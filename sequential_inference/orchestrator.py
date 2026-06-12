# Markham Lee (C) 2026
# https://github.com/MarkhamLee/vision-pipeline-benchmarks
# Sequential inference orchestrator.
# Runs two YOLO models one after the other per frame, persists
# performance telemetry to InfluxDB, analytics to PostgreSQL,
# and writes a final markdown report with run-level and per-video stats.
import sys
import time
from collections import defaultdict
from copy import deepcopy
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = CURRENT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from base_orchestrator import BaseOrchestrator  # noqa: E402
from data_utils.data_clients import InfluxClient, PostgresClient  # noqa: E402
from models.model_loader import CudaYoloLoader  # noqa: E402
from utils.logging_utils import console_logging  # noqa: E402
from utils.pipeline_utils import send_slack_webhook_basic  # noqa: E402

logger = console_logging('sequential-orchestrator')
model_loader = CudaYoloLoader()


class SequentialOrchestrator(BaseOrchestrator):
    def __init__(self,
                 config: dict,
                 influx_client,
                 influx_bucket: str,
                 pg_pool,
                 slack_webhook: str,
                 reports_dir: str | Path = REPO_ROOT / 'reports') -> None:
        self.config = config
        self.influx_client = influx_client
        self.influx_bucket = influx_bucket
        self.pg_pool = pg_pool
        self.slack_pipeline_completion_webhook = slack_webhook
        self.reports_dir = Path(reports_dir)
        self.reports_dir.mkdir(exist_ok=True)

        pipeline_cfg = config.get('pipeline', {})
        source_cfg = config.get('source', {})
        telemetry_cfg = config.get('telemetry', {})

        self.model1_path = pipeline_cfg['model1_path']
        self.model2_path = pipeline_cfg['model2_path']
        self.model1_confidence = pipeline_cfg.get('model1_confidence', 0.4)
        self.model2_confidence = pipeline_cfg.get('model2_confidence', 0.4)
        self.model1_class = pipeline_cfg.get('model1_class_number')
        self.model1_class_name = pipeline_cfg.get('model1_class_name')
        self.model2_class = pipeline_cfg.get('model2_class_number')
        self.model2_class_name = pipeline_cfg.get('model2_class_name')
        self.model1_name = pipeline_cfg.get('model1_name', self.model1_path)
        self.model2_name = pipeline_cfg.get('model2_name', self.model2_path)

        self.source_id = source_cfg.get('source_id', 'unknown')
        self.source_type = source_cfg.get('type', 'unknown')
        self.source_path = source_cfg.get('path', '')
        self.flush_interval_s = pipeline_cfg.get('flush_interval_seconds', 60)
        self.influx_measurement = pipeline_cfg.\
            get('influx_db_measurement',
                'sequential_pipeline_telemetry')
        self.postgres_table = pipeline_cfg.get('postgres_table',
                                               'sequential_analytics_data')

        self.hardware_label = telemetry_cfg.get('hardware_label',
                                                'unknown-hardware')
        self.pipeline_label = telemetry_cfg.get('pipeline_label', 'sequential')
        self.site_label = telemetry_cfg.get('site_label', 'local')

        self.model1 = model_loader.load_yolo_model(self.model1_path)
        self.model2 = model_loader.load_yolo_model(self.model2_path)
        logger.info('Models loaded: %s | %s', self.model1_path,
                    self.model2_path)

        self.influx_base = {
            'measurement': self.influx_measurement,
            'tags': {
                'pipeline': self.pipeline_label,
                'source_id': self.source_id,
                'site': self.site_label,
                'hardware': self.hardware_label,
                'model1_path': self.model1_path,
                'model2_path': self.model2_path,
                'model1_name': self.model1_name,
                'model2_name': self.model2_name,
            }
        }

        self._run_started_at = None
        self._run_completed_at = None
        self._run_totals = self._empty_aggregate()
        self._video_summaries = []
        self._current_video = None

    def run(self, source) -> dict:
        self._run_started_at = time.strftime('%Y-%m-%d %H:%M:%S')
        run_wall_start = time.perf_counter()
        interval = self._new_interval_state()

        logger.info(
            'Sequential pipeline started | source_id=%s | hardware=%s',
            self.source_id,
            self.hardware_label
        )

        for item in source.frames():
            if isinstance(item, dict) and item.get('event') == 'video_start':
                if interval['frame_latencies_ms']:
                    elapsed = time.monotonic() - interval['started_monotonic']
                    self._flush(interval, elapsed)
                self._start_video(item)
                interval = self._new_interval_state()
                continue

            if isinstance(item, dict) and item.get('event') == 'video_end':
                if interval['frame_latencies_ms']:
                    elapsed = time.monotonic() - interval['started_monotonic']
                    self._flush(interval, elapsed)
                    interval = self._new_interval_state()
                self._finalize_video()
                continue

            if isinstance(item, dict) and item.get('event') == 'frame':
                frame = item['frame']
            else:
                frame = item

            t_frame_start = time.perf_counter()

            t_m1_start = time.perf_counter()
            results1 = self.model1.predict(
                frame,
                conf=self.model1_confidence,
                classes=[self.model1_class],
                verbose=False,
            )
            t_m1_end = time.perf_counter()
            m1_latency_ms = (t_m1_end - t_m1_start) * 1000
            m1_count = self._extract_count(results1)

            t_m2_start = time.perf_counter()
            results2 = self.model2.predict(
                frame,
                conf=self.model2_confidence,
                classes=[self.model2_class],
                verbose=False,
            )
            t_m2_end = time.perf_counter()
            m2_latency_ms = (t_m2_end - t_m2_start) * 1000
            m2_count = self._extract_count(results2)

            t_frame_end = time.perf_counter()
            frame_latency_ms = (t_frame_end - t_frame_start) * 1000

            interval['frame_latencies_ms'].append(frame_latency_ms)
            interval['model1_latencies_ms'].append(m1_latency_ms)
            interval['model2_latencies_ms'].append(m2_latency_ms)
            interval['counts_m1'][self.model1_class_name].append(m1_count)
            interval['counts_m2'][self.model2_class_name].append(m2_count)

            self._update_aggregate(
                self._run_totals,
                frame_latency_ms,
                m1_latency_ms,
                m2_latency_ms,
                m1_count,
                m2_count
            )

            if self._current_video is not None:
                self._update_aggregate(
                    self._current_video['aggregate'],
                    frame_latency_ms,
                    m1_latency_ms,
                    m2_latency_ms,
                    m1_count,
                    m2_count
                )

            elapsed = time.monotonic() - interval['started_monotonic']
            if elapsed >= self.flush_interval_s:
                self._flush(interval, elapsed)
                interval = self._new_interval_state()

        elapsed = time.monotonic() - interval['started_monotonic']
        if interval['frame_latencies_ms']:
            self._flush(interval, elapsed)

        if self._current_video is not None:
            self._finalize_video()

        run_wall_elapsed_s = time.perf_counter() - run_wall_start
        self._run_completed_at = time.strftime('%Y-%m-%d %H:%M:%S')
        summary = self._build_run_summary(run_wall_elapsed_s)
        self._write_run_report(summary)

        duration_min = round(run_wall_elapsed_s / 60, 2)
        effective_inference_fps = round(summary["effective_inference_fps"], 2)
        effective_pipeline_fps = round(summary["effective_pipeline_fps"], 2)

        completion_message = (
            f'Sequential pipeline with run ID: {self.source_id}, completed in '
            f'{duration_min} minutes | total frames: {summary["total_frames"]} | '  # noqa: E501
            f'Effective inference FPS: {effective_inference_fps} | '
            f'Effective Pipeline FPS: {effective_pipeline_fps} | '
            f'hardware: {self.hardware_label}'
        )
        logger.info(completion_message)
        send_slack_webhook_basic(
            self.slack_pipeline_completion_webhook,
            completion_message
        )
        return summary

    def _new_interval_state(self) -> dict:
        return {
            'started_monotonic': time.monotonic(),
            'frame_latencies_ms': [],
            'model1_latencies_ms': [],
            'model2_latencies_ms': [],
            'counts_m1': defaultdict(list),
            'counts_m2': defaultdict(list),
            'video_path': self._current_video['video_path'] if self._current_video else None,  # noqa: E501
            'video_name': self._current_video['video_name'] if self._current_video else None,  # noqa: E501
            'video_fps': self._current_video['native_fps'] if self._current_video else None,  # noqa: E501
            'video_width': self._current_video['width'] if self._current_video else None,  # noqa: E501
            'video_height': self._current_video['height'] if self._current_video else None,  # noqa: E501
            'video_duration_s': self._current_video['duration_s'] if self._current_video else None,  # noqa: E501
        }

    def _empty_aggregate(self) -> dict:
        return {
            'frame_count': 0,
            'frame_latency_sum_ms': 0.0,
            'model1_latency_sum_ms': 0.0,
            'model2_latency_sum_ms': 0.0,
            'model1_count_sum': 0,
            'model2_count_sum': 0,
        }

    def _update_aggregate(self,
                          aggregate: dict,
                          frame_latency_ms: float,
                          model1_latency_ms: float,
                          model2_latency_ms: float,
                          model1_count: int,
                          model2_count: int) -> None:
        aggregate['frame_count'] += 1
        aggregate['frame_latency_sum_ms'] += frame_latency_ms
        aggregate['model1_latency_sum_ms'] += model1_latency_ms
        aggregate['model2_latency_sum_ms'] += model2_latency_ms
        aggregate['model1_count_sum'] += model1_count
        aggregate['model2_count_sum'] += model2_count

    def _start_video(self, item: dict) -> None:
        if self._current_video is not None:
            self._finalize_video()
        self._current_video = {
            'video_path': item.get('video_path', ''),
            'video_name': item.get('video_name', ''),
            'width': item.get('width'),
            'height': item.get('height'),
            'native_fps': item.get('native_fps'),
            'frame_total': item.get('frame_total'),
            'duration_s': item.get('duration_s'),
            'wall_start': time.perf_counter(),
            'aggregate': self._empty_aggregate(),
        }
        logger.info('Video started | %s | %sx%s | native_fps=%.3f',
                    self._current_video['video_name'],
                    self._current_video['width'],
                    self._current_video['height'],
                    self._current_video['native_fps'] or 0.0)

    def _finalize_video(self) -> None:
        if self._current_video is None:
            return
        wall_elapsed_s = time.\
            perf_counter() - self._current_video['wall_start']
        aggregate = deepcopy(self._current_video['aggregate'])
        frame_count = aggregate['frame_count']
        model1_latency_sum_ms = aggregate['model1_latency_sum_ms']
        model2_latency_sum_ms = aggregate['model2_latency_sum_ms']
        total_inference_time_s = (model1_latency_sum_ms +
                                  model2_latency_sum_ms) / 1000
        summary = {
            'video_name': self._current_video['video_name'],
            'video_path': self._current_video['video_path'],
            'width': self._current_video['width'],
            'height': self._current_video['height'],
            'native_fps': round(self._current_video['native_fps'] or 0.0, 3),
            'frame_total': self._current_video['frame_total'],
            'duration_s': round(self._current_video['duration_s'] or 0.0, 3),
            'processed_frames': frame_count,
            'avg_model1_latency_ms': self.
            _safe_avg(model1_latency_sum_ms, frame_count),
            'avg_model2_latency_ms': self.
            _safe_avg(model2_latency_sum_ms, frame_count),
            'avg_combined_inference_latency_ms': self.
            _safe_avg(model1_latency_sum_ms + model2_latency_sum_ms,
                      frame_count),
            'avg_frame_latency_ms': self.
            _safe_avg(aggregate['frame_latency_sum_ms'], frame_count),
            'effective_inference_fps': round(frame_count / total_inference_time_s, 3) if total_inference_time_s > 0 else 0.0,  # noqa: E501
            'effective_pipeline_fps': round(frame_count / wall_elapsed_s, 3) if wall_elapsed_s > 0 else 0.0,  # noqa: E501
            'avg_model1_count': self._safe_avg(aggregate['model1_count_sum'], frame_count),  # noqa: E501
            'avg_model2_count': self._safe_avg(aggregate['model2_count_sum'], frame_count),  # noqa: E501
            'wall_elapsed_s': round(wall_elapsed_s, 3),
        }
        self._video_summaries.append(summary)
        logger.info('Video completed | %s | frames=%d | infer_fps=%.2f | pipeline_fps=%.2f',  # noqa: E501
                    summary['video_name'], summary['processed_frames'],
                    summary['effective_inference_fps'], summary['effective_pipeline_fps'])  # noqa: E501
        self._current_video = None

    def _flush(self, interval: dict, elapsed_s: float) -> None:
        frame_count = len(interval['frame_latencies_ms'])
        avg_model1_latency_ms = self._avg(interval['model1_latencies_ms'])
        avg_model2_latency_ms = self._avg(interval['model2_latencies_ms'])
        avg_combined_inference_latency_ms = round(avg_model1_latency_ms +
                                                  avg_model2_latency_ms, 3)
        avg_frame_latency_ms = self._avg(interval['frame_latencies_ms'])
        avg_m1 = self._avg(interval['counts_m1'][self.model1_class_name])
        avg_m2 = self._avg(interval['counts_m2'][self.model2_class_name])

        total_inference_time_s = (sum(interval['model1_latencies_ms'])
                                  + sum(interval['model2_latencies_ms'])) / 1000  # noqa: E501
        effective_inference_fps = frame_count / total_inference_time_s if total_inference_time_s > 0 else 0.0  # noqa: E501
        effective_pipeline_fps = frame_count\
            / elapsed_s if elapsed_s > 0 else 0.0

        t_write_start = time.perf_counter()
        influx_data = {
            'avg_model1_latency_ms': avg_model1_latency_ms,
            'avg_model2_latency_ms': avg_model2_latency_ms,
            'avg_combined_inference_latency_ms': avg_combined_inference_latency_ms,  # noqa: E501
            'avg_frame_latency_ms': avg_frame_latency_ms,
            'effective_inference_fps': round(effective_inference_fps, 3),
            'effective_pipeline_fps': round(effective_pipeline_fps, 3),
            'frame_count': frame_count,
            'video_native_fps': round(interval['video_fps'] or 0.0, 3)
            if interval['video_fps'] is not None else 0.0,
            'video_width': int(interval['video_width'] or 0),
            'video_height': int(interval['video_height'] or 0),
            'video_duration_s': round(interval['video_duration_s'] or 0.0, 3)
            if interval['video_duration_s'] is not None else 0.0,
        }
        influx_base = deepcopy(self.influx_base)
        if interval['video_name']:
            influx_base['tags']['video_name'] = interval['video_name']
            influx_base['tags']['video_path'] = interval['video_path'] or ''
        InfluxClient.write_influx_data(self.influx_client,
                                       influx_base,
                                       influx_data,
                                       self.influx_bucket)

        PostgresClient.write_detection_data(
            table_name=self.postgres_table,
            pool=self.pg_pool,
            source_id=self.source_id,
            model1_class=self.model1_class_name,
            model1_count=round(avg_m1),
            model2_class=self.model2_class_name,
            model2_count=round(avg_m2),
        )
        write_overhead_ms = (time.perf_counter() - t_write_start) * 1000

        logger.info(
            'Flush | frames=%d | frame_latency=%.1fms | m1=%.1fms | m2=%.1fms | combined_infer=%.1fms | '  # noqa: E501
            'infer_fps=%.2f | pipeline_fps=%.2f | video=%s | native_fps=%.2f | write_overhead=%.1fms',  # noqa: E501
            frame_count,
            avg_frame_latency_ms,
            avg_model1_latency_ms,
            avg_model2_latency_ms,
            avg_combined_inference_latency_ms,
            effective_inference_fps,
            effective_pipeline_fps,
            interval['video_name'] or 'n/a',
            interval['video_fps'] or 0.0,
            write_overhead_ms,
        )

    def _build_run_summary(self, run_wall_elapsed_s: float) -> dict:
        total_frames = self._run_totals['frame_count']
        total_inference_time_s = (
            self._run_totals['model1_latency_sum_ms'] +
            self._run_totals['model2_latency_sum_ms']
        ) / 1000
        return {
            'source_id': self.source_id,
            'pipeline': self.pipeline_label,
            'hardware_label': self.hardware_label,
            'site_label': self.site_label,
            'source_type': self.source_type,
            'source_path': self.source_path,
            'model1_path': self.model1_path,
            'model2_path': self.model2_path,
            'model1_name': self.model1_name,
            'model2_name': self.model2_name,
            'influx_measurement': self.influx_measurement,
            'postgres_table': self.postgres_table,
            'flush_interval_s': self.flush_interval_s,
            'total_frames': total_frames,
            'run_wall_elapsed_s': round(run_wall_elapsed_s, 3),
            'effective_pipeline_fps': round(total_frames / run_wall_elapsed_s, 3) if run_wall_elapsed_s > 0 else 0.0,  # noqa: E501
            'effective_inference_fps': round(total_frames / total_inference_time_s, 3) if total_inference_time_s > 0 else 0.0,  # noqa: E501
            'avg_model1_latency_ms': self._safe_avg(self._run_totals['model1_latency_sum_ms'], total_frames),  # noqa: E501
            'avg_model2_latency_ms': self._safe_avg(self._run_totals['model2_latency_sum_ms'], total_frames),  # noqa: E501
            'avg_combined_inference_latency_ms': self._safe_avg(
                self._run_totals['model1_latency_sum_ms'] + self._run_totals['model2_latency_sum_ms'], total_frames  # noqa: E501
            ),
            'avg_frame_latency_ms': self._safe_avg(self._run_totals['frame_latency_sum_ms'], total_frames),  # noqa: E501
            'run_started_at': self._run_started_at,
            'run_completed_at': self._run_completed_at,
            'videos': self._video_summaries,
        }

    def _write_run_report(self, summary: dict) -> None:
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        report_path = self.reports_dir / f'run_report_{self.source_id}_{timestamp}.md'  # noqa: E501
        lines = [
            f'# Sequential Run Report: {summary["source_id"]}',
            '',
            '## Run',
            f'- Pipeline: {summary["pipeline"]}',
            f'- Source ID: {summary["source_id"]}',
            f'- Source type: {summary.get("source_type", "unknown")}',
            f'- Source path: {summary.get("source_path", "")}',
            f'- Hardware: {summary["hardware_label"]}',
            f'- Site: {summary["site_label"]}',
            f'- Started: {summary.get("run_started_at")}',
            f'- Completed: {summary.get("run_completed_at")}',
            '',
            '## Models',
            f'- Model 1: {summary.get("model1_name")} ({summary.get("model1_path")})',  # noqa: E501
            f'- Model 2: {summary.get("model2_name")} ({summary.get("model2_path")})',  # noqa: E501
            '',
            '## Stores',
            f'- Influx measurement: {summary.get("influx_measurement")}',
            f'- PostgreSQL table: {summary.get("postgres_table")}',
            '',
            '## Overall Performance',
            f'- Total frames: {summary["total_frames"]}',
            f'- Run wall elapsed seconds: {summary["run_wall_elapsed_s"]}',
            f'- Effective inference FPS: {summary.get("effective_inference_fps")}',  # noqa: E501
            f'- Effective pipeline FPS: {summary.get("effective_pipeline_fps")}',  # noqa: E501
            f'- Average model 1 latency ms: {summary.get("avg_model1_latency_ms")}',  # noqa: E501
            f'- Average model 2 latency ms: {summary.get("avg_model2_latency_ms")}',  # noqa: E501
            f'- Average combined inference latency ms: {summary.get("avg_combined_inference_latency_ms")}',  # noqa: E501
            f'- Average frame latency ms: {summary.get("avg_frame_latency_ms")}',  # noqa: E501
            '',
        ]
        if summary.get('videos'):
            lines.extend(['## Per-Video Performance', ''])
            for video in summary['videos']:
                resolution = f"{video['width']}x{video['height']}" if video.get('width') and video.get('height') else 'unknown'  # noqa: E501
                lines.extend([
                    f"### {video['video_name']}",
                    f"- Path: {video['video_path']}",
                    f"- Duration seconds: {video['duration_s']}",
                    f"- Native FPS: {video['native_fps']}",
                    f"- Resolution: {resolution}",
                    f"- Frames processed: {video['processed_frames']}",
                    f"- Effective inference FPS: {video['effective_inference_fps']}",  # noqa: E501
                    f"- Effective pipeline FPS: {video['effective_pipeline_fps']}",  # noqa: E501
                    f"- Average model 1 latency ms: {video['avg_model1_latency_ms']}",  # noqa: E501
                    f"- Average model 2 latency ms: {video['avg_model2_latency_ms']}",  # noqa: E501
                    f"- Average combined inference latency ms: {video['avg_combined_inference_latency_ms']}",  # noqa: E501
                    f"- Average frame latency ms: {video['avg_frame_latency_ms']}",  # noqa: E501
                    '',
                ])
        lines.extend([
            '## Notes',
            '- Report summarizes run-level behavior plus per-video behavior for folder inputs.',  # noqa: E501
            '- Config snapshots are stored as a separate YAML file in the reports folder',  # noqa: E501
            '- Interval-level telemetry is available in InfluxDB.',
        ])
        report_path.write_text('\n'.join(lines), encoding='utf-8')
        logger.info('Run report saved: %s', report_path)

    def get_metrics(self) -> dict:
        return {
            'pipeline': self.pipeline_label,
            'source_id': self.source_id,
            'hardware_label': self.hardware_label,
            'model1': self.model1_path,
            'model2': self.model2_path,
            'flush_interval_s': self.flush_interval_s,
            'influx_measurement': self.influx_measurement,
            'postgres_table': self.postgres_table,
        }

    @staticmethod
    def _extract_count(results) -> int:
        return sum(len(result.boxes) for result in results)

    @staticmethod
    def _avg(values: list[float | int]) -> float:
        return round(sum(values) / len(values), 3) if values else 0.0

    @staticmethod
    def _safe_avg(total: float, count: int) -> float:
        return round(total / count, 3) if count else 0.0
