# Markham Lee (C) 2026
# https://github.com/MarkhamLee/vision-pipeline-benchmarks
# Orchestrator for the async/sequential-inference pipeline variant.
# Runs two YOLO models sequentially on a single thread per frame,
# while frame loading and data I/O remain async/parallel.
# Use this alongside the parallel async orchestrator to isolate the
# impact of per-model thread context switching on inference throughput.
import asyncio
import contextlib
import sys
import time
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

CURRENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = CURRENT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from base_orchestrator import BaseOrchestrator  # noqa: E402
from data_utils.data_clients import InfluxClient, PostgresClient  # noqa: E402
from models.model_loader import CudaYoloLoader  # noqa: E402
from report_creation.run_report_builder import RunReportBuilder  # noqa: E402
from utils.logging_utils import console_logging  # noqa: E402
from utils.pipeline_utils import send_slack_webhook_basic  # noqa: E402

logger = console_logging('hybrid-inference-async-orchestrator')
model_loader = CudaYoloLoader()
_SENTINEL = object()


@dataclass(slots=True)
class FrameItem:
    frame_id: int
    frame: Any
    enqueued_at: float


@dataclass(slots=True)
class InferenceResultItem:
    """Combined result for both models from a single inference pass."""
    frame_id: int
    queue_wait_ms: float
    m1_latency_ms: float
    m2_latency_ms: float
    frame_latency_ms: float   # m1 + m2 — true serial elapsed
    m1_count: int
    m2_count: int


@dataclass(slots=True)
class ControlEvent:
    event: str
    payload: dict


class AsyncSequentialOrchestrator(BaseOrchestrator):
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
        self.report_builder = RunReportBuilder(self.reports_dir)

        pipeline_cfg = config.get('pipeline', {})
        source_cfg = config.get('source', {})
        telemetry_cfg = config.get('telemetry', {})

        self.model1_path = pipeline_cfg['model1_path']
        self.model2_path = pipeline_cfg['model2_path']
        self.model1_confidence = pipeline_cfg.get('model1_confidence', 0.4)
        self.model2_confidence = pipeline_cfg.get('model2_confidence', 0.4)
        self.model1_class = pipeline_cfg['model1_class_number']
        self.model2_class = pipeline_cfg['model2_class_number']
        self.model1_class_name = pipeline_cfg['model1_class_name']
        self.model2_class_name = pipeline_cfg['model2_class_name']
        self.model1_name = pipeline_cfg.get('model1_name', self.model1_path)
        self.model2_name = pipeline_cfg.get('model2_name', self.model2_path)

        self.source_id = source_cfg.get('source_id', 'unknown')
        self.source_type = source_cfg.get('type', 'unknown')
        self.source_path = source_cfg.get('path', '')
        self.flush_interval_s = pipeline_cfg.get('flush_interval_seconds', 60)
        self.influx_measurement = pipeline_cfg.get(
            'influx_db_measurement',
            'async_sequential_pipeline_telemetry'
        )
        self.postgres_table = pipeline_cfg['postgres_table']
        self.frame_queue_size = pipeline_cfg.get('frame_queue_size', 64)
        self.result_queue_size = pipeline_cfg.get('result_queue_size', 256)
        self.flush_timeout_s = pipeline_cfg.get('flush_timeout_seconds', 10)

        self.hardware_label = telemetry_cfg.get('hardware_label',
                                                'unknown-hardware')
        self.pipeline_label = telemetry_cfg.get('pipeline_label',
                                                'async-sequential')
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

        self._error: Exception | None = None
        self._failed_task_name: str | None = None
        self._pending_frame_buckets = 0
        self._final_flush_success = True
        self._frame_queue: asyncio.Queue | None = None
        self._result_queue: asyncio.Queue | None = None
        self._run_started_at: str | None = None
        self._run_completed_at: str | None = None
        self._current_video: dict | None = None
        self._video_summaries: list[dict] = []
        self._run_totals = self._new_aggregate_bucket()

    async def run(self, source) -> dict:
        self._frame_queue = asyncio.Queue(maxsize=self.frame_queue_size)
        self._result_queue = asyncio.Queue(maxsize=self.result_queue_size)

        self._run_started_at = time.strftime('%Y-%m-%d %H:%M:%S')
        run_wall_start = time.perf_counter()
        logger.info(
            'Async-sequential pipeline started | source_id=%s | hardware=%s',
            self.source_id,
            self.hardware_label
        )

        tasks = [
            asyncio.create_task(
                self._frame_loader(source, self._frame_queue),
                name='frame-loader'
            ),
            asyncio.create_task(
                self._inference_worker(self._frame_queue,
                                       self._result_queue),
                name='inference-worker'
            ),
            asyncio.create_task(
                self._aggregator(self._result_queue),
                name='aggregator'
            ),
        ]

        try:
            done, pending = await asyncio.wait(
                tasks, return_when=asyncio.FIRST_EXCEPTION)
            for task in done:
                exc = task.exception()
                if exc:
                    self._error = exc
                    self._failed_task_name = task.get_name()
                    logger.exception(
                        'Task failure | task=%s | error=%s',
                        task.get_name(),
                        exc,
                    )
                    raise exc
            if pending:
                await asyncio.gather(*pending)
        finally:
            for task in tasks:
                if not task.done():
                    task.cancel()
            for task in tasks:
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await task

        if self._current_video is not None:
            self._finalize_video()

        run_wall_elapsed_s = time.perf_counter() - run_wall_start
        self._run_completed_at = time.strftime('%Y-%m-%d %H:%M:%S')
        summary = self._build_run_summary(run_wall_elapsed_s)
        self.report_builder.write(summary)

        duration_min = round(run_wall_elapsed_s / 60, 2)
        effective_inference_fps = round(summary['effective_inference_fps'], 3)
        effective_pipeline_fps = round(summary['effective_pipeline_fps'], 3)
        status = 'failed' if self._error else 'completed'
        completion_message = (
            f'Async-sequential pipeline with run ID: {self.source_id}, {status} in '  # noqa: E501
            f'{duration_min} minutes | total frames: {summary["total_frames"]} | '  # noqa: E501
            f'Effective inference FPS: {effective_inference_fps} | '
            f'Effective Pipeline FPS: {effective_pipeline_fps} | '
            f'hardware: {self.hardware_label} | '
            f'failed_task: {self._failed_task_name or "none"} | '
            f'pending_buckets: {self._pending_frame_buckets} | '
            f'final_flush_success: {self._final_flush_success}'
        )
        logger.info(completion_message)
        try:
            send_slack_webhook_basic(
                self.slack_pipeline_completion_webhook,
                completion_message
            )
        except Exception:
            logger.exception('Slack completion notification failed')
        return summary

    async def _frame_loader(self,
                            source,
                            queue: asyncio.Queue) -> None:
        """Push frames and control events onto the single frame queue."""
        frame_id = 0
        try:
            for item in source.frames():
                if isinstance(item, dict) and item.get('event') in {
                    'video_start', 'video_end'
                }:
                    await queue.put(
                        ControlEvent(event=item['event'], payload=item))
                    continue

                frame = (item['frame']
                         if isinstance(item, dict)
                         and item.get('event') == 'frame'
                         else item)
                await queue.put(FrameItem(
                    frame_id=frame_id,
                    frame=frame,
                    enqueued_at=time.perf_counter()
                ))
                frame_id += 1
        finally:
            await queue.put(_SENTINEL)
            logger.info('Frame loader completed | frames=%d', frame_id)

    async def _inference_worker(self,
                                frame_queue: asyncio.Queue,
                                result_queue: asyncio.Queue) -> None:
        """Run both models sequentially on one thread per frame."""
        while True:
            item = await frame_queue.get()
            try:
                if item is _SENTINEL:
                    await result_queue.put(_SENTINEL)
                    logger.info('Inference worker completed')
                    return

                if isinstance(item, ControlEvent):
                    await result_queue.put(item)
                    continue

                queue_wait_ms = (time.perf_counter() - item.enqueued_at) * 1000

                result = await asyncio.to_thread(
                    self._run_inference_sequential,
                    item.frame,
                )

                await result_queue.put(InferenceResultItem(
                    frame_id=item.frame_id,
                    queue_wait_ms=round(queue_wait_ms, 3),
                    m1_latency_ms=result['m1_latency_ms'],
                    m2_latency_ms=result['m2_latency_ms'],
                    frame_latency_ms=result['frame_latency_ms'],
                    m1_count=result['m1_count'],
                    m2_count=result['m2_count'],
                ))
            finally:
                frame_queue.task_done()

    def _run_inference_sequential(self, frame) -> dict:
        """Blocking: run model1 then model2 on the calling thread."""
        t_m1 = time.perf_counter()
        results1 = self.model1.predict(
            frame,
            conf=self.model1_confidence,
            classes=[self.model1_class],
            verbose=False,
        )
        m1_latency_ms = (time.perf_counter() - t_m1) * 1000

        t_m2 = time.perf_counter()
        results2 = self.model2.predict(
            frame,
            conf=self.model2_confidence,
            classes=[self.model2_class],
            verbose=False,
        )
        m2_latency_ms = (time.perf_counter() - t_m2) * 1000

        return {
            'm1_latency_ms': round(m1_latency_ms, 3),
            'm2_latency_ms': round(m2_latency_ms, 3),
            'frame_latency_ms': round(m1_latency_ms + m2_latency_ms, 3),
            'm1_count': self._extract_count(results1),
            'm2_count': self._extract_count(results2),
        }

    async def _aggregator(self, result_queue: asyncio.Queue) -> None:
        """Consume InferenceResultItems; one result arrives per frame."""
        interval = self._new_interval_state()

        while True:
            item = await result_queue.get()
            try:
                if item is _SENTINEL:
                    break

                if isinstance(item, ControlEvent):
                    payload = item.payload

                    if item.event == 'video_start':
                        if interval['frame_latencies_ms']:
                            elapsed = (time.monotonic()
                                       - interval['started_monotonic'])
                            await self._flush_interval(interval, elapsed)
                        self._start_video(payload)
                        interval = self._new_interval_state()
                        continue

                    if item.event == 'video_end':
                        if interval['frame_latencies_ms']:
                            elapsed = (time.monotonic()
                                       - interval['started_monotonic'])
                            await self._flush_interval(interval, elapsed)
                        interval = self._new_interval_state()
                        self._finalize_video()
                        continue

                # InferenceResultItem — no pairing needed
                interval['frame_latencies_ms'].append(item.frame_latency_ms)
                interval['model1_latencies_ms'].append(item.m1_latency_ms)
                interval['model2_latencies_ms'].append(item.m2_latency_ms)
                interval['queue_wait_ms'].append(item.queue_wait_ms)
                interval['counts_m1'][self.model1_class_name].append(
                    item.m1_count)
                interval['counts_m2'][self.model2_class_name].append(
                    item.m2_count)

                self._update_aggregate(
                    self._run_totals,
                    item.frame_latency_ms,
                    item.m1_latency_ms,
                    item.m2_latency_ms,
                    item.m1_count,
                    item.m2_count,
                )
                if self._current_video is not None:
                    self._update_aggregate(
                        self._current_video['aggregate'],
                        item.frame_latency_ms,
                        item.m1_latency_ms,
                        item.m2_latency_ms,
                        item.m1_count,
                        item.m2_count,
                    )

                elapsed = time.monotonic() - interval['started_monotonic']
                if (interval['frame_latencies_ms']
                        and elapsed >= self.flush_interval_s):
                    await self._flush_interval(interval, elapsed)
                    interval = self._new_interval_state()
            finally:
                result_queue.task_done()

        elapsed = time.monotonic() - interval['started_monotonic']
        if interval['frame_latencies_ms']:
            try:
                await self._flush_interval(interval, elapsed)
            except Exception:
                self._final_flush_success = False
                raise

        logger.info('Aggregator completed')

    async def _flush_interval(self,
                              interval: dict,
                              elapsed_s: float) -> None:
        try:
            await asyncio.wait_for(
                self._flush(interval, elapsed_s),
                timeout=self.flush_timeout_s
            )
        except Exception:
            self._final_flush_success = False
            raise

    async def _flush(self, interval: dict, elapsed_s: float) -> None:
        avg_frame_latency_ms = self._avg(interval['frame_latencies_ms'])
        avg_model1_latency_ms = self._avg(interval['model1_latencies_ms'])
        avg_model2_latency_ms = self._avg(interval['model2_latencies_ms'])
        avg_queue_wait_ms = self._avg(interval['queue_wait_ms'])
        avg_m1 = self._avg(interval['counts_m1'][self.model1_class_name])
        avg_m2 = self._avg(interval['counts_m2'][self.model2_class_name])
        frame_count = len(interval['frame_latencies_ms'])
        # Serial inference: frame_latency = m1 + m2, so frame_latency_ms_sum
        # is the correct denominator for effective_inference_fps
        total_inference_time_s = sum(interval['frame_latencies_ms']) / 1000
        effective_inference_fps = round(
            frame_count / total_inference_time_s, 3)  \
            if total_inference_time_s > 0 else 0.0
        effective_pipeline_fps = round(
            frame_count / elapsed_s, 3) if elapsed_s > 0 else 0.0
        avg_combined_inference_latency_ms = round(
            avg_model1_latency_ms + avg_model2_latency_ms, 3)

        queue_depth = self._frame_queue.qsize() if self._frame_queue else -1
        result_queue_depth = (self._result_queue.qsize()
                              if self._result_queue else -1)

        influx_data = {
            'avg_model1_latency_ms': avg_model1_latency_ms,
            'avg_model2_latency_ms': avg_model2_latency_ms,
            'avg_combined_inference_latency_ms': avg_combined_inference_latency_ms,  # noqa: E501
            'avg_frame_latency_ms': avg_frame_latency_ms,
            'avg_queue_wait_ms': avg_queue_wait_ms,
            'effective_inference_fps': effective_inference_fps,
            'effective_pipeline_fps': effective_pipeline_fps,
            'frame_count': frame_count,
            'frame_queue_size': self.frame_queue_size,
            'result_queue_size': self.result_queue_size,
            'frame_queue_depth': queue_depth,
            'result_queue_depth': result_queue_depth,
            'video_native_fps': round(interval['video_fps'] or 0.0, 3)
            if interval['video_fps'] is not None else 0.0,
            'video_width': int(interval['video_width'] or 0),
            'video_height': int(interval['video_height'] or 0),
            'video_duration_s': round(interval['video_duration_s'] or 0.0, 3)
            if interval['video_duration_s'] is not None else 0.0,
            f'avg_{self.model1_class_name}_count': avg_m1,
            f'avg_{self.model2_class_name}_count': avg_m2,
        }

        t_write_start = time.perf_counter()
        influx_base = deepcopy(self.influx_base)
        if interval['video_name']:
            influx_base['tags']['video_name'] = interval['video_name']
            influx_base['tags']['video_path'] = interval['video_path'] or ''

        await asyncio.to_thread(
            InfluxClient.write_influx_data,
            self.influx_client,
            influx_base,
            influx_data,
            self.influx_bucket,
        )
        await PostgresClient.write_detection_data_async(
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
            'Flush | frames=%d | frame_latency=%.3fms | m1=%.3fms | m2=%.3fms | '  # noqa: E501
            'queue_wait=%.3fms | combined_infer=%.3fms | infer_fps=%.3f | '
            'pipeline_fps=%.3f | video=%s | native_fps=%.3f | write_overhead=%.3fms | '  # noqa: E501
            'q_depth=%d | rq_depth=%d',
            frame_count,
            avg_frame_latency_ms,
            avg_model1_latency_ms,
            avg_model2_latency_ms,
            avg_queue_wait_ms,
            avg_combined_inference_latency_ms,
            effective_inference_fps,
            effective_pipeline_fps,
            interval['video_name'] or 'n/a',
            interval['video_fps'] or 0.0,
            write_overhead_ms,
            queue_depth,
            result_queue_depth,
        )

    def _start_video(self, item: dict) -> None:
        if self._current_video is not None:
            self._finalize_video()
        self._current_video = {
            'video_path': item.get('video_path', ''),
            'video_name': item.get('video_name', ''),
            'width': item.get('width', item.get('video_width')),
            'height': item.get('height', item.get('video_height')),
            'native_fps': item.get('native_fps'),
            'frame_total': item.get('frame_total', item.get('total_frames')),
            'duration_s': item.get('duration_s', item.get('video_duration_s')),
            'wall_start': time.perf_counter(),
            'aggregate': self._new_aggregate_bucket(),
        }
        logger.info(
            'Video started | %s | %sx%s | native_fps=%.3f',
            self._current_video['video_name'],
            self._current_video['width'] or 0,
            self._current_video['height'] or 0,
            self._current_video['native_fps'] or 0.0,
        )

    def _finalize_video(self) -> None:
        if self._current_video is None:
            return
        wall_elapsed_s = (time.perf_counter()
                          - self._current_video['wall_start'])
        aggregate = deepcopy(self._current_video['aggregate'])
        frame_count = aggregate['frame_count']
        model1_latency_sum_ms = aggregate['model1_latency_ms_sum']
        model2_latency_sum_ms = aggregate['model2_latency_ms_sum']
        # Serial: frame_latency_ms_sum == m1 + m2 sum; use it for infer FPS
        total_inference_time_s = aggregate['frame_latency_ms_sum'] / 1000

        summary = {
            'video_name': self._current_video['video_name'],
            'video_path': self._current_video['video_path'],
            'width': self._current_video['width'],
            'height': self._current_video['height'],
            'native_fps': round(self._current_video['native_fps'] or 0.0, 3),
            'frame_total': self._current_video['frame_total'],
            'duration_s': round(
                self._current_video['duration_s'] or 0.0, 3),
            'processed_frames': frame_count,
            'avg_model1_latency_ms': round(self._safe_avg(
                model1_latency_sum_ms, frame_count), 3),
            'avg_model2_latency_ms': round(self._safe_avg(
                model2_latency_sum_ms, frame_count), 3),
            'avg_combined_inference_latency_ms': round(self._safe_avg(
                model1_latency_sum_ms + model2_latency_sum_ms,
                frame_count), 3),
            'avg_frame_latency_ms': round(self._safe_avg(
                aggregate['frame_latency_ms_sum'], frame_count), 3),
            'effective_inference_fps': round(
                frame_count / total_inference_time_s, 3)
            if total_inference_time_s > 0 else 0.0,
            'effective_pipeline_fps': round(
                frame_count / wall_elapsed_s, 3)
            if wall_elapsed_s > 0 else 0.0,
            'avg_model1_count': round(self._safe_avg(
                aggregate['model1_count_sum'], frame_count), 3),
            'avg_model2_count': round(self._safe_avg(
                aggregate['model2_count_sum'], frame_count), 3),
            'wall_elapsed_s': round(wall_elapsed_s, 3),
        }
        self._video_summaries.append(summary)
        logger.info(
            'Video completed | %s | frames=%d | infer_fps=%.3f | pipeline_fps=%.3f',  # noqa: E501
            summary['video_name'],
            summary['processed_frames'],
            summary['effective_inference_fps'],
            summary['effective_pipeline_fps'],
        )
        self._current_video = None

    def _new_interval_state(self) -> dict:
        current = self._current_video or {}
        return {
            'started_monotonic': time.monotonic(),
            'frame_latencies_ms': [],
            'model1_latencies_ms': [],
            'model2_latencies_ms': [],
            'queue_wait_ms': [],
            'counts_m1': defaultdict(list),
            'counts_m2': defaultdict(list),
            'video_path': current.get('video_path'),
            'video_name': current.get('video_name'),
            'video_fps': current.get('native_fps'),
            'video_width': current.get('width'),
            'video_height': current.get('height'),
            'video_duration_s': current.get('duration_s'),
        }

    @staticmethod
    def _new_aggregate_bucket() -> dict:
        return {
            'frame_count': 0,
            'frame_latency_ms_sum': 0.0,
            'model1_latency_ms_sum': 0.0,
            'model2_latency_ms_sum': 0.0,
            'model1_count_sum': 0,
            'model2_count_sum': 0,
        }

    @staticmethod
    def _update_aggregate(bucket: dict,
                          frame_latency_ms: float,
                          model1_latency_ms: float,
                          model2_latency_ms: float,
                          model1_count: int,
                          model2_count: int) -> None:
        bucket['frame_count'] += 1
        bucket['frame_latency_ms_sum'] += frame_latency_ms
        bucket['model1_latency_ms_sum'] += model1_latency_ms
        bucket['model2_latency_ms_sum'] += model2_latency_ms
        bucket['model1_count_sum'] += model1_count
        bucket['model2_count_sum'] += model2_count

    def _build_run_summary(self, run_wall_elapsed_s: float) -> dict:
        total_frames = self._run_totals['frame_count']
        total_inference_time_s = (
            self._run_totals['frame_latency_ms_sum'] / 1000
        )
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
            'effective_pipeline_fps': round(
                total_frames / run_wall_elapsed_s, 3)
            if run_wall_elapsed_s > 0 else 0.0,
            'effective_inference_fps': round(
                total_frames / total_inference_time_s, 3)
            if total_inference_time_s > 0 else 0.0,
            'avg_model1_latency_ms': round(self._safe_avg(
                self._run_totals['model1_latency_ms_sum'], total_frames), 3),
            'avg_model2_latency_ms': round(self._safe_avg(
                self._run_totals['model2_latency_ms_sum'], total_frames), 3),
            'avg_combined_inference_latency_ms': round(self._safe_avg(
                self._run_totals['model1_latency_ms_sum'] +
                self._run_totals['model2_latency_ms_sum'],
                total_frames,
            ), 3),
            'avg_frame_latency_ms': round(self._safe_avg(
                self._run_totals['frame_latency_ms_sum'], total_frames), 3),
            'run_started_at': self._run_started_at,
            'run_completed_at': self._run_completed_at,
            'videos': self._video_summaries,
        }

    def get_metrics(self) -> dict:
        return {
            'pipeline': self.pipeline_label,
            'source_id': self.source_id,
            'hardware_label': self.hardware_label,
            'model1': self.model1_path,
            'model2': self.model2_path,
            'flush_interval_s': self.flush_interval_s,
            'flush_timeout_s': self.flush_timeout_s,
            'influx_measurement': self.influx_measurement,
            'postgres_table': self.postgres_table,
            'frame_queue_size': self.frame_queue_size,
            'result_queue_size': self.result_queue_size,
        }

    @staticmethod
    def _extract_count(results) -> int:
        return sum(len(result.boxes) for result in results)

    @staticmethod
    def _avg(values: list[float | int]) -> float:
        return round(sum(values) / len(values), 3) if values else 0.0

    @staticmethod
    def _safe_avg(total: float, count: int) -> float:
        return total / count if count else 0.0
