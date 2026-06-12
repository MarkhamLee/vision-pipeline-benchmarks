# Markham Lee (C) 2026
# https://github.com/MarkhamLee/vision-pipeline-benchmarks
# Orchestrator for the async/parallel inference pipeline.
# Loads frames, runs inference, collates data, generates reports
# and pushes telemetry data to InfluxDB and analytics data to Postgres
import asyncio
import contextlib
import sys
import time
from collections import defaultdict
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
from utils.logging_utils import console_logging  # noqa: E402
from utils.pipeline_utils import send_slack_webhook_basic  # noqa: E402

logger = console_logging('async-orchestrator')
model_loader = CudaYoloLoader()
_SENTINEL = object()


@dataclass(slots=True)
class FrameItem:
    frame_id: int
    frame: Any
    enqueued_at: float


@dataclass(slots=True)
class ModelResultItem:
    frame_id: int
    model_key: str
    class_name: str
    count: int
    worker_latency_ms: float
    queue_wait_ms: float


@dataclass(slots=True)
class ControlEvent:
    event: str
    payload: dict


class AsyncOrchestrator(BaseOrchestrator):
    def __init__(self,
                 config: dict,
                 influx_client,
                 influx_bucket: str,
                 pg_pool,
                 slack_webhook: str) -> None:
        self.config = config
        self.influx_client = influx_client
        self.influx_bucket = influx_bucket
        self.pg_pool = pg_pool
        self.slack_pipeline_completion_webhook = slack_webhook

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
        self.flush_interval_s = pipeline_cfg.get('flush_interval_seconds', 60)
        self.influx_measurement = pipeline_cfg.get(
            'influx_db_measurement',
            'async_pipeline_telemetry'
        )
        self.postgres_table = pipeline_cfg['postgres_table']
        self.frame_queue_size = pipeline_cfg.get('frame_queue_size', 64)
        self.result_queue_size = pipeline_cfg.get('result_queue_size', 256)
        self.flush_timeout_s = pipeline_cfg.get('flush_timeout_seconds', 10)

        self.hardware_label = telemetry_cfg.get('hardware_label',
                                                'unknown-hardware')
        self.pipeline_label = telemetry_cfg.get('pipeline_label', 'async')
        self.site_label = telemetry_cfg.get('site_label', 'local')

        self.model1 = model_loader.load_yolo_model(self.model1_path)
        self.model2 = model_loader.load_yolo_model(self.model2_path)
        logger.info('Models loaded: %s | %s', self.model1_path, self.model2_path)  # noqa: E501

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
        self._frame_queue_1: asyncio.Queue | None = None
        self._frame_queue_2: asyncio.Queue | None = None
        self._result_queue: asyncio.Queue | None = None
        self._run_started_at: str | None = None
        self._run_completed_at: str | None = None
        self._current_video: dict | None = None
        self._video_summaries: list[dict] = []
        self._run_totals = self._new_aggregate_bucket()

    async def run(self, source) -> dict:
        self._frame_queue_1 = asyncio.Queue(maxsize=self.frame_queue_size)
        self._frame_queue_2 = asyncio.Queue(maxsize=self.frame_queue_size)
        self._result_queue = asyncio.Queue(maxsize=self.result_queue_size)

        self._run_started_at = time.strftime('%Y-%m-%d %H:%M:%S')
        run_wall_start = time.perf_counter()
        logger.info(
            'Async pipeline started | source_id=%s | hardware=%s',
            self.source_id,
            self.hardware_label
        )

        tasks = [
            asyncio.create_task(
                self._frame_loader(source,
                                   self._frame_queue_1,
                                   self._frame_queue_2),
                name='frame-loader'
            ),
            asyncio.create_task(
                self._model_worker(
                    model=self.model1,
                    model_key='model1',
                    class_id=self.model1_class,
                    class_name=self.model1_class_name,
                    confidence=self.model1_confidence,
                    input_queue=self._frame_queue_1,
                    result_queue=self._result_queue,
                ),
                name='model1-worker'
            ),
            asyncio.create_task(
                self._model_worker(
                    model=self.model2,
                    model_key='model2',
                    class_id=self.model2_class,
                    class_name=self.model2_class_name,
                    confidence=self.model2_confidence,
                    input_queue=self._frame_queue_2,
                    result_queue=self._result_queue,
                ),
                name='model2-worker'
            ),
            asyncio.create_task(self._aggregator(self._result_queue),
                                name='aggregator'),
        ]

        try:
            done, pending = await asyncio.wait(tasks,
                                               return_when=asyncio.FIRST_EXCEPTION)  # noqa: E501
            for task in done:
                exc = task.exception()
                if exc:
                    self._error = exc
                    self._failed_task_name = task.get_name()
                    logger.exception(
                        'Async task failure | task=%s | error=%s',
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
        self._write_run_report(summary)

        duration_min = round(run_wall_elapsed_s / 60, 2)
        effective_inference_fps = round(summary['effective_inference_fps'], 2)
        effective_pipeline_fps = round(summary['effective_pipeline_fps'], 2)
        status = 'failed' if self._error else 'completed'
        completion_message = (
            f'Async pipeline with run ID: {self.source_id}, {status} in '
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
                            q1: asyncio.Queue,
                            q2: asyncio.Queue) -> None:
        frame_id = 0
        try:
            for item in source.frames():
                if isinstance(item, dict) and item.get('event') in {'video_start', 'video_end'}:  # noqa: E501
                    control = ControlEvent(event=item['event'], payload=item)
                    await q1.put(control)
                    await q2.put(control)
                    continue

                frame = item['frame'] if isinstance(item, dict) and item.get('event') == 'frame' else item  # noqa: E501
                frame_item = FrameItem(
                    frame_id=frame_id,
                    frame=frame,
                    enqueued_at=time.perf_counter()
                )
                await q1.put(frame_item)
                await q2.put(frame_item)
                frame_id += 1
        finally:
            await q1.put(_SENTINEL)
            await q2.put(_SENTINEL)
            logger.info('Frame loader completed | frames=%d', frame_id)

    async def _model_worker(self,
                            model,
                            model_key: str,
                            class_id: int,
                            class_name: str,
                            confidence: float,
                            input_queue: asyncio.Queue,
                            result_queue: asyncio.Queue) -> None:
        while True:
            item = await input_queue.get()
            try:
                if item is _SENTINEL:
                    await result_queue.put(_SENTINEL)
                    logger.info('%s worker completed', model_key)
                    return

                if isinstance(item, ControlEvent):
                    await result_queue.put(item)
                    continue

                started = time.perf_counter()
                queue_wait_ms = (started - item.enqueued_at) * 1000
                results = await asyncio.to_thread(
                    model.predict,
                    item.frame,
                    conf=confidence,
                    classes=[class_id],
                    verbose=False,
                )
                ended = time.perf_counter()
                worker_latency_ms = (ended - started) * 1000
                count = self._extract_count(results)

                await result_queue.put(
                    ModelResultItem(
                        frame_id=item.frame_id,
                        model_key=model_key,
                        class_name=class_name,
                        count=count,
                        worker_latency_ms=worker_latency_ms,
                        queue_wait_ms=queue_wait_ms,
                    )
                )
            finally:
                input_queue.task_done()

    async def _aggregator(self, result_queue: asyncio.Queue) -> None:
        interval = self._new_interval_state()
        pending: dict[int, dict[str, ModelResultItem]] = {}
        sentinel_count = 0
        control_barrier_counts: dict[tuple[str, str], int] = defaultdict(int)

        while True:
            item = await result_queue.get()
            try:
                if item is _SENTINEL:
                    sentinel_count += 1
                    if sentinel_count == 2:
                        break
                    continue

                if isinstance(item, ControlEvent):
                    payload = item.payload
                    key = (item.event, payload.get('video_path', ''))
                    control_barrier_counts[key] += 1
                    if control_barrier_counts[key] < 2:
                        continue
                    del control_barrier_counts[key]

                    if item.event == 'video_start':
                        if interval['frame_latencies_ms']:
                            elapsed = time.\
                                monotonic() - interval['started_monotonic']
                            await self._flush_interval(interval, elapsed)
                        self._start_video(payload)
                        interval = self._new_interval_state()
                        continue

                    if item.event == 'video_end':
                        if interval['frame_latencies_ms']:
                            elapsed = time.\
                                monotonic() - interval['started_monotonic']
                            await self._flush_interval(interval, elapsed)
                            interval = self._new_interval_state()
                        self._finalize_video()
                        continue

                bucket = pending.setdefault(item.frame_id, {})
                bucket[item.model_key] = item

                if 'model1' in bucket and 'model2' in bucket:
                    m1 = bucket['model1']
                    m2 = bucket['model2']
                    combined_latency_ms = max(m1.worker_latency_ms,
                                              m2.worker_latency_ms)

                    interval['frame_latencies_ms'].append(combined_latency_ms)
                    interval['model1_latencies_ms'].\
                        append(m1.worker_latency_ms)
                    interval['model2_latencies_ms'].\
                        append(m2.worker_latency_ms)
                    interval['queue_wait_ms'].append(max(m1.queue_wait_ms,
                                                         m2.queue_wait_ms))
                    interval['counts_m1'][m1.class_name].append(m1.count)
                    interval['counts_m2'][m2.class_name].append(m2.count)

                    self._update_aggregate(
                        self._run_totals,
                        combined_latency_ms,
                        m1.worker_latency_ms,
                        m2.worker_latency_ms,
                        m1.count,
                        m2.count,
                    )
                    if self._current_video is not None:
                        self._update_aggregate(
                            self._current_video['aggregate'],
                            combined_latency_ms,
                            m1.worker_latency_ms,
                            m2.worker_latency_ms,
                            m1.count,
                            m2.count,
                        )
                        self._current_video['frames_processed'] += 1

                    del pending[item.frame_id]

                    elapsed = time.monotonic() - interval['started_monotonic']
                    if interval['frame_latencies_ms'] and elapsed >= self.flush_interval_s:  # noqa: E501
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

        self._pending_frame_buckets = len(pending)
        if pending:
            logger.warning(
                'Aggregator exiting with %d unmatched frame result buckets',
                len(pending)
            )
        logger.info('Aggregator completed')

    async def _flush_interval(self, interval: dict, elapsed_s: float) -> None:
        try:
            await asyncio.wait_for(self._flush(interval, elapsed_s),
                                   timeout=self.flush_timeout_s)
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
        effective_inference_fps = 1000 / avg_frame_latency_ms if avg_frame_latency_ms > 0 else 0.0  # noqa: E501
        effective_pipeline_fps = frame_count / elapsed_s if elapsed_s > 0 else 0.0  # noqa: E501

        queue_1_depth = self._frame_queue_1.qsize() if self._frame_queue_1 else -1  # noqa: E501
        queue_2_depth = self._frame_queue_2.qsize() if self._frame_queue_2 else -1  # noqa: E501
        result_queue_depth = self._result_queue.qsize() if self._result_queue else -1  # noqa: E501

        influx_data = {
            'avg_frame_latency_ms': avg_frame_latency_ms,
            'avg_model1_latency_ms': avg_model1_latency_ms,
            'avg_model2_latency_ms': avg_model2_latency_ms,
            'avg_queue_wait_ms': avg_queue_wait_ms,
            'combined_infer_ms': avg_frame_latency_ms,
            'infer_fps': round(effective_inference_fps, 3),
            'pipeline_fps': round(effective_pipeline_fps, 3),
            'frame_count': frame_count,
            'frame_queue_size': self.frame_queue_size,
            'result_queue_size': self.result_queue_size,
            'frame_queue_1_depth': queue_1_depth,
            'frame_queue_2_depth': queue_2_depth,
            'result_queue_depth': result_queue_depth,
            'native_fps': interval['native_fps'] or 0.0,
            'video_width': interval['video_width'] or 0,
            'video_height': interval['video_height'] or 0,
            'video_duration_s': interval['video_duration_s'] or 0.0,
            f'avg_{self.model1_class_name}_count': avg_m1,
            f'avg_{self.model2_class_name}_count': avg_m2,
        }

        t_write_start = time.perf_counter()
        await asyncio.to_thread(
            InfluxClient.write_influx_data,
            self.influx_client,
            self.influx_base,
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
            'Flush | frames=%d | frame_latency=%.1fms | m1=%.1fms | m2=%.1fms | '  # noqa: E501
            'queue_wait=%.1fms | combined_infer=%.1fms | infer_fps=%.2f | '
            'pipeline_fps=%.2f | video=%s | native_fps=%.2f | write_overhead=%.1fms | '  # noqa: E501
            'q1_depth=%d | q2_depth=%d | rq_depth=%d',
            frame_count,
            avg_frame_latency_ms,
            avg_model1_latency_ms,
            avg_model2_latency_ms,
            avg_queue_wait_ms,
            avg_frame_latency_ms,
            effective_inference_fps,
            effective_pipeline_fps,
            interval['video_name'] or 'n/a',
            interval['native_fps'] or 0.0,
            write_overhead_ms,
            queue_1_depth,
            queue_2_depth,
            result_queue_depth,
        )

    def _start_video(self, item: dict) -> None:
        self._current_video = {
            'video_name': item.get('video_name') or 'unknown',
            'video_path': item.get('video_path', ''),
            'video_width': item.get('video_width', item.get('width')),
            'video_height': item.get('video_height', item.get('height')),
            'native_fps': item.get('native_fps'),
            'video_duration_s': item.get('video_duration_s', item.get('duration_s')),  # noqa: E501
            'total_frames': item.get('total_frames', item.get('frame_total')),
            'frames_processed': 0,
            'aggregate': self._new_aggregate_bucket(),
        }
        logger.info(
            'Video started | %s | %sx%s | native_fps=%.3f',
            self._current_video['video_name'],
            self._current_video['video_width'] or 0,
            self._current_video['video_height'] or 0,
            self._current_video['native_fps'] or 0.0,
        )

    def _finalize_video(self) -> None:
        if self._current_video is None:
            return
        aggregate = self._current_video['aggregate']
        frame_count = aggregate['frame_count']
        avg_frame_latency = self._safe_avg(aggregate['frame_latency_ms_sum'],
                                           frame_count)
        avg_m1_latency = self._safe_avg(aggregate['model1_latency_ms_sum'],
                                        frame_count)
        avg_m2_latency = self._safe_avg(aggregate['model2_latency_ms_sum'],
                                        frame_count)
        avg_m1_count = self._safe_avg(aggregate['model1_count_sum'],
                                      frame_count)
        avg_m2_count = self._safe_avg(aggregate['model2_count_sum'],
                                      frame_count)
        effective_inference_fps = round(1000 / avg_frame_latency, 2) if avg_frame_latency > 0 else 0.0  # noqa: E501

        summary = {
            'video_name': self._current_video['video_name'],
            'video_path': self._current_video['video_path'],
            'video_width': self._current_video['video_width'],
            'video_height': self._current_video['video_height'],
            'native_fps': self._current_video['native_fps'],
            'video_duration_s': self._current_video['video_duration_s'],
            'total_frames': self._current_video['total_frames'],
            'frames_processed': self._current_video['frames_processed'],
            'avg_frame_latency_ms': round(avg_frame_latency, 2),
            'avg_model1_latency_ms': round(avg_m1_latency, 2),
            'avg_model2_latency_ms': round(avg_m2_latency, 2),
            'avg_model1_count': round(avg_m1_count, 2),
            'avg_model2_count': round(avg_m2_count, 2),
            'effective_inference_fps': effective_inference_fps,
        }
        self._video_summaries.append(summary)
        logger.info(
            'Video completed | %s | frames=%d | infer_fps=%.2f',
            summary['video_name'],
            summary['frames_processed'],
            summary['effective_inference_fps'],
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
            'video_name': current.get('video_name'),
            'video_path': current.get('video_path'),
            'video_width': current.get('video_width'),
            'video_height': current.get('video_height'),
            'native_fps': current.get('native_fps'),
            'video_duration_s': current.get('video_duration_s'),
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
        frame_count = self._run_totals['frame_count']
        avg_frame_latency = self.\
            _safe_avg(self._run_totals['frame_latency_ms_sum'],
                      frame_count)
        avg_model1_latency = self.\
            _safe_avg(self._run_totals['model1_latency_ms_sum'],
                      frame_count)
        avg_model2_latency = self.\
            _safe_avg(self._run_totals['model2_latency_ms_sum'],
                      frame_count)
        avg_model1_count = self.\
            _safe_avg(self._run_totals['model1_count_sum'],
                      frame_count)
        avg_model2_count = self.\
            _safe_avg(self._run_totals['model2_count_sum'],
                      frame_count)
        effective_inference_fps = round(1000 / avg_frame_latency, 2) if avg_frame_latency > 0 else 0.0  # noqa: E501
        effective_pipeline_fps = round(frame_count / run_wall_elapsed_s, 2) if run_wall_elapsed_s > 0 else 0.0  # noqa: E501

        return {
            'run_started_at': self._run_started_at,
            'run_completed_at': self._run_completed_at,
            'source_id': self.source_id,
            'pipeline': self.pipeline_label,
            'hardware_label': self.hardware_label,
            'model1_path': self.model1_path,
            'model2_path': self.model2_path,
            'model1_class_name': self.model1_class_name,
            'model2_class_name': self.model2_class_name,
            'total_frames': frame_count,
            'run_duration_seconds': round(run_wall_elapsed_s, 2),
            'avg_frame_latency_ms': round(avg_frame_latency, 2),
            'avg_model1_latency_ms': round(avg_model1_latency, 2),
            'avg_model2_latency_ms': round(avg_model2_latency, 2),
            'avg_model1_count': round(avg_model1_count, 2),
            'avg_model2_count': round(avg_model2_count, 2),
            'effective_inference_fps': effective_inference_fps,
            'effective_pipeline_fps': effective_pipeline_fps,
            'failed_task': self._failed_task_name,
            'pending_buckets': self._pending_frame_buckets,
            'final_flush_success': self._final_flush_success,
            'videos': self._video_summaries,
        }

    def _write_run_report(self, summary: dict) -> None:
        reports_dir = REPO_ROOT / 'reports'
        reports_dir.mkdir(exist_ok=True)
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        report_path = reports_dir / f'run_report_{self.source_id}_{timestamp}.md'  # noqa: E501

        lines = [
            f'# Async Run Report: {summary["source_id"]}',
            '',
            f'- Started: {summary["run_started_at"]}',
            f'- Completed: {summary["run_completed_at"]}',
            f'- Pipeline: {summary["pipeline"]}',
            f'- Hardware: {summary["hardware_label"]}',
            f'- Model 1: {summary["model1_path"]}',
            f'- Model 2: {summary["model2_path"]}',
            f'- Total frames: {summary["total_frames"]}',
            f'- Run duration seconds: {summary["run_duration_seconds"]}',
            f'- Average frame latency ms: {summary["avg_frame_latency_ms"]}',
            f'- Average model1 latency ms: {summary["avg_model1_latency_ms"]}',
            f'- Average model2 latency ms: {summary["avg_model2_latency_ms"]}',
            f'- Average {summary["model1_class_name"]} count: {summary["avg_model1_count"]}',  # noqa: E501
            f'- Average {summary["model2_class_name"]} count: {summary["avg_model2_count"]}',  # noqa: E501
            f'- Effective inference FPS: {summary["effective_inference_fps"]}',
            f'- Effective pipeline FPS: {summary["effective_pipeline_fps"]}',
            f'- Failed task: {summary["failed_task"] or "none"}',
            f'- Pending buckets: {summary["pending_buckets"]}',
            f'- Final flush success: {summary["final_flush_success"]}',
            '',
            '## Video summaries',
            '',
        ]

        if summary['videos']:
            for video in summary['videos']:
                lines.extend([
                    f'### {video["video_name"]}',
                    '',
                    f'- Path: {video["video_path"]}',
                    f'- Resolution: {video["video_width"]}x{video["video_height"]}',  # noqa: E501
                    f'- Native FPS: {video["native_fps"]}',
                    f'- Duration seconds: {video["video_duration_s"]}',
                    f'- Total frames: {video["total_frames"]}',
                    f'- Frames processed: {video["frames_processed"]}',
                    f'- Average frame latency ms: {video["avg_frame_latency_ms"]}',  # noqa: E501
                    f'- Average model1 latency ms: {video["avg_model1_latency_ms"]}',  # noqa: E501
                    f'- Average model2 latency ms: {video["avg_model2_latency_ms"]}',  # noqa: E501
                    f'- Average {summary["model1_class_name"]} count: {video["avg_model1_count"]}',  # noqa: E501
                    f'- Average {summary["model2_class_name"]} count: {video["avg_model2_count"]}',  # noqa: E501
                    f'- Effective inference FPS: {video["effective_inference_fps"]}',  # noqa: E501
                    '',
                ])
        else:
            lines.append('- No per-video summaries recorded.')
            lines.append('')

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
