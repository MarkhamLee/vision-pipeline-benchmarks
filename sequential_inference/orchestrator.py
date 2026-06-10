# Markham Lee (C) 2023 - 2026
# https://github.com/MarkhamLee/vision-pipeline-benchmarks
# Sequential inference orchestrator.
# Runs two YOLO models one after the other per frame, each scoped
# to a single class via config. Telemetry goes to InfluxDB;
# per-interval detection counts go to PostgreSQL.
import time
from collections import defaultdict
from base_orchestrator import BaseOrchestrator
from models.model_loader import CudaYoloLoader
from utils.data_clients import InfluxClient, PostgresClient
from utils.logging_utils import console_logging

logger = console_logging('sequential-orchestrator')

# Instantiated once at module level — shared across both model loads
model_loader = CudaYoloLoader()


class SequentialOrchestrator(BaseOrchestrator):

    def __init__(self,
                 config: dict,
                 influx_client,
                 influx_bucket: str,
                 pg_pool) -> None:

        self.config = config
        self.influx_client = influx_client
        self.influx_bucket = influx_bucket
        self.pg_pool = pg_pool

        pipeline_cfg = config.get('pipeline', {})

        # Model paths and per-model inference config
        self.model1_path = pipeline_cfg['model1_path']
        self.model2_path = pipeline_cfg['model2_path']
        self.model1_confidence = pipeline_cfg.get('model1_confidence', 0.4)
        self.model2_confidence = pipeline_cfg.get('model2_confidence', 0.4)

        # Class filter: pass as list to YOLO classes= param
        # Config uses model1_class_number / model2_class_number (integers)
        self.model1_class = pipeline_cfg.get('model1_class_number')
        self.model1_class_name = pipeline_cfg.\
            get('model1_class_names',
                'model1')
        self.model2_class = pipeline_cfg.get('model2_class')
        self.model2_class_name = pipeline_cfg.get('model2_class_name'
                                                  'model2')

        # Pipeline-level config
        self.source_id = config['source'].get('source_id', 'unknown')
        self.flush_interval_s = pipeline_cfg.get('flush_interval_seconds', 60)

        # DB routing from config
        self.influx_measurement = pipeline_cfg.get(
            'influx_db_measurement', 'sequential_pipeline_telemetry'
        )
        self.postgres_table = pipeline_cfg.get(
            'postgres_table', 'sequential_analytics_data'
        )

        # Load models via the hardware-aware loader
        self.model1 = model_loader.load_yolo_model(self.model1_path)
        self.model2 = model_loader.load_yolo_model(self.model2_path)
        logger.info('Models loaded: %s | %s',
                    self.model1_path,
                    self.model2_path)

        # InfluxDB base payload — measurement name now comes from config
        self.influx_base = {
            "measurement": self.influx_measurement,
            "tags": {
                "pipeline": "sequential",
                "source_id": self.source_id,
                "model1": self.model1_path,
                "model2": self.model2_path
            }
        }

    def run(self, source) -> None:
        """Main loop — reads frames, runs both models sequentially,
        flushes metrics every flush_interval_s seconds."""

        frame_times = []
        counts_m1: dict[str, list[int]] = defaultdict(list)
        counts_m2: dict[str, list[int]] = defaultdict(list)
        interval_start = time.monotonic()
        frame_count = 0

        logger.info('Sequential pipeline started | source_id=%s',
                    self.source_id)

        for frame in source.frames():
            t_frame_start = time.perf_counter()

            # Model 1 — scoped to its configured class
            results1 = self.model1.predict(
                frame,
                conf=self.model1_confidence,
                classes=[self.model1_class],
                verbose=False
            )
            m1_count = self._extract_count(results1)

            # Model 2 — scoped to its configured class
            results2 = self.model2.predict(
                frame,
                conf=self.model2_confidence,
                classes=[self.model2_class],
                verbose=False
            )
            m2_count = self._extract_count(results2)

            t_frame_end = time.perf_counter()
            frame_latency_ms = (t_frame_end - t_frame_start) * 1000
            frame_times.append(frame_latency_ms)

            # Use class names from config as dict keys — generic, not hardcoded
            counts_m1[self.model1_class_name].append(m1_count)
            counts_m2[self.model2_class_name].append(m2_count)
            frame_count += 1

            # Flush on interval
            elapsed = time.monotonic() - interval_start
            if elapsed >= self.flush_interval_s:
                self._flush(frame_times, counts_m1, counts_m2, elapsed)
                frame_times.clear()
                counts_m1.clear()
                counts_m2.clear()
                interval_start = time.monotonic()

        # Flush any remaining data at end of source
        elapsed = time.monotonic() - interval_start
        if frame_times:
            self._flush(frame_times, counts_m1, counts_m2, elapsed)

        logger.info('Sequential pipeline complete | total frames: %d',
                    frame_count)

    def _extract_count(self, results) -> int:
        """Returns total detection count from a YOLO results object.
        Since each model is already class-filtered via classes=[], we just
        count all detections without label inspection."""
        return sum(len(result.boxes) for result in results)

    def _flush(self, frame_times: list[float],
               counts_m1: dict, counts_m2: dict,
               elapsed_s: float) -> None:
        """Compute interval averages, write to InfluxDB and PostgreSQL,
        and log effective FPS including DB write overhead."""

        avg_latency_ms = sum(frame_times) / len(frame_times)
        avg_m1 = self._avg(counts_m1[self.model1_class_name])
        avg_m2 = self._avg(counts_m2[self.model2_class_name])

        t_write_start = time.perf_counter()

        # InfluxDB: inference telemetry — field names use config class names
        influx_data = {
            "avg_frame_latency_ms": round(avg_latency_ms, 3),
            f"avg_{self.model1_class_name}_count": avg_m1,
            f"avg_{self.model2_class_name}_count": avg_m2,
            "frame_count": len(frame_times)
        }
        InfluxClient.write_influx_data(
            self.influx_client,
            self.influx_base,
            influx_data,
            self.influx_bucket
        )

        # PostgreSQL: counts — table name comes from config
        PostgresClient.write_detection_data(
            pool=self.pg_pool,
            source_id=self.source_id,
            model1_class=self.model1_class_name,
            model1_count=round(avg_m1),
            model2_class=self.model2_class_name,
            model2_count=round(avg_m2)
        )

        t_write_end = time.perf_counter()
        write_overhead_ms = (t_write_end - t_write_start) * 1000
        effective_fps = len(frame_times) / elapsed_s

        logger.info(
            'Flush | frames=%d | avg_latency=%.1fms | eff_fps=%.2f | '
            'write_overhead=%.1fms | %s=%.1f | %s=%.1f',
            len(frame_times), avg_latency_ms, effective_fps,
            write_overhead_ms,
            self.model1_class_name, avg_m1,
            self.model2_class_name, avg_m2
        )

    def get_metrics(self) -> dict:
        """Required by BaseOrchestrator — returns pipeline config snapshot."""
        return {
            "pipeline": "sequential",
            "model1": self.model1_path,
            "model2": self.model2_path,
            "source_id": self.source_id,
            "flush_interval_s": self.flush_interval_s,
            "influx_measurement": self.influx_measurement,
            "postgres_table": self.postgres_table
        }

    @staticmethod
    def _avg(values: list[int]) -> float:
        return round(sum(values) / len(values), 2) if values else 0.0
