from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class VideoReportSummary:
    video_name: str
    video_path: str = ''
    width: int | None = None
    height: int | None = None
    native_fps: float = 0.0
    frame_total: int | None = None
    duration_s: float = 0.0
    processed_frames: int = 0
    avg_model1_latency_ms: float = 0.0
    avg_model2_latency_ms: float = 0.0
    avg_combined_inference_latency_ms: float = 0.0
    avg_frame_latency_ms: float = 0.0
    effective_inference_fps: float = 0.0
    effective_pipeline_fps: float = 0.0
    avg_model1_count: float = 0.0
    avg_model2_count: float = 0.0
    wall_elapsed_s: float = 0.0

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> 'VideoReportSummary':
        normalized = {
            'video_name': data.get('video_name') or 'unknown',
            'video_path': data.get('video_path', ''),
            'width': data.get('width', data.get('video_width')),
            'height': data.get('height', data.get('video_height')),
            'native_fps': round(float(data.get('native_fps') or 0.0), 3),
            'frame_total': data.get('frame_total', data.get('total_frames')),
            'duration_s': round(float(data.get('duration_s',
                                               data.get('video_duration_s') or 0.0)), 3),  # noqa: E501
            'processed_frames': int(data.get('processed_frames',
                                             data.get('frames_processed') or 0)),  # noqa: E501
            'avg_model1_latency_ms': round(float(data.
                                                 get('avg_model1_latency_ms') or 0.0), 3),  # noqa: E501
            'avg_model2_latency_ms': round(float(data.
                                                 get('avg_model2_latency_ms') or 0.0), 3),  # noqa: E501
            'avg_combined_inference_latency_ms': round(float(data.
                                                             get('avg_combined_inference_latency_ms',  # noqa: E501
                                                                 data.get('combined_infer_ms') or 0.0)), 3),  # noqa: E501
            'avg_frame_latency_ms': round(float(data.get('avg_frame_latency_ms') or 0.0), 3),  # noqa: E501
            'effective_inference_fps': round(float(data.get('effective_inference_fps') or 0.0), 3),  # noqa: E501
            'effective_pipeline_fps': round(float(data.get('effective_pipeline_fps') or 0.0), 3),  # noqa: E501
            'avg_model1_count': round(float(data.get('avg_model1_count') or 0.0), 3),  # noqa: E501
            'avg_model2_count': round(float(data.get('avg_model2_count') or 0.0), 3),  # noqa: E501
            'wall_elapsed_s': round(float(data.get('wall_elapsed_s') or 0.0), 3),  # noqa: E501
        }
        return cls(**normalized)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class RunReportSummary:
    source_id: str
    pipeline: str
    hardware_label: str
    site_label: str = 'local'
    source_type: str = 'unknown'
    source_path: str = ''
    model1_path: str = ''
    model2_path: str = ''
    model1_name: str = ''
    model2_name: str = ''
    influx_measurement: str = ''
    postgres_table: str = ''
    flush_interval_s: float = 0.0
    total_frames: int = 0
    run_wall_elapsed_s: float = 0.0
    effective_pipeline_fps: float = 0.0
    effective_inference_fps: float = 0.0
    avg_model1_latency_ms: float = 0.0
    avg_model2_latency_ms: float = 0.0
    avg_combined_inference_latency_ms: float = 0.0
    avg_frame_latency_ms: float = 0.0
    run_started_at: str | None = None
    run_completed_at: str | None = None
    videos: list[VideoReportSummary] = field(default_factory=list)

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> 'RunReportSummary':
        videos_raw = data.get('videos') or []
        normalized_videos = [
            video if isinstance(video,
                                VideoReportSummary) else VideoReportSummary.from_mapping(video)  # noqa: E501
            for video in videos_raw
        ]
        normalized = {
            'source_id': data.get('source_id', 'unknown'),
            'pipeline': data.get('pipeline', 'unknown'),
            'hardware_label': data.get('hardware_label', 'unknown-hardware'),
            'site_label': data.get('site_label', 'local'),
            'source_type': data.get('source_type', 'unknown'),
            'source_path': data.get('source_path', ''),
            'model1_path': data.get('model1_path', ''),
            'model2_path': data.get('model2_path', ''),
            'model1_name': data.get('model1_name', data.get('model1_path', '')),  # noqa: E501
            'model2_name': data.get('model2_name', data.get('model2_path', '')),  # noqa: E501
            'influx_measurement': data.get('influx_measurement', ''),
            'postgres_table': data.get('postgres_table', ''),
            'flush_interval_s': round(float(data.get('flush_interval_s') or 0.0), 3),  # noqa: E501
            'total_frames': int(data.get('total_frames') or 0),
            'run_wall_elapsed_s': round(float(data.get('run_wall_elapsed_s', data.get('run_duration_seconds') or 0.0)), 3),  # noqa: E501
            'effective_pipeline_fps': round(float(data.get('effective_pipeline_fps') or 0.0), 3),  # noqa: E501
            'effective_inference_fps': round(float(data.get('effective_inference_fps') or 0.0), 3),  # noqa: E501
            'avg_model1_latency_ms': round(float(data.get('avg_model1_latency_ms') or 0.0), 3),  # noqa: E501
            'avg_model2_latency_ms': round(float(data.get('avg_model2_latency_ms') or 0.0), 3),  # noqa: E501
            'avg_combined_inference_latency_ms': round(float(data.get('avg_combined_inference_latency_ms') or 0.0), 3),  # noqa: E501
            'avg_frame_latency_ms': round(float(data.get('avg_frame_latency_ms') or 0.0), 3),  # noqa: E501
            'run_started_at': data.get('run_started_at'),
            'run_completed_at': data.get('run_completed_at'),
            'videos': normalized_videos,
        }
        return cls(**normalized)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload['videos'] = [video.to_dict() for video in self.videos]
        return payload
