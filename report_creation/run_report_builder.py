# report_creation/run_report_builder.py
import time
from pathlib import Path


class RunReportBuilder:
    def __init__(self, reports_dir: str | Path) -> None:
        self.reports_dir = Path(reports_dir)
        self.reports_dir.mkdir(exist_ok=True)

    def write(self, summary: dict) -> Path:
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        report_path = self.reports_dir / f'run_report_{summary["source_id"]}_{timestamp}.md'  # noqa: E501

        lines = [
            f'# Run Report: {summary["source_id"]}',
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
                resolution = (
                    f"{video['width']}x{video['height']}"
                    if video.get('width') and video.get('height')
                    else 'unknown'
                )
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

        report_path.write_text('\\n'.join(lines), encoding='utf-8')
        return report_path
