# Run Report: async_test_run3

## Run
- Pipeline: async
- Source ID: async_test_run3
- Source type: folder
- Source path: ../video_processing/test_videos
- Hardware: NVIDIA 4060 mobile
- Site: local-dev
- Started: 2026-06-23 17:43:02
- Completed: 2026-06-23 17:44:16

## Models
- Model 1: fine-tuned-car-model_v1 (models/yolov8m.pt)
- Model 2: fine-tuned-people-model_v1 (models/yolov8m.pt)

## Stores
- Influx measurement: async-test1_telemetry
- PostgreSQL table: async_benchmark_analytics_data

## Overall Performance
- Total frames: 2348
- Run wall elapsed seconds: 73.145
- Effective inference FPS: 29.226
- Effective pipeline FPS: 32.101
- Average model 1 latency ms: 30.976
- Average model 2 latency ms: 30.688
- Average combined inference latency ms: 61.664
- Average frame latency ms: 34.216

## Per-Video Performance

### 13912687_1080_1920_30fps.mp4
- Path: ..\video_processing\test_videos\13912687_1080_1920_30fps.mp4
- Source Video Duration(seconds): 27.1
- Processing ratio (<1 = faster than real time): 1.1
- Native FPS: 30.0
- Resolution: 1080x1920
- Frames processed: 813
- Video processing time seconds: 31.002
- Effective inference FPS: 23.942
- Effective pipeline FPS: 26.224
- Average model 1 latency ms: 38.066
- Average model 2 latency ms: 38.066
- Average combined inference latency ms: 76.132
- Average frame latency ms: 41.768

### 5278-182817488.mp4
- Path: ..\video_processing\test_videos\5278-182817488.mp4
- Source Video Duration(seconds): 61.4
- Processing ratio (<1 = faster than real time): 0.7
- Native FPS: 25.0
- Resolution: 1840x1034
- Frames processed: 1535
- Video processing time seconds: 41.831
- Effective inference FPS: 33.095
- Effective pipeline FPS: 36.696
- Average model 1 latency ms: 27.221
- Average model 2 latency ms: 26.78
- Average combined inference latency ms: 54.002
- Average frame latency ms: 30.216

## Notes
- Processing ratio is a critical metric, <1 means the pipeline is processing videos faster than their native FPS.
- Report summarizes run-level behavior plus per-video behavior for folder inputs.
- Config snapshots are stored as a separate YAML file in the reports folder
- Interval-level telemetry is available in InfluxDB.