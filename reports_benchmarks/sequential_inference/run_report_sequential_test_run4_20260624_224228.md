# Run Report: sequential_test_run4

## Run
- Pipeline: async
- Source ID: sequential_test_run4
- Source type: folder
- Source path: ../video_processing/test_videos
- Hardware: NVIDIA 4060 mobile
- Site: local-dev
- Started: 2026-06-24 22:41:03
- Completed: 2026-06-24 22:42:28

## Models
- Model 1: models/yolov8m.pt (models/yolov8m.pt)
- Model 2: models/yolov8m.pt (models/yolov8m.pt)

## Stores
- Influx measurement: sequential_benchmark_telemetry
- PostgreSQL table: sequential_benchmark_analytics_data

## Overall Performance
- Total frames: 2348
- Run wall elapsed seconds: 84.684
- Effective inference FPS: 31.244
- Effective pipeline FPS: 27.726
- Average model 1 latency ms: 17.925
- Average model 2 latency ms: 14.081
- Average combined inference latency ms: 32.007
- Average frame latency ms: 32.032

## Per-Video Performance

### 13912687_1080_1920_30fps.mp4
- Path: ..\video_processing\test_videos\13912687_1080_1920_30fps.mp4
- Source Video Duration(seconds): 27.1
- Processing ratio (<1 = faster than real time): 1.2
- Native FPS: 30.0
- Resolution: 1080x1920
- Frames processed: 813
- Video processing time seconds: 32.543
- Effective inference FPS: 28.1
- Effective pipeline FPS: 24.982
- Average model 1 latency ms: 21.451
- Average model 2 latency ms: 14.137
- Average combined inference latency ms: 35.588
- Average frame latency ms: 35.613

### 5278-182817488.mp4
- Path: ..\video_processing\test_videos\5278-182817488.mp4
- Source Video Duration(seconds): 61.4
- Processing ratio (<1 = faster than real time): 0.8
- Native FPS: 25.0
- Resolution: 1840x1034
- Frames processed: 1535
- Video processing time seconds: 52.087
- Effective inference FPS: 33.212
- Effective pipeline FPS: 29.47
- Average model 1 latency ms: 16.058
- Average model 2 latency ms: 14.052
- Average combined inference latency ms: 30.11
- Average frame latency ms: 30.135

## Notes
- Processing ratio is a critical metric, <1 means the pipeline is processing videos faster than their native FPS.
- Report summarizes run-level behavior plus per-video behavior for folder inputs.
- Config snapshots are stored as a separate YAML file in the reports folder
- Interval-level telemetry is available in InfluxDB.