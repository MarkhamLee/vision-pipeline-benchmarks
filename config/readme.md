## Pipeline Configurations

This folder contains sample config files for running the benchmark pipelines. The current configs are built around Ultralytics YOLO models, but support for additional model types may be added later. These benchmarks are designed to run in a production-style environment, so you will need supporting infrastructure such as InfluxDB and PostgreSQL before running them.

### Setting up the configs

Use the following steps to prepare a config file:

1. Select the models you want to benchmark. Place local model weights in the appropriate `models` folder for the pipeline you are running, or use a supported off-the-shelf model such as YOLOv8 and allow the pipeline to download it.
2. Enter the class numbers and class names for the two target classes you want to track.
3. Set queue sizes, timeouts, and related pipeline parameters, or keep the default values if they fit your test.
4. Set the PostgreSQL table name and InfluxDB measurement name used for reporting.
5. Set the hardware label for the GPU being used. *Note: only NVIDIA GPUs are currently supported.*
6. Set the `flush_interval` in the config file. This controls how often the pipeline writes aggregated benchmark data to InfluxDB and PostgreSQL.