## Async and Parallel Inference Pipeline

A high-level overview of the benchmarking pipelines is available [here](../README.md).

This async pipeline variant is designed to reduce GPU idle time by separating frame loading, inference, data collation, and data I/O into independent async tasks in order to benchmark multi-model inference pipelines.

### High-Level Architecture and Workflow

![High Level Architectre](../images/async_architecture_v1.png)


1. A frame loader reads frames from the configured source and places each frame into a separate queue for each model. 
2. Each model has its own independent worker that allows inference to be run on each frame in parallel.  
3. The results aggregator collects and combines the inference outputs and telemetry data (e.g. latency) for each frame. 
4. At the interval set in the config file, the pipeline writes aggregated metrics to the data stores: 
    * Pipeline performance/telemetry data, e.g. inference latency, is written to InfluxDB 
    * Model inference data, e.g., detected objects, is written to Postgres 
5. At the end of the run, the pipeline logs completion details, reports overall run status, and saves the active config snapshot in the `reports` folder.

### How to Use

1. General setup and infrastructure instructions are available [here](../docs/readme.md).
2. Refer to the config [instructions](../config/readme.md) for details on preparing the benchmark configuration. The async and sequential pipelines use nearly identical model and source settings, with the async pipeline adding queue and flush controls.

