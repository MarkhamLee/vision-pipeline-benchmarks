## Vision Pipeline Benchmarks

This repo contains benchmarking tools for use in optimizing multi-modal computer vision inference pipelines. You can try different approaches (sequential vs async), swap models in and out and get a report of how fast the pipeline ran.  The goal is to build a tool that makes pipeline benchmarking quick and easy. 



For computer vision projects deployed at the edge the key constraints are often:
* **Time:** You are monitoring real‑world events and need to deliver alerts or results to people who can react to emergencies and other issues detected by cameras. 
* **Hardware:** You may be running on low‑power devices, or you need to run ML models on several video streams in parallel.
* **Economics:** vision AI product feasibility often comes down to how well you optimize time and hardware usage, which drives unit economics and customer adoption.

The tools in this repo are based on the approaches, "one off tools" and the like I've built in the past to optimize around the above parameters. For the moment the benchmarks are built around the YOLO computer vision models from [Ultralytics](https://github.com/ultralytics/ultralytics) running on NVIDIA GPUs, but they will be expanded to support other models and hardware in the future. 

### How does it work?

1) You start with a YAML configuration file, where you specify:
	1) Models, along with key parameters like classes and confidence thresholds
	2) Paths to source videos and other key factors, e.g., processing window for RTSP, creds for RTSP
	3) Database related fields like table names 
2) When you run a benchmark it outputs: 
	1) Video analytics to Postgres, e.g., object counts, scene analysis and LLM narration 
	2) Pipeline telemetry: effective FPS, latency for each model, latency for API calls for writing to databases and other I/O 
	3) A markdown report that captures run details, including config parameters and summary metrics such as average FPS and latency.
    4) A Slack message at the end of the pipeline run with data including total duration, inferencing FPS and effective pipeline FPS.
Since this is meant to simulate production pipelines, the benchmarks require the user to have an InfluxDB and PostgresSQL instance available to write data to. 


### Infrastructure

I maintain a [K3s cluster](https://github.com/MarkhamLee/k3s-powered-private-cloud-homelab) that I use to host a “private cloud” running applications that support home automation, personal projects, my professional work, and experiments like this one. This is where I host the InfluxDB and Postgres databases used to store benchmark data, plus a Grafana instance for visualization. 


### Repo Structure 

* **Sequential Inference:** benchmarks for a pipeline where the models run in sequence either for simplicity or because one model depends on the output from another. In this pipeline the models run in sequence, and the inferencing is periodically blocked for API calls, I/O, etc. 
* **Async Parallel Inference:** two or more models need to run on the same frame or image, with the results combined or analyzed together afterwards. In this pipeline the goals is to minimize *GPU Idle Time*:
	* Models run in parallel
	* Independent frame loaders continuously prepare visual data so the GPU is never waiting on a frame.
	* All API calls and I/O run as “fire‑and‑forget” workers so the GPU is not blocked on uploads or database writes.


### Roadmap 
* V1.0 – Async and sequential pipelines with analytics and telemetry persisted to Postgres and InfluxDB respectively.
* V1.5 – Benchmarks comparing multi‑threaded vs truly independent async pipelines, plus VLM narration to periodically describe recent frames or segments of video.
* V2.0 – Single command to run all pipeline types and generate a unified comparison report.
* V2.5 – Support for running models on non‑NVIDIA hardware (for example RKNN, Hailo).