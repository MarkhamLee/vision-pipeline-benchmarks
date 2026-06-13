## Basic Usage Instructions

### Alerting and Data Infrastructure

1. Ensure you have running instances of InfluxDB and PostgreSQL. The pipelines expect the following environment variables for connectivity:

   * `VISION_PIPELINE_COMPLETION_WEBHOOK`
   * `INFLUX_TOKEN`
   * `INFLUX_ORG`
   * `INFLUX_URL`
   * `INFLUX_BUCKET`
   * `PG_HOST`
   * `VISION_PIPELINE_PG_USER`
   * `VISION_PIPELINE_PG_PASSWORD`

   There are `.sql` files in the [data utils folder](../data_utils/README.md) that you can use to create the required PostgreSQL tables.

2. (Optional) Create a channel in Slack for receiving pipeline completion alerts, then go to [https://api.slack.com](https://api.slack.com) to create a webhook for that channel. If you do not want Slack alerting, remove or comment out the alerting calls in the code.

   Once you have the webhook URL, store it in the `VISION_PIPELINE_COMPLETION_WEBHOOK` environment variable.

### Configuration and Running a Benchmark

1. Refer to the config [instructions](../config/readme.md) for details on setting up the config files. The async and sequential pipelines use nearly identical configurations.

2. Place the videos you want to benchmark in the `video_processing/test_videos` folder, or configure an RTSP feed in the config file.

3. Run the `main.py` script in the folder of the pipeline you want to execute (async or sequential). 