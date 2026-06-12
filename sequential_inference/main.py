# Markham Lee (C) 2023 - 2026
# https://github.com/MarkhamLee/vision-pipeline-benchmarks
# Entrypoint for the sequential inference pipeline.
# Loads config, initialises clients, runs the orchestrator,
# and saves a timestamped config snapshot to reports/.
import os

import sys
import shutil
import yaml
from datetime import datetime
from pathlib import Path
from orchestrator import SequentialOrchestrator  # noqa: E402

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

from utils.logging_utils import console_logging  # noqa: E402
from video_processing.video_source import VideoSource  # noqa: E402
from data_utils.data_clients import InfluxClient, PostgresClient  # noqa: E402


logger = console_logging('sequential-main')

CONFIG_PATH = '../config/pipeline_config_prod.yaml'
REPORTS_DIR = Path('../reports')


def load_config(path: str = CONFIG_PATH) -> dict:
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def save_config_snapshot(source_path: str = CONFIG_PATH) -> None:
    """Copy the active config to reports/ with a timestamp suffix,
    so each benchmark run has a record of exactly what was used."""
    REPORTS_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    source_id = load_config(source_path).\
        get('source',
            {}).get('source_id',
                    'run')
    dest = REPORTS_DIR / f"config_snapshot_{source_id}_{timestamp}.yaml"
    shutil.copy2(source_path, dest)
    logger.info('Config snapshot saved: %s', dest)


def main():

    config = load_config()

    # Save a config snapshot before the run starts
    save_config_snapshot()

    # get Slack webhook for pipeline completion notifications
    SLACK_WEBHOOK = os.environ['VISION_PIPELINE_COMPLETION_WEBHOOK']

    # InfluxDB config
    influx_client = InfluxClient.influx_client(
        token=os.environ['INFLUX_TOKEN'],
        org=os.environ['INFLUX_ORG'],
        url=os.environ['INFLUX_URL']
    )
    influx_bucket = os.environ['INFLUX_BUCKET']

    pipeline_cfg = config.get('pipeline', {})

    # PostgreSQL
    pg_conninfo = (
        f"host={os.environ['PG_HOST']} "
        f"port={os.environ.get('PG_PORT', '5432')} "
        f"dbname={pipeline_cfg.get('postgres_table')} "
        f"user={os.environ['VISION_PIPELINE_PG_USER']} "
        f"password={os.environ['VISION_PIPELINE_PG_PASSWORD']}"
    )
    pg_pool = PostgresClient.postgres_client(pg_conninfo)

    # Video source
    source = VideoSource(config['source'])

    # Run
    orchestrator = SequentialOrchestrator(
        config=config,
        influx_client=influx_client,
        influx_bucket=influx_bucket,
        pg_pool=pg_pool,
        slack_webhook=SLACK_WEBHOOK
    )

    try:
        orchestrator.run(source)
    except KeyboardInterrupt:
        logger.info('Sequential pipeline stopped by user')
    finally:
        pg_pool.close()


if __name__ == '__main__':
    main()
