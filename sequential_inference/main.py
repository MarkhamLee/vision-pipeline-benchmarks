# Markham Lee (C) 2026
# https://github.com/MarkhamLee/vision-pipeline-benchmarks
# Entrypoint for the sequential inference pipeline.
# Loads config, initialises clients, runs the orchestrator,
# saves a timestamped config snapshot, and writes a final run report.
import os
import shutil
import sys
import yaml
from datetime import datetime
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = CURRENT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data_utils.data_clients import InfluxClient, PostgresClient  # noqa: E402
from sequential_inference.\
    orchestrator import SequentialOrchestrator  # noqa: E402
from utils.logging_utils import console_logging  # noqa: E402
from video_processing.video_source import VideoSource  # noqa: E402
from utils.pipeline_utils import validate_env  # noqa: E402

logger = console_logging('sequential-main')

CONFIG_PATH = REPO_ROOT / 'config' / 'pipeline_config_prod.yaml'
REPORTS_DIR = REPO_ROOT / 'reports'

REQUIRED_ENV_VARS = (
    'VISION_PIPELINE_COMPLETION_WEBHOOK',
    'INFLUX_TOKEN',
    'INFLUX_ORG',
    'INFLUX_URL',
    'INFLUX_BUCKET',
    'PG_HOST',
    'VISION_PIPELINE_PG_USER',
    'VISION_PIPELINE_PG_PASSWORD',
)
REQUIRED_PIPELINE_KEYS = (
    'model1_path',
    'model2_path',
    'model1_class_number',
    'model2_class_number',
    'model1_class_name',
    'model2_class_name',
    'postgres_database',
    'postgres_table',
)


def load_config(path: Path = CONFIG_PATH) -> dict:
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


# Validate configs - might move this to a util method but leaving
# here for now as different pipelines might have significant different
# configs in the future
def validate_config(config: dict) -> None:
    if 'source' not in config or not isinstance(config['source'], dict):
        raise ValueError('Missing required config section: source')
    if 'pipeline' not in config or not isinstance(config['pipeline'], dict):
        raise ValueError('Missing required config section: pipeline')

    source_cfg = config['source']
    pipeline_cfg = config['pipeline']

    if not source_cfg.get('type'):
        raise ValueError('Missing required config key: source.type')
    if source_cfg.get('type') == 'folder' and not source_cfg.get('path'):
        raise ValueError('Missing required config key: source.path')
    if source_cfg.get('type') == 'rtsp' and not source_cfg.get('rtsp_url'):
        raise ValueError('Missing required config key: source.rtsp_url')

    missing_pipeline = [key for key in REQUIRED_PIPELINE_KEYS if key not in pipeline_cfg]  # noqa: E501
    if missing_pipeline:
        raise ValueError(
            f'Missing required pipeline config keys: {", ".join(missing_pipeline)}'  # noqa: E501
        )

    empty_required_strings = [
        key for key in REQUIRED_PIPELINE_KEYS
        if isinstance(pipeline_cfg.get(key), str) and not pipeline_cfg[key].strip()  # noqa: E501
    ]
    if empty_required_strings:
        raise ValueError(
            f'Empty required pipeline config values: {", ".join(empty_required_strings)}'  # noqa: E501
        )


def save_config_snapshot(config: dict,
                         source_path: Path = CONFIG_PATH) -> Path:
    REPORTS_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    source_id = config.get('source', {}).get('source_id', 'run')
    dest = REPORTS_DIR / f'config_snapshot_{source_id}_{timestamp}.yaml'
    shutil.copy2(source_path, dest)
    logger.info('Config snapshot saved: %s', dest)
    return dest


def build_pg_conninfo(db_name: str) -> str:
    return (
        f"host={os.environ['PG_HOST']} "
        f"port={os.environ.get('PG_PORT', '5432')} "
        f"dbname={db_name} "
        f"user={os.environ['VISION_PIPELINE_PG_USER']} "
        f"password={os.environ['VISION_PIPELINE_PG_PASSWORD']}"
    )


def main() -> None:

    validate_env(REQUIRED_ENV_VARS)
    config = load_config()
    save_config_snapshot(config)
    validate_config(config)

    logger.info('Startup validation complete | platform=%s', sys.platform)

    slack_webhook = os.environ['VISION_PIPELINE_COMPLETION_WEBHOOK']
    influx_client = InfluxClient.influx_client(
        token=os.environ['INFLUX_TOKEN'],
        org=os.environ['INFLUX_ORG'],
        url=os.environ['INFLUX_URL'],
    )
    influx_bucket = os.environ['INFLUX_BUCKET']

    pipeline_cfg = config.get('pipeline', {})
    db_name = pipeline_cfg.get('postgres_database')
    if not db_name:
        raise ValueError('Missing required config key: pipeline.postgres_database')  # noqa: E501

    pg_pool = PostgresClient.postgres_client(build_pg_conninfo(db_name))
    source = VideoSource(config['source'])

    orchestrator = SequentialOrchestrator(
        config=config,
        influx_client=influx_client,
        influx_bucket=influx_bucket,
        pg_pool=pg_pool,
        slack_webhook=slack_webhook,
        reports_dir=REPORTS_DIR,
    )

    try:
        orchestrator.run(source)
    except KeyboardInterrupt:
        logger.info('Sequential pipeline stopped by user')
    finally:
        pg_pool.close()
        logger.info('Sequential PostgreSQL pool closed')


if __name__ == '__main__':
    main()
