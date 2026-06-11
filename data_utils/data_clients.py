# Markham Lee (C) 2023 - 2026
# https://github.com/MarkhamLee/internet-and-iot-data-platform
# Data clients for InfluxDB (telemetry) and
# PostgreSQL (detection counts + LLM narration)
from __future__ import annotations
import os
# import psycopg
import sys
from concurrent.futures import ThreadPoolExecutor
from influxdb_client import InfluxDBClient
from influxdb_client.client.write_api import SYNCHRONOUS, WriteOptions
from psycopg_pool import AsyncConnectionPool, ConnectionPool

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

from utils.logging_utils import console_logging  # noqa: E402

logger = console_logging('data-client-logging')

# Shared thread pool for fire-and-forget InfluxDB writes
_executor = ThreadPoolExecutor(max_workers=4)


class InfluxClient():

    def __init__(self) -> None:
        pass

    @staticmethod
    def influx_client(token: str, org: str, url: str):
        """Returns a synchronous write API client
        for the sequential pipeline."""
        try:
            write_client = InfluxDBClient(url=url, token=token, org=org)
            write_api = write_client.write_api(write_options=SYNCHRONOUS)
            logger.info('InfluxDB client created successfully')
            return write_api

        except Exception as e:
            logger.exception('InfluxDB client creation failed: %s', e)
            raise

    @staticmethod
    def influx_client_async(token: str, org: str, url: str,
                            batch_size: int = 50,
                            flush_interval: int = 5_000):
        """Returns a batched write API client for the async pipeline."""
        try:
            write_client = InfluxDBClient(url=url, token=token, org=org)
            write_api = write_client.write_api(
                write_options=WriteOptions(
                    batch_size=batch_size,
                    flush_interval=flush_interval
                )
            )
            logger.info('InfluxDB async client created successfully')
            return write_api

        except Exception as e:
            logger.exception('InfluxDB async client creation failed: %s', e)
            raise

    @staticmethod
    def write_influx_data(client: object, base: dict, data: dict, bucket: str):
        """Synchronous write — use in sequential pipeline."""
        payload = {**base, "fields": data}
        try:
            client.write(bucket=bucket, record=payload)
            logger.info('InfluxDB write successful')

        except Exception as e:
            logger.exception('InfluxDB write failed: %s', e)

    @staticmethod
    def write_influx_data_async(client: object, base: dict,
                                data: dict, bucket: str):
        """Fire-and-forget write — submits to thread pool,
        never blocks caller."""
        def _write():
            payload = {**base, "fields": data}
            try:
                client.write(bucket=bucket, record=payload)
            except Exception as e:
                logger.exception('InfluxDB async write failed: %s', e)

        _executor.submit(_write)


class PostgresClient():

    def __init__(self) -> None:
        pass

    @staticmethod
    def postgres_client(conninfo: str) -> ConnectionPool:
        """
        Returns a synchronous connection pool for the sequential pipeline.
        conninfo example:
        'host=localhost port=5432 dbname=mydb user=x password=y'
        """
        try:
            pool = ConnectionPool(conninfo)
            logger.info('PostgreSQL connection pool created successfully')
            return pool

        except Exception as e:
            logger.exception('PostgreSQL connection pool creation failed: %s',
                             e)
            raise

    @staticmethod
    async def postgres_client_async(conninfo: str) -> AsyncConnectionPool:
        """
        Returns an async connection pool for the async pipeline.
        Awaitable — call with:
        pool = await PostgresClient.postgres_client_async(conninfo)
        """
        try:
            pool = AsyncConnectionPool(conninfo)
            await pool.open()
            logger.\
                info('PostgreSQL async connection pool created successfully')
            return pool

        except Exception as e:
            logger.\
                exception('PostgreSQL async connection pool creation failed: %s', e)  # noqa: E501
            raise

    @staticmethod
    def write_detection_data(pool: ConnectionPool,
                             source_id: str,
                             model1_class: str, model1_count: int,
                             model2_class: str, model2_count: int,
                             narration: str | None = None):
        """
        Synchronous insert into detection_results.
        narration is optional — pass None for count-only writes.
        """
        sql = """
            INSERT INTO vision_pipeline_benchmark_analytics
                (source_id, model1_class, model1_count,
                 model2_class, model2_count, narration)
            VALUES (%s, %s, %s, %s, %s, %s)
        """
        try:
            with pool.connection() as conn:
                conn.execute(sql, (source_id, model1_class, model1_count,
                                   model2_class, model2_count, narration))
            logger.info('PostgreSQL write successful')

        except Exception as e:
            logger.exception('PostgreSQL write failed: %s', e)

    @staticmethod
    async def write_detection_data_async(pool: AsyncConnectionPool,
                                         source_id: str,
                                         model1_class: str, model1_count: int,
                                         model2_class: str, model2_count: int,
                                         narration: str | None = None):
        """
        Native async insert — awaitable, no thread pool needed.
        Use in the async pipeline.
        """
        sql = """
            INSERT INTO vision_pipeline_benchmark_analytics
                (source_id, model1_class, model1_count,
                 model2_class, model2_count, narration)
            VALUES (%s, %s, %s, %s, %s, %s)
        """
        try:
            async with pool.connection() as conn:
                await conn.execute(sql, (source_id,
                                         model1_class,
                                         model1_count,
                                         model2_class,
                                         model2_count,
                                         narration))
            logger.info('PostgreSQL async write successful')

        except Exception as e:
            logger.exception('PostgreSQL async write failed: %s', e)
