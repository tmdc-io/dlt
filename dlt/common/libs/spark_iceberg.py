"""Spark-based Iceberg operations: append, replace, and merge via PySpark.

Replaces PyIceberg's single-process paths with distributed Spark execution.

Requires ``pyspark`` and the Iceberg Spark runtime JAR to be available in the
environment (already present in ``tabulario/spark-iceberg`` Docker images).
"""

from __future__ import annotations

import os
import time
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from dlt.common import logger

if TYPE_CHECKING:
    from pyspark.sql import SparkSession


def _build_spark_session(
    catalog_name: str,
    catalog_config: Optional[Dict[str, Any]] = None,
) -> "SparkSession":
    """Create a SparkSession pre-configured for the Iceberg REST catalog.

    Catalog connection details are resolved in this order:
    1. Explicit ``catalog_config`` dict (keys like ``uri``, ``s3.endpoint``, …)
    2. Environment variables (``PYICEBERG_CATALOG__DEFAULT__*``, ``AWS_*``)
    """
    from pyspark.sql import SparkSession

    builder = (
        SparkSession.builder
        .master("local[1]")
        .appName("dlt-iceberg-merge")
        .config("spark.sql.shuffle.partitions", "2")
        .config(
            "spark.sql.extensions",
            "org.apache.iceberg.spark.extensions.IcebergSparkSessionExtensions",
        )
        .config("spark.sql.defaultCatalog", catalog_name)
    )

    cfg = catalog_config or {}

    cat_uri = cfg.get("uri") or os.environ.get("PYICEBERG_CATALOG__DEFAULT__URI", "")
    cat_warehouse = cfg.get("warehouse") or os.environ.get(
        "PYICEBERG_CATALOG__DEFAULT__WAREHOUSE", ""
    )
    s3_endpoint = cfg.get("s3.endpoint") or os.environ.get(
        "PYICEBERG_CATALOG__DEFAULT__S3__ENDPOINT", ""
    )
    s3_access_key = cfg.get("s3.access-key-id") or os.environ.get("AWS_ACCESS_KEY_ID", "")
    s3_secret_key = cfg.get("s3.secret-access-key") or os.environ.get(
        "AWS_SECRET_ACCESS_KEY", ""
    )
    s3_region = cfg.get("s3.region") or os.environ.get("AWS_REGION", "us-east-1")
    s3_path_style = cfg.get("s3.path-style-access", "true")

    spark_confs: Dict[str, str] = {
        f"spark.sql.catalog.{catalog_name}": "org.apache.iceberg.spark.SparkCatalog",
        f"spark.sql.catalog.{catalog_name}.catalog-impl": "org.apache.iceberg.rest.RESTCatalog",
        f"spark.sql.catalog.{catalog_name}.uri": cat_uri,
        f"spark.sql.catalog.{catalog_name}.warehouse": cat_warehouse,
        f"spark.sql.catalog.{catalog_name}.io-impl": "org.apache.iceberg.aws.s3.S3FileIO",
        f"spark.sql.catalog.{catalog_name}.s3.endpoint": s3_endpoint,
        f"spark.sql.catalog.{catalog_name}.s3.access-key-id": s3_access_key,
        f"spark.sql.catalog.{catalog_name}.s3.secret-access-key": s3_secret_key,
        f"spark.sql.catalog.{catalog_name}.s3.path-style-access": str(s3_path_style),
        "spark.hadoop.fs.s3a.endpoint": s3_endpoint,
        "spark.hadoop.fs.s3a.access.key": s3_access_key,
        "spark.hadoop.fs.s3a.secret.key": s3_secret_key,
        "spark.hadoop.fs.s3a.path.style.access": str(s3_path_style),
        "spark.hadoop.fs.s3a.region": s3_region,
        "spark.hadoop.fs.s3a.impl": "org.apache.hadoop.fs.s3a.S3AFileSystem",
    }

    for k, v in spark_confs.items():
        if v:
            builder = builder.config(k, v)

    spark = builder.getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    return spark


def merge_iceberg_table_spark(
    file_paths: List[str],
    table_id: str,
    join_cols: List[str],
    catalog_name: str = "rest",
    catalog_config: Optional[Dict[str, Any]] = None,
    batch_size: int = 2,
) -> None:
    """Run ``MERGE INTO`` on an Iceberg table via Spark.

    Processes files in batches to avoid OOM.  Records the current snapshot
    before starting; if any batch fails the table is rolled back to the
    original snapshot via PyIceberg so the operation is all-or-nothing.
    """
    from pyiceberg.catalog import load_catalog

    spark = _build_spark_session(catalog_name, catalog_config)
    full_table = f"{catalog_name}.{table_id}"

    py_catalog = load_catalog("default")
    py_table = py_catalog.load_table(table_id)
    original_snapshot = py_table.current_snapshot()
    original_snapshot_id = original_snapshot.snapshot_id if original_snapshot else None

    if original_snapshot_id:
        logger.info(
            f"[spark-merge] Saved rollback point: snapshot {original_snapshot_id}"
        )

    target_cols = [c for c in spark.table(full_table).columns]
    non_pk = [c for c in target_cols if c not in join_cols]

    on_clause = " AND ".join(f"t.{c} = s.{c}" for c in join_cols)
    set_clause = ", ".join(f"t.{c} = s.{c}" for c in non_pk)
    insert_cols = ", ".join(target_cols)
    insert_vals = ", ".join(f"s.{c}" for c in target_cols)

    merge_sql = f"""
    MERGE INTO {full_table} t
    USING __dlt_spark_updates s
    ON {on_clause}
    WHEN MATCHED THEN UPDATE SET {set_clause}
    WHEN NOT MATCHED THEN INSERT ({insert_cols}) VALUES ({insert_vals})
    """

    t0 = time.time()
    total_rows = 0
    n_files = len(file_paths)

    try:
        for i in range(0, n_files, batch_size):
            batch = file_paths[i : i + batch_size]
            batch_num = i // batch_size + 1
            total_batches = (n_files + batch_size - 1) // batch_size

            try:
                updates = spark.read.parquet(*batch)
                batch_rows = updates.count()
                total_rows += batch_rows
                updates.createOrReplaceTempView("__dlt_spark_updates")

                logger.info(
                    f"[spark-merge] Batch {batch_num}/{total_batches}: "
                    f"MERGE {batch_rows:,} rows from {len(batch)} file(s)"
                )
                spark.sql(merge_sql)
                logger.info(f"[spark-merge] Batch {batch_num} done")
                del updates
                spark.catalog.clearCache()
            finally:
                try:
                    spark.catalog.dropTempView("__dlt_spark_updates")
                except Exception:
                    pass
    except Exception:
        if original_snapshot_id:
            logger.error(
                f"[spark-merge] Merge failed — rolling back to snapshot "
                f"{original_snapshot_id}"
            )
            py_table.refresh()
            py_table.manage_snapshots().set_current_snapshot(
                original_snapshot_id
            ).commit()
            logger.info("[spark-merge] Rollback complete")
        else:
            logger.error("[spark-merge] Merge failed — no snapshot to rollback to")
        raise

    elapsed = time.time() - t0
    logger.info(
        f"[spark-merge] MERGE completed in {elapsed:.1f}s "
        f"({total_rows:,} rows, {n_files} files)"
    )


def write_iceberg_table_spark(
    file_paths: List[str],
    table_id: str,
    write_disposition: str,
    catalog_name: str = "rest",
    catalog_config: Optional[Dict[str, Any]] = None,
    batch_size: int = 5,
) -> None:
    """Append or replace data in an Iceberg table via Spark.

    Processes files in batches to avoid OOM in memory-constrained environments.
    """
    spark = _build_spark_session(catalog_name, catalog_config)
    full_table = f"{catalog_name}.{table_id}"

    t0 = time.time()
    total_rows = 0
    n_files = len(file_paths)

    if write_disposition == "replace":
        logger.info(f"[spark-write] REPLACE: loading all {n_files} file(s) at once")
        df = spark.read.parquet(*file_paths)
        df.writeTo(full_table).overwritePartitions()
        total_rows = df.count()
    else:
        for i in range(0, n_files, batch_size):
            batch = file_paths[i : i + batch_size]
            batch_num = i // batch_size + 1
            total_batches = (n_files + batch_size - 1) // batch_size
            logger.info(
                f"[spark-write] Batch {batch_num}/{total_batches}: "
                f"{len(batch)} file(s) [{i+1}-{min(i+batch_size, n_files)}/{n_files}]"
            )
            df = spark.read.parquet(*batch)
            df.writeTo(full_table).append()
            batch_rows = df.count()
            total_rows += batch_rows
            del df
            spark.catalog.clearCache()
            logger.info(f"[spark-write] Batch {batch_num} done: {batch_rows:,} rows appended")

    elapsed = time.time() - t0
    logger.info(
        f"[spark-write] {write_disposition.upper()} completed in {elapsed:.1f}s "
        f"({total_rows:,} rows, {n_files} files)"
    )
