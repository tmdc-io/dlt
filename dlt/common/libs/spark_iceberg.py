"""Spark-based Iceberg write operations.

Distributed writes via PySpark for large-scale data loading.
"""

from typing import List, Optional, Dict, Any

from dlt.common import logger
from dlt.common.schema.typing import TWriteDisposition, TTableSchema


def _ensure_namespace(spark: "SparkSession", table_id: str) -> None:  # type: ignore[name-defined]
    """Create catalog namespace (schema) if it doesn't exist."""
    parts = table_id.split(".")
    if len(parts) >= 3:
        namespace = ".".join(parts[:-1])
        spark.sql(f"CREATE NAMESPACE IF NOT EXISTS {namespace}")


def _table_exists(spark: "SparkSession", table_id: str) -> bool:  # type: ignore[name-defined]
    """Return True if the table is registered in the Spark catalog.

    Uses ``spark.catalog.tableExists`` which correctly resolves the default
    catalog for two-part ``schema.table`` names, and accepts fully-qualified
    ``catalog.schema.table`` names as well.
    """
    try:
        return spark.catalog.tableExists(table_id)
    except Exception:
        return False


def _build_spark_session(spark_config: Optional[Dict[str, str]]) -> "SparkSession":  # type: ignore[name-defined]
    """Build or get a SparkSession, optionally injecting catalog config."""
    try:
        from pyspark.sql import SparkSession
    except ImportError:
        raise ImportError("PySpark not installed. Install with: pip install pyspark")

    builder = SparkSession.builder.appName("dlt-iceberg")
    if spark_config:
        for k, v in spark_config.items():
            builder = builder.config(k, str(v))
    return builder.getOrCreate()


def write_iceberg_table_spark(
    table_id: str,
    file_paths: List[str],
    write_disposition: TWriteDisposition,
    spark_config: Optional[Dict[str, str]] = None,
    gc_collect_interval: int = 10,
    upload_chunk_size: int = 8 * 1024 * 1024,
) -> None:
    """Write parquet files to an Iceberg table via Spark.

    Args:
        table_id: Fully qualified table name (catalog.schema.table)
        file_paths: List of parquet files to load
        write_disposition: ``append`` or ``replace``
        spark_config: Spark/Iceberg configuration key-value pairs injected into SparkSession
        gc_collect_interval: Unused — kept for API compatibility
        upload_chunk_size: Unused — kept for API compatibility
    """
    if write_disposition not in ("append", "replace"):
        raise ValueError(
            f"write_iceberg_table_spark: unsupported write_disposition={write_disposition!r}. "
            "Use 'append' or 'replace'. For merge, call merge_iceberg_table_spark."
        )

    spark = _build_spark_session(spark_config)
    logger.info(
        f"[spark-iceberg] write table={table_id} disposition={write_disposition}"
        f" files={len(file_paths)}"
    )

    df = spark.read.parquet(*file_paths)
    writer = df.writeTo(table_id).tableProperty("format-version", "2")

    if write_disposition == "replace":
        _ensure_namespace(spark, table_id)
        writer.createOrReplace()
    else:  # append
        if _table_exists(spark, table_id):
            writer.append()
        else:
            _ensure_namespace(spark, table_id)
            writer.create()

    logger.info(f"[spark-iceberg] write complete table={table_id}")


def merge_iceberg_table_spark(
    table_id: str,
    file_paths: List[str],
    schema: TTableSchema,
    load_table_name: str,
    spark_config: Optional[Dict[str, str]] = None,
    gc_collect_interval: int = 10,
    upload_chunk_size: int = 8 * 1024 * 1024,
) -> None:
    """Upsert parquet files into an Iceberg table via Spark.

    Implements merge as: read existing → anti-join on primary keys (unchanged rows)
    → union with incoming rows → atomic createOrReplace. This avoids reliance on
    Iceberg row-level delete support, which varies across catalog implementations.

    Args:
        table_id: Fully qualified table name (catalog.schema.table)
        file_paths: List of parquet files to load
        schema: DLT table schema (used to extract primary key columns)
        load_table_name: Table name used for logging
        spark_config: Spark/Iceberg configuration key-value pairs injected into SparkSession
        gc_collect_interval: Unused — kept for API compatibility
        upload_chunk_size: Unused — kept for API compatibility
    """
    from dlt.common.schema.utils import get_columns_names_with_prop

    spark = _build_spark_session(spark_config)
    logger.info(f"[spark-iceberg] merge table={table_id} files={len(file_paths)}")

    incoming_df = spark.read.parquet(*file_paths)

    primary_keys = get_columns_names_with_prop(schema, "primary_key", include_incomplete=True)
    if not primary_keys:
        logger.warning(
            f"[spark-iceberg] no primary_key defined on {load_table_name!r},"
            " falling back to append"
        )
        if _table_exists(spark, table_id):
            incoming_df.writeTo(table_id).append()
        else:
            _ensure_namespace(spark, table_id)
            incoming_df.writeTo(table_id).tableProperty("format-version", "2").create()
        logger.info(f"[spark-iceberg] merge complete (append fallback) table={table_id}")
        return

    if not _table_exists(spark, table_id):
        # No existing data — treat first merge as a plain create
        _ensure_namespace(spark, table_id)
        incoming_df.writeTo(table_id).tableProperty("format-version", "2").create()
        logger.info(f"[spark-iceberg] merge complete (initial create) table={table_id}")
        return

    # spark.table() correctly resolves the default catalog for 2-part schema.table names.
    # spark.read.format("iceberg").load() does NOT — it looks for a "default_iceberg" catalog.
    existing_df = spark.table(table_id)
    # Rows in target not matched by any incoming primary key → keep as-is
    unchanged_df = existing_df.join(
        incoming_df.select(primary_keys), on=primary_keys, how="left_anti"
    )
    # Unchanged rows + all incoming rows (updated + new) = full upsert result
    result_df = unchanged_df.unionByName(incoming_df)
    result_df.writeTo(table_id).tableProperty("format-version", "2").createOrReplace()

    logger.info(f"[spark-iceberg] merge complete table={table_id}")


def create_table_spark(
    table_id: str,
    table_location: str,
    schema: Any,
    partition_spec: Optional[Any] = None,
    spark_config: Optional[Dict[str, str]] = None,
) -> None:
    """Create an empty Iceberg table via Spark.

    Args:
        table_id: Fully qualified table name (catalog.schema.table)
        table_location: Physical storage location (e.g. ``s3://bucket/path``)
        schema: Schema for the new table.  Accepts either a PyArrow ``pa.Schema``
            or a PyIceberg ``Schema`` (auto-converted to Arrow).
        partition_spec: Reserved for future use; partitioning via this path is not
            yet implemented — pass ``None``.
        spark_config: Spark/Iceberg configuration key-value pairs injected into SparkSession
    """
    import pyarrow as pa

    # Normalise: accept pyiceberg Schema → convert to Arrow first
    if not isinstance(schema, pa.Schema):
        try:
            from pyiceberg.io.pyarrow import schema_to_pyarrow

            schema = schema_to_pyarrow(schema)
        except Exception as e:
            raise TypeError(
                f"create_table_spark: cannot convert schema of type"
                f" {type(schema).__name__!r} to a PyArrow schema: {e}"
            ) from e

    spark = _build_spark_session(spark_config)
    logger.info(f"[spark-iceberg] create_table table={table_id} location={table_location}")

    spark_schema = _arrow_schema_to_spark(schema)
    empty_df = spark.createDataFrame([], spark_schema)
    empty_df.writeTo(table_id).option("location", table_location).tableProperty(
        "format-version", "2"
    ).create()

    logger.info(f"[spark-iceberg] create_table complete table={table_id}")


def _arrow_schema_to_spark(arrow_schema: Any) -> Any:
    """Convert a PyArrow schema to a PySpark StructType."""
    try:
        from pyspark.sql import SparkSession
        from pyspark.sql.pandas.types import from_arrow_schema

        return from_arrow_schema(arrow_schema)
    except Exception as e:
        logger.warning(
            f"[spark-iceberg] arrow→spark schema conversion failed: {e}; "
            "falling back to from_arrow_type"
        )
        # Older PySpark versions expose this differently
        try:
            from pyspark.sql.types import StructType, StructField
            from pyspark.sql.pandas.types import from_arrow_type

            return StructType(
                [StructField(f.name, from_arrow_type(f.type), f.nullable) for f in arrow_schema]
            )
        except Exception as e2:
            raise RuntimeError(
                f"Cannot convert Arrow schema to Spark schema: {e2}"
            ) from e2
