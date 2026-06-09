"""Iceberg load job using Spark for distributed writes.

Selected when ``DLT_ICEBERG_WRITE_ENGINE=spark`` is set on the filesystem destination.
Spark configuration (catalog URI, credentials, etc.) is passed via
``FilesystemDestinationClientConfiguration.iceberg_spark_config``.
"""

from typing import Any, Optional, Dict

from dlt.common import logger
from dlt.common.schema.exceptions import SchemaCorruptedException

from dlt.destinations.impl.filesystem.filesystem import TableFormatLoadFilesystemJob


class IcebergSparkLoadFilesystemJob(TableFormatLoadFilesystemJob):
    """Load parquet files to an Iceberg table using PySpark.

    Flow
    ----
    1.  Read Spark config from ``job_client.config.iceberg_spark_config``.
    2.  If the target table does not yet exist in the catalog, create it.
    3.  Write / merge parquet files into the Iceberg table.
    """

    def run(self) -> None:
        from dlt.common.libs.spark_iceberg import (
            write_iceberg_table_spark,
            merge_iceberg_table_spark,
        )

        spark_config: Optional[Dict[str, str]] = getattr(
            self._job_client.config, "iceberg_spark_config", None
        )

        # schema.table — the catalog is selected via spark.sql.defaultCatalog in spark_config.
        table_id = f"{self._job_client.dataset_name}.{self.load_table_name}"
        write_disposition = self._load_table["write_disposition"]

        logger.info(
            f"[spark-iceberg] job table={table_id} files={len(self.file_paths)}"
            f" disposition={write_disposition}"
        )

        # Table creation on first load and catalog namespace creation are handled
        # inside write_iceberg_table_spark / merge_iceberg_table_spark.
        if write_disposition == "merge":
            merge_iceberg_table_spark(
                table_id=table_id,
                file_paths=self.file_paths,
                schema=self._load_table,
                load_table_name=self.load_table_name,
                spark_config=spark_config,
            )
        else:
            write_iceberg_table_spark(
                table_id=table_id,
                file_paths=self.file_paths,
                write_disposition=write_disposition,
                spark_config=spark_config,
            )

        logger.info(f"[spark-iceberg] job complete table={table_id}")

    def _resolve_partition_spec(self, arrow_schema: "pa.Schema") -> Any:  # type: ignore[name-defined]
        """Return the partition spec list, or ``None`` if unpartitioned.

        Validates that identity partition columns are not duplicated across
        legacy ``partition_columns`` hints and new ``partition`` hints.
        """
        from dlt.destinations.impl.filesystem.iceberg_adapter import (
            parse_partition_hints,
            create_identity_specs,
        )

        legacy_columns = self._partition_columns
        hint_specs = parse_partition_hints(self._load_table)

        for spec in hint_specs:
            if spec.transform == "identity" and spec.source_column in legacy_columns:
                raise SchemaCorruptedException(
                    self._schema.name,
                    f"Column '{spec.source_column}' is defined both as a partition column "
                    "and in partition hints.",
                )

        identity_specs = create_identity_specs(legacy_columns)
        all_specs = identity_specs + hint_specs
        return all_specs if all_specs else None
