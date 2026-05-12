import dataclasses

from typing import Final, Optional, Type, Dict, Any

from dlt.common import logger
from dlt.common.configuration import configspec, resolve_type
from dlt.common.destination.client import (
    CredentialsConfiguration,
    DestinationClientStagingConfiguration,
)
from dlt.common.storages import FilesystemConfigurationWithLocalFiles

from dlt.destinations.impl.filesystem.typing import TCurrentDateTime, TExtraPlaceholders
from dlt.destinations.path_utils import check_layout, get_unused_placeholders


@configspec
class FilesystemDestinationClientConfiguration(FilesystemConfigurationWithLocalFiles, DestinationClientStagingConfiguration):  # type: ignore[misc]
    destination_type: Final[str] = dataclasses.field(  # type: ignore[misc]
        default="filesystem", init=False, repr=False, compare=False
    )
    current_datetime: Optional[TCurrentDateTime] = None
    extra_placeholders: Optional[TExtraPlaceholders] = None
    max_state_files: int = 100
    """Maximum number of pipeline state files to keep; 0 or negative value disables cleanup."""
    always_refresh_views: bool = False
    """Always refresh table scanner views by setting the newest table metadata or globbing table files"""
    iceberg_gc_collect_interval: int = 0
    """How often (in batches) to run gc.collect() during streamed Iceberg writes. Set to 0 to disable."""
    iceberg_write_engine: str = "pyiceberg"
    """Engine to use for all Iceberg writes (append, replace, merge/upsert):
    'pyiceberg' (default, single-process) or 'spark' (distributed via PySpark)."""
    spark_catalog_name: str = "rest"
    """Spark Iceberg catalog name (used when iceberg_write_engine='spark')."""

    @resolve_type("credentials")
    def resolve_credentials_type(self) -> Type[CredentialsConfiguration]:
        return super().resolve_credentials_type()

    def on_resolved(self) -> None:
        # Validate layout and show unused placeholders
        _, layout_placeholders = check_layout(self.layout, self.extra_placeholders)
        unused_placeholders = get_unused_placeholders(
            layout_placeholders, list((self.extra_placeholders or {}).keys())
        )
        if unused_placeholders:
            logger.info(f"Found unused layout placeholders: {', '.join(unused_placeholders)}")
