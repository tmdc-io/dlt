import dataclasses

from typing import Final, Optional, Type, Dict, Any, ClassVar, List

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

    state_bucket_url: Optional[str] = None
    """If set, dlt internal tables (_dlt_*) are written here instead of bucket_url.
    e.g. Microsoft Fabric OneLake /Tables vs /Files."""

    iceberg_spark_config: Optional[Dict[str, str]] = None
    """Spark/Iceberg configuration key-value pairs injected into SparkSession when
    ``DLT_ICEBERG_WRITE_ENGINE=spark`` is set.  Use this to configure the REST catalog URI,
    storage credentials, and any other ``spark.*`` properties.

    Example::

        destination.filesystem(
            iceberg_spark_config={
                "spark.sql.catalog.rest": "org.apache.iceberg.spark.SparkCatalog",
                "spark.sql.catalog.rest.catalog-impl": "org.apache.iceberg.rest.RESTCatalog",
                "spark.sql.catalog.rest.uri": "https://my-catalog.example.com",
            }
        )

    .. note::
        This field is a plain ``Dict`` and cannot be populated via nested ``__`` env vars
        (e.g. ``DESTINATION__FILESYSTEM__ICEBERG_SPARK_CONFIG__KEY=value`` will not work).
        Pass the dict directly in Python or via a dlt secrets/config file instead.
    """

    __config_gen_annotations__: ClassVar[List[str]] = ["iceberg_spark_config"]

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
