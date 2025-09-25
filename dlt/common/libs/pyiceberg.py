import os
from typing import Dict, Any, List, Optional

from fsspec import AbstractFileSystem

from dlt import version
from dlt.common import logger
from dlt.common.destination.exceptions import DestinationUndefinedEntity
from dlt.common.time import precise_time
from dlt.common.libs.pyarrow import cast_arrow_schema_types
from dlt.common.libs.utils import load_open_tables
from dlt.common.pipeline import SupportsPipeline
from dlt.common.schema.typing import TWriteDisposition, TTableSchema
from dlt.common.schema.utils import get_first_column_name_with_prop, get_columns_names_with_prop
from dlt.common.utils import assert_min_pkg_version
from dlt.common.exceptions import MissingDependencyException
from dlt.common.storages.configuration import FileSystemCredentials, FilesystemConfiguration
from dlt.common.configuration.specs import CredentialsConfiguration, AwsCredentials, AnyAzureCredentials, AzureCredentialsWithoutDefaults, GcpServiceAccountCredentials
from dlt.common.pendulum import pendulum

from dlt.common.configuration.specs.mixins import WithPyicebergConfig

from dlt.destinations.impl.filesystem.filesystem import FilesystemClient


try:
    from pyiceberg.table import Table as IcebergTable
    from pyiceberg.catalog import Catalog as IcebergCatalog
    from pyiceberg.exceptions import NoSuchTableError
    from pyiceberg.catalog import load_catalog
    import pyarrow as pa
    import pyiceberg.io.pyarrow as _pio
except ModuleNotFoundError:
    raise MissingDependencyException(
        "dlt pyiceberg helpers",
        [f"{version.DLT_PKG_NAME}[pyiceberg]"],
        "Install `pyiceberg` so dlt can create Iceberg tables in the `filesystem` destination.",
    )


# TODO: remove with pyiceberg's release after 0.9.1
_orig_get_kwargs = _pio._get_parquet_writer_kwargs


def _patched_get_parquet_writer_kwargs(table_properties):  # type: ignore[no-untyped-def]
    """Return the original kwargs **plus** store_decimal_as_integer=True."""
    kwargs = _orig_get_kwargs(table_properties)
    kwargs.setdefault("store_decimal_as_integer", True)
    return kwargs


_pio._get_parquet_writer_kwargs = _patched_get_parquet_writer_kwargs


import google.auth
from google.auth.transport.requests import Request


def get_access_token(service_account_file, scopes):
    """
    Retrieves an access token from Google Cloud Platform using service account credentials.

    Args:
        service_account_file: Path to the service account JSON key file.
        scopes: List of OAuth scopes required for your application.

    Returns:
        The access token as a string.
    """

    credentials, name = google.auth.load_credentials_from_file(
        service_account_file, scopes=scopes)

    request = Request()
    credentials.refresh(request)  # Forces token refresh if needed
    return credentials

def ensure_iceberg_compatible_arrow_schema(schema: pa.Schema) -> pa.Schema:
    ARROW_TO_ICEBERG_COMPATIBLE_ARROW_TYPE_MAP = {
        pa.types.is_time32: pa.time64("us"),
        pa.types.is_decimal256: pa.string(),  # pyarrow does not allow downcasting to decimal128
        pa.types.is_dictionary: lambda t_: t_.value_type,
    }
    return cast_arrow_schema_types(schema, ARROW_TO_ICEBERG_COMPATIBLE_ARROW_TYPE_MAP)


def ensure_iceberg_compatible_arrow_data(data: pa.Table) -> pa.Table:
    schema = ensure_iceberg_compatible_arrow_schema(data.schema)
    return data.cast(schema)


def write_iceberg_table(
    table: IcebergTable,
    data: pa.Table,
    write_disposition: TWriteDisposition,
) -> None:
    start_ts = precise_time()
    if write_disposition == "append":
        table.append(ensure_iceberg_compatible_arrow_data(data))
    elif write_disposition == "replace":
        table.overwrite(ensure_iceberg_compatible_arrow_data(data))
    logger.debug(
        f"pyiceberg: {write_disposition} arrow with {data.num_rows} rows to table {table.name()} at"
        f" location {table.location()} took {(precise_time() - start_ts)} seconds."
    )


def merge_iceberg_table(
    table: IcebergTable,
    data: pa.Table,
    schema: TTableSchema,
    load_table_name: str,
) -> None:
    """Merges in-memory Arrow data into on-disk Iceberg table."""
    strategy = schema["x-merge-strategy"]  # type: ignore[typeddict-item]
    if strategy == "upsert":
        # evolve schema
        with table.update_schema() as update:
            update.union_by_name(ensure_iceberg_compatible_arrow_schema(data.schema))

        if "parent" in schema:
            join_cols = [get_first_column_name_with_prop(schema, "unique")]
        else:
            join_cols = get_columns_names_with_prop(schema, "primary_key")

        # TODO: replace the batching method with transaction with pyiceberg's release after 0.9.1
        for rb in data.to_batches(max_chunksize=1_000):
            batch_tbl = pa.Table.from_batches([rb])
            batch_tbl = ensure_iceberg_compatible_arrow_data(batch_tbl)

            table.upsert(
                df=batch_tbl,
                join_cols=join_cols,
                when_matched_update_all=True,
                when_not_matched_insert_all=True,
                case_sensitive=True,
            )
    else:
        raise ValueError(
            f'Merge strategy "{strategy}" is not supported for Iceberg tables. '
            f'Table: "{load_table_name}".'
        )


def get_rest_catalog(credentials: FileSystemCredentials) -> IcebergCatalog:
    """Creates and returns a RestCatalog for Iceberg."""
    # Ensure METASTORE_URL is set in the environment
    if "METASTORE_URL" not in os.environ:
        raise Exception("Missing env: METASTORE_URL.")

    # Handle AWS credentials
    if isinstance(credentials, AwsCredentials):
        session_credentials = credentials.to_pyiceberg_fileio_config()
        return load_catalog(
            name="lakehouse_catalog",
            **{
                "uri": os.environ.get("METASTORE_URL"),
                "s3.access-key-id": session_credentials["s3.access-key-id"],
                "s3.secret-access-key": session_credentials["s3.secret-access-key"],
                "s3.session-token": session_credentials.get("s3.session-token", ""),
                "s3.region": session_credentials.get("s3.region", "us-east-1"),
                "s3.endpoint": session_credentials.get("s3.endpoint"),
                "s3.connect-timeout": session_credentials.get("s3.connect-timeout", 300),
            }
        )
    elif isinstance(credentials, AzureCredentialsWithoutDefaults):
        session_credentials = credentials.to_pyiceberg_fileio_config()
        return load_catalog(
            name="lakehouse_catalog",
            **{
                "uri": os.environ.get("METASTORE_URL"),
                "py-io-impl": "pyiceberg.io.fsspec.FsspecFileIO",
                "adls.connection-string": session_credentials.get("adls.connection-string"),
                "adls.account-name": session_credentials["adls.account-name"],
                "adls.account-key": session_credentials["adls.account-key"]
            }
        )

    elif isinstance(credentials, GcpServiceAccountCredentials):
        # GCS_JSON_KEY_FILE_PATH get this env var for service account file
        service_account_file = os.environ.get("GCS_JSON_KEY_FILE_PATH", None)
        if service_account_file is None:
            raise Exception("GCS_JSON_KEY_FILE_PATH env var is not set, cannot create GCS Catalog")
        if not credentials.get("scopes"):
            credentials.scopes = ["https://www.googleapis.com/auth/cloud-platform"]
        token = get_access_token(service_account_file, credentials.scopes)
        return load_catalog(
            name="lakehouse_catalog",
            **{
                "uri": os.environ.get("METASTORE_URL"),
                "py-io-impl": "pyiceberg.io.fsspec.FsspecFileIO",
                "gcs.project-id": credentials.get("project_id"),
                "gcs.oauth2.token": token,
                "gcs.oauth2.token-expires-at": (pendulum.now().timestamp() + (5 * 60)) * 1000, # 5 minutes
            }
        )
    else:
        raise ValueError("Unsupported or unknown credentials type.")


def get_sql_catalog(
    catalog_name: str,
    uri: str,
    credentials: FileSystemCredentials,
    properties: Dict[str, Any] = None,
) -> IcebergCatalog:  # noqa: F821
    assert_min_pkg_version(
        pkg_name="sqlalchemy",
        version="2.0.18",
        msg=(
            "`sqlalchemy>=2.0.18` is needed for `iceberg` table format on `filesystem` destination."
        ),
    )

    from pyiceberg.catalog.sql import SqlCatalog

    return SqlCatalog(
        catalog_name,
        uri=uri,
        **_get_fileio_config(credentials),
        **(properties or {}),
    )




# def ensure_pyiceberg_local_path(location: str) -> str:
#     """Converts local absolute paths into file urls."""


def evolve_table(
    catalog: IcebergCatalog,
    client: FilesystemClient,
    table_id: str,
    table_location: str,
    schema: Optional[pa.Schema] = None,
) -> IcebergTable:
    try:
        table = catalog.load_table(table_id)
    except NoSuchTableError:
        # add table to catalog
        metadata_path = f"{table_location.rstrip('/')}/metadata"
        if client.fs_client.exists(metadata_path):
            # found metadata; register existing table
            table = register_table(
                table_id, metadata_path, catalog, client.fs_client, client.config
            )
        else:
            raise

    # evolve schema
    if schema is not None:
        with table.update_schema() as update:
            update.union_by_name(ensure_iceberg_compatible_arrow_schema(schema))

    return table


def create_table(
    catalog: IcebergCatalog,
    table_id: str,
    table_location: str,
    schema: pa.Schema,
    partition_columns: Optional[List[str]] = None,
    partition_specs: Optional[List[Dict[str, Any]]] = None,
) -> None:
    # found no metadata; create new table

    with catalog.create_table_transaction(
        table_id,
        schema=ensure_iceberg_compatible_arrow_schema(schema),
        location=table_location,
    ) as txn:
        # add partitioning
        if partition_columns or partition_specs:
            with txn.update_spec() as update_spec:
                # Legacy: simple identity partitioning (dlt standard)
                if partition_columns:
                    for col in partition_columns:
                        update_spec.add_identity(col)

                # Enhanced: advanced partitioning with transforms (new feature)
                if partition_specs:
                    _add_partition_specs(update_spec, partition_specs)


def extract_partition_specs_from_schema(
    table_schema: Dict[str, Any],
    arrow_schema: pa.Schema
) -> Optional[List[Dict[str, Any]]]:
    """Extract partition specifications from dlt table schema.

    Priority system:
    - If ANY column uses advanced partitioning, ALL legacy partitioning is ignored
    - If NO columns use advanced partitioning, legacy partitioning is used

    Advanced formats:
    - {"partition": {"index": 1, "type": "day"}}
    - {"partition": [{"index": 1, "type": "year"}, {"index": 2, "type": "month"}]}

    Legacy format:
    - {"partition": True} - identity partitioning (existing dlt standard)

    Args:
        table_schema: dlt table schema containing column hints
        arrow_schema: PyArrow schema for field type validation

    Returns:
        List of partition specifications or None if no partitions found
    """
    partition_specs = []
    legacy_partitions = []
    has_advanced_partitioning = False
    columns = table_schema.get("columns", {})

    # First pass: collect advanced and legacy partitions separately
    for column_name, column_config in columns.items():
        partition_hint = column_config.get("partition")
        if not partition_hint:
            continue

        # Handle list of partitions (advanced)
        if isinstance(partition_hint, list):
            has_advanced_partitioning = True
            for spec in partition_hint:
                if isinstance(spec, dict) and "index" in spec:
                    partition_specs.append({
                        "column": column_name,
                        "index": spec["index"],
                        "type": spec["type"],
                        "bucket_count": spec.get("bucket_count"),
                        "name": spec.get("name")
                    })

        # Handle single partition with index (advanced)
        elif isinstance(partition_hint, dict) and "index" in partition_hint:
            has_advanced_partitioning = True
            partition_specs.append({
                "column": column_name,
                "index": partition_hint["index"],
                "type": partition_hint["type"],
                "bucket_count": partition_hint.get("bucket_count"),
                "name": partition_hint.get("name")
            })

        # Handle boolean partition (legacy)
        elif partition_hint is True:
            legacy_partitions.append({
                "column": column_name,
                "index": 9999,  # Put legacy partitions last
                "type": "identity",
                "bucket_count": None,
                "name": None
            })

    # Priority logic: advanced takes precedence
    if has_advanced_partitioning:
        # Use only advanced partitioning, ignore legacy
        if legacy_partitions:
            logger.info(
                f"Advanced partitioning detected. Ignoring {len(legacy_partitions)} legacy partition(s): "
                f"{[p['column'] for p in legacy_partitions]}"
            )
        final_specs = partition_specs
    else:
        # No advanced partitioning, use legacy
        final_specs = legacy_partitions

    if not final_specs:
        return None

    # Sort by index to preserve user-specified order
    final_specs.sort(key=lambda x: x["index"])
    return final_specs


def _add_partition_specs(update_spec, partition_specs: List[Dict[str, Any]]) -> None:
    """Add partition specifications to Iceberg update spec.

    Attempts to add all user-specified partitions. If PyIceberg rejects any
    partition (e.g., multiple time partitions on same column), logs a warning
    and continues with remaining partitions instead of failing completely.
    """
    from pyiceberg.transforms import (
        IdentityTransform, BucketTransform, TruncateTransform,
        YearTransform, MonthTransform, DayTransform, HourTransform
    )

    transform_map = {
        "identity": IdentityTransform,
        "bucket": BucketTransform,
        "truncate": TruncateTransform,
        "year": YearTransform,
        "month": MonthTransform,
        "day": DayTransform,
        "hour": HourTransform,
    }

    for spec in partition_specs:
        column = spec["column"]
        transform_type = spec["type"]
        bucket_count = spec.get("bucket_count")
        name = spec.get("name")

        try:
            if transform_type == "identity":
                update_spec.add_identity(column)
            elif transform_type == "bucket":
                if not bucket_count:
                    raise ValueError(f"bucket_count required for bucket transform on {column}")
                transform = BucketTransform(bucket_count)
                if name:
                    update_spec.add_field(column, transform, name)
                else:
                    update_spec.add_field(column, transform)
            else:
                # Time-based or other transforms
                transform_class = transform_map.get(transform_type)
                if not transform_class:
                    raise ValueError(f"Unsupported partition type: {transform_type}")

                transform = transform_class()
                if name:
                    update_spec.add_field(column, transform, name)
                else:
                    update_spec.add_field(column, transform)

        except Exception as e:
            logger.warning(
                f"Failed to add {transform_type} partition on {column}: {e}. "
                f"This may be due to PyIceberg limitations with multiple time partitions on the same column."
            )
            # Continue with other partitions instead of failing completely
            continue


def get_iceberg_tables(
    pipeline: SupportsPipeline,
    *tables: str,
    schema_name: Optional[str] = None,
    include_dlt_tables: bool = False,
) -> Dict[str, IcebergTable]:
    """Returns Iceberg tables in `pipeline.default_schema (default)` or `schema_name` as `pyiceberg.Table` objects.

    Returned object is a dictionary with table names as keys and `Tables` objects as values.
    Optionally filters dictionary by table names specified as `*tables*`.
    Raises ValueError if table name specified as `*tables` is not found. You may try to switch to other
    schemas via `schema_name` argument.
    """
    return load_open_tables(
        pipeline, "iceberg", *tables, schema_name=schema_name, include_dlt_tables=include_dlt_tables
    )


def _get_fileio_config(credentials: CredentialsConfiguration) -> Dict[str, Any]:
    if isinstance(credentials, WithPyicebergConfig):
        return credentials.to_pyiceberg_fileio_config()
    return {}


def get_last_metadata_file(
    metadata_path: str, fs_client: AbstractFileSystem, config: FilesystemConfiguration
) -> str:
    # TODO: read version-hint.txt first and save it in filesystem
    try:
        metadata_files = [f for f in fs_client.ls(metadata_path) if f.endswith(".json")]
    except FileNotFoundError:
        raise DestinationUndefinedEntity(FileNotFoundError(metadata_path))
    if len(metadata_files) == 0:
        raise DestinationUndefinedEntity(FileNotFoundError(metadata_path))
    return make_location(sorted(metadata_files)[-1], config)


def register_table(
    identifier: str,
    metadata_path: str,
    catalog: IcebergCatalog,
    fs_client: AbstractFileSystem,
    config: FilesystemConfiguration,
) -> IcebergTable:
    # last_metadata_file = get_last_metadata_file(metadata_path, fs_client, config)
    # return catalog.register_table(identifier, last_metadata_file)
    return catalog.load_table(identifier)


def make_location(path: str, config: FilesystemConfiguration) -> str:
    # don't use file protocol for local files because duckdb does not support it
    # https://github.com/duckdb/duckdb/issues/13669
    location = config.make_url(path)
    if config.is_local_filesystem and os.name == "nt":
        # pyiceberg cannot deal with windows absolute urls
        location = location.replace("file:///", "file://")
    return location
