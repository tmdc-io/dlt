import os
from typing import Dict, Any, List, Optional, Tuple, Union
from pathlib import Path
import warnings

from fsspec import AbstractFileSystem
from packaging.version import Version

from dlt import version
from dlt.common import logger
from dlt.common.time import precise_time
from dlt.common.destination.exceptions import DestinationUndefinedEntity
from dlt.common.libs.pyarrow import cast_arrow_schema_types
from dlt.common.libs.utils import load_open_tables
from dlt.common.pipeline import SupportsPipeline
from dlt.common.schema.typing import TWriteDisposition, TTableSchema
from dlt.common.schema.utils import get_first_column_name_with_prop, get_columns_names_with_prop
from dlt.common.utils import assert_min_pkg_version
from dlt.common.exceptions import MissingDependencyException
from dlt.common.storages.configuration import FileSystemCredentials, FilesystemConfiguration
from dlt.common.configuration.specs import CredentialsConfiguration
from dlt.common.data_writers.buffered import BufferedDataWriter
from dlt.common.configuration.specs.mixins import WithPyicebergConfig
from dlt.common.configuration.inject import with_config
from dlt.common.configuration import configspec
from dlt.common.configuration.specs import BaseConfiguration

from dlt.destinations.impl.filesystem.filesystem import FilesystemClient


try:
    import pyiceberg
    from pyiceberg.table import Table as IcebergTable
    from pyiceberg.catalog import Catalog as IcebergCatalog
    from pyiceberg.exceptions import NoSuchTableError
    from pyiceberg.partitioning import (
        UNPARTITIONED_PARTITION_SPEC,
        PartitionSpec as IcebergPartitionSpec,
    )
    import pyarrow as pa
    from pydantic import BaseModel, ConfigDict, Field
except ModuleNotFoundError:
    raise MissingDependencyException(
        "dlt pyiceberg helpers",
        [f"{version.DLT_PKG_NAME}[pyiceberg]"],
        "Install `pyiceberg` so dlt can create Iceberg tables in the `filesystem` destination.",
    )

pyiceberg_semver = Version(pyiceberg.__version__)

if pyiceberg_semver < Version("0.10.0"):
    import pyiceberg.io.pyarrow as _pio

    _orig_get_kwargs = _pio._get_parquet_writer_kwargs

    def _patched_get_parquet_writer_kwargs(table_properties):  # type: ignore[no-untyped-def]
        """Return the original kwargs **plus** store_decimal_as_integer=True."""
        kwargs = _orig_get_kwargs(table_properties)
        kwargs.setdefault("store_decimal_as_integer", True)
        return kwargs

    _pio._get_parquet_writer_kwargs = _patched_get_parquet_writer_kwargs


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


# Internal streaming constants — not public API, not environment-tunable.
_UPLOAD_CHUNK_BYTES = 8 * 1024 * 1024  # read window when uploading a parquet file to remote IO
_UPSERT_BATCH_ROWS = 1_000            # max rows per in-memory batch during merge/upsert
_GC_INTERVAL_BATCHES = 10            # call gc.collect() every N batches to bound RSS growth


def _inject_iceberg_field_ids(arrow_schema: pa.Schema, table: IcebergTable) -> pa.Schema:
    """Return arrow_schema with Iceberg field-ID metadata and nullability injected per field.

    add_files validates that parquet required/optional matches the Iceberg schema.
    Primary-key and dlt internal columns are required in the Iceberg schema, so the
    parquet fields must also be non-nullable. Field IDs are also needed so add_files
    can map columns without falling back to name mapping.
    """
    try:
        from pyiceberg.io.pyarrow import schema_to_pyarrow

        iceberg_arrow = schema_to_pyarrow(table.schema())
        iceberg_by_name = {f.name: f for f in iceberg_arrow}
        new_fields = []
        for f in arrow_schema:
            iceberg_f = iceberg_by_name.get(f.name)
            if iceberg_f is not None:
                new_fields.append(
                    pa.field(
                        f.name,
                        f.type,
                        nullable=iceberg_f.nullable,
                        metadata=iceberg_f.metadata or f.metadata,
                    )
                )
            else:
                new_fields.append(f)
        return pa.schema(new_fields, metadata=arrow_schema.metadata)
    except Exception as e:
        logger.debug(f"pyiceberg: field-id injection skipped, falling back to name mapping: {e}")
        return arrow_schema


def _upload_parquet_to_remote(
    arrow_table: pa.Table,
    data_location: str,
    table_io: Any,
    prefix: str = "batch",
    iceberg_table: Optional[IcebergTable] = None,
    upload_chunk_bytes: int = _UPLOAD_CHUNK_BYTES,
) -> str:
    """Write an Arrow table to a temp Parquet file and upload it to remote storage.

    If iceberg_table is provided, Iceberg field IDs are injected into the
    Parquet schema metadata so add_files can match columns without a name mapping.
    """
    import uuid
    import tempfile

    import pyarrow.parquet as pq

    write_table = arrow_table
    if iceberg_table is not None:
        schema_with_ids = _inject_iceberg_field_ids(arrow_table.schema, iceberg_table)
        if schema_with_ids is not arrow_table.schema:
            write_table = pa.Table.from_arrays(
                [arrow_table.column(name) for name in arrow_table.schema.names],
                schema=schema_with_ids,
            )

    with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as tmp:
        temp_path = tmp.name
    try:
        pq.write_table(
            write_table, temp_path, compression="snappy", store_decimal_as_integer=True
        )

        remote_path = f"{data_location}/{prefix}-{uuid.uuid4()}.parquet"
        output = table_io.new_output(remote_path)
        with open(temp_path, "rb") as fh, output.create() as remote_file:
            while True:
                chunk = fh.read(upload_chunk_bytes)
                if not chunk:
                    break
                remote_file.write(chunk)
    finally:
        os.remove(temp_path)

    return remote_path


def _delete_files(table_io: Any, paths: List[str]) -> None:
    """Best-effort delete of pre-written remote files on commit failure."""
    for path in paths:
        try:
            table_io.delete(path)
        except Exception as e:
            logger.warning(f"pyiceberg: failed to delete orphan file {path}: {e}")


def write_iceberg_table(
    table: IcebergTable,
    data: Union[pa.Table, pa.RecordBatchReader],
    write_disposition: TWriteDisposition,
) -> None:
    start_ts = precise_time()

    mode = "streamed" if isinstance(data, pa.RecordBatchReader) else "in-memory"
    logger.info(
        f"[pyiceberg-write] enter"
        f" table={table.name()} disposition={write_disposition} mode={mode}"
    )

    if isinstance(data, pa.RecordBatchReader):
        _, upload_chunk_bytes = get_iceberg_config_tuning()
        _write_iceberg_table_streamed(table, data, write_disposition, upload_chunk_bytes)
    else:
        if write_disposition == "append":
            table.append(ensure_iceberg_compatible_arrow_data(data))
        elif write_disposition == "replace":
            table.overwrite(ensure_iceberg_compatible_arrow_data(data))
        logger.debug(
            f"pyiceberg: {write_disposition} arrow with {data.num_rows} rows to table"
            f" {table.name()} at location {table.location()} took"
            f" {(precise_time() - start_ts)} seconds."
        )


def _write_iceberg_table_streamed(
    table: IcebergTable,
    reader: pa.RecordBatchReader,
    write_disposition: TWriteDisposition,
    upload_chunk_bytes: int = _UPLOAD_CHUNK_BYTES,
) -> None:
    """Streams Arrow batches as individual parquet files via Iceberg's IO.

    Memory stays constant: only one batch + one parquet file in memory at a time.

    For unpartitioned tables: all files committed in ONE atomic snapshot via txn.add_files().
    For partitioned tables: uses txn.append() per batch (partition-aware, multiple snapshots).
    """
    start_ts = precise_time()
    is_partitioned = table.spec() != UNPARTITIONED_PARTITION_SPEC

    if is_partitioned:
        total_rows, batch_count, data_files_desc = _write_streamed_partitioned(
            table, reader, write_disposition
        )
    else:
        total_rows, batch_count, n_files = _write_streamed_unpartitioned_atomic(
            table, reader, write_disposition, upload_chunk_bytes
        )
        data_files_desc = f"{n_files} data files (atomic)"

    logger.debug(
        f"pyiceberg: streamed {write_disposition} with {total_rows} rows in {batch_count}"
        f" batches ({data_files_desc}) to table {table.name()} at location"
        f" {table.location()} took {(precise_time() - start_ts)} seconds."
    )


def _write_streamed_partitioned(
    table: IcebergTable,
    reader: pa.RecordBatchReader,
    write_disposition: TWriteDisposition,
) -> Tuple[int, int, str]:
    """Write streamed batches to a partitioned Iceberg table using txn.append().

    Each batch creates its own snapshot (pyiceberg limitation), but partition-aware
    file layout is handled correctly by pyiceberg's writer.
    """
    import gc

    from pyiceberg.expressions import AlwaysTrue

    total_rows = 0
    batch_count = 0

    with table.transaction() as txn:
        if write_disposition == "replace" and table.current_snapshot():
            txn.delete(delete_filter=AlwaysTrue())

        for batch in reader:
            batch_count += 1
            batch_table = ensure_iceberg_compatible_arrow_data(pa.Table.from_batches([batch]))
            txn.append(batch_table)
            total_rows += batch_table.num_rows
            del batch_table

            if batch_count % _GC_INTERVAL_BATCHES == 0:
                gc.collect()
            if batch_count % 10 == 0:
                logger.debug(
                    f"pyiceberg: streamed {batch_count} batches, {total_rows} rows so far"
                )

    return total_rows, batch_count, "via pyiceberg writer (partition-aware)"


def _write_streamed_unpartitioned_atomic(
    table: IcebergTable,
    reader: pa.RecordBatchReader,
    write_disposition: TWriteDisposition,
    upload_chunk_bytes: int = _UPLOAD_CHUNK_BYTES,
) -> Tuple[int, int, int]:
    """Write streamed batches to an unpartitioned Iceberg table, committed atomically.

    Each batch is written to disk independently (constant memory), then all file
    paths are registered in a single transaction commit. This ensures readers
    never see partial writes.
    """
    import gc

    from pyiceberg.expressions import AlwaysTrue

    data_location = f"{table.location()}/data"
    accumulated_files: List[str] = []
    total_rows = 0
    batch_count = 0

    try:
        # Stream-write each batch to disk (constant memory per batch)
        for batch in reader:
            batch_count += 1
            batch_table = ensure_iceberg_compatible_arrow_data(pa.Table.from_batches([batch]))
            file_path = _upload_parquet_to_remote(
                batch_table,
                data_location,
                table.io,
                prefix="batch",
                iceberg_table=table,
                upload_chunk_bytes=upload_chunk_bytes,
            )
            accumulated_files.append(file_path)
            total_rows += batch_table.num_rows
            del batch_table

            if batch_count % _GC_INTERVAL_BATCHES == 0:
                gc.collect()
            if batch_count % 10 == 0:
                logger.debug(
                    f"pyiceberg: streamed {batch_count} batches, {total_rows} rows so far"
                )

        # Atomic commit: all files registered in ONE snapshot
        with table.transaction() as txn:
            if write_disposition == "replace" and table.current_snapshot():
                txn.delete(delete_filter=AlwaysTrue())
            txn.add_files(accumulated_files, check_duplicate_files=False)

    except Exception:
        _delete_files(table.io, accumulated_files)
        raise

    return total_rows, batch_count, len(accumulated_files)


def merge_iceberg_table(
    table: IcebergTable,
    data: Union[pa.Table, pa.RecordBatchReader],
    schema: TTableSchema,
    load_table_name: str,
) -> None:
    """Merges Arrow data into on-disk Iceberg table.

    Accepts pa.Table or streaming RecordBatchReader.
    """
    strategy = schema["x-merge-strategy"]  # type: ignore[typeddict-item]
    mode = "streamed" if isinstance(data, pa.RecordBatchReader) else "in-memory"
    logger.info(
        f"[pyiceberg-merge] enter"
        f" table={load_table_name} strategy={strategy} mode={mode}"
    )
    if strategy in ("upsert", "insert-only"):
        arrow_schema = ensure_iceberg_compatible_arrow_schema(data.schema)

        with table.update_schema() as update:
            update.union_by_name(arrow_schema)

        if "parent" in schema:
            join_cols = [get_first_column_name_with_prop(schema, "unique")]
        else:
            join_cols = get_columns_names_with_prop(schema, "primary_key")

        _, upload_chunk_bytes = get_iceberg_config_tuning()
        _upsert_iceberg_table(
            table,
            data,
            join_cols,
            strategy,
            upload_chunk_bytes,
        )
    else:
        raise ValueError(
            f'Merge strategy "{strategy}" is not supported for Iceberg tables. '
            f'Table: "{load_table_name}".'
        )


def _process_upsert_batch(
    batch_tbl: pa.Table,
    table: IcebergTable,
    join_cols: List[str],
    strategy: str,
    has_existing_data: bool,
) -> Tuple[pa.Table, Optional[pa.Table]]:
    """Classify one batch into inserts and updates without applying either.

    Returns ``(rows_to_insert, rows_to_update)``.
    rows_to_update is None when strategy is 'insert-only' or there are no matches.
    Keeping classification separate from writes lets the caller commit overwrite
    and append in **separate transactions**, which is required to avoid SIGILL in
    PyIceberg when two writer-style operations share one transaction.
    """
    if not has_existing_data:
        return batch_tbl, None

    from pyiceberg.table import upsert_util
    from pyiceberg.io.pyarrow import expression_to_pyarrow
    from pyiceberg.expressions.visitors import bind

    matched_predicate = upsert_util.create_match_filter(batch_tbl, join_cols)
    matched_existing = table.scan(
        row_filter=matched_predicate, case_sensitive=True
    ).to_arrow()

    rows_to_update: Optional[pa.Table] = None
    if strategy == "upsert":
        candidate = upsert_util.get_rows_to_update(batch_tbl, matched_existing, join_cols)
        if len(candidate) > 0:
            rows_to_update = candidate

    if len(matched_existing) > 0:
        expr_match = upsert_util.create_match_filter(matched_existing, join_cols)
        expr_bound = bind(table.schema(), expr_match, case_sensitive=True)
        expr_arrow = expression_to_pyarrow(expr_bound)
        rows_to_insert = batch_tbl.filter(~expr_arrow)
    else:
        rows_to_insert = batch_tbl

    del matched_existing
    return rows_to_insert, rows_to_update


def _upsert_iceberg_table(
    table: IcebergTable,
    data: Union[pa.Table, pa.RecordBatchReader],
    join_cols: List[str],
    strategy: str,
    upload_chunk_bytes: int = _UPLOAD_CHUNK_BYTES,
) -> None:
    """Upserts Arrow data into an Iceberg table, one transaction per operation type per batch.

    Overwrite and insert are always committed in **separate transactions** — even for
    unpartitioned tables — because PyIceberg crashes (SIGILL) when two writer-style
    operations (overwrite + append, or overwrite + add_files) share one transaction.

    Each batch is committed independently so later batches can observe earlier writes,
    keeping duplicate keys that span batches correct.
    """
    import gc
    from pyiceberg.table import upsert_util

    start_ts = precise_time()
    is_partitioned = table.spec() != UNPARTITIONED_PARTITION_SPEC
    total_updated = 0
    total_inserted = 0
    batch_count = 0

    batches = (
        data
        if isinstance(data, pa.RecordBatchReader)
        else data.to_batches(max_chunksize=_UPSERT_BATCH_ROWS)
    )

    for batch in batches:
        batch_count += 1
        batch_tbl = ensure_iceberg_compatible_arrow_data(pa.Table.from_batches([batch]))
        has_existing_data = table.current_snapshot() is not None

        rows_to_insert, rows_to_update = _process_upsert_batch(
            batch_tbl, table, join_cols, strategy, has_existing_data
        )

        # Txn 1: overwrite matched rows (never combined with append/add_files)
        if rows_to_update is not None:
            overwrite_filter = upsert_util.create_match_filter(rows_to_update, join_cols)
            with table.transaction() as txn:
                txn.overwrite(rows_to_update, overwrite_filter=overwrite_filter)
            total_updated += len(rows_to_update)

        # Txn 2: insert new rows (separate transaction, safe for both partitioned and not)
        if len(rows_to_insert) > 0:
            total_inserted += len(rows_to_insert)
            if is_partitioned:
                with table.transaction() as txn:
                    txn.append(rows_to_insert)
            else:
                remote_path = _upload_parquet_to_remote(
                    rows_to_insert,
                    f"{table.location()}/data",
                    table.io,
                    prefix="upsert",
                    iceberg_table=table,
                    upload_chunk_bytes=upload_chunk_bytes,
                )
                try:
                    with table.transaction() as txn:
                        txn.add_files([remote_path], check_duplicate_files=False)
                except Exception:
                    _delete_files(table.io, [remote_path])
                    raise

        del batch_tbl
        if batch_count % _GC_INTERVAL_BATCHES == 0:
            gc.collect()
        if batch_count % 10 == 0:
            logger.debug(
                f"pyiceberg: upsert streamed {batch_count} batches,"
                f" {total_inserted} inserts, {total_updated} updates so far"
            )

    logger.debug(
        f"pyiceberg: upsert {total_updated} updated, {total_inserted} inserted"
        f" in {batch_count} batches"
        f" into table {table.name()} took {(precise_time() - start_ts)} seconds."
    )


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


class CatalogNotFoundError(Exception):
    """Raised when a catalog cannot be found in the specified configuration method"""

    pass


class PyicebergCatalogConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    type: str = Field(..., description="Iceberg catalog type")  # noqa
    uri: str = Field(..., description="Iceberg catalog URI")
    warehouse: str = Field(..., description="Warehouse name")


@configspec
class IcebergConfig(BaseConfiguration):
    # Iceberg catalog configuration
    iceberg_catalog_name: str = "default"
    """Name of the Iceberg catalog to use. Corresponds to catalog name in .pyiceberg.yaml"""

    iceberg_catalog_type: Optional[str] = "sql"
    """Type of Iceberg catalog: 'sql', 'rest', 'glue', 'hive', etc."""

    iceberg_catalog_config: Optional[Dict[str, Any]] = None
    """
    Optional dictionary with complete catalog configuration.
    If provided, will be used instead of loading from .pyiceberg.yaml.
    Example for REST catalog:
        {
            'type': 'rest',
            'uri': 'https://catalog.example.com',
            'warehouse': 'my_warehouse',
            'credential': 'token',
            'scope': 'PRINCIPAL_ROLE:ALL'
        }
    Example for SQL catalog:
        {
            'type': 'sql',
            'uri': 'postgresql://user:pass@localhost/catalog'
        }

    Example for secrets.toml:
        [iceberg_catalog]
        iceberg_catalog_name = "default"
        iceberg_catalog_type = "rest"

        [iceberg_catalog.iceberg_catalog_config]
        uri = "http://localhost:8181/catalog"
        warehouse = "default"
        header.X-Iceberg-Access-Delegation = "remote-signing"
        py-io-impl = "pyiceberg.io.fsspec.FsspecFileIO"
        s3.endpoint = "https://cool-bucket.com/"
        s3.access-key-id = "cool-bucket-access-key"
        s3.secret-access-key = "cool-bucket-secret-key"
        s3.region = "cool-bucket-region"
    """

    # Performance tuning — set via env var:
    #   ICEBERG_CATALOG__ICEBERG_UPLOAD_CHUNK_BYTES=33554432
    # Arrow batch size is controlled via DATA_WRITER__BUFFER_MAX_ITEMS (default 5 000).

    iceberg_upload_chunk_bytes: int = _UPLOAD_CHUNK_BYTES
    """Bytes per read when uploading a parquet file to remote storage (default 8 MB)."""


def _load_catalog_from_pyiceberg(
    catalog_name: str,
) -> IcebergCatalog:
    """Load Iceberg catalog through pyiceberg load_catalog mechanism. See https://py.iceberg.apache.org/configuration/#setting-configuration-values

    Args:
        catalog_name: Name of the catalog to load from YAML

    Returns:
        IcebergCatalog instance loaded from YAML configuration

    Raises:
        CatalogNotFoundError: If no .pyiceberg.yaml file found or catalog not in file

    Search paths (in order):
        1. PYICEBERG_HOME environment variable (pyiceberg standard)
        2. DLT run directory
        3. DLT settings directory

    Example .pyiceberg.yaml:
        catalog:
          my_catalog:
            type: rest
            uri: https://catalog.example.com
            warehouse: my_warehouse
            credential: token
    """
    from pyiceberg.catalog import load_catalog
    import dlt

    active_run_context = dlt.current.run_context()

    # Search through potential paths for Iceberg Config
    search_paths = []

    pyiceberg_home = os.environ.get("PYICEBERG_HOME")
    if pyiceberg_home:
        search_paths.append(Path(pyiceberg_home) / ".pyiceberg.yaml")

    # Add nilus-specific paths
    search_paths.extend(
        [
            Path(active_run_context.run_dir) / ".pyiceberg.yaml",
            Path(active_run_context.get_setting(".pyiceberg.yaml")),
        ]
    )

    # Search for the first existing config file and confirm 'catalog:' is present
    no_config_file_found = True
    for path in search_paths:
        if path.exists():
            logger.debug(f"Searching for catalog configuration in: {path}")
            with open(path, "r", encoding="utf-8") as f:
                contents = f.read()
                if "catalog:" in contents:
                    no_config_file_found = False
                    break

    # Check if any PYICEBERG_CATALOG_* environment variable is set
    pyiceberg_env_var = any(key.startswith("PYICEBERG_CATALOG_") for key in os.environ)

    # If no config file was found, raise error
    if no_config_file_found and not pyiceberg_env_var:
        raise CatalogNotFoundError(
            "No .pyiceberg.yaml file found. Searched in:"
            f" {', '.join(str(p) for p in search_paths)}. No PYICEBERG_CATALOG_* environment"
            " variables found."
        )

    return load_catalog(catalog_name)


def _load_catalog_from_config(
    catalog_name: str,
    config_dict: Dict[str, Any],
    credentials: Optional[FileSystemCredentials] = None,
) -> IcebergCatalog:
    """Load Iceberg catalog from configuration dictionary

    Args:
        catalog_name: Name of the catalog
        config_dict: Dictionary with catalog configuration (type, uri, warehouse, etc.)

    Returns:
        IcebergCatalog instance

    Raises:
        CatalogNotFoundError: If config_dict is None or empty

    Example:
        config = {
            'type': 'rest',
            'uri': 'https://catalog.example.com',
            'warehouse': 'my_warehouse',
            'credential': 'token'
        }
        catalog = load_catalog_from_config('my_catalog', config)
    """
    from pyiceberg.catalog import load_catalog

    # Validate config
    PyicebergCatalogConfig(**config_dict)

    if not config_dict:
        raise CatalogNotFoundError("No configuration dictionary provided")

    logger.info(f"Loading catalog '{catalog_name}' from provided configuration")

    if credentials:
        config_dict.update(_get_fileio_config(credentials))

    return load_catalog(catalog_name, **config_dict)


@with_config(spec=BufferedDataWriter.BufferedDataWriterConfiguration)
def _get_writer_config(
    buffer_max_items: int,
    file_max_items: Optional[int],
) -> Tuple[int, Optional[int]]:
    """Resolve data_writer config — defaults owned by BufferedDataWriterConfiguration."""
    return buffer_max_items, file_max_items


@with_config(spec=IcebergConfig, sections="iceberg_catalog")
def get_iceberg_config_tuning(
    iceberg_upload_chunk_bytes: int = _UPLOAD_CHUNK_BYTES,
) -> Tuple[int, int]:
    """Return (parquet_batch_size, upload_chunk_bytes) resolved from dlt config / env vars.

    Arrow batch size mirrors BufferedDataWriter: min(buffer_max_items, file_max_items).
    Defaults are owned by dlt (BufferedDataWriterConfiguration), not hardcoded here.
    """
    buffer_max_items, file_max_items = _get_writer_config()
    # Mirror BufferedDataWriter logic: batch cannot exceed the file item limit
    parquet_batch_size = min(buffer_max_items, file_max_items or buffer_max_items)
    return parquet_batch_size, iceberg_upload_chunk_bytes


@with_config(spec=IcebergConfig, sections="iceberg_catalog")
def get_catalog(
    iceberg_catalog_name: str = "default",
    iceberg_catalog_type: Optional[str] = None,
    iceberg_catalog_config: Optional[Dict[str, Any]] = None,
    credentials: Optional[FileSystemCredentials] = None,
) -> IcebergCatalog:
    """Get an Iceberg catalog using multiple configuration methods.

    This function tries to load a catalog in the following priority order:
    1. From explicit config dictionary (if iceberg_catalog_config provided)
    2. From .pyiceberg.yaml file or from environment variables (PYICEBERG_*). Resolved by pyiceberg load_catalog mechanism. See https://py.iceberg.apache.org/configuration/#setting-configuration-values
    4. Fall back to in-memory SQLite catalog

    Args:
        iceberg_catalog_name: Name of the catalog (default: "default")
        iceberg_catalog_type: Type of catalog ('sql' or 'rest')
        iceberg_catalog_config: Optional dictionary with complete catalog configuration
        credentials: Optional filesystem credentials. This is ONLY used for backward compatibility with in-memory SQLite catalog.

    Returns:
        IcebergCatalog instance

    Examples:

        # Load from config dict
        config = {'type': 'rest', 'uri': 'https://...', 'warehouse': 'wh'}
        catalog = get_catalog('my_catalog', iceberg_catalog_type='rest', iceberg_catalog_config=config)

        # Load from .pyiceberg.yaml
        catalog = get_catalog('my_catalog', iceberg_catalog_type='sql')

        # Load from environment variables
        # (set PYICEBERG_CATALOG_TYPE, PYICEBERG_CATALOG_URI, etc.)
        catalog = get_catalog('my_catalog', iceberg_catalog_type='rest')

    """
    logger.info(f"Attempting to load Iceberg catalog: {iceberg_catalog_name}")

    # Validate catalog type
    supported_catalog_types = ["sql", "rest"]
    if iceberg_catalog_type not in supported_catalog_types:
        raise ValueError(f"Unsupported catalog type: {iceberg_catalog_type}. Use 'sql' or 'rest'.")

    # Priority 1: Explicit config dictionary (most specific and comes from secrets.toml)
    if iceberg_catalog_config:
        try:
            return _load_catalog_from_config(iceberg_catalog_name, iceberg_catalog_config)
        except CatalogNotFoundError as e:
            logger.warning(f"Failed to load catalog from config dict: {e}")

    # Priority 2: .pyiceberg.yaml file (PyIceberg standard)
    try:
        return _load_catalog_from_pyiceberg(iceberg_catalog_name)
    except CatalogNotFoundError as e:
        logger.debug(f"Catalog not found in .pyiceberg.yaml: {e}")

    # Priority 3: Fall back to in-memory SQLite (backward compatibility)
    logger.info(
        "No catalog configuration found, using in-memory SQLite catalog (backward compatibility)"
    )
    return get_sql_catalog(iceberg_catalog_name, "sqlite:///:memory:", credentials)


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
    schema: Union[pa.Schema, "pyiceberg.schema.Schema"],
    partition_columns: Optional[List[str]] = None,
    partition_spec: Optional[IcebergPartitionSpec] = UNPARTITIONED_PARTITION_SPEC,
    properties: Optional[Dict[str, str]] = None,
) -> None:
    if isinstance(schema, pa.Schema):
        schema = ensure_iceberg_compatible_arrow_schema(schema)

    if partition_columns:
        warnings.warn(
            "partition_columns is deprecated. Use partition_spec instead.", DeprecationWarning
        )
        with catalog.create_table_transaction(
            table_id,
            schema=schema,
            location=table_location,
            properties=properties or {},
        ) as txn:
            # add partitioning
            with txn.update_spec() as update_spec:
                for col in partition_columns:
                    update_spec.add_identity(col)
    else:
        catalog.create_table(
            identifier=table_id,
            schema=schema,
            location=table_location,
            partition_spec=partition_spec,
            properties=properties or {},
        )


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
    last_metadata_file = get_last_metadata_file(metadata_path, fs_client, config)
    return catalog.register_table(identifier, last_metadata_file)


def make_location(path: str, config: FilesystemConfiguration) -> str:
    # don't use file protocol for local files because duckdb does not support it
    # https://github.com/duckdb/duckdb/issues/13669
    location = config.make_url(path)
    if config.is_local_filesystem and os.name == "nt":
        # pyiceberg cannot deal with windows absolute urls
        location = location.replace("file:///", "file://")
    return location
