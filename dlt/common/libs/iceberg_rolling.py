"""Roll Iceberg data files using the encoded Parquet stream position."""
from collections import OrderedDict
from shutil import copyfileobj
from tempfile import TemporaryFile
from uuid import uuid4

import pyarrow as pa
import pyarrow.parquet as pq
from pyiceberg.exceptions import CommitStateUnknownException
from pyiceberg.expressions import AlwaysTrue
from pyiceberg.io import pyarrow as iceberg_arrow
from pyiceberg.manifest import DataFile, DataFileContent, FileFormat
from pyiceberg.schema import sanitize_column_names
from pyiceberg.table import DOWNCAST_NS_TIMESTAMP_TO_US_ON_WRITE, Table, TableProperties
from pyiceberg.table.locations import load_location_provider
from pyiceberg.typedef import Record
from pyiceberg.utils.config import Config

from dlt.common.libs.pyiceberg import _delete_files, ensure_iceberg_compatible_arrow_data
from dlt.common.schema.typing import TWriteDisposition


class _OpenFile:
    def __init__(self, key, schema, options):
        self.key = key
        self.stream = TemporaryFile()
        try:
            self.writer = pq.ParquetWriter(self.stream, schema, **options)
        except BaseException:
            self.stream.close()
            raise
        self.bytes_per_row = None

    def write_group(self, data, offset, target, row_limit):
        # Use the previous encoded row group to approach the target. Limit input
        # size independently so a very compressible column cannot exhaust memory.
        raw_bytes_per_row = max(1, data.nbytes / data.num_rows)
        budget = min(target, max(target // 16, target - self.stream.tell()))
        rows = max(1, min(
            int(budget / (self.bytes_per_row or raw_bytes_per_row)),
            int(64 * 1024 * 1024 / raw_bytes_per_row),
            row_limit,
            data.num_rows - offset,
        ))
        before = self.stream.tell()
        self.writer.write_table(data.slice(offset, rows), row_group_size=rows)
        self.bytes_per_row = max(1, self.stream.tell() - before) / rows
        return rows

    def close(self):
        try:
            self.writer.close()
        finally:
            self.stream.close()


def _input_tables(batches):
    """Group small input batches so each does not become a tiny Parquet row group."""
    pending = []
    size = 0
    for batch in batches:
        if not batch.num_rows:
            continue
        if pending and (batch.schema != pending[0].schema or size + batch.nbytes > 8 * 1024 * 1024):
            yield pa.Table.from_batches(pending)
            pending = []
            size = 0
        pending.append(batch)
        size += batch.nbytes
        if size >= 8 * 1024 * 1024 or len(pending) >= 256:
            yield pa.Table.from_batches(pending)
            pending = []
            size = 0
    if pending:
        yield pa.Table.from_batches(pending)


def write_sized_files(
    table: Table, batches, disposition: TWriteDisposition, upload_chunk: int, overwrite_filter=None
) -> None:
    props = table.properties
    target = int(props[TableProperties.WRITE_TARGET_FILE_SIZE_BYTES])
    row_limit = int(props.get(TableProperties.PARQUET_ROW_GROUP_LIMIT,
                              TableProperties.PARQUET_ROW_GROUP_LIMIT_DEFAULT))
    if min(target, row_limit, upload_chunk) <= 0:
        raise ValueError("Iceberg file size, row group row limit and upload chunk must be positive")
    options = iceberg_arrow._get_parquet_writer_kwargs(props)
    options.setdefault("store_decimal_as_integer", True)
    schema = table.schema()
    file_schema = sanitize_column_names(schema)
    spec = table.spec()
    locations = load_location_provider(table.location(), props)
    stats_plan = iceberg_arrow.compute_statistics_plan(file_schema, props)
    column_ids = iceberg_arrow.parquet_path_to_id_mapping(file_schema)
    downcast = Config().get_bool(DOWNCAST_NS_TIMESTAMP_TO_US_ON_WRITE) or False
    open_files = OrderedDict()
    uploaded = []
    files = []

    def finish(current):
        try:
            current.writer.close()
            size = current.stream.tell()  # includes the Parquet footer
            stats = iceberg_arrow.data_file_statistics_from_parquet_metadata(
                current.writer.writer.metadata, stats_plan, column_ids
            )
            path = locations.new_data_location(f"{uuid4()}.parquet", current.key)
            current.stream.seek(0)
            with table.io.new_output(path).create() as remote:
                uploaded.append(path)
                copyfileobj(current.stream, remote, upload_chunk)
            # PyIceberg 0.9 constructs DataFile directly.
            make_file = getattr(DataFile, "from_args", DataFile)
            files.append(make_file(
                content=DataFileContent.DATA, file_path=path, file_format=FileFormat.PARQUET,
                partition=current.key.partition if current.key else Record(),
                file_size_in_bytes=size, spec_id=spec.spec_id, sort_order_id=None,
                equality_ids=None, key_metadata=None, **stats.to_serialized_dict(),
            ))
        finally:
            current.stream.close()

    previous_schema = None
    try:
        for data in _input_tables(batches):
            data = ensure_iceberg_compatible_arrow_data(data)
            if data.schema != previous_schema:
                iceberg_arrow._check_pyarrow_schema_compatible(schema, data.schema, downcast)
                input_schema = iceberg_arrow.pyarrow_to_schema(
                    data.schema, name_mapping=schema.name_mapping, downcast_ns_timestamp_to_us=downcast
                )
                previous_schema = data.schema
            if spec.is_unpartitioned():
                partitions = [(None, data)]
            else:
                partitions = ((part.partition_key, part.arrow_table_partition)
                              for part in iceberg_arrow._determine_partitions(spec, schema, data))
            for key, subset in partitions:
                partition = tuple(v.value for v in key.field_values) if key else ()
                projected = pa.Table.from_batches([
                    iceberg_arrow._to_requested_schema(
                        file_schema, input_schema, part, downcast_ns_timestamp_to_us=downcast,
                        include_field_ids=True,
                    ) for part in subset.to_batches()
                ])
                offset = 0
                while offset < projected.num_rows:
                    if partition not in open_files:
                        if len(open_files) == 32:
                            finish(open_files.popitem(last=False)[1])
                        open_files[partition] = _OpenFile(key, projected.schema, options)
                    current = open_files[partition]
                    open_files.move_to_end(partition)
                    offset += current.write_group(projected, offset, target, row_limit)
                    if current.stream.tell() >= target:
                        finish(open_files.pop(partition))
        while open_files:
            finish(open_files.popitem()[1])
        with table.transaction() as txn:
            if overwrite_filter is not None:
                txn.delete(delete_filter=overwrite_filter)
            elif disposition == "replace" and table.current_snapshot():
                txn.delete(delete_filter=AlwaysTrue())
            if files:
                with txn._append_snapshot_producer({}) as append:
                    for data_file in files:
                        append.append_data_file(data_file)
    except CommitStateUnknownException:
        raise  # The catalog may already reference the uploaded files.
    except Exception:
        _delete_files(table.io, uploaded)
        raise
    finally:
        for current in open_files.values():
            current.close()
