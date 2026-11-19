from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from pyiceberg.catalog import load_catalog
from pyiceberg.partitioning import PartitionField, PartitionSpec
from pyiceberg.schema import Schema
from pyiceberg.transforms import IdentityTransform
from pyiceberg.types import LongType, NestedField, StringType

from dlt.common.libs.pyiceberg import merge_iceberg_table, write_iceberg_table


def test_compressed_target_across_batches(tmp_path: Path) -> None:
    target = 256 * 1024
    catalog = load_catalog(
        "local", type="sql", uri="sqlite:///:memory:", warehouse=tmp_path.as_uri()
    )
    catalog.create_namespace("ns")
    schema = Schema(NestedField(1, "id", LongType()), NestedField(2, "payload", StringType()),
                    NestedField(3, "region", LongType()))
    data = pa.table({"id": range(300_000), "payload": ["x" * 128] * 300_000,
                     "region": [0, 1] * 150_000})
    for name, spec in (
        ("plain", PartitionSpec()),
        ("partitioned", PartitionSpec(PartitionField(
            source_id=3, field_id=1000, transform=IdentityTransform(), name="region"
        ))),
    ):
        table = catalog.create_table(("ns", name), schema=schema, partition_spec=spec,
                                     properties={"write.target-file-size-bytes": str(target),
                                                 "write.parquet.compression-codec": "gzip"})
        write_iceberg_table(table, data.to_reader(max_chunksize=4096), "append")
        files = [task.file for task in table.scan().plan_files()]
        sizes = [f.file_size_in_bytes for f in files]
        partitions = 1 if name == "plain" else 2
        assert len(files) > partitions
        assert sum(size < target for size in sizes) <= partitions
        assert max(sizes) < target * 1.2
        for f in files:
            with table.io.new_input(f.file_path).open() as stream:
                assert stream.seek(0, 2) == f.file_size_in_bytes
                assert pq.read_metadata(stream).row_group(0).column(0).compression == "GZIP"
        assert sorted(table.scan(selected_fields=("id",)).to_arrow()["id"].to_pylist()) == list(range(300_000))
        if name == "partitioned":
            rows = table.scan(row_filter="region == 1").to_arrow()["id"].to_pylist()
            assert sorted(rows) == list(range(1, 300_000, 2))
        write_iceberg_table(table, data.slice(0, 13).to_reader(), "replace")
        assert table.scan().to_arrow().num_rows == 13
        merge_iceberg_table(
            table, pa.table({"id": [2, 20], "payload": ["update", "insert"], "region": [0, 1]}),
            {"x-merge-strategy": "upsert", "columns": {"id": {"name": "id", "data_type": "bigint", "primary_key": True}}},
            name,
        )
        result = table.scan().to_arrow().to_pylist()
        assert len(result) == 14
        assert {row["payload"] for row in result if row["id"] in (2, 20)} == {"update", "insert"}
