"""Spark-based Iceberg operations: append, replace, and merge via PySpark.

Replaces PyIceberg's single-process paths with distributed Spark execution.

Requires ``pyspark``. The Iceberg Spark runtime + AWS bundle JARs are pulled
from Maven Central on first SparkSession creation via ``spark.jars.packages``
(cached in ``~/.ivy2``). Override via the env vars below if needed:

- ``DLT_ICEBERG_SPARK_PACKAGES`` — full comma-separated Maven coords list.
- ``DLT_ICEBERG_VERSION`` — Iceberg version (default ``1.10.1``).
- ``DLT_ICEBERG_SPARK_RUNTIME`` — Spark+Scala suffix (default ``4.0_2.13``).
"""

from __future__ import annotations

import contextlib
import glob
import os
import re
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple, TYPE_CHECKING

from dlt.common import logger

if TYPE_CHECKING:
    from pyspark.sql import SparkSession


_DEFAULT_ICEBERG_VERSION = "1.10.1"
_DEFAULT_SPARK_RUNTIME = "4.0_2.13"

#: Directories scanned (in order) by :func:`_find_cached_iceberg_jars` for
#: pre-staged Iceberg Spark JARs. Public — callers can prepend their own
#: paths to ship JARs from inside a package without going through ``~/.ivy2``.
#: Example::
#:
#:     from dlt.common.libs import spark_iceberg
#:     spark_iceberg.ICEBERG_JARS_DIRS.insert(0, Path("/app/nilus/iceberg_jars"))
ICEBERG_JARS_DIRS: List[Path] = [
    Path.home() / ".ivy2.5.2" / "jars",
    Path.home() / ".ivy2" / "jars",
]


def _iceberg_jar_specs() -> List[Tuple[str, str, str]]:
    """Return (group, artifact, version) tuples for required Iceberg JARs."""
    iceberg_version = os.environ.get("DLT_ICEBERG_VERSION", _DEFAULT_ICEBERG_VERSION)
    spark_runtime = os.environ.get("DLT_ICEBERG_SPARK_RUNTIME", _DEFAULT_SPARK_RUNTIME)
    return [
        ("org.apache.iceberg", f"iceberg-spark-runtime-{spark_runtime}", iceberg_version),
        ("org.apache.iceberg", "iceberg-aws-bundle", iceberg_version),
    ]


def _resolve_iceberg_packages() -> str:
    """Maven coordinates for Iceberg Spark runtime + AWS bundle."""
    explicit = os.environ.get("DLT_ICEBERG_SPARK_PACKAGES")
    if explicit:
        return explicit
    return ",".join(f"{g}:{a}:{v}" for g, a, v in _iceberg_jar_specs())


def _find_cached_iceberg_jars() -> Optional[List[str]]:
    """If all required Iceberg JARs already live on disk, return their paths.

    Scans :data:`ICEBERG_JARS_DIRS` in order. Using ``spark.jars`` with file
    paths instead of ``spark.jars.packages`` skips Spark's Ivy resolver
    entirely, removing the noisy bootstrap output. Returns ``None`` if any
    JAR is missing, so the caller falls back to Maven.
    """
    if os.environ.get("DLT_ICEBERG_SKIP_JAR_CACHE", "").lower() in {"1", "true", "yes"}:
        return None

    found: List[str] = []
    for group, artifact, version in _iceberg_jar_specs():
        candidate = None
        for jar_dir in ICEBERG_JARS_DIRS:
            patterns = [
                jar_dir / f"{group}_{artifact}-{version}.jar",
                jar_dir / f"{artifact}-{version}.jar",
            ]
            for p in patterns:
                if p.is_file():
                    candidate = str(p)
                    break
            if candidate:
                break
            glob_hits = glob.glob(str(jar_dir / f"*{artifact}*{version}*.jar"))
            if glob_hits:
                candidate = glob_hits[0]
                break
        if not candidate:
            return None
        found.append(candidate)
    return found


_QUIET_LOG4J2_PROPERTIES = """\
status = error
name = dlt-iceberg-quiet
appenders = console
appender.console.type = Console
appender.console.name = console
appender.console.target = SYSTEM_ERR
appender.console.layout.type = PatternLayout
appender.console.layout.pattern = %d{yy/MM/dd HH:mm:ss} %p %c{1}: %m%n%ex
rootLogger.level = error
rootLogger.appenderRefs = console
rootLogger.appenderRef.console.ref = console
"""


def _ensure_quiet_log4j_config() -> str:
    """Write a quiet log4j2 config to a stable temp path and return its file URL.

    Providing this file via ``-Dlog4j2.configurationFile=...`` stops Spark from
    falling back to its built-in defaults, which are the source of the noisy
    "Using Spark's default log4j profile / Setting default log level" stderr
    prints. Cached on disk so we do not rewrite on every call.
    """
    target = Path(tempfile.gettempdir()) / "dlt-spark-iceberg-log4j2.properties"
    if not target.is_file():
        target.write_text(_QUIET_LOG4J2_PROPERTIES)
    return target.as_uri()


# Lines that Spark / the JVM hardcode to System.err and cannot be silenced via
# log4j2 config. We drop them at the OS file descriptor level (see
# ``_filter_jvm_stderr``); everything else is written through unchanged.
_JVM_NOISE_PATTERNS: Tuple["re.Pattern[str]", ...] = (
    re.compile(r'^Setting default log level to "'),
    re.compile(r"^To adjust logging level use sc\.setLogLevel"),
    re.compile(r'^Setting Spark log level to "'),
    re.compile(r"^WARNING: Using incubator modules: jdk\.incubator\.vector"),
    re.compile(r"^WARN(?:ING)? NativeCodeLoader: "),
    re.compile(r"^Using Spark's default log4j profile"),
)


def _is_jvm_noise(line: str) -> bool:
    return any(p.search(line) for p in _JVM_NOISE_PATTERNS)


@contextlib.contextmanager
def _filter_jvm_stderr() -> Iterator[None]:
    """Drop a known-noisy subset of JVM/Spark stderr lines.

    Spark's ``SparkContext`` init and the JVM print messages directly via
    ``System.err`` that bypass log4j entirely (e.g. ``Setting default log
    level to "WARN"``). They cannot be suppressed through configuration, so we
    swap fd 2 with a pipe, run a pump thread that filters lines, and write
    survivors back to the real stderr. Set
    ``DLT_ICEBERG_NO_STDERR_FILTER=1`` to disable.
    """
    if os.environ.get("DLT_ICEBERG_NO_STDERR_FILTER", "").lower() in {"1", "true", "yes"}:
        yield
        return

    try:
        saved_stderr_fd = os.dup(2)
    except OSError:
        yield
        return

    r_fd, w_fd = os.pipe()
    os.dup2(w_fd, 2)
    os.close(w_fd)

    stop_event = threading.Event()

    def _pump() -> None:
        buf = b""
        try:
            with os.fdopen(r_fd, "rb", buffering=0) as r:
                while True:
                    try:
                        chunk = r.read(4096)
                    except OSError:
                        break
                    if not chunk:
                        break
                    buf += chunk
                    while b"\n" in buf:
                        line, buf = buf.split(b"\n", 1)
                        if not _is_jvm_noise(line.decode("utf-8", "replace")):
                            os.write(saved_stderr_fd, line + b"\n")
                if buf and not _is_jvm_noise(buf.decode("utf-8", "replace")):
                    os.write(saved_stderr_fd, buf)
        finally:
            stop_event.set()

    pump = threading.Thread(target=_pump, name="dlt-spark-stderr-filter", daemon=True)
    pump.start()
    try:
        yield
    finally:
        # Restore stderr; the pipe's last writer reference (fd 2) now drops,
        # giving the reader EOF so the pump thread can exit cleanly.
        os.dup2(saved_stderr_fd, 2)
        os.close(saved_stderr_fd)
        stop_event.wait(timeout=2)


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

    quiet_log4j_uri = _ensure_quiet_log4j_config()
    log4j_jvm_opt = f"-Dlog4j2.configurationFile={quiet_log4j_uri}"

    # In Spark local mode the driver JVM is spawned by ``spark-submit`` before
    # SparkConf is consulted, so ``spark.driver.extraJavaOptions`` is silently
    # ignored. We have to slip the log4j override into ``PYSPARK_SUBMIT_ARGS``
    # via ``--driver-java-options`` so it is picked up at JVM launch.
    existing_submit_args = os.environ.get("PYSPARK_SUBMIT_ARGS", "pyspark-shell")
    if "log4j2.configurationFile" not in existing_submit_args:
        os.environ["PYSPARK_SUBMIT_ARGS"] = (
            f'--driver-java-options="{log4j_jvm_opt}" {existing_submit_args}'
        )

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
        .config("spark.log.level", "ERROR")
        .config("spark.ui.showConsoleProgress", "false")
        .config("spark.driver.extraJavaOptions", log4j_jvm_opt)
        .config("spark.executor.extraJavaOptions", log4j_jvm_opt)
    )

    cached_jars = _find_cached_iceberg_jars()
    if cached_jars:
        builder = builder.config("spark.jars", ",".join(cached_jars))
    else:
        builder = builder.config("spark.jars.packages", _resolve_iceberg_packages())

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

    rest_apikey_header = (
        cfg.get("header.apikey")
        or os.environ.get("PYICEBERG_CATALOG__DEFAULT__HEADER__APIKEY")
        or os.environ.get("DATAOS_RUN_AS_APIKEY")
        or ""
    )

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
        f"spark.sql.catalog.{catalog_name}.header.apikey": rest_apikey_header,
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

    with _filter_jvm_stderr():
        spark = builder.getOrCreate()
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

    py_catalog = load_catalog("default", **(catalog_config or {}))
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
