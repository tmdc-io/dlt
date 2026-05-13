"""Spark-based Iceberg operations: append, replace, and merge via PySpark.

Replaces PyIceberg's single-process paths with distributed Spark execution.

Requires ``pyspark``. Cloud-specific Iceberg bundle JARs are pulled from Maven
Central on first SparkSession creation via ``spark.jars.packages`` (cached in
``~/.ivy2``). The cloud backend (S3/Azure/GCS/local) is auto-detected from the
warehouse URL scheme. Override via the env vars below if needed:

- ``DLT_ICEBERG_SPARK_PACKAGES`` — full comma-separated Maven coords list
  (skips auto-detection entirely).
- ``DLT_ICEBERG_VERSION`` — Iceberg version (default ``1.10.1``).
- ``DLT_ICEBERG_SPARK_RUNTIME`` — Spark+Scala suffix (default ``4.0_2.13``).
"""

from __future__ import annotations

import contextlib
import glob
import os
import re
import shutil
import tempfile
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple, TYPE_CHECKING

from dlt.common import logger

if TYPE_CHECKING:
    from pyspark.sql import SparkSession


_DEFAULT_ICEBERG_VERSION = "1.10.1"
_DEFAULT_SPARK_RUNTIME = "4.0_2.13"

# Default version of the Hadoop Azure FileSystem jar used when
# ``ResolvingFileIO`` falls back to ``HadoopFileIO`` for ``abfss://`` paths.
# Spark 4.x bundles Hadoop 3.4.x. Override via ``DLT_HADOOP_AZURE_VERSION`` if
# you upgrade Spark/Hadoop.
_DEFAULT_HADOOP_AZURE_VERSION = "3.4.1"

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


_CLOUD_BUNDLE: Dict[str, Optional[str]] = {
    "s3": "iceberg-aws-bundle",
    "azure": "iceberg-azure-bundle",
    "gcs": "iceberg-gcp-bundle",
    "local": None,
}


def _detect_cloud(warehouse_uri: str) -> str:
    """Detect cloud backend from warehouse URI scheme.

    Returns one of: ``s3``, ``azure``, ``gcs``, ``local``. Defaults to ``s3``
    when scheme is unknown to preserve backward compatibility with existing
    S3-only setups.
    """
    uri = (warehouse_uri or "").lower().strip()
    if uri.startswith(("s3://", "s3a://", "s3n://")):
        return "s3"
    if uri.startswith(("abfss://", "abfs://", "wasbs://", "wasb://")):
        return "azure"
    if uri.startswith("gs://"):
        return "gcs"
    if uri.startswith(("file://", "/")) or not uri:
        return "local"
    return "s3"


def _hadoop_fs_specs(cloud: str) -> List[Tuple[str, str, str]]:
    """Return Hadoop FileSystem support jars needed for ``cloud``.

    Iceberg's ``ResolvingFileIO`` only auto-routes ``s3://``/``gs://`` to its
    native FileIOs. For ``abfss://`` it falls back to ``HadoopFileIO`` which
    needs Hadoop's Azure FileSystem implementation (``hadoop-azure``). The
    ``iceberg-azure-bundle`` does NOT include this — it only ships Iceberg's
    native ``ADLSFileIO``.

    For ABFSS-only workloads, ``hadoop-azure`` is the only extra jar
    required: its other compile-scope deps (``azure-storage``,
    ``jetty-util-ajax``, ``wildfly-openssl``) are needed only by the legacy
    ``wasb://`` code paths or for optional native-OpenSSL TLS acceleration,
    neither of which apply to modern Azure (ADLS Gen2) deployments.

    Returns an empty list for clouds where Spark's bundled Hadoop already
    has the FileSystem class (``s3a`` is provided by ``hadoop-aws`` which is
    typically bundled, and ``iceberg-aws-bundle`` covers Iceberg's S3FileIO).
    """
    if cloud != "azure":
        return []

    hadoop_azure = os.environ.get(
        "DLT_HADOOP_AZURE_VERSION", _DEFAULT_HADOOP_AZURE_VERSION
    )
    return [
        ("org.apache.hadoop", "hadoop-azure", hadoop_azure),
    ]


def _iceberg_jar_specs(cloud: str) -> List[Tuple[str, str, str]]:
    """Return (group, artifact, version) tuples for required Iceberg JARs.

    Always includes the Spark runtime; appends the cloud-specific Iceberg
    bundle when one exists for ``cloud``, and any extra Hadoop FileSystem
    support jars (e.g. ``hadoop-azure`` for ``abfss://``).
    """
    iceberg_version = os.environ.get("DLT_ICEBERG_VERSION", _DEFAULT_ICEBERG_VERSION)
    spark_runtime = os.environ.get("DLT_ICEBERG_SPARK_RUNTIME", _DEFAULT_SPARK_RUNTIME)
    specs: List[Tuple[str, str, str]] = [
        ("org.apache.iceberg", f"iceberg-spark-runtime-{spark_runtime}", iceberg_version),
    ]
    bundle = _CLOUD_BUNDLE.get(cloud)
    if bundle:
        specs.append(("org.apache.iceberg", bundle, iceberg_version))
    specs.extend(_hadoop_fs_specs(cloud))
    return specs


def _resolve_iceberg_packages(cloud: str) -> str:
    """Maven coordinates for Iceberg Spark runtime + cloud-specific bundle."""
    explicit = os.environ.get("DLT_ICEBERG_SPARK_PACKAGES")
    if explicit:
        return explicit
    return ",".join(f"{g}:{a}:{v}" for g, a, v in _iceberg_jar_specs(cloud))


def _stage_jars_into_spark_home(jars: List[str]) -> bool:
    """Symlink (or copy) ``jars`` into ``$SPARK_HOME/jars/``.

    This is the most robust way to put extra JARs on Spark's classpath.
    ``$SPARK_HOME/jars/`` is loaded by Spark's bootstrap launcher onto the
    JVM system classloader BEFORE PySpark, before any user code, before any
    SparkConf is read. Every classloader Spark creates afterwards (driver
    classloader, executor MutableURLClassLoader, REPL classloader,
    Hadoop's ``Configuration.classLoader``) chains to it as parent and
    therefore sees these classes.

    Why this matters: in Spark ``local[*]`` mode, ``spark.executor.extraClassPath``
    set programmatically via ``SparkConf`` is silently ignored — the local
    executor reuses the driver JVM but takes its classpath from the launcher,
    not from the conf. That is why ``HadoopFileIO`` fails with
    ``ClassNotFoundException: SecureAzureBlobFileSystem`` on the executor
    side even when the driver-side ``FileSystem.get()`` works.

    Returns ``True`` if every jar is now present in ``$SPARK_HOME/jars/``,
    ``False`` if the directory is unknown / not writable / a copy fails. The
    caller should fall back to ``spark.jars`` + ``spark.driver.extraClassPath``
    + ``--jars`` on ``PYSPARK_SUBMIT_ARGS`` in that case.
    """
    spark_home_str = os.environ.get("SPARK_HOME")
    if not spark_home_str:
        try:
            from pyspark import find_spark_home as _fsh

            spark_home_str = _fsh._find_spark_home()  # type: ignore[attr-defined]
        except Exception:
            return False

    spark_jars_dir = Path(spark_home_str) / "jars"
    if not spark_jars_dir.is_dir() or not os.access(spark_jars_dir, os.W_OK):
        return False

    for jar_path in jars:
        src = Path(jar_path)
        if not src.is_file():
            return False
        target = spark_jars_dir / src.name
        if target.exists() or target.is_symlink():
            try:
                if target.resolve() == src.resolve():
                    continue
            except OSError:
                pass
            continue
        try:
            target.symlink_to(src.resolve())
        except OSError:
            try:
                shutil.copy2(src, target)
            except OSError:
                return False
    return True


def _find_cached_iceberg_jars(cloud: str) -> Optional[List[str]]:
    """If all required Iceberg JARs already live on disk, return their paths.

    Scans :data:`ICEBERG_JARS_DIRS` in order. Using ``spark.jars`` with file
    paths instead of ``spark.jars.packages`` skips Spark's Ivy resolver
    entirely, removing the noisy bootstrap output. Returns ``None`` if any
    JAR is missing, so the caller falls back to Maven.
    """
    if os.environ.get("DLT_ICEBERG_SKIP_JAR_CACHE", "").lower() in {"1", "true", "yes"}:
        return None

    found: List[str] = []
    for group, artifact, version in _iceberg_jar_specs(cloud):
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

    cfg = catalog_config or {}

    cat_uri = cfg.get("uri") or os.environ.get("PYICEBERG_CATALOG__DEFAULT__URI", "")
    cat_warehouse = cfg.get("warehouse") or os.environ.get(
        "PYICEBERG_CATALOG__DEFAULT__WAREHOUSE", ""
    )
    rest_apikey_header = (
        cfg.get("header.apikey")
        or os.environ.get("PYICEBERG_CATALOG__DEFAULT__HEADER__APIKEY")
        or os.environ.get("DATAOS_RUN_AS_APIKEY")
        or ""
    )

    cloud = _detect_cloud(cat_warehouse)
    logger.debug(f"[spark-iceberg] Detected cloud backend: {cloud} (warehouse={cat_warehouse!r})")

    cached_jars = _find_cached_iceberg_jars(cloud)
    if cached_jars:
        # Try the bulletproof path first: stage jars into ``$SPARK_HOME/jars/``.
        # That directory is loaded onto the JVM system classloader by Spark's
        # launcher BEFORE PySpark / SparkConf / executor setup runs, so every
        # classloader (driver, executor in local mode, Hadoop's
        # ``Configuration.classLoader``) sees the classes unconditionally.
        # This is the only mechanism that survives Spark ``local[*]`` mode's
        # well-known habit of silently dropping ``spark.executor.extraClassPath``
        # set via ``SparkConf`` — see :func:`_stage_jars_into_spark_home` for
        # the gory details.
        staged = _stage_jars_into_spark_home(cached_jars)
        if staged:
            logger.info(
                f"[spark-iceberg] Staged {len(cached_jars)} jar(s) into "
                f"$SPARK_HOME/jars (bootstrap classpath)"
            )
        else:
            # Fallback: register the jars via every Spark/JVM mechanism we can.
            # Each of these covers a different classloader path; we set all
            # three because no single one works in every Spark mode.
            #   1. ``spark.jars`` — Spark internals + executor (cluster mode).
            #   2. ``spark.driver.extraClassPath`` — explicit driver classpath.
            #   3. ``--jars`` on ``PYSPARK_SUBMIT_ARGS`` — consumed by
            #      ``spark-submit`` BEFORE the JVM starts, so jars hit the
            #      JVM classpath at launch.
            jars_csv = ",".join(cached_jars)
            cp_sep = os.pathsep  # ":" on Linux/macOS, ";" on Windows
            cp_string = cp_sep.join(cached_jars)
            builder = (
                builder
                .config("spark.jars", jars_csv)
                .config("spark.driver.extraClassPath", cp_string)
                .config("spark.executor.extraClassPath", cp_string)
            )
            submit_args_now = os.environ.get("PYSPARK_SUBMIT_ARGS", "pyspark-shell")
            if "--jars" not in submit_args_now:
                os.environ["PYSPARK_SUBMIT_ARGS"] = f"--jars {jars_csv} {submit_args_now}"
            logger.warning(
                f"[spark-iceberg] $SPARK_HOME/jars staging unavailable — "
                f"falling back to spark.jars + extraClassPath + --jars "
                f"({len(cached_jars)} jar(s)). In Spark local mode this may "
                f"not reach the executor classloader."
            )
    else:
        builder = builder.config("spark.jars.packages", _resolve_iceberg_packages(cloud))

    spark_confs: Dict[str, str] = {
        f"spark.sql.catalog.{catalog_name}": "org.apache.iceberg.spark.SparkCatalog",
        f"spark.sql.catalog.{catalog_name}.catalog-impl": "org.apache.iceberg.rest.RESTCatalog",
        f"spark.sql.catalog.{catalog_name}.uri": cat_uri,
        f"spark.sql.catalog.{catalog_name}.warehouse": cat_warehouse,
        f"spark.sql.catalog.{catalog_name}.header.apikey": rest_apikey_header,
    }
    spark_confs.update(_cloud_spark_confs(cloud, catalog_name, cfg))

    for k, v in spark_confs.items():
        if v:
            builder = builder.config(k, v)

    # Diagnostic to stdout: show the auth-relevant Spark conf we end up
    # with. Critical for debugging prod ``CredentialUnavailable`` errors:
    # if the auth keys are missing here, the depot did not give us the
    # creds. If they are present here but Iceberg still can't auth, the
    # REST catalog server is overriding them.
    auth_keys_emitted = sorted(
        k for k, val in spark_confs.items()
        if val and ("adls.auth" in k or "adls.sas-token" in k
                    or "adls.connection-string" in k or "fs.azure" in k
                    or "s3.access-key-id" in k or "fs.s3a.access" in k
                    or "gcs.project-id" in k)
    )
    print(
        f"[dlt][spark-iceberg] cloud={cloud} catalog={catalog_name} "
        f"warehouse={cat_warehouse!r} "
        f"auth_keys_in_spark_conf={auth_keys_emitted}",
        flush=True,
    )

    with _filter_jvm_stderr():
        spark = builder.getOrCreate()
    return spark


def _cloud_spark_confs(
    cloud: str, catalog_name: str, cfg: Dict[str, Any]
) -> Dict[str, str]:
    """Cloud-specific Spark/Hadoop properties for the Iceberg catalog.

    Returns FileIO impl + Hadoop FS credentials for the detected backend.
    Reads from ``cfg`` first (PyIceberg catalog config), then falls back to
    standard cloud env vars (``AWS_*``, ``AZURE_*``, ``GOOGLE_*``).
    """
    cat_prefix = f"spark.sql.catalog.{catalog_name}"

    if cloud == "s3":
        endpoint = cfg.get("s3.endpoint") or os.environ.get(
            "PYICEBERG_CATALOG__DEFAULT__S3__ENDPOINT", ""
        )
        access_key = cfg.get("s3.access-key-id") or os.environ.get("AWS_ACCESS_KEY_ID", "")
        secret_key = cfg.get("s3.secret-access-key") or os.environ.get(
            "AWS_SECRET_ACCESS_KEY", ""
        )
        region = cfg.get("s3.region") or os.environ.get("AWS_REGION", "us-east-1")
        path_style = str(cfg.get("s3.path-style-access", "true"))
        return {
            f"{cat_prefix}.io-impl": "org.apache.iceberg.aws.s3.S3FileIO",
            f"{cat_prefix}.s3.endpoint": endpoint,
            f"{cat_prefix}.s3.access-key-id": access_key,
            f"{cat_prefix}.s3.secret-access-key": secret_key,
            f"{cat_prefix}.s3.path-style-access": path_style,
            "spark.hadoop.fs.s3a.endpoint": endpoint,
            "spark.hadoop.fs.s3a.access.key": access_key,
            "spark.hadoop.fs.s3a.secret.key": secret_key,
            "spark.hadoop.fs.s3a.path.style.access": path_style,
            "spark.hadoop.fs.s3a.region": region,
            "spark.hadoop.fs.s3a.impl": "org.apache.hadoop.fs.s3a.S3AFileSystem",
        }

    if cloud == "azure":
        # ADLS Gen2 (abfss://) — preferred. account.key OR sas.token OR
        # connection-string can be supplied via cfg or env.
        account = cfg.get("adls.account-name") or os.environ.get(
            "AZURE_STORAGE_ACCOUNT_NAME", ""
        )
        account_key = cfg.get("adls.account-key") or os.environ.get(
            "AZURE_STORAGE_ACCOUNT_KEY", ""
        )
        sas_token = cfg.get("adls.sas-token") or os.environ.get("AZURE_STORAGE_SAS_TOKEN", "")
        conn_string = cfg.get("adls.connection-string") or os.environ.get(
            "AZURE_STORAGE_CONNECTION_STRING", ""
        )
        # Diagnostic to stdout (DataOS captures stdout, not python logger).
        # Helps distinguish "depot didn't pass creds" from "creds reached
        # us but Iceberg ignored them".
        print(
            f"[dlt][spark-iceberg][azure-creds] cfg_keys={sorted(cfg.keys())} "
            f"account={'<set>' if account else '<EMPTY>'} "
            f"account_key={'<set>' if account_key else '<EMPTY>'} "
            f"sas_token={'<set>' if sas_token else '<EMPTY>'} "
            f"conn_string={'<set>' if conn_string else '<EMPTY>'}",
            flush=True,
        )

        confs: Dict[str, str] = {
            f"{cat_prefix}.io-impl": "org.apache.iceberg.azure.adlsv2.ADLSFileIO",
            # Hadoop FS class registrations for ``abfss://`` / ``abfs://``.
            # Without these Spark throws ``ClassNotFoundException`` when
            # ``ResolvingFileIO`` falls back to ``HadoopFileIO`` for ADLS Gen2
            # paths. The auto-discovery via ``core-default.xml`` inside
            # ``hadoop-azure.jar`` is shadowed by Spark's bundled
            # ``hadoop-client-api`` jar, so we register them explicitly.
            "spark.hadoop.fs.abfss.impl": (
                "org.apache.hadoop.fs.azurebfs.SecureAzureBlobFileSystem"
            ),
            "spark.hadoop.fs.abfs.impl": (
                "org.apache.hadoop.fs.azurebfs.AzureBlobFileSystem"
            ),
            "spark.hadoop.fs.AbstractFileSystem.abfss.impl": (
                "org.apache.hadoop.fs.azurebfs.Abfss"
            ),
            "spark.hadoop.fs.AbstractFileSystem.abfs.impl": (
                "org.apache.hadoop.fs.azurebfs.Abfs"
            ),
        }
        # Iceberg's ADLSFileIO uses these EXACT property names (see
        # ``org.apache.iceberg.azure.AzureProperties``). Earlier we used
        # convenience names like ``adls.account-name`` which ADLSFileIO
        # silently ignored, causing it to fall through to
        # ``DefaultAzureCredential`` (env vars / Azure CLI / managed
        # identity) — all of which fail in a stock K8s pod with
        # ``CredentialUnavailableException``. The SAS/connection-string
        # variants are PER-ACCOUNT and require the
        # ``<account>.dfs.core.windows.net`` suffix to be honored.
        if account:
            confs[f"{cat_prefix}.adls.auth.shared-key.account.name"] = account
        if account_key:
            if account:
                confs[f"{cat_prefix}.adls.auth.shared-key.account.key"] = account_key
                # Hadoop fallback (only used if Iceberg routes via HadoopFileIO).
                confs[f"spark.hadoop.fs.azure.account.key.{account}.dfs.core.windows.net"] = (
                    account_key
                )
        if sas_token and account:
            confs[f"{cat_prefix}.adls.sas-token.{account}.dfs.core.windows.net"] = sas_token
            confs[f"spark.hadoop.fs.azure.sas.fixed.token.{account}.dfs.core.windows.net"] = (
                sas_token
            )
        if conn_string and account:
            confs[f"{cat_prefix}.adls.connection-string.{account}.dfs.core.windows.net"] = (
                conn_string
            )
        return confs

    if cloud == "gcs":
        project_id = cfg.get("gcs.project-id") or os.environ.get(
            "GOOGLE_CLOUD_PROJECT", ""
        )
        creds_path = cfg.get("gcs.credentials-path") or os.environ.get(
            "GOOGLE_APPLICATION_CREDENTIALS", ""
        )
        confs = {
            f"{cat_prefix}.io-impl": "org.apache.iceberg.gcp.gcs.GCSFileIO",
        }
        if project_id:
            confs[f"{cat_prefix}.gcs.project-id"] = project_id
            confs["spark.hadoop.fs.gs.project.id"] = project_id
        if creds_path:
            confs[f"{cat_prefix}.gcs.service-account-key-file"] = creds_path
            confs["spark.hadoop.google.cloud.auth.service.account.json.keyfile"] = (
                creds_path
            )
        confs["spark.hadoop.fs.gs.impl"] = (
            "com.google.cloud.hadoop.fs.gcs.GoogleHadoopFileSystem"
        )
        return confs

    # Local filesystem — Iceberg picks the right FileIO automatically.
    return {f"{cat_prefix}.io-impl": "org.apache.iceberg.io.ResolvingFileIO"}


@contextlib.contextmanager
def _wap_session(
    spark: "SparkSession",
    full_table: str,
    table_id: str,
    catalog_name: str,
    py_table: Any,
    op: str,
) -> Iterator[str]:
    """Run a block of writes against an isolated Iceberg WAP staging branch.

    Yields the branch name. All writes inside the ``with`` block are routed
    to the staging branch via ``spark.wap.branch``. On clean exit, every
    snapshot created on the branch (since the base) is published onto
    ``main`` via ``system.cherrypick_snapshot`` — this is the canonical
    WAP publish that does NOT require an ancestor relationship. On
    exception the branch is dropped and ``main`` is left untouched —
    readers never observe partial state.

    First-time tables (no prior snapshot) get a no-op seed snapshot so a
    branch can be created. On failure such tables are visible as empty,
    never partially populated.
    """
    logger.info(f"[wap] enter op={op!r} table={full_table!r}")
    # Iceberg silently ignores spark.wap.branch unless this table property is
    # set. Without it Spark writes to main and the staging branch stays empty,
    # silently breaking atomicity. Set idempotently on every run.
    spark.sql(
        f"ALTER TABLE {full_table} SET TBLPROPERTIES ('write.wap.enabled'='true')"
    )

    if py_table.current_snapshot() is None:
        logger.info(f"[{op}] First run on empty table — committing seed snapshot")
        spark.sql(f"INSERT INTO {full_table} SELECT * FROM {full_table} WHERE 1 = 0")
        py_table.refresh()

    base_snapshot_id = py_table.current_snapshot().snapshot_id
    # Branch name must be unique across concurrent loads. dlt's loader uses a
    # thread pool, so multiple jobs in the same process can hit this within the
    # same wall-clock second — pid+time alone collides. Add thread id + a 6-hex
    # uuid suffix to make it collision-proof.
    wap_branch = (
        f"dlt_wap_{int(time.time())}"
        f"_{os.getpid()}_{threading.get_ident()}_{uuid.uuid4().hex[:6]}"
    )

    logger.info(
        f"[{op}] Creating staging branch {wap_branch!r} from snapshot "
        f"{base_snapshot_id} (main left untouched until success)"
    )
    spark.sql(f"ALTER TABLE {full_table} CREATE BRANCH {wap_branch}")
    spark.conf.set("spark.wap.branch", wap_branch)

    try:
        yield wap_branch
        spark.conf.unset("spark.wap.branch")

        # With write.wap.enabled=true, every staged commit on the branch chains
        # onto the base snapshot — so the branch IS a descendant of main and
        # fast_forward is valid. This is a SINGLE atomic ref-pointer move at
        # the catalog level: main jumps from base → branch.head in one commit.
        # No partial state is possible: either the publish succeeds and ALL
        # batches are visible, or it fails and main stays at base. Compare
        # cherrypick_snapshot which is one commit per staged snapshot (N calls,
        # mid-process death = partial main).
        py_table.refresh()
        branch_snap = py_table.snapshot_by_name(wap_branch)
        if branch_snap is None or branch_snap.snapshot_id == base_snapshot_id:
            logger.info(f"[{op}] No new snapshots on staging branch — nothing to publish")
        else:
            logger.info(
                f"[{op}] Publishing branch {wap_branch!r} onto main via fast_forward "
                f"(branch head = {branch_snap.snapshot_id}, base = {base_snapshot_id}) — "
                f"single atomic commit"
            )
            spark.sql(
                f"CALL {catalog_name}.system.fast_forward("
                f"table => '{table_id}', branch => 'main', to => '{wap_branch}')"
            )
    except Exception:
        logger.error(
            f"[{op}] Operation failed — discarding staging branch {wap_branch!r}. "
            f"Main remains at snapshot {base_snapshot_id}."
        )
        raise
    finally:
        try:
            spark.conf.unset("spark.wap.branch")
        except Exception:
            pass
        try:
            spark.sql(f"ALTER TABLE {full_table} DROP BRANCH IF EXISTS {wap_branch}")
        except Exception as cleanup_err:
            logger.warning(
                f"[{op}] Failed to drop staging branch {wap_branch!r}: "
                f"{cleanup_err}. Branch can be removed manually."
            )


def merge_iceberg_table_spark(
    file_paths: List[str],
    table_id: str,
    join_cols: List[str],
    catalog_name: str = "rest",
    catalog_config: Optional[Dict[str, Any]] = None,
    batch_size: int = 2,
) -> None:
    """Run ``MERGE INTO`` on an Iceberg table via Spark, atomically.

    All per-batch MERGE statements run inside :func:`_wap_session`, so the
    operation is fully all-or-nothing — readers either see all merged rows
    or none.
    """
    from pyiceberg.catalog import load_catalog

    spark = _build_spark_session(catalog_name, catalog_config)
    full_table = f"{catalog_name}.{table_id}"

    py_catalog = load_catalog("default", **(catalog_config or {}))
    py_table = py_catalog.load_table(table_id)

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
    total_batches = (n_files + batch_size - 1) // batch_size

    with _wap_session(spark, full_table, table_id, catalog_name, py_table, "spark-merge") as wap_branch:
        for i in range(0, n_files, batch_size):
            batch = file_paths[i : i + batch_size]
            batch_num = i // batch_size + 1

            try:
                updates = spark.read.parquet(*batch)
                batch_rows = updates.count()
                total_rows += batch_rows
                updates.createOrReplaceTempView("__dlt_spark_updates")

                logger.info(
                    f"[spark-merge] Batch {batch_num}/{total_batches}: "
                    f"MERGE {batch_rows:,} rows from {len(batch)} file(s) → "
                    f"branch {wap_branch!r}"
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
    """Append or replace data in an Iceberg table via Spark, atomically.

    Both ``append`` and ``replace`` are routed through :func:`_wap_session`,
    so partial-write states are never visible to readers. ``replace`` first
    deletes existing data on the staging branch, then appends new batches;
    on success ``main`` is fast-forwarded in a single atomic commit.

    Files are processed in batches of ``batch_size`` to bound peak memory.
    """
    from pyiceberg.catalog import load_catalog

    spark = _build_spark_session(catalog_name, catalog_config)
    full_table = f"{catalog_name}.{table_id}"

    py_catalog = load_catalog("default", **(catalog_config or {}))
    py_table = py_catalog.load_table(table_id)

    op = "spark-write"
    t0 = time.time()
    total_rows = 0
    n_files = len(file_paths)
    total_batches = (n_files + batch_size - 1) // batch_size

    with _wap_session(spark, full_table, table_id, catalog_name, py_table, op) as wap_branch:
        if write_disposition == "replace":
            logger.info(
                f"[{op}] REPLACE on branch {wap_branch!r}: deleting existing rows"
            )
            spark.sql(f"DELETE FROM {full_table} WHERE TRUE")

        for i in range(0, n_files, batch_size):
            batch = file_paths[i : i + batch_size]
            batch_num = i // batch_size + 1

            df = spark.read.parquet(*batch)
            batch_rows = df.count()
            total_rows += batch_rows

            logger.info(
                f"[{op}] Batch {batch_num}/{total_batches}: APPEND "
                f"{batch_rows:,} rows from {len(batch)} file(s) → "
                f"branch {wap_branch!r}"
            )
            df.writeTo(full_table).append()
            del df
            spark.catalog.clearCache()
            logger.info(f"[{op}] Batch {batch_num} done")

    elapsed = time.time() - t0
    logger.info(
        f"[{op}] {write_disposition.upper()} completed in {elapsed:.1f}s "
        f"({total_rows:,} rows, {n_files} files)"
    )
