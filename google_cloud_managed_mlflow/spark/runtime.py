# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Spark session, topology, and scratch GCS helpers.

The ``gcs_*`` helpers intentionally use raw ADC ``google.cloud.storage.Client``
rather than ``GcpGCSArtifactRepository`` because user-supplied ``dfs_tmpdir`` /
``MLFLOW_DFS_TMP`` scratch buckets sit outside the Managed MLflow namespace or
registered-model storage root, so CAB downscoped tokens minted by
``GcpGCSArtifactRepository`` would fail storage-root containment checks.
"""

from __future__ import annotations

import contextlib
import enum
import logging
import os
import urllib.parse
import uuid
from collections.abc import Iterator
from typing import Any

_logger = logging.getLogger(__name__)


class SparkTopology(enum.Enum):
    """Execution and default-filesystem topology of a SparkSession."""

    SPARK_CONNECT = "spark_connect"
    LOCAL = "local"
    DISTRIBUTED_SHARED_DFS = "distributed_shared_dfs"
    DISTRIBUTED_NO_SHARED_DFS = "distributed_no_shared_dfs"


def is_spark_connect(spark: Any) -> bool:
    """Return True if ``spark`` is a Spark Connect session."""
    if spark is None:
        return False
    try:
        from pyspark.sql.connect.session import SparkSession as ConnectSession

        if isinstance(spark, ConnectSession):
            return True
    except ImportError:
        pass
    module_name = getattr(type(spark), "__module__", "")
    return isinstance(module_name, str) and module_name.startswith(
        "pyspark.sql.connect"
    )


def active_session() -> Any:
    """Return the active SparkSession (classic or Connect), or None."""
    try:
        from pyspark.sql import SparkSession

        return SparkSession.getActiveSession()
    except Exception:
        return None


def require_active_session() -> Any:
    """Return the active SparkSession or raise an ``MlflowException``."""
    session = active_session()
    if session is None:
        from mlflow.exceptions import MlflowException
        from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE

        raise MlflowException(
            "An active SparkSession is required for Spark model operations.",
            error_code=INVALID_PARAMETER_VALUE,
        )
    return session


def is_local_master(spark: Any) -> bool:
    """Return True if ``spark`` is a classic session with a ``local`` master."""
    if spark is None or is_spark_connect(spark):
        return False
    master = ""
    try:
        conf = getattr(spark, "conf", None)
        if conf is not None:
            master = str(conf.get("spark.master", "") or "")
    except Exception:
        master = ""
    if not master:
        try:
            sc = getattr(spark, "sparkContext", None)
            if sc is not None:
                master = str(getattr(sc, "master", "") or "")
        except Exception:
            master = ""
    return master.strip().lower().startswith("local")


def default_fs_scheme(spark: Any) -> str:
    """Return the lowercase Hadoop defaultFS URI scheme, or ``""`` on Connect."""
    if spark is None or is_spark_connect(spark):
        return ""
    raw_fs: str | None = None
    try:
        conf = getattr(spark, "conf", None)
        if conf is not None:
            val = conf.get("spark.hadoop.fs.defaultFS", None)
            if val:
                raw_fs = str(val)
    except Exception:
        raw_fs = None
    if not raw_fs:
        try:
            sc = getattr(spark, "sparkContext", None)
            jsc = getattr(sc, "_jsc", None) if sc is not None else None
            if jsc is not None:
                val = jsc.hadoopConfiguration().get("fs.defaultFS", "file:///")
                if val:
                    raw_fs = str(val)
        except Exception:
            raw_fs = None
    if not raw_fs:
        return "file"
    scheme = urllib.parse.urlsplit(raw_fs.strip()).scheme.lower()
    return scheme or "file"


def classify_topology(spark: Any) -> SparkTopology:
    """Classify ``spark`` by Connect vs. classic master and defaultFS."""
    if is_spark_connect(spark):
        return SparkTopology.SPARK_CONNECT
    if is_local_master(spark):
        return SparkTopology.LOCAL
    if default_fs_scheme(spark) not in ("file", ""):
        return SparkTopology.DISTRIBUTED_SHARED_DFS
    return SparkTopology.DISTRIBUTED_NO_SHARED_DFS


def is_distributed_without_shared_dfs(spark: Any) -> bool:
    """Return True for Spark Connect or non-local clusters lacking shared DFS."""
    return classify_topology(spark) in (
        SparkTopology.SPARK_CONNECT,
        SparkTopology.DISTRIBUTED_NO_SHARED_DFS,
    )


def validate_save_dfs_tmpdir(spark: Any, dfs_tmpdir: str | None) -> None:
    """Validate ``dfs_tmpdir`` for ``mlflow.spark.save_model`` against topology."""
    if spark is None:
        return
    from mlflow.exceptions import MlflowException
    from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE

    from google_cloud_managed_mlflow.cab_credentials import has_gs_scheme

    eff_tmp = str(dfs_tmpdir or "").strip()
    is_gs = has_gs_scheme(eff_tmp)
    topology = classify_topology(spark)

    if topology is SparkTopology.SPARK_CONNECT:
        if not is_gs:
            raise MlflowException(
                "Spark Connect save_model requires a 'gs://' scratch location "
                "via dfs_tmpdir='gs://<scratch-bucket>/mlflow-tmp' or "
                "MLFLOW_DFS_TMP.",
                error_code=INVALID_PARAMETER_VALUE,
            )
        return

    scheme = urllib.parse.urlsplit(eff_tmp).scheme.lower()
    if topology is not SparkTopology.LOCAL and scheme == "file":
        raise MlflowException(
            "Refusing explicit 'file:' dfs_tmpdir for save_model on a "
            "non-local Spark session to prevent silent executor weight data "
            "loss. Pass dfs_tmpdir='gs://<scratch-bucket>/mlflow-tmp' or "
            "a shared DFS URI.",
            error_code=INVALID_PARAMETER_VALUE,
        )

    if topology is SparkTopology.DISTRIBUTED_NO_SHARED_DFS:
        is_remote_dfs = is_gs or (bool(scheme) and scheme != "file")
        if not is_remote_dfs:
            raise MlflowException(
                "Refusing local file: dfs_tmpdir staging for save_model on a "
                "distributed Spark session without shared DFS to prevent silent "
                "executor weight data loss. Pass "
                "dfs_tmpdir='gs://<scratch-bucket>/mlflow-tmp' (or set "
                "MLFLOW_DFS_TMP).",
                error_code=INVALID_PARAMETER_VALUE,
            )


def _parse_gs_prefix(gs_prefix: str) -> tuple[str, str]:
    """Validate a non-root ``gs://<bucket>/<prefix>`` scratch URI."""
    from google_cloud_managed_mlflow.cab_credentials import (
        validate_gcs_artifact_uri,
    )

    bucket_name, prefix = validate_gcs_artifact_uri(gs_prefix)
    if not prefix:
        from mlflow.exceptions import MlflowException
        from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE

        raise MlflowException(
            f"Scratch GCS URI must include a non-empty object prefix, "
            f"got {gs_prefix!r}.",
            error_code=INVALID_PARAMETER_VALUE,
        )
    return bucket_name, prefix


def gcs_upload_directory(local_dir: str, gs_prefix: str) -> list[tuple[str, str]]:
    """Upload every file under ``local_dir`` to ``gs_prefix`` using raw ADC.

    Uses raw ADC ``google.cloud.storage.Client`` because scratch ``dfs_tmpdir``
    buckets sit outside Managed MLflow CAB storage roots.
    """
    from google.cloud import storage

    bucket_name, prefix = _parse_gs_prefix(gs_prefix)
    uploaded: list[tuple[str, str]] = []
    with contextlib.closing(storage.Client()) as client:
        bucket = client.bucket(bucket_name)
        for root, _, files in os.walk(local_dir):
            for filename in sorted(files):
                local_path = os.path.join(root, filename)
                rel_path = os.path.relpath(local_path, local_dir).replace(os.sep, "/")
                blob_name = f"{prefix}/{rel_path}"
                bucket.blob(blob_name).upload_from_filename(local_path)
                uploaded.append((bucket_name, blob_name))
    return uploaded


def gcs_download_prefix(gs_prefix: str, local_dir: str) -> int:
    """Download all non-directory blobs under ``gs_prefix/`` using raw ADC.

    Uses raw ADC ``google.cloud.storage.Client`` because scratch ``dfs_tmpdir``
    buckets sit outside Managed MLflow CAB storage roots.
    """
    from google.cloud import storage

    bucket_name, prefix = _parse_gs_prefix(gs_prefix)
    list_prefix = f"{prefix}/"
    os.makedirs(local_dir, exist_ok=True)
    count = 0
    with contextlib.closing(storage.Client()) as client:
        for blob in client.bucket(bucket_name).list_blobs(prefix=list_prefix):
            blob_name = str(blob.name)
            if (
                not blob_name
                or blob_name.endswith("/")
                or not blob_name.startswith(list_prefix)
            ):
                continue
            rel_path = blob_name[len(list_prefix) :].lstrip("/")
            if not rel_path:
                continue
            parts = rel_path.split("/")
            if any(part in ("", ".", "..") for part in parts):
                from mlflow.exceptions import MlflowException
                from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE

                raise MlflowException(
                    f"Unsafe GCS object key relative path {rel_path!r} "
                    f"under {gs_prefix!r}.",
                    error_code=INVALID_PARAMETER_VALUE,
                )
            dst_path = os.path.join(local_dir, *parts)
            os.makedirs(os.path.dirname(dst_path), exist_ok=True)
            blob.download_to_filename(dst_path)
            count += 1
    return count


def gcs_delete_prefix(gs_prefix: str) -> None:
    """Best-effort delete all blobs under ``gs_prefix/`` using raw ADC.

    Uses raw ADC ``google.cloud.storage.Client`` because scratch ``dfs_tmpdir``
    buckets sit outside Managed MLflow CAB storage roots.

    Raises:
        MlflowException: If ``gs_prefix`` is invalid or has an empty object prefix.
    """
    bucket_name, prefix = _parse_gs_prefix(gs_prefix)
    list_prefix = f"{prefix}/"
    try:
        from google.cloud import storage

        with contextlib.closing(storage.Client()) as client:
            bucket = client.bucket(bucket_name)
            for blob in bucket.list_blobs(prefix=list_prefix):
                try:
                    blob.delete()
                except Exception:
                    _logger.debug(
                        "Best-effort delete failed for gs://%s/%s",
                        bucket_name,
                        blob.name,
                        exc_info=True,
                    )
    except Exception:
        _logger.debug(
            "Best-effort cleanup failed for GCS prefix %s", gs_prefix, exc_info=True
        )


@contextlib.contextmanager
def gcs_scratch_dir(dfs_tmpdir: str, subdir: str) -> Iterator[str]:
    """Yield ``<dfs_tmpdir>/mlflow-spark-<uuid>/<subdir>`` and clean up on exit."""
    from google_cloud_managed_mlflow.cab_credentials import (
        validate_gcs_artifact_uri,
    )

    validate_gcs_artifact_uri(dfs_tmpdir)
    clean_subdir = subdir.strip("/")
    if not clean_subdir:
        from mlflow.exceptions import MlflowException
        from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE

        raise MlflowException(
            "Scratch subdirectory must be non-empty.",
            error_code=INVALID_PARAMETER_VALUE,
        )
    run_prefix = f"{dfs_tmpdir.rstrip('/')}/mlflow-spark-{uuid.uuid4().hex}"
    try:
        yield f"{run_prefix}/{clean_subdir}"
    finally:
        gcs_delete_prefix(run_prefix)
