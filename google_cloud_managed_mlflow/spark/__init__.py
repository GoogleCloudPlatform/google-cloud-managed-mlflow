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

"""Spark integration for Managed MLflow Credential Access Boundary (CAB) tokens.

The bundled ``mlflow-gcs-cab-provider.jar`` adds the ``mlflow-gs://`` Hadoop
filesystem. Spark reads and writes MLflow artifacts through
``mlflow-gs://<scope>@<bucket>/<path>`` while MLflow metadata keeps canonical
``gs://`` URIs. The driver mints one downscoped token per scope and broadcasts
it to executors, which never call the MLflow control plane.
"""

from __future__ import annotations

import atexit
import os
import pkgutil
import shutil
import tempfile
import threading
from pathlib import Path

JAR_NAME = "mlflow-gcs-cab-provider.jar"
SCHEME = "mlflow-gs"
FILESYSTEM_CLASS = "com.google.cloud.mlflow.gcs.MlflowCabFileSystem"
ABSTRACT_FILESYSTEM_CLASS = "com.google.cloud.mlflow.gcs.MlflowCabAbstractFileSystem"
CONTROL_SOURCE_CLASS = "com.google.cloud.mlflow.gcs.MlflowCabDriverControlSource"
SPARK_PLUGIN_CLASS = "com.google.cloud.mlflow.gcs.MlflowCabSparkPlugin"

_EXTRACT_LOCK = threading.Lock()
_extracted_jar: str | None = None


def bundled_jar_path() -> str:
    """Return a filesystem path of the bundled provider JAR.

    When the package is installed normally this is the JAR inside the package.
    When it is imported from a zip archive (for example a wheel shipped with
    ``spark-submit --py-files``) the JAR is extracted once per process into a
    private (``0700``) temporary directory that is removed at interpreter exit.
    """
    global _extracted_jar
    packaged = Path(__file__).resolve().parent.parent / "jars" / JAR_NAME
    if os.path.isfile(packaged):
        return str(packaged)
    with _EXTRACT_LOCK:
        if _extracted_jar is not None and os.path.isfile(_extracted_jar):
            return _extracted_jar
        data = pkgutil.get_data("google_cloud_managed_mlflow", f"jars/{JAR_NAME}")
        if not data:
            raise FileNotFoundError(f"Bundled provider JAR {JAR_NAME} is missing.")
        directory = tempfile.mkdtemp(prefix="mlflow-cab-provider-")
        os.chmod(directory, 0o700)
        atexit.register(shutil.rmtree, directory, True)
        target = os.path.join(directory, JAR_NAME)
        with open(target, "xb") as handle:
            handle.write(data)
        _extracted_jar = target
        return target


def to_cab_uri(gcs_uri: str, scope_id: str) -> str:
    """Translate ``gs://bucket/path`` into ``mlflow-gs://<scope>@bucket/path``."""
    if not gcs_uri.startswith("gs://"):
        raise ValueError(f"Not a gs:// URI: {gcs_uri!r}")
    remainder = gcs_uri[len("gs://") :]
    return f"{SCHEME}://{scope_id}@{remainder}"


def from_cab_uri(cab_uri: str) -> tuple[str, str]:
    """Translate ``mlflow-gs://<scope>@bucket/path`` into ``(gs_uri, scope_id)``."""
    prefix = f"{SCHEME}://"
    if not cab_uri.startswith(prefix):
        raise ValueError(f"Not an {prefix} URI: {cab_uri!r}")
    remainder = cab_uri[len(prefix) :]
    authority, sep, path = remainder.partition("/")
    scope_id, at, bucket = authority.partition("@")
    if not at or not scope_id or not bucket:
        raise ValueError(f"Missing <scope>@<bucket> authority in {cab_uri!r}")
    gcs_uri = f"gs://{bucket}/{path}" if sep else f"gs://{bucket}"
    return gcs_uri, scope_id


__all__ = [
    "ABSTRACT_FILESYSTEM_CLASS",
    "CONTROL_SOURCE_CLASS",
    "FILESYSTEM_CLASS",
    "JAR_NAME",
    "SCHEME",
    "SPARK_PLUGIN_CLASS",
    "bundled_jar_path",
    "from_cab_uri",
    "to_cab_uri",
]

from google_cloud_managed_mlflow.spark import hooks as _hooks  # noqa: E402

_hooks.install()
