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

"""Transparent MLflow Spark hooks for Managed MLflow ``mlflow-gs://`` storage.

Every hook first checks whether a Managed MLflow tracking (or registry) URI is
active and the artifact location is ``gs://``; otherwise MLflow's original
implementation runs unchanged.
"""

from __future__ import annotations

import contextlib
import dataclasses
import functools
import importlib.abc
import importlib.machinery
import logging
import os
import posixpath
import shutil
import sys
import tempfile
import threading
import types
import uuid
from collections.abc import Callable, Iterator
from typing import Any

from mlflow.entities import FileInfo
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import INTERNAL_ERROR
from mlflow.store.artifact.artifact_repo import ArtifactRepository

from google_cloud_managed_mlflow.auth import (
    AUTH_PROVIDER_NAME,
    ensure_auth_provider_registered,
)
from google_cloud_managed_mlflow.spark import (
    SCHEME,
    cab_scope as cab,
    from_cab_uri,
    runtime,
    to_cab_uri,
)
from google_cloud_managed_mlflow.uri import (
    ParsedGcpUri,
    is_gcp_mlflow_uri,
    parse_gcp_uri,
)

_logger = logging.getLogger(__name__)

PATCH_MARKER = "__mlflow_cab_patched__"
SPARK_MODEL_SUBDIR = "sparkml"
STAGING_DIR = ".mlflow-cab-staging"
SCHEME_PREFIX = f"{SCHEME}://"
_LOCAL = threading.local()
_HOOK_LOCK = threading.Lock()
_HOOKS: dict[str, list[Callable[[types.ModuleType], None]]] = {}
_FINDER_INSTALLED = False


class _PostImportFinder(importlib.abc.MetaPathFinder):
    """Run registered callbacks immediately after target modules are imported."""

    def find_spec(
        self,
        fullname: str,
        path: Any,
        target: types.ModuleType | None = None,
    ) -> importlib.machinery.ModuleSpec | None:
        if fullname not in _HOOKS:
            return None
        for finder in sys.meta_path:
            if isinstance(finder, _PostImportFinder):
                continue
            find_spec = getattr(finder, "find_spec", None)
            if find_spec is None:
                continue
            spec: importlib.machinery.ModuleSpec | None = find_spec(
                fullname, path, target
            )
            if spec is not None and spec.loader is not None:
                spec.loader = _PostImportLoader(spec.loader, fullname)
                return spec
        return None


class _PostImportLoader(importlib.abc.Loader):
    def __init__(self, loader: importlib.abc.Loader, fullname: str) -> None:
        self._loader = loader
        self._fullname = fullname

    def create_module(
        self, spec: importlib.machinery.ModuleSpec
    ) -> types.ModuleType | None:
        create = getattr(self._loader, "create_module", None)
        return create(spec) if callable(create) else None

    def exec_module(self, module: types.ModuleType) -> None:
        self._loader.exec_module(module)
        for callback in list(_HOOKS.get(self._fullname, ())):
            callback(module)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._loader, name)


def when_imported(fullname: str, callback: Callable[[types.ModuleType], None]) -> None:
    """Invoke ``callback(module)`` now if loaded, and on future imports."""
    global _FINDER_INSTALLED
    with _HOOK_LOCK:
        if not _FINDER_INSTALLED:
            sys.meta_path.insert(0, _PostImportFinder())
            _FINDER_INSTALLED = True
        callbacks = _HOOKS.setdefault(fullname, [])
        if callback not in callbacks:
            callbacks.append(callback)
    loaded = sys.modules.get(fullname)
    if isinstance(loaded, types.ModuleType):
        callback(loaded)


@dataclasses.dataclass(frozen=True)
class ModelLocation:
    """Canonical ``gs://`` model location and the read scope that covers it."""

    gcs_uri: str
    read_scope: cab.CabScope


def _ensure_tracking_auth() -> None:
    ensure_auth_provider_registered()
    if not os.environ.get("MLFLOW_TRACKING_AUTH"):
        os.environ["MLFLOW_TRACKING_AUTH"] = AUTH_PROVIDER_NAME


def _parse_uri_or_none(raw_uri: str | None) -> ParsedGcpUri | None:
    if not raw_uri or not is_gcp_mlflow_uri(raw_uri):
        return None
    try:
        parsed = parse_gcp_uri(raw_uri)
    except MlflowException:
        return None
    _ensure_tracking_auth()
    return parsed


def managed_namespace(tracking_uri: str | None = None) -> ParsedGcpUri | None:
    """Return the active Managed MLflow Namespace, or None."""
    import mlflow

    uri = tracking_uri or mlflow.get_tracking_uri()
    parsed = _parse_uri_or_none(uri)
    if parsed is not None and parsed.is_namespace:
        return parsed
    return None


def managed_registry(registry_uri: str | None = None) -> ParsedGcpUri | None:
    """Return the active Managed MLflow Registry, or None."""
    import mlflow

    uri = registry_uri or mlflow.get_registry_uri()
    parsed = _parse_uri_or_none(uri)
    if parsed is not None and parsed.is_registry:
        return parsed
    return None


def is_managed_context(
    tracking_uri: str | None = None, registry_uri: str | None = None
) -> bool:
    """True when tracking or registry URIs point at Managed MLflow."""
    return (
        managed_namespace(tracking_uri) is not None
        or managed_registry(registry_uri) is not None
    )


def _mark(func: Callable[..., Any], original: Callable[..., Any]) -> Callable[..., Any]:
    setattr(func, PATCH_MARKER, True)
    func.__wrapped__ = original  # type: ignore[attr-defined]
    return func


def _is_patched(func: Any) -> bool:
    return bool(getattr(func, PATCH_MARKER, False))


def scope_for_gcs_uri(
    gcs_uri: str,
    *,
    write: bool,
    tracking_uri: str | None = None,
    registry_uri: str | None = None,
) -> cab.CabScope | None:
    """Return the CAB scope covering ``gcs_uri`` (Vault first, then Namespace)."""
    registry = managed_registry(registry_uri)
    vault = cab.lookup_vault_prefix(gcs_uri, registry)
    if vault is not None:
        _, model_resource, reg = vault
        return cab.registered_model_scope(
            reg, model_resource, write=write, artifact_uri=gcs_uri
        )
    namespace = managed_namespace(tracking_uri)
    if namespace is None:
        return None
    return cab.namespace_scope(namespace, write=write, artifact_uri=gcs_uri)


def resolve_model_location(model_uri: str) -> ModelLocation | None:
    """Resolve ``runs:/``, ``models:/``, or ``gs://`` model URIs for reading."""
    if not is_managed_context():
        return None
    gcs_uri: str | None
    if model_uri.startswith("runs:/"):
        from mlflow.store.artifact.runs_artifact_repo import RunsArtifactRepository

        gcs_uri = RunsArtifactRepository.get_underlying_uri(model_uri)
    elif model_uri.startswith("models:/"):
        from mlflow.store.artifact.models_artifact_repo import ModelsArtifactRepository

        base, sub_path = ModelsArtifactRepository.split_models_uri(model_uri)
        _, _, underlying = ModelsArtifactRepository._get_model_uri_infos(base)
        gcs_uri = posixpath.join(underlying, sub_path) if sub_path else underlying
    elif model_uri.startswith("gs://"):
        gcs_uri = model_uri
    else:
        return None
    if not gcs_uri or not gcs_uri.startswith("gs://"):
        return None
    gcs_uri = gcs_uri.rstrip("/")
    scope = scope_for_gcs_uri(gcs_uri, write=False)
    if scope is None:
        return None
    return ModelLocation(gcs_uri=gcs_uri, read_scope=scope)


class MlflowGsArtifactRepository(ArtifactRepository):
    """Spark-driver-backed artifact repository for ``mlflow-gs://`` and ``gs://``."""

    def __init__(
        self,
        artifact_uri: str,
        tracking_uri: str | None = None,
        registry_uri: str | None = None,
        *,
        read_scope: cab.CabScope | None = None,
        write_scope: cab.CabScope | None = None,
    ) -> None:
        super().__init__(artifact_uri, tracking_uri, registry_uri)
        self._explicit_scope_id: str | None = None
        if artifact_uri.startswith(SCHEME_PREFIX):
            gcs_uri, scope_id = from_cab_uri(artifact_uri)
            self._gcs_uri = gcs_uri.rstrip("/")
            self._explicit_scope_id = scope_id
        else:
            self._gcs_uri = artifact_uri.rstrip("/")
        self._read_scope = read_scope
        self._write_scope = write_scope

    def _target_gcs_uri(self, subpath: str | None = None) -> str:
        clean = (subpath or "").strip("/")
        return posixpath.join(self._gcs_uri, clean) if clean else self._gcs_uri

    def _cab_uri_for(
        self,
        spark: Any,
        target_gcs_uri: str,
        *,
        write: bool,
        reuse_active: bool = False,
    ) -> str:
        if self._explicit_scope_id is not None:
            return to_cab_uri(target_gcs_uri, self._explicit_scope_id)
        scope = self._write_scope if write else self._read_scope
        if scope is None and reuse_active:
            active_scopes = runtime._state(spark).get("scopes", {})
            for scope_id, info in active_scopes.items():
                res = info.get("result") or {}
                uri = str(
                    res.get("artifactUri") or info.get("artifact_uri") or ""
                ).rstrip("/")
                if uri and (
                    target_gcs_uri == uri or target_gcs_uri.startswith(f"{uri}/")
                ):
                    return to_cab_uri(target_gcs_uri, str(scope_id))
        if scope is None:
            boundary_uri = (
                posixpath.dirname(target_gcs_uri)
                if target_gcs_uri.endswith("/MLmodel")
                else target_gcs_uri
            )
            scope = scope_for_gcs_uri(
                boundary_uri,
                write=write,
                tracking_uri=self.tracking_uri,
                registry_uri=self.registry_uri,
            )
        if scope is None:
            raise MlflowException(
                f"Cannot resolve a Managed MLflow CAB scope for {target_gcs_uri}.",
                error_code=INTERNAL_ERROR,
            )
        scope_id = runtime.ensure_scope(spark, scope)
        return to_cab_uri(target_gcs_uri, scope_id)

    def log_artifact(self, local_file: str, artifact_path: str | None = None) -> None:
        spark = _require_session()
        target_dir = self._target_gcs_uri(artifact_path)
        cab_uri = self._cab_uri_for(spark, target_dir, write=True, reuse_active=True)
        runtime.write_single_file(
            spark, cab_uri, local_file, rel_path=os.path.basename(local_file)
        )

    def log_artifacts(self, local_dir: str, artifact_path: str | None = None) -> None:
        spark = _require_session()
        target_dir = self._target_gcs_uri(artifact_path)
        cab_uri = self._cab_uri_for(spark, target_dir, write=True, reuse_active=True)
        runtime.write_directory(spark, cab_uri, local_dir)

    def list_artifacts(self, path: str | None = None) -> list[FileInfo]:
        spark = _require_session()
        rel_prefix = (path or "").strip("/")
        target = self._target_gcs_uri(rel_prefix)
        cab_uri = self._cab_uri_for(spark, target, write=False, reuse_active=True)
        entries = runtime.list_directory(spark, cab_uri)
        infos: list[FileInfo] = []
        for entry in entries:
            name = str(entry.get("name", ""))
            if not name:
                continue
            rel = f"{rel_prefix}/{name}" if rel_prefix else name
            is_dir = bool(entry.get("isDir", False))
            size = None if is_dir else int(entry.get("size", 0))
            infos.append(FileInfo(rel, is_dir, size))
        infos.sort(key=lambda info: info.path)
        return infos

    def download_artifacts(
        self, artifact_path: str = "", dst_path: str | None = None
    ) -> str:
        spark = _require_session()
        dest_root = (
            os.path.abspath(dst_path)
            if dst_path is not None
            else tempfile.mkdtemp(prefix="mlflow-cab-dl-")
        )
        os.makedirs(dest_root, exist_ok=True)
        rel = (artifact_path or "").strip("/")
        target = self._target_gcs_uri(rel)
        cab_uri = self._cab_uri_for(spark, target, write=False, reuse_active=False)
        with tempfile.TemporaryDirectory(prefix="mlflow-cab-extract-") as tmp:
            try:
                is_dir, extracted = runtime.read_to_directory(spark, cab_uri, tmp)
            except MlflowException:
                if not rel:
                    return dest_root
                raise
            if not is_dir:
                filename = extracted[0] if extracted else posixpath.basename(target)
                src_file = os.path.join(tmp, *filename.split("/"))
                out_file = (
                    os.path.join(dest_root, *rel.split("/"))
                    if rel
                    else os.path.join(dest_root, posixpath.basename(target))
                )
                os.makedirs(os.path.dirname(out_file), exist_ok=True)
                shutil.copy2(src_file, out_file)
                return out_file
            out_dir = os.path.join(dest_root, *rel.split("/")) if rel else dest_root
            os.makedirs(out_dir, exist_ok=True)
            for item in os.listdir(tmp):
                src_item = os.path.join(tmp, item)
                dst_item = os.path.join(out_dir, item)
                if os.path.isdir(src_item):
                    shutil.copytree(src_item, dst_item, dirs_exist_ok=True)
                else:
                    shutil.copy2(src_item, dst_item)
            return out_dir

    def _download_file(self, remote_file_path: str, local_path: str) -> None:
        spark = _require_session()
        target = self._target_gcs_uri(remote_file_path)
        cab_uri = self._cab_uri_for(spark, target, write=False, reuse_active=False)
        with tempfile.TemporaryDirectory(prefix="mlflow-cab-file-") as tmp:
            _, extracted = runtime.read_to_directory(spark, cab_uri, tmp)
            if not extracted:
                raise MlflowException(
                    f"Artifact file {target} was not found.", error_code=INTERNAL_ERROR
                )
            src_file = os.path.join(tmp, *extracted[0].split("/"))
            os.makedirs(os.path.dirname(os.path.abspath(local_path)), exist_ok=True)
            shutil.copy2(src_file, local_path)

    def delete_artifacts(self, artifact_path: str | None = None) -> None:
        spark = _require_session()
        target = self._target_gcs_uri(artifact_path)
        cab_uri = self._cab_uri_for(spark, target, write=True, reuse_active=True)
        runtime.delete_path(spark, cab_uri)


def repository_for(
    location: ModelLocation, artifact_uri: str
) -> MlflowGsArtifactRepository:
    """Read-only ``MlflowGsArtifactRepository`` for ``artifact_uri``."""
    return MlflowGsArtifactRepository(
        artifact_uri.rstrip("/"),
        read_scope=location.read_scope,
        write_scope=None,
    )


def read_mlmodel(location_or_uri: ModelLocation | str) -> Any:
    """Download only the ``MLmodel`` manifest and return the parsed Model."""
    from mlflow.models import Model

    with tempfile.TemporaryDirectory(prefix="mlflow-cab-mlmodel-") as tmp:
        if isinstance(location_or_uri, ModelLocation):
            repo = repository_for(location_or_uri, location_or_uri.gcs_uri)
            local = repo.download_artifacts("MLmodel", dst_path=tmp)
        else:
            import mlflow

            uri = str(location_or_uri).rstrip("/")
            resolved = resolve_model_location(uri)
            if resolved is not None:
                repo = repository_for(resolved, resolved.gcs_uri)
                local = repo.download_artifacts("MLmodel", dst_path=tmp)
            else:
                local = mlflow.artifacts.download_artifacts(
                    artifact_uri=f"{uri}/MLmodel", dst_path=tmp
                )
        return Model.load(local)


def _save_spark_model(spark_model: Any, path: str) -> None:
    writer = getattr(spark_model, "write", None)
    if callable(writer):
        writer().overwrite().save(path)
    else:
        spark_model.save(path)


def _spark_errors() -> tuple[type[BaseException], ...]:
    errors: list[type[BaseException]] = []
    try:
        from py4j.protocol import Py4JError

        errors.append(Py4JError)
    except ImportError:  # pragma: no cover
        pass
    try:
        from pyspark.errors import PySparkException

        errors.append(PySparkException)
    except ImportError:  # pragma: no cover
        pass
    return tuple(errors) or (Exception,)


def _require_session() -> Any:
    spark = runtime.active_session()
    if spark is None:
        raise MlflowException(
            "An active SparkSession is required for Spark model artifacts.",
            error_code=INTERNAL_ERROR,
        )
    return spark


def cab_path_for(spark: Any, gcs_uri: str, scope: cab.CabScope) -> str:
    """Register ``scope`` on the driver and return the ``mlflow-gs://`` path."""
    scope_id = runtime.ensure_scope(spark, scope)
    return to_cab_uri(gcs_uri, scope_id)


# --------------------------------------------------------------------- mlflow.spark


def patch_mlflow_spark(module: types.ModuleType) -> None:
    """Install hooks on ``mlflow.spark`` (idempotent)."""
    target: Any = module
    original_maybe_save = module._maybe_save_model
    if not _is_patched(original_maybe_save):

        @functools.wraps(original_maybe_save)
        def _maybe_save_model(spark_model: Any, model_dir: str) -> bool:
            namespace = managed_namespace()
            if namespace is None or not str(model_dir).startswith("gs://"):
                return bool(original_maybe_save(spark_model, model_dir))
            spark = _require_session()
            target_dir = str(model_dir).rstrip("/")
            scope = cab.namespace_scope(namespace, write=True, artifact_uri=target_dir)
            try:
                path = cab_path_for(spark, f"{target_dir}/{SPARK_MODEL_SUBDIR}", scope)
                _save_spark_model(spark_model, path)
            except _spark_errors() as exc:
                raise MlflowException(
                    f"Failed to save Spark model to {target_dir}/{SPARK_MODEL_SUBDIR} "
                    "through "
                    f"Managed MLflow CAB credentials: {cab.redact(str(exc))[:800]}",
                    error_code=INTERNAL_ERROR,
                ) from None
            return True

        target._maybe_save_model = _mark(_maybe_save_model, original_maybe_save)

    original_tmp = module.generate_tmp_dfs_path
    if not _is_patched(original_tmp):

        @functools.wraps(original_tmp)
        def generate_tmp_dfs_path(dfs_tmp: str) -> str:
            staging = getattr(_LOCAL, "staging", None)
            if staging is None:
                return str(original_tmp(dfs_tmp))
            return str(staging())

        target.generate_tmp_dfs_path = _mark(generate_tmp_dfs_path, original_tmp)

    hadoop_fs = module._HadoopFileSystem
    original_copy = hadoop_fs.copy_to_local_file.__func__
    if not _is_patched(original_copy):

        def copy_to_local_file(cls: Any, src: str, dst: str, remove_src: bool) -> Any:
            staged = getattr(_LOCAL, "staged", None)
            if staged is None or str(src) not in staged:
                return original_copy(cls, src, dst, remove_src)
            entry: StagedPrefix = staged[str(src)]
            _download_dir(entry, dst)
            if remove_src:
                _delete_staged(entry)
            return None

        hadoop_fs.copy_to_local_file = classmethod(
            _mark(copy_to_local_file, original_copy)
        )

    original_save = module.save_model
    if not _is_patched(original_save):

        @functools.wraps(original_save)
        def save_model(spark_model: Any, path: str, *args: Any, **kwargs: Any) -> Any:
            import mlflow

            namespace = managed_namespace()
            dfs_tmpdir = kwargs.get("dfs_tmpdir")
            if len(args) >= 4:
                dfs_tmpdir = args[3]
            if namespace is None or dfs_tmpdir is not None:
                return original_save(spark_model, path, *args, **kwargs)
            with _managed_staging(namespace, mlflow.active_run()):
                return original_save(spark_model, path, *args, **kwargs)

        target.save_model = _mark(save_model, original_save)

    original_load = module.load_model
    if not _is_patched(original_load):

        @functools.wraps(original_load)
        def load_model(
            model_uri: str, dfs_tmpdir: str | None = None, dst_path: str | None = None
        ) -> Any:
            location = (
                resolve_model_location(str(model_uri)) if dfs_tmpdir is None else None
            )
            if location is None:
                return original_load(
                    model_uri, dfs_tmpdir=dfs_tmpdir, dst_path=dst_path
                )
            return load_spark_model(str(model_uri), location)

        target.load_model = _mark(load_model, original_load)


@dataclasses.dataclass
class StagedPrefix:
    """One temporary ``mlflow-gs://`` staging prefix used by ``save_model``."""

    gcs_uri: str
    read_scope: cab.CabScope
    write_scope: cab.CabScope
    project: str
    deleted: bool = False

    def repository(self) -> MlflowGsArtifactRepository:
        """Driver-backed repository over the staging prefix (read + delete)."""
        return MlflowGsArtifactRepository(
            self.gcs_uri,
            read_scope=self.read_scope,
            write_scope=self.write_scope,
        )


@contextlib.contextmanager
def _managed_staging(namespace: ParsedGcpUri, run: Any) -> Iterator[None]:
    """Route ``mlflow.spark.save_model`` DFS staging through a CAB-governed prefix."""
    spark = _require_session()
    staged: dict[str, StagedPrefix] = {}
    artifact_root = str(run.info.artifact_uri).rstrip("/") if run is not None else None

    def _staging() -> str:
        token = uuid.uuid4().hex
        if artifact_root is not None:
            gcs_uri = f"{artifact_root}/{STAGING_DIR}/{token}"
            write_scope = cab.namespace_scope(
                namespace, write=True, artifact_uri=gcs_uri
            )
            scope_id = runtime.ensure_scope(spark, write_scope)
        else:
            write_scope = cab.namespace_scope(
                namespace, write=True, artifact_path=f"{STAGING_DIR}/{token}"
            )
            scope_id = runtime.ensure_scope(spark, write_scope)
            gcs_uri = runtime.scope_artifact_uri(spark, write_scope)
        read_scope = cab.namespace_scope(namespace, write=False, artifact_uri=gcs_uri)
        path = to_cab_uri(gcs_uri, scope_id)
        staged[path] = StagedPrefix(gcs_uri, read_scope, write_scope, namespace.project)
        return path

    _LOCAL.staging = _staging
    _LOCAL.staged = staged
    try:
        yield
    finally:
        _LOCAL.staging = None
        _LOCAL.staged = None
        for entry in staged.values():
            if not entry.deleted:
                try:
                    _delete_staged(entry)
                except Exception:
                    _logger.warning(
                        "Failed to delete MLflow staging prefix %s",
                        entry.gcs_uri,
                        exc_info=True,
                    )
            try:
                runtime.release_scope(spark, entry.write_scope)
            except Exception:
                _logger.debug("Failed to release staging scope", exc_info=True)


def _delete_staged(entry: StagedPrefix) -> None:
    entry.repository().delete_artifacts()
    entry.deleted = True


def _download_dir(entry: StagedPrefix, dst: str) -> None:
    repo = entry.repository()
    os.makedirs(dst, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="mlflow-cab-stage-") as tmp:
        downloaded = repo.download_artifacts("", dst_path=tmp)
        for name in os.listdir(downloaded):
            source = os.path.join(downloaded, name)
            target = os.path.join(dst, name)
            if os.path.isdir(source):
                shutil.copytree(source, target, dirs_exist_ok=True)
            else:
                shutil.copy2(source, target)


def _connect_python_pipeline(spark: Any, location: ModelLocation, meta: Any) -> Any:
    """Decoded pipeline for Python-stage models on Spark Connect, else None."""
    if not runtime.is_spark_connect(spark):
        return None
    from google_cloud_managed_mlflow.spark import udf as managed_udf

    data_uri, stages = managed_udf.read_pipeline_stages(location, meta)
    if not managed_udf.has_python_stages(stages):
        return None
    return managed_udf.decode_xgboost_pipeline(data_uri, location, stages)


def load_spark_model(model_uri: str, location: ModelLocation) -> Any:
    """Load a Spark model directly from CAB-governed storage (no local weights).

    On Spark Connect, pipelines with Python-only stages (``SparkXGB*``, whose
    readers require a ``SparkContext``) load as a transformer that evaluates the
    decoded pipeline with a pandas UDF and appends the prediction column.
    """
    from mlflow.utils.class_utils import _get_class_from_string

    model = read_mlmodel(location)
    flavor = model.flavors.get("spark", {})
    model_data = flavor.get("model_data", SPARK_MODEL_SUBDIR)
    model_class = flavor.get("model_class") or "pyspark.ml.pipeline.PipelineModel"
    spark = _require_session()
    if str(model_class).startswith("pyspark.ml.connect."):
        raise MlflowException(
            "pyspark.ml.connect models are stored locally and are not supported here.",
            error_code=INTERNAL_ERROR,
        )
    decoded = _connect_python_pipeline(spark, location, model)
    if decoded is not None:
        from google_cloud_managed_mlflow.spark import udf as managed_udf

        return managed_udf.connect_pipeline_model(decoded)
    path = cab_path_for(spark, f"{location.gcs_uri}/{model_data}", location.read_scope)
    try:
        from pyspark.ml.pipeline import PipelineModel

        if model_class in (
            "pyspark.ml.pipeline.PipelineModel",
            "pyspark.ml.PipelineModel",
        ):
            return PipelineModel.load(path)
        loader = _get_class_from_string(model_class)
        return loader.load(path)
    except _spark_errors() as exc:
        raise MlflowException(
            f"Failed to load Spark model from {location.gcs_uri}: "
            f"{cab.redact(str(exc))[:800]}",
            error_code=INTERNAL_ERROR,
        ) from None


# ---------------------------------------------------------------- mlflow.pyspark.ml

XGBOOST_ALLOWLIST_ENTRIES = (
    "xgboost.spark.*",
    "xgboost.spark.core.SparkXGBClassifierModel",
    "xgboost.spark.core.SparkXGBRegressorModel",
    "xgboost.spark.core.SparkXGBRankerModel",
    "xgboost.spark.estimator.SparkXGBClassifierModel",
    "xgboost.spark.estimator.SparkXGBRegressorModel",
    "xgboost.spark.estimator.SparkXGBRankerModel",
)


def patch_pyspark_ml(module: types.ModuleType) -> None:
    """Install hooks on ``mlflow.pyspark.ml`` (idempotent)."""
    target: Any = module
    original_allowlist = module._read_log_model_allowlist
    if not _is_patched(original_allowlist):

        @functools.wraps(original_allowlist)
        def _read_log_model_allowlist() -> Any:
            if not is_managed_context():
                return original_allowlist()
            import importlib.resources

            builtin = importlib.resources.files(module.__name__).joinpath(
                "log_model_allowlist.txt"
            )
            spark = runtime.active_session()
            custom = None
            if spark is not None:
                try:
                    custom = spark.conf.get(
                        "spark.mlflow.pysparkml.autolog.logModelAllowlistFile", None
                    )
                except Exception:
                    custom = None
            allowlist: set[str]
            with importlib.resources.as_file(builtin) as builtin_path:
                try:
                    allowlist = set(
                        module._read_log_model_allowlist_from_file(
                            custom or builtin_path.as_posix()
                        )
                    )
                except Exception:
                    _logger.exception(
                        "Reading allowlist %s failed; using built-in.", custom
                    )
                    allowlist = set(
                        module._read_log_model_allowlist_from_file(
                            builtin_path.as_posix()
                        )
                    )
            allowlist.update(XGBOOST_ALLOWLIST_ENTRIES)
            return allowlist

        target._read_log_model_allowlist = _mark(
            _read_log_model_allowlist, original_allowlist
        )

    original_autolog = module.autolog
    if not _is_patched(original_autolog):
        import inspect

        autolog_signature = inspect.signature(original_autolog)

        @functools.wraps(original_autolog)
        def autolog(*args: Any, **kwargs: Any) -> Any:
            result = original_autolog(*args, **kwargs)
            try:
                bound = autolog_signature.bind_partial(*args, **kwargs)
                disable = bool(bound.arguments.get("disable", False))
            except TypeError:
                disable = False
            if not disable:
                _guard_estimator_fits()
            return result

        target.autolog = _mark(autolog, original_autolog)


def _all_estimator_classes() -> list[type]:
    from pyspark.ml.base import Estimator

    seen: list[type] = [Estimator]
    stack: list[type] = [Estimator]
    while stack:
        cls = stack.pop()
        for sub in cls.__subclasses__():
            if sub not in seen:
                seen.append(sub)
                stack.append(sub)
    return seen


def _guard_estimator_fits() -> None:
    """Wrap autolog-patched ``fit`` methods to protect caller-owned caches.

    MLflow autologging persists the training DataFrame and unpersists it at the
    end of ``fit``. If the caller had already cached the DataFrame, that would
    evict the caller's cache; if logging failed midway, the autolog cache would
    leak. The guard keeps caller caches and always releases autolog-only caches.
    """
    for cls in _all_estimator_classes():
        current = cls.__dict__.get("fit")
        if current is None or _is_patched(current):
            continue
        code = getattr(current, "__code__", None)
        filename = str(getattr(code, "co_filename", "")).replace(os.sep, "/")
        if "mlflow/utils/autologging_utils" not in filename:
            continue
        guarded: Any = cls
        guarded.fit = _mark(_make_guarded_fit(current), current)


def _make_guarded_fit(patched_fit: Callable[..., Any]) -> Callable[..., Any]:
    @functools.wraps(patched_fit)
    def fit(self: Any, dataset: Any, *args: Any, **kwargs: Any) -> Any:
        if not is_managed_context() or not hasattr(dataset, "is_cached"):
            return patched_fit(self, dataset, *args, **kwargs)
        try:
            was_cached = bool(dataset.is_cached)
        except Exception:
            return patched_fit(self, dataset, *args, **kwargs)
        original_unpersist = dataset.__dict__.get("unpersist")
        if was_cached:
            dataset.unpersist = lambda *a, **k: dataset  # keep the caller's cache
        try:
            return patched_fit(self, dataset, *args, **kwargs)
        finally:
            if was_cached:
                if original_unpersist is None:
                    dataset.__dict__.pop("unpersist", None)
                else:
                    dataset.unpersist = original_unpersist
            else:
                try:
                    if dataset.is_cached:
                        dataset.unpersist()
                except Exception:  # pragma: no cover - best effort cleanup
                    _logger.debug("Failed to release autolog cache", exc_info=True)

    return fit


# --------------------------------------------------------------------- mlflow.pyfunc


def patch_pyfunc(module: types.ModuleType) -> None:
    """Install hooks on ``mlflow.pyfunc`` (idempotent)."""
    target: Any = module
    original_load = module.load_model
    if not _is_patched(original_load):

        @functools.wraps(original_load)
        def load_model(model_uri: Any, *args: Any, **kwargs: Any) -> Any:
            uri = str(model_uri)
            location = resolve_model_location(uri) if _is_remote_uri(uri) else None
            if location is not None:
                meta = read_mlmodel(location)
                if _is_spark_flavor(meta):
                    return _spark_pyfunc(uri, location, meta)
            return original_load(model_uri, *args, **kwargs)

        target.load_model = _mark(load_model, original_load)

    original_udf = module.spark_udf
    if not _is_patched(original_udf):

        @functools.wraps(original_udf)
        def spark_udf(spark: Any, model_uri: Any, *args: Any, **kwargs: Any) -> Any:
            uri = str(model_uri)
            env_manager = kwargs.get("env_manager", args[1] if len(args) > 1 else None)
            prebuilt = kwargs.get("prebuilt_env_uri")
            location = (
                resolve_model_location(uri)
                if _is_remote_uri(uri)
                and env_manager in (None, "local")
                and not prebuilt
                else None
            )
            if location is None:
                return original_udf(spark, model_uri, *args, **kwargs)
            from google_cloud_managed_mlflow.spark import udf as managed_udf

            result_type = kwargs.get("result_type", args[0] if args else None)
            params = kwargs.get("params")
            model_config = kwargs.get("model_config")
            meta = read_mlmodel(location)
            if _is_spark_flavor(meta):
                return managed_udf.spark_flavor_udf(
                    spark, uri, location, meta, result_type
                )
            return managed_udf.python_flavor_udf(
                spark, uri, location, meta, result_type, params, model_config
            )

        target.spark_udf = _mark(spark_udf, original_udf)


def _is_remote_uri(uri: str) -> bool:
    return uri.startswith(("runs:/", "models:/", "gs://"))


def _is_spark_flavor(meta: Any) -> bool:
    pyfunc_conf = meta.flavors.get("python_function", {})
    return (
        "spark" in meta.flavors and pyfunc_conf.get("loader_module") == "mlflow.spark"
    )


def _spark_pyfunc(model_uri: str, location: ModelLocation, meta: Any) -> Any:
    import mlflow.pyfunc
    import mlflow.spark

    spark = _require_session()
    decoded = _connect_python_pipeline(spark, location, meta)
    if decoded is not None:
        from google_cloud_managed_mlflow.spark import udf as managed_udf

        return mlflow.pyfunc.PyFuncModel(
            model_meta=meta, model_impl=managed_udf.LocalPipelinePredictor(decoded)
        )
    spark_model = load_spark_model(model_uri, location)
    wrapper = mlflow.spark._PyFuncModelWrapper(
        spark, spark_model, signature=meta.signature
    )
    return mlflow.pyfunc.PyFuncModel(model_meta=meta, model_impl=wrapper)


def _patch_artifact_registry(module: types.ModuleType) -> None:
    """Register ``mlflow-gs`` and wrap stock ``gs`` when Spark + Managed MLflow run."""
    registry = getattr(module, "_artifact_repository_registry", None)
    if registry is None:
        return
    mapping = getattr(registry, "_registry", None)
    if not isinstance(mapping, dict):
        return
    mapping[SCHEME] = MlflowGsArtifactRepository
    current_gs = mapping.get("gs")
    if current_gs is None or _is_patched(current_gs):
        return
    if getattr(current_gs, "__name__", "") == "GCSArtifactRepository":

        def _gs_spark_fallback(
            artifact_uri: str,
            tracking_uri: str | None = None,
            registry_uri: str | None = None,
        ) -> Any:
            if (
                is_managed_context(tracking_uri, registry_uri)
                and runtime.active_session() is not None
            ):
                return MlflowGsArtifactRepository(
                    artifact_uri, tracking_uri=tracking_uri, registry_uri=registry_uri
                )
            return current_gs(
                artifact_uri, tracking_uri=tracking_uri, registry_uri=registry_uri
            )

        mapping["gs"] = _mark(_gs_spark_fallback, current_gs)


def install() -> None:
    """Install lazy post-import hooks for ``mlflow.spark`` and ``mlflow-gs``."""
    when_imported(
        "mlflow.store.artifact.artifact_repository_registry", _patch_artifact_registry
    )
    when_imported("mlflow.spark", patch_mlflow_spark)
    when_imported("mlflow.pyspark.ml", patch_pyspark_ml)
    when_imported("mlflow.pyfunc", patch_pyfunc)


install()
