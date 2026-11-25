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

"""Lazy post-import MLflow Spark, PySpark ML, and PyFunc hooks."""

from __future__ import annotations

import atexit
import dataclasses
import functools
import importlib.abc
import importlib.machinery
import inspect
import logging
import os
import shutil
import sys
import tempfile
import threading
import types
from collections.abc import Callable
from typing import Any, cast

from google_cloud_managed_mlflow.spark import runtime

_logger = logging.getLogger(__name__)

PATCH_MARKER = "__mlflow_gcp_spark_patched__"
SPARK_MODEL_SUBDIR = "sparkml"
_LOCAL = threading.local()
_HOOK_LOCK = threading.Lock()
_HOOKS: dict[str, list[Callable[[types.ModuleType], None]]] = {}
_APPLIED_PATCHES: list[tuple[Any, str, Any]] = []
_FINDER_INSTALLED = False


@dataclasses.dataclass
class _ProtectedDatasetEntry:
    ref_count: int
    caller_owned: bool
    # Strong reference prevents id(dataset) recycling while ref_count > 0.
    dataset: Any
    orig_unpersist: Any


_GUARD_LOCK = threading.Lock()
_PROTECTED_DATASETS: dict[int, _ProtectedDatasetEntry] = {}

XGBOOST_ALLOWLIST_ENTRIES = (
    "xgboost.spark.*",
    "xgboost.spark.core.SparkXGBClassifierModel",
    "xgboost.spark.core.SparkXGBRegressorModel",
    "xgboost.spark.core.SparkXGBRankerModel",
    "xgboost.spark.estimator.SparkXGBClassifierModel",
    "xgboost.spark.estimator.SparkXGBRegressorModel",
    "xgboost.spark.estimator.SparkXGBRankerModel",
)


def _run_hook_safely(
    callback: Callable[[types.ModuleType], None], module: types.ModuleType
) -> None:
    try:
        callback(module)
    except Exception:
        _logger.warning(
            "Failed to apply Managed MLflow Spark hook %s to %s; "
            "falling back to stock MLflow behavior.",
            getattr(callback, "__name__", repr(callback)),
            module.__name__,
            exc_info=True,
        )


class _PostImportFinder(importlib.abc.MetaPathFinder):
    """Run registered callbacks immediately after target modules are imported."""

    def find_spec(
        self,
        fullname: str,
        path: Any,
        target: types.ModuleType | None = None,
    ) -> importlib.machinery.ModuleSpec | None:
        with _HOOK_LOCK:
            has_hook = fullname in _HOOKS
        if not has_hook:
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
                if not isinstance(spec.loader, _PostImportLoader):
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
        with _HOOK_LOCK:
            callbacks = list(_HOOKS.get(self._fullname, ()))
        for callback in callbacks:
            _run_hook_safely(callback, module)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._loader, name)


def _attach_post_init_spec(spec: Any, fullname: str, loaded: types.ModuleType) -> None:
    """Defer post-import callbacks until ``spec._initializing`` clears.

    Cold ``import mlflow.spark`` imports ``mlflow.tracking._model_registry.utils``
    and triggers plugin entry-point discovery while ``mlflow.spark`` is still
    mid-execution (``_initializing == True`` and target functions are not yet
    bound). Waiting for ``_initializing = False`` avoids mid-import
    ``AttributeError`` while still patching before ``import mlflow.spark`` returns.
    """
    orig_cls = spec.__class__
    if getattr(orig_cls, "_mlflow_gcp_post_init", False):
        return

    class _PostInitSpec(orig_cls):  # type: ignore[valid-type, misc]
        _mlflow_gcp_post_init = True

        def __setattr__(self, name: str, value: Any) -> None:
            super().__setattr__(name, value)
            if name == "_initializing" and not value:
                self.__class__ = orig_cls
                if sys.modules.get(fullname) is loaded:
                    with _HOOK_LOCK:
                        callbacks = list(_HOOKS.get(fullname, ()))
                    for cb in callbacks:
                        _run_hook_safely(cb, loaded)

    try:
        spec.__class__ = _PostInitSpec
    except TypeError:
        _logger.debug(
            "Unable to attach post-init spec wrapper for %s", fullname, exc_info=True
        )


def when_imported(fullname: str, callback: Callable[[types.ModuleType], None]) -> None:
    """Invoke ``callback(module)`` now if initialized, or on import completion."""
    global _FINDER_INSTALLED
    call_now: types.ModuleType | None = None
    with _HOOK_LOCK:
        if not _FINDER_INSTALLED:
            sys.meta_path.insert(0, _PostImportFinder())
            _FINDER_INSTALLED = True
        callbacks = _HOOKS.setdefault(fullname, [])
        if callback not in callbacks:
            callbacks.append(callback)
        loaded = sys.modules.get(fullname)
        if isinstance(loaded, types.ModuleType):
            spec = getattr(loaded, "__spec__", None)
            if spec is not None and getattr(spec, "_initializing", False):
                _attach_post_init_spec(spec, fullname, loaded)
            else:
                call_now = loaded
    if call_now is not None:
        _run_hook_safely(callback, call_now)


def install_spark_hooks() -> None:
    """Install lazy post-import hooks for MLflow Spark, PySpark ML, and PyFunc."""
    try:
        when_imported("mlflow.pyspark.ml", patch_pyspark_ml)
        when_imported("mlflow.spark", patch_mlflow_spark)
        when_imported("mlflow.pyfunc", patch_pyfunc)
    except Exception:
        _logger.debug("Failed to install Spark hooks", exc_info=True)


def reset_spark_hooks_for_testing() -> None:
    """Restore patched MLflow/PySpark callables and clear global hook state."""
    global _FINDER_INSTALLED
    with _HOOK_LOCK:
        _HOOKS.clear()
        applied = list(reversed(_APPLIED_PATCHES))
        _APPLIED_PATCHES.clear()
        sys.meta_path[:] = [
            finder
            for finder in sys.meta_path
            if not isinstance(finder, _PostImportFinder)
        ]
        _FINDER_INSTALLED = False

    with _GUARD_LOCK:
        for entry in _PROTECTED_DATASETS.values():
            try:
                if entry.orig_unpersist is None:
                    entry.dataset.__dict__.pop("unpersist", None)
                else:
                    entry.dataset.unpersist = entry.orig_unpersist
            except (AttributeError, TypeError):
                pass
        _PROTECTED_DATASETS.clear()

    _LOCAL.log_model_dfs_tmpdir = None

    for target, attr, original in applied:
        try:
            if _is_patched(getattr(target, attr, None)):
                setattr(target, attr, original)
        except Exception:
            _logger.debug(
                "Failed to restore %s.%s during test reset", target, attr, exc_info=True
            )


def _mark(func: Callable[..., Any], original: Callable[..., Any]) -> Callable[..., Any]:
    setattr(func, PATCH_MARKER, True)
    func.__wrapped__ = original  # type: ignore[attr-defined]
    return func


def _is_patched(func: Any) -> bool:
    return bool(getattr(func, PATCH_MARKER, False))


def _patch_attr(
    target: Any, attr: str, wrapper: Callable[..., Any], original: Any
) -> None:
    marked = _mark(wrapper, original)
    setattr(target, attr, marked)
    with _HOOK_LOCK:
        _APPLIED_PATCHES.append((target, attr, original))


def _ensure_managed_registrations(eff_tracking: str, eff_registry: str) -> None:
    from google_cloud_managed_mlflow.auth import ensure_auth_provider_registered
    from google_cloud_managed_mlflow.uri import (
        is_gcp_registry_uri,
        is_gcp_tracking_uri,
    )

    ensure_auth_provider_registered()
    if is_gcp_tracking_uri(eff_tracking):
        from google_cloud_managed_mlflow.tracking_store import (
            ensure_tracking_store_registered,
        )

        ensure_tracking_store_registered()
    if is_gcp_registry_uri(eff_registry):
        from google_cloud_managed_mlflow.registry_store import (
            ensure_registry_store_registered,
        )

        ensure_registry_store_registered()


def ensure_managed_context(
    tracking_uri: str | None = None,
    registry_uri: str | None = None,
) -> bool:
    """Return True if targeting Managed MLflow, registering auth/stores if active.

    For a pure, side-effect-free predicate without store/auth registration, use
    ``google_cloud_managed_mlflow.uri.is_gcp_context``.
    """
    from google_cloud_managed_mlflow.uri import resolve_managed_uris

    eff_tracking, eff_registry, managed = resolve_managed_uris(
        tracking_uri, registry_uri
    )
    if managed:
        _ensure_managed_registrations(eff_tracking, eff_registry)
    return managed


is_managed_context = ensure_managed_context


def _spark_errors() -> tuple[type[BaseException], ...]:
    errors: list[type[BaseException]] = []
    try:
        from py4j.protocol import Py4JError

        errors.append(Py4JError)
    except ImportError:
        pass
    try:
        from pyspark.errors import PySparkException

        errors.append(PySparkException)
    except ImportError:
        pass
    return tuple(errors) or (Exception,)


def _save_spark_model(spark_model: Any, path: str) -> None:
    writer = getattr(spark_model, "write", None)
    if callable(writer):
        writer().overwrite().save(path)
    else:
        spark_model.save(path)


def _save_spark_model_via_gcs_scratch(
    spark_model: Any, dfs_tmpdir: str, local_dst_dir: str
) -> None:
    """Save ``spark_model`` to a scratch ``gs://`` prefix and download locally."""
    from mlflow.exceptions import MlflowException
    from mlflow.protos.databricks_pb2 import INTERNAL_ERROR

    os.makedirs(local_dst_dir, exist_ok=True)
    with runtime.gcs_scratch_dir(dfs_tmpdir, SPARK_MODEL_SUBDIR) as staged_gs:
        _save_spark_model(spark_model, staged_gs)
        if runtime.gcs_download_prefix(staged_gs, local_dst_dir) <= 0:
            raise MlflowException(
                f"Spark model save to {staged_gs} produced no files.",
                error_code=INTERNAL_ERROR,
            )


def _load_spark_class(model_class: str) -> Callable[[str], Any]:
    if model_class in (
        "pyspark.ml.pipeline.PipelineModel",
        "pyspark.ml.PipelineModel",
    ):
        from pyspark.ml.pipeline import PipelineModel

        return cast(Callable[[str], Any], PipelineModel.load)
    from mlflow.utils.class_utils import _get_class_from_string

    cls: Any = _get_class_from_string(model_class)
    return cast(Callable[[str], Any], cls.load)


# --------------------------------------------------------------------- mlflow.pyfunc


def _merge_udf_extra_env(existing: Any, eff_registry: str | None) -> dict[str, str]:
    merged: dict[str, str] = dict(existing) if existing else {}
    if eff_registry:
        merged.setdefault("MLFLOW_REGISTRY_URI", eff_registry)
    if "MLFLOW_GCP_ENDPOINT_OVERRIDE" in os.environ:
        merged.setdefault(
            "MLFLOW_GCP_ENDPOINT_OVERRIDE",
            os.environ["MLFLOW_GCP_ENDPOINT_OVERRIDE"],
        )
    return merged


def patch_pyfunc(module: types.ModuleType) -> None:
    """Install hooks on ``mlflow.pyfunc`` (idempotent)."""
    original_udf = getattr(module, "spark_udf", None)
    if not callable(original_udf) or _is_patched(original_udf):
        return

    udf_signature = inspect.signature(original_udf)

    @functools.wraps(original_udf)
    def spark_udf(spark: Any, model_uri: Any, *args: Any, **kwargs: Any) -> Any:
        from google_cloud_managed_mlflow.uri import (
            is_gcp_registry_uri,
            resolve_managed_uris,
        )

        eff_tracking, eff_registry, managed = resolve_managed_uris()
        if not managed:
            return original_udf(spark, model_uri, *args, **kwargs)
        _ensure_managed_registrations(eff_tracking, eff_registry)
        reg_to_inject = eff_registry if is_gcp_registry_uri(eff_registry) else None
        if reg_to_inject is None and "MLFLOW_GCP_ENDPOINT_OVERRIDE" not in os.environ:
            return original_udf(spark, model_uri, *args, **kwargs)

        if "extra_env" in udf_signature.parameters:
            bound = udf_signature.bind_partial(spark, model_uri, *args, **kwargs)
            bound.arguments["extra_env"] = _merge_udf_extra_env(
                bound.arguments.get("extra_env"), reg_to_inject
            )
            return original_udf(*bound.args, **bound.kwargs)

        new_kwargs = dict(kwargs)
        new_kwargs["extra_env"] = _merge_udf_extra_env(
            kwargs.get("extra_env"), reg_to_inject
        )
        return original_udf(spark, model_uri, *args, **new_kwargs)

    _patch_attr(module, "spark_udf", spark_udf, original_udf)


# ---------------------------------------------------------------- mlflow.pyspark.ml


def _guard_on_import(_module: types.ModuleType) -> None:
    _guard_estimator_fits()


def patch_pyspark_ml(module: types.ModuleType) -> None:
    """Install hooks on ``mlflow.pyspark.ml`` (idempotent)."""
    original_allowlist = getattr(module, "_read_log_model_allowlist", None)
    if callable(original_allowlist) and not _is_patched(original_allowlist):

        @functools.wraps(original_allowlist)
        def _read_log_model_allowlist() -> Any:
            if not ensure_managed_context():
                return original_allowlist()
            # Upstream _read_log_model_allowlist accesses spark.sparkContext._conf,
            # raising [JVM_ATTRIBUTE_NOT_SUPPORTED] on Spark Connect and restricted
            # clusters; read via spark.conf instead across all Managed MLflow runs.
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

        _patch_attr(
            module,
            "_read_log_model_allowlist",
            _read_log_model_allowlist,
            original_allowlist,
        )

    original_autolog = getattr(module, "autolog", None)
    if callable(original_autolog) and not _is_patched(original_autolog):
        autolog_signature = inspect.signature(original_autolog)

        @functools.wraps(original_autolog)
        def autolog(*args: Any, **kwargs: Any) -> Any:
            if not ensure_managed_context():
                return original_autolog(*args, **kwargs)
            result = original_autolog(*args, **kwargs)
            try:
                bound = autolog_signature.bind_partial(*args, **kwargs)
                disable = bool(bound.arguments.get("disable", False))
            except TypeError:
                disable = False
            if not disable:
                _guard_estimator_fits()
                when_imported("xgboost.spark", _guard_on_import)
                when_imported("xgboost.spark.estimator", _guard_on_import)
            return result

        _patch_attr(module, "autolog", autolog, original_autolog)


def _all_estimator_classes() -> list[type[Any]]:
    try:
        from pyspark.ml.base import Estimator
    except ImportError:
        return []

    seen: list[type[Any]] = [Estimator]
    stack: list[type[Any]] = [Estimator]
    while stack:
        cls = stack.pop()
        for sub in cls.__subclasses__():
            if sub not in seen:
                seen.append(sub)
                stack.append(sub)
    return seen


def _is_autolog_patched_fit(cls: type[Any], current: Any) -> bool:
    if (
        "_gorilla_active_patch_fit" in cls.__dict__
        or "_gorilla_original_fit" in cls.__dict__
    ):
        return True
    code = getattr(current, "__code__", None)
    filename = str(getattr(code, "co_filename", "")).replace(os.sep, "/")
    return "mlflow/utils/autologging_utils" in filename


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
        if not _is_autolog_patched_fit(cls, current):
            continue
        _patch_attr(cls, "fit", _make_guarded_fit(current), current)


def _make_guarded_fit(patched_fit: Callable[..., Any]) -> Callable[..., Any]:
    @functools.wraps(patched_fit)
    def fit(self: Any, dataset: Any, *args: Any, **kwargs: Any) -> Any:
        if not ensure_managed_context() or not hasattr(dataset, "is_cached"):
            return patched_fit(self, dataset, *args, **kwargs)

        ds_id = id(dataset)
        slotted_or_unreadable = False
        with _GUARD_LOCK:
            entry = _PROTECTED_DATASETS.get(ds_id)
            if entry is None:
                try:
                    caller_owned = bool(dataset.is_cached)
                    orig_unpersist = dataset.__dict__.get("unpersist")
                    dataset.unpersist = lambda *a, **k: dataset
                except (AttributeError, TypeError):
                    slotted_or_unreadable = True
                except Exception:
                    slotted_or_unreadable = True
                else:
                    _PROTECTED_DATASETS[ds_id] = _ProtectedDatasetEntry(
                        ref_count=1,
                        caller_owned=caller_owned,
                        dataset=dataset,
                        orig_unpersist=orig_unpersist,
                    )
            else:
                entry.ref_count += 1
        if slotted_or_unreadable:
            return patched_fit(self, dataset, *args, **kwargs)
        try:
            return patched_fit(self, dataset, *args, **kwargs)
        finally:
            should_unpersist = False
            with _GUARD_LOCK:
                entry = _PROTECTED_DATASETS.get(ds_id)
                if entry is not None:
                    entry.ref_count -= 1
                    if entry.ref_count <= 0:
                        _PROTECTED_DATASETS.pop(ds_id, None)
                        try:
                            if entry.orig_unpersist is None:
                                entry.dataset.__dict__.pop("unpersist", None)
                            else:
                                entry.dataset.unpersist = entry.orig_unpersist
                        except (AttributeError, TypeError):
                            pass
                        should_unpersist = not entry.caller_owned
            if should_unpersist:
                try:
                    if dataset.is_cached:
                        dataset.unpersist()
                except Exception:  # pragma: no cover - best effort cleanup
                    _logger.debug("Failed to release autolog cache", exc_info=True)

    return fit


# --------------------------------------------------------------------- mlflow.spark


class _SubpathArtifactRepo:
    """Scope ``download_artifacts`` on a ``models:/`` repo to a relative subpath."""

    def __init__(self, repo: Any, subpath: str) -> None:
        self._repo = repo
        self._subpath = subpath.strip("/")

    def download_artifacts(
        self, artifact_path: str = "", dst_path: str | None = None
    ) -> str:
        rel = artifact_path.strip("/")
        target = f"{self._subpath}/{rel}" if rel else self._subpath
        return cast(str, self._repo.download_artifacts(target, dst_path=dst_path))


def _resolve_spark_artifact_target(model_uri: str) -> tuple[str, Any] | None:
    """Resolve ``runs:/``, ``models:/``, or ``gs://`` URI to ``(gcs_uri, repo)``."""
    from google_cloud_managed_mlflow.cab_credentials import has_gs_scheme

    if not (model_uri.startswith(("runs:/", "models:/")) or has_gs_scheme(model_uri)):
        return None

    from mlflow.store.artifact.artifact_repository_registry import (
        get_artifact_repository,
    )

    if model_uri.startswith("runs:/"):
        from mlflow.store.artifact.runs_artifact_repo import RunsArtifactRepository

        gcs_uri = str(RunsArtifactRepository.get_underlying_uri(model_uri)).rstrip("/")
        if not has_gs_scheme(gcs_uri):
            return None
        repo = get_artifact_repository(gcs_uri)
        return (gcs_uri, repo)

    if model_uri.startswith("models:/"):
        from mlflow.tracking.artifact_utils import _get_root_uri_and_artifact_path

        from google_cloud_managed_mlflow.cab_credentials import validate_artifact_path

        root_uri, subpath = _get_root_uri_and_artifact_path(model_uri)
        models_repo = get_artifact_repository(root_uri)
        inner = getattr(models_repo, "repo", None)
        underlying = str(getattr(inner, "artifact_uri", "") or "")
        if not has_gs_scheme(underlying):
            return None
        gcs_uri = underlying.rstrip("/")
        clean_sub = (subpath or "").strip("/")
        if clean_sub:
            clean_sub = validate_artifact_path(clean_sub)
            return (
                f"{gcs_uri}/{clean_sub}",
                _SubpathArtifactRepo(models_repo, clean_sub),
            )
        return (gcs_uri, models_repo)

    gcs_uri = model_uri.rstrip("/")
    repo = get_artifact_repository(gcs_uri)
    return (gcs_uri, repo)


def _save_staged_metadata(
    module: types.ModuleType,
    spark_model: Any,
    path: str,
    bound_args: dict[str, Any],
) -> None:
    from mlflow.models import Model

    mlflow_model = bound_args.get("mlflow_model")
    if mlflow_model is None:
        mlflow_model = Model()
    metadata = bound_args.get("metadata")
    if metadata is not None:
        mlflow_model.metadata = metadata

    signature = bound_args.get("signature")
    input_example = bound_args.get("input_example")
    if signature is None and input_example is not None:
        try:
            from mlflow.models.utils import _Example

            input_ex = _Example(input_example).inference_data
            spark = runtime.active_session()
            if spark is not None:
                input_example_spark_df = spark.createDataFrame(input_ex)
                model_for_sig = spark_model
                try:
                    model_for_sig = spark_model.copy()
                except Exception:
                    _logger.debug(
                        "Failed to copy the model, using the original model.",
                        exc_info=True,
                    )
                infer_sig = getattr(module, "_infer_spark_model_signature", None)
                if callable(infer_sig):
                    signature = infer_sig(model_for_sig, input_example_spark_df)
                else:
                    import mlflow.pyspark.ml

                    signature = mlflow.pyspark.ml._infer_spark_model_signature(
                        model_for_sig, input_example_spark_df
                    )
        except Exception as exc:
            _logger.warning(
                "Failed to infer Spark model signature from input_example: %r", exc
            )
            _logger.debug("", exc_info=True)
    elif signature is False:
        signature = None

    module._save_model_metadata(
        dst_dir=path,
        spark_model=spark_model,
        mlflow_model=mlflow_model,
        conda_env=bound_args.get("conda_env"),
        code_paths=bound_args.get("code_paths"),
        signature=signature,
        input_example=input_example,
        pip_requirements=bound_args.get("pip_requirements"),
        extra_pip_requirements=bound_args.get("extra_pip_requirements"),
    )


def _load_via_hadoop_dfs(
    module: types.ModuleType,
    loader: Callable[[str], Any],
    local_model_dir: str,
    eff_dfs_tmp: str | None,
) -> Any:
    dfs_target = module.generate_tmp_dfs_path(eff_dfs_tmp or "/tmp/mlflow")
    module._HadoopFileSystem.copy_from_local_file(
        local_model_dir, dfs_target, remove_src=False
    )
    try:
        return loader(dfs_target)
    finally:
        try:
            module._HadoopFileSystem.delete(dfs_target)
        except Exception:
            _logger.debug(
                "Failed to clean up DFS staging path %s", dfs_target, exc_info=True
            )


def _load_spark_model_impl(
    module: types.ModuleType,
    original_load: Callable[..., Any],
    model_uri: str,
    dfs_tmpdir: str | None = None,
    dst_path: str | None = None,
) -> Any:
    from google_cloud_managed_mlflow.cab_credentials import (
        has_gs_scheme,
        validate_artifact_path,
    )
    from google_cloud_managed_mlflow.uri import resolve_managed_uris

    eff_tracking, eff_registry, managed = resolve_managed_uris()
    if not managed:
        return original_load(model_uri, dfs_tmpdir=dfs_tmpdir, dst_path=dst_path)
    _ensure_managed_registrations(eff_tracking, eff_registry)

    resolved = _resolve_spark_artifact_target(str(model_uri))
    if resolved is None:
        return original_load(model_uri, dfs_tmpdir=dfs_tmpdir, dst_path=dst_path)

    from mlflow.exceptions import MlflowException
    from mlflow.models import Model
    from mlflow.protos.databricks_pb2 import (
        PERMISSION_DENIED,
        RESOURCE_DOES_NOT_EXIST,
    )

    gcs_uri, repo = resolved
    with tempfile.TemporaryDirectory(prefix="mlflow-spark-mlmodel-") as meta_tmp:
        mlmodel_file = repo.download_artifacts("MLmodel", dst_path=meta_tmp)
        meta = Model.load(mlmodel_file)

    if "spark" not in meta.flavors:
        raise MlflowException(
            'Model does not have the "spark" flavor',
            error_code=RESOURCE_DOES_NOT_EXIST,
        )

    flavor = meta.flavors["spark"]
    model_data = validate_artifact_path(
        str(flavor.get("model_data") or SPARK_MODEL_SUBDIR).strip("/")
    )
    model_class = str(flavor.get("model_class") or "pyspark.ml.pipeline.PipelineModel")
    if model_class.startswith("pyspark.ml.connect."):
        return original_load(model_uri, dfs_tmpdir=dfs_tmpdir, dst_path=dst_path)

    downloaded_root: str | None = None
    if dst_path is not None:
        downloaded_root = repo.download_artifacts("", dst_path=dst_path)

    raw_code_subpath = flavor.get("code")
    if raw_code_subpath:
        code_subpath = validate_artifact_path(str(raw_code_subpath).strip("/"))
        if downloaded_root is not None:
            code_root = downloaded_root
        else:
            # Keep code_root until process exit because sys.path custom stage
            # modules may be imported lazily after load_model() returns.
            code_root = tempfile.mkdtemp(prefix="mlflow-spark-code-")
            atexit.register(shutil.rmtree, code_root, True)
            repo.download_artifacts(code_subpath, dst_path=code_root)
        add_code = getattr(module, "_add_code_from_conf_to_system_path", None)
        if callable(add_code):
            add_code(code_root, flavor)

    spark = runtime.require_active_session()
    loader = _load_spark_class(model_class)
    eff_dfs_tmp = dfs_tmpdir or os.environ.get("MLFLOW_DFS_TMP")

    direct_exc: BaseException | None = None
    if dfs_tmpdir is None:
        try:
            return loader(f"{gcs_uri}/{model_data}")
        except _spark_errors() as exc:
            if isinstance(exc, MlflowException):
                raise
            direct_exc = exc
            _logger.debug(
                "Direct Spark load from %s/%s failed; trying staged fallback.",
                gcs_uri,
                model_data,
                exc_info=True,
            )

    if has_gs_scheme(eff_dfs_tmp):
        with runtime.gcs_scratch_dir(str(eff_dfs_tmp), model_data) as staged_gs:
            if downloaded_root is not None:
                runtime.gcs_upload_directory(
                    os.path.join(downloaded_root, model_data), staged_gs
                )
            else:
                with tempfile.TemporaryDirectory(prefix="mlflow-spark-stage-") as tmp:
                    local_model_dir = repo.download_artifacts(model_data, dst_path=tmp)
                    runtime.gcs_upload_directory(local_model_dir, staged_gs)
            return loader(staged_gs)

    topology = runtime.classify_topology(spark)
    if topology is runtime.SparkTopology.LOCAL:
        if downloaded_root is not None:
            local_model_dir = os.path.join(downloaded_root, model_data)
        else:
            # Keep local_root until process exit because PySpark readers may lazily
            # access staged Parquet metadata/files after load() returns.
            local_root = tempfile.mkdtemp(prefix="mlflow-spark-local-")
            atexit.register(shutil.rmtree, local_root, True)
            local_model_dir = repo.download_artifacts(model_data, dst_path=local_root)
        return loader(f"file://{os.path.abspath(local_model_dir)}")

    if topology is runtime.SparkTopology.DISTRIBUTED_SHARED_DFS:
        if downloaded_root is not None:
            return _load_via_hadoop_dfs(
                module,
                loader,
                os.path.join(downloaded_root, model_data),
                eff_dfs_tmp,
            )
        with tempfile.TemporaryDirectory(prefix="mlflow-spark-dfs-") as tmp:
            local_model_dir = repo.download_artifacts(model_data, dst_path=tmp)
            return _load_via_hadoop_dfs(module, loader, local_model_dir, eff_dfs_tmp)

    detail = f" ({direct_exc})" if direct_exc is not None else ""
    raise MlflowException(
        f"Direct Spark load from {gcs_uri}/{model_data} failed{detail}. "
        "Distributed Spark sessions without a shared DFS require either "
        "direct GCS read access for the Spark workload identity or "
        "dfs_tmpdir='gs://<scratch-bucket>/mlflow-tmp' (or MLFLOW_DFS_TMP).",
        error_code=PERMISSION_DENIED,
    ) from direct_exc


def _maybe_save_spark_model_impl(
    original_maybe_save: Callable[..., Any],
    spark_model: Any,
    model_dir: str,
) -> bool:
    from google_cloud_managed_mlflow.cab_credentials import has_gs_scheme

    if not ensure_managed_context() or not has_gs_scheme(model_dir):
        return bool(original_maybe_save(spark_model, model_dir))

    from mlflow.exceptions import MlflowException
    from mlflow.protos.databricks_pb2 import PERMISSION_DENIED
    from mlflow.store.artifact.artifact_repository_registry import (
        get_artifact_repository,
    )

    spark = runtime.require_active_session()
    target_dir = str(model_dir).rstrip("/")
    try:
        _save_spark_model(spark_model, f"{target_dir}/{SPARK_MODEL_SUBDIR}")
        return True
    except _spark_errors() as exc:
        if isinstance(exc, MlflowException):
            raise
        eff_dfs_tmp = getattr(_LOCAL, "log_model_dfs_tmpdir", None) or os.environ.get(
            "MLFLOW_DFS_TMP"
        )
        if has_gs_scheme(eff_dfs_tmp):
            with tempfile.TemporaryDirectory(prefix="mlflow-spark-save-") as tmp:
                local_sparkml = os.path.join(tmp, SPARK_MODEL_SUBDIR)
                _save_spark_model_via_gcs_scratch(
                    spark_model, str(eff_dfs_tmp), local_sparkml
                )
                get_artifact_repository(target_dir).log_artifacts(
                    local_sparkml, artifact_path=SPARK_MODEL_SUBDIR
                )
            return True

        if not runtime.is_distributed_without_shared_dfs(spark):
            return False

        raise MlflowException(
            f"Direct Spark save to {target_dir}/{SPARK_MODEL_SUBDIR} failed "
            f"({exc}). Refusing local /tmp staging on a distributed Spark "
            "session without shared DFS to prevent silent executor weight data "
            "loss. Grant GCS write access to the Spark workload identity or "
            "set dfs_tmpdir='gs://<scratch-bucket>/mlflow-tmp' (or "
            "MLFLOW_DFS_TMP).",
            error_code=PERMISSION_DENIED,
        ) from exc


def _save_spark_model_impl(
    module: types.ModuleType,
    original_save: Callable[..., Any],
    save_signature: inspect.Signature,
    spark_model: Any,
    path: str,
    *args: Any,
    **kwargs: Any,
) -> Any:
    if not ensure_managed_context():
        return original_save(spark_model, path, *args, **kwargs)

    is_connect_model_fn = getattr(module, "_is_spark_connect_model", lambda _m: False)
    if is_connect_model_fn(spark_model):
        return original_save(spark_model, path, *args, **kwargs)

    from google_cloud_managed_mlflow.cab_credentials import has_gs_scheme

    bound = save_signature.bind_partial(spark_model, path, *args, **kwargs)
    eff_dfs_tmp = bound.arguments.get("dfs_tmpdir") or os.environ.get("MLFLOW_DFS_TMP")
    eff_tmp_str = str(eff_dfs_tmp or "")

    spark = runtime.active_session()
    runtime.validate_save_dfs_tmpdir(spark, eff_dfs_tmp)

    # Stock mlflow.spark.save_model uses _HadoopFileSystem._jvm / copy_to_local_file,
    # which crashes on Spark Connect (no JVM) and throws Wrong FS for gs:// dfs_tmpdir
    # when fs.defaultFS=file:///. Stage via Python GCS helpers whenever gs:// is used.
    if spark is not None and has_gs_scheme(eff_tmp_str):
        bound.apply_defaults()
        validate_model = getattr(module, "_validate_model", None)
        if callable(validate_model):
            validate_model(spark_model)
        validate_env = getattr(module, "_validate_env_arguments", None)
        if callable(validate_env):
            validate_env(
                bound.arguments.get("conda_env"),
                bound.arguments.get("pip_requirements"),
                bound.arguments.get("extra_pip_requirements"),
            )

        try:
            from pyspark.ml import PipelineModel

            if not isinstance(spark_model, PipelineModel):
                spark_model = PipelineModel([spark_model])
        except ImportError:
            pass

        sparkml_data_path = os.path.abspath(os.path.join(path, SPARK_MODEL_SUBDIR))
        _save_spark_model_via_gcs_scratch(spark_model, eff_tmp_str, sparkml_data_path)
        _save_staged_metadata(module, spark_model, path, dict(bound.arguments))
        return None

    return original_save(spark_model, path, *args, **kwargs)


def patch_mlflow_spark(module: types.ModuleType) -> None:
    """Install hooks on ``mlflow.spark`` (idempotent)."""
    original_load = getattr(module, "load_model", None)
    if callable(original_load) and not _is_patched(original_load):

        @functools.wraps(original_load)
        def load_model(
            model_uri: str,
            dfs_tmpdir: str | None = None,
            dst_path: str | None = None,
        ) -> Any:
            return _load_spark_model_impl(
                module,
                original_load,
                model_uri,
                dfs_tmpdir=dfs_tmpdir,
                dst_path=dst_path,
            )

        _patch_attr(module, "load_model", load_model, original_load)

    original_log = getattr(module, "log_model", None)
    if callable(original_log) and not _is_patched(original_log):
        log_signature = inspect.signature(original_log)

        @functools.wraps(original_log)
        def log_model(*args: Any, **kwargs: Any) -> Any:
            if not ensure_managed_context():
                return original_log(*args, **kwargs)
            dfs_tmpdir = kwargs.get("dfs_tmpdir")
            if "dfs_tmpdir" in log_signature.parameters:
                try:
                    bound = log_signature.bind_partial(*args, **kwargs)
                    dfs_tmpdir = bound.arguments.get("dfs_tmpdir")
                except TypeError:
                    pass
            prev_dfs_tmpdir = getattr(_LOCAL, "log_model_dfs_tmpdir", None)
            _LOCAL.log_model_dfs_tmpdir = dfs_tmpdir
            try:
                return original_log(*args, **kwargs)
            finally:
                _LOCAL.log_model_dfs_tmpdir = prev_dfs_tmpdir

        _patch_attr(module, "log_model", log_model, original_log)

    original_maybe_save = getattr(module, "_maybe_save_model", None)
    if callable(original_maybe_save) and not _is_patched(original_maybe_save):

        @functools.wraps(original_maybe_save)
        def _maybe_save_model(spark_model: Any, model_dir: str) -> bool:
            return _maybe_save_spark_model_impl(
                original_maybe_save, spark_model, model_dir
            )

        _patch_attr(module, "_maybe_save_model", _maybe_save_model, original_maybe_save)

    original_save = getattr(module, "save_model", None)
    if callable(original_save) and not _is_patched(original_save):
        save_signature = inspect.signature(original_save)

        @functools.wraps(original_save)
        def save_model(spark_model: Any, path: str, *args: Any, **kwargs: Any) -> Any:
            return _save_spark_model_impl(
                module,
                original_save,
                save_signature,
                spark_model,
                path,
                *args,
                **kwargs,
            )

        _patch_attr(module, "save_model", save_model, original_save)
