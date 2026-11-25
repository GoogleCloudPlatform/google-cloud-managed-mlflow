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

"""Tests for Google Cloud Managed MLflow Spark, PySpark ML, and PyFunc hooks."""

from __future__ import annotations

import concurrent.futures
import importlib.machinery
import os
import subprocess
import sys
import threading
import types
import unittest.mock
from pathlib import Path
from typing import Any, cast

import mlflow
import mlflow.pyfunc
import mlflow.pyspark.ml
import mlflow.spark
import pytest
from mlflow.exceptions import MlflowException
from mlflow.models import Model
from mlflow.protos.databricks_pb2 import (
    INTERNAL_ERROR,
    INVALID_PARAMETER_VALUE,
    PERMISSION_DENIED,
    RESOURCE_DOES_NOT_EXIST,
)

from google_cloud_managed_mlflow.spark import hooks, install_spark_hooks, runtime

_GCP_TRACKING_URI = (
    "https://mlflow.googleapis.com/v1/projects/my-project"
    "/locations/us-central1/namespaces/my-namespace"
)
_GCP_REGISTRY_URI = (
    "https://mlflow.googleapis.com/v1/projects/my-project"
    "/locations/us-central1/registries/my-registry"
)
_REPO_ROOT = str(Path(__file__).resolve().parent.parent)


@pytest.fixture(autouse=True)
def _install_and_isolate(monkeypatch: pytest.MonkeyPatch) -> None:
    install_spark_hooks()
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    monkeypatch.delenv("MLFLOW_REGISTRY_URI", raising=False)
    monkeypatch.delenv("MLFLOW_TRACKING_AUTH", raising=False)
    monkeypatch.delenv("MLFLOW_GCP_ENDPOINT_OVERRIDE", raising=False)
    monkeypatch.delenv("MLFLOW_DFS_TMP", raising=False)

    import mlflow.tracking._model_registry.utils as mr_utils
    import mlflow.tracking._tracking_service.utils as tr_utils

    monkeypatch.setattr(tr_utils, "_tracking_uri", None, raising=False)
    monkeypatch.setattr(mr_utils, "_registry_uri", None, raising=False)


def _set_gcp_tracking(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MLFLOW_TRACKING_URI", _GCP_TRACKING_URI)


def _set_gcp_registry(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MLFLOW_REGISTRY_URI", _GCP_REGISTRY_URI)


def _make_spark_error(
    monkeypatch: pytest.MonkeyPatch, msg: str = "403 Forbidden"
) -> BaseException:
    try:
        from pyspark.errors import PySparkException

        err_cls: Any = type("_TestSparkError", (PySparkException,), {})
        return cast(BaseException, err_cls(msg))
    except ImportError:
        pass
    try:
        from py4j.protocol import Py4JError

        return cast(BaseException, Py4JError(msg))
    except ImportError:
        pass

    fake_py4j = types.ModuleType("py4j")
    fake_protocol = types.ModuleType("py4j.protocol")

    class _FakePy4JError(Exception):
        pass

    fake_protocol.Py4JError = _FakePy4JError  # type: ignore[attr-defined]
    fake_py4j.protocol = fake_protocol  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "py4j", fake_py4j)
    monkeypatch.setitem(sys.modules, "py4j.protocol", fake_protocol)
    return _FakePy4JError(msg)


def _make_pyspark_exception(
    monkeypatch: pytest.MonkeyPatch, msg: str = "403 Forbidden"
) -> BaseException:
    try:
        from pyspark.errors import PySparkException

        err_cls: Any = type("_TestPySparkError", (PySparkException,), {})
        return cast(BaseException, err_cls(msg))
    except ImportError:
        fake_pyspark = types.ModuleType("pyspark")
        fake_errors = types.ModuleType("pyspark.errors")

        class _FakePySparkException(Exception):
            pass

        fake_errors.PySparkException = _FakePySparkException  # type: ignore[attr-defined]
        fake_pyspark.errors = fake_errors  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "pyspark", fake_pyspark)
        monkeypatch.setitem(sys.modules, "pyspark.errors", fake_errors)
        return _FakePySparkException(msg)


class _ConfStub:
    def __init__(self, values: dict[str, Any] | None = None) -> None:
        self._values = dict(values or {})

    def get(self, key: str, default: Any = None) -> Any:
        return self._values.get(key, default)


class _HadoopConfStub:
    def __init__(self, default_fs: str = "file:///") -> None:
        self._default_fs = default_fs

    def get(self, key: str, default: Any = None) -> Any:
        if key == "fs.defaultFS":
            return self._default_fs
        return default


class _JscStub:
    def __init__(self, default_fs: str = "file:///") -> None:
        self._hadoop_conf = _HadoopConfStub(default_fs)

    def hadoopConfiguration(self) -> _HadoopConfStub:
        return self._hadoop_conf


class _SparkContextStub:
    def __init__(self, master: str = "local[*]", default_fs: str = "file:///") -> None:
        self.master = master
        self._jsc = _JscStub(default_fs)


class _ClassicSparkStub:
    def __init__(
        self,
        master: str = "local[*]",
        default_fs: str = "file:///",
        conf: dict[str, Any] | None = None,
    ) -> None:
        merged_conf = dict(conf or {})
        merged_conf.setdefault("spark.master", master)
        merged_conf.setdefault("spark.hadoop.fs.defaultFS", default_fs)
        self.conf = _ConfStub(merged_conf)
        self.sparkContext = _SparkContextStub(master=master, default_fs=default_fs)


def _make_connect_spark_stub(conf: dict[str, Any] | None = None) -> Any:
    """Construct a Spark Connect session stub without a sparkContext."""

    class SparkSession:
        def __init__(self) -> None:
            self.conf = _ConfStub(conf)

    SparkSession.__module__ = "pyspark.sql.connect.session"
    return SparkSession()


def _install_fake_pyspark_ml(
    monkeypatch: pytest.MonkeyPatch,
    *,
    pipeline_mod: types.ModuleType | None = None,
    base_mod: types.ModuleType | None = None,
    eval_mod: types.ModuleType | None = None,
    pipeline_cls: Any = None,
) -> tuple[types.ModuleType, types.ModuleType]:
    """Install fake ``pyspark.ml`` submodules while keeping ``pyspark.ml`` synced."""
    fake_pyspark = sys.modules.get("pyspark") or types.ModuleType("pyspark")
    fake_ml = types.ModuleType("pyspark.ml")
    if pipeline_mod is not None:
        monkeypatch.setattr(fake_ml, "pipeline", pipeline_mod, raising=False)
        monkeypatch.setitem(sys.modules, "pyspark.ml.pipeline", pipeline_mod)
    if base_mod is not None:
        monkeypatch.setattr(fake_ml, "base", base_mod, raising=False)
        monkeypatch.setitem(sys.modules, "pyspark.ml.base", base_mod)
    if eval_mod is not None:
        monkeypatch.setattr(fake_ml, "evaluation", eval_mod, raising=False)
        monkeypatch.setitem(sys.modules, "pyspark.ml.evaluation", eval_mod)
    if pipeline_cls is not None:
        monkeypatch.setattr(fake_ml, "PipelineModel", pipeline_cls, raising=False)
    monkeypatch.setattr(fake_pyspark, "ml", fake_ml, raising=False)
    monkeypatch.setitem(sys.modules, "pyspark", fake_pyspark)
    monkeypatch.setitem(sys.modules, "pyspark.ml", fake_ml)
    return fake_pyspark, fake_ml


class _FakePipelineModelModule:
    """Helper to register a fake pyspark.ml.pipeline.PipelineModel loader."""

    @staticmethod
    def install(
        monkeypatch: pytest.MonkeyPatch, load_fn: Any, *, wrap_cls: bool = False
    ) -> None:
        fake_pipeline = types.ModuleType("pyspark.ml.pipeline")

        class PipelineModel:
            def __init__(self, stages: Any = None) -> None:
                self.stages = stages or []

            @classmethod
            def load(cls, path: str) -> Any:
                return load_fn(path)

        fake_pipeline.PipelineModel = PipelineModel  # type: ignore[attr-defined]
        _install_fake_pyspark_ml(
            monkeypatch,
            pipeline_mod=fake_pipeline,
            pipeline_cls=PipelineModel if wrap_cls else None,
        )


# ---------------------------------------------------------------------------
# 1. Cold-Import Isolation
# ---------------------------------------------------------------------------


def test_cold_import_isolation() -> None:
    cmd = [
        sys.executable,
        "-c",
        (
            "import sys; "
            "import google_cloud_managed_mlflow; "
            "from google_cloud_managed_mlflow.spark import install_spark_hooks; "
            "install_spark_hooks(); "
            "assert 'mlflow' not in sys.modules, 'mlflow was eagerly imported'; "
            "assert 'pyspark' not in sys.modules, 'pyspark was eagerly imported'; "
            "print('OK')"
        ),
    ]
    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        check=True,
        env={**os.environ, "PYTHONPATH": _REPO_ROOT},
    )
    assert proc.stdout.strip() == "OK"


# ---------------------------------------------------------------------------
# 2. Non-GCP MLflow URI Coexistence
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("tracking_uri", ["http://localhost:5000", "sqlite:///tmp.db"])
def test_non_gcp_coexistence_delegates_unchanged(
    monkeypatch: pytest.MonkeyPatch, tracking_uri: str, tmp_path: Path
) -> None:
    monkeypatch.setenv("MLFLOW_TRACKING_URI", tracking_uri)
    assert not hooks.is_managed_context()

    # Real patched modules are already marked
    assert getattr(mlflow.pyfunc.spark_udf, hooks.PATCH_MARKER, False)
    assert getattr(
        mlflow.pyspark.ml._read_log_model_allowlist, hooks.PATCH_MARKER, False
    )
    assert getattr(mlflow.pyspark.ml.autolog, hooks.PATCH_MARKER, False)
    assert getattr(mlflow.spark.load_model, hooks.PATCH_MARKER, False)
    assert getattr(mlflow.spark.log_model, hooks.PATCH_MARKER, False)
    assert getattr(mlflow.spark._maybe_save_model, hooks.PATCH_MARKER, False)
    assert getattr(mlflow.spark.save_model, hooks.PATCH_MARKER, False)

    # spark_udf
    udf_sentinel = object()
    called: list[tuple[Any, Any]] = []

    def fake_udf(
        spark: Any,
        model_uri: Any,
        result_type: Any = None,
        env_manager: Any = None,
        params: Any = None,
        extra_env: Any = None,
        prebuilt_env_uri: Any = None,
        model_config: Any = None,
    ) -> Any:
        called.append((model_uri, extra_env))
        return udf_sentinel

    fake_pyfunc = types.ModuleType("mlflow.pyfunc")
    fake_pyfunc.spark_udf = fake_udf  # type: ignore[attr-defined]
    hooks.patch_pyfunc(fake_pyfunc)
    assert (
        fake_pyfunc.spark_udf(None, "models:/m/1", extra_env={"K": "V"}) is udf_sentinel
    )
    assert called == [("models:/m/1", {"K": "V"})]

    # pyspark.ml._read_log_model_allowlist & autolog
    fake_pyspark_ml = types.ModuleType("mlflow.pyspark.ml")
    allowlist_sentinel = {"pyspark.ml.classification.LogisticRegressionModel"}
    autolog_calls: list[dict[str, Any]] = []

    def fake_allowlist() -> set[str]:
        return allowlist_sentinel

    def fake_autolog(disable: bool = False, **kwargs: Any) -> str:
        autolog_calls.append({"disable": disable, **kwargs})
        return "orig-autolog"

    fake_pyspark_ml._read_log_model_allowlist = fake_allowlist  # type: ignore[attr-defined]
    fake_pyspark_ml.autolog = fake_autolog  # type: ignore[attr-defined]
    hooks.patch_pyspark_ml(fake_pyspark_ml)

    assert fake_pyspark_ml._read_log_model_allowlist() is allowlist_sentinel
    assert fake_pyspark_ml.autolog(disable=False) == "orig-autolog"
    assert autolog_calls == [{"disable": False}]

    # mlflow.spark load_model, log_model, _maybe_save_model, save_model
    fake_spark_mod = types.ModuleType("mlflow.spark")
    load_calls: list[Any] = []
    log_calls: list[Any] = []
    maybe_calls: list[Any] = []
    save_calls: list[Any] = []

    def fake_load(
        model_uri: str, dfs_tmpdir: str | None = None, dst_path: str | None = None
    ) -> str:
        load_calls.append((model_uri, dfs_tmpdir, dst_path))
        return "orig-loaded"

    def fake_log(
        spark_model: Any,
        artifact_path: str,
        dfs_tmpdir: str | None = None,
        **kwargs: Any,
    ) -> str:
        log_calls.append((spark_model, artifact_path, dfs_tmpdir))
        return "orig-logged"

    def fake_maybe(spark_model: Any, model_dir: str) -> bool:
        maybe_calls.append((spark_model, model_dir))
        return False

    def fake_save(
        spark_model: Any,
        path: str,
        dfs_tmpdir: str | None = None,
        **kwargs: Any,
    ) -> str:
        save_calls.append((spark_model, path, dfs_tmpdir))
        return "orig-saved"

    fake_spark_mod.load_model = fake_load  # type: ignore[attr-defined]
    fake_spark_mod.log_model = fake_log  # type: ignore[attr-defined]
    fake_spark_mod._maybe_save_model = fake_maybe  # type: ignore[attr-defined]
    fake_spark_mod.save_model = fake_save  # type: ignore[attr-defined]
    hooks.patch_mlflow_spark(fake_spark_mod)

    assert fake_spark_mod.load_model("runs:/r1/m", dfs_tmpdir="/tmp/d") == "orig-loaded"
    assert fake_spark_mod.log_model("m", "art", dfs_tmpdir="/tmp/d") == "orig-logged"
    assert fake_spark_mod._maybe_save_model("m", "gs://my-bucket/p") is False
    assert (
        fake_spark_mod.save_model("m", str(tmp_path), dfs_tmpdir="/tmp/d")
        == "orig-saved"
    )
    assert load_calls == [("runs:/r1/m", "/tmp/d", None)]
    assert log_calls == [("m", "art", "/tmp/d")]
    assert maybe_calls == [("m", "gs://my-bucket/p")]
    assert save_calls == [("m", str(tmp_path), "/tmp/d")]


# ---------------------------------------------------------------------------
# 3. Bootstrap Before import mlflow.pyspark.ml
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "bootstrap_stmt",
    [
        (
            "from google_cloud_managed_mlflow.tracking_store import "
            "build_tracking_store; "
            f"build_tracking_store('{_GCP_TRACKING_URI}')"
        ),
        (
            "from google_cloud_managed_mlflow.auth import "
            "GoogleCloudAuthProvider; "
            "GoogleCloudAuthProvider().get_auth()"
        ),
        (
            "from google_cloud_managed_mlflow.tracking_store import "
            "ensure_tracking_store_registered; "
            "ensure_tracking_store_registered()"
        ),
        (
            "from google_cloud_managed_mlflow.registry_store import "
            "ensure_registry_store_registered; "
            "ensure_registry_store_registered()"
        ),
        (
            "from google_cloud_managed_mlflow.registry_store import "
            "build_registry_store; "
            f"build_registry_store('{_GCP_REGISTRY_URI}')"
        ),
        (
            "import os; "
            f"os.environ['MLFLOW_TRACKING_URI'] = '{_GCP_TRACKING_URI}'; "
            "from google_cloud_managed_mlflow.gcs_artifact_repo import "
            "GcpGCSArtifactRepository; "
            "GcpGCSArtifactRepository('gs://my-bucket/artifacts')"
        ),
    ],
)
def test_bootstrap_before_import_patches_post_import(bootstrap_stmt: str) -> None:
    script = (
        "import sys; "
        f"{bootstrap_stmt}; "
        "assert 'mlflow.pyspark.ml' not in sys.modules; "
        "import mlflow.pyspark.ml, mlflow.spark, mlflow.pyfunc; "
        "assert getattr(mlflow.pyspark.ml._read_log_model_allowlist, "
        "'__mlflow_gcp_spark_patched__', False); "
        "assert getattr(mlflow.pyspark.ml.autolog, "
        "'__mlflow_gcp_spark_patched__', False); "
        "assert getattr(mlflow.spark.load_model, "
        "'__mlflow_gcp_spark_patched__', False); "
        "assert getattr(mlflow.spark.log_model, "
        "'__mlflow_gcp_spark_patched__', False); "
        "assert getattr(mlflow.spark._maybe_save_model, "
        "'__mlflow_gcp_spark_patched__', False); "
        "assert getattr(mlflow.spark.save_model, "
        "'__mlflow_gcp_spark_patched__', False); "
        "assert getattr(mlflow.pyfunc.spark_udf, "
        "'__mlflow_gcp_spark_patched__', False); "
        "print('OK')"
    )
    proc = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        check=True,
        env={**os.environ, "PYTHONPATH": _REPO_ROOT},
    )
    assert proc.stdout.strip() == "OK"


def test_models_artifact_repo_installs_spark_hooks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from google_cloud_managed_mlflow.models_artifact_repo import (
        GcpModelsArtifactRepository,
    )

    _set_gcp_tracking(monkeypatch)
    fake_client = unittest.mock.MagicMock()
    fake_client.get_logged_model.return_value.artifact_location = (
        "gs://my-bucket/models/m-123/artifacts"
    )
    with (
        unittest.mock.patch(
            "google_cloud_managed_mlflow.spark.install_spark_hooks"
        ) as mock_install,
        unittest.mock.patch("mlflow.MlflowClient", return_value=fake_client),
    ):
        repo = GcpModelsArtifactRepository("models:/m-123")
        assert repo.is_logged_model_uri is True
        assert mock_install.call_count >= 1


# ---------------------------------------------------------------------------
# 4. patch_pyfunc: spark_udf
# ---------------------------------------------------------------------------


def test_pyfunc_spark_udf_injects_registry_env_kw_and_positional(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_gcp_tracking(monkeypatch)
    _set_gcp_registry(monkeypatch)
    monkeypatch.setenv("MLFLOW_GCP_ENDPOINT_OVERRIDE", "https://127.0.0.1:8443")

    captured: list[dict[str, str] | None] = []

    def fake_udf(
        spark: Any,
        model_uri: Any,
        result_type: Any = None,
        env_manager: Any = None,
        params: dict[str, Any] | None = None,
        extra_env: dict[str, str] | None = None,
        prebuilt_env_uri: str | None = None,
        model_config: Any = None,
    ) -> str:
        captured.append(extra_env)
        return "udf-ok"

    fake_mod = types.ModuleType("mlflow.pyfunc")
    fake_mod.spark_udf = fake_udf  # type: ignore[attr-defined]
    hooks.patch_pyfunc(fake_mod)

    # Keyword extra_env preserves existing keys and adds MLFLOW_REGISTRY_URI + override
    user_env = {"CUSTOM_VAR": "123"}
    assert (
        fake_mod.spark_udf(None, "models:/my-model@champion", extra_env=user_env)
        == "udf-ok"
    )
    assert user_env == {"CUSTOM_VAR": "123"}  # caller dict not mutated
    assert captured[-1] == {
        "CUSTOM_VAR": "123",
        "MLFLOW_REGISTRY_URI": _GCP_REGISTRY_URI,
        "MLFLOW_GCP_ENDPOINT_OVERRIDE": "https://127.0.0.1:8443",
    }

    # Positional extra_env does not overwrite user-specified MLFLOW_REGISTRY_URI
    custom_reg = (
        "https://mlflow.googleapis.com/v1/projects/other-project"
        "/locations/us-central1/registries/other-reg"
    )
    fake_mod.spark_udf(
        None,
        "models:/my-model@champion",
        "double",
        None,
        None,
        {"MLFLOW_REGISTRY_URI": custom_reg, "FOO": "bar"},
    )
    assert captured[-1] == {
        "MLFLOW_REGISTRY_URI": custom_reg,
        "FOO": "bar",
        "MLFLOW_GCP_ENDPOINT_OVERRIDE": "https://127.0.0.1:8443",
    }

    # No extra_env passed
    fake_mod.spark_udf(None, "models:/my-model@champion")
    assert captured[-1] == {
        "MLFLOW_REGISTRY_URI": _GCP_REGISTRY_URI,
        "MLFLOW_GCP_ENDPOINT_OVERRIDE": "https://127.0.0.1:8443",
    }


def test_pyfunc_spark_udf_skips_injection_without_gcp_registry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_gcp_tracking(monkeypatch)
    # No MLFLOW_REGISTRY_URI -> resolved registry URI is the namespace tracking URI
    captured: list[dict[str, str] | None] = []

    def fake_udf(
        spark: Any,
        model_uri: Any,
        result_type: Any = None,
        env_manager: Any = None,
        params: dict[str, Any] | None = None,
        extra_env: dict[str, str] | None = None,
    ) -> str:
        captured.append(extra_env)
        return "udf-ok"

    fake_mod = types.ModuleType("mlflow.pyfunc")
    fake_mod.spark_udf = fake_udf  # type: ignore[attr-defined]
    hooks.patch_pyfunc(fake_mod)

    fake_mod.spark_udf(None, "runs:/r1/m")
    assert captured[-1] is None


# ---------------------------------------------------------------------------
# 5. patch_pyspark_ml: allowlist & cache guard
# ---------------------------------------------------------------------------


def test_pyspark_ml_allowlist_connect_and_custom_conf(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _set_gcp_tracking(monkeypatch)
    connect_spark = _make_connect_spark_stub()

    with unittest.mock.patch.object(
        runtime, "active_session", return_value=connect_spark
    ):
        allowlist = mlflow.pyspark.ml._read_log_model_allowlist()

    assert "xgboost.spark.estimator.SparkXGBClassifierModel" in allowlist
    assert "pyspark.ml.classification.LogisticRegressionModel" in allowlist

    # Custom allowlist via spark.conf (never touches sparkContext)
    custom_file = tmp_path / "custom_allowlist.txt"
    custom_file.write_text("custom.pkg.MyEstimatorModel\n", encoding="utf-8")
    connect_with_custom = _make_connect_spark_stub(
        {"spark.mlflow.pysparkml.autolog.logModelAllowlistFile": str(custom_file)}
    )
    with unittest.mock.patch.object(
        runtime, "active_session", return_value=connect_with_custom
    ):
        custom_allowlist = mlflow.pyspark.ml._read_log_model_allowlist()

    assert "custom.pkg.MyEstimatorModel" in custom_allowlist
    assert "xgboost.spark.estimator.SparkXGBClassifierModel" in custom_allowlist


def test_pyspark_ml_autolog_cache_guard_preserves_caller_and_releases_on_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_gcp_tracking(monkeypatch)

    class _BaseEstimator:
        pass

    class _MockEstimator(_BaseEstimator):
        fit: Any = None

    unpersist_calls: list[str] = []

    class _Dataset:
        def __init__(self, cached: bool) -> None:
            self.is_cached = cached

        def unpersist(self) -> _Dataset:
            self.is_cached = False
            unpersist_calls.append("unpersist")
            return self

    # Simulate mlflow/utils/autologging_utils/safety.py filename on patched fit
    code_ns: dict[str, Any] = {}
    exec(
        compile(
            "def fit(self, dataset, should_fail=False):\n"
            "    dataset.is_cached = True\n"
            "    try:\n"
            "        if should_fail:\n"
            "            raise RuntimeError('fit boom')\n"
            "        return 'fitted'\n"
            "    finally:\n"
            "        if not should_fail:\n"
            "            dataset.unpersist()\n",
            "/tmp/mlflow/utils/autologging_utils/safety.py",
            "exec",
        ),
        code_ns,
    )
    _MockEstimator.fit = code_ns["fit"]

    fake_base = types.ModuleType("pyspark.ml.base")
    fake_base.Estimator = _BaseEstimator  # type: ignore[attr-defined]
    _install_fake_pyspark_ml(monkeypatch, base_mod=fake_base)

    hooks._guard_estimator_fits()
    est = _MockEstimator()

    # Case 1: Caller-owned cache (dataset.is_cached=True) is preserved!
    caller_df = _Dataset(cached=True)
    assert est.fit(caller_df) == "fitted"
    assert caller_df.is_cached is True
    assert unpersist_calls == []
    # Restore check: caller can still unpersist afterwards
    caller_df.unpersist()
    assert caller_df.is_cached is False
    unpersist_calls.clear()

    # Case 2: Autolog-only cache is unpersisted even when fit raises!
    uncached_df = _Dataset(cached=False)
    with pytest.raises(RuntimeError, match="fit boom"):
        est.fit(uncached_df, should_fail=True)
    assert uncached_df.is_cached is False
    assert unpersist_calls == ["unpersist"]


# ---------------------------------------------------------------------------
# 6. patch_mlflow_spark: load_model
# ---------------------------------------------------------------------------


def _write_mlmodel_file(
    dst_dir: Path,
    *,
    model_data: str = "sparkml",
    model_class: str = "pyspark.ml.pipeline.PipelineModel",
) -> Path:
    dst_dir.mkdir(parents=True, exist_ok=True)
    mlmodel_path = dst_dir / "MLmodel"
    m = Model()
    m.add_flavor("spark", model_data=model_data, model_class=model_class)
    m.save(str(mlmodel_path))
    return mlmodel_path


class _FakeArtifactRepo:
    def __init__(
        self, artifact_uri: str, model_class: str = "pyspark.ml.pipeline.PipelineModel"
    ) -> None:
        self.artifact_uri = artifact_uri
        self.model_class = model_class
        self.downloaded_paths: list[str] = []

    def download_artifacts(
        self, artifact_path: str = "", dst_path: str | None = None
    ) -> str:
        self.downloaded_paths.append(artifact_path)
        assert dst_path is not None
        root = Path(dst_path)
        if artifact_path == "MLmodel":
            return str(_write_mlmodel_file(root, model_class=self.model_class))
        if artifact_path == "":
            _write_mlmodel_file(root, model_class=self.model_class)
            data_dir = root / "sparkml"
            data_dir.mkdir(parents=True, exist_ok=True)
            (data_dir / "part-000").write_text("weights", encoding="utf-8")
            return str(root)
        target = root / artifact_path
        target.mkdir(parents=True, exist_ok=True)
        (target / "part-000").write_text("weights", encoding="utf-8")
        return str(target)


@pytest.mark.parametrize(
    ("model_uri", "expected_load_path"),
    [
        (
            "runs:/run-123/my_model",
            "gs://my-bucket/ns/runs/run-123/artifacts/my_model/sparkml",
        ),
        (
            "models:/my-model@champion",
            "gs://my-bucket/reg/models/my-model/v1/sparkml",
        ),
        ("gs://my-bucket/direct/model", "gs://my-bucket/direct/model/sparkml"),
    ],
)
@pytest.mark.parametrize("session_kind", ["connect", "batch"])
def test_load_model_direct_gcs_across_uris_and_sessions(
    monkeypatch: pytest.MonkeyPatch,
    model_uri: str,
    expected_load_path: str,
    session_kind: str,
) -> None:
    _set_gcp_tracking(monkeypatch)
    _set_gcp_registry(monkeypatch)

    spark = (
        _make_connect_spark_stub()
        if session_kind == "connect"
        else _ClassicSparkStub(master="yarn", default_fs="file:///")
    )

    run_gcs = "gs://my-bucket/ns/runs/run-123/artifacts/my_model"
    reg_gcs = "gs://my-bucket/reg/models/my-model/v1"

    def fake_get_repo(uri: str) -> Any:
        if uri == "models:/my-model@champion":
            outer = _FakeArtifactRepo(uri)
            outer.repo = _FakeArtifactRepo(reg_gcs)  # type: ignore[attr-defined]
            return outer
        return _FakeArtifactRepo(uri)

    loaded_paths: list[str] = []

    def _record_load(path: str) -> str:
        loaded_paths.append(path)
        return f"loaded:{path}"

    _FakePipelineModelModule.install(monkeypatch, _record_load)

    with (
        unittest.mock.patch.object(runtime, "active_session", return_value=spark),
        unittest.mock.patch(
            "mlflow.store.artifact.runs_artifact_repo.RunsArtifactRepository.get_underlying_uri",
            return_value=run_gcs,
        ),
        unittest.mock.patch(
            "mlflow.store.artifact.artifact_repository_registry.get_artifact_repository",
            side_effect=fake_get_repo,
        ),
        unittest.mock.patch.object(
            mlflow.spark._HadoopFileSystem,
            "_jvm",
            side_effect=AssertionError("_HadoopFileSystem._jvm must not be touched"),
        ),
    ):
        result = mlflow.spark.load_model(model_uri)

    assert result == f"loaded:{expected_load_path}"
    assert loaded_paths == [expected_load_path]


def test_load_model_403_falls_back_to_gcs_scratch_and_cleans_up(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_gcp_tracking(monkeypatch)
    spark = _make_connect_spark_stub()
    spark_err = _make_spark_error(monkeypatch, "403 Access Denied")
    gcs_uri = "gs://my-bucket/ns/runs/r1/artifacts/model"
    repo = _FakeArtifactRepo(gcs_uri)

    attempts: list[str] = []

    def fake_load(path: str) -> str:
        attempts.append(path)
        if path.startswith("gs://my-bucket/"):
            raise spark_err
        return f"loaded:{path}"

    _FakePipelineModelModule.install(monkeypatch, fake_load)

    uploaded: list[tuple[str, str]] = []
    deleted: list[str] = []

    def fake_upload(local_dir: str, gs_prefix: str) -> list[tuple[str, str]]:
        uploaded.append((local_dir, gs_prefix))
        return [("my-scratch", "tmp/sparkml/part-000")]

    with (
        unittest.mock.patch.object(runtime, "active_session", return_value=spark),
        unittest.mock.patch(
            "mlflow.store.artifact.artifact_repository_registry.get_artifact_repository",
            return_value=repo,
        ),
        unittest.mock.patch.object(
            runtime, "gcs_upload_directory", side_effect=fake_upload
        ),
        unittest.mock.patch.object(
            runtime, "gcs_delete_prefix", side_effect=deleted.append
        ),
    ):
        # First call fails direct gs:// -> falls back via MLFLOW_DFS_TMP
        monkeypatch.setenv("MLFLOW_DFS_TMP", "gs://my-scratch/tmp")
        res = mlflow.spark.load_model(gcs_uri)

    assert res.startswith("loaded:gs://my-scratch/tmp/mlflow-spark-")
    assert len(attempts) == 2
    assert len(uploaded) == 1
    assert len(deleted) == 1
    assert not deleted[0].endswith("/sparkml")
    assert uploaded[0][1] == f"{deleted[0]}/sparkml"


def test_load_model_local_master_fallback_to_file_uri(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _set_gcp_tracking(monkeypatch)
    spark = _ClassicSparkStub(master="local[*]", default_fs="file:///")
    spark_err = _make_spark_error(monkeypatch, "No FileSystem for scheme: gs")
    gcs_uri = "gs://my-bucket/ns/runs/r1/artifacts/model"
    repo = _FakeArtifactRepo(gcs_uri)

    attempts: list[str] = []

    def fake_load(path: str) -> str:
        attempts.append(path)
        if path.startswith("gs://"):
            raise spark_err
        return f"loaded:{path}"

    _FakePipelineModelModule.install(monkeypatch, fake_load)
    dst_dir = tmp_path / "out"

    with (
        unittest.mock.patch.object(runtime, "active_session", return_value=spark),
        unittest.mock.patch(
            "mlflow.store.artifact.artifact_repository_registry.get_artifact_repository",
            return_value=repo,
        ),
    ):
        res = mlflow.spark.load_model(gcs_uri, dst_path=str(dst_dir))

    assert res.startswith("loaded:file://")
    assert (dst_dir / "MLmodel").is_file()
    assert len(attempts) == 2


def test_load_model_hdfs_fallback_uses_hadoop_fs_and_cleans_up(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_gcp_tracking(monkeypatch)
    spark = _ClassicSparkStub(master="yarn", default_fs="hdfs://namenode:8020")
    spark_err = _make_spark_error(monkeypatch, "403 Forbidden")
    gcs_uri = "gs://my-bucket/ns/runs/r1/artifacts/model"
    repo = _FakeArtifactRepo(gcs_uri)

    def fake_load(path: str) -> str:
        if path.startswith("gs://"):
            raise spark_err
        return f"loaded:{path}"

    _FakePipelineModelModule.install(monkeypatch, fake_load)

    copied: list[tuple[str, str, bool]] = []
    deleted_dfs: list[str] = []

    with (
        unittest.mock.patch.object(runtime, "active_session", return_value=spark),
        unittest.mock.patch(
            "mlflow.store.artifact.artifact_repository_registry.get_artifact_repository",
            return_value=repo,
        ),
        unittest.mock.patch.object(
            mlflow.spark,
            "generate_tmp_dfs_path",
            return_value="hdfs://namenode:8020/tmp/mlflow/staged",
        ),
        unittest.mock.patch.object(
            mlflow.spark._HadoopFileSystem,
            "copy_from_local_file",
            side_effect=lambda s, d, remove_src=False: copied.append(
                (s, d, remove_src)
            ),
        ),
        unittest.mock.patch.object(
            mlflow.spark._HadoopFileSystem,
            "delete",
            side_effect=deleted_dfs.append,
        ),
    ):
        res = mlflow.spark.load_model(gcs_uri)

    assert res == "loaded:hdfs://namenode:8020/tmp/mlflow/staged"
    assert len(copied) == 1
    assert copied[0][1:] == ("hdfs://namenode:8020/tmp/mlflow/staged", False)
    assert deleted_dfs == ["hdfs://namenode:8020/tmp/mlflow/staged"]


def test_load_model_distributed_403_without_scratch_raises_permission_denied(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_gcp_tracking(monkeypatch)
    spark = _make_connect_spark_stub()
    spark_err = _make_pyspark_exception(monkeypatch, "403 Forbidden")
    gcs_uri = "gs://my-bucket/ns/runs/r1/artifacts/model"
    repo = _FakeArtifactRepo(gcs_uri)

    def fake_load(_path: str) -> Any:
        raise spark_err

    _FakePipelineModelModule.install(monkeypatch, fake_load)

    with (
        unittest.mock.patch.object(runtime, "active_session", return_value=spark),
        unittest.mock.patch(
            "mlflow.store.artifact.artifact_repository_registry.get_artifact_repository",
            return_value=repo,
        ),
        pytest.raises(MlflowException) as exc_info,
    ):
        mlflow.spark.load_model(gcs_uri)

    assert (
        exc_info.value.error_code == MlflowException("", PERMISSION_DENIED).error_code
    )
    assert "dfs_tmpdir" in str(exc_info.value)


def test_load_model_delegates_pyspark_ml_connect_class_to_original_load(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_gcp_tracking(monkeypatch)
    gcs_uri = "gs://my-bucket/ns/runs/r1/artifacts/model"
    repo = _FakeArtifactRepo(
        gcs_uri,
        model_class="pyspark.ml.connect.classification.LogisticRegressionModel",
    )
    orig_calls: list[tuple[str, str | None, str | None]] = []

    def fake_orig_load(
        model_uri: str, dfs_tmpdir: str | None = None, dst_path: str | None = None
    ) -> str:
        orig_calls.append((model_uri, dfs_tmpdir, dst_path))
        return "connect-loaded"

    fake_mod = types.ModuleType("mlflow.spark")
    fake_mod.load_model = fake_orig_load  # type: ignore[attr-defined]
    fake_mod.log_model = lambda *a, **kw: None  # type: ignore[attr-defined]
    fake_mod._maybe_save_model = lambda *a, **kw: False  # type: ignore[attr-defined]
    fake_mod.save_model = lambda *a, **kw: None  # type: ignore[attr-defined]
    hooks.patch_mlflow_spark(fake_mod)

    with unittest.mock.patch(
        "mlflow.store.artifact.artifact_repository_registry.get_artifact_repository",
        return_value=repo,
    ):
        res = fake_mod.load_model(gcs_uri, dfs_tmpdir="gs://s/t", dst_path="/tmp/d")

    assert res == "connect-loaded"
    assert orig_calls == [(gcs_uri, "gs://s/t", "/tmp/d")]


# ---------------------------------------------------------------------------
# 7. patch_mlflow_spark: log_model / _maybe_save_model / save_model
# ---------------------------------------------------------------------------


class _ModelStub:
    def __init__(self, on_save: Any) -> None:
        self._on_save = on_save

    def write(self) -> Any:
        outer = self

        class _Writer:
            def overwrite(self) -> _Writer:
                return self

            def save(self, path: str) -> None:
                outer._on_save(path)

        return _Writer()


def _make_pipeline_model_stub(on_save: Any) -> Any:
    """Return a _ModelStub that is also a PipelineModel when pyspark is installed."""
    try:
        from pyspark.ml import PipelineModel

        stub_cls: Any = type(
            "_PipelineStub",
            (PipelineModel,),
            {"write": lambda _self: _ModelStub(on_save).write()},
        )
        return stub_cls([])
    except ImportError:
        return _ModelStub(on_save)


@pytest.mark.parametrize("session_kind", ["connect", "batch"])
def test_maybe_save_model_direct_gcs_succeeds(
    monkeypatch: pytest.MonkeyPatch, session_kind: str
) -> None:
    _set_gcp_tracking(monkeypatch)
    spark = (
        _make_connect_spark_stub()
        if session_kind == "connect"
        else _ClassicSparkStub(master="yarn", default_fs="file:///")
    )
    saved_paths: list[str] = []
    model = _ModelStub(saved_paths.append)

    with unittest.mock.patch.object(runtime, "active_session", return_value=spark):
        ok = mlflow.spark._maybe_save_model(
            model, "gs://my-bucket/ns/runs/r1/artifacts/m"
        )

    assert ok is True
    assert saved_paths == ["gs://my-bucket/ns/runs/r1/artifacts/m/sparkml"]


def test_log_model_and_maybe_save_403_falls_back_to_gcs_scratch_and_cleans_up(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_gcp_tracking(monkeypatch)
    spark = _make_connect_spark_stub()
    spark_err = _make_pyspark_exception(monkeypatch, "403 Forbidden")

    save_attempts: list[str] = []

    def on_save(path: str) -> None:
        save_attempts.append(path)
        if path.startswith("gs://my-bucket/"):
            raise spark_err

    model = _ModelStub(on_save)
    logged_artifacts: list[tuple[str, str | None]] = []
    deleted_prefixes: list[str] = []

    fake_repo = unittest.mock.MagicMock()
    fake_repo.log_artifacts.side_effect = lambda local_dir, artifact_path=None: (
        logged_artifacts.append((local_dir, artifact_path))
    )

    def fake_download_prefix(gs_prefix: str, local_dir: str) -> int:
        os.makedirs(local_dir, exist_ok=True)
        Path(local_dir, "part-000").write_text("data", encoding="utf-8")
        return 1

    # Verify log_model binds dfs_tmpdir into _LOCAL and restores it
    fake_spark_mod = types.ModuleType("mlflow.spark")

    def orig_load(model_uri: str, dfs_tmpdir: Any = None, dst_path: Any = None) -> Any:
        return None

    def orig_log(
        spark_model: Any,
        artifact_path: str,
        dfs_tmpdir: str | None = None,
    ) -> str:
        assert (
            fake_spark_mod._maybe_save_model(
                spark_model, "gs://my-bucket/ns/runs/r1/artifacts/m"
            )
            is True
        )
        return "logged-ok"

    def orig_maybe(spark_model: Any, model_dir: str) -> bool:
        return False

    def orig_save(spark_model: Any, path: str, dfs_tmpdir: Any = None) -> None:
        return None

    fake_spark_mod.load_model = orig_load  # type: ignore[attr-defined]
    fake_spark_mod.log_model = orig_log  # type: ignore[attr-defined]
    fake_spark_mod._maybe_save_model = orig_maybe  # type: ignore[attr-defined]
    fake_spark_mod.save_model = orig_save  # type: ignore[attr-defined]
    hooks.patch_mlflow_spark(fake_spark_mod)

    with (
        unittest.mock.patch.object(runtime, "active_session", return_value=spark),
        unittest.mock.patch.object(
            runtime, "gcs_download_prefix", side_effect=fake_download_prefix
        ),
        unittest.mock.patch.object(
            runtime, "gcs_delete_prefix", side_effect=deleted_prefixes.append
        ),
        unittest.mock.patch(
            "mlflow.store.artifact.artifact_repository_registry.get_artifact_repository",
            return_value=fake_repo,
        ),
    ):
        res = fake_spark_mod.log_model(model, "m", dfs_tmpdir="gs://my-scratch/tmp")

    assert res == "logged-ok"
    assert getattr(hooks._LOCAL, "log_model_dfs_tmpdir", None) is None
    assert len(save_attempts) == 2
    assert save_attempts[1].startswith("gs://my-scratch/tmp/mlflow-spark-")
    assert logged_artifacts[0][1] == "sparkml"
    assert len(deleted_prefixes) == 1
    assert save_attempts[1] == f"{deleted_prefixes[0]}/sparkml"


@pytest.mark.parametrize("session_kind", ["connect", "batch_file"])
def test_maybe_save_403_distributed_without_scratch_raises_permission_denied(
    monkeypatch: pytest.MonkeyPatch, session_kind: str
) -> None:
    _set_gcp_tracking(monkeypatch)
    spark = (
        _make_connect_spark_stub()
        if session_kind == "connect"
        else _ClassicSparkStub(master="yarn", default_fs="file:///")
    )
    spark_err = _make_spark_error(monkeypatch, "403 Forbidden")
    model = _ModelStub(lambda _p: (_ for _ in ()).throw(spark_err))

    with (
        unittest.mock.patch.object(runtime, "active_session", return_value=spark),
        pytest.raises(MlflowException) as exc_info,
    ):
        mlflow.spark._maybe_save_model(model, "gs://my-bucket/ns/runs/r1/artifacts/m")

    assert (
        exc_info.value.error_code == MlflowException("", PERMISSION_DENIED).error_code
    )
    assert "dfs_tmpdir" in str(exc_info.value)


@pytest.mark.parametrize(
    ("master", "default_fs"),
    [("local[*]", "file:///"), ("yarn", "hdfs://namenode:8020")],
)
def test_maybe_save_403_local_or_hdfs_returns_false(
    monkeypatch: pytest.MonkeyPatch, master: str, default_fs: str
) -> None:
    _set_gcp_tracking(monkeypatch)
    spark = _ClassicSparkStub(master=master, default_fs=default_fs)
    spark_err = _make_spark_error(monkeypatch, "403 Forbidden")
    model = _ModelStub(lambda _p: (_ for _ in ()).throw(spark_err))

    with unittest.mock.patch.object(runtime, "active_session", return_value=spark):
        ok = mlflow.spark._maybe_save_model(
            model, "gs://my-bucket/ns/runs/r1/artifacts/m"
        )
    assert ok is False


def test_save_model_distributed_file_without_remote_dfs_tmpdir_raises(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _set_gcp_tracking(monkeypatch)
    spark = _ClassicSparkStub(master="yarn", default_fs="file:///")
    model = _ModelStub(lambda _p: None)

    with (
        unittest.mock.patch.object(runtime, "active_session", return_value=spark),
        pytest.raises(MlflowException) as exc_info,
    ):
        mlflow.spark.save_model(model, str(tmp_path / "out"))

    assert (
        exc_info.value.error_code
        == MlflowException("", INVALID_PARAMETER_VALUE).error_code
    )
    assert "dfs_tmpdir" in str(exc_info.value)


def test_save_model_connect_with_gs_dfs_tmpdir_stages_writes_metadata_and_cleans_up(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _set_gcp_tracking(monkeypatch)
    spark = _make_connect_spark_stub()
    saved_paths: list[str] = []
    model = _make_pipeline_model_stub(saved_paths.append)
    deleted: list[str] = []

    try:
        import pyspark  # noqa: F401
    except ImportError:
        fake_pyspark = types.ModuleType("pyspark")
        fake_pyspark.__version__ = "4.0.0"  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "pyspark", fake_pyspark)

    def fake_download_prefix(gs_prefix: str, local_dir: str) -> int:
        os.makedirs(local_dir, exist_ok=True)
        Path(local_dir, "part-000").write_text("weights", encoding="utf-8")
        return 1

    out_dir = tmp_path / "saved_connect_model"
    with (
        unittest.mock.patch.object(runtime, "active_session", return_value=spark),
        unittest.mock.patch.object(mlflow.spark, "_validate_model", autospec=True),
        unittest.mock.patch.object(
            mlflow.spark, "_validate_env_arguments", autospec=True
        ),
        unittest.mock.patch.object(
            mlflow.spark,
            "_save_model_metadata",
            wraps=mlflow.spark._save_model_metadata,
        ) as spy_save_meta,
        unittest.mock.patch.object(
            runtime, "gcs_download_prefix", side_effect=fake_download_prefix
        ),
        unittest.mock.patch.object(
            runtime, "gcs_delete_prefix", side_effect=deleted.append
        ),
    ):
        mlflow.spark.save_model(
            model,
            str(out_dir),
            dfs_tmpdir="gs://my-scratch/tmp",
            pip_requirements=["pyspark==4.0.0"],
        )

    assert len(saved_paths) == 1
    assert saved_paths[0].startswith("gs://my-scratch/tmp/mlflow-spark-")
    assert (out_dir / "sparkml" / "part-000").is_file()
    assert (out_dir / "MLmodel").is_file()
    loaded_meta = Model.load(str(out_dir / "MLmodel"))
    assert "spark" in loaded_meta.flavors
    assert len(deleted) == 1
    assert saved_paths[0] == f"{deleted[0]}/sparkml"
    spy_save_meta.assert_called_once()


# ---------------------------------------------------------------------------
# 8. runtime.py helpers
# ---------------------------------------------------------------------------


def test_runtime_topology_and_gcs_helpers(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    connect = _make_connect_spark_stub()
    local = _ClassicSparkStub(master="local[2]", default_fs="file:///")
    batch_file = _ClassicSparkStub(master="yarn", default_fs="file:///")
    gce_hdfs = _ClassicSparkStub(master="yarn", default_fs="hdfs://nn:8020")

    assert runtime.is_spark_connect(None) is False
    assert runtime.is_spark_connect(connect) is True
    assert runtime.is_spark_connect(local) is False

    assert runtime.is_local_master(connect) is False
    assert runtime.is_local_master(local) is True
    assert runtime.is_local_master(batch_file) is False

    assert runtime.default_fs_scheme(connect) == ""
    assert runtime.default_fs_scheme(local) == "file"
    assert runtime.default_fs_scheme(gce_hdfs) == "hdfs"

    assert runtime.is_distributed_without_shared_dfs(connect) is True
    assert runtime.is_distributed_without_shared_dfs(local) is False
    assert runtime.is_distributed_without_shared_dfs(batch_file) is True
    assert runtime.is_distributed_without_shared_dfs(gce_hdfs) is False

    with unittest.mock.patch.object(runtime, "active_session", return_value=None):
        with pytest.raises(MlflowException):
            runtime.require_active_session()

    # Test GCS upload / download / delete helpers with mock storage.Client
    src_dir = tmp_path / "src"
    (src_dir / "sub").mkdir(parents=True)
    (src_dir / "a.txt").write_text("hello", encoding="utf-8")
    (src_dir / "sub" / "b.txt").write_text("world", encoding="utf-8")

    stored_blobs: dict[str, bytes] = {}

    class _FakeBlob:
        def __init__(self, name: str) -> None:
            self.name = name

        def upload_from_filename(self, filename: str) -> None:
            stored_blobs[self.name] = Path(filename).read_bytes()

        def download_to_filename(self, filename: str) -> None:
            Path(filename).write_bytes(stored_blobs[self.name])

        def delete(self) -> None:
            stored_blobs.pop(self.name, None)

    class _FakeBucket:
        def blob(self, name: str) -> _FakeBlob:
            return _FakeBlob(name)

        def list_blobs(self, prefix: str = "") -> list[_FakeBlob]:
            blobs = [_FakeBlob(f"{prefix}")]  # directory marker should be skipped
            for k in sorted(stored_blobs):
                if k.startswith(prefix):
                    blobs.append(_FakeBlob(k))
            return blobs

    fake_client = unittest.mock.MagicMock()
    fake_client.bucket.return_value = _FakeBucket()

    with unittest.mock.patch("google.cloud.storage.Client", return_value=fake_client):
        uploaded = runtime.gcs_upload_directory(str(src_dir), "gs://my-scratch/p1")
        assert uploaded == [
            ("my-scratch", "p1/a.txt"),
            ("my-scratch", "p1/sub/b.txt"),
        ]

        dst_dir = tmp_path / "dst"
        count = runtime.gcs_download_prefix("gs://my-scratch/p1", str(dst_dir))
        assert count == 2
        assert (dst_dir / "a.txt").read_text(encoding="utf-8") == "hello"
        assert (dst_dir / "sub" / "b.txt").read_text(encoding="utf-8") == "world"

        runtime.gcs_delete_prefix("gs://my-scratch/p1")
        assert stored_blobs == {}


# ---------------------------------------------------------------------------
# 9. Regression Tests (Lifecycle, Topology, Concurrency, Isolation)
# ---------------------------------------------------------------------------


def test_cold_subprocess_import_mlflow_spark_first_with_entry_points(
    tmp_path: Path,
) -> None:
    """Importing mlflow.spark/pyspark.ml/pyfunc cold with entry points patches all 7."""
    dist_info = tmp_path / "google_cloud_managed_mlflow-0.1.0.dist-info"
    dist_info.mkdir()
    (dist_info / "METADATA").write_text(
        "Metadata-Version: 2.1\nName: google-cloud-managed-mlflow\nVersion: 0.1.0\n",
        encoding="utf-8",
    )
    (dist_info / "entry_points.txt").write_text(
        "[mlflow.request_auth_provider]\n"
        "google-cloud-auth = "
        "google_cloud_managed_mlflow.auth:GoogleCloudAuthProvider\n\n"
        "[mlflow.tracking_store]\n"
        "https = google_cloud_managed_mlflow.tracking_store:build_tracking_store\n\n"
        "[mlflow.model_registry_store]\n"
        "https = google_cloud_managed_mlflow.registry_store:build_registry_store\n\n"
        "[mlflow.artifact_repository]\n"
        "gs = google_cloud_managed_mlflow.gcs_artifact_repo:"
        "GcpGCSArtifactRepository\n"
        "models = google_cloud_managed_mlflow.models_artifact_repo:"
        "GcpModelsArtifactRepository\n",
        encoding="utf-8",
    )

    script = (
        "import mlflow.spark\n"
        "import mlflow.pyspark.ml\n"
        "import mlflow.pyfunc\n"
        "from google_cloud_managed_mlflow.registry_store import (\n"
        "    ensure_registry_store_registered,\n"
        ")\n"
        "from google_cloud_managed_mlflow.tracking_store import (\n"
        "    ensure_tracking_store_registered,\n"
        ")\n"
        "marker = '__mlflow_gcp_spark_patched__'\n"
        "assert getattr(mlflow.spark.load_model, marker, False)\n"
        "assert getattr(mlflow.spark.log_model, marker, False)\n"
        "assert getattr(mlflow.spark._maybe_save_model, marker, False)\n"
        "assert getattr(mlflow.spark.save_model, marker, False)\n"
        "assert getattr(mlflow.pyspark.ml._read_log_model_allowlist, marker, False)\n"
        "assert getattr(mlflow.pyspark.ml.autolog, marker, False)\n"
        "assert getattr(mlflow.pyfunc.spark_udf, marker, False)\n"
        "assert ensure_registry_store_registered() is True\n"
        "assert ensure_tracking_store_registered() is True\n"
        "print('COLD_IMPORT_OK')\n"
    )
    env = {
        **os.environ,
        "MLFLOW_TRACKING_URI": _GCP_TRACKING_URI,
        "PYTHONPATH": f"{tmp_path}{os.pathsep}{_REPO_ROOT}",
    }
    proc = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        check=True,
        env=env,
    )
    assert proc.stdout.strip().splitlines()[-1] == "COLD_IMPORT_OK"


def test_when_imported_mid_init_defers_and_finder_not_double_wrapped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Mid-import _initializing=True modules run callbacks once init completes."""
    mod_name = "test_fake_mid_init_mod"
    fake_mod = types.ModuleType(mod_name)
    spec = importlib.machinery.ModuleSpec(mod_name, None)
    spec._initializing = True  # type: ignore[attr-defined]
    fake_mod.__spec__ = spec
    monkeypatch.setitem(sys.modules, mod_name, fake_mod)

    observed: list[str] = []

    def cb(mod: types.ModuleType) -> None:
        observed.append(str(getattr(mod, "ready", "missing")))

    hooks.when_imported(mod_name, cb)
    # Second registration while initializing is idempotent
    hooks.when_imported(mod_name, cb)
    assert observed == []

    fake_mod.ready = "yes"  # type: ignore[attr-defined]
    orig_cls = importlib.machinery.ModuleSpec
    spec._initializing = False  # type: ignore[attr-defined]
    assert observed == ["yes"]
    assert type(spec) is orig_cls

    # _PostImportFinder does not double-wrap an already wrapped _PostImportLoader
    inner_loader = importlib.machinery.SourceFileLoader("dummy", "/tmp/dummy.py")
    wrapped_loader = hooks._PostImportLoader(inner_loader, "mlflow.spark")
    fake_spec = importlib.machinery.ModuleSpec("mlflow.spark", wrapped_loader)

    class _CachingFinder(importlib.abc.MetaPathFinder):
        def find_spec(
            self,
            fullname: str,
            path: Any,
            target: types.ModuleType | None = None,
        ) -> importlib.machinery.ModuleSpec | None:
            return fake_spec if fullname == "mlflow.spark" else None

    monkeypatch.setattr(sys, "meta_path", [hooks._PostImportFinder(), _CachingFinder()])
    finder = sys.meta_path[0]
    assert finder.find_spec("mlflow.spark", None) is fake_spec
    assert fake_spec.loader is wrapped_loader


def test_save_model_wraps_standalone_estimator_and_infers_signature(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _set_gcp_tracking(monkeypatch)
    spark = _make_connect_spark_stub()
    spark.createDataFrame = lambda data: ("spark-df", data)

    wrapped_saved: list[tuple[str, Any]] = []

    class _StandaloneModel:
        def copy(self) -> _StandaloneModel:
            return self

    class _FakePipelineModel:
        def __init__(self, stages: Any = None) -> None:
            self.stages = list(stages or [])

        def copy(self) -> _FakePipelineModel:
            return self

        def write(self) -> Any:
            outer = self

            class _Writer:
                def overwrite(self) -> _Writer:
                    return self

                def save(self, path: str) -> None:
                    wrapped_saved.append((path, outer))

            return _Writer()

    _install_fake_pyspark_ml(monkeypatch, pipeline_cls=_FakePipelineModel)

    def fake_download_prefix(gs_prefix: str, local_dir: str) -> int:
        os.makedirs(local_dir, exist_ok=True)
        Path(local_dir, "part-000").write_text("weights", encoding="utf-8")
        return 1

    inferred_calls: list[tuple[Any, Any]] = []
    meta_calls: list[dict[str, Any]] = []

    def fake_infer_sig(spark_model: Any, spark_df: Any) -> str:
        inferred_calls.append((spark_model, spark_df))
        return "inferred-sig"

    standalone = _StandaloneModel()
    out_dir = tmp_path / "standalone_out"
    with (
        unittest.mock.patch.object(runtime, "active_session", return_value=spark),
        unittest.mock.patch.object(mlflow.spark, "_validate_model", autospec=True),
        unittest.mock.patch.object(
            mlflow.spark, "_validate_env_arguments", autospec=True
        ),
        unittest.mock.patch.object(
            mlflow.pyspark.ml,
            "_infer_spark_model_signature",
            side_effect=fake_infer_sig,
        ),
        unittest.mock.patch.object(
            mlflow.spark,
            "_save_model_metadata",
            autospec=True,
            side_effect=lambda **kw: meta_calls.append(kw),
        ),
        unittest.mock.patch.object(
            runtime, "gcs_download_prefix", side_effect=fake_download_prefix
        ),
        unittest.mock.patch.object(runtime, "gcs_delete_prefix"),
    ):
        mlflow.spark.save_model(
            standalone,
            str(out_dir),
            dfs_tmpdir="gs://my-scratch/tmp",
            input_example=[{"x": 1.0}],
        )

    assert len(wrapped_saved) == 1
    assert isinstance(wrapped_saved[0][1], _FakePipelineModel)
    assert wrapped_saved[0][1].stages == [standalone]
    assert len(inferred_calls) == 1
    assert isinstance(inferred_calls[0][0], _FakePipelineModel)
    assert meta_calls[0]["signature"] == "inferred-sig"
    assert isinstance(meta_calls[0]["spark_model"], _FakePipelineModel)

    # signature=False skips inference and passes signature=None
    inferred_calls.clear()
    meta_calls.clear()
    with (
        unittest.mock.patch.object(runtime, "active_session", return_value=spark),
        unittest.mock.patch.object(mlflow.spark, "_validate_model", autospec=True),
        unittest.mock.patch.object(
            mlflow.spark, "_validate_env_arguments", autospec=True
        ),
        unittest.mock.patch.object(
            mlflow.pyspark.ml,
            "_infer_spark_model_signature",
            side_effect=fake_infer_sig,
        ),
        unittest.mock.patch.object(
            mlflow.spark,
            "_save_model_metadata",
            autospec=True,
            side_effect=lambda **kw: meta_calls.append(kw),
        ),
        unittest.mock.patch.object(
            runtime, "gcs_download_prefix", side_effect=fake_download_prefix
        ),
        unittest.mock.patch.object(runtime, "gcs_delete_prefix"),
    ):
        mlflow.spark.save_model(
            standalone,
            str(tmp_path / "standalone_no_sig"),
            dfs_tmpdir="gs://my-scratch/tmp",
            input_example=[{"x": 1.0}],
            signature=False,
        )

    assert inferred_calls == []
    assert meta_calls[0]["signature"] is None


def test_save_model_classic_distributed_gs_staging_and_topology_rejections(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _set_gcp_tracking(monkeypatch)
    batch_file_spark = _ClassicSparkStub(master="yarn", default_fs="file:///")
    saved_paths: list[str] = []
    model = _make_pipeline_model_stub(saved_paths.append)
    deleted: list[str] = []
    meta_calls: list[dict[str, Any]] = []

    def fake_download_prefix(gs_prefix: str, local_dir: str) -> int:
        os.makedirs(local_dir, exist_ok=True)
        Path(local_dir, "part-000").write_text("weights", encoding="utf-8")
        return 1

    out_dir = tmp_path / "classic_batch_gs"
    with (
        unittest.mock.patch.object(
            runtime, "active_session", return_value=batch_file_spark
        ),
        unittest.mock.patch.object(mlflow.spark, "_validate_model", autospec=True),
        unittest.mock.patch.object(
            mlflow.spark, "_validate_env_arguments", autospec=True
        ),
        unittest.mock.patch.object(
            mlflow.spark,
            "_save_model_metadata",
            autospec=True,
            side_effect=lambda **kw: meta_calls.append(kw),
        ),
        unittest.mock.patch.object(
            runtime, "gcs_download_prefix", side_effect=fake_download_prefix
        ),
        unittest.mock.patch.object(
            runtime, "gcs_delete_prefix", side_effect=deleted.append
        ),
        unittest.mock.patch.object(
            mlflow.spark._HadoopFileSystem,
            "copy_to_local_file",
            side_effect=AssertionError("HadoopFS must not run for gs:// dfs_tmpdir"),
        ),
    ):
        mlflow.spark.save_model(model, str(out_dir), dfs_tmpdir="gs://my-scratch/tmp")

    assert len(saved_paths) == 1
    assert len(deleted) == 1
    assert saved_paths[0] == f"{deleted[0]}/sparkml"
    assert (out_dir / "sparkml" / "part-000").is_file()
    assert len(meta_calls) == 1

    # Spark Connect rejects hdfs:// dfs_tmpdir
    connect_spark = _make_connect_spark_stub()
    with (
        unittest.mock.patch.object(
            runtime, "active_session", return_value=connect_spark
        ),
        pytest.raises(MlflowException) as connect_exc,
    ):
        mlflow.spark.save_model(
            model, str(tmp_path / "bad_connect"), dfs_tmpdir="hdfs://nn:8020/tmp"
        )
    assert (
        connect_exc.value.error_code
        == MlflowException("", INVALID_PARAMETER_VALUE).error_code
    )

    # YARN + HDFS rejects explicit file:///tmp dfs_tmpdir
    yarn_hdfs_spark = _ClassicSparkStub(
        master="yarn", default_fs="hdfs://namenode:8020"
    )
    with (
        unittest.mock.patch.object(
            runtime, "active_session", return_value=yarn_hdfs_spark
        ),
        pytest.raises(MlflowException) as hdfs_exc,
    ):
        mlflow.spark.save_model(
            model, str(tmp_path / "bad_yarn"), dfs_tmpdir="file:///tmp/mlflow"
        )
    assert (
        hdfs_exc.value.error_code
        == MlflowException("", INVALID_PARAMETER_VALUE).error_code
    )


def test_save_and_log_model_connect_model_delegates_without_dfs_tmpdir(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _set_gcp_tracking(monkeypatch)
    connect_spark = _make_connect_spark_stub()
    connect_model = object()
    save_calls: list[Any] = []
    log_calls: list[Any] = []

    fake_spark_mod = types.ModuleType("mlflow.spark")
    fake_spark_mod._is_spark_connect_model = (  # type: ignore[attr-defined]
        lambda m: m is connect_model
    )

    def _orig_log(m: Any, art: str, **kw: Any) -> str:
        log_calls.append((m, art, kw))
        return "logged-connect"

    def _orig_save(m: Any, p: str, dfs_tmpdir: Any = None, **kw: Any) -> str:
        save_calls.append((m, p, dfs_tmpdir))
        return "saved-connect"

    fake_spark_mod.load_model = lambda *a, **kw: None  # type: ignore[attr-defined]
    fake_spark_mod._maybe_save_model = lambda *a, **kw: False  # type: ignore[attr-defined]
    fake_spark_mod.log_model = _orig_log  # type: ignore[attr-defined]
    fake_spark_mod.save_model = _orig_save  # type: ignore[attr-defined]
    hooks.patch_mlflow_spark(fake_spark_mod)

    with unittest.mock.patch.object(
        runtime, "active_session", return_value=connect_spark
    ):
        assert (
            fake_spark_mod.save_model(connect_model, str(tmp_path)) == "saved-connect"
        )
        assert fake_spark_mod.log_model(connect_model, "m") == "logged-connect"

    assert save_calls == [(connect_model, str(tmp_path), None)]
    assert log_calls == [(connect_model, "m", {})]


def test_hybrid_registry_gcp_tracking_local_does_not_pollute_auth_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from mlflow.store.tracking.rest_store import RestStore

    from google_cloud_managed_mlflow.tracking_store import build_tracking_store

    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    monkeypatch.setenv("MLFLOW_REGISTRY_URI", _GCP_REGISTRY_URI)

    captured: list[dict[str, str] | None] = []

    def _fake_udf(spark: Any, uri: str, extra_env: dict[str, str] | None = None) -> str:
        captured.append(extra_env)
        return "udf-ok"

    fake_pyfunc = types.ModuleType("mlflow.pyfunc")
    fake_pyfunc.spark_udf = _fake_udf  # type: ignore[attr-defined]
    hooks.patch_pyfunc(fake_pyfunc)

    assert fake_pyfunc.spark_udf(None, "models:/my-model@champion") == "udf-ok"
    assert "MLFLOW_TRACKING_AUTH" not in os.environ
    assert captured[-1] == {"MLFLOW_REGISTRY_URI": _GCP_REGISTRY_URI}

    store = build_tracking_store("http://localhost:5000")
    assert type(store) is RestStore


def test_load_model_with_code_paths_adds_code_to_system_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_gcp_tracking(monkeypatch)
    spark = _make_connect_spark_stub()
    gcs_uri = "gs://my-bucket/ns/runs/r1/artifacts/model"

    class _CodeRepo:
        artifact_uri = gcs_uri

        def download_artifacts(
            self, artifact_path: str = "", dst_path: str | None = None
        ) -> str:
            assert dst_path is not None
            root = Path(dst_path)
            root.mkdir(parents=True, exist_ok=True)
            if artifact_path == "MLmodel":
                mlmodel_path = root / "MLmodel"
                m = Model()
                m.add_flavor(
                    "spark",
                    model_data="sparkml",
                    model_class="pyspark.ml.pipeline.PipelineModel",
                    code="code",
                )
                m.save(str(mlmodel_path))
                return str(mlmodel_path)
            code_dir = root / artifact_path
            code_dir.mkdir(parents=True, exist_ok=True)
            (code_dir / "custom_stage.py").write_text("X = 1\n", encoding="utf-8")
            return str(code_dir)

    events: list[tuple[str, Any]] = []

    def _record_load(path: str) -> str:
        events.append(("load", path))
        return f"loaded:{path}"

    _FakePipelineModelModule.install(monkeypatch, _record_load)

    def fake_add_code(local_path: str, conf: dict[str, Any]) -> None:
        events.append(("add_code", (local_path, conf.get("code"))))

    with (
        unittest.mock.patch.object(runtime, "active_session", return_value=spark),
        unittest.mock.patch(
            "mlflow.store.artifact.artifact_repository_registry.get_artifact_repository",
            return_value=_CodeRepo(),
        ),
        unittest.mock.patch.object(
            mlflow.spark,
            "_add_code_from_conf_to_system_path",
            side_effect=fake_add_code,
        ),
    ):
        res = mlflow.spark.load_model(gcs_uri)

    assert res == f"loaded:{gcs_uri}/sparkml"
    assert [e[0] for e in events] == ["add_code", "load"]
    assert events[0][1][1] == "code"


def test_patch_hooks_are_idempotent() -> None:
    first_udf = mlflow.pyfunc.spark_udf
    first_allowlist = mlflow.pyspark.ml._read_log_model_allowlist
    first_autolog = mlflow.pyspark.ml.autolog
    first_load = mlflow.spark.load_model
    first_log = mlflow.spark.log_model
    first_maybe = mlflow.spark._maybe_save_model
    first_save = mlflow.spark.save_model

    install_spark_hooks()
    hooks.patch_pyfunc(mlflow.pyfunc)
    hooks.patch_pyspark_ml(mlflow.pyspark.ml)
    hooks.patch_mlflow_spark(mlflow.spark)

    assert mlflow.pyfunc.spark_udf is first_udf
    assert mlflow.pyspark.ml._read_log_model_allowlist is first_allowlist
    assert mlflow.pyspark.ml.autolog is first_autolog
    assert mlflow.spark.load_model is first_load
    assert mlflow.spark.log_model is first_log
    assert mlflow.spark._maybe_save_model is first_maybe
    assert mlflow.spark.save_model is first_save
    assert not getattr(first_udf.__wrapped__, hooks.PATCH_MARKER, False)
    assert not getattr(first_load.__wrapped__, hooks.PATCH_MARKER, False)


def test_concurrent_guarded_fit_preserves_and_restores_unpersist(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_gcp_tracking(monkeypatch)
    unpersist_calls: list[str] = []

    class _CachedDataset:
        is_cached = True

        def unpersist(self) -> _CachedDataset:
            unpersist_calls.append("unpersist")
            self.is_cached = False
            return self

    barrier = threading.Barrier(4)

    def fake_autolog_fit(self: Any, dataset: Any) -> str:
        barrier.wait(timeout=5.0)
        dataset.unpersist()
        return "ok"

    guarded_fit = hooks._make_guarded_fit(fake_autolog_fit)
    ds = _CachedDataset()

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(guarded_fit, object(), ds) for _ in range(4)]
        results = [f.result() for f in futures]

    assert results == ["ok", "ok", "ok", "ok"]
    assert ds.is_cached is True
    assert unpersist_calls == []
    assert hooks._PROTECTED_DATASETS == {}
    ds.unpersist()
    assert unpersist_calls == ["unpersist"]
    assert ds.is_cached is False


def test_real_mlflow_pyspark_ml_autolog_applies_cache_guard(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_gcp_tracking(monkeypatch)
    try:
        from pyspark.ml.base import Estimator as RealEstimator
    except ImportError:

        class _StubEstimator:
            def fit(self, dataset: Any, params: Any = None) -> str:
                return "fitted"

        class _StubModel:
            def transform(self, dataset: Any, params: Any = None) -> Any:
                return dataset

        class _StubEvaluator:
            def evaluate(self, dataset: Any, params: Any = None) -> float:
                return 1.0

        fake_base = types.ModuleType("pyspark.ml.base")
        fake_eval = types.ModuleType("pyspark.ml.evaluation")
        fake_base.Estimator = _StubEstimator  # type: ignore[attr-defined]
        fake_base.Model = _StubModel  # type: ignore[attr-defined]
        fake_eval.Evaluator = _StubEvaluator  # type: ignore[attr-defined]
        _install_fake_pyspark_ml(monkeypatch, base_mod=fake_base, eval_mod=fake_eval)
        RealEstimator = _StubEstimator

    mlflow.pyspark.ml.autolog(disable=False, silent=True)
    current_fit = RealEstimator.__dict__.get("fit")
    assert getattr(current_fit, hooks.PATCH_MARKER, False) is True


def test_gcs_scratch_dir_cleans_parent_prefix_and_rejects_empty_prefix() -> None:
    for invalid_uri in (
        "gs://my-scratch/",
        "gs://my-scratch",
        "gs://my-scratch/tmp/..",
    ):
        with pytest.raises(MlflowException) as exc_info:
            runtime.gcs_delete_prefix(invalid_uri)
        assert (
            exc_info.value.error_code
            == MlflowException("", INVALID_PARAMETER_VALUE).error_code
        )

    deleted_prefixes: list[str] = []
    with (
        unittest.mock.patch.object(
            runtime, "gcs_delete_prefix", side_effect=deleted_prefixes.append
        ),
        pytest.raises(RuntimeError, match="boom"),
    ):
        with runtime.gcs_scratch_dir("gs://my-scratch/tmp", "sparkml") as staged:
            assert staged.startswith("gs://my-scratch/tmp/mlflow-spark-")
            assert staged.endswith("/sparkml")
            raise RuntimeError("boom")

    assert len(deleted_prefixes) == 1
    assert deleted_prefixes[0].startswith("gs://my-scratch/tmp/mlflow-spark-")
    assert not deleted_prefixes[0].endswith("/sparkml")


def test_concurrent_uncached_dataset_stays_cached_mid_fit_and_unpersists_after(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_gcp_tracking(monkeypatch)
    unpersist_calls: list[str] = []

    class _UncachedDataset:
        is_cached = False

        def unpersist(self) -> _UncachedDataset:
            unpersist_calls.append("unpersist")
            self.is_cached = False
            return self

    barrier = threading.Barrier(2)
    observed_mid_fit: list[bool] = []

    def fake_autolog_fit(self: Any, dataset: Any) -> str:
        dataset.is_cached = True
        barrier.wait(timeout=5.0)
        # Simulate MLflow autologging's internal unpersist() before exit
        dataset.unpersist()
        observed_mid_fit.append(bool(dataset.is_cached))
        return "ok"

    guarded_fit = hooks._make_guarded_fit(fake_autolog_fit)
    ds = _UncachedDataset()

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(guarded_fit, object(), ds) for _ in range(2)]
        results = [f.result() for f in futures]

    assert results == ["ok", "ok"]
    assert observed_mid_fit == [True, True]
    assert ds.is_cached is False
    assert unpersist_calls == ["unpersist"]
    assert hooks._PROTECTED_DATASETS == {}

    # Slotted dataset without __dict__ falls back cleanly without crashing
    class _SlottedDataset:
        __slots__ = ("is_cached",)

        def __init__(self) -> None:
            self.is_cached = True

        def unpersist(self) -> _SlottedDataset:
            return self

    slotted_ds = _SlottedDataset()
    slotted_guarded = hooks._make_guarded_fit(lambda _self, _ds: "slotted-ok")
    assert slotted_guarded(object(), slotted_ds) == "slotted-ok"


@pytest.mark.parametrize(
    ("pre_module", "target_module"),
    [
        ("google_cloud_managed_mlflow.tracking_store", "mlflow.pyspark.ml"),
        ("google_cloud_managed_mlflow.registry_store", "mlflow.spark"),
    ],
)
def test_pre_mlflow_import_installs_hooks_in_cold_subprocess(
    pre_module: str, target_module: str
) -> None:
    script = (
        "import sys\n"
        f"import {pre_module}\n"
        "assert 'mlflow' not in sys.modules\n"
        f"import {target_module}\n"
        "import mlflow.pyspark.ml, mlflow.spark, mlflow.pyfunc\n"
        "marker = '__mlflow_gcp_spark_patched__'\n"
        "assert getattr(mlflow.pyspark.ml._read_log_model_allowlist, marker, False)\n"
        "assert getattr(mlflow.pyspark.ml.autolog, marker, False)\n"
        "assert getattr(mlflow.spark.load_model, marker, False)\n"
        "assert getattr(mlflow.spark.log_model, marker, False)\n"
        "assert getattr(mlflow.spark._maybe_save_model, marker, False)\n"
        "assert getattr(mlflow.spark.save_model, marker, False)\n"
        "assert getattr(mlflow.pyfunc.spark_udf, marker, False)\n"
        "print('PRE_MLFLOW_OK')\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        check=True,
        env={**os.environ, "PYTHONPATH": _REPO_ROOT},
    )
    assert proc.stdout.strip().splitlines()[-1] == "PRE_MLFLOW_OK"


def test_load_model_models_uri_with_subpath_and_dst_path_reuse(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _set_gcp_tracking(monkeypatch)
    _set_gcp_registry(monkeypatch)
    spark = _ClassicSparkStub(master="local[*]", default_fs="file:///")
    spark_err = _make_spark_error(monkeypatch, "No FileSystem for scheme: gs")

    reg_gcs = "gs://my-bucket/reg/models/my-model/v1"
    requested_targets: list[str] = []

    class _RootModelsRepo:
        def __init__(self) -> None:
            self.repo = _FakeArtifactRepo(reg_gcs)

        def download_artifacts(
            self, artifact_path: str = "", dst_path: str | None = None
        ) -> str:
            requested_targets.append(artifact_path)
            assert dst_path is not None
            root = Path(dst_path)
            if artifact_path == "submodel/MLmodel":
                return str(_write_mlmodel_file(root / "submodel"))
            if artifact_path == "submodel":
                sub_dir = root / "submodel"
                _write_mlmodel_file(sub_dir)
                data_dir = sub_dir / "sparkml"
                data_dir.mkdir(parents=True, exist_ok=True)
                (data_dir / "part-000").write_text("weights", encoding="utf-8")
                return str(sub_dir)
            raise AssertionError(f"Unexpected download target: {artifact_path!r}")

    attempts: list[str] = []

    def fake_load(path: str) -> str:
        attempts.append(path)
        if path.startswith("gs://"):
            raise spark_err
        return f"loaded:{path}"

    _FakePipelineModelModule.install(monkeypatch, fake_load)
    dst_dir = tmp_path / "dst"

    with (
        unittest.mock.patch.object(runtime, "active_session", return_value=spark),
        unittest.mock.patch(
            "mlflow.store.artifact.artifact_repository_registry.get_artifact_repository",
            return_value=_RootModelsRepo(),
        ),
    ):
        res = mlflow.spark.load_model(
            "models:/my-model@champion/submodel", dst_path=str(dst_dir)
        )

    assert attempts[0] == f"{reg_gcs}/submodel/sparkml"
    assert res == f"loaded:file://{dst_dir / 'submodel' / 'sparkml'}"
    # dst_path already downloaded submodel -> model_data was NOT re-downloaded
    assert requested_targets == ["submodel/MLmodel", "submodel"]


def test_load_model_missing_spark_flavor_raises_resource_does_not_exist(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_gcp_tracking(monkeypatch)
    gcs_uri = "gs://my-bucket/ns/runs/r1/artifacts/sklearn_model"

    class _NonSparkRepo:
        artifact_uri = gcs_uri

        def download_artifacts(
            self, artifact_path: str = "", dst_path: str | None = None
        ) -> str:
            assert dst_path is not None
            mlmodel_path = Path(dst_path) / "MLmodel"
            m = Model()
            m.add_flavor("python_function", loader_module="mlflow.sklearn")
            m.save(str(mlmodel_path))
            return str(mlmodel_path)

    with (
        unittest.mock.patch(
            "mlflow.store.artifact.artifact_repository_registry.get_artifact_repository",
            return_value=_NonSparkRepo(),
        ),
        pytest.raises(MlflowException) as exc_info,
    ):
        mlflow.spark.load_model(gcs_uri)

    assert (
        exc_info.value.error_code
        == MlflowException("", RESOURCE_DOES_NOT_EXIST).error_code
    )
    assert "spark" in str(exc_info.value)


def test_pyfunc_spark_udf_propagates_endpoint_override_without_registry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MLFLOW_GCP_ENDPOINT_OVERRIDE", "https://127.0.0.1:8443")
    monkeypatch.setenv(
        "MLFLOW_TRACKING_URI",
        "https://127.0.0.1:8443/v1/projects/my-project"
        "/locations/us-central1/namespaces/my-namespace",
    )

    captured: list[dict[str, str] | None] = []

    def fake_udf(
        spark: Any,
        model_uri: Any,
        result_type: Any = None,
        env_manager: Any = None,
        params: dict[str, Any] | None = None,
        extra_env: dict[str, str] | None = None,
    ) -> str:
        captured.append(extra_env)
        return "udf-ok"

    fake_mod = types.ModuleType("mlflow.pyfunc")
    fake_mod.spark_udf = fake_udf  # type: ignore[attr-defined]
    hooks.patch_pyfunc(fake_mod)

    assert fake_mod.spark_udf(None, "runs:/run-123/model") == "udf-ok"
    assert captured == [{"MLFLOW_GCP_ENDPOINT_OVERRIDE": "https://127.0.0.1:8443"}]


def test_autolog_disable_true_and_scratch_validation_guards(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _set_gcp_tracking(monkeypatch)
    mlflow.pyspark.ml.autolog(disable=True, silent=True)
    for cls in hooks._all_estimator_classes():
        assert not getattr(cls.__dict__.get("fit"), hooks.PATCH_MARKER, False)

    # Empty scratch download raises INTERNAL_ERROR
    model = _ModelStub(lambda _p: None)
    with (
        unittest.mock.patch.object(runtime, "gcs_download_prefix", return_value=0),
        unittest.mock.patch.object(runtime, "gcs_delete_prefix"),
        pytest.raises(MlflowException) as empty_exc,
    ):
        hooks._save_spark_model_via_gcs_scratch(
            model, "gs://my-scratch/tmp", str(tmp_path / "out")
        )
    assert empty_exc.value.error_code == MlflowException("", INTERNAL_ERROR).error_code

    # Unsafe blob key p1/../escape.txt rejected in gcs_download_prefix
    unsafe_blob = unittest.mock.MagicMock()
    unsafe_blob.name = "p1/../escape.txt"
    fake_bucket = unittest.mock.MagicMock()
    fake_bucket.list_blobs.return_value = [unsafe_blob]
    fake_client = unittest.mock.MagicMock()
    fake_client.bucket.return_value = fake_bucket
    with (
        unittest.mock.patch("google.cloud.storage.Client", return_value=fake_client),
        pytest.raises(MlflowException) as unsafe_exc,
    ):
        runtime.gcs_download_prefix("gs://my-scratch/p1", str(tmp_path / "dl"))
    assert (
        unsafe_exc.value.error_code
        == MlflowException("", INVALID_PARAMETER_VALUE).error_code
    )

    # Empty subdir "/" rejected in gcs_scratch_dir
    with pytest.raises(MlflowException) as subdir_exc:
        with runtime.gcs_scratch_dir("gs://my-scratch/tmp", "/"):
            pass
    assert (
        subdir_exc.value.error_code
        == MlflowException("", INVALID_PARAMETER_VALUE).error_code
    )
