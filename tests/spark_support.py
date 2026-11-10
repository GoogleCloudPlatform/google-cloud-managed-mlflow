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

"""Shared fixtures and helpers for the Spark end-to-end tests.

These tests start real Spark JVMs with two remote executors, the bundled
provider JAR, and a shaded Cloud Storage connector JAR whose path is given by
``MLFLOW_CAB_TEST_GCS_CONNECTOR_JAR``. All traffic stays on 127.0.0.1: Cloud
Storage goes to :class:`tests.fakes.gcs.FakeGcs` (which rejects any request
outside a token's prefix or mode) and Managed MLflow to
:class:`tests.fakes.mlflow_server.FakeManagedMlflow`. Executors are given an
unreachable metadata server, so any executor-side token minting would fail.
"""

from __future__ import annotations

import dataclasses
import os
import sys
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from tests.fakes.gcs import FakeGcs
from tests.fakes.mlflow_server import FakeManagedMlflow

pyspark = pytest.importorskip("pyspark")

CONNECTOR_ENV = "MLFLOW_CAB_TEST_GCS_CONNECTOR_JAR"
ROWS = 1000


def _connector_jar() -> str:
    path = os.environ.get(CONNECTOR_ENV, "")
    if not path or not os.path.isfile(path):
        pytest.skip(
            f"Set {CONNECTOR_ENV} to a shaded gcs-connector JAR to run Spark tests."
        )
    return path


@dataclasses.dataclass
class SparkEnv:
    """Shared paths and fakes for Spark tests."""

    gcs: FakeGcs
    server: FakeManagedMlflow
    connector: str
    workdir: Path
    executor_tmp: Path


@pytest.fixture(scope="module")
def spark_env(
    fake_gcs: FakeGcs,
    fake_mlflow: FakeManagedMlflow,
    tmp_path_factory: pytest.TempPathFactory,
) -> Iterator[SparkEnv]:
    connector = _connector_jar()
    workdir = tmp_path_factory.mktemp("spark")
    executor_tmp = workdir / "executor-tmp"
    executor_tmp.mkdir()
    saved = dict(os.environ)
    os.environ.update(
        {
            "GCE_METADATA_HOST": fake_mlflow.metadata_host,
            "MLFLOW_GCP_ENDPOINT_OVERRIDE": fake_mlflow.origin,
            "STORAGE_EMULATOR_HOST": fake_gcs.endpoint,
            "PYSPARK_PYTHON": sys.executable,
            "PYSPARK_DRIVER_PYTHON": sys.executable,
            "SPARK_LOCAL_DIRS": str(workdir / "local"),
            "PYTHONDONTWRITEBYTECODE": "1",
        }
    )
    os.environ.pop("SPARK_HOME", None)
    try:
        yield SparkEnv(fake_gcs, fake_mlflow, connector, workdir, executor_tmp)
    finally:
        os.environ.clear()
        os.environ.update(saved)


def _common_conf(env: SparkEnv) -> dict[str, str]:
    return {
        "spark.hadoop.fs.gs.storage.root.url": env.gcs.endpoint + "/",
        "spark.hadoop.fs.defaultFS": "file:///",
        "spark.executorEnv.GCE_METADATA_HOST": "127.0.0.1:9",
        "spark.executorEnv.TMPDIR": str(env.executor_tmp),
        "spark.executorEnv.PYTHONDONTWRITEBYTECODE": "1",
        "spark.sql.shuffle.partitions": "4",
        "spark.default.parallelism": "4",
        "spark.ui.enabled": "false",
        "spark.task.maxFailures": "1",
        "spark.local.dir": str(env.workdir / "local"),
        "spark.driver.bindAddress": "127.0.0.1",
        "spark.driver.host": "127.0.0.1",
    }


def _wait_for_executors(spark: Any, count: int) -> None:
    deadline = time.time() + 60
    while time.time() < deadline:
        status = spark.sparkContext._jsc.sc().getExecutorMemoryStatus()
        if status.size() >= count + 1:
            return
        time.sleep(0.5)
    raise AssertionError("executors did not register")


def _features(spark: Any, rows: int = ROWS, partitions: int = 4) -> Any:
    rng = np.random.default_rng(7)
    x1 = rng.normal(size=rows)
    x2 = rng.normal(size=rows)
    label = 3.0 * x1 - 2.0 * x2 + 0.5
    data = [
        (float(a), float(b), float(c)) for a, b, c in zip(x1, x2, label, strict=True)
    ]
    return spark.createDataFrame(data, ["x1", "x2", "label"]).repartition(partitions)


def _fit_pipeline(frame: Any) -> Any:
    from pyspark.ml import Pipeline
    from pyspark.ml.feature import VectorAssembler
    from pyspark.ml.regression import LinearRegression

    return Pipeline(
        stages=[
            VectorAssembler(inputCols=["x1", "x2"], outputCol="features"),
            LinearRegression(featuresCol="features", labelCol="label", maxIter=5),
        ]
    ).fit(frame)


def _fs_gs_snapshot(spark: Any) -> dict[str, str]:
    conf = spark.sparkContext._jsc.hadoopConfiguration()
    iterator = conf.iterator()
    snapshot: dict[str, str] = {}
    while iterator.hasNext():
        entry = iterator.next()
        if str(entry.getKey()).startswith("fs.gs."):
            snapshot[str(entry.getKey())] = str(entry.getValue())
    return snapshot


def _python_flavor_models(local: Any) -> list[tuple[str, str, Any, bool]]:
    """Log XGBoost and PyTorch models trained on ``local`` (pandas).

    Returns ``(flavor, model_uri, expected predictions ordered by x1,
    float32_inputs)``; PyTorch signatures are float32 tensors.
    """
    import mlflow
    import torch
    import xgboost

    features = local[["x1", "x2"]]
    ordered = local.sort_values("x1")[["x1", "x2"]]
    models: list[tuple[str, str, Any, bool]] = []
    regressor = xgboost.XGBRegressor(n_estimators=20, max_depth=3).fit(
        features, local["label"]
    )
    with mlflow.start_run():
        info = mlflow.xgboost.log_model(
            regressor,
            name="model",
            input_example=features[:3],
            pip_requirements=["xgboost"],
        )
    models.append(("xgboost", info.model_uri, regressor.predict(ordered), False))
    torch.manual_seed(0)
    network = torch.nn.Sequential(
        torch.nn.Linear(2, 8), torch.nn.Tanh(), torch.nn.Linear(8, 1)
    ).eval()
    with torch.no_grad():
        values = torch.from_numpy(ordered.to_numpy(dtype=np.float32))
        expected = network(values).numpy()[:, 0]
    example = features[:4].to_numpy(dtype=np.float32)
    for fmt in ("pickle", "pt2"):
        with mlflow.start_run():
            info = mlflow.pytorch.log_model(
                network,
                name="model",
                input_example=example,
                serialization_format=fmt,
                pip_requirements=["torch"],
            )
        models.append((f"pytorch-{fmt}", info.model_uri, expected, True))
    return models


def _assert_no_executor_minting(server: FakeManagedMlflow) -> None:
    roles = {call.role for call in server.cab_calls_snapshot()}
    assert "java-executor" not in roles
    assert roles <= {"java-driver", "python-client"}


def _executor_log_files(spark: Any, app_id: str | None = None) -> list[Path]:
    """stdout/stderr of this application's executors (local-cluster work dir)."""
    from pyspark.find_spark_home import _find_spark_home

    app = app_id or str(spark.sparkContext.applicationId)
    app_dir = Path(_find_spark_home()) / "work" / app
    return sorted(path for path in app_dir.glob("*/std*") if path.is_file())


def _assert_no_token_leaks(
    spark: Any,
    env: SparkEnv,
    *,
    app_id: str | None = None,
    extra_logs: tuple[Path, ...] = (),
) -> None:
    conf = (
        spark.conf.getAll.items()
        if app_id is not None
        else spark.sparkContext.getConf().getAll()
    )
    for key, value in conf:
        assert "ya29." not in str(value), key
    logs = _executor_log_files(spark, app_id)
    assert any(path.name == "stderr" for path in logs), "executor logs not found"
    for path in [*logs, *extra_logs]:
        assert path.is_file(), path
        assert "ya29." not in path.read_text(errors="replace"), path


def _gcs_names(env: SparkEnv, bucket: str, gcs_uri: str) -> list[str]:
    return env.gcs.list_names(bucket, gcs_uri.split("/", 3)[3])


# ------------------------------------------------- executors without mlflow / plugin

BLOCKED_MODULES = "mlflow,google_cloud_managed_mlflow"
_SITECUSTOMIZE = '''\
"""Test-only: make chosen top-level modules unimportable in this interpreter."""
import importlib.abc
import os
import sys

_BLOCKED = tuple(
    m for m in os.environ.get("MLFLOW_CAB_TEST_BLOCKED_MODULES", "").split(",") if m
)
_LOG = os.environ.get("MLFLOW_CAB_TEST_BLOCK_LOG", "")


class _Blocker(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".", 1)[0] not in _BLOCKED:
            return None
        if _LOG:
            with open(_LOG, "a", encoding="utf-8") as handle:
                handle.write(f"{os.getpid()} {fullname}\\n")
        raise ModuleNotFoundError(
            f"No module named {fullname!r} (blocked for this test)", name=fullname
        )


if _BLOCKED:
    sys.meta_path.insert(0, _Blocker())
'''


def _blocked_worker_conf(env: SparkEnv, name: str) -> dict[str, str]:
    """Executor settings that block ``mlflow`` and the plugin in Python workers."""
    site = env.workdir / f"{name}-worker-site"
    site.mkdir(exist_ok=True)
    (site / "sitecustomize.py").write_text(_SITECUSTOMIZE)
    log = env.workdir / f"{name}-blocked-imports.log"
    log.touch()
    return {
        "spark.executorEnv.PYTHONPATH": str(site),
        "spark.executorEnv.MLFLOW_CAB_TEST_BLOCKED_MODULES": BLOCKED_MODULES,
        "spark.executorEnv.MLFLOW_CAB_TEST_BLOCK_LOG": str(log),
    }


def _blocked_imports(env: SparkEnv, name: str) -> list[str]:
    """Modules that executor Python workers tried (and failed) to import."""
    log = env.workdir / f"{name}-blocked-imports.log"
    lines = log.read_text().splitlines() if log.exists() else []
    return [line.split(" ", 1)[1] for line in lines if " " in line]


def _assert_workers_never_imported_plugin(env: SparkEnv, name: str) -> None:
    attempts = _blocked_imports(env, name)
    assert not [m for m in attempts if m.startswith("google_cloud_managed_mlflow")]


def _driver_log_options(env: SparkEnv, name: str) -> tuple[str, Path]:
    """``spark.driver.extraJavaOptions`` sending driver logs to a file."""
    log = env.workdir / f"{name}-driver.log"
    config = env.workdir / f"{name}-log4j2.properties"
    config.write_text(
        "\n".join(
            [
                "rootLogger.level = info",
                "rootLogger.appenderRef.file.ref = file",
                "appender.file.type = File",
                "appender.file.name = file",
                f"appender.file.fileName = {log}",
                "appender.file.layout.type = PatternLayout",
                "appender.file.layout.pattern = %d %p %c: %m%n%ex",
                "logger.cab.name = com.google.cloud.mlflow",
                "logger.cab.level = debug",
                "",
            ]
        )
    )
    return f"-Dlog4j2.configurationFile=file:{config}", log


# ------------------------------------------------------- client token supplier (F3)


def _client_supplier_round(
    spark: Any,
    server: FakeManagedMlflow,
    monkeypatch: Any,
    base: str,
    *,
    via_env: bool = False,
) -> tuple[list[dict[str, str]], Any, dict[str, str]]:
    """Push client-minted tokens for ~16s with a short TTL; record ensure options."""
    from google_cloud_managed_mlflow.spark import cab_scope as cab, hooks, runtime

    namespace = hooks.managed_namespace()
    assert namespace is not None
    server.token_ttl_seconds = 6.0
    monkeypatch.setenv(cab.REFRESH_MARGIN_ENV, "3")
    recorded: list[dict[str, str]] = []
    original = runtime.control

    def spy(session: Any, action: str, **options: Any) -> dict[str, str]:
        if action == "ensure":
            recorded.append({k: str(v) for k, v in options.items() if v is not None})
        return original(session, action, **options)

    monkeypatch.setattr(runtime, "control", spy)
    scope = cab.namespace_scope(
        namespace,
        write=True,
        artifact_uri=f"{base}/client",
    )
    if via_env:
        monkeypatch.setenv(runtime.SUPPLIER_ENV, "client")
    else:
        spark.conf.set(runtime.SUPPLIER_CONF, "client")
    try:
        path = hooks.cab_path_for(spark, f"{base}/client/data", scope)
        spark.range(0, 200, numPartitions=4).write.parquet(path)
        deadline = time.time() + 16
        while time.time() < deadline:
            assert spark.read.parquet(path).count() == 200
            time.sleep(0.5)
        stats = runtime.driver_stats(spark, scope.scope_id)
        runtime.release_scope(spark, scope)
    finally:
        if not via_env:
            spark.conf.unset(runtime.SUPPLIER_CONF)
    return recorded, scope, stats


def _assert_client_supplier_round(
    server: FakeManagedMlflow,
    recorded: list[dict[str, str]],
    stats: dict[str, str],
    boundary: str,
) -> dict[str, str]:
    """Checks a client-supplier round; returns the last ensure options sent."""
    assert stats["supplier"] == "CLIENT", stats
    assert int(stats["scopeClientPushes"]) >= 3, stats
    assert stats["scopeMints"] == "0", stats
    assert stats["executorCabRpcs"] == "0", stats
    calls = [c for c in server.cab_calls_snapshot() if c.boundary == boundary]
    assert len([c for c in calls if c.role == "python-client"]) >= 3
    assert not [c for c in calls if c.role != "python-client"]
    pushes = [o for o in recorded if o.get("supplier") == "client"]
    assert len(pushes) >= 3
    for options in recorded:
        assert "token" not in options
        assert all("ya29." not in value for value in options.values()), options
    assert all(o.get("tokenSecret") for o in pushes)
    return dict(pushes[-1], action="probe")


# ------------------------------------------------------- run-less save_model (F5)

STAGING_PREFIX = "experiments/.mlflow-cab-staging/"
_WRITE_OPS = ("insert", "resumable_start")


def _run_less_save_round(
    spark: Any,
    env: SparkEnv,
    model: Any,
    target: Path,
    failed_target: Path,
    monkeypatch: Any,
) -> None:
    """Run-less save_model succeeds and leaves no staging objects or scopes."""
    import mlflow

    from google_cloud_managed_mlflow.spark import hooks, runtime
    from tests.fakes.mlflow_server import NAMESPACE_BUCKET

    released: list[Any] = []
    original_release = runtime.release_scope

    def spy_release(session: Any, scope: Any) -> None:
        released.append(scope)
        original_release(session, scope)

    monkeypatch.setattr(runtime, "release_scope", spy_release)
    assert mlflow.active_run() is None
    env.gcs.clear_records()
    mlflow.spark.save_model(model, str(target))
    assert (target / "MLmodel").is_file()
    assert (target / "sparkml" / "metadata").is_dir()
    assert env.gcs.list_names(NAMESPACE_BUCKET, STAGING_PREFIX) == []
    writes = [
        r
        for r in env.gcs.records_snapshot()
        if STAGING_PREFIX in r.name and r.op in _WRITE_OPS
    ]
    assert writes, "the model must be staged through Managed MLflow storage"
    assert len(released) == 1

    def fail_download(entry: Any, dst: str) -> None:
        raise RuntimeError("simulated download failure")

    monkeypatch.setattr(hooks, "_download_dir", fail_download)
    env.gcs.clear_records()
    try:
        mlflow.spark.save_model(model, str(failed_target))
    except Exception as exc:
        assert "simulated download failure" in str(exc)
    else:
        raise AssertionError("save_model should have failed")
    staged = [r for r in env.gcs.records_snapshot() if STAGING_PREFIX in r.name]
    assert any(r.op in _WRITE_OPS for r in staged)
    assert any(r.op == "delete" for r in staged)
    assert env.gcs.list_names(NAMESPACE_BUCKET, STAGING_PREFIX) == []
    assert len(released) == 2
    state = runtime._state(spark)["scopes"]
    for scope in released:
        assert scope.scope_id not in state
        assert runtime.driver_stats(spark, scope.scope_id)["active"] == "false"
