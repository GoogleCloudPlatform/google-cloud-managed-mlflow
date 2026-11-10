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

"""Spark end-to-end tests: Spark Connect with the provider uploaded on demand.

The Connect server only has the Cloud Storage connector preinstalled; the
plugin uploads the provider JAR with ``SparkSession.addArtifacts``. See
:mod:`tests.spark_support`.
"""

from __future__ import annotations

import dataclasses
import glob
import os
import shutil
import socket
import subprocess
import time
import uuid
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import pytest

from google_cloud_managed_mlflow.spark import (
    CONTROL_SOURCE_CLASS,
    cab_scope as cab,
    hooks,
    runtime,
)
from tests.conftest import ManagedEnv
from tests.fakes.mlflow_server import NAMESPACE_BUCKET, NAMESPACE_ROOT
from tests.spark_support import (
    SparkEnv,
    _assert_client_supplier_round,
    _assert_no_executor_minting,
    _assert_no_token_leaks,
    _assert_workers_never_imported_plugin,
    _blocked_worker_conf,
    _client_supplier_round,
    _common_conf,
    _features,
    _fit_pipeline,
    _gcs_names,
    _run_less_save_round,
    pyspark,
    spark_env,
)

pytestmark = pytest.mark.spark
__all__ = ["spark_env"]

CONNECT = "connect"


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


@dataclasses.dataclass(frozen=True)
class ConnectServer:
    """A running Spark Connect server (its log is the driver log)."""

    url: str
    log: Path


@pytest.fixture(scope="module")
def connect_server(spark_env: SparkEnv) -> Iterator[ConnectServer]:
    pyspark_dir = Path(pyspark.__file__).resolve().parent
    connect_jars = sorted(glob.glob(str(pyspark_dir / "jars" / "spark-connect_*.jar")))
    if not connect_jars:
        pytest.skip("spark-connect JAR not bundled with this PySpark")
    port = _free_port()
    server_dir = spark_env.workdir / "connect-server"
    server_dir.mkdir()
    command = [
        str(pyspark_dir / "bin" / "spark-submit"),
        "--master",
        "local-cluster[2,1,1024]",
    ]
    conf = _common_conf(spark_env)
    conf["spark.connect.grpc.binding.port"] = str(port)
    conf["spark.connect.grpc.binding.address"] = "127.0.0.1"
    conf.update(_blocked_worker_conf(spark_env, CONNECT))
    for key, value in conf.items():
        command += ["--conf", f"{key}={value}"]
    command += ["--jars", f"{connect_jars[0]},{spark_env.connector}"]
    command += [
        "--class",
        "org.apache.spark.sql.connect.service.SparkConnectServer",
        connect_jars[0],
    ]
    log_path = server_dir / "server.log"
    with log_path.open("w") as log:
        process = subprocess.Popen(
            command,
            cwd=server_dir,
            env=dict(os.environ),
            stdout=log,
            stderr=subprocess.STDOUT,
        )
    try:
        deadline = time.time() + 120
        while time.time() < deadline:
            if process.poll() is not None:
                raise AssertionError(log_path.read_text()[-4000:])
            with socket.socket() as probe:
                if probe.connect_ex(("127.0.0.1", port)) == 0:
                    break
            time.sleep(0.5)
        else:
            raise AssertionError("Spark Connect server did not start")
        yield ConnectServer(f"sc://127.0.0.1:{port}", log_path)
    finally:
        process.terminate()
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            process.kill()
        assert "ya29." not in log_path.read_text(errors="replace")
        shutil.copy(log_path, spark_env.workdir / "connect-server.log")


@pytest.fixture(scope="module")
def connect_spark(connect_server: ConnectServer) -> Iterator[Any]:
    from pyspark.sql import SparkSession

    spark = SparkSession.builder.remote(connect_server.url).getOrCreate()
    try:
        yield spark
    finally:
        spark.stop()
        runtime._SESSION_STATE.clear()


def test_connect_add_artifact_provider_and_executor_reads(
    managed: ManagedEnv, connect_spark: Any, spark_env: SparkEnv
) -> None:
    assert runtime.is_spark_connect(connect_spark)
    assert runtime.ensure_provider(connect_spark) == "add_artifact"
    namespace = hooks.managed_namespace()
    assert namespace is not None
    base = f"{NAMESPACE_ROOT}/connect/{uuid.uuid4().hex}"
    scope = cab.namespace_scope(namespace, write=True, artifact_uri=base)
    path = hooks.cab_path_for(connect_spark, f"{base}/data", scope)
    connect_spark.range(0, 500, numPartitions=4).write.parquet(path)
    assert connect_spark.read.parquet(path).count() == 500
    assert any(
        n.endswith(".parquet") for n in _gcs_names(spark_env, NAMESPACE_BUCKET, base)
    )
    stats = runtime.driver_stats(connect_spark, scope.scope_id)
    assert stats["executorCabRpcs"] == "0"
    assert stats["scopeMints"] == "1"
    _assert_no_executor_minting(managed.server)
    assert spark_env.gcs.denied() == []
    conf = connect_spark.conf.getAll
    assert all("ya29." not in str(v) for v in conf.values())


def test_connect_spark_model_log_register_and_load(
    managed: ManagedEnv, connect_spark: Any, spark_env: SparkEnv
) -> None:
    frame = _features(connect_spark)
    model = _fit_pipeline(frame)
    expected = [
        row.prediction for row in model.transform(frame).orderBy("x1").collect()
    ]
    name = f"connect-{uuid.uuid4().hex[:8]}"
    with mlflow.start_run():
        info = mlflow.spark.log_model(model, "spark-model", registered_model_name=name)
    loaded = mlflow.spark.load_model(info.model_uri)
    got = [row.prediction for row in loaded.transform(frame).orderBy("x1").collect()]
    np.testing.assert_allclose(got, expected, rtol=1e-9)
    from_registry = mlflow.spark.load_model(f"models:/{name}/1")
    got = [
        row.prediction for row in from_registry.transform(frame).orderBy("x1").collect()
    ]
    np.testing.assert_allclose(got, expected, rtol=1e-9)
    _assert_no_executor_minting(managed.server)
    assert spark_env.gcs.denied() == []
    _assert_workers_never_imported_plugin(spark_env, CONNECT)


def test_connect_client_supplier_wraps_and_renews_tokens(
    managed: ManagedEnv,
    connect_spark: Any,
    connect_server: ConnectServer,
    spark_env: SparkEnv,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    base = f"{NAMESPACE_ROOT}/connect-client/{uuid.uuid4().hex}"
    recorded, _, stats = _client_supplier_round(
        connect_spark, managed.server, monkeypatch, base, via_env=True
    )
    probe = _assert_client_supplier_round(
        managed.server, recorded, stats, f"{base}/client"
    )
    frame = connect_spark.read.format(CONTROL_SOURCE_CLASS).options(**probe).load()
    capsys.readouterr()
    frame.explain(True)
    plans = capsys.readouterr().out
    assert plans and "ya29." not in plans
    _assert_no_executor_minting(managed.server)
    assert spark_env.gcs.denied() == []
    _assert_no_token_leaks(
        connect_spark,
        spark_env,
        app_id=str(connect_spark.conf.get("spark.app.id")),
        extra_logs=(connect_server.log,),
    )


def test_connect_run_less_save_model_cleans_up_staging(
    managed: ManagedEnv,
    connect_spark: Any,
    spark_env: SparkEnv,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from pyspark.ml import PipelineModel

    frame = _features(connect_spark, rows=200)
    model = _fit_pipeline(frame)
    target = tmp_path / "saved"
    _run_less_save_round(
        connect_spark, spark_env, model, target, tmp_path / "failed", monkeypatch
    )
    expected = [r.prediction for r in model.transform(frame).orderBy("x1").collect()]
    reloaded = PipelineModel.load(str(target / "sparkml"))
    got = [r.prediction for r in reloaded.transform(frame).orderBy("x1").collect()]
    np.testing.assert_allclose(got, expected, rtol=1e-9)
    _assert_no_executor_minting(managed.server)
    assert spark_env.gcs.denied() == []


def test_connect_second_sequential_session(
    managed: ManagedEnv,
    connect_server: ConnectServer,
    connect_spark: Any,
    spark_env: SparkEnv,
) -> None:
    from pyspark.sql import SparkSession

    namespace = hooks.managed_namespace()
    assert namespace is not None
    base = f"{NAMESPACE_ROOT}/sessions/{uuid.uuid4().hex}"
    first_scope = cab.namespace_scope(namespace, write=True, artifact_uri=base)
    first_path = hooks.cab_path_for(connect_spark, f"{base}/data", first_scope)
    connect_spark.range(0, 50, numPartitions=2).write.parquet(first_path)
    second = SparkSession.builder.remote(connect_server.url).create()
    try:
        assert second.session_id != connect_spark.session_id
        assert runtime.ensure_provider(second) == "add_artifact"
        read_scope = cab.namespace_scope(namespace, write=False, artifact_uri=base)
        path = hooks.cab_path_for(second, f"{base}/data", read_scope)
        assert second.read.parquet(path).count() == 50
        second_scopes = runtime._state(second)["scopes"]
        assert read_scope.scope_id in second_scopes
        assert first_scope.scope_id not in second_scopes
        assert runtime.driver_stats(second)["executorCabRpcs"] == "0"
    finally:
        second.stop()
    assert connect_spark.read.parquet(first_path).count() == 50
    _assert_no_executor_minting(managed.server)
    assert spark_env.gcs.denied() == []
