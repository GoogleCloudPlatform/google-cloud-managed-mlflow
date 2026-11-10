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

"""Spark end-to-end tests: classic PySpark on a local cluster with two executors.

The provider JAR and the Cloud Storage connector are passed with ``spark.jars``
and the optional Spark plugin is enabled. See :mod:`tests.spark_support`.
"""

from __future__ import annotations

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
    SPARK_PLUGIN_CLASS,
    bundled_jar_path,
    cab_scope as cab,
    hooks,
    runtime,
)
from tests.conftest import ManagedEnv
from tests.fakes.mlflow_server import NAMESPACE_BUCKET, NAMESPACE_ROOT, REGISTRY_BUCKET
from tests.spark_support import (
    SparkEnv,
    _assert_client_supplier_round,
    _assert_no_executor_minting,
    _assert_no_token_leaks,
    _blocked_worker_conf,
    _client_supplier_round,
    _common_conf,
    _driver_log_options,
    _features,
    _fit_pipeline,
    _fs_gs_snapshot,
    _gcs_names,
    _run_less_save_round,
    _wait_for_executors,
    spark_env,
)

pytestmark = pytest.mark.spark
__all__ = ["spark_env"]


CLASSIC = "classic"


@pytest.fixture(scope="module")
def classic_spark(spark_env: SparkEnv) -> Iterator[Any]:
    from pyspark.sql import SparkSession

    builder = SparkSession.builder.master("local-cluster[2,1,1024]").appName(
        "mlflow-cab-e2e"
    )
    conf = _common_conf(spark_env)
    conf.update(_blocked_worker_conf(spark_env, CLASSIC))
    java_options, _ = _driver_log_options(spark_env, CLASSIC)
    conf["spark.driver.extraJavaOptions"] = java_options
    for key, value in conf.items():
        builder = builder.config(key, value)
    builder = builder.config(
        "spark.jars", f"{bundled_jar_path()},{spark_env.connector}"
    )
    builder = builder.config("spark.plugins", SPARK_PLUGIN_CLASS)
    spark = builder.getOrCreate()
    try:
        _wait_for_executors(spark, 2)
        yield spark
    finally:
        spark.stop()
        runtime._SESSION_STATE.clear()


def _driver_log(env: SparkEnv) -> Path:
    return env.workdir / f"{CLASSIC}-driver.log"


def test_classic_spark_model_log_register_and_load(
    managed: ManagedEnv, classic_spark: Any, spark_env: SparkEnv
) -> None:
    frame = _features(classic_spark)
    model = _fit_pipeline(frame)
    expected = [
        row.prediction for row in model.transform(frame).orderBy("x1").collect()
    ]
    before = _fs_gs_snapshot(classic_spark)
    name = f"spark-{uuid.uuid4().hex[:8]}"
    with mlflow.start_run():
        info = mlflow.spark.log_model(model, "spark-model", registered_model_name=name)
    all_ns_names = spark_env.gcs.list_names(NAMESPACE_BUCKET, "experiments/")
    assert any("/sparkml/stages/" in n and n.endswith(".parquet") for n in all_ns_names)
    assert not any(".mlflow-cab-staging" in n for n in all_ns_names)
    vault = mlflow.MlflowClient().get_model_version_download_uri(name, "1")
    assert vault.startswith(f"{managed.server.storage_location(name)}/versions/")
    vault_names = _gcs_names(spark_env, REGISTRY_BUCKET, vault)
    assert any(n.endswith("/MLmodel") for n in vault_names)
    assert any("/sparkml/stages/" in n and n.endswith(".parquet") for n in vault_names)

    round_trip = mlflow.spark.load_model(info.model_uri)
    got = [
        row.prediction for row in round_trip.transform(frame).orderBy("x1").collect()
    ]
    np.testing.assert_allclose(got, expected, rtol=1e-9)
    loaded = mlflow.spark.load_model(f"models:/{name}/1")
    got = [row.prediction for row in loaded.transform(frame).orderBy("x1").collect()]
    np.testing.assert_allclose(got, expected, rtol=1e-9)

    stats = runtime.driver_stats(classic_spark)
    assert stats["executorCabRpcs"] == "0"
    assert stats["remoteExecutors"] == "2"
    assert int(stats["broadcasts"]) >= 1
    _assert_no_executor_minting(managed.server)
    assert _fs_gs_snapshot(classic_spark) == before
    assert spark_env.gcs.denied() == []
    _assert_no_token_leaks(
        classic_spark, spark_env, extra_logs=(_driver_log(spark_env),)
    )


def test_classic_scope_isolation_and_timed_renewal(
    managed: ManagedEnv,
    classic_spark: Any,
    spark_env: SparkEnv,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    namespace = hooks.managed_namespace()
    assert namespace is not None
    base = f"{NAMESPACE_ROOT}/isolation/{uuid.uuid4().hex}"
    write_a = cab.namespace_scope(namespace, write=True, artifact_uri=f"{base}/a")
    write_b = cab.namespace_scope(namespace, write=True, artifact_uri=f"{base}/b")
    path_a = hooks.cab_path_for(classic_spark, f"{base}/a/data", write_a)
    path_b = hooks.cab_path_for(classic_spark, f"{base}/b/data", write_b)
    classic_spark.range(0, 400, numPartitions=4).write.parquet(path_a)
    classic_spark.range(0, 300, numPartitions=4).write.parquet(path_b)
    assert classic_spark.read.parquet(path_a).count() == 400
    assert classic_spark.read.parquet(path_b).count() == 300

    # A path under scope A's id but outside A's prefix is rejected by storage.
    cross = path_a.replace("/a/data", "/b/data")
    with pytest.raises(Exception, match="403|Forbidden|AccessDenied|does not cover"):
        classic_spark.read.parquet(cross).count()
    spark_env.gcs.clear_records()

    # Timed renewal: short TTL plus a small lead window forces >= 3 rotations.
    managed.server.token_ttl_seconds = 6.0
    monkeypatch.setenv(cab.REFRESH_MARGIN_ENV, "3")
    rotating = cab.namespace_scope(
        namespace,
        write=True,
        artifact_uri=f"{base}/rotating",
    )
    path_r = hooks.cab_path_for(classic_spark, f"{base}/rotating/data", rotating)
    classic_spark.range(0, 100, numPartitions=4).write.parquet(path_r)
    deadline = time.time() + 16
    iterations = 0
    while time.time() < deadline:
        assert classic_spark.read.parquet(path_r).count() == 100
        iterations += 1
        time.sleep(0.5)
    stats = runtime.driver_stats(classic_spark, rotating.scope_id)
    assert int(stats["scopeMints"]) >= 4, stats
    assert stats["executorCabRpcs"] == "0"
    assert iterations >= 5
    assert [r for r in spark_env.gcs.denied()] == []
    driver_calls = [
        c
        for c in managed.server.cab_calls_snapshot()
        if c.role == "java-driver" and c.boundary == f"{base}/rotating"
    ]
    assert len(driver_calls) >= 4
    _assert_no_executor_minting(managed.server)
    _assert_no_token_leaks(classic_spark, spark_env)


def test_classic_client_supplier_wraps_and_renews_tokens(
    managed: ManagedEnv,
    classic_spark: Any,
    spark_env: SparkEnv,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base = f"{NAMESPACE_ROOT}/client-supplier/{uuid.uuid4().hex}"
    recorded, _, stats = _client_supplier_round(
        classic_spark, managed.server, monkeypatch, base
    )
    probe = _assert_client_supplier_round(
        managed.server, recorded, stats, f"{base}/client"
    )
    frame = classic_spark.read.format(CONTROL_SOURCE_CLASS).options(**probe).load()
    plans = frame._jdf.queryExecution().toString()
    assert probe["tokenSecret"] and "ya29." not in plans
    _assert_no_executor_minting(managed.server)
    assert spark_env.gcs.denied() == []
    _assert_no_token_leaks(
        classic_spark, spark_env, extra_logs=(_driver_log(spark_env),)
    )


def test_classic_run_less_save_model_cleans_up_staging(
    managed: ManagedEnv,
    classic_spark: Any,
    spark_env: SparkEnv,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from pyspark.ml import PipelineModel

    frame = _features(classic_spark, rows=200)
    model = _fit_pipeline(frame)
    target = tmp_path / "saved"
    _run_less_save_round(
        classic_spark, spark_env, model, target, tmp_path / "failed", monkeypatch
    )
    expected = [r.prediction for r in model.transform(frame).orderBy("x1").collect()]
    reloaded = PipelineModel.load(str(target / "sparkml"))
    got = [r.prediction for r in reloaded.transform(frame).orderBy("x1").collect()]
    np.testing.assert_allclose(got, expected, rtol=1e-9)
    _assert_no_executor_minting(managed.server)
    assert spark_env.gcs.denied() == []
