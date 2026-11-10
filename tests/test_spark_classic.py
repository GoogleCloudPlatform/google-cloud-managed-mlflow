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
from tests import spark_models
from tests.conftest import ManagedEnv
from tests.fakes.mlflow_server import NAMESPACE_BUCKET, NAMESPACE_ROOT, REGISTRY_BUCKET
from tests.spark_support import (
    ROWS,
    SparkEnv,
    _assert_client_supplier_round,
    _assert_no_executor_minting,
    _assert_no_token_leaks,
    _assert_workers_never_imported_plugin,
    _blocked_imports,
    _blocked_worker_conf,
    _client_supplier_round,
    _common_conf,
    _driver_log_options,
    _features,
    _fit_pipeline,
    _fs_gs_snapshot,
    _gcs_names,
    _python_flavor_models,
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
    # Executors behave like serverless runtimes: no mlflow, no plugin.
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


def test_classic_spark_model_log_register_load_and_udf(
    managed: ManagedEnv, classic_spark: Any, spark_env: SparkEnv
) -> None:
    from pyspark.sql import functions as F

    frame = _features(classic_spark)
    model = _fit_pipeline(frame)
    expected = [
        row.prediction for row in model.transform(frame).orderBy("x1").collect()
    ]
    before = _fs_gs_snapshot(classic_spark)
    name = f"spark-{uuid.uuid4().hex[:8]}"
    with mlflow.start_run() as run:
        info = mlflow.spark.log_model(model, "spark-model", registered_model_name=name)
        artifact_root = run.info.artifact_uri
    run_names = _gcs_names(spark_env, NAMESPACE_BUCKET, f"{artifact_root}/spark-model")
    assert any("/sparkml/stages/" in n and n.endswith(".parquet") for n in run_names)
    assert not any(".mlflow-cab-staging" in n for n in run_names)
    vault = mlflow.MlflowClient().get_model_version_download_uri(name, "1")
    assert vault.startswith(f"{managed.server.storage_location(name)}/versions/")
    vault_names = _gcs_names(spark_env, REGISTRY_BUCKET, vault)
    assert any(n.endswith("/MLmodel") for n in vault_names)
    assert any("/sparkml/stages/" in n and n.endswith(".parquet") for n in vault_names)

    # The URI returned by log_model loads with tracking and registry URIs set.
    round_trip = mlflow.spark.load_model(info.model_uri)
    got = [
        row.prediction for row in round_trip.transform(frame).orderBy("x1").collect()
    ]
    np.testing.assert_allclose(got, expected, rtol=1e-9)
    loaded = mlflow.spark.load_model(f"models:/{name}/1")
    got = [row.prediction for row in loaded.transform(frame).orderBy("x1").collect()]
    np.testing.assert_allclose(got, expected, rtol=1e-9)

    pyfunc_model = mlflow.pyfunc.load_model(f"runs:/{run.info.run_id}/spark-model")
    sample = frame.orderBy("x1").limit(50).toPandas()
    np.testing.assert_allclose(
        np.asarray(pyfunc_model.predict(sample[["x1", "x2"]])), expected[:50], rtol=1e-9
    )

    udf = mlflow.pyfunc.spark_udf(classic_spark, f"models:/{name}/1")
    scored = frame.withColumn("p_cols", udf("x1", "x2")).withColumn(
        "p_struct", udf(F.struct("x1", "x2"))
    )
    rows = scored.orderBy("x1").collect()
    assert len(rows) == ROWS
    np.testing.assert_allclose([r.p_cols for r in rows], expected, rtol=1e-9)
    np.testing.assert_allclose([r.p_struct for r in rows], expected, rtol=1e-9)

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


def test_classic_python_flavor_spark_udf_parity(
    managed: ManagedEnv, classic_spark: Any, spark_env: SparkEnv
) -> None:
    from sklearn.linear_model import LinearRegression

    frame = _features(classic_spark)
    local = frame.toPandas()
    model = LinearRegression().fit(local[["x1", "x2"]], local["label"])
    name = f"skl-{uuid.uuid4().hex[:8]}"
    with mlflow.start_run():
        info = mlflow.sklearn.log_model(
            model,
            name="model",
            registered_model_name=name,
            input_example=local[["x1", "x2"]][:3],
        )
    expected = model.predict(local.sort_values("x1")[["x1", "x2"]])
    for uri in (info.model_uri, f"models:/{name}/1"):
        calls_before = len(managed.server.cab_calls_snapshot())
        udf = mlflow.pyfunc.spark_udf(classic_spark, uri, result_type="double")
        calls_after_build = len(managed.server.cab_calls_snapshot())
        rows = frame.withColumn("p", udf("x1", "x2")).orderBy("x1").collect()
        assert len(rows) == ROWS
        np.testing.assert_allclose([r.p for r in rows], expected, rtol=1e-9)
        assert len(managed.server.cab_calls_snapshot()) == calls_after_build
        assert calls_after_build >= calls_before
    assert not list(spark_env.executor_tmp.glob("mlflow-cab-udf-*"))
    # Workers had no mlflow (they tried it, then loaded natively) and never
    # needed the plugin.
    assert "mlflow.pyfunc" in _blocked_imports(spark_env, CLASSIC) or "mlflow" in (
        _blocked_imports(spark_env, CLASSIC)
    )
    _assert_workers_never_imported_plugin(spark_env, CLASSIC)
    _assert_no_executor_minting(managed.server)
    assert spark_env.gcs.denied() == []


def test_classic_xgboost_and_pytorch_udfs_without_mlflow_on_workers(
    managed: ManagedEnv, classic_spark: Any, spark_env: SparkEnv
) -> None:
    from pyspark.sql import functions as F

    frame = _features(classic_spark)
    for flavor, model_uri, expected, float32 in _python_flavor_models(frame.toPandas()):
        udf = mlflow.pyfunc.spark_udf(classic_spark, model_uri, result_type="double")
        columns = [
            F.col(c).cast("float") if float32 else F.col(c) for c in ("x1", "x2")
        ]
        rows = frame.withColumn("p", udf(*columns)).orderBy("x1").collect()
        assert len(rows) == ROWS, flavor
        np.testing.assert_allclose([r.p for r in rows], expected, rtol=1e-5)
    assert not list(spark_env.executor_tmp.glob("mlflow-cab-udf-*"))
    _assert_workers_never_imported_plugin(spark_env, CLASSIC)
    assert runtime.driver_stats(classic_spark)["executorCabRpcs"] == "0"
    _assert_no_executor_minting(managed.server)
    assert spark_env.gcs.denied() == []


def test_classic_sparkxgb_pipelines_udf_and_load_parity(
    managed: ManagedEnv, classic_spark: Any, spark_env: SparkEnv
) -> None:
    frame = _features(classic_spark)
    inputs = spark_models.with_inputs(frame)
    for stage_name, model in spark_models.fit_pipelines(frame).items():
        columns, _ = spark_models.PIPELINES[stage_name]
        expected = spark_models.predictions(model, frame)
        name = f"{stage_name}-{uuid.uuid4().hex[:6]}"
        with mlflow.start_run() as run:
            mlflow.spark.log_model(model, stage_name, registered_model_name=name)
        for uri in (f"runs:/{run.info.run_id}/{stage_name}", f"models:/{name}/1"):
            udf = mlflow.pyfunc.spark_udf(classic_spark, uri)
            rows = inputs.withColumn("p", udf(*columns)).orderBy("x1").collect()
            assert len(rows) == ROWS
            np.testing.assert_allclose([r.p for r in rows], expected, rtol=1e-6)
        loaded = mlflow.spark.load_model(f"models:/{name}/1")
        got = [r.prediction for r in loaded.transform(inputs).orderBy("x1").collect()]
        np.testing.assert_allclose(got, expected, rtol=1e-6)
    _assert_workers_never_imported_plugin(spark_env, CLASSIC)
    assert runtime.driver_stats(classic_spark)["executorCabRpcs"] == "0"
    _assert_no_executor_minting(managed.server)
    assert spark_env.gcs.denied() == []


def test_classic_autolog_preserves_caller_cache_and_logs_model(
    managed: ManagedEnv, classic_spark: Any
) -> None:
    from pyspark.ml.feature import VectorAssembler
    from pyspark.ml.regression import LinearRegression

    assembled = VectorAssembler(inputCols=["x1", "x2"], outputCol="features").transform(
        _features(classic_spark, rows=200)
    )
    cached = assembled.cache()
    cached.count()
    uncached = assembled.select("features", "label")
    mlflow.pyspark.ml.autolog(log_models=True)
    try:
        with mlflow.start_run() as run:
            LinearRegression(maxIter=3).fit(cached)
        assert cached.is_cached, "autologging must not evict the caller's cache"
        with mlflow.start_run():
            LinearRegression(maxIter=3).fit(uncached)
        assert not uncached.is_cached, "autologging must release its own cache"
    finally:
        mlflow.pyspark.ml.autolog(disable=True)
        cached.unpersist()
    client = mlflow.MlflowClient()
    params = client.get_run(run.info.run_id).data.params
    assert params.get("maxIter") == "3"
    logged = {info.path for info in client.list_artifacts(run.info.run_id, "model")}
    assert {"model/MLmodel", "model/sparkml"} <= logged
    _assert_no_executor_minting(managed.server)


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
    # What Catalyst sees for the exact options the client sent.
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
