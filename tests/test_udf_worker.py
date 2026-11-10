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

"""Executor-side UDF bodies: self-contained pickles and loading without MLflow.

Serverless executors provide numpy, pandas, scikit-learn, XGBoost, and PyTorch
but neither ``mlflow`` nor this plugin. The pickled UDF bodies must therefore
reference neither package and must load models natively when ``import mlflow``
fails, producing the same predictions as ``mlflow.pyfunc``.
"""

from __future__ import annotations

import dataclasses
import json
import math
import subprocess
import sys
import tempfile
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import pandas as pd
import pytest
import torch
import xgboost
from mlflow.pyfunc.model import PythonModel
from pyspark import cloudpickle as spark_cloudpickle
from sklearn.linear_model import LinearRegression

from google_cloud_managed_mlflow.spark import udf as managed_udf

PLUGIN_MODULE = b"google_cloud_managed_mlflow"
_X = np.linspace(-2.0, 2.0, 64)
FEATURES = pd.DataFrame({"x1": _X, "x2": np.cos(3.0 * _X)})
TARGET = 3.0 * FEATURES["x1"].to_numpy() - 2.0 * FEATURES["x2"].to_numpy() + 0.5


@dataclasses.dataclass(frozen=True)
class SavedModel:
    """A saved MLflow model directory and its ground-truth predictions."""

    path: Path
    expected: np.ndarray
    rtol: float
    frame: pd.DataFrame = dataclasses.field(default_factory=lambda: FEATURES)


def _python_spec(path: Path, *, allow_pickle: bool = True) -> dict[str, Any]:
    from mlflow.models import Model

    meta = Model.load(str(path / "MLmodel"))
    flavors = managed_udf._plain(meta.flavors)
    spec: dict[str, Any] = {
        "kind": "python_model",
        "payload": managed_udf._zip_directory(str(path)),
        "flavors": flavors,
        "input_names": ["x1", "x2"],
        "result_kind": "double",
        "params": None,
        "model_config": None,
        "allow_pickle": allow_pickle,
    }
    native = managed_udf._native_sklearn_payload(str(path), flavors)
    if native is not None:
        spec["native_sklearn"] = native
    return spec


def _batches(frame: pd.DataFrame) -> Iterator[pd.DataFrame]:
    yield frame.iloc[:40].reset_index(drop=True)
    yield frame.iloc[40:].reset_index(drop=True)


def _run(spec: dict[str, Any], frame: pd.DataFrame = FEATURES) -> np.ndarray:
    predict = managed_udf.worker_predict_fn(spec)
    outputs = list(predict(_batches(frame)))
    return np.asarray(pd.concat(outputs, ignore_index=True).to_numpy())


@pytest.fixture(scope="module")
def saved_models(tmp_path_factory: pytest.TempPathFactory) -> dict[str, SavedModel]:
    root = tmp_path_factory.mktemp("udf-models")
    models: dict[str, SavedModel] = {}
    linear = LinearRegression().fit(FEATURES, TARGET)
    for fmt in ("skops", "cloudpickle", "pickle"):
        path = root / f"sklearn-{fmt}"
        mlflow.sklearn.save_model(
            linear,
            str(path),
            serialization_format=fmt,
            pip_requirements=["scikit-learn"],
        )
        models[f"sklearn-{fmt}"] = SavedModel(path, linear.predict(FEATURES), 0.0)
    regressor = xgboost.XGBRegressor(n_estimators=20, max_depth=3).fit(FEATURES, TARGET)
    path = root / "xgboost-regressor"
    mlflow.xgboost.save_model(regressor, str(path), pip_requirements=["xgboost"])
    models["xgboost-regressor"] = SavedModel(path, regressor.predict(FEATURES), 1e-6)
    booster = xgboost.train(
        {"max_depth": 3}, xgboost.DMatrix(FEATURES, label=TARGET), num_boost_round=20
    )
    path = root / "xgboost-booster"
    mlflow.xgboost.save_model(booster, str(path), pip_requirements=["xgboost"])
    models["xgboost-booster"] = SavedModel(
        path, booster.predict(xgboost.DMatrix(FEATURES)), 1e-6
    )
    torch.manual_seed(0)
    network = torch.nn.Sequential(
        torch.nn.Linear(2, 8), torch.nn.Tanh(), torch.nn.Linear(8, 1)
    ).eval()
    inputs = FEATURES.to_numpy(dtype=np.float32)
    with torch.no_grad():
        torch_expected = network(torch.from_numpy(inputs)).numpy()[:, 0]
    torch_expected = torch_expected.astype(np.float64)
    for kind, model_object, options in (
        ("pytorch-module", network, {"serialization_format": "pickle"}),
        ("pytorch-pt2", network, {"input_example": inputs[:4]}),
        (
            "pytorch-scripted",
            torch.jit.script(network),
            {"serialization_format": "pickle"},
        ),
    ):
        path = root / kind
        mlflow.pytorch.save_model(
            model_object, str(path), pip_requirements=["torch"], **options
        )
        # A pt2 signature is a float32 tensor spec that MLflow enforces exactly.
        frame = FEATURES.astype(np.float32) if kind == "pytorch-pt2" else FEATURES
        models[kind] = SavedModel(path, torch_expected, 1e-6, frame)
    return models


MODEL_KINDS = [
    "sklearn-skops",
    "sklearn-cloudpickle",
    "sklearn-pickle",
    "xgboost-regressor",
    "xgboost-booster",
    "pytorch-module",
    "pytorch-pt2",
    "pytorch-scripted",
]


@pytest.mark.parametrize("kind", MODEL_KINDS)
def test_worker_matches_mlflow_with_and_without_mlflow(
    saved_models: dict[str, SavedModel], kind: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = saved_models[kind]
    spec = _python_spec(model.path)
    loaded = mlflow.pyfunc.load_model(str(model.path))
    reference = np.asarray(loaded.predict(model.frame))
    np.testing.assert_allclose(
        reference.reshape(-1), model.expected, rtol=model.rtol or 1e-12
    )
    with_mlflow = _run(spec, model.frame)
    np.testing.assert_allclose(with_mlflow, model.expected, rtol=model.rtol or 1e-12)
    # Executors without MLflow: `import mlflow` fails with name == "mlflow".
    monkeypatch.setitem(sys.modules, "mlflow", None)
    without_mlflow = _run(spec, model.frame)
    np.testing.assert_allclose(without_mlflow, model.expected, rtol=model.rtol or 1e-12)


def test_pickled_udfs_reference_neither_plugin_nor_mlflow(
    saved_models: dict[str, SavedModel], tmp_path: Path
) -> None:
    worker_tmp = tmp_path / "worker-tmp"
    worker_tmp.mkdir()
    pickles: dict[str, list[str]] = {}
    for kind in MODEL_KINDS:
        model = saved_models[kind]
        data = spark_cloudpickle.dumps(
            managed_udf.worker_predict_fn(_python_spec(model.path))
        )
        assert PLUGIN_MODULE not in data, kind
        path = tmp_path / f"{kind}.pkl"
        path.write_bytes(data)
        frame_path = tmp_path / f"{kind}.frame.pkl"
        model.frame.to_pickle(frame_path)
        pickles[kind] = [str(path), str(frame_path)]
    # A fresh interpreter where neither package can be imported, like an executor.
    script = "\n".join(
        [
            "import json, pickle, sys",
            "sys.modules['mlflow'] = None",
            "sys.modules['google_cloud_managed_mlflow'] = None",
            "import pandas as pd",
            "result = {}",
            "for kind, (path, frame_path) in json.loads(sys.argv[1]).items():",
            "    frame = pd.read_pickle(frame_path)",
            "    with open(path, 'rb') as handle:",
            "        predict = pickle.load(handle)",
            "    outputs = list(predict(iter([frame])))",
            "    result[kind] = pd.concat(outputs).tolist()",
            "print(json.dumps(result))",
        ]
    )
    completed = subprocess.run(
        [sys.executable, "-c", script, json.dumps(pickles)],
        capture_output=True,
        text=True,
        check=False,
        cwd=tmp_path,
        env={"TMPDIR": str(worker_tmp), "PYTHONDONTWRITEBYTECODE": "1"},
    )
    assert completed.returncode == 0, completed.stderr[-3000:]
    results = json.loads(completed.stdout.strip().splitlines()[-1])
    for kind in MODEL_KINDS:
        model = saved_models[kind]
        np.testing.assert_allclose(
            results[kind], model.expected, rtol=model.rtol or 1e-12
        )
    assert list(worker_tmp.iterdir()) == []


def test_worker_removes_its_extraction_directory(
    saved_models: dict[str, SavedModel],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    monkeypatch.setitem(sys.modules, "mlflow", None)
    predict = managed_udf.worker_predict_fn(
        _python_spec(saved_models["sklearn-pickle"].path)
    )
    outputs = predict(_batches(FEATURES))
    next(outputs)
    assert len(list(tmp_path.glob("mlflow-cab-udf-*"))) == 1
    list(outputs)
    assert list(tmp_path.glob("mlflow-cab-udf-*")) == []


def test_pickle_deserialization_gate_applies_without_mlflow(
    saved_models: dict[str, SavedModel], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(sys.modules, "mlflow", None)
    spec = _python_spec(saved_models["sklearn-pickle"].path, allow_pickle=False)
    with pytest.raises(RuntimeError, match="MLFLOW_ALLOW_PICKLE_DESERIALIZATION"):
        _run(spec)
    monkeypatch.setenv("MLFLOW_ALLOW_PICKLE_DESERIALIZATION", "false")
    with pytest.raises(RuntimeError, match="MLFLOW_ALLOW_PICKLE_DESERIALIZATION"):
        _run(_python_spec(saved_models["pytorch-module"].path))


class _Doubler(PythonModel):
    def predict(self, context: Any, model_input: Any, params: Any = None) -> Any:
        return model_input["x1"] * 2


def test_custom_python_models_need_mlflow_on_executors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "custom"
    mlflow.pyfunc.save_model(
        str(path), python_model=_Doubler(), pip_requirements=["mlflow"]
    )
    spec = _python_spec(path)
    np.testing.assert_allclose(_run(spec), FEATURES["x1"].to_numpy() * 2)
    monkeypatch.setitem(sys.modules, "mlflow", None)
    with pytest.raises(RuntimeError, match="Install mlflow on the executors"):
        _run(spec)


# ------------------------------------------------------- decoded SparkXGB pipelines


def _pipeline_spec(
    booster: xgboost.Booster,
    task: str,
    stages: list[dict[str, Any]],
    features_col: str,
    kinds: dict[str, str],
) -> dict[str, Any]:
    return {
        "kind": "xgboost_pipeline",
        "booster": bytes(booster.save_raw("json")),
        "task": task,
        "missing": math.nan,
        "input_cols": list(kinds),
        "input_kinds": kinds,
        "stages": stages,
        "features_col": features_col,
        "result_kind": "double",
    }


def _scaled(raw: np.ndarray) -> tuple[np.ndarray, list[float], list[float]]:
    mean = raw.mean(axis=0)
    std = raw.std(axis=0, ddof=1)
    return (raw - mean) / std, mean.tolist(), std.tolist()


ASSEMBLER = {
    "type": "assembler",
    "input_cols": ["x1", "x2"],
    "output_col": "features",
    "handle_invalid": "error",
}


@pytest.mark.parametrize("classes", [2, 3])
def test_decoded_classifier_pipeline_matches_xgboost(classes: int) -> None:
    rng = np.random.default_rng(3)
    raw = rng.normal(size=(300, 2))
    scaled, mean, std = _scaled(raw)
    score = scaled[:, 0] - 0.5 * scaled[:, 1]
    labels = np.digitize(
        score, np.quantile(score, np.linspace(0, 1, classes + 1)[1:-1])
    )
    classifier = xgboost.XGBClassifier(n_estimators=15, max_depth=3).fit(
        scaled.astype(np.float32), labels
    )
    scaler = {
        "type": "scaler",
        "input_col": "features",
        "output_col": "scaled",
        "with_mean": True,
        "with_std": True,
        "mean": mean,
        "std": std,
    }
    spec = _pipeline_spec(
        classifier.get_booster(),
        "classifier",
        [ASSEMBLER, scaler],
        "scaled",
        {"x1": "scalar", "x2": "scalar"},
    )
    frame = pd.DataFrame({"x1": raw[:, 0], "x2": raw[:, 1]})
    got = _run(spec, frame)
    expected = classifier.predict(scaled.astype(np.float32)).astype(np.float64)
    np.testing.assert_array_equal(got, expected)
    assert set(np.unique(got)) <= set(range(classes))


def test_decoded_regressor_pipeline_on_vector_input_matches_xgboost() -> None:
    from pyspark.ml.linalg import Vectors

    rng = np.random.default_rng(5)
    raw = rng.normal(size=(200, 3))
    target = raw @ np.array([1.0, -2.0, 0.5])
    regressor = xgboost.XGBRegressor(n_estimators=10, max_depth=2).fit(
        raw.astype(np.float32), target
    )
    spec = _pipeline_spec(
        regressor.get_booster(), "regressor", [], "features", {"features": "vector"}
    )
    expected = regressor.predict(raw.astype(np.float32)).astype(np.float64)
    arrays = pd.DataFrame({"features": list(raw)})
    np.testing.assert_allclose(_run(spec, arrays), expected, rtol=1e-6)
    pipeline = managed_udf.XGBoostPipeline(
        spec=spec,
        input_cols=["features"],
        vector_inputs=frozenset({"features"}),
        prediction_col="prediction",
    )
    vectors = pd.DataFrame({"features": [Vectors.dense(row) for row in raw]})
    local = managed_udf.LocalPipelinePredictor(pipeline).predict(vectors)
    np.testing.assert_allclose(local, expected, rtol=1e-6)


def test_decoded_assembler_honors_handle_invalid() -> None:
    rng = np.random.default_rng(9)
    raw = rng.normal(size=(50, 2))
    regressor = xgboost.XGBRegressor(n_estimators=5, max_depth=2).fit(
        raw.astype(np.float32), raw[:, 0]
    )
    frame = pd.DataFrame({"x1": raw[:, 0], "x2": raw[:, 1]})
    frame.loc[3, "x2"] = np.nan
    skip = dict(ASSEMBLER, handle_invalid="skip")
    spec = _pipeline_spec(
        regressor.get_booster(),
        "regressor",
        [skip],
        "features",
        {"x1": "scalar", "x2": "scalar"},
    )
    got = _run(spec, frame)
    assert np.isnan(got[3]) and not np.isnan(np.delete(got, 3)).any()
    strict = _pipeline_spec(
        regressor.get_booster(),
        "regressor",
        [ASSEMBLER],
        "features",
        {"x1": "scalar", "x2": "scalar"},
    )
    with pytest.raises(ValueError, match="handleInvalid=error"):
        _run(strict, frame)
