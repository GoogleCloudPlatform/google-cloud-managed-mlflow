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

"""Executor-side bodies of the Managed MLflow ``spark_udf`` replacements.

:mod:`google_cloud_managed_mlflow.spark.udf` compiles this file's source under a
neutral module name, so the closures returned by :func:`make_predict_fn` are
serialized *by value* and reference only the standard library plus runtimes that
executors already provide (``numpy``, ``pandas``, and the model's framework).
Executors therefore need neither ``mlflow`` nor this plugin, never fetch
artifacts, and never hold credentials.

Only the standard library may be imported at module level, and every helper
used on executors is a nested closure of :func:`make_predict_fn`.
"""

from __future__ import annotations

import io
import os
import pickle
import shutil
import tempfile
import zipfile
from collections.abc import Callable, Iterator
from typing import Any

WORKER_TEMP_PREFIX = "mlflow-cab-udf-"
SINGLE_INPUT_FIELD = "__mlflow_cab_input__"


def make_predict_fn(spec: dict[str, Any]) -> Callable[[Iterator[Any]], Iterator[Any]]:
    """Return the scalar-iterator pandas UDF body described by ``spec``.

    ``spec`` contains only ``str``, ``bytes``, numbers, ``bool``, ``None``,
    lists, and dicts:

    * ``kind="python_model"``: ``payload`` (zip of the MLflow model directory),
      ``flavors`` (the ``MLmodel`` flavors), ``input_names``, ``params``,
      ``model_config``, ``allow_pickle``, and an optional ``native_sklearn``
      (cloudpickle of the fitted estimator for executors without ``skops``).
    * ``kind="xgboost_pipeline"``: ``booster`` (XGBoost JSON model),
      ``task`` (``classifier``/``regressor``), ``missing``, ``input_cols``,
      ``input_kinds`` (``scalar``/``vector`` per column), and ``stages`` (decoded
      ``VectorAssembler``/``StandardScalerModel`` parameters).

    Both kinds also carry ``result_kind``, the Spark result type's simple
    string (``double``, ``float``, ``int``, ``bigint``, ``boolean``,
    ``string``, or ``array<...>``).
    """
    result_kind = str(spec.get("result_kind") or "double")

    def to_series(result: Any, size: int) -> Any:
        """Convert model output into one pandas Series of ``size`` rows."""
        import numpy as np
        import pandas as pd

        wants_array = result_kind.startswith("array")
        if hasattr(result, "detach") and hasattr(result, "cpu"):
            result = result.detach().cpu().numpy()
        if isinstance(result, pd.DataFrame):
            if wants_array:
                series = pd.Series(result.to_numpy().tolist())
            else:
                series = result.iloc[:, 0]
        elif isinstance(result, pd.Series):
            series = result
        else:
            array = np.asarray(result)
            if array.ndim > 1:
                if wants_array:
                    return pd.Series(array.tolist())
                array = array[:, 0]
            elif array.ndim == 0:
                array = np.full(size, array.item())
            series = pd.Series(array)
        series = series.reset_index(drop=True)
        casts = {
            "double": "float64",
            "float": "float32",
            "int": "int32",
            "bigint": "int64",
            "boolean": "bool",
        }
        if result_kind in casts:
            return series.astype(casts[result_kind])
        if result_kind == "string":
            return series.map(str)
        return series

    def as_frame(batch: Any, names: list[str]) -> Any:
        """Rebuild the model input DataFrame from the struct column batch."""
        import pandas as pd

        if isinstance(batch, pd.Series):
            batch = batch.to_frame()
        if list(batch.columns) == [SINGLE_INPUT_FIELD]:
            column = batch[SINGLE_INPUT_FIELD]
            first = next((v for v in column if v is not None), None)
            if isinstance(first, dict):
                return pd.DataFrame(list(column), index=batch.index)
            if first is not None and hasattr(first, "asDict"):
                return pd.DataFrame([v.asDict() for v in column], index=batch.index)
            return column.to_frame(names[0] if names else "0")
        return batch

    def allow_deserialization() -> None:
        """Honor MLflow's deserialization gate on both the client and executor."""
        local = os.environ.get("MLFLOW_ALLOW_PICKLE_DESERIALIZATION", "true")
        if not spec.get("allow_pickle", True) or local.strip().lower() in (
            "false",
            "0",
            "no",
        ):
            raise RuntimeError(
                "Deserializing this model is disallowed because "
                "MLFLOW_ALLOW_PICKLE_DESERIALIZATION is not 'true'."
            )

    def native_sklearn(workdir: str, flavors: dict[str, Any]) -> Callable[[Any], Any]:
        conf = flavors.get("sklearn") or {}
        pyfunc_conf = flavors.get("python_function") or {}
        method_name = str(pyfunc_conf.get("predict_fn") or "predict")
        allow_deserialization()
        prebuilt = spec.get("native_sklearn")
        if isinstance(prebuilt, bytes):
            model = pickle.loads(prebuilt)
        else:
            relative = pyfunc_conf.get("model_path") or conf.get("pickled_model")
            path = os.path.join(workdir, str(relative or "model.pkl"))
            fmt = str(conf.get("serialization_format") or "pickle")
            if fmt == "skops":
                import skops.io

                model = skops.io.load(path, trusted=conf.get("skops_trusted_types"))
            elif fmt == "joblib":
                import joblib

                model = joblib.load(path)
            else:
                with open(path, "rb") as handle:
                    # cloudpickle output is standard pickle data; unpickling it only
                    # needs cloudpickle importable for by-value objects it references.
                    model = pickle.load(handle)
        predict: Callable[[Any], Any] = getattr(model, method_name)
        return predict

    def native_xgboost(workdir: str, flavors: dict[str, Any]) -> Callable[[Any], Any]:
        import xgboost

        conf = flavors.get("xgboost") or {}
        path = os.path.join(workdir, str(conf.get("data") or "model.xgb"))
        class_name = str(conf.get("model_class") or "Booster").rsplit(".", 1)[-1]
        model = getattr(xgboost, class_name)()
        model.load_model(path)
        if isinstance(model, xgboost.Booster):
            booster = model

            def predict_booster(frame: Any) -> Any:
                return booster.predict(xgboost.DMatrix(frame), validate_features=True)

            return predict_booster

        def predict_estimator(frame: Any) -> Any:
            return model.predict(frame, validate_features=True)

        return predict_estimator

    def native_pytorch(workdir: str, flavors: dict[str, Any]) -> Callable[[Any], Any]:
        import numpy as np
        import pandas as pd
        import torch

        conf = flavors.get("pytorch") or {}
        data = os.path.join(workdir, str(conf.get("model_data") or "data"))
        pickled = os.path.join(data, "model.pth") if os.path.isdir(data) else data
        exported = os.path.join(data, "model.pt2")
        scripted = os.path.join(data, "scripted_model.pt")
        exported_model = False
        if os.path.isfile(pickled):
            allow_deserialization()
            try:
                # Full nn.Module pickle: PyTorch >= 2.6 defaults to weights_only=True.
                model = torch.load(pickled, map_location="cpu", weights_only=False)
            except Exception:
                model = torch.jit.load(pickled, map_location="cpu")
        elif os.path.isfile(exported):
            allow_deserialization()
            model = torch.export.load(exported).module()
            exported_model = True
        else:
            model = torch.jit.load(scripted, map_location="cpu")
        if not exported_model:
            model.eval()

        def predict_torch(frame: Any) -> Any:
            values = (
                frame.to_numpy(dtype=np.float32)
                if isinstance(frame, pd.DataFrame)
                else np.asarray(frame, dtype=np.float32)
            )
            with torch.no_grad():
                output = model(torch.from_numpy(values))
            if not isinstance(output, torch.Tensor):
                raise TypeError("The PyTorch model must return a single tensor.")
            result = pd.DataFrame(output.detach().cpu().numpy())
            if isinstance(frame, pd.DataFrame):
                result.index = frame.index
            return result

        return predict_torch

    def load_python_model(workdir: str) -> Callable[[Any], Any]:
        """Prefer ``mlflow.pyfunc``; fall back to native loaders without MLflow."""
        flavors: dict[str, Any] = dict(spec.get("flavors") or {})
        try:
            import mlflow.pyfunc as worker_pyfunc
        except ModuleNotFoundError as exc:
            missing = str(exc.name or "")
            if missing != "mlflow" and not missing.startswith("mlflow."):
                raise
            if "sklearn" in flavors:
                return native_sklearn(workdir, flavors)
            if "xgboost" in flavors:
                return native_xgboost(workdir, flavors)
            if "pytorch" in flavors:
                return native_pytorch(workdir, flavors)
            raise RuntimeError(
                "mlflow is not installed on this executor and the model's flavors "
                f"{sorted(flavors)} have no native loader (supported without mlflow: "
                "sklearn, xgboost, pytorch). Install mlflow on the executors."
            ) from exc
        model_config = spec.get("model_config")
        if model_config is not None:
            model = worker_pyfunc.load_model(workdir, model_config=model_config)
        else:
            model = worker_pyfunc.load_model(workdir)
        params = spec.get("params")

        def predict_pyfunc(frame: Any) -> Any:
            if params:
                return model.predict(frame, params=params)
            return model.predict(frame)

        return predict_pyfunc

    def python_model(iterator: Iterator[Any]) -> Iterator[Any]:
        names = [str(n) for n in spec.get("input_names") or []]
        workdir = tempfile.mkdtemp(prefix=WORKER_TEMP_PREFIX)
        try:
            with zipfile.ZipFile(io.BytesIO(spec["payload"])) as archive:
                archive.extractall(workdir)
            predict = load_python_model(workdir)
            for batch in iterator:
                frame = as_frame(batch, names)
                if len(frame) == 0:
                    continue
                yield to_series(predict(frame), len(frame))
        finally:
            shutil.rmtree(workdir, ignore_errors=True)

    def features(frame: Any) -> Any:
        """Evaluate the decoded feature stages (double precision, like the JVM)."""
        import numpy as np

        columns: dict[str, Any] = {}
        kinds = dict(spec.get("input_kinds") or {})
        for name in spec.get("input_cols") or []:
            values = frame[name]
            if kinds.get(name) == "vector":
                rows = [v.toArray() if hasattr(v, "toArray") else v for v in values]
                columns[name] = np.stack(
                    [np.asarray(v, dtype=np.float64) for v in rows]
                )
            else:
                columns[name] = values.to_numpy(dtype=np.float64, na_value=np.nan)
        skip = np.zeros(len(frame), dtype=bool)
        for stage in spec.get("stages") or []:
            if stage["type"] == "assembler":
                parts = []
                for name in stage["input_cols"]:
                    block = columns[name]
                    parts.append(block.reshape(len(frame), -1))
                assembled = np.hstack(parts) if parts else np.zeros((len(frame), 0))
                invalid = np.isnan(assembled).any(axis=1)
                mode = stage.get("handle_invalid") or "error"
                if invalid.any() and mode == "error":
                    raise ValueError(
                        "Encountered null or NaN in VectorAssembler inputs "
                        f"{stage['input_cols']} (handleInvalid=error)."
                    )
                if mode == "skip":
                    skip |= invalid
                columns[stage["output_col"]] = assembled
            elif stage["type"] == "scaler":
                values = columns[stage["input_col"]].reshape(len(frame), -1).copy()
                std = np.asarray(stage["std"], dtype=np.float64)
                mean = np.asarray(stage["mean"], dtype=np.float64)
                scale = np.zeros_like(std)
                nonzero = std != 0.0
                scale[nonzero] = 1.0 / std[nonzero]
                if stage.get("with_mean") and stage.get("with_std"):
                    values = (values - mean) * scale
                elif stage.get("with_mean"):
                    values = values - mean
                elif stage.get("with_std"):
                    values = values * scale
                columns[stage["output_col"]] = values
        matrix = columns[spec["features_col"]].reshape(len(frame), -1)
        return matrix, skip

    def xgboost_pipeline(iterator: Iterator[Any]) -> Iterator[Any]:
        import numpy as np
        import xgboost
        from scipy.special import expit, softmax

        booster = xgboost.Booster()
        booster.load_model(bytearray(spec["booster"]))
        best = booster.attr("best_iteration")
        iteration_range = (0, int(best) + 1) if best is not None else (0, 0)
        missing = float(spec.get("missing", float("nan")))
        classifier = spec.get("task") == "classifier"
        linear = '"name":"gblinear"' in booster.save_config().replace(" ", "")
        for batch in iterator:
            frame = as_frame(batch, [])
            if len(frame) == 0:
                continue
            matrix, skip = features(frame)
            # SparkXGB models cast feature vectors to float32 before predicting.
            data = matrix.astype(np.float32)
            predict_type = "margin" if classifier else "value"
            if linear:
                dmatrix = xgboost.DMatrix(data, missing=missing)
                output = booster.predict(
                    dmatrix,
                    output_margin=classifier,
                    iteration_range=iteration_range,
                    validate_features=False,
                )
            else:
                output = booster.inplace_predict(
                    data,
                    iteration_range=iteration_range,
                    predict_type=predict_type,
                    missing=missing,
                    validate_features=False,
                )
            if classifier:
                # Mirrors SparkXGBClassifierModel: argmax of class probabilities.
                if output.ndim == 1:
                    one = expit(output)
                    probs = np.vstack((1.0 - one, one)).transpose()
                else:
                    probs = softmax(output, axis=1)
                prediction = np.argmax(probs, axis=1).astype(np.float64)
            else:
                prediction = np.asarray(output, dtype=np.float64)
            if skip.any():
                prediction = prediction.astype(np.float64)
                prediction[skip] = np.nan
            yield to_series(prediction, len(frame))

    if spec.get("kind") == "xgboost_pipeline":
        return xgboost_pipeline
    return python_model
