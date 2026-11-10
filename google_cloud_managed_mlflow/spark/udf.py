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

"""``mlflow.pyfunc.spark_udf`` replacements for Managed MLflow models.

* Python flavors (sklearn, xgboost, pytorch, ...): the model is downloaded once
  on the client with a ``READ_ONLY`` CAB token and shipped to Python workers
  inside the UDF; workers never fetch artifacts or hold credentials, and their
  temporary extraction directory is removed when each task finishes.
* Spark flavor, JVM stages only: the fitted pipeline is loaded once on the driver
  through ``mlflow-gs://`` and evaluated row-wise inside a JVM UDF.
* Spark flavor with Python ``SparkXGB*`` stages: the saved stage metadata, scaler
  statistics, and booster are decoded on the client (no Py4J and no
  ``PipelineModel.load``), so the same pandas UDF works on classic Spark and
  Spark Connect.

Executor-side code lives in :mod:`google_cloud_managed_mlflow.spark._udf_worker`
and is compiled here under a neutral module name: the pickled UDFs reference
neither this plugin nor ``mlflow``, so executors without either package (for
example serverless runtimes) can run them.
"""

from __future__ import annotations

import builtins
import dataclasses
import functools
import importlib.resources
import io
import json
import math
import os
import posixpath
import re
import shutil
import tempfile
import uuid
import zipfile
from collections.abc import Callable, Iterator
from typing import Any

import mlflow
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE

from google_cloud_managed_mlflow.spark import hooks, runtime

MAX_INLINE_MODEL_BYTES = 512 * 1024 * 1024
SINGLE_INPUT_FIELD = "__mlflow_cab_input__"
WORKER_MODULE_NAME = "mlflow_cab_udf_worker"
_WORKER_FILENAME = "<mlflow-cab-udf-worker>"
_IMPLICIT_FEATURES = "__mlflow_cab_features__"
_JVM_STAGE_PREFIX = "org.apache.spark."
_XGBOOST_MODELS = {
    "SparkXGBClassifierModel": "classifier",
    "SparkXGBRegressorModel": "regressor",
    "SparkXGBRankerModel": "regressor",
}


@functools.lru_cache(maxsize=1)
def _worker_namespace() -> dict[str, Any]:
    source = (
        importlib.resources.files("google_cloud_managed_mlflow.spark")
        .joinpath("_udf_worker.py")
        .read_text(encoding="utf-8")
    )
    namespace: dict[str, Any] = {
        "__name__": WORKER_MODULE_NAME,
        "__builtins__": builtins,
    }
    code = compile(source, _WORKER_FILENAME, "exec", dont_inherit=True)
    exec(code, namespace)  # packaged plugin source, compiled under a neutral name
    return namespace


def worker_predict_fn(spec: dict[str, Any]) -> Callable[[Iterator[Any]], Iterator[Any]]:
    """Build the executor-side UDF body for ``spec`` (pickles by value)."""
    factory: Callable[[dict[str, Any]], Callable[[Iterator[Any]], Iterator[Any]]]
    factory = _worker_namespace()["make_predict_fn"]
    return factory(spec)


def _pandas_udf(spec: dict[str, Any], spark_type: Any) -> Any:
    import pandas as pd
    from pyspark.sql.functions import pandas_udf

    predict = worker_predict_fn(spec)
    # Scalar-iterator pandas UDF over one struct column: Iterator[DataFrame].
    predict.__annotations__ = {
        "iterator": Iterator[pd.DataFrame],
        "return": Iterator[pd.Series],
    }
    return pandas_udf(predict, returnType=spark_type)


def _resolve_result_type(meta: Any, result_type: Any) -> Any:
    from pyspark.sql.types import DataType, DoubleType

    if result_type is None:
        try:
            from mlflow.pyfunc import _infer_spark_udf_return_type

            schema = meta.get_output_schema()
            if schema is not None:
                return _infer_spark_udf_return_type(schema)
        except Exception:
            pass
        return DoubleType()
    if isinstance(result_type, DataType):
        return result_type
    from mlflow.pyfunc import _parse_spark_datatype

    return _parse_spark_datatype(str(result_type))


def _zip_directory(path: str) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for root, _, files in os.walk(path):
            for name in sorted(files):
                full = os.path.join(root, name)
                archive.write(full, os.path.relpath(full, path))
    return buffer.getvalue()


def _column(value: Any) -> Any:
    from pyspark.sql import functions

    return functions.col(value) if isinstance(value, str) else value


def _plain(value: Any) -> Any:
    """Deep copy of MLmodel data restricted to JSON-compatible primitives."""
    return json.loads(json.dumps(value, default=str))


def pickle_deserialization_allowed() -> bool:
    """Client-side value of MLflow's ``MLFLOW_ALLOW_PICKLE_DESERIALIZATION``."""
    raw = os.environ.get("MLFLOW_ALLOW_PICKLE_DESERIALIZATION", "true")
    return raw.strip().lower() not in ("false", "0", "no")


def _native_sklearn_payload(local_dir: str, flavors: dict[str, Any]) -> bytes | None:
    """Cloudpickle of a ``skops`` sklearn model for executors without skops."""
    conf = flavors.get("sklearn") or {}
    if conf.get("serialization_format") != "skops":
        return None
    try:
        import cloudpickle
        import mlflow.sklearn

        model = mlflow.sklearn.load_model(local_dir)
        data: bytes = cloudpickle.dumps(model)
    except Exception:  # pragma: no cover - executors may still have skops/mlflow
        return None
    return data


def python_flavor_udf(
    spark: Any,
    model_uri: str,
    location: hooks.ModelLocation,
    meta: Any,
    result_type: Any,
    params: dict[str, Any] | None,
    model_config: dict[str, Any] | None,
) -> Any:
    """Spark UDF for Python-flavor models with a single client-side download."""
    from pyspark.sql import functions

    spark_type = _resolve_result_type(meta, result_type)
    flavors = _plain(meta.flavors)
    staging = tempfile.mkdtemp(prefix="mlflow-cab-udf-src-")
    try:
        local = mlflow.artifacts.download_artifacts(
            artifact_uri=model_uri, dst_path=staging
        )
        payload = _zip_directory(local)
        native = _native_sklearn_payload(local, flavors)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    if len(payload) > MAX_INLINE_MODEL_BYTES:
        raise MlflowException(
            f"Model archive ({len(payload)} bytes) is too large to ship inside a UDF.",
            error_code=INVALID_PARAMETER_VALUE,
        )
    input_schema = meta.get_input_schema()
    input_names = list(input_schema.input_names()) if input_schema is not None else []
    spec: dict[str, Any] = {
        "kind": "python_model",
        "payload": payload,
        "flavors": flavors,
        "input_names": [str(n) for n in input_names],
        "result_kind": spark_type.simpleString(),
        "params": _plain(params) if params else None,
        "model_config": _plain(model_config) if model_config is not None else None,
        "allow_pickle": pickle_deserialization_allowed(),
    }
    if native is not None:
        spec["native_sklearn"] = native
    udf = _pandas_udf(spec, spark_type)

    def apply(*cols: Any) -> Any:
        if not cols:
            if not input_names:
                raise MlflowException(
                    "Cannot apply the UDF without columns: the model has no "
                    "input names.",
                    error_code=INVALID_PARAMETER_VALUE,
                )
            cols = tuple(input_names)
        if len(cols) == 1:
            struct_col = functions.struct(_column(cols[0]).alias(SINGLE_INPUT_FIELD))
        else:
            names = (
                [str(n) for n in input_names]
                if len(input_names) >= len(cols)
                else [str(i) for i in range(len(cols))]
            )
            struct_col = functions.struct(
                *[_column(c).alias(names[i]) for i, c in enumerate(cols)]
            )
        return udf(struct_col)

    apply.metadata = meta  # type: ignore[attr-defined]
    return apply


# ----------------------------------------------------------- Spark ML pipelines


@dataclasses.dataclass(frozen=True)
class SavedStage:
    """Metadata of one saved Spark ML stage (``metadata/part-*`` JSON)."""

    directory: str
    class_name: str
    params: dict[str, Any]

    @property
    def simple_name(self) -> str:
        """Class name without its package."""
        return self.class_name.rsplit(".", 1)[-1]

    @property
    def is_jvm(self) -> bool:
        """True for stages implemented in the JVM (Spark ML)."""
        return self.class_name.startswith(_JVM_STAGE_PREFIX)


class PipelineReader:
    """Reads saved Spark ML metadata with the model's ``READ_ONLY`` CAB scope."""

    def __init__(self, data_uri: str, location: hooks.ModelLocation) -> None:
        self.data_uri = data_uri.rstrip("/")
        self._repo = hooks.repository_for(location, self.data_uri)
        self._tmp = tempfile.mkdtemp(prefix="mlflow-cab-sparkml-")

    def close(self) -> None:
        """Remove downloaded metadata files."""
        shutil.rmtree(self._tmp, ignore_errors=True)

    def __enter__(self) -> PipelineReader:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def _files(self, directory: str, pattern: str) -> list[str]:
        infos = self._repo.list_artifacts(directory)
        names = [posixpath.basename(i.path) for i in infos if not i.is_dir]
        return sorted(n for n in names if re.match(pattern, n))

    def _download(self, relative: str) -> str:
        local: str = self._repo.download_artifacts(relative, dst_path=self._tmp)
        return local

    def text_parts(self, directory: str) -> list[str]:
        """Lines of every ``part-*`` text file under ``directory``, in order."""
        lines: list[str] = []
        for name in self._files(directory, r"^part-\d+"):
            if name.endswith(".crc"):
                continue
            path = self._download(posixpath.join(directory, name))
            with open(path, encoding="utf-8") as handle:
                text = handle.read()
            text = text.replace("\r\n", "\n").replace("\r", "\n")
            lines.extend(line for line in text.split("\n") if line)
        return lines

    def metadata(self, directory: str) -> dict[str, Any]:
        """Parsed ``metadata`` JSON of the stage or pipeline at ``directory``."""
        lines = self.text_parts(posixpath.join(directory, "metadata"))
        if not lines:
            raise MlflowException(
                f"No Spark ML metadata found under {self.data_uri}/{directory}.",
                error_code=INVALID_PARAMETER_VALUE,
            )
        parsed: dict[str, Any] = json.loads(lines[0])
        return parsed

    def stages(self) -> list[SavedStage]:
        """Stages in pipeline order (a bare model is a single stage)."""
        root = self.metadata("")
        if not str(root.get("class", "")).endswith("PipelineModel"):
            return [_saved_stage("", root)]
        uids = list((root.get("paramMap") or {}).get("stageUids") or [])
        listed = [i.path for i in self._repo.list_artifacts("stages") if i.is_dir]
        by_uid: dict[str, str] = {}
        for path in listed:
            base = posixpath.basename(path.rstrip("/"))
            _, _, uid = base.partition("_")
            by_uid[uid] = posixpath.join("stages", base)
        result = []
        for uid in uids:
            directory = by_uid.get(uid)
            if directory is None:
                raise MlflowException(
                    f"Saved pipeline stage {uid} is missing under {self.data_uri}.",
                    error_code=INVALID_PARAMETER_VALUE,
                )
            result.append(_saved_stage(directory, self.metadata(directory)))
        return result

    def scaler_statistics(self, directory: str) -> tuple[list[float], list[float]]:
        """``(mean, std)`` of a saved ``StandardScalerModel``."""
        import pyarrow.parquet as pq

        data_dir = posixpath.join(directory, "data")
        for name in self._files(data_dir, r"^part-.*\.parquet$"):
            table = pq.read_table(self._download(posixpath.join(data_dir, name)))
            for row in table.to_pylist():
                return _dense(row["mean"]), _dense(row["std"])
        raise MlflowException(
            f"StandardScalerModel statistics are missing under {directory}.",
            error_code=INVALID_PARAMETER_VALUE,
        )

    def booster(self, directory: str) -> bytes:
        """Raw XGBoost JSON model written by the ``SparkXGB*Model`` writer."""
        text = "".join(self.text_parts(posixpath.join(directory, "model")))
        if not text:
            raise MlflowException(
                f"XGBoost model data is missing under {directory}.",
                error_code=INVALID_PARAMETER_VALUE,
            )
        return text.encode("utf-8")


def _saved_stage(directory: str, metadata: dict[str, Any]) -> SavedStage:
    params = dict(metadata.get("defaultParamMap") or {})
    params.update(metadata.get("paramMap") or {})
    return SavedStage(directory, str(metadata.get("class", "")), params)


def _dense(vector: dict[str, Any]) -> list[float]:
    values = [float(v) for v in vector.get("values") or []]
    if int(vector.get("type", 1)) == 1:
        return values
    dense = [0.0] * int(vector.get("size") or 0)
    for index, value in zip(vector.get("indices") or [], values, strict=True):
        dense[int(index)] = value
    return dense


def read_pipeline_stages(
    location: hooks.ModelLocation, meta: Any
) -> tuple[str, list[SavedStage]]:
    """Return the Spark model data URI and its saved stages (client-side)."""
    model_data = meta.flavors.get("spark", {}).get(
        "model_data", hooks.SPARK_MODEL_SUBDIR
    )
    data_uri = f"{location.gcs_uri}/{model_data}"
    with PipelineReader(data_uri, location) as reader:
        return data_uri, reader.stages()


def has_python_stages(stages: list[SavedStage]) -> bool:
    """True if any stage is implemented in Python (for example ``SparkXGB*``)."""
    return any(not stage.is_jvm for stage in stages)


@dataclasses.dataclass(frozen=True)
class XGBoostPipeline:
    """Client-decoded ``[VectorAssembler] [StandardScalerModel] SparkXGB*Model``."""

    spec: dict[str, Any]
    input_cols: list[str]
    vector_inputs: frozenset[str]
    prediction_col: str


def decode_xgboost_pipeline(
    data_uri: str,
    location: hooks.ModelLocation,
    stages: list[SavedStage],
    result_kind: str = "double",
) -> XGBoostPipeline:
    """Decode a pipeline ending in a ``SparkXGB*Model`` into a worker spec."""
    if not stages or stages[-1].simple_name not in _XGBOOST_MODELS:
        names = [s.simple_name for s in stages]
        raise MlflowException(
            f"Unsupported pipeline stages {names}: Python-stage pipelines must end "
            "with SparkXGBClassifierModel, SparkXGBRegressorModel, or "
            "SparkXGBRankerModel and may only be preceded by VectorAssembler and "
            "StandardScalerModel. Use mlflow.spark.load_model(uri).transform(df) on "
            "classic Spark instead.",
            error_code=INVALID_PARAMETER_VALUE,
        )
    produced: set[str] = set()
    required: list[str] = []
    vectors: set[str] = set()
    decoded: list[dict[str, Any]] = []

    def need(column: str, vector: bool) -> None:
        if column not in produced and column not in required:
            required.append(column)
            if vector:
                vectors.add(column)

    with PipelineReader(data_uri, location) as reader:
        for stage in stages[:-1]:
            params = stage.params
            if stage.simple_name == "VectorAssembler":
                inputs = [str(c) for c in params.get("inputCols") or []]
                for column in inputs:
                    need(column, False)
                decoded.append(
                    {
                        "type": "assembler",
                        "input_cols": inputs,
                        "output_col": str(params["outputCol"]),
                        "handle_invalid": str(params.get("handleInvalid") or "error"),
                    }
                )
                produced.add(str(params["outputCol"]))
            elif stage.simple_name == "StandardScalerModel":
                mean, std = reader.scaler_statistics(stage.directory)
                need(str(params["inputCol"]), True)
                decoded.append(
                    {
                        "type": "scaler",
                        "input_col": str(params["inputCol"]),
                        "output_col": str(params["outputCol"]),
                        "with_mean": bool(params.get("withMean", False)),
                        "with_std": bool(params.get("withStd", True)),
                        "mean": mean,
                        "std": std,
                    }
                )
                produced.add(str(params["outputCol"]))
            else:
                raise MlflowException(
                    f"Stage {stage.class_name} is not supported before a SparkXGB "
                    "model in a Spark UDF (supported: VectorAssembler, "
                    "StandardScalerModel).",
                    error_code=INVALID_PARAMETER_VALUE,
                )
        final = stages[-1]
        params = final.params
        if params.get("base_margin_col"):
            raise MlflowException(
                "SparkXGB models with base_margin_col are not supported in a "
                "Spark UDF.",
                error_code=INVALID_PARAMETER_VALUE,
            )
        features_cols = [str(c) for c in params.get("features_cols") or []]
        if features_cols:
            for column in features_cols:
                need(column, False)
            decoded.append(
                {
                    "type": "assembler",
                    "input_cols": features_cols,
                    "output_col": _IMPLICIT_FEATURES,
                    "handle_invalid": "keep",
                }
            )
            features_col = _IMPLICIT_FEATURES
        else:
            features_col = str(params.get("featuresCol") or "features")
            need(features_col, True)
        booster = reader.booster(final.directory)
    missing = params.get("missing")
    spec = {
        "kind": "xgboost_pipeline",
        "booster": booster,
        "task": _XGBOOST_MODELS[final.simple_name],
        "missing": math.nan if missing is None else float(missing),
        "input_cols": required,
        "input_kinds": {c: ("vector" if c in vectors else "scalar") for c in required},
        "stages": decoded,
        "features_col": features_col,
        "result_kind": result_kind,
    }
    return XGBoostPipeline(
        spec=spec,
        input_cols=list(required),
        vector_inputs=frozenset(vectors),
        prediction_col=str(params.get("predictionCol") or "prediction"),
    )


def _xgboost_struct(pipeline: XGBoostPipeline, cols: tuple[Any, ...]) -> Any:
    from pyspark.ml.functions import vector_to_array
    from pyspark.sql import functions

    required = pipeline.input_cols
    if not cols:
        cols = tuple(required)
    if len(cols) == 1 and len(required) != 1:
        return _column(cols[0])
    if len(cols) != len(required):
        raise MlflowException(
            f"Expected {len(required)} input columns {required}, got {len(cols)}.",
            error_code=INVALID_PARAMETER_VALUE,
        )
    fields = []
    for value, name in zip(cols, required, strict=True):
        column = _column(value)
        if name in pipeline.vector_inputs:
            column = vector_to_array(column, "float64")
        fields.append(column.alias(name))
    return functions.struct(*fields)


def xgboost_pipeline_udf(pipeline: XGBoostPipeline, spark_type: Any) -> Any:
    """Column function evaluating a decoded SparkXGB pipeline on executors."""
    udf = _pandas_udf(pipeline.spec, spark_type)

    def apply(*cols: Any) -> Any:
        return udf(_xgboost_struct(pipeline, cols))

    apply.input_columns = list(pipeline.input_cols)  # type: ignore[attr-defined]
    return apply


def spark_flavor_udf(
    spark: Any,
    model_uri: str,
    location: hooks.ModelLocation,
    meta: Any,
    result_type: Any,
) -> Any:
    """Spark UDF for Spark ML pipelines (JVM row-wise or client-decoded XGBoost)."""
    from pyspark.sql import functions

    spark_type = _resolve_result_type(meta, result_type)
    data_uri, stages = read_pipeline_stages(location, meta)
    if has_python_stages(stages):
        pipeline = decode_xgboost_pipeline(
            data_uri, location, stages, spark_type.simpleString()
        )
        decoded = xgboost_pipeline_udf(pipeline, spark_type)
        decoded.metadata = meta
        return decoded
    path = hooks.cab_path_for(spark, data_uri, location.read_scope)
    name = f"mlflow_cab_spark_model_{uuid.uuid4().hex[:16]}"
    result = runtime.control(spark, "register_model_udf", name=name, modelPath=path)
    if result.get("ok") != "true":
        raise MlflowException(
            "Unable to build a Spark-flavor UDF for this pipeline: "
            f"{result.get('error', '')}",
            error_code=INVALID_PARAMETER_VALUE,
        )
    input_cols = [c for c in result.get("inputCols", "").split(",") if c]

    def apply(*cols: Any) -> Any:
        if not cols:
            cols = tuple(input_cols)
        if len(cols) == 1 and len(input_cols) != 1:
            struct_col = _column(cols[0])
        else:
            if len(cols) != len(input_cols):
                raise MlflowException(
                    f"Expected {len(input_cols)} input columns {input_cols}, "
                    f"got {len(cols)}.",
                    error_code=INVALID_PARAMETER_VALUE,
                )
            struct_col = functions.struct(
                *[_column(c).alias(n) for c, n in zip(cols, input_cols, strict=True)]
            )
        return functions.call_function(name, struct_col).cast(spark_type)

    apply.metadata = meta  # type: ignore[attr-defined]
    apply.input_columns = input_cols  # type: ignore[attr-defined]
    return apply


def connect_pipeline_model(pipeline: XGBoostPipeline) -> Any:
    """A Spark ML ``Model`` for Spark Connect that appends the prediction column.

    ``xgboost.spark`` reads and writes models through ``SparkContext``, which
    Spark Connect clients do not have; this model evaluates the same decoded
    pipeline with a pandas UDF instead.
    """
    from pyspark.ml import Model
    from pyspark.sql.types import DoubleType

    apply = xgboost_pipeline_udf(pipeline, DoubleType())

    class ManagedXGBoostPipelineModel(Model):
        """Transforms DataFrames by appending the SparkXGB prediction column."""

        def _transform(self, dataset: Any) -> Any:
            return dataset.withColumn(pipeline.prediction_col, apply())

    model = ManagedXGBoostPipelineModel()
    model.input_columns = list(pipeline.input_cols)
    model.prediction_col = pipeline.prediction_col
    return model


class LocalPipelinePredictor:
    """``python_function`` implementation scoring a decoded pipeline in-process."""

    def __init__(self, pipeline: XGBoostPipeline) -> None:
        self._pipeline = pipeline
        self._predict = worker_predict_fn(pipeline.spec)

    def predict(self, data: Any, params: dict[str, Any] | None = None) -> Any:
        """Score ``data`` (pandas DataFrame) and return a numpy array."""
        import numpy as np
        import pandas as pd

        del params
        frame = data if isinstance(data, pd.DataFrame) else pd.DataFrame(data)
        batch = frame[self._pipeline.input_cols].reset_index(drop=True)
        outputs = list(self._predict(iter([batch])))
        if not outputs:
            return np.asarray([], dtype=np.float64)
        return pd.concat(outputs, ignore_index=True).to_numpy()
