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

"""SparkXGB pipelines shared by the Spark end-to-end tests.

``xgboost.spark`` models are written and read through a ``SparkContext``, so
Spark Connect tests consume models produced by a classic Spark job, as real
workflows do. Run ``python -m tests.spark_models <output-dir>`` to train the
pipelines with classic local Spark and save them with the stock (non-Managed)
``mlflow.spark.save_model`` plus their expected predictions.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

# name -> (input columns, task)
PIPELINES = {
    "xgb-1-stage": (["features"], "classifier"),
    "xgb-2-stage": (["x1", "x2"], "regressor"),
    "xgb-3-stage": (["x1", "x2"], "classifier"),
}


def with_inputs(frame: Any) -> Any:
    """Add the assembled ``features`` vector and the binary ``label_cls``."""
    from pyspark.ml.feature import VectorAssembler
    from pyspark.sql import functions

    assembled = VectorAssembler(inputCols=["x1", "x2"], outputCol="features")
    return assembled.transform(frame).withColumn(
        "label_cls", (functions.col("label") > 0.5).cast("double")
    )


def fit_pipelines(frame: Any) -> dict[str, Any]:
    """1-, 2-, and 3-stage pipelines ending in a SparkXGB model."""
    from pyspark.ml import Pipeline, PipelineModel
    from pyspark.ml.feature import StandardScaler, VectorAssembler
    from xgboost.spark import SparkXGBClassifier, SparkXGBRegressor

    data = with_inputs(frame).cache()
    common: dict[str, Any] = {"n_estimators": 12, "max_depth": 3, "num_workers": 1}
    one = SparkXGBClassifier(features_col="features", label_col="label_cls", **common)
    two = Pipeline(
        stages=[
            VectorAssembler(inputCols=["x1", "x2"], outputCol="assembled"),
            SparkXGBRegressor(features_col="assembled", label_col="label", **common),
        ]
    )
    three = Pipeline(
        stages=[
            VectorAssembler(inputCols=["x1", "x2"], outputCol="raw"),
            StandardScaler(
                inputCol="raw", outputCol="scaled", withMean=True, withStd=True
            ),
            SparkXGBClassifier(features_col="scaled", label_col="label_cls", **common),
        ]
    )
    models = {
        "xgb-1-stage": PipelineModel(stages=[one.fit(data)]),
        "xgb-2-stage": two.fit(data),
        "xgb-3-stage": three.fit(data),
    }
    data.unpersist()
    return models


def predictions(model: Any, frame: Any) -> list[float]:
    """``prediction`` column ordered by ``x1``."""
    rows = model.transform(with_inputs(frame)).orderBy("x1").collect()
    return [float(row.prediction) for row in rows]


def main(output: str) -> None:
    import mlflow
    from pyspark.sql import SparkSession

    from tests.spark_support import _features

    root = Path(output)
    spark = (
        SparkSession.builder.master("local[2]")
        .appName("mlflow-cab-sparkxgb-models")
        .config("spark.ui.enabled", "false")
        .config("spark.sql.shuffle.partitions", "4")
        .getOrCreate()
    )
    try:
        frame = _features(spark)
        expected: dict[str, list[float]] = {}
        for name, model in fit_pipelines(frame).items():
            mlflow.spark.save_model(model, str(root / name))
            expected[name] = predictions(model, frame)
        (root / "expected.json").write_text(json.dumps(expected))
    finally:
        spark.stop()


if __name__ == "__main__":
    main(sys.argv[1])
