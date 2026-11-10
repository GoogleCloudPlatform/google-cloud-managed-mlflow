/*
 * Copyright 2026 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.google.cloud.mlflow.gcs;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.PredictionModel;
import org.apache.spark.ml.Transformer;
import org.apache.spark.ml.feature.StandardScalerModel;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.ml.linalg.Vector;
import org.apache.spark.ml.linalg.Vectors;
import org.apache.spark.sql.Row;

/**
 * Row-wise, executor-side evaluator for fitted Spark ML pipelines.
 *
 * <p>Used by the Spark-flavor {@code mlflow.pyfunc.spark_udf} replacement so predictions run
 * inside a plain JVM UDF: the pipeline is loaded once on the driver (one READ_ONLY CAB scope) and
 * shipped inside the task closure, so executors never spawn local Spark sessions, never contact
 * MLflow, and never stage model files on local disk. Stage semantics mirror Spark ML's own
 * {@code transform} implementations.
 */
final class MlflowCabLocalPipeline implements Serializable {

  private static final long serialVersionUID = 1L;

  /** One row-wise stage. */
  interface Stage extends Serializable {
    void apply(Map<String, Object> values);

    String describe();
  }

  private final Stage[] stages;
  private final String[] inputCols;
  private final String outputCol;

  private MlflowCabLocalPipeline(List<Stage> stages, String[] inputCols, String outputCol) {
    this.stages = stages.toArray(new Stage[0]);
    this.inputCols = inputCols;
    this.outputCol = outputCol;
  }

  /** Builds an evaluator, failing with a descriptive message for unsupported stages. */
  static MlflowCabLocalPipeline from(PipelineModel model) {
    List<Stage> stages = new ArrayList<>();
    Set<String> produced = new LinkedHashSet<>();
    Set<String> required = new LinkedHashSet<>();
    String output = null;
    for (Transformer transformer : model.stages()) {
      if (transformer instanceof VectorAssembler) {
        VectorAssembler assembler = (VectorAssembler) transformer;
        for (String col : assembler.getInputCols()) {
          if (!produced.contains(col)) {
            required.add(col);
          }
        }
        stages.add(
            new AssemblerStage(
                assembler.getInputCols(), assembler.getOutputCol(), assembler.getHandleInvalid()));
        produced.add(assembler.getOutputCol());
      } else if (transformer instanceof StandardScalerModel) {
        StandardScalerModel scaler = (StandardScalerModel) transformer;
        if (!produced.contains(scaler.getInputCol())) {
          required.add(scaler.getInputCol());
        }
        stages.add(
            new ScalerStage(
                scaler.mean().toArray(),
                scaler.std().toArray(),
                scaler.getWithMean(),
                scaler.getWithStd(),
                scaler.getInputCol(),
                scaler.getOutputCol()));
        produced.add(scaler.getOutputCol());
      } else if (transformer instanceof PredictionModel) {
        PredictionModel<?, ?> predictor = (PredictionModel<?, ?>) transformer;
        if (!produced.contains(predictor.getFeaturesCol())) {
          required.add(predictor.getFeaturesCol());
        }
        stages.add(
            new PredictorStage(predictor, predictor.getFeaturesCol(), predictor.getPredictionCol()));
        produced.add(predictor.getPredictionCol());
        output = predictor.getPredictionCol();
      } else {
        throw new UnsupportedOperationException(
            "Row-wise evaluation does not support stage "
                + transformer.getClass().getName()
                + "; supported stages are VectorAssembler, StandardScalerModel, and prediction "
                + "models. Use mlflow.spark.load_model(uri).transform(df) instead.");
      }
    }
    if (output == null) {
      throw new UnsupportedOperationException(
          "Pipeline has no prediction stage; row-wise evaluation requires a prediction model");
    }
    return new MlflowCabLocalPipeline(stages, required.toArray(new String[0]), output);
  }

  String[] inputCols() {
    return inputCols.clone();
  }

  List<String> stageNames() {
    List<String> names = new ArrayList<>();
    for (Stage stage : stages) {
      names.add(stage.describe());
    }
    return names;
  }

  /** Evaluates one input row (a struct whose fields are the pipeline input columns). */
  Double predict(Row row) {
    if (row == null) {
      return null;
    }
    Map<String, Object> values = new HashMap<>();
    for (int i = 0; i < inputCols.length; i++) {
      Object value;
      int index = fieldIndex(row, inputCols[i]);
      value = index >= 0 ? row.get(index) : (i < row.length() ? row.get(i) : null);
      values.put(inputCols[i], value);
    }
    for (Stage stage : stages) {
      stage.apply(values);
      if (values.containsKey(SKIP_MARKER)) {
        return null;
      }
    }
    Object prediction = values.get(outputCol);
    return prediction instanceof Number ? ((Number) prediction).doubleValue() : null;
  }

  private static int fieldIndex(Row row, String name) {
    try {
      return row.schema() == null ? -1 : row.fieldIndex(name);
    } catch (IllegalArgumentException | UnsupportedOperationException e) {
      return -1;
    }
  }

  static final String SKIP_MARKER = "\u0000mlflow-cab-skip";

  /** Mirrors VectorAssembler for numeric, boolean, and vector inputs. */
  static final class AssemblerStage implements Stage {
    private static final long serialVersionUID = 1L;
    private final String[] inputCols;
    private final String outputCol;
    private final String handleInvalid;

    AssemblerStage(String[] inputCols, String outputCol, String handleInvalid) {
      this.inputCols = inputCols.clone();
      this.outputCol = outputCol;
      this.handleInvalid = handleInvalid == null ? "error" : handleInvalid;
    }

    @Override
    public void apply(Map<String, Object> values) {
      List<Double> out = new ArrayList<>();
      for (String col : inputCols) {
        Object value = values.get(col);
        if (value instanceof Vector) {
          for (double v : ((Vector) value).toArray()) {
            out.add(v);
          }
        } else if (value instanceof Number) {
          double v = ((Number) value).doubleValue();
          if (Double.isNaN(v) && !"keep".equals(handleInvalid)) {
            invalid(values, col);
            return;
          }
          out.add(v);
        } else if (value instanceof Boolean) {
          out.add(((Boolean) value) ? 1.0 : 0.0);
        } else if (value == null) {
          if ("keep".equals(handleInvalid)) {
            out.add(Double.NaN);
          } else {
            invalid(values, col);
            return;
          }
        } else {
          throw new IllegalArgumentException(
              "VectorAssembler input column " + col + " has unsupported type "
                  + value.getClass().getName());
        }
      }
      double[] arr = new double[out.size()];
      for (int i = 0; i < arr.length; i++) {
        arr[i] = out.get(i);
      }
      values.put(outputCol, Vectors.dense(arr));
    }

    private void invalid(Map<String, Object> values, String col) {
      if ("skip".equals(handleInvalid)) {
        values.put(SKIP_MARKER, Boolean.TRUE);
        return;
      }
      throw new IllegalArgumentException(
          "Encountered null or NaN in VectorAssembler input column " + col
              + " (handleInvalid=" + handleInvalid + ")");
    }

    @Override
    public String describe() {
      return "VectorAssembler";
    }
  }

  /** Mirrors StandardScalerModel's shift/scale transform (scale = 1/std, or 0 when std == 0). */
  static final class ScalerStage implements Stage {
    private static final long serialVersionUID = 1L;
    private final double[] shift;
    private final double[] scale;
    private final boolean withMean;
    private final boolean withStd;
    private final String inputCol;
    private final String outputCol;

    ScalerStage(
        double[] mean,
        double[] std,
        boolean withMean,
        boolean withStd,
        String inputCol,
        String outputCol) {
      this.shift = mean.clone();
      this.scale = new double[std.length];
      for (int i = 0; i < std.length; i++) {
        this.scale[i] = std[i] != 0.0 ? 1.0 / std[i] : 0.0;
      }
      this.withMean = withMean;
      this.withStd = withStd;
      this.inputCol = inputCol;
      this.outputCol = outputCol;
    }

    @Override
    public void apply(Map<String, Object> values) {
      Object value = values.get(inputCol);
      if (!(value instanceof Vector)) {
        throw new IllegalArgumentException("StandardScaler input column " + inputCol + " is not a vector");
      }
      double[] arr = ((Vector) value).toArray();
      for (int i = 0; i < arr.length; i++) {
        if (withMean && withStd) {
          arr[i] = (arr[i] - shift[i]) * scale[i];
        } else if (withMean) {
          arr[i] = arr[i] - shift[i];
        } else if (withStd) {
          arr[i] = arr[i] * scale[i];
        }
      }
      values.put(outputCol, Vectors.dense(arr));
    }

    @Override
    public String describe() {
      return "StandardScalerModel";
    }
  }

  /** Delegates to the fitted model's public single-row {@code predict(Vector)} API. */
  static final class PredictorStage implements Stage {
    private static final long serialVersionUID = 1L;
    private final PredictionModel<?, ?> model;
    private final String featuresCol;
    private final String predictionCol;

    PredictorStage(PredictionModel<?, ?> model, String featuresCol, String predictionCol) {
      this.model = model;
      this.featuresCol = featuresCol;
      this.predictionCol = predictionCol;
    }

    @Override
    @SuppressWarnings("unchecked")
    public void apply(Map<String, Object> values) {
      Object features = values.get(featuresCol);
      if (!(features instanceof Vector)) {
        throw new IllegalArgumentException("Prediction features column " + featuresCol + " is not a vector");
      }
      double prediction = ((PredictionModel<Vector, ?>) model).predict((Vector) features);
      values.put(predictionCol, prediction);
    }

    @Override
    public String describe() {
      return model.getClass().getSimpleName();
    }
  }
}
