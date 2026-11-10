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

import java.util.Collections;
import java.util.HashMap;
import java.util.Map;
import java.util.logging.Level;
import java.util.logging.Logger;
import org.apache.spark.SparkContext;
import org.apache.spark.api.plugin.DriverPlugin;
import org.apache.spark.api.plugin.ExecutorPlugin;
import org.apache.spark.api.plugin.PluginContext;
import org.apache.spark.api.plugin.SparkPlugin;

/**
 * Optional Spark plugin ({@code spark.plugins=com.google.cloud.mlflow.gcs.MlflowCabSparkPlugin})
 * that installs the provider classloader wiring on the driver <em>and</em> on every executor at
 * startup. Recommended together with {@code --jars} or a static {@code /usr/lib/spark/jars/}
 * install; sessions that receive the JAR through {@code spark.addArtifact} get the equivalent
 * driver wiring from the driver control source instead.
 */
public final class MlflowCabSparkPlugin implements SparkPlugin {

  private static final Logger LOG = Logger.getLogger(MlflowCabSparkPlugin.class.getName());

  static volatile boolean driverInitialized;
  static volatile boolean executorInitialized;

  @Override
  public DriverPlugin driverPlugin() {
    return new Driver();
  }

  @Override
  public ExecutorPlugin executorPlugin() {
    return new Executor();
  }

  /** Driver half: installs classloader wiring on the driver's base Hadoop configurations. */
  static final class Driver implements DriverPlugin {
    @Override
    public Map<String, String> init(SparkContext sc, PluginContext pluginContext) {
      Map<String, String> extra = new HashMap<>();
      try {
        MlflowCabClassLoaders.install(sc.hadoopConfiguration());
        Map<String, String> result = MlflowCabClassLoaders.installOnDriver(sc);
        extra.put("mlflow.cab.driver.serviceLoaderScheme", result.get("serviceLoaderScheme"));
      } catch (Exception | LinkageError e) {
        LOG.log(Level.WARNING, "MLflow CAB driver plugin init failed: {0}", MlflowCabRedaction.describe(e));
      }
      driverInitialized = true;
      return Collections.unmodifiableMap(extra);
    }
  }

  /** Executor half: registers the scheme with Hadoop under the executor's user classloader. */
  static final class Executor implements ExecutorPlugin {
    @Override
    public void init(PluginContext ctx, Map<String, String> extraConf) {
      try {
        MlflowCabClassLoaders.installOnExecutor();
      } catch (Exception | LinkageError e) {
        LOG.log(Level.WARNING, "MLflow CAB executor plugin init failed: {0}", MlflowCabRedaction.describe(e));
      }
      executorInitialized = true;
    }
  }
}
