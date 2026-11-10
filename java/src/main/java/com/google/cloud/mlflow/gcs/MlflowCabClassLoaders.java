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

import java.io.IOException;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;

/**
 * Classloader wiring shared by the Spark plugin (driver and executors) and the driver control
 * source. Works identically for {@code --jars}/{@code spark.jars}, {@code spark.addArtifact}, and
 * static {@code /usr/lib/spark/jars/} installs.
 */
public final class MlflowCabClassLoaders {

  static final String FS_IMPL_KEY = "fs." + MlflowCabConstants.SCHEME + ".impl";
  static final String AFS_IMPL_KEY = "fs.AbstractFileSystem." + MlflowCabConstants.SCHEME + ".impl";

  private MlflowCabClassLoaders() {}

  /** Returns the classloader that defined the provider classes. */
  static ClassLoader providerLoader() {
    ClassLoader own = MlflowCabClassLoaders.class.getClassLoader();
    return own == null ? ClassLoader.getSystemClassLoader() : own;
  }

  /** Returns true if {@code loader} resolves the provider's filesystem to this exact class. */
  static boolean resolvesProvider(ClassLoader loader) {
    if (loader == null) {
      return false;
    }
    try {
      return Class.forName(MlflowCabFileSystem.class.getName(), false, loader)
          == MlflowCabFileSystem.class;
    } catch (ClassNotFoundException | LinkageError e) {
      return false;
    }
  }

  /**
   * Makes {@code conf} resolve the provider classes and registers the scheme implementation keys
   * (never touching {@code fs.gs.*}).
   *
   * @return true if the configuration's classloader was augmented
   */
  static boolean install(Configuration conf) {
    boolean augmented = false;
    ClassLoader current = conf.getClassLoader();
    if (!resolvesProvider(current)) {
      ClassLoader base = current == null ? ClassLoader.getSystemClassLoader() : current;
      conf.setClassLoader(new MlflowCabDelegatingClassLoader(base, providerLoader()));
      augmented = true;
    }
    conf.set(FS_IMPL_KEY, MlflowCabFileSystem.class.getName());
    conf.set(AFS_IMPL_KEY, MlflowCabAbstractFileSystem.class.getName());
    return augmented;
  }

  /**
   * Driver-side installation on the SparkContext and active-session base configurations from
   * which every later job configuration is copied.
   */
  public static Map<String, String> installOnDriver() {
    return installOnDriver(null);
  }

  /**
   * Same as {@link #installOnDriver()} for a known driver SparkContext (typed as {@code Object} so
   * this class never links Spark), used while the context is still being constructed.
   */
  static Map<String, String> installOnDriver(Object sparkContext) {
    Map<String, String> result = new LinkedHashMap<>();
    int confs = 0;
    int augmented = 0;
    try {
      List<Configuration> bases = MlflowCabSparkBridge.driverBaseConfigurations(sparkContext);
      for (Configuration conf : bases) {
        confs++;
        if (install(conf)) {
          augmented++;
        }
      }
    } catch (LinkageError e) {
      result.put("sparkUnavailable", "true");
    }
    result.put("driverConfs", Integer.toString(confs));
    result.put("driverConfsAugmented", Integer.toString(augmented));
    result.put("serviceLoaderScheme", Boolean.toString(registerWithHadoopServiceLoader()));
    return result;
  }

  /**
   * Executor-side installation: ensures Hadoop's JVM-wide ServiceLoader registry knows the {@code
   * mlflow-gs} scheme when the JAR is in a child classloader (for example {@code --jars}), so
   * executor tasks resolve the filesystem even from configurations created outside task threads.
   */
  public static Map<String, String> installOnExecutor() {
    Map<String, String> result = new LinkedHashMap<>();
    result.put("serviceLoaderScheme", Boolean.toString(registerWithHadoopServiceLoader()));
    result.put("providerLoader", providerLoader().getClass().getName());
    return result;
  }

  /** Triggers Hadoop's filesystem ServiceLoader scan with the provider visible. */
  static boolean registerWithHadoopServiceLoader() {
    Thread thread = Thread.currentThread();
    ClassLoader previous = thread.getContextClassLoader();
    ClassLoader base = previous == null ? ClassLoader.getSystemClassLoader() : previous;
    try {
      if (!resolvesProvider(base)) {
        thread.setContextClassLoader(new MlflowCabDelegatingClassLoader(base, providerLoader()));
      }
      Configuration probe = new Configuration(false);
      probe.setClassLoader(thread.getContextClassLoader());
      Class<? extends FileSystem> resolved =
          FileSystem.getFileSystemClass(MlflowCabConstants.SCHEME, probe);
      return resolved == MlflowCabFileSystem.class;
    } catch (IOException | RuntimeException | LinkageError e) {
      return false;
    } finally {
      thread.setContextClassLoader(previous);
    }
  }
}
