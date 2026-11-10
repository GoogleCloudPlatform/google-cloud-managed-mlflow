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

import java.io.ByteArrayOutputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.lang.reflect.Method;
import java.util.ArrayList;
import java.util.List;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.security.UserGroupInformation;
import org.apache.spark.SparkContext;
import org.apache.spark.SparkContext$;
import org.apache.spark.SparkEnv;
import org.apache.spark.scheduler.SchedulerBackend;
import org.apache.spark.scheduler.cluster.CoarseGrainedSchedulerBackend;
import org.apache.spark.sql.SparkSession;

/**
 * Isolates every reference to Spark classes so the filesystem and token provider also work in
 * plain Hadoop processes where Spark is absent. All entry points degrade gracefully when Spark
 * classes cannot be linked.
 */
final class MlflowCabSparkBridge {

  static final String DRIVER_ID = "driver";

  private MlflowCabSparkBridge() {}

  /** Returns the Spark executor id of this JVM, {@code "driver"}, or null outside Spark. */
  static String executorIdOrNull() {
    try {
      SparkEnv env = SparkEnv.get();
      return env == null ? null : env.executorId();
    } catch (LinkageError e) {
      return null;
    }
  }

  /** Returns true on the Spark driver or in a non-Spark (standalone Hadoop) JVM. */
  static boolean isDriverOrStandalone() {
    String id = executorIdOrNull();
    return id == null || DRIVER_ID.equals(id);
  }

  /** Returns true if a Spark driver is running in this JVM. */
  static boolean isSparkDriver() {
    return DRIVER_ID.equals(executorIdOrNull());
  }

  /**
   * Returns the active or default session without throwing (the {@code active()} accessor throws
   * a checked exception when no session exists), or null.
   */
  static SparkSession activeSession() {
    try {
      scala.Option<SparkSession> session = SparkSession.getActiveSession();
      if (session.isEmpty()) {
        session = SparkSession.getDefaultSession();
      }
      return session.isDefined() ? session.get() : null;
    } catch (LinkageError | RuntimeException e) {
      return null;
    }
  }

  /**
   * Returns the running SparkContext without ever creating one (unlike {@code
   * SparkContext.getOrCreate()}, which would recurse when called while a context is still being
   * constructed, for example from a driver plugin), or null.
   */
  static SparkContext activeContext() {
    try {
      SparkSession session = activeSession();
      if (session != null) {
        return session.sparkContext();
      }
    } catch (LinkageError | RuntimeException e) {
      // Fall back to the context registry below.
    }
    try {
      scala.Option<SparkContext> active = SparkContext$.MODULE$.getActive();
      return active.isDefined() ? active.get() : null;
    } catch (LinkageError | RuntimeException e) {
      return null;
    }
  }

  /**
   * Broadcasts the driver's current-user Hadoop credentials (including CAB secret keys) to all
   * registered executors through Spark's encrypted scheduler RPC, and records them for executors
   * that register later.
   *
   * @return number of remote executors notified, 0 for single-JVM schedulers, -1 if not a driver
   */
  static int broadcastCurrentUserCredentials() throws IOException {
    if (!isSparkDriver()) {
      return -1;
    }
    try {
      SparkContext sc = activeContext();
      if (sc == null) {
        return 0;
      }
      SchedulerBackend backend = sc.schedulerBackend();
      if (!(backend instanceof CoarseGrainedSchedulerBackend)) {
        return 0;
      }
      ByteArrayOutputStream bytes = new ByteArrayOutputStream();
      try (DataOutputStream out = new DataOutputStream(bytes)) {
        UserGroupInformation.getCurrentUser().getCredentials().writeTokenStorageToStream(out);
      }
      ((CoarseGrainedSchedulerBackend) backend).updateDelegationTokens(bytes.toByteArray());
      return Math.max(0, sc.statusTracker().getExecutorInfos().length - 1);
    } catch (LinkageError e) {
      throw new IOException("Spark scheduler does not support credential broadcast", e);
    }
  }

  /** Returns the number of live remote executors, or -1 outside a Spark driver. */
  static int remoteExecutorCount() {
    if (!isSparkDriver()) {
      return -1;
    }
    try {
      SparkContext sc = activeContext();
      return sc == null ? -1 : Math.max(0, sc.statusTracker().getExecutorInfos().length - 1);
    } catch (LinkageError | RuntimeException e) {
      return -1;
    }
  }

  /**
   * Returns the driver-side Hadoop configurations that later per-job configurations are copied
   * from: the SparkContext configuration and the active session's shared-state configuration.
   *
   * @param explicit the context to use (for example the one handed to a driver plugin while it is
   *     still being constructed), or null to look up the running context
   */
  static List<Configuration> driverBaseConfigurations(Object explicit) {
    List<Configuration> result = new ArrayList<>();
    if (!isSparkDriver()) {
      return result;
    }
    try {
      SparkContext sc = explicit instanceof SparkContext ? (SparkContext) explicit : activeContext();
      if (sc == null) {
        return result;
      }
      result.add(sc.hadoopConfiguration());
    } catch (LinkageError | RuntimeException e) {
      return result;
    }
    try {
      Object session = activeSession();
      if (session == null) {
        return result;
      }
      Method sharedState = session.getClass().getMethod("sharedState");
      Object shared = sharedState.invoke(session);
      Method hadoopConf = shared.getClass().getMethod("hadoopConf");
      Object conf = hadoopConf.invoke(shared);
      if (conf instanceof Configuration && !result.contains(conf)) {
        result.add((Configuration) conf);
      }
    } catch (ReflectiveOperationException | LinkageError | RuntimeException e) {
      // No active classic session (e.g. plain SparkContext); the context configuration suffices.
    }
    return result;
  }

  /** Returns a fresh Hadoop configuration derived from the active session, or null. */
  static Configuration activeSessionHadoopConf() {
    if (!isSparkDriver()) {
      return null;
    }
    try {
      Object session = activeSession();
      if (session == null) {
        SparkContext sc = activeContext();
        return sc == null ? null : new Configuration(sc.hadoopConfiguration());
      }
      Method sessionState = session.getClass().getMethod("sessionState");
      Object state = sessionState.invoke(session);
      Method newHadoopConf = state.getClass().getMethod("newHadoopConf");
      Object conf = newHadoopConf.invoke(state);
      return conf instanceof Configuration ? (Configuration) conf : null;
    } catch (ReflectiveOperationException | LinkageError | RuntimeException e) {
      try {
        SparkContext sc = activeContext();
        return sc == null ? null : new Configuration(sc.hadoopConfiguration());
      } catch (LinkageError | RuntimeException e2) {
        return null;
      }
    }
  }
}
