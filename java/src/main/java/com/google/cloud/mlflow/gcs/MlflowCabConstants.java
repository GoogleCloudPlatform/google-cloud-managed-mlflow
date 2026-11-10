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

/** Configuration keys, defaults, and identifiers shared by the MLflow CAB Hadoop integration. */
public final class MlflowCabConstants {

  /** Hadoop filesystem scheme served by {@link MlflowCabFileSystem}. */
  public static final String SCHEME = "mlflow-gs";

  /** Scheme of the wrapped Cloud Storage connector filesystem. */
  public static final String GCS_SCHEME = "gs";

  /** Release version embedded in the User-Agent of control-plane requests. */
  public static final String VERSION = "0.2.0";

  /** Product token used in the User-Agent header of control-plane requests. */
  public static final String USER_AGENT_PRODUCT = "mlflow-gcs-cab-provider";

  /**
   * Public operator environment variable that shortens (or lengthens) the lead window before
   * credential expiry at which the driver renews CAB tokens.
   */
  public static final String ENV_REFRESH_MARGIN_SECONDS = "MLFLOW_CAB_REFRESH_MARGIN_SECONDS";

  /** Non-secret Hadoop/Spark equivalent of {@link #ENV_REFRESH_MARGIN_SECONDS}. */
  public static final String CONF_REFRESH_MARGIN_SECONDS = "fs.mlflow.cab.refresh.margin.seconds";

  /** Default renewal lead window (seconds) before CAB token expiry. */
  public static final long DEFAULT_REFRESH_MARGIN_SECONDS = 300L;

  /** Public environment variable that opts into a custom or loopback MLflow endpoint origin. */
  public static final String ENV_ENDPOINT_OVERRIDE = "MLFLOW_GCP_ENDPOINT_OVERRIDE";

  /** Non-secret Hadoop/Spark equivalent of {@link #ENV_ENDPOINT_OVERRIDE}. */
  public static final String CONF_ENDPOINT_OVERRIDE = "fs.mlflow.cab.endpoint.override";

  /** Inner-configuration key naming the CAB scope served by a filesystem instance. */
  public static final String CONF_ACTIVE_SCOPE = "fs.mlflow.cab.active.scope";

  /** Optional inner delegate filesystem class (defaults to the Cloud Storage connector). */
  public static final String CONF_DELEGATE_IMPL = "fs.mlflow.cab.delegate.impl";

  /** Maximum time an executor waits for a driver-broadcast token before failing. */
  public static final String CONF_EXECUTOR_TOKEN_WAIT_MS = "fs.mlflow.cab.executor.token.wait.ms";

  /** Default executor wait for an in-flight driver token broadcast. */
  public static final long DEFAULT_EXECUTOR_TOKEN_WAIT_MS = 20_000L;

  /** Maximum attempts for transient control-plane failures. */
  public static final String CONF_HTTP_MAX_ATTEMPTS = "fs.mlflow.cab.http.max.attempts";

  /** Default maximum attempts for transient control-plane failures. */
  public static final int DEFAULT_HTTP_MAX_ATTEMPTS = 6;

  /** Initial backoff for transient control-plane failures. */
  public static final String CONF_HTTP_INITIAL_BACKOFF_MS = "fs.mlflow.cab.http.initial.backoff.ms";

  /** Default initial backoff for transient control-plane failures. */
  public static final long DEFAULT_HTTP_INITIAL_BACKOFF_MS = 250L;

  /** Fully qualified name of the access token provider handed to the connector. */
  public static final String PROVIDER_CLASS =
      "com.google.cloud.mlflow.gcs.MlflowCabAccessTokenProvider";

  /** Fully qualified name of the delegation token binding handed to the connector. */
  public static final String BINDING_CLASS =
      "com.google.cloud.mlflow.gcs.MlflowCabDelegationTokenBinding";

  /** Fully qualified name of the Cloud Storage connector filesystem. */
  public static final String GHFS_CLASS = "com.google.cloud.hadoop.fs.gcs.GoogleHadoopFileSystem";

  /** Prefix of the UGI secret-key alias that carries a scope's CAB token. */
  public static final String UGI_ALIAS_PREFIX = "mlflow.cab.token.";

  private MlflowCabConstants() {}
}
