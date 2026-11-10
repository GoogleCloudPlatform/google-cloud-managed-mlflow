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

import com.google.cloud.hadoop.util.AccessBoundary;
import com.google.cloud.hadoop.util.AccessTokenProvider;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.time.Instant;
import java.util.List;
import org.apache.hadoop.conf.Configuration;

/**
 * Cloud Storage connector {@link AccessTokenProvider} that serves already-downscoped CAB tokens.
 *
 * <p>The provider is only ever configured on the private configuration cloned by {@link
 * MlflowCabFileSystem}; it reads the scope id from {@code fs.mlflow.cab.active.scope} and obtains
 * tokens from {@link MlflowCabTokenService}: minted/renewed on the driver, read from broadcast UGI
 * credentials on executors (which never call the MLflow control plane).
 */
public final class MlflowCabAccessTokenProvider implements AccessTokenProvider {

  private Configuration conf;
  private String scopeId;

  /** Public no-arg constructor required by the connector's reflective instantiation. */
  public MlflowCabAccessTokenProvider() {}

  MlflowCabAccessTokenProvider(Configuration conf) {
    setConf(conf);
  }

  @Override
  public void setConf(Configuration configuration) {
    this.conf = configuration;
    this.scopeId =
        configuration == null ? null : configuration.getTrimmed(MlflowCabConstants.CONF_ACTIVE_SCOPE);
  }

  @Override
  public Configuration getConf() {
    return conf;
  }

  String scopeId() {
    return scopeId;
  }

  @Override
  public AccessToken getAccessToken() {
    try {
      MlflowCabToken token = MlflowCabTokenService.get().tokenFor(scopeId, conf);
      return new AccessToken(token.secret(), Instant.ofEpochMilli(token.expiresAtMs()));
    } catch (IOException e) {
      throw new UncheckedIOException(
          new IOException("Unable to obtain MLflow CAB token: " + MlflowCabRedaction.describe(e)));
    }
  }

  @Override
  public AccessToken getAccessToken(List<AccessBoundary> accessBoundaries) {
    // Tokens are already downscoped server-side to the MLflow artifact prefix.
    return getAccessToken();
  }

  @Override
  public void refresh() throws IOException {
    MlflowCabTokenService.get().forceRefresh(scopeId);
  }

  @Override
  public String toString() {
    return "MlflowCabAccessTokenProvider{scope=" + scopeId + "}";
  }
}
