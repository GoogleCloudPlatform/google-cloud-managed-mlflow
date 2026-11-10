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

import com.google.cloud.hadoop.fs.gcs.auth.AbstractDelegationTokenBinding;
import com.google.cloud.hadoop.util.AccessTokenProvider;
import java.io.IOException;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.security.UserGroupInformation;
import org.apache.hadoop.security.token.delegation.web.DelegationTokenIdentifier;

/**
 * Cloud Storage connector delegation token binding for MLflow CAB scopes.
 *
 * <p>CAB secrets never travel inside Hadoop delegation token identifiers: the binding hands the
 * connector an {@link MlflowCabAccessTokenProvider} whose tokens come from driver-broadcast UGI
 * secret keys, while identifiers only carry owner/renewer metadata.
 */
public final class MlflowCabDelegationTokenBinding extends AbstractDelegationTokenBinding {

  /** Delegation token kind for MLflow CAB bindings. */
  public static final Text KIND = new Text("MLFLOW_CAB_TOKEN");

  /** Public no-arg constructor required by the connector's reflective instantiation. */
  public MlflowCabDelegationTokenBinding() {
    super(KIND);
  }

  @Override
  public AccessTokenProvider deployUnbonded() throws IOException {
    return new MlflowCabAccessTokenProvider(getConfig());
  }

  @Override
  public AccessTokenProvider bindToTokenIdentifier(DelegationTokenIdentifier retrievedIdentifier)
      throws IOException {
    return new MlflowCabAccessTokenProvider(getConfig());
  }

  @Override
  public DelegationTokenIdentifier createTokenIdentifier(Text renewer) throws IOException {
    UserGroupInformation user = UserGroupInformation.getCurrentUser();
    Text owner = new Text(user.getUserName());
    return new MlflowCabTokenIdentifier(owner, renewer == null ? owner : renewer, owner);
  }

  @Override
  public DelegationTokenIdentifier createTokenIdentifier() throws IOException {
    return createTokenIdentifier(null);
  }

  @Override
  public DelegationTokenIdentifier createEmptyIdentifier() {
    return new MlflowCabTokenIdentifier();
  }
}
