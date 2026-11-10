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

import org.apache.hadoop.io.Text;
import org.apache.hadoop.security.token.delegation.web.DelegationTokenIdentifier;

/** Secret-free Hadoop delegation token identifier for MLflow CAB bindings. */
public final class MlflowCabTokenIdentifier extends DelegationTokenIdentifier {

  /** Public no-arg constructor required for Writable deserialization. */
  public MlflowCabTokenIdentifier() {
    super(MlflowCabDelegationTokenBinding.KIND);
  }

  MlflowCabTokenIdentifier(Text owner, Text renewer, Text realUser) {
    super(MlflowCabDelegationTokenBinding.KIND, owner, renewer, realUser);
  }

  @Override
  public Text getKind() {
    return MlflowCabDelegationTokenBinding.KIND;
  }
}
