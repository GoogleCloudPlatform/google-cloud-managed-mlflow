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

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import org.apache.hadoop.conf.Configuration;

/**
 * Builds the private configuration used by the wrapped Cloud Storage connector.
 *
 * <p>The caller's configuration is cloned and never modified, so customer {@code fs.gs.*}
 * settings and cached {@code gs://} filesystem instances are left untouched. Only the clone
 * receives the CAB-specific connector keys and the provider's classloader.
 */
final class MlflowCabInnerConfiguration {

  static final String KEY_DISABLE_CACHE = "fs.gs.impl.disable.cache";
  static final String KEY_AUTH_TYPE = "fs.gs.auth.type";
  static final String KEY_TOKEN_PROVIDER = "fs.gs.auth.access.token.provider";
  static final String KEY_DT_BINDING = "fs.gs.delegation.token.binding";
  static final String KEY_CONFLICT_CHECK = "fs.gs.create.items.conflict.check.enable";
  static final String KEY_DIR_REPAIR = "fs.gs.implicit.dir.repair.enable";
  static final String KEY_HNS_FOLDERS = "fs.gs.hierarchical.namespace.folders.enable";
  static final String IMPERSONATION_PREFIX = "fs.gs.auth.impersonation.";

  private MlflowCabInnerConfiguration() {}

  static Configuration build(Configuration caller, String scopeId) {
    Configuration inner = caller == null ? new Configuration() : new Configuration(caller);
    ClassLoader own = MlflowCabInnerConfiguration.class.getClassLoader();
    if (own != null) {
      inner.setClassLoader(own);
    }
    inner.set(KEY_DISABLE_CACHE, "true");
    inner.set(KEY_AUTH_TYPE, "ACCESS_TOKEN_PROVIDER");
    inner.set(KEY_TOKEN_PROVIDER, MlflowCabConstants.PROVIDER_CLASS);
    if (bindingVisibleToConnector()) {
      inner.set(KEY_DT_BINDING, MlflowCabConstants.BINDING_CLASS);
    } else {
      inner.unset(KEY_DT_BINDING);
    }
    inner.set(KEY_CONFLICT_CHECK, "false");
    inner.set(KEY_DIR_REPAIR, "false");
    inner.set(KEY_HNS_FOLDERS, "false");
    List<String> impersonation = new ArrayList<>();
    for (Map.Entry<String, String> entry : inner) {
      if (entry.getKey().startsWith(IMPERSONATION_PREFIX)) {
        impersonation.add(entry.getKey());
      }
    }
    for (String key : impersonation) {
      inner.unset(key);
    }
    if (scopeId != null) {
      inner.set(MlflowCabConstants.CONF_ACTIVE_SCOPE, scopeId);
    } else {
      inner.unset(MlflowCabConstants.CONF_ACTIVE_SCOPE);
    }
    return inner;
  }

  /**
   * The connector instantiates delegation token bindings with {@code Class.forName} from its own
   * classloader, so the binding is only configured when that loader resolves this exact class
   * (static image installs). Otherwise the provider path, which the connector loads through the
   * configuration's classloader, carries the identical driver-broadcast tokens.
   */
  static boolean bindingVisibleToConnector() {
    try {
      ClassLoader own = MlflowCabInnerConfiguration.class.getClassLoader();
      Class<?> connector = Class.forName(MlflowCabConstants.GHFS_CLASS, false, own);
      Class<?> binding =
          Class.forName(MlflowCabConstants.BINDING_CLASS, false, connector.getClassLoader());
      return binding == MlflowCabDelegationTokenBinding.class;
    } catch (ClassNotFoundException | LinkageError e) {
      return false;
    }
  }
}
