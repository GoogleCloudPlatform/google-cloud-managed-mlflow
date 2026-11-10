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

import com.google.cloud.hadoop.fs.gcs.GoogleHadoopFileSystem;
import java.io.IOException;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.util.ReflectionUtils;

/**
 * Creates the wrapped {@code gs://} filesystem. Kept separate so that loading {@link
 * MlflowCabFileSystem} (for example during Hadoop's ServiceLoader scan) never requires the Cloud
 * Storage connector to be present.
 */
final class MlflowCabGcsDelegate {

  private MlflowCabGcsDelegate() {}

  static FileSystem create(Configuration inner) throws IOException {
    Class<?> custom = inner.getClass(MlflowCabConstants.CONF_DELEGATE_IMPL, null);
    if (custom == null) {
      return new GoogleHadoopFileSystem();
    }
    if (!FileSystem.class.isAssignableFrom(custom)) {
      throw new IOException(
          MlflowCabConstants.CONF_DELEGATE_IMPL + " must name a Hadoop FileSystem implementation");
    }
    return (FileSystem) ReflectionUtils.newInstance(custom, inner);
  }
}
