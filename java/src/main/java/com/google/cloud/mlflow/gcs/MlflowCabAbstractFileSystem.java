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
import java.net.URI;
import java.net.URISyntaxException;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.DelegateToFileSystem;

/**
 * {@link org.apache.hadoop.fs.AbstractFileSystem} for the {@code mlflow-gs} scheme, delegating to
 * {@link MlflowCabFileSystem} so {@code FileContext} users (for example Spark structured streaming
 * checkpoint managers and some committers) resolve {@code mlflow-gs://} URIs.
 *
 * <p>Hadoop resolves AbstractFileSystems through {@code fs.AbstractFileSystem.mlflow-gs.impl};
 * the bundled Python plugin sets that key automatically, and the class is also listed in {@code
 * META-INF/services/org.apache.hadoop.fs.AbstractFileSystem}.
 */
public class MlflowCabAbstractFileSystem extends DelegateToFileSystem {

  /** Constructor invoked reflectively by {@code AbstractFileSystem.createFileSystem}. */
  public MlflowCabAbstractFileSystem(URI theUri, Configuration conf)
      throws IOException, URISyntaxException {
    super(theUri, new MlflowCabFileSystem(), conf, MlflowCabConstants.SCHEME, false);
  }

  /**
   * No-arg constructor so generic {@code ServiceLoader} iteration can instantiate the provider; it
   * binds to a bucket-less placeholder URI and performs no I/O.
   */
  public MlflowCabAbstractFileSystem() throws IOException, URISyntaxException {
    this(URI.create(MlflowCabConstants.SCHEME + ":///"), new Configuration(false));
  }

  @Override
  public int getUriDefaultPort() {
    return -1;
  }
}
