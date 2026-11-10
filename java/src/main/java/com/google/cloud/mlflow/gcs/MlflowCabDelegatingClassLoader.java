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
import java.net.URL;
import java.util.Enumeration;

/**
 * Parent-first classloader that falls back to the provider's own classloader.
 *
 * <p>Installed on Hadoop {@code Configuration}s whose classloader cannot see the provider (for
 * example when the JAR arrives through {@code spark.addArtifact} after the session's base
 * configuration was created). Every class the original loader could resolve keeps resolving to
 * the same definition; only classes it cannot find are served by the provider's loader.
 */
final class MlflowCabDelegatingClassLoader extends ClassLoader {

  static {
    registerAsParallelCapable();
  }

  private final ClassLoader fallback;

  MlflowCabDelegatingClassLoader(ClassLoader primary, ClassLoader fallback) {
    super(primary);
    this.fallback = fallback;
  }

  ClassLoader fallback() {
    return fallback;
  }

  @Override
  protected Class<?> findClass(String name) throws ClassNotFoundException {
    return fallback.loadClass(name);
  }

  @Override
  protected URL findResource(String name) {
    return fallback.getResource(name);
  }

  @Override
  protected Enumeration<URL> findResources(String name) throws IOException {
    return fallback.getResources(name);
  }

  @Override
  public String toString() {
    return "MlflowCabDelegatingClassLoader{parent=" + getParent() + "}";
  }
}
