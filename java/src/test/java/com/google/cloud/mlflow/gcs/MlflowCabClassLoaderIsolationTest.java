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

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.io.File;
import java.lang.reflect.Method;
import java.net.URI;
import java.net.URL;
import java.net.URLClassLoader;
import java.util.ArrayList;
import java.util.List;
import org.junit.jupiter.api.Test;

/**
 * Simulates an executor (or Spark Connect session) where Hadoop and the Cloud Storage connector
 * live in the parent ("system") classloader and the provider JAR lives only in a child loader, as
 * with {@code --jars} or {@code spark.addArtifact}.
 */
class MlflowCabClassLoaderIsolationTest {

  private static final String SCOPE = "fedcba9876543210fedcba98";

  private static URL[] split(String classpath, boolean providerOnly, String providerDir) throws Exception {
    List<URL> urls = new ArrayList<>();
    for (String entry : classpath.split(File.pathSeparator)) {
      if (entry.isEmpty()) {
        continue;
      }
      boolean isProvider = new File(entry).getCanonicalPath().equals(new File(providerDir).getCanonicalPath());
      if (isProvider == providerOnly) {
        urls.add(new File(entry).toURI().toURL());
      }
    }
    return urls.toArray(new URL[0]);
  }

  @Test
  void childLoaderProviderWorksWhenParentLacksIt() throws Exception {
    String runtimeClasspath = System.getProperty("mlflow.cab.test.dependency.classpath");
    String providerDir = System.getProperty("mlflow.cab.test.provider.dir");
    assertNotNull(runtimeClasspath, "build_jar.py must pass -Dmlflow.cab.test.dependency.classpath");
    assertNotNull(providerDir, "build_jar.py must pass -Dmlflow.cab.test.provider.dir");

    // Loaders are intentionally left open: Hadoop registers JVM shutdown hooks from the parent
    // loader that must still resolve their classes at exit.
    URLClassLoader parent =
        new URLClassLoader(split(runtimeClasspath, false, providerDir), ClassLoader.getPlatformClassLoader());
    URLClassLoader child =
        new URLClassLoader(new URL[] {new File(providerDir).toURI().toURL()}, parent);
    {
      // The "system" side cannot see the provider (the original executor ClassNotFound failure).
      assertThrows(
          ClassNotFoundException.class,
          () -> Class.forName(MlflowCabConstants.PROVIDER_CLASS, false, parent));
      Class<?> confClass = Class.forName("org.apache.hadoop.conf.Configuration", true, parent);
      Object executorConf = confClass.getConstructor().newInstance();
      confClass.getMethod("setClassLoader", ClassLoader.class).invoke(executorConf, parent);
      Method getClassByNameOrNull = confClass.getMethod("getClassByNameOrNull", String.class);
      assertNull(getClassByNameOrNull.invoke(executorConf, MlflowCabConstants.PROVIDER_CLASS));

      // Loading the filesystem from the child loader wires its private configuration so the
      // connector resolves the provider through the configuration's classloader.
      Class<?> fsClass = Class.forName(MlflowCabFileSystem.class.getName(), true, child);
      assertSame(child, fsClass.getClassLoader());
      Object fs = fsClass.getConstructor().newInstance();
      fsClass
          .getMethod("initialize", URI.class, confClass)
          .invoke(fs, URI.create("mlflow-gs://" + SCOPE + "@bucket/"), executorConf);
      Method innerConfiguration = fsClass.getDeclaredMethod("innerConfiguration");
      innerConfiguration.setAccessible(true);
      Object inner = innerConfiguration.invoke(fs);
      Class<?> provider =
          (Class<?>) confClass.getMethod("getClassByName", String.class).invoke(inner, MlflowCabConstants.PROVIDER_CLASS);
      assertSame(child, provider.getClassLoader());
      Object instance = provider.getConstructor().newInstance();
      assertNotNull(instance);
      // The connector instantiates bindings via Class.forName from its own (parent) loader, so the
      // binding key must not be configured in this layout.
      assertNull(confClass.getMethod("get", String.class).invoke(inner, "fs.gs.delegation.token.binding"));
      Class<?> binding = Class.forName(MlflowCabConstants.BINDING_CLASS, true, child);
      assertNotNull(binding.getConstructor().newInstance());

      // Executor/driver base configuration installation keeps parent-first resolution.
      Class<?> loaders = Class.forName(MlflowCabClassLoaders.class.getName(), true, child);
      Method install = loaders.getDeclaredMethod("install", confClass);
      install.setAccessible(true);
      assertEquals(Boolean.TRUE, install.invoke(null, executorConf));
      ClassLoader installed = (ClassLoader) confClass.getMethod("getClassLoader").invoke(executorConf);
      assertSame(
          child,
          ((Class<?>) getClassByNameOrNull.invoke(executorConf, MlflowCabFileSystem.class.getName()))
              .getClassLoader());
      assertSame(
          parent,
          ((Class<?>) getClassByNameOrNull.invoke(executorConf, "org.apache.hadoop.fs.Path"))
              .getClassLoader());
      assertTrue(installed.getClass().getName().endsWith("MlflowCabDelegatingClassLoader"));
      assertEquals(
          MlflowCabFileSystem.class.getName(),
          confClass.getMethod("get", String.class).invoke(executorConf, "fs.mlflow-gs.impl"));
      // Idempotent: a second install does not wrap again.
      assertEquals(Boolean.FALSE, install.invoke(null, executorConf));
    }
  }
}
