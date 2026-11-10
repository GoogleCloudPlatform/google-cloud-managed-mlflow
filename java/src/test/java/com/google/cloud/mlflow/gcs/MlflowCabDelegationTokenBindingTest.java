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
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertInstanceOf;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.google.cloud.hadoop.fs.gcs.GoogleHadoopFileSystem;
import com.google.cloud.hadoop.fs.gcs.auth.GcsDelegationTokens;
import com.google.cloud.hadoop.util.AccessTokenProvider;
import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.File;
import java.lang.reflect.Method;
import java.net.URL;
import java.net.URLClassLoader;
import java.nio.charset.StandardCharsets;
import java.security.PrivilegedExceptionAction;
import java.time.Instant;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.security.UserGroupInformation;
import org.apache.hadoop.security.token.Token;
import org.apache.hadoop.security.token.delegation.web.DelegationTokenIdentifier;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Test;

/**
 * Exercises the Cloud Storage connector's delegation-token runtime path with the provider on the
 * same classloader as the connector, as when the provider JAR is copied into the Spark image's JAR
 * directory next to {@code gcs-connector} (static install). The build runs this suite once against
 * connector 3.1.x and once against 4.0.x.
 */
class MlflowCabDelegationTokenBindingTest {

  private static final String TOKEN = "ya29." + "binding-cab-token";
  private static final Text SERVICE = new Text("gs://managed-bucket");

  @AfterEach
  void tearDown() {
    MlflowCabTokenService.setInstanceForTesting(null);
  }

  private static MlflowCabScope registerClientToken() throws Exception {
    MlflowCabTokenService service =
        new MlflowCabTokenService(
            MlflowCabTestSupport.env(Map.of()),
            System::currentTimeMillis,
            null,
            () -> 0,
            () -> true,
            MlflowCabTestSupport.staticCaller("ya29." + "unused"));
    MlflowCabTokenService.setInstanceForTesting(service);
    MlflowCabScope scope =
        MlflowCabScope.of(
            "https://mlflow.googleapis.com/v1alpha/projects/p/locations/l/namespaces/ns"
                + ":issueReadArtifactToken",
            "{\"runId\":\"r1\"}",
            "READ_ONLY",
            "gs://managed-bucket/runs/r1/artifacts");
    service.ensure(
        scope,
        MlflowCabTokenService.Supplier.CLIENT,
        TOKEN,
        Instant.now().plusSeconds(3600).toEpochMilli(),
        new Configuration(false),
        null);
    return scope;
  }

  @Test
  void connectorDeploysBindingAndServesBroadcastTokens() throws Exception {
    MlflowCabScope scope = registerClientToken();
    assertTrue(
        MlflowCabInnerConfiguration.bindingVisibleToConnector(),
        "provider and connector share a classloader in a static install");
    Configuration inner = MlflowCabInnerConfiguration.build(new Configuration(false), scope.scopeId());
    assertEquals(
        MlflowCabConstants.BINDING_CLASS, inner.get(MlflowCabInnerConfiguration.KEY_DT_BINDING));

    // The connector's own runtime: binding class resolved with Class.forName from its loader, and
    // start() deploys it (no delegation token of this kind in the owner's credentials: unbonded).
    GcsDelegationTokens tokens = new GcsDelegationTokens();
    tokens.bindToFileSystem(new GoogleHadoopFileSystem(), SERVICE);
    tokens.init(inner);
    tokens.start();
    try {
      assertFalse(tokens.isBoundToDT());
      AccessTokenProvider provider = tokens.getAccessTokenProvider();
      assertInstanceOf(MlflowCabAccessTokenProvider.class, provider);
      assertEquals(scope.scopeId(), ((MlflowCabAccessTokenProvider) provider).scopeId());
      assertEquals(TOKEN, provider.getAccessToken().getToken());
      assertFalse(provider.toString().contains("ya29."));
    } finally {
      tokens.stop();
    }

    MlflowCabDelegationTokenBinding binding =
        (MlflowCabDelegationTokenBinding)
            Class.forName(
                    inner.get(MlflowCabInnerConfiguration.KEY_DT_BINDING),
                    true,
                    GoogleHadoopFileSystem.class.getClassLoader())
                .getDeclaredConstructor()
                .newInstance();
    binding.init(inner);
    binding.start();
    try {
      assertEquals(MlflowCabDelegationTokenBinding.KIND, binding.getKind());
      AccessTokenProvider unbonded = binding.deployUnbonded();
      assertEquals(TOKEN, unbonded.getAccessToken().getToken());

      DelegationTokenIdentifier empty = binding.createEmptyIdentifier();
      assertEquals(MlflowCabDelegationTokenBinding.KIND, empty.getKind());
      DelegationTokenIdentifier issued = binding.createTokenIdentifier(new Text("yarn"));
      ByteArrayOutputStream bytes = new ByteArrayOutputStream();
      try (DataOutputStream out = new DataOutputStream(bytes)) {
        issued.write(out);
      }
      assertFalse(
          new String(bytes.toByteArray(), StandardCharsets.ISO_8859_1).contains("ya29."),
          "CAB secrets never travel inside delegation token identifiers");
      try (DataInputStream in =
          new DataInputStream(new ByteArrayInputStream(bytes.toByteArray()))) {
        empty.readFields(in);
      }
      assertEquals(new Text("yarn"), empty.getRenewer());
      assertEquals(TOKEN, binding.bindToTokenIdentifier(empty).getAccessToken().getToken());

      // A delegation token of this kind in the filesystem owner's credentials is decoded and the
      // connector binds to it through the same provider.
      Token<DelegationTokenIdentifier> token =
          new Token<>(issued.getBytes(), new byte[0], issued.getKind(), SERVICE);
      assertNotNull(token.decodeIdentifier(), "identifier kind is registered for decoding");
      UserGroupInformation owner = UserGroupInformation.createRemoteUser("dt-owner");
      owner.addToken(SERVICE, token);
      String viaToken =
          owner.doAs(
              (PrivilegedExceptionAction<String>)
                  () -> {
                    GcsDelegationTokens bound = new GcsDelegationTokens();
                    bound.bindToFileSystem(new GoogleHadoopFileSystem(), SERVICE);
                    bound.init(inner);
                    bound.start();
                    try {
                      assertTrue(bound.isBoundToDT());
                      assertEquals(token, bound.getBoundDT());
                      return bound.getAccessTokenProvider().getAccessToken().getToken();
                    } finally {
                      bound.stop();
                    }
                  });
      assertEquals(TOKEN, viaToken);
    } finally {
      binding.stop();
    }
  }

  /**
   * Rebuilds the static-install topology explicitly: one URLClassLoader holding Hadoop, the
   * connector, and the provider classes (no parent can see any of them).
   */
  @Test
  void singleUrlClassLoaderTopologyConfiguresAndDeploysTheBinding() throws Exception {
    String dependencies = System.getProperty("mlflow.cab.test.dependency.classpath");
    String providerDir = System.getProperty("mlflow.cab.test.provider.dir");
    assertNotNull(dependencies, "build_jar.py must pass -Dmlflow.cab.test.dependency.classpath");
    assertNotNull(providerDir, "build_jar.py must pass -Dmlflow.cab.test.provider.dir");
    List<URL> urls = new ArrayList<>();
    urls.add(new File(providerDir).toURI().toURL());
    for (String entry : dependencies.split(File.pathSeparator)) {
      if (!entry.isEmpty()) {
        urls.add(new File(entry).toURI().toURL());
      }
    }
    // Left open on purpose: Hadoop registers shutdown hooks that must still resolve classes.
    URLClassLoader shared =
        new URLClassLoader(urls.toArray(new URL[0]), ClassLoader.getPlatformClassLoader());
    Thread thread = Thread.currentThread();
    ClassLoader previous = thread.getContextClassLoader();
    thread.setContextClassLoader(shared);
    try {
      Class<?> innerConfiguration =
          Class.forName(MlflowCabInnerConfiguration.class.getName(), true, shared);
      assertSame(shared, innerConfiguration.getClassLoader());
      Method visible = innerConfiguration.getDeclaredMethod("bindingVisibleToConnector");
      visible.setAccessible(true);
      assertEquals(Boolean.TRUE, visible.invoke(null));

      Class<?> confClass = Class.forName("org.apache.hadoop.conf.Configuration", true, shared);
      Object callerConf = confClass.getConstructor(boolean.class).newInstance(false);
      Method build = innerConfiguration.getDeclaredMethod("build", confClass, String.class);
      build.setAccessible(true);
      Object inner = build.invoke(null, callerConf, "0123456789abcdef01234567");
      Method get = confClass.getMethod("get", String.class);
      assertEquals(
          MlflowCabConstants.BINDING_CLASS, get.invoke(inner, "fs.gs.delegation.token.binding"));

      Class<?> ghfs = Class.forName(MlflowCabConstants.GHFS_CLASS, true, shared);
      Class<?> textClass = Class.forName("org.apache.hadoop.io.Text", true, shared);
      Class<?> tokensClass =
          Class.forName("com.google.cloud.hadoop.fs.gcs.auth.GcsDelegationTokens", true, shared);
      Object tokens = tokensClass.getConstructor().newInstance();
      tokensClass
          .getMethod("bindToFileSystem", ghfs, textClass)
          .invoke(
              tokens,
              ghfs.getConstructor().newInstance(),
              textClass.getConstructor(String.class).newInstance("gs://managed-bucket"));
      tokensClass.getMethod("init", confClass).invoke(tokens, inner);
      tokensClass.getMethod("start").invoke(tokens);
      try {
        Object provider = tokensClass.getMethod("getAccessTokenProvider").invoke(tokens);
        assertEquals(MlflowCabConstants.PROVIDER_CLASS, provider.getClass().getName());
        assertSame(shared, provider.getClass().getClassLoader());
      } finally {
        tokensClass.getMethod("stop").invoke(tokens);
      }
    } finally {
      thread.setContextClassLoader(previous);
    }
  }
}
