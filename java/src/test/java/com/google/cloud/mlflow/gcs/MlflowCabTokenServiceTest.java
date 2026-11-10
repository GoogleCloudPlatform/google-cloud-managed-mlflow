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
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.security.PrivilegedExceptionAction;
import java.time.Instant;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.security.Credentials;
import org.apache.hadoop.security.UserGroupInformation;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Test;

class MlflowCabTokenServiceTest {

  private final List<MlflowCabTokenService> created = new ArrayList<>();

  @AfterEach
  void tearDown() {
    for (MlflowCabTokenService service : created) {
      service.shutdown();
    }
  }

  /** Mock MLflow CAB endpoint minting unique tokens with a configurable TTL. */
  private static MlflowCabTestSupport.FakeServer mlflow(AtomicInteger counter, long ttlMs)
      throws IOException {
    return new MlflowCabTestSupport.FakeServer(
        r -> {
          int n = counter.incrementAndGet();
          String token = "ya29." + "cab-" + r.path.replaceAll("[^a-zA-Z]", "") + "-" + n;
          String expire = Instant.ofEpochMilli(System.currentTimeMillis() + ttlMs).toString();
          return new MlflowCabTestSupport.Reply(
              200, MlflowCabTestSupport.cabResponse(token, expire, "gs://b/p/", n % 2 == 0));
        });
  }

  private MlflowCabTokenService service(
      String baseUrl, Map<String, String> extraEnv, AtomicBoolean driver, AtomicInteger broadcasts) {
    Map<String, String> envValues = new HashMap<>(extraEnv);
    envValues.put(MlflowCabConstants.ENV_ENDPOINT_OVERRIDE, baseUrl);
    MlflowCabTokenService service =
        new MlflowCabTokenService(
            MlflowCabTestSupport.env(envValues),
            System::currentTimeMillis,
            new MlflowCabIssuer(
                MlflowCabIssuer.newHttpClient(),
                System::currentTimeMillis,
                MlflowCabTestSupport.env(envValues)),
            () -> {
              broadcasts.incrementAndGet();
              return 2;
            },
            driver::get,
            MlflowCabTestSupport.staticCaller("ya29." + "driver-caller"));
    created.add(service);
    return service;
  }

  private static MlflowCabScope scope(String base, String resource, String rpc, String body, String mode) {
    return MlflowCabScope.of(base + "/v1alpha/" + resource + rpc, body, mode, "");
  }

  @Test
  void mintsOncePerScopeAndIsolatesMultiScopeTransitions() throws Exception {
    AtomicInteger rpcs = new AtomicInteger();
    AtomicInteger broadcasts = new AtomicInteger();
    try (MlflowCabTestSupport.FakeServer server = mlflow(rpcs, 3_600_000L)) {
      MlflowCabTokenService service =
          service(server.baseUrl(), Map.of(), new AtomicBoolean(true), broadcasts);
      String b = server.baseUrl();
      MlflowCabScope nsRw =
          scope(b, "projects/p/locations/l/namespaces/ns", ":issueReadWriteArtifactToken",
              "{\"runId\":\"r1\",\"artifactPath\":\"model\"}", "READ_WRITE");
      MlflowCabScope vaultRc =
          scope(b, "projects/p/locations/l/registries/r/registeredModels/m", ":issueReadWriteArtifactToken",
              "{\"artifactUri\":\"gs://vault/registeredModels/u/versions/1\"}", "READ_CREATE");
      MlflowCabScope vaultRo =
          scope(b, "projects/p/locations/l/registries/r/registeredModels/m", ":issueReadArtifactToken",
              "{\"artifactUri\":\"gs://vault/registeredModels/u/versions/1\"}", "READ_ONLY");
      Configuration conf = new Configuration(false);

      service.ensure(nsRw, MlflowCabTokenService.Supplier.CLUSTER, null, 0L, conf, null);
      service.ensure(nsRw, MlflowCabTokenService.Supplier.CLUSTER, null, 0L, conf, null);
      for (int i = 0; i < 5; i++) {
        service.tokenFor(nsRw.scopeId(), conf);
      }
      assertEquals(1, rpcs.get(), "one driver RPC per scope regardless of reads");
      String nsToken = service.tokenFor(nsRw.scopeId(), conf).secret();

      service.ensure(vaultRc, MlflowCabTokenService.Supplier.CLUSTER, null, 0L, conf, null);
      service.ensure(vaultRo, MlflowCabTokenService.Supplier.CLUSTER, null, 0L, conf, null);
      assertEquals(3, rpcs.get());
      String rcToken = service.tokenFor(vaultRc.scopeId(), conf).secret();
      String roToken = service.tokenFor(vaultRo.scopeId(), conf).secret();
      assertNotEquals(nsToken, rcToken);
      assertNotEquals(rcToken, roToken);
      assertNotEquals(nsRw.scopeId(), vaultRc.scopeId());
      assertNotEquals(vaultRc.scopeId(), vaultRo.scopeId());

      // Back to the namespace scope: still fresh, so no stale cache hit and no extra RPC.
      service.ensure(nsRw, MlflowCabTokenService.Supplier.CLUSTER, null, 0L, conf, null);
      assertEquals(nsToken, service.tokenFor(nsRw.scopeId(), conf).secret());
      assertEquals(3, rpcs.get());
      assertEquals(3, broadcasts.get(), "one broadcast per newly minted token");

      Map<String, String> stats = service.stats(nsRw.scopeId());
      assertEquals("3", stats.get("driverCabRpcs"));
      assertEquals("0", stats.get("executorCabRpcs"));
      for (String value : stats.values()) {
        assertFalse(value.contains("ya29."), value);
      }
      assertFalse(service.tokenFor(nsRw.scopeId(), conf).toString().contains("ya29."));
    }
  }

  @Test
  void shortTtlRotatesAtLeastThreeTimesUsingPublicMarginEnvVar() throws Exception {
    AtomicInteger rpcs = new AtomicInteger();
    AtomicInteger broadcasts = new AtomicInteger();
    try (MlflowCabTestSupport.FakeServer server = mlflow(rpcs, 3_000L)) {
      MlflowCabTokenService service =
          service(
              server.baseUrl(),
              Map.of(MlflowCabConstants.ENV_REFRESH_MARGIN_SECONDS, "2"),
              new AtomicBoolean(true),
              broadcasts);
      MlflowCabScope scope =
          scope(server.baseUrl(), "projects/p/locations/l/namespaces/ns", ":issueReadArtifactToken",
              "{\"artifactUri\":\"gs://b/p\"}", "READ_ONLY");
      Configuration conf = new Configuration(false);
      Map<String, String> first =
          service.ensure(scope, MlflowCabTokenService.Supplier.CLUSTER, null, 0L, conf, null);
      assertEquals("2000", first.get("refreshMarginMs"));
      List<String> seen = new ArrayList<>();
      long deadline = System.currentTimeMillis() + 8_000L;
      while (System.currentTimeMillis() < deadline) {
        String token = service.tokenFor(scope.scopeId(), conf).secret();
        if (seen.isEmpty() || !seen.get(seen.size() - 1).equals(token)) {
          seen.add(token);
        }
        // Executor view: the broadcast UGI secret always carries the latest generation.
        MlflowCabToken ugi = MlflowCabTokenService.readFromUgi(scope.scopeId());
        assertNotNull(ugi);
        Thread.sleep(100L);
      }
      int renewals = Integer.parseInt(service.stats(scope.scopeId()).get("scopeRenewals"));
      assertTrue(renewals >= 3, "expected >=3 renewals, got " + renewals);
      assertTrue(seen.size() >= 4, "expected >=4 distinct tokens, got " + seen.size());
      assertEquals(rpcs.get(), broadcasts.get());
    }
  }

  @Test
  void executorsReadBroadcastCredentialsAndNeverCallMlflow() throws Exception {
    AtomicInteger rpcs = new AtomicInteger();
    AtomicInteger broadcasts = new AtomicInteger();
    try (MlflowCabTestSupport.FakeServer server = mlflow(rpcs, 3_600_000L)) {
      AtomicBoolean driverRole = new AtomicBoolean(true);
      MlflowCabTokenService driver = service(server.baseUrl(), Map.of(), driverRole, broadcasts);
      MlflowCabScope scope =
          scope(server.baseUrl(), "projects/p/locations/l/namespaces/ns", ":issueReadArtifactToken",
              "{\"runId\":\"r9\"}", "READ_ONLY");
      Configuration conf = new Configuration(false);
      driver.ensure(scope, MlflowCabTokenService.Supplier.CLUSTER, null, 0L, conf, null);
      assertEquals(1, rpcs.get());

      // Simulate Spark's UpdateDelegationTokens: serialize the driver's credentials and add them
      // to a distinct executor UGI, exactly as CoarseGrainedExecutorBackend does.
      ByteArrayOutputStream bytes = new ByteArrayOutputStream();
      try (DataOutputStream out = new DataOutputStream(bytes)) {
        UserGroupInformation.getCurrentUser().getCredentials().writeTokenStorageToStream(out);
      }
      Credentials received = new Credentials();
      received.readTokenStorageStream(
          new DataInputStream(new ByteArrayInputStream(bytes.toByteArray())));
      UserGroupInformation executorUser = UserGroupInformation.createRemoteUser("executor-1");
      executorUser.addCredentials(received);

      AtomicBoolean executorRole = new AtomicBoolean(false);
      MlflowCabTokenService executor =
          service(server.baseUrl(), Map.of(), executorRole, new AtomicInteger());
      Configuration executorConf = new Configuration(false);
      executorConf.setLong(MlflowCabConstants.CONF_EXECUTOR_TOKEN_WAIT_MS, 200L);
      String token =
          executorUser.doAs(
              (PrivilegedExceptionAction<String>)
                  () -> executor.tokenFor(scope.scopeId(), executorConf).secret());
      assertEquals(driver.tokenFor(scope.scopeId(), conf).secret(), token);
      assertEquals(1, rpcs.get(), "executors must not issue CAB RPCs");
      assertEquals("0", executor.stats(null).get("driverCabRpcs"));
      assertThrows(IOException.class, () -> executor.ensure(scope,
          MlflowCabTokenService.Supplier.CLUSTER, null, 0L, conf, null));

      // An executor without the broadcast fails fast instead of minting.
      UserGroupInformation lonely = UserGroupInformation.createRemoteUser("executor-2");
      IOException missing =
          assertThrows(
              IOException.class,
              () ->
                  lonely.doAs(
                      (PrivilegedExceptionAction<MlflowCabToken>)
                          () -> executor.tokenFor("abcdefabcdefabcdefabcdef", executorConf)));
      assertTrue(missing.getMessage().contains("never mint"));
      assertEquals(1, rpcs.get());
    }
  }

  @Test
  void clientSuppliedTokensArePublishedWithoutDriverRpc() throws Exception {
    AtomicInteger rpcs = new AtomicInteger();
    AtomicInteger broadcasts = new AtomicInteger();
    try (MlflowCabTestSupport.FakeServer server = mlflow(rpcs, 3_600_000L)) {
      MlflowCabTokenService service =
          service(server.baseUrl(), Map.of(), new AtomicBoolean(true), broadcasts);
      MlflowCabScope scope =
          scope(server.baseUrl(), "projects/p/locations/l/namespaces/ns", ":issueReadArtifactToken",
              "{\"modelId\":\"m-1\"}", "READ_ONLY");
      long expiry = System.currentTimeMillis() + 600_000L;
      Map<String, String> result =
          service.ensure(
              scope, MlflowCabTokenService.Supplier.CLIENT, "ya29." + "client-pushed", expiry,
              new Configuration(false), null);
      assertEquals("CLIENT", result.get("supplier"));
      assertEquals("ya29." + "client-pushed", service.tokenFor(scope.scopeId(), null).secret());
      assertEquals(0, rpcs.get());
      assertEquals(1, broadcasts.get());
      for (String value : result.values()) {
        assertFalse(value.contains("ya29."));
      }
      assertThrows(
          IOException.class,
          () ->
              service.ensure(
                  scope, MlflowCabTokenService.Supplier.CLIENT, "", expiry, new Configuration(false), null));
    }
  }

  @Test
  void refreshMarginPrecedence() {
    MlflowCabTokenService service =
        new MlflowCabTokenService(
            MlflowCabTestSupport.env(Map.of(MlflowCabConstants.ENV_REFRESH_MARGIN_SECONDS, "120")),
            System::currentTimeMillis,
            null,
            () -> -1,
            () -> true,
            null);
    created.add(service);
    Configuration conf = new Configuration(false);
    assertEquals(120_000L, service.refreshMarginMs(conf, null));
    conf.set(MlflowCabConstants.CONF_REFRESH_MARGIN_SECONDS, "45");
    assertEquals(45_000L, service.refreshMarginMs(conf, null));
    assertEquals(7_000L, service.refreshMarginMs(conf, 7L));
    conf.set(MlflowCabConstants.CONF_REFRESH_MARGIN_SECONDS, "not-a-number");
    assertEquals(120_000L, service.refreshMarginMs(conf, null));
  }

  @Test
  void scopeIdsAreStableAndValidated() {
    String id = MlflowCabScope.computeScopeId("https://mlflow.googleapis.com/v1alpha/x", "{}");
    assertEquals(24, id.length());
    assertEquals(id, MlflowCabScope.computeScopeId("https://mlflow.googleapis.com/v1alpha/x", "{}"));
    assertTrue(MlflowCabScope.isValidScopeId(id));
    assertFalse(MlflowCabScope.isValidScopeId("ABC"));
    assertFalse(MlflowCabScope.isValidScopeId(null));
    MlflowCabToken token = new MlflowCabToken(id, "ya29." + "secret", 42L, 3L);
    MlflowCabToken decoded = MlflowCabToken.decode(id, token.encode());
    assertNotNull(decoded);
    assertEquals("ya29." + "secret", decoded.secret());
    assertEquals(42L, decoded.expiresAtMs());
    assertEquals(3L, decoded.generation());
    assertFalse(token.toString().contains("ya29."));
  }

  @Test
  void clientTokensAreUnwrappedOnlyFromOaepSha256CiphertextForThisJvm() throws Exception {
    MlflowCabTokenService service =
        new MlflowCabTokenService(
            MlflowCabTestSupport.env(Map.of()), System::currentTimeMillis, null, () -> 0,
            () -> true, null);
    created.add(service);
    String publicKey = service.clientWrapKey();
    assertEquals(publicKey, service.clientWrapKey(), "one key pair per service instance");
    java.security.spec.X509EncodedKeySpec spec =
        new java.security.spec.X509EncodedKeySpec(java.util.Base64.getDecoder().decode(publicKey));
    java.security.interfaces.RSAPublicKey rsa =
        (java.security.interfaces.RSAPublicKey)
            java.security.KeyFactory.getInstance("RSA").generatePublic(spec);
    assertEquals(2048, rsa.getModulus().bitLength());

    String shortToken = "ya29." + "short-client-token";
    StringBuilder longToken = new StringBuilder("ya29." + "c.");
    while (longToken.length() < 1500) {
      longToken.append("AbCdEfGhIj0123456789-_");
    }
    for (String token : new String[] {shortToken, longToken.toString()}) {
      String wrapped = MlflowCabTestSupport.wrapForDriver(publicKey, token, true);
      assertFalse(wrapped.contains("ya29."));
      assertEquals(token, service.unwrapClientToken(wrapped));
    }

    byte[] blocks =
        java.util.Base64.getDecoder()
            .decode(MlflowCabTestSupport.wrapForDriver(publicKey, longToken.toString(), true));
    byte[] tampered = blocks.clone();
    tampered[17] ^= 0x01;
    byte[] reordered = blocks.clone();
    System.arraycopy(blocks, 0, reordered, 256, 256);
    System.arraycopy(blocks, 256, reordered, 0, 256);
    MlflowCabTokenService other =
        new MlflowCabTokenService(
            MlflowCabTestSupport.env(Map.of()), System::currentTimeMillis, null, () -> 0,
            () -> true, null);
    created.add(other);
    List<String> rejected = new ArrayList<>();
    rejected.add(java.util.Base64.getEncoder().encodeToString(tampered));
    rejected.add(java.util.Base64.getEncoder().encodeToString(reordered));
    rejected.add(java.util.Base64.getEncoder().encodeToString(java.util.Arrays.copyOf(blocks, 300)));
    rejected.add(MlflowCabTestSupport.wrapForDriver(other.clientWrapKey(), shortToken, true));
    rejected.add(MlflowCabTestSupport.wrapForDriver(publicKey, shortToken, false));
    rejected.add(shortToken);
    rejected.add("");
    for (String bad : rejected) {
      IOException error = assertThrows(IOException.class, () -> service.unwrapClientToken(bad));
      assertFalse(String.valueOf(error.getMessage()).contains("ya29."), error.getMessage());
    }
  }
}
