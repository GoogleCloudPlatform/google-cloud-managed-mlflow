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
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.io.IOException;
import java.time.Instant;
import java.util.Map;
import java.util.concurrent.atomic.AtomicInteger;
import org.apache.hadoop.conf.Configuration;
import org.junit.jupiter.api.Test;

class MlflowCabIssuerTest {

  private static final String CALLER = "ya29." + "caller-secret-token";

  private static Configuration fastRetries() {
    Configuration conf = new Configuration(false);
    conf.setInt(MlflowCabConstants.CONF_HTTP_MAX_ATTEMPTS, 5);
    conf.setLong(MlflowCabConstants.CONF_HTTP_INITIAL_BACKOFF_MS, 5L);
    return conf;
  }

  private static MlflowCabIssuer issuer(String baseUrl) {
    return new MlflowCabIssuer(
        MlflowCabIssuer.newHttpClient(),
        System::currentTimeMillis,
        MlflowCabTestSupport.env(Map.of(MlflowCabConstants.ENV_ENDPOINT_OVERRIDE, baseUrl)));
  }

  @Test
  void parsesCamelCaseAndSnakeCaseResponses() throws Exception {
    String expire = Instant.now().plusSeconds(600).toString();
    try (MlflowCabTestSupport.FakeServer server =
        new MlflowCabTestSupport.FakeServer(
            r ->
                new MlflowCabTestSupport.Reply(
                    200,
                    MlflowCabTestSupport.cabResponse(
                        "ya29." + "cab-camel", expire, "gs://b/p/", !r.path.contains("snake"))))) {
      MlflowCabIssuer issuer = issuer(server.baseUrl());
      MlflowCabScope camel =
          MlflowCabScope.of(
              server.baseUrl() + "/v1alpha/projects/p/locations/l/namespaces/n:issueReadArtifactToken",
              "{\"artifactUri\":\"gs://b/p\"}",
              "READ_ONLY",
              "gs://b/p/");
      MlflowCabIssuer.Issued issued =
          issuer.issue(camel, MlflowCabTestSupport.staticCaller(CALLER), fastRetries(), "driver");
      assertEquals("ya29." + "cab-camel", issued.token);
      assertEquals(Instant.parse(expire).toEpochMilli(), issued.expiresAtMs);
      assertEquals("gs://b/p/", issued.artifactUri);
      assertEquals("Bearer " + CALLER, server.requests.get(0).header("Authorization"));
      assertTrue(server.requests.get(0).header("User-Agent").contains("role=driver"));
      assertEquals("{\"artifactUri\":\"gs://b/p\"}", server.requests.get(0).body);
      assertFalse(issued.toString().contains("ya29"));

      MlflowCabScope snake =
          MlflowCabScope.of(server.baseUrl() + "/snake:issueReadArtifactToken", "{}", "READ_ONLY", "");
      assertEquals(
          "ya29." + "cab-camel",
          issuer.issue(snake, MlflowCabTestSupport.staticCaller(CALLER), fastRetries(), "driver")
              .token);
    }
  }

  @Test
  void retriesTransient5xxAnd429ThenSucceeds() throws Exception {
    AtomicInteger calls = new AtomicInteger();
    String expire = Instant.now().plusSeconds(600).toString();
    try (MlflowCabTestSupport.FakeServer server =
        new MlflowCabTestSupport.FakeServer(
            r -> {
              int n = calls.incrementAndGet();
              if (n == 1) {
                return new MlflowCabTestSupport.Reply(
                    500, "{\"error\":{\"status\":\"INTERNAL\",\"message\":\"Invalid IAM type\"}}");
              }
              if (n == 2) {
                return new MlflowCabTestSupport.Reply(503, "unavailable");
              }
              if (n == 3) {
                return new MlflowCabTestSupport.Reply(429, "slow down", Map.of("Retry-After", "0"));
              }
              return new MlflowCabTestSupport.Reply(
                  200, MlflowCabTestSupport.cabResponse("ya29." + "after-retry", expire, "gs://b/", true));
            })) {
      MlflowCabScope scope =
          MlflowCabScope.of(server.baseUrl() + "/v1alpha/x:issueReadWriteArtifactToken", "{}", "READ_WRITE", "");
      MlflowCabIssuer.Issued issued =
          issuer(server.baseUrl())
              .issue(scope, MlflowCabTestSupport.staticCaller(CALLER), fastRetries(), "driver");
      assertEquals("ya29." + "after-retry", issued.token);
      assertEquals(4, issued.attempts);
      assertEquals(4, server.count.get());
    }
  }

  @Test
  void permanentErrorsAreRedactedAndNotRetried() throws Exception {
    try (MlflowCabTestSupport.FakeServer server =
        new MlflowCabTestSupport.FakeServer(
            r ->
                new MlflowCabTestSupport.Reply(
                    403,
                    "{\"error\":{\"status\":\"PERMISSION_DENIED\",\"message\":\"denied for ya29." + "leaked-token-value\"}}"))) {
      MlflowCabScope scope =
          MlflowCabScope.of(server.baseUrl() + "/v1alpha/x:issueReadArtifactToken", "{}", "READ_ONLY", "");
      IOException error =
          assertThrows(
              IOException.class,
              () ->
                  issuer(server.baseUrl())
                      .issue(scope, MlflowCabTestSupport.staticCaller(CALLER), fastRetries(), "driver"));
      assertTrue(error.getMessage().contains("HTTP 403"));
      assertTrue(error.getMessage().contains("PERMISSION_DENIED"));
      assertFalse(error.getMessage().contains("ya29."), error.getMessage());
      assertEquals(1, server.count.get());
    }
  }

  @Test
  void refusesUntrustedOriginsBeforeSendingAnything() throws Exception {
    try (MlflowCabTestSupport.FakeServer server =
        new MlflowCabTestSupport.FakeServer(r -> new MlflowCabTestSupport.Reply(200, "{}"))) {
      MlflowCabIssuer noOverride =
          new MlflowCabIssuer(
              MlflowCabIssuer.newHttpClient(), System::currentTimeMillis, MlflowCabTestSupport.env(Map.of()));
      for (String url :
          new String[] {
            server.baseUrl() + "/v1alpha/x:issueReadArtifactToken",
            "http://mlflow.example.com/x",
            "https://mlflow.googleapis.com.evil.example.com/x",
          }) {
        MlflowCabScope scope = MlflowCabScope.of(url, "{}", "READ_ONLY", "");
        assertThrows(
            IOException.class,
            () -> noOverride.issue(scope, MlflowCabTestSupport.staticCaller(CALLER), fastRetries(), "driver"));
      }
      assertEquals(0, server.count.get());
    }
  }

  @Test
  void doesNotFollowRedirects() throws Exception {
    try (MlflowCabTestSupport.FakeServer target =
            new MlflowCabTestSupport.FakeServer(r -> new MlflowCabTestSupport.Reply(200, "{}"));
        MlflowCabTestSupport.FakeServer redirector =
            new MlflowCabTestSupport.FakeServer(
                r ->
                    new MlflowCabTestSupport.Reply(
                        302, "", Map.of("Location", target.baseUrl() + "/steal")))) {
      MlflowCabScope scope =
          MlflowCabScope.of(redirector.baseUrl() + "/v1alpha/x:issueReadArtifactToken", "{}", "READ_ONLY", "");
      assertThrows(
          IOException.class,
          () ->
              issuer(redirector.baseUrl())
                  .issue(scope, MlflowCabTestSupport.staticCaller(CALLER), fastRetries(), "driver"));
      assertEquals(0, target.count.get());
    }
  }

  @Test
  void redactionScrubsTokensFromArbitraryText() {
    String text =
        "Authorization: Bearer ya29." + "abc.def {\"accessToken\":\"secret-value\"} access_token=xyz&x=1";
    String redacted = MlflowCabRedaction.redact(text);
    assertFalse(redacted.contains("ya29."));
    assertFalse(redacted.contains("secret-value"));
    assertFalse(redacted.contains("xyz"));
    assertTrue(redacted.contains("<redacted>"));
  }
}
