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
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.charset.StandardCharsets;
import java.time.Duration;
import java.time.Instant;
import java.time.OffsetDateTime;
import java.time.format.DateTimeParseException;
import java.util.Map;
import java.util.Optional;
import java.util.concurrent.ThreadLocalRandom;
import java.util.function.LongSupplier;
import java.util.function.UnaryOperator;
import org.apache.hadoop.conf.Configuration;

/**
 * Calls the Managed Service for MLflow {@code :issue*ArtifactToken} RPCs.
 *
 * <p>Requests are only sent to allowlisted origins, retry transient HTTP 429/500/502/503/504 and
 * connection failures with bounded exponential backoff, never follow redirects, and accept both
 * {@code camelCase} and {@code snake_case} response fields. Error messages are redacted.
 */
final class MlflowCabIssuer {

  /** Result of one successful issuance. */
  static final class Issued {
    final String token;
    final long expiresAtMs;
    final String artifactUri;
    final int attempts;

    Issued(String token, long expiresAtMs, String artifactUri, int attempts) {
      this.token = token;
      this.expiresAtMs = expiresAtMs;
      this.artifactUri = artifactUri;
      this.attempts = attempts;
    }

    @Override
    public String toString() {
      return "Issued{expiresAtMs=" + expiresAtMs + ", artifactUri=" + artifactUri + "}";
    }
  }

  private static final long DEFAULT_TTL_MS = 3_600_000L;
  private static final long MAX_BACKOFF_MS = 8_000L;

  private final HttpClient http;
  private final LongSupplier clock;
  private final UnaryOperator<String> env;

  MlflowCabIssuer(HttpClient http, LongSupplier clock, UnaryOperator<String> env) {
    this.http = http;
    this.clock = clock;
    this.env = env;
  }

  static HttpClient newHttpClient() {
    return HttpClient.newBuilder()
        .version(HttpClient.Version.HTTP_1_1)
        .followRedirects(HttpClient.Redirect.NEVER)
        .connectTimeout(Duration.ofSeconds(20))
        .build();
  }

  Issued issue(
      MlflowCabScope scope, MlflowCabCallerCredentials caller, Configuration conf, String role)
      throws IOException {
    MlflowCabEndpointPolicy.validate(scope.issueUrl(), conf, env);
    int maxAttempts =
        Math.max(
            1,
            conf == null
                ? MlflowCabConstants.DEFAULT_HTTP_MAX_ATTEMPTS
                : conf.getInt(
                    MlflowCabConstants.CONF_HTTP_MAX_ATTEMPTS,
                    MlflowCabConstants.DEFAULT_HTTP_MAX_ATTEMPTS));
    long backoffMs =
        Math.max(
            1L,
            conf == null
                ? MlflowCabConstants.DEFAULT_HTTP_INITIAL_BACKOFF_MS
                : conf.getLong(
                    MlflowCabConstants.CONF_HTTP_INITIAL_BACKOFF_MS,
                    MlflowCabConstants.DEFAULT_HTTP_INITIAL_BACKOFF_MS));
    String lastError = "no attempt made";
    boolean refreshedCaller = false;
    for (int attempt = 1; attempt <= maxAttempts; attempt++) {
      HttpResponse<String> response;
      try {
        HttpRequest request =
            HttpRequest.newBuilder(URI.create(scope.issueUrl()))
                .timeout(Duration.ofSeconds(60))
                .header("Content-Type", "application/json")
                .header("Authorization", "Bearer " + caller.token())
                .header("User-Agent", userAgent(role))
                .header(
                    "x-goog-api-client",
                    MlflowCabConstants.USER_AGENT_PRODUCT + "/" + MlflowCabConstants.VERSION)
                .POST(HttpRequest.BodyPublishers.ofString(scope.issueBody(), StandardCharsets.UTF_8))
                .build();
        response = http.send(request, HttpResponse.BodyHandlers.ofString(StandardCharsets.UTF_8));
      } catch (InterruptedException e) {
        Thread.currentThread().interrupt();
        throw new IOException("Interrupted while issuing CAB token for scope " + scope.scopeId(), e);
      } catch (IOException e) {
        lastError = MlflowCabRedaction.describe(e);
        if (attempt < maxAttempts) {
          sleep(nextBackoff(backoffMs, attempt, null));
          continue;
        }
        break;
      }
      int status = response.statusCode();
      if (status == 200) {
        return parse(response.body(), scope, attempt);
      }
      lastError = "HTTP " + status + ": " + summarizeError(response.body());
      if (status == 401 && !refreshedCaller) {
        refreshedCaller = true;
        caller.invalidate();
        continue;
      }
      if (isRetryable(status) && attempt < maxAttempts) {
        sleep(nextBackoff(backoffMs, attempt, response.headers().firstValue("Retry-After")));
        continue;
      }
      break;
    }
    throw new IOException(
        "MLflow CAB token issuance failed for scope "
            + scope.scopeId()
            + " ("
            + scope.mode()
            + "): "
            + lastError);
  }

  static boolean isRetryable(int status) {
    return status == 429 || status == 500 || status == 502 || status == 503 || status == 504;
  }

  private static String userAgent(String role) {
    return MlflowCabConstants.USER_AGENT_PRODUCT
        + "/"
        + MlflowCabConstants.VERSION
        + " (role="
        + role
        + "; java="
        + System.getProperty("java.specification.version", "unknown")
        + ")";
  }

  private Issued parse(String body, MlflowCabScope scope, int attempts) throws IOException {
    Map<String, Object> json;
    try {
      json = MlflowCabJson.parseObject(body);
    } catch (IllegalArgumentException e) {
      throw new IOException("CAB token response is not valid JSON for scope " + scope.scopeId());
    }
    Map<String, Object> credentials = MlflowCabJson.object(json, "credentials");
    String token =
        MlflowCabJson.firstString(
            credentials, "accessToken", "access_token", "token");
    if (token == null) {
      token = MlflowCabJson.firstString(json, "accessToken", "access_token");
    }
    if (token == null) {
      throw new IOException(
          "CAB token response did not include credentials.accessToken for scope "
              + scope.scopeId());
    }
    String expire = MlflowCabJson.firstString(credentials, "expireTime", "expire_time");
    if (expire == null) {
      expire = MlflowCabJson.firstString(json, "expireTime", "expire_time");
    }
    long expiresAtMs = clock.getAsLong() + DEFAULT_TTL_MS;
    if (expire != null) {
      expiresAtMs = parseTimestamp(expire, expiresAtMs);
    }
    String artifactUri = MlflowCabJson.firstString(json, "artifactUri", "artifact_uri");
    return new Issued(token, expiresAtMs, artifactUri == null ? "" : artifactUri, attempts);
  }

  static long parseTimestamp(String value, long fallback) {
    try {
      return Instant.parse(value).toEpochMilli();
    } catch (DateTimeParseException e) {
      try {
        return OffsetDateTime.parse(value).toInstant().toEpochMilli();
      } catch (DateTimeParseException e2) {
        return fallback;
      }
    }
  }

  private static String summarizeError(String body) {
    if (body == null || body.isEmpty()) {
      return "(empty response)";
    }
    try {
      Map<String, Object> json = MlflowCabJson.parseObject(body);
      Map<String, Object> error = MlflowCabJson.object(json, "error");
      if (error != null) {
        String status = MlflowCabJson.firstString(error, "status");
        String message = MlflowCabJson.firstString(error, "message");
        return MlflowCabRedaction.redact(
            (status == null ? "" : status + " ") + (message == null ? "" : message));
      }
      String message = MlflowCabJson.firstString(json, "message", "error_code");
      if (message != null) {
        return MlflowCabRedaction.redact(message);
      }
    } catch (IllegalArgumentException ignored) {
      // Non-JSON error body; fall through to a bounded redacted excerpt.
    }
    String excerpt = body.length() > 200 ? body.substring(0, 200) + "..." : body;
    return MlflowCabRedaction.redact(excerpt);
  }

  private static long nextBackoff(long initialMs, int attempt, Optional<String> retryAfter) {
    if (retryAfter != null && retryAfter.isPresent()) {
      try {
        long seconds = Long.parseLong(retryAfter.get().trim());
        return Math.min(MAX_BACKOFF_MS, Math.max(0L, seconds * 1000L));
      } catch (NumberFormatException ignored) {
        // Non-numeric Retry-After; use exponential backoff instead.
      }
    }
    long exp = initialMs << Math.min(attempt - 1, 16);
    long capped = Math.min(MAX_BACKOFF_MS, exp);
    return capped / 2 + ThreadLocalRandom.current().nextLong(capped / 2 + 1);
  }

  private static void sleep(long millis) throws IOException {
    try {
      Thread.sleep(millis);
    } catch (InterruptedException e) {
      Thread.currentThread().interrupt();
      throw new IOException("Interrupted during CAB issuance backoff", e);
    }
  }
}
