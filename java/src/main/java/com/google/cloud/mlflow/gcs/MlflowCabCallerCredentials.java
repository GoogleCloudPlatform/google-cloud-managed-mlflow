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
import java.net.URLEncoder;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.security.GeneralSecurityException;
import java.security.KeyFactory;
import java.security.PrivateKey;
import java.security.Signature;
import java.security.spec.PKCS8EncodedKeySpec;
import java.time.Duration;
import java.util.Base64;
import java.util.Map;
import java.util.function.LongSupplier;
import java.util.function.UnaryOperator;
import org.apache.hadoop.conf.Configuration;

/**
 * Resolves the driver's own Google OAuth token used to call the MLflow CAB issuance RPCs.
 *
 * <p>Follows the Application Default Credentials order: {@code GOOGLE_APPLICATION_CREDENTIALS}
 * ({@code authorized_user} or {@code service_account} JSON), the gcloud well-known ADC file, and
 * finally the Compute Engine metadata server (honoring the standard {@code GCE_METADATA_HOST}
 * override). Tokens are cached in memory and never logged.
 */
class MlflowCabCallerCredentials {

  static final String CLOUD_PLATFORM_SCOPE = "https://www.googleapis.com/auth/cloud-platform";
  private static final String DEFAULT_TOKEN_URI = "https://oauth2.googleapis.com/token";
  private static final String DEFAULT_METADATA_HOST = "169.254.169.254";
  private static final long EARLY_EXPIRY_MS = 60_000L;

  private final Configuration conf;
  private final UnaryOperator<String> env;
  private final LongSupplier clock;
  private final HttpClient http;
  private final Object lock = new Object();
  private volatile String cachedToken;
  private volatile long cachedExpiresAtMs;
  private volatile String lastSourceKind = "none";

  MlflowCabCallerCredentials(
      Configuration conf, UnaryOperator<String> env, LongSupplier clock, HttpClient http) {
    this.conf = conf;
    this.env = env;
    this.clock = clock;
    this.http = http;
  }

  /** Returns a valid caller access token, refreshing it when close to expiry. */
  String token() throws IOException {
    long now = clock.getAsLong();
    String current = cachedToken;
    if (current != null && now + EARLY_EXPIRY_MS < cachedExpiresAtMs) {
      return current;
    }
    synchronized (lock) {
      now = clock.getAsLong();
      if (cachedToken != null && now + EARLY_EXPIRY_MS < cachedExpiresAtMs) {
        return cachedToken;
      }
      fetchFresh();
      return cachedToken;
    }
  }

  /** Drops the cached token so the next call re-resolves credentials (e.g. after HTTP 401). */
  void invalidate() {
    synchronized (lock) {
      cachedToken = null;
      cachedExpiresAtMs = 0L;
    }
  }

  String lastSourceKind() {
    return lastSourceKind;
  }

  private void fetchFresh() throws IOException {
    String explicit = env.apply("GOOGLE_APPLICATION_CREDENTIALS");
    if (explicit != null && !explicit.trim().isEmpty()) {
      fromCredentialsFile(Paths.get(explicit.trim()), "google_application_credentials");
      return;
    }
    Path wellKnown = wellKnownAdcFile();
    if (wellKnown != null && Files.isRegularFile(wellKnown)) {
      fromCredentialsFile(wellKnown, "gcloud_adc_file");
      return;
    }
    fromMetadataServer();
  }

  private Path wellKnownAdcFile() {
    String sdkConfig = env.apply("CLOUDSDK_CONFIG");
    if (sdkConfig != null && !sdkConfig.trim().isEmpty()) {
      return Paths.get(sdkConfig.trim(), "application_default_credentials.json");
    }
    String home = env.apply("HOME");
    if (home == null || home.trim().isEmpty()) {
      return null;
    }
    return Paths.get(home.trim(), ".config", "gcloud", "application_default_credentials.json");
  }

  private void fromCredentialsFile(Path file, String kind) throws IOException {
    Map<String, Object> json;
    try {
      json = MlflowCabJson.parseObject(new String(Files.readAllBytes(file), StandardCharsets.UTF_8));
    } catch (IllegalArgumentException e) {
      throw new IOException("Credentials file is not valid JSON: " + file, e);
    }
    String type = MlflowCabJson.firstString(json, "type");
    if ("authorized_user".equals(type)) {
      String tokenUri = orDefault(MlflowCabJson.firstString(json, "token_uri"), DEFAULT_TOKEN_URI);
      String form =
          "grant_type=refresh_token"
              + "&client_id="
              + enc(MlflowCabJson.firstString(json, "client_id"))
              + "&client_secret="
              + enc(MlflowCabJson.firstString(json, "client_secret"))
              + "&refresh_token="
              + enc(MlflowCabJson.firstString(json, "refresh_token"));
      exchange(tokenUri, form, kind + ":authorized_user");
    } else if ("service_account".equals(type)) {
      String tokenUri = orDefault(MlflowCabJson.firstString(json, "token_uri"), DEFAULT_TOKEN_URI);
      String assertion = signJwtAssertion(json, tokenUri);
      String form =
          "grant_type="
              + enc("urn:ietf:params:oauth:grant-type:jwt-bearer")
              + "&assertion="
              + enc(assertion);
      exchange(tokenUri, form, kind + ":service_account");
    } else {
      throw new IOException(
          "Unsupported credentials type '" + type + "' in " + file
              + "; use authorized_user, service_account, or the metadata server.");
    }
  }

  private String signJwtAssertion(Map<String, Object> json, String tokenUri) throws IOException {
    String email = MlflowCabJson.firstString(json, "client_email");
    String pem = MlflowCabJson.firstString(json, "private_key");
    if (email == null || pem == null) {
      throw new IOException("service_account credentials require client_email and private_key");
    }
    long nowSec = clock.getAsLong() / 1000L;
    String keyId = MlflowCabJson.firstString(json, "private_key_id");
    String header =
        "{\"alg\":\"RS256\",\"typ\":\"JWT\""
            + (keyId == null ? "" : ",\"kid\":" + MlflowCabJson.quote(keyId))
            + "}";
    String claims =
        "{\"iss\":"
            + MlflowCabJson.quote(email)
            + ",\"scope\":"
            + MlflowCabJson.quote(CLOUD_PLATFORM_SCOPE)
            + ",\"aud\":"
            + MlflowCabJson.quote(tokenUri)
            + ",\"iat\":"
            + nowSec
            + ",\"exp\":"
            + (nowSec + 3600L)
            + "}";
    Base64.Encoder b64 = Base64.getUrlEncoder().withoutPadding();
    String signingInput =
        b64.encodeToString(header.getBytes(StandardCharsets.UTF_8))
            + "."
            + b64.encodeToString(claims.getBytes(StandardCharsets.UTF_8));
    try {
      String body =
          pem.replace("-----BEGIN PRIVATE KEY-----", "")
              .replace("-----END PRIVATE KEY-----", "")
              .replaceAll("\\s", "");
      PrivateKey key =
          KeyFactory.getInstance("RSA")
              .generatePrivate(new PKCS8EncodedKeySpec(Base64.getDecoder().decode(body)));
      Signature signer = Signature.getInstance("SHA256withRSA");
      signer.initSign(key);
      signer.update(signingInput.getBytes(StandardCharsets.US_ASCII));
      return signingInput + "." + b64.encodeToString(signer.sign());
    } catch (GeneralSecurityException | IllegalArgumentException e) {
      throw new IOException("Unable to sign service account assertion", e);
    }
  }

  private void exchange(String tokenUri, String form, String kind) throws IOException {
    MlflowCabEndpointPolicy.validate(tokenUri, conf, env);
    HttpRequest request =
        HttpRequest.newBuilder(URI.create(tokenUri))
            .timeout(Duration.ofSeconds(30))
            .header("Content-Type", "application/x-www-form-urlencoded")
            .POST(HttpRequest.BodyPublishers.ofString(form, StandardCharsets.UTF_8))
            .build();
    accept(send(request), kind);
  }

  private void fromMetadataServer() throws IOException {
    String host = env.apply("GCE_METADATA_HOST");
    if (host == null || host.trim().isEmpty()) {
      host = DEFAULT_METADATA_HOST;
    }
    URI uri =
        URI.create(
            "http://"
                + host.trim()
                + "/computeMetadata/v1/instance/service-accounts/default/token");
    HttpRequest request =
        HttpRequest.newBuilder(uri)
            .timeout(Duration.ofSeconds(10))
            .header("Metadata-Flavor", "Google")
            .GET()
            .build();
    IOException last = null;
    for (int attempt = 0; attempt < 3; attempt++) {
      try {
        accept(send(request), "metadata_server");
        return;
      } catch (IOException e) {
        last = e;
        sleepQuietly(200L * (attempt + 1));
      }
    }
    throw new IOException(
        "Unable to obtain a Google access token for MLflow CAB issuance (no credentials file and "
            + "the metadata server is unreachable): "
            + MlflowCabRedaction.describe(last),
        last);
  }

  private HttpResponse<String> send(HttpRequest request) throws IOException {
    try {
      return http.send(request, HttpResponse.BodyHandlers.ofString(StandardCharsets.UTF_8));
    } catch (InterruptedException e) {
      Thread.currentThread().interrupt();
      throw new IOException("Interrupted while fetching caller credentials", e);
    }
  }

  private void accept(HttpResponse<String> response, String kind) throws IOException {
    if (response.statusCode() != 200) {
      throw new IOException(
          "Caller credential exchange failed with HTTP "
              + response.statusCode()
              + " ("
              + kind
              + "): "
              + MlflowCabRedaction.redact(abbreviate(response.body())));
    }
    Map<String, Object> json;
    try {
      json = MlflowCabJson.parseObject(response.body());
    } catch (IllegalArgumentException e) {
      throw new IOException("Caller credential response is not valid JSON (" + kind + ")", e);
    }
    String token = MlflowCabJson.firstString(json, "access_token", "accessToken");
    if (token == null) {
      throw new IOException("Caller credential response did not include an access token");
    }
    Number expiresIn = MlflowCabJson.firstNumber(json, "expires_in", "expiresIn");
    long ttlMs = expiresIn == null ? 3_000_000L : expiresIn.longValue() * 1000L;
    cachedToken = token;
    cachedExpiresAtMs = clock.getAsLong() + ttlMs;
    lastSourceKind = kind;
  }

  private static String abbreviate(String body) {
    if (body == null) {
      return "";
    }
    return body.length() > 300 ? body.substring(0, 300) + "..." : body;
  }

  private static String orDefault(String value, String fallback) {
    return value == null || value.isEmpty() ? fallback : value;
  }

  private static String enc(String value) {
    return URLEncoder.encode(value == null ? "" : value, StandardCharsets.UTF_8);
  }

  private static void sleepQuietly(long millis) {
    try {
      Thread.sleep(millis);
    } catch (InterruptedException e) {
      Thread.currentThread().interrupt();
    }
  }
}
