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

import com.sun.net.httpserver.HttpExchange;
import com.sun.net.httpserver.HttpServer;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.net.InetSocketAddress;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.Function;
import java.util.function.UnaryOperator;

/** Loopback HTTP server and helpers shared by the JUnit suite (no external network). */
final class MlflowCabTestSupport {

  private MlflowCabTestSupport() {}

  /** A recorded HTTP request. */
  static final class Recorded {
    final String method;
    final String path;
    final String query;
    final Map<String, String> headers;
    final String body;

    Recorded(String method, String path, String query, Map<String, String> headers, String body) {
      this.method = method;
      this.path = path;
      this.query = query;
      this.headers = headers;
      this.body = body;
    }

    String header(String name) {
      return headers.get(name.toLowerCase(java.util.Locale.ROOT));
    }
  }

  /** A canned HTTP response. */
  static final class Reply {
    final int status;
    final String body;
    final Map<String, String> headers;
    final byte[] raw;

    Reply(int status, String body) {
      this(status, body, Collections.emptyMap());
    }

    Reply(int status, String body, Map<String, String> headers) {
      this.status = status;
      this.body = body;
      this.headers = headers;
      this.raw = null;
    }

    Reply(int status, byte[] raw, Map<String, String> headers) {
      this.status = status;
      this.body = null;
      this.headers = headers;
      this.raw = raw;
    }
  }

  /** Minimal programmable loopback HTTP server. */
  static final class FakeServer implements AutoCloseable {
    private final HttpServer server;
    final List<Recorded> requests = Collections.synchronizedList(new ArrayList<>());
    final AtomicInteger count = new AtomicInteger();
    private volatile Function<Recorded, Reply> handler;

    FakeServer(Function<Recorded, Reply> handler) throws IOException {
      this.handler = handler;
      server = HttpServer.create(new InetSocketAddress("127.0.0.1", 0), 0);
      server.createContext("/", this::handle);
      server.start();
    }

    void setHandler(Function<Recorded, Reply> newHandler) {
      handler = newHandler;
    }

    String baseUrl() {
      return "http://127.0.0.1:" + server.getAddress().getPort();
    }

    private void handle(HttpExchange exchange) throws IOException {
      Map<String, String> headers = new HashMap<>();
      for (Map.Entry<String, List<String>> entry : exchange.getRequestHeaders().entrySet()) {
        headers.put(
            entry.getKey().toLowerCase(java.util.Locale.ROOT),
            entry.getValue().isEmpty() ? "" : entry.getValue().get(0));
      }
      String body;
      try (InputStream in = exchange.getRequestBody()) {
        body = new String(readAll(in), StandardCharsets.UTF_8);
      }
      Recorded recorded =
          new Recorded(
              exchange.getRequestMethod(),
              exchange.getRequestURI().getRawPath(),
              exchange.getRequestURI().getRawQuery(),
              headers,
              body);
      requests.add(recorded);
      count.incrementAndGet();
      Reply reply;
      try {
        reply = handler.apply(recorded);
      } catch (RuntimeException e) {
        reply = new Reply(500, "{\"error\":{\"message\":\"handler failure\"}}");
      }
      byte[] bytes = reply.raw != null ? reply.raw : reply.body.getBytes(StandardCharsets.UTF_8);
      for (Map.Entry<String, String> entry : reply.headers.entrySet()) {
        exchange.getResponseHeaders().add(entry.getKey(), entry.getValue());
      }
      if (!reply.headers.containsKey("Content-Type")) {
        exchange.getResponseHeaders().add("Content-Type", "application/json");
      }
      exchange.sendResponseHeaders(reply.status, bytes.length == 0 ? -1 : bytes.length);
      if (bytes.length > 0) {
        try (OutputStream out = exchange.getResponseBody()) {
          out.write(bytes);
        }
      }
      exchange.close();
    }

    @Override
    public void close() {
      server.stop(0);
    }
  }

  static byte[] readAll(InputStream in) throws IOException {
    ByteArrayOutputStream out = new ByteArrayOutputStream();
    byte[] buffer = new byte[8192];
    int read;
    while ((read = in.read(buffer)) >= 0) {
      out.write(buffer, 0, read);
    }
    return out.toByteArray();
  }

  /** Builds an environment lookup backed by a map (no access to the real process env). */
  static UnaryOperator<String> env(Map<String, String> values) {
    Map<String, String> copy = new HashMap<>(values);
    return copy::get;
  }

  /** Returns a CAB issuance JSON response in camelCase or snake_case. */
  static String cabResponse(String token, String expireTime, String artifactUri, boolean camel) {
    if (camel) {
      return "{\"artifactUri\":\""
          + artifactUri
          + "\",\"credentials\":{\"accessToken\":\""
          + token
          + "\",\"expireTime\":\""
          + expireTime
          + "\"}}";
    }
    return "{\"artifact_uri\":\""
        + artifactUri
        + "\",\"credentials\":{\"access_token\":\""
        + token
        + "\",\"expire_time\":\""
        + expireTime
        + "\"}}";
  }

  /** Fixed caller credentials that never touch the network. */
  static MlflowCabCallerCredentials staticCaller(String token) {
    return new MlflowCabCallerCredentials(null, k -> null, System::currentTimeMillis, null) {
      @Override
      String token() {
        return token;
      }

      @Override
      void invalidate() {}
    };
  }

  /**
   * Wraps {@code token} for a driver's base64 X.509 {@code clientWrapKey} the way the Python client
   * does: UTF-8 slices of at most {@code keyBytes - 2*32 - 2 - 2} bytes, each prefixed with a
   * {@code [index, count]} header and encrypted with RSA-OAEP (SHA-256 digest, SHA-256 MGF1).
   */
  static String wrapForDriver(String publicKeyB64, String token, boolean sha256Mgf1)
      throws java.security.GeneralSecurityException {
    java.security.PublicKey key =
        java.security.KeyFactory.getInstance("RSA")
            .generatePublic(
                new java.security.spec.X509EncodedKeySpec(
                    java.util.Base64.getDecoder().decode(publicKeyB64)));
    byte[] data = token.getBytes(StandardCharsets.UTF_8);
    int chunk = MlflowCabTokenService.WRAP_BLOCK_BYTES - 2 * 32 - 2 - 2;
    int blocks = Math.max(1, (data.length + chunk - 1) / chunk);
    javax.crypto.Cipher cipher = javax.crypto.Cipher.getInstance(MlflowCabTokenService.WRAP_CIPHER);
    if (sha256Mgf1) {
      cipher.init(javax.crypto.Cipher.ENCRYPT_MODE, key, MlflowCabTokenService.oaepSha256());
    } else {
      // JCE default parameters for this transformation: SHA-256 digest but SHA-1 MGF1.
      cipher.init(javax.crypto.Cipher.ENCRYPT_MODE, key);
    }
    ByteArrayOutputStream out = new ByteArrayOutputStream();
    for (int i = 0; i < blocks; i++) {
      int from = i * chunk;
      int to = Math.min(data.length, from + chunk);
      byte[] block = new byte[2 + (to - from)];
      block[0] = (byte) i;
      block[1] = (byte) blocks;
      System.arraycopy(data, from, block, 2, to - from);
      byte[] encrypted = cipher.doFinal(block);
      out.write(encrypted, 0, encrypted.length);
    }
    return java.util.Base64.getEncoder().encodeToString(out.toByteArray());
  }
}
