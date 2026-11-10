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

import java.nio.charset.StandardCharsets;
import java.util.Objects;

/**
 * A short-lived downscoped bearer token bound to one CAB scope.
 *
 * <p>The secret is only reachable through the package-private {@link #secret()} accessor; {@link
 * #toString()} never renders it.
 */
public final class MlflowCabToken {

  private static final String FORMAT_VERSION = "v1";

  private final String scopeId;
  private final String secret;
  private final long expiresAtMs;
  private final long generation;

  MlflowCabToken(String scopeId, String secret, long expiresAtMs, long generation) {
    this.scopeId = Objects.requireNonNull(scopeId, "scopeId");
    this.secret = secret == null ? "" : secret;
    this.expiresAtMs = expiresAtMs;
    this.generation = generation;
  }

  String secret() {
    return secret;
  }

  public String scopeId() {
    return scopeId;
  }

  public long expiresAtMs() {
    return expiresAtMs;
  }

  public long generation() {
    return generation;
  }

  /** Returns true if the token is non-empty and valid for at least {@code minValidityMs}. */
  public boolean isUsable(long nowMs, long minValidityMs) {
    return !secret.isEmpty() && nowMs + minValidityMs < expiresAtMs;
  }

  /** Serializes the token for transport inside Hadoop {@code Credentials} secret keys. */
  byte[] encode() {
    String payload = FORMAT_VERSION + "\n" + expiresAtMs + "\n" + generation + "\n" + secret;
    return payload.getBytes(StandardCharsets.UTF_8);
  }

  /** Decodes a token previously produced by {@link #encode()}, or null if malformed. */
  static MlflowCabToken decode(String scopeId, byte[] raw) {
    if (raw == null || raw.length == 0) {
      return null;
    }
    String[] parts = new String(raw, StandardCharsets.UTF_8).split("\n", 4);
    if (parts.length != 4 || !FORMAT_VERSION.equals(parts[0])) {
      return null;
    }
    try {
      return new MlflowCabToken(
          scopeId, parts[3], Long.parseLong(parts[1].trim()), Long.parseLong(parts[2].trim()));
    } catch (NumberFormatException e) {
      return null;
    }
  }

  @Override
  public String toString() {
    return "MlflowCabToken{scope="
        + scopeId
        + ", expiresAtMs="
        + expiresAtMs
        + ", generation="
        + generation
        + ", secret=<redacted>}";
  }
}
