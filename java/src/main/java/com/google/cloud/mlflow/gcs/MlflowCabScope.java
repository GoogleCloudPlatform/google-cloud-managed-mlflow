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
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.Locale;
import java.util.Objects;
import java.util.regex.Pattern;

/**
 * Immutable description of one Credential Access Boundary (CAB) scope.
 *
 * <p>A scope is identified by the non-secret {@code (issueUrl, issueBody)} pair used to mint its
 * token. The derived {@link #scopeId()} is a stable, lowercase hex digest that is embedded in
 * {@code mlflow-gs://<scopeId>@<bucket>/...} paths so every filesystem operation is bound to
 * exactly one downscoped token, which keeps multi-scope transitions free of cross-scope bleed.
 */
public final class MlflowCabScope {

  private static final Pattern SCOPE_ID = Pattern.compile("^[0-9a-f]{16,64}$");

  private final String scopeId;
  private final String issueUrl;
  private final String issueBody;
  private final String mode;
  private final String boundary;

  MlflowCabScope(String issueUrl, String issueBody, String mode, String boundary) {
    this.issueUrl = Objects.requireNonNull(issueUrl, "issueUrl");
    this.issueBody = issueBody == null || issueBody.isEmpty() ? "{}" : issueBody;
    this.mode = mode == null ? "" : mode.toUpperCase(Locale.ROOT);
    this.boundary = boundary == null ? "" : boundary;
    this.scopeId = computeScopeId(this.issueUrl, this.issueBody);
  }

  /** Creates a scope for the given non-secret issuance request. */
  public static MlflowCabScope of(String issueUrl, String issueBody, String mode, String boundary) {
    return new MlflowCabScope(issueUrl, issueBody, mode, boundary);
  }

  /** Computes the stable scope identifier shared by the Python and Java components. */
  public static String computeScopeId(String issueUrl, String issueBody) {
    try {
      MessageDigest digest = MessageDigest.getInstance("SHA-256");
      byte[] hash =
          digest.digest((issueUrl + "\n" + issueBody).getBytes(StandardCharsets.UTF_8));
      StringBuilder hex = new StringBuilder();
      for (int i = 0; i < 12; i++) {
        hex.append(String.format("%02x", hash[i] & 0xff));
      }
      return hex.toString();
    } catch (NoSuchAlgorithmException e) {
      throw new IllegalStateException("SHA-256 is unavailable", e);
    }
  }

  /** Returns true if {@code value} is a syntactically valid scope identifier. */
  public static boolean isValidScopeId(String value) {
    return value != null && SCOPE_ID.matcher(value).matches();
  }

  public String scopeId() {
    return scopeId;
  }

  public String issueUrl() {
    return issueUrl;
  }

  public String issueBody() {
    return issueBody;
  }

  public String mode() {
    return mode;
  }

  public String boundary() {
    return boundary;
  }

  @Override
  public String toString() {
    return "MlflowCabScope{id=" + scopeId + ", mode=" + mode + ", boundary=" + boundary + "}";
  }
}
