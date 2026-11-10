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
import java.net.InetAddress;
import java.net.URI;
import java.net.URISyntaxException;
import java.net.UnknownHostException;
import java.util.Locale;
import java.util.Set;
import java.util.function.UnaryOperator;
import java.util.regex.Pattern;
import org.apache.hadoop.conf.Configuration;

/**
 * Destination policy for requests that carry Google credentials.
 *
 * <p>Mirrors the Python plugin's transport policy: credentials may only be sent to {@code
 * https://*.googleapis.com} on port 443, or to the exact {@code (scheme, host, port)} origin that
 * an operator configured through {@code MLFLOW_GCP_ENDPOINT_OVERRIDE} (or the non-secret Hadoop key
 * {@code fs.mlflow.cab.endpoint.override}). Plain HTTP is only ever permitted for loopback hosts.
 */
public final class MlflowCabEndpointPolicy {

  private static final Set<String> LOOPBACK_HOSTS = Set.of("localhost", "127.0.0.1", "::1");
  private static final String GOOGLEAPIS_SUFFIX = ".googleapis.com";
  private static final Pattern DNS_LABEL = Pattern.compile("^[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?$");
  private static final Pattern HOST_CHARS = Pattern.compile("^[a-z0-9_-]+(?:\\.[a-z0-9_-]+)*$");

  /** Normalized endpoint origin. */
  public static final class Origin {
    private final String scheme;
    private final String host;
    private final int port;

    Origin(String scheme, String host, int port) {
      this.scheme = scheme;
      this.host = host;
      this.port = port;
    }

    public String scheme() {
      return scheme;
    }

    public String host() {
      return host;
    }

    public int port() {
      return port;
    }

    boolean isLoopback() {
      return LOOPBACK_HOSTS.contains(host);
    }

    @Override
    public boolean equals(Object other) {
      if (!(other instanceof Origin)) {
        return false;
      }
      Origin that = (Origin) other;
      return scheme.equals(that.scheme) && host.equals(that.host) && port == that.port;
    }

    @Override
    public int hashCode() {
      return (scheme.hashCode() * 31 + host.hashCode()) * 31 + port;
    }

    @Override
    public String toString() {
      String display = host.contains(":") ? "[" + host + "]" : host;
      return scheme + "://" + display + ":" + port;
    }
  }

  private MlflowCabEndpointPolicy() {}

  /**
   * Parses and validates the scheme, host, and effective port of {@code rawUrl}.
   *
   * @throws IllegalArgumentException if the URL is malformed or uses non-loopback HTTP
   */
  public static Origin parse(String rawUrl) {
    if (rawUrl == null || rawUrl.isEmpty()) {
      throw new IllegalArgumentException("Endpoint URL must be a non-empty string.");
    }
    for (int i = 0; i < rawUrl.length(); i++) {
      char ch = rawUrl.charAt(i);
      if (ch < 0x20 || ch == 0x7f) {
        throw new IllegalArgumentException("Endpoint URL contains invalid control characters.");
      }
    }
    if (Character.isWhitespace(rawUrl.charAt(0))
        || Character.isWhitespace(rawUrl.charAt(rawUrl.length() - 1))) {
      throw new IllegalArgumentException(
          "Endpoint URL must not have leading or trailing whitespace.");
    }
    URI uri;
    try {
      uri = new URI(rawUrl);
    } catch (URISyntaxException e) {
      throw new IllegalArgumentException("Endpoint URL is malformed.", e);
    }
    String scheme = uri.getScheme() == null ? "" : uri.getScheme().toLowerCase(Locale.ROOT);
    if (!scheme.equals("https") && !scheme.equals("http")) {
      throw new IllegalArgumentException(
          "Invalid scheme '" + scheme + "': only https:// and loopback http:// are allowed.");
    }
    String rawAuthority = uri.getRawAuthority();
    if (rawAuthority == null
        || rawAuthority.contains("@")
        || rawAuthority.contains("\\")
        || uri.getRawUserInfo() != null) {
      throw new IllegalArgumentException("Endpoint URL must include a valid scheme and host.");
    }
    String rawHost = uri.getHost();
    if (rawHost == null || rawHost.isEmpty()) {
      throw new IllegalArgumentException("Endpoint URL must include a valid scheme and host.");
    }
    String host = rawHost.toLowerCase(Locale.ROOT);
    if (host.startsWith("[") && host.endsWith("]")) {
      String literal = host.substring(1, host.length() - 1);
      if (literal.contains("%") || !literal.contains(":")) {
        throw new IllegalArgumentException("Endpoint URL contains an invalid IPv6 host literal.");
      }
      try {
        InetAddress address = InetAddress.getByName(literal);
        host = address.isLoopbackAddress() ? "::1" : address.getHostAddress();
      } catch (UnknownHostException e) {
        throw new IllegalArgumentException("Endpoint URL contains an invalid IPv6 host literal.", e);
      }
    } else {
      while (host.endsWith(".")) {
        host = host.substring(0, host.length() - 1);
      }
      if (host.isEmpty() || !HOST_CHARS.matcher(host).matches()) {
        throw new IllegalArgumentException("Endpoint URL contains invalid characters in host.");
      }
    }
    int port = uri.getPort();
    if (port == 0) {
      throw new IllegalArgumentException("Endpoint URL is malformed or contains an invalid port.");
    }
    if (port < 0) {
      if (rawAuthority.endsWith(":")) {
        throw new IllegalArgumentException(
            "Endpoint URL is malformed or contains an invalid port.");
      }
      port = scheme.equals("https") ? 443 : 80;
    }
    Origin origin = new Origin(scheme, host, port);
    if (scheme.equals("http") && !origin.isLoopback()) {
      throw new IllegalArgumentException(
          "Insecure HTTP is only permitted on loopback addresses; got '" + host + "'.");
    }
    return origin;
  }

  /** Returns the operator override origin string (environment first, then Hadoop config). */
  public static String overrideValue(Configuration conf, UnaryOperator<String> env) {
    String fromEnv = env == null ? null : env.apply(MlflowCabConstants.ENV_ENDPOINT_OVERRIDE);
    if (fromEnv != null && !fromEnv.trim().isEmpty()) {
      return fromEnv.trim();
    }
    String fromConf = conf == null ? null : conf.getTrimmed(MlflowCabConstants.CONF_ENDPOINT_OVERRIDE);
    return fromConf == null || fromConf.isEmpty() ? null : fromConf;
  }

  /** Returns true if {@code origin} is an allowed credential destination. */
  public static boolean isAllowed(Origin origin, String overrideRaw) {
    if (origin.scheme().equals("https")
        && origin.port() == 443
        && origin.host().endsWith(GOOGLEAPIS_SUFFIX)) {
      boolean labelsOk = true;
      for (String label : origin.host().split("\\.", -1)) {
        if (!DNS_LABEL.matcher(label).matches()) {
          labelsOk = false;
          break;
        }
      }
      if (labelsOk) {
        return true;
      }
    }
    if (overrideRaw != null && !overrideRaw.isEmpty()) {
      try {
        return origin.equals(parse(overrideRaw));
      } catch (IllegalArgumentException e) {
        return false;
      }
    }
    return false;
  }

  /**
   * Validates that {@code url} may receive Google credentials.
   *
   * @throws IOException if the destination is malformed or not allowlisted
   */
  public static void validate(String url, Configuration conf, UnaryOperator<String> env)
      throws IOException {
    Origin origin;
    try {
      origin = parse(url);
    } catch (IllegalArgumentException e) {
      throw new IOException("Refusing to send Google credentials: " + e.getMessage(), e);
    }
    if (!isAllowed(origin, overrideValue(conf, env))) {
      throw new IOException(
          "Refusing to send Google credentials to untrusted origin '"
              + origin
              + "'. Allowed destinations are https://*.googleapis.com (port 443) or the exact "
              + "origin configured in "
              + MlflowCabConstants.ENV_ENDPOINT_OVERRIDE
              + " (required to opt into loopback endpoints).");
    }
  }
}
