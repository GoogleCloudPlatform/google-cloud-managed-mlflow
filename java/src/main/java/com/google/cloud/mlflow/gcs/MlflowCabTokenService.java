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

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.security.GeneralSecurityException;
import java.security.KeyPair;
import java.security.KeyPairGenerator;
import java.security.spec.MGF1ParameterSpec;
import java.util.Base64;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicLong;
import java.util.function.BooleanSupplier;
import java.util.function.LongSupplier;
import java.util.function.UnaryOperator;
import java.util.logging.Level;
import java.util.logging.Logger;
import javax.crypto.Cipher;
import javax.crypto.spec.OAEPParameterSpec;
import javax.crypto.spec.PSource;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.security.Credentials;
import org.apache.hadoop.security.UserGroupInformation;

/**
 * JVM-wide CAB token broker.
 *
 * <p><b>Driver:</b> mints at most one token per scope and renewal window (one MLflow RPC per
 * scope), stores it as a Hadoop {@code Credentials} secret key on the current UGI, and pushes the
 * updated credentials to every executor through Spark's scheduler RPC ({@code
 * updateDelegationTokens}), which also seeds executors that register later. A single daemon thread
 * renews tokens before expiry using the operator-tunable lead window ({@code
 * MLFLOW_CAB_REFRESH_MARGIN_SECONDS} / {@code fs.mlflow.cab.refresh.margin.seconds}).
 *
 * <p><b>Executors:</b> only ever read tokens from local UGI credentials and never contact the
 * MLflow control plane, so issuance cost is O(1) in the number of executors.
 *
 * <p><b>Client-supplied tokens</b> (for example a Spark Connect client whose cluster identity lacks
 * MLflow IAM) never cross the wire in plaintext: each driver JVM generates an in-memory ephemeral
 * RSA-2048 key pair, publishes only the X.509 public key ({@link #clientWrapKey()}), and accepts
 * client tokens solely as RSA-OAEP-SHA256 ciphertext ({@link #unwrapClientToken(String)}).
 */
public final class MlflowCabTokenService {

  /** Who supplies tokens for a scope. */
  public enum Supplier {
    /** The driver mints and renews tokens with its own Google identity. */
    CLUSTER,
    /** A remote client mints tokens and pushes them to the driver (no driver-side IAM needed). */
    CLIENT
  }

  /** Broadcasts the driver's current-user credentials; returns executors notified or -1. */
  interface Broadcaster {
    int broadcast() throws IOException;
  }

  private static final Logger LOG = Logger.getLogger(MlflowCabTokenService.class.getName());
  private static final long MIN_MINT_SPACING_MS = 1_000L;
  private static final long EXECUTOR_MIN_VALIDITY_MS = 1_000L;
  private static final long RENEWAL_TICK_MS = 500L;

  /** Client token wrapping: RSA-2048 with OAEP (SHA-256 digest and SHA-256 MGF1, empty label). */
  static final String WRAP_KEY_ALGORITHM = "RSA";

  static final int WRAP_KEY_BITS = 2048;
  static final String WRAP_CIPHER = "RSA/ECB/OAEPWithSHA-256AndMGF1Padding";

  /** Each RSA block carries a two-byte header (block index, block count) before the payload. */
  static final int WRAP_BLOCK_BYTES = WRAP_KEY_BITS / 8;

  static final int WRAP_HEADER_BYTES = 2;
  static final int WRAP_MAX_BLOCKS = 255;

  private static volatile MlflowCabTokenService instance;

  private final UnaryOperator<String> env;
  private final LongSupplier clock;
  private final MlflowCabIssuer issuer;
  private final Broadcaster broadcaster;
  private final BooleanSupplier driverRole;
  private final ConcurrentHashMap<String, ScopeState> scopes = new ConcurrentHashMap<>();
  private final AtomicLong driverRpcs = new AtomicLong();
  private final AtomicLong executorRpcs = new AtomicLong();
  private final AtomicLong broadcastCount = new AtomicLong();
  private final AtomicLong executorUgiReads = new AtomicLong();
  private final Object callerLock = new Object();
  private final Object wrapKeyLock = new Object();
  private volatile MlflowCabCallerCredentials caller;
  private volatile Configuration callerConf;
  private volatile KeyPair wrapKeys;
  private ScheduledExecutorService renewer;

  static final class ScopeState {
    final MlflowCabScope scope;
    final Object lock = new Object();
    final AtomicLong mints = new AtomicLong();
    final AtomicLong renewals = new AtomicLong();
    final AtomicLong publishes = new AtomicLong();
    final AtomicLong clientPushes = new AtomicLong();
    volatile Supplier supplier = Supplier.CLUSTER;
    volatile MlflowCabToken token;
    volatile Configuration conf;
    volatile boolean active = true;
    volatile long lastMintMs;
    volatile long refreshMarginMs = MlflowCabConstants.DEFAULT_REFRESH_MARGIN_SECONDS * 1000L;
    volatile long publishedGeneration = -1L;
    volatile long generation;
    volatile int lastExecutorsNotified = -1;
    volatile String lastError = "";
    volatile String artifactUri = "";

    ScopeState(MlflowCabScope scope) {
      this.scope = scope;
    }
  }

  MlflowCabTokenService(
      UnaryOperator<String> env,
      LongSupplier clock,
      MlflowCabIssuer issuer,
      Broadcaster broadcaster,
      BooleanSupplier driverRole,
      MlflowCabCallerCredentials callerOverride) {
    this.env = env;
    this.clock = clock;
    this.issuer = issuer;
    this.broadcaster = broadcaster;
    this.driverRole = driverRole;
    this.caller = callerOverride;
  }

  /** Returns the JVM-wide service instance. */
  public static MlflowCabTokenService get() {
    MlflowCabTokenService current = instance;
    if (current == null) {
      synchronized (MlflowCabTokenService.class) {
        if (instance == null) {
          instance =
              new MlflowCabTokenService(
                  System::getenv,
                  System::currentTimeMillis,
                  new MlflowCabIssuer(
                      MlflowCabIssuer.newHttpClient(), System::currentTimeMillis, System::getenv),
                  MlflowCabTokenService::sparkBroadcast,
                  MlflowCabTokenService::sparkDriverOrStandalone,
                  null);
        }
        current = instance;
      }
    }
    return current;
  }

  /** Replaces the JVM-wide instance (tests only). */
  static void setInstanceForTesting(MlflowCabTokenService replacement) {
    synchronized (MlflowCabTokenService.class) {
      if (instance != null && instance != replacement) {
        instance.shutdown();
      }
      instance = replacement;
    }
  }

  private static int sparkBroadcast() throws IOException {
    try {
      return MlflowCabSparkBridge.broadcastCurrentUserCredentials();
    } catch (LinkageError e) {
      return -1;
    }
  }

  private static boolean sparkDriverOrStandalone() {
    try {
      return MlflowCabSparkBridge.isDriverOrStandalone();
    } catch (LinkageError e) {
      return true;
    }
  }

  /** Returns the UGI secret-key alias holding a scope's token. */
  static Text aliasFor(String scopeId) {
    return new Text(MlflowCabConstants.UGI_ALIAS_PREFIX + scopeId);
  }

  /** Resolves the renewal lead window: explicit override, Hadoop key, environment, default. */
  long refreshMarginMs(Configuration conf, Long overrideSeconds) {
    long seconds = MlflowCabConstants.DEFAULT_REFRESH_MARGIN_SECONDS;
    String fromEnv = env.apply(MlflowCabConstants.ENV_REFRESH_MARGIN_SECONDS);
    if (fromEnv != null && !fromEnv.trim().isEmpty()) {
      seconds = parseSeconds(fromEnv, seconds);
    }
    String fromConf = conf == null ? null : conf.getTrimmed(MlflowCabConstants.CONF_REFRESH_MARGIN_SECONDS);
    if (fromConf != null && !fromConf.isEmpty()) {
      seconds = parseSeconds(fromConf, seconds);
    }
    if (overrideSeconds != null && overrideSeconds >= 0) {
      seconds = overrideSeconds;
    }
    return Math.max(0L, seconds) * 1000L;
  }

  private static long parseSeconds(String raw, long fallback) {
    try {
      long value = (long) Double.parseDouble(raw.trim());
      return value < 0 ? fallback : value;
    } catch (NumberFormatException e) {
      return fallback;
    }
  }

  /**
   * Returns this JVM's ephemeral client token wrapping key as base64 X.509 SubjectPublicKeyInfo
   * DER. The private key never leaves the JVM and is regenerated on every driver restart.
   */
  public String clientWrapKey() {
    return Base64.getEncoder().encodeToString(wrapKeys().getPublic().getEncoded());
  }

  /**
   * Decrypts a client-wrapped CAB token: base64 of one or more RSA-OAEP-SHA256 blocks, each holding
   * a {@code [block index, block count]} header followed by a UTF-8 slice of the token.
   *
   * @throws IOException if the ciphertext is malformed or was not produced for this JVM's key;
   *     messages never include any part of the input
   */
  String unwrapClientToken(String wrapped) throws IOException {
    if (wrapped == null || wrapped.trim().isEmpty()) {
      throw new IOException("Client-supplied CAB token is empty");
    }
    byte[] ciphertext;
    try {
      ciphertext = Base64.getDecoder().decode(wrapped.trim());
    } catch (IllegalArgumentException e) {
      throw new IOException("Client-wrapped CAB token is not valid base64");
    }
    int blocks = ciphertext.length / WRAP_BLOCK_BYTES;
    if (blocks == 0 || blocks > WRAP_MAX_BLOCKS || ciphertext.length % WRAP_BLOCK_BYTES != 0) {
      throw new IOException("Client-wrapped CAB token has an invalid length");
    }
    ByteArrayOutputStream plain = new ByteArrayOutputStream();
    try {
      Cipher cipher = Cipher.getInstance(WRAP_CIPHER);
      cipher.init(Cipher.DECRYPT_MODE, wrapKeys().getPrivate(), oaepSha256());
      for (int i = 0; i < blocks; i++) {
        byte[] block = cipher.doFinal(ciphertext, i * WRAP_BLOCK_BYTES, WRAP_BLOCK_BYTES);
        if (block.length < WRAP_HEADER_BYTES
            || (block[0] & 0xff) != i
            || (block[1] & 0xff) != blocks) {
          throw new IOException("Client-wrapped CAB token blocks are incomplete or out of order");
        }
        plain.write(block, WRAP_HEADER_BYTES, block.length - WRAP_HEADER_BYTES);
      }
    } catch (GeneralSecurityException e) {
      throw new IOException(
          "Unable to unwrap the client-supplied CAB token (" + e.getClass().getSimpleName() + ")");
    }
    String token = new String(plain.toByteArray(), StandardCharsets.UTF_8);
    if (token.isEmpty()) {
      throw new IOException("Client-supplied CAB token is empty");
    }
    return token;
  }

  /** OAEP with SHA-256 for both the digest and MGF1 (the JCE default MGF1 digest is SHA-1). */
  static OAEPParameterSpec oaepSha256() {
    return new OAEPParameterSpec(
        "SHA-256", "MGF1", MGF1ParameterSpec.SHA256, PSource.PSpecified.DEFAULT);
  }

  private KeyPair wrapKeys() {
    KeyPair current = wrapKeys;
    if (current != null) {
      return current;
    }
    synchronized (wrapKeyLock) {
      if (wrapKeys == null) {
        try {
          KeyPairGenerator generator = KeyPairGenerator.getInstance(WRAP_KEY_ALGORITHM);
          generator.initialize(WRAP_KEY_BITS);
          wrapKeys = generator.generateKeyPair();
        } catch (GeneralSecurityException e) {
          throw new IllegalStateException("RSA key generation is unavailable", e);
        }
      }
      return wrapKeys;
    }
  }

  /**
   * Registers (or refreshes) a scope on the driver, mints a token when needed, publishes it to UGI
   * credentials, and broadcasts it to executors.
   *
   * @return non-secret result attributes
   */
  public Map<String, String> ensure(
      MlflowCabScope scope,
      Supplier supplier,
      String clientToken,
      long clientExpiresAtMs,
      Configuration conf,
      Long refreshMarginSecondsOverride)
      throws IOException {
    if (!driverRole.getAsBoolean()) {
      throw new IOException("CAB scopes can only be registered on the Spark driver");
    }
    ScopeState state = scopes.computeIfAbsent(scope.scopeId(), id -> new ScopeState(scope));
    boolean minted = false;
    synchronized (state.lock) {
      state.active = true;
      state.supplier = supplier;
      state.conf = conf == null ? new Configuration(false) : new Configuration(conf);
      state.refreshMarginMs = refreshMarginMs(conf, refreshMarginSecondsOverride);
      if (supplier == Supplier.CLIENT) {
        if (clientToken == null || clientToken.isEmpty()) {
          throw new IOException("Client-supplied CAB token is empty for scope " + scope.scopeId());
        }
        MlflowCabToken current = state.token;
        if (current == null
            || !clientToken.equals(current.secret())
            || current.expiresAtMs() != clientExpiresAtMs) {
          state.generation++;
          state.token =
              new MlflowCabToken(scope.scopeId(), clientToken, clientExpiresAtMs, state.generation);
          state.clientPushes.incrementAndGet();
        }
      } else if (needsRenewal(state)) {
        mintLocked(state);
        minted = true;
      }
      if (state.publishedGeneration != state.generation) {
        publishLocked(state);
      }
    }
    if (supplier == Supplier.CLUSTER) {
      startRenewer();
    }
    Map<String, String> result = describe(state);
    result.put("minted", Boolean.toString(minted));
    return result;
  }

  /** Marks a scope inactive so the driver stops renewing it. */
  public boolean release(String scopeId) {
    ScopeState state = scopes.get(scopeId);
    if (state == null) {
      return false;
    }
    state.active = false;
    return true;
  }

  /** Returns a usable token for {@code scopeId} on the driver or an executor. */
  public MlflowCabToken tokenFor(String scopeId, Configuration conf) throws IOException {
    if (!MlflowCabScope.isValidScopeId(scopeId)) {
      throw new IOException(
          "mlflow-gs:// paths must carry a CAB scope id (mlflow-gs://<scope>@<bucket>/...)");
    }
    long now = clock.getAsLong();
    if (driverRole.getAsBoolean()) {
      ScopeState state = scopes.get(scopeId);
      if (state != null) {
        if (state.supplier == Supplier.CLUSTER && needsRenewal(state)) {
          synchronized (state.lock) {
            if (needsRenewal(state)) {
              mintLocked(state);
              publishLocked(state);
            }
          }
        }
        MlflowCabToken token = state.token;
        if (token != null && token.isUsable(now, 0L)) {
          return token;
        }
      }
      MlflowCabToken fromUgi = readFromUgi(scopeId);
      if (fromUgi != null && fromUgi.isUsable(now, 0L)) {
        return fromUgi;
      }
      throw new IOException(
          "CAB scope "
              + scopeId
              + " has no valid token on the driver; register it before reading or writing "
              + "mlflow-gs:// paths"
              + (state != null && !state.lastError.isEmpty() ? " (last error: " + state.lastError + ")" : ""));
    }
    long waitMs =
        conf == null
            ? MlflowCabConstants.DEFAULT_EXECUTOR_TOKEN_WAIT_MS
            : conf.getLong(
                MlflowCabConstants.CONF_EXECUTOR_TOKEN_WAIT_MS,
                MlflowCabConstants.DEFAULT_EXECUTOR_TOKEN_WAIT_MS);
    long deadline = now + Math.max(0L, waitMs);
    while (true) {
      executorUgiReads.incrementAndGet();
      MlflowCabToken token = readFromUgi(scopeId);
      long current = clock.getAsLong();
      if (token != null && token.isUsable(current, EXECUTOR_MIN_VALIDITY_MS)) {
        return token;
      }
      if (current >= deadline) {
        throw new IOException(
            "No valid CAB token for scope "
                + scopeId
                + " has been broadcast to this executor by the driver; executors never mint "
                + "tokens themselves");
      }
      try {
        Thread.sleep(50L);
      } catch (InterruptedException e) {
        Thread.currentThread().interrupt();
        throw new IOException("Interrupted while waiting for a broadcast CAB token", e);
      }
    }
  }

  /** Forces a driver-side re-mint (rate limited); a no-op on executors. */
  public void forceRefresh(String scopeId) throws IOException {
    if (!driverRole.getAsBoolean()) {
      return;
    }
    ScopeState state = scopes.get(scopeId);
    if (state == null || state.supplier != Supplier.CLUSTER) {
      return;
    }
    synchronized (state.lock) {
      if (clock.getAsLong() - state.lastMintMs >= 5_000L) {
        mintLocked(state);
        publishLocked(state);
      }
    }
  }

  /** Returns non-secret telemetry for one scope (or the whole service if scopeId is null). */
  public Map<String, String> stats(String scopeId) {
    Map<String, String> result = new LinkedHashMap<>();
    result.put("driverCabRpcs", Long.toString(driverRpcs.get()));
    result.put("executorCabRpcs", Long.toString(executorRpcs.get()));
    result.put("broadcasts", Long.toString(broadcastCount.get()));
    result.put("executorUgiReads", Long.toString(executorUgiReads.get()));
    result.put("scopes", Integer.toString(scopes.size()));
    if (scopeId != null) {
      ScopeState state = scopes.get(scopeId);
      if (state != null) {
        result.putAll(describe(state));
      }
    }
    return result;
  }

  private Map<String, String> describe(ScopeState state) {
    Map<String, String> result = new LinkedHashMap<>();
    MlflowCabToken token = state.token;
    result.put("scopeId", state.scope.scopeId());
    result.put("mode", state.scope.mode());
    result.put("supplier", state.supplier.name());
    result.put("active", Boolean.toString(state.active));
    result.put("generation", Long.toString(state.generation));
    result.put("expiresAtMs", token == null ? "0" : Long.toString(token.expiresAtMs()));
    result.put("scopeMints", Long.toString(state.mints.get()));
    result.put("scopeRenewals", Long.toString(state.renewals.get()));
    result.put("scopePublishes", Long.toString(state.publishes.get()));
    result.put("scopeClientPushes", Long.toString(state.clientPushes.get()));
    result.put("refreshMarginMs", Long.toString(state.refreshMarginMs));
    result.put("executorsNotified", Integer.toString(state.lastExecutorsNotified));
    result.put("artifactUri", state.artifactUri);
    result.put("lastError", state.lastError);
    return result;
  }

  private boolean needsRenewal(ScopeState state) {
    MlflowCabToken token = state.token;
    long now = clock.getAsLong();
    if (token == null || token.secret().isEmpty()) {
      return true;
    }
    if (now < token.expiresAtMs() - state.refreshMarginMs) {
      return false;
    }
    // Inside the renewal window: renew, but never more than once per spacing interval unless the
    // current token is already unusable.
    return now - state.lastMintMs >= MIN_MINT_SPACING_MS || !token.isUsable(now, 0L);
  }

  private void mintLocked(ScopeState state) throws IOException {
    MlflowCabCallerCredentials credentials = callerFor(state.conf);
    try {
      if (!driverRole.getAsBoolean()) {
        executorRpcs.incrementAndGet();
        throw new IOException("Refusing to mint a CAB token outside the driver");
      }
      MlflowCabIssuer.Issued issued = issuer.issue(state.scope, credentials, state.conf, "driver");
      driverRpcs.incrementAndGet();
      boolean renewal = state.token != null;
      state.generation++;
      state.token =
          new MlflowCabToken(state.scope.scopeId(), issued.token, issued.expiresAtMs, state.generation);
      if (issued.artifactUri != null && !issued.artifactUri.isEmpty()) {
        state.artifactUri = issued.artifactUri;
      }
      state.lastMintMs = clock.getAsLong();
      state.mints.incrementAndGet();
      if (renewal) {
        state.renewals.incrementAndGet();
      }
      state.lastError = "";
    } catch (IOException e) {
      state.lastError = MlflowCabRedaction.describe(e);
      throw new IOException(MlflowCabRedaction.redact(e.getMessage()), e.getCause());
    }
  }

  private void publishLocked(ScopeState state) throws IOException {
    MlflowCabToken token = state.token;
    if (token == null) {
      return;
    }
    Credentials credentials = new Credentials();
    credentials.addSecretKey(aliasFor(state.scope.scopeId()), token.encode());
    UserGroupInformation.getCurrentUser().addCredentials(credentials);
    int notified;
    try {
      notified = broadcaster.broadcast();
    } catch (IOException | RuntimeException e) {
      state.lastError = MlflowCabRedaction.describe(e);
      notified = -1;
    }
    if (notified >= 0) {
      broadcastCount.incrementAndGet();
    }
    state.lastExecutorsNotified = notified;
    state.publishes.incrementAndGet();
    state.publishedGeneration = token.generation();
  }

  /** Reads a scope token from the current UGI credentials (executor-safe, no network). */
  static MlflowCabToken readFromUgi(String scopeId) throws IOException {
    Credentials credentials = UserGroupInformation.getCurrentUser().getCredentials();
    return MlflowCabToken.decode(scopeId, credentials.getSecretKey(aliasFor(scopeId)));
  }

  private MlflowCabCallerCredentials callerFor(Configuration conf) {
    synchronized (callerLock) {
      if (caller == null) {
        callerConf = conf == null ? new Configuration(false) : conf;
        caller =
            new MlflowCabCallerCredentials(
                callerConf, env, clock, MlflowCabIssuer.newHttpClient());
      }
      return caller;
    }
  }

  private synchronized void startRenewer() {
    if (renewer != null) {
      return;
    }
    renewer =
        Executors.newSingleThreadScheduledExecutor(
            runnable -> {
              Thread thread = new Thread(runnable, "mlflow-cab-driver-token-renewer");
              thread.setDaemon(true);
              return thread;
            });
    renewer.scheduleWithFixedDelay(
        this::renewTick, RENEWAL_TICK_MS, RENEWAL_TICK_MS, TimeUnit.MILLISECONDS);
  }

  void renewTick() {
    for (ScopeState state : scopes.values()) {
      if (!state.active || state.supplier != Supplier.CLUSTER) {
        continue;
      }
      if (!needsRenewal(state)) {
        continue;
      }
      synchronized (state.lock) {
        if (!state.active || !needsRenewal(state)) {
          continue;
        }
        try {
          mintLocked(state);
          publishLocked(state);
        } catch (IOException | RuntimeException e) {
          LOG.log(
              Level.WARNING,
              "MLflow CAB token renewal failed for scope {0}: {1}",
              new Object[] {state.scope.scopeId(), MlflowCabRedaction.describe(e)});
        }
      }
    }
  }

  synchronized void shutdown() {
    if (renewer != null) {
      renewer.shutdownNow();
      renewer = null;
    }
  }
}
