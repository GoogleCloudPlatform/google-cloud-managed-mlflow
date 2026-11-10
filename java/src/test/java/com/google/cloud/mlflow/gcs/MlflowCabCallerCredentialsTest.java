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
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.io.IOException;
import java.net.URLDecoder;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.security.KeyPair;
import java.security.KeyPairGenerator;
import java.security.Signature;
import java.util.Base64;
import java.util.HashMap;
import java.util.Map;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

class MlflowCabCallerCredentialsTest {

  @TempDir Path tmp;

  private static String tokenJson(String token) {
    return "{\"access_token\":\"" + token + "\",\"expires_in\":3599,\"token_type\":\"Bearer\"}";
  }

  @Test
  void metadataServerTokenIsCached() throws Exception {
    try (MlflowCabTestSupport.FakeServer metadata =
        new MlflowCabTestSupport.FakeServer(
            r -> {
              assertEquals("Google", r.header("Metadata-Flavor"));
              assertEquals(
                  "/computeMetadata/v1/instance/service-accounts/default/token", r.path);
              return new MlflowCabTestSupport.Reply(200, tokenJson("ya29." + "metadata-token"));
            })) {
      Map<String, String> env = new HashMap<>();
      env.put("GCE_METADATA_HOST", metadata.baseUrl().substring("http://".length()));
      env.put("CLOUDSDK_CONFIG", tmp.resolve("empty-gcloud").toString());
      MlflowCabCallerCredentials caller =
          new MlflowCabCallerCredentials(
              null, MlflowCabTestSupport.env(env), System::currentTimeMillis, MlflowCabIssuer.newHttpClient());
      assertEquals("ya29." + "metadata-token", caller.token());
      assertEquals("ya29." + "metadata-token", caller.token());
      assertEquals(1, metadata.count.get());
      assertEquals("metadata_server", caller.lastSourceKind());
      caller.invalidate();
      caller.token();
      assertEquals(2, metadata.count.get());
    }
  }

  @Test
  void authorizedUserRefreshTokenGrant() throws Exception {
    try (MlflowCabTestSupport.FakeServer oauth =
        new MlflowCabTestSupport.FakeServer(
            r -> {
              assertTrue(r.body.contains("grant_type=refresh_token"));
              assertTrue(r.body.contains("refresh_token=rt-1"));
              return new MlflowCabTestSupport.Reply(200, tokenJson("ya29." + "user-token"));
            })) {
      Path file = tmp.resolve("adc.json");
      Files.writeString(
          file,
          "{\"type\":\"authorized_user\",\"client_id\":\"cid\",\"client_secret\":\"cs\","
              + "\"refresh_token\":\"rt-1\",\"token_uri\":\""
              + oauth.baseUrl()
              + "/token\"}");
      Map<String, String> env = new HashMap<>();
      env.put("GOOGLE_APPLICATION_CREDENTIALS", file.toString());
      env.put(MlflowCabConstants.ENV_ENDPOINT_OVERRIDE, oauth.baseUrl());
      MlflowCabCallerCredentials caller =
          new MlflowCabCallerCredentials(
              null, MlflowCabTestSupport.env(env), System::currentTimeMillis, MlflowCabIssuer.newHttpClient());
      assertEquals("ya29." + "user-token", caller.token());
      assertTrue(caller.lastSourceKind().contains("authorized_user"));

      Map<String, String> noOverride = new HashMap<>();
      noOverride.put("GOOGLE_APPLICATION_CREDENTIALS", file.toString());
      MlflowCabCallerCredentials untrusted =
          new MlflowCabCallerCredentials(
              null, MlflowCabTestSupport.env(noOverride), System::currentTimeMillis,
              MlflowCabIssuer.newHttpClient());
      assertThrows(IOException.class, untrusted::token);
      assertEquals(1, oauth.count.get());
    }
  }

  @Test
  void serviceAccountJwtAssertionIsSignedWithTheKey() throws Exception {
    KeyPairGenerator generator = KeyPairGenerator.getInstance("RSA");
    generator.initialize(2048);
    KeyPair pair = generator.generateKeyPair();
    String pem =
        "-----BEGIN PRIVATE KEY-----\\n"
            + Base64.getEncoder().encodeToString(pair.getPrivate().getEncoded())
            + "\\n-----END PRIVATE KEY-----\\n";
    try (MlflowCabTestSupport.FakeServer oauth =
        new MlflowCabTestSupport.FakeServer(
            r -> {
              String assertion = null;
              for (String part : r.body.split("&")) {
                if (part.startsWith("assertion=")) {
                  assertion = URLDecoder.decode(part.substring(10), StandardCharsets.UTF_8);
                }
              }
              try {
                String[] jwt = assertion.split("\\.");
                Signature verifier = Signature.getInstance("SHA256withRSA");
                verifier.initVerify(pair.getPublic());
                verifier.update((jwt[0] + "." + jwt[1]).getBytes(StandardCharsets.US_ASCII));
                boolean ok = verifier.verify(Base64.getUrlDecoder().decode(jwt[2]));
                String claims = new String(Base64.getUrlDecoder().decode(jwt[1]), StandardCharsets.UTF_8);
                if (!ok || !claims.contains("sa@example.iam.gserviceaccount.com")) {
                  return new MlflowCabTestSupport.Reply(401, "{}");
                }
              } catch (Exception e) {
                return new MlflowCabTestSupport.Reply(400, "{}");
              }
              return new MlflowCabTestSupport.Reply(200, tokenJson("ya29." + "sa-token"));
            })) {
      Path file = tmp.resolve("sa.json");
      Files.writeString(
          file,
          "{\"type\":\"service_account\",\"client_email\":\"sa@example.iam.gserviceaccount.com\","
              + "\"private_key_id\":\"k1\",\"private_key\":\""
              + pem
              + "\",\"token_uri\":\""
              + oauth.baseUrl()
              + "/token\"}");
      Map<String, String> env = new HashMap<>();
      env.put("GOOGLE_APPLICATION_CREDENTIALS", file.toString());
      env.put(MlflowCabConstants.ENV_ENDPOINT_OVERRIDE, oauth.baseUrl());
      MlflowCabCallerCredentials caller =
          new MlflowCabCallerCredentials(
              null, MlflowCabTestSupport.env(env), System::currentTimeMillis, MlflowCabIssuer.newHttpClient());
      assertEquals("ya29." + "sa-token", caller.token());
    }
  }
}
