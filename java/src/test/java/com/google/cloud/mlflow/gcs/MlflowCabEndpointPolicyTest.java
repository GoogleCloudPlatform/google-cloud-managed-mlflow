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
import java.util.Map;
import org.apache.hadoop.conf.Configuration;
import org.junit.jupiter.api.Test;

class MlflowCabEndpointPolicyTest {

  private static final Map<String, String> NO_ENV = Map.of();

  @Test
  void allowsGoogleApisOnPort443() throws Exception {
    for (String url :
        new String[] {
          "https://mlflow.googleapis.com/v1alpha/projects/p/locations/l/namespaces/n:issueReadArtifactToken",
          "https://MLFLOW.GOOGLEAPIS.COM./v1alpha/x",
          "https://autopush-mlflow.sandbox.googleapis.com/v1alpha/x",
          "https://mlflow.googleapis.com:443/x",
        }) {
      MlflowCabEndpointPolicy.validate(url, null, MlflowCabTestSupport.env(NO_ENV));
    }
  }

  @Test
  void rejectsUntrustedHostsPortsAndSchemes() {
    for (String url :
        new String[] {
          "https://mlflow.example.com/x",
          "https://evilgoogleapis.com/x",
          "https://googleapis.com/x",
          "https://mlflow.googleapis.com:8443/x",
          "https://mlflow.googleapis.com.evil.example.com/x",
          "https://evil.example.com@mlflow.googleapis.com/x",
          "https://bad_label.googleapis.com/x",
          "http://mlflow.googleapis.com/x",
          "ftp://mlflow.googleapis.com/x",
          "http://127.0.0.1:8080/x",
          "https://mlflow.googleapis.com:0/x",
          " https://mlflow.googleapis.com/x",
          "https://mlflow.googleapis.com/x\n",
          "",
        }) {
      assertThrows(
          IOException.class,
          () -> MlflowCabEndpointPolicy.validate(url, null, MlflowCabTestSupport.env(NO_ENV)),
          url);
    }
  }

  @Test
  void overrideOptsIntoExactOriginOnly() throws Exception {
    Map<String, String> env = Map.of(MlflowCabConstants.ENV_ENDPOINT_OVERRIDE, "http://127.0.0.1:5000");
    MlflowCabEndpointPolicy.validate(
        "http://127.0.0.1:5000/v1alpha/x", null, MlflowCabTestSupport.env(env));
    for (String bad :
        new String[] {"http://127.0.0.1:5001/x", "https://127.0.0.1:5000/x", "http://localhost:5000/x"}) {
      assertThrows(
          IOException.class,
          () -> MlflowCabEndpointPolicy.validate(bad, null, MlflowCabTestSupport.env(env)),
          bad);
    }
    Configuration conf = new Configuration(false);
    conf.set(MlflowCabConstants.CONF_ENDPOINT_OVERRIDE, "https://gateway.example.com:8443");
    MlflowCabEndpointPolicy.validate(
        "https://gateway.example.com:8443/v1alpha/x", conf, MlflowCabTestSupport.env(NO_ENV));
    assertThrows(
        IOException.class,
        () ->
            MlflowCabEndpointPolicy.validate(
                "https://gateway.example.com/v1alpha/x", conf, MlflowCabTestSupport.env(NO_ENV)));
    // Non-loopback HTTP overrides fail closed.
    assertFalse(
        MlflowCabEndpointPolicy.isAllowed(
            MlflowCabEndpointPolicy.parse("https://gateway.example.com/x"),
            "http://gateway.example.com"));
  }

  @Test
  void normalizesOrigins() {
    MlflowCabEndpointPolicy.Origin origin = MlflowCabEndpointPolicy.parse("https://[::1]:8443/x");
    assertEquals("::1", origin.host());
    assertEquals(8443, origin.port());
    assertTrue(MlflowCabEndpointPolicy.isAllowed(origin, "https://[0:0:0:0:0:0:0:1]:8443"));
    assertEquals(443, MlflowCabEndpointPolicy.parse("https://mlflow.googleapis.com").port());
    assertEquals(80, MlflowCabEndpointPolicy.parse("http://localhost").port());
  }
}
