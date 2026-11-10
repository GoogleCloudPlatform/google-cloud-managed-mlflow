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

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertInstanceOf;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.google.cloud.hadoop.fs.gcs.GoogleHadoopFileSystem;
import java.io.InputStream;
import java.net.URLDecoder;
import java.nio.charset.StandardCharsets;
import java.time.Instant;
import java.util.Map;
import java.util.TreeMap;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileStatus;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

/**
 * Wires {@link MlflowCabFileSystem} to the real Cloud Storage connector on the test classpath
 * (3.1.x or 4.0.x, depending on the build matrix leg) and a loopback JSON-API stub that only
 * accepts the downscoped CAB token, proving the provider is linked into the connector's auth path.
 */
class MlflowCabConnectorLinkageTest {

  private static final String CAB_TOKEN = "ya29." + "linkage-cab-token";
  private static final byte[] CONTENT = "hello-cab".getBytes(StandardCharsets.UTF_8);

  @BeforeEach
  void setUp() throws Exception {
    FileSystem.closeAll();
  }

  @AfterEach
  void tearDown() throws Exception {
    FileSystem.closeAll();
    MlflowCabTokenService.setInstanceForTesting(null);
  }

  private static String objectJson(String bucket, String name, long size) {
    return "{\"kind\":\"storage#object\",\"bucket\":\""
        + bucket
        + "\",\"name\":\""
        + name
        + "\",\"size\":\""
        + size
        + "\",\"generation\":\"1700000000000001\",\"metageneration\":\"1\","
        + "\"contentType\":\"application/octet-stream\",\"updated\":\"2026-11-07T00:00:00.000Z\","
        + "\"timeCreated\":\"2026-11-07T00:00:00.000Z\",\"storageClass\":\"STANDARD\"}";
  }

  @Test
  void connectorAuthenticatesWithDriverIssuedCabToken() throws Exception {
    AtomicInteger unauthorized = new AtomicInteger();
    String objectPath = "/storage/v1/b/managed-bucket/o/runs%2Fr1%2Fartifacts%2Fmodel%2Fdata.bin";
    try (MlflowCabTestSupport.FakeServer gcs =
        new MlflowCabTestSupport.FakeServer(
            r -> {
              if (!("Bearer " + CAB_TOKEN).equals(r.header("Authorization"))) {
                unauthorized.incrementAndGet();
                return new MlflowCabTestSupport.Reply(401, "{\"error\":{\"code\":401}}");
              }
              String path = r.path.replace("/download/storage/v1/", "/storage/v1/");
              boolean media = r.query != null && r.query.contains("alt=media");
              if (path.equals(objectPath)) {
                if (media) {
                  return new MlflowCabTestSupport.Reply(
                      200, CONTENT, Map.of("Content-Type", "application/octet-stream"));
                }
                return new MlflowCabTestSupport.Reply(
                    200, objectJson("managed-bucket", "runs/r1/artifacts/model/data.bin", CONTENT.length));
              }
              if (path.equals("/storage/v1/b/managed-bucket/o") && r.query != null) {
                String query = URLDecoder.decode(r.query, StandardCharsets.UTF_8);
                if (query.contains("prefix=runs/r1/artifacts/model/")) {
                  return new MlflowCabTestSupport.Reply(
                      200,
                      "{\"kind\":\"storage#objects\",\"items\":["
                          + objectJson("managed-bucket", "runs/r1/artifacts/model/data.bin", CONTENT.length)
                          + "]}");
                }
                return new MlflowCabTestSupport.Reply(200, "{\"kind\":\"storage#objects\"}");
              }
              return new MlflowCabTestSupport.Reply(404, "{\"error\":{\"code\":404,\"message\":\"Not Found\"}}");
            })) {
      MlflowCabTokenService service =
          new MlflowCabTokenService(
              MlflowCabTestSupport.env(Map.of()),
              System::currentTimeMillis,
              null,
              () -> 0,
              new AtomicBoolean(true)::get,
              MlflowCabTestSupport.staticCaller("ya29." + "unused"));
      MlflowCabTokenService.setInstanceForTesting(service);
      MlflowCabScope scope =
          MlflowCabScope.of(
              "https://mlflow.googleapis.com/v1alpha/projects/p/locations/l/namespaces/ns:issueReadArtifactToken",
              "{\"runId\":\"r1\",\"artifactPath\":\"model\"}",
              "READ_ONLY",
              "gs://managed-bucket/runs/r1/artifacts/model/");
      service.ensure(
          scope,
          MlflowCabTokenService.Supplier.CLIENT,
          CAB_TOKEN,
          Instant.now().plusSeconds(3600).toEpochMilli(),
          new Configuration(false),
          null);

      Configuration customer = new Configuration();
      // Customer environment (for example a private Cloud Storage endpoint); inherited by the clone.
      customer.set("fs.gs.storage.root.url", gcs.baseUrl() + "/");
      customer.set("fs.gs.auth.type", "UNAUTHENTICATED");
      customer.set("fs.gs.max.requests.per.batch", "1");
      Map<String, String> before = new TreeMap<>();
      for (Map.Entry<String, String> entry : customer) {
        if (entry.getKey().startsWith("fs.gs.")) {
          before.put(entry.getKey(), entry.getValue());
        }
      }

      Path file =
          new Path("mlflow-gs://" + scope.scopeId() + "@managed-bucket/runs/r1/artifacts/model/data.bin");
      MlflowCabFileSystem fs = (MlflowCabFileSystem) file.getFileSystem(customer);
      FileStatus status = fs.getFileStatus(file);
      assertEquals(file, status.getPath());
      assertEquals(CONTENT.length, status.getLen());
      try (InputStream in = fs.open(file)) {
        assertArrayEquals(CONTENT, MlflowCabTestSupport.readAll(in));
      }
      FileStatus[] listed = fs.listStatus(file.getParent());
      assertEquals(1, listed.length);
      assertEquals(file, listed[0].getPath());
      assertInstanceOf(GoogleHadoopFileSystem.class, fs.delegate());
      assertEquals(0, unauthorized.get(), "every connector request must carry the CAB token");
      assertTrue(gcs.count.get() >= 2);

      Map<String, String> after = new TreeMap<>();
      for (Map.Entry<String, String> entry : customer) {
        if (entry.getKey().startsWith("fs.gs.")) {
          after.put(entry.getKey(), entry.getValue());
        }
      }
      assertEquals(before, after, "customer fs.gs.* keys must not be mutated");
      assertEquals("UNAUTHENTICATED", customer.get("fs.gs.auth.type"));
      assertFalse(fs.innerConfiguration().get("fs.gs.auth.type").equals("UNAUTHENTICATED"));
    }
  }
}
