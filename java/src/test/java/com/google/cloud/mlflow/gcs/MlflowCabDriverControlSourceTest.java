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
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.nio.charset.StandardCharsets;
import java.time.Instant;
import java.util.Base64;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.zip.ZipEntry;
import java.util.zip.ZipInputStream;
import java.util.zip.ZipOutputStream;
import org.apache.spark.sql.connector.catalog.SupportsRead;
import org.apache.spark.sql.connector.catalog.Table;
import org.apache.spark.sql.connector.catalog.TableCapability;
import org.apache.spark.sql.connector.read.Scan;
import org.apache.spark.sql.types.StructType;
import org.apache.spark.sql.util.CaseInsensitiveStringMap;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Test;

class MlflowCabDriverControlSourceTest {

  @AfterEach
  void tearDown() {
    MlflowCabTokenService.setInstanceForTesting(null);
  }

  private static Map<String, Object> run(Map<String, String> options) {
    StructType schema =
        new MlflowCabDriverControlSource().inferSchema(new CaseInsensitiveStringMap(options));
    assertEquals(1, schema.fields().length);
    String json = schema.fields()[0].metadata().getString(MlflowCabDriverControlSource.RESULT_METADATA_KEY);
    assertFalse(json.contains("ya29."), json);
    return MlflowCabJson.parseObject(json);
  }

  @Test
  void probeAndUnknownActionsReturnStructuredResults() {
    Map<String, Object> probe = run(Map.of("action", "probe"));
    assertEquals("true", probe.get("ok"));
    assertEquals(MlflowCabConstants.VERSION, probe.get("version"));
    Map<String, Object> unknown = run(Map.of("action", "nope"));
    assertEquals("false", unknown.get("ok"));
    assertTrue(((String) unknown.get("error")).contains("Unknown"));
  }

  @Test
  void ensureNeverEchoesClientTokens() throws Exception {
    Map<String, Object> probe = run(Map.of("action", "probe"));
    String wrapKey = (String) probe.get("clientWrapKey");
    assertTrue(wrapKey != null && !wrapKey.isEmpty(), "probe must publish clientWrapKey");
    String token = "ya29." + "client-token-must-not-leak";
    Map<String, String> options = new HashMap<>();
    options.put("action", "ensure");
    options.put("issueUrl", "https://mlflow.googleapis.com/v1alpha/projects/p/locations/l/namespaces/n:issueReadArtifactToken");
    options.put("issueBody", "{\"runId\":\"r\"}");
    options.put("mode", "READ_ONLY");
    options.put("supplier", "client");
    options.put("tokenSecret", MlflowCabTestSupport.wrapForDriver(wrapKey, token, true));
    options.put("expireTimeMs", Long.toString(Instant.now().plusSeconds(600).toEpochMilli()));
    Map<String, Object> result = run(options);
    assertEquals("true", result.get("ok"), String.valueOf(result.get("error")));
    assertEquals("CLIENT", result.get("supplier"));
    String scopeId = (String) result.get("scopeId");
    assertEquals(token, MlflowCabTokenService.get().tokenFor(scopeId, null).secret());

    Map<String, String> plaintext = new HashMap<>(options);
    plaintext.remove("tokenSecret");
    plaintext.put("token", token);
    Map<String, Object> rejected = run(plaintext);
    assertEquals("false", rejected.get("ok"));
    assertTrue(((String) rejected.get("error")).contains("Plaintext client tokens are rejected"));

    Map<String, String> unwrapped = new HashMap<>(options);
    unwrapped.put("tokenSecret", token);
    Map<String, Object> rejectedSecret = run(unwrapped);
    assertEquals("false", rejectedSecret.get("ok"));
    assertTrue(((String) rejectedSecret.get("error")).contains("must be wrapped"));

    options.put("scopeId", "000000000000000000000000");
    Map<String, Object> mismatch = run(options);
    assertEquals("false", mismatch.get("ok"));
  }

  @Test
  void tablePlansZeroPartitions() {
    StructType schema =
        new MlflowCabDriverControlSource()
            .inferSchema(new CaseInsensitiveStringMap(Map.of("action", "probe")));
    Table table = new MlflowCabDriverControlSource().getTable(schema, new org.apache.spark.sql.connector.expressions.Transform[0], Map.of());
    assertTrue(table.capabilities().contains(TableCapability.BATCH_READ));
    Scan scan =
        ((SupportsRead) table).newScanBuilder(new CaseInsensitiveStringMap(Map.of())).build();
    assertEquals(0, scan.toBatch().planInputPartitions().length);
    assertEquals("mlflow-cab-control", new MlflowCabDriverControlSource().shortName());
  }

  @Test
  void archiveListAndDeleteActionsRoundTripOverCabPaths() throws Exception {
    InMemoryGcsFileSystem.reset();
    System.setProperty(MlflowCabConstants.CONF_DELEGATE_IMPL, InMemoryGcsFileSystem.class.getName());
    try {
      String scope = "0123456789abcdef01234567";
      String root = "mlflow-gs://" + scope + "@bucket/models/m1";
      Map<String, String> files = new LinkedHashMap<>();
      files.put("MLmodel", "artifact_path: model\n");
      files.put("sub/conda.yaml", "name: mlflow-env\n");

      Map<String, Object> written =
          run(Map.of("action", "write_archive", "dst", root, "archive", zipBase64(files)));
      assertEquals("true", written.get("ok"), String.valueOf(written.get("error")));
      assertEquals("2", written.get("filesWritten"));

      Map<String, Object> listed = run(Map.of("action", "list_status", "src", root));
      assertEquals("true", listed.get("ok"), String.valueOf(listed.get("error")));
      String entriesJson = (String) listed.get("entries");
      assertTrue(entriesJson.contains("\"MLmodel\""), entriesJson);
      assertTrue(entriesJson.contains("\"sub\""), entriesJson);

      Map<String, Object> readDir = run(Map.of("action", "read_archive", "src", root));
      assertEquals("true", readDir.get("ok"), String.valueOf(readDir.get("error")));
      assertEquals("true", readDir.get("isDirectory"));
      assertEquals(files, unzipBase64((String) readDir.get("archive")));

      Map<String, Object> readFile =
          run(Map.of("action", "read_archive", "src", root + "/MLmodel"));
      assertEquals("true", readFile.get("ok"), String.valueOf(readFile.get("error")));
      assertEquals("false", readFile.get("isDirectory"));
      assertEquals(
          Map.of("MLmodel", "artifact_path: model\n"),
          unzipBase64((String) readFile.get("archive")));

      Map<String, Object> traversal =
          run(
              Map.of(
                  "action",
                  "write_archive",
                  "dst",
                  root,
                  "archive",
                  zipBase64(Map.of("../escape", "no"))));
      assertEquals("false", traversal.get("ok"));

      Map<String, Object> nonCab =
          run(Map.of("action", "delete", "target", "gs://bucket/models/m1"));
      assertEquals("false", nonCab.get("ok"));

      Map<String, Object> deleted = run(Map.of("action", "delete", "target", root));
      assertEquals("true", deleted.get("ok"), String.valueOf(deleted.get("error")));
      assertEquals("true", deleted.get("deleted"));

      Map<String, Object> afterDelete = run(Map.of("action", "list_status", "src", root));
      assertEquals("true", afterDelete.get("ok"));
      assertEquals("[]", afterDelete.get("entries"));
    } finally {
      System.clearProperty(MlflowCabConstants.CONF_DELEGATE_IMPL);
    }
  }

  private static String zipBase64(Map<String, String> files) throws Exception {
    ByteArrayOutputStream baos = new ByteArrayOutputStream();
    try (ZipOutputStream zip = new ZipOutputStream(baos, StandardCharsets.UTF_8)) {
      for (Map.Entry<String, String> entry : files.entrySet()) {
        zip.putNextEntry(new ZipEntry(entry.getKey()));
        zip.write(entry.getValue().getBytes(StandardCharsets.UTF_8));
        zip.closeEntry();
      }
    }
    return Base64.getEncoder().encodeToString(baos.toByteArray());
  }

  private static Map<String, String> unzipBase64(String b64) throws Exception {
    Map<String, String> result = new LinkedHashMap<>();
    byte[] raw = Base64.getDecoder().decode(b64);
    try (ZipInputStream zip =
        new ZipInputStream(new ByteArrayInputStream(raw), StandardCharsets.UTF_8)) {
      ZipEntry entry;
      while ((entry = zip.getNextEntry()) != null) {
        if (!entry.isDirectory()) {
          result.put(entry.getName(), new String(zip.readAllBytes(), StandardCharsets.UTF_8));
        }
        zip.closeEntry();
      }
    }
    return result;
  }
}
