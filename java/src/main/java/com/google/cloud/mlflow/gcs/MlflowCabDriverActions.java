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

import java.io.BufferedReader;
import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.nio.charset.StandardCharsets;
import java.util.Arrays;
import java.util.Base64;
import java.util.Comparator;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.function.UnaryOperator;
import java.util.zip.ZipEntry;
import java.util.zip.ZipInputStream;
import java.util.zip.ZipOutputStream;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileStatus;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.LocatedFileStatus;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.fs.RemoteIterator;
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.Transformer;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.types.DataTypes;

/**
 * Driver-side actions triggered through {@link MlflowCabDriverControlSource}. Results only ever
 * contain non-secret attributes; failures are reported as redacted messages.
 */
final class MlflowCabDriverActions {

  /** Prefix of Google OAuth 2.0 access tokens; such values must never arrive in plaintext. */
  private static final String BEARER_TOKEN_PREFIX = "ya29.";

  /** Class recorded by PySpark's pipeline writer (for example for pipelines saved over Connect). */
  static final String PYTHON_PIPELINE_CLASS = "pyspark.ml.pipeline.PipelineModel";

  private MlflowCabDriverActions() {}

  static Map<String, String> execute(UnaryOperator<String> options) {
    Map<String, String> result = new LinkedHashMap<>();
    String action = lower(options.apply("action"), "probe");
    result.put("action", action);
    try {
      switch (action) {
        case "probe":
          probe(result);
          break;
        case "install":
          result.putAll(MlflowCabClassLoaders.installOnDriver());
          probe(result);
          break;
        case "ensure":
          ensure(options, result);
          break;
        case "release":
          result.put(
              "released",
              Boolean.toString(MlflowCabTokenService.get().release(required(options, "scopeId"))));
          break;
        case "stats":
          result.putAll(MlflowCabTokenService.get().stats(options.apply("scopeId")));
          result.put("remoteExecutors", Integer.toString(MlflowCabSparkBridge.remoteExecutorCount()));
          break;
        case "copy":
          copy(options, result);
          break;
        case "delete":
          delete(options, result);
          break;
        case "list_status":
          listStatus(options, result);
          break;
        case "read_archive":
          readArchive(options, result);
          break;
        case "write_archive":
          writeArchive(options, result);
          break;
        case "register_model_udf":
          registerModelUdf(options, result);
          break;
        default:
          throw new IllegalArgumentException("Unknown MLflow CAB driver action: " + action);
      }
      result.put("ok", "true");
    } catch (Exception | LinkageError e) {
      result.put("ok", "false");
      result.put("error", MlflowCabRedaction.describe(e));
    }
    return result;
  }

  private static void probe(Map<String, String> result) {
    result.put("version", MlflowCabConstants.VERSION);
    result.put("isDriver", Boolean.toString(MlflowCabSparkBridge.isSparkDriver()));
    result.put("pluginDriverInitialized", Boolean.toString(MlflowCabSparkPlugin.driverInitialized));
    result.put("providerLoader", MlflowCabClassLoaders.providerLoader().getClass().getName());
    result.put(
        "delegationBindingVisible",
        Boolean.toString(MlflowCabInnerConfiguration.bindingVisibleToConnector()));
    result.put("remoteExecutors", Integer.toString(MlflowCabSparkBridge.remoteExecutorCount()));
    // Public half of this driver JVM's ephemeral key; clients wrap pushed tokens with it.
    result.put("clientWrapKey", MlflowCabTokenService.get().clientWrapKey());
  }

  private static void ensure(UnaryOperator<String> options, Map<String, String> result)
      throws IOException {
    String issueUrl = required(options, "issueUrl");
    String issueBody = options.apply("issueBody");
    MlflowCabScope scope =
        MlflowCabScope.of(issueUrl, issueBody, options.apply("mode"), options.apply("boundary"));
    String expected = options.apply("scopeId");
    if (expected != null && !expected.isEmpty() && !expected.equals(scope.scopeId())) {
      throw new IllegalArgumentException("scopeId does not match the issuance request");
    }
    String plaintext = options.apply("token");
    if (plaintext != null && !plaintext.isEmpty()) {
      throw new IllegalArgumentException(
          "Plaintext client tokens are rejected; wrap them with the clientWrapKey returned by the "
              + "install action and pass the ciphertext as tokenSecret");
    }
    MlflowCabTokenService.Supplier supplier =
        "client".equals(lower(options.apply("supplier"), "cluster"))
            ? MlflowCabTokenService.Supplier.CLIENT
            : MlflowCabTokenService.Supplier.CLUSTER;
    String clientToken = null;
    if (supplier == MlflowCabTokenService.Supplier.CLIENT) {
      String wrapped = options.apply("tokenSecret");
      if (wrapped != null && wrapped.trim().startsWith(BEARER_TOKEN_PREFIX)) {
        throw new IllegalArgumentException(
            "tokenSecret must be wrapped with the driver's clientWrapKey; plaintext bearer tokens "
                + "are rejected");
      }
      clientToken = MlflowCabTokenService.get().unwrapClientToken(wrapped);
    }
    Configuration conf = MlflowCabSparkBridge.activeSessionHadoopConf();
    if (conf == null) {
      conf = new Configuration();
    }
    String override = options.apply("endpointOverride");
    if (override != null && !override.trim().isEmpty()) {
      conf.set(MlflowCabConstants.CONF_ENDPOINT_OVERRIDE, override.trim());
    }
    Long margin = parseLong(options.apply("refreshMarginSeconds"));
    Long expiresAtMs = parseLong(options.apply("expireTimeMs"));
    result.putAll(
        MlflowCabTokenService.get()
            .ensure(
                scope,
                supplier,
                clientToken,
                expiresAtMs == null ? 0L : expiresAtMs,
                conf,
                margin));
  }

  private static void copy(UnaryOperator<String> options, Map<String, String> result)
      throws IOException {
    Configuration conf = MlflowCabSparkBridge.activeSessionHadoopConf();
    if (conf == null) {
      conf = new Configuration();
    }
    copy(new Path(required(options, "src")), new Path(required(options, "dst")), conf, result);
  }

  /**
   * Create-only copy of a file or directory tree. {@code mlflow-gs} filesystems are opened as
   * private (uncached) instances and closed afterwards; shared cached instances of other schemes
   * ({@code gs}, {@code hdfs}, {@code file}, ...) are never closed here.
   */
  static void copy(Path src, Path dst, Configuration conf, Map<String, String> result)
      throws IOException {
    MlflowCabClassLoaders.install(conf);
    long[] counters = new long[3]; // files copied, files skipped, bytes copied
    FileSystem srcFs = open(src, conf);
    IOException failure = null;
    try {
      FileSystem dstFs = open(dst, conf);
      try {
        copyTree(srcFs, src, dstFs, dst, counters);
      } catch (IOException e) {
        failure = e;
      } finally {
        failure = closePrivate(dstFs, dst, failure);
      }
    } finally {
      failure = closePrivate(srcFs, src, failure);
    }
    if (failure != null) {
      throw failure;
    }
    result.put("filesCopied", Long.toString(counters[0]));
    result.put("filesSkipped", Long.toString(counters[1]));
    result.put("bytesCopied", Long.toString(counters[2]));
  }

  private static void copyTree(
      FileSystem srcFs, Path src, FileSystem dstFs, Path dst, long[] counters) throws IOException {
    FileStatus srcStatus = srcFs.getFileStatus(src);
    if (!srcStatus.isDirectory()) {
      copyFile(srcFs, srcStatus, dstFs, dst, counters);
      return;
    }
    Path srcRoot = srcStatus.getPath();
    String rootPath = srcRoot.toUri().getPath();
    if (!rootPath.endsWith("/")) {
      rootPath = rootPath + "/";
    }
    RemoteIterator<LocatedFileStatus> it = srcFs.listFiles(srcRoot, true);
    while (it.hasNext()) {
      LocatedFileStatus status = it.next();
      String filePath = status.getPath().toUri().getPath();
      if (!filePath.startsWith(rootPath)) {
        continue;
      }
      copyFile(srcFs, status, dstFs, new Path(dst, filePath.substring(rootPath.length())), counters);
    }
  }

  private static void copyFile(
      FileSystem srcFs, FileStatus status, FileSystem dstFs, Path target, long[] counters)
      throws IOException {
    FileStatus existing;
    try {
      existing = dstFs.getFileStatus(target);
    } catch (java.io.FileNotFoundException e) {
      existing = null;
    }
    if (existing != null) {
      if (!existing.isDirectory() && existing.getLen() == status.getLen()) {
        counters[1]++;
        return;
      }
      throw new IOException("Refusing to overwrite existing object " + target);
    }
    byte[] buffer = new byte[1 << 16];
    try (InputStream in = srcFs.open(status.getPath());
        OutputStream out = dstFs.create(target, false)) {
      int read;
      while ((read = in.read(buffer)) >= 0) {
        out.write(buffer, 0, read);
      }
    }
    counters[0]++;
    counters[2] += status.getLen();
  }

  private static boolean isCabPath(Path path) {
    String scheme = path.toUri().getScheme();
    return scheme != null && MlflowCabConstants.SCHEME.equalsIgnoreCase(scheme);
  }

  private static FileSystem open(Path path, Configuration conf) throws IOException {
    return isCabPath(path) ? FileSystem.newInstance(path.toUri(), conf) : path.getFileSystem(conf);
  }

  private static IOException closePrivate(FileSystem fs, Path path, IOException failure) {
    if (!isCabPath(path)) {
      return failure;
    }
    try {
      fs.close();
    } catch (IOException e) {
      if (failure == null) {
        return e;
      }
      failure.addSuppressed(e);
    }
    return failure;
  }

  private static Path requireCabPath(UnaryOperator<String> options, String key) {
    Path path = new Path(required(options, key));
    if (!isCabPath(path)) {
      throw new IllegalArgumentException(
          "Option '" + key + "' must be an mlflow-gs:// URI: " + path);
    }
    return path;
  }

  private static Configuration activeConf() {
    Configuration conf = MlflowCabSparkBridge.activeSessionHadoopConf();
    if (conf == null) {
      conf = new Configuration();
    }
    String delegate = System.getProperty(MlflowCabConstants.CONF_DELEGATE_IMPL);
    if (delegate != null && !delegate.isEmpty()) {
      conf.set(MlflowCabConstants.CONF_DELEGATE_IMPL, delegate);
    }
    MlflowCabClassLoaders.install(conf);
    return conf;
  }

  private static void delete(UnaryOperator<String> options, Map<String, String> result)
      throws IOException {
    Path target = requireCabPath(options, "target");
    Configuration conf = activeConf();
    FileSystem fs = open(target, conf);
    IOException failure = null;
    boolean deleted = false;
    try {
      deleted = fs.delete(target, true);
    } catch (java.io.FileNotFoundException e) {
      deleted = false;
    } catch (IOException e) {
      failure = e;
    } finally {
      failure = closePrivate(fs, target, failure);
    }
    if (failure != null) {
      throw failure;
    }
    result.put("deleted", Boolean.toString(deleted));
  }

  private static void listStatus(UnaryOperator<String> options, Map<String, String> result)
      throws IOException {
    Path src = requireCabPath(options, "src");
    Configuration conf = activeConf();
    FileSystem fs = open(src, conf);
    IOException failure = null;
    FileStatus[] statuses = new FileStatus[0];
    try {
      FileStatus rootStatus = fs.getFileStatus(src);
      if (rootStatus.isDirectory()) {
        statuses = fs.listStatus(src);
      }
    } catch (java.io.FileNotFoundException e) {
      statuses = new FileStatus[0];
    } catch (IOException e) {
      failure = e;
    } finally {
      failure = closePrivate(fs, src, failure);
    }
    if (failure != null) {
      throw failure;
    }
    Arrays.sort(statuses, Comparator.comparing(s -> s.getPath().getName()));
    StringBuilder entries = new StringBuilder("[");
    boolean first = true;
    for (FileStatus status : statuses) {
      String name = status.getPath().getName();
      if (name.endsWith(".crc")) {
        continue;
      }
      if (!first) {
        entries.append(',');
      }
      first = false;
      entries
          .append("{\"name\":")
          .append(MlflowCabJson.quote(name))
          .append(",\"isDir\":")
          .append(status.isDirectory())
          .append(",\"size\":")
          .append(status.getLen())
          .append('}');
    }
    entries.append(']');
    result.put("entries", entries.toString());
  }

  private static void readArchive(UnaryOperator<String> options, Map<String, String> result)
      throws IOException {
    Path src = requireCabPath(options, "src");
    Configuration conf = activeConf();
    FileSystem fs = open(src, conf);
    IOException failure = null;
    ByteArrayOutputStream baos = new ByteArrayOutputStream();
    long files = 0;
    long bytes = 0;
    boolean isDirectory = false;
    byte[] buffer = new byte[1 << 16];
    try (ZipOutputStream zip = new ZipOutputStream(baos, StandardCharsets.UTF_8)) {
      FileStatus status = fs.getFileStatus(src);
      isDirectory = status.isDirectory();
      if (!isDirectory) {
        zip.putNextEntry(new ZipEntry(src.getName()));
        try (InputStream in = fs.open(status.getPath())) {
          int read;
          while ((read = in.read(buffer)) >= 0) {
            zip.write(buffer, 0, read);
            bytes += read;
          }
        }
        zip.closeEntry();
        files = 1;
      } else {
        Path root = status.getPath();
        String rootPath = root.toUri().getPath();
        if (!rootPath.endsWith("/")) {
          rootPath = rootPath + "/";
        }
        RemoteIterator<LocatedFileStatus> it = fs.listFiles(root, true);
        while (it.hasNext()) {
          LocatedFileStatus item = it.next();
          String filePath = item.getPath().toUri().getPath();
          if (!filePath.startsWith(rootPath) || filePath.endsWith(".crc")) {
            continue;
          }
          String relative = filePath.substring(rootPath.length());
          if (relative.isEmpty()) {
            continue;
          }
          zip.putNextEntry(new ZipEntry(relative));
          try (InputStream in = fs.open(item.getPath())) {
            int read;
            while ((read = in.read(buffer)) >= 0) {
              zip.write(buffer, 0, read);
              bytes += read;
            }
          }
          zip.closeEntry();
          files++;
        }
      }
    } catch (IOException e) {
      failure = e;
    } finally {
      failure = closePrivate(fs, src, failure);
    }
    if (failure != null) {
      throw failure;
    }
    result.put("isDirectory", Boolean.toString(isDirectory));
    result.put("filesRead", Long.toString(files));
    result.put("bytesRead", Long.toString(bytes));
    result.put("archive", Base64.getEncoder().encodeToString(baos.toByteArray()));
  }

  private static void writeArchive(UnaryOperator<String> options, Map<String, String> result)
      throws IOException {
    Path dst = requireCabPath(options, "dst");
    byte[] archive = Base64.getDecoder().decode(required(options, "archive"));
    Configuration conf = activeConf();
    FileSystem fs = open(dst, conf);
    IOException failure = null;
    long files = 0;
    long bytes = 0;
    byte[] buffer = new byte[1 << 16];
    try (ZipInputStream zip =
        new ZipInputStream(new ByteArrayInputStream(archive), StandardCharsets.UTF_8)) {
      ZipEntry entry;
      while ((entry = zip.getNextEntry()) != null) {
        if (entry.isDirectory()) {
          zip.closeEntry();
          continue;
        }
        String name = validateRelativeEntryName(entry.getName());
        Path target = new Path(dst, name);
        try (OutputStream out = fs.create(target, true)) {
          int read;
          while ((read = zip.read(buffer)) >= 0) {
            out.write(buffer, 0, read);
            bytes += read;
          }
        }
        files++;
        zip.closeEntry();
      }
    } catch (IOException e) {
      failure = e;
    } finally {
      failure = closePrivate(fs, dst, failure);
    }
    if (failure != null) {
      throw failure;
    }
    result.put("filesWritten", Long.toString(files));
    result.put("bytesWritten", Long.toString(bytes));
  }

  private static String validateRelativeEntryName(String raw) {
    if (raw == null || raw.isEmpty() || raw.startsWith("/") || raw.contains("\\")) {
      throw new IllegalArgumentException("Invalid archive entry path: " + raw);
    }
    for (String segment : raw.split("/")) {
      if (segment.isEmpty() || ".".equals(segment) || "..".equals(segment)) {
        throw new IllegalArgumentException("Invalid archive entry path: " + raw);
      }
    }
    return raw;
  }

  private static void registerModelUdf(UnaryOperator<String> options, Map<String, String> result)
      throws IOException, ReflectiveOperationException {
    String name = required(options, "name");
    String modelPath = required(options, "modelPath");
    SparkSession session = MlflowCabSparkBridge.activeSession();
    if (session == null) {
      throw new IllegalStateException(
          "No active SparkSession on the Spark driver; cannot register the model UDF");
    }
    // The pipeline is read through mlflow-gs://, so the session's base Hadoop configurations must
    // resolve the provider (e.g. when the JAR arrived through spark.addArtifact).
    Configuration conf = session.sparkContext().hadoopConfiguration();
    MlflowCabClassLoaders.install(conf);
    MlflowCabClassLoaders.installOnDriver(session.sparkContext());
    PipelineModel model = loadPipeline(modelPath, conf);
    MlflowCabLocalPipeline pipeline = MlflowCabLocalPipeline.from(model);
    session.udf().register(name, new MlflowCabSparkModelUdf(pipeline), DataTypes.DoubleType);
    result.put("inputCols", String.join(",", pipeline.inputCols()));
    result.put("stages", String.join(",", pipeline.stageNames()));
  }

  /**
   * Loads a fitted pipeline of JVM stages. Pipelines written by PySpark's pipeline writer (for
   * example saved over Spark Connect) record {@value #PYTHON_PIPELINE_CLASS} as their class even
   * when every stage is a JVM stage; those stages are loaded one by one and assembled in {@code
   * stageUids} order.
   */
  static PipelineModel loadPipeline(String modelPath, Configuration conf)
      throws IOException, ReflectiveOperationException {
    Path root = new Path(modelPath);
    FileSystem fs = root.getFileSystem(conf);
    Map<String, Object> metadata = readMetadata(fs, root);
    if (!PYTHON_PIPELINE_CLASS.equals(MlflowCabJson.firstString(metadata, "class"))) {
      return PipelineModel.load(modelPath);
    }
    Map<String, Object> params = MlflowCabJson.object(metadata, "paramMap");
    Object stageUids = params == null ? null : params.get("stageUids");
    if (!(stageUids instanceof List<?> uids)) {
      throw new IOException("Saved pipeline metadata has no stageUids under " + modelPath);
    }
    Map<String, Path> byUid = new HashMap<>();
    for (FileStatus status : fs.listStatus(new Path(root, "stages"))) {
      String dirName = status.getPath().getName();
      int separator = dirName.indexOf('_');
      if (status.isDirectory() && separator > 0) {
        byUid.put(dirName.substring(separator + 1), status.getPath());
      }
    }
    Transformer[] stages = new Transformer[uids.size()];
    for (int i = 0; i < stages.length; i++) {
      String uid = String.valueOf(uids.get(i));
      Path directory = byUid.get(uid);
      if (directory == null) {
        throw new IOException("Saved pipeline stage " + uid + " is missing under " + modelPath);
      }
      String className = MlflowCabJson.firstString(readMetadata(fs, directory), "class");
      Object stage =
          Class.forName(className, true, stageClassLoader())
              .getMethod("load", String.class)
              .invoke(null, directory.toString());
      if (!(stage instanceof Transformer transformer)) {
        throw new IOException("Saved pipeline stage " + uid + " is not a fitted transformer");
      }
      stages[i] = transformer;
    }
    String uid = MlflowCabJson.firstString(metadata, "uid");
    return new PipelineModel(uid == null ? "PipelineModel" : uid, stages);
  }

  private static ClassLoader stageClassLoader() {
    ClassLoader context = Thread.currentThread().getContextClassLoader();
    return context != null ? context : PipelineModel.class.getClassLoader();
  }

  /** First JSON line of a saved Spark ML {@code metadata/part-*} file under {@code dir}. */
  private static Map<String, Object> readMetadata(FileSystem fs, Path dir) throws IOException {
    FileStatus[] parts =
        fs.listStatus(new Path(dir, "metadata"), path -> path.getName().startsWith("part-"));
    Arrays.sort(parts, Comparator.comparing((FileStatus status) -> status.getPath().getName()));
    for (FileStatus part : parts) {
      try (BufferedReader reader =
          new BufferedReader(
              new InputStreamReader(fs.open(part.getPath()), StandardCharsets.UTF_8))) {
        String line = reader.readLine();
        if (line != null && !line.isBlank()) {
          return MlflowCabJson.parseObject(line);
        }
      }
    }
    throw new IOException("No Spark ML metadata found under " + dir);
  }

  private static String required(UnaryOperator<String> options, String key) {
    String value = options.apply(key);
    if (value == null || value.trim().isEmpty()) {
      throw new IllegalArgumentException("Missing required option '" + key + "'");
    }
    return value.trim();
  }

  private static String lower(String value, String fallback) {
    return value == null || value.trim().isEmpty()
        ? fallback
        : value.trim().toLowerCase(Locale.ROOT);
  }

  private static Long parseLong(String value) {
    if (value == null || value.trim().isEmpty()) {
      return null;
    }
    try {
      return (long) Double.parseDouble(value.trim());
    } catch (NumberFormatException e) {
      return null;
    }
  }
}
