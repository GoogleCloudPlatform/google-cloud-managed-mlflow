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
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.io.FileNotFoundException;
import java.io.InputStream;
import java.io.OutputStream;
import java.net.URI;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.EnumSet;
import java.util.List;
import java.util.Map;
import java.util.ServiceLoader;
import java.util.TreeMap;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.AbstractFileSystem;
import org.apache.hadoop.fs.BlockLocation;
import org.apache.hadoop.fs.CreateFlag;
import org.apache.hadoop.fs.FSDataOutputStream;
import org.apache.hadoop.fs.FileContext;
import org.apache.hadoop.fs.FileStatus;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.LocatedFileStatus;
import org.apache.hadoop.fs.Options;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.fs.permission.FsPermission;
import org.apache.spark.sql.sources.DataSourceRegister;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

class MlflowCabFileSystemTest {

  private static final String SCOPE = "0123456789abcdef01234567";

  @BeforeEach
  void setUp() throws Exception {
    InMemoryGcsFileSystem.reset();
    FileSystem.closeAll();
  }

  @AfterEach
  void tearDown() throws Exception {
    FileSystem.closeAll();
  }

  private static Configuration callerConf() {
    Configuration conf = new Configuration();
    conf.set(MlflowCabConstants.CONF_DELEGATE_IMPL, InMemoryGcsFileSystem.class.getName());
    conf.set("fs.gs.auth.type", "APPLICATION_DEFAULT");
    conf.set("fs.gs.auth.impersonation.service.account", "customer-sa@example.iam.gserviceaccount.com");
    conf.set("fs.gs.project.id", "customer-project");
    return conf;
  }

  private static Map<String, String> fsGsKeys(Configuration conf) {
    Map<String, String> keys = new TreeMap<>();
    for (Map.Entry<String, String> entry : conf) {
      if (entry.getKey().startsWith("fs.gs.")) {
        keys.put(entry.getKey(), entry.getValue());
      }
    }
    return keys;
  }

  @Test
  void serviceLoaderResolvesSchemeWithoutExplicitImplKey() throws Exception {
    Configuration conf = new Configuration();
    assertNull(conf.get("fs.mlflow-gs.impl"));
    FileSystem fs = FileSystem.get(URI.create("mlflow-gs://b/p"), conf);
    assertInstanceOf(MlflowCabFileSystem.class, fs);
    assertEquals("mlflow-gs", fs.getScheme());
    assertEquals(URI.create("mlflow-gs://b"), fs.getUri());
    assertNull(conf.get("fs.mlflow-gs.impl"));
    assertSame(MlflowCabFileSystem.class, FileSystem.getFileSystemClass("mlflow-gs", conf));

    List<Class<?>> fileSystems = new ArrayList<>();
    ServiceLoader.load(FileSystem.class).stream().forEach(p -> fileSystems.add(p.type()));
    assertTrue(fileSystems.contains(MlflowCabFileSystem.class));
  }

  @Test
  void abstractFileSystemServiceAndFileContext() throws Exception {
    List<Class<?>> types = new ArrayList<>();
    ServiceLoader.load(AbstractFileSystem.class).stream().forEach(p -> types.add(p.type()));
    assertTrue(types.contains(MlflowCabAbstractFileSystem.class));
    boolean instantiated = false;
    for (AbstractFileSystem afs : ServiceLoader.load(AbstractFileSystem.class)) {
      if (afs instanceof MlflowCabAbstractFileSystem) {
        instantiated = true;
      }
    }
    assertTrue(instantiated, "ServiceLoader must be able to instantiate the provider");

    Configuration conf = callerConf();
    conf.set(MlflowCabClassLoaders.AFS_IMPL_KEY, MlflowCabAbstractFileSystem.class.getName());
    URI root = URI.create("mlflow-gs://" + SCOPE + "@bucket/");
    AbstractFileSystem created = AbstractFileSystem.createFileSystem(root, conf);
    assertInstanceOf(MlflowCabAbstractFileSystem.class, created);
    assertEquals(-1, created.getUriDefaultPort());

    FileContext context = FileContext.getFileContext(root, conf);
    Path file = new Path("mlflow-gs://" + SCOPE + "@bucket/fc/data.txt");
    try (FSDataOutputStream out =
        context.create(file, EnumSet.of(CreateFlag.CREATE), Options.CreateOpts.createParent())) {
      out.write("file-context".getBytes(StandardCharsets.UTF_8));
    }
    FileStatus status = context.getFileStatus(file);
    assertEquals("mlflow-gs", status.getPath().toUri().getScheme());
    assertEquals(SCOPE + "@bucket", status.getPath().toUri().getAuthority());
    assertTrue(InMemoryGcsFileSystem.OBJECTS.containsKey("bucket/fc/data.txt"));
  }

  @Test
  void dataSourceRegisterServiceIsPackaged() {
    List<Class<?>> sources = new ArrayList<>();
    ServiceLoader.load(DataSourceRegister.class).stream().forEach(p -> sources.add(p.type()));
    assertTrue(sources.contains(MlflowCabDriverControlSource.class));
  }

  @Test
  void translatesPathsBothWaysForReadWriteListRenameDelete() throws Exception {
    Configuration conf = callerConf();
    Path root = new Path("mlflow-gs://" + SCOPE + "@bucket/runs/r1/artifacts/model");
    FileSystem fs = root.getFileSystem(conf);
    assertInstanceOf(MlflowCabFileSystem.class, fs);
    assertEquals(SCOPE, ((MlflowCabFileSystem) fs).getScopeId());

    Path part = new Path(root, "sparkml/_temporary/0/part-00000.parquet");
    try (OutputStream out = fs.create(part, true)) {
      out.write("payload".getBytes(StandardCharsets.UTF_8));
    }
    Path committed = new Path(root, "sparkml/data/part-00000.parquet");
    assertTrue(fs.rename(part, committed));
    try (InputStream in = fs.open(committed)) {
      assertArrayEquals(
          "payload".getBytes(StandardCharsets.UTF_8), MlflowCabTestSupport.readAll(in));
    }
    FileStatus[] listing = fs.listStatus(new Path(root, "sparkml"));
    assertTrue(listing.length >= 1);
    for (FileStatus status : listing) {
      assertEquals("mlflow-gs", status.getPath().toUri().getScheme());
      assertEquals(SCOPE + "@bucket", status.getPath().toUri().getAuthority());
      assertEquals(status.getPath(), fs.makeQualified(status.getPath()));
    }
    FileStatus[] globbed = fs.globStatus(new Path(root, "sparkml/data/*.parquet"));
    assertEquals(1, globbed.length);
    assertEquals(committed, globbed[0].getPath());
    assertTrue(fs.exists(committed));
    assertTrue(fs.getFileStatus(new Path(root, "sparkml")).isDirectory());
    assertTrue(fs.delete(new Path(root, "sparkml"), true));
    assertThrows(FileNotFoundException.class, () -> fs.getFileStatus(committed));
    for (String seen : InMemoryGcsFileSystem.SEEN_PATHS) {
      assertTrue(seen.startsWith("gs://bucket/"), seen);
    }
    assertThrows(IllegalArgumentException.class, () -> fs.getFileStatus(new Path("gs://bucket/x")));

    Path relative = new Path("relative/file.txt");
    fs.setWorkingDirectory(root);
    assertEquals(new Path(root, "relative/file.txt"), fs.makeQualified(relative));
    assertNull(fs.getCanonicalServiceName());
  }

  @Test
  void callerConfigurationAndCustomerGcsCacheStayUntouched() throws Exception {
    Configuration conf = callerConf();
    conf.set("fs.gs.impl", InMemoryGcsFileSystem.class.getName());
    Map<String, String> before = fsGsKeys(conf);
    ClassLoader loaderBefore = conf.getClassLoader();
    FileSystem customer = FileSystem.get(URI.create("gs://customer-bucket/data"), conf);

    Path path = new Path("mlflow-gs://" + SCOPE + "@managed-bucket/x/y.txt");
    MlflowCabFileSystem fs = (MlflowCabFileSystem) path.getFileSystem(conf);
    try (OutputStream out = fs.create(path)) {
      out.write(1);
    }

    assertEquals(before, fsGsKeys(conf), "caller fs.gs.* keys must not change");
    assertSame(loaderBefore, conf.getClassLoader());
    assertSame(customer, FileSystem.get(URI.create("gs://customer-bucket/data"), conf));

    Configuration inner = fs.innerConfiguration();
    assertEquals("ACCESS_TOKEN_PROVIDER", inner.get("fs.gs.auth.type"));
    assertEquals(MlflowCabConstants.PROVIDER_CLASS, inner.get("fs.gs.auth.access.token.provider"));
    assertEquals("true", inner.get("fs.gs.impl.disable.cache"));
    assertEquals("false", inner.get("fs.gs.create.items.conflict.check.enable"));
    assertEquals("false", inner.get("fs.gs.implicit.dir.repair.enable"));
    assertEquals(SCOPE, inner.get(MlflowCabConstants.CONF_ACTIVE_SCOPE));
    assertNull(inner.get("fs.gs.auth.impersonation.service.account"));
    assertEquals("customer-project", inner.get("fs.gs.project.id"));
    assertSame(MlflowCabFileSystem.class.getClassLoader(), inner.getClassLoader());
    assertEquals(MlflowCabConstants.PROVIDER_CLASS, inner.getClassByName(MlflowCabConstants.PROVIDER_CLASS).getName());
    assertEquals(
        "customer-sa@example.iam.gserviceaccount.com",
        conf.get("fs.gs.auth.impersonation.service.account"));
    // The delegate was initialized with the private clone, never the caller configuration.
    for (Configuration seen : InMemoryGcsFileSystem.INIT_CONFS) {
      if ("ACCESS_TOKEN_PROVIDER".equals(seen.get("fs.gs.auth.type"))) {
        assertFalse(seen == conf);
      }
    }
  }

  @Test
  void rejectsInvalidScopeIdsAndBucketlessIo() throws Exception {
    Configuration conf = callerConf();
    MlflowCabFileSystem fs = new MlflowCabFileSystem();
    assertThrows(
        IllegalArgumentException.class,
        () -> fs.initialize(URI.create("mlflow-gs://NOT_HEX@bucket/"), conf));
    MlflowCabFileSystem bucketless = new MlflowCabFileSystem();
    bucketless.initialize(URI.create("mlflow-gs:///"), conf);
    assertThrows(Exception.class, () -> bucketless.getFileStatus(new Path("mlflow-gs:///x")));
  }

  @Test
  void createOnlyWritesSurfaceExistingObjects() throws Exception {
    Configuration conf = callerConf();
    Path path = new Path("mlflow-gs://" + SCOPE + "@vault/registeredModels/u1/versions/1/MLmodel");
    FileSystem fs = path.getFileSystem(conf);
    try (OutputStream out = fs.create(path, false)) {
      out.write(7);
    }
    assertThrows(
        Exception.class,
        () -> fs.create(path, FsPermission.getFileDefault(), false, 4096, (short) 1, 1024L, null));
  }

  @Test
  void toOuterPreservesLocatedFileStatusBlockLocations() throws Exception {
    MlflowCabFileSystem fs =
        (MlflowCabFileSystem)
            new Path("mlflow-gs://" + SCOPE + "@bucket/").getFileSystem(callerConf());
    BlockLocation[] blocks = {
      new BlockLocation(new String[] {"h1:9866"}, new String[] {"h1"}, 0L, 10L),
      new BlockLocation(new String[] {"h2:9866"}, new String[] {"h2"}, 10L, 5L)
    };
    FileStatus inner = new FileStatus(15L, false, 1, 1024L, 7L, new Path("gs://bucket/runs/r/a.bin"));
    FileStatus outer = fs.toOuter(new LocatedFileStatus(inner, blocks));
    assertInstanceOf(LocatedFileStatus.class, outer);
    assertArrayEquals(blocks, ((LocatedFileStatus) outer).getBlockLocations());
    assertEquals(new Path("mlflow-gs://" + SCOPE + "@bucket/runs/r/a.bin"), outer.getPath());
    assertEquals(15L, outer.getLen());
    FileStatus plain = fs.toOuter(inner);
    assertFalse(plain instanceof LocatedFileStatus);
    assertEquals(new Path("mlflow-gs://" + SCOPE + "@bucket/runs/r/a.bin"), plain.getPath());
  }

  private static void write(FileSystem fs, Path path, String text) throws Exception {
    try (OutputStream out = fs.create(path, false)) {
      out.write(text.getBytes(StandardCharsets.UTF_8));
    }
  }

  @Test
  void driverCopyHandlesTreesAndSingleFilesAndClosesOnlyPrivateInstances() throws Exception {
    Configuration conf = callerConf();
    String base = "mlflow-gs://" + SCOPE + "@bucket";
    FileSystem shared = FileSystem.get(URI.create(base + "/"), conf);
    write(shared, new Path(base + "/src/model/sparkml/a.bin"), "alpha");
    write(shared, new Path(base + "/src/model/sparkml/stages/0/b.bin"), "beta!");
    write(shared, new Path(base + "/src/single.txt"), "one");
    int closedBefore = InMemoryGcsFileSystem.CLOSED.get();

    Map<String, String> tree = new TreeMap<>();
    MlflowCabDriverActions.copy(
        new Path(base + "/src/model/sparkml"), new Path(base + "/dst/v1/sparkml"), conf, tree);
    assertEquals("2", tree.get("filesCopied"));
    assertEquals("10", tree.get("bytesCopied"));
    assertTrue(InMemoryGcsFileSystem.OBJECTS.containsKey("bucket/dst/v1/sparkml/a.bin"));
    assertTrue(InMemoryGcsFileSystem.OBJECTS.containsKey("bucket/dst/v1/sparkml/stages/0/b.bin"));
    // Two private (uncached) mlflow-gs instances were opened and closed with their delegates.
    assertEquals(closedBefore + 2, InMemoryGcsFileSystem.CLOSED.get());
    assertSame(shared, FileSystem.get(URI.create(base + "/"), conf), "shared instance untouched");
    assertTrue(shared.exists(new Path(base + "/dst/v1/sparkml/a.bin")));

    Map<String, String> again = new TreeMap<>();
    MlflowCabDriverActions.copy(
        new Path(base + "/src/model/sparkml"), new Path(base + "/dst/v1/sparkml"), conf, again);
    assertEquals("0", again.get("filesCopied"));
    assertEquals("2", again.get("filesSkipped"));

    Map<String, String> single = new TreeMap<>();
    MlflowCabDriverActions.copy(
        new Path(base + "/src/single.txt"), new Path(base + "/dst/v1/single.txt"), conf, single);
    assertEquals("1", single.get("filesCopied"));
    assertArrayEquals(
        "one".getBytes(StandardCharsets.UTF_8),
        InMemoryGcsFileSystem.OBJECTS.get("bucket/dst/v1/single.txt"));

    write(shared, new Path(base + "/src/other.txt"), "different-size");
    assertThrows(
        java.io.IOException.class,
        () ->
            MlflowCabDriverActions.copy(
                new Path(base + "/src/other.txt"),
                new Path(base + "/dst/v1/single.txt"),
                conf,
                new TreeMap<>()));

    java.io.File local = java.nio.file.Files.createTempDirectory("mlflow-cab-copy").toFile();
    FileSystem localFs = FileSystem.get(URI.create("file:///"), conf);
    Map<String, String> toLocal = new TreeMap<>();
    MlflowCabDriverActions.copy(
        new Path(base + "/src/model/sparkml"),
        new Path(local.toURI().toString() + "sparkml"),
        conf,
        toLocal);
    assertEquals("2", toLocal.get("filesCopied"));
    assertTrue(new java.io.File(local, "sparkml/stages/0/b.bin").isFile());
    assertSame(localFs, FileSystem.get(URI.create("file:///"), conf), "cached file:// kept open");
  }
}
