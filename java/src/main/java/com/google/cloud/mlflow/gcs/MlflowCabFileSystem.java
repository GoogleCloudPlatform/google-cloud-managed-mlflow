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

import java.io.FileNotFoundException;
import java.io.IOException;
import java.net.URI;
import java.util.EnumSet;
import java.util.Locale;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.CreateFlag;
import org.apache.hadoop.fs.FSDataInputStream;
import org.apache.hadoop.fs.FSDataOutputStream;
import org.apache.hadoop.fs.FileChecksum;
import org.apache.hadoop.fs.FileStatus;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.FsStatus;
import org.apache.hadoop.fs.LocatedFileStatus;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.fs.permission.FsPermission;
import org.apache.hadoop.util.Progressable;

/**
 * Hadoop {@link FileSystem} for {@code mlflow-gs://[<scope>@]<bucket>/<path>} URIs.
 *
 * <p>Wraps a private Cloud Storage connector instance that is initialized for {@code
 * gs://<bucket>} over a <em>clone</em> of the caller's configuration. Incoming {@code mlflow-gs}
 * paths are translated to {@code gs} for the delegate, and every outgoing path ({@link #getUri()},
 * working directory, and {@link FileStatus} results) is rewritten back to {@code mlflow-gs}, so
 * Hadoop and Spark committers, {@code makeQualified}, and {@code checkPath} never observe a scheme
 * mismatch. The optional {@code <scope>} user-info component binds the instance to exactly one
 * Credential Access Boundary token scope.
 *
 * <p>Registered through {@code META-INF/services/org.apache.hadoop.fs.FileSystem}, so a static
 * classpath install resolves the scheme without {@code fs.mlflow-gs.impl}.
 */
public class MlflowCabFileSystem extends FileSystem {

  private URI uri;
  private String bucket = "";
  private String scopeId;
  private Configuration innerConf;
  private Path workingDir;
  private final Object innerLock = new Object();
  private volatile FileSystem inner;

  /** Public no-arg constructor required by Hadoop's ServiceLoader discovery. */
  public MlflowCabFileSystem() {}

  @Override
  public String getScheme() {
    return MlflowCabConstants.SCHEME;
  }

  @Override
  public void initialize(URI name, Configuration conf) throws IOException {
    super.initialize(name, conf);
    setConf(conf);
    String scheme = name.getScheme();
    if (scheme == null || !MlflowCabConstants.SCHEME.equalsIgnoreCase(scheme)) {
      throw new IllegalArgumentException("Unsupported scheme for " + getClass().getSimpleName());
    }
    String authority = name.getRawAuthority() == null ? "" : name.getRawAuthority();
    int at = authority.lastIndexOf('@');
    if (at >= 0) {
      scopeId = authority.substring(0, at).toLowerCase(Locale.ROOT);
      bucket = authority.substring(at + 1).toLowerCase(Locale.ROOT);
      if (!MlflowCabScope.isValidScopeId(scopeId)) {
        throw new IllegalArgumentException("Invalid MLflow CAB scope id in mlflow-gs:// URI");
      }
    } else {
      scopeId = null;
      bucket = authority.toLowerCase(Locale.ROOT);
    }
    String normalizedAuthority = scopeId == null ? bucket : scopeId + "@" + bucket;
    uri =
        normalizedAuthority.isEmpty()
            ? URI.create(MlflowCabConstants.SCHEME + ":///")
            : URI.create(MlflowCabConstants.SCHEME + "://" + normalizedAuthority);
    innerConf = MlflowCabInnerConfiguration.build(conf, scopeId);
    workingDir = new Path(MlflowCabConstants.SCHEME, normalizedAuthority, "/");
  }

  /** Returns the CAB scope id bound to this instance, or null. */
  public String getScopeId() {
    return scopeId;
  }

  /** Returns the private configuration handed to the wrapped connector (for diagnostics). */
  Configuration innerConfiguration() {
    return innerConf;
  }

  /** Returns the wrapped {@code gs://} filesystem, creating it on first use. */
  FileSystem delegate() throws IOException {
    FileSystem current = inner;
    if (current != null) {
      return current;
    }
    synchronized (innerLock) {
      if (inner == null) {
        if (bucket.isEmpty()) {
          throw new IOException("mlflow-gs:// URI does not name a bucket: " + uri);
        }
        FileSystem created = MlflowCabGcsDelegate.create(innerConf);
        created.initialize(
            URI.create(MlflowCabConstants.GCS_SCHEME + "://" + bucket + "/"), innerConf);
        inner = created;
      }
      return inner;
    }
  }

  Path toInner(Path path) {
    Path qualified = makeQualified(path);
    String rawPath = qualified.toUri().getPath();
    return new Path(
        MlflowCabConstants.GCS_SCHEME, bucket, rawPath == null || rawPath.isEmpty() ? "/" : rawPath);
  }

  Path toOuter(Path innerPath) {
    String rawPath = innerPath.toUri().getPath();
    return new Path(
        MlflowCabConstants.SCHEME,
        uri.getRawAuthority(),
        rawPath == null || rawPath.isEmpty() ? "/" : rawPath);
  }

  /** Rewrites a delegate status to this scheme, preserving block locations when present. */
  FileStatus toOuter(FileStatus status) throws IOException {
    FileStatus copy = new FileStatus(status);
    copy.setPath(toOuter(status.getPath()));
    if (status instanceof LocatedFileStatus located) {
      return new LocatedFileStatus(copy, located.getBlockLocations());
    }
    return copy;
  }

  @Override
  public URI getUri() {
    return uri;
  }

  @Override
  public FSDataInputStream open(Path path, int bufferSize) throws IOException {
    return delegate().open(toInner(path), bufferSize);
  }

  @Override
  public FSDataOutputStream create(
      Path path,
      FsPermission permission,
      boolean overwrite,
      int bufferSize,
      short replication,
      long blockSize,
      Progressable progress)
      throws IOException {
    return delegate()
        .create(toInner(path), permission, overwrite, bufferSize, replication, blockSize, progress);
  }

  @Override
  public FSDataOutputStream createNonRecursive(
      Path path,
      FsPermission permission,
      EnumSet<CreateFlag> flags,
      int bufferSize,
      short replication,
      long blockSize,
      Progressable progress)
      throws IOException {
    return delegate()
        .createNonRecursive(
            toInner(path), permission, flags, bufferSize, replication, blockSize, progress);
  }

  @Override
  public FSDataOutputStream append(Path path, int bufferSize, Progressable progress)
      throws IOException {
    return delegate().append(toInner(path), bufferSize, progress);
  }

  @Override
  public boolean rename(Path src, Path dst) throws IOException {
    return delegate().rename(toInner(src), toInner(dst));
  }

  @Override
  public boolean delete(Path path, boolean recursive) throws IOException {
    return delegate().delete(toInner(path), recursive);
  }

  @Override
  public FileStatus[] listStatus(Path path) throws IOException {
    FileStatus[] statuses;
    try {
      statuses = delegate().listStatus(toInner(path));
    } catch (FileNotFoundException e) {
      throw new FileNotFoundException("File " + path + " does not exist.");
    }
    FileStatus[] result = new FileStatus[statuses.length];
    for (int i = 0; i < statuses.length; i++) {
      result[i] = toOuter(statuses[i]);
    }
    return result;
  }

  @Override
  public void setWorkingDirectory(Path newDir) {
    workingDir = makeQualified(newDir);
  }

  @Override
  public Path getWorkingDirectory() {
    return workingDir;
  }

  @Override
  public boolean mkdirs(Path path, FsPermission permission) throws IOException {
    return delegate().mkdirs(toInner(path), permission);
  }

  @Override
  public FileStatus getFileStatus(Path path) throws IOException {
    try {
      return toOuter(delegate().getFileStatus(toInner(path)));
    } catch (FileNotFoundException e) {
      throw new FileNotFoundException("File " + path + " does not exist.");
    }
  }

  @Override
  public long getDefaultBlockSize(Path path) {
    try {
      return delegate().getDefaultBlockSize(toInner(path));
    } catch (IOException e) {
      return 64L * 1024 * 1024;
    }
  }

  @Override
  public short getDefaultReplication(Path path) {
    return 3;
  }

  @Override
  public FileChecksum getFileChecksum(Path path) throws IOException {
    return delegate().getFileChecksum(toInner(path));
  }

  @Override
  public FsStatus getStatus(Path path) throws IOException {
    return delegate().getStatus(toInner(path == null ? workingDir : path));
  }

  @Override
  public void setTimes(Path path, long mtime, long atime) throws IOException {
    delegate().setTimes(toInner(path), mtime, atime);
  }

  @Override
  public void setPermission(Path path, FsPermission permission) throws IOException {
    delegate().setPermission(toInner(path), permission);
  }

  @Override
  public void setOwner(Path path, String username, String groupname) throws IOException {
    delegate().setOwner(toInner(path), username, groupname);
  }

  @Override
  public boolean hasPathCapability(Path path, String capability) throws IOException {
    return delegate().hasPathCapability(toInner(path), capability);
  }

  @Override
  public void concat(Path target, Path[] sources) throws IOException {
    Path[] translated = new Path[sources.length];
    for (int i = 0; i < sources.length; i++) {
      translated[i] = toInner(sources[i]);
    }
    delegate().concat(toInner(target), translated);
  }

  /** CAB secrets are never exported as Hadoop delegation tokens. */
  @Override
  public String getCanonicalServiceName() {
    return null;
  }

  @Override
  public void close() throws IOException {
    try {
      FileSystem current = inner;
      if (current != null) {
        current.close();
      }
    } finally {
      inner = null;
      super.close();
    }
  }

  @Override
  public String toString() {
    return "MlflowCabFileSystem{uri=" + uri + "}";
  }
}
