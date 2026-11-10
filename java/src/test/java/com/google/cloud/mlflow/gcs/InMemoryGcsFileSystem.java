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

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.EOFException;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.net.URI;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentSkipListMap;
import java.util.concurrent.CopyOnWriteArrayList;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FSDataInputStream;
import org.apache.hadoop.fs.FSDataOutputStream;
import org.apache.hadoop.fs.FileStatus;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.fs.PositionedReadable;
import org.apache.hadoop.fs.Seekable;
import org.apache.hadoop.fs.permission.FsPermission;
import org.apache.hadoop.util.Progressable;

/**
 * In-memory {@code gs://} test double used as the wrapped delegate so path translation and
 * configuration isolation can be verified without network access. Records every configuration it
 * was initialized with and every path it received.
 */
public final class InMemoryGcsFileSystem extends FileSystem {

  static final Map<String, byte[]> OBJECTS = new ConcurrentSkipListMap<>();
  static final List<Configuration> INIT_CONFS = new CopyOnWriteArrayList<>();
  static final List<String> SEEN_PATHS = new CopyOnWriteArrayList<>();
  static final Map<String, Integer> INSTANCES = new ConcurrentHashMap<>();
  static final java.util.concurrent.atomic.AtomicInteger CLOSED =
      new java.util.concurrent.atomic.AtomicInteger();

  private URI uri;
  private Path workingDir;

  public InMemoryGcsFileSystem() {}

  static void reset() {
    OBJECTS.clear();
    INIT_CONFS.clear();
    SEEN_PATHS.clear();
    INSTANCES.clear();
    CLOSED.set(0);
  }

  @Override
  public String getScheme() {
    return "gs";
  }

  @Override
  public void initialize(URI name, Configuration conf) throws IOException {
    super.initialize(name, conf);
    setConf(conf);
    uri = URI.create("gs://" + name.getAuthority());
    workingDir = new Path(uri.toString() + "/");
    INIT_CONFS.add(conf);
    INSTANCES.merge(name.getAuthority(), 1, Integer::sum);
  }

  @Override
  public void close() throws IOException {
    CLOSED.incrementAndGet();
    super.close();
  }

  private String key(Path path) {
    Path qualified = makeQualified(path);
    String scheme = qualified.toUri().getScheme();
    if (!"gs".equals(scheme)) {
      throw new IllegalArgumentException("Delegate received non-gs path: " + path);
    }
    SEEN_PATHS.add(qualified.toString());
    return qualified.toUri().getAuthority() + qualified.toUri().getPath();
  }

  @Override
  public URI getUri() {
    return uri;
  }

  @Override
  public FSDataInputStream open(Path path, int bufferSize) throws IOException {
    byte[] data = OBJECTS.get(key(path));
    if (data == null) {
      throw new FileNotFoundException(path.toString());
    }
    return new FSDataInputStream(new SeekableBytes(data));
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
    String key = key(path);
    if (!overwrite && OBJECTS.containsKey(key)) {
      throw new IOException("Object exists (create-only): " + path);
    }
    ByteArrayOutputStream buffer =
        new ByteArrayOutputStream() {
          @Override
          public void close() throws IOException {
            super.close();
            OBJECTS.put(key, toByteArray());
          }
        };
    return new FSDataOutputStream(buffer, null);
  }

  @Override
  public FSDataOutputStream append(Path path, int bufferSize, Progressable progress)
      throws IOException {
    throw new IOException("append unsupported");
  }

  @Override
  public boolean rename(Path src, Path dst) throws IOException {
    String from = key(src);
    String to = key(dst);
    boolean moved = false;
    for (String k : new ArrayList<>(OBJECTS.keySet())) {
      if (k.equals(from) || k.startsWith(from + "/")) {
        OBJECTS.put(to + k.substring(from.length()), OBJECTS.remove(k));
        moved = true;
      }
    }
    return moved;
  }

  @Override
  public boolean delete(Path path, boolean recursive) throws IOException {
    String target = key(path);
    boolean deleted = false;
    for (String k : new ArrayList<>(OBJECTS.keySet())) {
      if (k.equals(target) || k.startsWith(target + "/")) {
        OBJECTS.remove(k);
        deleted = true;
      }
    }
    return deleted;
  }

  @Override
  public FileStatus[] listStatus(Path path) throws IOException {
    String prefix = key(path);
    if (OBJECTS.containsKey(prefix)) {
      return new FileStatus[] {getFileStatus(path)};
    }
    String dirPrefix = prefix.endsWith("/") ? prefix : prefix + "/";
    List<FileStatus> result = new ArrayList<>();
    List<String> seenDirs = new ArrayList<>();
    for (Map.Entry<String, byte[]> entry : OBJECTS.entrySet()) {
      if (!entry.getKey().startsWith(dirPrefix)) {
        continue;
      }
      String rest = entry.getKey().substring(dirPrefix.length());
      int slash = rest.indexOf('/');
      if (slash < 0) {
        if (!rest.isEmpty()) {
          result.add(fileStatus(entry.getKey(), entry.getValue().length));
        }
      } else {
        String dir = dirPrefix + rest.substring(0, slash);
        if (!seenDirs.contains(dir)) {
          seenDirs.add(dir);
          result.add(dirStatus(dir));
        }
      }
    }
    if (result.isEmpty()) {
      throw new FileNotFoundException(path.toString());
    }
    return result.toArray(new FileStatus[0]);
  }

  private FileStatus fileStatus(String key, long len) {
    return new FileStatus(len, false, 1, 1024, 0L, new Path("gs://" + key));
  }

  private FileStatus dirStatus(String key) {
    return new FileStatus(0L, true, 1, 1024, 0L, new Path("gs://" + key));
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
    String k = key(path);
    OBJECTS.putIfAbsent(k + "/", new byte[0]);
    return true;
  }

  @Override
  public FileStatus getFileStatus(Path path) throws IOException {
    String k = key(path);
    byte[] data = OBJECTS.get(k);
    if (data != null) {
      return fileStatus(k, data.length);
    }
    String dirPrefix = k.endsWith("/") ? k : k + "/";
    for (String other : OBJECTS.keySet()) {
      if (other.startsWith(dirPrefix)) {
        return dirStatus(k.endsWith("/") ? k.substring(0, k.length() - 1) : k);
      }
    }
    throw new FileNotFoundException(path.toString());
  }

  /** Seekable byte-array stream. */
  static final class SeekableBytes extends ByteArrayInputStream
      implements Seekable, PositionedReadable {
    SeekableBytes(byte[] data) {
      super(data);
    }

    @Override
    public void seek(long newPos) throws IOException {
      if (newPos < 0 || newPos > count) {
        throw new EOFException("bad seek");
      }
      pos = (int) newPos;
    }

    @Override
    public long getPos() {
      return pos;
    }

    @Override
    public boolean seekToNewSource(long targetPos) {
      return false;
    }

    @Override
    public int read(long position, byte[] buffer, int offset, int length) {
      if (position >= count) {
        return -1;
      }
      int n = (int) Math.min(length, count - position);
      System.arraycopy(buf, (int) position, buffer, offset, n);
      return n;
    }

    @Override
    public void readFully(long position, byte[] buffer, int offset, int length) throws IOException {
      if (read(position, buffer, offset, length) < length) {
        throw new EOFException("short read");
      }
    }

    @Override
    public void readFully(long position, byte[] buffer) throws IOException {
      readFully(position, buffer, 0, buffer.length);
    }
  }
}
