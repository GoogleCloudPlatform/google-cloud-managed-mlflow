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

import java.util.Collections;
import java.util.EnumSet;
import java.util.Map;
import java.util.Set;
import org.apache.spark.sql.connector.catalog.SupportsRead;
import org.apache.spark.sql.connector.catalog.Table;
import org.apache.spark.sql.connector.catalog.TableCapability;
import org.apache.spark.sql.connector.catalog.TableProvider;
import org.apache.spark.sql.connector.expressions.Transform;
import org.apache.spark.sql.connector.read.Batch;
import org.apache.spark.sql.connector.read.InputPartition;
import org.apache.spark.sql.connector.read.PartitionReader;
import org.apache.spark.sql.connector.read.PartitionReaderFactory;
import org.apache.spark.sql.connector.read.Scan;
import org.apache.spark.sql.connector.read.ScanBuilder;
import org.apache.spark.sql.catalyst.InternalRow;
import org.apache.spark.sql.sources.DataSourceRegister;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.Metadata;
import org.apache.spark.sql.types.MetadataBuilder;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;
import org.apache.spark.sql.util.CaseInsensitiveStringMap;

/**
 * Spark DataSource V2 "control" source that runs MLflow CAB actions on the driver during plan
 * analysis.
 *
 * <p>Clients call {@code spark.read.format(<this class>).options(...).load().schema}; Spark
 * analyzes the relation on the driver (Spark Connect included, where clients have no JVM
 * gateway), which invokes {@link #inferSchema}. No Spark job or executor task is launched. The
 * action result is returned as JSON in the metadata of the single {@code result} field and never
 * contains secrets.
 */
public final class MlflowCabDriverControlSource implements TableProvider, DataSourceRegister {

  /** Metadata key under which the JSON action result is returned. */
  public static final String RESULT_METADATA_KEY = "mlflow_cab_result";

  @Override
  public String shortName() {
    return "mlflow-cab-control";
  }

  @Override
  public StructType inferSchema(CaseInsensitiveStringMap options) {
    Map<String, String> result = MlflowCabDriverActions.execute(options::get);
    Metadata metadata =
        new MetadataBuilder()
            .putString(RESULT_METADATA_KEY, MlflowCabJson.encodeFlat(result))
            .build();
    return new StructType(
        new StructField[] {new StructField("result", DataTypes.StringType, true, metadata)});
  }

  @Override
  public Table getTable(
      StructType schema, Transform[] partitioning, Map<String, String> properties) {
    return new ControlTable(schema);
  }

  /** Read-only, zero-partition table so analysis succeeds without scheduling any task. */
  static final class ControlTable implements SupportsRead {
    private final StructType schema;

    ControlTable(StructType schema) {
      this.schema = schema;
    }

    @Override
    public String name() {
      return "mlflow_cab_control";
    }

    @Override
    @Deprecated
    public StructType schema() {
      return schema;
    }

    @Override
    public Set<TableCapability> capabilities() {
      return Collections.unmodifiableSet(EnumSet.of(TableCapability.BATCH_READ));
    }

    @Override
    public ScanBuilder newScanBuilder(CaseInsensitiveStringMap options) {
      return () -> new EmptyScan(schema);
    }
  }

  /** Scan that plans zero input partitions. */
  static final class EmptyScan implements Scan, Batch {
    private final StructType schema;

    EmptyScan(StructType schema) {
      this.schema = schema;
    }

    @Override
    public StructType readSchema() {
      return schema;
    }

    @Override
    public Batch toBatch() {
      return this;
    }

    @Override
    public InputPartition[] planInputPartitions() {
      return new InputPartition[0];
    }

    @Override
    public PartitionReaderFactory createReaderFactory() {
      return new EmptyReaderFactory();
    }
  }

  /** Reader factory that is never invoked because there are no partitions. */
  static final class EmptyReaderFactory implements PartitionReaderFactory {
    private static final long serialVersionUID = 1L;

    @Override
    public PartitionReader<InternalRow> createReader(InputPartition partition) {
      throw new UnsupportedOperationException("The MLflow CAB control source has no data");
    }
  }
}
