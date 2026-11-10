# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Build, test, and package ``mlflow-gcs-cab-provider.jar`` reproducibly.

The provider is compiled with ``javac --release 17 -Xlint:all -Werror`` and its
JUnit 5 suite is executed twice: once against Cloud Storage connector 3.1.x and
once against 4.0.x. The packaged JAR is written deterministically (sorted
entries, fixed timestamps) so rebuilding identical sources yields an identical
SHA-256 digest.

Example::

    python scripts/build_jar.py --jars-dir /path/to/reference-jars \\
        --output google_cloud_managed_mlflow/jars/mlflow-gcs-cab-provider.jar

``--jars-dir`` (or ``MLFLOW_CAB_JARS_DIR``) must contain the compile-time
dependencies: ``gcs-connector-<version>.jar`` for every requested connector,
``hadoop-client-api-*.jar``, ``hadoop-client-runtime-*.jar``, Spark
``core``/``sql``/``sql-api``/``catalyst``/``common-utils`` JARs,
``scala-library-*.jar``, ``slf4j-api-*.jar``, and
``junit-platform-console-standalone-*.jar``. Spark ML JARs are taken from
``--spark-jars-dir`` (defaults to the installed ``pyspark`` distribution).
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import os
import re
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
JAVA_ROOT = REPO_ROOT / "java"
MAIN_SRC = JAVA_ROOT / "src" / "main" / "java"
MAIN_RES = JAVA_ROOT / "src" / "main" / "resources"
TEST_SRC = JAVA_ROOT / "src" / "test" / "java"
DEFAULT_OUTPUT = (
    REPO_ROOT / "google_cloud_managed_mlflow" / "jars" / "mlflow-gcs-cab-provider.jar"
)
DEFAULT_CONNECTORS = ("3.1.20", "4.0.7")
MAX_CLASS_MAJOR = 61  # Java 17
FIXED_ZIP_TIME = (1980, 1, 1, 0, 0, 0)
COMMON_JAR_PATTERNS = (
    "hadoop-client-api-*.jar",
    "hadoop-client-runtime-*.jar",
    "spark-core_*.jar",
    "spark-sql_*.jar",
    "spark-sql-api_*.jar",
    "spark-catalyst_*.jar",
    "spark-common-utils_*.jar",
    "scala-library-*.jar",
    "slf4j-api-*.jar",
)
SPARK_ML_PATTERNS = (
    "spark-mllib_*.jar",
    "spark-mllib-local_*.jar",
    "scala-reflect-*.jar",
)
# Test-runtime classpath: every other JAR of the Spark distribution (logging bridges,
# Kryo, ICU, Jackson, ...). The pinned compile-time JARs above stay first on the
# classpath, so these only supply classes the reference set does not contain.
TEST_RUNTIME_EXCLUDED_PREFIXES = (
    "spark-core_",
    "spark-sql_",
    "spark-sql-api_",
    "spark-catalyst_",
    "spark-common-utils_",
    "spark-mllib_",
    "spark-mllib-local_",
    "spark-connect",
    "hadoop-client-api-",
    "hadoop-client-runtime-",
    "scala-library-",
    "scala-reflect-",
    "slf4j-api-",
    "gcs-connector",
)
MANIFEST = (
    "Manifest-Version: 1.0\r\n"
    "Created-By: google-cloud-managed-mlflow scripts/build_jar.py\r\n"
    "Implementation-Title: mlflow-gcs-cab-provider\r\n"
    "Implementation-Vendor: Google LLC\r\n"
    "Implementation-Version: 0.2.0\r\n"
    "\r\n"
)


def _find_one(directory: Path, pattern: str) -> Path:
    matches = sorted(Path(p) for p in glob.glob(str(directory / pattern)))
    if not matches:
        raise SystemExit(f"Missing required JAR matching {pattern!r} in {directory}")
    return matches[-1]


def _java_tool(name: str, explicit: str | None) -> str:
    if explicit:
        return explicit
    java_home = os.environ.get("JAVA_HOME")
    candidates = []
    if java_home:
        candidates.append(Path(java_home) / "bin" / name)
    for jdk_dir in (
        "/usr/lib/jvm/java-21-openjdk-amd64/bin",
        "/usr/lib/jvm/java-17-openjdk-amd64/bin",
        "/usr/local/buildtools/java/jdk/bin",
    ):
        candidates.append(Path(jdk_dir) / name)
    for candidate in candidates:
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return str(candidate)
    found = shutil.which(name)
    if not found:
        raise SystemExit(f"Unable to locate {name}; set JAVA_HOME or pass --{name}")
    return found


def _default_spark_jars_dir() -> Path | None:
    try:
        import pyspark  # noqa: PLC0415
    except ImportError:
        return None
    jars = Path(pyspark.__file__).resolve().parent / "jars"
    return jars if jars.is_dir() else None


def _run(cmd: list[str], log_path: Path) -> int:
    with log_path.open("w", encoding="utf-8") as log:
        log.write("$ " + " ".join(cmd) + "\n")
        log.flush()
        proc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=False)
        log.write(f"\nexit={proc.returncode}\n")
    return proc.returncode


def _sources(root: Path) -> list[str]:
    return sorted(str(p) for p in root.rglob("*.java"))


def _class_major(data: bytes) -> int:
    if data[:4] != b"\xca\xfe\xba\xbe":
        raise ValueError("not a class file")
    return int.from_bytes(data[6:8], "big")


def _write_deterministic_jar(classes_dir: Path, output: Path) -> list[str]:
    entries: dict[str, bytes] = {}
    for path in sorted(classes_dir.rglob("*.class")):
        entries[path.relative_to(classes_dir).as_posix()] = path.read_bytes()
    for path in sorted(MAIN_RES.rglob("*")):
        if path.is_file():
            entries[path.relative_to(MAIN_RES).as_posix()] = path.read_bytes()
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_suffix(".jar.tmp")
    names = ["META-INF/MANIFEST.MF", *sorted(entries)]
    with zipfile.ZipFile(tmp, "w") as jar:
        for directory in ("META-INF/",):
            info = zipfile.ZipInfo(directory, FIXED_ZIP_TIME)
            info.external_attr = 0o40755 << 16
            jar.writestr(info, b"")
        info = zipfile.ZipInfo("META-INF/MANIFEST.MF", FIXED_ZIP_TIME)
        info.compress_type = zipfile.ZIP_DEFLATED
        info.external_attr = 0o644 << 16
        jar.writestr(info, MANIFEST.encode("utf-8"))
        for name in sorted(entries):
            info = zipfile.ZipInfo(name, FIXED_ZIP_TIME)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            jar.writestr(info, entries[name])
    os.replace(tmp, output)
    return names


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--jars-dir", default=os.environ.get("MLFLOW_CAB_JARS_DIR"))
    parser.add_argument("--spark-jars-dir", default=None)
    parser.add_argument("--connectors", default=",".join(DEFAULT_CONNECTORS))
    parser.add_argument("--work-dir", default=str(REPO_ROOT / "build" / "java"))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--javac", default=None)
    parser.add_argument("--java", default=None)
    parser.add_argument("--javap", default=None)
    parser.add_argument(
        "--extra-test-java",
        default="",
        help="Comma-separated additional java executables to rerun the suite with.",
    )
    parser.add_argument("--skip-tests", action="store_true")
    args = parser.parse_args(argv)

    if not args.jars_dir:
        raise SystemExit("Pass --jars-dir or set MLFLOW_CAB_JARS_DIR")
    jars_dir = Path(args.jars_dir)
    spark_jars = (
        Path(args.spark_jars_dir) if args.spark_jars_dir else _default_spark_jars_dir()
    )
    if spark_jars is None:
        raise SystemExit("Pass --spark-jars-dir (Spark ML JARs are required)")
    javac = _java_tool("javac", args.javac)
    java = _java_tool("java", args.java)
    javap = _java_tool("javap", args.javap)
    work = Path(args.work_dir)
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    logs = work / "logs"
    logs.mkdir()

    common = [_find_one(jars_dir, p) for p in COMMON_JAR_PATTERNS]
    common += [_find_one(spark_jars, p) for p in SPARK_ML_PATTERNS]
    test_runtime = sorted(
        p
        for p in spark_jars.glob("*.jar")
        if not p.name.startswith(TEST_RUNTIME_EXCLUDED_PREFIXES)
    )
    junit = _find_one(jars_dir, "junit-platform-console-standalone-*.jar")
    main_sources = _sources(MAIN_SRC)
    test_sources = _sources(TEST_SRC)
    summary: dict[str, object] = {"javac": javac, "connectors": {}}
    class_dirs: dict[str, Path] = {}
    exit_code = 0
    for version in [v.strip() for v in args.connectors.split(",") if v.strip()]:
        connector = _find_one(jars_dir, f"gcs-connector-{version}*.jar")
        classpath = os.pathsep.join(str(p) for p in [connector, *common])
        main_out = work / f"classes-{version}"
        test_out = work / f"test-classes-{version}"
        main_out.mkdir()
        test_out.mkdir()
        flags = ["--release", "17", "-Xlint:all", "-Werror", "-encoding", "UTF-8"]
        rc = _run(
            [javac, *flags, "-d", str(main_out), "-cp", classpath, *main_sources],
            logs / f"javac-main-{version}.log",
        )
        result: dict[str, object] = {"connector": connector.name, "javac_main_exit": rc}
        if rc == 0:
            class_dirs[version] = main_out
            rc_test = _run(
                [
                    javac,
                    *flags,
                    "-d",
                    str(test_out),
                    "-cp",
                    os.pathsep.join(
                        [str(main_out), str(MAIN_RES), classpath, str(junit)]
                    ),
                    *test_sources,
                ],
                logs / f"javac-test-{version}.log",
            )
            result["javac_test_exit"] = rc_test
            if rc_test == 0 and not args.skip_tests:
                dependency_cp = os.pathsep.join(
                    [classpath, *(str(p) for p in test_runtime)]
                )
                runtime_cp = os.pathsep.join(
                    [str(test_out), str(main_out), str(MAIN_RES), dependency_cp]
                )
                classpath = dependency_cp
                java_bins = [java] + [
                    j.strip() for j in args.extra_test_java.split(",") if j.strip()
                ]
                runs = []
                for idx, java_bin in enumerate(java_bins):
                    log = logs / f"junit-{version}-{idx}.log"
                    rc_run = _run(
                        [
                            java_bin,
                            f"-Dmlflow.cab.test.dependency.classpath={classpath}",
                            f"-Dmlflow.cab.test.provider.dir={main_out}",
                            "-Djava.net.preferIPv4Stack=true",
                            "-cp",
                            os.pathsep.join([str(junit), runtime_cp]),
                            "org.junit.platform.console.ConsoleLauncher",
                            "execute",
                            "--scan-class-path",
                            str(test_out),
                            "--details=summary",
                            "--disable-banner",
                            "--reports-dir",
                            str(work / f"junit-reports-{version}-{idx}"),
                        ],
                        log,
                    )
                    text = log.read_text(encoding="utf-8", errors="replace")
                    found = re.search(r"\[\s*(\d+) tests successful\s*\]", text)
                    failed = re.search(r"\[\s*(\d+) tests failed\s*\]", text)
                    runs.append(
                        {
                            "java": java_bin,
                            "exit": rc_run,
                            "tests_successful": int(found.group(1)) if found else None,
                            "tests_failed": int(failed.group(1)) if failed else None,
                            "log": str(log),
                        }
                    )
                    if rc_run != 0:
                        exit_code = 1
                result["junit_runs"] = runs
            elif rc_test != 0:
                exit_code = 1
        else:
            exit_code = 1
        summary_connectors = summary["connectors"]
        assert isinstance(summary_connectors, dict)
        summary_connectors[version] = result

    if exit_code != 0 or not class_dirs:
        print(summary)
        print(f"Build or tests failed; see logs under {logs}", file=sys.stderr)
        return 1

    # The bytecode must not depend on which connector version it was compiled against.
    reference_version, reference_dir = next(iter(class_dirs.items()))
    for version, directory in class_dirs.items():
        for path in sorted(reference_dir.rglob("*.class")):
            other = directory / path.relative_to(reference_dir)
            if not other.is_file() or other.read_bytes() != path.read_bytes():
                print(
                    f"Bytecode differs between connector {reference_version} and "
                    f"{version}: {path.relative_to(reference_dir)}",
                    file=sys.stderr,
                )
                return 1

    output = Path(args.output)
    names = _write_deterministic_jar(reference_dir, output)
    majors = set()
    with zipfile.ZipFile(output) as jar:
        for name in jar.namelist():
            if name.endswith(".class"):
                majors.add(_class_major(jar.read(name)))
    if not majors or max(majors) > MAX_CLASS_MAJOR:
        print(
            f"Unexpected class file major versions: {sorted(majors)}", file=sys.stderr
        )
        return 1
    javap_log = logs / "javap-verbose.log"
    rc = _run(
        [
            javap,
            "-v",
            "-cp",
            str(output),
            "com.google.cloud.mlflow.gcs.MlflowCabFileSystem",
        ],
        javap_log,
    )
    if rc != 0 or f"major version: {MAX_CLASS_MAJOR}" not in javap_log.read_text(
        "utf-8"
    ):
        print("javap verification failed", file=sys.stderr)
        return 1
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    print(
        f"connectors={sorted(class_dirs)} entries={len(names)} majors={sorted(majors)}"
    )
    print(f"jar={output}")
    print(f"sha256={digest}")
    connectors_summary = summary["connectors"]
    assert isinstance(connectors_summary, dict)
    for version, result in connectors_summary.items():
        print(f"connector {version}: {result}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
