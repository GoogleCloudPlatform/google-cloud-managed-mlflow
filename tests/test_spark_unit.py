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

"""Unit tests for ``mlflow-gs://`` Spark hooks, CAB scopes, and JAR packaging."""

from __future__ import annotations

import base64
import datetime
import hashlib
import importlib
import json
import shutil
import struct
import subprocess
import sys
import time
import zipfile
from pathlib import Path

import mlflow
import pytest
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec, padding, rsa
from mlflow.exceptions import MlflowException

import google_cloud_managed_mlflow
import google_cloud_managed_mlflow.spark.hooks as spark_hooks
from google_cloud_managed_mlflow.spark import (
    ABSTRACT_FILESYSTEM_CLASS,
    CONTROL_SOURCE_CLASS,
    FILESYSTEM_CLASS,
    JAR_NAME,
    bundled_jar_path,
    cab_scope as cab,
    runtime,
    to_cab_uri,
)
from google_cloud_managed_mlflow.uri import parse_gcp_uri
from tests.conftest import ManagedEnv
from tests.fakes.mlflow_server import NAMESPACE_ROOT

REPO_ROOT = Path(__file__).resolve().parent.parent
SERVICES = "META-INF/services"
BLOCK = 256
CHUNK = BLOCK - 2 * 32 - 2 - 2


# ----------------------------------------------- hook installation & JAR packaging


def test_mlflow_spark_hooks_are_installed_and_idempotent() -> None:
    import mlflow.pyfunc
    import mlflow.pyspark.ml
    import mlflow.spark

    for func in (
        mlflow.spark._maybe_save_model,
        mlflow.spark.save_model,
        mlflow.spark.load_model,
        mlflow.pyspark.ml._read_log_model_allowlist,
        mlflow.pyspark.ml.autolog,
        mlflow.pyfunc.load_model,
        mlflow.pyfunc.spark_udf,
    ):
        assert getattr(func, spark_hooks.PATCH_MARKER, False), func
        assert not getattr(func.__wrapped__, spark_hooks.PATCH_MARKER, False)
    spark_hooks.patch_mlflow_spark(mlflow.spark)
    spark_hooks.patch_pyspark_ml(mlflow.pyspark.ml)
    spark_hooks.patch_pyfunc(mlflow.pyfunc)
    assert not getattr(
        mlflow.spark.load_model.__wrapped__, spark_hooks.PATCH_MARKER, False
    )


def test_allowlist_works_without_active_spark_session_and_includes_xgboost(
    managed: ManagedEnv,
) -> None:
    import mlflow.pyspark.ml

    assert runtime.active_session() is None
    allowlist = mlflow.pyspark.ml._read_log_model_allowlist()
    assert "xgboost.spark.*" in allowlist
    assert "pyspark.ml.classification.LogisticRegressionModel" in allowlist
    assert managed.tracking_uri


def test_reimporting_plugin_does_not_duplicate_finders() -> None:
    importlib.reload(google_cloud_managed_mlflow)
    finders = [f for f in sys.meta_path if type(f).__name__ == "_PostImportFinder"]
    assert len(finders) == 1


def test_artifact_registry_registers_mlflow_gs_scheme() -> None:
    from mlflow.store.artifact.artifact_repository_registry import (
        _artifact_repository_registry,
    )

    assert (
        _artifact_repository_registry._registry["mlflow-gs"]
        is spark_hooks.MlflowGsArtifactRepository
    )


def test_to_cab_uri() -> None:
    assert to_cab_uri("gs://b/p/q", "abc") == "mlflow-gs://abc@b/p/q"
    with pytest.raises(ValueError):
        to_cab_uri("s3://b/p", "abc")


def _class_major(data: bytes) -> int:
    magic, _, major = struct.unpack(">IHH", data[:8])
    assert magic == 0xCAFEBABE
    return int(major)


def test_bundled_jar_has_service_metadata_and_java17_bytecode() -> None:
    jar = Path(bundled_jar_path())
    assert jar.name == JAR_NAME and jar.is_file()
    with zipfile.ZipFile(jar) as archive:
        names = set(archive.namelist())
        services = {
            f"{SERVICES}/org.apache.hadoop.fs.FileSystem": FILESYSTEM_CLASS,
            f"{SERVICES}/org.apache.hadoop.fs.AbstractFileSystem": (
                ABSTRACT_FILESYSTEM_CLASS
            ),
            f"{SERVICES}/org.apache.spark.sql.sources.DataSourceRegister": (
                CONTROL_SOURCE_CLASS
            ),
            f"{SERVICES}/org.apache.hadoop.security.token.TokenIdentifier": (
                "com.google.cloud.mlflow.gcs.MlflowCabTokenIdentifier"
            ),
        }
        for entry, cls in services.items():
            assert entry in names
            lines = [ln.strip() for ln in archive.read(entry).decode().splitlines()]
            assert cls in lines
            assert cls.replace(".", "/") + ".class" in names
        majors = {
            _class_major(archive.read(name))
            for name in names
            if name.endswith(".class")
        }
        assert majors == {61}
        for required in (
            "MlflowCabAccessTokenProvider",
            "MlflowCabDelegationTokenBinding",
            "MlflowCabSparkPlugin",
            "MlflowCabDriverControlSource",
        ):
            assert f"com/google/cloud/mlflow/gcs/{required}.class" in names


def test_gitignore_does_not_ignore_bundled_jar() -> None:
    if not (REPO_ROOT / ".git").exists() or shutil.which("git") is None:
        pytest.skip("not a git checkout")
    jar = Path(bundled_jar_path()).relative_to(REPO_ROOT)
    result = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "check-ignore", "-q", str(jar)], check=False
    )
    assert result.returncode == 1, "the bundled JAR must not be git-ignored"


@pytest.fixture(scope="module")
def built_wheel(tmp_path_factory: pytest.TempPathFactory) -> Path:
    if not (REPO_ROOT / "pyproject.toml").is_file():
        pytest.skip("source tree not available")
    tmp_path = tmp_path_factory.mktemp("wheel")
    source = tmp_path / "src"
    source.mkdir()
    for name in ("pyproject.toml", "README.md", "LICENSE"):
        if (REPO_ROOT / name).exists():
            shutil.copy(REPO_ROOT / name, source / name)
    shutil.copytree(
        REPO_ROOT / "google_cloud_managed_mlflow",
        source / "google_cloud_managed_mlflow",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    out = tmp_path / "dist"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "wheel",
            "--no-deps",
            "--no-build-isolation",
            "--no-index",
            "-q",
            "-w",
            str(out),
            str(source),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0 and "No module named pip" in result.stderr:
        pytest.skip("pip is not available")
    assert result.returncode == 0, result.stderr[-2000:]
    wheels = list(out.glob("google_cloud_managed_mlflow-*.whl"))
    assert len(wheels) == 1
    return wheels[0]


def test_wheel_bundles_identical_jar(built_wheel: Path) -> None:
    with zipfile.ZipFile(built_wheel) as archive:
        member = f"google_cloud_managed_mlflow/jars/{JAR_NAME}"
        assert member in archive.namelist()
        assert "google_cloud_managed_mlflow/spark/_udf_worker.py" in archive.namelist()
        assert "google_cloud_managed_mlflow/py.typed" in archive.namelist()
        bundled = hashlib.sha256(archive.read(member)).hexdigest()
        entry_points = archive.read(
            next(n for n in archive.namelist() if n.endswith("entry_points.txt"))
        ).decode()
    assert bundled == hashlib.sha256(Path(bundled_jar_path()).read_bytes()).hexdigest()
    assert (
        "google-cloud-auth = google_cloud_managed_mlflow.auth:GoogleCloudAuthProvider"
        in entry_points
    )
    assert (
        "mlflow-gs = google_cloud_managed_mlflow.spark.hooks:MlflowGsArtifactRepository"
        in (entry_points)
    )
    assert mlflow.__version__


def test_bundled_jar_path_extracts_from_zipimported_wheel(
    built_wheel: Path, tmp_path: Path
) -> None:
    script = "\n".join(
        [
            "import hashlib, json, os, stat, sys",
            "sys.path.insert(0, sys.argv[1])",
            "import google_cloud_managed_mlflow.spark as spark_pkg",
            "path = spark_pkg.bundled_jar_path()",
            "with open(path, 'rb') as handle:",
            "    digest = hashlib.sha256(handle.read()).hexdigest()",
            "print(json.dumps({",
            "    'module': spark_pkg.__file__,",
            "    'path': path,",
            "    'again': spark_pkg.bundled_jar_path(),",
            "    'mode': stat.S_IMODE(os.stat(os.path.dirname(path)).st_mode),",
            "    'sha256': digest,",
            "}))",
        ]
    )
    env = {"TMPDIR": str(tmp_path), "PYTHONDONTWRITEBYTECODE": "1"}
    completed = subprocess.run(
        [sys.executable, "-c", script, str(built_wheel)],
        capture_output=True,
        text=True,
        check=False,
        cwd=tmp_path,
        env=env,
    )
    assert completed.returncode == 0, completed.stderr[-3000:]
    result = json.loads(completed.stdout.strip().splitlines()[-1])
    assert ".whl" in result["module"]
    assert result["path"] == result["again"]
    assert Path(result["path"]).name == JAR_NAME
    assert Path(result["path"]).parent.parent == tmp_path
    assert result["mode"] == 0o700
    expected = hashlib.sha256(Path(bundled_jar_path()).read_bytes()).hexdigest()
    assert result["sha256"] == expected
    assert not Path(result["path"]).parent.exists()


# ----------------------------------------------------------- CAB scopes & issuance


def _namespace(env: ManagedEnv) -> cab.CabScope:
    resource = parse_gcp_uri(env.tracking_uri)
    return cab.namespace_scope(
        resource,
        write=False,
        artifact_uri=f"{NAMESPACE_ROOT}/0/run/artifacts",
    )


def test_namespace_and_registered_model_scope_bodies() -> None:
    ns = parse_gcp_uri(
        "https://us-central1-mlflow.googleapis.com/v1alpha/projects/p/locations/us-central1/namespaces/n"
    )
    reg = parse_gcp_uri(
        "https://us-central1-mlflow.googleapis.com/v1alpha/projects/p/locations/us-central1/registries/r"
    )
    read = cab.namespace_scope(ns, write=False, run_id="r1", artifact_path="/model/")
    write = cab.namespace_scope(ns, write=True, artifact_uri="gs://b/x/")
    assert read.issue_url.endswith("namespaces/n:issueReadArtifactToken")
    assert json.loads(read.issue_body) == {"runId": "r1", "artifactPath": "model"}
    assert read.mode == cab.READ_ONLY and not read.is_write
    assert write.issue_url.endswith(":issueReadWriteArtifactToken")
    assert json.loads(write.issue_body) == {"artifactUri": "gs://b/x"}
    assert write.mode == cab.READ_WRITE and write.is_write and not write.is_create_only
    assert read.scope_id != write.scope_id and len(read.scope_id) == 24

    model_write = cab.registered_model_scope(
        reg,
        "projects/p/locations/us-central1/registries/r/registeredModels/m",
        write=True,
        artifact_uri="gs://reg/registeredModels/u/versions/1/",
    )
    assert json.loads(model_write.issue_body) == {
        "artifactUri": "gs://reg/registeredModels/u/versions/1"
    }
    assert model_write.mode == cab.READ_CREATE and model_write.is_create_only


def test_namespace_scope_rejects_multiple_targets_and_traversal() -> None:
    ns = parse_gcp_uri(
        "https://us-central1-mlflow.googleapis.com/v1alpha/projects/p/locations/us-central1/namespaces/n"
    )
    with pytest.raises(MlflowException, match="only one of"):
        cab.namespace_scope(ns, write=False, run_id="r", model_id="m")
    with pytest.raises(MlflowException, match="Invalid artifact_path"):
        cab.namespace_scope(ns, write=False, run_id="r", artifact_path="a/../b")


def test_parse_token_response_and_redact() -> None:
    camel = cab.parse_token_response(
        {
            "artifactUri": "gs://b/p",
            "credentials": {
                "accessToken": "ya29." + "a",
                "expireTime": "2030-01-01T00:00:00.123456789Z",
            },
        }
    )
    snake = cab.parse_token_response(
        {
            "artifact_uri": "gs://b/p",
            "credentials": {
                "access_token": "ya29." + "b",
                "expire_time": "2030-01-01T00:00:00Z",
            },
        }
    )
    assert camel.token == "ya29." + "a" and snake.token == "ya29." + "b"
    assert camel.artifact_uri == snake.artifact_uri == "gs://b/p"
    assert camel.expiry == datetime.datetime(2030, 1, 1, 0, 0, 0, 123456)
    assert "ya29" not in repr(camel)
    text = (
        "Bearer ya29." + 'secret-value {"access_token": "abc", '
        '"accessToken":"def", "x": "y"}'
    )
    redacted = cab.redact(text)
    assert "ya29." + "secret" not in redacted and "abc" not in redacted


def test_issue_token_and_shared_credentials_refresh(
    managed: ManagedEnv, monkeypatch: pytest.MonkeyPatch
) -> None:
    managed.server.cab_failures.extend([500, 503])
    sleeps: list[float] = []
    scope = _namespace(managed)
    token = cab.issue_token(scope, sleep=sleeps.append)
    assert token.token.startswith("ya29." + "cab-read_only-python-client-")
    assert len(sleeps) == 2

    managed.server.token_ttl_seconds = 4.0
    monkeypatch.setenv(cab.REFRESH_MARGIN_ENV, "2.5")
    cab.reset_shared_credentials()
    credentials = cab.CabCredentials.shared(scope)
    first = credentials.current_token()
    rotations = {first.token}
    deadline = time.time() + 12
    while len(rotations) < 3 and time.time() < deadline:
        time.sleep(0.25)
        rotations.add(credentials.current_token().token)
    assert len(rotations) >= 3
    assert "ya29" not in repr(credentials)


# ---------------------------------------------------------- RSA-OAEP token wrapping


def _key_pair() -> tuple[rsa.RSAPrivateKey, str]:
    private = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    spki = private.public_key().public_bytes(
        serialization.Encoding.DER, serialization.PublicFormat.SubjectPublicKeyInfo
    )
    return private, base64.b64encode(spki).decode("ascii")


def _oaep(mgf_hash: hashes.HashAlgorithm) -> padding.OAEP:
    return padding.OAEP(
        mgf=padding.MGF1(algorithm=mgf_hash), algorithm=hashes.SHA256(), label=None
    )


def _unwrap(private: rsa.RSAPrivateKey, wrapped: str) -> str:
    data = base64.b64decode(wrapped)
    assert data and len(data) % BLOCK == 0
    blocks = [data[i : i + BLOCK] for i in range(0, len(data), BLOCK)]
    parts = []
    for index, block in enumerate(blocks):
        plain = private.decrypt(block, _oaep(hashes.SHA256()))
        assert plain[0] == index and plain[1] == len(blocks)
        parts.append(plain[2:])
    return b"".join(parts).decode("utf-8")


@pytest.mark.parametrize("length", [1, 40, CHUNK, CHUNK + 1, 1500])
def test_wrapped_tokens_decrypt_with_standard_oaep_sha256(length: int) -> None:
    private, public = _key_pair()
    token = "ya29." + "t" * length
    wrapped = runtime.wrap_client_token(public, token)
    assert "ya29." not in wrapped
    assert _unwrap(private, wrapped) == token


def test_non_rsa_and_malformed_keys_are_rejected() -> None:
    ec_spki = (
        ec.generate_private_key(ec.SECP256R1())
        .public_key()
        .public_bytes(
            serialization.Encoding.DER,
            serialization.PublicFormat.SubjectPublicKeyInfo,
        )
    )
    with pytest.raises(ValueError, match="not an RSA public key"):
        runtime.parse_rsa_public_key(ec_spki)
