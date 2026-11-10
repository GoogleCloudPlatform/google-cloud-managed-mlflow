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

"""Driver wiring for the CAB provider across Spark Connect and classic Spark.

All driver interactions go through the provider's DataSource V2 control source
(``spark.read.format(...).load().schema``), which runs on the driver during
analysis without launching Spark jobs and works identically over Spark Connect
(no JVM gateway) and classic PySpark.
"""

from __future__ import annotations

import base64
import datetime
import hashlib
import io
import json
import logging
import os
import threading
import time
import uuid
import weakref
import zipfile
from typing import Any

from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import INTERNAL_ERROR

from google_cloud_managed_mlflow.spark import (
    ABSTRACT_FILESYSTEM_CLASS,
    CONTROL_SOURCE_CLASS,
    FILESYSTEM_CLASS,
    bundled_jar_path,
    cab_scope as cab,
)

_logger = logging.getLogger(__name__)

RESULT_METADATA_KEY = "mlflow_cab_result"
SUPPLIER_CONF = "spark.mlflow.cab.token.supplier"
SUPPLIER_ENV = "MLFLOW_CAB_TOKEN_SUPPLIER"
SUPPLIERS = ("auto", "cluster", "client")
_SESSION_STATE: weakref.WeakKeyDictionary[Any, dict[str, Any]] = (
    weakref.WeakKeyDictionary()
)
_SESSION_LOCK = threading.RLock()


class ProviderNotLoadedError(MlflowException):
    """Raised when the provider JAR is not visible to the Spark driver."""


# ---------------------------------------------------------- client token wrapping
#
# Client-minted tokens are pushed to the driver only as RSA-OAEP ciphertext for
# the driver JVM's ephemeral public key (SHA-256 digest, MGF1-SHA-256, empty
# label), so plaintext bearer tokens never appear in data source options, plan
# strings, Spark Connect requests, listener events, or logs.

_RSA_ENCRYPTION_OID = bytes.fromhex("2a864886f70d010101")
_SHA256_BYTES = 32
_WRAP_HEADER_BYTES = 2
_WRAP_MAX_BLOCKS = 255


def _der_element(data: bytes, offset: int, tag: int) -> tuple[bytes, int]:
    """Return the value of the DER element with ``tag`` at ``offset`` and its end."""
    if offset + 2 > len(data) or data[offset] != tag:
        raise ValueError("unexpected DER structure")
    length = data[offset + 1]
    offset += 2
    if length & 0x80:
        count = length & 0x7F
        if count == 0 or count > 4 or offset + count > len(data):
            raise ValueError("invalid DER length")
        length = int.from_bytes(data[offset : offset + count], "big")
        offset += count
    end = offset + length
    if end > len(data):
        raise ValueError("truncated DER element")
    return data[offset:end], end


def parse_rsa_public_key(spki_der: bytes) -> tuple[int, int]:
    """Return ``(modulus, exponent)`` of an X.509 SubjectPublicKeyInfo RSA key."""
    spki, _ = _der_element(spki_der, 0, 0x30)
    algorithm, offset = _der_element(spki, 0, 0x30)
    oid, _ = _der_element(algorithm, 0, 0x06)
    if oid != _RSA_ENCRYPTION_OID:
        raise ValueError("not an RSA public key")
    bits, _ = _der_element(spki, offset, 0x03)
    if not bits or bits[0] != 0:
        raise ValueError("invalid RSA public key bit string")
    rsa_key, _ = _der_element(bits[1:], 0, 0x30)
    modulus, offset = _der_element(rsa_key, 0, 0x02)
    exponent, _ = _der_element(rsa_key, offset, 0x02)
    return int.from_bytes(modulus, "big"), int.from_bytes(exponent, "big")


def _mgf1_sha256(seed: bytes, length: int) -> bytes:
    output = b""
    counter = 0
    while len(output) < length:
        output += hashlib.sha256(seed + counter.to_bytes(4, "big")).digest()
        counter += 1
    return output[:length]


def rsa_oaep_sha256_encrypt(modulus: int, exponent: int, message: bytes) -> bytes:
    """RSAES-OAEP encryption (RFC 8017) with SHA-256, MGF1-SHA-256, empty label."""
    size = (modulus.bit_length() + 7) // 8
    if len(message) > size - 2 * _SHA256_BYTES - 2:
        raise ValueError("message too long for RSA-OAEP")
    block = (
        hashlib.sha256(b"").digest()
        + b"\x00" * (size - len(message) - 2 * _SHA256_BYTES - 2)
        + b"\x01"
        + message
    )
    seed = os.urandom(_SHA256_BYTES)
    db_mask = _mgf1_sha256(seed, size - _SHA256_BYTES - 1)
    masked_block = bytes(a ^ b for a, b in zip(block, db_mask, strict=True))
    seed_mask = _mgf1_sha256(masked_block, _SHA256_BYTES)
    masked_seed = bytes(a ^ b for a, b in zip(seed, seed_mask, strict=True))
    encoded = int.from_bytes(b"\x00" + masked_seed + masked_block, "big")
    return pow(encoded, exponent, modulus).to_bytes(size, "big")


def wrap_client_token(public_key_b64: str, token: str) -> str:
    """Encrypt ``token`` for the driver's base64 X.509 ``clientWrapKey``."""
    modulus, exponent = parse_rsa_public_key(base64.b64decode(public_key_b64))
    size = (modulus.bit_length() + 7) // 8
    chunk = size - 2 * _SHA256_BYTES - 2 - _WRAP_HEADER_BYTES
    data = token.encode("utf-8")
    blocks = max(1, -(-len(data) // chunk))
    if blocks > _WRAP_MAX_BLOCKS:
        raise ValueError("token too long to wrap")
    ciphertext = b"".join(
        rsa_oaep_sha256_encrypt(
            modulus,
            exponent,
            bytes((index, blocks)) + data[index * chunk : (index + 1) * chunk],
        )
        for index in range(blocks)
    )
    return base64.b64encode(ciphertext).decode("ascii")


def is_spark_connect(spark: Any) -> bool:
    """Return True for Spark Connect sessions (no JVM gateway on the client)."""
    try:
        from pyspark.sql.connect.session import SparkSession as ConnectSession
    except ImportError:
        return False
    return isinstance(spark, ConnectSession)


def active_session() -> Any:
    """Return the active (classic or Spark Connect) session, or None."""
    try:
        from pyspark.sql import SparkSession
    except ImportError:
        return None
    try:
        session = SparkSession.getActiveSession()
        if session is not None:
            return session
    except Exception:  # pragma: no cover - defensive
        pass
    with _SESSION_LOCK:
        known = list(_SESSION_STATE)
    for session in reversed(known):
        try:
            if is_spark_connect(session):
                if not getattr(session.client, "is_closed", True):
                    return session
            elif getattr(session, "_jsc", None) is not None:
                return session
        except Exception:  # pragma: no cover
            continue
    return None


def _state(spark: Any) -> dict[str, Any]:
    with _SESSION_LOCK:
        try:
            state = _SESSION_STATE.get(spark)
        except TypeError:  # pragma: no cover - unhashable sessions
            state = None
        if state is None:
            state = {"scopes": {}, "lock": threading.RLock()}
            try:
                _SESSION_STATE[spark] = state
            except TypeError:  # pragma: no cover
                pass
        return state


def control(spark: Any, action: str, **options: Any) -> dict[str, str]:
    """Run a driver-side provider action and return its (secret-free) result."""
    opts = {k: str(v) for k, v in options.items() if v is not None}
    opts["action"] = action
    opts["nonce"] = uuid.uuid4().hex
    try:
        schema = spark.read.format(CONTROL_SOURCE_CLASS).options(**opts).load().schema
    except Exception as exc:
        text = str(exc)
        if (
            "DATA_SOURCE_NOT_FOUND" in text
            or "ClassNotFoundException" in text
            or "Failed to find the data source" in text
            or "MlflowCabDriverControlSource" in text
        ):
            raise ProviderNotLoadedError(
                "The mlflow-gs provider JAR is not on the Spark driver classpath.",
                error_code=INTERNAL_ERROR,
            ) from None
        raise MlflowException(
            f"MLflow CAB driver action '{action}' failed: {cab.redact(text)[:500]}",
            error_code=INTERNAL_ERROR,
        ) from None
    field = schema.fields[0]
    metadata = field.metadata or {}
    raw = metadata.get(RESULT_METADATA_KEY)
    if not isinstance(raw, str):
        raise MlflowException(
            "MLflow CAB driver action returned no result metadata.",
            error_code=INTERNAL_ERROR,
        )
    result: dict[str, str] = {
        str(k): "" if v is None else str(v) for k, v in json.loads(raw).items()
    }
    return result


def _add_jar(spark: Any) -> str:
    jar = bundled_jar_path()
    if not os.path.isfile(jar):
        raise MlflowException(
            f"Bundled provider JAR not found at {jar}.", error_code=INTERNAL_ERROR
        )
    if is_spark_connect(spark):
        spark.addArtifacts(jar)
        return "add_artifact"
    sc = spark.sparkContext
    sc.addJar(jar)  # executors
    jvm = sc._jvm
    url = jvm.java.io.File(jar).toURI().toURL()
    loader = jvm.java.lang.Thread.currentThread().getContextClassLoader()
    try:
        loader.addURL(url)  # MutableURLClassLoader on the driver (spark-submit client)
    except Exception:
        array = sc._gateway.new_array(jvm.java.net.URL, 1)
        array[0] = url
        child = jvm.java.net.URLClassLoader(array, loader)
        jvm.java.lang.Thread.currentThread().setContextClassLoader(child)
    return "add_jar"


def _forward_operator_settings(spark: Any) -> None:
    settings = {
        "spark.hadoop.fs.mlflow-gs.impl": FILESYSTEM_CLASS,
        "spark.hadoop.fs.AbstractFileSystem.mlflow-gs.impl": ABSTRACT_FILESYSTEM_CLASS,
    }
    margin = os.environ.get(cab.REFRESH_MARGIN_ENV, "").strip()
    if margin:
        settings["spark.hadoop.fs.mlflow.cab.refresh.margin.seconds"] = margin
    override = os.environ.get("MLFLOW_GCP_ENDPOINT_OVERRIDE", "").strip()
    if override:
        settings["spark.hadoop.fs.mlflow.cab.endpoint.override"] = override
    for key, value in settings.items():
        try:
            if spark.conf.get(key, None) != value:
                spark.conf.set(key, value)
        except Exception:  # pragma: no cover - static/immutable conf
            _logger.debug("Unable to set %s", key, exc_info=True)


def ensure_provider(spark: Any) -> str:
    """Make the provider available on the driver and executors (idempotent).

    Returns how the provider was found: ``preinstalled`` (static classpath,
    ``--jars``, or ``spark.jars``), ``add_artifact`` (Spark Connect upload), or
    ``add_jar`` (classic dynamic wiring).
    """
    state = _state(spark)
    with state["lock"]:
        mode = state.get("provider_mode")
        if mode:
            return str(mode)
        try:
            result = control(spark, "install")
            mode = "preinstalled"
        except ProviderNotLoadedError:
            mode = _add_jar(spark)
            result = control(spark, "install")
        if result.get("ok") != "true":
            raise MlflowException(
                f"MLflow CAB provider installation failed: {result.get('error', '')}",
                error_code=INTERNAL_ERROR,
            )
        _forward_operator_settings(spark)
        state["provider_mode"] = mode
        state["install_result"] = result
        return mode


def _configured_supplier(spark: Any) -> str:
    """``auto``, ``cluster``, or ``client`` from the environment or session conf.

    Spark Connect redacts configuration values whose keys contain ``token``, so
    Spark Connect clients cannot read ``spark.mlflow.cab.token.supplier``; they
    set ``MLFLOW_CAB_TOKEN_SUPPLIER`` in the client environment instead (it
    takes precedence everywhere).
    """
    value = os.environ.get(SUPPLIER_ENV, "").strip()
    if not value:
        try:
            value = str(spark.conf.get(SUPPLIER_CONF, None) or "").strip()
        except Exception:  # pragma: no cover - unreadable conf
            value = ""
    supplier = value.lower() or "auto"
    if supplier not in SUPPLIERS:
        _logger.warning(
            "Unsupported CAB token supplier %r (expected one of %s); using 'auto'.",
            value,
            ", ".join(SUPPLIERS),
        )
        return "auto"
    return supplier


def ensure_scope(spark: Any, scope: cab.CabScope) -> str:
    """Register ``scope`` on the driver (one CAB RPC per scope) and return its id.

    The driver mints with its own identity (``cluster``). If that fails and the
    client can mint (for example a Spark Connect client whose cluster service
    account lacks MLflow IAM), the client pushes downscoped tokens to the driver
    (``client``) and keeps renewing them in the background.
    """
    ensure_provider(spark)
    state = _state(spark)
    scopes: dict[str, dict[str, Any]] = state["scopes"]
    supplier = _configured_supplier(spark)
    margin = cab.refresh_margin_seconds()
    override = os.environ.get("MLFLOW_GCP_ENDPOINT_OVERRIDE", "").strip() or None
    base: dict[str, Any] = {
        "scopeId": scope.scope_id,
        "issueUrl": scope.issue_url,
        "issueBody": scope.issue_body,
        "mode": scope.mode,
        "boundary": scope.boundary,
        "refreshMarginSeconds": (
            int(margin) if os.environ.get(cab.REFRESH_MARGIN_ENV) else None
        ),
        "endpointOverride": override,
    }
    with state["lock"]:
        known = scopes.get(scope.scope_id)
        if known is not None and known.get("supplier") == "client":
            _push_client_token(spark, scope, base, known)
            return scope.scope_id
        if supplier in ("auto", "cluster"):
            result = control(spark, "ensure", supplier="cluster", **base)
            if result.get("ok") == "true":
                scopes[scope.scope_id] = {"supplier": "cluster", "result": result}
                return scope.scope_id
            if supplier == "cluster":
                raise MlflowException(
                    f"Driver failed to mint CAB token: {result.get('error', '')}",
                    error_code=INTERNAL_ERROR,
                )
            _logger.info(
                "Driver could not mint CAB token (%s); pushing client-minted tokens.",
                result.get("error", ""),
            )
        entry: dict[str, Any] = {"supplier": "client"}
        scopes[scope.scope_id] = entry
        _push_client_token(spark, scope, base, entry)
        _start_client_renewer(spark, scope, base, entry)
        return scope.scope_id


def _client_wrap_key(spark: Any, refresh: bool = False) -> str:
    """The driver JVM's public token wrapping key (cached per session)."""
    state = _state(spark)
    key = None
    if not refresh:
        install = state.get("install_result") or {}
        key = state.get("wrap_key") or install.get("clientWrapKey")
    if not key:
        key = control(spark, "probe").get("clientWrapKey")
    if not key:
        raise MlflowException(
            "The Spark driver did not publish a client token wrapping key; upgrade "
            "the mlflow-gs provider JAR on the cluster.",
            error_code=INTERNAL_ERROR,
        )
    state["wrap_key"] = key
    return str(key)


def _push_client_token(
    spark: Any, scope: cab.CabScope, base: dict[str, Any], entry: dict[str, Any]
) -> None:
    credentials = cab.CabCredentials.shared(scope)
    token = credentials.current_token()
    if entry.get("pushed_expiry") == token.expires_at_ms:
        return
    result: dict[str, str] = {}
    for refresh in (False, True):
        wrapped = wrap_client_token(_client_wrap_key(spark, refresh), token.token)
        result = control(
            spark,
            "ensure",
            supplier="client",
            tokenSecret=wrapped,
            expireTimeMs=token.expires_at_ms,
            **base,
        )
        if result.get("ok") == "true" or "unwrap" not in result.get("error", ""):
            break
    if result.get("ok") != "true":
        raise MlflowException(
            "Failed to publish client CAB token to the driver: "
            f"{result.get('error', '')}",
            error_code=INTERNAL_ERROR,
        )
    entry["pushed_expiry"] = token.expires_at_ms
    entry["artifact_uri"] = token.artifact_uri
    entry["result"] = result


def _start_client_renewer(
    spark: Any, scope: cab.CabScope, base: dict[str, Any], entry: dict[str, Any]
) -> None:
    if entry.get("renewer") is not None:
        return
    session_ref = weakref.ref(spark)
    stop = threading.Event()

    def _run() -> None:
        failures = 0
        while failures < 10:
            credentials = cab.CabCredentials.shared(scope)
            expiry = credentials.expiry
            margin = cab.refresh_margin_seconds()
            delay = 1.0
            if expiry is not None:
                expires_at = expiry.replace(tzinfo=datetime.timezone.utc).timestamp()
                delay = min(60.0, max(0.5, expires_at - time.time() - margin))
            if stop.wait(delay):
                return
            session = session_ref()
            if session is None:
                return
            try:
                with _state(session)["lock"]:
                    if stop.is_set():
                        return
                    _push_client_token(session, scope, base, entry)
                failures = 0
            except Exception:
                failures += 1
                _logger.debug("Client CAB token renewal failed", exc_info=True)

    thread = threading.Thread(
        target=_run, name=f"mlflow-cab-client-renewer-{scope.scope_id}", daemon=True
    )
    entry["renewer"] = thread
    entry["stop"] = stop
    thread.start()


def scope_artifact_uri(spark: Any, scope: cab.CabScope) -> str:
    """Canonical ``gs://`` prefix the issuance RPC resolved for ``scope``."""
    entry = _state(spark)["scopes"].get(scope.scope_id) or {}
    result = entry.get("result") or {}
    uri = str(result.get("artifactUri") or entry.get("artifact_uri") or "")
    if not uri.startswith("gs://"):
        raise MlflowException(
            "The CAB issuance response did not include the resolved artifact URI.",
            error_code=INTERNAL_ERROR,
        )
    return uri.rstrip("/")


def release_scope(spark: Any, scope: cab.CabScope) -> None:
    """Stop renewing ``scope`` on the driver and in this client (best effort)."""
    state = _state(spark)
    with state["lock"]:
        entry = state["scopes"].pop(scope.scope_id, None)
    if entry is not None and entry.get("stop") is not None:
        entry["stop"].set()
    control(spark, "release", scopeId=scope.scope_id)


def driver_stats(spark: Any, scope_id: str | None = None) -> dict[str, str]:
    """Return driver-side token telemetry (no secrets)."""
    ensure_provider(spark)
    return control(spark, "stats", scopeId=scope_id)


def write_directory(
    spark: Any, cab_uri: str, local_dir: str, *, rel_prefix: str = ""
) -> None:
    """Upload all files under ``local_dir`` to ``cab_uri`` via the Spark driver."""
    ensure_provider(spark)
    buf = io.BytesIO()
    prefix = rel_prefix.strip("/")
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for root, _, files in os.walk(local_dir):
            for name in sorted(files):
                full = os.path.join(root, name)
                rel = os.path.relpath(full, local_dir).replace(os.sep, "/")
                arcname = f"{prefix}/{rel}" if prefix else rel
                zf.write(full, arcname=arcname)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    result = control(spark, "write_archive", dst=cab_uri, archive=b64)
    if result.get("ok") != "true":
        raise MlflowException(
            f"Driver write_archive to {cab_uri} failed: {result.get('error', '')}",
            error_code=INTERNAL_ERROR,
        )


def write_single_file(
    spark: Any, cab_uri: str, local_file: str, *, rel_path: str
) -> None:
    """Upload ``local_file`` to ``{cab_uri}/{rel_path}`` via the Spark driver."""
    ensure_provider(spark)
    buf = io.BytesIO()
    arcname = rel_path.strip("/") or os.path.basename(local_file)
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(local_file, arcname=arcname)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    result = control(spark, "write_archive", dst=cab_uri, archive=b64)
    if result.get("ok") != "true":
        raise MlflowException(
            f"Driver write_archive to {cab_uri} failed: {result.get('error', '')}",
            error_code=INTERNAL_ERROR,
        )


def read_to_directory(spark: Any, cab_uri: str, dst_dir: str) -> tuple[bool, list[str]]:
    """Download ``cab_uri`` into ``dst_dir`` via the Spark driver.

    Returns ``(is_directory, extracted_relative_paths)``.
    """
    ensure_provider(spark)
    result = control(spark, "read_archive", src=cab_uri)
    if result.get("ok") != "true":
        raise MlflowException(
            f"Driver read_archive from {cab_uri} failed: {result.get('error', '')}",
            error_code=INTERNAL_ERROR,
        )
    is_dir = result.get("isDirectory") == "true"
    raw = base64.b64decode(result.get("archive", ""))
    os.makedirs(dst_dir, exist_ok=True)
    extracted: list[str] = []
    with zipfile.ZipFile(io.BytesIO(raw), "r") as zf:
        for member in zf.infolist():
            name = member.filename.replace("\\", "/").strip("/")
            if not name or any(part in ("", ".", "..") for part in name.split("/")):
                raise MlflowException(
                    f"Unsafe archive entry path {member.filename!r}",
                    error_code=INTERNAL_ERROR,
                )
            if member.is_dir():
                continue
            target = os.path.join(dst_dir, *name.split("/"))
            os.makedirs(os.path.dirname(target), exist_ok=True)
            with zf.open(member, "r") as src_f, open(target, "wb") as dst_f:
                dst_f.write(src_f.read())
            extracted.append(name)
    return is_dir, extracted


def list_directory(spark: Any, cab_uri: str) -> list[dict[str, Any]]:
    """List direct children of ``cab_uri`` via the Spark driver."""
    ensure_provider(spark)
    result = control(spark, "list_status", src=cab_uri)
    if result.get("ok") != "true":
        raise MlflowException(
            f"Driver list_status on {cab_uri} failed: {result.get('error', '')}",
            error_code=INTERNAL_ERROR,
        )
    entries = json.loads(result.get("entries", "[]"))
    if not isinstance(entries, list):
        return []
    return [item for item in entries if isinstance(item, dict)]


def delete_path(spark: Any, cab_uri: str) -> bool:
    """Recursively delete ``cab_uri`` via the Spark driver."""
    ensure_provider(spark)
    result = control(spark, "delete", target=cab_uri)
    if result.get("ok") != "true":
        raise MlflowException(
            f"Driver delete on {cab_uri} failed: {result.get('error', '')}",
            error_code=INTERNAL_ERROR,
        )
    return result.get("deleted") == "true"
