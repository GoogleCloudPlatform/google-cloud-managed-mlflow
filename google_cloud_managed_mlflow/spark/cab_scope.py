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

"""CAB scope descriptors and client fallback credentials for ``mlflow-gs://``.

Normally the Spark driver JVM mints and renews downscoped tokens through
``MlflowCabTokenService`` using the cluster's identity. When the driver cannot
mint (for example on Spark Connect where only the client holds MLflow IAM), the
client mints via :class:`CabCredentials`, wraps the token with the driver's
ephemeral RSA-OAEP public key, and pushes it to the driver.
"""

from __future__ import annotations

import collections
import dataclasses
import datetime
import email.utils
import hashlib
import json
import logging
import os
import random
import re
import threading
import time
import urllib.parse
from typing import Any

import google.auth.credentials
import requests
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import (
    INTERNAL_ERROR,
    INVALID_PARAMETER_VALUE,
    PERMISSION_DENIED,
    RESOURCE_DOES_NOT_EXIST,
    UNAUTHENTICATED,
)

from google_cloud_managed_mlflow.auth import (
    GoogleCloudBearerAuth,
    create_secure_session,
    validate_transport_security,
)
from google_cloud_managed_mlflow.uri import (
    ParsedGcpUri,
    build_gcp_api_url,
    build_gcp_resource_url,
    parse_rfc3339_utc,
)

_logger = logging.getLogger(__name__)

REFRESH_MARGIN_ENV = "MLFLOW_CAB_REFRESH_MARGIN_SECONDS"
DEFAULT_REFRESH_MARGIN_SECONDS = 300.0
READ_ONLY = "READ_ONLY"
READ_WRITE = "READ_WRITE"
READ_CREATE = "READ_CREATE"
_RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 504})
_MAX_ATTEMPTS = 6
_INITIAL_BACKOFF_SECONDS = 0.25
_MAX_BACKOFF_SECONDS = 8.0
_MIN_REFRESH_SPACING_SECONDS = 1.0
_TOKEN_RE = re.compile(r"ya29\.[A-Za-z0-9._\-~+/=]*")
_SECRET_FIELD_RE = re.compile(
    r'(?i)("(?:access_?token|refresh_?token|token)"\s*:\s*)"[^"]*"'
)
_REGISTERED_MODEL_PATH_RE = re.compile(
    r"^gs://[^/]+/(?:.+/)?registeredModels/(?P<model_id>[A-Za-z0-9._-]+)(?:/|$)"
)

CAB_RPC_COUNTS: collections.Counter[str] = collections.Counter()
_COUNTS_LOCK = threading.Lock()
_VAULT_LOCK = threading.Lock()
_VAULT_PREFIXES: dict[str, tuple[str, ParsedGcpUri]] = {}


def redact(text: str) -> str:
    """Remove OAuth tokens and secret JSON fields from ``text``."""
    return _SECRET_FIELD_RE.sub(r'\1"<redacted>"', _TOKEN_RE.sub("<redacted>", text))


def refresh_margin_seconds() -> float:
    """Return the renewal lead window from ``MLFLOW_CAB_REFRESH_MARGIN_SECONDS``."""
    raw = os.environ.get(REFRESH_MARGIN_ENV, "").strip()
    if not raw:
        return DEFAULT_REFRESH_MARGIN_SECONDS
    try:
        value = float(raw)
    except ValueError:
        _logger.warning("Ignoring invalid %s=%r", REFRESH_MARGIN_ENV, raw)
        return DEFAULT_REFRESH_MARGIN_SECONDS
    return max(0.0, value)


def scope_id_for(issue_url: str, issue_body: str) -> str:
    """Stable scope identifier shared with the Java provider (SHA-256, 24 hex)."""
    digest = hashlib.sha256(f"{issue_url}\n{issue_body}".encode()).hexdigest()
    return digest[:24]


@dataclasses.dataclass(frozen=True)
class CabScope:
    """Non-secret description of one CAB token request."""

    issue_url: str
    issue_body: str
    mode: str
    boundary: str
    resource_name: str

    @property
    def scope_id(self) -> str:
        """Stable identifier used in ``mlflow-gs://<scope>@<bucket>`` paths."""
        return scope_id_for(self.issue_url, self.issue_body)

    @property
    def is_write(self) -> bool:
        """Return True for scopes that allow object creation."""
        return self.mode in (READ_WRITE, READ_CREATE)

    @property
    def is_create_only(self) -> bool:
        """Return True for RegisteredModel Vault scopes (no overwrite/delete)."""
        return self.mode == READ_CREATE


def _encode_body(body: dict[str, str]) -> str:
    return json.dumps(body, sort_keys=True, separators=(",", ":"))


def namespace_scope(
    namespace: ParsedGcpUri,
    *,
    write: bool,
    artifact_uri: str | None = None,
    run_id: str | None = None,
    model_id: str | None = None,
    artifact_path: str | None = None,
) -> CabScope:
    """Build a Namespace scope; at most one of artifact_uri/run_id/model_id."""
    targets = [t for t in (artifact_uri, run_id, model_id) if t]
    if len(targets) > 1:
        raise MlflowException(
            "Namespace CAB scopes accept only one of artifact_uri, run_id, "
            "or model_id.",
            error_code=INVALID_PARAMETER_VALUE,
        )
    body: dict[str, str] = {}
    if artifact_uri:
        body["artifactUri"] = artifact_uri.rstrip("/")
    if run_id:
        body["runId"] = run_id
    if model_id:
        body["modelId"] = model_id
    if artifact_path:
        cleaned = artifact_path.strip("/")
        if any(part in ("", ".", "..") for part in cleaned.split("/")):
            raise MlflowException(
                f"Invalid artifact_path {artifact_path!r} for a CAB scope.",
                error_code=INVALID_PARAMETER_VALUE,
            )
        body["artifactPath"] = cleaned
    verb = "issueReadWriteArtifactToken" if write else "issueReadArtifactToken"
    return CabScope(
        issue_url=build_gcp_api_url(namespace, verb=verb),
        issue_body=_encode_body(body),
        mode=READ_WRITE if write else READ_ONLY,
        boundary=(artifact_uri or "").rstrip("/"),
        resource_name=namespace.resource_name,
    )


def registered_model_scope(
    registry: ParsedGcpUri,
    registered_model_resource: str,
    *,
    write: bool,
    artifact_uri: str | None = None,
) -> CabScope:
    """Build a RegisteredModel Vault scope (never sends ``artifactPath``)."""
    body: dict[str, str] = {}
    if artifact_uri:
        body["artifactUri"] = artifact_uri.rstrip("/")
    verb = "issueReadWriteArtifactToken" if write else "issueReadArtifactToken"
    return CabScope(
        issue_url=build_gcp_resource_url(
            registry, registered_model_resource, verb=verb
        ),
        issue_body=_encode_body(body),
        mode=READ_CREATE if write else READ_ONLY,
        boundary=(artifact_uri or "").rstrip("/"),
        resource_name=registered_model_resource,
    )


def record_vault_prefix(
    storage_location: str, model_resource: str, registry: ParsedGcpUri
) -> None:
    """Cache a RegisteredModel Vault ``gs://`` prefix for scope lookup."""
    clean = storage_location.rstrip("/")
    if clean.startswith("gs://"):
        with _VAULT_LOCK:
            _VAULT_PREFIXES[clean] = (model_resource, registry)


def lookup_vault_prefix(
    gcs_uri: str, registry: ParsedGcpUri | None = None
) -> tuple[str, str, ParsedGcpUri] | None:
    """Resolve a RegisteredModel Vault ``(prefix, model_resource, registry)``."""
    clean = gcs_uri.rstrip("/")
    with _VAULT_LOCK:
        matches = [
            (prefix, model_resource, reg)
            for prefix, (model_resource, reg) in _VAULT_PREFIXES.items()
            if clean == prefix or clean.startswith(f"{prefix}/")
        ]
    if matches:
        matches.sort(key=lambda item: len(item[0]), reverse=True)
        return matches[0]
    if registry is None:
        return None
    found = _query_registry_for_uri(registry, clean)
    if found is not None:
        return found
    match = _REGISTERED_MODEL_PATH_RE.match(clean)
    if match:
        model_id = match.group("model_id")
        marker = f"/registeredModels/{model_id}"
        idx = clean.find(marker)
        prefix = clean[: idx + len(marker)] if idx >= 0 else clean
        model_resource = f"{registry.resource_name}/registeredModels/{model_id}"
        record_vault_prefix(prefix, model_resource, registry)
        return (prefix, model_resource, registry)
    return None


def _query_registry_for_uri(
    registry: ParsedGcpUri, gcs_uri: str
) -> tuple[str, str, ParsedGcpUri] | None:
    url = build_gcp_api_url(registry, subpath="registeredModels")
    validate_transport_security(url)
    page_token = ""
    with create_secure_session() as http:
        bearer = GoogleCloudBearerAuth.get_instance()
        for _ in range(10):
            params: dict[str, str] = {"pageSize": "100"}
            if page_token:
                params["pageToken"] = page_token
            query = urllib.parse.urlencode(params)
            try:
                resp = http.get(
                    f"{url}?{query}",
                    auth=bearer,
                    timeout=30,
                    allow_redirects=False,
                )
            except requests.exceptions.RequestException:
                return None
            if resp.status_code != 200:
                return None
            try:
                payload = resp.json()
            except ValueError:
                return None
            if not isinstance(payload, dict):
                return None
            models = payload.get("registeredModels") or payload.get("registered_models")
            if isinstance(models, list):
                for item in models:
                    if not isinstance(item, dict):
                        continue
                    name = item.get("name")
                    loc = item.get("storageLocation") or item.get("storage_location")
                    if isinstance(name, str) and isinstance(loc, str) and loc:
                        record_vault_prefix(loc, name, registry)
                        clean_loc = loc.rstrip("/")
                        if gcs_uri == clean_loc or gcs_uri.startswith(f"{clean_loc}/"):
                            return (clean_loc, name, registry)
            nxt = payload.get("nextPageToken") or payload.get("next_page_token")
            if not isinstance(nxt, str) or not nxt:
                break
            page_token = nxt
    return None


@dataclasses.dataclass(frozen=True)
class CabToken:
    """A minted downscoped token. ``repr`` never renders the secret."""

    token: str = dataclasses.field(repr=False)
    expiry: datetime.datetime
    artifact_uri: str

    @property
    def expires_at_ms(self) -> int:
        """Expiry as epoch milliseconds."""
        return int(self.expiry.replace(tzinfo=datetime.timezone.utc).timestamp() * 1000)


def _utcnow() -> datetime.datetime:
    return datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None)


def _first(mapping: Any, *keys: str) -> Any:
    if not isinstance(mapping, dict):
        return None
    for key in keys:
        value = mapping.get(key)
        if value not in (None, ""):
            return value
    return None


def parse_token_response(payload: Any) -> CabToken:
    """Parse a camelCase or snake_case CAB token response."""
    credentials = _first(payload, "credentials") or payload
    token = _first(credentials, "accessToken", "access_token") or _first(
        payload, "accessToken", "access_token"
    )
    if not isinstance(token, str):
        raise MlflowException(
            "CAB token response did not include credentials.accessToken.",
            error_code=INTERNAL_ERROR,
        )
    raw_expiry = _first(credentials, "expireTime", "expire_time") or _first(
        payload, "expireTime", "expire_time"
    )
    expiry = (
        parse_rfc3339_utc(raw_expiry)
        if isinstance(raw_expiry, str)
        else _utcnow() + datetime.timedelta(hours=1)
    )
    artifact_uri = _first(payload, "artifactUri", "artifact_uri") or ""
    return CabToken(token=token, expiry=expiry, artifact_uri=str(artifact_uri))


def _error_code_for(status: int) -> int:
    if status == 401:
        return UNAUTHENTICATED
    if status == 403:
        return PERMISSION_DENIED
    if status == 404:
        return RESOURCE_DOES_NOT_EXIST
    if 400 <= status < 500:
        return INVALID_PARAMETER_VALUE
    return INTERNAL_ERROR


def _error_summary(response: requests.Response) -> str:
    try:
        payload = response.json()
    except ValueError:
        return redact(response.text[:300])
    error = payload.get("error") if isinstance(payload, dict) else None
    if isinstance(error, dict):
        status = error.get("status") or ""
        message = error.get("message") or ""
        return redact(f"{status} {message}".strip())
    return redact(json.dumps(payload)[:300])


def _retry_after_seconds(response: requests.Response) -> float | None:
    value = str(response.headers.get("Retry-After") or "")
    if not value:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        parsed = email.utils.parsedate_to_datetime(value)
        return max(0.0, float(parsed.timestamp()) - time.time())


def _user_agent() -> str:
    from google_cloud_managed_mlflow import __version__

    return f"google-cloud-managed-mlflow/{__version__} (role=client)"


def issue_token(
    scope: CabScope,
    *,
    session: requests.Session | None = None,
    auth: requests.auth.AuthBase | None = None,
    max_attempts: int = _MAX_ATTEMPTS,
    sleep: Any = time.sleep,
) -> CabToken:
    """Call the scope's issuance RPC with bounded retries on transient errors."""
    validate_transport_security(scope.issue_url)
    owned = session is None
    http = session if session is not None else create_secure_session()
    bearer = auth if auth is not None else GoogleCloudBearerAuth.get_instance()
    last_error = "no attempt made"
    try:
        for attempt in range(1, max_attempts + 1):
            try:
                response = http.post(
                    scope.issue_url,
                    data=scope.issue_body.encode("utf-8"),
                    headers={
                        "Content-Type": "application/json",
                        "User-Agent": _user_agent(),
                    },
                    auth=bearer,
                    timeout=60,
                    allow_redirects=False,
                )
            except requests.exceptions.RequestException as exc:
                last_error = redact(f"{type(exc).__name__}: {exc}")
                if attempt < max_attempts:
                    sleep(_backoff(attempt, None))
                    continue
                break
            if response.status_code == 200:
                with _COUNTS_LOCK:
                    CAB_RPC_COUNTS[scope.scope_id] += 1
                return parse_token_response(response.json())
            last_error = f"HTTP {response.status_code}: {_error_summary(response)}"
            if response.status_code in _RETRYABLE_STATUS and attempt < max_attempts:
                sleep(_backoff(attempt, _retry_after_seconds(response)))
                continue
            raise MlflowException(
                f"MLflow CAB token issuance failed ({scope.mode}): {last_error}",
                error_code=_error_code_for(response.status_code),
            )
    finally:
        if owned:
            http.close()
    raise MlflowException(
        f"MLflow CAB token issuance failed ({scope.mode}): {last_error}",
        error_code=INTERNAL_ERROR,
    )


def _backoff(attempt: int, retry_after: float | None) -> float:
    if retry_after is not None:
        return min(_MAX_BACKOFF_SECONDS, retry_after)
    capped = min(_MAX_BACKOFF_SECONDS, _INITIAL_BACKOFF_SECONDS * 2.0 ** (attempt - 1))
    return capped / 2 + random.uniform(0, capped / 2)


class CabCredentials(google.auth.credentials.Credentials):
    """Auto-refreshing google-auth credentials backed by one CAB scope."""

    def __init__(self, scope: CabScope) -> None:
        super().__init__()  # type: ignore[no-untyped-call]
        self._scope = scope
        self._lock = threading.Lock()
        self._last_refresh = 0.0
        self._artifact_uri = scope.boundary
        self.refresh_count = 0

    @property
    def scope(self) -> CabScope:
        """The CAB scope backing these credentials."""
        return self._scope

    def __repr__(self) -> str:
        state = "set" if self.token else "unset"
        scope = self._scope
        return (
            f"<CabCredentials scope={scope.scope_id} mode={scope.mode} "
            f"token={state}>"
        )

    @property
    def expired(self) -> bool:
        if not self.expiry:
            return False
        now = _utcnow()
        if now >= self.expiry:
            return True
        margin = datetime.timedelta(seconds=refresh_margin_seconds())
        if now < self.expiry - margin:
            return False
        return time.monotonic() - self._last_refresh >= _MIN_REFRESH_SPACING_SECONDS

    def refresh(self, request: Any) -> None:
        token_before = self.token
        with self._lock:
            if self.token is not None and self.token != token_before and self.valid:
                return
            minted = issue_token(self._scope)
            self.token = minted.token
            self.expiry = minted.expiry
            if minted.artifact_uri:
                self._artifact_uri = minted.artifact_uri.rstrip("/")
            self._last_refresh = time.monotonic()
            self.refresh_count += 1

    def current_token(self) -> CabToken:
        """Return a valid token, refreshing first when needed."""
        if not self.valid:
            self.refresh(None)
        assert self.token is not None and self.expiry is not None
        return CabToken(
            token=self.token, expiry=self.expiry, artifact_uri=self._artifact_uri
        )

    @classmethod
    def shared(cls, scope: CabScope) -> CabCredentials:
        """Return the process-wide credentials for ``scope`` (one token per scope)."""
        with _SHARED_LOCK:
            credentials = _SHARED_CREDENTIALS.get(scope.scope_id)
            if credentials is None:
                credentials = cls(scope)
                _SHARED_CREDENTIALS[scope.scope_id] = credentials
            return credentials


_SHARED_LOCK = threading.Lock()
_SHARED_CREDENTIALS: dict[str, CabCredentials] = {}


def reset_shared_credentials() -> None:
    """Forget cached per-scope credentials and Vault prefixes (used by tests)."""
    with _SHARED_LOCK:
        _SHARED_CREDENTIALS.clear()
    with _COUNTS_LOCK:
        CAB_RPC_COUNTS.clear()
    with _VAULT_LOCK:
        _VAULT_PREFIXES.clear()
