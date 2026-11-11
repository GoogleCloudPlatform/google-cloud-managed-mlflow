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

"""Credential Access Boundary (CAB) credentials and token acquisition."""

from __future__ import annotations

import datetime
import enum
import re
import threading
import urllib.parse
import weakref
from collections import OrderedDict
from typing import Any, Final, NamedTuple, TypeVar

import google.auth.credentials
import requests
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import (
    INTERNAL_ERROR,
    INVALID_PARAMETER_VALUE,
    RESOURCE_DOES_NOT_EXIST,
    ErrorCode,
)

from google_cloud_managed_mlflow.auth import (
    GoogleCloudBearerAuth,
    create_secure_session,
)
from google_cloud_managed_mlflow.uri import (
    GcpMlflowApiError,
    ParsedGcpUri,
    build_gcp_resource_url,
    parse_gcp_uri,
    parse_rfc3339_utc,
    raise_for_aip193_error,
)

_EARLY_EXPIRY_BUFFER = datetime.timedelta(seconds=300)
_UNSAFE_PATH_CHARS: frozenset[str] = frozenset("'\"\\#?\u2028\u2029")
_RESOURCE_ID_RE = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")
_REGISTERED_MODEL_ID_RE = re.compile(r"^[a-zA-Z0-9_-]{1,256}$")
_CANONICAL_ARTIFACTS_BOUNDARY_RE = re.compile(
    r"/(?:runs/[a-zA-Z0-9_-]{1,64}|models/[a-zA-Z0-9_-]{1,64}|[0-9a-f]{32})/artifacts(?=/|$)"
)
# Intentional regex asymmetry: CAB token issuance requires strictly alphanumeric
# resource IDs (_REGISTERED_MODEL_ID_RE), whereas storage path matching allows
# arbitrary non-slash characters (_REGISTERED_MODEL_VERSION_BOUNDARY_RE) so that
# read-only protection and URI validation still cover non-CAB IDs (e.g. 'my.model').
_REGISTERED_MODEL_VERSION_BOUNDARY_RE = re.compile(
    r"(?P<model_root>/registeredModels/[^/]{1,256})/versions/[a-zA-Z0-9._-]{1,64}(?=/|$)"
)
_REGISTERED_MODEL_VERSIONS_PARENT_RE: Final[re.Pattern[str]] = re.compile(
    r"/registeredModels/[^/]{1,256}/versions$"
)
_FORBIDDEN_GCS_ENCODED_TOKENS: Final[frozenset[str]] = frozenset(
    {"%2e", "%2f", "%5c", "%00"}
)
_MAX_MODEL_SEARCH_PAGES: int = 100


class UnsupportedCabModelIdError(MlflowException):
    """Raised when a valid registered model ID uses characters unsupported by CAB."""


class ClosedCabCredentialsError(MlflowException):
    """Raised when attempting to refresh a closed CabTokenCredentials instance."""


class CabAccessMode(str, enum.Enum):
    """Access mode for Credential Access Boundary (CAB) tokens."""

    READ_ONLY = "READ_ONLY"
    READ_WRITE = "READ_WRITE"


class CabResourceKind(str, enum.Enum):
    """Resource kind for Credential Access Boundary (CAB) tokens."""

    NAMESPACE = "NAMESPACE"
    REGISTERED_MODEL = "REGISTERED_MODEL"


class RegisteredModelGcsMatch(NamedTuple):
    """Correlation match between a GCS artifact URI and a registered model."""

    version_root_uri: str
    registry_resource_url: str
    model_name: str


def has_gs_scheme(uri: Any) -> bool:
    """Return True if uri is a string starting with case-insensitive 'gs://'."""
    return isinstance(uri, str) and uri[:5].lower() == "gs://"


def strip_mlmodel_leaf(rel_path: str) -> str:
    """Strip trailing /MLmodel leaf segment from a normalized relative path."""
    if rel_path == "MLmodel":
        return ""
    if rel_path.endswith("/MLmodel"):
        return rel_path[: -len("/MLmodel")]
    return rel_path


def _has_canonical_namespace_artifacts_prefix(path_part: str, before_idx: int) -> bool:
    """Return True if path_part[:before_idx] has a canonical namespace boundary.

    Generic '/artifacts' segments in custom Registry storage roots are permitted.
    """
    return bool(_CANONICAL_ARTIFACTS_BOUNDARY_RE.search(path_part[:before_idx]))


def _match_registered_model_boundary(
    path_part: str, *, include_versions_parent: bool = False
) -> re.Match[str] | None:
    """Match registered model boundary, rejecting namespace /artifacts prefixes."""
    match = _REGISTERED_MODEL_VERSION_BOUNDARY_RE.search(path_part)
    if match is not None:
        if _has_canonical_namespace_artifacts_prefix(path_part, match.start()):
            return None
        return match
    if include_versions_parent:
        parent_match = _REGISTERED_MODEL_VERSIONS_PARENT_RE.search(path_part)
        if parent_match is not None:
            if _has_canonical_namespace_artifacts_prefix(
                path_part, parent_match.start()
            ):
                return None
            return parent_match
    return None


def _json_field(obj: dict[str, Any], camel_key: str, snake_key: str) -> Any:
    """Return a proto3-JSON field, preferring lowerCamelCase over snake_case.

    Google Cloud REST responses use lowerCamelCase (the proto3 JSON default);
    snake_case is also accepted for servers that preserve proto field names.
    """
    value = obj.get(camel_key)
    return obj.get(snake_key) if value is None else value


def _close_session(session: requests.Session | None) -> None:
    """Close a requests session safely during finalization."""
    if session is not None:
        try:
            session.close()
        except Exception:
            pass


def _has_unsafe_or_control_chars(value: str) -> bool:
    """Return True if value contains control chars, DEL, C1, or unsafe chars."""
    return any(
        ord(c) < 0x20
        or ord(c) == 0x7F
        or ("\x80" <= c <= "\x9f")
        or c in _UNSAFE_PATH_CHARS
        for c in value
    )


def _validate_relative_gcs_segments(raw_path: str, *, label: str) -> None:
    """Validate raw_path contains no percent-encoded traversal, //, or ./.. segments."""
    lower_path = raw_path.lower()
    if any(token in lower_path for token in _FORBIDDEN_GCS_ENCODED_TOKENS):
        raise MlflowException(
            f"Invalid {label}: "
            "percent-encoded traversal or delimiter tokens are not allowed.",
            error_code=INVALID_PARAMETER_VALUE,
        )
    if "//" in raw_path:
        raise MlflowException(
            f"Invalid {label}: empty path segments are not allowed.",
            error_code=INVALID_PARAMETER_VALUE,
        )
    trimmed = raw_path.strip("/")
    if not trimmed:
        return
    segments = trimmed.split("/")
    if any(seg in ("", ".", "..") for seg in segments):
        raise MlflowException(
            f"Invalid {label}: path traversal segments are not allowed.",
            error_code=INVALID_PARAMETER_VALUE,
        )


def _validate_artifact_path(artifact_path: str) -> str:
    """Validate and normalize a client artifact subpath.

    Matches backend GcsCabUriValidator.normalizeClientSubpath.
    """
    if not isinstance(artifact_path, str) or not artifact_path.strip():
        raise MlflowException(
            "artifact_path must be a non-empty relative path.",
            error_code=INVALID_PARAMETER_VALUE,
        )
    if artifact_path.startswith("/") or _has_unsafe_or_control_chars(artifact_path):
        raise MlflowException(
            f"Invalid artifact_path '{artifact_path}': path must be relative "
            "and cannot contain control or unsafe characters.",
            error_code=INVALID_PARAMETER_VALUE,
        )
    _validate_relative_gcs_segments(
        artifact_path, label=f"artifact_path '{artifact_path}'"
    )
    trimmed = artifact_path.rstrip("/")
    if not trimmed:
        raise MlflowException(
            f"Invalid artifact_path '{artifact_path}': "
            "empty path segments are not allowed.",
            error_code=INVALID_PARAMETER_VALUE,
        )
    return trimmed


def _validate_gcs_artifact_uri(artifact_uri: str) -> tuple[str, str]:
    """Validate and parse a GCS artifact URI.

    Returns:
        (bucket, rel_path) where rel_path is stripped of leading and trailing slashes.
    """
    if (
        not isinstance(artifact_uri, str)
        or not artifact_uri.strip()
        or _has_unsafe_or_control_chars(artifact_uri)
    ):
        raise MlflowException(
            f"Invalid GCS artifact URI '{artifact_uri}': "
            "URI must be a non-empty string without control or unsafe characters.",
            error_code=INVALID_PARAMETER_VALUE,
        )

    parsed = urllib.parse.urlsplit(artifact_uri)
    if parsed.scheme.lower() != "gs":
        raise MlflowException(
            f"Invalid GCS artifact URI '{artifact_uri}': must start with 'gs://'.",
            error_code=INVALID_PARAMETER_VALUE,
        )

    bucket = parsed.netloc
    if not bucket:
        raise MlflowException(
            f"Invalid GCS artifact URI '{artifact_uri}': bucket name cannot be empty.",
            error_code=INVALID_PARAMETER_VALUE,
        )

    if parsed.query or parsed.fragment:
        raise MlflowException(
            f"Invalid GCS artifact URI '{artifact_uri}': query parameters and "
            "fragments are not allowed.",
            error_code=INVALID_PARAMETER_VALUE,
        )

    raw_path = parsed.path
    _validate_relative_gcs_segments(
        raw_path, label=f"GCS artifact URI '{artifact_uri}'"
    )
    rel_path = raw_path.strip("/")
    return bucket, rel_path


def derive_cab_directory_uri(artifact_uri: str) -> str:
    """Derive the directory-level GCS URI prefix for CAB token issuance.

    Strips leaf file segments or trailing slashes, truncates run artifacts
    subpaths to the /artifacts boundary, and returns normalized directory URIs.

    Args:
        artifact_uri: The raw GCS artifact URI string.

    Returns:
        The normalized directory-level GCS URI.

    Raises:
        MlflowException: If artifact_uri does not start with 'gs://', contains
            unsafe or control characters, an empty bucket name, invalid path
            segments, or query parameters / fragments.
    """
    bucket, rel_path = _validate_gcs_artifact_uri(artifact_uri)
    if not rel_path:
        return f"gs://{bucket}"

    path_part = f"/{rel_path}"

    matches = list(_CANONICAL_ARTIFACTS_BOUNDARY_RE.finditer(path_part))
    if matches:
        return f"gs://{bucket}{path_part[: matches[0].end()]}"

    art_marker = "/artifacts/"
    if art_marker in path_part:
        idx = path_part.rindex(art_marker)
        return f"gs://{bucket}{path_part[: idx + len(art_marker) - 1]}"

    if not artifact_uri.endswith("/"):
        stripped = strip_mlmodel_leaf(rel_path)
        if stripped != rel_path:
            return f"gs://{bucket}/{stripped}" if stripped else f"gs://{bucket}"

    return f"gs://{bucket}/{rel_path}"


def derive_registered_model_cab_uri(artifact_uri: str) -> str:
    """Derive the version-root GCS URI prefix for RegisteredModel CAB issuance.

    Truncates subpaths beneath /registeredModels/<id>/versions/<ver>, strips
    leaf /MLmodel if present, and returns the normalized root GCS URI without
    truncating canonical namespace paths (/runs/<id>/artifacts,
    /models/<id>/artifacts, or /<32-hex>/artifacts).
    """
    bucket, rel_path = _validate_gcs_artifact_uri(artifact_uri)
    if not rel_path:
        return f"gs://{bucket}"

    path_part = f"/{rel_path}"
    match = _match_registered_model_boundary(path_part)
    if match:
        version_root = path_part[: match.end()].lstrip("/")
        return f"gs://{bucket}/{version_root}"

    if not artifact_uri.endswith("/"):
        stripped = strip_mlmodel_leaf(rel_path)
        if stripped != rel_path:
            return f"gs://{bucket}/{stripped}" if stripped else f"gs://{bucket}"

    return f"gs://{bucket}/{rel_path}"


def _validate_cab_resource_url(
    resource_url: str, resource_kind: CabResourceKind
) -> str:
    """Validate and normalize a Google Cloud CAB resource URL."""
    if not isinstance(resource_url, str) or not resource_url.strip():
        raise MlflowException(
            "resource_url must be a non-empty string.",
            error_code=INVALID_PARAMETER_VALUE,
        )
    if _has_unsafe_or_control_chars(resource_url):
        raise MlflowException(
            f"Invalid resource_url '{resource_url}': "
            "contains control or unsafe characters.",
            error_code=INVALID_PARAMETER_VALUE,
        )
    cleaned = resource_url.rstrip("/")
    if resource_kind == CabResourceKind.NAMESPACE:
        parsed = parse_gcp_uri(cleaned)
        if not parsed.is_namespace:
            raise MlflowException(
                f"Expected a namespace resource URL, got '{resource_url}'.",
                error_code=INVALID_PARAMETER_VALUE,
            )
        return parsed.resource_url
    if resource_kind == CabResourceKind.REGISTERED_MODEL:
        parts = cleaned.split("/registeredModels/")
        if len(parts) != 2:
            raise MlflowException(
                "RegisteredModel resource URL must follow format: "
                "'{registry_resource_url}/registeredModels/{model_id}', "
                f"got '{resource_url}'.",
                error_code=INVALID_PARAMETER_VALUE,
            )
        reg_url, model_seg = parts
        parsed_reg = parse_gcp_uri(reg_url)
        if not parsed_reg.is_registry:
            raise MlflowException(
                "Expected a model registry resource URL before '/registeredModels/', "
                f"got '{resource_url}'.",
                error_code=INVALID_PARAMETER_VALUE,
            )
        if not _REGISTERED_MODEL_ID_RE.fullmatch(model_seg):
            raise UnsupportedCabModelIdError(
                f"Invalid registered model resource ID '{model_seg}'.",
                error_code=INVALID_PARAMETER_VALUE,
            )
        return build_gcp_resource_url(
            parsed_reg, f"{parsed_reg.resource_name}/registeredModels/{model_seg}"
        )
    raise MlflowException(
        f"Unsupported CabResourceKind: {resource_kind}",
        error_code=INVALID_PARAMETER_VALUE,
    )


def _is_stale_registered_model_token_error(exc: GcpMlflowApiError) -> bool:
    """Return True if an RPC error indicates a stale registered model resource URL.

    When a registered model is deleted and recreated under the same display name,
    the model ID (resource name suffix) changes. A cached resource URL referring
    to the old ID will return HTTP 404 / NOT_FOUND from the CAB token endpoint,
    or HTTP 400/403 (PERMISSION_DENIED or FAILED_PRECONDITION) with a storage root
    containment error because the storage path corresponding to the new model
    version is not contained within the storage root of the old model resource.
    """
    if exc.http_status == 404 or exc.gcp_status == "NOT_FOUND":
        return True
    if (
        exc.http_status in (400, 403)
        or exc.gcp_status in ("PERMISSION_DENIED", "FAILED_PRECONDITION")
    ) and "not contained within storage root" in str(exc).lower():
        return True
    return False


class CabTokenCredentials(google.auth.credentials.Credentials):
    """STS Credential Access Boundary credentials for Google Cloud Storage."""

    def __init__(
        self,
        resource_url: str,
        resource_kind: CabResourceKind = CabResourceKind.NAMESPACE,
        access_mode: CabAccessMode = CabAccessMode.READ_ONLY,
        artifact_uri: str | None = None,
        run_id: str | None = None,
        model_id: str | None = None,
        artifact_path: str | None = None,
        session: requests.Session | None = None,
        auth: Any = None,
        registry_resource_url: str | None = None,
        model_name: str | None = None,
    ) -> None:
        super_init: Any = super().__init__
        super_init()

        validated_url = _validate_cab_resource_url(resource_url, resource_kind)

        if resource_kind == CabResourceKind.REGISTERED_MODEL:
            if access_mode != CabAccessMode.READ_ONLY:
                raise MlflowException(
                    "RegisteredModel artifact credentials only support READ_ONLY "
                    "access mode; Registered models are read-only.",
                    error_code=INVALID_PARAMETER_VALUE,
                )
            if artifact_path is not None or run_id is not None or model_id is not None:
                raise MlflowException(
                    "RegisteredModel artifact credentials only support artifact_uri; "
                    "run_id, model_id, and artifact_path are not supported.",
                    error_code=INVALID_PARAMETER_VALUE,
                )
        else:
            specified = [
                name
                for name, val in [
                    ("artifact_uri", artifact_uri),
                    ("run_id", run_id),
                    ("model_id", model_id),
                ]
                if val is not None
            ]
            if len(specified) > 1:
                raise MlflowException(
                    "At most one of artifact_uri, run_id, or model_id can be "
                    f"specified; got {specified}.",
                    error_code=INVALID_PARAMETER_VALUE,
                )

        if artifact_uri is not None:
            _validate_gcs_artifact_uri(artifact_uri)

        if run_id is not None:
            if not isinstance(run_id, str) or not _RESOURCE_ID_RE.fullmatch(run_id):
                raise MlflowException(
                    f"Invalid run_id '{run_id}': must match ^[a-zA-Z0-9_-]{{1,64}}$.",
                    error_code=INVALID_PARAMETER_VALUE,
                )

        if model_id is not None:
            if not isinstance(model_id, str) or not _RESOURCE_ID_RE.fullmatch(model_id):
                raise MlflowException(
                    f"Invalid model_id '{model_id}': "
                    "must match ^[a-zA-Z0-9_-]{{1,64}}$.",
                    error_code=INVALID_PARAMETER_VALUE,
                )

        if resource_kind == CabResourceKind.REGISTERED_MODEL:
            derived_reg_url = validated_url.split("/registeredModels/", 1)[0]
            if registry_resource_url is not None:
                parsed_reg = parse_gcp_uri(registry_resource_url)
                if (
                    not parsed_reg.is_registry
                    or parsed_reg.resource_url != derived_reg_url
                ):
                    raise MlflowException(
                        f"registry_resource_url '{registry_resource_url}' does not "
                        f"match resource_url '{resource_url}'.",
                        error_code=INVALID_PARAMETER_VALUE,
                    )
            self._registry_resource_url: str | None = derived_reg_url
        else:
            if registry_resource_url is not None or model_name is not None:
                raise MlflowException(
                    "registry_resource_url and model_name are only supported for "
                    "REGISTERED_MODEL.",
                    error_code=INVALID_PARAMETER_VALUE,
                )
            self._registry_resource_url = None

        self._model_name: str | None = None
        if model_name is not None:
            if not isinstance(model_name, str) or not model_name.strip():
                raise MlflowException(
                    "Registered model name must be non-empty.",
                    error_code=INVALID_PARAMETER_VALUE,
                )
            self._model_name = model_name

        self._closed: bool = False

        # Lock hierarchy: _refresh_lock -> _lock.
        # - _refresh_lock serializes in-flight token RPCs so property readers
        #   never block.
        # - _lock guards _closed, _resource_url, _access_mode, _session,
        #   _custom_session, _refresh_generation, _effective_buffer, token,
        #   and expiry. Never acquire _refresh_lock or perform network I/O
        #   while holding _lock.
        self._lock = threading.Lock()
        self._refresh_lock = threading.Lock()
        self._resource_url = validated_url
        self._resource_kind = resource_kind
        self._access_mode = access_mode
        self._artifact_uri = artifact_uri
        self._run_id = run_id
        self._model_id = model_id
        self._artifact_path: str | None = (
            _validate_artifact_path(artifact_path)
            if artifact_path is not None
            else None
        )
        self._custom_session = session is not None
        self._session: requests.Session | None = (
            session if session is not None else create_secure_session()
        )
        self._auth = auth if auth is not None else GoogleCloudBearerAuth.get_instance()
        self._refresh_generation: int = 0
        self._effective_buffer: datetime.timedelta = _EARLY_EXPIRY_BUFFER
        self.token: str | None = None
        self.expiry: datetime.datetime | None = None

        if not self._custom_session:
            weakref.finalize(self, _close_session, self._session)

    @property
    def closed(self) -> bool:
        """Return True if this credentials instance has been closed."""
        with self._lock:
            return self._closed

    def _raise_if_closed_locked(self) -> None:
        if self._closed:
            raise ClosedCabCredentialsError(
                "Cannot refresh closed CabTokenCredentials.",
                error_code=INVALID_PARAMETER_VALUE,
            )

    @property
    def resource_url(self) -> str:
        """Return the canonical resource URL."""
        with self._lock:
            return self._resource_url

    @property
    def registry_resource_url(self) -> str | None:
        """Return the registry resource URL if configured."""
        return self._registry_resource_url

    @property
    def access_mode(self) -> CabAccessMode:
        """Return the current access mode."""
        with self._lock:
            return self._access_mode

    @property
    def resource_kind(self) -> CabResourceKind:
        """Return the resource kind."""
        return self._resource_kind

    def _is_expired_locked(self) -> bool:
        if self.expiry is None:
            return True
        now = datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None)
        return now >= (self.expiry - self._effective_buffer)

    def _is_valid_locked(self) -> bool:
        return self.token is not None and not self._is_expired_locked()

    @property
    def expired(self) -> bool:
        """Return True if credentials are expired or within early refresh margin."""
        with self._lock:
            return self._is_expired_locked()

    @property
    def valid(self) -> bool:
        """Return True if credentials have a token and are not expired."""
        with self._lock:
            return self._is_valid_locked()

    @property
    def token_state(self) -> google.auth.credentials.TokenState:
        """Return token freshness state considering effective early refresh margin."""
        with self._lock:
            if self.token is None or self.expiry is None:
                return google.auth.credentials.TokenState.INVALID
            now = datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None)
            if now >= self.expiry:
                return google.auth.credentials.TokenState.INVALID
            if now >= (self.expiry - self._effective_buffer):
                return google.auth.credentials.TokenState.STALE
            return google.auth.credentials.TokenState.FRESH

    def with_access_mode(self, access_mode: CabAccessMode) -> CabTokenCredentials:
        """Create a new CabTokenCredentials instance with a different access mode."""
        res_url = self.resource_url
        return CabTokenCredentials(
            resource_url=res_url,
            resource_kind=self._resource_kind,
            access_mode=access_mode,
            artifact_uri=self._artifact_uri,
            run_id=self._run_id,
            model_id=self._model_id,
            artifact_path=self._artifact_path,
            session=self._session if self._custom_session else None,
            auth=self._auth,
            registry_resource_url=self._registry_resource_url,
            model_name=self._model_name,
        )

    def elevate_to_read_write(self) -> bool:
        """Elevate access mode to READ_WRITE and invalidate the cached token.

        Returns:
            True if this call changed the access mode, False if it was already
            READ_WRITE.
        """
        if self._resource_kind == CabResourceKind.REGISTERED_MODEL:
            raise MlflowException(
                "RegisteredModel artifact credentials cannot be elevated to "
                "READ_WRITE; Registered models are read-only.",
                error_code=INVALID_PARAMETER_VALUE,
            )
        with self._lock:
            if self._access_mode == CabAccessMode.READ_WRITE:
                return False
            self._access_mode = CabAccessMode.READ_WRITE
            self.token = None
            self.expiry = None
            self._effective_buffer = _EARLY_EXPIRY_BUFFER
            self._refresh_generation += 1
            return True

    def close(self) -> None:
        """Close any cached HTTP session owned by this instance."""
        session_to_close: requests.Session | None = None
        with self._lock:
            self._closed = True
            if self._session is not None and not self._custom_session:
                session_to_close = self._session
                self._session = None
        if session_to_close is not None:
            try:
                session_to_close.close()
            except Exception:
                pass

    def _build_token_request(
        self, access_mode: CabAccessMode
    ) -> tuple[str, dict[str, Any]]:
        """Construct the RPC verb and payload for the requested access mode."""
        verb = (
            "issueReadWriteArtifactToken"
            if access_mode == CabAccessMode.READ_WRITE
            else "issueReadArtifactToken"
        )
        payload: dict[str, Any] = {}
        if self._artifact_uri is not None:
            payload["artifactUri"] = self._artifact_uri
        elif self._run_id is not None:
            payload["runId"] = self._run_id
        elif self._model_id is not None:
            payload["modelId"] = self._model_id

        if self._artifact_path is not None:
            payload["artifactPath"] = self._artifact_path
        return verb, payload

    @staticmethod
    def _parse_token_response(
        response: requests.Response,
    ) -> tuple[str, datetime.datetime, datetime.timedelta]:
        """Validate and parse the STS credentials token response."""
        try:
            data = response.json()
        except ValueError as exc:
            raise MlflowException(
                "Invalid response from server: malformed JSON payload.",
                error_code=INTERNAL_ERROR,
            ) from exc
        if not isinstance(data, dict):
            raise MlflowException(
                "Invalid response from server: expected a JSON object.",
                error_code=INTERNAL_ERROR,
            )

        creds_obj = data.get("credentials")
        if not isinstance(creds_obj, dict):
            raise MlflowException(
                "Invalid response from server: missing 'credentials' object.",
                error_code=INTERNAL_ERROR,
            )

        token = _json_field(creds_obj, "accessToken", "access_token")
        if not token or not isinstance(token, str):
            raise MlflowException(
                "Invalid response from server: missing 'accessToken' in credentials.",
                error_code=INTERNAL_ERROR,
            )

        expire_time_str = _json_field(creds_obj, "expireTime", "expire_time")
        if not expire_time_str or not isinstance(expire_time_str, str):
            raise MlflowException(
                "Invalid response from server: missing 'expireTime' in credentials.",
                error_code=INTERNAL_ERROR,
            )

        parsed_expiry = parse_rfc3339_utc(expire_time_str)
        now = datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None)
        ttl = parsed_expiry - now
        if ttl > datetime.timedelta(0):
            effective_buffer = min(_EARLY_EXPIRY_BUFFER, ttl / 2)
        else:
            effective_buffer = datetime.timedelta(0)

        return token, parsed_expiry, effective_buffer

    def refresh(self, request: Any = None) -> None:
        """Acquire a new CAB token via single-flight thread-safe execution."""
        with self._lock:
            self._raise_if_closed_locked()
            gen_before = self._refresh_generation

        with self._refresh_lock:
            while True:
                with self._lock:
                    self._raise_if_closed_locked()
                    if (
                        self._refresh_generation != gen_before
                        and self._is_valid_locked()
                    ):
                        return
                    snap_access_mode = self._access_mode
                    snap_resource_url = self._resource_url
                    session = self._session
                    if session is None:
                        session = create_secure_session()
                        self._session = session
                        self._custom_session = False
                        weakref.finalize(self, _close_session, self._session)

                verb, payload = self._build_token_request(snap_access_mode)
                target_url = snap_resource_url
                try:
                    response = self._post_token_rpc(session, target_url, verb, payload)
                except GcpMlflowApiError as exc:
                    target_url, response = self._retry_if_stale_model(
                        session, target_url, verb, payload, exc
                    )

                token, parsed_expiry, effective_buffer = self._parse_token_response(
                    response
                )

                with self._lock:
                    self._raise_if_closed_locked()
                    if self._access_mode != snap_access_mode:
                        # Retain self-healed target_url before looping for
                        # updated access_mode.
                        self._resource_url = target_url
                        gen_before = self._refresh_generation
                        continue

                    self._resource_url = target_url
                    self.token = token
                    self.expiry = parsed_expiry
                    self._effective_buffer = effective_buffer
                    self._refresh_generation += 1
                    return

    def _post_token_rpc(
        self,
        session: requests.Session,
        resource_url: str,
        verb: str,
        payload: dict[str, Any],
    ) -> requests.Response:
        url = f"{resource_url}:{verb}"
        response = session.post(
            url,
            json=payload,
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
            auth=self._auth,
            timeout=30,
            allow_redirects=False,
        )
        if 300 <= response.status_code < 400:
            raise MlflowException(
                f"Unexpected HTTP redirect ({response.status_code}) "
                "from CAB token endpoint.",
                error_code=INVALID_PARAMETER_VALUE,
            )
        raise_for_aip193_error(response)
        return response

    def _retry_if_stale_model(
        self,
        session: requests.Session,
        current_url: str,
        verb: str,
        payload: dict[str, Any],
        exc: GcpMlflowApiError,
    ) -> tuple[str, requests.Response]:
        """Retry token acquisition if model was recreated or URL is stale."""
        if not (
            self._resource_kind == CabResourceKind.REGISTERED_MODEL
            and self._registry_resource_url is not None
            and self._model_name is not None
            and _is_stale_registered_model_token_error(exc)
        ):
            raise exc

        reg_url = self._registry_resource_url
        model_name = self._model_name
        evict_cached_model_resource_url(reg_url, model_name)
        try:
            refreshed_url = resolve_registered_model_resource_url(
                registry_resource_url=reg_url,
                model_name=model_name,
                session=session,
                auth=self._auth,
            )
        except Exception:
            raise exc from None

        if refreshed_url == current_url:
            raise exc

        try:
            response = self._post_token_rpc(session, refreshed_url, verb, payload)
        except GcpMlflowApiError as retry_exc:
            if _is_stale_registered_model_token_error(retry_exc):
                stale_root = _extract_model_storage_root(self._artifact_uri)
                if stale_root:
                    _evict_download_uris_by_model_root(
                        reg_url,
                        model_name,
                        stale_root,
                        keep_matching=False,
                    )
            raise retry_exc from None

        active_root = _extract_model_storage_root(self._artifact_uri)
        if active_root:
            _evict_download_uris_by_model_root(
                reg_url,
                model_name,
                active_root,
                keep_matching=True,
            )
        if self._artifact_uri is not None and is_registered_model_storage_uri(
            self._artifact_uri
        ):
            record_registered_model_download_uri(
                self._artifact_uri,
                reg_url,
                model_name,
            )
        return refreshed_url, response

    def __repr__(self) -> str:
        with self._lock:
            token_status = "set" if self.token is not None else "unset"
            mode = self._access_mode.value
            kind = self._resource_kind.value
        return (
            f"<CabTokenCredentials resource_kind={kind} "
            f"access_mode={mode} token={token_status}>"
        )


def _validate_registered_model_entry(
    res_name: Any,
    expected_prefix_re: re.Pattern[str],
) -> tuple[str, str]:
    """Validate resource name format and return (res_name, matched_id)."""
    if (
        not isinstance(res_name, str)
        or not res_name
        or _has_unsafe_or_control_chars(res_name)
    ):
        raise MlflowException(
            "Invalid SearchRegisteredModels response: matched registered "
            "model has invalid 'name'.",
            error_code=INTERNAL_ERROR,
        )
    m = expected_prefix_re.fullmatch(res_name)
    if m is None:
        raise MlflowException(
            "Invalid SearchRegisteredModels response: resource name "
            f"'{res_name}' does not match expected pattern.",
            error_code=INTERNAL_ERROR,
        )
    matched_id = m.group("model_id")
    if len(matched_id) > 256 or matched_id in (".", ".."):
        raise MlflowException(
            "Invalid SearchRegisteredModels response: resource ID "
            f"'{matched_id}' is invalid or too long.",
            error_code=INTERNAL_ERROR,
        )
    return res_name, matched_id


def _resolve_registered_model_resource_name(
    session: requests.Session,
    registry_resource_url: str | ParsedGcpUri,
    model_name: str,
    auth: Any = None,
) -> str:
    """Resolve an MLflow model name to its canonical resource name.

    Queries SearchRegisteredModels with the filter name = '<model_name>',
    then returns the canonical resource name of the first result whose
    displayName (or display_name, or name) equals model_name exactly.

    Args:
        session: Active HTTP session for making the request.
        registry_resource_url: Full URL or ParsedGcpUri of the model registry resource.
        model_name: Open-source MLflow registered model name.
        auth: Optional requests auth handler (defaults to singleton).

    Returns:
        Canonical Google Cloud resource name for the registered model.

    Raises:
        MlflowException: INVALID_PARAMETER_VALUE if model_name is empty or
            registry_resource_url is invalid; INTERNAL_ERROR if the server
            response is malformed; RESOURCE_DOES_NOT_EXIST if no result matches.
    """
    if not isinstance(model_name, str) or not model_name.strip():
        raise MlflowException(
            "Registered model name must be non-empty.",
            error_code=INVALID_PARAMETER_VALUE,
        )

    if isinstance(registry_resource_url, ParsedGcpUri):
        parsed_reg = registry_resource_url
    else:
        parsed_reg = parse_gcp_uri(registry_resource_url)

    if not parsed_reg.is_registry:
        raise MlflowException(
            f"Expected a model registry resource URL, got '{parsed_reg.resource_url}'.",
            error_code=INVALID_PARAMETER_VALUE,
        )

    clean_registry_url = parsed_reg.resource_url
    escaped_model_name = model_name.replace("\\", "\\\\").replace("'", "\\'")
    filter_value = f"name = '{escaped_model_name}'"
    encoded_filter = urllib.parse.quote(filter_value, safe="")
    base_search_url = (
        f"{clean_registry_url}/api/2.0/mlflow/registered-models/search"
        f"?filter={encoded_filter}"
    )

    auth_handler = auth if auth is not None else GoogleCloudBearerAuth.get_instance()
    page_token: str | None = None
    expected_prefix_re = re.compile(
        rf"^projects/[a-zA-Z0-9._:-]+/locations/{re.escape(parsed_reg.location)}"
        rf"/registries/{re.escape(parsed_reg.resource_id)}"
        r"/registeredModels/(?P<model_id>[^/]+)$"
    )

    fallback_candidate: tuple[str, str] | None = None

    for _ in range(_MAX_MODEL_SEARCH_PAGES):
        search_url = base_search_url
        if page_token:
            encoded_token = urllib.parse.quote(page_token, safe="")
            search_url = f"{base_search_url}&page_token={encoded_token}"

        response = session.get(
            search_url,
            headers={"Accept": "application/json"},
            auth=auth_handler,
            timeout=30,
            allow_redirects=False,
        )
        if 300 <= response.status_code < 400:
            raise MlflowException(
                f"Unexpected HTTP redirect ({response.status_code}) from "
                "SearchRegisteredModels endpoint.",
                error_code=INVALID_PARAMETER_VALUE,
            )
        raise_for_aip193_error(response)
        try:
            data = response.json()
        except ValueError as exc:
            raise MlflowException(
                "Invalid SearchRegisteredModels response: malformed JSON payload.",
                error_code=INTERNAL_ERROR,
            ) from exc
        if not isinstance(data, dict):
            raise MlflowException(
                "Invalid SearchRegisteredModels response: expected a JSON object.",
                error_code=INTERNAL_ERROR,
            )

        registered_models = _json_field(data, "registeredModels", "registered_models")
        if registered_models is None:
            registered_models = []
        if not isinstance(registered_models, list):
            raise MlflowException(
                "Invalid SearchRegisteredModels response: "
                "'registeredModels' must be a list.",
                error_code=INTERNAL_ERROR,
            )

        for model in registered_models:
            if not isinstance(model, dict):
                continue
            disp_name = _json_field(model, "displayName", "display_name")
            res_name = model.get("name")
            res_id = (
                res_name.rsplit("/registeredModels/", 1)[-1]
                if isinstance(res_name, str) and "/registeredModels/" in res_name
                else None
            )
            exact_match = (
                disp_name == model_name
                or res_name == model_name
                or (not disp_name and res_id == model_name)
            )
            if exact_match:
                valid_res_name, matched_id = _validate_registered_model_entry(
                    res_name, expected_prefix_re
                )
                if not _REGISTERED_MODEL_ID_RE.fullmatch(matched_id):
                    raise UnsupportedCabModelIdError(
                        f"Registered model resource ID '{matched_id}' contains "
                        "characters not supported by CAB token issuance.",
                        error_code=INVALID_PARAMETER_VALUE,
                    )
                return valid_res_name

            if res_id == model_name and fallback_candidate is None:
                fallback_candidate = _validate_registered_model_entry(
                    res_name, expected_prefix_re
                )

        page_token = _json_field(data, "nextPageToken", "next_page_token")
        if not page_token or not isinstance(page_token, str):
            break

    if fallback_candidate is not None:
        fallback_res_name, fallback_matched_id = fallback_candidate
        if not _REGISTERED_MODEL_ID_RE.fullmatch(fallback_matched_id):
            raise UnsupportedCabModelIdError(
                f"Registered model resource ID '{fallback_matched_id}' contains "
                "characters not supported by CAB token issuance.",
                error_code=INVALID_PARAMETER_VALUE,
            )
        return fallback_res_name

    raise MlflowException(
        f"Registered model '{model_name}' not found in registry.",
        error_code=RESOURCE_DOES_NOT_EXIST,
    )


K = TypeVar("K")
V = TypeVar("V")


def _lru_get(cache: OrderedDict[K, V], key: K) -> V | None:
    val = cache.get(key)
    if val is not None:
        cache.move_to_end(key)
    return val


def _lru_put(cache: OrderedDict[K, V], key: K, value: V, maxsize: int) -> None:
    if key in cache:
        cache.move_to_end(key)
    elif len(cache) >= maxsize:
        cache.popitem(last=False)
    cache[key] = value


_MAX_MODEL_RESOURCE_NAME_CACHE: int = 1024
_MODEL_RESOURCE_NAME_CACHE: OrderedDict[tuple[str, str], str] = OrderedDict()
_DOWNLOAD_URI_MODEL_MAP: OrderedDict[str, tuple[str, str]] = OrderedDict()
_MODEL_RESOURCE_NAME_CACHE_LOCK = threading.Lock()


def _parse_model_cache_key(
    registry_resource_url: str,
    model_name: str | None,
) -> tuple[str, str] | None:
    """Validate registry URL and model name, returning key tuple or None."""
    if not isinstance(model_name, str) or not model_name.strip():
        return None
    try:
        parsed_reg = parse_gcp_uri(registry_resource_url)
    except MlflowException:
        return None
    if not parsed_reg.is_registry:
        return None
    return (parsed_reg.resource_url, model_name)


def _parse_registered_model_gcs_path(
    uri: str | None, *, include_versions_parent: bool = False
) -> tuple[str, str, re.Match[str]] | None:
    """Validate GCS URI and match registered model boundary.

    Returns (bucket, rel_path, match) or None.
    """
    if not isinstance(uri, str) or not has_gs_scheme(uri):
        return None
    try:
        bucket, rel_path = _validate_gcs_artifact_uri(uri)
    except MlflowException:
        return None
    if not bucket or not rel_path:
        return None
    path_part = f"/{rel_path}"
    match = _match_registered_model_boundary(
        path_part, include_versions_parent=include_versions_parent
    )
    if match is None:
        return None
    return bucket, rel_path, match


def _extract_model_storage_root(uri: str | None) -> str | None:
    """Return model root 'gs://.../registeredModels/<id>' from version URI."""
    parsed = _parse_registered_model_gcs_path(uri)
    if parsed is None:
        return None
    bucket, rel_path, match = parsed
    path_part = f"/{rel_path}"
    prefix = path_part[: match.start()]
    return f"gs://{bucket}{prefix}{match.group('model_root')}"


def _evict_download_uris_by_model_root(
    registry_resource_url: str,
    model_name: str,
    model_root: str,
    *,
    keep_matching: bool,
) -> None:
    """Evict download URI mappings targeting model_name by matching model_root."""
    target = _parse_model_cache_key(registry_resource_url, model_name)
    if target is None:
        return
    with _MODEL_RESOURCE_NAME_CACHE_LOCK:
        candidates = [p for p, t in _DOWNLOAD_URI_MODEL_MAP.items() if t == target]
        for p in candidates:
            p_root = _extract_model_storage_root(p)
            matches = p_root == model_root
            should_evict = (not matches) if keep_matching else matches
            if should_evict:
                _DOWNLOAD_URI_MODEL_MAP.pop(p, None)


def is_registered_model_storage_uri(uri: str) -> bool:
    """Return True if uri is a GCS path under a model version boundary."""
    return _parse_registered_model_gcs_path(uri) is not None


def is_registered_model_storage_or_parent_uri(uri: str) -> bool:
    """Return True if uri is a GCS path matching registered model boundaries.

    Matches version boundary (/registeredModels/<id>/versions/<vid>) or parent
    boundary (/registeredModels/<id>/versions) while excluding canonical namespace
    artifact boundaries (/runs/<id>/artifacts, /models/<id>/artifacts). Includes a
    normalized segment fallback so redundant slashes or non-CAB characters (e.g.
    quotes in euc/auto mode) cannot bypass read-only protection.
    """
    if _parse_registered_model_gcs_path(uri, include_versions_parent=True) is not None:
        return True
    if not isinstance(uri, str) or not has_gs_scheme(uri):
        return False
    try:
        parsed = urllib.parse.urlsplit(uri)
    except Exception:
        return False
    if not parsed.netloc or parsed.query or parsed.fragment:
        return False
    lower_path = parsed.path.lower()
    if any(tok in lower_path for tok in _FORBIDDEN_GCS_ENCODED_TOKENS):
        return False
    unquoted_path = urllib.parse.unquote(parsed.path)
    segments = [seg for seg in unquoted_path.split("/") if seg]
    if not segments or any(seg in (".", "..") for seg in segments):
        return False
    norm_path = "/" + "/".join(segments)
    return (
        _match_registered_model_boundary(norm_path, include_versions_parent=True)
        is not None
    )


def reset_cab_credentials_cache() -> None:
    """Clear in-memory caches for CAB credential resolution."""
    with _MODEL_RESOURCE_NAME_CACHE_LOCK:
        _MODEL_RESOURCE_NAME_CACHE.clear()
        _DOWNLOAD_URI_MODEL_MAP.clear()


def evict_cached_model_resource_url(
    registry_resource_url: str,
    model_name: str,
) -> None:
    """Evict a model's cached canonical resource name without touching download URIs."""
    cache_key = _parse_model_cache_key(registry_resource_url, model_name)
    if cache_key is None:
        return
    with _MODEL_RESOURCE_NAME_CACHE_LOCK:
        _MODEL_RESOURCE_NAME_CACHE.pop(cache_key, None)


def _find_download_uri_entries_by_model_id_locked(
    reg_url: str,
    model_id: str,
) -> list[tuple[str, tuple[str, str]]]:
    """Find (prefix, target) entries in _DOWNLOAD_URI_MODEL_MAP matching model_id.

    Caller MUST hold _MODEL_RESOURCE_NAME_CACHE_LOCK.
    """
    model_root_suffix = f"/registeredModels/{model_id}"
    matches: list[tuple[str, tuple[str, str]]] = []
    for prefix, target in _DOWNLOAD_URI_MODEL_MAP.items():
        if target[0] == reg_url:
            root = _extract_model_storage_root(prefix)
            if root and root.endswith(model_root_suffix):
                matches.append((prefix, target))
    return matches


def invalidate_registered_model_resource_url(
    registry_resource_url: str,
    model_name: str,
) -> None:
    """Evict a registered model's cached canonical resource name and download URIs."""
    cache_key = _parse_model_cache_key(registry_resource_url, model_name)
    if cache_key is None:
        return
    reg_url, target_id = cache_key
    canonical_target_url = f"{reg_url}/registeredModels/{target_id}"
    with _MODEL_RESOURCE_NAME_CACHE_LOCK:
        cached_resolved = _MODEL_RESOURCE_NAME_CACHE.get(cache_key)
        target_urls: set[str] = {canonical_target_url}
        if isinstance(cached_resolved, str) and cached_resolved:
            target_urls.add(cached_resolved)
        target_ids: set[str] = {target_id}
        if isinstance(cached_resolved, str) and "/registeredModels/" in cached_resolved:
            cand_id = cached_resolved.rsplit("/registeredModels/", 1)[-1]
            if cand_id:
                target_ids.add(cand_id)

        evicted_keys: set[tuple[str, str]] = {cache_key}
        for tid in target_ids:
            evicted_keys.add((reg_url, tid))
        for key, resolved_url in list(_MODEL_RESOURCE_NAME_CACHE.items()):
            if key[0] == reg_url and resolved_url in target_urls:
                evicted_keys.add(key)
        for tid in target_ids:
            for _, target in _find_download_uri_entries_by_model_id_locked(
                reg_url, tid
            ):
                evicted_keys.add(target)
        for key in evicted_keys:
            _MODEL_RESOURCE_NAME_CACHE.pop(key, None)
        stale_prefixes = [
            p for p, target in _DOWNLOAD_URI_MODEL_MAP.items() if target in evicted_keys
        ]
        for p in stale_prefixes:
            _DOWNLOAD_URI_MODEL_MAP.pop(p, None)


# Identifier extraction for cache migration:
# 'bare_id' permits non-CAB compliant IDs (e.g., containing dots or special characters)
# so that cache eviction and download URI migrations still clean up legacy or non-CAB
# model entries. In contrast, 'canonical_id' enforces strict CAB alphanumeric validation
# (_REGISTERED_MODEL_ID_RE) before populating _MODEL_RESOURCE_NAME_CACHE so that CAB
# credential resolution never caches an un-issuable token target.
def _parse_rename_identifiers(
    canonical_resource_id: str | None,
) -> tuple[str | None, str | None]:
    """Extract (bare_id, canonical_id) from canonical_resource_id."""
    bare_id: str | None = None
    if isinstance(canonical_resource_id, str) and canonical_resource_id:
        cand = canonical_resource_id.split("/registeredModels/")[-1]
        if cand and "/" not in cand and cand not in (".", ".."):
            bare_id = cand
    canonical_id: str | None = (
        bare_id
        if (bare_id is not None and _REGISTERED_MODEL_ID_RE.fullmatch(bare_id))
        else None
    )
    return bare_id, canonical_id


def _migrate_download_uris_locked(
    old_target: tuple[str, str],
    new_target: tuple[str, str],
    bare_id: str | None,
) -> None:
    """Migrate download URIs matching old_target or bare_id to new_target.

    Caller MUST hold _MODEL_RESOURCE_NAME_CACHE_LOCK.
    """
    matching_prefixes: set[str] = set()
    for p, target in _DOWNLOAD_URI_MODEL_MAP.items():
        if target == old_target:
            matching_prefixes.add(p)
    if bare_id is not None:
        for p, _ in _find_download_uri_entries_by_model_id_locked(
            old_target[0], bare_id
        ):
            matching_prefixes.add(p)
    for p in matching_prefixes:
        _DOWNLOAD_URI_MODEL_MAP[p] = new_target


def rename_registered_model_cache_entries(
    registry_resource_url: str,
    old_name: str,
    new_name: str,
    *,
    canonical_resource_id: str | None = None,
) -> None:
    """Update cache entries when a registered model is renamed."""
    old_target = _parse_model_cache_key(registry_resource_url, old_name)
    new_target = _parse_model_cache_key(registry_resource_url, new_name)
    if old_target is None or new_target is None:
        return
    bare_id, canonical_id = _parse_rename_identifiers(canonical_resource_id)

    explicit_url: str | None = (
        f"{old_target[0]}/registeredModels/{canonical_id}"
        if canonical_id is not None
        else None
    )
    with _MODEL_RESOURCE_NAME_CACHE_LOCK:
        if old_target == new_target:
            if bare_id is not None:
                self_target_url = f"{old_target[0]}/registeredModels/{bare_id}"
                for key, resolved_url in list(_MODEL_RESOURCE_NAME_CACHE.items()):
                    if (
                        key[0] == old_target[0]
                        and key != new_target
                        and resolved_url == self_target_url
                    ):
                        _MODEL_RESOURCE_NAME_CACHE.pop(key, None)
                _migrate_download_uris_locked(old_target, new_target, bare_id)
            if explicit_url is not None:
                _lru_put(
                    _MODEL_RESOURCE_NAME_CACHE,
                    old_target,
                    explicit_url,
                    _MAX_MODEL_RESOURCE_NAME_CACHE,
                )
            else:
                _MODEL_RESOURCE_NAME_CACHE.pop(old_target, None)
            return

        cached_old_url = _MODEL_RESOURCE_NAME_CACHE.pop(old_target, None)
        _MODEL_RESOURCE_NAME_CACHE.pop(new_target, None)

        canonical_target_url: str | None = (
            f"{old_target[0]}/registeredModels/{bare_id}"
            if bare_id is not None
            else cached_old_url
        )
        if canonical_target_url is not None:
            for key, resolved_url in list(_MODEL_RESOURCE_NAME_CACHE.items()):
                if key[0] == old_target[0] and resolved_url == canonical_target_url:
                    popped = _MODEL_RESOURCE_NAME_CACHE.pop(key, None)
                    if cached_old_url is None:
                        cached_old_url = popped

        stale_new = [
            p for p, target in _DOWNLOAD_URI_MODEL_MAP.items() if target == new_target
        ]
        for p in stale_new:
            _DOWNLOAD_URI_MODEL_MAP.pop(p, None)

        _migrate_download_uris_locked(old_target, new_target, bare_id)

        carried_url = explicit_url or cached_old_url
        if carried_url is not None:
            _lru_put(
                _MODEL_RESOURCE_NAME_CACHE,
                new_target,
                carried_url,
                _MAX_MODEL_RESOURCE_NAME_CACHE,
            )


def record_registered_model_download_uri(
    download_uri: str,
    registry_resource_url: str,
    model_name: str,
) -> None:
    """Record mapping from GCS download URI directory to registry and model name."""
    target = _parse_model_cache_key(registry_resource_url, model_name)
    if target is None:
        return
    if _parse_registered_model_gcs_path(download_uri) is None:
        return
    prefix = derive_registered_model_cab_uri(download_uri).rstrip("/")
    with _MODEL_RESOURCE_NAME_CACHE_LOCK:
        _lru_put(
            _DOWNLOAD_URI_MODEL_MAP,
            prefix,
            target,
            _MAX_MODEL_RESOURCE_NAME_CACHE,
        )


def lookup_registered_model_for_gcs_uri(
    artifact_uri: str,
) -> RegisteredModelGcsMatch | None:
    """Look up recorded registered model info for a GCS artifact URI.

    Checks bucket membership in _DOWNLOAD_URI_MODEL_MAP and substring
    presence of 'registeredModels' in the unquoted path BEFORE running full
    _validate_gcs_artifact_uri. This fast-path avoids failing on valid
    Namespace artifact paths with quote characters while preserving fail-closed
    path-traversal rejection on recorded Registry buckets.
    """
    if not has_gs_scheme(artifact_uri):
        return None
    try:
        parsed_split = urllib.parse.urlsplit(artifact_uri)
    except Exception:
        return None
    if not parsed_split.netloc:
        return None

    bucket_prefix = f"gs://{parsed_split.netloc}"
    with _MODEL_RESOURCE_NAME_CACHE_LOCK:
        if not _DOWNLOAD_URI_MODEL_MAP:
            return None
        has_bucket_match = any(
            p == bucket_prefix or p.startswith(f"{bucket_prefix}/")
            for p in _DOWNLOAD_URI_MODEL_MAP
        )
        if not has_bucket_match:
            return None

    raw_path_unquoted = urllib.parse.unquote(parsed_split.path)
    if "registeredModels" not in raw_path_unquoted:
        return None

    bucket, rel_path = _validate_gcs_artifact_uri(artifact_uri)
    path_part = f"/{rel_path}" if rel_path else ""
    if path_part and "/registeredModels" in path_part:
        idx = path_part.find("/registeredModels")
        if _has_canonical_namespace_artifacts_prefix(path_part, idx):
            return None

    cand = f"gs://{bucket}/{rel_path}" if rel_path else f"gs://{bucket}"

    with _MODEL_RESOURCE_NAME_CACHE_LOCK:
        matches = [
            prefix
            for prefix in _DOWNLOAD_URI_MODEL_MAP
            if cand == prefix or cand.startswith(f"{prefix}/")
        ]
        if matches:
            best_prefix = max(matches, key=len)
            target = _lru_get(_DOWNLOAD_URI_MODEL_MAP, best_prefix)
            if target is not None:
                return RegisteredModelGcsMatch(best_prefix, target[0], target[1])

        if rel_path and _REGISTERED_MODEL_VERSIONS_PARENT_RE.search(f"/{rel_path}"):
            child_prefixes = [
                prefix
                for prefix in _DOWNLOAD_URI_MODEL_MAP
                if prefix.startswith(f"{cand}/") and "/" not in prefix[len(cand) + 1 :]
            ]
            if child_prefixes:
                # _DOWNLOAD_URI_MODEL_MAP maintains insertion/access LRU order
                # via OrderedDict, so child_prefixes[-1] selects the most recently
                # recorded child version prefix under this parent candidate directory.
                target = _lru_get(_DOWNLOAD_URI_MODEL_MAP, child_prefixes[-1])
                if target is not None:
                    return RegisteredModelGcsMatch(cand, target[0], target[1])

        return None


def resolve_registered_model_resource_url(
    registry_resource_url: str,
    model_name: str,
    session: requests.Session | None = None,
    auth: Any = None,
) -> str:
    """Resolve an MLflow model name to its canonical Google Cloud resource URL.

    Results are cached in a thread-safe, bounded cache. Callers may invalidate
    entries on 404/403 errors via invalidate_registered_model_resource_url().
    """
    if not isinstance(model_name, str) or not model_name.strip():
        raise MlflowException(
            "Registered model name must be non-empty.",
            error_code=INVALID_PARAMETER_VALUE,
        )
    parsed_reg = parse_gcp_uri(registry_resource_url)
    if not parsed_reg.is_registry:
        raise MlflowException(
            f"Expected a model registry resource URL, got '{registry_resource_url}'.",
            error_code=INVALID_PARAMETER_VALUE,
        )

    cache_key = (parsed_reg.resource_url, model_name)
    with _MODEL_RESOURCE_NAME_CACHE_LOCK:
        cached = _lru_get(_MODEL_RESOURCE_NAME_CACHE, cache_key)
        if cached is not None:
            return cached

    owned_session: requests.Session | None = None
    if session is None:
        owned_session = create_secure_session()
        active_session = owned_session
    else:
        active_session = session

    try:
        canonical_name = _resolve_registered_model_resource_name(
            session=active_session,
            registry_resource_url=parsed_reg,
            model_name=model_name,
            auth=auth,
        )
    except MlflowException as exc:
        if getattr(exc, "error_code", None) in (
            RESOURCE_DOES_NOT_EXIST,
            ErrorCode.Name(RESOURCE_DOES_NOT_EXIST),
        ):
            invalidate_registered_model_resource_url(
                parsed_reg.resource_url, model_name
            )
        raise
    finally:
        if owned_session is not None:
            owned_session.close()

    model_id = canonical_name.split("/registeredModels/")[-1]
    resolved_url = build_gcp_resource_url(
        parsed_reg, f"{parsed_reg.resource_name}/registeredModels/{model_id}"
    )

    with _MODEL_RESOURCE_NAME_CACHE_LOCK:
        _lru_put(
            _MODEL_RESOURCE_NAME_CACHE,
            cache_key,
            resolved_url,
            _MAX_MODEL_RESOURCE_NAME_CACHE,
        )

    return resolved_url
