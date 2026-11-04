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
from typing import Any

import google.auth.credentials
import requests
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import (
    INTERNAL_ERROR,
    INVALID_PARAMETER_VALUE,
    RESOURCE_DOES_NOT_EXIST,
)

from google_cloud_managed_mlflow.auth import (
    GoogleCloudBearerAuth,
    create_secure_session,
)
from google_cloud_managed_mlflow.uri import (
    build_gcp_resource_url,
    parse_gcp_uri,
    parse_rfc3339_utc,
    raise_for_aip193_error,
)

_EARLY_EXPIRY_BUFFER = datetime.timedelta(seconds=300)
_UNSAFE_PATH_CHARS: frozenset[str] = frozenset("'\"\\#?\u2028\u2029")
_RESOURCE_ID_RE = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")
_CANONICAL_ARTIFACTS_BOUNDARY_RE = re.compile(
    r"/(?:runs/[a-zA-Z0-9_-]{1,64}|models/[a-zA-Z0-9_-]{1,64}|[0-9a-f]{32})/artifacts(?=/|$)"
)
_MAX_MODEL_SEARCH_PAGES: int = 100


class CabAccessMode(str, enum.Enum):
    """Access mode for Credential Access Boundary (CAB) tokens."""

    READ_ONLY = "READ_ONLY"
    READ_WRITE = "READ_WRITE"


class CabResourceKind(str, enum.Enum):
    """Resource kind for Credential Access Boundary (CAB) tokens."""

    NAMESPACE = "NAMESPACE"
    # Reserved for model registry store integration; not yet wired into artifact I/O.
    REGISTERED_MODEL = "REGISTERED_MODEL"


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
    trimmed = artifact_path.rstrip("/")
    if not trimmed or "//" in trimmed:
        raise MlflowException(
            f"Invalid artifact_path '{artifact_path}': "
            "empty path segments are not allowed.",
            error_code=INVALID_PARAMETER_VALUE,
        )
    segments = trimmed.split("/")
    if any(seg in ("", ".", "..") for seg in segments):
        raise MlflowException(
            f"Invalid artifact_path '{artifact_path}': "
            "path traversal segments are not allowed.",
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
    rel_path = raw_path.strip("/")
    if rel_path:
        if "//" in raw_path.strip("/"):
            raise MlflowException(
                f"Invalid GCS artifact URI '{artifact_uri}': "
                "empty path segments are not allowed.",
                error_code=INVALID_PARAMETER_VALUE,
            )
        segments = rel_path.split("/")
        if any(seg in ("", ".", "..") for seg in segments):
            raise MlflowException(
                f"Invalid GCS artifact URI '{artifact_uri}': "
                "path traversal segments are not allowed.",
                error_code=INVALID_PARAMETER_VALUE,
            )

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

    if not artifact_uri.endswith("/") and rel_path.rsplit("/", 1)[-1] == "MLmodel":
        if "/" not in rel_path:
            return f"gs://{bucket}"
        return f"gs://{bucket}/{rel_path.rsplit('/', 1)[0]}"

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
        if not _RESOURCE_ID_RE.fullmatch(model_seg):
            raise MlflowException(
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
    ) -> None:
        super_init: Any = super().__init__
        super_init()

        validated_url = _validate_cab_resource_url(resource_url, resource_kind)

        if resource_kind == CabResourceKind.REGISTERED_MODEL:
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

        self._lock = threading.Lock()
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
    def resource_url(self) -> str:
        """Return the canonical resource URL."""
        return self._resource_url

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
        return CabTokenCredentials(
            resource_url=self._resource_url,
            resource_kind=self._resource_kind,
            access_mode=access_mode,
            artifact_uri=self._artifact_uri,
            run_id=self._run_id,
            model_id=self._model_id,
            artifact_path=self._artifact_path,
            session=self._session if self._custom_session else None,
            auth=self._auth,
        )

    def elevate_to_read_write(self) -> bool:
        """Elevate access mode to READ_WRITE and invalidate the cached token.

        Returns:
            True if this call changed the access mode, False if it was already
            READ_WRITE.
        """
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
        with self._lock:
            if self._session is not None and not self._custom_session:
                self._session.close()
                self._session = None

    def refresh(self, request: Any = None) -> None:
        """Acquire a new CAB token via single-flight thread-safe execution."""
        gen_before = self._refresh_generation
        with self._lock:
            if self._refresh_generation != gen_before and self._is_valid_locked():
                return
            self._refresh_locked()

    def _refresh_locked(self) -> None:
        """Execute the CAB issueToken RPC (caller holds _lock)."""
        verb = (
            "issueReadWriteArtifactToken"
            if self._access_mode == CabAccessMode.READ_WRITE
            else "issueReadArtifactToken"
        )
        url = f"{self._resource_url}:{verb}"

        payload: dict[str, Any] = {}
        if self._artifact_uri is not None:
            payload["artifactUri"] = self._artifact_uri
        elif self._run_id is not None:
            payload["runId"] = self._run_id
        elif self._model_id is not None:
            payload["modelId"] = self._model_id

        if self._artifact_path is not None:
            payload["artifactPath"] = self._artifact_path

        session = self._session
        if session is None:
            session = create_secure_session()
            self._session = session
            self._custom_session = False
            weakref.finalize(self, _close_session, self._session)

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
            self._effective_buffer = min(_EARLY_EXPIRY_BUFFER, ttl / 2)
        else:
            self._effective_buffer = datetime.timedelta(0)

        self.token = token
        self.expiry = parsed_expiry
        self._refresh_generation += 1

    def __repr__(self) -> str:
        with self._lock:
            token_status = "set" if self.token is not None else "unset"
            mode = self._access_mode.value
            kind = self._resource_kind.value
        return (
            f"<CabTokenCredentials resource_kind={kind} "
            f"access_mode={mode} token={token_status}>"
        )


def _resolve_registered_model_resource_name(
    session: requests.Session,
    registry_resource_url: str,
    model_name: str,
    auth: Any = None,
) -> str:
    """Resolve an MLflow model name to its canonical resource name.

    Queries SearchRegisteredModels with the filter name = '<model_name>',
    then returns the canonical resource name of the first result whose
    displayName (or display_name, or name) equals model_name exactly.

    Args:
        session: Active HTTP session for making the request.
        registry_resource_url: Full URL of the model registry resource.
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

    parsed_reg = parse_gcp_uri(registry_resource_url)
    if not parsed_reg.is_registry:
        raise MlflowException(
            f"Expected a model registry resource URL, got '{registry_resource_url}'.",
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
    expected_res_re = re.compile(
        rf"^projects/[a-zA-Z0-9._:-]+/locations/{re.escape(parsed_reg.location)}"
        rf"/registries/{re.escape(parsed_reg.resource_id)}"
        r"/registeredModels/[a-zA-Z0-9_-]{1,64}$"
    )

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
            if disp_name == model_name or model.get("name") == model_name:
                res_name = model.get("name")
                if not isinstance(res_name, str) or not res_name:
                    raise MlflowException(
                        "Invalid SearchRegisteredModels response: matched registered "
                        "model has no 'name'.",
                        error_code=INTERNAL_ERROR,
                    )
                if not expected_res_re.fullmatch(res_name):
                    raise MlflowException(
                        "Invalid SearchRegisteredModels response: resource name "
                        f"'{res_name}' does not match expected pattern.",
                        error_code=INTERNAL_ERROR,
                    )
                return res_name

        page_token = _json_field(data, "nextPageToken", "next_page_token")
        if not page_token or not isinstance(page_token, str):
            break

    raise MlflowException(
        f"Registered model '{model_name}' not found in registry.",
        error_code=RESOURCE_DOES_NOT_EXIST,
    )
