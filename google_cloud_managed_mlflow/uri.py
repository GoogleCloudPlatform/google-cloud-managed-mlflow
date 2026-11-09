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

"""Google Cloud URI primitives, RFC 3339 UTC helpers, and AIP-193."""

from __future__ import annotations

import datetime
import re
import urllib.parse
from typing import Any, NamedTuple

import requests
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import (
    ABORTED,
    CANCELLED,
    DATA_LOSS,
    DEADLINE_EXCEEDED,
    INTERNAL_ERROR,
    INVALID_PARAMETER_VALUE,
    INVALID_STATE,
    NOT_IMPLEMENTED,
    PERMISSION_DENIED,
    REQUEST_LIMIT_EXCEEDED,
    RESOURCE_ALREADY_EXISTS,
    RESOURCE_DOES_NOT_EXIST,
    TEMPORARILY_UNAVAILABLE,
    UNAUTHENTICATED,
    ErrorCode,
)

from google_cloud_managed_mlflow._validation import (
    _contains_control_chars,
    _EndpointOrigin,
    _is_allowed_destination,
    _parse_validated_endpoint_url,
)
from google_cloud_managed_mlflow.auth import validate_transport_security

_PATH_PATTERN = re.compile(
    r"^/(?P<api_version>v1alpha|v1)"
    r"/projects/(?P<project>[^/]+)"
    r"/locations/(?P<location>[^/]+)"
    r"/(?P<resource_type>namespaces|registries)"
    r"/(?P<resource_id>[^/]+)/?$"
)

_PROJECT_ID_PATTERN = re.compile(r"^[a-z0-9](?:[a-z0-9-.:]{0,61}[a-z0-9])?$")
_LOCATION_PATTERN = re.compile(r"^[a-z0-9-]+$")
_RESOURCE_ID_PATTERN = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")
_CUSTOM_VERB_PATTERN = re.compile(r"^[a-zA-Z][a-zA-Z0-9]*$")

_RFC3339_PATTERN = re.compile(
    r"^(\d{4})-(\d{2})-(\d{2})[Tt](\d{2}):(\d{2}):(\d{2})"
    r"(?:\.(\d+))?"
    r"(?:[Zz]|([+-]\d{2}):(\d{2}))$"
)

_FORBIDDEN_ENCODED_TOKENS = frozenset({"%2e", "%5c", "%00", "%3f", "%23", "%3a"})
_ENCODED_CONTROL_CHARS_PATTERN = re.compile(r"%(?:[01][0-9a-fA-F]|7[fF])")

_GCP_STATUS_TO_MLFLOW: dict[str, int] = {
    "INVALID_ARGUMENT": INVALID_PARAMETER_VALUE,
    "FAILED_PRECONDITION": INVALID_STATE,
    "OUT_OF_RANGE": INVALID_PARAMETER_VALUE,
    "UNAUTHENTICATED": UNAUTHENTICATED,
    "PERMISSION_DENIED": PERMISSION_DENIED,
    "NOT_FOUND": RESOURCE_DOES_NOT_EXIST,
    "ALREADY_EXISTS": RESOURCE_ALREADY_EXISTS,
    "RESOURCE_EXHAUSTED": REQUEST_LIMIT_EXCEEDED,
    "ABORTED": ABORTED,
    "UNAVAILABLE": TEMPORARILY_UNAVAILABLE,
    "DEADLINE_EXCEEDED": DEADLINE_EXCEEDED,
    "UNIMPLEMENTED": NOT_IMPLEMENTED,
    "CANCELLED": CANCELLED,
    "DATA_LOSS": DATA_LOSS,
    "INTERNAL": INTERNAL_ERROR,
}

_HTTP_STATUS_TO_MLFLOW: dict[int, int] = {
    400: INVALID_PARAMETER_VALUE,
    401: UNAUTHENTICATED,
    403: PERMISSION_DENIED,
    404: RESOURCE_DOES_NOT_EXIST,
    409: RESOURCE_ALREADY_EXISTS,
    429: REQUEST_LIMIT_EXCEEDED,
    499: CANCELLED,
    500: INTERNAL_ERROR,
    501: NOT_IMPLEMENTED,
    503: TEMPORARILY_UNAVAILABLE,
    504: DEADLINE_EXCEEDED,
}


class ParsedGcpUri(NamedTuple):
    """Parsed representation of a Google Cloud MLflow resource URI."""

    raw_uri: str
    scheme: str
    host: str
    port: int
    api_version: str
    project: str
    location: str
    resource_type: str
    resource_id: str

    @property
    def resource_name(self) -> str:
        """Return the canonical Google Cloud relative resource name."""
        return (
            f"projects/{self.project}/locations/{self.location}/"
            f"{self.resource_type}/{self.resource_id}"
        )

    @property
    def parent(self) -> str:
        """Return the parent location resource name."""
        return f"projects/{self.project}/locations/{self.location}"

    @property
    def origin(self) -> str:
        """Return the canonical scheme://host[:port] origin."""
        host_str = f"[{self.host}]" if ":" in self.host else self.host
        if (self.scheme == "https" and self.port == 443) or (
            self.scheme == "http" and self.port == 80
        ):
            return f"{self.scheme}://{host_str}"
        return f"{self.scheme}://{host_str}:{self.port}"

    @property
    def api_root(self) -> str:
        """Return the canonical API root URL (origin/api_version)."""
        return f"{self.origin}/{self.api_version}"

    @property
    def base_api_url(self) -> str:
        """Return the canonical base API URL (origin/api_version)."""
        return self.api_root

    @property
    def resource_url(self) -> str:
        """Return the canonical full resource URL."""
        return f"{self.origin}/{self.api_version}/{self.resource_name}"

    @property
    def full_endpoint_url(self) -> str:
        """Return the canonical full endpoint URL."""
        return self.resource_url

    @property
    def is_namespace(self) -> bool:
        """Return True if this URI references a tracking namespace."""
        return self.resource_type == "namespaces"

    @property
    def is_registry(self) -> bool:
        """Return True if this URI references a model registry."""
        return self.resource_type == "registries"


def parse_gcp_uri(uri: str) -> ParsedGcpUri:
    """Parse and validate a Google Cloud MLflow tracking or registry URI.

    Args:
        uri: The URI string to parse.

    Returns:
        ParsedGcpUri containing normalized URI components.

    Raises:
        MlflowException: If the URI is malformed, uses an unsupported scheme,
            violates transport security policy, attempts path traversal, or
            does not match the canonical Google Cloud resource structure.
    """
    if not uri or not isinstance(uri, str):
        raise MlflowException(
            "Google Cloud resource URI must be a non-empty string.",
            error_code=INVALID_PARAMETER_VALUE,
        )

    if _contains_control_chars(uri):
        raise MlflowException(
            "Google Cloud resource URI contains invalid control characters.",
            error_code=INVALID_PARAMETER_VALUE,
        )

    if uri[0].isspace() or uri[-1].isspace():
        raise MlflowException(
            "Google Cloud resource URI must not contain leading or trailing "
            "whitespace.",
            error_code=INVALID_PARAMETER_VALUE,
        )

    try:
        parsed = urllib.parse.urlsplit(uri)
    except ValueError as exc:
        raise MlflowException(
            "Google Cloud resource URI is malformed.",
            error_code=INVALID_PARAMETER_VALUE,
        ) from exc

    if parsed.query or parsed.fragment:
        raise MlflowException(
            "Google Cloud resource URI must not contain query parameters or fragments.",
            error_code=INVALID_PARAMETER_VALUE,
        )

    # Enforce transport security policy (scheme, host allowlist, TLS).
    validate_transport_security(uri)
    origin: _EndpointOrigin = _parse_validated_endpoint_url(uri)

    # Protect against path traversal before and after percent-decoding.
    lower_path = parsed.path.lower()
    if any(token in lower_path for token in _FORBIDDEN_ENCODED_TOKENS):
        raise MlflowException(
            "Google Cloud resource URI contains invalid encoded path characters.",
            error_code=INVALID_PARAMETER_VALUE,
        )

    if "\\" in parsed.path:
        raise MlflowException(
            "Google Cloud resource URI contains backslashes.",
            error_code=INVALID_PARAMETER_VALUE,
        )

    if "//" in parsed.path:
        raise MlflowException(
            "Google Cloud resource URI contains empty path segments.",
            error_code=INVALID_PARAMETER_VALUE,
        )

    raw_path = parsed.path.rstrip("/")

    unquoted_path = urllib.parse.unquote(raw_path)
    if "\x00" in unquoted_path or "\\" in unquoted_path or "//" in unquoted_path:
        raise MlflowException(
            "Google Cloud resource URI contains invalid decoded path characters.",
            error_code=INVALID_PARAMETER_VALUE,
        )

    for segment in unquoted_path.split("/"):
        if segment in (".", ".."):
            raise MlflowException(
                "Google Cloud resource URI contains relative path traversal segments.",
                error_code=INVALID_PARAMETER_VALUE,
            )

    match = _PATH_PATTERN.match(raw_path)
    if not match:
        raise MlflowException(
            f"Invalid Google Cloud resource URI path '{parsed.path}'. "
            "Expected format: /(v1alpha|v1)/projects/{project}/locations/{location}/"
            "{namespaces|registries}/{id}",
            error_code=INVALID_PARAMETER_VALUE,
        )

    api_version = match.group("api_version")
    project = match.group("project")
    location = match.group("location")
    resource_type = match.group("resource_type")
    resource_id = match.group("resource_id")

    if not _PROJECT_ID_PATTERN.fullmatch(project):
        raise MlflowException(
            f"Invalid Google Cloud project ID '{project}'.",
            error_code=INVALID_PARAMETER_VALUE,
        )

    if not _LOCATION_PATTERN.fullmatch(location):
        raise MlflowException(
            f"Invalid Google Cloud location '{location}'.",
            error_code=INVALID_PARAMETER_VALUE,
        )

    res_type_singular = "registry" if resource_type == "registries" else "namespace"
    if resource_id in (".", "..") or not _RESOURCE_ID_PATTERN.fullmatch(resource_id):
        raise MlflowException(
            f"Invalid Google Cloud {res_type_singular} ID '{resource_id}'.",
            error_code=INVALID_PARAMETER_VALUE,
        )

    return ParsedGcpUri(
        raw_uri=uri,
        scheme=origin.scheme,
        host=origin.hostname,
        port=origin.port,
        api_version=api_version,
        project=project,
        location=location,
        resource_type=resource_type,
        resource_id=resource_id,
    )


def is_gcp_mlflow_uri(uri: str) -> bool:
    """Return True if uri is structurally a Google Cloud MLflow URI.

    Performs structural host and path checks without raising exceptions.
    Does not validate TLS configuration, allowing callers to route URIs before
    transport enforcement occurs in parse_gcp_uri.
    """
    if not uri or not isinstance(uri, str):
        return False
    if _contains_control_chars(uri):
        return False
    if uri[0].isspace() or uri[-1].isspace():
        return False

    try:
        parsed = urllib.parse.urlsplit(uri)
        if parsed.query or parsed.fragment:
            return False
        origin = _parse_validated_endpoint_url(uri)
    except Exception:
        return False

    if not _is_allowed_destination(origin):
        return False

    lower_path = parsed.path.lower()
    if any(token in lower_path for token in _FORBIDDEN_ENCODED_TOKENS):
        return False
    if "\\" in parsed.path or "//" in parsed.path:
        return False

    raw_path = parsed.path.rstrip("/")
    match = _PATH_PATTERN.match(raw_path)
    if not match:
        return False

    project = match.group("project")
    location = match.group("location")
    resource_id = match.group("resource_id")

    if not _PROJECT_ID_PATTERN.fullmatch(project):
        return False
    if not _LOCATION_PATTERN.fullmatch(location):
        return False
    if resource_id in (".", "..") or not _RESOURCE_ID_PATTERN.fullmatch(resource_id):
        return False

    return True


def is_gcp_tracking_uri(uri: str) -> bool:
    """Return True if uri is structurally a Google Cloud MLflow tracking URI."""
    if not is_gcp_mlflow_uri(uri):
        return False
    try:
        parsed = urllib.parse.urlsplit(uri)
        raw_path = parsed.path.rstrip("/")
        match = _PATH_PATTERN.match(raw_path)
        return match is not None and match.group("resource_type") == "namespaces"
    except Exception:
        return False


def is_gcp_registry_uri(uri: str) -> bool:
    """Return True if uri is structurally a Google Cloud MLflow registry URI."""
    if not is_gcp_mlflow_uri(uri):
        return False
    try:
        parsed = urllib.parse.urlsplit(uri)
        raw_path = parsed.path.rstrip("/")
        match = _PATH_PATTERN.match(raw_path)
        return match is not None and match.group("resource_type") == "registries"
    except Exception:
        return False


def _validate_custom_verb(verb: str) -> str:
    """Validate and normalize an AIP-136 custom method verb.

    Args:
        verb: Custom method verb string (e.g. 'issueReadArtifactToken').

    Returns:
        Normalized verb without leading colon.

    Raises:
        MlflowException: If verb contains control characters, delimiters, or is
            not an AIP-136 alphanumeric identifier.
    """
    if not isinstance(verb, str):
        raise MlflowException(
            "Google Cloud API custom method verb must be a string.",
            error_code=INVALID_PARAMETER_VALUE,
        )
    if _contains_control_chars(verb):
        raise MlflowException(
            "Google Cloud API custom method verb contains invalid control characters.",
            error_code=INVALID_PARAMETER_VALUE,
        )
    if any(char in verb for char in ("?", "#", "/", "\\")):
        raise MlflowException(
            "Google Cloud API custom method verb cannot contain '?', '#', "
            "'/', or '\\\\'.",
            error_code=INVALID_PARAMETER_VALUE,
        )
    cleaned = verb.lstrip(":")
    if not _CUSTOM_VERB_PATTERN.fullmatch(cleaned):
        raise MlflowException(
            f"Google Cloud API custom method verb {verb!r} is invalid: must be an "
            "AIP-136 alphanumeric identifier (e.g. 'issueReadArtifactToken').",
            error_code=INVALID_PARAMETER_VALUE,
        )
    return cleaned


def validate_path_segments(raw_path: str, *, context: str = "suffix") -> None:
    """Validate a URL path component against traversal and delimiter attacks.

    Args:
        raw_path: The URL path component to validate.
        context: Label describing the parameter context for error messages.

    Raises:
        MlflowException: If path contains control characters, invalid encoded
            tokens, empty segments, or traversal segments.
    """
    if not isinstance(raw_path, str):
        raise MlflowException(
            f"Google Cloud API URL {context} must be a string.",
            error_code=INVALID_PARAMETER_VALUE,
        )
    if _contains_control_chars(raw_path):
        raise MlflowException(
            f"Google Cloud API URL {context} contains invalid control characters.",
            error_code=INVALID_PARAMETER_VALUE,
        )
    if "#" in raw_path:
        raise MlflowException(
            f"Google Cloud API URL {context} cannot contain URL fragment ('#').",
            error_code=INVALID_PARAMETER_VALUE,
        )
    if "\\" in raw_path:
        raise MlflowException(
            f"Google Cloud API URL {context} contains backslashes.",
            error_code=INVALID_PARAMETER_VALUE,
        )
    lower_path = raw_path.lower()
    if any(token in lower_path for token in _FORBIDDEN_ENCODED_TOKENS):
        raise MlflowException(
            f"Google Cloud API URL {context} contains invalid encoded path characters.",
            error_code=INVALID_PARAMETER_VALUE,
        )
    if _ENCODED_CONTROL_CHARS_PATTERN.search(raw_path):
        raise MlflowException(
            f"Google Cloud API URL {context} contains invalid encoded control "
            "characters.",
            error_code=INVALID_PARAMETER_VALUE,
        )

    path_part = raw_path.split("?")[0]
    # Iterative unquote check to guard against multi-layer percent-encoded traversal
    curr = path_part
    iterations = 0
    max_iterations = 10
    while "%" in curr and iterations < max_iterations:
        iterations += 1
        nxt = urllib.parse.unquote(curr)
        if nxt == curr:
            break
        curr = nxt
        lower_curr = curr.lower()
        if any(token in lower_curr for token in _FORBIDDEN_ENCODED_TOKENS):
            raise MlflowException(
                f"Google Cloud API URL {context} contains invalid encoded path "
                "characters.",
                error_code=INVALID_PARAMETER_VALUE,
            )
        if _ENCODED_CONTROL_CHARS_PATTERN.search(curr):
            raise MlflowException(
                f"Google Cloud API URL {context} contains invalid encoded control "
                "characters.",
                error_code=INVALID_PARAMETER_VALUE,
            )
        if _contains_control_chars(curr):
            raise MlflowException(
                f"Google Cloud API URL {context} contains invalid control characters.",
                error_code=INVALID_PARAMETER_VALUE,
            )
        if any(c in curr for c in ("\x00", "\\", "#", "?")):
            raise MlflowException(
                f"Google Cloud API URL {context} contains invalid decoded path "
                "characters.",
                error_code=INVALID_PARAMETER_VALUE,
            )
        if context == "resource_name":
            parts = curr.lstrip("/").split("/")
            if len(parts) >= 2 and parts[0] == "projects":
                if any(":" in seg for seg in [parts[0]] + parts[2:]):
                    raise MlflowException(
                        f"Google Cloud API URL {context} cannot contain "
                        "custom method verbs (':').",
                        error_code=INVALID_PARAMETER_VALUE,
                    )
            elif any(":" in seg for seg in parts):
                raise MlflowException(
                    f"Google Cloud API URL {context} cannot contain "
                    "custom method verbs (':').",
                    error_code=INVALID_PARAMETER_VALUE,
                )
        elif context == "subpath" and ":" in curr:
            raise MlflowException(
                f"Google Cloud API URL {context} cannot contain "
                "custom method verbs (':').",
                error_code=INVALID_PARAMETER_VALUE,
            )
        split_path = curr.replace("%2f", "/").replace("%2F", "/")
        if "//" in split_path:
            raise MlflowException(
                f"Google Cloud API URL {context} contains empty path segments ('//').",
                error_code=INVALID_PARAMETER_VALUE,
            )
        for segment in split_path.split("/"):
            if segment in (".", ".."):
                raise MlflowException(
                    f"Google Cloud API URL {context} contains relative path traversal "
                    "segments.",
                    error_code=INVALID_PARAMETER_VALUE,
                )

    if re.search(r"%[0-9a-fA-F]{2}", curr):
        raise MlflowException(
            f"Google Cloud API URL {context} contains excessively nested "
            "percent-encoding.",
            error_code=INVALID_PARAMETER_VALUE,
        )

    # Final check on fully decoded path_part
    unquoted_path = urllib.parse.unquote(path_part)
    if _contains_control_chars(unquoted_path):
        raise MlflowException(
            f"Google Cloud API URL {context} contains invalid control characters.",
            error_code=INVALID_PARAMETER_VALUE,
        )
    if any(c in unquoted_path for c in ("\x00", "\\", "#", "?")):
        raise MlflowException(
            f"Google Cloud API URL {context} contains invalid decoded path characters.",
            error_code=INVALID_PARAMETER_VALUE,
        )
    split_unquoted = unquoted_path.replace("%2f", "/").replace("%2F", "/")
    if "//" in split_unquoted:
        raise MlflowException(
            f"Google Cloud API URL {context} contains empty path segments ('//').",
            error_code=INVALID_PARAMETER_VALUE,
        )
    for segment in split_unquoted.split("/"):
        if segment in (".", ".."):
            raise MlflowException(
                f"Google Cloud API URL {context} contains relative path "
                "traversal segments.",
                error_code=INVALID_PARAMETER_VALUE,
            )


# Alias for backward compatibility
_validate_path_segments = validate_path_segments


def build_gcp_api_url(
    parsed_or_uri: ParsedGcpUri | str,
    suffix: str = "",
    *,
    verb: str | None = None,
    subpath: str | None = None,
) -> str:
    """Build a complete Google Cloud API URL from a parsed or raw URI.

    Args:
        parsed_or_uri: A ParsedGcpUri or string URI to build from.
        suffix: Optional URL suffix (custom verb starting with ':', query string
            starting with '?', root-relative path starting with 'projects/', or
            resource-relative subpath).
        verb: Optional AIP-136 custom method verb (e.g. 'issueReadArtifactToken').
        subpath: Optional REST subpath (e.g. 'api/2.0/mlflow/experiments/search').

    Returns:
        The full API URL string.
    """
    parsed = (
        parsed_or_uri
        if isinstance(parsed_or_uri, ParsedGcpUri)
        else parse_gcp_uri(parsed_or_uri)
    )

    # Separate suffix into path and query components
    if "?" in suffix:
        q_idx = suffix.index("?")
        path_suffix = suffix[:q_idx]
        query_part = suffix[q_idx:]  # includes leading '?'
    else:
        path_suffix = suffix
        query_part = ""

    if query_part:
        if _contains_control_chars(query_part):
            raise MlflowException(
                "Google Cloud API URL query string contains invalid control "
                "characters.",
                error_code=INVALID_PARAMETER_VALUE,
            )
        if "#" in query_part:
            raise MlflowException(
                "Google Cloud API URL query string cannot contain URL fragment ('#').",
                error_code=INVALID_PARAMETER_VALUE,
            )
        if subpath is not None or verb is not None:
            raise MlflowException(
                "Cannot specify 'subpath' or 'verb' when 'suffix' already contains a "
                "query string ('?').",
                error_code=INVALID_PARAMETER_VALUE,
            )

    if path_suffix:
        canonical_suffix = urllib.parse.unquote(path_suffix).strip().lstrip("/")
        if (
            canonical_suffix.startswith("projects/")
            or canonical_suffix.startswith(f"{parsed.api_version}/")
            or re.match(r"^v[0-9]+[a-zA-Z0-9]*/", canonical_suffix)
        ):
            raise MlflowException(
                "Google Cloud API URL suffix must be resource-relative and cannot "
                "start with 'projects/' or an API version prefix.",
                error_code=INVALID_PARAMETER_VALUE,
            )
        _validate_path_segments(path_suffix, context="suffix")

    path_part = path_suffix

    # Check if path_suffix already contains a custom verb
    if ":" in path_part or "%3a" in path_part.lower():
        if verb is not None:
            raise MlflowException(
                "Google Cloud API custom method verb cannot be specified both in "
                "'suffix' and 'verb' argument.",
                error_code=INVALID_PARAMETER_VALUE,
            )
        if subpath is not None:
            raise MlflowException(
                "Cannot append subpath after a custom method verb.",
                error_code=INVALID_PARAMETER_VALUE,
            )
        if "%3a" in path_part.lower():
            raise MlflowException(
                "Google Cloud API URL suffix contains invalid encoded colon ('%3a').",
                error_code=INVALID_PARAMETER_VALUE,
            )
        if path_part.count(":") > 1:
            raise MlflowException(
                "Google Cloud API URL cannot contain multiple custom method verbs.",
                error_code=INVALID_PARAMETER_VALUE,
            )
        prefix_part, existing_verb = path_part.rsplit(":", 1)
        _validate_custom_verb(existing_verb)
        if ":" in prefix_part or "%3a" in prefix_part.lower():
            raise MlflowException(
                "Google Cloud API URL cannot contain multiple custom method verbs.",
                error_code=INVALID_PARAMETER_VALUE,
            )

    if subpath is not None:
        if not isinstance(subpath, str):
            raise MlflowException(
                "Google Cloud API subpath must be a string.",
                error_code=INVALID_PARAMETER_VALUE,
            )
        if "?" in subpath or "#" in subpath:
            raise MlflowException(
                "Google Cloud API subpath cannot contain '?' or '#'.",
                error_code=INVALID_PARAMETER_VALUE,
            )
        if ":" in subpath or "%3a" in subpath.lower():
            raise MlflowException(
                "Google Cloud API subpath cannot contain custom method verbs (':'). "
                "Use the 'verb' argument instead.",
                error_code=INVALID_PARAMETER_VALUE,
            )
        canonical_sub = urllib.parse.unquote(subpath).strip().lstrip("/")
        if (
            canonical_sub.startswith("projects/")
            or canonical_sub.startswith(f"{parsed.api_version}/")
            or re.match(r"^v[0-9]+[a-zA-Z0-9]*/", canonical_sub)
        ):
            raise MlflowException(
                "Google Cloud API subpath must be resource-relative and cannot start "
                "with 'projects/' or an API version prefix.",
                error_code=INVALID_PARAMETER_VALUE,
            )
        _validate_path_segments(subpath.strip().lstrip("/"), context="subpath")
        sub_clean = subpath.strip().lstrip("/")
        if sub_clean:
            path_part = f"{path_part}/{sub_clean}" if path_part else sub_clean

    if verb is not None:
        verb_clean = _validate_custom_verb(verb)
        path_part = f"{path_part}:{verb_clean}" if path_part else f":{verb_clean}"

    base_url = f"{parsed.origin}/{parsed.api_version}/{parsed.resource_name}"
    if not path_part:
        return f"{base_url}{query_part}"

    if path_part.startswith(":"):
        return f"{base_url}{path_part}{query_part}"
    cleaned = path_part.lstrip("/")
    if not cleaned:
        return f"{base_url}{query_part}"
    if cleaned.startswith(":"):
        return f"{base_url}{cleaned}{query_part}"
    return f"{base_url}/{cleaned}{query_part}"


def build_gcp_resource_url(
    parsed_or_uri: ParsedGcpUri | str,
    resource_name: str,
    *,
    verb: str | None = None,
    subpath: str | None = None,
) -> str:
    """Build a complete Google Cloud API URL for an arbitrary sub-resource name.

    Args:
        parsed_or_uri: A ParsedGcpUri or string URI to derive origin and
            api_version from.
        resource_name: Root-relative resource name (e.g.
            projects/.../registeredModels/...).
        verb: Optional AIP-136 custom method verb.
        subpath: Optional REST subpath.

    Returns:
        The formatted API URL.
    """
    parsed = (
        parsed_or_uri
        if isinstance(parsed_or_uri, ParsedGcpUri)
        else parse_gcp_uri(parsed_or_uri)
    )
    if not isinstance(resource_name, str) or not resource_name.strip():
        raise MlflowException(
            "Google Cloud API resource name must be a non-empty string.",
            error_code=INVALID_PARAMETER_VALUE,
        )
    if "?" in resource_name or "#" in resource_name:
        raise MlflowException(
            "Google Cloud API resource name cannot contain '?' or '#'.",
            error_code=INVALID_PARAMETER_VALUE,
        )
    res_clean = resource_name.strip().lstrip("/")
    if not res_clean:
        raise MlflowException(
            "Google Cloud API resource name cannot be empty or only slashes.",
            error_code=INVALID_PARAMETER_VALUE,
        )

    # Reject custom verbs smuggled in resource_name
    if "%3a" in res_clean.lower():
        raise MlflowException(
            "Google Cloud API resource name cannot contain custom "
            "method verbs (':'). Use the 'verb' argument instead.",
            error_code=INVALID_PARAMETER_VALUE,
        )
    parts = res_clean.split("/")
    if len(parts) >= 2 and parts[0] == "projects" and ":" in parts[1]:
        if any(":" in seg for seg in [parts[0]] + parts[2:]):
            raise MlflowException(
                "Google Cloud API resource name cannot contain custom "
                "method verbs (':'). Use the 'verb' argument instead.",
                error_code=INVALID_PARAMETER_VALUE,
            )
    elif any(":" in seg for seg in parts):
        raise MlflowException(
            "Google Cloud API resource name cannot contain custom "
            "method verbs (':'). Use the 'verb' argument instead.",
            error_code=INVALID_PARAMETER_VALUE,
        )

    _validate_path_segments(res_clean, context="resource_name")

    expected_prefix = f"{parsed.resource_name}/"
    if not res_clean.startswith(expected_prefix):
        raise MlflowException(
            f"Google Cloud API resource name must belong to '{parsed.resource_name}'.",
            error_code=INVALID_PARAMETER_VALUE,
        )

    rel_resource = res_clean[len(expected_prefix) :]
    rel_segments = rel_resource.split("/")
    if any(not seg for seg in rel_segments) or len(rel_segments) < 2:
        raise MlflowException(
            "Google Cloud API resource name must include a valid collection and "
            f"resource ID under '{expected_prefix}'.",
            error_code=INVALID_PARAMETER_VALUE,
        )

    res_url = f"{parsed.origin}/{parsed.api_version}/{res_clean}"

    if subpath is not None:
        if not isinstance(subpath, str):
            raise MlflowException(
                "Google Cloud API subpath must be a string.",
                error_code=INVALID_PARAMETER_VALUE,
            )
        if "?" in subpath or "#" in subpath:
            raise MlflowException(
                "Google Cloud API subpath cannot contain '?' or '#'.",
                error_code=INVALID_PARAMETER_VALUE,
            )
        if ":" in subpath or "%3a" in subpath.lower():
            raise MlflowException(
                "Google Cloud API subpath cannot contain custom method verbs (':'). "
                "Use the 'verb' argument instead.",
                error_code=INVALID_PARAMETER_VALUE,
            )
        canonical_sub = urllib.parse.unquote(subpath).strip().lstrip("/")
        if (
            canonical_sub.startswith("projects/")
            or canonical_sub.startswith(f"{parsed.api_version}/")
            or re.match(r"^v[0-9]+[a-zA-Z0-9]*/", canonical_sub)
        ):
            raise MlflowException(
                "Google Cloud API subpath must be resource-relative and cannot start "
                "with 'projects/' or an API version prefix.",
                error_code=INVALID_PARAMETER_VALUE,
            )
        sub_clean = subpath.strip().lstrip("/")
        _validate_path_segments(sub_clean, context="subpath")
        if sub_clean:
            res_url = f"{res_url}/{sub_clean}"

    if verb is not None:
        verb_clean = _validate_custom_verb(verb)
        res_url = f"{res_url}:{verb_clean}"

    return res_url


class InvalidTimestampError(MlflowException, ValueError):
    """Raised when an RFC 3339 timestamp is invalid or malformed."""

    def __init__(self, message: str) -> None:
        base_init: Any = super().__init__
        base_init(message, error_code=INVALID_PARAMETER_VALUE)


def parse_rfc3339_utc(ts: str) -> datetime.datetime:
    """Parse an RFC 3339 timestamp string into a naive UTC datetime.

    Note:
        Returns a naive datetime in UTC (tzinfo=None) to match Google Auth
        and MLflow entity expiry contracts. Use parse_rfc3339_ms() when converting
        to epoch milliseconds to avoid local timezone interpretation by naive
        .timestamp().

        A custom regex parser is used instead of datetime.fromisoformat() because
        Python 3.10 lacks support for the trailing 'Z' shorthand and sub-microsecond
        (nanosecond) precision strings commonly returned by Google Cloud APIs.

    Args:
        ts: RFC 3339 formatted timestamp string.

    Returns:
        Naive datetime object representing UTC time.

    Raises:
        InvalidTimestampError: If the timestamp is invalid or malformed.
    """
    if not ts or not isinstance(ts, str):
        raise InvalidTimestampError("RFC 3339 timestamp must be a non-empty string.")

    match = _RFC3339_PATTERN.match(ts)
    if not match:
        raise InvalidTimestampError(f"Invalid RFC 3339 timestamp '{ts}'.")

    try:
        year = int(match.group(1))
        month = int(match.group(2))
        day = int(match.group(3))
        hour = int(match.group(4))
        minute = int(match.group(5))
        second = int(match.group(6))
        if second == 60:
            second = 59

        frac_str = match.group(7)
        microsecond = 0
        if frac_str:
            microsecond = int(frac_str[:6].ljust(6, "0"))

        tzinfo: datetime.tzinfo = datetime.timezone.utc
        raw_tz_h = match.group(8)
        raw_tz_m = match.group(9)
        if raw_tz_h is not None and raw_tz_m is not None:
            tz_sign = -1 if raw_tz_h.startswith("-") else 1
            tz_h = int(raw_tz_h.lstrip("+-"))
            tz_m = int(raw_tz_m)
            if tz_h > 23 or tz_m > 59:
                raise ValueError(f"Timezone offset out of range: {raw_tz_h}:{raw_tz_m}")
            tz_offset = datetime.timedelta(minutes=tz_sign * (tz_h * 60 + tz_m))
            tzinfo = datetime.timezone(tz_offset)

        dt = datetime.datetime(
            year, month, day, hour, minute, second, microsecond, tzinfo=tzinfo
        )
    except (ValueError, OverflowError) as exc:
        raise InvalidTimestampError(
            f"Invalid RFC 3339 timestamp '{ts}': {exc}"
        ) from exc

    return dt.astimezone(datetime.timezone.utc).replace(tzinfo=None)


def parse_rfc3339_ms(ts: str) -> int:
    """Parse an RFC 3339 timestamp string into integer milliseconds since Unix epoch."""
    utc_dt = parse_rfc3339_utc(ts)
    diff = utc_dt - datetime.datetime(1970, 1, 1)
    total_seconds = diff.days * 86400 + diff.seconds
    return total_seconds * 1000 + (diff.microseconds // 1000)


def ms_to_rfc3339(epoch_ms: int) -> str:
    """Convert integer epoch ms to an RFC 3339 UTC string with trailing 'Z'."""
    if isinstance(epoch_ms, bool) or not isinstance(epoch_ms, int):
        raise InvalidTimestampError(
            f"Epoch milliseconds must be an integer; got {type(epoch_ms).__name__}."
        )
    sec, ms = divmod(epoch_ms, 1000)
    try:
        dt = datetime.datetime(1970, 1, 1) + datetime.timedelta(seconds=sec)
        return f"{dt.strftime('%Y-%m-%dT%H:%M:%S')}.{ms:03d}Z"
    except (OverflowError, ValueError) as exc:
        raise InvalidTimestampError(
            f"Epoch milliseconds '{epoch_ms}' out of valid range: {exc}"
        ) from exc


class GcpMlflowApiError(MlflowException):
    """Exception representing an API error from Google Cloud Managed MLflow."""

    def __init__(
        self,
        message: str,
        *,
        http_status: int,
        gcp_status: str | None = None,
        error_details: dict[str, Any] | None = None,
        error_code: int = INTERNAL_ERROR,
        error_info_reason: str | None = None,
    ) -> None:
        base_init: Any = super().__init__
        base_init(message, error_code=error_code)
        self.message = message
        self.api_message = message
        self.http_status = http_status
        self.gcp_status = gcp_status
        self.error_details = error_details
        self.error_info_reason = error_info_reason


def map_aip193_status_to_error_code(
    gcp_status: str | None, http_status: int = 500
) -> int:
    """Map a Google Cloud RPC status string or HTTP status to an MLflow error code."""
    if gcp_status:
        normalized = gcp_status.strip().upper()
        if normalized in _GCP_STATUS_TO_MLFLOW:
            return _GCP_STATUS_TO_MLFLOW[normalized]
        try:
            return int(ErrorCode.Value(normalized))
        except (ValueError, TypeError):
            pass
    return _HTTP_STATUS_TO_MLFLOW.get(http_status, INTERNAL_ERROR)


_CANONICAL_ERROR_INFO_TYPES = frozenset(
    {"type.googleapis.com/google.rpc.ErrorInfo", "google.rpc.ErrorInfo"}
)


def _is_matching_error_info_domain(domain: str) -> bool:
    """Return True if domain matches Managed MLflow service endpoints."""
    d = domain.strip().lower()
    if not d.endswith(".googleapis.com"):
        return False
    parts = d.split(".")
    return parts[0] == "mlflow" or parts[0].endswith("-mlflow")


def _extract_mlflow_error_info_reason(
    error_details: dict[str, Any] | None,
) -> str | None:
    """Extract reason string from ErrorInfo details matching Managed MLflow domain."""
    if not isinstance(error_details, dict):
        return None
    raw_details = error_details.get("details")
    if not isinstance(raw_details, list):
        return None
    for detail in raw_details:
        if not isinstance(detail, dict):
            continue
        type_str = detail.get("@type")
        domain_str = detail.get("domain")
        if (
            type_str in _CANONICAL_ERROR_INFO_TYPES
            and isinstance(domain_str, str)
            and _is_matching_error_info_domain(domain_str)
        ):
            reason = detail.get("reason")
            if isinstance(reason, str) and reason.strip():
                return reason.strip()
    return None


def raise_for_aip193_error(response: requests.Response) -> None:
    """Inspect HTTP response and raise GcpMlflowApiError if status code is >= 400.

    Translates Google Cloud AIP-193 error envelopes, OSS MLflow error envelopes,
    and HTTP status codes into structured GcpMlflowApiError exceptions.
    """
    if response.status_code < 400:
        return

    payload: Any = None
    try:
        payload = response.json()
    except Exception:
        payload = None

    http_status = response.status_code
    gcp_status: str | None = None
    error_details: dict[str, Any] | None = None
    message: str | None = None
    error_info_reason: str | None = None

    if isinstance(payload, dict):
        if isinstance(payload.get("error"), dict):
            error_details = payload["error"]
            raw_status = error_details.get("status")
            if isinstance(raw_status, str):
                gcp_status = raw_status.strip().upper()
            raw_message = error_details.get("message")
            if raw_message is not None:
                message = str(raw_message)
        elif "error_code" in payload:
            raw_error_code = payload["error_code"]
            if isinstance(raw_error_code, str):
                gcp_status = raw_error_code.strip().upper()
            raw_message = payload.get("message")
            if raw_message is not None:
                message = str(raw_message)

    if not message:
        raw_text = getattr(response, "text", "") or ""
        message = (
            (raw_text[:500] + "...")
            if len(raw_text) > 500
            else (raw_text or f"HTTP {http_status}")
        )

    error_code = map_aip193_status_to_error_code(gcp_status, http_status)

    error_info_reason = _extract_mlflow_error_info_reason(error_details)
    if error_info_reason:
        resolved_code = getattr(ErrorCode, error_info_reason, None)
        if isinstance(resolved_code, int) and not isinstance(resolved_code, bool):
            error_code = resolved_code

    raise GcpMlflowApiError(
        message,
        http_status=http_status,
        gcp_status=gcp_status,
        error_details=error_details,
        error_code=error_code,
        error_info_reason=error_info_reason,
    )
