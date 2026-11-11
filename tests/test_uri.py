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

"""Unit tests for Google Cloud URI primitives, RFC 3339 UTC, and AIP-193."""

from __future__ import annotations

import contextlib
import datetime
import json
from collections.abc import Iterator
from typing import Any

import pytest
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
    ErrorCode,
)

from google_cloud_managed_mlflow.auth import GoogleCloudBearerAuth
from google_cloud_managed_mlflow.uri import (
    GcpMlflowApiError,
    InvalidTimestampError,
    ParsedGcpUri,
    build_gcp_api_url,
    build_gcp_resource_url,
    is_gcp_mlflow_uri,
    is_gcp_registry_uri,
    is_gcp_tracking_uri,
    map_aip193_status_to_error_code,
    ms_to_rfc3339,
    parse_gcp_uri,
    parse_rfc3339_ms,
    parse_rfc3339_utc,
    raise_for_aip193_error,
)

try:
    from mlflow.protos.databricks_pb2 import UNAUTHENTICATED
except ImportError:
    from mlflow.protos.databricks_pb2 import (
        CUSTOMER_UNAUTHORIZED as UNAUTHENTICATED,
    )


@contextlib.contextmanager
def _raises_invalid_param(
    match: str | None = None,
) -> Iterator[pytest.ExceptionInfo[MlflowException]]:
    """Assert the enclosed block raises MlflowException(INVALID_PARAMETER_VALUE)."""
    __tracebackhide__ = True
    with pytest.raises(MlflowException) as exc_info:
        yield exc_info
    assert exc_info.value.error_code == ErrorCode.Name(INVALID_PARAMETER_VALUE)
    if match is not None:
        assert match in str(exc_info.value)


def _make_response(
    status_code: int,
    payload: Any = None,
    raw_text: str | None = None,
    content_type: str = "application/json",
) -> requests.Response:
    """Construct an isolated requests.Response without network I/O."""
    resp = requests.Response()
    resp.status_code = status_code
    resp.headers["Content-Type"] = content_type
    if raw_text is not None:
        resp._content = raw_text.encode("utf-8")
    elif payload is not None:
        resp._content = json.dumps(payload).encode("utf-8")
    else:
        resp._content = b""
    return resp


@pytest.fixture(autouse=True)
def _clean_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure test isolation across environment variables and auth state."""
    GoogleCloudBearerAuth.reset_for_testing()
    monkeypatch.delenv("MLFLOW_GCP_ENDPOINT_OVERRIDE", raising=False)
    monkeypatch.delenv("MLFLOW_TRACKING_INSECURE_TLS", raising=False)


# ============================================================================
# 1. ParsedGcpUri properties
# ============================================================================


def test_parsed_gcp_uri_fields_and_properties() -> None:
    """Verify ParsedGcpUri properties for tracking namespaces and registries."""
    uri_ns = ParsedGcpUri(
        raw_uri="https://mlflow.googleapis.com/v1alpha/projects/my-proj/locations/us-central1/namespaces/default",
        scheme="https",
        host="mlflow.googleapis.com",
        port=443,
        api_version="v1alpha",
        project="my-proj",
        location="us-central1",
        resource_type="namespaces",
        resource_id="default",
    )
    assert (
        uri_ns.resource_name
        == "projects/my-proj/locations/us-central1/namespaces/default"
    )
    assert uri_ns.parent == "projects/my-proj/locations/us-central1"
    assert uri_ns.origin == "https://mlflow.googleapis.com"
    assert uri_ns.api_root == "https://mlflow.googleapis.com/v1alpha"
    assert (
        uri_ns.resource_url
        == "https://mlflow.googleapis.com/v1alpha/projects/my-proj/locations/us-central1/namespaces/default"
    )
    assert uri_ns.is_namespace is True
    assert uri_ns.is_registry is False

    uri_reg = ParsedGcpUri(
        raw_uri="https://mlflow.googleapis.com/v1/projects/my-proj/locations/us-central1/registries/prod-models",
        scheme="https",
        host="mlflow.googleapis.com",
        port=443,
        api_version="v1",
        project="my-proj",
        location="us-central1",
        resource_type="registries",
        resource_id="prod-models",
    )
    assert (
        uri_reg.resource_name
        == "projects/my-proj/locations/us-central1/registries/prod-models"
    )
    assert uri_reg.parent == "projects/my-proj/locations/us-central1"
    assert uri_reg.api_root == "https://mlflow.googleapis.com/v1"
    assert uri_reg.is_namespace is False
    assert uri_reg.is_registry is True


@pytest.mark.parametrize(
    ("scheme", "host", "port", "expected_origin"),
    [
        ("https", "mlflow.googleapis.com", 443, "https://mlflow.googleapis.com"),
        ("http", "localhost", 80, "http://localhost"),
        ("https", "mlflow.googleapis.com", 8443, "https://mlflow.googleapis.com:8443"),
        ("http", "127.0.0.1", 8080, "http://127.0.0.1:8080"),
        ("http", "::1", 8080, "http://[::1]:8080"),
        ("http", "::1", 80, "http://[::1]"),
        ("https", "::1", 443, "https://[::1]"),
        ("https", "::1", 8443, "https://[::1]:8443"),
        ("http", "2001:db8::1", 9000, "http://[2001:db8::1]:9000"),
    ],
)
def test_parsed_gcp_uri_origin_formatting(
    scheme: str, host: str, port: int, expected_origin: str
) -> None:
    """Verify origin formatting handles default, custom, and IPv6 ports."""
    parsed = ParsedGcpUri("", scheme, host, port, "", "", "", "", "")
    assert parsed.origin == expected_origin


# ============================================================================
# 2. parse_gcp_uri() validation
# ============================================================================


@pytest.mark.parametrize(
    ("raw_uri", "expected_fields"),
    [
        (
            "https://mlflow.googleapis.com/v1alpha/projects/my-proj/locations/us-central1/namespaces/default",
            {
                "scheme": "https",
                "host": "mlflow.googleapis.com",
                "port": 443,
                "api_version": "v1alpha",
                "project": "my-proj",
                "location": "us-central1",
                "resource_type": "namespaces",
                "resource_id": "default",
                "is_namespace": True,
                "is_registry": False,
            },
        ),
        (
            "https://mlflow.googleapis.com/v1/projects/my-proj/locations/us-central1/registries/prod-models",
            {
                "scheme": "https",
                "host": "mlflow.googleapis.com",
                "port": 443,
                "api_version": "v1",
                "project": "my-proj",
                "location": "us-central1",
                "resource_type": "registries",
                "resource_id": "prod-models",
                "is_namespace": False,
                "is_registry": True,
            },
        ),
        (
            "https://asia-east1-mlflow.googleapis.com/v1alpha/projects/test-project/locations/asia-east1/namespaces/demo-ns",
            {
                "host": "asia-east1-mlflow.googleapis.com",
                "location": "asia-east1",
                "resource_id": "demo-ns",
            },
        ),
        (
            "https://europe-west1-mlflow.googleapis.com/v1alpha/projects/dsml-testing/locations/europe-west1/namespaces/demo_test",
            {
                "host": "europe-west1-mlflow.googleapis.com",
                "location": "europe-west1",
                "resource_id": "demo_test",
            },
        ),
    ],
)
def test_parse_gcp_uri_valid_https(
    raw_uri: str, expected_fields: dict[str, Any]
) -> None:
    """Verify parse_gcp_uri parses valid production and regional Google Cloud URIs."""
    parsed = parse_gcp_uri(raw_uri)
    for field, expected_val in expected_fields.items():
        assert getattr(parsed, field) == expected_val


@pytest.mark.parametrize(
    ("uri", "expected_project"),
    [
        (
            "https://mlflow.googleapis.com/v1alpha/projects/example.com:my-proj/locations/us-central1/namespaces/default",
            "example.com:my-proj",
        ),
        (
            "https://mlflow.googleapis.com/v1alpha/projects/domain.co.uk:proj-123/locations/europe-west1/registries/reg-1",
            "domain.co.uk:proj-123",
        ),
    ],
)
def test_parse_gcp_uri_domain_scoped_projects(uri: str, expected_project: str) -> None:
    """Verify parse_gcp_uri supports domain-scoped project IDs."""
    parsed = parse_gcp_uri(uri)
    assert parsed.project == expected_project


def test_parse_gcp_uri_trailing_slash_tolerance() -> None:
    """Verify parse_gcp_uri tolerates trailing slashes and normalizes them."""
    uri = "https://mlflow.googleapis.com/v1/projects/p/locations/l/namespaces/n/"
    parsed = parse_gcp_uri(uri)
    assert parsed.project == "p"
    assert parsed.location == "l"
    assert parsed.resource_type == "namespaces"
    assert parsed.resource_id == "n"
    assert parsed.resource_name == "projects/p/locations/l/namespaces/n"
    assert (
        parsed.resource_url
        == "https://mlflow.googleapis.com/v1/projects/p/locations/l/namespaces/n"
    )
    assert is_gcp_mlflow_uri(uri) is True
    assert is_gcp_tracking_uri(uri) is True
    assert is_gcp_registry_uri(uri) is False


def test_parse_gcp_uri_resource_id_validation() -> None:
    """Verify resource ID validation enforces 64-char limit and no dots."""
    id_64 = "a" * 64
    parsed = parse_gcp_uri(
        f"https://mlflow.googleapis.com/v1/projects/p/locations/l/namespaces/{id_64}"
    )
    assert parsed.resource_id == id_64

    id_65 = "a" * 65
    with _raises_invalid_param("Invalid Google Cloud namespace ID"):
        parse_gcp_uri(
            f"https://mlflow.googleapis.com/v1/projects/p/locations/l/namespaces/{id_65}"
        )

    with _raises_invalid_param("Invalid Google Cloud namespace ID"):
        parse_gcp_uri(
            "https://mlflow.googleapis.com/v1/projects/p/locations/l/namespaces/name.with.dot"
        )


@pytest.mark.parametrize(
    ("override", "uri", "expected_host", "expected_port", "expected_origin"),
    [
        (
            "http://localhost:8080",
            "http://localhost:8080/v1alpha/projects/p/locations/l/namespaces/n",
            "localhost",
            8080,
            "http://localhost:8080",
        ),
        (
            "http://127.0.0.1:8080",
            "http://127.0.0.1:8080/v1alpha/projects/p/locations/l/namespaces/n",
            "127.0.0.1",
            8080,
            "http://127.0.0.1:8080",
        ),
        (
            "http://[::1]:8080",
            "http://[::1]:8080/v1alpha/projects/p/locations/l/namespaces/n",
            "::1",
            8080,
            "http://[::1]:8080",
        ),
        (
            "http://127.0.0.1",
            "http://127.0.0.1/v1alpha/projects/p/locations/l/namespaces/n",
            "127.0.0.1",
            80,
            "http://127.0.0.1",
        ),
    ],
)
def test_parse_gcp_uri_loopback_http_with_override(
    monkeypatch: pytest.MonkeyPatch,
    override: str,
    uri: str,
    expected_host: str,
    expected_port: int,
    expected_origin: str,
) -> None:
    """Verify loopback http endpoints succeed when opted into via override."""
    monkeypatch.setenv("MLFLOW_GCP_ENDPOINT_OVERRIDE", override)
    parsed = parse_gcp_uri(uri)
    assert parsed.scheme == "http"
    assert parsed.host == expected_host
    assert parsed.port == expected_port
    assert parsed.origin == expected_origin


@pytest.mark.parametrize(
    ("flag",),
    [("true",), ("1",), ("yes",), ("True",), ("TRUE",)],
)
def test_parse_gcp_uri_insecure_tls_flag_rejected(
    monkeypatch: pytest.MonkeyPatch, flag: str
) -> None:
    """Verify MLFLOW_TRACKING_INSECURE_TLS triggers immediate fail-closed rejection."""
    monkeypatch.setenv("MLFLOW_TRACKING_INSECURE_TLS", flag)
    uri = "https://mlflow.googleapis.com/v1alpha/projects/p/locations/l/namespaces/n"
    with _raises_invalid_param("Disabling TLS verification is prohibited"):
        parse_gcp_uri(uri)


@pytest.mark.parametrize(
    ("bad_uri", "expected_match"),
    [
        (
            "http://mlflow.googleapis.com/v1alpha/projects/p/locations/l/namespaces/n",
            "Insecure HTTP is only permitted on loopback",
        ),
        (
            "http://remote-server.com/v1alpha/projects/p/locations/l/namespaces/n",
            "Insecure HTTP is only permitted on loopback",
        ),
        (
            "https://example.com/v1alpha/projects/p/locations/l/namespaces/n",
            "Refusing to send Google Cloud credentials to untrusted origin",
        ),
        (
            "https://evil.com/v1alpha/projects/p/locations/l/namespaces/n",
            "Refusing to send Google Cloud credentials to untrusted origin",
        ),
        ("gcp://projects/p/locations/l/namespaces/n", "Invalid scheme 'gcp'"),
        (
            "gcp:projects/p/locations/l/namespaces/n",
            "Endpoint URL must include a valid scheme and host",
        ),
        (
            "google-cloud://projects/p/locations/l/namespaces/n",
            "Invalid scheme 'google-cloud'",
        ),
        (
            "file://localhost/v1alpha/projects/p/locations/l/namespaces/n",
            "Invalid scheme 'file'",
        ),
        (
            "file:///v1alpha/projects/p/locations/l/namespaces/n",
            "Endpoint URL must include a valid scheme and host",
        ),
    ],
)
def test_parse_gcp_uri_transport_and_origin_rejections(
    bad_uri: str, expected_match: str
) -> None:
    """Verify parse_gcp_uri fails closed on plaintext HTTP and untrusted origins."""
    with _raises_invalid_param(expected_match):
        parse_gcp_uri(bad_uri)


@pytest.mark.parametrize(
    ("traversal_uri", "match"),
    [
        (
            "https://mlflow.googleapis.com/v1alpha/projects/p/locations/l/namespaces/../namespaces/n",
            None,
        ),
        (
            "https://mlflow.googleapis.com/v1alpha/projects/p/locations/l/namespaces/./n",
            None,
        ),
        (
            "https://mlflow.googleapis.com//v1alpha/projects/p/locations/l/namespaces/n",
            "empty path segments",
        ),
        (
            "https://mlflow.googleapis.com/v1alpha//projects/p/locations/l/namespaces/n",
            "empty path segments",
        ),
        (
            "https://mlflow.googleapis.com/v1alpha/projects/p/locations/l/namespaces/n//",
            "empty path segments",
        ),
        (
            "https://mlflow.googleapis.com/v1alpha/projects/p/locations/l\\namespaces/n",
            "backslashes",
        ),
    ],
)
def test_parse_gcp_uri_path_traversal_rejections(
    traversal_uri: str, match: str | None
) -> None:
    """Verify parse_gcp_uri strictly rejects raw path traversal."""
    with _raises_invalid_param(match):
        parse_gcp_uri(traversal_uri)


@pytest.mark.parametrize(
    ("token",),
    [
        ("%2e",),
        ("%2E",),
        ("%2e%2e",),
        ("%2E%2E",),
        ("%5c",),
        ("%5C",),
        ("%00",),
    ],
)
def test_parse_gcp_uri_encoded_traversal_tokens(token: str) -> None:
    """Verify parse_gcp_uri strictly rejects percent-encoded traversal tokens."""
    uri = f"https://mlflow.googleapis.com/v1alpha/projects/p/locations/l/namespaces/{token}"
    with _raises_invalid_param("invalid encoded path characters"):
        parse_gcp_uri(uri)


@pytest.mark.parametrize(
    ("bad_uri", "match"),
    [
        (
            "https://mlflow.googleapis.com/v1alpha/projects/p/locations/l/namespaces/n?key=val",
            "query parameters or fragments",
        ),
        (
            "https://mlflow.googleapis.com/v1alpha/projects/p/locations/l/namespaces/n#frag",
            "query parameters or fragments",
        ),
        (
            " https://mlflow.googleapis.com/v1alpha/projects/p/locations/l/namespaces/n",
            "whitespace",
        ),
        (
            "https://mlflow.googleapis.com/v1alpha/projects/p/locations/"
            "l/namespaces/n ",
            "whitespace",
        ),
        ("", "non-empty string"),
    ],
)
def test_parse_gcp_uri_query_fragment_whitespace_rejections(
    bad_uri: str, match: str
) -> None:
    """Verify parse_gcp_uri rejects queries, fragments, whitespace, and empty."""
    with _raises_invalid_param(match):
        parse_gcp_uri(bad_uri)


@pytest.mark.parametrize(
    ("ch",),
    [("\t",), ("\r",), ("\n",), ("\x00",), ("\x1f",), ("\x7f",)],
)
def test_parse_gcp_uri_control_chars_rejections(ch: str) -> None:
    """Verify parse_gcp_uri rejects control characters in URI strings."""
    uri = (
        f"https://mlflow.googleapis.com/v1alpha/projects/p/locations/l/namespaces/{ch}n"
    )
    with _raises_invalid_param("control characters"):
        parse_gcp_uri(uri)


@pytest.mark.parametrize(
    ("malformed_uri", "match"),
    [
        (
            "https://mlflow.googleapis.com/v1alpha/projects/p/locations/l/experiments/e",
            "Invalid Google Cloud resource URI path",
        ),
        (
            "https://mlflow.googleapis.com/v1alpha/projects/p/locations/l/runs/r",
            "Invalid Google Cloud resource URI path",
        ),
        (
            "https://mlflow.googleapis.com/v1alpha/projects/p/locations/l/namespaces",
            "Invalid Google Cloud resource URI path",
        ),
        (
            "https://mlflow.googleapis.com/v1alpha/projects/INVALID_PROJECT/locations/l/namespaces/n",
            "Invalid Google Cloud project ID",
        ),
        (
            "https://mlflow.googleapis.com/v1alpha/projects/-bad-lead/locations/l/namespaces/n",
            "Invalid Google Cloud project ID",
        ),
        (
            "https://mlflow.googleapis.com/v1alpha/projects/my-proj/locations/US_CENTRAL1/namespaces/n",
            "Invalid Google Cloud location",
        ),
        (
            "https://mlflow.googleapis.com/v1alpha/projects/my-proj/locations/us-central1/namespaces/bad@id",
            "Invalid Google Cloud namespace ID",
        ),
        (
            "https://mlflow.googleapis.com/v1alpha/projects/my-proj/locations/us-central1/registries/bad@id",
            "Invalid Google Cloud registry ID",
        ),
        (
            "https://mlflow.googleapis.com/v1alpha/projects/my-proj/locations/us-central1/namespaces/bad$id",
            "Invalid Google Cloud namespace ID",
        ),
    ],
)
def test_parse_gcp_uri_malformed_structure_rejections(
    malformed_uri: str, match: str
) -> None:
    """Verify parse_gcp_uri rejects unsupported types and invalid identifiers."""
    with _raises_invalid_param(match):
        parse_gcp_uri(malformed_uri)


# ============================================================================
# 3. Structural recognition helpers
# ============================================================================


def test_is_gcp_mlflow_uri_structural_checks() -> None:
    """Verify is_gcp_mlflow_uri returns True for valid GCP URIs."""
    ns_uri = (
        "https://mlflow.googleapis.com/v1alpha/projects/my-proj/locations/"
        "us-central1/namespaces/default"
    )
    reg_uri = (
        "https://mlflow.googleapis.com/v1/projects/my-proj/locations/"
        "us-central1/registries/prod-models"
    )
    assert is_gcp_mlflow_uri(ns_uri) is True
    assert is_gcp_mlflow_uri(reg_uri) is True


@pytest.mark.parametrize(
    ("non_gcp_uri",),
    [
        (None,),
        ("",),
        ("   ",),
        (12345,),
        ("sqlite:///mlflow.db",),
        ("file:///tmp/mlruns",),
        ("databricks",),
        ("databricks-uc",),
        ("https://mlflow.example.com/api/2.0/mlflow/experiments/list",),
        ("http://localhost:5000",),
        ("http://mlflow.googleapis.com/v1alpha/projects/p/locations/l/namespaces/n",),
        (
            "https://mlflow.googleapis.com/v1alpha/projects/p/locations/l/namespaces/n?q=1",
        ),
        (
            "https://mlflow.googleapis.com/v1alpha/projects/p/locations/l/namespaces/n#frag",
        ),
        (
            "https://mlflow.googleapis.com/v1alpha/projects/INVALID_PROJ/locations/us-central1/namespaces/default",
        ),
        (
            "https://mlflow.googleapis.com/v1alpha/projects/my-proj/locations/INVALID_LOC/namespaces/default",
        ),
        (
            "https://mlflow.googleapis.com/v1alpha/projects/my-proj/locations/us-central1/namespaces/bad@id",
        ),
        (
            "https://mlflow.googleapis.com/v1alpha/projects/my-proj/locations/us-central1/namespaces/..",
        ),
        (
            "https://mlflow.googleapis.com/v1alpha/projects/my-proj/locations/us-central1/namespaces/\tdef",
        ),
        (
            " https://mlflow.googleapis.com/v1alpha/projects/my-proj/locations/"
            "us-central1/namespaces/default",
        ),
        (
            "https://mlflow.googleapis.com/v1alpha/projects/my-proj/locations/"
            "us-central1/namespaces/default ",
        ),
        (
            "https://mlflow.googleapis.com/v1alpha/projects/my-proj/locations/"
            "us-central1/namespaces/foo%00bar",
        ),
        (
            "https://mlflow.googleapis.com/v1alpha/projects/my-proj/locations/"
            "us-central1/namespaces/foo\\bar",
        ),
        ("https://mlflow.googleapis.com//v1alpha/projects/p/locations/l/namespaces/n",),
        ("https://mlflow.googleapis.com/v1alpha/projects/p/locations/l/experiments/e",),
    ],
)
def test_is_gcp_mlflow_uri_non_gcp_returns_false(non_gcp_uri: Any) -> None:
    """Verify is_gcp_mlflow_uri returns False for non-GCP URIs without raising."""
    assert is_gcp_mlflow_uri(non_gcp_uri) is False


def test_is_gcp_tracking_and_registry_uri() -> None:
    """Verify tracking and registry uri helpers discriminate resource types."""
    ns_uri = (
        "https://mlflow.googleapis.com/v1alpha/projects/my-proj/locations/"
        "us-central1/namespaces/default"
    )
    reg_uri = (
        "https://mlflow.googleapis.com/v1/projects/my-proj/locations/"
        "us-central1/registries/prod-models"
    )

    assert is_gcp_tracking_uri(ns_uri) is True
    assert is_gcp_tracking_uri(reg_uri) is False

    assert is_gcp_registry_uri(reg_uri) is True
    assert is_gcp_registry_uri(ns_uri) is False

    assert is_gcp_tracking_uri("databricks") is False
    assert is_gcp_registry_uri("databricks") is False


def test_structural_helpers_transport_independence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify structural helpers succeed even when parse_gcp_uri fails closed on TLS."""
    uri = (
        "https://mlflow.googleapis.com/v1alpha/projects/my-proj/locations/"
        "us-central1/namespaces/default"
    )
    monkeypatch.setenv("MLFLOW_TRACKING_INSECURE_TLS", "true")

    # Structural recognition returns True
    assert is_gcp_mlflow_uri(uri) is True
    assert is_gcp_tracking_uri(uri) is True
    assert is_gcp_registry_uri(uri) is False

    # parse_gcp_uri enforces transport policy and fails closed
    with _raises_invalid_param("Disabling TLS verification is prohibited"):
        parse_gcp_uri(uri)


# ============================================================================
# 4. build_gcp_api_url() URL construction
# ============================================================================


@pytest.mark.parametrize(
    ("suffix", "expected_suffix"),
    [
        ("", ""),
        ("/", ""),
        (":issueReadArtifactToken", ":issueReadArtifactToken"),
        ("/:issueReadArtifactToken", ":issueReadArtifactToken"),
        ("?max_results=10&page_token=abc", "?max_results=10&page_token=abc"),
        ("/?max_results=10", "?max_results=10"),
    ],
)
def test_build_gcp_api_url_custom_verbs_and_query(
    suffix: str, expected_suffix: str
) -> None:
    """Verify build_gcp_api_url constructs correct URLs for verbs and queries."""
    uri = (
        "https://mlflow.googleapis.com/v1alpha/projects/my-proj/locations/"
        "us-central1/namespaces/default"
    )
    expected = (
        f"https://mlflow.googleapis.com/v1alpha/projects/my-proj/locations/"
        f"us-central1/namespaces/default{expected_suffix}"
    )
    assert build_gcp_api_url(uri, suffix) == expected


def test_build_gcp_api_url_keyword_args() -> None:
    """Verify build_gcp_api_url supports verb and subpath keyword arguments."""
    uri = (
        "https://mlflow.googleapis.com/v1alpha/projects/my-proj/locations/"
        "us-central1/namespaces/default"
    )
    assert (
        build_gcp_api_url(uri, verb="issueReadArtifactToken")
        == f"{uri}:issueReadArtifactToken"
    )
    assert (
        build_gcp_api_url(uri, subpath="api/2.0/mlflow/experiments/search")
        == f"{uri}/api/2.0/mlflow/experiments/search"
    )


def test_build_gcp_resource_url() -> None:
    """Verify build_gcp_resource_url builds URLs for arbitrary sub-resources."""
    reg_uri = (
        "https://mlflow.googleapis.com/v1alpha/projects/p/locations/l/registries/r"
    )
    model_res = "projects/p/locations/l/registries/r/registeredModels/m"
    expected = (
        f"https://mlflow.googleapis.com/v1alpha/{model_res}:issueReadArtifactToken"
    )
    assert (
        build_gcp_resource_url(reg_uri, model_res, verb="issueReadArtifactToken")
        == expected
    )


def test_build_gcp_resource_url_encoded_slash_in_model_name() -> None:
    """Verify build_gcp_resource_url supports model names with encoded slash."""
    reg_uri = "https://mlflow.googleapis.com/v1/projects/p/locations/l/registries/r"
    resource_name = "projects/p/locations/l/registries/r/registeredModels/my%2Fmodel"
    url = build_gcp_resource_url(reg_uri, resource_name)
    assert url == (
        "https://mlflow.googleapis.com/v1/projects/p/locations/l/registries/r/"
        "registeredModels/my%2Fmodel"
    )


def test_build_gcp_resource_url_domain_scoped_project_and_encoded_slash() -> None:
    """Verify build_gcp_resource_url supports domain-scoped project with encoded
    slash.
    """
    reg_uri = (
        "https://mlflow.googleapis.com/v1/projects/example.com:my-proj/"
        "locations/us-central1/registries/r1"
    )
    resource_name = (
        "projects/example.com:my-proj/locations/us-central1/registries/r1/"
        "registeredModels/my%2Fmodel"
    )
    url = build_gcp_resource_url(reg_uri, resource_name)
    assert url == (
        "https://mlflow.googleapis.com/v1/projects/example.com:my-proj/"
        "locations/us-central1/registries/r1/registeredModels/my%2Fmodel"
    )


def test_build_gcp_resource_url_encoded_percent_in_model_name() -> None:
    """Verify build_gcp_resource_url supports model name with encoded percent (%25)."""
    reg_uri = "https://mlflow.googleapis.com/v1/projects/p1/locations/us-central1/registries/r1"
    resource_name = (
        "projects/p1/locations/us-central1/registries/r1/registeredModels/my%25model"
    )
    url = build_gcp_resource_url(reg_uri, resource_name)
    assert url == (
        "https://mlflow.googleapis.com/v1/projects/p1/locations/us-central1/"
        "registries/r1/registeredModels/my%25model"
    )


def test_build_gcp_url_subpath_segment_starting_with_dots() -> None:
    """Verify subpaths starting with '..' are allowed if not traversal, while '..' is
    rejected.
    """
    reg_uri = "https://mlflow.googleapis.com/v1/projects/p/locations/l/registries/r"
    url = build_gcp_api_url(reg_uri, subpath="..backup/file.txt")
    assert url == f"{reg_uri}/..backup/file.txt"

    res_url = build_gcp_resource_url(
        reg_uri,
        "projects/p/locations/l/registries/r/registeredModels/m",
        subpath="..backup/file.txt",
    )
    assert res_url == f"{reg_uri}/registeredModels/m/..backup/file.txt"

    with _raises_invalid_param():
        build_gcp_api_url(reg_uri, subpath="../file.txt")

    with _raises_invalid_param():
        build_gcp_api_url(reg_uri, subpath="dir/../file.txt")

    with _raises_invalid_param():
        build_gcp_resource_url(
            reg_uri,
            "projects/p/locations/l/registries/r/registeredModels/m",
            subpath="../file.txt",
        )

    with _raises_invalid_param():
        build_gcp_resource_url(
            reg_uri,
            "projects/p/locations/l/registries/r/registeredModels/m",
            subpath="dir/../file.txt",
        )


@pytest.mark.parametrize(
    ("sub",),
    [
        (
            "projects/my-proj/locations/us-central1/registeredModels/my-model:issueReadArtifactToken",
        ),
        (
            "/projects/my-proj/locations/us-central1/registeredModels/my-model:issueReadArtifactToken",
        ),
        (
            "v1alpha/projects/my-proj/locations/us-central1/registeredModels/my-model:issueReadArtifactToken",
        ),
        (
            "/v1alpha/projects/my-proj/locations/us-central1/registeredModels/my-model:issueReadArtifactToken",
        ),
    ],
)
def test_build_gcp_api_url_root_relative_projects(sub: str) -> None:
    """Verify root-relative projects/... suffixes are rejected to prevent escape."""
    uri = (
        "https://mlflow.googleapis.com/v1alpha/projects/my-proj/locations/"
        "us-central1/namespaces/default"
    )
    with _raises_invalid_param():
        build_gcp_api_url(uri, sub)


def test_build_gcp_api_url_accepts_parsed_or_str() -> None:
    """Verify build_gcp_api_url accepts both ParsedGcpUri and string input."""
    raw = (
        "https://mlflow.googleapis.com/v1alpha/projects/my-proj/locations/"
        "us-central1/namespaces/default"
    )
    parsed = parse_gcp_uri(raw)
    assert build_gcp_api_url(raw, ":ping") == build_gcp_api_url(parsed, ":ping")


@pytest.mark.parametrize(
    ("bad_suffix",),
    [
        ("/../x",),
        ("/./x",),
        ("%2e%2e/x",),
        ("%2e/x",),
        (r"\foo",),
        ("%00evil",),
        ("%252e%252e/x",),
        ("%252f../x",),
        ("%255cfoo",),
        ("foo#frag",),
        ("..#",),
        ("foo\tbar",),
        ("foo\nbar",),
    ],
)
def test_build_gcp_api_url_path_traversal_rejections(bad_suffix: str) -> None:
    """Verify build_gcp_api_url rejects path traversal in suffix."""
    uri = (
        "https://mlflow.googleapis.com/v1alpha/projects/my-proj/locations/"
        "us-central1/namespaces/default"
    )
    with _raises_invalid_param():
        build_gcp_api_url(uri, bad_suffix)


@pytest.mark.parametrize(
    ("valid_verb", "expected_suffix"),
    [
        ("issueReadArtifactToken", ":issueReadArtifactToken"),
        (":issueReadArtifactToken", ":issueReadArtifactToken"),
        ("issueWriteArtifactToken", ":issueWriteArtifactToken"),
        ("customVerb123", ":customVerb123"),
        ("get", ":get"),
    ],
)
def test_build_gcp_api_url_custom_verb_valid(
    valid_verb: str, expected_suffix: str
) -> None:
    """Verify valid AIP-136 verbs format properly on build_gcp_api_url."""
    uri = (
        "https://mlflow.googleapis.com/v1alpha/projects/my-proj/locations/"
        "us-central1/namespaces/default"
    )
    assert build_gcp_api_url(uri, verb=valid_verb) == f"{uri}{expected_suffix}"


@pytest.mark.parametrize(
    ("bad_verb",),
    [
        ("../evil",),
        ("/evil",),
        ("bad:verb",),
        ("bad-verb",),
        ("issueReadArtifactToken?injected=true",),
        ("issueReadArtifactToken#frag",),
        ("issueRead\tArtifactToken",),
        ("issueRead\nArtifactToken",),
        (r"evil\verb",),
        ("",),
        (":",),
        ("123bad",),
    ],
)
def test_build_gcp_api_url_custom_verb_rejections(bad_verb: str) -> None:
    """Verify build_gcp_api_url rejects invalid AIP-136 verbs."""
    uri = (
        "https://mlflow.googleapis.com/v1alpha/projects/my-proj/locations/"
        "us-central1/namespaces/default"
    )
    with _raises_invalid_param():
        build_gcp_api_url(uri, verb=bad_verb)


@pytest.mark.parametrize(
    ("bad_subpath",),
    [
        ("projects/other-project/locations/us-central1/namespaces/other",),
        ("/projects/other-project/locations/us-central1/namespaces/other",),
        ("search?injected=true",),
        ("search#frag",),
        ("..#",),
        ("../traversal",),
        ("%2e%2e/traversal",),
        ("sub\tpath",),
        (r"sub\path",),
    ],
)
def test_build_gcp_api_url_subpath_rejections(bad_subpath: str) -> None:
    """Verify build_gcp_api_url rejects invalid or escaping subpaths."""
    uri = (
        "https://mlflow.googleapis.com/v1alpha/projects/my-proj/locations/"
        "us-central1/namespaces/default"
    )
    with _raises_invalid_param():
        build_gcp_api_url(uri, subpath=bad_subpath)


@pytest.mark.parametrize(
    ("bad_resource_name", "verb"),
    [
        # Verb override via query injection in resource_name
        (
            (
                "projects/p/locations/l/registries/r/registeredModels/m:"
                "issueWriteArtifactToken?"
            ),
            "issueReadArtifactToken",
        ),
        # Fragment injection in resource_name
        (
            (
                "projects/p/locations/l/registries/r/registeredModels/m:"
                "issueWriteArtifactToken#"
            ),
            "issueReadArtifactToken",
        ),
        # Traversal in resource_name
        (
            ("projects/p/locations/l/registries/r/registeredModels/../other/models/m"),
            None,
        ),
        # Cross-project resource_name
        (
            "projects/attacker-proj/locations/l/registries/r/registeredModels/m",
            None,
        ),
        # Cross-location resource_name
        (
            "projects/p/locations/attacker-loc/registries/r/registeredModels/m",
            None,
        ),
        # Empty or whitespace resource_name
        ("", None),
        ("   ", None),
        ("/", None),
        ("//", None),
        # Control characters in resource_name
        ("projects/p/locations/l/registries/r/\tregisteredModels/m", None),
        # Custom method verb smuggled in resource_name
        (
            "projects/p/locations/l/registries/r/registeredModels/m:issueWriteArtifactToken",
            None,
        ),
        (":issueWriteArtifactToken", None),
        # Sibling registry or namespace escaping bound scope
        (
            "projects/p/locations/l/registries/other_r/registeredModels/m",
            None,
        ),
        ("projects/p/locations/l/namespaces/n/experiments/1", None),
        # Empty path segments and missing child segments
        ("projects/p/locations/l/registries/r//", None),
        ("projects/p/locations/l/registries/r/registeredModels/", None),
        ("projects/p/locations/l/registries/r/registeredModels", None),
        ("projects/p/locations/l//", None),
        ("projects/p/locations/l/registries/", None),
        # Percent-encoded colons in resource_name
        (
            "projects/p/locations/l/registries/r/registeredModels/m%3aissueWriteArtifactToken",
            None,
        ),
        (
            "projects/p/locations/l/registries/r/registeredModels/m%253aissueWriteArtifactToken",
            None,
        ),
        (
            "projects/example.com:my-proj/locations/l/registries/r/registeredModels/m%3aissueWriteArtifactToken",
            None,
        ),
    ],
)
def test_build_gcp_resource_url_security_validations(
    bad_resource_name: str, verb: str | None
) -> None:
    """Verify build_gcp_resource_url rejects injection & traversal in resource_name."""
    reg_uri = (
        "https://mlflow.googleapis.com/v1alpha/projects/p/locations/l/registries/r"
    )
    with _raises_invalid_param():
        build_gcp_resource_url(reg_uri, bad_resource_name, verb=verb)


@pytest.mark.parametrize(
    ("bad_suffix", "verb", "subpath"),
    [
        # Query injection in suffix with subpath or verb
        ("?pageToken=abc", "search", "experiments"),
        ("?pageToken=abc", "search", None),
        ("?pageToken=abc", None, "experiments"),
        # Custom verb in suffix with verb or subpath
        (":ping", "pong", None),
        (":ping", None, "experiments"),
        # Fragment or control characters in query part
        ("?foo#bar", None, None),
        ("?foo\tbar", None, None),
        # Custom verb smuggled in subpath
        ("", None, ":issueReadArtifactToken"),
        ("", None, "artifacts:download"),
        ("", "download", "artifacts:sub"),
        # Encoded colon custom verb smuggling
        ("models/m%3aevilVerb:get", None, None),
        ("models/m%253aevilVerb:get", None, None),
        (":customVerb%3aextra", None, None),
        # Encoded delimiter and traversal in suffix
        ("experiments/..%3f/foo", None, None),
        ("experiments/..%23/foo", None, None),
        ("experiments/1%3ffoo=bar", None, None),
        ("experiments/1%23fragment", None, None),
        ("experiments/%252525252e%252525252e", None, None),
        # API version prefix or projects/ escape in subpath
        ("", None, "v1alpha/projects/other/locations/l/namespaces/n"),
        ("", None, "projects/other/locations/l/namespaces/n"),
        ("", None, "%2fprojects%2fother%2flocations%2fl"),
        ("", None, "projects%2fother%2flocations%2fl"),
        # Cross-project or cross-location root-relative suffix
        (
            "projects/attacker-proj/locations/us-central1/namespaces/default",
            None,
            None,
        ),
        (
            "projects/my-proj/locations/europe-west1/namespaces/default",
            None,
            None,
        ),
    ],
)
def test_build_gcp_api_url_query_verb_and_cross_project_rejections(
    bad_suffix: str, verb: str | None, subpath: str | None
) -> None:
    """Verify build_gcp_api_url rejects query/verb mixing and cross-project scopes."""
    uri = (
        "https://mlflow.googleapis.com/v1alpha/projects/my-proj/locations/"
        "us-central1/namespaces/default"
    )
    with _raises_invalid_param():
        build_gcp_api_url(uri, bad_suffix, verb=verb, subpath=subpath)


# ============================================================================
# 5. RFC 3339 UTC helpers
# ============================================================================


@pytest.mark.parametrize(
    ("ts_str", "expected_dt"),
    [
        ("2026-11-03T12:30:45Z", datetime.datetime(2026, 11, 3, 12, 30, 45)),
        ("2026-11-03t12:30:45z", datetime.datetime(2026, 11, 3, 12, 30, 45)),
        (
            "2026-11-03T12:30:45.123456Z",
            datetime.datetime(2026, 11, 3, 12, 30, 45, 123456),
        ),
        (
            "2026-11-03T12:30:45.5Z",
            datetime.datetime(2026, 11, 3, 12, 30, 45, 500000),
        ),
        (
            "2026-11-03T12:30:45.123456789Z",
            datetime.datetime(2026, 11, 3, 12, 30, 45, 123456),
        ),
        (
            "2026-11-03T18:00:45+05:30",
            datetime.datetime(2026, 11, 3, 12, 30, 45),
        ),
        (
            "2026-11-03T05:30:45-07:00",
            datetime.datetime(2026, 11, 3, 12, 30, 45),
        ),
    ],
)
def test_parse_rfc3339_utc_formats(ts_str: str, expected_dt: datetime.datetime) -> None:
    """Verify parse_rfc3339_utc parses timestamps to naive UTC."""
    dt = parse_rfc3339_utc(ts_str)
    assert dt == expected_dt
    assert dt.tzinfo is None


def test_parse_rfc3339_utc_leap_second() -> None:
    """Verify parse_rfc3339_utc clamps leap second 60 to 59."""
    dt = parse_rfc3339_utc("2026-12-31T23:59:60Z")
    assert dt == datetime.datetime(2026, 12, 31, 23, 59, 59)


@pytest.mark.parametrize(
    ("bad_ts",),
    [
        ("2026-11-03T12:30:45",),
        ("2026-11-03T12:30:45.123",),
        ("2026-11-03T12:30:45+25:00",),
        ("2026-11-03T12:30:45+05:60",),
        ("2026-02-30T12:30:45Z",),
        ("2026-13-01T12:30:45Z",),
        ("2026-11-03T25:00:00Z",),
        ("",),
        ("not-a-timestamp",),
    ],
)
def test_parse_rfc3339_utc_rejections(bad_ts: str) -> None:
    """Verify parse_rfc3339_utc rejects naive timestamps and malformed inputs."""
    with pytest.raises((InvalidTimestampError, ValueError)):
        parse_rfc3339_utc(bad_ts)


@pytest.mark.parametrize(
    ("ts_str", "expected_ms"),
    [
        ("1970-01-01T00:00:00.000Z", 0),
        ("1970-01-01T00:00:00Z", 0),
        ("1970-01-01T00:00:01.000Z", 1000),
        ("1970-01-01T00:00:00.123Z", 123),
        ("1969-12-31T23:59:59.999Z", -1),
        ("1969-12-31T23:59:50.000Z", -10000),
        ("9999-12-31T23:59:59.999Z", 253402300799999),
    ],
)
def test_parse_rfc3339_ms_epoch_and_negative(ts_str: str, expected_ms: int) -> None:
    """Verify parse_rfc3339_ms parses Unix epoch, pre-1970, and future timestamps."""
    assert parse_rfc3339_ms(ts_str) == expected_ms


@pytest.mark.parametrize(
    ("ms",),
    [
        (0,),
        (1,),
        (-1,),
        (1000,),
        (-1000,),
        (123456789,),
        (-123456789,),
        (1730668800000,),
        (-500000000123,),
    ],
)
def test_ms_to_rfc3339_lossless_roundtrip(ms: int) -> None:
    """Verify lossless roundtrip between ms_to_rfc3339 and parse_rfc3339_ms."""
    encoded = ms_to_rfc3339(ms)
    decoded = parse_rfc3339_ms(encoded)
    assert decoded == ms, f"Failed roundtrip for ms={ms}: encoded as {encoded}"


@pytest.mark.parametrize(
    ("bad_val",),
    [(True,), (False,), ("1000",), (12.34,), (None,), ([100],), (10**25,)],
)
def test_ms_to_rfc3339_type_rejections(bad_val: Any) -> None:
    """Verify ms_to_rfc3339 rejects non-integer types, booleans, and overflow values."""
    with pytest.raises((InvalidTimestampError, MlflowException, ValueError)):
        ms_to_rfc3339(bad_val)


# ============================================================================
# 6. AIP-193 Error mapping
# ============================================================================


def test_gcp_mlflow_api_error_properties() -> None:
    """Verify GcpMlflowApiError preserves http_status, gcp_status, and details."""
    err = GcpMlflowApiError(
        "Resource not found",
        http_status=404,
        gcp_status="NOT_FOUND",
        error_details={"@type": "type.googleapis.com/google.rpc.ErrorInfo"},
        error_code=RESOURCE_DOES_NOT_EXIST,
    )
    assert isinstance(err, MlflowException)
    assert err.http_status == 404
    assert err.gcp_status == "NOT_FOUND"
    assert err.error_details == {"@type": "type.googleapis.com/google.rpc.ErrorInfo"}
    assert err.api_message == "Resource not found"
    assert err.message == "Resource not found"
    assert err.error_code == ErrorCode.Name(RESOURCE_DOES_NOT_EXIST)
    assert "Resource not found" in str(err)


@pytest.mark.parametrize(
    ("gcp_status", "expected_code"),
    [
        ("INVALID_ARGUMENT", INVALID_PARAMETER_VALUE),
        ("FAILED_PRECONDITION", INVALID_STATE),
        ("OUT_OF_RANGE", INVALID_PARAMETER_VALUE),
        ("UNAUTHENTICATED", UNAUTHENTICATED),
        ("PERMISSION_DENIED", PERMISSION_DENIED),
        ("NOT_FOUND", RESOURCE_DOES_NOT_EXIST),
        ("ALREADY_EXISTS", RESOURCE_ALREADY_EXISTS),
        ("RESOURCE_EXHAUSTED", REQUEST_LIMIT_EXCEEDED),
        ("ABORTED", ABORTED),
        ("UNAVAILABLE", TEMPORARILY_UNAVAILABLE),
        ("DEADLINE_EXCEEDED", DEADLINE_EXCEEDED),
        ("UNIMPLEMENTED", NOT_IMPLEMENTED),
        ("CANCELLED", CANCELLED),
        ("DATA_LOSS", DATA_LOSS),
        ("INTERNAL", INTERNAL_ERROR),
    ],
)
def test_map_aip193_status_canonical_codes(gcp_status: str, expected_code: int) -> None:
    """Verify map_aip193_status_to_error_code across Google RPC status codes."""
    assert map_aip193_status_to_error_code(gcp_status) == expected_code
    assert map_aip193_status_to_error_code(gcp_status.lower()) == expected_code


@pytest.mark.parametrize(
    ("http_code", "expected_code"),
    [
        (400, INVALID_PARAMETER_VALUE),
        (401, UNAUTHENTICATED),
        (403, PERMISSION_DENIED),
        (404, RESOURCE_DOES_NOT_EXIST),
        (409, RESOURCE_ALREADY_EXISTS),
        (429, REQUEST_LIMIT_EXCEEDED),
        (499, CANCELLED),
        (500, INTERNAL_ERROR),
        (501, NOT_IMPLEMENTED),
        (503, TEMPORARILY_UNAVAILABLE),
        (504, DEADLINE_EXCEEDED),
        (502, INTERNAL_ERROR),
    ],
)
def test_map_aip193_status_http_fallbacks(http_code: int, expected_code: int) -> None:
    """Verify map_aip193_status_to_error_code falls back to HTTP status code."""
    assert map_aip193_status_to_error_code(None, http_code) == expected_code


def test_raise_for_aip193_error_2xx_noop() -> None:
    """Verify raise_for_aip193_error is a no-op on HTTP 2xx success responses."""
    for code in (200, 201, 204):
        resp = _make_response(code, payload={"experiments": []})
        raise_for_aip193_error(resp)


@pytest.mark.parametrize(
    ("http_status", "gcp_status", "message", "expected_error_code"),
    [
        (
            404,
            "NOT_FOUND",
            "Namespace 'default' was not found.",
            RESOURCE_DOES_NOT_EXIST,
        ),
        (
            409,
            "ABORTED",
            "Concurrent modification conflict.",
            ABORTED,
        ),
        (
            501,
            "UNIMPLEMENTED",
            "Method is not implemented.",
            NOT_IMPLEMENTED,
        ),
    ],
)
def test_raise_for_aip193_error_google_rpc_envelopes(
    http_status: int,
    gcp_status: str,
    message: str,
    expected_error_code: int,
) -> None:
    """Verify raise_for_aip193_error maps Google RPC error envelopes."""
    resp = _make_response(
        http_status,
        payload={
            "error": {
                "code": http_status,
                "message": message,
                "status": gcp_status,
            }
        },
    )
    with pytest.raises(GcpMlflowApiError) as exc_info:
        raise_for_aip193_error(resp)
    assert exc_info.value.http_status == http_status
    assert exc_info.value.gcp_status == gcp_status
    assert exc_info.value.error_code == ErrorCode.Name(expected_error_code)
    assert message in str(exc_info.value)


def test_raise_for_aip193_error_oss_mlflow_envelopes() -> None:
    """Verify raise_for_aip193_error maps top-level OSS MLflow error envelopes."""
    resp = _make_response(
        400,
        payload={
            "error_code": "INVALID_PARAMETER_VALUE",
            "message": "Invalid run_id specified.",
        },
    )
    with pytest.raises(GcpMlflowApiError) as exc_info:
        raise_for_aip193_error(resp)
    assert exc_info.value.http_status == 400
    assert exc_info.value.gcp_status == "INVALID_PARAMETER_VALUE"
    assert exc_info.value.error_code == ErrorCode.Name(INVALID_PARAMETER_VALUE)
    assert "Invalid run_id specified." in str(exc_info.value)


def test_raise_for_aip193_error_html_and_non_dict_payloads() -> None:
    """Verify raise_for_aip193_error handles HTML error pages and non-dict JSON."""
    # HTML 502 Bad Gateway
    html_502 = _make_response(
        502,
        raw_text="<html><body>502 Bad Gateway</body></html>",
        content_type="text/html",
    )
    with pytest.raises(GcpMlflowApiError) as exc_502:
        raise_for_aip193_error(html_502)
    assert exc_502.value.http_status == 502
    assert exc_502.value.error_code == ErrorCode.Name(INTERNAL_ERROR)
    assert "502 Bad Gateway" in str(exc_502.value)

    # Non-dict JSON (array)
    array_resp = _make_response(500, payload=["unexpected", "error", "list"])
    with pytest.raises(GcpMlflowApiError) as exc_array:
        raise_for_aip193_error(array_resp)
    assert exc_array.value.http_status == 500
    assert exc_array.value.error_code == ErrorCode.Name(INTERNAL_ERROR)

    # Empty body
    empty_resp = _make_response(503, raw_text="")
    with pytest.raises(GcpMlflowApiError) as exc_empty:
        raise_for_aip193_error(empty_resp)
    assert exc_empty.value.http_status == 503
    assert exc_empty.value.error_code == ErrorCode.Name(TEMPORARILY_UNAVAILABLE)
    assert "HTTP 503" in str(exc_empty.value)


def test_raise_for_aip193_error_html_truncation() -> None:
    """Verify raise_for_aip193_error truncates large HTML error bodies to 500 chars."""
    body_2000 = "<html><body>" + ("x" * 2000) + "</body></html>"
    resp = _make_response(500, raw_text=body_2000, content_type="text/html")
    with pytest.raises(GcpMlflowApiError) as exc_info:
        raise_for_aip193_error(resp)
    assert exc_info.value.http_status == 500
    assert len(exc_info.value.message) == 503
    assert exc_info.value.message.endswith("...")
    assert exc_info.value.message == body_2000[:500] + "..."


def test_raise_for_aip193_error_resource_exhausted_429() -> None:
    """Verify HTTP 429 and RESOURCE_EXHAUSTED map to REQUEST_LIMIT_EXCEEDED."""
    resp = _make_response(
        429,
        payload={
            "error": {
                "code": 429,
                "message": "Quota exceeded.",
                "status": "RESOURCE_EXHAUSTED",
            }
        },
    )
    with pytest.raises(GcpMlflowApiError) as exc_info:
        raise_for_aip193_error(resp)
    assert exc_info.value.http_status == 429
    assert exc_info.value.gcp_status == "RESOURCE_EXHAUSTED"
    assert exc_info.value.error_code == ErrorCode.Name(REQUEST_LIMIT_EXCEEDED)


def test_raise_for_aip193_error_info_reason_extracted() -> None:
    """Verify AIP-193 ErrorInfo reason overrides status and sets error_info_reason."""
    resp = _make_response(
        400,
        payload={
            "error": {
                "code": 400,
                "message": "Resource already exists in target location",
                "status": "FAILED_PRECONDITION",
                "details": [
                    {
                        "@type": "type.googleapis.com/google.rpc.ErrorInfo",
                        "domain": "mlflow.googleapis.com",
                        "reason": "RESOURCE_ALREADY_EXISTS",
                    }
                ],
            }
        },
    )
    with pytest.raises(GcpMlflowApiError) as exc_info:
        raise_for_aip193_error(resp)
    assert exc_info.value.error_code == ErrorCode.Name(RESOURCE_ALREADY_EXISTS)
    assert exc_info.value.error_info_reason == "RESOURCE_ALREADY_EXISTS"
    assert exc_info.value.gcp_status == "FAILED_PRECONDITION"


def test_raise_for_aip193_error_failed_precondition_maps_to_invalid_state() -> None:
    """Verify FAILED_PRECONDITION status maps to INVALID_STATE error code."""
    resp = _make_response(
        400,
        payload={
            "error": {
                "code": 400,
                "message": "Precondition failed.",
                "status": "FAILED_PRECONDITION",
            }
        },
    )
    with pytest.raises(GcpMlflowApiError) as exc_info:
        raise_for_aip193_error(resp)
    assert exc_info.value.error_code == ErrorCode.Name(INVALID_STATE)
    assert exc_info.value.error_info_reason is None


def test_raise_for_aip193_error_http_504_maps_to_deadline_exceeded() -> None:
    """Verify HTTP 504 Gateway Timeout maps to DEADLINE_EXCEEDED error code."""
    resp = _make_response(504, raw_text="Gateway Timeout")
    with pytest.raises(GcpMlflowApiError) as exc_info:
        raise_for_aip193_error(resp)
    assert exc_info.value.error_code == ErrorCode.Name(DEADLINE_EXCEEDED)


def test_raise_for_aip193_error_info_reason_foreign_domain_ignored() -> None:
    """Verify ErrorInfo from foreign service domains is ignored."""
    resp = _make_response(
        400,
        payload={
            "error": {
                "code": 400,
                "message": "Resource already exists in foreign service",
                "status": "FAILED_PRECONDITION",
                "details": [
                    {
                        "@type": "type.googleapis.com/google.rpc.ErrorInfo",
                        "domain": "compute.googleapis.com",
                        "reason": "RESOURCE_ALREADY_EXISTS",
                    }
                ],
            }
        },
    )
    with pytest.raises(GcpMlflowApiError) as exc_info:
        raise_for_aip193_error(resp)
    assert exc_info.value.error_info_reason is None
    assert exc_info.value.error_code == ErrorCode.Name(INVALID_STATE)


def test_raise_for_aip193_error_info_reason_staging_domain_accepted() -> None:
    """Verify ErrorInfo from staging Managed MLflow domains is accepted."""
    resp = _make_response(
        400,
        payload={
            "error": {
                "code": 400,
                "message": "Resource already exists",
                "status": "FAILED_PRECONDITION",
                "details": [
                    {
                        "@type": "type.googleapis.com/google.rpc.ErrorInfo",
                        "domain": "staging-mlflow.googleapis.com",
                        "reason": "RESOURCE_ALREADY_EXISTS",
                    }
                ],
            }
        },
    )
    with pytest.raises(GcpMlflowApiError) as exc_info:
        raise_for_aip193_error(resp)
    assert exc_info.value.error_info_reason == "RESOURCE_ALREADY_EXISTS"
    assert exc_info.value.error_code == ErrorCode.Name(RESOURCE_ALREADY_EXISTS)


def test_raise_for_aip193_error_info_reason_unknown_reason_code() -> None:
    """Verify unknown reason sets error_info_reason but preserves status code."""
    resp = _make_response(
        400,
        payload={
            "error": {
                "code": 400,
                "message": "Precondition failed with unrecognized reason",
                "status": "FAILED_PRECONDITION",
                "details": [
                    {
                        "@type": "type.googleapis.com/google.rpc.ErrorInfo",
                        "domain": "mlflow.googleapis.com",
                        "reason": "NOT_A_CODE",
                    }
                ],
            }
        },
    )
    with pytest.raises(GcpMlflowApiError) as exc_info:
        raise_for_aip193_error(resp)
    assert exc_info.value.error_info_reason == "NOT_A_CODE"
    assert exc_info.value.error_code == ErrorCode.Name(INVALID_STATE)


def test_raise_for_aip193_error_info_reason_malformed_details() -> None:
    """Verify malformed ErrorInfo details structures are handled safely."""
    # Test non-list details: "string"
    resp_str = _make_response(
        400,
        payload={
            "error": {
                "code": 400,
                "message": "Malformed details",
                "status": "FAILED_PRECONDITION",
                "details": "string",
            }
        },
    )
    with pytest.raises(GcpMlflowApiError) as exc_info:
        raise_for_aip193_error(resp_str)
    assert exc_info.value.error_info_reason is None
    assert exc_info.value.error_code == ErrorCode.Name(INVALID_STATE)

    # Test non-dict items in details list: [None, 123]
    resp_items = _make_response(
        400,
        payload={
            "error": {
                "code": 400,
                "message": "Non-dict items in details",
                "status": "FAILED_PRECONDITION",
                "details": [None, 123],
            }
        },
    )
    with pytest.raises(GcpMlflowApiError) as exc_info:
        raise_for_aip193_error(resp_items)
    assert exc_info.value.error_info_reason is None
    assert exc_info.value.error_code == ErrorCode.Name(INVALID_STATE)
