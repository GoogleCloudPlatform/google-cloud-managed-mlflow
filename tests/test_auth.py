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

"""Tests for Google Cloud Managed MLflow authentication and transport security."""

from __future__ import annotations

import contextlib
import datetime
import http.server
import importlib
import importlib.metadata
import ipaddress
import json
import shutil
import ssl
import threading
import unittest.mock
import zipfile
from collections.abc import Callable, Iterator, MutableMapping
from pathlib import Path
from typing import Any

import google.auth.exceptions
import mlflow.tracking.request_auth.registry
import mlflow.utils.request_utils
import pytest
import requests
import requests.adapters
import requests.exceptions
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import NameOID
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import (
    INVALID_PARAMETER_VALUE,
    UNAUTHENTICATED,
    ErrorCode,
)
from mlflow.tracking import MlflowClient
from mlflow.tracking.request_auth.abstract_request_auth_provider import (
    RequestAuthProvider,
)

import google_cloud_managed_mlflow
from google_cloud_managed_mlflow._validation import (
    _EndpointOrigin,
    _is_allowed_destination,
)
from google_cloud_managed_mlflow.auth import (
    AUTH_PROVIDER_NAME,
    CLOUD_PLATFORM_SCOPE,
    GoogleCloudAuthProvider,
    GoogleCloudBearerAuth,
    _create_tls12_ssl_context,
    _is_tls_verification_disabled,
    _Tls12HttpAdapter,
    _validate_redirect_target,
    create_secure_session,
    ensure_auth_provider_registered,
    is_truthy_env,
    validate_transport_security,
)


class _MockCredentials:
    """Mock Google credentials for testing before_request behavior."""

    def __init__(
        self,
        token: str = "fake-adc-token-1",
        quota_project_id: str | None = None,
        expired: bool = False,
        on_refresh: Callable[[Any], None] | None = None,
        refresh_error: Exception | None = None,
    ) -> None:
        self.token = token
        self.quota_project_id = quota_project_id
        self.expired = expired
        self.refresh_count = 0
        self._on_refresh = on_refresh
        self.refresh_error = refresh_error

    def refresh(self, request: Any = None) -> None:
        if self._on_refresh is not None:
            self._on_refresh(request)
        if self.refresh_error is not None:
            raise self.refresh_error
        self.refresh_count += 1
        self.expired = False
        self.token = f"fake-adc-token-refreshed-{self.refresh_count}"

    def before_request(
        self, request: Any, method: str, url: str, headers: MutableMapping[str, str]
    ) -> None:
        if self.expired:
            self.refresh(request)
        headers["Authorization"] = f"Bearer {self.token}"
        if self.quota_project_id:
            headers["x-goog-user-project"] = self.quota_project_id


def _send_json_response(
    handler: http.server.BaseHTTPRequestHandler, status_code: int, payload: Any
) -> None:
    """Send a JSON HTTP response with Content-Length and close connection."""
    body = json.dumps(payload).encode("utf-8")
    handler.send_response(status_code)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(body)))
    handler.send_header("Connection", "close")
    handler.end_headers()
    handler.wfile.write(body)


def _prep(url: str) -> requests.PreparedRequest:
    return requests.Request("GET", url).prepare()


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


@contextlib.contextmanager
def _local_http_server(
    handler_cls: type[http.server.BaseHTTPRequestHandler],
    *,
    ssl_context: ssl.SSLContext | None = None,
) -> Iterator[tuple[http.server.HTTPServer, str]]:
    """Run a local loopback HTTP/HTTPS server on a background thread."""
    server = http.server.HTTPServer(("127.0.0.1", 0), handler_cls)
    scheme = "http"
    if ssl_context is not None:
        server.socket = ssl_context.wrap_socket(server.socket, server_side=True)
        scheme = "https"
    port = server.server_address[1]
    thread = threading.Thread(
        target=server.serve_forever, kwargs={"poll_interval": 0.02}, daemon=True
    )
    thread.start()
    try:
        yield server, f"{scheme}://127.0.0.1:{port}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2.0)


@pytest.fixture(autouse=True)
def _reset_auth_state() -> None:
    GoogleCloudBearerAuth.reset_for_testing()


def test_truthy_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for val in ("1", "true", "TRUE", "yes", "Yes", "on", "ON", " true "):
        monkeypatch.setenv("TEST_FLAG", val)
        assert is_truthy_env("TEST_FLAG") is True
    for val in ("0", "false", "no", "off", "", "random"):
        monkeypatch.setenv("TEST_FLAG", val)
        assert is_truthy_env("TEST_FLAG") is False
    monkeypatch.delenv("TEST_FLAG", raising=False)
    assert is_truthy_env("TEST_FLAG") is False


def test_tls_verification_disabled_check() -> None:
    assert _is_tls_verification_disabled(False) is True
    assert _is_tls_verification_disabled(None) is True
    assert _is_tls_verification_disabled("") is True
    assert _is_tls_verification_disabled("   ") is True
    assert _is_tls_verification_disabled(True) is False
    assert _is_tls_verification_disabled("/path/to/custom-ca.pem") is False


def test_validate_transport_security_allowed_schemes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("MLFLOW_GCP_ENDPOINT_OVERRIDE", raising=False)
    monkeypatch.delenv("MLFLOW_TRACKING_INSECURE_TLS", raising=False)
    for allowed_url in (
        "https://mlflow.googleapis.com",
        "https://mlflow.googleapis.com:443/api/2.0",
        "https://MLFLOW.GOOGLEAPIS.COM./api/2.0",
        "https://us-central1-mlflow.googleapis.com",
        "https://mlflow.rep.googleapis.com",
        "https://mlflow.mtls.googleapis.com",
        "https://psc.p.googleapis.com",
    ):
        validate_transport_security(allowed_url)


def test_validate_transport_security_untrusted_https_hosts_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("MLFLOW_GCP_ENDPOINT_OVERRIDE", raising=False)
    for untrusted_url in (
        "https://mlflow.example.com",
        "https://evilgoogleapis.com",
        "https://googleapis.com",
        "https://.googleapis.com",
        "https://mlflow..googleapis.com",
        "https://bad_label.googleapis.com/api",
        "https://-bad.googleapis.com/api",
        "https://bad-.googleapis.com/api",
        f"https://{'a' * 64}.googleapis.com/api",
        "https://mlflow.googleapis.com.evil.example.com",
        "https://mlflow.googleapis.com:8443/api/2.0",
        "https://[v1.evil.example.com:8443.googleapis.com]/api",
        "https://evil.example.com@mlflow.googleapis.com/api",
        "https://evil.example.com%2F.googleapis.com/api",
        "https://evil.example.com;mlflow.googleapis.com/api",
    ):
        with _raises_invalid_param():
            validate_transport_security(untrusted_url)

    auth = GoogleCloudBearerAuth()
    with unittest.mock.patch("google.auth.default") as mock_default:
        for probe_url in (
            "https://mlflow.example.com/api/2.0",
            "https://evil.example.com%2F.googleapis.com/api",
        ):
            with _raises_invalid_param():
                auth(_prep(probe_url))
        # Assign malformed bracketed host after prepare() so urllib3 does not
        # reject the URL before GoogleCloudBearerAuth.__call__ validates it.
        req_bracket = _prep("https://mlflow.example.com/api/2.0")
        req_bracket.url = "https://[v1.evil.example.com:8443.googleapis.com]/api"
        with _raises_invalid_param():
            auth(req_bracket)
        mock_default.assert_not_called()


def test_validate_transport_security_unconfigured_loopback_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("MLFLOW_GCP_ENDPOINT_OVERRIDE", raising=False)
    for loopback_url, expected_origin in (
        ("http://localhost", "http://localhost:80"),
        ("http://localhost:5000", "http://localhost:5000"),
        ("https://localhost:8443", "https://localhost:8443"),
        ("http://127.0.0.1:8080", "http://127.0.0.1:8080"),
        ("https://127.0.0.1:8443", "https://127.0.0.1:8443"),
        ("http://[::1]:8080", "http://[::1]:8080"),
        ("https://[::1]", "https://[::1]:443"),
    ):
        with _raises_invalid_param() as exc_info:
            validate_transport_security(loopback_url)
        assert expected_origin in str(exc_info.value)
        assert "MLFLOW_GCP_ENDPOINT_OVERRIDE" in str(exc_info.value)

    auth = GoogleCloudBearerAuth()
    with unittest.mock.patch("google.auth.default") as mock_default:
        with _raises_invalid_param():
            auth(_prep("http://127.0.0.1:5000/api/2.0"))
        mock_default.assert_not_called()


def test_validate_transport_security_endpoint_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # 1. Custom HTTPS endpoint with explicit port requires exact (scheme, host, port).
    monkeypatch.setenv(
        "MLFLOW_GCP_ENDPOINT_OVERRIDE", "https://custom-gateway.example.com:8443/v1"
    )
    for ok_url in (
        "https://custom-gateway.example.com:8443/api/2.0",
        "https://CUSTOM-GATEWAY.EXAMPLE.COM.:8443/api/2.0",
        "https://mlflow.googleapis.com/api/2.0",
    ):
        validate_transport_security(ok_url)

    auth = GoogleCloudBearerAuth(credentials=_MockCredentials(token="override-tok"))
    assert (
        auth(_prep("https://custom-gateway.example.com:8443/api/2.0")).headers.get(
            "Authorization"
        )
        == "Bearer override-tok"
    )

    for bad in (
        "https://custom-gateway.example.com/api/2.0",
        "https://custom-gateway.example.com:9443/api/2.0",
        "https://other-gateway.example.com:8443/api/2.0",
    ):
        with _raises_invalid_param(match="untrusted origin"):
            validate_transport_security(bad)
    with _raises_invalid_param(match="Insecure HTTP is only permitted on loopback"):
        validate_transport_security("http://custom-gateway.example.com:8443/api/2.0")

    # 2. Default HTTPS port 443 normalization on override.
    monkeypatch.setenv(
        "MLFLOW_GCP_ENDPOINT_OVERRIDE", "https://custom-gateway.example.com"
    )
    validate_transport_security("https://custom-gateway.example.com/api/2.0")
    validate_transport_security("https://custom-gateway.example.com:443/api/2.0")
    with _raises_invalid_param():
        validate_transport_security("https://custom-gateway.example.com:8443/api/2.0")

    # 3. Loopback HTTP/HTTPS opt-in enforces exact (scheme, hostname, effective_port).
    monkeypatch.setenv("MLFLOW_GCP_ENDPOINT_OVERRIDE", "http://127.0.0.1:5000")
    validate_transport_security("http://127.0.0.1:5000/api/2.0")
    for bad in (
        "http://127.0.0.1:5001/api/2.0",
        "http://127.0.0.1/api/2.0",
        "https://127.0.0.1:5000/api/2.0",
        "http://localhost:5000/api/2.0",
    ):
        with _raises_invalid_param():
            validate_transport_security(bad)

    monkeypatch.setenv("MLFLOW_GCP_ENDPOINT_OVERRIDE", "http://localhost")
    validate_transport_security("http://localhost/api/2.0")
    validate_transport_security("http://localhost:80/api/2.0")
    with _raises_invalid_param():
        validate_transport_security("https://localhost/api/2.0")

    monkeypatch.setenv("MLFLOW_GCP_ENDPOINT_OVERRIDE", "https://[::1]:8443")
    validate_transport_security("https://[::1]:8443/api/2.0")
    validate_transport_security("https://[0:0:0:0:0:0:0:1]:8443/api/2.0")
    for bad in ("http://[::1]:8443/api/2.0", "https://[::1]/api/2.0"):
        with _raises_invalid_param():
            validate_transport_security(bad)

    monkeypatch.setenv("MLFLOW_GCP_ENDPOINT_OVERRIDE", "http://[::1]:5000")
    validate_transport_security("http://[::1]:5000/api/2.0")
    validate_transport_security("http://[0:0:0:0:0:0:0:1]:5000/api/2.0")

    # 4. Non-loopback HTTP in MLFLOW_GCP_ENDPOINT_OVERRIDE itself fails closed.
    monkeypatch.setenv(
        "MLFLOW_GCP_ENDPOINT_OVERRIDE", "http://custom-gateway.example.com"
    )
    with _raises_invalid_param(match="untrusted origin"):
        validate_transport_security("https://custom-gateway.example.com/api/2.0")


@pytest.mark.parametrize(
    ("bad_override", "probe_origin"),
    [
        ("not-a-url", ("https", "custom-gateway.example.com", 443)),
        ("https://[::1", ("https", "::1", 443)),
        (
            "https://custom-gateway.example.com:0",
            ("https", "custom-gateway.example.com", 0),
        ),
        ("http://127.0.0.1:0", ("http", "127.0.0.1", 0)),
        (
            "https://custom-gateway.example.com:badport",
            ("https", "custom-gateway.example.com", 443),
        ),
        (
            r"https://mlflow.googleapis.com\@custom-gateway.example.com",
            ("https", "custom-gateway.example.com", 443),
        ),
        (
            "https://custom-gateway	.example.com",
            ("https", "custom-gateway.example.com", 443),
        ),
        (
            "https://custom gateway.example.com",
            ("https", "custom gateway.example.com", 443),
        ),
    ],
)
def test_malformed_endpoint_override_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
    bad_override: str,
    probe_origin: tuple[str, str, int],
) -> None:
    monkeypatch.setenv("MLFLOW_GCP_ENDPOINT_OVERRIDE", bad_override)
    assert _is_allowed_destination(_EndpointOrigin(*probe_origin)) is False


def test_validate_transport_security_remote_http_rejected() -> None:
    for remote_http in ("http://mlflow.googleapis.com", "http://mlflow.example.com"):
        with _raises_invalid_param(match="Insecure HTTP is only permitted on loopback"):
            validate_transport_security(remote_http)


def test_validate_transport_security_insecure_tls_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with _raises_invalid_param():
        validate_transport_security(
            "https://mlflow.googleapis.com", ignore_tls_verification=True
        )
    for truthy in ("1", "true", "yes", "on"):
        monkeypatch.setenv("MLFLOW_TRACKING_INSECURE_TLS", truthy)
        with _raises_invalid_param():
            validate_transport_security("https://mlflow.googleapis.com")


def test_validate_transport_security_invalid_urls() -> None:
    for bad_url in (
        "",
        "   ",
        "ftp://mlflow.googleapis.com",
        "not-a-url",
        "https://.",
        "https://[::1/api",
        "https://[::1%lo]:8443/api",
        "https://[::1%25lo]:8443/api",
        "https://mlflow.googleapis.com:0/api",
        "http://127.0.0.1:0/api",
        "https://mlflow.googleapis.com:badport/api",
        "https://evil.example.com\\@mlflow.googleapis.com/api",
        "https://mlflow\t.googleapis.com/api",
        "https://mlflow\n.googleapis.com/api",
        "ht\ttps://mlflow.googleapis.com/api",
        "https://evil .googleapis.com/api",
        "https://mlflow.googleapis.com/api\r\n",
        "https://mlflow.googleapis.com/api\x7f",
        "  https://mlflow.googleapis.com/api",
        "https://mlflow.googleapis.com/api  ",
    ):
        with _raises_invalid_param():
            validate_transport_security(bad_url)


def _generate_self_signed_cert(tmp_path: Path, prefix: str) -> tuple[Path, Path]:
    """Generate a self-signed loopback certificate + key via cryptography.x509."""
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, prefix)])
    now = datetime.datetime.now(datetime.timezone.utc)
    ski = x509.SubjectKeyIdentifier.from_public_key(key.public_key())
    builder = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(name)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - datetime.timedelta(minutes=5))
        .not_valid_after(now + datetime.timedelta(days=1))
        .add_extension(x509.BasicConstraints(ca=True, path_length=None), critical=True)
        .add_extension(
            x509.KeyUsage(
                digital_signature=True,
                content_commitment=False,
                key_encipherment=True,
                data_encipherment=False,
                key_agreement=False,
                key_cert_sign=True,
                crl_sign=True,
                encipher_only=False,
                decipher_only=False,
            ),
            critical=True,
        )
        .add_extension(ski, critical=False)
        .add_extension(
            x509.AuthorityKeyIdentifier.from_issuer_subject_key_identifier(ski),
            critical=False,
        )
        .add_extension(
            x509.SubjectAlternativeName(
                [x509.IPAddress(ipaddress.IPv4Address("127.0.0.1"))]
            ),
            critical=False,
        )
    )
    cert = builder.sign(key, hashes.SHA256())
    cert_path, key_path = tmp_path / f"{prefix}.pem", tmp_path / f"{prefix}.key"
    cert_path.write_bytes(cert.public_bytes(serialization.Encoding.PEM))
    key_path.write_bytes(
        key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.TraditionalOpenSSL,
            encryption_algorithm=serialization.NoEncryption(),
        )
    )
    return cert_path, key_path


def test_tls12_ssl_context_and_create_secure_session(tmp_path: Path) -> None:
    ctx = _create_tls12_ssl_context()
    assert ctx.minimum_version == ssl.TLSVersion.TLSv1_2
    if getattr(ctx, "post_handshake_auth", None) is not None:
        assert ctx.post_handshake_auth is True

    assert google_cloud_managed_mlflow.__all__ == [
        "__version__",
        "GcpTrackingRestStore",
        "build_tracking_store",
    ]

    ca1_cert, ca1_key = _generate_self_signed_cert(tmp_path, "ca1")
    ca2_cert, ca2_key = _generate_self_signed_cert(tmp_path, "ca2")
    combined_pem = tmp_path / "ca2_combined.pem"
    combined_pem.write_bytes(ca2_cert.read_bytes() + ca2_key.read_bytes())

    server_ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    server_ctx.minimum_version = ssl.TLSVersion.TLSv1_2
    server_ctx.load_cert_chain(certfile=str(ca1_cert), keyfile=str(ca1_key))
    server_ctx.verify_mode = ssl.CERT_OPTIONAL
    server_ctx.load_verify_locations(cafile=str(ca2_cert))

    class _SimpleTlsHandler(http.server.BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            has_client_cert = isinstance(self.connection, ssl.SSLSocket) and bool(
                self.connection.getpeercert()
            )
            _send_json_response(
                self, 200, {"tls": "ok", "client_cert": has_client_cert}
            )

        def log_message(self, *_: Any) -> None:
            pass

    with (
        _local_http_server(_SimpleTlsHandler, ssl_context=server_ctx) as (
            _,
            https_url,
        ),
        create_secure_session() as session,
    ):
        adapter = session.get_adapter("https://mlflow.googleapis.com")
        assert isinstance(adapter, _Tls12HttpAdapter)
        pm = adapter.proxy_manager_for("https://proxy.googleapis.com:8443")
        assert (
            pm.connection_pool_kw["ssl_context"].minimum_version
            == ssl.TLSVersion.TLSv1_2
        )

        # Request 1 pinned to ca1_cert succeeds over TLS 1.2+ (including client certs).
        assert session.get(https_url, verify=str(ca1_cert), timeout=2.0).json() == {
            "tls": "ok",
            "client_cert": False,
        }
        # Subsequent request reuses cached SSL context.
        assert (
            session.get(https_url, verify=str(ca1_cert), timeout=2.0).status_code == 200
        )
        assert session.get(
            https_url,
            verify=str(ca1_cert),
            cert=(str(ca2_cert), str(ca2_key)),
            timeout=2.0,
        ).json() == {"tls": "ok", "client_cert": True}
        assert session.get(
            https_url,
            verify=str(ca1_cert),
            cert=str(combined_pem),
            timeout=2.0,
        ).json() == {"tls": "ok", "client_cert": True}

        # Request 2 on same session pinned to ca2_cert must NOT trust ca1_cert.
        with pytest.raises(requests.exceptions.SSLError):
            session.get(https_url, verify=str(ca2_cert), timeout=2.0)

        # Missing custom CA path raises OSError cleanly without polluting cache.
        with pytest.raises(OSError):
            session.get(https_url, verify=str(tmp_path / "missing-ca.pem"), timeout=2.0)


def test_tls12_adapter_rejects_disabled_verify() -> None:
    adapter = _Tls12HttpAdapter()
    try:
        req = _prep("https://mlflow.googleapis.com")
        for bad_verify in (False, None, "", "   "):
            with _raises_invalid_param():
                adapter.send(req, verify=bad_verify)
    finally:
        adapter.close()


def test_tls12_adapter_ssl_context_caching_and_path_normalization(
    tmp_path: Path,
) -> None:
    """Verify Path vs str and list vs tuple normalization share cached SSLContexts."""
    ca_cert, ca_key = _generate_self_signed_cert(tmp_path, "pool_ca")
    cert2, key2 = _generate_self_signed_cert(tmp_path, "pool_cert")
    combined_pem = tmp_path / "combined.pem"
    combined_pem.write_bytes(cert2.read_bytes() + key2.read_bytes())

    adapter = _Tls12HttpAdapter()
    try:
        req = _prep("https://mlflow.googleapis.com")

        # 1. Path vs str for verify
        _, kw_path = adapter.build_connection_pool_key_attributes(req, verify=ca_cert)
        _, kw_str = adapter.build_connection_pool_key_attributes(
            req, verify=str(ca_cert)
        )
        assert kw_path["ssl_context"] is kw_str["ssl_context"]
        assert (str(ca_cert), None) in adapter._ssl_contexts
        assert len(adapter._ssl_contexts) == 1

        # 2. Path vs str for cert
        _, kw_cert_path = adapter.build_connection_pool_key_attributes(
            req, verify=True, cert=combined_pem
        )
        _, kw_cert_str = adapter.build_connection_pool_key_attributes(
            req, verify=True, cert=str(combined_pem)
        )
        assert kw_cert_path["ssl_context"] is kw_cert_str["ssl_context"]
        assert (True, str(combined_pem)) in adapter._ssl_contexts
        assert len(adapter._ssl_contexts) == 2

        # 3. [Path(c), Path(k)] list vs (c, k) tuple
        _, kw_list = adapter.build_connection_pool_key_attributes(
            req, verify=True, cert=[cert2, key2]
        )
        _, kw_tuple = adapter.build_connection_pool_key_attributes(
            req, verify=True, cert=(str(cert2), str(key2))
        )
        assert kw_list["ssl_context"] is kw_tuple["ssl_context"]
        assert (True, (str(cert2), str(key2))) in adapter._ssl_contexts
        assert len(adapter._ssl_contexts) == 3
    finally:
        adapter.close()


def test_bearer_auth_default_credentials_error_wrapped() -> None:
    auth = GoogleCloudBearerAuth()
    with unittest.mock.patch(
        "google.auth.default",
        side_effect=google.auth.exceptions.DefaultCredentialsError("ADC missing"),
    ):
        with pytest.raises(MlflowException) as exc_info:
            auth(_prep("https://mlflow.googleapis.com/test"))
        assert exc_info.value.error_code == ErrorCode.Name(UNAUTHENTICATED)
        assert "gcloud auth application-default login" in str(exc_info.value)
        assert "GOOGLE_APPLICATION_CREDENTIALS" in str(exc_info.value)
        assert isinstance(
            exc_info.value.__cause__, google.auth.exceptions.DefaultCredentialsError
        )


@pytest.mark.parametrize(
    ("initial_error", "exc_cls"),
    [
        (True, google.auth.exceptions.RefreshError),
        (False, google.auth.exceptions.RefreshError),
        (False, google.auth.exceptions.TransportError),
    ],
)
def test_bearer_auth_adc_recovery_on_auth_error(
    initial_error: bool,
    exc_cls: type[google.auth.exceptions.GoogleAuthError],
) -> None:
    close_spies: list[unittest.mock.MagicMock] = []
    captured_reqs: list[Any] = []

    def spy_session(request: Any) -> None:
        if request is not None and getattr(request, "session", None) is not None:
            spy = unittest.mock.MagicMock(wraps=request.session.close)
            request.session.close = spy
            close_spies.append(spy)
            captured_reqs.append(request)

    refresh_exc = exc_cls(
        "Initial ADC refresh failed" if initial_error else "ADC auth error"
    )
    stale_adc_creds = _MockCredentials(
        token="adc-token-stale",
        quota_project_id="proj-stale",
        expired=initial_error,
        on_refresh=spy_session,
        refresh_error=refresh_exc if initial_error else None,
    )
    fresh_adc_creds = _MockCredentials(
        token="adc-token-fresh",
        quota_project_id="proj-fresh",
        expired=True,
    )

    with contextlib.closing(GoogleCloudBearerAuth()) as auth:
        with unittest.mock.patch(
            "google.auth.default",
            side_effect=[
                (stale_adc_creds, "proj-stale"),
                (fresh_adc_creds, "proj-fresh"),
            ],
        ) as mock_default:
            if not initial_error:
                # 1. Initial ADC resolution and request succeed.
                out1 = auth(_prep("https://mlflow.googleapis.com/v1"))
                assert out1.headers.get("Authorization") == "Bearer adc-token-stale"
                assert auth.credentials is stale_adc_creds
                assert auth._auth_request is not None
                assert mock_default.call_count == 1
                # Arm credentials for mid-session failure.
                stale_adc_creds.expired = True
                stale_adc_creds.refresh_error = refresh_exc

            # 2. Failing request clears cached ADC creds, closes session,
            # and raises UNAUTHENTICATED.
            fail_req = _prep("https://mlflow.googleapis.com/v1")
            with pytest.raises(MlflowException) as exc_info:
                auth(fail_req)
            assert exc_info.value.error_code == ErrorCode.Name(UNAUTHENTICATED)
            assert "gcloud auth application-default login" in str(exc_info.value)
            assert "GOOGLE_APPLICATION_CREDENTIALS" in str(exc_info.value)
            assert exc_info.value.__cause__ is refresh_exc
            assert "Authorization" not in fail_req.headers
            assert len(close_spies) == 1
            assert close_spies[0].call_count == 1
            assert captured_reqs[0].session is None
            assert auth.credentials is None
            assert auth._auth_request is None
            assert mock_default.call_count == 1

            # 3. Subsequent request re-invokes google.auth.default() with fresh creds.
            out_rec = auth(_prep("https://mlflow.googleapis.com/v1"))
            assert mock_default.call_count == 2
            mock_default.assert_called_with(scopes=[CLOUD_PLATFORM_SCOPE])
            assert (
                out_rec.headers.get("Authorization")
                == "Bearer fake-adc-token-refreshed-1"
            )
            assert out_rec.headers.get("x-goog-user-project") == "proj-fresh"
            assert auth.credentials is fresh_adc_creds
            assert auth._auth_request is not None

            # 4. Valid recovered ADC credentials reused without re-calling default().
            auth(_prep("https://mlflow.googleapis.com/v1"))
            assert mock_default.call_count == 2


def _via_init(creds: _MockCredentials) -> GoogleCloudBearerAuth:
    return GoogleCloudBearerAuth(credentials=creds)


def _via_setter_after_adc(creds: _MockCredentials) -> GoogleCloudBearerAuth:
    auth = GoogleCloudBearerAuth()
    adc_initial = _MockCredentials(token="adc-before-override")
    with unittest.mock.patch(
        "google.auth.default", return_value=(adc_initial, "adc-proj-1")
    ) as mock_init:
        auth(_prep("https://mlflow.googleapis.com/test"))
        assert auth.credentials is adc_initial
        mock_init.assert_called_once()
    auth.credentials = creds
    return auth


@pytest.mark.parametrize(
    ("make_auth", "exc_cls"),
    [
        (_via_init, google.auth.exceptions.RefreshError),
        (_via_setter_after_adc, google.auth.exceptions.TransportError),
    ],
)
def test_bearer_auth_explicit_credentials_preserved_on_auth_error(
    make_auth: Callable[[_MockCredentials], GoogleCloudBearerAuth],
    exc_cls: type[google.auth.exceptions.GoogleAuthError],
) -> None:
    first_exc = exc_cls("Explicit auth failure")
    explicit_creds = _MockCredentials(
        token="explicit-tok", expired=True, refresh_error=first_exc
    )
    adc_creds_after_clear = _MockCredentials(token="adc-after-clear")

    with (
        contextlib.closing(make_auth(explicit_creds)) as auth,
        unittest.mock.patch(
            "google.auth.default",
            return_value=(adc_creds_after_clear, "adc-proj-2"),
        ) as mock_default,
    ):
        with pytest.raises(MlflowException) as exc_info:
            auth(_prep("https://mlflow.googleapis.com/test"))
        assert exc_info.value.error_code == ErrorCode.Name(UNAUTHENTICATED)
        assert exc_info.value.__cause__ is first_exc
        assert auth.credentials is explicit_creds
        mock_default.assert_not_called()

        explicit_creds.refresh_error = None
        out2 = auth(_prep("https://mlflow.googleapis.com/test"))
        assert out2.headers.get("Authorization") == "Bearer fake-adc-token-refreshed-1"
        assert auth.credentials is explicit_creds
        mock_default.assert_not_called()

        explicit_creds.expired = True
        explicit_creds.refresh_error = exc_cls("Explicit auth retry failure")
        with pytest.raises(MlflowException):
            auth(_prep("https://mlflow.googleapis.com/test"))
        assert auth.credentials is explicit_creds
        mock_default.assert_not_called()

        auth.credentials = None
        out_adc = auth(_prep("https://mlflow.googleapis.com/test"))
        assert out_adc.headers.get("Authorization") == "Bearer adc-after-clear"
        assert auth.credentials is adc_creds_after_clear
        mock_default.assert_called_once()


def test_bearer_auth_credentials_lifecycle_and_session_cleanup() -> None:
    auth = GoogleCloudBearerAuth.get_instance()
    assert auth.credentials is None
    assert "unset" in repr(auth)

    # 1. Lazy ADC resolution via google.auth.default.
    creds = _MockCredentials(token="lazy-token-secret")
    with unittest.mock.patch(
        "google.auth.default", return_value=(creds, "proj-lazy")
    ) as mock_default:
        out1 = auth(_prep("https://mlflow.googleapis.com/test"))
        mock_default.assert_called_once_with(scopes=[CLOUD_PLATFORM_SCOPE])
        assert out1.headers.get("Authorization") == "Bearer lazy-token-secret"
        assert auth.credentials is creds
        assert "set" in repr(auth)
        assert "lazy-token-secret" not in repr(auth)

    # 2. Valid token reuse vs expired refresh.
    assert creds.refresh_count == 0
    auth(_prep("https://mlflow.googleapis.com/test"))
    assert creds.refresh_count == 0
    creds.expired = True
    out_refreshed = auth(_prep("https://mlflow.googleapis.com/test"))
    assert creds.refresh_count == 1
    assert (
        out_refreshed.headers.get("Authorization")
        == "Bearer fake-adc-token-refreshed-1"
    )

    # 3. Session cleanup on credentials replacement, reset_for_testing, and close().
    assert auth._auth_request is not None
    assert auth._auth_request.session is not None
    session1 = auth._auth_request.session
    assert isinstance(
        session1.get_adapter("https://oauth2.googleapis.com/token"),
        _Tls12HttpAdapter,
    )
    with unittest.mock.patch.object(session1, "close", wraps=session1.close) as spy1:
        auth.credentials = _MockCredentials(token="token-2")
        spy1.assert_called_once()
        assert auth._auth_request is None

    auth(_prep("https://mlflow.googleapis.com/test"))
    assert auth._auth_request is not None
    assert auth._auth_request.session is not None
    session2 = auth._auth_request.session
    with unittest.mock.patch.object(session2, "close", wraps=session2.close) as spy2:
        GoogleCloudBearerAuth.reset_for_testing()
        spy2.assert_called_once()

    auth_closed = GoogleCloudBearerAuth(credentials=_MockCredentials())
    auth_closed(_prep("https://mlflow.googleapis.com/test"))
    assert auth_closed._auth_request is not None
    assert auth_closed._auth_request.session is not None
    session3 = auth_closed._auth_request.session
    with unittest.mock.patch.object(session3, "close", wraps=session3.close) as spy3:
        auth_closed.close()
        spy3.assert_called_once()
        assert auth_closed._auth_request is None


def test_bearer_auth_concurrent_single_flight_refresh() -> None:
    in_refresh = threading.Event()
    creds = _MockCredentials(
        token="initial-token",
        expired=True,
        on_refresh=lambda _: in_refresh.set(),
    )
    auth = GoogleCloudBearerAuth(credentials=creds)

    barrier = threading.Barrier(8)
    errors: list[BaseException] = []

    def worker() -> None:
        try:
            barrier.wait()
            auth(_prep("https://mlflow.googleapis.com/api"))
        except BaseException as exc:  # pragma: no cover - defensive capture
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors
    assert in_refresh.is_set()
    assert creds.refresh_count == 1


def test_bearer_auth_redirect_blocks_untrusted_and_allows_relative(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("MLFLOW_GCP_ENDPOINT_OVERRIDE", raising=False)
    auth = GoogleCloudBearerAuth(
        credentials=_MockCredentials(
            token="fake-adc-token-redirect",
            quota_project_id="redirect-test-project",
        )
    )

    # Verify duplicate calls do not register duplicate redirect hooks.
    prep = _prep("https://mlflow.googleapis.com/api")
    auth(prep)
    auth(prep)
    assert prep.hooks["response"].count(_validate_redirect_target) == 1

    # Verify _validate_redirect_target closes response.raw on untrusted host,
    # non-443 port, unconfigured loopback, malformed IPv6, invalid UTF-8, and
    # control/whitespace/encoded Location headers.
    raw_resp = requests.Response()
    raw_resp.status_code = 302
    raw_resp.url = "https://mlflow.googleapis.com/api"
    raw_resp.raw = unittest.mock.MagicMock()
    with unittest.mock.patch.object(raw_resp, "close") as close_mock:
        for bad_location in (
            "https://evil.example.com/steal",
            "https://mlflow.googleapis.com:8443/steal",
            "http://127.0.0.1:8080/steal",
            "https://[::1/steal",
            b"https://mlflow.googleapis.com/\xff".decode("latin1"),
            "https://mlflow\t.googleapis.com/steal",
            "https://evil .googleapis.com/steal",
            "https://evil.com%2e.googleapis.com/v1",
            "\x0b//mlflow.googleapis.com/v1",
            " //mlflow.googleapis.com/v1",
            "https://mlflow.googleapis.com  ",
            "%68ttps://evil.example.com/steal",
            "%68ttp://127.0.0.1:8080/steal",
            "%68ttps://localhost/steal",
            "\x7fhttps://mlflow.googleapis.com/v1",
        ):
            raw_resp.headers["Location"] = bad_location
            close_mock.reset_mock()
            with _raises_invalid_param():
                _validate_redirect_target(raw_resp)
            close_mock.assert_called_once()

    redirect_state = {"location": "https://evil.example.com/steal"}

    class _RedirectHandler(http.server.BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            if self.path.startswith("/redirect"):
                self.send_response(302)
                self.send_header("Location", redirect_state["location"])
                self.send_header("Content-Length", "0")
                self.send_header("Connection", "close")
                self.end_headers()
            else:
                _send_json_response(self, 200, {"ok": True, "path": self.path})

        def log_message(self, *_: Any) -> None:
            pass

    sent_urls: list[str] = []
    orig_send = requests.adapters.HTTPAdapter.send

    def tracking_send(
        self: requests.adapters.HTTPAdapter,
        request: requests.PreparedRequest,
        *args: Any,
        **kwargs: Any,
    ) -> requests.Response:
        sent_urls.append(request.url or "")
        return orig_send(self, request, *args, **kwargs)

    with (
        _local_http_server(_RedirectHandler) as (server, base_url),
        requests.Session() as session,
        unittest.mock.patch.object(
            requests.adapters.HTTPAdapter,
            "send",
            autospec=True,
            side_effect=tracking_send,
        ),
    ):
        # 0. Without MLFLOW_GCP_ENDPOINT_OVERRIDE, loopback is rejected before send.
        with _raises_invalid_param():
            session.get(f"{base_url}/redirect", auth=auth, timeout=2.0)
        assert sent_urls == []

        monkeypatch.setenv("MLFLOW_GCP_ENDPOINT_OVERRIDE", base_url)
        port = server.server_address[1]
        other_port = 5000 if port != 5000 else 5001

        # 1. Representative cross-origin, protocol-relative, encoded-scheme, and
        # mismatched-loopback redirects are blocked before connect.
        for bypass_target in (
            "https://evil.example.com/steal",
            "//evil.example.com/steal",
            "%68ttps://evil.example.com/steal",
            f"http://127.0.0.1:{other_port}/steal",
            f"https://127.0.0.1:{port}/steal",
        ):
            redirect_state["location"] = bypass_target
            with _raises_invalid_param():
                session.get(f"{base_url}/redirect", auth=auth, timeout=2.0)
            assert all(
                "evil.example.com" not in u and "steal" not in u for u in sent_urls
            )

        # 2. Same-origin relative redirect succeeds end-to-end.
        redirect_state["location"] = "/v1alpha/experiments"
        resp = session.get(f"{base_url}/redirect", auth=auth, timeout=2.0)
        assert resp.status_code == 200
        assert resp.json() == {
            "ok": True,
            "path": "/v1alpha/experiments",
        }


def test_ensure_auth_provider_registered_idempotency_and_fallbacks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reg_mod = mlflow.tracking.request_auth.registry
    reg_cls: Any = reg_mod.RequestAuthProviderRegistry

    # Case 1: Empty registry where register_entrypoints is a no-op.
    empty_reg = reg_cls()
    with unittest.mock.patch.object(empty_reg, "register_entrypoints") as mock_noop:
        monkeypatch.setattr(reg_mod, "_request_auth_provider_registry", empty_reg)
        assert ensure_auth_provider_registered() is True
        assert ensure_auth_provider_registered() is True
        mock_noop.assert_called_once()
        assert [p.get_name() for p in empty_reg] == [AUTH_PROVIDER_NAME]

    # Case 2: Empty registry where register_entrypoints populates google-cloud-auth.
    ep_reg = reg_cls()
    with unittest.mock.patch.object(
        ep_reg,
        "register_entrypoints",
        side_effect=lambda: ep_reg.register(GoogleCloudAuthProvider),
    ) as mock_pop:
        monkeypatch.setattr(reg_mod, "_request_auth_provider_registry", ep_reg)
        assert ensure_auth_provider_registered() is True
        assert ensure_auth_provider_registered() is True
        mock_pop.assert_called_once()
        assert [p.get_name() for p in ep_reg] == [AUTH_PROVIDER_NAME]

    # Case 3: Non-empty registry does not re-run register_entrypoints.
    class _OtherAuthProvider(RequestAuthProvider):
        def get_name(self) -> str:
            return "other-auth"

        def get_auth(self) -> Any:
            return None

    non_empty = reg_cls()
    non_empty.register(_OtherAuthProvider)
    with unittest.mock.patch.object(
        non_empty, "register_entrypoints"
    ) as mock_non_empty:
        monkeypatch.setattr(reg_mod, "_request_auth_provider_registry", non_empty)
        assert ensure_auth_provider_registered() is True
        assert ensure_auth_provider_registered() is True
        mock_non_empty.assert_not_called()
        assert [p.get_name() for p in non_empty] == ["other-auth", AUTH_PROVIDER_NAME]

    # Case 4: Defensive fallbacks on [], None, delattr, and sys.modules failure.
    bad_reg: Any
    for bad_reg in ([], None):
        monkeypatch.setattr(reg_mod, "_request_auth_provider_registry", bad_reg)
        assert ensure_auth_provider_registered() is False

    monkeypatch.delattr(reg_mod, "_request_auth_provider_registry")
    assert ensure_auth_provider_registered() is False

    with unittest.mock.patch.dict(
        "sys.modules", {"mlflow.tracking.request_auth.registry": None}
    ):
        assert ensure_auth_provider_registered() is False


def test_auth_provider_and_end_to_end_mlflow_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    received_headers: list[dict[str, str]] = []
    exp_payload = {
        "experiment": {
            "experiment_id": "test-exp-1",
            "name": "test-exp",
            "artifact_location": "gs://bucket/test-exp-1",
            "lifecycle_stage": "active",
        }
    }

    class _RecordingHandler(http.server.BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            received_headers.append(dict(self.headers))
            _send_json_response(self, 200, exp_payload)

        def log_message(self, *_: Any) -> None:
            pass

    reg_mod = mlflow.tracking.request_auth.registry
    reg_cls: Any = reg_mod.RequestAuthProviderRegistry
    monkeypatch.setattr(reg_mod, "_request_auth_provider_registry", reg_cls())

    with _local_http_server(_RecordingHandler) as (_, base_url):
        assert ensure_auth_provider_registered() is True
        provider = GoogleCloudAuthProvider()
        assert provider.get_name() == AUTH_PROVIDER_NAME
        auth = provider.get_auth()
        assert isinstance(auth, GoogleCloudBearerAuth)
        auth.credentials = _MockCredentials(
            token="fake-adc-token-1", quota_project_id="test-user-project"
        )

        monkeypatch.setenv("MLFLOW_TRACKING_AUTH", AUTH_PROVIDER_NAME)
        monkeypatch.delenv("MLFLOW_GCP_ENDPOINT_OVERRIDE", raising=False)

        client_untrusted = MlflowClient(tracking_uri=base_url)
        with pytest.raises(MlflowException) as exc_info:
            client_untrusted.get_experiment_by_name("test-exp")
        assert "untrusted origin" in str(exc_info.value)
        assert received_headers == []

        monkeypatch.setenv("MLFLOW_GCP_ENDPOINT_OVERRIDE", base_url)
        client = MlflowClient(tracking_uri=base_url)
        exp = client.get_experiment_by_name("test-exp")
        assert exp is not None
        assert exp.experiment_id == "test-exp-1"
        assert len(received_headers) >= 1
        assert received_headers[0].get("Authorization") == "Bearer fake-adc-token-1"
        assert received_headers[0].get("x-goog-user-project") == "test-user-project"


def test_non_interference_regression() -> None:
    """Importing the plugin must not monkey-patch MLflow's shared session factory."""
    session_fn = mlflow.utils.request_utils._get_request_session
    assert session_fn.__module__ == "mlflow.utils.request_utils"

    with mlflow.utils.request_utils._get_request_session(
        3, 1.0, 0.5, (429,), True, True
    ) as session:
        https_adapter = session.get_adapter("https://mlflow.example.com")
        assert not isinstance(https_adapter, _Tls12HttpAdapter)


def test_wheel_packaging_metadata_and_dynamic_version(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    repo_root = Path(__file__).resolve().parent.parent
    pkg_src, dist_dir, site_dir = (
        tmp_path / "pkg_src",
        tmp_path / "dist",
        tmp_path / "site",
    )
    pkg_src.mkdir()
    dist_dir.mkdir()
    site_dir.mkdir()

    for rel_file in ("pyproject.toml", "README.md", "LICENSE"):
        shutil.copy2(repo_root / rel_file, pkg_src / rel_file)
    shutil.copytree(
        repo_root / "google_cloud_managed_mlflow",
        pkg_src / "google_cloud_managed_mlflow",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )

    monkeypatch.chdir(pkg_src)
    # Poison package and runtime deps in sys.modules to ensure setuptools
    # extracts __version__ statically via AST rather than falling back to
    # module import.
    with unittest.mock.patch.dict(
        "sys.modules",
        {
            "google": None,
            "google_cloud_managed_mlflow": None,
            "google_cloud_managed_mlflow.auth": None,
            "mlflow": None,
            "requests": None,
        },
    ):
        build_meta: Any = importlib.import_module("setuptools.build_meta")
        wheel_name: str = build_meta.build_wheel(str(dist_dir))

    with zipfile.ZipFile(dist_dir / wheel_name) as zf:
        zf.extractall(site_dir)

    assert (site_dir / "google_cloud_managed_mlflow" / "py.typed").is_file()
    installed_dists = [
        d
        for d in importlib.metadata.distributions(path=[str(site_dir)])
        if d.metadata["Name"] == "google-cloud-managed-mlflow"
    ]
    assert len(installed_dists) == 1
    assert installed_dists[0].version == google_cloud_managed_mlflow.__version__
    requires = installed_dists[0].requires or []
    assert "mlflow>=3.17.2" in requires
    assert "requests>=2.32.3" in requires


def test_is_gcp_request_url_fail_closed_and_resolution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from google_cloud_managed_mlflow.auth import _is_gcp_request_url

    # Standard Google Cloud hosts
    assert _is_gcp_request_url("https://mlflow.googleapis.com/v1/projects/p") is True
    assert _is_gcp_request_url("https://us-central1-mlflow.googleapis.com/v1") is True
    assert _is_gcp_request_url("https://googleapis.com/test") is True

    # Non-GCP hosts
    assert _is_gcp_request_url("https://custom-mlflow.internal:5000/api/2.0") is False
    assert _is_gcp_request_url("https://mlflow.example.com/api/2.0") is False
    assert _is_gcp_request_url("") is False
    assert _is_gcp_request_url(None) is False

    # Malformed URL fails closed (returns True)
    assert _is_gcp_request_url("https://[invalid-ipv6/path") is True

    # Override endpoint matching
    monkeypatch.setenv("MLFLOW_GCP_ENDPOINT_OVERRIDE", "https://localhost:8080")
    assert _is_gcp_request_url("https://localhost:8080/api/2.0") is True
    assert _is_gcp_request_url("https://otherhost:8080/api/2.0") is False
