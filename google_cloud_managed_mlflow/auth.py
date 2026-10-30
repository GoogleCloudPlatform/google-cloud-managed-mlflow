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

"""Authentication and transport security for Google Cloud Managed MLflow."""

from __future__ import annotations

import ipaddress
import os
import re
import ssl
import threading
import urllib.parse
from collections.abc import MutableMapping
from typing import Any, ClassVar, NamedTuple, Protocol

import google.auth
import google.auth.exceptions
import google.auth.transport.requests
import requests
import requests.adapters
import requests.auth
import requests.utils
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE, UNAUTHENTICATED
from mlflow.tracking.request_auth.abstract_request_auth_provider import (
    RequestAuthProvider,
)

CLOUD_PLATFORM_SCOPE = "https://www.googleapis.com/auth/cloud-platform"
AUTH_PROVIDER_NAME = "google-cloud-auth"

_LOOPBACK_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})
_GOOGLEAPIS_DOMAIN_SUFFIX = ".googleapis.com"
_DNS_LABEL_RE = re.compile(r"^[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?$")
_UNBRACKETED_HOST_RE = re.compile(r"^[a-z0-9_-]+(?:\.[a-z0-9_-]+)*$")


class _EndpointOrigin(NamedTuple):
    """Normalized (scheme, hostname, effective_port) endpoint origin."""

    scheme: str
    hostname: str
    port: int

    def __str__(self) -> str:
        host_display = f"[{self.hostname}]" if ":" in self.hostname else self.hostname
        return f"{self.scheme}://{host_display}:{self.port}"


def _contains_control_chars(value: str) -> bool:
    """Return True if value contains ASCII C0 or DEL control characters.

    Must run before urllib.parse.urlsplit/urljoin, which silently strip
    '\\t', '\\r', and '\\n' (bpo-43882).
    """
    return any(ord(ch) < 0x20 or ord(ch) == 0x7F for ch in value)


class CredentialsProtocol(Protocol):
    """Structural interface for Google Cloud request credentials."""

    def before_request(
        self,
        request: Any,
        method: str,
        url: str,
        headers: MutableMapping[str, str],
    ) -> None:
        """Populate authentication headers prior to sending a request."""


def is_truthy_env(var_name: str) -> bool:
    """Return True if an environment variable has a truthy string value."""
    return os.environ.get(var_name, "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _is_tls_verification_disabled(verify: Any) -> bool:
    """Return True if TLS verification is explicitly disabled or empty."""
    return not verify or (isinstance(verify, str) and not verify.strip())


def _parse_validated_endpoint_url(raw_url: str) -> _EndpointOrigin:
    """Parse and validate URL scheme, host, effective port, and HTTP policy."""
    if not raw_url:
        raise MlflowException(
            "Endpoint URL must be a non-empty string.",
            error_code=INVALID_PARAMETER_VALUE,
        )

    if _contains_control_chars(raw_url):
        raise MlflowException(
            "Endpoint URL contains invalid control characters.",
            error_code=INVALID_PARAMETER_VALUE,
        )

    if raw_url[0].isspace() or raw_url[-1].isspace():
        raise MlflowException(
            "Endpoint URL must be a non-empty string without "
            "leading/trailing whitespace.",
            error_code=INVALID_PARAMETER_VALUE,
        )

    try:
        parsed = urllib.parse.urlsplit(raw_url)
        raw_port = parsed.port
    except ValueError as exc:
        raise MlflowException(
            "Endpoint URL is malformed or contains an invalid port.",
            error_code=INVALID_PARAMETER_VALUE,
        ) from exc

    if raw_port == 0:
        raise MlflowException(
            "Endpoint URL is malformed or contains an invalid port.",
            error_code=INVALID_PARAMETER_VALUE,
        )

    raw_hostname = (parsed.hostname or "").lower()
    if (
        not parsed.scheme
        or not raw_hostname.rstrip(".")
        or "@" in parsed.netloc
        or "\\" in parsed.netloc
        or any(ch.isspace() for ch in parsed.netloc)
    ):
        raise MlflowException(
            "Endpoint URL must include a valid scheme and host.",
            error_code=INVALID_PARAMETER_VALUE,
        )

    if "[" in parsed.netloc or "]" in parsed.netloc:
        try:
            if "%" in raw_hostname:
                raise ValueError("IPv6 zone identifiers are not supported.")
            hostname = ipaddress.IPv6Address(raw_hostname).compressed
        except ValueError as exc:
            raise MlflowException(
                "Endpoint URL contains an invalid IPv6 host literal.",
                error_code=INVALID_PARAMETER_VALUE,
            ) from exc
    else:
        hostname = raw_hostname.rstrip(".")
        if not _UNBRACKETED_HOST_RE.fullmatch(hostname):
            raise MlflowException(
                "Endpoint URL contains invalid characters in host.",
                error_code=INVALID_PARAMETER_VALUE,
            )

    scheme = parsed.scheme.lower()
    if scheme not in ("https", "http"):
        raise MlflowException(
            f"Invalid scheme '{scheme}': only https:// and loopback http:// "
            "are allowed.",
            error_code=INVALID_PARAMETER_VALUE,
        )

    if scheme == "http" and hostname not in _LOOPBACK_HOSTS:
        raise MlflowException(
            f"Insecure HTTP is only permitted on loopback addresses "
            f"({', '.join(sorted(_LOOPBACK_HOSTS))}); got '{hostname}'.",
            error_code=INVALID_PARAMETER_VALUE,
        )

    effective_port = (
        raw_port if raw_port is not None else (443 if scheme == "https" else 80)
    )
    return _EndpointOrigin(scheme, hostname, effective_port)


def _is_allowed_destination(origin: _EndpointOrigin) -> bool:
    """Return True if origin is an allowed Google Cloud or override destination."""
    if (
        origin.scheme == "https"
        and origin.port == 443
        and origin.hostname.endswith(_GOOGLEAPIS_DOMAIN_SUFFIX)
        and all(_DNS_LABEL_RE.fullmatch(label) for label in origin.hostname.split("."))
    ):
        return True

    override_raw = os.environ.get("MLFLOW_GCP_ENDPOINT_OVERRIDE", "").strip(" ")
    if override_raw:
        try:
            override_origin = _parse_validated_endpoint_url(override_raw)
        except MlflowException:
            return False
        return origin == override_origin

    return False


def validate_transport_security(
    endpoint_url: str, *, ignore_tls_verification: bool = False
) -> None:
    """Validate endpoint URL scheme, origin, and TLS enforcement.

    Allows ``https://*.googleapis.com`` (effective port 443) by default, or the
    exact ``(scheme, host, effective_port)`` origin configured via
    ``MLFLOW_GCP_ENDPOINT_OVERRIDE`` (required to opt into custom or loopback
    ``localhost`` / ``127.0.0.1`` / ``::1`` endpoints).

    Raises:
        MlflowException: If ``ignore_tls_verification`` is True or
            ``MLFLOW_TRACKING_INSECURE_TLS`` is truthy, scheme/port/host is
            malformed, HTTP is used for non-loopback hosts, or the origin is
            not allowlisted.
    """
    if ignore_tls_verification or is_truthy_env("MLFLOW_TRACKING_INSECURE_TLS"):
        raise MlflowException(
            "Disabling TLS verification is prohibited for Google Cloud endpoints "
            "(MLFLOW_TRACKING_INSECURE_TLS).",
            error_code=INVALID_PARAMETER_VALUE,
        )

    origin = _parse_validated_endpoint_url(endpoint_url)

    if not _is_allowed_destination(origin):
        raise MlflowException(
            f"Refusing to send Google Cloud credentials to untrusted origin "
            f"'{origin}' (untrusted host or port). "
            "Allowed destinations are https://*.googleapis.com (port 443) or "
            "the exact scheme, host, and port configured in "
            "MLFLOW_GCP_ENDPOINT_OVERRIDE (required to opt into loopback "
            "endpoints).",
            error_code=INVALID_PARAMETER_VALUE,
        )


def _resolve_redirect_url(base_url: str, raw_location: str) -> str:
    """Resolve a raw HTTP Location header using requests' redirect rules.

    Mirrors SessionRedirectMixin.get_redirect_target (latin1 -> utf-8) and
    resolve_redirects (requote_uri before urljoin on relative redirects,
    unquoting unreserved escapes like '%2e' -> '.' and '%68' -> 'h') so
    validation checks the exact URL requests follows.
    """
    try:
        decoded_location = raw_location.encode("latin1").decode("utf8")
    except UnicodeError as exc:
        raise MlflowException(
            "Redirect Location header is malformed.",
            error_code=INVALID_PARAMETER_VALUE,
        ) from exc

    if (
        not decoded_location
        or _contains_control_chars(decoded_location)
        or decoded_location[0].isspace()
        or decoded_location[-1].isspace()
    ):
        raise MlflowException(
            "Redirect Location header is malformed.",
            error_code=INVALID_PARAMETER_VALUE,
        )

    try:
        redirect_url = decoded_location
        if redirect_url.startswith("//"):
            parsed_base = urllib.parse.urlparse(base_url)
            redirect_url = f"{parsed_base.scheme}:{redirect_url}"
        parsed_redirect = urllib.parse.urlparse(redirect_url)
        redirect_url = parsed_redirect.geturl()
        if not parsed_redirect.netloc:
            return urllib.parse.urljoin(
                base_url, requests.utils.requote_uri(redirect_url)
            )
        return requests.utils.requote_uri(redirect_url)
    except ValueError as exc:
        raise MlflowException(
            "Redirect Location header is malformed.",
            error_code=INVALID_PARAMETER_VALUE,
        ) from exc


def _validate_redirect_target(response: requests.Response, **kwargs: Any) -> None:
    """Validate redirect Location targets before requests follows them."""
    if not response.is_redirect:
        return

    location = response.headers.get("Location", "")
    try:
        target_url = _resolve_redirect_url(response.url or "", location)
        validate_transport_security(target_url)
    except BaseException:
        if getattr(response, "raw", None) is not None:
            response.close()
        raise


def _create_tls12_ssl_context() -> ssl.SSLContext:
    """Create an SSL context enforcing TLS 1.2+ minimum protocol version."""
    ctx = ssl.create_default_context()
    ctx.minimum_version = ssl.TLSVersion.TLSv1_2
    if getattr(ctx, "post_handshake_auth", None) is not None:
        ctx.post_handshake_auth = True
    return ctx


def _build_pool_ssl_context(verify: Any = True, cert: Any = None) -> ssl.SSLContext:
    """Build an isolated TLS 1.2+ SSLContext for a specific (verify, cert) pair."""
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    ctx.verify_mode = ssl.CERT_REQUIRED
    ctx.check_hostname = True
    ctx.minimum_version = ssl.TLSVersion.TLSv1_2
    if getattr(ctx, "post_handshake_auth", None) is not None:
        ctx.post_handshake_auth = True

    ca_loc = (
        verify if isinstance(verify, str) else requests.utils.DEFAULT_CA_BUNDLE_PATH
    )
    if ca_loc:
        if os.path.isdir(ca_loc):
            ctx.load_verify_locations(capath=ca_loc)
        elif os.path.exists(ca_loc):
            ctx.load_verify_locations(cafile=ca_loc)
    else:
        ctx.load_default_certs()

    if cert is not None:
        if isinstance(cert, tuple) and len(cert) == 2:
            cert_path, key_path = cert
            if os.path.exists(cert_path) and os.path.exists(key_path):
                ctx.load_cert_chain(certfile=cert_path, keyfile=key_path)
        elif isinstance(cert, (str, bytes)) and os.path.exists(cert):
            ctx.load_cert_chain(certfile=cert)
    return ctx


def _cert_paths_exist(verify: Any, cert: Any) -> bool:
    """Return True if user-supplied verify/cert paths exist on disk."""
    if isinstance(verify, str) and not os.path.exists(verify):
        return False
    if isinstance(cert, tuple) and len(cert) == 2:
        return os.path.exists(cert[0]) and os.path.exists(cert[1])
    if isinstance(cert, (str, bytes)):
        return os.path.exists(cert)
    return True


class _Tls12HttpAdapter(requests.adapters.HTTPAdapter):
    """HTTPAdapter enforcing TLS 1.2+ and isolated per-CA verification."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self._ssl_contexts: dict[tuple[Any, Any], ssl.SSLContext] = {}
        self._ssl_lock = threading.Lock()
        super().__init__(*args, **kwargs)

    def init_poolmanager(
        self,
        connections: int,
        maxsize: int,
        block: bool = requests.adapters.DEFAULT_POOLBLOCK,
        **pool_kwargs: Any,
    ) -> None:
        pool_kwargs.setdefault("ssl_context", _create_tls12_ssl_context())
        super().init_poolmanager(connections, maxsize, block=block, **pool_kwargs)

    def proxy_manager_for(self, proxy: str, **proxy_kwargs: Any) -> Any:
        proxy_kwargs.setdefault("ssl_context", _create_tls12_ssl_context())
        proxy_kwargs.setdefault("proxy_ssl_context", _create_tls12_ssl_context())
        return super().proxy_manager_for(proxy, **proxy_kwargs)

    def build_connection_pool_key_attributes(
        self,
        request: requests.PreparedRequest,
        verify: bool | str,
        cert: str | tuple[str, str] | None = None,
    ) -> Any:
        host_params, pool_kwargs = super().build_connection_pool_key_attributes(
            request, verify, cert
        )
        raw_pool_kwargs: dict[str, Any] = dict(pool_kwargs)
        key = (verify, cert)
        with self._ssl_lock:
            ctx = self._ssl_contexts.get(key)
            if ctx is None:
                ctx = _build_pool_ssl_context(verify=verify, cert=cert)
                if _cert_paths_exist(verify, cert):
                    self._ssl_contexts[key] = ctx
        raw_pool_kwargs["ssl_context"] = ctx
        return host_params, raw_pool_kwargs

    def cert_verify(self, conn: Any, url: str, verify: Any, cert: Any) -> None:
        # Delegate to HTTPAdapter.cert_verify (untyped in types-requests) to set
        # conn.cert_reqs = "CERT_REQUIRED" and raise OSError on missing CA/cert
        # paths, then clear per-connection cert paths so urllib3 uses the isolated
        # SSLContext from build_connection_pool_key_attributes without reloading.
        base_cert_verify: Any = super().cert_verify
        base_cert_verify(conn, url, verify, cert)
        conn.ca_certs = None
        conn.ca_cert_dir = None
        conn.cert_file = None
        conn.key_file = None

    def send(
        self,
        request: requests.PreparedRequest,
        stream: bool = False,
        timeout: Any = None,
        verify: Any = True,
        cert: Any = None,
        proxies: Any = None,
    ) -> requests.Response:
        if _is_tls_verification_disabled(verify):
            raise MlflowException(
                "TLS verification cannot be disabled for Google Cloud connections.",
                error_code=INVALID_PARAMETER_VALUE,
            )
        return super().send(
            request,
            stream=stream,
            timeout=timeout,
            verify=verify,
            cert=cert,
            proxies=proxies,
        )


def create_secure_session() -> requests.Session:
    """Create a requests.Session enforcing TLS 1.2+ and certificate verification.

    Callers should close the returned session (or use it as a context manager)
    when finished. Used internally for ADC token refresh; callers needing
    session-level TLS 1.2+ enforcement on tracking, registry, or artifact
    traffic must use this session explicitly, as stock MLflow RestStore traffic
    uses MLflow's default session.
    """
    session = requests.Session()
    adapter = _Tls12HttpAdapter()
    session.mount("https://", adapter)
    return session


class GoogleCloudBearerAuth(requests.auth.AuthBase):
    """Requests auth handler using Google Cloud Application Default Credentials.

    Validates scheme, destination origin (allowing ``https://*.googleapis.com``
    at port 443 or the exact origin configured in ``MLFLOW_GCP_ENDPOINT_OVERRIDE``),
    ``MLFLOW_TRACKING_INSECURE_TLS``, and HTTP redirect targets on PreparedRequest,
    and refreshes ADC tokens over a TLS 1.2+ session. Note: requests.auth.AuthBase
    runs during prepare_request() and does not inspect Session.send(..., verify=...);
    TLS 1.2+ is enforced here only for ADC token refresh, so callers needing
    session-level TLS 1.2+ enforcement on tracking or registry traffic must use
    create_secure_session() themselves.
    """

    _instance: ClassVar[GoogleCloudBearerAuth | None] = None
    _instance_lock: ClassVar[threading.Lock] = threading.Lock()

    def __init__(self, credentials: CredentialsProtocol | None = None) -> None:
        self._lock = threading.Lock()
        self._creds: CredentialsProtocol | None = credentials
        self._creds_from_adc: bool = False
        self._auth_request: google.auth.transport.requests.Request | None = None

    def _close_auth_request_locked(self) -> None:
        """Close and clear the cached auth transport session (caller holds _lock)."""
        auth_request: Any = self._auth_request
        if auth_request is not None:
            session = getattr(auth_request, "session", None)
            if session is not None:
                session.close()
                auth_request.session = None
            self._auth_request = None

    def close(self) -> None:
        """Close any cached token-refresh session owned by this instance."""
        with self._lock:
            self._close_auth_request_locked()

    @classmethod
    def get_instance(cls) -> GoogleCloudBearerAuth:
        """Return the shared singleton GoogleCloudBearerAuth instance."""
        if cls._instance is None:
            with cls._instance_lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    @classmethod
    def reset_for_testing(cls) -> None:
        """Reset the singleton instance for testing isolation."""
        with cls._instance_lock:
            if cls._instance is not None:
                with cls._instance._lock:
                    cls._instance._close_auth_request_locked()
            cls._instance = None

    @property
    def credentials(self) -> CredentialsProtocol | None:
        """Return the active credentials instance, or None if not yet initialized."""
        with self._lock:
            return self._creds

    @credentials.setter
    def credentials(self, value: CredentialsProtocol | None) -> None:
        """Replace active credentials and close any cached refresh session.

        Setting None clears cached credentials so the next request re-resolves
        Application Default Credentials via google.auth.default().
        """
        with self._lock:
            self._close_auth_request_locked()
            self._creds = value
            self._creds_from_adc = False

    def __repr__(self) -> str:
        with self._lock:
            status = "set" if self._creds is not None else "unset"
        return f"<GoogleCloudBearerAuth credentials={status}>"

    def __call__(self, request: requests.PreparedRequest) -> requests.PreparedRequest:
        """Validate transport security and inject Google Cloud Bearer headers.

        Raises:
            MlflowException: With INVALID_PARAMETER_VALUE if the target URL or
                TLS settings violate transport policy, or UNAUTHENTICATED if
                Application Default Credentials cannot be obtained or refreshed.
        """
        validate_transport_security(request.url or "")
        response_hooks = request.hooks.setdefault("response", [])
        if _validate_redirect_target not in response_hooks:
            response_hooks.append(_validate_redirect_target)

        with self._lock:
            using_adc = self._creds is None or self._creds_from_adc
            try:
                creds = self._creds
                if creds is None:
                    creds, _ = google.auth.default(scopes=[CLOUD_PLATFORM_SCOPE])
                if self._auth_request is None:
                    self._auth_request = google.auth.transport.requests.Request(
                        session=create_secure_session()
                    )
                creds.before_request(
                    self._auth_request,
                    request.method or "GET",
                    request.url or "",
                    request.headers,
                )
                self._creds = creds
                self._creds_from_adc = using_adc
            except google.auth.exceptions.GoogleAuthError as exc:
                if using_adc:
                    self._creds = None
                    self._creds_from_adc = False
                    self._close_auth_request_locked()
                raise MlflowException(
                    "Failed to obtain Google Cloud Application Default Credentials "
                    f"(ADC): {exc}. Run 'gcloud auth application-default login' "
                    "or set the GOOGLE_APPLICATION_CREDENTIALS environment variable.",
                    error_code=UNAUTHENTICATED,
                ) from exc
        return request


class GoogleCloudAuthProvider(RequestAuthProvider):
    """MLflow request auth provider plugin for Google Cloud."""

    def get_name(self) -> str:
        """Return the auth provider name."""
        return AUTH_PROVIDER_NAME

    def get_auth(self) -> GoogleCloudBearerAuth:
        """Return the GoogleCloudBearerAuth instance."""
        return GoogleCloudBearerAuth.get_instance()


def _has_google_cloud_auth_provider(registry: Any) -> bool:
    """Return True if google-cloud-auth is already present in registry."""
    return any(
        getattr(provider, "get_name", lambda: None)() == AUTH_PROVIDER_NAME
        for provider in registry
    )


def ensure_auth_provider_registered() -> bool:
    """Ensure GoogleCloudAuthProvider is registered with MLflow's auth registry.

    Returns True when the provider is present or newly registered, or False if
    MLflow's request_auth registry internals are unavailable.
    """
    try:
        from mlflow.tracking.request_auth.registry import (
            _request_auth_provider_registry,
        )

        if _has_google_cloud_auth_provider(_request_auth_provider_registry):
            return True

        # MLflow's RequestAuthProviderRegistry implements __iter__ without __len__
        # or __bool__ (so `not registry` is always False), and register() appends
        # without deduplicating; only load entry points when the registry is empty.
        if next(iter(_request_auth_provider_registry), None) is None:
            register_entrypoints = getattr(
                _request_auth_provider_registry, "register_entrypoints", None
            )
            if callable(register_entrypoints):
                register_entrypoints()
            if _has_google_cloud_auth_provider(_request_auth_provider_registry):
                return True

        _request_auth_provider_registry.register(GoogleCloudAuthProvider)
        return True
    except (ImportError, AttributeError, TypeError):
        return False
