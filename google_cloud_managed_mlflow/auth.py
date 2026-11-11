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

import logging
import os
import ssl
import threading
import urllib.parse
from collections.abc import MutableMapping
from typing import Any, ClassVar, Protocol

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

from google_cloud_managed_mlflow._validation import (
    _contains_control_chars,
    _is_allowed_destination,
    _parse_validated_endpoint_url,
)

_logger = logging.getLogger(__name__)

CLOUD_PLATFORM_SCOPE = "https://www.googleapis.com/auth/cloud-platform"
AUTH_PROVIDER_NAME = "google-cloud-auth"

AUTH_MODE_ENV_VAR = "GOOGLE_CLOUD_MANAGED_MLFLOW_AUTH_MODE"
AUTH_MODE_AUTO = "auto"
AUTH_MODE_CAB = "cab"
AUTH_MODE_EUC = "euc"
VALID_AUTH_MODES: frozenset[str] = frozenset(
    {AUTH_MODE_AUTO, AUTH_MODE_CAB, AUTH_MODE_EUC}
)


def parse_auth_mode() -> str:
    """Return the validated auth mode configured in the environment."""
    mode = os.environ.get(AUTH_MODE_ENV_VAR, AUTH_MODE_AUTO).strip().lower()
    if mode not in VALID_AUTH_MODES:
        raise MlflowException(
            f"Invalid {AUTH_MODE_ENV_VAR} '{mode}'. "
            f"Expected one of: {', '.join(sorted(VALID_AUTH_MODES))}.",
            error_code=INVALID_PARAMETER_VALUE,
        )
    return mode


__all__ = [
    "AUTH_MODE_AUTO",
    "AUTH_MODE_CAB",
    "AUTH_MODE_ENV_VAR",
    "AUTH_MODE_EUC",
    "AUTH_PROVIDER_NAME",
    "CLOUD_PLATFORM_SCOPE",
    "CredentialsProtocol",
    "GoogleCloudAuthProvider",
    "GoogleCloudBearerAuth",
    "VALID_AUTH_MODES",
    "create_secure_session",
    "ensure_auth_provider_registered",
    "harden_mlflow_request_session",
    "is_truthy_env",
    "parse_auth_mode",
    "validate_transport_security",
]


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


def _is_gcp_request_url(url: str | None) -> bool:
    """Return True if url targets a Google Cloud or override endpoint.

    Fails closed: if parsing fails, returns True so that transport security
    and TLS checks are never bypassed on malformed or ambiguous URLs.
    """
    if not url:
        return False
    try:
        parsed = urllib.parse.urlsplit(url)
        hostname = (parsed.hostname or "").lower()
    except Exception:
        # Fail closed on unparseable URLs so TLS verification is not bypassed
        return True

    if not hostname:
        return False

    override = os.environ.get("MLFLOW_GCP_ENDPOINT_OVERRIDE", "").strip()
    if override:
        try:
            override_host = (urllib.parse.urlsplit(override).hostname or "").lower()
            if override_host and hostname == override_host:
                return True
        except Exception:
            return True

    return hostname == "googleapis.com" or hostname.endswith(".googleapis.com")


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


def _normalize_tls_paths(
    verify: bool | str | bytes | os.PathLike[Any],
    cert: Any = None,
) -> tuple[Any, Any]:
    """Normalize verify and cert paths to strings/tuples for pooling and SSLContext."""
    norm_verify: Any = (
        os.fspath(verify) if isinstance(verify, (str, bytes, os.PathLike)) else verify
    )
    if isinstance(norm_verify, bytes):
        norm_verify = os.fsdecode(norm_verify)

    norm_cert: Any = cert
    if isinstance(cert, (str, bytes, os.PathLike)):
        norm_cert = os.fsdecode(os.fspath(cert))
    elif isinstance(cert, (list, tuple)) and len(cert) == 2:
        norm_cert = (
            os.fsdecode(os.fspath(cert[0])),
            os.fsdecode(os.fspath(cert[1])),
        )
    elif isinstance(cert, list):
        norm_cert = tuple(cert)

    return norm_verify, norm_cert


def _build_pool_ssl_context(verify: Any = True, cert: Any = None) -> ssl.SSLContext:
    """Build an isolated TLS 1.2+ SSLContext for a specific (verify, cert) pair."""
    if _is_tls_verification_disabled(verify):
        ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        ctx.minimum_version = ssl.TLSVersion.TLSv1_2
        if getattr(ctx, "post_handshake_auth", None) is not None:
            ctx.post_handshake_auth = True
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        return ctx

    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    ctx.verify_mode = ssl.CERT_REQUIRED
    ctx.check_hostname = True
    ctx.minimum_version = ssl.TLSVersion.TLSv1_2
    if getattr(ctx, "post_handshake_auth", None) is not None:
        ctx.post_handshake_auth = True

    norm_verify, normalized_cert = _normalize_tls_paths(verify, cert)
    if isinstance(norm_verify, str):
        ca_loc: Any = norm_verify
    else:
        ca_loc = requests.utils.DEFAULT_CA_BUNDLE_PATH
    if ca_loc:
        if os.path.isdir(ca_loc):
            ctx.load_verify_locations(capath=ca_loc)
        elif os.path.exists(ca_loc):
            ctx.load_verify_locations(cafile=ca_loc)
    else:
        ctx.load_default_certs()

    if normalized_cert is not None:
        if isinstance(normalized_cert, tuple) and len(normalized_cert) == 2:
            cert_path, key_path = normalized_cert
            if os.path.exists(cert_path) and os.path.exists(key_path):
                ctx.load_cert_chain(certfile=cert_path, keyfile=key_path)
        elif isinstance(normalized_cert, str) and os.path.exists(normalized_cert):
            ctx.load_cert_chain(certfile=normalized_cert)
    return ctx


def _cert_paths_exist(verify: Any, cert: Any) -> bool:
    """Return True if user-supplied verify/cert paths exist on disk."""
    if isinstance(verify, (str, bytes, os.PathLike)) and not os.path.exists(verify):
        return False
    if isinstance(cert, (tuple, list)) and len(cert) == 2:
        return os.path.exists(cert[0]) and os.path.exists(cert[1])
    if isinstance(cert, (str, bytes, os.PathLike)):
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
        verify: bool | str | bytes | os.PathLike[Any],
        cert: Any = None,
    ) -> Any:
        norm_verify, norm_cert = _normalize_tls_paths(verify, cert)

        host_params, pool_kwargs = super().build_connection_pool_key_attributes(
            request, norm_verify, norm_cert
        )
        raw_pool_kwargs: dict[str, Any] = dict(pool_kwargs)
        key = (norm_verify, norm_cert)
        with self._ssl_lock:
            ctx = self._ssl_contexts.get(key)
            if ctx is None:
                ctx = _build_pool_ssl_context(verify=norm_verify, cert=norm_cert)
                if _cert_paths_exist(norm_verify, norm_cert):
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
        url = getattr(request, "url", None)
        if _is_tls_verification_disabled(verify) and _is_gcp_request_url(url):
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


def harden_mlflow_request_session() -> bool:
    """Harden MLflow's REST sessions to enforce TLS 1.2+ transport security.

    Ensures requests made by MLflow's RestStore use a TLS 1.2+ adapter for https.
    """
    try:
        import functools

        target_modules: list[Any] = []
        try:
            import mlflow.utils.request_utils as req_utils

            target_modules.append(req_utils)
        except ImportError:
            pass
        try:
            import mlflow.utils.rest_utils as rest_utils

            target_modules.append(rest_utils)
        except ImportError:
            pass

        def _create_hardened(func: Any) -> Any:
            @functools.wraps(func)
            def _hardened_cached_session(*args: Any, **kwargs: Any) -> requests.Session:
                session: requests.Session = func(*args, **kwargs)
                adapter = session.adapters.get("https://")
                if not isinstance(adapter, _Tls12HttpAdapter):
                    max_retries = getattr(
                        adapter, "max_retries", requests.adapters.DEFAULT_RETRIES
                    )
                    pool_connections = getattr(
                        adapter, "_pool_connections", requests.adapters.DEFAULT_POOLSIZE
                    )
                    pool_maxsize = getattr(
                        adapter, "_pool_maxsize", requests.adapters.DEFAULT_POOLSIZE
                    )
                    pool_block = getattr(
                        adapter, "_pool_block", requests.adapters.DEFAULT_POOLBLOCK
                    )
                    session.mount(
                        "https://",
                        _Tls12HttpAdapter(
                            max_retries=max_retries,
                            pool_connections=pool_connections,
                            pool_maxsize=pool_maxsize,
                            pool_block=pool_block,
                        ),
                    )
                return session

            _hardened_cached_session._is_hardened = True  # type: ignore[attr-defined]
            if hasattr(func, "cache_clear"):
                _hardened_cached_session.cache_clear = (  # type: ignore[attr-defined]
                    func.cache_clear
                )
            if hasattr(func, "cache_info"):
                _hardened_cached_session.cache_info = (  # type: ignore[attr-defined]
                    func.cache_info
                )
            return _hardened_cached_session

        for mod in target_modules:
            cached_func = getattr(mod, "_cached_get_request_session", None)
            if cached_func is not None and not getattr(
                cached_func, "_is_hardened", False
            ):
                hardened = _create_hardened(cached_func)
                mod._cached_get_request_session = hardened
                if hasattr(cached_func, "cache_clear"):
                    cached_func.cache_clear()
        return True
    except Exception as exc:
        _logger.warning(
            "Failed to harden MLflow REST session transport security: %s", exc
        )
        return False
