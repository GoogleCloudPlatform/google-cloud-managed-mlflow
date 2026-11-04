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

"""GCS artifact repository for Google Cloud Managed Service for MLflow."""

from __future__ import annotations

import contextlib
import importlib.metadata
import logging
import os
import posixpath
import re
import sys
import threading
import urllib.parse
import warnings
from collections.abc import Callable, Iterator
from typing import TYPE_CHECKING, Any, Final, Literal, TypeVar, cast

import google.api_core.exceptions
from google.cloud import storage as gcs_storage

_GS_REGISTRATION_WARNING = (
    r'Failure attempting to register artifact repository for scheme "gs"'
)
_IMPORTED_BEFORE_MLFLOW = "mlflow" not in sys.modules


@contextlib.contextmanager
def _quiet_gs_registration_warning() -> Iterator[None]:
    """Hide MLflow's "gs" registration warning while this module imports mlflow.

    If this module is the first to import mlflow, MLflow loads the "gs" entry
    point while this module is only partially initialized, warns "Failure
    attempting to register artifact repository for scheme \"gs\"" (an error under
    ``-W error``) and keeps its stock class; _ensure_gs_registered() repairs the
    registration at the bottom of this module. Only the filter added here is
    removed afterwards. Do not use warnings.catch_warnings(): it restores the
    whole filter list and would discard filters that mlflow and its
    dependencies install during their first import.
    """
    if "mlflow" in sys.modules:
        yield
        return
    warnings.filterwarnings(
        "ignore", message=_GS_REGISTRATION_WARNING, category=UserWarning
    )
    # With Python 3.14 context-aware warnings inside catch_warnings(), the new
    # filter goes to a context-local list, so warnings.filters[0] may belong to
    # someone else. Only remember the entry if it is the one just added.
    gs_filter: Any = None
    candidate = warnings.filters[0] if warnings.filters else None
    if (
        candidate is not None
        and candidate[0] == "ignore"
        and candidate[2] is UserWarning
        and getattr(candidate[1], "pattern", None) == _GS_REGISTRATION_WARNING
    ):
        gs_filter = candidate
    try:
        yield
    finally:
        if gs_filter is not None:
            # Remove by identity so that an equal filter added by others survives.
            filters = cast("list[Any]", warnings.filters)
            for index, item in enumerate(filters):
                if item is gs_filter:
                    del filters[index]
                    break


with _quiet_gs_registration_warning():
    from mlflow.exceptions import (
        MlflowException,
        _UnsupportedMultipartUploadException,
    )
    from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE
    from mlflow.store.artifact.gcs_artifact_repo import GCSArtifactRepository

# Import-cycle guard: MLflow loads the "gs" entry point while `import mlflow`
# is still running. If a caller imported google_cloud_managed_mlflow.auth (or
# .uri / .cab_credentials) first, those modules are only partially initialized
# at that point, so importing them at module scope here would make entry-point
# registration fail and MLflow would silently keep its stock GCS repository.
# Keep first-party imports deferred to call time; guarded by
# test_gs_registration_survives_first_import_order. If this module is
# itself imported first, MLflow cannot see the class yet; _ensure_gs_registered()
# at the bottom of this module repairs that case.
if TYPE_CHECKING:
    from google_cloud_managed_mlflow.cab_credentials import (
        CabAccessMode,
        CabTokenCredentials,
    )

AUTH_MODE_ENV_VAR = "GOOGLE_CLOUD_MANAGED_MLFLOW_AUTH_MODE"
AUTH_MODE_AUTO = "auto"
AUTH_MODE_CAB = "cab"
AUTH_MODE_EUC = "euc"
_SUPPORTED_AUTH_MODES = frozenset({AUTH_MODE_AUTO, AUTH_MODE_CAB, AUTH_MODE_EUC})

# Lowercase fragments of issue*ArtifactToken error messages, matched against
# str(exc).lower() by _fallback_reason. Message matching is a stopgap until the
# service exposes a machine-readable reason (AIP-193 ErrorInfo).
_SERVICE_AGENT_ERROR_SUBSTRINGS: Final[tuple[str, ...]] = (
    "service agent",
    "invalid service account configuration",
    "security token service rejected the token",
    "failed to get gcs credentials from security token service",
    "failed to generate access token for service account",
    "failed to get gcs credentials from iam credentials",
    "could not get credentials for service account",
    "does not have a valid service account configured",
)

_CONTAINMENT_ERROR_SUBSTRINGS: Final[tuple[str, ...]] = (
    "not contained within storage root",
    "not contained within namespace root",
    "does not have a storage_uri configured",
    "storage root uri must not be null or empty",
    "stored artifact root uri must not be null or empty",
)

_LIFECYCLE_ERROR_SUBSTRINGS: Final[tuple[str, ...]] = (
    "cannot issue read_write credentials",
    "cannot issue read-write credentials",
)

_MAX_CACHE_ENTRIES: Final[int] = 1024


def _bounded_add(cache: set[Any], item: Any) -> None:
    if len(cache) >= _MAX_CACHE_ENTRIES:
        cache.clear()
    cache.add(item)


_FallbackReason = Literal[
    "service_agent_provisioning",
    "token_rpc_unavailable",
    "storage_root_containment",
    "initial_storage_forbidden",
]

_SERVICE_AGENT_BUCKET_HINT = (
    " Server-side artifact features such as UI previews and model version "
    "creation will fail until the service agent is granted 'roles/storage.objectUser' "
    "on this bucket."
)
# The bucket-access hint only helps when the service agent is in play; it is
# noise when the token RPC is unavailable or the URI is outside the root.
_REASONS_WITHOUT_BUCKET_HINT: frozenset[_FallbackReason] = frozenset(
    {"token_rpc_unavailable", "storage_root_containment"}
)

logger = logging.getLogger(__name__)

# Dedup keys for the one-time fallback warning: (endpoint_target, bucket_name).
_WARNED_FALLBACK_TARGETS: set[tuple[str, str]] = set()
_WARNING_LOCK = threading.Lock()

_WARNED_CAB_INACTIVE_URIS: set[str] = set()
_WARNED_CAB_INACTIVE_LOCK = threading.Lock()

_PROVEN_CAB_ACCESS_TARGETS: set[tuple[str, str, CabAccessMode]] = set()
_PROVEN_CAB_ACCESS_LOCK = threading.Lock()

_NAMESPACE_RUN_OR_MODEL_URI_RE: Final[re.Pattern[str]] = re.compile(
    r"/(?:models/(?P<model_id>[a-zA-Z0-9_-]{1,64})"
    r"|runs/(?P<run_id>[a-zA-Z0-9_-]{1,64})"
    r"|(?P<hex_run_id>[0-9a-f]{32}))/artifacts(?:/(?P<subpath>.+))?$"
)


def _warn_cab_inactive_once(effective_tracking_uri: str) -> None:
    """Log a one-time warning in cab mode when CAB is inactive."""
    with _WARNED_CAB_INACTIVE_LOCK:
        if effective_tracking_uri in _WARNED_CAB_INACTIVE_URIS:
            return
        _bounded_add(_WARNED_CAB_INACTIVE_URIS, effective_tracking_uri)
    logger.warning(
        "%s=cab is set, but the effective tracking URI %r is not a Managed "
        "Service for MLflow namespace URI and no CAB credentials were provided. CAB is "
        "inactive; using ADC credentials. Configure a Managed Service for MLflow "
        "namespace tracking URI via mlflow.set_tracking_uri() or the "
        "MLFLOW_TRACKING_URI environment variable.",
        AUTH_MODE_ENV_VAR,
        effective_tracking_uri,
    )


T = TypeVar("T")


def _fallback_reason(
    exc: BaseException, *, cab_storage_succeeded: bool
) -> _FallbackReason | None:
    """Classify a CAB-backed storage failure for the auto-mode ADC fallback.

    This is the whole fallback policy. Callers consult it only in auto mode;
    cab mode always raises. A reason is returned when CAB cannot serve the
    request but the caller's own ADC may:

    (a) token RPC HTTP 400 or 403 with status FAILED_PRECONDITION or PERMISSION_DENIED
        (or no status) naming an unprovisioned service agent or storage root
        containment, or 500 / INTERNAL naming an unprovisioned service agent
        (e.g. STS or IAM credentials failure), unless it is a run/model
        lifecycle error ("Cannot issue READ_WRITE credentials"), which must surface;
    (b) token RPC 501 / UNIMPLEMENTED or 404 (with no GCP status): the server
        does not offer the RPC;
    (c) GCS Forbidden before any CAB-backed storage call on the repository has
        succeeded: tokens are minted without checking bucket IAM, so a service
        agent without bucket access fails on the first storage call. After a
        success, a 403 is a real authorization boundary and must surface.
    """
    # (c)
    if isinstance(exc, google.api_core.exceptions.Forbidden):
        return None if cab_storage_succeeded else "initial_storage_forbidden"

    from google_cloud_managed_mlflow.uri import GcpMlflowApiError

    if not isinstance(exc, GcpMlflowApiError):
        return None
    msg_lower = str(exc).lower()

    # (b)
    if (
        exc.http_status == 501
        or exc.gcp_status == "UNIMPLEMENTED"
        or (exc.http_status == 404 and exc.gcp_status is None)
    ):
        return "token_rpc_unavailable"

    # (a)
    if any(s in msg_lower for s in _LIFECYCLE_ERROR_SUBSTRINGS):
        return None

    if (
        exc.http_status in (400, 403)
        or exc.gcp_status in ("FAILED_PRECONDITION", "PERMISSION_DENIED")
    ) and any(s in msg_lower for s in _CONTAINMENT_ERROR_SUBSTRINGS):
        return "storage_root_containment"

    if (
        exc.http_status in (400, 403, 500)
        or exc.gcp_status in ("FAILED_PRECONDITION", "PERMISSION_DENIED", "INTERNAL")
    ) and any(s in msg_lower for s in _SERVICE_AGENT_ERROR_SUBSTRINGS):
        return "service_agent_provisioning"

    return None


def _is_object_key_within_path(object_key: str, path: str) -> bool:
    """Return whether an object-store key belongs to a path."""
    if not path:
        return True
    if path.endswith("/"):
        return object_key.startswith(path)
    return object_key == path or object_key.startswith(f"{path}/")


def _effective_uri(
    explicit: str | None, mlflow_getter: Callable[[], str | None], env_var: str
) -> str:
    """Return ``explicit``, else ``mlflow_getter()``, else the ``env_var`` value.

    The MLflow getter also honors mlflow.set_*_uri(). If it raises, the error is
    logged at DEBUG level and the environment variable is used instead.
    """
    if explicit:
        return explicit
    try:
        resolved = mlflow_getter()
        if resolved:
            return resolved
    except Exception:
        logger.debug("MLflow URI lookup failed; using %s.", env_var, exc_info=True)
    return os.environ.get(env_var, "")


def _get_effective_tracking_uri(tracking_uri: str | None) -> str:
    import mlflow

    return _effective_uri(
        tracking_uri, lambda: mlflow.get_tracking_uri(), "MLFLOW_TRACKING_URI"
    )


def _get_effective_registry_uri(registry_uri: str | None) -> str:
    import mlflow

    return _effective_uri(
        registry_uri, lambda: mlflow.get_registry_uri(), "MLFLOW_REGISTRY_URI"
    )


def _extract_namespace_cab_target(
    artifact_uri: str,
) -> tuple[str | None, str | None, str | None, str | None]:
    """Extract (target_uri, run_id, model_id, artifact_path) from artifact URI."""
    from google_cloud_managed_mlflow.cab_credentials import (
        derive_cab_directory_uri,
    )

    derived_uri = derive_cab_directory_uri(artifact_uri)
    parsed = urllib.parse.urlsplit(artifact_uri)
    raw_path = parsed.path.rstrip("/")
    if not artifact_uri.endswith("/") and raw_path.endswith("/MLmodel"):
        raw_path = raw_path[: -len("/MLmodel")]

    parsed_derived = urllib.parse.urlsplit(derived_uri)
    derived_path = parsed_derived.path.rstrip("/")

    match = _NAMESPACE_RUN_OR_MODEL_URI_RE.search(derived_path)
    if match:
        remainder = raw_path[len(derived_path) :].strip("/")
        subpath = remainder if remainder else None
        if match.group("model_id"):
            return (None, None, match.group("model_id"), subpath)
        extracted_run_id = match.group("run_id") or match.group("hex_run_id")
        return (None, extracted_run_id, None, subpath)

    return (derived_uri, None, None, None)


class GcpGCSArtifactRepository(GCSArtifactRepository):
    """GCS artifact repository for Google Cloud Managed Service for MLflow.

    CAB-downscoped credentials are used only when ``cab_credentials`` is injected
    or the Managed Service for MLflow namespace resolves from: (1) the explicit
    ``tracking_uri`` argument, if it is a Managed Service for MLflow namespace URI;
    (2) ``mlflow.get_tracking_uri()``; (3) the ``MLFLOW_TRACKING_URI``
    environment variable; otherwise MLflow's stock ADC-backed repository is
    used, in every mode. Registry URIs never enable CAB.
    GOOGLE_CLOUD_MANAGED_MLFLOW_AUTH_MODE selects ``auto`` (default: CAB,
    retried once with ADC when ``_fallback_reason`` allows it), ``cab``
    (never falls back to ADC once CAB is active; a Managed Service for MLflow
    registry URI without a namespace tracking URI raises INVALID_PARAMETER_VALUE)
    or ``euc`` (ADC only). An explicit ``client`` or ``credential_refresh_def``
    always bypasses CAB.

    Security note: after an auto-mode fallback the instance uses the caller's
    ADC for the rest of its lifetime, so neither least privilege nor server-side
    run-lifecycle enforcement applies to it; only cab mode guarantees both.
    """

    def __init__(
        self,
        artifact_uri: str,
        client: Any = None,
        credential_refresh_def: Any = None,
        tracking_uri: str | None = None,
        registry_uri: str | None = None,
        cab_credentials: CabTokenCredentials | None = None,
    ) -> None:
        from google_cloud_managed_mlflow.cab_credentials import CabAccessMode

        self._auth_mode = (
            os.environ.get(AUTH_MODE_ENV_VAR, AUTH_MODE_AUTO).strip().lower()
        )
        if self._auth_mode not in _SUPPORTED_AUTH_MODES:
            raise MlflowException(
                f"Invalid {AUTH_MODE_ENV_VAR} '{self._auth_mode}'. "
                f"Expected one of: {', '.join(sorted(_SUPPORTED_AUTH_MODES))}.",
                error_code=INVALID_PARAMETER_VALUE,
            )

        self._cab_credentials: CabTokenCredentials | None = None
        self._write_cab_credentials: CabTokenCredentials | None = None
        self._read_client: Any = None
        self._write_client: Any = None
        self._proven_cab_modes: set[CabAccessMode] = set()
        self._op_local = threading.local()
        self._fallback_lock = threading.Lock()
        self._endpoint_target = ""

        # An explicit client or refresh hook, or euc mode, keeps MLflow's own
        # credentials; otherwise CAB applies when configured.
        if (
            client is None
            and credential_refresh_def is None
            and self._auth_mode != AUTH_MODE_EUC
        ):
            if cab_credentials is None:
                cab_credentials = self._namespace_cab_credentials(
                    artifact_uri, tracking_uri, registry_uri
                )
            if cab_credentials is not None:
                self._cab_credentials = cab_credentials
                client = self._attach_cab_client(
                    cab_credentials, mode=CabAccessMode.READ_ONLY
                )
                self._read_client = client

        self._init_super(
            artifact_uri,
            client=client,
            credential_refresh_def=credential_refresh_def,
            tracking_uri=tracking_uri,
            registry_uri=registry_uri,
        )
        if self._read_client is None:
            self._read_client = getattr(self, "client", client)

    def _namespace_cab_credentials(
        self,
        artifact_uri: str,
        tracking_uri: str | None,
        registry_uri: str | None = None,
    ) -> CabTokenCredentials | None:
        """Resolve and construct CAB credentials for a namespace, or None."""
        from google_cloud_managed_mlflow.cab_credentials import (
            CabAccessMode,
            CabResourceKind,
            CabTokenCredentials,
        )
        from google_cloud_managed_mlflow.uri import (
            is_gcp_registry_uri,
            is_gcp_tracking_uri,
            parse_gcp_uri,
        )

        effective_tracking_uri = _get_effective_tracking_uri(tracking_uri)
        if not effective_tracking_uri or not is_gcp_tracking_uri(
            effective_tracking_uri
        ):
            if self._auth_mode == AUTH_MODE_CAB:
                effective_registry_uri = _get_effective_registry_uri(registry_uri)
                if is_gcp_registry_uri(effective_registry_uri) or is_gcp_registry_uri(
                    effective_tracking_uri
                ):
                    raise MlflowException(
                        f"{AUTH_MODE_ENV_VAR}=cab requires a Managed "
                        "Service for MLflow namespace tracking URI; "
                        "registered-model artifact credentials "
                        "are not supported yet. Use auto or euc mode.",
                        error_code=INVALID_PARAMETER_VALUE,
                    )
                _warn_cab_inactive_once(effective_tracking_uri)
            return None

        parsed_tracking = parse_gcp_uri(effective_tracking_uri)
        target_uri, run_id, model_id, artifact_path = _extract_namespace_cab_target(
            artifact_uri
        )

        return CabTokenCredentials(
            resource_url=parsed_tracking.resource_url,
            resource_kind=CabResourceKind.NAMESPACE,
            access_mode=CabAccessMode.READ_ONLY,
            artifact_uri=target_uri,
            run_id=run_id,
            model_id=model_id,
            artifact_path=artifact_path,
        )

    def _wrap_client_http_request(
        self, client: Any, mode: CabAccessMode, creds: Any
    ) -> None:
        """Wrap client._http.request to observe successful 2xx responses."""
        http = getattr(client, "_http", None)
        if http is None:
            return
        orig_request = getattr(http, "request", None)
        if not callable(orig_request):
            return
        if getattr(orig_request, "_is_cab_wrapped", False):
            return

        def wrapped_request(*args: Any, **kwargs: Any) -> Any:
            resp = orig_request(*args, **kwargs)
            status = getattr(resp, "status_code", None)
            if isinstance(status, int) and 200 <= status < 300:
                effective_mode = getattr(creds, "access_mode", None) or mode
                self._mark_cab_access_proven(effective_mode, creds)
            return resp

        wrapped_request._is_cab_wrapped = True  # type: ignore[attr-defined]
        try:
            http.request = wrapped_request
        except (AttributeError, TypeError):
            pass

    def _cab_access_cache_key(
        self, mode: CabAccessMode, creds: Any
    ) -> tuple[str, str, CabAccessMode]:
        res_url = getattr(
            creds,
            "resource_url",
            getattr(creds, "_resource_url", self._endpoint_target),
        )
        try:
            repo_any: Any = self
            bucket_name, _ = repo_any.parse_gcs_uri(self.artifact_uri)
        except Exception:
            bucket_name = ""
        return (res_url, bucket_name, mode)

    def _mark_cab_access_proven(self, mode: CabAccessMode, creds: Any) -> None:
        self._proven_cab_modes.add(mode)
        key = self._cab_access_cache_key(mode, creds)
        with _PROVEN_CAB_ACCESS_LOCK:
            _bounded_add(_PROVEN_CAB_ACCESS_TARGETS, key)

    def _is_cab_access_proven(self, mode: CabAccessMode, creds: Any) -> bool:
        if mode in self._proven_cab_modes:
            return True
        key = self._cab_access_cache_key(mode, creds)
        with _PROVEN_CAB_ACCESS_LOCK:
            return key in _PROVEN_CAB_ACCESS_TARGETS

    @property
    def _has_cab_gcs_succeeded(self) -> bool:
        from google_cloud_managed_mlflow.cab_credentials import CabAccessMode

        mode = getattr(self._op_local, "access_mode", None)
        if mode is None:
            if (
                getattr(self._cab_credentials, "access_mode", None)
                == CabAccessMode.READ_WRITE
            ):
                mode = CabAccessMode.READ_WRITE
            else:
                mode = CabAccessMode.READ_ONLY
        creds = (
            self._write_cab_credentials
            if mode == CabAccessMode.READ_WRITE
            else self._cab_credentials
        )
        if creds is None:
            creds = self._cab_credentials
        return self._is_cab_access_proven(mode, creds)

    @_has_cab_gcs_succeeded.setter
    def _has_cab_gcs_succeeded(self, val: bool) -> None:
        from google_cloud_managed_mlflow.cab_credentials import CabAccessMode

        mode = getattr(self._op_local, "access_mode", None)
        if mode is None:
            if (
                getattr(self._cab_credentials, "access_mode", None)
                == CabAccessMode.READ_WRITE
            ):
                mode = CabAccessMode.READ_WRITE
            else:
                mode = CabAccessMode.READ_ONLY
        creds = (
            self._write_cab_credentials
            if mode == CabAccessMode.READ_WRITE
            else self._cab_credentials
        )
        if creds is None:
            creds = self._cab_credentials
        if val:
            self._mark_cab_access_proven(mode, creds)
        else:
            self._proven_cab_modes.discard(mode)
            key = self._cab_access_cache_key(mode, creds)
            with _PROVEN_CAB_ACCESS_LOCK:
                _PROVEN_CAB_ACCESS_TARGETS.discard(key)

    def _attach_cab_client(
        self,
        cab_creds: CabTokenCredentials,
        mode: CabAccessMode | None = None,
    ) -> Any:
        from google_cloud_managed_mlflow.cab_credentials import CabAccessMode

        active_mode = mode if mode is not None else CabAccessMode.READ_ONLY
        self._endpoint_target = getattr(
            cab_creds, "resource_url", getattr(cab_creds, "_resource_url", "")
        )
        cab_client = gcs_storage.Client(project=None, credentials=cab_creds)
        # Disable google-cloud-storage's bucket metadata cache to prevent
        # google-cloud-storage >= 2.14 from issuing background storage.buckets.get
        # calls under prefix-scoped CAB tokens, which lack bucket-level get permissions.
        if hasattr(cab_client, "_bucket_metadata_cache"):
            try:
                cab_client._bucket_metadata_cache = None
            except AttributeError:
                pass
        self._wrap_client_http_request(cab_client, active_mode, cab_creds)
        return cab_client

    def _ensure_write_client(self) -> None:
        """Lazily create READ_WRITE CAB credentials and client."""
        with self._fallback_lock:
            if self._write_client is not None:
                return
            if self._cab_credentials is None:
                self._write_client = getattr(self, "client", None)
                return

            from google_cloud_managed_mlflow.cab_credentials import (
                CabAccessMode,
            )

            if (
                getattr(self._cab_credentials, "access_mode", None)
                == CabAccessMode.READ_WRITE
            ):
                self._write_cab_credentials = self._cab_credentials
                self._write_client = getattr(self, "client", None)
                return

            if self._write_cab_credentials is None:
                self._write_cab_credentials = self._cab_credentials.with_access_mode(
                    CabAccessMode.READ_WRITE
                )
            self._write_client = self._attach_cab_client(
                self._write_cab_credentials, mode=CabAccessMode.READ_WRITE
            )

    def _get_bucket(self, bucket: str) -> Any:
        """Return the GCS Bucket object using the appropriate client."""
        from google_cloud_managed_mlflow.cab_credentials import CabAccessMode

        mode = getattr(self._op_local, "access_mode", CabAccessMode.READ_ONLY)
        if (
            mode == CabAccessMode.READ_WRITE
            and self._write_client is not None
            and self.client is self._read_client
        ):
            return self._write_client.bucket(bucket)
        super_call: Any = super()
        return super_call._get_bucket(bucket)

    def _init_super(
        self,
        artifact_uri: str,
        client: Any = None,
        credential_refresh_def: Any = None,
        tracking_uri: str | None = None,
        registry_uri: str | None = None,
    ) -> None:
        """Call GCSArtifactRepository.__init__ with MLflow 3.x keyword arguments."""
        super().__init__(
            artifact_uri,
            client=client,
            credential_refresh_def=credential_refresh_def,
            tracking_uri=tracking_uri,
            registry_uri=registry_uri,
        )

    def _enable_write_mode(self) -> None:
        """Elevate or prepare CAB write credentials for mutation operations."""
        from google_cloud_managed_mlflow.cab_credentials import CabAccessMode

        self._op_local.access_mode = CabAccessMode.READ_WRITE
        self._ensure_write_client()

    def _execute_with_cab_fallback(
        self, fn: Callable[[], T], *, write: bool = False
    ) -> T:
        """Run a storage operation, retrying once with ADC per _fallback_reason.

        Only auto mode with active CAB credentials can fall back; afterwards the
        instance uses ADC for the rest of its lifetime.
        """
        from google_cloud_managed_mlflow.cab_credentials import CabAccessMode

        mode = CabAccessMode.READ_WRITE if write else CabAccessMode.READ_ONLY
        self._op_local.access_mode = mode

        if write and self._cab_credentials is not None:
            self._ensure_write_client()

        creds = self._write_cab_credentials if write else self._cab_credentials

        if self._auth_mode != AUTH_MODE_AUTO or self._cab_credentials is None:
            return fn()

        try:
            result = fn()
            self._mark_cab_access_proven(mode, creds)
            return result
        except Exception as exc:
            storage_proven = self._is_cab_access_proven(mode, creds)
            reason = _fallback_reason(exc, cab_storage_succeeded=storage_proven)
            if reason is None:
                raise
            if self._switch_to_adc(cause=exc):
                self._warn_fallback_once(reason, exc)
            return fn()

    def _switch_to_adc(self, *, cause: BaseException) -> bool:
        """Replace the CAB-backed client with an ADC client.

        Returns False if another thread already switched. If the ADC client
        cannot be created, raises with ``cause`` chained and leaves the CAB
        client and credentials untouched.
        """
        import google.auth

        with self._fallback_lock:
            stale_read = self._cab_credentials
            stale_write = self._write_cab_credentials
            if stale_read is None and stale_write is None:
                return False

            try:
                credentials, project = google.auth.default()
                adc_client = gcs_storage.Client(
                    project=project, credentials=credentials
                )
            except Exception as client_exc:
                raise client_exc from cause

            self.client = adc_client
            self._read_client = adc_client
            self._write_client = adc_client
            self._cab_credentials = None
            self._write_cab_credentials = None

        for stale in (stale_read, stale_write):
            if stale is not None:
                try:
                    stale.close()
                except Exception:
                    logger.debug(
                        "Could not close the replaced CAB credentials.", exc_info=True
                    )
        return True

    def _warn_fallback_once(
        self, reason: _FallbackReason, cause: BaseException
    ) -> None:
        """Warn once per (namespace endpoint, bucket) and process."""
        repo_any: Any = self
        bucket_name, _ = repo_any.parse_gcs_uri(self.artifact_uri)
        with _WARNING_LOCK:
            if (self._endpoint_target, bucket_name) in _WARNED_FALLBACK_TARGETS:
                return
            _bounded_add(_WARNED_FALLBACK_TARGETS, (self._endpoint_target, bucket_name))
        hint = (
            "" if reason in _REASONS_WITHOUT_BUCKET_HINT else _SERVICE_AGENT_BUCKET_HINT
        )
        logger.warning(
            "Google Cloud Managed Service for MLflow: using Application Default "
            "Credentials for bucket '%s' because scoped (CAB) credentials could "
            "not be used (%s: %s).%s",
            bucket_name,
            reason,
            cause,
            hint,
        )

    def log_artifact(
        self,
        local_file: str | os.PathLike[str],
        artifact_path: str | None = None,
    ) -> None:
        """Upload a local file to GCS artifact storage."""
        super_call: Any = super()
        self._execute_with_cab_fallback(
            lambda: super_call.log_artifact(local_file, artifact_path),
            write=True,
        )

    def log_artifacts(
        self,
        local_dir: str | os.PathLike[str],
        artifact_path: str | None = None,
    ) -> None:
        """Upload all files from a directory to GCS artifact storage."""
        super_call: Any = super()
        self._execute_with_cab_fallback(
            lambda: super_call.log_artifacts(local_dir, artifact_path),
            write=True,
        )

    def list_artifacts(self, path: str | None = None) -> list[Any]:
        """List artifacts at a given path within the repository."""
        super_call: Any = super()
        return self._execute_with_cab_fallback(
            lambda: super_call.list_artifacts(path),
            write=False,
        )

    def _download_file(self, remote_file_path: str, local_path: str) -> Any:
        """Download an artifact from GCS to local disk."""
        super_call: Any = super()
        return self._execute_with_cab_fallback(
            lambda: super_call._download_file(remote_file_path, local_path),
            write=False,
        )

    def delete_artifacts(self, artifact_path: str | None = None) -> None:
        """Delete artifacts under a given artifact path."""

        def _delete_impl() -> None:
            repo_any: Any = self
            (bucket_name, dest_path) = repo_any.parse_gcs_uri(self.artifact_uri)
            if artifact_path:
                dest_path = posixpath.join(dest_path, artifact_path)

            gcs_bucket = self._get_bucket(bucket_name)
            prefix = dest_path
            if self._cab_credentials is not None and dest_path:
                _, ext = posixpath.splitext(dest_path)
                leaf = posixpath.basename(dest_path)
                # Extensionless files (e.g. MLmodel, LICENSE, Dockerfile) must not
                # have a trailing slash appended, which would prevent list_blobs
                # from finding the exact blob.
                if not ext and leaf != "MLmodel" and not dest_path.endswith("/"):
                    prefix = f"{dest_path}/"

            blobs = list(gcs_bucket.list_blobs(prefix=prefix))
            if not blobs and prefix.endswith("/") and prefix.rstrip("/") == dest_path:
                # Fallback for any other extensionless single-file artifact that was
                # listed with a trailing slash.
                blobs = list(gcs_bucket.list_blobs(prefix=dest_path))

            for blob in blobs:
                if _is_object_key_within_path(blob.name, dest_path):
                    blob.delete()

        self._execute_with_cab_fallback(_delete_impl, write=True)

    def _reject_multipart_if_cab(self) -> None:
        """Make MLflow fall back to a single-part upload when CAB-backed.

        The stock multipart upload hands out signed part URLs, which CAB access
        tokens cannot sign.
        """
        if self._cab_credentials is not None:
            raise _UnsupportedMultipartUploadException()

    def create_multipart_upload(
        self,
        local_file: str,
        num_parts: int = 1,
        artifact_path: str | None = None,
    ) -> Any:
        """Create a multipart upload, unsupported when CAB credentials are used."""
        self._reject_multipart_if_cab()
        super_call: Any = super()
        return super_call.create_multipart_upload(
            local_file, num_parts=num_parts, artifact_path=artifact_path
        )

    def complete_multipart_upload(
        self,
        local_file: str,
        upload_id: str,
        parts: list[Any] | None = None,
        artifact_path: str | None = None,
    ) -> Any:
        """Complete a multipart upload, unsupported when CAB credentials are used."""
        self._reject_multipart_if_cab()
        super_call: Any = super()
        return super_call.complete_multipart_upload(
            local_file, upload_id, parts=parts, artifact_path=artifact_path
        )

    def abort_multipart_upload(
        self,
        local_file: str,
        upload_id: str,
        artifact_path: str | None = None,
    ) -> Any:
        """Abort a multipart upload, unsupported when CAB credentials are used."""
        self._reject_multipart_if_cab()
        super_call: Any = super()
        return super_call.abort_multipart_upload(
            local_file, upload_id, artifact_path=artifact_path
        )


def _ensure_gs_registered() -> None:
    """Register this class for "gs" if MLflow's registry loaded before it existed.

    When this module is the first import, its own mlflow imports load MLflow's
    artifact-repository registry before GcpGCSArtifactRepository is defined;
    the import-time filter suppresses MLflow's registration warning and keeps
    its stock GCSArtifactRepository. Re-register only if the registry still
    holds the stock class (never override another plugin) and this class is the
    installed "gs" entry point (never claim "gs" from an uninstalled source
    tree). If this module triggered the first mlflow import and the repair
    cannot run because the registry or its API is missing, log at WARNING.
    """
    try:
        registry_module = sys.modules.get(
            "mlflow.store.artifact.artifact_repository_registry"
        )
        registry = getattr(registry_module, "_artifact_repository_registry", None)
        if registry is None:
            if registry_module is not None and _IMPORTED_BEFORE_MLFLOW:
                logger.warning(
                    "Could not register GcpGCSArtifactRepository for 'gs': "
                    "MLflow's artifact repository registry was not found."
                )
            return
        current = registry.get_registered_artifact_repositories().get("gs")
        if current is not GCSArtifactRepository:
            return
        target = f"{__name__}:{GcpGCSArtifactRepository.__name__}"
        if any(
            ep.name == "gs" and ep.value.replace(" ", "") == target
            for ep in importlib.metadata.entry_points(
                group="mlflow.artifact_repository"
            )
        ):
            registry.register("gs", GcpGCSArtifactRepository)
    except Exception:
        logger.log(
            logging.WARNING if _IMPORTED_BEFORE_MLFLOW else logging.DEBUG,
            "Could not repair MLflow's 'gs' registration.",
            exc_info=True,
        )


_ensure_gs_registered()
