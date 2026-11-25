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

import enum
import logging
import os
import posixpath
import re
import sys
import threading
import urllib.parse
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Final, Literal, TypeVar

import google.api_core.exceptions
from google.cloud import storage as gcs_storage

from google_cloud_managed_mlflow._entrypoint_guard import (
    quiet_registration_warning,
    repair_artifact_entrypoint,
)

_IMPORTED_BEFORE_MLFLOW = (
    "mlflow.store.artifact.artifact_repository_registry" not in sys.modules
)


with quiet_registration_warning():
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
    from google_cloud_managed_mlflow.auth import (
        AUTH_MODE_AUTO as AUTH_MODE_AUTO,
        AUTH_MODE_CAB as AUTH_MODE_CAB,
        AUTH_MODE_ENV_VAR as AUTH_MODE_ENV_VAR,
        AUTH_MODE_EUC as AUTH_MODE_EUC,
    )
    from google_cloud_managed_mlflow.cab_credentials import (
        CabAccessMode,
        CabTokenCredentials,
        RegisteredModelGcsMatch,
    )

_AUTH_MODE_ATTRS: Final[frozenset[str]] = frozenset(
    {"AUTH_MODE_AUTO", "AUTH_MODE_CAB", "AUTH_MODE_ENV_VAR", "AUTH_MODE_EUC"}
)


class _Unset(enum.Enum):
    TOKEN = "UNSET"


_UNSET: Final = _Unset.TOKEN


def __getattr__(name: str) -> Any:
    if name in _AUTH_MODE_ATTRS:
        from google_cloud_managed_mlflow import auth

        return getattr(auth, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


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
    "unsupported_model_id_for_cab",
]

_SERVICE_AGENT_BUCKET_HINT = (
    " Server-side artifact features such as UI previews and model version "
    "creation will fail until the service agent is granted 'roles/storage.objectUser' "
    "on this bucket."
)
# The bucket-access hint only helps when the service agent is in play; it is
# noise when the token RPC is unavailable or the URI is outside the root.
_REASONS_WITHOUT_BUCKET_HINT: frozenset[_FallbackReason] = frozenset(
    {
        "token_rpc_unavailable",
        "storage_root_containment",
        "unsupported_model_id_for_cab",
    }
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


def _warn_cab_fallback_once(
    endpoint_target: str,
    bucket_name: str,
    reason: _FallbackReason,
    cause: BaseException,
) -> None:
    """Warn once per (endpoint_target, bucket_name) across the process."""
    with _WARNING_LOCK:
        if (endpoint_target, bucket_name) in _WARNED_FALLBACK_TARGETS:
            return
        _bounded_add(_WARNED_FALLBACK_TARGETS, (endpoint_target, bucket_name))
    hint = "" if reason in _REASONS_WITHOUT_BUCKET_HINT else _SERVICE_AGENT_BUCKET_HINT
    logger.warning(
        "Google Cloud Managed Service for MLflow: using Application Default "
        "Credentials for bucket '%s' because scoped (CAB) credentials could "
        "not be used (%s: %s).%s",
        bucket_name,
        reason,
        cause,
        hint,
    )


def _warn_cab_inactive_once(effective_tracking_uri: str) -> None:
    """Log a one-time warning in cab mode when CAB is inactive."""
    from google_cloud_managed_mlflow.auth import AUTH_MODE_ENV_VAR

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

    from google_cloud_managed_mlflow.cab_credentials import UnsupportedCabModelIdError

    if isinstance(exc, UnsupportedCabModelIdError):
        return "unsupported_model_id_for_cab"

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


def _extract_namespace_cab_target(
    artifact_uri: str,
) -> tuple[str | None, str | None, str | None, str | None]:
    """Extract (target_uri, run_id, model_id, artifact_path) from artifact URI."""
    from google_cloud_managed_mlflow.cab_credentials import (
        derive_cab_directory_uri,
        strip_mlmodel_leaf,
    )

    derived_uri = derive_cab_directory_uri(artifact_uri)
    parsed = urllib.parse.urlsplit(artifact_uri)
    rel = parsed.path.strip("/")
    if not artifact_uri.endswith("/"):
        rel = strip_mlmodel_leaf(rel).strip("/")
    raw_path = f"/{rel}" if rel else ""

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


def _match_registered_model_gcs_uri(
    artifact_uri: str,
    tracking_uri: str | None,
    registry_uri: str | None,
) -> RegisteredModelGcsMatch | None:
    """Match a GCS artifact URI to recorded registered model info."""
    from google_cloud_managed_mlflow.cab_credentials import (
        is_registered_model_storage_or_parent_uri,
        lookup_registered_model_for_gcs_uri,
    )
    from google_cloud_managed_mlflow.uri import (
        get_effective_tracking_uri,
        is_gcp_registry_uri,
        is_gcp_tracking_uri,
        parse_gcp_uri,
    )

    effective_tracking = get_effective_tracking_uri(tracking_uri)
    prefer_namespace = (
        registry_uri is None
        and is_gcp_tracking_uri(effective_tracking)
        and not is_registered_model_storage_or_parent_uri(artifact_uri)
    )
    if prefer_namespace:
        return None
    if registry_uri is not None and not is_gcp_registry_uri(registry_uri):
        return None
    matched = lookup_registered_model_for_gcs_uri(artifact_uri)
    if matched is not None and registry_uri is not None:
        if parse_gcp_uri(registry_uri).resource_url != matched.registry_resource_url:
            return None
    return matched


def build_registered_model_cab_credentials(
    *,
    registry_uri: str,
    model_name: str,
    version_root_uri: str,
    auth_mode: str,
) -> CabTokenCredentials | None:
    """Build CAB token credentials for a model with auto-mode fallback."""
    from google_cloud_managed_mlflow.auth import AUTH_MODE_AUTO
    from google_cloud_managed_mlflow.cab_credentials import (
        CabAccessMode,
        CabResourceKind,
        CabTokenCredentials,
        resolve_registered_model_resource_url,
    )

    try:
        canonical_url = resolve_registered_model_resource_url(
            registry_resource_url=registry_uri, model_name=model_name
        )
        return CabTokenCredentials(
            resource_url=canonical_url,
            resource_kind=CabResourceKind.REGISTERED_MODEL,
            access_mode=CabAccessMode.READ_ONLY,
            artifact_uri=version_root_uri,
            registry_resource_url=registry_uri,
            model_name=model_name,
        )
    except Exception as exc:
        if auth_mode != AUTH_MODE_AUTO:
            raise
        reason = _fallback_reason(exc, cab_storage_succeeded=False)
        if reason is None:
            raise
        bucket_name = urllib.parse.urlsplit(version_root_uri).netloc
        _warn_cab_fallback_once(registry_uri, bucket_name, reason, exc)
        return None


class GcpGCSArtifactRepository(GCSArtifactRepository):
    """GCS artifact repository for Google Cloud Managed Service for MLflow.

    Downscoped Credential Access Boundary (CAB) credentials are used for namespace
    run/model artifacts and registered models in Google Cloud Managed Service for
    MLflow. Registered model storage paths are strictly read-only and resolve to
    RegisteredModel CAB credentials based on recorded download URIs and the model
    registry resource URL.

    GOOGLE_CLOUD_MANAGED_MLFLOW_AUTH_MODE selects ``auto`` (default: CAB,
    retried once with ADC when allowable storage errors occur), ``cab``
    (never falls back to ADC once CAB is active; unresolvable paths or missing
    registry credentials fail closed with INVALID_PARAMETER_VALUE) or ``euc``
    (ADC only). An explicit ``client`` or ``credential_refresh_def`` always
    bypasses CAB.
    """

    def __init__(
        self,
        artifact_uri: str,
        client: Any = None,
        credential_refresh_def: Any = None,
        tracking_uri: str | None = None,
        registry_uri: str | None = None,
        cab_credentials: CabTokenCredentials | _Unset | None = _UNSET,
        is_registered_model: bool = False,
        *,
        _auth_mode: str | None = None,
    ) -> None:
        from google_cloud_managed_mlflow.auth import (
            AUTH_MODE_EUC,
            parse_auth_mode,
        )
        from google_cloud_managed_mlflow.cab_credentials import (
            CabAccessMode,
            CabResourceKind,
            is_registered_model_storage_or_parent_uri,
        )

        self._auth_mode = _auth_mode if _auth_mode is not None else parse_auth_mode()
        self._is_registered_model = is_registered_model

        if not self._is_registered_model and is_registered_model_storage_or_parent_uri(
            artifact_uri
        ):
            self._is_registered_model = True

        self._cab_credentials: CabTokenCredentials | None = None
        self._write_cab_credentials: CabTokenCredentials | None = None
        self._read_client: Any = None
        self._write_client: Any = None
        self._proven_cab_modes: set[CabAccessMode] = set()
        self._op_local = threading.local()
        self._fallback_lock = threading.Lock()
        self._endpoint_target = ""

        matched_model: RegisteredModelGcsMatch | _Unset | None = _UNSET
        if cab_credentials is not _UNSET:
            if (
                cab_credentials is not None
                and getattr(cab_credentials, "resource_kind", None)
                == CabResourceKind.REGISTERED_MODEL
            ):
                self._is_registered_model = True
        elif self._auth_mode != AUTH_MODE_EUC:
            matched_model = _match_registered_model_gcs_uri(
                artifact_uri, tracking_uri, registry_uri
            )
            if matched_model is not None:
                self._is_registered_model = True

        # An explicit client or refresh hook, or euc mode, keeps MLflow's own
        # credentials; otherwise CAB applies when configured.
        if (
            client is None
            and credential_refresh_def is None
            and self._auth_mode != AUTH_MODE_EUC
        ):
            resolved_cab_creds: CabTokenCredentials | None
            if cab_credentials is _UNSET:
                resolved_cab_creds = self._resolve_cab_credentials(
                    artifact_uri,
                    tracking_uri,
                    registry_uri,
                    matched_model=matched_model,
                )
            else:
                resolved_cab_creds = cab_credentials
            if resolved_cab_creds is not None:
                self._cab_credentials = resolved_cab_creds
                client = self._attach_cab_client(
                    resolved_cab_creds, mode=CabAccessMode.READ_ONLY
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

        from google_cloud_managed_mlflow.uri import is_gcp_context

        if (
            self._cab_credentials is not None
            or self._is_registered_model
            or is_gcp_context(tracking_uri=tracking_uri, registry_uri=registry_uri)
        ):
            from google_cloud_managed_mlflow.spark import install_spark_hooks

            install_spark_hooks()

    def _resolve_cab_credentials(
        self,
        artifact_uri: str,
        tracking_uri: str | None,
        registry_uri: str | None = None,
        matched_model: RegisteredModelGcsMatch | _Unset | None = _UNSET,
    ) -> CabTokenCredentials | None:
        """Resolve and construct CAB credentials for a namespace or model, or None."""
        from google_cloud_managed_mlflow.auth import (
            AUTH_MODE_CAB,
            AUTH_MODE_ENV_VAR,
        )
        from google_cloud_managed_mlflow.cab_credentials import (
            CabAccessMode,
            CabResourceKind,
            CabTokenCredentials,
            RegisteredModelGcsMatch,
            is_registered_model_storage_or_parent_uri,
        )
        from google_cloud_managed_mlflow.uri import (
            get_effective_registry_uri,
            get_effective_tracking_uri,
            is_gcp_registry_uri,
            is_gcp_tracking_uri,
            parse_gcp_uri,
        )

        matched: RegisteredModelGcsMatch | None = (
            matched_model
            if matched_model is not _UNSET
            else _match_registered_model_gcs_uri(
                artifact_uri, tracking_uri, registry_uri
            )
        )

        if isinstance(matched, RegisteredModelGcsMatch):
            return build_registered_model_cab_credentials(
                registry_uri=matched.registry_resource_url,
                model_name=matched.model_name,
                version_root_uri=matched.version_root_uri,
                auth_mode=self._auth_mode,
            )

        effective_tracking_uri = get_effective_tracking_uri(tracking_uri)
        if (
            not is_registered_model_storage_or_parent_uri(artifact_uri)
            and effective_tracking_uri
            and is_gcp_tracking_uri(effective_tracking_uri)
        ):
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

        effective_registry_uri = get_effective_registry_uri(registry_uri)
        if self._auth_mode == AUTH_MODE_CAB:
            if (
                is_gcp_registry_uri(effective_registry_uri)
                or is_gcp_registry_uri(effective_tracking_uri)
                or is_registered_model_storage_or_parent_uri(artifact_uri)
            ):
                raise MlflowException(
                    f"{AUTH_MODE_ENV_VAR}=cab: cannot resolve CAB credentials "
                    f"for artifact URI '{artifact_uri}'. The artifact URI does "
                    "not match a known namespace run/model path nor a "
                    "registered model path.",
                    error_code=INVALID_PARAMETER_VALUE,
                )
            _warn_cab_inactive_once(effective_tracking_uri)
            return None

        return None

    @classmethod
    def for_registered_model(
        cls,
        underlying_uri: str,
        *,
        model_name: str,
        tracking_uri: str | None,
        effective_tracking_uri: str,
        effective_registry_uri: str,
        auth_mode: str | None = None,
    ) -> GcpGCSArtifactRepository:
        """Construct a read-only GCS artifact repository for a registered model."""
        from google_cloud_managed_mlflow.auth import (
            AUTH_MODE_EUC,
            parse_auth_mode,
        )
        from google_cloud_managed_mlflow.cab_credentials import (
            derive_registered_model_cab_uri,
            is_registered_model_storage_or_parent_uri,
            record_registered_model_download_uri,
        )
        from google_cloud_managed_mlflow.uri import (
            is_gcp_tracking_uri,
        )

        is_model_storage_path = is_registered_model_storage_or_parent_uri(
            underlying_uri
        )
        if is_model_storage_path:
            record_registered_model_download_uri(
                underlying_uri, effective_registry_uri, model_name
            )

        active_mode = auth_mode or parse_auth_mode()
        if active_mode == AUTH_MODE_EUC:
            return cls(
                underlying_uri,
                tracking_uri=tracking_uri,
                registry_uri=effective_registry_uri,
                cab_credentials=None,
                is_registered_model=True,
                _auth_mode=active_mode,
            )

        if not is_model_storage_path and is_gcp_tracking_uri(effective_tracking_uri):
            return cls(
                underlying_uri,
                tracking_uri=effective_tracking_uri,
                registry_uri=effective_registry_uri,
                is_registered_model=True,
                _auth_mode=active_mode,
            )

        derived_cab_uri = derive_registered_model_cab_uri(underlying_uri)
        cab_creds = build_registered_model_cab_credentials(
            registry_uri=effective_registry_uri,
            model_name=model_name,
            version_root_uri=derived_cab_uri,
            auth_mode=active_mode,
        )
        return cls(
            underlying_uri,
            tracking_uri=tracking_uri,
            registry_uri=effective_registry_uri,
            cab_credentials=cab_creds,
            is_registered_model=True,
            _auth_mode=active_mode,
        )

    def _wrap_client_http_request(
        self, client: Any, mode: CabAccessMode, creds: Any
    ) -> None:
        """Wrap client._http.request to observe successful 2xx responses.

        High-level MLflow operations (such as log_artifacts or download_artifacts)
        execute multiple HTTP requests against GCS. Wrapping client._http.request
        ensures that any successful 2xx HTTP response is recorded at the transport
        layer as proven CAB storage access, preventing subsequent authorization
        errors (such as a 403 on a non-existent or restricted object) from incorrectly
        triggering fallback to ADC.
        """
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
        with self._fallback_lock:
            self._proven_cab_modes.add(mode)
        key = self._cab_access_cache_key(mode, creds)
        with _PROVEN_CAB_ACCESS_LOCK:
            _bounded_add(_PROVEN_CAB_ACCESS_TARGETS, key)

    def _is_cab_access_proven(self, mode: CabAccessMode, creds: Any) -> bool:
        with self._fallback_lock:
            if mode in self._proven_cab_modes:
                return True
        key = self._cab_access_cache_key(mode, creds)
        with _PROVEN_CAB_ACCESS_LOCK:
            return key in _PROVEN_CAB_ACCESS_TARGETS

    def _active_cab_mode_and_creds(self) -> tuple[CabAccessMode, Any]:
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
        return mode, creds

    @property
    def _has_cab_gcs_succeeded(self) -> bool:
        """Return whether CAB GCS access has been proven for the active mode."""
        mode, creds = self._active_cab_mode_and_creds()
        return self._is_cab_access_proven(mode, creds)

    @_has_cab_gcs_succeeded.setter
    def _has_cab_gcs_succeeded(self, val: bool) -> None:
        mode, creds = self._active_cab_mode_and_creds()
        if val:
            self._mark_cab_access_proven(mode, creds)
        else:
            with self._fallback_lock:
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
        self._endpoint_target = str(
            getattr(cab_creds, "registry_resource_url", None)
            or getattr(
                cab_creds, "resource_url", getattr(cab_creds, "_resource_url", "")
            )
            or ""
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
        """Enable thread-local write mode and initialize the READ_WRITE CAB client."""
        from google_cloud_managed_mlflow.cab_credentials import CabAccessMode

        self._op_local.access_mode = CabAccessMode.READ_WRITE
        self._ensure_write_client()

    def _was_closed_by_concurrent_adc_switch(
        self, creds: CabTokenCredentials | None, exc: BaseException
    ) -> bool:
        """Return True if an in-flight CAB call failed because another
        thread switched to ADC.
        """
        from google_cloud_managed_mlflow.cab_credentials import (
            ClosedCabCredentialsError,
        )

        if (
            creds is None
            or not isinstance(exc, ClosedCabCredentialsError)
            or not creds.closed
        ):
            return False
        with self._fallback_lock:
            return self._cab_credentials is None and self._write_cab_credentials is None

    def _execute_with_cab_fallback(
        self, fn: Callable[[], T], *, write: bool = False
    ) -> T:
        """Run a storage operation, retrying once with ADC per _fallback_reason.

        Only auto mode with active CAB credentials can fall back; afterwards the
        instance uses ADC for the rest of its lifetime.
        """
        from google_cloud_managed_mlflow.auth import AUTH_MODE_AUTO
        from google_cloud_managed_mlflow.cab_credentials import CabAccessMode

        mode = CabAccessMode.READ_WRITE if write else CabAccessMode.READ_ONLY
        self._op_local.access_mode = mode

        if write and self._cab_credentials is not None:
            self._enable_write_mode()

        creds = self._write_cab_credentials if write else self._cab_credentials

        if self._auth_mode != AUTH_MODE_AUTO or self._cab_credentials is None:
            return fn()

        try:
            result = fn()
            with self._fallback_lock:
                active_creds = (
                    self._write_cab_credentials if write else self._cab_credentials
                )
                still_cab = creds is not None and active_creds is creds
            if still_cab:
                self._mark_cab_access_proven(mode, creds)
            return result
        except Exception as exc:
            storage_proven = self._is_cab_access_proven(mode, creds)
            reason = _fallback_reason(exc, cab_storage_succeeded=storage_proven)
            if reason is not None:
                if self._switch_to_adc(cause=exc):
                    self._warn_fallback_once(reason, exc)
            elif not self._was_closed_by_concurrent_adc_switch(creds, exc):
                raise
            return fn()

    def _switch_to_adc(self, *, cause: BaseException) -> bool:
        """Replace the CAB-backed client with an ADC client.

        Returns False if another thread already switched. If the ADC client
        cannot be created, raises with ``cause`` chained and leaves the CAB
        client and credentials untouched.
        """
        with self._fallback_lock:
            stale_read = self._cab_credentials
            stale_write = self._write_cab_credentials
            if stale_read is None and stale_write is None:
                return False

            import google.auth

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
        _warn_cab_fallback_once(self._endpoint_target, bucket_name, reason, cause)

    def _reject_mutation_if_registered_model(self, operation: str) -> None:
        if self._is_registered_model:
            raise MlflowException(
                f"{operation} is not supported for RegisteredModel artifacts. "
                "Registered models are read-only.",
                error_code=INVALID_PARAMETER_VALUE,
            )

    def log_artifact(
        self,
        local_file: str | os.PathLike[str],
        artifact_path: str | None = None,
    ) -> None:
        """Upload a local file to GCS artifact storage."""
        self._reject_mutation_if_registered_model("log_artifact")
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
        self._reject_mutation_if_registered_model("log_artifacts")
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
        self._reject_mutation_if_registered_model("delete_artifacts")

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
        self._reject_mutation_if_registered_model("create_multipart_upload")
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
        self._reject_mutation_if_registered_model("complete_multipart_upload")
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
        self._reject_mutation_if_registered_model("abort_multipart_upload")
        self._reject_multipart_if_cab()
        super_call: Any = super()
        return super_call.abort_multipart_upload(
            local_file, upload_id, artifact_path=artifact_path
        )


def _ensure_gs_registered() -> None:
    """Register this class for 'gs' if MLflow's registry loaded before it existed."""
    repair_artifact_entrypoint(
        "gs",
        GcpGCSArtifactRepository,
        GCSArtifactRepository,
        imported_before_mlflow=_IMPORTED_BEFORE_MLFLOW,
    )


def reset_gcs_artifact_repo_state() -> None:
    """Reset cached warnings and proven CAB access state across the process."""
    with _WARNING_LOCK:
        _WARNED_FALLBACK_TARGETS.clear()
    with _WARNED_CAB_INACTIVE_LOCK:
        _WARNED_CAB_INACTIVE_URIS.clear()
    with _PROVEN_CAB_ACCESS_LOCK:
        _PROVEN_CAB_ACCESS_TARGETS.clear()


def ensure_artifact_repo_registered() -> None:
    """Ensure both 'gs' and 'models' artifact repositories are registered in MLflow."""
    with quiet_registration_warning():
        from mlflow.store.artifact.artifact_repository_registry import (
            _artifact_repository_registry,
        )

        _artifact_repository_registry.register("gs", GcpGCSArtifactRepository)

    from google_cloud_managed_mlflow.models_artifact_repo import (
        ensure_models_repo_registered,
    )

    ensure_models_repo_registered()


_ensure_gs_registered()
