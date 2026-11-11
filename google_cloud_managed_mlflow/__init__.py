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

"""Google Cloud Managed Service for MLflow plugin."""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

__version__ = "0.1.0"

if TYPE_CHECKING:
    from google_cloud_managed_mlflow.auth import (
        AUTH_PROVIDER_NAME as AUTH_PROVIDER_NAME,
        CLOUD_PLATFORM_SCOPE as CLOUD_PLATFORM_SCOPE,
        CredentialsProtocol as CredentialsProtocol,
        GoogleCloudAuthProvider as GoogleCloudAuthProvider,
        GoogleCloudBearerAuth as GoogleCloudBearerAuth,
        create_secure_session as create_secure_session,
        ensure_auth_provider_registered as ensure_auth_provider_registered,
        harden_mlflow_request_session as harden_mlflow_request_session,
        is_truthy_env as is_truthy_env,
        validate_transport_security as validate_transport_security,
    )
    from google_cloud_managed_mlflow.gcs_artifact_repo import (
        GcpGCSArtifactRepository as GcpGCSArtifactRepository,
        ensure_artifact_repo_registered as ensure_artifact_repo_registered,
    )
    from google_cloud_managed_mlflow.models_artifact_repo import (
        GcpModelsArtifactRepository as GcpModelsArtifactRepository,
    )
    from google_cloud_managed_mlflow.registry_store import (
        GcpModelRegistryRestStore as GcpModelRegistryRestStore,
        build_registry_store as build_registry_store,
        ensure_registry_store_registered as ensure_registry_store_registered,
    )
    from google_cloud_managed_mlflow.tracking_store import (
        GcpTrackingRestStore as GcpTrackingRestStore,
        build_tracking_store as build_tracking_store,
        ensure_tracking_store_registered as ensure_tracking_store_registered,
    )

__all__ = [
    "__version__",
    "GcpGCSArtifactRepository",
    "GcpModelRegistryRestStore",
    "GcpModelsArtifactRepository",
    "GcpTrackingRestStore",
    "build_registry_store",
    "build_tracking_store",
    "ensure_artifact_repo_registered",
    "ensure_registry_store_registered",
    "ensure_tracking_store_registered",
]

_LAZY_EXPORTS: dict[str, str] = {
    "GcpGCSArtifactRepository": "gcs_artifact_repo",
    "ensure_artifact_repo_registered": "gcs_artifact_repo",
    "GcpModelsArtifactRepository": "models_artifact_repo",
    "GcpModelRegistryRestStore": "registry_store",
    "build_registry_store": "registry_store",
    "ensure_registry_store_registered": "registry_store",
    "GcpTrackingRestStore": "tracking_store",
    "build_tracking_store": "tracking_store",
    "ensure_tracking_store_registered": "tracking_store",
    "AUTH_PROVIDER_NAME": "auth",
    "CLOUD_PLATFORM_SCOPE": "auth",
    "CredentialsProtocol": "auth",
    "GoogleCloudAuthProvider": "auth",
    "GoogleCloudBearerAuth": "auth",
    "create_secure_session": "auth",
    "ensure_auth_provider_registered": "auth",
    "harden_mlflow_request_session": "auth",
    "is_truthy_env": "auth",
    "validate_transport_security": "auth",
}


def __getattr__(name: str) -> Any:
    submodule = _LAZY_EXPORTS.get(name)
    if submodule is not None:
        mod = importlib.import_module(f"{__name__}.{submodule}")
        return getattr(mod, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(__all__) | set(_LAZY_EXPORTS))
