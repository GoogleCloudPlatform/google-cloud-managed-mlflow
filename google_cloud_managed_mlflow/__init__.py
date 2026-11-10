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
    from google_cloud_managed_mlflow.registry_store import (
        GcpModelRegistryRestStore,
        build_registry_store,
        ensure_registry_store_registered,
    )
    from google_cloud_managed_mlflow.tracking_store import (
        GcpTrackingRestStore as GcpTrackingRestStore,
        build_tracking_store as build_tracking_store,
    )

__all__ = [
    "__version__",
    "GcpModelRegistryRestStore",
    "GcpTrackingRestStore",
    "build_registry_store",
    "build_tracking_store",
    "ensure_registry_store_registered",
]

_REGISTRY_EXPORTS = frozenset(
    {
        "GcpModelRegistryRestStore",
        "build_registry_store",
        "ensure_registry_store_registered",
    }
)

_TRACKING_EXPORTS = frozenset(
    {
        "GcpTrackingRestStore",
        "build_tracking_store",
    }
)

_AUTH_EXPORTS = frozenset(
    {
        "AUTH_PROVIDER_NAME",
        "CLOUD_PLATFORM_SCOPE",
        "CredentialsProtocol",
        "GoogleCloudAuthProvider",
        "GoogleCloudBearerAuth",
        "create_secure_session",
        "ensure_auth_provider_registered",
        "harden_mlflow_request_session",
        "is_truthy_env",
        "validate_transport_security",
    }
)


def __getattr__(name: str) -> Any:
    if name.startswith("__"):
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    if name in _REGISTRY_EXPORTS:
        from google_cloud_managed_mlflow import registry_store as registry_store_mod

        return getattr(registry_store_mod, name)
    if name in _TRACKING_EXPORTS:
        tracking_store_mod = importlib.import_module(
            "google_cloud_managed_mlflow.tracking_store"
        )
        return getattr(tracking_store_mod, name)
    if name in _AUTH_EXPORTS:
        auth_mod = importlib.import_module("google_cloud_managed_mlflow.auth")
        return getattr(auth_mod, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
