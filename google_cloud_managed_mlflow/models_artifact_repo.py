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

"""Models artifact repository for Google Cloud Managed Service for MLflow."""

from __future__ import annotations

import logging
import sys
import urllib.parse
from typing import TYPE_CHECKING

from google_cloud_managed_mlflow._entrypoint_guard import (
    quiet_registration_warning,
    repair_artifact_entrypoint,
)

# Import-cycle guard: MLflow loads the "models" entry point while `import mlflow`
# is still running. Keep first-party imports deferred to call time. If this module
# is itself imported first, MLflow cannot see the class yet; _ensure_models_registered()
# at the bottom of this module repairs that case.
if TYPE_CHECKING:
    from mlflow.entities.file_info import FileInfo

_IMPORTED_BEFORE_MLFLOW = (
    "mlflow.store.artifact.artifact_repository_registry" not in sys.modules
)

with quiet_registration_warning():
    from mlflow.exceptions import MlflowException
    from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE
    from mlflow.store.artifact.artifact_repo import ArtifactRepository
    from mlflow.store.artifact.models_artifact_repo import ModelsArtifactRepository

logger = logging.getLogger(__name__)


class GcpModelsArtifactRepository(ModelsArtifactRepository):
    """Models artifact repository for Google Cloud Managed Service for MLflow.

    Provides artifact download and inspection for models registered in Google Cloud
    Managed Service for MLflow under the ``models:/`` URI scheme.
    """

    model_name: str | None  # type: ignore[assignment]
    model_version: str | None
    is_logged_model_uri: bool
    repo: ArtifactRepository  # type: ignore[assignment]
    _auth_mode: str
    _is_gcp_registry: bool

    def __init__(
        self,
        artifact_uri: str,
        tracking_uri: str | None = None,
        registry_uri: str | None = None,
    ) -> None:
        """Initialize the repository for models:/ URIs."""
        from google_cloud_managed_mlflow.auth import parse_auth_mode
        from google_cloud_managed_mlflow.registry_store import (
            raise_for_namespace_as_registry_uri,
        )
        from google_cloud_managed_mlflow.uri import (
            get_effective_tracking_uri,
            is_gcp_registry_uri,
            is_gcp_tracking_uri,
            parse_gcp_uri,
            resolve_effective_registry_uri,
        )

        self._auth_mode = parse_auth_mode()
        self.is_logged_model_uri = bool(self._is_logged_model_uri(artifact_uri))
        self._is_gcp_registry = False

        if not urllib.parse.urlsplit(artifact_uri).netloc:
            effective_tracking_uri = get_effective_tracking_uri(tracking_uri)
            effective_registry_uri = resolve_effective_registry_uri(
                registry_uri, effective_tracking_uri
            )

            if self.is_logged_model_uri:
                if is_gcp_tracking_uri(effective_tracking_uri):
                    self._init_gcp_logged_model(
                        artifact_uri=artifact_uri,
                        tracking_uri=tracking_uri,
                        registry_uri=registry_uri,
                        effective_tracking_uri=effective_tracking_uri,
                    )
                    return
            else:
                if is_gcp_tracking_uri(effective_registry_uri):
                    # Enforce TLS/host policy before raising the
                    # namespace-as-registry error.
                    parse_gcp_uri(effective_registry_uri)
                    raise_for_namespace_as_registry_uri(effective_registry_uri)

                if is_gcp_registry_uri(effective_registry_uri):
                    self._init_gcp_registered_model(
                        artifact_uri=artifact_uri,
                        tracking_uri=tracking_uri,
                        effective_tracking_uri=effective_tracking_uri,
                        effective_registry_uri=effective_registry_uri,
                    )
                    return

        super().__init__(
            artifact_uri, tracking_uri=tracking_uri, registry_uri=registry_uri
        )

    def _init_gcp_logged_model(
        self,
        artifact_uri: str,
        tracking_uri: str | None,
        registry_uri: str | None,
        effective_tracking_uri: str,
    ) -> None:
        """Initialize a logged model (models:/m-<id>) on a GCP tracking server.

        Directly invokes ArtifactRepository.__init__ instead of
        ModelsArtifactRepository.__init__ to prevent default Databricks/OSS
        resolution, populating the 4 attributes (model_name, model_version,
        is_logged_model_uri, repo) consumed by inherited ModelsArtifactRepository
        methods.
        """
        from mlflow import MlflowClient
        from mlflow.store.artifact.artifact_repository_registry import (
            get_artifact_repository,
        )
        from mlflow.store.artifact.utils.models import get_model_name_and_version

        from google_cloud_managed_mlflow.cab_credentials import has_gs_scheme
        from google_cloud_managed_mlflow.tracking_store import (
            ensure_tracking_store_registered,
        )
        from google_cloud_managed_mlflow.uri import parse_gcp_uri

        # Validate tracking URI transport security before creating client
        # so MLFLOW_TRACKING_INSECURE_TLS=true fails closed immediately.
        parse_gcp_uri(effective_tracking_uri)
        ensure_tracking_store_registered()

        ArtifactRepository.__init__(
            self,
            artifact_uri,
            tracking_uri=tracking_uri,
            registry_uri=registry_uri,
        )
        self.model_name = None
        self.model_version = None
        client = MlflowClient(tracking_uri=effective_tracking_uri)
        model_id = get_model_name_and_version(client, artifact_uri)[0]
        underlying_uri = client.get_logged_model(model_id).artifact_location
        if has_gs_scheme(underlying_uri):
            from google_cloud_managed_mlflow.gcs_artifact_repo import (
                GcpGCSArtifactRepository,
            )

            self.repo = GcpGCSArtifactRepository(
                underlying_uri,
                tracking_uri=effective_tracking_uri,
                registry_uri=None,
            )
        else:
            self.repo = get_artifact_repository(
                underlying_uri,
                tracking_uri=effective_tracking_uri,
                registry_uri=None,
            )

    def _init_gcp_registered_model(
        self,
        artifact_uri: str,
        tracking_uri: str | None,
        effective_tracking_uri: str,
        effective_registry_uri: str,
    ) -> None:
        """Initialize a registered model (models:/name/version) on a GCP registry.

        Directly invokes ArtifactRepository.__init__ instead of
        ModelsArtifactRepository.__init__ to prevent default Databricks/OSS
        resolution, populating the 4 attributes (model_name, model_version,
        is_logged_model_uri, repo) consumed by inherited ModelsArtifactRepository
        methods.
        """
        from mlflow import MlflowClient
        from mlflow.store.artifact.artifact_repository_registry import (
            get_artifact_repository,
        )
        from mlflow.store.artifact.utils.models import get_model_name_and_version

        from google_cloud_managed_mlflow.cab_credentials import has_gs_scheme
        from google_cloud_managed_mlflow.registry_store import (
            ensure_registry_store_registered,
        )
        from google_cloud_managed_mlflow.uri import (
            is_gcp_tracking_uri,
            parse_gcp_uri,
        )

        parse_gcp_uri(effective_registry_uri)
        ensure_registry_store_registered()
        if is_gcp_tracking_uri(effective_tracking_uri):
            from google_cloud_managed_mlflow.tracking_store import (
                ensure_tracking_store_registered,
            )

            ensure_tracking_store_registered()

        self._is_gcp_registry = True
        ArtifactRepository.__init__(
            self,
            artifact_uri,
            tracking_uri=tracking_uri,
            registry_uri=effective_registry_uri,
        )
        client = MlflowClient(
            registry_uri=effective_registry_uri,
            tracking_uri=effective_tracking_uri or None,
        )
        name, version = get_model_name_and_version(client, artifact_uri)
        self.model_name = name
        self.model_version = version
        underlying_uri = client.get_model_version_download_uri(name, version)

        if not has_gs_scheme(underlying_uri):
            self.repo = get_artifact_repository(
                underlying_uri,
                tracking_uri=tracking_uri,
                registry_uri=effective_registry_uri,
            )
            return

        from google_cloud_managed_mlflow.gcs_artifact_repo import (
            GcpGCSArtifactRepository,
        )

        self.repo = GcpGCSArtifactRepository.for_registered_model(
            underlying_uri,
            model_name=name,
            tracking_uri=tracking_uri,
            effective_tracking_uri=effective_tracking_uri,
            effective_registry_uri=effective_registry_uri,
            auth_mode=self._auth_mode,
        )

    def _check_writable_or_raise(self, operation: str, hint: str = "") -> None:
        if self._is_gcp_registry:
            suffix = f" {hint}" if hint else ""
            raise MlflowException(
                f"{operation} is not supported for models:/ URIs. "
                f"Registered models are read-only.{suffix}",
                error_code=INVALID_PARAMETER_VALUE,
            )

    def log_artifact(self, local_file: str, artifact_path: str | None = None) -> None:
        """Log a local file as an artifact."""
        self._check_writable_or_raise("log_artifact", "Use register_model instead.")
        super().log_artifact(local_file, artifact_path)  # type: ignore[no-untyped-call]

    def log_artifacts(self, local_dir: str, artifact_path: str | None = None) -> None:
        """Log files in a local directory as artifacts."""
        self._check_writable_or_raise("log_artifacts", "Use register_model instead.")
        super().log_artifacts(local_dir, artifact_path)  # type: ignore[no-untyped-call]

    def delete_artifacts(self, artifact_path: str | None = None) -> None:
        """Delete artifacts under a given path."""
        self._check_writable_or_raise("delete_artifacts")
        super().delete_artifacts(artifact_path)  # type: ignore[no-untyped-call]

    def list_artifacts(self, path: str | None = None) -> list[FileInfo]:
        """List artifacts directly under path.

        Adds default path=None omitted on base ModelsArtifactRepository.
        """
        return self.repo.list_artifacts(path)


def _ensure_models_registered() -> None:
    repair_artifact_entrypoint(
        "models",
        GcpModelsArtifactRepository,
        ModelsArtifactRepository,
        imported_before_mlflow=_IMPORTED_BEFORE_MLFLOW,
    )


def ensure_models_repo_registered() -> None:
    """Register GcpModelsArtifactRepository for 'models'."""
    with quiet_registration_warning():
        from mlflow.store.artifact.artifact_repository_registry import (
            _artifact_repository_registry,
        )

        _artifact_repository_registry.register("models", GcpModelsArtifactRepository)


_ensure_models_registered()
