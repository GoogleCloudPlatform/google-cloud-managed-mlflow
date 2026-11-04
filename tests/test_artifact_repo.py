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

"""Tests for GcpGCSArtifactRepository and CAB/EUC fallback behavior."""

from __future__ import annotations

import importlib.metadata
import logging
import os
import subprocess
import sys
import textwrap
import unittest.mock
from collections.abc import Callable
from pathlib import Path
from typing import Any, NoReturn

import google.api_core.exceptions
import mlflow.store.artifact.artifact_repository_registry as repo_registry_mod
import pytest
import requests
from mlflow.exceptions import MlflowException, _UnsupportedMultipartUploadException
from mlflow.protos.databricks_pb2 import (
    INVALID_PARAMETER_VALUE,
    ErrorCode,
)
from mlflow.store.artifact.gcs_artifact_repo import GCSArtifactRepository

import google_cloud_managed_mlflow.gcs_artifact_repo as repo_mod
from google_cloud_managed_mlflow.cab_credentials import (
    CabAccessMode,
    CabResourceKind,
    CabTokenCredentials,
)
from google_cloud_managed_mlflow.gcs_artifact_repo import (
    AUTH_MODE_ENV_VAR,
    GcpGCSArtifactRepository,
)
from google_cloud_managed_mlflow.uri import GcpMlflowApiError, parse_gcp_uri

_GCP_TRACKING_URI = (
    "https://mlflow.googleapis.com/v1/projects/my-proj"
    "/locations/us-central1/namespaces/my-ns"
)
_GCP_REGISTRY_URI = (
    "https://mlflow.googleapis.com/v1/projects/my-proj"
    "/locations/us-central1/registries/my-reg"
)
_ARTIFACT_URI = "gs://my-bucket/exp/run/artifacts"


@pytest.fixture(autouse=True)
def _isolate_plugin_state(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(repo_mod, "_WARNED_FALLBACK_TARGETS", set())
    monkeypatch.setattr(repo_mod, "_WARNED_CAB_INACTIVE_URIS", set())
    monkeypatch.setattr(repo_mod, "_PROVEN_CAB_ACCESS_TARGETS", set())
    monkeypatch.delenv(AUTH_MODE_ENV_VAR, raising=False)
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    monkeypatch.delenv("MLFLOW_REGISTRY_URI", raising=False)
    import mlflow.tracking._model_registry.utils as mr_utils
    import mlflow.tracking._tracking_service.utils as tr_utils

    monkeypatch.setattr(mr_utils, "_registry_uri", None, raising=True)
    monkeypatch.setattr(tr_utils, "_tracking_uri", None, raising=True)


def _make_mock_gcs_client() -> unittest.mock.MagicMock:
    client = unittest.mock.MagicMock()
    client._bucket_metadata_cache = {"stale": "entry"}
    return client


def _stub_super_init(monkeypatch: pytest.MonkeyPatch) -> unittest.mock.MagicMock:
    """Stub GCSArtifactRepository.__init__ with the installed MLflow's signature."""
    stub: unittest.mock.MagicMock = unittest.mock.create_autospec(
        GCSArtifactRepository.__init__, return_value=None
    )
    monkeypatch.setattr(GCSArtifactRepository, "__init__", stub)
    return stub


def _raise(exc: BaseException) -> Callable[[], NoReturn]:
    """Return a zero-argument storage operation that raises ``exc``."""

    def _op() -> NoReturn:
        raise exc

    return _op


def _repo_warnings(caplog: pytest.LogCaptureFixture) -> list[str]:
    """Return messages this module's logger emitted at WARNING or above."""
    return [
        r.getMessage()
        for r in caplog.records
        if r.name == repo_mod.__name__ and r.levelno >= logging.WARNING
    ]


def _parent_init_mlflow_3(
    self: Any,
    artifact_uri: str,
    client: Any = None,
    credential_refresh_def: Any = None,
    tracking_uri: str | None = None,
    registry_uri: str | None = None,
) -> None:
    pass


class TestEntryPointAndInit:
    """Tests for entry point registration, MLflow 2/3 init, and URI routing."""

    def test_pyproject_declares_gs_entry_point(self) -> None:
        pyproject_path = Path(__file__).resolve().parent.parent / "pyproject.toml"
        content = pyproject_path.read_text(encoding="utf-8")
        assert '[project.entry-points."mlflow.artifact_repository"]' in content
        assert (
            'gs = "google_cloud_managed_mlflow.gcs_artifact_repo:'
            'GcpGCSArtifactRepository"'
        ) in content

    def test_registry_resolves_gs_scheme(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import importlib.metadata

        _stub_super_init(monkeypatch)
        ep = importlib.metadata.EntryPoint(
            name="gs",
            value="google_cloud_managed_mlflow.gcs_artifact_repo:GcpGCSArtifactRepository",
            group="mlflow.artifact_repository",
        )
        reg = repo_registry_mod.ArtifactRepositoryRegistry()
        with unittest.mock.patch(
            "importlib.metadata.entry_points",
            return_value=importlib.metadata.EntryPoints([ep]),
        ):
            reg.register_entrypoints()

        monkeypatch.setattr(repo_registry_mod, "_artifact_repository_registry", reg)
        repo = repo_registry_mod.get_artifact_repository(_ARTIFACT_URI)
        assert isinstance(repo, GcpGCSArtifactRepository)
        assert repo._cab_credentials is None

    def test_init_super_passes_all_mlflow3_kwargs(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        parent = unittest.mock.create_autospec(_parent_init_mlflow_3, return_value=None)
        monkeypatch.setattr(GCSArtifactRepository, "__init__", parent)
        cab_client = _make_mock_gcs_client()
        with unittest.mock.patch(
            "google_cloud_managed_mlflow.gcs_artifact_repo.gcs_storage.Client",
            return_value=cab_client,
        ):
            GcpGCSArtifactRepository(
                _ARTIFACT_URI,
                tracking_uri=_GCP_TRACKING_URI,
                registry_uri=_GCP_REGISTRY_URI,
            )
        assert set(parent.call_args.kwargs) == {
            "client",
            "credential_refresh_def",
            "tracking_uri",
            "registry_uri",
        }
        assert parent.call_args.kwargs["client"] is cab_client
        assert parent.call_args.kwargs["tracking_uri"] == _GCP_TRACKING_URI
        assert parent.call_args.kwargs["registry_uri"] == _GCP_REGISTRY_URI

    def test_init_super_fails_closed_when_parent_rejects_client(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A parent that cannot take the CAB client must not build its own (ADC) one."""
        parent = unittest.mock.create_autospec(
            lambda self, artifact_uri: None, return_value=None
        )
        monkeypatch.setattr(GCSArtifactRepository, "__init__", parent)
        with unittest.mock.patch(
            "google_cloud_managed_mlflow.gcs_artifact_repo.gcs_storage.Client",
            return_value=_make_mock_gcs_client(),
        ):
            with pytest.raises(TypeError):
                GcpGCSArtifactRepository(_ARTIFACT_URI, tracking_uri=_GCP_TRACKING_URI)

    def test_non_gcp_uri_passthrough(self, monkeypatch: pytest.MonkeyPatch) -> None:
        super_init = _stub_super_init(monkeypatch)

        repo = GcpGCSArtifactRepository(
            _ARTIFACT_URI,
            tracking_uri="http://localhost:5000",
            registry_uri="file:///tmp/mlruns",
        )
        assert repo._cab_credentials is None
        super_init.assert_called_once()

    def test_cab_init_sets_project_none_and_disables_bucket_metadata_cache(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        super_init = _stub_super_init(monkeypatch)
        mock_client = _make_mock_gcs_client()

        with unittest.mock.patch(
            "google_cloud_managed_mlflow.gcs_artifact_repo.gcs_storage.Client",
            return_value=mock_client,
        ) as client_cls:
            repo = GcpGCSArtifactRepository(
                f"{_ARTIFACT_URI}/model/MLmodel",
                tracking_uri=_GCP_TRACKING_URI,
            )

        assert repo._cab_credentials is not None
        assert repo._cab_credentials.resource_kind == CabResourceKind.NAMESPACE
        assert repo._cab_credentials.access_mode == CabAccessMode.READ_ONLY
        assert repo._cab_credentials._artifact_uri == _ARTIFACT_URI
        client_cls.assert_called_once_with(
            project=None, credentials=repo._cab_credentials
        )
        assert mock_client._bucket_metadata_cache is None
        assert super_init.call_args.kwargs.get("client") is mock_client

    def test_registry_only_uri_does_not_initialize_cab_credentials(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        super_init = _stub_super_init(monkeypatch)

        repo = GcpGCSArtifactRepository(
            "gs://my-bucket/models/123/artifacts/MLmodel",
            tracking_uri="http://localhost:5000",
            registry_uri=_GCP_REGISTRY_URI,
        )

        assert repo._cab_credentials is None
        super_init.assert_called_once()
        assert super_init.call_args.kwargs.get("client") is None

        # Omitted tracking URI with GCP registry URI also yields stock repository
        super_init.reset_mock()
        repo2 = GcpGCSArtifactRepository(
            "gs://my-bucket/models/123/artifacts/MLmodel",
            registry_uri=_GCP_REGISTRY_URI,
        )
        assert repo2._cab_credentials is None
        super_init.assert_called_once()
        assert super_init.call_args.kwargs.get("client") is None

    @pytest.mark.parametrize(
        ("env", "kwargs"),
        [
            pytest.param(
                {}, {"registry_uri": _GCP_REGISTRY_URI}, id="registry-uri-argument"
            ),
            pytest.param(
                {"MLFLOW_REGISTRY_URI": _GCP_REGISTRY_URI},
                {},
                id="registry-uri-env-var",
            ),
            pytest.param(
                {},
                {"tracking_uri": _GCP_REGISTRY_URI},
                id="registry-uri-as-tracking-uri",
            ),
        ],
    )
    def test_cab_mode_registry_only_raises(
        self,
        monkeypatch: pytest.MonkeyPatch,
        env: dict[str, str],
        kwargs: dict[str, Any],
    ) -> None:
        monkeypatch.setenv(AUTH_MODE_ENV_VAR, "cab")
        with (
            unittest.mock.patch.dict(os.environ, env),
            pytest.raises(
                MlflowException,
                match=r"requires a Managed (?:Service for )?MLflow namespace",
            ) as exc_info,
        ):
            GcpGCSArtifactRepository(
                "gs://my-bucket/models/123/artifacts/MLmodel",
                **{"tracking_uri": "http://localhost:5000", **kwargs},
            )
        assert exc_info.value.error_code == ErrorCode.Name(INVALID_PARAMETER_VALUE)
        assert "requires a Managed" in str(exc_info.value)
        assert "namespace" in str(exc_info.value)

    def test_cab_mode_non_managed_registry_uri_does_not_raise(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(AUTH_MODE_ENV_VAR, "cab")
        super_init = _stub_super_init(monkeypatch)

        repo = GcpGCSArtifactRepository(
            "gs://my-bucket/models/123/artifacts/MLmodel",
            tracking_uri="http://localhost:5000",
            registry_uri="http://custom-registry:5000",
        )
        assert repo._cab_credentials is None
        super_init.assert_called_once()

    def test_mlflow_get_tracking_uri_consulted_when_param_and_env_unset(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
        import mlflow

        monkeypatch.setattr(mlflow, "get_tracking_uri", lambda: _GCP_TRACKING_URI)
        _stub_super_init(monkeypatch)

        repo = GcpGCSArtifactRepository(_ARTIFACT_URI)
        assert repo._cab_credentials is not None

    def test_precedence_explicit_tracking_uri_wins(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _stub_super_init(monkeypatch)
        ns_arg = (
            "https://mlflow.googleapis.com/v1/projects/p/locations/l/namespaces/arg"
        )
        ns_global = (
            "https://mlflow.googleapis.com/v1/projects/p/locations/l/namespaces/global"
        )
        ns_env = (
            "https://mlflow.googleapis.com/v1/projects/p/locations/l/namespaces/env"
        )

        import mlflow

        monkeypatch.setattr(mlflow, "get_tracking_uri", lambda: ns_global)
        monkeypatch.setenv("MLFLOW_TRACKING_URI", ns_env)

        repo = GcpGCSArtifactRepository(_ARTIFACT_URI, tracking_uri=ns_arg)
        assert repo._cab_credentials is not None
        assert repo._cab_credentials._resource_url == parse_gcp_uri(ns_arg).resource_url

    def test_precedence_global_uri_wins_over_env(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _stub_super_init(monkeypatch)
        ns_global = (
            "https://mlflow.googleapis.com/v1/projects/p/locations/l/namespaces/global"
        )
        ns_env = (
            "https://mlflow.googleapis.com/v1/projects/p/locations/l/namespaces/env"
        )

        import mlflow

        monkeypatch.setattr(mlflow, "get_tracking_uri", lambda: ns_global)
        monkeypatch.setenv("MLFLOW_TRACKING_URI", ns_env)

        repo = GcpGCSArtifactRepository(_ARTIFACT_URI)
        assert repo._cab_credentials is not None
        assert (
            repo._cab_credentials._resource_url == parse_gcp_uri(ns_global).resource_url
        )

    def test_precedence_env_var_fallback(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _stub_super_init(monkeypatch)
        ns_env = (
            "https://mlflow.googleapis.com/v1/projects/p/locations/l/namespaces/env"
        )

        import mlflow

        # Non-GCP global tracking URI takes precedence and disables CAB
        monkeypatch.setattr(mlflow, "get_tracking_uri", lambda: "file:///tmp/mlruns")
        monkeypatch.setenv("MLFLOW_TRACKING_URI", ns_env)
        repo = GcpGCSArtifactRepository(_ARTIFACT_URI)
        assert repo._cab_credentials is None

        # When global tracking URI is unset/None, MLFLOW_TRACKING_URI is used
        monkeypatch.setattr(mlflow, "get_tracking_uri", lambda: None)
        repo2 = GcpGCSArtifactRepository(_ARTIFACT_URI)
        assert repo2._cab_credentials is not None
        assert (
            repo2._cab_credentials._resource_url == parse_gcp_uri(ns_env).resource_url
        )

    def test_cab_mode_warns_once_when_built_without_managed_tracking_uri(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        _stub_super_init(monkeypatch)
        monkeypatch.setenv(AUTH_MODE_ENV_VAR, "cab")
        monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
        monkeypatch.delenv("MLFLOW_REGISTRY_URI", raising=False)
        monkeypatch.setattr(repo_mod, "_WARNED_CAB_INACTIVE_URIS", set())
        with unittest.mock.patch(
            "google_cloud_managed_mlflow.gcs_artifact_repo.gcs_storage.Client",
            return_value=_make_mock_gcs_client(),
        ):
            with caplog.at_level(logging.WARNING, logger=repo_mod.__name__):
                repo = GcpGCSArtifactRepository(
                    _ARTIFACT_URI, tracking_uri="file:///tmp/mlruns"
                )
                assert repo._cab_credentials is None
                GcpGCSArtifactRepository(
                    _ARTIFACT_URI, tracking_uri="file:///tmp/mlruns"
                )
        warnings = [
            r
            for r in caplog.records
            if r.name == repo_mod.__name__
            and r.levelno == logging.WARNING
            and "CAB is inactive" in r.getMessage()
        ]
        assert len(warnings) == 1
        assert "file:///tmp/mlruns" in warnings[0].getMessage()

    def test_cab_mode_patched_mlflow_get_registry_uri_raises(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(AUTH_MODE_ENV_VAR, "cab")
        monkeypatch.delenv("MLFLOW_REGISTRY_URI", raising=False)
        import mlflow

        monkeypatch.setattr(mlflow, "get_tracking_uri", lambda: "http://non-gcp-uri")
        monkeypatch.setattr(mlflow, "get_registry_uri", lambda: _GCP_REGISTRY_URI)
        _stub_super_init(monkeypatch)

        with pytest.raises(
            MlflowException,
            match=r"requires a Managed (?:Service for )?MLflow namespace",
        ) as exc_info:
            GcpGCSArtifactRepository(_ARTIFACT_URI, tracking_uri="http://non-gcp-uri")
        assert exc_info.value.error_code == ErrorCode.Name(INVALID_PARAMETER_VALUE)

    def test_init_super_var_keyword_signature_passes_all_kwargs(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        captured_kwargs: dict[str, Any] = {}

        def var_keyword_init(
            self: Any, artifact_uri: str, *args: Any, **kwargs: Any
        ) -> None:
            captured_kwargs.update(kwargs)
            self.artifact_uri = artifact_uri

        monkeypatch.setattr(GCSArtifactRepository, "__init__", var_keyword_init)
        mock_client = _make_mock_gcs_client()
        with unittest.mock.patch(
            "google_cloud_managed_mlflow.gcs_artifact_repo.gcs_storage.Client",
            return_value=mock_client,
        ):
            repo = GcpGCSArtifactRepository(
                _ARTIFACT_URI,
                tracking_uri=_GCP_TRACKING_URI,
                registry_uri=_GCP_REGISTRY_URI,
            )
        assert captured_kwargs["client"] is mock_client
        assert "credential_refresh_def" in captured_kwargs
        assert captured_kwargs["tracking_uri"] == _GCP_TRACKING_URI
        assert captured_kwargs["registry_uri"] == _GCP_REGISTRY_URI
        assert repo._cab_credentials is not None

    def test_effective_tracking_uri_falls_back_to_env_when_mlflow_raises(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        monkeypatch.setenv("MLFLOW_TRACKING_URI", _GCP_TRACKING_URI)
        import mlflow

        error = RuntimeError("tracking service down")
        monkeypatch.setattr(
            mlflow,
            "get_tracking_uri",
            unittest.mock.Mock(side_effect=error),
        )
        with caplog.at_level(logging.DEBUG, logger=repo_mod.__name__):
            assert repo_mod._get_effective_tracking_uri(None) == _GCP_TRACKING_URI
        debug_records = [
            r
            for r in caplog.records
            if r.name == repo_mod.__name__
            and r.levelno == logging.DEBUG
            and r.exc_info
            and r.exc_info[1] is error
        ]
        assert len(debug_records) == 1

    def test_effective_registry_uri_falls_back_to_env_when_mlflow_raises(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        monkeypatch.setenv("MLFLOW_REGISTRY_URI", _GCP_REGISTRY_URI)
        import mlflow

        error = RuntimeError("registry service down")
        monkeypatch.setattr(
            mlflow,
            "get_registry_uri",
            unittest.mock.Mock(side_effect=error),
        )
        with caplog.at_level(logging.DEBUG, logger=repo_mod.__name__):
            assert repo_mod._get_effective_registry_uri(None) == _GCP_REGISTRY_URI
        debug_records = [
            r
            for r in caplog.records
            if r.name == repo_mod.__name__
            and r.levelno == logging.DEBUG
            and r.exc_info
            and r.exc_info[1] is error
        ]
        assert len(debug_records) == 1


@pytest.mark.parametrize(
    ("exc", "cab_succeeded", "expected_reason"),
    [
        pytest.param(
            GcpMlflowApiError(
                "Service agent lacks necessary IAM permissions on bucket",
                http_status=400,
                gcp_status="FAILED_PRECONDITION",
            ),
            False,
            "service_agent_provisioning",
            id="provisioning-failed-precondition-initial",
        ),
        pytest.param(
            GcpMlflowApiError(
                "Security Token Service rejected the token for service account",
                http_status=400,
                gcp_status="FAILED_PRECONDITION",
            ),
            False,
            "service_agent_provisioning",
            id="provisioning-sts-rejected",
        ),
        pytest.param(
            GcpMlflowApiError(
                "Invalid service account configuration for namespace",
                http_status=400,
                gcp_status="FAILED_PRECONDITION",
            ),
            False,
            "service_agent_provisioning",
            id="provisioning-no-sa-configured",
        ),
        pytest.param(
            GcpMlflowApiError(
                "Failed to get GCS credentials from Security Token Service",
                http_status=400,
                gcp_status=None,
            ),
            False,
            "service_agent_provisioning",
            id="provisioning-http-400-no-gcp-status",
        ),
        pytest.param(
            GcpMlflowApiError("Not found", http_status=404),
            False,
            "token_rpc_unavailable",
            id="token-rpc-404-initial",
        ),
        pytest.param(
            GcpMlflowApiError("Not implemented", http_status=501),
            False,
            "token_rpc_unavailable",
            id="token-rpc-501-initial",
        ),
        pytest.param(
            GcpMlflowApiError(
                (
                    "Requested artifact URI 'gs://my-bucket/artifacts' "
                    "is not contained within storage root 'gs://other-root'"
                ),
                http_status=403,
                gcp_status="PERMISSION_DENIED",
            ),
            False,
            "storage_root_containment",
            id="containment-403-permission-denied",
        ),
        pytest.param(
            GcpMlflowApiError(
                (
                    "Requested artifact URI 'gs://my-bucket/artifacts' "
                    "is not contained within storage root 'gs://other-root'"
                ),
                http_status=403,
                gcp_status=None,
            ),
            False,
            "storage_root_containment",
            id="containment-403-no-gcp-status",
        ),
        pytest.param(
            google.api_core.exceptions.Forbidden("Forbidden on initial probe"),
            False,
            "initial_storage_forbidden",
            id="storage-forbidden-before-success",
        ),
        pytest.param(
            GcpMlflowApiError("Not found", http_status=404),
            True,
            "token_rpc_unavailable",
            id="token-rpc-404-after-storage-success",
        ),
        pytest.param(
            GcpMlflowApiError("Not implemented", http_status=501),
            True,
            "token_rpc_unavailable",
            id="token-rpc-501-after-storage-success",
        ),
        pytest.param(
            GcpMlflowApiError(
                (
                    "Requested artifact URI 'gs://my-bucket/artifacts' "
                    "is not contained within storage root 'gs://other-root'"
                ),
                http_status=403,
                gcp_status="PERMISSION_DENIED",
            ),
            True,
            "storage_root_containment",
            id="containment-403-after-storage-success",
        ),
        pytest.param(
            GcpMlflowApiError(
                "Service agent lacks necessary IAM permissions on bucket",
                http_status=400,
                gcp_status="FAILED_PRECONDITION",
            ),
            True,
            "service_agent_provisioning",
            id="provisioning-after-storage-success",
        ),
        pytest.param(
            GcpMlflowApiError(
                "Cannot issue READ_WRITE credentials for finalized run",
                http_status=400,
                gcp_status="FAILED_PRECONDITION",
            ),
            False,
            None,
            id="lifecycle-phrase-finalized-run",
        ),
        pytest.param(
            GcpMlflowApiError(
                "Cannot issue read-write credentials for finalized model",
                http_status=400,
                gcp_status="FAILED_PRECONDITION",
            ),
            False,
            None,
            id="lifecycle-phrase-finalized-model",
        ),
        pytest.param(
            GcpMlflowApiError(
                "Cannot issue READ_WRITE credentials: could not get credentials "
                "for service account sa@example.iam",
                http_status=400,
                gcp_status="FAILED_PRECONDITION",
            ),
            False,
            None,
            id="lifecycle-phrase-beats-provisioning-phrase",
        ),
        pytest.param(
            GcpMlflowApiError(
                "Could not get credentials for service account sa@example.iam",
                http_status=400,
                gcp_status="INVALID_ARGUMENT",
            ),
            False,
            "service_agent_provisioning",
            id="provisioning-400-invalid-argument",
        ),
        pytest.param(
            GcpMlflowApiError(
                "Could not get credentials for service account sa@example.iam",
                http_status=500,
                gcp_status="FAILED_PRECONDITION",
            ),
            False,
            "service_agent_provisioning",
            id="provisioning-500-failed-precondition",
        ),
        pytest.param(
            GcpMlflowApiError(
                "artifactPath is outside of run artifact root",
                http_status=400,
                gcp_status="INVALID_ARGUMENT",
            ),
            False,
            None,
            id="outside-artifact-root-400",
        ),
        pytest.param(
            GcpMlflowApiError(
                "Caller does not have permission",
                http_status=403,
                gcp_status="PERMISSION_DENIED",
            ),
            False,
            None,
            id="other-403-permission-denied",
        ),
        pytest.param(
            GcpMlflowApiError(
                "Could not get credentials for service account sa@example.iam",
                http_status=403,
                gcp_status="FAILED_PRECONDITION",
            ),
            False,
            "service_agent_provisioning",
            id="provisioning-403-failed-precondition",
        ),
        pytest.param(
            google.api_core.exceptions.Forbidden("Forbidden after success"),
            True,
            None,
            id="storage-forbidden-after-success",
        ),
        pytest.param(
            ValueError("Something else"), False, None, id="unrelated-exception"
        ),
        pytest.param(
            GcpMlflowApiError(
                "Failed to get GCS credentials from Security Token Service",
                http_status=500,
                gcp_status="INTERNAL",
            ),
            False,
            "service_agent_provisioning",
            id="backend-sts-internal-provisioning",
        ),
        pytest.param(
            GcpMlflowApiError(
                "Failed to generate access token for service account",
                http_status=403,
                gcp_status="PERMISSION_DENIED",
            ),
            False,
            "service_agent_provisioning",
            id="backend-permission-denied-provisioning",
        ),
        pytest.param(
            GcpMlflowApiError(
                "Artifact URI is not contained within storage root",
                http_status=400,
                gcp_status="FAILED_PRECONDITION",
            ),
            False,
            "storage_root_containment",
            id="backend-failed-precondition-containment",
        ),
        pytest.param(
            GcpMlflowApiError(
                "Run not found",
                http_status=404,
                gcp_status="NOT_FOUND",
            ),
            False,
            None,
            id="backend-not-found-no-fallback",
        ),
        pytest.param(
            GcpMlflowApiError(
                "404 Not Found",
                http_status=404,
                gcp_status=None,
            ),
            False,
            "token_rpc_unavailable",
            id="backend-404-no-status-token-rpc-unavailable",
        ),
    ],
)
def test_fallback_reason(
    exc: BaseException, cab_succeeded: bool, expected_reason: str | None
) -> None:
    assert (
        repo_mod._fallback_reason(exc, cab_storage_succeeded=cab_succeeded)
        == expected_reason
    )


class TestAuthModesAndFallback:
    """Tests for GOOGLE_CLOUD_MANAGED_MLFLOW_AUTH_MODE and fallback rules."""

    def test_invalid_auth_mode_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(AUTH_MODE_ENV_VAR, "unsupported_mode")
        with pytest.raises(MlflowException) as exc_info:
            GcpGCSArtifactRepository(_ARTIFACT_URI, tracking_uri=_GCP_TRACKING_URI)
        assert exc_info.value.error_code == ErrorCode.Name(INVALID_PARAMETER_VALUE)

    def test_euc_mode_skips_cab(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(AUTH_MODE_ENV_VAR, "EUC")
        super_init = _stub_super_init(monkeypatch)

        repo = GcpGCSArtifactRepository(_ARTIFACT_URI, tracking_uri=_GCP_TRACKING_URI)
        assert repo._cab_credentials is None
        super_init.assert_called_once()

    @pytest.mark.parametrize(
        "exc_factory",
        [
            lambda: GcpMlflowApiError(
                "Service agent lacks necessary IAM permissions on bucket",
                http_status=400,
                gcp_status="FAILED_PRECONDITION",
            ),
            lambda: GcpMlflowApiError(
                "Security Token Service rejected the token for service account",
                http_status=400,
                gcp_status="FAILED_PRECONDITION",
            ),
            lambda: GcpMlflowApiError(
                "Invalid service account configuration for namespace",
                http_status=400,
                gcp_status="FAILED_PRECONDITION",
            ),
            lambda: GcpMlflowApiError("Not implemented", http_status=501),
            lambda: GcpMlflowApiError("Not found", http_status=404),
            lambda: GcpMlflowApiError(
                "Requested artifact URI 'gs://my-bucket/artifacts' "
                "is not contained within storage root 'gs://other-root'",
                http_status=403,
                gcp_status="PERMISSION_DENIED",
            ),
            lambda: google.api_core.exceptions.Forbidden(
                "Service agent lacks bucket IAM on initial probe"
            ),
        ],
    )
    def test_auto_mode_fallback_and_deduplicated_warning(
        self,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
        exc_factory: Any,
    ) -> None:
        _stub_super_init(monkeypatch)
        cab_client = _make_mock_gcs_client()
        euc_client = _make_mock_gcs_client()

        with unittest.mock.patch(
            "google_cloud_managed_mlflow.gcs_artifact_repo.gcs_storage.Client",
            side_effect=[cab_client, euc_client, cab_client, euc_client],
        ) as client_patch:
            repo1 = GcpGCSArtifactRepository(
                _ARTIFACT_URI, tracking_uri=_GCP_TRACKING_URI
            )
            repo1.artifact_uri = _ARTIFACT_URI

            calls = 0

            def op1() -> str:
                nonlocal calls
                calls += 1
                if calls == 1:
                    raise exc_factory()
                return "ok-after-fallback"

            with caplog.at_level(logging.WARNING, logger=repo_mod.__name__):
                res1 = repo1._execute_with_cab_fallback(op1)

            assert res1 == "ok-after-fallback"
            assert calls == 2
            assert repo1._cab_credentials is None
            assert repo1.client is euc_client
            warnings = _repo_warnings(caplog)
            assert len(warnings) == 1
            assert "my-bucket" in warnings[0]

            # Second repo for same (endpoint, bucket) -> fallback without duplicate warn
            repo2 = GcpGCSArtifactRepository(
                _ARTIFACT_URI, tracking_uri=_GCP_TRACKING_URI
            )
            repo2.artifact_uri = _ARTIFACT_URI
            calls2 = 0

            def op2() -> str:
                nonlocal calls2
                calls2 += 1
                if calls2 == 1:
                    raise exc_factory()
                return "ok-2"

            with caplog.at_level(logging.WARNING, logger=repo_mod.__name__):
                res2 = repo2._execute_with_cab_fallback(op2)

            assert res2 == "ok-2"
            assert calls2 == 2
            assert repo2.client is euc_client
            assert len(_repo_warnings(caplog)) == 1
            assert client_patch.call_count == 4

    def test_auto_mode_storage_forbidden_after_success_raises(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _stub_super_init(monkeypatch)
        cab_client = _make_mock_gcs_client()

        with unittest.mock.patch(
            "google_cloud_managed_mlflow.gcs_artifact_repo.gcs_storage.Client",
            return_value=cab_client,
        ):
            repo = GcpGCSArtifactRepository(
                _ARTIFACT_URI, tracking_uri=_GCP_TRACKING_URI
            )

        # First operation succeeds -> marks CAB GCS as verified
        assert repo._execute_with_cab_fallback(lambda: "listed") == "listed"
        assert repo._has_cab_gcs_succeeded is True

        # Subsequent 403 must NOT fall back
        with pytest.raises(google.api_core.exceptions.Forbidden):
            repo._execute_with_cab_fallback(
                _raise(google.api_core.exceptions.Forbidden("Post-success 403"))
            )
        assert repo._cab_credentials is not None

    def test_read_then_write_forbidden_resets_success_and_falls_back(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Elevating to READ_WRITE resets success flag so write 403 falls back."""
        _stub_super_init(monkeypatch)
        cab_read_client = _make_mock_gcs_client()
        cab_write_client = _make_mock_gcs_client()
        euc_client = _make_mock_gcs_client()

        with unittest.mock.patch(
            "google_cloud_managed_mlflow.gcs_artifact_repo.gcs_storage.Client",
            side_effect=[cab_read_client, cab_write_client, euc_client],
        ):
            repo = GcpGCSArtifactRepository(
                _ARTIFACT_URI, tracking_uri=_GCP_TRACKING_URI
            )
            repo.artifact_uri = _ARTIFACT_URI

            # Step 1: Read operation succeeds
            assert repo._execute_with_cab_fallback(lambda: "read-ok") == "read-ok"
            assert repo._has_cab_gcs_succeeded is True
            assert repo._cab_credentials is not None
            assert repo._cab_credentials.access_mode.value == "READ_ONLY"

            # Step 2: Write elevation resets the flag
            repo._enable_write_mode()
            assert repo._has_cab_gcs_succeeded is False
            assert repo._write_cab_credentials is not None
            assert repo._write_cab_credentials.access_mode.value == "READ_WRITE"
            assert repo._cab_credentials.access_mode.value == "READ_ONLY"

            # Step 3: Write gets Forbidden on initial write call -> falls back to ADC
            calls = 0

            def write_op() -> str:
                nonlocal calls
                calls += 1
                if calls == 1:
                    raise google.api_core.exceptions.Forbidden(
                        "Service agent lacks write IAM"
                    )
                return "write-ok-adc"

            assert (
                repo._execute_with_cab_fallback(write_op, write=True) == "write-ok-adc"
            )
            assert calls == 2
            assert repo._cab_credentials is None
            assert repo._write_cab_credentials is None
            assert repo.client is euc_client

    def test_auto_mode_fallback_client_creation_failure_leaves_cab_intact(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        _stub_super_init(monkeypatch)
        cab_client = _make_mock_gcs_client()

        with unittest.mock.patch(
            "google_cloud_managed_mlflow.gcs_artifact_repo.gcs_storage.Client",
            side_effect=[cab_client, RuntimeError("ADC client init failed")],
        ):
            repo = GcpGCSArtifactRepository(
                _ARTIFACT_URI, tracking_uri=_GCP_TRACKING_URI
            )
            repo.artifact_uri = _ARTIFACT_URI
            repo.client = cab_client

            orig_exc = GcpMlflowApiError("Not found", http_status=404)
            with pytest.raises(
                RuntimeError, match="ADC client init failed"
            ) as exc_info:
                repo._execute_with_cab_fallback(_raise(orig_exc))

            assert exc_info.value.__cause__ is orig_exc
            assert repo._cab_credentials is not None
            assert repo.client is cab_client
            assert _repo_warnings(caplog) == []

    def test_write_on_already_read_write_credentials_does_not_reset_success_flag(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _stub_super_init(monkeypatch)
        mock_client = _make_mock_gcs_client()
        with unittest.mock.patch(
            "google_cloud_managed_mlflow.gcs_artifact_repo.gcs_storage.Client",
            return_value=mock_client,
        ):
            repo = GcpGCSArtifactRepository(
                _ARTIFACT_URI, tracking_uri=_GCP_TRACKING_URI
            )
        assert repo._cab_credentials is not None
        repo._cab_credentials.elevate_to_read_write()
        assert repo._cab_credentials.access_mode == "READ_WRITE"
        repo._has_cab_gcs_succeeded = True
        # Calling _enable_write_mode again when already READ_WRITE should
        # not reset the flag
        repo._enable_write_mode()
        assert repo._has_cab_gcs_succeeded is True

    @pytest.mark.parametrize(
        ("reason", "hint"),
        [
            pytest.param(
                "service_agent_provisioning",
                repo_mod._SERVICE_AGENT_BUCKET_HINT,
                id="provisioning",
            ),
            pytest.param(
                "initial_storage_forbidden",
                repo_mod._SERVICE_AGENT_BUCKET_HINT,
                id="initial-forbidden",
            ),
            pytest.param("token_rpc_unavailable", "", id="token-rpc-unavailable"),
            pytest.param("storage_root_containment", "", id="containment"),
        ],
    )
    def test_fallback_warning_content(
        self,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
        reason: repo_mod._FallbackReason,
        hint: str,
    ) -> None:
        _stub_super_init(monkeypatch)
        mock_client = _make_mock_gcs_client()
        with unittest.mock.patch(
            "google_cloud_managed_mlflow.gcs_artifact_repo.gcs_storage.Client",
            return_value=mock_client,
        ):
            repo = GcpGCSArtifactRepository(
                _ARTIFACT_URI, tracking_uri=_GCP_TRACKING_URI
            )
            repo.artifact_uri = _ARTIFACT_URI
        with caplog.at_level(logging.WARNING, logger=repo_mod.__name__):
            repo._warn_fallback_once(reason, RuntimeError("simulated error"))
        [message] = _repo_warnings(caplog)
        assert "for bucket 'my-bucket'" in message
        assert message.endswith(f"({reason}: simulated error).{hint}")

    def test_switch_to_adc_returns_false_once_already_switched(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _stub_super_init(monkeypatch)
        cab_creds = unittest.mock.MagicMock(spec=CabTokenCredentials)
        cab_client = _make_mock_gcs_client()
        adc_client = _make_mock_gcs_client()
        losing_adc_client = _make_mock_gcs_client()
        cause = GcpMlflowApiError("Not found", http_status=404)
        with unittest.mock.patch(
            "google_cloud_managed_mlflow.gcs_artifact_repo.gcs_storage.Client",
            side_effect=[cab_client, adc_client, losing_adc_client],
        ):
            repo = GcpGCSArtifactRepository(_ARTIFACT_URI, cab_credentials=cab_creds)
            assert repo._switch_to_adc(cause=cause) is True
            assert repo._switch_to_adc(cause=cause) is False
        assert repo.client is adc_client
        cab_creds.close.assert_called_once_with()

    def test_switch_to_adc_concurrent_calls_guarded_by_lock(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import threading

        _stub_super_init(monkeypatch)
        cab_creds = unittest.mock.MagicMock(spec=CabTokenCredentials)
        cab_client = _make_mock_gcs_client()
        adc_clients = [_make_mock_gcs_client() for _ in range(5)]
        with unittest.mock.patch(
            "google_cloud_managed_mlflow.gcs_artifact_repo.gcs_storage.Client",
            side_effect=[cab_client, *adc_clients],
        ):
            repo = GcpGCSArtifactRepository(_ARTIFACT_URI, cab_credentials=cab_creds)
            barrier = threading.Barrier(2)
            results: list[bool] = []

            def worker() -> None:
                barrier.wait()
                res = repo._switch_to_adc(cause=RuntimeError("test"))
                results.append(res)

            t1 = threading.Thread(target=worker)
            t2 = threading.Thread(target=worker)
            t1.start()
            t2.start()
            t1.join()
            t2.join()

            assert sorted(results) == [False, True]
            assert repo._cab_credentials is None
            assert repo.client in adc_clients
            cab_creds.close.assert_called_once_with()

    def test_fallback_warns_and_retries_when_closing_cab_credentials_fails(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        _stub_super_init(monkeypatch)
        cab_creds = unittest.mock.MagicMock(spec=CabTokenCredentials)
        cab_creds.close.side_effect = OSError("socket close failed")
        cab_client = _make_mock_gcs_client()
        adc_client = _make_mock_gcs_client()
        calls = []

        def op() -> str:
            calls.append(True)
            if len(calls) == 1:
                raise GcpMlflowApiError("Not found", http_status=404)
            return "recovered"

        with unittest.mock.patch(
            "google_cloud_managed_mlflow.gcs_artifact_repo.gcs_storage.Client",
            side_effect=[cab_client, adc_client],
        ):
            repo = GcpGCSArtifactRepository(_ARTIFACT_URI, cab_credentials=cab_creds)
            repo.artifact_uri = _ARTIFACT_URI
            repo.client = cab_client
            with caplog.at_level(logging.WARNING, logger=repo_mod.__name__):
                res = repo._execute_with_cab_fallback(op)

        assert res == "recovered"
        assert len(calls) == 2
        assert repo._cab_credentials is None
        assert repo.client is adc_client
        warnings = _repo_warnings(caplog)
        assert len(warnings) == 1
        assert "token_rpc_unavailable" in warnings[0]

    def test_fallback_installs_adc_client_before_releasing_cab_credentials(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _stub_super_init(monkeypatch)
        cab_creds = unittest.mock.MagicMock(spec=CabTokenCredentials)
        cab_client = _make_mock_gcs_client()
        adc_client = _make_mock_gcs_client()
        with unittest.mock.patch(
            "google_cloud_managed_mlflow.gcs_artifact_repo.gcs_storage.Client",
            side_effect=[cab_client, adc_client],
        ):
            repo = GcpGCSArtifactRepository(_ARTIFACT_URI, cab_credentials=cab_creds)
            repo.artifact_uri = _ARTIFACT_URI
            repo.client = cab_client
            client_at_close: list[Any] = []
            cab_creds.close.side_effect = lambda: client_at_close.append(repo.client)
            calls = 0

            def op() -> str:
                nonlocal calls
                calls += 1
                if calls == 1:
                    raise GcpMlflowApiError("Not found", http_status=404)
                return "ok"

            assert repo._execute_with_cab_fallback(op) == "ok"
        assert client_at_close == [adc_client]

    @pytest.mark.parametrize(
        "exc_factory",
        [
            lambda: GcpMlflowApiError(
                "Cannot issue READ_WRITE credentials for finalized run",
                http_status=400,
                gcp_status="FAILED_PRECONDITION",
            ),
            lambda: GcpMlflowApiError(
                "Cannot issue read-write credentials for finalized model",
                http_status=400,
                gcp_status="FAILED_PRECONDITION",
            ),
        ],
    )
    def test_auto_mode_strict_non_fallback_on_domain_errors(
        self,
        monkeypatch: pytest.MonkeyPatch,
        exc_factory: Any,
    ) -> None:
        _stub_super_init(monkeypatch)
        cab_client = _make_mock_gcs_client()

        with unittest.mock.patch(
            "google_cloud_managed_mlflow.gcs_artifact_repo.gcs_storage.Client",
            return_value=cab_client,
        ):
            repo = GcpGCSArtifactRepository(
                _ARTIFACT_URI, tracking_uri=_GCP_TRACKING_URI
            )

        with pytest.raises(GcpMlflowApiError):
            repo._execute_with_cab_fallback(_raise(exc_factory()))
        assert repo._cab_credentials is not None

    def test_cab_mode_never_falls_back(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(AUTH_MODE_ENV_VAR, "cab")
        _stub_super_init(monkeypatch)
        cab_client = _make_mock_gcs_client()

        with unittest.mock.patch(
            "google_cloud_managed_mlflow.gcs_artifact_repo.gcs_storage.Client",
            return_value=cab_client,
        ):
            repo = GcpGCSArtifactRepository(
                _ARTIFACT_URI, tracking_uri=_GCP_TRACKING_URI
            )

        with pytest.raises(google.api_core.exceptions.Forbidden):
            repo._execute_with_cab_fallback(
                _raise(google.api_core.exceptions.Forbidden("403 in cab mode"))
            )
        assert repo._cab_credentials is not None

    def test_cab_mode_containment_403_raises(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(AUTH_MODE_ENV_VAR, "cab")
        _stub_super_init(monkeypatch)
        cab_client = _make_mock_gcs_client()

        with unittest.mock.patch(
            "google_cloud_managed_mlflow.gcs_artifact_repo.gcs_storage.Client",
            return_value=cab_client,
        ):
            repo = GcpGCSArtifactRepository(
                _ARTIFACT_URI, tracking_uri=_GCP_TRACKING_URI
            )

        with pytest.raises(GcpMlflowApiError) as exc_info:
            repo._execute_with_cab_fallback(
                _raise(
                    GcpMlflowApiError(
                        "Requested artifact URI 'gs://my-bucket/artifacts' "
                        "is not contained within storage root 'gs://other-bucket'",
                        http_status=403,
                        gcp_status="PERMISSION_DENIED",
                    )
                )
            )
        assert exc_info.value.http_status == 403
        assert repo._cab_credentials is not None

    def test_auto_mode_other_403_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _stub_super_init(monkeypatch)
        cab_client = _make_mock_gcs_client()

        with unittest.mock.patch(
            "google_cloud_managed_mlflow.gcs_artifact_repo.gcs_storage.Client",
            return_value=cab_client,
        ):
            repo = GcpGCSArtifactRepository(
                _ARTIFACT_URI, tracking_uri=_GCP_TRACKING_URI
            )

        with pytest.raises(GcpMlflowApiError):
            repo._execute_with_cab_fallback(
                _raise(
                    GcpMlflowApiError(
                        "Permission denied on resource",
                        http_status=403,
                        gcp_status="PERMISSION_DENIED",
                    )
                )
            )
        assert repo._cab_credentials is not None


class TestArtifactOperations:
    """Tests for delete_artifacts, multipart guards, and Path inputs."""

    @pytest.mark.parametrize(
        "read_op",
        [
            pytest.param(lambda repo: repo.list_artifacts("sub"), id="list_artifacts"),
            pytest.param(
                lambda repo: repo._download_file("sub/a.txt", "/tmp/a.txt"),
                id="_download_file",
            ),
        ],
    )
    def test_read_operations_retry_with_adc_on_initial_storage_forbidden(
        self, monkeypatch: pytest.MonkeyPatch, read_op: Any
    ) -> None:
        _stub_super_init(monkeypatch)
        cab_client = _make_mock_gcs_client()
        adc_client = _make_mock_gcs_client()
        cab_creds = unittest.mock.MagicMock(spec=CabTokenCredentials)
        calls: list[Any] = []

        def fake_super_op(self_repo: Any, *args: Any, **kwargs: Any) -> list[str]:
            calls.append(self_repo.client)
            if self_repo.client is cab_client:
                raise google.api_core.exceptions.Forbidden("Initial bucket 403")
            return ["ok"]

        monkeypatch.setattr(GCSArtifactRepository, "list_artifacts", fake_super_op)
        monkeypatch.setattr(GCSArtifactRepository, "_download_file", fake_super_op)

        with unittest.mock.patch(
            "google_cloud_managed_mlflow.gcs_artifact_repo.gcs_storage.Client",
            side_effect=[cab_client, adc_client],
        ):
            repo = GcpGCSArtifactRepository(_ARTIFACT_URI, cab_credentials=cab_creds)
            repo.artifact_uri = _ARTIFACT_URI
            repo.client = cab_client
            result = read_op(repo)

        assert calls == [cab_client, adc_client]
        assert repo.client is adc_client
        assert repo._cab_credentials is None
        assert result == ["ok"]

    def test_log_artifacts_elevates_to_read_write_before_upload(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _stub_super_init(monkeypatch)
        cab_creds = unittest.mock.MagicMock(spec=CabTokenCredentials)
        cab_creds.access_mode = CabAccessMode.READ_ONLY
        rw_creds = unittest.mock.MagicMock(spec=CabTokenCredentials)
        rw_creds.access_mode = CabAccessMode.READ_WRITE
        cab_creds.with_access_mode.return_value = rw_creds

        cab_client = _make_mock_gcs_client()

        monkeypatch.setattr(
            GCSArtifactRepository, "log_artifacts", unittest.mock.MagicMock()
        )

        with unittest.mock.patch(
            "google_cloud_managed_mlflow.gcs_artifact_repo.gcs_storage.Client",
            return_value=cab_client,
        ):
            repo = GcpGCSArtifactRepository(_ARTIFACT_URI, cab_credentials=cab_creds)
            repo.log_artifacts("/tmp/local_dir")

        cab_creds.with_access_mode.assert_called_once_with(CabAccessMode.READ_WRITE)
        assert repo._write_cab_credentials is rw_creds
        assert repo._cab_credentials is not None
        assert repo._cab_credentials.access_mode == CabAccessMode.READ_ONLY

    def test_delete_artifacts_trailing_slash_prefix_normalization(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _stub_super_init(monkeypatch)
        cab_client = _make_mock_gcs_client()
        mock_bucket = unittest.mock.MagicMock()
        blob_match = unittest.mock.MagicMock()
        blob_match.name = "exp/run/artifacts/subdir/a.txt"
        blob_sibling = unittest.mock.MagicMock()
        blob_sibling.name = "exp/run/artifacts/subdir_other/b.txt"
        mock_bucket.list_blobs.return_value = [blob_match, blob_sibling]

        with unittest.mock.patch(
            "google_cloud_managed_mlflow.gcs_artifact_repo.gcs_storage.Client",
            return_value=cab_client,
        ):
            repo = GcpGCSArtifactRepository(
                _ARTIFACT_URI, tracking_uri=_GCP_TRACKING_URI
            )

        repo.artifact_uri = _ARTIFACT_URI
        repo_any: Any = repo
        repo_any.parse_gcs_uri = unittest.mock.MagicMock(
            return_value=("my-bucket", "exp/run/artifacts")
        )
        repo_any._get_bucket = unittest.mock.MagicMock(return_value=mock_bucket)

        repo.delete_artifacts("subdir")

        assert repo._cab_credentials is not None
        assert repo._cab_credentials.access_mode == CabAccessMode.READ_ONLY
        assert repo._write_cab_credentials is not None
        assert repo._write_cab_credentials.access_mode == CabAccessMode.READ_WRITE
        mock_bucket.list_blobs.assert_called_once_with(
            prefix="exp/run/artifacts/subdir/"
        )
        blob_match.delete.assert_called_once()
        blob_sibling.delete.assert_not_called()

    def test_delete_artifacts_file_extension_keeps_exact_prefix(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _stub_super_init(monkeypatch)
        cab_client = _make_mock_gcs_client()
        mock_bucket = unittest.mock.MagicMock()
        mock_bucket.list_blobs.return_value = []

        with unittest.mock.patch(
            "google_cloud_managed_mlflow.gcs_artifact_repo.gcs_storage.Client",
            return_value=cab_client,
        ):
            repo = GcpGCSArtifactRepository(
                _ARTIFACT_URI, tracking_uri=_GCP_TRACKING_URI
            )

        repo.artifact_uri = _ARTIFACT_URI
        repo_any: Any = repo
        repo_any.parse_gcs_uri = unittest.mock.MagicMock(
            return_value=("my-bucket", "exp/run/artifacts")
        )
        repo_any._get_bucket = unittest.mock.MagicMock(return_value=mock_bucket)

        repo.delete_artifacts("model.pkl")
        mock_bucket.list_blobs.assert_called_once_with(
            prefix="exp/run/artifacts/model.pkl"
        )

    def test_delete_artifacts_extensionless_file_mlmodel_keeps_exact_prefix(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _stub_super_init(monkeypatch)
        cab_client = _make_mock_gcs_client()
        mock_bucket = unittest.mock.MagicMock()
        blob_mlmodel = unittest.mock.MagicMock()
        blob_mlmodel.name = "exp/run/artifacts/MLmodel"
        mock_bucket.list_blobs.return_value = [blob_mlmodel]

        with unittest.mock.patch(
            "google_cloud_managed_mlflow.gcs_artifact_repo.gcs_storage.Client",
            return_value=cab_client,
        ):
            repo = GcpGCSArtifactRepository(
                _ARTIFACT_URI, tracking_uri=_GCP_TRACKING_URI
            )

        repo.artifact_uri = _ARTIFACT_URI
        repo_any: Any = repo
        repo_any.parse_gcs_uri = unittest.mock.MagicMock(
            return_value=("my-bucket", "exp/run/artifacts")
        )
        repo_any._get_bucket = unittest.mock.MagicMock(return_value=mock_bucket)

        repo.delete_artifacts("MLmodel")
        mock_bucket.list_blobs.assert_called_once_with(
            prefix="exp/run/artifacts/MLmodel"
        )
        blob_mlmodel.delete.assert_called_once()

    def test_delete_artifacts_extensionless_file_fallback_deletes_blob(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _stub_super_init(monkeypatch)
        cab_client = _make_mock_gcs_client()
        mock_bucket = unittest.mock.MagicMock()
        blob_license = unittest.mock.MagicMock()
        blob_license.name = "exp/run/artifacts/LICENSE"
        # First call with trailing slash returns empty; second call with exact
        # prefix returns the blob.
        mock_bucket.list_blobs.side_effect = [[], [blob_license]]

        with unittest.mock.patch(
            "google_cloud_managed_mlflow.gcs_artifact_repo.gcs_storage.Client",
            return_value=cab_client,
        ):
            repo = GcpGCSArtifactRepository(
                _ARTIFACT_URI, tracking_uri=_GCP_TRACKING_URI
            )

        repo.artifact_uri = _ARTIFACT_URI
        repo_any: Any = repo
        repo_any.parse_gcs_uri = unittest.mock.MagicMock(
            return_value=("my-bucket", "exp/run/artifacts")
        )
        repo_any._get_bucket = unittest.mock.MagicMock(return_value=mock_bucket)

        repo.delete_artifacts("LICENSE")
        assert mock_bucket.list_blobs.call_count == 2
        mock_bucket.list_blobs.assert_any_call(prefix="exp/run/artifacts/LICENSE/")
        mock_bucket.list_blobs.assert_any_call(prefix="exp/run/artifacts/LICENSE")
        blob_license.delete.assert_called_once()

    def test_multipart_upload_guarded_when_cab_backed(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _stub_super_init(monkeypatch)
        cab_creds = unittest.mock.MagicMock(spec=CabTokenCredentials)
        cab_client = _make_mock_gcs_client()

        with unittest.mock.patch(
            "google_cloud_managed_mlflow.gcs_artifact_repo.gcs_storage.Client",
            return_value=cab_client,
        ):
            repo = GcpGCSArtifactRepository(
                _ARTIFACT_URI,
                cab_credentials=cab_creds,
            )

        with pytest.raises(_UnsupportedMultipartUploadException):
            repo.create_multipart_upload("/tmp/file.bin", num_parts=2)

        with pytest.raises(_UnsupportedMultipartUploadException):
            repo.complete_multipart_upload("/tmp/file.bin", "upload-1", parts=[])

        with pytest.raises(_UnsupportedMultipartUploadException):
            repo.abort_multipart_upload("/tmp/file.bin", "upload-1")

    def test_pathlib_path_inputs_for_log_artifact_and_log_artifacts(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        _stub_super_init(monkeypatch)
        super_log_artifact = unittest.mock.MagicMock()
        super_log_artifacts = unittest.mock.MagicMock()
        monkeypatch.setattr(GCSArtifactRepository, "log_artifact", super_log_artifact)
        monkeypatch.setattr(GCSArtifactRepository, "log_artifacts", super_log_artifacts)

        cab_client = _make_mock_gcs_client()
        with unittest.mock.patch(
            "google_cloud_managed_mlflow.gcs_artifact_repo.gcs_storage.Client",
            return_value=cab_client,
        ):
            repo = GcpGCSArtifactRepository(
                _ARTIFACT_URI, tracking_uri=_GCP_TRACKING_URI
            )

        local_file = tmp_path / "artifact.txt"
        local_file.write_text("hello", encoding="utf-8")
        local_dir = tmp_path / "dir"
        local_dir.mkdir()

        repo.log_artifact(local_file, artifact_path="sub")
        assert repo._cab_credentials is not None
        assert repo._cab_credentials.access_mode == CabAccessMode.READ_ONLY
        assert repo._write_cab_credentials is not None
        assert repo._write_cab_credentials.access_mode == CabAccessMode.READ_WRITE
        super_log_artifact.assert_called_once_with(local_file, "sub")

        repo.log_artifacts(local_dir, artifact_path="sub_dir")
        super_log_artifacts.assert_called_once_with(local_dir, "sub_dir")

    def test_extract_namespace_cab_target_variants(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _stub_super_init(monkeypatch)
        cases: list[
            tuple[str, tuple[str | None, str | None, str | None, str | None]]
        ] = [
            (
                "gs://my-bucket/ns/runs/run-123/artifacts/sub/dir",
                (None, "run-123", None, "sub/dir"),
            ),
            (
                "gs://my-bucket/ns/0123456789abcdef0123456789abcdef/artifacts",
                (None, "0123456789abcdef0123456789abcdef", None, None),
            ),
            (
                "gs://my-bucket/ns/models/m-456/artifacts/weights",
                (None, None, "m-456", "weights"),
            ),
            (
                "gs://my-bucket/ns/models/m-456/artifacts/MLmodel",
                (None, None, "m-456", None),
            ),
            (
                "gs://my-bucket/custom/dir",
                ("gs://my-bucket/custom/dir", None, None, None),
            ),
            (
                "gs://my-bucket/ns/runs/run-1/artifacts/runs/run-2/artifacts/file",
                (None, "run-1", None, "runs/run-2/artifacts/file"),
            ),
            (
                "gs://my-bucket/exp/my-run/artifacts/sub/file",
                ("gs://my-bucket/exp/my-run/artifacts", None, None, None),
            ),
        ]
        for artifact_uri, expected_target in cases:
            assert (
                repo_mod._extract_namespace_cab_target(artifact_uri) == expected_target
            )
            with unittest.mock.patch(
                "google_cloud_managed_mlflow.gcs_artifact_repo.gcs_storage.Client"
            ):
                repo = GcpGCSArtifactRepository(
                    artifact_uri, tracking_uri=_GCP_TRACKING_URI
                )
            assert repo._cab_credentials is not None
            exp_uri, exp_run, exp_model, exp_path = expected_target
            assert repo._cab_credentials._artifact_uri == exp_uri
            assert repo._cab_credentials._run_id == exp_run
            assert repo._cab_credentials._model_id == exp_model
            assert repo._cab_credentials._artifact_path == exp_path

    def test_proven_cab_access_cached_across_repository_instances(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _stub_super_init(monkeypatch)
        with unittest.mock.patch(
            "google_cloud_managed_mlflow.gcs_artifact_repo.gcs_storage.Client"
        ):
            repo1 = GcpGCSArtifactRepository(
                _ARTIFACT_URI, tracking_uri=_GCP_TRACKING_URI
            )
            repo1.artifact_uri = _ARTIFACT_URI

            assert repo1._execute_with_cab_fallback(lambda: "read-ok") == "read-ok"
            assert repo1._has_cab_gcs_succeeded is True

            repo2 = GcpGCSArtifactRepository(
                _ARTIFACT_URI, tracking_uri=_GCP_TRACKING_URI
            )
            repo2.artifact_uri = _ARTIFACT_URI

            def forbidden_ro() -> None:
                raise google.api_core.exceptions.Forbidden("403 Forbidden RO")

            with pytest.raises(google.api_core.exceptions.Forbidden):
                repo2._execute_with_cab_fallback(forbidden_ro)
            assert repo2._cab_credentials is not None

            calls = [0]

            def write_op() -> str:
                calls[0] += 1
                if calls[0] == 1:
                    raise google.api_core.exceptions.Forbidden("403 Forbidden RW")
                return "write-ok-adc"

            assert (
                repo2._execute_with_cab_fallback(write_op, write=True) == "write-ok-adc"
            )
            assert repo2._cab_credentials is None

    def test_concurrent_read_and_write_clients_isolated(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        _stub_super_init(monkeypatch)
        mock_read_client = unittest.mock.MagicMock()
        mock_write_client = unittest.mock.MagicMock()

        with unittest.mock.patch(
            "google_cloud_managed_mlflow.gcs_artifact_repo.gcs_storage.Client",
            side_effect=[mock_read_client, mock_write_client],
        ):
            repo = GcpGCSArtifactRepository(
                _ARTIFACT_URI, tracking_uri=_GCP_TRACKING_URI
            )
            repo.artifact_uri = _ARTIFACT_URI
            repo.client = mock_read_client
            repo._GCS_UPLOAD_CHUNK_SIZE = 1024 * 1024
            repo._GCS_DEFAULT_TIMEOUT = 60

            read_bucket = unittest.mock.MagicMock()
            mock_read_client.bucket.return_value = read_bucket
            mock_blobs = unittest.mock.MagicMock()
            mock_blobs.pages = []
            read_bucket.list_blobs.return_value = mock_blobs

            write_bucket = unittest.mock.MagicMock()
            mock_write_client.bucket.return_value = write_bucket
            blob_mock = unittest.mock.MagicMock()
            write_bucket.blob.return_value = blob_mock

            # 1. list_artifacts (read operation)
            repo.list_artifacts()
            assert repo._read_client is mock_read_client
            assert read_bucket.list_blobs.called
            assert repo._cab_credentials is not None
            assert repo._cab_credentials.access_mode == CabAccessMode.READ_ONLY

            # 2. log_artifact (write operation)
            test_file = tmp_path / "sample.txt"
            test_file.write_text("content", encoding="utf-8")
            repo.log_artifact(str(test_file))
            assert repo._write_client is mock_write_client
            assert write_bucket.blob.called
            assert repo._cab_credentials.access_mode == CabAccessMode.READ_ONLY
            assert repo._write_cab_credentials is not None
            assert repo._write_cab_credentials.access_mode == CabAccessMode.READ_WRITE


@pytest.mark.parametrize(
    "first_import",
    [
        pytest.param("mlflow", id="mlflow"),
        pytest.param("google_cloud_managed_mlflow.auth", id="auth"),
        pytest.param("google_cloud_managed_mlflow.uri", id="uri"),
        pytest.param(
            "google_cloud_managed_mlflow.cab_credentials", id="cab_credentials"
        ),
        pytest.param(
            "google_cloud_managed_mlflow.gcs_artifact_repo",
            id="gcs_artifact_repo",
        ),
    ],
)
def test_gs_registration_survives_first_import_order(
    tmp_path: Path, first_import: str
) -> None:
    """Whatever is imported first, "gs" resolves to the CAB-capable repository."""
    dist_info = tmp_path / "google_cloud_managed_mlflow-0.0.0.dist-info"
    dist_info.mkdir()
    (dist_info / "METADATA").write_text(
        "Metadata-Version: 2.1\nName: google-cloud-managed-mlflow\nVersion: 0.0.0\n",
        encoding="utf-8",
    )
    (dist_info / "entry_points.txt").write_text(
        "[mlflow.artifact_repository]\n"
        "gs = google_cloud_managed_mlflow.gcs_artifact_repo:GcpGCSArtifactRepository\n",
        encoding="utf-8",
    )

    repo_root = str(Path(__file__).resolve().parent.parent)
    python_path = f"{tmp_path}{os.pathsep}{repo_root}"

    script = textwrap.dedent(
        f"""
        import {first_import}
        import mlflow
        from mlflow.store.artifact.artifact_repository_registry import (
            get_artifact_repository,
        )

        repo = get_artifact_repository("gs://my-bucket/path")
        assert type(repo).__name__ == "GcpGCSArtifactRepository", (
            f"Expected GcpGCSArtifactRepository, got {{type(repo).__name__}}"
        )
        assert getattr(repo, "_cab_credentials", None) is not None, (
            "Expected _cab_credentials to be initialized"
        )
        """
    )

    env = {
        k: v
        for k, v in os.environ.items()
        if not k.startswith("MLFLOW_")
        and not k.startswith("GOOGLE_CLOUD_MANAGED_MLFLOW_")
        and k != "PYTHONWARNINGS"
    }
    env["PYTHONPATH"] = python_path
    env["PYTHONSAFEPATH"] = "1"
    env["MLFLOW_TRACKING_URI"] = (
        "https://mlflow.googleapis.com/v1alpha/projects/p/locations/l/namespaces/n"
    )

    cmd = [sys.executable, "-W", "error::UserWarning", "-c", script]

    proc = subprocess.run(
        cmd,
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )
    assert proc.returncode == 0, (
        f"Subprocess failed (first_import={first_import}):\n"
        f"stdout: {proc.stdout}\nstderr: {proc.stderr}"
    )
    assert "Failure attempting to register artifact repository" not in proc.stderr


class _FakeArtifactRegistry:
    def __init__(self, gs: Any) -> None:
        self.repos: dict[str, Any] = {"gs": gs}

    def get_registered_artifact_repositories(self) -> dict[str, Any]:
        return self.repos

    def register(self, scheme: str, repository: Any) -> None:
        self.repos[scheme] = repository


_GS_ENTRY_POINT = importlib.metadata.EntryPoint(
    name="gs",
    value="google_cloud_managed_mlflow.gcs_artifact_repo:GcpGCSArtifactRepository",
    group="mlflow.artifact_repository",
)


@pytest.mark.parametrize(
    ("registered", "entry_points", "expected"),
    [
        pytest.param(
            GCSArtifactRepository,
            [_GS_ENTRY_POINT],
            GcpGCSArtifactRepository,
            id="replaces-stock-class",
        ),
        pytest.param(
            GCSArtifactRepository, [], GCSArtifactRepository, id="not-installed"
        ),
        pytest.param(object, [_GS_ENTRY_POINT], object, id="keeps-other-plugin"),
    ],
)
def test_ensure_gs_registered(
    monkeypatch: pytest.MonkeyPatch,
    registered: Any,
    entry_points: list[importlib.metadata.EntryPoint],
    expected: Any,
) -> None:
    registry = _FakeArtifactRegistry(registered)
    monkeypatch.setattr(repo_registry_mod, "_artifact_repository_registry", registry)
    monkeypatch.setattr(
        importlib.metadata,
        "entry_points",
        lambda **_: importlib.metadata.EntryPoints(entry_points),
    )
    repo_mod._ensure_gs_registered()
    assert registry.repos["gs"] is expected


def test_ensure_gs_registered_never_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    broken = unittest.mock.MagicMock()
    broken.get_registered_artifact_repositories.side_effect = RuntimeError("drift")
    monkeypatch.setattr(repo_registry_mod, "_artifact_repository_registry", broken)
    repo_mod._ensure_gs_registered()


def test_warning_filter_added_during_import_survives() -> None:
    """A warning filter registered during module import is not discarded."""
    script = textwrap.dedent(
        """
        import builtins
        import sys
        import warnings

        orig_import = builtins.__import__
        def custom_import(name, *args, **kwargs):
            mod = orig_import(name, *args, **kwargs)
            if name == "mlflow.exceptions":
                warnings.filterwarnings(
                    "ignore",
                    message="surviving_mlflow_filter",
                    category=UserWarning,
                )
            return mod
        builtins.__import__ = custom_import

        import google_cloud_managed_mlflow.gcs_artifact_repo
        warnings.warn("surviving_mlflow_filter", UserWarning)
        leftover = [
            f
            for f in warnings.filters
            if f[1] is not None and "register artifact repository" in f[1].pattern
        ]
        assert not leftover, (
            f"the plugin must remove its own warning filter: {leftover}"
        )
        """
    )
    repo_root = str(Path(__file__).resolve().parent.parent)
    env = {
        k: v
        for k, v in os.environ.items()
        if not k.startswith("MLFLOW_")
        and not k.startswith("GOOGLE_CLOUD_MANAGED_MLFLOW_")
        and k != "PYTHONWARNINGS"
    }
    env["PYTHONPATH"] = repo_root
    env["PYTHONSAFEPATH"] = "1"
    proc = subprocess.run(
        [sys.executable, "-W", "error::UserWarning", "-c", script],
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )
    assert proc.returncode == 0, (
        f"Subprocess failed:\nstdout: {proc.stdout}\nstderr: {proc.stderr}"
    )


class TestRealStorageClientFallback:
    """E2E tests using real storage.Client with CabTokenCredentials and mocked HTTP."""

    def test_token_rpc_failure_retried_with_adc(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Token-RPC failure (e.g. 400 service agent provisioning) falls back to ADC."""
        with unittest.mock.patch("google.auth.default", return_value=(None, "my-proj")):
            repo = GcpGCSArtifactRepository(
                _ARTIFACT_URI,
                tracking_uri=_GCP_TRACKING_URI,
            )

        assert repo._cab_credentials is not None
        session = repo._cab_credentials._session

        def fake_post_token_fail(*args: Any, **kwargs: Any) -> requests.Response:
            resp = requests.Response()
            resp.status_code = 400
            resp._content = (
                b'{"error": {"message": "The service agent is being provisioned."}}'
            )
            return resp

        monkeypatch.setattr(session, "post", fake_post_token_fail)

        calls = 0

        def storage_op() -> str:
            nonlocal calls
            calls += 1
            if repo._cab_credentials is not None:
                assert repo.client is not None
                repo.client._http.request(
                    "GET",
                    "https://storage.googleapis.com/storage/v1/b/my-bucket/o",
                )
            return "ok-adc"

        res = repo._execute_with_cab_fallback(storage_op)
        assert res == "ok-adc"
        assert repo._cab_credentials is None
        assert calls == 2
        assert any(
            "service_agent_provisioning" in record.message for record in caplog.records
        )

    def test_initial_gcs_forbidden_retried_with_adc(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Initial GCS 403 Forbidden falls back to ADC."""
        with unittest.mock.patch("google.auth.default", return_value=(None, "my-proj")):
            repo = GcpGCSArtifactRepository(
                _ARTIFACT_URI,
                tracking_uri=_GCP_TRACKING_URI,
            )

        assert repo._cab_credentials is not None
        session = repo._cab_credentials._session

        def fake_post_token_ok(*args: Any, **kwargs: Any) -> requests.Response:
            resp = requests.Response()
            resp.status_code = 200
            resp._content = (
                b'{"credentials": {"accessToken": "valid-token", '
                b'"expireTime": "2029-01-01T00:00:00Z"}}'
            )
            return resp

        monkeypatch.setattr(session, "post", fake_post_token_ok)

        orig_send = requests.Session.send

        def fake_send(
            self_session: requests.Session, req: requests.PreparedRequest, **kwargs: Any
        ) -> requests.Response:
            if req.url is not None and "storage.googleapis.com" in req.url:
                resp = requests.Response()
                resp.status_code = 403
                resp._content = b"Access Denied"
                return resp
            return orig_send(self_session, req, **kwargs)

        calls = 0

        def storage_op() -> str:
            nonlocal calls
            calls += 1
            if repo._cab_credentials is not None:
                assert repo.client is not None
                resp = repo.client._http.request(
                    "GET",
                    "https://storage.googleapis.com/storage/v1/b/my-bucket/o",
                )
                if resp.status_code == 403:
                    raise google.api_core.exceptions.Forbidden(
                        "403 Forbidden: Access Denied"
                    )
            return "ok-adc"

        monkeypatch.setattr(requests.Session, "send", fake_send)
        res = repo._execute_with_cab_fallback(storage_op)
        assert res == "ok-adc"
        assert repo._cab_credentials is None
        assert calls == 2
        assert any(
            "initial_storage_forbidden" in record.message for record in caplog.records
        )

    def test_subsequent_gcs_forbidden_surfaces_forbidden(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Subsequent GCS 403 Forbidden after proven 200 raises 403 without fallback."""
        with unittest.mock.patch("google.auth.default", return_value=(None, "my-proj")):
            repo = GcpGCSArtifactRepository(
                _ARTIFACT_URI,
                tracking_uri=_GCP_TRACKING_URI,
            )

        assert repo._cab_credentials is not None
        session = repo._cab_credentials._session

        def fake_post_token_ok(*args: Any, **kwargs: Any) -> requests.Response:
            resp = requests.Response()
            resp.status_code = 200
            resp._content = (
                b'{"credentials": {"accessToken": "valid-token", '
                b'"expireTime": "2029-01-01T00:00:00Z"}}'
            )
            return resp

        monkeypatch.setattr(session, "post", fake_post_token_ok)

        status_holder = [200]
        orig_send = requests.Session.send

        def fake_send(
            self_session: requests.Session, req: requests.PreparedRequest, **kwargs: Any
        ) -> requests.Response:
            if req.url is not None and "storage.googleapis.com" in req.url:
                resp = requests.Response()
                resp.status_code = status_holder[0]
                resp._content = b"{}" if status_holder[0] == 200 else b"Access Denied"
                return resp
            return orig_send(self_session, req, **kwargs)

        def storage_op() -> str:
            assert repo.client is not None
            resp = repo.client._http.request(
                "GET",
                "https://storage.googleapis.com/storage/v1/b/my-bucket/o",
            )
            if resp.status_code == 403:
                raise google.api_core.exceptions.Forbidden(
                    "403 Forbidden: Access Denied"
                )
            return "ok"

        monkeypatch.setattr(requests.Session, "send", fake_send)

        # Call 1: 200 OK -> proves CAB access
        assert repo._execute_with_cab_fallback(storage_op) == "ok"
        assert repo._has_cab_gcs_succeeded is True
        assert repo._cab_credentials is not None

        # Call 2: 403 Forbidden -> surfaces Forbidden exception
        status_holder[0] = 403
        with pytest.raises(google.api_core.exceptions.Forbidden):
            repo._execute_with_cab_fallback(storage_op)

        # CAB credentials remain in place; did not switch to ADC
        assert repo._cab_credentials is not None
