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

"""Tests for models artifact repository and registered model CAB support."""

from __future__ import annotations

import concurrent.futures
import contextlib
import json
import logging
import os
import re
import subprocess
import sys
import threading
import types
import warnings
from collections.abc import Generator
from pathlib import Path
from typing import Any, cast
from unittest import mock

import mlflow.tracking._model_registry.utils as mr_utils
import mlflow.tracking._tracking_service.utils as tr_utils
import pytest
import requests
from mlflow.exceptions import MlflowException, RestException
from mlflow.protos.databricks_pb2 import (
    INVALID_PARAMETER_VALUE,
    RESOURCE_DOES_NOT_EXIST,
    ErrorCode,
)
from mlflow.store.artifact.gcs_artifact_repo import (
    GCSArtifactRepository as StockGcsRepo,
)
from mlflow.store.artifact.models_artifact_repo import (
    ModelsArtifactRepository,
    ModelsArtifactRepository as StockRepo,
)

from google_cloud_managed_mlflow._entrypoint_guard import (
    quiet_registration_warning,
    repair_artifact_entrypoint,
)
from google_cloud_managed_mlflow.cab_credentials import (
    _DOWNLOAD_URI_MODEL_MAP,
    _MODEL_RESOURCE_NAME_CACHE,
    _MODEL_RESOURCE_NAME_CACHE_LOCK,
    CabAccessMode,
    CabResourceKind,
    CabTokenCredentials,
    UnsupportedCabModelIdError,
    _extract_model_storage_root,
    _lru_put,
    derive_registered_model_cab_uri,
    evict_cached_model_resource_url,
    invalidate_registered_model_resource_url,
    is_registered_model_storage_or_parent_uri,
    is_registered_model_storage_uri,
    lookup_registered_model_for_gcs_uri,
    record_registered_model_download_uri,
    rename_registered_model_cache_entries,
    reset_cab_credentials_cache,
    resolve_registered_model_resource_url,
)
from google_cloud_managed_mlflow.gcs_artifact_repo import (
    AUTH_MODE_AUTO,
    AUTH_MODE_CAB,
    AUTH_MODE_ENV_VAR,
    AUTH_MODE_EUC,
    GcpGCSArtifactRepository,
    _match_registered_model_gcs_uri,
    _warn_cab_fallback_once,
    ensure_artifact_repo_registered,
    reset_gcs_artifact_repo_state,
)
from google_cloud_managed_mlflow.models_artifact_repo import (
    GcpModelsArtifactRepository,
    ensure_models_repo_registered,
)
from google_cloud_managed_mlflow.registry_store import (
    _build_registry_store_class,
    _sync_model_cache_after_rpc,
)
from google_cloud_managed_mlflow.uri import GcpMlflowApiError

_GCP_TRACKING_URI = (
    "https://mlflow.googleapis.com/v1/projects/p1/locations/l1/namespaces/n1"
)
_GCP_REGISTRY_URI = (
    "https://mlflow.googleapis.com/v1/projects/p1/locations/l1/registries/r1"
)
_CANONICAL_MODEL_URL = f"{_GCP_REGISTRY_URI}/registeredModels/m1"
_MODEL_STORAGE_DL_URI = "gs://bkt/registeredModels/uid-m1/versions/1"
_EXP_2099 = "2099-01-01T00:00:00Z"


def _json_resp(payload: Any, status: int = 200) -> mock.MagicMock:
    m = mock.MagicMock(status_code=status)
    m.json.return_value = payload
    m.text = json.dumps(payload) if isinstance(payload, (dict, list)) else str(payload)
    return m


def _model_creds(**kw: Any) -> CabTokenCredentials:
    d = {
        "resource_url": _CANONICAL_MODEL_URL,
        "resource_kind": CabResourceKind.REGISTERED_MODEL,
        "access_mode": CabAccessMode.READ_ONLY,
        "artifact_uri": _MODEL_STORAGE_DL_URI,
        "registry_resource_url": _GCP_REGISTRY_URI,
        "model_name": "m1",
    }
    d.update(kw)
    return CabTokenCredentials(**d)  # type: ignore[arg-type]


@contextlib.contextmanager
def _mock_model_download(
    dl_uri: str = _MODEL_STORAGE_DL_URI,
    model_url: str = _CANONICAL_MODEL_URL,
    side_effect: Any = None,
) -> Generator[None, None, None]:
    target = (
        "google_cloud_managed_mlflow.cab_credentials."
        "resolve_registered_model_resource_url"
    )
    mv_target = "mlflow.store.artifact.utils.models.get_model_name_and_version"
    kw = {"side_effect": side_effect} if side_effect else {"return_value": model_url}
    with (
        mock.patch(target, **kw),
        mock.patch(
            "mlflow.MlflowClient.get_model_version_download_uri", return_value=dl_uri
        ),
        mock.patch(mv_target, return_value=("m1", "1")),
    ):
        yield


@pytest.fixture(autouse=True)
def _isolate_plugin_state(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> Generator[None, None, None]:
    monkeypatch.chdir(tmp_path)
    for v in (
        "MLFLOW_TRACKING_URI",
        "MLFLOW_REGISTRY_URI",
        AUTH_MODE_ENV_VAR,
        "MLFLOW_TRACKING_INSECURE_TLS",
    ):
        monkeypatch.delenv(v, raising=False)
    monkeypatch.setattr(mr_utils, "_registry_uri", None, raising=True)
    monkeypatch.setattr(tr_utils, "_tracking_uri", None, raising=True)
    reset_cab_credentials_cache()
    reset_gcs_artifact_repo_state()
    yield
    reset_cab_credentials_cache()
    reset_gcs_artifact_repo_state()


class TestEntryPointLifecycle:
    @pytest.mark.parametrize(
        "code",
        [
            pytest.param(
                "import mlflow, google_cloud_managed_mlflow",
                id="mlflow_then_plugin",
            ),
            pytest.param(
                "import mlflow, google_cloud_managed_mlflow.gcs_artifact_repo",
                id="mlflow_then_gcs_repo",
            ),
            pytest.param(
                "import google_cloud_managed_mlflow, mlflow",
                id="plugin_then_mlflow",
            ),
        ],
    )
    def test_lazy_import_does_not_eagerly_load_models_repo(self, code: str) -> None:
        full_code = (
            "import unittest.mock, importlib.metadata; "
            "unittest.mock.patch('importlib.metadata.entry_points', "
            "return_value=()).start(); "
            f"{code}; import sys; "
            "imported = "
            "'google_cloud_managed_mlflow.models_artifact_repo' in sys.modules; "
            "assert not imported, f'Eagerly imported: {imported}'"
        )
        repo_root = str(Path(__file__).resolve().parent.parent)
        res = subprocess.run(
            [sys.executable, "-c", full_code],
            capture_output=True,
            text=True,
            check=False,
            cwd=repo_root,
            env={**os.environ, "PYTHONPATH": repo_root},
        )
        assert res.returncode == 0, res.stderr

    def test_root_import_does_not_load_submodules(self) -> None:
        c = (
            "import google_cloud_managed_mlflow, sys; s = sys.modules; "
            "assert 'google_cloud_managed_mlflow.models_artifact_repo' not in s and "
            "'google_cloud_managed_mlflow.gcs_artifact_repo' not in s"
        )
        repo_root = str(Path(__file__).resolve().parent.parent)
        res = subprocess.run(
            [sys.executable, "-c", c],
            capture_output=True,
            text=True,
            check=False,
            cwd=repo_root,
            env={**os.environ, "PYTHONPATH": repo_root},
        )
        assert res.returncode == 0, res.stderr

    def test_root_package_lazy_exports(self) -> None:
        import google_cloud_managed_mlflow as pkg

        assert "ensure_tracking_store_registered" in pkg.__all__
        assert "ensure_artifact_repo_registered" in pkg.__all__
        assert all(getattr(pkg, name) is not None for name in pkg.__all__)
        assert pkg.__getattr__("build_registry_store") is not None
        assert "GcpModelsArtifactRepository" in pkg.__dir__()
        assert set(pkg.__all__).issubset(dir(pkg))
        assert "AUTH_PROVIDER_NAME" in dir(pkg)
        with pytest.raises(AttributeError, match="has no attribute 'unknown_attr'"):
            _ = pkg.unknown_attr

    def test_quiet_registration_warning(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Branch 1: quiet_registration_warning when registry already in sys.modules
        with quiet_registration_warning():
            pass

        # Branch 2: quiet_registration_warning when registry NOT in sys.modules
        monkeypatch.delitem(
            sys.modules,
            "mlflow.store.artifact.artifact_repository_registry",
            raising=False,
        )
        with warnings.catch_warnings(record=True) as recorded:
            warnings.simplefilter("always")
            with quiet_registration_warning():
                warn_msg = (
                    "Failure attempting to register artifact repository for "
                    'scheme "models"'
                )
                warnings.warn(warn_msg, UserWarning, stacklevel=2)
                warnings.warn("Preserved unrelated warning", UserWarning, stacklevel=2)
            # Registration warning is suppressed, unrelated warning is preserved
            assert len(recorded) == 1
            assert recorded[
                0
            ].message is not None and "Preserved unrelated warning" in str(
                recorded[0].message
            )

        # Branch coverage for quiet_registration_warning:
        with mock.patch("warnings.filters", []):
            with quiet_registration_warning():
                pass
        with mock.patch("warnings.filters", [("error", None, Warning, None, 0)]):
            with quiet_registration_warning():
                pass
        # Ensure filter inserted at index 0 is removed while preserving inner filters
        with warnings.catch_warnings():
            with quiet_registration_warning():
                warnings.filterwarnings("ignore", category=DeprecationWarning)
            assert any(f[2] is DeprecationWarning for f in warnings.filters)
            from google_cloud_managed_mlflow._entrypoint_guard import (
                _REGISTRATION_WARNING_PATTERN,
            )

            assert not any(
                getattr(f[1], "pattern", None) == _REGISTRATION_WARNING_PATTERN
                for f in warnings.filters
            )

        # Handle no-op filterwarnings when no matching entry is inserted
        with (
            mock.patch("warnings.filterwarnings", lambda *a, **k: None),
            mock.patch(
                "warnings.filters",
                [("ignore", re.compile("other"), UserWarning, None, 0)],
            ),
        ):
            with quiet_registration_warning():
                pass
        with warnings.catch_warnings():
            with quiet_registration_warning():
                cast("list[Any]", warnings.filters).clear()

    def test_entrypoint_repair_functions(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        fake_reg = {"models": StockRepo, "gs": StockGcsRepo}
        mock_reg = mock.MagicMock(
            get_registered_artifact_repositories=mock.MagicMock(return_value=fake_reg)
        )
        mock_reg.register.side_effect = fake_reg.__setitem__

        def _ep(n: str, v: str) -> types.SimpleNamespace:
            return types.SimpleNamespace(name=n, value=v)

        mod_name = "google_cloud_managed_mlflow.models_artifact_repo"
        cls_name = "GcpModelsArtifactRepository"
        eps = [
            _ep("models", f"{mod_name}:{cls_name}"),
            _ep(
                "gs",
                "google_cloud_managed_mlflow.gcs_artifact_repo:GcpGCSArtifactRepository",
            ),
        ]
        target = (
            "mlflow.store.artifact.artifact_repository_registry."
            "_artifact_repository_registry"
        )
        with (
            mock.patch(target, mock_reg),
            mock.patch("importlib.metadata.entry_points", return_value=eps),
        ):
            repair_artifact_entrypoint(
                "models",
                GcpModelsArtifactRepository,
                StockRepo,
                imported_before_mlflow=False,
            )
            assert fake_reg["models"] is GcpModelsArtifactRepository
            # Already repaired -> early return
            repair_artifact_entrypoint(
                "models",
                GcpModelsArtifactRepository,
                StockRepo,
                imported_before_mlflow=False,
            )

        fake_reg["models"] = StockRepo
        with (
            mock.patch(target, mock_reg),
            mock.patch(
                "importlib.metadata.entry_points",
                return_value=[_ep("unrelated", "m:M")],
            ),
        ):
            repair_artifact_entrypoint(
                "models",
                GcpModelsArtifactRepository,
                StockRepo,
                imported_before_mlflow=False,
            )
            assert fake_reg["models"] is StockRepo

        # ensure_*_registered calls register directly into _artifact_repository_registry
        with (
            mock.patch(target, mock_reg),
            mock.patch("importlib.metadata.entry_points", return_value=[]),
        ):
            ensure_artifact_repo_registered()
            ensure_models_repo_registered()
            assert fake_reg["gs"] is GcpGCSArtifactRepository
            assert fake_reg["models"] is GcpModelsArtifactRepository

        # None registry: warns only when module is loaded and
        # imported_before_mlflow=True
        reg_mod = "mlflow.store.artifact.artifact_repository_registry"
        stub_mod = types.SimpleNamespace(_artifact_repository_registry=None)
        with mock.patch.dict(sys.modules, {reg_mod: stub_mod}):
            caplog.clear()
            repair_artifact_entrypoint(
                "models",
                GcpModelsArtifactRepository,
                StockRepo,
                imported_before_mlflow=True,
            )
            assert any(
                r.levelno == logging.WARNING
                and "MLflow's artifact repository registry was not found" in r.message
                for r in caplog.records
            )

            caplog.clear()
            repair_artifact_entrypoint(
                "models",
                GcpModelsArtifactRepository,
                StockRepo,
                imported_before_mlflow=False,
            )
            assert caplog.records == []

        caplog.clear()
        with mock.patch.dict(sys.modules, {reg_mod: None}):
            repair_artifact_entrypoint(
                "models",
                GcpModelsArtifactRepository,
                StockRepo,
                imported_before_mlflow=True,
            )
        assert caplog.records == []

        caplog.clear()
        caplog.set_level(logging.DEBUG)
        err_reg = mock.MagicMock(
            get_registered_artifact_repositories=mock.MagicMock(
                side_effect=RuntimeError("err")
            )
        )
        for b in (True, False):
            with mock.patch(target, err_reg):
                repair_artifact_entrypoint(
                    "models",
                    GcpModelsArtifactRepository,
                    StockRepo,
                    imported_before_mlflow=b,
                )
        assert any(
            r.levelno == logging.WARNING
            and "Could not repair MLflow's 'models' registration" in r.message
            for r in caplog.records
        )
        assert any(
            r.levelno == logging.DEBUG
            and "Could not repair MLflow's 'models' registration" in r.message
            for r in caplog.records
        )

    def test_ensure_models_repo_registered_cold_no_warnings(self) -> None:
        repo_root = str(Path(__file__).resolve().parents[1])
        cmd = [
            sys.executable,
            "-W",
            "error::UserWarning",
            "-c",
            "import sys\n"
            f"sys.path.insert(0, {repo_root!r})\n"
            "from google_cloud_managed_mlflow.models_artifact_repo "
            "import ensure_models_repo_registered\n"
            "ensure_models_repo_registered()\n",
        ]
        res = subprocess.run(cmd, capture_output=True, text=True, check=False)
        assert res.returncode == 0, f"Failed with: {res.stderr}"


class TestModelsArtifactRepository:
    @pytest.mark.parametrize(
        (
            "uri",
            "dl_uri",
            "auth_mode",
            "use_trk",
            "expected_repo_type",
            "is_euc_mode",
        ),
        [
            pytest.param(
                "models:/m1/2",
                "gs://bkt/registeredModels/uid-m1/versions/2",
                AUTH_MODE_AUTO,
                False,
                GcpGCSArtifactRepository,
                False,
                id="auto_version_path",
            ),
            pytest.param(
                "models:/m1/Prod",
                _MODEL_STORAGE_DL_URI,
                AUTH_MODE_AUTO,
                False,
                GcpGCSArtifactRepository,
                False,
                id="auto_stage_path",
            ),
            pytest.param(
                "models:/m1@champ",
                _MODEL_STORAGE_DL_URI,
                AUTH_MODE_AUTO,
                False,
                GcpGCSArtifactRepository,
                False,
                id="auto_alias_path",
            ),
            pytest.param(
                "models:/m1/1",
                "s3://bkt/m1/v1",
                AUTH_MODE_AUTO,
                False,
                None,
                False,
                id="s3_scheme_non_gcs",
            ),
            pytest.param(
                "models:/m1/1",
                _MODEL_STORAGE_DL_URI,
                AUTH_MODE_EUC,
                False,
                None,
                True,
                id="euc_mode_no_cab_creds",
            ),
            pytest.param(
                "models:/m1/1",
                _MODEL_STORAGE_DL_URI,
                AUTH_MODE_AUTO,
                True,
                GcpGCSArtifactRepository,
                False,
                id="auto_tracking_as_registry",
            ),
        ],
    )
    def test_uri_resolutions(
        self,
        uri: str,
        dl_uri: str,
        auth_mode: str,
        use_trk: bool,
        expected_repo_type: type[Any] | None,
        is_euc_mode: bool,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv(AUTH_MODE_ENV_VAR, auth_mode)
        reg_uri = None if use_trk else _GCP_REGISTRY_URI
        trk_uri = _GCP_REGISTRY_URI if use_trk else None
        with _mock_model_download(dl_uri=dl_uri):
            repo = GcpModelsArtifactRepository(
                uri, tracking_uri=trk_uri, registry_uri=reg_uri
            )
            if is_euc_mode:
                assert getattr(repo.repo, "_cab_credentials", None) is None
            elif expected_repo_type is not None:
                assert isinstance(repo.repo, expected_repo_type)
            else:
                assert not isinstance(repo.repo, GcpGCSArtifactRepository)

    @pytest.mark.parametrize(
        (
            "auth_mode",
            "dl_uri",
            "insecure_tls",
            "api_error",
            "expected_exc",
            "match_pattern",
        ),
        [
            pytest.param(
                "invalid_mode",
                _MODEL_STORAGE_DL_URI,
                False,
                None,
                MlflowException,
                "Invalid .*AUTH_MODE",
                id="invalid_auth_mode",
            ),
            pytest.param(
                AUTH_MODE_CAB,
                _MODEL_STORAGE_DL_URI,
                False,
                GcpMlflowApiError("CAB failure", http_status=500),
                MlflowException,
                "CAB failure",
                id="cab_mode_500_fails_closed",
            ),
            pytest.param(
                AUTH_MODE_AUTO,
                _MODEL_STORAGE_DL_URI,
                True,
                None,
                MlflowException,
                "Disabling TLS",
                id="insecure_tls_fails_closed",
            ),
            pytest.param(
                AUTH_MODE_AUTO,
                "gs://bkt/registeredModels/uid-m1/versions/1/../1",
                False,
                None,
                MlflowException,
                "path traversal",
                id="path_traversal_fails_closed",
            ),
            pytest.param(
                AUTH_MODE_AUTO,
                "gs://bkt/registeredModels/uid-m1/versions/1/%2e%2e/1",
                False,
                None,
                MlflowException,
                "percent-encoded",
                id="encoded_traversal_fails_closed",
            ),
            pytest.param(
                AUTH_MODE_AUTO,
                _MODEL_STORAGE_DL_URI,
                False,
                GcpMlflowApiError("Not found", http_status=404),
                None,
                None,
                id="auto_mode_upstream_404_falls_back",
            ),
            pytest.param(
                AUTH_MODE_AUTO,
                _MODEL_STORAGE_DL_URI,
                False,
                GcpMlflowApiError(
                    "Model not found", http_status=404, gcp_status="NOT_FOUND"
                ),
                GcpMlflowApiError,
                "Model not found",
                id="auto_mode_model_not_found_fails_closed",
            ),
            pytest.param(
                AUTH_MODE_AUTO,
                _MODEL_STORAGE_DL_URI,
                False,
                GcpMlflowApiError("not contained within storage root", http_status=403),
                None,
                None,
                id="auto_mode_containment_403_falls_back",
            ),
        ],
    )
    def test_fail_closed_and_fallback_matrix(
        self,
        auth_mode: str,
        dl_uri: str,
        insecure_tls: bool,
        api_error: GcpMlflowApiError | None,
        expected_exc: type[Exception] | None,
        match_pattern: str | None,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv(AUTH_MODE_ENV_VAR, auth_mode)
        if insecure_tls:
            monkeypatch.setenv("MLFLOW_TRACKING_INSECURE_TLS", "true")

        def _resolve_url(*_a: Any, **_k: Any) -> str:
            if api_error is not None:
                raise api_error
            return _CANONICAL_MODEL_URL

        with _mock_model_download(dl_uri=dl_uri, side_effect=_resolve_url):
            if expected_exc is not None:
                with pytest.raises(expected_exc, match=match_pattern or ""):
                    GcpModelsArtifactRepository(
                        "models:/m1/1", registry_uri=_GCP_REGISTRY_URI
                    )
            else:
                repo = GcpModelsArtifactRepository(
                    "models:/m1/1", registry_uri=_GCP_REGISTRY_URI
                )
                assert isinstance(repo.repo, GcpGCSArtifactRepository)
                assert repo.repo._cab_credentials is None

    def test_e2e_issue_token_fallback_and_download_artifacts(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(AUTH_MODE_ENV_VAR, AUTH_MODE_AUTO)
        m_name = "projects/p1/locations/l1/registries/r1/registeredModels/uid-m1"
        search_resp = _json_resp(
            {"registeredModels": [{"name": m_name, "displayName": "m1"}]}
        )
        token_err = _json_resp(
            {"error": {"message": "not contained within storage root"}}, 403
        )

        mock_blob = mock.MagicMock()
        mock_blob.name = "registeredModels/uid-m1/versions/1/MLmodel"
        mock_blob.size = 10
        mock_blob.updated = 1234567890

        def _download_to_filename(path: Any, **_k: Any) -> None:
            Path(path).write_bytes(b"artifact_path: model\n")

        mock_blob.download_to_filename.side_effect = _download_to_filename

        mock_bucket = mock.MagicMock()
        mock_results = mock.MagicMock()
        mock_results.__iter__.return_value = [mock_blob]
        mock_results.pages = [mock.MagicMock(prefixes=[])]
        mock_bucket.list_blobs.return_value = mock_results
        mock_bucket.get_blob.return_value = mock_blob
        mock_bucket.blob.return_value = mock_blob

        mock_adc_client = mock.MagicMock()
        mock_adc_client.bucket.return_value = mock_bucket

        def _client_factory(*args: Any, credentials: Any = None, **kwargs: Any) -> Any:
            if isinstance(credentials, CabTokenCredentials):
                cab_client = mock.MagicMock()
                cab_bucket = mock.MagicMock()

                def _trigger_cab_refresh(*_a: Any, **_k: Any) -> Any:
                    credentials.refresh(mock.MagicMock())

                cab_bucket.list_blobs.side_effect = _trigger_cab_refresh
                cab_bucket.get_blob.side_effect = _trigger_cab_refresh
                cab_bucket.blob.side_effect = _trigger_cab_refresh
                cab_client.bucket.return_value = cab_bucket
                return cab_client
            return mock_adc_client

        with (
            mock.patch("requests.Session.get", return_value=search_resp),
            mock.patch("requests.Session.post", return_value=token_err) as mock_post,
            mock.patch("google.cloud.storage.Client", side_effect=_client_factory),
            mock.patch(
                "mlflow.MlflowClient.get_model_version_download_uri",
                return_value=_MODEL_STORAGE_DL_URI,
            ),
        ):
            repo = GcpModelsArtifactRepository(
                "models:/m1/1", registry_uri=_GCP_REGISTRY_URI
            )
            # 1. list_artifacts falls back to ADC
            artifacts = repo.list_artifacts()
            assert len(artifacts) == 1
            assert artifacts[0].path == "MLmodel"
            assert mock_post.called
            inner_repo: Any = repo.repo
            assert inner_repo._cab_credentials is None
            assert inner_repo._is_registered_model is True

            # 2. download_artifacts writes MLmodel and registered_model_meta on
            # ADC fallback
            dest = tmp_path / "model_dl"
            dest.mkdir(exist_ok=True)
            dl_fn: Any = repo.download_artifacts
            result_dir = dl_fn("", dst_path=str(dest))
            mlmodel_path = Path(result_dir) / "MLmodel"
            assert mlmodel_path.read_bytes() == b"artifact_path: model\n"
            if hasattr(ModelsArtifactRepository, "_add_registered_model_meta_file"):
                meta_path = Path(result_dir) / "registered_model_meta"
                assert meta_path.exists()
                meta_text = meta_path.read_text(encoding="utf-8")
                assert "model_name: m1" in meta_text
                assert "model_version: '1'" in meta_text

            # 3. Read-only enforcement remains active after fallback
            for m in ("log_artifact", "log_artifacts", "delete_artifacts"):
                with pytest.raises(
                    MlflowException, match="Registered models are read-only"
                ):
                    getattr(repo, m)("test_path")

            inner_repo = repo.repo
            for m in (
                "create_multipart_upload",
                "complete_multipart_upload",
                "abort_multipart_upload",
            ):
                with pytest.raises(
                    MlflowException, match="Registered models are read-only"
                ):
                    getattr(inner_repo, m)(
                        "file", "part" if "complete" in m or "abort" in m else 1
                    )

    def test_model_id_length_and_unsupported_chars(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        # 1. 200-char ID succeeds
        long_id = "m-" + ("a" * 198)
        assert len(long_id) == 200
        long_name = f"projects/p1/locations/l1/registries/r1/registeredModels/{long_id}"
        search_resp = _json_resp(
            {"registeredModels": [{"name": long_name, "displayName": "m_long"}]}
        )
        tok_resp = _json_resp(
            {"credentials": {"accessToken": "tok1", "expireTime": _EXP_2099}}
        )
        with (
            mock.patch("requests.Session.get", return_value=search_resp) as mock_get,
            mock.patch("requests.Session.post", return_value=tok_resp),
            mock.patch(
                "mlflow.MlflowClient.get_model_version_download_uri",
                return_value=f"gs://bkt/registeredModels/{long_id}/versions/1",
            ),
            mock.patch(
                "mlflow.store.artifact.utils.models.get_model_name_and_version",
                return_value=("m_long", "1"),
            ),
        ):
            repo = GcpModelsArtifactRepository(
                "models:/m_long/1", registry_uri=_GCP_REGISTRY_URI
            )
            assert isinstance(repo.repo, GcpGCSArtifactRepository)
            assert mock_get.called
            assert repo.repo._cab_credentials is not None
            assert (
                repo.repo._cab_credentials.resource_url
                == f"{_GCP_REGISTRY_URI}/registeredModels/{long_id}"
            )

        # 2. 257-char ID fails closed with INTERNAL_ERROR
        too_long = "m-" + ("a" * 255)
        bad_name = f"projects/p1/locations/l1/registries/r1/registeredModels/{too_long}"
        bad_search = _json_resp(
            {"registeredModels": [{"name": bad_name, "displayName": "m_toolong"}]}
        )
        with (
            mock.patch("requests.Session.get", return_value=bad_search),
            pytest.raises(MlflowException, match="invalid or too long"),
        ):
            resolve_registered_model_resource_url(_GCP_REGISTRY_URI, "m_toolong")

        # 3. Model ID with spaces fails closed in cab mode, falls back in auto mode
        space_name = "projects/p1/locations/l1/registries/r1/registeredModels/my model"
        space_search = _json_resp(
            {"registeredModels": [{"name": space_name, "displayName": "m_space"}]}
        )
        with (
            mock.patch("requests.Session.get", return_value=space_search),
            pytest.raises(UnsupportedCabModelIdError, match="not supported by CAB"),
        ):
            resolve_registered_model_resource_url(_GCP_REGISTRY_URI, "m_space")

        caplog.clear()
        monkeypatch.setenv(AUTH_MODE_ENV_VAR, AUTH_MODE_AUTO)
        with (
            mock.patch("requests.Session.get", return_value=space_search),
            mock.patch(
                "mlflow.MlflowClient.get_model_version_download_uri",
                return_value="gs://bkt/registeredModels/my-model/versions/1",
            ),
            mock.patch(
                "mlflow.store.artifact.utils.models.get_model_name_and_version",
                return_value=("m_space", "1"),
            ),
        ):
            auto_repo = GcpModelsArtifactRepository(
                "models:/m_space/1", registry_uri=_GCP_REGISTRY_URI
            )
            assert isinstance(auto_repo.repo, GcpGCSArtifactRepository)
            assert auto_repo.repo._cab_credentials is None
            assert any(
                "unsupported_model_id_for_cab" in r.message for r in caplog.records
            )

    def test_search_registered_models_fallback_candidate_and_precedence(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # 1. rm.name == res_id succeeds when displayName differs from immutable ID
        uid = "uid-123"
        res_name = f"projects/p1/locations/l1/registries/r1/registeredModels/{uid}"
        search_resp_diff_disp = _json_resp(
            {"registeredModels": [{"name": res_name, "displayName": "my-display"}]}
        )
        tok_resp = _json_resp(
            {"credentials": {"accessToken": "tok1", "expireTime": _EXP_2099}}
        )
        with (
            mock.patch("requests.Session.get", return_value=search_resp_diff_disp),
            mock.patch("requests.Session.post", return_value=tok_resp),
            mock.patch(
                "mlflow.MlflowClient.get_model_version_download_uri",
                return_value=f"gs://bkt/registeredModels/{uid}/versions/1",
            ),
            mock.patch(
                "mlflow.store.artifact.utils.models.get_model_name_and_version",
                return_value=(uid, "1"),
            ),
        ):
            resolved = resolve_registered_model_resource_url(_GCP_REGISTRY_URI, uid)
            expected_url = f"{_GCP_REGISTRY_URI}/registeredModels/{uid}"
            assert resolved == expected_url
            repo = GcpModelsArtifactRepository(
                f"models:/{uid}/1", registry_uri=_GCP_REGISTRY_URI
            )
            assert isinstance(repo.repo, GcpGCSArtifactRepository)
            assert repo.repo._cab_credentials is not None
            assert repo.repo._cab_credentials.resource_url == expected_url

        reset_cab_credentials_cache()

        # 2. Exact displayName match on page 2 wins over res_id fallback
        # candidate on page 1
        name_p1 = "projects/p1/locations/l1/registries/r1/registeredModels/target-name"
        page1 = _json_resp(
            {
                "registeredModels": [
                    {
                        "name": name_p1,
                        "displayName": "other-display",
                    }
                ],
                "nextPageToken": "token-p2",
            }
        )
        name_p2 = "projects/p1/locations/l1/registries/r1/registeredModels/uid-456"
        page2 = _json_resp(
            {
                "registeredModels": [
                    {
                        "name": name_p2,
                        "displayName": "target-name",
                    }
                ]
            }
        )
        with mock.patch("requests.Session.get", side_effect=[page1, page2]):
            resolved = resolve_registered_model_resource_url(
                _GCP_REGISTRY_URI, "target-name"
            )
            assert resolved == f"{_GCP_REGISTRY_URI}/registeredModels/uid-456"

        reset_cab_credentials_cache()

        # 3. Deferred UnsupportedCabModelIdError check:
        # Page 1 has invalid CAB ID fallback candidate, but page 2 has valid exact match
        inv_name = "projects/p1/locations/l1/registries/r1/registeredModels/invalid.id"
        page1_invalid = _json_resp(
            {
                "registeredModels": [
                    {
                        "name": inv_name,
                        "displayName": "other-display",
                    }
                ],
                "nextPageToken": "token-p2",
            }
        )
        valid_name = (
            "projects/p1/locations/l1/registries/r1/registeredModels/valid-cab-id"
        )
        page2_valid = _json_resp(
            {
                "registeredModels": [
                    {
                        "name": valid_name,
                        "displayName": "invalid.id",
                    }
                ]
            }
        )
        with mock.patch(
            "requests.Session.get", side_effect=[page1_invalid, page2_valid]
        ):
            resolved = resolve_registered_model_resource_url(
                _GCP_REGISTRY_URI, "invalid.id"
            )
            assert resolved == f"{_GCP_REGISTRY_URI}/registeredModels/valid-cab-id"

        reset_cab_credentials_cache()

        # 4. Deferred UnsupportedCabModelIdError raises when no exact match exists
        page1_invalid_only = _json_resp(
            {
                "registeredModels": [
                    {
                        "name": inv_name,
                        "displayName": "other-display",
                    }
                ]
            }
        )
        with (
            mock.patch("requests.Session.get", return_value=page1_invalid_only),
            pytest.raises(UnsupportedCabModelIdError, match="not supported by CAB"),
        ):
            resolve_registered_model_resource_url(_GCP_REGISTRY_URI, "invalid.id")

    def test_euc_mode_quote_in_run_artifact_subpath_succeeds_with_recorded_model(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(AUTH_MODE_ENV_VAR, AUTH_MODE_EUC)
        # Record a registered model so _DOWNLOAD_URI_MODEL_MAP contains 'gs://bkt'
        record_registered_model_download_uri(
            "gs://bkt/registeredModels/uid-1/versions/1", _GCP_REGISTRY_URI, "m1"
        )

        # Run artifact subpath containing quote character
        quote_uri = "gs://bkt/runs/r1/artifacts/user's/registeredModels/foo/versions/1"
        repo = GcpGCSArtifactRepository(quote_uri)
        assert repo._auth_mode == AUTH_MODE_EUC
        assert not repo._is_registered_model
        assert repo._cab_credentials is None

    def test_legacy_run_source_path_and_tracking_routing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("MLFLOW_TRACKING_URI", _GCP_TRACKING_URI)
        legacy_dl = "gs://bkt/experiments/1/runs/r1/artifacts/model"

        with _mock_model_download(dl_uri=legacy_dl):
            # ModelsArtifactRepo resolves namespace CAB while staying read-only
            repo = GcpModelsArtifactRepository(
                "models:/m1/1", registry_uri=_GCP_REGISTRY_URI
            )
            assert isinstance(repo.repo, GcpGCSArtifactRepository)
            assert repo.repo._is_registered_model
            with pytest.raises(
                MlflowException, match="Registered models are read-only"
            ):
                repo.log_artifact("test")

            # Legacy URI was NOT recorded in _DOWNLOAD_URI_MODEL_MAP
            assert not _DOWNLOAD_URI_MODEL_MAP

            # Opening GcpGCSArtifactRepository on run path gets writable namespace CAB
            run_repo = GcpGCSArtifactRepository(legacy_dl)
            assert not run_repo._is_registered_model
            assert run_repo._cab_credentials is not None
            assert run_repo._cab_credentials.resource_kind == CabResourceKind.NAMESPACE

    def test_logged_model_insecure_tls_fails_closed(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("MLFLOW_TRACKING_INSECURE_TLS", "true")
        with pytest.raises(MlflowException, match="Disabling TLS"):
            GcpModelsArtifactRepository(
                "models:/m-12345", tracking_uri=_GCP_TRACKING_URI
            )

    def test_read_only_rejection_and_sticky_state(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        with _mock_model_download():
            repo = GcpModelsArtifactRepository(
                "models:/m1/1", registry_uri=_GCP_REGISTRY_URI
            )
            for m in ("log_artifact", "log_artifacts", "delete_artifacts"):
                with pytest.raises(
                    MlflowException, match="Registered models are read-only"
                ):
                    getattr(repo, m)("test_path")
            with mock.patch.object(
                repo.repo, "list_artifacts", return_value=[]
            ) as mock_list:
                assert repo.list_artifacts() == []
                mock_list.assert_called_with(None)
                assert repo.list_artifacts("sub") == []
                mock_list.assert_called_with("sub")

        gcs_repo = GcpGCSArtifactRepository(
            _MODEL_STORAGE_DL_URI,
            registry_uri=_GCP_REGISTRY_URI,
            cab_credentials=_model_creds(),
        )
        assert gcs_repo._is_registered_model
        with mock.patch("google.cloud.storage.Client", return_value=mock.MagicMock()):
            gcs_repo._switch_to_adc(cause=RuntimeError("err"))
        assert gcs_repo._cab_credentials is None and gcs_repo._is_registered_model
        for m in ("log_artifact", "log_artifacts", "delete_artifacts"):
            with pytest.raises(
                MlflowException, match="Registered models are read-only"
            ):
                getattr(gcs_repo, m)("test_path")

        monkeypatch.setenv(AUTH_MODE_ENV_VAR, AUTH_MODE_EUC)
        record_registered_model_download_uri(
            _MODEL_STORAGE_DL_URI, _GCP_REGISTRY_URI, "m1"
        )
        euc = GcpGCSArtifactRepository(
            _MODEL_STORAGE_DL_URI, registry_uri=_GCP_REGISTRY_URI
        )
        assert euc._is_registered_model and euc._cab_credentials is None
        with pytest.raises(MlflowException, match="Registered models are read-only"):
            euc.log_artifact("test_path")

    def test_non_gcp_and_logged_models(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("MLFLOW_REGISTRY_URI", _GCP_REGISTRY_URI)
        sup_init = (
            "mlflow.store.artifact.models_artifact_repo."
            "ModelsArtifactRepository.__init__"
        )
        with mock.patch(sup_init) as mock_sup:
            assert (
                not GcpModelsArtifactRepository(
                    "models://prof@databricks/m/1"
                )._is_gcp_registry
                and mock_sup.called
            )
            assert GcpModelsArtifactRepository(
                "models:/m-12345", tracking_uri="http://localhost:5000"
            ).is_logged_model_uri

        uc_target = (
            "mlflow.store.artifact.unity_catalog_models_artifact_repo."
            "UnityCatalogModelsArtifactRepository.download_artifacts"
        )
        with (
            mock.patch(uc_target) as mock_uc_dl,
            mock.patch("os.path.isdir", return_value=False),
        ):
            repo = GcpModelsArtifactRepository(
                "models:/uc-m/1", registry_uri="databricks-uc"
            )
            repo.download_artifacts("path", lineage_header_info={"h": "v"})  # type: ignore[no-untyped-call]
            mock_uc_dl.assert_called_with("path", None, lineage_header_info={"h": "v"})

        mock_inner = mock.MagicMock()
        with (
            mock.patch("mlflow.MlflowClient.get_logged_model") as mock_glm,
            mock.patch(
                "google_cloud_managed_mlflow.gcs_artifact_repo.GcpGCSArtifactRepository",
                return_value=mock_inner,
            ),
            mock.patch(
                "mlflow.store.artifact.artifact_repository_registry.get_artifact_repository",
                return_value=mock_inner,
            ),
            mock.patch(
                "mlflow.store.artifact.utils.models.get_model_name_and_version",
                return_value=("m-12345", None),
            ),
        ):
            mock_glm.return_value.artifact_location = "gs://bkt/artifacts/m"
            logged_repo = GcpModelsArtifactRepository(
                "models:/m-12345", tracking_uri=_GCP_TRACKING_URI
            )
            assert logged_repo.is_logged_model_uri and not logged_repo._is_gcp_registry
            logged_repo.log_artifact("file.txt")
            mock_inner.log_artifact.assert_called_with("file.txt", None)
            logged_repo.log_artifacts("dir")
            mock_inner.log_artifacts.assert_called_with("dir", None)

        base = "mlflow.store.artifact.models_artifact_repo.ModelsArtifactRepository"
        with mock.patch.multiple(
            base,
            __init__=lambda *a, **k: None,
            log_artifact=mock.DEFAULT,
            log_artifacts=mock.DEFAULT,
            delete_artifacts=mock.DEFAULT,
        ) as mocks:
            non_gcp = GcpModelsArtifactRepository(
                "models:/m1/1", registry_uri="http://localhost:5000"
            )
            assert not non_gcp._is_gcp_registry
            for m in ("log_artifact", "log_artifacts", "delete_artifacts"):
                getattr(non_gcp, m)("x")
                mocks[m].assert_called_once()


class TestRegisteredModelCabAndCache:
    def test_cab_uri_and_lookup_cache_boundaries(self) -> None:
        long_uri = (
            "gs://bkt/projects/p/locations/l/registries/r/registeredModels/m/versions/1"
        )
        assert derive_registered_model_cab_uri(f"{long_uri}/sub/w.bin") == long_uri
        m_uri = "gs://bkt/models/m"
        assert derive_registered_model_cab_uri(f"{m_uri}/MLmodel") == m_uri
        assert is_registered_model_storage_uri(long_uri)
        assert not is_registered_model_storage_uri(
            "gs://bkt/experiments/1/runs/r1/artifacts/m"
        )

        # Encoded traversal check
        with pytest.raises(MlflowException, match="percent-encoded"):
            derive_registered_model_cab_uri(f"{long_uri}/%2e%2e/2")

        run_art = "gs://bkt/runs/r1/artifacts"
        run_uri = f"{run_art}/sub/model"
        assert derive_registered_model_cab_uri(run_uri) == run_uri
        record_registered_model_download_uri(run_uri, _GCP_REGISTRY_URI, "m1")
        ns_repo = GcpGCSArtifactRepository(run_art, tracking_uri=_GCP_TRACKING_URI)
        creds = ns_repo._cab_credentials
        assert creds is not None and creds.resource_kind == CabResourceKind.NAMESPACE

        # record_registered_model_download_uri truncates /versions/<v>/sub to
        # the version root (/versions/<v>), updating the single version-root
        # mapping to "m1-sub", and segment boundaries are preserved (/v/10 != /v/1).
        v1_uri = "gs://bkt/registeredModels/m/versions/1"
        record_registered_model_download_uri(v1_uri, _GCP_REGISTRY_URI, "m1")
        record_registered_model_download_uri(
            f"{v1_uri}/sub", _GCP_REGISTRY_URI, "m1-sub"
        )
        assert lookup_registered_model_for_gcs_uri(f"{v1_uri}0/a.txt") is None
        hit = lookup_registered_model_for_gcs_uri(f"{v1_uri}/sub/weights.bin")
        assert hit is not None and hit.model_name == "m1-sub"
        with pytest.raises(MlflowException, match="traversal"):
            lookup_registered_model_for_gcs_uri(f"{v1_uri}/../../v2")

    def test_cache_eviction_and_lru_lifecycle(self) -> None:
        store = _build_registry_store_class()(_GCP_REGISTRY_URI)
        store.get_host_creds = mock.MagicMock()

        v1 = "gs://bkt/registeredModels/uid-m1/versions/1"
        v2 = "gs://bkt/registeredModels/uid-m1/versions/2"
        record_registered_model_download_uri(v1, _GCP_REGISTRY_URI, "m1")
        record_registered_model_download_uri(v2, _GCP_REGISTRY_URI, "m1")
        _MODEL_RESOURCE_NAME_CACHE[(_GCP_REGISTRY_URI, "m1")] = _CANONICAL_MODEL_URL

        # evict_cached_model_resource_url does NOT wipe download URIs
        evict_cached_model_resource_url(_GCP_REGISTRY_URI, "m1")
        assert (_GCP_REGISTRY_URI, "m1") not in _MODEL_RESOURCE_NAME_CACHE
        assert len(_DOWNLOAD_URI_MODEL_MAP) == 2

        # Invalidate wipes both
        invalidate_registered_model_resource_url(_GCP_REGISTRY_URI, "m1")
        assert not _DOWNLOAD_URI_MODEL_MAP

        # Re-populate
        record_registered_model_download_uri(v1, _GCP_REGISTRY_URI, "m1")
        _MODEL_RESOURCE_NAME_CACHE[(_GCP_REGISTRY_URI, "m1")] = _CANONICAL_MODEL_URL

        # Failed CreateRegisteredModel on 409 does NOT wipe caches
        with (
            mock.patch(
                "mlflow.utils.rest_utils.http_request",
                return_value=_json_resp({"error_code": "RESOURCE_ALREADY_EXISTS"}, 409),
            ),
            pytest.raises(MlflowException),
        ):
            store.create_registered_model("m1")
        assert (_GCP_REGISTRY_URI, "m1") in _MODEL_RESOURCE_NAME_CACHE
        assert len(_DOWNLOAD_URI_MODEL_MAP) == 1

        # Self-rename does not wipe or corrupt caches
        from google_cloud_managed_mlflow.cab_credentials import (
            rename_registered_model_cache_entries,
        )

        rename_registered_model_cache_entries(_GCP_REGISTRY_URI, "m1", "m1")
        pl_self = {
            "name": f"{_GCP_REGISTRY_URI}/registeredModels/m1",
            "displayName": "m1",
        }
        with mock.patch(
            "mlflow.utils.rest_utils.http_request", return_value=_json_resp(pl_self)
        ):
            store.rename_registered_model("m1", "m1")
        assert (_GCP_REGISTRY_URI, "m1") in _MODEL_RESOURCE_NAME_CACHE
        assert len(_DOWNLOAD_URI_MODEL_MAP) == 1

        # RenameRegisteredModel with GCP resource format updates caches to m2
        pl2 = {
            "name": "projects/p1/locations/l1/registries/r1/registeredModels/m1",
            "displayName": "m2",
        }
        with mock.patch(
            "mlflow.utils.rest_utils.http_request", return_value=_json_resp(pl2)
        ):
            store.rename_registered_model("m1", "m2")
        assert (_GCP_REGISTRY_URI, "m1") not in _MODEL_RESOURCE_NAME_CACHE
        assert (_GCP_REGISTRY_URI, "m2") in _MODEL_RESOURCE_NAME_CACHE
        hit_m2 = lookup_registered_model_for_gcs_uri(f"{v1}/w.bin")
        assert hit_m2 is not None and hit_m2.model_name == "m2"

        # DeleteRegisteredModel wipes caches
        with mock.patch(
            "mlflow.utils.rest_utils.http_request", return_value=_json_resp({})
        ):
            store.delete_registered_model("m1")
        assert not _DOWNLOAD_URI_MODEL_MAP
        assert (_GCP_REGISTRY_URI, "m2") not in _MODEL_RESOURCE_NAME_CACHE

        # LRU bounds test
        for i in range(1025):
            record_registered_model_download_uri(
                f"gs://bkt/registeredModels/m/versions/{i}", _GCP_REGISTRY_URI, f"m{i}"
            )
            with _MODEL_RESOURCE_NAME_CACHE_LOCK:
                _lru_put(
                    _MODEL_RESOURCE_NAME_CACHE,
                    (_GCP_REGISTRY_URI, f"m{i}"),
                    f"{_GCP_REGISTRY_URI}/m{i}",
                    1024,
                )
        assert (
            len(_MODEL_RESOURCE_NAME_CACHE) <= 1024
            and len(_DOWNLOAD_URI_MODEL_MAP) <= 1024
        )
        assert (_GCP_REGISTRY_URI, "m0") not in _MODEL_RESOURCE_NAME_CACHE

    def test_oss_mlflow_download_and_list_artifacts_versions_correlation(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import mlflow.artifacts
        from mlflow.store.artifact.artifact_repository_registry import (
            _artifact_repository_registry,
            get_artifact_repository,
        )

        monkeypatch.setitem(
            _artifact_repository_registry._registry, "gs", GcpGCSArtifactRepository
        )
        monkeypatch.setitem(
            _artifact_repository_registry._registry,
            "models",
            GcpModelsArtifactRepository,
        )

        v1_uri = "gs://bkt/registeredModels/uid-m1/versions/1"
        v2_uri = "gs://bkt/registeredModels/uid-m1/versions/2"
        record_registered_model_download_uri(v1_uri, _GCP_REGISTRY_URI, "m1")
        record_registered_model_download_uri(v2_uri, _GCP_REGISTRY_URI, "m1")

        mock_blob1 = mock.MagicMock()
        mock_blob1.name = "registeredModels/uid-m1/versions/1/model.pkl"
        mock_blob1.size = 100
        mock_blob1.updated = 1234567890

        mock_blob2 = mock.MagicMock()
        mock_blob2.name = "registeredModels/uid-m1/versions/2/model.pkl"
        mock_blob2.size = 200
        mock_blob2.updated = 1234567891

        def _dl1(p: Any, **_k: Any) -> None:
            Path(p).write_bytes(b"v1_data")

        def _dl2(p: Any, **_k: Any) -> None:
            Path(p).write_bytes(b"v2_data")

        mock_blob1.download_to_filename.side_effect = _dl1
        mock_blob2.download_to_filename.side_effect = _dl2

        mock_bucket = mock.MagicMock()

        def _list_blobs(prefix: str = "", **_k: Any) -> mock.MagicMock:
            res = mock.MagicMock()
            if "versions/1" in prefix:
                res.__iter__.return_value = [mock_blob1]
            elif "versions/2" in prefix:
                res.__iter__.return_value = [mock_blob2]
            else:
                res.__iter__.return_value = [mock_blob1, mock_blob2]
            res.pages = [mock.MagicMock(prefixes=[])]
            return res

        mock_bucket.list_blobs.side_effect = _list_blobs
        mock_bucket.get_blob.side_effect = lambda n, *a, **k: (
            mock_blob1 if "versions/1" in n else mock_blob2
        )
        mock_bucket.blob.side_effect = lambda n, *a, **k: (
            mock_blob1 if "versions/1" in n else mock_blob2
        )

        mock_client = mock.MagicMock()
        mock_client.bucket.return_value = mock_bucket

        m_name = "projects/p1/locations/l1/registries/r1/registeredModels/uid-m1"
        search_resp = _json_resp(
            {"registeredModels": [{"name": m_name, "displayName": "m1"}]}
        )
        tok_resp = _json_resp(
            {"credentials": {"accessToken": "tok1", "expireTime": _EXP_2099}}
        )

        for mode in (AUTH_MODE_CAB, AUTH_MODE_AUTO):
            monkeypatch.setenv(AUTH_MODE_ENV_VAR, mode)
            with (
                mock.patch("requests.Session.get", return_value=search_resp),
                mock.patch("requests.Session.post", return_value=tok_resp),
                mock.patch("google.cloud.storage.Client", return_value=mock_client),
            ):
                items_v1 = mlflow.artifacts.list_artifacts(v1_uri)
                assert len(items_v1) == 1
                assert items_v1[0].path.endswith("model.pkl")

                items_v2 = mlflow.artifacts.list_artifacts(v2_uri)
                assert len(items_v2) == 1
                assert items_v2[0].path.endswith("model.pkl")

                out1 = tmp_path / f"dl_v1_{mode}"
                res1 = Path(
                    mlflow.artifacts.download_artifacts(
                        artifact_uri=v1_uri, dst_path=str(out1)
                    )
                )
                model_file1 = res1 if res1.is_file() else next(res1.rglob("model.pkl"))
                assert model_file1.read_bytes() == b"v1_data"

                out2 = tmp_path / f"dl_v2_{mode}"
                res2 = Path(
                    mlflow.artifacts.download_artifacts(
                        artifact_uri=v2_uri, dst_path=str(out2)
                    )
                )
                model_file2 = res2 if res2.is_file() else next(res2.rglob("model.pkl"))
                assert model_file2.read_bytes() == b"v2_data"

                parent_repo = get_artifact_repository(
                    "gs://bkt/registeredModels/uid-m1/versions"
                )
                assert isinstance(parent_repo, GcpGCSArtifactRepository)
                assert parent_repo._is_registered_model
                with pytest.raises(
                    MlflowException, match="Registered models are read-only"
                ):
                    parent_repo.log_artifact("local.txt")

    def test_per_call_namespace_uri_fails_closed(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        with pytest.raises(MlflowException) as exc_info1:
            GcpModelsArtifactRepository("models:/m1/1", registry_uri=_GCP_TRACKING_URI)
        assert exc_info1.value.error_code == ErrorCode.Name(INVALID_PARAMETER_VALUE)
        assert "registries" in str(exc_info1.value)

        monkeypatch.delenv("MLFLOW_REGISTRY_URI", raising=False)
        with pytest.raises(MlflowException) as exc_info2:
            GcpModelsArtifactRepository("models:/m1/1", tracking_uri=_GCP_TRACKING_URI)
        assert exc_info2.value.error_code == ErrorCode.Name(INVALID_PARAMETER_VALUE)
        assert "registries" in str(exc_info2.value)

    def test_ambiguous_rename_and_not_found_eviction(self) -> None:
        store_cls = _build_registry_store_class()
        store = store_cls(_GCP_REGISTRY_URI)

        old_uri = "gs://bkt/registeredModels/uid-old/versions/1"
        new_uri = "gs://bkt/registeredModels/uid-new/versions/1"
        record_registered_model_download_uri(old_uri, _GCP_REGISTRY_URI, "m1_old")
        record_registered_model_download_uri(new_uri, _GCP_REGISTRY_URI, "m1_new")
        with _MODEL_RESOURCE_NAME_CACHE_LOCK:
            _MODEL_RESOURCE_NAME_CACHE[(_GCP_REGISTRY_URI, "m1_old")] = (
                f"{_GCP_REGISTRY_URI}/registeredModels/m1_old"
            )
            _MODEL_RESOURCE_NAME_CACHE[(_GCP_REGISTRY_URI, "m1_new")] = (
                f"{_GCP_REGISTRY_URI}/registeredModels/m1_new"
            )

        err_503 = _json_resp(
            {"error": {"code": 503, "message": "Service Unavailable"}}, 503
        )
        with mock.patch("mlflow.utils.rest_utils.http_request", return_value=err_503):
            with pytest.raises((RestException, GcpMlflowApiError)):
                store.rename_registered_model(name="m1_old", new_name="m1_new")

        with _MODEL_RESOURCE_NAME_CACHE_LOCK:
            assert (_GCP_REGISTRY_URI, "m1_old") not in _MODEL_RESOURCE_NAME_CACHE
            assert (_GCP_REGISTRY_URI, "m1_new") not in _MODEL_RESOURCE_NAME_CACHE
        assert lookup_registered_model_for_gcs_uri(f"{old_uri}/model.pkl") is None
        assert lookup_registered_model_for_gcs_uri(f"{new_uri}/model.pkl") is None

        uri_del = "gs://bkt/registeredModels/uid-del/versions/1"
        record_registered_model_download_uri(uri_del, _GCP_REGISTRY_URI, "m_del")
        assert lookup_registered_model_for_gcs_uri(f"{uri_del}/model.pkl") is not None

        del_resp = _json_resp({"registeredModels": []})
        with mock.patch("requests.Session.get", return_value=del_resp):
            with pytest.raises(MlflowException) as exc_info:
                resolve_registered_model_resource_url(_GCP_REGISTRY_URI, "m_del")
            assert exc_info.value.error_code == ErrorCode.Name(RESOURCE_DOES_NOT_EXIST)

        with _MODEL_RESOURCE_NAME_CACHE_LOCK:
            assert (_GCP_REGISTRY_URI, "m_del") not in _MODEL_RESOURCE_NAME_CACHE
        assert lookup_registered_model_for_gcs_uri(f"{uri_del}/model.pkl") is None


class TestConcurrencyAndBounds:
    def test_cab_token_credentials_validation_and_close(self) -> None:
        with pytest.raises(MlflowException, match="must follow format"):
            _model_creds(resource_url="https://other.com/m1")
        with pytest.raises(MlflowException, match="only supported for"):
            _model_creds(
                resource_url=_GCP_TRACKING_URI,
                resource_kind=CabResourceKind.NAMESPACE,
            )
        creds = _model_creds()
        assert (
            creds.resource_url == _CANONICAL_MODEL_URL
            and creds.registry_resource_url == _GCP_REGISTRY_URI
        )
        assert (
            creds.with_access_mode(CabAccessMode.READ_ONLY).access_mode
            == CabAccessMode.READ_ONLY
        )
        with pytest.raises(MlflowException, match="READ_ONLY"):
            creds.with_access_mode(CabAccessMode.READ_WRITE)
        with pytest.raises(MlflowException, match="READ_ONLY"):
            _model_creds(access_mode=CabAccessMode.READ_WRITE)
        with pytest.raises(MlflowException, match="READ_WRITE"):
            creds.elevate_to_read_write()
        creds.close()
        with pytest.raises(MlflowException, match="closed"):
            creds.refresh(mock.MagicMock())

    def test_stale_retry_and_preserved_error(self) -> None:
        creds = _model_creds()
        err_msg = {"error": {"message": "not contained within storage root"}}
        stale_resp = _json_resp(err_msg, 403)
        tok = {"credentials": {"accessToken": "new-tok", "expireTime": _EXP_2099}}
        posts = [stale_resp, _json_resp(tok)]
        renamed = f"{_GCP_REGISTRY_URI}/registeredModels/m1-renamed"
        res_target = (
            "google_cloud_managed_mlflow.cab_credentials."
            "_resolve_registered_model_resource_name"
        )

        with (
            mock.patch(
                "requests.Session.post", side_effect=lambda *a, **k: posts.pop(0)
            ),
            mock.patch(res_target, return_value=renamed) as mock_res,
        ):
            creds.refresh(mock.MagicMock())
            assert creds.token == "new-tok" and creds.resource_url == renamed
            mock_res.assert_called_once()
            hit = lookup_registered_model_for_gcs_uri(f"{_MODEL_STORAGE_DL_URI}/w.bin")
            assert hit is not None and hit.model_name == "m1"

        # Check preserved original error when resolution fails or yields same url
        v2 = "gs://bkt/registeredModels/uid-m1/versions/2"
        record_registered_model_download_uri(v2, _GCP_REGISTRY_URI, "m1")
        c_same = _model_creds()
        with (
            mock.patch("requests.Session.post", return_value=stale_resp),
            mock.patch(res_target, return_value=_CANONICAL_MODEL_URL),
        ):
            with pytest.raises(GcpMlflowApiError) as exc_info:
                c_same.refresh(mock.MagicMock())
            assert exc_info.value.http_status == 403
            # Sibling version 2 was preserved because
            # refreshed_url == self._resource_url
            hit_v2 = lookup_registered_model_for_gcs_uri(f"{v2}/w.bin")
            assert hit_v2 is not None and hit_v2.model_name == "m1"

        # Deleted model (RESOURCE_DOES_NOT_EXIST) invalidates download map
        c_del = _model_creds()
        del_exc = MlflowException("del", error_code=RESOURCE_DOES_NOT_EXIST)
        with (
            mock.patch("requests.Session.post", return_value=stale_resp),
            mock.patch(res_target, side_effect=del_exc),
        ):
            with pytest.raises(GcpMlflowApiError):
                c_del.refresh(mock.MagicMock())
            assert not _DOWNLOAD_URI_MODEL_MAP

    def test_warning_dedup_and_pool(self, caplog: pytest.LogCaptureFixture) -> None:
        caplog.clear()
        for _ in range(5):
            _warn_cab_fallback_once(
                _GCP_REGISTRY_URI,
                "bkt",
                "initial_storage_forbidden",
                RuntimeError(),
            )
        msg = "using Application Default Credentials"
        assert sum(msg in r.message for r in caplog.records) == 1

        v1 = "gs://bkt/registeredModels/uid-m1/versions/1"
        record_registered_model_download_uri(v1, _GCP_REGISTRY_URI, "m1")
        uri = f"{v1}/w.bin"
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
            assert all(
                r is not None and r.model_name == "m1"
                for r in ex.map(lookup_registered_model_for_gcs_uri, [uri] * 32)
            )

    def test_stale_model_refresh_cache_preservation_and_root_eviction(self) -> None:
        old_uid = "uid-old"
        new_uid = "uid-new"
        old_uri = f"gs://bkt/registeredModels/{old_uid}/versions/1"
        new_v1 = f"gs://bkt/registeredModels/{new_uid}/versions/1"
        new_v2 = f"gs://bkt/registeredModels/{new_uid}/versions/2"
        new_url = f"{_GCP_REGISTRY_URI}/registeredModels/{new_uid}"

        record_registered_model_download_uri(old_uri, _GCP_REGISTRY_URI, "m1")
        record_registered_model_download_uri(new_v2, _GCP_REGISTRY_URI, "m1")

        creds = CabTokenCredentials(
            resource_url=f"{_GCP_REGISTRY_URI}/registeredModels/{old_uid}",
            resource_kind=CabResourceKind.REGISTERED_MODEL,
            access_mode=CabAccessMode.READ_ONLY,
            artifact_uri=new_v1,
            registry_resource_url=_GCP_REGISTRY_URI,
            model_name="m1",
        )

        stale_resp = _json_resp(
            {"error": {"message": "not contained within storage root"}}, 403
        )
        tok_resp = _json_resp(
            {"credentials": {"accessToken": "refreshed-tok", "expireTime": _EXP_2099}}
        )
        posts = [stale_resp, tok_resp]

        res_target = (
            "google_cloud_managed_mlflow.cab_credentials."
            "_resolve_registered_model_resource_name"
        )
        with (
            mock.patch(
                "requests.Session.post", side_effect=lambda *a, **k: posts.pop(0)
            ),
            mock.patch(res_target, return_value=new_url),
        ):
            creds.refresh(mock.MagicMock())
            assert creds.token == "refreshed-tok"
            assert creds.resource_url == new_url

        # 1. Refreshed model resource name is preserved in cache
        with _MODEL_RESOURCE_NAME_CACHE_LOCK:
            assert _MODEL_RESOURCE_NAME_CACHE.get((_GCP_REGISTRY_URI, "m1")) == new_url

        # 2. Sibling under active new_uid root is preserved
        hit_v2 = lookup_registered_model_for_gcs_uri(f"{new_v2}/model.pkl")
        assert hit_v2 is not None and hit_v2.model_name == "m1"

        # 3. Download URI under old_uid root is evicted
        hit_old = lookup_registered_model_for_gcs_uri(f"{old_uri}/model.pkl")
        assert hit_old is None


class TestRegisteredModelBoundaryEdgeCases:
    """Targeted boundary and edge-case coverage for registered models."""

    def test_cold_cache_and_uppercase_gs_read_only_and_cab_fail_closed(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        reset_cab_credentials_cache()
        uri = "GS://bkt/registeredModels/uid-m1/versions/1"
        assert is_registered_model_storage_or_parent_uri(uri) is True
        repo = GcpGCSArtifactRepository(uri)
        assert repo._is_registered_model is True
        dummy = tmp_path / "f.txt"
        dummy.write_text("content", encoding="utf-8")
        with pytest.raises(MlflowException, match="read-only"):
            repo.log_artifact(str(dummy))

        # Under cab mode, unregistered model path must not misroute to namespace CAB
        monkeypatch.setenv(AUTH_MODE_ENV_VAR, AUTH_MODE_CAB)
        with pytest.raises(MlflowException, match="cannot resolve CAB credentials"):
            GcpGCSArtifactRepository(uri, tracking_uri=_GCP_TRACKING_URI)

    def test_stale_token_double_failure_and_subpath_extract_model_storage_root(
        self,
    ) -> None:
        reset_cab_credentials_cache()
        subpath_root = _extract_model_storage_root(
            "gs://bkt/registeredModels/uid-new/versions/1/versions_meta"
        )
        assert subpath_root == "gs://bkt/registeredModels/uid-new"

        old_uri = "gs://bkt/registeredModels/uid-old/versions/1"
        record_registered_model_download_uri(old_uri, _GCP_REGISTRY_URI, "m1")

        creds = CabTokenCredentials(
            resource_url=f"{_GCP_REGISTRY_URI}/registeredModels/uid-old",
            resource_kind=CabResourceKind.REGISTERED_MODEL,
            access_mode=CabAccessMode.READ_ONLY,
            artifact_uri=old_uri,
            registry_resource_url=_GCP_REGISTRY_URI,
            model_name="m1",
        )

        stale_resp1 = _json_resp(
            {"error": {"message": "not contained within storage root"}}, 403
        )
        stale_resp2 = _json_resp(
            {"error": {"message": "not contained within storage root"}}, 403
        )
        posts = [stale_resp1, stale_resp2]

        res_target = (
            "google_cloud_managed_mlflow.cab_credentials."
            "_resolve_registered_model_resource_name"
        )
        with (
            mock.patch(
                "requests.Session.post", side_effect=lambda *a, **k: posts.pop(0)
            ),
            mock.patch(
                res_target,
                return_value=f"{_GCP_REGISTRY_URI}/registeredModels/uid-new",
            ),
        ):
            with pytest.raises(GcpMlflowApiError):
                creds.refresh(mock.MagicMock())

        # Double failure with keep_matching=False must evict the stale root
        assert lookup_registered_model_for_gcs_uri(f"{old_uri}/model.pkl") is None

    def test_cross_registry_mismatch_and_nested_prefixes_and_per_call_tracking_uri(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        reset_cab_credentials_cache()
        # 1. Cross-registry mismatch
        record_registered_model_download_uri(
            "gs://bkt/registeredModels/uid-m1/versions/1", _GCP_REGISTRY_URI, "m1"
        )
        matched = _match_registered_model_gcs_uri(
            "gs://bkt/registeredModels/uid-m1/versions/1",
            None,
            "https://mlflow.googleapis.com/v1/projects/p2/locations/l2/registries/r2",
        )
        assert matched is None

        # 2. Distinct-prefix longest-match check (parent model root vs version prefix)
        parent_prefix = "gs://bkt/registeredModels/uid-m1"
        version_prefix = "gs://bkt/registeredModels/uid-m1/versions/1"
        with _MODEL_RESOURCE_NAME_CACHE_LOCK:
            _DOWNLOAD_URI_MODEL_MAP[parent_prefix] = (_GCP_REGISTRY_URI, "m1-root")
            _DOWNLOAD_URI_MODEL_MAP[version_prefix] = (_GCP_REGISTRY_URI, "m1-v1")
        hit = lookup_registered_model_for_gcs_uri(
            "gs://bkt/registeredModels/uid-m1/versions/1/sub/model.pkl"
        )
        assert hit is not None and hit.model_name == "m1-v1"
        parent_hit = lookup_registered_model_for_gcs_uri(
            "gs://bkt/registeredModels/uid-m1/other/model.pkl"
        )
        assert parent_hit is not None and parent_hit.model_name == "m1-root"

        # 3. Per-call tracking_uri="http://localhost:5000" overrides global namespace
        monkeypatch.setenv("MLFLOW_TRACKING_URI", _GCP_TRACKING_URI)
        with mock.patch(
            "mlflow.store.artifact.models_artifact_repo.ModelsArtifactRepository.__init__",
            return_value=None,
        ):
            models_repo = GcpModelsArtifactRepository(
                "models:/m1/1", tracking_uri="http://localhost:5000"
            )
            assert models_repo._is_gcp_registry is False

    def test_concurrent_cab_fallback_no_closed_credentials_race(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Verify sibling threads calling fallback handle closed CAB credentials."""
        import google.api_core.exceptions

        monkeypatch.setenv(AUTH_MODE_ENV_VAR, AUTH_MODE_AUTO)
        repo = GcpGCSArtifactRepository(
            "gs://my-bucket/artifacts",
            tracking_uri=_GCP_TRACKING_URI,
            registry_uri=_GCP_REGISTRY_URI,
        )
        cab_creds = _model_creds()
        repo._cab_credentials = cab_creds

        mock_adc_client = mock.MagicMock()
        mock_adc_client.download_blob_to_file.return_value = None

        thread1_switched = threading.Event()
        thread2_entered = threading.Event()

        def op_thread1() -> str:
            if repo._cab_credentials is None:
                return "thread1_success"
            thread2_entered.wait(timeout=5.0)
            raise google.api_core.exceptions.Forbidden("initial 403")

        def op_thread2() -> str:
            if repo._cab_credentials is None:
                return "thread2_success"
            thread2_entered.set()
            thread1_switched.wait(timeout=5.0)
            cab_creds.refresh(mock.MagicMock())
            return "unexpected"

        orig_switch = repo._switch_to_adc

        def instrumented_switch(*args: Any, **kwargs: Any) -> bool:
            res = orig_switch(*args, **kwargs)
            thread1_switched.set()
            return res

        repo._switch_to_adc = instrumented_switch  # type: ignore[method-assign]

        with (
            mock.patch("google.auth.default", return_value=(mock.MagicMock(), "proj")),
            mock.patch("google.cloud.storage.Client", return_value=mock_adc_client),
            concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool,
        ):
            f2 = pool.submit(repo._execute_with_cab_fallback, op_thread2)
            f1 = pool.submit(repo._execute_with_cab_fallback, op_thread1)
            res1 = f1.result(timeout=10.0)
            res2 = f2.result(timeout=10.0)

        assert res1 == "thread1_success"
        assert res2 == "thread2_success"
        assert repo._cab_credentials is None
        assert repo.client is mock_adc_client

    def test_cache_invalidation_and_self_rename_alias_consistency(self) -> None:
        reg_url = _GCP_REGISTRY_URI
        canonical_target_url = f"{reg_url}/registeredModels/uid-1"

        # 1. Invalidation of display-name alias purges resource ID and URIs
        with _MODEL_RESOURCE_NAME_CACHE_LOCK:
            _MODEL_RESOURCE_NAME_CACHE[(reg_url, "display-a")] = canonical_target_url
            _MODEL_RESOURCE_NAME_CACHE[(reg_url, "uid-1")] = canonical_target_url
        record_registered_model_download_uri(
            "gs://bkt/registeredModels/uid-1/versions/1", reg_url, "display-a"
        )
        assert (
            lookup_registered_model_for_gcs_uri(
                "gs://bkt/registeredModels/uid-1/versions/1"
            )
            is not None
        )

        invalidate_registered_model_resource_url(reg_url, "display-a")
        with _MODEL_RESOURCE_NAME_CACHE_LOCK:
            assert (reg_url, "display-a") not in _MODEL_RESOURCE_NAME_CACHE
            assert (reg_url, "uid-1") not in _MODEL_RESOURCE_NAME_CACHE
            assert not _DOWNLOAD_URI_MODEL_MAP

        reset_cab_credentials_cache()

        # 2. Self-rename (old_target == new_target) cleans up stale aliases
        # and migrates download URIs
        with _MODEL_RESOURCE_NAME_CACHE_LOCK:
            _MODEL_RESOURCE_NAME_CACHE[(reg_url, "display-a")] = canonical_target_url
            _MODEL_RESOURCE_NAME_CACHE[(reg_url, "uid-1")] = canonical_target_url
        record_registered_model_download_uri(
            "gs://bkt/registeredModels/uid-1/versions/1", reg_url, "display-a"
        )
        record_registered_model_download_uri(
            "gs://bkt/registeredModels/uid-1/versions/2", reg_url, "uid-1"
        )

        canon_res_id = "projects/p1/locations/l1/registries/r1/registeredModels/uid-1"
        rename_registered_model_cache_entries(
            reg_url,
            "uid-1",
            "uid-1",
            canonical_resource_id=canon_res_id,
        )

        with _MODEL_RESOURCE_NAME_CACHE_LOCK:
            assert (reg_url, "display-a") not in _MODEL_RESOURCE_NAME_CACHE
            assert (reg_url, "uid-1") in _MODEL_RESOURCE_NAME_CACHE
            assert _DOWNLOAD_URI_MODEL_MAP[
                "gs://bkt/registeredModels/uid-1/versions/1"
            ] == (reg_url, "uid-1")
            assert _DOWNLOAD_URI_MODEL_MAP[
                "gs://bkt/registeredModels/uid-1/versions/2"
            ] == (reg_url, "uid-1")

    def test_defensive_readonly_on_double_slash_and_quote_paths_and_factory_auth_mode(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        assert (
            is_registered_model_storage_or_parent_uri(
                "gs://bkt/registeredModels/uid-m1/versions/1//model.pkl"
            )
            is True
        )
        assert (
            is_registered_model_storage_or_parent_uri(
                "gs://bkt/registeredModels/my'model/versions/1"
            )
            is True
        )
        assert (
            is_registered_model_storage_or_parent_uri(
                "gs://bkt/runs/r1/artifacts/user's/registeredModels/foo/versions/1"
            )
            is False
        )
        assert (
            is_registered_model_storage_or_parent_uri(
                "gs:///registeredModels/uid-m1/versions/1"
            )
            is False
        )
        assert (
            is_registered_model_storage_or_parent_uri(
                "gs://bkt/registeredModels/uid-m1/versions/1?x=1"
            )
            is False
        )
        assert (
            is_registered_model_storage_or_parent_uri(
                "gs://bkt/registeredModels/uid-m1/versions/%2e%2e/1"
            )
            is False
        )
        assert (
            is_registered_model_storage_or_parent_uri(
                "gs://bkt/registeredModels/uid-m1/versions/../1"
            )
            is False
        )
        assert (
            is_registered_model_storage_or_parent_uri("http://example.com/x") is False
        )

        monkeypatch.setenv(AUTH_MODE_ENV_VAR, AUTH_MODE_EUC)
        repo = GcpGCSArtifactRepository(
            "gs://bkt/registeredModels/uid-m1/versions/1//model.pkl"
        )
        assert repo._is_registered_model is True
        dummy = tmp_path / "f.txt"
        dummy.write_text("test", encoding="utf-8")
        with pytest.raises(MlflowException, match="read-only"):
            repo.log_artifact(str(dummy))

        monkeypatch.delenv(AUTH_MODE_ENV_VAR, raising=False)
        factory_repo = GcpGCSArtifactRepository.for_registered_model(
            "gs://bkt/registeredModels/uid-m1/versions/1",
            model_name="m1",
            tracking_uri=None,
            effective_tracking_uri="",
            effective_registry_uri=_GCP_REGISTRY_URI,
            auth_mode=AUTH_MODE_EUC,
        )
        assert factory_repo._auth_mode == AUTH_MODE_EUC

    def test_registry_store_rpc_cache_sync_create_and_rename_404(self) -> None:
        reset_cab_credentials_cache()
        reg_url = _GCP_REGISTRY_URI

        # 1. CreateRegisteredModel 200 invalidates clean_name
        with _MODEL_RESOURCE_NAME_CACHE_LOCK:
            _MODEL_RESOURCE_NAME_CACHE[(reg_url, "m_create")] = (
                f"{reg_url}/registeredModels/old"
            )
        _sync_model_cache_after_rpc(
            reg_url, "CreateRegisteredModel", "m_create", None, None
        )
        with _MODEL_RESOURCE_NAME_CACHE_LOCK:
            assert (reg_url, "m_create") not in _MODEL_RESOURCE_NAME_CACHE

        # 2. RenameRegisteredModel 404 evicts clean_name and keeps clean_new_name
        with _MODEL_RESOURCE_NAME_CACHE_LOCK:
            _MODEL_RESOURCE_NAME_CACHE[(reg_url, "m_old")] = (
                f"{reg_url}/registeredModels/old"
            )
            _MODEL_RESOURCE_NAME_CACHE[(reg_url, "m_new")] = (
                f"{reg_url}/registeredModels/new"
            )
        err_404 = GcpMlflowApiError("Not found", http_status=404)
        _sync_model_cache_after_rpc(
            reg_url, "RenameRegisteredModel", "m_old", "m_new", err_404
        )
        with _MODEL_RESOURCE_NAME_CACHE_LOCK:
            assert (reg_url, "m_old") not in _MODEL_RESOURCE_NAME_CACHE
            assert (reg_url, "m_new") in _MODEL_RESOURCE_NAME_CACHE

    def test_unrelated_bucket_quotes_and_shared_bucket_run_artifacts(
        self,
    ) -> None:
        reset_cab_credentials_cache()
        # 1. Unrelated bucket containing literal quote marks returns None without error
        assert (
            lookup_registered_model_for_gcs_uri(
                "gs://other-bkt/user's/registeredModels/m1/versions/1"
            )
            is None
        )

        # 2. Shared bucket run artifact containing registeredModels in path
        run_uri = "gs://bkt/runs/r1/artifacts/registeredModels/foo/versions/1/w.bin"
        assert is_registered_model_storage_uri(run_uri) is False
        assert is_registered_model_storage_or_parent_uri(run_uri) is False
        assert lookup_registered_model_for_gcs_uri(run_uri) is None

        # After recording gs://bkt/registeredModels/uid-1/versions/1, unrelated run
        # artifact with literal quotes in the same bucket returns None without error.
        rec_uri = "gs://bkt/registeredModels/uid-1/versions/1"
        record_registered_model_download_uri(rec_uri, _GCP_REGISTRY_URI, "m1")
        assert (
            lookup_registered_model_for_gcs_uri(
                "gs://bkt/runs/r1/artifacts/user's_file.txt"
            )
            is None
        )

        # Traversal on the recorded bucket raises MlflowException
        with pytest.raises(MlflowException, match="traversal"):
            lookup_registered_model_for_gcs_uri(
                "gs://bkt/registeredModels/uid-1/versions/1/../2"
            )

    def test_for_registered_model_euc_mode_with_quotes(self) -> None:
        quoted_uri = "gs://bkt/user's/registeredModels/m1/versions/1"
        repo = GcpGCSArtifactRepository.for_registered_model(
            quoted_uri,
            model_name="m1",
            tracking_uri=_GCP_TRACKING_URI,
            effective_tracking_uri=_GCP_TRACKING_URI,
            effective_registry_uri=_GCP_REGISTRY_URI,
            auth_mode=AUTH_MODE_EUC,
        )
        assert repo._is_registered_model is True
        assert repo._cab_credentials is None

    def test_init_gcp_logged_model_gs_vs_non_gs_delegation(self) -> None:
        mock_client = mock.MagicMock()
        mock_logged_model_gs = mock.MagicMock(
            artifact_location="gs://bkt/exp/runs/r1/artifacts/model"
        )
        mock_logged_model_s3 = mock.MagicMock(
            artifact_location="s3://bkt/exp/runs/r1/artifacts/model"
        )

        with (
            mock.patch("mlflow.MlflowClient", return_value=mock_client),
            mock.patch(
                "mlflow.store.artifact.utils.models.get_model_name_and_version",
                return_value=("m-123", "1"),
            ),
        ):
            # 1. gs:// scheme instantiates GcpGCSArtifactRepository directly
            mock_client.get_logged_model.return_value = mock_logged_model_gs
            repo_gs = GcpModelsArtifactRepository(
                "models:/m-123", tracking_uri=_GCP_TRACKING_URI
            )
            assert isinstance(repo_gs.repo, GcpGCSArtifactRepository)

            # 2. non-gs (e.g. s3://) delegates to get_artifact_repository
            mock_client.get_logged_model.return_value = mock_logged_model_s3
            with mock.patch(
                "mlflow.store.artifact.artifact_repository_registry.get_artifact_repository",
                return_value=mock.MagicMock(),
            ) as mock_get_repo:
                GcpModelsArtifactRepository(
                    "models:/m-123", tracking_uri=_GCP_TRACKING_URI
                )
                mock_get_repo.assert_called_once_with(
                    "s3://bkt/exp/runs/r1/artifacts/model",
                    tracking_uri=_GCP_TRACKING_URI,
                    registry_uri=None,
                )

    def test_unsupported_cab_model_id_is_storage_path_and_fails_closed_in_cab_mode(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        reset_cab_credentials_cache()
        uri = "gs://bkt/registeredModels/my.model/versions/1"
        assert is_registered_model_storage_or_parent_uri(uri) is True
        assert is_registered_model_storage_uri(uri) is True

        repo = GcpGCSArtifactRepository(uri)
        assert repo._is_registered_model is True
        f = tmp_path / "x.txt"
        f.write_text("x", encoding="utf-8")
        with pytest.raises(MlflowException, match="Registered models are read-only"):
            repo.log_artifact(str(f))

        # Constructing GcpGCSArtifactRepository with tracking_uri on registered
        # model path verifies _is_registered_model is True and NOT routed to CAB
        repo_tracking = GcpGCSArtifactRepository(uri, tracking_uri=_GCP_TRACKING_URI)
        assert repo_tracking._is_registered_model is True
        assert repo_tracking._cab_credentials is None

        monkeypatch.setenv(AUTH_MODE_ENV_VAR, AUTH_MODE_CAB)
        with (
            mock.patch(
                "google_cloud_managed_mlflow.cab_credentials.resolve_registered_model_resource_url",
                return_value=f"{_GCP_REGISTRY_URI}/registeredModels/my.model",
            ),
            pytest.raises(UnsupportedCabModelIdError),
        ):
            GcpGCSArtifactRepository.for_registered_model(
                uri,
                model_name="my.model",
                tracking_uri=_GCP_TRACKING_URI,
                effective_tracking_uri=_GCP_TRACKING_URI,
                effective_registry_uri=_GCP_REGISTRY_URI,
                auth_mode=AUTH_MODE_CAB,
            )

        # for_registered_model in AUTO mode falls back to ADC and remains read-only
        monkeypatch.setenv(AUTH_MODE_ENV_VAR, AUTH_MODE_AUTO)
        with mock.patch(
            "google_cloud_managed_mlflow.cab_credentials.resolve_registered_model_resource_url",
            return_value=f"{_GCP_REGISTRY_URI}/registeredModels/my.model",
        ):
            repo_auto = GcpGCSArtifactRepository.for_registered_model(
                uri,
                model_name="my.model",
                tracking_uri=_GCP_TRACKING_URI,
                effective_tracking_uri=_GCP_TRACKING_URI,
                effective_registry_uri=_GCP_REGISTRY_URI,
                auth_mode=AUTH_MODE_AUTO,
            )
            assert repo_auto._is_registered_model is True
            assert repo_auto._cab_credentials is None
            with pytest.raises(
                MlflowException, match="Registered models are read-only"
            ):
                repo_auto.log_artifact(str(f))

    def test_cold_cache_immutable_id_eviction_and_rename_null_model(self) -> None:
        reset_cab_credentials_cache()
        store = _build_registry_store_class()(_GCP_REGISTRY_URI)
        store.get_host_creds = mock.MagicMock()

        uri = "gs://bkt/registeredModels/uid-1/versions/1"
        record_registered_model_download_uri(uri, _GCP_REGISTRY_URI, "display-a")
        with _MODEL_RESOURCE_NAME_CACHE_LOCK:
            _MODEL_RESOURCE_NAME_CACHE.clear()

        # DeleteRegisteredModel by immutable ID "uid-1" must evict download URIs
        with mock.patch(
            "mlflow.utils.rest_utils.http_request", return_value=_json_resp({})
        ):
            store.delete_registered_model("uid-1")
        assert lookup_registered_model_for_gcs_uri(uri) is None

        # Cold-cache rename by immutable ID "uid-1" -> "display-b" migrates URIs
        record_registered_model_download_uri(uri, _GCP_REGISTRY_URI, "display-a")
        with _MODEL_RESOURCE_NAME_CACHE_LOCK:
            _MODEL_RESOURCE_NAME_CACHE.clear()
        with mock.patch(
            "mlflow.utils.rest_utils.http_request",
            return_value=_json_resp(
                {
                    "registeredModel": {
                        "name": f"{_GCP_REGISTRY_URI}/registeredModels/uid-1",
                        "displayName": "display-b",
                    }
                }
            ),
        ):
            store.rename_registered_model("uid-1", "display-b")
        hit = lookup_registered_model_for_gcs_uri(uri)
        assert hit is not None and hit.model_name == "display-b"

        # Cold-cache rename for non-CAB ID "my.model" -> "display-b" migrates URIs
        non_cab_uri = "gs://bkt/registeredModels/my.model/versions/1"
        record_registered_model_download_uri(non_cab_uri, _GCP_REGISTRY_URI, "my.model")
        with _MODEL_RESOURCE_NAME_CACHE_LOCK:
            _MODEL_RESOURCE_NAME_CACHE.clear()
        with mock.patch(
            "mlflow.utils.rest_utils.http_request",
            return_value=_json_resp(
                {
                    "registeredModel": {
                        "name": f"{_GCP_REGISTRY_URI}/registeredModels/my.model",
                        "displayName": "display-b",
                    }
                }
            ),
        ):
            store.rename_registered_model("my.model", "display-b")
        hit_non_cab = lookup_registered_model_for_gcs_uri(non_cab_uri)
        assert hit_non_cab is not None and hit_non_cab.model_name == "display-b"

        # Warm display-name alias eviction on rename
        with _MODEL_RESOURCE_NAME_CACHE_LOCK:
            _MODEL_RESOURCE_NAME_CACHE[(_GCP_REGISTRY_URI, "display-b")] = (
                f"{_GCP_REGISTRY_URI}/registeredModels/uid-1"
            )
            _MODEL_RESOURCE_NAME_CACHE[(_GCP_REGISTRY_URI, "uid-1")] = (
                f"{_GCP_REGISTRY_URI}/registeredModels/uid-1"
            )
        rename_registered_model_cache_entries(
            _GCP_REGISTRY_URI,
            "uid-1",
            "display-c",
            canonical_resource_id=f"{_GCP_REGISTRY_URI}/registeredModels/uid-1",
        )
        with _MODEL_RESOURCE_NAME_CACHE_LOCK:
            assert (_GCP_REGISTRY_URI, "display-b") not in _MODEL_RESOURCE_NAME_CACHE
            assert (_GCP_REGISTRY_URI, "display-c") in _MODEL_RESOURCE_NAME_CACHE

        # RenameRegisteredModel with "registeredModel": None must not crash
        with mock.patch(
            "mlflow.utils.rest_utils.http_request",
            return_value=_json_resp({"registeredModel": None}),
        ):
            store.rename_registered_model("m1", "m2")

    def test_mlflow_set_registry_uri_precedence_in_models_repo(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import mlflow

        monkeypatch.setenv("MLFLOW_TRACKING_URI", _GCP_TRACKING_URI)
        mlflow.set_registry_uri(_GCP_REGISTRY_URI)
        with _mock_model_download():
            repo = GcpModelsArtifactRepository("models:/m1/1")
            assert repo._is_gcp_registry is True
            assert repo.registry_uri == _GCP_REGISTRY_URI

    def test_resolve_effective_registry_uri_helper(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import mlflow.tracking._model_registry.utils as mr_utils

        from google_cloud_managed_mlflow.uri import resolve_effective_registry_uri

        # 1. explicit registry_uri
        assert (
            resolve_effective_registry_uri("https://reg", _GCP_TRACKING_URI)
            == "https://reg"
        )

        # 2. explicit env var
        monkeypatch.setenv("MLFLOW_REGISTRY_URI", "https://reg-env")
        assert (
            resolve_effective_registry_uri(None, _GCP_TRACKING_URI) == "https://reg-env"
        )
        monkeypatch.delenv("MLFLOW_REGISTRY_URI")

        # 3. context helper returns None, but _registry_uri attribute is set
        monkeypatch.setattr(
            mr_utils, "_get_registry_uri_from_context", lambda: None, raising=False
        )
        monkeypatch.setattr(
            mr_utils, "_registry_uri", "https://reg-attr", raising=False
        )
        assert (
            resolve_effective_registry_uri(None, _GCP_TRACKING_URI)
            == "https://reg-attr"
        )

        # 4. neither is set -> fallback to effective_tracking_uri
        monkeypatch.setattr(mr_utils, "_registry_uri", None, raising=False)
        assert (
            resolve_effective_registry_uri(None, _GCP_TRACKING_URI) == _GCP_TRACKING_URI
        )

    def test_registry_storage_roots_containing_artifacts_segment(
        self, tmp_path: Path
    ) -> None:
        """Verify Registry roots with /artifacts in path match model patterns."""
        reset_cab_credentials_cache()
        reg_uri1 = "gs://bkt/artifacts/registeredModels/uid-1/versions/1"
        reg_uri2 = "gs://bkt/ml/artifacts/registry/registeredModels/uid-1/versions/1"
        parent_uri = "gs://bkt/artifacts/registeredModels/uid-1/versions"

        # 1. is_registered_model_storage_uri
        assert is_registered_model_storage_uri(reg_uri1) is True
        assert is_registered_model_storage_uri(reg_uri2) is True
        assert is_registered_model_storage_uri(parent_uri) is False

        # 2. is_registered_model_storage_or_parent_uri
        assert is_registered_model_storage_or_parent_uri(reg_uri1) is True
        assert is_registered_model_storage_or_parent_uri(reg_uri2) is True
        assert is_registered_model_storage_or_parent_uri(parent_uri) is True

        # 3. Canonical run / model / experiment hash artifact URIs remain False
        run_uri = "gs://bkt/runs/r1/artifacts/registeredModels/uid-1/versions/1"
        model_uri = "gs://bkt/models/m1/artifacts/registeredModels/uid-1/versions/1"
        hex_uri = (
            "gs://bkt/0123456789abcdef0123456789abcdef"
            "/artifacts/registeredModels/uid-1/versions/1"
        )
        for non_reg_uri in (run_uri, model_uri, hex_uri):
            assert is_registered_model_storage_uri(non_reg_uri) is False
            assert is_registered_model_storage_or_parent_uri(non_reg_uri) is False

        # 4. record_registered_model_download_uri + lookup_registered_model_for_gcs_uri
        record_registered_model_download_uri(reg_uri1, _GCP_REGISTRY_URI, "model1")
        record_registered_model_download_uri(reg_uri2, _GCP_REGISTRY_URI, "model2")

        hit1 = lookup_registered_model_for_gcs_uri(f"{reg_uri1}/model.pkl")
        assert hit1 is not None and hit1.model_name == "model1"
        hit2 = lookup_registered_model_for_gcs_uri(f"{reg_uri2}/model.pkl")
        assert hit2 is not None and hit2.model_name == "model2"

        # 5. for_registered_model CAB routing when GCP tracking_uri is active
        model_res_name = "projects/p1/locations/l1/registries/r1/registeredModels/uid-1"
        search_resp = _json_resp(
            {
                "registeredModels": [
                    {
                        "name": model_res_name,
                        "displayName": "model1",
                    }
                ]
            }
        )
        with mock.patch("requests.Session.get", return_value=search_resp):
            repo = GcpGCSArtifactRepository.for_registered_model(
                reg_uri1,
                model_name="model1",
                tracking_uri=_GCP_TRACKING_URI,
                effective_tracking_uri=_GCP_TRACKING_URI,
                effective_registry_uri=_GCP_REGISTRY_URI,
            )
            assert repo._is_registered_model is True
            assert repo._cab_credentials is not None

        # 6. Cold-cache get_artifact_repository(dl_uri) read-only enforcement
        reset_cab_credentials_cache()
        ensure_artifact_repo_registered()
        from mlflow.store.artifact.artifact_repository_registry import (
            get_artifact_repository,
        )

        cold_repo = get_artifact_repository(reg_uri1)
        assert getattr(cold_repo, "_is_registered_model", False) is True
        assert getattr(cold_repo, "_cab_credentials", None) is None
        test_file = tmp_path / "artifact.txt"
        test_file.write_text("content", encoding="utf-8")
        with pytest.raises(MlflowException, match="Registered models are read-only"):
            cold_repo.log_artifact(str(test_file))

    def test_stale_refresh_non_blocking_property_access(self) -> None:
        """Verify reading token properties concurrently does not block on HTTP."""
        old_url = f"{_GCP_REGISTRY_URI}/registeredModels/uid-old"
        creds = CabTokenCredentials(
            resource_url=old_url,
            resource_kind=CabResourceKind.REGISTERED_MODEL,
            access_mode=CabAccessMode.READ_ONLY,
            registry_resource_url=_GCP_REGISTRY_URI,
            model_name="m1",
        )
        stale_resp = _json_resp(
            {"error": {"message": "not contained within storage root"}}, 403
        )
        tok_resp = _json_resp(
            {"credentials": {"accessToken": "new-tok", "expireTime": _EXP_2099}}
        )

        in_resolver = threading.Event()
        resume_resolver = threading.Event()

        def _blocking_resolve(*_a: Any, **_k: Any) -> str:
            in_resolver.set()
            resume_resolver.wait(timeout=5)
            return f"{_GCP_REGISTRY_URI}/registeredModels/uid-new"

        posts = [stale_resp, tok_resp]
        res_target = (
            "google_cloud_managed_mlflow.cab_credentials."
            "resolve_registered_model_resource_url"
        )

        refresh_done = threading.Event()

        def _do_refresh() -> None:
            creds.refresh(mock.MagicMock())
            refresh_done.set()

        with (
            mock.patch(
                "requests.Session.post", side_effect=lambda *a, **k: posts.pop(0)
            ),
            mock.patch(res_target, side_effect=_blocking_resolve),
        ):
            refresh_thread = threading.Thread(target=_do_refresh)
            refresh_thread.start()
            assert in_resolver.wait(timeout=5)

            # While refresh is blocked in resolve_registered_model_resource_url,
            # verify all 5 properties can be read immediately from another thread
            prop_values: dict[str, Any] = {}

            def _read_props() -> None:
                prop_values["token_state"] = creds.token_state
                prop_values["valid"] = creds.valid
                prop_values["expired"] = creds.expired
                prop_values["resource_url"] = creds.resource_url
                prop_values["access_mode"] = creds.access_mode

            reader_thread = threading.Thread(target=_read_props)
            reader_thread.start()
            reader_thread.join(timeout=1.0)
            assert not reader_thread.is_alive()

            assert prop_values["resource_url"] == old_url
            assert prop_values["access_mode"] == CabAccessMode.READ_ONLY

            resume_resolver.set()
            refresh_thread.join(timeout=5)
            assert refresh_done.is_set()
            assert creds.token == "new-tok"
            assert creds.resource_url == f"{_GCP_REGISTRY_URI}/registeredModels/uid-new"

    @pytest.mark.parametrize(
        ("retry_post_resp", "expected_exc_type"),
        [
            pytest.param(
                _json_resp({"error": {"message": "Service Unavailable"}}, 503),
                GcpMlflowApiError,
                id="http-503-unavailable",
            ),
            pytest.param(
                _json_resp({"credentials": {"accessToken": 12345}}),
                MlflowException,
                id="malformed-json-non-string-token",
            ),
            pytest.param(
                _json_resp({"credentials": {}}),
                MlflowException,
                id="malformed-json-missing-fields",
            ),
        ],
    )
    def test_stale_retry_failure_preserves_original_resource_url(
        self, retry_post_resp: requests.Response, expected_exc_type: type[Exception]
    ) -> None:
        old_url = f"{_GCP_REGISTRY_URI}/registeredModels/uid-old"
        new_url = f"{_GCP_REGISTRY_URI}/registeredModels/uid-new"
        creds = CabTokenCredentials(
            resource_url=old_url,
            resource_kind=CabResourceKind.REGISTERED_MODEL,
            access_mode=CabAccessMode.READ_ONLY,
            registry_resource_url=_GCP_REGISTRY_URI,
            model_name="m1",
        )
        stale_resp = _json_resp(
            {"error": {"message": "not contained within storage root"}}, 403
        )
        posts = [stale_resp, retry_post_resp]
        res_target = (
            "google_cloud_managed_mlflow.cab_credentials."
            "resolve_registered_model_resource_url"
        )
        with (
            mock.patch(
                "requests.Session.post", side_effect=lambda *a, **k: posts.pop(0)
            ),
            mock.patch(res_target, return_value=new_url),
        ):
            with pytest.raises(expected_exc_type):
                creds.refresh(mock.MagicMock())
            # resource_url must remain unchanged at old_url
            assert creds.resource_url == old_url

    def test_concurrent_elevate_and_close_during_refresh(self) -> None:
        """Verify concurrent elevate loops to mint RW, and close fails closed."""
        old_url = _GCP_TRACKING_URI
        creds = CabTokenCredentials(
            resource_url=old_url,
            resource_kind=CabResourceKind.NAMESPACE,
            access_mode=CabAccessMode.READ_ONLY,
        )

        in_first_post = threading.Event()
        resume_first_post = threading.Event()
        verbs_called: list[str] = []

        def _fake_post(
            session: requests.Session, url: str, verb: str, payload: dict[str, Any]
        ) -> requests.Response:
            verbs_called.append(verb)
            if verb == "issueReadArtifactToken":
                in_first_post.set()
                resume_first_post.wait(timeout=5)
            tok = {
                "credentials": {
                    "accessToken": f"tok-{verb}",
                    "expireTime": _EXP_2099,
                }
            }
            return _json_resp(tok)

        # 1. Concurrent elevate_to_read_write while refresh is in flight
        refresh_done = threading.Event()

        def _do_refresh() -> None:
            creds.refresh(mock.MagicMock())
            refresh_done.set()

        with mock.patch.object(creds, "_post_token_rpc", side_effect=_fake_post):
            t = threading.Thread(target=_do_refresh)
            t.start()
            assert in_first_post.wait(timeout=5)

            # Concurrently elevate
            creds.elevate_to_read_write()
            assert creds.access_mode == CabAccessMode.READ_WRITE

            resume_first_post.set()
            t.join(timeout=5)
            assert refresh_done.is_set()

            # Verified refresh loop ran again with issueReadWriteArtifactToken
            assert verbs_called == [
                "issueReadArtifactToken",
                "issueReadWriteArtifactToken",
            ]
            assert creds.token == "tok-issueReadWriteArtifactToken"

        # 2. Concurrent close while refresh is in flight
        creds2 = CabTokenCredentials(
            resource_url=old_url,
            resource_kind=CabResourceKind.NAMESPACE,
            access_mode=CabAccessMode.READ_ONLY,
        )
        in_second_post = threading.Event()
        resume_second_post = threading.Event()

        def _fake_post2(
            session: requests.Session, url: str, verb: str, payload: dict[str, Any]
        ) -> requests.Response:
            in_second_post.set()
            resume_second_post.wait(timeout=5)
            tok = {
                "credentials": {
                    "accessToken": "tok-close",
                    "expireTime": _EXP_2099,
                }
            }
            return _json_resp(tok)

        refresh_exc: list[Exception] = []

        def _do_refresh2() -> None:
            try:
                creds2.refresh(mock.MagicMock())
            except Exception as e:
                refresh_exc.append(e)

        with mock.patch.object(creds2, "_post_token_rpc", side_effect=_fake_post2):
            t2 = threading.Thread(target=_do_refresh2)
            t2.start()
            assert in_second_post.wait(timeout=5)

            # Concurrently close
            creds2.close()

            resume_second_post.set()
            t2.join(timeout=5)

            assert len(refresh_exc) == 1
            assert isinstance(refresh_exc[0], MlflowException)
            assert "Cannot refresh closed CabTokenCredentials" in str(refresh_exc[0])

    def test_gcs_artifact_repo_pep562_getattr(self) -> None:
        from google_cloud_managed_mlflow import gcs_artifact_repo

        assert gcs_artifact_repo.AUTH_MODE_AUTO == "auto"
        assert gcs_artifact_repo.AUTH_MODE_CAB == "cab"
        assert gcs_artifact_repo.AUTH_MODE_EUC == "euc"
        assert (
            gcs_artifact_repo.AUTH_MODE_ENV_VAR
            == "GOOGLE_CLOUD_MANAGED_MLFLOW_AUTH_MODE"
        )
        with pytest.raises(AttributeError, match="has no attribute 'NON_EXISTENT'"):
            _ = gcs_artifact_repo.NON_EXISTENT
