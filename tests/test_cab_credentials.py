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

"""Tests for Credential Access Boundary (CAB) credentials and helpers."""

from __future__ import annotations

import datetime
import gc
import threading
import unittest.mock
import urllib.parse
from typing import Any

import google.auth.credentials
import pytest
import requests
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import (
    INTERNAL_ERROR,
    INVALID_PARAMETER_VALUE,
    RESOURCE_DOES_NOT_EXIST,
    ErrorCode,
)

from google_cloud_managed_mlflow.cab_credentials import (
    CabAccessMode,
    CabResourceKind,
    CabTokenCredentials,
    _resolve_registered_model_resource_name,
    derive_cab_directory_uri,
)
from google_cloud_managed_mlflow.uri import GcpMlflowApiError

_NAMESPACE_URL = (
    "https://mlflow.googleapis.com/v1/projects/my-proj"
    "/locations/us-central1/namespaces/my-ns"
)
_REGISTRY_URL = (
    "https://mlflow.googleapis.com/v1/projects/my-proj"
    "/locations/us-central1/registries/my-reg"
)


def _future_rfc3339(seconds: int = 3600) -> str:
    dt = datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(
        seconds=seconds
    )
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def _make_json_response(status_code: int, payload: Any) -> requests.Response:
    resp = requests.Response()
    resp.status_code = status_code
    import json

    resp._content = json.dumps(payload).encode("utf-8")
    resp.headers["Content-Type"] = "application/json"
    return resp


def _make_raw_response(status_code: int, content: bytes) -> requests.Response:
    resp = requests.Response()
    resp.status_code = status_code
    resp._content = content
    resp.headers["Content-Type"] = "application/json"
    return resp


class TestDeriveCabDirectoryUri:
    """Tests for derive_cab_directory_uri."""

    @pytest.mark.parametrize(
        ("raw_uri", "expected"),
        [
            ("gs://my-bucket", "gs://my-bucket"),
            ("gs://my-bucket/", "gs://my-bucket"),
            ("gs://my-bucket/file.txt", "gs://my-bucket/file.txt"),
            ("gs://my-bucket/MLmodel", "gs://my-bucket"),
            ("gs://my-bucket/path/to/file.txt", "gs://my-bucket/path/to/file.txt"),
            ("gs://my-bucket/path/to/MLmodel", "gs://my-bucket/path/to"),
            ("gs://my-bucket/path/to/MLmodel/", "gs://my-bucket/path/to/MLmodel"),
            ("gs://my-bucket/path/to/dir/", "gs://my-bucket/path/to/dir"),
            ("gs://my-bucket/path/to/dir", "gs://my-bucket/path/to/dir"),
            ("gs://my-bucket/ns/release-v1.0", "gs://my-bucket/ns/release-v1.0"),
            (
                "gs://my-bucket/runs/run123/artifacts/sub/dir",
                "gs://my-bucket/runs/run123/artifacts",
            ),
            (
                "gs://my-bucket/models/m456/artifacts/model/MLmodel",
                "gs://my-bucket/models/m456/artifacts",
            ),
            (
                "gs://my-bucket/0123456789abcdef0123456789abcdef/artifacts/sub",
                "gs://my-bucket/0123456789abcdef0123456789abcdef/artifacts",
            ),
            (
                "gs://my-bucket/exp/run/artifacts/model/MLmodel",
                "gs://my-bucket/exp/run/artifacts",
            ),
            (
                "gs://my-bucket/exp/run/artifacts/sub/dir",
                "gs://my-bucket/exp/run/artifacts",
            ),
            (
                "gs://my-bucket/exp/run/artifacts",
                "gs://my-bucket/exp/run/artifacts",
            ),
            (
                "gs://0123456789abcdef0123456789abcdef/team-a/data",
                "gs://0123456789abcdef0123456789abcdef/team-a/data",
            ),
            (
                "gs://0123456789abcdef0123456789abcdef/artifacts/team-a/data",
                "gs://0123456789abcdef0123456789abcdef/artifacts",
            ),
        ],
    )
    def test_valid_uris(self, raw_uri: str, expected: str) -> None:
        assert derive_cab_directory_uri(raw_uri) == expected

    @pytest.mark.parametrize(
        "invalid_uri",
        [
            "",
            "   ",
            "s3://my-bucket/path",
            "https://storage.googleapis.com/my-bucket/path",
            "gs://",
            "gs:///missing-bucket",
            "gs://my-bucket/path//extra",
            "gs://my-bucket/path/../secret",
            "gs://my-bucket/path/./current",
            "gs://my-bucket/path?query=1",
            "gs://my-bucket/path#fragment",
            "gs://my-bucket/path'with'quotes",
            "gs://my-bucket/path\nwith\nnewline",
        ],
    )
    def test_invalid_uris_raise(self, invalid_uri: str) -> None:
        with pytest.raises(MlflowException) as exc_info:
            derive_cab_directory_uri(invalid_uri)
        assert exc_info.value.error_code == ErrorCode.Name(INVALID_PARAMETER_VALUE)


class TestCabTokenCredentialsRpcs:
    """Tests for all 4 STS CAB RPCs and payload validation."""

    def test_namespace_read_and_read_write_rpcs(self) -> None:
        session = unittest.mock.MagicMock(spec=requests.Session)
        session.post.side_effect = [
            _make_json_response(
                200,
                {
                    "credentials": {
                        "accessToken": "ro-token-secret",
                        "expireTime": _future_rfc3339(3600),
                    }
                },
            ),
            _make_json_response(
                200,
                {
                    "credentials": {
                        "accessToken": "rw-token-secret",
                        "expireTime": _future_rfc3339(3600),
                    }
                },
            ),
        ]

        creds = CabTokenCredentials(
            resource_url=f"{_NAMESPACE_URL}/",
            resource_kind=CabResourceKind.NAMESPACE,
            access_mode=CabAccessMode.READ_ONLY,
            artifact_uri="gs://my-bucket/exp/run/artifacts",
            session=session,
            auth=unittest.mock.MagicMock(),
        )
        creds.refresh()
        assert creds.token == "ro-token-secret"
        assert session.post.call_args_list[0].args[0] == (
            f"{_NAMESPACE_URL}:issueReadArtifactToken"
        )
        assert session.post.call_args_list[0].kwargs["json"] == {
            "artifactUri": "gs://my-bucket/exp/run/artifacts"
        }
        assert session.post.call_args_list[0].kwargs.get("allow_redirects") is False

        creds.elevate_to_read_write()
        assert creds.access_mode == CabAccessMode.READ_WRITE
        assert creds.token is None
        creds.refresh()
        assert creds.token == "rw-token-secret"
        assert session.post.call_args_list[1].args[0] == (
            f"{_NAMESPACE_URL}:issueReadWriteArtifactToken"
        )
        assert session.post.call_args_list[1].kwargs["json"] == {
            "artifactUri": "gs://my-bucket/exp/run/artifacts"
        }
        assert session.post.call_args_list[1].kwargs.get("allow_redirects") is False

    @pytest.mark.parametrize(
        ("kwargs", "expected_payload"),
        [
            (
                {"artifact_uri": "gs://my-bucket/p"},
                {"artifactUri": "gs://my-bucket/p"},
            ),
            (
                {"artifact_uri": "gs://my-bucket/p", "artifact_path": "sub/dir"},
                {"artifactUri": "gs://my-bucket/p", "artifactPath": "sub/dir"},
            ),
            (
                {"run_id": "run-123"},
                {"runId": "run-123"},
            ),
            (
                {"run_id": "run-123", "artifact_path": "model"},
                {"runId": "run-123", "artifactPath": "model"},
            ),
            (
                {"model_id": "m-456"},
                {"modelId": "m-456"},
            ),
            (
                {"model_id": "m-456", "artifact_path": "weights"},
                {"modelId": "m-456", "artifactPath": "weights"},
            ),
        ],
    )
    def test_namespace_payload_oneof_and_artifact_path(
        self,
        kwargs: dict[str, Any],
        expected_payload: dict[str, str],
    ) -> None:
        session = unittest.mock.MagicMock(spec=requests.Session)
        session.post.return_value = _make_json_response(
            200,
            {
                "credentials": {
                    "accessToken": "tok",
                    "expireTime": _future_rfc3339(3600),
                }
            },
        )
        creds = CabTokenCredentials(
            resource_url=_NAMESPACE_URL,
            resource_kind=CabResourceKind.NAMESPACE,
            session=session,
            auth=unittest.mock.MagicMock(),
            **kwargs,
        )
        creds.refresh()
        assert session.post.call_args.kwargs["json"] == expected_payload

    def test_namespace_rejects_multiple_oneof_fields(self) -> None:
        with pytest.raises(MlflowException) as exc_info:
            CabTokenCredentials(
                resource_url=_NAMESPACE_URL,
                resource_kind=CabResourceKind.NAMESPACE,
                artifact_uri="gs://my-bucket/p",
                run_id="run-1",
            )
        assert exc_info.value.error_code == ErrorCode.Name(INVALID_PARAMETER_VALUE)

        with pytest.raises(MlflowException):
            CabTokenCredentials(
                resource_url=_NAMESPACE_URL,
                resource_kind=CabResourceKind.NAMESPACE,
                run_id="run-1",
                model_id="m-1",
            )

    @pytest.mark.parametrize(
        "bad_path",
        [
            "../secret",
            "a/../b",
            "a/./b",
            "/absolute",
            "a//b",
            "path'with'quote",
            "path\nnewline",
        ],
    )
    def test_namespace_rejects_relative_traversal_artifact_path(
        self, bad_path: str
    ) -> None:
        with pytest.raises(MlflowException) as exc_info:
            CabTokenCredentials(
                resource_url=_NAMESPACE_URL,
                resource_kind=CabResourceKind.NAMESPACE,
                run_id="run-1",
                artifact_path=bad_path,
            )
        assert exc_info.value.error_code == ErrorCode.Name(INVALID_PARAMETER_VALUE)

    @pytest.mark.parametrize(
        "valid_path",
        [".hidden", ".mlflow", "v1..v2", "release-v1.0", "sub/dir", "model"],
    )
    def test_namespace_allows_valid_relative_subpaths(self, valid_path: str) -> None:
        creds = CabTokenCredentials(
            resource_url=_NAMESPACE_URL,
            resource_kind=CabResourceKind.NAMESPACE,
            run_id="run-1",
            artifact_path=valid_path,
        )
        assert creds._artifact_path == valid_path.rstrip("/")

    def test_registered_model_read_and_read_write_rpcs(self) -> None:
        session = unittest.mock.MagicMock(spec=requests.Session)
        session.post.return_value = _make_json_response(
            200,
            {
                "credentials": {
                    "access_token": "reg-ro-tok",
                    "expire_time": _future_rfc3339(3600),
                }
            },
        )

        model_url = f"{_REGISTRY_URL}/registeredModels/12345"
        creds = CabTokenCredentials(
            resource_url=model_url,
            resource_kind=CabResourceKind.REGISTERED_MODEL,
            access_mode=CabAccessMode.READ_ONLY,
            artifact_uri="gs://my-bucket/models/12345/artifacts",
            session=session,
            auth=unittest.mock.MagicMock(),
        )
        creds.refresh()
        assert creds.token == "reg-ro-tok"
        assert session.post.call_args_list[0].args[0] == (
            f"{model_url}:issueReadArtifactToken"
        )
        assert session.post.call_args_list[0].kwargs["json"] == {
            "artifactUri": "gs://my-bucket/models/12345/artifacts"
        }

        with pytest.raises(MlflowException, match="READ_WRITE"):
            creds.elevate_to_read_write()
        with pytest.raises(MlflowException, match="READ_ONLY"):
            CabTokenCredentials(
                resource_url=model_url,
                resource_kind=CabResourceKind.REGISTERED_MODEL,
                access_mode=CabAccessMode.READ_WRITE,
                artifact_uri="gs://my-bucket/models/12345/artifacts",
                session=session,
                auth=unittest.mock.MagicMock(),
            )

    @pytest.mark.parametrize(
        "forbidden_kwargs",
        [
            {"run_id": "run-1"},
            {"model_id": "model-1"},
            {"artifact_path": "sub"},
        ],
    )
    def test_registered_model_strict_artifact_uri_only(
        self, forbidden_kwargs: dict[str, Any]
    ) -> None:
        with pytest.raises(MlflowException) as exc_info:
            CabTokenCredentials(
                resource_url=f"{_REGISTRY_URL}/registeredModels/12345",
                resource_kind=CabResourceKind.REGISTERED_MODEL,
                artifact_uri="gs://my-bucket/p",
                **forbidden_kwargs,
            )
        assert exc_info.value.error_code == ErrorCode.Name(INVALID_PARAMETER_VALUE)

    def test_empty_resource_url_raises(self) -> None:
        with pytest.raises(MlflowException) as exc_info:
            CabTokenCredentials(resource_url="")
        assert exc_info.value.error_code == ErrorCode.Name(INVALID_PARAMETER_VALUE)

    @pytest.mark.parametrize(
        "malformed_payload",
        [
            {},
            {"credentials": "not-a-dict"},
            {"credentials": {"expireTime": _future_rfc3339(3600)}},
            {"credentials": {"accessToken": "tok"}},
        ],
    )
    def test_malformed_response_raises_internal_error(
        self, malformed_payload: dict[str, Any]
    ) -> None:
        session = unittest.mock.MagicMock(spec=requests.Session)
        session.post.return_value = _make_json_response(200, malformed_payload)
        creds = CabTokenCredentials(
            resource_url=_NAMESPACE_URL,
            session=session,
            auth=unittest.mock.MagicMock(),
        )
        with pytest.raises(MlflowException) as exc_info:
            creds.refresh()
        assert exc_info.value.error_code == ErrorCode.Name(INTERNAL_ERROR)

    @pytest.mark.parametrize("body", [[], ["credentials"], "token", 42, None])
    def test_non_object_response_raises_internal_error(self, body: Any) -> None:
        session = unittest.mock.MagicMock(spec=requests.Session)
        session.post.return_value = _make_json_response(200, body)
        creds = CabTokenCredentials(
            resource_url=_NAMESPACE_URL,
            session=session,
            auth=unittest.mock.MagicMock(),
        )
        with pytest.raises(MlflowException) as exc_info:
            creds.refresh()
        assert exc_info.value.error_code == ErrorCode.Name(INTERNAL_ERROR)

    def test_non_json_response_raises_internal_error(self) -> None:
        session = unittest.mock.MagicMock(spec=requests.Session)
        session.post.return_value = _make_raw_response(200, b"not-json")
        creds = CabTokenCredentials(
            resource_url=_NAMESPACE_URL,
            session=session,
            auth=unittest.mock.MagicMock(),
        )
        with pytest.raises(MlflowException) as exc_info:
            creds.refresh()
        assert exc_info.value.error_code == ErrorCode.Name(INTERNAL_ERROR)
        assert "malformed JSON payload" in str(exc_info.value)

    def test_aip193_error_propagates(self) -> None:
        session = unittest.mock.MagicMock(spec=requests.Session)
        session.post.return_value = _make_json_response(
            400,
            {
                "error": {
                    "code": 400,
                    "status": "FAILED_PRECONDITION",
                    "message": "Cannot issue READ_WRITE credentials for finalized run",
                }
            },
        )
        creds = CabTokenCredentials(
            resource_url=_NAMESPACE_URL,
            access_mode=CabAccessMode.READ_WRITE,
            session=session,
            auth=unittest.mock.MagicMock(),
        )
        with pytest.raises(GcpMlflowApiError) as exc_info:
            creds.refresh()
        assert exc_info.value.http_status == 400
        assert exc_info.value.gcp_status == "FAILED_PRECONDITION"


class TestCabTokenCredentialsLifecycle:
    """Tests for expiry buffer, concurrency, finalization, and repr."""

    def test_300s_early_expiry_buffer_and_token_state(self) -> None:
        session = unittest.mock.MagicMock(spec=requests.Session)
        creds = CabTokenCredentials(
            resource_url=_NAMESPACE_URL,
            session=session,
            auth=unittest.mock.MagicMock(),
        )

        # Uninitialized
        assert creds.expired is True
        assert creds.valid is False
        assert creds.token_state == google.auth.credentials.TokenState.INVALID

        now = datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None)

        # Fresh (> 300s remaining)
        creds.token = "tok"
        creds.expiry = now + datetime.timedelta(seconds=600)
        assert creds.expired is False
        assert creds.valid is True
        assert creds.token_state == google.auth.credentials.TokenState.FRESH

        # Within 300s buffer -> expired=True, valid=False, token_state=STALE
        creds.expiry = now + datetime.timedelta(seconds=250)
        assert creds.expired is True
        assert creds.valid is False
        assert creds.token_state == google.auth.credentials.TokenState.STALE

        # Past expiry -> expired=True, valid=False, token_state=INVALID
        creds.expiry = now - datetime.timedelta(seconds=5)
        assert creds.expired is True
        assert creds.valid is False
        assert creds.token_state == google.auth.credentials.TokenState.INVALID

        # No expiry -> INVALID
        creds.expiry = None
        assert creds.expired is True
        assert creds.valid is False
        assert creds.token_state == google.auth.credentials.TokenState.INVALID

    def test_elevate_to_read_write_idempotent_when_already_rw(self) -> None:
        session = unittest.mock.MagicMock(spec=requests.Session)
        creds = CabTokenCredentials(
            resource_url=_NAMESPACE_URL,
            access_mode=CabAccessMode.READ_WRITE,
            session=session,
            auth=unittest.mock.MagicMock(),
        )
        creds.token = "existing-rw-token"
        creds.expiry = datetime.datetime.now(datetime.timezone.utc).replace(
            tzinfo=None
        ) + datetime.timedelta(seconds=1000)

        creds.elevate_to_read_write()
        assert creds.token == "existing-rw-token"
        assert creds.valid is True

    def test_single_flight_thread_safe_refresh(self) -> None:
        call_count = 0
        count_lock = threading.Lock()
        in_post = threading.Event()
        release_post = threading.Event()

        def slow_post(*args: Any, **kwargs: Any) -> requests.Response:
            nonlocal call_count
            with count_lock:
                call_count += 1
            in_post.set()
            release_post.wait(timeout=5)
            return _make_json_response(
                200,
                {
                    "credentials": {
                        "accessToken": "shared-single-flight-token",
                        "expireTime": _future_rfc3339(3600),
                    }
                },
            )

        session = unittest.mock.MagicMock(spec=requests.Session)
        session.post.side_effect = slow_post

        creds = CabTokenCredentials(
            resource_url=_NAMESPACE_URL,
            session=session,
            auth=unittest.mock.MagicMock(),
        )

        num_threads = 12
        start_barrier = threading.Barrier(num_threads)
        all_entered = threading.Event()
        entered_count = 0
        entered_lock = threading.Lock()
        errors: list[BaseException] = []

        def worker() -> None:
            nonlocal entered_count
            try:
                start_barrier.wait(timeout=5)
                with entered_lock:
                    entered_count += 1
                    if entered_count == num_threads:
                        all_entered.set()
                creds.refresh()
            except BaseException as exc:  # pragma: no cover
                errors.append(exc)

        threads = [threading.Thread(target=worker) for _ in range(num_threads)]
        for t in threads:
            t.start()

        assert in_post.wait(timeout=5)
        assert all_entered.wait(timeout=5)
        release_post.set()

        for t in threads:
            t.join(timeout=5)

        assert not errors
        assert call_count == 1
        assert creds.token == "shared-single-flight-token"

    def test_close_and_weakref_finalize(self) -> None:
        owned_session = unittest.mock.MagicMock(spec=requests.Session)
        with unittest.mock.patch(
            "google_cloud_managed_mlflow.cab_credentials.create_secure_session",
            return_value=owned_session,
        ):
            creds = CabTokenCredentials(
                resource_url=_NAMESPACE_URL,
                auth=unittest.mock.MagicMock(),
            )
            creds.close()
            owned_session.close.assert_called_once()

        # Custom session is NOT closed by CabTokenCredentials.close()
        custom_session = unittest.mock.MagicMock(spec=requests.Session)
        creds_custom = CabTokenCredentials(
            resource_url=_NAMESPACE_URL,
            session=custom_session,
            auth=unittest.mock.MagicMock(),
        )
        creds_custom.close()
        custom_session.close.assert_not_called()

        # weakref.finalize closes owned session on GC
        gc_session = unittest.mock.MagicMock(spec=requests.Session)
        with unittest.mock.patch(
            "google_cloud_managed_mlflow.cab_credentials.create_secure_session",
            return_value=gc_session,
        ):
            creds_gc: CabTokenCredentials | None = CabTokenCredentials(
                resource_url=_NAMESPACE_URL,
                auth=unittest.mock.MagicMock(),
            )
            del creds_gc
            gc.collect()
            gc_session.close.assert_called_once()

    def test_redacted_repr(self) -> None:
        session = unittest.mock.MagicMock(spec=requests.Session)
        creds = CabTokenCredentials(
            resource_url=_NAMESPACE_URL,
            resource_kind=CabResourceKind.NAMESPACE,
            access_mode=CabAccessMode.READ_ONLY,
            session=session,
            auth=unittest.mock.MagicMock(),
        )
        assert "token=unset" in repr(creds)
        assert _NAMESPACE_URL not in repr(creds)

        creds.token = "super-secret-cab-token-value"
        rendered = repr(creds)
        assert "super-secret-cab-token-value" not in rendered
        assert _NAMESPACE_URL not in rendered
        assert "token=set" in rendered
        assert "resource_kind=NAMESPACE" in rendered
        assert "access_mode=READ_ONLY" in rendered

    def test_short_lived_token_clamps_early_expiry_buffer(self) -> None:
        session = unittest.mock.MagicMock(spec=requests.Session)
        session.post.return_value = _make_json_response(
            200,
            {
                "credentials": {
                    "accessToken": "short-tok",
                    "expireTime": _future_rfc3339(120),
                }
            },
        )
        creds = CabTokenCredentials(
            resource_url=_NAMESPACE_URL,
            session=session,
            auth=unittest.mock.MagicMock(),
        )
        creds.refresh()
        assert creds._effective_buffer <= datetime.timedelta(seconds=60)
        assert creds.valid is True
        assert creds.expired is False
        assert creds.token_state == google.auth.credentials.TokenState.FRESH

    def test_refresh_forces_renewal_when_token_valid_for_401_recovery(
        self,
    ) -> None:
        session = unittest.mock.MagicMock(spec=requests.Session)
        session.post.side_effect = [
            _make_json_response(
                200,
                {
                    "credentials": {
                        "accessToken": "tok-1",
                        "expireTime": _future_rfc3339(3600),
                    }
                },
            ),
            _make_json_response(
                200,
                {
                    "credentials": {
                        "accessToken": "tok-2",
                        "expireTime": _future_rfc3339(3600),
                    }
                },
            ),
        ]
        creds = CabTokenCredentials(
            resource_url=_NAMESPACE_URL,
            session=session,
            auth=unittest.mock.MagicMock(),
        )
        creds.refresh()
        assert creds.token == "tok-1"
        assert creds.valid is True
        creds.refresh()
        assert creds.token == "tok-2"
        assert session.post.call_count == 2

    def test_with_access_mode_creates_independent_instance(self) -> None:
        custom_session = unittest.mock.MagicMock(spec=requests.Session)
        ro_creds = CabTokenCredentials(
            resource_url=_NAMESPACE_URL,
            session=custom_session,
            auth=unittest.mock.MagicMock(),
        )
        assert ro_creds._custom_session is True
        rw_creds = ro_creds.with_access_mode(CabAccessMode.READ_WRITE)
        assert rw_creds.access_mode == CabAccessMode.READ_WRITE
        assert ro_creds.access_mode == CabAccessMode.READ_ONLY
        assert rw_creds._session is ro_creds._session
        assert rw_creds._custom_session is True

        # Internally created session:
        ro_internal = CabTokenCredentials(
            resource_url=_NAMESPACE_URL,
            auth=unittest.mock.MagicMock(),
        )
        assert ro_internal._custom_session is False
        rw_internal = ro_internal.with_access_mode(CabAccessMode.READ_WRITE)
        assert rw_internal.access_mode == CabAccessMode.READ_WRITE
        assert ro_internal.access_mode == CabAccessMode.READ_ONLY
        assert rw_internal._custom_session is False
        assert rw_internal._session is not ro_internal._session

    def test_resource_url_and_identifier_validation(self) -> None:
        with pytest.raises(MlflowException) as exc_info:
            CabTokenCredentials(
                "https://example.com/v1/projects/p/locations/l/namespaces/n"
            )
        assert exc_info.value.error_code == ErrorCode.Name(INVALID_PARAMETER_VALUE)

        with pytest.raises(MlflowException) as exc_info:
            CabTokenCredentials(_REGISTRY_URL, resource_kind=CabResourceKind.NAMESPACE)
        assert exc_info.value.error_code == ErrorCode.Name(INVALID_PARAMETER_VALUE)

        with pytest.raises(MlflowException) as exc_info:
            CabTokenCredentials(
                _NAMESPACE_URL, resource_kind=CabResourceKind.REGISTERED_MODEL
            )
        assert exc_info.value.error_code == ErrorCode.Name(INVALID_PARAMETER_VALUE)

        for bad_run in ["run/1", "../run", "a" * 65]:
            with pytest.raises(MlflowException) as exc_info:
                CabTokenCredentials(_NAMESPACE_URL, run_id=bad_run)
            assert exc_info.value.error_code == ErrorCode.Name(INVALID_PARAMETER_VALUE)

        for bad_model in ["model/1", "a" * 65]:
            with pytest.raises(MlflowException) as exc_info:
                CabTokenCredentials(_NAMESPACE_URL, model_id=bad_model)
            assert exc_info.value.error_code == ErrorCode.Name(INVALID_PARAMETER_VALUE)

    def test_redirect_rejected_in_refresh_and_model_search(self) -> None:
        session = unittest.mock.MagicMock(spec=requests.Session)
        resp302 = requests.Response()
        resp302.status_code = 302
        resp302.headers["Location"] = "https://evil.com"
        session.post.return_value = resp302
        creds = CabTokenCredentials(
            resource_url=_NAMESPACE_URL,
            session=session,
            auth=unittest.mock.MagicMock(),
        )
        with pytest.raises(MlflowException) as exc_info:
            creds.refresh()
        assert exc_info.value.error_code == ErrorCode.Name(INVALID_PARAMETER_VALUE)

        session.get.return_value = resp302
        with pytest.raises(MlflowException) as exc_info:
            _resolve_registered_model_resource_name(
                session=session,
                registry_resource_url=_REGISTRY_URL,
                model_name="model",
                auth=unittest.mock.MagicMock(),
            )
        assert exc_info.value.error_code == ErrorCode.Name(INVALID_PARAMETER_VALUE)


class TestResolveRegisteredModelResourceName:
    """Tests for resolve_registered_model_resource_name."""

    @pytest.mark.parametrize(
        ("list_key", "name_key"),
        [
            ("registeredModels", "displayName"),
            ("registered_models", "display_name"),
        ],
    )
    def test_resolves_renamed_model_by_display_name(
        self, list_key: str, name_key: str
    ) -> None:
        session = unittest.mock.MagicMock(spec=requests.Session)
        canonical = (
            "projects/my-proj/locations/us-central1"
            "/registries/my-reg/registeredModels/98765"
        )
        other = (
            "projects/my-proj/locations/us-central1"
            "/registries/my-reg/registeredModels/11111"
        )
        session.get.return_value = _make_json_response(
            200,
            {
                list_key: [
                    {
                        "name": other,
                        name_key: "other-model",
                    },
                    {
                        "name": canonical,
                        name_key: "renamed-model",
                    },
                ]
            },
        )

        resolved = _resolve_registered_model_resource_name(
            session=session,
            registry_resource_url=f"{_REGISTRY_URL}/",
            model_name="renamed-model",
            auth=unittest.mock.MagicMock(),
        )
        assert resolved == canonical
        called_url = session.get.call_args.args[0]
        assert called_url.startswith(
            f"{_REGISTRY_URL}/api/2.0/mlflow/registered-models/search?filter="
        )
        assert urllib.parse.quote("name = 'renamed-model'") in called_url

    def test_camel_case_list_key_takes_precedence(self) -> None:
        """When registeredModels is present, it takes precedence even if empty."""
        session = unittest.mock.MagicMock(spec=requests.Session)
        session.get.return_value = _make_json_response(
            200,
            {
                "registeredModels": [],
                "registered_models": [
                    {
                        "name": "projects/p/locations/l/registries/r/models/1",
                        "display_name": "target-model",
                    }
                ],
            },
        )
        with pytest.raises(MlflowException) as exc_info:
            _resolve_registered_model_resource_name(
                session=session,
                registry_resource_url=_REGISTRY_URL,
                model_name="target-model",
                auth=unittest.mock.MagicMock(),
            )
        assert exc_info.value.error_code == ErrorCode.Name(RESOURCE_DOES_NOT_EXIST)

    @pytest.mark.parametrize(
        "payload",
        [
            {
                "registeredModels": [
                    {
                        "name": "projects/p/locations/l/registries/r/models/1",
                        "displayName": "other-model",
                    }
                ]
            },
            {
                "registered_models": [
                    {
                        "name": "projects/p/locations/l/registries/r/models/1",
                        "display_name": "other-model",
                    }
                ]
            },
            {
                "registeredModels": [
                    {
                        "name": "projects/p/locations/l/registries/r/models/1",
                    }
                ]
            },
            {
                "registeredModels": [
                    "not-a-dict",
                    {
                        "name": "projects/p/locations/l/registries/r/models/1",
                        "displayName": "other-model",
                    },
                ]
            },
        ],
    )
    def test_never_returns_non_matching_model(self, payload: dict[str, Any]) -> None:
        session = unittest.mock.MagicMock(spec=requests.Session)
        session.get.return_value = _make_json_response(200, payload)
        with pytest.raises(MlflowException) as exc_info:
            _resolve_registered_model_resource_name(
                session=session,
                registry_resource_url=_REGISTRY_URL,
                model_name="target-model",
                auth=unittest.mock.MagicMock(),
            )
        assert exc_info.value.error_code == ErrorCode.Name(RESOURCE_DOES_NOT_EXIST)

    @pytest.mark.parametrize(
        "payload",
        [
            ["top-level-list"],
            {"registeredModels": 5},
            {"registered_models": "not-a-list"},
        ],
    )
    def test_malformed_response_raises_internal_error(self, payload: Any) -> None:
        session = unittest.mock.MagicMock(spec=requests.Session)
        session.get.return_value = _make_json_response(200, payload)
        with pytest.raises(MlflowException) as exc_info:
            _resolve_registered_model_resource_name(
                session=session,
                registry_resource_url=_REGISTRY_URL,
                model_name="target-model",
                auth=unittest.mock.MagicMock(),
            )
        assert exc_info.value.error_code == ErrorCode.Name(INTERNAL_ERROR)

    def test_filter_escaping(self) -> None:
        session = unittest.mock.MagicMock(spec=requests.Session)
        canonical = (
            "projects/my-proj/locations/us-central1"
            "/registries/my-reg/registeredModels/42"
        )
        session.get.return_value = _make_json_response(
            200,
            {"registeredModels": [{"name": canonical, "displayName": "model's\\name"}]},
        )
        resolved = _resolve_registered_model_resource_name(
            session=session,
            registry_resource_url=_REGISTRY_URL,
            model_name="model's\\name",
            auth=unittest.mock.MagicMock(),
        )
        assert resolved == canonical
        called_url = session.get.call_args.args[0]
        assert urllib.parse.quote("name = 'model\\'s\\\\name'", safe="") in called_url
        assert session.get.call_args.kwargs.get("allow_redirects") is False

    def test_empty_model_name_raises_invalid_parameter(self) -> None:
        session = unittest.mock.MagicMock(spec=requests.Session)
        with pytest.raises(MlflowException) as exc_info:
            _resolve_registered_model_resource_name(
                session=session,
                registry_resource_url=_REGISTRY_URL,
                model_name="",
            )
        assert exc_info.value.error_code == ErrorCode.Name(INVALID_PARAMETER_VALUE)

    def test_not_found_raises_resource_does_not_exist(self) -> None:
        session = unittest.mock.MagicMock(spec=requests.Session)
        session.get.return_value = _make_json_response(200, {"registered_models": []})
        with pytest.raises(MlflowException) as exc_info:
            _resolve_registered_model_resource_name(
                session=session,
                registry_resource_url=_REGISTRY_URL,
                model_name="missing-model",
                auth=unittest.mock.MagicMock(),
            )
        assert exc_info.value.error_code == ErrorCode.Name(RESOURCE_DOES_NOT_EXIST)

    @pytest.mark.parametrize(
        "entry",
        [
            {"displayName": "target-model"},
            {"displayName": "target-model", "name": ""},
            {"displayName": "target-model", "name": 12345},
            {"displayName": "target-model", "name": None},
        ],
    )
    def test_matched_entry_invalid_name_raises_internal_error(
        self, entry: dict[str, Any]
    ) -> None:
        session = unittest.mock.MagicMock(spec=requests.Session)
        session.get.return_value = _make_json_response(
            200, {"registeredModels": [entry]}
        )
        with pytest.raises(MlflowException) as exc_info:
            _resolve_registered_model_resource_name(
                session=session,
                registry_resource_url=_REGISTRY_URL,
                model_name="target-model",
                auth=unittest.mock.MagicMock(),
            )
        assert exc_info.value.error_code == ErrorCode.Name(INTERNAL_ERROR)

    def test_display_name_casing_precedence_at_entry_level(self) -> None:
        """When displayName is present but different, display_name must not match."""
        session = unittest.mock.MagicMock(spec=requests.Session)
        session.get.return_value = _make_json_response(
            200,
            {
                "registeredModels": [
                    {
                        "name": "projects/p/locations/l/registries/r/models/1",
                        "displayName": "other-model",
                        "display_name": "target-model",
                    }
                ]
            },
        )
        with pytest.raises(MlflowException) as exc_info:
            _resolve_registered_model_resource_name(
                session=session,
                registry_resource_url=_REGISTRY_URL,
                model_name="target-model",
                auth=unittest.mock.MagicMock(),
            )
        assert exc_info.value.error_code == ErrorCode.Name(RESOURCE_DOES_NOT_EXIST)

    def test_resolve_registered_model_pagination(self) -> None:
        canonical = (
            "projects/my-proj/locations/us-central1"
            "/registries/my-reg/registeredModels/m-target"
        )
        session = unittest.mock.MagicMock(spec=requests.Session)
        session.get.side_effect = [
            _make_json_response(
                200,
                {
                    "registeredModels": [{"displayName": "other"}],
                    "nextPageToken": "page-2",
                },
            ),
            _make_json_response(
                200,
                {
                    "registeredModels": [
                        {"name": canonical, "displayName": "target-model"}
                    ]
                },
            ),
        ]
        res = _resolve_registered_model_resource_name(
            session=session,
            registry_resource_url=_REGISTRY_URL,
            model_name="target-model",
            auth=unittest.mock.MagicMock(),
        )
        assert res == canonical
        assert session.get.call_count == 2
        assert "page_token=page-2" in session.get.call_args_list[1].args[0]

    def test_resolve_registered_model_matches_by_canonical_name(self) -> None:
        canonical = (
            "projects/my-proj/locations/us-central1"
            "/registries/my-reg/registeredModels/m-target"
        )
        session_name = unittest.mock.MagicMock(spec=requests.Session)
        session_name.get.return_value = _make_json_response(
            200,
            {
                "registeredModels": [
                    {"name": canonical, "displayName": "different-display"}
                ]
            },
        )
        res_by_name = _resolve_registered_model_resource_name(
            session=session_name,
            registry_resource_url=_REGISTRY_URL,
            model_name=canonical,
            auth=unittest.mock.MagicMock(),
        )
        assert res_by_name == canonical

    def test_resolve_registered_model_rejects_malformed_resource_name(
        self,
    ) -> None:
        session_bad = unittest.mock.MagicMock(spec=requests.Session)
        session_bad.get.return_value = _make_json_response(
            200,
            {
                "registeredModels": [
                    {"name": "malformed/resource/name", "displayName": "target"}
                ]
            },
        )
        with pytest.raises(MlflowException) as exc_info:
            _resolve_registered_model_resource_name(
                session=session_bad,
                registry_resource_url=_REGISTRY_URL,
                model_name="target",
                auth=unittest.mock.MagicMock(),
            )
        assert exc_info.value.error_code == ErrorCode.Name(INTERNAL_ERROR)

    def test_resolve_registered_model_non_json_raises_internal_error(self) -> None:
        session = unittest.mock.MagicMock(spec=requests.Session)
        session.get.return_value = _make_raw_response(200, b"not-json")
        with pytest.raises(MlflowException) as exc_info:
            _resolve_registered_model_resource_name(
                session=session,
                registry_resource_url=_REGISTRY_URL,
                model_name="my-model",
            )
        assert exc_info.value.error_code == ErrorCode.Name(INTERNAL_ERROR)
        assert "malformed JSON payload" in str(exc_info.value)
