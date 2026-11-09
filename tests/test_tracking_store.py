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

"""Tests for GcpTrackingRestStore and tracking store registration."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import unittest.mock
from collections.abc import Generator
from pathlib import Path
from typing import Any

import mlflow
import pytest
import requests
from mlflow.exceptions import MlflowException
from mlflow.protos import service_pb2
from mlflow.protos.databricks_pb2 import (
    INVALID_PARAMETER_VALUE,
    PERMISSION_DENIED,
    RESOURCE_DOES_NOT_EXIST,
    UNAUTHENTICATED,
)
from mlflow.store.tracking.rest_store import RestStore

import google_cloud_managed_mlflow.tracking_store as ts_mod
from google_cloud_managed_mlflow.tracking_store import (
    GcpTrackingRestStore,
    _ensure_experiment_fields,
    _ensure_logged_model_info_fields,
    _ensure_run_info_fields,
    build_tracking_store,
    ensure_tracking_store_registered,
    normalize_tracking_response,
    reset_tracking_store_cache,
)
from google_cloud_managed_mlflow.uri import GcpMlflowApiError

_GCP_TRACKING_URI = (
    "https://mlflow.googleapis.com/v1/projects/my-project"
    "/locations/us-central1/namespaces/my-namespace"
)
_GCP_REGISTRY_URI = (
    "https://mlflow.googleapis.com/v1/projects/my-project"
    "/locations/us-central1/registries/my-registry"
)
_NON_GCP_URI = "https://custom-mlflow.example.com"


@pytest.fixture(autouse=True)
def _isolate_tracking_state(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    import mlflow.tracking._tracking_service.utils as tr_utils

    monkeypatch.setattr(tr_utils, "_tracking_uri", None, raising=False)


@pytest.fixture(autouse=True)
def restore_mlflow_request_session() -> Generator[None, None, None]:
    import mlflow.utils.request_utils as req_utils
    import mlflow.utils.rest_utils as rest_utils

    session_attr = "_cached_get_request_session"
    orig_req = getattr(req_utils, session_attr, None)
    orig_rest = getattr(rest_utils, session_attr, None)
    try:
        yield
    finally:
        if orig_req is not None:
            setattr(req_utils, session_attr, orig_req)
            if hasattr(orig_req, "cache_clear"):
                orig_req.cache_clear()
        if orig_rest is not None:
            setattr(rest_utils, session_attr, orig_rest)
            if hasattr(orig_rest, "cache_clear"):
                orig_rest.cache_clear()


def _make_mock_response(
    status_code: int = 200,
    json_data: dict[str, Any] | None = None,
    text: str | None = None,
) -> requests.Response:
    resp = requests.Response()
    resp.status_code = status_code
    if json_data is not None:
        resp._content = json.dumps(json_data).encode("utf-8")
        resp.headers["Content-Type"] = "application/json"
    elif text is not None:
        resp._content = text.encode("utf-8")
    else:
        resp._content = b"{}"
        resp.headers["Content-Type"] = "application/json"
    return resp


# ---------------------------------------------------------------------------
# Test Case 1: build_tracking_store
# ---------------------------------------------------------------------------


def test_build_tracking_store_gcp_uri() -> None:
    store = build_tracking_store(
        _GCP_TRACKING_URI, artifact_uri="gs://bucket/artifacts"
    )
    assert isinstance(store, GcpTrackingRestStore)
    assert store.store_uri == _GCP_TRACKING_URI
    assert store.artifact_uri == "gs://bucket/artifacts"
    assert (
        store.namespace_resource
        == "projects/my-project/locations/us-central1/namespaces/my-namespace"
    )
    assert store.base_api_url == "https://mlflow.googleapis.com/v1"
    assert store.endpoint_url == _GCP_TRACKING_URI


def test_build_tracking_store_non_gcp_uri_fallback() -> None:
    with unittest.mock.patch(
        "mlflow.tracking._tracking_service.utils._get_rest_store"
    ) as mock_get_rest:
        sentinel_store = unittest.mock.MagicMock(spec=RestStore)
        mock_get_rest.return_value = sentinel_store

        result = build_tracking_store(_NON_GCP_URI, artifact_uri="gs://test-bucket/a")
        assert result is sentinel_store
        mock_get_rest.assert_called_once_with(
            _NON_GCP_URI, artifact_uri="gs://test-bucket/a"
        )


# ---------------------------------------------------------------------------
# Test Case 2: URI validation
# ---------------------------------------------------------------------------


def test_tracking_store_resolves_from_mlflow_get_tracking_uri(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MLFLOW_TRACKING_URI", _GCP_TRACKING_URI)
    store = GcpTrackingRestStore()
    assert store.store_uri == _GCP_TRACKING_URI
    assert store.namespace_resource.endswith("namespaces/my-namespace")


def test_tracking_store_empty_uri_raises() -> None:
    with pytest.raises(MlflowException) as exc_info:
        GcpTrackingRestStore("")
    assert exc_info.value.error_code == "INVALID_PARAMETER_VALUE"


def test_tracking_store_non_namespace_uri_raises() -> None:
    with pytest.raises(MlflowException) as exc_info:
        GcpTrackingRestStore(_GCP_REGISTRY_URI)
    assert exc_info.value.error_code == "INVALID_PARAMETER_VALUE"
    assert "namespaces" in str(exc_info.value)


def test_tracking_store_malformed_uri_raises() -> None:
    with pytest.raises(MlflowException) as exc_info:
        GcpTrackingRestStore("https://mlflow.googleapis.com/v1/invalid/path")
    assert exc_info.value.error_code == "INVALID_PARAMETER_VALUE"


# ---------------------------------------------------------------------------
# Test Case 3: Host credentials
# ---------------------------------------------------------------------------


def test_tracking_store_host_creds() -> None:
    store = GcpTrackingRestStore(_GCP_TRACKING_URI)
    creds = store.get_host_creds()
    assert creds.host == _GCP_TRACKING_URI
    assert creds.auth == "google-cloud-auth"
    assert creds.ignore_tls_verification is False


def test_tracking_store_host_creds_untrusted_origin() -> None:
    untrusted_uri = "https://evil.com/v1/projects/p/locations/l/namespaces/ns"
    with pytest.raises(MlflowException) as exc_info:
        GcpTrackingRestStore(untrusted_uri)
    assert exc_info.value.error_code == "INVALID_PARAMETER_VALUE"


# ---------------------------------------------------------------------------
# Test Case 4: Response Normalization
# ---------------------------------------------------------------------------


def test_ensure_run_info_fields_mappings_and_defaults() -> None:
    info = {
        "runId": "run-001",
        "artifactLocation": "gs://bucket/experiments/1/runs/run-001/artifacts",
        "startTime": "2026-11-09T12:00:00.000Z",
    }
    normalized = _ensure_run_info_fields(info)
    assert normalized["run_id"] == "run-001"
    assert "runId" not in normalized
    assert (
        normalized["artifact_uri"] == "gs://bucket/experiments/1/runs/run-001/artifacts"
    )
    assert "artifactLocation" not in normalized
    assert normalized["lifecycle_stage"] == "active"
    assert "status" not in normalized
    assert normalized["user_id"] == ""
    assert "experiment_id" not in normalized
    assert isinstance(normalized["start_time"], int)
    assert normalized["start_time"] > 0


def test_ensure_run_info_run_id_mapping() -> None:
    info = {"runId": "run-123"}
    normalized = _ensure_run_info_fields(info)
    assert normalized["run_id"] == "run-123"
    assert "runId" not in normalized


def test_ensure_experiment_fields_mappings_and_id_extraction() -> None:
    exp = {
        "name": "projects/my-p/locations/us-c1/namespaces/ns1/experiments/exp-99",
        "artifactLocation": "gs://my-bucket/experiments/exp-99",
    }
    normalized = _ensure_experiment_fields(exp)
    assert normalized["experiment_id"] == "exp-99"
    assert normalized["artifact_location"] == "gs://my-bucket/experiments/exp-99"
    assert "artifactLocation" not in normalized
    assert normalized["lifecycle_stage"] == "active"

    exp2 = {
        "name": "exp-1",
        "experimentId": "123",
        "creationTime": "2026-11-10T10:01:24.783324138Z",
        "updateTime": "2026-11-10T10:05:00.000Z",
    }
    normalized2 = _ensure_experiment_fields(exp2)
    assert normalized2["experiment_id"] == "123"
    assert isinstance(normalized2["creation_time"], int)
    assert isinstance(normalized2["last_update_time"], int)
    assert normalized2["creation_time"] > 0
    assert normalized2["last_update_time"] > 0


def test_ensure_logged_model_info_fields() -> None:
    model_info = {
        "model_id": "model-1",
        "artifactLocation": "gs://bucket/models/m1",
    }
    normalized = _ensure_logged_model_info_fields(model_info)
    assert normalized["artifact_uri"] == "gs://bucket/models/m1"
    assert "artifactLocation" not in normalized

    model_info2 = {
        "model_id": "model-2",
        "artifactUri": "gs://bucket/models/m2",
    }
    normalized2 = _ensure_logged_model_info_fields(model_info2)
    assert normalized2["artifact_uri"] == "gs://bucket/models/m2"
    assert "artifactUri" not in normalized2


def test_normalize_create_and_get_run() -> None:
    # Flat run dict
    flat = {
        "run_id": "r1",
        "artifactLocation": "gs://test-bucket/a",
    }
    res = normalize_tracking_response(service_pb2.CreateRun, flat)
    assert "run" in res
    assert "info" in res["run"]
    assert res["run"]["info"]["run_id"] == "r1"
    assert res["run"]["info"]["artifact_uri"] == "gs://test-bucket/a"
    assert res["run"]["data"] == {}

    # Already wrapped
    wrapped = {
        "run": {
            "info": {"run_id": "r2", "artifact_location": "gs://test-bucket/a2"},
            "data": {},
        }
    }
    res2 = normalize_tracking_response(service_pb2.GetRun, wrapped)
    assert res2["run"]["info"]["artifact_uri"] == "gs://test-bucket/a2"


def test_normalize_update_run() -> None:
    flat = {"run_id": "r1", "artifactLocation": "gs://test-bucket/a"}
    res = normalize_tracking_response(service_pb2.UpdateRun, flat)
    assert "run_info" in res
    assert res["run_info"]["run_id"] == "r1"
    assert res["run_info"]["artifact_uri"] == "gs://test-bucket/a"

    wrapped = {"run_info": {"run_id": "r2", "artifact_location": "gs://test-bucket/a2"}}
    res2 = normalize_tracking_response(service_pb2.UpdateRun, wrapped)
    assert res2["run_info"]["artifact_uri"] == "gs://test-bucket/a2"

    camel_case_style = {
        "runInfo": {
            "runId": "r3",
            "runName": "run3",
            "status": "FINISHED",
            "startTime": "1000",
            "endTime": "2000",
            "artifactUri": "gs://bucket-3/artifacts",
        }
    }
    res3 = normalize_tracking_response(service_pb2.UpdateRun, camel_case_style)
    assert "run_info" in res3
    assert res3["run_info"]["run_id"] == "r3"
    assert res3["run_info"]["status"] == "FINISHED"
    assert res3["run_info"]["run_name"] == "run3"
    assert res3["run_info"]["start_time"] == 1000
    assert res3["run_info"]["end_time"] == 2000
    assert res3["run_info"]["artifact_uri"] == "gs://bucket-3/artifacts"


def test_normalize_get_experiment() -> None:
    flat = {
        "name": "projects/p/locations/l/namespaces/n/experiments/10",
        "artifactLocation": "gs://bucket-10/artifacts",
    }
    res = normalize_tracking_response(service_pb2.GetExperiment, flat)
    assert "experiment" in res
    assert res["experiment"]["experiment_id"] == "10"
    assert res["experiment"]["artifact_location"] == "gs://bucket-10/artifacts"


def test_normalize_create_experiment() -> None:
    assert normalize_tracking_response(
        service_pb2.CreateExperiment, {"experiment_id": "123"}
    ) == {"experiment_id": "123"}
    assert normalize_tracking_response(
        service_pb2.CreateExperiment, {"experimentId": "123"}
    ) == {"experiment_id": "123"}
    assert normalize_tracking_response(
        service_pb2.CreateExperiment, {"experiment": {"experiment_id": "456"}}
    ) == {"experiment_id": "456"}


def test_normalize_search_runs() -> None:
    data = {
        "runs": [
            {"run_id": "r1", "artifactLocation": "gs://test-bucket/r1"},
            {"info": {"run_id": "r2", "artifactLocation": "gs://test-bucket/r2"}},
        ],
        "nextPageToken": "token-1",
    }
    res = normalize_tracking_response(service_pb2.SearchRuns, data)
    assert len(res["runs"]) == 2
    assert res["runs"][0]["info"]["run_id"] == "r1"
    assert res["runs"][0]["info"]["artifact_uri"] == "gs://test-bucket/r1"
    assert res["runs"][1]["info"]["run_id"] == "r2"
    assert res["runs"][1]["info"]["artifact_uri"] == "gs://test-bucket/r2"
    assert res["next_page_token"] == "token-1"


def test_normalize_search_experiments() -> None:
    data = {
        "experiments": [
            {
                "name": "projects/p/locations/l/namespaces/n/experiments/exp-1",
                "artifactLocation": "gs://test-bucket/e1",
            },
            {
                "experiment": {
                    "experiment_id": "exp-2",
                    "artifactLocation": "gs://test-bucket/e2",
                }
            },
        ],
        "nextPageToken": "token-2",
    }
    res = normalize_tracking_response(service_pb2.SearchExperiments, data)
    assert len(res["experiments"]) == 2
    assert res["experiments"][0]["experiment_id"] == "exp-1"
    assert res["experiments"][0]["artifact_location"] == "gs://test-bucket/e1"
    assert res["experiments"][1]["experiment_id"] == "exp-2"
    assert res["experiments"][1]["artifact_location"] == "gs://test-bucket/e2"
    assert res["next_page_token"] == "token-2"


def test_normalize_logged_models() -> None:
    if not hasattr(service_pb2, "CreateLoggedModel"):
        pytest.skip("LoggedModel APIs require MLflow >= 3.0")
    # CreateLoggedModel
    flat = {"model_id": "m1", "artifactLocation": "gs://test-bucket/m1"}
    res = normalize_tracking_response(service_pb2.CreateLoggedModel, flat)
    assert "model" in res
    assert res["model"]["info"]["artifact_uri"] == "gs://test-bucket/m1"

    # SearchLoggedModels
    data = {
        "models": [{"model_id": "m1", "artifactLocation": "gs://test-bucket/m1"}],
        "nextPageToken": "tok",
    }
    res_search = normalize_tracking_response(service_pb2.SearchLoggedModels, data)
    assert res_search["models"][0]["info"]["artifact_uri"] == "gs://test-bucket/m1"
    assert res_search["next_page_token"] == "tok"


def test_normalize_passthrough_apis() -> None:
    passthroughs = [
        "GetMetricHistory",
        "GetMetricHistoryBulkInterval",
        "SearchDatasets",
        "LogMetric",
        "LogParam",
        "LogBatch",
        "SetTag",
        "DeleteTag",
        "DeleteRun",
        "DeleteExperiment",
        "RestoreExperiment",
        "DeleteExperimentTag",
        "RestoreRun",
        "LogInputs",
        "LogOutputs",
        "LogModel",
        "LogLoggedModelParams",
        "DeleteLoggedModelTag",
    ]
    raw_payload = {"arbitrary_key": "arbitrary_value", "nested": [1, 2, 3]}
    for api_name in passthroughs:
        res = normalize_tracking_response(api_name, raw_payload)
        assert res == raw_payload


def test_normalization_idempotency() -> None:
    cases: list[tuple[Any, dict[str, Any]]] = [
        (
            service_pb2.CreateRun,
            {"run_id": "r1", "artifactLocation": "gs://test-bucket/r1"},
        ),
        (
            service_pb2.UpdateRun,
            {"run_id": "r1", "artifactLocation": "gs://test-bucket/r1"},
        ),
        (
            service_pb2.GetExperiment,
            {
                "name": "projects/p/locations/l/namespaces/n/experiments/exp-1",
                "artifactLocation": "gs://test-bucket/e1",
            },
        ),
        (service_pb2.CreateExperiment, {"experimentId": "123"}),
        (
            service_pb2.SearchRuns,
            {"runs": [{"run_id": "r1", "artifactLocation": "gs://test-bucket/r1"}]},
        ),
        (
            service_pb2.SearchExperiments,
            {
                "experiments": [
                    {
                        "name": "projects/p/locations/l/namespaces/n/experiments/exp-1",
                        "artifactLocation": "gs://test-bucket/e1",
                    }
                ]
            },
        ),
    ]
    if hasattr(service_pb2, "CreateLoggedModel"):
        cases.append(
            (
                service_pb2.CreateLoggedModel,
                {"model_id": "m1", "artifactLocation": "gs://test-bucket/m1"},
            )
        )
    if hasattr(service_pb2, "SearchLoggedModels"):
        cases.append(
            (
                service_pb2.SearchLoggedModels,
                {
                    "models": [
                        {"model_id": "m1", "artifactLocation": "gs://test-bucket/m1"}
                    ]
                },
            )
        )

    for api, payload in cases:
        pass1 = normalize_tracking_response(api, payload)
        pass2 = normalize_tracking_response(api, json.loads(json.dumps(pass1)))
        assert pass1 == pass2, f"Idempotency failed for {api}"


# ---------------------------------------------------------------------------
# Test Case 5: _call_endpoint
# ---------------------------------------------------------------------------


def test_call_endpoint_get_uses_params() -> None:
    store: Any = GcpTrackingRestStore(_GCP_TRACKING_URI)
    mock_resp = _make_mock_response(
        200,
        {
            "experiment": {
                "experiment_id": "1",
                "name": "exp1",
                "artifact_location": "gs://bucket-1/artifacts",
            }
        },
    )

    with unittest.mock.patch(
        "mlflow.utils.rest_utils.http_request", return_value=mock_resp
    ) as mock_http:
        store.get_experiment("1")
        mock_http.assert_called_once()
        _, kwargs = mock_http.call_args
        assert kwargs["method"] == "GET"
        assert "params" in kwargs
        assert kwargs["params"] == {"experiment_id": "1"}
        assert "json" not in kwargs


def test_call_endpoint_post_uses_json_body() -> None:
    store: Any = GcpTrackingRestStore(_GCP_TRACKING_URI)
    mock_resp = _make_mock_response(200, {"experiment_id": "123"})

    with unittest.mock.patch(
        "mlflow.utils.rest_utils.http_request", return_value=mock_resp
    ) as mock_http:
        store.create_experiment(
            "exp-test", artifact_location="gs://test-bucket/exp-test"
        )
        mock_http.assert_called_once()
        _, kwargs = mock_http.call_args
        assert kwargs["method"] == "POST"
        assert "json" in kwargs
        assert kwargs["json"]["name"] == "exp-test"
        assert "params" not in kwargs


def test_call_endpoint_delete_does_not_send_json_body() -> None:
    store: Any = GcpTrackingRestStore(_GCP_TRACKING_URI)
    mock_resp = _make_mock_response(200, {})

    with unittest.mock.patch(
        "mlflow.utils.rest_utils.http_request", return_value=mock_resp
    ) as mock_http:
        store.delete_experiment("1")
        mock_http.assert_called_once()
        _, kwargs = mock_http.call_args
        assert kwargs["method"] == "POST"
        # Explicit test with method="DELETE":
        store._call_endpoint(
            service_pb2.DeleteExperiment,
            json_body='{"experiment_id": "1"}',
            endpoint="/api/2.0/mlflow/experiments/delete",
            method="DELETE",
        )
        _, kwargs_del = mock_http.call_args
        assert kwargs_del["method"] == "DELETE"
        assert "json" not in kwargs_del
        assert kwargs_del["params"] == {"experiment_id": "1"}


def test_call_endpoint_security_validations() -> None:
    store: Any = GcpTrackingRestStore(_GCP_TRACKING_URI)

    # Positive test cases: valid endpoints (including path IDs, AIP colon methods,
    # and v3) pass
    mock_resp = _make_mock_response(
        200, {"experiment": {"experiment_id": "1", "name": "exp"}}
    )
    with unittest.mock.patch(
        "mlflow.utils.rest_utils.http_request", return_value=mock_resp
    ) as mock_http:
        store._call_endpoint(
            service_pb2.GetExperiment,
            endpoint="/api/2.0/mlflow/logged-models/m-123",
            method="GET",
        )
        assert (
            mock_http.call_args[1]["endpoint"] == "/api/2.0/mlflow/logged-models/m-123"
        )

        store._call_endpoint(
            service_pb2.GetExperiment,
            endpoint="/api/2.0/mlflow/experiments:search",
            method="POST",
            json_body='{"filter": ""}',
        )
        assert (
            mock_http.call_args[1]["endpoint"] == "/api/2.0/mlflow/experiments:search"
        )

        store._call_endpoint(
            service_pb2.GetExperiment,
            endpoint="/api/3.0/mlflow/traces/t-123",
            method="GET",
        )
        assert mock_http.call_args[1]["endpoint"] == "/api/3.0/mlflow/traces/t-123"

    # Invalid endpoint prefix
    with pytest.raises(MlflowException) as exc_info:
        store._call_endpoint(
            service_pb2.GetExperiment,
            endpoint="/evil/arbitrary/path",
        )
    assert exc_info.value.error_code == "INVALID_PARAMETER_VALUE"
    assert "Invalid MLflow endpoint path" in str(exc_info.value)

    # Path traversal
    with pytest.raises(MlflowException) as exc_info:
        store._call_endpoint(
            service_pb2.GetExperiment,
            endpoint="/api/2.0/mlflow/experiments/../../forbidden",
        )
    assert exc_info.value.error_code == "INVALID_PARAMETER_VALUE"

    # Query or fragment delimiter in endpoint
    with pytest.raises(MlflowException) as exc_info:
        store._call_endpoint(
            service_pb2.GetExperiment,
            endpoint="/api/2.0/mlflow/experiments/list?query=1",
        )
    assert exc_info.value.error_code == "INVALID_PARAMETER_VALUE"

    # Unsupported HTTP method
    with pytest.raises(MlflowException) as exc_info:
        store._call_endpoint(
            service_pb2.GetExperiment,
            endpoint="/api/2.0/mlflow/experiments/get",
            method="TRACE",
        )
    assert exc_info.value.error_code == "INVALID_PARAMETER_VALUE"
    assert "Unsupported HTTP method" in str(exc_info.value)


def test_call_endpoint_get_experiment_by_name_404_translates() -> None:
    store = GcpTrackingRestStore(_GCP_TRACKING_URI)
    mock_404 = _make_mock_response(
        404,
        {
            "error": {
                "code": 404,
                "message": "Experiment 'new_experiment' not found",
                "status": "NOT_FOUND",
            }
        },
    )

    with unittest.mock.patch(
        "mlflow.utils.rest_utils.http_request", return_value=mock_404
    ):
        with pytest.raises(MlflowException) as exc_info:
            store._call_endpoint(
                service_pb2.GetExperimentByName,
                json_body='{"experiment_name": "new_experiment"}',
            )
        assert exc_info.value.error_code == "RESOURCE_DOES_NOT_EXIST"

    # Verify that mlflow.set_experiment catches it and creates the experiment
    mock_create = _make_mock_response(200, {"experiment_id": "exp-new-id"})
    mock_get = _make_mock_response(
        200,
        {
            "experiment": {
                "experiment_id": "exp-new-id",
                "name": "new_experiment",
                "artifact_location": "gs://test-bucket/new_experiment",
                "lifecycle_stage": "active",
            }
        },
    )

    responses = [mock_404, mock_create, mock_get]

    def _mock_req(*args: Any, **kwargs: Any) -> requests.Response:
        return responses.pop(0)

    with (
        unittest.mock.patch(
            "mlflow.utils.rest_utils.http_request", side_effect=_mock_req
        ),
        unittest.mock.patch("mlflow.tracking.fluent._active_experiment_id", None),
    ):
        ensure_tracking_store_registered()
        mlflow.set_tracking_uri(_GCP_TRACKING_URI)
        exp = mlflow.set_experiment("new_experiment")
        assert exp.experiment_id == "exp-new-id"
        assert exp.name == "new_experiment"


def test_call_endpoint_aip193_error_mapping() -> None:
    store: Any = GcpTrackingRestStore(_GCP_TRACKING_URI)

    status_codes = [
        (400, INVALID_PARAMETER_VALUE),
        (401, UNAUTHENTICATED),
        (403, PERMISSION_DENIED),
        (404, RESOURCE_DOES_NOT_EXIST),
    ]

    for status_code, expected_code in status_codes:
        mock_resp = _make_mock_response(
            status_code,
            {
                "error": {
                    "code": status_code,
                    "message": f"Error {status_code}",
                    "status": "FAILED",
                }
            },
        )
        with unittest.mock.patch(
            "mlflow.utils.rest_utils.http_request", return_value=mock_resp
        ):
            with pytest.raises(GcpMlflowApiError) as exc_info:
                store.get_run("run-123")
            assert exc_info.value.http_status == status_code
            from mlflow.protos.databricks_pb2 import ErrorCode

            assert exc_info.value.error_code == ErrorCode.Name(expected_code)


def test_call_endpoint_lifecycle_passthrough_operations() -> None:
    store = GcpTrackingRestStore(_GCP_TRACKING_URI)
    mock_resp = _make_mock_response(200, {})

    lifecycle_calls: list[tuple[Any, dict[str, Any]]] = [
        (service_pb2.DeleteExperiment, {"experiment_id": "1"}),
        (service_pb2.RestoreExperiment, {"experiment_id": "1"}),
        (service_pb2.DeleteRun, {"run_id": "r1"}),
        (service_pb2.RestoreRun, {"run_id": "r1"}),
        (service_pb2.DeleteTag, {"run_id": "r1", "key": "k"}),
    ]
    if hasattr(service_pb2, "DeleteExperimentTag"):
        lifecycle_calls.append(
            (service_pb2.DeleteExperimentTag, {"experiment_id": "1", "key": "k"})
        )

    with unittest.mock.patch(
        "mlflow.utils.rest_utils.http_request", return_value=mock_resp
    ):
        for api, body in lifecycle_calls:
            proto = store._call_endpoint(api, json_body=json.dumps(body))
            assert proto is not None


def test_call_endpoint_204_no_content() -> None:
    store = GcpTrackingRestStore(_GCP_TRACKING_URI)
    mock_resp = requests.Response()
    mock_resp.status_code = 204
    mock_resp._content = b""

    with unittest.mock.patch(
        "mlflow.utils.rest_utils.http_request", return_value=mock_resp
    ):
        proto = store._call_endpoint(
            service_pb2.DeleteRun, json_body='{"run_id": "r1"}'
        )
        assert proto is not None


# ---------------------------------------------------------------------------
# Test Case 6: Import order independence
# ---------------------------------------------------------------------------


def test_import_tracking_store_before_mlflow() -> None:
    code = (
        "import sys\n"
        "import google_cloud_managed_mlflow.tracking_store as ts\n"
        "assert 'mlflow' not in sys.modules, 'mlflow imported prematurely!'\n"
        "from google_cloud_managed_mlflow.tracking_store import ("
        " GcpTrackingRestStore, build_tracking_store\n"
        ")\n"
        "import mlflow\n"
        "store = build_tracking_store("
        " 'https://mlflow.googleapis.com/v1/projects/p/locations/l/namespaces/n'\n"
        ")\n"
        "assert isinstance(store, GcpTrackingRestStore)\n"
        "print('OK')\n"
    )
    repo_root = str(Path(__file__).resolve().parent.parent)
    pythonpath = f"{repo_root}:{os.environ.get('PYTHONPATH', '')}".rstrip(":")
    env = {**os.environ, "PYTHONPATH": pythonpath}
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )
    assert result.returncode == 0, f"Error: {result.stderr}"
    assert "OK" in result.stdout


def test_import_mlflow_before_tracking_store() -> None:
    code = (
        "import mlflow\n"
        "import google_cloud_managed_mlflow.tracking_store as ts\n"
        "from google_cloud_managed_mlflow.tracking_store import ("
        " GcpTrackingRestStore, build_tracking_store\n"
        ")\n"
        "store = build_tracking_store("
        " 'https://mlflow.googleapis.com/v1/projects/p/locations/l/namespaces/n'\n"
        ")\n"
        "assert isinstance(store, GcpTrackingRestStore)\n"
        "print('OK')\n"
    )
    repo_root = str(Path(__file__).resolve().parent.parent)
    pythonpath = f"{repo_root}:{os.environ.get('PYTHONPATH', '')}".rstrip(":")
    env = {**os.environ, "PYTHONPATH": pythonpath}
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )
    assert result.returncode == 0, f"Error: {result.stderr}"
    assert "OK" in result.stdout


def test_ensure_tracking_store_registered() -> None:
    registered = ensure_tracking_store_registered()
    assert registered is True
    from mlflow.tracking._tracking_service.utils import _tracking_store_registry

    builder = _tracking_store_registry.get_store_builder(
        "https://mlflow.googleapis.com/v1/projects/p/locations/l/namespaces/n"
    )
    assert builder is build_tracking_store


def test_reset_tracking_store_cache() -> None:
    cls1 = ts_mod._get_tracking_store_cls()
    assert cls1 is not None
    reset_tracking_store_cache()
    assert ts_mod._TRACKING_STORE_CLS is None
    cls2 = ts_mod._get_tracking_store_cls()
    assert cls2 is not None


def test_search_datasets_raises_feature_disabled() -> None:
    store = build_tracking_store(
        "https://mlflow.googleapis.com/v1/projects/p/locations/l/namespaces/n"
    )
    from mlflow.exceptions import MlflowException

    with pytest.raises(MlflowException) as exc_info:
        store.search_datasets(experiment_ids=["123"])
    assert exc_info.value.error_code == "FEATURE_DISABLED"
    assert "Evaluation dataset search" in str(exc_info.value)


def test_top_level_package_lazy_auth_exports() -> None:
    import google_cloud_managed_mlflow as pkg

    assert hasattr(pkg, "GoogleCloudAuthProvider")
    assert hasattr(pkg, "create_secure_session")
    assert hasattr(pkg, "AUTH_PROVIDER_NAME")
    from google_cloud_managed_mlflow import (
        AUTH_PROVIDER_NAME,
        GoogleCloudAuthProvider,
    )

    assert AUTH_PROVIDER_NAME == "google-cloud-auth"
    assert GoogleCloudAuthProvider is not None

    # Dunder attribute probes immediately raise AttributeError without importing
    with pytest.raises(AttributeError):
        _ = pkg.__wrapped__
    with pytest.raises(AttributeError):
        _ = pkg.__unknown_dunder__

    # Unknown non-dunder attributes raise AttributeError
    with pytest.raises(AttributeError):
        _ = pkg.non_existent_attribute


def test_harden_cached_session_forwards_cache_methods() -> None:
    from unittest.mock import MagicMock

    import mlflow.utils.rest_utils as rest_utils

    from google_cloud_managed_mlflow import auth

    mock_func = MagicMock()
    mock_func.cache_clear = MagicMock()
    mock_func.cache_info = MagicMock()
    session_attr = "_cached_get_request_session"
    orig = getattr(rest_utils, session_attr, None)
    try:
        setattr(rest_utils, session_attr, mock_func)
        auth.harden_mlflow_request_session()
        current: Any = getattr(rest_utils, session_attr, None)
        assert current is not None
        assert hasattr(current, "cache_clear")
        assert hasattr(current, "cache_info")
        current.cache_clear()
        mock_func.cache_clear.assert_called()
    finally:
        setattr(rest_utils, session_attr, orig)


def test_harden_mlflow_request_session_preserves_retries_and_pool_settings() -> None:
    import mlflow.utils.request_utils as req_utils

    from google_cloud_managed_mlflow import auth
    from google_cloud_managed_mlflow.auth import _Tls12HttpAdapter

    custom_session = requests.Session()
    custom_adapter = requests.adapters.HTTPAdapter(
        max_retries=5,
        pool_connections=25,
        pool_maxsize=50,
        pool_block=True,
    )
    custom_session.mount("https://", custom_adapter)

    def mock_get_session(host_creds: Any = None) -> requests.Session:
        return custom_session

    setattr(req_utils, "_cached_get_request_session", mock_get_session)  # noqa: B010
    assert auth.harden_mlflow_request_session() is True

    hardened_session = req_utils._cached_get_request_session("https://example.com")
    adapter: Any = hardened_session.adapters.get("https://")
    assert isinstance(adapter, _Tls12HttpAdapter)
    retries = adapter.max_retries
    total_retries = getattr(retries, "total", retries)
    assert total_retries == 5
    raw_adapter: Any = adapter
    assert raw_adapter._pool_connections == 25
    assert raw_adapter._pool_maxsize == 50
    assert raw_adapter._pool_block is True


def test_non_gcp_insecure_tls_allowed_with_hardened_adapter() -> None:
    from google_cloud_managed_mlflow.auth import _Tls12HttpAdapter

    adapter = _Tls12HttpAdapter()
    req = requests.Request(
        "GET", "https://custom-mlflow.internal:5000/api/2.0/mlflow/experiments/list"
    ).prepare()

    with unittest.mock.patch(
        "requests.adapters.HTTPAdapter.send", return_value=_make_mock_response(200, {})
    ) as mock_super_send:
        resp = adapter.send(req, verify=False)
        assert resp.status_code == 200
        mock_super_send.assert_called_once()
        _, kwargs = mock_super_send.call_args
        assert kwargs["verify"] is False

    gcp_req = requests.Request(
        "GET",
        "https://mlflow.googleapis.com/v1/projects/p/locations/l/namespaces/n/experiments/1",
    ).prepare()
    with pytest.raises(MlflowException) as exc_info:
        adapter.send(gcp_req, verify=False)
    assert "TLS verification cannot be disabled for Google Cloud connections" in str(
        exc_info.value
    )


def test_rfc3339_parse_failure_in_ensure_run_info_and_experiment() -> None:
    from google_cloud_managed_mlflow.uri import InvalidTimestampError

    bad_run_info = {"startTime": "not-a-valid-timestamp"}
    with pytest.raises(InvalidTimestampError):
        _ensure_run_info_fields(bad_run_info)

    bad_exp = {"creationTime": "not-a-valid-timestamp"}
    with pytest.raises(InvalidTimestampError):
        _ensure_experiment_fields(bad_exp)


def test_get_experiment_display_name_and_resource_path_mapping() -> None:
    payload = {
        "name": "projects/my-p/locations/us-c1/namespaces/ns1/experiments/10",
        "displayName": "my-cool-exp",
        "artifactLocation": "gs://test-bucket/exp10",
    }
    normalized = _ensure_experiment_fields(payload.copy())
    assert normalized["name"] == "my-cool-exp"
    assert normalized["experiment_id"] == "10"
    assert normalized["artifact_location"] == "gs://test-bucket/exp10"
    assert "displayName" not in normalized

    res = normalize_tracking_response(service_pb2.GetExperiment, payload.copy())
    assert res["experiment"]["name"] == "my-cool-exp"
    assert res["experiment"]["experiment_id"] == "10"

    store: Any = GcpTrackingRestStore(_GCP_TRACKING_URI)
    mock_resp = _make_mock_response(200, payload)
    with unittest.mock.patch(
        "mlflow.utils.rest_utils.http_request", return_value=mock_resp
    ):
        exp = store.get_experiment("10")
        assert exp.name == "my-cool-exp"
        assert exp.experiment_id == "10"

    payload_both = {
        "displayName": "name1",
        "display_name": "name2",
    }
    norm_both = _ensure_experiment_fields(payload_both.copy())
    assert norm_both["name"] == "name1"
    assert "displayName" not in norm_both
    assert "display_name" not in norm_both
