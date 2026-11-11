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

"""Tests for GcpModelRegistryRestStore and model registry store registration."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import unittest.mock
from pathlib import Path
from typing import Any

import mlflow
import pytest
import requests
from mlflow.exceptions import MlflowException
from mlflow.protos import model_registry_pb2
from mlflow.store.model_registry.rest_store import RestStore

import google_cloud_managed_mlflow.registry_store as rs_mod
from google_cloud_managed_mlflow.registry_store import (
    GcpModelRegistryRestStore,
    _ensure_model_version_fields,
    _ensure_registered_model_fields,
    _sanitize_and_quote_model_name,
    _sanitize_prompt_filter,
    build_registry_store,
    ensure_registry_store_registered,
    normalize_registry_response,
    reset_registry_store_cache,
)
from google_cloud_managed_mlflow.uri import GcpMlflowApiError

_GCP_REGISTRY_URI = (
    "https://mlflow.googleapis.com/v1/projects/my-project"
    "/locations/us-central1/registries/my-registry"
)
_GCP_TRACKING_URI = (
    "https://mlflow.googleapis.com/v1/projects/my-project"
    "/locations/us-central1/namespaces/my-namespace"
)
_GCP_TRACKING_URI_DIFF_LOC = (
    "https://mlflow.googleapis.com/v1/projects/my-project"
    "/locations/us-east1/namespaces/my-namespace"
)
_CANONICAL_NS_NAME = "projects/my-project/locations/us-central1/namespaces/my-namespace"
_CANONICAL_NS_NAME_DIFF_LOC = (
    "projects/my-project/locations/us-east1/namespaces/my-namespace"
)
_NON_GCP_URI = "https://custom-mlflow.example.com"


@pytest.fixture(autouse=True)
def _isolate_registry_state(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    import mlflow.tracking._tracking_service.utils as tr_utils

    monkeypatch.setattr(tr_utils, "_tracking_uri", None, raising=False)


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
# build_registry_store & fallback
# ---------------------------------------------------------------------------


def test_build_registry_store_gcp_uri() -> None:
    store = build_registry_store(_GCP_REGISTRY_URI, tracking_uri=_GCP_TRACKING_URI)
    assert isinstance(store, rs_mod.GcpModelRegistryRestStore)
    assert store.store_uri == _GCP_REGISTRY_URI
    assert store.tracking_uri == _GCP_TRACKING_URI
    assert (
        store.registry_resource
        == "projects/my-project/locations/us-central1/registries/my-registry"
    )
    assert store.base_api_url == "https://mlflow.googleapis.com/v1"
    assert store.endpoint_url == _GCP_REGISTRY_URI


def test_build_registry_store_non_gcp_uri_fallback() -> None:
    with unittest.mock.patch(
        "mlflow.tracking._model_registry.utils._get_rest_store"
    ) as mock_get_rest:
        sentinel_store = unittest.mock.MagicMock(spec=RestStore)
        mock_get_rest.return_value = sentinel_store

        result = build_registry_store(_NON_GCP_URI)
        assert result is sentinel_store
        mock_get_rest.assert_called_once_with(_NON_GCP_URI)


# ---------------------------------------------------------------------------
# URI validation on store initialization
# ---------------------------------------------------------------------------


def test_registry_store_rejects_empty_uri() -> None:
    with pytest.raises(MlflowException) as exc_info:
        GcpModelRegistryRestStore("")
    assert exc_info.value.error_code == "INVALID_PARAMETER_VALUE"


def test_registry_store_rejects_namespace_uri() -> None:
    with pytest.raises(MlflowException) as exc_info:
        GcpModelRegistryRestStore(_GCP_TRACKING_URI)
    assert exc_info.value.error_code == "INVALID_PARAMETER_VALUE"
    assert "registries" in str(exc_info.value)


def test_registry_store_rejects_insecure_http() -> None:
    insecure = "http://mlflow.googleapis.com/v1/projects/p/locations/l/registries/r"
    with pytest.raises(MlflowException) as exc_info:
        GcpModelRegistryRestStore(insecure)
    assert exc_info.value.error_code == "INVALID_PARAMETER_VALUE"


# ---------------------------------------------------------------------------
# Source namespace resolution
# ---------------------------------------------------------------------------


def test_resolve_source_namespace_from_tracking_uri_canonical_string() -> None:
    store = GcpModelRegistryRestStore(
        _GCP_REGISTRY_URI, tracking_uri=_CANONICAL_NS_NAME
    )
    assert store._resolve_source_namespace() == _CANONICAL_NS_NAME


def test_resolve_source_namespace_from_tracking_uri_full_url() -> None:
    store = GcpModelRegistryRestStore(_GCP_REGISTRY_URI, tracking_uri=_GCP_TRACKING_URI)
    assert store._resolve_source_namespace() == _CANONICAL_NS_NAME


def test_resolve_source_namespace_from_mlflow_get_tracking_uri(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(mlflow, "get_tracking_uri", lambda: _GCP_TRACKING_URI)
    store = GcpModelRegistryRestStore(_GCP_REGISTRY_URI)
    assert store._resolve_source_namespace() == _CANONICAL_NS_NAME


def test_resolve_source_namespace_from_env_var(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MLFLOW_TRACKING_URI", _GCP_TRACKING_URI)
    store = GcpModelRegistryRestStore(_GCP_REGISTRY_URI)
    assert store._resolve_source_namespace() == _CANONICAL_NS_NAME


def test_resolve_source_namespace_cross_location_canonical_raises() -> None:
    store = GcpModelRegistryRestStore(
        _GCP_REGISTRY_URI, tracking_uri=_CANONICAL_NS_NAME_DIFF_LOC
    )
    with pytest.raises(MlflowException) as exc_info:
        store._resolve_source_namespace()
    assert exc_info.value.error_code == "INVALID_PARAMETER_VALUE"
    assert "Cross-location model registration is not supported" in str(exc_info.value)


def test_resolve_source_namespace_cross_location_url_raises() -> None:
    store = GcpModelRegistryRestStore(
        _GCP_REGISTRY_URI, tracking_uri=_GCP_TRACKING_URI_DIFF_LOC
    )
    with pytest.raises(MlflowException) as exc_info:
        store._resolve_source_namespace()
    assert exc_info.value.error_code == "INVALID_PARAMETER_VALUE"
    assert "Cross-location model registration is not supported" in str(exc_info.value)


def test_resolve_source_namespace_missing_raises() -> None:
    store = GcpModelRegistryRestStore(_GCP_REGISTRY_URI)
    with pytest.raises(MlflowException) as exc_info:
        store._resolve_source_namespace()
    assert exc_info.value.error_code == "INVALID_PARAMETER_VALUE"
    assert "requires a configured tracking namespace" in str(exc_info.value)


# ---------------------------------------------------------------------------
# Wire rewrites in _call_endpoint
# ---------------------------------------------------------------------------


def test_rewrite_create_registered_model() -> None:
    store = GcpModelRegistryRestStore(_GCP_REGISTRY_URI)
    mock_resp = _make_mock_response(
        200,
        {
            "registered_model": {
                "name": "model_1",
                "creation_timestamp": 123456,
            }
        },
    )
    payload = {
        "name": "model_1",
        "description": "test model description",
        "tags": [{"key": "env", "value": "prod"}],
    }
    with unittest.mock.patch(
        "mlflow.utils.rest_utils.http_request", return_value=mock_resp
    ) as mock_http:
        store._call_endpoint(
            model_registry_pb2.CreateRegisteredModel,
            json_body=json.dumps(payload),
        )
        mock_http.assert_called_once()
        _, kwargs = mock_http.call_args
        assert kwargs["method"] == "POST"
        assert kwargs["endpoint"] == "/registeredModels"
        assert kwargs["params"] == {"registered_model_id": "model_1"}
        assert kwargs["json"] == {
            "description": "test model description",
            "annotations": {"env": "prod"},
        }


def test_endpoint_url_concatenation_with_host_creds() -> None:
    store = GcpModelRegistryRestStore(_GCP_REGISTRY_URI)
    endpoint, _, _, _ = store._dispatch_request(
        "CreateRegisteredModel",
        model_registry_pb2.CreateRegisteredModel,
        {"name": "model_1"},
    )
    assert endpoint == "/registeredModels"
    url = f"{store.get_host_creds().host}{endpoint}"
    assert url == f"{_GCP_REGISTRY_URI}/registeredModels"


def test_rewrite_create_registered_model_missing_name_raises() -> None:
    store = GcpModelRegistryRestStore(_GCP_REGISTRY_URI)
    with pytest.raises(MlflowException) as exc_info:
        store._call_endpoint(
            model_registry_pb2.CreateRegisteredModel,
            json_body=json.dumps({"description": "test model"}),
        )
    assert exc_info.value.error_code == "INVALID_PARAMETER_VALUE"
    assert "Registered model name is required." in str(exc_info.value)


def test_rewrite_get_registered_model() -> None:
    store = GcpModelRegistryRestStore(_GCP_REGISTRY_URI)
    mock_resp = _make_mock_response(
        200,
        {
            "registered_model": {
                "name": "model_1",
                "creation_timestamp": 123456,
            }
        },
    )
    with unittest.mock.patch(
        "mlflow.utils.rest_utils.http_request", return_value=mock_resp
    ) as mock_http:
        store._call_endpoint(
            model_registry_pb2.GetRegisteredModel,
            json_body=json.dumps({"name": "model_1"}),
        )
        mock_http.assert_called_once()
        _, kwargs = mock_http.call_args
        assert kwargs["method"] == "GET"
        assert kwargs["endpoint"] == "/registeredModels/model_1"
        assert "params" not in kwargs
        assert "json" not in kwargs


def test_rewrite_update_registered_model() -> None:
    store = GcpModelRegistryRestStore(_GCP_REGISTRY_URI)
    mock_resp = _make_mock_response(
        200,
        {
            "registered_model": {
                "name": "model_1",
                "description": "updated",
            }
        },
    )
    with unittest.mock.patch(
        "mlflow.utils.rest_utils.http_request", return_value=mock_resp
    ) as mock_http:
        store._call_endpoint(
            model_registry_pb2.UpdateRegisteredModel,
            json_body=json.dumps({"name": "model_1", "description": "updated"}),
        )
        mock_http.assert_called_once()
        _, kwargs = mock_http.call_args
        assert kwargs["method"] == "PATCH"
        assert kwargs["endpoint"] == "/registeredModels/model_1"
        assert kwargs["params"] == {"update_mask": "description"}
        assert kwargs["json"] == {"description": "updated"}


def test_rewrite_delete_registered_model_empty_body() -> None:
    store = GcpModelRegistryRestStore(_GCP_REGISTRY_URI)
    mock_resp = _make_mock_response(200, {})
    with unittest.mock.patch(
        "mlflow.utils.rest_utils.http_request", return_value=mock_resp
    ) as mock_http:
        store._call_endpoint(
            model_registry_pb2.DeleteRegisteredModel,
            json_body=json.dumps({"name": "model_1"}),
        )
        mock_http.assert_called_once()
        _, kwargs = mock_http.call_args
        assert kwargs["method"] == "DELETE"
        assert kwargs["endpoint"] == "/registeredModels/model_1"
        assert "params" not in kwargs
        assert "json" not in kwargs


def test_rewrite_delete_model_version_empty_body() -> None:
    store = GcpModelRegistryRestStore(_GCP_REGISTRY_URI)
    mock_resp = _make_mock_response(200, {})
    with unittest.mock.patch(
        "mlflow.utils.rest_utils.http_request", return_value=mock_resp
    ) as mock_http:
        store._call_endpoint(
            model_registry_pb2.DeleteModelVersion,
            json_body=json.dumps({"name": "model_1", "version": 3}),
        )
        mock_http.assert_called_once()
        _, kwargs = mock_http.call_args
        assert kwargs["method"] == "DELETE"
        assert kwargs["endpoint"] == "/api/2.0/mlflow/model-versions/delete"
        assert kwargs["params"] == {"name": "model_1", "version": "3"}
        assert "json" not in kwargs


def test_rewrite_delete_registered_model_tag_empty_body() -> None:
    store = GcpModelRegistryRestStore(_GCP_REGISTRY_URI)
    mock_resp = _make_mock_response(200, {})
    with unittest.mock.patch(
        "mlflow.utils.rest_utils.http_request", return_value=mock_resp
    ) as mock_http:
        store._call_endpoint(
            model_registry_pb2.DeleteRegisteredModelTag,
            json_body=json.dumps({"name": "model_1", "key": "tag1"}),
        )
        mock_http.assert_called_once()
        _, kwargs = mock_http.call_args
        assert kwargs["method"] == "DELETE"
        assert kwargs["endpoint"] == "/api/2.0/mlflow/registered-models/delete-tag"
        assert kwargs["params"] == {"name": "model_1", "tag_key": "tag1"}
        assert "json" not in kwargs


def test_rewrite_delete_model_version_tag_empty_body() -> None:
    store = GcpModelRegistryRestStore(_GCP_REGISTRY_URI)
    mock_resp = _make_mock_response(200, {})
    with unittest.mock.patch(
        "mlflow.utils.rest_utils.http_request", return_value=mock_resp
    ) as mock_http:
        store._call_endpoint(
            model_registry_pb2.DeleteModelVersionTag,
            json_body=json.dumps({"name": "model_1", "version": 2, "key": "tag2"}),
        )
        mock_http.assert_called_once()
        _, kwargs = mock_http.call_args
        assert kwargs["method"] == "DELETE"
        assert kwargs["endpoint"] == "/api/2.0/mlflow/model-versions/delete-tag"
        assert kwargs["params"] == {
            "name": "model_1",
            "version": "2",
            "tag_key": "tag2",
        }
        assert "json" not in kwargs


def test_rewrite_delete_registered_model_alias_empty_body() -> None:
    store = GcpModelRegistryRestStore(_GCP_REGISTRY_URI)
    mock_resp = _make_mock_response(200, {})
    with unittest.mock.patch(
        "mlflow.utils.rest_utils.http_request", return_value=mock_resp
    ) as mock_http:
        store._call_endpoint(
            model_registry_pb2.DeleteRegisteredModelAlias,
            json_body=json.dumps({"name": "model_1", "alias": "champion"}),
        )
        mock_http.assert_called_once()
        _, kwargs = mock_http.call_args
        assert kwargs["method"] == "DELETE"
        assert kwargs["endpoint"] == "/api/2.0/mlflow/registered-models/alias"
        assert kwargs["params"] == {"name": "model_1", "alias": "champion"}
        assert "json" not in kwargs


def test_rewrite_create_model_version_injects_source_namespace() -> None:
    store = GcpModelRegistryRestStore(_GCP_REGISTRY_URI, tracking_uri=_GCP_TRACKING_URI)
    mock_resp = _make_mock_response(
        200,
        {
            "model_version": {
                "name": "model_1",
                "version": "1",
                "creation_timestamp": 123456,
            }
        },
    )
    with unittest.mock.patch(
        "mlflow.utils.rest_utils.http_request", return_value=mock_resp
    ) as mock_http:
        store._call_endpoint(
            model_registry_pb2.CreateModelVersion,
            json_body=json.dumps({"name": "model_1", "source": "gs://bucket/model"}),
        )
        mock_http.assert_called_once()
        _, kwargs = mock_http.call_args
        assert kwargs["method"] == "POST"
        assert kwargs["endpoint"] == "/api/2.0/mlflow/model-versions/create"
        assert kwargs["json"]["source_namespace"] == _CANONICAL_NS_NAME
        assert kwargs["json"]["source"] == "gs://bucket/model"
        assert "params" not in kwargs


def test_rewrite_other_methods_get_uses_params() -> None:
    store = GcpModelRegistryRestStore(_GCP_REGISTRY_URI)
    mock_resp = _make_mock_response(
        200,
        {
            "registered_models": [],
        },
    )
    with unittest.mock.patch(
        "mlflow.utils.rest_utils.http_request", return_value=mock_resp
    ) as mock_http:
        store._call_endpoint(
            model_registry_pb2.SearchRegisteredModels,
            json_body=json.dumps({"max_results": 10}),
        )
        mock_http.assert_called_once()
        _, kwargs = mock_http.call_args
        assert kwargs["method"] == "GET"
        assert kwargs["endpoint"] == "/api/2.0/mlflow/registered-models/search"
        assert kwargs["params"] == {"max_results": 10}


def test_rewrite_search_registered_models_strips_prompt_filter() -> None:
    store = GcpModelRegistryRestStore(_GCP_REGISTRY_URI)
    mock_resp = _make_mock_response(200, {"registered_models": []})
    with unittest.mock.patch(
        "mlflow.utils.rest_utils.http_request", return_value=mock_resp
    ) as mock_http:
        store._call_endpoint(
            model_registry_pb2.SearchRegisteredModels,
            json_body=json.dumps(
                {
                    "filter": (
                        "name = 'model_1' AND tag.`mlflow.prompt.is_prompt` != 'true'"
                    ),
                    "max_results": 10,
                }
            ),
        )
        mock_http.assert_called_once()
        _, kwargs = mock_http.call_args
        assert kwargs["method"] == "GET"
        assert kwargs["params"]["filter"] == "name = 'model_1'"
        assert "json" not in kwargs


def test_rewrite_search_registered_models_multi_clause_prompt_filter() -> None:
    store = GcpModelRegistryRestStore(_GCP_REGISTRY_URI)
    mock_resp = _make_mock_response(200, {"registered_models": []})
    with unittest.mock.patch(
        "mlflow.utils.rest_utils.http_request", return_value=mock_resp
    ) as mock_http:
        store._call_endpoint(
            model_registry_pb2.SearchRegisteredModels,
            json_body=json.dumps(
                {
                    "filter": (
                        "name = 'model_1' AND tag.`mlflow.prompt.is_prompt` != 'true' "
                        "AND tags.tier = 'gold'"
                    ),
                    "max_results": 10,
                }
            ),
        )
        mock_http.assert_called_once()
        _, kwargs = mock_http.call_args
        assert kwargs["params"]["filter"] == "name = 'model_1' AND tags.tier = 'gold'"


def test_rewrite_search_registered_models_standalone_prompt_filter() -> None:
    store = GcpModelRegistryRestStore(_GCP_REGISTRY_URI)
    mock_resp = _make_mock_response(200, {"registered_models": []})
    with unittest.mock.patch(
        "mlflow.utils.rest_utils.http_request", return_value=mock_resp
    ) as mock_http:
        store._call_endpoint(
            model_registry_pb2.SearchRegisteredModels,
            json_body=json.dumps(
                {
                    "filter": "tag.`mlflow.prompt.is_prompt` != 'true'",
                    "max_results": 10,
                }
            ),
        )
        mock_http.assert_called_once()
        _, kwargs = mock_http.call_args
        assert "filter" not in kwargs["params"]
        assert kwargs["params"] == {"max_results": 10}


def test_rewrite_other_methods_post_uses_body() -> None:
    store = GcpModelRegistryRestStore(_GCP_REGISTRY_URI)
    mock_resp = _make_mock_response(200, {})
    with unittest.mock.patch(
        "mlflow.utils.rest_utils.http_request", return_value=mock_resp
    ) as mock_http:
        store._call_endpoint(
            model_registry_pb2.SetRegisteredModelTag,
            json_body=json.dumps({"name": "m1", "key": "k", "value": "v"}),
        )
        mock_http.assert_called_once()
        _, kwargs = mock_http.call_args
        assert kwargs["method"] == "POST"
        assert kwargs["endpoint"] == "/api/2.0/mlflow/registered-models/set-tag"
        assert kwargs["json"] == {"name": "m1", "key": "k", "value": "v"}
        assert "params" not in kwargs


# ---------------------------------------------------------------------------
# Response Normalization
# ---------------------------------------------------------------------------


def test_normalize_raw_registered_model_response() -> None:
    raw_gcp_api = {
        "name": (
            "projects/my-project/locations/us-central1/registries/my-registry"
            "/registeredModels/my-model"
        ),
        "createTime": "2026-11-10T12:00:00.123456Z",
        "updateTime": "2026-11-10T12:05:00.000Z",
        "description": "Production model",
        "annotations": {"environment": "prod"},
        "aliases": {"champion": "1"},
        "latestVersions": [
            {
                "name": (
                    "projects/my-project/locations/us-central1/registries/my-registry"
                    "/registeredModels/my-model/versions/1"
                ),
                "version": 1,
                "createTime": "2026-11-10T12:00:00.000Z",
                "annotations": {"tier": "gold"},
            }
        ],
    }
    resp = normalize_registry_response(
        model_registry_pb2.GetRegisteredModel,
        raw_gcp_api,
        response_proto=model_registry_pb2.GetRegisteredModel.Response(),
    )
    assert "registered_model" in resp
    rm = resp["registered_model"]
    assert rm["name"] == "my-model"
    assert rm["creation_timestamp"] == 1794312000123
    assert rm["last_updated_timestamp"] == 1794312300000
    assert rm["description"] == "Production model"
    assert rm["tags"] == [{"key": "environment", "value": "prod"}]
    assert rm["aliases"] == [{"alias": "champion", "version": "1"}]
    assert len(rm["latest_versions"]) == 1
    v = rm["latest_versions"][0]
    assert v["name"] == "my-model"
    assert v["version"] == "1"
    assert v["status"] == "READY"
    assert v["current_stage"] == "None"
    assert v["tags"] == [{"key": "tier", "value": "gold"}]


def test_normalize_raw_model_version_response() -> None:
    raw_gcp_api = {
        "name": (
            "projects/my-project/locations/us-central1/registries/my-registry"
            "/registeredModels/my-model/versions/2"
        ),
        "version": 2,
        "createTime": "2026-11-10T12:00:00.000Z",
        "updateTime": "2026-11-10T12:05:00.000Z",
        "description": "v2 release",
        "annotations": {"stage": "eval"},
        "aliases": {"challenger": "2"},
    }
    resp = normalize_registry_response(
        model_registry_pb2.CreateModelVersion,
        raw_gcp_api,
        response_proto=model_registry_pb2.CreateModelVersion.Response(),
    )
    assert "model_version" in resp
    mv = resp["model_version"]
    assert mv["name"] == "my-model"
    assert mv["version"] == "2"
    assert mv["creation_timestamp"] == 1794312000000
    assert mv["last_updated_timestamp"] == 1794312300000
    assert mv["status"] == "READY"
    assert mv["current_stage"] == "None"
    assert mv["tags"] == [{"key": "stage", "value": "eval"}]
    assert mv["aliases"] == ["challenger"]


def test_normalize_search_registered_models() -> None:
    raw_resp = {
        "registeredModels": [
            {
                "name": "projects/p/locations/l/registries/r/registeredModels/m1",
                "createTime": "2026-11-10T12:00:00.000Z",
            }
        ],
        "nextPageToken": "token-123",
    }
    resp = normalize_registry_response(
        model_registry_pb2.SearchRegisteredModels,
        raw_resp,
        response_proto=model_registry_pb2.SearchRegisteredModels.Response(),
    )
    assert "registered_models" in resp
    assert resp["registered_models"][0]["name"] == "m1"
    assert resp["next_page_token"] == "token-123"


def test_normalize_search_model_versions() -> None:
    raw_resp = {
        "modelVersions": [
            {
                "name": (
                    "projects/p/locations/l/registries/r/registeredModels/m1/versions/1"
                ),
                "version": 1,
            }
        ],
        "nextPageToken": "token-456",
    }
    resp = normalize_registry_response(
        model_registry_pb2.SearchModelVersions,
        raw_resp,
        response_proto=model_registry_pb2.SearchModelVersions.Response(),
    )
    assert "model_versions" in resp
    assert resp["model_versions"][0]["name"] == "m1"
    assert resp["model_versions"][0]["version"] == "1"
    assert resp["next_page_token"] == "token-456"


def test_normalize_get_latest_versions_response() -> None:
    raw_resp = {
        "modelVersions": [
            {
                "model_version": {
                    "name": (
                        "projects/p/locations/l/registries/r/registeredModels/m1/versions/1"
                    ),
                    "version": 1,
                    "createTime": "2026-01-01T00:00:00.000Z",
                    "status": "READY",
                }
            },
            {
                "name": (
                    "projects/p/locations/l/registries/r/registeredModels/m1/versions/2"
                ),
                "version": 2,
                "createTime": "2026-01-02T00:00:00.000Z",
                "status": "READY",
            },
        ],
    }
    resp = normalize_registry_response(
        model_registry_pb2.GetLatestVersions,
        raw_resp,
        response_proto=model_registry_pb2.GetLatestVersions.Response(),
    )
    assert "model_versions" in resp
    assert len(resp["model_versions"]) == 2
    assert "model_version" not in resp["model_versions"][0]
    assert resp["model_versions"][0]["name"] == "m1"
    assert resp["model_versions"][0]["version"] == "1"
    assert resp["model_versions"][0]["creation_timestamp"] == 1767225600000
    assert resp["model_versions"][1]["name"] == "m1"
    assert resp["model_versions"][1]["version"] == "2"
    assert resp["model_versions"][1]["creation_timestamp"] == 1767312000000

    raw_resp_snake = {
        "model_versions": [
            {
                "model_version": {
                    "name": (
                        "projects/p/locations/l/registries/r/registeredModels/m2/versions/3"
                    ),
                    "version": 3,
                }
            }
        ]
    }
    resp_snake = normalize_registry_response(
        model_registry_pb2.GetLatestVersions,
        raw_resp_snake,
        response_proto=model_registry_pb2.GetLatestVersions.Response(),
    )
    assert "model_versions" in resp_snake
    assert "model_version" not in resp_snake["model_versions"][0]
    assert resp_snake["model_versions"][0]["name"] == "m2"
    assert resp_snake["model_versions"][0]["version"] == "3"


# ---------------------------------------------------------------------------
# Error Mapping & AIP-193
# ---------------------------------------------------------------------------


def test_call_endpoint_aip193_error_raised() -> None:
    store = GcpModelRegistryRestStore(_GCP_REGISTRY_URI)
    mock_resp = _make_mock_response(
        404,
        {
            "error": {
                "code": 404,
                "message": "Model 'not-found' not found",
                "status": "NOT_FOUND",
            }
        },
    )
    with unittest.mock.patch(
        "mlflow.utils.rest_utils.http_request", return_value=mock_resp
    ):
        with pytest.raises(GcpMlflowApiError) as exc_info:
            store._call_endpoint(
                model_registry_pb2.GetRegisteredModel,
                json_body=json.dumps({"name": "not-found"}),
            )
        assert exc_info.value.http_status == 404
        assert exc_info.value.error_code == "RESOURCE_DOES_NOT_EXIST"


def test_call_endpoint_204_no_content() -> None:
    store = GcpModelRegistryRestStore(_GCP_REGISTRY_URI)
    mock_resp = requests.Response()
    mock_resp.status_code = 204
    mock_resp._content = b""

    with unittest.mock.patch(
        "mlflow.utils.rest_utils.http_request", return_value=mock_resp
    ):
        proto = store._call_endpoint(
            model_registry_pb2.DeleteRegisteredModel,
            json_body='{"name": "m1"}',
        )
        assert proto is not None


# ---------------------------------------------------------------------------
# ensure_registry_store_registered
# ---------------------------------------------------------------------------


def test_ensure_registry_store_registered(monkeypatch: pytest.MonkeyPatch) -> None:
    import mlflow.tracking._model_registry.utils as reg_utils

    registry = reg_utils._get_store_registry()
    monkeypatch.setattr(registry, "_registry", dict(registry._registry))
    assert ensure_registry_store_registered() is True
    assert registry._registry.get("https") == build_registry_store


# ---------------------------------------------------------------------------
# PEP 562 __getattr__ and lazy loading
# ---------------------------------------------------------------------------


def test_pep562_getattr() -> None:
    store_cls = rs_mod.GcpModelRegistryRestStore
    assert store_cls is rs_mod._get_registry_store_cls()

    with pytest.raises(AttributeError):
        _ = rs_mod.NonExistentAttribute


def test_lazy_import_isolation() -> None:
    cmd = [
        sys.executable,
        "-c",
        (
            "import sys; "
            "import google_cloud_managed_mlflow; "
            "assert 'mlflow' not in sys.modules, 'mlflow was eagerly imported'; "
            "print('OK')"
        ),
    ]
    repo_root = str(Path(__file__).resolve().parent.parent)
    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        check=True,
        env={**os.environ, "PYTHONPATH": repo_root},
    )
    assert proc.stdout.strip() == "OK"


# ---------------------------------------------------------------------------
# Direct Field Normalizers and Download URI
# ---------------------------------------------------------------------------


def test_ensure_registered_model_fields_all_branches() -> None:
    d = {
        "name": "projects/p/locations/l/registries/r/registeredModels/my-model-id",
        "displayName": "my-display-model",
        "creationTimestamp": "12345",
        "lastUpdatedTimestamp": "67890",
        "tags": {"key1": "val1"},
    }
    normed = _ensure_registered_model_fields(d)
    assert normed["name"] == "my-model-id"
    assert "displayName" not in normed
    assert "display_name" not in normed
    assert normed["creation_timestamp"] == 12345
    assert normed["last_updated_timestamp"] == 67890
    assert normed["tags"] == [{"key": "key1", "value": "val1"}]


def test_ensure_model_version_fields_all_branches() -> None:
    d = {
        "modelId": "my-model-id",
        "modelVersionId": "5",
        "creationTimestamp": 100000,
        "lastUpdatedTimestamp": 200000,
        "tags": {"t": "v"},
        "aliases": {"champ": "5"},
    }
    normed = _ensure_model_version_fields(d)
    assert normed["name"] == "my-model-id"
    assert normed["version"] == "5"
    assert normed["creation_timestamp"] == 100000
    assert normed["last_updated_timestamp"] == 200000
    assert normed["status"] == "READY"
    assert normed["current_stage"] == "None"
    assert normed["tags"] == [{"key": "t", "value": "v"}]
    assert normed["aliases"] == ["champ"]


def test_normalize_get_model_version_download_uri() -> None:
    raw = {"artifactUri": "gs://bucket/model/path"}
    resp = normalize_registry_response(
        model_registry_pb2.GetModelVersionDownloadUri,
        raw,
        response_proto=model_registry_pb2.GetModelVersionDownloadUri.Response(),
    )
    assert resp["artifact_uri"] == "gs://bucket/model/path"


def test_reset_registry_store_cache() -> None:
    orig = rs_mod._REGISTRY_STORE_CLS
    try:
        cls1 = rs_mod._get_registry_store_cls()
        assert rs_mod._REGISTRY_STORE_CLS is cls1
        reset_registry_store_cache()
        assert rs_mod._REGISTRY_STORE_CLS is None
        cls2 = rs_mod._get_registry_store_cls()
        assert cls2 is not None
        assert rs_mod._REGISTRY_STORE_CLS is cls2
    finally:
        rs_mod._REGISTRY_STORE_CLS = orig


def test_normalize_last_updated_time_precedence_over_update_time() -> None:
    raw_gcp_api = {
        "name": (
            "projects/my-project/locations/us-central1/registries/my-registry"
            "/registeredModels/my-model"
        ),
        "createTime": "2026-11-10T12:00:00.000Z",
        "updateTime": "2026-11-10T12:05:00.000Z",
        "lastUpdatedTime": "2026-11-10T12:10:00.000Z",
    }
    resp = normalize_registry_response(
        model_registry_pb2.GetRegisteredModel,
        raw_gcp_api,
        response_proto=model_registry_pb2.GetRegisteredModel.Response(),
    )
    rm = resp["registered_model"]
    # 12:10:00.000Z is 1794312600000, 12:05:00.000Z is 1794312300000
    assert rm["last_updated_timestamp"] == 1794312600000


def test_delete_endpoints_validation_and_stripping() -> None:
    store = GcpModelRegistryRestStore(_GCP_REGISTRY_URI)

    # Missing model name raises
    with pytest.raises(MlflowException, match="Registered model name is required"):
        store._dispatch_delete_endpoint("DeleteRegisteredModel", {})

    # Strip /registeredModels/ prefix
    ep, params = store._dispatch_delete_endpoint(
        "DeleteRegisteredModel",
        {
            "name": (
                "projects/my-project/locations/us-central1/registries/my-registry"
                "/registeredModels/my-model"
            )
        },
    )
    assert ep == "/registeredModels/my-model"
    assert params is None

    # DeleteModelVersion missing version raises
    with pytest.raises(MlflowException, match="Model version is required"):
        store._dispatch_delete_endpoint("DeleteModelVersion", {"name": "my-model"})

    # DeleteRegisteredModelTag missing tag key raises
    with pytest.raises(MlflowException, match="Tag key is required"):
        store._dispatch_delete_endpoint(
            "DeleteRegisteredModelTag", {"name": "my-model"}
        )

    # DeleteModelVersionTag missing version raises
    with pytest.raises(MlflowException, match="Model version is required"):
        store._dispatch_delete_endpoint(
            "DeleteModelVersionTag", {"name": "my-model", "key": "k"}
        )

    # DeleteRegisteredModelAlias missing alias raises
    with pytest.raises(MlflowException, match="Alias is required"):
        store._dispatch_delete_endpoint(
            "DeleteRegisteredModelAlias", {"name": "my-model"}
        )


def test_get_and_update_registered_model_strip_prefix() -> None:
    store = GcpModelRegistryRestStore(_GCP_REGISTRY_URI)
    full_name = (
        "projects/my-project/locations/us-central1/registries/my-registry"
        "/registeredModels/my-model"
    )

    ep, method, params, body = store._dispatch_request(
        "GetRegisteredModel",
        model_registry_pb2.GetRegisteredModel,
        {"name": full_name},
    )
    assert ep == "/registeredModels/my-model"
    assert method == "GET"

    ep, method, params, body = store._dispatch_request(
        "UpdateRegisteredModel",
        model_registry_pb2.UpdateRegisteredModel,
        {"name": full_name, "description": "desc"},
    )
    assert ep == "/registeredModels/my-model"
    assert method == "PATCH"
    assert params == {"update_mask": "description"}
    assert body == {"description": "desc"}


def test_ensure_model_version_fields_unwraps_nested_model_version() -> None:
    nested = {
        "modelVersion": {
            "name": "my-model",
            "version": "1",
            "createTime": "2026-11-10T12:00:00.000Z",
        }
    }
    normed = _ensure_model_version_fields(nested)
    assert normed["name"] == "my-model"
    assert normed["version"] == "1"
    assert normed["status"] == "READY"


def test_create_model_version_honors_explicit_source_namespace() -> None:
    store = GcpModelRegistryRestStore(_GCP_REGISTRY_URI)
    payload = {
        "name": "my-model",
        "source": "gs://my-bucket/model",
        "source_namespace": (
            "projects/another-project/locations/us-central1/namespaces/explicit-ns"
        ),
    }
    ep, method, params, body = store._dispatch_request(
        "CreateModelVersion",
        model_registry_pb2.CreateModelVersion,
        payload,
    )
    assert ep == "/api/2.0/mlflow/model-versions/create"
    assert method == "POST"
    assert body is not None
    assert (
        body["source_namespace"]
        == "projects/another-project/locations/us-central1/namespaces/explicit-ns"
    )


def test_create_model_version_explicit_source_namespace_cross_location_raises() -> None:
    store = GcpModelRegistryRestStore(_GCP_REGISTRY_URI)
    payload = {
        "name": "my-model",
        "source": "gs://my-bucket/model",
        "source_namespace": (
            "projects/another-project/locations/europe-west1/namespaces/explicit-ns"
        ),
    }
    with pytest.raises(
        MlflowException, match="Cross-location model registration is not supported"
    ):
        store._dispatch_request(
            "CreateModelVersion",
            model_registry_pb2.CreateModelVersion,
            payload,
        )


# ---------------------------------------------------------------------------
# Prompt filter sanitization, URL quoting, and missing RPC wire tests
# ---------------------------------------------------------------------------


def test_sanitize_prompt_filter_variations() -> None:
    # 1. Stripping negative exclusion clauses
    assert _sanitize_prompt_filter("tag.`mlflow.prompt.is_prompt` != 'true'") == ""
    assert _sanitize_prompt_filter("tags.`mlflow.prompt.is_prompt` != 'true'") == ""
    assert _sanitize_prompt_filter("tags.mlflow.prompt.is_prompt != 'true'") == ""
    assert _sanitize_prompt_filter('tags.`mlflow.prompt.is_prompt` != "true"') == ""
    assert (
        _sanitize_prompt_filter(
            "name = 'm1' AND tag.`mlflow.prompt.is_prompt` != 'true'"
        )
        == "name = 'm1'"
    )
    assert (
        _sanitize_prompt_filter(
            "tag.`mlflow.prompt.is_prompt` != 'true' AND name = 'm1'"
        )
        == "name = 'm1'"
    )
    assert (
        _sanitize_prompt_filter(
            "name = 'm1' OR tag.`mlflow.prompt.is_prompt` != 'true'"
        )
        == "name = 'm1'"
    )
    assert (
        _sanitize_prompt_filter(
            "name = 'm1' AND tag.`mlflow.prompt.is_prompt` != 'true' "
            "AND tag.env = 'prod'"
        )
        == "name = 'm1' AND tag.env = 'prod'"
    )

    # 2. Rejecting positive prompt queries with FEATURE_DISABLED
    with pytest.raises(MlflowException) as exc_info1:
        _sanitize_prompt_filter("tag.`mlflow.prompt.is_prompt` = 'true'")
    assert exc_info1.value.error_code == "FEATURE_DISABLED"

    with pytest.raises(MlflowException) as exc_info2:
        _sanitize_prompt_filter("tags.`mlflow.prompt.is_prompt` = 'true'")
    assert exc_info2.value.error_code == "FEATURE_DISABLED"

    with pytest.raises(MlflowException) as exc_info3:
        _sanitize_prompt_filter(
            "name = 'm1' AND tag.`mlflow.prompt.is_prompt` = 'true'"
        )
    assert exc_info3.value.error_code == "FEATURE_DISABLED"


def test_sanitize_and_quote_model_name_valid_and_hostile() -> None:
    # Valid model names
    assert _sanitize_and_quote_model_name("my-model") == "my-model"
    assert _sanitize_and_quote_model_name("model with space") == "model%20with%20space"
    assert _sanitize_and_quote_model_name("model/subpath") == "model%2Fsubpath"

    # Full resource path extraction
    full_path = "projects/p/locations/l/registries/r/registeredModels/extracted-model"
    assert _sanitize_and_quote_model_name(full_path) == "extracted-model"

    # Hostile path traversal segments
    for hostile in ("../traversal", "foo/../bar", "..", ".", "./foo", "foo/."):
        with pytest.raises(MlflowException) as exc_info:
            _sanitize_and_quote_model_name(hostile)
        assert exc_info.value.error_code == "INVALID_PARAMETER_VALUE"

    # Hostile URL control characters
    for invalid in ("foo?param=1", "foo#fragment", "foo\\bar", "foo\x00bar"):
        with pytest.raises(MlflowException) as exc_info:
            _sanitize_and_quote_model_name(invalid)
        assert exc_info.value.error_code == "INVALID_PARAMETER_VALUE"

    # Empty or whitespace names
    for empty in ("", "   ", None):
        with pytest.raises(MlflowException) as exc_info:
            _sanitize_and_quote_model_name(empty)
        assert exc_info.value.error_code == "INVALID_PARAMETER_VALUE"


def test_dispatch_update_registered_model_description_none_vs_provided() -> None:
    store = GcpModelRegistryRestStore(_GCP_REGISTRY_URI)

    # When description is None: sends GET to read model without clearing description
    ep, method, params, body = store._dispatch_request(
        "UpdateRegisteredModel",
        model_registry_pb2.UpdateRegisteredModel,
        {"name": "my-model", "description": None},
    )
    assert ep == "/registeredModels/my-model"
    assert method == "GET"
    assert params is None
    assert body is None

    # When description is provided: sends PATCH with update_mask=description
    ep, method, params, body = store._dispatch_request(
        "UpdateRegisteredModel",
        model_registry_pb2.UpdateRegisteredModel,
        {"name": "my-model", "description": "new description"},
    )
    assert ep == "/registeredModels/my-model"
    assert method == "PATCH"
    assert params == {"update_mask": "description"}
    assert body == {"description": "new description"}


def test_build_registry_store_with_namespace_uri_raises() -> None:
    namespace_uri = (
        "https://mlflow.googleapis.com/v1/projects/my-project"
        "/locations/us-central1/namespaces/my-namespace"
    )
    with pytest.raises(MlflowException) as exc_info:
        build_registry_store(namespace_uri)
    assert "tracking namespace URI" in str(exc_info.value)
    assert "Please configure MLFLOW_REGISTRY_URI" in str(exc_info.value)


def test_wire_dispatch_additional_registry_rpcs() -> None:
    store = GcpModelRegistryRestStore(_GCP_REGISTRY_URI)

    # 1. RenameRegisteredModel
    ep, method, params, body = store._dispatch_request(
        "RenameRegisteredModel",
        model_registry_pb2.RenameRegisteredModel,
        {"name": "old-name", "new_name": "new-name"},
    )
    assert ep == "/api/2.0/mlflow/registered-models/rename"
    assert method == "POST"
    assert body == {"name": "old-name", "new_name": "new-name"}

    # 2. TransitionModelVersionStage
    ep, method, params, body = store._dispatch_request(
        "TransitionModelVersionStage",
        model_registry_pb2.TransitionModelVersionStage,
        {"name": "m1", "version": "1", "stage": "Production"},
    )
    assert ep == "/api/2.0/mlflow/model-versions/transition-stage"
    assert method == "POST"
    assert body == {"name": "m1", "version": "1", "stage": "Production"}

    # 3. GetModelVersionByAlias
    ep, method, params, body = store._dispatch_request(
        "GetModelVersionByAlias",
        model_registry_pb2.GetModelVersionByAlias,
        {"name": "m1", "alias": "champion"},
    )
    assert ep == "/api/2.0/mlflow/registered-models/alias"
    assert method == "GET"
    assert params == {"name": "m1", "alias": "champion"}

    # 4. SetRegisteredModelAlias
    ep, method, params, body = store._dispatch_request(
        "SetRegisteredModelAlias",
        model_registry_pb2.SetRegisteredModelAlias,
        {"name": "m1", "alias": "champion", "version": "1"},
    )
    assert ep == "/api/2.0/mlflow/registered-models/alias"
    assert method == "POST"
    assert body == {"name": "m1", "alias": "champion", "version": "1"}

    # 5. DeleteRegisteredModelAlias
    ep, method, params, body = store._dispatch_request(
        "DeleteRegisteredModelAlias",
        model_registry_pb2.DeleteRegisteredModelAlias,
        {"name": "m1", "alias": "champion"},
    )
    assert ep == "/api/2.0/mlflow/registered-models/alias"
    assert method == "DELETE"
    assert params == {"name": "m1", "alias": "champion"}

    # 6. GetModelVersionDownloadUri
    ep, method, params, body = store._dispatch_request(
        "GetModelVersionDownloadUri",
        model_registry_pb2.GetModelVersionDownloadUri,
        {"name": "m1", "version": "1"},
    )
    assert ep == "/api/2.0/mlflow/model-versions/get-download-uri"
    assert method == "GET"
    assert params == {"name": "m1", "version": "1"}

    # 7. UpdateModelVersion
    ep, method, params, body = store._dispatch_request(
        "UpdateModelVersion",
        model_registry_pb2.UpdateModelVersion,
        {"name": "m1", "version": "1", "description": "v1 desc"},
    )
    assert ep == "/api/2.0/mlflow/model-versions/update"
    assert method == "PATCH"
    assert body == {"name": "m1", "version": "1", "description": "v1 desc"}

    # 8. GetLatestVersions
    ep, method, params, body = store._dispatch_request(
        "GetLatestVersions",
        model_registry_pb2.GetLatestVersions,
        {"name": "m1", "stages": ["Production"]},
    )
    assert ep == "/api/2.0/mlflow/registered-models/get-latest-versions"
    assert method in ("GET", "POST")


def test_normalize_registry_response_assigns_subdicts_back() -> None:
    # Test that nested model_version is reassigned properly
    raw_mv = {
        "model_version": {
            "name": (
                "projects/p/locations/l/registries/r/registeredModels/m1/versions/1"
            ),
            "version": "1",
            "creationTimestamp": "1000",
        }
    }
    normalized_mv = normalize_registry_response(
        model_registry_pb2.GetModelVersion, raw_mv
    )
    assert "model_version" in normalized_mv
    assert normalized_mv["model_version"]["name"] == "m1"
    assert normalized_mv["model_version"]["version"] == "1"
    assert normalized_mv["model_version"]["creation_timestamp"] == 1000

    # Test that nested registered_model is reassigned properly
    raw_rm = {
        "registered_model": {
            "name": "projects/p/locations/l/registries/r/registeredModels/m1",
            "creationTimestamp": "2000",
        }
    }
    normalized_rm = normalize_registry_response(
        model_registry_pb2.GetRegisteredModel, raw_rm
    )
    assert "registered_model" in normalized_rm
    assert normalized_rm["registered_model"]["name"] == "m1"
    assert normalized_rm["registered_model"]["creation_timestamp"] == 2000
