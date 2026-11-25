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

"""Tracking REST store for Google Cloud Managed MLflow."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from mlflow.store.tracking.rest_store import RestStore

    from google_cloud_managed_mlflow.uri import ParsedGcpUri

    class GcpTrackingRestStore(RestStore):
        store_uri: str
        artifact_uri: str | None
        parsed_uri: ParsedGcpUri
        namespace_resource: str
        base_api_url: str
        endpoint_url: str

        def __init__(
            self,
            store_uri: str | None = None,
            artifact_uri: str | None = None,
            **kwargs: Any,
        ) -> None: ...
        def get_host_creds(self) -> Any: ...
        def search_datasets(self, *args: Any, **kwargs: Any) -> Any: ...
        def _call_endpoint(
            self,
            api: Any,
            json_body: str | dict[str, Any] | None = None,
            endpoint: str | None = None,
            retry_timeout_seconds: int | None = None,
            response_proto: Any = None,
            **kwargs: Any,
        ) -> Any: ...


__all__ = [
    "GcpTrackingRestStore",
    "build_tracking_store",
    "ensure_tracking_store_registered",
    "normalize_tracking_response",
    "reset_tracking_store_cache",
]

_TRACKING_STORE_CLS: type[Any] | None = None

_PASSTHROUGH_APIS = frozenset(
    {
        "GetMetricHistory",
        "GetMetricHistoryBulkInterval",
        "SearchDatasets",
        "LogMetric",
        "LogParam",
        "LogBatch",
        "SetTag",
        "DeleteTag",
        "SetExperimentTag",
        "UpdateExperiment",
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
    }
)

_ALLOWED_HTTP_METHODS = frozenset({"GET", "POST", "PATCH", "DELETE", "PUT"})


def reset_tracking_store_cache() -> None:
    """Reset class caches for testing isolation."""
    global _TRACKING_STORE_CLS
    _TRACKING_STORE_CLS = None


def _ensure_run_info_fields(info_dict: dict[str, Any]) -> dict[str, Any]:
    """Normalize and supply defaults for RunInfo fields."""
    if "artifactLocation" in info_dict:
        val = info_dict.pop("artifactLocation")
        if "artifact_uri" not in info_dict:
            info_dict["artifact_uri"] = val
    if "artifactUri" in info_dict:
        val = info_dict.pop("artifactUri")
        if "artifact_uri" not in info_dict:
            info_dict["artifact_uri"] = val
    if "artifact_location" in info_dict and "artifact_uri" not in info_dict:
        info_dict["artifact_uri"] = info_dict.pop("artifact_location")

    if "runId" in info_dict and "run_id" not in info_dict:
        info_dict["run_id"] = info_dict.pop("runId")
    if "runName" in info_dict and "run_name" not in info_dict:
        info_dict["run_name"] = info_dict.pop("runName")
    if "startTime" in info_dict and "start_time" not in info_dict:
        info_dict["start_time"] = info_dict.pop("startTime")
    if "endTime" in info_dict and "end_time" not in info_dict:
        info_dict["end_time"] = info_dict.pop("endTime")
    if "lifecycleStage" in info_dict and "lifecycle_stage" not in info_dict:
        info_dict["lifecycle_stage"] = info_dict.pop("lifecycleStage")
    if "userId" in info_dict and "user_id" not in info_dict:
        info_dict["user_id"] = info_dict.pop("userId")
    if "experimentId" in info_dict and "experiment_id" not in info_dict:
        info_dict["experiment_id"] = info_dict.pop("experimentId")

    info_dict.setdefault("lifecycle_stage", "active")
    info_dict.setdefault("start_time", 0)
    info_dict.setdefault("user_id", "")

    for key in ("start_time", "end_time"):
        val = info_dict.get(key)
        if isinstance(val, str):
            if val.isdigit():
                info_dict[key] = int(val)
            else:
                from google_cloud_managed_mlflow.uri import parse_rfc3339_ms

                info_dict[key] = parse_rfc3339_ms(val)

    return info_dict


def _ensure_experiment_fields(exp_dict: dict[str, Any]) -> dict[str, Any]:
    """Normalize and supply defaults for Experiment fields."""
    if "artifactLocation" in exp_dict:
        val = exp_dict.pop("artifactLocation")
        if "artifact_location" not in exp_dict:
            exp_dict["artifact_location"] = val
    if "artifact_uri" in exp_dict and "artifact_location" not in exp_dict:
        exp_dict["artifact_location"] = exp_dict.pop("artifact_uri")
    if "artifactUri" in exp_dict and "artifact_location" not in exp_dict:
        exp_dict["artifact_location"] = exp_dict.pop("artifactUri")
    if "lifecycleStage" in exp_dict and "lifecycle_stage" not in exp_dict:
        exp_dict["lifecycle_stage"] = exp_dict.pop("lifecycleStage")
    if "experimentId" in exp_dict and "experiment_id" not in exp_dict:
        exp_dict["experiment_id"] = str(exp_dict.pop("experimentId"))

    if "creationTime" in exp_dict and "creation_time" not in exp_dict:
        exp_dict["creation_time"] = exp_dict.pop("creationTime")
    elif "createTime" in exp_dict and "creation_time" not in exp_dict:
        exp_dict["creation_time"] = exp_dict.pop("createTime")
    elif "create_time" in exp_dict and "creation_time" not in exp_dict:
        exp_dict["creation_time"] = exp_dict.pop("create_time")

    if "lastUpdateTime" in exp_dict and "last_update_time" not in exp_dict:
        exp_dict["last_update_time"] = exp_dict.pop("lastUpdateTime")
    elif "updateTime" in exp_dict and "last_update_time" not in exp_dict:
        exp_dict["last_update_time"] = exp_dict.pop("updateTime")
    elif "update_time" in exp_dict and "last_update_time" not in exp_dict:
        exp_dict["last_update_time"] = exp_dict.pop("update_time")

    for key in ("creation_time", "last_update_time"):
        val = exp_dict.get(key)
        if isinstance(val, str):
            if val.isdigit():
                exp_dict[key] = int(val)
            else:
                from google_cloud_managed_mlflow.uri import parse_rfc3339_ms

                exp_dict[key] = parse_rfc3339_ms(val)

    exp_dict.setdefault("lifecycle_stage", "active")

    resource_name = exp_dict.get("name", "")
    if "experiment_id" not in exp_dict or not exp_dict["experiment_id"]:
        if isinstance(resource_name, str) and "/experiments/" in resource_name:
            exp_id = resource_name.split("/experiments/")[-1].split("/")[0]
            if exp_id:
                exp_dict["experiment_id"] = exp_id

    dn = exp_dict.pop("displayName", None)
    dn2 = exp_dict.pop("display_name", None)
    display_name = dn or dn2
    if display_name:
        exp_dict["name"] = display_name

    if "experiment_id" in exp_dict:
        exp_dict["experiment_id"] = str(exp_dict["experiment_id"])

    return exp_dict


def _ensure_logged_model_info_fields(info_dict: dict[str, Any]) -> dict[str, Any]:
    """Normalize LoggedModelInfo fields."""
    if "artifactLocation" in info_dict:
        val = info_dict.pop("artifactLocation")
        if "artifact_uri" not in info_dict:
            info_dict["artifact_uri"] = val
    if "artifactUri" in info_dict:
        val = info_dict.pop("artifactUri")
        if "artifact_uri" not in info_dict:
            info_dict["artifact_uri"] = val
    if "artifact_location" in info_dict and "artifact_uri" not in info_dict:
        info_dict["artifact_uri"] = info_dict.pop("artifact_location")

    if "modelId" in info_dict and "model_id" not in info_dict:
        info_dict["model_id"] = info_dict.pop("modelId")
    if "experimentId" in info_dict and "experiment_id" not in info_dict:
        info_dict["experiment_id"] = info_dict.pop("experimentId")

    for key in (
        "creation_timestamp_ms",
        "creationTimestampMs",
        "last_updated_timestamp_ms",
        "lastUpdatedTimestampMs",
    ):
        val = info_dict.get(key)
        if isinstance(val, str):
            if val.isdigit():
                info_dict[key] = int(val)
            else:
                from google_cloud_managed_mlflow.uri import parse_rfc3339_ms

                info_dict[key] = parse_rfc3339_ms(val)

    return info_dict


def normalize_tracking_response(api: Any, js_dict: dict[str, Any]) -> dict[str, Any]:
    """Normalize REST response dictionary to match MLflow proto expectations."""
    if not isinstance(js_dict, dict):
        return js_dict

    api_name = getattr(api, "__name__", str(api))

    if api_name in _PASSTHROUGH_APIS:
        return js_dict

    if api_name in ("CreateRun", "GetRun"):
        if "run" in js_dict and isinstance(js_dict["run"], dict):
            run = js_dict["run"]
            if "info" in run and isinstance(run["info"], dict):
                _ensure_run_info_fields(run["info"])
            else:
                run["info"] = _ensure_run_info_fields(dict(run))
            run.setdefault("data", {})
        elif "info" in js_dict and isinstance(js_dict["info"], dict):
            _ensure_run_info_fields(js_dict["info"])
            js_dict.setdefault("data", {})
            return {"run": js_dict}
        else:
            info = _ensure_run_info_fields(dict(js_dict))
            return {"run": {"info": info, "data": {}}}
        return js_dict

    if api_name == "UpdateRun":
        if "run_info" in js_dict and isinstance(js_dict["run_info"], dict):
            _ensure_run_info_fields(js_dict["run_info"])
            return js_dict
        if "runInfo" in js_dict and isinstance(js_dict["runInfo"], dict):
            js_dict["run_info"] = _ensure_run_info_fields(js_dict.pop("runInfo"))
            return js_dict
        if "run" in js_dict and isinstance(js_dict["run"], dict):
            run = js_dict["run"]
            raw_info = run.get("info") or run.get("runInfo") or run.get("run_info")
            info_dict = dict(raw_info) if isinstance(raw_info, dict) else dict(run)
            return {"run_info": _ensure_run_info_fields(info_dict)}
        if "info" in js_dict and isinstance(js_dict["info"], dict):
            return {"run_info": _ensure_run_info_fields(js_dict["info"])}
        return {"run_info": _ensure_run_info_fields(dict(js_dict))}

    if api_name in ("GetExperiment", "GetExperimentByName"):
        if "experiment" in js_dict and isinstance(js_dict["experiment"], dict):
            _ensure_experiment_fields(js_dict["experiment"])
            return js_dict
        return {"experiment": _ensure_experiment_fields(dict(js_dict))}

    if api_name == "CreateExperiment":
        if "experiment_id" in js_dict:
            js_dict["experiment_id"] = str(js_dict["experiment_id"])
            return js_dict
        if "experimentId" in js_dict:
            return {"experiment_id": str(js_dict["experimentId"])}
        if "experiment" in js_dict and isinstance(js_dict["experiment"], dict):
            exp = js_dict["experiment"]
            _ensure_experiment_fields(exp)
            if "experiment_id" in exp:
                return {"experiment_id": str(exp["experiment_id"])}
        exp = _ensure_experiment_fields(dict(js_dict))
        if "experiment_id" in exp:
            return {"experiment_id": str(exp["experiment_id"])}
        return js_dict

    if api_name == "SearchRuns":
        if "runs" in js_dict and isinstance(js_dict["runs"], list):
            for idx, r in enumerate(js_dict["runs"]):
                if isinstance(r, dict):
                    if "info" in r and isinstance(r["info"], dict):
                        _ensure_run_info_fields(r["info"])
                        r.setdefault("data", {})
                    elif "run" in r and isinstance(r["run"], dict):
                        sub_r = r["run"]
                        if "info" in sub_r and isinstance(sub_r["info"], dict):
                            _ensure_run_info_fields(sub_r["info"])
                            sub_r.setdefault("data", {})
                        else:
                            sub_r = {
                                "info": _ensure_run_info_fields(dict(sub_r)),
                                "data": {},
                            }
                        js_dict["runs"][idx] = sub_r
                    else:
                        js_dict["runs"][idx] = {
                            "info": _ensure_run_info_fields(dict(r)),
                            "data": {},
                        }
        if "nextPageToken" in js_dict and "next_page_token" not in js_dict:
            js_dict["next_page_token"] = js_dict.pop("nextPageToken")
        return js_dict

    if api_name == "SearchExperiments":
        if "experiments" in js_dict and isinstance(js_dict["experiments"], list):
            for idx, exp in enumerate(js_dict["experiments"]):
                if isinstance(exp, dict):
                    if "experiment" in exp and isinstance(exp["experiment"], dict):
                        exp = exp["experiment"]
                    _ensure_experiment_fields(exp)
                    js_dict["experiments"][idx] = exp
        if "nextPageToken" in js_dict and "next_page_token" not in js_dict:
            js_dict["next_page_token"] = js_dict.pop("nextPageToken")
        return js_dict

    if api_name in (
        "CreateLoggedModel",
        "GetLoggedModel",
        "FinalizeLoggedModel",
        "SetLoggedModelTags",
    ):
        if "model" in js_dict and isinstance(js_dict["model"], dict):
            model = js_dict["model"]
            if "info" in model and isinstance(model["info"], dict):
                _ensure_logged_model_info_fields(model["info"])
            else:
                model["info"] = _ensure_logged_model_info_fields(dict(model))
            model.setdefault("data", {})
            return js_dict
        if "info" in js_dict and isinstance(js_dict["info"], dict):
            _ensure_logged_model_info_fields(js_dict["info"])
            js_dict.setdefault("data", {})
            return {"model": js_dict}
        info = _ensure_logged_model_info_fields(dict(js_dict))
        return {"model": {"info": info, "data": {}}}

    if api_name == "SearchLoggedModels":
        if "models" in js_dict and isinstance(js_dict["models"], list):
            for idx, m in enumerate(js_dict["models"]):
                if isinstance(m, dict):
                    if "info" in m and isinstance(m["info"], dict):
                        _ensure_logged_model_info_fields(m["info"])
                        m.setdefault("data", {})
                    elif "model" in m and isinstance(m["model"], dict):
                        sub_m = m["model"]
                        if "info" in sub_m and isinstance(sub_m["info"], dict):
                            _ensure_logged_model_info_fields(sub_m["info"])
                            sub_m.setdefault("data", {})
                        else:
                            sub_m = {
                                "info": _ensure_logged_model_info_fields(dict(sub_m)),
                                "data": {},
                            }
                        js_dict["models"][idx] = sub_m
                    else:
                        js_dict["models"][idx] = {
                            "info": _ensure_logged_model_info_fields(dict(m)),
                            "data": {},
                        }
        if "nextPageToken" in js_dict and "next_page_token" not in js_dict:
            js_dict["next_page_token"] = js_dict.pop("nextPageToken")
        return js_dict

    return js_dict


def _build_tracking_store_class() -> type[Any]:
    """Lazily construct and return the GcpTrackingRestStore class."""
    import mlflow
    import mlflow.store.tracking.rest_store as rest_store_mod
    from mlflow.exceptions import MlflowException
    from mlflow.protos.databricks_pb2 import (
        FEATURE_DISABLED,
        INVALID_PARAMETER_VALUE,
    )
    from mlflow.store.tracking.rest_store import RestStore
    from mlflow.utils import proto_json_utils, rest_utils
    from mlflow.utils.rest_utils import MlflowHostCreds

    from google_cloud_managed_mlflow import auth, uri

    class GcpTrackingRestStore(RestStore):
        """Tracking REST store implementation for Google Cloud Managed MLflow."""

        def __init__(
            self,
            store_uri: str | None = None,
            artifact_uri: str | None = None,
            **kwargs: Any,
        ) -> None:
            resolved_uri = store_uri or mlflow.get_tracking_uri()
            if not resolved_uri:
                raise MlflowException(
                    "Google Cloud tracking store URI must be non-empty.",
                    error_code=INVALID_PARAMETER_VALUE,
                )
            parsed = uri.parse_gcp_uri(resolved_uri)
            if parsed.resource_type != "namespaces":
                raise MlflowException(
                    f"Tracking store URI must reference a 'namespaces' resource, "
                    f"got {parsed.resource_type!r} in {resolved_uri!r}.",
                    error_code=INVALID_PARAMETER_VALUE,
                )
            self.store_uri = resolved_uri
            self.artifact_uri = artifact_uri
            self.parsed_uri = parsed
            self.namespace_resource = parsed.resource_name
            self.base_api_url = parsed.base_api_url
            self.endpoint_url = parsed.full_endpoint_url
            auth.harden_mlflow_request_session()
            super_init: Any = super().__init__
            super_init(get_host_creds=self.get_host_creds)

        def get_host_creds(self) -> Any:
            """Validate transport security and return Google Cloud host credentials."""
            auth.validate_transport_security(self.endpoint_url)
            return MlflowHostCreds(
                host=self.endpoint_url,
                auth="google-cloud-auth",
                ignore_tls_verification=False,
            )

        def search_datasets(
            self,
            experiment_ids: list[str] | None = None,
            filter_string: str | None = None,
            max_results: int = 1000,
            order_by: list[str] | None = None,
            page_token: str | None = None,
        ) -> Any:
            """Search evaluation datasets (unsupported in Managed MLflow)."""
            msg = (
                "Evaluation dataset search (/api/3.0/mlflow/datasets/search) is not "
                "supported by Google Cloud Managed Service for MLflow namespaces."
            )
            raise MlflowException(msg, error_code=FEATURE_DISABLED)

        def _call_endpoint(
            self,
            api: Any,
            json_body: str | dict[str, Any] | None = None,
            endpoint: str | None = None,
            retry_timeout_seconds: int | None = None,
            response_proto: Any = None,
            **kwargs: Any,
        ) -> Any:
            """Invoke REST endpoint with security validation and response mapping."""
            if hasattr(self, "_V3_APIS") and api in self._V3_APIS:
                method_to_info = getattr(self, "_V3_METHOD_TO_INFO", {})
            else:
                method_to_info = getattr(self, "_METHOD_TO_INFO", None) or getattr(
                    rest_store_mod, "_METHOD_TO_INFO", {}
                )
            if endpoint:
                if api in method_to_info:
                    _, method = method_to_info[api]
                else:
                    method = kwargs.get("method", "GET")
            else:
                endpoint, method = method_to_info[api]

            if "method" in kwargs:
                method = kwargs["method"]

            if not isinstance(endpoint, str) or not endpoint.startswith(
                ("/api/2.0/mlflow/", "/api/3.0/mlflow/")
            ):
                msg = (
                    f"Invalid MLflow endpoint path: {endpoint!r}. "
                    "Must start with '/api/2.0/mlflow/' or '/api/3.0/mlflow/'."
                )
                raise MlflowException(msg, error_code=INVALID_PARAMETER_VALUE)
            uri.validate_path_segments(endpoint, context="endpoint")
            if "?" in endpoint or "#" in endpoint or ".." in endpoint:
                msg = (
                    "Invalid MLflow endpoint path contains forbidden delimiter "
                    f"or traversal characters: {endpoint!r}."
                )
                raise MlflowException(msg, error_code=INVALID_PARAMETER_VALUE)

            method = method.upper() if isinstance(method, str) else ""
            if method not in _ALLOWED_HTTP_METHODS:
                raise MlflowException(
                    f"Unsupported HTTP method: {method!r}.",
                    error_code=INVALID_PARAMETER_VALUE,
                )

            payload = (
                json.loads(json_body)
                if isinstance(json_body, str) and json_body
                else json_body
            )

            call_kwargs: dict[str, Any] = {
                "host_creds": self.get_host_creds(),
                "endpoint": endpoint,
                "method": method,
                "allow_redirects": False,
            }
            if retry_timeout_seconds is not None:
                call_kwargs["retry_timeout_seconds"] = retry_timeout_seconds
            if "extra_headers" in kwargs:
                call_kwargs["extra_headers"] = kwargs["extra_headers"]

            if method == "GET":
                call_kwargs["params"] = payload
            elif method == "DELETE":
                if payload:
                    call_kwargs["params"] = payload
            else:
                call_kwargs["json"] = payload

            response = rest_utils.http_request(**call_kwargs)

            if response.status_code >= 400:
                uri.raise_for_aip193_error(response)

            if response_proto is None:
                response_proto = api.Response()

            if response.status_code == 204 or not (response.text or "").strip():
                return response_proto

            response = rest_utils.verify_rest_response(response, endpoint)

            js_dict = json.loads(response.text)
            js_dict = normalize_tracking_response(api, js_dict)
            proto_json_utils.parse_dict(js_dict=js_dict, message=response_proto)
            return response_proto

    return GcpTrackingRestStore


def _get_tracking_store_cls() -> type[Any]:
    """Retrieve or lazily initialize the GcpTrackingRestStore class."""
    global _TRACKING_STORE_CLS
    if _TRACKING_STORE_CLS is None:
        _TRACKING_STORE_CLS = _build_tracking_store_class()
    return _TRACKING_STORE_CLS


def build_tracking_store(
    store_uri: str,
    artifact_uri: str | None = None,
    **kwargs: Any,
) -> Any:
    """Construct an MLflow tracking store instance for the given URI.

    Delegates to GcpTrackingRestStore if store_uri is a Google Cloud tracking
    URI; falls back to stock MLflow RestStore otherwise.
    """
    from google_cloud_managed_mlflow.uri import is_gcp_tracking_uri

    if not is_gcp_tracking_uri(store_uri):
        from mlflow.tracking._tracking_service.utils import _get_rest_store

        return _get_rest_store(store_uri, artifact_uri=artifact_uri, **kwargs)

    from google_cloud_managed_mlflow.spark import install_spark_hooks

    install_spark_hooks()
    cls = _get_tracking_store_cls()
    return cls(store_uri=store_uri, artifact_uri=artifact_uri, **kwargs)


def ensure_tracking_store_registered() -> bool:
    """Ensure Google Cloud tracking store is registered with MLflow.

    Registers https under _tracking_store_registry if present, ensures
    Google Cloud auth provider is registered, and hardens the request session.
    """
    from google_cloud_managed_mlflow.spark import install_spark_hooks

    install_spark_hooks()
    try:
        from google_cloud_managed_mlflow import auth

        auth.ensure_auth_provider_registered()
        auth.harden_mlflow_request_session()

        import mlflow.tracking._tracking_service.utils as tr_utils

        registry = getattr(tr_utils, "_tracking_store_registry", None)
        if registry is None:
            return False
        registry.register("https", build_tracking_store)
        return True
    except Exception:
        return False


def __getattr__(name: str) -> Any:
    """PEP 562 lazy attribute access for GcpTrackingRestStore."""
    if name == "GcpTrackingRestStore":
        return _get_tracking_store_cls()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


from google_cloud_managed_mlflow.spark import install_spark_hooks  # noqa: E402

install_spark_hooks()
