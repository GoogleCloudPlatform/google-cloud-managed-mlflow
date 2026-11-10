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

"""Model Registry REST store for Google Cloud Managed MLflow."""

from __future__ import annotations

import json
import os
import re
import sys
import urllib.parse
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from mlflow.store.model_registry.rest_store import RestStore

    from google_cloud_managed_mlflow.uri import ParsedGcpUri

    class GcpModelRegistryRestStore(RestStore):
        store_uri: str
        tracking_uri: str | None
        parsed_registry_uri: ParsedGcpUri
        registry_resource: str
        base_api_url: str
        endpoint_url: str

        def __init__(
            self,
            store_uri: str | None = None,
            tracking_uri: str | None = None,
            **kwargs: Any,
        ) -> None: ...
        def get_host_creds(self) -> Any: ...
        def _resolve_source_namespace(
            self, explicit_namespace: str | None = None
        ) -> str: ...
        def _dispatch_delete_endpoint(
            self, api_name: str, payload: dict[str, Any]
        ) -> tuple[str, dict[str, Any] | None]: ...
        def _dispatch_create_registered_model(
            self, payload: dict[str, Any]
        ) -> tuple[str, str, dict[str, Any] | None, dict[str, Any] | None]: ...
        def _dispatch_request(
            self, api_name: str, api: Any, payload: dict[str, Any]
        ) -> tuple[str, str, dict[str, Any] | None, dict[str, Any] | None]: ...
        def _call_endpoint(
            self,
            api: Any,
            json_body: str | dict[str, Any] | None = None,
            call_all_endpoints: bool = False,
            extra_headers: dict[str, Any] | None = None,
            response_proto: Any = None,
            **kwargs: Any,
        ) -> Any: ...


__all__ = [
    "GcpModelRegistryRestStore",
    "build_registry_store",
    "ensure_registry_store_registered",
    "normalize_registry_response",
    "reset_registry_store_cache",
]

_CANONICAL_NS_RE = re.compile(
    r"^projects/(?P<project>[a-z0-9-.:]+)/locations/(?P<location>[a-z0-9-]+)"
    r"/namespaces/(?P<namespace>[a-zA-Z0-9_-]+)$"
)

_POSITIVE_PROMPT_FILTER_RE = re.compile(
    r"tags?\.`?mlflow\.prompt\.is_prompt`?\s*=\s*[\x27\x22]true[\x27\x22]",
    re.IGNORECASE,
)

_NEGATIVE_PROMPT_FILTER_RE = re.compile(
    r"tags?\.`?mlflow\.prompt\.is_prompt`?\s*!=\s*[\x27\x22](?:true|false)[\x27\x22]",
    re.IGNORECASE,
)


def _sanitize_prompt_filter(filter_str: str) -> str:
    """Strip synthetic MLflow 3.x prompt filter clauses while preserving AND/OR syntax.

    Raises:
        MlflowException: If a positive prompt query is detected, since Managed MLflow
            does not support Prompt Registry.
    """
    if not filter_str:
        return ""
    if _POSITIVE_PROMPT_FILTER_RE.search(filter_str):
        from mlflow.exceptions import MlflowException
        from mlflow.protos.databricks_pb2 import FEATURE_DISABLED

        raise MlflowException(
            "Prompt registry is not supported by Google Cloud Managed MLflow "
            "registries.",
            error_code=FEATURE_DISABLED,
        )

    cleaned = _NEGATIVE_PROMPT_FILTER_RE.sub("", filter_str)
    # Collapse multiple consecutive AND / OR operators
    cleaned = re.sub(
        r"\b(?:AND|OR)\s+(?:\s*(?:AND|OR)\b)+",
        "AND",
        cleaned,
        flags=re.IGNORECASE,
    )
    # Strip leading or trailing AND / OR
    cleaned = re.sub(r"^\s*(?:AND|OR)\s+", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s+(?:AND|OR)\s*$", "", cleaned, flags=re.IGNORECASE)
    # Normalize multiple whitespace to single space
    cleaned = re.sub(r"\s{2,}", " ", cleaned)
    return cleaned.strip()


def _sanitize_and_quote_model_name(raw_name: Any) -> str:
    """Validate and percent-encode model name for URL path construction."""
    from mlflow.exceptions import MlflowException
    from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE

    if not raw_name or not isinstance(raw_name, str):
        raise MlflowException(
            "Registered model name is required.",
            error_code=INVALID_PARAMETER_VALUE,
        )
    name = (
        raw_name.split("/registeredModels/")[-1]
        if "/registeredModels/" in raw_name
        else raw_name
    )
    if not name.strip():
        raise MlflowException(
            "Registered model name cannot be empty or whitespace.",
            error_code=INVALID_PARAMETER_VALUE,
        )
    # Check for path traversal segments
    if (
        name in (".", "..")
        or "/../" in name
        or name.startswith("../")
        or name.endswith("/..")
        or "/./" in name
        or name.startswith("./")
        or name.endswith("/.")
    ):
        raise MlflowException(
            f"Invalid registered model name '{raw_name}': contains path traversal "
            "segments.",
            error_code=INVALID_PARAMETER_VALUE,
        )
    # Check for invalid URL control characters
    if any(c in name for c in ("\x00", "\\", "#", "?")):
        raise MlflowException(
            f"Invalid registered model name '{raw_name}': contains invalid URL "
            "characters.",
            error_code=INVALID_PARAMETER_VALUE,
        )
    return urllib.parse.quote(name, safe="")


_REGISTRY_STORE_CLS: type[Any] | None = None

_PASSTHROUGH_REGISTRY_APIS = frozenset(
    {
        "DeleteRegisteredModel",
        "DeleteModelVersion",
        "DeleteRegisteredModelTag",
        "DeleteModelVersionTag",
        "DeleteRegisteredModelAlias",
        "SetRegisteredModelTag",
        "SetModelVersionTag",
        "SetRegisteredModelAlias",
    }
)

_BODYLESS_DELETE_APIS = frozenset(
    {
        "DeleteRegisteredModel",
        "DeleteModelVersion",
        "DeleteRegisteredModelTag",
        "DeleteModelVersionTag",
        "DeleteRegisteredModelAlias",
    }
)

_TIMESTAMP_FIELDS: dict[str, tuple[str, ...]] = {
    "creation_timestamp": ("createTime", "create_time", "creationTimestamp"),
    "last_updated_timestamp": (
        "lastUpdatedTime",
        "last_updated_time",
        "lastUpdatedTimestamp",
        "updateTime",
        "update_time",
    ),
}


def reset_registry_store_cache() -> None:
    """Reset class caches for testing isolation."""
    global _REGISTRY_STORE_CLS
    _REGISTRY_STORE_CLS = None


def _normalize_timestamps(
    entity_dict: dict[str, Any],
    timestamp_fields: dict[str, tuple[str, ...]],
) -> None:
    """Convert string/numeric timestamps to integer milliseconds."""
    from google_cloud_managed_mlflow.uri import parse_rfc3339_ms

    for target_field, source_fields in timestamp_fields.items():
        for key in source_fields:
            if key in entity_dict:
                val = entity_dict.pop(key)
                if target_field not in entity_dict:
                    entity_dict[target_field] = val

        val = entity_dict.get(target_field)
        if isinstance(val, str):
            try:
                entity_dict[target_field] = int(val)
            except ValueError:
                try:
                    entity_dict[target_field] = parse_rfc3339_ms(val)
                except Exception:
                    pass
        elif isinstance(val, (int, float)):
            entity_dict[target_field] = int(val)


def _normalize_annotations_to_tags(entity_dict: dict[str, Any]) -> None:
    """Convert annotations dict to list of key-value pairs and merge into tags."""
    if "annotations" in entity_dict:
        ann = entity_dict.pop("annotations")
        if isinstance(ann, dict):
            tags = [{"key": k, "value": v} for k, v in ann.items()]
            if "tags" in entity_dict and isinstance(entity_dict["tags"], list):
                entity_dict["tags"].extend(tags)
            else:
                entity_dict["tags"] = tags

    if "tags" in entity_dict and isinstance(entity_dict["tags"], dict):
        entity_dict["tags"] = [
            {"key": k, "value": v} for k, v in entity_dict["tags"].items()
        ]


def _ensure_registered_model_fields(model_dict: dict[str, Any]) -> dict[str, Any]:
    """Normalize and supply defaults for RegisteredModel fields."""
    name = model_dict.get("name", "")
    if "/registeredModels/" in name:
        model_dict["name"] = name.split("/registeredModels/")[-1]
    elif name.startswith("projects/"):
        model_dict["name"] = name.split("/")[-1]

    # Drop displayName / display_name to ensure resource ID is always preserved as name
    model_dict.pop("displayName", None)
    model_dict.pop("display_name", None)

    _normalize_timestamps(model_dict, _TIMESTAMP_FIELDS)
    _normalize_annotations_to_tags(model_dict)

    if "aliases" in model_dict:
        aliases = model_dict["aliases"]
        if isinstance(aliases, dict):
            model_dict["aliases"] = [
                {"alias": k, "version": str(v)} for k, v in aliases.items()
            ]

    if "latest_versions" in model_dict and isinstance(
        model_dict["latest_versions"], list
    ):
        model_dict["latest_versions"] = [
            _ensure_model_version_fields(v) if isinstance(v, dict) else v
            for v in model_dict["latest_versions"]
        ]
    elif "latestVersions" in model_dict and isinstance(
        model_dict["latestVersions"], list
    ):
        model_dict["latest_versions"] = [
            _ensure_model_version_fields(v) if isinstance(v, dict) else v
            for v in model_dict.pop("latestVersions")
        ]

    return model_dict


def _ensure_model_version_fields(mv_dict: dict[str, Any]) -> dict[str, Any]:
    """Normalize and supply defaults for ModelVersion fields."""
    if not isinstance(mv_dict, dict):
        return mv_dict

    if "model_version" in mv_dict and isinstance(mv_dict["model_version"], dict):
        mv_dict = dict(mv_dict["model_version"])
    elif "modelVersion" in mv_dict and isinstance(mv_dict["modelVersion"], dict):
        mv_dict = dict(mv_dict["modelVersion"])

    name = mv_dict.get("name", "")
    if "/registeredModels/" in name:
        sub = name.split("/registeredModels/")[-1]
        if "/versions/" in sub:
            mv_dict["name"] = sub.split("/versions/")[0]
        else:
            mv_dict["name"] = sub.split("/")[0]
    elif name.startswith("projects/"):
        mv_dict["name"] = name.split("/")[-1]
    elif not name:
        if "model_id" in mv_dict:
            mv_dict["name"] = mv_dict["model_id"]
        elif "modelId" in mv_dict:
            mv_dict["name"] = mv_dict["modelId"]
        elif "registered_model_id" in mv_dict:
            mv_dict["name"] = mv_dict["registered_model_id"]
        elif "registeredModelId" in mv_dict:
            mv_dict["name"] = mv_dict["registeredModelId"]

    if "version" in mv_dict and mv_dict["version"] is not None:
        mv_dict["version"] = str(mv_dict["version"])
    elif "model_version_id" in mv_dict:
        mv_dict["version"] = str(mv_dict.pop("model_version_id"))
    elif "modelVersionId" in mv_dict:
        mv_dict["version"] = str(mv_dict.pop("modelVersionId"))
    elif "/versions/" in name:
        v = name.split("/versions/")[-1].split("/")[0]
        if v:
            mv_dict["version"] = str(v)

    _normalize_timestamps(mv_dict, _TIMESTAMP_FIELDS)

    if not mv_dict.get("status"):
        mv_dict["status"] = "READY"

    if "current_stage" not in mv_dict or mv_dict["current_stage"] is None:
        mv_dict["current_stage"] = "None"

    _normalize_annotations_to_tags(mv_dict)

    if "aliases" in mv_dict:
        aliases = mv_dict["aliases"]
        if isinstance(aliases, dict):
            mv_dict["aliases"] = list(aliases.keys())
        elif isinstance(aliases, list):
            mv_dict["aliases"] = [
                a["alias"] if isinstance(a, dict) and "alias" in a else str(a)
                for a in aliases
            ]

    return mv_dict


def normalize_registry_response(
    api: Any, js_dict: dict[str, Any], response_proto: Any = None
) -> dict[str, Any]:
    """Normalize REST response to match MLflow model registry proto expectations."""
    if not isinstance(js_dict, dict):
        return js_dict

    api_name = getattr(api, "__name__", str(api))

    if api_name in _PASSTHROUGH_REGISTRY_APIS:
        return js_dict

    if api_name == "GetModelVersionDownloadUri":
        if "artifactUri" in js_dict and "artifact_uri" not in js_dict:
            js_dict["artifact_uri"] = js_dict.pop("artifactUri")
        elif "artifactLocation" in js_dict and "artifact_uri" not in js_dict:
            js_dict["artifact_uri"] = js_dict.pop("artifactLocation")
        elif "artifact_location" in js_dict and "artifact_uri" not in js_dict:
            js_dict["artifact_uri"] = js_dict.pop("artifact_location")
        return js_dict

    if api_name in (
        "CreateRegisteredModel",
        "GetRegisteredModel",
        "UpdateRegisteredModel",
        "RenameRegisteredModel",
    ):
        if "registered_model" in js_dict and isinstance(
            js_dict["registered_model"], dict
        ):
            js_dict["registered_model"] = _ensure_registered_model_fields(
                js_dict["registered_model"]
            )
            return js_dict
        if "registeredModel" in js_dict and isinstance(
            js_dict["registeredModel"], dict
        ):
            js_dict["registered_model"] = _ensure_registered_model_fields(
                js_dict.pop("registeredModel")
            )
            return js_dict
        model_dict = _ensure_registered_model_fields(dict(js_dict))
        return {"registered_model": model_dict}

    if api_name in (
        "CreateModelVersion",
        "GetModelVersion",
        "UpdateModelVersion",
        "TransitionModelVersionStage",
        "GetModelVersionByAlias",
    ):
        if "model_version" in js_dict and isinstance(js_dict["model_version"], dict):
            js_dict["model_version"] = _ensure_model_version_fields(
                js_dict["model_version"]
            )
            return js_dict
        if "modelVersion" in js_dict and isinstance(js_dict["modelVersion"], dict):
            js_dict["model_version"] = _ensure_model_version_fields(
                js_dict.pop("modelVersion")
            )
            return js_dict
        mv_dict = _ensure_model_version_fields(dict(js_dict))
        return {"model_version": mv_dict}

    if api_name == "SearchRegisteredModels":
        if "registered_models" in js_dict and isinstance(
            js_dict["registered_models"], list
        ):
            for idx, m in enumerate(js_dict["registered_models"]):
                if isinstance(m, dict):
                    if "registered_model" in m and isinstance(
                        m["registered_model"], dict
                    ):
                        m = m["registered_model"]
                    js_dict["registered_models"][idx] = _ensure_registered_model_fields(
                        m
                    )
        elif "registeredModels" in js_dict and isinstance(
            js_dict["registeredModels"], list
        ):
            raw_list = js_dict.pop("registeredModels")
            js_dict["registered_models"] = [
                _ensure_registered_model_fields(
                    m.get("registered_model", m) if isinstance(m, dict) else m
                )
                if isinstance(m, dict)
                else m
                for m in raw_list
            ]
        if "nextPageToken" in js_dict and "next_page_token" not in js_dict:
            js_dict["next_page_token"] = js_dict.pop("nextPageToken")
        return js_dict

    if api_name in ("SearchModelVersions", "GetLatestVersions"):
        if "model_versions" in js_dict and isinstance(js_dict["model_versions"], list):
            for idx, v in enumerate(js_dict["model_versions"]):
                if isinstance(v, dict):
                    if "model_version" in v and isinstance(v["model_version"], dict):
                        v = v["model_version"]
                    js_dict["model_versions"][idx] = _ensure_model_version_fields(v)
        elif "modelVersions" in js_dict and isinstance(js_dict["modelVersions"], list):
            raw_list = js_dict.pop("modelVersions")
            js_dict["model_versions"] = [
                _ensure_model_version_fields(
                    v.get("model_version", v) if isinstance(v, dict) else v
                )
                if isinstance(v, dict)
                else v
                for v in raw_list
            ]
        if "nextPageToken" in js_dict and "next_page_token" not in js_dict:
            js_dict["next_page_token"] = js_dict.pop("nextPageToken")
        return js_dict

    if "registered_model" in js_dict and isinstance(js_dict["registered_model"], dict):
        _ensure_registered_model_fields(js_dict["registered_model"])
    elif "model_version" in js_dict and isinstance(js_dict["model_version"], dict):
        _ensure_model_version_fields(js_dict["model_version"])
    elif "registered_models" in js_dict and isinstance(
        js_dict["registered_models"], list
    ):
        for idx, m in enumerate(js_dict["registered_models"]):
            if isinstance(m, dict):
                js_dict["registered_models"][idx] = _ensure_registered_model_fields(m)
    elif "model_versions" in js_dict and isinstance(js_dict["model_versions"], list):
        for idx, v in enumerate(js_dict["model_versions"]):
            if isinstance(v, dict):
                js_dict["model_versions"][idx] = _ensure_model_version_fields(v)
    elif response_proto is not None and hasattr(response_proto, "DESCRIPTOR"):
        field_names = [f.name for f in response_proto.DESCRIPTOR.fields]
        if "registered_model" in field_names:
            return {"registered_model": _ensure_registered_model_fields(dict(js_dict))}
        if "model_version" in field_names:
            return {"model_version": _ensure_model_version_fields(dict(js_dict))}

    if "nextPageToken" in js_dict and "next_page_token" not in js_dict:
        js_dict["next_page_token"] = js_dict.pop("nextPageToken")

    return js_dict


def _build_registry_store_class() -> type[Any]:
    """Lazily construct and return the GcpModelRegistryRestStore class."""
    import mlflow
    import mlflow.store.model_registry.rest_store as rest_store_mod
    from mlflow.exceptions import MlflowException
    from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE
    from mlflow.store.model_registry.rest_store import RestStore
    from mlflow.utils import proto_json_utils, rest_utils
    from mlflow.utils.rest_utils import MlflowHostCreds

    from google_cloud_managed_mlflow import auth, uri

    class GcpModelRegistryRestStore(RestStore):
        """Model Registry REST store implementation for Google Cloud Managed MLflow."""

        def __init__(
            self,
            store_uri: str | None = None,
            tracking_uri: str | None = None,
            **kwargs: Any,
        ) -> None:
            if not store_uri:
                raise MlflowException(
                    "Google Cloud model registry store URI must be non-empty.",
                    error_code=INVALID_PARAMETER_VALUE,
                )
            auth.validate_transport_security(store_uri)
            parsed = uri.parse_gcp_uri(store_uri)
            if not parsed.is_registry:
                raise MlflowException(
                    f"Model registry store URI must reference a 'registries' resource, "
                    f"got {parsed.resource_type!r} in {store_uri!r}.",
                    error_code=INVALID_PARAMETER_VALUE,
                )
            self.store_uri = store_uri
            self.tracking_uri = tracking_uri
            self.parsed_registry_uri = parsed
            self.registry_resource = parsed.resource_name
            self.base_api_url = parsed.base_api_url
            self.endpoint_url = parsed.resource_url

            super().__init__(get_host_creds=self.get_host_creds)  # type: ignore[no-untyped-call]

        def get_host_creds(self) -> Any:
            """Validate transport security and return Google Cloud host credentials."""
            auth.validate_transport_security(self.endpoint_url)
            return MlflowHostCreds(
                host=self.endpoint_url,
                auth="google-cloud-auth",
                ignore_tls_verification=False,
            )

        def _resolve_source_namespace(
            self, explicit_namespace: str | None = None
        ) -> str:
            def _validate_location(cand_loc: str) -> None:
                if cand_loc != self.parsed_registry_uri.location:
                    loc = self.parsed_registry_uri.location
                    raise MlflowException(
                        f"Cross-location model registration is not supported: "
                        f"registry is in location '{loc}', "
                        f"but tracking namespace is in location '{cand_loc}'.",
                        error_code=INVALID_PARAMETER_VALUE,
                    )

            candidates = [
                explicit_namespace,
                self.tracking_uri,
                getattr(mlflow, "get_tracking_uri", lambda: None)(),
                os.environ.get("MLFLOW_TRACKING_URI"),
            ]
            for candidate in candidates:
                if not candidate or not isinstance(candidate, str):
                    continue
                candidate_str = candidate.strip()
                if not candidate_str:
                    continue

                match = _CANONICAL_NS_RE.match(candidate_str)
                if match:
                    _validate_location(match.group("location"))
                    return candidate_str

                try:
                    parsed = uri.parse_gcp_uri(candidate_str)
                except Exception:
                    continue

                if parsed.is_namespace:
                    _validate_location(parsed.location)
                    return parsed.resource_name

            raise MlflowException(
                "Google Cloud Managed MLflow requires a configured tracking namespace "
                "to create a model version. Please configure MLFLOW_TRACKING_URI or "
                "pass tracking_uri to the client.",
                error_code=INVALID_PARAMETER_VALUE,
            )

        def _dispatch_delete_endpoint(
            self, api_name: str, payload: dict[str, Any]
        ) -> tuple[str, dict[str, Any] | None]:
            raw_name = payload.get("name")
            quoted_name = _sanitize_and_quote_model_name(raw_name)
            name = (
                raw_name.split("/registeredModels/")[-1]
                if isinstance(raw_name, str) and "/registeredModels/" in raw_name
                else raw_name
            )
            if api_name == "DeleteRegisteredModel":
                return f"/registeredModels/{quoted_name}", None
            if api_name == "DeleteModelVersion":
                version = payload.get("version")
                if version is None:
                    raise MlflowException(
                        "Model version is required.",
                        error_code=INVALID_PARAMETER_VALUE,
                    )
                return (
                    "/api/2.0/mlflow/model-versions/delete",
                    {"name": name, "version": str(version)},
                )
            if api_name == "DeleteRegisteredModelTag":
                tag_key = payload.get("key") or payload.get("tag_key")
                if not tag_key:
                    raise MlflowException(
                        "Tag key is required.",
                        error_code=INVALID_PARAMETER_VALUE,
                    )
                return (
                    "/api/2.0/mlflow/registered-models/delete-tag",
                    {"name": name, "tag_key": tag_key},
                )
            if api_name == "DeleteModelVersionTag":
                version = payload.get("version")
                if version is None:
                    raise MlflowException(
                        "Model version is required.",
                        error_code=INVALID_PARAMETER_VALUE,
                    )
                tag_key = payload.get("key") or payload.get("tag_key")
                if not tag_key:
                    raise MlflowException(
                        "Tag key is required.",
                        error_code=INVALID_PARAMETER_VALUE,
                    )
                return (
                    "/api/2.0/mlflow/model-versions/delete-tag",
                    {
                        "name": name,
                        "version": str(version),
                        "tag_key": tag_key,
                    },
                )
            if api_name == "DeleteRegisteredModelAlias":
                alias = payload.get("alias")
                if not alias:
                    raise MlflowException(
                        "Alias is required.",
                        error_code=INVALID_PARAMETER_VALUE,
                    )
                return (
                    "/api/2.0/mlflow/registered-models/alias",
                    {"name": name, "alias": alias},
                )
            raise ValueError(f"Unsupported delete API: {api_name}")

        def _dispatch_create_registered_model(
            self, payload: dict[str, Any]
        ) -> tuple[str, str, dict[str, Any] | None, dict[str, Any] | None]:
            raw_name = payload.pop("name", None)
            _ = _sanitize_and_quote_model_name(raw_name)
            name = (
                raw_name.split("/registeredModels/")[-1]
                if isinstance(raw_name, str) and "/registeredModels/" in raw_name
                else raw_name
            )
            tags = payload.pop("tags", [])
            desc = payload.pop("description", None)
            if isinstance(tags, list):
                annotations = {
                    t["key"]: t["value"]
                    for t in tags
                    if isinstance(t, dict) and "key" in t and "value" in t
                }
            elif isinstance(tags, dict):
                annotations = tags
            else:
                annotations = {}
            endpoint = "/registeredModels"
            params = {"registered_model_id": name}
            body: dict[str, Any] = {"description": desc} if desc else {}
            if annotations:
                body["annotations"] = annotations
            return endpoint, "POST", params, body

        def _dispatch_request(
            self, api_name: str, api: Any, payload: dict[str, Any]
        ) -> tuple[str, str, dict[str, Any] | None, dict[str, Any] | None]:
            if api_name == "CreateRegisteredModel":
                return self._dispatch_create_registered_model(payload)

            if api_name == "GetRegisteredModel":
                raw_name = payload.get("name")
                quoted_name = _sanitize_and_quote_model_name(raw_name)
                return f"/registeredModels/{quoted_name}", "GET", None, None

            if api_name == "UpdateRegisteredModel":
                raw_name = payload.get("name")
                quoted_name = _sanitize_and_quote_model_name(raw_name)
                desc = payload.get("description")
                if desc is not None:
                    return (
                        f"/registeredModels/{quoted_name}",
                        "PATCH",
                        {"update_mask": "description"},
                        {"description": desc},
                    )
                return f"/registeredModels/{quoted_name}", "GET", None, None

            if api_name in _BODYLESS_DELETE_APIS:
                endpoint, params = self._dispatch_delete_endpoint(api_name, payload)
                return endpoint, "DELETE", params, None

            if api_name == "SearchRegisteredModels" and "filter" in payload:
                f_val = payload.get("filter")
                if f_val:
                    clean_f = _sanitize_prompt_filter(str(f_val))
                    if clean_f:
                        payload["filter"] = clean_f
                    else:
                        payload.pop("filter", None)
                else:
                    payload.pop("filter", None)

            if api_name == "CreateModelVersion":
                payload["source_namespace"] = self._resolve_source_namespace(
                    explicit_namespace=payload.get("source_namespace")
                )
                return (
                    "/api/2.0/mlflow/model-versions/create",
                    "POST",
                    None,
                    payload,
                )

            self_any: Any = self
            if hasattr(self, "_get_endpoint_from_method"):
                endpoint_sub, orig_method = self_any._get_endpoint_from_method(api)
            else:
                endpoint_sub, orig_method = rest_store_mod._METHOD_TO_INFO[api]
            if not endpoint_sub.startswith("/"):
                endpoint_sub = f"/{endpoint_sub}"
            endpoint = endpoint_sub
            if orig_method == "GET":
                return endpoint, "GET", payload, None
            return endpoint, orig_method, None, payload

        def _call_endpoint(
            self,
            api: Any,
            json_body: str | dict[str, Any] | None = None,
            call_all_endpoints: bool = False,
            extra_headers: dict[str, Any] | None = None,
            response_proto: Any = None,
            **kwargs: Any,
        ) -> Any:
            try:
                auth.harden_mlflow_request_session()
            except Exception:
                pass

            if json_body is None:
                payload: dict[str, Any] = {}
            elif isinstance(json_body, str):
                payload = json.loads(json_body) if json_body.strip() else {}
            elif isinstance(json_body, dict):
                payload = dict(json_body)
            else:
                payload = dict(json_body)

            api_name = getattr(api, "__name__", str(api))
            endpoint, method, params, body = self._dispatch_request(
                api_name, api, payload
            )

            call_kwargs: dict[str, Any] = {
                "host_creds": self.get_host_creds(),
                "endpoint": endpoint,
                "method": method,
                "allow_redirects": False,
            }
            if extra_headers is not None:
                call_kwargs["extra_headers"] = extra_headers
            if (
                "retry_timeout_seconds" in kwargs
                and kwargs["retry_timeout_seconds"] is not None
            ):
                call_kwargs["retry_timeout_seconds"] = kwargs["retry_timeout_seconds"]

            if params is not None:
                call_kwargs["params"] = params
            if body is not None:
                call_kwargs["json"] = body

            response = rest_utils.http_request(**call_kwargs)

            if response.status_code >= 400:
                uri.raise_for_aip193_error(response)

            if response_proto is None:
                self_any: Any = self
                if hasattr(self, "_get_response_from_method"):
                    try:
                        response_proto = self_any._get_response_from_method(api)
                    except Exception:
                        response_proto = getattr(api, "Response", lambda: None)()
                elif hasattr(api, "Response"):
                    response_proto = api.Response()

            if response.status_code == 204 or not (response.text or "").strip():
                return response_proto

            response = rest_utils.verify_rest_response(response, endpoint)

            js_dict = json.loads(response.text)
            js_dict = normalize_registry_response(
                api, js_dict, response_proto=response_proto
            )
            proto_json_utils.parse_dict(js_dict=js_dict, message=response_proto)
            return response_proto

    return GcpModelRegistryRestStore


def _get_registry_store_cls() -> type[Any]:
    """Retrieve or lazily initialize the GcpModelRegistryRestStore class."""
    global _REGISTRY_STORE_CLS
    if _REGISTRY_STORE_CLS is None:
        _REGISTRY_STORE_CLS = _build_registry_store_class()
    return _REGISTRY_STORE_CLS


def build_registry_store(
    store_uri: str,
    tracking_uri: str | None = None,
    **kwargs: Any,
) -> Any:
    """Construct an MLflow model registry store instance for the given URI.

    Delegates to GcpModelRegistryRestStore if store_uri is a Google Cloud registry
    URI; falls back to stock MLflow RestStore otherwise.
    """
    from google_cloud_managed_mlflow import uri

    if not uri.is_gcp_registry_uri(store_uri):
        if uri.is_gcp_mlflow_uri(store_uri):
            from mlflow.exceptions import MlflowException
            from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE

            raise MlflowException(
                f"The provided URI '{store_uri}' is a Managed MLflow tracking "
                "namespace URI, not a model registry URI. Managed MLflow requires "
                "a dedicated model registry URI (e.g., "
                "'https://mlflow.googleapis.com/v1/projects/<project>/"
                "locations/<location>/registries/<registry>'). "
                "Please configure MLFLOW_REGISTRY_URI or pass registry_uri explicitly.",
                error_code=INVALID_PARAMETER_VALUE,
            )
        from mlflow.tracking._model_registry.utils import _get_rest_store

        return _get_rest_store(store_uri, **kwargs)

    cls = _get_registry_store_cls()
    return cls(store_uri=store_uri, tracking_uri=tracking_uri, **kwargs)


def ensure_registry_store_registered() -> bool:
    """Ensure Google Cloud model registry store is registered with MLflow.

    Registers https under model registry store registry if present, ensures
    Google Cloud auth provider is registered, and hardens the request session.
    """
    try:
        from google_cloud_managed_mlflow import auth

        auth.ensure_auth_provider_registered()
        auth.harden_mlflow_request_session()

        import mlflow.tracking._model_registry.utils as reg_utils

        registry = None
        if hasattr(reg_utils, "_get_store_registry"):
            registry = reg_utils._get_store_registry()
        elif hasattr(reg_utils, "_model_registry_store_registry"):
            registry = reg_utils._model_registry_store_registry

        if registry is not None:
            registry.register("https", build_registry_store)
            return True
        return False
    except Exception:
        return False


def __getattr__(name: str) -> Any:
    """PEP 562 lazy attribute access for GcpModelRegistryRestStore."""
    if name == "GcpModelRegistryRestStore":
        return _get_registry_store_cls()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


if "mlflow" in sys.modules:
    ensure_registry_store_registered()
