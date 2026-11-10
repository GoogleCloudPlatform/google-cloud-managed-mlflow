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

"""In-process fake of the Managed MLflow ``v1alpha`` REST surface (loopback).

The fake wraps a real open-source MLflow tracking server (SQLite backend) and
adds the Managed MLflow specific behavior the plugin depends on:

* resource-prefixed routing (``/v1alpha/projects/p/locations/l/namespaces/n``
  and ``.../registries/r``) with bearer-token enforcement;
* Credential Access Boundary token RPCs for Namespaces and RegisteredModels,
  which register downscoped tokens with :class:`tests.fakes.gcs.FakeGcs`;
* RegisteredModel Vault ``storageLocation`` and server-side copy of model
  version artifacts into ``<storageLocation>/versions/<v>`` on version creation;
* a Compute Engine metadata token endpoint for JVM-side caller identity.
"""

from __future__ import annotations

import dataclasses
import datetime
import io
import json
import os
import re
import threading
import time
import urllib.parse
import uuid
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

from tests.fakes.gcs import READ_CREATE, READ_ONLY, READ_WRITE, FakeGcs

PROJECT = "test-project"
LOCATION = "us-central1"
NAMESPACE_ID = "ns1"
REGISTRY_ID = "reg1"
NAMESPACE_BUCKET = "mlflow-ns-bucket"
REGISTRY_BUCKET = "mlflow-reg-bucket"
NAMESPACE_ROOT = f"gs://{NAMESPACE_BUCKET}/experiments"
CALLER_TOKEN = "ya29." + "test-caller-python"
DRIVER_IDENTITY_TOKEN = "ya29." + "test-driver-identity"
MAX_SOURCE_LENGTH = 4096
_INVALID_NAME_CHARS = frozenset("/:.%\"'\\")
_SOURCE_SCHEMES = ("gs://", "models:/", "runs:/")
_DELETE_FIELDS = {"tag_key": "key"}

_RESOURCE_RE = re.compile(
    r"^/(?P<version>v1[a-z0-9]*)/projects/(?P<project>[^/]+)/"
    r"locations/(?P<location>[^/]+)/"
    r"(?P<kind>namespaces|registries)/(?P<rid>[^/:]+)(?P<rest>.*)$"
)

StartResponse = Callable[..., Any]
Environ = dict[str, Any]


@dataclasses.dataclass(frozen=True)
class CabCall:
    """Secret-free record of one CAB issuance RPC."""

    resource: str
    verb: str
    body: str
    mode: str
    role: str
    status: int
    boundary: str


@dataclasses.dataclass(frozen=True)
class RecordedRequest:
    """Record of one control-plane request (headers are never recorded)."""

    method: str
    path: str
    query: tuple[tuple[str, str], ...]
    body: str

    def params(self) -> dict[str, str]:
        """Query parameters as a dict."""
        return dict(self.query)


@dataclasses.dataclass
class _RegisteredModel:
    resource_id: str
    name: str
    uid: str
    storage_location: str
    description: str
    created: float
    updated: float


def _invalid_model_name(name: str) -> str | None:
    if not name.strip():
        return "name must not be empty"
    if len(name) > 256:
        return "name must be at most 256 characters"
    if set(name) & _INVALID_NAME_CHARS:
        return "name contains a character that is not allowed"
    return None


def _iso(epoch: float, nanos: bool = False) -> str:
    moment = datetime.datetime.fromtimestamp(epoch, tz=datetime.timezone.utc)
    base = moment.strftime("%Y-%m-%dT%H:%M:%S")
    if nanos:
        return f"{base}.{moment.microsecond:06d}123Z"
    return f"{base}.{moment.microsecond // 1000:03d}Z"


def _role(user_agent: str) -> str:
    if "role=driver" in user_agent:
        return "java-driver"
    if "role=executor" in user_agent:
        return "java-executor"
    # GcpGCSArtifactRepository mints from the client process through a plain
    # requests session, so its User-Agent carries no role tag.
    if "role=client" in user_agent or user_agent.startswith("python-requests/"):
        return "python-client"
    return "other"


class FakeManagedMlflow:
    """Loopback Managed MLflow server backed by a real MLflow tracking server."""

    def __init__(self, gcs: FakeGcs, workdir: Path) -> None:
        self.gcs = gcs
        self.workdir = workdir
        self.token_ttl_seconds = 3600.0
        self.snake_case_responses = False
        self.cab_failures: list[int] = []
        self.cab_calls: list[CabCall] = []
        self.rest_calls: list[tuple[str, str, int]] = []
        self.requests: list[RecordedRequest] = []
        self.source_namespaces: list[str] = []
        self.accepted_caller_tokens = {CALLER_TOKEN, DRIVER_IDENTITY_TOKEN}
        self.metadata_requests = 0
        self.download_uri_mode = "rewrite"
        self.namespace_cab_denied = False
        self.honor_request_ids = True
        self.rest_failures: dict[str, list[int]] = {}
        self.commit_then_fail: dict[str, list[int]] = {}
        self._request_ids: dict[str, str] = {}
        self._models: dict[str, _RegisteredModel] = {}
        self._lock = threading.RLock()
        self._server: Any = None
        self._thread: threading.Thread | None = None
        self._flask_app: Any = None
        self._handlers: Any = None

    # ----------------------------------------------------------------- lifecycle

    def start(self) -> FakeManagedMlflow:
        """Initialize the backing MLflow server stores and start serving."""
        db_uri = f"sqlite:///{self.workdir / 'mlflow.db'}"
        os.environ["_MLFLOW_SERVER_FILE_STORE"] = db_uri
        os.environ["_MLFLOW_SERVER_REGISTRY_STORE"] = db_uri
        os.environ["_MLFLOW_SERVER_ARTIFACT_ROOT"] = NAMESPACE_ROOT
        os.environ.setdefault("MLFLOW_SERVER_ALLOWED_HOSTS", "*")
        import mlflow
        from mlflow.server import app as flask_app, handlers
        from werkzeug.serving import make_server

        handlers._get_tracking_store()
        handlers._get_model_registry_store()
        local_store = f"sqlite:///{self.workdir / 'client-default.db'}"
        mlflow.set_tracking_uri(local_store)
        mlflow.set_registry_uri(local_store)
        self._flask_app = flask_app
        self._handlers = handlers
        self._server = make_server("127.0.0.1", 0, self, threaded=True)
        self._thread = threading.Thread(
            target=self._server.serve_forever, name="fake-managed-mlflow", daemon=True
        )
        self._thread.start()
        return self

    def stop(self) -> None:
        """Stop serving."""
        if self._server is not None:
            self._server.shutdown()
            self._server = None

    @property
    def origin(self) -> str:
        """``http://127.0.0.1:<port>``."""
        assert self._server is not None
        return f"http://127.0.0.1:{self._server.server_port}"

    @property
    def metadata_host(self) -> str:
        """``127.0.0.1:<port>`` for ``GCE_METADATA_HOST``."""
        assert self._server is not None
        return f"127.0.0.1:{self._server.server_port}"

    @property
    def tracking_uri(self) -> str:
        """Managed MLflow Namespace URI."""
        return f"{self.origin}/v1alpha/{self.namespace_resource}"

    @property
    def registry_uri(self) -> str:
        """Managed MLflow Registry URI."""
        return f"{self.origin}/v1alpha/{self.registry_resource}"

    @property
    def namespace_resource(self) -> str:
        """Namespace resource name."""
        return f"projects/{PROJECT}/locations/{LOCATION}/namespaces/{NAMESPACE_ID}"

    @property
    def registry_resource(self) -> str:
        """Registry resource name."""
        return f"projects/{PROJECT}/locations/{LOCATION}/registries/{REGISTRY_ID}"

    def reset_counters(self) -> None:
        """Forget recorded calls and injected failures; restore defaults."""
        with self._lock:
            self.cab_calls.clear()
            self.rest_calls.clear()
            self.requests.clear()
            self.source_namespaces.clear()
            self.cab_failures.clear()
            self.rest_failures.clear()
            self.commit_then_fail.clear()
            self.metadata_requests = 0
        self.snake_case_responses = False
        self.token_ttl_seconds = 3600.0
        self.download_uri_mode = "rewrite"
        self.namespace_cab_denied = False
        self.honor_request_ids = True

    def cab_calls_snapshot(self) -> list[CabCall]:
        """Copy of all CAB issuance records."""
        with self._lock:
            return list(self.cab_calls)

    def requests_snapshot(self) -> list[RecordedRequest]:
        """Copy of all recorded control-plane requests."""
        with self._lock:
            return list(self.requests)

    def _by_display_name(self, name: str) -> _RegisteredModel | None:
        with self._lock:
            for model in self._models.values():
                if model.name == name:
                    return model
        return None

    def _ensure_model_record(
        self, name: str, description: str = ""
    ) -> _RegisteredModel:
        existing = self._by_display_name(name)
        if existing is not None:
            return existing
        uid = uuid.uuid4().hex[:20]
        now = time.time()
        model = _RegisteredModel(
            resource_id=name,
            name=name,
            uid=uid,
            storage_location=f"gs://{REGISTRY_BUCKET}/registeredModels/{uid}",
            description=description,
            created=now,
            updated=now,
        )
        with self._lock:
            self._models[name] = model
        return model

    def storage_location(self, name: str) -> str:
        """Vault ``gs://`` prefix of a registered model (by MLflow name)."""
        model = self._by_display_name(name)
        if model is None:
            raise KeyError(name)
        return model.storage_location

    def resource_id(self, name: str) -> str:
        """Immutable resource ID of a registered model (by MLflow name)."""
        model = self._by_display_name(name)
        if model is None:
            raise KeyError(name)
        return model.resource_id

    # ---------------------------------------------------------------------- WSGI

    def __call__(
        self, environ: Environ, start_response: StartResponse
    ) -> Iterable[bytes]:
        path = str(environ.get("PATH_INFO", ""))
        method = str(environ.get("REQUEST_METHOD", "GET"))
        if path.startswith("/computeMetadata/"):
            return self._metadata(environ, start_response)
        auth = str(environ.get("HTTP_AUTHORIZATION", ""))
        token = auth[7:].strip() if auth.lower().startswith("bearer ") else ""
        if token not in self.accepted_caller_tokens:
            return self._reply(
                start_response,
                401,
                {
                    "error": {
                        "code": 401,
                        "status": "UNAUTHENTICATED",
                        "message": "bad token",
                    }
                },
                method,
                path,
            )
        body = self._body(environ)
        environ["wsgi.input"] = io.BytesIO(body)
        environ["CONTENT_LENGTH"] = str(len(body))
        query = urllib.parse.parse_qsl(
            str(environ.get("QUERY_STRING", "")), keep_blank_values=True
        )
        with self._lock:
            self.requests.append(
                RecordedRequest(
                    method, path, tuple(query), body.decode("utf-8", "replace")
                )
            )
        failure = self._pop_failure(self.rest_failures, path)
        if failure is not None:
            return self._transient(start_response, failure, method, path)
        after = self._pop_failure(self.commit_then_fail, path)
        if after is None:
            return self._route(environ, start_response, method, path)
        captured: dict[str, Any] = {}

        def _capture(status: str, headers: Any, exc_info: Any = None) -> None:
            captured["status"] = status
            captured["headers"] = headers

        chunks = list(self._route(environ, _capture, method, path))
        if str(captured.get("status", "")).startswith("200"):
            return self._transient(start_response, after, method, path)
        start_response(captured["status"], captured["headers"])
        return chunks

    def _pop_failure(self, failures: dict[str, list[int]], path: str) -> int | None:
        with self._lock:
            for suffix, statuses in failures.items():
                if statuses and path.endswith(suffix):
                    return statuses.pop(0)
        return None

    def _transient(
        self, start_response: StartResponse, status: int, method: str, path: str
    ) -> Iterable[bytes]:
        google_status = {429: "RESOURCE_EXHAUSTED", 500: "INTERNAL"}.get(
            status, "UNAVAILABLE"
        )
        return self._reply(
            start_response,
            status,
            {
                "error": {
                    "code": status,
                    "status": google_status,
                    "message": "injected transient failure",
                }
            },
            method,
            path,
        )

    def _route(
        self, environ: Environ, start_response: StartResponse, method: str, path: str
    ) -> Iterable[bytes]:
        match = _RESOURCE_RE.match(path)
        if match is None:
            return self._not_found(start_response, method, path)
        kind, rid, rest = match.group("kind"), match.group("rid"), match.group("rest")
        if kind == "namespaces":
            if rid != NAMESPACE_ID:
                return self._not_found(start_response, method, path)
            if rest in (":issueReadArtifactToken", ":issueReadWriteArtifactToken"):
                return self._namespace_cab(environ, start_response, rest[1:])
            if rest.startswith(
                ("/api/2.0/mlflow/", "/ajax-api/2.0/mlflow/", "/api/3.0/mlflow/")
            ):
                return self._forward(environ, start_response, rest)
            return self._not_found(start_response, method, path)
        if rid != REGISTRY_ID:
            return self._not_found(start_response, method, path)
        if rest == "/registeredModels" and method == "POST":
            return self._create_model(environ, start_response)
        if rest == "/registeredModels" and method == "GET":
            with self._lock:
                items = [self._model_json(m) for m in self._models.values()]
            return self._reply(
                start_response, 200, {"registeredModels": items}, "GET", path
            )
        model_match = re.match(r"^/registeredModels/([^/:]+)(:\w+)?$", rest)
        if model_match is not None:
            resource_id = urllib.parse.unquote(model_match.group(1))
            verb = (model_match.group(2) or "")[1:]
            if verb in ("issueReadArtifactToken", "issueReadWriteArtifactToken"):
                return self._model_cab(environ, start_response, resource_id, verb)
            if not verb and method == "GET":
                return self._get_model(start_response, resource_id, method, path)
            if not verb and method == "PATCH":
                return self._patch_model(environ, start_response, resource_id)
            if not verb and method == "DELETE":
                return self._delete_model(start_response, resource_id, method, path)
            return self._not_found(start_response, method, path)
        if rest == "/api/2.0/mlflow/registered-models/create" and method == "POST":
            return self._create_oss_model(environ, start_response, rest)
        if rest == "/api/2.0/mlflow/registered-models/search":
            return self._search_models(environ, start_response)
        if rest == "/api/2.0/mlflow/registered-models/rename" and method == "POST":
            return self._rename_model(environ, start_response, rest)
        if rest == "/api/2.0/mlflow/model-versions/get-download-uri":
            return self._download_uri(environ, start_response)
        if rest == "/api/2.0/mlflow/model-versions/create":
            return self._create_version(environ, start_response, rest)
        if method == "DELETE" and rest.startswith("/api/2.0/mlflow/"):
            return self._query_delete(environ, start_response, rest)
        if rest.startswith(("/api/2.0/mlflow/", "/api/3.0/mlflow/")):
            return self._forward(environ, start_response, rest)
        return self._not_found(start_response, method, path)

    # -------------------------------------------------------------------- helpers

    def _reply(
        self,
        start_response: StartResponse,
        status: int,
        payload: dict[str, Any],
        method: str,
        path: str,
    ) -> Iterable[bytes]:
        body = json.dumps(payload).encode()
        with self._lock:
            self.rest_calls.append((method, path, status))
        reason = {
            200: "OK",
            400: "Bad Request",
            401: "Unauthorized",
            403: "Forbidden",
        }.get(status, "Error")
        start_response(
            f"{status} {reason}",
            [("Content-Type", "application/json"), ("Content-Length", str(len(body)))],
        )
        return [body]

    def _not_found(
        self, start_response: StartResponse, method: str, path: str
    ) -> Iterable[bytes]:
        return self._reply(
            start_response,
            404,
            {
                "error": {
                    "code": 404,
                    "status": "NOT_FOUND",
                    "message": f"{method} {path}",
                }
            },
            method,
            path,
        )

    @staticmethod
    def _body(environ: Environ) -> bytes:
        length = int(environ.get("CONTENT_LENGTH") or "0")
        stream = environ["wsgi.input"]
        return bytes(stream.read(length)) if length else b""

    def _call_flask(
        self,
        environ: Environ,
        rest: str,
        body: bytes | None = None,
        query: str | None = None,
    ) -> tuple[int, list[tuple[str, str]], bytes]:
        inner = dict(environ)
        inner["PATH_INFO"] = rest
        inner["SCRIPT_NAME"] = ""
        if query is not None:
            inner["QUERY_STRING"] = query
        if body is not None:
            inner["wsgi.input"] = io.BytesIO(body)
            inner["CONTENT_LENGTH"] = str(len(body))
        captured: dict[str, Any] = {}

        def _start(
            status: str, headers: list[tuple[str, str]], exc_info: Any = None
        ) -> None:
            captured["status"] = int(status.split(" ", 1)[0])
            captured["headers"] = headers

        chunks = self._flask_app(inner, _start)
        try:
            payload = b"".join(chunks)
        finally:
            close = getattr(chunks, "close", None)
            if callable(close):
                close()
        return (
            int(captured.get("status", 500)),
            list(captured.get("headers", [])),
            payload,
        )

    def _forward(
        self, environ: Environ, start_response: StartResponse, rest: str
    ) -> Iterable[bytes]:
        status, headers, payload = self._call_flask(environ, rest)
        with self._lock:
            self.rest_calls.append((str(environ.get("REQUEST_METHOD")), rest, status))
        start_response(f"{status} X", headers)
        return [payload]

    def _metadata(
        self, environ: Environ, start_response: StartResponse
    ) -> Iterable[bytes]:
        if environ.get("HTTP_METADATA_FLAVOR") != "Google":
            start_response("403 Forbidden", [("Content-Length", "0")])
            return [b""]
        with self._lock:
            self.metadata_requests += 1
        path = str(environ.get("PATH_INFO", ""))
        if path.endswith("/token"):
            body = json.dumps(
                {
                    "access_token": DRIVER_IDENTITY_TOKEN,
                    "expires_in": 3599,
                    "token_type": "Bearer",
                }
            ).encode()
            start_response(
                "200 OK",
                [
                    ("Content-Type", "application/json"),
                    ("Metadata-Flavor", "Google"),
                    ("Content-Length", str(len(body))),
                ],
            )
            return [body]
        text = PROJECT.encode() if path.endswith("/project-id") else b"ok"
        start_response(
            "200 OK",
            [("Metadata-Flavor", "Google"), ("Content-Length", str(len(text)))],
        )
        return [text]

    # ------------------------------------------------------------------------ CAB

    def _mint(self, boundary: str, mode: str, role: str) -> tuple[str, float]:
        token = "ya29." + f"cab-{mode.lower()}-{role}-{uuid.uuid4().hex}"
        ttl = self.token_ttl_seconds
        self.gcs.grant(token, boundary, mode, ttl, label=f"{mode}:{boundary}")
        return token, time.time() + ttl

    def _cab_response(
        self,
        start_response: StartResponse,
        boundary: str,
        token: str,
        expiry: float,
        path: str,
    ) -> Iterable[bytes]:
        if self.snake_case_responses:
            payload: dict[str, Any] = {
                "artifact_uri": boundary,
                "credentials": {
                    "access_token": token,
                    "expire_time": _iso(expiry, True),
                },
            }
        else:
            payload = {
                "artifactUri": boundary,
                "credentials": {"accessToken": token, "expireTime": _iso(expiry, True)},
            }
        return self._reply(start_response, 200, payload, "POST", path)

    def _injected_failure(self) -> int | None:
        with self._lock:
            if self.cab_failures:
                return self.cab_failures.pop(0)
        return None

    def _record_cab(
        self,
        resource: str,
        verb: str,
        body: str,
        mode: str,
        role: str,
        status: int,
        boundary: str,
    ) -> None:
        with self._lock:
            self.cab_calls.append(
                CabCall(resource, verb, body, mode, role, status, boundary)
            )

    def _namespace_cab(
        self, environ: Environ, start_response: StartResponse, verb: str
    ) -> Iterable[bytes]:
        raw = self._body(environ).decode() or "{}"
        role = _role(str(environ.get("HTTP_USER_AGENT", "")))
        write = verb == "issueReadWriteArtifactToken"
        mode = READ_WRITE if write else READ_ONLY
        path = str(environ.get("PATH_INFO", ""))
        failure = self._injected_failure()
        if failure is not None:
            self._record_cab(
                self.namespace_resource, verb, raw, mode, role, failure, ""
            )
            return self._reply(
                start_response,
                failure,
                {
                    "error": {
                        "code": failure,
                        "status": "UNAVAILABLE",
                        "message": "injected",
                    }
                },
                "POST",
                path,
            )
        if self.namespace_cab_denied:
            self._record_cab(self.namespace_resource, verb, raw, mode, role, 403, "")
            return self._reply(
                start_response,
                403,
                {
                    "error": {
                        "code": 403,
                        "status": "PERMISSION_DENIED",
                        "message": "The caller has no access to the namespace.",
                    }
                },
                "POST",
                path,
            )
        request = json.loads(raw)
        targets = [k for k in ("artifactUri", "runId", "modelId") if request.get(k)]
        artifact_path = str(request.get("artifactPath") or "").strip("/")
        bad_path = any(seg in (".", "..") for seg in artifact_path.split("/"))
        if len(targets) > 1 or bad_path:
            self._record_cab(self.namespace_resource, verb, raw, mode, role, 400, "")
            return self._reply(
                start_response,
                400,
                {
                    "error": {
                        "code": 400,
                        "status": "INVALID_ARGUMENT",
                        "message": "at most one target and no dot segments",
                    }
                },
                "POST",
                path,
            )
        store = self._handlers._get_tracking_store()
        if not targets:
            boundary = NAMESPACE_ROOT
        elif request.get("artifactUri"):
            boundary = str(request["artifactUri"]).rstrip("/")
        elif request.get("runId"):
            run = store.get_run(request["runId"])
            if write and run.info.status != "RUNNING":
                self._record_cab(
                    self.namespace_resource, verb, raw, mode, role, 400, ""
                )
                return self._reply(
                    start_response,
                    400,
                    {
                        "error": {
                            "code": 400,
                            "status": "FAILED_PRECONDITION",
                            "message": "Run is not RUNNING.",
                        }
                    },
                    "POST",
                    path,
                )
            boundary = str(run.info.artifact_uri).rstrip("/")
        else:
            model = store.get_logged_model(request["modelId"])
            boundary = str(model.artifact_location).rstrip("/")
        if artifact_path:
            boundary = f"{boundary}/{artifact_path}"
        if boundary != NAMESPACE_ROOT and not boundary.startswith(NAMESPACE_ROOT + "/"):
            self._record_cab(
                self.namespace_resource, verb, raw, mode, role, 403, boundary
            )
            return self._reply(
                start_response,
                403,
                {
                    "error": {
                        "code": 403,
                        "status": "PERMISSION_DENIED",
                        "message": "Artifact URI is outside the namespace.",
                    }
                },
                "POST",
                path,
            )
        token, expiry = self._mint(boundary, mode, role)
        self._record_cab(self.namespace_resource, verb, raw, mode, role, 200, boundary)
        reported = boundary if targets else boundary + "/"
        return self._cab_response(start_response, reported, token, expiry, path)

    def _model_cab(
        self,
        environ: Environ,
        start_response: StartResponse,
        resource_id: str,
        verb: str,
    ) -> Iterable[bytes]:
        raw = self._body(environ).decode() or "{}"
        role = _role(str(environ.get("HTTP_USER_AGENT", "")))
        write = verb == "issueReadWriteArtifactToken"
        mode = READ_CREATE if write else READ_ONLY
        path = str(environ.get("PATH_INFO", ""))
        resource = f"{self.registry_resource}/registeredModels/{resource_id}"
        failure = self._injected_failure()
        if failure is not None:
            self._record_cab(resource, verb, raw, mode, role, failure, "")
            return self._reply(
                start_response,
                failure,
                {
                    "error": {
                        "code": failure,
                        "status": "UNAVAILABLE",
                        "message": "injected",
                    }
                },
                "POST",
                path,
            )
        request = json.loads(raw)
        unexpected = sorted(set(request) - {"artifactUri"})
        with self._lock:
            model = self._models.get(resource_id)
        if model is None or unexpected:
            status = 404 if model is None else 400
            self._record_cab(resource, verb, raw, mode, role, status, "")
            return self._reply(
                start_response,
                status,
                {
                    "error": {
                        "code": status,
                        "status": "INVALID_ARGUMENT",
                        "message": "bad",
                    }
                },
                "POST",
                path,
            )
        boundary = str(request.get("artifactUri") or model.storage_location).rstrip("/")
        root = model.storage_location
        if boundary != root and not boundary.startswith(root + "/"):
            self._record_cab(resource, verb, raw, mode, role, 403, boundary)
            return self._reply(
                start_response,
                403,
                {
                    "error": {
                        "code": 403,
                        "status": "PERMISSION_DENIED",
                        "message": "outside",
                    }
                },
                "POST",
                path,
            )
        token, expiry = self._mint(boundary, mode, role)
        self._record_cab(resource, verb, raw, mode, role, 200, boundary)
        return self._cab_response(start_response, boundary, token, expiry, path)

    # --------------------------------------------------------- RegisteredModels

    def _error(
        self,
        start_response: StartResponse,
        status: int,
        google_status: str,
        message: str,
        method: str,
        path: str,
    ) -> Iterable[bytes]:
        return self._reply(
            start_response,
            status,
            {"error": {"code": status, "status": google_status, "message": message}},
            method,
            path,
        )

    def _model_json(self, model: _RegisteredModel) -> dict[str, Any]:
        return {
            "name": f"{self.registry_resource}/registeredModels/{model.resource_id}",
            "displayName": model.name,
            "uid": model.uid,
            "storageLocation": model.storage_location,
            "description": model.description,
            "createTime": _iso(model.created),
            "updateTime": _iso(model.updated),
        }

    def _create_oss_model(
        self, environ: Environ, start_response: StartResponse, rest: str
    ) -> Iterable[bytes]:
        raw = self._body(environ)
        request = json.loads(raw or b"{}")
        name = str(request.get("name") or "")
        description = str(request.get("description") or "")
        status, headers, payload = self._call_flask(environ, rest, body=raw)
        if status == 200 and name:
            self._ensure_model_record(name, description)
        with self._lock:
            self.rest_calls.append(("POST", rest, status))
        start_response(f"{status} X", headers)
        return [payload]

    def _create_model(
        self, environ: Environ, start_response: StartResponse
    ) -> Iterable[bytes]:
        path = str(environ.get("PATH_INFO", ""))
        query = dict(urllib.parse.parse_qsl(str(environ.get("QUERY_STRING", ""))))
        request = json.loads(self._body(environ) or b"{}")
        resource_id = query.get("registeredModelId", "")
        display = str(request.get("displayName") or "")
        request_id = query.get("requestId", "")
        with self._lock:
            known = (
                self._request_ids.get(request_id) if self.honor_request_ids else None
            )
            existing = self._models.get(known) if known else None
        if existing is not None:
            return self._reply(
                start_response, 200, self._model_json(existing), "POST", path
            )
        problem = _invalid_model_name(resource_id) or _invalid_model_name(display)
        if problem is None and resource_id != display:
            problem = "registered_model_id must equal display_name"
        if problem is not None:
            return self._error(
                start_response, 400, "INVALID_ARGUMENT", problem, "POST", path
            )
        with self._lock:
            taken = resource_id in self._models
        if taken or self._by_display_name(display) is not None:
            return self._error(
                start_response, 409, "ALREADY_EXISTS", display, "POST", path
            )
        from mlflow.entities.model_registry import RegisteredModelTag

        annotations = request.get("annotations") or {}
        tags = [RegisteredModelTag(str(k), str(v)) for k, v in annotations.items()]
        store = self._handlers._get_model_registry_store()
        store.create_registered_model(display, tags, request.get("description") or None)
        uid = uuid.uuid4().hex[:20]
        now = time.time()
        model = _RegisteredModel(
            resource_id=resource_id,
            name=display,
            uid=uid,
            storage_location=f"gs://{REGISTRY_BUCKET}/registeredModels/{uid}",
            description=str(request.get("description") or ""),
            created=now,
            updated=now,
        )
        with self._lock:
            self._models[resource_id] = model
            if request_id:
                self._request_ids[request_id] = resource_id
        return self._reply(start_response, 200, self._model_json(model), "POST", path)

    def _get_model(
        self, start_response: StartResponse, resource_id: str, method: str, path: str
    ) -> Iterable[bytes]:
        with self._lock:
            model = self._models.get(resource_id)
        if model is None:
            return self._not_found(start_response, method, path)
        return self._reply(start_response, 200, self._model_json(model), method, path)

    def _patch_model(
        self, environ: Environ, start_response: StartResponse, resource_id: str
    ) -> Iterable[bytes]:
        path = str(environ.get("PATH_INFO", ""))
        request = json.loads(self._body(environ) or b"{}")
        with self._lock:
            model = self._models.get(resource_id)
        if model is None:
            return self._not_found(start_response, "PATCH", path)
        description = str(request.get("description") or "")
        self._handlers._get_model_registry_store().update_registered_model(
            model.name, description
        )
        with self._lock:
            model.description = description
            model.updated = time.time()
        return self._reply(start_response, 200, self._model_json(model), "PATCH", path)

    def _delete_model(
        self, start_response: StartResponse, resource_id: str, method: str, path: str
    ) -> Iterable[bytes]:
        with self._lock:
            model = self._models.pop(resource_id, None)
        if model is None:
            return self._not_found(start_response, method, path)
        self._handlers._get_model_registry_store().delete_registered_model(model.name)
        return self._reply(start_response, 200, {}, method, path)

    def _rename_model(
        self, environ: Environ, start_response: StartResponse, rest: str
    ) -> Iterable[bytes]:
        path = str(environ.get("PATH_INFO", ""))
        raw = self._body(environ)
        request = json.loads(raw or b"{}")
        old = str(request.get("name") or "")
        new = str(request.get("new_name") or "")
        model = self._by_display_name(old)
        if model is None:
            return self._not_found(start_response, "POST", path)
        problem = _invalid_model_name(new)
        if problem is not None:
            return self._error(
                start_response, 400, "INVALID_ARGUMENT", problem, "POST", path
            )
        status, _, payload = self._call_flask(environ, rest, body=raw)
        if status != 200:
            return self._reply(
                start_response, status, json.loads(payload or b"{}"), "POST", path
            )
        with self._lock:
            model.name = new
            model.updated = time.time()
        return self._reply(
            start_response,
            200,
            {"registeredModel": self._model_json(model)},
            "POST",
            path,
        )

    def _query_delete(
        self, environ: Environ, start_response: StartResponse, rest: str
    ) -> Iterable[bytes]:
        """``DELETE`` with HTTP/JSON transcoding: body ignored, query bound."""
        path = str(environ.get("PATH_INFO", ""))
        self._body(environ)
        fields: dict[str, str] = {}
        query = urllib.parse.parse_qsl(
            str(environ.get("QUERY_STRING", "")), keep_blank_values=True
        )
        for key, value in query:
            if key == "key":
                return self._error(
                    start_response,
                    400,
                    "INVALID_ARGUMENT",
                    "API key not valid. Please pass a valid API key.",
                    "DELETE",
                    path,
                )
            fields[_DELETE_FIELDS.get(key, key)] = value
        status, headers, payload = self._call_flask(
            environ, rest, body=json.dumps(fields).encode(), query=""
        )
        with self._lock:
            self.rest_calls.append(("DELETE", rest, status))
        start_response(f"{status} X", headers)
        return [payload]

    def _search_models(
        self, environ: Environ, start_response: StartResponse
    ) -> Iterable[bytes]:
        status, _, payload = self._call_flask(
            environ, "/api/2.0/mlflow/registered-models/search"
        )
        path = str(environ.get("PATH_INFO", ""))
        if status != 200:
            return self._reply(
                start_response, status, json.loads(payload or b"{}"), "GET", path
            )
        data = json.loads(payload or b"{}")
        items: list[dict[str, Any]] = []
        for entry in data.get("registered_models", []):
            model = self._by_display_name(str(entry.get("name", "")))
            if model is not None:
                items.append(self._model_json(model))
        result: dict[str, Any] = {"registeredModels": items}
        if data.get("next_page_token"):
            result["nextPageToken"] = data["next_page_token"]
        return self._reply(start_response, 200, result, "GET", path)

    def _download_uri(
        self, environ: Environ, start_response: StartResponse
    ) -> Iterable[bytes]:
        path = str(environ.get("PATH_INFO", ""))
        query = dict(urllib.parse.parse_qsl(str(environ.get("QUERY_STRING", ""))))
        name, version = query.get("name", ""), query.get("version", "")
        model = self._by_display_name(name)
        if model is None:
            return self._not_found(start_response, "GET", path)
        status, _, payload = self._call_flask(
            environ,
            "/api/2.0/mlflow/model-versions/get",
            query=urllib.parse.urlencode({"name": name, "version": version}),
        )
        if status != 200:
            return self._reply(
                start_response, status, json.loads(payload or b"{}"), "GET", path
            )
        if self.download_uri_mode == "rewrite":
            uri = f"{model.storage_location}/versions/{version}"
        else:
            stored = json.loads(payload or b"{}").get("model_version") or {}
            uri = str(stored.get("source") or "")
        return self._reply(start_response, 200, {"artifact_uri": uri}, "GET", path)

    def _resolve_source_gcs_uri(
        self, source: str, run_id: str, model_id: str = ""
    ) -> str | None:
        if source.startswith("gs://"):
            return source.rstrip("/")
        store = self._handlers._get_tracking_store()
        if source.startswith("runs:/"):
            body = source[len("runs:/") :].lstrip("/")
            rid, _, subpath = body.partition("/")
            run = store.get_run(rid)
            base = str(run.info.artifact_uri).rstrip("/")
            return f"{base}/{subpath}".rstrip("/") if subpath else base
        if source.startswith("models:/"):
            body = source[len("models:/") :].lstrip("/")
            if body.startswith("m-"):
                logged = store.get_logged_model(body.split("/", 1)[0])
                return str(logged.artifact_location).rstrip("/")
        if model_id:
            logged = store.get_logged_model(model_id)
            return str(logged.artifact_location).rstrip("/")
        if run_id:
            run = store.get_run(run_id)
            base = str(run.info.artifact_uri).rstrip("/")
            subpath = source.strip("/")
            return f"{base}/{subpath}".rstrip("/") if subpath else base
        return None

    def _copy_to_vault(
        self, src_gcs_uri: str, model: _RegisteredModel, version: str
    ) -> None:
        parsed_src = urllib.parse.urlsplit(src_gcs_uri)
        if parsed_src.scheme != "gs" or not parsed_src.netloc:
            return
        src_bucket = parsed_src.netloc
        src_prefix = parsed_src.path.strip("/")
        dst_prefix = f"registeredModels/{model.uid}/versions/{version}"
        names = self.gcs.list_names(src_bucket, src_prefix)
        for name in names:
            if name != src_prefix and not name.startswith(src_prefix + "/"):
                continue
            rel = name[len(src_prefix) :].lstrip("/")
            dst_name = f"{dst_prefix}/{rel}" if rel else dst_prefix
            data = self.gcs.get_object(src_bucket, name)
            if data is not None:
                self.gcs.put_object(REGISTRY_BUCKET, dst_name, data)

    def _create_version(
        self, environ: Environ, start_response: StartResponse, rest: str
    ) -> Iterable[bytes]:
        path = str(environ.get("PATH_INFO", ""))
        request = json.loads(self._body(environ) or b"{}")
        namespace = request.pop("source_namespace", None)
        name = str(request.get("name") or "")
        source = str(request.get("source") or "")
        run_id = str(request.get("run_id") or "")
        model_id = str(request.get("model_id") or "")
        if len(source) > MAX_SOURCE_LENGTH:
            return self._error(
                start_response,
                400,
                "INVALID_ARGUMENT",
                f"source must be at most {MAX_SOURCE_LENGTH} characters",
                "POST",
                path,
            )
        if ".." in source:
            return self._error(
                start_response,
                400,
                "INVALID_ARGUMENT",
                "source must not contain '..'",
                "POST",
                path,
            )
        if namespace:
            with self._lock:
                self.source_namespaces.append(str(namespace))
        model = self._ensure_model_record(name) if name else None
        status, headers, payload = self._call_flask(
            environ, rest, body=json.dumps(request).encode()
        )
        if status == 200 and model is not None:
            data = json.loads(payload or b"{}")
            mv = data.get("model_version") or {}
            version = str(mv.get("version") or "1")
            src_gcs = self._resolve_source_gcs_uri(source, run_id, model_id)
            if src_gcs is not None:
                self._copy_to_vault(src_gcs, model, version)
        with self._lock:
            self.rest_calls.append(("POST", rest, status))
        start_response(f"{status} X", headers)
        return [payload]
