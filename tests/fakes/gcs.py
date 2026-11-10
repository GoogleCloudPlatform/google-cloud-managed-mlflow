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

"""Loopback Cloud Storage JSON API fake that enforces CAB token boundaries.

The fake implements the subset of the JSON API used by the Python
``google-cloud-storage`` client and the Hadoop Cloud Storage connector
(metadata, media download with ranges, listing, media/multipart/resumable
uploads with preconditions, delete, rewrite/copy, compose, patch, and JSON
batch requests). Every request must carry a token minted by the fake Managed
MLflow server; tokens are bound to one bucket and object prefix plus a mode:

* ``READ_ONLY``: get and list.
* ``READ_WRITE``: get, list, create, overwrite, and delete.
* ``READ_CREATE``: get, list, and create; overwrite and delete are refused.

Buckets registered as ambient accept unauthenticated requests (customer-owned
buckets accessed with ambient credentials).
"""

from __future__ import annotations

import base64
import dataclasses
import datetime
import gzip
import hashlib
import http.server
import json
import re
import struct
import threading
import time
import urllib.parse
import uuid
from collections.abc import Mapping
from typing import Any

try:
    import google_crc32c as _google_crc32c
except ImportError:  # pragma: no cover
    _google_crc32c = None  # type: ignore[assignment]

READ_ONLY = "READ_ONLY"
READ_WRITE = "READ_WRITE"
READ_CREATE = "READ_CREATE"

_RANGE_RE = re.compile(r"^bytes=(\d*)-(\d*)$")
_CONTENT_RANGE_RE = re.compile(r"^bytes (?:(\d+)-(\d+)|\*)/(\d+|\*)$")

Response = tuple[int, dict[str, str], bytes]


@dataclasses.dataclass
class StoredObject:
    """One object generation held by the fake."""

    data: bytes
    generation: int
    metageneration: int
    content_type: str
    created: float
    updated: float
    metadata: dict[str, str]


@dataclasses.dataclass(frozen=True)
class Grant:
    """A downscoped token's boundary."""

    label: str
    bucket: str
    prefix: str
    mode: str
    expires_at: float


@dataclasses.dataclass(frozen=True)
class RequestRecord:
    """Secret-free record of one handled request."""

    method: str
    path: str
    op: str
    bucket: str
    name: str
    grant: str
    status: int
    user_agent: str


@dataclasses.dataclass
class _UploadSession:
    bucket: str
    name: str
    content_type: str
    metadata: dict[str, str]
    preconditions: dict[str, str]
    grant: Grant | None
    data: bytearray


def _rfc3339(epoch: float) -> str:
    moment = datetime.datetime.fromtimestamp(epoch, tz=datetime.timezone.utc)
    return moment.strftime("%Y-%m-%dT%H:%M:%S.") + f"{moment.microsecond // 1000:03d}Z"


def _crc32c_b64(data: bytes) -> str:
    if _google_crc32c is not None:
        crc = int(_google_crc32c.value(data))
    else:  # pragma: no cover - pure-Python CRC32C fallback
        crc = 0xFFFFFFFF
        for byte in data:
            crc ^= byte
            for _ in range(8):
                crc = (crc >> 1) ^ (0x82F63B78 if (crc & 1) else 0)
        crc ^= 0xFFFFFFFF
    return base64.b64encode(struct.pack(">I", crc)).decode()


def _md5_b64(data: bytes) -> str:
    return base64.b64encode(hashlib.md5(data).digest()).decode()  # noqa: S324


def _error(status: int, message: str) -> Response:
    reason = {
        400: "invalid",
        401: "required",
        403: "forbidden",
        404: "notFound",
        412: "conditionNotMet",
        416: "requestedRangeNotSatisfiable",
    }.get(status, "error")
    body = {
        "error": {
            "code": status,
            "message": message,
            "errors": [{"message": message, "domain": "global", "reason": reason}],
        }
    }
    return status, {"Content-Type": "application/json"}, json.dumps(body).encode()


def _json(status: int, payload: Mapping[str, Any]) -> Response:
    return status, {"Content-Type": "application/json"}, json.dumps(payload).encode()


def _split_multipart(body: bytes, boundary: str) -> list[tuple[dict[str, str], bytes]]:
    parts: list[tuple[dict[str, str], bytes]] = []
    delimiter = b"--" + boundary.encode()
    for chunk in body.split(delimiter)[1:]:
        if chunk.startswith(b"--"):
            break
        chunk = chunk[2:] if chunk.startswith(b"\r\n") else chunk.lstrip(b"\n")
        head, sep, content = chunk.partition(b"\r\n\r\n")
        if not sep:
            head, _, content = chunk.partition(b"\n\n")
        headers: dict[str, str] = {}
        for line in head.decode("latin-1").splitlines():
            key, _, value = line.partition(":")
            if key.strip():
                headers[key.strip().lower()] = value.strip()
        if content.endswith(b"\r\n"):
            content = content[:-2]
        elif content.endswith(b"\n"):
            content = content[:-1]
        parts.append((headers, content))
    return parts


def _boundary_of(content_type: str) -> str:
    match = re.search(r'boundary="?([^";]+)"?', content_type)
    if not match:
        raise ValueError(f"No multipart boundary in {content_type!r}")
    return match.group(1)


class FakeGcs:
    """In-memory, thread-safe Cloud Storage JSON API served on 127.0.0.1."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._objects: dict[tuple[str, str], StoredObject] = {}
        self._grants: dict[str, Grant] = {}
        self._sessions: dict[str, _UploadSession] = {}
        self._generation = 1_700_000_000_000_000
        self.ambient_buckets: set[str] = set()
        self.records: list[RequestRecord] = []
        self.errors: list[str] = []
        self._server: http.server.ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    # ----------------------------------------------------------------- lifecycle

    def start(self) -> FakeGcs:
        """Start serving on an ephemeral loopback port."""
        fake = self

        class _Handler(_GcsRequestHandler):
            owner = fake

        server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
        server.daemon_threads = True
        self._server = server
        self._thread = threading.Thread(
            target=server.serve_forever, name="fake-gcs", daemon=True
        )
        self._thread.start()
        return self

    def stop(self) -> None:
        """Stop the server."""
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._server = None

    @property
    def endpoint(self) -> str:
        """Root URL, e.g. ``http://127.0.0.1:1234``."""
        assert self._server is not None
        host, port = self._server.server_address[:2]
        return f"http://{host!s}:{port}"

    # ------------------------------------------------------------------- state

    def reset(self) -> None:
        """Drop all objects, grants, sessions, and records."""
        with self._lock:
            self._objects.clear()
            self._grants.clear()
            self._sessions.clear()
            self.records.clear()
            self.errors.clear()
            self.ambient_buckets.clear()

    def grant(
        self, token: str, gcs_uri: str, mode: str, ttl_seconds: float, label: str
    ) -> None:
        """Register ``token`` for ``gs://bucket/prefix`` with ``mode``."""
        parsed = urllib.parse.urlsplit(gcs_uri)
        if parsed.scheme != "gs" or not parsed.netloc:
            raise ValueError(f"Not a gs:// URI: {gcs_uri!r}")
        with self._lock:
            self._grants[token] = Grant(
                label=label,
                bucket=parsed.netloc,
                prefix=parsed.path.strip("/"),
                mode=mode,
                expires_at=time.time() + ttl_seconds,
            )

    def clear_records(self) -> None:
        """Forget request records and handler errors (objects and grants stay)."""
        with self._lock:
            self.records.clear()
            self.errors.clear()

    def put_object(self, bucket: str, name: str, data: bytes) -> None:
        """Create or replace an object directly (test setup)."""
        with self._lock:
            self._store(bucket, name, data, "application/octet-stream", {})

    def get_object(self, bucket: str, name: str) -> bytes | None:
        """Return object bytes or None."""
        with self._lock:
            obj = self._objects.get((bucket, name))
            return None if obj is None else obj.data

    def list_names(self, bucket: str, prefix: str = "") -> list[str]:
        """Return sorted object names in ``bucket`` under ``prefix``."""
        with self._lock:
            return sorted(
                n for b, n in self._objects if b == bucket and n.startswith(prefix)
            )

    def denied(self) -> list[RequestRecord]:
        """Requests rejected with 401 or 403."""
        with self._lock:
            return [r for r in self.records if r.status in (401, 403)]

    def records_snapshot(self) -> list[RequestRecord]:
        """Copy of all request records."""
        with self._lock:
            return list(self.records)

    # -------------------------------------------------------------- authorization

    def _grant_for(self, headers: Mapping[str, str]) -> tuple[Grant | None, str]:
        auth = headers.get("authorization", "")
        if not auth:
            return None, "none"
        if not auth.lower().startswith("bearer "):
            return None, "invalid"
        token = auth[7:].strip()
        with self._lock:
            grant = self._grants.get(token)
        if grant is None:
            return None, "invalid"
        if time.time() >= grant.expires_at:
            return None, f"expired:{grant.label}"
        return grant, grant.label

    def _check(
        self, grant: Grant | None, label: str, bucket: str, name: str, access: str
    ) -> Response | None:
        if grant is None:
            if bucket in self.ambient_buckets and label == "none":
                return None
            return _error(401, f"Anonymous or invalid credentials ({label}).")
        if grant.bucket != bucket:
            return _error(403, f"Token {grant.label} is not valid for bucket {bucket}.")
        prefix = grant.prefix
        if access == "list":
            inside = not prefix or (
                name.startswith(prefix)
                and (len(name) == len(prefix) or name[len(prefix)] == "/")
            )
        else:
            inside = not prefix or name == prefix or name.startswith(prefix + "/")
        if not inside:
            return _error(403, f"Token {grant.label} does not cover {bucket}/{name}.")
        if access in ("read", "list"):
            return None
        if grant.mode == READ_ONLY:
            return _error(403, f"Token {grant.label} is read-only.")
        if access in ("overwrite", "delete", "update") and grant.mode == READ_CREATE:
            return _error(
                403, f"Token {grant.label} cannot overwrite or delete objects."
            )
        return None

    # ------------------------------------------------------------------ objects

    def _store(
        self,
        bucket: str,
        name: str,
        data: bytes,
        content_type: str,
        metadata: dict[str, str],
    ) -> StoredObject:
        self._generation += 1
        now = time.time()
        previous = self._objects.get((bucket, name))
        obj = StoredObject(
            data=bytes(data),
            generation=self._generation,
            metageneration=1,
            content_type=content_type or "application/octet-stream",
            created=now if previous is None else previous.created,
            updated=now,
            metadata=dict(metadata),
        )
        self._objects[(bucket, name)] = obj
        return obj

    def _object_json(self, bucket: str, name: str, obj: StoredObject) -> dict[str, Any]:
        quoted = urllib.parse.quote(name, safe="")
        payload: dict[str, Any] = {
            "kind": "storage#object",
            "id": f"{bucket}/{name}/{obj.generation}",
            "selfLink": f"{self.endpoint}/storage/v1/b/{bucket}/o/{quoted}",
            "mediaLink": f"{self.endpoint}/download/storage/v1/b/{bucket}/o/{quoted}"
            f"?generation={obj.generation}&alt=media",
            "name": name,
            "bucket": bucket,
            "generation": str(obj.generation),
            "metageneration": str(obj.metageneration),
            "contentType": obj.content_type,
            "storageClass": "STANDARD",
            "size": str(len(obj.data)),
            "md5Hash": _md5_b64(obj.data),
            "crc32c": _crc32c_b64(obj.data),
            "etag": f"CP{obj.generation}",
            "timeCreated": _rfc3339(obj.created),
            "updated": _rfc3339(obj.updated),
            "timeStorageClassUpdated": _rfc3339(obj.created),
        }
        if obj.metadata:
            payload["metadata"] = dict(obj.metadata)
        return payload

    def _preconditions_fail(
        self,
        existing: StoredObject | None,
        params: Mapping[str, str],
        prefix: str = "if",
    ) -> Response | None:
        match = params.get(f"{prefix}GenerationMatch")
        if match is not None and match != "":
            expected = int(match)
            if expected == 0 and existing is not None:
                return _error(
                    412,
                    "At least one of the pre-conditions you specified did not hold.",
                )
            if expected != 0 and (existing is None or existing.generation != expected):
                return _error(
                    412,
                    "At least one of the pre-conditions you specified did not hold.",
                )
        not_match = params.get(f"{prefix}GenerationNotMatch")
        if not_match and existing is not None and existing.generation == int(not_match):
            return _error(
                412, "At least one of the pre-conditions you specified did not hold."
            )
        meta_match = params.get(f"{prefix}MetagenerationMatch")
        if meta_match and (
            existing is None or existing.metageneration != int(meta_match)
        ):
            return _error(
                412, "At least one of the pre-conditions you specified did not hold."
            )
        return None

    def _insert(
        self,
        grant: Grant | None,
        label: str,
        bucket: str,
        name: str,
        data: bytes,
        content_type: str,
        metadata: dict[str, str],
        params: Mapping[str, str],
    ) -> Response:
        denied = self._check(grant, label, bucket, name, "create")
        if denied is not None:
            return denied
        with self._lock:
            existing = self._objects.get((bucket, name))
            failed = self._preconditions_fail(existing, params)
            if failed is not None:
                return failed
            if existing is not None:
                denied = self._check(grant, label, bucket, name, "overwrite")
                if denied is not None:
                    return denied
            obj = self._store(bucket, name, data, content_type, metadata)
            return _json(200, self._object_json(bucket, name, obj))

    # ----------------------------------------------------------------- dispatch

    def dispatch(
        self, method: str, raw_target: str, headers: Mapping[str, str], body: bytes
    ) -> tuple[Response, str, str, str, str]:
        """Handle one request; returns (response, op, bucket, name, grant label)."""
        parsed = urllib.parse.urlsplit(raw_target)
        params = dict(urllib.parse.parse_qsl(parsed.query, keep_blank_values=True))
        segments = [urllib.parse.unquote(s) for s in parsed.path.split("/") if s != ""]
        grant, label = self._grant_for(headers)
        op, bucket, name = "unknown", "", ""
        try:
            if segments[:1] == ["batch"]:
                return self._batch(headers, body), "batch", "", "", label
            if segments[:3] == ["download", "storage", "v1"] and len(segments) == 7:
                op, bucket, name = "media", segments[4], segments[6]
                return (
                    self._media(grant, label, bucket, name, params, headers),
                    op,
                    bucket,
                    name,
                    label,
                )
            if segments[:3] == ["upload", "storage", "v1"] and len(segments) == 6:
                bucket = segments[4]
                op, name, response = self._upload(
                    method, grant, label, bucket, params, headers, body
                )
                return response, op, bucket, name, label
            if (
                segments[:2] != ["storage", "v1"]
                or len(segments) < 4
                or segments[2] != "b"
            ):
                return (
                    _error(404, f"Unknown route {parsed.path}"),
                    op,
                    bucket,
                    name,
                    label,
                )
            bucket = segments[3]
            rest = segments[4:]
            if not rest:
                op = "bucket"
                return self._bucket(grant, label, bucket), op, bucket, name, label
            if rest == ["o"] and method == "GET":
                op, name = "list", params.get("prefix", "")
                return self._list(grant, label, bucket, params), op, bucket, name, label
            if len(rest) == 2 and rest[0] == "o":
                name = rest[1]
                if method == "GET":
                    if params.get("alt") == "media":
                        op = "media"
                        return (
                            self._media(grant, label, bucket, name, params, headers),
                            op,
                            bucket,
                            name,
                            label,
                        )
                    op = "get"
                    return (
                        self._get(grant, label, bucket, name, params),
                        op,
                        bucket,
                        name,
                        label,
                    )
                if method == "DELETE":
                    op = "delete"
                    return (
                        self._delete(grant, label, bucket, name, params),
                        op,
                        bucket,
                        name,
                        label,
                    )
                if method in ("PATCH", "PUT"):
                    op = "patch"
                    return (
                        self._patch(grant, label, bucket, name, body),
                        op,
                        bucket,
                        name,
                        label,
                    )
            if (
                len(rest) == 3
                and rest[0] == "o"
                and rest[2] == "compose"
                and method == "POST"
            ):
                op, name = "compose", rest[1]
                return (
                    self._compose(grant, label, bucket, name, params, body),
                    op,
                    bucket,
                    name,
                    label,
                )
            if (
                len(rest) == 7
                and rest[0] == "o"
                and rest[2] in ("rewriteTo", "copyTo")
                and rest[3] == "b"
                and rest[5] == "o"
                and method == "POST"
            ):
                op, name = rest[2], rest[6]
                response = self._copy(
                    grant,
                    label,
                    bucket,
                    rest[1],
                    rest[4],
                    rest[6],
                    params,
                    body,
                    rest[2],
                )
                return response, op, rest[4], name, label
            if (
                len(rest) == 5
                and rest[0] == "o"
                and rest[2] == "moveTo"
                and rest[3] == "o"
                and method == "POST"
            ):
                op, name = "moveTo", rest[4]
                response = self._move(grant, label, bucket, rest[1], rest[4], params)
                return response, op, bucket, name, label
            return (
                _error(404, f"Unknown route {method} {parsed.path}"),
                op,
                bucket,
                name,
                label,
            )
        except Exception as exc:  # pragma: no cover - surfaced as HTTP 500 in tests
            with self._lock:
                self.errors.append(
                    f"{method} {parsed.path}: {type(exc).__name__}: {exc}"
                )
            return (
                _error(500, f"fake GCS error: {type(exc).__name__}: {exc}"),
                op,
                bucket,
                name,
                label,
            )

    def _bucket(self, grant: Grant | None, label: str, bucket: str) -> Response:
        if grant is None and not (bucket in self.ambient_buckets and label == "none"):
            return _error(401, "Anonymous caller.")
        if grant is not None and grant.bucket != bucket:
            return _error(403, "Bucket not covered by token.")
        now = _rfc3339(time.time())
        return _json(
            200,
            {
                "kind": "storage#bucket",
                "id": bucket,
                "name": bucket,
                "projectNumber": "1",
                "metageneration": "1",
                "location": "US-CENTRAL1",
                "storageClass": "STANDARD",
                "timeCreated": now,
                "updated": now,
                "hierarchicalNamespace": {"enabled": False},
            },
        )

    def _get(
        self,
        grant: Grant | None,
        label: str,
        bucket: str,
        name: str,
        params: Mapping[str, str],
    ) -> Response:
        denied = self._check(grant, label, bucket, name, "read")
        if denied is not None:
            return denied
        with self._lock:
            obj = self._objects.get((bucket, name))
            if obj is None or (
                params.get("generation") and str(obj.generation) != params["generation"]
            ):
                return _error(404, f"No such object: {bucket}/{name}")
            return _json(200, self._object_json(bucket, name, obj))

    def _media(
        self,
        grant: Grant | None,
        label: str,
        bucket: str,
        name: str,
        params: Mapping[str, str],
        headers: Mapping[str, str],
    ) -> Response:
        denied = self._check(grant, label, bucket, name, "read")
        if denied is not None:
            return denied
        with self._lock:
            obj = self._objects.get((bucket, name))
        if obj is None or (
            params.get("generation") and str(obj.generation) != params["generation"]
        ):
            return _error(404, f"No such object: {bucket}/{name}")
        data = obj.data
        total = len(data)
        base_headers = {
            "Content-Type": obj.content_type,
            "x-goog-generation": str(obj.generation),
            "x-goog-metageneration": str(obj.metageneration),
            "x-goog-stored-content-length": str(total),
            "x-goog-stored-content-encoding": "identity",
            "x-goog-hash": f"crc32c={_crc32c_b64(data)},md5={_md5_b64(data)}",
            "Accept-Ranges": "bytes",
        }
        range_header = headers.get("range", "").strip()
        match = _RANGE_RE.match(range_header) if range_header else None
        if match is None:
            return 200, base_headers, data
        start_raw, end_raw = match.groups()
        if start_raw == "":
            suffix = int(end_raw or "0")
            start, end = max(0, total - suffix), total - 1
        else:
            start = int(start_raw)
            end = int(end_raw) if end_raw else total - 1
        if total == 0 or start >= total:
            if total == 0 and start == 0:
                failed = dict(base_headers)
                failed["Content-Range"] = "bytes */0"
                return 416, failed, b""
            failed = dict(base_headers)
            failed["Content-Range"] = f"bytes */{total}"
            return 416, failed, b""
        end = min(end, total - 1)
        partial = dict(base_headers)
        partial["Content-Range"] = f"bytes {start}-{end}/{total}"
        return 206, partial, data[start : end + 1]

    def _list(
        self, grant: Grant | None, label: str, bucket: str, params: Mapping[str, str]
    ) -> Response:
        prefix = params.get("prefix", "")
        denied = self._check(grant, label, bucket, prefix, "list")
        if denied is not None:
            return denied
        delimiter = params.get("delimiter", "")
        include_trailing = params.get("includeTrailingDelimiter", "").lower() == "true"
        start_offset = params.get("startOffset", "")
        end_offset = params.get("endOffset", "")
        with self._lock:
            names = sorted(
                n
                for b, n in self._objects
                if b == bucket
                and n.startswith(prefix)
                and (not start_offset or n >= start_offset)
                and (not end_offset or n < end_offset)
            )
            entries: dict[str, str] = {}
            for name in names:
                rest = name[len(prefix) :]
                if delimiter and delimiter in rest:
                    rolled = prefix + rest[: rest.index(delimiter) + len(delimiter)]
                    entries.setdefault(rolled, "prefix")
                    if include_trailing and name == rolled:
                        entries[rolled + "\x00item"] = "item"
                    continue
                entries[name] = "item"
            ordered = sorted(entries.items())
            offset = int(params.get("pageToken") or "0")
            max_results = int(params.get("maxResults") or "1000")
            page = ordered[offset : offset + max_results]
            items: list[dict[str, Any]] = []
            prefixes: list[str] = []
            for key, kind in page:
                if kind == "prefix":
                    prefixes.append(key)
                else:
                    real = key.removesuffix("\x00item")
                    items.append(
                        self._object_json(bucket, real, self._objects[(bucket, real)])
                    )
        payload: dict[str, Any] = {"kind": "storage#objects"}
        if items:
            payload["items"] = items
        if prefixes:
            payload["prefixes"] = prefixes
        if offset + max_results < len(ordered):
            payload["nextPageToken"] = str(offset + max_results)
        return _json(200, payload)

    def _delete(
        self,
        grant: Grant | None,
        label: str,
        bucket: str,
        name: str,
        params: Mapping[str, str],
    ) -> Response:
        denied = self._check(grant, label, bucket, name, "delete")
        if denied is not None:
            return denied
        with self._lock:
            existing = self._objects.get((bucket, name))
            if existing is None:
                return _error(404, f"No such object: {bucket}/{name}")
            failed = self._preconditions_fail(existing, params)
            if failed is not None:
                return failed
            del self._objects[(bucket, name)]
        return 204, {}, b""

    def _patch(
        self, grant: Grant | None, label: str, bucket: str, name: str, body: bytes
    ) -> Response:
        denied = self._check(grant, label, bucket, name, "update")
        if denied is not None:
            return denied
        update = json.loads(body or b"{}")
        with self._lock:
            obj = self._objects.get((bucket, name))
            if obj is None:
                return _error(404, f"No such object: {bucket}/{name}")
            if isinstance(update.get("metadata"), dict):
                obj.metadata.update(
                    {str(k): str(v) for k, v in update["metadata"].items()}
                )
            if update.get("contentType"):
                obj.content_type = str(update["contentType"])
            obj.metageneration += 1
            obj.updated = time.time()
            return _json(200, self._object_json(bucket, name, obj))

    def _copy(
        self,
        grant: Grant | None,
        label: str,
        src_bucket: str,
        src_name: str,
        dst_bucket: str,
        dst_name: str,
        params: Mapping[str, str],
        body: bytes,
        verb: str,
    ) -> Response:
        denied = self._check(grant, label, src_bucket, src_name, "read")
        if denied is not None:
            return denied
        with self._lock:
            source = self._objects.get((src_bucket, src_name))
            if source is None:
                return _error(404, f"No such object: {src_bucket}/{src_name}")
            failed = self._preconditions_fail(source, params, prefix="ifSource")
            if failed is not None:
                return failed
            metadata = dict(source.metadata)
            content_type = source.content_type
            try:
                overrides = json.loads(body) if body.strip() else {}
            except ValueError:
                overrides = {}
            if isinstance(overrides, dict):
                if isinstance(overrides.get("metadata"), dict):
                    metadata.update(
                        {str(k): str(v) for k, v in overrides["metadata"].items()}
                    )
                content_type = str(overrides.get("contentType") or content_type)
            status, response_headers, payload = self._insert(
                grant,
                label,
                dst_bucket,
                dst_name,
                source.data,
                content_type,
                metadata,
                params,
            )
        if status != 200 or verb == "copyTo":
            return status, response_headers, payload
        resource = json.loads(payload)
        size = str(len(source.data))
        return _json(
            200,
            {
                "kind": "storage#rewriteResponse",
                "totalBytesRewritten": size,
                "objectSize": size,
                "done": True,
                "resource": resource,
            },
        )

    def _move(
        self,
        grant: Grant | None,
        label: str,
        bucket: str,
        src_name: str,
        dst_name: str,
        params: Mapping[str, str],
    ) -> Response:
        for name, access in (
            (src_name, "read"),
            (src_name, "delete"),
            (dst_name, "create"),
        ):
            denied = self._check(grant, label, bucket, name, access)
            if denied is not None:
                return denied
        with self._lock:
            source = self._objects.get((bucket, src_name))
            if source is None:
                return _error(404, f"No such object: {bucket}/{src_name}")
            failed = self._preconditions_fail(source, params, prefix="ifSource")
            if failed is not None:
                return failed
            status, headers, payload = self._insert(
                grant,
                label,
                bucket,
                dst_name,
                source.data,
                source.content_type,
                dict(source.metadata),
                params,
            )
            if status == 200:
                del self._objects[(bucket, src_name)]
            return status, headers, payload

    def _compose(
        self,
        grant: Grant | None,
        label: str,
        bucket: str,
        name: str,
        params: Mapping[str, str],
        body: bytes,
    ) -> Response:
        request = json.loads(body or b"{}")
        chunks: list[bytes] = []
        with self._lock:
            for source in request.get("sourceObjects", []):
                src_name = str(source.get("name", ""))
                denied = self._check(grant, label, bucket, src_name, "read")
                if denied is not None:
                    return denied
                obj = self._objects.get((bucket, src_name))
                if obj is None:
                    return _error(404, f"No such object: {bucket}/{src_name}")
                chunks.append(obj.data)
            destination = request.get("destination") or {}
            return self._insert(
                grant,
                label,
                bucket,
                name,
                b"".join(chunks),
                str(destination.get("contentType") or "application/octet-stream"),
                {},
                params,
            )

    def _upload(
        self,
        method: str,
        grant: Grant | None,
        label: str,
        bucket: str,
        params: Mapping[str, str],
        headers: Mapping[str, str],
        body: bytes,
    ) -> tuple[str, str, Response]:
        upload_type = params.get("uploadType", "")
        if method == "PUT" or (method == "POST" and params.get("upload_id")):
            return self._resumable_chunk(params, headers, body)
        if upload_type == "media":
            name = params.get("name", "")
            content_type = headers.get("content-type", "application/octet-stream")
            return (
                "insert",
                name,
                self._insert(
                    grant, label, bucket, name, body, content_type, {}, params
                ),
            )
        if upload_type == "multipart":
            parts = _split_multipart(
                body, _boundary_of(headers.get("content-type", ""))
            )
            if len(parts) < 2:
                return "insert", "", _error(400, "Multipart upload requires two parts.")
            metadata = json.loads(parts[0][1] or b"{}")
            name = str(metadata.get("name") or params.get("name", ""))
            content_type = str(
                metadata.get("contentType")
                or parts[1][0].get("content-type")
                or "application/octet-stream"
            )
            custom = (
                metadata.get("metadata")
                if isinstance(metadata.get("metadata"), dict)
                else {}
            )
            return (
                "insert",
                name,
                self._insert(
                    grant,
                    label,
                    bucket,
                    name,
                    parts[1][1],
                    content_type,
                    {str(k): str(v) for k, v in (custom or {}).items()},
                    params,
                ),
            )
        if upload_type == "resumable":
            metadata = json.loads(body) if body.strip() else {}
            name = str(metadata.get("name") or params.get("name", ""))
            denied = self._check(grant, label, bucket, name, "create")
            if denied is not None:
                return "resumable_start", name, denied
            session_id = uuid.uuid4().hex
            custom = (
                metadata.get("metadata")
                if isinstance(metadata.get("metadata"), dict)
                else {}
            )
            with self._lock:
                self._sessions[session_id] = _UploadSession(
                    bucket=bucket,
                    name=name,
                    content_type=str(
                        metadata.get("contentType")
                        or headers.get("x-upload-content-type")
                        or "application/octet-stream"
                    ),
                    metadata={str(k): str(v) for k, v in (custom or {}).items()},
                    preconditions={
                        k: v for k, v in params.items() if k.startswith(("if", "If"))
                    },
                    grant=grant,
                    data=bytearray(),
                )
            location = (
                f"{self.endpoint}/upload/storage/v1/b/{bucket}/o"
                f"?uploadType=resumable&upload_id={session_id}"
            )
            return "resumable_start", name, (200, {"Location": location}, b"")
        return "insert", "", _error(400, f"Unsupported uploadType {upload_type!r}")

    def _resumable_chunk(
        self, params: Mapping[str, str], headers: Mapping[str, str], body: bytes
    ) -> tuple[str, str, Response]:
        session_id = params.get("upload_id", "")
        with self._lock:
            session = self._sessions.get(session_id)
        if session is None:
            return "resumable_chunk", "", _error(404, "Unknown upload session.")
        content_range = headers.get("content-range", "").strip()
        total: int | None = None
        if content_range:
            match = _CONTENT_RANGE_RE.match(content_range)
            if match is None:
                return (
                    "resumable_chunk",
                    session.name,
                    _error(400, "Bad Content-Range."),
                )
            start_raw, _, total_raw = match.groups()
            if start_raw is not None:
                start = int(start_raw)
                del session.data[start:]
                session.data.extend(body)
            total = None if total_raw == "*" else int(total_raw)
        else:
            session.data.extend(body)
            total = len(session.data)
        if total is not None and len(session.data) >= total:
            label = session.grant.label if session.grant is not None else "none"
            response = self._insert(
                session.grant,
                label,
                session.bucket,
                session.name,
                bytes(session.data[:total]),
                session.content_type,
                session.metadata,
                session.preconditions,
            )
            with self._lock:
                self._sessions.pop(session_id, None)
            return "resumable_final", session.name, response
        progress: dict[str, str] = {}
        if session.data:
            progress["Range"] = f"bytes=0-{len(session.data) - 1}"
        return "resumable_chunk", session.name, (308, progress, b"")

    def _batch(self, headers: Mapping[str, str], body: bytes) -> Response:
        boundary = _boundary_of(headers.get("content-type", ""))
        out_boundary = f"batch_{uuid.uuid4().hex}"
        chunks: list[bytes] = []
        for part_headers, content in _split_multipart(body, boundary):
            request_line, _, rest = content.partition(b"\r\n")
            head, _, inner_body = rest.partition(b"\r\n\r\n")
            method, target, _ = request_line.decode("latin-1").split(" ", 2)
            inner_headers = {k.lower(): v for k, v in headers.items()}
            for line in head.decode("latin-1").splitlines():
                key, _, value = line.partition(":")
                if key.strip():
                    inner_headers[key.strip().lower()] = value.strip()
            if (
                inner_headers.get("content-encoding", "").lower() == "gzip"
                and inner_body
            ):
                inner_body = gzip.decompress(inner_body)
            split = urllib.parse.urlsplit(target)
            relative = split.path + (f"?{split.query}" if split.query else "")
            (status, response_headers, payload), op, bucket, name, label = (
                self.dispatch(method, relative, inner_headers, inner_body)
            )
            self._record(method, relative, op, bucket, name, label, status, headers)
            reason = http.HTTPStatus(status).phrase
            lines = [f"HTTP/1.1 {status} {reason}"]
            for key, value in response_headers.items():
                lines.append(f"{key}: {value}")
            lines.append(f"Content-Length: {len(payload)}")
            content_id = part_headers.get("content-id", "")
            response_id = content_id.replace("<", "<response-", 1) if content_id else ""
            chunk = (
                (
                    f"--{out_boundary}\r\nContent-Type: application/http\r\n"
                    + (f"Content-ID: {response_id}\r\n" if response_id else "")
                    + "\r\n"
                    + "\r\n".join(lines)
                    + "\r\n\r\n"
                ).encode("latin-1")
                + payload
                + b"\r\n"
            )
            chunks.append(chunk)
        chunks.append(f"--{out_boundary}--\r\n".encode())
        return (
            200,
            {"Content-Type": f"multipart/mixed; boundary={out_boundary}"},
            b"".join(chunks),
        )

    def _record(
        self,
        method: str,
        target: str,
        op: str,
        bucket: str,
        name: str,
        label: str,
        status: int,
        headers: Mapping[str, str],
    ) -> None:
        record = RequestRecord(
            method=method,
            path=urllib.parse.urlsplit(target).path,
            op=op,
            bucket=bucket,
            name=name,
            grant=label,
            status=status,
            user_agent=headers.get("user-agent", ""),
        )
        with self._lock:
            self.records.append(record)


class _GcsRequestHandler(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    owner: FakeGcs

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        return

    def _read_body(self) -> bytes:
        if self.headers.get("Transfer-Encoding", "").lower() == "chunked":
            data = bytearray()
            while True:
                size_line = self.rfile.readline().strip()
                size = int(size_line.split(b";")[0] or b"0", 16)
                if size == 0:
                    self.rfile.readline()
                    break
                data.extend(self.rfile.read(size))
                self.rfile.readline()
            return bytes(data)
        length = int(self.headers.get("Content-Length") or "0")
        return self.rfile.read(length) if length else b""

    def _decoded_body(self) -> bytes:
        body = self._read_body()
        if self.headers.get("Content-Encoding", "").lower() == "gzip" and body:
            return gzip.decompress(body)
        return body

    def _handle(self) -> None:
        body = self._decoded_body()
        headers = {k.lower(): v for k, v in self.headers.items()}
        (status, response_headers, payload), op, bucket, name, label = (
            self.owner.dispatch(self.command, self.path, headers, body)
        )
        if op != "batch":
            self.owner._record(
                self.command, self.path, op, bucket, name, label, status, headers
            )
        self.send_response(status)
        for key, value in response_headers.items():
            self.send_header(key, value)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        if self.command != "HEAD" and payload:
            self.wfile.write(payload)

    do_GET = _handle
    do_POST = _handle
    do_PUT = _handle
    do_PATCH = _handle
    do_DELETE = _handle
    do_HEAD = _handle
