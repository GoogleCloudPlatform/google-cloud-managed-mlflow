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

"""Validation and URL parsing primitives for Google Cloud Managed MLflow."""

from __future__ import annotations

import ipaddress
import os
import re
import urllib.parse
from typing import NamedTuple

from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE

_LOOPBACK_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})
_GOOGLEAPIS_DOMAIN_SUFFIX = ".googleapis.com"
_DNS_LABEL_RE = re.compile(r"^[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?$")
_UNBRACKETED_HOST_RE = re.compile(r"^[a-z0-9_-]+(?:\.[a-z0-9_-]+)*$")


class _EndpointOrigin(NamedTuple):
    """Normalized (scheme, hostname, effective_port) endpoint origin."""

    scheme: str
    hostname: str
    port: int

    def __str__(self) -> str:
        host_display = f"[{self.hostname}]" if ":" in self.hostname else self.hostname
        return f"{self.scheme}://{host_display}:{self.port}"


def _contains_control_chars(s: str) -> bool:
    """Return True if s contains ASCII C0 or DEL control characters.

    Must run before urllib.parse.urlsplit/urljoin, which silently strip
    '\\t', '\\r', and '\\n' (bpo-43882).
    """
    return any(ord(ch) < 0x20 or ord(ch) == 0x7F for ch in s)


def _parse_validated_endpoint_url(raw_url: str) -> _EndpointOrigin:
    """Parse and validate URL scheme, host, effective port, and HTTP policy."""
    if not raw_url:
        raise MlflowException(
            "Endpoint URL must be a non-empty string.",
            error_code=INVALID_PARAMETER_VALUE,
        )

    if _contains_control_chars(raw_url):
        raise MlflowException(
            "Endpoint URL contains invalid control characters.",
            error_code=INVALID_PARAMETER_VALUE,
        )

    if raw_url[0].isspace() or raw_url[-1].isspace():
        raise MlflowException(
            "Endpoint URL must be a non-empty string without "
            "leading/trailing whitespace.",
            error_code=INVALID_PARAMETER_VALUE,
        )

    try:
        parsed = urllib.parse.urlsplit(raw_url)
        raw_port = parsed.port
    except ValueError as exc:
        raise MlflowException(
            "Endpoint URL is malformed or contains an invalid port.",
            error_code=INVALID_PARAMETER_VALUE,
        ) from exc

    if raw_port == 0:
        raise MlflowException(
            "Endpoint URL is malformed or contains an invalid port.",
            error_code=INVALID_PARAMETER_VALUE,
        )

    raw_hostname = (parsed.hostname or "").lower()
    if (
        not parsed.scheme
        or not raw_hostname.rstrip(".")
        or "@" in parsed.netloc
        or "\\" in parsed.netloc
        or any(ch.isspace() for ch in parsed.netloc)
    ):
        raise MlflowException(
            "Endpoint URL must include a valid scheme and host.",
            error_code=INVALID_PARAMETER_VALUE,
        )

    if "[" in parsed.netloc or "]" in parsed.netloc:
        try:
            if "%" in raw_hostname:
                raise ValueError("IPv6 zone identifiers are not supported.")
            hostname = ipaddress.IPv6Address(raw_hostname).compressed
        except ValueError as exc:
            raise MlflowException(
                "Endpoint URL contains an invalid IPv6 host literal.",
                error_code=INVALID_PARAMETER_VALUE,
            ) from exc
    else:
        hostname = raw_hostname.rstrip(".")
        if not _UNBRACKETED_HOST_RE.fullmatch(hostname):
            raise MlflowException(
                "Endpoint URL contains invalid characters in host.",
                error_code=INVALID_PARAMETER_VALUE,
            )

    scheme = parsed.scheme.lower()
    if scheme not in ("https", "http"):
        raise MlflowException(
            f"Invalid scheme '{scheme}': only https:// and loopback http:// "
            "are allowed.",
            error_code=INVALID_PARAMETER_VALUE,
        )

    if scheme == "http" and hostname not in _LOOPBACK_HOSTS:
        raise MlflowException(
            f"Insecure HTTP is only permitted on loopback addresses "
            f"({', '.join(sorted(_LOOPBACK_HOSTS))}); got '{hostname}'.",
            error_code=INVALID_PARAMETER_VALUE,
        )

    effective_port = (
        raw_port if raw_port is not None else (443 if scheme == "https" else 80)
    )
    return _EndpointOrigin(scheme, hostname, effective_port)


def _is_allowed_destination(origin: _EndpointOrigin) -> bool:
    """Return True if origin is an allowed Google Cloud or override destination."""
    if (
        origin.scheme == "https"
        and origin.port == 443
        and origin.hostname.endswith(_GOOGLEAPIS_DOMAIN_SUFFIX)
        and all(_DNS_LABEL_RE.fullmatch(label) for label in origin.hostname.split("."))
    ):
        return True

    override_raw = os.environ.get("MLFLOW_GCP_ENDPOINT_OVERRIDE", "").strip(" ")
    if override_raw:
        try:
            override_origin = _parse_validated_endpoint_url(override_raw)
        except MlflowException:
            return False
        return origin == override_origin

    return False
