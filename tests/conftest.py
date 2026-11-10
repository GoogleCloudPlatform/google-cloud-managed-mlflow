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

"""Pytest configuration: global test isolation plus hermetic fake Google services.

* Plugin cache/singleton resets and a mocked ``google.auth.default`` for unit tests.
* Loopback-only networking plus fake Cloud Storage / Managed MLflow for Spark tests.
"""

from __future__ import annotations

import dataclasses
import os
import socket
import sys
import unittest.mock
from collections.abc import Generator, Iterator, MutableMapping
from pathlib import Path
from typing import Any

import mlflow
import pytest
from google.auth.credentials import AnonymousCredentials

from google_cloud_managed_mlflow.auth import (
    AUTH_PROVIDER_NAME,
    GoogleCloudBearerAuth,
    ensure_auth_provider_registered,
)
from google_cloud_managed_mlflow.spark import cab_scope as cab
from tests.fakes.gcs import FakeGcs
from tests.fakes.mlflow_server import CALLER_TOKEN, FakeManagedMlflow


def _reset_plugin_caches() -> None:
    if "google_cloud_managed_mlflow.auth" in sys.modules:
        from google_cloud_managed_mlflow.auth import GoogleCloudBearerAuth

        GoogleCloudBearerAuth.reset_for_testing()

    if "google_cloud_managed_mlflow.cab_credentials" in sys.modules:
        from google_cloud_managed_mlflow.cab_credentials import (
            reset_cab_credentials_cache,
        )

        reset_cab_credentials_cache()

    if "google_cloud_managed_mlflow.gcs_artifact_repo" in sys.modules:
        from google_cloud_managed_mlflow.gcs_artifact_repo import (
            reset_gcs_artifact_repo_state,
        )

        reset_gcs_artifact_repo_state()


@pytest.fixture(autouse=True)
def _reset_plugin_global_caches() -> Generator[None, None, None]:
    """Ensure in-memory caches and singletons are cleanly reset between test cases."""
    _reset_plugin_caches()
    try:
        yield
    finally:
        _reset_plugin_caches()


_AMBIENT_GCP_ENV_VARS = (
    "GOOGLE_APPLICATION_CREDENTIALS",
    "GOOGLE_CLOUD_PROJECT",
    "GCLOUD_PROJECT",
    "CLOUDSDK_CORE_PROJECT",
)


@pytest.fixture(autouse=True)
def _mock_google_auth_default(
    monkeypatch: pytest.MonkeyPatch,
) -> Generator[None, None, None]:
    """Provide fallback mock credentials so unit tests never depend on ambient ADC."""
    for env_var in _AMBIENT_GCP_ENV_VARS:
        monkeypatch.delenv(env_var, raising=False)
    with unittest.mock.patch(
        "google.auth.default",
        side_effect=lambda *args, **kwargs: (
            AnonymousCredentials(),
            "mock-default-project",
        ),
    ):
        yield


_LOOPBACK_NAMES = {"localhost", "127.0.0.1", "::1", "ip6-localhost"}


def _is_loopback(host: Any) -> bool:
    if host is None:
        return True
    text = host.decode() if isinstance(host, bytes) else str(host)
    return text in _LOOPBACK_NAMES or text.startswith("127.")


@pytest.fixture(autouse=True, scope="session")
def _hermetic_environment(tmp_path_factory: pytest.TempPathFactory) -> Iterator[None]:
    """Block non-loopback DNS and hide ambient Google credentials for all tests."""
    original = socket.getaddrinfo

    def guarded(host: Any, *args: Any, **kwargs: Any) -> Any:
        if not _is_loopback(host):
            raise OSError(f"Non-loopback network access is blocked in tests: {host!r}")
        return original(host, *args, **kwargs)

    saved = dict(os.environ)
    sdk_config = tmp_path_factory.mktemp("gcloud-config")
    os.environ.pop("GOOGLE_APPLICATION_CREDENTIALS", None)
    os.environ["CLOUDSDK_CONFIG"] = str(sdk_config)
    os.environ["GCE_METADATA_HOST"] = "127.0.0.1:9"
    os.environ["GCE_METADATA_IP"] = "127.0.0.1:9"
    os.environ["NO_GCE_CHECK"] = "True"
    os.environ["MLFLOW_DISABLE_TELEMETRY"] = "true"
    os.environ["DO_NOT_TRACK"] = "true"
    socket.getaddrinfo = guarded
    try:
        yield
    finally:
        socket.getaddrinfo = original
        os.environ.clear()
        os.environ.update(saved)


class StaticCredentials:
    """Caller credentials that attach a fixed fake bearer token."""

    def __init__(self, token: str = CALLER_TOKEN) -> None:
        self.token = token

    def before_request(
        self, request: Any, method: str, url: str, headers: MutableMapping[str, str]
    ) -> None:
        headers["authorization"] = f"Bearer {self.token}"


@pytest.fixture(scope="session")
def fake_gcs() -> Iterator[FakeGcs]:
    """Session-wide fake Cloud Storage."""
    gcs = FakeGcs().start()
    try:
        yield gcs
    finally:
        gcs.stop()


@pytest.fixture(scope="session")
def fake_mlflow(
    fake_gcs: FakeGcs, tmp_path_factory: pytest.TempPathFactory
) -> Iterator[FakeManagedMlflow]:
    """Session-wide fake Managed MLflow server."""
    server = FakeManagedMlflow(
        fake_gcs, tmp_path_factory.mktemp("managed-mlflow")
    ).start()
    try:
        yield server
    finally:
        server.stop()


@dataclasses.dataclass
class ManagedEnv:
    """Handles to the fakes plus the configured Managed MLflow URIs."""

    gcs: FakeGcs
    server: FakeManagedMlflow
    tracking_uri: str
    registry_uri: str
    workdir: Path


@pytest.fixture
def managed(
    fake_gcs: FakeGcs,
    fake_mlflow: FakeManagedMlflow,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> Iterator[ManagedEnv]:
    """Point MLflow at the fake Managed MLflow Namespace and Registry."""
    fake_mlflow.reset_counters()
    fake_gcs.clear_records()
    monkeypatch.setenv("MLFLOW_GCP_ENDPOINT_OVERRIDE", fake_mlflow.origin)
    monkeypatch.setenv("STORAGE_EMULATOR_HOST", fake_gcs.endpoint)
    monkeypatch.delenv(cab.REFRESH_MARGIN_ENV, raising=False)
    ensure_auth_provider_registered()
    monkeypatch.setenv("MLFLOW_TRACKING_AUTH", AUTH_PROVIDER_NAME)
    GoogleCloudBearerAuth.reset_for_testing()
    GoogleCloudBearerAuth.get_instance().credentials = StaticCredentials()
    cab.reset_shared_credentials()
    previous_tracking = mlflow.get_tracking_uri()
    previous_registry = mlflow.get_registry_uri()
    mlflow.set_tracking_uri(fake_mlflow.tracking_uri)
    mlflow.set_registry_uri(fake_mlflow.registry_uri)
    try:
        yield ManagedEnv(
            gcs=fake_gcs,
            server=fake_mlflow,
            tracking_uri=fake_mlflow.tracking_uri,
            registry_uri=fake_mlflow.registry_uri,
            workdir=tmp_path,
        )
    finally:
        while mlflow.active_run() is not None:
            mlflow.end_run()
        mlflow.set_tracking_uri(previous_tracking)
        mlflow.set_registry_uri(previous_registry)
        GoogleCloudBearerAuth.reset_for_testing()
        cab.reset_shared_credentials()
