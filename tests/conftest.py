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

"""Pytest configuration and global test isolation fixtures."""

from __future__ import annotations

import sys
from collections.abc import Generator

import pytest


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
