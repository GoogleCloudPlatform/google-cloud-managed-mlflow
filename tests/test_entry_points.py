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

"""Tests validating entry point definitions and loadability."""

from __future__ import annotations

import importlib.metadata
from pathlib import Path

import pytest

EXPECTED_ENTRY_POINTS: tuple[tuple[str, str, str], ...] = (
    (
        "mlflow.request_auth_provider",
        "google-cloud-auth",
        "google_cloud_managed_mlflow.auth:GoogleCloudAuthProvider",
    ),
    (
        "mlflow.tracking_store",
        "https",
        "google_cloud_managed_mlflow.tracking_store:build_tracking_store",
    ),
    (
        "mlflow.model_registry_store",
        "https",
        "google_cloud_managed_mlflow.registry_store:build_registry_store",
    ),
    (
        "mlflow.artifact_repository",
        "gs",
        "google_cloud_managed_mlflow.gcs_artifact_repo:GcpGCSArtifactRepository",
    ),
    (
        "mlflow.artifact_repository",
        "models",
        "google_cloud_managed_mlflow.models_artifact_repo:GcpModelsArtifactRepository",
    ),
    (
        "mlflow.artifact_repository",
        "mlflow-gs",
        "google_cloud_managed_mlflow.spark.hooks:MlflowGsArtifactRepository",
    ),
)


def test_pyproject_entry_points_match_expected() -> None:
    """Verifies entry point declarations in pyproject.toml via stdlib tomllib."""
    tomllib = pytest.importorskip("tomllib", reason="tomllib requires Python >= 3.11")

    pyproject_path = Path(__file__).resolve().parent.parent / "pyproject.toml"
    assert pyproject_path.is_file(), f"pyproject.toml not found at {pyproject_path}"

    with pyproject_path.open("rb") as f:
        data = tomllib.load(f)

    raw_groups: dict[str, dict[str, str]] = data["project"]["entry-points"]
    declared = {
        (group, name, value)
        for group, entries in raw_groups.items()
        for name, value in entries.items()
    }
    assert declared == set(EXPECTED_ENTRY_POINTS)


def test_installed_distribution_entry_points_match_expected() -> None:
    """Verifies entry point declarations in the installed distribution."""
    try:
        dist = importlib.metadata.distribution("google-cloud-managed-mlflow")
    except importlib.metadata.PackageNotFoundError:
        pytest.skip("google-cloud-managed-mlflow distribution is not installed")

    installed = [(ep.group, ep.name, ep.value) for ep in dist.entry_points]
    assert sorted(installed) == sorted(EXPECTED_ENTRY_POINTS)

    for installed_ep in dist.entry_points:
        assert callable(installed_ep.load())


@pytest.mark.parametrize(("group", "name", "value"), EXPECTED_ENTRY_POINTS)
def test_entry_point_loadable(group: str, name: str, value: str) -> None:
    """Verifies entry point resolution and loadability."""
    ep = importlib.metadata.EntryPoint(name=name, value=value, group=group)
    loaded = ep.load()
    assert loaded is not None
    assert callable(loaded)
