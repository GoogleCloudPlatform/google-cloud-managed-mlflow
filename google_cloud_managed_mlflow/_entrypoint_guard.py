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

"""Private stdlib-only entry-point guard and registration repair helpers."""

from __future__ import annotations

import contextlib
import importlib.metadata
import logging
import sys
import warnings
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from collections.abc import Iterator

logger = logging.getLogger(__name__)

# Matches warnings from
# mlflow.store.artifact.artifact_repository_registry.register_entrypoints
# (MLflow >= 3.17.2).
_REGISTRATION_WARNING_PATTERN = (
    r'Failure attempting to register artifact repository for scheme "(?:gs|models)"'
)


@contextlib.contextmanager
def quiet_registration_warning() -> Iterator[None]:
    """Context manager to suppress MLflow entry-point registration warnings.

    Temporarily silences warnings emitted when MLflow discovers this package's
    entry points while an import cycle or early-load condition is active. Filter
    is removed afterwards. Do not use warnings.catch_warnings(): it restores the
    whole filter list and would discard filters that mlflow and its
    dependencies install during their first import.
    """
    if "mlflow.store.artifact.artifact_repository_registry" in sys.modules:
        yield
        return
    warnings.filterwarnings(
        "ignore", message=_REGISTRATION_WARNING_PATTERN, category=UserWarning
    )
    # With Python 3.14 context-aware warnings inside catch_warnings(), the new
    # filter goes to a context-local list, so warnings.filters[0] may belong to
    # someone else. Only remember the entry if it is the one just added.
    filter_item: Any = None
    candidate = warnings.filters[0] if warnings.filters else None
    cand_pat = getattr(candidate[1], "pattern", None) if candidate is not None else None
    if (
        candidate is not None
        and candidate[0] == "ignore"
        and candidate[2] is UserWarning
        and cand_pat == _REGISTRATION_WARNING_PATTERN
    ):
        filter_item = candidate
    try:
        yield
    finally:
        if filter_item is not None:
            filters = cast("list[Any]", warnings.filters)
            for index, item in enumerate(filters):
                if item is filter_item:
                    del filters[index]
                    break


def repair_artifact_entrypoint(
    scheme: str,
    repo_cls: type[Any],
    stock_cls: type[Any],
    *,
    imported_before_mlflow: bool,
) -> None:
    """Repair an artifact repository scheme registration in MLflow's registry.

    If MLflow loaded early before our entry points could resolve without an import
    cycle, this function re-registers our repository class once prerequisites are met.
    """
    class_name = repo_cls.__name__
    try:
        registry_module = sys.modules.get(
            "mlflow.store.artifact.artifact_repository_registry"
        )
        registry = getattr(registry_module, "_artifact_repository_registry", None)
        if registry is None:
            if registry_module is not None and imported_before_mlflow:
                logger.warning(
                    f"Could not register {class_name} for '{scheme}': "
                    "MLflow's artifact repository registry was not found."
                )
            return
        current = registry.get_registered_artifact_repositories().get(scheme)
        if current is not stock_cls:
            return
        target = f"{repo_cls.__module__}:{class_name}"
        if any(
            ep.name == scheme and ep.value.replace(" ", "") == target
            for ep in importlib.metadata.entry_points(
                group="mlflow.artifact_repository"
            )
        ):
            registry.register(scheme, repo_cls)
    except Exception:
        logger.log(
            logging.WARNING if imported_before_mlflow else logging.DEBUG,
            f"Could not repair MLflow's '{scheme}' registration.",
            exc_info=True,
        )
