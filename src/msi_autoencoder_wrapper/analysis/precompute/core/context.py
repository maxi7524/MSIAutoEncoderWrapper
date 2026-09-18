"""Runtime context passed from the orchestrator to analysis plugins."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..model_catalog.resolver import ResolvedModelCatalog
from .artifacts import ArtifactStore
from .resources import ResourceManager


@dataclass
class AnalysisContext:
    """Resolved inputs and shared services for one strategy invocation.

    :param settings: Fully resolved analysis settings.
    :type settings: dict[str, typing.Any]
    :param catalog: Concrete aliases resolved from all configured sources.
    :type catalog: ResolvedModelCatalog
    :param artifacts: Result-directory and provenance writer.
    :type artifacts: ArtifactStore
    :param resources: In-memory resource pool shared by all strategy stages.
    :type resources: ResourceManager
    :param allow_cpu: Whether an explicitly requested CPU fallback is accepted.
    :type allow_cpu: bool
    """

    settings: dict[str, Any]
    catalog: ResolvedModelCatalog
    artifacts: ArtifactStore
    resources: ResourceManager
    allow_cpu: bool = False
