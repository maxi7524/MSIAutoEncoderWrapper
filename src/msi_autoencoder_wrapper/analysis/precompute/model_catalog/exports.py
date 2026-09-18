"""Persist resolved aliases and the shared visualization contract for notebooks."""

from __future__ import annotations

from pathlib import Path

from .resolver import ResolvedModelCatalog


def write_catalog_artifacts(directory: Path, catalog: ResolvedModelCatalog) -> None:
    """Write auditable model and visualization tables beside the execution plan.

    :param directory: Strategy control directory, normally ``<notebook_dir>/precompute``.
    :type directory: pathlib.Path
    :param catalog: Resolved aliases and selected model records.
    :type catalog: ResolvedModelCatalog
    """
    directory.mkdir(parents=True, exist_ok=True)
    catalog.records.to_csv(directory / "resolved_model_catalog.csv", index=False)
    catalog.styles.to_csv(directory / "visualization_contract.csv", index=False)
