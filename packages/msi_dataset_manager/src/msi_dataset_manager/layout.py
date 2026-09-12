"""Canonical paths shared by dataset-management operations."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DatasetWorkspaceLayout:
    """Resolve source images, cohort configuration, and merged outputs.

    :param workspace_path: Root containing ``datasets`` and ``configs``.
    :type workspace_path: pathlib.Path | str
    """

    workspace_path: Path

    def __init__(self, workspace_path: Path | str) -> None:
        object.__setattr__(self, "workspace_path", Path(workspace_path).resolve())

    @property
    def datasets_dir(self) -> Path:
        """Return the shared canonical source and merged-image directory."""
        return self.workspace_path / "datasets"

    def dataset_dir(self, dataset_id: str) -> Path:
        """Return ``datasets/<dataset_id>`` without provider path components."""
        return self.datasets_dir / dataset_id

    def imzml_path(self, dataset_id: str) -> Path:
        """Return the canonical imzML path for a source or merged dataset."""
        return self.dataset_dir(dataset_id) / f"{dataset_id}.imzML"

    def composed_catalog_path(self, cohort_id: str) -> Path:
        """Return the only SQLite catalogue, created during composition."""
        return self.dataset_dir(cohort_id) / f"{cohort_id}.sqlite"

    def composition_path(self, cohort_id: str) -> Path:
        """Return the normalized composition configuration path."""
        return self.dataset_dir(cohort_id) / "composition.json"

    def dataset_metadata_path(self, dataset_id: str) -> Path:
        """Return the normalized source-dataset metadata artifact path."""
        return self.dataset_dir(dataset_id) / "dataset_metadata.json"

    def cohort_metadata_path(self, cohort_id: str) -> Path:
        """Return the normalized metadata artifact for one composed cohort."""
        return self.dataset_dir(cohort_id) / "cohort_metadata.json"

    def database_annotations_dir(self, dataset_id: str) -> Path:
        """Return the sidecar directory for external candidate annotations."""
        return self.dataset_dir(dataset_id) / "database_annotations"

    def candidate_catalog_path(self, dataset_id: str) -> Path:
        """Return the normalized external-candidate SQLite catalog path."""
        return self.database_annotations_dir(dataset_id) / "candidates.sqlite"

    def candidate_catalog_manifest_path(self, dataset_id: str) -> Path:
        """Return the provenance manifest for one candidate catalog."""
        return self.database_annotations_dir(dataset_id) / "manifest.json"
