"""Persistent storage contracts for synthetic pretraining artifacts."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
import shutil
import time
from typing import Callable, Mapping
from uuid import uuid4

import numpy as np

from ...utils.logger import get_custom_logger

logger = get_custom_logger(__name__)

# REMARK: Version 2 invalidates provisional artifacts created before the
# persistent cache and batch-rendering contract were finalized.
ARTIFACT_SCHEMA_VERSION = 2


@dataclass(frozen=True)
class SyntheticManifest:
    """Compact declarations for one complete precomputed population."""

    component_ids: np.ndarray
    blank_centers: np.ndarray
    anchor_bins: np.ndarray

    def __post_init__(self) -> None:
        if self.component_ids.ndim != 2:
            raise ValueError("Manifest component IDs must have shape (N, K).")
        sample_count = self.component_ids.shape[0]
        if self.blank_centers.shape != (sample_count,):
            raise ValueError("Manifest blank centers must have shape (N,).")
        if self.anchor_bins.shape != (sample_count,):
            raise ValueError("Manifest anchor bins must have shape (N,).")


@dataclass(frozen=True)
class SyntheticPrecomputeArtifact:
    """Persistent sparse basis bank and compact population manifests."""

    fingerprint: str
    artifact_key: str
    axis: np.ndarray
    basis_rows: np.ndarray
    basis_columns: np.ndarray
    basis_values: np.ndarray
    prototype_target_indices: np.ndarray
    prototype_anchor_bins: np.ndarray
    manifests: Mapping[str, SyntheticManifest]
    blank_peak_radius: int
    normalization: str

    @property
    def feature_count(self) -> int:
        """Return active axis width."""
        return int(self.axis.size)

    @property
    def prototype_count(self) -> int:
        """Return immutable labelled component count."""
        return int(self.prototype_target_indices.size)


class SyntheticArtifactStore:
    """Persist immutable artifacts with atomic lock protection."""

    def __init__(self, root: Path) -> None:
        self.root = root

    def load_or_build(
        self,
        *,
        artifact_key: str,
        fingerprint: str,
        builder: Callable[[], SyntheticPrecomputeArtifact],
    ) -> SyntheticPrecomputeArtifact:
        """Load matching output or build it exactly once under a file lock."""
        self.root.mkdir(parents=True, exist_ok=True)
        target = self.root / f"{_safe_key(artifact_key)}-{fingerprint[:16]}"
        loaded = self._load(target, fingerprint)
        if loaded is not None:
            logger.info("Loaded synthetic precompute artifact: %s.", target)
            return loaded
        lock_path = self.root / f".{_safe_key(artifact_key)}-{fingerprint[:16]}.lock"
        with _artifact_lock(lock_path):
            loaded = self._load(target, fingerprint)
            if loaded is not None:
                logger.info("Loaded synthetic artifact after lock: %s.", target)
                return loaded
            artifact = builder()
            temporary = self.root / f".{target.name}.{uuid4().hex}.tmp"
            temporary.mkdir(parents=True)
            try:
                self._save(temporary, artifact)
                os.replace(temporary, target)
            except BaseException:
                shutil.rmtree(temporary, ignore_errors=True)
                raise
            logger.info(
                "Built synthetic artifact: %s prototypes=%s populations=%s.",
                target,
                artifact.prototype_count,
                tuple(artifact.manifests),
            )
            return artifact

    def _load(
        self,
        directory: Path,
        fingerprint: str,
    ) -> SyntheticPrecomputeArtifact | None:
        """Load complete matching arrays, rejecting partial or stale artifacts."""
        metadata_path = directory / "metadata.json"
        arrays_path = directory / "artifact.npz"
        if not metadata_path.is_file() or not arrays_path.is_file():
            return None
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if (
            metadata.get("schema_version") != ARTIFACT_SCHEMA_VERSION
            or metadata.get("fingerprint") != fingerprint
        ):
            return None
        with np.load(arrays_path, allow_pickle=False) as arrays:
            manifests = {
                name: SyntheticManifest(
                    component_ids=arrays[f"{name}__component_ids"],
                    blank_centers=arrays[f"{name}__blank_centers"],
                    anchor_bins=arrays[f"{name}__anchor_bins"],
                )
                for name in metadata["population_names"]
            }
            return SyntheticPrecomputeArtifact(
                fingerprint=str(metadata["fingerprint"]),
                artifact_key=str(metadata["artifact_key"]),
                axis=arrays["axis"],
                basis_rows=arrays["basis_rows"],
                basis_columns=arrays["basis_columns"],
                basis_values=arrays["basis_values"],
                prototype_target_indices=arrays["prototype_target_indices"],
                prototype_anchor_bins=arrays["prototype_anchor_bins"],
                manifests=manifests,
                blank_peak_radius=int(metadata["blank_peak_radius"]),
                normalization=str(metadata["normalization"]),
            )

    def _save(
        self,
        directory: Path,
        artifact: SyntheticPrecomputeArtifact,
    ) -> None:
        """Write arrays before atomically publishing completion metadata."""
        arrays: dict[str, np.ndarray] = {
            "axis": artifact.axis,
            "basis_rows": artifact.basis_rows,
            "basis_columns": artifact.basis_columns,
            "basis_values": artifact.basis_values,
            "prototype_target_indices": artifact.prototype_target_indices,
            "prototype_anchor_bins": artifact.prototype_anchor_bins,
        }
        for name, manifest in artifact.manifests.items():
            arrays[f"{name}__component_ids"] = manifest.component_ids
            arrays[f"{name}__blank_centers"] = manifest.blank_centers
            arrays[f"{name}__anchor_bins"] = manifest.anchor_bins
        np.savez_compressed(directory / "artifact.npz", **arrays)
        (directory / "metadata.json").write_text(
            json.dumps(
                {
                    "schema_version": ARTIFACT_SCHEMA_VERSION,
                    "artifact_key": artifact.artifact_key,
                    "fingerprint": artifact.fingerprint,
                    "blank_peak_radius": artifact.blank_peak_radius,
                    "normalization": artifact.normalization,
                    "population_names": list(artifact.manifests),
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )


@contextmanager
def _artifact_lock(path: Path, timeout_seconds: float = 300.0):
    """Serialize a builder and prevent publication of partial artifacts."""
    deadline = time.monotonic() + timeout_seconds
    descriptor: int | None = None
    while descriptor is None:
        try:
            descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"Timed out waiting for synthetic artifact lock: {path}."
                )
            time.sleep(0.1)
    try:
        os.write(descriptor, str(os.getpid()).encode("ascii"))
        yield
    finally:
        os.close(descriptor)
        path.unlink(missing_ok=True)


def _safe_key(value: str) -> str:
    """Normalize an artifact key into one filesystem-safe component."""
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-") or "synthetic"
