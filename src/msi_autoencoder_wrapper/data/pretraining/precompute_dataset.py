"""Batch-time rendering of compact synthetic pretraining manifests."""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from ..batches import SpectrumBatch
from ..spaces import SpectrumSpace
from ..supervision_masks import simulated_negative_mask_key
from ..targets import TargetBatch, TargetSchema
from .precompute_artifact import SyntheticPrecomputeArtifact


class PrecomputedSyntheticDataset(Dataset):
    """Expose static manifest rows through the existing ``SpectrumBatch`` API."""

    def __init__(
        self,
        artifact: SyntheticPrecomputeArtifact,
        *,
        population: str,
        schemas: Mapping[str, TargetSchema],
        dtype: torch.dtype,
        row_indices: np.ndarray | None = None,
        fixed_epoch: bool = False,
        seed: int = 0,
    ) -> None:
        if population not in artifact.manifests:
            raise ValueError(f"Unknown synthetic artifact population: {population}.")
        self.artifact = artifact
        self.population = population
        self.manifest = artifact.manifests[population]
        self.schemas = dict(schemas)
        self.dtype = dtype
        self.fixed_epoch = fixed_epoch
        self.seed = int(seed)
        self.row_indices = (
            np.arange(self.manifest.component_ids.shape[0], dtype=np.int64)
            if row_indices is None
            else np.asarray(row_indices, dtype=np.int64)
        )
        if self.row_indices.ndim != 1 or not self.row_indices.size:
            raise ValueError("Synthetic dataset needs at least one manifest row.")
        if bool((self.row_indices < 0).any()) or bool(
            (self.row_indices >= self.manifest.component_ids.shape[0]).any()
        ):
            raise ValueError("Synthetic dataset indices are outside the manifest.")

        # Batch renderer state
        self.space = SpectrumSpace(
            torch.as_tensor(artifact.axis, dtype=torch.float64),
            normalization=artifact.normalization,
        )
        sparse_indices = (
            torch.as_tensor(
                np.stack((artifact.basis_rows, artifact.basis_columns)),
                dtype=torch.long,
            )
            if artifact.basis_rows.size
            else torch.empty((2, 0), dtype=torch.long)
        )
        self._basis = torch.sparse_coo_tensor(
            sparse_indices,
            torch.as_tensor(artifact.basis_values, dtype=torch.float32),
            size=(artifact.prototype_count, artifact.feature_count),
        ).coalesce()
        self._transposed_basis = self._basis.transpose(0, 1).coalesce()
        self._prototype_targets = torch.as_tensor(
            artifact.prototype_target_indices,
            dtype=torch.long,
        )
        self.epoch = 0

    def __len__(self) -> int:
        """Return selected manifest-row count."""
        return int(self.row_indices.size)

    def __getitem__(self, index: int) -> int:
        """Return one lightweight manifest row identifier."""
        if index < 0 or index >= len(self):
            raise IndexError(index)
        return int(self.row_indices[index])

    def set_epoch(self, epoch: int) -> None:
        """Select per-epoch weights without rebuilding static population state."""
        if epoch < 0:
            raise ValueError("epoch must be nonnegative.")
        if not self.fixed_epoch:
            self.epoch = int(epoch)

    def collate_fn(self, rows: Sequence[int]) -> SpectrumBatch:
        """Vectorially render one batch and construct complete molecular targets."""
        row_ids = np.asarray(rows, dtype=np.int64)
        component_ids = self.manifest.component_ids[row_ids]  # (B, K)
        blank_centers = self.manifest.blank_centers[row_ids]  # (B,)
        spectra = self._render_batch(component_ids, blank_centers, row_ids)  # (B, M)
        return SpectrumBatch(
            sample_ids=torch.as_tensor(row_ids, dtype=torch.long),
            spectra=spectra.to(dtype=self.dtype),
            space=self.space,
            targets=self._build_targets(component_ids),
            metadata={
                "synthetic_precompute": {
                    "artifact_key": self.artifact.artifact_key,
                    "fingerprint": self.artifact.fingerprint,
                    "population": self.population,
                    "epoch": self.epoch,
                }
            },
        )

    def _render_batch(
        self,
        component_ids: np.ndarray,
        blank_centers: np.ndarray,
        row_ids: np.ndarray,
    ) -> torch.Tensor:
        """Render sparse basis mixtures and generic blank-bin peaks."""
        ids = torch.as_tensor(component_ids, dtype=torch.long)  # (B, K)
        valid = ids >= 0  # (B, K)
        batch_size = ids.shape[0]
        safe_ids = ids.clamp_min(0)  # (B, K)
        weights = _component_weights(
            row_ids=row_ids,
            component_mask=valid.numpy(),
            seed=self.seed,
            epoch=self.epoch,
        )  # (B, K)
        composition = torch.zeros(
            (batch_size, self.artifact.prototype_count),
            dtype=torch.float32,
        )  # (B, P)
        if self.artifact.prototype_count:
            composition.scatter_add_(1, safe_ids, weights)
            spectra = torch.sparse.mm(
                self._transposed_basis,
                composition.transpose(0, 1),
            ).transpose(0, 1)  # (B, M)
        else:
            spectra = torch.zeros(
                (batch_size, self.artifact.feature_count),
                dtype=torch.float32,
            )  # (B, M)
        _add_blank_profiles(
            spectra,
            torch.as_tensor(blank_centers, dtype=torch.long),
            radius=self.artifact.blank_peak_radius,
        )
        denominator = {
            "tic": spectra.sum(dim=1, keepdim=True),
            "max": spectra.amax(dim=1, keepdim=True),
            "l2": torch.linalg.vector_norm(spectra, dim=1, keepdim=True),
            "none": torch.ones((batch_size, 1), dtype=spectra.dtype),
        }[self.artifact.normalization]  # (B, 1)
        spectra = spectra / denominator.clamp_min(
            torch.finfo(spectra.dtype).tiny
        )  # (B, M)
        if not bool(torch.isfinite(spectra).all()) or bool((spectra < 0).any()):
            raise ValueError("Batch synthetic renderer produced invalid spectra.")
        return spectra

    def _build_targets(self, component_ids: np.ndarray) -> TargetBatch:
        """Create complete positives and known negatives for molecular targets."""
        batch_size = int(component_ids.shape[0])
        values = {
            name: torch.zeros((batch_size, schema.class_count), dtype=torch.float32)
            for name, schema in self.schemas.items()
        }
        masks = {
            name: torch.zeros((batch_size, schema.class_count), dtype=torch.bool)
            for name, schema in self.schemas.items()
        }
        ids = torch.as_tensor(component_ids, dtype=torch.long)  # (B, K)
        valid = ids >= 0  # (B, K)
        if self.artifact.prototype_count:
            labels = self._prototype_targets[ids.clamp_min(0)]  # (B, K)
            values["molecule"].scatter_add_(
                1,
                labels,
                valid.to(dtype=torch.float32),
            )
            values["molecule"].clamp_(0.0, 1.0)
        masks["molecule"].fill_(True)
        masks[simulated_negative_mask_key("molecule")] = (
            values["molecule"] < 0.5
        )  # (B, C_molecule)
        return TargetBatch(values=values, masks=masks, schemas=self.schemas)


def _component_weights(
    *,
    row_ids: np.ndarray,
    component_mask: np.ndarray,
    seed: int,
    epoch: int,
) -> torch.Tensor:
    """Generate stateless vectorized Dirichlet(1) component weights."""
    positions = np.arange(component_mask.shape[1], dtype=np.uint64)[None, :]
    seed_state = np.uint64(
        (int(seed) * 0x9E3779B97F4A7C15) & ((1 << 64) - 1)
    )
    epoch_state = np.uint64(
        ((int(epoch) + 1) * 0xBF58476D1CE4E5B9) & ((1 << 64) - 1)
    )
    states = (
        np.asarray(row_ids, dtype=np.uint64)[:, None]
        ^ seed_state
        ^ epoch_state
        ^ (positions * np.uint64(0x94D049BB133111EB))
    )
    uniform = _splitmix64_uniform(states)  # (B, K)
    exponential = -np.log(np.clip(uniform, np.finfo(np.float64).tiny, 1.0))
    exponential *= component_mask
    denominator = exponential.sum(axis=1, keepdims=True)
    weights = np.divide(
        exponential,
        denominator,
        out=np.zeros_like(exponential),
        where=denominator > 0,
    ).astype(np.float32)
    return torch.as_tensor(weights)  # (B, K)


def _splitmix64_uniform(values: np.ndarray) -> np.ndarray:
    """Map uint64 states to deterministic open-unit-interval values."""
    state = np.asarray(values, dtype=np.uint64) + np.uint64(0x9E3779B97F4A7C15)
    state = (state ^ (state >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    state = (state ^ (state >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    state = state ^ (state >> np.uint64(31))
    return ((state >> np.uint64(11)).astype(np.float64) + 0.5) / float(1 << 53)


def _add_blank_profiles(
    spectra: torch.Tensor,
    centers: torch.Tensor,
    *,
    radius: int,
) -> None:
    """Add normalized triangular profiles at blank axis anchors."""
    active = centers >= 0  # (B,)
    if not bool(active.any()):
        return
    offsets = torch.arange(-radius, radius + 1, dtype=torch.long)  # (W,)
    positions = centers[:, None] + offsets[None, :]  # (B, W)
    valid = (
        active[:, None]
        & (positions >= 0)
        & (positions < spectra.shape[1])
    )  # (B, W)
    profile = 1.0 - offsets.abs().to(dtype=spectra.dtype) / float(radius + 1)  # (W,)
    values = profile.expand_as(positions).clone() * valid.to(dtype=spectra.dtype)  # (B, W)
    values = values / values.sum(dim=1, keepdim=True).clamp_min(
        torch.finfo(spectra.dtype).tiny
    )  # (B, W)
    spectra.scatter_add_(
        1,
        positions.clamp(0, spectra.shape[1] - 1),
        values,
    )
