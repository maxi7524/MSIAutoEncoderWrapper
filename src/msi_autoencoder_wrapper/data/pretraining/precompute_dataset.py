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
            torch.as_tensor(artifact.axis, dtype=self.dtype),
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
        blank_centers = self.manifest.blank_centers[row_ids]  # (B, K)
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
                    "requested_target_indices": torch.as_tensor(
                        self.manifest.requested_target_indices[row_ids],
                        dtype=torch.long,
                    ),
                    "component_kinds": torch.as_tensor(
                        self.manifest.component_kinds[row_ids],
                        dtype=torch.uint8,
                    ),
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
        annotated = ids >= 0  # (B, K)
        blank = blank_centers >= 0  # (B, K)
        batch_size = ids.shape[0]
        safe_ids = ids.clamp_min(0)  # (B, K)
        weights = _component_weights(
            row_ids=row_ids,
            annotated_mask=annotated.numpy(),
            blank_mask=blank,
            annotated_concentrations=self.manifest.annotated_concentrations[
                row_ids
            ],
            blank_concentrations=self.manifest.blank_concentrations[row_ids],
            seed=self.seed,
            epoch=self.epoch,
        )  # (B, K)
        composition = torch.zeros(
            (batch_size, self.artifact.prototype_count),
            dtype=torch.float32,
        )  # (B, P)
        if self.artifact.prototype_count:
            composition.scatter_add_(
                1,
                safe_ids,
                weights * annotated.to(dtype=weights.dtype),
            )
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
            weights,
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
    annotated_mask: np.ndarray,
    blank_mask: np.ndarray,
    annotated_concentrations: np.ndarray,
    blank_concentrations: np.ndarray,
    seed: int,
    epoch: int,
) -> torch.Tensor:
    """Generate row-stable Dirichlet weights with annotated-priority priors.

    REMARK: One independent NumPy generator per manifest row makes the result
    invariant to DataLoader batch composition and ordering. The loop is over
    the batch dimension only; spectrum rendering remains sparse and batched.
    """
    row_ids = np.asarray(row_ids, dtype=np.int64)
    annotated_mask = np.asarray(annotated_mask, dtype=bool)
    blank_mask = np.asarray(blank_mask, dtype=bool)
    if annotated_mask.shape != blank_mask.shape:
        raise ValueError("Annotated and blank masks must have the same shape.")
    batch_size, slot_count = annotated_mask.shape
    if row_ids.shape != (batch_size,):
        raise ValueError("row_ids must have shape (B,).")
    annotated_concentrations = np.asarray(
        annotated_concentrations,
        dtype=np.float64,
    )
    blank_concentrations = np.asarray(blank_concentrations, dtype=np.float64)
    if annotated_concentrations.shape != (batch_size,) or (
        blank_concentrations.shape != (batch_size,)
    ):
        raise ValueError("Dirichlet concentrations must have shape (B,).")

    weights = np.zeros((batch_size, slot_count), dtype=np.float32)
    for batch_index, row_id in enumerate(row_ids):
        valid = annotated_mask[batch_index] | blank_mask[batch_index]  # (K,)
        alpha = np.where(
            annotated_mask[batch_index],
            annotated_concentrations[batch_index],
            blank_concentrations[batch_index],
        )[valid]  # (K_valid,)
        if not alpha.size or not np.isfinite(alpha).all() or bool((alpha <= 0).any()):
            raise ValueError("Every rendered row needs positive finite concentrations.")
        generator = np.random.default_rng(
            np.random.SeedSequence([int(seed), int(epoch), int(row_id)])
        )
        weights[batch_index, valid] = generator.dirichlet(alpha).astype(
            np.float32,
            copy=False,
        )
    return torch.as_tensor(weights)  # (B, K)


def _add_blank_profiles(
    spectra: torch.Tensor,
    centers: torch.Tensor,
    weights: torch.Tensor,
    *,
    radius: int,
) -> None:
    """Add weighted normalized triangular profiles at blank axis anchors."""
    if centers.ndim != 2 or weights.shape != centers.shape:
        raise ValueError("Blank centers and weights must have shape (B, K).")
    active = centers >= 0  # (B, K)
    if not bool(active.any()):
        return
    offsets = torch.arange(-radius, radius + 1, dtype=torch.long)  # (W,)
    positions = centers[:, :, None] + offsets[None, None, :]  # (B, K, W)
    valid = (
        active[:, :, None]
        & (positions >= 0)
        & (positions < spectra.shape[1])
    )  # (B, K, W)
    profile = 1.0 - offsets.abs().to(dtype=spectra.dtype) / float(radius + 1)  # (W,)
    values = profile.expand_as(positions).clone() * valid.to(dtype=spectra.dtype)  # (B, K, W)
    values = values / values.sum(dim=2, keepdim=True).clamp_min(
        torch.finfo(spectra.dtype).tiny
    )  # (B, K, W)
    values = values * weights[:, :, None]  # (B, K, W)
    batch_size = spectra.shape[0]
    spectra.scatter_add_(
        1,
        positions.clamp(0, spectra.shape[1] - 1).reshape(batch_size, -1),
        values.reshape(batch_size, -1),
    )
