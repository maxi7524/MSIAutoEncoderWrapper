"""Shared projection primitives for inverse binning on a reconstruction axis."""

from __future__ import annotations

import torch

from ...data import InverseSpectrumBatch, SpectrumBatch, SpectrumSpace


def nearest_axis_indices(reference_axis: torch.Tensor, query_axis: torch.Tensor) -> torch.Tensor:
    """Return the nearest reference-axis index for every query coordinate."""
    right = torch.searchsorted(reference_axis, query_axis).clamp(0, reference_axis.numel() - 1)
    left = (right - 1).clamp(0, reference_axis.numel() - 1)
    choose_right = (reference_axis[right] - query_axis).abs() < (query_axis - reference_axis[left]).abs()
    return torch.where(choose_right, right, left)


def pack_point_projection(
    batch: SpectrumBatch,
    keep_mask: torch.Tensor,
    reconstruction_axis: torch.Tensor,
) -> InverseSpectrumBatch:
    """Project selected source points to their nearest reconstruction coordinates."""
    source_axis = batch.space.mass_axis.to(device=batch.device, dtype=reconstruction_axis.dtype)
    target_axis = reconstruction_axis.to(device=batch.device, dtype=source_axis.dtype)
    target_indices = nearest_axis_indices(target_axis, source_axis)  # (F,)
    batch_size, target_depth = batch.batch_size, target_axis.numel()

    projected = torch.zeros((batch_size, target_depth), device=batch.device, dtype=batch.spectra.dtype)
    projected_mask = torch.zeros((batch_size, target_depth), device=batch.device, dtype=torch.long)
    expanded_indices = target_indices.unsqueeze(0).expand(batch_size, -1)  # (B, F)
    projected.scatter_add_(1, expanded_indices, torch.where(keep_mask, batch.spectra, torch.zeros_like(batch.spectra)))
    projected_mask.scatter_add_(1, expanded_indices, keep_mask.long())
    return _pack(batch, target_axis, projected, projected_mask > 0)


def pack_region_projection(
    batch: SpectrumBatch,
    keep_mask: torch.Tensor,
    reconstruction_axis: torch.Tensor,
) -> InverseSpectrumBatch:
    """Assign every reconstruction coordinate to its nearest selected source bin."""
    source_axis = batch.space.mass_axis.to(device=batch.device, dtype=reconstruction_axis.dtype)
    target_axis = reconstruction_axis.to(device=batch.device, dtype=source_axis.dtype)
    source_indices = nearest_axis_indices(source_axis, target_axis)  # (G,)
    projected = batch.spectra[:, source_indices]  # (B, G)
    projected_mask = keep_mask[:, source_indices]  # (B, G)
    projected_mask &= (target_axis >= source_axis[0]).unsqueeze(0)
    projected_mask &= (target_axis <= source_axis[-1]).unsqueeze(0)
    return _pack(batch, target_axis, projected, projected_mask)


def _pack(
    batch: SpectrumBatch,
    axis: torch.Tensor,
    values: torch.Tensor,
    mask: torch.Tensor,
) -> InverseSpectrumBatch:
    """Pack a dense reconstruction mask in row-major order."""
    counts = mask.sum(dim=1)  # (B,)
    offsets = torch.zeros(batch.batch_size + 1, device=batch.device, dtype=torch.long)
    offsets[1:] = counts.cumsum(dim=0)
    rows, columns = torch.nonzero(mask, as_tuple=True)
    reconstruction_space = SpectrumSpace(
        mass_axis=axis,
        representation="reconstruction",
        normalization=batch.space.normalization,
        axis_unit=batch.space.axis_unit,
    )
    return InverseSpectrumBatch(
        sample_ids=batch.sample_ids,
        mass_values=axis[columns],
        intensities=values[rows, columns],
        offsets=offsets,
        source_space=batch.space,
        reconstruction_space=reconstruction_space,
        normalization_trace=batch.normalization_trace,
    )
