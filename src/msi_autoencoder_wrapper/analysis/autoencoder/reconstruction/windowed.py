"""Local (m/z-windowed) decomposition of the training Masserstein objective.

The training objective is the exact one-dimensional Wasserstein-1 distance between
TIC-normalized spectra,

.. math::

    W_1(x, \\hat x) = \\int |F_{\\hat x}(m) - F_x(m)|\\,dm
                    = \\sum_j |C_j|\\,(m_{j+1} - m_j),

with :math:`C_j` the cumulative mass difference up to bin :math:`j`. This module
reports, for every half-open m/z window :math:`W = [a, b)`, three complementary
quantities:

``contribution``
    :math:`\\int_W |F_{\\hat x} - F_x|\\,dm`. The windows partition the axis, so the
    contributions sum exactly to the global objective. A contribution also contains
    mass that is merely *transported through* the window because of an imbalance at
    lower m/z.
``within``
    :math:`W_1` between the two spectra restricted to the window and renormalized to
    unit mass there. It measures the local shape error only and is undefined when
    either restricted mass is below a threshold.
``input_mass`` / ``output_mass``
    The TIC fraction of each spectrum inside the window. Their signed difference is
    the local mass imbalance; the imbalance entering a window equals the sum of the
    imbalances of all preceding windows.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import torch

from ....metrics import SpectrumMasserstein
from ....normalization import ScalarNormalization
from ....utils.exceptions import raise_validation_error
from ....utils.logger import get_custom_logger

logger = get_custom_logger(__name__)


@dataclass(frozen=True)
class WindowedMasserstein:
    """Per-spectrum local Masserstein decomposition.

    :param edges: Window edges in m/z, shape ``(W + 1,)``.
    :type edges: numpy.ndarray
    :param total: Global training Masserstein cost per spectrum, shape ``(N,)``.
    :type total: numpy.ndarray
    :param contribution: Exact additive window contributions, shape ``(N, W)``.
    :type contribution: numpy.ndarray
    :param within: Renormalized within-window cost, shape ``(N, W)``; ``nan`` where
        either restricted mass is below the configured threshold.
    :type within: numpy.ndarray
    :param input_mass: TIC fraction of the input inside each window, shape ``(N, W)``.
    :type input_mass: numpy.ndarray
    :param output_mass: TIC fraction of the reconstruction inside each window,
        shape ``(N, W)``.
    :type output_mass: numpy.ndarray
    """

    edges: np.ndarray
    total: np.ndarray
    contribution: np.ndarray
    within: np.ndarray
    input_mass: np.ndarray
    output_mass: np.ndarray

    @property
    def mass_imbalance(self) -> np.ndarray:
        """Signed local imbalance ``output_mass - input_mass``, shape ``(N, W)``."""
        return self.output_mass - self.input_mass


def window_edges(mass_axis: np.ndarray, width: float) -> np.ndarray:
    """Return window edges aligned to integer multiples of ``width``.

    Aligning the edges to multiples of the width, instead of to the first bin, makes
    windows of different spectral axes directly comparable: ``200-300`` denotes the
    same physical interval on a 200-900 and on a 100-3000 axis.

    :param mass_axis: Strictly increasing bin centres, shape ``(M,)``.
    :type mass_axis: numpy.ndarray
    :param width: Window width in m/z units; must be positive.
    :type width: float
    :return: Edges covering every bin centre, shape ``(W + 1,)``.
    :rtype: numpy.ndarray
    :raises ValidationError: If the axis is not increasing or the width is invalid.
    """
    axis = np.asarray(mass_axis, dtype=np.float64)
    if axis.ndim != 1 or axis.size < 1 or width <= 0 or not np.all(np.diff(axis) > 0):
        raise_validation_error("WindowedMasserstein", "Need an increasing axis and a positive width.")
    start = np.floor(axis[0] / width) * width
    stop = (np.floor(axis[-1] / width) + 1.0) * width
    count = int(round((stop - start) / width))
    return start + width * np.arange(count + 1, dtype=np.float64)


def window_assignment(mass_axis: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Assign every bin centre to its half-open window ``[a, b)``.

    :param mass_axis: Bin centres, shape ``(M,)``.
    :type mass_axis: numpy.ndarray
    :param edges: Increasing window edges, shape ``(W + 1,)``.
    :type edges: numpy.ndarray
    :return: Window index per bin, shape ``(M,)``.
    :rtype: numpy.ndarray
    :raises ValidationError: If a bin centre lies outside the edges.
    """
    axis = np.asarray(mass_axis, dtype=np.float64)
    assignment = np.searchsorted(edges, axis, side="right") - 1  # (M,)
    if np.any(assignment < 0) or np.any(assignment >= len(edges) - 1):
        raise_validation_error("WindowedMasserstein", "Window edges must cover every bin centre.")
    return assignment.astype(np.int64)


def _segment_window_lengths(mass_axis: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Length of every inter-centre segment ``[m_j, m_{j+1}]`` inside every window.

    :param mass_axis: Bin centres, shape ``(M,)``.
    :type mass_axis: numpy.ndarray
    :param edges: Window edges, shape ``(W + 1,)``.
    :type edges: numpy.ndarray
    :return: Overlap lengths, shape ``(M - 1, W)``; each row sums to the segment length.
    :rtype: numpy.ndarray
    """
    axis = np.asarray(mass_axis, dtype=np.float64)
    left = axis[:-1, None]  # (M - 1, 1)
    right = axis[1:, None]  # (M - 1, 1)
    lower = edges[None, :-1]  # (1, W)
    upper = edges[None, 1:]  # (1, W)
    return np.clip(np.minimum(right, upper) - np.maximum(left, lower), 0.0, None)  # (M - 1, W)


def windowed_masserstein(
    inputs: np.ndarray,
    outputs: np.ndarray,
    mass_axis: np.ndarray,
    edges: np.ndarray,
    *,
    minimum_window_mass: float = 1e-3,
    batch_size: int = 1024,
    device: str | torch.device = "cpu",
    criterion_options: Optional[Mapping[str, Any]] = None,
    total_atol: float = 1e-4,
    total_rtol: float = 1e-4,
) -> WindowedMasserstein:
    """Decompose the training Masserstein objective into m/z windows.

    The global cost is evaluated by the training metric itself
    (:class:`~msi_autoencoder_wrapper.metrics.SpectrumMasserstein`), and the window
    contributions are recomputed from the same TIC normalization. Their sum is checked
    against the global cost, so a divergence between the decomposition and the training
    objective raises instead of silently producing inconsistent tables.

    :param inputs: Model inputs, shape ``(N, M)``; nonnegative.
    :type inputs: numpy.ndarray
    :param outputs: Reconstructions, shape ``(N, M)``; nonnegative.
    :type outputs: numpy.ndarray
    :param mass_axis: Bin centres, shape ``(M,)``.
    :type mass_axis: numpy.ndarray
    :param edges: Window edges from :func:`window_edges`, shape ``(W + 1,)``.
    :type edges: numpy.ndarray
    :param minimum_window_mass: Minimum TIC fraction of both restricted spectra for a
        defined within-window cost.
    :type minimum_window_mass: float
    :param batch_size: Spectra evaluated together.
    :type batch_size: int
    :param device: Torch device.
    :type device: str | torch.device
    :param criterion_options: Masserstein criterion parameters used in training.
    :type criterion_options: Mapping[str, Any] | None
    :param total_atol: Absolute tolerance of the additivity check.
    :type total_atol: float
    :param total_rtol: Relative tolerance of the additivity check. ``float32`` cumulative
        sums over several thousand bins accumulate rounding of order ``1e-6`` per unit
        mass, so ``1e-4`` leaves two orders of margin.
    :type total_rtol: float
    :return: Decomposition arrays.
    :rtype: WindowedMasserstein
    :raises ValidationError: If shapes disagree or the contributions do not add up to
        the training objective.
    """
    original = np.asarray(inputs, dtype=np.float32)
    reconstructed = np.asarray(outputs, dtype=np.float32)
    axis = np.asarray(mass_axis, dtype=np.float32)
    if original.ndim != 2 or original.shape != reconstructed.shape or axis.shape != (original.shape[1],):
        raise_validation_error("WindowedMasserstein", "Inputs, outputs and axis are misaligned.")
    if batch_size < 1 or minimum_window_mass < 0:
        raise_validation_error("WindowedMasserstein", "batch_size must be positive and the mass threshold nonnegative.")

    # Static geometry
    ## Window membership of bins and of inter-centre transport segments
    edges = np.asarray(edges, dtype=np.float64)
    assignment = window_assignment(axis, edges)  # (M,)
    window_count = len(edges) - 1
    lengths = torch.as_tensor(_segment_window_lengths(axis, edges), dtype=torch.float32, device=device)  # (M - 1, W)
    membership = torch.zeros(axis.size, window_count, dtype=torch.float32, device=device)  # (M, W)
    membership[torch.arange(axis.size), torch.as_tensor(assignment)] = 1.0
    window_bins = [np.flatnonzero(assignment == window) for window in range(window_count)]

    ## Training metric and normalization, shared with the objective
    options = dict(criterion_options or {})
    options["reduction"] = "none"
    metric = SpectrumMasserstein(**options).to(device)
    normalization = ScalarNormalization(kind="tic", epsilon=metric.epsilon)
    axis_tensor = torch.as_tensor(axis, device=device)
    ### REMARK: SpectrumMasserstein freezes its transport geometry on first use, so
    ### every window needs its own instance bound to that window's sub-axis.
    window_metrics = [SpectrumMasserstein(**options).to(device) for _ in window_bins]

    # Batched decomposition
    totals, contributions, within, input_mass, output_mass = [], [], [], [], []
    with torch.no_grad():
        for start in range(0, len(original), batch_size):
            stop = min(start + batch_size, len(original))
            target = torch.as_tensor(original[start:stop], device=device)  # (B, M)
            prediction = torch.as_tensor(reconstructed[start:stop], device=device)  # (B, M)

            ### Global objective and exact additive contributions
            total = metric(prediction, target, mass_axis=axis_tensor)  # (B,)
            target_normalized, _ = normalization.transform(target)  # (B, M)
            prediction_normalized, _ = normalization.transform(prediction)  # (B, M)
            cumulative = torch.cumsum(prediction_normalized - target_normalized, dim=1)  # (B, M)
            contribution = cumulative[:, :-1].abs() @ lengths  # (B, W)

            ### Local masses
            mass_in = target_normalized @ membership  # (B, W)
            mass_out = prediction_normalized @ membership  # (B, W)

            ### Renormalized within-window cost
            local = torch.full((stop - start, window_count), float("nan"), device=device)  # (B, W)
            for window, bins in enumerate(window_bins):
                if bins.size < 2:
                    local[:, window] = 0.0
                    continue
                index = torch.as_tensor(bins, device=device)
                local[:, window] = window_metrics[window](
                    prediction_normalized[:, index], target_normalized[:, index], mass_axis=axis_tensor[index],
                )  # (B,)
            defined = (mass_in >= minimum_window_mass) & (mass_out >= minimum_window_mass)  # (B, W)
            local = torch.where(defined, local, torch.full_like(local, float("nan")))

            totals.append(total.cpu().numpy())
            contributions.append(contribution.cpu().numpy())
            within.append(local.cpu().numpy())
            input_mass.append(mass_in.cpu().numpy())
            output_mass.append(mass_out.cpu().numpy())

    result = WindowedMasserstein(
        edges=edges,
        total=np.concatenate(totals).astype(np.float32),
        contribution=np.concatenate(contributions).astype(np.float32),
        within=np.concatenate(within).astype(np.float32),
        input_mass=np.concatenate(input_mass).astype(np.float32),
        output_mass=np.concatenate(output_mass).astype(np.float32),
    )

    # Additivity contract
    ## REMARK: The decomposition is only meaningful while it reproduces the training
    ## objective; a silent mismatch would make every local table inconsistent.
    summed = result.contribution.sum(axis=1)  # (N,)
    if not np.allclose(summed, result.total, atol=total_atol, rtol=total_rtol):
        worst = float(np.max(np.abs(summed - result.total)))
        raise_validation_error(
            "WindowedMasserstein",
            f"Window contributions do not sum to the training Masserstein cost (max deviation {worst:.3e}).",
        )
    logger.debug("Windowed Masserstein: %s spectra, %s windows.", len(original), window_count)
    return result
