"""Trained-model reconstruction vs. raw X, swept across binning steps.

Answers the model-in-the-loop counterpart of :mod:`.binner_forward_analysis`'s
question: instead of ``B(X)`` vs ``X`` (pure binning error, no model involved), this
compares ``model(B(X))`` vs ``X`` — how well a *trained* autoencoder's reconstruction
recovers the original raw spectrum, at each binning step it was trained on. Reuses
the exact same matching/metric machinery as :mod:`.binner_forward_analysis`
(``match_spectral_points`` + ``metrics.spectral_points.spectral_point_metrics``) by
treating the model's dense reconstruction (one value per forward-binner grid point) as
a candidate point set, so records from both modules share one schema and can be
overlaid on the same :func:`.binner_forward_analysis.plot_metric_by_delta_m` /
:func:`.binner_forward_analysis.plot_forward_tradeoff` plots — directly showing how
much of the total (binning + model) error the model adds on top of binning alone.

Models are loaded from a :mod:`..experiments.campaign_reader` grid campaign via
:class:`~msi_autoencoder_wrapper.models.model_loader.ModelLoader` (real trained
weights, not the parameter-counting, weights-free reconstruction used in
:mod:`..reconstruction.architecture_overview_analysis`).

REMARK: every model in this campaign was trained on ``PixelDataset``-normalized
spectra (``data.dataset.parameters.normalization``, typically ``"tic"``) — the
dataset divides each *binned* spectrum by its own scale (L1/Linf/L2 depending on the
kind) before the model ever sees it, and the reconstruction target is that same
normalized representation. Feeding the model raw, unnormalized binned intensities is
out-of-distribution input and produces a degenerate reconstruction that looks like
"the model predicts almost nothing" when compared against a raw-scale reference (the
two are off by orders of magnitude, not actually comparable). Every function here
therefore reads each task's own saved ``normalization``/``normalization_epsilon`` and
applies the *identical* formula (:func:`_dataset_normalize`, matching
``PixelDataset._normalize`` exactly) to both the model's input and the raw reference
spectrum before any comparison or plot — there is deliberately no "raw" or "max"
comparison option: only the scale the model was actually trained on is meaningful.
"""

from __future__ import annotations

import time
from collections import defaultdict
from typing import Any, Mapping, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm.auto import tqdm

from ....metrics import match_spectral_points, spectral_point_metrics
from ....models.model_loader import ModelLoader
from ....readers.base_reader import MSIBaseReader
from ....readers.spatial import SpatialImage
from ....utils.exceptions import raise_validation_error
from ....utils.logger import get_custom_logger
from ....visualization import VisualizationTheme, resolve_theme
from ....visualization.spatial import plot_spatial_image
from ....visualization.spectra import plot_sparse_spectrum_match
from ..experiments.campaign_reader import CampaignTask
from ..reconstruction.architecture_overview_analysis import architecture_grid_parameters
from .binner_forward_analysis import crop_spectrum
from .inverse_binner_analysis import MAXIMIZE_METRICS
from .precompute import BinningPrecompute

logger = get_custom_logger(__name__)


def group_tasks_by_binning(
    tasks: Sequence[CampaignTask],
    architecture_name: str,
) -> dict[float, list[CampaignTask]]:
    """Group one architecture's completed tasks (every repetition) by binning step.

    :param tasks: Campaign tasks with loaded ``grid_parameters``/``model_config`` (see
        :func:`~..experiments.campaign_reader.read_campaign`).
    :type tasks: Sequence[CampaignTask]
    :param architecture_name: This campaign's ``grid_parameters.architectures.name``
        to select (see :func:`~..reconstruction.architecture_overview_analysis.
        architecture_grid_parameters`).
    :type architecture_name: str
    :return: ``{binning_step: [task, ...]}``, sorted by ``binning_step``, tasks in
        their original order (every repetition kept, not deduplicated).
    :rtype: dict[float, list[CampaignTask]]
    """
    grouped: dict[float, list[CampaignTask]] = defaultdict(list)
    for task in tasks:
        if task.status != "completed" or task.model_config is None:
            continue
        grid = architecture_grid_parameters(task)
        if grid["name"] != architecture_name:
            continue
        grouped[grid["binning_step"]].append(task)
    return dict(sorted(grouped.items()))


def split_spectrum_ids(task: CampaignTask, split_name: str) -> np.ndarray:
    """Return one task's saved dataset-split pixel indices.

    :param task: Campaign task with a loaded ``model_config``.
    :type task: CampaignTask
    :param split_name: ``train``, ``validation``, or ``test`` (the keys saved under
        ``data.dataset.parameters.split.assignments`` at training time).
    :type split_name: str
    :return: Sorted, deduplicated pixel indices.
    :rtype: numpy.ndarray
    :raises ValidationError: If ``model_config`` is missing or has no such split.
    """
    if task.model_config is None:
        raise_validation_error(
            "ModelReconstructionAnalysis",
            f"Task '{task.task_id}' has no loaded model configuration.",
        )
    assignments = task.model_config["data"]["dataset"]["parameters"]["split"]["assignments"]
    if split_name not in assignments:
        raise_validation_error(
            "ModelReconstructionAnalysis",
            f"Unknown split '{split_name}'; available: {sorted(assignments)}.",
        )
    return np.unique(np.asarray(assignments[split_name], dtype=np.int64))


def sample_split_ids(
    task: CampaignTask,
    split_name: str,
    sample_size: int,
    seed: int,
) -> np.ndarray:
    """Return a reproducible random subsample of one saved dataset split.

    Splits (especially ``train``) can hold tens of thousands of pixels — passing the
    full split straight to :class:`~.precompute.BinningPrecompute` would attempt to
    read/bin/reconstruct all of them. This draws a fixed-size, seeded subsample
    instead, the same way the notebook-level ``SAMPLE_SIZE``/``RANDOM_SEED`` config
    already controls :class:`~.precompute.BinningPrecompute`'s own random sample.

    :param sample_size: Requested pixel count; capped to the split's size.
    :type sample_size: int
    :param seed: Sample-draw seed.
    :type seed: int
    :return: Sorted pixel indices, ready to pass as ``BinningPrecompute(...,
        spectrum_ids=...)``.
    :rtype: numpy.ndarray
    """
    ids = split_spectrum_ids(task, split_name)
    if sample_size >= ids.size:
        return ids
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(ids, size=sample_size, replace=False))


def load_reconstruction_model(
    task: CampaignTask,
    device: str | torch.device = "cpu",
) -> tuple[torch.nn.Module, tuple[float, float], tuple[str, float]]:
    """Load one task's trained weights, ready for evaluation-mode inference.

    :param task: Completed campaign task with a ``result.model_path``.
    :type task: CampaignTask
    :param device: Torch execution device.
    :type device: str | torch.device
    :return: Model in ``eval()`` mode on ``device``; the exact ``(x_min, x_max)`` its
        forward binner used at training time (required to rebuild an identical binner
        via :meth:`~.precompute.BinningPrecompute.forward` — a mismatch would silently
        change the input dimensionality the model expects); and the exact
        ``(normalization_kind, normalization_epsilon)`` its dataset used (``"none"``,
        ``"tic"``, ``"max"``, or ``"l2"`` — required by :func:`_dataset_normalize`, see
        this module's docstring for why this is not optional).
    :rtype: tuple[torch.nn.Module, tuple[float, float], tuple[str, float]]
    :raises ValidationError: If the task has no model artifact.
    """
    if task.result is None or not task.result.get("model_path"):
        raise_validation_error(
            "ModelReconstructionAnalysis",
            f"Task '{task.task_id}' has no model artifact to load.",
        )
    model, config, _ = ModelLoader.load_artifact(task.result["model_path"])
    model.to(device).eval()
    binner_parameters = config["data"]["context"]["components"]["binner"]["parameters"]
    dataset_parameters = config["data"]["dataset"]["parameters"]
    normalization = (
        str(dataset_parameters.get("normalization") or "none"),
        float(dataset_parameters.get("normalization_epsilon", 1e-12)),
    )
    return (
        model,
        (float(binner_parameters["x_min"]), float(binner_parameters["x_max"])),
        normalization,
    )


def _dataset_normalize(values: np.ndarray, kind: str, epsilon: float) -> np.ndarray:
    """Normalize exactly like ``PixelDataset._normalize`` (identical formula/epsilon).

    Works on both a single spectrum (1D) and a batch (2D, one spectrum per row) via
    ``axis=-1`` reduction. Spectra whose scale falls at or below ``epsilon`` are
    zeroed, matching the training-time dataset's degenerate-spectrum handling exactly.

    :param values: Binned spectrum or batch of spectra.
    :type values: numpy.ndarray
    :param kind: ``"none"``, ``"tic"``, ``"max"``, or ``"l2"``.
    :type kind: str
    :param epsilon: Positive denominator safety threshold.
    :type epsilon: float
    :return: Normalized array, same shape as ``values``.
    :rtype: numpy.ndarray
    :raises ValidationError: If ``kind`` is not a recognized normalization.
    """
    if kind == "none":
        return np.asarray(values, dtype=np.float32)
    array = np.asarray(values, dtype=np.float32)
    absolute = np.abs(array)
    if kind == "tic":
        denominator = np.sum(absolute, axis=-1, keepdims=True)
    elif kind == "max":
        denominator = np.max(absolute, axis=-1, keepdims=True, initial=0.0)
    elif kind == "l2":
        denominator = np.linalg.norm(array, axis=-1, keepdims=True)
    else:
        raise_validation_error(
            "ModelReconstructionAnalysis",
            f"Unsupported dataset normalization kind '{kind}'.",
        )
    safe = np.where(denominator > epsilon, denominator, 1.0)
    normalized = array / safe
    return np.where(denominator > epsilon, normalized, 0.0).astype(np.float32, copy=False)


def _reconstruct(
    model: torch.nn.Module,
    values: np.ndarray,
    device: str | torch.device,
    batch_size: int,
    normalization_kind: str,
    normalization_epsilon: float,
) -> np.ndarray:
    """Run batched forward-pass reconstruction over a ``[N, D]`` binned-spectrum matrix.

    Normalizes ``values`` with :func:`_dataset_normalize` before the forward pass —
    see this module's docstring for why this is required, not optional. The returned
    reconstruction is therefore in that same normalized space, not raw scale.
    """
    normalized_values = _dataset_normalize(values, normalization_kind, normalization_epsilon)
    outputs: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(normalized_values), batch_size):
            batch = torch.as_tensor(
                normalized_values[start : start + batch_size], dtype=torch.float32, device=device
            )
            outputs.append(model(batch)["reconstruction"].detach().cpu().numpy())
    return np.concatenate(outputs).astype(np.float32, copy=False)


def model_reconstruction_records(
    precompute: BinningPrecompute,
    tasks_by_binning: Mapping[float, Sequence[CampaignTask]],
    tolerance: float = 0.01,
    tolerance_unit: str = "Da",
    device: str | torch.device = "cpu",
    batch_size: int = 64,
) -> list[dict[str, Any]]:
    """Long-form records of every metric, for every sampled spectrum, task, and delta_m.

    For each ``(binning_step, task)`` pair: rebuilds the forward binner at that task's
    exact training-time ``(x_min, x_max)`` (via :meth:`~.precompute.BinningPrecompute.
    forward`, cached across repetitions sharing a binning step), runs the task's
    trained model on every sampled spectrum's ``B(X)`` (normalized exactly like
    training — see this module's docstring), then matches the reconstruction against
    the identically-normalized raw spectrum with ``match_spectral_points(...,
    "one_to_one")`` — the same matching :func:`~.binner_forward_analysis.
    forward_sweep_records` uses, just with the model's reconstruction standing in for
    ``B(X)`` as the candidate.

    :param precompute: Shared raw-spectra cache. Its own ``forward``/``inverse``
        caches are reused/extended in place.
    :type precompute: BinningPrecompute
    :param tasks_by_binning: One or more tasks (typically every repetition) per
        binning step, e.g. :func:`group_tasks_by_binning`'s output.
    :type tasks_by_binning: Mapping[float, Sequence[CampaignTask]]
    :return: One row per (spectrum, task, delta_m, metric) — same fields as
        :func:`~.binner_forward_analysis.forward_sweep_records` plus ``task_id``/
        ``repetition``, so records from both functions can be concatenated and
        grouped identically. ``normalization`` is fixed to whatever each task's own
        dataset used at training time (not a caller-supplied sweep — see this
        module's docstring).
    :rtype: list[dict[str, Any]]
    """
    logger.info(
        "START model-reconstruction sweep: %s spectra x %s binning step(s).",
        precompute.spectrum_ids.size, len(tasks_by_binning),
    )
    started = time.perf_counter()
    records: list[dict[str, Any]] = []
    for binning_step, tasks in tasks_by_binning.items():
        for task in tasks:
            model, (x_min, x_max), (normalization_kind, normalization_epsilon) = load_reconstruction_model(task, device)
            binner, forward_cache = precompute.forward(binning_step, x_min, x_max)
            grid_mz = np.asarray(binner.GetXAxis(), dtype=np.float32)
            dimension = binner.GetXAxisDepth()
            stacked = np.stack(
                [forward_cache[int(spectrum_id)] for spectrum_id in precompute.spectrum_ids]
            )

            # Batched inference
            ## One forward pass per batch instead of per spectrum; input is normalized
            ## exactly like training (see _reconstruct).
            task_started = time.perf_counter()
            reconstructed = _reconstruct(model, stacked, device, batch_size, normalization_kind, normalization_epsilon)
            logger.info(
                "Reconstructed %s spectra with task '%s' (Δm/z=%s, normalization=%s) in %.2fs.",
                len(stacked), task.task_id, binning_step, normalization_kind, time.perf_counter() - task_started,
            )

            for index, spectrum_id in enumerate(
                tqdm(precompute.spectrum_ids, desc=f"Matching {task.task_id} Δm={binning_step:g}", unit="spectrum")
            ):
                raw_mz, raw_y = crop_spectrum(*precompute.raw(spectrum_id), binner.x_min, binner.x_max)
                normalized_raw = _dataset_normalize(raw_y, normalization_kind, normalization_epsilon)
                recon_y = reconstructed[index]
                match = match_spectral_points(raw_mz, normalized_raw, grid_mz, recon_y, tolerance, tolerance_unit, "one_to_one")
                values = spectral_point_metrics(raw_mz, normalized_raw, grid_mz, recon_y, match)
                for metric, value in values.items():
                    records.append({
                        "spectrum_id": int(spectrum_id), "task_id": task.task_id, "repetition": task.repetition,
                        "delta_m": float(binning_step), "normalization": normalization_kind, "metric": metric,
                        "value": float(value), "feature_dimension": int(dimension),
                        "x_min": float(binner.x_min), "x_max": float(binner.x_max),
                    })
    logger.info("DONE model-reconstruction sweep: %s records in %.2fs.", len(records), time.perf_counter() - started)
    return records


def plot_ranked_model_reconstructions(
    precompute: BinningPrecompute,
    records: Sequence[Mapping[str, Any]],
    task: CampaignTask,
    metric: str,
    n_best: int = 2,
    n_worst: int = 2,
    tolerance: float = 0.01,
    tolerance_unit: str = "Da",
    zoom_window_da: float = 30.0,
    device: str | torch.device = "cpu",
    theme: VisualizationTheme | str | None = None,
):
    """Mirror-plot one task's ``n_best``/``n_worst`` reconstructed spectra.

    Reuses ``visualization.spectra.plot_sparse_spectrum_match`` exactly as
    :func:`~.inverse_binner_analysis.plot_ranked_spectra` does: reference (raw X,
    normalized exactly like training — see this module's docstring) plotted positive,
    reconstruction mirrored negative, matched pairs connected, unmatched points
    marked, matched-intensity residual below. Each selected spectrum gets **two**
    rows: the full spectrum, and a second row zoomed to a ``zoom_window_da``-wide
    window **centered on the single highest-intensity reference peak** — deliberately
    one fixed-width window around the single densest point, not a window spanning
    several peaks (which can span an arbitrarily wide, mostly-empty range if those
    peaks are far apart and says nothing about local reconstruction quality).

    :param records: Output of :func:`model_reconstruction_records`, filtered to
        ``task`` internally (by ``task_id``/``metric``) to select which spectra rank
        best/worst; each task carries exactly one ``normalization`` in ``records``,
        so no explicit selection is needed here.
    :type records: Sequence[Mapping[str, Any]]
    :param task: Which task's spectra to rank and plot (a single repetition — pass
        the campaign's best-performing task for the binning step being inspected,
        e.g. from :func:`~..reconstruction.training_dynamics_analysis.
        final_performance_summary`'s ``best_task_id``).
    :type task: CampaignTask
    :param zoom_window_da: Total width (not half-width) of the zoomed window, in Da.
    :type zoom_window_da: float
    :return: Figure and axes, or ``(None, None)`` if no finite record matched.
    :rtype: tuple[matplotlib.figure.Figure, numpy.ndarray] | tuple[None, None]
    """
    resolved = resolve_theme(theme)
    grid = architecture_grid_parameters(task)
    selected = [
        record for record in records
        if record["task_id"] == task.task_id and record["metric"] == metric
        and np.isfinite(record["value"])
    ]
    ordered = sorted(selected, key=lambda record: record["value"], reverse=metric in MAXIMIZE_METRICS)
    chosen = ordered[:n_best] + (ordered[-n_worst:] if n_worst else [])
    if not chosen:
        logger.warning(
            "No finite records for task_id=%s, metric=%s — skipping.",
            task.task_id, metric,
        )
        return None, None
    logger.info(
        "Plotting %s best + %s worst reconstructed spectra for task_id=%s, metric=%s.",
        n_best, n_worst, task.task_id, metric,
    )

    model, (x_min, x_max), (normalization_kind, normalization_epsilon) = load_reconstruction_model(task, device)
    binner, forward_cache = precompute.forward(grid["binning_step"], x_min, x_max)
    grid_mz = np.asarray(binner.GetXAxis(), dtype=np.float32)

    figure, axes = plt.subplots(
        4 * len(chosen), 1, figsize=(resolved.figure_size[0], 3 * 4 * len(chosen)), dpi=resolved.figure_dpi, squeeze=False,
    )
    for row, record in enumerate(chosen):
        spectrum_id = record["spectrum_id"]
        raw_mz, raw_y = crop_spectrum(*precompute.raw(spectrum_id), binner.x_min, binner.x_max)
        normalized_raw = _dataset_normalize(raw_y, normalization_kind, normalization_epsilon)
        recon_y = _reconstruct(
            model, forward_cache[int(spectrum_id)][None, :], device, 1, normalization_kind, normalization_epsilon
        )[0]
        match = match_spectral_points(raw_mz, normalized_raw, grid_mz, recon_y, tolerance, tolerance_unit, "one_to_one")
        rank_label = "best" if row < n_best else "worst"
        candidate_label = f"{grid['name']}: reconstruction"

        # Full spectrum
        full_signal, full_residual = axes[4 * row, 0], axes[4 * row + 1, 0]
        plot_sparse_spectrum_match(raw_mz, normalized_raw, grid_mz, recon_y, match, axes=(full_signal, full_residual), candidate_label=candidate_label, theme=resolved)
        full_signal.set_title(
            f"{grid['name']} Δm/z={grid['binning_step']:g} | {metric} {rank_label} | spectrum {spectrum_id} | {record['value']:.4g}",
            loc=resolved.title_location,
        )

        # Zoom around the reference spectrum's single densest peak
        ## Same match/data as the full-spectrum row above; only the visible x-range
        ## differs, so the residual panel's matched/unmatched markers stay consistent.
        zoom_signal, zoom_residual = axes[4 * row + 2, 0], axes[4 * row + 3, 0]
        plot_sparse_spectrum_match(raw_mz, normalized_raw, grid_mz, recon_y, match, axes=(zoom_signal, zoom_residual), candidate_label=candidate_label, theme=resolved)
        window = _zoom_window(raw_mz, normalized_raw, zoom_window_da)
        if window is not None:
            zoom_signal.set_xlim(window)
            zoom_residual.set_xlim(window)
            zoom_signal.set_title(f"zoom: ±{zoom_window_da / 2:g} Da around the densest reference peak", loc=resolved.title_location)
    figure.tight_layout()
    return figure, axes


def _zoom_window(
    mz: np.ndarray,
    intensity: np.ndarray,
    window_da: float,
) -> Optional[tuple[float, float]]:
    """Return a ``window_da``-wide window centered on the single highest-intensity peak."""
    if mz.size == 0 or window_da <= 0:
        return None
    center = float(mz[int(np.argmax(intensity))])
    half_width = window_da / 2.0
    return center - half_width, center + half_width


def spatial_reconstruction_error(
    reader: MSIBaseReader,
    records: Sequence[Mapping[str, Any]],
    metric: str = "wasserstein",
) -> SpatialImage:
    """Map one per-spectrum metric onto the reader's native spatial grid.

    Intended for ``records`` covering exactly one task (one architecture, one binning
    step, one repetition) — spectra outside ``records`` (any pixel not swept, e.g. a
    different split) are left as ``NaN`` (transparent in :func:`~msi_autoencoder_wrapper.
    visualization.spatial.plot_spatial_image`), so calling this on a per-split
    ``records`` list directly produces a mask over only that split's pixels — no
    separate masking step is needed.

    :param reader: Reader the sampled ``spectrum_id`` values are indices into.
    :type reader: MSIBaseReader
    :param records: Output of :func:`model_reconstruction_records` (or any subset of
        it already filtered to one task).
    :type records: Sequence[Mapping[str, Any]]
    :param metric: Which per-spectrum metric to map.
    :type metric: str
    :return: Values arranged on the reader's native ``(z, y, x)`` coordinate grid.
    :rtype: SpatialImage
    """
    values = np.full(reader.GetNumberOfSpectra(), np.nan, dtype=np.float32)
    for record in records:
        if record["metric"] == metric:
            values[record["spectrum_id"]] = record["value"]
    return reader.MapSpectrumValuesToImage(values)


def plot_spatial_error_by_split(
    images_by_split: Mapping[str, SpatialImage],
    metric: str = "wasserstein",
    theme: VisualizationTheme | str | None = None,
) -> tuple[Any, np.ndarray]:
    """Plot one spatial error heatmap per dataset split, side by side, on one shared scale.

    A shared color scale (computed once across every split's finite values) is what
    makes the panels comparable — without it, each split's independent auto-scaling
    would make "does error concentrate in the same region regardless of split" or
    "is one split uniformly worse" impossible to read off the plots directly (the same
    reasoning as :func:`~..reconstruction.training_dynamics_analysis.
    plot_training_curves_by_binning`'s shared axis limits).

    :param images_by_split: :func:`spatial_reconstruction_error` output keyed by split
        name (e.g. ``"train"``, ``"validation"``, ``"test"``); panels are drawn in
        this mapping's order.
    :type images_by_split: Mapping[str, SpatialImage]
    :param metric: Metric name used in panel titles.
    :type metric: str
    :param theme: Global graphical strategy.
    :type theme: VisualizationTheme | str | None
    :return: Figure and flattened axes.
    :rtype: tuple[matplotlib.figure.Figure, numpy.ndarray]
    """
    resolved = resolve_theme(theme)
    finite_values = [
        image.values[np.isfinite(image.values)] for image in images_by_split.values()
    ]
    finite = np.concatenate(finite_values) if finite_values else np.asarray([])
    value_range = (
        (float(np.min(finite)), float(np.max(finite))) if finite.size else None
    )
    figure, axes = plt.subplots(
        1,
        len(images_by_split),
        figsize=(6 * len(images_by_split), 6),
        dpi=resolved.figure_dpi,
        squeeze=False,
    )
    for axis, (split_name, image) in zip(axes[0], images_by_split.items()):
        plot_spatial_image(
            image.values,
            title=f"{split_name}: {metric}",
            ax=axis,
            cmap=resolved.error_colormap,
            value_range=value_range,
            theme=resolved,
        )
    figure.tight_layout()
    return figure, axes.ravel()
