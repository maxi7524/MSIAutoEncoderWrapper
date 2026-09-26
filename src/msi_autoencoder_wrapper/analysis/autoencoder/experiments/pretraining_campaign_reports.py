"""Registered analyses of the pretraining-campaign notebooks.

Every function below turns the shared cache (:mod:`.pretraining_campaign_precompute`)
into the canonical tables of exactly one notebook. Quantities needed by several
notebooks (class selection, segmentation, latent summaries) are recomputed through the
same deterministic helper functions instead of being read from another analysis'
results, so every analysis can be (re)produced on its own.

Symbols used in the documentation of the tables:

* ``N``: pixels of one population; ``M``: bins of one axis; ``W``: m/z windows;
  ``C``: head classes; ``I``: METASPACE ions of the held-out images; ``D``: latent
  dimension.
"""

from __future__ import annotations

import json
from itertools import combinations
from typing import Any, Optional

import numpy as np
import pandas as pd
from scipy.ndimage import distance_transform_edt
from scipy.signal import find_peaks
from sklearn.decomposition import PCA

from ....utils.logger import get_custom_logger
from ..latent.predictive_geometry import (
    geometry_tables,
    intrinsic_dimension_estimate,
    label_structure_correlation,
    pairwise_geometry_battery,
    structure_summary,
)
from ..latent.sphere_geometry import canonicalize, normalize_to_constant_norm, structure_test, two_nn_intrinsic_dimension
from ..reconstruction.metrics import peak_matching_errors
from ..spatial import (
    adjusted_rand_between,
    align_labels,
    assemble_image,
    fit_segmentation,
    image_extent,
    presence_auc,
    segment_contingency,
    segment_marker_table,
    spatial_agreement,
)
from .pretraining_campaign_inference import SUPPORT_EDGES, class_strata
from .pretraining_campaign_precompute import (
    PIXEL_METRICS,
    PREDICTION_POPULATIONS,
    WINDOW_ARRAYS,
    CampaignCache,
    register,
)

logger = get_custom_logger(__name__)

IDENTITY = ("model_id", "model_alias", "display_label", "axis", "repetition")
RECONSTRUCTION_POPULATIONS = ("train", "test", "test_extended", "heldout_image")
LATENT_POPULATIONS = ("train", "test", "heldout_image")
HISTORY_BOOKKEEPING = {"epoch", "duration", "checkpoint_scope", "is_best", "best_loss"}
QUANTILES = np.round(np.linspace(0.0, 1.0, 51), 4)


# --------------------------------------------------
# Section: shared helpers
# --------------------------------------------------

def _identity(row: Any) -> dict:
    """Identity columns of one catalog record."""
    return {column: getattr(row, column) for column in IDENTITY}


def _rows(cache: CampaignCache, population: str) -> pd.DataFrame:
    """Pixel identities of one population in array-row order."""
    pixels = cache.population_frame()
    return pixels[pixels.population == population].sort_values("row").reset_index(drop=True)


def _summary(values: np.ndarray) -> dict:
    """Distribution summary over finite values."""
    values = np.asarray(values, dtype=np.float64)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {"mean": np.nan, "median": np.nan, "q10": np.nan, "q90": np.nan, "q99": np.nan,
                "pixels": 0, "defined_fraction": 0.0}
    return {"mean": finite.mean(), "median": np.median(finite), "q10": np.quantile(finite, 0.1),
            "q90": np.quantile(finite, 0.9), "q99": np.quantile(finite, 0.99), "pixels": int(finite.size),
            "defined_fraction": finite.size / values.size}


def _windows(cache: CampaignCache, axis: str) -> pd.DataFrame:
    """Window table of one axis: index, edges and ``lower-upper`` label."""
    edges = cache.axis_meta(axis)["window_edges"]
    return pd.DataFrame({"window": np.arange(len(edges) - 1), "window_lower": edges[:-1], "window_upper": edges[1:],
                         "window_label": [f"{lower:.0f}-{upper:.0f}" for lower, upper in zip(edges[:-1], edges[1:])]})


def _sample_rows(count: int, size: int, seed: int, salt: int) -> np.ndarray:
    """Deterministic sorted row sample shared by every model of one population."""
    return np.sort(np.random.default_rng([seed, salt]).choice(count, size=min(size, count), replace=False))


def _available_targets(arrays: dict) -> np.ndarray:
    """Available positive annotations, shape ``(N, C)``."""
    return np.asarray(arrays["targets"]).astype(bool) & np.asarray(arrays["mask"])


def _window_frame(values: np.ndarray, windows: pd.DataFrame, quantity: str, identity: dict) -> list[dict]:
    """Per-window distribution summaries of one ``(N, W)`` array."""
    rows = []
    for window in windows.itertuples():
        rows.append({**identity, "quantity": quantity, "window": window.window, "window_label": window.window_label,
                     "window_lower": window.window_lower, "window_upper": window.window_upper,
                     **_summary(values[:, window.window])})
    return rows


def _heldout_images(cache: CampaignCache) -> pd.DataFrame:
    """Held-out pixel identities with image-local coordinates."""
    return _rows(cache, "heldout_image")


def _probabilities(logits: np.ndarray) -> np.ndarray:
    """Logistic transform of head logits."""
    return 1.0 / (1.0 + np.exp(-np.asarray(logits, dtype=np.float64)))


# --------------------------------------------------
# Section: part 0 — training histories
# --------------------------------------------------

def _representative_table(cache: CampaignCache, axes: list[str]) -> pd.DataFrame:
    """Metric ranks of the repetitions of the given axes with the representative flag.

    Every image in the part-1 reports shows the representative model of its axis
    (see :func:`~.pretraining_campaign_inference.representative_ranking`), never a
    repetition mean, which would blur the images.
    """
    frames = []
    for axis_name in axes:
        identity = cache.axis_models(axis_name)[list(IDENTITY)]
        frames.append(cache.representative_ranking(axis_name).drop(columns="repetition").merge(
            identity, on="model_id", how="left", validate="many_to_one"))
    return pd.concat(frames, ignore_index=True)


def _history(cache: CampaignCache) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Long-form loss history and per-epoch bookkeeping of every selected model.

    Epoch records hold training values under the component name and validation values
    under ``validation_<component>``; the final record with ``split == 'test'`` holds
    the test evaluation of the restored best checkpoint. Epochs are counted per phase
    (a staged pretraining restarts at epoch 1), so every table carries ``phase`` and
    ``phase_index`` (position of the phase in the model's history); ``variant`` and
    ``stage`` identify the pretraining lineage.
    """
    from . import pretraining_campaign as campaign

    values, epochs = [], []
    for row in cache.models.itertuples():
        history = campaign.read_history(row.artifact)
        identity = {**_identity(row), "variant": row.variant, "stage": row.role}
        last_epoch, phases = 0, []
        for record in history:
            metrics = record.get("metrics", {})
            phase = record.get("phase")
            if phase not in phases:
                phases.append(phase)
            position = phases.index(phase)
            if record.get("split") == "test":
                for component, value in metrics.items():
                    if component not in HISTORY_BOOKKEEPING:
                        values.append({**identity, "phase": phase, "phase_index": position, "epoch": last_epoch,
                                       "split": "test", "component": component, "value": value})
                continue
            last_epoch = int(metrics["epoch"])
            epochs.append({**identity, "phase": phase, "phase_index": position, "epoch": last_epoch,
                           "duration_seconds": metrics.get("duration"), "is_best": bool(metrics.get("is_best")),
                           "checkpoint_scope": metrics.get("checkpoint_scope")})
            for key, value in metrics.items():
                if key in HISTORY_BOOKKEEPING:
                    continue
                split, component = (("validation", key[len("validation_"):]) if key.startswith("validation_")
                                    else ("train", key))
                values.append({**identity, "phase": phase, "phase_index": position, "epoch": last_epoch,
                               "split": split, "component": component, "value": value})
    return pd.DataFrame(values), pd.DataFrame(epochs)


@register("loss_overview", per_axis=False, requires=(), scope="all", version=2)
def loss_overview(cache: CampaignCache, axis: Optional[str]) -> dict:
    """Final, best and test values of every loss component, and training-health flags.

    Tables: ``history`` (long form), ``final_losses`` (model x phase x component x
    split), ``training_health`` (one row per model and phase) and ``group_summary``
    (model alias x phase x component x split, across repetitions).
    """
    history, epochs = _history(cache)
    identity = [*IDENTITY, "variant", "stage"]
    keys = [*identity, "phase", "phase_index", "component", "split"]
    epoch_values = history[history.split != "test"]
    final = epoch_values.sort_values("epoch").groupby(keys, as_index=False).last()[[*keys, "epoch", "value"]]
    final = final.rename(columns={"value": "final_value", "epoch": "final_epoch"})
    best_index = epoch_values.groupby(keys).value.idxmin()
    best = epoch_values.loc[best_index, [*keys, "epoch", "value"]].rename(
        columns={"value": "minimum_value", "epoch": "minimum_epoch"})
    test = history[history.split == "test"][[*identity, "phase", "component", "value"]].rename(
        columns={"value": "test_value", "phase": "test_phase"})
    final_losses = final.merge(best, on=keys).merge(test, on=[*identity, "component"], how="left")

    ## Health flags per model and phase
    health = []
    tolerance = 0.05
    for (model_id, phase), frame in epochs.groupby(["model_id", "phase"], sort=False):
        values = history[(history.model_id == model_id) & (history.phase == phase)]
        validation = values[(values.split == "validation") & (values.component == "total_loss")].sort_values("epoch")
        best_epochs = frame[frame.is_best].epoch
        minimum = float(validation.value.min()) if len(validation) else np.nan
        final_value = float(validation.value.iloc[-1]) if len(validation) else np.nan
        health.append({**{column: frame.iloc[0][column] for column in identity}, "phase": phase,
                       "phase_index": int(frame.phase_index.iloc[0]),
                       "epochs": int(frame.epoch.max()), "restored_epoch": int(best_epochs.max()) if len(best_epochs) else None,
                       "non_finite_values": int((~np.isfinite(values.value.astype(float))).sum()),
                       "total_duration_seconds": float(frame.duration_seconds.sum()),
                       "validation_total_final": final_value, "validation_total_minimum": minimum,
                       "validation_increase_after_minimum": bool(
                           final_value > minimum * (1 + tolerance) if minimum > 0 else final_value > minimum + tolerance),
                       "test_evaluated": bool((history[history.model_id == model_id].split == "test").any())})
    group = final_losses.groupby(["model_alias", "display_label", "axis", "variant", "stage", "phase", "component",
                                  "split"]).final_value.agg(["mean", "std", "min", "max", "count"]).reset_index()
    return {"history": history, "final_losses": final_losses, "training_health": pd.DataFrame(health),
            "group_summary": group, "metadata": {"divergence_tolerance": tolerance}}


@register("training_dynamics", per_axis=False, requires=(), scope="all", version=2)
def training_dynamics(cache: CampaignCache, axis: Optional[str]) -> dict:
    """Epoch trajectories of every loss component and their spread across repetitions.

    Tables: ``history`` (long form), ``group_trajectories`` (alias x phase x epoch x
    component x split: mean, sd, min, max, n) and ``epochs`` (duration and checkpoint
    flags).
    """
    history, epochs = _history(cache)
    trajectories = history[history.split != "test"].groupby(
        ["model_alias", "display_label", "axis", "variant", "stage", "phase", "phase_index", "component", "split",
         "epoch"]).value.agg(["mean", "std", "min", "max", "count"]).reset_index()
    return {"history": history, "group_trajectories": trajectories, "epochs": epochs, "metadata": {}}


# --------------------------------------------------
# Section: part 0 — split and population audit
# --------------------------------------------------

@register("split_population_audit", per_axis=False, scope="all", version=2)
def split_population_audit(cache: CampaignCache, axis: Optional[str]) -> dict:
    """Data-contract verification, population sizes, class coverage and acquisition ranges.

    Tables: ``verification``, ``population_sizes``, ``test_extension``,
    ``class_coverage``, ``heldout_label_coverage``, ``acquisition_windows``,
    ``acquisition_range``, ``history_consistency`` and ``metaspace_alignment``.
    """
    settings = cache.settings
    verification = cache.plan["verification"]
    pixels = cache.population_frame()
    sizes = pixels.groupby(["population", "dataset_id", "dataset_name"]).size().reset_index(name="pixels")
    visibility = []
    for axis_name in cache.axes:
        for population in RECONSTRUCTION_POPULATIONS:
            visible = np.asarray(cache.population(axis_name, population)["visible"], dtype=bool)
            visibility.append({"axis": axis_name, "population": population, "pixels": visible.size,
                               "annotation_visible": int(visible.sum())})
    coverage = []
    for axis_name in cache.axes:
        classes = cache.axis_table(axis_name, "classes")
        for population in RECONSTRUCTION_POPULATIONS:
            coverage.append(classes[["class_index", "class_name", "mz", "class_set"]].assign(
                axis=axis_name, population=population, positives=classes[f"{population}_positives"]))
    heldout_coverage = []
    metaspace = np.load(cache.population_root / "heldout_metaspace.npy", mmap_mode="r")
    for axis_name in cache.axes:
        ions = cache.axis_table(axis_name, "heldout_ions")
        ions["metaspace_positive_pixels"] = [int(np.nansum(np.asarray(metaspace[:, index]) > 0))
                                             for index in ions.ion_index]
        heldout_coverage.append(ions.drop(columns="bins").assign(axis=axis_name))

    ## Acquisition: mean input TIC fraction per m/z window and dataset
    minimum = float(settings["windows"].get("minimum_mass", 1e-3))
    acquisition = []
    for axis_name in cache.axes:
        reference = cache.axis_models(axis_name).iloc[0].model_id
        windows = _windows(cache, axis_name)
        for population in ("train", "heldout_image"):
            input_mass = cache.model_pixels(reference, population)["input_mass"]  # (N, W)
            datasets = _rows(cache, population).dataset_id.to_numpy()
            for dataset_id in np.unique(datasets):
                means = input_mass[datasets == dataset_id].mean(axis=0)  # (W,)
                acquisition.append(windows.assign(axis=axis_name, population=population, dataset_id=dataset_id,
                                                  mean_input_mass=means))
    acquisition = pd.concat(acquisition, ignore_index=True)
    signal = acquisition[acquisition.mean_input_mass >= minimum]
    ranges = signal.groupby(["axis", "population", "dataset_id"]).agg(
        lowest_window_lower=("window_lower", "min"), highest_window_upper=("window_upper", "max"),
        windows_with_signal=("window", "count")).reset_index()

    ## Training-time test Masserstein versus the recomputed value
    consistency = []
    for row in cache.models.itertuples():
        summary = cache.model_summary(row.model_id)
        recomputed = summary["populations"]["test"]["masserstein_mean"]
        recorded = summary.get("history_test_masserstein")
        consistency.append({**_identity(row), "recorded_test_masserstein": recorded,
                            "recomputed_test_masserstein": recomputed,
                            "relative_difference": (recomputed - recorded) / recorded if recorded else np.nan})
    complete = json.loads((cache.population_root / "complete.json").read_text())
    return {"verification": verification, "population_sizes": sizes,
            "population_visibility": pd.DataFrame(visibility),
            "test_extension": cache.shared_table("test_extension"),
            "class_coverage": pd.concat(coverage, ignore_index=True),
            "heldout_label_coverage": pd.concat(heldout_coverage, ignore_index=True),
            "acquisition_windows": acquisition, "acquisition_range": ranges,
            "history_consistency": pd.DataFrame(consistency),
            "metaspace_alignment": cache.shared_table("metaspace_alignment"),
            "metadata": {"heldout_datasets": complete["heldout_datasets"],
                         "heldout_empty_pixels": complete["heldout_empty_pixels"],
                         "minimum_window_mass": minimum,
                         ## REMARK: the trainer builds validation and test loaders from the
                         ## phase configuration, including `supervision_sampling`; recorded
                         ## validation/test losses are therefore estimates on a P:U-resampled
                         ## population, not on the plain partition evaluated here.
                         "history_note": "training-time validation/test values use the phase supervision "
                                         "sampler (P:U resampling with replacement); the recomputed value "
                                         "is the plain mean over every test pixel"}}


# --------------------------------------------------
# Section: part 1 — reconstruction
# --------------------------------------------------

@register("reconstruction_global", per_axis=True)
def reconstruction_global(cache: CampaignCache, axis: str) -> dict:
    """Per-pixel reconstruction metrics per population, image and repetition.

    Tables: ``summary`` (model x population x metric), ``quantiles`` (51 quantiles),
    ``image_breakdown`` (model x population x dataset x metric), ``pixel_sample``
    (fixed seeded pixels, identical for every model) and ``gaps`` (differences of
    population means per model).
    """
    sample_settings = cache.settings.get("pixel_sample", {})
    summary, quantiles, images, samples = [], [], [], []
    for row in cache.axis_models(axis).itertuples():
        for position, population in enumerate(RECONSTRUCTION_POPULATIONS):
            arrays = cache.model_pixels(row.model_id, population)
            identities = _rows(cache, population)
            sample = _sample_rows(len(identities), int(sample_settings.get("size", 2000)),
                                  int(sample_settings.get("seed", 42)), position)
            frame = identities.loc[sample, ["source_id", "dataset_id", "row"]].assign(
                **_identity(row), population=population)
            for metric in PIXEL_METRICS:
                values = arrays[metric]
                summary.append({**_identity(row), "population": population, "metric": metric, **_summary(values)})
                quantiles.append(pd.DataFrame({**_identity(row), "population": population, "metric": metric,
                                               "quantile": QUANTILES, "value": np.quantile(values, QUANTILES)}))
                frame[metric] = values[sample]
                grouped = pd.DataFrame({"dataset_id": identities.dataset_id, "value": values}).groupby("dataset_id").value
                images.append(grouped.agg(["mean", "median", "count"]).reset_index().assign(
                    **_identity(row), population=population, metric=metric))
            samples.append(frame)
    summary = pd.DataFrame(summary)
    means = summary.pivot_table(index=[*IDENTITY, "metric"], columns="population", values="mean").reset_index()
    gaps = means.assign(test_minus_train=means.test - means.train, heldout_minus_test=means.heldout_image - means.test,
                        heldout_minus_train=means.heldout_image - means.train)
    return {"summary": summary, "quantiles": pd.concat(quantiles, ignore_index=True),
            "image_breakdown": pd.concat(images, ignore_index=True),
            "pixel_sample": pd.concat(samples, ignore_index=True), "gaps": gaps,
            "metadata": {"metrics": list(PIXEL_METRICS), "populations": list(RECONSTRUCTION_POPULATIONS)}}


@register("reconstruction_windowed", per_axis=True)
def reconstruction_windowed(cache: CampaignCache, axis: str) -> dict:
    """Local (100 Da) Masserstein decomposition per population, image and bin.

    Tables: ``window_summary`` (model x population x window x quantity),
    ``window_images`` (per dataset means), ``window_gaps`` (population differences of
    window means) and ``feature_errors`` (per-bin mean errors, overall and per held-out
    image).
    """
    windows = _windows(cache, axis)
    summary, images, features = [], [], []
    for row in cache.axis_models(axis).itertuples():
        for population in RECONSTRUCTION_POPULATIONS:
            arrays = cache.model_pixels(row.model_id, population)
            quantities = {name: arrays[name] for name in WINDOW_ARRAYS}
            quantities["mass_imbalance"] = arrays["output_mass"] - arrays["input_mass"]
            total = arrays["masserstein"][:, None]  # (N, 1)
            quantities["contribution_share"] = np.divide(arrays["contribution"], total,
                                                         out=np.full_like(arrays["contribution"], np.nan),
                                                         where=total > 0)
            identity = {**_identity(row), "population": population}
            datasets = _rows(cache, population).dataset_id.to_numpy()
            for name, values in quantities.items():
                summary.extend(_window_frame(values, windows, name, identity))
                for dataset_id in np.unique(datasets):
                    selected = values[datasets == dataset_id]
                    with np.errstate(all="ignore"):
                        means = np.nanmean(selected, axis=0)
                    images.append(windows.assign(**identity, quantity=name, dataset_id=dataset_id, mean=means,
                                                 pixels=int(selected.shape[0])))
            features.append(cache.model_table(row.model_id, f"{population}_features").assign(
                **identity))
    summary = pd.DataFrame(summary)
    means = summary.pivot_table(index=[*IDENTITY, "quantity", "window", "window_label"], columns="population",
                                values="mean").reset_index()
    gaps = means.assign(test_minus_train=means.test - means.train, heldout_minus_test=means.heldout_image - means.test)
    return {"window_summary": summary, "window_images": pd.concat(images, ignore_index=True), "window_gaps": gaps,
            "feature_errors": pd.concat(features, ignore_index=True),
            "metadata": {"window_width": cache.settings["windows"]["width"],
                         "minimum_window_mass": cache.settings["windows"].get("minimum_mass", 1e-3)}}


@register("reconstruction_spectra", per_axis=True)
def reconstruction_spectra(cache: CampaignCache, axis: str) -> dict:
    """Example spectra, their window costs and peak-level reconstruction errors.

    Cases per population: fixed seeded rows plus every model's best and worst
    Masserstein rows. Tables: ``cases`` (input/output spectra of the representative
    model's cases), ``case_windows`` (window decomposition of every case, all
    repetitions), ``peak_matching`` (all repetitions) and ``representative_model``.
    """
    meta = cache.axis_meta(axis)
    windows = _windows(cache, axis)
    representative = cache.representative(axis).model_id
    spectra, case_windows, peaks = [], [], []
    for row in cache.axis_models(axis).itertuples():
        for population in RECONSTRUCTION_POPULATIONS:
            with np.load(cache.model_directory(row.model_id) / f"{population}_cases.npz") as archive:
                cases = {key: archive[key] for key in archive.files}
            identities = _rows(cache, population)
            arrays = cache.model_pixels(row.model_id, population)
            identity = {**_identity(row), "population": population}
            ## REMARK: a row can be fixed and extreme at once; the extreme kind wins and
            ## ``fixed`` keeps the membership of the seeded sample.
            kinds = np.where(cases["best"], "best", np.where(cases["worst"], "worst", "fixed"))
            for position, case_row in enumerate(cases["rows"]):
                base = {**identity, "row": int(case_row), "source_id": int(identities.source_id[case_row]),
                        "dataset_id": identities.dataset_id[case_row], "case_kind": kinds[position],
                        "fixed": bool(cases["fixed"][position]),
                        "masserstein": float(arrays["masserstein"][case_row])}
                if row.model_id == representative:
                    spectra.append(pd.DataFrame({**base, "bin": np.arange(meta["mass_axis"].size),
                                                 "mz": meta["mass_axis"], "input": cases["input"][position],
                                                 "output": cases["output"][position]}))
                case_windows.append(windows.assign(**base, **{name: arrays[name][case_row] for name in WINDOW_ARRAYS}))
            matched = peak_matching_errors(cases["input"], cases["output"], meta["mass_axis"])
            peaks.append(pd.DataFrame({**identity, "row": cases["rows"][matched["spectrum_index"]],
                                       "case_kind": kinds[matched["spectrum_index"]],
                                       **{key: matched[key] for key in ("peak_mz", "mz_error",
                                                                        "relative_intensity_error",
                                                                        "original_intensity", "detected")}}))
    return {"cases": pd.concat(spectra, ignore_index=True), "case_windows": pd.concat(case_windows, ignore_index=True),
            "peak_matching": pd.concat(peaks, ignore_index=True),
            "representative_model": _representative_table(cache, [axis]),
            "metadata": {"representative_model": representative,
                         "zoom_half_width": float(cache.settings.get("spectrum_zoom", {}).get("half_width", 15.0))}}


def _least_squares_slope(x: np.ndarray, y: np.ndarray) -> float:
    """Slope of the least-squares line of ``y`` on ``x``; ``nan`` for constant ``x``."""
    centred = x - x.mean()
    denominator = float(np.dot(centred, centred))
    return float(np.dot(centred, y - y.mean()) / denominator) if denominator > 0 else np.nan


@register("heldout_image_reconstruction", per_axis=True)
def heldout_image_reconstruction(cache: CampaignCache, axis: str) -> dict:
    """Masserstein error of the held-out images, pixel by pixel.

    The decoder output is TIC-normalized like its input, so the total mass of every
    pixel is reproduced by construction (``tic_error`` is zero); what can go wrong is how
    that mass is distributed along m/z. The maps show the representative model's global
    metrics, its exact window decomposition and the input and output mass of every
    displayed window; the repetition SD shows where the repetitions disagree.

    Tables: ``pixel_maps`` (one row per held-out pixel), ``window_mass_relation``
    (model x image x displayed window: agreement of the input and output window-mass
    images), ``image_metrics`` (model x image x metric), ``image_vs_test``,
    ``browser_pixels`` (representative model: best, median and worst pixel of every
    image), ``display_windows`` and ``representative_model``.
    """
    windows = _windows(cache, axis)
    models = cache.axis_models(axis)
    representative = cache.representative(axis)
    pixels = _heldout_images(cache)
    raw = cache.population(axis, "heldout_image")
    display_windows = cache.axis_table(axis, "display_windows")
    if len(display_windows) != len(windows):
        raise ValueError("Display windows and analysis windows disagree.")
    shown = windows[display_windows.displayed.astype(bool).to_numpy()]
    images = {dataset_id: frame.row.to_numpy() for dataset_id, frame in pixels.groupby("dataset_id")}
    stacks: dict[str, list[np.ndarray]] = {}
    image_rows, comparison, relation = [], [], []
    chosen: dict[str, np.ndarray] = {}
    for row in models.itertuples():
        arrays = cache.model_pixels(row.model_id, "heldout_image")
        test = cache.model_pixels(row.model_id, "test")
        if row.model_id == representative.model_id:
            chosen = arrays
        ## Image-level summaries of the global metrics
        for metric in PIXEL_METRICS:
            stacks.setdefault(metric, []).append(arrays[metric])
            for dataset_id, rows in images.items():
                values = arrays[metric][rows]
                image_rows.append({**_identity(row), "dataset_id": dataset_id, "metric": metric, **_summary(values)})
                comparison.append({**_identity(row), "dataset_id": dataset_id, "metric": metric,
                                   "image_mean": float(values.mean()), "test_mean": float(np.mean(test[metric])),
                                   "difference": float(values.mean() - np.mean(test[metric]))})
        ## Agreement of the input and output window-mass images
        for window in shown.itertuples():
            for dataset_id, rows in images.items():
                x = arrays["input_mass"][rows, window.window]  # (N_d,)
                y = arrays["output_mass"][rows, window.window]  # (N_d,)
                agreement = spatial_agreement(x, y)
                relation.append({**_identity(row), "dataset_id": dataset_id, "window": window.window,
                                 "window_label": window.window_label, "window_lower": window.window_lower,
                                 "pearson": agreement["pearson"], "spearman": agreement["spearman"],
                                 "slope": _least_squares_slope(x, y), "mean_input_mass": float(x.mean()),
                                 "mean_output_mass": float(y.mean()), "pixels": int(rows.size)})

    # Per-pixel maps of the representative model
    maps = pixels[["row", "source_id", "dataset_id", "dataset_name", "x", "y"]].copy()
    maps["input_tic"] = np.asarray(raw["tic"])
    for metric in PIXEL_METRICS:
        maps[metric] = chosen[metric]
        maps[f"{metric}_sd"] = np.stack(stacks[metric]).std(axis=0)  # (N,)
    imbalance = chosen["output_mass"] - chosen["input_mass"]  # (N, W)
    for window in windows.itertuples():
        maps[f"contribution__{window.window_label}"] = chosen["contribution"][:, window.window]
        maps[f"within__{window.window_label}"] = chosen["within"][:, window.window]
        maps[f"mass_imbalance__{window.window_label}"] = imbalance[:, window.window]
    for window in shown.itertuples():
        maps[f"input_mass__{window.window_label}"] = chosen["input_mass"][:, window.window]
        maps[f"output_mass__{window.window_label}"] = chosen["output_mass"][:, window.window]

    ## Deterministic browser pixels: best, median and worst Masserstein of the representative model
    bins = np.asarray(cache.axis_array(axis, "display_bins"))
    display_input = cache.axis_array(axis, "display_input")
    reconstruction = cache.model_array(representative.model_id, "display_reconstruction")
    mass_axis = cache.axis_meta(axis)["mass_axis"]
    browser = []
    for dataset_id, frame in maps.groupby("dataset_id"):
        ordered = frame.sort_values("masserstein").row.to_numpy()
        for label, pixel in (("best", ordered[0]), ("median", ordered[len(ordered) // 2]), ("worst", ordered[-1])):
            browser.append(pd.DataFrame({**_identity(representative), "dataset_id": dataset_id, "pixel_kind": label,
                                         "row": int(pixel), "x": int(maps.x[pixel]), "y": int(maps.y[pixel]),
                                         "masserstein": float(maps.masserstein[pixel]),
                                         "bin": bins, "mz": mass_axis[bins],
                                         "input": np.asarray(display_input[pixel], dtype=np.float32),
                                         "output": np.asarray(reconstruction[pixel], dtype=np.float32)}))
    population_directory = cache.population_root / cache.settings["axes"][axis]["directory"]
    return {"pixel_maps": maps, "window_mass_relation": pd.DataFrame(relation),
            "image_metrics": pd.DataFrame(image_rows), "image_vs_test": pd.DataFrame(comparison),
            "browser_pixels": pd.concat(browser, ignore_index=True), "display_windows": display_windows,
            "representative_model": _representative_table(cache, [axis]),
            "metadata": {"representative_model": representative.model_id,
                         "displayed_windows": shown.window_label.tolist(),
                         "display_input": str(population_directory / "display_input.npy"),
                         "display_bins": str(population_directory / "display_bins.npy"),
                         "display_reconstruction": str(cache.model_directory(representative.model_id)
                                                       / "display_reconstruction.npy")}}


def _image_channels(spectra: np.ndarray, images: dict[str, np.ndarray], mass_axis: np.ndarray,
                    display_bins: np.ndarray, options: dict) -> pd.DataFrame:
    """Peak-apex m/z channels of every held-out image.

    A channel is the apex bin of a peak of the image's mean input spectrum whose height
    is at least ``minimum_height_fraction`` of the image's base peak; apexes closer than
    ``minimum_distance_bins`` keep the higher one, so neighbouring bins of one peak do
    not appear as separate channels. Only bins with a stored display reconstruction are
    kept (windows with negligible held-out mass are not stored).

    :param spectra: Held-out input spectra, shape ``(N_h, M)``.
    :param images: Held-out rows of every image.
    :return: One row per image and channel (``bin``, ``mz``, ``mean_input``, ``height_fraction``).
    :rtype: pandas.DataFrame
    """
    fraction = float(options.get("minimum_height_fraction", 1e-3))
    distance = int(options.get("minimum_distance_bins", 3))
    displayed = np.isin(np.arange(mass_axis.size), display_bins)  # (M,)
    frames = []
    for dataset_id, rows in images.items():
        mean = np.asarray(spectra[rows], dtype=np.float64).mean(axis=0)  # (M,)
        apexes, _ = find_peaks(mean, height=fraction * mean.max(), distance=distance)
        apexes = apexes[displayed[apexes]]
        frames.append(pd.DataFrame({"dataset_id": dataset_id, "bin": apexes, "mz": mass_axis[apexes],
                                    "mean_input": mean[apexes], "height_fraction": mean[apexes] / mean.max()}))
    return pd.concat(frames, ignore_index=True)


@register("reconstructed_ion_images", per_axis=True)
def reconstructed_ion_images(cache: CampaignCache, axis: str) -> dict:
    """Appearance of the reconstructed held-out images, channel by channel.

    Two views. (1) m/z channels: every peak-apex channel of every held-out image
    (:func:`_image_channels`) is compared as an image, input versus the representative
    model's reconstruction, by Pearson correlation over the image's pixels (shape,
    independent of scale) together with the relative L1 error and the intensity ratio;
    per image the ``per_kind`` best, median and worst channels by the configured
    criterion are selected for display with their error image. (2) Annotated ions:
    METASPACE, input and reconstruction of every METASPACE annotation of the image.
    Inputs and reconstructions are TIC-normalized, as seen by the model.

    Tables: ``channel_metrics`` (image x channel), ``channel_selection``,
    ``channel_pixels`` (selected channels: input, output and error per pixel),
    ``ion_agreement`` (model x ion), ``image_fidelity`` (model x image median
    agreement), ``ion_pixels`` (per pixel of every annotated ion: METASPACE, binary
    annotation target, input, representative output) and ``representative_model``.
    """
    options = cache.settings.get("image_channels", {})
    per_kind = int(options.get("per_kind", 2))
    criterion = str(options.get("criterion", "pearson"))
    representative = cache.representative(axis)
    pixels = _heldout_images(cache)
    images = {dataset_id: frame.row.to_numpy() for dataset_id, frame in pixels.groupby("dataset_id")}
    raw = cache.population(axis, "heldout_image")
    spectra = raw["spectra"]
    display_bins = np.asarray(cache.axis_array(axis, "display_bins"))
    column_of_bin = {int(value): position for position, value in enumerate(display_bins)}
    reconstruction = cache.model_array(representative.model_id, "display_reconstruction")  # (N_h, K)

    # m/z channel images of the representative model
    channels = _image_channels(spectra, images, cache.axis_meta(axis)["mass_axis"], display_bins, options)
    metrics, selected_pixels, selection = [], [], []
    for dataset_id, frame in channels.groupby("dataset_id", sort=False):
        rows = images[dataset_id]
        columns = [column_of_bin[int(value)] for value in frame["bin"]]
        observed = np.asarray(spectra[rows][:, frame["bin"].to_numpy()], dtype=np.float64)  # (N_d, C_d)
        produced = np.asarray(reconstruction[rows][:, columns], dtype=np.float64)  # (N_d, C_d)
        for position, channel in enumerate(frame.itertuples()):
            x, y = observed[:, position], produced[:, position]
            agreement = spatial_agreement(x, y)
            total = float(x.sum())
            metrics.append({**_identity(representative), "dataset_id": dataset_id, "bin": int(channel.bin),
                            "mz": float(channel.mz), "mean_input": float(channel.mean_input),
                            "height_fraction": float(channel.height_fraction), "pearson": agreement["pearson"],
                            "spearman": agreement["spearman"],
                            "relative_l1": float(np.abs(y - x).sum() / total) if total > 0 else np.nan,
                            "intensity_ratio": float(y.sum() / total) if total > 0 else np.nan,
                            "column": position})
        ## Best, median and worst channels of the image by the criterion
        ranked = pd.DataFrame(metrics[-len(frame):]).dropna(subset=[criterion]).sort_values(criterion, ascending=False)
        middle = max(len(ranked) // 2 - per_kind // 2, 0)
        for kind, chosen in (("best", ranked.head(per_kind)), ("median", ranked.iloc[middle:middle + per_kind]),
                             ("worst", ranked.tail(per_kind).iloc[::-1])):
            for rank, record in enumerate(chosen.itertuples()):
                selection.append({"dataset_id": dataset_id, "kind": kind, "rank": rank, "bin": record.bin,
                                  "mz": record.mz, "pearson": record.pearson, "spearman": record.spearman,
                                  "relative_l1": record.relative_l1, "intensity_ratio": record.intensity_ratio,
                                  "channels": len(ranked)})
                x, y = observed[:, record.column], produced[:, record.column]
                selected_pixels.append(pixels.loc[rows, ["row", "dataset_id", "x", "y"]].assign(
                    kind=kind, rank=rank, bin=record.bin, mz=record.mz, input=x, output=y, error=y - x))
    channel_metrics = pd.DataFrame(metrics).drop(columns="column")

    # METASPACE annotations: agreement for every model, pixels of the representative model
    ions = cache.axis_table(axis, "heldout_ions")
    metaspace = np.load(cache.population_root / "heldout_metaspace.npy", mmap_mode="r")  # (N_h, I)
    observed_ions = np.asarray(cache.axis_array(axis, "heldout_ion_input"))  # (N_h, I)
    targets = _available_targets(raw)  # (N_h, C)
    models = cache.axis_models(axis)
    outputs = {row.model_id: np.asarray(cache.model_array(row.model_id, "heldout_image_ions")) for row in models.itertuples()}
    agreement_rows, pixel_rows = [], []
    for ion in ions[ions.in_axis].itertuples():
        rows = pixels.dataset_id.eq(ion.dataset_id).to_numpy()
        reference = np.asarray(metaspace[rows, ion.ion_index], dtype=np.float64)
        source = observed_ions[rows, ion.ion_index]
        base = {"ion_index": ion.ion_index, "dataset_id": ion.dataset_id, "class_name": ion.class_name, "mz": ion.mz,
                "class_set": ion.class_set, "head_column": ion.head_column}
        for row in models.itertuples():
            output = outputs[row.model_id][rows, ion.ion_index]
            input_output = spatial_agreement(source, output)
            agreement_rows.append({**_identity(row), **base,
                                   "input_output_spearman": input_output["spearman"],
                                   "input_output_pearson": input_output["pearson"],
                                   "metaspace_input_spearman": spatial_agreement(reference, source)["spearman"],
                                   "metaspace_output_spearman": spatial_agreement(reference, output)["spearman"],
                                   "intensity_ratio": float(output.sum() / source.sum()) if source.sum() > 0 else np.nan,
                                   "mean_absolute_relative_error": float(np.abs(output - source).sum() / source.sum())
                                   if source.sum() > 0 else np.nan, "pixels": int(rows.sum())})
        target = (targets[rows, ion.head_column].astype(np.float32) if ion.head_column >= 0
                  else np.full(int(rows.sum()), np.nan, dtype=np.float32))
        pixel_rows.append(pixels.loc[rows, ["row", "dataset_id", "x", "y"]].assign(
            **base, metaspace=reference, target=target, input=source,
            output=outputs[representative.model_id][rows, ion.ion_index]))
    agreement = pd.DataFrame(agreement_rows)
    fidelity = agreement.groupby([*IDENTITY, "dataset_id"])[
        ["input_output_spearman", "metaspace_input_spearman", "metaspace_output_spearman", "intensity_ratio"]
    ].median().reset_index()
    return {"channel_metrics": channel_metrics, "channel_selection": pd.DataFrame(selection),
            "channel_pixels": pd.concat(selected_pixels, ignore_index=True),
            "ion_agreement": agreement, "image_fidelity": fidelity,
            "ion_pixels": pd.concat(pixel_rows, ignore_index=True),
            "representative_model": _representative_table(cache, [axis]),
            "metadata": {"representative_model": representative.model_id, "criterion": criterion,
                         "per_kind": per_kind, "ions_in_axis": int(ions.in_axis.sum()), "ions_total": len(ions)}}


# --------------------------------------------------
# Section: part 1 — latent space
# --------------------------------------------------

def _latent_codes(cache: CampaignCache, row: Any, population: str) -> tuple[np.ndarray, np.ndarray]:
    """Raw codes ``z`` and canonicalized codes ``u`` of one model, shape ``(N, D)`` each."""
    summary = cache.model_summary(row.model_id)
    z = cache.model_pixels(row.model_id, population)["latent"]
    return z, canonicalize(z, np.asarray(summary["gamma"]), np.asarray(summary["beta"]))


def _latent_samples(cache: CampaignCache, axis: str) -> dict[str, np.ndarray]:
    """Shared latent sample rows per population (identical for every model and axis)."""
    latent = cache.settings.get("latent", {})
    return {population: _sample_rows(len(_rows(cache, population)), int(latent.get("sample_size", 2000)),
                                     int(latent.get("seed", 42)), 10 + position)
            for position, population in enumerate(LATENT_POPULATIONS)}


def _spectrum_usage(eigenvalues: np.ndarray) -> dict:
    """Effective rank and participation ratio of a nonnegative variance spectrum."""
    values = np.clip(np.asarray(eigenvalues, dtype=np.float64), 0.0, None)
    total = values.sum()
    proportions = values[values > 0] / total
    return {"trace": total, "effective_rank": float(np.exp(-np.sum(proportions * np.log(proportions)))),
            "participation_ratio": float(total ** 2 / np.sum(values ** 2))}


def latent_model_summary(cache: CampaignCache, row: Any, samples: dict[str, np.ndarray],
                         populations: tuple[str, ...] = LATENT_POPULATIONS) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Dimension usage, TwoNN, angular structure and label correlation of one model.

    :param cache: Cache view.
    :type cache: CampaignCache
    :param row: Model record.
    :type row: typing.Any
    :param samples: Shared sampled rows per population (:func:`_latent_samples`).
    :type samples: dict[str, numpy.ndarray]
    :param populations: Populations to describe.
    :type populations: tuple[str, ...]
    :return: Long-form ``geometry`` records (``space`` z or u) and ``eigenvalues``.
    :rtype: tuple[pandas.DataFrame, pandas.DataFrame]
    """
    latent = cache.settings.get("latent", {})
    seed = int(latent.get("seed", 42))
    geometry, eigenvalues = [], []
    for population in populations:
        z, u = _latent_codes(cache, row, population)
        targets = _available_targets(cache.population(row.axis, population))
        sample = samples[population]
        identity = {**_identity(row), "population": population}
        for space, values in (("z", z), ("u", u)):
            tables = geometry_tables(values, targets, sample, k=int(latent.get("neighbours", 10)))
            geometry.append(tables["geometry"].assign(**identity, space=space))
            eigenvalues.append(tables["eigenvalues"].assign(**identity, space=space))
        extra = [{"metric": key, "value": value} for key, value in structure_summary(
            u, sample, np.random.default_rng([seed, 1]), pair_count=int(latent.get("pair_count", 5000))).items()]
        two_nn = intrinsic_dimension_estimate(u, sample)
        ## REMARK: TwoNN above the ambient dimension indicates a collapsed cloud,
        ## not an intrinsic dimension; it is reported as missing (as in the sweeps).
        admissible = bool(np.isfinite(two_nn) and two_nn <= u.shape[1])
        extra.append({"metric": "two_nn_intrinsic_dimension", "value": two_nn if admissible else np.nan})
        extra.append({"metric": "two_nn_admissible", "value": float(admissible)})
        correlation = label_structure_correlation(u, sample, targets, np.random.default_rng([seed, 2]),
                                                  pair_count=int(latent.get("pair_count", 5000)))
        extra.extend({"metric": f"label_correlation_{key}", "value": value} for key, value in correlation.items())
        geometry.append(pd.DataFrame(extra).assign(**identity, space="u"))
    return pd.concat(geometry, ignore_index=True), pd.concat(eigenvalues, ignore_index=True)


def _latent_summaries(cache: CampaignCache, axis: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Dimension usage, TwoNN, angular structure and label correlation per model and population.

    :return: Long-form ``geometry`` records (``space`` z or u) and ``eigenvalues``.
    """
    samples = _latent_samples(cache, axis)
    results = [latent_model_summary(cache, row, samples) for row in cache.axis_models(axis).itertuples()]
    return (pd.concat([geometry for geometry, _ in results], ignore_index=True),
            pd.concat([eigenvalues for _, eigenvalues in results], ignore_index=True))


def _data_geometry(cache: CampaignCache, axis: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Dimension of the input spectra themselves (model independent).

    Linear dimension from the PCA spectrum of the sampled TIC-normalized inputs and a
    nonlinear TwoNN estimate on the same rows, rescaled to constant norm so that the
    angular estimator applies unchanged.
    """
    latent = cache.settings.get("latent", {})
    components = int(latent.get("data_components", 50))
    samples = _latent_samples(cache, axis)
    rows, spectrum = [], []
    for population in LATENT_POPULATIONS:
        spectra = np.asarray(cache.population(axis, population)["spectra"][samples[population]], dtype=np.float64)
        spectra = spectra[np.linalg.norm(spectra, axis=1) > 0]  # (S, M)
        centered = spectra - spectra.mean(axis=0, keepdims=True)
        singular = np.linalg.svd(centered, compute_uv=False)  # (min(S, M),)
        eigen = singular ** 2 / len(spectra)
        ratio = eigen / eigen.sum()
        cumulative = np.cumsum(ratio)
        record = {"axis": axis, "population": population, "rows": len(spectra), **_spectrum_usage(eigen),
                  "two_nn_intrinsic_dimension": two_nn_intrinsic_dimension(normalize_to_constant_norm(spectra))}
        for level in (0.8, 0.9, 0.95, 0.99):
            record[f"components_for_{int(level * 100)}pct_variance"] = int(np.searchsorted(cumulative, level) + 1)
        rows.append(record)
        spectrum.append(pd.DataFrame({"axis": axis, "population": population,
                                      "component": np.arange(1, min(components, ratio.size) + 1),
                                      "explained_variance_ratio": ratio[:components],
                                      "cumulative": cumulative[:components]}))
    return pd.DataFrame(rows), pd.concat(spectrum, ignore_index=True)


def _reproducibility(cache: CampaignCache, axis_pairs: list[tuple[Any, Any]], populations: tuple[str, ...]) -> pd.DataFrame:
    """Pairwise representation similarity of two models on identical sampled pixels."""
    latent = cache.settings.get("latent", {})
    rows = []
    for population in populations:
        sample = _latent_samples(cache, axis_pairs[0][0].axis)[population] if axis_pairs else None
        codes: dict[str, np.ndarray] = {}
        for left, right in axis_pairs:
            for row in (left, right):
                if row.model_id not in codes:
                    codes[row.model_id] = _latent_codes(cache, row, population)[1]
            battery = pairwise_geometry_battery(codes[left.model_id], codes[right.model_id], sample,
                                                k=int(latent.get("neighbours", 10)))
            rows.append({"population": population, "left_model": left.model_id, "right_model": right.model_id,
                         "left_axis": left.axis, "right_axis": right.axis,
                         "left_repetition": left.repetition, "right_repetition": right.repetition, **battery})
    return pd.DataFrame(rows)


@register("latent_geometry", per_axis=True)
def latent_geometry(cache: CampaignCache, axis: str) -> dict:
    """Latent dimension usage, angular structure, reproducibility and sensitivity.

    Tables: ``geometry`` (long form, spaces z and u), ``eigenvalues``,
    ``data_geometry`` and ``data_spectrum`` (dimension of the inputs),
    ``angular_samples`` (raw cos-theta pairs), ``latent_quantiles`` (per latent
    dimension), ``reproducibility`` (pairs of repetitions) and ``sensitivity``.
    """
    latent = cache.settings.get("latent", {})
    seed = int(latent.get("seed", 42))
    samples = _latent_samples(cache, axis)
    geometry, eigenvalues = _latent_summaries(cache, axis)
    data_geometry, data_spectrum = _data_geometry(cache, axis)
    angular, quantiles = [], []
    levels = np.array([0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99])
    models = cache.axis_models(axis)
    for row in models.itertuples():
        for population in LATENT_POPULATIONS:
            z, u = _latent_codes(cache, row, population)
            result = structure_test(u[samples[population]], np.random.default_rng([seed, 1]),
                                    pair_count=int(latent.get("pair_count", 5000)), return_samples=True)
            angular.append(pd.DataFrame({**_identity(row), "population": population,
                                         "cos_theta": np.asarray(result["cos_theta_samples"])}))
            values = np.quantile(z, levels, axis=0)  # (Q, D)
            for dimension in range(z.shape[1]):
                quantiles.append({**_identity(row), "population": population, "dimension": dimension,
                                  "mean": float(z[:, dimension].mean()), "sd": float(z[:, dimension].std()),
                                  **{f"q{int(level * 100):02d}": float(values[index, dimension])
                                     for index, level in enumerate(levels)}})
    pairs = list(combinations(list(models.itertuples()), 2))
    return {"geometry": geometry, "eigenvalues": eigenvalues, "data_geometry": data_geometry,
            "data_spectrum": data_spectrum, "angular_samples": pd.concat(angular, ignore_index=True),
            "latent_quantiles": pd.DataFrame(quantiles),
            "reproducibility": _reproducibility(cache, pairs, ("test", "heldout_image")),
            "sensitivity": cache.collect("sensitivity", models),
            "metadata": {"sample_size": int(latent.get("sample_size", 2000)), "seed": seed,
                         "pair_count": int(latent.get("pair_count", 5000))}}


# --------------------------------------------------
# Section: part 1 — segmentation
# --------------------------------------------------

def _segmentations(cache: CampaignCache, axis: str, *, refits: bool = True) -> dict:
    """Gaussian-mixture segmentation of every model's latent codes and of input PCA.

    The mixture is fitted on a seeded sample of training pixels and applied to the
    held-out pixels. The input-PCA reference uses the same procedure on the first
    ``D`` principal components of the TIC-normalized inputs, so it shows which spatial
    structure is already linearly present in the spectra. The representative model is
    additionally fitted on ``resamples`` further training subsamples (seeds ``seed + i``),
    which isolates the variability of the mixture fit from that of the model. Every
    partition is aligned (Hungarian matching) to the representative model's partition of
    the same ``k``, so equal indices (colours) denote matching segments.

    :return: ``bic`` rows, ``labels`` keyed by ``(representation, repetition, k)``,
        ``resamples`` keyed by ``(resample, k)``, the selected ``k`` (lowest mean latent
        BIC) and the reference repetition.
    """
    settings = cache.settings.get("segmentation", {})
    seed = int(settings.get("seed", 42))
    sample_size = int(settings.get("fit_sample_size", 20000))
    k_values = [int(value) for value in settings.get("k_values", (4, 6, 8, 12))]
    resample_count = int(settings.get("resamples", 5)) if refits else 0
    models = cache.axis_models(axis)
    representative = cache.representative(axis)
    bic, labels, resamples = [], {}, {}
    for row in models.itertuples():
        train = cache.model_pixels(row.model_id, "train")["latent"]
        heldout = cache.model_pixels(row.model_id, "heldout_image")["latent"]
        for k in k_values:
            fit = fit_segmentation(train, k, seed=seed, sample_size=sample_size)
            bic.append({"representation": "latent", "repetition": int(row.repetition), "model_id": row.model_id,
                        "k": k, "bic": fit.bic})
            labels[("latent", int(row.repetition), k)] = fit.predict(heldout)
            if row.model_id == representative.model_id:
                for resample in range(1, resample_count + 1):
                    resamples[(resample, k)] = fit_segmentation(train, k, seed=seed + resample,
                                                                sample_size=sample_size).predict(heldout)
    ## Input-PCA reference with the latent dimension as number of components
    dimension = cache.model_pixels(models.iloc[0].model_id, "heldout_image")["latent"].shape[1]
    train_spectra = cache.population(axis, "train")["spectra"]
    rows = _sample_rows(train_spectra.shape[0], sample_size, seed, 50)
    pca = PCA(n_components=dimension, random_state=seed).fit(np.asarray(train_spectra[rows], dtype=np.float32))
    projected_train = pca.transform(np.asarray(train_spectra[rows], dtype=np.float32))
    projected_heldout = pca.transform(np.asarray(cache.population(axis, "heldout_image")["spectra"], dtype=np.float32))
    for k in k_values:
        fit = fit_segmentation(projected_train, k, seed=seed, sample_size=sample_size)
        bic.append({"representation": "input_pca", "repetition": -1, "model_id": "", "k": k, "bic": fit.bic})
        labels[("input_pca", -1, k)] = fit.predict(projected_heldout)
    ## Alignment to the representative model
    reference_repetition = int(representative.repetition)
    for (representation, repetition, k), values in list(labels.items()):
        if (representation, repetition) != ("latent", reference_repetition):
            labels[(representation, repetition, k)] = align_labels(labels[("latent", reference_repetition, k)], values, k)
    for (resample, k), values in list(resamples.items()):
        resamples[(resample, k)] = align_labels(labels[("latent", reference_repetition, k)], values, k)
    bic = pd.DataFrame(bic)
    selected = int(bic[bic.representation == "latent"].groupby("k").bic.mean().idxmin())
    return {"bic": bic, "labels": labels, "resamples": resamples, "selected_k": selected, "k_values": k_values,
            "reference_repetition": reference_repetition, "representative_model": representative.model_id}


def _class_selection(cache: CampaignCache, axis: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-image class agreement with METASPACE and the best/worst class selection.

    Agreement of a class in one held-out image uses the METASPACE ion image of that
    image as reference and the head probability map as candidate. Classes are eligible
    for selection when both METASPACE and the annotations mark at least
    ``class_selection.minimum_positive_pixels`` pixels. Two criteria are selected
    independently: ``metaspace_spearman`` (spatial reproduction of METASPACE) and
    ``average_precision`` (the ranking the head was trained for, annotation retrieval,
    per image). Both are evaluated on the representative model, the model whose maps
    are displayed.

    :return: ``agreement`` (model x image x class) and ``selection``.
    """
    options = cache.settings.get("class_selection", {})
    count, minimum = int(options.get("count", 3)), int(options.get("minimum_positive_pixels", 20))
    ions = cache.axis_table(axis, "heldout_ions")
    ions = ions[ions.in_axis & (ions.head_column >= 0)]
    pixels = _heldout_images(cache)
    metaspace = np.load(cache.population_root / "heldout_metaspace.npy", mmap_mode="r")
    targets = _available_targets(cache.population(axis, "heldout_image"))
    rows = []
    for row in cache.axis_models(axis).itertuples():
        probabilities = np.asarray(cache.model_array(row.model_id, "heldout_image_probabilities"), dtype=np.float64)
        per_class = cache.model_table(row.model_id, "per_class")
        per_class = per_class[(per_class.evaluation == "heldout_image") & (per_class.population == "annotation_retrieval")
                              & (per_class.group != "all")].set_index(["group", "class_name"])
        for ion in ions.itertuples():
            selected = pixels.dataset_id.eq(ion.dataset_id).to_numpy()
            reference = np.asarray(metaspace[selected, ion.ion_index], dtype=np.float64)
            probability = probabilities[selected, ion.head_column]
            agreement = spatial_agreement(reference, probability)
            key = (ion.dataset_id, ion.class_name)
            rows.append({**_identity(row), "ion_index": ion.ion_index, "dataset_id": ion.dataset_id,
                         "class_name": ion.class_name, "mz": ion.mz, "class_set": ion.class_set,
                         "head_column": ion.head_column, "metaspace_spearman": agreement["spearman"],
                         "metaspace_pearson": agreement["pearson"],
                         "presence_auc": presence_auc(probability, np.nan_to_num(reference) > 0),
                         "average_precision": float(per_class.average_precision.get(key, np.nan)),
                         "metaspace_positive_pixels": int(np.nansum(reference > 0)),
                         "annotation_positive_pixels": int(targets[selected, ion.head_column].sum()),
                         "pixels": int(selected.sum())})
    agreement = pd.DataFrame(rows)
    chosen = agreement[agreement.model_id == cache.representative(axis).model_id]
    eligible = chosen[(chosen.metaspace_positive_pixels >= minimum) & (chosen.annotation_positive_pixels >= minimum)]
    selection = []
    for criterion in ("metaspace_spearman", "average_precision"):
        ordered = eligible.dropna(subset=[criterion]).sort_values(criterion, ascending=False)
        for kind, frame in (("best", ordered.head(count)), ("worst", ordered.tail(count).iloc[::-1])):
            for rank, record in enumerate(frame.itertuples()):
                selection.append({"criterion": criterion, "kind": kind, "rank": rank, "ion_index": record.ion_index,
                                  "dataset_id": record.dataset_id, "class_name": record.class_name, "mz": record.mz,
                                  "class_set": record.class_set, "head_column": record.head_column,
                                  "value": getattr(record, criterion)})
    return agreement, pd.DataFrame(selection)


@register("latent_segmentation", per_axis=True)
def latent_segmentation(cache: CampaignCache, axis: str) -> dict:
    """Latent segmentation of the held-out images and its relation to molecular classes.

    Every partition is aligned to the representative model's partition of the same
    ``k``. Tables: ``bic``, ``labels`` (held-out pixel x representation x repetition x k),
    ``rgb`` (first three principal components of the representative model's held-out
    codes scaled to [0, 1]), ``markers`` (most enriched bins per segment of the
    representative model and of input PCA), ``stability`` (ARI between repetitions,
    against input PCA and between mixture refits of the representative model on
    independent training subsamples), ``contingency`` (pixel overlap of the
    representative partition with every other partition), ``composition`` (segment
    fractions per image), ``class_overlap``, ``exploded_layers`` (layers of the
    image x m/z view for the selected classes) and ``representative_model``.
    """
    settings = cache.settings.get("segmentation", {})
    result = _segmentations(cache, axis)
    representative = cache.representative(axis)
    reference = result["reference_repetition"]
    pixels = _heldout_images(cache)
    base = pixels[["row", "dataset_id", "x", "y"]]
    labels = pd.concat([base.assign(representation=representation, repetition=repetition, k=k, segment=values)
                        for (representation, repetition, k), values in result["labels"].items()], ignore_index=True)
    ## RGB composite of the representative model's codes
    codes = cache.model_pixels(representative.model_id, "heldout_image")["latent"]  # (N, D)
    components = PCA(n_components=3, random_state=int(settings.get("seed", 42))).fit_transform(codes)  # (N, 3)
    low, high = np.percentile(components, 1, axis=0), np.percentile(components, 99, axis=0)
    scaled = np.clip((components - low) / np.where(high > low, high - low, 1.0), 0.0, 1.0)  # (N, 3)
    rgb = base.assign(**_identity(representative), r=scaled[:, 0], g=scaled[:, 1], b=scaled[:, 2])
    ## Markers of the displayed partitions and composition of every partition
    spectra = np.asarray(cache.population(axis, "heldout_image")["spectra"], dtype=np.float32)
    mass_axis = cache.axis_meta(axis)["mass_axis"]
    markers, composition = [], []
    for (representation, repetition, k), values in result["labels"].items():
        if (representation, repetition) in (("latent", reference), ("input_pca", -1)):
            table = segment_marker_table(spectra, values, mass_axis, top=int(settings.get("marker_top", 10)),
                                         minimum_mean=float(settings.get("marker_minimum_mean", 1e-5)))
            markers.append(table.assign(representation=representation, repetition=repetition, k=k))
        frame = pd.DataFrame({"dataset_id": pixels.dataset_id, "segment": values})
        fractions = frame.groupby("dataset_id").segment.value_counts(normalize=True).rename("fraction").reset_index()
        composition.append(fractions.assign(representation=representation, repetition=repetition, k=k))
    ## Stability and overlap: repetitions, input PCA and mixture refits
    stability, contingency = [], []
    repetitions = sorted({key[1] for key in result["labels"] if key[0] == "latent"})
    for k in result["k_values"]:
        main = result["labels"][("latent", reference, k)]
        for left, right in combinations(repetitions, 2):
            stability.append({"k": k, "comparison": "latent_repetitions", "left": left, "right": right,
                              "ari": adjusted_rand_between(result["labels"][("latent", left, k)],
                                                           result["labels"][("latent", right, k)])})
        others = [("latent_repetition", repetition, result["labels"][("latent", repetition, k)])
                  for repetition in repetitions if repetition != reference]
        others.append(("input_pca", -1, result["labels"][("input_pca", -1, k)]))
        others += [("mixture_refit", resample, values) for (resample, resample_k), values in result["resamples"].items()
                   if resample_k == k]
        for comparison, other, values in others:
            if comparison != "latent_repetition":
                stability.append({"k": k, "comparison": "latent_vs_input_pca" if comparison == "input_pca"
                                  else "mixture_refits", "left": reference, "right": other,
                                  "ari": adjusted_rand_between(main, values)})
            contingency.append(segment_contingency(main, values, k).assign(k=k, comparison=comparison, other=other))
        for repetition in repetitions:
            if repetition != reference:
                stability.append({"k": k, "comparison": "latent_vs_input_pca", "left": repetition, "right": -1,
                                  "ari": adjusted_rand_between(result["labels"][("latent", repetition, k)],
                                                               result["labels"][("input_pca", -1, k)])})
    ## Class overlap and exploded image x m/z layers for the selected classes
    _, selection = _class_selection(cache, axis)
    metaspace = np.load(cache.population_root / "heldout_metaspace.npy", mmap_mode="r")
    observed = np.asarray(cache.axis_array(axis, "heldout_ion_input"))
    probabilities = np.asarray(cache.model_array(representative.model_id, "heldout_image_probabilities"),
                               dtype=np.float32)  # (N_h, C)
    selected_k = result["selected_k"]
    segment = result["labels"][("latent", reference, selected_k)]
    overlap, layers = [], []
    for choice in selection.itertuples():
        rows = pixels.dataset_id.eq(choice.dataset_id).to_numpy()
        values = np.nan_to_num(np.asarray(metaspace[rows, choice.ion_index], dtype=np.float64))
        segments = segment[rows]
        for value in np.unique(segments):
            inside = segments == value
            overlap.append({"criterion": choice.criterion, "kind": choice.kind, "rank": choice.rank,
                            "dataset_id": choice.dataset_id, "class_name": choice.class_name, "segment": int(value),
                            "segment_pixels": int(inside.sum()),
                            "metaspace_positive_fraction": float((values[inside] > 0).mean()),
                            "metaspace_mean": float(values[inside].mean()),
                            "probability_mean": float(probabilities[rows, choice.head_column][inside].mean())})
        layers.append(pixels.loc[rows, ["row", "dataset_id", "x", "y"]].assign(
            criterion=choice.criterion, kind=choice.kind, rank=choice.rank, class_name=choice.class_name,
            mz=choice.mz, metaspace=values, input_ion=observed[rows, choice.ion_index],
            probability=probabilities[rows, choice.head_column], segment=segments))
    return {"bic": result["bic"], "labels": labels, "rgb": rgb, "markers": pd.concat(markers, ignore_index=True),
            "composition": pd.concat(composition, ignore_index=True), "stability": pd.DataFrame(stability),
            "contingency": pd.concat(contingency, ignore_index=True), "class_selection": selection,
            "class_overlap": pd.DataFrame(overlap),
            "exploded_layers": pd.concat(layers, ignore_index=True) if layers else pd.DataFrame(),
            "representative_model": _representative_table(cache, [axis]),
            "metadata": {"selected_k": selected_k, "k_values": result["k_values"], "reference_repetition": reference,
                         "representative_model": representative.model_id,
                         "resamples": int(settings.get("resamples", 5))}}


# --------------------------------------------------
# Section: part 1 — prediction
# --------------------------------------------------

def _class_frame(cache: CampaignCache, axis: str) -> pd.DataFrame:
    """Class catalogue with the stratum labels used by the inference level."""
    ions = cache.axis_table(axis, "heldout_ions")
    return class_strata(cache.axis_table(axis, "classes"), ions.class_name, float(cache.settings["windows"]["width"]))


def _stratum_metrics(cache: CampaignCache, models: pd.DataFrame, per_class: pd.DataFrame, grouping: str) -> pd.DataFrame:
    """Complete metric set per class stratum, with class and positive counts.

    Metrics come from the inference level, which runs the ranking evaluation on the
    columns of every stratum (macro and micro AP, ROC AUC, AP above prevalence,
    precision, recall, F1, micro prevalence and Hamming loss, training-supported
    classes). The ``frequency`` grouping (rare / medium / frequent) is the class scope
    of the whole-axis ranking table itself. Pooled confusion counts and class counts
    are summed from ``per_class``, which must carry the ``grouping`` column.

    :return: One row per model, data population, image group, ranking population and
        stratum, with one column per metric.
    :rtype: pandas.DataFrame
    """
    keys = [*IDENTITY, "evaluation", "group", "population", "stratum"]
    if grouping == "frequency":
        source = cache.collect("prediction", models)
        source = source[source.scope.isin(["rare", "medium", "frequent"])].assign(stratum=lambda frame: frame.scope)
    else:
        source = cache.collect("strata_prediction", models)
        source = source[source.grouping == grouping]
    metrics = source.pivot_table(index=keys, columns="metric", values="value", dropna=False).reset_index()
    metrics.columns.name = None
    classes = per_class.rename(columns={grouping: "stratum"})
    counts = classes.groupby(keys).agg(classes=("class_name", "size"), eligible_classes=("eligible", "sum"),
                                       positives=("positives", "sum")).reset_index()
    pooled = classes[classes.eligible.astype(bool)].groupby(keys)[
        ["true_positive", "false_positive", "false_negative"]].sum(min_count=1).reset_index()
    return metrics.merge(counts, on=keys, how="left").merge(pooled, on=keys, how="left").assign(grouping=grouping)


@register("prediction_global", per_axis=True)
def prediction_global(cache: CampaignCache, axis: str) -> dict:
    """Head metrics over the whole axis per population, image and repetition.

    Tables: ``prediction`` (aggregate metrics), ``image_prediction`` (per held-out image),
    ``per_class`` (whole populations and, with ``group`` = image, per held-out image),
    ``score_diagnostics``, ``gaps`` and ``representative_model``.
    """
    models = cache.axis_models(axis)
    prediction = cache.collect("prediction", models)
    per_class = cache.collect("per_class", models)
    diagnostics = cache.collect("score_diagnostics", models)
    overall = prediction[prediction.group == "all"]
    means = overall.pivot_table(index=[*IDENTITY, "population", "scope", "metric"], columns="evaluation",
                                values="value").reset_index()
    gaps = means.assign(test_minus_train=means.test - means.train,
                        heldout_minus_test=means.heldout_image - means.test,
                        combined_minus_test=means.test_combined - means.test)
    return {"prediction": overall, "image_prediction": prediction[prediction.group != "all"],
            "per_class": per_class, "score_diagnostics": diagnostics, "gaps": gaps,
            "representative_model": _representative_table(cache, [axis]),
            "metadata": {"evaluations": list(PREDICTION_POPULATIONS)}}


@register("prediction_local", per_axis=True)
def prediction_local(cache: CampaignCache, axis: str) -> dict:
    """Head metrics along the m/z axis (100 Da class windows).

    Tables: ``window_metrics`` (model x evaluation x ranking population x window, complete
    metric set) and ``class_positions`` (per-class AP with its m/z, for the AP-versus-m/z view).
    """
    classes = _class_frame(cache, axis)
    models = cache.axis_models(axis)
    per_class = cache.collect("per_class", models)
    per_class = per_class[per_class.group == "all"].merge(
        classes[["class_name", "mz", "mz_window", "mz_window_lower", "class_set"]], on="class_name", how="left")
    windows = _stratum_metrics(cache, models, per_class, "mz_window")
    windows = windows.merge(classes[["mz_window", "mz_window_lower"]].drop_duplicates(), left_on="stratum",
                            right_on="mz_window", how="left").drop(columns="mz_window")
    positions = per_class[[*IDENTITY, "evaluation", "population", "class_name", "mz", "mz_window", "class_set",
                           "train_positives", "positives", "prevalence", "eligible", "average_precision", "roc_auc",
                           "ap_above_prevalence"]]
    return {"window_metrics": windows, "class_positions": positions,
            "metadata": {"window_width": cache.settings["windows"]["width"]}}


@register("prediction_class_strata", per_axis=True)
def prediction_class_strata(cache: CampaignCache, axis: str) -> dict:
    """Head metrics by class stratum and the extension-range diagnostics.

    Strata: ``train_support`` (training positives), ``frequency`` (rare / medium /
    frequent, as in the ranking tables), ``class_set`` (common / boundary / extension;
    also per held-out image) and ``heldout_presence``. Every stratum carries the complete
    metric set of the whole-axis table. Tables: ``strata_metrics``, ``extension_diagnostics``
    (per class: predicted-positive rate, mean probabilities of positives and of the
    other entries, acquisition fraction of the class window), ``score_by_class_set``
    and ``unseen_heldout_classes``.
    """
    classes = _class_frame(cache, axis)
    models = cache.axis_models(axis)
    per_class = cache.collect("per_class", models).merge(
        classes[["class_name", "class_set", "train_support", "heldout_presence"]], on="class_name", how="left")
    strata = pd.concat([_stratum_metrics(cache, models, per_class, grouping)
                        for grouping in ("train_support", "frequency", "class_set", "heldout_presence")],
                       ignore_index=True)
    ## Extension diagnostics on the evaluation pixels (test + extension, held-out)
    minimum = float(cache.settings["windows"].get("minimum_mass", 1e-3))
    windows = _windows(cache, axis)
    window_of_class = np.searchsorted(windows.window_upper.to_numpy(), classes.mz.fillna(classes.bin_mz).to_numpy(),
                                      side="right")
    window_of_class = np.clip(window_of_class, 0, len(windows) - 1)  # (C,)
    diagnostics, scores = [], []
    test = cache.population(axis, "test")
    extended = cache.population(axis, "test_extended")
    visible = np.asarray(extended["visible"], dtype=bool)
    for row in models.itertuples():
        evaluations = {
            "test_combined": (
                _probabilities(np.concatenate([np.asarray(cache.model_array(row.model_id, "test_logits")),
                                               np.asarray(cache.model_array(row.model_id, "test_extended_logits"))[visible]])),
                np.concatenate([_available_targets(test), _available_targets(extended)[visible]]),
                np.concatenate([cache.model_pixels(row.model_id, "test")["input_mass"],
                                cache.model_pixels(row.model_id, "test_extended")["input_mass"][visible]])),
            "heldout_image": (
                np.asarray(cache.model_array(row.model_id, "heldout_image_probabilities"), dtype=np.float64),
                _available_targets(cache.population(axis, "heldout_image")),
                cache.model_pixels(row.model_id, "heldout_image")["input_mass"]),
        }
        for evaluation, (probabilities, targets, input_mass) in evaluations.items():
            acquired = input_mass[:, window_of_class] >= minimum  # (N, C)
            predicted = probabilities >= 0.5
            positives = targets.sum(axis=0)
            with np.errstate(all="ignore"):
                diagnostics.append(pd.DataFrame({
                    **_identity(row), "evaluation": evaluation, "class_name": classes.class_name,
                    "class_set": classes.class_set, "mz": classes.mz, "train_positives": classes.train_positives,
                    "positives": positives, "pixels": len(targets),
                    "predicted_positive_rate": predicted.mean(axis=0),
                    "mean_probability_positive": np.where(positives > 0, (probabilities * targets).sum(axis=0)
                                                          / np.maximum(positives, 1), np.nan),
                    "mean_probability_other": (probabilities * ~targets).sum(axis=0)
                                              / np.maximum((~targets).sum(axis=0), 1),
                    "acquired_fraction": acquired.mean(axis=0),
                    "recall_at_half": np.where(positives > 0, (predicted & targets).sum(axis=0)
                                               / np.maximum(positives, 1), np.nan)}))
            for class_set in classes.class_set.dropna().unique():
                columns = np.flatnonzero(classes.class_set.to_numpy() == class_set)
                for state, selected in (("positive", targets[:, columns]), ("other", ~targets[:, columns])):
                    values = probabilities[:, columns][selected]
                    if values.size:
                        scores.append({**_identity(row), "evaluation": evaluation, "class_set": class_set,
                                       "state": state, "entries": int(values.size),
                                       **{f"q{int(level * 100):02d}": float(np.quantile(values, level))
                                          for level in (0.1, 0.25, 0.5, 0.75, 0.9)}})
    ions = cache.axis_table(axis, "heldout_ions")
    unseen = ions[ions.head_column < 0].drop(columns="bins")
    return {"strata_metrics": strata, "extension_diagnostics": pd.concat(diagnostics, ignore_index=True),
            "score_by_class_set": pd.DataFrame(scores), "unseen_heldout_classes": unseen,
            "metadata": {"support_edges": [str(value) for value in SUPPORT_EDGES], "minimum_window_mass": minimum}}


@register("prediction_ion_images", per_axis=True)
def prediction_ion_images(cache: CampaignCache, axis: str) -> dict:
    """Head probability maps against METASPACE ion images of the held-out images.

    METASPACE ion images are intensities; the annotation target of a pixel is
    ``intensity > 0``, so a weak but non-zero pixel is as positive as a strong one.
    Tables: ``class_agreement`` (model x image x class), ``image_summary``,
    ``class_selection`` (both criteria, representative model) and ``ion_pixels``
    (selected classes, representative model: METASPACE intensity, binary target, input
    ion, reconstructed ion and probability), and ``representative_model``.
    """
    agreement, selection = _class_selection(cache, axis)
    representative = cache.representative(axis)
    pixels = _heldout_images(cache)
    metaspace = np.load(cache.population_root / "heldout_metaspace.npy", mmap_mode="r")
    observed = np.asarray(cache.axis_array(axis, "heldout_ion_input"))  # (N_h, I)
    probabilities = np.asarray(cache.model_array(representative.model_id, "heldout_image_probabilities"),
                               dtype=np.float32)  # (N_h, C)
    reconstructed = np.asarray(cache.model_array(representative.model_id, "heldout_image_ions"))  # (N_h, I)
    targets = _available_targets(cache.population(axis, "heldout_image"))
    layers = []
    for choice in selection.drop_duplicates(["dataset_id", "class_name"]).itertuples():
        rows = pixels.dataset_id.eq(choice.dataset_id).to_numpy()
        layers.append(pixels.loc[rows, ["row", "dataset_id", "x", "y"]].assign(
            class_name=choice.class_name, mz=choice.mz, class_set=choice.class_set,
            metaspace=np.asarray(metaspace[rows, choice.ion_index]), target=targets[rows, choice.head_column],
            input_ion=observed[rows, choice.ion_index], output_ion=reconstructed[rows, choice.ion_index],
            probability=probabilities[rows, choice.head_column]))
    summary = agreement.groupby([*IDENTITY, "dataset_id"])[["metaspace_spearman", "presence_auc",
                                                            "average_precision"]].median().reset_index()
    return {"class_agreement": agreement, "image_summary": summary, "class_selection": selection,
            "ion_pixels": pd.concat(layers, ignore_index=True) if layers else pd.DataFrame(),
            "representative_model": _representative_table(cache, [axis]),
            "metadata": {"selection": cache.settings.get("class_selection", {}),
                         "representative_model": representative.model_id}}


# --------------------------------------------------
# Section: part 1 — axis comparison
# --------------------------------------------------

def _paired_models(cache: CampaignCache) -> list[tuple[Any, Any]]:
    """Models of the first and last axis paired by repetition index."""
    first, last = cache.axes[0], cache.axes[-1]
    left = {int(row.repetition): row for row in cache.axis_models(first).itertuples()}
    right = {int(row.repetition): row for row in cache.axis_models(last).itertuples()}
    return [(left[repetition], right[repetition]) for repetition in sorted(set(left) & set(right))]


@register("axis_comparison_reconstruction", per_axis=False)
def axis_comparison_reconstruction(cache: CampaignCache, axis: Optional[str]) -> dict:
    """Reconstruction on the common range and on the extension of the wide axis.

    Both axes are evaluated on the same pixels (identical source identifiers), so every
    comparison is paired per pixel. Global Masserstein values are not comparable across
    axes because the wide axis transports over a longer m/z range; the comparison
    therefore uses the windows shared by both axes. Tables: ``common_windows``
    (per-pixel paired differences of window quantities), ``extension_windows`` (wide
    axis: input versus output mass and the fraction of pixels whose reconstruction
    leaves a window with signal empty), ``extension_images`` and ``image_maps``.
    """
    narrow, wide = cache.axes[0], cache.axes[-1]
    narrow_windows, wide_windows = _windows(cache, narrow), _windows(cache, wide)
    common = narrow_windows.merge(wide_windows, on="window_label", suffixes=("_narrow", "_wide"))
    reference = tuple(cache.settings.get("reference_mass_range", (200.0, 900.0)))
    extension = wide_windows[(wide_windows.window_upper <= reference[0]) | (wide_windows.window_lower >= reference[1])]
    minimum = float(cache.settings["windows"].get("minimum_mass", 1e-3))
    empty = 0.1 * minimum
    paired, extension_rows, extension_images = [], [], []
    for population in RECONSTRUCTION_POPULATIONS:
        datasets = _rows(cache, population).dataset_id.to_numpy()
        for left, right in _paired_models(cache):
            a = cache.model_pixels(left.model_id, population)
            b = cache.model_pixels(right.model_id, population)
            if not np.array_equal(a["source_ids"], b["source_ids"]):
                raise ValueError(f"Population {population} is not paired across axes.")
            for window in common.itertuples():
                for quantity in ("within", "contribution", "output_mass", "input_mass"):
                    x, y = a[quantity][:, window.window_narrow], b[quantity][:, window.window_wide]
                    difference = y - x
                    finite = np.isfinite(difference)
                    paired.append({"population": population, "repetition": int(left.repetition),
                                   "window_label": window.window_label, "quantity": quantity,
                                   "narrow_mean": float(np.nanmean(x)), "wide_mean": float(np.nanmean(y)),
                                   "median_difference": float(np.median(difference[finite])) if finite.any() else np.nan,
                                   "wide_larger_fraction": float((difference[finite] > 0).mean()) if finite.any() else np.nan,
                                   "paired_pixels": int(finite.sum())})
            for window in extension.itertuples():
                input_mass, output_mass = b["input_mass"][:, window.window], b["output_mass"][:, window.window]
                with_signal = input_mass >= minimum
                extension_rows.append({"population": population, "repetition": int(right.repetition),
                                       "window_label": window.window_label, "window_lower": window.window_lower,
                                       "mean_input_mass": float(input_mass.mean()),
                                       "mean_output_mass": float(output_mass.mean()),
                                       "pixels_with_signal": int(with_signal.sum()),
                                       "emptied_fraction": float((output_mass[with_signal] < empty).mean())
                                       if with_signal.any() else np.nan})
                for dataset_id in np.unique(datasets):
                    rows = datasets == dataset_id
                    extension_images.append({"population": population, "repetition": int(right.repetition),
                                             "window_label": window.window_label, "dataset_id": dataset_id,
                                             "mean_input_mass": float(input_mass[rows].mean()),
                                             "mean_output_mass": float(output_mass[rows].mean())})
    ## Held-out maps of the representative model of each axis on identical pixels
    pixels = _heldout_images(cache)
    maps = pixels[["row", "dataset_id", "x", "y"]].copy()
    for axis_name, label, windows in ((narrow, "narrow", common.window_narrow), (wide, "wide", common.window_wide)):
        values = cache.model_pixels(cache.representative(axis_name).model_id, "heldout_image")
        columns = windows.to_numpy()
        maps[f"masserstein_{label}"] = values["masserstein"]
        maps[f"common_contribution_{label}"] = values["contribution"][:, columns].sum(axis=1)
        with np.errstate(all="ignore"):
            maps[f"common_within_{label}"] = np.nanmean(values["within"][:, columns], axis=1)
        if label == "wide" and len(extension):
            columns = extension.window.to_numpy()
            maps["extension_contribution_wide"] = values["contribution"][:, columns].sum(axis=1)
            maps["extension_input_mass_wide"] = values["input_mass"][:, columns].sum(axis=1)
            maps["extension_output_mass_wide"] = values["output_mass"][:, columns].sum(axis=1)
    return {"common_windows": pd.DataFrame(paired), "extension_windows": pd.DataFrame(extension_rows),
            "extension_images": pd.DataFrame(extension_images), "image_maps": maps,
            "representative_model": _representative_table(cache, [narrow, wide]),
            "metadata": {"narrow_axis": narrow, "wide_axis": wide, "reference_mass_range": list(reference),
                         "signal_mass": minimum, "empty_mass": empty}}


@register("axis_comparison_latent_capacity", per_axis=False)
def axis_comparison_latent_capacity(cache: CampaignCache, axis: Optional[str]) -> dict:
    """Latent capacity of both axes at the same latent dimension.

    Tables: ``capacity`` (latent dimension usage per axis, repetition and population),
    ``data_geometry`` (dimension of the inputs per axis), ``eigenvalues``,
    ``cross_axis_similarity`` (all pairs of models, within and across axes, on identical
    pixels), ``segmentation_agreement`` (ARI between axes per paired repetition and
    between the representative models, per k), ``side_by_side_segments`` and
    ``segment_contingency`` (representative partitions, wide aligned to narrow, every k)
    and ``representative_model``.
    """
    capacity, eigenvalues, data = [], [], []
    for axis_name in cache.axes:
        geometry, eigen = _latent_summaries(cache, axis_name)
        capacity.append(geometry)
        eigenvalues.append(eigen)
        data.append(_data_geometry(cache, axis_name)[0])
    models = list(cache.models.itertuples())
    similarity = _reproducibility(cache, list(combinations(models, 2)), ("test", "heldout_image"))
    similarity["pair_kind"] = np.where(similarity.left_axis == similarity.right_axis, "within_axis", "across_axes")
    segmentations = {axis_name: _segmentations(cache, axis_name, refits=False) for axis_name in cache.axes}
    narrow, wide = cache.axes[0], cache.axes[-1]
    agreement = []
    for k in segmentations[narrow]["k_values"]:
        for left, right in _paired_models(cache):
            agreement.append({"k": k, "repetition": int(left.repetition), "ari": adjusted_rand_between(
                segmentations[narrow]["labels"][("latent", int(left.repetition), k)],
                segmentations[wide]["labels"][("latent", int(right.repetition), k)])})
    ## Representative partitions of both axes, wide aligned to narrow, for every k
    pixels = _heldout_images(cache)
    side_by_side, contingency = [], []
    for k in segmentations[narrow]["k_values"]:
        narrow_labels = segmentations[narrow]["labels"][("latent", segmentations[narrow]["reference_repetition"], k)]
        wide_labels = align_labels(narrow_labels, segmentations[wide]["labels"][
            ("latent", segmentations[wide]["reference_repetition"], k)], k)
        agreement.append({"k": k, "repetition": -1, "comparison": "representatives",
                          "ari": adjusted_rand_between(narrow_labels, wide_labels)})
        side_by_side.append(pixels[["row", "dataset_id", "x", "y"]].assign(k=k, segment_narrow=narrow_labels,
                                                                         segment_wide=wide_labels))
        contingency.append(segment_contingency(narrow_labels, wide_labels, k).assign(k=k))
    agreement = pd.DataFrame(agreement)
    agreement["comparison"] = agreement.comparison.fillna("paired_repetitions")
    return {"capacity": pd.concat(capacity, ignore_index=True), "data_geometry": pd.concat(data, ignore_index=True),
            "eigenvalues": pd.concat(eigenvalues, ignore_index=True), "cross_axis_similarity": similarity,
            "segmentation_agreement": agreement, "side_by_side_segments": pd.concat(side_by_side, ignore_index=True),
            "segment_contingency": pd.concat(contingency, ignore_index=True),
            "representative_model": _representative_table(cache, [narrow, wide]),
            "metadata": {"narrow_axis": narrow, "wide_axis": wide, "selected_k": segmentations[narrow]["selected_k"],
                         "segmentation_selected_k": {name: value["selected_k"] for name, value in segmentations.items()}}}


@register("axis_comparison_prediction", per_axis=False)
def axis_comparison_prediction(cache: CampaignCache, axis: Optional[str]) -> dict:
    """Head metrics of both axes on common classes, and the wide axis on its extra classes.

    Tables: ``common_class_pairs`` (per common class and evaluation: AP of both axes per
    repetition), ``class_set_summary`` (macro metrics per axis and class set),
    ``agreement_pairs`` (per image and common class: METASPACE Spearman of both axes)
    and ``ion_image_pairs`` (pixel maps of selected common classes, representative model of
    each axis, with the binary annotation target), ``representative_model``.
    """
    narrow, wide = cache.axes[0], cache.axes[-1]
    frames, labelled = {}, {}
    for axis_name in cache.axes:
        per_class = cache.collect("per_class", cache.axis_models(axis_name))
        labelled[axis_name] = per_class.merge(_class_frame(cache, axis_name)[["class_name", "class_set"]],
                                              on="class_name", how="left")
        frames[axis_name] = labelled[axis_name][labelled[axis_name].group == "all"]
    keys = ["evaluation", "population", "class_name", "repetition"]
    columns = [*keys, "average_precision", "roc_auc", "eligible", "positives", "train_positives"]
    common = frames[narrow][columns].merge(frames[wide][columns + []], on=keys, suffixes=("_narrow", "_wide"))
    common["ap_difference"] = common.average_precision_wide - common.average_precision_narrow
    summary = pd.concat([_stratum_metrics(cache, cache.axis_models(axis_name), frame, "class_set")
                         for axis_name, frame in labelled.items()], ignore_index=True)
    ## Spatial agreement with METASPACE on common classes present in both heads
    agreements = {axis_name: _class_selection(cache, axis_name) for axis_name in cache.axes}
    left = agreements[narrow][0].groupby(["dataset_id", "class_name", "repetition"]).metaspace_spearman.mean()
    right = agreements[wide][0].groupby(["dataset_id", "class_name", "repetition"]).metaspace_spearman.mean()
    pairs = pd.concat([left.rename("spearman_narrow"), right.rename("spearman_wide")], axis=1, join="inner").reset_index()
    ## Pixel maps of the narrow-axis selections for both axes
    pixels = _heldout_images(cache)
    metaspace = np.load(cache.population_root / "heldout_metaspace.npy", mmap_mode="r")
    maps = []
    heads = {axis_name: cache.axis_table(axis_name, "classes").set_index("class_name").class_index
             for axis_name in cache.axes}
    shown = {axis_name: np.asarray(cache.model_array(cache.representative(axis_name).model_id,
                                                     "heldout_image_probabilities"), dtype=np.float32)
             for axis_name in cache.axes}  # (N_h, C_axis)
    targets = _available_targets(cache.population(narrow, "heldout_image"))  # (N_h, C_narrow)
    for choice in agreements[narrow][1].drop_duplicates(["dataset_id", "class_name"]).itertuples():
        if choice.class_name not in heads[wide].index:
            continue
        rows = pixels.dataset_id.eq(choice.dataset_id).to_numpy()
        maps.append(pixels.loc[rows, ["row", "dataset_id", "x", "y"]].assign(
            class_name=choice.class_name, mz=choice.mz, metaspace=np.asarray(metaspace[rows, choice.ion_index]),
            target=targets[rows, heads[narrow][choice.class_name]],
            probability_narrow=shown[narrow][rows, heads[narrow][choice.class_name]],
            probability_wide=shown[wide][rows, heads[wide][choice.class_name]]))
    return {"common_class_pairs": common, "class_set_summary": summary, "agreement_pairs": pairs,
            "ion_image_pairs": pd.concat(maps, ignore_index=True) if maps else pd.DataFrame(),
            "representative_model": _representative_table(cache, [narrow, wide]),
            "metadata": {"narrow_axis": narrow, "wide_axis": wide}}


@register("error_localization", per_axis=False)
def error_localization(cache: CampaignCache, axis: Optional[str]) -> dict:
    """Where errors occur: m/z window, image, input intensity, image border and class support.

    Tables: ``window_errors`` (axis x population x image x window x repetition),
    ``tic_dependence`` (Masserstein by raw-TIC decile), ``edge_dependence``
    (held-out Masserstein by distance to the tissue border), ``class_errors`` (per
    class: AP next to the reconstruction cost of the class window) and ``worst_windows``
    (largest held-out minus test window gaps).
    """
    windows_rows, tic_rows, edge_rows, class_rows = [], [], [], []
    pixels = _heldout_images(cache)
    ## Distance of every held-out pixel to the border of its image
    distance = np.zeros(len(pixels))
    for dataset_id, frame in pixels.groupby("dataset_id"):
        extent = image_extent(frame.x.to_numpy(), frame.y.to_numpy())
        mask = np.isfinite(assemble_image(frame.x.to_numpy(), frame.y.to_numpy(), np.ones(len(frame)), extent))
        edt = distance_transform_edt(np.pad(mask, 1))[1:-1, 1:-1]  # (H, W)
        distance[frame.index] = edt[frame.y.to_numpy() - extent[1], frame.x.to_numpy() - extent[0]]
    edge_bins = pd.cut(distance, [0, 1.5, 2.5, 5.5, 10.5, np.inf], labels=["1", "2", "3-5", "6-10", ">10"])
    for axis_name in cache.axes:
        windows = _windows(cache, axis_name)
        classes = _class_frame(cache, axis_name)
        for row in cache.axis_models(axis_name).itertuples():
            identity = _identity(row)
            for population in ("train", "test", "heldout_image"):
                arrays = cache.model_pixels(row.model_id, population)
                datasets = _rows(cache, population).dataset_id.to_numpy()
                groups = ["all", *np.unique(datasets)] if population == "heldout_image" else ["all"]
                for group in groups:
                    selected = np.ones(len(datasets), dtype=bool) if group == "all" else datasets == group
                    with np.errstate(all="ignore"):
                        windows_rows.append(windows.assign(**identity, population=population, group=group,
                                                           contribution=arrays["contribution"][selected].mean(axis=0),
                                                           within=np.nanmean(arrays["within"][selected], axis=0),
                                                           imbalance=(arrays["output_mass"] - arrays["input_mass"])[
                                                               selected].mean(axis=0)))
                tic = np.asarray(cache.population(axis_name, population)["tic"])
                deciles = pd.qcut(tic, 10, labels=False, duplicates="drop")
                frame = pd.DataFrame({"decile": deciles, "tic": tic, "masserstein": arrays["masserstein"],
                                      "cosine_similarity": arrays["cosine_similarity"]})
                tic_rows.append(frame.groupby("decile").agg(tic_median=("tic", "median"),
                                                            masserstein_mean=("masserstein", "mean"),
                                                            cosine_mean=("cosine_similarity", "mean"),
                                                            pixels=("tic", "size")).reset_index().assign(
                    **identity, population=population))
                if population == "heldout_image":
                    frame = pd.DataFrame({"edge_distance": edge_bins, "dataset_id": datasets,
                                          "masserstein": arrays["masserstein"]})
                    edge_rows.append(frame.groupby(["dataset_id", "edge_distance"], observed=True).masserstein.agg(
                        ["mean", "median", "count"]).reset_index().assign(**identity))
            ## Class-level: AP next to the within-window cost of the class window (test)
            within = np.nanmean(cache.model_pixels(row.model_id, "test")["within"], axis=0)  # (W,)
            per_class = cache.model_table(row.model_id, "per_class")
            per_class = per_class[(per_class.group == "all") & (per_class.evaluation == "test_combined")
                                  & (per_class.population == "annotation_retrieval")]
            merged = per_class[["class_name", "average_precision", "eligible", "positives"]].merge(
                classes[["class_name", "mz", "mz_window_lower", "class_set", "train_positives"]], on="class_name")
            index = np.clip(np.searchsorted(windows.window_upper.to_numpy(), merged.mz.fillna(merged.mz_window_lower),
                                            side="right"), 0, len(windows) - 1)
            class_rows.append(merged.assign(**identity, window_within_test=within[index]))
    windows_frame = pd.concat(windows_rows, ignore_index=True)
    means = windows_frame[windows_frame.group == "all"].pivot_table(
        index=["axis", "window_label"], columns="population", values="within").reset_index()
    means["heldout_minus_test"] = means.heldout_image - means.test
    worst = means.sort_values("heldout_minus_test", ascending=False)
    return {"window_errors": windows_frame, "tic_dependence": pd.concat(tic_rows, ignore_index=True),
            "edge_dependence": pd.concat(edge_rows, ignore_index=True), "class_errors": pd.concat(class_rows, ignore_index=True),
            "worst_windows": worst, "metadata": {"edge_bins": ["1", "2", "3-5", "6-10", ">10"]}}


@register("decision_summary", per_axis=False)
def decision_summary(cache: CampaignCache, axis: Optional[str]) -> dict:
    """Compact set of the most informative metrics per axis and repetition.

    Tables: ``decision_metrics`` (long form: family, metric, population, value per model)
    and ``decision_table`` (mean, SD and count across repetitions).
    """
    rows = []
    reference = tuple(cache.settings.get("reference_mass_range", (200.0, 900.0)))
    minimum = float(cache.settings["windows"].get("minimum_mass", 1e-3))
    for axis_name in cache.axes:
        windows = _windows(cache, axis_name)
        common = ((windows.window_lower >= reference[0]) & (windows.window_upper <= reference[1])).to_numpy()
        geometry, _ = _latent_summaries(cache, axis_name)
        agreement, _ = _class_selection(cache, axis_name)
        classes = _class_frame(cache, axis_name)
        for row in cache.axis_models(axis_name).itertuples():
            identity = _identity(row)

            def add(family: str, metric: str, population: str, value: float) -> None:
                rows.append({**identity, "family": family, "metric": metric, "population": population,
                             "value": float(value)})

            ## Reconstruction
            means = {}
            for population in RECONSTRUCTION_POPULATIONS:
                arrays = cache.model_pixels(row.model_id, population)
                means[population] = float(arrays["masserstein"].mean())
                add("reconstruction", "masserstein_mean", population, means[population])
                with np.errstate(all="ignore"):
                    add("reconstruction", "common_window_within_mean", population,
                        np.nanmean(arrays["within"][:, common]))
                if (~common).any():
                    with_signal = arrays["input_mass"][:, ~common] >= minimum
                    emptied = arrays["output_mass"][:, ~common][with_signal] < 0.1 * minimum
                    add("reconstruction", "extension_emptied_fraction", population,
                        emptied.mean() if emptied.size else np.nan)
            add("reconstruction", "masserstein_heldout_minus_test", "heldout_image",
                means["heldout_image"] - means["test"])
            ## Latent
            frame = geometry[(geometry.model_id == row.model_id) & (geometry.space == "u")]
            for population in ("test", "heldout_image"):
                values = frame[frame.population == population].set_index("metric").value
                for metric in ("effective_rank", "participation_ratio", "two_nn_intrinsic_dimension"):
                    add("latent", metric, population, values.get(metric, np.nan))
                add("latent", "cos_theta_sd_over_uniform", population,
                    values.get("observed_sd_cos_theta", np.nan) / values.get("uniform_baseline_sd_cos_theta", np.nan))
            sensitivity = cache.model_table(row.model_id, "sensitivity")
            for population, frame_s in sensitivity.groupby("population"):
                at = frame_s.iloc[(frame_s.epsilon - 0.01).abs().argsort()[:1]]
                add("latent", "sensitivity_angle_at_0.01", population, at.mean_angle_degrees.iloc[0])
            ## Prediction
            prediction = cache.model_table(row.model_id, "prediction")
            prediction = prediction[(prediction.group == "all") & (prediction.scope == "train_supported")]
            for evaluation in ("test_combined", "heldout_image"):
                for population in ("annotation_retrieval", "operational_pn"):
                    selected = prediction[(prediction.evaluation == evaluation) & (prediction.population == population)]
                    values = selected.set_index("metric").value
                    for metric in ("average_precision", "micro_average_precision"):
                        add("prediction", f"{population}_{metric}", evaluation, values.get(metric, np.nan))
            per_class = cache.model_table(row.model_id, "per_class")
            per_class = per_class[(per_class.group == "all") & (per_class.evaluation == "test_combined")
                                  & (per_class.population == "annotation_retrieval") & per_class.eligible.astype(bool)]
            per_class = per_class.merge(classes[["class_name", "class_set"]], on="class_name", how="left")
            for class_set, frame_c in per_class.groupby("class_set"):
                add("prediction", f"macro_ap_{class_set}_classes", "test_combined", frame_c.average_precision.mean())
            add("prediction", "metaspace_spearman_median", "heldout_image",
                agreement[agreement.model_id == row.model_id].metaspace_spearman.median())
    metrics = pd.DataFrame(rows)
    table = metrics.groupby(["axis", "display_label", "family", "metric", "population"]).value.agg(
        ["mean", "std", "count"]).reset_index()
    return {"decision_metrics": metrics, "decision_table": table, "metadata": {}}
