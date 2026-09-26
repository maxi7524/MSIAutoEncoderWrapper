"""Registered analyses comparing the pretraining variants and stages with the baselines.

Main thesis of these parts: synthetic pretraining improves on the model trained on real
data only. Every table therefore carries the baseline and every comparison is a paired
difference to the baseline of the same axis and repetition (see
:mod:`.pretraining_campaign_comparison` for the sign convention and the summaries).

Parts and analyses (settings keys):

* part 0 — ``synthetic_pretraining_curves``, ``finetuning_curves``;
* parts 2 and 3, once per axis and stage (``pretrained`` in part 2, ``frozen_head`` and
  ``unfrozen_head`` in part 3) — ``stage_overview``, ``stage_reconstruction``,
  ``stage_prediction``, ``stage_latent``, ``stage_segmentation``, ``stage_mechanisms``;
* part 3 stage comparison, per axis — ``stage_transitions``, ``stage_representation_change``,
  ``stage_images``;
* part 4, per axis — ``rare_classes``, ``overlap_classes``;
* part 5 — ``axis_reconstruction_common``, ``axis_prediction_common``, ``axis_benefit``,
  ``axis_latent_capacity``;
* part 6 — ``thesis_verdict``.

Symbols: ``N`` pixels, ``M`` bins, ``W`` m/z windows, ``C`` head classes, ``D`` latent
dimension, ``P`` class pairs.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.metrics import label_ranking_average_precision_score

from ....utils.logger import get_custom_logger
from ..heads.predictive_comparison import ranking_tables
from ..latent.predictive_geometry import geometry_tables, intrinsic_dimension_estimate, pairwise_geometry_battery
from ..spatial import adjusted_rand_between, align_labels, fit_segmentation, segment_contingency
from . import pretraining_campaign as campaign
from .predictive_precompute import resolve_device
from .pretraining_campaign_comparison import (
    METRIC_KEYS,
    NO_RANKING,
    angle_between,
    calibration_table,
    factorial_effects,
    logits_from_probabilities,
    metric_direction,
    paired_improvements,
    pairwise_discrimination,
    rank_variants,
    ridge_probe,
    summarize,
    true_label_ranks,
    variant_contrasts,
)
from .pretraining_campaign_precompute import CampaignCache, axis_directory, register
from .pretraining_campaign_reports import (
    _available_targets,
    _heldout_images,
    _history,
    _latent_codes,
    _latent_samples,
    _rows,
    _sample_rows,
    _windows,
    latent_model_summary,
)

logger = get_custom_logger(__name__)

IDENTITY = ("model_id", "model_alias", "display_label", "axis", "repetition", "variant", "stage", "lineage")
RECONSTRUCTION_POPULATIONS = ("train", "test", "test_extended", "heldout_image")
PIXEL_SUMMARY_METRICS = ("masserstein", "spectral_angle", "cosine_similarity", "mse")
PREDICTION_EVALUATIONS = ("test", "test_extended", "test_combined", "heldout_image")
PREDICTION_METRICS = ("average_precision", "micro_average_precision", "roc_auc", "ap_above_prevalence",
                      "precision", "recall", "f1", "micro_f1", "hamming_loss")
#: Identity columns of the loss-history tables (no lineage there).
HISTORY_IDENTITY = IDENTITY[:-1]
UNIT = ("axis", "repetition")
GROUP = ("axis", "variant", "stage")


# --------------------------------------------------
# Section: identities, settings and model sets
# --------------------------------------------------

def _identity(row: Any) -> dict:
    """Identity columns of one catalog record (``stage`` is the workflow role)."""
    return {"model_id": row.model_id, "model_alias": row.model_alias, "display_label": row.display_label,
            "axis": row.axis, "repetition": int(row.repetition), "variant": row.variant, "stage": row.role,
            "lineage": row.lineage}


def _comparison(cache: CampaignCache) -> dict:
    """Comparison settings with defaults."""
    options = dict(cache.settings.get("comparison", {}))
    options.setdefault("key_metrics", [
        {"family": "reconstruction", "metric": "masserstein"},
        {"family": "reconstruction", "metric": "spectral_angle"},
        {"family": "prediction", "metric": "average_precision", "ranking": "annotation_retrieval"},
        {"family": "prediction", "metric": "micro_average_precision", "ranking": "annotation_retrieval"},
        {"family": "prediction", "metric": "f1", "ranking": "annotation_retrieval"}])
    options.setdefault("selection_population", "test")
    options.setdefault("confidence", 0.95)
    return options


def _is_key(frame: pd.DataFrame, specifications: list[dict]) -> pd.Series:
    """Rows of a metric table that belong to the configured key metrics."""
    selected = pd.Series(False, index=frame.index)
    for spec in specifications:
        selected |= ((frame.family == spec["family"]) & (frame.metric == spec["metric"])
                     & (frame.ranking == spec.get("ranking", NO_RANKING)))
    return selected


def _baseline(frame: pd.DataFrame) -> pd.DataFrame:
    """Rows of the baseline variant."""
    return frame[frame.variant == campaign.BASELINE_VARIANT]


def _variants(frame: pd.DataFrame) -> pd.DataFrame:
    """Rows of the pretraining variants."""
    return frame[frame.variant != campaign.BASELINE_VARIANT]


def _workers(cache: CampaignCache) -> int:
    """Parallel CPU workers of the heavy per-model computations."""
    return int(cache.settings.get("workers", 1))


def _alias(cache: CampaignCache, axis: str, variant: str, stage: Optional[str]) -> str:
    """Model alias of one cell (the baseline alias when ``variant`` is the baseline)."""
    models = cache.models[(cache.models.axis == axis) & (cache.models.variant == variant)]
    if variant != campaign.BASELINE_VARIANT:
        models = models[models.role == stage]
    aliases = models.model_alias.unique()
    if len(aliases) != 1:
        raise ValueError(f"Expected one alias for {axis}/{variant}/{stage}, found {list(aliases)}.")
    return str(aliases[0])


# --------------------------------------------------
# Section: per-model metric tables
# --------------------------------------------------

def _pixel_means(cache: CampaignCache, model_id: str, population: str, keys: tuple[str, ...]) -> dict[str, float]:
    """Means of per-pixel arrays of one model (only the requested arrays are read)."""
    with np.load(cache.model_directory(model_id) / f"{population}_pixels.npz") as archive:
        return {key: float(np.mean(archive[key])) for key in keys}


def model_metrics(cache: CampaignCache, models: pd.DataFrame, *,
                  evaluations: tuple[str, ...] = PREDICTION_EVALUATIONS,
                  populations: tuple[str, ...] = RECONSTRUCTION_POPULATIONS) -> pd.DataFrame:
    """Reconstruction and head metrics of every model as one metric table.

    Reconstruction: population means of the per-pixel metrics. Head: the whole-axis
    ranking table (``scope`` train_supported, ``group`` all) for both ranking populations,
    plus ``heldout_image_per_image`` (mean over the held-out images of the per-image value).

    :param cache: Cache view.
    :type cache: CampaignCache
    :param models: Model records.
    :type models: pandas.DataFrame
    :param evaluations: Head evaluation populations.
    :type evaluations: tuple[str, ...]
    :param populations: Reconstruction populations.
    :type populations: tuple[str, ...]
    :return: Metric table (see :mod:`.pretraining_campaign_comparison`).
    :rtype: pandas.DataFrame
    """
    rows = []
    for row in models.itertuples():
        identity = _identity(row)
        for population in populations:
            for metric, value in _pixel_means(cache, row.model_id, population, PIXEL_SUMMARY_METRICS).items():
                rows.append({**identity, "family": "reconstruction", "metric": metric, "evaluation": population,
                             "ranking": NO_RANKING, "value": value})
        prediction = cache.model_table(row.model_id, "prediction")
        prediction = prediction[(prediction.scope == "train_supported") & prediction.metric.isin(PREDICTION_METRICS)]
        overall = prediction[(prediction.group == "all") & prediction.evaluation.isin(evaluations)]
        rows.extend({**identity, "family": "prediction", "metric": record.metric, "evaluation": record.evaluation,
                     "ranking": record.population, "value": float(record.value)} for record in overall.itertuples())
        if "heldout_image" in evaluations:
            images = prediction[(prediction.evaluation == "heldout_image") & (prediction.group != "all")]
            for (population, metric), frame in images.groupby(["population", "metric"]):
                rows.append({**identity, "family": "prediction", "metric": metric,
                             "evaluation": "heldout_image_per_image", "ranking": population,
                             "value": float(frame.value.mean())})
    frame = pd.DataFrame(rows)
    frame["direction"] = frame.metric.map(metric_direction)
    return frame


def improvements(metrics: pd.DataFrame) -> pd.DataFrame:
    """Paired improvements of every variant row over the baseline of the same axis and repetition."""
    return paired_improvements(_variants(metrics), _baseline(metrics), pair_on=UNIT)


def select_best_variant(cache: CampaignCache, axis: str, stage: str) -> tuple[str, pd.DataFrame]:
    """Best variant of one axis and stage: lowest mean rank of the mean improvement.

    Ranked on the configured selection population (withheld test pixels by default),
    never on the held-out images that are displayed.

    :param cache: Cache view containing the baselines and the stage models of the axis.
    :type cache: CampaignCache
    :param axis: Axis name.
    :type axis: str
    :param stage: Stage name.
    :type stage: str
    :return: Best variant and the variant ranking table.
    :rtype: tuple[str, pandas.DataFrame]
    """
    options = _comparison(cache)
    population = options["selection_population"]
    models = cache.models[(cache.models.axis == axis) & ((cache.models.role == stage)
                                                         | (cache.models.variant == campaign.BASELINE_VARIANT))]
    metrics = model_metrics(cache, models, evaluations=(population,), populations=(population,))
    metrics = metrics[_is_key(metrics, options["key_metrics"])]
    summary = summarize(improvements(metrics), [*GROUP, *METRIC_KEYS], confidence=options["confidence"])
    ranking = rank_variants(summary, options["key_metrics"], evaluation=population,
                            tie_breaker=options.get("tie_breaker"))
    return str(ranking.variant.iloc[0]), ranking.assign(axis=axis, stage=stage)


# --------------------------------------------------
# Section: part 0 — pretraining and fine-tuning histories
# --------------------------------------------------

@register("synthetic_pretraining_curves", per_axis=False, requires=("synthetic_reference",), scope="all")
def synthetic_pretraining_curves(cache: CampaignCache, axis: Optional[str]) -> dict:
    """Synthetic-phase learning curves, the synthetic-to-real gap and the synthetic exposure.

    Tables: ``curves`` (pretrained models, synthetic phases, long form), ``group_curves``
    (variant x phase x epoch x component x split across repetitions), ``real_test`` (test
    values of the zero-epoch real evaluation of every pretrained model next to the
    baseline test values), ``synthetic_real_gap`` (final synthetic validation versus real
    test Masserstein per model), ``exposure`` (synthetic tokens per variant population,
    class and kind) and ``variant_populations``.
    """
    history, _ = _history(cache.view(cache.models[cache.models.role.isin(["pretrained", cache.settings["baseline_role"]])]))
    pretrained = history[history.stage == "pretrained"]
    synthetic = pretrained[pretrained.split != "test"]
    group_curves = synthetic.groupby(["axis", "variant", "phase", "phase_index", "component", "split", "epoch"]).value.agg(
        ["mean", "std", "min", "max", "count"]).reset_index()
    real_test = history[history.split == "test"][[*HISTORY_IDENTITY, "phase", "component", "value"]]
    ## Synthetic validation at the end of the last synthetic phase versus the real test value
    final = (synthetic[(synthetic.split == "validation") & (synthetic.component == "masserstein")]
             .sort_values(["phase_index", "epoch"]).groupby("model_id").last().value.rename("synthetic_validation"))
    real = (pretrained[(pretrained.split == "test") & (pretrained.component == "masserstein")]
            .set_index("model_id").value.rename("real_test"))
    identity = cache.models.set_index("model_id")[["axis", "variant", "repetition"]]
    gap = pd.concat([final, real], axis=1, join="inner").join(identity).reset_index()
    gap["real_over_synthetic"] = gap.real_test / gap.synthetic_validation
    ## Populations seen by every variant and their exposure
    populations = []
    for task in cache.plan["tasks"].values():
        if task.workflow["role"] != "pretrained":
            continue
        variant = campaign.task_variant(task, cache.settings.get("variants") or {}, cache.plan["tasks"])
        for position, phase in enumerate(phase for phase in task.parameters["training"]["phases"]
                                         if "pretraining" in phase):
            populations.append({"axis": task.grid_parameters["axes"]["name"], "variant": variant,
                                "phase": phase["phase_name"], "phase_index": position,
                                "population": phase["pretraining"]["population"], "epochs": int(phase["epochs"])})
    variant_populations = pd.DataFrame(populations).drop_duplicates().reset_index(drop=True)
    exposure = []
    for axis_name in cache.axes:
        path = cache.synthetic_root / axis_directory(cache.settings, axis_name) / "exposure.csv"
        exposure.append(pd.read_csv(path).assign(axis=axis_name))
    return {"curves": synthetic, "group_curves": group_curves, "real_test": real_test, "synthetic_real_gap": gap,
            "exposure": pd.concat(exposure, ignore_index=True), "variant_populations": variant_populations,
            "metadata": {"note": "epochs restart in every phase; phase_index orders the phases of a staged variant"}}


@register("finetuning_curves", per_axis=False, requires=(), scope="all")
def finetuning_curves(cache: CampaignCache, axis: Optional[str]) -> dict:
    """Real-data learning curves of the fine-tuned models against the baselines, epoch by epoch.

    Both the baseline and every fine-tuning phase train 10 epochs on the same real data,
    so epoch ``e`` of a fine-tuning phase is paired with epoch ``e`` of the baseline of
    the same axis and repetition. Tables: ``curves`` (long form), ``epoch_improvements``
    (paired, sign-adjusted), ``epoch_summary`` (per variant, stage, component, split and
    epoch), ``convergence`` (least-squares slope over the last ``finetuning.slope_epochs``
    epochs per model; a negative slope of a loss means it was still decreasing) and
    ``epochs_to_baseline`` (first epoch at which a model reaches the final value of its
    paired baseline).
    """
    options = cache.settings.get("finetuning", {})
    slope_epochs = int(options.get("slope_epochs", 3))
    components = tuple(options.get("components", ("masserstein", "total_loss")))
    real_roles = [cache.settings["baseline_role"], "frozen_head", "unfrozen_head"]
    history, _ = _history(cache.view(cache.models[cache.models.role.isin(real_roles)]))
    curves = history[(history.split != "test") & history.component.isin(components)]
    baseline = _baseline(curves)[["axis", "repetition", "component", "split", "epoch", "value"]]
    paired = _variants(curves).merge(baseline.rename(columns={"value": "baseline_value"}),
                                     on=["axis", "repetition", "component", "split", "epoch"], how="inner")
    paired["improvement"] = (paired.value - paired.baseline_value) * paired.component.map(metric_direction)
    epoch_summary = summarize(paired, ["axis", "variant", "stage", "component", "split", "epoch"])
    ## Slope over the last epochs (loss units per epoch)
    slopes = []
    for (model_id, component, split), frame in curves.groupby(["model_id", "component", "split"]):
        tail = frame.sort_values("epoch").tail(slope_epochs)
        slope = np.polyfit(tail.epoch.astype(float), tail.value.astype(float), 1)[0] if len(tail) > 1 else np.nan
        slopes.append({**tail.iloc[-1][list(HISTORY_IDENTITY)].to_dict(), "component": component, "split": split,
                       "slope_last_epochs": float(slope), "final_value": float(tail.value.iloc[-1])})
    convergence = pd.DataFrame(slopes)
    ## First epoch reaching the paired baseline's final value
    final_baseline = (baseline.sort_values("epoch").groupby(["axis", "repetition", "component", "split"]).value.last()
                      .rename("baseline_final").reset_index())
    reached = _variants(curves).merge(final_baseline, on=["axis", "repetition", "component", "split"])
    sign = reached.component.map(metric_direction)
    reached["reached"] = (reached.value - reached.baseline_final) * sign >= 0
    first = []
    for (_, component, split), frame in reached.groupby(["model_id", "component", "split"]):
        hit = frame[frame.reached].epoch
        first.append({**frame.iloc[0][list(HISTORY_IDENTITY)].to_dict(), "component": component, "split": split,
                      "first_epoch_at_baseline_final": int(hit.min()) if len(hit) else np.nan,
                      "epochs": int(frame.epoch.max())})
    return {"curves": curves, "epoch_improvements": paired, "epoch_summary": epoch_summary,
            "convergence": convergence, "epochs_to_baseline": pd.DataFrame(first),
            "metadata": {"slope_epochs": slope_epochs, "components": list(components)}}


# --------------------------------------------------
# Section: parts 2 and 3 — one stage against the baselines
# --------------------------------------------------

@register("stage_overview", per_axis=True, scope="stage")
def stage_overview(cache: CampaignCache, axis: str, stage: str) -> dict:
    """Every variant of one stage against the baseline: values, paired improvements, ranking.

    Tables: ``model_metrics`` (every model incl. baselines), ``improvements`` (paired per
    repetition), ``improvement_summary`` (per variant: mean, sd, t interval, repetitions
    improving), ``value_summary`` (absolute values incl. the baseline),
    ``generalization_gap`` (held-out minus test per model and its paired improvement) and
    ``variant_ranking`` (selection on the test pixels).
    """
    options = _comparison(cache)
    metrics = model_metrics(cache, cache.models)
    paired = improvements(metrics)
    summary = summarize(paired, [*GROUP, *METRIC_KEYS], confidence=options["confidence"])
    values = summarize(metrics, [*GROUP, *METRIC_KEYS], value="value", confidence=options["confidence"])
    ## Generalization gap: held-out images minus withheld test pixels
    wide = metrics[metrics.evaluation.isin(["test", "heldout_image"])]
    pivot = wide.pivot_table(index=[*IDENTITY, "family", "metric", "ranking"], columns="evaluation",
                             values="value").reset_index()
    pivot = pivot.dropna(subset=["test", "heldout_image"])
    gap = pivot.assign(value=pivot.heldout_image - pivot.test, evaluation="heldout_minus_test")
    gap_paired = paired_improvements(_variants(gap), _baseline(gap), pair_on=UNIT)
    ranking = rank_variants(summary, options["key_metrics"], evaluation=options["selection_population"],
                            tie_breaker=options.get("tie_breaker"))
    return {"model_metrics": metrics, "improvements": paired, "improvement_summary": summary,
            "value_summary": values, "generalization_gap": gap_paired,
            "variant_ranking": ranking.assign(axis=axis, stage=stage),
            "metadata": {"best_variant": str(ranking.variant.iloc[0]), "stage": stage,
                         "selection_population": options["selection_population"],
                         "key_metrics": options["key_metrics"]}}


def other_baseline_cost(cache: CampaignCache, axis: str, representative: Any, population: str) -> np.ndarray:
    """Mean per-pixel Masserstein cost of the baseline repetitions other than the representative.

    Pixels selected as the worst of one model regress towards the mean for any other
    model; this cost is the reference for that effect (an improvement of a variant on such
    pixels is informative only beyond it).

    :return: Mean cost per pixel of the population, shape ``(N,)``.
    :rtype: numpy.ndarray
    """
    models = cache.models[(cache.models.axis == axis) & (cache.models.variant == campaign.BASELINE_VARIANT)
                          & (cache.models.model_id != representative.model_id)]
    return np.mean([cache.model_pixels(model_id, population)["masserstein"] for model_id in models.model_id], axis=0)


#: Pixel kinds of :func:`extreme_spectra` and the number of pixels per kind.
EXTREME_KINDS = (("baseline_worst", 2), ("variant_best", 1), ("variant_worst", 1), ("most_improved", 1),
                 ("most_worsened", 1))


def extreme_spectra(cache: CampaignCache, axis: str, baseline_row: Any,
                    variant_rows: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Held-out spectra of the baseline and of every variant representative on the same pixels.

    Pixels are chosen from the per-pixel Masserstein cost of the representatives: the
    baseline's worst pixels (shared by every variant, to see whether a variant repairs
    them), the variant's best and worst pixels and the pixels with the largest paired
    improvement and deterioration (baseline cost minus variant cost). Spectra are the
    stored display reconstructions (bins of windows with at least the display mass).

    :param baseline_row: Model record of the baseline representative.
    :type baseline_row: typing.Any
    :param variant_rows: Variant name mapped to its representative's model record.
    :type variant_rows: dict[str, typing.Any]
    :return: ``pixels`` (one row per variant and selected pixel: identity, kind, rank,
        image, coordinates, both costs and both largest-contribution windows) and ``spectra``
        (one row per variant, pixel and display bin: ``input``, ``baseline_output``, ``output``).
    :rtype: tuple[pandas.DataFrame, pandas.DataFrame]
    """
    pixels = _heldout_images(cache)
    windows = _windows(cache, axis)
    bins = np.asarray(cache.axis_array(axis, "display_bins"))  # (K,)
    display_input = cache.axis_array(axis, "display_input")  # (N_h, K)
    mass_axis = cache.axis_meta(axis)["mass_axis"]
    baseline = cache.model_pixels(baseline_row.model_id, "heldout_image")
    baseline_output = cache.model_array(baseline_row.model_id, "display_reconstruction")  # (N_h, K)
    baseline_cost = baseline["masserstein"]  # (N_h,)
    other_cost = other_baseline_cost(cache, axis, baseline_row, "heldout_image")  # (N_h,)
    selected, frames = [], []
    for variant, row in variant_rows.items():
        arrays = cache.model_pixels(row.model_id, "heldout_image")
        cost = arrays["masserstein"]  # (N_h,)
        gain = baseline_cost - cost  # (N_h,); positive = the variant is better on this pixel
        order = {"baseline_worst": np.argsort(baseline_cost)[::-1], "variant_best": np.argsort(cost),
                 "variant_worst": np.argsort(cost)[::-1], "most_improved": np.argsort(gain)[::-1],
                 "most_worsened": np.argsort(gain)}
        output = cache.model_array(row.model_id, "display_reconstruction")  # (N_h, K)
        chosen: set[int] = set()
        for kind, count in EXTREME_KINDS:
            ## REMARK: a pixel already shown for this variant is skipped, so kinds never repeat one spectrum.
            picked = [int(pixel) for pixel in order[kind] if int(pixel) not in chosen][:count]
            for rank, pixel in enumerate(picked):
                chosen.add(pixel)
                selected.append({
                    **_identity(row), "pixel_kind": kind, "rank": rank, "row": pixel,
                    "dataset_id": pixels.dataset_id.iloc[pixel], "x": int(pixels.x.iloc[pixel]),
                    "y": int(pixels.y.iloc[pixel]), "masserstein_baseline": float(baseline_cost[pixel]),
                    "masserstein_baseline_others": float(other_cost[pixel]), "masserstein": float(cost[pixel]),
                    "worst_window_baseline": windows.window_label.iloc[int(np.argmax(baseline["contribution"][pixel]))],
                    "worst_window": windows.window_label.iloc[int(np.argmax(arrays["contribution"][pixel]))]})
                if any(frame["variant"].iloc[0] == variant and frame["row"].iloc[0] == pixel for frame in frames):
                    continue
                frames.append(pd.DataFrame({
                    "variant": variant, "row": pixel, "bin": bins, "mz": mass_axis[bins],
                    "input": np.asarray(display_input[pixel], dtype=np.float32),
                    "baseline_output": np.asarray(baseline_output[pixel], dtype=np.float32),
                    "output": np.asarray(output[pixel], dtype=np.float32)}))
    spectra = pd.concat(frames, ignore_index=True)
    ## REMARK: intensities are TIC fractions; six decimals keep the display precision in a smaller CSV.
    return pd.DataFrame(selected), spectra.round({"mz": 4, "input": 6, "baseline_output": 6, "output": 6})


@register("stage_reconstruction", per_axis=True, scope="stage", version=5)
def stage_reconstruction(cache: CampaignCache, axis: str, stage: str) -> dict:
    """Local (100 Da) reconstruction of every variant of one stage against the baseline.

    Tables: ``window_summary`` (model x population x window: mean contribution to W1,
    mean within-window W1, mean signed and absolute mass imbalance),
    ``window_improvements`` (paired per repetition), ``window_improvement_summary``,
    ``image_breakdown`` (per held-out image and model, paired), ``feature_errors``
    (per-bin mean errors of the representative of the baseline and of every variant)
    and ``cases`` (fixed test and held-out spectra of the baseline and best-variant
    representatives) and ``extreme_spectra`` (held-out spectra of the baseline and every
    variant representative on the same selected pixels) with ``extreme_pixels``
    (:func:`extreme_spectra`).
    """
    options = _comparison(cache)
    windows = _windows(cache, axis)
    rows, images = [], []
    for row in cache.models.itertuples():
        identity = _identity(row)
        for population in RECONSTRUCTION_POPULATIONS:
            with np.load(cache.model_directory(row.model_id) / f"{population}_pixels.npz") as archive:
                arrays = {key: archive[key] for key in ("contribution", "within", "input_mass", "output_mass",
                                                        "masserstein", "spectral_angle")}
            imbalance = arrays["output_mass"] - arrays["input_mass"]  # (N, W)
            with np.errstate(all="ignore"):
                quantities = {"contribution": arrays["contribution"].mean(axis=0),
                              "within": np.nanmean(arrays["within"], axis=0),
                              "mass_imbalance": imbalance.mean(axis=0),
                              "absolute_mass_imbalance": np.abs(imbalance).mean(axis=0)}
            for quantity, values in quantities.items():
                rows.append(windows.assign(**identity, population=population, metric=quantity, value=values))
            if population == "heldout_image":
                datasets = _rows(cache, population).dataset_id.to_numpy()
                for dataset_id in np.unique(datasets):
                    selected = datasets == dataset_id
                    for metric in ("masserstein", "spectral_angle"):
                        images.append({**identity, "dataset_id": dataset_id, "metric": metric,
                                       "value": float(arrays[metric][selected].mean())})
    summary = pd.concat(rows, ignore_index=True)
    table = summary.assign(family="reconstruction_window", evaluation=summary.population + "@" + summary.window_label,
                           ranking=NO_RANKING)
    paired = paired_improvements(_variants(table), _baseline(table), pair_on=UNIT)
    window_summary = summarize(paired, [*GROUP, "population", "window", "window_label", "metric"],
                               confidence=options["confidence"])
    images = pd.DataFrame(images).assign(family="reconstruction", ranking=NO_RANKING)
    images["evaluation"] = images.dataset_id
    image_paired = paired_improvements(_variants(images), _baseline(images), pair_on=UNIT)
    ## Representatives: per-bin errors and fixed spectrum cases
    best, _ = select_best_variant(cache, axis, stage)
    baseline_row = cache.cell_representative(_alias(cache, axis, campaign.BASELINE_VARIANT, None))
    features, cases, variant_rows = [], [], {}
    meta = cache.axis_meta(axis)
    for variant in [campaign.BASELINE_VARIANT, *sorted(_variants(cache.models).variant.unique())]:
        row = baseline_row if variant == campaign.BASELINE_VARIANT else cache.cell_representative(
            _alias(cache, axis, variant, stage))
        if variant != campaign.BASELINE_VARIANT:
            variant_rows[variant] = row
        for population in ("test", "heldout_image"):
            table = cache.model_table(row.model_id, f"{population}_features")
            features.append(table[table.group == "all"].assign(**_identity(row), population=population))
            if variant not in (campaign.BASELINE_VARIANT, best):
                continue
            with np.load(cache.model_directory(row.model_id) / f"{population}_cases.npz") as archive:
                fixed = archive["fixed"]
                inputs, outputs, case_rows = archive["input"][fixed], archive["output"][fixed], archive["rows"][fixed]
            for position, case_row in enumerate(case_rows):
                cases.append(pd.DataFrame({**_identity(row), "population": population, "row": int(case_row),
                                           "bin": np.arange(meta["mass_axis"].size), "mz": meta["mass_axis"],
                                           "input": inputs[position], "output": outputs[position]}))
    return {"window_summary": summary, "window_improvements": paired, "window_improvement_summary": window_summary,
            "image_breakdown": image_paired, "feature_errors": pd.concat(features, ignore_index=True),
            "cases": pd.concat(cases, ignore_index=True),
            **dict(zip(("extreme_pixels", "extreme_spectra"), extreme_spectra(cache, axis, baseline_row, variant_rows))),
            "metadata": {"best_variant": best, "baseline_representative": baseline_row.model_id,
                         "window_width": cache.settings["windows"]["width"]}}


def _evaluation_scores(cache: CampaignCache, row: Any, axis: str, evaluation: str) -> tuple[np.ndarray, np.ndarray,
                                                                                           np.ndarray]:
    """Logits, available annotations and evaluable-state mask of one evaluation population.

    ``test_combined`` pools the test pixels with the visible test-extension pixels; the
    held-out logits are recovered from the stored float16 probabilities.

    :return: Scores ``(N, C)``, annotations ``(N, C)`` and availability ``(N, C)``.
    :rtype: tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]
    """
    if evaluation == "heldout_image":
        arrays = cache.population(axis, "heldout_image")
        scores = logits_from_probabilities(cache.model_array(row.model_id, "heldout_image_probabilities"))
        return scores, _available_targets(arrays), np.asarray(arrays["states"]) >= 0
    parts = [("test", slice(None)), ("test_extended", np.asarray(cache.population(axis, "test_extended")["visible"],
                                                                 dtype=bool))]
    if evaluation != "test_combined":
        parts = [(evaluation, slice(None))]
    scores, labels, available = [], [], []
    for name, rows in parts:
        arrays = cache.population(axis, name)
        scores.append(np.asarray(cache.model_array(row.model_id, f"{name}_logits"), dtype=np.float64)[rows])
        labels.append(_available_targets(arrays)[rows])
        available.append((np.asarray(arrays["states"]) >= 0)[rows])
    return np.concatenate(scores), np.concatenate(labels), np.concatenate(available)


@register("stage_prediction", per_axis=True, scope="stage")
def stage_prediction(cache: CampaignCache, axis: str, stage: str) -> dict:
    """Head metrics of every variant of one stage against the baseline, down to single classes.

    Tables: ``prediction`` (complete ranking table, whole axis and per held-out image),
    ``class_improvements`` (per class and variant: mean paired AP/recall/precision/F1
    improvement over repetitions), ``class_tests`` (Wilcoxon signed-rank test over classes
    per variant), ``strata`` (complete metric set per class stratum) and
    ``strata_improvements``, ``calibration_bins`` and ``calibration`` (ECE, Brier on the
    training-supported classes, relative to the annotations).
    """
    options = _comparison(cache)
    models = cache.models
    prediction = cache.collect("prediction", models)
    classes = cache.axis_table(axis, "classes")
    ## Per-class paired improvements
    per_class = cache.collect("per_class", models)
    per_class = per_class[(per_class.group == "all") & (per_class.population == "annotation_retrieval")
                          & per_class.evaluation.isin(["test_combined", "heldout_image"])
                          & per_class.eligible.astype(bool)]
    long = per_class.melt(id_vars=[*IDENTITY, "evaluation", "class_name"],
                          value_vars=["average_precision", "recall", "precision", "f1", "roc_auc"],
                          var_name="metric", value_name="value").assign(family="class", ranking="annotation_retrieval")
    long["evaluation"] = long.evaluation + "|" + long.class_name
    paired = paired_improvements(_variants(long), _baseline(long), pair_on=UNIT)
    paired[["evaluation", "class_name"]] = paired.evaluation.str.split("|", n=1, expand=True)
    class_summary = summarize(paired, [*GROUP, "evaluation", "class_name", "metric"], confidence=options["confidence"])
    class_summary = class_summary.merge(classes[["class_name", "mz", "class_set", "train_positives"]],
                                        on="class_name", how="left")
    tests = []
    for keys, frame in class_summary.groupby([*GROUP, "evaluation", "metric"]):
        values = frame["mean"].dropna().to_numpy()
        nonzero = values[values != 0]
        statistic, p_value = (stats.wilcoxon(nonzero) if nonzero.size >= 10 else (np.nan, np.nan))
        tests.append({**dict(zip([*GROUP, "evaluation", "metric"], keys)), "classes": int(values.size),
                      "median_improvement": float(np.median(values)) if values.size else np.nan,
                      "improved_fraction": float((values > 0).mean()) if values.size else np.nan,
                      "wilcoxon_statistic": float(statistic), "wilcoxon_p_value": float(p_value)})
    ## Class strata (complete metric set from the inference level)
    strata = cache.collect("strata_prediction", models)
    strata = strata[strata.group == "all"]
    frequency = prediction[(prediction.group == "all") & prediction.scope.isin(["rare", "medium", "frequent"])]
    frequency = frequency.assign(grouping="frequency", stratum=frequency.scope)
    strata = pd.concat([strata, frequency], ignore_index=True)
    strata_long = strata.assign(family="stratum", ranking=strata.population,
                                evaluation=strata.evaluation + "|" + strata.grouping + "|" + strata.stratum.astype(str))
    strata_paired = paired_improvements(_variants(strata_long), _baseline(strata_long), pair_on=UNIT)
    strata_summary = summarize(strata_paired, [*GROUP, *METRIC_KEYS], confidence=options["confidence"])
    ## Calibration on the training-supported classes
    supported = classes.train_positives.to_numpy() > 0
    bins, calibration = [], []
    for row in models.itertuples():
        for evaluation in ("test_combined", "heldout_image"):
            scores, labels, available = _evaluation_scores(cache, row, axis, evaluation)
            probabilities = 1.0 / (1.0 + np.exp(-scores[:, supported]))
            mask = available[:, supported]
            table, summary = calibration_table(probabilities[mask], labels[:, supported][mask])
            bins.append(table.assign(**_identity(row), evaluation=evaluation))
            calibration.append({**_identity(row), "evaluation": evaluation, **summary})
    calibration = pd.DataFrame(calibration)
    calibration_long = calibration.melt(id_vars=[*IDENTITY, "evaluation"],
                                        value_vars=["expected_calibration_error", "brier_score"],
                                        var_name="metric").assign(family="calibration", ranking=NO_RANKING)
    calibration_paired = paired_improvements(_variants(calibration_long), _baseline(calibration_long), pair_on=UNIT)
    return {"prediction": prediction, "class_improvements": class_summary, "class_tests": pd.DataFrame(tests),
            "strata": strata, "strata_improvements": strata_summary,
            "calibration_bins": pd.concat(bins, ignore_index=True), "calibration": calibration,
            "calibration_improvements": calibration_paired,
            "metadata": {"note": "calibration is measured against incomplete annotations (a lower bound of the "
                                 "true positive rate)"}}


def _probe_sample(cache: CampaignCache, axis: str) -> np.ndarray:
    """Training rows used to fit the linear probes (identical for every model)."""
    options = cache.settings.get("probe", {})
    count = len(_rows(cache, "train"))
    return _sample_rows(count, int(options.get("train_size", 20000)), int(options.get("seed", 42)), 60)


def probe_predictions(cache: CampaignCache, row: Any, axis: str, device: Any) -> pd.DataFrame:
    """Ridge probe of the annotations from the canonical latent codes of one model.

    The probe is fitted on a fixed training sample and evaluated with the same ranking
    function, evidence states and class scope as the head, so ``probe`` and ``head``
    average precision are directly comparable: their difference is what the head's
    decision function loses (or adds) relative to the linearly decodable information.

    :return: Ranking rows (``train_supported`` scope) of the probe per evaluation population.
    :rtype: pandas.DataFrame
    """
    options = cache.settings.get("probe", {})
    classes = cache.axis_table(axis, "classes")
    train_counts = classes.train_positives.to_numpy()
    sample = _probe_sample(cache, axis)
    _, u_train = _latent_codes(cache, row, "train")
    train_targets = _available_targets(cache.population(axis, "train"))[sample]
    evaluations = {}
    for evaluation in ("test_combined", "heldout_image"):
        if evaluation == "heldout_image":
            _, u = _latent_codes(cache, row, "heldout_image")
            arrays = cache.population(axis, "heldout_image")
            targets = _available_targets(arrays).astype(np.float32)
            states = np.asarray(arrays["states"])
        else:
            visible = np.asarray(cache.population(axis, "test_extended")["visible"], dtype=bool)
            codes, labels, state = [], [], []
            for name, rows in (("test", slice(None)), ("test_extended", visible)):
                arrays = cache.population(axis, name)
                codes.append(_latent_codes(cache, row, name)[1][rows])
                labels.append(_available_targets(arrays)[rows])
                state.append(np.asarray(arrays["states"])[rows])
            u, targets, states = np.concatenate(codes), np.concatenate(labels).astype(np.float32), np.concatenate(state)
        evaluations[evaluation] = (u, targets, states)
    scores = ridge_probe(u_train[sample], train_targets, [value[0] for value in evaluations.values()],
                         regularization=float(options.get("regularization", 1e-3)))
    frames = []
    for (evaluation, (_, targets, states)), values in zip(evaluations.items(), scores):
        table = ranking_tables(values.astype(np.float32), targets, states, train_counts, tuple(classes.class_name),
                               device=device, family=None)["prediction"]
        table = table[(table.scope == "train_supported") & (table.population == "annotation_retrieval")]
        frames.append(table.assign(**_identity(row), evaluation=evaluation))
    return pd.concat(frames, ignore_index=True)


def _latent_job(cache: CampaignCache, row: Any, samples: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Latent battery of one model on the test and held-out populations (worker function)."""
    return latent_model_summary(cache, row, samples, populations=("test", "heldout_image"))


@register("stage_latent", per_axis=True, scope="stage")
def stage_latent(cache: CampaignCache, axis: str, stage: str) -> dict:
    """Latent geometry, linear decodability and similarity to the baseline representation.

    Tables: ``geometry`` (battery of part 1 on test and held-out pixels, spaces z and u),
    ``eigenvalues``, ``geometry_improvements`` (paired), ``probe`` (ridge probe ranking
    metrics), ``probe_versus_head`` (probe minus head average precision per model),
    ``to_baseline`` (linear CKA, Procrustes distance and neighbourhood overlap to the
    baseline of the same repetition on identical pixels) and ``sensitivity``.
    """
    options = _comparison(cache)
    samples = _latent_samples(cache, axis)
    results = Parallel(n_jobs=_workers(cache))(delayed(_latent_job)(cache, row, samples)
                                               for row in cache.models.itertuples())
    identity = cache.models.set_index("model_id")[["variant", "role", "lineage"]].rename(columns={"role": "stage"})
    geometry = pd.concat([value for value, _ in results], ignore_index=True).join(identity, on="model_id")
    eigenvalues = pd.concat([value for _, value in results], ignore_index=True).join(identity, on="model_id")
    long = geometry[geometry.space == "u"].assign(family="latent", evaluation=lambda frame: frame.population, ranking=NO_RANKING)
    geometry_paired = paired_improvements(_variants(long), _baseline(long), pair_on=UNIT)
    ## Linear probe and the head on the same scope
    device = resolve_device(cache.settings, allow_cpu=True)
    probe = pd.concat([probe_predictions(cache, row, axis, device) for row in cache.models.itertuples()],
                      ignore_index=True)
    head = cache.collect("prediction", cache.models)
    head = head[(head.group == "all") & (head.scope == "train_supported") & (head.population == "annotation_retrieval")
                & head.evaluation.isin(["test_combined", "heldout_image"])]
    versus = probe[["model_id", "evaluation", "metric", "value"]].merge(
        head[["model_id", "evaluation", "metric", "value"]], on=["model_id", "evaluation", "metric"],
        suffixes=("_probe", "_head")).join(cache.models.set_index("model_id")[["axis", "repetition", "variant",
                                                                              "role"]], on="model_id")
    versus["probe_minus_head"] = versus.value_probe - versus.value_head
    ## Similarity to the baseline representation of the same repetition
    baselines = {int(row.repetition): row for row in _baseline(cache.models).itertuples()}
    similarity = []
    for row in _variants(cache.models).itertuples():
        reference = baselines.get(int(row.repetition))
        if reference is None:
            continue
        for population in ("test", "heldout_image"):
            battery = pairwise_geometry_battery(_latent_codes(cache, row, population)[1],
                                                _latent_codes(cache, reference, population)[1], samples[population],
                                                k=int(cache.settings.get("latent", {}).get("neighbours", 10)))
            similarity.append({**_identity(row), "population": population, "baseline_model": reference.model_id,
                               **battery})
    return {"geometry": geometry, "eigenvalues": eigenvalues,
            "geometry_improvements": summarize(geometry_paired, [*GROUP, "population", "metric"],
                                               confidence=options["confidence"]),
            "probe": probe, "probe_versus_head": versus, "to_baseline": pd.DataFrame(similarity),
            "sensitivity": cache.collect("sensitivity", cache.models),
            "metadata": {"populations": ["test", "heldout_image"],
                         "probe": {"train_size": int(len(_probe_sample(cache, axis))),
                                   "regularization": float(cache.settings.get("probe", {}).get("regularization",
                                                                                               1e-3))}}}


def _representative_rows(cache: CampaignCache, axis: str, stage: str) -> dict[str, Any]:
    """Representative model of the baseline and of every variant of one stage."""
    rows = {campaign.BASELINE_VARIANT: cache.cell_representative(_alias(cache, axis, campaign.BASELINE_VARIANT, None))}
    for variant in sorted(_variants(cache.models[cache.models.axis == axis]).variant.unique()):
        rows[variant] = cache.cell_representative(_alias(cache, axis, variant, stage))
    return rows


@register("stage_segmentation", per_axis=True, scope="stage")
def stage_segmentation(cache: CampaignCache, axis: str, stage: str) -> dict:
    """Latent segmentation of the held-out images by the representatives of one stage.

    Mixtures are fitted on a seeded training sample of every representative's codes and
    applied to the held-out pixels; every partition is aligned to the baseline
    representative's partition of the same ``k``. Tables: ``bic``, ``agreement`` (ARI to
    the baseline representative), ``labels`` (``segmentation.stage_k_values`` only),
    ``composition`` (segment fractions per image) and ``contingency`` (to the baseline).
    """
    settings = cache.settings.get("segmentation", {})
    seed = int(settings.get("seed", 42))
    sample_size = int(settings.get("fit_sample_size", 20000))
    k_values = [int(value) for value in settings.get("k_values", (4, 6, 8, 12))]
    stored_k = [int(value) for value in settings.get("stage_k_values", (4, 6))]
    representatives = _representative_rows(cache, axis, stage)
    pixels = _rows(cache, "heldout_image")[["row", "dataset_id", "x", "y"]]
    bic, labels = [], {}
    for variant, row in representatives.items():
        train = cache.model_pixels(row.model_id, "train")["latent"]
        heldout = cache.model_pixels(row.model_id, "heldout_image")["latent"]
        for k in k_values:
            fit = fit_segmentation(train, k, seed=seed, sample_size=sample_size)
            bic.append({**_identity(row), "k": k, "bic": fit.bic})
            labels[(variant, k)] = fit.predict(heldout)
    agreement, contingency, composition, stored = [], [], [], []
    for (variant, k), values in labels.items():
        reference = labels[(campaign.BASELINE_VARIANT, k)]
        aligned = values if variant == campaign.BASELINE_VARIANT else align_labels(reference, values, k)
        row = representatives[variant]
        agreement.append({**_identity(row), "k": k, "ari_to_baseline": adjusted_rand_between(reference, aligned)})
        if variant != campaign.BASELINE_VARIANT:
            contingency.append(segment_contingency(reference, aligned, k).assign(**_identity(row), k=k))
        fractions = pd.DataFrame({"dataset_id": pixels.dataset_id, "segment": aligned}).value_counts(
            normalize=False).rename("pixels").reset_index()
        composition.append(fractions.assign(**_identity(row), k=k))
        if k in stored_k:
            stored.append(pixels.assign(**_identity(row), k=k, segment=aligned))
    return {"bic": pd.DataFrame(bic), "agreement": pd.DataFrame(agreement),
            "labels": pd.concat(stored, ignore_index=True),
            "composition": pd.concat(composition, ignore_index=True),
            "contingency": pd.concat(contingency, ignore_index=True) if contingency else pd.DataFrame(),
            "metadata": {"k_values": k_values, "stored_k_values": stored_k,
                         "representatives": {variant: row.model_id for variant, row in representatives.items()}}}


@register("stage_mechanisms", per_axis=True, scope="stage", version=2)
def stage_mechanisms(cache: CampaignCache, axis: str, stage: str) -> dict:
    """Effect of every element of the synthetic-data construction within one stage.

    Tables: ``contrasts`` (configured paired contrasts between variants, per repetition),
    ``contrast_summary``, ``factorial`` (main effects and two-factor interactions of the
    configured factorial design, per repetition), ``factorial_summary``, ``model_values``
    (key metrics of every model incl. the real-only baseline) and ``value_summary``.
    """
    options = _comparison(cache)
    metrics = model_metrics(cache, cache.models)
    keyed = metrics[_is_key(metrics, options["key_metrics"])]
    metrics = _variants(keyed)
    contrasts = variant_contrasts(metrics, options.get("contrasts", {}))
    factors = {name: value.get("factors", {}) for name, value in (cache.settings.get("variants") or {}).items()}
    levels = options.get("factorial", {}).get("levels", {})
    factorial = factorial_effects(metrics, factors, levels) if levels else pd.DataFrame()
    return {"contrasts": contrasts,
            "contrast_summary": summarize(contrasts, ["axis", "stage", "contrast", *METRIC_KEYS],
                                          confidence=options["confidence"]) if len(contrasts) else pd.DataFrame(),
            "factorial": factorial,
            "factorial_summary": summarize(factorial, ["axis", "stage", "term", "order", *METRIC_KEYS],
                                           confidence=options["confidence"]) if len(factorial) else pd.DataFrame(),
            "model_values": keyed,
            "value_summary": summarize(keyed, [*GROUP, *METRIC_KEYS], value="value", confidence=options["confidence"]),
            "metadata": {"contrasts": options.get("contrasts", {}), "factorial_levels": levels}}


# --------------------------------------------------
# Section: part 3 — stage comparison per axis
# --------------------------------------------------

@register("stage_transitions", per_axis=True, scope="all")
def stage_transitions(cache: CampaignCache, axis: str) -> dict:
    """What fine-tuning changes: pretrained -> frozen head -> unfrozen head, per variant.

    Tables: ``model_metrics`` (baselines and every stage), ``stage_improvements`` (each
    stage against the baseline, summarized per variant and stage) and ``transitions``
    (paired within one pretraining lineage: ``frozen_head - pretrained``,
    ``unfrozen_head - pretrained`` and ``unfrozen_head - frozen_head``, sign-adjusted)
    with ``transition_summary``.
    """
    options = _comparison(cache)
    models = cache.models[cache.models.axis == axis]
    metrics = model_metrics(cache, models)
    stage_summary = summarize(improvements(metrics), [*GROUP, *METRIC_KEYS], confidence=options["confidence"])
    lineage = _variants(metrics)
    wide = lineage.pivot_table(index=["axis", "variant", "repetition", "lineage", *METRIC_KEYS], columns="stage",
                               values="value").reset_index()
    frames = []
    for name, (target, source) in {"frozen_minus_pretrained": ("frozen_head", "pretrained"),
                                   "unfrozen_minus_pretrained": ("unfrozen_head", "pretrained"),
                                   "unfrozen_minus_frozen": ("unfrozen_head", "frozen_head")}.items():
        if {target, source} <= set(wide.columns):
            frame = wide[["axis", "variant", "repetition", "lineage", *METRIC_KEYS, source, target]].dropna()
            frame = frame.assign(transition=name, difference=frame[target] - frame[source])
            frame["improvement"] = frame.difference * frame.metric.map(metric_direction)
            frames.append(frame.drop(columns=[source, target]))
    transitions = pd.concat(frames, ignore_index=True)
    return {"model_metrics": metrics, "stage_improvements": stage_summary, "transitions": transitions,
            "transition_summary": summarize(transitions, ["axis", "variant", "transition", *METRIC_KEYS],
                                            confidence=options["confidence"]),
            "metadata": {"stages": cache.stages}}


@register("stage_representation_change", per_axis=True, scope="all")
def stage_representation_change(cache: CampaignCache, axis: str) -> dict:
    """How far fine-tuning moves the representation and the head, and what the head loses.

    Tables: ``drift`` (per lineage and transition: quantiles of the per-pixel angle
    between canonical codes of two stages on test and held-out pixels), ``similarity``
    (CKA, Procrustes and neighbourhood overlap between stages and to the baseline of the
    same repetition), ``head_change`` (relative Frobenius change of every head tensor
    against the pretrained head) and ``probe_versus_head`` (every model of the axis).
    """
    models = cache.models[cache.models.axis == axis]
    samples = _latent_samples(cache, axis)
    neighbours = int(cache.settings.get("latent", {}).get("neighbours", 10))
    baselines = {int(row.repetition): row for row in _baseline(models).itertuples()}
    drift, similarity, heads = [], [], []
    levels = np.array([0.1, 0.25, 0.5, 0.75, 0.9])
    for lineage, frame in _variants(models).groupby("lineage"):
        stages = {row.role: row for row in frame.itertuples()}
        pairs = [("pretrained", "frozen_head"), ("pretrained", "unfrozen_head"), ("frozen_head", "unfrozen_head")]
        pairs += [(stage, "baseline") for stage in stages]
        for source, target in pairs:
            left = stages.get(source)
            right = baselines.get(int(frame.repetition.iloc[0])) if target == "baseline" else stages.get(target)
            if left is None or right is None:
                continue
            for population in ("test", "heldout_image"):
                u_left, u_right = _latent_codes(cache, left, population)[1], _latent_codes(cache, right, population)[1]
                battery = pairwise_geometry_battery(u_left, u_right, samples[population], k=neighbours)
                record = {"axis": axis, "variant": left.variant, "repetition": int(left.repetition),
                          "lineage": lineage, "source": source, "target": target, "population": population}
                similarity.append({**record, **battery})
                if target != "baseline":
                    angles = angle_between(u_left, u_right)
                    drift.append({**record, "mean_angle_degrees": float(angles.mean()),
                                  **{f"q{int(level * 100):02d}_angle_degrees": float(value)
                                     for level, value in zip(levels, np.quantile(angles, levels))}})
        ## Head tensors against the pretrained head
        if "pretrained" in stages:
            reference = campaign._head_tensors(stages["pretrained"].artifact, stages["pretrained"].head)
            for stage in ("frozen_head", "unfrozen_head"):
                if stage not in stages:
                    continue
                tensors = campaign._head_tensors(stages[stage].artifact, stages[stage].head)
                for name, value in reference.items():
                    change = float((tensors[name] - value).norm() / max(float(value.norm()), 1e-12))
                    heads.append({"axis": axis, "variant": stages[stage].variant,
                                  "repetition": int(stages[stage].repetition), "lineage": lineage, "stage": stage,
                                  "tensor": name, "relative_change": change})
    device = resolve_device(cache.settings, allow_cpu=True)
    probe = pd.concat([probe_predictions(cache, row, axis, device) for row in models.itertuples()], ignore_index=True)
    head = cache.collect("prediction", models)
    head = head[(head.group == "all") & (head.scope == "train_supported") & (head.population == "annotation_retrieval")
                & head.evaluation.isin(["test_combined", "heldout_image"])]
    versus = probe[["model_id", "evaluation", "metric", "value"]].merge(
        head[["model_id", "evaluation", "metric", "value"]], on=["model_id", "evaluation", "metric"],
        suffixes=("_probe", "_head")).join(models.set_index("model_id")[["axis", "repetition", "variant", "role",
                                                                         "lineage"]], on="model_id")
    versus["probe_minus_head"] = versus.value_probe - versus.value_head
    return {"drift": pd.DataFrame(drift), "similarity": pd.DataFrame(similarity), "head_change": pd.DataFrame(heads),
            "probe_versus_head": versus, "metadata": {"angle_space": "canonical u"}}


@register("stage_images", per_axis=True, scope="all")
def stage_images(cache: CampaignCache, axis: str) -> dict:
    """Held-out images of the baseline and of the best variant of every stage.

    The best variant of a stage is selected on the withheld test pixels
    (:func:`select_best_variant`); its representative repetition is shown. Tables:
    ``selection`` (variant ranking per stage), ``pixel_maps`` (per-pixel Masserstein and
    spectral angle), ``latent_rgb`` (first three principal components of each shown
    model's held-out codes, scaled to [0, 1]), ``class_maps`` (probability maps of the
    classes with the best and worst per-image AP of the baseline representative, with
    the METASPACE intensities and annotations) and ``channel_agreement`` (Pearson
    correlation of reconstructed and input ion images of every model, and its paired
    improvement).
    """
    models = cache.models[cache.models.axis == axis]
    shown = {"baseline": cache.cell_representative(_alias(cache, axis, campaign.BASELINE_VARIANT, None))}
    selections = []
    for stage in cache.stages:
        best, ranking = select_best_variant(cache, axis, stage)
        shown[stage] = cache.cell_representative(_alias(cache, axis, best, stage))
        selections.append(ranking)
    pixels = _rows(cache, "heldout_image")[["row", "dataset_id", "x", "y"]]
    maps, rgb = [], []
    seed = int(cache.settings.get("segmentation", {}).get("seed", 42))
    for label, row in shown.items():
        with np.load(cache.model_directory(row.model_id) / "heldout_image_pixels.npz") as archive:
            maps.append(pixels.assign(**_identity(row), shown_as=label, masserstein=archive["masserstein"],
                                      spectral_angle=archive["spectral_angle"]))
        codes = _latent_codes(cache, row, "heldout_image")[1]
        components = PCA(n_components=3, random_state=seed).fit_transform(codes)  # (N, 3)
        low, high = np.percentile(components, 1, axis=0), np.percentile(components, 99, axis=0)
        scaled = np.clip((components - low) / np.where(high > low, high - low, 1.0), 0.0, 1.0)  # (N, 3)
        rgb.append(pixels.assign(**_identity(row), shown_as=label, r=scaled[:, 0], g=scaled[:, 1], b=scaled[:, 2]))
    ## Classes with the best and worst per-image AP of the baseline representative
    options = cache.settings.get("class_selection", {})
    count, minimum = int(options.get("count", 3)), int(options.get("minimum_positive_pixels", 20))
    per_class = cache.model_table(shown["baseline"].model_id, "per_class")
    per_class = per_class[(per_class.evaluation == "heldout_image") & (per_class.group != "all")
                          & (per_class.population == "annotation_retrieval") & (per_class.positives >= minimum)]
    ordered = per_class.dropna(subset=["average_precision"]).sort_values("average_precision", ascending=False)
    chosen = pd.concat([ordered.head(count).assign(kind="best"), ordered.tail(count).assign(kind="worst")])
    ions = cache.axis_table(axis, "heldout_ions")
    metaspace = np.load(cache.population_root / "heldout_metaspace.npy", mmap_mode="r")
    targets = _available_targets(cache.population(axis, "heldout_image"))
    columns = {name: index for index, name in enumerate(cache.axis_table(axis, "classes").class_name)}
    class_maps = []
    for record in chosen.itertuples():
        selected = pixels.dataset_id.eq(record.group).to_numpy()
        column = columns[record.class_name]
        ion = ions[(ions.dataset_id == record.group) & (ions.class_name == record.class_name)]
        reference = (np.asarray(metaspace[selected, int(ion.ion_index.iloc[0])], dtype=np.float32)
                     if len(ion) else np.full(int(selected.sum()), np.nan, dtype=np.float32))
        for label, row in shown.items():
            probabilities = np.asarray(cache.model_array(row.model_id, "heldout_image_probabilities")[selected, column],
                                       dtype=np.float32)
            class_maps.append(pixels[selected].assign(**_identity(row), shown_as=label, kind=record.kind,
                                                      class_name=record.class_name, probability=probabilities,
                                                      metaspace=reference, annotated=targets[selected, column]))
    ## Reconstructed versus input ion images of every model
    input_ions = np.load(cache.population_root / axis_directory(cache.settings, axis) / "heldout_ion_input.npy",
                         mmap_mode="r")
    agreement = []
    in_axis = ions.in_axis.to_numpy(bool)
    for row in models.itertuples():
        reconstructed = cache.model_array(row.model_id, "heldout_image_ions")
        for dataset_id in pixels.dataset_id.unique():
            selected = pixels.dataset_id.eq(dataset_id).to_numpy()
            indices = ions[(ions.dataset_id == dataset_id) & in_axis].ion_index.to_numpy()
            left = np.asarray(input_ions[selected][:, indices], dtype=np.float64)
            right = np.asarray(reconstructed[selected][:, indices], dtype=np.float64)
            valid = (left.std(axis=0) > 0) & (right.std(axis=0) > 0)
            correlation = np.full(indices.size, np.nan)
            if valid.any():
                lc, rc = left[:, valid] - left[:, valid].mean(axis=0), right[:, valid] - right[:, valid].mean(axis=0)
                correlation[valid] = (lc * rc).sum(axis=0) / np.sqrt((lc ** 2).sum(axis=0) * (rc ** 2).sum(axis=0))
            agreement.append({**_identity(row), "dataset_id": dataset_id, "ions": int(indices.size),
                              "median_pearson": float(np.nanmedian(correlation)) if valid.any() else np.nan})
    agreement = pd.DataFrame(agreement)
    long = agreement.assign(family="ion_images", metric="median_pearson", evaluation=agreement.dataset_id, ranking=NO_RANKING,
                            value=agreement.median_pearson)
    return {"selection": pd.concat(selections, ignore_index=True), "pixel_maps": pd.concat(maps, ignore_index=True),
            "latent_rgb": pd.concat(rgb, ignore_index=True),
            "class_maps": pd.concat(class_maps, ignore_index=True) if class_maps else pd.DataFrame(),
            "channel_agreement": agreement,
            "channel_improvements": paired_improvements(_variants(long), _baseline(long), pair_on=UNIT),
            "metadata": {"shown": {label: row.model_id for label, row in shown.items()}}}


# --------------------------------------------------
# Section: part 4 — targeted mechanisms (rare and colliding classes)
# --------------------------------------------------

def _synthetic_classes(cache: CampaignCache, axis: str) -> pd.DataFrame:
    """Generator class flags of one axis aligned to the head columns (verified by name)."""
    directory = cache.synthetic_root / axis_directory(cache.settings, axis)
    synthetic = pd.read_csv(directory / "classes.csv")
    classes = cache.axis_table(axis, "classes")
    if tuple(synthetic.class_name) != tuple(classes.class_name):
        raise ValueError(f"Synthetic and head class columns of {axis} differ.")
    synthetic = synthetic.drop(columns="class_name").rename(columns={"eligible": "generator_eligible"})
    return classes[["class_index", "class_name", "mz", "class_set", "train_positives"]].merge(
        synthetic, on="class_index", validate="one_to_one")


def _class_level(cache: CampaignCache, models: pd.DataFrame, flags: pd.DataFrame,
                 evaluations: tuple[str, ...]) -> pd.DataFrame:
    """Per-class head metrics of every model with the generator flags."""
    per_class = cache.collect("per_class", models)
    per_class = per_class[(per_class.group == "all") & (per_class.population == "annotation_retrieval")
                          & per_class.evaluation.isin(evaluations)]
    return per_class.merge(flags.drop(columns=["class_name", "train_positives"]), on="class_index", how="left",
                           validate="many_to_one")


def _set_metrics(per_class: pd.DataFrame, sets: dict[str, pd.Series]) -> pd.DataFrame:
    """Macro head metrics over class sets (eligible classes only) as a metric table."""
    rows = []
    eligible = per_class[per_class.eligible.astype(bool)]
    for name, selector in sets.items():
        subset = eligible[selector.reindex(eligible.index).fillna(False).astype(bool)]
        grouped = subset.groupby([*IDENTITY, "evaluation"])
        for metric in ("average_precision", "recall", "precision", "f1", "roc_auc"):
            frame = grouped[metric].mean().rename("value").reset_index()
            rows.append(frame.assign(family="class_set", metric=metric, ranking=name,
                                     classes=grouped.size().to_numpy()))
    return pd.concat(rows, ignore_index=True)


def _matched_control(metrics: pd.DataFrame, pairs: dict[str, str]) -> pd.DataFrame:
    """Paired improvement of every treated variant over its matched control variant.

    Pairing unit: axis, stage and repetition; a control shares every construction
    element of the treated variant except the mechanism under test.
    """
    frames = []
    for treated, control in pairs.items():
        left = metrics[metrics.variant == treated]
        right = metrics[metrics.variant == control]
        if left.empty or right.empty:
            continue
        frame = paired_improvements(left, right, pair_on=("axis", "stage", "repetition"))
        frames.append(frame.assign(control=control))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


#: Defaults of :func:`rare_spectra`; ``settings.rare_spectra`` overrides them.
RARE_SPECTRA_DEFAULTS = {"stage": "unfrozen_head", "presence": 1e-3, "rare_frequency": 0.05,
                         "candidate_quantile": 0.98, "improved": 2, "worst": 1, "baseline_worst": 2,
                         "populations": ("train", "heldout_image")}


def bin_frequency(spectra: np.ndarray, presence: float, batch_size: int = 8192) -> np.ndarray:
    """Fraction of pixels in which every bin carries at least ``presence`` of the pixel's TIC.

    :param spectra: TIC-normalized spectra, shape ``(N, M)`` (may be memory-mapped).
    :type spectra: numpy.ndarray
    :param presence: Minimum TIC fraction of a present bin.
    :type presence: float
    :return: Occurrence frequency per bin, shape ``(M,)``.
    :rtype: numpy.ndarray
    """
    counts = np.zeros(spectra.shape[1], dtype=np.int64)  # (M,)
    for start in range(0, spectra.shape[0], batch_size):
        counts += (np.asarray(spectra[start:start + batch_size]) >= presence).sum(axis=0)
    return counts / max(spectra.shape[0], 1)


def rare_spectra(cache: CampaignCache, axis: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Spectra dominated by rarely occurring peaks: the baseline against every fine-tuned variant.

    A bin is rare when it is present (at least ``presence`` of the TIC) in fewer than
    ``rare_frequency`` of the training pixels; the rare mass of a pixel is its TIC fraction
    in rare bins and candidates are the pixels above the ``candidate_quantile`` of the rare
    mass of their population. Among the candidates, per variant representative of
    ``stage``: the pixels with the largest paired Masserstein improvement over the baseline
    representative, the variant's worst pixel, and the baseline's worst pixels (shared by
    every variant). The selected spectra are reconstructed by loading the representatives
    (full axis, a few pixels per model); the zoom centre is the strongest rare bin.

    :return: ``pixels`` (one row per variant, population and selected pixel) and ``spectra``
        (one row per variant, population, pixel and bin with ``input``, ``baseline_output``,
        ``output``, ``rare``).
    :rtype: tuple[pandas.DataFrame, pandas.DataFrame]
    """
    from ....models.model_loader import ModelLoader
    from .pretraining_campaign_inference import _forward

    options = {**RARE_SPECTRA_DEFAULTS, **cache.settings.get("rare_spectra", {})}
    device = resolve_device(cache.settings, allow_cpu=True)
    mass_axis = cache.axis_meta(axis)["mass_axis"]
    frequency = bin_frequency(cache.population(axis, "train")["spectra"], float(options["presence"]))  # (M,)
    rare = (frequency > 0) & (frequency < float(options["rare_frequency"]))  # (M,)
    baseline_row = cache.cell_representative(_alias(cache, axis, campaign.BASELINE_VARIANT, None))
    variant_rows = {variant: cache.cell_representative(_alias(cache, axis, variant, options["stage"]))
                    for variant in sorted(_variants(cache.models).variant.unique())}
    selected, needed = [], {}
    for population in options["populations"]:
        spectra = cache.population(axis, population)["spectra"]  # (N, M), memory-mapped
        identities = _rows(cache, population)
        rare_mass = np.zeros(spectra.shape[0])  # (N,)
        for start in range(0, spectra.shape[0], 8192):
            rare_mass[start:start + 8192] = np.asarray(spectra[start:start + 8192])[:, rare].sum(axis=1)
        candidates = np.flatnonzero(rare_mass >= np.quantile(rare_mass, float(options["candidate_quantile"])))
        baseline_cost = cache.model_pixels(baseline_row.model_id, population)["masserstein"][candidates]
        other_cost = other_baseline_cost(cache, axis, baseline_row, population)[candidates]
        shared = candidates[np.argsort(baseline_cost)[::-1][:int(options["baseline_worst"])]]
        for variant, row in variant_rows.items():
            cost = cache.model_pixels(row.model_id, population)["masserstein"][candidates]
            gain = baseline_cost - cost  # positive = the variant is better
            kinds = [("baseline_worst", shared),
                     ("most_improved", candidates[np.argsort(gain)[::-1][:int(options["improved"])]]),
                     ("variant_worst", candidates[np.argsort(cost)[::-1][:int(options["worst"])]])]
            for kind, rows in kinds:
                for rank, pixel in enumerate(rows):
                    position = int(np.flatnonzero(candidates == pixel)[0])
                    values = np.asarray(spectra[pixel], dtype=np.float32)  # (M,)
                    peak = int(np.flatnonzero(rare)[np.argmax(values[rare])])
                    selected.append({**_identity(row), "population": population, "pixel_kind": kind, "rank": rank,
                                     "row": int(pixel), "dataset_id": identities.dataset_id.iloc[pixel],
                                     "rare_mass": float(rare_mass[pixel]), "rare_peak_mz": float(mass_axis[peak]),
                                     "rare_peak_frequency": float(frequency[peak]),
                                     "masserstein_baseline": float(baseline_cost[position]),
                                     "masserstein_baseline_others": float(other_cost[position]),
                                     "masserstein": float(cost[position])})
                    needed.setdefault((population, int(pixel)), values)
    pixels = pd.DataFrame(selected)
    ## Reconstructions of the selected spectra, one model load per representative
    keys = sorted(needed)
    inputs = np.stack([needed[key] for key in keys])  # (P, M)
    outputs = {}
    for name, row in [(campaign.BASELINE_VARIANT, baseline_row), *variant_rows.items()]:
        model = ModelLoader.load_artifact(row.artifact, strict=True)[0].to(device).eval()
        outputs[name] = _forward(model, inputs, row.head, device)["reconstruction"]  # (P, M)
        del model
    frames = []
    for variant in variant_rows:
        shown = pixels[pixels.variant == variant][["population", "row"]].drop_duplicates()
        for population, pixel in shown.itertuples(index=False):
            position = keys.index((population, int(pixel)))
            frames.append(pd.DataFrame({"variant": variant, "population": population, "row": int(pixel),
                                        "bin": np.arange(mass_axis.size), "mz": mass_axis, "rare": rare,
                                        "input": inputs[position], "baseline_output": outputs[campaign.BASELINE_VARIANT][position],
                                        "output": outputs[variant][position]}))
    spectra = pd.concat(frames, ignore_index=True)
    ## REMARK: bins without signal in the input and both outputs carry no information and are dropped.
    spectra = spectra[(spectra[["input", "baseline_output", "output"]].abs().max(axis=1) >= 1e-6)]
    return pixels, spectra.round({"mz": 4, "input": 6, "baseline_output": 6, "output": 6})


@register("rare_classes", per_axis=True, scope="all", requires=("synthetic_reference", "inference"), version=3)
def rare_classes(cache: CampaignCache, axis: str) -> dict:
    """Does the rare-class quota improve exactly the classes it targets?

    Rare classes are the generator's targets (classes that received rare-bonus tokens:
    the least annotated quarter of the eligible classes). Tables: ``classes`` (flags,
    annotation counts, ranks), ``class_metrics`` (per model, evaluation and class),
    ``set_metrics`` (macro metrics over rare / other eligible classes per model),
    ``baseline_improvements`` and ``control_improvements`` (paired against the baseline
    and against the matched variant without the rare quota) with summaries,
    ``dose_response`` (per class and cell: mean AP improvement over the baseline against
    the annotation count), ``cutoff`` (classes just below versus just above the rare
    cutoff, which differ in the bonus but hardly in frequency), ``true_rank`` (median rank
    of annotated classes within their pixels, per class set), ``exposure`` and
    ``rare_spectra_pixels`` / ``rare_spectra`` (spectra dominated by rarely occurring peaks,
    the baseline against every fine-tuned variant on training and held-out pixels,
    :func:`rare_spectra`).
    """
    options = _comparison(cache)
    targeted = cache.settings.get("targeted", {}).get("rare", {})
    evaluations = tuple(targeted.get("evaluations", ("test_combined", "test_extended", "heldout_image")))
    models = cache.models[cache.models.axis == axis]
    flags = _synthetic_classes(cache, axis)
    per_class = _class_level(cache, models, flags, evaluations)
    generator_eligible = per_class.generator_eligible.astype(bool)
    sets = {"rare": per_class.rare.astype(bool) & generator_eligible,
            "other_eligible": ~per_class.rare.astype(bool) & generator_eligible}
    set_metrics = _set_metrics(per_class, sets)
    set_metrics = set_metrics.rename(columns={"ranking": "class_set"}).assign(ranking=lambda frame: frame.class_set)
    baseline_paired = paired_improvements(_variants(set_metrics), _baseline(set_metrics), pair_on=UNIT)
    control_paired = _matched_control(_variants(set_metrics), targeted.get("pairs", {}))
    ## Dose-response: per class improvement over the baseline against the annotation count
    ap = per_class[per_class.eligible.astype(bool)][[*IDENTITY, "evaluation", "class_index", "average_precision"]]
    ap = ap.assign(family="class", metric="average_precision", ranking=NO_RANKING,
                   evaluation=ap.evaluation + "|" + ap.class_index.astype(str), value=ap.average_precision)
    class_paired = paired_improvements(_variants(ap), _baseline(ap), pair_on=UNIT)
    class_paired[["evaluation", "class_index"]] = class_paired.evaluation.str.split("|", n=1, expand=True)
    class_paired["class_index"] = class_paired.class_index.astype(int)
    dose = class_paired.groupby([*GROUP, "evaluation", "class_index"]).improvement.agg(
        ["mean", "count"]).reset_index().merge(flags, on="class_index", how="left")
    ## Classes around the rare cutoff (ranked by annotation count among eligible classes)
    window = int(targeted.get("cutoff_window", 25))
    cutoff = flags.loc[flags.rare.astype(bool), "eligible_count_rank"].max()
    near = dose[(dose.eligible_count_rank > cutoff - window) & (dose.eligible_count_rank <= cutoff + window)]
    near = near.assign(side=np.where(near.eligible_count_rank <= cutoff, "rare_below_cutoff", "above_cutoff"))
    cutoff_table = near.groupby([*GROUP, "evaluation", "side"]).agg(
        mean_improvement=("mean", "mean"), classes=("class_index", "nunique"),
        mean_annotation_count=("annotation_count", "mean")).reset_index()
    ## Rank of annotated classes within their pixels
    ranks = []
    rare_columns = flags.rare.to_numpy(bool) & flags.generator_eligible.to_numpy(bool)
    other_columns = ~flags.rare.to_numpy(bool) & flags.generator_eligible.to_numpy(bool)
    for row in models.itertuples():
        for evaluation in ("test_combined", "heldout_image"):
            scores, labels, _ = _evaluation_scores(cache, row, axis, evaluation)
            rank = true_label_ranks(scores, labels)  # (N, C)
            for name, columns in (("rare", rare_columns), ("other_eligible", other_columns)):
                values = rank[:, columns]
                values = values[np.isfinite(values)]
                ranks.append({**_identity(row), "evaluation": evaluation, "class_set": name,
                              "annotated_entries": int(values.size),
                              "median_true_rank": float(np.median(values)) if values.size else np.nan,
                              "mean_true_rank": float(values.mean()) if values.size else np.nan})
    ranks = pd.DataFrame(ranks)
    exposure = pd.read_csv(cache.synthetic_root / axis_directory(cache.settings, axis) / "exposure.csv")
    spectra_pixels, spectra = rare_spectra(cache, axis)
    return {"rare_spectra_pixels": spectra_pixels, "rare_spectra": spectra, "classes": flags, "class_metrics": per_class, "set_metrics": set_metrics,
            "baseline_improvements": baseline_paired,
            "baseline_summary": summarize(baseline_paired, [*GROUP, "evaluation", "class_set", "metric"],
                                          confidence=options["confidence"]),
            "control_improvements": control_paired,
            "control_summary": summarize(control_paired, [*GROUP, "control", "evaluation", "class_set", "metric"],
                                         confidence=options["confidence"]) if len(control_paired) else pd.DataFrame(),
            "dose_response": dose, "cutoff": cutoff_table, "true_rank": ranks,
            "exposure": exposure[exposure.class_index.isin(flags.class_index[flags.rare.astype(bool)])],
            "metadata": {"rare_classes": int(flags.rare.sum()), "eligible_classes": int(flags.generator_eligible.sum()),
                         "cutoff_rank": float(cutoff), "cutoff_window": window, "pairs": targeted.get("pairs", {})}}


### REMARK: version 2 flips the sign of the false-positive lift improvement (lower lift is better).
@register("overlap_classes", per_axis=True, scope="all", requires=("synthetic_reference", "inference"), version=2)
def overlap_classes(cache: CampaignCache, axis: str) -> dict:
    """Does the overlap quota help to separate classes that collide in one bin?

    The overlap quota targets classes annotated at the same bin of the same training
    pixel (spectral collisions). Primary analysis, per collision pair and model: on
    pixels where exactly one class of the pair is annotated, the probability that the
    annotated class scores higher (discrimination), the logit margin, and the
    false-positive rate of the absent partner compared with pixels where both are absent.
    Secondary analysis: pixels grouped by their number of annotated classes (spatial
    co-occurrence), with the label-ranking average precision per group.

    Tables: ``pairs`` (collision pairs with support), ``pair_summary`` (per model and
    evaluation, pixel-weighted over pairs), ``pair_cells`` (per pair and cell, mean over
    repetitions), ``baseline_improvements``/``control_improvements`` (paired) with
    summaries, ``set_metrics`` (macro metrics over overlap / other eligible classes) and
    ``cooccurrence`` (LRAP per annotation-count group, paired against the baseline).
    """
    options = _comparison(cache)
    targeted = cache.settings.get("targeted", {}).get("overlap", {})
    minimum = int(targeted.get("minimum_pair_pixels", 5))
    edges = [int(value) for value in targeted.get("cooccurrence_edges", (1, 2, 4, 8))]
    models = cache.models[cache.models.axis == axis]
    flags = _synthetic_classes(cache, axis)
    pairs = pd.read_csv(cache.synthetic_root / axis_directory(cache.settings, axis) / "collision_pairs.csv")
    pair_array = pairs[["class_a", "class_b"]].to_numpy(np.int64)
    summaries, cells, cooccurrence = [], [], []
    for row in models.itertuples():
        for evaluation in ("test_combined", "heldout_image"):
            scores, labels, available = _evaluation_scores(cache, row, axis, evaluation)
            table = pairwise_discrimination(scores, labels, available, pair_array)
            table = table[table.pixels >= minimum]
            weights = table.pixels.to_numpy(float)
            record = {**_identity(row), "evaluation": evaluation, "pairs": int(len(table))}
            for metric in ("discrimination", "margin", "false_positive_rate_when_partner_present",
                           "false_positive_lift"):
                values = table[metric].to_numpy(float)
                finite = np.isfinite(values)
                record[metric] = float(np.average(values[finite], weights=weights[finite])) if finite.any() else np.nan
            summaries.append(record)
            cells.append(table.assign(**_identity(row), evaluation=evaluation))
            ## Spatial co-occurrence groups
            counts = labels.sum(axis=1)
            bounds = [*edges, np.inf]
            for low, high in zip(bounds[:-1], bounds[1:]):
                selected = (counts >= low) & (counts < high)
                if selected.sum() < 2:
                    continue
                cooccurrence.append({**_identity(row), "evaluation": evaluation,
                                     "labels_group": f"{low}-{int(high) - 1}" if np.isfinite(high) else f"{low}+",
                                     "pixels": int(selected.sum()),
                                     "value": float(label_ranking_average_precision_score(labels[selected],
                                                                                           scores[selected]))})
    summary = pd.DataFrame(summaries)
    long = summary.melt(id_vars=[*IDENTITY, "evaluation", "pairs"],
                        value_vars=["discrimination", "margin", "false_positive_rate_when_partner_present",
                                    "false_positive_lift"], var_name="metric").assign(family="collision", ranking=NO_RANKING)
    baseline_paired = paired_improvements(_variants(long), _baseline(long), pair_on=UNIT)
    control_paired = _matched_control(_variants(long), targeted.get("pairs", {}))
    pair_cells = pd.concat(cells, ignore_index=True).groupby(["axis", "variant", "stage", "evaluation", "class_a",
                                                              "class_b"]).agg(
        pixels=("pixels", "mean"), discrimination=("discrimination", "mean"), margin=("margin", "mean"),
        false_positive_lift=("false_positive_lift", "mean")).reset_index()
    ## Class-level macro metrics over the overlap set
    per_class = _class_level(cache, models, flags, ("test_combined", "heldout_image"))
    generator_eligible = per_class.generator_eligible.astype(bool)
    set_metrics = _set_metrics(per_class, {"overlap": per_class.overlap.astype(bool) & generator_eligible,
                                           "other_eligible": ~per_class.overlap.astype(bool) & generator_eligible})
    set_paired = paired_improvements(_variants(set_metrics), _baseline(set_metrics), pair_on=UNIT)
    set_control = _matched_control(_variants(set_metrics), targeted.get("pairs", {}))
    cooccurrence = pd.DataFrame(cooccurrence)
    cooccurrence_long = cooccurrence.assign(family="cooccurrence", metric="label_ranking_average_precision",
                                            ranking=cooccurrence.labels_group)
    cooccurrence_paired = paired_improvements(_variants(cooccurrence_long), _baseline(cooccurrence_long),
                                              pair_on=UNIT)
    return {"pairs": pairs, "pair_summary": summary, "pair_cells": pair_cells,
            "baseline_improvements": baseline_paired,
            "baseline_summary": summarize(baseline_paired, [*GROUP, "evaluation", "metric"],
                                          confidence=options["confidence"]),
            "control_improvements": control_paired,
            "control_summary": summarize(control_paired, [*GROUP, "control", "evaluation", "metric"],
                                         confidence=options["confidence"]) if len(control_paired) else pd.DataFrame(),
            "set_metrics": set_metrics,
            "set_summary": summarize(set_paired, [*GROUP, "evaluation", "ranking", "metric"],
                                     confidence=options["confidence"]),
            "set_control_summary": summarize(set_control, [*GROUP, "control", "evaluation", "ranking", "metric"],
                                             confidence=options["confidence"]) if len(set_control) else pd.DataFrame(),
            "cooccurrence": cooccurrence,
            "cooccurrence_summary": summarize(cooccurrence_paired, [*GROUP, "evaluation", "ranking"],
                                              confidence=options["confidence"]),
            "metadata": {"collision_pairs": int(len(pairs)), "minimum_pair_pixels": minimum,
                         "overlap_classes": int(flags.overlap.sum()), "pairs_controls": targeted.get("pairs", {}),
                         "cooccurrence_edges": edges}}


# --------------------------------------------------
# Section: part 5 — axis comparison
# --------------------------------------------------

def _common_window_metrics(cache: CampaignCache) -> pd.DataFrame:
    """Mean window quantities on the m/z windows shared by every axis, per model and population."""
    reference = tuple(cache.settings.get("reference_mass_range", (200.0, 900.0)))
    rows = []
    for axis_name in cache.axes:
        windows = _windows(cache, axis_name)
        common = ((windows.window_lower >= reference[0]) & (windows.window_upper <= reference[1])).to_numpy()
        for row in cache.axis_models(axis_name).itertuples():
            for population in ("test", "heldout_image"):
                with np.load(cache.model_directory(row.model_id) / f"{population}_pixels.npz") as archive:
                    contribution, within = archive["contribution"][:, common], archive["within"][:, common]
                with np.errstate(all="ignore"):
                    values = {"contribution": float(contribution.sum(axis=1).mean()),
                              "within": float(np.nanmean(within))}
                for metric, value in values.items():
                    rows.append({**_identity(row), "family": "common_windows", "metric": metric,
                                 "evaluation": population, "ranking": NO_RANKING, "value": value})
    return pd.DataFrame(rows)


def _common_class_metrics(cache: CampaignCache) -> pd.DataFrame:
    """Macro head metrics over the classes shared by every axis, per model and population."""
    class_sets = cache.shared_table("class_sets").set_index("class_name").class_set
    rows = []
    for axis_name in cache.axes:
        per_class = cache.collect("per_class", cache.axis_models(axis_name))
        per_class = per_class[(per_class.group == "all") & (per_class.population == "annotation_retrieval")
                              & per_class.evaluation.isin(["test_combined", "heldout_image"])
                              & per_class.eligible.astype(bool)]
        per_class = per_class[per_class.class_name.map(class_sets).eq("common")]
        grouped = per_class.groupby([*IDENTITY, "evaluation"])
        for metric in ("average_precision", "roc_auc"):
            frame = grouped[metric].mean().rename("value").reset_index()
            rows.append(frame.assign(family="common_classes", metric=metric, ranking="annotation_retrieval"))
    return pd.concat(rows, ignore_index=True)


def _axis_pairs(metrics: pd.DataFrame, axes: list[str]) -> pd.DataFrame:
    """Wide minus narrow axis, paired by variant, stage and repetition (same pixels, same seed)."""
    narrow, wide = axes[0], axes[-1]
    keys = ["variant", "stage", "repetition", *METRIC_KEYS]
    left = metrics[metrics.axis == narrow].set_index(keys).value
    right = metrics[metrics.axis == wide].set_index(keys).value
    frame = pd.concat([left.rename("narrow"), right.rename("wide")], axis=1, join="inner").reset_index()
    frame["difference"] = frame.wide - frame.narrow
    frame["improvement"] = frame.difference * frame.metric.map(metric_direction)
    return frame.assign(narrow_axis=narrow, wide_axis=wide)


def _benefit_interaction(metrics: pd.DataFrame, axes: list[str], confidence: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Improvement over the own-axis baseline on each axis and its axis difference.

    :return: Paired improvements per axis and the interaction (wide minus narrow
        improvement, paired by variant, stage and repetition) with its summary.
    """
    paired = improvements(metrics)
    narrow, wide = axes[0], axes[-1]
    keys = ["variant", "stage", "repetition", *METRIC_KEYS]
    left = paired[paired.axis == narrow].set_index(keys).improvement.rename("narrow_improvement")
    right = paired[paired.axis == wide].set_index(keys).improvement.rename("wide_improvement")
    interaction = pd.concat([left, right], axis=1, join="inner").reset_index()
    interaction["improvement"] = interaction.wide_improvement - interaction.narrow_improvement
    summary = summarize(interaction, ["variant", "stage", *METRIC_KEYS], confidence=confidence)
    return paired, summary


@register("axis_reconstruction_common", per_axis=False, scope="all")
def axis_reconstruction_common(cache: CampaignCache, axis: Optional[str]) -> dict:
    """Reconstruction on the shared m/z windows: every stage and variant on both axes.

    Global Masserstein is not comparable across axes; the shared windows are. Tables:
    ``metrics`` (per model), ``axis_differences`` (wide minus narrow, paired),
    ``axis_difference_summary``, ``improvements`` (over the own-axis baseline) and
    ``benefit_interaction`` (does the wide axis gain more from pretraining?).
    """
    confidence = _comparison(cache)["confidence"]
    metrics = _common_window_metrics(cache)
    differences = _axis_pairs(metrics, cache.axes)
    paired, interaction = _benefit_interaction(metrics, cache.axes, confidence)
    return {"metrics": metrics, "axis_differences": differences,
            "axis_difference_summary": summarize(differences, ["variant", "stage", *METRIC_KEYS],
                                                 confidence=confidence),
            "improvements": paired, "benefit_interaction": interaction,
            "metadata": {"reference_mass_range": list(cache.settings.get("reference_mass_range", (200.0, 900.0)))}}


@register("axis_prediction_common", per_axis=False, scope="all")
def axis_prediction_common(cache: CampaignCache, axis: Optional[str]) -> dict:
    """Head metrics on the classes shared by both axes, every stage and variant.

    Tables: ``metrics``, ``axis_differences``, ``axis_difference_summary``,
    ``improvements`` and ``benefit_interaction`` (as in the reconstruction comparison).
    """
    confidence = _comparison(cache)["confidence"]
    metrics = _common_class_metrics(cache)
    differences = _axis_pairs(metrics, cache.axes)
    paired, interaction = _benefit_interaction(metrics, cache.axes, confidence)
    return {"metrics": metrics, "axis_differences": differences,
            "axis_difference_summary": summarize(differences, ["variant", "stage", *METRIC_KEYS],
                                                 confidence=confidence),
            "improvements": paired, "benefit_interaction": interaction, "metadata": {}}


@register("axis_benefit", per_axis=False, scope="all")
def axis_benefit(cache: CampaignCache, axis: Optional[str]) -> dict:
    """Whether the gain from pretraining and fine-tuning depends on the m/z range.

    Combines the axis-comparable quantities (shared windows, shared classes) with the
    stage transitions: ``interaction`` (improvement over the own baseline, wide minus
    narrow) and ``transition_interaction`` (the same for the stage transitions).
    """
    confidence = _comparison(cache)["confidence"]
    metrics = pd.concat([_common_window_metrics(cache), _common_class_metrics(cache)], ignore_index=True)
    _, interaction = _benefit_interaction(metrics, cache.axes, confidence)
    variants = _variants(metrics)
    wide = variants.pivot_table(index=["axis", "variant", "repetition", *METRIC_KEYS], columns="stage",
                                values="value").reset_index()
    frames = []
    for name, (target, source) in {"frozen_minus_pretrained": ("frozen_head", "pretrained"),
                                   "unfrozen_minus_pretrained": ("unfrozen_head", "pretrained"),
                                   "unfrozen_minus_frozen": ("unfrozen_head", "frozen_head")}.items():
        if {target, source} <= set(wide.columns):
            frame = wide.assign(transition=name,
                                improvement=(wide[target] - wide[source]) * wide.metric.map(metric_direction))
            frames.append(frame[["axis", "variant", "repetition", "transition", *METRIC_KEYS, "improvement"]])
    transitions = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    transition_interaction = pd.DataFrame()
    if len(transitions):
        keys = ["variant", "repetition", "transition", *METRIC_KEYS]
        left = transitions[transitions.axis == cache.axes[0]].set_index(keys).improvement.rename("narrow")
        right = transitions[transitions.axis == cache.axes[-1]].set_index(keys).improvement.rename("wide")
        joined = pd.concat([left, right], axis=1, join="inner").reset_index()
        joined["improvement"] = joined.wide - joined.narrow
        transition_interaction = summarize(joined, ["variant", "transition", *METRIC_KEYS], confidence=confidence)
    return {"metrics": metrics, "interaction": interaction, "transitions": transitions,
            "transition_interaction": transition_interaction, "metadata": {}}


def _capacity_job(cache: CampaignCache, row: Any, samples: dict) -> list[dict]:
    """Dimension usage and TwoNN of one model's canonical codes (worker function)."""
    rows = []
    for population in ("test", "heldout_image"):
        _, u = _latent_codes(cache, row, population)
        targets = _available_targets(cache.population(row.axis, population))
        tables = geometry_tables(u, targets, samples[population],
                                 k=int(cache.settings.get("latent", {}).get("neighbours", 10)))
        values = tables["geometry"].set_index("metric").value
        two_nn = intrinsic_dimension_estimate(u, samples[population])
        for metric in ("effective_rank", "participation_ratio"):
            rows.append({**_identity(row), "family": "latent", "metric": metric, "evaluation": population,
                         "ranking": NO_RANKING, "value": float(values.get(metric, np.nan))})
        rows.append({**_identity(row), "family": "latent", "metric": "two_nn_intrinsic_dimension",
                     "evaluation": population, "ranking": NO_RANKING,
                     "value": float(two_nn) if np.isfinite(two_nn) and two_nn <= u.shape[1] else np.nan})
    return rows


@register("axis_latent_capacity", per_axis=False, scope="all")
def axis_latent_capacity(cache: CampaignCache, axis: Optional[str]) -> dict:
    """Latent capacity of the wide axis for every stage: dimension usage and TwoNN.

    Tables: ``metrics`` (per model), ``axis_differences`` (wide minus narrow, paired) with
    summary, and ``benefit_interaction``.
    """
    confidence = _comparison(cache)["confidence"]
    jobs = []
    for axis_name in cache.axes:
        samples = _latent_samples(cache, axis_name)
        jobs.extend(delayed(_capacity_job)(cache, row, samples) for row in cache.axis_models(axis_name).itertuples())
    metrics = pd.DataFrame([record for rows in Parallel(n_jobs=_workers(cache))(jobs) for record in rows])
    differences = _axis_pairs(metrics, cache.axes)
    _, interaction = _benefit_interaction(metrics, cache.axes, confidence)
    return {"metrics": metrics, "axis_differences": differences,
            "axis_difference_summary": summarize(differences, ["variant", "stage", *METRIC_KEYS],
                                                 confidence=confidence),
            "benefit_interaction": interaction, "metadata": {}}


# --------------------------------------------------
# Section: part 6 — thesis verdict
# --------------------------------------------------

@register("thesis_verdict", per_axis=False, scope="all", requires=("synthetic_reference", "inference"), version=2)
def thesis_verdict(cache: CampaignCache, axis: Optional[str]) -> dict:
    """Does synthetic pretraining improve on the real-data baseline, and where?

    ``verdict`` holds, per axis, stage and variant, the mean improvement of every key
    metric with its interval and a label: ``improves`` (interval above zero),
    ``worsens`` (below zero) or ``inconclusive``. ``targeted`` holds the same for the
    mechanism-specific class sets (rare classes against the matched control without
    the rare quota, overlap classes against the control without the overlap quota).
    ``model_values`` and ``value_summary`` give the absolute key metrics of every model,
    the real-only baseline included.
    """
    options = _comparison(cache)
    metrics = model_metrics(cache, cache.models, evaluations=("test_combined", "heldout_image"),
                            populations=("test", "heldout_image"))
    metrics = metrics[_is_key(metrics, options["key_metrics"])]
    summary = summarize(improvements(metrics), [*GROUP, *METRIC_KEYS], confidence=options["confidence"])

    def label(frame: pd.DataFrame) -> pd.DataFrame:
        return frame.assign(verdict=np.select([frame.ci_low > 0, frame.ci_high < 0], ["improves", "worsens"],
                                              default="inconclusive"))

    targeted = []
    for axis_name in cache.axes:
        models = cache.axis_models(axis_name)
        flags = _synthetic_classes(cache, axis_name)
        per_class = _class_level(cache, models, flags, ("test_combined", "heldout_image"))
        for mechanism, column in (("rare", "rare"), ("overlap", "overlap")):
            sets = _set_metrics(per_class, {mechanism: per_class[column].astype(bool)})
            sets = sets[sets.metric == "average_precision"]
            control = _matched_control(_variants(sets), cache.settings.get("targeted", {}).get(mechanism, {})
                                       .get("pairs", {}))
            if len(control):
                targeted.append(summarize(control, [*GROUP, "control", "evaluation", "metric"],
                                          confidence=options["confidence"]).assign(mechanism=mechanism))
    return {"verdict": label(summary), "model_values": metrics,
            "value_summary": summarize(metrics, [*GROUP, *METRIC_KEYS], value="value", confidence=options["confidence"]),
            "targeted": label(pd.concat(targeted, ignore_index=True)) if targeted else pd.DataFrame(),
            "metadata": {"confidence": options["confidence"], "key_metrics": options["key_metrics"]}}
