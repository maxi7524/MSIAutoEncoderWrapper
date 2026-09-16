"""Relate head prediction quality to historical and training-only ion strata."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ....utils.logger import get_custom_logger

logger = get_custom_logger(__name__)


def class_strata(prevalence: pd.DataFrame, regimes: pd.DataFrame, separation: pd.DataFrame,
                 train_states: pd.DataFrame) -> pd.DataFrame:
    """Join part-0 ion descriptions by identity and derive separate train-only strata.

    :param prevalence: Part-0 class_prevalence records, one row per class_name.
    :type prevalence: pandas.DataFrame
    :param regimes: Historical part-0 class_regimes at one recorded evidence rule.
    :type regimes: pandas.DataFrame
    :param separation: Part-0 class_separation with evidence_auc.
    :type separation: pandas.DataFrame
    :param train_states: Counts at the campaign evidence rule, computed on train only.
    :type train_states: pandas.DataFrame
    :return: Aligned taxonomy, thresholds and continuous class properties.
    :rtype: pandas.DataFrame
    :raises ValueError: If class identities are duplicated, missing or inconsistent.
    """
    frames = [prevalence, regimes, separation, train_states]
    reference = set(train_states.class_name)
    if any(frame.class_name.duplicated().any() for frame in frames):
        raise ValueError("Class strata require exactly one record per class identity.")
    if any(set(frame.class_name) != reference for frame in frames):
        raise ValueError("Part-0 and training class catalogues differ; do not join by position.")
    if len(regimes[["relative_threshold", "bin_radius"]].drop_duplicates()) != 1:
        raise ValueError("Historical regimes must describe one explicit evidence rule.")
    result = train_states.copy()
    result = result.merge(prevalence[["class_name", "mz", "prevalence"]].rename(columns={"prevalence": "part0_prevalence"}),
                          on="class_name", validate="one_to_one")
    result = result.merge(regimes[["class_name", "regime", "relative_threshold", "bin_radius", "negative_fraction_of_unannotated"]].rename(
        columns={"regime": "part0_regime", "relative_threshold": "part0_threshold", "bin_radius": "part0_radius",
                 "negative_fraction_of_unannotated": "part0_negative_fraction"}), on="class_name", validate="one_to_one")
    result = result.merge(separation[["class_name", "evidence_auc"]], on="class_name", validate="one_to_one")
    # Keep historical full-population descriptions separate from train-only strata.
    total = result.positive_entries + result.negative_entries + result.uncertain_entries  # (C,)
    unannotated = result.negative_entries + result.uncertain_entries  # (C,)
    result["train_prevalence"] = result.positive_entries / total.replace(0, np.nan)
    result["train_negative_fraction"] = result.negative_entries / unannotated.replace(0, np.nan)
    result["train_regime"] = np.select(
        [result.train_negative_fraction >= .95, result.train_prevalence < .01],
        ["undetectable", "rare"], default="common")
    result.loc[total == 0, "train_regime"] = "unavailable"
    result["train_support"] = np.select([result.positive_entries == 0, result.positive_entries < 10,
                                         result.positive_entries < 100],
                                        ["absent", "1-9 positives", "10-99 positives"], default="100+ positives")
    result["part0_evidence_separation"] = np.select(
        [result.evidence_auc.isna(), result.evidence_auc <= .5, result.evidence_auc <= .7],
        ["undefined", "AUC <= 0.5", "0.5 < AUC <= 0.7"], default="AUC > 0.7")
    lower = (np.floor(result.mz / 100) * 100).astype(int)  # (C,)
    result["mz_window"] = lower.astype(str) + "-" + (lower + 100).astype(str) + " m/z"
    logger.info("Joined %s class identities to historical and train-only strata.", len(result))
    return result


#: Threshold-free ranking metrics, macro-averaged over eligible classes in a stratum.
RANKING_METRICS = ("average_precision", "roc_auc", "ap_above_prevalence")
#: Threshold-dependent per-class metrics (already gated to threshold-anchored,
#: eligible classes by `predictive_comparison.ranking_tables`), macro-averaged the
#: same way as :data:`RANKING_METRICS`.
THRESHOLD_MACRO_METRICS = ("precision", "recall", "f1")
#: Micro (pooled-confusion-count) counterparts of :data:`THRESHOLD_MACRO_METRICS`,
#: derived per stratum from the raw `true_positive`/`false_positive`/`false_negative`
#: columns rather than from the macro ratios (see :func:`stratified_metrics`).
THRESHOLD_MICRO_METRICS = ("micro_precision", "micro_recall", "micro_f1", "hamming_loss")


def stratified_metrics(per_class: pd.DataFrame, strata: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Aggregate fixed class strata per model, preserving coverage and undefined AP.

    Alongside the macro-averaged ranking metrics, this also reports macro
    precision/recall/F1 and their micro (pooled-confusion-count) counterparts plus
    Hamming loss, mirroring the global threshold-based metrics
    `predictive_reports`/`part_4_prediction_global.ipynb` report at campaign scope,
    but broken down by an arbitrary class stratum instead. Both families are ``nan``
    for classes/models `ranking_tables` did not gate as threshold-anchored, so a
    stratum containing only non-threshold-anchored conditions correctly yields
    ``nan`` here too, not a spurious zero.

    :param per_class: Canonical per-class prediction records for every run and split.
    :type per_class: pandas.DataFrame
    :param strata: Output of :func:`class_strata`.
    :type strata: pandas.DataFrame
    :return: Enriched per-class records and long-form macro/micro metrics with
        denominators.
    :rtype: tuple[pandas.DataFrame, pandas.DataFrame]
    :raises ValueError: If any scored class lacks a stratum.
    """
    joined = per_class.merge(strata, on="class_name", how="left", validate="many_to_one", indicator=True)
    if not (joined._merge == "both").all():
        raise ValueError("Every evaluated class must have a verified stratum.")
    joined = joined.drop(columns="_merge")
    identity = ["model_id", "label", "condition", "repetition", "split", "population"]
    outputs = []
    for grouping in ("part0_regime", "train_regime", "train_support", "part0_evidence_separation", "mz_window"):
        groups = joined.groupby([*identity, grouping], observed=True, dropna=False)
        counts = groups.agg(total_classes=("class_name", "size"), eligible_classes=("eligible", "sum"),
                            positive_entries_evaluated=("positives", "sum"),
                            negative_entries_evaluated=("negatives", "sum")).reset_index()
        for metric in (*RANKING_METRICS, *THRESHOLD_MACRO_METRICS):
            valid = joined[metric].where(joined.eligible)  # (R,)
            means = joined.assign(_value=valid).groupby([*identity, grouping], observed=True, dropna=False)._value.mean().reset_index(name="value")
            frame = counts.merge(means, on=[*identity, grouping], validate="one_to_one")
            frame = frame.rename(columns={grouping: "stratum"}).assign(grouping=grouping, metric=metric)
            outputs.append(frame)
        ## Micro threshold metrics: pool raw confusion counts over eligible classes in
        ## the stratum, rather than re-deriving them from the macro ratios above
        ## (lossy whenever a class's precision is exactly 0, see `ranking_tables`).
        pooled = joined.assign(
            _tp=joined.true_positive.where(joined.eligible), _fp=joined.false_positive.where(joined.eligible),
            _fn=joined.false_negative.where(joined.eligible), _available=joined.available.where(joined.eligible),
        ).groupby([*identity, grouping], observed=True, dropna=False).agg(
            _tp_sum=("_tp", "sum"), _fp_sum=("_fp", "sum"), _fn_sum=("_fn", "sum"),
            _available_sum=("_available", "sum"), _defined=("_tp", "count")).reset_index()
        defined = (pooled._defined > 0).to_numpy()  # (G,); False when no class in the stratum is threshold-anchored
        true_positive = pooled._tp_sum.to_numpy()
        false_positive = pooled._fp_sum.to_numpy()
        false_negative = pooled._fn_sum.to_numpy()
        available = pooled._available_sum.to_numpy()
        micro_precision = np.divide(true_positive, true_positive + false_positive, out=np.full(len(pooled), np.nan),
                                    where=defined & ((true_positive + false_positive) > 0))
        micro_recall = np.divide(true_positive, true_positive + false_negative, out=np.full(len(pooled), np.nan),
                                 where=defined & ((true_positive + false_negative) > 0))
        micro_f1 = np.divide(2 * micro_precision * micro_recall, micro_precision + micro_recall,
                             out=np.full(len(pooled), np.nan),
                             where=defined & ((micro_precision + micro_recall) > 0))
        hamming_loss = np.divide(false_positive + false_negative, available, out=np.full(len(pooled), np.nan),
                                 where=defined & (available > 0))
        for metric, values in zip(THRESHOLD_MICRO_METRICS, (micro_precision, micro_recall, micro_f1, hamming_loss)):
            micro_frame = pooled[[*identity, grouping]].assign(value=values)
            frame = counts.merge(micro_frame, on=[*identity, grouping], validate="one_to_one")
            frame = frame.rename(columns={grouping: "stratum"}).assign(grouping=grouping, metric=metric)
            outputs.append(frame)
    return joined, pd.concat(outputs, ignore_index=True)
