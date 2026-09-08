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


def stratified_metrics(per_class: pd.DataFrame, strata: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Aggregate fixed class strata per model, preserving coverage and undefined AP.

    :param per_class: Canonical per-class prediction records for every run and split.
    :type per_class: pandas.DataFrame
    :param strata: Output of :func:`class_strata`.
    :type strata: pandas.DataFrame
    :return: Enriched per-class records and long-form macro metrics with denominators.
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
        for metric in ("average_precision", "roc_auc", "ap_above_prevalence"):
            valid = joined[metric].where(joined.eligible)  # (R,)
            means = joined.assign(_value=valid).groupby([*identity, grouping], observed=True, dropna=False)._value.mean().reset_index(name="value")
            frame = counts.merge(means, on=[*identity, grouping], validate="one_to_one")
            frame = frame.rename(columns={grouping: "stratum"}).assign(grouping=grouping, metric=metric)
            outputs.append(frame)
    return joined, pd.concat(outputs, ignore_index=True)
