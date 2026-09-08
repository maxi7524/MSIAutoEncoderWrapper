"""Threshold-free molecular ranking with explicit annotation/evidence populations."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.special import logsumexp
from sklearn.metrics import average_precision_score, roc_auc_score

from ....data.annotation_evidence import NEGATIVE, POSITIVE, UNAVAILABLE
from ....utils.logger import get_custom_logger
from .metrics import probabilities_from_logits

logger = get_custom_logger(__name__)


def positive_scores(logits: np.ndarray) -> np.ndarray:
    """Return stable positive log odds for binary or N/P/U outputs.

    :param logits: Finite logits of shape ``(N, C)`` or ``(N, C, 3)``.
    :type logits: numpy.ndarray
    :return: Positive ranking scores, shape ``(N, C)``.
    :rtype: numpy.ndarray
    :raises ValueError: If dimensions or values are invalid.
    """
    values = np.asarray(logits, dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("Head logits must be finite.")
    if values.ndim == 2:
        return values
    if values.ndim != 3 or values.shape[-1] != 3:
        raise ValueError("Expected binary (N, C) or N/P/U (N, C, 3) logits.")
    # REMARK: This is log(p_P / (1-p_P)), monotone in the existing positive
    # softmax probability. Ranking log odds avoids sigmoid saturation ties.
    return values[..., POSITIVE] - logsumexp(values[..., [NEGATIVE, 2]], axis=-1)  # (N, C)


def ranking_tables(
    logits: np.ndarray,
    targets: np.ndarray,
    states: np.ndarray,
    train_counts: np.ndarray,
    class_names: tuple[str, ...],
) -> dict[str, pd.DataFrame]:
    """Measure ranking under two fixed evaluation populations, without a cutoff.

    :param logits: Active head outputs, ``(N, C)`` or ``(N, C, 3)``.
    :param targets: Original binary annotations, ``(N, C)``.
    :param states: Shared evidence states, ``(N, C)``; -1 excludes unavailable entries.
    :param train_counts: Observed training positives per class, ``(C,)``.
    :param class_names: Stable names in target-column order.
    :return: Aggregate, per-class and per-state diagnostic DataFrames.
    :rtype: dict[str, pandas.DataFrame]
    :raises ValueError: If shapes, labels or evidence semantics disagree.
    """
    scores = positive_scores(logits)  # (N, C)
    targets, states = np.asarray(targets), np.asarray(states)
    train_counts = np.asarray(train_counts)
    if (scores.shape != targets.shape or states.shape != targets.shape
            or len(class_names) != scores.shape[1] or train_counts.shape != (scores.shape[1],)):
        raise ValueError("Scores, annotations, states and class catalogue must align.")
    if not np.isin(targets, [0, 1]).all() or not np.isin(states, [-1, 0, 1, 2]).all():
        raise ValueError("Invalid binary annotations or evidence-state codes.")
    available = states != UNAVAILABLE  # (N, C)
    if not np.array_equal((states == POSITIVE)[available], targets[available].astype(bool)):
        raise ValueError("Evidence positives must preserve available annotations exactly.")
    probabilities = probabilities_from_logits(np.asarray(logits, dtype=np.float64), "multi_label")
    masks = {
        "annotation_retrieval": available,
        "operational_pn": (states == POSITIVE) | (states == NEGATIVE),
    }
    rows, aggregate, diagnostics = [], [], []
    # Evaluate every class, retaining undefined values and their denominators
    for population, mask in masks.items():
        population_rows = []
        for c, name in enumerate(class_names):
            y = targets[mask[:, c], c]  # (N_available,)
            s = scores[mask[:, c], c]  # (N_available,)
            positives, negatives = int(y.sum()), int(len(y) - y.sum())
            # REMARK: AP=1 for a positive-only class is algebraically valid but
            # carries no discrimination evidence; exclude it from macro ranking.
            eligible = bool(train_counts[c] > 0 and positives > 0 and negatives > 0)
            row = {
                "population": population, "class_index": c, "class_name": name,
                "train_positives": int(train_counts[c]), "positives": positives,
                "negatives": negatives, "available": len(y), "eligible": eligible,
                "frequency": "rare" if train_counts[c] < 10 else "medium" if train_counts[c] < 100 else "frequent",
                "prevalence": positives / len(y) if len(y) else np.nan,
                "average_precision": average_precision_score(y, s) if eligible else np.nan,
                "roc_auc": roc_auc_score(y, s) if eligible else np.nan,
            }
            row["ap_above_prevalence"] = row["average_precision"] - row["prevalence"]
            population_rows.append(row)
        rows.extend(population_rows)
        frame = pd.DataFrame(population_rows)
        for scope in ("train_supported", "rare", "medium", "frequent"):
            selected = frame[frame.eligible & ((frame.frequency == scope) if scope != "train_supported" else True)]
            for metric in ("average_precision", "roc_auc", "ap_above_prevalence"):
                aggregate.append({"population": population, "scope": scope, "metric": metric,
                                  "value": selected[metric].mean(), "classes": len(selected)})
    # Score spread is a diagnostic, not evidence of molecular calibration
    for state, label in ((1, "P"), (0, "N"), (2, "U")):
        selected = states == state  # (N, C)
        for quantity, array in (("positive_probability", probabilities), ("positive_log_odds", scores)):
            values = array[selected]  # (E_state,)
            for quantile in (0.0, 0.1, 0.5, 0.9, 1.0):
                diagnostics.append({"state": label, "quantity": quantity, "quantile": quantile,
                                    "value": np.quantile(values, quantile) if values.size else np.nan,
                                    "entries": values.size})
    logger.info("Evaluated %s classes in two ranking populations.", scores.shape[1])
    return {"prediction": pd.DataFrame(aggregate), "per_class": pd.DataFrame(rows),
            "score_diagnostics": pd.DataFrame(diagnostics)}


def generalization_gaps(per_class: pd.DataFrame) -> pd.DataFrame:
    """Compare splits on identical eligible class support within each model.

    :param per_class: Per-class records with model, split and population identity.
    :type per_class: pandas.DataFrame
    :return: Train-minus-held-out gaps, matched on class before averaging.
    :rtype: pandas.DataFrame
    """
    rows = []
    for (model_id, population), group in per_class.groupby(["model_id", "population"]):
        for held_out in ("validation", "test"):
            left = group[(group.split == "train") & group.eligible]
            right = group[(group.split == held_out) & group.eligible]
            paired = left.merge(right, on="class_index", suffixes=("_train", "_held"), validate="one_to_one")
            for metric in ("average_precision", "roc_auc"):
                rows.append({"model_id": model_id, "population": population, "held_out": held_out,
                             "metric": metric, "classes": len(paired),
                             "train_value": paired[f"{metric}_train"].mean(),
                             "held_out_value": paired[f"{metric}_held"].mean(),
                             "value": (paired[f"{metric}_train"] - paired[f"{metric}_held"]).mean()})
    return pd.DataFrame(rows)
