"""Anatomical partition scores with explicit availability and noise semantics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.ndimage import distance_transform_edt
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score

from ....utils.logger import get_custom_logger

logger = get_custom_logger(__name__)


@dataclass(frozen=True)
class PartitionScores:
    """Retain scalar scores, class overlap, contingency and score-only matching."""

    summary: dict[str, float | int | None]
    regions: pd.DataFrame
    contingency: pd.DataFrame
    aligned: np.ndarray


def score_partition(reference: np.ndarray, clusters: np.ndarray, *, noise_id: int = -1) -> PartitionScores:
    """Evaluate a partition on exactly the pixels with available reference labels.

    Reference -1 means unavailable. Predicted noise is retained in ARI/AMI as one
    declared cluster and penalized in matched accuracy/Dice. Matching maximizes
    intersection using a rectangular assignment; extra clusters remain unmatched.
    It is used only for scoring, never to change predictions. Complexity is
    O(N log N + R*C + min(R,C)^2*max(R,C)), memory O(N + R*C).

    :param reference: Integer anatomical labels, shape ``(N,)``.
    :param clusters: Integer predicted cluster IDs, shape ``(N,)``.
    :param noise_id: Cluster value representing unassigned pixels.
    :return: Scores and tables with class/cluster IDs preserved.
    :raises ValueError: For invalid arrays, values or unavailable reference.
    """
    truth, pred = np.asarray(reference), np.asarray(clusters)
    if truth.ndim != 1 or pred.shape != truth.shape or truth.dtype.kind not in "iu" or pred.dtype.kind not in "iu" or np.any(truth < -1):
        raise ValueError("Reference and clusters must be equal-length integer vectors; reference IDs >= -1.")
    available = truth >= 0  # (N,)
    if not available.any():
        raise ValueError("No reference labels available for scoring.")
    y, p = truth[available], pred[available]  # (K,), (K,)
    classes, yi = np.unique(y, return_inverse=True)  # (R,), (K,)
    cluster_ids, pi = np.unique(p, return_inverse=True)  # (C,), (K,)
    counts = np.zeros((len(classes), len(cluster_ids)), dtype=np.int64)  # (R, C)
    np.add.at(counts, (yi, pi), 1)
    nonnoise = cluster_ids != noise_id  # (C,)
    columns = np.flatnonzero(nonnoise)  # (C_without_noise,)
    aligned = np.full(len(truth), -2, dtype=np.int64)  # (N,)
    mappings: dict[int, int] = {}
    if len(columns):
        rows, selected = linear_sum_assignment(-counts[:, columns])  # (J,), (J,)
        mappings = {int(cluster_ids[columns[col]]): int(classes[row]) for row, col in zip(rows, selected, strict=True)}
        for cluster, label in mappings.items():
            aligned[pred == cluster] = label
    records = []
    for label in classes:
        observed = available & (truth == label)  # (N,)
        predicted = available & (aligned == label)  # (N,)
        tp = int(np.sum(observed & predicted))
        actual, selected = int(observed.sum()), int(predicted.sum())
        records.append({"reference_class": int(label), "reference_pixels": actual, "predicted_pixels": selected,
                        "intersection": tp, "region_recall": tp / actual,
                        "region_precision": tp / selected if selected else 0.0,
                        "dice": 2 * tp / (actual + selected), "iou": tp / (actual + selected - tp),
                        "matched_cluster": next((c for c, v in mappings.items() if v == label), None)})
    regions = pd.DataFrame(records)
    covered = p != noise_id  # (K,)
    summary = {"reference_pixels": int(available.sum()), "reference_coverage": float(available.mean()),
               "reference_classes": len(classes), "predicted_clusters": int(nonnoise.sum()),
               "noise_fraction": float((~covered).mean()), "ari": float(adjusted_rand_score(y, p)),
               "ami": float(adjusted_mutual_info_score(y, p)),
               "ari_covered": float(adjusted_rand_score(y[covered], p[covered])) if covered.sum() >= 2 else None,
               "covered_fraction": float(covered.mean()),
               "matched_accuracy": float((aligned[available] == y).mean()),
               "macro_dice": float(regions.dice.mean()), "macro_iou": float(regions.iou.mean())}
    logger.debug("Scored %s reference pixels, %s classes and %s predicted clusters.", len(y), len(classes), int(nonnoise.sum()))
    return PartitionScores(summary, regions, pd.DataFrame(counts, index=classes, columns=cluster_ids), aligned)


def _boundary(labels: np.ndarray, available: np.ndarray) -> np.ndarray:
    """Find both endpoints of label-changing four-neighbour edges."""
    boundary = np.zeros(labels.shape, dtype=bool)  # (H, W)
    for left, right in [(np.s_[:-1, :], np.s_[1:, :]), (np.s_[:, :-1], np.s_[:, 1:])]:
        differs = available[left] & available[right] & (labels[left] != labels[right])  # (H-1, W) or (H, W-1)
        boundary[left] |= differs
        boundary[right] |= differs
    return boundary


def score_boundaries(reference: np.ndarray, aligned: np.ndarray, *, tolerance: float,
                     spacing: tuple[float, float] = (1.0, 1.0)) -> tuple[dict[str, float | int | None], pd.DataFrame]:
    """Score boundary agreement and class-specific confident interior accuracy.

    Geometry is a registered regular raster. Unknown reference pixels never define
    boundaries; predicted unmatched/noise pixels remain an error category. With
    spacing (1,1), distance is in raster pixels. Complexity O(H*W), memory O(H*W).

    :param reference: Reference raster, shape ``(H, W)``, -1 unavailable.
    :param aligned: Score-only matched predictions, shape ``(H, W)``.
    :param tolerance: Nonnegative boundary matching distance in spacing units.
    :param spacing: Positive row/column spacing in the same units as tolerance.
    :return: Boundary summary and interior accuracy table by reference class.
    :raises ValueError: For malformed rasters, spacing or tolerance.
    """
    truth, pred = np.asarray(reference), np.asarray(aligned)
    if truth.ndim != 2 or pred.shape != truth.shape or truth.dtype.kind not in "iu" or pred.dtype.kind not in "iu":
        raise ValueError("Boundary evaluation requires equal integer rasters.")
    if not np.isfinite(tolerance) or tolerance < 0 or len(spacing) != 2 or not np.isfinite(spacing).all() or min(spacing) <= 0:
        raise ValueError("Boundary tolerance and spacing are invalid.")
    available = truth >= 0  # (H, W)
    ref_boundary, pred_boundary = _boundary(truth, available), _boundary(pred, available)  # (H, W), (H, W)
    ref_count, pred_count = int(ref_boundary.sum()), int(pred_boundary.sum())
    ref_distance = distance_transform_edt(~ref_boundary, sampling=spacing) if ref_count else np.full(truth.shape, np.inf)  # (H, W)
    pred_distance = distance_transform_edt(~pred_boundary, sampling=spacing) if pred_count else np.full(truth.shape, np.inf)  # (H, W)
    precision = float(np.mean(ref_distance[pred_boundary] <= tolerance)) if pred_count else (0.0 if ref_count else None)
    recall = float(np.mean(pred_distance[ref_boundary] <= tolerance)) if ref_count else None
    f1 = None if precision is None or recall is None else (2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    records = []
    for label in np.unique(truth[available]):
        interior = available & (truth == label) & (ref_distance > tolerance)  # (H, W)
        count = int(interior.sum())
        records.append({"reference_class": int(label), "interior_pixels": count,
                        "interior_accuracy": float(np.mean(pred[interior] == label)) if count else None})
    return {"reference_boundary_pixels": ref_count, "predicted_boundary_pixels": pred_count,
            "boundary_precision": precision, "boundary_recall": recall, "boundary_f1": f1,
            "boundary_tolerance": float(tolerance)}, pd.DataFrame(records)
