"""Aggregation helpers specific to binning-analysis records."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np


def normalize_intensity(values: np.ndarray, normalization: str) -> np.ndarray:
    """Return raw, TIC-normalized, or maximum-normalized intensity values."""
    array = np.asarray(values, dtype=np.float32)
    if normalization == "raw": return array
    denominator = float(np.sum(array)) if normalization == "tic" else float(np.max(array, initial=0.0)) if normalization == "max" else None
    if denominator is None: raise ValueError("normalization must be raw, tic, or max")
    return array / denominator if denominator else np.zeros_like(array)


def summarize_metric(values: Iterable[float]) -> dict[str, float]:
    """Summarize a finite per-spectrum metric distribution."""
    array = np.asarray(list(values), dtype=float); array = array[np.isfinite(array)]
    if not array.size: return {name: np.nan for name in ("mean", "std", "min", "q25", "median", "q75", "q90", "q95", "q99", "max")} | {"count": 0.0}
    return {"count": float(array.size), "mean": float(np.mean(array)), "std": float(np.std(array)), "min": float(np.min(array)), **{f"q{int(q * 100)}": float(np.quantile(array, q)) for q in (.25, .5, .75, .9, .95, .99)}, "median": float(np.median(array)), "max": float(np.max(array))}


def summaries(records: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Group long-form records by comparison, normalization, and metric."""
    grouped: dict[tuple[str, str, str], list[float]] = {}
    for record in records: grouped.setdefault((record["comparison"], record["normalization"], record["metric"]), []).append(record["value"])
    return [{"comparison": key[0], "normalization": key[1], "metric": key[2], **summarize_metric(values)} for key, values in grouped.items()]
