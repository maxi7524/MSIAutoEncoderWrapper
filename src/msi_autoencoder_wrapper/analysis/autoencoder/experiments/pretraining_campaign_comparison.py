"""Model-independent comparison helpers of the pretraining-campaign analyses.

Everything here operates on arrays or long-form tables and never reads the cache, so
every helper can be tested on synthetic data with known results.

Conventions:

* A *metric table* has one row per model and metric with the columns ``axis``,
  ``repetition``, ``variant``, ``stage``, ``family``, ``metric``, ``evaluation``,
  ``ranking`` and ``value``. ``ranking`` is the ranking population of head metrics and
  :data:`NO_RANKING` for every other metric.
* The *improvement* of a model is its paired difference to the baseline model of the
  same axis and repetition, signed so that a positive value is always better
  (:func:`metric_direction`). Repetitions share one initialization seed across all
  grid groups, which makes the difference a paired comparison.
* Summaries over repetitions report the mean, the standard deviation, a Student-t
  confidence interval and the number of repetitions that improve.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from scipy import stats

#: Metrics for which a lower value is better; every other metric is higher-is-better.
LOWER_IS_BETTER = frozenset({
    "masserstein", "mse", "mae", "spectral_angle", "tic_error", "hamming_loss", "expected_calibration_error",
    "brier_score", "median_true_rank", "mean_true_rank", "contribution", "within", "absolute_mass_imbalance",
    "false_positive_rate_when_partner_present", "false_positive_lift", "total_loss",
})

#: ``ranking`` value of metrics without a ranking population (explicit, so that it
#: survives a CSV round trip, where an empty string would become missing).
NO_RANKING = "none"

#: Columns identifying one paired comparison unit (a model's metric in one population).
METRIC_KEYS = ("family", "metric", "evaluation", "ranking")


# --------------------------------------------------
# Section: paired differences and summaries
# --------------------------------------------------

def metric_direction(metric: str) -> int:
    """Return ``-1`` for lower-is-better metrics and ``+1`` otherwise.

    :param metric: Metric name.
    :type metric: str
    :return: Sign that turns a difference into an improvement.
    :rtype: int
    """
    return -1 if metric in LOWER_IS_BETTER else 1


def paired_improvements(metrics: pd.DataFrame, reference: pd.DataFrame, *,
                        pair_on: Sequence[str] = ("axis", "repetition"),
                        keys: Sequence[str] = METRIC_KEYS) -> pd.DataFrame:
    """Pair every metric row with the reference row of the same pairing unit.

    :param metrics: Metric table of the compared models.
    :type metrics: pandas.DataFrame
    :param reference: Metric table of the reference models (e.g. the baselines), one
        row per pairing unit and metric key.
    :type reference: pandas.DataFrame
    :param pair_on: Columns defining the pairing unit.
    :type pair_on: collections.abc.Sequence[str]
    :param keys: Metric key columns.
    :type keys: collections.abc.Sequence[str]
    :return: ``metrics`` with ``reference_value``, ``delta`` (value minus reference),
        ``improvement`` (sign-adjusted delta) and ``relative_improvement``; rows without
        a reference are dropped.
    :rtype: pandas.DataFrame
    :raises ValueError: If the reference is not unique per pairing unit and key.
    """
    columns = [*pair_on, *keys]
    base = reference[[*columns, "value"]].rename(columns={"value": "reference_value"})
    if base.duplicated(columns).any():
        raise ValueError("The reference table holds more than one value per pairing unit and metric.")
    frame = metrics.merge(base, on=columns, how="inner", validate="many_to_one")
    frame["delta"] = frame.value - frame.reference_value
    sign = frame.metric.map(metric_direction).astype(float)
    frame["improvement"] = frame.delta * sign
    with np.errstate(divide="ignore", invalid="ignore"):
        frame["relative_improvement"] = frame.improvement / frame.reference_value.abs()
    return frame


def summarize(frame: pd.DataFrame, keys: Sequence[str], value: str = "improvement",
              confidence: float = 0.95) -> pd.DataFrame:
    """Mean, spread, Student-t interval and sign counts of ``value`` per group.

    :param frame: Long-form table.
    :type frame: pandas.DataFrame
    :param keys: Grouping columns.
    :type keys: collections.abc.Sequence[str]
    :param value: Column to summarize.
    :type value: str
    :param confidence: Two-sided confidence level of the interval.
    :type confidence: float
    :return: One row per group with ``n``, ``mean``, ``sd``, ``ci_low``, ``ci_high``,
        ``positive`` and ``negative`` (counts of strictly positive/negative values).
    :rtype: pandas.DataFrame
    """
    grouped = frame.groupby(list(keys), dropna=False, sort=False)[value]
    result = grouped.agg(n="count", mean="mean", sd="std", median="median").reset_index()
    signs = frame.assign(_positive=frame[value] > 0, _negative=frame[value] < 0).groupby(
        list(keys), dropna=False, sort=False).agg(positive=("_positive", "sum"), negative=("_negative", "sum"))
    result = result.merge(signs.reset_index(), on=list(keys), how="left")
    critical = np.where(result.n > 1, stats.t.ppf(0.5 + confidence / 2.0, np.maximum(result.n - 1, 1)), np.nan)
    half = critical * result.sd / np.sqrt(result.n)
    result["ci_low"], result["ci_high"] = result["mean"] - half, result["mean"] + half
    return result


def rank_variants(summary: pd.DataFrame, selection: Sequence[dict], *, evaluation: str,
                  tie_breaker: dict | None = None) -> pd.DataFrame:
    """Rank variants by their mean improvement over the selection metrics.

    :param summary: Output of :func:`summarize` on improvements with ``variant`` among
        the keys.
    :type summary: pandas.DataFrame
    :param selection: Metric keys (``family``, ``metric`` and optional ``ranking``).
    :type selection: collections.abc.Sequence[dict]
    :param evaluation: Population on which variants are ranked.
    :type evaluation: str
    :param tie_breaker: Metric key deciding equal mean ranks (default: the first key).
    :type tie_breaker: dict | None
    :return: One row per variant with ``mean_rank`` and ``rank`` (1 = best), sorted.
    :rtype: pandas.DataFrame
    """
    frames = []
    for spec in selection:
        selected = summary[(summary.family == spec["family"]) & (summary.metric == spec["metric"])
                           & (summary.ranking == spec.get("ranking", NO_RANKING)) & (summary.evaluation == evaluation)]
        frames.append(selected[["variant", "mean"]].assign(
            key=f"{spec['family']}:{spec['metric']}", rank=selected["mean"].rank(ascending=False, method="average")))
    ranks = pd.concat(frames, ignore_index=True)
    tie = tie_breaker or selection[0]
    tie_key = f"{tie['family']}:{tie['metric']}"
    table = ranks.groupby("variant").agg(mean_rank=("rank", "mean")).reset_index()
    table = table.merge(ranks[ranks.key == tie_key][["variant", "rank"]].rename(columns={"rank": "tie_rank"}),
                        on="variant", how="left")
    table = table.sort_values(["mean_rank", "tie_rank", "variant"]).reset_index(drop=True)
    table["rank"] = np.arange(1, len(table) + 1)
    return table


# --------------------------------------------------
# Section: contrasts between variants
# --------------------------------------------------

def variant_contrasts(metrics: pd.DataFrame, contrasts: dict[str, dict], *,
                      unit: Sequence[str] = ("axis", "stage", "repetition")) -> pd.DataFrame:
    """Paired contrasts ``mean(plus variants) - mean(minus variants)`` per repetition.

    :param metrics: Metric table with ``variant``.
    :type metrics: pandas.DataFrame
    :param contrasts: Name mapped to ``{"plus": [...], "minus": [...]}``.
    :type contrasts: dict[str, dict]
    :param unit: Columns of one paired unit.
    :type unit: collections.abc.Sequence[str]
    :return: One row per contrast, unit and metric key with ``plus``, ``minus``,
        ``difference`` and ``improvement``; contrasts with a missing variant are skipped.
    :rtype: pandas.DataFrame
    """
    frames = []
    available = set(metrics.variant)
    keys = [*unit, *METRIC_KEYS]
    for name, specification in contrasts.items():
        plus, minus = list(specification["plus"]), list(specification["minus"])
        if not set(plus + minus) <= available:
            continue
        side = {label: metrics[metrics.variant.isin(members)].groupby(keys).value.mean().rename(label)
                for label, members in (("plus", plus), ("minus", minus))}
        frame = pd.concat(side.values(), axis=1, join="inner").reset_index()
        frame["difference"] = frame.plus - frame.minus
        frame["improvement"] = frame.difference * frame.metric.map(metric_direction).astype(float)
        frames.append(frame.assign(contrast=name, plus_variants="+".join(plus), minus_variants="+".join(minus)))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def factorial_effects(metrics: pd.DataFrame, factors: dict[str, dict], levels: dict[str, list], *,
                      unit: Sequence[str] = ("axis", "stage", "repetition")) -> pd.DataFrame:
    """Main effects and two-factor interactions of a two-level factorial design.

    Each factor has a reference and a treatment level (``levels[factor] = [reference,
    treatment]``). The main effect is ``mean(treatment) - mean(reference)`` over the
    design cells; the interaction of two factors is the difference between the means
    of the cells whose two factors are both at reference or both at treatment and the
    cells where they differ (the usual +/- coded contrast, not halved).

    :param metrics: Metric table with ``variant``.
    :type metrics: pandas.DataFrame
    :param factors: Factor levels per variant (the settings ``factors`` of each variant).
    :type factors: dict[str, dict]
    :param levels: Two levels per factor, reference first.
    :type levels: dict[str, list]
    :param unit: Columns of one paired unit.
    :type unit: collections.abc.Sequence[str]
    :return: One row per effect, unit and metric key with ``effect`` (raw difference)
        and ``improvement`` (sign-adjusted); empty when the design is incomplete.
    :rtype: pandas.DataFrame
    """
    design = {variant: values for variant, values in factors.items()
              if all(values.get(factor) in pair for factor, pair in levels.items())}
    selected = metrics[metrics.variant.isin(design)]
    if selected.empty:
        return pd.DataFrame()
    codes = pd.DataFrame({factor: {variant: 1.0 if values[factor] == levels[factor][1] else -1.0
                                   for variant, values in design.items()} for factor in levels})
    keys = [*unit, *METRIC_KEYS]
    effects = [(factor, codes[factor]) for factor in levels]
    names = list(levels)
    effects += [(f"{left}:{right}", codes[left] * codes[right])
                for position, left in enumerate(names) for right in names[position + 1:]]
    frames = []
    for name, code in effects:
        signed = selected.assign(code=selected.variant.map(code))
        positive = signed[signed.code > 0].groupby(keys).value.mean()
        negative = signed[signed.code < 0].groupby(keys).value.mean()
        frame = (positive - negative).rename("effect").dropna().reset_index()
        frame["improvement"] = frame.effect * frame.metric.map(metric_direction).astype(float)
        frames.append(frame.assign(term=name, order=1 if ":" not in name else 2))
    return pd.concat(frames, ignore_index=True)


# --------------------------------------------------
# Section: head scores
# --------------------------------------------------

def logits_from_probabilities(probabilities: np.ndarray, epsilon: float = 1e-4) -> np.ndarray:
    """Logits of stored (float16) probabilities, clipped away from 0 and 1.

    :param probabilities: Probabilities, any shape.
    :type probabilities: numpy.ndarray
    :param epsilon: Clipping margin; float16 resolves about ``5e-4`` near one.
    :type epsilon: float
    :return: Logits in float64.
    :rtype: numpy.ndarray
    """
    values = np.clip(np.asarray(probabilities, dtype=np.float64), epsilon, 1.0 - epsilon)
    return np.log(values) - np.log1p(-values)


def calibration_table(probabilities: np.ndarray, labels: np.ndarray, bins: int = 10) -> tuple[pd.DataFrame, dict]:
    """Reliability bins, expected calibration error and Brier score of multi-label scores.

    Every (pixel, class) entry is one binary prediction. With incomplete annotations the
    observed positive rate is a lower bound of the true one, so the calibration error is
    relative to the annotations.

    :param probabilities: Predicted probabilities ``(N, C)``.
    :type probabilities: numpy.ndarray
    :param labels: Binary annotations ``(N, C)``.
    :type labels: numpy.ndarray
    :param bins: Number of equal-width probability bins.
    :type bins: int
    :return: Per-bin ``entries``, ``mean_probability``, ``positive_rate`` and the
        summary ``expected_calibration_error``, ``brier_score``, ``mean_probability``,
        ``positive_rate`` and ``entries``.
    :rtype: tuple[pandas.DataFrame, dict]
    """
    values = np.asarray(probabilities, dtype=np.float64).ravel()
    truth = np.asarray(labels, dtype=np.float64).ravel()
    edges = np.linspace(0.0, 1.0, bins + 1)
    index = np.clip(np.digitize(values, edges[1:-1]), 0, bins - 1)
    counts = np.bincount(index, minlength=bins).astype(np.float64)
    confidence = np.bincount(index, weights=values, minlength=bins)
    positives = np.bincount(index, weights=truth, minlength=bins)
    with np.errstate(divide="ignore", invalid="ignore"):
        mean_probability = confidence / counts
        positive_rate = positives / counts
    table = pd.DataFrame({"bin": np.arange(bins), "lower": edges[:-1], "upper": edges[1:], "entries": counts,
                          "mean_probability": mean_probability, "positive_rate": positive_rate})
    total = max(values.size, 1)
    ece = float(np.nansum(np.abs(mean_probability - positive_rate) * counts) / total)
    summary = {"expected_calibration_error": ece, "brier_score": float(np.mean((values - truth) ** 2)),
               "mean_probability": float(values.mean()), "positive_rate": float(truth.mean()),
               "entries": int(values.size)}
    return table, summary


def true_label_ranks(scores: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Rank (1 = highest score) of every annotated class within its pixel.

    :param scores: Scores ``(N, C)``.
    :type scores: numpy.ndarray
    :param labels: Binary annotations ``(N, C)``.
    :type labels: numpy.ndarray
    :return: Ranks ``(N, C)`` as float with ``nan`` where the class is not annotated;
        ties receive the lowest (best) rank of the tie.
    :rtype: numpy.ndarray
    """
    values = np.asarray(scores, dtype=np.float64)
    order = np.argsort(-values, axis=1, kind="stable")
    ranks = np.empty_like(order)
    rows = np.arange(values.shape[0])[:, None]
    ranks[rows, order] = np.arange(1, values.shape[1] + 1)[None, :]
    ## Ties: the best rank among equal scores
    sorted_values = np.take_along_axis(values, order, axis=1)
    first = np.maximum.accumulate(np.where(np.diff(sorted_values, axis=1, prepend=np.inf) != 0,
                                           np.arange(values.shape[1])[None, :], 0), axis=1) + 1
    tied = np.empty_like(first)
    tied[rows, order] = first
    result = tied.astype(np.float64)
    result[~np.asarray(labels, dtype=bool)] = np.nan
    return result


def pairwise_discrimination(scores: np.ndarray, labels: np.ndarray, available: np.ndarray, pairs: np.ndarray, *,
                            chunk: int = 256) -> pd.DataFrame:
    """Separation of colliding classes on pixels where exactly one of them is annotated.

    For a pair ``(a, b)`` and the pixels where ``a`` is annotated and ``b`` is available
    but not annotated, the discrimination is ``P(s_a > s_b) + 0.5 P(s_a = s_b)`` and the
    margin is the mean of ``s_a - s_b``; the same is computed with the roles swapped, and
    the two directions are pooled with pixel weights. The false-positive rate of the
    absent partner (``s_b > 0``) is reported next to its rate on pixels where neither
    class is annotated, so ``false_positive_lift`` measures confusion caused by the
    present partner.

    :param scores: Logits ``(N, C)``.
    :type scores: numpy.ndarray
    :param labels: Binary annotations ``(N, C)``.
    :type labels: numpy.ndarray
    :param available: Availability of every entry ``(N, C)`` (evaluable annotation state).
    :type available: numpy.ndarray
    :param pairs: Class index pairs ``(P, 2)``.
    :type pairs: numpy.ndarray
    :param chunk: Pairs processed at once (memory ``N * chunk``).
    :type chunk: int
    :return: One row per pair with ``pixels`` (pixels with exactly one of the pair),
        ``discrimination``, ``margin``, ``false_positive_rate_when_partner_present``,
        ``false_positive_rate_when_both_absent`` and ``false_positive_lift``.
    :rtype: pandas.DataFrame
    """
    values = np.asarray(scores, dtype=np.float32)
    truth = np.asarray(labels, dtype=bool)
    usable = np.asarray(available, dtype=bool)
    pairs = np.asarray(pairs, dtype=np.int64).reshape(-1, 2)
    frames = []
    for start in range(0, len(pairs), chunk):
        block = pairs[start:start + chunk]
        a, b = block[:, 0], block[:, 1]
        totals = {"pixels": 0.0, "wins": 0.0, "margin": 0.0, "fp_present": 0.0, "fp_present_n": 0.0}
        for left, right in ((a, b), (b, a)):
            present = truth[:, left] & usable[:, right] & ~truth[:, right]  # (N, P)
            s_left, s_right = values[:, left], values[:, right]  # (N, P)
            totals["pixels"] = totals["pixels"] + present.sum(axis=0)
            wins = (s_left > s_right).astype(np.float32) + 0.5 * (s_left == s_right)  # (N, P)
            totals["wins"] = totals["wins"] + (wins * present).sum(axis=0)
            totals["margin"] = totals["margin"] + ((s_left - s_right) * present).sum(axis=0)
            totals["fp_present"] = totals["fp_present"] + ((s_right > 0) & present).sum(axis=0)
            totals["fp_present_n"] = totals["fp_present_n"] + present.sum(axis=0)
        neither = usable[:, a] & usable[:, b] & ~truth[:, a] & ~truth[:, b]  # (N, P)
        false_absent = (((values[:, a] > 0) & neither).sum(axis=0) + ((values[:, b] > 0) & neither).sum(axis=0))
        absent_n = 2.0 * neither.sum(axis=0)
        with np.errstate(divide="ignore", invalid="ignore"):
            frame = pd.DataFrame({
                "class_a": a, "class_b": b, "pixels": totals["pixels"].astype(np.int64),
                "discrimination": totals["wins"] / totals["pixels"], "margin": totals["margin"] / totals["pixels"],
                "false_positive_rate_when_partner_present": totals["fp_present"] / totals["fp_present_n"],
                "false_positive_rate_when_both_absent": false_absent / absent_n})
        frame["false_positive_lift"] = (frame.false_positive_rate_when_partner_present
                                        - frame.false_positive_rate_when_both_absent)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


# --------------------------------------------------
# Section: representations
# --------------------------------------------------

def ridge_probe(train_features: np.ndarray, train_targets: np.ndarray, evaluation_features: Iterable[np.ndarray],
                regularization: float = 1e-3) -> list[np.ndarray]:
    """Linear (ridge) probe of multi-label targets from latent codes.

    Features are standardized with the training statistics and a bias column is added;
    the weights solve ``(X^T X + lambda n I) W = X^T Y`` (bias not penalized). The probe
    scores rank classes; they are not calibrated probabilities.

    :param train_features: Training codes ``(N, D)``.
    :type train_features: numpy.ndarray
    :param train_targets: Training annotations ``(N, C)``.
    :type train_targets: numpy.ndarray
    :param evaluation_features: Codes ``(N_e, D)`` of every evaluation population.
    :type evaluation_features: collections.abc.Iterable[numpy.ndarray]
    :param regularization: Ridge strength relative to the number of training rows.
    :type regularization: float
    :return: Scores ``(N_e, C)`` per evaluation population.
    :rtype: list[numpy.ndarray]
    """
    x = np.asarray(train_features, dtype=np.float64)
    mean, scale = x.mean(axis=0), x.std(axis=0)
    scale = np.where(scale > 0, scale, 1.0)

    def design(values: np.ndarray) -> np.ndarray:
        standardized = (np.asarray(values, dtype=np.float64) - mean) / scale  # (N, D)
        return np.hstack([standardized, np.ones((standardized.shape[0], 1))])  # (N, D + 1)

    features = design(x)
    penalty = regularization * features.shape[0] * np.eye(features.shape[1])
    penalty[-1, -1] = 0.0
    weights = np.linalg.solve(features.T @ features + penalty,
                              features.T @ np.asarray(train_targets, dtype=np.float64))  # (D + 1, C)
    return [design(values) @ weights for values in evaluation_features]


def angle_between(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Angle in degrees between corresponding rows of two code matrices.

    :param left: Codes ``(N, D)``.
    :type left: numpy.ndarray
    :param right: Codes ``(N, D)``.
    :type right: numpy.ndarray
    :return: Angles ``(N,)``.
    :rtype: numpy.ndarray
    """
    a, b = np.asarray(left, dtype=np.float64), np.asarray(right, dtype=np.float64)
    a = a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-12)  # (N, D)
    b = b / np.maximum(np.linalg.norm(b, axis=1, keepdims=True), 1e-12)  # (N, D)
    ## REMARK: 2 atan2(|a - b|, |a + b|) keeps full precision for small angles, where
    ## arccos of the cosine loses about half of the significant digits.
    return np.degrees(2.0 * np.arctan2(np.linalg.norm(a - b, axis=1), np.linalg.norm(a + b, axis=1)))
