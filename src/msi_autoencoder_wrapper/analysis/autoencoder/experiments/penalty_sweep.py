"""Grid identity and training dynamics of regularization-penalty sweep campaigns.

A penalty sweep varies the ``regularization.contractive`` objective across a grid of
penalty metrics, input geometries and weights, holding the architecture, binning and
head configuration fixed. This is the complementary campaign shape to the one
:mod:`..reconstruction.training_dynamics_analysis` handles: that module binds to
``architecture_grid_parameters`` and therefore requires ``architectures`` x
``binning_steps`` axes, which a penalty sweep does not have (it would raise
``ValidationError`` on every task here). The per-epoch filtering rule is the same and
is deliberately kept identical; what differs is the grid key and the retention of
every individual objective term rather than only ``total_loss``.

Two questions this module answers without loading a single model:

1. did every run actually train (loss terms finite, penalty term responding, planned
   epochs reached), so a failed or degenerate cell is not silently read as a result;
2. how much wall time each penalty configuration costs per epoch, since spectral and
   exact-Jacobian estimators are materially more expensive than a Hutchinson
   Frobenius estimate and that cost is part of the decision.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Dict, Optional

from ....utils.logger import get_custom_logger
from .campaign_reader import CampaignTask

logger = get_custom_logger(__name__)

# Objective terms are campaign-specific (one key per configured criterion), so they are
# discovered from the history records rather than enumerated here. These keys are the
# per-epoch bookkeeping fields that are not objective values.
_NON_OBJECTIVE_METRIC_KEYS = frozenset(
    {"epoch", "duration", "checkpoint_scope", "is_best", "best_loss"}
)

# ContractiveLoss constructor defaults, applied when a campaign predates a parameter.
# Mirrors msi_autoencoder_wrapper.training.criterions.autoencoder.regularization
# .contractive_loss.MSIContractiveLoss.__init__.
_PENALTY_DEFAULTS = {
    "penalty_metric": "frobenius",
    "input_geometry": "euclidean",
    "hinge_alpha": 1.0,
    "hinge_threshold": None,
    "penalized_space": None,
    "calculation_method": None,
}

BASELINE_PENALTY_METRIC = "none"


@dataclass(frozen=True)
class PenaltySweepCell:
    """Semantic identity of one grid cell of a penalty sweep.

    A cell is the configuration shared by every repetition of one grid point. Two
    cells from different campaigns are the same cell if and only if every field here
    matches, which is what makes cross-campaign pooling safe.

    :param penalty_metric: ``frobenius``, ``spectral``, ``hinged``,
        ``spectral_plus_hinged``, or :data:`BASELINE_PENALTY_METRIC` when the task
        configures no contractive penalty at all.
    :type penalty_metric: str
    :param input_geometry: ``euclidean`` or ``fisher_rao``. Campaigns predating this
        parameter are reported as ``euclidean``, matching the constructor default.
    :type input_geometry: str
    :param weight: Objective weight of the contractive term, ``None`` for a baseline.
    :type weight: float | None
    :param hinge_threshold: Threshold ``tau``, ``None`` for non-hinged penalties.
    :type hinge_threshold: float | None
    :param hinge_alpha: Excess-sensitivity coefficient of ``spectral_plus_hinged``.
    :type hinge_alpha: float
    :param penalized_space: Latent space the penalty is applied in (e.g. ``u``).
    :type penalized_space: str | None
    :param calculation_method: Jacobian estimator, e.g. ``exact_autograd_jacobian`` or
        ``approximate_hutchinson_vjp``. Materially affects epoch duration.
    :type calculation_method: str | None
    """

    penalty_metric: str
    input_geometry: str
    weight: Optional[float]
    hinge_threshold: Optional[float]
    hinge_alpha: float
    penalized_space: Optional[str]
    calculation_method: Optional[str]

    def __post_init__(self) -> None:
        """Normalize absent numeric fields so a cell has one canonical identity.

        REMARK: a record that has been through a pandas frame carries ``nan`` where
        this class was constructed with ``None``, and ``nan is not None`` is true, so
        an unnormalized cell built from such a record produced a different label than
        the same cell built from the task manifest (``... / tau=nan``). Any lookup
        keyed on the label then failed. Normalizing here fixes identity, equality and
        hashing at once, rather than only the label.
        """
        for field_name in ("weight", "hinge_threshold"):
            value = getattr(self, field_name)
            if value is not None and not math.isfinite(value):
                object.__setattr__(self, field_name, None)
        if self.hinge_alpha is None or not math.isfinite(self.hinge_alpha):
            object.__setattr__(self, "hinge_alpha", _PENALTY_DEFAULTS["hinge_alpha"])

    @property
    def is_baseline(self) -> bool:
        """Whether this cell configures no contractive penalty."""
        return self.penalty_metric == BASELINE_PENALTY_METRIC

    @property
    def label(self) -> str:
        """Semantic plot label, e.g. ``fisher_rao / spectral / w=1e-03``.

        REMARK: the reproducibility contract forbids opaque labels such as
        ``grid_06``; every figure axis in the sweep notebooks uses this string so a
        reader can identify the configuration without consulting the campaign config.
        """
        if self.is_baseline:
            return "baseline (no contractive)"
        parts = [self.input_geometry, self.penalty_metric]
        parts.append("w=unset" if self.weight is None else f"w={self.weight:.0e}")
        if self.hinge_threshold is not None:
            parts.append(f"tau={self.hinge_threshold:g}")
        return " / ".join(parts)


def penalty_parameters(task: CampaignTask) -> Optional[Dict[str, Any]]:
    """Return the raw ``regularization.contractive`` mapping of one task, if any.

    :param task: Campaign task with populated ``grid_parameters``.
    :type task: CampaignTask
    :return: The contractive objective mapping (``target``, ``weight``, ``params``),
        or ``None`` when the task configures no contractive regularization.
    :rtype: Dict[str, Any] | None
    """
    objectives = task.grid_parameters.get("objectives")
    if not isinstance(objectives, dict):
        return None
    regularization = objectives.get("regularization")
    if not isinstance(regularization, dict):
        return None
    contractive = regularization.get("contractive")
    return contractive if isinstance(contractive, dict) else None


def sweep_cell(task: CampaignTask) -> PenaltySweepCell:
    """Resolve one task's semantic grid cell from its recorded grid parameters.

    Missing penalty parameters fall back to the ``ContractiveLoss`` constructor
    defaults rather than to ``None``, so a campaign that predates a parameter
    (notably ``input_geometry``) is pooled with later campaigns that set it
    explicitly to the same effective value.

    :param task: Campaign task with populated ``grid_parameters``.
    :type task: CampaignTask
    :return: The task's grid-cell identity.
    :rtype: PenaltySweepCell
    """
    contractive = penalty_parameters(task)
    if contractive is None:
        return PenaltySweepCell(
            penalty_metric=BASELINE_PENALTY_METRIC,
            input_geometry=_PENALTY_DEFAULTS["input_geometry"],
            weight=None,
            hinge_threshold=None,
            hinge_alpha=_PENALTY_DEFAULTS["hinge_alpha"],
            penalized_space=None,
            calculation_method=None,
        )

    params = contractive.get("params") or {}
    resolved = {key: params.get(key, default) for key, default in _PENALTY_DEFAULTS.items()}
    weight = contractive.get("weight")
    threshold = resolved["hinge_threshold"]
    return PenaltySweepCell(
        penalty_metric=str(resolved["penalty_metric"]),
        input_geometry=str(resolved["input_geometry"]),
        weight=None if weight is None else float(weight),
        hinge_threshold=None if threshold is None else float(threshold),
        hinge_alpha=float(resolved["hinge_alpha"]),
        penalized_space=resolved["penalized_space"],
        calculation_method=resolved["calculation_method"],
    )


def sweep_grid_frame(
    tasks: Sequence[CampaignTask], campaign_id: str
) -> list[Dict[str, Any]]:
    """Flatten a campaign's tasks into one long-form grid-coverage record list.

    One row per task, whatever its status, so incomplete or failed cells stay visible
    instead of being silently dropped by a completed-only filter.

    :param tasks: Tasks of one campaign (see
        :func:`~.entropy_status_reader.read_entropy_campaign`).
    :type tasks: Sequence[CampaignTask]
    :param campaign_id: Campaign identifier, carried into every row so several
        campaigns can be concatenated and still be distinguishable.
    :type campaign_id: str
    :return: Records with ``campaign``, ``task_id``, ``model_name``, ``status``,
        ``repetition``, ``cell_label`` and every :class:`PenaltySweepCell` field.
    :rtype: list[Dict[str, Any]]
    """
    rows: list[Dict[str, Any]] = []
    for task in tasks:
        cell = sweep_cell(task)
        rows.append(
            {
                "campaign": campaign_id,
                "task_id": task.task_id,
                "model_name": f"{campaign_id}__{task.task_id}",
                "status": task.status,
                "repetition": task.repetition,
                "cell_label": cell.label,
                "penalty_metric": cell.penalty_metric,
                "input_geometry": cell.input_geometry,
                "weight": cell.weight,
                "hinge_threshold": cell.hinge_threshold,
                "hinge_alpha": cell.hinge_alpha,
                "penalized_space": cell.penalized_space,
                "calculation_method": cell.calculation_method,
                "recorded_epochs": (task.result or {}).get("epochs"),
            }
        )
    completed = sum(1 for row in rows if row["status"] == "completed")
    logger.info(
        "Campaign '%s': %s task(s), %s completed, %s distinct grid cell(s).",
        campaign_id,
        len(rows),
        completed,
        len({row["cell_label"] for row in rows}),
    )
    return rows


def training_dynamics_frame(
    tasks: Sequence[CampaignTask], campaign_id: str
) -> list[Dict[str, Any]]:
    """Flatten every task's per-epoch history into long-form records, keyed by cell.

    Only ``history`` entries carrying a ``metrics.epoch`` value are included; the
    trailing post-training evaluation entry (``split: test``, no ``epoch``/
    ``duration``) is excluded by the same rule
    :func:`~..reconstruction.training_dynamics_analysis.training_curves_frame` uses,
    and is available separately through :func:`final_evaluation_frame`.

    Unlike that function, every objective term present in the record is retained as
    its own row rather than collapsed into ``total_loss``: the point of a penalty
    sweep is precisely how the penalty term behaves relative to the others.

    :param tasks: Campaign tasks with loaded ``history`` (``load_artifacts=True``).
    :type tasks: Sequence[CampaignTask]
    :param campaign_id: Campaign identifier carried into every row.
    :type campaign_id: str
    :return: Records with ``campaign``, ``task_id``, ``cell_label``, penalty-cell
        fields, ``repetition``, ``epoch``, ``duration`` (seconds), ``objective``
        (term name, e.g. ``contractive``), ``split`` (``train``/``validation``),
        ``value``, ``is_best`` and ``best_loss``.
    :rtype: list[Dict[str, Any]]
    """
    rows: list[Dict[str, Any]] = []
    tasks_without_history = 0

    # Per-task traversal
    for task in tasks:
        if not task.history:
            tasks_without_history += 1
            continue
        cell = sweep_cell(task)
        cell_fields = {
            "campaign": campaign_id,
            "task_id": task.task_id,
            "model_name": f"{campaign_id}__{task.task_id}",
            "cell_label": cell.label,
            "penalty_metric": cell.penalty_metric,
            "input_geometry": cell.input_geometry,
            "weight": cell.weight,
            "calculation_method": cell.calculation_method,
            "repetition": task.repetition,
        }

        ## Per-epoch traversal
        for entry in task.history:
            metrics = entry.get("metrics", {})
            epoch = metrics.get("epoch")
            if epoch is None:
                continue

            ### One row per objective term, split into train and validation series
            for key, value in metrics.items():
                if key in _NON_OBJECTIVE_METRIC_KEYS:
                    continue
                is_validation = key.startswith("validation_")
                objective = key[len("validation_") :] if is_validation else key
                rows.append(
                    {
                        **cell_fields,
                        "epoch": int(epoch),
                        "duration": metrics.get("duration"),
                        "objective": objective,
                        "split": "validation" if is_validation else "train",
                        "value": None if value is None else float(value),
                        "is_best": metrics.get("is_best"),
                        "best_loss": metrics.get("best_loss"),
                    }
                )

    if tasks_without_history:
        logger.warning(
            "%s task(s) had no loaded training history and were skipped; "
            "read the campaign with load_artifacts=True to include them.",
            tasks_without_history,
        )
    logger.info(
        "Campaign '%s': built %s per-epoch objective row(s) from %s task(s).",
        campaign_id,
        len(rows),
        len(tasks) - tasks_without_history,
    )
    return rows


def final_evaluation_frame(
    tasks: Sequence[CampaignTask], campaign_id: str
) -> list[Dict[str, Any]]:
    """Extract the trailing post-training evaluation record of every task.

    The training loop appends one non-epoch entry per evaluated split (``split:
    test``) after training finishes. It is the model's objective value under the
    restored best checkpoint, which is not recoverable from the epoch curve.

    :param tasks: Campaign tasks with loaded ``history``.
    :type tasks: Sequence[CampaignTask]
    :param campaign_id: Campaign identifier carried into every row.
    :type campaign_id: str
    :return: Records with ``campaign``, ``task_id``, ``cell_label``, penalty-cell
        fields, ``repetition``, ``split``, ``objective`` and ``value``.
    :rtype: list[Dict[str, Any]]
    """
    rows: list[Dict[str, Any]] = []
    for task in tasks:
        if not task.history:
            continue
        cell = sweep_cell(task)
        for entry in task.history:
            metrics = entry.get("metrics", {})
            if metrics.get("epoch") is not None:
                continue
            split = entry.get("split")
            if split is None:
                continue
            for objective, value in metrics.items():
                if objective in _NON_OBJECTIVE_METRIC_KEYS:
                    continue
                rows.append(
                    {
                        "campaign": campaign_id,
                        "task_id": task.task_id,
                        "model_name": f"{campaign_id}__{task.task_id}",
                        "cell_label": cell.label,
                        "penalty_metric": cell.penalty_metric,
                        "input_geometry": cell.input_geometry,
                        "weight": cell.weight,
                        "repetition": task.repetition,
                        "split": str(split),
                        "objective": objective,
                        "value": None if value is None else float(value),
                    }
                )
    logger.info(
        "Campaign '%s': %s post-training evaluation row(s).", campaign_id, len(rows)
    )
    return rows


def run_duration_frame(dynamics_rows: Sequence[Dict[str, Any]]) -> list[Dict[str, Any]]:
    """Reduce per-epoch rows to one duration record per training run.

    Durations are recorded once per epoch, but :func:`training_dynamics_frame` emits
    one row per objective term, so the same epoch's duration appears several times.
    This function deduplicates on ``(task_id, epoch)`` before summing, which is the
    only correct reduction of that layout.

    :param dynamics_rows: Output of :func:`training_dynamics_frame`.
    :type dynamics_rows: Sequence[Dict[str, Any]]
    :return: One record per task with ``campaign``, ``task_id``, ``cell_label``,
        penalty-cell fields, ``repetition``, ``epochs`` (measured epoch count),
        ``total_duration`` and ``mean_epoch_duration`` (seconds).
    :rtype: list[Dict[str, Any]]
    """
    ## Deduplicate the per-objective fan-out back to one measurement per epoch
    ### REMARK: keyed by model_name, not task_id: task identifiers restart at
    ### task_000000 in every campaign, so pooling two campaigns on task_id alone
    ### silently merges unrelated runs.
    durations_by_task: dict[str, dict[int, float]] = {}
    identity_by_task: dict[str, Dict[str, Any]] = {}
    for row in dynamics_rows:
        duration = row.get("duration")
        if duration is None:
            continue
        run_key = row["model_name"]
        durations_by_task.setdefault(run_key, {})[int(row["epoch"])] = float(duration)
        identity_by_task.setdefault(
            run_key,
            {
                "campaign": row["campaign"],
                "task_id": row["task_id"],
                "model_name": row["model_name"],
                "cell_label": row["cell_label"],
                "penalty_metric": row["penalty_metric"],
                "input_geometry": row["input_geometry"],
                "weight": row["weight"],
                "calculation_method": row["calculation_method"],
                "repetition": row["repetition"],
            },
        )

    records = []
    for run_key, per_epoch in sorted(durations_by_task.items()):
        values = list(per_epoch.values())
        records.append(
            {
                **identity_by_task[run_key],
                "epochs": len(values),
                "total_duration": float(sum(values)),
                "mean_epoch_duration": float(sum(values) / len(values)),
            }
        )
    logger.info("Reduced per-epoch rows to %s run duration record(s).", len(records))
    return records


def epoch_duration_samples(
    dynamics_rows: Sequence[Dict[str, Any]],
) -> Dict[str, list[float]]:
    """Group individual epoch durations by grid cell, preserving every measurement.

    Returned as raw samples rather than as a mean/standard-deviation summary because
    the reproducibility contract requires distributions to be plotted from complete
    observations.

    :param dynamics_rows: Output of :func:`training_dynamics_frame`.
    :type dynamics_rows: Sequence[Dict[str, Any]]
    :return: Mapping from ``cell_label`` to every measured epoch duration in seconds,
        pooled across repetitions.
    :rtype: Dict[str, list[float]]
    """
    seen: set[tuple[str, int]] = set()
    samples: Dict[str, list[float]] = {}
    for row in dynamics_rows:
        duration = row.get("duration")
        if duration is None:
            continue
        key = (row["model_name"], int(row["epoch"]))
        if key in seen:
            continue
        seen.add(key)
        samples.setdefault(row["cell_label"], []).append(float(duration))
    return samples


def paired_contrast(
    treatment: Dict[int, float], reference: Dict[int, float]
) -> Dict[str, Any]:
    """Compare a swept cell against a reference cell, pairing on repetition index.

    Repetitions of two cells share their split and dataloader seeds and differ only
    in the run seeds derived from the repetition index, so repetition ``r`` of one
    cell and repetition ``r`` of another are matched observations. Pairing removes the
    between-repetition variance that an unpaired comparison would leave in the
    residual, which matters at five repetitions.

    The interval is a two-sided 95% Student-t interval on the paired differences. It
    is not a bootstrap: with five pairs a bootstrap resamples at most five distinct
    values and understates the tails.

    :param treatment: Metric value per repetition for the evaluated cell.
    :type treatment: Dict[int, float]
    :param reference: Metric value per repetition for the reference cell.
    :type reference: Dict[int, float]
    :return: ``pairs`` (number of matched repetitions), ``mean_difference``,
        ``ci_low``, ``ci_high`` and ``separated_from_zero``. Endpoints are ``None``
        when fewer than two pairs are available.
    :rtype: Dict[str, Any]
    """
    shared = sorted(set(treatment) & set(reference))
    differences = [
        float(treatment[repetition]) - float(reference[repetition])
        for repetition in shared
        if math.isfinite(treatment[repetition]) and math.isfinite(reference[repetition])
    ]
    if not differences:
        return {
            "pairs": 0,
            "mean_difference": None,
            "ci_low": None,
            "ci_high": None,
            "separated_from_zero": False,
        }

    mean_difference = sum(differences) / len(differences)
    if len(differences) < 2:
        return {
            "pairs": len(differences),
            "mean_difference": mean_difference,
            "ci_low": None,
            "ci_high": None,
            "separated_from_zero": False,
        }

    # Deferred import: scipy is only needed for the interval, and keeping it local
    # matches how the benchmarking module treats the same dependency.
    from scipy.stats import t as student_t

    variance = sum((value - mean_difference) ** 2 for value in differences) / (
        len(differences) - 1
    )
    deviation = math.sqrt(variance)
    margin = (
        float(student_t.ppf(0.975, len(differences) - 1))
        * deviation
        / math.sqrt(len(differences))
    )
    low, high = mean_difference - margin, mean_difference + margin
    return {
        "pairs": len(differences),
        "mean_difference": mean_difference,
        "ci_low": low,
        "ci_high": high,
        "separated_from_zero": bool(low > 0.0 or high < 0.0),
    }


def training_health_report(
    dynamics_rows: Sequence[Dict[str, Any]],
    grid_rows: Sequence[Dict[str, Any]],
    planned_epochs: Optional[int] = None,
) -> list[Dict[str, Any]]:
    """Flag runs whose training did not behave, one record per task.

    Checks applied per run, each of which invalidates that run's results in a
    different way:

    - ``non_finite_objectives``: any recorded objective value is NaN or infinite,
      which means the run's remaining epochs are meaningless;
    - ``penalty_increased``: the contractive term's final training value exceeds its
      first, i.e. the penalty was not actually minimized. This is expected to be
      informative at large weights, where the penalty can dominate and destabilize;
    - ``short_run``: fewer measured epochs than ``planned_epochs``, i.e. early
      stopping or a crash;
    - ``no_improvement``: the trainer never recorded an improving checkpoint after
      the first epoch, so the restored best model is the first epoch's.

    :param dynamics_rows: Output of :func:`training_dynamics_frame`.
    :type dynamics_rows: Sequence[Dict[str, Any]]
    :param grid_rows: Output of :func:`sweep_grid_frame`, used to report cells whose
        tasks produced no history at all.
    :type grid_rows: Sequence[Dict[str, Any]]
    :param planned_epochs: Configured epoch budget; when ``None`` the maximum
        measured epoch count across the campaign is used as the reference.
    :type planned_epochs: int | None
    :return: One record per task with the boolean flags above, ``epochs``,
        ``first_penalty``/``final_penalty`` (training-split contractive values, or
        ``None`` for a baseline) and ``healthy``.
    :rtype: list[Dict[str, Any]]
    """
    ## Collect per-run series needed by the individual checks
    ### REMARK: keyed by model_name for the same reason as run_duration_frame:
    ### task_id is only unique within one campaign.
    by_task: dict[str, Dict[str, Any]] = {}
    for row in dynamics_rows:
        task = by_task.setdefault(
            row["model_name"],
            {
                "campaign": row["campaign"],
                "task_id": row["task_id"],
                "model_name": row["model_name"],
                "cell_label": row["cell_label"],
                "penalty_metric": row["penalty_metric"],
                "input_geometry": row["input_geometry"],
                "weight": row["weight"],
                "repetition": row["repetition"],
                "epochs_seen": set(),
                "non_finite": False,
                "penalty_by_epoch": {},
                "improved_after_first": False,
            },
        )
        epoch = int(row["epoch"])
        task["epochs_seen"].add(epoch)
        value = row.get("value")
        if value is not None and not math.isfinite(value):
            task["non_finite"] = True
        if row["objective"] == "contractive" and row["split"] == "train":
            task["penalty_by_epoch"][epoch] = value
        if epoch > 1 and row.get("is_best"):
            task["improved_after_first"] = True

    reference_epochs = planned_epochs
    if reference_epochs is None and by_task:
        reference_epochs = max(len(task["epochs_seen"]) for task in by_task.values())

    ## Emit one record per task
    records = []
    for task in by_task.values():
        penalties = [
            task["penalty_by_epoch"][epoch]
            for epoch in sorted(task["penalty_by_epoch"])
            if task["penalty_by_epoch"][epoch] is not None
        ]
        first_penalty = penalties[0] if penalties else None
        final_penalty = penalties[-1] if penalties else None
        penalty_increased = (
            first_penalty is not None
            and final_penalty is not None
            and final_penalty > first_penalty
        )
        epochs = len(task["epochs_seen"])
        short_run = reference_epochs is not None and epochs < reference_epochs
        no_improvement = not task["improved_after_first"]
        records.append(
            {
                "campaign": task["campaign"],
                "task_id": task["task_id"],
                "model_name": task["model_name"],
                "cell_label": task["cell_label"],
                "penalty_metric": task["penalty_metric"],
                "input_geometry": task["input_geometry"],
                "weight": task["weight"],
                "repetition": task["repetition"],
                "epochs": epochs,
                "first_penalty": first_penalty,
                "final_penalty": final_penalty,
                "non_finite_objectives": task["non_finite"],
                "penalty_increased": penalty_increased,
                "short_run": short_run,
                "no_improvement": no_improvement,
                "healthy": not (
                    task["non_finite"] or penalty_increased or short_run or no_improvement
                ),
            }
        )

    ## Tasks that were expected but produced no usable history at all
    tasks_with_history = {record["model_name"] for record in records}
    for row in grid_rows:
        if row["model_name"] in tasks_with_history:
            continue
        records.append(
            {
                "campaign": row["campaign"],
                "task_id": row["task_id"],
                "model_name": row["model_name"],
                "cell_label": row["cell_label"],
                "penalty_metric": row["penalty_metric"],
                "input_geometry": row["input_geometry"],
                "weight": row["weight"],
                "repetition": row["repetition"],
                "epochs": 0,
                "first_penalty": None,
                "final_penalty": None,
                "non_finite_objectives": False,
                "penalty_increased": False,
                "short_run": True,
                "no_improvement": True,
                "healthy": False,
            }
        )

    unhealthy = sum(1 for record in records if not record["healthy"])
    if unhealthy:
        logger.warning(
            "%s of %s run(s) failed at least one training health check.",
            unhealthy,
            len(records),
        )
    else:
        logger.info("All %s run(s) passed the training health checks.", len(records))
    return sorted(records, key=lambda record: (record["campaign"], record["task_id"]))
