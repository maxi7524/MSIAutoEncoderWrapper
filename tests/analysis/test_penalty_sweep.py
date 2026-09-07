"""Tests for regularization-penalty sweep grid identity and training dynamics."""

from __future__ import annotations

from typing import Any, Optional

import pytest

from msi_autoencoder_wrapper.analysis.autoencoder.experiments.campaign_reader import (
    CampaignTask,
)
from msi_autoencoder_wrapper.analysis.autoencoder.experiments.penalty_sweep import (
    BASELINE_PENALTY_METRIC,
    epoch_duration_samples,
    final_evaluation_frame,
    paired_contrast,
    run_duration_frame,
    sweep_cell,
    sweep_grid_frame,
    training_dynamics_frame,
    training_health_report,
)


def _epoch_entry(
    epoch: int,
    *,
    duration: float = 50.0,
    contractive: Optional[float] = 100.0,
    total_loss: float = 10.0,
    is_best: bool = False,
    best_loss: float = 10.0,
) -> dict:
    """Build one per-epoch history entry in the trainer's recorded shape."""
    metrics: dict[str, Any] = {
        "epoch": epoch,
        "duration": duration,
        "masserstein": 15.0,
        "total_loss": total_loss,
        "validation_total_loss": total_loss + 1.0,
        "checkpoint_scope": "validation",
        "is_best": is_best,
        "best_loss": best_loss,
    }
    if contractive is not None:
        metrics["contractive"] = contractive
        metrics["validation_contractive"] = contractive * 0.9
    return {"phase": "joint_predictive", "metrics": metrics}


def _test_split_entry() -> dict:
    """Build the trailing post-training evaluation entry (no epoch, no duration)."""
    return {
        "phase": "joint_predictive",
        "split": "test",
        "metrics": {"masserstein": 9.1, "contractive": 1137.5, "total_loss": 10.3},
    }


def _task(
    task_id: str,
    *,
    penalty: Optional[dict] = None,
    repetition: int = 0,
    history: Optional[list] = None,
    status: str = "completed",
) -> CampaignTask:
    """Build a CampaignTask carrying one contractive grid cell."""
    objectives: dict[str, Any] = {"heads": {}, "reconstruction": {}}
    if penalty is not None:
        objectives["regularization"] = {"contractive": penalty}
    return CampaignTask(
        task_id=task_id,
        status=status,
        grid_parameters={"objectives": objectives},
        repetition=repetition,
        result={"epochs": len(history or [])},
        model_config=None,
        history=history,
    )


_LEGACY_PENALTY = {
    "target": "ContractiveLoss",
    "weight": 1e-3,
    "params": {
        "calculation_method": "approximate_hutchinson_vjp",
        "num_probes": 5,
        "penalty_metric": "spectral",
        "penalized_space": "u",
    },
}

_FISHER_HINGE_PENALTY = {
    "target": "ContractiveLoss",
    "weight": 1e-2,
    "params": {
        "calculation_method": "exact_autograd_jacobian",
        "penalty_metric": "spectral_plus_hinged",
        "penalized_space": "u",
        "input_geometry": "fisher_rao",
        "hinge_threshold": 7.0,
        "hinge_alpha": 2.0,
    },
}


class TestSweepCell:
    """Grid-cell identity resolution from recorded task parameters."""

    def test_absent_input_geometry_resolves_to_the_constructor_default(self) -> None:
        # A campaign predating `input_geometry` is euclidean, not unknown; pooling it
        # with a later campaign that sets it explicitly must produce one cell, not two.
        cell = sweep_cell(_task("task_000000", penalty=_LEGACY_PENALTY))

        assert cell.input_geometry == "euclidean"
        assert cell.penalty_metric == "spectral"
        assert cell.weight == pytest.approx(1e-3)
        assert cell.hinge_threshold is None
        assert cell.hinge_alpha == pytest.approx(1.0)
        assert not cell.is_baseline

    def test_hinged_penalty_parameters_are_retained(self) -> None:
        cell = sweep_cell(_task("task_000000", penalty=_FISHER_HINGE_PENALTY))

        assert cell.input_geometry == "fisher_rao"
        assert cell.penalty_metric == "spectral_plus_hinged"
        assert cell.hinge_threshold == pytest.approx(7.0)
        assert cell.hinge_alpha == pytest.approx(2.0)
        assert cell.calculation_method == "exact_autograd_jacobian"

    def test_task_without_contractive_regularization_is_a_baseline(self) -> None:
        cell = sweep_cell(_task("task_000000", penalty=None))

        assert cell.penalty_metric == BASELINE_PENALTY_METRIC
        assert cell.is_baseline
        assert cell.weight is None

    def test_labels_are_semantic_and_distinguish_geometry(self) -> None:
        legacy = sweep_cell(_task("task_000000", penalty=_LEGACY_PENALTY)).label
        fisher = sweep_cell(_task("task_000001", penalty=_FISHER_HINGE_PENALTY)).label

        assert legacy == "euclidean / spectral / w=1e-03"
        assert fisher == "fisher_rao / spectral_plus_hinged / w=1e-02 / tau=7"
        assert sweep_cell(_task("task_000002")).label == "baseline (no contractive)"


class TestSweepGridFrame:
    """Grid coverage records."""

    def test_incomplete_tasks_stay_visible(self) -> None:
        tasks = [
            _task("task_000000", penalty=_LEGACY_PENALTY),
            _task("task_000001", penalty=_LEGACY_PENALTY, status="failed"),
        ]

        rows = sweep_grid_frame(tasks, "campaign-a")

        assert len(rows) == 2
        assert {row["status"] for row in rows} == {"completed", "failed"}
        assert rows[0]["model_name"] == "campaign-a__task_000000"


class TestTrainingDynamicsFrame:
    """Per-epoch objective records."""

    def test_post_training_entry_is_excluded_from_the_epoch_curve(self) -> None:
        task = _task(
            "task_000000",
            penalty=_LEGACY_PENALTY,
            history=[_epoch_entry(1), _epoch_entry(2), _test_split_entry()],
        )

        rows = training_dynamics_frame([task], "campaign-a")

        assert {row["epoch"] for row in rows} == {1, 2}
        assert all(row["duration"] is not None for row in rows)

    def test_every_objective_term_is_kept_as_its_own_row(self) -> None:
        task = _task(
            "task_000000", penalty=_LEGACY_PENALTY, history=[_epoch_entry(1)]
        )

        rows = training_dynamics_frame([task], "campaign-a")
        train_rows = [row for row in rows if row["split"] == "train"]

        assert {row["objective"] for row in train_rows} == {
            "masserstein",
            "contractive",
            "total_loss",
        }
        # Bookkeeping keys must not be mistaken for objective values.
        assert "is_best" not in {row["objective"] for row in rows}
        assert "duration" not in {row["objective"] for row in rows}

    def test_validation_series_is_separated_from_the_training_series(self) -> None:
        task = _task(
            "task_000000", penalty=_LEGACY_PENALTY, history=[_epoch_entry(1)]
        )

        rows = training_dynamics_frame([task], "campaign-a")
        validation = [row for row in rows if row["split"] == "validation"]

        assert {row["objective"] for row in validation} == {
            "total_loss",
            "contractive",
        }

    def test_tasks_without_history_are_skipped_not_raised(self) -> None:
        tasks = [
            _task("task_000000", penalty=_LEGACY_PENALTY, history=[_epoch_entry(1)]),
            _task("task_000001", penalty=_LEGACY_PENALTY, history=None),
        ]

        rows = training_dynamics_frame(tasks, "campaign-a")

        assert {row["task_id"] for row in rows} == {"task_000000"}


class TestFinalEvaluationFrame:
    """Post-training evaluation records."""

    def test_trailing_split_entry_is_extracted(self) -> None:
        task = _task(
            "task_000000",
            penalty=_LEGACY_PENALTY,
            history=[_epoch_entry(1), _test_split_entry()],
        )

        rows = final_evaluation_frame([task], "campaign-a")

        assert {row["split"] for row in rows} == {"test"}
        assert {row["objective"] for row in rows} == {
            "masserstein",
            "contractive",
            "total_loss",
        }


class TestRunDurationFrame:
    """Reduction of the per-objective fan-out back to per-run durations."""

    def test_duration_is_not_multiplied_by_the_number_of_objectives(self) -> None:
        # Three objective terms per epoch produce three rows carrying the same
        # duration; summing rows naively would triple every reported run time.
        task = _task(
            "task_000000",
            penalty=_LEGACY_PENALTY,
            history=[_epoch_entry(1, duration=50.0), _epoch_entry(2, duration=30.0)],
        )

        records = run_duration_frame(training_dynamics_frame([task], "campaign-a"))

        assert len(records) == 1
        assert records[0]["epochs"] == 2
        assert records[0]["total_duration"] == pytest.approx(80.0)
        assert records[0]["mean_epoch_duration"] == pytest.approx(40.0)

    def test_task_identifiers_repeating_across_campaigns_stay_separate(self) -> None:
        # Regression: task identifiers restart at task_000000 in every campaign, so
        # keying the reduction on task_id alone merged a baseline run into a swept
        # cell and silently reported the wrong duration for both.
        swept = _task(
            "task_000000",
            penalty=_LEGACY_PENALTY,
            history=[_epoch_entry(1, duration=60.0)],
        )
        baseline = _task(
            "task_000000", penalty=None, history=[_epoch_entry(1, duration=30.0, contractive=None)]
        )

        rows = training_dynamics_frame([swept], "sweep-campaign")
        rows += training_dynamics_frame([baseline], "baseline-campaign")
        records = run_duration_frame(rows)

        assert len(records) == 2
        durations = {record["campaign"]: record["mean_epoch_duration"] for record in records}
        assert durations["sweep-campaign"] == pytest.approx(60.0)
        assert durations["baseline-campaign"] == pytest.approx(30.0)

    def test_epoch_duration_samples_keep_every_measurement(self) -> None:
        task = _task(
            "task_000000",
            penalty=_LEGACY_PENALTY,
            history=[_epoch_entry(1, duration=50.0), _epoch_entry(2, duration=30.0)],
        )

        samples = epoch_duration_samples(training_dynamics_frame([task], "campaign-a"))

        assert samples["euclidean / spectral / w=1e-03"] == pytest.approx([50.0, 30.0])


class TestTrainingHealthReport:
    """Per-run training validity checks."""

    def _report(self, tasks, campaign="campaign-a", planned_epochs=3):
        rows = training_dynamics_frame(tasks, campaign)
        grid = sweep_grid_frame(tasks, campaign)
        return training_health_report(rows, grid, planned_epochs=planned_epochs)

    def test_a_well_behaved_run_passes_every_check(self) -> None:
        task = _task(
            "task_000000",
            penalty=_LEGACY_PENALTY,
            history=[
                _epoch_entry(1, contractive=300.0, is_best=True),
                _epoch_entry(2, contractive=200.0, is_best=True),
                _epoch_entry(3, contractive=100.0, is_best=True),
            ],
        )

        record = self._report([task])[0]

        assert record["healthy"]
        assert record["first_penalty"] == pytest.approx(300.0)
        assert record["final_penalty"] == pytest.approx(100.0)

    def test_non_finite_objective_is_flagged(self) -> None:
        task = _task(
            "task_000000",
            penalty=_LEGACY_PENALTY,
            history=[
                _epoch_entry(1, is_best=True),
                _epoch_entry(2, total_loss=float("nan"), is_best=True),
                _epoch_entry(3, is_best=True),
            ],
        )

        record = self._report([task])[0]

        assert record["non_finite_objectives"]
        assert not record["healthy"]

    def test_penalty_that_grows_over_training_is_flagged(self) -> None:
        task = _task(
            "task_000000",
            penalty=_LEGACY_PENALTY,
            history=[
                _epoch_entry(1, contractive=100.0, is_best=True),
                _epoch_entry(2, contractive=200.0, is_best=True),
                _epoch_entry(3, contractive=400.0, is_best=True),
            ],
        )

        record = self._report([task])[0]

        assert record["penalty_increased"]
        assert not record["healthy"]

    def test_short_run_is_flagged_against_the_planned_budget(self) -> None:
        task = _task(
            "task_000000",
            penalty=_LEGACY_PENALTY,
            history=[_epoch_entry(1, is_best=True), _epoch_entry(2, is_best=True)],
        )

        record = self._report([task], planned_epochs=3)[0]

        assert record["short_run"]
        assert record["epochs"] == 2

    def test_run_that_never_improved_after_the_first_epoch_is_flagged(self) -> None:
        task = _task(
            "task_000000",
            penalty=_LEGACY_PENALTY,
            history=[
                _epoch_entry(1, is_best=True),
                _epoch_entry(2, is_best=False),
                _epoch_entry(3, is_best=False),
            ],
        )

        record = self._report([task])[0]

        assert record["no_improvement"]

    def test_task_without_any_history_is_reported_as_a_failed_run(self) -> None:
        # A cell whose artifacts never arrived must not silently vanish from the
        # report, otherwise a partially synced campaign reads as fully healthy.
        tasks = [
            _task("task_000000", penalty=_LEGACY_PENALTY, history=[_epoch_entry(1, is_best=True)]),
            _task("task_000001", penalty=_LEGACY_PENALTY, history=None),
        ]

        records = self._report(tasks, planned_epochs=1)
        missing = [record for record in records if record["task_id"] == "task_000001"]

        assert len(missing) == 1
        assert missing[0]["epochs"] == 0
        assert not missing[0]["healthy"]


class TestPairedContrast:
    """Repetition-paired comparison of one cell against a reference."""

    def test_only_shared_repetitions_are_paired(self) -> None:
        result = paired_contrast({0: 1.0, 1: 2.0, 5: 9.0}, {0: 0.5, 1: 1.0})

        assert result["pairs"] == 2
        assert result["mean_difference"] == pytest.approx(0.75)

    def test_a_consistent_shift_separates_from_zero(self) -> None:
        treatment = {index: 0.80 + 0.001 * index for index in range(5)}
        reference = {index: 0.70 + 0.001 * index for index in range(5)}

        result = paired_contrast(treatment, reference)

        assert result["mean_difference"] == pytest.approx(0.10)
        assert result["separated_from_zero"]
        assert result["ci_low"] > 0.0

    def test_noise_dominated_difference_does_not_separate_from_zero(self) -> None:
        treatment = {0: 0.80, 1: 0.60, 2: 0.90, 3: 0.55, 4: 0.85}
        reference = {0: 0.75, 1: 0.85, 2: 0.60, 3: 0.80, 4: 0.62}

        result = paired_contrast(treatment, reference)

        assert not result["separated_from_zero"]
        assert result["ci_low"] < 0.0 < result["ci_high"]

    def test_pairing_removes_between_repetition_variance(self) -> None:
        # Repetitions differ wildly, but the within-repetition gap is constant. An
        # unpaired comparison would drown in the between-repetition spread; the
        # paired one must resolve the shift.
        reference = {0: 0.10, 1: 0.50, 2: 0.90, 3: 0.30, 4: 0.70}
        treatment = {index: value + 0.05 for index, value in reference.items()}

        result = paired_contrast(treatment, reference)

        assert result["mean_difference"] == pytest.approx(0.05)
        assert result["separated_from_zero"]

    def test_non_finite_observations_are_dropped(self) -> None:
        result = paired_contrast(
            {0: 1.0, 1: float("nan"), 2: 3.0}, {0: 0.0, 1: 0.0, 2: 1.0}
        )

        assert result["pairs"] == 2

    def test_no_shared_repetitions_yields_an_empty_contrast(self) -> None:
        result = paired_contrast({0: 1.0}, {7: 1.0})

        assert result["pairs"] == 0
        assert result["mean_difference"] is None
        assert not result["separated_from_zero"]

    def test_single_pair_has_no_interval(self) -> None:
        result = paired_contrast({0: 1.0}, {0: 0.5})

        assert result["pairs"] == 1
        assert result["mean_difference"] == pytest.approx(0.5)
        assert result["ci_low"] is None
        assert not result["separated_from_zero"]


class TestCellIdentityNormalization:
    """Canonical identity of a cell built from records of different provenance."""

    def test_a_nan_threshold_is_treated_as_absent(self) -> None:
        # Regression: records that have been through a pandas frame carry nan where
        # the manifest path carries None, and `nan is not None` is true, so the two
        # produced different labels ("... / tau=nan") and every label-keyed lookup
        # failed with an out-of-bounds index.
        from_manifest = sweep_cell(_task("task_000000", penalty=_LEGACY_PENALTY))
        from_frame = type(from_manifest)(
            penalty_metric="spectral", input_geometry="euclidean", weight=1e-3,
            hinge_threshold=float("nan"), hinge_alpha=1.0,
            penalized_space="u", calculation_method="approximate_hutchinson_vjp",
        )

        assert from_frame.hinge_threshold is None
        assert from_frame.label == from_manifest.label
        assert from_frame == from_manifest

    def test_a_nan_weight_does_not_break_the_label(self) -> None:
        cell = sweep_cell(_task("task_000000", penalty=_LEGACY_PENALTY))
        degenerate = type(cell)(
            penalty_metric="spectral", input_geometry="euclidean", weight=float("nan"),
            hinge_threshold=None, hinge_alpha=1.0, penalized_space="u", calculation_method=None,
        )

        assert degenerate.weight is None
        assert "nan" not in degenerate.label

    def test_a_nan_hinge_alpha_falls_back_to_the_constructor_default(self) -> None:
        cell = sweep_cell(_task("task_000000", penalty=_LEGACY_PENALTY))
        degenerate = type(cell)(
            penalty_metric="spectral", input_geometry="euclidean", weight=1e-3,
            hinge_threshold=None, hinge_alpha=float("nan"), penalized_space="u",
            calculation_method=None,
        )

        assert degenerate.hinge_alpha == pytest.approx(1.0)

    def test_normalized_cells_hash_together(self) -> None:
        cell = sweep_cell(_task("task_000000", penalty=_FISHER_HINGE_PENALTY))
        same = type(cell)(
            penalty_metric="spectral_plus_hinged", input_geometry="fisher_rao",
            weight=1e-2, hinge_threshold=7.0, hinge_alpha=2.0, penalized_space="u",
            calculation_method="exact_autograd_jacobian",
        )

        assert len({cell, same}) == 1
