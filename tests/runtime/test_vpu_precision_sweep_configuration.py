"""Regression coverage for the final controlled VPU precision sweep."""

from __future__ import annotations

from pathlib import Path

from msi_autoencoder_wrapper.runtime.configuration.loading import load_experiment_config
from msi_autoencoder_wrapper.runtime.planning.plan import build_plan


ROOT = Path(
    "assets/experiments/autoencoder_architecture/experiment_runs_configs/"
    "14_09_26_predictive_final"
)


def test_vpu_precision_sweep_is_paired_and_excludes_simulated_negatives() -> None:
    """Every alpha value is paired with no-contrastive and InfoNCE training."""
    plan = build_plan(load_experiment_config(ROOT / "vpu_precision_sweep.yaml"))

    assert len(plan.tasks) == 50
    observed_alpha: set[float] = set()
    contrastive_pairs: dict[float, set[str]] = {}
    for task in plan.tasks:
        parameters = task.parameters
        target_settings = parameters["factory_parameters"]["dataset"]["parameters"][
            "annotation_settings"
        ]["targets"]["molecule"]
        phase = parameters["training"]["phases"][0]
        heads = parameters["factory_parameters"]["predictive"]["heads"]
        configured_losses = phase["criterions"]["heads"]

        assert "simulated_negative" not in target_settings
        assert phase["supervision_sampling"]["proportions"] == {
            "positive": 1,
            "unlabelled": 1,
        }
        assert set(heads) == {"molecule_vpu"}
        criterion = next(iter(configured_losses["molecule_vpu"].values()))
        assert criterion["target"] == "VariationalPULoss"
        assert criterion["params"]["beta"] == 1.0
        assert criterion["params"]["consistency_weight"] == 0.0
        alpha = criterion["params"]["alpha"]
        observed_alpha.add(alpha)
        contrastive = phase["criterions"]["contrastive"]
        condition = "none" if not contrastive else "label_invariant_infonce"
        if contrastive:
            info_nce = contrastive["peak_permutation"]
            assert info_nce["target"] == "InfoNCELoss"
            assert info_nce["weight"] == 0.1
            assert info_nce["params"]["peak_selection_method"] == "permutation_label_invariant"
        contrastive_pairs.setdefault(alpha, set()).add(condition)

    assert observed_alpha == {1.0, 1.25, 1.5, 2.0, 3.0}
    assert all(
        conditions == {"none", "label_invariant_infonce"}
        for conditions in contrastive_pairs.values()
    )
