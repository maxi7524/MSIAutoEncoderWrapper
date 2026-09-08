"""Paired campaign configuration and registered-component compatibility."""

from pathlib import Path

import pytest
import yaml

from msi_autoencoder_wrapper.runtime import build_plan, load_experiment_config
from msi_autoencoder_wrapper.training.criterions.criterions_manager import CriterionsManager


ROOT = Path(__file__).resolve().parents[2] / "assets/experiments/autoencoder_architecture/experiment_runs_configs"
FILES = [
    "07_09_26_predictive_expanded/bce_baseline_experiment.yaml",
    "07_09_26_predictive_expanded/bce_contrastive_experiment.yaml",
    "07_09_26_predictive_expanded/pn_bce_experiment.yaml",
    "07_09_26_predictive_expanded/pnu_ce_experiment.yaml",
    "07_09_26_predictive_expanded/vpu_experiment.yaml",
    "07_09_26_predictive_expanded/pu_ranking_experiment.yaml",
    "07_09_26_predictive_expanded/chemical_class_experiment.yaml",
    "07_09_26_pretrain/pretraining_experiment.yaml",
    "07_09_26_pretrain/chemical_pretraining_experiment.yaml",
]


@pytest.mark.parametrize("relative", FILES)
def test_campaign_preserves_baseline_population_and_builds_every_loss(relative):
    baseline = load_experiment_config(ROOT / "05_09_26_contractive_expaned/bce_baseline_experiment.yaml")
    path = ROOT / relative
    config = load_experiment_config(path)
    assert "# TODO" in path.read_text()
    assert config["seeds"] == baseline["seeds"]
    assert config["runs"] == baseline["runs"]
    factory = config["task"]["parameters"]["factory_parameters"]
    original = baseline["task"]["parameters"]["factory_parameters"]
    for key in ("project_path", "image_path", "reader", "annotations", "inverse_binning"):
        assert factory[key] == original[key]
    data = factory["dataset"]["parameters"]
    for key in ("source", "normalization", "annotation_settings", "subset", "split"):
        assert data[key] == original["dataset"]["parameters"][key]

    plan = build_plan(config)
    signatures = set()
    CriterionsManager.discover_criterions()
    for task in plan.tasks:
        head_specs = {name: {"target_field": value["target_field"]}
                      for name, value in task.parameters["factory_parameters"]["predictive"]["heads"].items()}
        phases = task.parameters["training"]["phases"]
        assert "pretraining" not in phases[-1]
        for phase in phases:
            losses = phase["criterions"]
            serialized = yaml.safe_dump(losses)
            assert "ContractiveLoss" not in serialized
            assert "NNPU" not in serialized and "class_prior" not in serialized
            if losses.get("contrastive"):
                assert losses["contrastive"]["peak_permutation"]["target"] == "InfoNCELoss"
            if "pretraining" in phase:
                assert "contrastive" not in losses
                ion = losses.get("heads", {}).get("molecule_primary", {})
                assert all(loss["target"] == "MultiLabelBCELoss" for loss in ion.values())
            if serialized not in signatures:
                composite = CriterionsManager.build_model_composite_loss("autoencoder", losses, head_specs=head_specs)
                assert composite.loss_functions
                signatures.add(serialized)

    if "07_09_26_pretrain" in relative:
        assert factory["split_reference_binning"] == original["binning"]
        axes = {tuple(t.parameters["factory_parameters"]["binning"]["parameters"][k] for k in ("x_min", "x_max"))
                for t in plan.tasks}
        assert axes == {(200., 900.), (100., 3000.)}
        assert any(len(t.parameters["training"]["phases"]) == 1 for t in plan.tasks)
        # Every synthetic setting has a paired elemental-head ablation.
        combinations = {}
        for task in plan.tasks:
            phase = task.parameters["training"]["phases"][0]
            if "pretraining" not in phase:
                continue
            heads = phase["criterions"].get("heads", {})
            key = (tuple(phase["pretraining"]["modes"]),
                   yaml.safe_dump(phase["criterions"]["reconstruction"]),
                   "molecule_primary" in heads, "chemical_class" in heads)
            combinations.setdefault(key, set()).add("element_counts" in heads)
        assert all(values == {False, True} for values in combinations.values())
        assert all(
            "contrastive" not in task.parameters["training"]["phases"][-1]["criterions"]
            for task in plan.tasks
        )
    elif path.name == "bce_baseline_experiment.yaml":
        assert all(
            "contrastive" not in task.parameters["training"]["phases"][0]["criterions"]
            for task in plan.tasks
        )
    elif path.name == "bce_contrastive_experiment.yaml":
        assert all(
            "contrastive" in task.parameters["training"]["phases"][0]["criterions"]
            for task in plan.tasks
        )
    else:
        # Every non-BCE head condition has one otherwise identical InfoNCE control.
        objective_pairs = {}
        for task in plan.tasks:
            losses = task.parameters["training"]["phases"][0]["criterions"]
            key_losses = dict(losses)
            has_contrastive = bool(key_losses.pop("contrastive", {}))
            key = yaml.safe_dump(key_losses, sort_keys=True)
            objective_pairs.setdefault(key, set()).add(has_contrastive)
        assert all(conditions == {False, True} for conditions in objective_pairs.values())


@pytest.mark.parametrize(
    ("filename", "grid_cells"),
    [
        ("pn_bce_experiment.yaml", 8),
        ("pnu_ce_experiment.yaml", 4),
        ("vpu_experiment.yaml", 6),
        ("pu_ranking_experiment.yaml", 4),
    ],
)
def test_presence_head_grids_expand_the_intended_loss_parameters(filename, grid_cells):
    """Head grids vary evidence, confidence, and contrastive conditions only."""
    config = load_experiment_config(ROOT / "07_09_26_predictive_expanded" / filename)
    plan = build_plan(config)
    assert len(plan.tasks) == grid_cells * 5
    for task in plan.tasks:
        loss = task.parameters["training"]["phases"][0]["criterions"]
        assert loss["heads"]["molecule_primary"]
        assert next(iter(loss["heads"]["molecule_primary"].values()))["weight"] == 0.2
