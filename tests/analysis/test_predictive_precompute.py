"""Artifact-layout and complete report workflow tests on deterministic miniature data."""

from copy import deepcopy
import json
from pathlib import Path

import nbformat
import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn
import yaml

from msi_autoencoder_wrapper.analysis.autoencoder.experiments import predictive_campaign as campaign
from msi_autoencoder_wrapper.analysis.autoencoder.experiments import predictive_precompute as cache
from msi_autoencoder_wrapper.analysis.autoencoder.experiments.sweep_evaluation import MaterializedSplit
from msi_autoencoder_wrapper.data.annotation_evidence import IonCatalogue

REPOSITORY = Path(__file__).resolve().parents[2]
NOTEBOOKS = REPOSITORY / "assets/experiments/autoencoder_architecture/notebooks/07_09_26_predictive_expanded"


class _TinyModel(nn.Module):
    """Deterministic encoder, LayerNorm, decoder and both campaign head shapes."""

    def __init__(self):
        super().__init__()
        with torch.random.fork_rng():
            torch.manual_seed(17)
            self.encoder = nn.Module()
            self.encoder.bottleneck_layer = nn.Sequential(nn.Linear(6, 3), nn.LayerNorm(3))
            self.decoder = nn.Sequential(nn.Linear(3, 6), nn.Softmax(dim=-1))
            self.binary = nn.Linear(3, 2)
            self.pnu = nn.Linear(3, 6)

    def forward(self, x):
        z = self.encoder.bottleneck_layer(x)  # (B, D)
        return {"latent_space": z, "reconstruction": self.decoder(z),
                "head_binary": self.binary(z), "head_pnu": self.pnu(z).reshape(-1, 2, 3)}


@pytest.fixture
def miniature_campaign(tmp_path):
    """One model per head with real manifest/config/weight files and analytical inputs."""
    settings = campaign.load_settings(NOTEBOOKS / "analysis_settings.yaml")
    settings.update(workspace=str(tmp_path), model_store=str(tmp_path / "models" / "kidney"),
                    cache_directory=str(tmp_path / "cache"), geometry_sample_size=4, batch_size=3,
                    expected_seeds=1, case_count=2,
                    shortlist=["pnu (ThreeStateCrossEntropyLoss)", "binary (ClassBalancedMultiLabelBCELoss)"])
    sources = []
    for source, head, target in [("predictive_initial", "pnu", "ThreeStateCrossEntropyLoss"),
                                  ("historical_bce", "binary", "ClassBalancedMultiLabelBCELoss")]:
        status = tmp_path / "configs" / "entropy-runs" / source / "plan" / "status"
        status.mkdir(parents=True)
        sources.append(dict(name=source, status_directory=str(status), required=True,
                            experiment_name=source, role="baseline" if source == "historical_bce" else "candidate"))
        objective = {"reconstruction": {"masserstein": {"target": "MassersteinLoss", "weight": 1.0}},
                     "heads": {head: {head: {"target": target, "weight": .2}}}}
        artifact = Path(settings["model_store"]) / f"{source}__task_000000" / "config"
        artifact.mkdir(parents=True)
        config = {"data": {"context": {"components": {"reader": {"parameters": {"file_path": "/old/workspace/datasets/kidney/kidney.imzML"}}}},
                           "dataset": {"parameters": {"normalization": "tic", "split": {"assignments": {"train": [0, 1], "validation": [2, 3], "test": [4, 5]}}}}},
                  "model": {"components": {"encoder": {"type": "test"}, "decoder": {"type": "test"}}},
                  "training": {"parameters": {"phases": [{"criterions": objective}]}}}
        (artifact / "config.json").write_text(json.dumps(config))
        (artifact / "history.json").write_text(json.dumps([{"phase": "joint", "metrics": {"epoch": 1, "validation_masserstein": .1}},
                                                         {"phase": "joint", "split": "test", "metrics": {"masserstein": .2}}]))
        torch.save(_TinyModel().state_dict(), artifact / "weights.pt")
        record = {"records": {"task_000000": {"status": "completed", "task": {"repetition": 0,
                  "grid_parameters": {"objectives": objective}, "runtime": {"model_name": "not_present"},
                  "reproducibility": {"derived_run_seeds": {"model_initialization": 1000, "training": 2000}}}}}}
        (status / "task_000000.yaml").write_text(yaml.safe_dump(record))
        (status / "task_000000-progress.yaml").write_text("status: running")
    settings["sources"] = sources
    x = np.array([[5, 1, 0, 0, 0, 0], [0, 1, 5, 0, 0, 0], [4, 0, 0, 1, 0, 1],
                  [0, 0, 4, 1, 0, 1], [2, 0, 2, 0, 1, 1], [0, 0, 0, 4, 1, 1]], dtype=np.float32) / 6
    y = np.array([[1, 0], [0, 1], [1, 0], [0, 1], [1, 1], [0, 0]], dtype=np.float32)
    splits = {name: MaterializedSplit(name, torch.from_numpy(x.copy()), y.copy(), np.ones_like(y, dtype=bool),
                                      np.arange(6), 6, 1., 42) for name in ("train", "validation", "test")}
    catalogue = IonCatalogue(("A", "B"), ((0,), (2,)), 6)
    return settings, (splits, catalogue, np.arange(6, dtype=np.float32) + 200)


def test_inventory_duplicate_grid_and_relocation(miniature_campaign):
    settings, _ = miniature_campaign
    models, sources = campaign.inventory(settings)
    assert len(models) == 2
    assert models.ready.all()
    assert sources.available.all()
    assert models.data_contract.nunique() == 1
    assert models.training_contract.nunique() == 1
    assert models["head"].tolist() == ["pnu", "binary"]
    assert set(campaign.training_history(models).record_type) == {"epoch", "evaluation"}
    grid = campaign.configured_grid(settings["experiment_config"])
    assert len(grid) == 12
    assert grid.condition.nunique() == 8
    assert grid.duplicate_of.notna().sum() == 4
    saved = json.loads((Path(models.iloc[0].artifact) / "config" / "config.json").read_text())
    local = campaign.relocated_data(saved, settings["workspace"])
    assert local["data"]["context"]["components"]["reader"]["parameters"]["file_path"].startswith(settings["workspace"])
    assert saved["data"]["context"]["components"]["reader"]["parameters"]["file_path"].startswith("/old/")


def test_split_overlap_and_configuration_mismatch_are_detected(miniature_campaign):
    settings, _ = miniature_campaign
    models, _ = campaign.inventory(settings)
    path = Path(models.iloc[0].artifact) / "config" / "config.json"
    config = json.loads(path.read_text())
    config["training"]["parameters"]["phases"][0]["criterions"]["heads"]["pnu"]["pnu"]["weight"] = .3
    path.write_text(json.dumps(config))
    changed, _ = campaign.inventory(settings)
    assert not changed.iloc[0].ready
    config["data"]["dataset"]["parameters"]["split"]["assignments"]["test"] = [0, 4]
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="overlap"):
        campaign.inventory(settings)


def test_inference_restores_training_mode_and_selects_only_active_head():
    model = _TinyModel().train()
    outputs = cache.infer_model(model, torch.ones((5, 6)) / 6, "pnu", batch_size=2)
    assert model.training
    assert outputs["logits"].shape == (5, 2, 3)
    assert outputs["reconstruction"].shape == (5, 6)
    np.testing.assert_allclose(outputs["reconstruction"].sum(axis=1), 1, atol=1e-6, rtol=0)


def test_required_source_with_no_completed_candidate_is_rejected(miniature_campaign):
    settings, prepared = miniature_campaign
    models, _ = campaign.inventory(settings)
    models.loc[models.source == "predictive_initial", "ready"] = False
    with pytest.raises(ValueError, match="predictive_initial has no ready"):
        cache.precompute(models, settings, prepared=prepared, model_loader=lambda path: _TinyModel())


def test_full_cache_resume_and_all_notebook_cells(miniature_campaign, monkeypatch, tmp_path):
    settings, prepared = miniature_campaign
    models, _ = campaign.inventory(settings)
    calls = []

    def load_model(path):
        calls.append(path)
        return _TinyModel()

    run = cache.precompute(models, settings, prepared=prepared, model_loader=load_model)
    assert len(calls) == 2
    first = cache.load_table(settings, "prediction")
    cache.precompute(models, settings, prepared=prepared, model_loader=load_model)
    assert len(calls) == 2
    pd.testing.assert_frame_equal(first, cache.load_table(settings, "prediction"))
    assert (run / "sample_indices.csv").is_file()
    manifest = json.loads((Path(settings["cache_directory"]) / "latest.json").read_text())
    (Path(manifest["models"][0]["directory"]) / "prediction.csv").unlink()
    cache.precompute(models, settings, prepared=prepared, model_loader=load_model)
    assert len(calls) == 3  # Only the incomplete model is recomputed.
    changed = deepcopy(settings)
    changed["probe_penalty"] = .2
    with pytest.raises(ValueError, match="changed"):
        cache.load_table(changed, "prediction")

    # Execute all real notebook cells with only the unavailable dataset/checkpoint
    # construction replaced. Computation, joins, plotting and exports remain real.
    monkeypatch.setattr(campaign, "load_settings", lambda path: settings)
    monkeypatch.setattr(cache, "prepare_splits", lambda models, settings: prepared)
    monkeypatch.setattr(cache.ModelLoader, "load_artifact", lambda path, strict=True: (_TinyModel(), {}, Path(path)))
    import matplotlib.pyplot as plt
    monkeypatch.chdir(REPOSITORY)
    # Provide a small historical catalogue for the real class-stratification notebook.
    population = tmp_path / "part_0_1_annotation_population_results"
    evidence = tmp_path / "part_0_2_evidence_threshold_selection_results"
    population.mkdir()
    evidence.mkdir()
    pd.DataFrame({"class_name": ["A", "B"], "mz": [200., 202.],
                  "prevalence": [.005, .5]}).to_csv(population / "class_prevalence.csv", index=False)
    pd.DataFrame({"class_name": ["B", "A"], "regime": ["common", "rare"],
                  "relative_threshold": [.005, .005], "bin_radius": [1, 1],
                  "negative_fraction_of_unannotated": [.3, .4]}).to_csv(evidence / "class_regimes.csv", index=False)
    pd.DataFrame({"class_name": ["A", "B"], "evidence_auc": [.8, .4]}).to_csv(evidence / "class_separation.csv", index=False)
    for path in sorted(NOTEBOOKS.glob("part_[1-9]_*.ipynb")):
        notebook = nbformat.read(path, as_version=4)
        nbformat.validate(notebook)
        namespace = {"__name__": "__main__"}
        for number, cell in enumerate(notebook.cells):
            if cell.cell_type == "code":
                source = cell.source
                if "RESULTS = notebook_directory" in source:
                    # Output files are confined to the fixture directory.
                    source = source.replace('RESULTS = notebook_directory /', f'RESULTS = Path({str(tmp_path)!r}) /')
                exec(compile(source, f"{path.name}:cell-{number}", "exec"), namespace)
                if "RESULTS = notebook_directory" in cell.source:
                    namespace["notebook_directory"] = tmp_path
                plt.close("all")
        assert list(namespace["RESULTS"].glob("*.csv")), path.name
        if path.name.startswith("part_3_"):
            # Pooled aggregation must reach the reported table, not only the macro means.
            reported = namespace["units"]
            assert {"average_precision", "micro_average_precision", "micro_roc_auc"} <= set(reported.metric)
            assert set(namespace["ranking_curves"].curve) == {"precision_recall", "roc"}
        if path.name.startswith("part_4_"):
            separation = namespace["validation_separation"]
            assert {"positive_vs_uncertain_auc", "negative_vs_uncertain_auc"} <= set(separation.columns)
            histograms = namespace["validation_histograms"]
            assert set(histograms.state) == {"P", "N", "U"}
            # Shared bin edges are what makes the stored counts overlayable across models.
            assert histograms.groupby(["quantity", "model_id"]).left_edge.apply(tuple).nunique() == 2
        if path.name.startswith("part_8_"):
            assert set(namespace["strata"].class_name) == {"A", "B"}
            assert set(namespace["head_probe"].split) == {"validation", "test"}
            comparison = namespace["head_probe"]
            assert np.isfinite(comparison.query("population == 'annotation_retrieval'")["value"]).all()
            np.testing.assert_array_equal(comparison["value"].isna(),
                                          comparison.value_head.isna() | comparison.value_probe.isna())
        if path.name.startswith("part_9_"):
            agreement = namespace["agreement"]
            assert set(agreement.left) | set(agreement.right) <= set(namespace["shortlist"])
            np.testing.assert_allclose(agreement.difference, agreement.value_left - agreement.value_right)
    weights = Path(models.iloc[0].artifact) / "config" / "weights.pt"
    weights.write_bytes(weights.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="Checkpoint bytes changed"):
        cache.load_table(settings, "prediction")


def test_empty_chemistry_default_does_not_change_data_contract(miniature_campaign):
    settings, _ = miniature_campaign
    models, _ = campaign.inventory(settings)
    config_path = Path(models.iloc[0].artifact) / 'config' / 'config.json'
    config = json.loads(config_path.read_text())
    config['data']['dataset']['parameters']['chemistry'] = {}
    config_path.write_text(json.dumps(config))
    equivalent, _ = campaign.inventory(settings)
    assert equivalent.data_contract.nunique() == 1
    config['data']['dataset']['parameters']['chemistry'] = {'version': 'different'}
    config_path.write_text(json.dumps(config))
    changed, _ = campaign.inventory(settings)
    assert changed.data_contract.nunique() == 2


def test_history_components_separate_split_prefix_and_comparability():
    history = pd.DataFrame({
        "metric": ["masserstein", "validation_masserstein", "molecule_pnu__pnu_ce",
                   "validation_molecule_pnu__pnu_ce", "total_loss", "epoch"],
        "value": [.1, .2, .3, .4, .5, 1.],
    })
    result = campaign.history_components(history)
    assert result.history_split.tolist() == ["train", "validation", "train", "validation", "train", "train"]
    assert result.component.tolist() == ["reconstruction", "reconstruction", "head", "head", "total", "bookkeeping"]
    assert result.component_name.tolist() == ["masserstein", "masserstein", "pnu_ce", "pnu_ce", "total_loss", "epoch"]
    # Only the shared reconstruction cost may share a vertical axis across objectives.
    assert result.comparable.tolist() == [True, True, False, False, False, False]
    assert campaign.history_components(history.iloc[:0]).empty
