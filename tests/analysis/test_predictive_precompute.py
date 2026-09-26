"""Artifact-layout and complete report workflow tests on deterministic miniature data."""

from copy import deepcopy
import dataclasses
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
from msi_autoencoder_wrapper.analysis.autoencoder.experiments import predictive_reports as reports
from msi_autoencoder_wrapper.analysis.autoencoder.experiments.sweep_evaluation import MaterializedSplit
from msi_autoencoder_wrapper.data.annotation_evidence import IonCatalogue

REPOSITORY = Path(__file__).resolve().parents[2]
NOTEBOOKS = REPOSITORY / "assets/experiments/autoencoder_architecture/notebooks/14_09_predictive_expanded"


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


def test_source_digest_excludes_training_only_subtrees(tmp_path):
    (tmp_path / "analysis").mkdir()
    (tmp_path / "analysis" / "predictive_precompute.py").write_text("A = 1\n")
    (tmp_path / "training").mkdir()
    (tmp_path / "training" / "some_loss.py").write_text("B = 1\n")
    before = cache._source_digest(tmp_path, exclude=cache.TRAINING_ONLY_SOURCES)
    (tmp_path / "training" / "some_loss.py").write_text("B = 2\n")  # Training-only edit.
    after_training_edit = cache._source_digest(tmp_path, exclude=cache.TRAINING_ONLY_SOURCES)
    assert before == after_training_edit
    (tmp_path / "analysis" / "predictive_precompute.py").write_text("A = 2\n")  # Analysis edit.
    after_analysis_edit = cache._source_digest(tmp_path, exclude=cache.TRAINING_ONLY_SOURCES)
    assert after_analysis_edit != before
    # Without the exclusion, the same training-only edit would be visible: the match
    # above is TRAINING_ONLY_SOURCES doing its job, not incidental hash luck.
    unscoped_after_training_edit = cache._source_digest(tmp_path)
    assert unscoped_after_training_edit != after_training_edit


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


def test_task_objective_uses_complete_resolved_training_criteria() -> None:
    """Resolved grid metadata may omit fixed reconstruction and contrastive terms."""
    head = {"molecule_vpu": {"vpu": {"target": "VariationalPULoss", "weight": 0.2}}}
    complete = {
        "reconstruction": {"masserstein": {"target": "MassersteinLoss", "weight": 1.0}},
        "heads": head,
        "contrastive": {},
    }
    task = {
        "grid_parameters": {"objectives": {"heads": head}},
        "parameters": {"training": {"phases": [{"criterions": complete}]}},
    }

    assert campaign._task_objective(task) == complete


def test_inventory_can_restrict_a_source_to_selected_condition_labels(miniature_campaign):
    """Source filters exclude unrelated objectives from a shared campaign download."""
    settings, _ = miniature_campaign
    settings["sources"][0]["include_labels"] = ["pnu (ThreeStateCrossEntropyLoss)"]

    models, _ = campaign.inventory(settings)

    assert set(models["source"]) == {"predictive_initial", "historical_bce"}
    assert models.loc[models.source == "predictive_initial", "label"].tolist() == [
        "pnu (ThreeStateCrossEntropyLoss)"
    ]


def test_configured_grid_expands_all_runtime_grid_axes():
    """Coverage auditing must match the planner when contrastive is a separate axis."""
    config = REPOSITORY / "assets/experiments/autoencoder_architecture/experiment_runs_configs/14_09_26_predictive_final/vpu_precision_sweep.yaml"

    grid = campaign.configured_grid(config)

    assert len(grid) == 10  # 5 VPU settings x 2 contrastive settings.
    assert grid.label.eq("vpu_alpha_1_beta_1 (VariationalPULoss)").sum() == 2
    assert grid.condition.nunique() == 10


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


def test_geometry_table_carries_structure_and_label_correlation_metrics_for_u_only(miniature_campaign):
    settings, prepared = miniature_campaign
    models, _ = campaign.inventory(settings)
    cache.precompute(models, settings, prepared=prepared, model_loader=lambda path: _TinyModel())
    geometry = cache.load_table(settings, "geometry")
    new_metrics = {"observed_mean_cos_theta", "observed_sd_cos_theta", "uniform_baseline_sd_cos_theta",
                  "effective_dimension", "two_nn_intrinsic_dimension", "label_correlation_spearman_r",
                  "label_correlation_p_value", "label_correlation_pairs"}
    present = set(geometry.metric)
    assert new_metrics <= present
    for metric in new_metrics:
        assert set(geometry.query("metric == @metric").space) == {"u"}, metric
    # Every ready model and non-train split contributes one row per new metric.
    per_model_split = geometry.query("metric == 'two_nn_intrinsic_dimension'")
    assert len(per_model_split) == len(models) * 2  # validation + test, not train.


def test_peak_matching_is_computed_on_the_bounded_case_set(miniature_campaign):
    settings, prepared = miniature_campaign
    models, _ = campaign.inventory(settings)
    run = cache.precompute(models, settings, prepared=prepared, model_loader=lambda path: _TinyModel())
    peak_matching = cache.load_table(settings, "peak_matching")
    cases = cache.load_table(settings, "spectrum_cases")
    assert not peak_matching.empty
    assert {"peak_mz", "mz_error", "relative_intensity_error", "original_intensity", "detected"} <= set(peak_matching.columns)
    # Every matched-peak row must come from a position actually selected as a case.
    assert set(peak_matching.row_position) <= set(cases.row_position)
    assert (run / "sample_indices.csv").is_file()


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
    pd.testing.assert_frame_equal(first, cache.load_table(changed, "prediction"))

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


def test_simulated_negative_schema_does_not_change_data_contract(miniature_campaign):
    """Regression: an older artifact omitting `simulated_negative` must still match a
    newer one that records it, since the analysis applies its own shared evidence
    policy at evaluation time regardless of what a model's own training run logged.
    Caught for real comparing `predictive_initial` (omits the key) against
    `predictive_expanded` (records it) on the live kidney campaign: both trained on
    byte-identical split assignments, yet `_data_contract` disagreed before the fix.
    """
    settings, _ = miniature_campaign
    models, _ = campaign.inventory(settings)
    paths = [Path(row.artifact) / 'config' / 'config.json' for row in models.itertuples()]
    configs = [json.loads(path.read_text()) for path in paths]
    for config in configs:
        # Both artifacts start with the same `targets.molecule` branch already present
        # (as on the real campaign), just missing `simulated_negative` on either side.
        config['data']['dataset']['parameters']['annotation_settings'] = {
            "targets": {"molecule": {"empty_spectrum_policy": "exclude"}}}
    configs[0]['data']['dataset']['parameters']['annotation_settings']['targets']['molecule'].update(
        simulated_negative={"type": "SignalEvidenceSimulatedNegative", "parameters": {"relative_threshold": 0.0119}},
        simulated_negative_metadata_key=None)
    for path, config in zip(paths, configs):
        path.write_text(json.dumps(config))
    equivalent, _ = campaign.inventory(settings)
    assert equivalent.data_contract.nunique() == 1

    # A real difference elsewhere in the same branch must still be caught.
    configs[1]['data']['dataset']['parameters']['annotation_settings']['targets']['molecule']['empty_spectrum_policy'] = "unlabelled"
    paths[1].write_text(json.dumps(configs[1]))
    changed, _ = campaign.inventory(settings)
    assert changed.data_contract.nunique() == 2


@pytest.fixture()
def routine_settings(tmp_path: Path) -> dict:
    """A minimal predictive settings mapping with one configured analysis routine."""
    return {
        "device": "cuda",
        "settings_path": str(tmp_path / "analysis_settings.yaml"),
        "analyses": {"campaign_training_dynamics": {"output_directory": str(tmp_path / "results")}},
    }


def test_analysis_settings_merge_shared_and_specific_entries(routine_settings):
    merged = cache.analysis_settings(routine_settings, "campaign_training_dynamics")
    assert merged["device"] == "cuda"
    assert isinstance(merged["output_directory"], Path)
    assert "analyses" not in merged


def test_unknown_analysis_is_rejected(routine_settings):
    with pytest.raises(KeyError):
        cache.analysis_settings(routine_settings, "not_a_routine")


def test_unconfigured_but_registered_analysis_is_rejected(routine_settings):
    routine_settings["analyses"] = {}
    with pytest.raises(KeyError):
        cache.analysis_settings(routine_settings, "campaign_training_dynamics")


def test_analysis_device_policy_matches_the_base_cache(routine_settings, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="without CUDA"):
        cache.resolve_device(routine_settings)
    assert cache.resolve_device(routine_settings, allow_cpu=True).type == "cpu"


def test_analysis_run_command_forms(routine_settings):
    background = cache.run_analysis_command(routine_settings, "campaign_training_dynamics")
    assert background.startswith("nohup ") and background.rstrip().endswith("&")
    assert "--analysis campaign_training_dynamics" in background
    assert "campaign_training_dynamics.log" in background
    foreground = cache.run_analysis_command(routine_settings, "campaign_training_dynamics", background=False)
    assert not foreground.startswith("nohup")
    assert "-m msi_autoencoder_wrapper.analysis.autoencoder.experiments.predictive_precompute" in foreground


def test_analysis_table_and_metadata_name_the_producing_command_when_absent(routine_settings):
    with pytest.raises(FileNotFoundError) as failure:
        cache.load_analysis_table(routine_settings, "campaign_training_dynamics", "inventory")
    assert "--analysis campaign_training_dynamics" in str(failure.value)
    with pytest.raises(FileNotFoundError) as failure:
        cache.load_analysis_metadata(routine_settings, "campaign_training_dynamics")
    assert "--analysis campaign_training_dynamics" in str(failure.value)


def test_analysis_routine_registration_survives_module_execution_order():
    # Regression, mirrors contractive_precompute's own guard test: a routine appended
    # below the `__main__` guard would be defined after the CLI already parsed
    # `--analysis`, so it would show as registered on import but be rejected at the
    # command line.
    source = Path(cache.__file__).read_text(encoding="utf-8")
    guard = 'if __name__ == "__main__":'
    assert source.index(guard) > source.rindex("@_routine("), (
        "the entry-point guard must be the last statement, after every routine"
    )


def _models_frame(model_ids: list[str]) -> pd.DataFrame:
    return pd.DataFrame({"model_id": model_ids, "source": "predictive_initial", "role": "candidate",
                         "condition": [f"c{index}" for index in range(len(model_ids))],
                         "label": model_ids, "repetition": 0})


def test_run_duration_frame_deduplicates_epoch_bookkeeping_and_sums_durations():
    # m1's epoch-1 duration is logged twice (the per-objective-term fan-out real training
    # histories can produce) and must be counted once; its total_loss row must be ignored.
    history = pd.DataFrame({
        "model_id": ["m1", "m1", "m1", "m1", "m2"],
        "metric": ["duration", "duration", "duration", "total_loss", "duration"],
        "epoch": [1, 1, 2, 1, 1],
        "value": [10.0, 10.0, 12.0, 0.5, 5.0],
    })
    result = campaign.run_duration_frame(history, _models_frame(["m1", "m2"])).set_index("model_id")
    assert result.loc["m1", "epochs"] == 2
    assert result.loc["m1", "total_duration"] == pytest.approx(22.0)
    assert result.loc["m1", "mean_epoch_duration"] == pytest.approx(11.0)
    assert result.loc["m2", "epochs"] == 1
    assert result.loc["m2", "total_duration"] == pytest.approx(5.0)


def test_run_duration_frame_is_empty_but_well_formed_without_duration_rows():
    history = pd.DataFrame({"model_id": ["m1"], "metric": ["total_loss"], "epoch": [1], "value": [0.5]})
    result = campaign.run_duration_frame(history, _models_frame(["m1"]))
    assert result.empty
    assert {"model_id", "epochs", "total_duration", "mean_epoch_duration"} <= set(result.columns)


def _epoch_row(model_id: str, epoch: int, metric: str, value: float, *,
               history_split: str = "train", is_best: bool = False, record_type: str = "epoch") -> dict:
    return {"model_id": model_id, "epoch": epoch, "metric": metric, "value": value,
           "history_split": history_split, "is_best": is_best, "record_type": record_type}


def test_epoch_duration_samples_deduplicates_and_groups_by_label():
    history = pd.DataFrame({
        "model_id": ["m1", "m1", "m1", "m2"],
        "metric": ["duration", "duration", "duration", "duration"],
        "epoch": [1, 1, 2, 1],
        "value": [10.0, 10.0, 12.0, 5.0],
    })
    models = pd.DataFrame({"model_id": ["m1", "m2"], "label": ["bce_global", "bce_global"]})
    samples = campaign.epoch_duration_samples(history, models)
    # m1's epoch-1 duplicate must collapse to one sample; both models share one label.
    assert sorted(samples["bce_global"]) == [5.0, 10.0, 12.0]


def test_training_health_report_flags_each_failure_mode_independently():
    history = pd.DataFrame([
        # m_healthy: three epochs, loss strictly decreasing, improves after epoch 1.
        _epoch_row("m_healthy", 1, "total_loss", 1.0, is_best=True),
        _epoch_row("m_healthy", 2, "total_loss", 0.7, is_best=True),
        _epoch_row("m_healthy", 3, "total_loss", 0.5, is_best=True),
        # m_short: only one epoch measured, while the campaign otherwise reaches three.
        _epoch_row("m_short", 1, "total_loss", 1.0, is_best=True),
        # m_worse: loss increases and the trainer never improves past epoch 1.
        _epoch_row("m_worse", 1, "total_loss", 1.0, is_best=True),
        _epoch_row("m_worse", 2, "total_loss", 1.2, is_best=False),
        _epoch_row("m_worse", 3, "total_loss", 1.4, is_best=False),
        # m_broken: a non-finite value anywhere invalidates the run outright.
        _epoch_row("m_broken", 1, "total_loss", float("nan"), is_best=True),
        _epoch_row("m_broken", 2, "total_loss", 0.4, is_best=True),
        _epoch_row("m_broken", 3, "total_loss", 0.3, is_best=True),
    ])
    models = _models_frame(["m_healthy", "m_short", "m_worse", "m_broken"])
    report = campaign.training_health_report(history, models).set_index("model_id")

    assert report.loc["m_healthy", "healthy"]
    assert not report.loc["m_healthy", ["non_finite_objectives", "loss_increased", "short_run", "no_improvement"]].any()

    assert not report.loc["m_short", "healthy"]
    assert report.loc["m_short", "short_run"]

    assert not report.loc["m_worse", "healthy"]
    assert report.loc["m_worse", "loss_increased"]
    assert report.loc["m_worse", "no_improvement"]

    assert not report.loc["m_broken", "healthy"]
    assert report.loc["m_broken", "non_finite_objectives"]


def test_training_health_report_respects_an_explicit_planned_epoch_budget():
    history = pd.DataFrame([_epoch_row("m1", epoch, "total_loss", 1.0 - .1 * epoch, is_best=True)
                            for epoch in (1, 2, 3)])
    # Three measured epochs is short against an explicit five-epoch budget, even though
    # it is the campaign's own maximum (the implicit fallback would call this healthy).
    report = campaign.training_health_report(history, _models_frame(["m1"]), planned_epochs=5).set_index("model_id")
    assert report.loc["m1", "short_run"]


def test_campaign_training_dynamics_routine_runs_end_to_end(miniature_campaign, tmp_path):
    settings, _ = miniature_campaign
    settings["analyses"] = {"campaign_training_dynamics": {"output_directory": str(tmp_path / "dynamics_results")}}
    output = cache.precompute_analysis(settings, "campaign_training_dynamics", allow_cpu=True)

    inventory = cache.load_analysis_table(settings, "campaign_training_dynamics", "inventory")
    assert len(inventory) == 2 and inventory.ready.all()
    sources = cache.load_analysis_table(settings, "campaign_training_dynamics", "sources")
    assert sources.available.all()
    coverage = cache.load_analysis_table(settings, "campaign_training_dynamics", "condition_coverage")
    # One row per declared condition in the real campaign grid; none of them match this
    # fixture's synthetic, unrelated objective, so coverage is correctly all zero here.
    assert len(coverage) == 8
    assert (coverage.ready_tasks == 0).all()

    dynamics = cache.load_analysis_table(settings, "campaign_training_dynamics", "training_dynamics")
    evaluation = cache.load_analysis_table(settings, "campaign_training_dynamics", "final_evaluation")
    assert len(dynamics) == 4  # 2 epoch-type rows (epoch, validation_masserstein) per model
    assert len(evaluation) == 2  # 1 evaluation-type row per model

    health = cache.load_analysis_table(settings, "campaign_training_dynamics", "training_health").set_index("model_id")
    # This fixture's minimal one-epoch history never logs an improvement past epoch 1,
    # so both models are correctly flagged unhealthy for that reason alone.
    assert not health.healthy.any()
    assert health.no_improvement.all()
    assert not health[["non_finite_objectives", "loss_increased", "short_run"]].any().any()

    durations = cache.load_analysis_table(settings, "campaign_training_dynamics", "run_durations")
    assert durations.empty  # no `duration` metric is logged in this fixture's history

    metadata = cache.load_analysis_metadata(settings, "campaign_training_dynamics")
    assert metadata["analysis"] == "campaign_training_dynamics"
    assert metadata["tasks"] == 2
    assert (output / "training_health.csv").is_file()


def test_campaign_training_dynamics_does_not_require_cuda(monkeypatch, tmp_path):
    """Manifest-only campaign diagnostics remain available before GPU inference."""
    settings = {
        "device": "cuda",
        "analyses": {
            "campaign_training_dynamics": {"output_directory": str(tmp_path / "results")},
        },
    }

    def unexpected_cuda_resolution(*_args, **_kwargs):
        raise AssertionError("Campaign diagnostics must not resolve a CUDA device.")

    monkeypatch.setattr(cache, "resolve_device", unexpected_cuda_resolution)
    monkeypatch.setitem(
        cache.ROUTINES,
        "campaign_training_dynamics",
        lambda _settings: {"inventory": pd.DataFrame({"model_id": ["model"]}), "metadata": {}},
    )

    output = cache.precompute_analysis(settings, "campaign_training_dynamics")

    assert (output / "inventory.csv").is_file()


def test_campaign_training_dynamics_notebook_runs_end_to_end(miniature_campaign, monkeypatch, tmp_path):
    settings, _ = miniature_campaign
    models, _ = campaign.inventory(settings)
    # This fixture's default one-epoch history is enough for the routine tests above, but
    # the notebook's loss-trajectory and duration-violin cells need more than one epoch to
    # draw anything real: replace it with a richer, multi-epoch history (total_loss
    # decreasing, an improving checkpoint past epoch 1, and a per-epoch duration reading).
    first_epoch_duration = {"predictive_initial/task_000000": 12.0, "historical_bce/task_000000": 9.0}
    for model in models.to_dict("records"):
        entries = [{"phase": "joint", "metrics": {"epoch": epoch, "duration": first_epoch_duration[model["model_id"]] + epoch,
                                                  "total_loss": 1.0 - 0.1 * epoch, "is_best": epoch > 1}}
                  for epoch in (1, 2, 3)]
        entries.append({"phase": "joint", "split": "test", "metrics": {"masserstein": .2}})
        (Path(model["artifact"]) / "config" / "history.json").write_text(json.dumps(entries))

    settings["planned_epochs"] = 15
    settings["analyses"] = {"campaign_training_dynamics": {"output_directory": str(tmp_path / "results")}}
    cache.precompute_analysis(settings, "campaign_training_dynamics", allow_cpu=True)

    monkeypatch.setattr(campaign, "load_settings", lambda path: settings)
    monkeypatch.chdir(REPOSITORY)
    import matplotlib.pyplot as plt
    path = REPOSITORY / "assets/experiments/autoencoder_architecture/notebooks/14_09_predictive_expanded/part_0_0_campaign_training_dynamics.ipynb"
    notebook = nbformat.read(path, as_version=4)
    nbformat.validate(notebook)
    namespace = {"__name__": "__main__"}
    for number, cell in enumerate(notebook.cells):
        if cell.cell_type == "code":
            exec(compile(cell.source, f"{path.name}:cell-{number}", "exec"), namespace)
            plt.close("all")

    assert list(namespace["ARTIFACT_DIR"].glob("*.csv"))
    assert set(namespace["duration_samples"]) == set(namespace["ordered_labels"])
    assert len(namespace["conditions"]) == 1  # only predictive_initial's one synthetic condition
    assert namespace["baseline_epoch_seconds"] == pytest.approx(11.0)  # (10 + 11 + 12) / 3


def test_reconstruction_local_routine_compares_two_different_head_types(miniature_campaign, monkeypatch, tmp_path):
    settings, prepared = miniature_campaign
    monkeypatch.setattr(cache, "prepare_splits", lambda models, settings: prepared)
    monkeypatch.setattr(cache.ModelLoader, "load_artifact", lambda path, strict=True: (_TinyModel(), {}, Path(path)))

    settings["compared"] = {"pnu": "pnu (ThreeStateCrossEntropyLoss)", "baseline": "binary (ClassBalancedMultiLabelBCELoss)"}
    settings["cases_per_category"] = 1
    settings["top_k"] = 2  # the fixture's catalogue has only 2 classes
    settings["analyses"] = {"reconstruction_local": {"output_directory": str(tmp_path / "results")}}

    output = cache.precompute_analysis(settings, "reconstruction_local", allow_cpu=True)

    grid = cache.load_analysis_table(settings, "reconstruction_local", "grid")
    assert set(grid.label) == set(settings["compared"].values())
    cases = cache.load_analysis_table(settings, "reconstruction_local", "selected_cases")
    assert {"W pnu", "W baseline"} <= set(cases.columns)
    spectra = cache.load_analysis_table(settings, "reconstruction_local", "displayed_spectra")
    assert set(spectra.series) >= {"input", "pnu", "baseline"}
    curves = cache.load_analysis_table(settings, "reconstruction_local", "amplitude_curves")
    assert set(curves.model) == {"pnu", "baseline", "between models"}
    # The disagreement curve carries no per-model latent angle; every real model curve does.
    assert curves.query("model == 'between models'").median_angle_degrees.isna().all()
    assert curves.query("model != 'between models'").median_angle_degrees.notna().all()

    metadata = cache.load_analysis_metadata(settings, "reconstruction_local")
    assert metadata["compared"] == settings["compared"]
    assert set(metadata["compared_heads"].values()) == {"pnu", "binary"}
    assert (output / "amplitude_curves.csv").is_file()


def test_reconstruction_local_rejects_a_condition_absent_from_the_inventory(miniature_campaign, monkeypatch, tmp_path):
    settings, prepared = miniature_campaign
    monkeypatch.setattr(cache, "prepare_splits", lambda models, settings: prepared)
    monkeypatch.setattr(cache.ModelLoader, "load_artifact", lambda path, strict=True: (_TinyModel(), {}, Path(path)))
    settings["compared"] = {"pnu": "pnu (ThreeStateCrossEntropyLoss)", "missing": "not a real condition"}
    settings["analyses"] = {"reconstruction_local": {"output_directory": str(tmp_path / "results")}}

    with pytest.raises(ValueError, match="absent from the ready inventory"):
        cache.precompute_analysis(settings, "reconstruction_local", allow_cpu=True)


def test_reconstruction_local_notebook_runs_end_to_end(miniature_campaign, monkeypatch, tmp_path):
    settings, prepared = miniature_campaign
    monkeypatch.setattr(cache, "prepare_splits", lambda models, settings: prepared)
    monkeypatch.setattr(cache.ModelLoader, "load_artifact", lambda path, strict=True: (_TinyModel(), {}, Path(path)))

    settings["compared"] = {"pnu": "pnu (ThreeStateCrossEntropyLoss)", "baseline": "binary (ClassBalancedMultiLabelBCELoss)"}
    settings["cases_per_category"] = 1
    settings["top_k"] = 2  # the fixture's catalogue has only 2 classes
    settings["display_amplitudes"] = [0.0, 0.5, 1.0]
    settings["curve_amplitudes"] = [0.0, 0.5, 1.0]
    settings["analyses"] = {"reconstruction_local": {"output_directory": str(tmp_path / "results")}}
    cache.precompute_analysis(settings, "reconstruction_local", allow_cpu=True)

    monkeypatch.setattr(campaign, "load_settings", lambda path: settings)
    monkeypatch.chdir(REPOSITORY)
    import matplotlib.pyplot as plt
    path = REPOSITORY / "assets/experiments/autoencoder_architecture/notebooks/14_09_predictive_expanded/part_1_reconstruction_local.ipynb"
    notebook = nbformat.read(path, as_version=4)
    nbformat.validate(notebook)
    namespace = {"__name__": "__main__"}
    for number, cell in enumerate(notebook.cells):
        if cell.cell_type == "code":
            exec(compile(cell.source, f"{path.name}:cell-{number}", "exec"), namespace)
            plt.close("all")

    assert list(namespace["ARTIFACT_DIR"].glob("*.csv"))
    assert namespace["HEADS"] == {"pnu": "pnu", "baseline": "binary"}
    assert len(namespace["shown_rows"]) >= 1
    assert set(namespace["curve_frame"].model) == {"pnu", "baseline", "between models"}


def test_reconstruction_global_routine_without_per_image_identity(miniature_campaign, monkeypatch, tmp_path):
    settings, prepared = miniature_campaign
    monkeypatch.setattr(cache, "prepare_splits", lambda models, settings: prepared)
    monkeypatch.setattr(cache.ModelLoader, "load_artifact", lambda path, strict=True: (_TinyModel(), {}, Path(path)))
    models, _ = campaign.inventory(settings)
    cache.precompute(models, settings, prepared=prepared, model_loader=lambda path: _TinyModel())

    settings["global_perturbation_amplitudes"] = [0.0, 1.0]
    settings["global_perturbation_sample_pixels"] = 6
    settings["analyses"] = {"reconstruction_global": {"output_directory": str(tmp_path / "results")}}
    cache.precompute_analysis(settings, "reconstruction_global", allow_cpu=True)

    # The fixture's MaterializedSplit carries no `sample_ids` (default None), so the
    # per-image breakdown must degrade gracefully to an empty, correctly-shaped table
    # rather than crash the CSV round trip (an entirely columnless frame does).
    image_breakdown = cache.load_analysis_table(settings, "reconstruction_global", "image_breakdown")
    assert image_breakdown.empty
    assert {"model_id", "image_key", "mean_masserstein"} <= set(image_breakdown.columns)

    drift = cache.load_analysis_table(settings, "reconstruction_global", "perturbation_drift")
    assert set(drift.model_id) == set(models.model_id)
    assert "times_baseline" in drift.columns
    # At amplitude 0 the baseline's own drift against itself is exactly 0, so its ratio
    # to itself is the undefined 0/0 (correctly NaN); only amplitude > 0 is checked.
    baseline_positive_amplitude = drift.query("role == 'baseline' and amplitude > 0")
    np.testing.assert_allclose(baseline_positive_amplitude.times_baseline, 1.0)

    summary = cache.load_analysis_table(settings, "reconstruction_global", "reconstruction_summary")
    assert {"metric", "statistic", "value"} <= set(summary.columns)

    metadata = cache.load_analysis_metadata(settings, "reconstruction_global")
    assert metadata["analysis"] == "reconstruction_global"
    assert metadata["per_image_identity_available"] is False
    assert metadata["models"] == 2


def test_reconstruction_global_routine_groups_by_image_when_available(miniature_campaign, monkeypatch, tmp_path):
    settings, (splits, catalogue, axis) = miniature_campaign
    # A CohortDataset-style sample identity, attached to the test split only.
    test_split = splits["test"]
    sample_ids = np.array([{"image_key": f"image-{position % 2}", "spectrum_id": position}
                           for position in range(test_split.sampled)], dtype=object)
    splits = {**splits, "test": dataclasses.replace(test_split, sample_ids=sample_ids)}
    prepared = (splits, catalogue, axis)
    monkeypatch.setattr(cache, "prepare_splits", lambda models, settings: prepared)
    monkeypatch.setattr(cache.ModelLoader, "load_artifact", lambda path, strict=True: (_TinyModel(), {}, Path(path)))
    models, _ = campaign.inventory(settings)
    cache.precompute(models, settings, prepared=prepared, model_loader=lambda path: _TinyModel())

    settings["global_perturbation_amplitudes"] = [0.0, 1.0]
    settings["global_perturbation_sample_pixels"] = 6
    settings["analyses"] = {"reconstruction_global": {"output_directory": str(tmp_path / "results")}}
    cache.precompute_analysis(settings, "reconstruction_global", allow_cpu=True)

    image_breakdown = cache.load_analysis_table(settings, "reconstruction_global", "image_breakdown")
    assert not image_breakdown.empty
    assert set(image_breakdown.image_key) == {"image-0", "image-1"}
    assert set(image_breakdown.model_id) == set(models.model_id)

    metadata = cache.load_analysis_metadata(settings, "reconstruction_global")
    assert metadata["per_image_identity_available"] is True


def test_reconstruction_global_notebook_runs_end_to_end(miniature_campaign, monkeypatch, tmp_path):
    settings, (splits, catalogue, axis) = miniature_campaign
    test_split = splits["test"]
    sample_ids = np.array([{"image_key": f"image-{position % 2}", "spectrum_id": position}
                           for position in range(test_split.sampled)], dtype=object)
    splits = {**splits, "test": dataclasses.replace(test_split, sample_ids=sample_ids)}
    prepared = (splits, catalogue, axis)
    monkeypatch.setattr(cache, "prepare_splits", lambda models, settings: prepared)
    monkeypatch.setattr(cache.ModelLoader, "load_artifact", lambda path, strict=True: (_TinyModel(), {}, Path(path)))
    real_models, real_sources = campaign.inventory(settings)
    # A second repetition of the same condition label: on the real campaign every
    # label has several repetitions, so any notebook cell that does
    # `identity.set_index("label")` without deduplicating first breaks on the real
    # data even though it looks fine against a fixture with one row per label.
    # `reconstruction_global` re-derives its own model list via `campaign.inventory`
    # rather than accepting one, so the patched inventory (not a locally spliced
    # DataFrame) is what actually needs to carry the duplicate.
    duplicate = real_models.iloc[[0]].copy()
    duplicate["model_id"] = duplicate["model_id"] + "_repeat"
    duplicate["repetition"] = 1
    models_with_repeat = pd.concat([real_models, duplicate], ignore_index=True)
    monkeypatch.setattr(campaign, "inventory", lambda settings: (models_with_repeat, real_sources))
    cache.precompute(models_with_repeat, settings, prepared=prepared, model_loader=lambda path: _TinyModel())

    settings["global_perturbation_amplitudes"] = [0.0, 1.0]
    settings["global_perturbation_sample_pixels"] = 6
    settings["analyses"] = {"reconstruction_global": {"output_directory": str(tmp_path / "results")}}
    cache.precompute_analysis(settings, "reconstruction_global", allow_cpu=True)

    monkeypatch.setattr(campaign, "load_settings", lambda path: settings)
    monkeypatch.chdir(REPOSITORY)
    import matplotlib.pyplot as plt
    path = REPOSITORY / "assets/experiments/autoencoder_architecture/notebooks/14_09_predictive_expanded/part_2_reconstruction_global.ipynb"
    notebook = nbformat.read(path, as_version=4)
    nbformat.validate(notebook)
    namespace = {"__name__": "__main__"}
    for number, cell in enumerate(notebook.cells):
        if cell.cell_type == "code":
            exec(compile(cell.source, f"{path.name}:cell-{number}", "exec"), namespace)
            plt.close("all")

    assert list(namespace["ARTIFACT_DIR"].glob("*.csv"))
    assert namespace["PER_IMAGE_AVAILABLE"] is True
    assert set(namespace["ordered_labels"]) == set(namespace["identity"].label)
    # The duplicated-label row above must not have been silently dropped or collapsed.
    assert namespace["identity"].label.duplicated().any()
    assert namespace["masserstein_summary"].index.is_unique
    assert "verdict" in namespace["at_largest"].columns


def test_latent_geometry_routine_computes_sensitivity_and_structure_samples(miniature_campaign, monkeypatch, tmp_path):
    settings, prepared = miniature_campaign
    monkeypatch.setattr(cache, "prepare_splits", lambda models, settings: prepared)
    monkeypatch.setattr(cache.ModelLoader, "load_artifact", lambda path, strict=True: (_TinyModel(), {}, Path(path)))
    models, _ = campaign.inventory(settings)

    settings["pair_count"] = 10  # the fixture's evaluation split has only 6 pixels
    settings["epsilons"] = [0.01, 0.1]
    settings["analyses"] = {"latent_geometry": {"output_directory": str(tmp_path / "results")}}
    output = cache.precompute_analysis(settings, "latent_geometry", allow_cpu=True)

    sensitivity = cache.load_analysis_table(settings, "latent_geometry", "angular_sensitivity")
    assert set(sensitivity.model_id) == set(models.model_id)
    assert set(sensitivity.epsilon) == {0.01, 0.1}
    assert (sensitivity.mean_angle_degrees >= 0).all()

    structure = cache.load_analysis_table(settings, "latent_geometry", "angular_structure")
    assert set(structure.model_id) == set(models.model_id)
    assert {"observed_mean_cos_theta", "observed_sd_cos_theta"} <= set(structure.columns)

    samples = cache.load_analysis_table(settings, "latent_geometry", "angular_structure_samples")
    assert set(samples.model_id) == set(models.model_id)
    # `structure_test` may return fewer than `pair_count` distinct pairs on a tiny
    # sample (6 pixels here); every model must still get at least one, and the same
    # count as every other model since they share the same rng draw and pair_count.
    counts = samples.groupby("model_id").size()
    assert (counts > 0).all()
    assert counts.nunique() == 1

    metadata = cache.load_analysis_metadata(settings, "latent_geometry")
    assert metadata["analysis"] == "latent_geometry"
    assert metadata["models"] == 2
    assert (output / "angular_sensitivity.csv").is_file()


def test_latent_geometry_notebook_runs_end_to_end(miniature_campaign, monkeypatch, tmp_path):
    settings, prepared = miniature_campaign
    # `neighbours` is part of the base cache's provenance identity, so it must be set to
    # a value valid for this tiny fixture (geometry_similarity's trustworthiness/continuity
    # need n_neighbors < n_samples/2) BEFORE the base precompute runs, not after — changing
    # it afterwards would make `load_table` see a provenance mismatch and refuse to load.
    settings["neighbours"] = 1
    monkeypatch.setattr(cache, "prepare_splits", lambda models, settings: prepared)
    monkeypatch.setattr(cache.ModelLoader, "load_artifact", lambda path, strict=True: (_TinyModel(), {}, Path(path)))
    models, _ = campaign.inventory(settings)
    cache.precompute(models, settings, prepared=prepared, model_loader=lambda path: _TinyModel())

    settings["pair_count"] = 10  # the fixture's evaluation split has only 6 pixels
    settings["epsilons"] = [0.01, 0.1]
    settings["analyses"] = {"latent_geometry": {"output_directory": str(tmp_path / "results")}}
    cache.precompute_analysis(settings, "latent_geometry", allow_cpu=True)

    monkeypatch.setattr(campaign, "load_settings", lambda path: settings)
    monkeypatch.chdir(REPOSITORY)
    import matplotlib.pyplot as plt
    path = REPOSITORY / "assets/experiments/autoencoder_architecture/notebooks/14_09_predictive_expanded/part_3_latent_geometry.ipynb"
    notebook = nbformat.read(path, as_version=4)
    nbformat.validate(notebook)
    namespace = {"__name__": "__main__"}
    for number, cell in enumerate(notebook.cells):
        if cell.cell_type == "code":
            exec(compile(cell.source, f"{path.name}:cell-{number}", "exec"), namespace)
            plt.close("all")

    assert list(namespace["ARTIFACT_DIR"].glob("*.csv"))
    # The fixture's two models are each the only member of their own condition, so no
    # same-condition/different-seed pair exists — the graceful-empty branch is what
    # this asserts, not a real reproducibility comparison.
    assert namespace["reproducibility"].empty
    assert set(namespace["CONDITION_ORDER"]) == set(models.label)
    assert "geometry_baseline_contrasts" in namespace
    assert (namespace["ARTIFACT_DIR"] / "geometry_baseline_contrasts.csv").is_file()


def test_prediction_global_notebook_runs_end_to_end(miniature_campaign, monkeypatch, tmp_path):
    settings, prepared = miniature_campaign
    monkeypatch.setattr(cache, "prepare_splits", lambda models, settings: prepared)
    monkeypatch.setattr(cache.ModelLoader, "load_artifact", lambda path, strict=True: (_TinyModel(), {}, Path(path)))
    models, _ = campaign.inventory(settings)
    cache.precompute(models, settings, prepared=prepared, model_loader=lambda path: _TinyModel())

    monkeypatch.setattr(campaign, "load_settings", lambda path: settings)
    monkeypatch.chdir(REPOSITORY)
    import matplotlib.pyplot as plt
    path = REPOSITORY / "assets/experiments/autoencoder_architecture/notebooks/14_09_predictive_expanded/part_4_prediction_global.ipynb"
    notebook = nbformat.read(path, as_version=4)
    nbformat.validate(notebook)
    namespace = {"__name__": "__main__"}
    for number, cell in enumerate(notebook.cells):
        if cell.cell_type == "code":
            source = cell.source
            if 'RESULTS = NOTEBOOK_DIR / "part_4_prediction_global_results"' in source:
                source = source.replace('RESULTS = NOTEBOOK_DIR / "part_4_prediction_global_results"',
                                        f'RESULTS = Path({str(tmp_path)!r})')
            exec(compile(source, f"{path.name}:cell-{number}", "exec"), namespace)
            plt.close("all")

    assert list(namespace["RESULTS"].glob("*.csv"))
    assert set(namespace["CONDITION_ORDER"]) == set(models.label)
    assert namespace["baseline_label"] == "binary (ClassBalancedMultiLabelBCELoss)"
    assert {"average_precision", "roc_auc", "ap_above_prevalence"} <= set(namespace["summary_table"].columns)


def test_prediction_global_vs_baseline_notebook_runs_end_to_end(miniature_campaign, monkeypatch, tmp_path):
    settings, prepared = miniature_campaign
    monkeypatch.setattr(cache, "prepare_splits", lambda models, settings: prepared)
    monkeypatch.setattr(cache.ModelLoader, "load_artifact", lambda path, strict=True: (_TinyModel(), {}, Path(path)))
    models, _ = campaign.inventory(settings)
    cache.precompute(models, settings, prepared=prepared, model_loader=lambda path: _TinyModel())

    monkeypatch.setattr(campaign, "load_settings", lambda path: settings)
    monkeypatch.chdir(REPOSITORY)
    import matplotlib.pyplot as plt
    path = REPOSITORY / "assets/experiments/autoencoder_architecture/notebooks/14_09_predictive_expanded/part_5_prediction_global_vs_baseline.ipynb"
    notebook = nbformat.read(path, as_version=4)
    nbformat.validate(notebook)
    namespace = {"__name__": "__main__"}
    for number, cell in enumerate(notebook.cells):
        if cell.cell_type == "code":
            source = cell.source
            if 'RESULTS = NOTEBOOK_DIR / "part_5_prediction_global_vs_baseline_results"' in source:
                source = source.replace('RESULTS = NOTEBOOK_DIR / "part_5_prediction_global_vs_baseline_results"',
                                        f'RESULTS = Path({str(tmp_path)!r})')
            exec(compile(source, f"{path.name}:cell-{number}", "exec"), namespace)
            plt.close("all")

    assert list(namespace["RESULTS"].glob("*"))
    assert set(namespace["selected"].label) <= set(models.label)
    assert {"label", "pairs", "mean_difference", "ci_low", "ci_high"} <= set(namespace["selected"].columns)


def test_prediction_local_notebook_runs_end_to_end(miniature_campaign, monkeypatch, tmp_path):
    settings, prepared = miniature_campaign
    monkeypatch.setattr(cache, "prepare_splits", lambda models, settings: prepared)
    monkeypatch.setattr(cache.ModelLoader, "load_artifact", lambda path, strict=True: (_TinyModel(), {}, Path(path)))
    models, _ = campaign.inventory(settings)
    cache.precompute(models, settings, prepared=prepared, model_loader=lambda path: _TinyModel())

    # Small historical catalogue for the real class-stratification notebook, matching
    # the fixture's two classes "A"/"B" (mirrors `test_full_cache_resume_and_all_notebook_cells`).
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

    monkeypatch.setattr(campaign, "load_settings", lambda path: settings)
    monkeypatch.chdir(REPOSITORY)
    import matplotlib.pyplot as plt
    path = REPOSITORY / "assets/experiments/autoencoder_architecture/notebooks/14_09_predictive_expanded/part_6_prediction_local.ipynb"
    notebook = nbformat.read(path, as_version=4)
    nbformat.validate(notebook)
    namespace = {"__name__": "__main__"}
    for number, cell in enumerate(notebook.cells):
        if cell.cell_type == "code":
            source = cell.source
            if "NOTEBOOK_DIR = SETTINGS_PATH.parent" in source:
                source = source.replace("NOTEBOOK_DIR = SETTINGS_PATH.parent", f"NOTEBOOK_DIR = Path({str(tmp_path)!r})")
            exec(compile(source, f"{path.name}:cell-{number}", "exec"), namespace)
            plt.close("all")

    assert list(namespace["RESULTS"].glob("*"))
    assert set(namespace["strata"].class_name) == {"A", "B"}
    assert set(namespace["head_probe"].split) == {"validation", "test"}
    assert np.isfinite(namespace["head_probe"].query("population == 'annotation_retrieval'")["value_head"]).all()
    assert set(namespace["gaps"].metric) >= {"average_precision", "roc_auc"}


def test_decision_summary_notebook_runs_end_to_end(miniature_campaign, monkeypatch, tmp_path):
    settings, prepared = miniature_campaign
    monkeypatch.setattr(cache, "prepare_splits", lambda models, settings: prepared)
    monkeypatch.setattr(cache.ModelLoader, "load_artifact", lambda path, strict=True: (_TinyModel(), {}, Path(path)))
    models, _ = campaign.inventory(settings)
    # Richer, multi-epoch history (as in the part_0_0 notebook test) so `run_durations`
    # carries real, non-degenerate values for the "affordable" decision criterion.
    first_epoch_duration = {"predictive_initial/task_000000": 12.0, "historical_bce/task_000000": 9.0}
    for model in models.to_dict("records"):
        entries = [{"phase": "joint", "metrics": {"epoch": epoch, "duration": first_epoch_duration[model["model_id"]] + epoch,
                                                  "total_loss": 1.0 - 0.1 * epoch, "is_best": epoch > 1}}
                  for epoch in (1, 2, 3)]
        entries.append({"phase": "joint", "split": "test", "metrics": {"masserstein": .2}})
        (Path(model["artifact"]) / "config" / "history.json").write_text(json.dumps(entries))
    cache.precompute(models, settings, prepared=prepared, model_loader=lambda path: _TinyModel())

    # Sibling notebooks' own derived tables, computed directly with the same shared
    # helpers those notebooks use (rather than executing their full cell sequences).
    dynamics_results = tmp_path / "part_0_0_campaign_training_dynamics_results"
    prediction_results = tmp_path / "part_4_prediction_global_results"
    dynamics_results.mkdir()
    prediction_results.mkdir()
    history = campaign.history_components(campaign.training_history(models))
    campaign.run_duration_frame(history, models).to_csv(dynamics_results / "run_durations.csv", index=False)
    prediction = cache.load_table(settings, "prediction")
    contrast_source = prediction.query("split == 'validation' and population == 'annotation_retrieval' and "
                                       "scope == 'train_supported' and metric in ['average_precision', 'roc_auc', 'ap_above_prevalence']")
    units = reports.experimental_units(contrast_source, models, ["metric"])
    _, contrasts = reports.paired_comparisons(units, ["metric"])
    reports.baseline_contrasts(contrasts, models).to_csv(prediction_results / "baseline_contrasts.csv", index=False)

    monkeypatch.setattr(campaign, "load_settings", lambda path: settings)
    monkeypatch.chdir(REPOSITORY)
    import matplotlib.pyplot as plt
    path = REPOSITORY / "assets/experiments/autoencoder_architecture/notebooks/14_09_predictive_expanded/part_7_decision_summary_geometry_vs_prediction.ipynb"
    notebook = nbformat.read(path, as_version=4)
    nbformat.validate(notebook)
    namespace = {"__name__": "__main__"}
    for number, cell in enumerate(notebook.cells):
        if cell.cell_type == "code":
            source = cell.source
            if "NOTEBOOK_DIR = SETTINGS_PATH.parent" in source:
                source = source.replace("NOTEBOOK_DIR = SETTINGS_PATH.parent", f"NOTEBOOK_DIR = Path({str(tmp_path)!r})")
            exec(compile(source, f"{path.name}:cell-{number}", "exec"), namespace)
            plt.close("all")

    assert list(namespace["ARTIFACT_DIR"].glob("*"))
    decision_frame = namespace["decision_frame"]
    assert {"beats_baseline", "keeps_variation", "affordable", "carried_forward"} <= set(decision_frame.columns)
    assert len(decision_frame) == 1  # one swept condition (predictive_initial) against the baseline
    # historical_bce's own epoch durations (10/11/12s) against itself: cost_multiple == 1.
    assert namespace["baseline_epoch_seconds"] == pytest.approx(11.0)


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
