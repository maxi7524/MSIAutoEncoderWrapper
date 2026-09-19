"""Unit tests for catalog-driven common analysis precompute orchestration."""

from pathlib import Path

import pandas as pd
import pytest

from msi_autoencoder_wrapper.analysis.precompute.core.artifacts import ArtifactStore
from msi_autoencoder_wrapper.analysis.precompute.core.contracts import ArtifactSpec
from msi_autoencoder_wrapper.analysis.precompute.core.context import AnalysisContext
from msi_autoencoder_wrapper.analysis.precompute.core.planner import build_plan
from msi_autoencoder_wrapper.analysis.precompute.core.resources import ResourceManager
from msi_autoencoder_wrapper.analysis.precompute.core.runner import _resolve_output_directories
from msi_autoencoder_wrapper.analysis.precompute.notebook_inputs import select_catalog_frame
from msi_autoencoder_wrapper.analysis.precompute.core.strategy_loader import load_strategy
from msi_autoencoder_wrapper.analysis.precompute.model_catalog.exports import write_catalog_artifacts
from msi_autoencoder_wrapper.analysis.precompute.model_catalog.resolver import resolve_model_catalog
from msi_autoencoder_wrapper.analysis.autoencoder.experiments import contractive_precompute, predictive_precompute
from msi_autoencoder_wrapper.analysis.autoencoder.evidence import precompute_plugin as evidence_plugin
from msi_autoencoder_wrapper.visualization.analysis_catalog import (
    load_visualization_contract,
    model_style_map,
    theme_from_visualization_contract,
)


@pytest.fixture
def joint_inventory() -> pd.DataFrame:
    """Two campaign sources with reused grid ids and two repetitions each."""
    rows = []
    for source, grid_ids in (("vpu", ("grid_0000", "grid_0005")),
                             ("contractive", ("grid_0000", "grid_0001"))):
        for grid_id in grid_ids:
            for repetition in (0, 1):
                rows.append({
                    "source": source,
                    "grid_id": grid_id,
                    "task_id": f"{source}-{grid_id}-{repetition}",
                    "model_id": f"{source}/{grid_id}/{repetition}",
                    "repetition": repetition,
                    "label": "vpu (VariationalPULoss)",
                    "ready": True,
                })
    return pd.DataFrame(rows)


def test_catalog_uses_technical_selector_and_persists_visual_encoding(joint_inventory, tmp_path):
    """One raw label may resolve to four stable aliases without any condition branch."""
    settings = {
        "models": {
            "vpu": {
                "display_label": "vpu",
                "select": {"source": "vpu", "grid_id": "grid_0000"},
                "tags": {"contrastive": False},
                "visualization": {"order": 0, "color": "#0072B2"},
            },
            "vpu_contractive": {
                "display_label": "vpu + contractive",
                "select": {"source": "contractive", "grid_id": "grid_0000"},
                "tags": {"contractive": True},
                "visualization": {"order": 1, "color": "#D55E00", "line_style": "dashed"},
            },
        },
        "groups": {"comparison": ["vpu", "vpu_contractive"]},
    }

    catalog = resolve_model_catalog(settings, joint_inventory)

    assert catalog.aliases("comparison") == ("vpu", "vpu_contractive")
    assert catalog.records.groupby("model_alias").repetition.nunique().to_dict() == {
        "vpu": 2,
        "vpu_contractive": 2,
    }
    assert set(catalog.records.label) == {"vpu", "vpu + contractive"}

    write_catalog_artifacts(tmp_path, catalog)
    contract = load_visualization_contract(tmp_path / "visualization_contract.csv")
    theme = theme_from_visualization_contract(contract)
    styles = model_style_map(contract)
    assert theme.color_for_model("vpu") == "#0072B2"
    assert styles["vpu + contractive"]["line_style"] == "dashed"


def test_default_precompute_directory_is_beside_the_settings_file(tmp_path):
    """A legacy YAML without a precompute block never creates a repository-root folder."""
    repository = tmp_path / "repository"
    settings_directory = repository / "assets" / "notebooks" / "campaign"
    settings = {"analyses": {"local": {"output_directory": "assets/results/local"}}}

    _resolve_output_directories(settings, repository, settings_directory / "precompute")

    assert settings["precompute"]["output_directory"] == str(settings_directory / "precompute")
    assert settings["analyses"]["local"]["output_directory"] == str(repository / "assets/results/local")


def test_notebook_catalog_filter_maps_only_selected_models_to_display_labels():
    """Notebook presentation never re-implements a condition-specific selector."""
    catalog = pd.DataFrame({
        "model_id": ["first", "second"],
        "model_alias": ["base", "contrastive"],
        "display_label": ["VPU", "VPU + contrastive"],
        "source": ["campaign", "campaign"],
        "condition": ["base-condition", "contrastive-condition"],
    })
    frame = pd.DataFrame({
        "model_id": ["first", "discard"],
        "label": ["raw", "raw"],
        "metric": [0.1, 0.2],
    })

    selected = select_catalog_frame(frame, catalog)

    assert selected.to_dict("records") == [{"model_id": "first", "label": "VPU", "metric": 0.1}]


def test_catalog_rejects_ambiguous_alias_before_gpu_work(joint_inventory):
    """A raw label/source selector that spans two cells is not silently deduplicated."""
    settings = {
        "models": {
            "ambiguous": {
                "select": {"source": "vpu", "label": "vpu (VariationalPULoss)"},
            }
        }
    }

    with pytest.raises(ValueError, match="exactly once per repetition"):
        resolve_model_catalog(settings, joint_inventory)


def test_resource_manager_constructs_shared_resource_once():
    """Several plugins can ask for one decoded split/model resource safely."""
    resources = ResourceManager()
    calls = 0

    def build():
        nonlocal calls
        calls += 1
        return object()

    assert resources.get_or_create("split:test", build) is resources.get_or_create("split:test", build)
    assert calls == 1


def test_prepare_splits_uses_the_shared_resource_manager(joint_inventory, monkeypatch):
    """Decoded partitions are prepared once when several plugins request them."""
    calls = 0
    expected = ({"test": object()}, object(), object())
    settings = {
        "target_field": "metaspace",
        "pixel_fraction": 1.0,
        "sample_seed": 13,
        "batch_size": 16,
        "_precompute_resources": ResourceManager(),
    }
    models = joint_inventory[joint_inventory.source == "vpu"].copy()
    models["data_contract"] = "shared-contract"

    def build(_models, _settings):
        nonlocal calls
        calls += 1
        return expected

    monkeypatch.setattr(predictive_precompute, "_prepare_splits_uncached", build)

    assert predictive_precompute.prepare_splits(models, settings) is expected
    assert predictive_precompute.prepare_splits(models, settings) is expected
    assert calls == 1


def test_resource_manager_keeps_one_active_model_without_reloading_checkpoint():
    """Checkpoint objects remain reusable on CPU while only one is active on device."""
    class Model:
        def __init__(self):
            self.devices = []
            self.evaluations = 0

        def to(self, device):
            self.devices.append(device)
            return self

        def eval(self):
            self.evaluations += 1
            return self

    resources = ResourceManager()
    first_calls = second_calls = 0

    def first_factory():
        nonlocal first_calls
        first_calls += 1
        return Model()

    def second_factory():
        nonlocal second_calls
        second_calls += 1
        return Model()

    first = resources.activate_model("first", first_factory, "cuda")
    second = resources.activate_model("second", second_factory, "cuda")
    first_again = resources.activate_model("first", first_factory, "cuda")

    assert first is first_again
    assert first_calls == 1 and second_calls == 1
    assert first.devices == ["cuda", "cpu", "cuda"]
    assert second.devices == ["cuda", "cpu"]


def test_contractive_context_and_checkpoint_helpers_use_common_resources(monkeypatch):
    """The legacy contractive routines participate in the shared resource contract."""
    resources = ResourceManager()
    settings = {
        "workspace": "/workspace",
        "model_context": "kidney",
        "target_field": "molecule",
        "splits": ["train", "test"],
        "pixel_fraction": 1.0,
        "sample_seed": 42,
        "decode_cache": "/cache",
        "model_store": "/models",
        "device": "cuda",
        "_precompute_device": "cuda",
        "_precompute_resources": resources,
    }
    grid = pd.DataFrame({"model_name": ["run-a"]})
    context_calls = 0

    def build_context(_settings, _grid):
        nonlocal context_calls
        context_calls += 1
        return object(), {"test": object()}

    monkeypatch.setattr(contractive_precompute, "_prepare_context_uncached", build_context)
    first_context = contractive_precompute.prepare_context(settings, grid)
    assert first_context is contractive_precompute.prepare_context(settings, grid)
    assert context_calls == 1

    class Model:
        def to(self, _device):
            return self

        def eval(self):
            return self

    class Manager:
        def __init__(self):
            self.calls = 0

        def load_model(self, **_kwargs):
            self.calls += 1
            return Model()

    class Wrapper:
        def __init__(self):
            self.models_manager = Manager()

    wrapper = Wrapper()
    first_model = contractive_precompute.load_model_resource(settings, wrapper, "run-a")
    assert first_model is contractive_precompute.load_model_resource(settings, wrapper, "run-a")
    assert wrapper.models_manager.calls == 1


def test_heads_strategy_preserves_declared_dependency_safe_order(joint_inventory, tmp_path):
    """The planner validates strategy order rather than inferring a different workflow."""
    catalog = resolve_model_catalog({"models": {"vpu": {"select": {"source": "vpu", "grid_id": "grid_0000"}}}}, joint_inventory)
    settings = {
        "analyses": {
            "campaign_training_dynamics": {"output_directory": str(tmp_path / "part_0_results")},
            "reconstruction_local": {"output_directory": str(tmp_path / "part_1_results")},
        },
    }
    context = AnalysisContext(settings, catalog, ArtifactStore(tmp_path / "precompute"), ResourceManager())
    strategy = load_strategy("autoencoder.heads_general_analysis")

    plan = build_plan(strategy, context)

    assert [stage.name for stage in plan.stages] == [
        "autoencoder.reconstruction.campaign_training_dynamics",
        "autoencoder.heads.shared_inference",
        "autoencoder.reconstruction.local",
    ]


def test_heads_strategy_adds_evidence_before_shared_inference_when_configured(joint_inventory, tmp_path):
    """An evidence notebook cache is an explicit optional stage, not a cross-campaign input."""
    catalog = resolve_model_catalog(
        {"models": {"vpu": {"select": {"source": "vpu", "grid_id": "grid_0000"}}}},
        joint_inventory,
    )
    settings = {
        "annotation_evidence_cache": str(tmp_path / "evidence.npz"),
        "analyses": {
            "campaign_training_dynamics": {"output_directory": str(tmp_path / "part_0_results")},
        },
    }
    context = AnalysisContext(settings, catalog, ArtifactStore(tmp_path / "precompute"), ResourceManager())

    plan = build_plan(load_strategy("autoencoder.heads_general_analysis"), context)

    assert [stage.name for stage in plan.stages] == [
        "autoencoder.reconstruction.campaign_training_dynamics",
        "autoencoder.evidence.annotation_population",
        "autoencoder.heads.shared_inference",
    ]


def test_annotation_evidence_plugin_uses_campaign_threshold_boundary(joint_inventory, monkeypatch, tmp_path):
    """A configured threshold is added exactly, rather than interpolated from an old cache."""
    cache_path = tmp_path / "evidence.npz"
    settings = {
        "annotation_evidence_cache": str(cache_path),
        "evidence": {"relative_threshold": 0.0119, "bin_radius": 1},
        "experiment_config": str(tmp_path / "campaign.yaml"),
        "target_field": "molecule",
        "batch_size": 2048,
        "device": "cuda",
    }
    catalog = resolve_model_catalog(
        {"models": {"vpu": {"select": {"source": "vpu", "grid_id": "grid_0000"}}}},
        joint_inventory,
    )
    context = AnalysisContext(settings, catalog, ArtifactStore(tmp_path / "precompute"), ResourceManager())
    observed = {}

    monkeypatch.setattr(evidence_plugin, "build_population_dataset", lambda _path: (object(), object()))
    monkeypatch.setattr(evidence_plugin.predictive_precompute, "resolve_device", lambda *_args, **_kwargs: "cuda")

    class Result:
        def save(self, path):
            Path(path).write_text("complete evidence cache")

    class Precompute:
        def __init__(self, _dataset, **kwargs):
            observed.update(kwargs)

        def run(self, *, progress):
            assert progress is True
            return Result()

    monkeypatch.setattr(evidence_plugin, "EvidencePrecompute", Precompute)
    plugin = evidence_plugin.annotation_evidence_plugin()
    plugin.run(context)

    assert cache_path.is_file()
    assert observed["batch_size"] == 512
    assert observed["bin_radii"] == (0, 1, 2)
    assert 0.0119 in observed["candidate_relative_thresholds"]
    assert (observed["grid"].relative_edges == 0.0119).any()
    context.artifacts.verify(context, plugin.provides[0])


def test_artifact_store_verifies_shared_file_artifact(tmp_path):
    """A strategy contract can verify a shared artifact that is one file, not a directory."""
    output = tmp_path / "artifact.npz"
    output.write_text("complete")
    context = AnalysisContext({"output": str(output)}, None, ArtifactStore(tmp_path), ResourceManager())

    context.artifacts.verify(context, ArtifactSpec("archive", root_setting="output", path_kind="file"))


def test_reconstruction_local_resolves_catalog_aliases_before_any_model_load(joint_inventory, monkeypatch):
    """The local analysis receives four unambiguous aliases, not raw duplicate labels."""
    settings = {
        "device": "cpu",
        "batch_size": 2,
        "sample_seed": 1,
        "models": {
            "vpu": {"display_label": "vpu", "select": {"source": "vpu", "grid_id": "grid_0000"}},
            "vpu_contrastive": {"display_label": "vpu + contrastive", "select": {"source": "vpu", "grid_id": "grid_0005"}},
            "vpu_contractive": {"display_label": "vpu + contractive", "select": {"source": "contractive", "grid_id": "grid_0000"}},
            "vpu_contractive_contrastive": {"display_label": "vpu + contractive + contrastive", "select": {"source": "contractive", "grid_id": "grid_0001"}},
        },
        "selected_models": [
            "vpu",
            "vpu_contrastive",
            "vpu_contractive",
            "vpu_contractive_contrastive",
        ],
        "repetition": 0,
    }
    catalog = resolve_model_catalog(settings, joint_inventory)
    settings["_resolved_model_catalog"] = catalog.records

    class SelectionObserved(Exception):
        pass

    def stop_after_selection(models, _settings):
        assert models.model_alias.tolist() == [
            "vpu",
            "vpu_contrastive",
            "vpu_contractive",
            "vpu_contractive_contrastive",
        ]
        assert models.label.tolist() == [
            "vpu",
            "vpu + contrastive",
            "vpu + contractive",
            "vpu + contractive + contrastive",
        ]
        raise SelectionObserved

    monkeypatch.setattr(predictive_precompute, "prepare_splits", stop_after_selection)
    with pytest.raises(SelectionObserved):
        predictive_precompute.reconstruction_local(settings)
