"""Cache contracts, legacy adoption, analysis reuse and model cells of the pretraining analyses."""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from msi_autoencoder_wrapper.analysis.autoencoder.experiments import pretraining_campaign as campaign
from msi_autoencoder_wrapper.analysis.autoencoder.experiments import pretraining_campaign_cache as contracts
from msi_autoencoder_wrapper.analysis.autoencoder.experiments import pretraining_campaign_precompute as family


def _settings(tmp_path) -> dict:
    return {"cache_directory": str(tmp_path / "cache"), "repository_root": str(tmp_path),
            "settings_path": str(tmp_path / "analysis_settings.yaml"),
            "axes": {"axis-a": {"directory": "axis_a", "label": "A"}}}


def _passing(directory) -> pd.DataFrame:
    return pd.DataFrame([{"check": "content", "subject": directory.name, "passed": True, "detail": ""}])


def _failing(directory) -> pd.DataFrame:
    return pd.DataFrame([{"check": "content", "subject": directory.name, "passed": False, "detail": "differs"}])


def test_level_is_created_once_and_reused_by_contract(tmp_path) -> None:
    settings = _settings(tmp_path)
    contract = {"level": "populations", "schema": 1, "settings": {"windows": (100.0, 1e-3)}}
    first = contracts.resolve_level(settings, "populations", contract)
    assert not first.complete
    (first.directory / "complete.json").write_text("{}")
    ## The tuple is compared in its stored JSON form
    second = contracts.resolve_level(settings, "populations", contract)
    assert second.directory == first.directory and second.complete
    changed = contracts.resolve_level(settings, "populations", {**contract, "schema": 2})
    assert changed.directory != first.directory


@pytest.mark.parametrize("validator, adopted", [(_passing, True), (_failing, False)])
def test_legacy_directory_is_adopted_only_after_passing_validation(tmp_path, validator, adopted) -> None:
    settings = _settings(tmp_path)
    legacy = tmp_path / "cache" / "populations" / "legacy"
    legacy.mkdir(parents=True)
    (legacy / "complete.json").write_text("{}")
    contract = {"level": "populations", "schema": 1}
    level = contracts.resolve_level(settings, "populations", contract, validator)

    assert (level.directory == legacy) is adopted
    assert level.complete is adopted and level.adopted is adopted
    if adopted:
        stored = json.loads((legacy / contracts.CONTRACT_FILE).read_text())
        assert stored["contract"] == contract
        assert stored["audit"]["validation"][0]["passed"]
    else:
        ## A rejected directory is left untouched and the level starts in a new directory
        assert contracts.read_contract(legacy) is None


def test_task_parameters_are_compared_outside_resolver_rewritten_entries(tmp_path) -> None:
    parameters = {"training": {"epochs": 10}, "factory_parameters": {"dataset": {"parameters": {
        "subset": {"fraction": 0.2}, "target_specs": {"molecule": {"type": "multi_label"}}}}}}
    resolved = json.loads(json.dumps(parameters))
    resolved["factory_parameters"]["dataset"]["parameters"]["subset"] = {"method": "source_indices"}
    resolved["factory_parameters"]["dataset"]["parameters"]["target_specs"]["molecule"]["class_mapping"] = {"a": 0}
    resolved["resolved"] = {"split_manifest": str(tmp_path / "split.yaml")}
    (tmp_path / "split.yaml").write_text("assignments: {}")
    assert contracts.comparable_task_parameters(resolved) == contracts.comparable_task_parameters(parameters)

    directory = tmp_path / "plan"
    directory.mkdir()
    (directory / "resolved_parameters.json").write_text(json.dumps({"task_1": resolved}))
    task = SimpleNamespace(task_id="task_1", parameters=parameters)
    assert contracts.validate_legacy_plan(directory, [task]).passed.all()
    changed = SimpleNamespace(task_id="task_1", parameters={**parameters, "training": {"epochs": 11}})
    report = contracts.validate_legacy_plan(directory, [changed]).set_index("check")
    assert not report.loc["task_parameters", "passed"]


def test_stage_results_have_their_own_directories(tmp_path) -> None:
    settings = {**_settings(tmp_path), "analyses": {"overview": {"stages": {
        "pretrained": {"output_directory": "part_2", "result_directory": "part_2_01_results"},
        "frozen_head": {"output_directory": "part_3", "stage_directory": "frozen_head",
                        "result_directory": "part_3_01_results"}}}}}
    assert family.results_directory(settings, "overview", "axis-a", "pretrained") == (
        tmp_path / "part_2/axis_a/part_2_01_results")
    assert family.results_directory(settings, "overview", "axis-a", "frozen_head") == (
        tmp_path / "part_3/axis_a/frozen_head/part_3_01_results")
    with pytest.raises(ValueError, match="per stage"):
        family.results_directory(settings, "overview", "axis-a")
    with pytest.raises(ValueError, match="scope"):
        family.register("x", per_axis=True, scope="unknown")


def test_results_are_current_only_for_identical_inputs(tmp_path) -> None:
    cache = SimpleNamespace(keys={"plan": "p1", "populations": "q1", "inference": "i1"})
    inputs = {"analysis": "a", "models": ["m0", "m1"], "cache": {"plan": "x"}}
    (tmp_path / "metadata.json").write_text(json.dumps({"inputs": inputs}))
    assert family._is_current(tmp_path, inputs, cache)
    assert not family._is_current(tmp_path, {**inputs, "models": ["m0"]}, cache)
    ## Results written before input records: same models and same cache directories
    (tmp_path / "metadata.json").write_text(json.dumps({"models": ["m1", "m0"], "cache_keys": dict(cache.keys)}))
    assert family._is_current(tmp_path, inputs, cache)
    stale = SimpleNamespace(keys={**cache.keys, "inference": "i2"})
    assert not family._is_current(tmp_path, inputs, stale)


def _task(phases: list[dict], role: str = "pretrained", task_id: str = "task_x", parent: str | None = None
          ) -> SimpleNamespace:
    return SimpleNamespace(task_id=task_id, parameters={"training": {"phases": phases}},
                           workflow={"role": role, "parent_task_id": parent})


def _phase(name: str, *, epochs: int = 10, head: str | None = "vpu", freeze: tuple = (), synthetic: bool = False
           ) -> dict:
    phase = {"phase_name": name, "epochs": epochs, "freeze": list(freeze),
             "criterions": {"reconstruction": {"m": {"target": "MassersteinLoss"}},
                            "heads": {"h": {head: {"target": "VariationalPULoss" if head == "vpu"
                                                   else "MultiLabelBCELoss", "weight": 0.2}}}}}
    if synthetic:
        phase["pretraining"] = {"population": "axis"}
    return phase


def test_variants_are_identified_by_their_synthetic_phases() -> None:
    variants = {"joint": {"phases": ["synthetic_joint"]}, "staged": {"phases": ["single", "permutation"]}}
    staged = _task([_phase("single", head="bce", synthetic=True), _phase("permutation", head="bce", synthetic=True),
                    _phase("real", epochs=0)])
    assert campaign.task_variant(staged, variants) == "staged"
    assert campaign.task_variant(_task([_phase("real")], role="real_only"), variants) == campaign.BASELINE_VARIANT
    ## Unconfigured phase sequences keep their phase names and receive no alias
    assert campaign.task_variant(_task([_phase("other", synthetic=True)]), variants) == "other"
    with pytest.raises(ValueError, match="matches variants"):
        campaign.task_variant(_task([_phase("other", synthetic=True)]), {"x": {"phases": ["other"]},
                                                                         "y": {"phases": ["other"]}})


def test_fine_tuning_tasks_inherit_variant_and_head_family_from_their_parent() -> None:
    parent = _task([_phase("synthetic_joint", head="bce", synthetic=True), _phase("real_test", epochs=0)],
                   task_id="task_p")
    frozen = _task([_phase("real", freeze=("heads.h",))], role="frozen_head", task_id="task_f", parent="task_p")
    unfrozen = _task([_phase("real")], role="unfrozen_head", task_id="task_u", parent="task_p")
    tasks = {task.task_id: task for task in (parent, frozen, unfrozen)}
    variants = {"joint": {"phases": ["synthetic_joint"]}}

    assert campaign.task_variant(frozen, variants, tasks) == "joint"
    assert campaign._head_objective(frozen, tasks)["family"] == "MultiLabelBCELoss"
    assert campaign._head_objective(unfrozen, tasks)["family"] == "VariationalPULoss"
    with pytest.raises(ValueError, match="Parent task"):
        campaign.lineage_phases(frozen)


def test_head_family_is_the_loss_that_last_trained_the_head() -> None:
    pretrained = _task([_phase("synthetic", head="bce", synthetic=True), _phase("real_test", epochs=0)])
    frozen = _task([_phase("synthetic", head="bce", synthetic=True), _phase("real", freeze=("heads.h",))])
    unfrozen = _task([_phase("synthetic", head="bce", synthetic=True), _phase("real")])

    assert campaign._head_objective(pretrained)["family"] == "MultiLabelBCELoss"
    assert campaign._head_objective(frozen)["family"] == "MultiLabelBCELoss"
    assert campaign._head_objective(unfrozen)["family"] == "VariationalPULoss"
    ## The evaluated objective is always the last (real-data) phase
    assert json.loads(campaign._head_objective(frozen)["objective_json"])["heads"]["h"]["vpu"]["weight"] == 0.2


def test_cell_models_are_generated_per_axis_variant_and_stage() -> None:
    settings = {"axes": {"axis-a": {"directory": "axis_a"}, "axis-b": {"directory": "axis_b"}},
                "models": {"baseline_a": {"select": {"axis": "axis-a", "role": "real_only"}}},
                "variants": {"v1": {"phases": ["p"], "label": "V1", "color": "#111111", "factors": {"rare": 1}}},
                "stages": {"pretrained": {"label": "pre", "line_style": "dotted"},
                           "frozen_head": {"label": "frozen"}}}
    campaign.expand_cell_models(settings)
    models = settings["models"]

    assert list(models)[0] == "baseline_a"
    assert list(models)[1:] == ["v1__pretrained__axis_a", "v1__frozen_head__axis_a",
                                "v1__pretrained__axis_b", "v1__frozen_head__axis_b"]
    cell = models["v1__pretrained__axis_b"]
    assert cell["select"] == {"axis": "axis-b", "role": "pretrained", "variant": "v1"}
    assert cell["visualization"]["color"] == "#111111" and cell["visualization"]["line_style"] == "dotted"
    assert cell["tags"]["rare"] == 1


def test_model_configs_compare_without_serialization_only_fields() -> None:
    resolved = {"family": "autoencoder", "components": {
        "encoder": {"type": "CNNEncoder", "parameters": {"latent_dim": 10}},
        "heads": {"molecule": {"type": "LinearClassificationHead", "parameters": {"output_dim": 5}}}}}
    saved = {"family": "autoencoder", "preset": None, "components": {
        "encoder": {"type": "CNNEncoder", "module": "pkg.cnn_encoder", "parameters": {"latent_dim": 10}},
        "heads": {"molecule": {"type": "LinearClassificationHead", "module": "pkg.head",
                               "parameters": {"output_dim": 5}}}}}
    assert campaign.comparable_model_config(resolved) == campaign.comparable_model_config(saved)
    ## A real architecture difference is still detected
    saved["components"]["heads"]["molecule"]["parameters"]["output_dim"] = 6
    assert campaign.comparable_model_config(resolved) != campaign.comparable_model_config(saved)


def test_array_agreement_requires_identical_undefined_entries_and_tolerates_rounding() -> None:
    from msi_autoencoder_wrapper.analysis.autoencoder.experiments.pretraining_campaign_inference import (
        array_agreement,
    )

    stored = {"within": np.array([[1.0, np.nan], [2.0, 3.0]]), "tic_error": np.array([0.0, 4e-7]),
              "masserstein": np.array([3.0, 4.0])}
    close = {"within": np.array([[1.0 + 1e-6, np.nan], [2.0, 3.0]]), "tic_error": np.array([1e-7, 0.0]),
             "masserstein": np.array([3.0 + 1e-5, 4.0])}
    assert array_agreement(close, stored)[0] == []
    moved_nan = {**close, "within": np.array([[1.0, 5.0], [2.0, 3.0]])}
    assert array_agreement(moved_nan, stored)[0] == ["within"]
    shifted = {**close, "masserstein": np.array([3.1, 4.0])}
    assert array_agreement(shifted, stored)[0] == ["masserstein"]
