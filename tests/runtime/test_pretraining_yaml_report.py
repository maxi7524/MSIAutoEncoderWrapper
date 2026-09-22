"""Tests for the campaign-local YAML planning report."""

from __future__ import annotations

from copy import deepcopy
import importlib.util
from pathlib import Path
import sys
from types import ModuleType

import yaml

from msi_autoencoder_wrapper.runtime import build_plan, load_experiment_config


CAMPAIGN = (
    Path(__file__).parents[2]
    / "assets/experiments/autoencoder_architecture/experiment_runs_configs"
    / "segmentation_model/20_09_26_metaspace_base_pretrain"
)


def _load_script(name: str) -> ModuleType:
    """Import one campaign-local script without package side effects."""
    path = CAMPAIGN / name
    specification = importlib.util.spec_from_file_location(path.stem, path)
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


def test_report_contains_all_materialized_roles_and_seed_sources() -> None:
    """The static report exposes task lineage and all configured seed levels."""
    report = _load_script("generate_yaml_report.py")
    content = report.render_report(CAMPAIGN / "pretraining_experiment.yaml")

    assert "Planned tasks: **310**; logical groups: **110**" in content
    assert "'pretrained': 100" in content
    assert "'frozen_head': 100" in content
    assert "'unfrozen_head': 100" in content
    assert "| Axis | Population | Artifact seed | Construction |" in content
    assert "| axis-200-900 | axis | 42 |" in content
    assert "| axis-100-3000 | axis | 42 |" in content
    assert "| 0 | 42 | 43 | 1000 |" in content
    assert "__pretrained |" in content
    assert "__frozen_head |" in content
    assert "__unfrozen_head |" in content
    task_section = content.split("## Every planned model task\n", 1)[1]
    assert sum(line.startswith("| task_") for line in task_section.splitlines()) == 310
    assert "| axes | axis-100-3000 |" in content
    assert "| architectures | conv1d-ae-32-16-8-latent-10 |" in content


def test_report_uses_explicit_execution_id_for_all_model_names() -> None:
    """The reviewed report names models in the repaired Entropy campaign."""
    report = _load_script("generate_yaml_report.py")
    campaign_id = "metaspace-pretrain-repaired-20260923-01"
    content = report.render_report(
        CAMPAIGN / "pretraining_experiment.yaml", execution_id=campaign_id
    )

    assert f"Execution campaign ID: {campaign_id}" in content
    assert content.count(f"{campaign_id}__grid_") == 310
    assert "kidney-metaspace-pretraining-final__cfg_b8e15a957f6c__grid_" not in content
    assert "Source: pretraining_experiment.yaml" in content


def test_report_escapes_nested_configuration_in_tables() -> None:
    """Structured parameters remain readable without breaking Markdown cells."""
    report = _load_script("generate_yaml_report.py")

    assert report._cell({"field": "a|b"}) == '{"field":"a\\|b"}'
    assert report._cell("line one\nline two") == "line one<br>line two"


def test_old_plan_comparison_detects_seed_change(tmp_path: Path) -> None:
    """Seed parity is checked on a logical trial, independently of task IDs."""
    report = _load_script("generate_yaml_report.py")
    task = build_plan(load_experiment_config(CAMPAIGN / "pretraining_experiment.yaml")).tasks[0]
    plan_path = tmp_path / "resolved-experiment.yaml"
    plan_path.write_text("{}\n", encoding="utf-8")
    task_dir = tmp_path / "tasks"
    task_dir.mkdir()
    descriptor = {
        "grid_parameters": task.grid_parameters,
        "repetition": task.repetition,
        "reproducibility": deepcopy(task.reproducibility),
    }
    task_path = task_dir / "task_000000.yaml"
    task_path.write_text(yaml.safe_dump(descriptor), encoding="utf-8")
    assert "'same': 4" in "\n".join(report._old_seed_comparison(plan_path, (task,)))

    descriptor["reproducibility"]["derived_run_seeds"]["training"] += 1
    task_path.write_text(yaml.safe_dump(descriptor), encoding="utf-8")
    assert "'DIFFERENT': 1" in "\n".join(report._old_seed_comparison(plan_path, (task,)))
