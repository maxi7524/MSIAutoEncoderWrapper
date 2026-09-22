#!/usr/bin/env python3
"""Render a reviewable Markdown report from the complete pretraining YAML plan."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any

import yaml

from msi_autoencoder_wrapper.runtime import build_plan, load_experiment_config
from msi_autoencoder_wrapper.runtime.naming import run_identifier


DEFAULT_YAML = Path(__file__).with_name("pretraining_experiment.yaml")


def _cell(value: Any) -> str:
    """Render one Markdown table cell without breaking table structure."""
    if isinstance(value, (dict, list, tuple)):
        value = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return str(value).replace("|", "\\|").replace("\n", "<br>")


def _table(headers: list[str], rows: list[list[Any]]) -> list[str]:
    """Return one Markdown table from headers and data rows."""
    return [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
        *("| " + " | ".join(_cell(item) for item in row) + " |" for row in rows),
    ]


def _trial_key(task: Any) -> tuple[str, str, int]:
    """Identify one logical trial independently of its task ID."""
    grid = task.grid_parameters if hasattr(task, "grid_parameters") else task["grid_parameters"]
    repetition = task.repetition if hasattr(task, "repetition") else task["repetition"]
    return (
        grid["axes"]["name"],
        ",".join(phase["phase_name"] for phase in grid["schedules"]),
        int(repetition),
    )


def _old_seed_comparison(old_plan: Path, tasks: tuple[Any, ...]) -> list[str]:
    """Compare old and new seed streams by logical trial, not task numbering."""
    previous: dict[tuple[str, str, int], dict[str, Any]] = {}
    for path in sorted((old_plan.parent / "tasks").glob("task_*.yaml")):
        with path.open(encoding="utf-8") as stream:
            descriptor = yaml.safe_load(stream)
        previous[_trial_key(descriptor)] = descriptor["reproducibility"]
    counts: Counter[str] = Counter()
    for task in tasks:
        old = previous.get(_trial_key(task))
        if old is not None:
            for group, name in (
                ("common_seeds", "split"), ("common_seeds", "dataloader"),
                ("derived_run_seeds", "model_initialization"),
                ("derived_run_seeds", "training"),
            ):
                counts[
                    "same" if old[group][name] == task.reproducibility[group][name]
                    else "DIFFERENT"
                ] += 1
    return ["## Old-plan seed comparison", "", str(dict(counts)), ""]


def render_report(
    yaml_path: Path,
    old_plan: Path | None = None,
    execution_id: str | None = None,
) -> str:
    """Build a Markdown audit of planned seeds, phases, objectives and models.

    :param yaml_path: Source experiment YAML.
    :type yaml_path: pathlib.Path
    :param old_plan: Optional legacy resolved-experiment file for seed comparison.
    :type old_plan: pathlib.Path | None
    :param execution_id: Optional explicit Entropy execution campaign ID.
    :type execution_id: str | None
    :return: Reviewable Markdown document.
    :rtype: str
    """
    config = load_experiment_config(yaml_path)
    plan = build_plan(config)
    roles = Counter(task.workflow["role"] for task in plan.tasks if task.workflow)
    groups = {task.workflow["group_id"] for task in plan.tasks if task.workflow}
    derived_id = f"{plan.experiment_name}__cfg_{plan.config_fingerprint[:12]}"
    campaign_id = execution_id or derived_id
    lines = [
        "# Experiment YAML report", "",
        f"Source: {yaml_path.name}<br>",
        f"Experiment: {plan.experiment_name}<br>",
        f"Configuration SHA-256: {plan.config_fingerprint}<br>",
        f"Configuration-derived identifier: {derived_id}<br>",
        f"Execution campaign ID: {campaign_id}<br>",
        f"Planned tasks: **{len(plan.tasks)}**; logical groups: **{len(groups)}**; roles: `{dict(roles)}`.",
        "", "This is a static plan report; it does not prove completed training.", "",
        "## Seed streams", "",
    ]
    seed_rows = []
    for repetition in sorted({task.repetition for task in plan.tasks}):
        seeds = next(task.reproducibility for task in plan.tasks if task.repetition == repetition)
        seed_rows.append([
            repetition, seeds["common_seeds"]["split"], seeds["common_seeds"]["dataloader"],
            seeds["run_seeds"]["model_initialization"],
            seeds["derived_run_seeds"]["model_initialization"],
            seeds["run_seeds"]["training"], seeds["derived_run_seeds"]["training"],
        ])
    lines += _table(
        ["Rep", "Split", "DataLoader", "Model base", "Model derived", "Training base", "Training derived"],
        seed_rows,
    )
    lines += ["", "## Dataset and model configuration", ""]
    factory = config["task"]["parameters"]["factory_parameters"]
    for field, value in (
        ("Factory", config["task"]["parameters"]["factory"]),
        ("Project", factory["project_path"]), ("Image", factory["image_path"]),
        ("Reader", factory["reader"]), ("Dataset", factory["dataset"]["strategy"]),
        ("Cohort", factory["cohort_selection"]),
        ("Annotation settings", factory["dataset"]["parameters"]["annotation_settings"]),
        ("Subset", factory["dataset"]["parameters"]["subset"]),
        ("Split", factory["dataset"]["parameters"]["split"]),
        ("Predictive head", factory["predictive"]),
        ("Training defaults", config["task"]["parameters"]["training"]),
    ):
        lines += _table(["Field", "Configured value"], [[field, value]]) if field == "Factory" else [
            f"| {_cell(field)} | {_cell(value)} |"
        ]
    lines += ["", "## Grid dimensions", ""]
    grid_rows = [
        [dimension, item["name"], {k: v for k, v in item.items() if k != "name"}]
        for dimension, axis in config["grid"].items()
        if isinstance(axis, dict) and "values" in axis
        for item in axis["values"] if isinstance(item, dict) and "name" in item
    ]
    lines += _table(["Dimension", "Name", "Parameters"], grid_rows)
    lines += ["", "## Phase and objective catalogue", ""]
    phase_rows = []
    seen = set()
    for task in plan.tasks:
        for phase in task.parameters["training"]["phases"]:
            name = phase["phase_name"]
            if name in seen:
                continue
            seen.add(name)
            pretraining = phase.get("pretraining")
            population = pretraining.get("population", "real") if isinstance(pretraining, dict) else "real"
            phase_rows.append([name, task.workflow["role"], phase.get("epochs"), phase.get("batch_size"),
                               population, phase.get("optimizer"), phase.get("criterions")])
    lines += _table(["Phase", "Role", "Epochs", "Batch", "Population", "Optimizer", "Objective"], phase_rows)
    lines += ["", "### Selected synthetic populations", ""]
    population_rows = []
    seen_axes = set()
    for task in plan.tasks:
        if task.workflow["role"] != "pretrained":
            continue
        axis_name = task.grid_parameters["axes"]["name"]
        if axis_name in seen_axes:
            continue
        seen_axes.add(axis_name)
        pretraining = next(
            phase["pretraining"] for phase in task.parameters["training"]["phases"]
            if isinstance(phase.get("pretraining"), dict)
        )
        for name, value in sorted(pretraining["artifact"]["populations"].items()):
            population_rows.append([axis_name, name, pretraining["artifact"]["seed"], value])
    lines += _table(["Axis", "Population", "Artifact seed", "Construction"], population_rows)
    if old_plan is not None:
        lines += [""] + _old_seed_comparison(old_plan, plan.tasks)
    lines += ["", "## Every planned model task", ""]
    rows = []
    for task in plan.tasks:
        descriptor = asdict(task)
        grid = task.grid_parameters
        rows.append([
            task.task_id, task.grid_id, task.workflow["group_id"],
            grid["axes"]["name"], ",".join(phase["phase_name"] for phase in grid["schedules"]),
            task.repetition, task.workflow["role"], ",".join(task.depends_on),
            ",".join(phase["phase_name"] for phase in task.parameters["training"]["phases"]),
            task.reproducibility["derived_run_seeds"], run_identifier(campaign_id, descriptor),
        ])
    lines += _table(
        ["Task", "Grid", "Group", "Axis", "Schedule", "Rep", "Role", "Depends on",
         "Actual phases", "Derived seeds", "Model name"], rows,
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    """Write a Markdown report for one campaign YAML."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("yaml", type=Path, nargs="?", default=DEFAULT_YAML)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--old-plan", type=Path)
    parser.add_argument("--campaign-id")
    args = parser.parse_args()
    output = args.output or args.yaml.with_name("yaml_report.md")
    output.write_text(render_report(args.yaml, args.old_plan, args.campaign_id), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
