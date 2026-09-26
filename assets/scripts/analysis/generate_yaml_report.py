#!/usr/bin/env python3
"""Render an auditable Markdown report from an experiment YAML plan."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import yaml

from msi_autoencoder_wrapper.runtime import build_plan, load_experiment_config
from msi_autoencoder_wrapper.runtime.naming import campaign_identifier, run_identifier
from msi_autoencoder_wrapper.utils.logger import get_custom_logger


logger = get_custom_logger(__name__)


def _cell(value: Any) -> str:
    """Format one compact, Markdown-safe table cell."""
    if isinstance(value, (dict, list)):
        value = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return str(value).replace("|", "\\|").replace("\n", "<br>")


def _table(headers: list[str], rows: list[list[Any]]) -> list[str]:
    """Render a Markdown table with a stable column order."""
    return [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
        *("| " + " | ".join(_cell(value) for value in row) + " |" for row in rows),
    ]


def _phase_loss_key(criterions: Any) -> str:
    """Assign a content-derived short identifier to one objective graph."""
    import hashlib

    encoded = json.dumps(criterions or {}, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:10]


def _grid_signature(task: Any) -> tuple[str, str, int]:
    """Identify a comparable axis, schedule and repetition without task numbering."""
    grid = task.grid_parameters
    axis = grid.get("axes", {})
    axis_name = axis.get("name", str(axis)) if isinstance(axis, dict) else str(axis)
    schedule = grid.get("schedules", [])
    phase_names = ",".join(
        phase.get("phase_name", "?") for phase in schedule if isinstance(phase, dict)
    )
    return axis_name, phase_names, task.repetition


def _old_seed_comparison(path: Path, tasks: tuple[Any, ...]) -> list[str]:
    """Compare every logical run against a prior resolved plan's seed streams.

    :param path: Prior ``resolved-experiment.yaml``.
    :type path: pathlib.Path
    :param tasks: Current statically planned tasks.
    :type tasks: tuple[typing.Any, ...]
    :return: Markdown section reporting exact seed parity and missing signatures.
    :rtype: list[str]
    """
    # Prior resolved plans can exceed 100 MB because they repeat resolved dataset
    # metadata for every task. Parse one descriptor at a time to bound memory.
    task_paths = sorted((path.parent / "tasks").glob("task_*.yaml"))
    if not task_paths:
        raise ValueError("Comparison plan requires adjacent tasks/task_*.yaml files.")

    previous: dict[tuple[str, str, int], dict[str, Any]] = {}
    loader = getattr(yaml, "CSafeLoader", yaml.SafeLoader)
    for task_path in task_paths:
        with task_path.open(encoding="utf-8") as stream:
            task = yaml.load(stream, Loader=loader)
        if not isinstance(task, dict):
            continue
        grid = task.get("grid_parameters", {})
        axis = grid.get("axes", {})
        axis_name = axis.get("name", str(axis)) if isinstance(axis, dict) else str(axis)
        schedule = grid.get("schedules", [])
        phase_names = ",".join(
            phase.get("phase_name", "?") for phase in schedule if isinstance(phase, dict)
        )
        previous[(axis_name, phase_names, task.get("repetition"))] = task.get("reproducibility", {})

    current = {}
    for task in tasks:
        current.setdefault(_grid_signature(task), task)
    rows = []
    for signature, task in sorted(current.items()):
        old = previous.get(signature)
        if old is None:
            rows.append([*signature, "missing", "missing", "missing", "missing"])
            continue
        old_repro = old
        new_repro = task.reproducibility
        rows.append(
            [
                *signature,
                "same" if old_repro.get("common_seeds", {}).get("split") == new_repro["common_seeds"]["split"] else "DIFFERENT",
                "same" if old_repro.get("common_seeds", {}).get("dataloader") == new_repro["common_seeds"]["dataloader"] else "DIFFERENT",
                "same" if old_repro.get("derived_run_seeds", {}).get("model_initialization") == new_repro["derived_run_seeds"]["model_initialization"] else "DIFFERENT",
                "same" if old_repro.get("derived_run_seeds", {}).get("training") == new_repro["derived_run_seeds"]["training"] else "DIFFERENT",
            ]
        )
    counts = Counter(value for row in rows for value in row[3:])
    return [
        "## Comparison with prior plan",
        "",
        f"Source: `{path.resolve()}`. Compared {len(rows)} logical axis/schedule/repetition runs; seed cells: {dict(counts)}.",
        "Equal seed values establish matching configured random streams. They do not prove identical input data, code version, or stochastic execution across different task layouts.",
        "",
        *_table(["Axis", "Schedule phases", "Rep", "Split", "DataLoader", "Model init", "Training"], rows),
        "",
    ]


def render_report(config_path: Path, *, compare_plan: Path | None = None) -> str:
    """Build and render one static experiment plan.

    :param config_path: Experiment source YAML.
    :type config_path: pathlib.Path
    :param compare_plan: Optional prior resolved plan for seed comparison.
    :type compare_plan: pathlib.Path | None
    :return: Markdown report.
    :rtype: str
    """
    config = load_experiment_config(config_path)
    plan = build_plan(config)
    tasks = plan.tasks
    campaign_id = campaign_identifier(plan.experiment_name, plan.config_fingerprint)
    roles = Counter((task.workflow or {}).get("role", "single") for task in tasks)
    groups: dict[str, list[Any]] = defaultdict(list)
    for task in tasks:
        group = (task.workflow or {}).get("group_id", f"{task.grid_id}__rep_{task.repetition:02d}")
        groups[group].append(task)

    lines = [
        "# Experiment YAML report",
        "",
        f"Source: `{config_path.resolve()}`  ",
        f"Experiment: `{plan.experiment_name}`  ",
        f"Configuration SHA-256: `{plan.config_fingerprint}`  ",
        f"Campaign identifier: `{campaign_id}`  ",
        f"Planned tasks: **{len(tasks)}**; logical groups: **{len(groups)}**; roles: `{dict(roles)}`.",
        "",
        "This is a static plan report. It does not assert that data, training, or model materialization completed.",
        "",
        "## Seed streams",
        "",
    ]
    seed_rows = []
    for repetition in sorted({task.repetition for task in tasks}):
        sample = next(task for task in tasks if task.repetition == repetition)
        repro = sample.reproducibility
        seed_rows.append([
            repetition,
            repro["common_seeds"]["split"],
            repro["common_seeds"]["dataloader"],
            repro["run_seeds"]["model_initialization"],
            repro["derived_run_seeds"]["model_initialization"],
            repro["run_seeds"]["training"],
            repro["derived_run_seeds"]["training"],
        ])
    lines += _table(
        ["Rep", "Split", "DataLoader", "Model base", "Model derived", "Training base", "Training derived"],
        seed_rows,
    )
    lines += [
        "",
        "Derived seed = first 32 bits of SHA-256(`base:purpose:repetition`) masked to 31 bits. The same repetition has the same derived model and training seeds in every grid cell and role. Different repetitions have different derived seeds.",
        "The model seed is set before factory construction; the training seed is set before fit. The DataLoader seed is set per phase. The synthetic artifact has its own seed and reuses static rows across repetitions.",
        "",
        "## Dataset and model configuration",
        "",
    ]
    factory = config["task"]["parameters"].get("factory_parameters", {})
    dataset = factory.get("dataset", {})
    dataset_parameters = dataset.get("parameters", {}) if isinstance(dataset, dict) else {}
    lines += _table(
        ["Field", "Configured value"],
        [
            ["Factory", config["task"]["parameters"].get("factory")],
            ["Project", factory.get("project_path")],
            ["Image", factory.get("image_path")],
            ["Reader", factory.get("reader")],
            ["Dataset", dataset.get("strategy") if isinstance(dataset, dict) else dataset],
            ["Cohort", factory.get("cohort_selection")],
            ["Annotation settings", dataset_parameters.get("annotation_settings")],
            ["Subset", dataset_parameters.get("subset")],
            ["Split", dataset_parameters.get("split")],
            ["Split reference binning", factory.get("split_reference_binning")],
            ["Normalization", dataset_parameters.get("normalization")],
            ["Targets", dataset_parameters.get("target_specs")],
            ["Predictive head", factory.get("predictive")],
            ["Training defaults", {
                key: value for key, value in config["task"]["parameters"].get("training", {}).items()
                if key != "phases"
            }],
        ],
    )
    lines += ["", "## Grid dimensions", ""]
    lines += _table(
        ["Dimension", "Name", "Parameters"],
        [
            [dimension, value.get("name"), {key: item for key, item in value.items() if key != "name"}]
            for dimension in ("axes", "architectures")
            for value in config["grid"][dimension]["values"]
        ],
    )
    lines += ["", "## Phase and objective catalogue", ""]
    phase_rows: dict[tuple[str, str], list[Any]] = {}
    objectives: dict[str, Any] = {}
    populations: dict[tuple[str, str], tuple[Any, Any]] = {}
    for task in tasks:
        for phase in task.parameters.get("training", {}).get("phases", []):
            loss_key = _phase_loss_key(phase.get("criterions"))
            objectives[loss_key] = phase.get("criterions", {})
            pretrain = phase.get("pretraining", {})
            population = pretrain.get("population", "real")
            if pretrain.get("kind") == "precomputed_synthetic":
                artifact = pretrain.get("artifact", {})
                axis, _, _ = _grid_signature(task)
                populations[(axis, population)] = (
                    artifact.get("seed"), artifact.get("populations", {}).get(population)
                )
            row_key = (phase.get("phase_name", "?"), str(population))
            phase_rows[row_key] = [
                phase.get("phase_name"),
                (task.workflow or {}).get("role", "single"),
                phase.get("epochs", 10),
                phase.get("batch_size"),
                population,
                phase.get("pretraining", {}).get("validation_samples", ""),
                phase.get("freeze", []),
                phase.get("supervision_sampling", ""),
                phase.get("optimizer", "default"),
                loss_key,
            ]
    lines += _table(
        ["Phase", "Role", "Epochs", "Batch", "Population", "Synthetic val rows", "Freeze", "Supervision", "Optimizer", "Objective ID"],
        list(phase_rows.values()),
    )
    lines += ["", "### Complete objective definitions", ""]
    for key, objective in sorted(objectives.items()):
        lines += [f"`{key}`", "", "```json", json.dumps(objective, indent=2, sort_keys=True), "```", ""]
    lines += ["### Selected synthetic populations", ""]
    lines += _table(
        ["Axis", "Population", "Artifact seed", "Construction"],
        [[axis, name, seed, value] for (axis, name), (seed, value) in sorted(populations.items())],
    )
    lines += ["", "## Every planned model task", ""]
    task_rows = []
    for task in tasks:
        axis, schedule, _ = _grid_signature(task)
        workflow = task.workflow or {}
        phases = task.parameters.get("training", {}).get("phases", [])
        task_rows.append([
            task.task_id, task.grid_id, workflow.get("group_id", ""), axis,
            schedule, task.repetition, workflow.get("role", "single"),
            ", ".join(task.depends_on) or "—",
            ", ".join(phase.get("phase_name", "?") for phase in phases),
            task.reproducibility["derived_run_seeds"]["model_initialization"],
            task.reproducibility["derived_run_seeds"]["training"],
            run_identifier(campaign_id, {
                "grid_id": task.grid_id,
                "repetition": task.repetition,
                "workflow": task.workflow,
            }),
        ])
    lines += _table(
        ["Task", "Grid", "Group", "Axis", "Schedule", "Rep", "Role", "Depends on", "Actual phases", "Model seed", "Training seed", "Model name"],
        task_rows,
    )
    lines += [""]
    if compare_plan is not None:
        lines += _old_seed_comparison(compare_plan, tasks)
    return "\n".join(lines)


def main() -> int:
    """Write ``yaml_report.md`` adjacent to the source YAML by default."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path, nargs="?", default=Path(__file__).with_name("pretraining_experiment.yaml"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--compare-plan", type=Path)
    args = parser.parse_args()
    output = args.output or args.config.with_name("yaml_report.md")
    report = render_report(args.config, compare_plan=args.compare_plan)
    output.write_text(report, encoding="utf-8")
    logger.info("Wrote plan report to %s.", output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
