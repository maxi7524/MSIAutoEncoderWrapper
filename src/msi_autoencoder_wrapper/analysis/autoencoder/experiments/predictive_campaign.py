"""Audit and relocate predictive campaigns without editing training artifacts."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from ....utils.logger import get_custom_logger
from .entropy_status_reader import _resolve_artifact_directory

logger = get_custom_logger(__name__)

#: Logged training scalars that record bookkeeping rather than an objective component.
BOOKKEEPING_METRICS = ("epoch", "duration", "best_loss")


def fingerprint(value: Any) -> str:
    """Hash a JSON-compatible semantic value deterministically.

    :param value: Configuration or metadata.
    :return: SHA-256 hexadecimal digest.
    :rtype: str
    """
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def objective_identity(objective: dict) -> dict:
    """Identify the trained head from the loss, never from all constructed heads.

    :param objective: Resolved criterion mapping for one training condition.
    :return: Semantic head identity and exact objective hash.
    :rtype: dict
    :raises ValueError: If the condition does not activate exactly one head loss.
    """
    active = [(head, name, spec) for head, losses in objective.get("heads", {}).items()
              for name, spec in losses.items() if float(spec.get("weight", 1.0)) != 0]
    if len(active) != 1:
        raise ValueError("This head-selection analysis requires exactly one active head loss.")
    head, name, spec = active[0]
    family = spec["target"]
    params = spec.get("params", {})
    weight_mode = params.get("positive_weight_mode", "none")
    label = f"{name} ({family})"
    return {"condition": fingerprint(objective)[:12], "label": label, "head": head,
            "family": family, "weight_mode": weight_mode,
            "head_weight": spec.get("weight", 1.0),
            "negative_weight": params.get("negative_weight", 0.0),
            "evidence": json.dumps(params.get("evidence", {}), sort_keys=True),
            "objective_json": json.dumps(objective, sort_keys=True)}


def configured_grid(config_path: Path | str) -> pd.DataFrame:
    """Audit declared cells, including exact duplicate objective/architecture pairs.

    :param config_path: Campaign YAML.
    :return: One row per declared cell; duplicates point to the first cell index.
    :rtype: pandas.DataFrame
    """
    config = yaml.safe_load(Path(config_path).read_text())
    rows, first = [], {}
    for architecture in config["grid"]["architectures"]["values"]:
        for objective in config["grid"]["objectives"]["values"]:
            identity = objective_identity(objective)
            key = fingerprint([architecture, objective])
            position = len(rows)
            rows.append({"grid_position": position, "architecture": architecture["name"],
                         **identity, "duplicate_of": first.get(key),
                         "repetitions": config["runs"]["repetitions"]})
            first.setdefault(key, position)
    return pd.DataFrame(rows)


def load_settings(path: Path | str) -> dict:
    """Resolve all analysis paths relative to the settings file's repository.

    :param path: Analysis settings YAML.
    :return: Settings with absolute paths and repository root.
    :rtype: dict
    :raises ValueError: If no repository root or invalid numeric settings exist.
    """
    path = Path(path).resolve()
    root = next((p for p in path.parents if (p / "pyproject.toml").is_file()), None)
    if root is None:
        raise ValueError("Analysis settings must live inside the repository.")
    settings = yaml.safe_load(path.read_text())
    for key in ("workspace", "model_store", "experiment_config", "cache_directory"):
        settings[key] = str((root / settings[key]).resolve())
    for source in settings["sources"]:
        if source.get("status_directory"):
            source["status_directory"] = str((root / source["status_directory"]).resolve())
    settings.setdefault("case_count", 6)
    settings.setdefault("shortlist", [])
    if settings["batch_size"] < 1 or settings["geometry_sample_size"] < 3:
        raise ValueError("batch_size >=1 and geometry_sample_size >=3 are required.")
    if not 0 < settings["pixel_fraction"] <= 1:
        raise ValueError("pixel_fraction must be in (0, 1].")
    if settings["case_count"] < 1 or settings["case_count"] > settings["geometry_sample_size"]:
        raise ValueError("case_count must be between one and geometry_sample_size.")
    if not isinstance(settings["shortlist"], list):
        raise ValueError("shortlist must be a list of condition labels.")
    settings["repository_root"] = str(root)
    return settings


def _discover_status(settings: dict, source: dict) -> Path | None:
    """Resolve one explicit status directory or an unambiguous experiment match."""
    if source.get("status_directory"):
        path = Path(source["status_directory"])
        return path if path.is_dir() else None
    workspace = Path(settings["workspace"])
    candidates = list((workspace / "configs" / "execution").glob("*/status"))
    candidates += list((workspace / "configs" / "entropy-runs").glob("*/plan/status"))
    matched = []
    for path in candidates:
        manifests = sorted(p for p in path.glob("task_*.yaml") if not p.name.endswith("-progress.yaml"))
        if not manifests:
            continue
        payload = yaml.safe_load(manifests[0].read_text()) or {}
        records = payload.get("records", {})
        names = [str(record.get("task", {}).get("runtime", {}).get("model_name", "")) for record in records.values()]
        experiment = source["experiment_name"]
        if path.parent.name.startswith(experiment) or any(name.startswith(experiment + "__cfg_") for name in names):
            matched.append(path)
    if len(matched) > 1:
        raise ValueError(f"Multiple campaigns match {source['experiment_name']}: {matched}. Set status_directory.")
    return matched[0] if matched else None


def relocated_data(config: dict, workspace: Path | str, path_remap: dict | None = None) -> dict:
    """Relocate saved workspace paths in a copy of the configuration.

    :param config: Saved consolidated configuration.
    :param workspace: Local workspace containing the downloaded datasets.
    :param path_remap: Optional explicit old-prefix to local-prefix mapping.
    :return: Relocated deep copy; source files remain unchanged.
    :rtype: dict
    """
    local = str(Path(workspace).resolve())
    remap = dict(path_remap or {})
    reader = config["data"]["context"]["components"].get("reader", {})
    old_path = reader.get("parameters", {}).get("file_path", "")
    if "/datasets/" in old_path:
        remap.setdefault(old_path.split("/datasets/", 1)[0], local)

    def visit(value):
        if isinstance(value, dict):
            return {key: visit(item) for key, item in value.items()}
        if isinstance(value, list):
            return [visit(item) for item in value]
        if isinstance(value, str):
            for old, new in sorted(remap.items(), key=lambda item: -len(item[0])):
                if value == old or value.startswith(old.rstrip("/") + "/"):
                    return str(new).rstrip("/") + value[len(old.rstrip("/")):]
        return value

    result = deepcopy(config)
    result["data"] = visit(result["data"])
    return result


def _data_contract(config: dict, settings: dict) -> dict:
    """Retain complete dataset selection, assignments and preprocessing semantics."""
    data = relocated_data(config, settings["workspace"], settings.get("path_remap"))["data"]
    # REMARK: PixelDataset.__init__ uses dict(chemistry or {}). Older artifacts
    # omit this field; newer exports serialize {}. These are identical defaults,
    # whereas a nonempty frozen chemical snapshot must remain in the contract.
    parameters = data.get("dataset", {}).get("parameters", {})
    if not parameters.get("chemistry"):
        parameters.pop("chemistry", None)
    assignments = data.get("dataset", {}).get("parameters", {}).get("split", {}).get("assignments")
    if not assignments or not all(name in assignments for name in ("train", "validation", "test")):
        raise ValueError("Saved train/validation/test assignments are required; never regenerate a split.")
    memberships = [set(assignments[name]) for name in ("train", "validation", "test")]
    if any(len(memberships[i]) != len(assignments[name]) for i, name in enumerate(("train", "validation", "test"))):
        raise ValueError("Duplicate indices in saved split assignments.")
    if any(memberships[i] & memberships[j] for i in range(3) for j in range(i)):
        raise ValueError("Saved split assignments overlap.")
    return data


def inventory(settings: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Audit both local and entropy manifests, saved configs and checkpoint coverage.

    :param settings: Resolved settings from :func:`load_settings`.
    :return: Model inventory and source coverage tables. Missing sources are reported.
    :rtype: tuple[pandas.DataFrame, pandas.DataFrame]
    :raises ValueError: If manifests conflict, split contracts are invalid or inference is ambiguous.
    """
    rows, sources = [], []
    for source in settings["sources"]:
        if not source.get("enabled", True):
            continue
        status = _discover_status(settings, source)
        sources.append({"source": source["name"], "required": source.get("required", False),
                        "status_directory": str(status or ""), "available": status is not None})
        if status is None:
            logger.warning("Campaign source is not downloaded: %s", source["name"])
            continue
        campaign_id = status.parent.parent.name if status.parent.name == "plan" else status.parent.name
        seen = set()
        for manifest in sorted(status.glob("task_*.yaml")):
            if manifest.name.endswith("-progress.yaml"):
                continue
            for task_id, record in (yaml.safe_load(manifest.read_text()) or {}).get("records", {}).items():
                if task_id in seen:
                    raise ValueError(f"Duplicate task record in {status}: {task_id}")
                seen.add(task_id)
                task = record["task"]
                objective = task.get("grid_parameters", {}).get("objectives")
                if objective is None:
                    objective = task["parameters"]["training"]["phases"][-1]["criterions"]
                identity = objective_identity(objective)
                artifact = _resolve_artifact_directory(Path(settings["model_store"]), task.get("runtime", {}).get("model_name"), campaign_id, task_id)
                # Local manifests can use a non-namespaced model_path.
                if artifact is None:
                    saved_path = (record.get("result") or {}).get("model_path")
                    if saved_path and (Path(saved_path) / "config" / "config.json").is_file():
                        artifact = Path(saved_path) / "config"
                config = json.loads((artifact / "config.json").read_text()) if artifact and (artifact / "config.json").is_file() else None
                reproducibility = task.get("reproducibility", {})
                seeds = reproducibility.get("derived_run_seeds", {})
                row = {"source": source["name"], "role": source.get("role", "candidate"),
                       "model_id": f"{source['name']}/{task_id}", "task_id": task_id,
                       "grid_id": task.get("grid_id", ""), "repetition": task.get("repetition"),
                       "status": record.get("status", "unknown"), **identity,
                       "initialization_seed": seeds.get("model_initialization"),
                       "training_seed": seeds.get("training"),
                       "artifact": str(artifact.parent) if artifact else "",
                       "weights_available": bool(artifact and (artifact / "weights.pt").is_file()),
                       "history_available": bool(artifact and (artifact / "history.json").is_file()),
                       "manifest": str(manifest), "data_contract": "", "backbone_contract": "",
                       "training_contract": "", "config_matches_manifest": False}
                if config:
                    row["data_contract"] = fingerprint(_data_contract(config, settings))
                    components = config["model"].get("components", {})
                    row["backbone_contract"] = fingerprint({key: components.get(key) for key in ("encoder", "decoder")})
                    training = deepcopy(config["training"]["parameters"])
                    saved_objective = training["phases"][-1]["criterions"]
                    row["config_matches_manifest"] = fingerprint(saved_objective) == fingerprint(objective)
                    for key in ("runtime", "seed", "continuation"):
                        training.pop(key, None)
                    for phase in training["phases"]:
                        phase.get("criterions", {}).pop("heads", None)
                    row["training_contract"] = fingerprint(training)
                row["ready"] = bool(row["status"] == "completed" and row["weights_available"] and config and row["config_matches_manifest"])
                rows.append(row)
    frame = pd.DataFrame(rows)
    if not frame.empty:
        # Same condition and paired seeds is one experimental unit, even when
        # the YAML schedules it twice. Keep duplicate diagnostics, never inflate n.
        frame["duplicate_seed_condition"] = frame.duplicated(
            ["source", "condition", "backbone_contract", "repetition", "initialization_seed", "training_seed"], keep=False)
    logger.info("Audited %s model manifests across %s sources.", len(frame), len(sources))
    return frame, pd.DataFrame(sources)


def training_history(models: pd.DataFrame) -> pd.DataFrame:
    """Preserve epoch trajectories and final evaluations as distinct records.

    :param models: Inventory table.
    :return: Long-form scalar history with epoch/evaluation record type.
    :rtype: pandas.DataFrame
    """
    rows = []
    for model in models.to_dict("records"):
        path = Path(model["artifact"]) / "config" / "history.json"
        if not model["artifact"] or not path.is_file():
            continue
        for entry in json.loads(path.read_text()):
            metrics = entry.get("metrics", {})
            for metric, value in metrics.items():
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    rows.append({"model_id": model["model_id"], "label": model["label"],
                                 "phase": entry.get("phase"), "epoch": metrics.get("epoch"),
                                 "record_type": "epoch" if metrics.get("epoch") is not None else "evaluation",
                                 "split": entry.get("split", ""), "metric": metric, "value": value,
                                 "is_best": metrics.get("is_best", False)})
    return pd.DataFrame(rows)


def history_components(history: pd.DataFrame) -> pd.DataFrame:
    """Separate logged training scalars into objective components and their split.

    The trainer writes one flat scalar name per logged quantity, with the evaluated
    split encoded as a ``validation_`` prefix and the head losses as
    ``<head>__<loss name>``. Reading such a name directly in a notebook invites two
    mistakes: plotting a validation curve as if it were a training curve, and
    overlaying head losses from different objectives on one axis even though a
    cross-entropy, a masked binary cross-entropy and a variational PU bound are not
    on a common scale. This function makes both distinctions explicit.

    Only the reconstruction component is marked comparable across conditions: every
    run optimizes the same Masserstein cost with the same parameters. The total loss
    contains the head term and therefore inherits its scale, and each head loss is
    defined only within its own objective.

    :param history: Long-form output of :func:`training_history`.
    :type history: pandas.DataFrame
    :return: The same records with ``component``, ``component_name``,
        ``history_split`` and ``comparable`` columns added.
    :rtype: pandas.DataFrame
    """
    if history.empty:
        return history.assign(component=[], component_name=[], history_split=[], comparable=[])
    result = history.copy()
    ## The evaluated split is a prefix on the scalar name, not the record's split field
    validation = result.metric.str.startswith("validation_")
    stem = result.metric.mask(validation, result.metric.str.removeprefix("validation_"))
    result["history_split"] = np.where(validation, "validation", "train")
    result["component"] = np.select(
        [stem.isin(BOOKKEEPING_METRICS), stem == "total_loss", stem.str.contains("__")],
        ["bookkeeping", "total", "head"], default="reconstruction")
    result["component_name"] = np.where(result.component == "head", stem.str.split("__").str[-1], stem)
    result["comparable"] = result.component == "reconstruction"
    return result


def coverage_table(grid: pd.DataFrame, models: pd.DataFrame, *, source: str = "predictive_initial") -> pd.DataFrame:
    """Compare declared conditions with manifest and completed-seed coverage.

    :param grid: Declared grid from :func:`configured_grid`.
    :param models: Current inventory, possibly empty.
    :param source: Source holding the declared campaign.
    :return: One row per condition, including conditions with no downloaded task.
    :rtype: pandas.DataFrame
    """
    rows = []
    for condition, declared in grid.groupby("condition"):
        observed = models[(models.source == source) & (models.condition == condition)] if not models.empty else pd.DataFrame()
        completed = observed[observed.ready] if not observed.empty else pd.DataFrame()
        expected = int(declared.repetitions.iloc[0])
        seeds = completed[["repetition", "initialization_seed", "training_seed"]].dropna().drop_duplicates() if not completed.empty else pd.DataFrame()
        rows.append({"condition": condition, "label": declared.label.iloc[0], "declared_cells": len(declared),
                     "expected_tasks": int(declared.repetitions.sum()), "expected_seeds": expected,
                     "downloaded_tasks": len(observed), "ready_tasks": len(completed),
                     "ready_seeds": len(seeds), "complete_seed_coverage": len(seeds) == expected})
    return pd.DataFrame(rows)
