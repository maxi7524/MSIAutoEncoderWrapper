#!/usr/bin/env python3
"""Audit each recovered legacy task, log, checkpoint and model bundle."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import numpy._core.multiarray as numpy_multiarray
import torch
import yaml

from msi_autoencoder_wrapper.models.model_loader import ModelLoader
from msi_autoencoder_wrapper.runtime import build_plan, load_experiment_config
from msi_autoencoder_wrapper.runtime.output import task_fingerprint
from msi_autoencoder_wrapper.utils.logger import get_custom_logger


logger = get_custom_logger(__name__)
LOG_NAME = re.compile(r"task_(\d+)_(\d+)\.log$")
PHASE_START = re.compile(r"Initiating sequential training loop phase: (\S+)")
EPOCH_SUMMARY = re.compile(r"=== Epoch Summary \[([^]]+)\]\s+(\d+)/(\d+)")
PHASE_END = re.compile(r"Restored the best checkpoint after phase '([^']+)'\.")
SNAPSHOT = re.compile(r"Stored completed phase '([^']+)' as snapshot '([^']+)'\.")
FINAL_SAVE = re.compile(r"Active model '([^']+)' saved under context")
ALLOWED_GLOBALS = frozenset(
    {"numpy.dtype", "numpy.ndarray", "numpy._core.multiarray._reconstruct"}
)


def _yaml(path: Path) -> dict[str, Any]:
    """Read one YAML mapping with bounded per-file memory."""
    with path.open(encoding="utf-8") as stream:
        value = yaml.load(stream, Loader=getattr(yaml, "CSafeLoader", yaml.SafeLoader))
    if not isinstance(value, dict):
        raise ValueError(f"Expected mapping: {path}")
    return value


def _log(path: Path) -> dict[str, Any]:
    """Extract concrete phase evidence from one scheduler log file."""
    content = path.read_text(encoding="utf-8", errors="replace")
    epochs: dict[str, set[int]] = defaultdict(set)
    for phase, number, _total in EPOCH_SUMMARY.findall(content):
        epochs[phase].add(int(number))
    return {
        "name": path.name,
        "phases": PHASE_START.findall(content),
        "epochs": epochs,
        "ended": PHASE_END.findall(content),
        "snapshots": SNAPSHOT.findall(content),
        "saved": FINAL_SAVE.findall(content),
        "error": "| ERROR " in content or "Traceback (most recent call last)" in content,
    }


def _state_valid(state: Any) -> bool:
    """Require nonempty finite tensors with string state-dictionary keys."""
    return bool(
        isinstance(state, dict)
        and state
        and all(
            isinstance(name, str)
            and isinstance(tensor, torch.Tensor)
            and bool(torch.isfinite(tensor).all())
            for name, tensor in state.items()
        )
    )


def _checkpoint(path: Path, fingerprint: str) -> dict[str, Any]:
    """Validate legacy checkpoint data using restricted PyTorch deserialization."""
    result = {"exists": path.is_file(), "valid": False, "snapshot": False, "reason": "missing"}
    if not result["exists"]:
        return result
    try:
        unexpected = set(torch.serialization.get_unsafe_globals_in_checkpoint(path)) - ALLOWED_GLOBALS
        if unexpected:
            raise ValueError(f"unexpected pickle globals: {sorted(unexpected)}")
        allowed = [np.dtype, np.ndarray, numpy_multiarray._reconstruct, np.dtypes.UInt32DType]
        with torch.serialization.safe_globals(allowed):
            payload = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(payload, dict) or payload.get("task_fingerprint") != fingerprint:
            raise ValueError("checkpoint fingerprint mismatch")
        if not _state_valid(payload.get("model_state")):
            raise ValueError("invalid checkpoint model_state")
        snapshot = payload.get("phase_snapshots", {}).get("synthetic_pretrained")
        result["snapshot"] = _state_valid(snapshot)
        if result["snapshot"]:
            model_state = payload["model_state"]
            result["snapshot"] = snapshot.keys() == model_state.keys() and all(
                snapshot[name].shape == model_state[name].shape
                and snapshot[name].dtype == model_state[name].dtype
                for name in snapshot
            )
        result.update(valid=True, reason="valid", phase_index=payload.get("phase_index"), epoch=payload.get("epoch"))
    except (OSError, RuntimeError, ValueError, TypeError) as error:
        result["reason"] = str(error)
    return result


def _model(path: Path) -> dict[str, Any]:
    """Validate final persisted weights with the public model loader."""
    files = [path / "config" / name for name in ("config.json", "history.json", "weights.pt")]
    result: dict[str, Any] = {"valid": False, "history": [], "reason": "missing model bundle"}
    if not all(item.is_file() for item in files):
        return result
    try:
        config = json.loads(files[0].read_text(encoding="utf-8"))
        history = json.loads(files[1].read_text(encoding="utf-8"))
        if not isinstance(config, dict) or not isinstance(history, list):
            raise ValueError("invalid model config or history")
        model, _, resolved = ModelLoader.load_artifact(path)
        if resolved != path.resolve() or not _state_valid(model.state_dict()):
            raise ValueError("loaded model has invalid state")
        result.update(valid=True, history=history, reason="valid")
    except (OSError, RuntimeError, ValueError, TypeError, KeyError) as error:
        result["reason"] = str(error)
    return result


def _signature(task: dict[str, Any]) -> tuple[str, str, int]:
    """Identify a trial independently of old and new task identifiers."""
    grid = task["grid_parameters"]
    return (
        grid["axes"]["name"],
        ",".join(phase["phase_name"] for phase in grid["schedules"]),
        int(task["repetition"]),
    )


def _scientific_phases(task: dict[str, Any]) -> list[dict[str, Any]]:
    """Ignore only workflow bookkeeping when comparing old and new phases."""
    phases = []
    for phase in task["grid_parameters"]["schedules"]:
        phases.append({
            key: value for key, value in phase.items()
            if key not in {"workflow_role", "save_model_state_as", "restore_model_state_from"}
        })
    return phases


def audit(root: Path, yaml_path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Evaluate each old task using its own descriptor, status, logs and artifacts.

    :param root: Recovery directory containing ``entropy-run`` and ``models``.
    :type root: pathlib.Path
    :param yaml_path: Current experiment YAML for exact seed/config parity.
    :type yaml_path: pathlib.Path
    :return: Per-task and per-log evidence records.
    :rtype: tuple[list[dict[str, typing.Any]], list[dict[str, typing.Any]]]
    """
    plan = root / "entropy-run" / "plan"
    campaign_id = root.name.removesuffix("-recovery")
    current = build_plan(load_experiment_config(yaml_path))
    new_trials = {}
    for task in current.tasks:
        new_trials.setdefault(_signature({"grid_parameters": task.grid_parameters, "repetition": task.repetition}), task)

    logs: dict[str, list[dict[str, Any]]] = defaultdict(list)
    log_rows = []
    for path in sorted((root / "entropy-run" / "logs").glob("task_*_*.log")):
        match = LOG_NAME.fullmatch(path.name)
        if match is None:
            continue
        task_id = f"task_{int(match.group(2)):06d}"
        observation = _log(path)
        observation["task_id"] = task_id
        logs[task_id].append(observation)
        log_rows.append(observation)

    rows = []
    paths = sorted((plan / "tasks").glob("task_*.yaml"))
    for index, path in enumerate(paths, start=1):
        task = _yaml(path)
        task_id = task["task_id"]
        fingerprint = task_fingerprint(task)
        status_path = plan / "status" / f"{task_id}.yaml"
        status = _yaml(status_path)["records"][task_id] if status_path.is_file() else {}
        phases = task["parameters"]["training"]["phases"]
        expected = {phase["phase_name"]: int(phase.get("epochs", 10)) for phase in phases}
        observations = logs[task_id]
        started = {phase for log in observations for phase in log["phases"]}
        ended = {phase for log in observations for phase in log["ended"]}
        epochs: dict[str, set[int]] = defaultdict(set)
        for log in observations:
            for phase, numbers in log["epochs"].items():
                epochs[phase].update(numbers)
        final_name = f"{campaign_id}__{task_id}"
        final_logged = any(final_name in log["saved"] for log in observations)
        snapshot_logged = any(
            name == "synthetic_pretrained"
            for log in observations for _phase, name in log["snapshots"]
        )
        model_path = root / "models" / "kidney" / final_name
        model = _model(model_path)
        checkpoint = _checkpoint(plan / "checkpoints" / f"{task_id}.pt", fingerprint)
        history_epochs: dict[str, set[int]] = defaultdict(set)
        for entry in model["history"]:
            if isinstance(entry, dict):
                number = entry.get("metrics", {}).get("epoch")
                if isinstance(number, int):
                    history_epochs[entry.get("phase", "?")].add(number)
        epoch_log_ok = all(len(epochs[name]) >= count for name, count in expected.items())
        history_ok = all(len(history_epochs[name]) >= count for name, count in expected.items())
        log_ok = bool(set(expected) <= started and epoch_log_ok and list(expected)[-1] in ended and final_logged)
        recorded_path = status.get("result", {}).get("model_path") if isinstance(status.get("result"), dict) else None
        status_ok = bool(
            status.get("status") == "completed"
            and status.get("task_fingerprint") == fingerprint
            and isinstance(recorded_path, str)
            and Path(recorded_path).name == final_name
        )
        synthetic = any(phase.get("pretraining") for phase in phases)
        synthetic_epoch_ok = all(
            len(epochs[phase["phase_name"]]) >= int(phase.get("epochs", 10))
            for phase in phases if phase.get("pretraining")
        )
        new = new_trials.get(_signature(task))
        seeds_match = bool(new and new.reproducibility == task["reproducibility"])
        phases_match = bool(new and _scientific_phases(task) == _scientific_phases({"grid_parameters": new.grid_parameters}))
        rows.append({
            "task_id": task_id, "signature": _signature(task), "status": status.get("status", "missing"),
            "logs": observations, "expected": expected, "epochs": epochs,
            "model": model, "checkpoint": checkpoint, "status_ok": status_ok,
            "history_ok": history_ok, "log_ok": log_ok, "synthetic": synthetic,
            "snapshot_ok": bool(synthetic and checkpoint["valid"] and checkpoint["snapshot"] and snapshot_logged and synthetic_epoch_ok),
            "final_ok": bool(status_ok and model["valid"] and history_ok and log_ok),
            "seeds_match": seeds_match, "phases_match": phases_match,
        })
        if index % 10 == 0:
            logger.info("Audited %s/%s recovered tasks.", index, len(paths))
    return rows, log_rows


def _table(headers: list[str], rows: list[list[Any]]) -> list[str]:
    """Render escaped Markdown table rows."""
    def cell(value: Any) -> str:
        return str(value).replace("|", "\\|").replace("\n", "<br>")

    return [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
        *("| " + " | ".join(cell(item) for item in row) + " |" for row in rows),
    ]


def render(rows: list[dict[str, Any]], logs: list[dict[str, Any]]) -> str:
    """Create a report with an individual verdict for every task and log."""
    baseline = [row for row in rows if not row["synthetic"]]
    synthetic = [row for row in rows if row["synthetic"]]
    lines = [
        "# Recovered campaign audit",
        "",
        f"Inspected **{len(rows)}** task descriptors and **{len(logs)}** individual scheduler logs. Statuses: `{dict(Counter(row['status'] for row in rows))}`.",
        f"Complete final model bundles: **{sum(row['final_ok'] for row in rows)}**; valid pretraining snapshots: **{sum(row['snapshot_ok'] for row in synthetic)}**; valid real-only baselines: **{sum(row['final_ok'] for row in baseline)}**.",
        "",
        "The legacy synthetic task saved one final model after its unfrozen phase. Its checkpoint may retain `synthetic_pretrained` as a separate tensor state. A snapshot is not a standalone model bundle. The frozen-head model was overwritten during the later unfrozen phase.",
        "Equal configured seeds do not establish identical stochastic trajectories after splitting one legacy task into separate new tasks.",
        "",
        "## Every task",
        "",
    ]
    task_rows = []
    for row in rows:
        axis, schedule, repetition = row["signature"]
        epoch_count = "; ".join(
            f"{phase} {len(row['epochs'][phase])}/{count}"
            for phase, count in row["expected"].items() if count
        )
        task_rows.append([
            row["task_id"], axis, schedule, repetition, row["status"],
            ", ".join(log["name"] for log in row["logs"]) or "missing",
            epoch_count, "yes" if row["model"]["valid"] else row["model"]["reason"],
            "yes" if row["checkpoint"]["valid"] else row["checkpoint"]["reason"],
            "yes" if row["snapshot_ok"] else "no",
            "yes" if row["final_ok"] else "no",
            "yes" if row["seeds_match"] else "no",
            "yes" if row["phases_match"] else "no",
        ])
    lines += _table(
        ["Task", "Axis", "Schedule", "Rep", "Status", "Logs", "Epochs in logs", "Model loads", "Checkpoint", "Snapshot", "Final usable", "Seed parity", "Phase parity"],
        task_rows,
    )
    lines += ["", "## Every scheduler log", ""]
    log_rows = []
    for log in sorted(logs, key=lambda item: (item["task_id"], item["name"])):
        log_rows.append([
            log["task_id"], log["name"], ", ".join(log["phases"]) or "—",
            ", ".join(f"{phase}:{len(numbers)}" for phase, numbers in log["epochs"].items()) or "—",
            ", ".join(log["ended"]) or "—",
            ", ".join(name for _phase, name in log["snapshots"]) or "—",
            "yes" if log["saved"] else "no", "yes" if log["error"] else "no",
        ])
    lines += _table(
        ["Task", "Log", "Phases started", "Epoch summaries", "Phases finished", "Snapshots", "Any model save", "Error marker"],
        log_rows,
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    """Write a Markdown audit without mutating the recovered campaign."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recovery_root", type=Path)
    parser.add_argument("--current-yaml", type=Path, default=Path(__file__).with_name("pretraining_experiment.yaml"))
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("recovery_audit.md"))
    args = parser.parse_args()
    rows, logs = audit(args.recovery_root, args.current_yaml)
    args.output.write_text(render(rows, logs), encoding="utf-8")
    logger.info("Wrote recovery audit to %s.", args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
