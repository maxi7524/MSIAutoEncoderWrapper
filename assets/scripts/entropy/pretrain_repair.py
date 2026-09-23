#!/usr/bin/env python3
"""Export audited legacy pretraining weights and import them into a staged plan."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import shutil
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import numpy._core.multiarray as numpy_multiarray
import torch
import yaml

from msi_autoencoder_wrapper.models.model_loader import ModelLoader
from msi_autoencoder_wrapper.runtime import build_plan, load_experiment_config
from msi_autoencoder_wrapper.runtime.naming import run_identifier
from msi_autoencoder_wrapper.runtime.planning.graph import verify_plan_graph
from msi_autoencoder_wrapper.runtime.output import is_completed_task, task_fingerprint, update_manifest
from msi_autoencoder_wrapper.utils.logger import get_custom_logger


logger = get_custom_logger(__name__)
CAMPAIGN_DIRECTORY = (
    Path(__file__).parents[2]
    / "experiments/autoencoder_architecture/experiment_runs_configs"
    / "segmentation_model/20_09_26_metaspace_base_pretrain"
)
DEFAULT_YAML = CAMPAIGN_DIRECTORY / "pretraining_experiment.yaml"
EXPECTED_ROLES = {"real_only": 10, "pretrained": 98}
ALLOWED_CHECKPOINT_GLOBALS = frozenset(
    {"numpy.dtype", "numpy.ndarray", "numpy._core.multiarray._reconstruct"}
)


def _sha256(path: Path) -> str:
    """Hash one file without loading it into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path: Path) -> Any:
    """Read one JSON document."""
    return json.loads(path.read_text(encoding="utf-8"))


def _yaml(path: Path) -> dict[str, Any]:
    """Read one YAML mapping with the C loader when available."""
    with path.open(encoding="utf-8") as stream:
        value = yaml.load(stream, Loader=getattr(yaml, "CSafeLoader", yaml.SafeLoader))
    if not isinstance(value, dict):
        raise ValueError(f"Expected YAML mapping: {path}")
    return value


def _plan_task_ids(path: Path) -> set[str]:
    """Validate a large aggregate plan without materializing every task in RAM."""
    task_ids: list[str] = []
    schema_version: str | None = None
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.startswith("runtime_schema_version: "):
                schema_version = line.partition(": ")[2].strip()
            elif line.startswith("- task_id: "):
                task_ids.append(line.partition(": ")[2].strip())
    if schema_version not in {"2", "3"} or len(task_ids) != 310 or len(set(task_ids)) != 310:
        raise ValueError("Target must be a complete 310-task schema-2 or schema-3 plan.")
    with path.open(encoding="utf-8") as stream:
        for _event in yaml.parse(stream, Loader=getattr(yaml, "CLoader", yaml.Loader)):
            pass
    return set(task_ids)


def _write_json(path: Path, value: Any) -> None:
    """Write canonical, human-readable JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def _signature(task: Any) -> dict[str, Any]:
    """Identify one scientific trial independently of runtime task numbering."""
    grid = task.grid_parameters if hasattr(task, "grid_parameters") else task["grid_parameters"]
    repetition = task.repetition if hasattr(task, "repetition") else task["repetition"]
    return {
        "axis": grid["axes"]["name"],
        "architecture": grid["architectures"]["name"],
        "phases": [phase["phase_name"] for phase in grid["schedules"]],
        "repetition": int(repetition),
    }


def _scientific_schedule(task: Any) -> list[dict[str, Any]]:
    """Exclude only workflow bookkeeping from a resolved schedule."""
    grid = task.grid_parameters if hasattr(task, "grid_parameters") else task["grid_parameters"]
    return [
        {
            key: value for key, value in phase.items()
            if key not in {"workflow_role", "save_model_state_as", "restore_model_state_from"}
        }
        for phase in grid["schedules"]
    ]


def _canonical_model(value: Any) -> Any:
    """Remove persistence-only fields from a model blueprint recursively."""
    if isinstance(value, dict):
        return {
            key: _canonical_model(item)
            for key, item in value.items()
            if key != "module" and not (key == "preset" and item is None)
        }
    if isinstance(value, list):
        return [_canonical_model(item) for item in value]
    return value


def _load_snapshot(path: Path, fingerprint: str) -> dict[str, torch.Tensor]:
    """Load an audited pretraining snapshot with restricted deserialization."""
    unexpected = set(torch.serialization.get_unsafe_globals_in_checkpoint(path))
    if unexpected - ALLOWED_CHECKPOINT_GLOBALS:
        raise ValueError(f"Unexpected checkpoint globals: {sorted(unexpected)}")
    allowed = [np.dtype, np.ndarray, numpy_multiarray._reconstruct, np.dtypes.UInt32DType]
    with torch.serialization.safe_globals(allowed):
        payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or payload.get("task_fingerprint") != fingerprint:
        raise ValueError(f"Checkpoint fingerprint mismatch: {path}")
    state = payload.get("phase_snapshots", {}).get("synthetic_pretrained")
    if not isinstance(state, dict) or not state or not all(
        isinstance(name, str) and isinstance(tensor, torch.Tensor)
        and bool(torch.isfinite(tensor).all())
        for name, tensor in state.items()
    ):
        raise ValueError(f"Invalid synthetic_pretrained state: {path}")
    return state


def _load_recovery_audit() -> Any:
    """Reuse the campaign's individual log/status/model audit as export gate."""
    path = CAMPAIGN_DIRECTORY / "recovery_audit.py"
    specification = importlib.util.spec_from_file_location("pretraining_recovery_audit", path)
    if specification is None or specification.loader is None:
        raise RuntimeError(f"Cannot load recovery audit: {path}")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def export_bundle(recovery_root: Path, output: Path, yaml_path: Path) -> dict[str, Any]:
    """Materialize only individually audited baseline and pretraining states.

    :param recovery_root: Local recovery root containing old plan, logs and models.
    :type recovery_root: pathlib.Path
    :param output: New empty bundle directory.
    :type output: pathlib.Path
    :param yaml_path: Current source experiment YAML.
    :type yaml_path: pathlib.Path
    :return: Portable bundle manifest.
    :rtype: dict[str, typing.Any]
    :raises ValueError: If an expected source result or target mapping is invalid.
    """
    if output.exists():
        raise FileExistsError(f"Bundle destination already exists: {output}")
    audit = _load_recovery_audit()
    observations, _logs = audit.audit(recovery_root, yaml_path)
    targets = build_plan(load_experiment_config(yaml_path))
    by_signature = {
        json.dumps(_signature(task), sort_keys=True): task
        for task in targets.tasks
        if task.workflow and task.workflow["role"] in EXPECTED_ROLES
    }
    if len(by_signature) != 110:
        raise ValueError("Target plan does not have 110 distinct importable logical trials.")
    selected = [
        row for row in observations
        if (not row["synthetic"] and row["final_ok"])
        or (row["synthetic"] and row["final_ok"] and row["snapshot_ok"])
    ]
    role_counts = Counter("pretrained" if row["synthetic"] else "real_only" for row in selected)
    if dict(role_counts) != EXPECTED_ROLES:
        raise ValueError(f"Audited source role counts changed: {dict(role_counts)}")
    if any(not row["seeds_match"] or not row["phases_match"] for row in selected):
        raise ValueError("Source seed or phase parity failed for an import candidate.")

    # Verified source-data reference
    ## The old split proves sample-ID coverage, not unchanged source file bytes.
    source_split = recovery_root / "entropy-run/plan/resolved/splits/split-0000.yaml"
    split = _yaml(source_split)
    if set(split.get("assignments", {})) != {"train", "validation", "test"}:
        raise ValueError("Recovered split lacks train/validation/test assignments.")
    output.mkdir(parents=True)
    shutil.copy2(source_split, output / "source_split.yaml")
    campaign_id = recovery_root.name.removesuffix("-recovery")
    entries: list[dict[str, Any]] = []

    # Model-bundle construction
    ## Each output directory is named for the exact target task and role.
    for index, row in enumerate(selected, start=1):
        old_task_id = row["task_id"]
        old_task = _yaml(recovery_root / "entropy-run/plan/tasks" / f"{old_task_id}.yaml")
        signature = _signature(old_task)
        target = by_signature.get(json.dumps(signature, sort_keys=True))
        if target is None:
            raise ValueError(f"No new-plan target for old task {old_task_id}: {signature}")
        role = "pretrained" if row["synthetic"] else "real_only"
        if target.workflow["role"] != role:
            raise ValueError(f"Role mismatch for {old_task_id} -> {target.task_id}")
        old_model = (
            recovery_root / "models/kidney" / f"{campaign_id}__{old_task_id}" / "config"
        )
        destination = output / "models" / f"{target.task_id}__{role}" / "config"
        destination.mkdir(parents=True)
        shutil.copy2(old_model / "config.json", destination / "config.json")
        source_history = _json(old_model / "history.json")
        if role == "pretrained":
            phase_names = {phase["phase_name"] for phase in target.parameters["training"]["phases"]}
            history = [entry for entry in source_history if entry.get("phase") in phase_names]
            _write_json(destination / "history.json", history)
            checkpoint = recovery_root / "entropy-run/plan/checkpoints" / f"{old_task_id}.pt"
            snapshot = _load_snapshot(checkpoint, task_fingerprint(old_task))
            torch.save(snapshot, destination / "weights.pt")
            checkpoint_hash = _sha256(checkpoint)
        else:
            shutil.copy2(old_model / "history.json", destination / "history.json")
            shutil.copy2(old_model / "weights.pt", destination / "weights.pt")
            checkpoint_hash = None
        bundle_model = destination.parent
        ModelLoader.load_artifact(bundle_model, strict=True)
        files = {
            name: _sha256(destination / name)
            for name in ("config.json", "history.json", "weights.pt")
        }
        entries.append({
            "source_task_id": old_task_id,
            "target_task_id": target.task_id,
            "role": role,
            "signature": signature,
            "reproducibility": old_task["reproducibility"],
            "scientific_schedule": _scientific_schedule(old_task),
            "source_checkpoint_sha256": checkpoint_hash,
            "model_directory": f"models/{target.task_id}__{role}",
            "files_sha256": files,
            "source_logs": [log["name"] for log in row["logs"]],
        })
        if index % 10 == 0:
            logger.info("Exported %s/%s audited repair models.", index, len(selected))

    manifest = {
        "schema_version": 1,
        "source_campaign": campaign_id,
        "source_yaml_sha256": _sha256(yaml_path),
        "source_split_sha256": _sha256(output / "source_split.yaml"),
        "data_identity_level": "sample_ids_only",
        "expected_target_task_count": len(targets.tasks),
        "entries": sorted(entries, key=lambda entry: entry["target_task_id"]),
    }
    _write_json(output / "manifest.json", manifest)
    return manifest


def _validate_split(source: dict[str, Any], target: dict[str, Any]) -> None:
    """Require identical sample IDs and partition assignments."""
    for key in ("seed", "dataset_fingerprint", "assignments"):
        if source.get(key) != target.get(key):
            raise ValueError(f"Source and target split differ in {key}.")


def _validate_bundle(
    bundle: Path,
    plan_directory: Path,
    yaml_path: Path,
    campaign_id: str,
    workspace: Path,
) -> list[dict[str, Any]]:
    """Validate the full bundle and all target tasks before any import write."""
    manifest = _json(bundle / "manifest.json")
    if manifest.get("schema_version") != 1:
        raise ValueError("Unsupported repair bundle schema.")
    if _sha256(yaml_path) != manifest.get("source_yaml_sha256"):
        raise ValueError("Source YAML bytes differ from the exported bundle.")
    if _sha256(bundle / "source_split.yaml") != manifest.get("source_split_sha256"):
        raise ValueError("Source split file checksum mismatch.")
    source_split = _yaml(bundle / "source_split.yaml")
    target_splits = sorted((plan_directory / "resolved/splits").glob("split-*.yaml"))
    if len(target_splits) != 1:
        raise ValueError(f"Expected one shared target split, found {len(target_splits)}.")
    _validate_split(source_split, _yaml(target_splits[0]))
    manifest_task_ids = _plan_task_ids(plan_directory / "resolved-experiment.yaml")
    with (plan_directory / "resolved-experiment.yaml").open(encoding="utf-8") as stream:
        if stream.readline().strip() == "runtime_schema_version: 3":
            verify_plan_graph(plan_directory, expected_count=310)
    descriptor_task_ids = {path.stem for path in (plan_directory / "tasks").glob("task_*.yaml")}
    if len(manifest_task_ids) != 310 or manifest_task_ids != descriptor_task_ids:
        raise ValueError("Aggregate plan and individual target descriptors disagree.")
    entries = manifest.get("entries")
    if not isinstance(entries, list) or len(entries) != 108:
        raise ValueError("Repair bundle must contain exactly 108 model entries.")
    counts = Counter(entry.get("role") for entry in entries)
    if dict(counts) != EXPECTED_ROLES:
        raise ValueError(f"Repair bundle role counts changed: {dict(counts)}")
    if len({entry["target_task_id"] for entry in entries}) != len(entries):
        raise ValueError("Duplicate target task in repair bundle.")

    # Full preflight
    ## Confirm all file hashes, scientific identities and architecture loads.
    validated = []
    for entry in entries:
        task_id = entry["target_task_id"]
        task = _yaml(plan_directory / "tasks" / f"{task_id}.yaml")
        role = entry["role"]
        if task["task_id"] != task_id or task.get("workflow", {}).get("role") != role:
            raise ValueError(f"Target role mismatch: {task_id}")
        if _signature(task) != entry["signature"]:
            raise ValueError(f"Target scientific identity mismatch: {task_id}")
        if task["reproducibility"] != entry["reproducibility"]:
            raise ValueError(f"Target seed mismatch: {task_id}")
        if _scientific_schedule(task) != entry["scientific_schedule"]:
            raise ValueError(f"Target schedule mismatch: {task_id}")
        if role == "pretrained" and task.get("depends_on"):
            raise ValueError(f"Pretrained parent unexpectedly depends on another task: {task_id}")
        model_name = task["parameters"].get("model_name")
        expected_name = run_identifier(campaign_id, task)
        if model_name != expected_name:
            raise ValueError(f"Target model name is not role-qualified: {task_id}")
        if entry["model_directory"] != f"models/{task_id}__{role}":
            raise ValueError(f"Unsafe bundle model directory for {task_id}")
        model_dir = bundle / entry["model_directory"]
        if model_dir.name != f"{task_id}__{role}":
            raise ValueError(f"Unsafe bundle model directory: {model_dir}")
        source_config_dir = model_dir / "config"
        if set(entry["files_sha256"]) != {"config.json", "history.json", "weights.pt"}:
            raise ValueError(f"Incomplete bundle file manifest: {task_id}")
        for name, expected_hash in entry["files_sha256"].items():
            if name not in {"config.json", "history.json", "weights.pt"}:
                raise ValueError(f"Unexpected bundle file name: {name}")
            if _sha256(source_config_dir / name) != expected_hash:
                raise ValueError(f"Bundle checksum mismatch: {task_id}/{name}")
        old_model = _json(source_config_dir / "config.json")["model"]
        target_blueprint = _yaml(Path(task["parameters"]["resolved"]["model_config"]))["model"]
        if _canonical_model(old_model) != _canonical_model(target_blueprint):
            raise ValueError(f"Target model configuration differs: {task_id}")
        state = torch.load(source_config_dir / "weights.pt", map_location="cpu", weights_only=True)
        model, _, _ = ModelLoader.build({"model": target_blueprint})
        model.load_state_dict(state, strict=True)
        if any(not bool(torch.isfinite(tensor).all()) for tensor in model.state_dict().values()):
            raise ValueError(f"Non-finite target weights: {task_id}")
        target_dir = workspace / "models" / "kidney" / expected_name
        status_path = plan_directory / "status" / f"{task_id}.yaml"
        if target_dir.exists() or status_path.exists():
            raise FileExistsError(f"Target model or status already exists: {task_id}")
        validated.append({
            "task": task,
            "entry": entry,
            "model_dir": model_dir,
            "target_dir": target_dir,
            "status_path": status_path,
            "target_blueprint": target_blueprint,
        })
    return validated


def import_bundle(
    bundle: Path,
    plan_directory: Path,
    yaml_path: Path,
    campaign_id: str,
    workspace: Path,
    *,
    apply: bool,
    accept_sample_id_only: bool,
) -> int:
    """Verify and optionally import one bundle into a staged server campaign.

    :param bundle: Exported local bundle copied to the server.
    :type bundle: pathlib.Path
    :param plan_directory: Server-materialized plan directory.
    :type plan_directory: pathlib.Path
    :param yaml_path: Unmodified source experiment YAML.
    :type yaml_path: pathlib.Path
    :param campaign_id: Server campaign ID used by Entropy staging.
    :type campaign_id: str
    :param workspace: Persistent server workspace, not node-local staging.
    :type workspace: pathlib.Path
    :param apply: Create model bundles and completed statuses after verification.
    :type apply: bool
    :param accept_sample_id_only: Acknowledge that old source bytes cannot be proven.
    :type accept_sample_id_only: bool
    :return: Number of individually verified import candidates.
    :rtype: int
    """
    if apply and not accept_sample_id_only:
        raise ValueError(
            "Import requires --accept-sample-id-only: old source spectra and "
            "annotation bytes were not recovered."
        )
    validated = _validate_bundle(bundle, plan_directory, yaml_path, campaign_id, workspace)
    logger.info("Verified %s repair entries; apply=%s.", len(validated), apply)
    if not apply:
        return len(validated)

    # Recoverable per-model installation
    ## Publish a model directory before its status; a failed copy never marks it done.
    for item in validated:
        target = item["target_dir"]
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = Path(tempfile.mkdtemp(prefix=f".{target.name}-", dir=target.parent))
        try:
            source_config = item["model_dir"] / "config"
            output_config = temporary / "config"
            output_config.mkdir()
            shutil.copy2(source_config / "weights.pt", output_config / "weights.pt")
            shutil.copy2(source_config / "history.json", output_config / "history.json")
            _write_json(output_config / "config.json", {"model": item["target_blueprint"]})
            ModelLoader.load_artifact(temporary, strict=True)
            temporary.rename(target)
            task = item["task"]
            entry = item["entry"]
            history = _json(target / "config/history.json")
            update_manifest(
                item["status_path"], task["task_id"],
                {
                    "status": "completed",
                    "task_fingerprint": task_fingerprint(task),
                    "result": {
                        "model_path": str(target.resolve()),
                        "epochs": len(history),
                        "recovery": {
                            "source_campaign": _json(bundle / "manifest.json")["source_campaign"],
                            "source_task_id": entry["source_task_id"],
                            "source_checkpoint_sha256": entry["source_checkpoint_sha256"],
                            "weights_sha256": _sha256(target / "config/weights.pt"),
                            "data_identity_level": "sample_ids_only",
                        },
                    },
                },
            )
            if not is_completed_task(item["status_path"], task):
                raise RuntimeError(f"Imported status failed fingerprint validation: {task['task_id']}")
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)
    logger.info("Imported %s parent/baseline models; 202 tasks remain pending.", len(validated))
    return len(validated)


def verify_installed(bundle: Path, plan_directory: Path, campaign_id: str, workspace: Path) -> tuple[int, int]:
    """Verify installed model/status pairs against bundle hashes and target tasks.

    :param bundle: Exported recovery bundle with its manifest.
    :type bundle: pathlib.Path
    :param plan_directory: Materialized campaign plan directory.
    :type plan_directory: pathlib.Path
    :param campaign_id: Campaign ID used in staged task names.
    :type campaign_id: str
    :param workspace: Workspace containing installed kidney models.
    :type workspace: pathlib.Path
    :return: Verified recovered models and all completed task counts.
    :rtype: tuple[int, int]
    :raises ValueError: If an imported model or task status is inconsistent.
    """
    manifest = _json(bundle / "manifest.json")
    entries = manifest.get("entries", [])
    if len(entries) != 108 or dict(Counter(item["role"] for item in entries)) != EXPECTED_ROLES:
        raise ValueError("Recovery manifest must contain 108 expected role entries.")
    task_paths = {
        path.stem: path
        for path in sorted((plan_directory / "tasks").glob("task_*.yaml"))
    }
    if len(task_paths) != 310:
        raise ValueError("Expected a complete 310-task target plan.")

    # Verify each recovered artifact and status against the target descriptor.
    for entry in entries:
        task_id = entry["target_task_id"]
        task = _yaml(task_paths[task_id])
        role = entry["role"]
        if task.get("workflow", {}).get("role") != role:
            raise ValueError(f"Recovered task role changed: {task_id}")
        model_name = run_identifier(campaign_id, task)
        if task["parameters"].get("model_name") != model_name:
            raise ValueError(f"Recovered task model name changed: {task_id}")
        status_path = plan_directory / "status" / f"{task_id}.yaml"
        if not is_completed_task(status_path, task):
            raise ValueError(f"Recovered task status is not completed: {task_id}")
        record = _yaml(status_path)["records"][task_id]
        if Path(record.get("result", {}).get("model_path", "")).name != model_name:
            raise ValueError(f"Recovered status model path differs: {task_id}")
        model_dir = workspace / "models/kidney" / model_name
        config_dir = model_dir / "config"
        if _sha256(config_dir / "weights.pt") != entry["files_sha256"]["weights.pt"]:
            raise ValueError(f"Recovered model weights differ: {task_id}")
        if _sha256(config_dir / "history.json") != entry["files_sha256"]["history.json"]:
            raise ValueError(f"Recovered model history differs: {task_id}")
        ModelLoader.load_artifact(model_dir, strict=True)
    completed = 0
    for status_path in sorted((plan_directory / "status").glob("task_*.yaml")):
        task = _yaml(task_paths[status_path.stem])
        if not is_completed_task(status_path, task):
            raise ValueError(f"Invalid completed task status: {status_path.stem}")
        completed += 1
    logger.info("Verified %s recovered models in kidney; %s/%s tasks completed.", len(entries), completed, len(task_paths))
    return len(entries), completed


def main() -> int:
    """Run the export or server import command."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    export = commands.add_parser("export")
    export.add_argument("--recovery-root", type=Path, required=True)
    export.add_argument("--output", type=Path, required=True)
    export.add_argument("--yaml", type=Path, default=DEFAULT_YAML)
    imported = commands.add_parser("import")
    imported.add_argument("--bundle", type=Path, required=True)
    imported.add_argument("--plan-directory", type=Path, required=True)
    imported.add_argument("--yaml", type=Path, default=DEFAULT_YAML)
    imported.add_argument("--campaign-id", required=True)
    imported.add_argument("--workspace", type=Path, required=True)
    imported.add_argument("--apply", action="store_true")
    imported.add_argument("--accept-sample-id-only", action="store_true")
    verified = commands.add_parser("verify-installed")
    verified.add_argument("--bundle", type=Path, required=True)
    verified.add_argument("--plan-directory", type=Path, required=True)
    verified.add_argument("--campaign-id", required=True)
    verified.add_argument("--workspace", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "export":
        manifest = export_bundle(args.recovery_root, args.output, args.yaml)
        logger.info("Exported %s audited model bundles to %s.", len(manifest["entries"]), args.output)
    elif args.command == "import":
        import_bundle(
            args.bundle, args.plan_directory, args.yaml, args.campaign_id,
            args.workspace, apply=args.apply,
            accept_sample_id_only=args.accept_sample_id_only,
        )
    else:
        verify_installed(args.bundle, args.plan_directory, args.campaign_id, args.workspace)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
