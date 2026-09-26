"""Content contracts, reuse and validation of the pretraining-campaign cache levels.

Every shared cache level (``plan``, ``populations``, ``synthetic_reference``,
``inference``) is identified by a *contract*: the semantic inputs its content depends
on (upstream contract digest, the relevant settings and an explicit schema version),
never the raw bytes of the campaign YAML or a digest of whole source packages. A level
directory is reused whenever its ``contract.json`` equals the requested contract.

REMARK: The previous keys hashed raw YAML bytes and whole source packages. A comment in
the campaign YAML, the role selection or an unrelated edit anywhere below
``models/architectures`` then silently invalidated hours of inference. Source digests
are therefore recorded for audit only; a level is recomputed when its contract changes,
and :data:`SCHEMAS` is bumped whenever the computation of a level changes on purpose.

Directories written before contracts existed (*legacy* directories) are never trusted
blindly. They are adopted only after a content validator re-derives what can be
re-derived cheaply (task parameters, split, decoded pixel samples, model outputs on
sampled rows) and finds it identical; the validation report is stored beside the
adopted contract. A failed validation never deletes anything: the directory is ignored
and the level is recomputed into a new directory.

The module is also a command-line status report that computes nothing::

    python -m msi_autoencoder_wrapper.analysis.autoencoder.experiments.pretraining_campaign_cache \\
        --settings <analysis_settings.yaml>
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

import pandas as pd

from ....utils.logger import get_custom_logger
from .predictive_campaign import fingerprint

logger = get_custom_logger(__name__)

#: Schema version per cache level; bump when a level's computation changes on purpose.
SCHEMAS = {"plan": 1, "populations": 1, "synthetic_reference": 1, "inference": 1}

#: File recording the contract (and audit information) of a level directory.
CONTRACT_FILE = "contract.json"

#: Completion marker of each level (a directory without it is incomplete).
COMPLETION_FILES = {"plan": "resolved_parameters.json", "populations": "complete.json",
                    "synthetic_reference": "complete.json", "inference": "complete.json"}

#: Parameter paths the plan resolver rewrites on purpose (``*`` matches one key).
RESOLVER_REWRITTEN_PATHS = (
    ("factory_parameters", "dataset", "parameters", "subset"),
    ("factory_parameters", "dataset", "parameters", "target_specs", "*", "class_mapping"),
)


# --------------------------------------------------
# Section: contracts
# --------------------------------------------------

def normalized(contract: dict) -> dict:
    """Contract in its stored JSON form (tuples become lists), the form compared on reuse."""
    return json.loads(json.dumps(contract, default=str))


def contract_digest(contract: dict) -> str:
    """Short digest naming the directory of a new level.

    :param contract: Level contract.
    :type contract: dict
    :return: First 16 hexadecimal digits of the contract fingerprint.
    :rtype: str
    """
    return fingerprint(contract)[:16]


def plan_contract(settings: dict, tasks: list[Any]) -> dict:
    """Contract of the plan level: the resolved tasks and their unresolved parameters.

    :param settings: Resolved analysis settings.
    :type settings: dict
    :param tasks: Reference tasks resolved by the level.
    :type tasks: list
    :return: Contract mapping.
    :rtype: dict
    """
    from .pretraining_campaign import task_parameter_fingerprint

    return {"level": "plan", "schema": SCHEMAS["plan"], "campaign_id": settings["campaign_id"],
            "tasks": {task.task_id: task_parameter_fingerprint(task) for task in tasks}}


def populations_contract(settings: dict, plan: dict) -> dict:
    """Contract of the population level.

    :param settings: Resolved analysis settings.
    :type settings: dict
    :param plan: Contract of the plan level.
    :type plan: dict
    :return: Contract mapping.
    :rtype: dict
    """
    relevant = {key: settings.get(key) for key in ("target_field", "evidence", "populations", "windows", "metaspace",
                                                   "display", "cases", "reference_mass_range")}
    relevant["merged_store"] = Path(settings["merged_store"]).name
    return {"level": "populations", "schema": SCHEMAS["populations"], "plan": contract_digest(plan),
            "settings": relevant}


def synthetic_reference_contract(plan: dict, artifacts: dict[str, dict]) -> dict:
    """Contract of the synthetic-reference level.

    :param plan: Contract of the plan level.
    :type plan: dict
    :param artifacts: Synthetic artifact declaration per axis (the phase ``pretraining``
        block without the selected population).
    :type artifacts: dict[str, dict]
    :return: Contract mapping.
    :rtype: dict
    """
    return {"level": "synthetic_reference", "schema": SCHEMAS["synthetic_reference"],
            "plan": contract_digest(plan), "artifacts": {axis: fingerprint(value) for axis, value in artifacts.items()}}


def inference_contract(settings: dict, populations: dict) -> dict:
    """Contract of the inference level (shared by every per-model directory).

    Only the settings that change a stored per-model array are part of it. The
    representative-model rule and the display selection are recomputed on every run.

    :param settings: Resolved analysis settings.
    :type settings: dict
    :param populations: Contract of the population level.
    :type populations: dict
    :return: Contract mapping.
    :rtype: dict
    """
    latent = settings.get("latent", {})
    sensitivity = {key: latent.get(key) for key in ("seed", "sensitivity_sample_size", "epsilons")}
    return {"level": "inference", "schema": SCHEMAS["inference"], "populations": contract_digest(populations),
            "settings": {"windows": settings.get("windows"), "cases": settings.get("cases"),
                         "target_field": settings.get("target_field"), "sensitivity": sensitivity}}


# --------------------------------------------------
# Section: level lookup and adoption
# --------------------------------------------------

@dataclass(frozen=True)
class LevelDirectory:
    """Directory chosen for one cache level.

    :param directory: Level directory.
    :type directory: pathlib.Path
    :param complete: Whether the level content is complete and reusable.
    :type complete: bool
    :param adopted: Whether a legacy directory was adopted in this call.
    :type adopted: bool
    """

    directory: Path
    complete: bool
    adopted: bool = False


def read_contract(directory: Path) -> Optional[dict]:
    """Contract stored in a level directory, or ``None`` for legacy directories."""
    path = directory / CONTRACT_FILE
    return json.loads(path.read_text())["contract"] if path.is_file() else None


def write_contract(directory: Path, contract: dict, audit: Optional[dict] = None) -> None:
    """Record the contract of a level directory atomically.

    :param directory: Level directory.
    :type directory: pathlib.Path
    :param contract: Level contract.
    :type contract: dict
    :param audit: Information not used for reuse decisions (source digests, the
        validation report of an adopted legacy directory).
    :type audit: dict | None
    """
    temporary = directory / f"{CONTRACT_FILE}.tmp"
    temporary.write_text(json.dumps({"contract": contract, "audit": audit or {}}, indent=2, default=str))
    temporary.replace(directory / CONTRACT_FILE)


def resolve_level(settings: dict, level: str, contract: dict,
                  legacy_validator: Optional[Callable[[Path], pd.DataFrame]] = None) -> LevelDirectory:
    """Find the directory of one level: reuse, adopt a validated legacy one, or create.

    Search order: (1) any directory whose stored contract equals ``contract``; (2) any
    complete legacy directory (no contract) accepted by ``legacy_validator``, which then
    receives the contract; (3) a new directory named by the contract digest.

    :param settings: Resolved analysis settings (``cache_directory``).
    :type settings: dict
    :param level: Level name, one of :data:`SCHEMAS`.
    :type level: str
    :param contract: Requested contract.
    :type contract: dict
    :param legacy_validator: Returns check rows (``check``, ``passed``, ``detail``) for a
        legacy directory; every row must pass for the directory to be adopted.
    :type legacy_validator: collections.abc.Callable | None
    :return: Chosen directory and whether its content is complete.
    :rtype: LevelDirectory
    """
    contract = normalized(contract)
    root = Path(settings["cache_directory"]) / level
    root.mkdir(parents=True, exist_ok=True)
    candidates = sorted((path for path in root.iterdir() if path.is_dir()), key=lambda path: path.name)
    marker = COMPLETION_FILES[level]
    ## Contract match
    for directory in candidates:
        if read_contract(directory) == contract:
            complete = (directory / marker).is_file()
            logger.info("Cache level '%s': contract match in %s (complete=%s).", level, directory, complete)
            return LevelDirectory(directory, complete)
    ## Legacy adoption after content validation
    if legacy_validator is not None:
        for directory in candidates:
            if (directory / CONTRACT_FILE).is_file() or not (directory / marker).is_file():
                continue
            logger.info("Cache level '%s': validating legacy directory %s.", level, directory)
            try:
                report = legacy_validator(directory)
            except (FileNotFoundError, KeyError, ValueError) as error:
                logger.warning("Legacy directory %s cannot be validated: %s", directory, error)
                continue
            failed = report[~report.passed.astype(bool)]
            if failed.empty and len(report):
                write_contract(directory, contract, {"adopted_legacy_directory": True,
                                                     "validation": report.to_dict("records")})
                logger.info("Cache level '%s': adopted %s after %s passed checks.", level, directory, len(report))
                return LevelDirectory(directory, True, adopted=True)
            logger.warning("Legacy directory %s rejected; failed checks:\n%s", directory,
                           failed.to_string(index=False))
    directory = root / contract_digest(contract)
    if directory.exists() and read_contract(directory) not in (None, contract):
        raise ValueError(f"Cache directory {directory} holds a different contract.")
    directory.mkdir(parents=True, exist_ok=True)
    if read_contract(directory) is None:
        write_contract(directory, contract)
    complete = (directory / marker).is_file()
    logger.info("Cache level '%s': using %s (complete=%s).", level, directory, complete)
    return LevelDirectory(directory, complete)


# --------------------------------------------------
# Section: legacy plan validation
# --------------------------------------------------

def _without(value: Any, path: tuple[str, ...]) -> Any:
    """Copy of a nested mapping without the entries matched by ``path``."""
    if not isinstance(value, dict) or not path:
        return value
    head, rest = path[0], path[1:]
    result = {}
    for key, item in value.items():
        if head in ("*", key):
            if not rest:
                continue
            result[key] = _without(item, rest)
        else:
            result[key] = item
    return result


def comparable_task_parameters(parameters: dict) -> dict:
    """Task parameters without the entries the plan resolver rewrites on purpose.

    :param parameters: Unresolved task parameters or stored resolved parameters
        (the ``resolved`` file paths are dropped as well).
    :type parameters: dict
    :return: JSON-normalized parameters comparable across resolution.
    :rtype: dict
    """
    value = json.loads(json.dumps({key: item for key, item in parameters.items() if key != "resolved"}, default=str))
    for path in RESOLVER_REWRITTEN_PATHS:
        value = _without(value, path)
    return value


def validate_legacy_plan(directory: Path, tasks: list[Any]) -> pd.DataFrame:
    """Check that a legacy resolved plan was produced from the current task parameters.

    :param directory: Legacy plan directory with ``resolved_parameters.json``.
    :type directory: pathlib.Path
    :param tasks: Reference tasks the level must contain.
    :type tasks: list
    :return: Check rows.
    :rtype: pandas.DataFrame
    """
    stored = json.loads((directory / "resolved_parameters.json").read_text())
    rows = [{"check": "task_set", "subject": directory.name,
             "passed": set(stored) == {task.task_id for task in tasks},
             "detail": f"stored={len(stored)} requested={len(tasks)}"}]
    for task in tasks:
        if task.task_id not in stored:
            continue
        equal = comparable_task_parameters(stored[task.task_id]) == comparable_task_parameters(task.parameters)
        files = all(Path(path).is_file() for path in stored[task.task_id]["resolved"].values())
        rows.append({"check": "task_parameters", "subject": task.task_id, "passed": equal,
                     "detail": "identical outside resolver-rewritten entries" if equal else "parameters differ"})
        rows.append({"check": "resolved_files", "subject": task.task_id, "passed": files,
                     "detail": "every resolved artifact file exists"})
    return pd.DataFrame(rows)


# --------------------------------------------------
# Section: status report (computes nothing)
# --------------------------------------------------

def cache_status(settings_path: Path | str) -> pd.DataFrame:
    """Report which cache levels and model evaluations can be reused.

    :param settings_path: Analysis settings YAML.
    :type settings_path: pathlib.Path | str
    :return: One row per level plus one summary row of per-model inference markers.
    :rtype: pandas.DataFrame
    """
    from . import pretraining_campaign as campaign

    settings = campaign.load_settings(settings_path)
    tasks = campaign.campaign_tasks(settings)
    models = campaign.inventory(settings)
    plan = normalized(plan_contract(settings, campaign.reference_tasks(settings, tasks)))
    populations = normalized(populations_contract(settings, plan))
    inference = normalized(inference_contract(settings, populations))
    rows = []
    for level, contract in (("plan", plan), ("populations", populations), ("inference", inference)):
        root = Path(settings["cache_directory"]) / level
        directories = [path for path in root.iterdir() if path.is_dir()] if root.is_dir() else []
        matching = [path for path in directories if read_contract(path) == contract]
        legacy = [path for path in directories if read_contract(path) is None]
        rows.append({"level": level, "contract": contract_digest(contract),
                     "matching_directory": matching[0].name if matching else "",
                     "legacy_candidates": ",".join(path.name for path in legacy)})
    ## Per-model evaluations in the matching directory, else in legacy candidates
    root = Path(settings["cache_directory"]) / "inference"
    directories = [path for path in (root.iterdir() if root.is_dir() else []) if path.is_dir()]
    matching = [path for path in directories if read_contract(path) == inference]
    searched = matching or [path for path in directories if read_contract(path) is None]
    for directory in searched:
        evaluated = sum((directory / model_id / "complete.json").is_file() for model_id in models.model_id)
        rows.append({"level": "inference models", "contract": "matching" if matching else "legacy candidate",
                     "matching_directory": directory.name,
                     "legacy_candidates": f"{evaluated} of {len(models)} models evaluated"})
    return pd.DataFrame(rows)


def main(argv: Optional[list[str]] = None) -> int:
    """Print the cache status of one analysis settings file."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--settings", required=True, help="Analysis settings YAML.")
    arguments = parser.parse_args(argv)
    print(cache_status(arguments.settings).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
