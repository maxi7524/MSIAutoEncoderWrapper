"""Shared cache, analysis registry and result I/O of the pretraining-campaign analyses.

The computation is a strict three-level DAG so that no notebook, and no analysis, ever
depends on the output of another analysis:

1. **Shared cache** (outside the repository, reused across runs):

   ``campaign_plan``
       resolves and verifies the data contract of the selected models;
   ``populations``
       decodes every pixel population once per spectral axis, together with targets,
       evidence states, coordinates, class catalogue and METASPACE reference images;
   ``synthetic_reference``
       regenerates the synthetic pretraining artifacts and derives the class sets their
       quotas target (rare classes, bin-colliding classes);
   ``inference``
       evaluates every model once on every population and stores per-pixel metrics,
       latent codes, windowed Masserstein arrays and ranking tables.

2. **Registered analyses** (``ANALYSES``) read only the shared cache and write the
   canonical CSV tables of exactly one notebook (per axis when the analysis is
   per-axis).

3. **Notebooks** read only their own result directory through :func:`load_table`.

Every shared level is identified by a content contract
(:mod:`.pretraining_campaign_cache`); a matching contract is reused, a changed one
produces a new directory instead of silently reusing stale arrays, and directories
written before contracts existed are adopted only after content validation.

Analyses have a model *scope*: ``baseline`` (the real-data baselines only, part 1),
``all`` (every selected model) or ``stage`` (the baselines plus one pretraining stage,
run once per axis and stage). An analysis whose recorded inputs are unchanged is not
recomputed.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import shlex
import traceback
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd

from ....utils.logger import get_custom_logger
from . import pretraining_campaign as campaign
from .predictive_precompute import _source_digest, provenance

logger = get_custom_logger(__name__)

#: Per-pixel scalar reconstruction metrics stored for every population.
PIXEL_METRICS = ("masserstein", "mse", "mae", "cosine_similarity", "spectral_angle", "tic_error")

#: Window arrays of the local Masserstein decomposition stored for every population.
WINDOW_ARRAYS = ("contribution", "within", "input_mass", "output_mass")

#: Populations on which ranking metrics are evaluated (``test_combined`` = test and
#: test_extended pixels together).
PREDICTION_POPULATIONS = ("train", "test", "test_extended", "test_combined", "heldout_image")


# --------------------------------------------------
# Section: analysis registry and result paths
# --------------------------------------------------

@dataclass(frozen=True)
class RegisteredAnalysis:
    """One analysis producing the canonical tables of one notebook.

    :param name: Analysis key in the settings ``analyses`` block.
    :type name: str
    :param function: ``function(cache, axis) -> {table: DataFrame, "metadata": dict}``;
        ``axis`` is ``None`` for campaign-level analyses.
    :type function: collections.abc.Callable
    :param per_axis: Whether one result directory is produced per spectral axis.
    :type per_axis: bool
    :param requires: Shared stages that must be complete before the analysis runs.
    :type requires: tuple[str, ...]
    :param scope: Model scope: ``baseline``, ``all`` or ``stage`` (see module docstring).
    :type scope: str
    :param version: Bumped when the analysis output changes although its function
        source does not (e.g. through a shared helper).
    :type version: int
    """

    name: str
    function: Callable[..., dict]
    per_axis: bool
    requires: tuple[str, ...]
    scope: str = "baseline"
    version: int = 1
    source_sha256: str = ""


ANALYSES: dict[str, RegisteredAnalysis] = {}

#: Model scopes of registered analyses.
SCOPES = ("baseline", "all", "stage")


def register(name: str, *, per_axis: bool, requires: tuple[str, ...] = ("inference",), scope: str = "baseline",
             version: int = 1) -> Callable:
    """Register one analysis function under its settings key.

    :param name: Analysis key.
    :type name: str
    :param per_axis: Whether the analysis runs once per spectral axis.
    :type per_axis: bool
    :param requires: Shared stages it reads (subset of ``campaign_plan``,
        ``populations``, ``synthetic_reference``, ``inference``).
    :type requires: tuple[str, ...]
    :param scope: Model scope, one of :data:`SCOPES`. ``stage`` analyses are called as
        ``function(cache, axis, stage)`` and are always per axis.
    :type scope: str
    :param version: Output version (see :class:`RegisteredAnalysis`).
    :type version: int
    :return: Decorator returning the function unchanged.
    :rtype: collections.abc.Callable
    :raises ValueError: If the scope is unknown.
    """
    if scope not in SCOPES:
        raise ValueError(f"Unknown analysis scope '{scope}'; expected one of {SCOPES}.")

    def decorator(function: Callable[..., dict]) -> Callable[..., dict]:
        ## REMARK: the source digest is taken at import time; reading the source later
        ## would hash an edited file if it changed while a long run was in progress.
        digest = hashlib.sha256(inspect.getsource(function).encode()).hexdigest()
        ANALYSES[name] = RegisteredAnalysis(name, function, per_axis or scope == "stage", tuple(requires), scope,
                                            version, digest)
        return function

    return decorator


def axis_directory(settings: dict, axis: str) -> str:
    """Return the folder name used for one spectral axis."""
    return str(settings["axes"][axis]["directory"])


def results_directory(settings: dict, analysis: str, axis: Optional[str] = None,
                      stage: Optional[str] = None) -> Path:
    """Return the result directory of one analysis (and axis, and stage).

    Campaign-level analyses write to ``output_directory``. Per-axis analyses write to
    ``output_directory / <axis directory> / result_directory``. Stage analyses configure
    one entry per stage under ``stages`` and write to ``output_directory / <axis
    directory> / [stage_directory] / result_directory``.

    :param settings: Resolved settings.
    :type settings: dict
    :param analysis: Analysis key.
    :type analysis: str
    :param axis: Axis name for per-axis analyses.
    :type axis: str | None
    :param stage: Stage name for stage analyses.
    :type stage: str | None
    :return: Absolute directory.
    :rtype: pathlib.Path
    :raises ValueError: If the analysis is not configured or the axis/stage arguments are
        inconsistent.
    """
    configured = settings.get("analyses", {}).get(analysis)
    if configured is None:
        raise ValueError(f"Analysis '{analysis}' is not configured in {settings['settings_path']}.")
    if "stages" in configured:
        if stage not in configured["stages"] or axis is None:
            raise ValueError(f"Analysis '{analysis}' is per stage; pass an axis and one of {list(configured['stages'])}.")
        entry = configured["stages"][stage]
        root = Path(entry["output_directory"])
        if not root.is_absolute():
            root = Path(settings["repository_root"]) / root
        root = root / axis_directory(settings, axis)
        if entry.get("stage_directory"):
            root = root / entry["stage_directory"]
        return root / entry["result_directory"]
    root = Path(configured["output_directory"])
    if not root.is_absolute():
        root = Path(settings["repository_root"]) / root
    if "result_directory" in configured:
        if axis is None:
            raise ValueError(f"Analysis '{analysis}' is per-axis; pass one of {list(settings['axes'])}.")
        return root / axis_directory(settings, axis) / configured["result_directory"]
    return root


def _json_default(value: Any) -> Any:
    """Serialize numpy scalars and paths in metadata."""
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)


def write_results(directory: Path, tables: dict[str, Any]) -> None:
    """Write every table as CSV and the ``metadata`` entry as JSON.

    :param directory: Result directory; created when absent.
    :type directory: pathlib.Path
    :param tables: DataFrames keyed by file stem plus a ``metadata`` mapping.
    :type tables: dict[str, typing.Any]
    """
    directory.mkdir(parents=True, exist_ok=True)
    metadata = dict(tables.get("metadata", {}))
    metadata["tables"] = sorted(key for key in tables if key != "metadata")
    for name, frame in tables.items():
        if name != "metadata":
            frame.to_csv(directory / f"{name}.csv", index=False)
    (directory / "metadata.json").write_text(json.dumps(metadata, indent=2, default=_json_default))


def run_command(settings: dict, analysis: Optional[str] = None, *, background: bool = True) -> str:
    """Return the canonical command producing the tables of one analysis.

    :param settings: Resolved settings.
    :type settings: dict
    :param analysis: Analysis key; ``None`` runs every configured analysis.
    :type analysis: str | None
    :param background: Detach the command and log beside the settings file.
    :type background: bool
    :return: Shell command generated from the same entry point as the runner.
    :rtype: str
    """
    from ...precompute.cli import run_precompute_command

    command = run_precompute_command(settings["settings_path"], background=False)
    if analysis is not None:
        command += f" --analysis {shlex.quote(analysis)}"
    if not background:
        return command
    log = Path(settings["settings_path"]).parent / "shared_precompute.log"
    return f"nohup {command} > {shlex.quote(str(log))} 2>&1 &"


def load_table(settings: dict, analysis: str, table: str, axis: Optional[str] = None,
               stage: Optional[str] = None) -> pd.DataFrame:
    """Load one canonical result table, or explain how to produce it.

    :param settings: Resolved settings.
    :type settings: dict
    :param analysis: Analysis key.
    :type analysis: str
    :param table: Table stem.
    :type table: str
    :param axis: Axis name for per-axis analyses.
    :type axis: str | None
    :param stage: Stage name for stage analyses.
    :type stage: str | None
    :return: Table contents.
    :rtype: pandas.DataFrame
    :raises FileNotFoundError: With the producing command when the table is missing.
    """
    path = results_directory(settings, analysis, axis, stage) / f"{table}.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Missing '{path}'. Produce it with:\n  {run_command(settings, analysis)}")
    return pd.read_csv(path)


def load_metadata(settings: dict, analysis: str, axis: Optional[str] = None, stage: Optional[str] = None) -> dict:
    """Load the provenance metadata written with one analysis.

    :param settings: Resolved settings.
    :type settings: dict
    :param analysis: Analysis key.
    :type analysis: str
    :param axis: Axis name for per-axis analyses.
    :type axis: str | None
    :param stage: Stage name for stage analyses.
    :type stage: str | None
    :return: Metadata mapping.
    :rtype: dict
    :raises FileNotFoundError: With the producing command when the file is missing.
    """
    path = results_directory(settings, analysis, axis, stage) / "metadata.json"
    if not path.is_file():
        raise FileNotFoundError(f"Missing '{path}'. Produce it with:\n  {run_command(settings, analysis)}")
    return json.loads(path.read_text())


# --------------------------------------------------
# Section: shared cache view
# --------------------------------------------------

@dataclass
class CampaignCache:
    """Resolved locations and lazy readers of the shared cache levels.

    :param settings: Resolved settings.
    :type settings: dict
    :param models: Selected catalog records (one row per model repetition).
    :type models: pandas.DataFrame
    """

    settings: dict
    models: pd.DataFrame
    plan: dict = field(default_factory=dict)
    population_root: Optional[Path] = None
    synthetic_root: Optional[Path] = None
    inference_root: Optional[Path] = None
    keys: dict = field(default_factory=dict)
    contracts: dict = field(default_factory=dict)
    _pixels: Optional[tuple] = field(default=None, repr=False)

    # Model selection
    @property
    def axes(self) -> list[str]:
        """Configured axes that have at least one selected model, in settings order."""
        present = set(self.models.axis)
        return [axis for axis in self.settings["axes"] if axis in present]

    @property
    def stages(self) -> list[str]:
        """Configured pretraining stages that have at least one selected model."""
        present = set(self.models.role)
        return [stage for stage in self.settings.get("stages", {}) if stage in present]

    def axis_models(self, axis: str) -> pd.DataFrame:
        """Selected models of one axis ordered by repetition."""
        return self.models[self.models.axis == axis].sort_values("repetition").reset_index(drop=True)

    def view(self, models: pd.DataFrame) -> "CampaignCache":
        """The same cache restricted to a subset of the selected models.

        :param models: Rows of :attr:`models`.
        :type models: pandas.DataFrame
        :return: Cache view sharing every resolved location.
        :rtype: CampaignCache
        """
        return replace(self, models=models.reset_index(drop=True))

    def baseline_view(self) -> "CampaignCache":
        """View restricted to the real-data baseline models."""
        return self.view(self.models[self.models.role == self.settings["baseline_role"]])

    def stage_view(self, axis: str, stage: str) -> "CampaignCache":
        """View of one axis restricted to the baselines and the models of one stage."""
        selected = (self.models.axis == axis) & self.models.role.isin([self.settings["baseline_role"], stage])
        return self.view(self.models[selected])

    def cells(self, axis: Optional[str] = None) -> pd.DataFrame:
        """One row per model alias (cell) with its axis, variant and stage, in display order."""
        frame = self.models if axis is None else self.models[self.models.axis == axis]
        columns = ["model_alias", "display_label", "axis", "variant", "role"]
        return (frame.sort_values(["display_order", "repetition"]).drop_duplicates("model_alias")[columns]
                .reset_index(drop=True))

    # Population level
    def axis_meta(self, axis: str) -> dict[str, np.ndarray]:
        """Mass axis, bin edges and window edges of one axis."""
        with np.load(self.population_root / axis_directory(self.settings, axis) / "axis.npz") as archive:
            return {key: archive[key] for key in archive.files}

    def population(self, axis: str, population: str) -> dict[str, np.ndarray]:
        """Memory-mapped arrays of one decoded population."""
        directory = self.population_root / axis_directory(self.settings, axis) / population
        return {path.stem: np.load(path, mmap_mode="r") for path in sorted(directory.glob("*.npy"))}

    def population_frame(self) -> pd.DataFrame:
        """Pixel identities, datasets and coordinates of every population (read once)."""
        path = self.population_root / "populations.csv"
        if self._pixels is None or self._pixels[0] != path:
            self._pixels = (path, pd.read_csv(path))
        return self._pixels[1].copy()

    def shared_table(self, name: str) -> pd.DataFrame:
        """One campaign-level table of the population level."""
        return pd.read_csv(self.population_root / f"{name}.csv")

    def axis_table(self, axis: str, name: str) -> pd.DataFrame:
        """One per-axis table of the population level."""
        return pd.read_csv(self.population_root / axis_directory(self.settings, axis) / f"{name}.csv")

    def axis_array(self, axis: str, name: str) -> np.ndarray:
        """One per-axis array of the population level."""
        return np.load(self.population_root / axis_directory(self.settings, axis) / f"{name}.npy", mmap_mode="r")

    # Inference level
    def model_directory(self, model_id: str) -> Path:
        """Directory of one evaluated model."""
        return self.inference_root / model_id

    def model_pixels(self, model_id: str, population: str) -> dict[str, np.ndarray]:
        """Per-pixel arrays of one model on one population."""
        with np.load(self.model_directory(model_id) / f"{population}_pixels.npz") as archive:
            return {key: archive[key] for key in archive.files}

    def model_table(self, model_id: str, name: str) -> pd.DataFrame:
        """One table written by the inference level for one model."""
        return pd.read_csv(self.model_directory(model_id) / f"{name}.csv")

    def model_array(self, model_id: str, name: str) -> np.ndarray:
        """One array written by the inference level for one model."""
        return np.load(self.model_directory(model_id) / f"{name}.npy", mmap_mode="r")

    def representative_ranking(self, axis: str) -> pd.DataFrame:
        """Metric ranks of the baseline repetitions of one axis with the representative flag."""
        return pd.read_csv(self.inference_root / f"representative__{axis_directory(self.settings, axis)}.csv")

    def cell_representative_ranking(self, alias: str) -> pd.DataFrame:
        """Metric ranks of the repetitions of one model alias with the representative flag."""
        return pd.read_csv(self.inference_root / f"representative__cell__{alias}.csv")

    def cell_representative(self, alias: str) -> Any:
        """Model record of the representative repetition of one model alias."""
        ranking = self.cell_representative_ranking(alias)
        chosen = ranking.model_id[ranking.representative.astype(bool)].iloc[0]
        return next(row for row in self.models.itertuples() if row.model_id == chosen)

    def representative(self, axis: str) -> Any:
        """Model record (``itertuples`` row) of the representative model of one axis."""
        ranking = self.representative_ranking(axis)
        chosen = ranking.model_id[ranking.representative.astype(bool)].iloc[0]
        return next(row for row in self.axis_models(axis).itertuples() if row.model_id == chosen)

    def model_summary(self, model_id: str) -> dict:
        """Inference summary (layer-norm parameters, verification values) of one model."""
        return json.loads((self.model_directory(model_id) / "summary.json").read_text())

    def collect(self, name: str, models: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Concatenate one per-model inference table with model identity columns."""
        frames = []
        for row in (self.models if models is None else models).itertuples():
            frame = self.model_table(row.model_id, name)
            frames.append(frame.assign(model_id=row.model_id, model_alias=row.model_alias,
                                       display_label=row.display_label, axis=row.axis,
                                       repetition=row.repetition, variant=row.variant, stage=row.role,
                                       lineage=row.lineage))
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _cache(context: Any) -> CampaignCache:
    """Return the in-process cache view shared by all stages of one run."""
    def create() -> CampaignCache:
        records = context.catalog.records.copy()
        return CampaignCache(context.settings, records.reset_index(drop=True))

    return context.resources.get_or_create("pretraining:cache", create)


def reference_task_ids(cache: CampaignCache) -> dict[str, str]:
    """Resolved reference (baseline) task of every axis, the first one in plan order.

    :param cache: Cache with the resolved plan.
    :type cache: CampaignCache
    :return: Task identifier per axis name.
    :rtype: dict[str, str]
    """
    tasks = cache.plan["tasks"]
    references: dict[str, str] = {}
    for task_id in cache.plan["parameters"]:
        references.setdefault(tasks[task_id].grid_parameters["axes"]["name"], task_id)
    return references


def _source_key(settings: dict, prefixes: tuple[str, ...]) -> str:
    """Digest of the package sources a cache level depends on."""
    return _source_digest(Path(settings["repository_root"]) / "src" / "msi_autoencoder_wrapper", prefixes)


def _atomic_json(path: Path, payload: dict) -> None:
    """Write JSON through a temporary file."""
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2, default=_json_default))
    temporary.replace(path)


# --------------------------------------------------
# Section: stage 1 — campaign plan
# --------------------------------------------------

def run_campaign_plan(context: Any) -> None:
    """Resolve (or reuse) the data contract of the reference tasks and verify every model.

    :param context: Precompute context of the common runner.
    :type context: msi_autoencoder_wrapper.analysis.precompute.core.context.AnalysisContext
    :raises ValueError: If the resolved contract disagrees with a saved artifact.
    """
    from . import pretraining_campaign_cache as contracts

    cache = _cache(context)
    settings = cache.settings
    tasks = campaign.campaign_tasks(settings)
    references = campaign.reference_tasks(settings, tasks)
    contract = contracts.plan_contract(settings, references)
    level = contracts.resolve_level(settings, "plan", contract,
                                    lambda directory: contracts.validate_legacy_plan(directory, references))
    manifest = level.directory / "resolved_parameters.json"
    if level.complete:
        logger.info("Reusing resolved campaign plan %s.", level.directory)
        parameters = json.loads(manifest.read_text())
    else:
        parameters = campaign.resolve_campaign_plan(references, level.directory)
    by_id = {task.task_id: task for task in tasks}
    cache.plan = {"directory": level.directory, "parameters": parameters,
                  "verification": campaign.verify_campaign_plan(settings, cache.models, parameters, by_id),
                  "tasks": by_id}
    cache.keys["plan"] = level.directory.name
    cache.contracts["plan"] = contracts.normalized(contract)


# --------------------------------------------------
# Section: provenance
# --------------------------------------------------

_PROVENANCE: dict[str, dict] = {}


def provenance_record(settings: dict) -> dict:
    """Commit, package versions and campaign YAML digest, computed once per process."""
    key = settings["settings_path"]
    if key not in _PROVENANCE:
        record = provenance(settings)
        _PROVENANCE[key] = {name: record[name] for name in ("git_commit", "versions", "experiment_sha256")}
    return _PROVENANCE[key]


# --------------------------------------------------
# Section: registered analyses
# --------------------------------------------------

#: Cache levels in dependency order and the level behind every ``requires`` entry.
_LEVEL_CHAIN = ("plan", "populations", "synthetic_reference", "inference")
_REQUIREMENT_LEVELS = {"campaign_plan": "plan", "populations": "populations",
                       "synthetic_reference": "synthetic_reference", "inference": "inference"}

#: Settings that never change an analysis result (runtime and selection bookkeeping).
_RUNTIME_SETTINGS = ("batch_size", "device", "workers", "settings_path", "repository_root", "precompute",
                     "analyses", "models", "groups")


def analysis_inputs(cache: CampaignCache, analysis: RegisteredAnalysis, axis: Optional[str],
                    stage: Optional[str]) -> dict:
    """Everything one analysis result depends on, recorded in its metadata.

    :param cache: Cache view of the analysis (its models are the analysed models).
    :type cache: CampaignCache
    :param analysis: Registered analysis.
    :type analysis: RegisteredAnalysis
    :param axis: Axis of per-axis analyses.
    :type axis: str | None
    :param stage: Stage of stage analyses.
    :type stage: str | None
    :return: Input record (compared on reuse).
    :rtype: dict
    """
    from .predictive_campaign import fingerprint

    settings = {key: value for key, value in cache.settings.items()
                if not key.startswith("_") and key not in _RUNTIME_SETTINGS}
    models = cache.models if axis is None else cache.models[cache.models.axis == axis]
    ## Cache levels the analysis reads, with every level upstream of them
    positions = [_LEVEL_CHAIN.index(_REQUIREMENT_LEVELS[name]) for name in analysis.requires]
    levels = _LEVEL_CHAIN[:max(positions) + 1] if positions else ()
    return {"analysis": analysis.name, "version": analysis.version,
            "function_sha256": analysis.source_sha256,
            "axis": axis, "stage": stage, "models": sorted(models.model_id),
            "cache": {level: fingerprint(cache.contracts[level])[:16] for level in levels if level in cache.contracts},
            "settings_sha256": fingerprint(settings)}


def _is_current(directory: Path, inputs: dict, cache: CampaignCache) -> bool:
    """Whether a result directory was produced from exactly these inputs.

    Results written before input records existed are current when they name the same
    models and the same (adopted) cache directories.
    """
    path = directory / "metadata.json"
    if not path.is_file():
        return False
    metadata = json.loads(path.read_text())
    if "inputs" in metadata:
        return metadata["inputs"] == json.loads(json.dumps(inputs, default=str))
    legacy_keys = {level: cache.keys.get(level) for level in ("plan", "populations", "inference")}
    return (sorted(metadata.get("models", [])) == inputs["models"]
            and {level: metadata.get("cache_keys", {}).get(level) for level in legacy_keys} == legacy_keys)


def run_analysis(context: Any, name: str) -> None:
    """Run one registered analysis for every configured axis (and stage) and write its tables.

    A result whose recorded inputs are unchanged is kept. With
    ``precompute.continue_on_error`` a failing axis/stage writes ``failure.json`` (with
    the traceback) into its result directory and the remaining ones still run.

    :param context: Precompute context of the common runner.
    :type context: msi_autoencoder_wrapper.analysis.precompute.core.context.AnalysisContext
    :param name: Registered analysis key.
    :type name: str
    """
    from . import pretraining_campaign_reports  # noqa: F401  (populates ANALYSES)
    from . import pretraining_campaign_stage_reports  # noqa: F401  (populates ANALYSES)

    analysis = ANALYSES[name]
    cache = _cache(context)
    continue_on_error = bool(cache.settings.get("precompute", {}).get("continue_on_error", False))
    # Model scope and the (axis, stage) runs of the analysis
    if analysis.scope == "baseline":
        scoped = cache.baseline_view()
        runs = [(scoped, axis, None) for axis in (scoped.axes if analysis.per_axis else [None])]
    elif analysis.scope == "all":
        runs = [(cache, axis, None) for axis in (cache.axes if analysis.per_axis else [None])]
    else:
        configured = cache.settings["analyses"][name].get("stages", {})
        runs = [(cache.stage_view(axis, stage), axis, stage) for axis in cache.axes for stage in cache.stages
                if stage in configured]
    for scoped, axis, stage in runs:
        directory = results_directory(cache.settings, name, axis, stage)
        inputs = analysis_inputs(scoped, analysis, axis, stage)
        label = "".join(f" on {value}" for value in (axis, stage) if value)
        if _is_current(directory, inputs, cache):
            logger.info("Analysis '%s'%s is current; keeping %s.", name, label, directory)
            continue
        logger.info("Running analysis '%s'%s.", name, label)
        try:
            tables = (analysis.function(scoped, axis, stage) if analysis.scope == "stage"
                      else analysis.function(scoped, axis))
        except Exception:
            if not continue_on_error:
                raise
            ## REMARK: analyses are independent leaves of the cache DAG; one failure must
            ## not cost the other results of an unattended run. The traceback is kept
            ## beside the (missing) tables and the error is logged.
            logger.error("Analysis '%s'%s failed; continuing.", name, label, exc_info=True)
            directory.mkdir(parents=True, exist_ok=True)
            _atomic_json(directory / "failure.json", {"inputs": inputs, "traceback": traceback.format_exc()})
            continue
        models = scoped.models if axis is None else scoped.models[scoped.models.axis == axis]
        metadata = {"analysis": name, "axis": axis, "stage": stage, "cache_keys": dict(cache.keys),
                    "models": models.model_id.tolist(), "inputs": inputs,
                    "provenance": provenance_record(cache.settings), **tables.pop("metadata", {})}
        (directory / "failure.json").unlink(missing_ok=True)
        write_results(directory, {**tables, "metadata": metadata})
