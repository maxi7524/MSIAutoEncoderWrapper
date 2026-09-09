"""Precompute for the contractive-penalty analyses, runnable outside a notebook.

The heavy part of these analyses is loading tens of trained models and running them over
a decoded split. That work is done here, once, and written to the analysis directory as
long-form tables; the notebooks read those tables and do nothing but derive cheap
summaries and draw figures. Splitting it this way is what makes a notebook re-runnable in
seconds and its numbers regenerable without opening it.

Two entry points, one implementation:

- ``python -m msi_autoencoder_wrapper.analysis.autoencoder.experiments.contractive_precompute
  --settings <analysis_settings.yaml> --analysis prediction_sweep`` for a background run;
- :func:`precompute` for the same thing from Python.

The counterpart is :func:`load_table`, which every notebook uses instead of computing.

This module mirrors the conventions of :mod:`.predictive_precompute` (settings file,
recorded provenance, one table per concern) rather than introducing a second precompute
style; what it adds is the command-line entry point, so a long run can be detached from
the notebook kernel.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import numpy as np
import pandas as pd
import torch
import yaml

from ....utils.logger import get_custom_logger
from ..heads.metrics import probabilities_from_logits
from . import penalty_sweep as sweep
from . import sweep_evaluation as evaluation
from .entropy_status_reader import read_entropy_campaign

logger = get_custom_logger(__name__)

#: Routines this module can run, populated by :func:`_routine`.
ROUTINES: Dict[str, Callable[[dict], Dict[str, Any]]] = {}


def _routine(name: str) -> Callable:
    """Register one named precompute routine."""

    def register(function: Callable[[dict], Dict[str, Any]]) -> Callable:
        ROUTINES[name] = function
        return function

    return register


# --------------------------------------------------
# Section: settings, devices and provenance
# --------------------------------------------------

def read_settings(path: Path | str) -> dict:
    """Load an ``analysis_settings.yaml`` and resolve its paths.

    :param path: Settings file, relative to the repository root or absolute.
    :type path: pathlib.Path | str
    :return: Settings with path-valued entries resolved to :class:`~pathlib.Path`.
    :rtype: dict
    :raises FileNotFoundError: If the settings file does not exist.
    """
    settings_path = Path(path)
    if not settings_path.is_file():
        raise FileNotFoundError(f"No analysis settings at '{settings_path}'.")
    settings = yaml.safe_load(settings_path.read_text(encoding="utf-8")) or {}
    for key in ("workspace", "model_store", "decode_cache", "analysis_directory"):
        if settings.get(key):
            settings[key] = Path(settings[key])
    settings["settings_path"] = settings_path
    return settings


def resolve_device(settings: dict, allow_cpu: bool = False) -> torch.device:
    """Resolve the configured device, refusing a silent fall back to the processor.

    REMARK: an earlier run of these analyses spent hours on the processor because the
    environment had been re-synced to a CPU build of torch and nothing checked. The
    configured device is therefore enforced rather than treated as a preference.

    :param settings: Analysis settings carrying a ``device`` entry.
    :type settings: dict
    :param allow_cpu: Permit running on the processor even when ``cuda`` is configured.
    :type allow_cpu: bool
    :return: The device to compute on.
    :rtype: torch.device
    :raises RuntimeError: If ``cuda`` is configured but unavailable and ``allow_cpu``
        was not requested.
    """
    requested = str(settings.get("device", "cuda"))
    if requested.startswith("cuda") and not torch.cuda.is_available():
        if not allow_cpu:
            raise RuntimeError(
                f"Settings request device '{requested}' but this interpreter has "
                f"torch {torch.__version__} without CUDA. Install the GPU extra "
                "(`uv sync --extra cu118`) or pass --allow-cpu to accept a slow run."
            )
        logger.warning("CUDA unavailable; continuing on the processor by request.")
        return torch.device("cpu")
    return torch.device(requested)


def provenance(settings: dict, extra: Optional[dict] = None) -> dict:
    """Record what produced a set of tables.

    :param settings: Analysis settings.
    :type settings: dict
    :param extra: Routine-specific entries to merge in.
    :type extra: dict | None
    :return: Provenance mapping, JSON-serializable.
    :rtype: dict
    """
    sources = {
        name: hashlib.sha256(Path(path).read_bytes()).hexdigest()
        for name, path in (
            ("sweep_evaluation", "src/msi_autoencoder_wrapper/analysis/autoencoder/experiments/sweep_evaluation.py"),
            ("penalty_sweep", "src/msi_autoencoder_wrapper/analysis/autoencoder/experiments/penalty_sweep.py"),
            ("contractive_precompute", __file__),
        )
        if Path(path).is_file()
    }
    record = {
        "settings_path": str(settings.get("settings_path", "")),
        "campaigns": settings.get("campaigns", {}),
        "device": str(settings.get("device")),
        "sample_seed": settings.get("sample_seed"),
        "pixel_fraction": settings.get("pixel_fraction"),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "source_sha256": sources,
    }
    try:
        record["git_commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):  # pragma: no cover - not a repository
        record["git_commit"] = None
    return {**record, **(extra or {})}


# --------------------------------------------------
# Section: shared campaign and split preparation
# --------------------------------------------------

def campaign_grid(settings: dict, names: list[str]) -> pd.DataFrame:
    """Read the named campaigns' completed runs into one grid frame.

    :param settings: Analysis settings with ``workspace``, ``model_store`` and a
        ``campaigns`` mapping from short name to campaign identifier.
    :type settings: dict
    :param names: Campaign short names to include, in order.
    :type names: list[str]
    :return: One row per completed run, with its grid-cell identity.
    :rtype: pandas.DataFrame
    :raises KeyError: If a requested campaign is absent from the settings.
    """
    rows = []
    for name in names:
        if name not in settings["campaigns"]:
            raise KeyError(f"Campaign '{name}' is not defined in the analysis settings.")
        campaign_id = settings["campaigns"][name]
        status = Path(settings["workspace"]) / "configs" / "entropy-runs" / campaign_id / "plan" / "status"
        tasks = read_entropy_campaign(status, Path(settings["model_store"]), load_artifacts=False)
        rows += [row for row in sweep.sweep_grid_frame(tasks, campaign_id) if row["status"] == "completed"]
    frame = pd.DataFrame(rows)
    logger.info("Resolved %s completed run(s) across %s cell(s).", len(frame), frame["cell_label"].nunique())
    return frame


def prepare_context(settings: dict, grid_frame: pd.DataFrame):
    """Establish the dataset context and materialize the requested splits once.

    Saved model configurations record the absolute reader paths of the compute node that
    trained them, so those paths are linked to the local workspace before the single
    ``load_configuration`` call that needs them; see
    :func:`~.sweep_evaluation.link_campaign_scratch_workspace`.

    :param settings: Analysis settings.
    :type settings: dict
    :param grid_frame: Runs to analyse; its first row supplies the dataset context.
    :type grid_frame: pandas.DataFrame
    :return: ``(wrapper, splits)`` with one materialized split per configured name.
    :rtype: tuple
    """
    from .... import MSIAutoEncoderWrapper

    workspace = Path(settings["workspace"])
    scratch_root = settings.get("scratch_workspace_root")
    if scratch_root:
        for campaign_id in set(grid_frame["campaign"]):
            evaluation.link_campaign_scratch_workspace(f"{scratch_root}/{campaign_id}/workspace", workspace)

    wrapper = MSIAutoEncoderWrapper(project_path=str(workspace))
    wrapper.workspace.set_default_image_path(settings["model_context"])
    wrapper.load_configuration(model_name=grid_frame.iloc[0]["model_name"])
    partitions = wrapper.active_dataset.create_partitions()

    splits = {
        name: evaluation.materialize_split(
            partitions, name, settings["target_field"],
            fraction=float(settings.get("pixel_fraction", 1.0)),
            seed=int(settings.get("sample_seed", 42)),
            cache_directory=settings.get("decode_cache"),
        )
        for name in settings.get("splits", ["train", "test"])
    }
    for name, split in splits.items():
        logger.info("Split '%s': %s/%s pixel(s).", name, split.sampled, split.total)
    return wrapper, splits


# --------------------------------------------------
# Section: routines
# --------------------------------------------------

@_routine("prediction_sweep")
def prediction_sweep(settings: dict) -> Dict[str, Any]:
    """Head metrics for every run of the pooled sweep, on both splits.

    Produces exactly the tables ``part_1`` consumes: long-form prediction metrics in
    both class scopes, per-class metrics for the first repetition of each cell, the
    calibration diagnostic, the grid, the decoded sample indices and the fair-scope
    class selection.

    :param settings: Analysis settings for this routine.
    :type settings: dict
    :return: Tables keyed by file name, plus ``metadata``.
    :rtype: Dict[str, Any]
    """
    grid_frame = campaign_grid(settings, settings["campaigns_used"])
    wrapper, splits = prepare_context(settings, grid_frame)
    head = settings["head"]
    threshold = float(settings.get("threshold", 0.5))
    batch_size = int(settings.get("batch_size", 256))

    fair_mask = evaluation.fair_scope_mask(splits["train"], splits["test"])
    scopes = {evaluation.ALL_CLASSES_SCOPE: None, evaluation.FAIR_SCOPE: fair_mask}

    def load_model(model_name: str):
        return wrapper.models_manager.load_model(
            img_name=settings["model_context"], model_name=model_name, strict=True
        )

    ## Head metrics for every run, both splits, both scopes
    evaluated = evaluation.evaluate_campaign_models(
        grid_frame.to_dict("records"), splits, load_model,
        head_name=head, scopes=scopes, threshold=threshold, batch_size=batch_size,
        collect_per_class=False, collect_geometry=False,
    )
    prediction_frame = pd.DataFrame(evaluated["prediction"])

    ## Per-class metrics and the calibration diagnostic, first repetition of each cell
    per_class_rows, spread_rows = [], []
    for record in grid_frame.query("repetition == 0").to_dict("records"):
        model = load_model(record["model_name"])
        model.eval()
        logits = evaluation.head_logits(model, splits["test"].spectra, head, batch_size=batch_size)

        cell = sweep.PenaltySweepCell(
            penalty_metric=record["penalty_metric"], input_geometry=record["input_geometry"],
            weight=record["weight"], hinge_threshold=record["hinge_threshold"],
            hinge_alpha=record["hinge_alpha"], penalized_space=record["penalized_space"],
            calculation_method=record["calculation_method"],
        )
        per_class_rows += evaluation.per_class_metrics_frame(
            logits, splits["test"], cell, model_name=record["model_name"],
            campaign=record["campaign"], repetition=record["repetition"],
            class_mask=fair_mask, threshold=threshold,
        )

        probabilities = probabilities_from_logits(logits[:, fair_mask], "multi_label")
        available = splits["test"].mask[:, fair_mask].astype(bool)
        values = probabilities[available]
        spread_rows.append({
            "cell_label": record["cell_label"],
            "mean_probability": float(values.mean()),
            "std_probability": float(values.std()),
            "mean_logit_std": float(logits[:, fair_mask].std(axis=0).mean()),
        })

    positive_rate = float(
        (splits["test"].targets[:, fair_mask] * splits["test"].mask[:, fair_mask]).sum()
        / splits["test"].mask[:, fair_mask].sum()
    )

    return {
        "prediction_metrics": prediction_frame,
        "per_class_metrics": pd.DataFrame(per_class_rows),
        "calibration_diagnostic": pd.DataFrame(spread_rows),
        "grid": grid_frame,
        "sample_indices": pd.DataFrame({
            "split": np.repeat(list(splits), [split.sampled for split in splits.values()]),
            "dataset_index": np.concatenate([split.indices for split in splits.values()]),
        }),
        "fair_scope_classes": pd.DataFrame({"class_index": np.flatnonzero(fair_mask)}),
        "metadata": provenance(settings, {
            "analysis": "prediction_sweep",
            "head": head,
            "target_field": settings["target_field"],
            "threshold": threshold,
            "splits": {name: {"sampled": split.sampled, "total": split.total} for name, split in splits.items()},
            "classes_total": int(fair_mask.size),
            "classes_fair_scope": int(fair_mask.sum()),
            "empirical_positive_rate_fair_scope": positive_rate,
            "models_evaluated": int(len(grid_frame)),
        }),
    }


# --------------------------------------------------
# Section: running and loading
# --------------------------------------------------

def analysis_settings(settings: dict, analysis: str) -> dict:
    """Merge the shared settings with one analysis's own overrides.

    :param settings: Parsed settings file.
    :type settings: dict
    :param analysis: Registered routine name.
    :type analysis: str
    :return: Effective settings for that routine, including ``output_directory``.
    :rtype: dict
    :raises KeyError: If the analysis is not configured or not registered.
    """
    if analysis not in ROUTINES:
        raise KeyError(f"Unknown analysis '{analysis}'; registered: {sorted(ROUTINES)}.")
    configured = (settings.get("analyses") or {}).get(analysis)
    if configured is None:
        raise KeyError(f"Analysis '{analysis}' is not configured in the settings file.")
    merged = {key: value for key, value in settings.items() if key != "analyses"}
    merged.update(configured)
    merged["output_directory"] = Path(configured["output_directory"])
    return merged


def precompute(settings: dict, analysis: str, *, allow_cpu: bool = False) -> Path:
    """Run one routine and write its tables next to the notebook that reads them.

    :param settings: Parsed settings file, as returned by :func:`read_settings`.
    :type settings: dict
    :param analysis: Registered routine name.
    :type analysis: str
    :param allow_cpu: Permit a processor run when ``cuda`` is configured.
    :type allow_cpu: bool
    :return: The directory the tables were written to.
    :rtype: pathlib.Path
    """
    effective = analysis_settings(settings, analysis)
    device = resolve_device(effective, allow_cpu=allow_cpu)
    logger.info("Running '%s' on %s.", analysis, device)

    output = effective["output_directory"]
    output.mkdir(parents=True, exist_ok=True)
    produced = ROUTINES[analysis](effective)

    for name, value in produced.items():
        if name == "metadata":
            (output / "metadata.json").write_text(json.dumps(value, indent=2) + "\n")
        else:
            value.to_csv(output / f"{name}.csv", index=False)
    logger.info("Wrote %s artifact(s) to %s", len(produced), output)
    return output


def load_table(settings: dict, analysis: str, table: str) -> pd.DataFrame:
    """Read one precomputed table.

    :param settings: Parsed settings file.
    :type settings: dict
    :param analysis: Registered routine name.
    :type analysis: str
    :param table: Table name without its extension.
    :type table: str
    :return: The stored table.
    :rtype: pandas.DataFrame
    :raises FileNotFoundError: If the table has not been produced yet, with the command
        that produces it.
    """
    effective = analysis_settings(settings, analysis)
    path = effective["output_directory"] / f"{table}.csv"
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing '{path}'. Produce it first:\n  {run_command(settings, analysis)}"
        )
    return pd.read_csv(path)


def load_metadata(settings: dict, analysis: str) -> dict:
    """Read a routine's recorded provenance.

    :param settings: Parsed settings file.
    :type settings: dict
    :param analysis: Registered routine name.
    :type analysis: str
    :return: The provenance mapping.
    :rtype: dict
    :raises FileNotFoundError: If the routine has not been run yet.
    """
    effective = analysis_settings(settings, analysis)
    path = effective["output_directory"] / "metadata.json"
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing '{path}'. Produce it first:\n  {run_command(settings, analysis)}"
        )
    return json.loads(path.read_text())


def run_command(settings: dict, analysis: str, *, background: bool = True) -> str:
    """Return the shell command that produces one analysis's tables.

    Kept here rather than written out in each notebook so the instruction cannot drift
    from the entry point it describes.

    :param settings: Parsed settings file.
    :type settings: dict
    :param analysis: Registered routine name.
    :type analysis: str
    :param background: Wrap the command in ``nohup`` and detach it, so a long run
        survives the notebook kernel.
    :type background: bool
    :return: A copy-pasteable shell command.
    :rtype: str
    """
    settings_path = settings.get("settings_path", "analysis_settings.yaml")
    module = "msi_autoencoder_wrapper.analysis.autoencoder.experiments.contractive_precompute"
    command = f"python -m {module} --settings {settings_path} --analysis {analysis}"
    if not background:
        return command
    log = Path(analysis_settings(settings, analysis)["output_directory"]) / f"{analysis}.log"
    return f"nohup {command} > {log} 2>&1 &"


def main(argv: Optional[list[str]] = None) -> int:
    """Command-line entry point, so a long run can be detached from a notebook."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--settings", required=True, help="Path to analysis_settings.yaml.")
    parser.add_argument("--analysis", required=True, choices=sorted(ROUTINES), help="Routine to run.")
    parser.add_argument("--allow-cpu", action="store_true", help="Accept a processor run.")
    arguments = parser.parse_args(argv)

    settings = read_settings(arguments.settings)
    output = precompute(settings, arguments.analysis, allow_cpu=arguments.allow_cpu)
    print(f"wrote {arguments.analysis} tables to {output}")
    return 0




@_routine("latent_geometry_sweep")
def latent_geometry_sweep(settings: dict) -> Dict[str, Any]:
    """Latent geometry, structure, stability, sensitivity and RSA for the pooled sweep.

    One encoding per model feeds every statistic. Splitting these into a loop per
    statistic family, as this analysis was first written, loads and encodes each model
    four times over.

    :param settings: Analysis settings for this routine.
    :type settings: dict
    :return: Tables keyed by file name, plus ``metadata``.
    :rtype: Dict[str, Any]
    """
    from ..latent import sphere_geometry as geometry

    grid_frame = campaign_grid(settings, settings["campaigns_used"])
    wrapper, splits = prepare_context(settings, grid_frame)
    test_split = splits["test"]
    batch_size = int(settings.get("batch_size", 256))
    seed = int(settings.get("sample_seed", 42))
    pair_count = int(settings.get("pair_count", 5000))
    neighbours = int(settings.get("knn_neighbors", 10))
    epsilons = tuple(settings.get("epsilons", (0.001, 0.003, 0.01, 0.03, 0.1)))

    def load_model(model_name: str):
        return wrapper.models_manager.load_model(
            img_name=settings["model_context"], model_name=model_name, strict=True
        )

    sensitivity_rng = np.random.default_rng(seed + 3)
    rsa_rng = np.random.default_rng(seed + 4)
    structure_rng = np.random.default_rng(seed + 1)
    rsa_labels = test_split.targets * test_split.mask
    sensitivity_inputs = test_split.spectra.numpy()

    geometry_rows, structure_rows, sensitivity_rows, rsa_rows, cos_rows = [], [], [], [], []
    latent_by_cell: Dict[tuple, Dict[int, np.ndarray]] = {}
    cell_by_label, campaign_by_label = {}, {}

    # One pass over every model
    for position, record in enumerate(grid_frame.to_dict("records"), start=1):
        model = load_model(record["model_name"])
        model.eval()
        gamma, beta = geometry.encoder_layer_norm_parameters(model)
        cell = sweep.PenaltySweepCell(
            penalty_metric=record["penalty_metric"], input_geometry=record["input_geometry"],
            weight=record["weight"], hinge_threshold=record["hinge_threshold"],
            hinge_alpha=record["hinge_alpha"], penalized_space=record["penalized_space"],
            calculation_method=record["calculation_method"],
        )
        cell_by_label.setdefault(cell.label, cell)
        campaign_by_label.setdefault(cell.label, record["campaign"])

        ## Single encoding of the shared sample, reused by every statistic below
        latent = evaluation.canonical_latent(model, test_split.spectra, batch_size=batch_size)

        geometry_rows += evaluation.latent_geometry_frame(
            latent, cell, split_name="test", model_name=record["model_name"],
            campaign=record["campaign"], repetition=record["repetition"],
        )
        if record["repetition"] is not None:
            latent_by_cell.setdefault((cell.label, "test"), {})[int(record["repetition"])] = latent

        ## Angular structure, on the first repetition of each cell
        if record["repetition"] == 0:
            result = geometry.structure_test(latent, structure_rng, pair_count=pair_count, return_samples=True)
            samples = np.asarray(result["cos_theta_samples"])
            cos_rows.append(pd.DataFrame({"cell_label": cell.label, "cos_theta": samples}))
            structure_rows.append({
                "cell_label": cell.label,
                **{key: value for key, value in result.items() if key != "cos_theta_samples"},
            })

        ## Encoder sensitivity to isotropic input perturbation
        def encode(batch: np.ndarray, _model=model, _gamma=gamma, _beta=beta) -> np.ndarray:
            with torch.no_grad():
                device = next(_model.parameters()).device
                codes = _model(torch.as_tensor(batch, dtype=torch.float32).to(device))["latent_space"]
            return geometry.canonicalize(codes.cpu().numpy(), _gamma, _beta)

        curve = geometry.angular_sensitivity_curve(encode, sensitivity_inputs, epsilons, sensitivity_rng)
        for epsilon, angle in zip(curve["epsilon"], curve["mean_angle_degrees"]):
            sensitivity_rows.append({
                "cell_label": cell.label, "campaign": record["campaign"],
                "penalty_metric": record["penalty_metric"], "input_geometry": record["input_geometry"],
                "weight": record["weight"], "repetition": record["repetition"],
                "epsilon": float(epsilon), "mean_angle_degrees": float(angle),
            })

        ## Agreement between latent similarity and label similarity
        rsa_rows.append({
            "cell_label": cell.label, "campaign": record["campaign"],
            "penalty_metric": record["penalty_metric"], "input_geometry": record["input_geometry"],
            "weight": record["weight"], "repetition": record["repetition"],
            **geometry.rsa_spearman(latent, rsa_labels, rsa_rng, pair_count=pair_count),
        })

        if position % 20 == 0 or position == len(grid_frame):
            logger.info("Encoded and summarized %s/%s model(s).", position, len(grid_frame))

    ## Cross-repetition agreement, once every cell's encodings are in hand
    stability_rows = []
    for (cell_label, split_name), latents in latent_by_cell.items():
        if len(latents) < 2:
            continue
        stability_rows += evaluation.representation_stability_frame(
            latents, cell_by_label[cell_label], split_name=split_name,
            campaign=campaign_by_label[cell_label], knn_neighbors=neighbours,
        )

    return {
        "latent_geometry": pd.DataFrame(geometry_rows),
        "representation_stability": pd.DataFrame(stability_rows),
        "angular_sensitivity": pd.DataFrame(sensitivity_rows),
        "angular_structure": pd.DataFrame(structure_rows),
        "angular_structure_samples": pd.concat(cos_rows, ignore_index=True) if cos_rows else pd.DataFrame(),
        "representational_similarity": pd.DataFrame(rsa_rows),
        "grid": grid_frame,
        "sample_indices": pd.DataFrame({"dataset_index": test_split.indices}),
        "metadata": provenance(settings, {
            "analysis": "latent_geometry_sweep",
            "pair_count": pair_count,
            "knn_neighbors": neighbours,
            "epsilons": list(epsilons),
            "test_sampled": test_split.sampled,
            "test_total": test_split.total,
            "models_evaluated": int(len(grid_frame)),
        }),
    }


@_routine("hinge_refinement")
def hinge_refinement(settings: dict) -> Dict[str, Any]:
    """Prediction, geometry, stability, sensitivity and training health for selected cells.

    Restricted to the cells named in the settings, which is what makes this analysis
    readable where the pooled sweep is not: nine cells rather than twenty-eight.

    :param settings: Analysis settings for this routine.
    :type settings: dict
    :return: Tables keyed by file name, plus ``metadata``.
    :rtype: Dict[str, Any]
    """
    from ..latent import sphere_geometry as geometry

    selected = list(settings["selected_cells"])
    batch_size = int(settings.get("batch_size", 256))
    seed = int(settings.get("sample_seed", 42))
    threshold = float(settings.get("threshold", 0.5))
    neighbours = int(settings.get("knn_neighbors", 10))
    epsilons = tuple(settings.get("epsilons", (0.001, 0.003, 0.01, 0.03, 0.1)))
    planned_epochs = int(settings.get("planned_epochs", 15))

    ## Every task of every campaign, with history: the health and cost checks need the
    ## incomplete and failed runs too, so this is deliberately not filtered to completed.
    all_grid_rows, dynamics_rows = [], []
    for name in settings["campaigns_used"]:
        campaign_id = settings["campaigns"][name]
        status = Path(settings["workspace"]) / "configs" / "entropy-runs" / campaign_id / "plan" / "status"
        tasks = read_entropy_campaign(status, Path(settings["model_store"]), load_artifacts=True)
        all_grid_rows += sweep.sweep_grid_frame(tasks, campaign_id)
        dynamics_rows += sweep.training_dynamics_frame(tasks, campaign_id)

    grid_frame = pd.DataFrame(
        [row for row in all_grid_rows if row["status"] == "completed" and row["cell_label"] in selected]
    )
    missing = sorted(set(selected) - set(grid_frame["cell_label"]))
    if missing:
        raise KeyError(f"Selected cells absent from the campaigns: {missing}")

    health_frame = pd.DataFrame(
        sweep.training_health_report(dynamics_rows, all_grid_rows, planned_epochs=planned_epochs)
    )
    run_frame = pd.DataFrame(sweep.run_duration_frame(dynamics_rows))

    wrapper, splits = prepare_context(settings, grid_frame)
    fair_mask = evaluation.fair_scope_mask(splits["train"], splits["test"])
    scopes = {evaluation.ALL_CLASSES_SCOPE: None, evaluation.FAIR_SCOPE: fair_mask}

    def load_model(model_name: str):
        return wrapper.models_manager.load_model(
            img_name=settings["model_context"], model_name=model_name, strict=True
        )

    ## Two of the latent statistics build an (N, N) matrix, so the row count is capped at
    ## the evaluation split's size; the larger training split is summarized on a
    ## matched-size subsample drawn once and shared by every model.
    geometry_sample = splits["test"].sampled

    evaluated = evaluation.evaluate_campaign_models(
        grid_frame.to_dict("records"), splits, load_model,
        head_name=settings["head"], scopes=scopes, threshold=threshold, batch_size=batch_size,
        collect_per_class=False, collect_geometry=True,
        geometry_sample_size=geometry_sample, geometry_seed=seed,
    )
    latent_by_cell = evaluated["latent_by_cell"]

    ## Cross-repetition agreement on the evaluation split
    cell_by_label, campaign_by_label = {}, {}
    for record in grid_frame.to_dict("records"):
        label = record["cell_label"]
        if label in cell_by_label:
            continue
        cell_by_label[label] = sweep.PenaltySweepCell(
            penalty_metric=record["penalty_metric"], input_geometry=record["input_geometry"],
            weight=record["weight"], hinge_threshold=record["hinge_threshold"],
            hinge_alpha=record["hinge_alpha"], penalized_space=record["penalized_space"],
            calculation_method=record["calculation_method"],
        )
        campaign_by_label[label] = record["campaign"]

    stability_rows = []
    for (cell_label, split_name), latents in latent_by_cell.items():
        if len(latents) < 2 or split_name != "test":
            continue
        stability_rows += evaluation.representation_stability_frame(
            latents, cell_by_label[cell_label], split_name=split_name,
            campaign=campaign_by_label[cell_label], knn_neighbors=neighbours,
        )

    ## Encoder sensitivity to isotropic perturbation, every repetition
    sensitivity_rng = np.random.default_rng(seed + 3)
    sensitivity_inputs = splits["test"].spectra.numpy()
    sensitivity_rows = []
    for record in grid_frame.to_dict("records"):
        model = load_model(record["model_name"])
        model.eval()
        gamma, beta = geometry.encoder_layer_norm_parameters(model)

        def encode(batch: np.ndarray, _model=model, _gamma=gamma, _beta=beta) -> np.ndarray:
            with torch.no_grad():
                device = next(_model.parameters()).device
                latent = _model(torch.as_tensor(batch, dtype=torch.float32).to(device))["latent_space"]
            return geometry.canonicalize(latent.cpu().numpy(), _gamma, _beta)

        curve = geometry.angular_sensitivity_curve(encode, sensitivity_inputs, epsilons, sensitivity_rng)
        for epsilon, angle in zip(curve["epsilon"], curve["mean_angle_degrees"]):
            sensitivity_rows.append({
                "cell_label": record["cell_label"], "campaign": record["campaign"],
                "weight": record["weight"], "repetition": record["repetition"],
                "epsilon": float(epsilon), "mean_angle_degrees": float(angle),
            })

    return {
        "prediction_metrics": pd.DataFrame(evaluated["prediction"]),
        "latent_geometry": pd.DataFrame(evaluated["geometry"]),
        "representation_stability": pd.DataFrame(stability_rows),
        "angular_sensitivity": pd.DataFrame(sensitivity_rows),
        "training_health": health_frame,
        "run_durations": run_frame,
        "grid": grid_frame,
        "sample_indices": pd.DataFrame({
            "split": np.repeat(list(splits), [split.sampled for split in splits.values()]),
            "dataset_index": np.concatenate([split.indices for split in splits.values()]),
        }),
        "fair_scope_classes": pd.DataFrame({"class_index": np.flatnonzero(fair_mask)}),
        "metadata": provenance(settings, {
            "analysis": "hinge_refinement",
            "selected_cells": selected,
            "threshold": threshold,
            "knn_neighbors": neighbours,
            "epsilons": list(epsilons),
            "planned_epochs": planned_epochs,
            "geometry_sample_size": int(geometry_sample),
            "splits": {name: {"sampled": split.sampled, "total": split.total} for name, split in splits.items()},
            "classes_fair_scope": int(fair_mask.sum()),
            "classes_total": int(fair_mask.size),
            "models_evaluated": int(len(grid_frame)),
        }),
    }


@_routine("campaign_training_dynamics")
def campaign_training_dynamics(settings: dict) -> Dict[str, Any]:
    """Grid coverage, per-epoch objectives and post-training evaluation of every campaign.

    No model is loaded: everything comes from the campaigns' recorded training histories.
    It is nevertheless a precompute rather than notebook code, because reading and
    flattening every history is the one step that scales with the campaign size.

    :param settings: Analysis settings for this routine.
    :type settings: dict
    :return: Tables keyed by file name, plus ``metadata``.
    :rtype: Dict[str, Any]
    """
    grid_rows, dynamics_rows, evaluation_rows = [], [], []
    for name in settings["campaigns_used"]:
        campaign_id = settings["campaigns"][name]
        status = Path(settings["workspace"]) / "configs" / "entropy-runs" / campaign_id / "plan" / "status"
        tasks = read_entropy_campaign(status, Path(settings["model_store"]), load_artifacts=True)
        grid_rows += sweep.sweep_grid_frame(tasks, campaign_id)
        dynamics_rows += sweep.training_dynamics_frame(tasks, campaign_id)
        evaluation_rows += sweep.final_evaluation_frame(tasks, campaign_id)

    planned_epochs = int(settings.get("planned_epochs", 15))
    return {
        "grid_coverage": pd.DataFrame(grid_rows),
        "training_dynamics": pd.DataFrame(dynamics_rows),
        "final_evaluation": pd.DataFrame(evaluation_rows),
        "training_health": pd.DataFrame(
            sweep.training_health_report(dynamics_rows, grid_rows, planned_epochs=planned_epochs)
        ),
        "run_durations": pd.DataFrame(sweep.run_duration_frame(dynamics_rows)),
        "metadata": provenance(settings, {
            "analysis": "campaign_training_dynamics",
            "planned_epochs": planned_epochs,
            "tasks": len(grid_rows),
            "epoch_rows": len(dynamics_rows),
        }),
    }


@_routine("perturbation_anatomy")
def perturbation_anatomy(settings: dict) -> Dict[str, Any]:
    """What each perturbation does to real spectra, measured under four input metrics.

    No trained model is evaluated; a saved configuration is loaded only to resolve the
    dataset. The spectra the figures draw are persisted alongside the numerical tables,
    because a figure of a spectrum cannot be redrawn from summary statistics.

    :param settings: Analysis settings for this routine.
    :type settings: dict
    :return: Tables keyed by file name, plus ``metadata``.
    :rtype: Dict[str, Any]
    """
    from ..latent import perturbations as perturbation

    context_cell = settings["context_cell"]
    seed = int(settings.get("sample_seed", 42))
    pixels = int(settings.get("anatomy_pixels", 2000))
    shown = int(settings.get("shown_spectra", 3))
    perturbation_settings = perturbation.PerturbationSettings(**settings.get("perturbation", {}))

    grid_rows = []
    for name in settings["campaigns_used"]:
        campaign_id = settings["campaigns"][name]
        status = Path(settings["workspace"]) / "configs" / "entropy-runs" / campaign_id / "plan" / "status"
        tasks = read_entropy_campaign(status, Path(settings["model_store"]), load_artifacts=False)
        grid_rows += [
            row for row in sweep.sweep_grid_frame(tasks, campaign_id)
            if row["status"] == "completed" and row["cell_label"] == context_cell and row["repetition"] == 0
        ]
    grid_frame = pd.DataFrame(grid_rows)

    wrapper, splits = prepare_context(settings, grid_frame)
    test_split = splits["test"]
    device = resolve_device(settings, allow_cpu=True)

    generator = np.random.default_rng(seed)
    selection = (
        np.sort(generator.choice(test_split.sampled, pixels, replace=False))
        if test_split.sampled > pixels else np.arange(test_split.sampled)
    )
    clean = test_split.spectra[selection].to(device)  # (N, M)

    # The binner's own grid, not a reconstructed one.
    mass_axis = np.asarray(wrapper.active_context.binner.GetXAxis(), dtype=float)

    perturbation_generator = torch.Generator(device="cpu").manual_seed(seed + 400)
    applied = {
        name: perturbation.perturb_spectra(
            clean, name, settings=perturbation_settings, generator=perturbation_generator,
        )
        for name in perturbation.PERTURBATION_NAMES
    }

    ## Primal lengths of each perturbation under every candidate metric
    norm_rows = []
    for name, result in applied.items():
        norms = perturbation.perturbation_norms(
            result.spectra - clean, clean,
            gaussian_sigma_bins=perturbation_settings.base_width_sigma,
            bin_spacing=float(np.median(np.diff(mass_axis))),
        )
        for index in range(clean.shape[0]):
            norm_rows.append({
                "perturbation": name, "row": index,
                "dataset_index": int(test_split.indices[selection][index]),
                **{key: float(value[index]) for key, value in norms.items()},
            })

    ## The spectra the figures draw, chosen deterministically by detected structure
    peak_counts = (clean > clean.mean(dim=1, keepdim=True)).sum(dim=1).cpu().numpy()
    shown_rows = np.argsort(-peak_counts)[:shown]
    spectra_rows = []
    for row in shown_rows:
        row = int(row)
        series = {"clean": clean[row].cpu().numpy()}
        series.update({name: result.spectra[row].cpu().numpy() for name, result in applied.items()})
        for label, values in series.items():
            spectra_rows.append(pd.DataFrame({
                "row": row,
                "dataset_index": int(test_split.indices[selection][row]),
                "series": label,
                "mz": mass_axis,
                "intensity": values,
            }))

    return {
        "perturbation_norms": pd.DataFrame(norm_rows),
        "perturbation_diagnostics": pd.DataFrame(perturbation.perturbation_diagnostics_frame(applied)),
        "displayed_spectra": pd.concat(spectra_rows, ignore_index=True),
        "analysed_spectra": pd.DataFrame({"dataset_index": test_split.indices[selection]}),
        "grid": grid_frame,
        "metadata": provenance(settings, {
            "analysis": "perturbation_anatomy",
            "context_cell": context_cell,
            "perturbations": list(perturbation.PERTURBATION_NAMES),
            "perturbation_settings": settings.get("perturbation", {}),
            "analysed_spectra": int(clean.shape[0]),
            "displayed_rows": [int(row) for row in shown_rows],
            "displayed_dataset_indices": [int(test_split.indices[selection][int(row)]) for row in shown_rows],
            "mass_axis_range": [float(mass_axis[0]), float(mass_axis[-1])],
        }),
    }


@_routine("perturbation_response")
def perturbation_response(settings: dict) -> Dict[str, Any]:
    """Angular response of every selected model to four chemical perturbation families.

    The full-strength response and the amplitude sweep share one model load and one
    clean encoding per model. The notebook version walked the model set twice, once per
    figure, which doubled the loading cost for no additional measurement.

    :param settings: Analysis settings for this routine.
    :type settings: dict
    :return: Tables keyed by file name, plus ``metadata``.
    :rtype: Dict[str, Any]
    """
    from ..latent import perturbations as perturbation
    from ..latent import sphere_geometry as geometry
    from ..latent.sensitivity import canonical_direction, paired_direction_angles

    grid_frame = campaign_grid(settings, settings["campaigns_used"])
    selected = settings.get("selected_cells")
    if selected:
        grid_frame = grid_frame[grid_frame["cell_label"].isin(selected)].reset_index(drop=True)
        missing = sorted(set(selected) - set(grid_frame["cell_label"]))
        if missing:
            raise KeyError(f"Selected cells absent from the campaigns: {missing}")

    wrapper, splits = prepare_context(settings, grid_frame)
    test_split = splits["test"]
    device = resolve_device(settings, allow_cpu=True)
    batch_size = int(settings.get("batch_size", 256))
    seed = int(settings.get("sample_seed", 42))
    repeats_requested = int(settings.get("noise_repeats", 4))
    amplitudes = tuple(settings.get("amplitudes", (0.0, 0.125, 0.25, 0.5, 0.75, 1.0)))
    response_pixels = int(settings.get("response_pixels", 2000))
    perturbation_settings = perturbation.PerturbationSettings(**settings.get("perturbation", {}))

    generator = np.random.default_rng(seed)
    selection = (
        np.sort(generator.choice(test_split.sampled, response_pixels, replace=False))
        if test_split.sampled > response_pixels else np.arange(test_split.sampled)
    )
    clean_spectra = test_split.spectra[selection].to(device)  # (N, M)

    def load_model(model_name: str):
        return wrapper.models_manager.load_model(
            img_name=settings["model_context"], model_name=model_name, strict=True
        )

    def canonical_directions(model, spectra: torch.Tensor) -> torch.Tensor:
        gamma, beta = geometry.encoder_layer_norm_parameters(model)
        gamma_tensor = torch.as_tensor(gamma, device=spectra.device)
        beta_tensor = torch.as_tensor(beta, device=spectra.device)
        directions = []
        with torch.no_grad():
            for start in range(0, len(spectra), batch_size):
                latent = model(spectra[start : start + batch_size])["latent_space"]  # (B, D)
                directions.append(canonical_direction(latent, gamma_tensor, beta_tensor)[1])
        return torch.cat(directions)  # (N, D)

    ## Fixed noise realizations, shared by every model so the comparison is paired on the draw
    repeat_generator = torch.Generator(device="cpu").manual_seed(seed + 500)
    perturbed_by_family = {
        name: [
            perturbation.perturb_spectra(
                clean_spectra, name, settings=perturbation_settings, generator=repeat_generator,
            ).spectra
            for _ in range(1 if name in perturbation.DETERMINISTIC_PERTURBATIONS else repeats_requested)
        ]
        for name in perturbation.PERTURBATION_NAMES
    }

    ## One model load per run feeds both the full-strength response and the sweep
    response_rows, amplitude_rows = [], []
    for position, record in enumerate(grid_frame.to_dict("records"), start=1):
        model = load_model(record["model_name"])
        model.eval()
        clean_direction = canonical_directions(model, clean_spectra)  # (N, D)

        for name, draws in perturbed_by_family.items():
            ### `paired_direction_angles` returns radians; the mean is taken over draws
            ### before conversion so the pixel stays the sampling unit.
            angles = torch.stack([
                paired_direction_angles(clean_direction, canonical_directions(model, draw))
                for draw in draws
            ]).mean(dim=0)  # (N,) radians
            degrees = np.degrees(angles.cpu().numpy())  # (N,)
            for pixel_index, value in enumerate(degrees):
                response_rows.append({
                    "cell_label": record["cell_label"], "campaign": record["campaign"],
                    "penalty_metric": record["penalty_metric"], "weight": record["weight"],
                    "repetition": record["repetition"], "perturbation": name,
                    "pixel": int(test_split.indices[selection][pixel_index]),
                    "angle_degrees": float(value),
                })

            ### The amplitude sweep uses the first draw only, on the first repetition
            if record["repetition"] == 0:
                for amplitude in amplitudes:
                    blended = perturbation.interpolate_perturbation(clean_spectra, draws[0], amplitude)
                    values = np.degrees(
                        paired_direction_angles(clean_direction, canonical_directions(model, blended)).cpu().numpy()
                    )
                    amplitude_rows.append({
                        "cell_label": record["cell_label"], "perturbation": name,
                        "amplitude": float(amplitude),
                        "median_angle_degrees": float(np.median(values)),
                        "mean_angle_degrees": float(np.mean(values)),
                    })

        if position % 10 == 0 or position == len(grid_frame):
            logger.info("Perturbation response: %s/%s model(s).", position, len(grid_frame))

    response_frame = pd.DataFrame(response_rows)
    return {
        "perturbation_response": response_frame,
        "response_medians": (
            response_frame.groupby(["cell_label", "repetition", "perturbation"])["angle_degrees"]
            .median().reset_index()
        ),
        "amplitude_curves": pd.DataFrame(amplitude_rows),
        "grid": grid_frame,
        "response_pixels": pd.DataFrame({"dataset_index": test_split.indices[selection]}),
        "metadata": provenance(settings, {
            "analysis": "perturbation_response",
            "perturbations": list(perturbation.PERTURBATION_NAMES),
            "deterministic_perturbations": sorted(perturbation.DETERMINISTIC_PERTURBATIONS),
            "perturbation_settings": settings.get("perturbation", {}),
            "noise_repeats": repeats_requested,
            "amplitudes": list(amplitudes),
            "response_pixels": int(len(selection)),
            "test_split_pixels": int(test_split.sampled),
            "models_evaluated": int(len(grid_frame)),
        }),
    }


@_routine("spectrum_reconstruction")
def spectrum_reconstruction(settings: dict) -> Dict[str, Any]:
    """Reconstructions, drift and prediction response of two models under perturbation.

    Unlike the other routines this one must persist spectra, not only summary
    statistics: the figures draw reconstructed spectra, and a spectrum cannot be redrawn
    from a median. The displayed series are therefore stored in long form on the
    binner's own m/z grid.

    :param settings: Analysis settings for this routine.
    :type settings: dict
    :return: Tables keyed by file name, plus ``metadata``.
    :rtype: Dict[str, Any]
    """
    from ..latent import perturbations as perturbation
    from ..latent import sphere_geometry as geometry
    from ..latent.sensitivity import canonical_direction, paired_direction_angles
    from ..reconstruction.metrics import masserstein_distances

    compared = dict(settings["compared"])            # display label -> cell label
    repetition = int(settings.get("repetition", 0))
    batch_size = int(settings.get("batch_size", 256))
    seed = int(settings.get("sample_seed", 42))
    top_k = int(settings.get("top_k", 10))
    cases_per_category = int(settings.get("cases_per_category", 2))
    display_amplitudes = tuple(settings.get("display_amplitudes", (0.0, 0.1, 0.5, 1.0)))
    curve_amplitudes = tuple(settings.get("curve_amplitudes", (0.0, 0.05, 0.1, 0.25, 0.5, 0.75, 1.0)))
    sample_pixels = int(settings.get("sample_pixels", 2000))
    perturbation_settings = perturbation.PerturbationSettings(**settings.get("perturbation", {}))

    grid_frame = campaign_grid(settings, settings["campaigns_used"])
    grid_frame = grid_frame[
        grid_frame["cell_label"].isin(compared.values()) & (grid_frame["repetition"] == repetition)
    ].reset_index(drop=True)
    model_name_by_cell = dict(zip(grid_frame["cell_label"], grid_frame["model_name"]))

    wrapper, splits = prepare_context(settings, grid_frame)
    test_split = splits["test"]
    device = resolve_device(settings, allow_cpu=True)

    generator = np.random.default_rng(seed)
    selection = (
        np.sort(generator.choice(test_split.sampled, sample_pixels, replace=False))
        if test_split.sampled > sample_pixels else np.arange(test_split.sampled)
    )
    clean = test_split.spectra[selection].to(device)  # (N, M)
    mass_axis = np.asarray(wrapper.active_context.binner.GetXAxis(), dtype=float)

    models = {}
    for label, cell_label in compared.items():
        model = wrapper.models_manager.load_model(
            img_name=settings["model_context"], model_name=model_name_by_cell[cell_label], strict=True
        )
        model.eval()
        models[label] = model

    def reconstruct(model, spectra: torch.Tensor) -> torch.Tensor:
        outputs = []
        with torch.no_grad():
            for start in range(0, len(spectra), batch_size):
                outputs.append(model(spectra[start : start + batch_size])["reconstruction"])  # (B, M)
        return torch.cat(outputs)  # (N, M)

    def latent_directions(model, spectra: torch.Tensor) -> torch.Tensor:
        gamma, beta = geometry.encoder_layer_norm_parameters(model)
        gamma_tensor = torch.as_tensor(gamma, device=spectra.device)
        beta_tensor = torch.as_tensor(beta, device=spectra.device)
        directions = []
        with torch.no_grad():
            for start in range(0, len(spectra), batch_size):
                latent = model(spectra[start : start + batch_size])["latent_space"]
                directions.append(canonical_direction(latent, gamma_tensor, beta_tensor)[1])
        return torch.cat(directions)  # (N, D)

    def head_probabilities(model, spectra: torch.Tensor) -> torch.Tensor:
        outputs = []
        with torch.no_grad():
            for start in range(0, len(spectra), batch_size):
                logits = model(spectra[start : start + batch_size])[f"head_{settings['head']}"]
                outputs.append(torch.sigmoid(logits))
        return torch.cat(outputs)  # (N, C)

    def transport(left: torch.Tensor, right: torch.Tensor) -> np.ndarray:
        return masserstein_distances(
            left.detach().cpu().numpy(), right.detach().cpu().numpy(), mass_axis, device=str(device)
        )

    clean_reconstruction = {label: reconstruct(model, clean) for label, model in models.items()}
    clean_direction = {label: latent_directions(model, clean) for label, model in models.items()}
    clean_probability = {label: head_probabilities(model, clean) for label, model in models.items()}
    clean_top_k = {label: torch.topk(value, top_k, dim=1).indices for label, value in clean_probability.items()}

    ## Stratified case selection by each model's own clean reconstruction cost
    clean_cost = {label: transport(clean, clean_reconstruction[label]) for label in compared}
    selection_rows = []
    for label in compared:
        order = np.argsort(clean_cost[label])
        middle = len(order) // 2
        for category, rows in (
            ("best", order[:cases_per_category]),
            ("median", order[middle : middle + cases_per_category]),
            ("worst", order[-cases_per_category:][::-1]),
        ):
            for rank, row in enumerate(rows):
                selection_rows.append({
                    "selected_by": label, "category": category, "rank": rank,
                    "row": int(row), "dataset_index": int(test_split.indices[selection][int(row)]),
                    **{f"W {other}": float(clean_cost[other][int(row)]) for other in compared},
                })
    case_frame = pd.DataFrame(selection_rows)
    category_order = ("best", "median", "worst")
    ordered = sorted(selection_rows, key=lambda r: (category_order.index(r["category"]), r["selected_by"], r["rank"]))
    shown_rows = list(dict.fromkeys(record["row"] for record in ordered))

    targets = {
        name: perturbation.perturb_spectra(
            clean, name, settings=perturbation_settings,
            generator=torch.Generator(device="cpu").manual_seed(seed + 700 + index),
        ).spectra
        for index, name in enumerate(perturbation.PERTURBATION_NAMES)
    }

    ## Spectra the figures draw: clean and every displayed amplitude, per selected case
    spectra_rows, panel_rows = [], []
    for row in shown_rows:
        series = {"input": clean[row].cpu().numpy()}
        series.update({label: clean_reconstruction[label][row].cpu().numpy() for label in compared})
        for label, values in series.items():
            spectra_rows.append(pd.DataFrame({
                "row": row, "dataset_index": int(test_split.indices[selection][row]),
                "perturbation": "clean", "amplitude": 0.0, "series": label,
                "mz": mass_axis, "intensity": values,
            }))

        for name, target in targets.items():
            for amplitude in display_amplitudes:
                blended = perturbation.interpolate_perturbation(clean, target, amplitude)[row : row + 1]
                traces, angles = {}, {}
                for label, model in models.items():
                    traces[label] = reconstruct(model, blended)[0].cpu().numpy()
                    angles[label] = float(np.degrees(paired_direction_angles(
                        clean_direction[label][row : row + 1], latent_directions(model, blended),
                    ).cpu().numpy()[0]))
                spectrum_in = blended[0].cpu().numpy()
                costs = dict(zip(
                    traces,
                    masserstein_distances(
                        np.tile(spectrum_in[None, :], (len(traces), 1)),
                        np.stack(list(traces.values())), mass_axis, device=str(device),
                    ).tolist(),
                ))
                for label, values in {"input": spectrum_in, **traces}.items():
                    spectra_rows.append(pd.DataFrame({
                        "row": row, "dataset_index": int(test_split.indices[selection][row]),
                        "perturbation": name, "amplitude": float(amplitude), "series": label,
                        "mz": mass_axis, "intensity": values,
                    }))
                panel_rows.append({
                    "dataset_index": int(test_split.indices[selection][row]),
                    "perturbation": name, "amplitude": float(amplitude),
                    **{f"W {label}": costs[label] for label in compared},
                    **{f"angle {label}": angles[label] for label in compared},
                })

    ## Population curves: drift, disagreement and prediction response against amplitude
    curve_rows, distribution_rows = [], []
    for name, target in targets.items():
        for amplitude in curve_amplitudes:
            blended = perturbation.interpolate_perturbation(clean, target, amplitude)
            reconstructions = {}
            for label, model in models.items():
                reconstructions[label] = reconstruct(model, blended)
                angles = np.degrees(paired_direction_angles(
                    clean_direction[label], latent_directions(model, blended)
                ).cpu().numpy())
                drift = transport(reconstructions[label], clean_reconstruction[label])
                euclidean = torch.linalg.vector_norm(
                    reconstructions[label] - clean_reconstruction[label], dim=1
                ).cpu().numpy()

                probability = head_probabilities(model, blended)
                perturbed_top_k = torch.topk(probability, top_k, dim=1).indices
                matches = clean_top_k[label].unsqueeze(2) == perturbed_top_k.unsqueeze(1)  # (N, K, K)
                retention = (matches.any(dim=2).sum(dim=1).float() / top_k).cpu().numpy()
                probability_drift = (probability - clean_probability[label]).abs().mean(dim=1).cpu().numpy()

                curve_rows.append({
                    "perturbation": name, "amplitude": float(amplitude), "model": label,
                    "median_angle_degrees": float(np.median(angles)),
                    "median_drift_masserstein": float(np.median(drift)),
                    "median_drift_euclidean": float(np.median(euclidean)),
                    "median_retention": float(np.median(retention)),
                    "q25_retention": float(np.quantile(retention, 0.25)),
                    "q75_retention": float(np.quantile(retention, 0.75)),
                    "median_probability_drift": float(np.median(probability_drift)),
                    "unchanged_pixels_fraction": float(np.mean(retention == 1.0)),
                })
                if amplitude in (0.1, 0.5, 1.0):
                    distribution_rows.append(pd.DataFrame({
                        "perturbation": name, "model": label, "amplitude": float(amplitude),
                        "row": np.arange(len(retention)),
                        "dataset_index": test_split.indices[selection],
                        "angle_degrees": angles, "drift_masserstein": drift, "retention": retention,
                    }))

            disagreement = transport(reconstructions[list(compared)[0]], reconstructions[list(compared)[1]])
            curve_rows.append({
                "perturbation": name, "amplitude": float(amplitude), "model": "between models",
                "median_angle_degrees": np.nan,
                "median_drift_masserstein": float(np.median(disagreement)),
                "median_drift_euclidean": float(np.median(torch.linalg.vector_norm(
                    reconstructions[list(compared)[0]] - reconstructions[list(compared)[1]], dim=1
                ).cpu().numpy())),
                "median_retention": np.nan, "q25_retention": np.nan, "q75_retention": np.nan,
                "median_probability_drift": np.nan, "unchanged_pixels_fraction": np.nan,
            })
            if amplitude in (0.0, 1.0):
                distribution_rows.append(pd.DataFrame({
                    "perturbation": name, "model": "between models", "amplitude": float(amplitude),
                    "row": np.arange(len(disagreement)),
                    "dataset_index": test_split.indices[selection],
                    "angle_degrees": np.nan, "drift_masserstein": disagreement, "retention": np.nan,
                }))

    return {
        "selected_cases": case_frame,
        "case_panels": pd.DataFrame(panel_rows),
        "displayed_spectra": pd.concat(spectra_rows, ignore_index=True),
        "amplitude_curves": pd.DataFrame(curve_rows),
        "response_distributions": pd.concat(distribution_rows, ignore_index=True),
        "grid": grid_frame,
        "analysed_spectra": pd.DataFrame({"dataset_index": test_split.indices[selection]}),
        "metadata": provenance(settings, {
            "analysis": "spectrum_reconstruction",
            "compared": compared,
            "compared_models": {label: model_name_by_cell[cell] for label, cell in compared.items()},
            "repetition": repetition,
            "top_k": top_k,
            "cases_per_category": cases_per_category,
            "display_amplitudes": list(display_amplitudes),
            "curve_amplitudes": list(curve_amplitudes),
            "perturbation_settings": settings.get("perturbation", {}),
            "analysed_spectra": int(clean.shape[0]),
            "displayed_rows": [int(row) for row in shown_rows],
            "displayed_dataset_indices": [int(test_split.indices[selection][int(r)]) for r in shown_rows],
            "mass_axis_range": [float(mass_axis[0]), float(mass_axis[-1])],
        }),
    }


if __name__ == "__main__":  # pragma: no cover - command-line entry
    sys.exit(main())
