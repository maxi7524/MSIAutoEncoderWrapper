"""Shared, resumable inference and numerical tables for the predictive campaign.

Beyond the full-campaign inference cache (:func:`precompute`/:func:`load_table`, one
provenance-keyed run per unique settings/code/checkpoint combination), this module also
exposes a second, independent entry point for analyses that need bulk computation of
their own beyond what that cache already provides one table per notebook, in the same
style as :mod:`.contractive_precompute`'s named-routine registry:

- ``python -m msi_autoencoder_wrapper.analysis.autoencoder.experiments.predictive_precompute
  --settings <analysis_settings.yaml> --analysis campaign_training_dynamics`` for a
  background run;
- :func:`precompute_analysis` for the same thing from Python.

The counterpart is :func:`load_analysis_table`. Both are named distinctly from
:func:`precompute`/:func:`load_table` (the full-campaign cache) because the two serve
different scopes and are not interchangeable: the full-campaign cache is one inference
pass shared by every notebook, while an "analysis" routine here produces the smaller,
notebook-specific derived tables that the full-campaign cache alone does not cover.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import numpy as np
import pandas as pd
import torch

from ....data.annotation_evidence import IonCatalogue, SignalEvidencePolicy
from ....models.model_loader import ModelLoader
from ....utils.logger import get_custom_logger
from ..heads.predictive_comparison import (
    THREE_STATE_FAMILIES, ranking_tables, score_histograms, state_confusion, state_separation,
)
from ..latent.predictive_geometry import (
    geometry_tables, intrinsic_dimension_estimate, label_structure_correlation, ridge_probe, structure_summary,
)
from ..latent.sphere_geometry import canonicalize, encoder_layer_norm_parameters
from ..reconstruction.metrics import masserstein_distances, peak_matching_errors, reconstruction_metrics
from . import predictive_campaign as campaign
from .predictive_campaign import fingerprint, relocated_data
from .sweep_evaluation import MaterializedSplit, materialize_split

logger = get_custom_logger(__name__)

#: Named analysis routines, populated by :func:`_routine`. See the module docstring:
#: distinct from, and layered on top of, the full-campaign inference cache below.
ROUTINES: Dict[str, Callable[[dict], Dict[str, Any]]] = {}


def _routine(name: str) -> Callable:
    """Register one named analysis routine."""

    def register(function: Callable[[dict], Dict[str, Any]]) -> Callable:
        ROUTINES[name] = function
        return function

    return register

#: Settings that change the stored numbers. Anything outside this set is
#: presentational (shortlists, expected seed counts, cache location) and must not
#: invalidate hours of inference when a notebook's reporting choices change.
INFERENCE_SETTINGS = ("workspace", "model_store", "experiment_config", "sources", "target_field",
                      "device", "batch_size", "pixel_fraction", "sample_seed", "geometry_sample_size",
                      "neighbours", "probe_penalty", "case_count", "evidence", "path_remap")

#: Provenance entries that identify the computation itself. ``git_commit`` is
#: recorded for traceability but deliberately excluded: the analysed source content
#: is already covered by ``source_sha256``, so committing a notebook must not
#: invalidate an hours-long inference cache.
PROVENANCE_IDENTITY = ("settings", "source_sha256", "experiment_sha256", "versions",
                       "input_sha256", "class_names", "catalogue_bins")

#: Source subtrees and modules that determine how a saved split is decoded into
#: spectra, targets and availability masks. The decoded-tensor cache is keyed on
#: these instead of the whole package, so editing an analysis or plotting module
#: does not force a full re-decode of every partition.
DECODING_SOURCES = ("data", "readers", "binners", "normalization", "models/datasets",
                    "analysis/autoencoder/experiments/sweep_evaluation.py")

#: Subtrees that determine how a checkpoint is *trained*, never how an already-trained
#: checkpoint is *loaded and evaluated*: `precompute()` only calls `ModelLoader.load_artifact`
#: and forward passes, it never constructs a criterion. Excluded from `source_sha256` so that
#: active work on losses, pretraining or training-only data augmentation does not invalidate
#: this campaign's hours-long inference cache; a head or dataset module actually used for
#: *inference* stays under `models`/`analysis`/`data` proper and keeps invalidating it.
TRAINING_ONLY_SOURCES = ("training", "data/pretraining", "data/simulated_negatives")


def provenance_identity(record: dict) -> dict:
    """Reduce a provenance record to the entries that must match for cache reuse.

    :param record: Output of :func:`provenance`, optionally with input digests added.
    :type record: dict
    :return: Comparable subset; traceability-only entries are dropped.
    :rtype: dict
    """
    return {key: record[key] for key in PROVENANCE_IDENTITY if key in record}


def _source_digest(source_root: Path, prefixes: tuple[str, ...] | None = None, *,
                   exclude: tuple[str, ...] = ()) -> str:
    """Hash the content of every selected Python source file, path included.

    :param source_root: Package root whose modules are hashed.
    :param prefixes: Optional relative directory or file prefixes; ``None`` hashes all.
    :param exclude: Relative directory or file prefixes dropped after ``prefixes`` is
        applied, e.g. subtrees known not to affect the hashed computation's output.
    :return: SHA-256 digest over ordered (relative path, bytes) pairs.
    :rtype: str
    """
    def _matches(relative: str, patterns: tuple[str, ...]) -> bool:
        return any(relative == pattern or relative.startswith(pattern.rstrip("/") + "/") for pattern in patterns)

    digest = hashlib.sha256()
    for path in sorted(source_root.rglob("*.py")):
        relative = path.relative_to(source_root).as_posix()
        if prefixes is not None and not _matches(relative, prefixes):
            continue
        if exclude and _matches(relative, exclude):
            continue
        digest.update(relative.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _array_hash(*arrays: np.ndarray) -> str:
    """Hash array shape, dtype and complete contents without a giant bytes copy."""
    digest = hashlib.sha256()
    for array in arrays:
        array = np.ascontiguousarray(array)
        digest.update(str((array.shape, array.dtype)).encode())
        digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


def provenance(settings: dict) -> dict:
    """Record code, configuration, package versions and explicit random choices.

    :param settings: Resolved analysis configuration.
    :return: JSON-compatible provenance including hashes of analysis source files.
    :rtype: dict
    """
    root = Path(settings["repository_root"])
    source_root = root / "src" / "msi_autoencoder_wrapper"
    # REMARK: The full digest (training-only subtrees excluded, see TRAINING_ONLY_SOURCES)
    # covers every reused metric, model and data implementation, local edits included, so an
    # analysis change invalidates the stored numbers. The narrower decoding digest keys only
    # the decoded tensors.
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, text=True, capture_output=True, check=False)
    return {"settings": {key: settings[key] for key in INFERENCE_SETTINGS if key in settings},
            "source_sha256": _source_digest(source_root, exclude=TRAINING_ONLY_SOURCES),
            "decoding_sha256": _source_digest(source_root, DECODING_SOURCES),
            "git_commit": commit.stdout.strip(),
            "experiment_sha256": hashlib.sha256(Path(settings["experiment_config"]).read_bytes()).hexdigest(),
            "versions": {name: importlib.metadata.version(name) for name in ("numpy", "pandas", "scipy", "scikit-learn", "torch")}}


def resolve_device(settings: dict, allow_cpu: bool = False) -> torch.device:
    """Resolve the configured device, refusing a silent fall back to the processor.

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


def prepare_splits(models: pd.DataFrame, settings: dict) -> tuple[dict, IonCatalogue, np.ndarray]:
    """Restore the saved dataset once and materialize exact stored partitions.

    :param models: Ready models sharing an identical saved data contract.
    :param settings: Resolved analysis settings.
    :return: Materialized splits, ion catalogue and physical m/z axis.
    :rtype: tuple[dict, IonCatalogue, numpy.ndarray]
    :raises ValueError: If models use different data contracts or splits are empty.
    """
    from msi_autoencoder_wrapper import MSIAutoEncoderWrapper

    if models.empty or models.data_contract.nunique() != 1:
        raise ValueError("Select ready models with exactly one identical data contract.")
    config = json.loads((Path(models.iloc[0].artifact) / "config" / "config.json").read_text())
    local_config = relocated_data(config, settings["workspace"], settings.get("path_remap"))
    # Restore through the public loader using an ephemeral, relocated config copy.
    wrapper = MSIAutoEncoderWrapper(project_path=settings["workspace"])
    with tempfile.TemporaryDirectory(prefix="predictive-analysis-") as directory:
        config_dir = Path(directory) / "config"
        config_dir.mkdir()
        (config_dir / "config.json").write_text(json.dumps(local_config))
        wrapper.load_experiment(directory, load_model=False)
    partitions = wrapper.active_dataset.create_partitions()
    catalogue = IonCatalogue.from_dataset(wrapper.active_dataset, settings["target_field"])
    axis = np.asarray(wrapper.active_context.binner.GetXAxis())  # (M,)
    # REMARK: The existing decoder cache omits assignments from its key. Namespace
    # it by the full saved data contract AND the identity of the source files and
    # input files that actually determine decoding, to prevent reuse across
    # same-sized but different partitions or replaced input files.
    file_identity = []
    for component in local_config["data"]["context"]["components"].values():
        for key, value in component.get("parameters", {}).items():
            if key in {"path", "file_path", "image_path"} and isinstance(value, str):
                path = Path(value)
                if path.is_file():
                    info = path.stat()
                    file_identity.append((str(path), info.st_size, info.st_mtime_ns))
                    if path.suffix.lower() == ".imzml":
                        binary = path.with_suffix(".ibd")
                        if binary.is_file():
                            info = binary.stat()
                            file_identity.append((str(binary), info.st_size, info.st_mtime_ns))
    implementation = provenance(settings)
    cache = Path(settings["cache_directory"]) / "decoded" / fingerprint(
        [models.iloc[0].data_contract, file_identity, implementation["decoding_sha256"], implementation["versions"]]
    )
    splits = {}
    for name in ("train", "validation", "test"):
        if len(getattr(partitions, name)) == 0:
            raise ValueError(f"Saved split {name} is empty.")
        splits[name] = materialize_split(partitions, name, settings["target_field"],
                                          fraction=settings["pixel_fraction"], seed=settings["sample_seed"],
                                          decode_batch_size=settings["batch_size"], cache_directory=cache)
    return splits, catalogue, axis


def infer_model(model: Any, spectra: torch.Tensor, head: str, *, batch_size: int) -> dict[str, np.ndarray]:
    """Run one active head, encoder and decoder together in evaluation mode.

    :param model: Loaded model; previous training/evaluation mode is restored.
    :param spectra: Model-scale spectra, shape ``(N, M)``.
    :param head: Active head key resolved from the training objective.
    :param batch_size: Positive inference batch size.
    :return: CPU arrays for logits, latent and reconstruction.
    :rtype: dict[str, numpy.ndarray]
    :raises ValueError: If inputs or model outputs are invalid.
    """
    if batch_size < 1 or len(spectra) == 0:
        raise ValueError("Inference needs nonempty spectra and a positive batch size.")
    device = next(model.parameters()).device
    was_training = model.training
    result = {"logits": [], "latent": [], "reconstruction": []}
    keys = {"logits": f"head_{head}", "latent": "latent_space", "reconstruction": "reconstruction"}
    model.eval()
    try:
        with torch.inference_mode():
            for start in range(0, len(spectra), batch_size):
                outputs = model(spectra[start:start + batch_size].to(device))
                for name, key in keys.items():
                    values = outputs[key]  # (B, C[, 3]), (B, D), or (B, M)
                    if not bool(torch.isfinite(values).all()):
                        raise ValueError(f"Non-finite {key} output.")
                    result[name].append(values.detach().cpu().numpy())
    finally:
        model.train(was_training)
    return {name: np.concatenate(chunks, axis=0) for name, chunks in result.items()}  # (N, ...)


def reconstruction_tables(split: MaterializedSplit, outputs: np.ndarray, axis: np.ndarray, *, options: dict,
                          batch_size: int, device: str | torch.device = "cpu") -> dict[str, pd.DataFrame]:
    """Reuse training-compatible reconstruction costs and preserve pixel distributions.

    :param split: Materialized model inputs and sample identities.
    :param outputs: Decoder predictions, ``(N, M)``.
    :param axis: Physical bin coordinates, ``(M,)``.
    :param options: Saved Masserstein parameters from this model's objective.
    :param batch_size: Cost computation batch size.
    :param device: Device the Masserstein transport cost is evaluated on.
    :return: Pixel costs, feature errors and run summaries.
    :rtype: dict[str, pandas.DataFrame]
    """
    x = split.spectra.numpy()  # (N, M)
    values = reconstruction_metrics(x, outputs)
    values["masserstein"] = masserstein_distances(x, outputs, axis, batch_size=batch_size,
                                                  device=device, criterion_options=options)
    pixel_keys = ("mse", "mae", "cosine_similarity", "spectral_angle", "tic_error", "masserstein")
    pixels = pd.DataFrame({"row_position": np.arange(len(x)), "split_position": split.indices,
                           "annotation_count": (split.targets * split.mask).sum(axis=1),
                           "input_max": x.max(axis=1), "input_nonzero_bins": (x > 0).sum(axis=1),
                           **{key: values[key] for key in pixel_keys}})
    rows = []
    for metric in pixel_keys:
        array = np.asarray(values[metric])
        for statistic, value in (("mean", array.mean()), ("median", np.median(array)),
                                 ("q90", np.quantile(array, .9)), ("q99", np.quantile(array, .99))):
            rows.append({"metric": metric, "statistic": statistic, "value": value, "pixels": len(x)})
    residual = outputs - x  # (N, M)
    features = pd.DataFrame({"mz": axis, "mean_absolute_error": np.abs(residual).mean(axis=0),
                             "mean_signed_error": residual.mean(axis=0), "mean_input": x.mean(axis=0)})
    return {"reconstruction_pixels": pixels, "reconstruction": pd.DataFrame(rows), "reconstruction_features": features}


def precompute(models: pd.DataFrame, settings: dict, *, prepared: tuple | None = None, model_loader=None) -> Path:
    """Evaluate completed runs once and reuse only exact provenance-matched outputs.

    :param models: Full campaign inventory; only ready rows are evaluated.
    :param settings: Resolved settings; controls sampling and shared evidence policy.
    :param prepared: Optional pre-materialized splits/catalogue/axis for integration tests.
    :param model_loader: Optional artifact loader returning a torch model.
    :return: Cache directory containing an atomic completed manifest.
    :rtype: pathlib.Path
    :raises ValueError: If no ready models, incompatible contracts or invalid artifacts exist.
    """
    ready = models[models.ready].copy()
    if ready.empty:
        raise ValueError("No completed models with matching config and weights. Download artifacts first.")
    for source in settings["sources"]:
        if source.get("enabled", True) and source.get("required", False) and not (ready.source == source["name"]).any():
            raise ValueError(f"Required source {source['name']} has no ready model; inspect campaign audit.")
    if ready.data_contract.nunique() != 1:
        raise ValueError("Data contracts differ. Inspect inventory and disable incompatible sources before inference.")
    root = Path(settings["cache_directory"])
    root.mkdir(parents=True, exist_ok=True)
    run_metadata = provenance(settings)
    splits, catalogue, axis = prepared if prepared is not None else prepare_splits(ready, settings)
    policy = SignalEvidencePolicy(**settings["evidence"])
    train = splits["train"]
    train_counts = (train.targets * train.mask).sum(axis=0)  # (C,)
    state_arrays, geometry_indices, input_digests = {}, {}, {}
    for name, split in splits.items():
        state_arrays[name] = policy.classify(split.spectra, torch.as_tensor(split.targets),
                                             torch.as_tensor(split.mask), catalogue).numpy()  # (N, C)
        geometry_indices[name] = np.sort(np.random.default_rng(settings["sample_seed"]).choice(
            split.sampled, min(settings["geometry_sample_size"], split.sampled), replace=False))  # (S,)
        input_digests[name] = _array_hash(split.spectra.numpy(), split.targets, split.mask, split.indices, axis)
    run_metadata["input_sha256"] = input_digests
    run_metadata["class_names"] = list(catalogue.class_names)
    run_metadata["catalogue_bins"] = catalogue.bins
    run_key = fingerprint(provenance_identity(run_metadata))
    run_dir = root / "evaluations" / run_key
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "metadata.json").write_text(json.dumps(run_metadata, indent=2))
    ready.to_csv(run_dir / "inventory.csv", index=False)
    geometry_sets = {name: set(indices) for name, indices in geometry_indices.items()}
    sample_rows = [{"split": name, "row_position": row, "split_position": int(position),
                    "geometry_selected": row in geometry_sets[name]}
                   for name, split in splits.items() for row, position in enumerate(split.indices)]
    pd.DataFrame(sample_rows).to_csv(run_dir / "sample_indices.csv", index=False)
    pd.DataFrame({"class_index": np.arange(len(catalogue.class_names)), "class_name": catalogue.class_names,
                  "train_positives": train_counts}).to_csv(run_dir / "classes.csv", index=False)
    train_states = state_arrays["train"]
    pd.DataFrame({"class_name": catalogue.class_names,
                  "positive_entries": (train_states == 1).sum(axis=0),
                  "negative_entries": (train_states == 0).sum(axis=0),
                  "uncertain_entries": (train_states == 2).sum(axis=0),
                  "relative_threshold": policy.relative_threshold,
                  "bin_radius": policy.bin_radius}).to_csv(run_dir / "train_class_states.csv", index=False)
    records = []
    # One checkpoint at a time; one shared forward pass per model and split
    for row in ready.to_dict("records"):
        artifact_key = ModelLoader.artifact_fingerprint(row["artifact"])
        model_dir = run_dir / fingerprint([row["model_id"], artifact_key])
        marker = model_dir / "complete.json"
        records.append({"model_id": row["model_id"], "directory": str(model_dir), "artifact_sha256": artifact_key})
        complete = json.loads(marker.read_text()) if marker.is_file() else {}
        expected_tables = complete.get("tables", [])
        if (expected_tables and all((model_dir / f"{key}.csv").is_file() for key in expected_tables)
                and all((model_dir / f"{name}_latent.npz").is_file() for name in splits)):
            logger.info("Reusing completed inference for %s.", row["model_id"])
            continue
        model_dir.mkdir(parents=True, exist_ok=True)
        marker.unlink(missing_ok=True)
        model = model_loader(row["artifact"]) if model_loader else ModelLoader.load_artifact(row["artifact"], strict=True)[0]
        model.to(settings["device"])
        gamma, beta = encoder_layer_norm_parameters(model)
        objective = json.loads(row["objective_json"])
        reconstruction_specs = list(objective.get("reconstruction", {}).values())
        masserstein = next((spec for spec in reconstruction_specs if spec["target"] == "MassersteinLoss"), None)
        if masserstein is None:
            raise ValueError("Masserstein reconstruction parameters must be recorded for this campaign.")
        collected = {}
        train_latent = None
        for name, split in splits.items():
            logger.info("Evaluating %s: %s (%s pixels).", row["model_id"], name, split.sampled)
            output = infer_model(model, split.spectra, row["head"], batch_size=settings["batch_size"])
            device = settings["device"]
            tables = ranking_tables(output["logits"], split.targets, state_arrays[name], train_counts,
                                    catalogue.class_names, device=device, family=row["family"])
            ### How each head orders operational negatives against unlabelled entries
            tables["state_separation"] = state_separation(output["logits"], state_arrays[name], train_counts,
                                                          catalogue.class_names, device=device, family=row["family"])
            tables["score_histograms"] = score_histograms(output["logits"], state_arrays[name], train_counts,
                                                           device=device, family=row["family"])
            ### Does the head's own N/P/U decision actually separate N from U, not
            ### just rank them (that is `state_separation` above)? Only meaningful
            ### for the two families that make an actual three-way decision; every
            ### other model still writes the (empty, correctly shaped) table so
            ### `load_table` can keep assuming every model has every table file.
            if row["family"] in THREE_STATE_FAMILIES:
                confusion = state_confusion(output["logits"], state_arrays[name], train_counts,
                                            catalogue.class_names, device=device)
                tables["state_confusion"] = confusion["state_confusion"]
                tables["state_confusion_summary"] = confusion["state_confusion_summary"]
            else:
                tables["state_confusion"] = pd.DataFrame(columns=["class_index", "class_name", "train_positives",
                                                                   "true_state", "predicted_state", "count",
                                                                   "true_state_total", "share"])
                tables["state_confusion_summary"] = pd.DataFrame(columns=["class_index", "class_name",
                                                                          "train_positives", "entries", "accuracy",
                                                                          "majority_baseline_accuracy",
                                                                          "chi2_statistic", "chi2_p_value", "cramers_v"])
            if name == "train":
                train_latent = output["latent"]
            else:
                if train_latent is None or not np.asarray(train.mask).all():
                    raise ValueError("The annotation probe needs train-first splits and fully available training labels.")
                probe_scores = ridge_probe(train_latent, train.targets, output["latent"], penalty=settings["probe_penalty"])  # (N, C)
                # REMARK: `family` is deliberately omitted here — the ridge probe is
                # always a plain linear score regardless of which loss trained the
                # real head, so it must keep the default shape-only convention.
                probe = ranking_tables(probe_scores, split.targets, state_arrays[name], train_counts,
                                       catalogue.class_names, device=device)
                tables["probe_prediction"] = probe["prediction"]
                tables["probe_per_class"] = probe["per_class"]
            tables.update(reconstruction_tables(split, output["reconstruction"], axis,
                                                options=masserstein.get("params", {}),
                                                batch_size=settings["batch_size"], device=device))
            # Persist shared representative spectra: fixed positions, not selected by
            # whichever model wins. Worst cases remain available in the pixel table.
            sample = geometry_indices[name]
            u = canonicalize(output["latent"], gamma, beta)  # (N, D)
            if name != "train":
                available_targets = split.targets * split.mask
                for space, latent in (("z", output["latent"]), ("u", u)):
                    computed = geometry_tables(latent, available_targets, sample, k=settings["neighbours"])
                    if space == "u":
                        ### Structure-vs-null, intrinsic dimension and label correlation are only
                        ### defined on the canonicalized sphere (`sphere_geometry.structure_test`'s
                        ### uniform baseline assumes the zero-row-sum/constant-norm constraint that
                        ### canonicalization enforces), so they are computed for u only, never z.
                        extra = [{"metric": key, "value": value}
                                for key, value in structure_summary(
                                    latent, sample, np.random.default_rng(settings["sample_seed"])).items()]
                        extra.append({"metric": "two_nn_intrinsic_dimension",
                                     "value": intrinsic_dimension_estimate(latent, sample)})
                        correlation = label_structure_correlation(
                            latent, sample, available_targets, np.random.default_rng(settings["sample_seed"] + 1))
                        extra.extend({"metric": f"label_correlation_{key}", "value": value}
                                    for key, value in correlation.items())
                        computed["geometry"] = pd.concat([computed["geometry"], pd.DataFrame(extra)], ignore_index=True)
                    for key, frame in computed.items():
                        tables.setdefault(key, pd.DataFrame())
                        tables[key] = pd.concat([tables[key], frame.assign(space=space)], ignore_index=True)
            # All model-specific worst examples are explicitly labelled as selected.
            worst = np.argsort(tables["reconstruction_pixels"].masserstein.to_numpy())[-settings["case_count"]:]
            cases = np.unique(np.concatenate([sample[:settings["case_count"]], worst]))  # (K_cases,)
            case_rows = pd.DataFrame({"row_position": np.repeat(cases, len(axis)),
                                      "mz": np.tile(axis, len(cases)),
                                      "input": split.spectra.numpy()[cases].reshape(-1),
                                      "reconstruction": output["reconstruction"][cases].reshape(-1),
                                      "selected_as_worst": np.repeat(np.isin(cases, worst), len(axis))})
            tables["spectrum_cases"] = case_rows
            ### Peak-level reconstruction fidelity (position vs. intensity error), on the same
            ### bounded case set as the spectrum examples above, not the full split: matching is
            ### per-spectrum and its cost scales with peak count, so it is confined to the cases
            ### already selected for inspection rather than run over every pixel.
            peaks = peak_matching_errors(split.spectra.numpy()[cases], output["reconstruction"][cases], axis)
            tables["peak_matching"] = pd.DataFrame({
                "row_position": cases[peaks["spectrum_index"]], "peak_mz": peaks["peak_mz"],
                "mz_error": peaks["mz_error"], "relative_intensity_error": peaks["relative_intensity_error"],
                "original_intensity": peaks["original_intensity"], "detected": peaks["detected"],
                "selected_as_worst": np.isin(cases[peaks["spectrum_index"]], worst)})
            np.savez_compressed(model_dir / f"{name}_latent.npz", z=output["latent"][sample], u=u[sample], rows=sample)
            for key, frame in tables.items():
                collected.setdefault(key, []).append(frame.assign(split=name, model_id=row["model_id"], label=row["label"],
                                                                   condition=row["condition"], repetition=row["repetition"]))
        for key, frames in collected.items():
            pd.concat(frames, ignore_index=True).to_csv(model_dir / f"{key}.csv", index=False)
        marker.write_text(json.dumps({"artifact_sha256": artifact_key, "tables": sorted(collected)}))
        del model
    manifest = {"run_directory": str(run_dir), "models": records, "run_key": run_key}
    temporary = root / "latest.tmp.json"
    temporary.write_text(json.dumps(manifest, indent=2))
    temporary.replace(root / "latest.json")
    logger.info("Completed analysis cache: %s", run_dir)
    return run_dir


def load_table(settings: dict, table: str) -> pd.DataFrame:
    """Read canonical per-run tables from the latest fully completed precomputation.

    :param settings: Resolved settings.
    :param table: Table stem, e.g. ``prediction`` or ``geometry``.
    :return: All individual model records; no repetition averaging is applied.
    :rtype: pandas.DataFrame
    :raises FileNotFoundError: If precomputation has not completed.
    :raises ValueError: If settings/code changed since the completed precomputation.
    """
    root = Path(settings["cache_directory"])
    path = root / "latest.json"
    if not path.is_file():
        raise FileNotFoundError("Run part_1_campaign_audit and part_2_shared_inference first.")
    manifest = json.loads(path.read_text())
    stored = provenance_identity(json.loads((Path(manifest["run_directory"]) / "metadata.json").read_text()))
    current = provenance_identity(provenance(settings))
    if any(stored.get(key) != value for key, value in current.items()):
        raise ValueError("Settings, source code or package versions changed; rerun shared inference.")
    inventory = pd.read_csv(Path(manifest["run_directory"]) / "inventory.csv").set_index("model_id")
    for row in manifest["models"]:
        if not (Path(row["directory"]) / "complete.json").is_file():
            raise ValueError("A cached model is incomplete; rerun shared inference.")
        if ModelLoader.artifact_fingerprint(inventory.loc[row["model_id"], "artifact"]) != row["artifact_sha256"]:
            raise ValueError("Checkpoint bytes changed; rerun shared inference.")
    frames = [pd.read_csv(Path(row["directory"]) / f"{table}.csv") for row in manifest["models"]]
    return pd.concat(frames, ignore_index=True)


# --------------------------------------------------
# Section: named analysis routines
# --------------------------------------------------

@_routine("campaign_training_dynamics")
def campaign_training_dynamics(settings: dict) -> Dict[str, Any]:
    """Manifest coverage, per-epoch objectives and training health of every task.

    No model is loaded: everything comes from the campaign's recorded manifests and
    training histories. Mirrors :func:`.contractive_precompute.campaign_training_dynamics`
    for a predictive objective (multiple sources/roles instead of one grid of penalty
    cells; no single scalar "penalty" term, so :func:`.predictive_campaign.training_health_report`
    checks the training loss trajectory instead).

    :param settings: Analysis settings for this routine.
    :type settings: dict
    :return: Tables keyed by file name, plus ``metadata``.
    :rtype: Dict[str, Any]
    """
    models, sources = campaign.inventory(settings)
    grid = campaign.configured_grid(settings["experiment_config"])
    candidate_source = next(source["name"] for source in settings["sources"]
                            if source.get("role", "candidate") != "baseline")
    coverage = campaign.coverage_table(grid, models, source=candidate_source)
    history = campaign.history_components(campaign.training_history(models))
    planned_epochs = settings.get("planned_epochs")
    return {
        "inventory": models,
        "sources": sources,
        "condition_coverage": coverage,
        "training_dynamics": history[history.record_type == "epoch"],
        "final_evaluation": history[history.record_type == "evaluation"],
        "training_health": campaign.training_health_report(history, models, planned_epochs=planned_epochs),
        "run_durations": campaign.run_duration_frame(history, models),
        "metadata": {**provenance(settings), "analysis": "campaign_training_dynamics",
                    "planned_epochs": planned_epochs, "tasks": len(models), "epoch_rows": len(history)},
    }


@_routine("reconstruction_local")
def reconstruction_local(settings: dict) -> Dict[str, Any]:
    """Reconstructions, drift and prediction response of two models under perturbation.

    Ported from :func:`.contractive_precompute.spectrum_reconstruction`. The "exactly
    two models" convention is unchanged and enforced the same way: every figure loop
    below is written generically over ``compared``, except the between-model
    disagreement panels, which read only ``list(compared)[0]``/``[1]``. Unlike the
    contractive grid (one geometry/weight axis, repetitions differing only in seed), the
    two compared models here may use entirely different heads and reconstruction
    objectives, so each is resolved and run independently; :func:`infer_model` (one
    forward pass, all three outputs) replaces the three separate single-purpose forward
    passes the contractive routine runs.

    ``compared`` maps a display label to a condition ``label`` (as in the ``inventory``
    table), not to a raw ``model_id``, mirroring how the contractive routine's
    ``compared`` maps to a ``cell_label`` rather than a ``model_name``: a condition label
    is what a settings file should name, a model id is an artifact of how many
    repetitions happened to be downloaded. ``repetition`` disambiguates *within* the
    swept `predictive_initial` grid; the historical baseline is a single run and is
    matched on label alone, since it was not trained as part of that repetition-indexed
    grid and its own ``repetition`` field carries no comparable meaning.

    Unlike the other routines this one must persist spectra, not only summary
    statistics: the figures draw reconstructed spectra, and a spectrum cannot be
    redrawn from a median. The displayed series are therefore stored in long form on
    the decoded m/z grid.

    :param settings: Analysis settings for this routine.
    :type settings: dict
    :return: Tables keyed by file name, plus ``metadata``.
    :rtype: Dict[str, Any]
    :raises ValueError: If a named condition is absent from the ready inventory.
    """
    from ..heads.predictive_comparison import positive_scores
    from ..latent import perturbations as perturbation
    from ..latent.sensitivity import canonical_direction, paired_direction_angles

    compared = dict(settings["compared"])            # display label -> condition label
    repetition = int(settings.get("repetition", 0))
    batch_size = int(settings.get("batch_size", 256))
    seed = int(settings.get("sample_seed", 42))
    top_k = int(settings.get("top_k", 10))
    cases_per_category = int(settings.get("cases_per_category", 2))
    display_amplitudes = tuple(settings.get("display_amplitudes", (0.0, 0.1, 0.5, 1.0)))
    curve_amplitudes = tuple(settings.get("curve_amplitudes", (0.0, 0.05, 0.1, 0.25, 0.5, 0.75, 1.0)))
    sample_pixels = int(settings.get("sample_pixels", 2000))
    perturbation_settings = perturbation.PerturbationSettings(**settings.get("perturbation", {}))
    device = resolve_device(settings, allow_cpu=True)

    all_models, _ = campaign.inventory(settings)
    ready = all_models[all_models.ready & all_models.label.isin(compared.values())]
    ready = ready[(ready.source != "predictive_initial") | (ready.repetition == repetition)]
    selected = ready.drop_duplicates("label").set_index("label")
    missing = set(compared.values()) - set(selected.index)
    if missing:
        raise ValueError(f"'compared' names condition(s) absent from the ready inventory: {sorted(missing)}")

    splits, _, axis = prepare_splits(selected.reset_index(drop=True), settings)
    test_split = splits["test"]

    generator = np.random.default_rng(seed)
    selection = (np.sort(generator.choice(test_split.sampled, sample_pixels, replace=False))
                if test_split.sampled > sample_pixels else np.arange(test_split.sampled))
    clean = test_split.spectra[selection]  # (N, M)

    models, heads, canon_params = {}, {}, {}
    for label, condition_label in compared.items():
        row = selected.loc[condition_label]
        model = ModelLoader.load_artifact(row["artifact"], strict=True)[0]
        model.to(device)
        model.eval()
        models[label] = model
        heads[label] = row["head"]
        canon_params[label] = encoder_layer_norm_parameters(model)

    def infer(label: str, spectra: torch.Tensor) -> dict[str, np.ndarray]:
        return infer_model(models[label], spectra, heads[label], batch_size=batch_size)

    def direction(label: str, latent: np.ndarray) -> torch.Tensor:
        gamma, beta = canon_params[label]
        return canonical_direction(torch.as_tensor(latent, dtype=torch.float32),
                                   torch.as_tensor(gamma, dtype=torch.float32),
                                   torch.as_tensor(beta, dtype=torch.float32))[1]

    def transport(left: np.ndarray, right: np.ndarray) -> np.ndarray:
        return masserstein_distances(left, right, axis, batch_size=batch_size, device=str(device))

    clean_outputs = {label: infer(label, clean) for label in compared}
    clean_reconstruction = {label: clean_outputs[label]["reconstruction"] for label in compared}
    clean_direction = {label: direction(label, clean_outputs[label]["latent"]) for label in compared}
    clean_score = {label: torch.as_tensor(positive_scores(clean_outputs[label]["logits"])) for label in compared}
    clean_top_k = {label: torch.topk(value, top_k, dim=1).indices for label, value in clean_score.items()}

    ## Stratified case selection by each model's own clean reconstruction cost
    clean_cost = {label: transport(clean.numpy(), clean_reconstruction[label]) for label in compared}
    selection_rows = []
    for label in compared:
        order = np.argsort(clean_cost[label])
        middle = len(order) // 2
        for category, rows in (("best", order[:cases_per_category]), ("median", order[middle:middle + cases_per_category]),
                              ("worst", order[-cases_per_category:][::-1])):
            for rank, row in enumerate(rows):
                selection_rows.append({"selected_by": label, "category": category, "rank": rank,
                                       "row": int(row), "dataset_index": int(test_split.indices[selection][int(row)]),
                                       **{f"W {other}": float(clean_cost[other][int(row)]) for other in compared}})
    case_frame = pd.DataFrame(selection_rows)
    category_order = ("best", "median", "worst")
    ordered = sorted(selection_rows, key=lambda r: (category_order.index(r["category"]), r["selected_by"], r["rank"]))
    shown_rows = list(dict.fromkeys(record["row"] for record in ordered))

    targets = {name: perturbation.perturb_spectra(clean, name, settings=perturbation_settings,
                                                  generator=torch.Generator(device="cpu").manual_seed(seed + 700 + index)).spectra
              for index, name in enumerate(perturbation.PERTURBATION_NAMES)}

    ## Spectra the figures draw: clean and every displayed amplitude, per selected case
    spectra_rows, panel_rows = [], []
    for row in shown_rows:
        series = {"input": clean[row].numpy()}
        series.update({label: clean_reconstruction[label][row] for label in compared})
        for label, values in series.items():
            spectra_rows.append(pd.DataFrame({"row": row, "dataset_index": int(test_split.indices[selection][row]),
                                              "perturbation": "clean", "amplitude": 0.0, "series": label,
                                              "mz": axis, "intensity": values}))
        for name, target in targets.items():
            for amplitude in display_amplitudes:
                blended = perturbation.interpolate_perturbation(clean, target, amplitude)[row:row + 1]
                traces, angles = {}, {}
                for label in compared:
                    output = infer(label, blended)
                    traces[label] = output["reconstruction"][0]
                    angles[label] = float(np.degrees(paired_direction_angles(
                        clean_direction[label][row:row + 1], direction(label, output["latent"]),
                    ).cpu().numpy()[0]))
                spectrum_in = blended[0].numpy()
                costs = dict(zip(traces, transport(np.tile(spectrum_in[None, :], (len(traces), 1)),
                                                   np.stack(list(traces.values())))))
                for label, values in {"input": spectrum_in, **traces}.items():
                    spectra_rows.append(pd.DataFrame({"row": row, "dataset_index": int(test_split.indices[selection][row]),
                                                      "perturbation": name, "amplitude": float(amplitude), "series": label,
                                                      "mz": axis, "intensity": values}))
                panel_rows.append({"dataset_index": int(test_split.indices[selection][row]), "perturbation": name,
                                   "amplitude": float(amplitude), **{f"W {label}": costs[label] for label in compared},
                                   **{f"angle {label}": angles[label] for label in compared}})

    ## Population curves: drift, disagreement and prediction response against amplitude
    curve_rows, distribution_rows = [], []
    for name, target in targets.items():
        for amplitude in curve_amplitudes:
            blended = perturbation.interpolate_perturbation(clean, target, amplitude)
            reconstructions = {}
            for label in compared:
                output = infer(label, blended)
                reconstructions[label] = output["reconstruction"]
                angles = np.degrees(paired_direction_angles(clean_direction[label], direction(label, output["latent"])).cpu().numpy())
                drift = transport(reconstructions[label], clean_reconstruction[label])
                euclidean = np.linalg.norm(reconstructions[label] - clean_reconstruction[label], axis=1)

                score = torch.as_tensor(positive_scores(output["logits"]))
                perturbed_top_k = torch.topk(score, top_k, dim=1).indices
                matches = clean_top_k[label].unsqueeze(2) == perturbed_top_k.unsqueeze(1)  # (N, K, K)
                retention = (matches.any(dim=2).sum(dim=1).float() / top_k).numpy()
                # REMARK: predictive heads are not all binary-sigmoid (a PNU head is a
                # 3-way softmax per class), so `positive_scores` (already used by
                # `ranking_tables`) supplies a head-agnostic positive log-odds score in
                # place of the contractive routine's plain sigmoid probability; the drift
                # below is therefore a score drift, not literally a probability drift.
                score_drift = (score - clean_score[label]).abs().mean(dim=1).numpy()

                curve_rows.append({"perturbation": name, "amplitude": float(amplitude), "model": label,
                                   "median_angle_degrees": float(np.median(angles)),
                                   "median_drift_masserstein": float(np.median(drift)),
                                   "median_drift_euclidean": float(np.median(euclidean)),
                                   "median_retention": float(np.median(retention)),
                                   "q25_retention": float(np.quantile(retention, .25)),
                                   "q75_retention": float(np.quantile(retention, .75)),
                                   "median_score_drift": float(np.median(score_drift)),
                                   "unchanged_pixels_fraction": float(np.mean(retention == 1.0))})
                if amplitude in (0.1, 0.5, 1.0):
                    distribution_rows.append(pd.DataFrame({"perturbation": name, "model": label, "amplitude": float(amplitude),
                                                           "row": np.arange(len(retention)), "dataset_index": test_split.indices[selection],
                                                           "angle_degrees": angles, "drift_masserstein": drift, "retention": retention}))

            left_label, right_label = list(compared)[0], list(compared)[1]
            disagreement = transport(reconstructions[left_label], reconstructions[right_label])
            curve_rows.append({"perturbation": name, "amplitude": float(amplitude), "model": "between models",
                               "median_angle_degrees": np.nan, "median_drift_masserstein": float(np.median(disagreement)),
                               "median_drift_euclidean": float(np.median(np.linalg.norm(
                                   reconstructions[left_label] - reconstructions[right_label], axis=1))),
                               "median_retention": np.nan, "q25_retention": np.nan, "q75_retention": np.nan,
                               "median_score_drift": np.nan, "unchanged_pixels_fraction": np.nan})
            if amplitude in (0.0, 1.0):
                distribution_rows.append(pd.DataFrame({"perturbation": name, "model": "between models", "amplitude": float(amplitude),
                                                       "row": np.arange(len(disagreement)), "dataset_index": test_split.indices[selection],
                                                       "angle_degrees": np.nan, "drift_masserstein": disagreement, "retention": np.nan}))

    return {
        "selected_cases": case_frame,
        "case_panels": pd.DataFrame(panel_rows),
        "displayed_spectra": pd.concat(spectra_rows, ignore_index=True),
        "amplitude_curves": pd.DataFrame(curve_rows),
        "response_distributions": pd.concat(distribution_rows, ignore_index=True),
        "grid": selected.reset_index(),
        "analysed_spectra": pd.DataFrame({"dataset_index": test_split.indices[selection]}),
        "metadata": {**provenance(settings), "analysis": "reconstruction_local", "compared": compared,
                    "compared_model_ids": {label: selected.loc[condition_label, "model_id"]
                                           for label, condition_label in compared.items()},
                    "compared_heads": heads, "repetition": repetition, "top_k": top_k,
                    "cases_per_category": cases_per_category, "display_amplitudes": list(display_amplitudes),
                    "curve_amplitudes": list(curve_amplitudes), "perturbation_settings": settings.get("perturbation", {}),
                    "analysed_spectra": int(clean.shape[0]), "displayed_rows": [int(row) for row in shown_rows],
                    "displayed_dataset_indices": [int(test_split.indices[selection][int(r)]) for r in shown_rows],
                    "mass_axis_range": [float(axis[0]), float(axis[-1])]},
    }


@_routine("reconstruction_global")
def reconstruction_global(settings: dict) -> Dict[str, Any]:
    """Campaign-wide reconstruction error, per-image spread and a coarse drift check.

    Unlike :func:`reconstruction_local` (exactly two named models, per-case detail
    under a fine perturbation sweep, individual spectra persisted for plotting), this
    routine covers every ready model at once: it is a population check, not an
    inspectable one, so no case selection or spectrum persistence happens here.

    Three things beyond the base full-campaign cache are computed:

    - the per-pixel error table (already in the base ``reconstruction_pixels`` table)
      joined to a source-image identity, when the dataset records one
      (:func:`.sweep_evaluation.MaterializedSplit.sample_ids`; a merged-store
      ``PixelDataset``'s plain integer ids are not resolved further here, only a
      ``CohortDataset``'s own ``image_key`` is used directly — resolving a merged
      store's ``dataset_id``/``image_path`` would need the live annotation reader,
      which :func:`prepare_splits` does not currently expose; ``per_image_identity_available``
      in the metadata says which case this run is in);
    - a coarse, campaign-wide perturbation drift table: every ready model, a handful of
      amplitudes (not `reconstruction_local`'s full curve), reusing the same shared
      evaluation sample and perturbation targets so every model sees identical inputs;
    - that drift table's ratio against the ``baseline`` role's own drift, at matching
      perturbation/amplitude, so a distribution-level shift can be read directly
      instead of eyeballed off the raw numbers.

    :param settings: Analysis settings for this routine.
    :type settings: dict
    :return: Tables keyed by file name, plus ``metadata``.
    :rtype: Dict[str, Any]
    :raises ValueError: If no ready models exist.
    """
    from ..latent import perturbations as perturbation

    all_models, _ = campaign.inventory(settings)
    ready = all_models[all_models.ready].reset_index(drop=True)
    if ready.empty:
        raise ValueError("No ready models to aggregate.")
    identity = ready[["model_id", "source", "role", "condition", "label", "repetition"]]
    batch_size = int(settings.get("batch_size", 256))
    evaluation_split_name = settings.get("evaluation_split", "test")

    splits, _, axis = prepare_splits(ready, settings)
    evaluation_split = splits[evaluation_split_name]

    pixels = load_table(settings, "reconstruction_pixels")
    pixels = pixels[pixels.split == evaluation_split_name].copy()

    ## Per-image breakdown, only where the dataset records a source-image identity
    ## REMARK: defined columns even when empty, so an empty table still round-trips
    ## through `pandas.read_csv` (a columnless DataFrame does not).
    image_breakdown = pd.DataFrame(columns=["model_id", "image_key", "mean_masserstein", "median_masserstein",
                                            "pixels", "source", "role", "condition", "label", "repetition"])
    sample_ids = evaluation_split.sample_ids
    per_image_identity_available = (sample_ids is not None and len(sample_ids) > 0
                                    and isinstance(sample_ids[0], dict) and "image_key" in sample_ids[0])
    if per_image_identity_available:
        image_by_position = dict(zip(evaluation_split.indices, (entry["image_key"] for entry in sample_ids)))
        pixels["image_key"] = pixels["split_position"].map(image_by_position)
        image_breakdown = (
            pixels.groupby(["model_id", "image_key"])
            .agg(mean_masserstein=("masserstein", "mean"), median_masserstein=("masserstein", "median"), pixels=("masserstein", "size"))
            .reset_index().merge(identity, on="model_id", validate="many_to_one")
        )

    ## Coarse perturbation-based drift: every ready model, a handful of amplitudes
    perturbation_settings = perturbation.PerturbationSettings(**settings.get("perturbation", {}))
    amplitudes = tuple(settings.get("global_perturbation_amplitudes", (0.0, 1.0)))
    seed = int(settings.get("sample_seed", 42))
    sample_pixels = int(settings.get("global_perturbation_sample_pixels", 500))
    device = resolve_device(settings, allow_cpu=True)

    generator = np.random.default_rng(seed)
    selection = (np.sort(generator.choice(evaluation_split.sampled, sample_pixels, replace=False))
                if evaluation_split.sampled > sample_pixels else np.arange(evaluation_split.sampled))
    clean = evaluation_split.spectra[selection]  # (N, M)
    targets = {name: perturbation.perturb_spectra(clean, name, settings=perturbation_settings,
                                                  generator=torch.Generator(device="cpu").manual_seed(seed + 900 + index)).spectra
              for index, name in enumerate(perturbation.PERTURBATION_NAMES)}

    drift_rows = []
    for row in ready.to_dict("records"):
        model = ModelLoader.load_artifact(row["artifact"], strict=True)[0]
        model.to(device)
        model.eval()
        clean_reconstruction = infer_model(model, clean, row["head"], batch_size=batch_size)["reconstruction"]
        for name, target in targets.items():
            for amplitude in amplitudes:
                blended = perturbation.interpolate_perturbation(clean, target, amplitude)
                reconstructed = infer_model(model, blended, row["head"], batch_size=batch_size)["reconstruction"]
                drift = masserstein_distances(clean_reconstruction, reconstructed, axis, batch_size=batch_size, device=str(device))
                drift_rows.append({"model_id": row["model_id"], "perturbation": name, "amplitude": float(amplitude),
                                   "median_drift_masserstein": float(np.median(drift)), "pixels": len(drift)})
        del model
    drift_frame = pd.DataFrame(drift_rows).merge(identity, on="model_id", validate="many_to_one")

    ## Distribution shift vs. baseline: same perturbation/amplitude, ratio to the
    ## baseline role's own median drift (undefined, left NaN, without a baseline role)
    baseline_ids = set(ready[ready.role == "baseline"].model_id)
    baseline_drift = (drift_frame[drift_frame.model_id.isin(baseline_ids)]
                      .groupby(["perturbation", "amplitude"]).median_drift_masserstein.mean())
    keys = list(zip(drift_frame.perturbation, drift_frame.amplitude))
    drift_frame["baseline_median_drift_masserstein"] = [baseline_drift.get(key, np.nan) for key in keys]
    drift_frame["times_baseline"] = drift_frame.median_drift_masserstein / drift_frame.baseline_median_drift_masserstein

    return {
        "reconstruction_summary": load_table(settings, "reconstruction"),
        "error_distributions": pixels,
        "image_breakdown": image_breakdown,
        "perturbation_drift": drift_frame,
        "metadata": {**provenance(settings), "analysis": "reconstruction_global",
                    "evaluation_split": evaluation_split_name, "amplitudes": list(amplitudes),
                    "sample_pixels": int(clean.shape[0]), "models": len(ready),
                    "per_image_identity_available": per_image_identity_available},
    }


@_routine("latent_geometry")
def latent_geometry(settings: dict) -> Dict[str, Any]:
    """Encoder sensitivity to input perturbation and raw angular-structure samples.

    Everything else contractive's ``latent_geometry_sweep`` computes (trace, effective
    rank, participation ratio, 2NN intrinsic dimension, RSA, cross-model similarity
    battery) is already produced by the base full-campaign inference cache
    (``predictive_geometry.geometry_tables``/``structure_summary``/``label_structure_correlation``,
    wired into :func:`precompute`'s per-model loop) and is read directly from the base
    ``geometry`` table by the notebook — reused, not rebuilt here; likewise, cross-model
    reproducibility uses the already-existing :func:`.predictive_reports.geometry_similarity`.

    Only two things are genuinely missing there: encoder sensitivity under isotropic
    input perturbation (:func:`~.latent.sphere_geometry.angular_sensitivity_curve`, used
    only inside the contractive sweep until now) and the raw cos-theta samples behind
    the base layer's structure summary (:func:`~.latent.predictive_geometry.structure_summary`
    keeps only the reduced statistics). Both need one extra encoding pass per model,
    which this routine performs once and persists, so the notebook does not load any
    model itself.

    :param settings: Analysis settings for this routine.
    :type settings: dict
    :return: Tables keyed by file name, plus ``metadata``.
    :rtype: Dict[str, Any]
    :raises ValueError: If no ready models exist.
    """
    from ..latent.sphere_geometry import angular_sensitivity_curve, structure_test

    all_models, _ = campaign.inventory(settings)
    ready = all_models[all_models.ready].reset_index(drop=True)
    if ready.empty:
        raise ValueError("No ready models to encode.")
    identity = ready[["model_id", "source", "role", "condition", "label", "repetition"]]
    evaluation_split_name = settings.get("evaluation_split", "test")
    batch_size = int(settings.get("batch_size", 256))
    seed = int(settings.get("sample_seed", 42))
    pair_count = int(settings.get("pair_count", 5000))
    epsilons = tuple(settings.get("epsilons", (0.001, 0.003, 0.01, 0.03, 0.1)))
    device = resolve_device(settings, allow_cpu=True)

    splits, _, _ = prepare_splits(ready, settings)
    evaluation_split = splits[evaluation_split_name]
    sensitivity_inputs = evaluation_split.spectra.numpy()
    structure_rng = np.random.default_rng(seed + 1)
    sensitivity_rng = np.random.default_rng(seed + 3)

    sensitivity_rows, structure_rows, cos_rows = [], [], []
    for row in ready.to_dict("records"):
        model = ModelLoader.load_artifact(row["artifact"], strict=True)[0]
        model.to(device)
        model.eval()
        gamma, beta = encoder_layer_norm_parameters(model)
        latent = infer_model(model, evaluation_split.spectra, row["head"], batch_size=batch_size)["latent"]
        u = canonicalize(latent, gamma, beta)  # (N, D)

        ## Raw angular structure, kept row-level (the base layer keeps only the summary)
        result = structure_test(u, structure_rng, pair_count=pair_count, return_samples=True)
        samples = np.asarray(result["cos_theta_samples"])
        cos_rows.append(pd.DataFrame({"model_id": row["model_id"], "cos_theta": samples}))
        structure_rows.append({"model_id": row["model_id"],
                               **{key: value for key, value in result.items() if key != "cos_theta_samples"}})

        ## Encoder sensitivity to isotropic input perturbation
        def encode(batch: np.ndarray, _model=model, _gamma=gamma, _beta=beta) -> np.ndarray:
            with torch.no_grad():
                encode_device = next(_model.parameters()).device
                codes = _model(torch.as_tensor(batch, dtype=torch.float32).to(encode_device))["latent_space"]
            return canonicalize(codes.cpu().numpy(), _gamma, _beta)

        curve = angular_sensitivity_curve(encode, sensitivity_inputs, epsilons, sensitivity_rng)
        for epsilon, angle in zip(curve["epsilon"], curve["mean_angle_degrees"]):
            sensitivity_rows.append({"model_id": row["model_id"], "epsilon": float(epsilon),
                                     "mean_angle_degrees": float(angle)})
        del model

    empty_samples = pd.DataFrame(columns=["model_id", "cos_theta"])
    return {
        "angular_sensitivity": pd.DataFrame(sensitivity_rows).merge(identity, on="model_id", validate="many_to_one"),
        "angular_structure": pd.DataFrame(structure_rows).merge(identity, on="model_id", validate="many_to_one"),
        "angular_structure_samples": (pd.concat(cos_rows, ignore_index=True) if cos_rows else empty_samples)
                                     .merge(identity, on="model_id", validate="many_to_one"),
        "metadata": {**provenance(settings), "analysis": "latent_geometry", "evaluation_split": evaluation_split_name,
                    "pair_count": pair_count, "epsilons": list(epsilons), "models": len(ready)},
    }


# --------------------------------------------------
# Section: running and loading named analysis routines
# --------------------------------------------------

def analysis_settings(settings: dict, analysis: str) -> dict:
    """Merge the shared settings with one analysis routine's own overrides.

    :param settings: Parsed settings file, as returned by :func:`.predictive_campaign.load_settings`.
    :type settings: dict
    :param analysis: Registered routine name.
    :type analysis: str
    :return: Effective settings for that routine, including ``output_directory``.
    :rtype: dict
    :raises KeyError: If the analysis is not registered or not configured.
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


def precompute_analysis(settings: dict, analysis: str, *, allow_cpu: bool = False) -> Path:
    """Run one named analysis routine and write its tables to its own directory.

    :param settings: Parsed settings file.
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
    logger.info("Running analysis '%s' on %s.", analysis, device)

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


def load_analysis_table(settings: dict, analysis: str, table: str) -> pd.DataFrame:
    """Read one table produced by a named analysis routine.

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
        raise FileNotFoundError(f"Missing '{path}'. Produce it first:\n  {run_analysis_command(settings, analysis)}")
    return pd.read_csv(path)


def load_analysis_metadata(settings: dict, analysis: str) -> dict:
    """Read a named analysis routine's recorded provenance.

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
        raise FileNotFoundError(f"Missing '{path}'. Produce it first:\n  {run_analysis_command(settings, analysis)}")
    return json.loads(path.read_text())


def run_analysis_command(settings: dict, analysis: str, *, background: bool = True) -> str:
    """Return the shell command that produces one named analysis routine's tables.

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
    module = "msi_autoencoder_wrapper.analysis.autoencoder.experiments.predictive_precompute"
    command = f"python -m {module} --settings {settings_path} --analysis {analysis}"
    if not background:
        return command
    log = Path(analysis_settings(settings, analysis)["output_directory"]) / f"{analysis}.log"
    return f"nohup {command} > {log} 2>&1 &"


def main(argv: Optional[list[str]] = None) -> int:
    """Command-line entry point, so a long analysis run can be detached from a notebook."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--settings", required=True, help="Path to analysis_settings.yaml.")
    parser.add_argument("--analysis", required=True, choices=sorted(ROUTINES), help="Routine to run.")
    parser.add_argument("--allow-cpu", action="store_true", help="Accept a processor run.")
    arguments = parser.parse_args(argv)

    settings = campaign.load_settings(arguments.settings)
    settings["settings_path"] = arguments.settings
    output = precompute_analysis(settings, arguments.analysis, allow_cpu=arguments.allow_cpu)
    print(f"wrote {arguments.analysis} tables to {output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
