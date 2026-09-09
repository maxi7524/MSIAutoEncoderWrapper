"""Shared, resumable inference and numerical tables for the predictive campaign."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from ....data.annotation_evidence import IonCatalogue, SignalEvidencePolicy
from ....models.model_loader import ModelLoader
from ....utils.logger import get_custom_logger
from ..heads.predictive_comparison import ranking_tables, score_histograms, state_separation
from ..latent.predictive_geometry import geometry_tables, ridge_probe
from ..latent.sphere_geometry import canonicalize, encoder_layer_norm_parameters
from ..reconstruction.metrics import masserstein_distances, reconstruction_metrics
from .predictive_campaign import fingerprint, relocated_data
from .sweep_evaluation import MaterializedSplit, materialize_split

logger = get_custom_logger(__name__)

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


def provenance_identity(record: dict) -> dict:
    """Reduce a provenance record to the entries that must match for cache reuse.

    :param record: Output of :func:`provenance`, optionally with input digests added.
    :type record: dict
    :return: Comparable subset; traceability-only entries are dropped.
    :rtype: dict
    """
    return {key: record[key] for key in PROVENANCE_IDENTITY if key in record}


def _source_digest(source_root: Path, prefixes: tuple[str, ...] | None = None) -> str:
    """Hash the content of every selected Python source file, path included.

    :param source_root: Package root whose modules are hashed.
    :param prefixes: Optional relative directory or file prefixes; ``None`` hashes all.
    :return: SHA-256 digest over ordered (relative path, bytes) pairs.
    :rtype: str
    """
    digest = hashlib.sha256()
    for path in sorted(source_root.rglob("*.py")):
        relative = path.relative_to(source_root).as_posix()
        if prefixes is not None and not any(relative == prefix or relative.startswith(prefix.rstrip("/") + "/")
                                            for prefix in prefixes):
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
    # REMARK: The full digest covers every reused metric, model and data
    # implementation, local edits included, so an analysis change invalidates the
    # stored numbers. The narrower decoding digest keys only the decoded tensors.
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, text=True, capture_output=True, check=False)
    return {"settings": {key: settings[key] for key in INFERENCE_SETTINGS if key in settings},
            "source_sha256": _source_digest(source_root),
            "decoding_sha256": _source_digest(source_root, DECODING_SOURCES),
            "git_commit": commit.stdout.strip(),
            "experiment_sha256": hashlib.sha256(Path(settings["experiment_config"]).read_bytes()).hexdigest(),
            "versions": {name: importlib.metadata.version(name) for name in ("numpy", "pandas", "scipy", "scikit-learn", "torch")}}


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
                                    catalogue.class_names, device=device)
            ### How each head orders operational negatives against unlabelled entries
            tables["state_separation"] = state_separation(output["logits"], state_arrays[name], train_counts,
                                                          catalogue.class_names, device=device)
            tables["score_histograms"] = score_histograms(output["logits"], state_arrays[name], train_counts, device=device)
            if name == "train":
                train_latent = output["latent"]
            else:
                if train_latent is None or not np.asarray(train.mask).all():
                    raise ValueError("The annotation probe needs train-first splits and fully available training labels.")
                probe_scores = ridge_probe(train_latent, train.targets, output["latent"], penalty=settings["probe_penalty"])  # (N, C)
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
                for space, latent in (("z", output["latent"]), ("u", u)):
                    for key, frame in geometry_tables(latent, split.targets * split.mask, sample, k=settings["neighbours"]).items():
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
