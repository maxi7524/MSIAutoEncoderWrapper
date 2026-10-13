"""Script entry point for reference preparation and auditable analysis reports."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shlex
import subprocess
import sys
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from scipy.ndimage import binary_dilation

from .reference_numerical_validation import validate_numerics
from ..heads.metrics import evaluate_head, per_class_metrics, probabilities_from_logits
from ..latent.diagnostics import gradient_diagnostics, representation_diagnostics
from ..reconstruction.diagnostics import reconstruction_diagnostics
from ..spatial.evaluation_data import (
    EvaluationImage, load_blocksam, load_cardinal, load_evaluation_image, load_poseidon, save_evaluation_image,
)
from ..spatial.partition_analysis import chemical_cluster_profiles, cluster_latents
from ..spatial.reference import PixelMask, load_mask, rasterize_regions, save_mask
from ..spatial.reference_metrics import score_boundaries, score_partition
from ....visualization.predictive import distributions, heatmap
from ....visualization.spectra.views import plot_spectrum_comparison
from ....visualization.spatial.reference import plot_partition_comparison, plot_reference_comparison, plot_source_annotation, plot_source_signal_and_mask
from ....utils.logger import get_custom_logger

logger = get_custom_logger(__name__)


def _hash(path: Path) -> str:
    """Stream a file hash without materializing an entire raw dataset."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_safe(value: Any) -> Any:
    """Convert unavailable/nonfinite quantities to JSON null, preserving booleans."""
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(_json_safe(value), indent=2, allow_nan=False) + "\n")


def _directories(output: Path) -> tuple[Path, Path, Path]:
    report, plots, numbers = [output / name for name in ["reports", "plots", "numerical_values"]]
    for p in [report, plots, numbers]:
        p.mkdir(parents=True, exist_ok=True)
    return report, plots, numbers


def load_settings(path: Path | str) -> dict[str, Any]:
    """Read the shared settings; relative paths resolve against the repository.

    :param path: YAML settings file.
    :return: Settings with absolute root/output paths and source settings identity.
    :raises ValueError: For a missing repository or malformed seed settings.
    """
    source = Path(path).resolve()
    settings = yaml.safe_load(source.read_text())
    root = source.parent
    while root != root.parent and not (root / "pyproject.toml").exists():
        root = root.parent
    if not (root / "pyproject.toml").exists():
        raise ValueError("Settings must reside inside the repository.")
    settings["repository_root"] = str(root)
    settings["settings_path"] = str(source)
    settings["output"] = str(root / settings["output"])
    if len(settings["seeds"]) < 3 or len(set(settings["seeds"])) != len(settings["seeds"]):
        raise ValueError("Validation requires at least three distinct analysis seeds.")
    return settings


def run_command(settings_path: Path | str, analysis: str = "demo") -> str:
    """Generate the canonical CLI command for a notebook or missing-artifact error.

    :param settings_path: Shared YAML file.
    :param analysis: Registered analysis name.
    :return: Shell-quoted command without launching it.
    """
    return shlex.join([sys.executable, "-m", __name__ if __name__ != "__main__" else
                       "msi_autoencoder_wrapper.analysis.autoencoder.experiments.reference_validation_precompute",
                       "--settings", str(Path(settings_path).resolve()), "--analysis", analysis])


def load_table(settings: dict[str, Any], table: str) -> pd.DataFrame:
    """Load a canonical numerical table, or report the command that creates it.

    :param settings: Shared resolved settings.
    :param table: CSV basename under numerical_values.
    :return: Saved table; no model inference occurs.
    :raises FileNotFoundError: If precomputation has not run.
    """
    if Path(table).name != table:
        raise ValueError("Expected a table basename.")
    path = Path(settings["output"]) / "numerical_values" / table
    if not path.is_file():
        raise FileNotFoundError(f"Missing {path}. Run: {run_command(settings['settings_path'])}")
    return pd.read_csv(path)


def _provenance(settings: dict[str, Any], output: Path, device: str, analysis: str = "demo") -> None:
    """Persist source hashes, actual device, settings and produced artifact hashes."""
    root = Path(settings["repository_root"])
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    diff = subprocess.check_output(["git", "diff", "--binary"], cwd=root)
    sources = list((root / "src/msi_autoencoder_wrapper/analysis/autoencoder/spatial").glob("*.py"))
    sources += [Path(__file__), Path(__file__).with_name("reference_numerical_validation.py"),
                root / "src/msi_autoencoder_wrapper/analysis/autoencoder/reconstruction/diagnostics.py",
                root / "src/msi_autoencoder_wrapper/analysis/autoencoder/latent/diagnostics.py",
                root / "src/msi_autoencoder_wrapper/visualization/spatial/reference.py"]
    sources += [Path(__file__).with_name("spectral_partition_comparison.py"), root / "src/msi_autoencoder_wrapper/models/model_loader.py"]
    if analysis == "spectral_partitions":
        sources += list((root / "src/msi_autoencoder_wrapper/models/architectures").rglob("*.py"))
    record = {"git_commit": commit, "tracked_diff_sha256": hashlib.sha256(diff).hexdigest(), "device": device,
              "device_name": torch.cuda.get_device_name(device) if device.startswith("cuda") else "CPU",
              "seeds": settings["seeds"], "settings_sha256": _hash(Path(settings["settings_path"])),
              "source_hashes": {str(p.relative_to(root)): _hash(p) for p in sources},
              "versions": {name: importlib.metadata.version(name) for name in ["torch", "numpy", "scipy", "scikit-learn", "matplotlib"]},
              "artifact_hashes": {str(p.relative_to(output)): _hash(p) for p in sorted(output.rglob("*")) if p.is_file() and not p.name.startswith("provenance")}}
    _write_json(output / f"provenance_{analysis}.json", record)


def prepare_manual_annotation(settings: dict[str, Any]) -> Path:
    """Rasterize the explicit reviewable Poseidon trace and create its comparison.

    :param settings: Paths, digitization file and uncertainty width.
    :return: JPEG path; source mask is in H&E image coordinates, not MSI coordinates.
    """
    root, output = Path(settings["repository_root"]), Path(settings["output"])
    reports, plots, numbers = _directories(output)
    definition_path = root / settings["manual_annotation"]["regions"]
    definition = json.loads(definition_path.read_text())
    image_path = root / settings["manual_annotation"]["source_image"]
    image = plt.imread(image_path)  # (H, W, C)
    shape = image.shape[:2]
    if list(shape) != definition["source_image_shape"]:
        raise ValueError("Manual trace does not match the source image shape.")
    canvas_h, canvas_w = definition["digitization_canvas"]
    scaled = []
    for region in definition["regions"]:
        vertices = np.asarray(region["vertices"], dtype=float) * [shape[1] / canvas_w, shape[0] / canvas_h]  # (V, 2)
        scaled.append(dict(region, vertices=vertices.tolist()))
    labels, conflicts = rasterize_regions(shape, scaled)  # (H, W), (H, W)
    # Reserve an explicit uncertainty band around every manually traced contour
    boundary = np.zeros(shape, dtype=bool)  # (H, W)
    boundary[:-1] |= labels[:-1] != labels[1:]
    boundary[1:] |= labels[:-1] != labels[1:]
    boundary[:, :-1] |= labels[:, :-1] != labels[:, 1:]
    boundary[:, 1:] |= labels[:, :-1] != labels[:, 1:]
    width = settings["manual_annotation"]["uncertainty_pixels"]
    uncertain = binary_dilation(boundary, iterations=width) if width else boundary
    labels[uncertain] = -1
    metadata = {"dataset_id": "poseidon_ccrcc", "image_id": "histology_annotated", "coordinate_space": "histology_image",
                "class_names": definition["class_names"], "usage_role": "external_test_only",
                "annotation_status": definition["annotation_status"], "image_shape": list(shape),
                "index_convention": "zero_based_row_major_y_times_width_plus_x", "registration_status": "pending",
                "provenance": dict(definition["provenance"], source_image_sha256=_hash(image_path), regions_sha256=_hash(definition_path)),
                "uncertainty_pixels": width, "conflict_pixels": int(conflicts.sum()), "accepted_for_scoring": False}
    mask = PixelMask(np.arange(labels.size), labels.ravel(), metadata)
    destination = root / settings["manual_annotation"]["destination"]
    save_mask(mask, destination)
    reloaded = load_mask(destination)
    np.testing.assert_array_equal(reloaded.labels, labels.ravel())
    (destination / "regions.json").write_text(json.dumps(dict(definition, scaled_regions=scaled), indent=2) + "\n")
    figure, _ = plot_reference_comparison(image, labels, definition["class_names"], definition["colors"],
                                         title="manual_annotation | H&E image indices | registration pending")
    preview = plots / "manual_annotation_comparison.jpeg"
    figure.savefig(preview)
    plt.close(figure)
    from shutil import copyfile
    copyfile(preview, destination / preview.name)
    counts = pd.DataFrame([{"class_id": i, "class_name": name, "pixels": int(np.sum(labels == int(i)))} for i, name in definition["class_names"].items()])
    counts.to_csv(numbers / "manual_annotation_counts.csv", index=False)
    text = "# Manual annotation\n\nProvisional digitization of visible expert contours. These are H&E image pixel indices; MSI registration remains pending. Unknown pixels include unannotated tissue, conflicts and a contour uncertainty band. This artifact is available for visual review and is not accepted MSI ground truth.\n\n![Source and indexed mask](../plots/manual_annotation_comparison.jpeg)\n\n" + counts.to_markdown(index=False) + "\n"
    (reports / "manual_annotation.md").write_text(text)
    return preview


def prepare_datasets(settings: dict[str, Any]) -> None:
    """Decode reference releases once and retain canonical evaluation caches.

    :param settings: Source dataset and cache paths, exact acquisition selections.
    """
    root, output = Path(settings["repository_root"]), Path(settings["output"])
    reports, plots, numbers = _directories(output)
    records = []
    for item in settings["datasets"]:
        destination = root / settings["cache_root"] / item["dataset_id"] / item["image_id"]
        if (destination / "image_manifest.json").is_file():
            image = load_evaluation_image(destination)
        else:
            source = root / item["directory"]
            if item["dataset_id"] == "poseidon_ccrcc":
                image = load_poseidon(source, assay=item.get("assay", "Lipids"))
            elif item["dataset_id"] == "cardinal_rcc":
                image = load_cardinal(source, sample_name=item["image_id"])
            elif item["dataset_id"] == "blocksam_kidney":
                image = load_blocksam(source, sample_name=item["image_id"])
            else:
                raise ValueError("Unknown evaluation dataset adapter.")
            image.metadata["source_hashes"] = {p.name: _hash(p) for p in sorted((source / "raw").glob("*")) if p.is_file()}
            save_evaluation_image(image, destination)
        image.assert_use("reporting")
        labels = image.annotation.labels if image.annotation else np.full(len(image.pixel_ids), -1)
        names = image.annotation.metadata["class_names"] if image.annotation else {}
        figure, _ = plot_source_annotation(image.xy, labels, names, title=f"{image.dataset_id} / {image.image_id} | source annotation")
        figure.savefig(plots / f"source_annotation_{image.dataset_id}.png");plt.close(figure)
        feature = int(np.argmin(np.abs(image.mass_axis - settings["reference_preview_mz"])))
        signal = np.asarray(image.values[:, feature])  # (N,)
        context = "diagnostic labels" if image.annotation else "MSI mask unavailable; H&E reference separate"
        if image.annotation and len(np.unique(labels[labels >= 0])) == 1:
            context += "; selected image has one class only"
        figure, _ = plot_source_signal_and_mask(image.xy, signal, labels, names,
                  title=f"{image.dataset_id} / {image.image_id} | {context}",
                  signal_label=f"Observed intensity at m/z {image.mass_axis[feature]:.3f}")
        preview = plots / f"reference_dataset_{image.dataset_id}.jpeg"
        figure.savefig(preview);plt.close(figure)
        from shutil import copyfile
        source_reference = root / item["directory"] / "reference"
        source_reference.mkdir(exist_ok=True)
        copyfile(preview, source_reference / "source_signal_and_mask.jpeg")

        records.append({"dataset_id": image.dataset_id, "image_id": image.image_id, "pixels": len(image.pixel_ids),
                        "features": len(image.mass_axis), "reference_pixels": int(image.annotation.available.sum()) if image.annotation else 0,
                        "representation": image.metadata["representation"], "coordinate_units": image.metadata["coordinate_units"],
                        "input_compatibility": "not_approved", "cache": str(destination.relative_to(root))})
    table = pd.DataFrame(records)
    table.to_csv(numbers / "dataset_inventory.csv", index=False)
    (reports / "dataset_inventory.md").write_text("# External evaluation data\n\nSource decoding and index validation only. Input compatibility with each trained encoder must be checked separately; no zero-filling of unreported features is allowed.\n\n" + table.to_markdown(index=False) + "\n")
    with (reports / "dataset_inventory.md").open("a") as handle:
        for item in settings["datasets"]:
            handle.write(f"\n![Source signal and available mask: {item['dataset_id']}](../plots/reference_dataset_{item['dataset_id']}.jpeg)\n")
    _provenance(settings, output, "cpu", "datasets")


def write_prediction_report(image: EvaluationImage, outputs: dict[str, np.ndarray], settings: dict[str, Any],
                            output: Path | str, *, context: str) -> dict[str, Any]:
    """Analyze identity-aligned retained model outputs without fitting a neural model.

    :param image: Source image with a compatible input/annotation contract.
    :param outputs: pixel_ids, latents, reconstruction; optionally molecule_logits,
        molecule_targets and molecule_availability, all keyed by the same pixel IDs.
    :param settings: Fixed clustering, geometry, prevalence and device parameters.
    :param output: Report destination containing reports/plots/numerical_values/.
    :param context: Model/config/seed identity or explicit synthetic fixture label.
    :return: Scalar result summary; detailed canonical tables are persisted.
    :raises ValueError: For missing/inconsistent identities or output shapes.
    """
    reports, plots, numbers = _directories(Path(output))
    for field, expected in [("dataset_id", image.dataset_id), ("image_id", image.image_id)]:
        if field not in outputs or np.asarray(outputs[field]).size != 1 or str(np.asarray(outputs[field]).item()) != expected:
            raise ValueError(f"Prediction bundle {field} does not match the source identity.")
    ids = np.asarray(outputs["pixel_ids"])
    if ids.dtype.kind not in "iu" or not np.array_equal(ids, image.pixel_ids):
        raise ValueError("Inference outputs must retain the exact source pixel IDs and order.")
    z, reconstruction = outputs["latents"], outputs["reconstruction"]
    if z.shape[0] != len(ids) or reconstruction.shape != image.values.shape:
        raise ValueError("Retained model output shapes disagree with source data.")
    truth = image.annotation.align_to(ids) if image.annotation else None  # (N,) or unavailable
    summary, eig = representation_diagnostics(z)
    eig.to_csv(numbers / "latent_spectrum.csv", index=False)
    rec = reconstruction_diagnostics(image.values, reconstruction, image.mass_axis,
                                    np.asarray(settings["reference_prevalence"]), rare_threshold=settings["rare_threshold"], device=settings["device"])
    for name, table in rec.items():
        table.to_csv(numbers / f"reconstruction_{name}.csv", index=False)
    fig = distributions(rec["spectra"].assign(label=context), value="mae", group="label", title=f"{context}: per-spectrum reconstruction MAE")
    fig.savefig(plots / "reconstruction_mae.png")
    plt.close(fig)
    worst_row = int(rec["worst_errors"].iloc[0].row_position)
    fig, _ = plot_spectrum_comparison(image.mass_axis, image.values[worst_row], reconstruction[worst_row], clip_reconstruction=False)
    fig.suptitle(f"{context}: largest feature error; source_pixel_id={int(ids[worst_row])}")
    fig.savefig(plots / "worst_reconstruction.png");plt.close(fig)
    rec["features"].assign(rarity=lambda f: np.where(f.rare, "rare", "common_or_unseen")).to_csv(numbers / "feature_error_by_rarity.csv", index=False)
    fig = distributions(rec["features"].assign(label=lambda f: np.where(f.rare, "rare", "common_or_unseen")),
                        value="mae", group="label", title=f"{context}: feature MAE by frozen prevalence")
    fig.savefig(plots / "rare_feature_errors.png");plt.close(fig)
    gradient_keys = {k: torch.as_tensor(v) for k, v in outputs.items() if k.startswith("gradient_")}
    if gradient_keys:
        gradient_diagnostics(gradient_keys).to_csv(numbers / "model_gradient_snapshots.csv", index=False)
    partitions = []
    for method in settings["clustering"]["methods"]:
        for seed in settings["seeds"]:
            labels = cluster_latents(z, method=method, seed=seed, n_clusters=settings["clustering"]["n_clusters"],
                                     min_cluster_size=settings["clustering"]["min_cluster_size"])  # (N,)
            pd.DataFrame({"pixel_id": ids, "cluster": labels}).to_csv(numbers / f"clusters_{method}_{seed}.csv", index=False)
            if truth is None:
                continue
            score = score_partition(truth, labels)
            row = dict(score.summary, method=method, seed=seed)
            score.regions.to_csv(numbers / f"regions_{method}_{seed}.csv", index=False)
            score.contingency.to_csv(numbers / f"contingency_{method}_{seed}.csv", index_label="reference_class")
            # Raster scoring requires a declared source grid; physical units remain explicit
            if image.metadata["coordinate_units"] == "source_grid" and np.array_equal(image.xy, np.round(image.xy)):
                xy = image.xy.astype(int) - image.xy.min(axis=0).astype(int)  # (N, 2)
                shape = (int(xy[:, 1].max()) + 1, int(xy[:, 0].max()) + 1)
                if np.prod(shape) > settings["max_raster_pixels"] or len(np.unique(xy, axis=0)) != len(xy):
                    raise ValueError("Raster dimensions/duplicate coordinates violate the evaluation contract.")
                ref = np.full(shape, -1, dtype=int);aligned = np.full(shape, -2, dtype=int)  # (H, W), (H, W)
                ref[xy[:, 1], xy[:, 0]], aligned[xy[:, 1], xy[:, 0]] = truth, score.aligned
                boundary, interior = score_boundaries(ref, aligned, tolerance=settings["boundary_tolerance_pixels"])
                row.update(boundary)
                interior.to_csv(numbers / f"interiors_{method}_{seed}.csv", index=False)
                if seed == settings["seeds"][0]:
                    fig, _ = plot_partition_comparison(ref, aligned, title=f"{context} | {method} | seed={seed}")
                    fig.savefig(plots / f"segmentation_{method}.png");plt.close(fig)
            partitions.append(row)
            if "molecule_targets" in outputs:
                profiles, chemical = chemical_cluster_profiles(labels, outputs["molecule_targets"], outputs["molecule_availability"],
                                                               annotations_complete=settings["annotations_complete"])
                profiles.to_csv(numbers / f"chemical_profiles_{method}_{seed}.csv", index=False)
                chemical.to_csv(numbers / f"chemical_agreement_{method}_{seed}.csv", index=False)
                if seed == settings["seeds"][0]:
                    fig = heatmap(profiles, index="cluster", columns="class_index", values="observed_positive_fraction",
                                  title=f"{context}: {method} chemical annotation profiles", colorbar_label="Observed positive fraction")
                    fig.savefig(plots / f"chemical_profiles_{method}.png");plt.close(fig)
    partition_table = pd.DataFrame(partitions)
    partition_table.to_csv(numbers / "segmentation_scores.csv", index=False)
    if len(partition_table):
        fig = distributions(partition_table.rename(columns={"method": "label"}), value="macro_dice", group="label", title=f"{context}: segmentation Dice across clustering seeds")
        fig.savefig(plots / "segmentation_dice.png");plt.close(fig)
    head = None
    if "molecule_logits" in outputs:
        head = evaluate_head(outputs["molecule_logits"], outputs["molecule_targets"], "multi_label",
                             mask=outputs["molecule_availability"], threshold=settings["prediction_threshold"])
        probabilities = probabilities_from_logits(outputs["molecule_logits"], "multi_label")  # (N, C)
        pd.DataFrame(per_class_metrics(probabilities, outputs["molecule_targets"], settings["prediction_threshold"], outputs["molecule_availability"])).to_csv(numbers / "prediction_classes.csv", index=False)
    result = {"model_gradient_snapshots_available": bool(gradient_keys), "context": context, "latent": summary, "prediction": head, "segmentation_available": truth is not None,
              "metric_priority": ["segmentation", "prediction_precision", "reconstruction", "latent_and_gradients"]}
    _write_json(numbers / "summary.json", result)
    text = f"# Model analysis: {context}\n\nMetric priority: segmentation → prediction precision → reconstruction → latent/gradient stability. Cluster matching uses reference labels for scoring only. Clustering seeds are not independently trained models.\n\n"
    if truth is not None:
        text += "## Segmentation\n\n" + partition_table.to_markdown(index=False) + "\n\n![Segmentation Dice](../plots/segmentation_dice.png)\n\n"
        for method in settings["clustering"]["methods"]:
            if (plots / f"segmentation_{method}.png").exists():
                text += f"![Tissue reference, clusters and errors: {method}](../plots/segmentation_{method}.png)\n\n"
    else:
        text += "## Segmentation\n\nNot evaluable: no registered numeric reference.\n\n"
    text += "## Prediction\n\n" + (pd.DataFrame([head]).to_markdown(index=False) if head else "Not evaluable: no independent ion targets.") + "\n\n"
    text += "## Reconstruction\n\n![Error distribution](../plots/reconstruction_mae.png)\n\n" + rec["summary"].to_markdown(index=False) + "\n\n### Largest errors\n\n" + rec["worst_errors"].head(10).to_markdown(index=False) + "\n\n"
    text += "![Largest reconstruction failure](../plots/worst_reconstruction.png)\n\n![Rare feature errors](../plots/rare_feature_errors.png)\n\n"
    if "molecule_targets" in outputs:
        text += "## Chemical clustering\n\n"
        for method in settings["clustering"]["methods"]:
            text += f"![Chemical profiles: {method}](../plots/chemical_profiles_{method}.png)\n\n"
        text += "Rates describe observed annotations; incomplete PU zeros are not confirmed absences.\n\n"
    text += "## Latent geometry\n\n" + pd.DataFrame([summary]).to_markdown(index=False) + "\n\nChemical profiles and per-class agreement, interior errors, eigenvalues and full per-spectrum errors are in numerical_values/. Missing quantities are explicitly unavailable. Model gradient snapshots, when supplied, are saved separately; absent snapshots cannot establish a trained model's gradient health.\n"
    (reports / "model_analysis.md").write_text(text)
    return result


def run_demo(settings: dict[str, Any], *, allow_cpu: bool = False) -> None:
    """Validate metrics and export a synthetic example of future agent reports.

    :param settings: Shared seed/device/output/metric settings.
    :param allow_cpu: Explicit opt-out when configured CUDA is unavailable.
    """
    output = Path(settings["output"])
    reports, plots, numbers = _directories(output)
    device = settings["device"]
    if device.startswith("cuda") and not torch.cuda.is_available():
        if not allow_cpu:
            raise RuntimeError("CUDA is required. Run outside a device-restricted sandbox, or pass --allow-cpu explicitly.")
        device = "cpu"
    numerical, gradients = validate_numerics(device=device, seed=settings["seeds"][0],
                                            atol=settings["numerical_atol"], rtol=settings["numerical_rtol"])
    numerical.to_csv(numbers / "numerical_checks.csv", index=False)
    gradients.to_csv(numbers / "gradient_checks.csv", index=False)
    side = settings["synthetic"]["side"]
    yy, xx = np.mgrid[:side, :side]  # (H, W), (H, W)
    truth = (xx.ravel() >= side // 2).astype(int)  # (N,)
    axis = np.arange(settings["synthetic"]["features"], dtype=float) + 100  # (M,)
    records = []
    for seed in settings["seeds"]:
        rng = np.random.default_rng(seed)
        x = np.zeros((len(truth), len(axis)))  # (N, M)
        x[:, 5] = 1
        x[:, 10] = truth + .2
        rare_rows = rng.choice(len(x), max(1, len(x)//20), replace=False)  # (K,)
        x[rare_rows, -1] = .4
        prevalence = np.zeros(len(axis));prevalence[5]=1;prevalence[10]=1;prevalence[-1]=.05  # (M,)
        for quality in ["separable", "mixed"]:
            z = rng.normal(0, .2 if quality == "separable" else 1.5, (len(x), 4))  # (N, D)
            z[:, 0] += truth * 3
            y = np.maximum(x + rng.normal(0, .01 if quality == "separable" else .08, x.shape), 0)  # (N, M)
            y[:, -1] = 0  # Deliberate rare-feature failure for verifier inspection
            targets = np.column_stack([1-truth, truth]).astype(int)  # (N, C)
            logits = (targets * 2 - 1) * (4 if quality == "separable" else 1) + rng.normal(0, .1 if quality == "separable" else 2, targets.shape)  # (N, C)
            mask = PixelMask(np.arange(len(x)), truth, {"dataset_id": "analytic_fixture", "image_id": "two_regions",
                             "coordinate_space": "msi_source", "class_names": {"0": "region_left", "1": "region_right"},
                             "usage_role": "external_test_only", "annotation_status": "analytic_fixture",
                             "provenance": {"source": "synthetic_known_partition"}})
            image = EvaluationImage("analytic_fixture", "two_regions", np.arange(len(x)), x, axis,
                                    np.column_stack([xx.ravel(), yy.ravel()]), mask,
                                    {"usage_role": "external_test_only", "representation": "analytic_fixture", "coordinate_units": "source_grid"})
            current = dict(settings, device=device, reference_prevalence=prevalence.tolist(), annotations_complete=True)
            case_root = output / "cases" / f"{quality}_seed_{seed}"
            result = write_prediction_report(image, {"dataset_id": np.asarray(image.dataset_id), "image_id": np.asarray(image.image_id), "pixel_ids": image.pixel_ids, "latents": z, "reconstruction": y,
                                              "molecule_logits": logits, "molecule_targets": targets,
                                              "molecule_availability": np.ones(targets.shape, dtype=bool)}, current, case_root,
                                              context=f"SYNTHETIC {quality}; analysis_seed={seed}")
            scores = pd.read_csv(case_root / "numerical_values/segmentation_scores.csv")
            records.extend(dict(row, fixture=quality, model_seed=seed) for row in scores.to_dict("records"))
            np.savez_compressed(case_root / "numerical_values/fixture_inputs.npz", pixel_ids=image.pixel_ids, xy=image.xy, truth=truth,
                                inputs=x, reconstruction=y, latents=z, molecule_targets=targets, molecule_logits=logits,
                                rare_rows=rare_rows, mass_axis=axis, reference_prevalence=prevalence)
    table = pd.DataFrame(records)
    table.to_csv(numbers / "fixture_segmentation_scores.csv", index=False)
    # Negative controls make stability alone insufficient for accepting a model
    for label, partition in [("perfect", truth + 10), ("collapsed", np.zeros(len(truth),dtype=int)), ("all_noise", np.full(len(truth),-1))]:
        check = score_partition(truth, partition)
        records_control = dict(check.summary, fixture=label)
        _write_json(numbers / f"control_{label}.json", records_control)
    assert score_partition(truth, truth+10).summary["matched_accuracy"] == 1
    assert score_partition(truth, np.zeros(len(truth),dtype=int)).summary["matched_accuracy"] == .5
    assert score_partition(truth, np.full(len(truth),-1)).summary["matched_accuracy"] == 0
    good = table[(table.fixture=="separable") & (table.method=="kmeans")]
    bad = table[(table.fixture=="mixed") & (table.method=="kmeans")]
    assert good.matched_accuracy.mean() > bad.matched_accuracy.mean()
    fig = distributions(table[table.method=="kmeans"].assign(label=lambda f:f.fixture), value="matched_accuracy", group="label",
                           title="SYNTHETIC verifier: region assignment across three independent input seeds")
    fig.savefig(plots / "fixture_segmentation_accuracy.png");plt.close(fig)
    mask_preview = prepare_manual_annotation(settings)
    text = "# Analysis validation report\n\n**Synthetic verifier validation, not trained-model results.** Three independent fixture seeds test analysis sensitivity; they do not satisfy the three-training-repetition requirement for future ablations.\n\n"
    text += "## Anatomical reference for review\n\n![manual_annotation](../plots/manual_annotation_comparison.jpeg)\n\n[Mask provenance and counts](manual_annotation.md). H&E→MSI registration is pending.\n\n"
    text += "## Numerical verification\n\n" + numerical.to_markdown(index=False) + "\n\n### Gradient snapshots\n\n" + gradients.to_markdown(index=False) + "\n\n"
    text += "## Segmentation sensitivity\n\n![Assignment accuracy](../plots/fixture_segmentation_accuracy.png)\n\n" + table.to_markdown(index=False) + "\n\n"
    text += "## Complete report examples\n\n"
    for quality in ["separable", "mixed"]:
        for seed in settings["seeds"]:
            text += f"- [{quality}, seed {seed}](../cases/{quality}_seed_{seed}/reports/model_analysis.md)\n"
    text += "\n## Interpretation\n\nThe exact, collapsed and all-noise controls produce matched accuracies 1, 0.5 and 0. More overlap in the synthetic latent worsens K-means tissue assignment. Rare-feature errors are deliberately injected and appear in the worst-error and rare-feature tables. CPU/CUDA checks reuse existing Masserstein and InfoNCE code. The real references are decoded separately; this report contains no encoder inference, model training or automatically selected best model.\n"
    (reports / "validation_report.md").write_text(text)
    _provenance(settings, output, device)
    logger.info("Validation report written to %s; manual mask preview %s.", reports, mask_preview)



def publish_notebook_views(settings: dict[str, Any]) -> None:
    """Link notebook result views to the one canonical report artifact set.

    :param settings: Shared settings containing output and settings paths.
    :raises ValueError: If an existing result link points to another artifact set.
    """
    import os
    folder = Path(settings["settings_path"]).parent
    output = Path(settings["output"])
    for stem in ["part_0_1_reference_datasets", "part_0_2_model_analysis_contract"]:
        destination = folder / f"{stem}_results"
        destination.mkdir(exist_ok=True)
        for name in ["reports", "plots", "numerical_values", "cases"]:
            target = output / name
            link = destination / name
            if not target.exists():
                continue
            if link.is_symlink() and link.resolve() == target.resolve():
                continue
            if link.exists() or link.is_symlink():
                raise ValueError(f"Existing notebook result destination conflicts with canonical artifacts: {link}.")
            link.symlink_to(os.path.relpath(target, destination), target_is_directory=True)
        _write_json(destination / "artifact_source.json", {"canonical_output": str(output),
                                                          "settings_sha256": _hash(Path(settings["settings_path"]))})

def main(argv: list[str] | None = None) -> int:
    """Run one registered analysis from a shared YAML settings file."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--settings", required=True)
    parser.add_argument("--analysis", choices=["demo", "datasets", "masks", "bundle", "spectral_partitions"], default="demo")
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument("--image-cache")
    parser.add_argument("--prediction-bundle")
    args = parser.parse_args(argv)
    settings = load_settings(args.settings)
    if args.analysis == "demo":
        run_demo(settings, allow_cpu=args.allow_cpu)
    elif args.analysis == "spectral_partitions":
        from .spectral_partition_comparison import run_spectral_comparison
        run_spectral_comparison(settings, allow_cpu=args.allow_cpu)
    elif args.analysis == "datasets":
        prepare_datasets(settings)
    elif args.analysis == "masks":
        prepare_manual_annotation(settings)
        _provenance(settings, Path(settings["output"]), "cpu", "masks")
    else:
        if not args.image_cache or not args.prediction_bundle:
            parser.error("bundle requires --image-cache and --prediction-bundle")
        image = load_evaluation_image(args.image_cache)
        image.assert_model_input(settings["model_input_representation"])
        with np.load(args.prediction_bundle, allow_pickle=False) as bundle:
            outputs = {k: bundle[k] for k in bundle.files}
        write_prediction_report(image, outputs, settings, settings["output"], context=settings["model_context"])
        _provenance(settings, Path(settings["output"]), settings["device"])
    publish_notebook_views(settings)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
