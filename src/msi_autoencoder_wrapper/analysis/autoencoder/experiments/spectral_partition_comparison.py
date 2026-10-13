"""Raw-spectrum versus trained-encoder partition comparisons on known images."""

from __future__ import annotations

import gc
import hashlib
from pathlib import Path
import time
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from ....binners.binners_strategies.linear_binner import LinearBinning
from ....models.model_loader import ModelLoader
from ....utils.logger import get_custom_logger
from ....visualization.predictive import distributions
from ....visualization.spatial.reference import plot_clustering_methods, plot_spectral_region_profiles
from ..spatial.gpu_partition_methods import cluster_on_cuda, latent_rank_on_cuda
from ..spatial.reference_metrics import score_boundaries, score_partition
from ..spatial.synthetic_partitions import infer_frozen_encoder, spectral_partition

logger = get_custom_logger(__name__)


def _state_hash(model: torch.nn.Module) -> str:
    """Hash parameters and buffers to detect inference-state mutation."""
    digest = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        value = tensor.detach().cpu().numpy()
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(str(value.shape).encode())
        digest.update(value.tobytes())
    return digest.hexdigest()


def run_spectral_comparison(settings: dict[str, Any], *, allow_cpu: bool = False) -> None:
    """Generate, encode and cluster the exact same spectral fixture observations.

    Existing checkpoints are selected by fixed IDs, with three repetitions per
    family. No optimization, head fitting or test-dependent parameter selection
    occurs. Ground-truth labels are excluded from encoder/clustering inputs.

    :param settings: Shared paths, models, source axis, seeds and fixed method settings.
    :param allow_cpu: Legacy CLI flag; CPU fallback is forbidden for this analysis.
    :raises ValueError: For incompatible checkpoints or benchmark configuration.
    :raises AssertionError: For failed numerical or identity contracts.
    """
    from .reference_validation_precompute import _directories, _provenance, _write_json

    root, output = Path(settings["repository_root"]), Path(settings["output"])
    reports, plots, numbers = _directories(output)
    options = settings["spectral_partition_comparison"]
    device = settings["device"]
    if not device.startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError("Spectral comparison requires CUDA; CPU fallback is forbidden.")
    if options["inference_dtype"] != "float32":
        raise ValueError("The GPU comparison uses the original float32 checkpoint dtype.")
    axis = LinearBinning(**options["model_axis"]).grid.copy()  # (M,)
    model_specs = options["models"]
    groups: dict[str, set[int]] = {}
    for spec in model_specs:
        groups.setdefault(spec["family"], set()).add(spec["repetition"])
    if not groups or any(len(reps) < 3 for reps in groups.values()):
        raise ValueError("Each trained model family requires at least three fixed repetitions.")
    models = []
    checks, records, galleries, profile_records, region_records = [], [], [], [], []
    # Load fixed historical checkpoints and verify their source input contract
    for spec in model_specs:
        artifact = root / spec["artifact"]
        fingerprint = ModelLoader.artifact_fingerprint(artifact)
        model, config, directory = ModelLoader.load_artifact(artifact, strict=True)
        encoder = config["model"]["components"]["encoder"]["parameters"]
        binner = config.get("data", {}).get("context", {}).get("components", {}).get("binner", {}).get("parameters")
        if encoder["input_dim"] != len(axis) or binner != options["model_axis"]:
            raise ValueError(f"Checkpoint input axis differs from the spectral benchmark: {directory}.")
        normalization = config["data"]["dataset"]["parameters"].get("normalization")
        if normalization != "tic":
            raise ValueError("Selected checkpoint does not declare the expected TIC input.")
        model.eval().to(device=device, dtype=torch.float32)
        models.append(dict(spec, model=model, fingerprint=fingerprint, state_hash=_state_hash(model), numerical_checked=False))
    # The same realized spectral rows are shared by every representation
    with torch.inference_mode():
        for layout in options["layouts"]:
            classes = 2 if layout == "two_stripes" else options["many_classes"]
            for noise in options["relative_amplitude_noise"]:
                for input_seed in settings["seeds"]:
                    key = f"{layout}_noise_{noise:g}_seed_{input_seed}"
                    fixture = spectral_partition(layout, side=options["side"], classes=classes, mass_axis=axis,
                                                 peak_positions=options["peak_positions"], peak_width_mz=options["peak_width_mz"],
                                                 amplitude_noise=noise, seed=input_seed, device=device)
                    np.savez_compressed(numbers / f"spectral_inputs_{key}.npz", reference=fixture.reference,
                                        pixel_ids=fixture.pixel_ids, xy=fixture.xy, spectra=fixture.spectra,
                                        amplitudes=fixture.amplitudes, prototypes=fixture.prototypes, mass_axis=axis)
                    entries = [dict(family="raw_tic_spectra", repetition=-1, model=None, fingerprint=None), *models]
                    for entry in entries:
                        family, repetition = entry["family"], entry["repetition"]
                        representation = "raw_tic_spectra" if entry["model"] is None else f"{family}_rep_{repetition:02d}"
                        reconstructed = None
                        if entry["model"] is None:
                            values = fixture.spectra
                        else:
                            model = entry["model"]
                            # Numerically verify actual checkpoint outputs before trusting them
                            if not entry["numerical_checked"]:
                                retained = fixture.spectra[:19]
                                first_z, first_r = infer_frozen_encoder(model, retained, device=device, batch_size=19)
                                second_z, second_r = infer_frozen_encoder(model, retained, device=device, batch_size=19)
                                for name, expected, actual in [("latent", first_z, second_z), ("reconstruction", first_r, second_r)]:
                                    np.testing.assert_allclose(actual, expected, atol=settings["numerical_atol"], rtol=settings["numerical_rtol"])
                                    checks.append({"family": family, "repetition": repetition, "check": f"gpu_repeat_{name}",
                                                   "device": device, "compute_dtype": "float32", "max_absolute_error": float(np.max(np.abs(expected-actual))),
                                                   "atol": settings["numerical_atol"], "rtol": settings["numerical_rtol"], "passed": True})
                                assert _state_hash(model) == entry["state_hash"]
                                entry["numerical_checked"] = True
                            values, reconstructed = infer_frozen_encoder(model, fixture.spectra, device=device, batch_size=options["batch_size"])
                            np.savez_compressed(numbers / f"encoder_{key}_{representation}.npz", pixel_ids=fixture.pixel_ids, latent=values)
                        effective_rank = latent_rank_on_cuda(values) if entry["model"] is not None else None
                        predictions, summaries = {}, {}
                        shown = input_seed == options["display_input_seed"] and (entry["model"] is None or repetition == options["display_model_repetition"])
                        for method in options["clustering_methods"]:
                            for cluster_seed in options["clustering_seeds"]:
                                torch.cuda.synchronize()
                                start = time.perf_counter()
                                labels, metadata = cluster_on_cuda(values, method=method, classes=classes,
                                                        seed=cluster_seed, min_samples=options["min_cluster_size"],
                                                        radius_quantile=options["density_radius_quantile"],
                                                        gmm_regularization=options["gmm_regularization"], max_iter=options["max_iter"])
                                torch.cuda.synchronize()
                                elapsed = time.perf_counter() - start
                                score = score_partition(fixture.labels, labels)
                                raster = np.full(fixture.reference.shape, -2, dtype=np.int64)  # (H, W)
                                raster.ravel()[fixture.pixel_ids] = score.aligned
                                boundary, interiors = score_boundaries(fixture.reference, raster, tolerance=settings["boundary_tolerance_pixels"])
                                scored = dict(score.summary, **boundary, **metadata, layout=layout, classes=classes, amplitude_noise=noise,
                                              input_seed=input_seed, cluster_seed=cluster_seed, family=family, model_repetition=repetition,
                                              representation=representation, clustering_method=method, runtime_seconds=elapsed,
                                              input_dimensions=values.shape[1], effective_rank=effective_rank,
                                              reconstruction_mae=float(np.abs(reconstructed-fixture.spectra).mean()) if reconstructed is not None else None)
                                records.append(scored)
                                regions = score.regions.merge(interiors, on="reference_class", how="left")
                                region_records.extend(dict(row, layout=layout, amplitude_noise=noise, input_seed=input_seed,
                                                           family=family, model_repetition=repetition, clustering_method=method,
                                                           cluster_seed=cluster_seed) for row in regions.to_dict("records"))
                                np.savez_compressed(numbers / f"partition_{key}_{representation}_{method}_{cluster_seed}.npz",
                                                    pixel_ids=fixture.pixel_ids, cluster=labels, score_only_aligned=score.aligned)
                                if cluster_seed == options["clustering_seeds"][0]:
                                    predictions[method], summaries[method] = raster, score.summary
                        if shown:
                            title = f"{layout}; classes={classes}; log-amplitude sigma={noise:g}; input seed={input_seed}\n{representation}; cluster seed={options['clustering_seeds'][0]}"
                            stem = f"spectral_{key}_{representation}"
                            for error_map in [False, True]:
                                fig, _ = plot_clustering_methods(fixture.reference, predictions, summaries, title=title, errors=error_map)
                                path = plots / f"{stem}{'_errors' if error_map else ''}.jpeg"
                                fig.savefig(path);plt.close(fig)
                            fig, _ = plot_spectral_region_profiles(axis, fixture.spectra, fixture.labels, title=title, reconstructions=reconstructed)
                            fig.savefig(plots / f"{stem}_spectra.jpeg");plt.close(fig)
                            for label in range(classes):
                                selected = fixture.labels == label
                                inputs = fixture.spectra[selected]
                                mean, low, high = inputs.mean(axis=0), *np.quantile(inputs, [.1,.9], axis=0)
                                prediction = reconstructed[selected].mean(axis=0) if reconstructed is not None else np.full(len(axis),np.nan)
                                profile_records.extend({"layout":layout,"amplitude_noise":noise,"representation":representation,
                                                        "region":label,"mz":float(mz),"input_mean":float(a),"input_q10":float(b),
                                                        "input_q90":float(c),"reconstruction_mean":float(d)}
                                                       for mz,a,b,c,d in zip(axis,mean,low,high,prediction,strict=True))
                            galleries.append({"layout":layout,"noise":noise,"representation":representation,"plot_stem":stem})
                        logger.info("Spectral comparison completed: %s / %s.", key, representation)
        # Evaluation must leave all checkpoint parameters and buffers unchanged
        for entry in models:
            assert _state_hash(entry["model"]) == entry["state_hash"]
            assert ModelLoader.artifact_fingerprint(root / entry["artifact"]) == entry["fingerprint"]
            checks.append({"family":entry["family"],"repetition":entry["repetition"],"check":"checkpoint_and_state_preserved","device":device,"passed":True})
    table = pd.DataFrame(records)
    table.to_csv(numbers / "spectral_partition_scores.csv",index=False)
    pd.DataFrame(region_records).to_csv(numbers / "spectral_partition_regions.csv",index=False)
    pd.DataFrame(profile_records).to_csv(numbers / "spectral_region_profiles.csv",index=False)
    pd.DataFrame(checks).to_csv(numbers / "spectral_model_numerical_checks.csv",index=False)
    pd.DataFrame(galleries).to_csv(numbers / "spectral_partition_gallery.csv",index=False)
    grouped = table.groupby(["layout","amplitude_noise","family","clustering_method"])[["matched_accuracy","macro_dice","boundary_f1","noise_fraction"]].agg(["mean","std"])
    grouped.columns = ["_".join(column) for column in grouped.columns]
    grouped.reset_index().to_csv(numbers / "spectral_partition_summary.csv",index=False)
    fig = distributions(table.assign(label=lambda f:f.family+" / "+f.clustering_method),value="macro_dice",group="label",
                        title="Synthetic spectral partitions: all geometries, noise levels and repetitions")
    fig.savefig(plots / "spectral_partition_distributions.jpeg");plt.close(fig)
    manifest = {"model_axis":options["model_axis"],"input_seeds":settings["seeds"],"cluster_seeds":options["clustering_seeds"],
                "models":[{k:v for k,v in item.items() if k not in {"model","numerical_checked"}} for item in models],
                "inference_device":device,"inference_dtype":"float32","clustering_backend":"torch_cuda","reference_labels_used_for":"scoring_and_descriptive_profiles_only",
                "spectral_generator":"Gaussian peak mixtures with cyclic abundance permutations; independent component lognormal noise",
                "not_actual_pretraining_corpus":True,"historical_checkpoint_comparison_not_controlled_ablation":True}
    _write_json(numbers / "spectral_partition_manifest.json",manifest)
    text = "# Spectral partition comparison\n\nThe same synthetic input spectra are clustered directly and after encoding by six existing trained checkpoints. Reference geometry/labels never enter the encoder or clustering. No models were trained.\n\n"
    text += "## Fixed protocol\n\nInference, spectral mixture generation, clustering and latent effective-rank computation use CUDA float32. CPU/CUDA reference comparisons are excluded. GPU repeatability checks use identical batches and the original frozen atol=1e-5 / rtol=1e-4; a preliminary exact-equality check failed on normal GPU roundoff (maximum 2.8e-9), with its log retained; checkpoint/state hashes are preserved. File decoding, scoring of exported integer labels and figure/report generation use host utilities. Previous CPU clustering and float64 inference artifacts are archived separately.\n\nThree input seeds; three clean-B0 checkpoints; three historical pretraining checkpoints with unfrozen heads. Two log-amplitude noise levels. Reference geometries: two stripes, five stripes, five equal-width radial regions. Circular exterior is unavailable; the small central region tests class imbalance.\n\n"
    text += "K-means and diagonal GMM use the prespecified fixture class count. CUDA DBSCAN infers its count and may assign noise; its radius is the frozen 0.95 quantile of 12-neighbour distances (including self), computed separately for each representation without labels. DBSCAN replaces archived CPU HDBSCAN and is a distinct analysis pipeline. Raw spectra use TIC; learned representations are the actual `latent_space` output, with no projection or reference-dependent scaling. No spatial coordinates enter clustering.\n\n"
    text += "## All scores\n\n" + grouped.reset_index().to_markdown(index=False) + "\n\n![All repetitions](../plots/spectral_partition_distributions.jpeg)\n\n"
    text += "## Same-image method comparison\n\nDisplay input seed and model repetition were selected in settings before evaluation. Every other repetition remains in CSV/NPZ; no best-seed selection occurs. Shared colours come from score-only class matching; grey denotes unmatched/noise and yellow denotes assignment errors.\n\n"
    for item in galleries:
        stem=item["plot_stem"]
        text+=f"### {item['layout']} / sigma={item['noise']} / {item['representation']}\n\n![Reference and all clustering methods](../plots/{stem}.jpeg)\n\n![Assignment errors](../plots/{stem}_errors.jpeg)\n\n![Observed spectral distributions and decoder profiles](../plots/{stem}_spectra.jpeg)\n\n"
    text += "## Model numerical verification\n\n"+pd.DataFrame(checks).to_markdown(index=False)+"\n\n"
    text += "## Interpretation limits\n\nThese Gaussian-mixture images diagnose separation, noise rejection, small regions, boundaries and representation sensitivity. They are not tissue or molecular ground truth and do not reproduce the pretraining corpus. Historical model families differ in data/objectives/budgets; differences are not a causal pretraining ablation. A method performing well here must also be validated on compatible biological data. Noise coverage, per-region errors and convergence must be considered alongside average Dice.\n"
    (reports / "spectral_partition_comparison.md").write_text(text)
    del models
    gc.collect()
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    _provenance(settings,output,device,"spectral_partitions")
