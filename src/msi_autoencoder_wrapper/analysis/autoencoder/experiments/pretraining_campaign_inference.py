"""Inference level of the pretraining-campaign cache: one evaluation per model and population.

Every selected checkpoint is evaluated once on every decoded population. The level
stores per-pixel reconstruction metrics with their exact m/z-window decomposition,
latent codes, ranking tables (whole axis, per held-out image and per class stratum),
held-out probability maps and ion intensities, spectrum cases and the encoder's angular
sensitivity. After every model of an axis is evaluated, the representative repetition of
every model alias (one baseline or one variant x stage cell) is selected
(:func:`representative_ranking`) and only its held-out reconstruction is stored for
display.
"""

from __future__ import annotations

import gc
import json
import os
import time
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
import pandas as pd
import torch

from ....models.model_loader import ModelLoader
from ....utils.logger import get_custom_logger
from ..heads.predictive_comparison import ranking_tables
from ..latent.sphere_geometry import angular_sensitivity_curve, canonicalize, encoder_layer_norm_parameters
from ..reconstruction.metrics import reconstruction_metrics
from ..reconstruction.windowed import windowed_masserstein
from ..spatial.ion_images import ion_intensities
from . import pretraining_campaign as campaign
from .predictive_precompute import resolve_device
from .pretraining_campaign_precompute import (
    PIXEL_METRICS,
    WINDOW_ARRAYS,
    _atomic_json,
    _cache,
    _source_key,
    axis_directory,
)

logger = get_custom_logger(__name__)

#: Package-relative sources that determine how a loaded checkpoint is evaluated.
INFERENCE_SOURCES = ("models/architectures", "models/model_loader.py", "metrics", "normalization",
                     "analysis/autoencoder/reconstruction/metrics.py", "analysis/autoencoder/reconstruction/windowed.py",
                     "analysis/autoencoder/heads/predictive_comparison.py",
                     "analysis/autoencoder/latent/sphere_geometry.py", "analysis/autoencoder/spatial/ion_images.py",
                     "analysis/autoencoder/experiments/pretraining_campaign_inference.py")

#: Class groupings whose strata receive the complete ranking-metric set.
STRATUM_GROUPINGS = ("class_set", "mz_window", "train_support", "heldout_presence")
#: Training-support buckets (left-closed) of :func:`class_strata`.
SUPPORT_EDGES = (0, 1, 10, 100, 1000, np.inf)
SUPPORT_LABELS = ("0", "1-9", "10-99", "100-999", "1000+")
#: Default selection of the representative model; ``settings.representative_model`` overrides it.
REPRESENTATIVE_DEFAULTS = {
    "population": "test", "evaluation": "annotation_retrieval", "scope": "train_supported",
    "metrics": {"masserstein": "lower", "spectral_angle": "lower", "average_precision": "higher",
                "micro_average_precision": "higher", "f1": "higher"},
    "tie_breaker": "average_precision",
}


#: Rows per population re-evaluated when a legacy inference directory is validated.
VALIDATION_ROWS = 64
#: Absolute tolerance of ``tic_error`` (float32 rounding of a unit-sum spectrum).
VALIDATION_TIC_TOLERANCE = 1e-5


def _forward(model: Any, spectra: np.ndarray, head: str, device: torch.device) -> dict[str, np.ndarray]:
    """One evaluation-mode forward pass on a small batch."""
    with torch.inference_mode():
        outputs = model(torch.as_tensor(np.array(spectra, dtype=np.float32), device=device))
    result = {"logits": outputs[f"head_{head}"], "latent": outputs["latent_space"],
              "reconstruction": outputs["reconstruction"]}
    for name, values in result.items():
        if not bool(torch.isfinite(values).all()):
            raise ValueError(f"Non-finite {name} output.")
    return {name: values.detach().float().cpu().numpy() for name, values in result.items()}


def _evaluate_population(model: Any, row: Any, arrays: dict[str, np.ndarray], meta: dict[str, np.ndarray],
                         settings: dict, device: torch.device, *, groups: np.ndarray | None = None,
                         ion_supports: list[np.ndarray] | None = None) -> dict[str, Any]:
    """Evaluate one model on one decoded population in bounded batches.

    :return: Per-pixel arrays, logits, per-bin error aggregates (overall and per group)
        and optional ion intensities.
    :rtype: dict[str, typing.Any]
    """
    spectra = arrays["spectra"]
    count, width = spectra.shape
    batch_size = int(settings["batch_size"])
    options = json.loads(row.masserstein_params_json)
    windows = settings["windows"]
    group_values = np.zeros(count, dtype=object) if groups is None else groups
    group_names = list(dict.fromkeys(group_values.tolist()))
    sums = {group: {"abs": np.zeros(width), "signed": np.zeros(width), "input": np.zeros(width),
                    "output": np.zeros(width), "pixels": 0} for group in ["all", *group_names]}
    collected: dict[str, list] = {key: [] for key in ("logits", "latent", *PIXEL_METRICS, *WINDOW_ARRAYS)}
    ions = []
    for start in range(0, count, batch_size):
        stop = min(start + batch_size, count)
        inputs = np.array(spectra[start:stop], dtype=np.float32)  # (B, M); copied out of the memory map
        output = _forward(model, inputs, row.head, device)
        reconstruction = output["reconstruction"]  # (B, M)
        ### Pixel metrics: training Masserstein with its exact window decomposition
        local = windowed_masserstein(inputs, reconstruction, meta["mass_axis"], meta["window_edges"],
                                     minimum_window_mass=float(windows.get("minimum_mass", 1e-3)),
                                     batch_size=batch_size, device=device, criterion_options=options)
        metrics = reconstruction_metrics(inputs, reconstruction)
        collected["masserstein"].append(local.total)
        for name in PIXEL_METRICS[1:]:
            collected[name].append(np.asarray(metrics[name], dtype=np.float32))
        for name in WINDOW_ARRAYS:
            collected[name].append(getattr(local, name))
        collected["logits"].append(output["logits"])
        collected["latent"].append(output["latent"])
        ### Per-bin error aggregates, overall and per group (e.g. held-out image)
        residual = reconstruction - inputs  # (B, M)
        batch_groups = group_values[start:stop]
        for group in ["all", *group_names]:
            selected = slice(None) if group == "all" else batch_groups == group
            target = sums[group]
            target["abs"] += np.abs(residual[selected]).sum(axis=0)
            target["signed"] += residual[selected].sum(axis=0)
            target["input"] += inputs[selected].sum(axis=0)
            target["output"] += reconstruction[selected].sum(axis=0)
            target["pixels"] += int(np.count_nonzero(selected)) if group != "all" else stop - start
        if ion_supports is not None:
            ions.append(ion_intensities(reconstruction, ion_supports))
    result = {key: np.concatenate(values, axis=0) for key, values in collected.items()}
    features = []
    for group, values in sums.items():
        pixels = max(values["pixels"], 1)
        features.append(pd.DataFrame({"group": group, "bin": np.arange(width), "mz": meta["mass_axis"],
                                      "mean_abs_error": values["abs"] / pixels,
                                      "mean_signed_error": values["signed"] / pixels,
                                      "mean_input": values["input"] / pixels,
                                      "mean_output": values["output"] / pixels, "pixels": values["pixels"]}))
    result["features"] = pd.concat(features, ignore_index=True)
    if ion_supports is not None:
        result["ions"] = np.concatenate(ions, axis=0)
    return result


def _ranking(logits: np.ndarray, targets: np.ndarray, states: np.ndarray, train_counts: np.ndarray,
             class_names: tuple[str, ...], row: Any, device: torch.device) -> dict:
    """Ranking tables of one head on prepared (row- and column-selected) arrays.

    :param logits: Head outputs, shape ``(N, C)``.
    :param targets: Available binary annotations as float, shape ``(N, C)``.
    :param states: Evidence states, shape ``(N, C)``.
    :return: ``prediction``, ``per_class`` and ``score_diagnostics`` tables.
    :rtype: dict
    """
    tables = ranking_tables(logits, targets, states, train_counts, class_names, device=device, family=row.family)
    return {key: tables[key] for key in ("prediction", "per_class", "score_diagnostics")}


def _ranking_inputs(logits: dict[str, np.ndarray], populations: dict) -> dict[str, tuple[np.ndarray, ...]]:
    """Logits, available targets and states of every prediction population.

    ``test_combined`` pools the test pixels with the visible test-extension pixels.

    :return: Population name mapped to ``(logits, targets, states)``, each ``(N, C)``.
    :rtype: dict[str, tuple[numpy.ndarray, ...]]
    """
    inputs = {}
    for name in ("train", "test", "test_extended", "heldout_image"):
        arrays = populations[name]
        targets = np.asarray(arrays["targets"]).astype(np.float32) * np.asarray(arrays["mask"])  # (N, C)
        inputs[name] = (logits[name], targets, np.asarray(arrays["states"]))
    visible = np.asarray(populations["test_extended"]["visible"], dtype=bool)
    inputs["test_combined"] = tuple(np.concatenate([test, extended[visible]])
                                    for test, extended in zip(inputs["test"], inputs["test_extended"]))
    return inputs


def class_strata(classes: pd.DataFrame, heldout_classes: Iterable[str], width: float) -> pd.DataFrame:
    """Stratum labels of every head class for :data:`STRATUM_GROUPINGS`.

    :param classes: Per-axis class catalogue (``class_name``, ``mz``, ``bin_mz``,
        ``train_positives``, ``class_set``).
    :type classes: pandas.DataFrame
    :param heldout_classes: Class names annotated in at least one held-out image.
    :type heldout_classes: collections.abc.Iterable[str]
    :param width: Width of the m/z windows.
    :type width: float
    :return: Catalogue with ``mz_window_lower``, ``mz_window``, ``train_support`` and
        ``heldout_presence``; missing class sets become ``unassigned``.
    :rtype: pandas.DataFrame
    """
    frame = classes.copy()
    mz = frame.mz.fillna(frame.bin_mz)
    frame["mz_window_lower"] = np.floor(mz / width) * width
    frame["mz_window"] = [f"{lower:.0f}-{lower + width:.0f}" for lower in frame.mz_window_lower]
    frame["train_support"] = pd.cut(frame.train_positives, np.asarray(SUPPORT_EDGES, dtype=float), right=False,
                                    labels=list(SUPPORT_LABELS)).astype(str)
    frame["heldout_presence"] = np.where(frame.class_name.isin(set(heldout_classes)), "in_heldout", "not_in_heldout")
    frame["class_set"] = frame.class_set.fillna("unassigned")
    return frame


def representative_ranking(settings: dict, models: pd.DataFrame, directory: Callable[[str], Path]) -> pd.DataFrame:
    """Rank the repetitions of one axis and mark its representative model.

    Every configured metric ranks the models (1 = best, average ranks for ties) on the
    configured population, which defaults to the withheld test pixels so that the model
    shown on the held-out images is not chosen on those images. The representative has
    the lowest mean rank; the tie breaker metric decides equal mean ranks. Pixel metrics
    are population means of the per-pixel values; the others are read from the
    whole-axis ranking table (``evaluation`` population, ``scope`` class scope).

    :param settings: Analysis settings (``representative_model`` overrides the defaults).
    :type settings: dict
    :param models: Evaluated models of one axis.
    :type models: pandas.DataFrame
    :param directory: Maps a model identifier to its inference directory.
    :type directory: collections.abc.Callable[[str], pathlib.Path]
    :return: One row per model and metric with ``value``, ``rank``, ``mean_rank`` and
        ``representative``.
    :rtype: pandas.DataFrame
    """
    options = {**REPRESENTATIVE_DEFAULTS, **settings.get("representative_model", {})}
    rows = []
    for row in models.itertuples():
        with np.load(directory(row.model_id) / f"{options['population']}_pixels.npz") as archive:
            pixel_means = {name: float(np.mean(archive[name])) for name in options["metrics"] if name in archive.files}
        prediction = pd.read_csv(directory(row.model_id) / "prediction.csv")
        ranking = prediction[(prediction.evaluation == options["population"]) & (prediction.group == "all")
                             & (prediction.population == options["evaluation"])
                             & (prediction.scope == options["scope"])].set_index("metric").value
        for metric, direction in options["metrics"].items():
            value = pixel_means[metric] if metric in pixel_means else float(ranking.get(metric, np.nan))
            rows.append({"model_id": row.model_id, "repetition": int(row.repetition), "metric": metric,
                         "direction": direction, "value": value})
    frame = pd.DataFrame(rows)
    signed = np.where(frame.direction.eq("lower"), frame.value, -frame.value)
    frame["rank"] = frame.assign(signed=signed).groupby("metric").signed.rank(method="average")
    mean_rank = frame.groupby("model_id")["rank"].mean()
    tie = frame[frame.metric == options["tie_breaker"]].set_index("model_id")["rank"]
    order = pd.DataFrame({"mean_rank": mean_rank, "tie": tie.reindex(mean_rank.index)}).sort_values(["mean_rank", "tie"])
    frame["mean_rank"] = frame.model_id.map(mean_rank)
    frame["representative"] = frame.model_id.eq(order.index[0])
    return frame


def _write_display(row: Any, spectra: np.ndarray, bins: np.ndarray, directory: Path, settings: dict,
                   device: torch.device) -> None:
    """Store the representative model's held-out reconstruction restricted to the display bins.

    :param spectra: Held-out input spectra, shape ``(N_h, M)``.
    :param bins: Display bins, shape ``(K,)``.
    """
    path = directory / "display_reconstruction.npy"
    if path.is_file() and np.load(path, mmap_mode="r").shape == (len(spectra), bins.size):
        logger.info("Reusing display reconstruction of %s.", row.model_id)
        return
    logger.info("Writing display reconstruction of the representative model %s.", row.model_id)
    model = ModelLoader.load_artifact(row.artifact, strict=True)[0].to(device).eval()
    batch_size = int(settings["batch_size"])
    blocks = [_forward(model, spectra[start:start + batch_size], row.head, device)["reconstruction"][:, bins]
              .astype(np.float16) for start in range(0, len(spectra), batch_size)]  # [(B, K)]
    temporary = path.with_suffix(".tmp.npy")
    np.save(temporary, np.concatenate(blocks, axis=0))  # (N_h, K)
    os.replace(temporary, path)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()


def run_inference(context: Any) -> None:
    """Evaluate every selected model once on every decoded population.

    :param context: Precompute context of the common runner.
    :type context: msi_autoencoder_wrapper.analysis.precompute.core.context.AnalysisContext
    """
    from . import pretraining_campaign_cache as contracts

    cache = _cache(context)
    settings = cache.settings
    device = resolve_device(settings, allow_cpu=context.allow_cpu)
    contract = contracts.inference_contract(settings, cache.contracts["populations"])
    level = contracts.resolve_level(settings, "inference", contract,
                                    lambda directory: validate_legacy_inference(directory, cache, device))
    root = level.directory
    key = root.name
    cache.inference_root, cache.keys["inference"] = root, key
    cache.contracts["inference"] = contracts.normalized(contract)
    pending = [row.model_id for row in cache.models.itertuples() if not _evaluated(root / row.model_id, row)]
    logger.info("Inference: %s of %s models already evaluated, %s to evaluate.", len(cache.models) - len(pending),
                len(cache.models), len(pending))
    started, evaluated_now = time.monotonic(), 0
    pixels = cache.population_frame()
    heldout_groups = pixels[pixels.population == "heldout_image"].sort_values("row").dataset_id.to_numpy(object)
    case_count = int(settings.get("cases", {}).get("per_population", 6))
    width = float(settings["windows"]["width"])
    history_split = "test"

    for axis in cache.axes:
        meta = cache.axis_meta(axis)
        classes = cache.axis_table(axis, "classes")
        class_names = tuple(classes.class_name)
        train_counts = classes.train_positives.to_numpy()
        ions = cache.axis_table(axis, "heldout_ions")
        supports = [np.asarray(json.loads(value), dtype=np.int64) for value in ions.bins]
        display_bins = np.asarray(cache.axis_array(axis, "display_bins"))
        strata = class_strata(classes, ions.class_name, width)
        populations = {name: cache.population(axis, name) for name in campaign.POPULATIONS}
        for row in cache.axis_models(axis).itertuples():
            directory = root / row.model_id
            if row.model_id not in pending:
                logger.debug("Reusing inference of %s.", row.model_id)
                continue
            artifact_key = ModelLoader.artifact_fingerprint(row.artifact)
            marker = directory / "complete.json"
            directory.mkdir(parents=True, exist_ok=True)
            marker.unlink(missing_ok=True)
            model_started = time.monotonic()
            logger.info("Evaluating %s on %s (%s/%s).", row.model_id, device, evaluated_now + 1, len(pending))
            model = ModelLoader.load_artifact(row.artifact, strict=True)[0].to(device).eval()
            gamma, beta = encoder_layer_norm_parameters(model)
            logits_by_population = {}
            summary = {"model_id": row.model_id, "axis": axis, "repetition": int(row.repetition), "head": row.head,
                       "family": row.family, "artifact_sha256": artifact_key,
                       "gamma": gamma.tolist(), "beta": beta.tolist(), "populations": {}}
            for name, arrays in populations.items():
                heldout = name == "heldout_image"
                evaluated = _evaluate_population(
                    model, row, arrays, meta, settings, device,
                    groups=heldout_groups if heldout else None,
                    ion_supports=supports if heldout else None)
                np.savez(directory / f"{name}_pixels.npz", source_ids=np.asarray(arrays["source_ids"]),
                         latent=evaluated["latent"], **{key: evaluated[key] for key in (*PIXEL_METRICS, *WINDOW_ARRAYS)})
                evaluated["features"].to_csv(directory / f"{name}_features.csv", index=False)
                logits_by_population[name] = evaluated["logits"]
                summary["populations"][name] = {"pixels": int(len(evaluated["masserstein"])),
                                                "masserstein_mean": float(evaluated["masserstein"].mean())}
                if heldout:
                    np.save(directory / "heldout_image_probabilities.npy",
                            (1.0 / (1.0 + np.exp(-evaluated["logits"]))).astype(np.float16))
                    np.save(directory / "heldout_image_ions.npy", evaluated["ions"])
                elif name in ("test", "test_extended"):
                    np.save(directory / f"{name}_logits.npy", evaluated["logits"])
                ### Spectrum cases: fixed rows and this model's best and worst Masserstein rows
                fixed = np.load(cache.population_root / f"case_rows_{name}.npy")
                order = np.argsort(evaluated["masserstein"])
                best, worst = order[:case_count], order[-case_count:]
                case_rows = np.unique(np.concatenate([fixed, best, worst]))
                output = _forward(model, np.asarray(arrays["spectra"])[case_rows], row.head, device)
                np.savez(directory / f"{name}_cases.npz", rows=case_rows, fixed=np.isin(case_rows, fixed),
                         best=np.isin(case_rows, best), worst=np.isin(case_rows, worst),
                         input=np.asarray(arrays["spectra"])[case_rows], output=output["reconstruction"])
                del evaluated
                gc.collect()

            # Encoder angular sensitivity on fixed rows with fixed perturbation directions
            sensitivity = _angular_sensitivity(model, row, populations, gamma, beta, settings, device)
            sensitivity.to_csv(directory / "sensitivity.csv", index=False)

            # Ranking tables per prediction population, per held-out image and per class stratum
            tables: dict[str, list[pd.DataFrame]] = {"prediction": [], "per_class": [], "score_diagnostics": []}

            def add(population: str, group: str, computed: dict) -> None:
                for table, frame in computed.items():
                    tables[table].append(frame.assign(evaluation=population, group=group))

            inputs = _ranking_inputs(logits_by_population, populations)
            for name, values in inputs.items():
                add(name, "all", _ranking(*values, train_counts, class_names, row, device))
            images = {dataset_id: np.flatnonzero(heldout_groups == dataset_id)
                      for dataset_id in dict.fromkeys(heldout_groups.tolist())}
            for dataset_id, rows in images.items():
                add("heldout_image", dataset_id, _ranking(*(values[rows] for values in inputs["heldout_image"]),
                                                          train_counts, class_names, row, device))
            for table, frames in tables.items():
                pd.concat(frames, ignore_index=True).to_csv(directory / f"{table}.csv", index=False)
            ## Complete metric set per class stratum (whole populations; per image for class sets)
            stratum_frames = []
            for grouping in STRATUM_GROUPINGS:
                labels = strata[grouping].to_numpy(object)
                for stratum in pd.unique(labels):
                    columns = np.flatnonzero(labels == stratum)
                    names = tuple(class_names[column] for column in columns)
                    scopes = [(name, "all", values) for name, values in inputs.items()]
                    if grouping == "class_set":
                        scopes += [("heldout_image", dataset_id, tuple(values[rows] for values in inputs["heldout_image"]))
                                   for dataset_id, rows in images.items()]
                    for name, group, values in scopes:
                        computed = _ranking(*(array[:, columns] for array in values), train_counts[columns], names,
                                            row, device)["prediction"]
                        stratum_frames.append(computed[computed.scope == "train_supported"].assign(
                            evaluation=name, group=group, grouping=grouping, stratum=stratum))
            pd.concat(stratum_frames, ignore_index=True).to_csv(directory / "strata_prediction.csv", index=False)
            del inputs

            ## History consistency: final test Masserstein recorded during training
            history = campaign.read_history(row.artifact)
            recorded = [record for record in history if record.get("split") == history_split]
            summary["history_test_masserstein"] = (float(recorded[-1]["metrics"]["masserstein"])
                                                   if recorded else None)
            summary["evaluation_seconds"] = time.monotonic() - model_started
            _atomic_json(directory / "summary.json", summary)
            _atomic_json(marker, {"artifact_sha256": artifact_key})
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
            ## Progress with the remaining-time estimate of this run
            evaluated_now += 1
            elapsed = time.monotonic() - started
            logger.info("Evaluated %s in %.0f s; %s/%s done, about %.1f h remaining.", row.model_id,
                        summary["evaluation_seconds"], evaluated_now, len(pending),
                        elapsed / evaluated_now * (len(pending) - evaluated_now) / 3600.0)

        # Representative repetition of every model alias and its display reconstruction
        models = cache.axis_models(axis)
        for alias, frame in models.groupby("model_alias", sort=False):
            ranking = representative_ranking(settings, frame, cache.model_directory)
            ranking.to_csv(root / f"representative__cell__{alias}.csv", index=False)
            chosen_id = ranking.model_id[ranking.representative].iloc[0]
            chosen = next(row for row in frame.itertuples() if row.model_id == chosen_id)
            logger.info("Representative of %s: %s (mean rank %.2f).", alias, chosen_id,
                        float(ranking.mean_rank[ranking.representative].iloc[0]))
            _write_display(chosen, populations["heldout_image"]["spectra"], display_bins,
                           cache.model_directory(chosen_id), settings, device)
        ## The part-1 reports read the baseline representative under the axis name
        baseline = models[models.role == settings["baseline_role"]]
        if not baseline.empty:
            representative_ranking(settings, baseline, cache.model_directory).to_csv(
                root / f"representative__{axis_directory(settings, axis)}.csv", index=False)
        del populations
        gc.collect()
    _atomic_json(root / "complete.json", {"key": key, "populations": cache.keys["populations"],
                                          "models": cache.models.model_id.tolist(),
                                          "sources_sha256": _source_key(settings, INFERENCE_SOURCES)})


def _angular_sensitivity(model: Any, row: Any, populations: dict, gamma: np.ndarray, beta: np.ndarray,
                         settings: dict, device: torch.device) -> pd.DataFrame:
    """Mean latent angle moved per relative input perturbation, on test and held-out rows.

    The evaluated rows and the random perturbation directions depend only on the
    configured seed, so every model is perturbed identically (paired comparison).
    """
    latent = settings.get("latent", {})
    seed = int(latent.get("seed", 42))
    size = int(latent.get("sensitivity_sample_size", 1000))
    epsilons = tuple(float(value) for value in latent.get("epsilons", (0.001, 0.003, 0.01, 0.03, 0.1)))
    batch_size = int(settings["batch_size"])

    def encode(batch: np.ndarray) -> np.ndarray:
        codes = [_forward(model, batch[start:start + batch_size], row.head, device)["latent"]
                 for start in range(0, len(batch), batch_size)]
        return canonicalize(np.concatenate(codes), gamma, beta)  # (S, D)

    frames = []
    for position, name in enumerate(("test", "heldout_image")):
        spectra = populations[name]["spectra"]
        rows = np.sort(np.random.default_rng([seed, position]).choice(len(spectra), min(size, len(spectra)),
                                                                       replace=False))
        curve = angular_sensitivity_curve(encode, np.asarray(spectra[rows], dtype=np.float32), epsilons,
                                          np.random.default_rng([seed, 3, position]))
        frames.append(pd.DataFrame({"population": name, "epsilon": curve["epsilon"],
                                    "mean_angle_degrees": curve["mean_angle_degrees"], "rows": rows.size}))
    return pd.concat(frames, ignore_index=True)


def array_agreement(recomputed: dict[str, np.ndarray], stored: dict[str, np.ndarray], *, relative: float = 1e-4,
                    absolute: float = 1e-6) -> tuple[list[str], dict[str, float], dict[str, float]]:
    """Compare recomputed per-pixel arrays with stored ones.

    Undefined entries (``nan``, e.g. the within-window W1 of a window without mass) must
    coincide exactly; defined entries must agree within ``relative * scale + absolute``,
    where ``scale`` is the largest stored magnitude of the array.

    REMARK: ``tic_error`` is zero by construction (TIC-normalized decoder output), so its
    stored values are float32 rounding noise and it uses :data:`VALIDATION_TIC_TOLERANCE`
    as absolute tolerance.

    :param recomputed: Arrays computed now, keyed by name.
    :type recomputed: dict[str, numpy.ndarray]
    :param stored: Stored arrays on the same rows.
    :type stored: dict[str, numpy.ndarray]
    :param relative: Relative tolerance.
    :type relative: float
    :param absolute: Absolute tolerance of every array but ``tic_error``.
    :type absolute: float
    :return: Failing array names, maximum absolute differences and scales.
    :rtype: tuple[list[str], dict[str, float], dict[str, float]]
    """
    failed, differences, scales = [], {}, {}
    for key, values in recomputed.items():
        left, right = np.asarray(values, dtype=np.float64), np.asarray(stored[key], dtype=np.float64)
        defined = ~np.isnan(right)
        differences[key] = float(np.max(np.abs(left[defined] - right[defined]))) if defined.any() else 0.0
        scales[key] = (float(np.max(np.abs(right[defined]))) if defined.any() else 0.0) or 1.0
        tolerance = relative * scales[key] + (VALIDATION_TIC_TOLERANCE if key == "tic_error" else absolute)
        if not np.array_equal(np.isnan(left), ~defined) or differences[key] > tolerance:
            failed.append(key)
    return failed, differences, scales


def _evaluated(directory: Path, row: Any) -> bool:
    """Whether a model directory holds a complete evaluation of the current weights."""
    marker = directory / "complete.json"
    return (marker.is_file() and json.loads(marker.read_text()).get("artifact_sha256")
            == ModelLoader.artifact_fingerprint(row.artifact))


def validate_legacy_inference(root: Path, cache: Any, device: torch.device) -> pd.DataFrame:
    """Re-evaluate every model of a legacy inference directory on sampled rows.

    For every evaluated model the current code re-computes, on a seeded sample of test
    and held-out rows, the reconstruction metrics, the windowed Masserstein arrays, the
    latent codes and the head outputs and compares them with the stored arrays.
    Tolerances cover single-precision GPU reductions over differently sized batches.
    The sensitivity grid and the fixed case rows must match the settings exactly.

    :param root: Legacy inference directory.
    :type root: pathlib.Path
    :param cache: Cache with the adopted population level.
    :type cache: CampaignCache
    :param device: Evaluation device.
    :type device: torch.device
    :return: Check rows (``check``, ``subject``, ``passed``, ``detail``).
    :rtype: pandas.DataFrame
    """
    settings = cache.settings
    rows: list[dict] = []
    complete = json.loads((root / "complete.json").read_text())
    rows.append({"check": "populations", "subject": root.name,
                 "passed": complete.get("populations") == cache.keys["populations"],
                 "detail": f"built on {complete.get('populations')}, current {cache.keys['populations']}"})
    latent = settings.get("latent", {})
    epsilons = [float(value) for value in latent.get("epsilons", (0.001, 0.003, 0.01, 0.03, 0.1))]
    seed = int(latent.get("seed", 42))
    for row in cache.models.itertuples():
        directory = root / row.model_id
        if not _evaluated(directory, row):
            continue
        meta = cache.axis_meta(row.axis)
        options = json.loads(row.masserstein_params_json)
        model = ModelLoader.load_artifact(row.artifact, strict=True)[0].to(device).eval()
        for position, name in enumerate(("test", "heldout_image")):
            arrays = cache.population(row.axis, name)
            with np.load(directory / f"{name}_pixels.npz") as archive:
                stored = {key: archive[key] for key in archive.files}
            sample = np.sort(np.random.default_rng([seed, 91, position]).choice(
                len(stored["masserstein"]), size=min(VALIDATION_ROWS, len(stored["masserstein"])), replace=False))
            inputs = np.asarray(arrays["spectra"][sample], dtype=np.float32)  # (S, M)
            output = _forward(model, inputs, row.head, device)
            local = windowed_masserstein(inputs, output["reconstruction"], meta["mass_axis"], meta["window_edges"],
                                         minimum_window_mass=float(settings["windows"].get("minimum_mass", 1e-3)),
                                         batch_size=VALIDATION_ROWS, device=device, criterion_options=options)
            metrics = reconstruction_metrics(inputs, output["reconstruction"])
            recomputed = {"masserstein": local.total, "latent": output["latent"],
                          **{key: getattr(local, key) for key in WINDOW_ARRAYS},
                          **{key: np.asarray(metrics[key]) for key in PIXEL_METRICS[1:]}}
            failed, differences, scales = array_agreement(recomputed, {key: stored[key][sample] for key in recomputed})
            passed = not failed
            if name == "test":
                logits = np.asarray(np.load(directory / "test_logits.npy", mmap_mode="r")[sample])
                head_difference = float(np.max(np.abs(output["logits"] - logits)))
                head_passed = head_difference <= 1e-3 * max(float(np.max(np.abs(logits))), 1.0)
            else:
                probabilities = np.asarray(np.load(directory / "heldout_image_probabilities.npy",
                                                   mmap_mode="r")[sample], dtype=np.float64)
                head_difference = float(np.max(np.abs(1.0 / (1.0 + np.exp(-output["logits"])) - probabilities)))
                head_passed = head_difference <= 2e-3  # float16 storage
            worst = max(differences, key=lambda key: differences[key] / scales[key])
            rows.append({"check": "pixel_outputs", "subject": f"{row.model_id}/{name}", "passed": passed,
                         "detail": f"{sample.size} rows; failing: {failed}" if failed else
                                   f"{sample.size} rows; largest relative difference {differences[worst] / scales[worst]:.2e} "
                                   f"({worst})"})
            rows.append({"check": "head_outputs", "subject": f"{row.model_id}/{name}", "passed": head_passed,
                         "detail": f"max abs difference {head_difference:.2e}"})
        sensitivity = pd.read_csv(directory / "sensitivity.csv")
        rows.append({"check": "sensitivity_grid", "subject": row.model_id,
                     "passed": sorted(set(sensitivity.epsilon.round(12))) == sorted(round(value, 12) for value in epsilons),
                     "detail": f"epsilons {sorted(set(sensitivity.epsilon))}"})
        for population in campaign.POPULATIONS:
            with np.load(directory / f"{population}_cases.npz") as archive:
                fixed = archive["rows"][archive["fixed"]]
            expected = np.load(cache.population_root / f"case_rows_{population}.npy")
            rows.append({"check": "case_rows", "subject": f"{row.model_id}/{population}",
                         "passed": np.array_equal(np.sort(fixed), np.sort(expected)), "detail": ""})
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return pd.DataFrame(rows)
