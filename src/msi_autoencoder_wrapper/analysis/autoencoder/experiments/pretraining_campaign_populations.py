"""Population level of the pretraining-campaign cache: decoded pixels and references.

Every population is decoded once per spectral axis through the training dataset's
reader, binner and normalization and stored as memory-mappable arrays together with
targets, evidence states, pixel identities and coordinates, the class catalogue, the
aligned METASPACE ion images of the held-out datasets and the display subset.
"""

from __future__ import annotations

import gc
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from ....data.annotation_evidence import IonCatalogue, SignalEvidencePolicy
from ....utils.logger import get_custom_logger
from ..reconstruction.windowed import window_assignment, window_edges
from ..spatial.coordinates import MergedPixelMap, imzml_coordinates
from ..spatial.ion_images import ion_bin_indices, ion_intensities, spatial_agreement
from . import pretraining_campaign as campaign
from .pretraining_campaign_precompute import (
    CampaignCache,
    _atomic_json,
    _cache,
    _source_key,
    axis_directory,
    provenance_record,
    reference_task_ids,
)

logger = get_custom_logger(__name__)

#: Package-relative sources that determine how populations are decoded.
POPULATION_SOURCES = ("data", "readers", "binners", "normalization", "models/datasets", "runtime/workflows",
                      "analysis/autoencoder/experiments/pretraining_campaign.py",
                      "analysis/autoencoder/experiments/pretraining_campaign_populations.py",
                      "analysis/autoencoder/reconstruction/windowed.py",
                      "analysis/autoencoder/spatial/coordinates.py", "analysis/autoencoder/spatial/ion_images.py")


def _save_population(directory: Path, arrays: dict[str, np.ndarray]) -> None:
    """Persist one population as separate ``.npy`` files (memory-mappable)."""
    directory.mkdir(parents=True, exist_ok=True)
    for name, values in arrays.items():
        np.save(directory / f"{name}.npy", np.ascontiguousarray(values))


def _classify_states(spectra: np.ndarray, targets: np.ndarray, mask: np.ndarray,
                     catalogue: IonCatalogue, policy: SignalEvidencePolicy, chunk: int = 8192) -> np.ndarray:
    """Evidence states P/N/U (1/0/2, -1 unavailable) in bounded chunks, shape ``(N, C)``."""
    states = np.empty(targets.shape, dtype=np.int8)
    for start in range(0, len(spectra), chunk):
        stop = min(start + chunk, len(spectra))
        states[start:stop] = policy.classify(
            torch.as_tensor(np.asarray(spectra[start:stop])), torch.as_tensor(targets[start:stop].astype(np.float32)),
            torch.as_tensor(mask[start:stop]), catalogue).numpy()
    return states


def _heldout_datasets(settings: dict, parameters: dict) -> list[str]:
    """Held-out dataset identifiers: explicit setting or the campaign's cohort exclusion."""
    configured = settings["populations"].get("heldout_dataset_ids")
    if configured:
        return list(configured)
    selection = parameters["factory_parameters"].get("cohort_selection") or {}
    excluded = (selection.get("parameters") or {}).get("excluded_dataset_ids") or []
    if not excluded:
        raise ValueError("No held-out datasets: set populations.heldout_dataset_ids or a cohort exclusion.")
    return list(excluded)


#: Rows per population re-decoded when a legacy population directory is validated.
VALIDATION_ROWS = 64


def run_populations(context: Any) -> None:
    """Decode every population once per axis and derive all model-independent tables.

    :param context: Precompute context of the common runner.
    :type context: msi_autoencoder_wrapper.analysis.precompute.core.context.AnalysisContext
    """
    from . import pretraining_campaign_cache as contracts

    cache = _cache(context)
    settings = cache.settings
    contract = contracts.populations_contract(settings, cache.contracts["plan"])
    level = contracts.resolve_level(settings, "populations", contract,
                                    lambda directory: validate_legacy_populations(directory, cache))
    root = level.directory
    cache.population_root, cache.keys["populations"] = root, root.name
    cache.contracts["populations"] = contracts.normalized(contract)
    if level.complete:
        logger.info("Reusing decoded populations %s.", root)
        return
    key = root.name
    target_field = settings["target_field"]
    batch_size = int(settings["batch_size"])
    parameters = cache.plan["parameters"]
    representative = reference_task_ids(cache)
    first_parameters = parameters[representative[cache.axes[0]]]

    # Population identities
    pixel_map = MergedPixelMap.from_store(settings["merged_store"])
    heldout_datasets = _heldout_datasets(settings, first_parameters)
    heldout_candidates = np.concatenate([np.arange(start, stop, dtype=np.int64)
                                         for start, stop in pixel_map.dataset_ranges(heldout_datasets)])
    assignments = campaign.split_assignments(first_parameters)
    subset_ids = np.unique(np.concatenate(list(assignments.values())))
    policy = SignalEvidencePolicy(**settings["evidence"])
    logger.info("Populations: train=%s test=%s heldout candidates=%s (%s).", assignments["train"].size,
                assignments["test"].size, heldout_candidates.size, heldout_datasets)

    # Per-axis decoding of train, test and held-out candidates
    state: dict[str, dict] = {}
    test_positives: dict[str, int] = {}
    candidates: dict[str, np.ndarray] = {}
    for axis in cache.axes:
        logger.info("Decoding populations of axis %s.", axis)
        task_parameters = parameters[representative[axis]]
        wrapper = campaign.build_axis_wrapper(task_parameters)
        cohort = wrapper.active_dataset
        heldout_dataset = campaign.build_population_dataset(
            wrapper, task_parameters, pixel_map.dataset_ranges(heldout_datasets))
        class_names = tuple(cohort.get_target_schemas()[target_field].class_names)
        catalogue = IonCatalogue.from_dataset(cohort, target_field)
        if tuple(catalogue.class_names) != class_names:
            raise ValueError("Ion catalogue and target columns disagree.")
        binner = wrapper.active_context.binner
        mass_axis = np.asarray(binner.GetXAxis(), dtype=np.float64)
        axis_dir = root / axis_directory(settings, axis)
        axis_dir.mkdir(parents=True, exist_ok=True)
        np.savez(axis_dir / "axis.npz", mass_axis=mass_axis,
                 bin_edges=np.asarray(binner.bin_edges.cpu() if torch.is_tensor(binner.bin_edges)
                                      else binner.bin_edges, dtype=np.float64),
                 window_edges=window_edges(mass_axis, float(settings["windows"]["width"])))
        ## REMARK: the visible population is the annotation-policy selection of the
        ## training dataset; it is protected API but has no public equivalent.
        visible = np.asarray(cohort._annotation_visible_source_indices(), dtype=np.int64)
        heldout_visible = np.asarray(heldout_dataset._annotation_visible_source_indices(), dtype=np.int64)
        for name in ("train", "test"):
            decoded = campaign.decode_source_spectra(cohort, assignments[name], target_field=target_field,
                                                     batch_size=batch_size)
            decoded["states"] = _classify_states(decoded["spectra"], decoded["targets"], decoded["mask"],
                                                 catalogue, policy)
            decoded["source_ids"] = assignments[name]
            decoded["visible"] = np.isin(assignments[name], visible)
            _save_population(axis_dir / name, decoded)
            if name == "test":
                counts = (decoded["targets"].astype(bool) & decoded["mask"]).sum(axis=0)
                test_positives.update({class_name: int(count) for class_name, count in zip(class_names, counts)})
            del decoded
            gc.collect()
        heldout = campaign.decode_source_spectra(heldout_dataset, heldout_candidates, target_field=target_field,
                                                 batch_size=batch_size)
        pool = np.setdiff1d(visible, subset_ids)
        for class_name, ids in campaign.class_positive_ids(cohort, class_names).items():
            candidates[class_name] = np.union1d(candidates.get(class_name, np.empty(0, np.int64)),
                                                np.intersect1d(ids, pool))
        state[axis] = {"wrapper": wrapper, "cohort": cohort, "class_names": class_names, "catalogue": catalogue,
                       "mass_axis": mass_axis, "visible": visible, "heldout": heldout,
                       "heldout_visible": heldout_visible}

    # Test extension shared by every axis
    extension = settings["populations"].get("test_extension", {})
    extended_ids, extension_report = campaign.select_test_extension(
        test_positives, candidates, minimum=int(extension.get("minimum_positives", 20)),
        seed=int(extension.get("seed", 42)))
    extension_report.to_csv(root / "test_extension.csv", index=False)
    logger.info("Test extension: %s pixels for %s deficient classes.", extended_ids.size,
                int((extension_report.added > 0).sum()))

    # Held-out pixels non-empty on every axis (paired across axes)
    nonempty = np.logical_and.reduce([state[axis]["heldout"]["tic"] > 0 for axis in cache.axes])
    heldout_ids = heldout_candidates[nonempty]
    empty_counts = {axis: int((state[axis]["heldout"]["tic"] <= 0).sum()) for axis in cache.axes}

    for axis in cache.axes:
        values = state[axis]
        axis_dir = root / axis_directory(settings, axis)
        ## Held-out population
        heldout = {name: array[nonempty] for name, array in values.pop("heldout").items()}
        heldout["states"] = _classify_states(heldout["spectra"], heldout["targets"], heldout["mask"],
                                             values["catalogue"], policy)
        heldout["source_ids"] = heldout_ids
        heldout["visible"] = np.isin(heldout_ids, values["heldout_visible"])
        _save_population(axis_dir / "heldout_image", heldout)
        values["heldout_spectra"] = heldout["spectra"]
        values["heldout_targets"] = heldout["targets"] & heldout["mask"]
        ## Test extension population
        decoded = campaign.decode_source_spectra(values["cohort"], extended_ids, target_field=target_field,
                                                 batch_size=batch_size)
        decoded["states"] = _classify_states(decoded["spectra"], decoded["targets"], decoded["mask"],
                                             values["catalogue"], policy)
        decoded["source_ids"] = extended_ids
        decoded["visible"] = np.isin(extended_ids, values["visible"])
        _save_population(axis_dir / "test_extended", decoded)
        del decoded, heldout
        gc.collect()

    # Pixel identities and coordinates
    identities = []
    for name, ids in (("train", assignments["train"]), ("test", assignments["test"]),
                      ("test_extended", extended_ids), ("heldout_image", heldout_ids)):
        frame = pixel_map.resolve(ids).assign(population=name, row=np.arange(ids.size))
        identities.append(frame)
    pixels = pd.concat(identities, ignore_index=True)
    pixels["x"] = -1
    pixels["y"] = -1
    image_paths = pixel_map.datasets.set_index("dataset_id").image_path
    for dataset_id in heldout_datasets:
        coordinates = imzml_coordinates(image_paths[dataset_id])  # (P, 2)
        rows = pixels.dataset_id.eq(dataset_id)
        pixels.loc[rows, ["x", "y"]] = coordinates[pixels.loc[rows, "source_pixel"].to_numpy()]
    pixels.to_csv(root / "populations.csv", index=False)

    # Class catalogue and class sets
    class_mz = campaign.class_mz_table(settings["merged_store"]).set_index("class_name").mz
    reference_range = tuple(settings.get("reference_mass_range", (200.0, 900.0)))
    class_sets = campaign.class_set_labels({axis: state[axis]["class_names"] for axis in cache.axes},
                                           class_mz, reference_range)
    class_sets.to_csv(root / "class_sets.csv", index=False)
    for axis in cache.axes:
        values = state[axis]
        axis_dir = root / axis_directory(settings, axis)
        counts = {}
        for name in ("train", "test", "test_extended", "heldout_image"):
            targets = np.load(axis_dir / name / "targets.npy", mmap_mode="r")
            mask = np.load(axis_dir / name / "mask.npy", mmap_mode="r")
            counts[f"{name}_positives"] = (np.asarray(targets).astype(bool) & np.asarray(mask)).sum(axis=0)
        classes = pd.DataFrame({"class_index": np.arange(len(values["class_names"])),
                                "class_name": values["class_names"],
                                "bins": [json.dumps(list(bins)) for bins in values["catalogue"].bins],
                                "bin_mz": [float(values["mass_axis"][bins[0]]) for bins in values["catalogue"].bins],
                                **counts})
        classes = classes.merge(class_sets[["class_name", "mz", "class_set"]], on="class_name", how="left",
                                validate="one_to_one")
        classes.to_csv(axis_dir / "classes.csv", index=False)

    # METASPACE reference images of the held-out datasets
    heldout_pixels = pixels[pixels.population == "heldout_image"].reset_index(drop=True)
    _write_metaspace(root, settings, cache, state, heldout_pixels, heldout_datasets, pixel_map)

    # Display subset and fixed spectrum cases
    _write_display_and_cases(root, settings, cache, state, pixels)

    for values in state.values():
        values.clear()
    state.clear()
    gc.collect()
    _atomic_json(root / "complete.json", {"key": key, "plan": cache.keys["plan"], "heldout_datasets": heldout_datasets,
                                          "heldout_empty_pixels": empty_counts,
                                          "test_extension_pixels": int(extended_ids.size),
                                          "sources_sha256": _source_key(settings, POPULATION_SOURCES),
                                          "provenance": provenance_record(settings)})
    logger.info("Decoded populations written to %s.", root)


def _write_metaspace(root: Path, settings: dict, cache: CampaignCache, state: dict, heldout_pixels: pd.DataFrame,
                     heldout_datasets: list[str], pixel_map: MergedPixelMap) -> None:
    """Align METASPACE ion images to held-out pixels and verify the coordinate offset."""
    metaspace = settings.get("metaspace", {})
    offset = tuple(metaspace.get("coordinate_offset", (-1, -1)))
    candidates = [tuple(value) for value in metaspace.get("candidate_offsets", [(-1, -1), (0, 0), (-1, 0), (0, -1),
                                                                                (1, 1)])]
    if offset not in candidates:
        candidates.append(offset)
    directories = pixel_map.datasets.set_index("dataset_id").image_path.map(lambda path: str(Path(path).parent))
    xy = heldout_pixels[["x", "y"]].to_numpy(np.int64)  # (N_h, 2)
    ion_rows, blocks, alignment_rows = [], [], []
    reference_axis = cache.axes[0]
    reference_edges = cache.axis_meta(reference_axis)["bin_edges"]
    radius = int(settings["evidence"].get("bin_radius", 1))
    for dataset_id in heldout_datasets:
        annotations, coordinates, intensities = campaign.read_metaspace_images(directories[dataset_id])
        ## REMARK: the export lists a formula and adduct once per METASPACE database; the
        ## repeated rows carry identical ion images, so the first row of every class is kept.
        unique = ~annotations.class_name.duplicated().to_numpy()
        if not unique.all():
            logger.info("Dropping %s repeated METASPACE annotations of %s.", int((~unique).sum()), dataset_id)
        annotations, intensities = annotations[unique].reset_index(drop=True), intensities[unique]  # (A_d, P)
        rows = heldout_pixels.dataset_id.eq(dataset_id).to_numpy()
        ## Offset verification against the reference-axis input ion images
        supports = ion_bin_indices(annotations.mz.to_numpy(), reference_edges, radius)
        observed = ion_intensities(state[reference_axis]["heldout_spectra"][rows], supports)  # (N_d, A)
        for candidate in candidates:
            aligned, coverage = campaign.align_metaspace(coordinates, intensities, xy[rows], candidate)
            correlations = [spatial_agreement(aligned[:, ion], observed[:, ion])["spearman"]
                            for ion in range(len(annotations)) if supports[ion].size]
            alignment_rows.append({"dataset_id": dataset_id, "offset_x": candidate[0], "offset_y": candidate[1],
                                   "coverage": coverage, "median_spearman": float(np.nanmedian(correlations))
                                   if correlations else np.nan, "ions": len(correlations),
                                   "configured": candidate == offset})
        aligned, _ = campaign.align_metaspace(coordinates, intensities, xy[rows], offset)
        block = np.full((len(heldout_pixels), len(annotations)), np.nan, dtype=np.float32)  # (N_h, A_d)
        block[rows] = aligned
        blocks.append(block)
        ion_rows.append(annotations.assign(dataset_id=dataset_id))
    alignment = pd.DataFrame(alignment_rows)
    alignment.to_csv(root / "metaspace_alignment.csv", index=False)
    best = alignment.loc[alignment.groupby("dataset_id").median_spearman.idxmax()]
    if not best.configured.all():
        raise ValueError("The configured METASPACE coordinate offset is not the best-aligned candidate:\n"
                         f"{alignment.to_string(index=False)}")
    ions = pd.concat(ion_rows, ignore_index=True)
    ions.insert(0, "ion_index", np.arange(len(ions)))
    ions.to_csv(root / "heldout_ions.csv", index=False)
    np.save(root / "heldout_metaspace.npy", np.concatenate(blocks, axis=1))  # (N_h, I)
    ## Input ion images per axis, same ion order
    class_sets = pd.read_csv(root / "class_sets.csv").set_index("class_name")
    for axis in cache.axes:
        axis_dir = root / axis_directory(settings, axis)
        supports = ion_bin_indices(ions.mz.to_numpy(), cache.axis_meta(axis)["bin_edges"], radius)
        np.save(axis_dir / "heldout_ion_input.npy", ion_intensities(state[axis]["heldout_spectra"], supports))
        head = {name: index for index, name in enumerate(state[axis]["class_names"])}
        table = ions[["ion_index", "dataset_id", "class_name", "mz"]].copy()
        table["bins"] = [json.dumps(support.tolist()) for support in supports]
        table["in_axis"] = [support.size > 0 for support in supports]
        table["head_column"] = table.class_name.map(head).fillna(-1).astype(int)
        table["class_set"] = np.where(table.head_column >= 0,
                                      class_sets.class_set.reindex(table.class_name).to_numpy(), "not_in_head")
        table.to_csv(axis_dir / "heldout_ions.csv", index=False)


def display_selection(spectra: np.ndarray, mass_axis: np.ndarray, edges: np.ndarray,
                      threshold: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Display bins: the bins of the m/z windows whose mean held-out TIC fraction reaches ``threshold``.

    :param spectra: Held-out spectra, shape ``(N_h, M)``.
    :type spectra: numpy.ndarray
    :param mass_axis: Bin centres, shape ``(M,)``.
    :type mass_axis: numpy.ndarray
    :param edges: Window edges, shape ``(W + 1,)``.
    :type edges: numpy.ndarray
    :param threshold: Minimum mean window mass.
    :type threshold: float
    :return: Display bins ``(K,)``, mean window masses ``(W,)`` and kept windows.
    :rtype: tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]
    """
    assignment = window_assignment(mass_axis, edges)  # (M,)
    membership = np.zeros((assignment.size, len(edges) - 1), dtype=np.float32)  # (M, W)
    membership[np.arange(assignment.size), assignment] = 1.0
    window_mass = (spectra @ membership).mean(axis=0)  # (W,)
    kept_windows = np.flatnonzero(window_mass >= threshold)
    return np.flatnonzero(np.isin(assignment, kept_windows)), window_mass, kept_windows


def case_rows(pixels: pd.DataFrame, settings: dict) -> dict[str, np.ndarray]:
    """Fixed seeded spectrum-case rows of every population (same rows for every model)."""
    cases = settings.get("cases", {})
    count, seed = int(cases.get("per_population", 6)), int(cases.get("seed", 42))
    rows = {}
    for position, (name, frame) in enumerate(pixels.groupby("population", sort=False)):
        generator = np.random.default_rng([seed, position])
        rows[name] = np.sort(generator.choice(len(frame), size=min(count, len(frame)), replace=False))
    return rows


def _write_display_and_cases(root: Path, settings: dict, cache: CampaignCache, state: dict,
                             pixels: pd.DataFrame) -> None:
    """Choose display bins with signal and fixed spectrum-case rows per population."""
    threshold = float(settings.get("display", {}).get("minimum_window_mass_fraction", 0.01))
    for axis in cache.axes:
        axis_dir = root / axis_directory(settings, axis)
        meta = cache.axis_meta(axis)
        spectra = state[axis]["heldout_spectra"]  # (N_h, M)
        bins, window_mass, kept_windows = display_selection(spectra, meta["mass_axis"], meta["window_edges"],
                                                            threshold)
        np.save(axis_dir / "display_bins.npy", bins)
        np.save(axis_dir / "display_input.npy", spectra[:, bins].astype(np.float16))  # (N_h, K)
        pd.DataFrame({"window_lower": meta["window_edges"][:-1], "window_upper": meta["window_edges"][1:],
                      "heldout_mean_mass": window_mass,
                      "displayed": np.isin(np.arange(window_mass.size), kept_windows)}).to_csv(
            axis_dir / "display_windows.csv", index=False)
    for name, rows in case_rows(pixels, settings).items():
        np.save(root / f"case_rows_{name}.npy", rows)


# --------------------------------------------------
# Section: validation of a legacy population directory
# --------------------------------------------------

def validate_legacy_populations(root: Path, cache: CampaignCache) -> pd.DataFrame:
    """Re-derive the population level on samples and compare with a legacy directory.

    Checks: the plan it was built from; held-out datasets; train/test identities against
    the resolved split; window edges; class columns; a seeded sample of every population
    re-decoded through the current reader, binner, normalization and evidence policy
    (spectra, targets, mask, states); the complete test extension; the case rows; the
    display bins; and the configured METASPACE offset.

    :param root: Legacy population directory.
    :type root: pathlib.Path
    :param cache: Cache with the resolved plan.
    :type cache: CampaignCache
    :return: Check rows (``check``, ``subject``, ``passed``, ``detail``).
    :rtype: pandas.DataFrame
    """
    settings = cache.settings
    rows: list[dict] = []

    def check(name: str, subject: str, passed: bool, detail: str = "") -> None:
        rows.append({"check": name, "subject": subject, "passed": bool(passed), "detail": detail})

    complete = json.loads((root / "complete.json").read_text())
    check("plan", root.name, complete.get("plan") == cache.keys["plan"],
          f"built from {complete.get('plan')}, current plan {cache.keys['plan']}")
    parameters = cache.plan["parameters"]
    references = reference_task_ids(cache)
    first_parameters = parameters[references[cache.axes[0]]]
    heldout_datasets = _heldout_datasets(settings, first_parameters)
    check("heldout_datasets", root.name, complete.get("heldout_datasets") == heldout_datasets, str(heldout_datasets))
    assignments = campaign.split_assignments(first_parameters)
    subset_ids = np.unique(np.concatenate(list(assignments.values())))
    pixel_map = MergedPixelMap.from_store(settings["merged_store"])
    policy = SignalEvidencePolicy(**settings["evidence"])
    target_field = settings["target_field"]
    seed = int(settings.get("cases", {}).get("seed", 42))
    test_positives: dict[str, int] = {}
    candidates: dict[str, np.ndarray] = {}
    for axis in cache.axes:
        axis_dir = root / axis_directory(settings, axis)
        ## Identities and axis geometry
        for name in ("train", "test"):
            stored = np.load(axis_dir / name / "source_ids.npy")
            check("split_identities", f"{axis}/{name}", np.array_equal(stored, assignments[name]), f"{stored.size} ids")
        with np.load(axis_dir / "axis.npz") as archive:
            mass_axis, edges = archive["mass_axis"], archive["window_edges"]
        check("window_edges", axis, np.array_equal(edges, window_edges(mass_axis, float(settings["windows"]["width"]))))
        ## Re-decoding of seeded samples
        wrapper = campaign.build_axis_wrapper(parameters[references[axis]])
        cohort = wrapper.active_dataset
        heldout_dataset = campaign.build_population_dataset(wrapper, parameters[references[axis]],
                                                            pixel_map.dataset_ranges(heldout_datasets))
        class_names = tuple(cohort.get_target_schemas()[target_field].class_names)
        classes = pd.read_csv(axis_dir / "classes.csv")
        check("class_columns", axis, tuple(classes.class_name) == class_names, f"{len(class_names)} classes")
        catalogue = IonCatalogue.from_dataset(cohort, target_field)
        for position, name in enumerate(campaign.POPULATIONS):
            stored = {key: np.load(axis_dir / name / f"{key}.npy", mmap_mode="r")
                      for key in ("source_ids", "spectra", "targets", "mask", "states")}
            sample = np.sort(np.random.default_rng([seed, 97, position]).choice(
                len(stored["source_ids"]), size=min(VALIDATION_ROWS, len(stored["source_ids"])), replace=False))
            dataset = heldout_dataset if name == "heldout_image" else cohort
            decoded = campaign.decode_source_spectra(dataset, np.asarray(stored["source_ids"][sample]),
                                                     target_field=target_field, batch_size=VALIDATION_ROWS)
            states = _classify_states(decoded["spectra"], decoded["targets"], decoded["mask"], catalogue, policy)
            check("decoded_spectra", f"{axis}/{name}",
                  np.allclose(decoded["spectra"], stored["spectra"][sample], rtol=1e-6, atol=1e-9),
                  f"{sample.size} rows, max abs diff "
                  f"{float(np.max(np.abs(decoded['spectra'] - stored['spectra'][sample]))):.3g}")
            for key, values in (("targets", decoded["targets"]), ("mask", decoded["mask"]), ("states", states)):
                check(f"decoded_{key}", f"{axis}/{name}", np.array_equal(values, stored[key][sample]),
                      f"{sample.size} rows")
        ## Inputs of the test extension
        test_targets = np.load(axis_dir / "test" / "targets.npy").astype(bool)
        test_mask = np.load(axis_dir / "test" / "mask.npy")
        test_positives.update({class_name: int(count) for class_name, count in
                               zip(class_names, (test_targets & test_mask).sum(axis=0))})
        visible = np.asarray(cohort._annotation_visible_source_indices(), dtype=np.int64)
        pool = np.setdiff1d(visible, subset_ids)
        for class_name, ids in campaign.class_positive_ids(cohort, class_names).items():
            candidates[class_name] = np.union1d(candidates.get(class_name, np.empty(0, np.int64)),
                                                np.intersect1d(ids, pool))
        ## Display bins from the stored held-out spectra (same float32 computation)
        threshold = float(settings.get("display", {}).get("minimum_window_mass_fraction", 0.01))
        heldout_spectra = np.asarray(np.load(axis_dir / "heldout_image" / "spectra.npy", mmap_mode="r"))  # (N_h, M)
        bins, _, _ = display_selection(heldout_spectra, mass_axis, edges, threshold)
        check("display_bins", axis, np.array_equal(bins, np.load(axis_dir / "display_bins.npy")), f"{bins.size} bins")
        del heldout_spectra
        del wrapper, cohort, heldout_dataset
        gc.collect()
    ## Complete test extension and case rows
    extension = settings["populations"].get("test_extension", {})
    extended_ids, _ = campaign.select_test_extension(test_positives, candidates,
                                                     minimum=int(extension.get("minimum_positives", 20)),
                                                     seed=int(extension.get("seed", 42)))
    stored_extension = np.load(root / axis_directory(settings, cache.axes[0]) / "test_extended" / "source_ids.npy")
    check("test_extension", root.name, np.array_equal(extended_ids, stored_extension), f"{extended_ids.size} pixels")
    pixels = pd.read_csv(root / "populations.csv")
    for name, values in case_rows(pixels, settings).items():
        check("case_rows", name, np.array_equal(values, np.load(root / f"case_rows_{name}.npy")))
    ## Configured METASPACE offset
    alignment = pd.read_csv(root / "metaspace_alignment.csv")
    configured = alignment[alignment.configured.astype(bool)]
    offset = tuple(settings.get("metaspace", {}).get("coordinate_offset", (-1, -1)))
    check("metaspace_offset", root.name,
          len(configured) > 0 and all((row.offset_x, row.offset_y) == offset for row in configured.itertuples()),
          str(offset))
    return pd.DataFrame(rows)
