"""Precompute of the chemical label-graph analysis (GCN expansion notebooks).

Stages:

1. ``chemistry.taxonomies`` — normalize local taxonomy exports once into
   ``taxonomies.derived_directory`` (shared, reused across runs).
2. ``chemistry.collections`` — per METASPACE collection: annotations, ion table,
   candidate resolution and class evidence for every taxonomy and resolution
   mode, and the decoded ion-image cache (shared, ``cache_directory``).
3. Seven analyses, each writing only its own ``part_*_results`` directory.

No analysis reads the results of another analysis; shared inputs come only from
stages 1–2.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import yaml

from msi_dataset_manager.annotations.chemistry import (
    RESOLUTION_MODES,
    ChemicalTaxonomy,
    class_assignments,
    load_chebi_taxonomy,
    load_classyfire_taxonomy,
    load_lipidmaps_taxonomy,
    load_swisslipids_taxonomy,
    resolve_candidates,
)

from .....utils.logger import get_custom_logger
from ....chemistry import consistency, graph_metrics, mz_distribution, spatial
from ....chemistry.annotations import ion_table, read_annotation_collection
from ....chemistry.ion_images import cache_ion_images, load_ion_images
from ....chemistry.label_graph import EVIDENCE_COLUMNS, GraphVariant, LabelGraph, build_label_graph, variant_grid
from ...cli import run_precompute_command
from ...core.contracts import AnalysisPlugin, ArtifactSpec, PrecomputeStrategy

logger = get_custom_logger(__name__)

STRATEGY = "chemistry.class_graph_analysis"
_LOADERS = {
    "lipidmaps": lambda files, version, options: load_lipidmaps_taxonomy(files["table"], version),
    "swisslipids": lambda files, version, options: load_swisslipids_taxonomy(files["table"], version),
    "classyfire": lambda files, version, options: load_classyfire_taxonomy(
        files["ontology"], files["compounds"], version, alternative_parents=bool(options.get("alternative_parents", False))),
    "chebi": lambda files, version, options: load_chebi_taxonomy(files["ontology"], version),
}
COLLECTION_FILES = ("annotations.csv.gz", "ions.csv", "candidates.csv.gz", "evidence.csv.gz", "fingerprint.json")


# Settings and paths
def read_settings(path: Path | str) -> dict[str, Any]:
    """Load the analysis settings and resolve the repository root.

    :param path: ``analysis_settings.yaml`` beside the notebooks.
    :type path: pathlib.Path | str
    :return: Settings with absolute ``settings_path`` and ``repository_root``.
    :rtype: dict[str, typing.Any]
    :raises FileNotFoundError: If the settings file does not exist.
    :raises ValueError: If the file is outside the repository or not a mapping.
    """
    settings_path = Path(path).resolve()
    if not settings_path.is_file():
        raise FileNotFoundError(f"No analysis settings at '{settings_path}'.")
    root = next((parent for parent in settings_path.parents if (parent / "pyproject.toml").is_file()), None)
    if root is None:
        raise ValueError("Analysis settings must live inside the repository.")
    settings = yaml.safe_load(settings_path.read_text(encoding="utf-8")) or {}
    if not isinstance(settings, dict):
        raise ValueError("Chemical graph analysis settings must be a YAML mapping.")
    settings["settings_path"] = str(settings_path)
    settings["repository_root"] = str(root)
    return settings


def _absolute(settings: Mapping[str, Any], value: str | Path) -> Path:
    """Resolve a repository-relative path."""
    path = Path(value)
    return path if path.is_absolute() else Path(settings["repository_root"]) / path


def _output(settings: Mapping[str, Any], analysis: str) -> Path:
    """Return the absolute result directory of one analysis."""
    return _absolute(settings, settings["analyses"][analysis]["output_directory"])


def _slug(value: str) -> str:
    """Return a filename-safe identifier."""
    return re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_")


def taxonomy_directory(settings: Mapping[str, Any], name: str) -> Path:
    """Return the derived directory of one normalized taxonomy snapshot."""
    source = settings["taxonomies"]["sources"][name]
    return _absolute(settings, settings["taxonomies"]["derived_directory"]) / f"{name}_{_slug(str(source['version']))}"


def collection_directory(settings: Mapping[str, Any], name: str) -> Path:
    """Return the cache directory of one METASPACE collection."""
    return _absolute(settings, settings["cache_directory"]) / name


def load_result(settings: Mapping[str, Any], analysis: str, name: str) -> pd.DataFrame:
    """Load one canonical result table for a notebook.

    :param settings: Settings from :func:`read_settings`.
    :type settings: collections.abc.Mapping[str, typing.Any]
    :param analysis: Analysis key in ``analyses``.
    :type analysis: str
    :param name: CSV file name inside the analysis result directory.
    :type name: str
    :return: Loaded table.
    :rtype: pandas.DataFrame
    :raises FileNotFoundError: With the producing command if the table is missing.
    """
    path = _output(settings, analysis) / name
    if not path.is_file():
        command = run_precompute_command(settings["settings_path"])
        raise FileNotFoundError(f"Missing '{path}'. Produce it with:\n  {command} --analysis {analysis}")
    return pd.read_csv(path, keep_default_na=True)


def result_path(settings: Mapping[str, Any], analysis: str, name: str) -> Path:
    """Return an existing non-tabular result artifact (for example an ``.npz`` map).

    :raises FileNotFoundError: With the producing command if the artifact is missing.
    """
    path = _output(settings, analysis) / name
    if not path.exists():
        command = run_precompute_command(settings["settings_path"])
        raise FileNotFoundError(f"Missing '{path}'. Produce it with:\n  {command} --analysis {analysis}")
    return path


def focus_variants(settings: Mapping[str, Any]) -> list[GraphVariant]:
    """Return the configured focus variants with defaults for omitted axes."""
    variants = []
    for mapping in settings["focus_variants"]:
        variants.append(GraphVariant(
            taxonomies=tuple(mapping["taxonomies"]), resolution=mapping["resolution"],
            assignment=mapping["assignment"], attachment=mapping.get("attachment", "all_ancestors"),
            alignment=mapping.get("alignment", "none"), adduct_links=bool(mapping.get("adduct_links", False)),
            pruning=mapping.get("pruning", "none")))
    return variants


def _write_metadata(directory: Path, payload: Mapping[str, Any]) -> None:
    """Write provenance of one result directory."""
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "metadata.json").write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")


def compute_device(context: Any) -> str:
    """Return the enforced torch device of vectorized computations.

    :param context: Precompute context with ``settings`` and ``allow_cpu``.
    :type context: typing.Any
    :return: ``"cuda"`` or, only with an explicit CPU opt-in, ``"cpu"``.
    :rtype: str
    :raises RuntimeError: If CUDA is configured but unavailable and CPU fallback was
        not explicitly allowed.
    """
    import torch

    configured = str(context.settings.get("device", "cuda"))
    if configured.startswith("cuda") and not torch.cuda.is_available():
        if not context.allow_cpu:
            raise RuntimeError("CUDA is configured but unavailable; rerun with --allow-cpu to accept the CPU.")
        logger.warning("CUDA unavailable; running vectorized computations on the CPU by explicit request.")
        return "cpu"
    return configured


def _inventory(_: dict[str, Any]) -> pd.DataFrame:
    """Expose the model-free analysis as one inventory record for the common runner."""
    return pd.DataFrame([{"model_id": "chemical_label_graph", "source": "metaspace_annotations",
                          "grid_id": "label_graph", "task_id": "label_graph", "repetition": 0,
                          "label": "Chemical label graph", "ready": True}])


# Shared stage 1: taxonomies
def _build_taxonomies(context: Any) -> None:
    """Normalize every configured taxonomy export once."""
    settings = context.settings
    source_directory = _absolute(settings, settings["taxonomies"]["source_directory"])
    records = []
    for name in settings["taxonomies"]["order"]:
        source = settings["taxonomies"]["sources"][name]
        target = taxonomy_directory(settings, name)
        if not (target / "manifest.json").is_file():
            logger.info("Normalizing taxonomy %s from %s.", name, source_directory)
            files = {key: source_directory / value for key, value in source["files"].items()}
            _LOADERS[source["loader"]](files, str(source["version"]), dict(source.get("options") or {})).save(target)
        manifest = json.loads((target / "manifest.json").read_text())
        records.append({"name": name, "version": manifest["version"], "directory": str(target), **manifest["counts"]})
    output = Path(settings["precompute"]["output_directory"])
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(output / "taxonomies.csv", index=False)


def load_taxonomies(context: Any) -> dict[str, ChemicalTaxonomy]:
    """Return memoized normalized taxonomies in configured order."""
    def build() -> dict[str, ChemicalTaxonomy]:
        return {name: ChemicalTaxonomy.load(taxonomy_directory(context.settings, name))
                for name in context.settings["taxonomies"]["order"]}

    return context.resources.get_or_create("chemistry_taxonomies", build)


# Shared stage 2: collections
def _collection_fingerprint(settings: Mapping[str, Any], name: str, dataset_ids: list[str]) -> dict[str, Any]:
    """Return the inputs that determine a collection cache."""
    return {"datasets": dataset_ids,
            "taxonomies": {n: str(settings["taxonomies"]["sources"][n]["version"]) for n in settings["taxonomies"]["order"]},
            "resolutions": list(RESOLUTION_MODES)}


def _build_collections(context: Any) -> None:
    """Materialize annotations, class evidence and ion images per collection."""
    settings = context.settings
    taxonomies = load_taxonomies(context)
    for name, collection in settings["collections"].items():
        root = _absolute(settings, collection["dataset_root"])
        excluded = set(collection.get("exclude") or [])
        dataset_ids = sorted(path.name for path in root.iterdir() if path.name not in excluded
                             and (path / "annotations.csv").is_file() and (path / "pixel_intensities.csv").is_file())
        directory = collection_directory(settings, name)
        fingerprint = _collection_fingerprint(settings, name, dataset_ids)
        fingerprint_path = directory / "fingerprint.json"
        ## Ion image cache (independent of taxonomies)
        for dataset_id in dataset_ids:
            cache_ion_images(root / dataset_id, directory / "images")
        if fingerprint_path.is_file() and json.loads(fingerprint_path.read_text()) == fingerprint:
            logger.info("Collection %s cache is current.", name)
            continue
        ## Annotations and ions
        annotations = read_annotation_collection(root, dataset_ids=dataset_ids)
        ions = ion_table(annotations)
        resolver_input = ions.assign(identifiers=ions.identifiers.map(json.loads))
        ## Candidate resolution and class evidence
        candidates, evidence = [], []
        for taxonomy_name, taxonomy in taxonomies.items():
            for mode in RESOLUTION_MODES:
                resolved = resolve_candidates(resolver_input, taxonomy, mode)
                candidates.append(resolved.assign(taxonomy=taxonomy_name, resolution=mode))
                evidence.append(class_assignments(resolved, taxonomy).assign(taxonomy=taxonomy_name, resolution=mode))
                logger.info("Collection %s, %s/%s: %s ions resolved.", name, taxonomy_name, mode, resolved.ion_id.nunique())
        directory.mkdir(parents=True, exist_ok=True)
        annotations.to_csv(directory / "annotations.csv.gz", index=False)
        ions.to_csv(directory / "ions.csv", index=False)
        pd.concat(candidates, ignore_index=True).to_csv(directory / "candidates.csv.gz", index=False)
        pd.concat(evidence, ignore_index=True).loc[:, list(EVIDENCE_COLUMNS)].to_csv(directory / "evidence.csv.gz", index=False)
        fingerprint_path.write_text(json.dumps(fingerprint, indent=2) + "\n")


def load_collection(context: Any, name: str) -> dict[str, Any]:
    """Return memoized collection tables and ion-image cache paths."""
    def build() -> dict[str, Any]:
        directory = collection_directory(context.settings, name)
        evidence = pd.read_csv(directory / "evidence.csv.gz")
        evidence["certain"] = evidence.certain.astype(bool)
        return {
            "annotations": pd.read_csv(directory / "annotations.csv.gz"),
            "ions": pd.read_csv(directory / "ions.csv"),
            "candidates": pd.read_csv(directory / "candidates.csv.gz"),
            "evidence": evidence,
            "images": sorted((directory / "images").glob("*.npz")),
        }

    return context.resources.get_or_create(f"chemistry_collection_{name}", build)


def _graph(context: Any, collection: str, variant: GraphVariant) -> LabelGraph:
    """Build one label graph with the configured pruning parameters."""
    data = load_collection(context, collection)
    configuration = context.settings["graph"]
    return build_label_graph(variant, data["ions"], data["evidence"], load_taxonomies(context),
                             min_class_support=int(configuration["min_class_support"]),
                             max_class_fraction=float(configuration["max_class_fraction"]))


def _grid(context: Any) -> list[GraphVariant]:
    """Return the configured complete variant grid."""
    return variant_grid(context.settings["taxonomies"]["order"], context.settings["graph"]["axes"])


# Analysis: annotation inventory
def _annotation_inventory(context: Any) -> None:
    """Describe annotations, databases, ambiguity and taxonomy resolution coverage."""
    output = _output(context.settings, "annotation_inventory")
    ion_rows, dataset_rows, database_rows, coverage_rows = [], [], [], []
    for name in context.settings["collections"]:
        data = load_collection(context, name)
        annotations, ions, candidates = data["annotations"], data["ions"], data["candidates"]
        counts = candidates.groupby(["taxonomy", "resolution", "ion_id"]).size().rename("candidates").reset_index()
        wide = counts.pivot_table(index="ion_id", columns=["taxonomy", "resolution"], values="candidates", fill_value=0)
        wide.columns = [f"candidates__{t}__{r}" for t, r in wide.columns]
        ion_rows.append(ions.drop(columns=["identifiers", "compound_names"]).merge(wide, left_on="ion_id", right_index=True, how="left")
                        .fillna({column: 0 for column in wide.columns}).assign(collection=name))
        for dataset_id, group in annotations.groupby("dataset_id"):
            dataset_rows.append({"collection": name, "dataset_id": dataset_id, "annotations": len(group),
                                 "ions": group.ion_id.nunique(), "fdr_le_0_05": float((group.fdr <= 0.05).mean()),
                                 "off_sample_fraction": float(group.off_sample.mean()),
                                 "median_possible_compounds": float(group.identifiers.map(lambda v: len(json.loads(v))).median())})
        for database, group in annotations.groupby("database"):
            database_rows.append({"collection": name, "database": database, "annotations": len(group), "ions": group.ion_id.nunique()})
        for (taxonomy, resolution), group in counts.groupby(["taxonomy", "resolution"]):
            coverage_rows.append({"collection": name, "taxonomy": taxonomy, "resolution": resolution,
                                  "ions_total": len(ions), "ions_resolved": group.ion_id.nunique(),
                                  "coverage": group.ion_id.nunique() / len(ions),
                                  "median_candidates": float(group.candidates.median()),
                                  "unique_candidate_fraction": float((group.candidates == 1).mean())})
    pd.concat(ion_rows, ignore_index=True).to_csv(output / "ions.csv", index=False)
    pd.DataFrame(dataset_rows).to_csv(output / "datasets.csv", index=False)
    pd.DataFrame(database_rows).to_csv(output / "databases.csv", index=False)
    pd.DataFrame(coverage_rows).to_csv(output / "resolution_coverage.csv", index=False)
    _write_metadata(output, {"analysis": "annotation_inventory", "collections": context.settings["collections"],
                             "resolutions": list(RESOLUTION_MODES)})


# Analysis: taxonomy structure
def _taxonomy_structure(context: Any) -> None:
    """Describe hierarchy shape, level-wise ambiguity and cross-taxonomy agreement."""
    configuration = context.settings["analyses"]["taxonomy_structure"]
    output = _output(context.settings, "taxonomy_structure")
    taxonomies = load_taxonomies(context)
    summary = []
    for name, taxonomy in taxonomies.items():
        parent_counts = taxonomy.class_edges.groupby("child_id").size()
        memberships = taxonomy.memberships.groupby("compound_id").size()
        depth = pd.Series(taxonomy.class_depth)
        summary.append({"taxonomy": name, "version": taxonomy.version, "classes": len(taxonomy.classes),
                        "class_edges": len(taxonomy.class_edges), "compounds": len(taxonomy.compounds),
                        "classified_compound_fraction": memberships.size / max(len(taxonomy.compounds), 1),
                        "direct_memberships_per_compound": float(memberships.mean()) if memberships.size else float("nan"),
                        "multi_parent_class_fraction": float((parent_counts > 1).sum() / max(len(taxonomy.classes), 1)),
                        "roots": int((depth == 0).sum()), "max_depth": int(depth.max()),
                        "alignments_exact": int((taxonomy.alignments.predicate == "EXACT").sum()),
                        "alignments_declared": int(taxonomy.alignments.predicate.isin(["EXACT", "BROAD", "NARROW"]).sum())})
    pd.DataFrame(summary).to_csv(output / "taxonomy_summary.csv", index=False)
    depth_rows = [{"taxonomy": name, "depth": d, "classes": int(count)}
                  for name, taxonomy in taxonomies.items()
                  for d, count in pd.Series(taxonomy.class_depth).value_counts().sort_index().items()]
    pd.DataFrame(depth_rows).to_csv(output / "class_depths.csv", index=False)

    profiles, agreements, unique_rows = [], [], []
    for name in context.settings["collections"]:
        evidence = load_collection(context, name)["evidence"]
        profiles.append(graph_metrics.level_profile(evidence, taxonomies).assign(collection=name))
        for resolution in RESOLUTION_MODES:
            agreements.append(graph_metrics.taxonomy_agreement(
                evidence, taxonomies, resolution=resolution, max_depth=int(configuration["agreement_max_depth"]),
                min_ions=int(configuration["agreement_min_ions"])).assign(collection=name, resolution=resolution))
        ## Unique certain class per ion and depth (contingency input)
        certain = evidence[evidence.certain].copy()
        certain["depth"] = [taxonomies[t].class_depth.get(c, -1) for t, c in zip(certain.taxonomy, certain.class_id)]
        certain = certain[certain.depth <= int(configuration["agreement_max_depth"])]
        unique = certain[certain.groupby(["resolution", "taxonomy", "depth", "ion_id"]).class_id.transform("size") == 1]
        names = pd.concat([t.classes[["class_id", "name"]] for t in taxonomies.values()])
        unique_rows.append(unique.merge(names, on="class_id", how="left")
                           [["resolution", "taxonomy", "depth", "ion_id", "class_id", "name"]].assign(collection=name))
    pd.concat(profiles, ignore_index=True).to_csv(output / "level_profile.csv", index=False)
    pd.concat(agreements, ignore_index=True).to_csv(output / "taxonomy_agreement.csv", index=False)
    pd.concat(unique_rows, ignore_index=True).to_csv(output / "ion_unique_classes.csv", index=False)
    _write_metadata(output, {"analysis": "taxonomy_structure", **configuration})


# Analysis: class distribution along the m/z axis
def _class_mz_distribution(context: Any) -> None:
    """Measure whether m/z bins identify chemical classes (PN) or not (PU)."""
    configuration = context.settings["analyses"]["class_mz_distribution"]
    output = _output(context.settings, "class_mz_distribution")
    taxonomies = load_taxonomies(context)
    tables: dict[str, list[pd.DataFrame]] = {}
    for name in context.settings["collections"]:
        data = load_collection(context, name)
        adducts = sorted(data["annotations"].adduct.unique())
        result = mz_distribution.class_mz_profile(
            taxonomies, data["evidence"], data["ions"], adducts=adducts,
            mz_range=tuple(float(v) for v in configuration["mz_range"]),
            depths=tuple(int(d) for d in configuration["depths"]),
            bin_widths=tuple(float(w) for w in configuration["bin_widths"]),
            purity_threshold=float(configuration["purity_threshold"]),
            min_class_members=int(configuration["min_class_members"]))
        for key, frame in result.items():
            tables.setdefault(key, []).append(frame.assign(collection=name, adducts="+".join(adducts)))
    for key, frames in tables.items():
        pd.concat(frames, ignore_index=True).to_csv(output / f"{key}.csv", index=False)
    _write_metadata(output, {"analysis": "class_mz_distribution", **configuration})


# Analysis: graph variants
def _graph_variants(context: Any) -> None:
    """Compute structural metrics of every variant and export focus graphs."""
    settings = context.settings
    configuration = settings["graph"]
    output = _output(settings, "graph_variants")
    device = compute_device(context)
    rows, nodes, edges = [], [], []
    variants = _grid(context)
    focus = {variant.variant_id for variant in focus_variants(settings)}
    for name in settings["collections"]:
        data = load_collection(context, name)
        for position, variant in enumerate(variants + [v for v in focus_variants(settings) if v.variant_id not in {x.variant_id for x in variants}]):
            graph = _graph(context, name, variant)
            metrics = graph_metrics.structure_metrics(
                graph, trainable_support=int(configuration["trainable_support"]),
                propagation_steps=tuple(int(k) for k in configuration["propagation_steps"]), device=device)
            rows.append({"collection": name, **variant.to_record(), **metrics})
            if variant.variant_id in focus:
                ion_records = data["ions"][["ion_id", "formula", "adduct", "dataset_count"]].assign(
                    node_type="ion", resolved=graph.resolved)
                class_records = graph.classes.rename(columns={"class_id": "node_id"}).assign(node_type="class")
                nodes.append(pd.concat([ion_records.rename(columns={"ion_id": "node_id"}), class_records], ignore_index=True)
                             .assign(collection=name, variant_id=variant.variant_id))
                edges.append(graph.edges.assign(collection=name, variant_id=variant.variant_id))
            if position % 100 == 0:
                logger.info("Graph variants %s: %s/%s.", name, position, len(variants))
    pd.DataFrame(rows).to_csv(output / "variant_metrics.csv", index=False)
    pd.concat(nodes, ignore_index=True).to_csv(output / "focus_graph_nodes.csv", index=False)
    pd.concat(edges, ignore_index=True).to_csv(output / "focus_graph_edges.csv", index=False)
    _write_metadata(output, {"analysis": "graph_variants", "graph": configuration,
                             "variant_count": len(variants), "focus_variants": sorted(focus)})


# Analysis: spatial class maps
def _spatial_class_maps(context: Any) -> None:
    """Project ion images onto graph classes for every image and focus variant."""
    settings = context.settings
    configuration = settings["analyses"]["spatial_class_maps"]
    output = _output(settings, "spatial_class_maps")
    (output / "maps").mkdir(parents=True, exist_ok=True)
    summary, composition, footprints, map_index, image_rows = [], [], [], [], []
    max_depth, top = int(configuration["max_map_depth"]), int(configuration["top_classes_per_depth"])
    for name in settings["collections"]:
        data = load_collection(context, name)
        images = [load_ion_images(path) for path in data["images"]]
        ranked = sorted(images, key=lambda image: (-len(image.ion_ids), image.dataset_id))
        selected = {image.dataset_id for image in ranked[: int(configuration["map_images_per_collection"])]}
        image_rows.extend({"collection": name, "dataset_id": image.dataset_id, "ions": len(image.ion_ids),
                           "pixels": image.coordinates.shape[0], "map_selected": image.dataset_id in selected}
                          for image in images)
        for variant_index, variant in enumerate(focus_variants(settings)):
            graph = _graph(context, name, variant)
            classes = graph.classes.reset_index(drop=True)
            is_a = graph.edges[graph.edges.relation == "is_a"].groupby("source").target.agg(lambda v: ";".join(sorted(v)))
            classes["parents"] = classes.class_id.map(is_a).fillna("")
            for image in images:
                abundance, mask, resolved = spatial.class_abundance(image, graph)  # (P_+, C), (P,), (P_+,)
                effective = spatial.effective_class_count(abundance)  # (P_+,)
                summary.append({"collection": name, "dataset_id": image.dataset_id, "variant_id": variant.variant_id,
                                "pixels": int(mask.sum()), "resolved_signal_mean": float(resolved.mean()),
                                "resolved_signal_median": float(np.median(resolved)),
                                "effective_classes_mean": float(np.nanmean(effective)) if np.isfinite(effective).any() else float("nan")})
                means = abundance.mean(axis=0)  # (C,)
                shallow = (classes.depth <= max_depth) & (means > 0)
                composition.append(classes[shallow].drop(columns="parents").assign(collection=name, dataset_id=image.dataset_id,
                                                           variant_id=variant.variant_id, mean_abundance=means[shallow.to_numpy()]))
                if image.dataset_id not in selected:
                    continue
                ## Maps per taxonomy and depth
                arrays: dict[str, np.ndarray] = {"coordinates": image.coordinates, "mask": mask,
                                                 "resolved": resolved.astype(np.float16), "effective": effective.astype(np.float16)}
                for (taxonomy, depth), group in classes[classes.depth.between(0, max_depth)].groupby(["taxonomy", "depth"]):
                    columns = group.index.to_numpy()
                    block = abundance[:, columns]  # (P_+, C_d)
                    key = f"{taxonomy}_d{depth}"
                    order = np.argsort(-block.mean(axis=0))[:top]
                    arrays[f"{key}__classes"] = group.class_id.to_numpy()[order].astype(str)
                    arrays[f"{key}__names"] = group.name.fillna("").to_numpy()[order].astype(str)
                    arrays[f"{key}__abundance"] = block[:, order].T.astype(np.float16)  # (K, P_+)
                    arrays[f"{key}__all_classes"] = group.class_id.to_numpy().astype(str)
                    arrays[f"{key}__all_names"] = group.name.fillna("").to_numpy().astype(str)
                    arrays[f"{key}__dominant"] = np.where(block.max(axis=1) > 0, block.argmax(axis=1), -1).astype(np.int32)
                    arrays[f"{key}__dominant_share"] = block.max(axis=1).astype(np.float16)
                file = f"maps/{name}__{image.dataset_id}__v{variant_index}.npz"
                np.savez_compressed(output / file, **arrays)
                map_index.append({"collection": name, "dataset_id": image.dataset_id, "variant_index": variant_index,
                                  "variant_id": variant.variant_id, "file": file})
                ## Pixel footprint: pixel with the largest annotated signal
                pixel = int(np.argmax(image.intensities[:, mask].sum(axis=0)))
                present = abundance[pixel] > 0
                footprints.append(classes[present].assign(collection=name, dataset_id=image.dataset_id,
                                                          variant_id=variant.variant_id, abundance=abundance[pixel, present],
                                                          x=int(image.coordinates[mask][pixel, 0]), y=int(image.coordinates[mask][pixel, 1])))
            logger.info("Spatial maps %s: variant %s done.", name, variant.variant_id)
    pd.DataFrame(image_rows).to_csv(output / "images.csv", index=False)
    pd.DataFrame(summary).to_csv(output / "image_summary.csv", index=False)
    pd.concat(composition, ignore_index=True).to_csv(output / "class_composition.csv", index=False)
    pd.concat(footprints, ignore_index=True).to_csv(output / "pixel_footprints.csv", index=False)
    pd.DataFrame(map_index).to_csv(output / "map_index.csv", index=False)
    _write_metadata(output, {"analysis": "spatial_class_maps", **configuration,
                             "focus_variants": [v.variant_id for v in focus_variants(settings)]})


# Analysis: graph vs co-localization
def _graph_colocalization(context: Any) -> None:
    """Test whether graph proximity predicts spatial co-localization of ions."""
    settings = context.settings
    configuration = settings["analyses"]["graph_colocalization"]
    output = _output(settings, "graph_colocalization")
    seed, permutations = int(configuration["seed"]), int(configuration["n_permutations"])
    steps = tuple(int(k) for k in configuration["propagation_steps"])
    device = compute_device(context)
    mantel_rows, cka_rows, coherence_rows, samples = [], [], [], []
    focus = {variant.variant_id for variant in focus_variants(settings)}
    for name in settings["collections"]:
        data = load_collection(context, name)
        ion_ids = tuple(data["ions"].ion_id)
        reference, pair_counts = consistency.colocalization((load_ion_images(p) for p in data["images"]), ion_ids, device=device)  # (A, A)
        ## Ions observed together often enough for kernel alignment
        frequent = data["ions"].dataset_count.to_numpy() >= int(configuration["min_images_per_ion"])  # (A,)
        spatial_kernel = np.nan_to_num(reference[np.ix_(frequent, frequent)], nan=0.0)
        np.fill_diagonal(spatial_kernel, 1.0)
        seen: set[tuple] = set()
        for variant in _grid(context):
            graph = _graph(context, name, variant)
            key = (variant.taxonomies, variant.resolution, variant.assignment, variant.pruning)
            if key not in seen:
                ### Membership similarity is independent of attachment, alignment and adduct links
                seen.add(key)
                similarity = consistency.tanimoto_similarity(graph)  # (A, A)
                result = consistency.mantel_test(similarity, reference, n_permutations=permutations, seed=seed, device=device)
                mantel_rows.append({"collection": name, "taxonomies": "+".join(variant.taxonomies),
                                    "taxonomy_label": variant.taxonomy_label, "resolution": variant.resolution,
                                    "assignment": variant.assignment, "pruning": variant.pruning, **result})
            for k, kernel in consistency.propagated_kernels(graph, steps, frequent, device=device).items():  # (n, n)
                cka_rows.append({"collection": name, **variant.to_record(), "steps": k,
                                 "ions": int(frequent.sum()), "resolved_ions": int(graph.resolved[frequent].sum()),
                                 "cka": consistency.linear_cka(kernel, spatial_kernel)})
            if variant.variant_id in focus:
                coherence_rows.append(consistency.class_spatial_coherence(graph, reference).assign(
                    collection=name, variant_id=variant.variant_id))
                ### Deterministic pair sample for distribution plots
                similarity = consistency.tanimoto_similarity(graph)
                upper = np.triu_indices(len(ion_ids), k=1)
                valid = np.isfinite(similarity[upper]) & np.isfinite(reference[upper])
                chosen = np.flatnonzero(valid)
                chosen = np.random.default_rng(seed).choice(chosen, size=min(20000, chosen.size), replace=False)
                samples.append(pd.DataFrame({"collection": name, "variant_id": variant.variant_id,
                                             "ion_a": np.asarray(ion_ids)[upper[0][chosen]], "ion_b": np.asarray(ion_ids)[upper[1][chosen]],
                                             "similarity": similarity[upper][chosen], "colocalization": reference[upper][chosen],
                                             "images": pair_counts[upper][chosen]}))
        logger.info("Graph co-localization %s done.", name)
    pd.DataFrame(mantel_rows).to_csv(output / "mantel.csv", index=False)
    pd.DataFrame(cka_rows).to_csv(output / "kernel_alignment.csv", index=False)
    pd.concat(coherence_rows, ignore_index=True).to_csv(output / "class_coherence.csv", index=False)
    pd.concat(samples, ignore_index=True).to_csv(output / "pair_sample.csv", index=False)
    _write_metadata(output, {"analysis": "graph_colocalization", **configuration, "pair_sample_size": 20000})


# Analysis: label-space embedding of pixels
def _label_space_embedding(context: Any) -> None:
    """Compare pixel segmentations in ion space and in graph label spaces."""
    settings = context.settings
    configuration = settings["analyses"]["label_space_embedding"]
    output = _output(settings, "label_space_embedding")
    device = compute_device(context)
    rows = []
    for name in settings["collections"]:
        data = load_collection(context, name)
        for variant in focus_variants(settings):
            graph = _graph(context, name, variant)
            for path in data["images"]:
                image = load_ion_images(path)
                for record in spatial.clustering_agreement(
                        image, graph, steps=tuple(int(k) for k in configuration["propagation_steps"]),
                        n_clusters=int(configuration["n_clusters"]), sample_size=int(configuration["sample_size"]),
                        seed=int(configuration["seed"]), device=device):
                    rows.append({"collection": name, "dataset_id": image.dataset_id, "variant_id": variant.variant_id, **record})
            logger.info("Label-space embedding %s: variant %s done.", name, variant.variant_id)
    pd.DataFrame(rows).to_csv(output / "clustering.csv", index=False)
    _write_metadata(output, {"analysis": "label_space_embedding", **configuration})


def build_strategy() -> PrecomputeStrategy:
    """Build the complete chemical label-graph precompute workflow."""
    def analysis(name: str, analysis_name: str, files: tuple[str, ...], run: Any) -> AnalysisPlugin:
        return AnalysisPlugin(name=f"chemistry.{name}", requires=("model_catalog", "taxonomies", "collections"),
                              provides=(ArtifactSpec(analysis_name, analysis_name, files + ("metadata.json",)),),
                              run=run)

    return PrecomputeStrategy(
        name=STRATEGY,
        model_type="chemistry",
        stages=(
            AnalysisPlugin(name="chemistry.taxonomies", requires=("model_catalog",),
                           provides=(ArtifactSpec("taxonomies"),), run=_build_taxonomies,
                           enabled_when_configured=False),
            AnalysisPlugin(name="chemistry.collections", requires=("model_catalog", "taxonomies"),
                           provides=(ArtifactSpec("collections"),), run=_build_collections,
                           enabled_when_configured=False),
            analysis("annotation_inventory", "annotation_inventory",
                     ("ions.csv", "datasets.csv", "databases.csv", "resolution_coverage.csv"), _annotation_inventory),
            analysis("taxonomy_structure", "taxonomy_structure",
                     ("taxonomy_summary.csv", "class_depths.csv", "level_profile.csv", "taxonomy_agreement.csv",
                      "ion_unique_classes.csv"), _taxonomy_structure),
            analysis("class_mz_distribution", "class_mz_distribution",
                     ("universe_summary.csv", "universe_classes.csv", "observed_ions.csv", "class_mz_quantiles.csv",
                      "mz_histogram.csv"), _class_mz_distribution),
            analysis("graph_variants", "graph_variants",
                     ("variant_metrics.csv", "focus_graph_nodes.csv", "focus_graph_edges.csv"), _graph_variants),
            analysis("spatial_class_maps", "spatial_class_maps",
                     ("images.csv", "image_summary.csv", "class_composition.csv", "pixel_footprints.csv", "map_index.csv"),
                     _spatial_class_maps),
            analysis("graph_colocalization", "graph_colocalization",
                     ("mantel.csv", "kernel_alignment.csv", "class_coherence.csv", "pair_sample.csv"), _graph_colocalization),
            analysis("label_space_embedding", "label_space_embedding", ("clustering.csv",), _label_space_embedding),
        ),
        load_settings=read_settings,
        inventory=_inventory,
    )
