"""Synthetic-reference level: the pretraining artifacts and the class sets they target.

The campaign's synthetic data are regenerated here through the runtime's own entry
point (:func:`~msi_autoencoder_wrapper.data.pretraining.precomputed.build_precomputed_synthetic_partitions`)
on the resolved training dataset of every axis. The artifact store is keyed by the
generator's request fingerprint (train split, annotation index, axis, configuration), so
the regenerated artifact is the one the models were pretrained on, and an existing
artifact with that fingerprint is reused.

From the artifact and the generator statistics the level derives, per axis:

``classes.csv``
    per head class: eligibility, the generator's ``rare`` and ``overlap`` targets, the
    train annotation count and its rank among eligible classes;
``exposure.csv``
    synthetic tokens per population, class and token kind (what each variant saw);
``population_summary.csv``
    rows and tokens per population and kind;
``collision_pairs.csv``
    pairs of classes annotated at the same bin of the same training pixel, the spectral
    collisions the overlap quota targets;
``checks.csv``
    consistency of the derived sets with the manifests.

REMARK: The ``rare`` and ``overlap`` sets are read from the manifests (classes that
received rare- or overlap-bonus tokens), not re-implemented; the generator's own
statistics and the collision pairs only verify them.
"""

from __future__ import annotations

import gc
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ....data.pretraining import precompute_builder as builder_module
from ....data.pretraining.precomputed import PrecomputedSyntheticConfig, build_precomputed_synthetic_partitions
from ....utils.logger import get_custom_logger
from . import pretraining_campaign as campaign
from .pretraining_campaign_precompute import _atomic_json, _cache, axis_directory, provenance_record, reference_task_ids

logger = get_custom_logger(__name__)

#: Token kinds of the synthetic manifests (constants of the generator).
TOKEN_KINDS = {builder_module._KIND_SINGLE: "single", builder_module._KIND_BASE_CLASS: "base_class",
               builder_module._KIND_BLANK: "blank", builder_module._KIND_OVERLAP_BONUS: "overlap_bonus",
               builder_module._KIND_RARE_BONUS: "rare_bonus"}


# --------------------------------------------------
# Section: artifact declarations of the campaign
# --------------------------------------------------

def artifact_declarations(tasks: dict[str, Any], baseline_role: str) -> dict[str, dict]:
    """Synthetic artifact declaration of every axis, shared by all its pretraining phases.

    :param tasks: Planned tasks keyed by task identifier.
    :type tasks: dict
    :param baseline_role: Role without synthetic phases.
    :type baseline_role: str
    :return: Per axis the ``pretraining`` block of the first synthetic phase (with its
        ``population``); empty when the campaign has no synthetic phase.
    :rtype: dict[str, dict]
    :raises ValueError: If two phases of one axis declare different artifacts.
    """
    declarations: dict[str, dict] = {}
    for task in tasks.values():
        if task.workflow["role"] == baseline_role:
            continue
        axis = task.grid_parameters["axes"]["name"]
        for phase in task.parameters["training"]["phases"]:
            if "pretraining" not in phase:
                continue
            block = json.loads(json.dumps(phase["pretraining"], default=str))
            current = declarations.setdefault(axis, block)
            if current["artifact"] != block["artifact"]:
                raise ValueError(f"Axis {axis} declares more than one synthetic artifact.")
    return declarations


# --------------------------------------------------
# Section: derived tables
# --------------------------------------------------

def exposure_table(manifests: dict[str, Any], class_count: int) -> pd.DataFrame:
    """Synthetic tokens and rows per population, class and token kind.

    :param manifests: Population manifests of one artifact.
    :type manifests: dict[str, msi_autoencoder_wrapper.data.pretraining.precompute_artifact.SyntheticManifest]
    :param class_count: Number of head classes ``C``.
    :type class_count: int
    :return: ``population``, ``class_index``, ``kind``, ``tokens`` and ``rows`` (rows
        containing at least one token of that class and kind); zero rows are omitted.
    :rtype: pandas.DataFrame
    """
    frames = []
    for population, manifest in manifests.items():
        targets = np.asarray(manifest.requested_target_indices)  # (N, K)
        kinds = np.asarray(manifest.component_kinds)  # (N, K)
        for kind, name in TOKEN_KINDS.items():
            selected = (kinds == kind) & (targets >= 0)
            if not selected.any():
                continue
            tokens = np.bincount(targets[selected], minlength=class_count)  # (C,)
            row_ids, columns = np.nonzero(selected)
            pairs = np.unique(np.stack([row_ids, targets[row_ids, columns]], axis=1), axis=0)  # (P, 2)
            rows = np.bincount(pairs[:, 1], minlength=class_count)  # (C,)
            present = np.flatnonzero(tokens)
            frames.append(pd.DataFrame({"population": population, "class_index": present, "kind": name,
                                        "tokens": tokens[present], "rows": rows[present]}))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(
        columns=["population", "class_index", "kind", "tokens", "rows"])


def population_summary(manifests: dict[str, Any]) -> pd.DataFrame:
    """Rows and tokens per population and token kind."""
    rows = []
    for population, manifest in manifests.items():
        kinds = np.asarray(manifest.component_kinds)
        record = {"population": population, "rows": int(kinds.shape[0])}
        for kind, name in TOKEN_KINDS.items():
            record[f"tokens_{name}"] = int((kinds == kind).sum())
        rows.append(record)
    return pd.DataFrame(rows)


def collision_pairs(index: Any, train_ids: np.ndarray, class_names: tuple[str, ...],
                    feature_count: int) -> tuple[pd.DataFrame, np.ndarray]:
    """Class pairs annotated at the same bin of the same training pixel.

    Follows the generator's annotation summary: train pixels of the mapped annotation
    index, annotation identities mapped to head columns, in-axis bin coordinates.

    :param index: Mapped annotation index of the training dataset.
    :type index: typing.Any
    :param train_ids: Training source identifiers.
    :type train_ids: numpy.ndarray
    :param class_names: Head class names (``formula|adduct``).
    :type class_names: tuple[str, ...]
    :param feature_count: Number of bins ``M``.
    :type feature_count: int
    :return: ``class_a`` < ``class_b`` with ``co_annotations`` (pixel-bin occurrences)
        and ``pixels``, and the annotation count per class ``(C,)`` (pixel-bin occurrences,
        the generator's ``class_counts``).
    :rtype: tuple[pandas.DataFrame, numpy.ndarray]
    """
    lookup = {name: position for position, name in enumerate(class_names)}
    identity_targets = np.asarray([lookup.get("|".join(identity), -1) for identity in index.annotation_identities],
                                  dtype=np.int64)
    spectrum_ids = np.asarray(index.spectrum_ids, dtype=np.int64)
    offsets = np.asarray(index.spectrum_offsets, dtype=np.int64)
    rows = np.repeat(np.arange(spectrum_ids.size), np.diff(offsets))  # (E,)
    in_train = np.isin(spectrum_ids[rows], np.asarray(train_ids, dtype=np.int64))
    targets = identity_targets[np.asarray(index.annotation_indices, dtype=np.int64)]  # (E,)
    coordinates = np.asarray(index.coordinate_indices, dtype=np.int64)  # (E,)
    keep = in_train & (targets >= 0) & (coordinates >= 0) & (coordinates < feature_count)
    ## Unique (pixel, bin, class) triples, as the generator counts each label once per bin
    triples = np.unique(np.stack([rows[keep], coordinates[keep], targets[keep]], axis=1), axis=0)  # (T, 3)
    counts = np.bincount(triples[:, 2], minlength=len(class_names))  # (C,)
    frame = pd.DataFrame(triples, columns=["pixel", "bin", "target"])
    frame["size"] = frame.groupby(["pixel", "bin"]).target.transform("size")
    shared = frame[frame["size"] > 1]
    pairs = shared.merge(shared, on=["pixel", "bin"], suffixes=("_a", "_b"))
    pairs = pairs[pairs.target_a < pairs.target_b]
    table = pairs.groupby(["target_a", "target_b"]).agg(co_annotations=("pixel", "size"),
                                                        pixels=("pixel", "nunique")).reset_index()
    table = table.rename(columns={"target_a": "class_a", "target_b": "class_b"})
    table["class_a_name"] = [class_names[value] for value in table.class_a]
    table["class_b_name"] = [class_names[value] for value in table.class_b]
    return table, counts


def rare_reference(counts: np.ndarray, eligible: np.ndarray, fraction: float) -> np.ndarray:
    """The generator's rare rule: the ``ceil(fraction * |eligible|)`` least annotated eligible classes.

    :param counts: Annotation count per class ``(C,)``.
    :type counts: numpy.ndarray
    :param eligible: Eligible class indices.
    :type eligible: numpy.ndarray
    :param fraction: Rare fraction of the population declaration.
    :type fraction: float
    :return: Sorted rare class indices (ties broken by class index).
    :rtype: numpy.ndarray
    """
    count = max(1, int(np.ceil(len(eligible) * fraction)))
    ordered = sorted((int(counts[target]), int(target)) for target in eligible)
    return np.sort(np.asarray([target for _, target in ordered[:count]], dtype=np.int64))


# --------------------------------------------------
# Section: stage
# --------------------------------------------------

def run_synthetic_reference(context: Any) -> None:
    """Regenerate (or load) the synthetic artifacts and derive the targeted class sets.

    :param context: Precompute context of the common runner.
    :type context: msi_autoencoder_wrapper.analysis.precompute.core.context.AnalysisContext
    :raises ValueError: If the derived sets disagree with the generator statistics.
    """
    from . import pretraining_campaign_cache as contracts

    cache = _cache(context)
    settings = cache.settings
    declarations = artifact_declarations(cache.plan["tasks"], settings["baseline_role"])
    declarations = {axis: value for axis, value in declarations.items() if axis in cache.axes}
    if not declarations:
        logger.info("No synthetic phases among the selected tasks; synthetic reference skipped.")
        return
    contract = contracts.synthetic_reference_contract(cache.contracts["plan"], {
        axis: {key: value for key, value in block.items() if key != "population"}
        for axis, block in declarations.items()})
    level = contracts.resolve_level(settings, "synthetic_reference", contract)
    root = level.directory
    cache.synthetic_root, cache.keys["synthetic_reference"] = root, root.name
    cache.contracts["synthetic_reference"] = contracts.normalized(contract)
    if level.complete:
        logger.info("Reusing synthetic reference %s.", root)
        return
    parameters = cache.plan["parameters"]
    references = reference_task_ids(cache)
    fingerprints = {}
    for axis, declaration in declarations.items():
        logger.info("Building the synthetic reference of axis %s.", axis)
        # Artifact through the runtime entry point
        block = deepcopy(declaration)
        directory = Path(block["artifact"]["cache_directory"])
        if not directory.is_absolute():
            block["artifact"]["cache_directory"] = str(Path(settings["repository_root"]) / directory)
        wrapper = campaign.build_axis_wrapper(parameters[references[axis]])
        dataset = wrapper.active_dataset
        artifact = build_precomputed_synthetic_partitions(dataset, block)["train"].artifact
        fingerprints[axis] = artifact.fingerprint
        config = PrecomputedSyntheticConfig.from_mapping(block)
        class_names = tuple(dataset.get_target_schemas()[settings["target_field"]].class_names)
        ## Generator statistics on the same training dataset
        builder = builder_module.SyntheticPrecomputeBuilder(dataset, config, fingerprint=artifact.fingerprint)
        eligible = np.asarray(builder.eligible_targets, dtype=np.int64)
        # Class sets read from the manifests
        exposure = exposure_table(dict(artifact.manifests), len(class_names))
        specifications = {population.name: population for population in config.populations}
        rare_populations = [name for name, spec in specifications.items() if spec.rare_bonus_per_class > 0]
        overlap_populations = [name for name, spec in specifications.items() if spec.overlap_bonus_per_class > 0]
        rare = np.unique(exposure[(exposure.kind == "rare_bonus")
                                  & exposure.population.isin(rare_populations)].class_index.to_numpy(np.int64))
        overlap = np.unique(exposure[(exposure.kind == "overlap_bonus")
                                     & exposure.population.isin(overlap_populations)].class_index.to_numpy(np.int64))
        ## Collision pairs and annotation counts of the training pixels
        ### REMARK: the generator's own train-identity reader, so both passes see the same pixels.
        train_ids = np.asarray(builder_module._partition_source_ids(dataset.create_partitions().train),
                               dtype=np.int64)
        pairs, counts = collision_pairs(dataset.get_mapped_annotation_index(), train_ids, class_names,
                                        int(np.asarray(dataset.active_context.binner.GetXAxis()).size))
        # Consistency with the generator
        fractions = {specifications[name].rare_fraction for name in rare_populations}
        rows = [
            {"check": "annotation_counts", "passed": np.array_equal(counts, np.asarray(builder.class_counts)),
             "detail": "collision-pair pass reproduces the generator class counts"},
            {"check": "overlap_targets", "passed": set(overlap.tolist()) == set(eligible.tolist())
             & set(np.asarray(builder.overlap_targets).tolist()), "detail": f"{overlap.size} overlap classes"},
            {"check": "overlap_from_pairs", "passed": set(overlap.tolist()) == set(eligible.tolist())
             & set(np.concatenate([pairs.class_a, pairs.class_b]).tolist()),
             "detail": f"{len(pairs)} collision pairs"},
        ]
        for fraction in sorted(fractions):
            expected = rare_reference(counts, eligible, fraction)
            rows.append({"check": f"rare_targets_fraction_{fraction:g}", "passed": np.array_equal(expected, rare),
                         "detail": f"{rare.size} rare classes"})
        checks = pd.DataFrame(rows)
        if not checks.passed.all():
            raise ValueError(f"Synthetic reference of {axis} is inconsistent:\n{checks.to_string(index=False)}")
        ranks = pd.Series(counts[eligible], index=eligible).rank(method="first").reindex(np.arange(len(class_names)))
        classes = pd.DataFrame({"class_index": np.arange(len(class_names)), "class_name": class_names,
                                "eligible": np.isin(np.arange(len(class_names)), eligible),
                                "rare": np.isin(np.arange(len(class_names)), rare),
                                "overlap": np.isin(np.arange(len(class_names)), overlap),
                                "annotation_count": counts, "eligible_count_rank": ranks.to_numpy()})
        axis_dir = root / axis_directory(settings, axis)
        axis_dir.mkdir(parents=True, exist_ok=True)
        classes.to_csv(axis_dir / "classes.csv", index=False)
        exposure.assign(class_name=[class_names[value] for value in exposure.class_index]).to_csv(
            axis_dir / "exposure.csv", index=False)
        population_summary(dict(artifact.manifests)).to_csv(axis_dir / "population_summary.csv", index=False)
        pairs.to_csv(axis_dir / "collision_pairs.csv", index=False)
        checks.to_csv(axis_dir / "checks.csv", index=False)
        logger.info("Synthetic reference of %s: %s eligible, %s rare, %s overlap classes, %s collision pairs.",
                    axis, eligible.size, rare.size, overlap.size, len(pairs))
        del wrapper, dataset, artifact, builder
        gc.collect()
    _atomic_json(root / "complete.json", {"artifact_fingerprints": fingerprints,
                                          "provenance": provenance_record(settings)})
