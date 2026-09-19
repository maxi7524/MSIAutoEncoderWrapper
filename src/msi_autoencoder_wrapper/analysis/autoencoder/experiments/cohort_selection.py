"""Reproducible selection of externally held-out MSI images from a cohort.

The selection is deliberately based only on lightweight cohort metadata.  It reads
the annotation CSV and the header of the accompanying pixel-intensity CSV, rather
than loading imzML spectra or a dense intensity matrix.  This makes it suitable as a
preflight check before expensive model construction while retaining the two properties
that determine whether an external test selection is safe:

* selected images are near the cohort centre in pixel and annotation population;
* every molecular label remains represented by at least one training image.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import subprocess
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from ....utils.exceptions import raise_validation_error
from ....utils.logger import get_custom_logger

logger = get_custom_logger(__name__)

_COORDINATE_PREFIX = "x"
_COORDINATE_SEPARATOR = "_y"


@dataclass(frozen=True)
class CohortImageInventory:
    """Metadata required to select one external test image.

    :param image_key: Directory name and stable cohort member identity.
    :type image_key: str
    :param dataset_name: Human-readable dataset name recorded by METASPACE.
    :type dataset_name: str
    :param pixel_count: Number of coordinate columns in ``pixel_intensities.csv``.
    :type pixel_count: int
    :param annotation_records: Number of annotation records, including source-database
        duplicates.
    :type annotation_records: int
    :param labels: Unique ``formula|adduct`` labels available in the image.
    :type labels: frozenset[str]
    """

    image_key: str
    dataset_name: str
    pixel_count: int
    annotation_records: int
    labels: frozenset[str]

    @property
    def unique_label_count(self) -> int:
        """Return the number of unique molecular labels."""
        return len(self.labels)


@dataclass(frozen=True)
class CohortSelection:
    """A lossless held-out image selection and diagnostics.

    :param heldout_image_keys: Stable image identities reserved exclusively for test.
    :type heldout_image_keys: tuple[str, ...]
    :param candidate_image_keys: Middle-population images considered by the optimizer.
    :type candidate_image_keys: tuple[str, ...]
    :param all_label_count: Unique labels across every eligible image.
    :type all_label_count: int
    :param retained_label_count: Labels still available from the training images.
    :type retained_label_count: int
    :param score: Deterministic centrality score of the selected combination.
    :type score: float
    """

    heldout_image_keys: tuple[str, ...]
    candidate_image_keys: tuple[str, ...]
    all_label_count: int
    retained_label_count: int
    score: float

    @property
    def uncovered_label_count(self) -> int:
        """Return labels that would disappear from the training population."""
        return self.all_label_count - self.retained_label_count


def scan_cohort(
    dataset_root: Path | str,
    *,
    mass_range: tuple[float, float] | None = None,
) -> tuple[CohortImageInventory, ...]:
    """Read metadata for every complete image directory in a cohort.

    A complete image has ``annotations.csv`` and ``pixel_intensities.csv``.  The
    latter is read only up to the header, so the operation is independent of the
    number of annotation-intensity rows.

    :param dataset_root: Directory containing one subdirectory per image.
    :type dataset_root: pathlib.Path | str
    :param mass_range: Optional inclusive m/z interval of the planned model axis.
        Rows outside it cannot create a target column and are excluded from all
        annotation counts and label sets.
    :type mass_range: tuple[float, float] | None
    :return: Sorted immutable inventory records.
    :rtype: tuple[CohortImageInventory, ...]
    :raises ValidationError: If no complete image is found or an annotations file has
        no usable ``formula``/``adduct`` columns.
    """
    root = Path(dataset_root)
    inventories: list[CohortImageInventory] = []
    for image_directory in sorted(path for path in root.iterdir() if path.is_dir()):
        annotations_path = image_directory / "annotations.csv"
        intensities_path = image_directory / "pixel_intensities.csv"
        if not annotations_path.is_file() or not intensities_path.is_file():
            logger.debug("Skipping incomplete cohort member '%s'.", image_directory)
            continue
        inventories.append(
            _scan_image(
                image_directory,
                annotations_path,
                intensities_path,
                mass_range=mass_range,
            )
        )

    if not inventories:
        raise_validation_error(
            "CohortSelection",
            f"No complete image directories found under '{root}'.",
        )
    logger.info("Scanned %s complete cohort image(s) from '%s'.", len(inventories), root)
    return tuple(inventories)


def select_middle_heldout_images(
    inventories: Sequence[CohortImageInventory],
    *,
    heldout_count: int = 3,
    middle_quantiles: tuple[float, float] = (0.2, 0.8),
) -> CohortSelection:
    """Choose central, lossless images as an external test set.

    Each candidate must lie inside the specified quantile interval for the three
    log-scaled quantities: pixel count, annotation-record count and unique-label
    count.  The optimizer then examines every candidate combination of the requested
    size.  It accepts only combinations whose removal leaves every molecular label in
    the training population, and minimizes robust distance to the cohort median.
    Ties are resolved lexicographically, so no random state is involved.

    :param inventories: Complete image metadata returned by :func:`scan_cohort`.
    :type inventories: collections.abc.Sequence[CohortImageInventory]
    :param heldout_count: Number of externally held-out images.
    :type heldout_count: int
    :param middle_quantiles: Inclusive lower and upper quantiles in ``[0, 1]``.
    :type middle_quantiles: tuple[float, float]
    :return: Deterministic held-out selection and label-coverage diagnostics.
    :rtype: CohortSelection
    :raises ValidationError: If the cohort cannot form a lossless selection.
    """
    if heldout_count < 1 or heldout_count >= len(inventories):
        raise_validation_error(
            "CohortSelection",
            "'heldout_count' must be at least one and smaller than the cohort size.",
        )
    lower, upper = middle_quantiles
    if not 0.0 <= lower <= upper <= 1.0:
        raise_validation_error(
            "CohortSelection",
            "'middle_quantiles' must satisfy 0 <= lower <= upper <= 1.",
        )

    metrics = _metric_matrix(inventories)  # (N, 3)
    quantile_bounds = np.quantile(metrics, [lower, upper], axis=0)  # (2, 3)
    candidate_mask = np.logical_and(
        metrics >= quantile_bounds[0], metrics <= quantile_bounds[1]
    ).all(axis=1)  # (N,)
    candidates = tuple(
        inventory
        for inventory, is_candidate in zip(inventories, candidate_mask)
        if bool(is_candidate)
    )
    if len(candidates) < heldout_count:
        raise_validation_error(
            "CohortSelection",
            "The middle-population filter leaves fewer images than 'heldout_count'.",
        )

    all_labels = frozenset().union(*(inventory.labels for inventory in inventories))
    metric_median = np.median(metrics, axis=0)  # (3,)
    metric_mad = np.median(np.abs(metrics - metric_median), axis=0)  # (3,)
    metric_scale = np.where(metric_mad > 0.0, metric_mad, 1.0)  # (3,)

    best: tuple[float, tuple[str, ...], tuple[CohortImageInventory, ...]] | None = None
    for selected in itertools.combinations(candidates, heldout_count):
        heldout_keys = tuple(inventory.image_key for inventory in selected)
        training_labels = frozenset().union(
            *(inventory.labels for inventory in inventories if inventory not in selected)
        )
        if training_labels != all_labels:
            continue
        selected_metrics = _metric_matrix(selected)  # (H, 3)
        score = float(np.mean(np.abs((selected_metrics - metric_median) / metric_scale)))
        candidate = (score, heldout_keys, selected)
        if best is None or candidate[:2] < best[:2]:
            best = candidate

    if best is None:
        raise_validation_error(
            "CohortSelection",
            "No lossless held-out combination exists in the middle-population pool.",
        )
    score, heldout_keys, selected = best
    retained_labels = frozenset().union(
        *(inventory.labels for inventory in inventories if inventory not in selected)
    )
    return CohortSelection(
        heldout_image_keys=heldout_keys,
        candidate_image_keys=tuple(inventory.image_key for inventory in candidates),
        all_label_count=len(all_labels),
        retained_label_count=len(retained_labels),
        score=score,
    )


def write_cohort_selection(
    settings: Mapping[str, Any],
    *,
    settings_path: Path | str | None = None,
) -> CohortSelection:
    """Run the cohort scan, write canonical tables and return the selection.

    :param settings: Mapping with ``dataset_root``, ``output_directory``,
        ``heldout_count`` and optional ``excluded_image_keys``/``middle_quantiles``.
    :type settings: collections.abc.Mapping[str, typing.Any]
    :param settings_path: Source settings path recorded in provenance when available.
    :type settings_path: pathlib.Path | str | None
    :return: Selected held-out images and coverage diagnostics.
    :rtype: CohortSelection
    """
    start_time = time.perf_counter()
    dataset_root = Path(settings["dataset_root"])
    output_directory = Path(settings["output_directory"])
    excluded_image_keys = frozenset(settings.get("excluded_image_keys", ()))
    inventories = tuple(
        inventory
        for inventory in scan_cohort(
            dataset_root,
            mass_range=_mass_range(settings.get("mass_range")),
        )
        if inventory.image_key not in excluded_image_keys
    )
    selection = select_middle_heldout_images(
        inventories,
        heldout_count=int(settings.get("heldout_count", 3)),
        middle_quantiles=tuple(settings.get("middle_quantiles", (0.2, 0.8))),
    )

    # Persist auditable population and coverage tables.
    ## These tables intentionally contain no dense spectral matrix or annotation JSON.
    output_directory.mkdir(parents=True, exist_ok=True)
    heldout_set = frozenset(selection.heldout_image_keys)
    summary = _inventory_frame(inventories, heldout_set, selection.candidate_image_keys)
    summary.to_csv(output_directory / "cohort_image_summary.csv", index=False)
    summary.loc[summary["is_heldout"]].to_csv(
        output_directory / "heldout_selection.csv", index=False
    )
    _label_coverage_frame(inventories, heldout_set).to_csv(
        output_directory / "training_label_coverage.csv", index=False
    )

    elapsed_seconds = time.perf_counter() - start_time
    provenance = {
        "dataset_root": str(dataset_root.resolve()),
        "settings_path": None if settings_path is None else str(Path(settings_path).resolve()),
        "excluded_image_keys": sorted(excluded_image_keys),
        "heldout_count": len(selection.heldout_image_keys),
        "middle_quantiles": list(settings.get("middle_quantiles", (0.2, 0.8))),
        "mass_range": settings.get("mass_range"),
        "heldout_image_keys": list(selection.heldout_image_keys),
        "candidate_image_count": len(selection.candidate_image_keys),
        "all_label_count": selection.all_label_count,
        "retained_label_count": selection.retained_label_count,
        "uncovered_label_count": selection.uncovered_label_count,
        "selection_score": selection.score,
        "analysis_seconds": elapsed_seconds,
        "git_commit": _git_commit(),
    }
    (output_directory / "provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    logger.info(
        "Selected held-out images %s; preserved %s/%s labels in %.3f seconds.",
        selection.heldout_image_keys,
        selection.retained_label_count,
        selection.all_label_count,
        elapsed_seconds,
    )
    return selection


def load_cohort_selection_results(output_directory: Path | str) -> dict[str, pd.DataFrame]:
    """Load the canonical tables written by :func:`write_cohort_selection`.

    :param output_directory: Result directory configured for the cohort analysis.
    :type output_directory: pathlib.Path | str
    :return: Frames keyed by ``images``, ``heldout`` and ``label_coverage``.
    :rtype: dict[str, pandas.DataFrame]
    :raises FileNotFoundError: If the analysis has not been executed yet.
    """
    directory = Path(output_directory)
    return {
        "images": pd.read_csv(directory / "cohort_image_summary.csv"),
        "heldout": pd.read_csv(directory / "heldout_selection.csv"),
        "label_coverage": pd.read_csv(directory / "training_label_coverage.csv"),
    }


def _scan_image(
    image_directory: Path,
    annotations_path: Path,
    intensities_path: Path,
    *,
    mass_range: tuple[float, float] | None,
) -> CohortImageInventory:
    """Extract one image's annotation labels and header-only pixel count."""
    labels: set[str] = set()
    annotation_records = 0
    dataset_name = image_directory.name
    with annotations_path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        required_fields = {"formula", "adduct"}
        if mass_range is not None:
            required_fields.add("mz")
        if not reader.fieldnames or not required_fields.issubset(reader.fieldnames):
            raise_validation_error(
                "CohortSelection",
                f"'{annotations_path}' lacks required columns {sorted(required_fields)}.",
            )
        for row in reader:
            if mass_range is not None:
                try:
                    mass = float(row.get("mz") or "nan")
                except ValueError:
                    mass = float("nan")
                if not math.isfinite(mass) or not mass_range[0] <= mass <= mass_range[1]:
                    continue
            annotation_records += 1
            formula = (row.get("formula") or "").strip()
            adduct = (row.get("adduct") or "").strip()
            if formula and adduct:
                labels.add(f"{formula}|{adduct}")
            if row.get("datasetName"):
                dataset_name = str(row["datasetName"])
    with intensities_path.open(newline="", encoding="utf-8") as stream:
        header = next(csv.reader(stream), [])
    pixel_count = sum(_is_coordinate_column(column) for column in header)
    if not labels:
        raise_validation_error(
            "CohortSelection",
            f"'{annotations_path}' contains no usable formula/adduct labels.",
        )
    if pixel_count == 0:
        raise_validation_error(
            "CohortSelection",
            f"'{intensities_path}' contains no coordinate columns.",
        )
    return CohortImageInventory(
        image_key=image_directory.name,
        dataset_name=dataset_name,
        pixel_count=pixel_count,
        annotation_records=annotation_records,
        labels=frozenset(labels),
    )


def _mass_range(value: Any) -> tuple[float, float] | None:
    """Validate an optional inclusive axis range declared in the settings."""
    if value is None:
        return None
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise_validation_error("CohortSelection", "mass_range must contain [minimum, maximum].")
    lower, upper = (float(item) for item in value)
    if not math.isfinite(lower) or not math.isfinite(upper) or lower >= upper:
        raise_validation_error("CohortSelection", "mass_range must be finite and increasing.")
    return lower, upper


def _is_coordinate_column(column: str) -> bool:
    """Return whether a pixel-intensity header field has the ``x<int>_y<int>`` form."""
    if not column.startswith(_COORDINATE_PREFIX) or _COORDINATE_SEPARATOR not in column:
        return False
    x, y = column[1:].split(_COORDINATE_SEPARATOR, maxsplit=1)
    return x.isdecimal() and y.isdecimal()


def _metric_matrix(inventories: Sequence[CohortImageInventory]) -> np.ndarray:
    """Return log-scaled population metrics with shape ``(N, 3)``."""
    return np.asarray(
        [
            (
                math.log1p(inventory.pixel_count),
                math.log1p(inventory.annotation_records),
                math.log1p(inventory.unique_label_count),
            )
            for inventory in inventories
        ],
        dtype=np.float64,
    )  # (N, 3)


def _inventory_frame(
    inventories: Sequence[CohortImageInventory],
    heldout_keys: frozenset[str],
    candidate_keys: Sequence[str],
) -> pd.DataFrame:
    """Materialize the image-level population table without nested labels."""
    candidate_set = frozenset(candidate_keys)
    frame = pd.DataFrame(
        {
            "image_key": [inventory.image_key for inventory in inventories],
            "dataset_name": [inventory.dataset_name for inventory in inventories],
            "pixel_count": [inventory.pixel_count for inventory in inventories],
            "annotation_records": [inventory.annotation_records for inventory in inventories],
            "unique_label_count": [inventory.unique_label_count for inventory in inventories],
            "is_middle_candidate": [inventory.image_key in candidate_set for inventory in inventories],
            "is_heldout": [inventory.image_key in heldout_keys for inventory in inventories],
        }
    )
    for column in ("pixel_count", "annotation_records", "unique_label_count"):
        frame[f"{column}_percentile"] = frame[column].rank(pct=True, method="average")
    return frame.sort_values("image_key", kind="stable").reset_index(drop=True)


def _label_coverage_frame(
    inventories: Sequence[CohortImageInventory],
    heldout_keys: frozenset[str],
) -> pd.DataFrame:
    """Return image-frequency coverage of every label before and after selection."""
    all_counts: Counter[str] = Counter()
    train_counts: Counter[str] = Counter()
    for inventory in inventories:
        all_counts.update(inventory.labels)
        if inventory.image_key not in heldout_keys:
            train_counts.update(inventory.labels)
    return pd.DataFrame(
        {
            "label": sorted(all_counts),
            "all_image_count": [all_counts[label] for label in sorted(all_counts)],
            "training_image_count": [train_counts[label] for label in sorted(all_counts)],
        }
    )


def _git_commit() -> str | None:
    """Return the current commit when the analysis runs inside a repository."""
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=Path.cwd(),
        text=True,
        capture_output=True,
        check=False,
    )
    return result.stdout.strip() or None


def _load_settings(path: Path) -> dict[str, Any]:
    """Load one YAML settings file and resolve paths relative to its location."""
    with path.open(encoding="utf-8") as stream:
        settings = yaml.safe_load(stream) or {}
    if not isinstance(settings, dict):
        raise_validation_error("CohortSelection", "Settings YAML must contain a mapping.")
    for key in ("dataset_root", "output_directory"):
        if key not in settings:
            raise_validation_error("CohortSelection", f"Settings require '{key}'.")
        candidate = Path(settings[key])
        settings[key] = str(candidate if candidate.is_absolute() else (path.parent / candidate).resolve())
    return settings


def main(argv: Sequence[str] | None = None) -> int:
    """Run cohort selection from a YAML file.

    :param argv: Optional command-line arguments, excluding the executable name.
    :type argv: collections.abc.Sequence[str] | None
    :return: Process exit status.
    :rtype: int
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--settings", required=True, type=Path)
    arguments = parser.parse_args(argv)
    settings_path = arguments.settings.resolve()
    write_cohort_selection(_load_settings(settings_path), settings_path=settings_path)
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI wrapper
    raise SystemExit(main())
