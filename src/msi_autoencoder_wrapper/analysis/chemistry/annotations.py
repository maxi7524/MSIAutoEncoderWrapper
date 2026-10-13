"""METASPACE annotation collections used to build chemical label graphs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import pandas as pd

from ...utils.logger import get_custom_logger

logger = get_custom_logger(__name__)

ANNOTATION_COLUMNS = (
    "dataset_id", "ion_id", "formula", "adduct", "mz", "fdr", "msm", "database",
    "off_sample", "identifiers", "compound_names",
)


def _possible_compounds(source_record_json: str) -> tuple[list[str], list[str]]:
    """Return database identifiers and names of METASPACE possible compounds."""
    record = json.loads(source_record_json) if isinstance(source_record_json, str) and source_record_json else {}
    identifiers, names = [], []
    for compound in record.get("possibleCompounds") or []:
        names.append(str(compound.get("name") or ""))
        identifiers.extend(str(item.get("databaseId")) for item in compound.get("information") or [] if item.get("databaseId"))
    return identifiers, names


def read_annotation_collection(
    dataset_root: Path | str,
    *,
    dataset_ids: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Read every ``annotations.csv`` of one METASPACE dataset collection.

    One row is one METASPACE annotation of an ion in one dataset and one annotation
    database. An ion is identified by ``formula|adduct`` exactly as in the molecule
    target vocabulary of the training datasets.

    :param dataset_root: Directory containing one sub-directory per METASPACE dataset.
    :type dataset_root: pathlib.Path | str
    :param dataset_ids: Optional explicit dataset subset; all complete datasets when
        ``None``.
    :type dataset_ids: collections.abc.Sequence[str] | None
    :return: Long annotation table with :data:`ANNOTATION_COLUMNS`; ``identifiers``
        and ``compound_names`` are JSON lists.
    :rtype: pandas.DataFrame
    :raises FileNotFoundError: If a requested dataset has no ``annotations.csv``.
    """
    root = Path(dataset_root)
    selected = list(dataset_ids) if dataset_ids is not None else sorted(
        path.name for path in root.iterdir()
        if (path / "annotations.csv").is_file() and (path / "pixel_intensities.csv").is_file()
    )
    frames = []
    for dataset_id in selected:
        path = root / dataset_id / "annotations.csv"
        if not path.is_file():
            raise FileNotFoundError(f"Dataset '{dataset_id}' has no annotations at '{path}'.")
        raw = pd.read_csv(path)
        parsed = [_possible_compounds(value) for value in raw.source_record_json]
        records = [json.loads(value) if isinstance(value, str) else {} for value in raw.source_record_json]
        frames.append(pd.DataFrame({
            "dataset_id": dataset_id,
            "ion_id": raw.formula.astype(str) + "|" + raw.adduct.astype(str),
            "formula": raw.formula.astype(str),
            "adduct": raw.adduct.astype(str),
            "mz": raw.mz.astype(float),
            "fdr": raw.fdr.astype(float),
            "msm": [float(record.get("msmScore") or float("nan")) for record in records],
            "database": raw.database_name.astype(str) + " " + raw.database_version.astype(str),
            "off_sample": [bool(record.get("offSample")) for record in records],
            "identifiers": [json.dumps(sorted(set(ids))) for ids, _ in parsed],
            "compound_names": [json.dumps(sorted(set(names))) for _, names in parsed],
        }))
    table = pd.concat(frames, ignore_index=True)
    logger.info("Read %s annotations of %s ions from %s datasets in %s.",
                len(table), table.ion_id.nunique(), len(selected), root)
    return table.loc[:, list(ANNOTATION_COLUMNS)]


def ion_table(annotations: pd.DataFrame) -> pd.DataFrame:
    """Aggregate annotation rows to one record per ion.

    :param annotations: Output of :func:`read_annotation_collection`.
    :type annotations: pandas.DataFrame
    :return: One row per ``ion_id`` with formula, adduct, median ``m/z``, number of
        datasets, best (minimum) FDR, annotation databases, the union of reported
        identifiers (JSON list), and the off-sample fraction.
    :rtype: pandas.DataFrame
    """
    rows = []
    for ion_id, group in annotations.groupby("ion_id", sort=True):
        identifiers = sorted({value for item in group.identifiers for value in json.loads(item)})
        names = sorted({value for item in group.compound_names for value in json.loads(item)})
        rows.append({
            "ion_id": ion_id,
            "formula": group.formula.iloc[0],
            "adduct": group.adduct.iloc[0],
            "mz": float(group.mz.median()),
            "dataset_count": int(group.dataset_id.nunique()),
            "best_fdr": float(group.fdr.min()),
            "databases": ";".join(sorted(set(group.database))),
            "identifiers": json.dumps(identifiers),
            "identifier_count": len(identifiers),
            "compound_names": json.dumps(names),
            "off_sample_fraction": float(group.off_sample.mean()),
        })
    return pd.DataFrame(rows)
