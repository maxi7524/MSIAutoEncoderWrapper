"""Import manually downloaded METASPACE files into the canonical catalog."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping

from ..annotations import read_canonical_csv_annotations
from ..imzml import PyImzMLReader
from ..utils.logger import get_custom_logger
from ..catalog.sqlite_catalog import DatasetCatalog


logger = get_custom_logger(__name__)


def import_local_dataset(
    *,
    catalog: DatasetCatalog,
    source: str,
    dataset_id: str,
    name: str,
    imzml_path: Path | str,
    annotations_path: Path | str,
    pixel_intensities_path: Path | str,
    metadata: Mapping[str, Any] | None = None,
) -> Dict[str, int]:
    """Import one local imzML pair and its METASPACE CSV exports.

    :param catalog: Destination canonical SQLite catalog.
    :type catalog: DatasetCatalog
    :param source: Source strategy name.
    :type source: str
    :param dataset_id: Stable source dataset identifier.
    :type dataset_id: str
    :param name: Human-readable dataset name.
    :type name: str
    :param imzml_path: Existing imzML file with a sibling ``.ibd`` file.
    :type imzml_path: pathlib.Path | str
    :param annotations_path: METASPACE annotation table CSV.
    :type annotations_path: pathlib.Path | str
    :param pixel_intensities_path: METASPACE pixel-intensity CSV.
    :type pixel_intensities_path: pathlib.Path | str
    :param metadata: Complete known source metadata.
    :type metadata: Mapping[str, Any] | None
    :return: Imported spectrum, annotation, and spatial-link counts.
    :rtype: Dict[str, int]
    :raises ValidationError: If files or annotation rows cannot be matched.
    """
    imzml = Path(imzml_path)
    _, records = read_canonical_csv_annotations(
        imzml,
        annotations_path,
        pixel_intensities_path,
    )
    reader = PyImzMLReader(imzml)
    spatial_links = sum(len(record["spectrum_ids"]) for record in records)

    catalog.upsert_dataset(
        source=source,
        dataset_id=dataset_id,
        name=name,
        metadata=dict(metadata or {}),
        local_path=imzml.parent,
        status="materialized",
    )
    catalog.replace_annotations(source=source, dataset_id=dataset_id, annotations=records)
    result = {
        "spectra": reader.GetNumberOfSpectra(),
        "annotations": len(records),
        "spatial_links": spatial_links,
    }
    logger.info("Imported local dataset %s: %s", dataset_id, result)
    return result
