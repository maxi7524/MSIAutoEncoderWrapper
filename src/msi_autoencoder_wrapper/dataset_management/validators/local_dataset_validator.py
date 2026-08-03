"""Validation of local canonical MSI datasets."""

from __future__ import annotations

from pathlib import Path

from ...utils.exceptions import raise_validation_error


def validate_imzml_pair(dataset_directory: Path | str, dataset_id: str) -> Path:
    """Validate and return the expected imzML file in a downloaded dataset.

    :param dataset_directory: Materialized dataset directory.
    :type dataset_directory: pathlib.Path | str
    :param dataset_id: Stable source dataset identifier and file stem.
    :type dataset_id: str
    :return: Validated imzML path.
    :rtype: pathlib.Path
    :raises ValidationError: If either required file is missing.
    """
    imzml_path = Path(dataset_directory) / f"{dataset_id}.imzML"
    if not imzml_path.is_file() or not imzml_path.with_suffix(".ibd").is_file():
        raise_validation_error(
            "CanonicalDataset",
            f"Dataset '{dataset_id}' does not contain a complete imzML/ibd pair.",
        )
    return imzml_path
