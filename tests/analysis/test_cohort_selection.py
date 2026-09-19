"""Tests for metadata-only external cohort selection."""

from __future__ import annotations

import csv
from pathlib import Path

from msi_autoencoder_wrapper.analysis.autoencoder.experiments.cohort_selection import (
    load_cohort_selection_results,
    scan_cohort,
    select_middle_heldout_images,
    write_cohort_selection,
)


def _write_image(
    root: Path,
    image_key: str,
    *,
    pixels: int,
    labels: tuple[tuple[str, str], ...],
    records: int,
) -> None:
    """Write a small metadata fixture with deterministic annotation duplicates."""
    image_directory = root / image_key
    image_directory.mkdir()
    with (image_directory / "annotations.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["datasetName", "formula", "adduct"])
        writer.writeheader()
        for index in range(records):
            formula, adduct = labels[index % len(labels)]
            writer.writerow(
                {"datasetName": f"dataset-{image_key}", "formula": formula, "adduct": adduct}
            )
    header = ["formula", "adduct", *(f"x{index}_y0" for index in range(pixels))]
    with (image_directory / "pixel_intensities.csv").open("w", newline="", encoding="utf-8") as stream:
        csv.writer(stream).writerow(header)


def test_selects_lossless_middle_images_and_writes_tables(tmp_path: Path) -> None:
    """Selection excludes central images while retaining every training label."""
    dataset_root = tmp_path / "datasets"
    dataset_root.mkdir()
    common = (("C1", "+H"), ("C2", "+H"))
    _write_image(dataset_root, "small", pixels=10, labels=common, records=2)
    _write_image(dataset_root, "middle_a", pixels=20, labels=common, records=3)
    _write_image(dataset_root, "middle_b", pixels=21, labels=common, records=3)
    _write_image(dataset_root, "middle_c", pixels=22, labels=common, records=3)
    _write_image(dataset_root, "large", pixels=100, labels=common + (("C3", "+H"),), records=20)

    inventories = scan_cohort(dataset_root)
    selection = select_middle_heldout_images(
        inventories,
        heldout_count=2,
        middle_quantiles=(0.2, 0.8),
    )

    assert selection.uncovered_label_count == 0
    assert set(selection.heldout_image_keys).issubset(set(selection.candidate_image_keys))

    output_directory = tmp_path / "result"
    written = write_cohort_selection(
        {
            "dataset_root": dataset_root,
            "output_directory": output_directory,
            "heldout_count": 2,
            "middle_quantiles": (0.2, 0.8),
        }
    )
    results = load_cohort_selection_results(output_directory)
    assert written == selection
    assert results["heldout"]["image_key"].tolist() == list(selection.heldout_image_keys)
    assert (results["label_coverage"]["training_image_count"] > 0).all()
