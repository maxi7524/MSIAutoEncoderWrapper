#!/usr/bin/env python3
"""Build the representative kidney MSI image used by the toy models.

The toy image joins two METASPACE kidney sections side by side into one processed
imzML file restricted to an m/z window, with the canonical ``annotations.csv`` +
``pixel_intensities.csv`` pair, so the library reads it exactly like every other
workspace image.

Selection (all values are module constants and recorded in ``provenance.json``):

- source images ``SOURCE_DATASET_IDS`` (two sections of the same series, d28-2017-2 and
  d28-2006-2, sharing seven annotated ions inside the window);
- every ``LATTICE_STRIDE``-th pixel in x and y of every source (1 = all pixels);
- peaks inside ``[MZ_MIN, MZ_MAX]`` only;
- every annotated ion inside the window of any source (union, one row per ion). An ion
  that is not annotated in one source has no positive pixels there, i.e. its pixels are
  unlabelled for that ion, which is the positive-unlabelled reading of METASPACE.

Sources are placed left to right with ``SOURCE_GAP`` empty columns between them;
``provenance.json`` records the column offset of every source.

Usage::

    .venv/bin/python assets/experiments/autoencoder_architecture/experiment_runs_configs/representative_model/build_representative_dataset.py
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd
from pyimzml.ImzMLParser import ImzMLParser
from pyimzml.ImzMLWriter import ImzMLWriter

from msi_autoencoder_wrapper.utils.logger import get_custom_logger

logger = get_custom_logger(__name__)

REPOSITORY_ROOT = Path(__file__).resolve().parents[5]
DATASETS_DIRECTORY = REPOSITORY_ROOT / "data/kidney_workspace/datasets"
SOURCE_DATASET_IDS = ("2024-02-20_01h57m32s", "2024-02-20_01h54m41s")
OUTPUT_DATASET_ID = "representative_dataset"
OUTPUT_DIRECTORY = DATASETS_DIRECTORY / OUTPUT_DATASET_ID

LATTICE_STRIDE = 1
SOURCE_GAP = 4
MZ_MIN = 200.0
MZ_MAX = 950.0
MZ_MATCH_DECIMALS = 3


def _read_source(dataset_id: str) -> tuple[ImzMLParser, pd.DataFrame, pd.DataFrame]:
    """Parser, in-window annotations (one row per ion) and pixel intensities of one source."""
    directory = DATASETS_DIRECTORY / dataset_id
    annotations = pd.read_csv(directory / "annotations.csv", dtype=str, keep_default_na=False)
    annotations["mz_key"] = annotations["mz"].astype(float).round(MZ_MATCH_DECIMALS)
    annotations = annotations[(annotations.mz_key >= MZ_MIN) & (annotations.mz_key <= MZ_MAX)]
    annotations = annotations.drop_duplicates("mz_key").sort_values("mz_key")
    intensities = pd.read_csv(directory / "pixel_intensities.csv", dtype=str, keep_default_na=False)
    intensities = intensities.set_index("source_annotation_id")
    return ImzMLParser(str(directory / f"{dataset_id}.imzML")), annotations, intensities


def build(output_directory: Path = OUTPUT_DIRECTORY) -> dict:
    """Write the representative imzML image and its annotation CSV pair.

    :param output_directory: Target dataset directory; created when absent.
    :type output_directory: pathlib.Path
    :return: Provenance summary written to ``provenance.json``.
    :rtype: dict
    :raises ValueError: If the selection is empty.
    """
    output_directory.mkdir(parents=True, exist_ok=True)
    sources = {dataset_id: _read_source(dataset_id) for dataset_id in SOURCE_DATASET_IDS}

    # Ion union
    ## The first source annotating an ion provides its annotation row.
    union = pd.concat([annotations for _, annotations, _ in sources.values()]).drop_duplicates("mz_key")
    union = union.sort_values("mz_key").reset_index(drop=True)
    logger.info("Selected %s annotated ions across %s sources.", len(union), len(sources))

    # Pixel placement and imzML output
    ## Sources are placed left to right; coordinates are zero-based in CSV and one-based in imzML.
    imzml_path = output_directory / f"{OUTPUT_DATASET_ID}.imzML"
    column_values: dict[str, dict[float, str]] = {}  # target column -> ion m/z -> intensity
    source_records, offset, dropped = [], 0, 0
    with ImzMLWriter(str(imzml_path), mode="processed", spec_type="centroid") as writer:
        for dataset_id, (parser, annotations, intensities) in sources.items():
            coordinates = np.asarray(parser.coordinates)[:, :2] - 1  # (N, 2)
            on_lattice = np.all(coordinates % LATTICE_STRIDE == 0, axis=1)  # (N,)
            compact = coordinates // LATTICE_STRIDE  # (N, 2)
            ion_rows = {row.mz_key: intensities.loc[row.source_annotation_id].to_dict()
                        for row in annotations.itertuples()}
            written = 0
            for index in np.flatnonzero(on_lattice):
                mzs, values = parser.getspectrum(int(index))
                window = (mzs >= MZ_MIN) & (mzs <= MZ_MAX)
                if not window.any():
                    dropped += 1
                    continue
                x, y = int(compact[index, 0]) + offset, int(compact[index, 1])
                writer.addSpectrum(mzs[window], values[window], (x + 1, y + 1, 1))
                source_column = f"x{coordinates[index, 0]}_y{coordinates[index, 1]}"
                column_values[f"x{x}_y{y}"] = {
                    mz: (row.get(source_column) or "0.0") for mz, row in ion_rows.items()
                }
                written += 1
            width = int(compact[on_lattice, 0].max()) + 1
            source_records.append({"dataset_id": dataset_id, "x_offset": offset, "width": width,
                                   "pixel_count": written,
                                   "ions": [float(value) for value in annotations.mz_key]})
            logger.info("Source %s: %s pixels at x offset %s.", dataset_id, written, offset)
            offset += width + SOURCE_GAP
    if not column_values:
        raise ValueError("The pixel selection is empty.")

    # Annotation CSV pair
    ## Rewrite the dataset identifier; intensities are zero where a source does not annotate the ion.
    annotation_rows = union.drop(columns="mz_key").copy()
    annotation_rows["datasetId"] = OUTPUT_DATASET_ID
    annotation_rows["datasetName"] = f"{OUTPUT_DATASET_ID} ({' + '.join(SOURCE_DATASET_IDS)})"
    annotation_rows.to_csv(output_directory / "annotations.csv", index=False, quoting=csv.QUOTE_MINIMAL)
    metadata = pd.DataFrame({
        "source_annotation_id": union.source_annotation_id,
        "mol_formula": union.formula,
        "adduct": union.adduct,
        "mz": union.mz,
        "moleculeNames": "",
        "moleculeIds": "",
    })
    pixel_values = pd.DataFrame(
        {column: [values.get(mz, "0.0") for mz in union.mz_key] for column, values in column_values.items()}
    )
    pd.concat([metadata, pixel_values], axis=1).to_csv(output_directory / "pixel_intensities.csv", index=False)

    # Provenance
    labels = pixel_values.astype(float).to_numpy() > 0  # (C, P)
    provenance = {
        "source_dataset_ids": list(SOURCE_DATASET_IDS),
        "output_dataset_id": OUTPUT_DATASET_ID,
        "lattice_stride": LATTICE_STRIDE,
        "source_gap": SOURCE_GAP,
        "mz_window": [MZ_MIN, MZ_MAX],
        "pixel_count": int(labels.shape[1]),
        "dropped_empty_pixels": dropped,
        "sources": source_records,
        "ions": [
            {
                "source_annotation_id": row.source_annotation_id,
                "formula": row.formula,
                "adduct": row.adduct,
                "mz": float(row.mz),
                "positive_pixels": int(count),
            }
            for row, count in zip(union.itertuples(), labels.sum(axis=1))
        ],
    }
    (output_directory / "provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    logger.info("Wrote %s", imzml_path)
    return provenance


def main() -> int:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, default=OUTPUT_DIRECTORY)
    args = parser.parse_args()
    provenance = build(args.output)
    print(json.dumps({key: value for key, value in provenance.items() if key != "ions"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
