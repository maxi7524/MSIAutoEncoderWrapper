"""Regression tests for canonical paired annotation CSV imports."""

from pathlib import Path

import numpy as np
from pyimzml.ImzMLWriter import ImzMLWriter

from msi_dataset_manager.annotations import read_canonical_csv_annotations


def test_isobaric_ions_are_matched_by_formula_adduct_and_mz(
    tmp_path: Path,
) -> None:
    """Different ions sharing m/z retain their respective spatial values."""
    annotations = tmp_path / "annotations.csv"
    intensities = tmp_path / "pixel_intensities.csv"
    image = tmp_path / "image.imzML"
    with ImzMLWriter(str(image), mode="processed") as writer:
        writer.addSpectrum(
            np.asarray([100.5]),
            np.asarray([1.0]),
            (1, 1, 1),
        )
    annotations.write_text(
        "datasetId,datasetName,formula,adduct,mz,fdr\n"
        "one,One,C1,[M]-,100.5,0.05\n"
        "one,One,C2,-H,100.5,0.05\n",
        encoding="utf-8",
    )
    intensities.write_text(
        "mol_formula,adduct,mz,x0_y0\n"
        "C1,[M]-,100.5,3.0\n"
        "C2,-H,100.5,7.0\n",
        encoding="utf-8",
    )

    _, records = read_canonical_csv_annotations(
        image,
        annotations,
        intensities,
    )

    assert [record["formula"] for record in records] == ["C1", "C2"]
    assert records[0]["spectrum_values"][0] == 3.0
    assert records[1]["spectrum_values"][0] == 7.0
