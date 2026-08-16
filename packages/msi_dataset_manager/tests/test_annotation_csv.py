"""Regression tests for canonical paired annotation CSV imports."""

from pathlib import Path

import numpy as np
from pyimzml.ImzMLWriter import ImzMLWriter

from msi_dataset_manager.sources.strategies.metaspace.csv import (
    read_metaspace_annotation_export,
)


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
        "schema_version,source,source_annotation_id,datasetId,datasetName,formula,adduct,mz,fdr\n"
        "1,metaspace,a1,one,One,C1,[M]-,100.5,0.05\n"
        "1,metaspace,a2,one,One,C2,-H,100.5,0.05\n",
        encoding="utf-8",
    )
    intensities.write_text(
        "mol_formula,adduct,mz,x0_y0\n"
        "C1,[M]-,100.5,3.0\n"
        "C2,-H,100.5,7.0\n",
        encoding="utf-8",
    )

    export = read_metaspace_annotation_export(
        dataset_id="one",
        directory=tmp_path,
        imzml_path=image,
    )
    records = export.records

    assert [record["formula"] for record in records] == ["C1", "C2"]
    assert records[0]["spectrum_values"][0] == 3.0
    assert records[1]["spectrum_values"][0] == 7.0
