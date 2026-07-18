"""Build the small MSI test fixture from the original bladder image."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from pyimzml.ImzMLParser import ImzMLParser
from pyimzml.ImzMLWriter import ImzMLWriter

SOURCE_SPECTRUM_INDICES = (0, 1, 2, 260, 261, 262)


def build_fixture(
    source_path: Path,
    output_path: Path,
    spectrum_indices: Sequence[int] = SOURCE_SPECTRUM_INDICES,
) -> None:
    """Write selected source spectra to a compact processed imzML fixture.

    :param source_path: Original bladder image imzML path.
    :type source_path: pathlib.Path
    :param output_path: Destination imzML path. The writer creates the matching
        ibd file next to it.
    :type output_path: pathlib.Path
    :param spectrum_indices: Source spectrum indices retained in the fixture.
    :type spectrum_indices: Sequence[int]
    :raises FileNotFoundError: If the original image is unavailable.

    The fixture preserves the source spectra and coordinates without resampling.
    Its only side effect is writing the destination imzML/ibd pair.
    """
    if not source_path.is_file():
        raise FileNotFoundError(f"Source imzML file does not exist: {source_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    parser = ImzMLParser(str(source_path))

    with ImzMLWriter(
        str(output_path),
        mode="processed",
        spec_type="profile",
    ) as writer:
        for spectrum_index in spectrum_indices:
            mass_axis, intensities = parser.getspectrum(spectrum_index)
            writer.addSpectrum(
                mass_axis,
                intensities,
                parser.coordinates[spectrum_index],
            )


def main() -> None:
    """Parse command-line arguments and generate the fixture."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_path", type=Path)
    parser.add_argument("output_path", type=Path)
    arguments = parser.parse_args()
    build_fixture(arguments.source_path, arguments.output_path)


if __name__ == "__main__":
    main()
