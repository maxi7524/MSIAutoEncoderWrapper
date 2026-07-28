"""Sequentially merge selected source spectra into one rectangular imzML image."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from pyimzml.ImzMLWriter import ImzMLWriter

from ...readers.strategies.pyimzml_reader import PyImzMLReader
from ...utils.exceptions import raise_validation_error, raise_workspace_error
from ...utils.logger import get_custom_logger
from ..catalog.sqlite_catalog import DatasetCatalog


logger = get_custom_logger(__name__)


@dataclass(frozen=True)
class ImzMLMergeInput:
    """Describe one source dataset participating in an imzML merge.

    :param source: Data-source adapter key.
    :type source: str
    :param dataset_id: Stable source dataset identifier.
    :type dataset_id: str
    :param imzml_path: Local source imzML path.
    :type imzml_path: pathlib.Path
    :param spectrum_ids: Optional source spectrum indices to retain. ``None``
        retains every spectrum.
    :type spectrum_ids: Sequence[int] | None
    """

    source: str
    dataset_id: str
    imzml_path: Path
    spectrum_ids: Optional[Sequence[int]] = None


class ImzMLMerger:
    """Create a rectangular imzML and persist minimal index provenance."""

    def __init__(self, catalog: DatasetCatalog) -> None:
        self.catalog = catalog

    def merge(
        self,
        *,
        inputs: Sequence[ImzMLMergeInput],
        output_path: Path | str,
        merged_dataset_id: str,
        row_width: Optional[int] = None,
    ) -> Path:
        """Write selected spectra sequentially and register index mappings.

        :param inputs: Ordered source datasets and selected source spatial IDs.
        :type inputs: Sequence[ImzMLMergeInput]
        :param output_path: Destination imzML path. Existing output is rejected.
        :type output_path: pathlib.Path | str
        :param merged_dataset_id: Catalog identifier for the produced dataset.
        :type merged_dataset_id: str
        :param row_width: Optional output image width. The default is the
            smallest near-square width that contains all spectra.
        :type row_width: int | None
        :return: Written imzML path.
        :rtype: pathlib.Path
        :raises ValidationError: If inputs or selected indices are invalid.
        :raises WorkspaceConfigError: If output creation fails.
        """
        if not inputs:
            raise_validation_error("ImzMLMerge", "At least one input dataset is required.")
        output = Path(output_path).with_suffix(".imzML")
        if output.exists() or output.with_suffix(".ibd").exists():
            raise_workspace_error(
                "ImzMLMerge",
                f"Output pair already exists for '{output}'.",
            )
        output.parent.mkdir(parents=True, exist_ok=True)

        prepared = self._prepare_inputs(inputs)
        spectrum_count = sum(len(spectrum_ids) for _, spectrum_ids in prepared)
        if spectrum_count == 0:
            raise_validation_error("ImzMLMerge", "The selected inputs contain no spectra.")
        width = row_width or int(math.ceil(math.sqrt(spectrum_count)))
        if width <= 0:
            raise_validation_error("ImzMLMerge", "row_width must be greater than zero.")

        temporary = output.with_name(f"{output.stem}.partial.imzML")
        temporary_ibd = temporary.with_suffix(".ibd")
        temporary.unlink(missing_ok=True)
        temporary_ibd.unlink(missing_ok=True)
        mappings: List[Dict[str, object]] = []

        try:
            with ImzMLWriter(str(temporary), mode="processed") as writer:
                merged_index = 0
                for merge_input, (reader, spectrum_ids) in zip(inputs, prepared):
                    for source_spectrum_id in spectrum_ids:
                        mass_axis, intensities = reader.GetSpectrum(source_spectrum_id)
                        x_position = merged_index % width + 1
                        y_position = merged_index // width + 1
                        writer.addSpectrum(
                            mass_axis,
                            intensities,
                            (x_position, y_position, 1),
                            userParams=[
                                {"name": "source", "value": merge_input.source},
                                {"name": "source_dataset_id", "value": merge_input.dataset_id},
                                {"name": "source_spectrum_id", "value": str(source_spectrum_id)},
                            ],
                        )
                        mappings.append(
                            {
                                "source": merge_input.source,
                                "source_dataset_id": merge_input.dataset_id,
                                "source_spectrum_id": source_spectrum_id,
                                "merged_spectrum_index": merged_index,
                            }
                        )
                        merged_index += 1
            temporary.replace(output)
            temporary_ibd.replace(output.with_suffix(".ibd"))
        except Exception as error:
            temporary.unlink(missing_ok=True)
            temporary_ibd.unlink(missing_ok=True)
            raise_workspace_error(
                "ImzMLMerge",
                f"Failed to create merged dataset '{output}': {error}",
            )

        self.catalog.register_merged_dataset(merged_dataset_id, output)
        self.catalog.replace_spectrum_mappings(merged_dataset_id, mappings)
        logger.info(
            "Merged %s spectra from %s datasets into %s",
            spectrum_count,
            len(inputs),
            output,
        )
        return output

    @staticmethod
    def _prepare_inputs(
        inputs: Sequence[ImzMLMergeInput],
    ) -> List[tuple[PyImzMLReader, List[int]]]:
        prepared: List[tuple[PyImzMLReader, List[int]]] = []
        for merge_input in inputs:
            if not merge_input.imzml_path.is_file():
                raise_validation_error(
                    "ImzMLMerge",
                    f"Input imzML does not exist: {merge_input.imzml_path}",
                )
            reader = PyImzMLReader(merge_input.imzml_path)
            count = reader.GetNumberOfSpectra()
            spectrum_ids = (
                list(range(count))
                if merge_input.spectrum_ids is None
                else [int(value) for value in merge_input.spectrum_ids]
            )
            invalid = [value for value in spectrum_ids if value < 0 or value >= count]
            if invalid:
                raise_validation_error(
                    "ImzMLMerge",
                    f"Dataset '{merge_input.dataset_id}' has invalid spectrum IDs: {invalid}",
                )
            prepared.append((reader, spectrum_ids))
        return prepared
