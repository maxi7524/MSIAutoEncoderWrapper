"""Sequentially merge selected source spectra into one rectangular imzML image."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

from pyimzml.ImzMLWriter import ImzMLWriter
from tqdm.auto import tqdm

from ...imzml import PyImzMLReader
from ...utils.exceptions import raise_validation_error, raise_workspace_error
from ...utils.logger import get_custom_logger
from .selection import select_merge_spectrum_ids


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
    :param spectrum_ids: Optional explicit source spectrum indices to retain.
        ``None`` applies annotation-aware selection.
    :type spectrum_ids: Sequence[int] | None
    """

    source: str
    dataset_id: str
    imzml_path: Path
    spectrum_ids: Optional[Sequence[int]] = None
    annotation_records: Sequence[Mapping[str, Any]] = ()


@dataclass(frozen=True)
class ImzMLMergeResult:
    """Return the composed pair path and exact source-pixel mappings."""

    path: Path
    pixel_mappings: List[Dict[str, object]]


class ImzMLMerger:
    """Create a rectangular imzML and return its source-pixel provenance."""

    def merge(
        self,
        *,
        inputs: Sequence[ImzMLMergeInput],
        output_path: Path | str,
        row_width: Optional[int] = None,
        unannotated_ratio: Optional[float] = None,
        unannotated_amount: Optional[int] = None,
        max_fdr: Optional[float] = None,
        random_seed: int = 0,
    ) -> ImzMLMergeResult:
        """Write selected spectra sequentially and return index mappings.

        :param inputs: Ordered source datasets and selected source spatial IDs.
        :type inputs: Sequence[ImzMLMergeInput]
        :param output_path: Destination imzML path. Existing output is rejected.
        :type output_path: pathlib.Path | str
        :param row_width: Optional output image width. The default is the
            smallest near-square width that contains all spectra.
        :type row_width: int | None
        :param unannotated_ratio: Optional number of unannotated spectra per
            annotated spectrum.
        :type unannotated_ratio: float | None
        :param unannotated_amount: Optional absolute unannotated spectrum count.
            ``None`` retains all available unannotated spectra and ``0`` retains
            none when no ratio is supplied.
        :type unannotated_amount: int | None
        :param random_seed: Reproducible unannotated sampling seed.
        :type random_seed: int
        :return: Written path and exact source-to-merged pixel mappings.
        :rtype: ImzMLMergeResult
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

        prepared = self._prepare_inputs(
            inputs,
            unannotated_ratio=unannotated_ratio,
            unannotated_amount=unannotated_amount,
            max_fdr=max_fdr,
            random_seed=random_seed,
        )
        spectrum_count = sum(len(spectrum_ids) for _, spectrum_ids in prepared)
        if spectrum_count == 0:
            raise_validation_error("ImzMLMerge", "The selected inputs contain no spectra.")
        width = row_width or int(math.ceil(math.sqrt(spectrum_count)))
        if width <= 0:
            raise_validation_error("ImzMLMerge", "row_width must be greater than zero.")

        # Merge progress estimation
        ## imzML writing does not expose a byte callback. Estimate source bytes
        ## read per selected spectrum, while the postfix reports completed source
        ## images. The final output size can differ due to imzML serialization.
        estimated_bytes = _estimate_selected_input_bytes(inputs, prepared)
        progress = tqdm(
            total=estimated_bytes,
            desc="Merge source data",
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            dynamic_ncols=True,
        )

        temporary = output.with_name(f"{output.stem}.partial.imzML")
        temporary_ibd = temporary.with_suffix(".ibd")
        temporary.unlink(missing_ok=True)
        temporary_ibd.unlink(missing_ok=True)
        pixel_mappings: List[Dict[str, object]] = []

        try:
            with ImzMLWriter(str(temporary), mode="processed") as writer:
                merged_index = 0
                for input_index, (merge_input, (reader, spectrum_ids)) in enumerate(
                    zip(inputs, prepared),
                    start=1,
                ):
                    source_bytes = _source_pair_size(merge_input.imzml_path)
                    source_spectrum_count = reader.GetNumberOfSpectra()
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
                        pixel_mappings.append(
                            {
                                "source": merge_input.source,
                                "source_dataset_id": merge_input.dataset_id,
                                "source_spectrum_id": source_spectrum_id,
                                "merged_spectrum_index": merged_index,
                            }
                        )
                        merged_index += 1
                        progress.update(max(1, source_bytes // source_spectrum_count))
                    progress.set_postfix(
                        images=f"{input_index}/{len(inputs)}",
                        spectra=merged_index,
                    )
            temporary.replace(output)
            temporary_ibd.replace(output.with_suffix(".ibd"))
        except Exception as error:
            temporary.unlink(missing_ok=True)
            temporary_ibd.unlink(missing_ok=True)
            raise_workspace_error(
                "ImzMLMerge",
                f"Failed to create merged dataset '{output}': {error}",
            )
        finally:
            progress.close()

        logger.info(
            "Merged %s spectra from %s datasets into %s",
            spectrum_count,
            len(inputs),
            output,
        )
        return ImzMLMergeResult(path=output, pixel_mappings=pixel_mappings)

    def _prepare_inputs(
        self,
        inputs: Sequence[ImzMLMergeInput],
        *,
        unannotated_ratio: Optional[float],
        unannotated_amount: Optional[int],
        max_fdr: Optional[float],
        random_seed: int,
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
            explicit_selection = merge_input.spectrum_ids is not None
            candidate_ids = (
                list(range(count))
                if not explicit_selection
                else [int(value) for value in merge_input.spectrum_ids or ()]
            )
            invalid = [value for value in candidate_ids if value < 0 or value >= count]
            if invalid:
                raise_validation_error(
                    "ImzMLMerge",
                    f"Dataset '{merge_input.dataset_id}' has invalid spectrum IDs: {invalid}",
                )
            spectrum_ids = (
                candidate_ids
                if explicit_selection
                else select_merge_spectrum_ids(
                    candidate_ids=candidate_ids,
                    annotated_ids=_annotated_spectrum_ids(
                        merge_input.annotation_records,
                        max_fdr=max_fdr,
                    ),
                    unannotated_ratio=unannotated_ratio,
                    unannotated_amount=unannotated_amount,
                    random_seed=random_seed,
                    seed_namespace=f"{merge_input.source}:{merge_input.dataset_id}",
                )
            )
            prepared.append((reader, spectrum_ids))
        return prepared


def _source_pair_size(imzml_path: Path) -> int:
    """Return the byte size of one source imzML/ibd pair."""
    ibd_path = imzml_path.with_suffix(".ibd")
    return imzml_path.stat().st_size + ibd_path.stat().st_size


def _estimate_selected_input_bytes(
    inputs: Sequence[ImzMLMergeInput],
    prepared: Sequence[tuple[PyImzMLReader, Sequence[int]]],
) -> int:
    """Estimate bytes read for the selected spectra of every source image."""
    return sum(
        max(
            1,
            _source_pair_size(merge_input.imzml_path)
            * len(spectrum_ids)
            // reader.GetNumberOfSpectra(),
        )
        for merge_input, (reader, spectrum_ids) in zip(inputs, prepared)
    )


def _annotated_spectrum_ids(
    records: Sequence[Mapping[str, Any]],
    *,
    max_fdr: Optional[float],
) -> set[int]:
    """Return source spectrum indices supported by an accepted annotation."""
    selected: set[int] = set()
    for record in records:
        fdr = record.get("fdr")
        if max_fdr is not None and (
            fdr is None or float(fdr) > float(max_fdr)
        ):
            continue
        selected.update(int(value) for value in record.get("spectrum_ids", ()))
    return selected
