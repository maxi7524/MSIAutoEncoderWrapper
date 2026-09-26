"""Map merged-store spectrum identifiers to source images and pixel coordinates."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ....utils.logger import get_custom_logger

logger = get_custom_logger(__name__)


@dataclass(frozen=True)
class MergedPixelMap:
    """Vectorized view of the ``pixel_segments`` table of a merged annotation store.

    A merged spectrum ``s`` inside segment ``g`` originates from source spectrum
    ``source_pixel_start[g] + (s - merged_pixel_start[g]) * source_step[g]`` of dataset
    ``dataset_index[g]``. This is the relation implemented per spectrum by
    ``MergedAnnotationReader.get_spectrum_metadata``; it is evaluated here for many
    spectra at once.

    :param segments: One row per segment with the schema columns of ``pixel_segments``.
    :type segments: pandas.DataFrame
    :param datasets: One row per dataset with ``dataset_index``, ``dataset_id``,
        ``name`` and ``image_path``.
    :type datasets: pandas.DataFrame
    """

    segments: pd.DataFrame
    datasets: pd.DataFrame

    @classmethod
    def from_store(cls, sqlite_path: Path | str) -> "MergedPixelMap":
        """Read the segment and dataset tables of a merged store (read-only).

        :param sqlite_path: Merged annotation store, e.g. ``datasets/kidney/kidney.sqlite``.
        :type sqlite_path: pathlib.Path | str
        :return: Segment map ordered by merged start index.
        :rtype: MergedPixelMap
        :raises FileNotFoundError: If the store does not exist.
        """
        path = Path(sqlite_path)
        if not path.is_file():
            raise FileNotFoundError(f"Merged annotation store not found: {path}")
        with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as connection:
            segments = pd.read_sql(
                "SELECT dataset_index, merged_pixel_start, segment_length, source_pixel_start, source_step "
                "FROM pixel_segments ORDER BY merged_pixel_start", connection)
            datasets = pd.read_sql(
                "SELECT dataset_index, source_dataset_id AS dataset_id, name, source_imzml_path AS image_path "
                "FROM datasets_metadata", connection)
        logger.debug("Merged pixel map: %s segments, %s datasets.", len(segments), len(datasets))
        return cls(segments.reset_index(drop=True), datasets)

    def dataset_ranges(self, dataset_ids: list[str]) -> list[tuple[int, int]]:
        """Return the half-open merged ranges belonging to the given datasets.

        :param dataset_ids: Source dataset identifiers.
        :type dataset_ids: list[str]
        :return: Sorted ``(start, stop)`` merged-spectrum intervals.
        :rtype: list[tuple[int, int]]
        :raises ValueError: If an identifier is not present in the store.
        """
        known = set(self.datasets.dataset_id)
        missing = sorted(set(dataset_ids) - known)
        if missing:
            raise ValueError(f"Datasets absent from the merged store: {missing}.")
        indices = self.datasets.loc[self.datasets.dataset_id.isin(dataset_ids), "dataset_index"]
        rows = self.segments[self.segments.dataset_index.isin(indices)]
        return [(int(start), int(start + length))
                for start, length in zip(rows.merged_pixel_start, rows.segment_length)]

    def resolve(self, merged_ids: np.ndarray) -> pd.DataFrame:
        """Resolve merged spectrum identifiers to their source dataset and spectrum.

        :param merged_ids: Merged spectrum identifiers, shape ``(N,)``.
        :type merged_ids: numpy.ndarray
        :return: ``source_id``, ``dataset_id``, ``dataset_name`` and ``source_pixel``
            in the input order.
        :rtype: pandas.DataFrame
        :raises ValueError: If an identifier lies outside every segment.
        """
        ids = np.asarray(merged_ids, dtype=np.int64)
        starts = self.segments.merged_pixel_start.to_numpy(np.int64)
        position = np.searchsorted(starts, ids, side="right") - 1  # (N,)
        lengths = self.segments.segment_length.to_numpy(np.int64)
        offset = ids - starts[np.clip(position, 0, None)]
        if np.any(position < 0) or np.any(offset >= lengths[np.clip(position, 0, None)]):
            raise ValueError("Some merged spectrum identifiers lie outside every pixel segment.")
        segment = self.segments.iloc[position]
        source_pixel = segment.source_pixel_start.to_numpy(np.int64) + offset * segment.source_step.to_numpy(np.int64)
        names = self.datasets.set_index("dataset_index")
        return pd.DataFrame({
            "source_id": ids,
            "dataset_id": names.dataset_id.reindex(segment.dataset_index).to_numpy(),
            "dataset_name": names.name.reindex(segment.dataset_index).to_numpy(),
            "source_pixel": source_pixel,
        })


def imzml_coordinates(image_path: Path | str) -> np.ndarray:
    """Return the ``(x, y)`` pixel coordinates stored in an imzML file.

    :param image_path: Path to the ``.imzML`` file.
    :type image_path: pathlib.Path | str
    :return: Integer coordinates, shape ``(P, 2)``, in spectrum order and in the file's
        own (normally 1-based) convention.
    :rtype: numpy.ndarray
    """
    from pyimzml.ImzMLParser import ImzMLParser

    parser = ImzMLParser(str(image_path))
    coordinates = np.asarray(parser.coordinates, dtype=np.int64)[:, :2]  # (P, 2)
    logger.debug("Read %s coordinates from %s.", len(coordinates), image_path)
    return coordinates
