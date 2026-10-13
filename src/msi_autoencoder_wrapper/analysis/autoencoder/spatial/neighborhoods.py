"""Physical neighbours on source pixels with an explicit allowed population."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

from ....utils.logger import get_custom_logger

logger = get_custom_logger(__name__)


@dataclass
class PhysicalPixelIndex:
    """Store one source image's allowed pixels in physical coordinates.

    Source IDs are metadata indices, not positions in an arbitrary minibatch.
    Build O(N log N), storage O(N); each radius query is output-sensitive with
    worst-case O(N). No all-pairs distance matrix is materialized.

    :param pixel_ids: Unique allowed source IDs, shape ``(N,)``.
    :param xy: Source grid coordinates, shape ``(N, 2)``.
    :param spacing_um: Confirmed positive X/Y pixel spacing in micrometres.
    """

    pixel_ids: np.ndarray
    xy: np.ndarray
    spacing_um: tuple[float, float]

    def __post_init__(self) -> None:
        ids, xy, spacing = np.asarray(self.pixel_ids), np.asarray(self.xy, dtype=float), np.asarray(self.spacing_um, dtype=float)
        if ids.ndim != 1 or not len(ids) or ids.dtype.kind not in "iu" or np.any(ids < 0) or len(np.unique(ids)) != len(ids):
            raise ValueError("Allowed source pixel IDs must be nonempty, unique and nonnegative.")
        if xy.shape != (len(ids), 2) or not np.isfinite(xy).all() or spacing.shape != (2,) or not np.isfinite(spacing).all() or np.any(spacing <= 0):
            raise ValueError("Coordinates and confirmed physical spacing are required.")
        if len(np.unique(xy, axis=0)) != len(ids):
            raise ValueError("Distinct source pixels must have distinct coordinates within one section.")
        self.pixel_ids = ids.copy()  # (N,)
        self.xy = xy.copy()  # (N, 2)
        self._physical = xy * spacing  # (N, 2)
        self._tree = cKDTree(self._physical)
        self._rows = {int(i): j for j, i in enumerate(ids)}

    def neighbours(self, anchor_id: int, radius_um: float) -> tuple[np.ndarray, np.ndarray]:
        """Return allowed neighbours and physical distances sorted by source ID.

        Radius includes its boundary; self is excluded. No neighbour cap silently
        changes the physical selection. Only IDs supplied during construction can
        occur, so validation/test pixels are excluded by the caller's population.

        :param anchor_id: Source ID within the allowed population.
        :param radius_um: Finite nonnegative physical radius.
        :return: Neighbour source IDs and distances, both shape ``(K,)``.
        :raises ValueError: For invalid radius or unknown anchor.
        """
        if anchor_id not in self._rows or not np.isfinite(radius_um) or radius_um < 0:
            raise ValueError("Anchor and physical radius are invalid.")
        row = self._rows[anchor_id]
        rows = np.asarray(self._tree.query_ball_point(self._physical[row], radius_um), dtype=np.int64)  # (K+1,)
        rows = rows[rows != row]  # (K,)
        rows = rows[np.argsort(self.pixel_ids[rows])]  # (K,)
        distances = np.linalg.norm(self._physical[rows] - self._physical[row], axis=1)  # (K,)
        logger.debug("Source pixel %s has %s allowed neighbours within %s um.", anchor_id, len(rows), radius_um)
        return self.pixel_ids[rows], distances


def source_indices_from_merged_map(pixel_map, merged_ids: np.ndarray,
                                   source_coordinates: dict[str, np.ndarray],
                                   spacing_um: dict[str, tuple[float, float]]) -> tuple[dict[str, PhysicalPixelIndex], object]:
    """Compose the existing compact merge map with source image metadata.

    The caller supplies only permitted merged IDs (e.g. the persisted train split).
    One index is built per original image, so coordinates from different images
    cannot produce an edge. Source arrays retain original spectrum ordering.

    :param pixel_map: Existing MergedPixelMap instance.
    :param merged_ids: Allowed original merged spectrum IDs, shape ``(N,)``.
    :param source_coordinates: Source XY arrays keyed by dataset ID.
    :param spacing_um: Confirmed physical X/Y spacing keyed by dataset ID.
    :return: Per-source physical indices and merged-to-source mapping DataFrame.
    :raises ValueError: For absent source metadata or invalid source indices.
    """
    ids = np.asarray(merged_ids)
    if ids.ndim != 1 or ids.dtype.kind not in "iu" or not len(ids) or len(np.unique(ids)) != len(ids):
        raise ValueError("Allowed merged IDs must be a nonempty unique integer vector.")
    mapping = pixel_map.resolve(ids)
    indices = {}
    for dataset_id, rows in mapping.groupby("dataset_id", sort=False):
        if dataset_id not in source_coordinates or dataset_id not in spacing_um:
            raise ValueError(f"Source coordinates/confirmed spacing missing for {dataset_id}.")
        xy = np.asarray(source_coordinates[dataset_id])  # (P_source, 2)
        source_ids = rows.source_pixel.to_numpy(dtype=np.int64)  # (N_source_allowed,)
        if xy.ndim != 2 or xy.shape[1] != 2 or np.any(source_ids < 0) or np.any(source_ids >= len(xy)):
            raise ValueError("Merged map refers to an invalid source coordinate index.")
        indices[str(dataset_id)] = PhysicalPixelIndex(source_ids, xy[source_ids], spacing_um[dataset_id])
    return indices, mapping
