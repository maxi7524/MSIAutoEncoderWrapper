"""Decode-once cache of METASPACE ion images exported as ``pixel_intensities.csv``."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..autoencoder.experiments.pretraining_campaign import read_metaspace_images
from ...utils.logger import get_custom_logger

logger = get_custom_logger(__name__)


@dataclass(frozen=True)
class IonImages:
    """Ion images of one dataset on its METASPACE pixel grid.

    :param dataset_id: METASPACE dataset identifier.
    :type dataset_id: str
    :param ion_ids: ``formula|adduct`` identities, shape ``(A,)``.
    :type ion_ids: numpy.ndarray
    :param coordinates: METASPACE pixel coordinates ``(x, y)``, shape ``(P, 2)``.
    :type coordinates: numpy.ndarray
    :param intensities: Non-negative ion intensities, shape ``(A, P)``.
    :type intensities: numpy.ndarray
    """

    dataset_id: str
    ion_ids: np.ndarray
    coordinates: np.ndarray
    intensities: np.ndarray

    def tic_normalized(self) -> tuple[np.ndarray, np.ndarray]:
        """Return intensities divided by the per-pixel sum over annotated ions.

        :return: Normalized intensities ``(A, P_+)`` for pixels with positive
            annotated signal, and the boolean pixel mask ``(P,)`` selecting them.
        :rtype: tuple[numpy.ndarray, numpy.ndarray]
        """
        total = self.intensities.sum(axis=0)  # (P,)
        mask = total > 0
        return self.intensities[:, mask] / total[mask], mask  # (A, P_+), (P,)

    def grid(self, values: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
        """Place per-pixel values on the rectangular image grid.

        :param values: Values for all pixels ``(P,)`` or for ``mask`` pixels.
        :type values: numpy.ndarray
        :param mask: Optional pixel mask ``(P,)`` that produced ``values``.
        :type mask: numpy.ndarray | None
        :return: Image ``(H, W)`` with ``nan`` outside the measured pixels.
        :rtype: numpy.ndarray
        """
        coordinates = self.coordinates if mask is None else self.coordinates[mask]
        image = np.full((self.coordinates[:, 1].max() + 1, self.coordinates[:, 0].max() + 1),
                        np.nan, dtype=np.float32)  # (H, W)
        image[coordinates[:, 1], coordinates[:, 0]] = values
        return image


def cache_ion_images(dataset_directory: Path | str, cache_directory: Path | str) -> Path:
    """Convert one ``pixel_intensities.csv`` to a compressed ``.npz`` once.

    Duplicate rows of the same ion (one per annotation database) carry the same ion
    image; the first row is retained.

    :param dataset_directory: METASPACE dataset directory.
    :type dataset_directory: pathlib.Path | str
    :param cache_directory: Directory receiving ``<dataset_id>.npz``.
    :type cache_directory: pathlib.Path | str
    :return: Cache file path.
    :rtype: pathlib.Path
    """
    source = Path(dataset_directory)
    target = Path(cache_directory) / f"{source.name}.npz"
    if target.is_file():
        return target
    annotations, coordinates, intensities = read_metaspace_images(source)  # (P, 2), (A, P)
    keep = ~annotations.class_name.duplicated().to_numpy()
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(".partial.npz")
    np.savez_compressed(temporary, ion_ids=annotations.class_name.to_numpy()[keep].astype(str),
                        coordinates=coordinates.astype(np.int32),
                        intensities=np.nan_to_num(intensities[keep], nan=0.0).astype(np.float32))
    temporary.replace(target)
    logger.info("Cached %s ion images of dataset %s (%s pixels).", int(keep.sum()), source.name, coordinates.shape[0])
    return target


def load_ion_images(path: Path | str) -> IonImages:
    """Load one cache written by :func:`cache_ion_images`.

    :param path: ``<dataset_id>.npz`` cache file.
    :type path: pathlib.Path | str
    :return: Ion images of the dataset.
    :rtype: IonImages
    """
    with np.load(path, allow_pickle=False) as payload:
        return IonImages(Path(path).stem, payload["ion_ids"], payload["coordinates"], payload["intensities"])
