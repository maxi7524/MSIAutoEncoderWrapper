"""Lossless, identity-preserving anatomical annotations for one source image."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from matplotlib.path import Path as PolygonPath

from ....utils.logger import get_custom_logger

logger = get_custom_logger(__name__)
SCHEMA_ID = "msi_pixel_annotation"
UNKNOWN = -1


@dataclass(frozen=True)
class PixelMask:
    """Store disjoint labels keyed by original pixel identity.

    :param pixel_ids: Unique nonnegative integer source indices, shape ``(N,)``.
    :param labels: Integer class IDs, or -1 for unavailable labels, shape ``(N,)``.
    :param metadata: Dataset/image identity, coordinate space and class vocabulary.
    :raises ValueError: If identity, labels or metadata violate the schema.
    """

    pixel_ids: np.ndarray
    labels: np.ndarray
    metadata: Mapping[str, Any]

    def __post_init__(self) -> None:
        ids, labels = np.asarray(self.pixel_ids), np.asarray(self.labels)
        if ids.ndim != 1 or labels.shape != ids.shape or ids.dtype.kind not in "iu" or labels.dtype.kind not in "iu":
            raise ValueError("Pixel IDs and labels must be equal-length integer vectors.")
        if len(ids) == 0 or np.any(ids < 0) or len(np.unique(ids)) != len(ids):
            raise ValueError("Pixel IDs must be nonempty, unique and nonnegative.")
        required = {"dataset_id", "image_id", "coordinate_space", "class_names", "usage_role", "annotation_status", "provenance"}
        if not required.issubset(self.metadata):
            raise ValueError(f"Missing mask metadata: {sorted(required - self.metadata.keys())}.")
        if self.metadata["coordinate_space"] not in {"histology_image", "msi_source"}:
            raise ValueError("Coordinate space must be histology_image or msi_source.")
        classes = self.metadata["class_names"]
        if not classes or any(not str(k).isdigit() for k in classes) or len(set(classes.values())) != len(classes):
            raise ValueError("Class names must have nonnegative integer keys and unique names.")
        if not set(np.unique(labels)).issubset({UNKNOWN, *map(int, classes)}):
            raise ValueError("Mask contains undefined class IDs.")
        for field in ("dataset_id", "image_id", "usage_role", "annotation_status"):
            if not isinstance(self.metadata[field], str) or not self.metadata[field]:
                raise ValueError(f"{field} must be a nonempty string.")
        # Owned immutable arrays prevent stale hashes after a caller mutates inputs.
        ids, labels = ids.astype(np.int64, copy=True), labels.astype(np.int32, copy=True)  # (N,), (N,)
        ids.flags.writeable = labels.flags.writeable = False
        object.__setattr__(self, "pixel_ids", ids)
        object.__setattr__(self, "labels", labels)
        object.__setattr__(self, "metadata", json.loads(json.dumps(dict(self.metadata), allow_nan=False)))

    @property
    def available(self) -> np.ndarray:
        """Return availability derived exclusively from label IDs, shape ``(N,)``."""
        return self.labels != UNKNOWN

    def align_to(self, pixel_ids: np.ndarray) -> np.ndarray:
        """Join requested unique source IDs without relying on array row order.

        :param pixel_ids: Requested unique source IDs, shape ``(K,)``.
        :return: Labels in the requested order, shape ``(K,)``.
        :raises ValueError: If IDs are malformed, duplicated or absent.
        """
        requested = np.asarray(pixel_ids)
        if requested.ndim != 1 or requested.dtype.kind not in "iu" or len(np.unique(requested)) != len(requested):
            raise ValueError("Requested pixel IDs must be a unique integer vector.")
        order = np.argsort(self.pixel_ids)  # (N,)
        sorted_ids = self.pixel_ids[order]  # (N,)
        positions = np.searchsorted(sorted_ids, requested)  # (K,)
        valid = positions < len(sorted_ids)  # (K,)
        if not np.all(valid) or not np.array_equal(sorted_ids[positions], requested):
            raise ValueError("Requested pixel IDs are missing from this annotation.")
        return self.labels[order[positions]]  # (K,)


def save_mask(mask: PixelMask, directory: Path | str) -> Path:
    """Write a numeric NPZ and versioned JSON manifest with SHA-256.

    :param mask: Validated source annotation.
    :param directory: Destination directory; existing files are replaced.
    :return: Manifest path.
    :raises ValueError: If metadata cannot be serialized as strict JSON.
    """
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    numeric = root / "mask.npz"
    np.savez_compressed(numeric, pixel_ids=mask.pixel_ids, labels=mask.labels)
    manifest = dict(mask.metadata, schema_id=SCHEMA_ID, schema_version=1, numeric_file="mask.npz",
                    numeric_sha256=hashlib.sha256(numeric.read_bytes()).hexdigest(), unknown_id=UNKNOWN,
                    pixel_count=len(mask.pixel_ids))
    path = root / "annotation_manifest.json"
    path.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    logger.info("Saved %s annotated pixels of %s to %s.", int(mask.available.sum()), len(mask.labels), root)
    return path


def load_mask(directory: Path | str) -> PixelMask:
    """Load and verify the annotation schema, numeric hash and pixel count.

    :param directory: Directory created by :func:`save_mask`.
    :return: Annotation preserving source indices.
    :raises ValueError: For corrupt or unsupported artifacts.
    :raises FileNotFoundError: For missing artifacts.
    """
    root = Path(directory)
    meta = json.loads((root / "annotation_manifest.json").read_text())
    if meta.get("schema_id") != SCHEMA_ID or meta.get("schema_version") != 1 or meta.get("unknown_id") != UNKNOWN:
        raise ValueError("Unsupported pixel annotation schema.")
    if meta.get("numeric_file") != "mask.npz":
        raise ValueError("Numeric file must be mask.npz.")
    numeric = root / "mask.npz"
    if hashlib.sha256(numeric.read_bytes()).hexdigest() != meta.get("numeric_sha256"):
        raise ValueError("Annotation checksum mismatch.")
    with np.load(numeric, allow_pickle=False) as data:
        mask = PixelMask(data["pixel_ids"], data["labels"], meta)
    if len(mask.pixel_ids) != meta["pixel_count"]:
        raise ValueError("Annotation pixel count mismatch.")
    return mask


def rasterize_regions(shape: tuple[int, int], regions: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    """Rasterize explicitly traced polygons at pixel centres.

    Unannotated pixels and overlaps between different classes receive -1. Polygons
    of the same class may overlap. Complexity is O(H*W*V), for total polygon vertices
    V, with O(H*W) auxiliary storage; bounded boxes avoid a full array per polygon.

    :param shape: Image height and width.
    :param regions: Dictionaries with ``class_id`` and ``vertices`` in image X/Y.
    :return: Label raster and conflicting-overlap mask, both shape ``(H, W)``.
    :raises ValueError: For invalid dimensions, vertices or class IDs.
    """
    h, w = shape
    if h <= 0 or w <= 0:
        raise ValueError("Image dimensions must be positive.")
    labels = np.full((h, w), UNKNOWN, dtype=np.int32)  # (H, W)
    conflict = np.zeros((h, w), dtype=bool)  # (H, W)
    for region in regions:
        vertices = np.asarray(region["vertices"], dtype=np.float64)  # (V_i, 2)
        label = region["class_id"]
        if not isinstance(label, int) or label < 0 or vertices.ndim != 2 or vertices.shape[1] != 2 or len(vertices) < 3 or not np.isfinite(vertices).all():
            raise ValueError("Each region needs a nonnegative class ID and at least three finite XY vertices.")
        x0, y0 = np.maximum(np.floor(vertices.min(axis=0)).astype(int), 0)
        x1, y1 = np.minimum(np.ceil(vertices.max(axis=0)).astype(int), [w, h])
        if x1 <= x0 or y1 <= y0:
            continue
        yy, xx = np.mgrid[y0:y1, x0:x1]  # (h_i, w_i), (h_i, w_i)
        points = np.column_stack([xx.ravel() + 0.5, yy.ravel() + 0.5])  # (P_i, 2)
        inside = PolygonPath(vertices).contains_points(points).reshape(xx.shape)  # (h_i, w_i)
        current = labels[y0:y1, x0:x1]  # (h_i, w_i)
        overlap = inside & (current != UNKNOWN) & (current != label)  # (h_i, w_i)
        conflict[y0:y1, x0:x1] |= overlap
        current[inside] = label
    labels[conflict] = UNKNOWN
    return labels, conflict


def project_histology_mask(mask: PixelMask, xy: np.ndarray, pixel_ids: np.ndarray,
                            registration: Mapping[str, Any], image_id: str) -> PixelMask:
    """Sample H&E labels at MSI centres using an independently accepted transform.

    This initial adapter samples centres, not footprint area fractions. Use a
    separately reviewed uncertainty band for boundaries. No fit occurs here.

    :param mask: Row-major histology image annotation.
    :param xy: MSI source XY centres, shape ``(N, 2)``.
    :param pixel_ids: Original MSI source indices, shape ``(N,)``.
    :param registration: Accepted MSI-to-H&E affine 3×3 matrix and provenance.
    :param image_id: Target MSI image identity.
    :return: MSI-indexed mask; outside-image pixels are unavailable.
    :raises ValueError: For unaccepted registration or inconsistent coordinates.
    """
    if mask.metadata["coordinate_space"] != "histology_image" or registration.get("status") != "accepted":
        raise ValueError("An independently accepted histology registration is required.")
    matrix = np.asarray(registration["msi_to_histology"], dtype=float)  # (3, 3)
    coords = np.asarray(xy, dtype=float)  # (N, 2)
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all() or not np.allclose(matrix[2], [0, 0, 1]) or abs(np.linalg.det(matrix)) < 1e-12:
        raise ValueError("Registration must be a finite invertible affine transform.")
    if coords.ndim != 2 or coords.shape != (len(pixel_ids), 2) or not np.isfinite(coords).all():
        raise ValueError("MSI coordinates must be finite N by 2.")
    h, w = mask.metadata["image_shape"]
    if not np.array_equal(mask.pixel_ids, np.arange(h * w)):
        raise ValueError("Histology mask must use complete row-major image indices.")
    homogeneous = np.column_stack([coords, np.ones(len(coords))])  # (N, 3)
    projected = homogeneous @ matrix.T  # (N, 3)
    inside = (projected[:, 0] >= 0) & (projected[:, 0] < w) & (projected[:, 1] >= 0) & (projected[:, 1] < h)  # (N,)
    labels = np.full(len(coords), UNKNOWN, dtype=np.int32)  # (N,)
    px = np.floor(projected[inside, :2]).astype(np.int64)  # (K, 2)
    labels[inside] = mask.labels[px[:, 1] * w + px[:, 0]]  # (K,)
    meta = dict(mask.metadata, image_id=image_id, coordinate_space="msi_source", registration=dict(registration))
    meta.pop("image_shape", None)
    return PixelMask(np.asarray(pixel_ids), labels, meta)
