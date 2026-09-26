"""Spatial analyses of pixel-level model outputs: images, ion images and segmentation."""

from .coordinates import MergedPixelMap, imzml_coordinates
from .images import assemble_image, image_extent
from .ion_images import ion_bin_indices, ion_intensities, presence_auc, spatial_agreement
from .segmentation import (
    SegmentationFit,
    adjusted_rand_between,
    align_labels,
    fit_segmentation,
    segment_contingency,
    segment_marker_table,
)

__all__ = [
    "MergedPixelMap",
    "SegmentationFit",
    "adjusted_rand_between",
    "align_labels",
    "assemble_image",
    "fit_segmentation",
    "image_extent",
    "imzml_coordinates",
    "ion_bin_indices",
    "ion_intensities",
    "presence_auc",
    "segment_contingency",
    "segment_marker_table",
    "spatial_agreement",
]
