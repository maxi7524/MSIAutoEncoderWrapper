"""Tests for stable and serializable pixel-spectrum normalization."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from msi_autoencoder_wrapper.models.datasets.strategies.pixel_dataset import (
    PixelDataset,
)
from msi_autoencoder_wrapper.utils.exceptions import ValidationError


def test_image_and_latent_sources_have_safe_normalization_defaults() -> None:
    """Images default to TIC scaling while latent components remain unchanged."""
    image_dataset = PixelDataset(source="image")
    latent_dataset = PixelDataset(source="latent")

    normalized = image_dataset._normalize(np.array([1.0, 2.0, 1.0], dtype=np.float32))

    assert normalized.sum() == pytest.approx(1.0)
    assert image_dataset.get_config()["normalization"] == "tic"
    assert latent_dataset.get_config()["normalization"] == "none"


def test_invalid_pixel_normalization_uses_global_validation_error() -> None:
    """Unsupported normalization names use the shared error format."""
    with pytest.raises(ValidationError, match="normalization"):
        PixelDataset(normalization="unsupported")


def test_pixel_dataset_assigns_metadata_and_multilabel_molecule_targets() -> None:
    """Semantic annotations become deterministic classes only in PixelDataset."""
    class Reader:
        def GetNumberOfSpectra(self):
            return 1

        def GetSpectrum(self, spectrum_id):
            return np.array([1.0]), np.array([2.0], dtype=np.float32)

    class AnnotationReader:
        def get_dataset_metadata(self):
            return {"metadata": {"condition": "disease"}}

        def get_spectrum_metadata(self, spectrum_id):
            return self.get_dataset_metadata()

        def get_annotations(self):
            return [
                {"formula": "B", "adduct": "+H"},
                {"formula": "A", "adduct": "+H"},
            ]

        def get_spectrum_annotations(self, spectrum_id):
            return [{"formula": "B", "adduct": "+H"}]

    class Context:
        annotation_reader = AnnotationReader()

        class Binner:
            @staticmethod
            def transform_spectrum(xs, ys):
                return torch.as_tensor(ys)

        binner = Binner()

        @staticmethod
        def get_data_reader(source):
            return Reader()

    dataset = PixelDataset(
        active_context=Context(),
        normalization="none",
        target_specs={
            "condition": {"type": "single_label"},
            "molecule": {"type": "multi_label"},
        },
    )

    spectrum_id, spectrum, targets, masks = dataset[0]

    assert spectrum_id == 0
    assert torch.equal(spectrum, torch.tensor([2.0]))
    assert dataset.get_class_mappings() == {
        "condition": {"disease": 0},
        "molecule": {"A|+H": 0, "B|+H": 1},
    }
    assert targets["condition"].item() == 0
    assert torch.equal(targets["molecule"], torch.tensor([0.0, 1.0]))
    assert masks["condition"].item()
    assert masks["molecule"].item()


def test_pixel_dataset_masks_missing_metadata_targets() -> None:
    """Missing metadata remains distinguishable from an explicit class zero."""
    class AnnotationReader:
        def get_dataset_metadata(self):
            return {"metadata": {}}

        def get_spectrum_metadata(self, spectrum_id):
            return {"metadata": {}}

        def get_annotations(self):
            return []

    class Context:
        annotation_reader = AnnotationReader()

    dataset = PixelDataset(
        active_context=Context(),
        target_specs={
            "condition": {
                "type": "single_label",
                "class_mapping": {"healthy": 0, "disease": 1},
            }
        },
    )

    _, _, targets, masks = dataset._sample(0, torch.ones(3))

    assert targets["condition"].item() == 0
    assert not masks["condition"].item()


def test_pixel_dataset_rejects_sparse_explicit_class_indices() -> None:
    """Explicit mappings remain safe for direct tensor indexing."""
    with pytest.raises(ValidationError, match="contiguous from zero"):
        PixelDataset(
            target_specs={
                "condition": {
                    "type": "single_label",
                    "class_mapping": {"healthy": 0, "disease": 2},
                }
            }
        )
