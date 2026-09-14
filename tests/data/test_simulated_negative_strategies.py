"""Signal-evidence reliable-negative strategy integration tests."""

from __future__ import annotations

import numpy as np
from pathlib import Path
import torch

from msi_dataset_manager.annotations.index import build_annotation_index

from msi_autoencoder_wrapper.data.simulated_negatives import SimulatedNegativeManager
from msi_autoencoder_wrapper.data.supervision_masks import simulated_negative_mask_key
from msi_autoencoder_wrapper.data.supervision_sampling import collect_supervision_masks
from msi_autoencoder_wrapper.models.datasets.strategies.pixel_dataset import PixelDataset


def _evidence_dataset() -> PixelDataset:
    """Return two pixels with positive, low-evidence, and U ion states."""
    class Reader:
        def GetNumberOfSpectra(self):
            return 2

        def GetSpectrum(self, spectrum_id):
            values = (
                np.array([1.0, 1.0], dtype=np.float32)
                if spectrum_id == 0
                else np.array([0.01, 1.0], dtype=np.float32)
            )
            return np.array([1.0, 2.0], dtype=np.float32), values

    class AnnotationReader:
        def get_dataset_metadata(self):
            return {"metadata": {}}

        def get_spectrum_metadata(self, spectrum_id):
            return self.get_dataset_metadata()

        def get_annotations(self):
            return [
                {"formula": "A", "adduct": "+H"},
                {"formula": "B", "adduct": "+H"},
            ]

        def get_spectrum_annotations(self, spectrum_id):
            return self.get_annotations() if spectrum_id == 0 else []

        def get_spectrum_annotation_index(self, spectrum_ids):
            indices = [0] if spectrum_ids is None else spectrum_ids
            entries = {0: [(("A", "+H"), 1.0), (("B", "+H"), 2.0)]}
            return build_annotation_index(
                spectrum_ids=indices,
                entries={index: entries[index] for index in indices if index in entries},
            )

    class Context:
        annotation_reader = AnnotationReader()
        binner = type(
            "Binner",
            (),
            {
                "transform_spectrum": staticmethod(
                    lambda mass_values, intensities: torch.as_tensor(intensities)
                ),
                "map_mass_values_to_bins": staticmethod(
                    lambda values: np.asarray(values, dtype=np.int32) - 1
                ),
                "GetXAxis": staticmethod(lambda: np.array([1.0, 2.0], dtype=np.float32)),
            },
        )()

        @staticmethod
        def get_data_reader(source):
            return Reader()

    return PixelDataset(
        active_context=Context(),
        normalization="tic",
        target_specs={"molecule": {"type": "multi_label"}},
        annotation_settings={
            "targets": {
                "molecule": {
                    "unobserved_label_policy": "unlabelled",
                    "simulated_negative": {
                        "type": "SignalEvidenceSimulatedNegative",
                        "parameters": {
                            "relative_threshold": 0.0119,
                            "bin_radius": 0,
                            "batch_size": 2,
                        },
                    },
                }
            }
        },
    )


def test_signal_evidence_strategy_is_loaded_from_portable_config() -> None:
    """The data-level registry resolves the declared evidence provider."""
    strategy = SimulatedNegativeManager.load_config(
        {
            "type": "SignalEvidenceSimulatedNegative",
            "parameters": {"relative_threshold": 0.0119},
        }
    )

    assert type(strategy).__name__ == "SignalEvidenceSimulatedNegative"


def test_pixel_dataset_emits_evidence_derived_simulated_negative_mask() -> None:
    """Low evidence becomes N_sim, high evidence remains U, and P is preserved."""
    dataset = _evidence_dataset()
    batch = dataset.get_target_batch([0, 1])
    simulated_negative = batch.masks[simulated_negative_mask_key("molecule")]

    assert batch.values["molecule"].tolist() == [[1.0, 1.0], [0.0, 0.0]]
    assert simulated_negative.tolist() == [[False, False], [True, False]]
    assert not bool(simulated_negative[batch.values["molecule"] > 0.5].any())


def test_evidence_masks_feed_the_pnu_sampler_without_reinterpreting_u() -> None:
    """The sampler sees P, N_sim, and U as distinct source memberships."""
    dataset = _evidence_dataset()
    targets, availability, simulated_negative = collect_supervision_masks(dataset, "molecule")

    assert targets.shape == availability.shape == simulated_negative.shape == (2, 2)
    assert simulated_negative.tolist() == [[False, False], [True, False]]
    assert bool((availability & ~simulated_negative).any())


def test_evidence_mask_cache_is_reused_for_an_identical_dataset_contract(
    tmp_path: Path,
) -> None:
    """Evidence masks are persisted once and reused by a fresh dataset instance."""
    wrapper = type("Wrapper", (), {"_project_path": str(tmp_path)})()
    first_dataset = _evidence_dataset()
    first_dataset.active_context._wrapper = wrapper
    first_mask = first_dataset.get_target_batch([0, 1]).masks[
        simulated_negative_mask_key("molecule")
    ]
    cache_files = list((tmp_path / "cache/evidence/simulated_negatives").glob("*.npz"))

    second_dataset = _evidence_dataset()
    second_dataset.active_context._wrapper = wrapper
    second_mask = second_dataset.get_target_batch([0, 1]).masks[
        simulated_negative_mask_key("molecule")
    ]

    assert len(cache_files) == 1
    assert list((tmp_path / "cache/evidence/simulated_negatives").glob("*.npz")) == cache_files
    assert torch.equal(first_mask, second_mask)
