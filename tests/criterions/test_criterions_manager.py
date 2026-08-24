"""Tests for criterion discovery, creation, and composition."""

from __future__ import annotations

import pytest
import torch

from msi_autoencoder_wrapper.data import SpectrumBatch, SpectrumSpace
from msi_autoencoder_wrapper.training.criterions.criterions_manager import (
    CriterionsManager,
)
from msi_autoencoder_wrapper.training.criterions.autoencoder.contrastive.infoNCE_loss import (
    MSIInfoNCELoss,
    ProtectedIntervalIndex,
)
from msi_autoencoder_wrapper.training.criterions.autoencoder.regularization.contractive_loss import (
    MSIContractiveLoss,
)
from msi_autoencoder_wrapper.utils.exceptions import IncompatibleInterfaceError


def test_criterion_discovery_returns_uniform_component_information() -> None:
    """Criterion discovery populates the model-scoped registry and metadata shape."""
    CriterionsManager.discover_criterions()
    available = CriterionsManager.get_available_criterions("autoencoder")

    assert set(available) == {
        "reconstruction",
        "contrastive",
        "head",
        "regularization",
    }
    assert "MSELoss" in available["reconstruction"]
    assert "SobolevLoss" in available["reconstruction"]
    assert "InfoNCELoss" in available["contrastive"]
    assert "ContractiveLoss" in available["regularization"]
    assert "MultiLabelBCELoss" in available["head"]
    assert "ClassBalancedMultiLabelBCELoss" in available["head"]
    assert "NNPUMultiLabelLoss" in available["head"]
    assert "MaskedCrossEntropyLoss" in available["head"]
    assert set(available["reconstruction"]["MSELoss"]) == {
        "docstring",
        "parameters",
    }
    assert available["reconstruction"]["MSELoss"]["parameters"]["reduction"] == "mean"


def test_named_head_losses_bind_head_id_to_its_target_field() -> None:
    composite = CriterionsManager.build_model_composite_loss(
        "autoencoder",
        {
            "heads": {
                "condition_a": {
                    "ce": {
                        "target": "MaskedCrossEntropyLoss",
                        "weight": 1.0,
                        "params": {},
                    }
                }
            }
        },
        head_specs={"condition_a": {"target_field": "condition"}},
    )

    criterion = composite.loss_functions["condition_a__ce"]

    assert criterion.head_id == "condition_a"
    assert criterion.target_field == "condition"


@pytest.mark.parametrize("criterion_name", ["MSELoss", "SobolevLoss"])
def test_reconstruction_criterions_return_differentiable_scalars(criterion_name: str) -> None:
    """Reconstruction criteria accept the standardized model output mapping."""
    criterion = CriterionsManager._REGISTRY["autoencoder"]["reconstruction"][criterion_name]()
    original = torch.rand(4, 16)
    reconstruction = torch.rand(4, 16, requires_grad=True)

    loss = criterion({"reconstruction": reconstruction}, (torch.arange(4), original))
    loss.backward()

    assert loss.ndim == 0
    assert reconstruction.grad is not None
    assert isinstance(criterion.export_config(), dict)


@pytest.mark.parametrize("criterion_name", ["MSELoss", "SobolevLoss", "MassersteinLoss"])
def test_reconstruction_criterions_reject_negative_intensities(
    criterion_name: str,
) -> None:
    """Every reconstruction objective enforces the MSI intensity domain."""
    criterion = CriterionsManager._REGISTRY["autoencoder"]["reconstruction"][
        criterion_name
    ]()
    original = torch.rand(2, 8)
    reconstruction = torch.rand(2, 8)
    reconstruction[0, 0] = -0.1

    with pytest.raises(IncompatibleInterfaceError, match="non-negative intensities"):
        criterion(
            {"reconstruction": reconstruction},
            (torch.arange(2), original),
        )


def test_composite_loss_combines_registered_criterions() -> None:
    """The manager builds a weighted loss and exposes component metrics."""
    composite = CriterionsManager.build_composite_loss(
        "autoencoder",
        {
            "reconstruction": {
                "MSELoss": {"weight": 1.0, "params": {}},
                "SobolevLoss": {"weight": 0.25, "params": {}},
            },
        },
    )
    original = torch.rand(4, 16)
    reconstruction = torch.rand(4, 16, requires_grad=True)

    loss, metrics = composite(
        {"reconstruction": reconstruction}, (torch.arange(4), original)
    )

    assert loss.ndim == 0
    assert set(metrics) == {"MSELoss", "SobolevLoss", "total_loss"}
    loss.backward()
    assert reconstruction.grad is not None


def test_composite_loss_accepts_ready_criterion_instance() -> None:
    """A user-created criterion is retained in the composite loss."""
    criterion = CriterionsManager._REGISTRY["autoencoder"]["reconstruction"]["MSELoss"]()

    composite = CriterionsManager.build_composite_loss(
        "autoencoder",
        {"reconstruction": {"custom_mse": criterion}},
    )

    assert composite.loss_functions["custom_mse"] is criterion


def test_criterion_output_contract_uses_global_interface_error() -> None:
    """Missing model output fields raise the standardized interface exception."""
    criterion_class = CriterionsManager._REGISTRY["autoencoder"]["reconstruction"]["MSELoss"]
    with pytest.raises(
        IncompatibleInterfaceError,
        match=r"\[RECONSTRUCTIONCRITERION INTERFACE ERROR\]",
    ):
        criterion_class()({}, (torch.arange(1), torch.zeros(1, 4)))


def test_info_nce_builds_peak_bank_and_routes_augmented_projection() -> None:
    """Contrastive hooks sample envelopes, double the batch, and backpropagate."""

    class PeakDataset:
        def __len__(self) -> int:
            return 6

        def __getitem__(self, index: int):
            spectrum = torch.zeros(16)
            spectrum[3:6] = torch.tensor([0.5, 2.0 + index, 0.5])
            spectrum[10:13] = torch.tensor([0.25, 1.0, 0.25])
            return index, spectrum

    criterion = MSIInfoNCELoss(
        max_peaks_per_spectrum=1,
        peak_sample_size=4,
        peak_sample_seed=3,
    )
    cache = {}
    criterion.on_phase_start(torch.nn.Identity(), PeakDataset(), cache)
    batch = (
        torch.arange(4),
        torch.stack([PeakDataset()[index][1] for index in range(4)]),
    )
    augmented_batch = criterion.on_batch_start(batch, cache)
    projection = torch.randn(8, 3, requires_grad=True)

    value = criterion({"projection": projection}, augmented_batch)
    value.backward()

    assert cache["chemical_peak_bank"]
    assert augmented_batch[1].shape == (8, 16)
    assert torch.isfinite(value)
    assert projection.grad is not None


@pytest.mark.parametrize(
    "calculation_method",
    ["exact_autograd_jacobian", "approximate_hutchinson_vjp"],
)
def test_contractive_loss_matches_linear_encoder_jacobian(
    calculation_method: str,
) -> None:
    """Exact and one-dimensional Hutchinson methods recover the analytic norm."""
    encoder = torch.nn.Linear(3, 1, bias=False)
    encoder.weight.data.copy_(torch.tensor([[1.0, -2.0, 0.5]]))
    spectra = torch.randn(4, 3)
    batch = (torch.arange(4), spectra)
    criterion = MSIContractiveLoss(
        calculation_method=calculation_method,
        num_probes=5,
    )
    batch = criterion.on_batch_start(batch, {})
    latent = encoder(batch[1])  # (B=4, D=1)

    loss = criterion({"latent_space": latent}, batch)  # ()
    loss.backward()

    assert loss.item() == pytest.approx(5.25)
    assert encoder.weight.grad is not None
    assert torch.isfinite(encoder.weight.grad).all()


@pytest.mark.parametrize(
    ("method", "protected"),
    [
        ("permutation_random", None),
        (
            "permutation_label_invariant",
            ProtectedIntervalIndex(
                spectrum_ids=torch.tensor([7]).numpy(),
                offsets=torch.tensor([0, 1]).numpy(),
                left=torch.tensor([0], dtype=torch.int32).numpy(),
                right=torch.tensor([3], dtype=torch.int32).numpy(),
            ),
        ),
    ],
)
def test_peak_permutation_preserves_tic_and_spectrum_specific_annotations(
    method: str,
    protected: ProtectedIntervalIndex | None,
) -> None:
    """Peak permutation preserves TIC and never modifies protected intervals."""
    criterion = MSIInfoNCELoss(
        peak_selection_method=method,
        permuted_peaks_per_view=2,
        preserve_input_normalization=True,
    )
    spectrum = torch.tensor(
        [[1.0, 3.0, 2.0, 0.0, 4.0, 1.0, 0.0, 2.0, 2.0]]
    )  # (B=1, M=9)
    batch = (torch.tensor([7]), spectrum)
    cache = {
        criterion._cache_key: {
            "catalogue": ((0, 3), (3, 6), (6, 9)),
            "protected": protected,
        }
    }

    augmented_batch = criterion.on_batch_start(batch, cache)
    augmented = augmented_batch[1][1:]  # (B=1, M=9)

    assert augmented.sum().item() == pytest.approx(spectrum.sum().item())
    if protected is not None:
        assert torch.equal(augmented[0, :3], spectrum[0, :3])


def test_peak_permutation_preserves_each_typed_batch_spectrum_tic() -> None:
    """Every augmented model input retains the typed batch TIC contract."""
    criterion = MSIInfoNCELoss(
        peak_selection_method="permutation_random",
        permuted_peaks_per_view=3,
        preserve_input_normalization=True,
    )
    spectra = torch.tensor(
        [
            [0.40, 0.10, 0.05, 0.05, 0.10, 0.05, 0.15, 0.05, 0.05],
            [0.02, 0.03, 0.05, 0.10, 0.20, 0.10, 0.30, 0.10, 0.10],
        ],
        dtype=torch.float32,
    )  # (B=2, M=9)
    batch = SpectrumBatch(
        sample_ids=torch.tensor([7, 8]),
        spectra=spectra,
        space=SpectrumSpace(
            mass_axis=torch.arange(9, dtype=torch.float32),
            normalization="tic",
        ),
    )
    cache = {
        criterion._cache_key: {
            "catalogue": ((0, 2), (2, 6), (6, 9)),
            "protected": None,
        }
    }

    augmented_batch = criterion.on_batch_start(batch, cache)
    model_input = augmented_batch.model_input()  # (2B, M)

    assert torch.allclose(
        model_input.sum(dim=1),
        torch.ones(2 * batch.batch_size),
        atol=1e-6,
    )
