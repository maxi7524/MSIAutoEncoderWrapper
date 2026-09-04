"""Tests for criterion discovery, creation, and composition."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from msi_autoencoder_wrapper.data import SpectrumBatch, SpectrumSpace
from msi_autoencoder_wrapper.training.criterions.criterions_manager import (
    CriterionsManager,
)
from msi_autoencoder_wrapper.training.criterions.autoencoder.contrastive.infoNCE_loss import (
    MSIInfoNCELoss,
    ProtectedIntervalIndex,
    _build_permutation_groups,
    _permute_envelopes_batch,
)
from msi_autoencoder_wrapper.training.criterions.autoencoder.regularization.contractive_loss import (
    MSIContractiveLoss,
)
from msi_autoencoder_wrapper.training.criterions.autoencoder.regularization.uniformity_loss import (
    MSIUniformityLoss,
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
    assert "UniformityLoss" in available["regularization"]
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
        max_peaks_per_spectrum=2,
        peak_sample_size=4,
        peak_sample_seed=3,
        permutation_bank_size=16,
        permuted_peaks_per_view=2,
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


class _LinearAutoencoder(torch.nn.Module):
    """Minimal model exposing the autoencoder output contract for loss tests."""

    def __init__(self, weight: torch.Tensor) -> None:
        super().__init__()
        self.encoder = torch.nn.Linear(weight.shape[1], weight.shape[0], bias=False)
        self.encoder.weight.data.copy_(weight)
        self.forward_calls = 0

    def forward(self, spectra: torch.Tensor) -> dict[str, torch.Tensor]:
        """Return the test encoder output."""
        self.forward_calls += 1
        return {"latent_space": self.encoder(spectra)}


def test_spectral_contractive_loss_matches_materialized_linear_jacobian() -> None:
    """Power iteration recovers the dominant singular-value penalty."""
    model = _LinearAutoencoder(torch.diag(torch.tensor([2.0, 0.25])))
    spectra = torch.randn(4, 2)
    criterion = MSIContractiveLoss(
        calculation_method="exact_autograd_jacobian",
        penalty_metric="spectral",
    )
    criterion.on_phase_start(model, object(), {})
    batch = criterion.on_batch_start((torch.arange(4), spectra), {})

    torch.manual_seed(7)
    loss = criterion(model(batch[1]), batch)
    materialized = model.encoder.weight.detach()
    expected = torch.linalg.svdvals(materialized)[0].square()

    loss.backward()

    assert loss.item() == pytest.approx(expected.item(), rel=1.0e-4)
    assert model.encoder.weight.grad is not None
    assert model.forward_calls == 1


class _NonlinearAutoencoder(torch.nn.Module):
    """Tanh encoder used to verify spectral VJP directions."""

    def __init__(self, weight: torch.Tensor) -> None:
        super().__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(3, 3, bias=False),
            torch.nn.Tanh(),
        )
        self.encoder[0].weight.data.copy_(weight)

    def forward(self, spectra: torch.Tensor) -> dict[str, torch.Tensor]:
        """Return a nonlinear latent representation."""
        return {"latent_space": self.encoder(spectra)}


def test_spectral_contractive_loss_detaches_power_iteration_directions() -> None:
    """Power iteration estimates a nonlinear Jacobian singular value correctly."""
    torch.manual_seed(62)
    model = _NonlinearAutoencoder(2.0 * torch.randn(3, 3))
    spectra = 2.0 * torch.randn(1, 3)
    criterion = MSIContractiveLoss(penalty_metric="spectral")
    criterion.on_phase_start(model, object(), {})
    batch = criterion.on_batch_start((torch.arange(1), spectra), {})

    torch.manual_seed(1062)
    loss = criterion(model(batch[1]), batch)
    latent = model(batch[1])["latent_space"]
    jacobian_rows = [
        torch.autograd.grad(latent[0, index], batch[1], retain_graph=True)[0][0]
        for index in range(latent.shape[1])
    ]
    expected = torch.linalg.svdvals(torch.stack(jacobian_rows))[0].square()

    assert loss.item() == pytest.approx(expected.item(), rel=1.0e-5)


def test_contractive_loss_rejects_batch_normalized_encoder() -> None:
    """Per-spectrum Jacobian regularization requires batch-separable encoders."""
    model = _LinearAutoencoder(torch.eye(2))
    model.encoder = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.BatchNorm1d(2))
    criterion = MSIContractiveLoss()

    with pytest.raises(IncompatibleInterfaceError, match="LayerNorm instead of BatchNorm"):
        criterion.on_phase_start(model, object(), {})


def test_hinged_contractive_loss_has_zero_input_gradient_below_threshold() -> None:
    """The hinge stops further contraction once the Frobenius norm is small."""
    encoder = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.Tanh())
    spectra = torch.randn(3, 2)
    criterion = MSIContractiveLoss(
        calculation_method="exact_autograd_jacobian",
        penalty_metric="hinged",
        hinge_threshold=100.0,
    )
    batch = criterion.on_batch_start((torch.arange(3), spectra), {})
    loss = criterion({"latent_space": encoder(batch[1])}, batch)
    input_gradient = torch.autograd.grad(loss, batch[1])[0]

    assert loss.item() == pytest.approx(0.0)
    assert torch.equal(input_gradient, torch.zeros_like(input_gradient))


class _LayerNormEncoder(torch.nn.Module):
    """Small affine-LayerNorm encoder matching the canonicalization contract."""

    def __init__(self) -> None:
        super().__init__()
        self.bottleneck_layer = torch.nn.Sequential(
            torch.nn.Linear(3, 3, bias=False),
            torch.nn.LayerNorm(3),
        )

    def forward(self, spectra: torch.Tensor) -> torch.Tensor:
        """Return post-LayerNorm latents."""
        return self.bottleneck_layer(spectra)


class _CanonicalizedAutoencoder(torch.nn.Module):
    """Expose a canonicalizable encoder using the autoencoder output contract."""

    def __init__(self) -> None:
        super().__init__()
        self.encoder = _LayerNormEncoder()

    def forward(self, spectra: torch.Tensor) -> dict[str, torch.Tensor]:
        """Return the post-LayerNorm representation."""
        return {"latent_space": self.encoder(spectra)}


def _contractive_penalty_for_gamma(
    gamma: float,
    penalized_space: str,
) -> float:
    """Evaluate a contractive loss after setting the bottleneck affine scale."""
    torch.manual_seed(11)
    model = _CanonicalizedAutoencoder()
    layer_norm = model.encoder.bottleneck_layer[-1]
    assert isinstance(layer_norm, torch.nn.LayerNorm)
    layer_norm.weight.data.fill_(gamma)
    layer_norm.bias.data.copy_(torch.tensor([0.2, -0.1, 0.4]))
    criterion = MSIContractiveLoss(
        calculation_method="exact_autograd_jacobian",
        penalized_space=penalized_space,
    )
    criterion.on_phase_start(model, object(), {})
    batch = criterion.on_batch_start((torch.arange(4), torch.randn(4, 3)), {})
    return criterion(model(batch[1]), batch).item()


def test_canonicalized_contractive_penalty_is_invariant_to_layer_norm_gamma() -> None:
    """Canonicalized penalties cannot be reduced by affine LayerNorm rescaling."""
    u_gamma_one = _contractive_penalty_for_gamma(1.0, "u")
    u_gamma_three = _contractive_penalty_for_gamma(3.0, "u")
    z_gamma_one = _contractive_penalty_for_gamma(1.0, "z")
    z_gamma_three = _contractive_penalty_for_gamma(3.0, "z")

    assert u_gamma_three == pytest.approx(u_gamma_one, rel=1.0e-5)
    assert z_gamma_three == pytest.approx(9.0 * z_gamma_one, rel=1.0e-5)


def test_uniformity_loss_prefers_spread_canonicalized_latents() -> None:
    """A spread set on the LayerNorm sphere has lower uniformity loss."""
    model = _CanonicalizedAutoencoder()
    layer_norm = model.encoder.bottleneck_layer[-1]
    assert isinstance(layer_norm, torch.nn.LayerNorm)
    layer_norm.weight.data.fill_(2.0)
    layer_norm.bias.data.copy_(torch.tensor([0.3, -0.2, 0.1]))
    criterion = MSIUniformityLoss()
    criterion.on_phase_start(model, object(), {})
    scale = float(np.sqrt(1.5))
    clustered_u = torch.tensor([[scale, -scale, 0.0]]).repeat(4, 1)
    spread_u = torch.tensor(
        [
            [scale, -scale, 0.0],
            [-scale, scale, 0.0],
            [scale, 0.0, -scale],
            [-scale, 0.0, scale],
        ],
        requires_grad=True,
    )
    clustered_z = clustered_u * layer_norm.weight + layer_norm.bias
    spread_z = spread_u * layer_norm.weight + layer_norm.bias

    batch = (torch.arange(4), torch.zeros(4, 3))
    clustered_loss = criterion({"latent_space": clustered_z}, batch)
    spread_loss = criterion({"latent_space": spread_z}, batch)
    spread_loss.backward()

    assert criterion.requires_input_grad is False
    assert spread_loss < clustered_loss
    assert spread_u.grad is not None
    assert torch.isfinite(spread_u.grad).all()


def test_uniformity_loss_is_finite_for_large_pairwise_penalties() -> None:
    """Log-sum-exp evaluation avoids exponential underflow for distant pairs."""
    model = _CanonicalizedAutoencoder()
    criterion = MSIUniformityLoss(temperature=200.0)
    criterion.on_phase_start(model, object(), {})
    scale = float(np.sqrt(1.5))
    canonical_u = torch.tensor([[scale, -scale, 0.0], [-scale, scale, 0.0]])
    layer_norm = model.encoder.bottleneck_layer[-1]
    assert isinstance(layer_norm, torch.nn.LayerNorm)
    latent = canonical_u * layer_norm.weight + layer_norm.bias

    loss = criterion({"latent_space": latent}, (torch.arange(2), torch.zeros(2, 3)))

    assert torch.isfinite(loss)


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
            "permutation_groups": np.asarray(
                [
                    (
                        ((3, 6), (6, 9))
                        if protected is not None
                        else ((0, 3), (3, 6))
                    )
                ],
                dtype=np.int32,
            ),
            "max_envelope_width": 3,
            "device_group_cache": {},
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
            "permutation_groups": np.asarray(
                [[(0, 2), (2, 6), (6, 9)]],
                dtype=np.int32,
            ),
            "max_envelope_width": 4,
            "device_group_cache": {},
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


def test_peak_permutation_bank_contains_non_overlapping_triples() -> None:
    """The reusable random bank resolves overlap before batch training."""
    groups = _build_permutation_groups(
        ((0, 3), (2, 5), (5, 7), (8, 11), (12, 14)),
        group_size=3,
        group_count=128,
        seed=7,
    )

    assert groups.shape == (128, 3, 2)
    for group in groups:
        ordered = group[np.argsort(group[:, 0])]
        assert np.all(ordered[:-1, 1] <= ordered[1:, 0])


def test_peak_permutation_cyclically_swaps_three_envelopes() -> None:
    """One precomputed triple is gathered, permuted, and scattered batchwise."""
    criterion = MSIInfoNCELoss(
        permutation_bank_size=1,
        permuted_peaks_per_view=3,
        permutation_selection_attempts=1,
        preserve_input_normalization=True,
    )
    spectrum = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]])
    cache = {
        criterion._cache_key: {
            "catalogue": ((0, 2), (2, 4), (4, 6)),
            "permutation_groups": np.asarray(
                [[(0, 2), (2, 4), (4, 6)]],
                dtype=np.int32,
            ),
            "max_envelope_width": 2,
            "device_group_cache": {},
            "protected": None,
        }
    }

    augmented_batch = criterion.on_batch_start(
        (torch.tensor([0]), spectrum),
        cache,
    )

    assert torch.allclose(
        augmented_batch[1][1:],
        torch.tensor([[3.0, 4.0, 5.0, 6.0, 1.0, 2.0]]),
    )


def test_vectorized_peak_permutation_matches_variable_width_interpolation() -> None:
    """Batched gather/scatter matches linear interpolation for unequal envelopes."""
    spectrum = torch.tensor([[1.0, 2.0, 3.0, 6.0, 9.0, 4.0, 8.0, 12.0, 16.0]])
    selected = torch.tensor([[[0, 2], [2, 5], [5, 9]]])

    augmented = _permute_envelopes_batch(
        spectrum,
        selected,
        torch.tensor([True]),
        max_envelope_width=4,
        preserve_mass=False,
    )
    expected = spectrum.clone()
    intervals = ((0, 2), (2, 5), (5, 9))
    for destination_index, (left, right) in enumerate(intervals):
        source_left, source_right = intervals[(destination_index + 1) % 3]
        expected[0, left:right] = torch.nn.functional.interpolate(
            spectrum[0, source_left:source_right].view(1, 1, -1),
            size=right - left,
            mode="linear",
            align_corners=False,
        ).view(-1)

    assert torch.allclose(augmented, expected)


def test_label_invariant_permutation_rejects_a_protected_bank_group() -> None:
    """A sampled group crossing this spectrum's annotation leaves its view unchanged."""
    criterion = MSIInfoNCELoss(
        peak_selection_method="permutation_label_invariant",
        permutation_bank_size=1,
        permuted_peaks_per_view=2,
        permutation_selection_attempts=1,
    )
    spectrum = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    protected = ProtectedIntervalIndex(
        spectrum_ids=np.asarray([12], dtype=np.int64),
        offsets=np.asarray([0, 1], dtype=np.int64),
        left=np.asarray([0], dtype=np.int32),
        right=np.asarray([2], dtype=np.int32),
    )
    cache = {
        criterion._cache_key: {
            "catalogue": ((0, 2), (2, 4)),
            "permutation_groups": np.asarray(
                [[(0, 2), (2, 4)]],
                dtype=np.int32,
            ),
            "max_envelope_width": 2,
            "device_group_cache": {},
            "protected": protected,
        }
    }

    augmented_batch = criterion.on_batch_start(
        (torch.tensor([12]), spectrum),
        cache,
    )

    assert torch.equal(augmented_batch[1][1:], spectrum)
