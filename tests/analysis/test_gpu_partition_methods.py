"""CUDA-only analytical contracts for the spectral comparison backend."""

import numpy as np
import pytest
import torch

from msi_autoencoder_wrapper.analysis.autoencoder.spatial.gpu_partition_methods import (
    cluster_on_cuda, latent_rank_on_cuda,
)
from msi_autoencoder_wrapper.analysis.autoencoder.spatial.synthetic_partitions import (
    infer_frozen_encoder, spectral_partition,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required; no CPU fallback")


@pytest.fixture
def cuda_options():
    return dict(classes=3, seed=101, min_samples=8, radius_quantile=.95,
                gmm_regularization=1e-6, max_iter=100)


@pytest.fixture
def cuda_spectra():
    return spectral_partition("many_stripes", side=16, classes=3,
        mass_axis=np.linspace(100, 200, 101), peak_positions=[120., 140., 160., 180.],
        peak_width_mz=1., amplitude_noise=0., seed=101, device="cuda")


@pytest.mark.parametrize("method", ["kmeans", "gmm_diag", "dbscan"])
def test_cuda_methods_recover_analytical_prototypes(method, cuda_spectra, cuda_options):
    labels, metadata = cluster_on_cuda(cuda_spectra.spectra, method=method, **cuda_options)
    repeated, _ = cluster_on_cuda(cuda_spectra.spectra, method=method, **cuda_options)
    np.testing.assert_array_equal(labels, repeated)
    assert labels.shape == cuda_spectra.labels.shape and labels.dtype == np.int64
    assert metadata["backend"] == "torch_cuda" and metadata["compute_dtype"] == "float32"
    assert metadata["converged"]
    region_clusters = [np.unique(labels[cuda_spectra.labels == region]) for region in range(3)]
    assert all(len(ids) == 1 and ids[0] >= 0 for ids in region_clusters)
    assert len({int(ids[0]) for ids in region_clusters}) == 3


@pytest.mark.parametrize("method", ["kmeans", "gmm_diag", "dbscan"])
def test_cuda_constant_input_has_single_finite_partition(method, cuda_options):
    labels, metadata = cluster_on_cuda(np.ones((24, 4), dtype=np.float32), method=method, **cuda_options)
    assert np.unique(labels).tolist() == [0]
    assert metadata["converged"]


def test_cuda_density_components_and_noise(cuda_options):
    # Exact binary spacings make the analytical radius independent of rounding.
    values = np.array([[0., 0.], [0., 1/128], [8., 8.], [8., 8+1/128], [32., 32.]], dtype=np.float32)
    labels, metadata = cluster_on_cuda(values, method="dbscan", **dict(cuda_options, classes=2, min_samples=2, radius_quantile=0.))
    assert labels[0] == labels[1] >= 0
    assert labels[2] == labels[3] >= 0 and labels[0] != labels[2]
    assert labels[4] == -1 and metadata["density_radius"] == 1/128


@pytest.mark.parametrize("invalid", [np.full((12, 3), np.nan), np.full((12, 3), np.inf), np.empty((0, 3)), np.ones(12)])
def test_cuda_rejects_invalid_values(invalid, cuda_options):
    with pytest.raises(ValueError):
        cluster_on_cuda(invalid, method="kmeans", **cuda_options)


def test_cuda_rejects_invalid_ridge_and_unknown_method(cuda_options):
    values = np.ones((24, 4), dtype=np.float32)
    with pytest.raises(ValueError):
        cluster_on_cuda(values, method="gmm_diag", **dict(cuda_options, gmm_regularization=np.nan))
    with pytest.raises(ValueError):
        cluster_on_cuda(values, method="hdbscan", **cuda_options)


def test_cuda_latent_rank_constant_rank_one_and_scale():
    assert latent_rank_on_cuda(np.zeros((12, 3), dtype=np.float32)) == 0.
    x = np.arange(-6, 6, dtype=np.float32)[:, None] * np.array([[1., 2., -3.]], dtype=np.float32)
    # Covariance eigensolver roundoff can produce tiny positive eigenvalues.
    assert latent_rank_on_cuda(x) == pytest.approx(1., abs=1e-5)
    assert latent_rank_on_cuda(x * 1e20) == pytest.approx(1., abs=1e-5)
    with pytest.raises(ValueError):
        latent_rank_on_cuda(np.full((12, 3), np.nan))


def test_cuda_spectral_generator_seed_tic_and_exact_prototypes(cuda_spectra):
    other = spectral_partition("many_stripes", side=16, classes=3,
        mass_axis=np.linspace(100, 200, 101), peak_positions=[120., 140., 160., 180.],
        peak_width_mz=1., amplitude_noise=0., seed=101, device="cuda")
    np.testing.assert_array_equal(cuda_spectra.spectra, other.spectra)
    values = torch.as_tensor(cuda_spectra.spectra, device="cuda")  # (N, M)
    assert torch.isfinite(values).all() and (values >= 0).all()
    torch.testing.assert_close(values.sum(dim=1), torch.ones(len(values), device="cuda"), atol=1e-6, rtol=1e-6)


class IdentityFixtureEncoder(torch.nn.Module):
    """Deterministic CUDA encoder fixture with mixed module training flags."""

    def __init__(self):
        super().__init__()
        self.dropout = torch.nn.Dropout(.5)
        self.register_parameter("scale", torch.nn.Parameter(torch.tensor(2., device="cuda")))

    def forward(self, spectra):
        return {"latent_space": self.dropout(spectra[:, :3]) * self.scale, "reconstruction": spectra}


def test_cuda_frozen_inference_repeat_flags_state_and_batch(cuda_spectra):
    model = IdentityFixtureEncoder().train()
    model.dropout.eval()
    original = model.scale.detach().clone()
    first, _ = infer_frozen_encoder(model, cuda_spectra.spectra, device="cuda", batch_size=13)
    second, _ = infer_frozen_encoder(model, cuda_spectra.spectra, device="cuda", batch_size=101)
    np.testing.assert_array_equal(first, second)
    assert model.training and not model.dropout.training
    torch.testing.assert_close(model.scale, original, atol=0, rtol=0)


@pytest.mark.parametrize("method", ["kmeans", "gmm_diag", "dbscan"])
def test_cuda_rejects_unrepresentable_distances(method, cuda_options):
    values = np.zeros((24, 4), dtype=np.float32)
    values[12:] = 3e38
    with pytest.raises(ValueError, match="overflowed"):
        cluster_on_cuda(values, method=method, **cuda_options)
