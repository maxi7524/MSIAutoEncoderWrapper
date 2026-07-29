"""Behavioral tests for cached single-image autoencoder analysis."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch
import matplotlib.pyplot as plt
from torch.utils.data import Dataset

from msi_autoencoder_wrapper.analysis import (
    AutoencoderAnalysis,
    AutoencoderMultiAnalysis,
)
from msi_autoencoder_wrapper.visualization import VisualizationTheme
from msi_autoencoder_wrapper.analysis.autoencoder.visualizations import (
    plot_metric_distribution,
    plot_projection,
    plot_spatial_image,
    plot_spectrum_comparison,
)
from msi_autoencoder_wrapper.analysis.autoencoder import (
    plot_metric_distribution,
    plot_projection,
    plot_spatial_image,
    plot_spectrum_comparison,
)
from msi_autoencoder_wrapper.readers.spatial import SpatialImage


class _Dataset(Dataset):
    target_specs = {
        "condition": {"type": "single_label"},
        "molecule": {"type": "multi_label"},
    }

    def __init__(self) -> None:
        self.values = torch.tensor(
            [
                [1.0, 2.0, 0.0],
                [0.0, 2.0, 2.0],
                [3.0, 0.0, 1.0],
                [1.0, 1.0, 2.0],
            ]
        )

    def __len__(self) -> int:
        return len(self.values)

    def __getitem__(self, index: int):
        targets = {
            "condition": torch.tensor(index % 2),
            "molecule": torch.tensor([index % 2, (index + 1) % 2]).float(),
        }
        masks = {"condition": torch.tensor(True), "molecule": torch.tensor(True)}
        return index, self.values[index], targets, masks

    def get_class_mappings(self):
        return {
            "condition": {"healthy": 0, "disease": 1},
            "molecule": {"A": 0, "B": 1},
        }


class _Model(torch.nn.Module):
    head_specs = {
        "condition_head": {"target_field": "condition"},
        "molecule_head": {"target_field": "molecule"},
    }

    def forward(self, values: torch.Tensor):
        latent = values[:, :2]
        return {
            "latent_space": latent,
            "reconstruction": values * 0.5,
            "head_condition_head": torch.stack((latent[:, 0], latent[:, 1]), dim=1),
            "head_molecule_head": torch.stack(
                (latent[:, 0] - 1.0, latent[:, 1] - 1.0), dim=1
            ),
        }


class _Reader:
    coordinates = ((0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0))

    def GetNumberOfSpectra(self) -> int:
        return len(self.coordinates)

    def MapSpectrumValuesToImage(self, values: np.ndarray) -> SpatialImage:
        grid = np.asarray(values).reshape(1, 2, 2)
        return SpatialImage(grid, np.ones_like(grid, dtype=bool), (0, 1, 0, 1, 0, 0))

    def GetIonImage(self, mz, tolerance, aggregation="mean") -> SpatialImage:
        values = np.full((1, 2, 2), mz + tolerance, dtype=np.float32)
        return SpatialImage(
            values, np.ones_like(values, dtype=bool), (0, 1, 0, 1, 0, 0)
        )


class _Binner:
    def __call__(self, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
        result = np.zeros(3, dtype=np.float32)
        for mz, intensity in zip(xs, ys):
            matches = np.flatnonzero(self.GetXAxis() == mz)
            if len(matches):
                result[matches[0]] += intensity
        return result

    def GetXAxis(self) -> np.ndarray:
        return np.asarray([100.0, 101.0, 102.0])

    def GetBinIndices(self, mz: float, tolerance: float = 0.0) -> np.ndarray:
        axis = self.GetXAxis()
        return np.flatnonzero(np.abs(axis - mz) <= tolerance)


class _InverseBinner:
    def __call__(self, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        selected = np.flatnonzero(values > 0.5)
        return np.asarray([100.0, 101.0, 102.0])[selected], values[selected]


def _wrapper(trained: bool = True):
    model = _Model()
    reader = _Reader()
    context = SimpleNamespace(
        reader=reader,
        binner=_Binner(),
        inverse_binner=_InverseBinner(),
    )
    interface = SimpleNamespace(is_trained=trained)
    manager = SimpleNamespace(autoencoder=interface, batch_size=2)
    return SimpleNamespace(
        active_dataset=_Dataset(),
        active_context=context,
        active_model=model,
        models_manager=manager,
        device="cpu",
    )


class _MultiWrapper:
    def __init__(self) -> None:
        source = _wrapper()
        self.active_dataset = source.active_dataset
        self.active_context = source.active_context
        self.active_model = source.active_model
        self.models_manager = source.models_manager
        self.device = "cpu"

    def load_configuration(self, model_name: str) -> None:
        self.active_model = _Model()
        scale = 0.5 if model_name == "baseline" else 0.75

        def forward(values):
            outputs = _Model.forward(self.active_model, values)
            outputs["reconstruction"] = values * scale
            return outputs

        self.active_model.forward = forward


def test_prepare_caches_metrics_arrays_and_generic_heads() -> None:
    """One pass retains configured arrays and evaluates both target types."""
    analysis = AutoencoderAnalysis(_wrapper())

    estimate = analysis.estimate_prepare_size()
    prepared = analysis.prepare(batch_size=2)
    head_metrics = analysis.evaluate_heads()

    assert estimate.sample_count == 4
    assert estimate.total_bytes > 0
    assert prepared.arrays["inputs"].shape == (4, 3)
    assert prepared.arrays["latents"].shape == (4, 2)
    assert prepared.feature_metrics["feature_mse"].shape == (3,)
    assert set(head_metrics) == {"condition_head", "molecule_head"}
    assert "accuracy" in head_metrics["condition_head"]
    assert "micro_f1" in head_metrics["molecule_head"]


def test_cached_results_support_spatial_and_selection_views() -> None:
    """Spatial views and spectrum selection reuse prepared values."""
    analysis = AutoencoderAnalysis(_wrapper())
    analysis.prepare()

    error_image = analysis.reconstruction_error_image("mse")
    ion_images = analysis.ion_image_comparison(101.0)
    head_image = analysis.head_probability_image("molecule_head", 0)

    assert error_image.values.shape == (1, 2, 2)
    assert set(ion_images) == {"input", "reconstruction", "residual"}
    assert head_image.values.shape == (1, 2, 2)
    assert len(analysis.select_spectra(selection="worst", count=2)) == 2
    assert analysis.reconstruction_summary("mae")["count"] == 4.0
    assert analysis.latent_projection().shape == (4, 2)
    assert analysis.latent_image(0).values.shape == (1, 2, 2)


def test_prepare_retention_is_explicit() -> None:
    """Discarded large matrices are not recomputed implicitly."""
    analysis = AutoencoderAnalysis(_wrapper())
    analysis.prepare(retain={"latents"})

    assert set(analysis.prepared.arrays) == {"latents"}
    with pytest.raises(Exception, match="inputs.*not retained"):
        analysis.ion_image_comparison(101.0)


def test_initialization_rejects_untrained_model() -> None:
    """Analysis validates inference readiness immediately."""
    with pytest.raises(Exception, match="not marked as trained"):
        AutoencoderAnalysis(_wrapper(trained=False))


def test_binner_and_inverse_binner_have_separate_reports() -> None:
    """Preprocessing diagnostics do not mix forward and inverse transforms."""
    wrapper = _wrapper()
    wrapper.active_context.reader.GetSpectrum = lambda index: (
        np.asarray([100.0, 101.0, 102.0]),
        wrapper.active_dataset.values[index].numpy(),
    )
    analysis = AutoencoderAnalysis(wrapper)

    forward_report = analysis.binner_report([0, 1])
    inverse_report = analysis.inverse_binner_report([0, 1])

    assert forward_report["finite_fraction"] == 1.0
    assert forward_report["mean_tic_ratio"] == pytest.approx(1.0)
    assert inverse_report["mean_mse"] >= 0.0


def test_visualizations_reuse_precomputed_values() -> None:
    """Visualization helpers consume arrays without invoking the model."""
    analysis = AutoencoderAnalysis(_wrapper())
    prepared = analysis.prepare()

    figures = [
        plot_metric_distribution(prepared)[0],
        plot_spatial_image(analysis.reconstruction_error_image().values, "MSE")[0],
        plot_spectrum_comparison(
            analysis.binner.GetXAxis(),
            prepared.arrays["inputs"][0],
            prepared.arrays["reconstructions"][0],
        )[0],
        plot_projection(analysis.latent_projection())[0],
    ]

    assert all(figure.axes for figure in figures)


def test_visualizations_consume_precomputed_results() -> None:
    """Plot helpers accept cached arrays without invoking the model."""
    analysis = AutoencoderAnalysis(_wrapper())
    prepared = analysis.prepare()

    figures = [
        plot_metric_distribution(prepared)[0],
        plot_projection(analysis.latent_projection())[0],
        plot_spatial_image(
            analysis.reconstruction_error_image().values,
            "Reconstruction error",
        )[0],
        plot_spectrum_comparison(
            analysis.binner.GetXAxis(),
            prepared.arrays["inputs"][0],
            prepared.arrays["reconstructions"][0],
        )[0],
    ]

    assert all(figure.axes for figure in figures)
    for figure in figures:
        plt.close(figure)


def test_grouped_single_model_api_and_theme() -> None:
    analysis = AutoencoderAnalysis(
        _wrapper(),
        theme=VisualizationTheme(model_overrides={"active_model": "#123456"}),
    )
    analysis.prepare()

    assert analysis.reconstruction.summary("mse")["count"] == 4.0
    assert analysis.latent.statistics()["mean"].shape == (2,)
    assert "molecule_head" in analysis.heads.evaluate()
    assert analysis.heads.probability_image("molecule_head", 0).values.shape == (
        1,
        2,
        2,
    )
    assert analysis.theme.color_for_model("active_model") == "#123456"


def test_multi_analysis_shares_inputs_and_compares_models() -> None:
    analysis = AutoencoderMultiAnalysis(
        _MultiWrapper(), ["baseline", "candidate"]
    )
    prepared = analysis.prepare()

    assert prepared.models["baseline"].arrays["inputs"] is prepared.models[
        "candidate"
    ].arrays["inputs"]
    ranking = analysis.reconstruction.compare_metric("mse")
    assert [row["model"] for row in ranking] == ["candidate", "baseline"]
    assert set(analysis.reconstruction.metric_image()) == {"baseline", "candidate"}
    assert set(analysis.heads.probability_image("molecule_head", 0)) == {
        "baseline",
        "candidate",
    }
