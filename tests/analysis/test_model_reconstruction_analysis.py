"""Tests for trained-model reconstruction vs. raw X, swept across binning steps."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from msi_autoencoder_wrapper.analysis.autoencoder.binning.model_reconstruction_analysis import (
    _dataset_normalize,
    _reconstruct,
    _zoom_window,
    group_tasks_by_binning,
    load_reconstruction_model,
    model_reconstruction_records,
    plot_ranked_model_reconstructions,
    plot_spatial_error_by_split,
    sample_split_ids,
    spatial_reconstruction_error,
    split_spectrum_ids,
)
from msi_autoencoder_wrapper.analysis.autoencoder.binning.precompute import BinningPrecompute
from msi_autoencoder_wrapper.analysis.autoencoder.experiments import read_campaign
from msi_autoencoder_wrapper.utils.exceptions import ValidationError

from tests.analysis._campaign_fixtures import EXPERIMENT_NAME, write_campaign_task
from tests.mocks.components import MockMSIReader


def _precompute() -> BinningPrecompute:
    return BinningPrecompute(MockMSIReader(), sample_size=3, seed=1).load_raw()


def _write_inference_ready_task(
    tmp_path: Path,
    *,
    task_id: str,
    binning_step: float,
    repetition: int = 0,
    architecture_name: str = "mlp-ae",
) -> tuple[BinningPrecompute, object]:
    """Write a task whose config exactly matches a real BinningPrecompute binner."""
    precompute = _precompute()
    binner = precompute.forward_binner(binning_step)
    write_campaign_task(
        tmp_path,
        task_id=task_id,
        architecture_name=architecture_name,
        preset="MLPAutoencoder",
        binning_step=binning_step,
        repetition=repetition,
        input_dim=binner.GetXAxisDepth(),
        binner_range=(binner.x_min, binner.x_max, binning_step),
        save_weights=True,
    )
    tasks = read_campaign(tmp_path, EXPERIMENT_NAME)
    return precompute, tasks[0]


def test_group_tasks_by_binning_filters_architecture_and_status(tmp_path: Path) -> None:
    write_campaign_task(tmp_path, task_id="task_a_045", architecture_name="mlp-ae", preset="MLPAutoencoder", binning_step=0.45, repetition=0, input_dim=6)
    write_campaign_task(tmp_path, task_id="task_a_050", architecture_name="mlp-ae", preset="MLPAutoencoder", binning_step=0.50, repetition=0, input_dim=5)
    write_campaign_task(tmp_path, task_id="task_b_045", architecture_name="conv-ae", preset="CNNAutoencoder", binning_step=0.45, repetition=0, input_dim=6)
    write_campaign_task(tmp_path, task_id="task_a_failed", architecture_name="mlp-ae", preset="MLPAutoencoder", binning_step=0.45, repetition=1, input_dim=6, status="failed")
    tasks = read_campaign(tmp_path, EXPERIMENT_NAME)

    grouped = group_tasks_by_binning(tasks, "mlp-ae")

    assert set(grouped.keys()) == {0.45, 0.50}
    assert [task.task_id for task in grouped[0.45]] == ["task_a_045"]  # failed repetition excluded


def test_split_spectrum_ids_and_sample_split_ids(tmp_path: Path) -> None:
    write_campaign_task(tmp_path, task_id="task_000000", architecture_name="mlp-ae", preset="MLPAutoencoder", binning_step=0.5, repetition=0, input_dim=6)
    task = read_campaign(tmp_path, EXPERIMENT_NAME)[0]

    train_ids = split_spectrum_ids(task, "train")
    np.testing.assert_array_equal(train_ids, np.arange(8))

    with pytest.raises(ValidationError):
        split_spectrum_ids(task, "not_a_split")

    full = sample_split_ids(task, "train", sample_size=100, seed=1)
    np.testing.assert_array_equal(full, train_ids)

    subsample = sample_split_ids(task, "train", sample_size=3, seed=1)
    assert subsample.size == 3
    assert set(subsample.tolist()).issubset(set(train_ids.tolist()))
    assert list(subsample) == sorted(subsample)  # sorted output


def test_load_reconstruction_model_requires_a_result(tmp_path: Path) -> None:
    write_campaign_task(tmp_path, task_id="task_000000", architecture_name="mlp-ae", preset="MLPAutoencoder", binning_step=0.5, repetition=0, input_dim=6, status="failed")
    task = read_campaign(tmp_path, EXPERIMENT_NAME)[0]

    with pytest.raises(ValidationError):
        load_reconstruction_model(task)


def test_load_reconstruction_model_matches_training_time_binner_range_and_normalization(
    tmp_path: Path,
) -> None:
    precompute, task = _write_inference_ready_task(tmp_path, task_id="task_000000", binning_step=5.0)
    binner = precompute.forward_binner(5.0)

    model, (x_min, x_max), (normalization_kind, normalization_epsilon) = load_reconstruction_model(task)

    assert x_min == pytest.approx(binner.x_min)
    assert x_max == pytest.approx(binner.x_max)
    assert normalization_kind == "tic"  # matches _campaign_fixtures._model_config
    assert normalization_epsilon == pytest.approx(1e-12)
    import torch
    with torch.no_grad():
        output = model(torch.zeros(1, binner.GetXAxisDepth()))
    assert output["reconstruction"].shape == (1, binner.GetXAxisDepth())


def test_dataset_normalize_matches_pixel_dataset_tic_formula() -> None:
    values = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)

    normalized = _dataset_normalize(values, "tic", 1e-12)

    np.testing.assert_allclose(normalized, values / 10.0)


def test_dataset_normalize_zeroes_degenerate_spectrum() -> None:
    normalized = _dataset_normalize(np.zeros(4, dtype=np.float32), "tic", 1e-12)

    np.testing.assert_array_equal(normalized, np.zeros(4, dtype=np.float32))


def test_dataset_normalize_none_is_identity() -> None:
    values = np.array([5.0, -3.0], dtype=np.float32)

    np.testing.assert_array_equal(_dataset_normalize(values, "none", 1e-12), values)


def test_dataset_normalize_is_row_independent_for_batches() -> None:
    values = np.array([[1.0, 1.0, 2.0], [10.0, 10.0, 20.0]], dtype=np.float32)

    normalized = _dataset_normalize(values, "tic", 1e-12)

    np.testing.assert_allclose(normalized[0], normalized[1])


def test_reconstruct_is_invariant_to_input_scale_under_tic_normalization(
    tmp_path: Path,
) -> None:
    """Regression test: feeding raw, unnormalized magnitudes into the model (instead
    of first matching the training-time TIC normalization) makes the reconstruction
    depend on the input's arbitrary raw scale, which is the exact bug this module's
    docstring REMARK warns about. Two inputs that are the same spectrum shape scaled
    by a large constant factor must produce an *identical* reconstruction once
    `_reconstruct` normalizes them the same way the model was trained on.
    """
    precompute, task = _write_inference_ready_task(tmp_path, task_id="task_000000", binning_step=5.0)
    model, _, (normalization_kind, normalization_epsilon) = load_reconstruction_model(task)
    binner = precompute.forward_binner(5.0)
    rng = np.random.default_rng(0)
    spectrum = rng.uniform(0.0, 5.0, size=binner.GetXAxisDepth()).astype(np.float32)

    reconstruction_a = _reconstruct(model, spectrum[None, :], "cpu", 1, normalization_kind, normalization_epsilon)
    reconstruction_b = _reconstruct(model, (spectrum * 1000.0)[None, :], "cpu", 1, normalization_kind, normalization_epsilon)

    np.testing.assert_allclose(reconstruction_a, reconstruction_b, rtol=1e-4, atol=1e-6)


def test_model_reconstruction_records_has_expected_schema(tmp_path: Path) -> None:
    precompute, task = _write_inference_ready_task(tmp_path, task_id="task_000000", binning_step=5.0)

    records = model_reconstruction_records(precompute, {5.0: [task]})

    assert records
    assert {record["task_id"] for record in records} == {"task_000000"}
    assert {record["repetition"] for record in records} == {0}
    assert {record["delta_m"] for record in records} == {5.0}
    assert {record["normalization"] for record in records} == {"tic"}
    assert {"wasserstein", "cosine_similarity"}.issubset({record["metric"] for record in records})
    binner = precompute.forward_binner(5.0)
    assert all(record["feature_dimension"] == binner.GetXAxisDepth() for record in records)
    spectrum_ids = {record["spectrum_id"] for record in records}
    assert spectrum_ids == set(int(sid) for sid in precompute.spectrum_ids)


def test_plot_ranked_model_reconstructions_smoke(tmp_path: Path) -> None:
    precompute, task = _write_inference_ready_task(tmp_path, task_id="task_000000", binning_step=5.0)
    records = model_reconstruction_records(precompute, {5.0: [task]})

    figure, axes = plot_ranked_model_reconstructions(precompute, records, task, metric="wasserstein", n_best=1, n_worst=1)

    assert figure is not None
    assert axes.shape[0] == 4 * (1 + 1)  # full + zoom, signal + residual, per chosen spectrum


def test_plot_ranked_model_reconstructions_returns_none_without_finite_records(tmp_path: Path) -> None:
    precompute, task = _write_inference_ready_task(tmp_path, task_id="task_000000", binning_step=5.0)

    figure, axes = plot_ranked_model_reconstructions(precompute, [], task, metric="wasserstein")

    assert figure is None
    assert axes is None


def test_zoom_window_centers_on_single_highest_peak_with_fixed_width() -> None:
    mz = np.array([100.0, 101.0, 150.0, 151.0, 200.0], dtype=np.float32)
    intensity = np.array([1.0, 5.0, 2.0, 9.0, 0.5], dtype=np.float32)

    window = _zoom_window(mz, intensity, window_da=30.0)

    # the single highest peak is at mz=151 (intensity 9.0); window is fixed-width,
    # centered there, regardless of where any other peak sits.
    assert window == pytest.approx((136.0, 166.0))


def test_zoom_window_returns_none_for_empty_spectrum() -> None:
    assert _zoom_window(np.asarray([]), np.asarray([]), window_da=30.0) is None


def test_spatial_reconstruction_error_maps_values_and_leaves_others_nan() -> None:
    reader = MockMSIReader()
    records = [
        {"spectrum_id": 0, "metric": "wasserstein", "value": 12.5},
        {"spectrum_id": 2, "metric": "wasserstein", "value": 7.0},
        {"spectrum_id": 0, "metric": "cosine_similarity", "value": 0.9},  # other metric, ignored
    ]

    image = spatial_reconstruction_error(reader, records, metric="wasserstein")

    position_0 = reader.GetSpectrumPosition(0)
    position_2 = reader.GetSpectrumPosition(2)
    min_x, _, min_y, _, min_z, _ = image.extent
    target_0 = (position_0[2] - min_z, position_0[1] - min_y, position_0[0] - min_x)
    target_2 = (position_2[2] - min_z, position_2[1] - min_y, position_2[0] - min_x)
    assert image.values[target_0] == pytest.approx(12.5)
    assert image.values[target_2] == pytest.approx(7.0)
    # Every position not covered by a spectrum_id in records is left NaN, not zero.
    untouched_count = np.isnan(image.values).sum()
    assert untouched_count == image.values.size - 2


def test_plot_spatial_error_by_split_shares_value_range() -> None:
    reader = MockMSIReader()
    low_records = [{"spectrum_id": 0, "metric": "wasserstein", "value": 1.0}]
    high_records = [{"spectrum_id": 0, "metric": "wasserstein", "value": 99.0}]
    images_by_split = {
        "train": spatial_reconstruction_error(reader, low_records, metric="wasserstein"),
        "test": spatial_reconstruction_error(reader, high_records, metric="wasserstein"),
    }

    figure, axes = plot_spatial_error_by_split(images_by_split, metric="wasserstein")

    assert len(axes) == 2
    train_clim = axes[0].images[0].get_clim()
    test_clim = axes[1].images[0].get_clim()
    assert train_clim == test_clim
    assert train_clim[0] == pytest.approx(1.0)
    assert train_clim[1] == pytest.approx(99.0)
