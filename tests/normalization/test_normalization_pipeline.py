"""Tests for reversible, device-preserving normalization pipelines."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from msi_autoencoder_wrapper.core.wrapper import MSIAutoEncoderWrapper
from msi_autoencoder_wrapper.metrics import MetricRequirements, validate_metric_compatibility
from msi_autoencoder_wrapper.normalization import (
    NormalizationCapabilities,
    NormalizationPipeline,
    NormalizationTrace,
)
from msi_autoencoder_wrapper.training.engine.base_trainer import MSIPyTorchTrainer
from msi_autoencoder_wrapper.models.datasets.splitting import DatasetSplitter, SplitConfig
from tests.mocks.components import MockMSIReader


def test_dense_stacked_normalization_round_trip() -> None:
    """Ordered samplewise steps invert in reverse order without leaving Torch."""
    pipeline = NormalizationPipeline.from_config(
        {
            "stage": "binned",
            "steps": {
                "tic": {"type": "tic"},
                "peak": {"type": "max"},
            },
        }
    )
    source = torch.tensor([[1.0, 2.0, 3.0], [0.0, 4.0, 0.0]])

    normalized, trace = pipeline.transform(source)
    restored = pipeline.inverse(normalized, trace)

    assert normalized.device == source.device
    assert len(trace.states) == 2
    assert torch.allclose(restored, source)


def test_packed_raw_normalization_round_trip() -> None:
    """Packed spectra use segmented reductions and one scale per sample."""
    pipeline = NormalizationPipeline.from_config(
        {"stage": "raw", "steps": {"tic": {"type": "tic"}}}
    )
    source = torch.tensor([1.0, 3.0, 2.0, 6.0])
    sample_indices = torch.tensor([0, 0, 1, 1])

    normalized, trace = pipeline.transform(
        source,
        sample_indices=sample_indices,
        batch_size=2,
    )
    restored = pipeline.inverse(
        normalized,
        trace,
        sample_indices=sample_indices,
    )

    assert torch.allclose(normalized, torch.tensor([0.25, 0.75, 0.25, 0.75]))
    assert torch.allclose(restored, source)


def test_pipeline_configuration_round_trip_preserves_policy() -> None:
    """Saved configuration retains order, stage, and reconstruction behavior."""
    source = {
        "stage": "binned",
        "steps": {"tic": {"type": "tic", "epsilon": 1e-9}},
        "reconstruction": {
            "output_space": "source",
            "denormalization_stage": "after_inverse_binning",
        },
    }
    pipeline = NormalizationPipeline.from_config(source)
    restored = NormalizationPipeline.from_config(pipeline.get_config())

    restored.set_denormalization("after_decode")
    restored.set_output_space("normalized")

    assert tuple(restored.steps) == ("tic",)
    assert restored.stage == "binned"
    assert restored.reconstruction.output_space == "normalized"
    assert restored.reconstruction.denormalization_stage == "after_decode"


def test_metric_requirements_reject_nonlinear_intensity_trace() -> None:
    """Metrics reject traces whose capabilities invalidate their semantics."""
    trace = NormalizationTrace(
        pipeline_name="nonlinear",
        stage="binned",
        capabilities=NormalizationCapabilities(
            invertible=True,
            preserves_nonnegativity=True,
            preserves_linear_intensity=False,
            samplewise_scalar=False,
            can_inverse_after_binning=True,
            can_inverse_after_inverse_binning=False,
        ),
    )
    requirements = MetricRequirements(
        requires_nonnegative=True,
        requires_linear_intensity=True,
    )

    with pytest.raises(Exception, match="linear intensity"):
        validate_metric_compatibility(requirements, trace)


@pytest.mark.parametrize(
    "split",
    [
        {"train": 0.7, "validation": 0.2, "test": 0.2},
        {"train": 0.7, "validation": 0.3},
        {"train": 0.0, "validation": 0.5, "test": 0.5},
    ],
)
def test_training_split_requires_complete_valid_proportions(split: dict[str, float]) -> None:
    """Training partitions must be complete and sum to one."""
    with pytest.raises(Exception):
        SplitConfig(fractions=split)


def test_training_split_is_deterministic_and_exhaustive() -> None:
    """Split rounding assigns every sample exactly once."""
    dataset = torch.utils.data.TensorDataset(torch.arange(11))
    split = {"train": 0.6, "validation": 0.2, "test": 0.2}

    config = {"strategy": "random", "fractions": split, "seed": 7}
    first = DatasetSplitter.split(dataset, config)
    second = DatasetSplitter.split(dataset, config)

    assert sum(len(partition) for _, partition in first.items()) == len(dataset)
    assert first.manifest.assignments == second.manifest.assignments


def test_context_normalization_can_be_replaced_updated_removed_and_saved(
    tmp_path: Path,
    msi_fixture_path: Path,
) -> None:
    """The context manager owns configuration while ActiveContext owns runtime use."""
    wrapper = MSIAutoEncoderWrapper(project_path=str(tmp_path))
    wrapper.context_manager.set_reader(
        MockMSIReader(msi_fixture_path),
        str(msi_fixture_path),
    )
    wrapper.workspace.set_active_image(str(msi_fixture_path))

    wrapper.context_manager.set_normalization(
        {
            "stage": "binned",
            "steps": {"tic": {"type": "tic"}},
            "reconstruction": {
                "denormalization_stage": "after_inverse_binning"
            },
        }
    )
    wrapper.context_manager.update_normalization(
        {"steps": {"peak": {"type": "max"}}}
    )

    assert tuple(wrapper.active_context.normalization.steps) == ("tic", "peak")
    assert tuple(
        wrapper.context_manager.get_context_config()["normalization"]["steps"]
    ) == ("tic", "peak")

    wrapper.context_manager.remove_normalization("tic")
    assert tuple(wrapper.active_context.normalization.steps) == ("peak",)
    wrapper.context_manager.clear_normalization()
    assert wrapper.active_context.normalization is None
