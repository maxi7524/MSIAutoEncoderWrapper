"""Trainer configuration builds state-aware training loaders."""

from __future__ import annotations

from types import SimpleNamespace

import torch

from msi_autoencoder_wrapper.data import TargetBatch
from msi_autoencoder_wrapper.data.supervision_masks import simulated_negative_mask_key
from msi_autoencoder_wrapper.data.supervision_sampling import SupervisionMaskBatchSampler
from msi_autoencoder_wrapper.training.engine.base_trainer import MSIPyTorchTrainer


class _MaskDataset(torch.utils.data.Dataset):
    """Minimal dense dataset exposing P/U/N_sim target masks."""

    targets = torch.tensor([[1.0], [1.0], [0.0], [0.0], [0.0], [0.0]])
    simulated_negative = torch.tensor([[False], [False], [True], [True], [False], [False]])

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        return (
            index,
            torch.tensor([float(index)]),
            {"molecule": self.targets[index]},
            {
                "molecule": torch.ones(1, dtype=torch.bool),
                simulated_negative_mask_key("molecule"): self.simulated_negative[index],
            },
        )

    def get_target_batch(self, indices):
        selected = list(indices)
        return TargetBatch(
            values={"molecule": self.targets[selected]},
            masks={
                "molecule": torch.ones(len(selected), 1, dtype=torch.bool),
                simulated_negative_mask_key("molecule"): self.simulated_negative[selected],
            },
            schemas={},
        )


def test_trainer_builds_supervision_mask_batch_sampler():
    """Phase-level sampling settings replace ordinary DataLoader shuffling."""
    trainer = MSIPyTorchTrainer(SimpleNamespace(models_manager=SimpleNamespace(batch_size=6)))
    loader = trainer._build_dataloader(
        dataset=_MaskDataset(),
        phase_config={
            "batch_size": 6,
            "dataloader": {"num_workers": 0},
            "supervision_sampling": {
                "target_field": "molecule",
                "proportions": {
                    "positive": 1,
                    "simulated_negative": 1,
                    "unlabelled": 1,
                },
                "steps_per_epoch": 1,
                "seed": 5,
            },
        },
        device="cpu",
    )
    assert isinstance(loader.batch_sampler, SupervisionMaskBatchSampler)
    batch = next(iter(loader))
    selected_targets = _MaskDataset.targets[batch[0]]
    selected_negative = _MaskDataset.simulated_negative[batch[0]]
    assert int((selected_targets > 0.5).sum()) == 2
    assert int(selected_negative.sum()) == 2
    assert int(((selected_targets == 0) & ~selected_negative).sum()) == 2


def test_subset_supervision_sampling_has_no_invalid_epoch_provider():
    """A static ``Subset`` must not install a callback returning ``None``."""
    trainer = MSIPyTorchTrainer(SimpleNamespace(models_manager=SimpleNamespace(batch_size=6)))
    dataset = torch.utils.data.Subset(_MaskDataset(), [0, 1, 2, 3, 4, 5])
    loader = trainer._build_dataloader(
        dataset=dataset,
        phase_config={
            "batch_size": 6,
            "dataloader": {"num_workers": 0},
            "supervision_sampling": {
                "target_field": "molecule",
                "proportions": {
                    "positive": 1,
                    "simulated_negative": 1,
                    "unlabelled": 1,
                },
                "steps_per_epoch": 1,
                "seed": 5,
            },
        },
        device="cpu",
    )

    sampler = loader.batch_sampler
    assert isinstance(sampler, SupervisionMaskBatchSampler)
    assert sampler.supervision_provider is None
    sampler.set_epoch(0)
    assert len(next(iter(sampler))) == 6
