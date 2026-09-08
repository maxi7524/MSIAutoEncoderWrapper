"""Integration of synthetic pretraining and real-data adaptation in one model."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import Subset

from msi_autoencoder_wrapper.data import SpectrumBatch, SpectrumSpace, TargetBatch, TargetSchema
from msi_autoencoder_wrapper.models.datasets.annotations.index import MappedSpectrumAnnotationIndex
from msi_autoencoder_wrapper.models.datasets.splitting.partitions import DatasetPartitions, SplitManifest
from msi_autoencoder_wrapper.training.criterions.criterions_manager import CriterionsManager
from msi_autoencoder_wrapper.training.engine.base_trainer import MSIPyTorchTrainer


class PhaseDataset:
    """An analytical real-data stand-in with disjoint, immutable partitions."""

    source, dtype = "image", torch.float32

    def __init__(self):
        self.active_context = SimpleNamespace(binner=SimpleNamespace(GetXAxis=lambda: np.arange(10.) + 100))
        self.partition_calls = 0
        self.samples_read = []

    def __len__(self):
        return 8

    def __getitem__(self, index):
        self.samples_read.append(index)
        spectrum = torch.zeros(10)
        spectrum[1 if index % 2 == 0 else 8] = 1
        return index, spectrum, {"molecule": torch.tensor([float(index % 2 == 0), float(index % 2 == 1)])}, {"molecule": torch.ones(2, dtype=torch.bool)}

    def create_partitions(self):
        return DatasetPartitions(Subset(self, [0, 1, 2, 3]), Subset(self, [4, 5]), Subset(self, [6, 7]),
                                 SplitManifest("predefined", 42, {"train": (0, 1, 2, 3), "validation": (4, 5), "test": (6, 7)}))

    def get_target_schemas(self):
        return {"molecule": TargetSchema("molecule", "multi_label", ("C2H4|+H", "C3H6|+H"))}

    def collate_fn(self, samples):
        return SpectrumBatch(
            sample_ids=torch.tensor([s[0] for s in samples]),
            spectra=torch.stack([s[1] for s in samples]),
            space=SpectrumSpace(torch.arange(10.) + 100, normalization="tic"),
            targets=TargetBatch(
                {"molecule": torch.stack([s[2]["molecule"] for s in samples])},
                {"molecule": torch.stack([s[3]["molecule"] for s in samples])},
                self.get_target_schemas(),
            ),
        )

    def get_target_batch(self, indices):
        return TargetBatch({"molecule": torch.tensor([[float(i % 2 == 0), float(i % 2 == 1)] for i in indices])},
                           {"molecule": torch.ones(len(indices), 2, dtype=torch.bool)}, self.get_target_schemas())

    def get_mapped_annotation_index(self):
        return MappedSpectrumAnnotationIndex(np.array([0, 1]), np.array([0, 1, 2]), np.array([0, 1]),
                                             np.array([1, 8]), (("C2H4", "+H"), ("C3H6", "+H")), np.arange(10.) + 100, "binner")


class PhaseModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Linear(10, 3)
        self.decoder = nn.Sequential(nn.Linear(3, 10), nn.Softplus())
        self.heads = nn.ModuleDict({"ion": nn.Linear(3, 2)})
        self.head_specs = {"ion": {"target_field": "molecule"}}
        self.seen = []

    def forward(self, x):
        self.seen.append(x.detach().clone())
        latent = self.encoder(x)
        return {"latent_space": latent, "projection": latent, "reconstruction": self.decoder(latent), "head_ion": self.heads["ion"](latent)}


@pytest.mark.parametrize("contrastive", [False, True])
def test_pretraining_then_real_adaptation_preserves_model_and_split(contrastive):
    torch.manual_seed(42)
    dataset, model = PhaseDataset(), PhaseModel()
    wrapper = SimpleNamespace(active_context=SimpleNamespace(reader=object(), _instantiated_image_key="fixture"),
                              active_model=model, active_dataset=dataset, device="cpu",
                              models_manager=SimpleNamespace(active_model_type="autoencoder"),
                              workspace=SimpleNamespace(active_img_name="fixture"))
    CriterionsManager.discover_criterions()
    common = {"epochs": 1, "batch_size": 4, "dataloader": {"shuffle": False},
              "optimizer": {"type": "SGD", "params": {"lr": .01}}}
    config = {"seed": 42, "test_mode": True, "checkpoint": {"enabled": False}, "phases": [
        {**common, "phase_name": "synthetic", "pretraining": {"samples": 8, "validation_samples": 4, "modes": ["single_random"]},
         "criterions": {"reconstruction": {"mse": {"target": "MSELoss"}}, "heads": {"ion": {"bce": {"target": "MultiLabelBCELoss"}}}}},
        {**common, "phase_name": "real", "criterions": {"reconstruction": {"mse": {"target": "MSELoss"}},
         "heads": {"ion": {"vpu": {"target": "VariationalPULoss", "params": {"consistency_weight": .1, "negative_weight": 1., "evidence": {"bin_radius": 0}}}}}}},
    ]}
    if contrastive:
        config["phases"][1]["criterions"]["contrastive"] = {"infonce": {
            "target": "InfoNCELoss", "weight": .1, "params": {
                "peak_selection_method": "permutation_label_invariant", "peak_sample_size": 4,
                "permutation_bank_size": 4, "permuted_peaks_per_view": 2,
            },
        }}
    initial = model.encoder.weight.detach().clone()
    history = MSIPyTorchTrainer(wrapper).fit(config)
    assert {entry["phase"] for entry in history} == {"synthetic", "real"}
    assert history[-1]["split"] == "test"
    if contrastive:
        assert sorted(dataset.samples_read[:4]) == [0, 1, 2, 3]  # Train-only contrastive peak catalogue.
        assert dataset.samples_read[4:] == [0, 1, 2, 3, 4, 5, 6, 7]
    else:
        assert dataset.samples_read == [0, 1, 2, 3, 4, 5, 6, 7]
    assert not torch.equal(initial, model.encoder.weight)
    assert wrapper.active_model is model and wrapper.active_dataset is dataset
    assert model.seen[0].shape == (4, 10)
    assert not bool(model.seen[0][:, [1, 8]].any())
