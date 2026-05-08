import pytest
import torch
import numpy as np
from unittest.mock import MagicMock

@pytest.fixture
def base_dims():
    """Standard dimensions for MSI data."""
    return {"input_dim": 1024, "latent_dim": 32, "batch_size": 4}

@pytest.fixture
def architecture_params(base_dims):
    """Hyperparameters for CNN architectures."""
    return {
        "input_dim": base_dims["input_dim"],
        "latent_dim": base_dims["latent_dim"],
        "channels": [1, 16, 32],
        "kernels": [3, 3],
        "strides": [2, 2]
    }

@pytest.fixture
def dummy_input(base_dims):
    """Standard batch input."""
    return torch.randn(base_dims["batch_size"], base_dims["input_dim"])



@pytest.fixture
def mock_msi_dataset(base_dims):
    """
    Robust mock of MSIPyTorchDataset.
    Satisfies deep attribute access like dataset.img.normalization 
    and unpacking in estimate_max_peak_width.
    """
    dataset = MagicMock()
    
    # 1. Mock internal 'img' object with metadata
    dataset.img = MagicMock()
    dataset.img.normalization = "TIC"
    dataset.img.GetXAxis.return_value = np.linspace(100, 1000, base_dims["input_dim"])
    dataset.img.GetXAxisDepth.return_value = base_dims["input_dim"]

    # 2. Mock grid/binner methods used in SetHyperparameters
    dataset.GetGridXAxis.return_value = np.linspace(100, 1000, base_dims["input_dim"])
    dataset.GetGridXAxisDepth.return_value = base_dims["input_dim"]
    dataset.__len__.return_value = 100

    # 3. Fix unpacking error: MSILoader[idx] must return (idx, spectrum)
    # We use side_effect to ensure each call returns a valid tuple
    dataset.__getitem__.side_effect = lambda i: (i, torch.abs(torch.randn(base_dims["input_dim"])))

    # 4. Mock PeakBank for ContrastiveCriterion
    # Each entry: (start, end, values)
    dataset.peak_bank = [(10, 20, torch.randn(10)) for _ in range(50)]

    return dataset