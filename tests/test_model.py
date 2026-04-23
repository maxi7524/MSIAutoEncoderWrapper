import pytest
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from unittest.mock import MagicMock
from ims_contrastive_model import IMSContrastiveModel
from ims_contrastive_model import IMSPyTorchDataset

@pytest.fixture
def mock_ims_loader():
    """Creates a mock IMSPyTorchDataset with a defined m/z grid."""
    loader = MagicMock(spec=IMSPyTorchDataset)
    # Define a grid of 1000 m/z bins
    loader.grid = np.linspace(100, 1000, 1000)
    # Mock __len__ and __getitem__ for basic iteration
    loader.__len__.return_value = 100
    loader.__getitem__.return_value = torch.randn(1000)
    return loader

@pytest.fixture
def model_instance(mock_ims_loader):
    """Initializes a model instance for testing."""
    return IMSContrastiveModel(
        IMSLoader=mock_ims_loader,
        latent_dim=32,
        epochs=1,
        batch_size=16
    )

def test_model_initialization(model_instance, mock_ims_loader):
    """Verifies if the model initializes with correct parameters."""
    assert model_instance._latent_dim == 32
    assert model_instance.model is not None
    # Check if input_dim in hyperparameters matches the loader grid
    assert model_instance._hyperparameters['input_dim'] == len(mock_ims_loader.grid)

def test_encode_decode_flow(model_instance):
    """Verifies the spectral encoding and decoding pipeline."""
    batch_size = 5
    dummy_input = torch.randn(batch_size, 1000)
    
    # Test encoding
    latent = model_instance.encode(dummy_input)
    assert latent.shape == (batch_size, 32)
    # Check L2 normalization (norm should be 1.0)
    norms = np.linalg.norm(latent, axis=1)
    assert np.allclose(norms, np.ones(batch_size), atol=1e-5)
    
    # Test decoding
    reconstructed = model_instance.decode(torch.from_numpy(latent))
    assert reconstructed.shape == (batch_size, 1000)

def test_save_and_load(model_instance, tmp_path):
    """Tests if saving and loading preserves model weights and config."""
    save_dir = tmp_path / "test_model"
    
    # 1. Save the model
    model_instance.save(save_dir)
    assert (save_dir / "model_weights.pt").exists()
    assert (save_dir / "config.json").exists()
    
    # 2. Create a new instance and load data
    new_model = IMSContrastiveModel(
        IMSLoader=model_instance.IMSLoader,
        latent_dim=32
    )
    new_model.load(save_dir)
    
    # 3. Verify parameters match
    for p1, p2 in zip(model_instance.model.parameters(), new_model.model.parameters()):
        assert torch.equal(p1, p2)
    
    assert new_model._latent_dim == model_instance._latent_dim

def test_fit_integration(model_instance, tmp_path, mock_ims_loader):
    """Verifies if the training process (fit) runs and updates history."""
    # We mock the data to be a list of tensors for the DataLoader
    data = [torch.randn(1000) for _ in range(32)]
    
    # Wrap the loader in a simple PyTorch DataLoader for the internal fit call
    # In actual use, fit() handles this, but here we check if history is updated
    model_instance.fit(save_dir=tmp_path)
    
    # Check if training history was recorded
    assert hasattr(model_instance, 'history')
    assert len(model_instance.history) > 0
    assert "total_loss" in model_instance.history[0]

def test_transform_output(model_instance):
    """Verifies that transform() processes the entire dataset."""
    # The mock loader has 100 spectra (len=100)
    latent_map = model_instance.transform()
    
    assert isinstance(latent_map, np.ndarray)
    assert latent_map.shape == (100, 32)



@pytest.mark.parametrize("input_type", ["numpy", "torch"])
def test_model_data_type_flexibility(model_instance, input_type):
    """
    Test if encode and decode methods correctly handle both 
    numpy arrays and torch tensors.
    """
    batch_size = 4
    input_dim = model_instance._hyperparameters['input_dim']
    latent_dim = model_instance._latent_dim
    
    # 1. Prepare raw data
    raw_data_np = np.random.rand(batch_size, input_dim).astype(np.float32)
    
    if input_type == "torch":
        data_to_test = torch.from_numpy(raw_data_np)
    else:
        data_to_test = raw_data_np

    # 2. Test ENCODE
    # Should work regardless of input_type and return numpy
    latent_space = model_instance.encode(data_to_test)
    
    assert isinstance(latent_space, np.ndarray), f"Encode should return numpy even if input is {input_type}"
    assert latent_space.shape == (batch_size, latent_dim)

    # 3. Test DECODE
    # Prepare latent data (again as numpy or torch)
    if input_type == "torch":
        latent_to_test = torch.from_numpy(latent_space)
    else:
        latent_to_test = latent_space
        
    reconstructed = model_instance.decode(latent_to_test)
    
    assert isinstance(reconstructed, np.ndarray), f"Decode should return numpy even if input is {input_type}"
    assert reconstructed.shape == (batch_size, input_dim)

def test_model_device_consistency_during_io(model_instance):
    """
    Verify if _prepare_input correctly moves data to the model's device.
    """
    # Create data on CPU (typical for numpy/standard torch tensors)
    data = np.random.rand(2, model_instance._hyperparameters['input_dim']).astype(np.float32)
    
    # Access the internal helper to see if it moves data to self._device
    # Even if self._device is 'cpu', the result must be a torch.Tensor
    tensor_ready = model_instance._prepare_input(data)
    
    assert isinstance(tensor_ready, torch.Tensor)
    assert tensor_ready.device.type == model_instance._device.type
    assert tensor_ready.dtype == torch.float32