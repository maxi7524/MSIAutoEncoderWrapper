import pytest
import torch
import numpy as np
from ims_contrastive_model.architecture import ContrastiveAutoencoder

@pytest.fixture
def model_params():
    """Defines sample model parameters for testing."""
    return {
        'input_dim': 1000,
        'latent_dim': 32,
        'channels': [1, 4, 8],
        'kernels': [7, 5],
        'strides': [2, 2]
    }

def test_forward_shapes(model_params):
    """Verifies if the output dimensions are correct."""
    model = ContrastiveAutoencoder(**model_params)
    batch_size = 4
    x = torch.randn(batch_size, model_params['input_dim'])
    
    z, x_hat = model(x)
    
    # Check latent space (p=2 normalization should result in a unit norm)
    assert z.shape == (batch_size, model_params['latent_dim'])
    assert torch.allclose(torch.norm(z, p=2, dim=1), torch.ones(batch_size), atol=1e-5)
    
    # Check reconstruction shape
    assert x_hat.shape == (batch_size, model_params['input_dim'])

def test_model_trainability(model_params):
    """Verifies if model weights are actually updating (no dead gradients)."""
    model = ContrastiveAutoencoder(**model_params)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    x = torch.randn(8, model_params['input_dim'])
    
    # Save weights before training
    initial_params = [p.clone() for p in model.parameters()]
    
    # Perform a single training step (dummy loss)
    z, x_hat = model(x)
    loss = torch.mean(z**2) + torch.mean((x - x_hat)**2)
    loss.backward()
    optimizer.step()
    
    # Verify that at least some parameters have changed
    any_changed = any(not torch.equal(p1, p2) for p1, p2 in zip(initial_params, model.parameters()))
    assert any_changed, "Model parameters did not change after the optimization step!"

def test_no_nan_outputs(model_params):
    """Checks robustness against extreme input values."""
    model = ContrastiveAutoencoder(**model_params)
    x = torch.zeros(4, model_params['input_dim']) # Test with all zeros
    
    z, x_hat = model(x)
    
    assert not torch.isnan(z).any(), "Latent space contains NaNs!"
    assert not torch.isnan(x_hat).any(), "Reconstruction contains NaNs!"