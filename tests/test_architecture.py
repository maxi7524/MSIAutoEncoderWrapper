import pytest
import torch
import numpy as np
from IMSAutoEncoderWrapper.architecture import ContrastiveAutoencoder, ContrastiveLoss, ReshapeLayer

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

@pytest.fixture
def device():
    """Returns the available device (cpu)."""
    return torch.device("cpu")

def test_reshape_layer():
    """Tests if ReshapeLayer correctly changes tensor dimensions."""
    layer = ReshapeLayer([8, 10])
    x = torch.randn(4, 80)
    output = layer(x)
    assert output.shape == (4, 8, 10)

def test_forward_shapes(model_params):
    """Verifies if the model output dimensions are correct."""
    model = ContrastiveAutoencoder(**model_params)
    batch_size = 4
    x = torch.randn(batch_size, model_params['input_dim'])
    
    z, x_hat = model(x)
    
    # Check latent space (p=2 normalization should result in a unit norm)
    assert z.shape == (batch_size, model_params['latent_dim'])
    norms = torch.norm(z, p=2, dim=1)
    assert torch.allclose(norms, torch.ones(batch_size), atol=1e-5)
    
    # Check reconstruction shape
    assert x_hat.shape == (batch_size, model_params['input_dim'])

def test_contrastive_loss_logic(device):
    """Tests if the loss function returns the correct keys and values."""
    criterion = ContrastiveLoss(device=device, temperature=0.5)
    
    batch_size = 8
    latent_dim = 32
    input_dim = 100
    
    # Simulate two views (augmentations) of the same batch
    emb_i = torch.randn(batch_size, latent_dim)
    emb_j = emb_i + torch.randn(batch_size, latent_dim) * 0.1 # close to each other
    emb_i = torch.nn.functional.normalize(emb_i, p=2, dim=1)
    emb_j = torch.nn.functional.normalize(emb_j, p=2, dim=1)
    
    inputs = torch.randn(batch_size, input_dim)
    outputs = inputs + torch.randn(batch_size, input_dim) * 0.1
    
    loss, loss_dict = criterion(emb_i, emb_j, inputs, outputs)
    
    assert isinstance(loss, torch.Tensor)
    assert "contrastive_loss" in loss_dict
    assert "std_loss" in loss_dict
    assert "decoder_loss" in loss_dict
    assert loss_dict["total_loss"] > 0

def test_model_trainability(model_params):
    """Verifies if model weights are actually updating (no dead gradients)."""
    model = ContrastiveAutoencoder(**model_params)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    x = torch.randn(8, model_params['input_dim'])
    
    initial_params = [p.clone() for p in model.parameters() if p.requires_grad]
    
    # Single training step
    z, x_hat = model(x)
    loss = torch.mean(z**2) + torch.mean((x - x_hat)**2)
    loss.backward()
    optimizer.step()
    
    # Verify that at least some parameters changed
    any_changed = any(not torch.equal(p1, p2) for p1, p2 in zip(initial_params, 
                  [p for p in model.parameters() if p.requires_grad]))
    assert any_changed, "Model parameters did not change after the optimization step!"

@pytest.mark.parametrize("input_size", [500, 1000, 2048])
def test_different_input_sizes(input_size, model_params):
    """Tests model stability for different spectral lengths."""
    params = model_params.copy()
    params['input_dim'] = input_size
    model = ContrastiveAutoencoder(**params)
    
    x = torch.randn(2, input_size)
    z, x_hat = model(x)
    
    assert z.shape == (2, params['latent_dim'])
    assert x_hat.shape == (2, input_size)

def test_no_nan_outputs(model_params):
    """Checks robustness against extreme input values (zeros)."""
    model = ContrastiveAutoencoder(**model_params)
    x = torch.zeros(4, model_params['input_dim'])
    
    z, x_hat = model(x)
    
    assert not torch.isnan(z).any(), "Latent space contains NaNs!"
    assert not torch.isnan(x_hat).any(), "Reconstruction contains NaNs!"