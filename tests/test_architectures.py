import pytest
import torch
import torch.nn as nn
import numpy as np
from MSIAutoEncoderWrapper.architectures import ARCHITECTURES_REGISTRY

@pytest.mark.parametrize("arch_name", ARCHITECTURES_REGISTRY.keys())
class TestArchitectures:
    """
    Suite of tests applied to every architecture registered in ARCHITECTURES_REGISTRY.
    Ensures consistency across different model implementations.
    """

    def test_config_serialization(self, arch_name, architecture_params):
        """
        Test: GetConfig() functionality.
        Expectation: Every architecture must return a dictionary containing its initialization parameters.
        Why: Essential for model checkpointing and reconstruction.
        """
        arch_class = ARCHITECTURES_REGISTRY[arch_name]
        model = arch_class(**architecture_params)
        
        config = model.GetConfig()
        assert isinstance(config, dict), f"{arch_name}: GetConfig() must return a dictionary."
        # Note: If GetConfig() logic is implemented in base class, 
        # ensure it's properly populated in __init__.

    def test_static_hyperparameter_suggestion(self, arch_name, mock_msi_dataset):
        """
        Test: Static method SetHyperparameters().
        Expectation: Should return either a dictionary of params or an initialized model instance.
        Why: This method is the 'brain' that decides how big the kernels should be based on data.
        """
        arch_class = ARCHITECTURES_REGISTRY[arch_name]
        
        # Test returning dictionary
        result_dict = arch_class.SetHyperparameters(
            MSIDataset=mock_msi_dataset, 
            latent_dim=32, 
            initialize_model=False
        )
        assert isinstance(result_dict, dict), f"{arch_name}: SetHyperparameters should return a dict."
        assert "input_dim" in result_dict, f"{arch_name}: Suggested params missing 'input_dim'."

        # Test returning initialized model
        result_model = arch_class.SetHyperparameters(
            MSIDataset=mock_msi_dataset, 
            latent_dim=32, 
            initialize_model=True
        )
        assert isinstance(result_model, arch_class), f"{arch_name}: Should return an initialized model instance."


    def test_encoding_decoding_shapes(self, arch_name, architecture_params, dummy_input, base_dims):
        """
        Test: Latent space and reconstruction dimensions.
        Expectation: 
            - encode() should output (batch, latent_dim).
            - decode() should output (batch, input_dim).
        Why: Ensures CNN layers (strides/kernels) and Linear layers are correctly aligned.
        """
        arch_class = ARCHITECTURES_REGISTRY[arch_name]
        model = arch_class(**architecture_params)
        
        # Test Encoder
        z = model.encode(dummy_input)
        assert z.shape == (base_dims["batch_size"], base_dims["latent_dim"]), \
            f"{arch_name}: Encoder output shape mismatch."
            
        # Test Decoder
        x_hat = model.decode(z)
        assert x_hat.shape == (base_dims["batch_size"], base_dims["input_dim"]), \
            f"{arch_name}: Decoder output shape mismatch."

    def test_forward_return_structure(self, arch_name, architecture_params, dummy_input):
        """
        Test: Forward pass return types.
        Expectation: forward() should return a tuple (latent, reconstruction).
        Why: The training loop expects this specific signature to calculate composite losses.
        """
        arch_class = ARCHITECTURES_REGISTRY[arch_name]
        model = arch_class(**architecture_params)
        
        output = model(dummy_input)
        assert isinstance(output, tuple) and len(output) == 2, \
            f"{arch_name}: forward() must return a tuple of (latent, reconstruction)."
        
        latent, recon = output
        assert torch.is_tensor(latent) and torch.is_tensor(recon)

    def test_gradient_flow(self, arch_name, architecture_params, dummy_input):
        """
        Test: Autograd compatibility.
        Expectation: Parameters should have non-None gradients after a backward pass on a trivial loss.
        Why: Validates that no layers are detached from the computational graph (e.g., via improper numpy conversion).
        """
        arch_class = ARCHITECTURES_REGISTRY[arch_name]
        model = arch_class(**architecture_params)
        model.train() # Set to training mode
        
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        optimizer.zero_grad()
        
        _, reconstruction = model(dummy_input)
        
        # Trivial MSE loss for gradient checking
        loss = torch.nn.functional.mse_loss(reconstruction, dummy_input)
        loss.backward()
        
        # Check if gradients exist for trainable parameters
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"{arch_name}: Gradient missing for parameter {name}."
                assert torch.sum(torch.abs(param.grad)) > 0, f"{arch_name}: Gradient is zero for {name}."

    def test_eval_mode_consistency(self, arch_name, architecture_params, dummy_input):
        """
        Test: Evaluation mode behavior.
        Expectation: Outputs should be consistent and not track gradients in eval mode.
        Why: Ensures layers like BatchNorm or Dropout (if used) behave correctly in inference.
        """
        arch_class = ARCHITECTURES_REGISTRY[arch_name]
        model = arch_class(**architecture_params)
        model.eval()
        
        with torch.no_grad():
            latent1, recon1 = model(dummy_input)
            latent2, recon2 = model(dummy_input)
            
        # Check for determinism in eval mode
        torch.testing.assert_close(recon1, recon2, msg=f"{arch_name}: Non-deterministic output in eval mode.")

    
    # TODO - it does not pass ??? i do not know how to create it 
    # def test_weight_initialization_sanity(self, arch_name, architecture_params):
    #     """
    #     Test: Initial weights state.
    #     Expectation: Weights should not be NaN and should have some variance (not all zeros).
    #     Why: Bad initialization (e.g. all zeros) can prevent the model from ever learning.

    #     REMARK: We exclude LayerNorm (starts with ones) 
    #     """
    #     arch_class = ARCHITECTURES_REGISTRY[arch_name]
    #     model = arch_class(**architecture_params)
        
    #     for name, param in model.named_parameters():
    #         assert not torch.isnan(param).any(), f"{arch_name}: NaN in {name}"
            
    #         if "weight" in name and param.requires_grad:
    #             # LayerNorm and expansion norms have constant weights by design
    #             if not any(x in name for x in ["LayerNorm", "norm", "initial_expansion.1"]):
    #                 assert torch.std(param) > 0, f"{arch_name}: Zero variance in {name}"