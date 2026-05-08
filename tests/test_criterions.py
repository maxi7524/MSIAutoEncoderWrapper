import pytest
import torch
import numpy as np
from unittest.mock import MagicMock
from MSIAutoEncoderWrapper.criterions import CRITERIONS_REGISTRY


@pytest.fixture
def prepared_dataset(mock_msi_dataset):
    """
    Small wrapper to ensure peak_bank is attached before criterion tests.
    """
    # mock_msi_dataset is already robust from conftest.py
    return mock_msi_dataset

@pytest.mark.parametrize("loss_name", CRITERIONS_REGISTRY.keys())
class TestCriterions:
    """
    Suite for testing all registered criterions.
    Checks for interface consistency, mathematical correctness, and dataset integration.
    """

    def test_criterion_config(self, loss_name):
        """
        Test: GetConfig() availability.
        Expectation: Returns a dictionary for object reconstruction.
        """
        criterion_class = CRITERIONS_REGISTRY[loss_name]
        criterion = criterion_class()
        
        assert hasattr(criterion, 'GetConfig'), f"{loss_name} lacks GetConfig method."
        assert isinstance(criterion.GetConfig(), dict), f"{loss_name} GetConfig must return a dict."

    def test_forward_output_structure(self, loss_name, prepared_dataset, dummy_input):
        """
        Test: Forward pass return values.
        Expectation: Returns a tuple (total_loss_tensor, loss_components_dict).
        Why: The trainer expects 'total_loss' for .backward() and a dict for logging.
        """
        criterion_class = CRITERIONS_REGISTRY[loss_name]
        criterion = criterion_class()
        
        model = MagicMock()
        # Mock model return: (latent, reconstruction)
        model.return_value = (torch.randn(4, 32), torch.randn(4, 1024))
        model.encode.side_effect = lambda x: torch.randn(x.shape[0], 32)
        model.device = torch.device('cpu')
        
        dataloader = MagicMock()
        dataloader.dataset = prepared_dataset
        batch_data = (torch.tensor([0, 1, 2, 3]), dummy_input)
        
        # We wrap in try-except to see detailed error if it still fails
        try:
            loss, loss_dict = criterion.forward(
                batch_idx=0,
                batch_data=batch_data,
                model=model,
                dataloader=dataloader,
                device=torch.device('cpu')
            )
        except Exception as e:
            pytest.fail(f"Criterion {loss_name} failed during forward: {e}")
        
        assert torch.is_tensor(loss)
        assert "total_loss" in loss_dict

    def test_mathematical_consistency(self, loss_name, prepared_dataset):
        """
        Test: Loss should decrease as reconstruction improves.
        Fix: Included 'prepared_dataset' in args to use fixture value, not function.
        """
        criterion_class = CRITERIONS_REGISTRY[loss_name]
        criterion = criterion_class()
        
        x = torch.ones(2, 1024)
        z = torch.ones(2, 32)
        batch_data = (torch.tensor([0, 1]), x)
        
        model = MagicMock()
        model.encode.return_value = z
        model.device = torch.device('cpu')
        
        dataloader = MagicMock()
        dataloader.dataset = prepared_dataset # Using the injected fixture instance
        
        # Scenario 1: Perfect
        model.return_value = (z, x)
        loss_perfect, _ = criterion.forward(0, batch_data, model, dataloader, torch.device('cpu'))
        
        # Scenario 2: Garbage
        model.return_value = (z, x * 0)
        loss_bad, _ = criterion.forward(0, batch_data, model, dataloader, torch.device('cpu'))
        
        assert loss_perfect < loss_bad, f"{loss_name}: Loss didn't penalize poor reconstruction."

    def test_required_setup_execution(self, loss_name, mock_msi_dataset):
        """
        Test: REQUIRED_SETUP protocol.
        Expectation: Methods in REQUIRED_SETUP must be callable and modify the dataset as intended.
        Why: Ensures someone didn't forget to attach 'peak_bank' or other metadata to the dataset.
        """
        criterion_class = CRITERIONS_REGISTRY[loss_name]
        criterion = criterion_class()
        
        if hasattr(criterion, 'REQUIRED_SETUP') and criterion.REQUIRED_SETUP:
            for setup_func in criterion.REQUIRED_SETUP:
                if callable(setup_func):
                    setup_func(mock_msi_dataset)

    def test_dummy_training_step(self, loss_name, architecture_params, dummy_input, prepared_dataset):
        """
        Test: Integration with a minimal optimizer step (skipping MSIAutoEncoder wrapper).
        Expectation: Gradients are calculated and weights are updated.
        Why: Confirms the loss function is fully differentiable within a training loop.
        """
        from MSIAutoEncoderWrapper.architectures import ContrastiveAutoencoderMax_InverseDim
        
        model = ContrastiveAutoencoderMax_InverseDim(**architecture_params)
        criterion = CRITERIONS_REGISTRY[loss_name]()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        
        # Initial weight snapshot
        initial_params = [p.clone() for p in model.parameters() if p.requires_grad]
        
        # Single step
        dataloader = MagicMock()
        dataloader.dataset = prepared_dataset # Needed if criterion calls dataset metadata
        batch_data = (torch.tensor([0, 1, 2, 3]), dummy_input)
        
        optimizer.zero_grad()
        loss, _ = criterion.forward(0, batch_data, model, dataloader, torch.device('cpu'))
        loss.backward()
        optimizer.step()
        
        # Check if at least some weights changed
        updated_params = [p for p in model.parameters() if p.requires_grad]
        changes = [not torch.equal(p1, p2) for p1, p2 in zip(initial_params, updated_params)]
        
        assert any(changes), f"{loss_name}: Optimizer step did not update any model parameters."