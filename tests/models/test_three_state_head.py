"""Three-state component dimensions, serialization, and encoder gradients."""

import torch

from msi_autoencoder_wrapper.models.architectures.types.autoencoders.heads.three_state_head import ThreeStateHead


def test_three_state_head_roundtrip_preserves_ion_count_and_gradients():
    torch.manual_seed(42)
    head = ThreeStateHead(latent_dim=4, output_dim=2, hidden_dim=5)
    restored = ThreeStateHead(**head._config)
    restored.load_state_dict(head.state_dict())
    latent = torch.tensor([[1., 0., -1., 2.]], requires_grad=True)
    logits = head(latent)
    assert logits.shape == (1, 2, 3)
    torch.testing.assert_close(restored(latent), logits, rtol=0, atol=0)
    probabilities = head.presence_probability(logits)
    assert probabilities.shape == (1, 2)
    probabilities.sum().backward()
    assert bool(torch.isfinite(latent.grad).all())
    assert bool(latent.grad.abs().sum() > 0)
