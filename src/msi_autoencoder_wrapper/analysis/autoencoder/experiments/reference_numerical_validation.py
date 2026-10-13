"""Executable CPU/CUDA and analytical checks for reused model computations."""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from scipy.stats import wasserstein_distance

from ....metrics import SpectrumMasserstein
from ....metrics.strategies.embedding import info_nce
from ....utils.logger import get_custom_logger
from ..latent.diagnostics import gradient_diagnostics

logger = get_custom_logger(__name__)


def validate_numerics(*, device: str, seed: int, atol: float = 1e-5, rtol: float = 1e-4) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Check analytical transport, finite gradients and CPU/device agreement.

    Float32 error limits allow accumulation over 32 spectral bins and 16 contrastive
    pairs, relative to independent float64 references. They are validation settings,
    not relaxed after observing an error. Float64 gradcheck uses eps=1e-6,
    atol=1e-5, rtol=1e-3 and avoids zero cumulative transport differences.

    :param device: Required Torch device; unavailable CUDA raises.
    :param seed: Fixed RNG seed for generated admissible inputs.
    :param atol: Absolute float32 comparison tolerance.
    :param rtol: Relative float32 comparison tolerance.
    :return: Check records and named device gradient snapshots.
    :raises RuntimeError: If CUDA was required but unavailable.
    :raises AssertionError: If an executable numerical contract fails.
    """
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("Configured CUDA device is unavailable; explicit CPU opt-out is required.")
    rng = np.random.default_rng(seed)
    axis = np.arange(32, dtype=float)
    x, y = rng.uniform(.1, 1, (8, 32)), rng.uniform(.1, 1, (8, 32))  # (B, M), (B, M)
    oracle = np.asarray([wasserstein_distance(axis, axis, u_weights=a, v_weights=b) for a, b in zip(x, y, strict=True)])  # (B,)
    records = []
    snapshots = {}
    for backend in ["cpu", device] if device != "cpu" else ["cpu"]:
        prediction = torch.tensor(y, dtype=torch.float32, device=backend, requires_grad=True)  # (B, M)
        target = torch.tensor(x, dtype=torch.float32, device=backend)  # (B, M)
        metric = SpectrumMasserstein(reduction="none").to(backend)
        values = metric(prediction, target, mass_axis=torch.tensor(axis, device=backend))  # (B,)
        np.testing.assert_allclose(values.detach().cpu().numpy(), oracle, atol=atol, rtol=rtol)
        values.sum().backward()
        assert prediction.grad is not None and torch.isfinite(prediction.grad).all()
        records.append({"check": "masserstein_independent_oracle", "device": backend, "max_absolute_error": float(np.max(np.abs(values.detach().cpu().numpy()-oracle))), "atol": atol, "rtol": rtol, "passed": True})
        snapshots[f"masserstein_{backend}"] = prediction.grad
    # Numerical Jacobian check is a verification fixture, not a model training step
    pred64 = torch.tensor(y[:2], dtype=torch.float64, requires_grad=True)  # (2, M)
    target64 = torch.tensor(x[:2], dtype=torch.float64)  # (2, M)
    metric64 = SpectrumMasserstein(reduction="sum")
    checked = torch.autograd.gradcheck(lambda p: metric64(p, target64), (pred64,), eps=1e-6, atol=1e-5, rtol=1e-3)
    assert checked
    records.append({"check": "masserstein_float64_gradcheck", "device": "cpu", "max_absolute_error": None, "atol": 1e-5, "rtol": 1e-3, "passed": True})
    # Paired contrastive gradient comparison with identical saved inputs
    a, b = rng.normal(size=(16, 10)), rng.normal(size=(16, 10))  # (B, D), (B, D)
    reference_loss = reference_grad = None
    for backend in ["cpu", device] if device != "cpu" else ["cpu"]:
        first = torch.tensor(a, dtype=torch.float32, device=backend, requires_grad=True)  # (B, D)
        second = torch.tensor(b, dtype=torch.float32, device=backend)  # (B, D)
        loss = info_nce(first, second, temperature=.07).mean()  # ()
        loss.backward()
        assert torch.isfinite(loss) and first.grad is not None and torch.isfinite(first.grad).all()
        actual_loss, actual_grad = float(loss.detach().cpu()), first.grad.detach().cpu().numpy()
        if reference_loss is None:
            reference_loss, reference_grad = actual_loss, actual_grad
        np.testing.assert_allclose(actual_loss, reference_loss, atol=atol, rtol=rtol)
        np.testing.assert_allclose(actual_grad, reference_grad, atol=atol, rtol=rtol)
        records.append({"check": "infonce_cpu_device_loss_and_gradient", "device": backend, "max_absolute_error": float(np.max(np.abs(actual_grad-reference_grad))), "atol": atol, "rtol": rtol, "passed": True})
        snapshots[f"infonce_{backend}"] = first.grad
    logger.info("Completed %s numerical checks on %s.", len(records), device)
    return pd.DataFrame(records), gradient_diagnostics(snapshots)
