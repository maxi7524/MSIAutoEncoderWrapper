"""Deterministic seed application for isolated experiment tasks."""

from __future__ import annotations

import random

import numpy as np
import torch


def set_execution_seed(seed: int) -> torch.Generator:
    """Seed Python, NumPy, Torch and return a DataLoader generator."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


def seed_worker(worker_id: int) -> None:
    """Seed NumPy and Python from the worker seed assigned by PyTorch."""
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)
