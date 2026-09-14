"""Configurable sources of reliable simulated-negative supervision."""

from .base import SimulatedNegativeStrategy
from .manager import SimulatedNegativeManager

__all__ = ["SimulatedNegativeManager", "SimulatedNegativeStrategy"]
