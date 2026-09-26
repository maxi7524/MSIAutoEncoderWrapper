"""Candidate dictionaries and deterministic synthetic deconvolution data."""

from .dictionary import GlobalCandidateDictionary
from .synthetic import SyntheticDeconvolutionConfig, SyntheticDeconvolutionGenerator

__all__ = [
    "GlobalCandidateDictionary",
    "SyntheticDeconvolutionConfig",
    "SyntheticDeconvolutionGenerator",
]
