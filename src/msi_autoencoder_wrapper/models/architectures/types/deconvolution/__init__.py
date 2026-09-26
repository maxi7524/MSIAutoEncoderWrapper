"""Torch-native deconvolution primitives for MSI candidate dictionaries."""

from .contracts import DeconvolutionBatch, DeconvolutionResult
from .data.dictionary import GlobalCandidateDictionary
from .data.synthetic import SyntheticDeconvolutionConfig, SyntheticDeconvolutionGenerator
from .evaluation import (
    catalogue_condition_counts,
    deconvolution_metrics,
    identifiability_experiment,
    projected_gradient_convergence,
    projected_gradient_gradcheck,
    sample_global_subdictionary,
)
from .solvers.projected_gradient import NonnegativeProjectedGradientSolver

__all__ = [
    "catalogue_condition_counts",
    "DeconvolutionBatch",
    "DeconvolutionResult",
    "GlobalCandidateDictionary",
    "NonnegativeProjectedGradientSolver",
    "SyntheticDeconvolutionConfig",
    "SyntheticDeconvolutionGenerator",
    "deconvolution_metrics",
    "identifiability_experiment",
    "projected_gradient_convergence",
    "projected_gradient_gradcheck",
    "sample_global_subdictionary",
]
