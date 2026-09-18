"""Reusable orchestration for reproducible analysis precomputations.

The package deliberately separates three concerns: resolving the concrete trained
models selected by an analysis YAML, executing an explicitly ordered precompute
strategy, and persisting the resulting notebook inputs.  Domain modules retain the
scientific calculations; this package never reimplements their metrics.
"""

from .core.contracts import AnalysisPlugin, ArtifactSpec, PrecomputeStrategy
from .core.runner import run_precompute

__all__ = ["AnalysisPlugin", "ArtifactSpec", "PrecomputeStrategy", "run_precompute"]
