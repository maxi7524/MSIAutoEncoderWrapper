"""Independent normalization components for MSI intensity spaces."""

from .base import (
    BaseNormalization,
    DenormalizationStage,
    NormalizationCapabilities,
    NormalizationStage,
    NormalizationTrace,
    OutputSpace,
)
from .pipeline import NormalizationPipeline, ReconstructionPolicy
from .output import (
    SUPPORTED_OUTPUT_NORMALIZATIONS,
    SpectrumOutputNormalization,
    build_output_normalization,
)
from .strategies import ScalarNormalization

__all__ = [
    "BaseNormalization",
    "DenormalizationStage",
    "NormalizationCapabilities",
    "NormalizationPipeline",
    "NormalizationStage",
    "NormalizationTrace",
    "OutputSpace",
    "ReconstructionPolicy",
    "ScalarNormalization",
    "SUPPORTED_OUTPUT_NORMALIZATIONS",
    "SpectrumOutputNormalization",
    "build_output_normalization",
]
