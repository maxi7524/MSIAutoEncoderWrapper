"""Analysis of the signal-evidence rule that turns missing annotations into labels.

The training objectives ``SignalMaskedBCELoss`` and ``ThreeStateCrossEntropyLoss``
do not read annotations alone: every ion without an annotation in a pixel is split
into an operational negative ``N`` and an uncertain ``U`` by
:class:`~msi_autoencoder_wrapper.data.annotation_evidence.SignalEvidencePolicy`.
This package measures the population that rule acts on, so its thresholds can be
selected from evidence instead of being assumed.

Modules are organized by analytical question:

``precompute``
    One streaming pass over the pixel population producing every histogram and
    per-pixel summary the other modules consume.
``threshold_analysis``
    How the P/N/U yield, the positive contradiction rate and the class-level
    separation respond to the relative and absolute thresholds.
``class_population_analysis``
    Per-ion annotation prevalence and evidence regime (rare, undetectable,
    background-dominated) over the 200-900 m/z window.
``alignment_analysis``
    Whether annotated m/z positions actually land on the bin carrying the peak,
    which is what the ``bin_radius`` dilation is supposed to absorb.
"""

from .precompute import (
    EvidencePrecompute,
    EvidenceStatistics,
    build_evidence_grid,
    build_population_dataset,
)

__all__ = [
    "EvidencePrecompute",
    "EvidenceStatistics",
    "build_evidence_grid",
    "build_population_dataset",
]
