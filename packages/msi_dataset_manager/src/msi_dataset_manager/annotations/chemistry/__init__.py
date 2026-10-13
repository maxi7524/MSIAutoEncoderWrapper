"""Versioned chemical descriptions attached to existing MSI annotations."""

from .formula import parse_formula
from .providers import ChemicalCandidate, LipidMapsProvider, SnapshotProvider
from .store import enrich_annotations, read_chemistry
from .taxonomy import (
    RESOLUTION_MODES,
    ChemicalTaxonomy,
    canonical_formula,
    class_assignments,
    normalize_identifier,
    resolve_candidates,
)
from .taxonomy_sources import (
    load_chebi_taxonomy,
    load_classyfire_taxonomy,
    load_lipidmaps_taxonomy,
    load_swisslipids_taxonomy,
)

__all__ = ["parse_formula", "ChemicalCandidate", "LipidMapsProvider", "SnapshotProvider",
           "enrich_annotations", "read_chemistry", "RESOLUTION_MODES", "ChemicalTaxonomy",
           "canonical_formula", "class_assignments", "normalize_identifier", "resolve_candidates",
           "load_chebi_taxonomy", "load_classyfire_taxonomy", "load_lipidmaps_taxonomy",
           "load_swisslipids_taxonomy"]
