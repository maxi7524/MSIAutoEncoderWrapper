"""IsoSpec-backed fine isotopic envelopes on the active binned axis."""

from __future__ import annotations

from typing import Any

import IsoSpecPy
import numpy as np

from msi_dataset_manager.annotations.candidates import resolve_ionic_composition
from msi_dataset_manager.annotations.candidates.model import ELECTRON_MASS

from .base import (
    SyntheticRepresentationContext,
    SyntheticRepresentationStrategy,
    register_representation_strategy,
)


@register_representation_strategy("isospec_envelope")
class IsoSpecEnvelopeRepresentation(SyntheticRepresentationStrategy):
    """Render weighted sums of adduct-aware IsoSpec isotope envelopes.

    IsoSpec computes infinitely resolved isotopologue masses and natural
    abundances. The adapter applies the ionic electron correction, maps every
    line through the active binner, and sums line intensities that share a bin.
    """

    def __init__(
        self,
        probability_coverage: float = 0.999,
        max_isotopologues: int = 4096,
        occupancy_alpha: float = 1.0,
        intensity_log_sigma: float = 0.5,
    ) -> None:
        """Configure envelope truncation and independent ion abundances.

        :param probability_coverage: Minimum joint natural-abundance mass
            retained by IsoSpec before a safety cap is applied.
        :type probability_coverage: float
        :param max_isotopologues: Maximum retained fine-structure lines per ion.
        :type max_isotopologues: int
        :param occupancy_alpha: Symmetric Dirichlet concentration for relative
            ion occupancy in a mixture.
        :type occupancy_alpha: float
        :param intensity_log_sigma: Log-normal spread of per-ion response.
        :type intensity_log_sigma: float
        """
        if not 0.0 < probability_coverage <= 1.0:
            raise ValueError("probability_coverage must belong to (0, 1].")
        if (
            isinstance(max_isotopologues, bool)
            or not isinstance(max_isotopologues, int)
            or max_isotopologues < 1
        ):
            raise ValueError("max_isotopologues must be a positive integer.")
        if occupancy_alpha <= 0:
            raise ValueError("occupancy_alpha must be positive.")
        if intensity_log_sigma < 0:
            raise ValueError("intensity_log_sigma must be nonnegative.")
        self.probability_coverage = float(probability_coverage)
        self.max_isotopologues = max_isotopologues
        self.occupancy_alpha = float(occupancy_alpha)
        self.intensity_log_sigma = float(intensity_log_sigma)
        self._envelope_cache: dict[tuple[str, str, int], tuple[np.ndarray, np.ndarray]] = {}

    def render(self, rng, context, definition):
        """Return a binned weighted sum of all declared molecular ions.

        :raises ValueError: If a component has no declared formula/adduct label
            or if an ion cannot be mapped to the active axis.
        """
        ions = [
            (int(label_index), float(component.intensity_weight))
            for component in definition.components
            for label_index in component.label_indices
        ]
        if not ions:
            raise ValueError(
                "isospec_envelope requires declared molecular components; use a "
                "reconstruction-only representation for unlabelled samples."
            )
        occupancy = rng.dirichlet(np.full(len(ions), self.occupancy_alpha))  # (J,)
        response = rng.lognormal(
            mean=0.0,
            sigma=self.intensity_log_sigma,
            size=len(ions),
        )  # (J,)
        component_weights = np.asarray([weight for _, weight in ions], dtype=np.float64)  # (J,)
        weights = occupancy * response * component_weights  # (J,)
        total_weight = weights.sum()  # ()
        if total_weight <= 0:
            raise ValueError("The declared molecular components have zero total intensity.")
        weights /= total_weight
        spectrum = np.zeros(context.feature_count, dtype=np.float64)  # (M,)

        # Mixture assembly
        ## Every declared ion contributes its full theoretical isotope envelope.
        for (label_index, _), weight in zip(ions, weights, strict=True):
            formula, adduct, charge = self._ion_identity(context, label_index)
            masses, probabilities = self._envelope(formula, adduct, charge)  # (I,), (I,)
            mz_values = (masses - charge * ELECTRON_MASS) / abs(charge)  # (I,)
            bin_indices = self._map_masses(context, mz_values)  # (I,)
            valid = (bin_indices >= 0) & (bin_indices < context.feature_count)  # (I,)
            if not bool(valid.any()):
                raise ValueError(
                    f"IsoSpec envelope for '{formula}|{adduct}' has no line on the active axis."
                )
            np.add.at(
                spectrum,
                bin_indices[valid],
                weight * probabilities[valid],
            )
        return spectrum

    def _ion_identity(
        self,
        context: SyntheticRepresentationContext,
        label_index: int,
    ) -> tuple[str, str, int]:
        """Resolve formula/adduct/charge from source metadata or stable labels."""
        metadata: dict[str, Any] = dict(context.source.get_label_metadata(label_index))
        name = context.source.class_names[label_index]
        if "formula" in metadata and "adduct" in metadata:
            formula = str(metadata["formula"])
            adduct = str(metadata["adduct"])
            _, charge = resolve_ionic_composition(
                formula,
                adduct,
                metadata.get("charge"),
            )
            return formula, adduct, charge
        parts = name.split("|", 2)
        if len(parts) < 2:
            raise ValueError(
                "IsoSpec labels must use 'formula|adduct' or source metadata."
            )
        formula, adduct = parts[:2]
        _, charge = resolve_ionic_composition(formula, adduct)
        return formula, adduct, charge

    def _envelope(
        self,
        formula: str,
        adduct: str,
        charge: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return cached fine isotopologue masses and natural abundances."""
        key = formula, adduct, charge
        cached = self._envelope_cache.get(key)
        if cached is not None:
            return cached
        composition, resolved_charge = resolve_ionic_composition(formula, adduct, charge)
        ionic_formula = _format_formula(composition)
        distribution = IsoSpecPy.IsoTotalProb(
            formula=ionic_formula,
            prob_to_cover=self.probability_coverage,
        )
        lines = np.asarray(list(distribution), dtype=np.float64)  # (I, 2)
        if lines.ndim != 2 or lines.shape[1] != 2 or not len(lines):
            raise ValueError(f"IsoSpec returned no envelope lines for '{ionic_formula}'.")
        masses, probabilities = lines[:, 0], lines[:, 1]  # (I,), (I,)
        if len(masses) > self.max_isotopologues:
            selected = np.argsort(probabilities)[-self.max_isotopologues:]  # (I_cap,)
            masses, probabilities = masses[selected], probabilities[selected]  # (I_cap,), (I_cap,)
        retained_probability = probabilities.sum()  # ()
        if retained_probability <= 0:
            raise ValueError(f"IsoSpec returned zero retained probability for '{ionic_formula}'.")
        # REMARK: The requested mixture weight is the integrated ion signal,
        # rather than the arbitrary probability mass remaining after coverage
        # truncation and the explicit isotopologue safety cap.
        probabilities = probabilities / retained_probability  # (I_retained,)
        ordered = np.argsort(masses)  # (I_retained,)
        result = masses[ordered], probabilities[ordered]  # (I_retained,), (I_retained,)
        self._envelope_cache[key] = result
        return result

    @staticmethod
    def _map_masses(
        context: SyntheticRepresentationContext,
        mz_values: np.ndarray,
    ) -> np.ndarray:
        """Map theoretical lines with the active binner or a nearest-axis fallback."""
        if context.mass_to_bin is not None:
            return np.asarray(context.mass_to_bin(mz_values), dtype=np.int64)  # (I,)
        if context.mass_axis is None:
            raise ValueError("IsoSpec rendering requires an active mass axis.")
        axis = np.asarray(context.mass_axis, dtype=np.float64)  # (M,)
        insertion = np.searchsorted(axis, mz_values)  # (I,)
        right = np.clip(insertion, 0, len(axis) - 1)  # (I,)
        left = np.clip(insertion - 1, 0, len(axis) - 1)  # (I,)
        return np.where(
            np.abs(axis[right] - mz_values) < np.abs(axis[left] - mz_values),
            right,
            left,
        ).astype(np.int64)  # (I,)


def _format_formula(composition: dict[str, int]) -> str:
    """Create a deterministic IsoSpec formula from positive elemental counts."""
    ordered = []
    for element in ("C", "H"):
        if element in composition:
            ordered.append(element)
    ordered.extend(sorted(set(composition) - {"C", "H"}))
    return "".join(f"{element}{composition[element]}" for element in ordered)
