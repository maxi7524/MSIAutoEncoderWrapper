"""Canonical candidate compounds, ions, and METASPACE-compatible masses."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping

from ..chemistry.formula import parse_formula


CALCULATOR_VERSION = "metaspace-neutral-formula-adduct-v1"
ELECTRON_MASS = 0.000548579909065

# Neutral atomic masses. Applying the final charge correction once gives the
# ionic masses used by METASPACE-style formula/adduct annotations.
MONOISOTOPIC_MASSES = {
    "H": 1.00782503223,
    "C": 12.0,
    "N": 14.00307400443,
    "O": 15.99491461957,
    "P": 30.97376199842,
    "S": 31.9720711744,
    "F": 18.99840316273,
    "Cl": 34.968852682,
    "Br": 78.9183376,
    "I": 126.9044719,
    "Na": 22.989769282,
    "K": 38.9637064864,
    "Li": 7.0160034366,
    "Ag": 106.9050916,
}

_ADDUCTS: dict[str, tuple[dict[str, int], int]] = {
    "+H": ({"H": 1}, 1),
    "+2H": ({"H": 2}, 2),
    "+3H": ({"H": 3}, 3),
    "+Na": ({"Na": 1}, 1),
    "+K": ({"K": 1}, 1),
    "+Li": ({"Li": 1}, 1),
    "+NH4": ({"N": 1, "H": 4}, 1),
    "+Ag": ({"Ag": 1}, 1),
    "-H": ({"H": -1}, -1),
    "-2H": ({"H": -2}, -2),
    "-3H": ({"H": -3}, -3),
    "+Cl": ({"Cl": 1}, -1),
    "+Br": ({"Br": 1}, -1),
    "[M]+": ({}, 1),
    "[M]-": ({}, -1),
}


@dataclass(frozen=True)
class CandidateClass:
    """One provider-supplied chemical-class hierarchy node."""

    namespace: str
    identifier: str
    name: str
    depth: int


@dataclass(frozen=True)
class CandidateCompound:
    """One structure-aware compound candidate from an external provider."""

    provider: str
    provider_version: str
    identifier: str
    name: str
    formula: str
    monoisotopic_mass: float | None = None
    smiles: str | None = None
    inchi: str | None = None
    inchi_key: str | None = None
    classes: tuple[CandidateClass, ...] = ()
    source_record: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        parse_formula(self.formula)
        if not all((self.provider, self.provider_version, self.identifier, self.name)):
            raise ValueError("Candidate compound identity fields must be non-empty.")

    @property
    def key(self) -> str:
        """Return a stable provider-scoped compound identity."""
        return f"{self.provider}:{self.provider_version}:{self.identifier}"

    def get_config(self) -> dict[str, Any]:
        """Return a portable JSON-compatible compound record."""
        value = asdict(self)
        value["source_record"] = dict(self.source_record or {})
        return value


@dataclass(frozen=True)
class CandidateIon:
    """One formula/adduct ion candidate available to an MSI dataset."""

    compound_key: str
    formula: str
    adduct: str
    charge: int
    polarity: str
    theoretical_mz: float
    calculator_version: str = CALCULATOR_VERSION

    @property
    def key(self) -> str:
        """Return a stable candidate-ion identity."""
        return f"{self.compound_key}|{self.adduct}|{self.charge}"


def default_adducts(polarity: str | None) -> tuple[str, ...]:
    """Return the METASPACE default adduct set for one polarity."""
    if str(polarity).casefold() == "negative":
        return ("-H", "+Cl")
    return ("+H", "+Na", "+K")


def calculate_theoretical_mz(
    formula: str,
    adduct: str,
    charge: int | None = None,
) -> tuple[float, int]:
    """Calculate the monoisotopic m/z for a METASPACE-style formula/adduct.

    METASPACE receives a neutral molecular formula and an adduct separately.
    This function applies the adduct's elemental delta to the neutral mass and
    then applies one electron-mass correction for the final charge state.

    :param formula: Neutral elemental formula.
    :type formula: str
    :param adduct: METASPACE adduct notation, for example ``+H`` or ``-H``.
    :type adduct: str
    :param charge: Optional explicit final charge. It must agree with a known
        adduct's charge convention.
    :type charge: int | None
    :return: Theoretical monoisotopic m/z and the resolved charge.
    :rtype: tuple[float, int]
    :raises ValueError: If the formula contains an unsupported mass element, an
        adduct is unknown, or the charge is zero or inconsistent.
    """
    composition = parse_formula(formula)
    delta, inferred_charge = _ADDUCTS.get(str(adduct), ({}, 0))
    if not inferred_charge:
        raise ValueError(f"Unsupported METASPACE adduct '{adduct}'.")
    resolved_charge = inferred_charge if charge is None else int(charge)
    if resolved_charge == 0 or resolved_charge != inferred_charge:
        raise ValueError("The explicit charge must match the selected adduct.")
    mass = _composition_mass(composition)
    mass += _composition_mass(delta)
    ion_mass = mass - resolved_charge * ELECTRON_MASS
    return ion_mass / abs(resolved_charge), resolved_charge


def make_candidate_ions(
    compound: CandidateCompound,
    *,
    polarity: str | None,
    adducts: tuple[str, ...] | None = None,
) -> tuple[CandidateIon, ...]:
    """Create all requested formula/adduct candidates for one compound."""
    selected = adducts or default_adducts(polarity)
    result = []
    for adduct in selected:
        theoretical_mz, charge = calculate_theoretical_mz(compound.formula, adduct)
        ion_polarity = "Positive" if charge > 0 else "Negative"
        result.append(
            CandidateIon(
                compound_key=compound.key,
                formula=compound.formula,
                adduct=adduct,
                charge=charge,
                polarity=ion_polarity,
                theoretical_mz=theoretical_mz,
            )
        )
    return tuple(result)


def _composition_mass(composition: Mapping[str, int]) -> float:
    """Return a neutral atomic-composition mass."""
    unsupported = sorted(set(composition) - set(MONOISOTOPIC_MASSES))
    if unsupported:
        raise ValueError(f"Unsupported monoisotopic mass elements: {unsupported}.")
    return sum(MONOISOTOPIC_MASSES[element] * count for element, count in composition.items())
