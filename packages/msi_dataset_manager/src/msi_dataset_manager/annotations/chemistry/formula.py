"""Strict elemental compositions for neutral molecular formula annotations."""

import re
from collections import Counter

ELEMENTS = frozenset("H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn Ga Ge As Se Br Kr Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe Cs Ba La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Hf Ta W Re Os Ir Pt Au Hg Tl Pb Bi Po At Rn Fr Ra Ac Th Pa U Np Pu Am Cm Bk Cf Es Fm Md No Lr Rf Db Sg Bh Hs Mt Ds Rg Cn Nh Fl Mc Lv Ts Og".split())


def parse_formula(formula: str) -> dict[str, int]:
    """Parse an explicit elemental formula without silently dropping syntax.

    :param formula: Neutral formula such as ``C6H12O6``; repeated elements are added.
    :type formula: str
    :return: Element symbols mapped to positive atom counts.
    :rtype: dict[str, int]
    :raises ValueError: For empty formulas, unknown elements, charges, isotopic
        notation, parentheses, hydrates, or nonpositive counts. These extensions
        require explicit preprocessing rather than an implicit numerical convention.
    """
    if not isinstance(formula, str) or not formula:
        raise ValueError("A nonempty neutral elemental formula is required.")
    matches = list(re.finditer(r"([A-Z][a-z]?)([0-9]*)", formula))
    if "".join(match.group(0) for match in matches) != formula:
        raise ValueError(f"Unsupported molecular formula syntax: {formula!r}.")
    counts = Counter()
    for match in matches:
        element, number = match.groups()
        count = int(number) if number else 1
        if element not in ELEMENTS or count < 1:
            raise ValueError(f"Invalid element or count in formula: {formula!r}.")
        counts[element] += count
    return dict(sorted(counts.items()))
