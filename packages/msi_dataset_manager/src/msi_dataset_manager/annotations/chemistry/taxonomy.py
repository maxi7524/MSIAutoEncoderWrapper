"""Uniform chemical class taxonomies and isomer-preserving ion class assignment.

Every supported source (LIPID MAPS, ClassyFire/ChemOnt, ChEBI, SwissLipids) is
normalized to the same six tables, so that graph construction and analysis never
depend on provider-specific layouts:

``classes``
    ``class_id``, ``name``, ``rank`` (provider rank name, possibly empty).
``class_edges``
    ``child_id``, ``parent_id`` — direct ``is_a`` relations between classes.
``compounds``
    ``compound_id``, ``name``, ``formula`` (canonical neutral Hill formula).
``memberships``
    ``compound_id``, ``class_id`` — direct class membership of one compound.
``aliases``
    ``alias``, ``compound_id`` — namespace-qualified external identifiers
    (own identifier, cross-references, secondary accessions) of one compound.
``alignments``
    ``class_id``, ``target_id``, ``predicate`` — declared correspondence of a class
    to a class of another taxonomy (``EXACT``, ``BROAD``, ``NARROW``, ``RELATED``).

A class hierarchy may be a tree (LIPID MAPS, ChemOnt) or a directed acyclic graph
(ChEBI, SwissLipids, ChemOnt memberships with alternative parents). The closure
operations below are defined for the general DAG case.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from functools import cached_property
import json
from pathlib import Path
import re
from typing import Iterable, Mapping

import pandas as pd

from .formula import parse_formula
from ...utils.logger import get_custom_logger

logger = get_custom_logger(__name__)

TAXONOMY_SCHEMA_VERSION = 1
TAXONOMY_COLUMNS: dict[str, tuple[str, ...]] = {
    "classes": ("class_id", "name", "rank"),
    "class_edges": ("child_id", "parent_id"),
    "compounds": ("compound_id", "name", "formula"),
    "memberships": ("compound_id", "class_id"),
    "aliases": ("alias", "compound_id"),
    "alignments": ("class_id", "target_id", "predicate"),
}
RESOLUTION_MODES = ("metaspace_ids", "formula")

_IDENTIFIER_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"^(?:hmdb:)?HMDB(\d+)$", re.IGNORECASE), "HMDB"),
    (re.compile(r"^(?:lipidmaps:|LIPIDMAPS:)?(LM[A-Z]{2}[0-9A-Z]{6,})$"), "LIPIDMAPS"),
    (re.compile(r"^(?:swisslipids:)?(SLM:\d+)$", re.IGNORECASE), "SWISSLIPIDS"),
    (re.compile(r"^CHEBI:(\d+)$", re.IGNORECASE), "CHEBI"),
    (re.compile(r"^(?:kegg\.compound:|kegg:)?(C\d{5})$"), "KEGG"),
    (re.compile(r"^(?:drugbank:)?(DB\d{5})$", re.IGNORECASE), "DRUGBANK"),
)


def normalize_identifier(value: object) -> str | None:
    """Map a raw provider or cross-reference identifier to a qualified alias.

    Supported namespaces are HMDB (normalized to seven digits so that HMDB 2.x
    five-digit accessions equal HMDB 4/5 accessions), LIPID MAPS, SwissLipids,
    ChEBI, KEGG compound, and DrugBank. Prefixes used by OBO cross-references
    (``hmdb:``, ``kegg.compound:``, ``lipidmaps:``, ``drugbank:``) are accepted.

    :param value: Raw identifier, for example ``HMDB00220``, ``C00249``,
        ``SLM:000056508`` or ``kegg.compound:C00249``.
    :type value: object
    :return: Alias such as ``HMDB:HMDB0000220`` or ``None`` for an unsupported or
        ambiguous identifier (for example bare numbers).
    :rtype: str | None
    """
    text = str(value or "").strip()
    for pattern, namespace in _IDENTIFIER_PATTERNS:
        match = pattern.match(text)
        if match is None:
            continue
        token = match.group(1)
        if namespace == "HMDB":
            return f"HMDB:HMDB{int(token):07d}"
        if namespace == "CHEBI":
            return f"CHEBI:{int(token)}"
        if namespace == "SWISSLIPIDS":
            return f"SWISSLIPIDS:{token.upper()}"
        return f"{namespace}:{token.upper()}"
    return None


def canonical_formula(formula: str, charge: int = 0) -> str | None:
    """Return the neutral Hill-ordered formula of a possibly charged species.

    :param formula: Elemental formula of the represented species.
    :type formula: str
    :param charge: Net charge of that species. A charged species is neutralized by
        removing ``charge`` hydrogens (adding them for anions), which reproduces the
        neutral formula of protonation microspecies.
    :type charge: int
    :return: Canonical neutral formula, or ``None`` if the formula is not a strict
        elemental formula (generic ``R`` groups, isotopes) or cannot be neutralized.
    :rtype: str | None

    REMARK: Neutralization by hydrogen transfer is exact for protonation
    microspecies (carboxylates, ammonium ions), which dominate pH-specific
    exports. It is not chemically meaningful for permanent ions such as
    quaternary ammonium cations without a counter-charge; such species receive a
    formula that METASPACE would annotate as ``[M]+`` instead.
    """
    try:
        counts = dict(parse_formula(str(formula)))
    except ValueError:
        return None
    if charge:
        hydrogen = counts.get("H", 0) - int(charge)
        if hydrogen < 0:
            return None
        if hydrogen:
            counts["H"] = hydrogen
        else:
            counts.pop("H", None)
    if not counts:
        return None
    ## Hill order: C, H, then alphabetical; without carbon all elements alphabetical
    if "C" in counts:
        order = ["C"] + (["H"] if "H" in counts else []) + sorted(set(counts) - {"C", "H"})
    else:
        order = sorted(counts)
    return "".join(f"{element}{counts[element] if counts[element] != 1 else ''}" for element in order)


@dataclass(frozen=True, eq=False)
class ChemicalTaxonomy:
    """One versioned class hierarchy with compound memberships and aliases.

    :param name: Stable taxonomy name, used as class identifier namespace.
    :type name: str
    :param version: Source release or retrieval snapshot.
    :type version: str
    :param tables: Mapping from table name to a frame with the columns of
        :data:`TAXONOMY_COLUMNS`.
    :type tables: collections.abc.Mapping[str, pandas.DataFrame]
    :raises ValueError: If a table or column is missing, identifiers are not unique,
        or an edge references an unknown class.
    """

    name: str
    version: str
    tables: Mapping[str, pd.DataFrame] = field(repr=False)

    def __post_init__(self) -> None:
        missing = [
            f"{table}.{column}"
            for table, columns in TAXONOMY_COLUMNS.items()
            for column in columns
            if table not in self.tables or column not in self.tables[table].columns
        ]
        if missing:
            raise ValueError(f"Taxonomy '{self.name}' misses required columns: {missing}.")
        if self.classes.class_id.duplicated().any() or self.compounds.compound_id.duplicated().any():
            raise ValueError(f"Taxonomy '{self.name}' has duplicated class or compound identifiers.")
        known = set(self.classes.class_id)
        unknown = (set(self.class_edges.child_id) | set(self.class_edges.parent_id) | set(self.memberships.class_id)) - known
        if unknown:
            raise ValueError(f"Taxonomy '{self.name}' references {len(unknown)} undefined classes.")

    # Table accessors
    @property
    def classes(self) -> pd.DataFrame:
        """Return class records."""
        return self.tables["classes"]

    @property
    def class_edges(self) -> pd.DataFrame:
        """Return direct child-parent class relations."""
        return self.tables["class_edges"]

    @property
    def compounds(self) -> pd.DataFrame:
        """Return compound records with canonical neutral formulae."""
        return self.tables["compounds"]

    @property
    def memberships(self) -> pd.DataFrame:
        """Return direct compound-class memberships."""
        return self.tables["memberships"]

    @property
    def aliases(self) -> pd.DataFrame:
        """Return qualified external identifiers of compounds."""
        return self.tables["aliases"]

    @property
    def alignments(self) -> pd.DataFrame:
        """Return declared correspondences to classes of other taxonomies."""
        return self.tables["alignments"]

    # Hierarchy closure
    @cached_property
    def parents(self) -> dict[str, tuple[str, ...]]:
        """Return direct parents per class."""
        grouped = self.class_edges.groupby("child_id").parent_id.agg(lambda values: tuple(sorted(set(values))))
        return grouped.to_dict()

    @cached_property
    def class_depth(self) -> dict[str, int]:
        """Return the shortest ``is_a`` distance of every class from a root class.

        Roots are classes without parents. In a DAG the shortest distance is used,
        so a class reachable through a short and a long path receives the smaller
        level; the longest path would overstate specificity of multiply-inherited
        classes.
        """
        children: dict[str, list[str]] = {}
        for child, parent in self.class_edges.itertuples(index=False):
            children.setdefault(parent, []).append(child)
        depth: dict[str, int] = {}
        queue = deque((class_id, 0) for class_id in self.classes.class_id if class_id not in self.parents)
        while queue:
            class_id, level = queue.popleft()
            if class_id in depth:
                continue
            depth[class_id] = level
            queue.extend((child, level + 1) for child in children.get(class_id, ()))
        return depth

    @cached_property
    def _ancestor_cache(self) -> dict[str, frozenset[str]]:
        return {}

    def ancestors(self, class_id: str) -> frozenset[str]:
        """Return a class together with all its transitive ``is_a`` ancestors.

        :param class_id: Class identifier.
        :type class_id: str
        :return: Reflexive-transitive ancestor set.
        :rtype: frozenset[str]
        """
        cache = self._ancestor_cache
        if class_id in cache:
            return cache[class_id]
        result, stack = {class_id}, list(self.parents.get(class_id, ()))
        while stack:
            parent = stack.pop()
            if parent in result:
                continue
            if parent in cache:
                result |= cache[parent]
                continue
            result.add(parent)
            stack.extend(self.parents.get(parent, ()))
        cache[class_id] = frozenset(result)
        return cache[class_id]

    @cached_property
    def direct_classes(self) -> dict[str, frozenset[str]]:
        """Return direct class memberships per compound."""
        grouped = self.memberships.groupby("compound_id").class_id.agg(frozenset)
        return grouped.to_dict()

    def compound_closure(self, compound_id: str) -> frozenset[str]:
        """Return every class of one compound, including inherited ancestors.

        :param compound_id: Compound identifier.
        :type compound_id: str
        :return: Union of the ancestor closures of all direct memberships; empty for
            an unclassified compound.
        :rtype: frozenset[str]
        """
        classes: set[str] = set()
        for class_id in self.direct_classes.get(compound_id, ()):
            classes |= self.ancestors(class_id)
        return frozenset(classes)

    # Lookup indices
    @cached_property
    def alias_index(self) -> dict[str, tuple[str, ...]]:
        """Return compounds per qualified alias."""
        grouped = self.aliases.groupby("alias").compound_id.agg(lambda values: tuple(sorted(set(values))))
        return grouped.to_dict()

    @cached_property
    def formula_index(self) -> dict[str, tuple[str, ...]]:
        """Return compounds per canonical neutral formula."""
        grouped = self.compounds.groupby("formula").compound_id.agg(lambda values: tuple(sorted(values)))
        return grouped.to_dict()

    @cached_property
    def compound_formula(self) -> dict[str, str]:
        """Return the canonical neutral formula per compound."""
        return dict(zip(self.compounds.compound_id, self.compounds.formula))

    # Persistence
    def save(self, directory: Path | str) -> Path:
        """Write the taxonomy as compressed CSV tables and a manifest.

        :param directory: Destination directory; created when absent.
        :type directory: pathlib.Path | str
        :return: The destination directory.
        :rtype: pathlib.Path
        """
        target = Path(directory)
        target.mkdir(parents=True, exist_ok=True)
        for table, columns in TAXONOMY_COLUMNS.items():
            self.tables[table].loc[:, list(columns)].to_csv(target / f"{table}.csv.gz", index=False)
        manifest = {
            "schema_version": TAXONOMY_SCHEMA_VERSION,
            "name": self.name,
            "version": self.version,
            "counts": {table: int(len(frame)) for table, frame in self.tables.items()},
        }
        (target / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        logger.info("Saved taxonomy %s (%s) to %s.", self.name, self.version, target)
        return target

    @classmethod
    def load(cls, directory: Path | str) -> "ChemicalTaxonomy":
        """Read a taxonomy written by :meth:`save`.

        :param directory: Directory containing ``manifest.json`` and the tables.
        :type directory: pathlib.Path | str
        :return: Validated taxonomy.
        :rtype: ChemicalTaxonomy
        :raises ValueError: If the schema version is unsupported.
        """
        source = Path(directory)
        manifest = json.loads((source / "manifest.json").read_text())
        if manifest.get("schema_version") != TAXONOMY_SCHEMA_VERSION:
            raise ValueError(f"Unsupported taxonomy schema in '{source}'.")
        tables = {
            table: pd.read_csv(source / f"{table}.csv.gz", dtype=str, keep_default_na=False)
            for table in TAXONOMY_COLUMNS
        }
        return cls(manifest["name"], manifest["version"], tables)


def resolve_candidates(
    ions: pd.DataFrame,
    taxonomy: ChemicalTaxonomy,
    mode: str,
) -> pd.DataFrame:
    """Resolve annotated ions to the compound candidates of one taxonomy.

    ``metaspace_ids`` uses the identifiers that METASPACE reported as possible
    compounds of the ion (from its annotation database), translated through the
    taxonomy's own identifiers and cross-references. ``formula`` ignores those
    identifiers and admits every compound of the taxonomy with the ion's neutral
    formula, i.e. the complete isomer ambiguity of an MS1 annotation.

    :param ions: One row per ion with ``ion_id``, ``formula`` and ``identifiers``
        (iterable of raw identifiers; may be empty for ``formula`` mode).
    :type ions: pandas.DataFrame
    :param taxonomy: Target taxonomy.
    :type taxonomy: ChemicalTaxonomy
    :param mode: One of :data:`RESOLUTION_MODES`.
    :type mode: str
    :return: Long table ``ion_id``, ``compound_id``; ions without candidates are absent.
    :rtype: pandas.DataFrame
    :raises ValueError: If ``mode`` is unknown.

    REMARK: In ``metaspace_ids`` mode a cross-reference can point to a conjugate or
    salt form with a different formula. Candidates whose canonical formula differs
    from the ion formula are discarded, because they cannot explain the observed ion.
    """
    if mode not in RESOLUTION_MODES:
        raise ValueError(f"Unknown resolution mode '{mode}'; expected one of {RESOLUTION_MODES}.")
    rows: list[tuple[str, str]] = []
    for ion_id, formula, identifiers in ions[["ion_id", "formula", "identifiers"]].itertuples(index=False):
        neutral = canonical_formula(formula)
        if neutral is None:
            continue
        if mode == "formula":
            candidates: Iterable[str] = taxonomy.formula_index.get(neutral, ())
        else:
            aliases = {alias for alias in map(normalize_identifier, identifiers or ()) if alias}
            candidates = {
                compound
                for alias in aliases
                for compound in taxonomy.alias_index.get(alias, ())
                if taxonomy.compound_formula.get(compound) == neutral
            }
        rows.extend((ion_id, compound) for compound in sorted(candidates))
    return pd.DataFrame(rows, columns=["ion_id", "compound_id"])


def class_assignments(candidates: pd.DataFrame, taxonomy: ChemicalTaxonomy) -> pd.DataFrame:
    """Aggregate candidate class closures into per-ion class evidence.

    For an ion :math:`m` with candidate set :math:`S(m)` and class closure
    :math:`C(s)` of candidate :math:`s`, the support fraction of class :math:`c` is

    .. math::

        f(m, c) = |\\{s \\in S(m) : c \\in C(s)\\}| / |S(m)|.

    ``certain`` classes (:math:`f = 1`) are the intersection of all candidate
    closures and ``possible`` classes (:math:`f > 0`) their union, matching
    :func:`msi_dataset_manager.annotations.chemistry.store.read_chemistry`.
    Unclassified candidates count in the denominator, so a class is certain only if
    every candidate structure belongs to it.

    :param candidates: Output of :func:`resolve_candidates`.
    :type candidates: pandas.DataFrame
    :param taxonomy: Taxonomy that produced the candidates.
    :type taxonomy: ChemicalTaxonomy
    :return: Table ``ion_id``, ``class_id``, ``support_fraction``, ``certain``,
        ``candidate_count``.
    :rtype: pandas.DataFrame
    """
    rows: list[tuple[str, str, float, bool, int]] = []
    for ion_id, group in candidates.groupby("ion_id", sort=True):
        compounds = group.compound_id.tolist()
        counts: dict[str, int] = {}
        for compound in compounds:
            for class_id in taxonomy.compound_closure(compound):
                counts[class_id] = counts.get(class_id, 0) + 1
        total = len(compounds)
        rows.extend((ion_id, class_id, count / total, count == total, total) for class_id, count in sorted(counts.items()))
    return pd.DataFrame(rows, columns=["ion_id", "class_id", "support_fraction", "certain", "candidate_count"])
