"""Readers normalizing local taxonomy exports to :class:`ChemicalTaxonomy`.

All readers work offline on exports stored in
``assets/local/metabolite_databases/`` and keep every compound of the source,
independent of any dataset ``m/z`` range, because isomer ambiguity must be judged
against the complete compound population of a taxonomy.
"""

from __future__ import annotations

from collections import deque
import csv
import gzip
from pathlib import Path
import re
from typing import IO, Iterable, Iterator
import xml.etree.ElementTree as ElementTree

import pandas as pd

from .taxonomy import ChemicalTaxonomy, canonical_formula, normalize_identifier
from ...utils.logger import get_custom_logger

logger = get_custom_logger(__name__)

CHEMONT_RANKS = {1: "kingdom", 2: "superclass", 3: "class", 4: "subclass"}
_LIPIDMAPS_LEVELS = (("core", "category"), ("main_class", "main_class"),
                     ("sub_class", "sub_class"), ("class_level4", "class_level4"))
_LIPIDMAPS_CODE = re.compile(r"^(?P<name>.*?)\s*\[(?P<code>[A-Z0-9]+)\]\s*$")
_OBO_SYNONYM = re.compile(r'^"(?P<text>.*)"\s+(?P<scope>EXACT|BROAD|NARROW|RELATED)\s+(?P<type>\S+)?\s*\[(?P<xref>[^\]]*)\]')
_CHEBI_CONJUGATE_RELATIONS = ("RO:0018033", "RO:0018034")


# Shared helpers
def _frame(rows: Iterable[tuple], columns: tuple[str, ...]) -> pd.DataFrame:
    """Return a de-duplicated string frame with fixed columns."""
    return pd.DataFrame(list(rows), columns=list(columns)).drop_duplicates().reset_index(drop=True)


def _iter_obo_stanzas(stream: Iterable[str]) -> Iterator[dict[str, list[str]]]:
    """Yield every ``[Term]`` stanza as a mapping from tag to all its values.

    :param stream: Text lines of an OBO 1.2/1.4 document.
    :type stream: collections.abc.Iterable[str]
    :return: Iterator over stanza dictionaries; trailing ``! comment`` text of
        reference tags is removed.
    :rtype: collections.abc.Iterator[dict[str, list[str]]]
    """
    current: dict[str, list[str]] | None = None
    for raw_line in stream:
        line = raw_line.rstrip("\n")
        if line.startswith("["):
            if current is not None:
                yield current
            current = {} if line.strip() == "[Term]" else None
            continue
        if current is None or ": " not in line:
            continue
        tag, value = line.split(": ", maxsplit=1)
        if tag in {"is_a", "relationship", "alt_id"}:
            value = value.split(" ! ", maxsplit=1)[0].strip()
        current.setdefault(tag, []).append(value)
    if current is not None:
        yield current


def _depth_from_roots(edges: pd.DataFrame, nodes: Iterable[str]) -> dict[str, int]:
    """Return shortest root distance for a child-parent edge list."""
    parents = edges.groupby("child_id").parent_id.agg(list).to_dict()
    children: dict[str, list[str]] = {}
    for child, parent in edges[["child_id", "parent_id"]].itertuples(index=False):
        children.setdefault(parent, []).append(child)
    depth: dict[str, int] = {}
    queue = deque((node, 0) for node in nodes if node not in parents)
    while queue:
        node, level = queue.popleft()
        if node in depth:
            continue
        depth[node] = level
        queue.extend((child, level + 1) for child in children.get(node, ()))
    return depth


# LIPID MAPS
def load_lipidmaps_taxonomy(path: Path | str, version: str) -> ChemicalTaxonomy:
    """Read the LIPID MAPS Structure Database TSV export.

    Classes are identified by their bracketed LIPID MAPS codes (``GP``, ``GP01``,
    ``GP0101``, ...), which are stable across releases unlike display names. A
    compound is a direct member of its most specific populated level only; the
    coarser levels follow by closure.

    :param path: ``lipidmaps_<version>.tsv`` export.
    :type path: pathlib.Path | str
    :param version: Retrieval snapshot label.
    :type version: str
    :return: LIPID MAPS taxonomy.
    :rtype: ChemicalTaxonomy
    """
    ## REMARK: Imported lazily; candidates.model imports chemistry.formula, so a
    ## module-level import of the candidate package would be circular.
    from ..candidates.providers import _lipidmaps_table_lines

    text = Path(path).read_text(encoding="utf-8-sig")
    ## REMARK: Parsed like the candidate provider (csv module, tab delimiter); some
    ## records contain quoted tabs that the pandas C tokenizer splits incorrectly.
    rows = csv.DictReader(_lipidmaps_table_lines(text), delimiter="\t")
    classes, edges, compounds, memberships, aliases = [], [], [], [], []
    for row in rows:
        record = {key: (value or "") for key, value in row.items() if key is not None}
        formula = canonical_formula(record.get("formula", ""))
        identifier = record.get("lm_id", "").strip()
        if not identifier or formula is None:
            continue
        ## Hierarchy path of the compound
        path_ids: list[str] = []
        for column, rank in _LIPIDMAPS_LEVELS:
            match = _LIPIDMAPS_CODE.match(record.get(column, "").strip())
            if match is None:
                continue
            class_id = f"LIPID_MAPS:{match['code']}"
            classes.append((class_id, match["name"], rank))
            if path_ids:
                edges.append((class_id, path_ids[-1]))
            path_ids.append(class_id)
        compound_id = f"LIPIDMAPS:{identifier}"
        compounds.append((compound_id, record.get("name") or record.get("sys_name") or identifier, formula))
        if path_ids:
            memberships.append((compound_id, path_ids[-1]))
        ## Identifier aliases
        raw_aliases = [identifier, record.get("hmdb_id", ""), record.get("kegg_id", "")]
        chebi = record.get("chebi_id", "").strip()
        raw_aliases.append(f"CHEBI:{chebi}" if chebi.isdigit() else chebi)
        aliases.extend((alias, compound_id) for alias in map(normalize_identifier, raw_aliases) if alias)
    class_frame = _frame(classes, ("class_id", "name", "rank")).drop_duplicates("class_id")
    taxonomy = ChemicalTaxonomy("LIPID_MAPS", version, {
        "classes": class_frame,
        "class_edges": _frame(edges, ("child_id", "parent_id")),
        "compounds": _frame(compounds, ("compound_id", "name", "formula")).drop_duplicates("compound_id"),
        "memberships": _frame(memberships, ("compound_id", "class_id")),
        "aliases": _frame(aliases, ("alias", "compound_id")),
        "alignments": _frame([], ("class_id", "target_id", "predicate")),
    })
    logger.info("Loaded LIPID MAPS taxonomy: %s classes, %s compounds.", len(class_frame), len(taxonomy.compounds))
    return taxonomy


# ClassyFire / ChemOnt
def _chemont_classes(path: Path | str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Read ChemOnt classes, is_a edges, and cross-taxonomy class synonyms."""
    classes, edges, alignments = [], [], []
    with open(path, encoding="utf-8") as stream:
        for stanza in _iter_obo_stanzas(stream):
            identifier = stanza.get("id", [""])[0]
            if not identifier.startswith("CHEMONTID:"):
                continue
            class_id = f"ClassyFire:{identifier}"
            classes.append((class_id, stanza.get("name", [identifier])[0]))
            edges.extend((class_id, f"ClassyFire:{parent}") for parent in stanza.get("is_a", []))
            for synonym in stanza.get("synonym", []):
                match = _OBO_SYNONYM.match(synonym)
                if match is None or not match["xref"]:
                    continue
                for reference in match["xref"].split(","):
                    reference = reference.strip()
                    if reference.startswith("LIPIDMAPS:"):
                        target = f"LIPID_MAPS:{reference.split(':', 1)[1]}"
                    elif reference.startswith("CHEBI:"):
                        target = f"ChEBI:{reference}"
                    else:
                        continue
                    alignments.append((class_id, target, match["scope"]))
    edge_frame = _frame(edges, ("child_id", "parent_id"))
    class_frame = _frame(classes, ("class_id", "name"))
    depth = _depth_from_roots(edge_frame, class_frame.class_id)
    class_frame["rank"] = [CHEMONT_RANKS.get(depth.get(c, -1), f"level_{depth.get(c, -1)}") for c in class_frame.class_id]
    class_frame.loc[class_frame.class_id.map(depth).eq(0), "rank"] = "root"
    return class_frame, edge_frame, _frame(alignments, ("class_id", "target_id", "predicate"))


def _iter_hmdb_metabolites(stream: IO[bytes]) -> Iterator[dict[str, object]]:
    """Stream HMDB metabolites with taxonomy and cross-references only."""
    def local(tag: str) -> str:
        return tag.rsplit("}", maxsplit=1)[-1]

    for _, element in ElementTree.iterparse(stream, events=("end",)):
        if local(element.tag) != "metabolite":
            continue
        record: dict[str, object] = {"secondary": [], "alternative_parents": []}
        for child in element:
            tag = local(child.tag)
            if tag == "secondary_accessions":
                record["secondary"] = [(item.text or "").strip() for item in child]
            elif tag == "taxonomy":
                for item in child:
                    item_tag = local(item.tag)
                    if item_tag == "direct_parent":
                        record["direct_parent"] = (item.text or "").strip()
                    elif item_tag == "alternative_parents":
                        record["alternative_parents"] = [(value.text or "").strip() for value in item]
            elif tag in {"accession", "name", "chemical_formula", "kegg_id", "chebi_id", "drugbank_id"}:
                record[tag] = (child.text or "").strip()
        yield record
        element.clear()


def load_classyfire_taxonomy(
    chemont_path: Path | str,
    hmdb_path: Path | str,
    version: str,
    *,
    alternative_parents: bool = False,
) -> ChemicalTaxonomy:
    """Combine the ChemOnt ontology with HMDB ClassyFire compound assignments.

    ChemOnt supplies the class tree with stable ``CHEMONTID`` identifiers and the
    declared synonyms to LIPID MAPS and ChEBI classes. HMDB supplies, for every
    metabolite, the ClassyFire ``direct_parent`` and the ``alternative_parents``.
    The canonical ClassyFire lineage (kingdom, superclass, class, subclass, direct
    parent) is the ancestor path of the direct parent, so with
    ``alternative_parents=False`` every compound lies on one path of the tree.
    Alternative parents add further, often unrelated branches (for example both
    *Lipids and lipid-like molecules* and *Organic acids and derivatives*), which
    makes a compound a member of several classes at the same depth.

    :param chemont_path: ``ChemOnt_2_1.obo``.
    :type chemont_path: pathlib.Path | str
    :param hmdb_path: Extracted ``hmdb_metabolites.xml``.
    :type hmdb_path: pathlib.Path | str
    :param version: Combined snapshot label, e.g. ``chemont-2.1_hmdb-5.0``.
    :type version: str
    :param alternative_parents: Also use HMDB alternative parents as direct
        memberships.
    :type alternative_parents: bool
    :return: ClassyFire taxonomy over HMDB compounds.
    :rtype: ChemicalTaxonomy

    REMARK: HMDB stores class names, not ChemOnt identifiers. Names are matched
    case-insensitively to ChemOnt; unmatched names (renamed or newer classes) are
    counted and logged rather than silently invented as new classes.
    """
    class_frame, edge_frame, alignments = _chemont_classes(chemont_path)
    by_name = {name.casefold(): class_id for class_id, name in zip(class_frame.class_id, class_frame.name)}
    compounds, memberships, aliases = [], [], []
    unmatched: dict[str, int] = {}
    with open(hmdb_path, "rb") as stream:
        for record in _iter_hmdb_metabolites(stream):
            compound_id = normalize_identifier(record.get("accession", ""))
            formula = canonical_formula(str(record.get("chemical_formula", "")))
            if compound_id is None or formula is None:
                continue
            compounds.append((compound_id, record.get("name") or compound_id, formula))
            ## ClassyFire memberships
            parents = [record.get("direct_parent", "")]
            if alternative_parents:
                parents += list(record.get("alternative_parents", []))
            for name in parents:
                if not name:
                    continue
                class_id = by_name.get(str(name).casefold())
                if class_id is None:
                    unmatched[str(name)] = unmatched.get(str(name), 0) + 1
                    continue
                memberships.append((compound_id, class_id))
            ## Identifier aliases
            chebi = str(record.get("chebi_id", ""))
            raw = [record.get("accession", ""), *record.get("secondary", []), record.get("kegg_id", ""),
                   record.get("drugbank_id", ""), f"CHEBI:{chebi}" if chebi.isdigit() else ""]
            aliases.extend((alias, compound_id) for alias in map(normalize_identifier, raw) if alias)
    if unmatched:
        logger.warning("ClassyFire: %s HMDB class names (%s memberships) have no ChemOnt term.",
                       len(unmatched), sum(unmatched.values()))
    taxonomy = ChemicalTaxonomy("ClassyFire", version, {
        "classes": class_frame,
        "class_edges": edge_frame,
        "compounds": _frame(compounds, ("compound_id", "name", "formula")).drop_duplicates("compound_id"),
        "memberships": _frame(memberships, ("compound_id", "class_id")),
        "aliases": _frame(aliases, ("alias", "compound_id")),
        "alignments": alignments,
    })
    logger.info("Loaded ClassyFire taxonomy: %s classes, %s compounds.", len(class_frame), len(taxonomy.compounds))
    return taxonomy


# ChEBI
def load_chebi_taxonomy(path: Path | str, version: str) -> ChemicalTaxonomy:
    """Read the complete ChEBI ontology as a structural class DAG.

    Neutral, formula-bearing entities are compounds. Every ``is_a`` ancestor of a
    compound is a class; only ``is_a`` is followed, so role (``has_role``) and
    part-of relations do not enter the chemical hierarchy. Charged entities are not
    compounds, because METASPACE annotates neutral formulae with adducts; their
    identifiers become aliases of the neutral conjugate acid or base.

    :param path: ``chebi_<version>.obo.gz``.
    :type path: pathlib.Path | str
    :param version: ChEBI release.
    :type version: str
    :return: ChEBI taxonomy.
    :rtype: ChemicalTaxonomy
    """
    # Parse ontology terms
    terms: dict[str, dict[str, list[str]]] = {}
    with gzip.open(path, mode="rt", encoding="utf-8") as stream:
        for stanza in _iter_obo_stanzas(stream):
            identifier = stanza.get("id", [""])[0]
            if identifier.startswith("CHEBI:") and stanza.get("is_obsolete", ["false"])[0] != "true":
                terms[identifier] = stanza

    def property_value(stanza: dict[str, list[str]], name: str) -> str:
        for value in stanza.get("property_value", []):
            key, _, remainder = value.partition(" ")
            if key.endswith(name):
                return remainder.split('"')[1] if '"' in remainder else remainder.split(" ")[0]
        return ""

    # Select neutral compounds
    formulas: dict[str, str] = {}
    for identifier, stanza in terms.items():
        charge = property_value(stanza, "charge") or "0"
        try:
            neutral = int(charge) == 0
        except ValueError:
            neutral = False
        formula = canonical_formula(property_value(stanza, "generalized_empirical_formula")) if neutral else None
        if formula is not None:
            formulas[identifier] = formula

    # Class closure reachable from compounds
    parents = {identifier: [p for p in stanza.get("is_a", []) if p in terms] for identifier, stanza in terms.items()}
    class_terms: set[str] = set()
    stack = [parent for compound in formulas for parent in parents[compound]]
    while stack:
        term = stack.pop()
        if term in class_terms:
            continue
        class_terms.add(term)
        stack.extend(parents[term])
    classes = [(f"ChEBI:{term}", terms[term].get("name", [term])[0], "") for term in class_terms]
    edges = [(f"ChEBI:{term}", f"ChEBI:{parent}") for term in class_terms for parent in parents[term]]
    memberships = [(compound, f"ChEBI:{parent}") for compound in formulas for parent in parents[compound]]
    compounds = [(compound, terms[compound].get("name", [compound])[0], formula) for compound, formula in formulas.items()]

    # Identifier aliases, including charged conjugates
    aliases: list[tuple[str, str]] = []
    for identifier, stanza in terms.items():
        targets = [identifier] if identifier in formulas else [
            relation.split(" ")[1] for relation in stanza.get("relationship", [])
            if relation.split(" ")[0] in _CHEBI_CONJUGATE_RELATIONS and relation.split(" ")[1] in formulas
        ]
        if not targets:
            continue
        raw = [identifier, *stanza.get("alt_id", [])]
        raw += [xref.split(" ")[0] for xref in stanza.get("xref", [])
                if xref.split(":")[0] in {"hmdb", "kegg.compound", "lipidmaps", "drugbank"}]
        aliases.extend((alias, target) for alias in map(normalize_identifier, raw) if alias for target in targets)
    taxonomy = ChemicalTaxonomy("ChEBI", version, {
        "classes": _frame(classes, ("class_id", "name", "rank")),
        "class_edges": _frame(edges, ("child_id", "parent_id")),
        "compounds": _frame(compounds, ("compound_id", "name", "formula")),
        "memberships": _frame(memberships, ("compound_id", "class_id")),
        "aliases": _frame(aliases, ("alias", "compound_id")),
        "alignments": _frame([], ("class_id", "target_id", "predicate")),
    })
    logger.info("Loaded ChEBI taxonomy: %s classes, %s neutral compounds.", len(classes), len(compounds))
    return taxonomy


# SwissLipids
def load_swisslipids_taxonomy(path: Path | str, version: str) -> ChemicalTaxonomy:
    """Read the SwissLipids ``lipids.tsv.gz`` export.

    ``Category`` and ``Class`` entries form the class DAG through their
    ``Lipid class*`` references. Molecular, structural and isomeric subspecies are
    collapsed into their root entity by following ``Parent``; only root entities
    (normally ``Species``, i.e. sum composition such as ``PC(34:1)``) are compounds.
    This matches the resolution of an MS1 formula annotation and prevents the
    combinatorial enumeration of acyl-chain positions and double-bond geometries
    from dominating isomer counts.

    :param path: ``swisslipids_<version>.tsv.gz``.
    :type path: pathlib.Path | str
    :param version: Retrieval snapshot label.
    :type version: str
    :return: SwissLipids taxonomy.
    :rtype: ChemicalTaxonomy

    REMARK: SwissLipids formulae are given at pH 7.3 together with the net charge;
    they are neutralized with :func:`canonical_formula`.
    """
    columns = ["Lipid ID", "Level", "Name", "Lipid class*", "Parent", "Formula (pH7.3)",
               "Charge (pH7.3)", "CHEBI", "LIPID MAPS", "HMDB"]
    frame = pd.read_csv(path, sep="\t", dtype=str, usecols=columns, encoding="latin-1",
                        keep_default_na=False)

    def split(value: str) -> list[str]:
        return [item.strip() for item in value.split("|") if item.strip()]

    is_class = frame.Level.isin(["Category", "Class"])
    class_ids = set(frame.loc[is_class, "Lipid ID"])
    classes = [(f"SwissLipids:{i}", name, level.lower()) for i, name, level in
               frame.loc[is_class, ["Lipid ID", "Name", "Level"]].itertuples(index=False)]
    edges = [(f"SwissLipids:{i}", f"SwissLipids:{parent}") for i, value in
             frame.loc[is_class, ["Lipid ID", "Lipid class*"]].itertuples(index=False)
             for parent in split(value) if parent in class_ids]

    # Collapse subspecies into root entities
    entities = frame.loc[~is_class].set_index("Lipid ID")
    parent_of = {i: split(value)[0] for i, value in entities.Parent.items() if split(value)}
    root_cache: dict[str, str] = {}

    def root(identifier: str) -> str:
        chain = []
        while identifier in parent_of and identifier not in root_cache and parent_of[identifier] in entities.index:
            chain.append(identifier)
            identifier = parent_of[identifier]
        resolved = root_cache.get(identifier, identifier)
        for item in chain:
            root_cache[item] = resolved
        return resolved

    compounds, memberships, aliases = [], [], []
    records = entities[["Name", "Lipid class*", "Formula (pH7.3)", "Charge (pH7.3)", "CHEBI", "LIPID MAPS", "HMDB"]]
    for identifier, name, lipid_classes, formula_text, charge_text, chebi, lipidmaps, hmdb in records.itertuples():
        target = root(identifier)
        if target == identifier:
            try:
                charge = int(charge_text or 0)
            except ValueError:
                continue
            formula = canonical_formula(formula_text, charge)
            if formula is None:
                continue
            compounds.append((f"SWISSLIPIDS:{identifier}", name, formula))
            memberships.extend((f"SWISSLIPIDS:{identifier}", f"SwissLipids:{c}")
                               for c in split(lipid_classes) if c in class_ids)
        raw = [identifier, *split(lipidmaps), *split(hmdb)]
        raw += [f"CHEBI:{value}" if value.isdigit() else value for value in split(chebi)]
        aliases.extend((alias, f"SWISSLIPIDS:{target}") for alias in map(normalize_identifier, raw) if alias)
    compound_frame = _frame(compounds, ("compound_id", "name", "formula"))
    known = set(compound_frame.compound_id)
    alias_frame = _frame(aliases, ("alias", "compound_id"))
    taxonomy = ChemicalTaxonomy("SwissLipids", version, {
        "classes": _frame(classes, ("class_id", "name", "rank")),
        "class_edges": _frame(edges, ("child_id", "parent_id")),
        "compounds": compound_frame,
        "memberships": _frame(memberships, ("compound_id", "class_id")),
        "aliases": alias_frame[alias_frame.compound_id.isin(known)].reset_index(drop=True),
        "alignments": _frame([], ("class_id", "target_id", "predicate")),
    })
    logger.info("Loaded SwissLipids taxonomy: %s classes, %s root compounds.", len(classes), len(compound_frame))
    return taxonomy
