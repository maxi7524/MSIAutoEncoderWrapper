"""Offline snapshots and explicit LIPID MAPS access for chemical enrichment."""

from __future__ import annotations

from dataclasses import dataclass, asdict
import json
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import quote
from urllib.request import urlopen

from .formula import parse_formula
from ...utils.logger import get_custom_logger

logger = get_custom_logger(__name__)


@dataclass(frozen=True)
class ChemicalCandidate:
    """Retain one candidate structure and all known ancestor classes.

    :param identifier: Provider-specific stable compound identifier.
    :param formula: Neutral elemental formula.
    :param classes: Namespaced hierarchy nodes, including ancestors.
    :param smiles: Optional structure string.
    :param inchi_key: Optional stable structure key.
    """

    identifier: str
    formula: str
    classes: tuple[str, ...] = ()
    smiles: str | None = None
    inchi_key: str | None = None

    def __post_init__(self) -> None:
        parse_formula(self.formula)
        if not self.identifier or not all(isinstance(c, str) and c for c in self.classes):
            raise ValueError("Candidate identifiers and class names must be nonempty strings.")

    def get_config(self) -> dict[str, Any]:
        """Return a portable, JSON-compatible candidate record."""
        return asdict(self)


class ChemistryProvider(Protocol):
    """Contract for versioned, independently cached chemical lookup providers."""

    name: str
    version: str

    def lookup(self, formula: str, identifiers: tuple[str, ...]) -> tuple[ChemicalCandidate, ...]:
        """Return every compatible candidate, preserving structural ambiguity."""
        ...


class SnapshotProvider:
    """Read a frozen local export, including ClassyFire/ChemOnt or lipid classes.

    :param path: JSON containing provider, version, and a candidates list.
    :type path: pathlib.Path | str
    :raises ValueError: If the snapshot is unversioned or has duplicate identities.
    """

    def __init__(self, path: Path | str) -> None:
        record = json.loads(Path(path).read_text())
        self.name, self.version = record["provider"], record["version"]
        if not self.name or not self.version:
            raise ValueError("Chemical snapshots require a provider and version.")
        self.candidates = tuple(ChemicalCandidate(**{**c, "classes": tuple(c.get("classes", ()))})
                                for c in record["candidates"])
        if len({c.identifier for c in self.candidates}) != len(self.candidates):
            raise ValueError("Chemical snapshot identifiers must be unique.")

    def lookup(self, formula, identifiers=()):
        """Prefer supplied identifiers, otherwise retain every formula candidate."""
        matched = tuple(c for c in self.candidates if c.identifier in identifiers)
        if matched:
            return matched
        composition = parse_formula(formula)
        return tuple(c for c in self.candidates if parse_formula(c.formula) == composition)


class LipidMapsProvider:
    """Fetch LIPID MAPS records outside training using the documented REST interface.

    :param version: Explicit database release or retrieval-snapshot label.
    :param timeout: HTTP timeout in seconds.
    :param base_url: REST service root; configurable for a maintained mirror.
    :raises ValueError: If the version or timeout is invalid.

    REMARK: The live endpoint is not release-pinned. The enrichment store freezes
    returned records; the supplied version must identify that retrieval snapshot.
    """

    name = "LIPID_MAPS"

    def __init__(self, version: str, timeout: float = 30,
                 base_url: str = "https://www.lipidmaps.org/rest") -> None:
        if not version or timeout <= 0:
            raise ValueError("A snapshot version and positive HTTP timeout are required.")
        self.version, self.timeout, self.base_url = version, timeout, base_url.rstrip("/")

    def _fetch(self, field: str, value: str, output: str = "all") -> list[dict[str, Any]]:
        url = f"{self.base_url}/compound/{field}/{quote(value, safe='')}/{output}/json"
        with urlopen(url, timeout=self.timeout) as response:
            payload = json.load(response)
        if not payload:
            return []
        if isinstance(payload, list):
            return payload
        if isinstance(payload, dict) and any(key in payload for key in ("lm_id", "category", "main_class")):
            return [payload]
        if isinstance(payload, dict) and all(isinstance(v, dict) for v in payload.values()):
            return list(payload.values())
        raise ValueError("Unrecognized LIPID MAPS response; no annotation was changed.")

    def lookup(self, formula, identifiers=()):
        """Fetch candidates by LM identifier, falling back to the molecular formula."""
        parse_formula(formula)
        lm_ids = tuple(value for value in identifiers if value.startswith("LM"))
        records = [record for identifier in lm_ids for record in self._fetch("lm_id", identifier)] if lm_ids else self._fetch("formula", formula)
        candidates = []
        for record in records:
            identifier = str(record["lm_id"])
            hierarchy = self._fetch("lm_id", identifier, "classification")
            fields = {**record, **(hierarchy[0] if hierarchy else {})}
            classes = tuple(f"LIPID_MAPS:{fields[key]}" for key in
                            ("category", "main_class", "sub_class", "class_level4") if fields.get(key))
            candidates.append(ChemicalCandidate(identifier, str(record["formula"]), classes,
                                                record.get("smiles"), record.get("inchi_key")))
        logger.debug("LIPID MAPS returned %s candidates for one formula.", len(candidates))
        return tuple(candidates)
