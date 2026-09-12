"""Remote metabolite providers normalized to candidate compounds."""

from __future__ import annotations

import csv
from dataclasses import dataclass
import gzip
from io import StringIO, TextIOWrapper
from pathlib import Path
import tempfile
from typing import Any, Iterable, Protocol
from urllib.request import Request, urlopen
import xml.etree.ElementTree as ElementTree
import zipfile

from .model import CandidateClass, CandidateCompound
from ..chemistry.formula import parse_formula
from ...utils.logger import get_custom_logger


logger = get_custom_logger(__name__)


class CandidateProvider(Protocol):
    """Retrieve a complete external compound collection in canonical form."""

    name: str
    version: str

    def fetch_candidates(
        self,
        snapshot_path: Path | str | None = None,
    ) -> tuple[CandidateCompound, ...]:
        """Return provider data normalized to :class:`CandidateCompound`."""


@dataclass(frozen=True)
class LIPIDMAPSCandidateProvider:
    """Retrieve the public LIPID MAPS Structure Database table.

    :param version: Explicit release or retrieval-snapshot identifier.
    :type version: str
    :param timeout: HTTP timeout in seconds.
    :type timeout: float
    :param download_url: Public TSV export endpoint.
    :type download_url: str
    """

    version: str
    timeout: float = 60.0
    download_url: str = "https://www.lipidmaps.org/rest/compound/lm_id/LM/all/download"
    name: str = "LIPID_MAPS"

    def snapshot_filename(self) -> str:
        """Return the stable local filename of this source snapshot."""
        return f"lipidmaps_{_safe_filename_component(self.version)}.tsv"

    def materialize_snapshot(
        self,
        cache_dir: Path | str,
        *,
        refresh_cache: bool = False,
    ) -> Path:
        """Download the provider source once and retain it in ``cache_dir``."""
        return _materialize_snapshot(
            url=self.download_url,
            target=Path(cache_dir) / self.snapshot_filename(),
            timeout=self.timeout,
            refresh_cache=refresh_cache,
        )

    def fetch_candidates(
        self,
        snapshot_path: Path | str | None = None,
    ) -> tuple[CandidateCompound, ...]:
        """Normalize every public LIPID MAPS compound record."""
        if snapshot_path is None:
            response = _open_url(self.download_url, self.timeout)
            text = response.read().decode("utf-8-sig")
            response.close()
        else:
            text = Path(snapshot_path).read_text(encoding="utf-8-sig")
        rows = csv.DictReader(_lipidmaps_table_lines(text), delimiter="\t")
        candidates = tuple(
            candidate
            for row in rows
            if (candidate := _lipidmaps_candidate(row, self.version)) is not None
        )
        if not candidates:
            raise ValueError("LIPID MAPS download contained no usable compound records.")
        logger.info("Retrieved %s LIPID MAPS candidate compounds.", len(candidates))
        return candidates


@dataclass(frozen=True)
class HMDBCandidateProvider:
    """Retrieve the public HMDB metabolite XML export.

    HMDB's documented API requires separate access. This provider uses its
    public XML export and streams it into normalized records. The export can
    be retained as a local source snapshot for deterministic re-use.

    :param version: Explicit HMDB release identifier.
    :type version: str
    :param timeout: HTTP timeout in seconds.
    :type timeout: float
    :param download_url: Public HMDB metabolite ZIP export.
    :type download_url: str
    """

    version: str
    timeout: float = 300.0
    download_url: str = "https://hmdb.ca/system/downloads/current/hmdb_metabolites.zip"
    name: str = "HMDB"

    def snapshot_filename(self) -> str:
        """Return the stable local filename of this source snapshot."""
        return (
            f"hmdb_{_safe_filename_component(self.version)}/hmdb_metabolites.xml"
        )

    def materialize_snapshot(
        self,
        cache_dir: Path | str,
        *,
        refresh_cache: bool = False,
    ) -> Path:
        """Download and extract the provider source once into ``cache_dir``."""
        target = Path(cache_dir) / self.snapshot_filename()
        if target.is_file() and not refresh_cache:
            logger.info("Using local source snapshot at %s", target)
            return target
        archive_path = _materialize_temporary_snapshot(
            url=self.download_url,
            timeout=self.timeout,
            suffix=".zip",
        )
        try:
            return _extract_hmdb_xml(archive_path=archive_path, target=target)
        finally:
            archive_path.unlink(missing_ok=True)

    def fetch_candidates(
        self,
        snapshot_path: Path | str | None = None,
    ) -> tuple[CandidateCompound, ...]:
        """Stream-normalize the HMDB metabolite export."""
        if snapshot_path is None:
            archive_path = _materialize_temporary_snapshot(
                url=self.download_url,
                timeout=self.timeout,
                suffix=".zip",
            )
            try:
                candidates = _parse_hmdb_archive(archive_path, self.version)
            finally:
                archive_path.unlink(missing_ok=True)
        else:
            source = Path(snapshot_path)
            if source.suffix.lower() == ".zip":
                candidates = _parse_hmdb_archive(source, self.version)
            else:
                with source.open("rb") as stream:
                    candidates = tuple(_parse_hmdb_xml(stream, self.version))
        if not candidates:
            raise ValueError("HMDB download contained no usable compound records.")
        logger.info("Retrieved %s HMDB candidate compounds.", len(candidates))
        return candidates


@dataclass(frozen=True)
class ChEBICandidateProvider:
    """Retrieve the full ChEBI ontology and normalize chemical entities.

    The compressed full OBO export contains chemical fields as well as the
    ontology edges needed to preserve direct chemical-class relationships.

    :param version: Explicit ChEBI release or retrieval-snapshot identifier.
    :type version: str
    :param timeout: HTTP timeout in seconds.
    :type timeout: float
    :param download_url: Public ChEBI full OBO export endpoint.
    :type download_url: str
    """

    version: str
    timeout: float = 120.0
    download_url: str = (
        "https://ftp.ebi.ac.uk/pub/databases/chebi/ontology/chebi.obo.gz"
    )
    name: str = "ChEBI"

    def snapshot_filename(self) -> str:
        """Return the stable local filename of this source snapshot."""
        return f"chebi_{_safe_filename_component(self.version)}.obo.gz"

    def materialize_snapshot(
        self,
        cache_dir: Path | str,
        *,
        refresh_cache: bool = False,
    ) -> Path:
        """Download the provider source once and retain it in ``cache_dir``."""
        return _materialize_snapshot(
            url=self.download_url,
            target=Path(cache_dir) / self.snapshot_filename(),
            timeout=self.timeout,
            refresh_cache=refresh_cache,
        )

    def fetch_candidates(
        self,
        snapshot_path: Path | str | None = None,
    ) -> tuple[CandidateCompound, ...]:
        """Normalize formula-bearing ChEBI ontology terms."""
        response: Any | None = None
        if snapshot_path is None:
            response = _open_url(self.download_url, self.timeout)
            stream = TextIOWrapper(gzip.GzipFile(fileobj=response), encoding="utf-8")
        else:
            stream = gzip.open(Path(snapshot_path), mode="rt", encoding="utf-8")
        try:
            candidates = tuple(_parse_chebi_obo(stream, self.version))
        finally:
            stream.close()
            if response is not None:
                response.close()
        if not candidates:
            raise ValueError("ChEBI download contained no usable compound records.")
        logger.info("Retrieved %s ChEBI candidate compounds.", len(candidates))
        return candidates


def _open_url(url: str, timeout: float) -> Any:
    """Open one public provider URL with a stable descriptive user agent."""
    request = Request(url, headers={"User-Agent": "msi-dataset-manager/0.1"})
    return urlopen(request, timeout=timeout)


def _materialize_snapshot(
    *,
    url: str,
    target: Path,
    timeout: float,
    refresh_cache: bool,
) -> Path:
    """Download one source artifact atomically unless an existing one is reused."""
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_file() and not refresh_cache:
        logger.info("Using local source snapshot at %s", target)
        return target
    temporary = target.with_suffix(target.suffix + ".partial")
    temporary.unlink(missing_ok=True)
    try:
        response = _open_url(url, timeout)
        with temporary.open("wb") as stream:
            while chunk := response.read(1024 * 1024):
                stream.write(chunk)
        response.close()
        temporary.replace(target)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    logger.info("Downloaded source snapshot to %s", target)
    return target


def _materialize_temporary_snapshot(
    *,
    url: str,
    timeout: float,
    suffix: str,
) -> Path:
    """Download one ephemeral provider artifact for direct, uncached use."""
    handle = tempfile.NamedTemporaryFile(prefix="msi-provider-", suffix=suffix, delete=False)
    target = Path(handle.name)
    handle.close()
    try:
        response = _open_url(url, timeout)
        with target.open("wb") as stream:
            while chunk := response.read(1024 * 1024):
                stream.write(chunk)
        response.close()
    except Exception:
        target.unlink(missing_ok=True)
        raise
    return target


def _extract_hmdb_xml(*, archive_path: Path, target: Path) -> Path:
    """Extract the single HMDB XML document atomically from a ZIP archive."""
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".partial")
    temporary.unlink(missing_ok=True)
    try:
        with zipfile.ZipFile(archive_path) as archive:
            names = [name for name in archive.namelist() if name.endswith(".xml")]
            if len(names) != 1:
                raise ValueError("HMDB archive must contain exactly one metabolite XML file.")
            with archive.open(names[0]) as source, temporary.open("wb") as destination:
                while chunk := source.read(1024 * 1024):
                    destination.write(chunk)
        temporary.replace(target)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    logger.info("Extracted HMDB XML source snapshot to %s", target)
    return target


def _parse_hmdb_archive(path: Path, version: str) -> tuple[CandidateCompound, ...]:
    """Parse one HMDB ZIP export without writing its XML to disk."""
    with zipfile.ZipFile(path) as archive:
        names = [name for name in archive.namelist() if name.endswith(".xml")]
        if len(names) != 1:
            raise ValueError("HMDB archive must contain exactly one metabolite XML file.")
        with archive.open(names[0]) as stream:
            return tuple(_parse_hmdb_xml(stream, version))


def _safe_filename_component(value: str) -> str:
    """Return an ASCII filename component derived from a source version."""
    return "".join(character if character.isalnum() else "-" for character in value)


def _lipidmaps_candidate(
    row: dict[str, str],
    version: str,
) -> CandidateCompound | None:
    identifier = _value(row, "LM_ID", "lm_id")
    formula = _value(row, "FORMULA", "formula")
    name = _value(row, "NAME", "name", "SYSTEMATIC_NAME", "systematic_name")
    if not all((identifier, formula, name)):
        return None
    try:
        parse_formula(formula)
    except ValueError:
        return None
    hierarchy = (
        ("category", _value(row, "CATEGORY", "category", "CORE", "core")),
        ("main_class", _value(row, "MAIN_CLASS", "main_class")),
        ("sub_class", _value(row, "SUB_CLASS", "sub_class")),
        ("class_level4", _value(row, "CLASS_LEVEL4", "class_level4")),
    )
    classes = tuple(
        CandidateClass(
            namespace="LIPID_MAPS",
            identifier=f"LIPID_MAPS:{label}:{value}",
            name=value,
            depth=position,
        )
        for position, (label, value) in enumerate(hierarchy)
        if value
    )
    return CandidateCompound(
        provider="LIPID_MAPS",
        provider_version=version,
        identifier=identifier,
        name=name,
        formula=formula,
        monoisotopic_mass=_optional_float(_value(row, "EXACT_MASS", "exactmass")),
        smiles=_value(row, "SMILES", "smiles") or None,
        inchi=_value(row, "INCHI", "inchi") or None,
        inchi_key=_value(row, "INCHI_KEY", "inchi_key") or None,
        classes=classes,
        source_record=row,
    )


def _parse_hmdb_xml(stream: Any, version: str) -> Iterable[CandidateCompound]:
    """Stream HMDB metabolites without retaining the source XML tree."""
    for _, element in ElementTree.iterparse(stream, events=("end",)):
        if _tag(element) != "metabolite":
            continue
        values = {_tag(child): (child.text or "").strip() for child in element}
        identifier = values.get("accession", "")
        formula = values.get("chemical_formula", "")
        name = values.get("name", "")
        if identifier and formula and name:
            try:
                parse_formula(formula)
            except ValueError:
                pass
            else:
                taxonomy = next(
                    (child for child in element if _tag(child) == "taxonomy"),
                    None,
                )
                classes = _hmdb_classes(taxonomy)
                yield CandidateCompound(
                    provider="HMDB",
                    provider_version=version,
                    identifier=identifier,
                    name=name,
                    formula=formula,
                    monoisotopic_mass=_optional_float(
                        values.get("monisotopic_molecular_weight", "")
                    ),
                    smiles=values.get("smiles") or None,
                    inchi=values.get("inchi") or None,
                    inchi_key=values.get("inchikey") or None,
                    classes=classes,
                    source_record={
                        "accession": identifier,
                        "status": values.get("status"),
                        "update_date": values.get("update_date"),
                    },
                )
        element.clear()


def _parse_chebi_obo(stream: Iterable[str], version: str) -> Iterable[CandidateCompound]:
    """Stream formula-bearing ChEBI OBO terms into canonical compounds."""
    for term in _iter_obo_terms(stream):
        if term.get("is_obsolete") == "true":
            continue
        identifier = term.get("id", "")
        name = term.get("name", "")
        formula = _obo_property_value(term.get("formula", ""))
        if not identifier.startswith("CHEBI:") or not name or not formula:
            continue
        try:
            parse_formula(formula)
        except ValueError:
            continue
        parents = tuple(_obo_parent(value) for value in term.get("is_a", []))
        classes = tuple(
            CandidateClass(
                namespace="ChEBI",
                identifier=parent_identifier,
                name=parent_name,
                depth=depth,
            )
            for depth, (parent_identifier, parent_name) in enumerate(parents)
        )
        yield CandidateCompound(
            provider="ChEBI",
            provider_version=version,
            identifier=identifier,
            name=name,
            formula=formula,
            monoisotopic_mass=_optional_float(
                _obo_property_value(term.get("monoisotopicmass", ""))
            ),
            smiles=_obo_property_value(term.get("smiles", "")) or None,
            inchi=_obo_property_value(term.get("inchi", "")) or None,
            inchi_key=_obo_property_value(term.get("inchikey", "")) or None,
            classes=classes,
            source_record={
                "id": identifier,
                "parents": [parent_identifier for parent_identifier, _ in parents],
            },
        )


def _iter_obo_terms(stream: Iterable[str]) -> Iterable[dict[str, Any]]:
    """Yield minimal dictionaries for OBO ``[Term]`` blocks."""
    current: dict[str, Any] | None = None
    for raw_line in stream:
        line = raw_line.strip()
        if line == "[Term]":
            if current is not None:
                yield current
            current = {}
            continue
        if not line or line.startswith("["):
            if current is not None and not line:
                yield current
                current = None
            continue
        if current is None or ": " not in line:
            continue
        key, value = line.split(": ", maxsplit=1)
        if key == "property_value":
            property_name = _obo_property_key(value)
            canonical_name = {
                "formula": "formula",
                "generalized_empirical_formula": "formula",
                "monoisotopicmass": "monoisotopicmass",
                "monoisotopic_mass": "monoisotopicmass",
                "smiles": "smiles",
                "smiles_string": "smiles",
                "inchi": "inchi",
                "inchi_string": "inchi",
                "inchikey": "inchikey",
                "inchi_key_string": "inchikey",
            }.get(property_name)
            if canonical_name is not None:
                current[canonical_name] = value
        elif key == "is_a":
            current.setdefault(key, []).append(value)
        elif key in {"id", "name", "is_obsolete"}:
            current[key] = value
    if current is not None:
        yield current


def _obo_parent(value: str) -> tuple[str, str]:
    """Return the direct ChEBI parent identifier and display name from OBO."""
    identifier, _, name = value.partition(" ! ")
    return identifier, name or identifier


def _obo_property_key(value: str) -> str:
    """Return a source-independent final property-name component from OBO."""
    identifier = value.split(" ", maxsplit=1)[0]
    return identifier.rsplit("/", maxsplit=1)[-1].rsplit(":", maxsplit=1)[-1]


def _obo_property_value(value: str) -> str:
    """Return the quoted OBO property value without its datatype suffix."""
    _, separator, remainder = value.partition('"')
    if not separator:
        return ""
    quoted, separator, _ = remainder.partition('"')
    return quoted if separator else ""


def _hmdb_classes(taxonomy: ElementTree.Element | None) -> tuple[CandidateClass, ...]:
    """Map HMDB's taxonomy hierarchy into canonical class nodes."""
    if taxonomy is None:
        return ()
    levels = ("kingdom", "super_class", "class", "sub_class", "direct_parent")
    values = {_tag(child): (child.text or "").strip() for child in taxonomy}
    return tuple(
        CandidateClass(
            namespace="HMDB",
            identifier=f"HMDB:{level}:{values[level]}",
            name=values[level],
            depth=position,
        )
        for position, level in enumerate(levels)
        if values.get(level)
    )


def _tag(element: ElementTree.Element) -> str:
    """Return an XML local name without an HMDB namespace."""
    return element.tag.rsplit("}", maxsplit=1)[-1]


def _value(row: dict[str, str], *names: str) -> str:
    """Return the first populated spelling of one source column."""
    return next((str(row.get(name) or "").strip() for name in names if row.get(name)), "")


def _lipidmaps_table_lines(text: str) -> list[str]:
    """Return TSV lines beginning at the LIPID MAPS column header.

    Current LIPID MAPS exports prepend a one-line release date, whereas older
    exports begin directly with the header. The explicit header search supports
    both documented forms without treating the release date as a column name.
    """
    lines = text.splitlines()
    for index, line in enumerate(lines):
        fields = {field.strip().lower() for field in line.split("\t")}
        if {"lm_id", "formula"}.issubset(fields):
            return lines[index:]
    raise ValueError("LIPID MAPS export is missing its lm_id/formula TSV header.")


def _optional_float(value: str) -> float | None:
    """Return a finite provider mass when one is available."""
    try:
        return float(value) if value else None
    except ValueError:
        return None
