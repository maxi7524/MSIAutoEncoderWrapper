"""PRIDE Archive adapter backed by the official :mod:`pridepy` client."""

#TODO - implement methods wihch are already in metasapce (cache, download and coutning values) - jak już to dodajemy to zróbmy to do końca

from __future__ import annotations

import csv
import hashlib
import re
import tempfile
import xml.etree.ElementTree as ElementTree
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence
from urllib.parse import quote

from ....utils.exceptions import raise_external_service_error, raise_project_config_error
from ....utils.logger import get_custom_logger
from ..base import DatasetSource
from ..source_manager import DatasetSourceManager


logger = get_custom_logger(__name__)

_PAIR_SEPARATOR = "__"
_ANNOTATION_SUFFIXES = (
    ".annotations.csv",
    "_annotations.csv",
    ".annotations.tsv",
    "_annotations.tsv",
)
_MOLECULE_FIELDS = ("formula", "sumFormula", "annotation_id", "id", "molecule")
_MOLECULE_KEY_FIELDS = _MOLECULE_FIELDS + ("adduct", "mz")


@DatasetSourceManager.register_source("pride")
class PrideDatasetSource(DatasetSource):
    """Discover complete, annotated imzML pairs in PRIDE Archive.

    :param project_client: Optional injected ``pridepy`` project client.
    :type project_client: Any | None
    :param download_client: Optional injected ``pridepy`` download client.
    :type download_client: Any | None
    :param file_provider: Optional injected PRIDE file provider.
    :type file_provider: Any | None
    :param checksum_provider: Optional callable returning filename-to-MD5 mappings.
    :type checksum_provider: Any | None

    A PRIDE project may contain several MSI acquisitions. Each complete
    ``.imzML``/``.ibd`` pair is exposed as one logical dataset. Molecular
    annotations are accepted only from an explicit sidecar table; the adapter
    never infers molecule identities from measured m/z values.
    """

    source_name = "pride"

    def __init__(
        self,
        *,
        project_client: Optional[Any] = None,
        download_client: Optional[Any] = None,
        file_provider: Optional[Any] = None,
        checksum_provider: Optional[Any] = None,
    ) -> None:
        self._project_client = project_client
        self._download_client = download_client
        self._file_provider = file_provider
        self._checksum_provider = checksum_provider
        self._pairs: Dict[str, Dict[str, Any]] = {}
        self._download_directories: Dict[str, Path] = {}
        self._config: Dict[str, Any] = {}
        self._search_diagnostics: List[Dict[str, Any]] = []
        self._accepted_records: List[Dict[str, Any]] = []

    @staticmethod
    def get_available_filters() -> Dict[str, Any]:
        """Return notebook-friendly documentation for supported filters."""
        return {
            "keyword": {"type": "string", "default": "imzML"},
            "filter": {"type": "string", "description": "Raw PRIDE API filter."},
            "organisms": {"type": "list[string]", "api_field": "organisms"},
            "organism_parts": {
                "type": "list[string]",
                "api_field": "organismsPart",
            },
            "diseases": {"type": "list[string]", "api_field": "diseases"},
            "instruments": {"type": "list[string]", "api_field": "instruments"},
            "experiment_types": {
                "type": "list[string]",
                "api_field": "experimentTypes",
            },
            "submission_types": {
                "type": "list[string]",
                "api_field": "submissionType",
            },
            "publication_date_from": {"type": "YYYY-MM-DD | null"},
            "publication_date_to": {"type": "YYYY-MM-DD | null"},
            "required_metadata_fields": {"type": "list[string]"},
            "single_value_metadata_fields": {"type": "list[string]"},
            "max_pair_size_bytes": {"type": "integer | null"},
            "exclude_dataset_ids": {"type": "list[string]"},
            "require_annotation_source": {
                "type": "boolean",
                "required": True,
                "choices": [True, False],
            },
            "require_checksum": {
                "type": "boolean",
                "required": True,
                "choices": [True, False],
            },
            "page_size": {"type": "positive integer", "default": 100},
            "max_projects": {"type": "positive integer", "default": 500},
        }

    def get_rejected_records(self) -> List[Dict[str, Any]]:
        """Return records rejected by the most recent discovery call."""
        return [dict(record) for record in self._search_diagnostics]

    def get_accepted_records(self) -> List[Dict[str, Any]]:
        """Return image pairs accepted by the most recent filtering call."""
        return [dict(record) for record in self._accepted_records]

    @property
    def project_client(self) -> Any:
        """Return the injected project client or create the official client."""
        if self._project_client is None:
            try:
                from pridepy.project.project import Project
            except ImportError:
                self._missing_pridepy()
            self._project_client = Project()
        return self._project_client

    @property
    def download_client(self) -> Any:
        """Return the injected download client or create the official client."""
        if self._download_client is None:
            try:
                from pridepy.download.client import Client
            except ImportError:
                self._missing_pridepy()
            self._download_client = Client()
        return self._download_client

    @property
    def file_provider(self) -> Any:
        """Return the injected file provider or create ``PrideProvider``."""
        if self._file_provider is None:
            try:
                from pridepy.download.pride import PrideProvider
            except ImportError:
                self._missing_pridepy()
            self._file_provider = PrideProvider()
        return self._file_provider

    def filter(self, filters: Mapping[str, Any]) -> List[Dict[str, Any]]:
        """Return complete annotated PRIDE image pairs matching all filters.

        :param filters: PRIDE project, file, and metadata filter configuration.
        :type filters: Mapping[str, Any]
        :return: Provider records, one per accepted imzML/ibd pair.
        :rtype: List[Dict[str, Any]]
        """
        options = _SearchOptions(filters)
        self._config = dict(filters)
        self._search_diagnostics = []
        records: List[Dict[str, Any]] = []
        rejected = 0
        for project in self._search_projects(options):
            accession = str(project.get("accession", ""))
            if not accession:
                rejected += 1
                logger.warning("Rejected PRIDE search record without an accession")
                continue
            project_metadata = self.project_client.get_by_accession(accession)
            reason = _metadata_rejection_reason(project_metadata, options)
            if reason:
                rejected += 1
                _log_rejection(accession, reason, self._search_diagnostics)
                continue
            files = list(self.file_provider.list_files(accession))
            checksums = self._get_checksums(accession) if options.require_checksum else {}
            project_records, project_rejected = _pair_records(
                project_metadata,
                files,
                checksums,
                options,
                self._search_diagnostics,
            )
            rejected += project_rejected
            for record in project_records:
                dataset_id = str(record["dataset_id"])
                if dataset_id in options.exclude_dataset_ids:
                    rejected += 1
                    _log_rejection(
                        accession,
                        f"dataset excluded by configuration: {dataset_id}",
                        self._search_diagnostics,
                        dataset_id=dataset_id,
                    )
                    continue
                self._pairs[dataset_id] = record["metadata"]["pride_files"]
                records.append(record)
        logger.info(
            "PRIDE discovery accepted %s image pairs and rejected %s records",
            len(records),
            rejected,
        )
        self._accepted_records = records
        return self.get_accepted_records()

    def get_dataset_metadata(self, dataset_id: str) -> Dict[str, Any]:
        """Return complete project and file metadata for one PRIDE image pair."""
        pair = self._resolve_pair(dataset_id)
        project = self.project_client.get_by_accession(pair["project_accession"])
        return _dataset_record(project, pair)

    def get_annotations(
        self,
        dataset_id: str,
        options: Optional[Mapping[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Read curated molecule-to-pixel links from the downloaded sidecar.

        Supported CSV/TSV tables must contain a molecule identity (``formula``,
        ``sumFormula``, ``annotation_id``, ``id``, or ``molecule``) and a
        ``spectrum_id`` column. Unsupported schemas fail visibly.
        """
        del options
        pair = self._resolve_pair(dataset_id)
        directory = self._download_directories.get(dataset_id)
        if directory is None:
            raise_project_config_error(
                "PrideSource",
                f"Dataset '{dataset_id}' must be downloaded before loading annotations.",
            )
        annotation_name = pair["annotation"]["fileName"]
        annotation_path = directory / annotation_name
        records = _read_annotation_table(annotation_path)
        logger.info("Loaded %s PRIDE annotation rows for %s", len(records), dataset_id)
        return records

    def download_dataset(self, dataset_id: str, destination: Path | str) -> Path:
        """Download and validate one imzML/ibd pair and its annotation table."""
        pair = self._resolve_pair(dataset_id)
        target = Path(destination)
        target.mkdir(parents=True, exist_ok=True)
        file_names = [
            pair["imzml"]["fileName"],
            pair["ibd"]["fileName"],
            pair["annotation"]["fileName"],
        ]
        try:
            self.download_client.download_files_by_list(
                accession=pair["project_accession"],
                file_names=file_names,
                output_folder=str(target),
                skip_if_downloaded_already=True,
                protocol=str(self._config.get("protocol", "ftp")),
                checksum_check=bool(self._config.get("require_checksum", True)),
                parallel_files=int(self._config.get("parallel_files", 1)),
                download_threads=int(self._config.get("download_threads", 1)),
            )
        except Exception as error:
            raise_external_service_error(
                "PRIDE",
                f"Download failed for dataset '{dataset_id}': {error}",
            )
        source_imzml = target / pair["imzml"]["fileName"]
        source_ibd = target / pair["ibd"]["fileName"]
        canonical_imzml = target / f"{dataset_id}.imzML"
        canonical_ibd = target / f"{dataset_id}.ibd"
        if source_imzml != canonical_imzml:
            source_imzml.replace(canonical_imzml)
        if source_ibd != canonical_ibd:
            source_ibd.replace(canonical_ibd)
        _validate_pair_uuid(canonical_imzml, canonical_ibd)
        self._download_directories[dataset_id] = target
        return target

    def _search_projects(self, options: "_SearchOptions") -> Iterable[Mapping[str, Any]]:
        yielded = 0
        page = 0
        while yielded < options.max_projects:
            page_records = self.project_client.search_by_keywords_and_filters(
                keyword=quote(options.keyword),
                query_filter=quote(options.api_filter, safe="=,[]|"),
                page_size=options.page_size,
                page=page,
                sort_direction=options.sort_direction,
                sort_fields=options.sort_fields,
            )
            if not isinstance(page_records, list):
                raise_external_service_error(
                    "PRIDE", "Project search returned a non-list response."
                )
            for project in page_records:
                if yielded >= options.max_projects:
                    return
                yielded += 1
                yield project
            if len(page_records) < options.page_size:
                return
            page += 1

    def _get_checksums(self, accession: str) -> Dict[str, str]:
        if self._checksum_provider is not None:
            return dict(self._checksum_provider(accession))
        try:
            from pridepy.download.client import Client
            from pridepy.download.pride import PrideProvider
        except ImportError:
            self._missing_pridepy()
        with tempfile.TemporaryDirectory(prefix="pride-checksums-") as directory:
            path = PrideProvider.save_checksum_file(accession, directory)
            return Client.read_checksum_file(path)

    def _resolve_pair(self, dataset_id: str) -> Dict[str, Any]:
        cached = self._pairs.get(dataset_id)
        if cached is not None:
            return cached
        accession = dataset_id.split(_PAIR_SEPARATOR, 1)[0]
        project = self.project_client.get_by_accession(accession)
        files = list(self.file_provider.list_files(accession))
        checksums = (
            self._get_checksums(accession)
            if self._config.get("require_checksum", True)
            else {}
        )
        options = _SearchOptions(self._config)
        records, _ = _pair_records(project, files, checksums, options, [])
        for record in records:
            pair = record["metadata"]["pride_files"]
            self._pairs[str(record["dataset_id"])] = pair
        if dataset_id not in self._pairs:
            raise_project_config_error(
                "PrideSource",
                f"PRIDE dataset '{dataset_id}' is unavailable or no longer satisfies its filters.",
            )
        return self._pairs[dataset_id]

    @staticmethod
    def _missing_pridepy() -> None:
        raise_project_config_error(
            "PrideSource",
            "The official pridepy client is not installed. Install pridepy==0.0.16.",
        )


class _SearchOptions:
    """Validated PRIDE discovery settings."""

    def __init__(self, values: Mapping[str, Any]) -> None:
        self.keyword = str(values.get("keyword", "imzML"))
        self.page_size = int(values.get("page_size", 100))
        self.max_projects = int(values.get("max_projects", 500))
        self.sort_direction = str(values.get("sort_direction", "DESC"))
        self.sort_fields = str(values.get("sort_fields", "submissionDate"))
        self.require_checksum = bool(values.get("require_checksum", True))
        self.require_annotation_source = bool(values.get("require_annotation_source", True))
        self.max_pair_size_bytes = _optional_int(values.get("max_pair_size_bytes"))
        self.required_metadata_fields = tuple(values.get("required_metadata_fields", ()))
        self.single_value_metadata_fields = tuple(values.get("single_value_metadata_fields", ()))
        self.exclude_dataset_ids = {str(value) for value in values.get("exclude_dataset_ids", ())}
        self.api_filter = _build_api_filter(values)
        if self.page_size <= 0 or self.max_projects <= 0:
            raise_project_config_error(
                "PrideSource", "page_size and max_projects must be positive."
            )
        if not self.require_annotation_source:
            raise_project_config_error(
                "PrideSource",
                "require_annotation_source must remain true because unannotated "
                "PRIDE images are unsupported.",
            )
        if not self.require_checksum:
            raise_project_config_error(
                "PrideSource",
                "require_checksum must remain true so every downloaded file can be verified.",
            )


def _build_api_filter(values: Mapping[str, Any]) -> str:
    explicit = values.get("filter")
    if explicit:
        return str(explicit)
    mappings = {
        "organisms": "organisms",
        "organism_parts": "organismsPart",
        "diseases": "diseases",
        "instruments": "instruments",
        "experiment_types": "experimentTypes",
        "submission_types": "submissionType",
    }
    clauses = []
    for config_name, api_name in mappings.items():
        selected = values.get(config_name)
        if selected:
            clauses.append(f"{api_name}=={' '.join(str(item) for item in selected)}")
    date_from = values.get("publication_date_from")
    date_to = values.get("publication_date_to")
    if date_from or date_to:
        clauses.append(f"publicationDate=range=[{date_from or '*'} TO {date_to or '*'}]")
    return ",".join(clauses)


def _metadata_rejection_reason(
    project: Mapping[str, Any], options: _SearchOptions
) -> Optional[str]:
    for field in options.required_metadata_fields:
        value = project.get(field)
        if value is None or value == "" or value == []:
            return f"missing required metadata field '{field}'"
    for field in options.single_value_metadata_fields:
        value = project.get(field)
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)) and len(value) > 1:
            return f"ambiguous metadata field '{field}' contains {len(value)} values"
    return None


def _pair_records(
    project: Mapping[str, Any],
    files: Sequence[Mapping[str, Any]],
    checksums: Mapping[str, str],
    options: _SearchOptions,
    diagnostics: List[Dict[str, Any]],
) -> tuple[List[Dict[str, Any]], int]:
    accession = str(project["accession"])
    by_stem: Dict[str, Dict[str, List[Mapping[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    annotation_files: Dict[str, Mapping[str, Any]] = {}
    for file_record in files:
        name = str(file_record.get("fileName", ""))
        lower_name = name.casefold()
        if lower_name.endswith(".imzml"):
            by_stem[lower_name[:-6]]["imzml"].append(file_record)
        elif lower_name.endswith(".ibd"):
            by_stem[lower_name[:-4]]["ibd"].append(file_record)
        elif lower_name.endswith((".csv", ".tsv")):
            annotation_files[lower_name] = file_record

    accepted: List[Dict[str, Any]] = []
    rejected = 0
    for normalized_stem, parts in sorted(by_stem.items()):
        imzml_name = (parts.get("imzml") or [{}])[0].get(
            "fileName", normalized_stem
        )
        display_stem = Path(str(imzml_name)).stem
        if len(parts.get("imzml", [])) != 1 or len(parts.get("ibd", [])) != 1:
            rejected += 1
            _log_rejection(
                accession,
                f"incomplete or ambiguous imzML/ibd pair '{display_stem}'",
                diagnostics,
            )
            continue
        annotation = _annotation_for_stem(normalized_stem, annotation_files)
        if annotation is None and options.require_annotation_source:
            rejected += 1
            candidates = ", ".join(
                sorted(record["fileName"] for record in annotation_files.values())
            )
            detail = f"; tabular project files: {candidates}" if candidates else ""
            _log_rejection(
                accession,
                f"unsupported annotation format for image '{display_stem}'{detail}",
                diagnostics,
            )
            continue
        imzml = dict(parts["imzml"][0])
        ibd = dict(parts["ibd"][0])
        selected_files = [imzml, ibd] + ([dict(annotation)] if annotation else [])
        missing_checksums = [
            item["fileName"]
            for item in selected_files
            if item["fileName"] not in checksums
        ]
        if options.require_checksum and missing_checksums:
            rejected += 1
            _log_rejection(
                accession,
                f"missing checksums for: {', '.join(missing_checksums)}",
                diagnostics,
            )
            continue
        pair_size = sum(int(item.get("fileSizeBytes") or 0) for item in (imzml, ibd))
        if options.max_pair_size_bytes is not None and pair_size > options.max_pair_size_bytes:
            rejected += 1
            _log_rejection(
                accession,
                f"image '{display_stem}' exceeds maximum size ({pair_size} bytes)",
                diagnostics,
            )
            continue
        dataset_id = _dataset_id(accession, display_stem)
        pair = {
            "project_accession": accession,
            "imzml": imzml,
            "ibd": ibd,
            "annotation": dict(annotation) if annotation else None,
            "checksums": {
                item["fileName"]: checksums.get(item["fileName"])
                for item in selected_files
            },
        }
        accepted.append(_dataset_record(project, pair, dataset_id=dataset_id, total_size=pair_size))
    return accepted, rejected


def _annotation_for_stem(
    normalized_stem: str,
    files: Mapping[str, Mapping[str, Any]],
) -> Optional[Mapping[str, Any]]:
    matches = [files.get(f"{normalized_stem}{suffix}") for suffix in _ANNOTATION_SUFFIXES]
    present = [match for match in matches if match is not None]
    return present[0] if len(present) == 1 else None


def _dataset_record(
    project: Mapping[str, Any],
    pair: Mapping[str, Any],
    *,
    dataset_id: Optional[str] = None,
    total_size: Optional[int] = None,
) -> Dict[str, Any]:
    stem = Path(str(pair["imzml"]["fileName"])).stem
    resolved_id = dataset_id or _dataset_id(str(project["accession"]), stem)
    selected_files = [pair["imzml"], pair["ibd"]] + (
        [pair["annotation"]] if pair.get("annotation") else []
    )
    resolved_size = total_size if total_size is not None else sum(
        int(item.get("fileSizeBytes") or 0) for item in selected_files
    )
    return {
        "dataset_id": resolved_id,
        "name": stem,
        "metadata": {
            "project_accession": project["accession"],
            "project_url": f"https://www.ebi.ac.uk/pride/archive/projects/{project['accession']}",
            "project": dict(project),
            "image_stem": stem,
            "imzml_size_bytes": int(pair["imzml"].get("fileSizeBytes") or 0),
            "ibd_size_bytes": int(pair["ibd"].get("fileSizeBytes") or 0),
            "annotation_size_bytes": int((pair.get("annotation") or {}).get("fileSizeBytes") or 0),
            "total_size_bytes": resolved_size,
            "download_size_bytes": resolved_size
            + int((pair.get("annotation") or {}).get("fileSizeBytes") or 0),
            "annotation_status": "supported" if pair.get("annotation") else "missing",
            "pride_files": dict(pair),
        },
    }


def _dataset_id(accession: str, stem: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", stem).strip("-._") or "image"
    digest = hashlib.sha256(stem.encode("utf-8")).hexdigest()[:10]
    return f"{accession}{_PAIR_SEPARATOR}{slug[:80]}-{digest}"


def _read_annotation_table(path: Path) -> List[Dict[str, Any]]:
    if not path.is_file():
        raise_project_config_error("PrideAnnotations", f"Annotation file is missing: '{path}'.")
    delimiter = "\t" if path.suffix.casefold() == ".tsv" else ","
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = [dict(row) for row in csv.DictReader(stream, delimiter=delimiter)]
    if not rows:
        raise_project_config_error("PrideAnnotations", f"Annotation table is empty: '{path.name}'.")
    fields = set(rows[0])
    if not fields.intersection(_MOLECULE_FIELDS):
        raise_project_config_error(
            "PrideAnnotations",
            f"Unsupported annotation format in '{path.name}': molecule identity column is missing.",
        )
    if "spectrum_id" not in fields:
        raise_project_config_error(
            "PrideAnnotations",
            f"Unsupported annotation format in '{path.name}': expected a spectrum_id column.",
        )
    grouped: Dict[tuple[str, ...], Dict[str, Any]] = {}
    for position, row in enumerate(rows):
        identity = tuple(str(row.get(field, "")) for field in _MOLECULE_KEY_FIELDS)
        record = grouped.setdefault(
            identity,
            {
                **row,
                "annotation_id": row.get("annotation_id")
                or row.get("id")
                or str(position),
                "spectrum_ids": [],
            },
        )
        record["spectrum_ids"].append(int(row["spectrum_id"]))
    return list(grouped.values())


def _validate_pair_uuid(imzml_path: Path, ibd_path: Path) -> None:
    uuid_value = None
    for _, element in ElementTree.iterparse(imzml_path, events=("start",)):
        if element.attrib.get("accession") == "IMS:1000080":
            uuid_value = element.attrib.get("value")
            break
    if not uuid_value:
        raise_project_config_error("PrideSource", f"imzML UUID is missing in '{imzml_path.name}'.")
    expected = re.sub(r"[^0-9a-fA-F]", "", uuid_value).lower()
    with ibd_path.open("rb") as stream:
        actual = stream.read(16).hex()
    if expected != actual:
        raise_project_config_error(
            "PrideSource", f"imzML/ibd UUID mismatch for '{imzml_path.stem}'."
        )


def _optional_int(value: Any) -> Optional[int]:
    return None if value is None else int(value)


def _log_rejection(
    accession: str,
    reason: str,
    diagnostics: List[Dict[str, Any]],
    *,
    dataset_id: Optional[str] = None,
) -> None:
    project_url = f"https://www.ebi.ac.uk/pride/archive/projects/{accession}"
    diagnostics.append(
        {
            "project_accession": accession,
            "dataset_id": dataset_id,
            "reason": reason,
            "project_url": project_url,
        }
    )
    logger.warning(
        "Rejected PRIDE record %s: %s (%s)",
        accession,
        reason,
        project_url,
    )
