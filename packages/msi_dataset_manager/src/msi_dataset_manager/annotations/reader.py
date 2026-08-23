"""Public annotation-reader factory owned by the dataset manager."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

from ..sources.source_manager import DatasetSourceManager
from ..utils.exceptions import raise_validation_error
from .merge.reader import MergedAnnotationReader
from .index import SpectrumAnnotationIndex, build_annotation_index


class AnnotationReader:
    """Create a normalized source or merged annotation reader."""

    @staticmethod
    def load(
        *,
        type: str,
        path: Path | str,
        source: Optional[str] = None,
        dataset_id: Optional[str] = None,
        image_path: Path | str | None = None,
        default_filters: Optional[Mapping[str, Any]] = None,
    ) -> Any:
        """Load exactly one supported annotation representation.

        :param type: ``source`` for source-owned exports or ``merge`` for SQLite.
        :type type: str
        :param path: Source dataset directory or merged SQLite path.
        :type path: pathlib.Path | str
        :param source: Registered source adapter for ``type='source'``.
        :type source: str | None
        :param dataset_id: Source dataset identifier. Defaults to the imzML stem.
        :type dataset_id: str | None
        :param image_path: Explicit source or merged imzML path.
        :type image_path: pathlib.Path | str | None
        :param default_filters: Filters applied by default to read operations.
        :type default_filters: Mapping[str, Any] | None
        :return: Reader implementing the normalized annotation interface.
        :rtype: Any
        """
        reader_type = str(type)
        if reader_type == "merge":
            return MergedAnnotationReader(
                path,
                image_path=image_path,
                default_filters=default_filters,
            )
        if reader_type != "source":
            raise_validation_error(
                "AnnotationReader",
                "type must be either 'source' or 'merge'.",
            )
        if source is None:
            raise_validation_error(
                "AnnotationReader",
                "source is required when type='source'.",
            )
        directory = Path(path)
        resolved_image = _resolve_source_image(directory, image_path, dataset_id)
        resolved_dataset_id = str(dataset_id or resolved_image.stem)
        export = DatasetSourceManager.read_annotation_export(
            source=source,
            dataset_id=resolved_dataset_id,
            directory=directory,
            imzml_path=resolved_image,
        )
        return SourceAnnotationReader(
            export=export,
            image_path=resolved_image,
            default_filters=default_filters,
        )


class SourceAnnotationReader:
    """Expose one normalized source export through the common reader API."""

    def __init__(
        self,
        *,
        export: Any,
        image_path: Path,
        default_filters: Optional[Mapping[str, Any]] = None,
    ) -> None:
        self.export = export
        self.image_path = image_path
        self.default_filters = dict(default_filters or {})
        self.active_context: Any = None
        self._spectrum_annotation_indices: Dict[
            tuple[tuple[int, ...] | None, tuple[tuple[str, str], ...]],
            SpectrumAnnotationIndex,
        ] = {}
        self._config = {
            "type": "source",
            "path": str(image_path.parent),
            "source": export.source,
            "dataset_id": export.dataset_id,
            "image_path": str(image_path),
            "default_filters": self.default_filters,
        }

    def get_dataset_metadata(self) -> Dict[str, Any]:
        """Return normalized dataset metadata retained by the source adapter."""
        return {
            "source": self.export.source,
            "dataset_id": self.export.dataset_id,
            "name": self.export.metadata.get("name", self.image_path.stem),
            "image_path": str(self.image_path),
            "metadata": dict(self.export.metadata),
        }

    def get_annotations(
        self,
        filters: Optional[Mapping[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Return source references after applying explicit filters."""
        effective = {**self.default_filters, **dict(filters or {})}
        return [
            dict(record)
            for record in self.export.records
            if _matches_source_filters(record, effective)
        ]

    def get_spectrum_annotations(
        self,
        spectrum_id: int,
        filters: Optional[Mapping[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Return source references present in one source spectrum."""
        target = int(spectrum_id)
        return [
            record
            for record in self.get_annotations(filters)
            if target in record.get("spectrum_ids", ())
        ]

    def get_spectrum_annotation_index(
        self,
        spectrum_ids: Any,
        filters: Optional[Mapping[str, Any]] = None,
    ) -> SpectrumAnnotationIndex:
        """Return all requested source annotations in one compact bulk index.

        :param spectrum_ids: Spectrum identifiers to represent, including rows
            without annotations. ``None`` stores annotated rows only.
        :type spectrum_ids: Iterable[int] | None
        :param filters: Optional source-reference filters.
        :type filters: Mapping[str, Any] | None
        :return: Cached CSR spectrum annotation index.
        :rtype: SpectrumAnnotationIndex
        """
        selected = (
            None
            if spectrum_ids is None
            else tuple(sorted({int(value) for value in spectrum_ids}))
        )
        effective = {**self.default_filters, **dict(filters or {})}
        filter_key = tuple(sorted((str(key), repr(value)) for key, value in effective.items()))
        cache_key = (selected, filter_key)
        cached = self._spectrum_annotation_indices.get(cache_key)
        if cached is not None:
            return cached
        selected_set = set(selected) if selected is not None else None
        entries: Dict[int, List[tuple[tuple[str, str], float]]] = {}
        for record in self.get_annotations(filters):
            identity = (str(record.get("formula", "")), str(record.get("adduct", "")))
            mz = record.get("mz")
            resolved_mz = float(mz) if mz is not None else float("nan")
            for spectrum_id in record.get("spectrum_ids", ()):
                resolved_id = int(spectrum_id)
                if selected_set is None or resolved_id in selected_set:
                    entries.setdefault(resolved_id, []).append((identity, resolved_mz))
        represented_ids = list(entries) if selected is None else list(selected)
        index = build_annotation_index(represented_ids, entries)
        self._spectrum_annotation_indices[cache_key] = index
        return index

    def get_spectrum_metadata(self, spectrum_id: int) -> Dict[str, Any]:
        """Return dataset metadata and the requested source spectrum index."""
        metadata = self.get_dataset_metadata()
        metadata["source_spectrum_id"] = int(spectrum_id)
        return metadata

    def get_spectrum_mask(self, spectrum_id: int, mask: str) -> bool:
        """Return the built-in annotation-presence mask for one source pixel."""
        if mask not in {"annotated", "annotation"}:
            raise_validation_error("SourceAnnotationReader", f"Unknown mask '{mask}'.")
        return bool(self.get_spectrum_annotations(int(spectrum_id)))

    def get_config(self) -> Dict[str, Any]:
        """Return the reader configuration used by wrapper context persistence."""
        return dict(self._config)


def _resolve_source_image(
    directory: Path,
    image_path: Path | str | None,
    dataset_id: Optional[str],
) -> Path:
    if image_path is not None:
        candidate = Path(image_path)
    elif dataset_id is not None:
        candidate = directory / f"{dataset_id}.imzML"
    else:
        candidates = sorted(directory.glob("*.imzML"))
        if len(candidates) != 1:
            raise_validation_error(
                "AnnotationReader",
                f"Expected one imzML file in '{directory}', found {len(candidates)}.",
            )
        candidate = candidates[0]
    if not candidate.is_file() or not candidate.with_suffix(".ibd").is_file():
        raise_validation_error(
            "AnnotationReader",
            f"Incomplete source imzML/ibd pair: '{candidate}'.",
        )
    return candidate


def _matches_source_filters(
    record: Mapping[str, Any],
    filters: Mapping[str, Any],
) -> bool:
    max_fdr = filters.get("max_fdr")
    if max_fdr is not None and (
        record.get("fdr") is None or float(record["fdr"]) > float(max_fdr)
    ):
        return False
    return all(
        key == "max_fdr" or record.get(key) == value
        for key, value in filters.items()
    )
