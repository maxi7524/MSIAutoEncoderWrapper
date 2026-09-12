"""Durable local snapshots of external candidate-provider exports."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Protocol, Sequence

from ...utils.logger import get_custom_logger


SOURCE_MANIFEST_FILENAME = "sources_manifest.json"
SOURCE_MANIFEST_SCHEMA_VERSION = 1
DEFAULT_CANDIDATE_SOURCE_CACHE_DIR = Path("assets/local/metabolite_databases")
logger = get_custom_logger(__name__)


class SnapshotProvider(Protocol):
    """Provider interface required for durable source materialization."""

    name: str
    version: str
    download_url: str

    def materialize_snapshot(
        self,
        cache_dir: Path | str,
        *,
        refresh_cache: bool = False,
    ) -> Path:
        """Return a local snapshot, downloading only when necessary."""


def materialize_candidate_sources(
    *,
    cache_dir: Path | str,
    providers: Sequence[SnapshotProvider],
    refresh_cache: bool = False,
) -> tuple[Path, ...]:
    """Download or re-use all provider exports and publish their manifest.

    :param cache_dir: Directory retaining raw, versioned provider exports.
    :type cache_dir: pathlib.Path or str
    :param providers: Providers with a durable snapshot implementation.
    :type providers: collections.abc.Sequence[SnapshotProvider]
    :param refresh_cache: Replace every already-cached export when true.
    :type refresh_cache: bool
    :returns: Absolute paths of provider source snapshots in provider order.
    :rtype: tuple[pathlib.Path, ...]
    :raises ValueError: If no provider is supplied.
    """
    if not providers:
        raise ValueError("At least one candidate provider is required.")
    target_directory = Path(cache_dir).resolve()
    target_directory.mkdir(parents=True, exist_ok=True)

    snapshots = tuple(
        provider.materialize_snapshot(
            target_directory,
            refresh_cache=refresh_cache,
        ).resolve()
        for provider in providers
    )
    manifest_path = target_directory / SOURCE_MANIFEST_FILENAME
    existing_sources = {
        (str(source["name"]), str(source["version"])): source
        for source in _read_manifest(manifest_path).get("sources", [])
        if isinstance(source, dict)
        and source.get("name") is not None
        and source.get("version") is not None
    }
    for provider, path in zip(providers, snapshots, strict=True):
        existing_sources[(provider.name, provider.version)] = {
            "name": provider.name,
            "version": provider.version,
            "source_url": provider.download_url,
            "local_path": str(path),
            "size_bytes": path.stat().st_size,
        }
    manifest = {
        "schema_version": SOURCE_MANIFEST_SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "sources": sorted(
            existing_sources.values(),
            key=lambda source: (str(source["name"]), str(source["version"])),
        ),
    }
    _write_manifest(manifest_path, manifest)
    logger.info(
        "Materialized %s external candidate-source snapshots in %s.",
        len(snapshots),
        target_directory,
    )
    return snapshots


def _write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    """Write the source provenance manifest atomically."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        temporary.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _read_manifest(path: Path) -> dict[str, Any]:
    """Return an existing source manifest or an empty manifest structure."""
    if not path.is_file():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else {}
