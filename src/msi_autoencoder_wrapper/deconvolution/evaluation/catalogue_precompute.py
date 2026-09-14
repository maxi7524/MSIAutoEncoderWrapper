"""Materialize a reproducible candidate catalogue for deconvolution notebooks."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Mapping

import yaml

from msi_dataset_manager.annotations.candidates import (
    ChEBICandidateProvider,
    HMDBCandidateProvider,
    LIPIDMAPSCandidateProvider,
    build_candidate_catalog,
)
from msi_dataset_manager.metadata import write_dataset_metadata

from ...utils.logger import get_custom_logger


logger = get_custom_logger(__name__)


def read_settings(path: Path | str) -> dict[str, Any]:
    """Load the shared deconvolution notebook settings file.

    :param path: YAML settings path.
    :type path: pathlib.Path | str
    :return: Parsed settings mapping.
    :rtype: dict[str, typing.Any]
    :raises ValueError: If the file does not contain a YAML mapping.
    """
    settings_path = Path(path)
    with settings_path.open(encoding="utf-8") as stream:
        settings = yaml.safe_load(stream)
    if not isinstance(settings, dict):
        raise ValueError("Deconvolution analysis settings must be a YAML mapping.")
    return settings


def catalogue_path(settings: Mapping[str, Any], *, notebook_dir: Path | str) -> Path:
    """Resolve the cached catalogue path for one notebook campaign.

    :param settings: Shared notebook settings mapping.
    :param notebook_dir: Directory containing the notebook and its result folders.
    :type settings: collections.abc.Mapping[str, typing.Any]
    :type notebook_dir: pathlib.Path | str
    :return: Expected ``candidates.sqlite`` path.
    :rtype: pathlib.Path
    :raises ValueError: If required catalogue settings are absent.
    """
    catalogue = settings.get("catalogue")
    if not isinstance(catalogue, Mapping):
        raise ValueError("analysis settings require a catalogue mapping.")
    workspace_relative_path = catalogue.get("workspace_relative_path")
    dataset_id = catalogue.get("dataset_id")
    if not isinstance(workspace_relative_path, str) or not isinstance(dataset_id, str):
        raise ValueError("catalogue workspace_relative_path and dataset_id are required.")
    return (
        Path(notebook_dir)
        / workspace_relative_path
        / "datasets"
        / dataset_id
        / "database_annotations"
        / "candidates.sqlite"
    )


def precompute_catalogue(
    settings: Mapping[str, Any],
    *,
    settings_path: Path | str,
) -> Path:
    """Build one cached catalogue from the configured local provider snapshots.

    :param settings: Shared notebook settings mapping.
    :param settings_path: Source YAML path used to resolve repository-relative
        source-cache and workspace paths.
    :type settings: collections.abc.Mapping[str, typing.Any]
    :type settings_path: pathlib.Path | str
    :return: Materialized dataset-specific candidate catalogue path.
    :rtype: pathlib.Path
    """
    source_path = Path(settings_path).resolve()
    repository_root = next(
        path for path in (source_path.parent, *source_path.parents)
        if (path / "pyproject.toml").is_file()
    )
    catalogue = settings.get("catalogue")
    if not isinstance(catalogue, Mapping):
        raise ValueError("analysis settings require a catalogue mapping.")
    source_cache_relative_path = catalogue.get("source_cache_relative_path")
    workspace_relative_path = catalogue.get("workspace_relative_path")
    dataset_id = catalogue.get("dataset_id")
    mz_min = catalogue.get("mz_min")
    mz_max = catalogue.get("mz_max")
    adducts = catalogue.get("adducts")
    providers = catalogue.get("providers")
    if not all(isinstance(value, str) for value in (source_cache_relative_path, workspace_relative_path, dataset_id)):
        raise ValueError("catalogue path and dataset settings must be strings.")
    if not isinstance(adducts, list) or not all(isinstance(value, str) for value in adducts):
        raise ValueError("catalogue adducts must be a list of strings.")
    if not isinstance(providers, Mapping):
        raise ValueError("catalogue providers must be a mapping of source versions.")
    workspace_path = source_path.parent / workspace_relative_path
    source_cache_dir = repository_root / source_cache_relative_path
    logger.info(
        "Materializing deconvolution candidate catalogue from local sources at %s.",
        source_cache_dir,
    )
    write_dataset_metadata(
        workspace_path=workspace_path,
        source="metaspace",
        dataset_id=dataset_id,
        name="annotation-base-initial",
        metadata={"mz_min": float(mz_min), "mz_max": float(mz_max)},
    )
    result = build_candidate_catalog(
        workspace_path=workspace_path,
        dataset_id=dataset_id,
        providers=(
            LIPIDMAPSCandidateProvider(version=str(providers["lipidmaps"])),
            HMDBCandidateProvider(version=str(providers["hmdb"])),
            ChEBICandidateProvider(version=str(providers["chebi"])),
        ),
        adducts=tuple(adducts),
        source_cache_dir=source_cache_dir,
    )
    logger.info("Materialized deconvolution candidate catalogue at %s.", result)
    return result


def run_command(settings_path: Path | str) -> str:
    """Return the canonical background-capable catalogue materialization command.

    :param settings_path: YAML settings path.
    :type settings_path: pathlib.Path | str
    :return: Shell command that performs the reproducible precomputation.
    :rtype: str
    """
    return (
        "uv run python -m msi_autoencoder_wrapper.deconvolution.evaluation.catalogue_precompute "
        f"--settings {Path(settings_path)}"
    )


def main() -> int:
    """Run candidate-catalogue precomputation from the command line.

    :return: Process exit status.
    :rtype: int
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--settings", required=True, type=Path)
    arguments = parser.parse_args()
    settings = read_settings(arguments.settings)
    print(precompute_catalogue(settings, settings_path=arguments.settings))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
