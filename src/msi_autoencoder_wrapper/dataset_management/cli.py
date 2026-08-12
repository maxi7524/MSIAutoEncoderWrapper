"""Command-line entry points for dataset query, download, and merge stages."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Sequence

from .catalog.sqlite_catalog import DatasetCatalog
from .operations import ImzMLMergeInput, ImzMLMerger, import_local_dataset
from .operations.download import (
    materialize_and_merge_selection,
    materialize_selection,
)
from .operations.query import query_to_selection
from .sources.source_manager import DatasetSourceManager
from .sources.profiles import read_source_profiles


def build_parser() -> argparse.ArgumentParser:
    """Build the dataset-management argument parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    query = commands.add_parser("query", help="Query metadata without downloads")
    _add_workspace_argument(query)
    query.add_argument("--source", default="metaspace")
    query.add_argument("--filters", type=Path, required=True)
    query.add_argument("--selection", type=Path, required=True)

    download = commands.add_parser("download", help="Materialize a query selection")
    _add_workspace_argument(download)
    download.add_argument("--source", default="metaspace")
    download.add_argument("--selection", type=Path, required=True)
    download.add_argument("--annotation-options", type=Path)
    download.add_argument("--dataset-id", action="append", dest="dataset_ids")
    download.add_argument(
        "--profiles",
        type=Path,
        help="CSV with a mandatory 'key' column and optional metadata columns",
    )
    download.add_argument("--manifest", type=Path)

    download_merge = commands.add_parser(
        "download-merge",
        help="Download and append one dataset at a time",
    )
    _add_workspace_argument(download_merge)
    download_merge.add_argument("--source", default="metaspace")
    download_merge.add_argument("--selection", type=Path, required=True)
    download_merge.add_argument("--output", type=Path, required=True)
    download_merge.add_argument("--merged-dataset-id", required=True)
    download_merge.add_argument("--row-width", type=int, required=True)
    download_merge.add_argument("--annotation-options", type=Path)
    download_merge.add_argument("--dataset-id", action="append", dest="dataset_ids")
    download_merge.add_argument("--keep-downloads", action="store_true")
    _add_unannotated_sampling_arguments(download_merge)

    merge = commands.add_parser("merge", help="Merge local imzML inputs")
    _add_workspace_argument(merge)
    merge.add_argument("--config", type=Path, required=True)

    local_import = commands.add_parser(
        "import-local", help="Import an already downloaded imzML pair and CSV annotations"
    )
    _add_workspace_argument(local_import)
    local_import.add_argument("--config", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    """Execute a dataset-management command."""
    arguments = build_parser().parse_args(argv)
    repository_root = _repository_root()
    arguments.workspace_path = _resolve_cli_path(arguments.workspace_path, repository_root)
    datasets_dir = arguments.workspace_path / "datasets"
    catalog = DatasetCatalog(datasets_dir / "catalog.sqlite")

    print(f"Repository root: {repository_root}")
    print(f"Workspace path: {arguments.workspace_path}")
    print(f"Dataset catalog: {catalog.path}")

    if arguments.command == "merge":
        config_path = _resolve_cli_path(arguments.config, repository_root)
        print(f"Merge configuration: {config_path}")
        _run_merge(config_path, catalog)
        return
    if arguments.command == "import-local":
        config_path = _resolve_cli_path(arguments.config, repository_root)
        config = _read_json(config_path)
        result = import_local_dataset(
            catalog=catalog,
            source=str(config["source"]),
            dataset_id=str(config["dataset_id"]),
            name=str(config.get("name", config["dataset_id"])),
            imzml_path=_resolve_config_path(config["imzml_path"], config_path.parent),
            annotations_path=_resolve_config_path(
                config["annotations_path"], config_path.parent
            ),
            pixel_intensities_path=_resolve_config_path(
                config["pixel_intensities_path"], config_path.parent
            ),
            metadata=config.get("metadata", {}),
        )
        print(
            "Imported {spectra} spectra, {annotations} annotations, and "
            "{spatial_links} spatial links.".format(**result)
        )
        return

    DatasetSourceManager.discover_strategies()
    resolved_profiles_path = (
        _resolve_cli_path(arguments.profiles, repository_root)
        if arguments.command == "download" and arguments.profiles is not None
        else None
    )
    source = DatasetSourceManager.get_source(
        arguments.source,
        **(
            {
                "client_options": {
                    "api_key": read_source_profiles(resolved_profiles_path)[0]["key"]
                }
            }
            if resolved_profiles_path is not None
            else {}
        ),
        **(
            {"load_catalog": False}
            if arguments.command in {"download", "download-merge"}
            else {}
        ),
    )
    if arguments.command == "query":
        filters_path = _resolve_cli_path(arguments.filters, repository_root)
        selection_path = _resolve_cli_path(arguments.selection, repository_root)
        print(f"Query filters: {filters_path}")
        print(f"Query selection: {selection_path}")
        query_to_selection(
            source=source,
            filters=_read_json(filters_path),
            catalog=catalog,
            selection_path=selection_path,
        )
        return
    arguments.selection = _resolve_cli_path(arguments.selection, repository_root)
    if arguments.annotation_options is not None:
        arguments.annotation_options = _resolve_cli_path(
            arguments.annotation_options, repository_root
        )
    if arguments.command == "download-merge":
        materialize_and_merge_selection(
            source=source,
            selection_path=arguments.selection,
            datasets_dir=datasets_dir,
            catalog=catalog,
            output_path=_resolve_cli_path(arguments.output, repository_root),
            merged_dataset_id=arguments.merged_dataset_id,
            row_width=arguments.row_width,
            annotation_options=(
                _read_json(arguments.annotation_options)
                if arguments.annotation_options is not None
                else None
            ),
            dataset_ids=arguments.dataset_ids,
            keep_downloads=arguments.keep_downloads,
            unannotated_ratio=arguments.unannotated_ratio,
            unannotated_amount=arguments.unannotated_amount,
            random_seed=arguments.random_seed,
        )
        return
    materialize_selection(
        source=source,
        selection_path=arguments.selection,
        datasets_dir=datasets_dir,
        catalog=catalog,
        annotation_options=(
            _read_json(arguments.annotation_options)
            if arguments.annotation_options is not None
            else None
        ),
        dataset_ids=arguments.dataset_ids,
        profiles_path=resolved_profiles_path,
        source_factory=(
            lambda key, _profile: DatasetSourceManager.get_source(
                arguments.source,
                client_options={"api_key": key},
                load_catalog=False,
            )
        ),
        manifest_path=(
            _resolve_cli_path(arguments.manifest, repository_root)
            if arguments.manifest is not None
            else None
        ),
    )


def _run_merge(config_path: Path, catalog: DatasetCatalog) -> None:
    config = _read_json(config_path)
    config_directory = config_path.parent
    inputs = [
        ImzMLMergeInput(
            source=str(item["source"]),
            dataset_id=str(item["dataset_id"]),
            imzml_path=_resolve_config_path(item["imzml_path"], config_directory),
            spectrum_ids=item.get("spectrum_ids"),
        )
        for item in config["inputs"]
    ]
    ImzMLMerger(catalog).merge(
        inputs=inputs,
        output_path=_resolve_config_path(config["output_path"], config_directory),
        merged_dataset_id=str(config["merged_dataset_id"]),
        row_width=config.get("row_width"),
        unannotated_ratio=config.get("unannotated_ratio"),
        unannotated_amount=config.get("unannotated_amount"),
        random_seed=int(config.get("random_seed", 0)),
    )


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _add_workspace_argument(parser: argparse.ArgumentParser) -> None:
    """Add the shared repository-aware workspace option to one command."""
    parser.add_argument(
        "--workspace-path",
        dest="workspace_path",
        type=Path,
        default=Path("workspace"),
        help="Workspace path relative to the repository root (default: workspace)",
    )


def _add_unannotated_sampling_arguments(parser: argparse.ArgumentParser) -> None:
    """Add optional merge sampling controls to a CLI parser."""
    parser.add_argument("--unannotated-ratio", type=float)
    parser.add_argument("--unannotated-amount", type=int)
    parser.add_argument("--random-seed", type=int, default=0)


def _repository_root() -> Path:
    """Return the checkout root containing ``pyproject.toml``."""
    for candidate in Path(__file__).resolve().parents:
        if (candidate / "pyproject.toml").is_file():
            return candidate
    raise RuntimeError("Could not locate the repository root from the installed CLI module.")


def _resolve_cli_path(path: Path | str, repository_root: Path) -> Path:
    """Resolve a CLI path against the repository root."""
    candidate = Path(path)
    return (
        candidate.resolve()
        if candidate.is_absolute()
        else (repository_root / candidate).resolve()
    )


def _resolve_config_path(path: Path | str, config_directory: Path) -> Path:
    """Resolve a path declared inside a configuration beside that file."""
    candidate = Path(path)
    return (
        candidate.resolve()
        if candidate.is_absolute()
        else (config_directory / candidate).resolve()
    )


if __name__ == "__main__":
    main()
