"""Command-line entry points for dataset catalog, download, and merge stages."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Sequence

from ..datasets.imzml_merger import ImzMLMergeInput, ImzMLMerger
from ..workspace.dataset_catalog import DatasetCatalog
from .pipeline import (
    discover_to_manifest,
    materialize_and_merge_manifest,
    materialize_manifest,
)
from .source_manager import DatasetSourceManager


def build_parser() -> argparse.ArgumentParser:
    """Build the dataset-management argument parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    catalog = commands.add_parser("catalog", help="Discover metadata without downloads")
    catalog.add_argument("--workspace", type=Path, required=True)
    catalog.add_argument("--source", default="metaspace")
    catalog.add_argument("--filters", type=Path, required=True)
    catalog.add_argument("--manifest", type=Path, required=True)

    download = commands.add_parser("download", help="Materialize a discovery manifest")
    download.add_argument("--workspace", type=Path, required=True)
    download.add_argument("--source", default="metaspace")
    download.add_argument("--manifest", type=Path, required=True)
    download.add_argument("--annotation-options", type=Path)
    download.add_argument("--dataset-id", action="append", dest="dataset_ids")

    download_merge = commands.add_parser(
        "download-merge",
        help="Download and append one dataset at a time",
    )
    download_merge.add_argument("--workspace", type=Path, required=True)
    download_merge.add_argument("--source", default="metaspace")
    download_merge.add_argument("--manifest", type=Path, required=True)
    download_merge.add_argument("--output", type=Path, required=True)
    download_merge.add_argument("--merged-dataset-id", required=True)
    download_merge.add_argument("--row-width", type=int, required=True)
    download_merge.add_argument("--annotation-options", type=Path)
    download_merge.add_argument("--dataset-id", action="append", dest="dataset_ids")
    download_merge.add_argument("--keep-downloads", action="store_true")

    merge = commands.add_parser("merge", help="Merge local imzML inputs")
    merge.add_argument("--workspace", type=Path, required=True)
    merge.add_argument("--config", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    """Execute a dataset-management command."""
    arguments = build_parser().parse_args(argv)
    datasets_dir = arguments.workspace / "datasets"
    catalog = DatasetCatalog(datasets_dir / "catalog.sqlite")

    if arguments.command == "merge":
        _run_merge(arguments.config, catalog)
        return

    DatasetSourceManager.discover_strategies()
    source = DatasetSourceManager.get_source(arguments.source)
    if arguments.command == "catalog":
        discover_to_manifest(
            source=source,
            filters=_read_json(arguments.filters),
            catalog=catalog,
            manifest_path=arguments.manifest,
        )
        return
    if arguments.command == "download-merge":
        materialize_and_merge_manifest(
            source=source,
            manifest_path=arguments.manifest,
            datasets_dir=datasets_dir,
            catalog=catalog,
            output_path=arguments.output,
            merged_dataset_id=arguments.merged_dataset_id,
            row_width=arguments.row_width,
            annotation_options=(
                _read_json(arguments.annotation_options)
                if arguments.annotation_options is not None
                else None
            ),
            dataset_ids=arguments.dataset_ids,
            keep_downloads=arguments.keep_downloads,
        )
        return
    materialize_manifest(
        source=source,
        manifest_path=arguments.manifest,
        datasets_dir=datasets_dir,
        catalog=catalog,
        annotation_options=(
            _read_json(arguments.annotation_options)
            if arguments.annotation_options is not None
            else None
        ),
        dataset_ids=arguments.dataset_ids,
    )


def _run_merge(config_path: Path, catalog: DatasetCatalog) -> None:
    config = _read_json(config_path)
    inputs = [
        ImzMLMergeInput(
            source=str(item["source"]),
            dataset_id=str(item["dataset_id"]),
            imzml_path=Path(item["imzml_path"]),
            spatial_ids=item.get("spatial_ids"),
        )
        for item in config["inputs"]
    ]
    ImzMLMerger(catalog).merge(
        inputs=inputs,
        output_path=Path(config["output_path"]),
        merged_dataset_id=str(config["merged_dataset_id"]),
        row_width=config.get("row_width"),
    )


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
