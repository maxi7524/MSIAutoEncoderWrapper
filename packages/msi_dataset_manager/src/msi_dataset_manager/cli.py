"""Command-line entry points for dataset query, download, and composition."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Sequence

from .layout import DatasetWorkspaceLayout
from .operations import compose_cohort
from .operations.download import (
    create_materialization_manifest,
    format_materialization_plan,
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
    download.add_argument(
        "--dry-run",
        action="store_true",
        help="Scan files, write and print the manifest, then exit without provider access",
    )

    compose = commands.add_parser(
        "compose", help="Create a cohort dataset from canonical local folders"
    )
    _add_workspace_argument(compose)
    compose.add_argument("--cohort-id", required=True)
    compose.add_argument("--source", default="metaspace")
    compose.add_argument("--selection", type=Path)
    compose.add_argument("--config", type=Path)
    compose.add_argument("--dataset-id", action="append", dest="dataset_ids")
    compose.add_argument("--row-width", type=int)
    compose.add_argument("--max-fdr", type=float)
    compose.add_argument("--minimum-dataset-occurrence", type=int, default=1)
    _add_unannotated_sampling_arguments(compose)

    return parser


def main(argv: Sequence[str] | None = None) -> None:
    """Dispatch the explicit ``query``, ``download``, and ``compose`` stages.

    ``query`` writes a frozen selection, ``download`` regenerates and executes
    a filesystem-derived manifest, and ``compose`` merges local inputs while
    creating the only SQLite catalogue. No command implicitly invokes another
    stage, which keeps provider access visible and independently repeatable.
    """
    # Shared path resolution
    arguments = build_parser().parse_args(argv)
    invocation_directory = _repository_root()
    arguments.workspace_path = _resolve_cli_path(
        arguments.workspace_path, invocation_directory
    )
    layout = DatasetWorkspaceLayout(arguments.workspace_path)
    datasets_dir = layout.datasets_dir
    cohort_id = _resolve_cohort_id(arguments)
    resolved_profiles_path = None
    source_profiles = None

    print(f"Invocation directory: {invocation_directory}")
    print(f"Workspace path: {arguments.workspace_path}")
    print(f"Cohort: {cohort_id}")

    # Composition dispatch
    ## Compose reads only canonical local files and owns the output SQLite.
    if arguments.command == "compose":
        print(f"Output dataset catalog: {layout.composed_catalog_path(cohort_id)}")
        composition_config = (
            _read_json(_resolve_cli_path(arguments.config, invocation_directory))
            if arguments.config is not None
            else {}
        )
        selection_ids: list[str] = []
        if arguments.selection is not None:
            selection = _read_json(
                _resolve_cli_path(arguments.selection, invocation_directory)
            )
            selection_ids = [str(item["dataset_id"]) for item in selection["datasets"]]
        dataset_ids = arguments.dataset_ids or composition_config.get(
            "dataset_ids", selection_ids
        )
        output = compose_cohort(
            workspace_path=arguments.workspace_path,
            cohort_id=arguments.cohort_id,
            source=str(composition_config.get("source", arguments.source)),
            dataset_ids=dataset_ids,
            row_width=composition_config.get("row_width", arguments.row_width),
            max_fdr=composition_config.get("max_fdr", arguments.max_fdr),
            minimum_dataset_occurrence=int(
                composition_config.get(
                    "minimum_dataset_occurrence",
                    arguments.minimum_dataset_occurrence,
                )
            ),
            unannotated_ratio=composition_config.get(
                "unannotated_ratio", arguments.unannotated_ratio
            ),
            unannotated_amount=composition_config.get(
                "unannotated_amount", arguments.unannotated_amount
            ),
            random_seed=int(
                composition_config.get("random_seed", arguments.random_seed)
            ),
            config=composition_config,
        )
        print(f"Composed dataset: {output}")
        print(f"Composition config: {layout.composition_path(arguments.cohort_id)}")
        return

    # Download planning
    ## Always replace the manifest from the current selection and filesystem.
    ## Dry-run returns here, before strategy discovery or provider construction.
    if arguments.command == "download":
        arguments.selection = _resolve_cli_path(
            arguments.selection, invocation_directory
        )
        _validate_selection_workspace(
            selection_path=arguments.selection,
            workspace_path=arguments.workspace_path,
        )
        annotation_options_path = (
            _resolve_cli_path(arguments.annotation_options, invocation_directory)
            if arguments.annotation_options is not None
            else None
        )
        manifest_path = (
            _resolve_cli_path(arguments.manifest, invocation_directory)
            if arguments.manifest is not None
            else None
        )
        if arguments.profiles is not None:
            resolved_profiles_path = _resolve_cli_path(
                arguments.profiles, invocation_directory
            )
            source_profiles = read_source_profiles(resolved_profiles_path)
        manifest = create_materialization_manifest(
            selection_path=arguments.selection,
            datasets_dir=datasets_dir,
            source_name=arguments.source,
            annotation_options=(
                _read_json(annotation_options_path)
                if annotation_options_path is not None
                else None
            ),
            dataset_ids=arguments.dataset_ids,
            manifest_path=manifest_path,
            dry_run=arguments.dry_run,
        )
        print(f"Manifest: {manifest['manifest_path']}")
        print(
            "Datasets directory (absolute): "
            f"{manifest['workspace_datasets_path']}"
        )
        print(
            "Datasets directory (relative to invocation): "
            f"{manifest['workspace_datasets_path_relative_to_invocation']}"
        )
        print(format_materialization_plan(manifest))
        if source_profiles is not None:
            print(
                "Provider profiles available: "
                f"{len(source_profiles)}. METASPACE does not expose the remaining "
                "download quota for a profile before a request."
            )
        if arguments.dry_run:
            print("Dry run complete: no provider was initialized and no download started.")
            return

    # Provider construction
    ## Query loads discovery metadata; download uses the frozen selection and
    ## therefore explicitly disables the provider catalogue request.
    DatasetSourceManager.discover_strategies()
    source = DatasetSourceManager.get_source(
        arguments.source,
        **(
            {
                "client_options": {
                    "api_key": source_profiles[0]["key"]
                }
            }
            if resolved_profiles_path is not None
            else {}
        ),
        **(
            {"load_catalog": False}
            if arguments.command == "download"
            else {}
        ),
    )
    if arguments.command == "query":
        # Query dispatch
        ## Persist selection criteria and records only; SQLite is a compose output.
        filters_path = _resolve_cli_path(arguments.filters, invocation_directory)
        selection_path = _resolve_cli_path(arguments.selection, invocation_directory)
        print(f"Query filters: {filters_path}")
        print(f"Query selection: {selection_path}")
        query_to_selection(
            source=source,
            filters=_read_json(filters_path),
            selection_path=selection_path,
        )
        return

    # Download execution
    ## Execute exactly the fresh manifest printed above. The operation updates
    ## that manifest after every dataset-level state transition.
    materialize_selection(
        source=source,
        profiles_path=resolved_profiles_path,
        source_factory=(
            lambda key, _profile: DatasetSourceManager.get_source(
                arguments.source,
                client_options={"api_key": key},
                load_catalog=False,
            )
        ),
        manifest=manifest,
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
        help="Workspace path relative to the invocation directory (default: workspace)",
    )


def _add_unannotated_sampling_arguments(parser: argparse.ArgumentParser) -> None:
    """Add optional merge sampling controls to a CLI parser."""
    parser.add_argument("--unannotated-ratio", type=float)
    parser.add_argument("--unannotated-amount", type=int)
    parser.add_argument("--random-seed", type=int, default=0)


def _resolve_cohort_id(arguments: argparse.Namespace) -> str:
    """Resolve the cohort label printed by CLI and required by composition."""
    explicit = getattr(arguments, "cohort_id", None)
    if explicit:
        return str(explicit)
    selection = getattr(arguments, "selection", None)
    if selection is not None:
        parent_name = Path(selection).parent.name
        if parent_name:
            return parent_name
    config = getattr(arguments, "config", None)
    if config is not None:
        return Path(config).stem
    return "default"


def _validate_selection_workspace(
    *,
    selection_path: Path,
    workspace_path: Path,
) -> None:
    """Reject a selection belonging to a different workspace before API access.

    :param selection_path: Resolved ``selection.json`` path.
    :type selection_path: pathlib.Path
    :param workspace_path: Resolved target workspace path.
    :type workspace_path: pathlib.Path
    :raises ValueError: If the conventional selection layout identifies a
        different workspace.
    """
    resolved_selection = selection_path.resolve()
    parents = resolved_selection.parents
    if (
        len(parents) < 4
        or parents[1].name != "datasets"
        or parents[2].name != "configs"
    ):
        return
    selection_workspace = parents[3]
    resolved_workspace = workspace_path.resolve()
    if selection_workspace == resolved_workspace:
        return
    raise ValueError(
        "Selection/workspace mismatch detected before provider access: "
        f"selection '{resolved_selection}' belongs to workspace "
        f"'{selection_workspace}', but --workspace-path resolves to "
        f"'{resolved_workspace}'. Use --workspace-path {selection_workspace}."
    )


def _repository_root() -> Path:
    """Return the invocation directory used to resolve relative CLI paths.

    The legacy name is retained for API compatibility. An installed standalone
    package must not infer paths from its own installation directory.
    """
    return Path.cwd().resolve()


def _resolve_cli_path(path: Path | str, repository_root: Path) -> Path:
    """Resolve a CLI path against the supplied invocation directory."""
    candidate = Path(path)
    return (
        candidate.resolve()
        if candidate.is_absolute()
        else (repository_root / candidate).resolve()
    )


if __name__ == "__main__":
    main()
