"""Command-line entry points for dataset query, download, and composition."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Sequence

from .layout import DatasetWorkspaceLayout
from .utils.exceptions import DatasetManagerError
from .operations import compose_cohort
from .operations.composition import create_composition_manifest
from .operations.download import (
    create_download_manifest,
    download_from_manifest,
    bash_print_download_manifest,
)
from .operations.query import query_to_selection
from .sources.profiles import read_source_profiles
from .sources.source_manager import DatasetSourceManager


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
        "--metaspace-api-keys",
        dest="metaspace_api_keys",
        type=Path,
        help=(
            "CSV of Metaspace API keys with a mandatory 'key' column; "
            "additional columns are optional metadata"
        ),
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
    """Run one command and render expected dataset-manager failures for a CLI."""
    try:
        _main(argv)
    except DatasetManagerError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(2) from None


def _main(argv: Sequence[str] | None = None) -> None:
    """Dispatch one explicit dataset-management stage.

    ``query`` writes a frozen selection, ``download`` creates and executes a
    filesystem-derived manifest, and ``compose`` merges local inputs while
    creating the composed SQLite catalogue. Each command returns immediately
    after its own workflow, so provider access and output artifacts remain
    explicit and independently repeatable.
    """
    # Shared argument and workspace resolution
    ## Parse the selected subcommand, resolve its workspace against the
    ## invocation directory, and derive the cohort label used for reporting.
    arguments = build_parser().parse_args(argv)
    invocation_directory = _repository_root()
    arguments.workspace_path = _resolve_cli_path(
        arguments.workspace_path, invocation_directory
    )
    layout = DatasetWorkspaceLayout(arguments.workspace_path)
    cohort_id = _resolve_cohort_id(arguments)

    print(f"Invocation directory: {invocation_directory}")
    print(f"Workspace path: {arguments.workspace_path}")
    print(f"Cohort: {cohort_id}")

    # Command dispatch
    ## Query constructs a provider, applies filter.json, validates provider
    ## records, and writes the frozen selection.json artifact.
    if arguments.command == "query":
        _run_query(arguments, invocation_directory)
        return

    ## Compose consumes canonical local source folders and produces the merged
    ## imzML/ibd pair, its SQLite catalog, and normalized composition.json.
    if arguments.command == "compose":
        _run_compose(arguments, layout, invocation_directory)
        return

    ## Download first creates a local manifest from selection.json and the
    ## current filesystem. A dry run stops before provider construction; a real
    ## run downloads the selected files from that manifest.
    if arguments.command == "download":
        manifest, metaspace_api_keys_path, metaspace_api_keys = _create_download_manifest(
            arguments, layout.datasets_dir, invocation_directory
        )
        if arguments.dry_run:
            return

        # Download provider construction
        ## Download uses frozen selection data and does not load the provider's
        ## discovery catalogue again.
        DatasetSourceManager.discover_strategies()
        source = DatasetSourceManager.get_source(
            arguments.source,
            **(
                {"client_options": {"api_key": metaspace_api_keys[0]["key"]}}
                if metaspace_api_keys_path is not None
                else {}
            ),
            load_catalog=False,
        )
        _download_from_manifest(
            arguments=arguments,
            source=source,
            manifest=manifest,
            metaspace_api_keys_path=metaspace_api_keys_path,
        )
        return

    raise ValueError(f"Unsupported command: {arguments.command}")

# --------------------------------------------------
# Section: CLI COMMANDS
# --------------------------------------------------

def _run_query(arguments: argparse.Namespace, invocation_directory: Path) -> None:
    """Query provider metadata and persist the frozen selection JSON.

    The provider catalogue is used only during discovery. The operation writes
    the accepted provider records and effective filters to ``selection.json``;
    it does not download binaries, create a manifest, or create SQLite state.
    """
    # Provider discovery and query
    ## Query is the only stage that loads the provider catalogue and applies
    ## filter.json; it writes metadata to selection.json, not binary files.
    DatasetSourceManager.discover_strategies()
    source = DatasetSourceManager.get_source(arguments.source)
    filters_path = _resolve_cli_path(arguments.filters, invocation_directory)
    selection_path = _resolve_cli_path(arguments.selection, invocation_directory)
    print(f"Query filters: {filters_path}")
    print(f"Query selection: {selection_path}")
    query_to_selection(
        source=source,
        filters=_read_json(filters_path),
        selection_path=selection_path,
    )


def _run_compose(
    arguments: argparse.Namespace,
    layout: DatasetWorkspaceLayout,
    invocation_directory: Path,
) -> None:
    """Compose local source datasets into one cohort and its catalog.

    Dataset IDs are resolved from explicit CLI arguments, then the composition
    config, then the optional selection. The operation validates local pairs,
    imports annotations, merges spectra, writes provenance, and stores the
    normalized composition configuration.
    """
    # Composition input resolution
    ## Explicit dataset IDs take precedence over the config and selection;
    ## configuration values provide reusable composition defaults.
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

    # Cohort composition
    ## Compose validates local pairs, imports annotations, merges spectra, writes
    ## provenance to SQLite, and persists the normalized composition config.
    print(f"Output dataset catalog: {layout.composed_catalog_path(arguments.cohort_id)}")
    composition_manifest = create_composition_manifest(
        workspace_path=arguments.workspace_path,
        source=str(composition_config.get("source", arguments.source)),
        dataset_ids=dataset_ids,
    )
    print(
        "Composition inputs: "
        f"{len(composition_manifest['dataset_ids'])} available, "
        f"{len(composition_manifest['missing_dataset_ids'])} unavailable."
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
        random_seed=int(composition_config.get("random_seed", arguments.random_seed)),
        config=composition_config,
        composition_manifest=composition_manifest,
    )
    print(f"Composed dataset: {output}")
    print(f"Composition config: {layout.composition_path(arguments.cohort_id)}")


def _create_download_manifest(
    arguments: argparse.Namespace,
    datasets_dir: Path,
    invocation_directory: Path,
) -> tuple[Dict[str, Any], Path | None, list[Dict[str, str]] | None]:
    """Create and report the filesystem-derived download manifest.

    The manifest is the local execution plan between ``selection.json`` and
    provider I/O. It records which dataset and annotation files are reusable,
    which must be downloaded, and where execution state is persisted.
    """
    # Manifest creation
    ## Selection data is combined with the current filesystem state to classify
    ## each dataset as reusable or requiring dataset/annotation download.
    arguments.selection = _resolve_cli_path(arguments.selection, invocation_directory)
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
    metaspace_api_keys_path = None
    metaspace_api_keys = None
    if arguments.metaspace_api_keys is not None:
        metaspace_api_keys_path = _resolve_cli_path(
            arguments.metaspace_api_keys,
            invocation_directory,
        )
        metaspace_api_keys = read_source_profiles(metaspace_api_keys_path)
    manifest = create_download_manifest(
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

    # Manifest reporting
    ## This output explains the planned file operations before any API access.
    print(f"Manifest: {manifest['manifest_path']}")
    print(f"Datasets directory (absolute): {manifest['workspace_datasets_path']}")
    print(
        "Datasets directory (relative to invocation): "
        f"{manifest['workspace_datasets_path_relative_to_invocation']}"
    )
    print(bash_print_download_manifest(manifest))
    if metaspace_api_keys is not None:
        print(
            "Metaspace API keys available: "
            f"{len(metaspace_api_keys)}. METASPACE does not expose the remaining "
            "quota for a key before a request."
        )
    if arguments.dry_run:
        print("Dry run complete: no provider was initialized and no download started.")
    return manifest, metaspace_api_keys_path, metaspace_api_keys


def _download_from_manifest(
    *,
    arguments: argparse.Namespace,
    source: Any,
    manifest: Dict[str, Any],
    metaspace_api_keys_path: Path | None,
) -> None:
    """Download datasets and annotations described by a prepared manifest.

    This is the CLI boundary where the local plan becomes provider I/O. The
    underlying operation updates the manifest after each dataset-level state
    transition and may rotate through configured Metaspace API keys.
    """
    # Manifest execution
    ## This is the boundary where the local plan becomes provider I/O.
    download_from_manifest(
        source=source,
        metaspace_api_keys_path=metaspace_api_keys_path,
        source_factory=(
            lambda key, _profile: DatasetSourceManager.get_source(
                arguments.source,
                client_options={"api_key": key},
                load_catalog=False,
            )
        ),
        manifest=manifest,
    )

# --------------------------------------------------
# Section: Inner helpers
# --------------------------------------------------


def _read_json(path: Path) -> Dict[str, Any]:
    """Read one UTF-8 JSON object from ``path``."""
    return json.loads(path.read_text(encoding="utf-8"))


def _add_workspace_argument(parser: argparse.ArgumentParser) -> None:
    """Add the shared repository-aware workspace option to a subcommand."""
    parser.add_argument(
        "--workspace-path",
        dest="workspace_path",
        type=Path,
        default=Path("workspace"),
        help="Workspace path relative to the invocation directory (default: workspace)",
    )


def _add_unannotated_sampling_arguments(parser: argparse.ArgumentParser) -> None:
    """Add optional controls for selecting unannotated spectra during compose."""
    parser.add_argument("--unannotated-ratio", type=float)
    parser.add_argument("--unannotated-amount", type=int)
    parser.add_argument("--random-seed", type=int, default=0)


def _resolve_cohort_id(arguments: argparse.Namespace) -> str:
    """Resolve the cohort label used by reporting and composition output paths."""
    explicit = getattr(arguments, "cohort_id", None)
    if explicit:
        return str(explicit)
    selection = getattr(arguments, "selection", None)
    if selection is not None and Path(selection).parent.name:
        return Path(selection).parent.name
    config = getattr(arguments, "config", None)
    if config is not None:
        return Path(config).stem
    return "default"


def _validate_selection_workspace(*, selection_path: Path, workspace_path: Path) -> None:
    """Reject a selection belonging to a different conventional workspace.

    The check is applied before provider construction so an invalid selection
    cannot trigger external API access.
    """
    resolved_selection = selection_path.resolve()
    parents = resolved_selection.parents
    if len(parents) < 4 or parents[1].name != "datasets" or parents[2].name != "configs":
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

    The installed package must resolve user-provided relative paths from the
    command invocation directory rather than from its installation directory.
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
