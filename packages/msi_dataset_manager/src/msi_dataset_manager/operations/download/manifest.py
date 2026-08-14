"""Create and persist filesystem-derived dataset download plans."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

from ...operations.annotation_csv import has_complete_annotation_csv
from ...validators import validate_selection

# --------------------------------------------------
# Section: Main functionality 
# --------------------------------------------------

def create_download_manifest(
    *,
    selection_path: Path | str,
    datasets_dir: Path | str,
    source_name: str,
    annotation_options: Optional[Mapping[str, Any]] = None,
    dataset_ids: Optional[Sequence[str]] = None,
    manifest_path: Optional[Path | str] = None,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """Scan local datasets and persist the complete materialization plan.

    :param selection_path: Frozen selection JSON produced by query.
    :param datasets_dir: Root directory containing source dataset directories.
    :param source_name: Source expected by the selection.
    :param annotation_options: Explicit provider annotation options.
    :param dataset_ids: Optional subset of selected dataset identifiers.
    :param manifest_path: Optional destination for the manifest JSON.
    :param dry_run: Whether the caller intends to inspect without execution.
    :return: Manifest whose ``planned_actions`` are the sole execution plan.
    :raises ValueError: If selection and requested source do not match.
    """
    selection_file = Path(selection_path).resolve()
    selection = json.loads(selection_file.read_text(encoding="utf-8"))
    validate_selection(selection)
    if selection.get("source") != source_name:
        raise ValueError(
            f"Selection source '{selection.get('source')}' does not match "
            f"requested source '{source_name}'."
        )
    resolved_options = resolve_download_annotation_options(selection, annotation_options)
    selected = set(dataset_ids or ())
    records = [
        record
        for record in selection.get("datasets", [])
        if not selected or str(record["dataset_id"]) in selected
    ]
    dataset_root = Path(datasets_dir).resolve()

    # Filesystem scan
    ## The manifest records the local state once. Execution uses the resulting
    ## planned actions rather than rediscovering the selection remotely.
    entries = []
    for record in records:
        dataset_id = str(record["dataset_id"])
        directory = dataset_root / dataset_id
        data_present = has_complete_pair(directory, dataset_id)
        annotations_present = has_complete_annotation_csv(directory, dataset_id)
        actions = []
        if not data_present:
            actions.append("download_dataset")
        if not annotations_present:
            actions.append("download_annotations")
        entries.append(
            {
                "dataset_id": dataset_id,
                "name": str(record.get("name", dataset_id)),
                "directory": str(directory),
                "directory_relative_to_invocation": relative_display(directory),
                "data_present": data_present,
                "annotations_present": annotations_present,
                "planned_actions": actions,
                "execution": {
                    "dataset": "reused" if data_present else "pending",
                    "annotations": "reused" if annotations_present else "pending",
                },
                "final_state": {
                    "data_present": data_present,
                    "annotations_present": annotations_present,
                },
            }
        )
    target = (
        Path(manifest_path).resolve()
        if manifest_path is not None
        else selection_file.parent / "materialization.json"
    )
    manifest: Dict[str, Any] = {
        "schema_version": 2,
        "dry_run": bool(dry_run),
        "status": "planned",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "invocation_directory": str(Path.cwd().resolve()),
        "workspace_datasets_path": str(dataset_root),
        "workspace_datasets_path_relative_to_invocation": relative_display(dataset_root),
        "selection_path": str(selection_file),
        "manifest_path": str(target),
        "source": source_name,
        "annotation_options": resolved_options,
        "datasets": entries,
        "summary": manifest_summary(entries),
    }
    write_json_atomic(target, manifest)
    return manifest

# --------------------------------------------------
# Section: CLI printers 
# --------------------------------------------------

def bash_print_download_manifest(manifest: Mapping[str, Any]) -> str:
    """Return a compact table containing local state and planned actions."""
    headers = ("Dataset ID", "Data", "Annotations", "Actions", "Destination")
    rows = [
        (
            str(entry["dataset_id"]),
            "yes" if entry["data_present"] else "no",
            "yes" if entry["annotations_present"] else "no",
            ", ".join(entry["planned_actions"]) or "none",
            str(entry["directory_relative_to_invocation"]),
        )
        for entry in manifest.get("datasets", [])
    ]
    widths = [
        max(len(headers[index]), *(len(row[index]) for row in rows))
        for index in range(len(headers))
    ]
    line = "  ".join(headers[index].ljust(widths[index]) for index in range(len(headers)))
    divider = "  ".join("-" * width for width in widths)
    body = [
        "  ".join(row[index].ljust(widths[index]) for index in range(len(headers)))
        for row in rows
    ]
    return "\n".join([line, divider, *body, bash_print_manifest_summary(manifest, len(rows))])


def bash_print_manifest_summary(manifest: Mapping[str, Any], entry_count: int) -> str:
    """Return the one-line summary displayed below a download manifest."""
    summary = manifest.get("summary", {})
    return (
        "Plan: "
        f"{summary.get('selected', entry_count)} selected; "
        f"{summary.get('download_dataset', 0)} dataset downloads remaining; "
        f"{summary.get('download_annotations', 0)} annotation downloads remaining."
    )


def manifest_summary(entries: Sequence[Mapping[str, Any]]) -> Dict[str, int]:
    """Summarize a freshly scanned materialization plan."""
    return {
        "selected": len(entries),
        "data_present": sum(bool(entry["data_present"]) for entry in entries),
        "annotations_present": sum(
            bool(entry["annotations_present"]) for entry in entries
        ),
        "download_dataset": sum(
            "download_dataset" in entry["planned_actions"] for entry in entries
        ),
        "download_annotations": sum(
            "download_annotations" in entry["planned_actions"] for entry in entries
        ),
    }


def final_manifest_summary(entries: Sequence[Mapping[str, Any]]) -> Dict[str, int]:
    """Summarize final filesystem state after executing a plan."""
    return {
        "data_present": sum(
            bool(entry["final_state"]["data_present"]) for entry in entries
        ),
        "annotations_present": sum(
            bool(entry["final_state"]["annotations_present"]) for entry in entries
        ),
    }


# --------------------------------------------------
# Section: Inner helpers 
# --------------------------------------------------

def has_complete_pair(directory: Path, dataset_id: str) -> bool:
    """Return whether a non-empty canonical imzML/ibd pair exists locally."""
    imzml_path = directory / f"{dataset_id}.imzML"
    ibd_path = directory / f"{dataset_id}.ibd"
    return (
        imzml_path.is_file()
        and imzml_path.stat().st_size > 0
        and ibd_path.is_file()
        and ibd_path.stat().st_size > 0
    )


def resolve_download_annotation_options(
    selection: Mapping[str, Any],
    annotation_options: Optional[Mapping[str, Any]],
) -> Dict[str, Any]:
    """Resolve one annotation FDR across discovery and materialization."""
    from ...utils.exceptions import raise_validation_error

    resolved = dict(annotation_options or {})
    if "fdr" in resolved:
        raise_validation_error(
            "AnnotationOptions",
            "Use 'annotation_fdr' instead of the ambiguous legacy key 'fdr'.",
        )
    selection_filters = selection.get("filters", {})
    selection_fdr = (
        selection_filters.get("annotation_fdr")
        if isinstance(selection_filters, Mapping)
        else None
    )
    requested_fdr = resolved.get("annotation_fdr")
    if selection_fdr is not None and requested_fdr is not None:
        if float(selection_fdr) != float(requested_fdr):
            raise_validation_error(
                "AnnotationOptions",
                "The annotation_fdr used for download must match the value "
                f"stored in the selection ({selection_fdr}).",
            )
    if selection_fdr is not None:
        resolved["annotation_fdr"] = float(selection_fdr)
    return resolved



def relative_display(path: Path) -> str:
    """Return an invocation-relative path when possible, otherwise absolute."""
    try:
        return str(path.resolve().relative_to(Path.cwd().resolve()))
    except ValueError:
        return str(path.resolve())


def write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    """Write one JSON artifact atomically without provider credentials."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
