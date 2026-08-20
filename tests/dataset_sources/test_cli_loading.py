"""Tests for repository-aware dataset source CLI loading."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from msi_dataset_manager.cli import (
    build_parser,
    _repository_root,
    _resolve_cli_path,
    _validate_selection_workspace,
    main,
)


def test_compose_accepts_annotation_aware_sampling_limits() -> None:
    """Composition exposes reproducible unannotated spectrum controls."""
    arguments = build_parser().parse_args(
        [
            "compose",
            "--source",
            "metaspace",
            "--cohort-id",
            "merged",
            "--unannotated-ratio",
            "3.0",
            "--unannotated-amount",
            "200",
            "--random-seed",
            "42",
        ]
    )

    assert arguments.unannotated_ratio == 3.0
    assert arguments.unannotated_amount == 200
    assert arguments.random_seed == 42


@pytest.mark.parametrize("removed_command", ["merge", "import-local", "download-merge"])
def test_cli_rejects_redundant_workflow_commands(removed_command: str) -> None:
    """The public CLI exposes only independent query, download, and compose stages."""
    with pytest.raises(SystemExit):
        build_parser().parse_args([removed_command])


def test_download_rejects_selection_from_another_workspace_before_api_access(
    tmp_path: Path,
) -> None:
    """A workspace typo cannot bypass existing files and consume provider quota."""
    selection = (
        tmp_path
        / "kidney_workspace"
        / "configs"
        / "datasets"
        / "kidney"
        / "selection.json"
    )

    with pytest.raises(ValueError, match="Selection/workspace mismatch"):
        _validate_selection_workspace(
            selection_path=selection,
            workspace_path=tmp_path / "kidney_data",
        )


def test_download_accepts_selection_from_target_workspace(tmp_path: Path) -> None:
    """A conventional selection and its owning workspace pass validation."""
    workspace = tmp_path / "kidney_workspace"
    selection = workspace / "configs" / "datasets" / "kidney" / "selection.json"

    _validate_selection_workspace(
        selection_path=selection,
        workspace_path=workspace,
    )


def test_download_dry_run_writes_plan_without_initializing_provider(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Dry-run scans disk and writes the real plan before any provider access."""
    workspace = tmp_path / "kidney_workspace"
    config_dir = workspace / "configs" / "datasets" / "kidney"
    config_dir.mkdir(parents=True)
    selection = config_dir / "selection.json"
    selection.write_text(
        '{"source":"metaspace","datasets":['
        '{"dataset_id":"one","name":"One"}]}'
        ,
        encoding="utf-8",
    )

    def forbidden_provider_access(*args: object, **kwargs: object) -> None:
        raise AssertionError("provider access is forbidden during dry-run")

    monkeypatch.setattr(
        "msi_dataset_manager.cli.DatasetSourceManager.discover_strategies",
        forbidden_provider_access,
    )
    main(
        [
            "download",
            "--workspace-path",
            str(workspace),
            "--selection",
            str(selection),
            "--source",
            "metaspace",
            "--dry-run",
        ]
    )

    manifest = json.loads((config_dir / "materialization.json").read_text())
    assert manifest["dry_run"] is True
    assert manifest["status"] == "planned"
    assert manifest["datasets"][0]["planned_actions"] == [
        "download_dataset",
        "download_annotations",
    ]
    output = capsys.readouterr().out
    assert "Datasets directory (absolute)" in output
    assert "download_dataset, download_annotations" in output


def test_cli_paths_use_the_invocation_directory() -> None:
    """Relative CLI paths use the invocation directory."""
    repository_root = _repository_root()

    assert _resolve_cli_path("workspace", repository_root) == repository_root / "workspace"


def test_dataset_cli_help_does_not_import_models_or_training() -> None:
    """Dataset management starts without importing model and training trees."""
    code = """
import sys
from msi_dataset_manager.cli import build_parser
build_parser().format_help()
assert not any(name.startswith('msi_autoencoder_wrapper.models') for name in sys.modules)
assert not any(name.startswith('msi_autoencoder_wrapper.training') for name in sys.modules)
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
