"""Tests for repository-aware dataset source CLI loading."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from msi_autoencoder_wrapper.dataset_management.cli import (
    _repository_root,
    _resolve_cli_path,
    _resolve_config_path,
)


def test_cli_and_config_paths_have_distinct_stable_roots(tmp_path: Path) -> None:
    """CLI paths use the repository while config paths use the config folder."""
    repository_root = _repository_root()
    config_directory = tmp_path / "configs"

    assert _resolve_cli_path("workspace", repository_root) == repository_root / "workspace"
    assert _resolve_config_path("../data/input.imzML", config_directory) == (
        tmp_path / "data" / "input.imzML"
    ).resolve()


def test_dataset_cli_help_does_not_import_models_or_training() -> None:
    """Dataset management starts without importing model and training trees."""
    code = """
import sys
from msi_autoencoder_wrapper.dataset_management.cli import build_parser
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
