"""Tests for verified and ownership-checked RAM staging."""

from __future__ import annotations

from pathlib import Path

import pytest

from msi_autoencoder_wrapper.runtime.staging import (
    cleanup_staging_directory,
    copy_verified,
    create_staging_directory,
    restore_results,
)


def test_staging_copy_is_verified_and_owned_cleanup_is_complete(tmp_path: Path) -> None:
    """A complete source tree is copied and its owned staging root is removed."""
    source = tmp_path / "source"
    source.mkdir()
    (source / "data.bin").write_bytes(b"spectra")
    staging = create_staging_directory(tmp_path / "ram", "execution-1")

    copy_verified(source, staging / "inputs")

    assert (staging / "inputs" / "data.bin").read_bytes() == b"spectra"
    cleanup_staging_directory(staging, "execution-1")
    assert not staging.exists()


def test_cleanup_refuses_a_directory_without_the_matching_marker(tmp_path: Path) -> None:
    """Cleanup cannot remove an arbitrary path supplied as a staging directory."""
    directory = tmp_path / "not-owned"
    directory.mkdir()

    with pytest.raises(ValueError, match="unowned"):
        cleanup_staging_directory(directory, "execution-1")


def test_result_restore_continues_after_one_missing_artifact(tmp_path: Path) -> None:
    """Completed results are preserved even when another run produced no artifact."""
    staging = tmp_path / "staging"
    (staging / "completed").mkdir(parents=True)
    (staging / "completed" / "model.pt").write_bytes(b"weights")
    persistent = tmp_path / "persistent"
    persistent.mkdir()

    with pytest.raises(OSError, match="one or more"):
        restore_results(
            staging,
            config_directory=persistent,
            results=[
                {"source": "missing", "destination": "failed"},
                {"source": "completed", "destination": "completed"},
            ],
        )

    assert (persistent / "completed" / "model.pt").read_bytes() == b"weights"
