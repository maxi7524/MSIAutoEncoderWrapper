"""Tests for ordered and failure-tolerant Quarto rendering."""

from __future__ import annotations

import subprocess
from pathlib import Path

import yaml

from msi_autoencoder_wrapper.execution.reporting import render_reports


def test_reports_run_in_order_and_continue_after_a_failure(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """One failed report does not prevent later configured reports."""
    for name in ("first.qmd", "second.qmd", "third.qmd"):
        (tmp_path / name).write_text("---\ntitle: test\n---\n", encoding="utf-8")
    calls: list[str] = []

    def run(command: list[str], check: bool) -> None:
        assert check
        calls.append(Path(command[2]).name)
        if calls[-1] == "second.qmd":
            raise subprocess.CalledProcessError(1, command)

    monkeypatch.setattr(subprocess, "run", run)
    manifest = tmp_path / "manifest.yaml"

    success = render_reports(
        ["first.qmd", "second.qmd", "third.qmd"],
        config_directory=tmp_path,
        manifest_path=manifest,
    )

    assert not success
    assert calls == ["first.qmd", "second.qmd", "third.qmd"]
    records = yaml.safe_load(manifest.read_text(encoding="utf-8"))["records"]
    assert records["first"]["status"] == "completed"
    assert records["second"]["status"] == "failed"
    assert records["third"]["status"] == "completed"
