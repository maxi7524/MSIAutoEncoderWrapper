"""CLI for complete analysis precompute strategies."""

from __future__ import annotations

import argparse
import shlex
from pathlib import Path

import yaml

from .core.runner import run_precompute


def run_precompute_command(settings_path: Path | str, *, background: bool = True) -> str:
    """Return the canonical shell command for one complete configured strategy.

    :param settings_path: Analysis YAML beside the notebook collection.
    :type settings_path: pathlib.Path | str
    :param background: Return a detached command with a notebook-local log when true.
    :type background: bool
    :return: Copy-pasteable command generated from the same entry point as the runner.
    :rtype: str
    """
    path = Path(settings_path)
    command = (
        "uv run --extra cu118 python -m msi_autoencoder_wrapper.analysis.precompute "
        f"--settings {shlex.quote(str(path))}"
    )
    if not background:
        return command
    log = path.parent / "shared_precompute.log"
    return f"nohup {command} > {shlex.quote(str(log))} 2>&1 &"


def main(argv: list[str] | None = None) -> int:
    """Parse CLI arguments and execute one complete configured strategy."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--settings", required=True, help="Path to analysis_settings.yaml.")
    parser.add_argument("--strategy",
                        help="Optional override of YAML precompute.strategy.")
    parser.add_argument("--allow-cpu", action="store_true", help="Permit a deliberate CPU fallback.")
    parser.add_argument("--dry-run", action="store_true", help="Validate without model inference.")
    parser.add_argument("--analysis", action="append", dest="analyses",
                        help="Produce only this analysis (repeatable) and the shared stages it needs.")
    arguments = parser.parse_args(argv)
    strategy = arguments.strategy
    if strategy is None:
        payload = yaml.safe_load(Path(arguments.settings).read_text()) or {}
        strategy = (payload.get("precompute") or {}).get("strategy")
    if not strategy:
        parser.error("Set precompute.strategy in the YAML or pass --strategy.")
    run_precompute(arguments.settings, strategy, allow_cpu=arguments.allow_cpu,
                   dry_run=arguments.dry_run, analyses=arguments.analyses)
    return 0
