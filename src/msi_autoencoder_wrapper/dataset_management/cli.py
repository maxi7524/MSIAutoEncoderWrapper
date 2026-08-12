"""Compatibility entry point for :mod:`msi_dataset_manager.cli`."""

from msi_dataset_manager.cli import (
    _repository_root,
    _resolve_cli_path,
    _resolve_config_path,
    _resolve_cohort_id,
    build_parser,
    main,
)

__all__ = ["build_parser", "main"]
