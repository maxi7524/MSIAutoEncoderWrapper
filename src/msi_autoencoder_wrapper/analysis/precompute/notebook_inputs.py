"""Stable notebook loaders for common-precompute control artifacts."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .cli import run_precompute_command
from ...visualization.analysis_catalog import load_visualization_contract, theme_from_visualization_contract


def _precompute_directory(settings: dict) -> Path:
    """Resolve the control-artifact directory declared by one analysis YAML.

    :param settings: Settings returned by a domain analysis settings loader.
    :type settings: dict
    :return: Absolute directory containing the common precompute control artifacts.
    :rtype: pathlib.Path
    """
    settings_path = Path(settings["settings_path"]).resolve()
    configured = (settings.get("precompute") or {}).get("output_directory")
    directory = Path(configured) if configured else settings_path.parent / "precompute"
    if directory.is_absolute():
        return directory
    root = Path(settings.get("repository_root", settings_path.parents[4]))
    return root / directory


def load_model_catalog(settings: dict) -> pd.DataFrame:
    """Load the exact selected model records produced by common precompute.

    :param settings: Settings returned by a domain analysis settings loader.
    :type settings: dict
    :return: Selected model repetitions with stable aliases and display labels.
    :rtype: pandas.DataFrame
    :raises FileNotFoundError: If the common precompute strategy has not completed.
    :raises ValueError: If the persisted catalog does not provide the notebook contract.
    """
    path = _precompute_directory(settings) / "resolved_model_catalog.csv"
    if not path.is_file():
        command = run_precompute_command(settings["settings_path"])
        raise FileNotFoundError(f"Missing '{path}'. Produce the complete inputs first:\n  {command}")
    catalog = pd.read_csv(path)
    required = {"model_id", "model_alias", "display_label", "source", "condition"}
    missing = sorted(required - set(catalog.columns))
    if missing:
        raise ValueError(f"Resolved model catalog is missing required column(s): {missing}.")
    if catalog.model_id.duplicated().any():
        raise ValueError("Resolved model catalog contains duplicate model identifiers.")
    return catalog


def load_visualization_theme(settings: dict):
    """Load the catalog-defined presentation colours for one notebook campaign.

    :param settings: Settings returned by a domain analysis settings loader.
    :type settings: dict
    :return: Shared theme with the model colour overrides recorded by precompute.
    :rtype: msi_autoencoder_wrapper.visualization.theme.VisualizationTheme
    """
    path = _precompute_directory(settings) / "visualization_contract.csv"
    if not path.is_file():
        command = run_precompute_command(settings["settings_path"])
        raise FileNotFoundError(f"Missing '{path}'. Produce the complete inputs first:\n  {command}")
    return theme_from_visualization_contract(load_visualization_contract(path))


def select_catalog_frame(frame: pd.DataFrame, catalog: pd.DataFrame) -> pd.DataFrame:
    """Restrict one precomputed table to catalog models and apply presentation labels.

    :param frame: Table loaded from a canonical shared or per-analysis CSV artifact.
    :type frame: pandas.DataFrame
    :param catalog: Records returned by :func:`load_model_catalog`.
    :type catalog: pandas.DataFrame
    :return: Filtered table with labels mapped from the persisted catalog where possible.
    :rtype: pandas.DataFrame
    """
    result = frame.copy()
    labels = catalog.set_index("model_id")["display_label"]
    if "model_id" in result:
        result = result[result["model_id"].isin(labels.index)].copy()
        result["label"] = result["model_id"].map(labels)
    for model_column, label_column in (("left_model", "left"), ("right_model", "right")):
        if model_column in result:
            result = result[result[model_column].isin(labels.index)].copy()
            result[label_column] = result[model_column].map(labels)
    if {"source", "condition"}.issubset(result.columns):
        label_counts = catalog.groupby(["source", "condition"])["display_label"].nunique()
        if (label_counts <= 1).all():
            by_condition = catalog.drop_duplicates(["source", "condition"])
            mapping = by_condition.set_index(["source", "condition"])["display_label"]
            keys = pd.MultiIndex.from_frame(result[["source", "condition"]])
            mapped = mapping.reindex(keys).to_numpy()
            if len(mapped) == len(result):
                result["label"] = mapped
    return result
