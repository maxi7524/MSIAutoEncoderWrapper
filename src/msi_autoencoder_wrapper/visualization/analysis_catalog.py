"""Shared visualization encodings exported by analysis precompute model catalogs."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .theme import VisualizationTheme, resolve_theme


_REQUIRED_COLUMNS = {"model_alias", "display_label", "display_order", "color", "line_style", "marker"}


def load_visualization_contract(path: Path | str) -> pd.DataFrame:
    """Load and validate a catalog-generated visualization contract.

    :param path: ``visualization_contract.csv`` created by common precompute.
    :type path: pathlib.Path | str
    :return: One ordered row per logical model alias.
    :rtype: pandas.DataFrame
    :raises ValueError: If a required field is missing or display labels collide.
    """
    frame = pd.read_csv(path)
    missing = sorted(_REQUIRED_COLUMNS - set(frame.columns))
    if missing:
        raise ValueError(f"Visualization contract is missing required columns: {missing}.")
    if frame.model_alias.duplicated().any():
        raise ValueError("Visualization contract contains duplicate model aliases.")
    if frame.display_label.duplicated().any():
        raise ValueError("Visualization contract contains duplicate display labels.")
    return frame.sort_values(["display_order", "model_alias"]).reset_index(drop=True)


def theme_from_visualization_contract(
    contract: pd.DataFrame,
    theme: VisualizationTheme | str | None = None,
) -> VisualizationTheme:
    """Return a theme with catalog-defined stable display-label colours.

    :param contract: Validated rows returned by :func:`load_visualization_contract`.
    :type contract: pandas.DataFrame
    :param theme: Base shared visualization theme or preset.
    :type theme: VisualizationTheme | str | None
    :return: Theme whose model overrides match every catalog display label.
    :rtype: VisualizationTheme
    """
    resolved = resolve_theme(theme)
    colors = dict(zip(contract.display_label, contract.color))
    return resolved.with_overrides(model_overrides={**resolved.model_overrides, **colors})


def model_style_map(contract: pd.DataFrame) -> dict[str, dict[str, str]]:
    """Return line and marker styles keyed by presentation label.

    :param contract: Validated rows returned by :func:`load_visualization_contract`.
    :type contract: pandas.DataFrame
    :return: Mapping from display label to color, line style, and marker.
    :rtype: dict[str, dict[str, str]]
    """
    return {
        row.display_label: {
            "color": row.color,
            "line_style": row.line_style,
            "marker": row.marker,
        }
        for row in contract.itertuples(index=False)
    }
