"""Global visual-language configuration shared by all MSI plots."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Mapping, Sequence

import matplotlib as mpl

from ..utils.exceptions import raise_validation_error


@dataclass(frozen=True)
class VisualizationTheme:
    """Define the complete graphical strategy used by visualization helpers.

    The theme controls semantic colors, overlap transparency, line hierarchy,
    spatial colormaps, labels, legends, grids, and layer order. Plot functions
    consume these semantic settings instead of embedding local style choices.

    :param name: Human-readable theme identifier.
    :type name: str
    :param model_palette: Ordered colors assigned to analyzed models.
    :type model_palette: Sequence[str]
    :param class_palette: Ordered colors assigned to annotation classes.
    :type class_palette: Sequence[str]
    :param figure_size: Default Matplotlib figure size.
    :type figure_size: tuple[float, float]
    :param figure_dpi: Default figure resolution.
    :type figure_dpi: int
    :param model_overrides: Optional stable model-name color mapping.
    :type model_overrides: Mapping[str, str]
    """

    name: str = "diagnostic_light"
    model_palette: Sequence[str] = (
        "#0072B2",
        "#D55E00",
        "#009E73",
        "#CC79A7",
        "#E69F00",
        "#56B4E9",
        "#7A7A7A",
        "#6A3D9A",
    )
    class_palette: Sequence[str] = (
        "#2563EB",
        "#F97316",
        "#16A34A",
        "#9333EA",
        "#DC2626",
        "#0891B2",
        "#CA8A04",
        "#DB2777",
    )
    model_overrides: Mapping[str, str] = field(default_factory=dict)
    figure_size: tuple[float, float] = (10.5, 6.2)
    figure_dpi: int = 140
    background_color: str = "#FFFFFF"
    panel_color: str = "#FFFFFF"
    text_color: str = "#202124"
    muted_text_color: str = "#5F6368"
    spine_color: str = "#4B5563"
    grid_color: str = "#CBD5E1"
    input_color: str = "#64748B"
    residual_color: str = "#E11D48"
    baseline_color: str = "#111827"
    true_positive_color: str = "#16A34A"
    false_positive_color: str = "#DC2626"
    false_negative_color: str = "#F59E0B"
    ground_truth_color: str = "#2563EB"
    primary_alpha: float = 0.85
    secondary_alpha: float = 0.60
    overlapping_signal_alpha: float = 0.38
    uncertainty_alpha: float = 0.20
    mask_alpha: float = 0.32
    residual_alpha: float = 0.95
    input_line_width: float = 1.15
    reconstruction_line_width: float = 1.25
    residual_line_width: float = 1.35
    reference_line_width: float = 0.9
    line_width: float = 1.35
    marker_size: float = 4.5
    marker_alpha: float = 0.78
    marker_edge_width: float = 0.6
    distribution_fill_alpha: float = 0.18
    distribution_edge_alpha: float = 0.85
    distribution_line_width: float = 1.05
    spine_line_width: float = 0.75
    baseline_line_style: str = "--"
    image_colormap: str = "cividis"
    error_colormap: str = "magma"
    residual_colormap: str = "coolwarm"
    probability_colormap: str = "viridis"
    correctness_colormap: str = "RdYlGn"
    image_origin: str = "lower"
    image_interpolation: str = "nearest"
    shared_image_scale: bool = True
    title_font_size: float = 13.0
    label_font_size: float = 10.0
    tick_font_size: float = 9.0
    title_location: str = "left"
    grid_visible: bool = True
    grid_alpha: float = 0.30
    grid_line_width: float = 0.55
    grid_axis: str = "y"
    legend_location: str = "best"
    legend_frame: bool = False
    legend_columns: int = 1
    legend_font_size: float = 8.5
    input_zorder: int = 1
    reconstruction_zorder: int = 2
    residual_zorder: int = 3
    annotation_zorder: int = 4

    def __post_init__(self) -> None:
        """Validate values that affect every rendered plot.

        :raises ValidationError: If palettes are empty or transparency values
            fall outside the Matplotlib range.
        """
        if not self.model_palette or not self.class_palette:
            raise_validation_error(
                "VisualizationTheme", "Model and class palettes cannot be empty."
            )
        for field_name in (
            "primary_alpha",
            "secondary_alpha",
            "marker_alpha",
            "distribution_fill_alpha",
            "distribution_edge_alpha",
            "overlapping_signal_alpha",
            "uncertainty_alpha",
            "mask_alpha",
            "residual_alpha",
            "grid_alpha",
        ):
            value = float(getattr(self, field_name))
            if not 0.0 <= value <= 1.0:
                raise_validation_error(
                    "VisualizationTheme",
                    f"{field_name} must be between zero and one.",
                )
        if self.grid_axis not in {"both", "x", "y"}:
            raise_validation_error(
                "VisualizationTheme",
                "grid_axis must be one of: both, x, y.",
            )

    def apply(self) -> None:
        """Apply this theme to Matplotlib defaults before figure creation."""
        mpl.rcParams.update(
            {
                "figure.facecolor": self.background_color,
                "figure.dpi": self.figure_dpi,
                "savefig.facecolor": self.background_color,
                "savefig.bbox": "tight",
                "axes.facecolor": self.panel_color,
                "axes.edgecolor": self.spine_color,
                "axes.linewidth": self.spine_line_width,
                "axes.labelcolor": self.text_color,
                "axes.labelsize": self.label_font_size,
                "axes.titlesize": self.title_font_size,
                "axes.titleweight": "normal",
                "axes.axisbelow": True,
                "axes.grid": self.grid_visible,
                "axes.grid.axis": self.grid_axis,
                "grid.color": self.grid_color,
                "grid.alpha": self.grid_alpha,
                "grid.linewidth": self.grid_line_width,
                "xtick.color": self.muted_text_color,
                "ytick.color": self.muted_text_color,
                "xtick.labelsize": self.tick_font_size,
                "ytick.labelsize": self.tick_font_size,
                "lines.linewidth": self.line_width,
                "lines.markersize": self.marker_size,
                "legend.fontsize": self.legend_font_size,
                "legend.frameon": self.legend_frame,
                "text.color": self.text_color,
            }
        )

    def color_for_model(self, model_name: str, model_index: int = 0) -> str:
        """Return a stable semantic color for one model.

        :param model_name: Model identifier.
        :type model_name: str
        :param model_index: Position in the analyzed model collection.
        :type model_index: int
        :return: Matplotlib-compatible color.
        :rtype: str
        """
        return self.model_overrides.get(
            model_name,
            self.model_palette[model_index % len(self.model_palette)],
        )

    def color_for_class(self, class_index: int) -> str:
        """Return a stable color for one annotation class.

        :param class_index: Zero-based class index.
        :type class_index: int
        :return: Matplotlib-compatible color.
        :rtype: str
        """
        return self.class_palette[class_index % len(self.class_palette)]

    def with_overrides(self, **kwargs: Any) -> "VisualizationTheme":
        """Return an immutable theme copy with selected local overrides.

        :param kwargs: Dataclass fields to replace.
        :return: Independent theme instance.
        :rtype: VisualizationTheme
        """
        return replace(self, **kwargs)


THEME_PRESETS: Mapping[str, VisualizationTheme] = {
    "diagnostic_light": VisualizationTheme(),
    "publication_light": VisualizationTheme(
        name="publication_light",
        figure_size=(7.2, 4.4),
        figure_dpi=220,
        grid_visible=False,
        overlapping_signal_alpha=0.50,
        uncertainty_alpha=0.15,
        title_font_size=12.0,
        label_font_size=9.5,
        tick_font_size=8.5,
        legend_font_size=8.0,
        line_width=1.15,
        marker_size=4.0,
        legend_frame=False,
    ),
}


def resolve_theme(
    theme: VisualizationTheme | str | None,
) -> VisualizationTheme:
    """Resolve a theme object or registered preset name.

    :param theme: Theme instance, preset name, or ``None`` for the default.
    :type theme: VisualizationTheme | str | None
    :return: Resolved immutable theme.
    :rtype: VisualizationTheme
    :raises ValidationError: If a preset name is unknown.
    """
    if theme is None:
        resolved = THEME_PRESETS["diagnostic_light"]
    elif isinstance(theme, VisualizationTheme):
        resolved = theme
    elif theme not in THEME_PRESETS:
        raise_validation_error(
            "VisualizationTheme",
            f"Unknown theme preset '{theme}'. Available: {sorted(THEME_PRESETS)}.",
        )
    else:
        resolved = THEME_PRESETS[theme]
    resolved.apply()
    return resolved
