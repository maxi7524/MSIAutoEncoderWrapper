"""Presentation figures for the representative (toy) model and the concept slides.

All figures of the conceptual presentation share the style defined here, so a
change of font size, palette or export resolution is made in one place. Functions
receive precomputed tables and never load models.
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Mapping, Optional, Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# --------------------------------------------------
# Section: style
# --------------------------------------------------

PRESENTATION_RC = {
    "font.family": "DejaVu Sans",
    "font.size": 15,
    "axes.titlesize": 17,
    "axes.labelsize": 15,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "legend.fontsize": 13,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.edgecolor": "#3F3F46",
    "axes.labelcolor": "#18181B",
    "xtick.color": "#3F3F46",
    "ytick.color": "#3F3F46",
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "savefig.bbox": "tight",
    "savefig.dpi": 200,
}
"""Matplotlib parameters of every presentation figure."""

INK = "#18181B"
MUTED = "#71717A"
ACCENT = "#DC2626"
SPHERE_GRID = "#D4D4D8"
VARIANT_COLORS = {
    "m0": "#71717A",
    "m1": "#0072B2",
    "m2": "#D55E00",
    "m3": "#009E73",
    "m4": "#CC79A7",
}
"""Stable colour of each toy objective family (variant-name prefix) across all figures."""


def variant_color(variant: str) -> str:
    """Colour of a toy variant, keyed by its ``m<k>`` prefix."""
    return VARIANT_COLORS.get(variant.split("-")[0], INK)


@contextmanager
def presentation_style() -> Iterator[None]:
    """Apply the presentation Matplotlib style within a ``with`` block."""
    with mpl.rc_context(PRESENTATION_RC):
        yield


def save_figure(figure: plt.Figure, directory: Path | str, name: str, *, formats: Sequence[str] = ("png",)) -> list[Path]:
    """Save one figure under a stable name in every requested format.

    :param figure: Figure to save.
    :param directory: Output directory; created when absent.
    :param name: File stem.
    :param formats: File extensions.
    :return: Written paths.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    paths = []
    for extension in formats:
        path = directory / f"{name}.{extension}"
        figure.savefig(path)
        paths.append(path)
    return paths


# --------------------------------------------------
# Section: pixel maps
# --------------------------------------------------

def sphere_rgb(coordinates: np.ndarray) -> np.ndarray:
    """Colour points of the unit sphere by position: ``RGB = (p + 1) / 2``.

    Nearby points on the sphere receive similar colours, so the image of the
    colour map shows the spatial layout of the latent geometry.

    :param coordinates: Unit vectors, shape ``(N, 3)``.
    :return: RGB values in ``[0, 1]``, shape ``(N, 3)``.
    """
    return np.clip((np.asarray(coordinates) + 1.0) / 2.0, 0.0, 1.0)


def stretched_sphere_rgb(coordinates: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Colour sphere points by position, stretched to the range of ``reference``.

    The latent points of a small model often occupy a cap of the sphere, where
    ``(p + 1) / 2`` gives nearly one colour. Each coordinate is therefore mapped
    linearly from its range in ``reference`` (shared by all compared panels) to
    ``[0.1, 0.9]``.

    :param coordinates: Unit vectors to colour, shape ``(N, 3)``.
    :param reference: Unit vectors defining the colour range, shape ``(K, 3)``.
    :return: RGB values, shape ``(N, 3)``.
    """
    low = reference.min(axis=0)  # (3,)
    span = np.maximum(reference.max(axis=0) - low, 1e-9)  # (3,)
    return np.clip(0.1 + 0.8 * (np.asarray(coordinates) - low) / span, 0.0, 1.0)


def camera_facing(coordinates: np.ndarray) -> tuple[float, float]:
    """Elevation and azimuth (degrees) of a camera looking at the mean direction.

    :param coordinates: Unit vectors, shape ``(N, 3)``.
    :return: ``(elevation, azimuth)`` for ``Axes3D.view_init``.
    """
    mean = np.asarray(coordinates).mean(axis=0)
    mean = mean / np.linalg.norm(mean)
    return float(np.degrees(np.arcsin(mean[2]))), float(np.degrees(np.arctan2(mean[1], mean[0])))


def sphere_columns(frame: pd.DataFrame) -> list[str]:
    """Sphere-coordinate columns present in a table (3 for ``S^2``, 2 for ``S^1``)."""
    return [column for column in ("sphere_a", "sphere_b", "sphere_c") if column in frame]


def relative_angles(coordinates: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Signed angle (rad) of circle points relative to the mean direction of ``reference``."""
    mean = np.asarray(reference).mean(axis=0)
    origin = np.arctan2(mean[1], mean[0])
    return np.angle(np.exp(1j * (np.arctan2(coordinates[:, 1], coordinates[:, 0]) - origin)))


def position_colors(coordinates: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Colour points by latent position: stretched RGB on ``S^2``, stretched viridis on ``S^1``."""
    if coordinates.shape[1] == 3:
        return stretched_sphere_rgb(coordinates, reference)
    angles = relative_angles(coordinates, reference)
    reference_angles = relative_angles(reference, reference)
    low, high = reference_angles.min(), reference_angles.max()
    return mpl.colormaps["viridis"](np.clip((angles - low) / max(high - low, 1e-9), 0.0, 1.0))[:, :3]


def latent_axes(figure: plt.Figure, spec, dimension: int) -> plt.Axes:
    """3-D axes for ``S^2`` coordinates, plain axes for ``S^1``."""
    return figure.add_subplot(spec, projection="3d") if dimension == 3 else figure.add_subplot(spec)


def plot_pixel_map(
    ax: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    colors: np.ndarray,
    *,
    title: Optional[str] = None,
    background: str = "#F4F4F5",
) -> None:
    """Draw pixels of one image as coloured cells on their spatial grid.

    :param ax: Target axes.
    :param x: Zero-based column indices, shape ``(N,)``.
    :param y: Zero-based row indices, shape ``(N,)``.
    :param colors: RGB(A) colours, shape ``(N, 3)`` or ``(N, 4)``.
    :param title: Optional panel title.
    :param background: Colour of positions without a pixel.
    """
    x = np.asarray(x, dtype=int)
    y = np.asarray(y, dtype=int)
    colors = np.asarray(colors, dtype=float)
    image = np.ones((y.max() + 1, x.max() + 1, 4))
    image[..., :3] = mpl.colors.to_rgb(background)
    image[y, x, : colors.shape[1]] = colors
    if colors.shape[1] == 3:
        image[y, x, 3] = 1.0
    ax.imshow(image, interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    if title:
        ax.set_title(title)


# --------------------------------------------------
# Section: sphere
# --------------------------------------------------

def draw_sphere(ax: plt.Axes, *, elevation: float = 22.0, azimuth: float = 38.0) -> None:
    """Draw a light unit-sphere wireframe and hide the 3-D axis furniture."""
    longitudes = np.linspace(0, 2 * np.pi, 13)
    latitudes = np.linspace(-np.pi / 2, np.pi / 2, 7)
    grid = np.linspace(0, 2 * np.pi, 200)
    for longitude in longitudes:
        ax.plot(np.cos(grid) * np.cos(longitude), np.cos(grid) * np.sin(longitude), np.sin(grid),
                color=SPHERE_GRID, lw=0.6, zorder=0)
    for latitude in latitudes:
        ax.plot(np.cos(latitude) * np.cos(grid), np.cos(latitude) * np.sin(grid),
                np.full_like(grid, np.sin(latitude)), color=SPHERE_GRID, lw=0.6, zorder=0)
    ax.set_box_aspect((1, 1, 1), zoom=1.3)
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1)
    ax.view_init(elev=elevation, azim=azimuth)
    ax.set_axis_off()


def front_facing(coordinates: np.ndarray, *, elevation: float, azimuth: float) -> np.ndarray:
    """Return a boolean mask of sphere points facing the camera."""
    elevation_rad, azimuth_rad = np.radians(elevation), np.radians(azimuth)
    camera = np.array([np.cos(elevation_rad) * np.cos(azimuth_rad),
                       np.cos(elevation_rad) * np.sin(azimuth_rad),
                       np.sin(elevation_rad)])
    return np.asarray(coordinates) @ camera >= 0.0


def draw_circle(ax: plt.Axes, *, reference: Optional[np.ndarray] = None, margin: float = 0.12) -> None:
    """Draw the unit circle; with ``reference``, zoom to the arc it occupies.

    :param reference: Circle points defining the zoom window, shape ``(K, 2)``.
    :param margin: Extra space around the reference points (circle units).
    """
    grid = np.linspace(0, 2 * np.pi, 720)
    ax.plot(np.cos(grid), np.sin(grid), color=SPHERE_GRID, lw=1.2, zorder=0)
    ax.set_aspect("equal")
    if reference is None:
        ax.plot(0, 0, marker="+", color=SPHERE_GRID, ms=10, zorder=0)
        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-1.2, 1.2)
    else:
        low = reference.min(axis=0) - margin
        high = reference.max(axis=0) + margin
        half = max(high - low) / 2
        centre = (low + high) / 2
        ax.set_xlim(centre[0] - half, centre[0] + half)
        ax.set_ylim(centre[1] - half, centre[1] + half)
    ax.set_axis_off()


def plot_circle_points(
    ax: plt.Axes,
    coordinates: np.ndarray,
    colors,
    *,
    size: float = 30.0,
    jitter: float = 0.05,
    seed: int = 0,
    title: Optional[str] = None,
    reference: Optional[np.ndarray] = None,
) -> None:
    """Scatter ``S^1`` points; a small radial jitter separates coinciding points.

    :param jitter: Half-width of the uniform radial offset (display only).
    :param reference: Points defining the zoomed arc (all compared panels).
    """
    draw_circle(ax, reference=reference)
    radius = 1.0 + np.random.default_rng(seed).uniform(-jitter, jitter, len(coordinates))
    ax.scatter(coordinates[:, 0] * radius, coordinates[:, 1] * radius, c=colors, s=size,
               edgecolors="white", linewidths=0.4, zorder=2)
    if title:
        ax.set_title(title)


def plot_circle_rings(
    ax: plt.Axes,
    coordinates: np.ndarray,
    regions: np.ndarray,
    region_colors: Mapping[str, str],
    *,
    background: Sequence[str] = (),
    ring_step: float = 0.075,
    title: Optional[str] = None,
) -> None:
    """``S^1`` points with one concentric ring per category.

    On the circle the angle is the only latent coordinate, so every category is drawn at
    its own radius (the angle is unchanged). Background categories share the innermost
    ring; the unit circle itself is the reference.
    """
    draw_circle(ax)
    ax.set_xlim(-1.75, 1.75)
    ax.set_ylim(-1.75, 1.75)
    angles = np.arctan2(coordinates[:, 1], coordinates[:, 0])  # (N,)
    foreground = [name for name in region_colors if name not in background and (regions == name).any()]
    rng = np.random.default_rng(0)
    for name in [*background, *foreground]:
        mask = regions == name
        if not mask.any():
            continue
        is_background = name in background
        radius = 0.88 if is_background else 1.0 + ring_step * (1 + foreground.index(name))
        radius = radius + rng.uniform(-0.015, 0.015, mask.sum())
        ax.scatter(radius * np.cos(angles[mask]), radius * np.sin(angles[mask]), s=4 if is_background else 14,
                   c=region_colors[name], edgecolors="none", alpha=0.6 if is_background else 0.9, zorder=2)
    if title:
        ax.set_title(title)


def plot_sphere_points(
    ax: plt.Axes,
    coordinates: np.ndarray,
    colors: np.ndarray | str,
    *,
    size: float = 34.0,
    elevation: float = 22.0,
    azimuth: float = 38.0,
    back_alpha: float = 0.18,
    edgecolor: str = "white",
    title: Optional[str] = None,
    reference: Optional[np.ndarray] = None,
) -> None:
    """Scatter unit-sphere points, fading those on the hidden hemisphere.

    :param ax: 3-D axes.
    :param coordinates: Unit vectors, shape ``(N, 3)``.
    :param colors: One colour or RGB(A) per point.
    :param size: Marker area.
    :param back_alpha: Opacity of points on the far hemisphere.
    """
    coordinates = np.asarray(coordinates)
    if coordinates.shape[1] == 2:
        plot_circle_points(ax, coordinates, colors, size=size, title=title, reference=reference)
        return
    draw_sphere(ax, elevation=elevation, azimuth=azimuth)
    front = front_facing(coordinates, elevation=elevation, azimuth=azimuth)
    rgba = mpl.colors.to_rgba_array(colors, alpha=None) if not isinstance(colors, str) else mpl.colors.to_rgba_array([colors] * len(coordinates))
    sizes = np.broadcast_to(np.asarray(size, dtype=float), (len(coordinates),))
    for mask, alpha in ((~front, back_alpha), (front, 1.0)):
        if not mask.any():
            continue
        point_colors = rgba[mask].copy()
        point_colors[:, 3] = alpha
        ax.scatter(coordinates[mask, 0], coordinates[mask, 1], coordinates[mask, 2], c=point_colors,
                   s=sizes[mask], edgecolors=edgecolor if alpha == 1.0 else "none", linewidths=0.3,
                   depthshade=False)
    if title:
        ax.set_title(title, pad=0)


def plotly_sphere_html(
    frames: Mapping[str, pd.DataFrame],
    path: Path | str,
    *,
    color_column: str,
    color_map: Mapping[str, str],
    hover_columns: Sequence[str] = (),
    title: str = "",
    show_legend: bool = True,
    display_column: str | None = None,
) -> Path:
    """Write an interactive, rotatable unit-sphere scatter with one view per variant.

    :param frames: Variant label -> table with ``sphere_a``, ``sphere_b``, ``sphere_c``
        and ``color_column``.
    :param path: Output HTML file; plotly.js is loaded from its CDN.
    :param color_column: Categorical column defining trace colours.
    :param color_map: Category -> colour.
    :param hover_columns: Extra columns shown on hover.
    :param title: Figure title.
    :param show_legend: Whether Plotly draws the category legend.
    :param display_column: Optional boolean column defining the sampled pixel subset.
        When provided, each variant receives sampled and all-pixel menu entries.
    :return: Written path.
    """
    import plotly.graph_objects as go

    grid = np.linspace(0, 2 * np.pi, 60)
    sphere_u, sphere_v = np.meshgrid(grid, np.linspace(0, np.pi, 30))
    surface = go.Surface(
        x=np.cos(sphere_u) * np.sin(sphere_v), y=np.sin(sphere_u) * np.sin(sphere_v), z=np.cos(sphere_v),
        opacity=0.12, showscale=False, colorscale=[[0, "#A1A1AA"], [1, "#A1A1AA"]], hoverinfo="skip",
    )
    figure = go.Figure()
    figure.add_trace(surface)
    button_trace_indices = []
    trace_index = 1
    for variant_index, (label, frame) in enumerate(frames.items()):
        if display_column is None:
            views = [("", frame)]
        else:
            if display_column not in frame:
                raise KeyError(f"Frame {label!r} has no display column {display_column!r}.")
            sampled = frame.loc[frame[display_column].fillna(False).astype(bool)]
            views = [("sample", sampled), ("all pixels", frame)]

        for view_index, (view_label, view_frame) in enumerate(views):
            active_traces = []
            for category, colour in color_map.items():
                subset = view_frame[view_frame[color_column] == category]
                if subset.empty:
                    continue
                hover = subset[list(hover_columns)].astype(str).agg("<br>".join, axis=1) if hover_columns else None
                figure.add_trace(go.Scatter3d(
                    x=subset.sphere_a, y=subset.sphere_b, z=subset.sphere_c, mode="markers", name=str(category),
                    marker={"size": 4.5, "color": colour, "line": {"width": 0.5, "color": "white"}},
                    text=hover, hoverinfo="text+name", visible=variant_index == 0 and view_index == 0,
                    legendgroup=str(category), showlegend=True,
                ))
                active_traces.append(trace_index)
                trace_index += 1

            button_label = f"{label} · {view_label}" if view_label else label
            button_trace_indices.append((button_label, active_traces))

    buttons = []
    for label, active_traces in button_trace_indices:
        active_set = set(active_traces)
        visible = [True] + [index in active_set for index in range(1, trace_index)]
        buttons.append({"label": label, "method": "restyle", "args": [{"visible": visible}]})
    axis = {"visible": False, "range": [-1.05, 1.05]}
    figure.update_layout(
        title=title, template="plotly_white", showlegend=show_legend,
        margin={"l": 0, "r": 0, "t": 40, "b": 0},
        scene={"xaxis": axis, "yaxis": axis, "zaxis": axis, "aspectmode": "cube"},
        updatemenus=[{"buttons": buttons, "direction": "down", "x": 0.02, "y": 0.98,
                      "xanchor": "left", "yanchor": "top", "active": 0, "showactive": True}],
        legend={"itemsizing": "constant", "x": 0.60, "xanchor": "left", "y": 0.96, "yanchor": "top"},
    )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.write_html(path, include_plotlyjs="cdn", full_html=True)
    return path


# --------------------------------------------------
# Section: toy-model figures
# --------------------------------------------------

def plot_region_and_ions(
    pixels: pd.DataFrame,
    classes: pd.DataFrame,
    region_colors: Mapping[str, str],
    *,
    selected_ions: Optional[Sequence[Mapping]] = None,
    ion_columns: int = 5,
) -> plt.Figure:
    """Category map of the toy image next to the image of every selected ion.

    :param pixels: Pixel table with ``x``, ``y``, ``region`` and ``label::<class>`` columns.
    :param classes: Class table with ``class_name`` and ``mz``.
    :param region_colors: Category -> colour, in legend order.
    :param selected_ions: ``{"mz", "color"}`` per ion to show; all classes (black) when omitted.
    :param ion_columns: Number of ion panels per row.
    :return: Figure.
    """
    if selected_ions is None:
        selected_ions = [{"mz": mz, "color": INK} for mz in classes.sort_values("mz").mz]
    rows = int(np.ceil(len(selected_ions) / ion_columns))
    width = pixels.x.max() + 1
    height = pixels.y.max() + 1
    aspect = width / height
    columns = min(ion_columns, len(selected_ions))
    figure = plt.figure(figsize=(16, 16 / (2.6 + columns) * 2.6 / aspect * 0.62 + 1.2))
    grid = figure.add_gridspec(rows, columns + 1, width_ratios=[2.6] + [1] * columns, wspace=0.06,
                               hspace=0.25, bottom=0.14)
    region_ax = figure.add_subplot(grid[:, 0])
    plot_pixel_map(region_ax, pixels.x, pixels.y,
                   np.array([mpl.colors.to_rgb(region_colors[r]) for r in pixels.region]),
                   title=f"toy image: {len(pixels)} pixels")
    handles = [mpl.patches.Patch(color=colour, label=name) for name, colour in region_colors.items()
               if (pixels.region == name).any()]
    figure.legend(handles=handles, loc="lower center", ncol=len(handles), frameon=False, fontsize=12,
                  bbox_to_anchor=(0.5, 0.0))
    for position, ion in enumerate(selected_ions):
        row = classes.loc[(classes.mz - float(ion["mz"])).abs().idxmin()]
        ax = figure.add_subplot(grid[position // ion_columns, 1 + position % ion_columns])
        present = pixels[f"label::{row.class_name}"].to_numpy() > 0
        colors = np.where(present[:, None], mpl.colors.to_rgb(ion["color"]), mpl.colors.to_rgb("#E4E4E7"))
        plot_pixel_map(ax, pixels.x, pixels.y, colors)
        ax.set_title(f"m/z {row.mz:.1f}", fontsize=13, color=ion["color"])
    return figure


def plot_sphere_with_image(
    axes_sphere: plt.Axes,
    axes_image: plt.Axes,
    frame: pd.DataFrame,
    region_colors: Mapping[str, str],
    *,
    title: str,
    reference: np.ndarray,
    elevation: float,
    azimuth: float,
    background: Sequence[str] = (),
) -> None:
    """Sphere and image, both coloured by the display category of every pixel.

    Points of ``background`` categories are drawn first and small, so the categories of
    interest stay visible on the sphere.

    :param axes_sphere: 3-D axes for the sphere.
    :param axes_image: 2-D axes for the pixel map.
    :param frame: Latent table of one variant with ``sphere_a/b/c``, ``x``, ``y``, ``region``;
        an optional boolean ``display`` column selects the points drawn on the sphere.
    :param region_colors: Region -> colour.
    :param title: Panel title.
    :param reference: Coordinates defining the shared colour range, shape ``(K, 3)``.
    """
    coordinates = frame[sphere_columns(frame)].to_numpy()
    shown = frame["display"].to_numpy(dtype=bool) if "display" in frame else np.ones(len(frame), dtype=bool)
    regions = frame.region.to_numpy()[shown]
    is_background = np.isin(regions, list(background))
    if coordinates.shape[1] == 2:
        plot_circle_rings(axes_sphere, coordinates[shown], regions, region_colors, background=background,
                          title=title)
        plot_pixel_map(axes_image, frame.x, frame.y,
                       np.array([mpl.colors.to_rgb(region_colors[r]) for r in frame.region]))
        return
    order = np.argsort(~is_background, kind="stable")  # background first, classes on top
    plot_sphere_points(axes_sphere, coordinates[shown][order], [region_colors[r] for r in regions[order]],
                       elevation=elevation, azimuth=azimuth, title=title,
                       size=np.where(is_background[order], 5.0, 22.0))
    plot_pixel_map(axes_image, frame.x, frame.y,
                   np.array([mpl.colors.to_rgb(region_colors[r]) for r in frame.region]))


def plot_variant_pair(
    frames: Sequence[tuple[str, pd.DataFrame]],
    region_colors: Mapping[str, str],
    *,
    background: Sequence[str] = (),
) -> plt.Figure:
    """Side-by-side variants: sphere (region colours) above image (sphere colours).

    The camera looks at the mean latent direction of all shown variants and the image
    colours share one range, so the panels are directly comparable.

    :param frames: ``(label, latent table)`` per variant, in display order.
    :param region_colors: Region -> colour, in legend order.
    :return: Figure.
    """
    pooled = np.concatenate([frame[sphere_columns(frame)].to_numpy() for _, frame in frames])
    dimension = pooled.shape[1]
    elevation, azimuth = camera_facing(pooled) if dimension == 3 else (0.0, 0.0)
    figure = plt.figure(figsize=(6.2 * len(frames), 9.4))
    grid = figure.add_gridspec(2, len(frames), height_ratios=[1.35, 1.0], hspace=0.02, wspace=0.05)
    for column, (label, frame) in enumerate(frames):
        sphere_ax = latent_axes(figure, grid[0, column], dimension)
        image_ax = figure.add_subplot(grid[1, column])
        plot_sphere_with_image(sphere_ax, image_ax, frame, region_colors, title=label, reference=pooled,
                               elevation=elevation, azimuth=azimuth, background=background)
    handles = [mpl.patches.Patch(color=colour, label=name) for name, colour in region_colors.items()]
    figure.legend(handles=handles, loc="lower center", ncol=len(handles), frameon=False, fontsize=12,
                  bbox_to_anchor=(0.5, -0.01))
    return figure


def plot_ion_presence_spheres(
    frames: Sequence[tuple[str, pd.DataFrame]],
    ions: Sequence[tuple[str, str]],
    *,
    present_color: str = "#DC2626",
    absent_color: str = "#D4D4D8",
) -> plt.Figure:
    """Sphere per variant (rows) and ion (columns), coloured by annotation presence.

    :param frames: ``(label, latent table with label::<class> columns)`` per variant.
    :param ions: ``(row label, label column)`` per ion.
    :return: Figure.
    """
    pooled = np.concatenate([frame[sphere_columns(frame)].to_numpy() for _, frame in frames])
    dimension = pooled.shape[1]
    elevation, azimuth = camera_facing(pooled) if dimension == 3 else (0.0, 0.0)
    figure = plt.figure(figsize=(4.6 * len(ions), 4.6 * len(frames)))
    grid = figure.add_gridspec(len(frames), len(ions))
    for row, (label, frame) in enumerate(frames):
        if "display" in frame:
            frame = frame[frame["display"].to_numpy(dtype=bool)]
        for column, (ion_label, column_name) in enumerate(ions):
            ax = latent_axes(figure, grid[row, column], dimension)
            present = frame[column_name].to_numpy() > 0
            order = np.argsort(present)  # present points drawn last
            coordinates = frame[sphere_columns(frame)].to_numpy()[order]
            colors = np.where(present[order], present_color, absent_color)
            plot_sphere_points(ax, coordinates, list(colors), size=20, elevation=elevation, azimuth=azimuth,
                               title=f"m/z {ion_label}" if row == 0 else None, reference=pooled)
            if column == 0:
                text = ax.text2D if dimension == 3 else ax.text
                text(-0.06, 0.5, label, transform=ax.transAxes, rotation=90, va="center", ha="center",
                     fontsize=15)
    figure.subplots_adjust(wspace=0.0, hspace=0.05)
    return figure


def plot_perturbation_clouds(
    frames: Sequence[tuple[str, pd.DataFrame]],
    region_of: Mapping[int, str],
    region_colors: Mapping[str, str],
    *,
    moved_kind: str = "perturbed",
) -> plt.Figure:
    """Perturbed copies of a few spectra (small points) around their originals (large).

    :param frames: ``(label, display-cloud table)`` per variant.
    :param moved_kind: ``kind`` value of the moved copies (``perturbed`` or ``view``).
    :param region_of: Source identifier -> region of the display pixels.
    :param region_colors: Region -> colour.
    :return: Figure.
    """
    pooled = np.concatenate([frame[sphere_columns(frame)].to_numpy() for _, frame in frames])
    dimension = pooled.shape[1]
    elevation, azimuth = camera_facing(pooled) if dimension == 3 else (0.0, 0.0)
    figure = plt.figure(figsize=(6.4 * len(frames), 6.4))
    grid = figure.add_gridspec(1, len(frames))
    for column, (label, frame) in enumerate(frames):
        ax = latent_axes(figure, grid[0, column], dimension)
        if dimension == 3:
            draw_sphere(ax, elevation=elevation, azimuth=azimuth)
        else:
            draw_circle(ax, reference=pooled, margin=0.05)
        for kind, size, edge in ((moved_kind, 10, "none"), ("original", 150, INK)):
            subset = frame[frame.kind == kind]
            colors = [region_colors[region_of[int(source)]] for source in subset.source_id]
            points = [subset[c] for c in sphere_columns(subset)]
            style = {"c": colors, "s": size, "edgecolors": edge, "linewidths": 1.2 if kind == "original" else 0,
                     "alpha": 0.55 if kind == moved_kind else 1.0}
            if dimension == 3:
                ax.scatter(*points, depthshade=False, **style)
            else:
                ax.scatter(*points, zorder=3 if kind == "original" else 2, **style)
        ax.set_title(label, pad=0)
    return figure


def plot_angle_curves(
    summary: pd.DataFrame,
    *,
    x: str,
    labels: Mapping[str, str],
    xlabel: str,
    ylabel: str = "latent angle  $\\angle(u, u')$  [deg]",
    dashed: Sequence[str] = (),
) -> plt.Figure:
    """Median latent angle (line) with interquartile band per variant.

    :param summary: Columns ``variant``, ``x``, ``median``, ``q25``, ``q75``.
    :param x: Name of the abscissa column.
    :param labels: Variant -> display label.
    :param dashed: Variants drawn with dashed lines (e.g. the weaker weight level).
    :return: Figure.
    """
    figure, ax = plt.subplots(figsize=(10, 5.6))
    for variant, frame in summary.groupby("variant", sort=False):
        colour = variant_color(variant)
        frame = frame.sort_values(x)
        ax.plot(frame[x], frame["median"], marker="o", color=colour, lw=2.2, label=labels.get(variant, variant),
                ls="--" if variant in dashed else "-")
        ax.fill_between(frame[x], frame["q25"], frame["q75"], color=colour, alpha=0.15, lw=0)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(frameon=False)
    ax.grid(axis="y", color="#E4E4E7", lw=0.8)
    return figure


def plot_angle_distributions(
    angles: pd.DataFrame,
    *,
    group: str,
    labels: Mapping[str, str],
    group_labels: Mapping[str, str],
    value: str = "latent_angle_deg",
    ylabel: str = "latent angle  $\\angle(u(x), u(\\tilde x))$  [deg]",
) -> plt.Figure:
    """Per-variant distributions (box + all points) of an angle, split by a grouping column.

    :param angles: Long table with ``variant``, ``group`` and ``value`` columns.
    :param group: Column separating the boxes of one variant.
    :param labels: Variant -> display label.
    :param group_labels: Group value -> legend label.
    :return: Figure.
    """
    variants = list(dict.fromkeys(angles.variant))
    groups = list(dict.fromkeys(angles[group]))
    width = 0.8 / len(groups)
    figure, ax = plt.subplots(figsize=(max(12, 2.0 * len(variants)), 5.8))
    rng = np.random.default_rng(0)
    for g_index, group_value in enumerate(groups):
        for v_index, variant in enumerate(variants):
            values = angles[(angles.variant == variant) & (angles[group] == group_value)][value].to_numpy()
            position = v_index + (g_index - (len(groups) - 1) / 2) * width
            colour = variant_color(variant)
            ax.scatter(position + rng.uniform(-width / 3, width / 3, len(values)), values, s=5, color=colour,
                       alpha=0.18 if g_index == 0 else 0.10, lw=0)
            box = ax.boxplot(values, positions=[position], widths=width * 0.7, showfliers=False, patch_artist=True)
            for patch in box["boxes"]:
                patch.set(facecolor="none", edgecolor=colour, lw=1.8, hatch=None if g_index == 0 else "//")
            for key in ("whiskers", "caps", "medians"):
                for line in box[key]:
                    line.set(color=colour, lw=1.6)
    ax.set_xticks(range(len(variants)), [labels.get(v, v).replace(" + ", "\n+ ") for v in variants], rotation=0,
                  fontsize=11)
    ax.set_ylabel(ylabel)
    handles = [mpl.patches.Patch(facecolor="white", edgecolor=INK, hatch=None if i == 0 else "//",
                                 label=group_labels.get(g, g)) for i, g in enumerate(groups)]
    ax.legend(handles=handles, frameon=False)
    ax.grid(axis="y", color="#E4E4E7", lw=0.8)
    return figure
