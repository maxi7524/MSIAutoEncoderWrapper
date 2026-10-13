"""Plots of chemical taxonomies, label graphs and their projection on MSI images."""

from __future__ import annotations

from typing import Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.patches import Patch

from .metrics import plot_violin_with_points
from .theme import VisualizationTheme, resolve_theme

# Taxonomy colours: lipid-specific taxonomies in green hues, broad ones in blue/purple.
TAXONOMY_COLORS = {"LIPID_MAPS": "#15803D", "SwissLipids": "#65A30D", "ClassyFire": "#1D4ED8", "ChEBI": "#7C3AED"}
TAXONOMY_LABELS = {"LM": "LIPID_MAPS", "SL": "SwissLipids", "CF": "ClassyFire", "ChEBI": "ChEBI"}
COLLECTION_COLORS = {"kidney": "#B45309", "brain": "#0E7490"}
RESOLUTION_STYLES = {"metaspace_ids": "o", "formula": "s"}
ION_COLOR = "#9CA3AF"


def taxonomy_color(label: str) -> str:
    """Return the colour of a taxonomy or taxonomy combination label.

    Single taxonomies use :data:`TAXONOMY_COLORS`; combinations use a grey whose
    darkness grows with the number of combined taxonomies.

    :param label: Taxonomy name or combination label such as ``LM+CF``.
    :type label: str
    :return: Hex colour.
    :rtype: str
    """
    if label in TAXONOMY_COLORS:
        return TAXONOMY_COLORS[label]
    parts = label.split("+")
    if len(parts) == 1:
        return TAXONOMY_COLORS.get(TAXONOMY_LABELS.get(label, label), "#374151")
    return {2: "#9CA3AF", 3: "#6B7280", 4: "#111827"}.get(len(parts), "#374151")


def violin_groups(
    ax: Axes,
    frame: pd.DataFrame,
    *,
    group: str,
    value: str,
    order: Sequence[str],
    colors: Mapping[str, str] | None = None,
    reference: float | None = None,
    max_points: int = 400,
    seed: int = 0,
    theme: VisualizationTheme | str | None = None,
) -> Axes:
    """Draw one violin with overlaid observations per group.

    The violin uses every finite observation; at most ``max_points`` observations
    per group (a seeded uniform subset) are drawn as points.

    :param ax: Target axes.
    :type ax: matplotlib.axes.Axes
    :param frame: Long table.
    :type frame: pandas.DataFrame
    :param group: Grouping column.
    :type group: str
    :param value: Value column.
    :type value: str
    :param order: Group order along x.
    :type order: collections.abc.Sequence[str]
    :param colors: Optional group colours.
    :type colors: collections.abc.Mapping[str, str] | None
    :param reference: Optional horizontal reference level.
    :type reference: float | None
    :param max_points: Maximum overlaid points per group.
    :type max_points: int
    :param seed: Seed of the displayed subset.
    :type seed: int
    :param theme: Visualization theme.
    :type theme: VisualizationTheme | str | None
    :return: The axes.
    :rtype: matplotlib.axes.Axes
    """
    resolved = resolve_theme(theme)
    generator = np.random.default_rng(seed)
    for position, name in enumerate(order):
        values = frame.loc[frame[group] == name, value].to_numpy(dtype=float)
        values = values[np.isfinite(values)]
        overlay = values if values.size <= max_points else generator.choice(values, size=max_points, replace=False)
        plot_violin_with_points(values, position, ax=ax, color=(colors or {}).get(name, resolved.model_palette[position % len(resolved.model_palette)]),
                                overlay_values=overlay, point_size=8, theme=resolved)
    ax.set_xticks(range(len(order)), order, rotation=30, ha="right")
    ax.set_ylabel(value)
    if reference is not None:
        ax.axhline(reference, color=resolved.baseline_color, linestyle=resolved.baseline_line_style,
                   linewidth=resolved.reference_line_width)
    return ax


def layered_layout(nodes: pd.DataFrame, edges: pd.DataFrame) -> dict[str, tuple[float, float]]:
    """Place ions on the bottom row and classes on rows by hierarchy depth.

    Row of a class: ``max_depth - depth + 1`` (roots on top). With several
    taxonomies, each taxonomy occupies its own vertical band of equal width, so that
    classes of different hierarchies are not interleaved. Within a band and row,
    classes are ordered by the barycentre of already placed neighbours and spread
    evenly; ions are spread over the full width ordered by their neighbours.

    :param nodes: ``node_id``, ``node_type`` (``ion``/``class``), ``depth`` and
        optionally ``taxonomy``.
    :type nodes: pandas.DataFrame
    :param edges: ``source``, ``target``.
    :type edges: pandas.DataFrame
    :return: Position per node identifier.
    :rtype: dict[str, tuple[float, float]]
    """
    classes = nodes[nodes.node_type == "class"]
    top = int(classes.depth.max()) if len(classes) else 0
    taxonomy_of = dict(zip(nodes.node_id, nodes.get("taxonomy", pd.Series([""] * len(nodes))).fillna("")))
    row = {node: 0 if kind == "ion" else top - int(depth) + 1
           for node, kind, depth in zip(nodes.node_id, nodes.node_type, nodes.depth.fillna(0))}
    bands = sorted({taxonomy_of[node] for node in classes.node_id}) or [""]
    neighbours: dict[str, list[str]] = {}
    for source, target in edges[["source", "target"]].itertuples(index=False):
        neighbours.setdefault(source, []).append(target)
        neighbours.setdefault(target, []).append(source)
    ## Common width: the most populated row of any band
    sizes = pd.Series([(row[n], taxonomy_of[n]) for n in classes.node_id]).value_counts() if len(classes) else pd.Series(dtype=int)
    band_width = float(max([int(sizes.max()) if len(sizes) else 1, 1]))
    width = band_width * len(bands)
    x: dict[str, float] = {}
    ## Ions: spread over the full width, grouped by their first class neighbour
    ions = sorted((n for n, r in row.items() if r == 0),
                  key=lambda n: (taxonomy_of.get(min(neighbours.get(n, [""])), ""), min(neighbours.get(n, [""]))))
    x.update({node: (i + 0.5) * width / max(len(ions), 1) for i, node in enumerate(ions)})
    ## Classes: per row and band, barycentric order
    for level in range(1, top + 2):
        for band_index, band in enumerate(bands):
            members = [n for n, r in row.items() if r == level and taxonomy_of[n] == band]
            if not members:
                continue
            estimate = {n: np.mean([x[m] for m in neighbours.get(n, []) if m in x] or [band_index * band_width])
                        for n in members}
            ordered = sorted(members, key=lambda n: estimate[n])
            x.update({n: band_index * band_width + (i + 0.5) * band_width / len(ordered) for i, n in enumerate(ordered)})
    return {node: (x[node], float(row[node])) for node in row}


def draw_label_graph(
    ax: Axes,
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
    *,
    label_classes: bool = True,
    max_label_length: int = 28,
    theme: VisualizationTheme | str | None = None,
) -> Axes:
    """Draw a label subgraph with ions at the bottom and classes above.

    Node colour encodes the taxonomy of class nodes (ions are grey); node size grows
    with ``support``. Edge style encodes the relation: solid ``belongs_to``, thick
    ``is_a``, dashed ``equivalent_to``, dotted ``same_formula``.

    :param ax: Target axes.
    :type ax: matplotlib.axes.Axes
    :param nodes: Subgraph nodes (see :func:`layered_layout`), optionally with
        ``taxonomy``, ``name`` and ``support``.
    :type nodes: pandas.DataFrame
    :param edges: Subgraph edges with ``relation``.
    :type edges: pandas.DataFrame
    :param label_classes: Write class names next to class nodes.
    :type label_classes: bool
    :param max_label_length: Truncation length of class names.
    :type max_label_length: int
    :param theme: Visualization theme.
    :type theme: VisualizationTheme | str | None
    :return: The axes.
    :rtype: matplotlib.axes.Axes
    """
    resolved = resolve_theme(theme)
    position = layered_layout(nodes, edges)
    styles = {"belongs_to": ("-", 0.5, 0.35), "is_a": ("-", 1.4, 0.8), "equivalent_to": ("--", 1.2, 0.9),
              "same_formula": (":", 1.0, 0.8)}
    for source, target, relation in edges[["source", "target", "relation"]].itertuples(index=False):
        if source not in position or target not in position:
            continue
        style, width, alpha = styles.get(relation, ("-", 0.5, 0.3))
        (x0, y0), (x1, y1) = position[source], position[target]
        ax.plot([x0, x1], [y0, y1], linestyle=style, linewidth=width, alpha=alpha, color=resolved.spine_color, zorder=1)
    for record in nodes.itertuples(index=False):
        x, y = position[record.node_id]
        is_ion = record.node_type == "ion"
        color = ION_COLOR if is_ion else TAXONOMY_COLORS.get(getattr(record, "taxonomy", ""), "#374151")
        support = getattr(record, "support", np.nan)
        size = 18 if is_ion else 30 + 6 * np.sqrt(support if np.isfinite(support) else 1)
        ax.scatter([x], [y], s=size, color=color, edgecolor="white", linewidth=0.5, zorder=3)
        if label_classes and not is_ion:
            name = str(getattr(record, "name", record.node_id))
            ax.text(x, y + 0.12, name[:max_label_length], fontsize=6.5, ha="center", va="bottom", rotation=20, zorder=4)
    handles = [Patch(color=ION_COLOR, label="ion (formula|adduct)")] + [
        Patch(color=c, label=n) for n, c in TAXONOMY_COLORS.items() if n in set(nodes.get("taxonomy", pd.Series(dtype=str)))]
    ax.legend(handles=handles, fontsize=7, loc="upper right", frameon=False)
    ax.set_yticks([])
    ax.set_xticks([])
    ax.set_ylabel("ions (bottom) → class depth → roots (top)")
    return ax


def image_from_pixels(coordinates: np.ndarray, mask: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Place masked pixel values on the image grid.

    :param coordinates: Pixel coordinates ``(P, 2)`` as ``(x, y)``.
    :type coordinates: numpy.ndarray
    :param mask: Pixel mask ``(P,)``.
    :type mask: numpy.ndarray
    :param values: Values of masked pixels ``(P_+,)``.
    :type values: numpy.ndarray
    :return: Image ``(H, W)`` with ``nan`` background.
    :rtype: numpy.ndarray
    """
    image = np.full((coordinates[:, 1].max() + 1, coordinates[:, 0].max() + 1), np.nan, dtype=np.float32)
    selected = coordinates[mask]
    image[selected[:, 1], selected[:, 0]] = values
    return image


def continuous_map(ax: Axes, image: np.ndarray, *, title: str, colorbar_label: str, vmin: float | None = None,
                   vmax: float | None = None, theme: VisualizationTheme | str | None = None) -> Axes:
    """Show one continuous per-pixel quantity with a colour bar."""
    resolved = resolve_theme(theme)
    handle = ax.imshow(image, cmap=resolved.image_colormap, vmin=vmin, vmax=vmax, interpolation="nearest")
    ax.set_title(title, fontsize=9)
    ax.set_axis_off()
    plt.colorbar(handle, ax=ax, fraction=0.046, pad=0.02, label=colorbar_label)
    return ax


def categorical_map(ax: Axes, image: np.ndarray, labels: Sequence[str], *, title: str,
                    theme: VisualizationTheme | str | None = None) -> Axes:
    """Show integer class indices (``-1`` = no class) with a legend of class names.

    Only indices present in the image receive colours, assigned in order of pixel
    frequency from a 20-colour qualitative palette, so that no two shown classes
    share a colour (up to 20 classes; rarer classes beyond that are grey).
    """
    resolve_theme(theme)
    values = image[np.isfinite(image) & (image >= 0)].astype(int)
    present = list(pd.Series(values).value_counts().index) if values.size else []
    palette = [plt.get_cmap("tab20")(i) for i in range(20)]
    colors = {index: (palette[rank] if rank < 20 else (0.7, 0.7, 0.7, 1.0)) for rank, index in enumerate(present)}
    rgb = np.full(image.shape + (4,), 1.0)
    rgb[..., 3] = 0.0
    for index, color in colors.items():
        rgb[image == index] = color
    ax.imshow(rgb, interpolation="nearest")
    ax.legend(handles=[Patch(color=colors[i], label=str(labels[i])[:34]) for i in present[:20]],
              fontsize=6.5, loc="upper left", bbox_to_anchor=(1.0, 1.0), frameon=False)
    ax.set_title(title, fontsize=9)
    ax.set_axis_off()
    return ax


def mz_class_profiles(ax: Axes, histogram: pd.DataFrame, classes: Sequence[str], names: Mapping[str, str], *,
                      population: str, bin_width: float = 5.0, theme: VisualizationTheme | str | None = None) -> Axes:
    """Draw per-class m/z histograms as offset, normalized profiles (ridge lines).

    :param ax: Target axes.
    :type ax: matplotlib.axes.Axes
    :param histogram: Rows ``class_id``, ``population``, ``mz_left``, ``count``.
    :type histogram: pandas.DataFrame
    :param classes: Classes drawn bottom-to-top.
    :type classes: collections.abc.Sequence[str]
    :param names: Display names per class.
    :type names: collections.abc.Mapping[str, str]
    :param population: ``universe`` or ``observed``.
    :type population: str
    :param bin_width: Histogram bin width used by the precompute (Da).
    :type bin_width: float
    :param theme: Visualization theme.
    :type theme: VisualizationTheme | str | None
    :return: The axes.
    :rtype: matplotlib.axes.Axes
    """
    resolved = resolve_theme(theme)
    subset = histogram[histogram.population == population]
    if subset.empty:
        return ax
    ## REMARK: Only non-empty bins are stored; the full grid is restored with zeros so
    ## that step profiles do not bridge empty m/z ranges.
    grid = np.arange(subset.mz_left.min(), subset.mz_left.max() + bin_width / 2, bin_width)  # (G,)
    for offset, class_id in enumerate(classes):
        rows = subset[subset.class_id == class_id]
        if rows.empty:
            continue
        counts = rows.groupby("mz_left")["count"].sum().reindex(np.round(grid, 6), fill_value=0)
        counts.index = grid
        height = counts.to_numpy(float) / counts.max() * 0.85
        color = resolved.class_palette[offset % len(resolved.class_palette)]
        ax.fill_between(grid + bin_width / 2, offset, offset + height, step="mid", color=color, alpha=0.45, linewidth=0)
        ax.plot(grid + bin_width / 2, offset + height, drawstyle="steps-mid", color=color, linewidth=0.8)
    ax.set_yticks(range(len(classes)), [str(names.get(c, c))[:40] for c in classes], fontsize=7)
    ax.set_xlabel("m/z")
    return ax
