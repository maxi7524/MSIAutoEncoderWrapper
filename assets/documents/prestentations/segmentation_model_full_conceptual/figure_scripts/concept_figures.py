#!/usr/bin/env python3
"""Concept figures of the presentation that do not depend on a trained model.

Figures (written to ``../figures``):

- ``msi_pixel_spectrum``: one kidney section (TIC image) and the spectrum of one pixel;
- ``msi_binning``: centroid peaks of that pixel and their accumulation into 0.55 m/z bins;
- ``atlas_globe_chart``: orthographic view of the globe and a flat chart of Europe;
- ``europe_capitals_map`` / ``europe_capitals_axis``: PubMed mass-spectrometry
  publication shares of European capitals on the map and on one real axis;
- ``masserstein_reconstructed_channels``: best/median/worst reconstructed channel images of
  the real model, read from the tables of ``part_1_05_reconstructed_ion_images``.

Inputs: the source kidney image in the kidney workspace, the Natural Earth outlines
and the PubMed counts in ``../figure_data`` (see ``pubmed_mass_spectrometry_capitals.csv``
for the exact queries and retrieval date).

Usage::

    .venv/bin/python assets/documents/prestentations/segmentation_model_full_conceptual/figure_scripts/concept_figures.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import ConnectionPatch, Polygon, Rectangle
from pyimzml.ImzMLParser import ImzMLParser

from msi_autoencoder_wrapper.utils.logger import get_custom_logger
from msi_autoencoder_wrapper.visualization.representative_toy import (
    ACCENT,
    INK,
    MUTED,
    presentation_style,
    save_figure,
)

logger = get_custom_logger(__name__)

PRESENTATION_DIRECTORY = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = Path(__file__).resolve().parents[5]
FIGURE_DIRECTORY = PRESENTATION_DIRECTORY / "figures"
DATA_DIRECTORY = PRESENTATION_DIRECTORY / "figure_data"
SOURCE_IMAGE = REPOSITORY_ROOT / "data/kidney_workspace/datasets/2024-02-20_01h46m58s/2024-02-20_01h46m58s.imzML"
HIGHLIGHT_PIXEL = (30, 60)  # zero-based (x, y) inside the cortex
BIN_STEP = 0.55
BINNING_WINDOW = (780.0, 792.0)
EUROPE_EXTENT = (-11.0, 32.0, 34.0, 62.0)  # lon_min, lon_max, lat_min, lat_max
RECONSTRUCTION_RESULTS = (REPOSITORY_ROOT / "assets/experiments/autoencoder_architecture/notebooks/segmentation_model"
                          / "20_09_metaspace_pretrain/part_1_real_baseline/axis_100_3000"
                          / "part_1_05_reconstructed_ion_images_results")
RECONSTRUCTION_IMAGE = "2024-02-20_01h46m58s"
OUTLIER_CITY = "Valletta/Msida"
LAND = "#E4E4E7"
BORDER = "#A1A1AA"


# --------------------------------------------------
# Section: MSI data
# --------------------------------------------------

def _load_source_image() -> tuple[np.ndarray, ImzMLParser, np.ndarray]:
    """Return the TIC image, the parser and zero-based pixel coordinates."""
    parser = ImzMLParser(str(SOURCE_IMAGE))
    coordinates = np.asarray(parser.coordinates)[:, :2] - 1  # (N, 2)
    tic = np.array([parser.getspectrum(index)[1].sum() for index in range(len(coordinates))])  # (N,)
    image = np.full((coordinates[:, 1].max() + 1, coordinates[:, 0].max() + 1), np.nan)
    image[coordinates[:, 1], coordinates[:, 0]] = tic
    return image, parser, coordinates


def msi_pixel_spectrum(image: np.ndarray, parser: ImzMLParser, coordinates: np.ndarray) -> None:
    """Image with one highlighted pixel connected to its spectrum."""
    index = int(np.flatnonzero((coordinates[:, 0] == HIGHLIGHT_PIXEL[0]) & (coordinates[:, 1] == HIGHLIGHT_PIXEL[1]))[0])
    mzs, intensities = parser.getspectrum(index)
    with presentation_style():
        figure, (image_ax, spectrum_ax) = plt.subplots(1, 2, figsize=(14, 5.6), gridspec_kw={"width_ratios": [1, 2.4]})
        image_ax.imshow(image, cmap="magma", interpolation="nearest")
        image_ax.add_patch(Rectangle((HIGHLIGHT_PIXEL[0] - 2.5, HIGHLIGHT_PIXEL[1] - 2.5), 5, 5, fill=False,
                                     edgecolor="#22D3EE", lw=2.5))
        image_ax.set_title("image  $I$")
        image_ax.set_axis_off()
        spectrum_ax.vlines(mzs, 0, intensities / intensities.max(), color=INK, lw=1.0)
        spectrum_ax.set_xlabel("m/z")
        spectrum_ax.set_ylabel("intensity (relative)")
        spectrum_ax.set_title(f"spectrum of pixel $(x, y) = ({HIGHLIGHT_PIXEL[0]}, {HIGHLIGHT_PIXEL[1]})$")
        spectrum_ax.set_ylim(0, 1.05)
        figure.add_artist(ConnectionPatch(
            xyA=(HIGHLIGHT_PIXEL[0] + 2.5, HIGHLIGHT_PIXEL[1]), coordsA="data", axesA=image_ax,
            xyB=(0.0, 0.8), coordsB="axes fraction", axesB=spectrum_ax,
            arrowstyle="-|>", color="#0891B2", lw=2.0, mutation_scale=22))
        figure.tight_layout()
        save_figure(figure, FIGURE_DIRECTORY, "msi_pixel_spectrum")
        plt.close(figure)


def msi_binning(parser: ImzMLParser, coordinates: np.ndarray) -> None:
    """Centroid peaks inside a window and their sums over equidistant bins."""
    index = int(np.flatnonzero((coordinates[:, 0] == HIGHLIGHT_PIXEL[0]) & (coordinates[:, 1] == HIGHLIGHT_PIXEL[1]))[0])
    mzs, intensities = parser.getspectrum(index)
    edges = np.arange(100.0, 3000.0 + BIN_STEP, BIN_STEP, dtype=np.float32).astype(np.float64)  # (M + 1,)
    window = (mzs >= BINNING_WINDOW[0]) & (mzs <= BINNING_WINDOW[1])
    visible_edges = edges[(edges >= BINNING_WINDOW[0]) & (edges <= BINNING_WINDOW[1])]
    first_bin = int(np.searchsorted(edges, BINNING_WINDOW[0], side="right") - 1)
    binned, _ = np.histogram(mzs[window], bins=edges[first_bin:first_bin + len(visible_edges) + 2],
                             weights=intensities[window])
    centers = (edges[first_bin:first_bin + len(binned)] + edges[first_bin + 1:first_bin + len(binned) + 1]) / 2
    scale = intensities[window].max()
    with presentation_style():
        figure, (raw_ax, bin_ax) = plt.subplots(2, 1, figsize=(14, 6.4), sharex=True)
        raw_ax.vlines(mzs[window], 0, intensities[window] / scale, color=INK, lw=1.6)
        raw_ax.set_ylabel("intensity")
        raw_ax.set_title("measured peaks  $(m/z, \\ \\mathrm{intensity}) \\in \\mathbb{R}_+ \\times \\mathbb{R}_+$")
        for ax in (raw_ax, bin_ax):
            for edge in visible_edges:
                ax.axvline(edge, color=MUTED, lw=0.6, ls=":")
        bin_ax.bar(centers, binned / scale, width=BIN_STEP * 0.92, color="#0891B2", alpha=0.85)
        bin_ax.set_ylabel("bin sum")
        bin_ax.set_xlabel("m/z")
        bin_ax.set_title(f"binned:  $B_i = [\\,100 + {BIN_STEP}\\,(i-1),\\ 100 + {BIN_STEP}\\,i\\,)$,  "
                         f"$i = 1, \\dots, {len(edges) - 1}$", pad=24)
        bin_ax.set_xlim(*BINNING_WINDOW)
        labelled = np.arange(0, len(centers), 3)
        index_axis = bin_ax.secondary_xaxis("top")
        index_axis.set_xticks(centers[labelled],
                              [f"$B_{{{first_bin + 1 + position}}}$" for position in labelled])
        index_axis.tick_params(labelsize=11, colors=MUTED, length=0)
        index_axis.spines["top"].set_visible(False)
        figure.tight_layout()
        save_figure(figure, FIGURE_DIRECTORY, "msi_binning")
        plt.close(figure)
    logger.info("Binning axis 100-3000 m/z at %s: %s bins.", BIN_STEP, len(edges) - 1)


# --------------------------------------------------
# Section: maps
# --------------------------------------------------

def _country_rings(path: Path) -> list[np.ndarray]:
    """Exterior rings (lon, lat) of every country polygon."""
    features = json.loads(path.read_text())["features"]
    rings = []
    for feature in features:
        geometry = feature["geometry"]
        polygons = geometry["coordinates"] if geometry["type"] == "MultiPolygon" else [geometry["coordinates"]]
        rings.extend(np.asarray(polygon[0]) for polygon in polygons)
    return rings


def _orthographic(lon: np.ndarray, lat: np.ndarray, lon0: float, lat0: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Orthographic projection; returns x, y and a visibility mask."""
    lon, lat, lon0, lat0 = map(np.radians, (lon, lat, lon0, lat0))
    x = np.cos(lat) * np.sin(lon - lon0)
    y = np.cos(lat0) * np.sin(lat) - np.sin(lat0) * np.cos(lat) * np.cos(lon - lon0)
    visible = np.sin(lat0) * np.sin(lat) + np.cos(lat0) * np.cos(lat) * np.cos(lon - lon0) >= 0
    return x, y, visible


def atlas_globe_chart(rings: list[np.ndarray], chart_rings: list[np.ndarray]) -> None:
    """Globe (the manifold) and a flat chart of one region (a map of the atlas)."""
    lon0, lat0 = 15.0, 45.0
    lon_min, lon_max, lat_min, lat_max = EUROPE_EXTENT
    with presentation_style():
        figure, (globe_ax, chart_ax) = plt.subplots(1, 2, figsize=(14, 6.2), gridspec_kw={"width_ratios": [1, 1.25]})
        globe_ax.add_patch(plt.Circle((0, 0), 1, color="#EFF6FF", ec=MUTED, lw=1.2))
        for ring in rings:
            x, y, visible = _orthographic(ring[:, 0], ring[:, 1], lon0, lat0)
            if visible.mean() > 0.98:
                globe_ax.add_patch(Polygon(np.c_[x, y], closed=True, fc=LAND, ec=BORDER, lw=0.4))
        ## Graticule and the charted region
        for meridian in range(-180, 180, 30):
            lat = np.linspace(-90, 90, 200)
            x, y, visible = _orthographic(np.full_like(lat, meridian), lat, lon0, lat0)
            globe_ax.plot(np.where(visible, x, np.nan), np.where(visible, y, np.nan), color="#CBD5E1", lw=0.5)
        for parallel in range(-60, 90, 30):
            lon = np.linspace(-180, 180, 400)
            x, y, visible = _orthographic(lon, np.full_like(lon, parallel), lon0, lat0)
            globe_ax.plot(np.where(visible, x, np.nan), np.where(visible, y, np.nan), color="#CBD5E1", lw=0.5)
        box_lon = np.r_[np.linspace(lon_min, lon_max, 50), np.full(50, lon_max), np.linspace(lon_max, lon_min, 50), np.full(50, lon_min)]
        box_lat = np.r_[np.full(50, lat_min), np.linspace(lat_min, lat_max, 50), np.full(50, lat_max), np.linspace(lat_max, lat_min, 50)]
        x, y, _ = _orthographic(box_lon, box_lat, lon0, lat0)
        globe_ax.plot(x, y, color=ACCENT, lw=2.0)
        globe_ax.text(0, -1.12, "manifold  $S^2$", ha="center", va="top")
        globe_ax.set_xlim(-1.05, 1.05)
        globe_ax.set_ylim(-1.25, 1.05)
        globe_ax.set_aspect("equal")
        globe_ax.set_axis_off()
        _draw_europe(chart_ax, chart_rings)
        for spine in chart_ax.spines.values():
            spine.set_visible(True)
            spine.set_color(ACCENT)
            spine.set_linewidth(2.0)
        chart_ax.set_xticks([])
        chart_ax.set_yticks([])
        chart_ax.set_title("chart  $\\varphi: U \\subset S^2 \\to \\mathbb{R}^2$", pad=10)
        figure.add_artist(ConnectionPatch(xyA=(0.62, 0.62), coordsA="axes fraction", axesA=globe_ax,
                                          xyB=(-0.02, 0.6), coordsB="axes fraction", axesB=chart_ax,
                                          arrowstyle="-|>", color=ACCENT, lw=2.0, mutation_scale=22,
                                          connectionstyle="arc3,rad=-0.2"))
        figure.tight_layout()
        save_figure(figure, FIGURE_DIRECTORY, "atlas_globe_chart")
        plt.close(figure)


def _draw_europe(ax: plt.Axes, rings: list[np.ndarray]) -> None:
    """Equirectangular Europe with latitude-corrected aspect."""
    lon_min, lon_max, lat_min, lat_max = EUROPE_EXTENT
    for ring in rings:
        if ring[:, 0].max() < lon_min - 5 or ring[:, 0].min() > lon_max + 5 or ring[:, 1].max() < lat_min - 5 or ring[:, 1].min() > lat_max + 5:
            continue
        ax.add_patch(Polygon(ring, closed=True, fc=LAND, ec=BORDER, lw=0.5))
    ax.set_xlim(lon_min, lon_max)
    ax.set_ylim(lat_min, lat_max)
    ax.set_aspect(1.0 / np.cos(np.radians((lat_min + lat_max) / 2)))


def europe_capitals(rings: list[np.ndarray]) -> None:
    """Capitals on the map, then a deliberately contrived one-dimensional score.

    Shown capitals: every capital with at least as many publications as Warsaw, the next
    smaller one (Dublin) and Valletta/Msida (conference venue). With ``x = log10 n`` the
    score is the quartic ``p(x) = (x - r1)(x - r2)(x - r3)(x - r4)`` whose roots lie between
    Valletta and Dublin, Dublin and Warsaw, Warsaw and the next larger capital, and the
    fourth and third largest capitals. Its sign places Valletta, Warsaw and the three
    largest centres above the threshold ``p = 0`` and all other capitals below it. The axis
    shows ``sign(p) |p|^(1/4)`` so that all values are readable.
    """
    counts = pd.read_csv(DATA_DIRECTORY / "pubmed_mass_spectrometry_capitals.csv")
    reference = int(counts.loc[counts.city == "Warsaw", "pubmed_count"].iloc[0])
    below = counts[counts.pubmed_count < reference].nlargest(1, "pubmed_count").city
    counts = counts[(counts.pubmed_count >= reference) | counts.city.isin(below) | (counts.city == OUTLIER_CITY)].copy()
    counts["share"] = counts.pubmed_count / counts.pubmed_count.sum()
    ## Quartic score with roots between the groups that must be separated
    log_counts = np.sort(np.log10(counts.pubmed_count.to_numpy()))
    warsaw = np.log10(reference)
    position = int(np.searchsorted(log_counts, warsaw))
    roots = np.array([
        2.0,
        (log_counts[position - 1] + warsaw) / 2,
        (warsaw + log_counts[position + 1]) / 2,
        (log_counts[-4] + log_counts[-3]) / 2,
    ])
    polynomial = np.prod(np.log10(counts.pubmed_count.to_numpy())[:, None] - roots[None, :], axis=1)
    counts["score"] = np.sign(polynomial) * np.abs(polynomial) ** 0.25
    counts["positive"] = polynomial > 0
    threshold = 0.0
    inner = counts.loc[counts.city != OUTLIER_CITY, "score"].to_numpy()
    emphasised = ("Warsaw", OUTLIER_CITY, *below)
    with presentation_style():
        figure, ax = plt.subplots(figsize=(10, 8.4))
        _draw_europe(ax, rings)
        ax.scatter(counts.longitude, counts.latitude, s=40 + 900 * counts.share / counts.share.max(),
                   c="#0891B2", alpha=0.75, edgecolors="white", linewidths=0.8, zorder=3)
        for row in counts.itertuples():
            ax.text(row.longitude + 0.5, row.latitude + 0.35, row.city.split("/")[0], fontsize=11,
                    color=INK if row.city in emphasised else MUTED, zorder=4,
                    fontweight="bold" if row.city in emphasised else "normal")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title("capitals: position $(\\lambda, \\phi) \\in \\mathbb{R}^2$, marker area = publications")
        figure.tight_layout()
        save_figure(figure, FIGURE_DIRECTORY, "europe_capitals_map")
        plt.close(figure)

        ## Broken axis: the capitals on [0, max inner score] and the outlier far to the right
        figure, (left, right) = plt.subplots(1, 2, figsize=(14, 4.2), sharey=True,
                                             gridspec_kw={"width_ratios": [5, 1], "wspace": 0.04})
        jitter = np.random.default_rng(0).uniform(-0.22, 0.22, len(counts))
        sizes = 30 + 700 * counts.pubmed_count / counts.pubmed_count.max()
        colors = np.where(counts.positive, "#0072B2", "#E69F00")
        for ax in (left, right):
            ax.scatter(counts.score, jitter, s=sizes, c=colors, edgecolors="white", linewidths=0.8, zorder=3)
            ax.set_ylim(-0.8, 0.9)
            ax.set_yticks([])
        left.axvline(threshold, color=MUTED, ls="--", lw=1.4)
        left.text(threshold, 0.8, "  threshold  $p = 0$", color=MUTED, fontsize=12, va="top")
        span = inner.max() - inner.min()
        left.set_xlim(inner.min() - 0.08 * span, inner.max() + 0.12 * span)
        outlier = float(counts.loc[counts.city == OUTLIER_CITY, "score"].iloc[0])
        right.set_xlim(outlier - 0.1 * span, outlier + 0.1 * span)
        left.spines["left"].set_visible(False)
        right.spines["left"].set_visible(False)
        for x, ax in ((1.0, left), (0.0, right)):
            ax.plot([x - 0.008, x + 0.008], [-0.03, 0.03], transform=ax.transAxes, color=INK, lw=1.2, clip_on=False)
        for row, offset in zip(counts.itertuples(), jitter):
            if row.city in (*emphasised, "London", "Paris", "Berlin", "Madrid"):
                ax = right if row.city == OUTLIER_CITY else left
                label = "Valletta / Msida" if row.city == OUTLIER_CITY else row.city
                ax.annotate(label, (row.score, offset), xytext=(0, 16), textcoords="offset points",
                            ha="center", fontsize=12, color=INK,
                            fontweight="bold" if row.city in emphasised else "normal")
        figure.supxlabel("importance  $\\mathrm{sign}(p)\\,|p|^{1/4}$", fontsize=15, y=0.02)
        figure.suptitle("$p(x) = (x - r_1)(x - r_2)(x - r_3)(x - r_4)$,  $x = \\log_{10} n$,  $n$ = publications", fontsize=16)
        handles = [plt.Line2D([], [], marker="o", ls="", color="#0072B2", ms=10, label="above threshold"),
                   plt.Line2D([], [], marker="o", ls="", color="#E69F00", ms=10, label="below threshold")]
        left.legend(handles=handles, frameon=False, loc="lower right", fontsize=12)
        figure.subplots_adjust(bottom=0.2, top=0.86)
        save_figure(figure, FIGURE_DIRECTORY, "europe_capitals_axis")
        plt.close(figure)
    logger.info("Capital score roots %s; positive: %s.", roots.round(3).tolist(),
                counts.loc[counts.positive, "city"].tolist())


# --------------------------------------------------
# Section: model schematic
# --------------------------------------------------

def autoencoder_schematic() -> None:
    """Encoder, sphere-valued latent, decoder, head and projector with their dimensions."""
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    def box(ax, x, y, width, height, text, colour, *, text_colour=INK, size=15):
        ax.add_patch(FancyBboxPatch((x - width / 2, y - height / 2), width, height,
                                    boxstyle="round,pad=0.02,rounding_size=0.08", fc=colour, ec="none"))
        ax.text(x, y, text, ha="center", va="center", fontsize=size, color=text_colour)

    def arrow(ax, start, end, text=None, *, offset=(0.0, 0.18)):
        ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=18, lw=1.8, color=MUTED))
        if text:
            ax.text((start[0] + end[0]) / 2 + offset[0], (start[1] + end[1]) / 2 + offset[1], text,
                    ha="center", va="bottom", fontsize=13, color=MUTED)

    with presentation_style():
        figure, ax = plt.subplots(figsize=(14, 5.2))
        ## Main reconstruction path
        box(ax, 1.0, 2.0, 1.7, 2.4, "spectrum\n$x \\in \\Delta^{M-1}$\n$M = 5273$", "#E0F2FE")
        box(ax, 3.55, 2.0, 1.6, 1.9, "encoder\nConv1d ×3\nLinear\n$a \\in \\mathbb{R}^{L}$", "#F4F4F5")
        box(ax, 6.1, 2.0, 2.05, 1.5, "LayerNorm\n$u = (a - \\mu_a)/\\sigma_a$", "#FEF3C7")
        box(ax, 8.75, 2.0, 1.9, 1.9, "$u \\in S^{L-2}$\n$L = 10 \\Rightarrow S^{8}$\n$z = \\gamma \\odot u + \\beta$",
            "#FDE68A")
        box(ax, 11.35, 2.0, 1.6, 1.9, "decoder\nLinear\nConvT ×3", "#F4F4F5")
        box(ax, 13.9, 2.0, 1.7, 2.4, "reconstruction\n$\\hat x \\in \\Delta^{M-1}$", "#E0F2FE")
        for left, right in ((1.85, 2.75), (4.35, 5.02), (7.18, 7.8), (9.7, 10.55), (12.15, 13.05)):
            arrow(ax, (left, 2.0), (right, 2.0))
        ## Auxiliary outputs of the latent
        box(ax, 7.3, 0.1, 2.3, 0.8, "head  $h(z) \\in \\mathbb{R}^{C}$", "#DCFCE7")
        box(ax, 10.3, 0.1, 2.5, 0.8, "projector  $g(z) \\in \\mathbb{R}^{64}$", "#FCE7F3")
        arrow(ax, (8.5, 1.05), (7.7, 0.5))
        arrow(ax, (9.0, 1.05), (9.9, 0.5))
        ## Loss annotations
        ax.text(7.45, 3.6, "Masserstein  $W_1(x, \\hat x)$", ha="center", fontsize=14, color=ACCENT)
        ax.annotate("", xy=(13.9, 3.3), xytext=(1.0, 3.3),
                    arrowprops={"arrowstyle": "<->", "color": ACCENT, "lw": 1.4})
        ax.text(7.3, -0.6, "labels (PU)", ha="center", fontsize=12, color=MUTED)
        ax.text(10.3, -0.6, "contrastive", ha="center", fontsize=12, color=MUTED)
        ax.text(4.85, 0.35, "contractive:  $\\partial u / \\partial x$", ha="center", fontsize=12, color=MUTED)
        ax.set_xlim(0, 14.9)
        ax.set_ylim(-0.9, 3.8)
        ax.set_axis_off()
        save_figure(figure, FIGURE_DIRECTORY, "autoencoder_schematic")
        plt.close(figure)


# --------------------------------------------------
# Section: real-model reconstruction
# --------------------------------------------------

def reconstructed_channels() -> None:
    """Input and reconstructed channel images of one held-out image (real model).

    Reads the tables of ``part_1_05_reconstructed_ion_images`` (axis 100-3000,
    representative repetition). One row per selected channel: best, median and
    worst Pearson agreement; each image has its own 1st-99th percentile range.
    """
    selection = pd.read_csv(RECONSTRUCTION_RESULTS / "channel_selection.csv")
    pixels = pd.read_csv(RECONSTRUCTION_RESULTS / "channel_pixels.csv")
    selection = selection[(selection.dataset_id == RECONSTRUCTION_IMAGE) & (selection["rank"] == 0)]
    selection = selection.set_index("kind").loc[["best", "median", "worst"]].reset_index()
    with presentation_style():
        figure, axes = plt.subplots(2, len(selection), figsize=(13, 8.6))
        for column, row in enumerate(selection.itertuples()):
            channel = pixels[(pixels.dataset_id == RECONSTRUCTION_IMAGE) & (pixels.kind == row.kind)
                             & (pixels["rank"] == 0)]
            for line, source in enumerate(("input", "output")):
                values = channel[source].to_numpy()
                image = np.full((channel.y.max() + 1, channel.x.max() + 1), np.nan)
                image[channel.y, channel.x] = values
                low, high = np.percentile(values, [1, 99])
                axes[line, column].imshow(image, cmap="viridis", vmin=low, vmax=high, interpolation="nearest")
                axes[line, column].set_xticks([])
                axes[line, column].set_yticks([])
                for spine in axes[line, column].spines.values():
                    spine.set_visible(False)
            axes[0, column].set_title(f"{row.kind}: m/z {row.mz:.2f}\n$r = {row.pearson:.2f}$")
        axes[0, 0].set_ylabel("input $x$")
        axes[1, 0].set_ylabel("reconstruction $\\hat x$")
        figure.tight_layout()
        save_figure(figure, FIGURE_DIRECTORY, "masserstein_reconstructed_channels")
        plt.close(figure)


def main() -> int:
    """Render every concept figure."""
    reconstructed_channels()
    autoencoder_schematic()
    image, parser, coordinates = _load_source_image()
    msi_pixel_spectrum(image, parser, coordinates)
    msi_binning(parser, coordinates)
    rings = _country_rings(DATA_DIRECTORY / "ne_50m_admin_0_countries_europe.geojson")
    atlas_globe_chart(_country_rings(DATA_DIRECTORY / "ne_110m_admin_0_countries.geojson"), rings)
    europe_capitals(rings)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
