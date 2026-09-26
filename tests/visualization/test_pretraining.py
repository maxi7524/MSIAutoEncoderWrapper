"""Structural checks of the pretraining-campaign figures on small synthetic tables."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from msi_autoencoder_wrapper.visualization import pretraining as figures  # noqa: E402


@pytest.fixture
def pixels() -> pd.DataFrame:
    """Two 3 x 2 images with one value column and one segment column."""
    rows = []
    for dataset in ("a", "b"):
        for x in range(1, 4):
            for y in range(1, 3):
                rows.append({"dataset_id": dataset, "x": x, "y": y, "row": len(rows), "value": float(x * y),
                             "segment": (x + y) % 3, "class_name": "C1|-H", "mz": 300.0})
    return pd.DataFrame(rows)


def test_image_panels_draw_one_image_per_row_and_column(pixels: pd.DataFrame) -> None:
    figure = figures.image_panels(pixels, value_columns=["value", "value"], column_titles=["v1", "v2"],
                                  datasets=["a", "b"])
    images = [ax.images[0] for ax in figure.axes if ax.images]
    assert len(images) == 4
    assert images[0].get_array().shape == (2, 3)
    plt.close(figure)


def test_label_panels_and_exploded_view_use_all_segments(pixels: pd.DataFrame) -> None:
    figure = figures.label_panels(pixels, label_columns=["segment"], column_titles=["seg"], datasets=["a", "b"], k=3)
    assert len([ax for ax in figure.axes if ax.images]) == 2
    plt.close(figure)
    layers = pd.concat([pixels.assign(metaspace=pixels.value), pixels.assign(class_name="C2|-H", mz=250.0,
                                                                              metaspace=pixels.value)])
    figure = figures.exploded_view(layers, dataset="a", classes=["C1|-H", "C2|-H"], value="metaspace", k=3)
    labels = [tick.get_text() for tick in figure.axes[0].get_zticklabels()]
    ## Layers are ordered by m/z above the segment plane
    assert labels[:3] == ["segments", "C2|-H (250.0)", "C1|-H (300.0)"]
    plt.close(figure)


def test_distribution_and_repetition_figures_accept_long_tables() -> None:
    generator = np.random.default_rng(0)
    frame = pd.DataFrame({"population": np.repeat(["train", "test"], 50), "metric_a": generator.random(100)})
    means = pd.DataFrame({"population": ["train", "test"], "metric": "metric_a", "mean": [0.5, 0.4]})
    figure = figures.population_distributions(frame, means, ["metric_a"], ["train", "test"], columns=1)
    assert len(figure.axes[0].collections) > 0
    plt.close(figure)
    points = pd.DataFrame({"group": ["g"] * 4, "hue": ["x", "x", "y", "y"], "value": [1.0, 2.0, 3.0, 4.0]})
    figure = figures.repetition_points(points, value="value", group="group", hue="hue", order=["g"],
                                       hue_order=["x", "y"], colors={"x": "red", "y": "blue"}, ylabel="v")
    ## One scatter (PathCollection) per group/hue; mean bars are separate line collections
    from matplotlib.collections import PathCollection

    assert sum(isinstance(item, PathCollection) for item in figure.axes[0].collections) == 2
    plt.close(figure)


def test_sparse_positive_images_keep_a_usable_colour_range() -> None:
    ## 1 % of pixels positive: the 99th percentile of all pixels is zero
    values = np.zeros(1000)
    values[:10] = np.linspace(1.0, 10.0, 10)
    assert figures._color_range(values, percentile=99.0, diverging=False, positive=False)[1] > 0
    low, high = figures._color_range(values, percentile=99.0, diverging=False, positive=True)
    assert low == 0.0 and high == pytest.approx(np.percentile(values[:10], 99.0))
    ## A binary target keeps the range [0, 1]
    assert figures._color_range(np.r_[np.zeros(999), 1.0], percentile=99.0, diverging=False,
                                positive=True) == (0.0, 1.0)


def test_row_scale_is_shared_by_unsigned_panels_and_error_is_symmetric(pixels: pd.DataFrame) -> None:
    frame = pixels.assign(output=pixels.value * 2.0, error=pixels.value)
    figure = figures.image_panels(frame, value_columns=["value", "output", "error"], column_titles=["a", "b", "c"],
                                  datasets=["a"], shared_scale="row", diverging_columns=["error"])
    first, second, third = (ax.images[0] for ax in figure.axes if ax.images)
    assert first.get_clim() == second.get_clim()
    low, high = third.get_clim()
    assert low == pytest.approx(-high)
    plt.close(figure)


def test_spectrum_zoom_draws_signal_and_residual_per_range() -> None:
    mz = np.arange(200.0, 400.0, 0.5)
    observed = np.exp(-0.5 * ((mz - 300.0) / 0.4) ** 2)
    cases = pd.DataFrame({"population": "test", "row": 3, "mz": mz, "input": observed, "output": 0.8 * observed,
                          "case_kind": "worst", "masserstein": 1.0})
    figure = figures.spectrum_zoom(cases, keys=[("test", 3)], half_width=15.0,
                                   worst_windows={("test", 3): ("300-400", 300.0, 400.0)})
    ## Axes are created per range as (signal, residual) pairs
    signal, residual = figure.axes[0::2], figure.axes[1::2]
    assert len(signal) == 3 and len(residual) == 3
    assert signal[1].get_xlim() == pytest.approx((285.0, 315.0))
    assert residual[2].get_xlim() == pytest.approx((300.0, 400.0))
    plt.close(figure)


def _summary() -> pd.DataFrame:
    """Two variants x two stages of one metric with intervals."""
    return pd.DataFrame({"variant": ["p", "q", "p", "q"], "stage": ["s1", "s1", "s2", "s2"], "metric": "m",
                         "mean": [0.1, -0.2, 0.3, 0.0], "ci_low": [0.0, -0.4, 0.1, -0.1],
                         "ci_high": [0.2, 0.0, 0.5, 0.1]})


def test_improvement_forest_draws_one_interval_per_variant_and_selects_rows() -> None:
    summary = _summary()
    paired = pd.DataFrame({"variant": ["p"] * 3, "stage": "s1", "metric": "m", "improvement": [0.0, 0.1, 0.2]})
    figure = figures.improvement_forest(summary, panels=[("s1", {"stage": "s1"}), ("both", {"stage": ["s1", "s2"]})],
                                        variants=["p", "q", "missing"], labels={"p": "P"},
                                        colors={"p": "red", "q": "blue"}, paired=paired)
    first, second = figure.axes[:2]
    ## Panel 1: two intervals (one per present variant); panel 2 selects both stages
    assert len(first.collections[0].get_segments()) == 1
    assert sum(len(collection.get_segments()) for collection in first.collections
               if hasattr(collection, "get_segments")) == 2
    assert sum(len(collection.get_segments()) for collection in second.collections
               if hasattr(collection, "get_segments")) == 4
    assert [tick.get_text() for tick in first.get_yticklabels()] == ["P", "q", "missing"]
    plt.close(figure)


def test_stage_paths_and_epoch_curves_follow_the_given_order() -> None:
    figure = figures.stage_paths(_summary(), stages=["s1", "s2"], stage_labels={"s1": "one"}, variants=["p", "q"],
                                 labels={}, colors={"p": "red", "q": "blue"})
    line = figure.axes[0].lines[0]
    assert line.get_ydata().tolist() == [0.1, 0.3]
    assert [tick.get_text() for tick in figure.axes[0].get_xticklabels()] == ["one", "s2"]
    plt.close(figure)
    curves = pd.DataFrame({"variant": ["p", "p", "q"], "epoch": [2, 1, 1], "mean": [2.0, 1.0, 5.0],
                           "ci_low": [1.5, 0.5, 4.0], "ci_high": [2.5, 1.5, 6.0]})
    figure = figures.epoch_curves(curves, levels=["p", "q"], labels={}, colors={}, low="ci_low", high="ci_high",
                                  reference=0.0)
    assert figure.axes[0].lines[0].get_xdata().tolist() == [1, 2]
    plt.close(figure)


def test_reliability_diagram_pools_bins_weighted_by_entries() -> None:
    bins = pd.DataFrame({"variant": "p", "bin": [0, 0], "entries": [1, 3], "mean_probability": [0.1, 0.2],
                         "positive_rate": [0.0, 0.4]})
    figure = figures.reliability_diagram(bins, levels=["p"], labels={}, colors={})
    line = figure.axes[0].lines[0]
    assert line.get_xdata().tolist() == pytest.approx([(0.1 + 3 * 0.2) / 4])
    assert line.get_ydata().tolist() == pytest.approx([1.2 / 4])
    plt.close(figure)


def test_rgb_panels_select_columns_by_any_column(pixels: pd.DataFrame) -> None:
    frame = pd.concat([pixels.assign(shown_as=name, r=0.1, g=0.5, b=0.9) for name in ("baseline", "best")])
    figure = figures.rgb_panels(frame, repetitions=["baseline", "best"], datasets=["a"], column="shown_as")
    assert [ax.get_title() for ax in figure.axes] == ["baseline", "best"]
    plt.close(figure)


def test_forest_rows_can_be_contrasts_and_overlays_draw_every_model() -> None:
    summary = _summary().rename(columns={"variant": "contrast"})
    figure = figures.improvement_forest(summary, panels=[("s1", {"stage": "s1"})], variants=["q", "p"], labels={},
                                        colors={}, row="contrast")
    assert [tick.get_text() for tick in figure.axes[0].get_yticklabels()] == ["q", "p"]
    plt.close(figure)
    mz = np.arange(10.0)
    cases = pd.concat([pd.DataFrame({"population": "test", "row": 1, "mz": mz, "input": np.exp(-(mz - 5) ** 2),
                                     "output": value, "variant": name})
                       for name, value in (("baseline", 0.1), ("best", 0.2))])
    figure = figures.spectrum_overlays(cases, keys=[("test", 1)], levels=["baseline", "best"], labels={}, colors={},
                                       zoom=2.0)
    ax = figure.axes[0]
    ## Input plus one line per model, zoomed around the apex at m/z 5
    assert len(ax.lines) == 3
    assert ax.get_xlim() == (3.0, 7.0)
    plt.close(figure)


@pytest.fixture
def variant_values() -> pd.DataFrame:
    """Three repetitions of a baseline, a better and a worse variant (higher is better)."""
    return pd.DataFrame({"variant": ["baseline"] * 3 + ["good"] * 3 + ["bad"] * 3,
                         "metric": "ap", "value": [0.5, 0.52, 0.48, 0.7, 0.72, 0.68, 0.3, 0.32, 0.28]})


def test_comparison_table_colours_against_baseline_and_frames_the_best(variant_values: pd.DataFrame) -> None:
    styler = figures.comparison_table(variant_values, index="variant", columns="metric", baseline="baseline",
                                      direction={"ap": "higher"}, order=["baseline", "good", "bad"], top=1)
    styles = styler._compute().ctx
    ## Cell positions: row 1 = good, row 2 = bad
    good = dict(styles[(1, 0)])
    bad = dict(styles[(2, 0)])
    assert good["background-color"].startswith("rgba(26, 152, 80") and good["border"] == "2px solid black"
    assert bad["background-color"].startswith("rgba(215, 48, 39") and "border" not in bad
    assert styler.data.loc["good", "ap"].startswith("0.700 +/- ")
    ## Lower-is-better flips the colours
    flipped = figures.comparison_table(variant_values, index="variant", columns="metric", baseline="baseline",
                                       direction={"ap": "lower"}, order=["baseline", "good", "bad"])._compute().ctx
    assert dict(flipped[(1, 0)])["background-color"].startswith("rgba(215, 48, 39")


def test_value_violins_colour_groups_by_side_of_the_baseline(variant_values: pd.DataFrame) -> None:
    figure = figures.value_violins(variant_values, value="value", group="variant", order=["baseline", "good", "bad"],
                                   labels={}, baseline="baseline", direction="higher", ylabel="AP")
    from matplotlib.colors import to_hex
    scatters = [collection for collection in figure.axes[0].collections if collection.get_offsets().shape[0] == 3]
    colours = [to_hex(collection.get_facecolor()[0]) for collection in scatters]
    assert colours[1] == figures.BETTER_COLOR and colours[2] == figures.WORSE_COLOR
    plt.close(figure)


def test_paired_spectra_draws_three_ranges_per_pixel_with_display_gaps() -> None:
    mz = np.r_[np.arange(200.0, 260.0, 0.5), np.arange(400.0, 460.0, 0.5)]
    observed = np.exp(-0.5 * ((mz - 230.0) / 0.4) ** 2)
    spectra = pd.DataFrame({"pixel_kind": "most_improved", "rank": 0, "row": 7, "dataset_id": "a", "x": 1, "y": 2,
                            "masserstein_baseline": 5.0, "masserstein": 3.0, "worst_window": "200-300",
                            "mz": mz, "input": observed, "baseline_output": 0.5 * observed, "output": 0.8 * observed})
    figure = figures.paired_spectra(spectra, variant_label="P1", variant_color="#0072B2")
    assert len(figure.axes) == 6
    ## The gap between 259.5 and 400 m/z is broken, not interpolated
    line = figure.axes[0].lines[0]
    assert np.isnan(line.get_xdata()).sum() == 1
    assert figure.axes[2].get_xlim() == pytest.approx((215.0, 245.0))
    plt.close(figure)


def test_paired_spectra_centres_on_the_given_peak_and_restores_dropped_bins() -> None:
    bins = np.arange(0, 400)
    mz = 100.0 + 0.5 * bins
    observed = np.exp(-0.5 * ((mz - 250.0) / 0.4) ** 2)
    observed[[0, -1]] = 1e-3  # signal at both ends of the axis, zeros in between are dropped
    frame = pd.DataFrame({"population": "train", "pixel_kind": "most_improved", "rank": 0, "row": 3, "bin": bins,
                          "mz": mz, "input": observed, "baseline_output": 0.2 * observed, "output": 0.7 * observed,
                          "masserstein_baseline": 4.0, "masserstein": 2.0, "rare_peak_mz": 180.0})
    kept = frame[frame.input > 1e-6]  # zero bins dropped as in the precompute table
    figure = figures.paired_spectra(kept, variant_label="P4", variant_color="#009E73", center="rare_peak_mz", dense=True)
    signal = figure.axes[0::2]
    assert signal[1].get_xlim() == pytest.approx((165.0, 195.0))
    assert signal[2].get_xlim() == pytest.approx((120.0, 240.0))
    ## Dense restoration: the full-range line spans every bin again
    assert len(signal[0].lines[0].get_xdata()) == bins.size
    plt.close(figure)
