"""Semantics of the shared predictive-campaign figures."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest
from matplotlib.collections import PathCollection

from msi_autoencoder_wrapper.visualization import predictive as plots
from msi_autoencoder_wrapper.visualization.theme import resolve_theme


def _units(labels=("balanced_bce (ClassBalancedMultiLabelBCELoss)", "vpu (VariationalPULoss)")):
    return pd.DataFrame([
        dict(label=label, model_id=f"{label}-{seed}", metric=metric, value=value + seed / 100)
        for label, value in zip(labels, (.6, .7))
        for metric in ("average_precision", "roc_auc")
        for seed in range(3)
    ])


def test_related_losses_keep_related_shades_and_unknown_labels_stay_distinct():
    theme = resolve_theme(None)
    global_shade = plots._color("bce_global_positive_penalty (PositiveWeightedMultiLabelBCELoss)", theme)
    per_class_shade = plots._color("bce_per_class_positive_penalty (PositiveWeightedMultiLabelBCELoss)", theme)
    other_family = plots._color("vpu (VariationalPULoss)", theme)
    # Same family, different weighting: a lighter shade of one base colour.
    assert global_shade != per_class_shade
    assert np.allclose(np.argsort(global_shade), np.argsort(per_class_shade))
    assert not np.allclose(global_shade, other_family)
    # Labels outside the known families must not all collapse onto one colour.
    assert plots._color("stratum A", theme) != plots._color("stratum B", theme)


def test_group_order_is_respected_missing_groups_dropped_and_ticks_shortened():
    frame = _units()
    figure = plots.distributions(frame.query("metric == 'average_precision'"),
                                 order=["vpu (VariationalPULoss)", "absent",
                                        "balanced_bce (ClassBalancedMultiLabelBCELoss)"])
    ticks = [text.get_text() for text in figure.axes[0].get_xticklabels()]
    # The loss class is carried by colour, so the axis shows the condition name only.
    assert ticks == ["vpu", "balanced_bce"]
    # Shortening is display-only: the colour must still come from the full label.
    theme = resolve_theme(None)
    assert plots._color("vpu (VariationalPULoss)", theme) != plots._color("vpu", theme)


def test_metric_panels_draw_one_panel_per_metric_and_the_reference_level():
    figure = plots.metric_panels(_units(), metrics=["average_precision", "roc_auc"],
                                 references={"average_precision": .65}, columns=2)
    visible = [axis for axis in figure.axes if axis.get_visible()]
    assert [axis.get_title() for axis in visible] == ["average_precision", "roc_auc"]
    assert [line.get_ydata()[0] for line in visible[0].lines if line.get_linestyle() == "--"] == [.65]
    assert not [line for line in visible[1].lines if line.get_linestyle() == "--"]


def test_histogram_overlay_uses_stored_counts_as_a_density_over_the_bin_width():
    frame = pd.DataFrame({"label": "vpu (VariationalPULoss)", "model_id": "run",
                          "left_edge": [0., .5], "right_edge": [.5, 1.], "count": [3, 1]})
    figure = plots.histogram_overlay(frame, x_label="score")
    step = figure.axes[0].patches[0]
    # Four stored entries over bins of width 0.5 give densities 1.5 and 0.5.
    np.testing.assert_allclose(step.get_data().values, [1.5, .5])
    counted = plots.histogram_overlay(frame, x_label="score", normalize=False)
    np.testing.assert_allclose(counted.axes[0].patches[0].get_data().values, [3., 1.])


def test_paired_intervals_sort_by_effect_and_mark_the_zero_reference():
    frame = pd.DataFrame({"label": ["a", "b"], "mean_difference": [.05, -.02],
                          "ci_low": [.01, -.05], "ci_high": [.09, .01], "pairs": [5, 5]})
    figure = plots.paired_intervals(frame)
    axis = figure.axes[0]
    assert [text.get_text() for text in axis.get_yticklabels()] == ["b (n=5)", "a (n=5)"]
    assert [line.get_xdata()[0] for line in axis.lines if line.get_linestyle() == "--"] == [0.]


def test_spectrum_comparison_draws_the_shared_input_once_and_one_line_per_run():
    rows = []
    for label in ("vpu (VariationalPULoss)", "pnu_ce (ThreeStateCrossEntropyLoss)"):
        for mz, value in zip((200., 201., 202.), (.5, .3, .2)):
            rows.append(dict(row_position=7, label=label, model_id=f"{label}-0", mz=mz,
                             input=value, reconstruction=value / 2, selected_as_worst=False))
    figure = plots.spectrum_comparison(pd.DataFrame(rows), row_position=7)
    intensity, residual = figure.axes[0], figure.axes[1]
    assert len(intensity.lines) == 3  # One shared input plus one reconstruction per condition.
    input_line = next(line for line in intensity.lines if line.get_label() == "input")
    # The shared reference must sit above the reconstruction band, not under it.
    assert input_line.get_zorder() > max(line.get_zorder() for line in intensity.lines
                                        if line is not input_line)
    assert sorted(line.get_label() for line in intensity.lines) == ["input", "pnu_ce", "vpu"]
    # The residual panel holds one line per run plus the zero reference.
    assert len(residual.lines) == 3
    with pytest.raises(ValueError, match="not present"):
        plots.spectrum_comparison(pd.DataFrame(rows), row_position=99)


def test_class_comparison_keeps_both_values_and_orders_rows_by_the_difference():
    frame = pd.DataFrame({"class_name": [f"ion{index}" for index in range(4)],
                          "value_left": [.9, .5, .4, .1], "value_right": [.1, .5, .6, .9]})
    figure = plots.class_comparison(frame, left_label="a", right_label="b", count=2)
    axis = figure.axes[0]
    assert [text.get_text() for text in axis.get_yticklabels()] == ["ion3", "ion2", "ion1", "ion0"]
    scatters = [child for child in axis.collections if isinstance(child, PathCollection)]
    assert len(scatters) == 8  # Two markers for each of the four retained ions.


def test_facets_hide_unused_panels_and_forward_arguments():
    frame = _units().assign(panel=lambda part: part.metric)
    figure = plots.facets(frame, facet="panel", plot=plots.distributions, columns=3, value="value")
    assert [axis.get_visible() for axis in figure.axes] == [True, True, False]
