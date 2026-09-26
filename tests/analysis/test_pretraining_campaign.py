"""Deterministic checks of the pretraining-campaign data contracts and plan selection."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from msi_autoencoder_wrapper.analysis.autoencoder.experiments.pretraining_campaign import (
    align_metaspace,
    class_set_labels,
    expand_cell_models,
    read_metaspace_images,
    select_test_extension,
)
from msi_autoencoder_wrapper.analysis.autoencoder.experiments.pretraining_campaign_inference import (
    class_strata,
    representative_ranking,
)
from msi_autoencoder_wrapper.analysis.autoencoder.experiments.pretraining_campaign_reports import _image_channels
from msi_autoencoder_wrapper.analysis.precompute.core.contracts import AnalysisPlugin, ArtifactSpec
from msi_autoencoder_wrapper.analysis.precompute.core.planner import ExecutionPlan, restrict_plan


def test_extension_tops_up_deficient_classes_and_counts_shared_pixels() -> None:
    ## Pixel 7 is positive for both classes, so drawing it for "a" also serves "b"
    candidates = {"a": np.array([1, 2, 7]), "b": np.array([7, 8, 9]), "c": np.array([3])}
    selected, report = select_test_extension({"a": 1, "b": 2, "c": 0}, candidates, minimum=3, seed=0)
    report = report.set_index("class_name")

    assert report.loc["a", "added"] == 2 and report.loc["a", "final_positives"] == 3
    assert report.loc["c", "added"] == 1 and report.loc["c", "shortfall"] == 2
    assert set(selected) >= {3}
    assert report.loc["b", "final_positives"] >= 3
    assert np.array_equal(selected, np.unique(selected))


def test_extension_is_deterministic_and_skips_satisfied_classes() -> None:
    candidates = {"a": np.arange(100), "b": np.arange(100, 200)}
    first, _ = select_test_extension({"a": 0, "b": 50}, candidates, minimum=10, seed=4)
    second, report = select_test_extension({"a": 0, "b": 50}, candidates, minimum=10, seed=4)

    np.testing.assert_array_equal(first, second)
    assert first.size == 10 and np.all(first < 100)
    assert report.set_index("class_name").loc["b", "added"] == 0


def test_class_sets_separate_common_boundary_and_extension() -> None:
    axes = {"narrow": ("a", "b"), "wide": ("a", "b", "c", "d")}
    mz = pd.Series({"a": 300.0, "b": 850.0, "c": 150.0, "d": 899.0})
    frame = class_set_labels(axes, mz, (200.0, 900.0)).set_index("class_name")

    assert frame.class_set.to_dict() == {"a": "common", "b": "common", "c": "extension", "d": "boundary"}
    assert not frame.loc["c", "on_narrow"] and frame.loc["c", "on_wide"]


def test_metaspace_images_align_by_offset_and_mark_missing_pixels(tmp_path) -> None:
    pd.DataFrame({"source_annotation_id": ["1", "2"], "mol_formula": ["C1", "C2"], "adduct": ["-H", "+Cl"],
                  "mz": [150.0, 300.0], "moleculeNames": ["", ""], "moleculeIds": ["", ""],
                  "x0_y0": [1.0, 5.0], "x1_y0": [2.0, 6.0], "x0_y1": [3.0, 7.0]}).to_csv(
        tmp_path / "pixel_intensities.csv", index=False)
    annotations, coordinates, intensities = read_metaspace_images(tmp_path)

    assert annotations.class_name.tolist() == ["C1|-H", "C2|+Cl"]
    ## imzML coordinates are 1-based; offset (-1, -1) maps (2, 1) -> (1, 0)
    aligned, coverage = align_metaspace(coordinates, intensities, np.array([[2, 1], [1, 2], [5, 5]]), (-1, -1))
    np.testing.assert_array_equal(aligned[:2], [[2.0, 6.0], [3.0, 7.0]])
    assert np.isnan(aligned[2]).all()
    assert coverage == pytest.approx(2 / 3)


def _plugin(name: str, requires: tuple[str, ...], artifact: str, analysis: str | None = None) -> AnalysisPlugin:
    return AnalysisPlugin(name=name, requires=requires, provides=(ArtifactSpec(artifact, analysis),),
                          run=lambda context: None, enabled_when_configured=analysis is None)


def test_restricted_plan_keeps_requested_analyses_and_their_shared_dependencies() -> None:
    plan = ExecutionPlan((
        _plugin("plan", ("model_catalog",), "plan"),
        _plugin("populations", ("plan",), "populations"),
        _plugin("inference", ("populations",), "inference"),
        _plugin("losses", ("model_catalog",), "losses", "losses"),
        _plugin("maps", ("model_catalog", "inference"), "maps", "maps"),
    ))

    assert [stage.name for stage in restrict_plan(plan, ["losses"]).stages] == ["losses"]
    assert [stage.name for stage in restrict_plan(plan, ["maps"]).stages] == ["plan", "populations", "inference", "maps"]
    with pytest.raises(ValueError):
        restrict_plan(plan, ["unknown"])


def test_class_positive_ids_follow_the_mapped_annotation_index() -> None:
    from types import SimpleNamespace

    from msi_autoencoder_wrapper.analysis.autoencoder.experiments.pretraining_campaign import class_positive_ids
    from msi_autoencoder_wrapper.models.datasets.annotations.index import MappedSpectrumAnnotationIndex

    ## Spectrum 5 carries a and b (b twice through two coordinates), spectrum 9 only a
    index = MappedSpectrumAnnotationIndex(
        spectrum_ids=np.array([5, 9]), spectrum_offsets=np.array([0, 3, 4]),
        annotation_indices=np.array([0, 1, 1, 0]), coordinate_indices=np.array([10, 20, 21, 10]),
        annotation_identities=(("A", "-H"), ("B", "-H")), coordinate_axis=np.arange(30.0),
        coordinate_system="binner")
    dataset = SimpleNamespace(get_mapped_annotation_index=lambda: index)
    positives = class_positive_ids(dataset, ("A|-H", "B|-H", "C|-H"))

    assert positives["A|-H"].tolist() == [5, 9]
    assert positives["B|-H"].tolist() == [5]
    assert positives["C|-H"].size == 0


def test_result_directories_are_per_axis_and_missing_tables_name_the_producing_command(tmp_path) -> None:
    from msi_autoencoder_wrapper.analysis.autoencoder.experiments.pretraining_campaign_precompute import (
        load_table,
        results_directory,
        write_results,
    )

    settings = {"repository_root": str(tmp_path), "settings_path": str(tmp_path / "analysis_settings.yaml"),
                "axes": {"axis-a": {"directory": "axis_a"}},
                "analyses": {"global": {"output_directory": "part/global_results"},
                             "local": {"output_directory": "part", "result_directory": "part_1_01_local_results"}}}
    assert results_directory(settings, "global") == tmp_path / "part/global_results"
    assert results_directory(settings, "local", "axis-a") == tmp_path / "part/axis_a/part_1_01_local_results"
    with pytest.raises(ValueError):
        results_directory(settings, "local")
    with pytest.raises(FileNotFoundError, match="--analysis local"):
        load_table(settings, "local", "summary", "axis-a")

    write_results(results_directory(settings, "local", "axis-a"),
                  {"summary": pd.DataFrame({"value": [1.5]}), "metadata": {"seed": 3}})
    assert load_table(settings, "local", "summary", "axis-a").value.tolist() == [1.5]


def test_class_strata_label_support_windows_presence_and_missing_sets() -> None:
    classes = pd.DataFrame({"class_name": ["a", "b", "c", "d"], "mz": [150.0, np.nan, 899.9, 1210.0],
                            "bin_mz": [150.0, 250.2, 899.9, 1210.0], "train_positives": [0, 5, 100, 2000],
                            "class_set": ["extension", "common", np.nan, "extension"]})
    strata = class_strata(classes, ["b", "d"], 100.0).set_index("class_name")

    assert strata.train_support.tolist() == ["0", "1-9", "100-999", "1000+"]
    ## Missing annotation m/z falls back to the bin m/z
    assert strata.mz_window.tolist() == ["100-200", "200-300", "800-900", "1200-1300"]
    assert strata.heldout_presence.tolist() == ["not_in_heldout", "in_heldout", "not_in_heldout", "in_heldout"]
    assert strata.loc["c", "class_set"] == "unassigned"


def _write_model(directory, masserstein: float, spectral_angle: float, average_precision: float) -> None:
    directory.mkdir()
    np.savez(directory / "test_pixels.npz", masserstein=np.full(4, masserstein), spectral_angle=np.full(4, spectral_angle))
    rows = [{"evaluation": "test", "group": "all", "population": "annotation_retrieval", "scope": "train_supported",
             "metric": metric, "value": value}
            for metric, value in (("average_precision", average_precision), ("micro_average_precision", 0.5),
                                  ("f1", 0.3))]
    ## A held-out row must not influence the selection
    rows.append({"evaluation": "heldout_image", "group": "all", "population": "annotation_retrieval",
                 "scope": "train_supported", "metric": "average_precision", "value": 1.0})
    pd.DataFrame(rows).to_csv(directory / "prediction.csv", index=False)


def test_representative_model_has_lowest_mean_test_rank_and_tie_breaker(tmp_path) -> None:
    _write_model(tmp_path / "m0", masserstein=3.0, spectral_angle=0.20, average_precision=0.70)
    _write_model(tmp_path / "m1", masserstein=2.9, spectral_angle=0.19, average_precision=0.69)
    _write_model(tmp_path / "m2", masserstein=3.1, spectral_angle=0.21, average_precision=0.71)
    models = pd.DataFrame({"model_id": ["m0", "m1", "m2"], "repetition": [0, 1, 2]})
    ranking = representative_ranking({}, models, lambda model_id: tmp_path / model_id)

    ## m1 wins both reconstruction metrics and loses AP; equal metrics share average ranks
    assert ranking.groupby("model_id").mean_rank.first().idxmin() == "m1"
    assert ranking[ranking.representative].model_id.unique().tolist() == ["m1"]
    assert (ranking[ranking.metric == "micro_average_precision"]["rank"] == 2.0).all()
    ## With only AP and the reconstruction metric in opposite order, the tie breaker decides
    tied = representative_ranking({"representative_model": {"metrics": {"masserstein": "lower",
                                                                        "average_precision": "higher"}}},
                                  models.iloc[:2], lambda model_id: tmp_path / model_id)
    assert tied[tied.representative].model_id.unique().tolist() == ["m0"]


def test_image_channels_are_peak_apexes_with_display_and_height_filters() -> None:
    mass_axis = np.arange(20, dtype=np.float64) + 100.0
    mean = np.zeros(20)
    mean[[3, 4, 5]] = [0.5, 1.0, 0.5]  # one peak: apex 4 only
    mean[[10, 11, 12]] = [0.2, 0.4, 0.2]
    mean[16] = 1e-5  # below the height threshold
    spectra = np.tile(mean, (3, 1))
    channels = _image_channels(spectra, {"a": np.arange(3)}, mass_axis, display_bins=np.array([4, 5, 16]),
                               options={"minimum_height_fraction": 1e-3, "minimum_distance_bins": 3})

    ## The apex at bin 11 has no stored reconstruction and is dropped
    assert channels["bin"].tolist() == [4]
    assert channels.mz.tolist() == [104.0]
    assert channels.height_fraction.tolist() == [1.0]


def test_generated_cells_have_unique_aliases_and_display_labels_across_axes() -> None:
    settings = {"axes": {"axis-a": {"directory": "axis_a", "label": "a m/z"}, "axis-b": {"directory": "axis_b", "label": "b m/z"}},
                "variants": {"P1": {"phases": ["p"], "label": "P1", "color": "#000000"}},
                "stages": {"pretrained": {"label": "pretrained"}, "unfrozen_head": {"label": "unfrozen"}},
                "models": {"baseline_a": {"display_label": "baseline, a m/z"}}}
    expand_cell_models(settings)
    labels = [model["display_label"] for model in settings["models"].values()]

    ## Regression: the same variant and stage on two axes must not share a display label
    assert len(settings["models"]) == 5
    assert len(set(labels)) == len(labels)
    assert settings["models"]["P1__pretrained__axis_b"]["display_label"] == "P1, pretrained, b m/z"


def test_bin_frequency_counts_pixels_with_enough_tic_per_bin() -> None:
    from msi_autoencoder_wrapper.analysis.autoencoder.experiments.pretraining_campaign_stage_reports import bin_frequency

    spectra = np.array([[0.5, 0.0005, 0.4995], [0.9, 0.1, 0.0], [1.0, 0.0, 0.0], [0.2, 0.2, 0.6]])
    ## Bin 1 is present (>= 1e-3) in two of four pixels; batches must not change the result
    assert bin_frequency(spectra, 1e-3, batch_size=3).tolist() == [1.0, 0.5, 0.5]
