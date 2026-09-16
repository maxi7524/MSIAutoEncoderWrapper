"""Semantic class joins, historical thresholds and stratum denominator regressions."""
import numpy as np
import pandas as pd
import pytest
from msi_autoencoder_wrapper.analysis.autoencoder.heads.class_stratification import class_strata, stratified_metrics


@pytest.fixture
def taxonomy():
    names = ['rare-ion', 'common-ion', 'silent-ion']
    prevalence = pd.DataFrame(dict(class_name=names, mz=[250., 450., 650.], prevalence=[.001, .1, .001]))
    regimes = pd.DataFrame(dict(class_name=names, regime=['rare', 'common', 'undetectable'],
                                relative_threshold=.005309, bin_radius=1, negative_fraction_of_unannotated=[.2, .3, .99]))
    separation = pd.DataFrame(dict(class_name=names[::-1], evidence_auc=[.4, .8, .6]))
    train = pd.DataFrame(dict(class_name=names, positive_entries=[1, 20, 1], negative_entries=[98, 30, 20],
                              uncertain_entries=[1, 50, 179], relative_threshold=.0119, bin_radius=1))
    return prevalence, regimes, separation, train


def test_strata_join_names_preserve_history_and_use_training_counts(taxonomy):
    result = class_strata(*taxonomy).set_index('class_name')
    assert result.loc['rare-ion', 'part0_regime'] == 'rare'
    assert result.loc['rare-ion', 'train_regime'] == 'undetectable'
    assert result.loc['silent-ion', 'part0_regime'] == 'undetectable'
    assert result.loc['silent-ion', 'train_regime'] == 'rare'
    assert result.loc['rare-ion', 'evidence_auc'] == .6
    assert result.part0_threshold.iloc[0] != result.relative_threshold.iloc[0]


def test_missing_or_duplicate_class_is_rejected(taxonomy):
    prevalence, regimes, separation, train = taxonomy
    with pytest.raises(ValueError, match='catalogues differ'):
        class_strata(prevalence.iloc[:2], regimes, separation, train)
    with pytest.raises(ValueError, match='exactly one'):
        class_strata(prevalence, pd.concat([regimes, regimes.iloc[:1]]), separation, train)


def _per_class_scores(strata, **overrides):
    base = dict(class_name=strata.class_name, model_id='run', label='head', condition='condition',
               repetition=0, split='validation', population='annotation_retrieval',
               eligible=[True, True, False], positives=[1, 4, 0], negatives=[9, 6, 10], available=[10, 10, 10],
               average_precision=[.5, .8, np.nan], roc_auc=[.7, .9, np.nan], ap_above_prevalence=[.4, .4, np.nan],
               precision=[np.nan, np.nan, np.nan], recall=[np.nan, np.nan, np.nan], f1=[np.nan, np.nan, np.nan],
               true_positive=[np.nan, np.nan, np.nan], false_positive=[np.nan, np.nan, np.nan],
               false_negative=[np.nan, np.nan, np.nan])
    base.update(overrides)
    return pd.DataFrame(base)


def test_empty_eligible_stratum_stays_nan_with_zero_denominator(taxonomy):
    strata = class_strata(*taxonomy)
    scores = _per_class_scores(strata)
    joined, metrics = stratified_metrics(scores, strata)
    assert len(joined) == 3
    group = metrics.query("grouping == 'part0_regime' and stratum == 'undetectable'")
    assert group.value.isna().all()
    assert group.eligible_classes.eq(0).all()
    assert group.total_classes.eq(1).all()


def test_stratum_without_threshold_anchored_classes_leaves_threshold_metrics_nan(taxonomy):
    # No class carries precision/recall/f1/confusion counts (as `ranking_tables`
    # leaves them for a non-threshold-anchored family): every threshold metric,
    # macro and micro alike, must stay `nan`, not fall back to a spurious zero.
    strata = class_strata(*taxonomy)
    scores = _per_class_scores(strata)
    _, metrics = stratified_metrics(scores, strata)
    threshold_metrics = metrics.query("metric in ['precision', 'recall', 'f1', 'micro_precision', "
                                      "'micro_recall', 'micro_f1', 'hamming_loss']")
    assert threshold_metrics.value.isna().all()


def test_stratum_pools_micro_threshold_metrics_from_raw_confusion_counts(taxonomy):
    # `rare-ion` and `silent-ion` both have 1 training positive, so both fall in the
    # `train_support` stratum '1-9 positives' (`common-ion` has 20, a different
    # stratum, and is irrelevant here). Pooling their confusion counts must
    # reproduce the standard micro precision/recall/F1/Hamming-loss definitions
    # exactly, and must differ from the plain mean of their own precision/recall/F1
    # (the macro convention) -- otherwise this test could not tell the two apart.
    strata = class_strata(*taxonomy)
    assert set(strata.loc[strata.train_support == '1-9 positives', 'class_name']) == {'rare-ion', 'silent-ion'}
    # class order is [rare-ion, common-ion, silent-ion]; common-ion's values are unused.
    scores = _per_class_scores(
        strata, eligible=[True, True, True],
        precision=[1.0, np.nan, .1], recall=[.8, np.nan, 1.0], f1=[8 / 9, np.nan, 2 / 11],
        true_positive=[8.0, np.nan, 1.0], false_positive=[0.0, np.nan, 9.0], false_negative=[2.0, np.nan, 0.0],
    )
    _, metrics = stratified_metrics(scores, strata)
    stratum = metrics.query("grouping == 'train_support' and stratum == '1-9 positives'").set_index("metric").value
    total_tp, total_fp, total_fn, total_available = 8.0 + 1.0, 0.0 + 9.0, 2.0 + 0.0, 10 + 10
    expected_precision = total_tp / (total_tp + total_fp)
    expected_recall = total_tp / (total_tp + total_fn)
    expected_f1 = 2 * expected_precision * expected_recall / (expected_precision + expected_recall)
    assert stratum.micro_precision == pytest.approx(expected_precision)
    assert stratum.micro_recall == pytest.approx(expected_recall)
    assert stratum.micro_f1 == pytest.approx(expected_f1)
    assert stratum.hamming_loss == pytest.approx((total_fp + total_fn) / total_available)
    # Macro is the plain mean of the stratum's own per-class ratios, which differs
    # from the pooled micro value above (0.55 vs. 0.5).
    assert stratum.precision == pytest.approx((1.0 + .1) / 2)
    assert stratum.precision != pytest.approx(expected_precision)
