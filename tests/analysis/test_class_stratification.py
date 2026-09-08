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


def test_empty_eligible_stratum_stays_nan_with_zero_denominator(taxonomy):
    strata = class_strata(*taxonomy)
    scores = pd.DataFrame(dict(class_name=strata.class_name, model_id='run', label='head', condition='condition',
                               repetition=0, split='validation', population='annotation_retrieval',
                               eligible=[True, True, False], positives=[1, 4, 0], negatives=[9, 6, 10],
                               average_precision=[.5, .8, np.nan], roc_auc=[.7, .9, np.nan],
                               ap_above_prevalence=[.4, .4, np.nan]))
    joined, metrics = stratified_metrics(scores, strata)
    assert len(joined) == 3
    group = metrics.query("grouping == 'part0_regime' and stratum == 'undetectable'")
    assert group.value.isna().all()
    assert group.eligible_classes.eq(0).all()
    assert group.total_classes.eq(1).all()
