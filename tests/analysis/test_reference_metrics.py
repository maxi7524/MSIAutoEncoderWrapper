"""Independent oracles and invariants for anatomical segmentation scores."""

from itertools import permutations

import numpy as np
import pytest

from msi_autoencoder_wrapper.analysis.autoencoder.spatial.reference_metrics import score_boundaries, score_partition
from msi_autoencoder_wrapper.analysis.autoencoder.latent.diagnostics import representation_diagnostics
from msi_autoencoder_wrapper.analysis.autoencoder.reconstruction.diagnostics import reconstruction_diagnostics


def test_perfect_partition_and_noise_penalty():
    truth=np.array([0,0,0,1,1,1,-1])
    perfect=score_partition(truth,np.array([8,8,8,4,4,4,99]))
    assert perfect.summary['ari']==pytest.approx(1)
    assert perfect.summary['matched_accuracy']==1
    assert perfect.summary['macro_dice']==1
    noisy=score_partition(truth,np.array([8,-1,8,4,4,-1,99]))
    assert noisy.summary['matched_accuracy']==pytest.approx(4/6)
    assert noisy.summary['noise_fraction']==pytest.approx(2/6)
    assert noisy.summary['ari_covered']==1
    assert noisy.summary['macro_dice']==pytest.approx(.8)


def test_rectangular_matching_against_exhaustive_oracle():
    rng=np.random.default_rng(42)
    for _ in range(50):
        truth=rng.integers(0,2,12);pred=rng.integers(0,3,12)
        result=score_partition(truth,pred)
        actual_classes=np.unique(truth)
        actual_clusters=np.unique(pred)
        # Exhaustively assign distinct predicted clusters to each anatomical class.
        if len(actual_clusters)>=len(actual_classes):
            expected=max(sum(np.sum((truth==label)&(pred==cluster)) for label,cluster in zip(actual_classes,p))
                         for p in permutations(actual_clusters,len(actual_classes)))
            assert result.summary['matched_accuracy']==pytest.approx(expected/len(truth))


def test_label_and_row_permutation_invariants():
    truth=np.array([0,0,0,1,1,1,2,2,2]);pred=np.array([3,3,4,4,4,4,8,-1,8])
    original=score_partition(truth,pred)
    transformed=score_partition(np.array([4,7,9])[truth],np.where(pred>=0,pred+100,-1))
    for metric in ['ari','ami','macro_dice','macro_iou','matched_accuracy','noise_fraction']:
        assert original.summary[metric]==pytest.approx(transformed.summary[metric],abs=1e-12)
    order=np.random.default_rng(10).permutation(len(truth))
    second=score_partition(truth[order],pred[order])
    assert second.summary==original.summary


def test_collapse_and_unknown_only():
    truth=np.array([0,0,1,1]);collapsed=score_partition(truth,np.zeros(4,dtype=int))
    assert collapsed.summary['ari']==0
    assert collapsed.summary['matched_accuracy']==.5
    all_noise=score_partition(truth,np.full(4,-1))
    assert all_noise.summary['macro_dice']==0
    assert all_noise.summary['matched_accuracy']==0
    with pytest.raises(ValueError):score_partition(np.full(4,-1),truth)


def test_boundary_shift_and_interior_metrics():
    reference=np.zeros((10,12),dtype=int);reference[:,6:]=1
    perfect,interior=score_boundaries(reference,reference,tolerance=0)
    assert perfect['boundary_f1']==1
    assert (interior.interior_accuracy==1).all()
    pred=np.zeros_like(reference);pred[:,9:]=1
    shifted,_=score_boundaries(reference,pred,tolerance=0)
    assert shifted['boundary_f1']==0
    relaxed,_=score_boundaries(reference,pred,tolerance=3)
    assert relaxed['boundary_f1']==1
    scaled,_=score_boundaries(reference,pred,tolerance=60,spacing=(20,20))
    assert scaled['boundary_f1']==relaxed['boundary_f1']


def test_unknown_holes_do_not_create_tissue_boundaries():
    truth=np.zeros((6,6),dtype=int);truth[2:4,2:4]=-1
    result,_=score_boundaries(truth,truth,tolerance=0)
    assert result['reference_boundary_pixels']==0
    assert result['boundary_f1'] is None


def test_latent_rank_zero_rank_one_and_rotation_scale_invariants():
    collapse,_=representation_diagnostics(np.ones((8,4)))
    assert collapse['effective_rank']==0
    assert collapse['numerical_rank']==0
    z=np.arange(8.)[:,None]@np.array([[1.,2.,3.,4.]])
    rank,_=representation_diagnostics(z)
    assert rank['numerical_rank']==1
    assert rank['effective_rank']==pytest.approx(1)
    rng=np.random.default_rng(5);z=rng.normal(size=(20,4));q=np.linalg.qr(rng.normal(size=(4,4)))[0]
    original,_=representation_diagnostics(z)
    altered,_=representation_diagnostics((z@q)*1e100)
    for k in ['effective_rank','participation_ratio']:assert original[k]==pytest.approx(altered[k],abs=1e-12)


def test_reconstruction_rare_error_and_largest_failures():
    x=np.array([[1.,0.,0.],[1.,0.,2.],[1.,0.,0.]])
    y=x.copy();y[1,2]=0
    result=reconstruction_diagnostics(x,y,np.array([100.,101.,102.]),np.array([1.,0.,.1]),rare_threshold=.2)
    assert result['worst_errors'].iloc[0].absolute_error==2
    assert result['worst_errors'].iloc[0].rare
    rare=result['features'].iloc[2]
    assert rare.positive_relative_error==1
    assert rare.missed_positive_fraction==1
    assert result['spectra'].rare_mae.max()==2
    perfect=reconstruction_diagnostics(x,x,np.array([100.,101.,102.]),np.array([1.,0.,.1]),rare_threshold=.2)
    assert perfect['spectra'].mae.max()==0
    assert perfect['spectra'].masserstein.max()==0


def test_extreme_nonfinite_and_undefined_reconstruction_domains():
    axis=np.array([100.,101.]);prevalence=np.ones(2)
    with pytest.raises(ValueError):
        reconstruction_diagnostics(np.full((2,2),1e100),np.ones((2,2)),axis,prevalence,rare_threshold=.1)
    with pytest.raises(ValueError):
        representation_diagnostics(np.array([[1.,np.nan],[0.,1.]]))
    zero=reconstruction_diagnostics(np.zeros((2,2)),np.zeros((2,2)),axis,prevalence,rare_threshold=.1)
    assert zero['features'].positive_relative_error.isna().all()
    assert zero['spectra'].masserstein.eq(0).all()


def test_reconstruction_scaling_preserves_normalized_transport():
    x=np.array([[1.,2.,3.],[3.,2.,1.]])
    y=x[:,::-1].copy();axis=np.array([100.,101.,102.]);prevalence=np.ones(3)
    base=reconstruction_diagnostics(x,y,axis,prevalence,rare_threshold=.1)
    scaled=reconstruction_diagnostics(x*1e6,y*1e6,axis,prevalence,rare_threshold=.1)
    np.testing.assert_allclose(base['spectra'].masserstein,scaled['spectra'].masserstein,rtol=1e-6,atol=1e-6)
    np.testing.assert_allclose(base['features'].mae*1e6,scaled['features'].mae,rtol=1e-6)
