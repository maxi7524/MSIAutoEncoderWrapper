"""End-to-end report contract on a small known source image, without training."""

import json

import numpy as np
import pytest

from msi_autoencoder_wrapper.analysis.autoencoder.experiments.reference_validation_precompute import write_prediction_report
from msi_autoencoder_wrapper.analysis.autoencoder.experiments.reference_numerical_validation import validate_numerics
from msi_autoencoder_wrapper.analysis.autoencoder.spatial.evaluation_data import EvaluationImage
from msi_autoencoder_wrapper.analysis.autoencoder.spatial.reference import PixelMask
from msi_autoencoder_wrapper.analysis.autoencoder.spatial.partition_analysis import chemical_cluster_profiles
from msi_autoencoder_wrapper.analysis.autoencoder.heads.metrics import evaluate_head


@pytest.fixture
def report_fixture():
    y,x=np.mgrid[:4,:6];truth=(x.ravel()>=3).astype(int);n=len(truth)
    mask=PixelMask(np.arange(n),truth,{'dataset_id':'fixture','image_id':'analytic', 'coordinate_space':'msi_source',
        'class_names':{'0':'left','1':'right'},'usage_role':'external_test_only','annotation_status':'verified_fixture','provenance':{'source':'analytic'}})
    values=np.column_stack([np.ones(n),truth,.1*np.ones(n)])
    image=EvaluationImage('fixture','analytic',np.arange(n),values,np.array([100.,101.,102.]),np.column_stack([x.ravel(),y.ravel()]),mask,
        {'usage_role':'external_test_only','representation':'analytic_fixture','coordinate_units':'source_grid'})
    outputs={'dataset_id':np.asarray(image.dataset_id),'image_id':np.asarray(image.image_id),'pixel_ids':image.pixel_ids,'latents':np.column_stack([truth,truth*.1]),'reconstruction':values.copy()}
    settings={'reference_prevalence':[1.,.5,1.],'rare_threshold':.1,'device':'cpu','seeds':[1,2,3],
        'clustering':{'methods':['kmeans'],'n_clusters':2,'min_cluster_size':3},'boundary_tolerance_pixels':0.,'max_raster_pixels':100,
        'annotations_complete':True,'prediction_threshold':.5}
    return image,outputs,settings


def test_complete_report_and_numeric_artifacts(tmp_path,report_fixture):
    image,outputs,settings=report_fixture
    result=write_prediction_report(image,outputs,settings,tmp_path,context='ANALYTIC FIXTURE')
    assert result['segmentation_available']
    assert (tmp_path/'reports/model_analysis.md').is_file()
    assert (tmp_path/'plots/segmentation_kmeans.png').is_file()
    assert (tmp_path/'plots/reconstruction_mae.png').is_file()
    summary=json.loads((tmp_path/'numerical_values/summary.json').read_text())
    assert summary['latent']['numerical_rank']==1
    assert (tmp_path/'numerical_values/interiors_kmeans_1.csv').is_file()
    outputs['pixel_ids']=outputs['pixel_ids'][::-1]
    with pytest.raises(ValueError):write_prediction_report(image,outputs,settings,tmp_path,context='incorrect join')


def test_prediction_precision_known_counts_and_extreme_logits():
    target=np.array([[1,0],[0,1],[1,0]])
    logits=np.array([[1000.,-1000.],[1000.,1000.],[-1000.,-1000.]])
    result=evaluate_head(logits,target,'multi_label',mask=np.ones(target.shape,bool),threshold=.5)
    assert result['micro_precision']==pytest.approx(2/3)
    assert result['micro_recall']==pytest.approx(2/3)


def test_chemical_overlap_requires_confirmed_absences():
    clusters=np.array([0,0,1,1]);target=np.array([[1,0],[1,0],[0,1],[0,1]])
    profiles,scores=chemical_cluster_profiles(clusters,target,np.ones(target.shape,bool),annotations_complete=True)
    assert (scores.ari==1).all()
    assert profiles.observed_positive_fraction.tolist()==[1,0,0,1]
    _,incomplete=chemical_cluster_profiles(clusters,target,np.ones(target.shape,bool),annotations_complete=False)
    assert incomplete.ari.isna().all()


def test_cpu_numerical_reference_and_gradcheck():
    checks,gradients=validate_numerics(device='cpu',seed=101)
    assert checks.passed.all()
    assert gradients.finite.all()


def test_cuda_numerical_reference_and_gradients():
    import torch
    if not torch.cuda.is_available():pytest.skip('CUDA unavailable in this execution environment')
    checks,gradients=validate_numerics(device='cuda',seed=101)
    assert checks.passed.all()
    assert gradients.finite.all()
    assert 'cuda' in checks.device.tolist()


def test_gradient_snapshots_distinguish_missing_zero_nonfinite_and_valid():
    import torch
    from msi_autoencoder_wrapper.analysis.autoencoder.latent.diagnostics import gradient_diagnostics
    table=gradient_diagnostics({'missing':None,'zero':torch.zeros(3),'invalid':torch.tensor([float('nan')]),'valid':torch.tensor([3.,4.])}).set_index('name')
    assert not table.loc['missing','present']
    assert table.loc['zero','zero_fraction']==1
    assert not table.loc['invalid','finite']
    assert table.loc['valid','finite']
    assert table.loc['valid','norm']==5


def test_gradient_norm_scaling_avoids_overflow():
    import torch
    from msi_autoencoder_wrapper.analysis.autoencoder.latent.diagnostics import gradient_diagnostics
    table=gradient_diagnostics({'large':torch.tensor([1e200,1e200],dtype=torch.float64)})
    assert table.iloc[0]['norm']==pytest.approx(2**.5*1e200,rel=1e-14)
