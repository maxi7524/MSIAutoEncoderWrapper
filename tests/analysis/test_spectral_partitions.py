"""Known spatial/spectral fixtures and frozen-model inference contracts."""

import numpy as np
import pytest
import torch

from msi_autoencoder_wrapper.analysis.autoencoder.spatial.synthetic_partitions import (
    cluster_partition_representation, infer_frozen_encoder, spatial_reference, spectral_partition,
)
from msi_autoencoder_wrapper.analysis.autoencoder.spatial.reference_metrics import score_partition


@pytest.fixture
def benchmark_options():
    return {'side':24,'classes':4,'mass_axis':np.linspace(100,200,101),'peak_positions':[120.,140.,160.,180.],
            'peak_width_mz':1.,'amplitude_noise':.05,'seed':101}


@pytest.mark.parametrize('layout',['many_stripes','concentric_rings'])
def test_reference_class_coverage_and_geometry(layout):
    ref=spatial_reference(layout,side=32,classes=5)
    assert ref.shape==(32,32)
    assert set(np.unique(ref[ref>=0]))==set(range(5))
    if layout=='concentric_rings':
        np.testing.assert_array_equal(ref,ref[::-1])
        np.testing.assert_array_equal(ref,ref[:,::-1])
        assert (ref==-1).sum()>0
        assert (ref==0).sum()<(ref==4).sum()
    else:assert (ref>=0).all()


def test_two_stripes_known_indices_and_invalid_dimensions():
    ref=spatial_reference('two_stripes',side=8,classes=2)
    np.testing.assert_array_equal(ref[0],[0,0,0,0,1,1,1,1])
    with pytest.raises(ValueError):spatial_reference('two_stripes',side=8,classes=3)
    with pytest.raises(ValueError):spatial_reference('concentric_rings',side=4,classes=3)


def test_spectra_reproducibility_identity_tic_and_noise(benchmark_options):
    first=spectral_partition('concentric_rings',**benchmark_options)
    second=spectral_partition('concentric_rings',**benchmark_options)
    np.testing.assert_array_equal(first.spectra,second.spectra)
    np.testing.assert_array_equal(first.labels,first.reference.ravel()[first.pixel_ids])
    np.testing.assert_allclose(first.spectra.sum(axis=1),1,atol=1e-6,rtol=1e-6)
    assert np.isfinite(first.spectra).all() and (first.spectra>=0).all()
    assert len(np.unique(first.prototypes,axis=0))==benchmark_options['classes']
    other=spectral_partition('concentric_rings',**dict(benchmark_options,seed=102))
    assert not np.array_equal(first.spectra,other.spectra)
    np.testing.assert_array_equal(first.reference,other.reference)


def test_noiseless_spectra_have_exact_region_prototypes(benchmark_options):
    fixture=spectral_partition('many_stripes',**dict(benchmark_options,amplitude_noise=0))
    for label in range(benchmark_options['classes']):
        values=fixture.spectra[fixture.labels==label]
        np.testing.assert_array_equal(values,np.repeat(values[:1],len(values),axis=0))


@pytest.mark.parametrize('method',['kmeans','hdbscan','gmm_diag'])
def test_cluster_methods_recover_separated_raw_spectra(method,benchmark_options):
    fixture=spectral_partition('many_stripes',**dict(benchmark_options,amplitude_noise=.01))
    labels,metadata=cluster_partition_representation(fixture.spectra,method=method,classes=benchmark_options['classes'],
        seed=101,min_cluster_size=10,gmm_regularization=1e-6)
    scores=score_partition(fixture.labels,labels)
    assert scores.summary['matched_accuracy']>.99
    if method=='gmm_diag':assert metadata['converged']


class FixtureEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout=torch.nn.Dropout(.5)
        self.projection=torch.nn.Linear(101,3)
    def forward(self,x):
        return {'latent_space':self.projection(self.dropout(x)),'reconstruction':x}


def test_frozen_inference_batch_invariance_and_flag_restoration(benchmark_options):
    fixture=spectral_partition('many_stripes',**benchmark_options)
    torch.manual_seed(101)
    model=FixtureEncoder().train();model.dropout.eval()
    original={k:v.clone() for k,v in model.state_dict().items()}
    first,_=infer_frozen_encoder(model,fixture.spectra,device='cpu',batch_size=13)
    second,_=infer_frozen_encoder(model,fixture.spectra,device='cpu',batch_size=101)
    np.testing.assert_allclose(first,second,atol=1e-7,rtol=1e-6)
    assert model.training and not model.dropout.training
    for key,value in model.state_dict().items():torch.testing.assert_close(value,original[key],atol=0,rtol=0)
    with pytest.raises(ValueError):infer_frozen_encoder(model,np.zeros_like(fixture.spectra),device='cpu',batch_size=13)
