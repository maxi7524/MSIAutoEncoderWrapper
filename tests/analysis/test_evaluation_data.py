"""Source-adapter and canonical-cache contracts on deterministic structures."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from msi_autoencoder_wrapper.analysis.autoencoder.spatial.evaluation_data import (
    blocksam_from_objects, cardinal_from_arrays, load_evaluation_image,
    poseidon_from_frames, save_evaluation_image,
)


@pytest.fixture
def coordinate_frame():
    return pd.DataFrame({'X':[0.,1.,2.],'Y':[0.,0.,0.]})


def test_poseidon_axis_order_identity_and_representation_guard(coordinate_frame):
    assay=pd.DataFrame({'200.0':[1.,2.,3.],'100.0':[4.,5.,6.]})
    image=poseidon_from_frames(coordinate_frame,assay,assay_name='Lipids')
    np.testing.assert_array_equal(image.mass_axis,[100.,200.])
    np.testing.assert_array_equal(image.values[:,0],[4.,5.,6.])
    image.assert_use('fixed_clustering')
    for purpose in ['training','pretraining','preprocessing_fit','model_selection']:
        with pytest.raises(ValueError):image.assert_use(purpose)
    with pytest.raises(ValueError):image.assert_model_input('source_binned_intensity')
    assay.index=[3,2,1]
    with pytest.raises(ValueError):poseidon_from_frames(coordinate_frame,assay,assay_name='Lipids')


def test_blocksam_source_rows_and_sample_namespace():
    infos=[SimpleNamespace(name=np.array([name]),className=np.array([label]),
             x=np.array([i]),y=np.array([0]),patNum=np.array([1]))
             for i,(name,label) in enumerate([('a','Normal'),('b','RCC'),('a','Normal')])]
    image=blocksam_from_objects({'pixelInfoAll':infos,'xall':np.arange(6).reshape(3,2),'filtered_mz':np.array([100.,200.])},'a')
    assert image.pixel_ids.tolist()==[0,2]
    assert image.annotation.labels.tolist()==[0,0]
    with pytest.raises(ValueError):blocksam_from_objects({'pixelInfoAll':infos,'xall':np.zeros((1,2)),'filtered_mz':np.array([100.,200.])},'a')


def test_cardinal_missing_diagnosis_source_alignment_and_roundtrip(tmp_path):
    frame=pd.DataFrame({'sample':['a','b','a'],'x':[1,1,2],'y':[1,1,1],'diagnosis':['normal','cancer',np.nan]})
    image=cardinal_from_arrays(frame,np.array([100.,200.]),np.arange(6).reshape(2,3),'a')
    assert image.annotation.labels.tolist()==[0,-1]
    assert image.pixel_ids.tolist()==[0,2]
    np.testing.assert_array_equal(image.values,[[0,3],[2,5]])
    save_evaluation_image(image,tmp_path)
    got=load_evaluation_image(tmp_path)
    np.testing.assert_array_equal(got.values,image.values)
    np.testing.assert_array_equal(got.annotation.labels,image.annotation.labels)
    with (tmp_path/'values.npy').open('ab') as f:f.write(b'bad')
    with pytest.raises(ValueError):load_evaluation_image(tmp_path)


def test_poseidon_positional_contract_is_explicit(coordinate_frame):
    assay=pd.DataFrame({'100.0':[1.,2.,3.]})
    coordinate_frame.index=[2,4,6]
    with pytest.raises(ValueError):poseidon_from_frames(coordinate_frame,assay,assay_name='Lipids')
    image=poseidon_from_frames(coordinate_frame,assay,assay_name='Lipids',row_alignment='author_positional')
    assert image.metadata['source_coordinate_row_names']==['2','4','6']
    assert image.metadata['source_assay_row_names']==['0','1','2']
    assert image.pixel_ids.tolist()==[0,1,2]
