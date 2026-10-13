"""Identity, rasterization and registration contracts for anatomical references."""

import json

import numpy as np
import pytest

from msi_autoencoder_wrapper.analysis.autoencoder.spatial.reference import (
    PixelMask, load_mask, project_histology_mask, rasterize_regions, save_mask,
)
from msi_autoencoder_wrapper.analysis.autoencoder.spatial.neighborhoods import PhysicalPixelIndex


@pytest.fixture
def mask_metadata():
    return {"dataset_id": "fixture", "image_id": "section", "coordinate_space": "msi_source",
            "class_names": {"0": "cortex", "1": "tumor"}, "usage_role": "external_test_only",
            "annotation_status": "verified_fixture", "provenance": {"source": "analytic"}}


def test_mask_roundtrip_and_permuted_identity(tmp_path, mask_metadata):
    mask = PixelMask(np.array([100, 4, 20]), np.array([1, -1, 0]), mask_metadata)
    save_mask(mask, tmp_path)
    restored = load_mask(tmp_path)
    np.testing.assert_array_equal(restored.align_to(np.array([20, 100, 4])), [0, 1, -1])
    assert restored.available.tolist() == [True, False, True]
    with pytest.raises(ValueError):restored.align_to(np.array([5]))
    with pytest.raises(ValueError):restored.align_to(np.array([20, 20]))
    with pytest.raises(ValueError):restored.pixel_ids[0] = 7


def test_checksum_and_schema_corruption(tmp_path, mask_metadata):
    save_mask(PixelMask(np.arange(2), np.array([0, 1]), mask_metadata), tmp_path)
    manifest = tmp_path / "annotation_manifest.json"
    obj = json.loads(manifest.read_text());obj['schema_version'] = 2;manifest.write_text(json.dumps(obj))
    with pytest.raises(ValueError):load_mask(tmp_path)
    obj['schema_version'] = 1;manifest.write_text(json.dumps(obj))
    with (tmp_path / 'mask.npz').open('ab') as f:f.write(b'corruption')
    with pytest.raises(ValueError):load_mask(tmp_path)


@pytest.mark.parametrize('ids,labels', [([1,1],[0,1]), ([1,-1],[0,1]), ([0,1],[0,8]), ([0.,1.],[0,1])])
def test_invalid_identity_rejected(ids, labels, mask_metadata):
    with pytest.raises(ValueError):PixelMask(np.asarray(ids),np.asarray(labels),mask_metadata)


def test_analytic_polygon_areas_overlap_unknown_and_order_invariance():
    regions = [{'class_id':0,'vertices':[[0,0],[3,0],[3,3],[0,3]]},
               {'class_id':1,'vertices':[[2,0],[4,0],[4,2],[2,2]]}]
    raster, conflict = rasterize_regions((4,5),regions)
    assert conflict.sum() == 2
    assert (raster==0).sum() == 7
    assert (raster==1).sum() == 2
    second, _ = rasterize_regions((4,5),list(reversed(regions)))
    np.testing.assert_array_equal(raster,second)


def test_affine_registration_identity_translation_and_rejection(mask_metadata):
    meta = dict(mask_metadata,coordinate_space='histology_image',image_shape=[2,2])
    mask = PixelMask(np.arange(4),np.array([0,1,1,-1]),meta)
    registration={'status':'accepted','msi_to_histology':np.eye(3).tolist()}
    projected = project_histology_mask(mask,np.array([[.5,.5],[1.5,.5],[4,4]]),np.array([7,8,9]),registration,'msi')
    np.testing.assert_array_equal(projected.labels,[0,1,-1])
    registration['msi_to_histology'][0][2]=1
    projected = project_histology_mask(mask,np.array([[.5,.5]]),np.array([7]),registration,'msi')
    assert projected.labels[0]==1
    registration['status']='pending'
    with pytest.raises(ValueError):project_histology_mask(mask,np.array([[0.,0.]]),np.array([7]),registration,'msi')


def test_physical_neighbors_brute_force_oracle_and_permutations():
    rng=np.random.default_rng(42)
    coords=np.column_stack([np.arange(20),rng.permutation(20)])
    ids=np.arange(20)*11+3
    spacing=(20.,80.)
    for order in [np.arange(20),rng.permutation(20)]:
        index=PhysicalPixelIndex(ids[order],coords[order],spacing)
        for i in range(20):
            got,dist=index.neighbours(int(ids[i]),200.)
            oracle=[]
            for j in range(20):
                d=sum(((coords[j,k]-coords[i,k])*spacing[k])**2 for k in range(2))**.5
                if j!=i and d<=200:oracle.append((ids[j],d))
            np.testing.assert_array_equal(got,[p[0] for p in oracle])
            np.testing.assert_allclose(dist,[p[1] for p in oracle],atol=1e-12)


def test_neighbours_radius_boundary_and_excluded_population():
    index=PhysicalPixelIndex(np.array([50,90]),np.array([[0,0],[1,0]]),(20.,20.))
    assert index.neighbours(50,20.)[0].tolist()==[90]
    assert index.neighbours(50,19.9)[0].size==0
    assert index.neighbours(50,0)[0].size==0
    with pytest.raises(ValueError):index.neighbours(60,100)


def test_compact_merge_map_feeds_source_neighbours_without_cross_image_edges():
    import pandas as pd
    from msi_autoencoder_wrapper.analysis.autoencoder.spatial.coordinates import MergedPixelMap
    from msi_autoencoder_wrapper.analysis.autoencoder.spatial.neighborhoods import source_indices_from_merged_map
    segments=pd.DataFrame({'dataset_index':[0,1],'merged_pixel_start':[0,3],'segment_length':[3,3],
                           'source_pixel_start':[0,0],'source_step':[1,1]})
    datasets=pd.DataFrame({'dataset_index':[0,1],'dataset_id':['a','b'],'name':['a','b'],'image_path':['a.imzML','b.imzML']})
    pixel_map=MergedPixelMap(segments,datasets)
    coords=np.array([[0,0],[1,0],[2,0]])
    indices,mapping=source_indices_from_merged_map(pixel_map,np.array([0,2,3,4]),{'a':coords,'b':coords}, {'a':(20.,20.),'b':(80.,80.)})
    assert mapping.source_id.tolist()==[0,2,3,4]
    assert indices['a'].neighbours(0,20.)[0].size==0
    assert indices['b'].neighbours(0,80.)[0].tolist()==[1]
    with pytest.raises(ValueError):source_indices_from_merged_map(pixel_map,np.array([0]),{'a':coords},{})
