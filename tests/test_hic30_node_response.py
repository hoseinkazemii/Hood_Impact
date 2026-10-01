"""Scientific checks for pair dependence, missing variation and map ranking."""
import numpy as np

from analyze_hic30_node_response import (
    FAMILY_OF, WITHIN_PAIRS, rank_columns, score_fields,
    within_permutation_indices, permutation_maxima, leave_design_out,
    spatial_candidates, identical_rank_profiles,
)


def test_constant_geometry_is_unidentifiable_not_zero_effect():
    result = score_fields(np.ones((6, 3)), np.arange(6.)[:,None])
    assert np.isnan(result).all()


def test_monotone_and_inverse_distance_association_with_ties():
    x=np.array([0.,0.,1.,2.,4.,5.])
    result=score_fields(np.column_stack([x,-x,np.ones(6)]),x**3)
    np.testing.assert_allclose(result[:2,0],[1.,-1.])
    assert np.isnan(result[2,0])


def test_family_centering_removes_pure_family_offset():
    groups=np.array([FAMILY_OF[int(a)] for a,_ in WITHIN_PAIRS])
    x=groups[:,None].astype(float)
    r,valid=rank_columns(x,groups)
    assert not valid.any()
    assert np.count_nonzero(r)==0
    y=np.arange(len(groups),dtype=float)
    assert np.isnan(score_fields(x,y,groups)).all()


def test_two_design_families_have_no_pair_ordering_information():
    groups=np.array([FAMILY_OF[int(a)] for a,_ in WITHIN_PAIRS])
    x=np.arange(len(groups),dtype=float)
    r,_=rank_columns(x,groups)
    assert r[groups==1].item()==0
    assert r[groups==3].item()==0


def test_permutations_preserve_shared_design_incidence_and_families():
    indices=within_permutation_indices()
    assert indices.shape==(576,14)
    assert len(np.unique(indices,axis=0))==576
    for ix in indices:
        assert sorted(ix)==list(range(14))
        for i,(a,b) in enumerate(WITHIN_PAIRS):
            assert FAMILY_OF[int(a)]==FAMILY_OF[int(WITHIN_PAIRS[ix[i],0])]
        # The images of a triangle are another triangle, never arbitrary pairs.
        tri=[i for i,p in enumerate(WITHIN_PAIRS) if tuple(p) in [(0,1),(0,2),(1,2)]]
        assert len(set(WITHIN_PAIRS[ix[tri]].ravel()))==3


def test_permutation_scan_includes_the_observed_maximum():
    rng=np.random.default_rng(31)
    x=rng.normal(size=(14,5))
    y=x[:,:2].copy()
    maximum=permutation_maxima(x,y,np.ones(5,dtype=bool))
    assert maximum.shape==(576,2)
    assert np.max(maximum)>=1-1e-12
    assert np.all(maximum<=1+1e-12)
    # The exact identity relabeling gives the known perfect matches.
    np.testing.assert_allclose(maximum[0],[1,1],atol=1e-12)


def test_single_design_dependency_is_flagged_on_deletion():
    x=np.zeros((14,1))
    x[np.any(WITHIN_PAIRS==0,axis=1),0]=1
    y=x.copy()
    q,p,count=leave_design_out(x,y)
    assert np.isnan(q).all()
    assert np.isnan(p).all()
    assert count.item()<8


def test_spatial_candidates_are_positive_supported_and_separated_by_part():
    xyz=np.array([[0,0,0],[1,0,0],[100,0,0],[0,0,10],[200,0,0]],float)
    score=np.array([.9,.85,.8,.7,np.nan])
    part=np.array([0,0,0,1,0])
    assert spatial_candidates(score,xyz,part)==[0,2,3]
    assert spatial_candidates(np.full(5,-.2),xyz,part)==[]


def test_cochanging_patches_are_marked_indistinguishable():
    values=np.array([0.,0.,1.,2.,4.,5.])
    x=np.column_stack([values,values*3,values,np.ones(6)])
    group,size,count=identical_rank_profiles(x,np.zeros(6),np.array([True,True,False,True]))
    assert count==1
    np.testing.assert_array_equal(group,[0,0,-1,-1])
    np.testing.assert_array_equal(size,[2,2,0,0])
