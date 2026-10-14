"""Verify compiled reduction order and complete grouped logistic fits."""
import numpy as np
import pytest
from scipy import sparse
from pegs_shared.compiled_moments import grouped_moment_sums
from pegs_shared.grouped_regression import GroupedDesign
from pegs_shared.regression import compute_logistic_beta_tildes
from pegs_shared.sparse_regression import marginal_backend
from pegs_shared.gibbs_features import resolve_features


@pytest.mark.parametrize('seed', [0,1,2])
@pytest.mark.parametrize('size,width', [(0,0),(0,7),(20,7),(20000,301)])
def test_sums_are_bitwise_equal_to_numpy(seed,size,width):
    rng=np.random.default_rng(seed)
    local=rng.integers(0,width,size=size);values=rng.normal(size=size)
    counts=rng.integers(1,1000,size=size).astype(float)
    p=rng.random(size);v=p*(1-p)
    expected=tuple(np.bincount(local,weights=counts*w,minlength=width)
                   for w in (p,v,values*p,values*v,values*values*v))
    actual=grouped_moment_sums(local,values,counts,p,v,width)
    for a,b in zip(actual,expected):np.testing.assert_array_equal(a,b)


@pytest.mark.parametrize('mask_kind', ['none','partial','empty'])
def test_real_grouped_masks_and_extreme_probabilities(mask_kind):
    rng=np.random.default_rng(6)
    X=sparse.csc_matrix(rng.choice([0.,0.,.2,1.,-3.],size=(100,8)))
    design=GroupedDesign(X,3)
    mask=None if mask_kind=='none' else (rng.random(24)>.5 if mask_kind=='partial' else np.zeros(24,bool))
    size=24 if mask is None else mask.sum()
    beta,alpha=rng.normal(size=(2,size))*100
    with marginal_backend('grouped'):
        expected=design.moments(beta,alpha,.99,mask)
    with marginal_backend('grouped',enable=['compiled-moments']):
        actual=design.moments(beta,alpha,.99,mask)
    for a,b in zip(actual,expected):np.testing.assert_array_equal(a,b)


@pytest.mark.parametrize('pseudo',[False,True])
def test_full_regression_including_covariance_is_exact(pseudo):
    rng=np.random.default_rng(49)
    X=sparse.csc_matrix(rng.choice([0.,0.,.2,.5,1.],size=(180,12)))
    y=rng.integers(0,2,size=(3,180)).astype(float)
    R=sparse.diags([np.full(179,.1),np.ones(180),np.full(179,.1)],[-1,0,1],format='csr')
    kw=dict(append_pseudo=pseudo,resid_correlation_matrix=[R]*3)
    with marginal_backend('grouped'):a=compute_logistic_beta_tildes(X,y.copy(),**kw)
    with marginal_backend('grouped',enable=['compiled-moments']):b=compute_logistic_beta_tildes(X,y.copy(),**kw)
    for x,z in zip(a,b):
        if x is None:assert z is None
        else:np.testing.assert_array_equal(x,z)


def test_missing_prerequisite_is_reported_not_enabled():
    chosen=resolve_features('legacy',enable=['compiled-moments'])
    assert 'compiled-moments' not in chosen['active']
    assert chosen['inactive']['compiled-moments']==['grouped-regression']
