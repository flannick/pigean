"""Direct quadratic forms versus the dense influence reference and sparse path."""
from pathlib import Path
import sys
import pickle

import numpy as np
import pytest
from scipy import sparse
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from pegs_shared.covariance import marginal_inflation, AnnotationCorrelation
from pegs_shared.regression import compute_beta_tildes
from pegs_shared.sparse_regression import marginal_backend


@pytest.mark.parametrize('parallel',[False,True])
@pytest.mark.parametrize('pseudo',[False,True])
@pytest.mark.parametrize('kind',['ordinary','cancellation','identity'])
def test_direct_quadratic_full_fit_and_rng(parallel,pseudo,kind):
    rng=np.random.default_rng(152)
    n,k=70,31
    X=rng.normal(size=(n,k))*(rng.random((n,k))<.17)
    X[:,0]=0;X[:,1]=1
    F=rng.normal(size=(n,4));F/=np.linalg.norm(F,axis=1)[:,None]
    R=(.4*(F@F.T)+.6*np.eye(n) if kind=='ordinary' else
       (1-1e-12)*np.ones((n,n))+1e-12*np.eye(n) if kind=='cancellation' else np.eye(n))
    r=sparse.csr_matrix(R)
    xp=sparse.csc_matrix(np.vstack([X,rng.normal(size=k)]) if pseudo else X)
    # Near cancellation, preserve the previous sparse fallback decisions.
    reference=marginal_inflation(xp,r,pseudo=pseudo,block_size=7,
                                backend='sparse' if kind=='cancellation' else 'legacy')
    features=['compiled-prefilter']+(['parallel-prefilter'] if parallel else [])
    before=pickle.dumps(np.random.get_state())
    with marginal_backend('sparse-regression',enable=features):
        result=marginal_inflation(xp,r,pseudo=pseudo,backend='sparse',block_size=7)
    np.testing.assert_allclose(result,reference,rtol=2e-10,atol=2e-12)
    assert pickle.dumps(np.random.get_state())==before
    y=rng.random((3,n));x=sparse.csc_matrix(X)
    with marginal_backend('sparse-regression'):
        a=compute_beta_tildes(x,y.copy(),resid_correlation_matrix=r)
    with marginal_backend('sparse-regression',enable=features):
        b=compute_beta_tildes(x,y.copy(),resid_correlation_matrix=r)
    for aa,bb in zip(a,b):
        if aa is not None:np.testing.assert_allclose(aa,bb,rtol=2e-10,atol=2e-12,equal_nan=True)


def test_logistic_and_operator_fallback():
    rng=np.random.default_rng(34)
    X=sparse.csc_matrix(rng.random((45,7))<.2,dtype=float)
    for R in (sparse.eye(45),AnnotationCorrelation(X,np.ones(7),np.ones(45))):
        kw=dict(beta=np.linspace(-30,30,7),alpha=np.zeros(7),backend='sparse')
        with marginal_backend('sparse-regression'):a=marginal_inflation(X,R,**kw)
        with marginal_backend('sparse-regression',enable=['compiled-prefilter']):b=marginal_inflation(X,R,**kw)
        np.testing.assert_array_equal(a,b)
