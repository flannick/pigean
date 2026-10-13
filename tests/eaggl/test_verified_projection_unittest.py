import json
from pathlib import Path
import sys
from types import SimpleNamespace
import numpy as np
import pytest
from scipy import sparse
from scipy.optimize import minimize

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src'))
from eaggl.verified_projection import project
from eaggl.projection_context import save_context,restore_context,SETTINGS


@pytest.mark.parametrize('cap,mass',[(None,None),(1.,None),(.35,None),(None,.4),(1.,.7),(.3,1.5)])
@pytest.mark.parametrize('weighted',[False,True])
def test_independent_convex_reference(cap,mass,weighted):
    rng=np.random.default_rng(45)
    W=rng.uniform(0,.5,(15,4));V=rng.uniform(0,1,(15,5));la=np.array([.5,.8,1.,1.5]);phi=.03
    left=rng.uniform(.1,1,(15,2)) if weighted else None
    right=rng.uniform(.1,1,(5,2)) if weighted else None
    H,diagnostic=project(W,sparse.csc_matrix(V),left=left,right=right,phi=phi,lambdas=la,cap=cap,mass=mass)
    S=np.ones_like(V) if left is None else left@right.T+1e-10
    for j,h in enumerate(H):
        def loss(x):
            return .5*np.sum(S[:,j]*(V[:,j]-W@x)**2)+.5*np.sum(phi/la*x*x)+1e-10*x.sum()
        def gradient(x):
            return W.T@(S[:,j]*(W@x-V[:,j]))+phi/la*x+1e-10
        constraints=[] if mass is None else [dict(type='ineq',fun=lambda x:mass-x.sum(),jac=lambda x:-np.ones_like(x))]
        reference=minimize(loss,np.zeros(4),jac=gradient,method='SLSQP',bounds=[(0,cap)]*4,
                           constraints=constraints,options=dict(ftol=1e-13,maxiter=5000))
        assert reference.success
        assert abs(loss(h)-reference.fun)<1e-8
        assert np.min(h)>=0
        if cap is not None:
            assert h.max()<=cap+1e-12
        if mass is not None:
            assert h.sum()<=mass+1e-12
    assert diagnostic['converged'],diagnostic


def test_degenerate_zero_and_duplicate_bases():
    W=np.array([[1.,1.,0.],[.5,.5,0.],[0.,0.,0.]])
    V=np.array([[1.,0.],[.5,0.],[0.,0.]])
    H,d=project(W,V,cap=1.)
    assert d['converged']
    np.testing.assert_allclose(W@H.T,V,atol=1e-8)
    np.testing.assert_array_equal(H[1],0.)
    H,d=project(np.zeros((3,2)),np.ones((3,2)),cap=1.)
    np.testing.assert_array_equal(H,0.);assert d['converged']


@pytest.mark.parametrize('bad',[dict(phi=-1),dict(phi=float('nan')),dict(cap=0),dict(mass=-1),dict(tol=0),dict(tol=float('nan')),dict(tol=float('inf')),dict(lambdas=[0,1])])
def test_bad_solver_settings(bad):
    with pytest.raises(ValueError):
        project(np.eye(2),np.eye(2),**bad)


def options():
    return SimpleNamespace(project_phenos_from_gene_sets=False,factor_output_scope='all',cluster_row_min_max_loading=0,
        trait_linkage_source='combined',trait_linkage_threshold=1.,trait_linkage_computation_mode='sparse_full',
        pheno_capture_input='weighted_thresholded',trait_factor_linkage_factor_gene_threshold=.05,
        trait_factor_linkage_nnls_min_loading=0.,trait_factor_linkage_nnls_max_value=1.)


def state():
    return SimpleNamespace(genes=['A','B','C'],exp_gene_factors=np.array([[.1234567,.2],[.3,.456789],[.5,.1]]),
        gene_prob_factor_vector=np.array([[.2],[.5],[.7]]),gene_factor_gene_mask=np.array([True,False,True]),
        supplied_factor_ids=(2,5),background_prior=.05,internal_last_projection_phi=.01,
        internal_last_projection_tol=1e-6,internal_projection_method='verified_quadratic',
        internal_projection_tol=1e-8,internal_projection_max_cycles=2000,_record_params=lambda *a,**k:None)


def test_context_restores_exact_basis_weights_and_order(tmp_path):
    original=state();path=str(tmp_path/'context.npz');save_context(original,path,options())
    loaded=state();loaded.genes=['C','A','B'];ix=[2,0,1]
    loaded.exp_gene_factors=np.round(original.exp_gene_factors[ix],4);loaded.gene_prob_factor_vector=None
    restore_context(loaded,path,options())
    np.testing.assert_array_equal(loaded.exp_gene_factors,original.exp_gene_factors[ix])
    np.testing.assert_array_equal(loaded.gene_prob_factor_vector,original.gene_prob_factor_vector[ix])
    np.testing.assert_array_equal(loaded.gene_factor_gene_mask,original.gene_factor_gene_mask[ix])
    assert loaded.internal_loaded_projection_phi==.01


@pytest.mark.parametrize('change',['ids','genes','basis','prior','solver','threshold'])
def test_context_rejects_incompatible_supplied_run(tmp_path,change):
    path=str(tmp_path/'context.npz');save_context(state(),path,options())
    loaded=state();opts=options()
    if change=='ids':loaded.supplied_factor_ids=(1,2)
    if change=='genes':loaded.genes=['A','B','D']
    if change=='basis':loaded.exp_gene_factors[0,0]+=.1
    if change=='prior':loaded.background_prior=.1
    if change=='solver':loaded.internal_projection_method='legacy'
    if change=='threshold':opts.trait_linkage_threshold=2.
    with pytest.raises(ValueError):restore_context(loaded,path,opts)


def test_context_rejects_modified_arrays(tmp_path):
    path=tmp_path/'context.npz';save_context(state(),str(path),options())
    with np.load(path,allow_pickle=False) as a:arrays={name:a[name] for name in a.files}
    arrays['weights']=arrays['weights']+.2
    np.savez_compressed(path,**arrays)
    with pytest.raises(ValueError,match='integrity'):restore_context(state(),str(path),options())
