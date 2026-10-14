"""Independent compatible/counterexample and production-conditional checks."""
from pathlib import Path
import sys
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'research'))
from conditional_compatibility import audit_slab_precision


def test_heterogeneous_variances_require_weighted_symmetry():
    V=np.array([[1.,.5],[.5,1.]])
    r=audit_slab_precision(V,np.array([.5,.2]),np.array([.5,.8]))
    assert not r['symmetric']
    assert r['witness']['q_ij']==.5
    assert r['witness']['q_ji']==.125
    V[1,0]=.125
    assert audit_slab_precision(V,np.array([.5,.2]),np.array([.5,.8]))['positive_definite']


def test_symmetry_does_not_imply_proper_density():
    r=audit_slab_precision(np.array([[1.,3.],[3.,1.]]),np.ones(2),np.ones(2))
    assert r['symmetric'] and not r['positive_definite']


@pytest.mark.parametrize('variance', [np.array([0.,1.]),np.array([np.nan,1.])])
def test_invalid_inputs_rejected(variance):
    with pytest.raises(ValueError):audit_slab_precision(np.eye(2),np.ones(2),variance)


def test_precision_matches_derivative_of_actual_production_slab_means():
    from pigean.state import _update_inner_beta_gene_set_batch as update
    V=np.array([[1.,.5],[.5,1.]])
    se2=np.array([[1.,4.]]);slab=np.ones((1,2));shrink=slab/(slab+se2)
    variance=shrink*se2
    def mean(j,other):
        sample=np.zeros((1,1,2));sample[0,0,1-j]=other
        pm=np.zeros_like(sample);prob=np.zeros_like(sample)
        mask=np.zeros((1,2),dtype=bool);mask[0,j]=True
        update(compute_mask_m=mask,compute_mask_v=np.array([True]),alpha_shrink=1.,
            rand_ps_t=np.ones_like(sample),rand_norms_t=np.zeros_like(sample),
            hdmp_hdmpn_m=shrink,c_const_m=np.ones((1,2)),d_const_m=np.ones((1,2)),
            hdmpn_m=slab+se2,se2s_m=se2,norm_scale_m=np.sqrt(variance),
            assume_independent=False,beta_tildes_m=np.array([[.1,.2]]),sampled_beta_t=sample,
            V=V,multiple_V=False,sparse_V=False,use_X=False,X_orig=None,
            scale_factors_m=np.ones((1,2)),mean_shifts_m=np.zeros((1,2)),
            betas_trace_out=None,betas_trace_gene_sets=None,account_for_V_diag_m=False,
            V_diag_m=np.ones((1,2)),posterior_nonzero_prob_t=prob,
            posterior_beta_mean_t=pm,gauss_seidel=False,residual_beta_tilde_t=np.zeros_like(sample))
        return pm[0,0,j]/prob[0,0,j]
    q=[-(mean(j,1e-5)-mean(j,-1e-5))/(2e-5*variance[0,j]) for j in range(2)]
    np.testing.assert_allclose(q,[.5,.125],rtol=1e-9)
    assert not audit_slab_precision(V,shrink[0],variance[0])['symmetric']
