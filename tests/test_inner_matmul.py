"""Dense BLAS dispatch preserves conditional definitions and consumes no RNG."""
import copy
import pickle
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from pigean.state import _update_inner_beta_gene_set_batch
from pegs_shared.sparse_regression import marginal_backend


@pytest.mark.parametrize('features', [['inner-matmul'], ['inner-fused'], ['inner-matmul', 'inner-fused']])
@pytest.mark.parametrize('gauss_seidel', [False, True])
@pytest.mark.parametrize('active', [[True, True, True], [True, False, True], [False]*3])
def test_dense_updates_and_rng(gauss_seidel, active, features):
    rng = np.random.default_rng(729)
    c,p,g = 4,3,31
    v = rng.normal(size=(g,g)); v = v.T@v/g
    common = dict(compute_mask_v=np.array(active),
        compute_mask_m=(rng.random((p,g))>.5)&np.array(active)[:,None],
        alpha_shrink=1.,rand_ps_t=rng.random((c,p,g)),rand_norms_t=rng.normal(size=(c,p,g)),
        hdmp_hdmpn_m=rng.random((p,g)),c_const_m=np.full((p,g),.4),
        d_const_m=np.full((p,g),.8),hdmpn_m=np.full((p,g),.6),se2s_m=np.full((p,g),.3),
        norm_scale_m=np.full((p,g),.2),assume_independent=False,beta_tildes_m=rng.normal(size=(p,g)),
        sampled_beta_t=rng.normal(size=(c,p,g)),V=v,multiple_V=False,sparse_V=False,
        use_X=False,X_orig=None,scale_factors_m=np.ones((p,g)),mean_shifts_m=np.zeros((p,g)),
        betas_trace_out=None,betas_trace_gene_sets=None,account_for_V_diag_m=True,
        V_diag_m=np.tile(v.diagonal(),(p,1)),posterior_nonzero_prob_t=np.ones((c,p,g)),
        posterior_beta_mean_t=np.zeros((c,p,g)),gauss_seidel=gauss_seidel,
        residual_beta_tilde_t=np.zeros((c,p,g)))
    before = pickle.dumps(np.random.get_state())
    a,b = copy.deepcopy(common),copy.deepcopy(common)
    for _ in range(5):
        with marginal_backend('legacy'):
            _update_inner_beta_gene_set_batch(**a)
        with marginal_backend('legacy',enable=features):
            _update_inner_beta_gene_set_batch(**b)
        for key in ('sampled_beta_t','posterior_nonzero_prob_t','posterior_beta_mean_t','residual_beta_tilde_t'):
            np.testing.assert_allclose(a[key],b[key],rtol=1e-12,atol=1e-12)
    assert pickle.dumps(np.random.get_state())==before
