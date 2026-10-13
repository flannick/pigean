"""Validate complete HuGE output with all cap/phantom/absence modes."""
from pathlib import Path
import sys
from itertools import product

import numpy as np
import pytest
from scipy import sparse
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from pigean.state import PigeanState
from pegs_shared.sparse_regression import marginal_backend


@pytest.mark.parametrize('cap,scale,phantom,absence',list(product([False,True],repeat=4)))
@pytest.mark.parametrize('extreme',[False,True])
def test_distillation_equivalence(cap,scale,phantom,absence,extreme):
    rng=np.random.default_rng(719)
    genes=[f'g{i}' for i in range(60)]
    state=PigeanState(background_prior=.05,batch_size=10)
    state.genes=genes
    bf=rng.exponential(size=(60,13))*(rng.random((60,13))<.1)
    bf[:,0]=0  # Empty signal.
    bf[1,1]=1e5;bf[1:,2]=0;bf[0,2]=3  # Single-candidate signal.
    bf=sparse.csc_matrix(bf)
    # Stored zero participates in the reference's sparse pattern too.
    bf.data[3]=0
    priors=rng.normal(size=(3,60))*(35 if extreme else 2)
    kw=dict(huge_signal_bfs=bf,huge_signal_posteriors=rng.uniform(.1,.8,13),
        huge_signal_sum_gene_cond_probabilities=rng.uniform(.1,.9,13),
        huge_signal_mean_gene_pos=np.full(13,3.),huge_signal_max_closest_gene_prob=.9,
        cap_region_posterior=cap,scale_region_posterior=scale,phantom_region_posterior=phantom,
        allow_evidence_of_absence=absence,gene_covariates=None,gene_covariates_mask=None,
        gene_covariates_mat_inv=None,gene_covariate_names=None,gene_covariate_intercept_index=None,
        gene_prob_genes=genes,rel_prior_log_bf=priors)
    with marginal_backend('legacy'):old=state._distill_huge_signal_bfs(**kw)
    with marginal_backend('legacy',enable=['huge-prior-softmax']):new=state._distill_huge_signal_bfs(**kw)
    for a,b in zip(old[:2],new[:2]):np.testing.assert_allclose(a,b,rtol=1e-10,atol=1e-11,equal_nan=True)
    assert old[2:]==new[2:]
