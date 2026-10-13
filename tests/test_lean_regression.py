"""No unused chain stacks; preserve batching, masks, and numerical outputs."""
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from scipy import sparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from pegs_shared.regression import compute_logistic_beta_tildes
from pegs_shared.sparse_regression import marginal_backend, use_lean_regression


def example():
    rng = np.random.default_rng(381)
    x = rng.choice([0., .2, .7, 1.], size=(80, 6))
    x[:, 0] = 0
    x[:, 1] = rng.normal(size=80)
    factor = rng.normal(size=(80, 4))
    factor /= np.sqrt(np.sum(factor**2, axis=1))[:, None]
    r = .3*factor@factor.T + .7*np.eye(80)
    return rng, sparse.csc_matrix(x), sparse.csr_matrix(r)


@pytest.mark.parametrize('pseudo', [False, True])
@pytest.mark.parametrize('chains', [1, 3])
def test_no_stack_fit_matches_parent_exactly(monkeypatch, pseudo, chains):
    rng, x, r = example()
    y = (rng.random((chains, x.shape[0])) < .3).astype(float)
    if chains > 1:
        y[-1] = 0  # Recursive removal/reinsertion of a degenerate chain.
    else:
        y = y[0]
    for supplied_stack in (None, sparse.hstack([x]*chains, format='csc')):
        kw = dict(resid_correlation_matrix=r, append_pseudo=pseudo, X_stacked=supplied_stack)
        with marginal_backend('compact-covariance'):
            expected = compute_logistic_beta_tildes(x, y.copy(), **kw)
        with monkeypatch.context() as patch:
            def forbidden(*args, **kwargs):
                raise AssertionError('Lean regression allocated a chain annotation stack')
            patch.setattr(sparse, 'hstack', forbidden)
            with marginal_backend('lean-regression'):
                actual = compute_logistic_beta_tildes(x, y.copy(), **kw)
        for a, b in zip(actual, expected):
            if a is None or b is None:
                assert a is b
            else:
                np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize('chains,max_mb', [(1, 100), (5, 100), (5, 2.5), (5, .5)])
def test_epoch_stack_budget_preserves_batch_boundaries(monkeypatch, chains, max_mb):
    import pigean.state as state_module
    _, x, _ = example()
    state = SimpleNamespace(X_orig=x, Y_for_regression=np.zeros(x.shape[0]),
                            genes_missing=None, gene_sets_missing=None, _get_X_size_mb=lambda: 1.)
    monkeypatch.setattr(state_module, '_maybe_unsubset_gene_sets', lambda *a, **kw: None)
    with marginal_backend('compact-covariance'):
        old = state_module._initialize_gibbs_epoch_state(state, chains, x.shape[1], True, max_mb, lambda *a: None)
    def forbidden(*args, **kwargs):
        raise AssertionError('Lean epoch allocated a chain annotation stack')
    monkeypatch.setattr(sparse, 'hstack', forbidden)
    with marginal_backend('lean-regression'):
        new = state_module._initialize_gibbs_epoch_state(state, chains, x.shape[1], True, max_mb, lambda *a: None)
    assert new['X_hstacked'] is None
    assert new['stack_batch_size'] == old['stack_batch_size']
    assert new['num_stack_batches'] == old['num_stack_batches']
    for key in ('sum_betas_m', 'sum_Ys_m', 'sum_priors_m'):
        np.testing.assert_array_equal(new[key], old[key])


@pytest.mark.parametrize('begin,end', [(0, 2), (2, 4), (4, 5)])
def test_full_and_partial_chain_batches_match_parent(monkeypatch, begin, end):
    import pigean.state as state_module
    rng, x, r = example()
    dense = x.toarray()
    state = SimpleNamespace(X_orig=x, scale_factors=dense.std(axis=0), mean_shifts=dense.mean(axis=0),
                            _compute_logistic_beta_tildes=compute_logistic_beta_tildes)
    # State adapter passes scale and mean positionally, shared implementation keyword-only.
    state._compute_logistic_beta_tildes = lambda a,y,scale,mean,**kw: compute_logistic_beta_tildes(a,y,scale_factors=scale,mean_shifts=mean,**kw)
    mask = np.array([False, True, False, True, True, False])
    y = (rng.random((5, x.shape[0])) < .3).astype(float)
    kw = dict(state=state, pre_gene_set_filter_mask=mask, p_sample_m=y, begin=begin, end=end,
              y_corr_sparse=r, stack_batch_size=2, X_hstacked=sparse.hstack([x]*2, format='csc'))
    with marginal_backend('compact-covariance'):
        expected = state_module._compute_gibbs_logistic_beta_tildes_batch(**kw)
    def forbidden(*args, **kwargs):
        raise AssertionError('Lean batch allocated a chain annotation stack')
    monkeypatch.setattr(sparse, 'hstack', forbidden)
    kw['X_hstacked'] = None
    with marginal_backend('lean-regression'):
        actual = state_module._compute_gibbs_logistic_beta_tildes_batch(**kw)
    for a,b in zip(actual, expected):
        if a is None or b is None:
            assert a is b
        else:
            np.testing.assert_array_equal(a,b)


def test_pipeline_scopes_lean_engine(monkeypatch):
    from pigean import pipeline
    config = SimpleNamespace(run_kwargs={}, num_chains=10, total_num_iter=500)
    monkeypatch.setattr(pipeline, 'build_main_gibbs_stage_config', lambda opts: config)
    seen=[]
    state=SimpleNamespace(run_gibbs=lambda:seen.append(use_lean_regression()))
    for engine in ['lean-regression', 'compact-covariance', 'legacy']:
        pipeline.run_main_gibbs_stage(None,state,SimpleNamespace(gibbs_engine=engine),{'run_gibbs':True})
    assert seen == [True,False,False]
    assert not use_lean_regression()
