"""Check grouped calculations against ungrouped legacy regression formulas."""
import sys
from pathlib import Path
import numpy as np
import pytest
from scipy import sparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from pegs_shared.covariance import marginal_inflation, AnnotationCorrelation
from pegs_shared.grouped_regression import GroupedDesign
from pegs_shared.regression import compute_logistic_beta_tildes, compute_beta_tildes
from pegs_shared.sparse_regression import marginal_backend, logistic_moments


def example():
    rng = np.random.default_rng(62)
    x = rng.choice([0., 0., 0., .25, .6, 1.], size=(180, 12))
    x[:, 0] = 0
    x[:, 1] = 1
    x[:, 2] = rng.normal(size=180)  # No grouping opportunity, still exact.
    r = .4*(np.arange(180)[:, None]//5 == np.arange(180)[None, :]//5)+.6*np.eye(180)
    return rng, sparse.csc_matrix(x), sparse.csr_matrix(r)


@pytest.mark.parametrize('pseudo', [False, True])
@pytest.mark.parametrize('logistic', [False, True])
def test_influence_matches_dense_reference_and_distinct_correlations(pseudo, logistic):
    rng, x, r = example()
    if pseudo:
        x = sparse.vstack([x, np.ones((1, x.shape[1]))], format='csc')
    group = GroupedDesign(x)
    for matrix in (r, sparse.eye(180, format='csr'), r):
        kw = dict(pseudo=pseudo)
        if logistic:
            kw.update(beta=rng.normal(size=x.shape[1]), alpha=rng.normal(size=x.shape[1]))
        a = marginal_inflation(x, matrix, **kw)
        b = group.inflation(matrix, **kw)
        np.testing.assert_allclose(b, a, rtol=1e-9, atol=1e-11)


def test_moments_follow_partial_chain_masks():
    rng, x, _ = example()
    grouped = GroupedDesign(x, 3)
    stacked = sparse.hstack([x]*3, format='csc')
    for mask in [np.ones(36, bool), rng.random(36)<.5, np.zeros(36, bool)]:
        beta, alpha = rng.normal(size=(2, mask.sum()))
        a = logistic_moments(stacked[:,mask], beta, alpha, .99)
        b = grouped.moments(beta, alpha, .99, mask)
        np.testing.assert_allclose(b, a, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize('pseudo', [False, True])
def test_full_fit_and_linear_parity(pseudo):
    rng, x, r = example()
    y = np.array([rng.random(180)<.25, rng.random(180)<.4, np.zeros(180)]).astype(float)
    rs = [r, sparse.eye(180, format='csr'), r]
    for fn, kw in [(compute_logistic_beta_tildes, {'append_pseudo':pseudo}), (compute_beta_tildes, {})]:
        a = fn(x, y.copy(), resid_correlation_matrix=rs, **kw)
        with marginal_backend('grouped'):
            b = fn(x, y.copy(), resid_correlation_matrix=rs, **kw)
        for aa, bb in zip(a,b):
            if aa is None:
                assert bb is None
            else:
                np.testing.assert_allclose(bb, aa, rtol=2e-8, atol=2e-10)


def test_operator_fallback_and_validation():
    _, x, _ = example()
    op = AnnotationCorrelation(x, np.ones(x.shape[1]), np.ones(x.shape[0]))
    grouped = GroupedDesign(x)
    np.testing.assert_allclose(grouped.inflation(op), marginal_inflation(x, op))
    with pytest.raises(ValueError):
        grouped.inflation(sparse.eye(x.shape[0]-1))


def test_unsorted_sparse_negative_correlation_entries():
    rng, x, _ = example()
    r = sparse.diags([np.full(179, -.2), np.ones(180), np.full(179, -.2)], [-1,0,1], format='csr')
    for row in range(180):
        start, end = r.indptr[row:row+2]
        r.indices[start:end] = r.indices[start:end][::-1]
        r.data[start:end] = r.data[start:end][::-1]
    r.has_sorted_indices = False
    kw = dict(beta=rng.normal(size=x.shape[1]), alpha=rng.normal(size=x.shape[1]))
    np.testing.assert_allclose(GroupedDesign(x).inflation(r, **kw),
                               marginal_inflation(x, r, **kw), rtol=1e-9, atol=1e-11)


def test_pipeline_selects_grouped_backend(monkeypatch):
    from types import SimpleNamespace
    from pigean import pipeline
    from pegs_shared.sparse_regression import use_grouped_backend
    config = SimpleNamespace(run_kwargs={}, num_chains=10, total_num_iter=500)
    monkeypatch.setattr(pipeline, 'build_main_gibbs_stage_config', lambda opts: config)
    values=[]
    state = SimpleNamespace(run_gibbs=lambda: values.append(use_grouped_backend()))
    for engine in ['grouped-regression', 'sparse-regression', 'legacy']:
        pipeline.run_main_gibbs_stage(SimpleNamespace(log=lambda *args: None),state,SimpleNamespace(gibbs_engine=engine),{'run_gibbs':True})
    assert values == [True, False, False]
    assert not use_grouped_backend()
