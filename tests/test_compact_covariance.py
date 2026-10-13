"""Exact edge preparation with signed weights, pseudocounts, and changing R."""
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from scipy import sparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from pegs_shared.covariance import marginal_inflation, AnnotationCorrelation
from pegs_shared.covariance_edges import compact_edges
from pegs_shared.grouped_regression import GroupedDesign
from pegs_shared.regression import compute_logistic_beta_tildes
from pegs_shared.sparse_regression import marginal_backend, use_compact_covariance


def example():
    rng = np.random.default_rng(381)
    x = rng.choice([0., .2, .7, 1.], size=(80, 6))
    x[:, 0] = 0
    x[:, 1] = rng.normal(size=80)  # Continuous: bounded unaggregated path.
    factor = rng.normal(size=(80, 4))
    factor /= np.sqrt(np.sum(factor**2, axis=1))[:, None]
    r = .3*factor@factor.T + .7*np.eye(80)
    return rng, sparse.csc_matrix(x), sparse.csr_matrix(r)


@pytest.mark.parametrize('pseudo', [False, True])
@pytest.mark.parametrize('logistic', [False, True])
def test_compact_inflation_matches_parent_and_dense(pseudo, logistic):
    rng, x, r = example()
    if pseudo:
        x = sparse.vstack([x, np.ones((1, x.shape[1]))], format='csc')
    # Force unsorted indices. Direct lookup must not assume sorted R rows.
    for i in range(r.shape[0]):
        start, end = r.indptr[i:i+2]
        r.indices[start:end] = r.indices[start:end][::-1]
        r.data[start:end] = r.data[start:end][::-1]
    r.has_sorted_indices = False
    with marginal_backend('compact-covariance') as cache:
        actual = GroupedDesign(x)
        for matrix in (r, sparse.eye(80, format='csr'), r):
            kw = dict(pseudo=pseudo)
            if logistic:
                kw.update(beta=rng.normal(size=x.shape[1]), alpha=rng.normal(size=x.shape[1]))
            a = actual.inflation(matrix, **kw)
            np.testing.assert_allclose(a, marginal_inflation(x, matrix, **kw), rtol=1e-9, atol=1e-11)
            with marginal_backend('grouped-cached'):
                b = GroupedDesign(x).inflation(matrix, **kw)
            np.testing.assert_allclose(a, b, rtol=1e-9, atol=1e-11)
        assert cache.covariance_stats['preparations'] == 2
        assert cache.covariance_stats['stored_pairs'] < cache.covariance_stats['input_edges']
        assert cache.covariance_stats['aggregated_columns'] > 0


def test_signed_pair_reduction_preserves_quadratic_and_absolute_scale():
    rng, x, r = example()
    parent = GroupedDesign(x)._prepare_correlation(r, False)
    old_l, old_r, old_w = parent[3:6]
    group = GroupedDesign(x)
    (l, rr, w, aw), stats = compact_edges(group.X, group.groups, r)
    delta = rng.normal(size=len(group.values))
    for old, new in [(old_w*delta[old_l]*delta[old_r], w*delta[l]*delta[rr]),
                     (np.abs(old_w*delta[old_l]*delta[old_r]), aw*np.abs(delta[l])*np.abs(delta[rr]))]:
        np.testing.assert_allclose(
            np.bincount(group.columns[l], weights=new, minlength=group.k),
            np.bincount(group.columns[old_l], weights=old, minlength=group.k), rtol=1e-12, atol=1e-12)
    assert np.any(aw > np.abs(w)+1e-6)  # Cancellation exists within a group pair.
    assert stats['stored_pairs'] < stats['input_edges']


def test_cancellation_still_uses_reference_fallback(monkeypatch):
    import pegs_shared.covariance as covariance
    x = sparse.csc_matrix(np.tile([0., 1.], 40)[:, None])
    r = sparse.csr_matrix((1-1e-12)*np.ones((80, 80)) + 1e-12*np.eye(80))
    original = covariance.marginal_inflation
    calls = []
    def record(*args, **kw):
        calls.append(True)
        return original(*args, **kw)
    monkeypatch.setattr(covariance, 'marginal_inflation', record)
    with marginal_backend('compact-covariance'):
        actual = GroupedDesign(x).inflation(r)
    assert calls
    np.testing.assert_allclose(actual, original(x, r), rtol=1e-12)


def test_new_design_rebuilds_mutated_correlation_and_selected_columns():
    rng, x, r = example()
    with marginal_backend('compact-covariance') as cache:
        for selected in ([5, 1, 2], [2, 4], [0, 3]):
            xx = x[:, selected]
            a = GroupedDesign(xx).inflation(r)
            np.testing.assert_allclose(a, marginal_inflation(xx, r), rtol=1e-9, atol=1e-11)
            # Keep identity/shape but alter off-diagonal weights each regression.
            coo = r.tocoo()
            coo.data[coo.row != coo.col] *= .5
            r.data[:] = coo.tocsr().data
        assert cache.covariance_stats['preparations'] == 3


@pytest.mark.parametrize('pseudo', [False, True])
def test_full_logistic_fit_matches_parent_and_legacy(pseudo):
    rng, x, r = example()
    y = np.array([rng.random(80)<.3, rng.random(80)<.45, np.zeros(80)]).astype(float)
    kw = dict(resid_correlation_matrix=[r, sparse.eye(80, format='csr'), r], append_pseudo=pseudo)
    results = []
    for engine in ['legacy', 'grouped-cached', 'compact-covariance']:
        with marginal_backend(engine):
            results.append(compute_logistic_beta_tildes(x, y.copy(), **kw))
    for ref in results[:2]:
        for a, b in zip(results[2], ref):
            if a is None or b is None:
                assert a is b
            else:
                np.testing.assert_allclose(a, b, rtol=2e-8, atol=2e-10)


def test_operator_fallback_invalid_shape_and_backend_scope(monkeypatch):
    from pigean import pipeline
    _, x, _ = example()
    op = AnnotationCorrelation(x, np.ones(x.shape[1]), np.ones(x.shape[0]))
    with marginal_backend('compact-covariance'):
        a = GroupedDesign(x)
        np.testing.assert_allclose(a.inflation(op), marginal_inflation(x, op))
        with pytest.raises(ValueError):
            a.inflation(sparse.eye(79))
    config = SimpleNamespace(run_kwargs={}, num_chains=10, total_num_iter=500)
    monkeypatch.setattr(pipeline, 'build_main_gibbs_stage_config', lambda opts: config)
    seen = []
    state = SimpleNamespace(run_gibbs=lambda: seen.append(use_compact_covariance()))
    for engine in ['compact-covariance', 'cached-grouped-regression', 'legacy']:
        pipeline.run_main_gibbs_stage(None, state, SimpleNamespace(gibbs_engine=engine), {'run_gibbs': True})
    assert seen == [True, False, False]
    assert not use_compact_covariance()
