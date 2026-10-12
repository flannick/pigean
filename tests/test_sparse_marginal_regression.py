"""Numerical equivalence to the independent, established dense formulas."""
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy import sparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from pegs_shared.covariance import AnnotationCorrelation, marginal_inflation
from pegs_shared.regression import compute_beta_tildes, compute_logistic_beta_tildes
from pegs_shared.sparse_regression import marginal_backend, use_sparse_backend


@pytest.mark.parametrize('pseudo', [False, True])
@pytest.mark.parametrize('logistic', [False, True])
def test_covariance_matches_dense_influence_formula(pseudo, logistic):
    rng = np.random.default_rng(912)
    x = rng.normal(size=(140, 19)) * (rng.random((140, 19)) < .12)
    x[:, 0] = 0
    x[:, 1] = 1  # Exercises cancellation/zero-variance fallback.
    r = sparse.eye(140, format='lil')
    for i in range(0, 140, 2):
        r[i, i+1] = r[i+1, i] = -.35 if i % 4 else .7
    if pseudo:
        x = np.vstack([x, np.ones(19)])
    kw = dict(pseudo=pseudo, block_size=7)
    if logistic:
        kw.update(beta=rng.normal(size=19)*3, alpha=rng.normal(size=19)*2)
    a = marginal_inflation(sparse.csc_matrix(x), r, **kw)
    b = marginal_inflation(sparse.csc_matrix(x), r, backend='sparse', **kw)
    np.testing.assert_allclose(b, a, rtol=2e-10, atol=2e-12)


@pytest.mark.parametrize('pseudo', [False, True])
@pytest.mark.parametrize('binary', [False, True])
def test_full_multi_trait_regression_parity(pseudo, binary):
    rng = np.random.default_rng(412)
    x = (rng.random((180, 12)) < .15).astype(float)
    if not binary:
        x *= rng.uniform(.1, 2, x.shape)
    y = np.array([rng.random(180)<.2, rng.random(180)<.4, np.zeros(180)]).astype(float)
    r = .3*(np.arange(180)[:, None]//6 == np.arange(180)[None, :]//6)+.7*np.eye(180)
    rs = [sparse.csr_matrix(r), sparse.eye(180), sparse.eye(180)]
    kw = dict(append_pseudo=pseudo, resid_correlation_matrix=rs, rel_tol=1e-8)
    a = compute_logistic_beta_tildes(sparse.csc_matrix(x), y.copy(), **kw)
    with marginal_backend('sparse'):
        b = compute_logistic_beta_tildes(sparse.csc_matrix(x), y.copy(), **kw)
    for aa, bb in zip(a, b):
        if aa is None:
            assert bb is None
        else:
            np.testing.assert_allclose(bb, aa, rtol=2e-8, atol=2e-10)
    a = compute_beta_tildes(sparse.csc_matrix(x), y[:2], resid_correlation_matrix=rs[:2])
    with marginal_backend('sparse'):
        b = compute_beta_tildes(sparse.csc_matrix(x), y[:2], resid_correlation_matrix=rs[:2])
    for aa, bb in zip(a, b):
        if aa is not None:
            np.testing.assert_allclose(bb, aa, rtol=2e-10, atol=2e-12)


def test_operator_fallback_and_extreme_coefficients():
    rng = np.random.default_rng(12)
    x = sparse.csc_matrix(rng.random((80, 5)) < .2, dtype=float)
    op = AnnotationCorrelation(x, np.ones(5), np.ones(80))
    kw = dict(beta=np.array([-150, -40, 0, 40, 150]), alpha=np.array([0, 10, -10, 10, 0]))
    for r in [op, sparse.eye(80)]:
        a = marginal_inflation(x, r, **kw)
        b = marginal_inflation(x, r, backend='sparse', **kw)
        np.testing.assert_allclose(b, a, rtol=2e-10, atol=2e-12)


def test_backend_context_is_scoped_and_validated():
    assert not use_sparse_backend()
    with pytest.raises(RuntimeError):
        with marginal_backend('sparse'):
            assert use_sparse_backend()
            raise RuntimeError('test cleanup')
    assert not use_sparse_backend()
    with pytest.raises(ValueError):
        with marginal_backend('unknown'):
            pass


def test_cli_dispatch_enables_only_requested_backend(monkeypatch):
    from types import SimpleNamespace
    from pigean import pipeline
    config = SimpleNamespace(run_kwargs={'sentinel': 4}, num_chains=10, total_num_iter=500)
    monkeypatch.setattr(pipeline, 'build_main_gibbs_stage_config', lambda options: config)
    seen = []
    state = SimpleNamespace(run_gibbs=lambda **kwargs: seen.append((use_sparse_backend(), kwargs)))
    for engine in ('sparse-regression', 'legacy'):
        pipeline.run_main_gibbs_stage(None, state, SimpleNamespace(gibbs_engine=engine), {'run_gibbs': True})
        assert not use_sparse_backend()
    assert seen == [(True, {'sentinel': 4}), (False, {'sentinel': 4})]
