import copy
import random
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy import sparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from pigean.inner_cache import InnerCorrelationCache
from pigean.state import PigeanState, _update_inner_beta_gene_set_batch
from pegs_shared.sparse_regression import marginal_backend, use_cached_inner, use_lean_regression


@pytest.mark.parametrize('kind', ['dense', 'multiple', 'sparse'])
@pytest.mark.parametrize('gauss_seidel', [False, True])
@pytest.mark.parametrize('max_bytes', [0, 16 * 1024 * 1024])
def test_conditional_sequence_exact(kind, gauss_seidel, max_bytes):
    rng = np.random.RandomState(19)
    c, p, g = 4, 3, 7
    v = rng.normal(size=(g, g))
    v = v.T @ v / 30 + np.eye(g)
    V = np.stack([v, v * 1.1, v * .9]) if kind == 'multiple' else v
    if kind == 'sparse':
        V = sparse.csc_matrix(v)
    diag = np.diagonal(V, axis1=1, axis2=2) if kind == 'multiple' else np.tile(v.diagonal(), (p, 1))
    base = dict(alpha_shrink=.8, hdmp_hdmpn_m=rng.uniform(.1, .9, (p, g)),
                c_const_m=rng.uniform(.1, .9, (p, g)), d_const_m=np.ones((p, g)),
                hdmpn_m=np.ones((p, g)), se2s_m=np.full((p, g), .3),
                norm_scale_m=np.full((p, g), .2), assume_independent=False,
                beta_tildes_m=rng.normal(size=(p, g)), sampled_beta_t=rng.normal(size=(c, p, g)),
                V=V, multiple_V=kind == 'multiple', sparse_V=kind == 'sparse', use_X=False,
                X_orig=None, scale_factors_m=np.ones((p, g)), mean_shifts_m=np.zeros((p, g)),
                betas_trace_out=None, betas_trace_gene_sets=None, account_for_V_diag_m=True,
                V_diag_m=diag, posterior_nonzero_prob_t=np.ones((c, p, g)),
                posterior_beta_mean_t=np.zeros((c, p, g)), gauss_seidel=gauss_seidel,
                residual_beta_tilde_t=np.zeros((c, p, g)))
    old, new = copy.deepcopy(base), copy.deepcopy(base)
    cache = InnerCorrelationCache(new['V'], kind == 'multiple', kind == 'sparse', max_bytes=max_bytes)
    # Repeated batches, asynchronously stopped parallels, reactivation and an
    # empty selection; compare every conditional, not just the final summaries.
    batch = rng.random((p, g)) > .4
    for active in ([True, True, True], [True, True, True], [True, False, True],
                   [False, True, False], [False, False, False], [True, True, True]):
        active = np.array(active)
        common = dict(compute_mask_m=batch & active[:, None], compute_mask_v=active,
                      rand_ps_t=rng.random((c, p, g)), rand_norms_t=rng.normal(size=(c, p, g)))
        _update_inner_beta_gene_set_batch(**old, **common)
        _update_inner_beta_gene_set_batch(**new, **common, correlation_cache=cache)
        for field in ('sampled_beta_t', 'posterior_nonzero_prob_t',
                      'posterior_beta_mean_t', 'residual_beta_tilde_t'):
            np.testing.assert_array_equal(old[field], new[field])
    assert cache.payload_bytes <= max_bytes
    assert cache.hits == (2 if max_bytes else 0)


def test_cache_eviction_and_lifetime():
    V = np.arange(100.).reshape(10, 10)
    cache = InnerCorrelationCache(V, max_entries=1, max_bytes=500)
    active = np.array([True])
    a = np.array([[True] + [False] * 9])
    b = np.roll(a, 1, axis=1)
    original = cache.get(a, active)[2].copy()
    cache.get(b, active)
    assert len(cache.entries) == 1
    np.testing.assert_array_equal(cache.get(a, active)[2], original)
    assert cache.misses == 3 and cache.hits == 0
    assert cache.payload_bytes <= 500
    # A later fit must see its new V, not values retained from a prior fit.
    other = InnerCorrelationCache(V * 2)
    np.testing.assert_array_equal(other.get(a, active)[2], original * 2)


@pytest.mark.parametrize('extra_features', [['inner-cache64'], ['inner-matmul', 'inner-fused', 'inner-batching', 'inner-cache64']])
@pytest.mark.parametrize('kind', ['dense', 'sparse', 'multiple', 'X', 'independent'])
@pytest.mark.parametrize('hyper', [False, True])
def test_full_inner_fit_and_rng_exact(kind, hyper, extra_features):
    rng = np.random.RandomState(42)
    X = sparse.csc_matrix(rng.random((30, 6)) > .5, dtype=float)
    mean = np.asarray(X.mean(axis=0)).ravel()
    scale = np.sqrt(np.asarray(X.power(2).mean(axis=0)).ravel() - mean ** 2)
    centered = (X.toarray() - mean) / scale
    V = centered.T @ centered / X.shape[0]
    if kind == 'sparse':
        V = sparse.csc_matrix(V)
    elif kind == 'multiple':
        V = np.stack([V, V * .95])
    elif kind in ('X', 'independent'):
        V = None
    fits = []
    for engine in ('lean-regression', 'cached-inner'):
        state = PigeanState(background_prior=.05, batch_size=10)
        state.sigma2, state.sigma_power, state.p = .02, None, .1
        state.scale_factors, state.mean_shifts = scale, mean
        state.is_dense_gene_set = np.zeros(6, dtype=bool)
        state.ps, state.sigma2s = None, None
        state.X_orig = X
        np.random.seed(1729)
        random.seed(1729)
        with marginal_backend(engine, enable=extra_features if engine == 'cached-inner' else []):
            result = state._calculate_non_inf_betas(
                None, beta_tildes=np.array([.8, .4, -.1, .2, .7, .3]),
                ses=np.full(6, .2), V=V, assume_independent=kind == 'independent',
                ps=np.full(6, .1), sigma2s=np.full(6, .02),
                num_chains=4, max_num_iter=24, max_num_burn_in=8, min_num_iter=3,
                update_hyper_p=hyper, update_hyper_sigma=hyper)
        fits.append((result, state.p, state.sigma2, np.random.random(30),
                     [random.random() for _ in range(30)]))
    for a, b in zip(*fits):
        if extra_features:
            np.testing.assert_allclose(a, b, rtol=1e-10, atol=1e-12)
        else:
            np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(fits[0][-2], fits[1][-2])
    assert fits[0][-1] == fits[1][-1]


def test_backend_scope():
    assert not use_cached_inner()
    with marginal_backend('cached-inner'):
        assert use_cached_inner() and use_lean_regression()
        with marginal_backend('lean-regression'):
            assert not use_cached_inner()
        assert use_cached_inner()
    assert not use_cached_inner()
