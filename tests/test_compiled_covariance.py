"""Compiled edge ordering, cancellation, fallbacks and optional dependency."""
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
from scipy import sparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from pegs_shared.covariance_edges import compact_edges
from pegs_shared.grouped_regression import GroupedDesign
from pegs_shared.regression import compute_logistic_beta_tildes
from pegs_shared.sparse_regression import marginal_backend, use_cached_inner, use_compiled_covariance


def example(pseudo=False):
    rng = np.random.default_rng(309)
    x = rng.choice([0., .2, .7, 1.], size=(80, 6))
    x[:, 0] = 0
    x[:, 1] = rng.normal(size=80)  # Force unaggregated continuous column.
    factor = rng.normal(size=(80, 4))
    factor /= np.sqrt(np.sum(factor**2, axis=1))[:, None]
    r = sparse.csr_matrix(.3 * factor @ factor.T + .7 * np.eye(80))
    x = sparse.csc_matrix(x)
    if pseudo:
        x = sparse.vstack([x, np.ones((1, 6))], format='csc')
    return rng, x, r


@pytest.mark.parametrize('pseudo', [False, True])
@pytest.mark.parametrize('xindex,rindex', [(np.int32, np.int32), (np.int64, np.int64), (np.int32, np.int64), (np.int64, np.int32)])
def test_exact_signed_unsorted_edges(pseudo, xindex, rindex):
    _, x, r = example(pseudo)
    d = GroupedDesign(x)
    d.X.indptr = d.X.indptr.astype(xindex)
    d.X.indices = d.X.indices.astype(xindex)
    r.indptr = r.indptr.astype(rindex)
    r.indices = r.indices.astype(rindex)
    for row in range(r.shape[0]):
        a, b = r.indptr[row:row + 2]
        r.indices[a:b] = r.indices[a:b][::-1]
        r.data[a:b] = r.data[a:b][::-1]
    r.has_sorted_indices = False
    expected, old_stats = compact_edges(d.X, d.groups, r, pseudo, compiled=False)
    actual, stats = compact_edges(d.X, d.groups, r, pseudo, compiled=True)
    assert stats == old_stats and stats['aggregated_columns'] > 0
    assert np.any(actual[3] > np.abs(actual[2]) + 1e-6)
    for a, b in zip(actual, expected):
        np.testing.assert_array_equal(a, b)
        assert a.dtype == b.dtype


@pytest.mark.parametrize('case', ['empty', 'diagonal', 'zeros', 'cancellation', 'bin_limit'])
def test_edge_boundaries(case):
    if case == 'bin_limit':
        n = 257  # 257**2 exceeds the retained pair-bin cap.
        x = sparse.csc_matrix(np.arange(1., n + 1)[:, None])
        r = sparse.csr_matrix(np.ones((n, n)))
    else:
        n = 8
        x = sparse.csc_matrix(np.ones((n, 2)))
        if case == 'empty':
            x = sparse.csc_matrix((n, 0))
        r = sparse.eye(n, format='csr') if case == 'diagonal' else sparse.csr_matrix(np.ones((n, n)))
        if case == 'zeros':
            r.data[:] = 0  # Keep explicit zero edges.
        elif case == 'cancellation':
            r.data[::2] = -1
    d = GroupedDesign(x)
    a, sa = compact_edges(d.X, d.groups, r, compiled=True)
    b, sb = compact_edges(d.X, d.groups, r, compiled=False)
    assert sa == sb
    for aa, bb in zip(a, b):
        np.testing.assert_array_equal(aa, bb)
    if case == 'bin_limit':
        assert sa['aggregated_columns'] == 0


@pytest.mark.parametrize('pseudo', [False, True])
def test_regression_and_changed_covariance_exact(pseudo):
    rng, x, r = example()
    y = np.array([rng.random(80) < .3, rng.random(80) < .45, np.zeros(80)]).astype(float)
    results = []
    for engine in ('cached-inner', 'compiled-covariance'):
        with marginal_backend(engine):
            results.append(compute_logistic_beta_tildes(
                x, y.copy(), resid_correlation_matrix=[r, sparse.eye(80, format='csr'), r],
                append_pseudo=pseudo))
    for a, b in zip(*results):
        np.testing.assert_array_equal(a, b)
    for selected in ([5, 1, 2], [2, 4], [0, 3]):
        values = []
        for engine in ('cached-inner', 'compiled-covariance'):
            with marginal_backend(engine):
                values.append(GroupedDesign(x[:, selected]).inflation(r))
        np.testing.assert_array_equal(*values)
        r.data *= .8
        r.setdiag(1.)


def test_cancellation_and_float32_fallback(monkeypatch):
    import pegs_shared.covariance as covariance
    original = covariance.marginal_inflation
    calls = []
    def record(*args, **kwargs):
        calls.append(True)
        return original(*args, **kwargs)
    monkeypatch.setattr(covariance, 'marginal_inflation', record)
    x = sparse.csc_matrix(np.tile([0., 1.], 40)[:, None])
    r = sparse.csr_matrix((1 - 1e-12) * np.ones((80, 80)) + 1e-12 * np.eye(80))
    with marginal_backend('compiled-covariance'):
        actual = GroupedDesign(x).inflation(r)
    assert calls
    np.testing.assert_array_equal(actual, original(x, r))
    d = GroupedDesign(x)
    a, sa = compact_edges(d.X, d.groups, r.astype(np.float32), compiled=True)
    b, sb = compact_edges(d.X, d.groups, r.astype(np.float32), compiled=False)
    assert sa == sb
    for aa, bb in zip(a, b):
        np.testing.assert_array_equal(aa, bb)


def test_backend_and_missing_optional_dependency():
    assert not use_compiled_covariance()
    with marginal_backend('compiled-covariance'):
        assert use_compiled_covariance() and use_cached_inner()
    assert not use_compiled_covariance()
    # A fresh process proves baseline execution does not import the compiler,
    # and explicitly choosing it fails clearly when the dependency is missing.
    code = '''
import sys
sys.modules['numba'] = None
from scipy import sparse
from pegs_shared.grouped_regression import GroupedDesign
from pegs_shared.sparse_regression import marginal_backend
x = sparse.csc_matrix([[1.], [0.]])
with marginal_backend('cached-inner'):
    GroupedDesign(x).inflation(sparse.eye(2, format='csr'))
try:
    with marginal_backend('compiled-covariance'):
        GroupedDesign(x).inflation(sparse.eye(2, format='csr'))
except ImportError as e:
    assert 'optional numba' in str(e)
else:
    raise AssertionError('Missing optional dependency did not fail')
'''
    proc = subprocess.run([sys.executable, '-c', code], env={**os.environ, 'PYTHONPATH': str(ROOT / 'src')},
                          text=True, capture_output=True)
    assert proc.returncode == 0, proc.stderr
