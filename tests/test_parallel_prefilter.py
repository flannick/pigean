"""Exact block boundaries, bounded workers, RNG and cleanup for prefilter."""
from pathlib import Path
import pickle
import random
import sys
import threading

import numpy as np
import pytest
from scipy import sparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from pegs_shared.covariance import marginal_inflation
from pegs_shared.regression import compute_beta_tildes
import pegs_shared.sparse_regression as module


def fixture(k, pseudo=False):
    rng = np.random.default_rng(483)
    x = sparse.csc_matrix(rng.choice([0., .2, .7, 1.], size=(40, k)))
    factor = rng.normal(size=(40, 3))
    factor /= np.sqrt(np.sum(factor**2, axis=1))[:, None]
    r = sparse.csr_matrix(.3 * factor @ factor.T + .7 * np.eye(40))
    if pseudo:
        x = sparse.vstack([x, np.ones((1, k))], format='csc')
    return rng, x, r


@pytest.mark.parametrize('k', [1, 512, 513, 1025])
@pytest.mark.parametrize('pseudo', [False, True])
def test_partial_chunks_exact_and_rng(k, pseudo):
    _, x, r = fixture(k, pseudo)
    np_state, py_state = pickle.dumps(np.random.get_state()), random.getstate()
    values = []
    for engine in ('compiled-covariance', 'parallel-prefilter'):
        with module.marginal_backend(engine):
            values.append(marginal_inflation(x, r, pseudo=pseudo, backend='sparse', block_size=512))
    np.testing.assert_array_equal(*values)
    assert pickle.dumps(np.random.get_state()) == np_state
    assert random.getstate() == py_state


def test_linear_fit_cancellation_and_logistic_serial_path():
    rng, x, r = fixture(1025)
    y = rng.random((3, 40))
    values = []
    for engine in ('compiled-covariance', 'parallel-prefilter'):
        with module.marginal_backend(engine):
            values.append(compute_beta_tildes(x, y.copy(), resid_correlation_matrix=r))
    for a, b in zip(*values):
        np.testing.assert_array_equal(a, b)
    # Same serial cancellation fallback for ill-conditioned correlation.
    nearly_constant = sparse.csr_matrix((1 - 1e-12) * np.ones((40, 40)) + 1e-12 * np.eye(40))
    with module.marginal_backend('parallel-prefilter'):
        result = marginal_inflation(x, nearly_constant, backend='sparse', block_size=512)
    with module.marginal_backend('compiled-covariance'):
        reference = marginal_inflation(x, nearly_constant, backend='sparse', block_size=512)
    np.testing.assert_array_equal(result, reference)
    with module.marginal_backend('parallel-prefilter'):
        pool = module._PREFILTER_POOL.get()
        marginal_inflation(x, r, beta=np.zeros(x.shape[1]), alpha=np.zeros(x.shape[1]), backend='sparse', block_size=512)
        assert not pool._threads  # Logistic fits do not start workers.


def test_two_workers_and_cleanup_on_error(monkeypatch):
    _, x, r = fixture(1025)
    original = module._sparse_inflation_serial
    ids = set()
    lock = threading.Lock()
    barrier = threading.Barrier(2)
    def fail_on_second(*args, **kwargs):
        with lock:
            ids.add(threading.get_ident())
        barrier.wait(timeout=10)
        if args[0].shape[1] == 1:
            raise RuntimeError('intentional worker failure')
        return original(*args, **kwargs)
    monkeypatch.setattr(module, '_sparse_inflation_serial', fail_on_second)
    with pytest.raises(RuntimeError, match='intentional worker failure'):
        with module.marginal_backend('parallel-prefilter'):
            pool = module._PREFILTER_POOL.get()
            marginal_inflation(x, r, backend='sparse', block_size=512)
    assert len(ids) == 2
    assert all(not thread.is_alive() for thread in pool._threads)
    assert module._PREFILTER_POOL.get() is None
    assert not module.use_compiled_covariance()


def test_nested_scope_restores_executor():
    with module.marginal_backend('parallel-prefilter'):
        pool = module._PREFILTER_POOL.get()
        with module.marginal_backend('compiled-covariance'):
            assert module._PREFILTER_POOL.get() is None
        assert module._PREFILTER_POOL.get() is pool
    assert module._PREFILTER_POOL.get() is None
