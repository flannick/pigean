"""Opt-in sparse arithmetic for the existing marginal-regression estimator."""
from contextlib import contextmanager
from contextvars import ContextVar

import numpy as np
from scipy import sparse
from scipy.special import expit
from .gibbs_features import ACTIVE, enabled, resolve_features


_BACKEND = ContextVar('marginal_regression_backend', default='legacy')
_STRUCTURE_CACHE = ContextVar('annotation_structure_cache', default=None)
_PREFILTER_POOL = ContextVar('prefilter_pool', default=None)


@contextmanager
def marginal_backend(name, *, enable=(), disable=()):
    selection = resolve_features(name, enable, disable)
    feature_token = ACTIVE.set(frozenset(selection['active']))
    cache = None
    pool = None
    try:
        if enabled('annotation-cache') or enabled('compact-covariance'):
            from .structure_cache import ColumnGroupingCache
            cache = ColumnGroupingCache()
            cache.covariance_stats = dict(preparations=0, input_edges=0,
                stored_pairs=0, aggregated_columns=0,
                peak_edge_payload_bytes=0, peak_lookup_scratch_bytes=0)
        if enabled('parallel-prefilter'):
            from concurrent.futures import ThreadPoolExecutor
            pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix='pigean-prefilter')
        pool_token = _PREFILTER_POOL.set(pool)
        token = _BACKEND.set(name)
        cache_token = _STRUCTURE_CACHE.set(cache)
        try:
            yield cache
        finally:
            _STRUCTURE_CACHE.reset(cache_token)
            _BACKEND.reset(token)
            _PREFILTER_POOL.reset(pool_token)
    finally:
        try:
            if pool is not None:
                pool.shutdown(wait=True, cancel_futures=True)
        finally:
            ACTIVE.reset(feature_token)


def use_sparse_backend():
    return enabled('sparse-regression')


def use_grouped_backend():
    return enabled('grouped-regression')


def use_compact_covariance():
    return enabled('compact-covariance')


def use_lean_regression():
    return enabled('lean-regression')


def use_cached_inner():
    return enabled('cached-inner')


def use_compiled_covariance():
    return enabled('compiled-covariance')


def annotation_structure_cache():
    return _STRUCTURE_CACHE.get() if enabled('annotation-cache') else None


def covariance_statistics():
    cache = _STRUCTURE_CACHE.get()
    return None if cache is None else cache.covariance_stats


def logistic_moments(X, beta, alpha, max_probability):
    """Sums of p, v, x*p, x*v, and x*x*v over explicitly nonzero x.

    The legacy fit handles implicit-zero observations separately. Keep its
    exponential and clipping conventions, including its probability cap.
    """
    a = sparse.csc_matrix(X, dtype=float, copy=True)
    a.sum_duplicates()
    a.eliminate_zeros()
    columns = np.repeat(np.arange(a.shape[1]), np.diff(a.indptr))
    eta = np.minimum(a.data * beta[columns] + alpha[columns], 100)
    ex = np.exp(eta)
    p = np.minimum(ex / (1 + ex), max_probability)
    v = p * (1 - p)
    def total(values):
        return np.bincount(columns, weights=values, minlength=a.shape[1])
    return tuple(total(values) for values in (p, v, a.data*p, a.data*v, a.data*a.data*v))


def sparse_inflation(X, r, beta, alpha, pseudo, fallback, block_size):
    """Optionally run independent linear chunks on two shared-memory workers."""
    calculate_serial = _sparse_inflation_serial
    if enabled('compiled-prefilter') and beta is None:
        from .compiled_prefilter import linear_inflation
        calculate_serial = linear_inflation
    pool = _PREFILTER_POOL.get()
    if pool is None or beta is not None or X.shape[1] <= block_size:
        return calculate_serial(X, r, beta, alpha, pseudo, fallback, block_size)
    # Preserve the original block boundaries and reduction order within each
    # coefficient. Each worker owns its X copy and temporaries; r is read-only.
    k = X.shape[1]
    split = ((k + block_size - 1) // block_size + 1) // 2 * block_size
    def calculate(begin, end):
        return calculate_serial(
            X[:, begin:end], r, None, None, pseudo, fallback, block_size)
    futures = [pool.submit(calculate, begin, end) for begin, end in ((0, split), (split, k))]
    return np.concatenate([future.result() for future in futures])


def _sparse_inflation_serial(X, r, beta, alpha, pseudo, fallback, block_size):
    """Compute the same quadratic form using u = c*1 + sparse_delta.

    At implicit zeros, logistic variance depends only on the intercept. The
    independent pseudocount contributes to centering but not to r's dimension.
    Operator correlations retain the reference path. Near cancellation
    also falls back to that path rather than clipping away numerical errors.
    """
    a = sparse.csc_matrix(X, dtype=float, copy=True)
    a.sum_duplicates()
    a.eliminate_zeros()
    n, k = a.shape
    real_n = n - int(pseudo)
    r1 = np.asarray(r.sum(axis=1)).ravel()
    total_r = r1.sum()
    output = np.empty(k)
    for begin in range(0, k, block_size):
        end = min(k, begin + block_size)
        b = a[:, begin:end].copy()
        columns = np.repeat(np.arange(end-begin), np.diff(b.indptr))
        if beta is None:
            mean = np.asarray(b.sum(axis=0)).ravel() / n
            constant = -mean
            delta = b
        else:
            bb, aa = beta[begin:end], alpha[begin:end]
            p0 = expit(np.clip(aa, -100, 100))
            v0 = p0 * (1-p0)
            p = expit(np.clip(b.data*bb[columns] + aa[columns], -100, 100))
            v = p * (1-p)
            total_v = n*v0 + np.bincount(columns, weights=v-v0[columns], minlength=end-begin)
            weighted_x = np.bincount(columns, weights=v*b.data, minlength=end-begin)
            mean = np.divide(weighted_x, total_v, out=np.zeros(end-begin), where=total_v > 0)
            constant = -np.sqrt(v0)*mean
            delta = b.copy()
            delta.data = np.sqrt(v)*(b.data-mean[columns]) - constant[columns]
        pseudo_sq = (delta[-1, :].toarray().ravel()+constant)**2 if pseudo else 0
        d = delta[:real_n, :]
        dsum = np.asarray(d.sum(axis=0)).ravel()
        d2 = np.asarray(d.multiply(d).sum(axis=0)).ravel()
        independent = d2 + 2*constant*dsum + real_n*constant**2 + pseudo_sq
        rd = r @ d
        quadratic = np.asarray(d.multiply(rd).sum(axis=0)).ravel()
        cross = 2*constant*np.asarray(d.T @ r1).ravel()
        background = constant**2*total_r
        correlated = quadratic + cross + background + pseudo_sq
        # Detect cancellation relative to the expanded terms, not just the sum.
        scale = np.abs(quadratic)+np.abs(cross)+np.abs(background)+pseudo_sq
        ind_scale = d2 + np.abs(2*constant*dsum) + real_n*constant**2 + pseudo_sq
        unsafe = ((independent <= 1e-10*ind_scale) | (correlated <= 1e-10*scale)
                  | ~np.isfinite(independent) | ~np.isfinite(correlated))
        ratio = np.divide(correlated, independent, out=np.ones(end-begin), where=independent > 0)
        ratio[unsafe] = 1
        output[begin:end] = np.sqrt(ratio)
        if np.any(unsafe):
            output[begin:end][unsafe] = fallback(
                b[:, unsafe], r,
                beta=None if beta is None else beta[begin:end][unsafe],
                alpha=None if alpha is None else alpha[begin:end][unsafe],
                pseudo=pseudo, block_size=block_size, backend='legacy')
    return output
