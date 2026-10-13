"""Exact supported-path caching, content invalidation, cap and lifetime checks."""
from pathlib import Path
import sys
import numpy as np
from scipy import sparse
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from pigean.state import PigeanState
from pigean.static_correlation import StaticCorrelationCache
from pegs_shared.sparse_regression import marginal_backend,static_correlation_cache


def test_content_invalidation_masks_and_cap():
    rng=np.random.default_rng(611)
    state=PigeanState(background_prior=.05,batch_size=10)
    state.X_orig=sparse.csc_matrix(rng.random((71,29))<.2,dtype=float)
    state.mean_shifts=np.asarray(state.X_orig.mean(axis=0)).ravel()
    state.scale_factors=np.sqrt(state.mean_shifts*(1-state.mean_shifts))
    state.y_corr_cholesky=None
    cache=StaticCorrelationCache(block_columns=7)
    for step in range(7):
        if step==2:state.X_orig.data[0]*=.8
        if step==3:state.mean_shifts[1]*=.9
        if step==4:state.scale_factors[2]*=1.1
        mask=rng.random(29)>.4
        expected=state._calculate_V_internal(state.X_orig[:,mask],None,state.mean_shifts[mask],state.scale_factors[mask])
        np.testing.assert_array_equal(cache.get(state,mask),expected)
    assert cache.builds==4 and cache.hits==3
    assert not cache.matrix.flags.writeable
    assert StaticCorrelationCache(max_bytes=0).get(state,mask) is None
    state.y_corr_cholesky=np.eye(71)
    assert cache.get(state,mask) is None and cache.matrix is None


def test_scope_isolation_and_no_prerequisites():
    assert static_correlation_cache() is None
    with marginal_backend('legacy',enable=['static-correlation']):
        cache=static_correlation_cache()
        assert cache is not None
        with marginal_backend('legacy'):assert static_correlation_cache() is None
        assert static_correlation_cache() is cache
    assert static_correlation_cache() is None
