"""Numerical equivalence, invalidation, and lifetime of structural reuse."""
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy import sparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from pegs_shared.grouped_regression import GroupedDesign
from pegs_shared.regression import compute_logistic_beta_tildes
from pegs_shared.sparse_regression import marginal_backend, annotation_structure_cache
from pegs_shared.structure_cache import ColumnGroupingCache


def design():
    rng = np.random.default_rng(917)
    x = rng.choice([0., 0., .25, .6, 1.], size=(80, 9))
    x[:, 0] = 0
    x[:, 1] = 1
    x[:, 2] = rng.normal(size=80)
    return rng, sparse.csc_matrix(x)


def test_exact_column_keys_detect_mutations_and_bound_memory():
    cache = ColumnGroupingCache(max_bytes=200, max_entries=2)
    rows = np.array([1, 3, 4], dtype=np.int32)
    data = np.array([.25, 1., .25])
    first = cache.column_groups(8, rows, data)
    assert cache.column_groups(8, rows.copy(), data.copy()) is first
    with pytest.raises(ValueError):
        first[0][0] = 8
    for n, r, v in [(9, rows, data), (8, rows+1, data), (8, rows, data+.1)]:
        got = cache.column_groups(n, r, v)
        for a, b in zip(got, np.unique(v, return_inverse=True, return_counts=True)):
            np.testing.assert_array_equal(a, b)
        assert cache.payload_bytes <= 200
        assert len(cache.entries) <= 2
    assert cache.hits == 1 and cache.misses == 4
    cache.column_groups(8, rows, data)  # Original entry must have been evicted.
    assert cache.misses == 5
    tiny = ColumnGroupingCache(max_bytes=1)
    tiny.column_groups(8, rows, data)
    assert not tiny.entries and tiny.payload_bytes == 0


def test_selection_reordering_and_changed_values_rebuild_exact_design():
    _, x = design()
    attrs = ('columns', 'values', 'counts', 'groups', 'zeros',
             'chain_columns', 'chain_values', 'chain_counts')
    changed = x.copy()
    changed.data[0] += .125
    variants = [x, x[:, [5, 2, 0, 1]], x[:, [1, 2, 5]], changed,
                sparse.vstack([x, np.ones((1, 9))], format='csc')]
    references = [GroupedDesign(v, 3) for v in variants]
    with marginal_backend('grouped-cached') as cache:
        for v, ref in zip(variants, references):
            actual = GroupedDesign(v, 3)
            for attr in attrs:
                np.testing.assert_array_equal(getattr(actual, attr), getattr(ref, attr))
            assert actual._correlations == {}
        assert cache.hits >= 7
        assert cache.misses >= 10


def test_nested_and_exceptional_contexts_restore_and_release_cache():
    assert annotation_structure_cache() is None
    with marginal_backend('grouped-cached') as first:
        assert annotation_structure_cache() is first
        with marginal_backend('legacy'):
            assert annotation_structure_cache() is None
        with pytest.raises(RuntimeError):
            with marginal_backend('grouped-cached') as second:
                assert second is not first
                raise RuntimeError('deliberate unwind')
        assert annotation_structure_cache() is first
    assert annotation_structure_cache() is None
    with marginal_backend('grouped-cached') as fresh:
        assert fresh is not first and not fresh.entries


@pytest.mark.parametrize('pseudo', [False, True])
def test_changed_response_covariance_and_chain_count_match_uncached(pseudo):
    rng, x = design()
    y = np.array([rng.random(80)<.2, rng.random(80)<.4, np.zeros(80)]).astype(float)
    r = sparse.diags([np.full(79, -.15), np.ones(80), np.full(79, -.15)], [-1,0,1], format='csr')
    with marginal_backend('grouped-cached') as cache:
        for columns, outcomes, correlation in [
                (list(range(9)), y, r),
                ([8, 2, 1, 5], 1-y[:2], sparse.eye(80, format='csr')),
                ([1, 2, 8], y[0], .7*r+.3*sparse.eye(80))]:
            xx = x[:, columns]
            kw = dict(resid_correlation_matrix=correlation, append_pseudo=pseudo)
            got = compute_logistic_beta_tildes(xx, outcomes.copy(), **kw)
            for backend in ('grouped', 'legacy'):
                with marginal_backend(backend):
                    ref = compute_logistic_beta_tildes(xx, outcomes.copy(), **kw)
                for field, (a, b) in enumerate(zip(got, ref)):
                    if a is None or b is None:
                        assert a is b
                    elif backend == 'grouped':
                        # The new cache must also preserve inherited behavior
                        # for singular columns, without any tolerance relaxation.
                        np.testing.assert_array_equal(a, b)
                    else:
                        if field == 1:
                            # Existing grouped and legacy SE arithmetic differs
                            # on non-identifiable constant columns (cancellation).
                            # All coefficients/other fields are still checked;
                            # full SE parity is required for varying columns.
                            varying = np.var(xx.toarray(), axis=0) > 0
                            a, b = a[..., varying], b[..., varying]
                        np.testing.assert_allclose(a, b, rtol=2e-8, atol=2e-10)
        assert cache.hits >= 7


def test_noncanonical_sparse_columns_are_canonicalized_before_caching():
    x = sparse.csc_matrix((np.array([.2, .3, 0., 1.]),
                           np.array([2, 2, 1, 0]), np.array([0, 4])), shape=(4, 1))
    with marginal_backend('grouped-cached') as cache:
        a = GroupedDesign(x)
        b = GroupedDesign(sparse.csc_matrix(x.toarray()))
        assert cache.hits == 1
        np.testing.assert_array_equal(a.groups, b.groups)
        np.testing.assert_array_equal(a.values, b.values)


def test_pipeline_scopes_cache_to_one_gibbs_run(monkeypatch):
    from types import SimpleNamespace
    from pigean import pipeline
    config = SimpleNamespace(run_kwargs={}, num_chains=10, total_num_iter=500)
    monkeypatch.setattr(pipeline, 'build_main_gibbs_stage_config', lambda opts: config)
    seen = []
    state = SimpleNamespace(run_gibbs=lambda: seen.append(annotation_structure_cache()))
    for engine in ['cached-grouped-regression', 'grouped-regression', 'cached-grouped-regression']:
        pipeline.run_main_gibbs_stage(None, state, SimpleNamespace(gibbs_engine=engine), {'run_gibbs':True})
    assert seen[0] is not None and seen[1] is None and seen[2] is not None
    assert seen[0] is not seen[2]
    assert annotation_structure_cache() is None
