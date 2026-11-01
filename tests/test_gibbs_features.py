"""Independent subsets, prerequisite fallbacks and old-preset preservation."""
from itertools import combinations
from pathlib import Path
import sys

import numpy as np
import pytest
from scipy import sparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from pegs_shared.gibbs_features import FEATURES, PRESETS, ACTIVE, DEFAULT_FEATURES, resolve_features
from pegs_shared.sparse_regression import marginal_backend, annotation_structure_cache
from pegs_shared.regression import compute_logistic_beta_tildes


def subsets():
    original = FEATURES[:8]
    return [subset for n in range(len(original)+1) for subset in combinations(original, n)]


def test_original_presets_and_overrides():
    for name, expected in PRESETS.items():
        assert set(resolve_features(name)['active']) == expected | DEFAULT_FEATURES
        with marginal_backend(name):
            assert ACTIVE.get() == expected | DEFAULT_FEATURES
        assert ACTIVE.get() == DEFAULT_FEATURES
    with marginal_backend('parallel-prefilter', disable=['annotation-cache']):
        assert annotation_structure_cache() is None
        assert 'compiled-covariance' in ACTIVE.get()
    with marginal_backend('legacy', enable=['cached-inner']):
        assert ACTIVE.get() == {'cached-inner'} | DEFAULT_FEATURES
        with marginal_backend('legacy'):
            assert ACTIVE.get() == DEFAULT_FEATURES
        assert ACTIVE.get() == {'cached-inner'} | DEFAULT_FEATURES


def test_fail_closed_and_inactive_prerequisites():
    with pytest.raises(ValueError, match='Unknown'):
        resolve_features(enable=['typo'])
    with pytest.raises(ValueError, match='both'):
        resolve_features(enable=['cached-inner'], disable=['cached-inner'])
    selected = resolve_features('parallel-prefilter', disable=['grouped-regression'])
    assert set(selected['active']) == {'sparse-regression', 'cached-inner', 'parallel-prefilter'} | DEFAULT_FEATURES
    assert set(selected['inactive']) == {'annotation-cache', 'compact-covariance', 'lean-regression', 'compiled-covariance'}
    assert ACTIVE.get() == DEFAULT_FEATURES


def test_comma_lists_repetition_whitespace_and_duplicates():
    expected = resolve_features('parallel-prefilter',
                                enable=['inner-matmul', 'inner-fused'],
                                disable=['annotation-cache', 'compiled-covariance'])
    observed = resolve_features('parallel-prefilter',
                                enable=[' inner-matmul, inner-fused ', 'inner-matmul'],
                                disable=['annotation-cache,compiled-covariance', 'annotation-cache'])
    assert observed == expected
    with marginal_backend('parallel-prefilter',
                          enable=['inner-matmul,inner-fused'],
                          disable=['annotation-cache,compiled-covariance']):
        assert ACTIVE.get() == set(expected['active'])
        assert annotation_structure_cache() is None
    assert ACTIVE.get() == DEFAULT_FEATURES
    assert resolve_features(enable='inner-matmul,inner-fused')['active'] == ['inner-fused', 'inner-matmul', 'inner-rhat-stable']


@pytest.mark.parametrize('value', ['', ' ', ',', 'inner-matmul,', ',inner-matmul', 'inner-matmul,,inner-fused'])
@pytest.mark.parametrize('option', ['enable', 'disable'])
def test_empty_list_members_are_rejected(value, option):
    with pytest.raises(ValueError, match='Empty Gibbs feature name'):
        resolve_features(**{option: [value]})


def test_comma_lists_preserve_validation_and_prerequisites():
    with pytest.raises(ValueError, match='Unknown Gibbs features: typo'):
        resolve_features(enable=['inner-matmul,typo'])
    with pytest.raises(ValueError, match='Cannot both enable and disable: inner-fused'):
        resolve_features(enable=['inner-matmul,inner-fused'], disable=['inner-fused,cached-inner'])
    result = resolve_features('parallel-prefilter', disable=['grouped-regression,cached-inner'])
    assert set(result['active']) == {'sparse-regression', 'parallel-prefilter'} | DEFAULT_FEATURES
    assert set(result['inactive']) == {'annotation-cache', 'compact-covariance', 'lean-regression', 'compiled-covariance'}


@pytest.mark.parametrize('name', ['inner-early-check', 'inner-mean', 'inner-residual'])
def test_rejected_experiments_are_not_available(name):
    with pytest.raises(ValueError, match='Unknown Gibbs features'):
        resolve_features(enable=[name])


@pytest.mark.parametrize('features', subsets())
def test_every_original_subset_runs_and_preserves_estimates(features):
    rng = np.random.default_rng(639)
    X = sparse.csc_matrix(rng.choice([0., 0., .4, 1.], size=(25, 5)))
    Y = rng.random((2, 25))
    factor = rng.normal(size=(25, 3))
    factor /= np.sqrt(np.sum(factor**2, axis=1))[:, None]
    R = sparse.csr_matrix(.2 * factor @ factor.T + .8 * np.eye(25))
    kw = dict(resid_correlation_matrix=R)
    with marginal_backend('legacy'):
        expected = compute_logistic_beta_tildes(X, Y.copy(), **kw)
    with marginal_backend('legacy', enable=features):
        observed = compute_logistic_beta_tildes(X, Y.copy(), **kw)
    for a, b in zip(expected, observed):
        if a is None or b is None:
            assert a is b
        else:
            np.testing.assert_allclose(a, b, rtol=1e-10, atol=1e-11, equal_nan=True)
    assert ACTIVE.get() == DEFAULT_FEATURES


def test_new_switches_are_independent_and_presets_unchanged():
    new = FEATURES[8:]
    for n in range(len(new)+1):
        for subset in combinations(new,n):
            selected = resolve_features('parallel-prefilter',enable=subset)
            assert set(selected['active']) == set(FEATURES[:8]) | set(subset) | DEFAULT_FEATURES
            for item in subset:
                without = resolve_features('parallel-prefilter',enable=set(subset)-{item},disable=[item])
                assert set(without['active']) == set(selected['active'])-{item}
    assert not set(new) & set().union(*PRESETS.values())


@pytest.mark.parametrize('enable,expected', [(None, set()), (['inner-matmul,inner-fused'], {'inner-matmul', 'inner-fused'})])
def test_pipeline_scopes_features_without_changing_sampler_arguments(monkeypatch, enable, expected):
    from types import SimpleNamespace
    from unittest.mock import Mock
    from pigean import pipeline

    kwargs = {'num_chains': 10, 'total_num_iter': 500}
    config = SimpleNamespace(run_kwargs=kwargs, num_chains=10, total_num_iter=500)
    monkeypatch.setattr(pipeline, 'build_main_gibbs_stage_config', lambda options: config)
    def sample(**passed):
        assert passed == kwargs
        assert ACTIVE.get() == expected | DEFAULT_FEATURES
    state = SimpleNamespace(run_gibbs=Mock(side_effect=sample))
    options = SimpleNamespace(gibbs_engine='legacy', gibbs_enable=enable, gibbs_disable=None)
    result = pipeline.run_main_gibbs_stage(Mock(), state, options, {'run_gibbs': True})
    state.run_gibbs.assert_called_once_with(**kwargs)
    assert result.ran
    assert ACTIVE.get() == DEFAULT_FEATURES
