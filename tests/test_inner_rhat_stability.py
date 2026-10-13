"""Exercise real inner stopping decisions and the default numerical boundary."""
import pickle
from contextlib import nullcontext

import numpy as np
import pytest

import pigean.state as state
from pigean.inner_diagnostics import stabilize_rhat_mean_boundary
from pegs_shared.gibbs_features import ACTIVE, PRESETS, DEFAULT_FEATURES, resolve_features
from pegs_shared.sparse_regression import marginal_backend


def stopping_check(monkeypatch, rhat, enabled, use_max=False):
    rhat = np.asarray(rhat, dtype=float)[None, :]
    n = rhat.shape[1]
    tensors = [np.zeros((4, 1, n)) for _ in range(5)]
    monkeypatch.setattr(state, '_calculate_r_tensor_from_chain_sums',
                        lambda *a: (np.ones_like(rhat), np.ones_like(rhat),
                                    rhat, np.ones((4, 1)), tensors[0].copy()))
    rng_before = pickle.dumps(np.random.get_state())
    with (nullcontext() if enabled is None else marginal_backend(
            'legacy', enable=['inner-rhat-stable'] if enabled else [],
            disable=[] if enabled else ['inner-rhat-stable'])):
        result = state._update_inner_beta_rhat_and_outliers(
            tensors[0], tensors[1], 11, np.array([True]), use_max, 10,
            tensors[2], tensors[3], tensors[4], np.array([True]), 1.01, 1)
    assert pickle.dumps(np.random.get_state()) == rng_before
    # The diagnostic R array is the original, including near-one/nonfinite entries.
    assert result[0] is rhat
    for tensor in tensors:
        np.testing.assert_array_equal(tensor, np.zeros_like(tensor))
    return bool(result[2][0])  # still burning in


def test_one_step_rounding_no_longer_changes_actual_stopping(monkeypatch):
    exact = [1.0199, 1.0]
    below = [1.0199, np.nextafter(1.0, 0.0)]
    assert not stopping_check(monkeypatch, exact, False)
    assert stopping_check(monkeypatch, below, False)
    assert not stopping_check(monkeypatch, exact, True)
    assert not stopping_check(monkeypatch, below, True)


@pytest.mark.parametrize('values', [[1., 1.], [.99, .98], [1.02, .99],
                                   [1.01, 1.01], [np.nan, 1.], [np.inf, 1.]])
def test_material_and_degenerate_decisions_preserved(monkeypatch, values):
    assert stopping_check(monkeypatch, values, True) == stopping_check(monkeypatch, values, False)


def test_max_based_stopping_and_raw_values_unchanged(monkeypatch):
    values = [1.0199, np.nextafter(1., 0.)]
    assert stopping_check(monkeypatch, values, True, use_max=True)
    assert stopping_check(monkeypatch, values, False, use_max=True)


def test_snap_scope_input_immutability_and_nonfinite_values():
    values = np.array([0., .99, 1.-1e-10, 1.-16*np.spacing(1.),
                       np.nextafter(1., 0.), 1., np.nextafter(1., 2.),
                       1.+16*np.spacing(1.), 1.+17*np.spacing(1.),
                       1.+1e-10, 1.01, np.nan, np.inf, -np.inf])
    before = values.copy()
    expected = values.copy()
    expected[3:8] = 1.
    np.testing.assert_array_equal(stabilize_rhat_mean_boundary(values), expected)
    np.testing.assert_array_equal(values, before)


def test_feature_independent_on_by_default_and_context_restored():
    assert all('inner-rhat-stable' not in v for v in PRESETS.values())
    assert all('inner-rhat-stable' in resolve_features(p)['active'] for p in PRESETS)
    assert resolve_features('legacy', enable=['inner-rhat-stable'])['active'] == ['inner-rhat-stable']
    assert resolve_features('legacy', disable=['inner-rhat-stable'])['active'] == []
    with marginal_backend('legacy', enable=['inner-rhat-stable,inner-matmul']):
        assert ACTIVE.get() == {'inner-rhat-stable', 'inner-matmul'}
    assert ACTIVE.get() == DEFAULT_FEATURES


def test_direct_python_default_uses_stable_boundary(monkeypatch):
    assert not stopping_check(monkeypatch, [1.0199, np.nextafter(1., 0.)], None)


def test_explicit_disable_nested_context_and_exception_restore():
    with marginal_backend('legacy', disable=['inner-rhat-stable']):
        assert not ACTIVE.get()
        with pytest.raises(RuntimeError):
            with marginal_backend('legacy'):
                assert ACTIVE.get() == DEFAULT_FEATURES
                raise RuntimeError('test restoration')
        assert not ACTIVE.get()
    assert ACTIVE.get() == DEFAULT_FEATURES
