"""Warm starts preserve outer rows; explicit opt-out preserves old trajectories."""
import ast
import inspect
import pickle
import random
from pathlib import Path

import numpy as np
import pytest
from scipy import sparse

import pigean.state as state_module
from pegs_shared.gibbs_features import DEFAULT_FEATURES, PRESETS, enabled, resolve_features
from pegs_shared.sparse_regression import marginal_backend


def native_helper(inner=4, outer=10, annotations=3):
    # Execute the real nested helper without needing a full regression fixture.
    source = Path(state_module.__file__).read_text()
    node = next(n for n in ast.walk(ast.parse(source)) if isinstance(n, ast.FunctionDef)
                and n.name == '_broadcast_initial_values_to_chain_parallel_gene_set_tensor')
    def bail(message):
        raise ValueError(message)
    env = dict(np=np, num_gene_sets=annotations, num_parallel=outer, num_chains=inner,
               chain_parallel_gene_set_shape=(inner, outer, annotations), bail=bail,
               gibbs_feature_enabled=enabled)
    exec(compile(ast.Module(body=[node], type_ignores=[]), 'native-warm-helper', 'exec'), env)
    return env[node.name]


@pytest.mark.parametrize('name,clamp', [('init_betas', False), ('init_postp', True)])
def test_labelled_outer_rows_and_legacy_opt_out(name, clamp):
    values = np.arange(30.).reshape(10, 3) / 30
    before = values.copy(); rng = pickle.dumps(np.random.get_state()); helper = native_helper()
    corrected = helper(values, name, clamp)
    np.testing.assert_array_equal(corrected, np.broadcast_to(values, (4, 10, 3)))
    with marginal_backend('legacy', disable=['warm-start-layout']):
        legacy = helper(values, name, clamp)
    np.testing.assert_array_equal(legacy, np.tile(values, 4).reshape(4, 10, 3))
    assert not np.array_equal(legacy, corrected)
    corrected[0, 0, 0] = -99
    assert corrected[1, 0, 0] == before[0, 0]
    np.testing.assert_array_equal(values, before)
    assert pickle.dumps(np.random.get_state()) == rng


@pytest.mark.parametrize('shape', [(3,), (1, 3), (4, 10, 3)])
def test_unambiguous_inputs_and_none_preserved(shape):
    values = np.arange(np.prod(shape), dtype=float).reshape(shape)
    helper = native_helper(); corrected = helper(values, 'beta')
    with marginal_backend('legacy', disable=['warm-start-layout']):
        np.testing.assert_array_equal(helper(values, 'beta'), corrected)
    assert helper(None, 'beta') is None


def test_single_outer_problem_preserved():
    helper = native_helper(outer=1)
    values = np.array([[.1, .4, .8]])
    np.testing.assert_array_equal(helper(values, 'beta'), np.broadcast_to(values, (4, 1, 3)))


def test_sanitization_clamping_and_invalid_shapes():
    values = np.tile([np.nan, np.inf, -2.], (10, 1)); values[1] = [-np.inf, .7, 4.]
    helper = native_helper()
    for clamp in [False, True]:
        expected = np.nan_to_num(values, nan=0, posinf=0, neginf=0)
        if clamp: expected = np.clip(expected, 0, 1)
        np.testing.assert_array_equal(helper(values, 'probe', clamp), np.broadcast_to(expected, (4, 10, 3)))
    for bad in [np.zeros(2), np.zeros((2, 3)), np.zeros((10, 2)), np.zeros((4, 9, 3)), np.zeros((1, 1, 1, 3))]:
        with pytest.raises(ValueError): helper(bad, 'probe')


def run_fit(initial, enabled_layout):
    rng = np.random.RandomState(42)
    X = sparse.csc_matrix(rng.random((30, 6)))
    state = state_module.PigeanState(background_prior=.05, batch_size=10)
    state.sigma2, state.sigma_power, state.p = .02, None, .1
    state.scale_factors, state.mean_shifts = np.ones(6), np.zeros(6)
    state.is_dense_gene_set = np.zeros(6, dtype=bool)
    state.ps, state.sigma2s, state.X_orig = None, None, X
    np.random.seed(171)
    random.seed(171)
    with marginal_backend('legacy', disable=[] if enabled_layout else ['warm-start-layout']):
        result = state._calculate_non_inf_betas(
            .1, beta_tildes=rng.normal(size=(3, 6)), ses=np.full((3, 6), .3),
            V=np.eye(6), num_chains=4, max_num_iter=15, min_num_iter=15,
            max_num_burn_in=5, update_hyper_sigma=False, update_hyper_p=False,
            init_betas=initial, init_postp=np.full_like(initial, .4), return_sample=True)
    return result, pickle.dumps((np.random.get_state(), random.getstate()))


@pytest.mark.parametrize('corrected', [False, True])
def test_real_inner_fit_matches_explicit_tensor_and_rng(corrected):
    rows = np.arange(18.).reshape(3, 6) / 100
    tensor = (np.broadcast_to(rows, (4, 3, 6)).copy() if corrected
              else np.tile(rows, 4).reshape(4, 3, 6))
    expected, expected_rng = run_fit(tensor, False)
    actual, actual_rng = run_fit(rows, corrected)
    for x, y in zip(actual, expected): np.testing.assert_array_equal(x, y)
    assert actual_rng == expected_rng


def test_actual_first_batch_receives_own_outer_rows(monkeypatch):
    original = state_module._update_inner_beta_gene_set_batch
    signature = inspect.signature(original)
    rows = np.arange(18.).reshape(3, 6) / 100
    class ObservedEntry(Exception): pass
    def capture(*args, **kwargs):
        values = signature.bind(*args, **kwargs).arguments
        np.testing.assert_array_equal(values['sampled_beta_t'], np.broadcast_to(rows, (4, 3, 6)))
        raise ObservedEntry()
    monkeypatch.setattr(state_module, '_update_inner_beta_gene_set_batch', capture)
    with pytest.raises(ObservedEntry): run_fit(rows, True)


def test_feature_has_no_arithmetic_dependencies():
    assert 'warm-start-layout' in DEFAULT_FEATURES
    for preset, features in PRESETS.items():
        assert 'warm-start-layout' not in features
        ordinary = set(resolve_features(preset)['active'])
        disabled = set(resolve_features(preset, disable=['warm-start-layout'])['active'])
        assert disabled == ordinary - {'warm-start-layout'}
