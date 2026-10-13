"""Regression and adverse-input checks for the integrated fast backend."""
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from eaggl import factor
from eaggl.cached_solver import CachedObjective, ConfirmationPolicy, fit
from eaggl.spectral_objective import PreciseObjective
from eaggl.spectral_solver import fit as original_fit


@pytest.mark.parametrize('seed', range(10))
def test_cached_updates_match_original_including_line_search(seed):
    rng = np.random.default_rng(seed)
    target = rng.uniform(0, .3, (9, 9))
    target = (target + target.T)/2
    kwargs = dict(K=4, phi=.03, sparsity=.001)
    w = rng.uniform(0, .3, (9, 4))
    expected = original_fit(PreciseObjective(target, **kwargs), w,
        max_iter=250, required_checks=251, window=1000, stationarity_tol=1e-6)
    actual = fit(CachedObjective(target, **kwargs), w, max_iter=250, observe_decisions=True)
    np.testing.assert_array_equal(actual[0], expected[0])
    np.testing.assert_array_equal(actual[1], expected[1])
    assert actual[2]['line_search_decision_mismatches'] == 0
    assert actual[2]['accepted_updates'] == 250


def test_adaptive_confirmation_is_required_and_fixed_mode_uses_budget():
    obj = CachedObjective(np.zeros((4, 4)), phi=0, K=2)
    initial = np.zeros((4, 2))
    early = fit(obj, initial, max_iter=3000, stop_adaptively=True)
    assert early[2]['status'] == 'iteration_cap'
    assert not early[2]['adaptive_confirmed']
    result = fit(obj, initial, max_iter=4000, stop_adaptively=True)
    assert result[2]['status'] == 'confirmed_stability'
    assert result[2]['iterations'] == 3100
    assert result[2]['adaptive_confirmed']
    fixed = fit(obj, initial, max_iter=4000)
    assert fixed[2]['iterations'] == 4000
    np.testing.assert_array_equal(fixed[0], result[0])


def test_failed_stability_resets_confirmation_and_nonfinite_cannot_pass():
    obj = CachedObjective(np.ones((3, 3))*.1, K=2)
    policy = ConfirmationPolicy(obj, confirmation=100)
    w = np.full((3, 2), .1)
    lambdas = obj.evaluate(w)[3]
    good = dict(gradient_mapping_over_W=0., window_W_change=0., strong_window_W_change=0.)
    for step in range(5):
        policy.observe(step*25, w, lambdas, 1., good, 0.)
    assert policy.anchor is not None
    policy.observe(125, w, lambdas, 1., dict(good, window_W_change=float('nan')), 0.)
    assert policy.anchor is None
    assert policy.passes == 0
    assert policy.stop is None


def test_objective_increase_cannot_be_declared_confirmation():
    obj = CachedObjective(np.ones((3, 3))*.1, K=2)
    policy = ConfirmationPolicy(obj, confirmation=25, consecutive=1)
    w = np.full((3, 2), .1)
    good = dict(gradient_mapping_over_W=0., window_W_change=0., strong_window_W_change=0.)
    policy.observe(0, w, obj.evaluate(w)[3], 1., good, 0.)
    policy.observe(25, w, obj.evaluate(w)[3], 2., good, 0.)
    assert policy.stop is None
    assert policy.failed_confirmations == 1


def test_fixed_budget_off_checkpoint_still_reports_loading_drift():
    obj = CachedObjective(np.zeros((4, 4)), phi=0, K=2)
    result = fit(obj, np.zeros((4, 2)), max_iter=1005)
    assert result[2]['window_W_change'] == 0
    assert result[2]['strong_window_W_change'] == 0


def test_unavailable_diagnostics_are_null_not_invalid_json_numbers():
    import json
    from eaggl.outputs import _json_safe
    payload = dict(runs=[dict(window=float('nan'), residual=float('inf'), iterations=5)])
    decoded = json.loads(json.dumps(_json_safe(payload), allow_nan=False))
    assert decoded['runs'][0] == dict(window=None, residual=None, iterations=5)


@pytest.mark.parametrize('kwargs', [dict(max_iter=0), dict(max_iter=2.5),
    dict(check_every=0), dict(window=26), dict(stationarity_tol=float('nan')),
    dict(confirmation=0), dict(required_checks=0)])
def test_invalid_budgets_and_tolerances_raise(kwargs):
    obj = CachedObjective(np.ones((3, 3))*.1, K=2)
    with pytest.raises(ValueError):
        fit(obj, np.full((3, 2), .1), **kwargs)


@pytest.mark.parametrize('bad', [float('nan'), float('inf')])
def test_invalid_initial_values_raise(bad):
    obj = CachedObjective(np.ones((3, 3))*.1, K=2)
    w = np.full((3, 2), .1)
    w[0, 0] = bad
    with pytest.raises(ValueError):
        fit(obj, w)


@pytest.mark.parametrize('model, method, budget, tolerance', [
    ('gene_by_gene', 'auto', 20000, 1e-6),
    ('gene_by_annotation', 'auto', 100, 1e-4),
    ('gene_by_gene', 'legacy', 100, 1e-4)])
def test_stage_resolves_mode_specific_defaults(model, method, budget, tolerance):
    options = SimpleNamespace(discovery_model=model, symnmf_update_method=method,
                              max_num_iterations=None, rel_tol=None)
    runtime = SimpleNamespace(_record_params=lambda *a, **k: None)
    domain = SimpleNamespace(bail=lambda s: (_ for _ in ()).throw(ValueError(s)))
    with patch.object(factor, 'extract_factor_workflow', return_value=SimpleNamespace(workflow_id='test')), \
         patch.object(factor, 'extract_factor_inputs'), \
         patch.object(factor, 'build_factor_execution_config'), \
         patch.object(factor, 'run_factor_model'):
        factor.run_main_factor_stage(domain, runtime, options, {}, {})
    assert options.max_num_iterations == budget
    assert options.rel_tol == tolerance
    assert runtime.internal_symnmf_update_method == ('legacy' if model == 'gene_by_annotation' else
                                                    'projected_gradient' if method == 'auto' else method)
