"""Mathematical and compatibility checks for opt-in numerical discovery."""
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.optimize import minimize

from eaggl import cli, factor
from eaggl.spectral_objective import PreciseObjective
from eaggl.spectral_solver import project_rows
from eaggl.symnmf_numerics import projected_gradient_fit


def test_numerical_defaults_are_mode_specific():
    options, _ = cli.parser.parse_args([])
    assert options.symnmf_update_method == 'auto'
    assert options.symnmf_stopping == 'adaptive'
    assert options.fixed_basis_projection_method == 'legacy'
    assert options.max_num_iterations is None
    assert options.rel_tol is None
    assert options.projection_context_in is None
    assert options.projection_context_out is None
    assert options.factor_runs == 1
    assert not options.consensus_nmf


def test_profiled_off_diagonal_gradient_and_stable_change():
    rng = np.random.default_rng(85)
    target = rng.uniform(0, .2, (5, 5))
    target = (target + target.T) / 2
    objective = PreciseObjective(target, K=3, phi=.08, sparsity=.003)
    w = rng.uniform(.05, .2, (5, 3))
    value, gradient, error, lambdas = objective.evaluate(w)
    residual = target - w @ w.T
    np.fill_diagonal(residual, 0)
    assert error == pytest.approx(np.sum(residual ** 2), abs=1e-12)
    for index in np.ndindex(w.shape):
        direction = np.zeros_like(w)
        direction[index] = 1e-6
        derivative = (objective.evaluate(w + direction)[0] - objective.evaluate(w - direction)[0]) / 2e-6
        assert gradient[index] == pytest.approx(derivative, abs=1e-8)
    for magnitude in (1e-2, 1e-4, 1e-7):
        delta = rng.normal(0, magnitude, w.shape)
        change = objective.change(w, delta, gradient)
        assert change == pytest.approx(objective.evaluate(w + delta)[0] - value, abs=1e-12)


def test_row_projection_agrees_with_independent_constrained_optimizer():
    values = np.array([[.7, .9, -.2], [.1, .2, .3], [-.1, -.2, -.3], [1.1, .8, .4]])
    projected = project_rows(values)
    for row, expected in zip(values, projected):
        reference = minimize(lambda x: .5 * np.sum((x - row) ** 2), np.zeros(3),
            jac=lambda x: x - row, method='SLSQP', bounds=[(0, None)] * 3,
            constraints=[dict(type='ineq', fun=lambda x: 1 - x.sum(), jac=lambda x: -np.ones(3))],
            options=dict(ftol=1e-13, maxiter=1000))
        assert reference.success
        np.testing.assert_allclose(reference.x, expected, atol=1e-8)
    np.testing.assert_allclose(project_rows(projected), projected, atol=1e-15)
    assert np.all(projected >= 0)
    assert np.all(projected.sum(axis=1) <= 1)


def adapter_settings():
    return dict(phi_eff=.001, C=13., b0=.1, sparsity=0., row_sum_cap=True,
                max_iter=5, rel_tol=1e-6)


@pytest.mark.parametrize('change', [dict(row_sum_cap=False), dict(views=np.ones((2, 4, 4))),
                                    dict(weights=np.ones((4, 4)))])
def test_unsupported_objectives_fail_explicitly(change):
    settings = adapter_settings()
    target = np.full((4, 4), .1)
    weights = np.ones((4, 4)) - np.eye(4)
    if 'weights' in change:
        weights = change['weights']
    else:
        settings.update(change)
    with pytest.raises(ValueError, match='off-diagonal targets only'):
        projected_gradient_fit(np.full((4, 2), .1), target, weights, **settings)


@pytest.mark.parametrize('change', [dict(rel_tol=float('nan')), dict(rel_tol=float('inf')),
                                    dict(phi_eff=-1), dict(b0=0), dict(C=float('nan'))])
def test_invalid_numerical_settings_fail_explicitly(change):
    settings = dict(adapter_settings(), **change)
    with pytest.raises(ValueError, match='Invalid projected-gradient'):
        projected_gradient_fit(np.full((4, 2), .1), np.full((4, 4), .1),
                               np.ones((4, 4)) - np.eye(4), **settings)


def test_backend_runs_full_budget_and_is_deterministic():
    settings = adapter_settings()
    target = np.full((4, 4), .1)
    weights = np.ones((4, 4)) - np.eye(4)
    w = np.full((4, 2), .1)
    first = projected_gradient_fit(w, target, weights, **settings)
    second = projected_gradient_fit(w, target, weights, **settings)
    np.testing.assert_array_equal(first[0], second[0])
    diagnostics = first[-1]
    assert diagnostics['iterations'] == settings['max_iter']
    assert not diagnostics['converged']
    assert diagnostics['hit_iteration_cap']
    assert np.max(np.diff(diagnostics['objective_trace'])) <= 1e-12


@pytest.mark.parametrize('tolerance', [float('nan'), float('inf'), 0., -1.])
def test_factor_stage_rejects_nonfinite_or_nonpositive_tolerance(tolerance):
    domain = SimpleNamespace(bail=lambda message: (_ for _ in ()).throw(ValueError(message)))
    with pytest.raises(ValueError, match='positive'):
        factor.run_main_factor_stage(domain, SimpleNamespace(),
                                     SimpleNamespace(max_num_iterations=100, rel_tol=tolerance), {}, {})


def test_projected_backend_cannot_be_silently_ignored_in_annotation_mode():
    domain = SimpleNamespace(bail=lambda message: (_ for _ in ()).throw(ValueError(message)))
    options = SimpleNamespace(max_num_iterations=100, rel_tol=1e-4,
                              symnmf_update_method='projected_gradient', discovery_model='gene_by_annotation')
    with pytest.raises(ValueError, match='gene_by_gene'):
        factor.run_main_factor_stage(domain, SimpleNamespace(), options, {}, {})
