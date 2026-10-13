"""Profiled SymNMF objective and optimized numerical adapter."""
import numpy as np


def project_nonnegative_rows(values, row_sum_cap=True):
    result = np.maximum(np.asarray(values, dtype=float), 0.0)
    if not row_sum_cap:
        return result
    selected = np.sum(result, axis=1) > 1.0
    if np.any(selected):
        rows = np.asarray(values, dtype=float)[selected]
        ordered = np.sort(rows, axis=1)[:, ::-1]
        cumulative = np.cumsum(ordered, axis=1) - 1.0
        ranks = np.arange(1, rows.shape[1] + 1)
        rho = np.sum(ordered - cumulative / ranks > 0.0, axis=1) - 1
        theta = cumulative[np.arange(len(rows)), rho] / (rho + 1)
        result[selected] = np.maximum(rows - theta[:, None], 0.0)
    return result


def profiled_objective(W, target, weights, phi_eff, C, b0, sparsity, views=None):
    lambdas = (0.5 * np.sum(W * W, axis=0) + b0) / C
    reconstruction = W @ W.T
    if views is None:
        error = float(np.sum(weights * np.square(target - reconstruction)))
    else:
        error = float(np.mean(np.sum(weights[None] * np.square(views - reconstruction[None]), axis=(1, 2))))
    penalty = phi_eff * np.sum((0.5 * np.sum(W * W, axis=0) + b0) / lambdas + C * np.log(lambdas))
    objective = 0.5 * error + penalty + sparsity * np.sum(W)
    return float(objective), float(error), lambdas


def profiled_gradient(W, target, weights, phi_eff, C, b0, sparsity):
    lambdas = (0.5 * np.sum(W * W, axis=0) + b0) / C
    residual = weights * (W @ W.T - target)
    return (residual + residual.T) @ W + phi_eff * W / lambdas[None] + sparsity


def projected_gradient_fit(W, target, weights, *, phi_eff, C, b0, sparsity,
                           row_sum_cap, max_iter, rel_tol, views=None, log_fn=None,
                           stopping='adaptive'):
    """Optimize the logged objective; adaptive stability is an approximate gate."""
    from .cached_solver import CachedObjective, fit

    expected = np.ones_like(target)
    np.fill_diagonal(expected, 0.0)
    if views is not None or not row_sum_cap or not np.array_equal(weights, expected):
        raise ValueError('Projected-gradient SymNMF supports row-capped, single-view, off-diagonal targets only')
    settings = np.array([phi_eff, C, b0, sparsity, rel_tol], dtype=float)
    if not np.isfinite(settings).all() or phi_eff < 0 or C <= 0 or b0 <= 0 or sparsity < 0 or rel_tol <= 0:
        raise ValueError('Invalid projected-gradient SymNMF settings')
    if stopping not in ('adaptive', 'fixed'):
        raise ValueError('Unknown projected-gradient stopping policy')
    objective = CachedObjective(target, K=W.shape[1], phi=0.0, sparsity=sparsity)
    objective.phi_eff = float(phi_eff)
    objective.C = float(C)
    objective.b0 = float(b0)
    initial = W.copy()
    W, lambdas, summary, trace, _, _ = fit(
        objective, W, max_iter=max_iter, stationarity_tol=rel_tol,
        stop_adaptively=stopping == 'adaptive', log=log_fn)
    endpoint_passed = bool(summary['status'] != 'line_search_failed' and
                     summary['gradient_mapping_over_W'] <= rel_tol and
                     summary['window_W_change'] <= 1e-4 and summary['strong_window_W_change'] <= 1e-4)
    converged = endpoint_passed and (stopping == 'fixed' or summary['adaptive_confirmed'])
    diagnostics = dict(
        iterations=summary['iterations'], converged=converged,
        hit_iteration_cap=bool(summary['iterations'] >= max_iter and not converged),
        failed_line_search=summary['status'] == 'line_search_failed',
        final_delambda=summary['scale_change'],
        projected_gradient_relative=summary['gradient_mapping_over_W'],
        objective_trace=[row['objective'] for row in trace],
        objective_initial=objective.evaluate(initial)[0], objective_final=summary['objective'],
        stopping_policy=stopping, stopping_status=summary['status'],
        endpoint_passed=endpoint_passed, adaptive_confirmed=summary['adaptive_confirmed'],
        failed_confirmations=summary['failed_confirmations'],
        window_W_change=summary['window_W_change'],
        strong_window_W_change=summary['strong_window_W_change'])
    error = objective.evaluate(W)[2]
    return W, 0.5 * error, summary['objective'], lambdas, error, diagnostics
