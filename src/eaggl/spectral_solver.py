"""Spectral projected-gradient solver for nonnegative off-diagonal SymNMF.

This optimizes the currently logged profiled objective, not the legacy update's
implicit doubled penalty. The EAGGL adapter is opt-in and uses a fixed budget.
"""
from collections import deque
import numpy as np


def project_rows(values):
    """Euclidean projection onto nonnegative rows with sum at most one."""
    result = np.maximum(values, 0)
    selected = result.sum(axis=1) > 1
    if selected.any():
        rows = values[selected]
        ordered = np.sort(rows, axis=1)[:, ::-1]
        cumulative = np.cumsum(ordered, axis=1) - 1
        ranks = np.arange(1, rows.shape[1] + 1)
        rho = (ordered - cumulative / ranks > 0).sum(axis=1) - 1
        theta = cumulative[np.arange(len(rows)), rho] / (rho + 1)
        result[selected] = np.maximum(rows - theta[:, None], 0)
    # Sorting/cumulative sums can overshoot the simplex by a few ulps.
    totals = result.sum(axis=1)
    overflow = totals > 1
    result[overflow] *= ((1 - 4*np.finfo(float).eps) / totals[overflow])[:, None]
    return result


class OffDiagonalObjective:
    def __init__(self, target, *, phi=.01, a0=10., K=30, sparsity=0.):
        raw = np.asarray(target, dtype=float)
        if raw.ndim != 2 or raw.shape[0] == 0 or raw.shape[0] != raw.shape[1]:
            raise ValueError('Target must be a nonempty square matrix')
        if not np.isfinite(raw).all() or (raw < 0).any():
            raise ValueError('Target must be finite and nonnegative')
        if not np.allclose(raw,raw.T,atol=1e-12,rtol=1e-12):
            raise ValueError('Target must be symmetric')
        if K < 1 or phi < 0 or a0 <= 1 or sparsity < 0:
            raise ValueError('Invalid rank or nonnegative regularization parameters')
        self.M = raw.copy()
        np.fill_diagonal(self.M, 0)
        self.target_norm_squared = float(np.sum(self.M ** 2))
        self.phi_eff = float(raw.std() ** 2 * phi)
        self.C = raw.shape[0] / 2 + a0 + 1
        self.b0 = max(3.14 * (a0 - 1) * float(raw.mean()) / (2*K), 1e-50)
        self.sparsity = sparsity
        self.initial_scale = max(float(raw.max()), 1e-6)

    def evaluate(self, W):
        # Trace identities avoid forming W W^T on every line-search trial.
        gram = W.T @ W
        MW = self.M @ W
        row_energy = np.sum(W * W, axis=1)
        error = self.target_norm_squared - 2*np.sum(W*MW) + np.sum(gram*gram) - np.sum(row_energy*row_energy)
        if error < 0:
            assert error >= -1e-10 * max(self.target_norm_squared, 1)
            error = 0.
        lambdas = (.5 * np.sum(W*W, axis=0) + self.b0) / self.C
        penalty = self.phi_eff * self.C * np.sum(1 + np.log(lambdas))
        value = .5*error + penalty + self.sparsity*W.sum()
        gradient = 2*(W @ gram - row_energy[:, None]*W - MW) + self.phi_eff*W/lambdas + self.sparsity
        return float(value), gradient, float(error), lambdas


def fit(objective, initial, *, max_iter=10000, stationarity_tol=1e-4,
        window=100, window_tol=1e-3, required_checks=5, check_every=25,
        log=None):
    initial=np.asarray(initial,dtype=float)
    if initial.ndim!=2 or initial.shape[0]!=objective.M.shape[0] or initial.shape[1]<1 or not np.isfinite(initial).all():
        raise ValueError('Invalid initial loading matrix')
    if min(max_iter,window,required_checks,check_every)<1:
        raise ValueError('Iteration and diagnostic counts must be positive')
    W = project_rows(initial)
    value, gradient, error, lambdas = objective.evaluate(W)
    history = deque([W.copy()], maxlen=window + 1)
    snapshots = {0: W.copy()}
    traces = []
    step = 1/max(1, 6*np.linalg.norm(W.T @ W, ord=2) + objective.phi_eff/lambdas.min())
    consecutive_passes = 0
    status = 'iteration_cap'
    for iteration in range(1, max_iter + 1):
        accepted = False
        for backtrack in range(60):
            candidate = project_rows(W - step*gradient)
            delta = candidate - W
            trial_value, trial_gradient, trial_error, trial_lambda = objective.evaluate(candidate)
            directional = float(np.sum(gradient * delta))
            slack = objective.descent_slack(value) if hasattr(objective, 'descent_slack') else 0.
            change = (objective.change(W, delta, gradient)
                      if hasattr(objective, 'change') else trial_value-value)
            descent = change <= 1e-4 * directional + slack
            if descent:
                accepted = True
                break
            step *= .5
        if not accepted:
            status = 'line_search_failed'
            break
        relative_step = float(np.linalg.norm(delta)/max(np.linalg.norm(candidate),1e-12))
        scale_change = float(np.max(np.abs(trial_lambda-lambdas)/np.maximum(lambdas,1e-50)))
        y = trial_gradient-gradient
        sy = float(np.sum(delta*y))
        spectral_step = float(np.sum(delta*delta))/sy if sy > 1e-20 else step*1.2
        step = float(np.clip(spectral_step,1e-12,1e4))
        W, value, gradient, error, lambdas = candidate, trial_value, trial_gradient, trial_error, trial_lambda
        history.append(W.copy())
        mapping = W-project_rows(W-gradient)
        stationarity = float(np.linalg.norm(mapping)/max(np.linalg.norm(W),1e-12))
        window_change = (float(np.linalg.norm(W-history[0])/max(np.linalg.norm(W),1e-12))
                         if len(history)==window+1 else np.nan)
        strong = np.argsort(np.sum(W*W,axis=0))[-min(5,W.shape[1]):]
        strong_window = (float(np.max(np.linalg.norm(W[:,strong]-history[0][:,strong],axis=0) /
                            np.maximum(np.linalg.norm(W[:,strong],axis=0),1e-12)))
                         if len(history)==window+1 else np.nan)
        if iteration % check_every == 0:
            passed = stationarity <= stationarity_tol and window_change <= window_tol and strong_window <= window_tol
            consecutive_passes = consecutive_passes+1 if passed else 0
        traces.append(dict(iteration=iteration, objective=value, reconstruction_error=error,
                           gradient_mapping_over_W=stationarity, window_W_change=window_change,
                           strong_window_W_change=strong_window, relative_step=relative_step,
                           scale_change=scale_change, line_search_backtracks=backtrack,
                           armijo_roundoff_slack=slack,
                           objective_change=change, armijo_bound=1e-4*directional+slack,
                           lambda_above_output_threshold=int((lambdas>1e-3).sum())))
        if iteration in (100,500,1000,2500,5000,10000):
            snapshots[iteration] = W.copy()
        if log and (iteration == 1 or iteration%500 == 0):
            log(f'iteration={iteration} objective={value:.8g} stationarity={stationarity:.4g} '
                f'window_change={window_change:.4g} strong_window={strong_window:.4g}')
        if consecutive_passes >= required_checks:
            status = 'passed_stationarity_and_window_checks'
            break
    value, gradient, error, lambdas = objective.evaluate(W)
    mapping = W-project_rows(W-gradient)
    final_stationarity = float(np.linalg.norm(mapping)/max(np.linalg.norm(W),1e-12))
    if traces:
        final_window = traces[-1]['window_W_change']
        final_strong = traces[-1]['strong_window_W_change']
    else:
        final_window = final_strong = float('nan')
    snapshots[iteration] = W.copy()
    summary = dict(status=status, iterations=iteration, accepted_updates=len(traces), objective=value,
                   relative_squared_reconstruction_error=(error/objective.target_norm_squared
                       if objective.target_norm_squared else (0. if error==0 else float('inf'))),
                   gradient_mapping_over_W=final_stationarity, window_W_change=final_window,
                   strong_window_W_change=final_strong, consecutive_passes=consecutive_passes,
                   scale_change=traces[-1]['scale_change'] if traces else float('nan'),
                   lambda_above_output_threshold=int((lambdas>1e-3).sum()),
                   max_row_sum=float(W.sum(axis=1).max()), min_loading=float(W.min()),
                   objective_increases=int(np.sum(np.diff([r['objective'] for r in traces])>1e-8)))
    return W, lambdas, summary, traces, snapshots
