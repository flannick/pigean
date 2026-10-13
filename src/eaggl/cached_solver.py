"""Cached line search with conservative fallback and confirmed adaptive stopping.

Stability gates are approximate, not a certificate of a local/global optimum.
"""
from collections import deque
import time

import numpy as np
from .spectral_solver import project_rows
from .spectral_objective import PreciseObjective


class CachedObjective(PreciseObjective):
    def evaluate_cached(self, W):
        gram = W.T @ W
        MW = self.M @ W
        row_energy = np.sum(W * W, axis=1)
        error = (self.target_norm_squared - 2*np.sum(W*MW)
                 + np.sum(gram*gram) - np.sum(row_energy*row_energy))
        if error < 0:
            if error < -1e-10 * max(self.target_norm_squared, 1):
                raise FloatingPointError('Reconstruction error exceeded roundoff tolerance')
            error = 0.
        lambdas = (.5 * np.sum(W*W, axis=0) + self.b0) / self.C
        penalty = self.phi_eff * self.C * np.sum(1 + np.log(lambdas))
        value = .5*error + penalty + self.sparsity*W.sum()
        gradient = (2*(W @ gram - row_energy[:, None]*W - MW)
                    + self.phi_eff*W/lambdas + self.sparsity)
        if (not np.isfinite(value) or not np.isfinite(gradient).all()
                or not np.isfinite(lambdas).all() or (lambdas <= 0).any()):
            raise FloatingPointError('Nonfinite projected-gradient objective or gradient')
        return (float(value), gradient, float(error), lambdas,
                dict(gram=gram, MW=MW, energy=row_energy))

    def cached_change(self, W, D, gradient, old, new):
        G = old['gram']
        A = W.T @ D
        B = D.T @ D
        energy = old['energy']
        diagonal_change = 2*np.sum(W*D, axis=1)+np.sum(D*D, axis=1)
        squared_change = (2*np.sum(G*B)+2*np.trace(A@A)+4*np.trace(A@B)
                          +np.sum(B*B)-np.sum(diagonal_change**2))
        base = .5*np.sum(W*W, axis=0)+self.b0
        lambdas = base/self.C
        data_gradient = gradient-self.phi_eff*W/lambdas-self.sparsity
        # Reuse M@candidate and M@W; near a decision boundary use the original
        # cancellation-controlled M@D calculation instead.
        RD = W@A-energy[:, None]*D-(new['MW']-old['MW'])
        loss_change = np.sum(data_gradient*D)+np.sum(D*RD)+.5*squared_change
        penalty_change = 0.
        if self.phi_eff:
            ratio = (np.sum(W*D, axis=0)+.5*np.sum(D*D, axis=0))/base
            large_drop = ratio < -.5
            terms = np.empty_like(ratio)
            terms[~large_drop] = np.log1p(ratio[~large_drop])
            updated = .5*np.sum((W+D)**2, axis=0)+self.b0
            terms[large_drop] = np.log(updated[large_drop]/base[large_drop])
            penalty_change = self.phi_eff*self.C*np.sum(terms)
        value = float(loss_change+penalty_change+self.sparsity*D.sum())
        # A conservative screening guard, empirically checked against original
        # decisions, not a formal floating-point error certificate.
        guard = (256*np.finfo(float).eps*len(W)*np.linalg.norm(D)
                 *(np.linalg.norm(old['MW'])+np.linalg.norm(new['MW'])
                   +np.sqrt(self.target_norm_squared)*np.linalg.norm(D)))
        return value, float(guard)


def diagnostics(W, gradient, previous):
    norm = max(np.linalg.norm(W), 1e-12)
    mapping = W-project_rows(W-gradient)
    stationarity = float(np.linalg.norm(mapping)/norm)
    if previous is None:
        return dict(gradient_mapping_over_W=stationarity,
                    window_W_change=float('nan'), strong_window_W_change=float('nan'))
    strong = np.argsort(np.sum(W*W, axis=0))[-min(5, W.shape[1]):]
    return dict(gradient_mapping_over_W=stationarity,
        window_W_change=float(np.linalg.norm(W-previous)/norm),
        strong_window_W_change=float(np.max(np.linalg.norm(
            W[:, strong]-previous[:, strong], axis=0)/np.maximum(
                np.linalg.norm(W[:, strong], axis=0), 1e-12))))


class ConfirmationPolicy:
    """Require stable endpoints and a prospective 2,000-update confirmation."""
    def __init__(self, objective, *, confirmation=2000, consecutive=5,
                 stationarity_tol=1e-6, window_tol=1e-4):
        if (confirmation < 1 or consecutive < 1 or
                not np.isfinite([stationarity_tol, window_tol]).all() or
                stationarity_tol <= 0 or window_tol <= 0):
            raise ValueError('Invalid adaptive confirmation settings')
        self.objective = objective
        self.confirmation = confirmation
        self.required = consecutive
        self.stationarity_tol = stationarity_tol
        self.window_tol = window_tol
        self.passes = 0
        self.anchor = None
        self.stop = None
        self.failed_confirmations = 0

    def observe(self, iteration, W, lambdas, value, diagnostic, elapsed):
        passed = (diagnostic['gradient_mapping_over_W'] <= self.stationarity_tol
                  and diagnostic['window_W_change'] <= self.window_tol
                  and diagnostic['strong_window_W_change'] <= self.window_tol)
        self.passes = self.passes+1 if passed else 0
        if not passed:
            self.anchor = None
        if self.stop is not None:
            return
        threshold = min(1e-3, 1.5*self.objective.b0/self.objective.C)
        count = int(((lambdas > threshold) & (W.sum(axis=0) > 1e-3)).sum())
        if self.anchor is None and self.passes >= self.required:
            self.anchor = dict(iteration=iteration, W=W.copy(), value=value,
                               count=count)
        if self.anchor is None or iteration-self.anchor['iteration'] < self.confirmation:
            return
        old = self.anchor
        strong = np.argsort(np.sum(old['W']**2, axis=0))[-min(5, W.shape[1]):]
        drift = float(np.linalg.norm(W-old['W'])/max(np.linalg.norm(old['W']), 1e-12))
        strong_drift = float(np.max(np.linalg.norm(W[:, strong]-old['W'][:, strong], axis=0)
            /np.maximum(np.linalg.norm(old['W'][:, strong], axis=0), 1e-12)))
        gain = float((old['value']-value)/max(abs(old['value']), 1.))
        if (self.passes >= self.required and drift <= 1e-3 and strong_drift <= 1e-3
                and -1e-12 <= gain <= 1e-6 and count == old['count']):
            self.stop = dict(iteration=iteration, W=W.copy(), solver_seconds=elapsed,
                             confirmation_W_change=drift,
                             confirmation_strong_W_change=strong_drift,
                             confirmation_objective_gain=gain)
        else:
            self.failed_confirmations += 1
            self.anchor = dict(iteration=iteration, W=W.copy(), value=value, count=count)


def fit(objective, initial, *, max_iter=20000, check_every=25, window=1000,
        cache_line_search=True, observe_decisions=False, checkpoints=(),
        stop_adaptively=False, stationarity_tol=1e-6, window_tol=1e-4,
        confirmation=2000, required_checks=5, log=None):
    initial = np.asarray(initial, dtype=float)
    if (initial.ndim != 2 or initial.shape[0] != objective.M.shape[0]
            or initial.shape[1] < 1 or not np.isfinite(initial).all()):
        raise ValueError('Invalid initial loading matrix')
    if (any(not isinstance(v, (int, np.integer)) or isinstance(v, bool) or v < 1
            for v in (max_iter, window, check_every, confirmation, required_checks))
            or window % check_every):
        raise ValueError('Positive budgets and a window divisible by check interval required')
    W = project_rows(initial)
    value, gradient, error, lambdas, cache = objective.evaluate_cached(W)
    history = deque([(0, W.copy())], maxlen=window//check_every+1)
    step = 1/max(1, 6*np.linalg.norm(W.T @ W, ord=2)+objective.phi_eff/lambdas.min())
    policy = ConfirmationPolicy(objective, confirmation=confirmation,
        consecutive=required_checks, stationarity_tol=stationarity_tol, window_tol=window_tol)
    traces = [dict(iteration=0, objective=value, solver_seconds=0.,
                   **diagnostics(W, gradient, None))]
    snapshots = {0: W.copy()}
    fallback_count = decision_checks = decision_mismatches = 0
    total_backtracks = 0
    started = time.perf_counter()
    status = 'iteration_cap'
    previous_lambdas = lambdas
    for iteration in range(1, max_iter+1):
        accepted = False
        for backtrack in range(60):
            candidate = project_rows(W-step*gradient)
            delta = candidate-W
            trial_value, trial_gradient, trial_error, trial_lambda, trial_cache = (
                objective.evaluate_cached(candidate))
            directional = float(np.sum(gradient*delta))
            slack = objective.descent_slack(value)
            bound = 1e-4*directional+slack
            if cache_line_search:
                change, guard = objective.cached_change(W, delta, gradient, cache, trial_cache)
                if not np.isfinite(change) or not np.isfinite(guard) or abs(change-bound) <= guard:
                    change = objective.change(W, delta, gradient)
                    fallback_count += 1
            else:
                change = objective.change(W, delta, gradient)
            descent = change <= bound
            if observe_decisions:
                original = objective.change(W, delta, gradient)
                decision_checks += 1
                decision_mismatches += int((original <= bound) != descent)
            if descent:
                accepted = True
                break
            step *= .5
        if not accepted:
            status = 'line_search_failed'
            break
        total_backtracks += backtrack
        y = trial_gradient-gradient
        sy = float(np.sum(delta*y))
        spectral_step = float(np.sum(delta*delta))/sy if sy > 1e-20 else step*1.2
        step = float(np.clip(spectral_step, 1e-12, 1e4))
        previous_lambdas = lambdas
        W, value, gradient, error, lambdas, cache = (
            candidate, trial_value, trial_gradient, trial_error, trial_lambda, trial_cache)
        if iteration % check_every == 0:
            history.append((iteration, W.copy()))
            previous = history[0][1] if iteration-history[0][0] == window else None
            diagnostic = diagnostics(W, gradient, previous)
            elapsed = time.perf_counter()-started
            policy.observe(iteration, W, lambdas, value, diagnostic, elapsed)
            traces.append(dict(iteration=iteration, objective=value,
                               solver_seconds=elapsed, **diagnostic))
        if iteration in checkpoints:
            snapshots[iteration] = W.copy()
        if stop_adaptively and policy.stop is not None:
            status = 'confirmed_stability'
            break
        if log and (iteration == 1 or iteration % 500 == 0):
            log('iteration=%d objective=%.8g stopping=%s' % (
                iteration, value, 'confirming' if policy.anchor is not None else 'checking'))
    elapsed = time.perf_counter()-started
    stop = policy.stop
    if stop is None:
        stop = dict(iteration=iteration, W=W.copy(), solver_seconds=elapsed)
    final_diagnostic = diagnostics(W, gradient, history[0][1]
        if iteration-history[0][0] >= window else None)
    if traces[-1]['iteration'] != iteration:
        traces.append(dict(iteration=iteration, objective=value,
                           solver_seconds=elapsed, **final_diagnostic))
    summary = dict(status=status, iterations=iteration, solver_seconds=elapsed,
        adaptive_iterations=stop['iteration'], adaptive_seconds=stop['solver_seconds'],
        adaptive_confirmed=policy.stop is not None,
        failed_confirmations=policy.failed_confirmations,
        precise_fallbacks=fallback_count, total_backtracks=total_backtracks,
        observed_line_search_decisions=decision_checks,
        line_search_decision_mismatches=decision_mismatches,
        objective=value, reconstruction_error=error,
        accepted_updates=iteration - int(status == 'line_search_failed'),
        scale_change=float(np.max(np.abs(lambdas-previous_lambdas)
            /np.maximum(previous_lambdas, 1e-50))),
        **final_diagnostic)
    return W, lambdas, summary, traces, snapshots, stop
