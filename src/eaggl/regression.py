from __future__ import annotations

import numpy as np
import scipy.sparse as sparse

from pegs_cli_errors import DataValidationError
from pegs_shared.regression import (
    compute_beta_tildes as pegs_compute_beta_tildes,
    compute_multivariate_beta_tildes as pegs_compute_multivariate_beta_tildes,
    finalize_regression_outputs as pegs_finalize_regression_outputs,
)


def finalize_regression(beta_tildes, ses, se_inflation_factors, *, log_fn, warn_fn, trace_level):
    return pegs_finalize_regression_outputs(
        beta_tildes,
        ses,
        se_inflation_factors,
        log_fn=log_fn,
        warn_fn=warn_fn,
        trace_level=trace_level,
    )


def compute_beta_tildes(
    state,
    X,
    Y,
    y_var=None,
    scale_factors=None,
    mean_shifts=None,
    resid_correlation_matrix=None,
    *,
    finalize_regression_fn,
    bail_fn,
    log_fun,
    debug_level,
):
    return pegs_compute_beta_tildes(
        X,
        Y,
        y_var=y_var,
        scale_factors=scale_factors,
        mean_shifts=mean_shifts,
        resid_correlation_matrix=resid_correlation_matrix,
        calc_x_shift_scale_fn=state._calc_X_shift_scale,
        finalize_regression_fn=finalize_regression_fn,
        bail_fn=bail_fn,
        log_fun=log_fun,
        debug_level=debug_level,
    )


def compute_multivariate_beta_tildes(
    state,
    X,
    Y,
    resid_correlation_matrix=None,
    add_intercept=True,
    covs=None,
    *,
    finalize_regression_fn,
):
    del state
    return pegs_compute_multivariate_beta_tildes(
        X,
        Y,
        resid_correlation_matrix=resid_correlation_matrix,
        add_intercept=add_intercept,
        covs=covs,
        finalize_regression_fn=finalize_regression_fn,
    )


def compute_robust_betas(
    state,
    X,
    Y,
    resid_correlation_matrix=None,
    covs=None,
    add_intercept=True,
    delta=1.0,
    max_iter=100,
    tol=1e-6,
    rel_tol=0.01,
    *,
    finalize_regression_fn,
    log_fn,
    debug_level,
):
    log_fn("Calculating robust beta tildes", debug_level)

    Y = Y.T
    if len(Y.shape) == 1:
        Y = Y[:, np.newaxis]

    n_phenos = Y.shape[1]
    n_factors = X.shape[1]

    if add_intercept:
        X = np.hstack((X, np.ones((X.shape[0], 1))))

    if covs is not None:
        if len(covs.shape) == 1:
            covs = covs[:, np.newaxis]
        X = np.hstack((X, covs))

    def _huber_loss(residuals, delta_value):
        return np.where(np.abs(residuals) <= delta_value, 0.5 * residuals ** 2, delta_value * (np.abs(residuals) - 0.5 * delta_value))

    def _huber_weight(residuals, delta_value):
        residuals[residuals == 0] = delta_value
        return np.where((np.abs(residuals) > 0) & (np.abs(residuals) <= delta_value), 1, delta_value / np.abs(residuals))

    W = np.linalg.lstsq(X, Y, rcond=None)[0]
    X_x_pheno = np.repeat(X[np.newaxis, :, :], Y.shape[1], axis=0)

    for _iteration in range(max_iter):
        Y_pred = np.dot(X, W)
        residuals = Y - Y_pred
        weights = _huber_weight(residuals, delta)

        X_x_pheno_w = np.multiply(X_x_pheno.T, weights).T
        XTwX = np.einsum("pgf,gh->pfh", X_x_pheno_w, X)

        wY = np.multiply(weights, Y)
        XTwY = np.einsum("pgf,gp->fp", X_x_pheno, wY)

        XTwX_inv = np.linalg.inv(XTwX)
        W_new = np.einsum("phf,fp->hp", XTwX_inv, XTwY)

        if np.linalg.norm(W_new - W, ord="fro") < tol:
            break

        if np.max(np.abs(W_new - W) / (np.abs(W_new) + np.abs(W) + 1e-20)) < rel_tol:
            break

        W = W_new

    Y_pred = np.dot(X, W)
    residuals = Y - Y_pred
    betas = W.T

    n = X.shape[0]
    p = X.shape[1]
    sse = np.sum(_huber_loss(residuals, delta), axis=0)
    sigma2 = sse / (n - p)

    XtX = X.T @ X
    XtX_inv = np.linalg.inv(XtX)
    diag_inv = np.diag(XtX_inv)
    base_ses = np.sqrt(sigma2[:, None] * diag_inv[None, :])

    if resid_correlation_matrix is None:
        ses = base_ses
    else:
        if len(resid_correlation_matrix) != n_phenos:
            raise DataValidationError("resid_correlation_matrix must match number of phenotypes.")

        ses = np.zeros_like(base_ses)

        for p_idx in range(n_phenos):
            R_p = resid_correlation_matrix[p_idx]
            w_vec = np.sqrt(weights[:, p_idx])
            WeightedX = X * w_vec[:, None]

            if sparse.issparse(R_p):
                WeightedX_R = R_p.dot(WeightedX)
            else:
                WeightedX_R = R_p @ WeightedX

            XtRprimeX = WeightedX.T @ WeightedX_R
            var_betas_p = XtX_inv @ XtRprimeX @ XtX_inv
            se_p = np.sqrt(np.diag(var_betas_p))
            ses[p_idx, :] = se_p

    if covs is not None or add_intercept:
        betas = betas[:, :n_factors]
        ses = ses[:, :n_factors]

    return finalize_regression_fn(betas, ses, se_inflation_factors=None)
