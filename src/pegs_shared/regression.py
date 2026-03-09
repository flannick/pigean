import copy

import numpy as np
import scipy.linalg
import scipy.stats
import scipy.sparse as sparse

from pegs_shared.cli import _default_bail


def finalize_regression_outputs(beta_tildes, ses, se_inflation_factors, *, log_fn=None, warn_fn=None, trace_level=0):
    if se_inflation_factors is not None:
        ses *= se_inflation_factors

    if np.prod(ses.shape) > 0:
        empty_mask = np.logical_and(beta_tildes == 0, ses <= 0)
        max_se = np.max(ses)

        if np.sum(empty_mask) > 0 and log_fn is not None:
            log_fn("Zeroing out %d betas due to negative ses" % (np.sum(empty_mask)), trace_level)

        ses[empty_mask] = max_se * 100 if max_se > 0 else 100
        beta_tildes[ses <= 0] = 0

    z_scores = np.zeros(beta_tildes.shape)
    ses_positive_mask = ses > 0
    z_scores[ses_positive_mask] = beta_tildes[ses_positive_mask] / ses[ses_positive_mask]
    if np.any(~ses_positive_mask) and warn_fn is not None:
        warn_fn("There were %d gene sets with negative ses; setting z-scores to 0" % (np.sum(~ses_positive_mask)))
    p_values = 2 * scipy.stats.norm.cdf(-np.abs(z_scores))
    return (beta_tildes, ses, z_scores, p_values, se_inflation_factors)


def compute_beta_tildes(
    X,
    Y,
    *,
    y_var=None,
    scale_factors=None,
    mean_shifts=None,
    resid_correlation_matrix=None,
    calc_x_shift_scale_fn=None,
    finalize_regression_fn=None,
    bail_fn=None,
    log_fun=None,
    debug_level=0,
):
    if finalize_regression_fn is None:
        finalize_regression_fn = finalize_regression_outputs
    if bail_fn is None:
        bail_fn = _default_bail
    if log_fun is None:
        log_fun = lambda *args, **kwargs: None

    log_fun("Calculating beta tildes")

    if X.shape[0] == 0 or X.shape[1] == 0:
        bail_fn("Can't compute beta tildes on no gene sets!")

    if len(Y.shape) == 2:
        len_Y = Y.shape[1]
        Y_mean = np.mean(Y, axis=1, keepdims=True)
    else:
        len_Y = Y.shape[0]
        Y_mean = np.mean(Y)

    if mean_shifts is None or scale_factors is None:
        if calc_x_shift_scale_fn is None:
            if sparse.issparse(X):
                mean_shifts = np.array(X.mean(axis=0)).ravel()
                scale_factors = np.sqrt(np.array(X.multiply(X).mean(axis=0)).ravel() - np.square(mean_shifts))
            else:
                mean_shifts = np.mean(X, axis=0)
                scale_factors = np.std(X, axis=0)
        else:
            (mean_shifts, scale_factors) = calc_x_shift_scale_fn(X)

    if y_var is None:
        if len(Y.shape) == 1:
            y_var = np.var(Y)
        else:
            y_var = np.var(Y, axis=1)

    if sparse.issparse(X):
        X_sum = X.sum(axis=0).A1.T[:, np.newaxis]
    else:
        X_sum = np.asarray(X.sum(axis=0, keepdims=True).T)

    if len(Y.shape) == 1:
        X_sum = X_sum.squeeze(axis=1)

    dot_product = (X.T.dot(Y.T) - X_sum * Y_mean.T).T / len_Y

    variances = np.power(scale_factors, 2)
    variances[variances == 0] = 1

    beta_tildes = scale_factors * dot_product / variances

    if len(Y.shape) == 2:
        ses = np.outer(np.sqrt(y_var), scale_factors)
    else:
        ses = np.sqrt(y_var) * scale_factors

    if len_Y > 1:
        ses /= (np.sqrt(variances * (len_Y - 1)))

    se_inflation_factors = None
    if resid_correlation_matrix is not None:
        log_fun("Adjusting standard errors for correlations", debug_level)

        if type(resid_correlation_matrix) is list:
            resid_correlation_matrix_list = resid_correlation_matrix
            assert len(resid_correlation_matrix_list) == beta_tildes.shape[0]
        else:
            resid_correlation_matrix_list = [resid_correlation_matrix]

        se_inflation_factors = np.zeros(beta_tildes.shape)

        for i in range(len(resid_correlation_matrix_list)):
            r_X = resid_correlation_matrix_list[i].dot(X)
            if sparse.issparse(X):
                r_X_col_means = r_X.multiply(X).sum(axis=0).A1 / X.shape[0]
            else:
                r_X_col_means = np.sum(r_X * X, axis=0) / X.shape[0]

            cor_variances = r_X_col_means - np.square(r_X_col_means)
            cor_variances[cor_variances < variances] = variances[cor_variances < variances]
            cur_se_inflation_factors = np.sqrt(cor_variances / variances)

            if len(resid_correlation_matrix_list) == 1:
                se_inflation_factors = cur_se_inflation_factors
                if len(beta_tildes.shape) == 2:
                    se_inflation_factors = np.tile(se_inflation_factors, beta_tildes.shape[0]).reshape(beta_tildes.shape)
                break
            else:
                se_inflation_factors[i, :] = cur_se_inflation_factors

    return finalize_regression_fn(beta_tildes, ses, se_inflation_factors)


def compute_logistic_beta_tildes(
    X,
    Y,
    *,
    scale_factors=None,
    mean_shifts=None,
    resid_correlation_matrix=None,
    convert_to_dichotomous=True,
    rel_tol=0.01,
    X_stacked=None,
    append_pseudo=True,
    calc_x_shift_scale_fn=None,
    finalize_regression_fn=None,
    bail_fn=None,
    log_fun=None,
    debug_level=0,
    trace_level=0,
    runtime_Y=None,
    runtime_Y_for_regression=None,
):
    if finalize_regression_fn is None:
        finalize_regression_fn = finalize_regression_outputs
    if bail_fn is None:
        bail_fn = _default_bail
    if log_fun is None:
        log_fun = lambda *args, **kwargs: None

    log_fun("Calculating logistic beta tildes")

    if X.shape[0] == 0 or X.shape[1] == 0:
        bail_fn("Can't compute beta tildes on no gene sets!")

    if runtime_Y is not None and (Y is runtime_Y or Y is runtime_Y_for_regression):
        Y = copy.copy(Y)

    if mean_shifts is None or scale_factors is None:
        if calc_x_shift_scale_fn is None:
            if sparse.issparse(X):
                mean_shifts = np.array(X.mean(axis=0)).ravel()
                scale_factors = np.sqrt(np.array(X.multiply(X).mean(axis=0)).ravel() - np.square(mean_shifts))
            else:
                mean_shifts = np.mean(X, axis=0)
                scale_factors = np.std(X, axis=0)
        else:
            (mean_shifts, scale_factors) = calc_x_shift_scale_fn(X)

    if len(Y.shape) == 1:
        orig_vector = True
        Y = Y[np.newaxis, :]
    else:
        orig_vector = False

    if convert_to_dichotomous:
        if np.sum(np.logical_and(Y != 0, Y != 1)) > 0:
            Y[np.isnan(Y)] = 0
            mult_sum = 1
            Y_sums = np.sum(Y, axis=1).astype(int) * mult_sum
            Y_sorted = np.sort(Y, axis=1)[:, ::-1]
            threshold_val = np.diag(Y_sorted[:, Y_sums])

            true_mask = (Y.T > threshold_val).T
            Y[true_mask] = 1
            Y[~true_mask] = 0
            log_fun("Converting values to dichotomous outcomes; y=1 for input y > %s" % threshold_val, debug_level)

    log_fun("Outcomes: %d=1, %d=0; mean=%.3g" % (np.sum(Y == 1), np.sum(Y == 0), np.mean(Y)), trace_level)

    if np.var(Y) == 0:
        bail_fn("Error: need at least one sample with a different outcome")

    len_Y = Y.shape[1]
    num_chains = Y.shape[0]

    if append_pseudo:
        log_fun("Appending pseudo counts", trace_level)
        Y_means = np.mean(Y, axis=1)[:, np.newaxis]
        Y = np.hstack((Y, Y_means))
        X = sparse.csc_matrix(sparse.vstack((X, sparse.csr_matrix(np.ones((1, X.shape[1]))))))

        if X_stacked is not None:
            X_stacked = sparse.csc_matrix(sparse.vstack((X_stacked, sparse.csr_matrix(np.ones((1, X_stacked.shape[1]))))))

    if X_stacked is None:
        if num_chains > 1:
            X_stacked = sparse.hstack([X] * num_chains)
        else:
            X_stacked = X

    num_non_zero = np.tile((X != 0).sum(axis=0).A1, num_chains)
    num_zero = X_stacked.shape[0] - num_non_zero

    beta_tildes = np.zeros(X.shape[1] * num_chains)
    alpha_tildes = np.zeros(X.shape[1] * num_chains)
    it = 0
    compute_mask = np.full(len(beta_tildes), True)
    diverged_mask = np.full(len(beta_tildes), False)

    def __compute_Y_R(_X, _beta_tildes, _alpha_tildes, max_cap=0.999):
        exp_X_stacked_beta_alpha = _X.multiply(_beta_tildes)
        exp_X_stacked_beta_alpha.data += (_X != 0).multiply(_alpha_tildes).data
        max_val = 100
        overflow_mask = exp_X_stacked_beta_alpha.data > max_val
        exp_X_stacked_beta_alpha.data[overflow_mask] = max_val
        np.exp(exp_X_stacked_beta_alpha.data, out=exp_X_stacked_beta_alpha.data)

        Y_pred = copy.copy(exp_X_stacked_beta_alpha)
        Y_pred.data = Y_pred.data / (1 + Y_pred.data)
        Y_pred.data[Y_pred.data > max_cap] = max_cap
        R = copy.copy(Y_pred)
        R.data = Y_pred.data * (1 - Y_pred.data)
        return (Y_pred, R)

    def __compute_Y_R_zero(_alpha_tildes):
        Y_pred_zero = np.exp(_alpha_tildes)
        Y_pred_zero = Y_pred_zero / (1 + Y_pred_zero)
        R_zero = Y_pred_zero * (1 - Y_pred_zero)
        return (Y_pred_zero, R_zero)

    max_it = 100
    log_fun("Performing IRLS...")
    while True:
        it += 1
        prev_beta_tildes = copy.copy(beta_tildes)
        prev_alpha_tildes = copy.copy(alpha_tildes)

        (Y_pred, R) = __compute_Y_R(X_stacked[:, compute_mask], beta_tildes[compute_mask], alpha_tildes[compute_mask])

        max_val = 100
        overflow_mask = alpha_tildes > max_val
        alpha_tildes[overflow_mask] = max_val

        (Y_pred_zero, R_zero) = __compute_Y_R_zero(alpha_tildes[compute_mask])

        Y_sum_per_chain = np.sum(Y, axis=1)
        Y_sum = np.tile(Y_sum_per_chain, X.shape[1])

        X_r_X_beta = X_stacked[:, compute_mask].power(2).multiply(R).sum(axis=0).A1.ravel()
        X_r_X_alpha = R.sum(axis=0).A1.ravel() + R_zero * num_zero[compute_mask]
        X_r_X_beta_alpha = X_stacked[:, compute_mask].multiply(R).sum(axis=0).A1.ravel()
        denom = X_r_X_beta * X_r_X_alpha - np.square(X_r_X_beta_alpha)

        diverged = np.logical_or(np.logical_or(X_r_X_beta == 0, X_r_X_beta_alpha == 0), denom == 0)

        if np.sum(diverged) > 0:
            log_fun("%d beta_tildes diverged" % np.sum(diverged), trace_level)
            not_diverged = ~diverged
            cur_indices = np.where(compute_mask)[0]
            compute_mask[cur_indices[diverged]] = False
            diverged_mask[cur_indices[diverged]] = True

            Y_pred = sparse.csc_matrix(Y_pred)
            R = sparse.csc_matrix(R)
            Y_pred = Y_pred[:, not_diverged]
            R = R[:, not_diverged]
            Y_pred_zero = Y_pred_zero[not_diverged]
            R_zero = R_zero[not_diverged]
            X_r_X_beta = X_r_X_beta[not_diverged]
            X_r_X_alpha = X_r_X_alpha[not_diverged]
            X_r_X_beta_alpha = X_r_X_beta_alpha[not_diverged]
            denom = denom[not_diverged]

        if np.sum(np.isnan(X_r_X_beta) | np.isnan(X_r_X_alpha) | np.isnan(X_r_X_beta_alpha)) > 0:
            bail_fn("Error: something went wrong")

        R_inv_Y_T_beta = X_stacked[:, compute_mask].multiply(Y_pred).sum(axis=0).A1.ravel() - X.T.dot(Y.T).T.ravel()[compute_mask]
        R_inv_Y_T_alpha = (Y_pred.sum(axis=0).A1.ravel() + Y_pred_zero * num_zero[compute_mask]) - Y_sum[compute_mask]

        beta_tilde_row = (X_r_X_beta * prev_beta_tildes[compute_mask] + X_r_X_beta_alpha * prev_alpha_tildes[compute_mask] - R_inv_Y_T_beta)
        alpha_tilde_row = (X_r_X_alpha * prev_alpha_tildes[compute_mask] + X_r_X_beta_alpha * prev_beta_tildes[compute_mask] - R_inv_Y_T_alpha)

        beta_tildes[compute_mask] = (X_r_X_alpha * beta_tilde_row - X_r_X_beta_alpha * alpha_tilde_row) / denom
        alpha_tildes[compute_mask] = (X_r_X_beta * alpha_tilde_row - X_r_X_beta_alpha * beta_tilde_row) / denom

        diff = np.abs(beta_tildes - prev_beta_tildes)
        diff_denom = np.abs(beta_tildes + prev_beta_tildes)
        diff_denom[diff_denom == 0] = 1
        rel_diff = diff / diff_denom

        compute_mask[np.logical_or(rel_diff < rel_tol, beta_tildes == 0)] = False
        if np.sum(compute_mask) == 0:
            log_fun("Converged after %d iterations" % it, trace_level)
            break
        if it == max_it:
            log_fun("Stopping with %d still not converged" % np.sum(compute_mask), trace_level)
            diverged_mask[compute_mask] = True
            break

    while True:
        if np.sum(diverged_mask) > 0:
            beta_tildes[diverged_mask] = 0
            alpha_tildes[diverged_mask] = Y_sum[diverged_mask] / len_Y

        max_coeff = 100
        (Y_pred, V) = __compute_Y_R(X_stacked, beta_tildes, alpha_tildes)

        params_too_large_mask = np.logical_or(np.abs(alpha_tildes) > max_coeff, np.abs(beta_tildes) > max_coeff)
        alpha_tildes[np.abs(alpha_tildes) > max_coeff] = max_coeff

        p_const = np.exp(alpha_tildes) / (1 + np.exp(alpha_tildes))
        variance_denom = (V.sum(axis=0).A1 + p_const * (1 - p_const) * (len_Y - (X_stacked != 0).sum(axis=0).A1))
        denom_zero = variance_denom == 0
        variance_denom[denom_zero] = 1

        variances = X_stacked.power(2).multiply(V).sum(axis=0).A1 - np.power(X_stacked.multiply(V).sum(axis=0).A1, 2) / variance_denom
        variances[denom_zero] = 100

        additional_diverged_mask = np.logical_and(~diverged_mask, np.logical_or(np.logical_or(variances < 0, denom_zero), params_too_large_mask))
        if np.sum(additional_diverged_mask) > 0:
            diverged_mask = np.logical_or(diverged_mask, additional_diverged_mask)
        else:
            break

    se_inflation_factors = None
    if resid_correlation_matrix is not None:
        if type(resid_correlation_matrix) is list:
            raise NotImplementedError("Vectorized correlations not yet implemented for logistic regression")

        if append_pseudo:
            resid_correlation_matrix = sparse.hstack((resid_correlation_matrix, np.zeros(resid_correlation_matrix.shape[0])[:, np.newaxis]))
            new_bottom_row = np.zeros((1, resid_correlation_matrix.shape[1]))
            new_bottom_row[0, -1] = 1
            resid_correlation_matrix = sparse.vstack((resid_correlation_matrix, new_bottom_row)).tocsc()

        cor_variances = copy.copy(variances)
        r_X = resid_correlation_matrix.dot(X)
        r_X = (X != 0).multiply(r_X)

        cor_variances = sparse.hstack([r_X.multiply(X)] * num_chains).multiply(V).sum(axis=0).A1 - sparse.hstack([r_X] * num_chains).multiply(V).sum(axis=0).A1 / (V.sum(axis=0).A1 + p_const * (1 - p_const) * (len_Y - (X_stacked != 0).sum(axis=0).A1))
        variances[variances == 0] = 1
        se_inflation_factors = np.sqrt(cor_variances / variances)

    if num_chains > 1:
        beta_tildes = beta_tildes.reshape(num_chains, X.shape[1])
        alpha_tildes = alpha_tildes.reshape(num_chains, X.shape[1])
        variances = variances.reshape(num_chains, X.shape[1])
        diverged_mask = diverged_mask.reshape(num_chains, X.shape[1])
        if se_inflation_factors is not None:
            se_inflation_factors = se_inflation_factors.reshape(num_chains, X.shape[1])
    else:
        beta_tildes = beta_tildes[np.newaxis, :]
        alpha_tildes = alpha_tildes[np.newaxis, :]
        variances = variances[np.newaxis, :]
        diverged_mask = diverged_mask[np.newaxis, :]
        if se_inflation_factors is not None:
            se_inflation_factors = se_inflation_factors[np.newaxis, :]

    variances[:, scale_factors == 0] = 1
    beta_tildes = scale_factors * beta_tildes
    variances[variances == 0] = 1e-10
    ses = scale_factors / np.sqrt(variances)

    if orig_vector:
        beta_tildes = np.squeeze(beta_tildes, axis=0)
        alpha_tildes = np.squeeze(alpha_tildes, axis=0)
        variances = np.squeeze(variances, axis=0)
        ses = np.squeeze(ses, axis=0)
        diverged_mask = np.squeeze(diverged_mask, axis=0)

        if se_inflation_factors is not None:
            se_inflation_factors = np.squeeze(se_inflation_factors, axis=0)

    return finalize_regression_fn(beta_tildes, ses, se_inflation_factors) + (alpha_tildes, diverged_mask)


def correct_beta_tildes(
    runtime,
    beta_tildes,
    ses,
    se_inflation_factors,
    total_qc_metrics,
    total_qc_metrics_directions,
    *,
    correct_mean=True,
    correct_var=True,
    add_missing=True,
    add_ignored=True,
    correct_ignored=False,
    fit=True,
    compute_beta_tildes_fn=None,
    log_fn=None,
    warn_fn=None,
    trace_level=0,
    debug_level=0,
):
    if compute_beta_tildes_fn is None:
        compute_beta_tildes_fn = runtime._compute_beta_tildes
    if log_fn is None:
        log_fn = lambda *args, **kwargs: None
    if warn_fn is None:
        warn_fn = lambda *args, **kwargs: None

    if len(beta_tildes.shape) == 1:
        beta_tildes = beta_tildes[np.newaxis, :]
    if len(ses.shape) == 1:
        ses = ses[np.newaxis, :]
    if se_inflation_factors is not None and len(se_inflation_factors.shape) == 1:
        se_inflation_factors = se_inflation_factors[np.newaxis, :]

    remove_mask = np.full(beta_tildes.shape[1], False)

    if total_qc_metrics is None:
        if runtime.gene_covariates is None:
            warn_fn("--correct-huge was not used, so skipping correction")
    else:
        if fit or runtime.total_qc_metric_betas is None:
            if add_missing and runtime.beta_tildes_missing is not None:
                beta_tildes = np.hstack((beta_tildes, np.tile(runtime.beta_tildes_missing, beta_tildes.shape[0]).reshape(beta_tildes.shape[0], len(runtime.beta_tildes_missing))))
                ses = np.hstack((ses, np.tile(runtime.ses_missing, ses.shape[0]).reshape(ses.shape[0], len(runtime.ses_missing))))
                if se_inflation_factors is not None:
                    se_inflation_factors = np.hstack((se_inflation_factors, np.tile(runtime.se_inflation_factors_missing, se_inflation_factors.shape[0]).reshape(se_inflation_factors.shape[0], len(runtime.se_inflation_factors_missing))))

                total_qc_metrics = np.vstack((total_qc_metrics, runtime.total_qc_metrics_missing))
                remove_mask = np.append(remove_mask, np.full(len(runtime.beta_tildes_missing), True))

            if add_ignored and runtime.beta_tildes_ignored is not None:
                beta_tildes = np.hstack((beta_tildes, np.tile(runtime.beta_tildes_ignored, beta_tildes.shape[0]).reshape(beta_tildes.shape[0], len(runtime.beta_tildes_ignored))))
                ses = np.hstack((ses, np.tile(runtime.ses_ignored, ses.shape[0]).reshape(ses.shape[0], len(runtime.ses_ignored))))
                if se_inflation_factors is not None:
                    se_inflation_factors = np.hstack((se_inflation_factors, np.tile(runtime.se_inflation_factors_ignored, se_inflation_factors.shape[0]).reshape(se_inflation_factors.shape[0], len(runtime.se_inflation_factors_ignored))))

                total_qc_metrics = np.vstack((total_qc_metrics, runtime.total_qc_metrics_ignored))
                remove_mask = np.append(remove_mask, np.full(len(runtime.beta_tildes_ignored), True))

            z_scores = np.zeros(beta_tildes.shape)
            z_scores[ses != 0] = np.abs(beta_tildes[ses != 0]) / ses[ses != 0]

            if runtime.huge_sparse_mode:
                log_fn("Too few genes from HuGE: using pre-computed correct betas", debug_level)
                runtime.total_qc_metric_intercept = runtime.total_qc_metric_intercept_defaults
                runtime.total_qc_metric2_intercept = runtime.total_qc_metric2_intercept_defaults
                runtime.total_qc_metric_betas = runtime.total_qc_metric_betas_defaults
                runtime.total_qc_metric2_betas = runtime.total_qc_metric2_betas_defaults
            else:
                z_scores_mask = np.all(np.logical_and(np.abs(z_scores - np.mean(z_scores)) <= 5 * np.std(z_scores), ses != 0), axis=0)
                metrics_mask = np.all(np.abs(total_qc_metrics - np.mean(total_qc_metrics, axis=0)) <= 5 * np.std(total_qc_metrics, axis=0), axis=1)
                pred_mask = np.logical_and(z_scores_mask, metrics_mask)

                intercept_mask = (np.std(total_qc_metrics, axis=0) == 0)
                if np.sum(intercept_mask) == 0:
                    total_qc_metrics = np.hstack((total_qc_metrics, np.ones((total_qc_metrics.shape[0], 1))))
                    if total_qc_metrics_directions is not None:
                        total_qc_metrics_directions = np.append(total_qc_metrics_directions, 0)
                    intercept_mask = np.append(intercept_mask, True)

                runtime.total_qc_metric_betas = np.zeros(len(intercept_mask))
                runtime.total_qc_metric2_betas = np.zeros(len(intercept_mask))

                (metric_beta_tildes_m, _metric_ses_m, _metric_z_scores_m, metric_p_values_m, _metric_se_inflation_factors_m) = compute_beta_tildes_fn(total_qc_metrics[pred_mask, :], z_scores[:, pred_mask], np.var(z_scores[:, pred_mask], axis=1), np.std(total_qc_metrics[pred_mask, :], axis=0), np.mean(total_qc_metrics[pred_mask, :], axis=0), resid_correlation_matrix=None, log_fun=lambda x, y=0: 1)

                log_fn("Mean marginal slopes are %s" % np.mean(metric_beta_tildes_m, axis=0), trace_level)

                keep_metrics = np.full(total_qc_metrics.shape[1], False)
                keep_metric_inds = np.where(np.any(metric_p_values_m < 0.05, axis=0))[0]
                keep_metrics[keep_metric_inds] = True
                keep_metrics = np.logical_or(keep_metrics, intercept_mask)
                if np.sum(keep_metrics) < total_qc_metrics.shape[1]:
                    log_fn("Not using %d non-significant metrics" % (total_qc_metrics.shape[1] - np.sum(keep_metrics)))

                if total_qc_metrics_directions is not None:
                    keep_metrics_dir = np.full(total_qc_metrics.shape[1], True)
                    keep_metric_dir_inds = np.where(np.any((metric_beta_tildes_m * total_qc_metrics_directions) < 0, axis=0))[0]
                    keep_metrics_dir[keep_metric_dir_inds] = False
                    if np.sum(keep_metrics_dir) < total_qc_metrics.shape[1]:
                        log_fn("Not using %d metrics with wrong sign" % (total_qc_metrics.shape[1] - np.sum(keep_metrics_dir)))
                    keep_metrics = np.logical_and(keep_metrics, keep_metrics_dir)

                total_qc_metrics_for_reg = total_qc_metrics
                if np.sum(keep_metrics) < total_qc_metrics.shape[1]:
                    total_qc_metrics_for_reg = total_qc_metrics[:, keep_metrics]

                total_qc_metrics_mat_inv = np.linalg.inv(total_qc_metrics_for_reg.T.dot(total_qc_metrics_for_reg))

                pred_slopes = total_qc_metrics_mat_inv.dot(total_qc_metrics_for_reg[pred_mask, :].T).dot(z_scores[:, pred_mask].T)
                pred2_slopes = total_qc_metrics_mat_inv.dot(total_qc_metrics_for_reg[pred_mask, :].T).dot(np.power(z_scores[:, pred_mask], 2).T)

                runtime.total_qc_metric_betas[keep_metrics] = np.mean(pred_slopes, axis=1)
                runtime.total_qc_metric2_betas[keep_metrics] = np.mean(pred2_slopes, axis=1)

                runtime.total_qc_metric_intercept = runtime.total_qc_metric_betas[intercept_mask]
                runtime.total_qc_metric2_intercept = runtime.total_qc_metric2_betas[intercept_mask]
                runtime.total_qc_metric_betas = runtime.total_qc_metric_betas[~intercept_mask]
                runtime.total_qc_metric2_betas = runtime.total_qc_metric2_betas[~intercept_mask]

                log_fn("Ran regression for %d gene sets" % np.sum(pred_mask), trace_level)

            desired_var = np.var(z_scores, axis=1)
            runtime.total_qc_metric_desired_var = desired_var

            log_fn("Mean slopes for mean are %s (+ %s)" % (runtime.total_qc_metric_betas, runtime.total_qc_metric_intercept), trace_level)
            if correct_var:
                log_fn("Mean slopes for square are %s (+ %s) " % (runtime.total_qc_metric2_betas, runtime.total_qc_metric2_intercept), trace_level)

            if runtime.gene_covariate_names is not None:
                param_names = ["%s_beta" % runtime.gene_covariate_names[i] for i in range(len(runtime.gene_covariate_names)) if i != runtime.gene_covariate_intercept_index] + ["%s2_beta" % runtime.gene_covariate_names[i] for i in range(len(runtime.gene_covariate_names)) if i != runtime.gene_covariate_intercept_index]
                param_values = np.append(runtime.total_qc_metric_betas, runtime.total_qc_metric2_betas)
                runtime._record_params(dict(zip(param_names, param_values)), record_only_first_time=True)

        else:
            z_scores = np.zeros(beta_tildes.shape)
            z_scores[ses != 0] = np.abs(beta_tildes[ses != 0]) / ses[ses != 0]

        intercept_mask = (np.std(total_qc_metrics, axis=0) == 0)
        pred_means = (total_qc_metrics[:, ~intercept_mask].dot(runtime.total_qc_metric_betas) + runtime.total_qc_metric_intercept).T
        pred_means2 = (total_qc_metrics[:, ~intercept_mask].dot(runtime.total_qc_metric2_betas) + runtime.total_qc_metric2_intercept).T
        pred_var = pred_means2 - np.square(pred_means)
        if len(pred_var.shape) == 1:
            pred_var = np.tile(pred_var, z_scores.shape[0]).reshape(z_scores.shape[0], len(pred_var))

        if correct_mean:
            pred_adjusted = ((z_scores - pred_means).T + runtime.total_qc_metric_intercept).T
        else:
            pred_adjusted = z_scores

        if correct_var:
            high_var_mask = np.logical_and(pred_var.T > runtime.total_qc_metric_desired_var, pred_var.T > 0).T
            pred_var[pred_var == 0] = 1
            variance_factors = (runtime.total_qc_metric_desired_var / pred_var.T).T
            pred_adjusted[high_var_mask] *= variance_factors[high_var_mask]

        inflate_mask = np.logical_and(np.abs(pred_adjusted) < np.abs(z_scores), beta_tildes != 0)

        new_ses = copy.copy(ses)
        if np.sum(inflate_mask) > 0:
            log_fn("Inflating %d standard errors" % (np.sum(inflate_mask)))

        new_ses[inflate_mask] = np.abs(beta_tildes[inflate_mask]) / np.abs(pred_adjusted[inflate_mask])

        if se_inflation_factors is not None:
            se_inflation_factors[inflate_mask] *= new_ses[inflate_mask] / ses[inflate_mask]

        ses = new_ses

    zero_se_mask = ses == 0
    assert np.sum(np.logical_and(zero_se_mask, beta_tildes != 0)) == 0
    z_scores = np.zeros(beta_tildes.shape)
    z_scores[~zero_se_mask] = beta_tildes[~zero_se_mask] / ses[~zero_se_mask]
    p_values = 2 * scipy.stats.norm.cdf(-np.abs(z_scores))

    if np.sum(remove_mask) > 0:
        if correct_ignored:
            runtime.beta_tildes_ignored = beta_tildes[0, remove_mask]
            runtime.ses_ignored = ses[0, remove_mask]
            runtime.z_scores_ignored = z_scores[0, remove_mask]
            runtime.p_values_ignored = p_values[0, remove_mask]
            if se_inflation_factors is not None:
                runtime.se_inflation_factors_ignored = se_inflation_factors[0, remove_mask]

        beta_tildes = beta_tildes[:, ~remove_mask]
        ses = ses[:, ~remove_mask]
        z_scores = z_scores[:, ~remove_mask]
        p_values = p_values[:, ~remove_mask]
        if se_inflation_factors is not None:
            se_inflation_factors = se_inflation_factors[:, ~remove_mask]

    if beta_tildes.shape[0] == 1:
        beta_tildes = np.squeeze(beta_tildes, axis=0)
        ses = np.squeeze(ses, axis=0)
        p_values = np.squeeze(p_values, axis=0)
        z_scores = np.squeeze(z_scores, axis=0)

        if se_inflation_factors is not None:
            se_inflation_factors = np.squeeze(se_inflation_factors, axis=0)

    return (beta_tildes, ses, z_scores, p_values, se_inflation_factors)


def compute_multivariate_beta_tildes(
    X,
    Y,
    *,
    resid_correlation_matrix=None,
    add_intercept=True,
    covs=None,
    finalize_regression_fn=None,
):
    if finalize_regression_fn is None:
        finalize_regression_fn = finalize_regression_outputs

    if covs is not None:
        if len(covs.shape) == 1:
            covs = covs[:, np.newaxis]
        X_design = np.hstack([X, covs])
    else:
        X_design = X

    if add_intercept:
        ones_col = np.ones((X_design.shape[0], 1))
        X_design = np.hstack([X_design, ones_col])

    n_obs, n_pred = X_design.shape
    n_phenos = Y.shape[0]
    Y_t = Y.T

    XtX = X_design.T @ X_design
    XtX_inv = np.linalg.inv(XtX)
    XtY = X_design.T @ Y_t
    betas = (XtX_inv @ XtY).T

    fitted = X_design @ betas.T
    residuals = Y_t - fitted

    df = n_obs - n_pred
    if df <= 0:
        raise ValueError("Degrees of freedom <= 0. Check the size of your input matrices.")

    sse = np.sum(residuals ** 2, axis=0)
    sigma2 = sse / df

    diag_xtx_inv = np.diag(XtX_inv)
    classical_ses = np.sqrt(sigma2[:, None] * diag_xtx_inv[None, :])
    final_ses = classical_ses.copy()

    if resid_correlation_matrix is not None:
        if len(resid_correlation_matrix) != n_phenos:
            raise ValueError("resid_correlation_matrix must be a list of length == n_phenos.")

        for p in range(n_phenos):
            R_p = resid_correlation_matrix[p]
            if sparse.issparse(R_p):
                XR_p = R_p.dot(X_design)
            else:
                XR_p = R_p @ X_design

            XtR_pX = X_design.T @ XR_p
            var_betas_p = XtX_inv @ XtR_pX @ XtX_inv
            final_ses[p, :] = np.sqrt(np.diag(var_betas_p))

    if covs is not None or add_intercept:
        n_factors = X.shape[1]
        betas = betas[:, :n_factors]
        final_ses = final_ses[:, :n_factors]

    return finalize_regression_fn(betas, final_ses, se_inflation_factors=None)
