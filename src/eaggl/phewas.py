from __future__ import annotations

import copy

import numpy as np
import scipy.sparse as sparse
import scipy.stats

from pegs_shared.io_common import construct_map_to_ind as pegs_construct_map_to_ind
from pegs_shared.io_common import open_text_with_retry
from pegs_shared.phewas import (
    accumulate_factor_phewas_outputs as pegs_accumulate_factor_phewas_outputs,
    accumulate_standard_phewas_outputs as pegs_accumulate_standard_phewas_outputs,
    append_phewas_metric_block as pegs_append_phewas_metric_block,
    prepare_phewas_phenos_from_file as pegs_prepare_phewas_phenos_from_file,
    read_phewas_file_batch as pegs_read_phewas_file_batch,
)

from .io import has_loaded_gene_phewas
from . import regression as eaggl_regression


def open_gz(file, flag=None):
    return open_text_with_retry(file, flag=flag)


_BINARY_FACTOR_PHEWAS_MODES = {
    "marginal_anchor_adjusted_binary",
    "marginal_unconditional_binary",
    "joint_anchor_adjusted_binary",
}

_LEGACY_FACTOR_PHEWAS_MODES = {
    "legacy_continuous_direct",
    "legacy_continuous_combined",
}

_FACTOR_PHEWAS_MODEL_METADATA = {
    "marginal_anchor_adjusted_binary": {
        "factor_model_scope": "marginal_one_factor",
        "outcome_surface": "binary_thresholded",
    },
    "marginal_unconditional_binary": {
        "factor_model_scope": "marginal_one_factor",
        "outcome_surface": "binary_thresholded",
    },
    "joint_anchor_adjusted_binary": {
        "factor_model_scope": "joint_all_factors",
        "outcome_surface": "binary_thresholded",
    },
    "legacy_continuous_direct": {
        "factor_model_scope": "joint_all_factors",
        "outcome_surface": "continuous_direct",
    },
    "legacy_continuous_combined": {
        "factor_model_scope": "joint_all_factors",
        "outcome_surface": "continuous_combined",
    },
}


def _ensure_factor_phewas_result_blocks(state):
    if getattr(state, "factor_phewas_result_blocks", None) is None:
        state.factor_phewas_result_blocks = []
    return state.factor_phewas_result_blocks


def _append_factor_phewas_result_block(
    state,
    *,
    phenos,
    analysis,
    mode,
    anchor_covariate,
    threshold_cutoff,
    se_type,
    coefficients,
    ses,
    z_scores,
    p_values,
    one_sided_p_values,
):
    metadata = _FACTOR_PHEWAS_MODEL_METADATA.get(mode, {})
    _ensure_factor_phewas_result_blocks(state).append(
        {
            "phenos": list(phenos),
            "analysis": analysis,
            "mode": mode,
            "model_name": mode,
            "factor_model_scope": metadata.get("factor_model_scope", "unknown"),
            "outcome_surface": metadata.get("outcome_surface", "unknown"),
            "anchor_covariate": anchor_covariate,
            "threshold_cutoff": threshold_cutoff,
            "se_type": se_type,
            "coefficients": np.asarray(coefficients, dtype=float),
            "ses": np.asarray(ses, dtype=float),
            "z_scores": np.asarray(z_scores, dtype=float),
            "p_values": np.asarray(p_values, dtype=float),
            "one_sided_p_values": np.asarray(one_sided_p_values, dtype=float),
        }
    )


def resolve_requested_factor_phewas_modes(options):
    raw_modes = getattr(options, "factor_phewas_modes", None)
    if raw_modes is None or raw_modes == []:
        raw_modes = [getattr(options, "factor_phewas_mode", "marginal_anchor_adjusted_binary")]
    elif isinstance(raw_modes, str):
        raw_modes = raw_modes.split(",")

    modes = []
    seen = set()
    for raw_mode in raw_modes:
        mode = str(raw_mode).strip()
        if mode == "" or mode in seen:
            continue
        seen.add(mode)
        modes.append(mode)
    return modes


def _factor_phewas_anchor_vector(state, anchor_covariate, bail_fn):
    if anchor_covariate == "none":
        return None
    if anchor_covariate == "direct":
        if state.Y is None:
            bail_fn("Default factor-PheWAS anchor adjustment requires direct anchor support (`Y`)")
        return np.asarray(state.Y, dtype=float)
    if anchor_covariate == "combined":
        if state.combined_prior_Ys is None:
            bail_fn("Combined anchor adjustment requires combined anchor support")
        return np.asarray(state.combined_prior_Ys, dtype=float)
    bail_fn("Unknown factor-PheWAS anchor covariate: %s" % anchor_covariate)


def _build_thresholded_hit_matrix(gene_pheno_combined_prior_Ys, gene_pheno_Y, cutoff):
    source = gene_pheno_combined_prior_Ys if gene_pheno_combined_prior_Ys is not None else gene_pheno_Y
    if source is None:
        return None
    source = np.asarray(source, dtype=float)
    if gene_pheno_combined_prior_Ys is not None:
        return np.asarray(source > cutoff, dtype=float)
    return np.asarray(source > 0, dtype=float)


def _fit_ols_with_selected_coefficients(X, Y, coefficient_indices, *, se_type="robust"):
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    if Y.ndim == 1:
        Y = Y[:, np.newaxis]

    xtx_inv = np.linalg.pinv(X.T @ X)
    beta_all = xtx_inv @ X.T @ Y
    residuals = Y - X @ beta_all

    if se_type == "none":
        dof = max(X.shape[0] - X.shape[1], 1)
        sigma2 = np.sum(residuals ** 2, axis=0) / dof
        covariances = np.einsum("m,ij->mij", sigma2, xtx_inv)
    else:
        hat_diag = np.einsum("ij,jk,ik->i", X, xtx_inv, X, optimize=True)
        denom = np.maximum(1.0 - hat_diag, 1e-8)
        scaled_residuals = (residuals ** 2) / (denom[:, np.newaxis] ** 2)
        score_cov = np.einsum("im,ip,iq->mpq", scaled_residuals, X, X, optimize=True)
        covariances = np.einsum("ab,mbc,cd->mad", xtx_inv, score_cov, xtx_inv, optimize=True)

    coefficient_indices = list(coefficient_indices)
    coefficients = beta_all[coefficient_indices, :]
    ses = np.zeros((len(coefficient_indices), Y.shape[1]), dtype=float)
    for coef_ind, source_index in enumerate(coefficient_indices):
        ses[coef_ind, :] = np.sqrt(np.maximum(covariances[:, source_index, source_index], 0))

    z_scores = np.zeros_like(coefficients)
    positive_mask = ses > 0
    z_scores[positive_mask] = coefficients[positive_mask] / ses[positive_mask]
    p_values = 2 * scipy.stats.norm.cdf(-np.abs(z_scores))
    one_sided_p_values = np.where(z_scores >= 0, p_values / 2.0, 1 - p_values / 2.0)
    return coefficients, ses, z_scores, p_values, one_sided_p_values


def build_phewas_input_values(state, run_for_factors=False, min_gene_factor_weight=0):
    if run_for_factors:
        input_values = state.exp_gene_factors
        factor_keep_mask = np.full(input_values.shape[0], True)
        if min_gene_factor_weight > 0:
            factor_keep_mask = np.any(state.exp_gene_factors > min_gene_factor_weight, axis=1)
        return input_values, factor_keep_mask

    default_value = (
        state.Y[:, np.newaxis]
        if state.Y is not None
        else state.combined_prior_Ys[:, np.newaxis]
        if state.combined_prior_Ys is not None
        else state.priors[:, np.newaxis]
    )
    input_values = np.hstack(
        (
            state.Y[:, np.newaxis] if state.Y is not None else default_value,
            state.combined_prior_Ys[:, np.newaxis] if state.combined_prior_Ys is not None else default_value,
            state.priors[:, np.newaxis] if state.priors is not None else default_value,
        )
    )
    input_values = np.exp(input_values + state.background_bf) / (1 + np.exp(input_values + state.background_bf))
    return input_values, None


def calculate_phewas_block(
    state,
    X_mat,
    Y_mat,
    *,
    max_num_burn_in,
    max_num_iter,
    min_num_iter,
    num_chains,
    r_threshold_burn_in,
    use_max_r_for_convergence,
    max_frac_sem,
    gauss_seidel,
    sparse_solution,
    sparse_frac_betas,
    non_inf_kwargs,
    X_orig=None,
    X_phewas_beta=None,
    Y_resid=None,
    multivariate=False,
    covs=None,
    huber=False,
    options,
    bail_fn,
    warn_fn,
    log_fn,
    info_level,
    debug_level,
    trace_level,
):
    del info_level
    bail = bail_fn
    log = log_fn
    DEBUG = debug_level
    TRACE = trace_level

    (mean_shifts, scale_factors) = state._calc_X_shift_scale(X_mat)
    cor_matrices = None

    beta_tildes = np.zeros((Y_mat.shape[0], X_mat.shape[1]))
    ses = np.zeros((Y_mat.shape[0], X_mat.shape[1]))
    z_scores = np.zeros((Y_mat.shape[0], X_mat.shape[1]))
    p_values = np.zeros((Y_mat.shape[0], X_mat.shape[1]))
    se_inflation_factors = np.zeros((Y_mat.shape[0], X_mat.shape[1]))

    cor_batch_size = int(np.ceil(beta_tildes.shape[0] / 4) if X_phewas_beta is not None and X_orig is not None else beta_tildes.shape[0])
    num_cor_batches = int(np.ceil(beta_tildes.shape[0] / cor_batch_size))
    for batch in range(num_cor_batches):
        log("Processing block batch %s" % (batch), TRACE)
        begin = batch * cor_batch_size
        end = min((batch + 1) * cor_batch_size, beta_tildes.shape[0])

        if X_phewas_beta is not None and X_orig is not None and not options.debug_skip_correlation:
            if X_phewas_beta.shape[0] != Y_mat.shape[0]:
                bail(
                    "When calling this, the phewas_betas must have same number of phenos as Y_mat: shapes are X_phewas=(%d,%d) vs. Y_mat=(%d,%d)"
                    % (X_phewas_beta.shape[0], X_phewas_beta.shape[1], Y_mat.shape[0], Y_mat.shape[1])
                )
            dot_threshold = 0.01 * 0.01
            log("Calculating correlation matrix for use in residuals", DEBUG)
            cor_matrices = state._sparse_correlation_with_dot_product_threshold(
                X_orig,
                X_phewas_beta[begin:end, :],
                dot_product_threshold=dot_threshold,
                Y=Y_resid[begin:end, :],
            )

            total = 0
            nnz = 0
            for cor_matrix in cor_matrices if type(cor_matrices) is list else [cor_matrices]:
                total += np.prod(cor_matrix.shape)
                nnz += cor_matrix.nnz
            log(
                "Sparsity of correlation matrix is %d/%d=%.3g (size %.3gMb)"
                % (nnz, total, float(nnz) / total, nnz * 8 / (1024 * 1024)),
                DEBUG,
            )

        if multivariate:
            if huber:
                (
                    beta_tildes[begin:end, :],
                    ses[begin:end, :],
                    z_scores[begin:end, :],
                    p_values[begin:end, :],
                    se_inflation_factors[begin:end, :],
                ) = eaggl_regression.compute_robust_betas(state, 
                    X_mat,
                    Y_mat[begin:end, :],
                    resid_correlation_matrix=cor_matrices,
                    covs=covs if not options.debug_skip_phewas_covs else None,
                    finalize_regression_fn=lambda *args, **kwargs: eaggl_regression.finalize_regression(
                        *args,
                        log_fn=log,
                        warn_fn=warn_fn,
                        trace_level=TRACE,
                        **kwargs,
                    ),
                    log_fn=log,
                    debug_level=DEBUG,
                )
            else:
                (
                    beta_tildes[begin:end, :],
                    ses[begin:end, :],
                    z_scores[begin:end, :],
                    p_values[begin:end, :],
                    se_inflation_factors[begin:end, :],
                ) = eaggl_regression.compute_multivariate_beta_tildes(state, 
                    X_mat,
                    Y_mat[begin:end, :],
                    resid_correlation_matrix=cor_matrices,
                    covs=covs if not options.debug_skip_phewas_covs else None,
                    finalize_regression_fn=lambda *args, **kwargs: eaggl_regression.finalize_regression(
                        *args,
                        log_fn=log,
                        warn_fn=warn_fn,
                        trace_level=TRACE,
                        **kwargs,
                    ),
                )
        else:
            (
                beta_tildes[begin:end, :],
                ses[begin:end, :],
                z_scores[begin:end, :],
                p_values[begin:end, :],
                se_inflation_factors[begin:end, :],
            ) = eaggl_regression.compute_beta_tildes(state, 
                X_mat,
                Y_mat[begin:end, :],
                scale_factors=scale_factors,
                mean_shifts=mean_shifts,
                resid_correlation_matrix=cor_matrices,
                finalize_regression_fn=lambda *args, **kwargs: eaggl_regression.finalize_regression(
                    *args,
                    log_fn=log,
                    warn_fn=warn_fn,
                    trace_level=TRACE,
                    **kwargs,
                ),
                bail_fn=bail,
                log_fun=log,
                debug_level=DEBUG,
            )

    one_sided_p_values = copy.copy(p_values)
    one_sided_p_values[z_scores < 0] = 1 - p_values[z_scores < 0] / 2.0
    one_sided_p_values[z_scores > 0] = p_values[z_scores > 0] / 2.0

    if multivariate:
        return (None, None, beta_tildes.T, ses.T, z_scores.T, p_values.T, one_sided_p_values.T)

    orig_ps = state.ps
    orig_sigma2s = state.sigma2s
    orig_p = state.p
    orig_sigma2_internal = state.sigma2
    orig_sigma_power = state.sigma_power
    state.ps = None
    state.sigma2s = None

    try:
        new_p = 0.5
        new_sigma2_internal = orig_sigma2_internal * (new_p / orig_p)
        state.set_p(new_p)
        state.set_sigma(new_sigma2_internal, orig_sigma_power, convert_sigma_to_internal_units=False)

        (betas_uncorrected, postp_uncorrected) = state._calculate_non_inf_betas(
            initial_p=state.p,
            assume_independent=True,
            beta_tildes=beta_tildes,
            ses=ses,
            V=None,
            X_orig=None,
            scale_factors=scale_factors,
            mean_shifts=mean_shifts,
            max_num_burn_in=max_num_burn_in,
            max_num_iter=max_num_iter,
            min_num_iter=min_num_iter,
            num_chains=num_chains,
            r_threshold_burn_in=r_threshold_burn_in,
            use_max_r_for_convergence=use_max_r_for_convergence,
            max_frac_sem=max_frac_sem,
            gauss_seidel=gauss_seidel,
            update_hyper_sigma=False,
            update_hyper_p=False,
            sparse_solution=sparse_solution,
            sparse_frac_betas=sparse_frac_betas,
            **non_inf_kwargs,
        )
    finally:
        state.ps = orig_ps
        state.sigma2s = orig_sigma2s
        state.p = orig_p
        state.sigma2 = orig_sigma2_internal
        state.sigma_power = orig_sigma_power

    return (
        (betas_uncorrected / scale_factors).T,
        postp_uncorrected.T,
        (beta_tildes / scale_factors).T,
        (ses / scale_factors).T,
        z_scores.T,
        p_values.T,
        one_sided_p_values.T,
    )


def append_phewas_metric_block(current_beta, current_beta_tilde, current_se, current_z, current_p_value, current_one_sided_p_value, beta, beta_tilde, se, z_score, p_value, one_sided_p_value):
    return pegs_append_phewas_metric_block(
        current_beta,
        current_beta_tilde,
        current_se,
        current_z,
        current_p_value,
        current_one_sided_p_value,
        beta,
        beta_tilde,
        se,
        z_score,
        p_value,
        one_sided_p_value,
    )


def prepare_phewas_phenos_from_file(state, gene_phewas_bfs_in, gene_phewas_bfs_id_col=None, gene_phewas_bfs_pheno_col=None, gene_phewas_bfs_log_bf_col=None, gene_phewas_bfs_combined_col=None, gene_phewas_bfs_prior_col=None, *, get_col_fn, warn_fn, log_fn, debug_level):
    phenos, pheno_to_ind, col_info = pegs_prepare_phewas_phenos_from_file(
        state,
        gene_phewas_bfs_in,
        gene_phewas_bfs_id_col=gene_phewas_bfs_id_col,
        gene_phewas_bfs_pheno_col=gene_phewas_bfs_pheno_col,
        gene_phewas_bfs_log_bf_col=gene_phewas_bfs_log_bf_col,
        gene_phewas_bfs_combined_col=gene_phewas_bfs_combined_col,
        gene_phewas_bfs_prior_col=gene_phewas_bfs_prior_col,
        open_text_fn=open_gz,
        get_col_fn=get_col_fn,
        construct_map_to_ind_fn=pegs_construct_map_to_ind,
        warn_fn=warn_fn,
        log_fn=log_fn,
        debug_level=debug_level,
    )
    return phenos, pheno_to_ind, {
        "id_col": col_info.id_col,
        "pheno_col": col_info.pheno_col,
        "bf_col": col_info.bf_col,
        "combined_col": col_info.combined_col,
        "prior_col": col_info.prior_col,
    }


def read_phewas_file_batch(state, gene_phewas_bfs_in, begin, cur_batch_size, pheno_to_ind, id_col, pheno_col, bf_col, combined_col, prior_col, *, warn_fn):
    col_info = {
        "id_col": id_col,
        "pheno_col": pheno_col,
        "bf_col": bf_col,
        "combined_col": combined_col,
        "prior_col": prior_col,
    }
    return pegs_read_phewas_file_batch(
        state,
        gene_phewas_bfs_in,
        begin=begin,
        cur_batch_size=cur_batch_size,
        pheno_to_ind=pheno_to_ind,
        col_info=col_info,
        open_text_fn=open_gz,
        warn_fn=warn_fn,
    )


def accumulate_standard_phewas_outputs(state, output_prefix, beta, beta_tilde, se, z_score, p_value):
    pegs_accumulate_standard_phewas_outputs(
        state,
        output_prefix,
        beta,
        beta_tilde,
        se,
        z_score,
        p_value,
    )


def accumulate_factor_phewas_outputs(state, output_prefix, beta_tilde, se, z_score, p_value, one_sided_p_value, huber=False):
    pegs_accumulate_factor_phewas_outputs(
        state,
        output_prefix,
        beta_tilde,
        se,
        z_score,
        p_value,
        one_sided_p_value,
        huber=huber,
    )


def _run_legacy_factor_phewas_batch(state, input_values, factor_keep_mask, gene_pheno_Y, gene_pheno_combined_prior_Ys, begin, end, phewas_beta_kwargs, *, mode, options):
    batch_phenos = state.phenos[begin:end]
    if gene_pheno_Y is not None:
        _, _, beta_tilde, se, z_score, p_value, one_sided_p_value = calculate_phewas_block(
            state,
            input_values[factor_keep_mask, :],
            gene_pheno_Y[factor_keep_mask, :].T,
            multivariate=True,
            covs=state.Y[factor_keep_mask],
            **phewas_beta_kwargs
        )
        _append_factor_phewas_result_block(
            state,
            phenos=batch_phenos,
            analysis="legacy_continuous_direct",
            mode=mode,
            anchor_covariate="direct",
            threshold_cutoff=np.nan,
            se_type="legacy",
            coefficients=beta_tilde,
            ses=se,
            z_scores=z_score,
            p_values=p_value,
            one_sided_p_values=one_sided_p_value,
        )

        if getattr(options, "factor_phewas_full_output", False) and not options.debug_skip_huber:
            _, _, beta_tilde, se, z_score, p_value, one_sided_p_value = calculate_phewas_block(
                state,
                input_values[factor_keep_mask, :],
                gene_pheno_Y[factor_keep_mask, :].T,
                multivariate=True,
                covs=state.Y[factor_keep_mask],
                huber=True,
                **phewas_beta_kwargs
            )
            _append_factor_phewas_result_block(
                state,
                phenos=batch_phenos,
                analysis="legacy_continuous_direct_huber",
                mode=mode,
                anchor_covariate="direct",
                threshold_cutoff=np.nan,
                se_type="legacy_huber",
                coefficients=beta_tilde,
                ses=se,
                z_scores=z_score,
                p_values=p_value,
                one_sided_p_values=one_sided_p_value,
            )

    if (
        (mode == "legacy_continuous_combined" or getattr(options, "factor_phewas_full_output", False))
        and gene_pheno_combined_prior_Ys is not None
        and not options.debug_skip_correlation
    ):
        _, _, beta_tilde, se, z_score, p_value, one_sided_p_value = calculate_phewas_block(
            state,
            input_values[factor_keep_mask, :],
            gene_pheno_combined_prior_Ys[factor_keep_mask, :].T,
            X_orig=state.X_orig[factor_keep_mask, :],
            X_phewas_beta=state.X_phewas_beta[begin:end, :] if state.X_phewas_beta is not None else None,
            Y_resid=gene_pheno_Y[factor_keep_mask, :].T,
            multivariate=True,
            covs=state.combined_prior_Ys[factor_keep_mask] if state.combined_prior_Ys is not None else state.Y[factor_keep_mask],
            **phewas_beta_kwargs
        )
        _append_factor_phewas_result_block(
            state,
            phenos=batch_phenos,
            analysis="legacy_continuous_combined",
            mode="legacy_continuous_combined",
            anchor_covariate="combined",
            threshold_cutoff=np.nan,
            se_type="legacy",
            coefficients=beta_tilde,
            ses=se,
            z_scores=z_score,
            p_values=p_value,
            one_sided_p_values=one_sided_p_value,
        )

        if not options.debug_skip_huber:
            _, _, beta_tilde, se, z_score, p_value, one_sided_p_value = calculate_phewas_block(
                state,
                input_values[factor_keep_mask, :],
                gene_pheno_combined_prior_Ys[factor_keep_mask, :].T,
                X_orig=state.X_orig[factor_keep_mask, :],
                X_phewas_beta=state.X_phewas_beta[begin:end, :] if state.X_phewas_beta is not None else None,
                Y_resid=gene_pheno_Y[factor_keep_mask, :].T,
                multivariate=True,
                covs=state.combined_prior_Ys[factor_keep_mask] if state.combined_prior_Ys is not None else state.Y[factor_keep_mask],
                huber=True,
                **phewas_beta_kwargs
            )
            _append_factor_phewas_result_block(
                state,
                phenos=batch_phenos,
                analysis="legacy_continuous_combined_huber",
                mode="legacy_continuous_combined",
                anchor_covariate="combined",
                threshold_cutoff=np.nan,
                se_type="legacy_huber",
                coefficients=beta_tilde,
                ses=se,
                z_scores=z_score,
                p_values=p_value,
                one_sided_p_values=one_sided_p_value,
            )


def _run_factor_phewas_batch_for_mode(state, gene_pheno_Y, gene_pheno_combined_prior_Ys, begin, end, phewas_beta_kwargs, *, mode, options):
    min_gene_factor_weight = (
        getattr(options, "factor_phewas_min_gene_factor_weight", 0.0)
        if mode in _LEGACY_FACTOR_PHEWAS_MODES
        else 0.0
    )
    input_values, factor_keep_mask = build_phewas_input_values(
        state,
        run_for_factors=True,
        min_gene_factor_weight=min_gene_factor_weight,
    )
    batch_phenos = state.phenos[begin:end]

    if mode in _LEGACY_FACTOR_PHEWAS_MODES:
        legacy_direct = mode == "legacy_continuous_direct"
        legacy_combined = mode == "legacy_continuous_combined"
        _run_legacy_factor_phewas_batch(
            state,
            input_values=input_values,
            factor_keep_mask=factor_keep_mask,
            gene_pheno_Y=gene_pheno_Y if legacy_direct else None,
            gene_pheno_combined_prior_Ys=gene_pheno_combined_prior_Ys if legacy_combined or getattr(options, "factor_phewas_full_output", False) else None,
            begin=begin,
            end=end,
            phewas_beta_kwargs=phewas_beta_kwargs,
            mode=mode,
            options=options,
        )
        return

    hit_matrix = _build_thresholded_hit_matrix(
        gene_pheno_combined_prior_Ys,
        gene_pheno_Y,
        getattr(options, "factor_phewas_thresholded_combined_cutoff", 1.0),
    )
    if hit_matrix is None:
        return

    se_type = getattr(options, "factor_phewas_se", "robust")
    anchor_covariate = (
        "none"
        if mode == "marginal_unconditional_binary"
        else getattr(options, "factor_phewas_anchor_covariate", "direct")
    )
    anchor_vector = _factor_phewas_anchor_vector(state, anchor_covariate, phewas_beta_kwargs["bail_fn"])
    intercept = np.ones((input_values.shape[0], 1), dtype=float)

    if mode == "joint_anchor_adjusted_binary":
        design_parts = [input_values, intercept]
        coefficient_indices = list(range(input_values.shape[1]))
        if anchor_vector is not None:
            design_parts.insert(1, anchor_vector[:, np.newaxis])
            coefficient_indices = list(range(input_values.shape[1]))
        X_design = np.hstack(design_parts)
        coefficients, ses, z_scores, p_values, one_sided_p_values = _fit_ols_with_selected_coefficients(
            X_design,
            hit_matrix,
            coefficient_indices,
            se_type=se_type,
        )
        _append_factor_phewas_result_block(
            state,
            phenos=batch_phenos,
            analysis="joint_anchor_adjusted_binary",
            mode=mode,
            anchor_covariate=anchor_covariate,
            threshold_cutoff=getattr(options, "factor_phewas_thresholded_combined_cutoff", 1.0),
            se_type=se_type,
            coefficients=coefficients,
            ses=ses,
            z_scores=z_scores,
            p_values=p_values,
            one_sided_p_values=one_sided_p_values,
        )
        return

    num_factors = input_values.shape[1]
    coefficients = np.zeros((num_factors, hit_matrix.shape[1]), dtype=float)
    ses = np.zeros_like(coefficients)
    z_scores = np.zeros_like(coefficients)
    p_values = np.zeros_like(coefficients)
    one_sided_p_values = np.zeros_like(coefficients)
    for factor_index in range(num_factors):
        design_parts = [input_values[:, factor_index][:, np.newaxis], intercept]
        if anchor_vector is not None:
            design_parts.insert(1, anchor_vector[:, np.newaxis])
        X_design = np.hstack(design_parts)
        cur_coef, cur_se, cur_z, cur_p, cur_one = _fit_ols_with_selected_coefficients(
            X_design,
            hit_matrix,
            [0],
            se_type=se_type,
        )
        coefficients[factor_index, :] = cur_coef[0, :]
        ses[factor_index, :] = cur_se[0, :]
        z_scores[factor_index, :] = cur_z[0, :]
        p_values[factor_index, :] = cur_p[0, :]
        one_sided_p_values[factor_index, :] = cur_one[0, :]

    _append_factor_phewas_result_block(
        state,
        phenos=batch_phenos,
        analysis=mode,
        mode=mode,
        anchor_covariate=anchor_covariate,
        threshold_cutoff=getattr(options, "factor_phewas_thresholded_combined_cutoff", 1.0),
        se_type=se_type,
        coefficients=coefficients,
        ses=ses,
        z_scores=z_scores,
        p_values=p_values,
        one_sided_p_values=one_sided_p_values,
    )


def run_factor_phewas_batch(state, gene_pheno_Y, gene_pheno_combined_prior_Ys, begin, end, phewas_beta_kwargs, *, options):
    for mode in resolve_requested_factor_phewas_modes(options):
        _run_factor_phewas_batch_for_mode(
            state,
            gene_pheno_Y=gene_pheno_Y,
            gene_pheno_combined_prior_Ys=gene_pheno_combined_prior_Ys,
            begin=begin,
            end=end,
            phewas_beta_kwargs=phewas_beta_kwargs,
            mode=mode,
            options=options,
        )


def run_standard_phewas_batch(state, input_values, gene_pheno_Y, gene_pheno_combined_prior_Ys, begin, end, phewas_beta_kwargs, *, options):
    if gene_pheno_Y is not None:
        beta, _, beta_tilde, se, z_score, p_value, _ = calculate_phewas_block(
            state,
            input_values,
            gene_pheno_Y.T,
            **phewas_beta_kwargs
        )
        assert beta.shape[0] == 3, "First dimension of beta should be 3, not (%s, %s)" % (beta.shape[0], beta.shape[1])
        accumulate_standard_phewas_outputs(state, "pheno_Y", beta, beta_tilde, se, z_score, p_value)

    if gene_pheno_combined_prior_Ys is not None and not options.debug_skip_correlation:
        beta, _, beta_tilde, se, z_score, p_value, _ = calculate_phewas_block(
            state,
            input_values,
            gene_pheno_combined_prior_Ys.T,
            X_orig=state.X_orig,
            X_phewas_beta=state.X_phewas_beta[begin:end, :] if state.X_phewas_beta is not None else None,
            Y_resid=gene_pheno_Y.T,
            **phewas_beta_kwargs
        )
        assert beta.shape[0] == 3, "First dimension of beta should be 3, not (%s, %s)" % (beta.shape[0], beta.shape[1])
        accumulate_standard_phewas_outputs(state, "pheno_combined_prior_Ys", beta, beta_tilde, se, z_score, p_value)


def run_phewas(state, gene_phewas_bfs_in=None, gene_phewas_bfs_id_col=None, gene_phewas_bfs_pheno_col=None, gene_phewas_bfs_log_bf_col=None, gene_phewas_bfs_combined_col=None, gene_phewas_bfs_prior_col=None, run_for_factors=False, max_num_burn_in=1000, max_num_iter=1100, min_num_iter=10, num_chains=10, r_threshold_burn_in=1.01, use_max_r_for_convergence=True, max_frac_sem=0.01, gauss_seidel=False, sparse_solution=False, sparse_frac_betas=None, batch_size=1500, min_gene_factor_weight=0, *, options, bail_fn, warn_fn, log_fn, info_level, debug_level, trace_level, **kwargs):
    bail = bail_fn
    warn = warn_fn
    log = log_fn
    INFO = info_level
    TRACE = trace_level

    if gene_phewas_bfs_in is None and not has_loaded_gene_phewas(state):
        bail("Require --gene-stats-in or --gene-phewas-bfs-in with a column for log_bf/Y in this operation")

    if run_for_factors:
        if state.exp_gene_set_factors is None:
            warn("Cannot run factor phewas without gene factors; skipping")
            return
        log("Running factor phewas", INFO)
        state.factor_phewas_result_blocks = []
    else:
        if state.genes is None:
            warn("Cannot run phewas without X matrix; skipping")
            return
        if state.Y is None and state.combined_prior_Ys is None and state.priors is None:
            warn("Cannot run phewas without Y values; skipping")
            return
        log("Running phewas", INFO)

    read_file = gene_phewas_bfs_in is not None
    col_info = None

    if read_file:
        phenos, pheno_to_ind, col_info = prepare_phewas_phenos_from_file(state, 
            gene_phewas_bfs_in=gene_phewas_bfs_in,
            gene_phewas_bfs_id_col=gene_phewas_bfs_id_col,
            gene_phewas_bfs_pheno_col=gene_phewas_bfs_pheno_col,
            gene_phewas_bfs_log_bf_col=gene_phewas_bfs_log_bf_col,
            gene_phewas_bfs_combined_col=gene_phewas_bfs_combined_col,
            gene_phewas_bfs_prior_col=gene_phewas_bfs_prior_col,
            get_col_fn=state._get_col,
            warn_fn=warn_fn,
            log_fn=log_fn,
            debug_level=debug_level,
        )
    else:
        phenos = state.phenos

    num_batches = int(np.ceil(len(phenos) / batch_size))
    input_values = None
    factor_keep_mask = None
    if not run_for_factors:
        input_values, factor_keep_mask = build_phewas_input_values(state, 
            run_for_factors=run_for_factors,
            min_gene_factor_weight=min_gene_factor_weight,
        )
    phewas_beta_kwargs = {
        "max_num_burn_in": max_num_burn_in,
        "max_num_iter": max_num_iter,
        "min_num_iter": min_num_iter,
        "num_chains": num_chains,
        "r_threshold_burn_in": r_threshold_burn_in,
        "use_max_r_for_convergence": use_max_r_for_convergence,
        "max_frac_sem": max_frac_sem,
        "gauss_seidel": gauss_seidel,
        "sparse_solution": sparse_solution,
        "sparse_frac_betas": sparse_frac_betas,
        "non_inf_kwargs": kwargs,
        "options": options,
        "bail_fn": bail_fn,
        "warn_fn": warn_fn,
        "log_fn": log_fn,
        "info_level": info_level,
        "debug_level": debug_level,
        "trace_level": trace_level,
    }

    for batch in range(num_batches):
        log("Getting phenos block batch %s" % (batch), TRACE)

        begin = batch * batch_size
        end = (batch + 1) * batch_size
        if end > len(phenos):
            end = len(phenos)

        cur_batch_size = end - begin
        log("Processing phenos %d-%d" % (begin + 1, end))

        if read_file:
            gene_pheno_Y, gene_pheno_combined_prior_Ys, gene_pheno_priors = read_phewas_file_batch(state, 
                gene_phewas_bfs_in=gene_phewas_bfs_in,
                begin=begin,
                cur_batch_size=cur_batch_size,
                pheno_to_ind=pheno_to_ind,
                id_col=col_info["id_col"],
                pheno_col=col_info["pheno_col"],
                bf_col=col_info["bf_col"],
                combined_col=col_info["combined_col"],
                prior_col=col_info["prior_col"],
                warn_fn=warn_fn,
            )
        else:
            gene_pheno_Y = state.gene_pheno_Y[:, begin:end].toarray() if state.gene_pheno_Y is not None else None
            gene_pheno_combined_prior_Ys = (
                state.gene_pheno_combined_prior_Ys[:, begin:end].toarray()
                if state.gene_pheno_combined_prior_Ys is not None
                else None
            )

        if run_for_factors:
            run_factor_phewas_batch(state, 
                gene_pheno_Y=gene_pheno_Y,
                gene_pheno_combined_prior_Ys=gene_pheno_combined_prior_Ys,
                begin=begin,
                end=end,
                phewas_beta_kwargs=phewas_beta_kwargs,
                options=options,
            )
        else:
            run_standard_phewas_batch(state, 
                input_values=input_values,
                gene_pheno_Y=gene_pheno_Y,
                gene_pheno_combined_prior_Ys=gene_pheno_combined_prior_Ys,
                begin=begin,
                end=end,
                phewas_beta_kwargs=phewas_beta_kwargs,
                options=options,
            )
