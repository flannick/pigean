from __future__ import annotations

import copy

import numpy as np
import scipy.sparse as sparse

from pegs_shared.io_common import open_text_with_retry
from pegs_utils import (
    accumulate_factor_phewas_outputs as pegs_accumulate_factor_phewas_outputs,
    accumulate_standard_phewas_outputs as pegs_accumulate_standard_phewas_outputs,
    append_phewas_metric_block as pegs_append_phewas_metric_block,
    construct_map_to_ind as pegs_construct_map_to_ind,
    prepare_phewas_phenos_from_file as pegs_prepare_phewas_phenos_from_file,
    read_phewas_file_batch as pegs_read_phewas_file_batch,
)

from .io import has_loaded_gene_phewas


def open_gz(file, flag=None):
    return open_text_with_retry(file, flag=flag)


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
    del warn_fn, info_level
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
                ) = state._compute_robust_betas(
                    X_mat,
                    Y_mat[begin:end, :],
                    resid_correlation_matrix=cor_matrices,
                    covs=covs if not options.debug_skip_phewas_covs else None,
                )
            else:
                (
                    beta_tildes[begin:end, :],
                    ses[begin:end, :],
                    z_scores[begin:end, :],
                    p_values[begin:end, :],
                    se_inflation_factors[begin:end, :],
                ) = state._compute_multivariate_beta_tildes(
                    X_mat,
                    Y_mat[begin:end, :],
                    resid_correlation_matrix=cor_matrices,
                    covs=covs if not options.debug_skip_phewas_covs else None,
                )
        else:
            (
                beta_tildes[begin:end, :],
                ses[begin:end, :],
                z_scores[begin:end, :],
                p_values[begin:end, :],
                se_inflation_factors[begin:end, :],
            ) = state._compute_beta_tildes(
                X_mat,
                Y_mat[begin:end, :],
                scale_factors=scale_factors,
                mean_shifts=mean_shifts,
                resid_correlation_matrix=cor_matrices,
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


def run_factor_phewas_batch(state, input_values, factor_keep_mask, gene_pheno_Y, gene_pheno_combined_prior_Ys, begin, end, phewas_beta_kwargs, *, options):
    if gene_pheno_Y is not None:
        _, _, beta_tilde, se, z_score, p_value, one_sided_p_value = state._calculate_phewas_block(
            input_values[factor_keep_mask, :],
            gene_pheno_Y[factor_keep_mask, :].T,
            multivariate=True,
            covs=state.Y[factor_keep_mask],
            **phewas_beta_kwargs
        )
        state._accumulate_factor_phewas_outputs("Y", beta_tilde, se, z_score, p_value, one_sided_p_value)

        if not options.debug_skip_huber:
            _, _, beta_tilde, se, z_score, p_value, one_sided_p_value = state._calculate_phewas_block(
                input_values[factor_keep_mask, :],
                gene_pheno_Y[factor_keep_mask, :].T,
                multivariate=True,
                covs=state.Y[factor_keep_mask],
                huber=True,
                **phewas_beta_kwargs
            )
            state._accumulate_factor_phewas_outputs("Y", beta_tilde, se, z_score, p_value, one_sided_p_value, huber=True)

    if gene_pheno_combined_prior_Ys is not None and not options.debug_skip_correlation:
        _, _, beta_tilde, se, z_score, p_value, one_sided_p_value = state._calculate_phewas_block(
            input_values[factor_keep_mask, :],
            gene_pheno_combined_prior_Ys[factor_keep_mask, :].T,
            X_orig=state.X_orig[factor_keep_mask, :],
            X_phewas_beta=state.X_phewas_beta[begin:end, :] if state.X_phewas_beta is not None else None,
            Y_resid=gene_pheno_Y[factor_keep_mask, :].T,
            multivariate=True,
            covs=state.combined_prior_Ys[factor_keep_mask] if state.combined_prior_Ys is not None else state.Y[factor_keep_mask],
            **phewas_beta_kwargs
        )
        state._accumulate_factor_phewas_outputs("combined_prior_Ys", beta_tilde, se, z_score, p_value, one_sided_p_value)

        if not options.debug_skip_huber:
            _, _, beta_tilde, se, z_score, p_value, one_sided_p_value = state._calculate_phewas_block(
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
            state._accumulate_factor_phewas_outputs("combined_prior_Ys", beta_tilde, se, z_score, p_value, one_sided_p_value, huber=True)


def run_standard_phewas_batch(state, input_values, gene_pheno_Y, gene_pheno_combined_prior_Ys, begin, end, phewas_beta_kwargs, *, options):
    if gene_pheno_Y is not None:
        beta, _, beta_tilde, se, z_score, p_value, _ = state._calculate_phewas_block(
            input_values,
            gene_pheno_Y.T,
            **phewas_beta_kwargs
        )
        assert beta.shape[0] == 3, "First dimension of beta should be 3, not (%s, %s)" % (beta.shape[0], beta.shape[1])
        state._accumulate_standard_phewas_outputs("pheno_Y", beta, beta_tilde, se, z_score, p_value)

    if gene_pheno_combined_prior_Ys is not None and not options.debug_skip_correlation:
        beta, _, beta_tilde, se, z_score, p_value, _ = state._calculate_phewas_block(
            input_values,
            gene_pheno_combined_prior_Ys.T,
            X_orig=state.X_orig,
            X_phewas_beta=state.X_phewas_beta[begin:end, :] if state.X_phewas_beta is not None else None,
            Y_resid=gene_pheno_Y.T,
            **phewas_beta_kwargs
        )
        assert beta.shape[0] == 3, "First dimension of beta should be 3, not (%s, %s)" % (beta.shape[0], beta.shape[1])
        state._accumulate_standard_phewas_outputs("pheno_combined_prior_Ys", beta, beta_tilde, se, z_score, p_value)


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
        phenos, pheno_to_ind, col_info = state._prepare_phewas_phenos_from_file(
            gene_phewas_bfs_in=gene_phewas_bfs_in,
            gene_phewas_bfs_id_col=gene_phewas_bfs_id_col,
            gene_phewas_bfs_pheno_col=gene_phewas_bfs_pheno_col,
            gene_phewas_bfs_log_bf_col=gene_phewas_bfs_log_bf_col,
            gene_phewas_bfs_combined_col=gene_phewas_bfs_combined_col,
            gene_phewas_bfs_prior_col=gene_phewas_bfs_prior_col,
        )
    else:
        phenos = state.phenos

    num_batches = int(np.ceil(len(phenos) / batch_size))
    input_values, factor_keep_mask = state._build_phewas_input_values(
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
            gene_pheno_Y, gene_pheno_combined_prior_Ys, gene_pheno_priors = state._read_phewas_file_batch(
                gene_phewas_bfs_in=gene_phewas_bfs_in,
                begin=begin,
                cur_batch_size=cur_batch_size,
                pheno_to_ind=pheno_to_ind,
                id_col=col_info["id_col"],
                pheno_col=col_info["pheno_col"],
                bf_col=col_info["bf_col"],
                combined_col=col_info["combined_col"],
                prior_col=col_info["prior_col"],
            )
        else:
            gene_pheno_Y = state.gene_pheno_Y[:, begin:end].toarray() if state.gene_pheno_Y is not None else None
            gene_pheno_combined_prior_Ys = (
                state.gene_pheno_combined_prior_Ys[:, begin:end].toarray()
                if state.gene_pheno_combined_prior_Ys is not None
                else None
            )

        if run_for_factors:
            state._run_factor_phewas_batch(
                input_values=input_values,
                factor_keep_mask=factor_keep_mask,
                gene_pheno_Y=gene_pheno_Y,
                gene_pheno_combined_prior_Ys=gene_pheno_combined_prior_Ys,
                begin=begin,
                end=end,
                phewas_beta_kwargs=phewas_beta_kwargs,
            )
        else:
            state._run_standard_phewas_batch(
                input_values=input_values,
                gene_pheno_Y=gene_pheno_Y,
                gene_pheno_combined_prior_Ys=gene_pheno_combined_prior_Ys,
                begin=begin,
                end=end,
                phewas_beta_kwargs=phewas_beta_kwargs,
            )
