from __future__ import annotations

import copy
import numpy as np
import pegs_shared.phewas as pegs_phewas
from scipy import sparse

from . import main_support as pigean_main_support
from . import runtime as pigean_runtime


_temporary_state_fields = pigean_runtime.temporary_state_fields
_STATE_FIELDS_SAMPLER_HYPER = pigean_runtime.STATE_FIELDS_SAMPLER_HYPER


def run_advanced_set_b_phewas_beta_sampling_if_requested(services, state, options, beta_sampling_kwargs):
    if not options.betas_uncorrected_from_phewas:
        return
    phewas_beta_sampling_kwargs = dict(beta_sampling_kwargs)
    phewas_beta_sampling_kwargs.update({
        "run_betas_using_phewas": options.betas_from_phewas,
        "run_uncorrected_using_phewas": True,
    })
    state.calculate_non_inf_betas(state.p, **phewas_beta_sampling_kwargs)


def run_advanced_set_b_output_phewas_if_requested(services, state, options):
    if not options.run_phewas:
        return
    if options.gene_phewas_bfs_prior_col is not None:
        services.log(
            "Ignoring --gene-phewas-bfs-prior-col for the gene-level PheWAS output stage; phenotype-side prior outputs are not implemented",
            services.INFO,
        )
    decision = pigean_main_support.resolve_gene_phewas_input_decision_for_stage(
        requested_input=options.run_phewas_input,
        reusable_inputs=[options.gene_phewas_bfs_in],
        read_gene_phewas=state.read_gene_phewas(),
        num_gene_phewas_filtered=state.num_gene_phewas_filtered,
    )
    services.log(
        "PheWAS stage 'output_phewas': mode=%s reason=%s" % (decision.mode, decision.reason),
        services.INFO,
    )
    bfs_to_use = decision.resolved_input

    phewas_config = pigean_main_support.build_phewas_stage_config(
        gene_phewas_bfs_in=bfs_to_use,
        gene_phewas_bfs_id_col=options.gene_phewas_bfs_id_col,
        gene_phewas_bfs_pheno_col=options.gene_phewas_bfs_pheno_col,
        gene_phewas_bfs_log_bf_col=options.gene_phewas_bfs_log_bf_col,
        gene_phewas_bfs_combined_col=options.gene_phewas_bfs_combined_col,
        gene_phewas_bfs_prior_col=None,
        min_value=getattr(options, "min_gene_phewas_read_value", None),
        phewas_comparison_set=getattr(options, "phewas_comparison_set", "matched"),
        max_num_burn_in=options.max_num_burn_in,
        max_num_iter=options.max_num_iter_betas,
        min_num_iter=options.min_num_iter_betas,
        num_chains=options.num_chains_betas,
        r_threshold_burn_in=options.r_threshold_burn_in_betas,
        use_max_r_for_convergence=options.use_max_r_for_convergence_betas,
        max_frac_sem=options.max_frac_sem_betas,
        gauss_seidel=options.gauss_seidel_betas,
        sparse_solution=options.sparse_solution,
        sparse_frac_betas=options.sparse_frac_betas,
    )
    run_kwargs = phewas_config.to_run_kwargs()
    run_kwargs["batch_size"] = 1500
    state.run_phewas(**run_kwargs)

    if options.phewas_stats_out:
        state.write_phewas_statistics(options.phewas_stats_out)


def prepare_phewas_phenos_from_file(
    state,
    gene_phewas_bfs_in,
    *,
    gene_phewas_bfs_id_col=None,
    gene_phewas_bfs_pheno_col=None,
    gene_phewas_bfs_log_bf_col=None,
    gene_phewas_bfs_combined_col=None,
    gene_phewas_bfs_prior_col=None,
    open_text_fn,
    get_col_fn,
    construct_map_to_ind_fn,
    warn_fn,
    log_fn,
    debug_level,
):
    phenos, pheno_to_ind, col_info = pegs_phewas.prepare_phewas_phenos_from_file(
        state,
        gene_phewas_bfs_in,
        gene_phewas_bfs_id_col=gene_phewas_bfs_id_col,
        gene_phewas_bfs_pheno_col=gene_phewas_bfs_pheno_col,
        gene_phewas_bfs_log_bf_col=gene_phewas_bfs_log_bf_col,
        gene_phewas_bfs_combined_col=gene_phewas_bfs_combined_col,
        gene_phewas_bfs_prior_col=gene_phewas_bfs_prior_col,
        open_text_fn=open_text_fn,
        get_col_fn=get_col_fn,
        construct_map_to_ind_fn=construct_map_to_ind_fn,
        warn_fn=warn_fn,
        log_fn=log_fn,
        debug_level=debug_level,
    )
    return phenos, pheno_to_ind, {
        'id_col': col_info.id_col,
        'pheno_col': col_info.pheno_col,
        'bf_col': col_info.bf_col,
        'combined_col': col_info.combined_col,
        'prior_col': col_info.prior_col,
    }


def read_phewas_file_batch(
    state,
    gene_phewas_bfs_in,
    *,
    begin,
    cur_batch_size,
    pheno_to_ind,
    id_col,
    pheno_col,
    bf_col,
    combined_col,
    prior_col,
    open_text_fn,
    warn_fn,
):
    return pegs_phewas.read_phewas_file_batch(
        state,
        gene_phewas_bfs_in,
        begin=begin,
        cur_batch_size=cur_batch_size,
        pheno_to_ind=pheno_to_ind,
        col_info={
            'id_col': id_col,
            'pheno_col': pheno_col,
            'bf_col': bf_col,
            'combined_col': combined_col,
            'prior_col': prior_col,
        },
        open_text_fn=open_text_fn,
        warn_fn=warn_fn,
    )


def accumulate_phewas_outputs(state, output_prefix, beta, beta_tilde, se, z_score, p_value):
    return pegs_phewas.accumulate_standard_phewas_outputs(state, output_prefix, beta, beta_tilde, se, z_score, p_value)


def accumulate_selected_gene_level_phewas_outputs(state, comparisons, beta, beta_tilde, se, z_score, p_value):
    return pegs_phewas.accumulate_selected_gene_level_phewas_outputs(
        state,
        comparisons,
        beta,
        beta_tilde,
        se,
        z_score,
        p_value,
    )


def build_phewas_input_values(state):
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
    return np.exp(input_values + state.background_bf) / (1 + np.exp(input_values + state.background_bf))


def get_enabled_gene_level_phewas_comparisons(state, comparison_set):
    comparisons = []
    for spec in pegs_phewas.iter_enabled_gene_level_phewas_comparisons(comparison_set):
        if spec["axis_name"] == "Y" and state.Y is None:
            continue
        if spec["axis_name"] == "combined_prior_Ys" and state.combined_prior_Ys is None:
            continue
        if spec["axis_name"] == "priors" and state.priors is None:
            continue
        comparisons.append(spec)
    return tuple(comparisons)


def calculate_combined_phewas_block_with_sparse_correlation(
    state,
    input_values,
    gene_pheno_combined_prior_Ys,
    *,
    begin,
    end,
    gene_pheno_Y,
    phewas_beta_kwargs,
):
    """
    Run the combined phenotype-support PheWAS block with sparse residual-correlation correction.

    The correction is only applied to the combined phenotype-support outcome family.
    Covariance is injected at the beta-tilde / SE stage through a sparse residual-correlation
    estimate induced by overlapping gene sets. The later non-infinitesimal shrinkage step
    still uses the independent approximation in `_calculate_non_inf_betas(...)`.
    """
    return calculate_phewas_block(
        state,
        input_values,
        gene_pheno_combined_prior_Ys.T,
        X_orig=state.X_orig,
        X_phewas_beta=state.X_phewas_beta[begin:end, :] if state.X_phewas_beta is not None else None,
        Y_resid=gene_pheno_Y.T if gene_pheno_Y is not None else None,
        **phewas_beta_kwargs,
    )


def stage_gene_level_phewas_file_once(
    state,
    gene_phewas_bfs_in,
    *,
    gene_phewas_bfs_id_col,
    gene_phewas_bfs_pheno_col,
    gene_phewas_bfs_log_bf_col,
    gene_phewas_bfs_combined_col,
    min_value,
    open_text_fn,
    get_col_fn,
    bail_fn,
    warn_fn,
):
    parsed = pegs_phewas.parse_gene_phewas_bfs_file(
        gene_phewas_bfs_in,
        gene_phewas_bfs_id_col=gene_phewas_bfs_id_col,
        gene_phewas_bfs_pheno_col=gene_phewas_bfs_pheno_col,
        gene_phewas_bfs_log_bf_col=gene_phewas_bfs_log_bf_col,
        gene_phewas_bfs_combined_col=gene_phewas_bfs_combined_col,
        gene_phewas_bfs_prior_col=None,
        min_value=min_value,
        max_num_entries_at_once=None,
        existing_phenos=state.phenos,
        existing_pheno_to_ind=state.pheno_to_ind,
        gene_to_ind=state.gene_to_ind,
        gene_label_map=state.gene_label_map,
        phewas_gene_to_x_gene=None,
        open_text_fn=open_text_fn,
        get_col_fn=get_col_fn,
        bail_fn=bail_fn,
        warn_fn=warn_fn,
    )
    num_prior_phenos = len(state.phenos) if state.phenos is not None else 0
    pegs_phewas.expand_phewas_state_for_added_phenos(state, len(parsed.phenos) - num_prior_phenos)
    state.phenos = parsed.phenos
    state.pheno_to_ind = parsed.pheno_to_ind
    state.num_gene_phewas_filtered = parsed.num_filtered

    shape = (len(state.genes), len(parsed.phenos))
    gene_pheno_Y = None
    gene_pheno_combined_prior_Ys = None
    if parsed.Ys is not None:
        gene_pheno_Y = sparse.csc_matrix((parsed.Ys, (parsed.row, parsed.col)), shape=shape)
    if parsed.combineds is not None:
        gene_pheno_combined_prior_Ys = sparse.csc_matrix((parsed.combineds, (parsed.row, parsed.col)), shape=shape)
    return parsed.phenos, gene_pheno_Y, gene_pheno_combined_prior_Ys


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
):
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
        pigean_main_support.log("Processing block batch %s" % (batch), pigean_main_support.TRACE)
        begin = batch * cor_batch_size
        end = min((batch + 1) * cor_batch_size, beta_tildes.shape[0])

        if X_phewas_beta is not None and X_orig is not None and not state.debug_skip_correlation:
            if X_phewas_beta.shape[0] != Y_mat.shape[0]:
                pigean_main_support.bail(
                    "When calling this, the phewas_betas must have same number of phenos as Y_mat: shapes are X_phewas=(%d,%d) vs. Y_mat=(%d,%d)"
                    % (X_phewas_beta.shape[0], X_phewas_beta.shape[1], Y_mat.shape[0], Y_mat.shape[1])
                )
            dot_threshold = 0.01 * 0.01
            pigean_main_support.log("Calculating correlation matrix for use in residuals", pigean_main_support.DEBUG)
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
            pigean_main_support.log(
                "Sparsity of correlation matrix is %d/%d=%.3g (size %.3gMb)" % (nnz, total, float(nnz) / total, nnz * 8 / (1024 * 1024)),
                pigean_main_support.DEBUG,
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
                    covs=covs if not state.debug_skip_phewas_covs else None,
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
                    covs=covs if not state.debug_skip_phewas_covs else None,
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

    with _temporary_state_fields(
        state,
        overrides={"ps": None, "sigma2s": None},
        restore_fields=_STATE_FIELDS_SAMPLER_HYPER,
    ) as hyper_snapshot:
        orig_p = hyper_snapshot["p"]
        orig_sigma2_internal = hyper_snapshot["sigma2"]
        orig_sigma_power = hyper_snapshot["sigma_power"]

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

    return (
        (betas_uncorrected / scale_factors).T,
        postp_uncorrected.T,
        (beta_tildes / scale_factors).T,
        (ses / scale_factors).T,
        z_scores.T,
        p_values.T,
        one_sided_p_values.T,
    )


def run_phewas(
    state,
    gene_phewas_bfs_in=None,
    gene_phewas_bfs_id_col=None,
    gene_phewas_bfs_pheno_col=None,
    gene_phewas_bfs_log_bf_col=None,
    gene_phewas_bfs_combined_col=None,
    gene_phewas_bfs_prior_col=None,
    min_value=None,
    phewas_comparison_set="matched",
    max_num_burn_in=1000,
    max_num_iter=1100,
    min_num_iter=10,
    num_chains=10,
    r_threshold_burn_in=1.01,
    use_max_r_for_convergence=True,
    max_frac_sem=0.01,
    gauss_seidel=False,
    sparse_solution=False,
    sparse_frac_betas=None,
    batch_size=1500,
    *,
    bail_fn,
    warn_fn,
    log_fn,
    info_level,
    debug_level,
    trace_level,
    open_text_fn,
    get_col_fn,
    construct_map_to_ind_fn,
    **kwargs,
):
    bail = bail_fn
    warn = warn_fn
    log = log_fn
    INFO = info_level
    TRACE = trace_level

    if gene_phewas_bfs_in is None and not state.read_gene_phewas():
        bail("Require --gene-stats-in or --gene-phewas-bfs-in with a column for log_bf/Y in this operation")

    if state.genes is None:
        warn("Cannot run phewas without X matrix; skipping")
        return
    if state.Y is None and state.combined_prior_Ys is None and state.priors is None:
        warn("Cannot run phewas without Y values; skipping")
        return

    log("Running phewas", INFO)
    read_file = gene_phewas_bfs_in is not None

    if read_file:
        phenos, staged_gene_pheno_Y, staged_gene_pheno_combined_prior_Ys = stage_gene_level_phewas_file_once(
            state,
            gene_phewas_bfs_in,
            gene_phewas_bfs_id_col=gene_phewas_bfs_id_col,
            gene_phewas_bfs_pheno_col=gene_phewas_bfs_pheno_col,
            gene_phewas_bfs_log_bf_col=gene_phewas_bfs_log_bf_col,
            gene_phewas_bfs_combined_col=gene_phewas_bfs_combined_col,
            min_value=min_value,
            open_text_fn=open_text_fn,
            get_col_fn=get_col_fn,
            bail_fn=bail_fn,
            warn_fn=warn_fn,
        )
    else:
        phenos = state.phenos
        staged_gene_pheno_Y = state.gene_pheno_Y
        staged_gene_pheno_combined_prior_Ys = state.gene_pheno_combined_prior_Ys

    enabled_comparisons = get_enabled_gene_level_phewas_comparisons(state, phewas_comparison_set)
    direct_comparisons = tuple(spec for spec in enabled_comparisons if spec["output_family"] == "pheno_Y")
    combined_comparisons = tuple(spec for spec in enabled_comparisons if spec["output_family"] == "pheno_combined_prior_Ys")

    num_batches = int(np.ceil(len(phenos) / batch_size))
    input_values = build_phewas_input_values(state)
    phewas_beta_kwargs = {
        'max_num_burn_in': max_num_burn_in,
        'max_num_iter': max_num_iter,
        'min_num_iter': min_num_iter,
        'num_chains': num_chains,
        'r_threshold_burn_in': r_threshold_burn_in,
        'use_max_r_for_convergence': use_max_r_for_convergence,
        'max_frac_sem': max_frac_sem,
        'gauss_seidel': gauss_seidel,
        'sparse_solution': sparse_solution,
        'sparse_frac_betas': sparse_frac_betas,
        'non_inf_kwargs': kwargs,
    }

    for batch in range(num_batches):
        log("Getting phenos block batch %s" % (batch), TRACE)
        begin = batch * batch_size
        end = (batch + 1) * batch_size
        if end > len(phenos):
            end = len(phenos)

        cur_batch_size = end - begin
        log("Processing phenos %d-%d" % (begin + 1, end))

        gene_pheno_Y = staged_gene_pheno_Y[:, begin:end].toarray() if staged_gene_pheno_Y is not None else None
        gene_pheno_combined_prior_Ys = (
            staged_gene_pheno_combined_prior_Ys[:, begin:end].toarray()
            if staged_gene_pheno_combined_prior_Ys is not None
            else None
        )

        if gene_pheno_Y is not None and direct_comparisons:
            beta, _, beta_tilde, se, Z, p_value, _ = calculate_phewas_block(state, input_values, gene_pheno_Y.T, **phewas_beta_kwargs)
            assert beta.shape[0] == 3, "First dimension of beta should be 3, not (%s, %s)" % (beta.shape[0], beta.shape[1])
            accumulate_selected_gene_level_phewas_outputs(state, direct_comparisons, beta, beta_tilde, se, Z, p_value)

        if gene_pheno_combined_prior_Ys is not None and not state.debug_skip_correlation and combined_comparisons:
            beta, _, beta_tilde, se, Z, p_value, _ = calculate_combined_phewas_block_with_sparse_correlation(
                state,
                input_values,
                gene_pheno_combined_prior_Ys,
                begin=begin,
                end=end,
                gene_pheno_Y=gene_pheno_Y,
                phewas_beta_kwargs=phewas_beta_kwargs,
            )
            assert beta.shape[0] == 3, "First dimension of beta should be 3, not (%s, %s)" % (beta.shape[0], beta.shape[1])
            accumulate_selected_gene_level_phewas_outputs(state, combined_comparisons, beta, beta_tilde, se, Z, p_value)
