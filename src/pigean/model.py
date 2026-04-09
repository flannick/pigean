from __future__ import annotations

import copy

import numpy as np
import scipy
import scipy.sparse as sparse

from pegs_shared.io_common import clean_chrom_name, open_text_with_retry
import pegs_shared.regression as pegs_regression
from . import runtime as pigean_runtime


def open_gz(file, flag=None):
    return open_text_with_retry(file, flag=flag)


def compute_beta_tildes(
    runtime,
    X,
    Y,
    y_var=None,
    scale_factors=None,
    mean_shifts=None,
    resid_correlation_matrix=None,
    *,
    calc_x_shift_scale_fn,
    bail_fn,
    log_fun,
    debug_level,
):
    return pegs_regression.compute_beta_tildes(
        X,
        Y,
        y_var=y_var,
        scale_factors=scale_factors,
        mean_shifts=mean_shifts,
        resid_correlation_matrix=resid_correlation_matrix,
        calc_x_shift_scale_fn=calc_x_shift_scale_fn,
        finalize_regression_fn=runtime._finalize_regression,
        bail_fn=bail_fn,
        log_fun=log_fun,
        debug_level=debug_level,
    )


def compute_multivariate_beta_tildes(
    runtime,
    X,
    Y,
    resid_correlation_matrix=None,
    add_intercept=True,
    covs=None,
):
    return pegs_regression.compute_multivariate_beta_tildes(
        X,
        Y,
        resid_correlation_matrix=resid_correlation_matrix,
        add_intercept=add_intercept,
        covs=covs,
        finalize_regression_fn=runtime._finalize_regression,
    )


def compute_logistic_beta_tildes(
    runtime,
    X,
    Y,
    scale_factors=None,
    mean_shifts=None,
    resid_correlation_matrix=None,
    convert_to_dichotomous=True,
    rel_tol=0.01,
    X_stacked=None,
    append_pseudo=True,
    *,
    calc_x_shift_scale_fn,
    bail_fn,
    log_fun,
    debug_level,
    trace_level,
):
    return pegs_regression.compute_logistic_beta_tildes(
        X,
        Y,
        scale_factors=scale_factors,
        mean_shifts=mean_shifts,
        resid_correlation_matrix=resid_correlation_matrix,
        convert_to_dichotomous=convert_to_dichotomous,
        rel_tol=rel_tol,
        X_stacked=X_stacked,
        append_pseudo=append_pseudo,
        calc_x_shift_scale_fn=calc_x_shift_scale_fn,
        finalize_regression_fn=runtime._finalize_regression,
        bail_fn=bail_fn,
        log_fun=log_fun,
        debug_level=debug_level,
        trace_level=trace_level,
        runtime_Y=runtime.Y,
        runtime_Y_for_regression=runtime.Y_for_regression,
    )


def finalize_regression(runtime, beta_tildes, ses, se_inflation_factors, *, log_fn, warn_fn, trace_level):
    return pegs_regression.finalize_regression_outputs(
        beta_tildes,
        ses,
        se_inflation_factors,
        log_fn=log_fn,
        warn_fn=warn_fn,
        trace_level=trace_level,
    )


def correct_beta_tildes(
    runtime,
    beta_tildes,
    ses,
    se_inflation_factors,
    total_qc_metrics,
    total_qc_metrics_directions,
    correct_mean=True,
    correct_var=True,
    add_missing=True,
    add_ignored=True,
    correct_ignored=False,
    fit=True,
    *,
    log_fn,
    warn_fn,
    trace_level,
    debug_level,
):
    return pegs_regression.correct_beta_tildes(
        runtime,
        beta_tildes,
        ses,
        se_inflation_factors,
        total_qc_metrics,
        total_qc_metrics_directions,
        correct_mean=correct_mean,
        correct_var=correct_var,
        add_missing=add_missing,
        add_ignored=add_ignored,
        correct_ignored=correct_ignored,
        fit=fit,
        compute_beta_tildes_fn=runtime._compute_beta_tildes,
        log_fn=log_fn,
        warn_fn=warn_fn,
        trace_level=trace_level,
        debug_level=debug_level,
    )


def build_inner_beta_sampler_common_kwargs(options):
    return dict(
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


def calc_priors_from_betas(X, betas_m, mean_shifts, scale_factors):
    # Compute per-chain log-prior odds (relative to background) from betas.
    return np.array(
        X.dot((betas_m / scale_factors).T)
        - np.sum(mean_shifts * betas_m / scale_factors, axis=1).T
    ).T


def finalize_gibbs_priors_for_sampling(
    state,
    priors_sample_m,
    priors_mean_m,
    priors_missing_sample_m,
    priors_missing_mean_m,
    adjust_priors,
    use_mean_betas,
    priors_percentage_max_sample_m,
    priors_percentage_max_mean_m,
    priors_adjustment_sample_m,
    priors_adjustment_mean_m,
    *,
    log_fn,
    trace_level,
):
    # Regress out gene-length trend from priors (when requested), then choose
    # mean/sample priors used for the next iteration's Y sampling.
    total_priors_m = np.hstack((priors_sample_m, priors_missing_sample_m))
    gene_N = state.get_gene_N()
    gene_N_missing = state.get_gene_N(get_missing=True)

    all_gene_N = gene_N
    if state.genes_missing is not None:
        assert gene_N_missing is not None
        all_gene_N = np.concatenate((all_gene_N, gene_N_missing))

    priors_slope = total_priors_m.dot(all_gene_N) / (total_priors_m.shape[1] * np.var(all_gene_N))

    if adjust_priors:
        log_fn(
            "Adjusting priors with slopes ranging from %.4g-%.4g"
            % (np.min(priors_slope), np.max(priors_slope)),
            trace_level,
        )
        priors_sample_m = priors_sample_m - np.outer(priors_slope, gene_N)
        priors_mean_m = priors_mean_m - np.outer(priors_slope, gene_N)

        if state.genes_missing is not None:
            priors_missing_sample_m = priors_missing_sample_m - np.outer(priors_slope, gene_N_missing)
            priors_missing_mean_m = priors_missing_mean_m - np.outer(priors_slope, gene_N_missing)

    priors_for_Y_m = priors_sample_m
    priors_percentage_max_for_Y_m = priors_percentage_max_sample_m
    priors_adjustment_for_Y_m = priors_adjustment_sample_m
    if use_mean_betas:
        priors_for_Y_m = priors_mean_m
        priors_percentage_max_for_Y_m = priors_percentage_max_mean_m
        priors_adjustment_for_Y_m = priors_adjustment_mean_m

    return {
        "priors_sample_m": priors_sample_m,
        "priors_mean_m": priors_mean_m,
        "priors_missing_sample_m": priors_missing_sample_m,
        "priors_missing_mean_m": priors_missing_mean_m,
        "priors_for_Y_m": priors_for_Y_m,
        "priors_percentage_max_for_Y_m": priors_percentage_max_for_Y_m,
        "priors_adjustment_for_Y_m": priors_adjustment_for_Y_m,
    }


def compute_gibbs_uncorrected_betas_and_defaults(
    state,
    full_beta_tildes_m,
    full_ses_m,
    full_scale_factors_m,
    full_mean_shifts_m,
    full_is_dense_gene_set_m,
    full_ps_m,
    full_sigma2s_m,
    passed_in_max_num_burn_in,
    max_num_iter_betas,
    min_num_iter_betas,
    num_chains_betas,
    r_threshold_burn_in_betas,
    use_max_r_for_convergence_betas,
    max_frac_sem_betas,
    max_allowed_batch_correlation,
    gauss_seidel_betas,
    sparse_solution,
    sparse_frac_betas,
):
    # Independent run provides sparse screening inputs and fallback values for
    # gene sets later filtered out from corrected-beta updates.
    (
        uncorrected_betas_sample_m,
        uncorrected_postp_sample_m,
        uncorrected_betas_mean_m,
        uncorrected_postp_mean_m,
    ) = state._calculate_non_inf_betas(
        assume_independent=True,
        initial_p=None,
        beta_tildes=full_beta_tildes_m,
        ses=full_ses_m,
        V=None,
        X_orig=None,
        scale_factors=full_scale_factors_m,
        mean_shifts=full_mean_shifts_m,
        is_dense_gene_set=full_is_dense_gene_set_m,
        ps=full_ps_m,
        sigma2s=full_sigma2s_m,
        return_sample=True,
        max_num_burn_in=passed_in_max_num_burn_in,
        max_num_iter=max_num_iter_betas,
        min_num_iter=min_num_iter_betas,
        num_chains=num_chains_betas,
        r_threshold_burn_in=r_threshold_burn_in_betas,
        use_max_r_for_convergence=use_max_r_for_convergence_betas,
        max_frac_sem=max_frac_sem_betas,
        max_allowed_batch_correlation=max_allowed_batch_correlation,
        gauss_seidel=gauss_seidel_betas,
        update_hyper_sigma=False,
        update_hyper_p=False,
        sparse_solution=sparse_solution,
        sparse_frac_betas=sparse_frac_betas,
        debug_gene_sets=state.gene_sets,
    )

    (
        default_betas_sample_m,
        default_postp_sample_m,
        default_betas_mean_m,
        default_postp_mean_m,
    ) = (
        copy.copy(uncorrected_betas_sample_m),
        copy.copy(uncorrected_postp_sample_m),
        copy.copy(uncorrected_betas_mean_m),
        copy.copy(uncorrected_postp_mean_m),
    )
    return {
        "uncorrected_betas_sample_m": uncorrected_betas_sample_m,
        "uncorrected_postp_sample_m": uncorrected_postp_sample_m,
        "uncorrected_betas_mean_m": uncorrected_betas_mean_m,
        "uncorrected_postp_mean_m": uncorrected_postp_mean_m,
        "default_betas_sample_m": default_betas_sample_m,
        "default_postp_sample_m": default_postp_sample_m,
        "default_betas_mean_m": default_betas_mean_m,
        "default_postp_mean_m": default_postp_mean_m,
    }


def _get_tracked_ignored_gene_set_mask(state):
    track_mask = getattr(state, "gene_set_track_beta_uncorrected_ignored", None)
    if track_mask is None:
        return None
    track_mask = np.asarray(track_mask, dtype=bool)
    if track_mask.size == 0 or not np.any(track_mask):
        return None
    return track_mask


def update_tracked_ignored_uncorrected_betas(
    state,
    *,
    beta_tildes,
    ses,
    scale_factors,
    mean_shifts,
    return_sample=False,
    debug_gene_sets=None,
    **inner_beta_kwargs,
):
    track_mask = _get_tracked_ignored_gene_set_mask(state)
    if (
        not getattr(state, "track_filtered_beta_uncorrected", False)
        or track_mask is None
        or getattr(state, "X_orig_ignored_gene_sets", None) is None
        or beta_tildes is None
        or ses is None
    ):
        return None

    beta_tildes_arr = np.asarray(beta_tildes)
    num_tracked = beta_tildes_arr.shape[-1] if beta_tildes_arr.ndim > 1 else beta_tildes_arr.shape[0]

    ignored_is_dense = np.zeros(num_tracked, dtype=bool)
    if getattr(state, "is_dense_gene_set_ignored", None) is not None:
        ignored_is_dense = np.asarray(state.is_dense_gene_set_ignored, dtype=bool)[track_mask]

    ignored_ps = np.full(num_tracked, state.p, dtype=float)
    if getattr(state, "ps_ignored", None) is not None and len(state.ps_ignored) == len(track_mask):
        ignored_ps = np.asarray(state.ps_ignored)[track_mask]

    ignored_sigma2s = np.full(num_tracked, state.sigma2, dtype=float)
    if getattr(state, "sigma2s_ignored", None) is not None and len(state.sigma2s_ignored) == len(track_mask):
        ignored_sigma2s = np.asarray(state.sigma2s_ignored)[track_mask]

    if debug_gene_sets is None and state.gene_sets_ignored is not None:
        debug_gene_sets = [state.gene_sets_ignored[i] for i in range(len(state.gene_sets_ignored)) if track_mask[i]]

    result = state._calculate_non_inf_betas(
        assume_independent=True,
        initial_p=None,
        beta_tildes=beta_tildes,
        ses=ses,
        V=None,
        X_orig=None,
        scale_factors=scale_factors,
        mean_shifts=mean_shifts,
        is_dense_gene_set=ignored_is_dense,
        ps=ignored_ps,
        sigma2s=ignored_sigma2s,
        return_sample=return_sample,
        update_hyper_sigma=False,
        update_hyper_p=False,
        debug_gene_sets=debug_gene_sets,
        **inner_beta_kwargs,
    )

    if return_sample:
        (
            tracked_betas_sample_m,
            tracked_postp_sample_m,
            tracked_betas_mean_m,
            tracked_postp_mean_m,
        ) = result
    else:
        tracked_betas_mean_m, tracked_postp_mean_m = result
        tracked_betas_sample_m = None
        tracked_postp_sample_m = None

    full_betas_uncorrected = np.zeros(len(state.gene_sets_ignored))
    full_postps = np.zeros(len(state.gene_sets_ignored))
    full_cond_betas = np.zeros(len(state.gene_sets_ignored))
    full_betas_uncorrected[track_mask] = tracked_betas_mean_m
    full_postps[track_mask] = tracked_postp_mean_m
    positive_postp_mask = tracked_postp_mean_m > 0
    tracked_cond_betas = np.array(tracked_betas_mean_m, copy=True)
    tracked_cond_betas[positive_postp_mask] = (
        tracked_betas_mean_m[positive_postp_mask] / tracked_postp_mean_m[positive_postp_mask]
    )
    full_cond_betas[track_mask] = tracked_cond_betas

    state.betas_uncorrected_ignored = full_betas_uncorrected
    state.non_inf_avg_postps_ignored = full_postps
    state.non_inf_avg_cond_betas_ignored = full_cond_betas

    return {
        "track_mask": track_mask,
        "betas_uncorrected_mean_m": tracked_betas_mean_m,
        "postp_mean_m": tracked_postp_mean_m,
        "betas_uncorrected_sample_m": tracked_betas_sample_m,
        "postp_sample_m": tracked_postp_sample_m,
    }


def compute_gibbs_tracked_ignored_uncorrected_betas(
    state,
    Y_sample_m,
    y_corr_sparse,
    *,
    inner_beta_kwargs_linear,
):
    track_mask = _get_tracked_ignored_gene_set_mask(state)
    if (
        not getattr(state, "track_filtered_beta_uncorrected", False)
        or track_mask is None
        or getattr(state, "X_orig_ignored_gene_sets", None) is None
        or state.X_orig_ignored_gene_sets.shape[1] != int(np.sum(track_mask))
    ):
        return None

    (
        ignored_beta_tildes_m,
        ignored_ses_m,
        ignored_z_scores_m,
        ignored_p_values_m,
        _ignored_se_inflation_factors_m,
        _ignored_alpha_tildes_m,
        _ignored_diverged_m,
    ) = state._compute_logistic_beta_tildes(
        state.X_orig_ignored_gene_sets,
        Y_sample_m,
        state.scale_factors_ignored[track_mask] if state.scale_factors_ignored is not None else None,
        state.mean_shifts_ignored[track_mask] if state.mean_shifts_ignored is not None else None,
        resid_correlation_matrix=y_corr_sparse,
    )

    ignored_setup = update_tracked_ignored_uncorrected_betas(
        state,
        beta_tildes=ignored_beta_tildes_m,
        ses=ignored_ses_m,
        scale_factors=np.tile(state.scale_factors_ignored[track_mask], (ignored_beta_tildes_m.shape[0], 1))
        if state.scale_factors_ignored is not None
        else None,
        mean_shifts=np.tile(state.mean_shifts_ignored[track_mask], (ignored_beta_tildes_m.shape[0], 1))
        if state.mean_shifts_ignored is not None
        else None,
        return_sample=True,
        debug_gene_sets=[state.gene_sets_ignored[i] for i in range(len(state.gene_sets_ignored)) if track_mask],
        **inner_beta_kwargs_linear,
    )
    if ignored_setup is None:
        return None
    ignored_setup["beta_tildes_m"] = ignored_beta_tildes_m
    ignored_setup["z_scores_m"] = ignored_z_scores_m
    ignored_setup["p_values_m"] = ignored_p_values_m
    return ignored_setup


def build_non_inf_beta_sampler_kwargs(inner_beta_kwargs):
    return {
        "max_num_burn_in": inner_beta_kwargs["passed_in_max_num_burn_in"],
        "max_num_iter": inner_beta_kwargs["max_num_iter_betas"],
        "min_num_iter": inner_beta_kwargs["min_num_iter_betas"],
        "num_chains": inner_beta_kwargs["num_chains_betas"],
        "r_threshold_burn_in": inner_beta_kwargs["r_threshold_burn_in_betas"],
        "use_max_r_for_convergence": inner_beta_kwargs["use_max_r_for_convergence_betas"],
        "max_frac_sem": inner_beta_kwargs["max_frac_sem_betas"],
        "max_allowed_batch_correlation": inner_beta_kwargs["max_allowed_batch_correlation"],
        "gauss_seidel": inner_beta_kwargs["gauss_seidel_betas"],
        "sparse_solution": inner_beta_kwargs["sparse_solution"],
        "sparse_frac_betas": inner_beta_kwargs["sparse_frac_betas"],
    }


def compute_gibbs_iteration_priors_from_betas(
    state,
    full_betas_sample_m,
    full_betas_mean_m,
    priors_missing_sample_m,
    priors_missing_mean_m,
):
    priors_sample_m = calc_priors_from_betas(
        state.X_orig,
        full_betas_sample_m,
        state.mean_shifts,
        state.scale_factors,
    )
    priors_mean_m = calc_priors_from_betas(
        state.X_orig,
        full_betas_mean_m,
        state.mean_shifts,
        state.scale_factors,
    )
    if state.genes_missing is not None:
        priors_missing_sample_m = calc_priors_from_betas(
            state.X_orig_missing_genes,
            full_betas_sample_m,
            state.mean_shifts,
            state.scale_factors,
        )
        priors_missing_mean_m = calc_priors_from_betas(
            state.X_orig_missing_genes,
            full_betas_mean_m,
            state.mean_shifts,
            state.scale_factors,
        )
    return (
        priors_sample_m,
        priors_mean_m,
        priors_missing_sample_m,
        priors_missing_mean_m,
    )


def calculate_gene_set_statistics(state, gwas_in=None, exomes_in=None, positive_controls_in=None, positive_controls_list=None, case_counts_in=None, ctrl_counts_in=None, gene_bfs_in=None, Y=None, show_progress=True, max_gene_set_p=None, run_logistic=True, max_for_linear=0.95, run_corrected_ols=False, use_sampling_for_betas=None, correct_betas_mean=True, correct_betas_var=True, gene_loc_file=None, gene_cor_file=None, gene_cor_file_gene_col=1, gene_cor_file_cor_start_col=10, skip_V=False, run_using_phewas=False, *, bail_fn, warn_fn, log_fn, info_level, debug_level, trace_level, run_read_y_stage_fn, **kwargs):
    bail = bail_fn
    warn = warn_fn
    log = log_fn
    INFO = info_level
    DEBUG = debug_level
    TRACE = trace_level
    _run_read_y_stage = run_read_y_stage_fn
    if state.X_orig is None:
        bail("Error: X is required")
    #now calculate the betas and p-values

    log("Calculating gene set statistics", INFO)

    if run_using_phewas:
        Y = state.gene_pheno_Y.T.toarray()
        if Y is None:
            bail("Need --gene-phewas-bfs in order to run beta calculation with phewas")

    if Y is None:
        Y = state.Y_for_regression

    if Y is None:
        if gwas_in is None and exomes_in is None and gene_bfs_in is None and positive_controls_in is None and positive_controls_list is None and case_counts_in is None and ctrl_counts_in is None:
            bail(
                "Need --gwas-in or --exomes-in or --gene-stats-in or "
                "--gene-list-in/--positive-controls-in or --case-counts_in"
            )

        log("Reading Y within calculate_gene_set_statistics; parameters may not be honored")
        _run_read_y_stage(
            state,
            gwas_in=gwas_in,
            exomes_in=exomes_in,
            positive_controls_in=positive_controls_in,
            positive_controls_list=positive_controls_list,
            case_counts_in=case_counts_in,
            ctrl_counts_in=ctrl_counts_in,
            gene_bfs_in=gene_bfs_in,
            **kwargs
        )
        Y = state.Y_for_regression

    if run_corrected_ols and state.y_corr is None:
        correlation_m = state._read_correlations(gene_cor_file, gene_loc_file, gene_cor_file_gene_col=gene_cor_file_gene_col, gene_cor_file_cor_start_col=gene_cor_file_cor_start_col)

        #convert X and Y to their new values
        min_correlation = 0.05
        state._set_Y(state.Y, state.Y_for_regression, state.Y_exomes, state.Y_positive_controls, state.Y_case_counts, Y_corr_m=correlation_m, store_corr_sparse=run_corrected_ols, skip_V=True, skip_scale_factors=True, min_correlation=min_correlation)
        Y = state.Y_for_regression

    #subset gene sets to remove empty ones first
    #number of gene sets in each gene set
    col_sums = state.get_col_sums(state.X_orig, num_nonzero=True)
    state.subset_gene_sets(col_sums > 0, keep_missing=False, ignore_missing=True, skip_V=True, skip_scale_factors=True, filter_reason="empty_after_gene_filter")

    state._set_scale_factors()

    #state.is_logistic = run_logistic

    #if the maximum Y is large, switch to logistic regression (to avoid being too strong)
    Y_to_use = Y
    Y = np.exp(Y_to_use + state.background_log_bf) / (1 + np.exp(Y_to_use + state.background_log_bf))

    if not run_logistic and np.max(Y) > max_for_linear and (use_sampling_for_betas is None or use_sampling_for_betas < 1):
        log("Switching to logistic sampling due to high Y values", DEBUG)
        run_logistic = True
        use_sampling_for_betas = 1

    if use_sampling_for_betas is not None:
        state._record_param("sampling_for_betas", use_sampling_for_betas)

    if use_sampling_for_betas is not None and use_sampling_for_betas > 0:

        #handy option in case we want to see what sampling looks like outside of gibbs
        if run_using_phewas:
            avg_beta_tildes = np.zeros((state.gene_pheno_Y.shape[1],len(state.gene_sets)))
            avg_z_scores = np.zeros((state.gene_pheno_Y.shape[1],len(state.gene_sets)))
        else:
            avg_beta_tildes = np.zeros(len(state.gene_sets))
            avg_z_scores = np.zeros(len(state.gene_sets))
        tot_its = 0
        for iteration_num in range(use_sampling_for_betas):
            log("Sampling iteration %d..." % (iteration_num+1))
            p_sample_m = np.zeros(Y.shape)
            p_sample_m[np.random.random(Y.shape) < Y] = 1
            Y_sample_m = p_sample_m

            (beta_tildes, ses, z_scores, p_values, se_inflation_factors, alpha_tildes, diverged) = state._compute_logistic_beta_tildes(state.X_orig, Y_sample_m, state.scale_factors, state.mean_shifts, resid_correlation_matrix=state.y_corr_sparse)

            avg_beta_tildes += beta_tildes
            avg_z_scores += z_scores
            tot_its += 1

        beta_tildes = avg_beta_tildes / tot_its

        z_scores = avg_z_scores / tot_its

        p_values = 2*scipy.stats.norm.cdf(-np.abs(z_scores))
        ses = np.full(beta_tildes.shape, 100.0)
        ses[z_scores != 0] = np.abs(beta_tildes[z_scores != 0] / z_scores[z_scores != 0])

        se_inflation_factors = None

    elif run_logistic:
        (beta_tildes, ses, z_scores, p_values, se_inflation_factors, alpha_tildes, diverged) = state._compute_logistic_beta_tildes(state.X_orig, Y, state.scale_factors, state.mean_shifts, resid_correlation_matrix=state.y_corr_sparse)

    else:
        #Technically, we could use the above code for this case, since X_blocks will returned unwhitened matrix
        #But, probably faster to keep sparse multiplication? Might be worth revisiting later to see if there actually is a performance gain
        #We can use original X here because whitening support was removed with GLS.
        assert(not state.scale_is_for_whitened)
        Y = copy.copy(Y)

        if len(Y.shape) > 1:
            y_var = np.var(Y, axis=1)
        else:
            y_var = np.var(Y)

        (beta_tildes, ses, z_scores, p_values, se_inflation_factors) = state._compute_beta_tildes(state.X_orig, Y, y_var, state.scale_factors, state.mean_shifts, resid_correlation_matrix=state.y_corr_sparse)

    if correct_betas_mean or correct_betas_var:
        (beta_tildes, ses, z_scores, p_values, se_inflation_factors) = state._correct_beta_tildes(beta_tildes, ses, se_inflation_factors, state.total_qc_metrics, state.total_qc_metrics_directions, correct_mean=correct_betas_mean, correct_var=correct_betas_var, fit=False)

    if run_using_phewas:
        (state.beta_tildes_phewas, state.z_scores_phewas, state.p_values_phewas, state.ses_phewas, state.se_inflation_factors_phewas) = (beta_tildes, z_scores, p_values, ses, se_inflation_factors)
        if len(state.beta_tildes_phewas.shape) == 1:
            state.beta_tildes_phewas = state.beta_tildes_phewas[np.newaxis,:]
            state.ses_phewas = state.ses_phewas[np.newaxis,:]
            state.z_scores_phewas = state.z_scores_phewas[np.newaxis,:]
            state.p_values_phewas = state.p_values_phewas[np.newaxis,:]
            if state.se_inflation_factors_phewas is not None:
                state.se_inflation_factors_phewas = state.se_inflation_factors_phewas[np.newaxis,:]
    else:
        (state.beta_tildes, state.z_scores, state.p_values, state.ses, state.se_inflation_factors) = (beta_tildes, z_scores, p_values, ses, se_inflation_factors)

        if state.gene_sets_missing is None:
            state.X_orig_missing_gene_sets = None
            state.mean_shifts_missing = None
            state.scale_factors_missing = None
            state.is_dense_gene_set_missing = None
            state.ps_missing = None
            state.sigma2s_missing = None

            state.beta_tildes_missing = None
            state.p_values_missing = None
            state.ses_missing = None
            state.z_scores_missing = None

            state.total_qc_metrics_missing = None
            state.mean_qc_metrics_missing = None

        if max_gene_set_p is not None:
            gene_set_mask = state.p_values <= max_gene_set_p
            if np.sum(gene_set_mask) == 0 and len(state.p_values) > 0:
                gene_set_mask = state.p_values == np.min(state.p_values)
            log("Keeping %d gene sets that passed threshold of p<%.3g" % (np.sum(gene_set_mask), max_gene_set_p))
            state.subset_gene_sets(
                gene_set_mask,
                keep_missing=not getattr(state, "track_filtered_beta_uncorrected", False),
                ignore_missing=getattr(state, "track_filtered_beta_uncorrected", False),
                skip_V=True,
                filter_reason="max_gene_set_p",
            )

            if len(state.gene_sets) < 1:
                log("No gene sets left!")
                return

    #state.max_gene_set_p = max_gene_set_p



def calculate_non_inf_betas(state, p, max_num_burn_in=1000, max_num_iter=1100, min_num_iter=10, num_chains=10, r_threshold_burn_in=1.01, use_max_r_for_convergence=True, max_frac_sem=0.01, gauss_seidel=False, update_hyper_sigma=True, update_hyper_p=True, sparse_solution=False, pre_filter_batch_size=None, pre_filter_small_batch_size=500, sparse_frac_betas=None, betas_trace_out=None, run_betas_using_phewas=False, run_uncorrected_using_phewas=False, independent_only=False, *, bail_fn, warn_fn, log_fn, info_level, debug_level, trace_level, **kwargs):
    bail = bail_fn
    warn = warn_fn
    log = log_fn
    INFO = info_level
    DEBUG = debug_level
    TRACE = trace_level

    run_using_phewas = run_betas_using_phewas or run_uncorrected_using_phewas

    log("Calculating betas")
    if independent_only and run_using_phewas:
        bail("Option --independent-betas-only is not supported with the PheWAS beta path")
    if run_using_phewas:
        (beta_tildes_to_use, ses_to_use) = (state.beta_tildes_phewas, state.ses_phewas)
    else:
        (beta_tildes_to_use, ses_to_use) = (state.beta_tildes, state.ses)
    tracked_ignored_mask = _get_tracked_ignored_gene_set_mask(state)
    ignored_beta_sampler_kwargs = build_non_inf_beta_sampler_kwargs(
        {
            "passed_in_max_num_burn_in": max_num_burn_in,
            "max_num_iter_betas": max_num_iter,
            "min_num_iter_betas": min_num_iter,
            "num_chains_betas": num_chains,
            "r_threshold_burn_in_betas": r_threshold_burn_in,
            "use_max_r_for_convergence_betas": use_max_r_for_convergence,
            "max_frac_sem_betas": max_frac_sem,
            "max_allowed_batch_correlation": kwargs.get("max_allowed_batch_correlation"),
            "gauss_seidel_betas": gauss_seidel,
            "sparse_solution": sparse_solution,
            "sparse_frac_betas": sparse_frac_betas,
        }
    )

    if not run_using_phewas or run_uncorrected_using_phewas:
        result_uncorrected = state._calculate_non_inf_betas(p, beta_tildes=beta_tildes_to_use, ses=ses_to_use, max_num_burn_in=max_num_burn_in, max_num_iter=max_num_iter, min_num_iter=min_num_iter, num_chains=num_chains, r_threshold_burn_in=r_threshold_burn_in, use_max_r_for_convergence=use_max_r_for_convergence, max_frac_sem=max_frac_sem, gauss_seidel=gauss_seidel, update_hyper_sigma=False, update_hyper_p=False, sparse_solution=sparse_solution, sparse_frac_betas=sparse_frac_betas, assume_independent=True, V=None, **kwargs)

    avg_betas_v = np.zeros(len(state.gene_sets))
    avg_postp_v = np.zeros(len(state.gene_sets))

    if run_using_phewas:
        initial_run_mask = np.full(len(state.gene_sets), True)
    else:
        (avg_betas_uncorrected_v, avg_postp_uncorrected_v) = result_uncorrected
        initial_run_mask = avg_betas_uncorrected_v != 0

    if independent_only:
        state.betas = None
        state.betas_r_hat = None
        state.betas_mcse = None
        state.betas_uncorrected = copy.copy(avg_betas_uncorrected_v)
        state.non_inf_avg_postps = copy.copy(avg_postp_uncorrected_v)
        state.non_inf_avg_cond_betas = copy.copy(avg_betas_uncorrected_v)
        positive_postp_mask = state.non_inf_avg_postps > 0
        state.non_inf_avg_cond_betas[positive_postp_mask] /= state.non_inf_avg_postps[positive_postp_mask]
        if state.gene_sets_missing is not None:
            if state.betas_missing is None:
                state.betas_missing = np.zeros(len(state.gene_sets_missing))
            if state.betas_uncorrected_missing is None:
                state.betas_uncorrected_missing = np.zeros(len(state.gene_sets_missing))
            if state.non_inf_avg_postps_missing is None:
                state.non_inf_avg_postps_missing = np.zeros(len(state.gene_sets_missing))
            if state.non_inf_avg_cond_betas_missing is None:
                state.non_inf_avg_cond_betas_missing = np.zeros(len(state.gene_sets_missing))
        if tracked_ignored_mask is not None and state.beta_tildes_ignored is not None and state.ses_ignored is not None:
            update_tracked_ignored_uncorrected_betas(
                state,
                beta_tildes=state.beta_tildes_ignored[tracked_ignored_mask],
                ses=state.ses_ignored[tracked_ignored_mask],
                scale_factors=state.scale_factors_ignored[tracked_ignored_mask] if state.scale_factors_ignored is not None else None,
                mean_shifts=state.mean_shifts_ignored[tracked_ignored_mask] if state.mean_shifts_ignored is not None else None,
                **ignored_beta_sampler_kwargs,
            )
        return

    run_mask = copy.copy(initial_run_mask)

    if pre_filter_batch_size is not None and np.sum(initial_run_mask) > pre_filter_batch_size:
        state._record_param("pre_filter_batch_size_orig", pre_filter_batch_size)

        num_batches = state._get_num_X_blocks(state.X_orig[:,initial_run_mask], batch_size=pre_filter_small_batch_size)
        if num_batches > 1:
            #try to run with small batches to see if we can zero out more
            gene_set_masks = state._compute_gene_set_batches(V=None, X_orig=state.X_orig[:,initial_run_mask], mean_shifts=state.mean_shifts[initial_run_mask], scale_factors=state.scale_factors[initial_run_mask], find_correlated_instead=pre_filter_small_batch_size)
            if len(gene_set_masks) > 0:
                if np.sum(gene_set_masks[-1]) == 1 and len(gene_set_masks) > 1:
                    #merge singletons at the end into the one before
                    gene_set_masks[-2][gene_set_masks[-1]] = True
                    gene_set_masks = gene_set_masks[:-1]
                if np.sum(gene_set_masks[0]) > 1:
                    V_data = []
                    V_rows = []
                    V_cols = []
                    for gene_set_mask in gene_set_masks:
                        V_block = state._calculate_V_internal(state.X_orig[:,initial_run_mask][:,gene_set_mask], state.y_corr_cholesky, state.mean_shifts[initial_run_mask][gene_set_mask], state.scale_factors[initial_run_mask][gene_set_mask])
                        orig_indices = np.where(gene_set_mask)[0]
                        V_rows += list(np.repeat(orig_indices, V_block.shape[0]))
                        V_cols += list(np.tile(orig_indices, V_block.shape[0]))
                        V_data += list(V_block.ravel())

                    V_sparse = sparse.csc_matrix((V_data, (V_rows, V_cols)), shape=(np.sum(initial_run_mask), np.sum(initial_run_mask)))

                    log("Running %d blocks to check for zeros..." % len(gene_set_masks), DEBUG)
                    (avg_betas_half_corrected_v, avg_postp_half_corrected_v) = state._calculate_non_inf_betas(p, V=V_sparse, X_orig=None, scale_factors=state.scale_factors[initial_run_mask], mean_shifts=state.mean_shifts[initial_run_mask], is_dense_gene_set=state.is_dense_gene_set[initial_run_mask], ps=state.ps[initial_run_mask], sigma2s=state.sigma2s[initial_run_mask], max_num_burn_in=max_num_burn_in, max_num_iter=max_num_iter, min_num_iter=min_num_iter, num_chains=num_chains, r_threshold_burn_in=r_threshold_burn_in, use_max_r_for_convergence=use_max_r_for_convergence, max_frac_sem=max_frac_sem, gauss_seidel=gauss_seidel, update_hyper_sigma=update_hyper_sigma, update_hyper_p=update_hyper_p, sparse_solution=sparse_solution, sparse_frac_betas=sparse_frac_betas, **kwargs)

                    add_zero_mask = avg_betas_half_corrected_v == 0

                    if np.any(add_zero_mask):
                        #need to convert these to the original gene sets
                        map_to_full = np.where(initial_run_mask)[0]
                        #get rows and then columns in subsetted
                        set_to_zero_full = np.where(add_zero_mask)
                        #map columns in subsetted to original
                        set_to_zero_full = map_to_full[set_to_zero_full]
                        orig_zero = np.sum(run_mask)
                        run_mask[set_to_zero_full] = False
                        new_zero = np.sum(run_mask)
                        log("Found %d additional zero gene sets" % (orig_zero - new_zero),DEBUG)

    if np.sum(~run_mask) > 0:
        log("Set additional %d gene sets to zero based on uncorrected betas" % np.sum(~run_mask))

    if np.sum(run_mask) == 0 and state.p_values is not None:
        run_mask[np.argmax(state.p_values)] = True

    if run_using_phewas:
        (beta_tildes_to_use, ses_to_use) = (state.beta_tildes_phewas[:,run_mask], state.ses_phewas[:,run_mask])
    else:
        (beta_tildes_to_use, ses_to_use) = (state.beta_tildes[run_mask], state.ses[run_mask])

    if not run_using_phewas or run_betas_using_phewas:
        result = state._calculate_non_inf_betas(p, beta_tildes=beta_tildes_to_use, ses=ses_to_use, X_orig=state.X_orig[:,run_mask], scale_factors=state.scale_factors[run_mask], mean_shifts=state.mean_shifts[run_mask], V=None, ps=state.ps[run_mask] if state.ps is not None else None, sigma2s=state.sigma2s[run_mask] if state.sigma2s is not None else None, is_dense_gene_set=state.is_dense_gene_set[run_mask], max_num_burn_in=max_num_burn_in, max_num_iter=max_num_iter, min_num_iter=min_num_iter, num_chains=num_chains, r_threshold_burn_in=r_threshold_burn_in, use_max_r_for_convergence=use_max_r_for_convergence, max_frac_sem=max_frac_sem, gauss_seidel=gauss_seidel, update_hyper_sigma=update_hyper_sigma, update_hyper_p=update_hyper_p, sparse_solution=sparse_solution, sparse_frac_betas=sparse_frac_betas, betas_trace_out=betas_trace_out, betas_trace_gene_sets=[state.gene_sets[i] for i in range(len(state.gene_sets)) if run_mask[i]], debug_gene_sets=[state.gene_sets[i] for i in range(len(state.gene_sets)) if run_mask[i]], **kwargs)

    if run_using_phewas:
        if run_betas_using_phewas:
            state.betas_phewas = _expand_phewas_gene_set_result(result[0], run_mask, len(state.gene_sets))
        if run_uncorrected_using_phewas:
            state.betas_uncorrected_phewas = _expand_phewas_gene_set_result(
                result_uncorrected[0],
                initial_run_mask,
                len(state.gene_sets),
            )

    else:
        (avg_betas_v[run_mask], avg_postp_v[run_mask]) = result

        if len(avg_betas_v.shape) == 2:
            avg_betas_v = np.mean(avg_betas_v, axis=0)
            avg_postp_v = np.mean(avg_postp_v, axis=0)

        state.betas = copy.copy(avg_betas_v)
        state.betas_uncorrected = copy.copy(avg_betas_uncorrected_v)

        state.non_inf_avg_postps = copy.copy(avg_postp_v)
        state.non_inf_avg_cond_betas = copy.copy(avg_betas_v)
        state.non_inf_avg_cond_betas[avg_postp_v > 0] /= avg_postp_v[avg_postp_v > 0]

        if state.gene_sets_missing is not None:
            if state.betas_missing is None:
                state.betas_missing = np.zeros(len(state.gene_sets_missing))
            if state.betas_uncorrected_missing is None:
                state.betas_uncorrected_missing = np.zeros(len(state.gene_sets_missing))
            if state.non_inf_avg_postps_missing is None:
                state.non_inf_avg_postps_missing = np.zeros(len(state.gene_sets_missing))
            if state.non_inf_avg_cond_betas_missing is None:
                state.non_inf_avg_cond_betas_missing = np.zeros(len(state.gene_sets_missing))
        if tracked_ignored_mask is not None and state.beta_tildes_ignored is not None and state.ses_ignored is not None:
            update_tracked_ignored_uncorrected_betas(
                state,
                beta_tildes=state.beta_tildes_ignored[tracked_ignored_mask],
                ses=state.ses_ignored[tracked_ignored_mask],
                scale_factors=state.scale_factors_ignored[tracked_ignored_mask] if state.scale_factors_ignored is not None else None,
                mean_shifts=state.mean_shifts_ignored[tracked_ignored_mask] if state.mean_shifts_ignored is not None else None,
                **ignored_beta_sampler_kwargs,
            )


def _expand_phewas_gene_set_result(values, mask, num_gene_sets):
    expanded_values = np.array(values, copy=True)
    if expanded_values.ndim == 1:
        expanded_values = expanded_values[np.newaxis, :]
    full_values = np.zeros((expanded_values.shape[0], num_gene_sets))
    full_values[:, mask] = expanded_values
    return full_values



def calculate_priors(state, max_gene_set_p=None, num_gene_batches=None, correct_betas_mean=True, correct_betas_var=True, gene_loc_file=None, gene_cor_file=None, gene_cor_file_gene_col=1, gene_cor_file_cor_start_col=10, p_noninf=None, run_logistic=True, max_for_linear=0.95, adjust_priors=False, tag="", *, bail_fn, warn_fn, log_fn, info_level, debug_level, trace_level, **kwargs):
    bail = bail_fn
    warn = warn_fn
    log = log_fn
    INFO = info_level
    DEBUG = debug_level
    TRACE = trace_level
    _temporary_unsubset_gene_sets = pigean_runtime.temporary_unsubset_gene_sets
    pegs_clean_chrom_name = clean_chrom_name
    # ==========================================================================
    # Prior Phase 0: Validate prerequisites and choose batching strategy.
    # ==========================================================================
    if state.X_orig is None:
        bail("X is required for this operation")
    if state.betas is None:
        bail("betas are required for this operation")

    use_X = False

    assert(state.gene_sets is not None)
    max_num_gene_batches_together = 10000
    #if 0, don't use any V
    num_gene_batches_parallel = int(max_num_gene_batches_together / len(state.gene_sets))
    if num_gene_batches_parallel == 0:
        use_X = True
        log("Using low memory X instead of V in priors", TRACE)
        num_gene_batches_parallel = 1

    loco = False
    if num_gene_batches is None:
        log("Doing leave-one-chromosome-out cross validation for priors computation")
        loco = True

    if num_gene_batches is not None and num_gene_batches < 2:
        # ==========================================================================
        # Prior Phase 1a: Single-pass projection from betas to priors.
        # ==========================================================================
        #this calculates the values for the non missing genes
        #use original X matrix here because we are rescaling betas back to those units
        priors = np.array(state.X_orig.dot(state.betas / state.scale_factors) - np.sum(state.mean_shifts * state.betas / state.scale_factors)).flatten()
        state.combined_prior_Ys = None
        state.combined_prior_Ys_for_regression = None
        state.combined_prior_Ys_adj = None
        state.combined_prior_Y_ses = None
        state.combined_Ds = None
        state.batches = None
    else:
        # ==========================================================================
        # Prior Phase 1b: Build batch metadata (LOCO or correlation-aware batches).
        # ==========================================================================

        if loco:
            if gene_loc_file is None:
                bail("Need --gene-loc-file for --loco")

            gene_chromosomes = {}
            batches = set()
            log("Reading gene locations")
            if state.gene_to_chrom is None:
                state.gene_to_chrom = {}
            if state.gene_to_pos is None:
                state.gene_to_pos = {}

            with open_gz(gene_loc_file) as gene_loc_fh:
                for line in gene_loc_fh:
                    cols = line.strip('\n').split()
                    if len(cols) != 6:
                        bail("Format for --gene-loc-file is:\n\tgene_id\tchrom\tstart\tstop\tstrand\tgene_name\nOffending line:\n\t%s" % line)
                    gene_name = cols[5]
                    if gene_name not in state.gene_to_ind:
                        continue

                    chrom = pegs_clean_chrom_name(cols[1])
                    pos1 = int(cols[2])
                    pos2 = int(cols[3])

                    state.gene_to_chrom[gene_name] = chrom
                    state.gene_to_pos[gene_name] = (pos1,pos2)

                    batches.add(chrom)
                    gene_chromosomes[gene_name] = chrom
            batches = sorted(batches)
            num_gene_batches = len(batches)
        else:
            #need sorted genes and correlation matrix to batch genes
            if state.y_corr is None:
                correlation_m = state._read_correlations(gene_cor_file, gene_loc_file, gene_cor_file_gene_col=gene_cor_file_gene_col, gene_cor_file_cor_start_col=gene_cor_file_cor_start_col)
                state._set_Y(state.Y, state.Y_for_regression, state.Y_exomes, state.Y_positive_controls, state.Y_case_counts, Y_corr_m=correlation_m, skip_V=True, skip_scale_factors=True, min_correlation=None)
            batches = range(num_gene_batches)

        gene_batch_size = int(len(state.genes) / float(num_gene_batches) + 1)
        state.batches = [None] * len(state.genes)
        priors = np.zeros(len(state.genes))

        #store a matrix of all beta_tildes across all batches
        full_matrix_shape = (len(batches), len(state.gene_sets) + (len(state.gene_sets_missing) if state.gene_sets_missing is not None else 0))
        full_beta_tildes_m = np.zeros(full_matrix_shape)
        full_ses_m = np.zeros(full_matrix_shape)
        full_z_scores_m = np.zeros(full_matrix_shape)
        full_se_inflation_factors_m = np.zeros(full_matrix_shape)
        full_p_values_m = np.zeros(full_matrix_shape)
        full_scale_factors_m = np.zeros(full_matrix_shape)
        full_ps_m = None
        if state.ps is not None:
            full_ps_m = np.zeros(full_matrix_shape)                
        full_sigma2s_m = None
        if state.sigma2s is not None:
            full_sigma2s_m = np.zeros(full_matrix_shape)                

        full_is_dense_gene_set_m = np.zeros(full_matrix_shape, dtype=bool)
        full_mean_shifts_m = np.zeros(full_matrix_shape)
        full_include_mask_m = np.zeros((len(batches), len(state.genes)), dtype=bool)
        full_priors_mask_m = np.zeros((len(batches), len(state.genes)), dtype=bool)

        # ==========================================================================
        # Prior Phase 2: Per-batch beta-tilde estimation on subsetted genes.
        # ==========================================================================
        # combine X_orig and X_orig_missing for batched prior calculations.
        with _temporary_unsubset_gene_sets(state, state.gene_sets_missing is not None, keep_missing=True, skip_V=True):

            for batch_ind in range(len(batches)):
                batch = batches[batch_ind]

                #specify:
                # (a) include_mask: the genes that are used for calculating beta tildes and betas for this batch
                # (b) priors_mask: the genes that we will calculate priors for
                #these are not exact complements because we may need to exlude some genes for both (i.e. a buffer)
                if loco:
                    include_mask = np.array([True] * len(state.genes))
                    priors_mask = np.array([False] * len(state.genes))
                    for i in range(len(state.genes)):
                        if state.genes[i] not in gene_chromosomes:
                            include_mask[i] = False
                            priors_mask[i] = True
                        elif gene_chromosomes[state.genes[i]] == batch:
                            include_mask[i] = False
                            priors_mask[i] = True
                        else:
                            include_mask[i] = True
                            priors_mask[i] = False
                    log("Batch %s: %d genes" % (batch, np.sum(priors_mask)))
                else:
                    begin = batch * gene_batch_size
                    end = (batch + 1) * gene_batch_size
                    if end > len(state.genes):
                        end = len(state.genes)
                    end = end - 1
                    log("Batch %d: genes %d - %d" % (batch+1, begin, end))


                    #include only genes not correlated with any in the current batch
                    include_mask = np.array([True] * len(state.genes))

                    include_mask_begin = begin - 1
                    while include_mask_begin > 0 and (begin - include_mask_begin) < len(state.y_corr) and state.y_corr[begin - include_mask_begin][include_mask_begin] > 0:
                        include_mask_begin -= 1
                    include_mask_begin += 1

                    include_mask_end = end + 1
                    while (include_mask_end - end) < len(state.y_corr) and state.y_corr[include_mask_end - end][end] > 0:
                        include_mask_end += 1
                    include_mask[include_mask_begin:include_mask_end] = False
                    include_mask_end -= 1

                    priors_mask = np.array([False] * len(state.genes))
                    priors_mask[begin:(end+1)] = True


                for i in range(len(state.genes)):
                    if priors_mask[i]:
                        state.batches[i] = batch

                #now subset Y
                Y = copy.copy(state.Y_for_regression)
                y_corr = None
                y_corr_sparse = None

                if state.y_corr is not None:
                    y_corr = copy.copy(state.y_corr)
                    if not loco:
                        #we cannot rely on chromosome boundaries to zero out correlations, so manually do this
                        for i in range(include_mask_begin - 1, include_mask_begin - y_corr.shape[0], -1):
                            y_corr[include_mask_begin - i:,i] = 0
                    #don't need to zero out anything for include_mask_end because correlations between after end and removed are all stored inside of the removed indices
                    y_corr = y_corr[:,include_mask]

                    if state.y_corr_sparse is not None:
                        y_corr_sparse = state.y_corr_sparse[include_mask,:][:,include_mask]

                Y = Y[include_mask]
                y_var = np.var(Y)

                #DO WE NEED THIS??
                #y_mean = np.mean(Y)
                #Y = Y - y_mean

                (mean_shifts, scale_factors) = state._calc_X_shift_scale(state.X_orig[include_mask,:])

                #if some gene sets became empty!
                assert(not np.any(np.logical_and(mean_shifts != 0, scale_factors == 0)))
                mean_shifts[mean_shifts == 0] = 0
                scale_factors[scale_factors == 0] = 1

                ps = state.ps
                sigma2s = state.sigma2s
                is_dense_gene_set = state.is_dense_gene_set

                #max_gene_set_p = state.max_gene_set_p if state.max_gene_set_p is not None else 1

                Y_to_use = Y
                D = np.exp(Y_to_use + state.background_log_bf) / (1 + np.exp(Y_to_use + state.background_log_bf))
                if np.max(D) > max_for_linear:
                    run_logistic = True

                #compute special beta tildes here
                if run_logistic:
                    (beta_tildes, ses, z_scores, p_values, se_inflation_factors, alpha_tildes, diverged) = state._compute_logistic_beta_tildes(state.X_orig[include_mask,:], D, scale_factors, mean_shifts, resid_correlation_matrix=y_corr_sparse)
                else:
                    (beta_tildes, ses, z_scores, p_values, se_inflation_factors) = state._compute_beta_tildes(state.X_orig[include_mask,:], Y, y_var, scale_factors, mean_shifts, resid_correlation_matrix=y_corr_sparse)

                if correct_betas_mean or correct_betas_var:
                    (beta_tildes, ses, z_scores, p_values, se_inflation_factors) = state._correct_beta_tildes(beta_tildes, ses, se_inflation_factors, state.total_qc_metrics, state.total_qc_metrics_directions, correct_mean=correct_betas_mean, correct_var=correct_betas_var, fit=False)

                #now determine those that have too many genes removed to be accurate
                mean_reduction = float(num_gene_batches - 1) / float(num_gene_batches)
                sd_reduction = np.sqrt(mean_reduction * (1 - mean_reduction))
                reduction = mean_shifts / state.mean_shifts
                ignore_mask = reduction < mean_reduction - 3 * sd_reduction
                if sum(ignore_mask) > 0:
                    log("Ignoring %d gene sets because there are too many genes are missing from this batch" % sum(ignore_mask))
                    for ind in np.array(range(len(ignore_mask)))[ignore_mask]:
                        log("%s: %.4g remaining (vs. %.4g +/- %.4g expected)" % (state.gene_sets[ind], reduction[ind], mean_reduction, sd_reduction), TRACE)
                #also zero out anything above the p-value threshold; this is a convenience for below
                #note that p-values are still preserved though for below
                ignore_mask = np.logical_or(ignore_mask, p_values > max_gene_set_p)

                beta_tildes[ignore_mask] = 0
                ses[ignore_mask] = max(state.ses) * 100

                full_beta_tildes_m[batch_ind,:] = beta_tildes
                full_ses_m[batch_ind,:] = ses
                full_z_scores_m[batch_ind,:] = z_scores
                full_se_inflation_factors_m[batch_ind,:] = se_inflation_factors
                full_p_values_m[batch_ind,:] = p_values
                full_scale_factors_m[batch_ind,:] = scale_factors
                full_mean_shifts_m[batch_ind,:] = mean_shifts
                if full_ps_m is not None:
                    full_ps_m[batch_ind,:] = ps
                if full_sigma2s_m is not None:
                    full_sigma2s_m[batch_ind,:] = sigma2s

                full_is_dense_gene_set_m[batch_ind,:] = is_dense_gene_set
                full_include_mask_m[batch_ind,:] = include_mask
                full_priors_mask_m[batch_ind,:] = priors_mask

            # ==========================================================================
            # Prior Phase 3: Fit non-inf betas per batch window and back-project priors.
            # ==========================================================================
            #now calculate everything
            if p_noninf is None or p_noninf >= 1:
                num_gene_batches_parallel = 1
            num_calculations = int(np.ceil(num_gene_batches / num_gene_batches_parallel))
            for calc in range(num_calculations):
                begin = calc * num_gene_batches_parallel
                end = (calc + 1) * num_gene_batches_parallel
                if end > num_gene_batches:
                    end = num_gene_batches

                log("Running calculations for batches %d-%d" % (begin, end))

                #ensure there is at least one gene set remaining
                max_gene_set_p_v = np.min(full_p_values_m[begin:end,:], axis=1)
                #max_gene_set_p_v[max_gene_set_p_v < (state.max_gene_set_p if state.max_gene_set_p is not None else 1)] = (state.max_gene_set_p if state.max_gene_set_p is not None else 1)
                max_gene_set_p_v[max_gene_set_p_v < (max_gene_set_p if max_gene_set_p is not None else 1)] = (max_gene_set_p if max_gene_set_p is not None else 1)

                #get the include mask; any batch has p <= threshold
                new_gene_set_mask = np.max(full_p_values_m[begin:end,:].T <= max_gene_set_p_v, axis=1)
                num_gene_set_mask = np.sum(new_gene_set_mask)

                #we unsubset genes to aid in batching; this caused sigma and p to be affected
                fraction_non_missing = np.mean(new_gene_set_mask)
                missing_scale_factor = state._get_fraction_non_missing() / fraction_non_missing
                if missing_scale_factor > 1 / state.p:
                    #threshold this here. otherwise set_p will cap p but set_sigma won't cap sigma
                    missing_scale_factor = 1 / state.p

                #orig_sigma2 = state.sigma2
                #orig_p = state.p
                #state.set_sigma(state.sigma2 * missing_scale_factor, state.sigma_power, sigma2_osc=state.sigma2_osc)
                #state.set_p(state.p * missing_scale_factor)

                #construct the V matrix
                if not use_X:
                    V_m = np.zeros((end-begin, num_gene_set_mask, num_gene_set_mask))
                    for i,j in zip(range(begin, end),range(end-begin)):
                        include_mask = full_include_mask_m[i,:]

                        V_m[j,:,:] = state._calculate_V_internal(state.X_orig[include_mask,:][:,new_gene_set_mask], None, full_mean_shifts_m[i,new_gene_set_mask], full_scale_factors_m[i,new_gene_set_mask])
                else:
                    V_m = None

                cur_beta_tildes = full_beta_tildes_m[begin:end,:][:,new_gene_set_mask]
                cur_ses = full_ses_m[begin:end,:][:,new_gene_set_mask]
                cur_se_inflation_factors = full_se_inflation_factors_m[begin:end,:][:,new_gene_set_mask]
                cur_scale_factors = full_scale_factors_m[begin:end,:][:,new_gene_set_mask]
                cur_mean_shifts = full_mean_shifts_m[begin:end,:][:,new_gene_set_mask]
                cur_is_dense_gene_set = full_is_dense_gene_set_m[begin:end,:][:,new_gene_set_mask]
                cur_ps = None
                if full_ps_m is not None:
                    cur_ps = full_ps_m[begin:end,:][:,new_gene_set_mask]
                cur_sigma2s = None
                if full_sigma2s_m is not None:
                    cur_sigma2s = full_sigma2s_m[begin:end,:][:,new_gene_set_mask]

                #only non inf now
                (betas, avg_postp) = state._calculate_non_inf_betas(None, beta_tildes=cur_beta_tildes, ses=cur_ses, V=V_m, X_orig=state.X_orig[include_mask,:][:,new_gene_set_mask], scale_factors=cur_scale_factors, mean_shifts=cur_mean_shifts, is_dense_gene_set=cur_is_dense_gene_set, ps=cur_ps, sigma2s=cur_sigma2s, update_hyper_sigma=False, update_hyper_p=False, num_missing_gene_sets=int((1 - fraction_non_missing) * len(state.gene_sets)), **kwargs)
                if len(betas.shape) == 1:
                    betas = betas[np.newaxis,:]


                for i,j in zip(range(begin, end),range(end-begin)):

                    priors[full_priors_mask_m[i,:]] = np.array(state.X_orig[full_priors_mask_m[i,:],:][:,new_gene_set_mask].dot(betas[j,:] / cur_scale_factors[j,:]))

                #now restore the p and sigma
                #state.set_sigma(orig_sigma2, state.sigma_power, sigma2_osc=state.sigma2_osc)
                #state.set_p(orig_p)

    # ==========================================================================
    # Prior Phase 4: Merge missing-gene priors, center values, and finalize.
    # ==========================================================================
    #now for the genes that were not included in X
    if state.X_orig_missing_genes is not None:
        #these can use the original betas because they were never included
        state.priors_missing = np.array(state.X_orig_missing_genes.dot(state.betas / state.scale_factors) - np.sum(state.mean_shifts * state.betas / state.scale_factors))
    else:
        state.priors_missing = np.array([])

    #store in member variable
    total_mean = np.mean(np.concatenate((priors, state.priors_missing)))
    state.priors = priors - total_mean
    state.priors_missing -= total_mean

    state.calculate_priors_adj(overwrite_priors=adjust_priors)



def run_cross_val(
    state,
    cross_val_num_explore_each_direction,
    folds=4,
    cross_val_max_num_tries=2,
    p=None,
    max_num_burn_in=1000,
    max_num_iter=1100,
    min_num_iter=10,
    num_chains=4,
    run_logistic=True,
    max_for_linear=0.95,
    run_corrected_ols=False,
    r_threshold_burn_in=1.01,
    use_max_r_for_convergence=True,
    max_frac_sem=0.01,
    gauss_seidel=False,
    sparse_solution=False,
    sparse_frac_betas=None,
    *,
    bail_fn,
    log_fn,
    debug_level,
    trace_level,
    **kwargs,
):
    bail = bail_fn
    log = log_fn
    DEBUG = debug_level
    TRACE = trace_level

    log("Running cross validation", DEBUG)

    if state.sigma2s is not None:
        candidate_sigma2s = state.sigma2s
    elif state.sigma2 is not None:
        candidate_sigma2s = np.array(state.sigma2).reshape((1,))
    else:
        bail("Need to have sigma set before running cross validation")

    if p is None:
        bail("Need to have p set before running cross validation")
    if state.X_orig is None:
        bail("Need to have X_orig set before running cross validation")

    Y_to_use = state.Y_for_regression
    if Y_to_use is None:
        Y_to_use = state.Y

    if Y_to_use is None:
        bail("Need to have Y set before running cross validation")

    D = np.exp(Y_to_use + state.background_log_bf) / (1 + np.exp(Y_to_use + state.background_log_bf))
    if not run_logistic and np.max(D) > max_for_linear:
        log("Switching to logistic sampling due to high Y values (max(D) = %.3g" % np.max(D), DEBUG)
        run_logistic = True

    beta_tildes_cv = np.zeros((folds, len(state.gene_sets)))
    alpha_tildes_cv = np.zeros((folds, len(state.gene_sets)))
    ses_cv = np.zeros((folds, len(state.gene_sets)))
    cv_val_masks = np.full((folds, len(Y_to_use)), False)
    for fold in range(folds):
        cv_mask = np.arange(len(Y_to_use)) % folds != fold
        cv_val_masks[fold,:] = ~cv_mask
        X_to_use = state.X_orig[cv_mask,:]
        if run_logistic:
            Y_cv = D[cv_mask]
            (beta_tildes_cv[fold,:], ses_cv[fold,:], _, _, _, alpha_tildes_cv[fold,:], _) = state._compute_logistic_beta_tildes(X_to_use, Y_cv, resid_correlation_matrix=state.y_corr_sparse[cv_mask,:][:,cv_mask])
        else:
            Y_cv = Y_to_use[cv_mask]
            (beta_tildes_cv[fold,:], ses_cv[fold,:], _, _, _) = state._compute_beta_tildes(X_to_use, Y_cv, resid_correlation_matrix=state.y_corr_sparse[cv_mask,:][:,cv_mask])

    cross_val_num_explore = cross_val_num_explore_each_direction * 2 + 1
    cross_val_num_explore_with_fold = cross_val_num_explore * folds

    candidate_sigma2s_m = np.tile(candidate_sigma2s, cross_val_num_explore).reshape(cross_val_num_explore, candidate_sigma2s.shape[0])
    candidate_sigma2s_m = (candidate_sigma2s_m.T * np.power(10.0, np.arange(-cross_val_num_explore_each_direction,cross_val_num_explore_each_direction+1))).T
    orig_index = cross_val_num_explore_each_direction

    for try_num in range(cross_val_max_num_tries):
        log("Sigmas to try: %s" % np.mean(candidate_sigma2s_m, axis=1), TRACE)
        candidate_sigma2s_m = np.tile(candidate_sigma2s_m, (folds, 1))

        beta_tildes_m = np.repeat(beta_tildes_cv, cross_val_num_explore, axis=0)
        ses_m = np.repeat(ses_cv, cross_val_num_explore, axis=0)
        scale_factors_m = np.tile(state.scale_factors, cross_val_num_explore_with_fold).reshape(cross_val_num_explore_with_fold, len(state.scale_factors))
        mean_shifts_m = np.tile(state.mean_shifts, cross_val_num_explore_with_fold).reshape(cross_val_num_explore_with_fold, len(state.mean_shifts))

        (betas_m, _postp_m) = state._calculate_non_inf_betas(initial_p=state.p, beta_tildes=beta_tildes_m, ses=ses_m, scale_factors=scale_factors_m, mean_shifts=mean_shifts_m, sigma2s=candidate_sigma2s_m, max_num_burn_in=max_num_burn_in, max_num_iter=max_num_iter, min_num_iter=min_num_iter, num_chains=num_chains, r_threshold_burn_in=r_threshold_burn_in, use_max_r_for_convergence=use_max_r_for_convergence, max_frac_sem=max_frac_sem, gauss_seidel=gauss_seidel, update_hyper_sigma=False, update_hyper_p=False, sparse_solution=sparse_solution, sparse_frac_betas=sparse_frac_betas, V=state._get_V(), **kwargs)

        rss = np.zeros(cross_val_num_explore)
        num_Y = 0
        Y_val = Y_to_use - np.mean(Y_to_use)

        for fold in range(folds):
            output_cv_mask = np.floor(np.arange(betas_m.shape[0]) / cross_val_num_explore) == fold
            cur_pred = state.X_orig[cv_val_masks[fold,:],:].dot((betas_m[output_cv_mask,:] / state.scale_factors).T).T
            rss += np.sum(np.square(cur_pred - Y_val[cv_val_masks[fold,:]]), axis=1)
            num_Y += np.sum(cv_val_masks[fold,:])

        rss /= num_Y
        best_result = np.argmin(rss)
        best_sigma2s = candidate_sigma2s_m[best_result,:]
        log("Got RSS values: %s" % (rss), TRACE)
        log("Best sigma is %.3g" % np.mean(best_sigma2s))
        log("Updating sigma from %.3g to %.3g" % (state.sigma2, np.mean(best_sigma2s)))
        if state.sigma2s is not None:
            state.sigma2s = best_sigma2s
            state.set_sigma(np.mean(best_sigma2s), state.sigma_power)
        else:
            assert(len(best_sigma2s.shape) == 1 and best_sigma2s.shape[0] == 1)
            state.set_sigma(best_sigma2s[0], state.sigma_power)

        if try_num + 1 < cross_val_max_num_tries and (best_result == 0 or best_result == (len(rss) - 1)) and best_result != orig_index:
            log("Expanding search further since best cross validation result was at boundary of search space", DEBUG)
            assert(state.sigma2s is not None or state.sigma2 is not None)
            if state.sigma2s is not None:
                candidate_sigma2s = state.sigma2s
            else:
                candidate_sigma2s = np.array(state.sigma2).reshape((1,))
            candidate_sigma2s_m = np.tile(candidate_sigma2s, cross_val_num_explore).reshape(cross_val_num_explore, candidate_sigma2s.shape[0])
            if best_result == 0:
                candidate_sigma2s_m = (candidate_sigma2s_m.T * np.power(10.0, np.arange(-cross_val_num_explore+1,1))).T
                orig_index = cross_val_num_explore - 1
            else:
                candidate_sigma2s_m = (candidate_sigma2s_m.T * np.power(10.0, np.arange(cross_val_num_explore))).T
                orig_index = 0
        else:
            break


def calculate_priors_adj(state, overwrite_priors=False, *, log_fn):
    if state.priors is None:
        return

    gene_N = state.get_gene_N()
    gene_N_missing = state.get_gene_N(get_missing=True)
    all_gene_N = gene_N
    if state.genes_missing is not None:
        assert(gene_N_missing is not None)
        all_gene_N = np.concatenate((all_gene_N, gene_N_missing))

    if state.genes_missing is not None:
        total_priors = np.concatenate((state.priors, state.priors_missing))
    else:
        total_priors = state.priors

    priors_slope = np.cov(total_priors, all_gene_N)[0,1] / np.var(all_gene_N)
    priors_intercept = np.mean(total_priors - all_gene_N * priors_slope)

    log_fn("Adjusting priors with slope %.4g" % priors_slope)
    priors_adj = state.priors - priors_slope * gene_N - priors_intercept
    if overwrite_priors:
        state.priors = priors_adj
    else:
        state.priors_adj = priors_adj
    if state.genes_missing is not None:
        priors_adj_missing = state.priors_missing - priors_slope * gene_N_missing
        if overwrite_priors:
            state.priors_missing = priors_adj_missing
        else:
            state.priors_adj_missing = priors_adj_missing
