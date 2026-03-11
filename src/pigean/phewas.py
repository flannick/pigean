from __future__ import annotations

import numpy as np
import pegs_shared.phewas as pegs_phewas

from . import main_support as pigean_main_support


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
    if not options.run_phewas_from_gene_phewas_stats_in:
        return
    decision = pigean_main_support.resolve_gene_phewas_input_decision_for_stage(
        requested_input=options.run_phewas_from_gene_phewas_stats_in,
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
        gene_phewas_bfs_prior_col=options.gene_phewas_bfs_prior_col,
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


def run_phewas(
    state,
    gene_phewas_bfs_in=None,
    gene_phewas_bfs_id_col=None,
    gene_phewas_bfs_pheno_col=None,
    gene_phewas_bfs_log_bf_col=None,
    gene_phewas_bfs_combined_col=None,
    gene_phewas_bfs_prior_col=None,
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
    col_info = None

    if read_file:
        phenos, pheno_to_ind, col_info = prepare_phewas_phenos_from_file(
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
    else:
        phenos = state.phenos

    num_batches = int(np.ceil(len(phenos) / batch_size))
    input_values = state._build_phewas_input_values()
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

        if read_file:
            gene_pheno_Y, gene_pheno_combined_prior_Ys, gene_pheno_priors = read_phewas_file_batch(
                state,
                gene_phewas_bfs_in,
                begin=begin,
                cur_batch_size=cur_batch_size,
                pheno_to_ind=pheno_to_ind,
                id_col=col_info['id_col'],
                pheno_col=col_info['pheno_col'],
                bf_col=col_info['bf_col'],
                combined_col=col_info['combined_col'],
                prior_col=col_info['prior_col'],
                open_text_fn=open_text_fn,
                warn_fn=warn_fn,
            )
        else:
            gene_pheno_Y = state.gene_pheno_Y[:,begin:end].toarray() if state.gene_pheno_Y is not None else None
            gene_pheno_combined_prior_Ys = state.gene_pheno_combined_prior_Ys[:,begin:end].toarray() if state.gene_pheno_combined_prior_Ys is not None else None

        if gene_pheno_Y is not None:
            beta, _, beta_tilde, se, Z, p_value, _ = state._calculate_phewas_block(input_values, gene_pheno_Y.T, **phewas_beta_kwargs)
            assert beta.shape[0] == 3, "First dimension of beta should be 3, not (%s, %s)" % (beta.shape[0], beta.shape[1])
            accumulate_phewas_outputs(state, 'pheno_Y', beta, beta_tilde, se, Z, p_value)

        if gene_pheno_combined_prior_Ys is not None and not state.debug_skip_correlation:
            beta, _, beta_tilde, se, Z, p_value, _ = state._calculate_phewas_block(
                input_values,
                gene_pheno_combined_prior_Ys.T,
                X_orig=state.X_orig,
                X_phewas_beta=state.X_phewas_beta[begin:end,:] if state.X_phewas_beta is not None else None,
                Y_resid=gene_pheno_Y.T,
                **phewas_beta_kwargs
            )
            assert beta.shape[0] == 3, "First dimension of beta should be 3, not (%s, %s)" % (beta.shape[0], beta.shape[1])
            accumulate_phewas_outputs(state, 'pheno_combined_prior_Ys', beta, beta_tilde, se, Z, p_value)
