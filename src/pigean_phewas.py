from __future__ import annotations


def run_advanced_set_b_phewas_beta_sampling_if_requested(domain, state, options, beta_sampling_kwargs):
    if not options.betas_uncorrected_from_phewas:
        return
    phewas_beta_sampling_kwargs = dict(beta_sampling_kwargs)
    phewas_beta_sampling_kwargs.update({
        "run_betas_using_phewas": options.betas_from_phewas,
        "run_uncorrected_using_phewas": True,
    })
    state.calculate_non_inf_betas(state.p, **phewas_beta_sampling_kwargs)


def run_advanced_set_b_output_phewas_if_requested(domain, state, options):
    if not options.run_phewas_from_gene_phewas_stats_in:
        return
    decision = domain.pegs_resolve_gene_phewas_input_decision_for_stage(
        requested_input=options.run_phewas_from_gene_phewas_stats_in,
        reusable_inputs=[options.gene_phewas_bfs_in],
        read_gene_phewas=state.read_gene_phewas(),
        num_gene_phewas_filtered=state.num_gene_phewas_filtered,
    )
    domain.log(
        "PheWAS stage 'output_phewas': mode=%s reason=%s" % (decision.mode, decision.reason),
        domain.INFO,
    )
    bfs_to_use = decision.resolved_input

    phewas_config = domain.pegs_build_phewas_stage_config(
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
