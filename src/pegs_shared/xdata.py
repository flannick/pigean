from pegs_shared.types import XData, XReadIngestionOptions, XReadPostOptions


def xdata_from_input_plan(input_plan):
    return XData.from_input_plan(input_plan)


def build_read_x_ingestion_options(local_vars):
    return XReadIngestionOptions(
        batch_all_for_hyper=local_vars["batch_all_for_hyper"],
        first_for_hyper=local_vars["first_for_hyper"],
        update_hyper_sigma=local_vars["update_hyper_sigma"],
        update_hyper_p=local_vars["update_hyper_p"],
        first_for_sigma_cond=local_vars["first_for_sigma_cond"],
        run_corrected_ols=local_vars["run_corrected_ols"],
        gene_cor_file=local_vars["gene_cor_file"],
        gene_loc_file=local_vars["gene_loc_file"],
        gene_cor_file_gene_col=local_vars["gene_cor_file_gene_col"],
        gene_cor_file_cor_start_col=local_vars["gene_cor_file_cor_start_col"],
        run_logistic=local_vars["run_logistic"],
        max_for_linear=local_vars["max_for_linear"],
        only_ids=local_vars["only_ids"],
        add_all_genes=local_vars["add_all_genes"],
        only_inc_genes=local_vars["only_inc_genes"],
        fraction_inc_genes=local_vars["fraction_inc_genes"],
        ignore_genes=local_vars["ignore_genes"],
        max_num_entries_at_once=local_vars["max_num_entries_at_once"],
        filter_gene_set_p=local_vars["filter_gene_set_p"],
        filter_gene_set_metric_z=local_vars["filter_gene_set_metric_z"],
        filter_using_phewas=local_vars["filter_using_phewas"],
    )


def build_read_x_post_options(local_vars, *, batches, num_ignored_gene_sets, ignored_for_fraction_inc):
    return XReadPostOptions(
        ignored_for_fraction_inc=ignored_for_fraction_inc,
        filter_gene_set_p=local_vars["filter_gene_set_p"],
        correct_betas_mean=local_vars["correct_betas_mean"],
        correct_betas_var=local_vars["correct_betas_var"],
        filter_using_phewas=local_vars["filter_using_phewas"],
        prune_gene_sets=local_vars["prune_gene_sets"],
        weighted_prune_gene_sets=local_vars["weighted_prune_gene_sets"],
        prune_deterministically=local_vars["prune_deterministically"],
        max_num_gene_sets_initial=local_vars["max_num_gene_sets_initial"],
        skip_betas=local_vars["skip_betas"],
        initial_p=local_vars["initial_p"],
        update_hyper_p=local_vars["update_hyper_p"],
        sigma_power=local_vars["sigma_power"],
        initial_sigma2_cond=local_vars["initial_sigma2_cond"],
        update_hyper_sigma=local_vars["update_hyper_sigma"],
        initial_sigma2=local_vars["initial_sigma2"],
        sigma_soft_threshold_95=local_vars["sigma_soft_threshold_95"],
        sigma_soft_threshold_5=local_vars["sigma_soft_threshold_5"],
        batches=batches,
        num_ignored_gene_sets=num_ignored_gene_sets,
        first_for_hyper=local_vars["first_for_hyper"],
        max_num_gene_sets_hyper=local_vars["max_num_gene_sets_hyper"],
        first_for_sigma_cond=local_vars["first_for_sigma_cond"],
        first_max_p_for_hyper=local_vars["first_max_p_for_hyper"],
        max_num_burn_in=local_vars["max_num_burn_in"],
        max_num_iter_betas=local_vars["max_num_iter_betas"],
        min_num_iter_betas=local_vars["min_num_iter_betas"],
        num_chains_betas=local_vars["num_chains_betas"],
        r_threshold_burn_in_betas=local_vars["r_threshold_burn_in_betas"],
        use_max_r_for_convergence_betas=local_vars["use_max_r_for_convergence_betas"],
        max_frac_sem_betas=local_vars["max_frac_sem_betas"],
        max_allowed_batch_correlation=local_vars["max_allowed_batch_correlation"],
        sigma_num_devs_to_top=local_vars["sigma_num_devs_to_top"],
        p_noninf_inflate=local_vars["p_noninf_inflate"],
        sparse_solution=local_vars["sparse_solution"],
        sparse_frac_betas=local_vars["sparse_frac_betas"],
        betas_trace_out=local_vars["betas_trace_out"],
        increase_filter_gene_set_p=local_vars["increase_filter_gene_set_p"],
        min_gene_set_size=local_vars["min_gene_set_size"],
        max_gene_set_size=local_vars["max_gene_set_size"],
        filter_gene_set_metric_z=local_vars["filter_gene_set_metric_z"],
        max_num_gene_sets=local_vars["max_num_gene_sets"],
    )


def initialize_matrix_and_gene_index_state(runtime, batch_size):
    XData.initialized_runtime_state(batch_size).apply_to_runtime(runtime)
