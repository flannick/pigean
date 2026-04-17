from __future__ import annotations

import numpy as np
import scipy.sparse as sparse

from pegs_shared.types import XReadCallbacks as PegsXReadCallbacks
from pegs_shared.types import XReadConfig as PegsXReadConfig
from pegs_shared.types import XReadPostCallbacks as PegsXReadPostCallbacks
from pegs_shared.xdata import (
    build_read_x_ingestion_options as pegs_build_read_x_ingestion_options,
    build_read_x_pipeline_config as pegs_build_read_x_pipeline_config,
    build_read_x_post_options as pegs_build_read_x_post_options,
    prepare_read_x_inputs as pegs_prepare_read_x_inputs,
    xdata_from_input_plan as pegs_xdata_from_input_plan,
)


def run_main_adaptive_read_x(
    state,
    options,
    mode_state,
    sigma2_cond,
    *,
    build_xin_to_p_noninf_index_map_fn,
    run_read_x_stage_fn,
    warn_fn,
    bail_fn,
    log_fn,
):
    pure_betas_run = (
        mode_state["run_beta"]
        and not mode_state["run_priors"]
        and not mode_state["run_naive_priors"]
        and not mode_state["run_gibbs"]
    )
    retain_all_beta_uncorrected = pure_betas_run and (
        options.retain_all_beta_uncorrected or options.independent_betas_only
    )
    track_filtered_beta_uncorrected = options.track_filtered_beta_uncorrected
    state.track_filtered_beta_uncorrected = track_filtered_beta_uncorrected
    xin_to_p_noninf_ind = build_xin_to_p_noninf_index_map_fn(
        options.X_in,
        options.X_list,
        options.Xd_in,
        options.Xd_list,
        options.p_noninf,
        warn_fn=warn_fn,
        bail_fn=bail_fn,
    )

    skip_betas = (
        (mode_state["run_huge"] or mode_state["run_beta_tilde"])
        and not (mode_state["run_beta"] or mode_state["run_priors"] or mode_state["run_naive_priors"] or mode_state["run_gibbs"])
    )
    read_x_retry_state = {
        "filter_gene_set_p": options.filter_gene_set_p,
        "force_reread": False,
    }
    while True:
        sigma2_internal_before_read = state.sigma2
        read_x_kwargs = dict(
            Xd_in=options.Xd_in,
            X_list=options.X_list,
            Xd_list=options.Xd_list,
            V_in=options.V_in,
            min_gene_set_size=options.min_gene_set_size,
            max_gene_set_size=options.max_gene_set_size,
            only_ids=None,
            only_inc_genes=None,
            fraction_inc_genes=None,
            add_all_genes=options.add_all_genes,
            prune_gene_sets=options.prune_gene_sets,
            weighted_prune_gene_sets=options.weighted_prune_gene_sets,
            prune_deterministically=options.prune_deterministically,
            x_sparsify=options.x_sparsify,
            add_ext=options.add_ext,
            add_top=options.add_top,
            add_bottom=options.add_bottom,
            filter_negative=options.filter_negative,
            threshold_weights=options.threshold_weights,
            cap_weights=options.cap_weights,
            permute_gene_sets=options.permute_gene_sets,
            max_gene_set_p=options.max_gene_set_read_p,
            filter_gene_set_p=read_x_retry_state["filter_gene_set_p"],
            filter_using_phewas=options.betas_uncorrected_from_phewas,
            increase_filter_gene_set_p=options.increase_filter_gene_set_p,
            max_num_gene_sets_initial=options.max_num_gene_sets_initial,
            max_num_gene_sets=options.max_num_gene_sets,
            max_num_gene_sets_hyper=options.max_num_gene_sets_hyper,
            skip_betas=skip_betas,
            run_logistic=not options.linear,
            max_for_linear=options.max_for_linear,
            filter_gene_set_metric_z=options.filter_gene_set_metric_z,
            initial_p=options.p_noninf,
            xin_to_p_noninf_ind=xin_to_p_noninf_ind,
            initial_sigma2=sigma2_internal_before_read,
            initial_sigma2_cond=sigma2_cond,
            sigma_power=options.sigma_power,
            sigma_soft_threshold_95=options.sigma_soft_threshold_95,
            sigma_soft_threshold_5=options.sigma_soft_threshold_5,
            run_corrected_ols=not options.ols,
            correct_betas_mean=options.correct_betas_mean,
            correct_betas_var=options.correct_betas_var,
            gene_loc_file=options.gene_loc_file,
            gene_cor_file=options.gene_cor_file,
            gene_cor_file_gene_col=options.gene_cor_file_gene_col,
            gene_cor_file_cor_start_col=options.gene_cor_file_cor_start_col,
            update_hyper_p=options.update_hyper_p,
            update_hyper_sigma=options.update_hyper_sigma,
            batch_all_for_hyper=options.batch_all_for_hyper,
            first_for_hyper=options.first_for_hyper,
            first_max_p_for_hyper=options.first_max_p_for_hyper,
            first_for_sigma_cond=options.first_for_sigma_cond,
            sigma_num_devs_to_top=options.sigma_num_devs_to_top,
            p_noninf_inflate=options.p_noninf_inflate,
            batch_separator=options.batch_separator,
            x_list_unlabeled_batching=options.x_list_unlabeled_batching,
            ignore_genes=options.ignore_genes,
            file_separator=options.file_separator,
            max_num_burn_in=options.max_num_burn_in,
            max_num_iter_betas=options.max_num_iter_betas,
            min_num_iter_betas=options.min_num_iter_betas,
            num_chains_betas=options.num_chains_betas,
            r_threshold_burn_in_betas=options.r_threshold_burn_in_betas,
            use_max_r_for_convergence_betas=options.use_max_r_for_convergence_betas,
            max_frac_sem_betas=options.max_frac_sem_betas,
            max_allowed_batch_correlation=options.max_allowed_batch_correlation,
            sparse_solution=options.sparse_solution,
            sparse_frac_betas=options.sparse_frac_betas,
            betas_trace_out=options.betas_trace_out,
            show_progress=not options.hide_progress,
            skip_V=(options.max_gene_set_read_p is not None),
            max_num_entries_at_once=options.max_read_entries_at_once,
            force_reread=read_x_retry_state["force_reread"],
            retain_all_beta_uncorrected=retain_all_beta_uncorrected,
            independent_betas_only=pure_betas_run and options.independent_betas_only,
            track_filtered_beta_uncorrected=track_filtered_beta_uncorrected,
        )
        run_read_x_stage_fn(state, options.X_in, **read_x_kwargs)

        should_reread = False
        new_filter_gene_set_p = read_x_retry_state["filter_gene_set_p"]
        if (
            options.min_num_gene_sets is not None
            and read_x_retry_state["filter_gene_set_p"] is not None
            and read_x_retry_state["filter_gene_set_p"] < 1
            and state.gene_sets is not None
            and len(state.gene_sets) < options.min_num_gene_sets
        ):
            fraction_to_increase = float(options.min_num_gene_sets) / (len(state.gene_sets) + 1)
            if fraction_to_increase > 1:
                new_filter_gene_set_p = read_x_retry_state["filter_gene_set_p"] * fraction_to_increase * 1.2
                if new_filter_gene_set_p > 1:
                    new_filter_gene_set_p = 1
                log_fn(
                    "Only read in %d gene sets; scaled --filter-gene-set-p to %.3g and re-reading gene sets"
                    % (len(state.gene_sets), new_filter_gene_set_p)
                )
                state.set_sigma(sigma2_internal_before_read, state.sigma_power)
                should_reread = True
        if not should_reread:
            break
        read_x_retry_state["filter_gene_set_p"] = new_filter_gene_set_p
        read_x_retry_state["force_reread"] = True


def run_read_x_stage(runtime, X_in, *, read_x_kwargs, build_read_x_pipeline_config_fn, bail_fn, read_x_pipeline_fn):
    read_x_pipeline_config = build_read_x_pipeline_config_fn(
        X_in,
        read_x_kwargs,
        bail_fn=bail_fn,
    )
    return read_x_pipeline_fn(runtime, read_x_pipeline_config)


def read_x_pipeline(
    runtime,
    read_x_pipeline_config,
    *,
    open_gz_fn,
    open_dense_fn,
    warn_fn,
    log_fn,
    info_level,
    debug_level,
    remove_tag_from_input_fn,
    record_read_x_counts_fn,
    ensure_gene_universe_fn,
    process_x_input_file_fn,
    normalize_dense_gene_rows_fn,
    build_sparse_x_from_dense_input_fn,
    reindex_x_rows_to_current_genes_fn,
    normalize_gene_set_weights_fn,
    partition_missing_gene_rows_fn,
    maybe_permute_gene_set_rows_fn,
    maybe_prefilter_x_block_fn,
    merge_missing_gene_rows_fn,
    finalize_added_x_block_fn,
    standardize_qc_metrics_after_x_read_fn,
    maybe_correct_gene_set_betas_after_x_read_fn,
    maybe_limit_initial_gene_sets_by_p_fn,
    maybe_prune_gene_sets_after_x_read_fn,
    initialize_hyper_defaults_after_x_read_fn,
    maybe_learn_batch_hyper_after_x_read_fn,
    maybe_adjust_overaggressive_p_filter_after_x_read_fn,
    apply_post_read_gene_set_size_and_qc_filters_fn,
    maybe_filter_zero_uncorrected_betas_after_x_read_fn,
    maybe_reduce_gene_sets_to_max_after_x_read_fn,
):
    if not read_x_pipeline_config.force_reread and runtime.X_orig is not None:
        return

    filter_using_phewas = read_x_pipeline_config.filter_using_phewas
    if filter_using_phewas and runtime.gene_pheno_Y is None:
        filter_using_phewas = False

    runtime._set_X(None, runtime.genes, None, skip_N=True)
    runtime._record_params({
        "filter_gene_set_p": read_x_pipeline_config.filter_gene_set_p,
        "filter_negative": read_x_pipeline_config.filter_negative,
        "threshold_weights": read_x_pipeline_config.threshold_weights,
        "cap_weights": read_x_pipeline_config.cap_weights,
        "max_num_gene_sets_initial": read_x_pipeline_config.max_num_gene_sets_initial,
        "max_num_gene_sets": read_x_pipeline_config.max_num_gene_sets,
        "max_num_gene_sets_hyper": read_x_pipeline_config.max_num_gene_sets_hyper,
        "filter_gene_set_metric_z": read_x_pipeline_config.filter_gene_set_metric_z,
        "num_chains_betas": read_x_pipeline_config.num_chains_betas,
        "sigma_num_devs_to_top": read_x_pipeline_config.sigma_num_devs_to_top,
        "p_noninf_inflate": read_x_pipeline_config.p_noninf_inflate,
    })

    x_input_plan = pegs_prepare_read_x_inputs(
        X_in=read_x_pipeline_config.X_in,
        X_list=read_x_pipeline_config.X_list,
        Xd_in=read_x_pipeline_config.Xd_in,
        Xd_list=read_x_pipeline_config.Xd_list,
        initial_p=read_x_pipeline_config.initial_p,
        xin_to_p_noninf_ind=read_x_pipeline_config.xin_to_p_noninf_ind,
        batch_separator=read_x_pipeline_config.batch_separator,
        file_separator=read_x_pipeline_config.file_separator,
        sparse_list_open_fn=open_gz_fn,
        dense_list_open_fn=open_dense_fn,
        x_list_unlabeled_batching=read_x_pipeline_config.x_list_unlabeled_batching,
        warn_fn=warn_fn,
    )
    xdata_seed = pegs_xdata_from_input_plan(x_input_plan)

    read_x_config = PegsXReadConfig(
        x_sparsify=read_x_pipeline_config.x_sparsify,
        min_gene_set_size=read_x_pipeline_config.min_gene_set_size,
        add_ext=read_x_pipeline_config.add_ext,
        add_top=read_x_pipeline_config.add_top,
        add_bottom=read_x_pipeline_config.add_bottom,
        threshold_weights=read_x_pipeline_config.threshold_weights,
        cap_weights=read_x_pipeline_config.cap_weights,
        permute_gene_sets=read_x_pipeline_config.permute_gene_sets,
        filter_gene_set_p=read_x_pipeline_config.filter_gene_set_p,
        filter_gene_set_metric_z=read_x_pipeline_config.filter_gene_set_metric_z,
        filter_using_phewas=filter_using_phewas,
        increase_filter_gene_set_p=read_x_pipeline_config.increase_filter_gene_set_p,
        filter_negative=read_x_pipeline_config.filter_negative,
    )
    read_x_callbacks = PegsXReadCallbacks(
        sparse_module=sparse,
        np_module=np,
        normalize_dense_gene_rows_fn=normalize_dense_gene_rows_fn,
        build_sparse_x_from_dense_input_fn=build_sparse_x_from_dense_input_fn,
        reindex_x_rows_to_current_genes_fn=reindex_x_rows_to_current_genes_fn,
        normalize_gene_set_weights_fn=normalize_gene_set_weights_fn,
        partition_missing_gene_rows_fn=partition_missing_gene_rows_fn,
        maybe_permute_gene_set_rows_fn=maybe_permute_gene_set_rows_fn,
        maybe_prefilter_x_block_fn=maybe_prefilter_x_block_fn,
        merge_missing_gene_rows_fn=merge_missing_gene_rows_fn,
        finalize_added_x_block_fn=finalize_added_x_block_fn,
    )

    read_x_locals = dict(vars(read_x_pipeline_config))
    read_x_locals["filter_using_phewas"] = filter_using_phewas
    ingestion_options = pegs_build_read_x_ingestion_options(read_x_locals)
    ingestion_state = xdata_seed.run_ingestion_stage(
        runtime,
        input_plan=x_input_plan,
        read_config=read_x_config,
        read_callbacks=read_x_callbacks,
        ingestion_options=ingestion_options,
        ensure_gene_universe_fn=ensure_gene_universe_fn,
        process_x_input_file_fn=process_x_input_file_fn,
        remove_tag_from_input_fn=remove_tag_from_input_fn,
        log_fn=log_fn,
        info_level=info_level,
        debug_level=debug_level,
    )

    post_options = pegs_build_read_x_post_options(
        read_x_locals,
        batches=ingestion_state["batches"],
        num_ignored_gene_sets=ingestion_state["num_ignored_gene_sets"],
        ignored_for_fraction_inc=ingestion_state["ignored_for_fraction_inc"],
    )
    post_callbacks = PegsXReadPostCallbacks(
        standardize_qc_metrics_after_x_read_fn=standardize_qc_metrics_after_x_read_fn,
        maybe_correct_gene_set_betas_after_x_read_fn=maybe_correct_gene_set_betas_after_x_read_fn,
        maybe_limit_initial_gene_sets_by_p_fn=maybe_limit_initial_gene_sets_by_p_fn,
        maybe_prune_gene_sets_after_x_read_fn=maybe_prune_gene_sets_after_x_read_fn,
        initialize_hyper_defaults_after_x_read_fn=initialize_hyper_defaults_after_x_read_fn,
        maybe_learn_batch_hyper_after_x_read_fn=maybe_learn_batch_hyper_after_x_read_fn,
        maybe_adjust_overaggressive_p_filter_after_x_read_fn=maybe_adjust_overaggressive_p_filter_after_x_read_fn,
        apply_post_read_gene_set_size_and_qc_filters_fn=apply_post_read_gene_set_size_and_qc_filters_fn,
        maybe_filter_zero_uncorrected_betas_after_x_read_fn=maybe_filter_zero_uncorrected_betas_after_x_read_fn,
        maybe_reduce_gene_sets_to_max_after_x_read_fn=maybe_reduce_gene_sets_to_max_after_x_read_fn,
        record_read_x_counts_fn=record_read_x_counts_fn,
    )
    xdata_seed.run_post_stage(
        runtime,
        post_options=post_options,
        post_callbacks=post_callbacks,
        log_fn=log_fn,
        debug_level=debug_level,
    )
