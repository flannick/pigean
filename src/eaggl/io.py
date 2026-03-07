from __future__ import annotations

import copy

import numpy as np
import scipy
import scipy.sparse as sparse

from pegs_shared.io_common import (
    clean_chrom_name,
    construct_map_to_ind,
    parse_gene_map_file,
    read_loc_file_with_gene_map,
)
from pegs_shared.types import XReadCallbacks, XReadConfig, XReadPostCallbacks
from pegs_shared.xdata import (
    build_read_x_pipeline_config,
    build_read_x_ingestion_options,
    build_read_x_post_options,
    xdata_from_input_plan,
)
from pegs_shared.ydata import sync_phewas_runtime_state
from pegs_utils import (
    load_and_apply_gene_phewas_bfs_to_runtime,
    load_and_apply_gene_set_phewas_statistics_to_runtime,
    load_and_apply_gene_set_statistics_to_runtime,
    record_read_x_counts,
)


def run_read_y_stage(domain, runtime, **read_kwargs):
    return read_y_pipeline(domain, runtime, **read_kwargs)


def read_y_pipeline(
    domain,
    runtime,
    gwas_in=None,
    huge_statistics_in=None,
    huge_statistics_out=None,
    exomes_in=None,
    positive_controls_in=None,
    positive_controls_list=None,
    case_counts_in=None,
    ctrl_counts_in=None,
    gene_bfs_in=None,
    gene_loc_file=None,
    gene_covs_in=None,
    hold_out_chrom=None,
    **kwargs
):
    unsupported_flags = []
    if gwas_in is not None:
        unsupported_flags.append("--gwas-in")
    if huge_statistics_in is not None:
        unsupported_flags.append("--huge-statistics-in")
    if huge_statistics_out is not None:
        unsupported_flags.append("--huge-statistics-out")
    if exomes_in is not None:
        unsupported_flags.append("--exomes-in")
    if case_counts_in is not None:
        unsupported_flags.append("--case-counts-in")
    if ctrl_counts_in is not None:
        unsupported_flags.append("--ctrl-counts-in")

    if len(unsupported_flags) > 0:
        domain.bail(
            "These inputs belong to pigean.py and are not supported in eaggl.py: %s. "
            "Run pigean.py first and pass outputs via --eaggl-bundle-in or --gene-stats-in/--gene-set-stats-in."
            % ", ".join(sorted(unsupported_flags))
        )

    if positive_controls_in is not None or positive_controls_list is not None:
        domain.warn("Ignoring positive-control inputs in eaggl.py read_Y; using --gene-stats-in values")

    if gene_bfs_in is None:
        domain.bail("Require --gene-stats-in for this operation")

    (Y1, extra_genes, extra_Y, gene_combined_map, gene_prior_map) = runtime._read_gene_bfs(gene_bfs_in, **kwargs)

    def _apply_hold_out_chrom(Y_values, extra_gene_names, extra_Y_values):
        if hold_out_chrom is None:
            return (Y_values, extra_gene_names, extra_Y_values)

        if runtime.gene_to_chrom is None:
            if gene_loc_file is None:
                domain.bail("Option --hold-out-chrom requires --gene-loc-file")
            (
                runtime.gene_chrom_name_pos,
                runtime.gene_to_chrom,
                runtime.gene_to_pos,
            ) = read_loc_file_with_gene_map(
                gene_loc_file,
                gene_label_map=runtime.gene_label_map,
                clean_chrom_fn=clean_chrom_name,
                warn_fn=domain.warn,
                bail_fn=domain.bail,
            )

        Y_values = np.array(Y_values, dtype=float)
        extra_gene_names = list(extra_gene_names)
        extra_Y_values = np.array(extra_Y_values, dtype=float)

        if runtime.genes is not None:
            Y_nan_mask = np.full(len(Y_values), False)
            for i, gene in enumerate(runtime.genes):
                if gene in runtime.gene_to_chrom and runtime.gene_to_chrom[gene] == hold_out_chrom:
                    Y_nan_mask[i] = True
            if np.sum(Y_nan_mask) > 0:
                Y_values[Y_nan_mask] = np.nan

        if len(extra_gene_names) > 0:
            keep_mask = np.full(len(extra_gene_names), True)
            for i, gene in enumerate(extra_gene_names):
                if gene in runtime.gene_to_chrom and runtime.gene_to_chrom[gene] == hold_out_chrom:
                    keep_mask[i] = False
            if np.sum(~keep_mask) > 0:
                extra_gene_names = [extra_gene_names[i] for i in range(len(extra_gene_names)) if keep_mask[i]]
                extra_Y_values = extra_Y_values[keep_mask]

        return (Y_values, extra_gene_names, extra_Y_values)

    (Y1, extra_genes, extra_Y) = _apply_hold_out_chrom(Y1, extra_genes, extra_Y)
    Y1_for_regression = copy.copy(Y1)
    extra_Y_for_regression = copy.copy(extra_Y)

    if runtime.genes is None:
        genes_union = []
        seen = set()
        for gene in extra_genes:
            if gene not in seen:
                genes_union.append(gene)
                seen.add(gene)

        runtime._set_X(runtime.X_orig, genes_union, runtime.gene_sets, skip_N=False)
        Y = np.array(extra_Y, dtype=float)
        Y_for_regression = np.array(extra_Y_for_regression, dtype=float)
        extra_genes = []
        extra_Y = np.array([])
        extra_Y_for_regression = np.array([])
    else:
        missing_value = np.nanmean(Y1) if len(Y1) > 0 else 0.0
        Y = np.array(Y1, dtype=float)
        Y[np.isnan(Y)] = missing_value
        Y_for_regression = np.array(Y1_for_regression, dtype=float)
        Y_for_regression[np.isnan(Y_for_regression)] = missing_value

    if len(extra_Y) > 0:
        Y = np.concatenate((Y, extra_Y))
        Y_for_regression = np.concatenate((Y_for_regression, extra_Y_for_regression))

        if runtime.X_orig is not None:
            runtime._set_X(
                sparse.csc_matrix(
                    (runtime.X_orig.data, runtime.X_orig.indices, runtime.X_orig.indptr),
                    shape=(runtime.X_orig.shape[0] + len(extra_Y), runtime.X_orig.shape[1]),
                ),
                runtime.genes,
                runtime.gene_sets,
                skip_V=True,
                skip_scale_factors=True,
                skip_N=False,
            )

        if runtime.genes is not None:
            runtime._set_X(runtime.X_orig, runtime.genes + extra_genes, runtime.gene_sets, skip_N=False)

    runtime._set_Y(Y, Y_for_regression, skip_V=True, skip_scale_factors=True)

    if gene_combined_map is not None:
        runtime.combined_prior_Ys = copy.copy(runtime.Y)
        for i, gene in enumerate(runtime.genes):
            if gene in gene_combined_map:
                runtime.combined_prior_Ys[i] = gene_combined_map[gene]

    if gene_prior_map is not None:
        runtime.priors = np.zeros(len(runtime.genes))
        for i, gene in enumerate(runtime.genes):
            if gene in gene_prior_map:
                runtime.priors[i] = gene_prior_map[gene]

    if gene_covs_in is not None:
        (cov_names, gene_covs, _, _) = runtime._read_gene_covs(gene_covs_in, **kwargs)
        cov_dirs = np.array([0] * len(cov_names))

        col_means = np.nanmean(gene_covs, axis=0)
        nan_indices = np.where(np.isnan(gene_covs))
        gene_covs[nan_indices] = np.take(col_means, nan_indices[1])

        if runtime.gene_covariates is not None:
            assert gene_covs.shape[0] == runtime.gene_covariates.shape[0]
            runtime.gene_covariates = np.hstack((runtime.gene_covariates, gene_covs))
            runtime.gene_covariate_names = runtime.gene_covariate_names + cov_names
            runtime.gene_covariate_directions = np.append(runtime.gene_covariate_directions, cov_dirs)
        else:
            runtime.gene_covariates = gene_covs
            runtime.gene_covariate_names = cov_names
            runtime.gene_covariate_directions = cov_dirs

    if runtime.gene_covariates is not None:
        constant_features = np.isclose(np.var(runtime.gene_covariates, axis=0), 0)
        if np.sum(constant_features) > 0:
            runtime.gene_covariates = runtime.gene_covariates[:, ~constant_features]
            runtime.gene_covariate_names = [runtime.gene_covariate_names[i] for i in np.where(~constant_features)[0]]
            runtime.gene_covariate_directions = np.array(
                [runtime.gene_covariate_directions[i] for i in np.where(~constant_features)[0]]
            )

        prune_threshold = 0.95
        cor_mat = np.abs(np.corrcoef(runtime.gene_covariates.T))
        np.fill_diagonal(cor_mat, 0)

        while True:
            if np.max(cor_mat) < prune_threshold:
                try:
                    np.linalg.inv(runtime.gene_covariates.T.dot(runtime.gene_covariates))
                    break
                except np.linalg.LinAlgError:
                    pass

            max_index = np.unravel_index(np.argmax(cor_mat), cor_mat.shape)
            if np.max(max_index) == runtime.gene_covariate_intercept_index:
                max_index = np.min(max_index)
            else:
                max_index = np.max(max_index)

            domain.log("Removing feature %s" % runtime.gene_covariate_names[max_index], domain.TRACE)
            runtime.gene_covariates = np.delete(runtime.gene_covariates, max_index, axis=1)
            del runtime.gene_covariate_names[max_index]
            runtime.gene_covariate_directions = np.delete(runtime.gene_covariate_directions, max_index)
            cor_mat = np.delete(np.delete(cor_mat, max_index, axis=1), max_index, axis=0)
            if len(runtime.gene_covariates) == 0:
                domain.bail("Error: something went wrong with matrix inversion. Still couldn't invert after removing all but one column")

        runtime.gene_covariate_intercept_index = np.where(np.isclose(np.var(runtime.gene_covariates, axis=0), 0))[0]
        if len(runtime.gene_covariate_intercept_index) == 0:
            runtime.gene_covariates = np.hstack((runtime.gene_covariates, np.ones(runtime.gene_covariates.shape[0])[:, np.newaxis]))
            runtime.gene_covariate_names.append("intercept")
            runtime.gene_covariate_directions = np.append(runtime.gene_covariate_directions, 0)
            runtime.gene_covariate_intercept_index = len(runtime.gene_covariate_names) - 1
        else:
            runtime.gene_covariate_intercept_index = runtime.gene_covariate_intercept_index[0]

        covariate_means = np.mean(runtime.gene_covariates, axis=0)
        covariate_sds = np.std(runtime.gene_covariates, axis=0)
        covariate_sds[covariate_sds == 0] = 1

        runtime.gene_covariates_mask = np.all(runtime.gene_covariates < covariate_means + 5 * covariate_sds, axis=1)
        runtime.gene_covariates_mat_inv = np.linalg.inv(
            runtime.gene_covariates[runtime.gene_covariates_mask, :].T.dot(runtime.gene_covariates[runtime.gene_covariates_mask, :])
        )
        gene_covariate_sds = np.std(runtime.gene_covariates, axis=0)
        gene_covariate_sds[gene_covariate_sds == 0] = 1
        runtime.gene_covariate_zs = (runtime.gene_covariates - np.mean(runtime.gene_covariates, axis=0)) / gene_covariate_sds

        Y_for_regression = runtime.Y_for_regression
        if runtime.Y_for_regression is not None:
            (Y_for_regression, _, _) = runtime._correct_huge(
                runtime.Y_for_regression,
                runtime.gene_covariates,
                runtime.gene_covariates_mask,
                runtime.gene_covariates_mat_inv,
                runtime.gene_covariate_names,
                runtime.gene_covariate_intercept_index,
            )

        (Y, runtime.Y_uncorrected, _) = runtime._correct_huge(
            runtime.Y,
            runtime.gene_covariates,
            runtime.gene_covariates_mask,
            runtime.gene_covariates_mat_inv,
            runtime.gene_covariate_names,
            runtime.gene_covariate_intercept_index,
        )

        runtime._set_Y(Y, Y_for_regression, runtime.Y_exomes, runtime.Y_positive_controls, runtime.Y_case_counts)
        runtime.gene_covariate_adjustments = runtime.Y_for_regression - runtime.Y_uncorrected


def run_read_x_stage(domain, runtime, X_in, **read_x_kwargs):
    read_x_pipeline_config = build_read_x_pipeline_config(
        X_in,
        read_x_kwargs,
        bail_fn=domain.bail,
    )
    return read_x_pipeline(domain, runtime, read_x_pipeline_config)


def read_x_pipeline(domain, runtime, read_x_pipeline_config):
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
    })

    x_input_plan = domain.pegs_prepare_read_x_inputs(
        read_x_pipeline_config.X_in,
        Xd_ins=read_x_pipeline_config.Xd_in,
        X_list=read_x_pipeline_config.X_list,
        Xd_list=read_x_pipeline_config.Xd_list,
        initial_p=read_x_pipeline_config.initial_p,
        batch_separator=read_x_pipeline_config.batch_separator,
        file_separator=read_x_pipeline_config.file_separator,
        bail_fn=domain.bail,
    )
    xdata_seed = xdata_from_input_plan(x_input_plan)

    read_x_config = XReadConfig(
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
    read_x_callbacks = XReadCallbacks(
        sparse_module=sparse,
        np_module=np,
        normalize_dense_gene_rows_fn=domain._normalize_dense_gene_rows,
        build_sparse_x_from_dense_input_fn=domain._build_sparse_x_from_dense_input,
        reindex_x_rows_to_current_genes_fn=domain._reindex_x_rows_to_current_genes,
        normalize_gene_set_weights_fn=domain._normalize_gene_set_weights,
        partition_missing_gene_rows_fn=domain._partition_missing_gene_rows,
        maybe_permute_gene_set_rows_fn=domain._maybe_permute_gene_set_rows,
        maybe_prefilter_x_block_fn=domain._maybe_prefilter_x_block,
        merge_missing_gene_rows_fn=domain._merge_missing_gene_rows,
        finalize_added_x_block_fn=domain._finalize_added_x_block,
    )

    read_x_locals = dict(vars(read_x_pipeline_config))
    read_x_locals["filter_using_phewas"] = filter_using_phewas
    ingestion_options = build_read_x_ingestion_options(read_x_locals)
    ingestion_state = xdata_seed.run_ingestion_stage(
        runtime,
        input_plan=x_input_plan,
        read_config=read_x_config,
        read_callbacks=read_x_callbacks,
        ingestion_options=ingestion_options,
        ensure_gene_universe_fn=domain._ensure_gene_universe_for_x,
        process_x_input_file_fn=domain._process_x_input_file,
        remove_tag_from_input_fn=domain.pegs_remove_tag_from_input,
        log_fn=domain.log,
        info_level=domain.INFO,
        debug_level=domain.DEBUG,
    )

    post_options = build_read_x_post_options(
        read_x_locals,
        batches=ingestion_state["batches"],
        num_ignored_gene_sets=ingestion_state["num_ignored_gene_sets"],
        ignored_for_fraction_inc=ingestion_state["ignored_for_fraction_inc"],
    )
    post_callbacks = XReadPostCallbacks(
        standardize_qc_metrics_after_x_read_fn=domain._standardize_qc_metrics_after_x_read,
        maybe_correct_gene_set_betas_after_x_read_fn=domain._maybe_correct_gene_set_betas_after_x_read,
        maybe_limit_initial_gene_sets_by_p_fn=domain._maybe_limit_initial_gene_sets_by_p,
        maybe_prune_gene_sets_after_x_read_fn=domain._maybe_prune_gene_sets_after_x_read,
        initialize_hyper_defaults_after_x_read_fn=domain._initialize_hyper_defaults_after_x_read,
        maybe_learn_batch_hyper_after_x_read_fn=domain._maybe_learn_batch_hyper_after_x_read,
        maybe_adjust_overaggressive_p_filter_after_x_read_fn=domain._maybe_adjust_overaggressive_p_filter_after_x_read,
        apply_post_read_gene_set_size_and_qc_filters_fn=domain._apply_post_read_gene_set_size_and_qc_filters,
        maybe_filter_zero_uncorrected_betas_after_x_read_fn=domain._maybe_filter_zero_uncorrected_betas_after_x_read,
        maybe_reduce_gene_sets_to_max_after_x_read_fn=domain._maybe_reduce_gene_sets_to_max_after_x_read,
        record_read_x_counts_fn=record_read_x_counts,
    )
    xdata_seed.run_post_stage(
        runtime,
        post_options=post_options,
        post_callbacks=post_callbacks,
        log_fn=domain.log,
        debug_level=domain.DEBUG,
    )


def log_runtime_environment_if_requested(domain, options):
    if options.hide_opts:
        return
    domain.log("Python version: %s" % domain.sys.version)
    domain.log("Numpy version: %s" % np.__version__)
    domain.log("Scipy version: %s" % scipy.__version__)
    domain.log("Options: %s" % options)


def read_gene_map(domain, runtime_state, gene_map_in, gene_map_orig_gene_col=1, gene_map_new_gene_col=2, allow_multi=False):
    runtime_state.gene_label_map = parse_gene_map_file(
        gene_map_in,
        gene_map_orig_gene_col=gene_map_orig_gene_col,
        gene_map_new_gene_col=gene_map_new_gene_col,
        allow_multi=allow_multi,
        bail_fn=domain.bail,
    )


def init_gene_locs(domain, runtime_state, gene_loc_file):
    domain.log("Reading --gene-loc-file %s" % gene_loc_file)
    (
        runtime_state.gene_chrom_name_pos,
        runtime_state.gene_to_chrom,
        runtime_state.gene_to_pos,
    ) = read_loc_file_with_gene_map(
        gene_loc_file,
        gene_label_map=runtime_state.gene_label_map,
        clean_chrom_fn=clean_chrom_name,
        warn_fn=domain.warn,
        bail_fn=domain.bail,
    )


def initialize_main_mappings(domain, runtime_state, options):
    if options.gene_map_in:
        read_gene_map(
            domain,
            runtime_state,
            options.gene_map_in,
            options.gene_map_orig_gene_col,
            options.gene_map_new_gene_col,
        )
    if options.gene_loc_file:
        init_gene_locs(domain, runtime_state, options.gene_loc_file)


def read_gene_set_statistics(
    domain,
    runtime_state,
    stats_in,
    *,
    stats_id_col=None,
    stats_exp_beta_tilde_col=None,
    stats_beta_tilde_col=None,
    stats_p_col=None,
    stats_se_col=None,
    stats_beta_col=None,
    stats_beta_uncorrected_col=None,
    ignore_negative_exp_beta=False,
    max_gene_set_p=None,
    min_gene_set_beta=None,
    min_gene_set_beta_uncorrected=None,
    return_only_ids=False,
):
    return load_and_apply_gene_set_statistics_to_runtime(
        runtime_state,
        stats_in,
        stats_id_col=stats_id_col,
        stats_exp_beta_tilde_col=stats_exp_beta_tilde_col,
        stats_beta_tilde_col=stats_beta_tilde_col,
        stats_p_col=stats_p_col,
        stats_se_col=stats_se_col,
        stats_beta_col=stats_beta_col,
        stats_beta_uncorrected_col=stats_beta_uncorrected_col,
        ignore_negative_exp_beta=ignore_negative_exp_beta,
        max_gene_set_p=max_gene_set_p,
        min_gene_set_beta=min_gene_set_beta,
        min_gene_set_beta_uncorrected=min_gene_set_beta_uncorrected,
        return_only_ids=return_only_ids,
        open_text_fn=domain.open_gz,
        get_col_fn=runtime_state._get_col,
        parse_log_fn=lambda message: domain.log(message, domain.INFO),
        apply_log_fn=lambda message: domain.log(message, domain.DEBUG),
        warn_fn=domain.warn,
        bail_fn=domain.bail,
    )


def read_gene_set_phewas_statistics(
    domain,
    runtime_state,
    stats_in,
    *,
    stats_id_col=None,
    stats_pheno_col=None,
    stats_beta_col=None,
    stats_beta_uncorrected_col=None,
    min_gene_set_beta=None,
    min_gene_set_beta_uncorrected=None,
    update_X=False,
    phenos_to_match=None,
    return_only_ids=False,
    max_num_entries_at_once=None,
):
    return load_and_apply_gene_set_phewas_statistics_to_runtime(
        runtime_state,
        stats_in,
        stats_id_col=stats_id_col,
        stats_pheno_col=stats_pheno_col,
        stats_beta_col=stats_beta_col,
        stats_beta_uncorrected_col=stats_beta_uncorrected_col,
        min_gene_set_beta=min_gene_set_beta,
        min_gene_set_beta_uncorrected=min_gene_set_beta_uncorrected,
        update_X=update_X,
        phenos_to_match=phenos_to_match,
        return_only_ids=return_only_ids,
        max_num_entries_at_once=max_num_entries_at_once,
        open_text_fn=domain.open_gz,
        get_col_fn=runtime_state._get_col,
        construct_map_to_ind_fn=construct_map_to_ind,
        warn_fn=domain.warn,
        bail_fn=domain.bail,
        log_fn=lambda message: domain.log(message, domain.DEBUG),
    )


def derive_factor_anchor_masks(domain, runtime, options):
    return domain.pegs_derive_factor_anchor_masks(
        genes=runtime.genes,
        phenos=runtime.phenos,
        anchor_genes=options.anchor_genes,
        anchor_phenos=options.anchor_phenos,
        bail_fn=domain.bail,
    )


def read_gene_phewas_bfs(
    domain,
    state,
    gene_phewas_bfs_in,
    gene_phewas_bfs_id_col=None,
    gene_phewas_bfs_pheno_col=None,
    anchor_genes=None,
    anchor_phenos=None,
    gene_phewas_bfs_log_bf_col=None,
    gene_phewas_bfs_combined_col=None,
    gene_phewas_bfs_prior_col=None,
    phewas_gene_to_X_gene_in=None,
    min_value=None,
    max_num_entries_at_once=None,
    **kwargs
):
    cached = dict(locals())
    cached.pop("domain", None)
    cached.pop("state", None)
    cached.pop("kwargs", None)
    state.cached_gene_phewas_call = cached

    if gene_phewas_bfs_in is None:
        domain.bail("Require --gene-stats-in or --gene-phewas-bfs-in for this operation")

    domain.log("Reading --gene-phewas-bfs-in file %s" % gene_phewas_bfs_in, domain.INFO)
    if state.genes is None:
        domain.bail("Need to initialixe --X before reading gene_phewas")

    phewas_gene_to_X_gene = None
    if phewas_gene_to_X_gene_in is not None:
        phewas_gene_to_X_gene = parse_gene_map_file(
            phewas_gene_to_X_gene_in,
            allow_multi=True,
            bail_fn=domain.bail,
        )

    load_and_apply_gene_phewas_bfs_to_runtime(
        state,
        gene_phewas_bfs_in,
        gene_phewas_bfs_id_col=gene_phewas_bfs_id_col,
        gene_phewas_bfs_pheno_col=gene_phewas_bfs_pheno_col,
        anchor_genes=anchor_genes,
        anchor_phenos=anchor_phenos,
        gene_phewas_bfs_log_bf_col=gene_phewas_bfs_log_bf_col,
        gene_phewas_bfs_combined_col=gene_phewas_bfs_combined_col,
        gene_phewas_bfs_prior_col=gene_phewas_bfs_prior_col,
        phewas_gene_to_x_gene=phewas_gene_to_X_gene,
        min_value=min_value,
        max_num_entries_at_once=max_num_entries_at_once,
        open_text_fn=domain.open_gz,
        get_col_fn=state._get_col,
        construct_map_to_ind_fn=construct_map_to_ind,
        warn_fn=domain.warn,
        bail_fn=domain.bail,
        log_fn=lambda message: domain.log(message, domain.DEBUG),
    )
    state.phewas_state = sync_phewas_runtime_state(state)


def has_loaded_gene_phewas(runtime):
    return (
        runtime.gene_pheno_Y is not None
        or runtime.gene_pheno_combined_prior_Ys is not None
        and runtime.gene_pheno_priors is not None
    )


def reread_gene_phewas_bfs(domain, state):
    if state.cached_gene_phewas_call is None:
        return
    domain.log("Rereading gene phewas bfs...")
    read_gene_phewas_bfs(domain, state, **state.cached_gene_phewas_call)
