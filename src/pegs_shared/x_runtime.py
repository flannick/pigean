from __future__ import annotations

import copy

import numpy as np


def assign_default_batches(batches, orig_files, batch_all_for_hyper, first_for_hyper):
    batches = list(batches)
    used_batches = set([str(b) for b in batches if b is not None])
    next_batch_num = 1

    def _generate_new_batch(new_batch_num):
        new_batch = "BATCH%d" % new_batch_num
        while new_batch in used_batches:
            new_batch_num += 1
            new_batch = "BATCH%d" % new_batch_num
        used_batches.add(new_batch)
        return new_batch, new_batch_num

    for i in range(len(batches)):
        if batches[i] is None:
            batches[i], next_batch_num = _generate_new_batch(next_batch_num)

            if batch_all_for_hyper:
                for j in range(i + 1, len(batches)):
                    batches[j] = batches[i]
                break
            for j in range(i + 1, len(batches)):
                if batches[j] is None and orig_files[i] == orig_files[j]:
                    batches[j] = batches[i]

        if first_for_hyper:
            for j in range(i + 1, len(batches)):
                if batches[j] != batches[i]:
                    batches[j] = None
            break
    return batches


def initialize_read_x_batch_seed_state(
    runtime,
    xdata_seed,
    batches,
    orig_files,
    *,
    batch_all_for_hyper,
    first_for_hyper,
    update_hyper_sigma,
    update_hyper_p,
    first_for_sigma_cond,
    record_params_fn=None,
    log_fn=None,
):
    if log_fn is None:
        log_fn = lambda _message: None

    batches = assign_default_batches(
        batches=batches,
        orig_files=orig_files,
        batch_all_for_hyper=batch_all_for_hyper,
        first_for_hyper=first_for_hyper,
    )

    if record_params_fn is not None:
        record_params_fn({"num_X_batches": len(batches)})

    if update_hyper_sigma or update_hyper_p:
        num_batched = len([x for x in batches if x is not None])
        num_unique_batched = len(set([x for x in batches if x is not None]))
        num_unbatched = len([x for x in batches if x is None])
        log_fn(
            "Will learn parameters for %d files as %d batches and fill in %d additional files from the first"
            % (num_batched, num_unique_batched, num_unbatched)
        )
    if first_for_sigma_cond:
        log_fn("Will fix conditional sigma from the first batch")

    num_ignored_gene_sets = np.zeros((len(batches)))

    xdata_seed.seed_runtime_read_x_state(runtime)

    return batches, num_ignored_gene_sets


def is_metric_qc_filter_active(filter_gene_set_metric_z):
    return filter_gene_set_metric_z is not None and filter_gene_set_metric_z > 0


def initialize_filtered_gene_set_state(runtime, update_hyper_p):
    runtime.gene_sets_ignored = []
    if runtime.gene_set_labels is not None:
        runtime.gene_set_labels_ignored = np.array([])

    runtime.col_sums_ignored = np.array([])
    runtime.scale_factors_ignored = np.array([])
    runtime.mean_shifts_ignored = np.array([])
    runtime.beta_tildes_ignored = np.array([])
    runtime.p_values_ignored = np.array([])
    runtime.ses_ignored = np.array([])
    runtime.z_scores_ignored = np.array([])
    runtime.se_inflation_factors_ignored = np.array([])

    runtime.beta_tildes = np.array([])
    runtime.p_values = np.array([])
    runtime.ses = np.array([])
    runtime.z_scores = np.array([])

    runtime.se_inflation_factors = None

    runtime.total_qc_metrics = None
    runtime.mean_qc_metrics = None
    runtime.total_qc_metrics_missing = None
    runtime.mean_qc_metrics_missing = None
    runtime.total_qc_metrics_ignored = None
    runtime.mean_qc_metrics_ignored = None
    runtime.total_qc_metrics_directions = None

    runtime.sigma2s = None
    runtime.sigma2s_missing = None
    if update_hyper_p is not None:
        runtime.ps = np.array([])
    else:
        runtime.ps = None
    runtime.ps_missing = None


def maybe_prepare_filtered_correlation(
    runtime,
    run_corrected_ols,
    gene_cor_file,
    gene_loc_file,
    gene_cor_file_gene_col,
    gene_cor_file_cor_start_col,
    min_correlation=0.05,
):
    if run_corrected_ols and runtime.y_corr is None:
        correlation_m = runtime._read_correlations(
            gene_cor_file,
            gene_loc_file,
            gene_cor_file_gene_col=gene_cor_file_gene_col,
            gene_cor_file_cor_start_col=gene_cor_file_cor_start_col,
        )
        runtime._set_Y(
            runtime.Y,
            runtime.Y_for_regression,
            runtime.Y_exomes,
            runtime.Y_positive_controls,
            runtime.Y_case_counts,
            Y_corr_m=correlation_m,
            store_corr_sparse=run_corrected_ols,
            skip_V=True,
            skip_scale_factors=True,
            min_correlation=min_correlation,
        )


def resolve_read_x_run_logistic(
    runtime,
    run_logistic,
    max_for_linear,
    background_log_bf,
    *,
    record_param_fn=None,
    log_fn=None,
):
    if log_fn is None:
        log_fn = lambda _message, _level=None: None

    if (
        not run_logistic
        and runtime.Y_for_regression is not None
        and np.max(np.exp(runtime.Y_for_regression + background_log_bf) / (1 + np.exp(runtime.Y_for_regression + background_log_bf))) > max_for_linear
    ):
        log_fn("Switching to logistic sampling due to high Y values")
        run_logistic = True

    if record_param_fn is not None:
        record_param_fn("read_X_run_logistic", run_logistic)
    return run_logistic


def record_read_x_counts(runtime, *, record_param_fn=None, log_fn=None):
    if record_param_fn is not None:
        record_param_fn("num_gene_sets_read", len(runtime.gene_sets))
        record_param_fn("num_genes_read", len(runtime.genes))
    if log_fn is not None:
        log_fn("Read %d gene sets and %d genes" % (len(runtime.gene_sets), len(runtime.genes)))


def standardize_qc_metrics_after_x_read(runtime):
    if runtime.total_qc_metrics is not None:
        total_qc_metrics = runtime.total_qc_metrics
        if runtime.total_qc_metrics_ignored is not None:
            total_qc_metrics = np.vstack((runtime.total_qc_metrics, runtime.total_qc_metrics_ignored))

        runtime.total_qc_metrics = (runtime.total_qc_metrics - np.mean(total_qc_metrics, axis=0)) / np.std(total_qc_metrics, axis=0)
        if runtime.total_qc_metrics_ignored is not None:
            runtime.total_qc_metrics_ignored = (
                runtime.total_qc_metrics_ignored - np.mean(total_qc_metrics, axis=0)
            ) / np.std(total_qc_metrics, axis=0)

    if runtime.mean_qc_metrics is not None:
        mean_qc_metrics = np.append(
            runtime.mean_qc_metrics,
            runtime.mean_qc_metrics_ignored if runtime.mean_qc_metrics_ignored is not None else [],
        )
        runtime.mean_qc_metrics = (runtime.mean_qc_metrics - np.mean(mean_qc_metrics)) / np.std(mean_qc_metrics)
        if runtime.mean_qc_metrics_ignored is not None:
            runtime.mean_qc_metrics_ignored = (
                runtime.mean_qc_metrics_ignored - np.mean(mean_qc_metrics)
            ) / np.std(mean_qc_metrics)


def maybe_correct_gene_set_betas_after_x_read(
    runtime,
    filter_gene_set_p,
    correct_betas_mean,
    correct_betas_var,
    filter_using_phewas,
    *,
    log_fn=None,
):
    if log_fn is None:
        log_fn = lambda _message: None

    if not (
        filter_gene_set_p is not None
        and (correct_betas_mean or correct_betas_var)
        and runtime.beta_tildes is not None
        and runtime.ses is not None
    ):
        return

    (
        runtime.beta_tildes,
        runtime.ses,
        runtime.z_scores,
        runtime.p_values,
        runtime.se_inflation_factors,
    ) = runtime._correct_beta_tildes(
        runtime.beta_tildes,
        runtime.ses,
        runtime.se_inflation_factors,
        runtime.total_qc_metrics,
        runtime.total_qc_metrics_directions,
        correct_mean=correct_betas_mean,
        correct_var=correct_betas_var,
        correct_ignored=True,
        fit=True,
    )
    newly_below_p_mask = runtime.p_values <= filter_gene_set_p
    if filter_using_phewas:
        newly_below_p_mask = np.full(len(runtime.p_values), True)

    if np.sum(newly_below_p_mask) == 0:
        newly_below_p_mask[np.argmin(runtime.p_values)] = True
    if np.sum(newly_below_p_mask) != len(newly_below_p_mask):
        log_fn(
            "Ignoring %d gene sets whose p-value increased after adjusting betas (kept %d)"
            % (np.sum(~newly_below_p_mask), np.sum(newly_below_p_mask))
        )
        runtime.subset_gene_sets(newly_below_p_mask, ignore_missing=True, keep_missing=False, skip_V=True)


def maybe_limit_initial_gene_sets_by_p(runtime, max_num_gene_sets_initial, *, log_fn=None):
    if log_fn is None:
        log_fn = lambda _message: None

    if runtime.p_values is None or max_num_gene_sets_initial is None:
        return

    if max_num_gene_sets_initial > 0 and max_num_gene_sets_initial < len(runtime.p_values):
        p_value_filter = np.partition(runtime.p_values, max_num_gene_sets_initial - 1)[max_num_gene_sets_initial - 1]
        log_fn("Keeping only %d most significant gene sets due to --max-num-gene-sets-initial" % max_num_gene_sets_initial)
        runtime.subset_gene_sets(runtime.p_values <= p_value_filter, ignore_missing=True, keep_missing=False, skip_V=True)


def maybe_prune_gene_sets_after_x_read(
    runtime,
    skip_betas,
    prune_gene_sets,
    prune_deterministically,
    weighted_prune_gene_sets,
):
    if skip_betas and runtime.Y is not None:
        return

    runtime._prune_gene_sets(
        prune_gene_sets,
        prune_deterministically=prune_deterministically,
        keep_missing=False,
        ignore_missing=True,
        skip_V=True,
    )

    if weighted_prune_gene_sets and runtime.Y is not None:
        gene_weights = np.exp(runtime.Y + runtime.background_log_bf) / (
            1 + np.exp(runtime.Y + runtime.background_log_bf)
        )
        runtime._prune_gene_sets(
            weighted_prune_gene_sets,
            prune_deterministically=prune_deterministically,
            keep_missing=False,
            ignore_missing=True,
            skip_V=True,
            gene_weights=gene_weights,
        )


def initialize_hyper_defaults_after_x_read(
    runtime,
    initial_p,
    update_hyper_p,
    sigma_power,
    initial_sigma2_cond,
    update_hyper_sigma,
    initial_sigma2,
    sigma_soft_threshold_95,
    sigma_soft_threshold_5,
    *,
    warn_fn=None,
    log_fn=None,
):
    if warn_fn is None:
        warn_fn = lambda _message: None
    if log_fn is None:
        log_fn = lambda _message: None

    if runtime.p is None:
        if initial_p is not None and type(initial_p) is list:
            runtime.set_p(np.mean(initial_p))
            if update_hyper_p:
                warn_fn("Since --update-hyper-p was passed, using average --p-noninf (%.3g) as initial condition" % runtime.p)
            if runtime.Y is not None:
                if runtime.ps is None:
                    num_gene_sets = len(runtime.gene_sets) if runtime.gene_sets is not None else 0
                    runtime.ps = np.full(num_gene_sets, runtime.p, dtype=float)
        else:
            runtime.set_p(initial_p)
    if runtime.sigma_power is None:
        runtime.set_sigma(runtime.sigma2, sigma_power)
    fixed_sigma_cond = False
    if runtime.sigma2 is None:
        if initial_sigma2_cond is not None:
            if not update_hyper_sigma:
                fixed_sigma_cond = True
            runtime.set_sigma(runtime.p * initial_sigma2_cond, runtime.sigma_power)
        else:
            runtime.set_sigma(initial_sigma2, runtime.sigma_power)

    if sigma_soft_threshold_95 is not None and sigma_soft_threshold_5 is not None:
        if sigma_soft_threshold_95 < 0 or sigma_soft_threshold_5 < 0:
            warn_fn("Ignoring sigma soft thresholding since both are not positive")
        else:
            frac_95 = float(sigma_soft_threshold_95) / len(runtime.genes)
            x1 = np.sqrt(frac_95 * (1 - frac_95))
            y1 = 0.95

            frac_5 = float(sigma_soft_threshold_5) / len(runtime.genes)
            x2 = np.sqrt(frac_5 * (1 - frac_5))
            y2 = 0.05
            L = 1

            if x2 < x1:
                warn_fn("--sigma-threshold-5 (%.3g) is less than --sigma-threshold-95 (%.3g); this is the opposite of what you usually want as it will threshold smaller gene sets rather than larger ones")

            runtime.sigma_threshold_k = -(np.log(1 / y2 - L) - np.log(1 / y1 - 1)) / (x2 - x1)
            runtime.sigma_threshold_xo = (x1 * np.log(1 / y2 - L) - x2 * np.log(1 / y1 - L)) / (np.log(1 / y2 - L) - np.log(1 / y1 - L))

            log_fn("Thresholding sigma with k=%.3g, xo=%.3g" % (runtime.sigma_threshold_k, runtime.sigma_threshold_xo))

    return fixed_sigma_cond


def maybe_adjust_overaggressive_p_filter_after_x_read(
    runtime,
    filter_gene_set_p,
    increase_filter_gene_set_p,
    filter_using_phewas,
    *,
    log_fn=None,
):
    if log_fn is None:
        log_fn = lambda _message: None

    if filter_gene_set_p is None or increase_filter_gene_set_p is None or runtime.p_values is None or runtime.p_values_ignored is None:
        return

    kept = float(len(runtime.p_values))
    total = kept + float(len(runtime.p_values_ignored))
    if total <= 0:
        return
    kept_fraction = kept / total
    if kept_fraction < increase_filter_gene_set_p:
        log_fn(
            "Kept fraction %.4g is below requested minimum %.4g after read_X; "
            "post-read adjustment cannot restore filtered sets, so keeping current set"
            % (kept_fraction, increase_filter_gene_set_p)
        )


def apply_post_read_gene_set_size_and_qc_filters(
    runtime,
    min_gene_set_size,
    max_gene_set_size,
    filter_gene_set_metric_z,
    *,
    log_fn=None,
):
    if log_fn is None:
        log_fn = lambda _message: None

    if runtime.X_orig is None:
        return

    col_sums = runtime.get_col_sums(runtime.X_orig, num_nonzero=True)
    size_ignore = col_sums < min_gene_set_size
    if np.sum(size_ignore) > 0:
        size_mask = ~size_ignore
        log_fn("Ignoring %d gene sets due to too few genes (kept %d)" % (np.sum(size_ignore), np.sum(size_mask)))
        runtime.subset_gene_sets(size_mask, keep_missing=False, skip_V=True)

    col_sums = runtime.get_col_sums(runtime.X_orig, num_nonzero=True)
    size_ignore = col_sums > max_gene_set_size
    if np.sum(size_ignore) > 0:
        size_mask = ~size_ignore
        log_fn("Ignoring %d gene sets due to too many genes (kept %d)" % (np.sum(size_ignore), np.sum(size_mask)))
        runtime.subset_gene_sets(size_mask, keep_missing=False, skip_V=True)

    if runtime.total_qc_metrics is not None and is_metric_qc_filter_active(filter_gene_set_metric_z):
        filter_mask = np.abs(runtime.mean_qc_metrics) < filter_gene_set_metric_z
        filter_ignore = ~filter_mask
        log_fn("Ignoring %d gene sets due to QC metric filters (kept %d)" % (np.sum(filter_ignore), np.sum(filter_mask)))
        runtime.subset_gene_sets(filter_mask, keep_missing=False, ignore_missing=True, skip_V=True)


def make_add_to_x_handler(runtime, read_config, read_callbacks, *, run_logistic):
    def _add_to_x(mat_info, genes, gene_sets, tag=None, skip_scale_factors=False, fname=None):
        if tag is not None:
            gene_sets = ["%s_%s" % (tag, gene_set) for gene_set in gene_sets]

        is_dense = False
        if isinstance(mat_info, tuple):
            (data, row, col) = mat_info
            cur_X = read_callbacks.sparse_module.csc_matrix((data, (row, col)), shape=(len(genes), len(gene_sets)))
            if cur_X.shape[1] == 0:
                return (0, 0)
        else:
            mat_info, genes = read_callbacks.normalize_dense_gene_rows_fn(mat_info, genes, runtime.gene_label_map)
            cur_X, gene_sets, should_skip_dense = read_callbacks.build_sparse_x_from_dense_input_fn(
                runtime,
                mat_info=mat_info,
                genes=genes,
                gene_sets=gene_sets,
                x_sparsify=read_config.x_sparsify,
                min_gene_set_size=read_config.min_gene_set_size,
                add_ext=read_config.add_ext,
                add_top=read_config.add_top,
                add_bottom=read_config.add_bottom,
                fname=fname,
            )
            if should_skip_dense:
                return (0, 0)
            cur_X, genes = read_callbacks.reindex_x_rows_to_current_genes_fn(runtime, cur_X=cur_X, genes=genes)

        cur_X = read_callbacks.normalize_gene_set_weights_fn(
            runtime,
            cur_X=cur_X,
            threshold_weights=read_config.threshold_weights,
            cap_weights=read_config.cap_weights,
        )
        (
            cur_X,
            genes,
            gene_sets,
            gene_ignored_N,
            cur_X_missing_genes_int,
            gene_ignored_N_missing_int,
            genes_missing_new,
            cur_X_missing_genes_new,
            gene_ignored_N_missing_new,
        ) = read_callbacks.partition_missing_gene_rows_fn(
            runtime,
            cur_X=cur_X,
            genes=genes,
            gene_sets=gene_sets,
        )

        cur_X = read_callbacks.maybe_permute_gene_set_rows_fn(
            runtime,
            cur_X=cur_X,
            permute_gene_sets=read_config.permute_gene_sets,
        )

        (
            cur_X,
            gene_sets,
            p_value_ignore,
            gene_ignored_N,
            cur_X_missing_genes_new,
            gene_ignored_N_missing_new,
            cur_X_missing_genes_int,
            gene_ignored_N_missing_int,
            total_qc_metrics,
            mean_qc_metrics,
            total_qc_metrics_directions,
        ) = read_callbacks.maybe_prefilter_x_block_fn(
            runtime,
            cur_X=cur_X,
            gene_sets=gene_sets,
            run_logistic=run_logistic,
            filter_gene_set_p=read_config.filter_gene_set_p,
            filter_gene_set_metric_z=read_config.filter_gene_set_metric_z,
            filter_using_phewas=read_config.filter_using_phewas,
            increase_filter_gene_set_p=read_config.increase_filter_gene_set_p,
            filter_negative=read_config.filter_negative,
            cur_X_missing_genes_new=cur_X_missing_genes_new,
            gene_ignored_N_missing_new=gene_ignored_N_missing_new,
            cur_X_missing_genes_int=cur_X_missing_genes_int,
            gene_ignored_N_missing_int=gene_ignored_N_missing_int,
            gene_ignored_N=gene_ignored_N,
        )

        runtime.is_dense_gene_set = read_callbacks.np_module.append(
            runtime.is_dense_gene_set,
            read_callbacks.np_module.full(len(gene_sets), is_dense),
        )

        num_new_gene_sets = len(gene_sets)
        num_old_gene_sets = len(runtime.gene_sets) if runtime.gene_sets is not None else 0
        if runtime.X_orig is not None:
            cur_X = read_callbacks.sparse_module.hstack((runtime.X_orig, cur_X))
            gene_sets = runtime.gene_sets + gene_sets

        cur_X, genes = read_callbacks.merge_missing_gene_rows_fn(
            runtime,
            cur_X=cur_X,
            genes=genes,
            num_old_gene_sets=num_old_gene_sets,
            num_new_gene_sets=num_new_gene_sets,
            cur_X_missing_genes_int=cur_X_missing_genes_int,
            gene_ignored_N_missing_int=gene_ignored_N_missing_int,
            cur_X_missing_genes_new=cur_X_missing_genes_new,
            gene_ignored_N_missing_new=gene_ignored_N_missing_new,
            genes_missing_new=genes_missing_new,
        )

        return read_callbacks.finalize_added_x_block_fn(
            runtime,
            cur_X=cur_X,
            genes=genes,
            gene_sets=gene_sets,
            skip_scale_factors=skip_scale_factors,
            p_value_ignore=p_value_ignore,
            gene_ignored_N=gene_ignored_N,
            total_qc_metrics=total_qc_metrics,
            mean_qc_metrics=mean_qc_metrics,
            total_qc_metrics_directions=total_qc_metrics_directions,
        )

    return _add_to_x


def ingest_x_inputs(
    runtime,
    X_ins,
    is_dense,
    batches,
    labels,
    initial_ps,
    num_ignored_gene_sets,
    *,
    only_ids,
    x_sparsify,
    min_gene_set_size,
    only_inc_genes,
    fraction_inc_genes,
    ignore_genes,
    max_num_entries_at_once,
    add_to_x_fn,
    process_x_input_file_fn,
    remove_tag_from_input_fn,
    log_fn,
    info_level,
    debug_level,
):
    ignored_for_fraction_inc = 0
    for input_index in range(len(X_ins)):
        X_in = X_ins[input_index]
        (X_in, tag) = remove_tag_from_input_fn(X_in)

        log_fn("Reading X %d of %d from --X-in file %s" % (input_index + 1, len(X_ins), X_in), info_level)

        num_too_small, ignored_for_fraction_inc, processed_input = process_x_input_file_fn(
            runtime,
            X_in=X_in,
            tag=tag,
            is_dense_input=is_dense[input_index],
            only_ids=only_ids,
            x_sparsify=x_sparsify,
            batch_value=batches[input_index],
            label_value=labels[input_index],
            initial_p_value=initial_ps[input_index] if initial_ps is not None else None,
            num_ignored_gene_sets=num_ignored_gene_sets,
            input_index=input_index,
            add_to_x_fn=add_to_x_fn,
            min_gene_set_size=min_gene_set_size,
            only_inc_genes=only_inc_genes,
            fraction_inc_genes=fraction_inc_genes,
            ignore_genes=ignore_genes,
            max_num_entries_at_once=max_num_entries_at_once,
        )
        if not processed_input:
            continue

        log_fn("Ignored %d gene sets due to too few genes" % num_too_small, debug_level)

    return ignored_for_fraction_inc


def run_read_x_ingestion(
    runtime,
    *,
    X_ins,
    is_dense,
    batches,
    labels,
    initial_ps,
    num_ignored_gene_sets,
    read_config,
    read_callbacks,
    run_logistic,
    only_ids,
    add_all_genes,
    only_inc_genes,
    fraction_inc_genes,
    ignore_genes,
    max_num_entries_at_once,
    ensure_gene_universe_fn,
    process_x_input_file_fn,
    remove_tag_from_input_fn,
    log_fn,
    info_level,
    debug_level,
):
    if only_inc_genes:
        add_all_genes = True

    ensure_gene_universe_fn(
        runtime,
        X_ins=X_ins,
        is_dense=is_dense,
        add_all_genes=add_all_genes,
        only_ids=only_ids,
        only_inc_genes=only_inc_genes,
        fraction_inc_genes=fraction_inc_genes,
    )

    add_to_x_fn = make_add_to_x_handler(
        runtime,
        read_config,
        read_callbacks,
        run_logistic=run_logistic,
    )

    return ingest_x_inputs(
        runtime,
        X_ins,
        is_dense,
        batches,
        labels,
        initial_ps,
        num_ignored_gene_sets,
        only_ids=only_ids,
        x_sparsify=read_config.x_sparsify,
        min_gene_set_size=read_config.min_gene_set_size,
        only_inc_genes=only_inc_genes,
        fraction_inc_genes=fraction_inc_genes,
        ignore_genes=ignore_genes,
        max_num_entries_at_once=max_num_entries_at_once,
        add_to_x_fn=add_to_x_fn,
        process_x_input_file_fn=process_x_input_file_fn,
        remove_tag_from_input_fn=remove_tag_from_input_fn,
        log_fn=log_fn,
        info_level=info_level,
        debug_level=debug_level,
    )
