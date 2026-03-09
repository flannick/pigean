import os
import csv
import gzip
import io
import re
import sys
import time
import urllib.error
import urllib.request
import copy

import numpy as np
import scipy.stats
import scipy.sparse as sparse

from pegs_shared.types import (
    AlignedGeneBfs,
    AlignedGeneCovariates,
    FactorInputData,
    HyperparameterData,
    ParsedGeneBfs,
    ParsedGeneCovariates,
    ParsedGenePhewasBfs,
    ParsedGeneSetStats,
    PhewasFileColumnInfo,
    PhewasInputResolution,
    PhewasRuntimeState,
    PhewasStageConfig,
    ReadXPipelineConfig,
    XData,
    XInputPlan,
    XReadCallbacks,
    XReadConfig,
    XReadIngestionOptions,
    XReadPostCallbacks,
    XReadPostOptions,
    YData,
)
from pegs_shared.cli import (
    _default_bail,
    apply_cli_config_overrides,
    callback_set_comma_separated_args,
    callback_set_comma_separated_args_as_float,
    callback_set_comma_separated_args_as_set,
    collect_cli_specified_dests,
    coerce_option_int_list,
    configure_random_seed,
    emit_stderr_warning,
    fail_removed_cli_aliases,
    format_removed_option_message,
    get_tar_write_mode_for_bundle_path,
    harmonize_cli_mode_args,
    initialize_cli_logging,
    is_path_like_dest,
    is_remote_path,
    iter_parser_options,
    json_safe,
    load_json_config,
    merge_dicts,
    open_optional_log_handle,
    resolve_config_path_value,
)
from pegs_shared.io_common import (
    GeneSetStatsTable,
    GeneStatsTable,
    TsvTable,
    clean_chrom_name,
    construct_map_to_ind,
    is_dig_open_data_ancestry_trait_spec,
    open_dig_open_data,
    open_text_auto,
    open_text_with_retry,
    parse_gene_map_file,
    read_loc_file_with_gene_map,
    read_tsv,
    resolve_column_index,
    urlopen_with_retry,
    write_tsv,
    is_dig_open_data_uri,
    is_gz_file,
)
from pegs_shared.xdata import (
    build_read_x_pipeline_config,
    build_read_x_ingestion_options,
    build_read_x_post_options,
    initialize_matrix_and_gene_index_state,
    prepare_read_x_inputs,
    xdata_from_input_plan,
)
from pegs_shared.ydata import (
    apply_hyperparameter_data_to_runtime,
    apply_phewas_runtime_state_to_runtime,
    apply_runtime_state_bundle_to_runtime,
    apply_y_data_to_runtime,
    build_y_data_from_inputs,
    hyperparameter_data_from_runtime,
    phewas_runtime_state_from_runtime,
    runtime_state_bundle_from_runtime,
    set_runtime_y_from_inputs,
    sync_hyperparameter_state,
    sync_phewas_runtime_state,
    sync_runtime_state_bundle,
    sync_y_state,
    y_data_from_runtime,
)
from pegs_shared.bundle import (
    BundleDefaultsApplication,
    BundleManifest,
    apply_bundle_defaults_to_options,
    collect_file_metadata,
    ensure_parent_dir_for_file,
    hash_file_sha256,
    is_huge_statistics_bundle_path,
    load_and_apply_bundle_defaults,
    load_bundle_manifest,
    read_prefixed_tar_bundle,
    require_existing_nonempty_file,
    resolve_bundle_default_inputs,
    safe_extract_tar_to_temp,
    stage_file_into_dir,
    write_bundle_from_specs,
    write_prefixed_tar_bundle,
)
from pegs_shared.gene_io import (
    align_gene_scalar_map,
    align_gene_vector_map,
    load_aligned_gene_bfs,
    load_aligned_gene_covariates,
    parse_gene_bfs_file,
    parse_gene_covariates_file,
    parse_gene_set_statistics_file,
)
from pegs_shared.huge_cache import (
    apply_huge_statistics_meta_to_runtime,
    build_huge_statistics_matrix_row_genes,
    build_huge_statistics_meta,
    build_huge_statistics_score_maps,
    coerce_runtime_state_dict,
    combine_runtime_huge_scores,
    get_huge_statistics_paths_for_prefix,
    load_huge_statistics_sparse_and_vectors,
    read_huge_statistics_covariates_if_present,
    read_huge_statistics_text_tables,
    read_numeric_vector_file,
    resolve_huge_statistics_gene_vectors,
    validate_huge_statistics_loaded_shapes,
    write_huge_statistics_runtime_vectors,
    write_huge_statistics_sparse_components,
    write_huge_statistics_text_tables,
    write_numeric_vector_file,
)
from pegs_shared.phewas import (
    accumulate_factor_phewas_outputs,
    accumulate_standard_phewas_outputs,
    append_phewas_metric_block,
    expand_phewas_state_for_added_phenos,
    parse_gene_phewas_bfs_file,
    prepare_phewas_phenos_from_file,
    read_phewas_file_batch,
    resolve_phewas_file_columns,
)
from pegs_shared.regression import (
    correct_beta_tildes,
    compute_beta_tildes,
    compute_logistic_beta_tildes,
    compute_multivariate_beta_tildes,
    finalize_regression_outputs,
)
from pegs_shared.runtime_matrix import (
    calc_shift_scale_for_dense_block,
    calc_X_shift_scale,
    calculate_V_internal,
    compute_banded_y_corr_cholesky,
    get_num_X_blocks,
    iterate_X_blocks_internal,
    set_runtime_p,
    set_runtime_sigma,
    set_runtime_x_from_inputs,
    whiten_matrix_with_banded_cholesky,
)

EAGGL_BUNDLE_SCHEMA = "pigean_eaggl_bundle/v1"
EAGGL_BUNDLE_ALLOWED_DEFAULT_INPUTS = set([
    "X_in",
    "gene_stats_in",
    "gene_set_stats_in",
    "gene_phewas_bfs_in",
    "gene_set_phewas_stats_in",
])


def derive_factor_anchor_masks(genes, phenos, anchor_genes=None, anchor_phenos=None, *, bail_fn=None):
    if bail_fn is None:
        bail_fn = _default_bail

    anchor_gene_mask = None
    anchor_pheno_mask = None

    if anchor_genes is not None:
        anchor_gene_mask = np.array([x in anchor_genes for x in genes])
        if np.sum(anchor_gene_mask) == 0:
            bail_fn("None of the anchor genes are in X")

    if anchor_phenos is not None:
        anchor_pheno_mask = np.array([x in anchor_phenos for x in phenos])
        if np.sum(anchor_pheno_mask) == 0:
            bail_fn("None of the anchor phenos are in gene pheno matrix")

    return FactorInputData(
        anchor_gene_mask=anchor_gene_mask,
        anchor_pheno_mask=anchor_pheno_mask,
    )


def resolve_gene_phewas_input_for_stage(
    requested_input,
    reusable_inputs,
    *,
    read_gene_phewas,
    num_gene_phewas_filtered,
):
    decision = resolve_gene_phewas_input_decision_for_stage(
        requested_input=requested_input,
        reusable_inputs=reusable_inputs,
        read_gene_phewas=read_gene_phewas,
        num_gene_phewas_filtered=num_gene_phewas_filtered,
    )
    return decision.resolved_input


def _normalize_optional_path(path):
    if path is None:
        return None
    return os.path.realpath(os.path.abspath(path))


def _paths_match(a, b):
    if a is None or b is None:
        return False
    return _normalize_optional_path(a) == _normalize_optional_path(b)


def resolve_gene_phewas_input_decision_for_stage(
    requested_input,
    reusable_inputs,
    *,
    read_gene_phewas,
    num_gene_phewas_filtered,
):
    if requested_input is None:
        return PhewasInputResolution()

    if not read_gene_phewas:
        return PhewasInputResolution(
            requested_input=requested_input,
            resolved_input=requested_input,
            mode="re_read_file",
            reason="matrix_not_loaded",
        )

    if num_gene_phewas_filtered != 0:
        return PhewasInputResolution(
            requested_input=requested_input,
            resolved_input=requested_input,
            mode="re_read_file",
            reason="loaded_matrix_filtered",
        )

    for candidate in reusable_inputs:
        if _paths_match(requested_input, candidate):
            return PhewasInputResolution(
                requested_input=requested_input,
                resolved_input=None,
                mode="reuse_loaded_matrix",
                reason="requested_input_matches_loaded_source",
            )

    return PhewasInputResolution(
        requested_input=requested_input,
        resolved_input=requested_input,
        mode="re_read_file",
        reason="requested_input_not_reusable",
    )


def build_phewas_stage_config(
    *,
    gene_phewas_bfs_in,
    gene_phewas_bfs_id_col,
    gene_phewas_bfs_pheno_col,
    gene_phewas_bfs_log_bf_col,
    gene_phewas_bfs_combined_col,
    gene_phewas_bfs_prior_col,
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
    run_for_factors=False,
    batch_size=None,
    min_gene_factor_weight=0.0,
):
    return PhewasStageConfig(
        gene_phewas_bfs_in=gene_phewas_bfs_in,
        gene_phewas_bfs_id_col=gene_phewas_bfs_id_col,
        gene_phewas_bfs_pheno_col=gene_phewas_bfs_pheno_col,
        gene_phewas_bfs_log_bf_col=gene_phewas_bfs_log_bf_col,
        gene_phewas_bfs_combined_col=gene_phewas_bfs_combined_col,
        gene_phewas_bfs_prior_col=gene_phewas_bfs_prior_col,
        max_num_burn_in=max_num_burn_in,
        max_num_iter=max_num_iter,
        min_num_iter=min_num_iter,
        num_chains=num_chains,
        r_threshold_burn_in=r_threshold_burn_in,
        use_max_r_for_convergence=use_max_r_for_convergence,
        max_frac_sem=max_frac_sem,
        gauss_seidel=gauss_seidel,
        sparse_solution=sparse_solution,
        sparse_frac_betas=sparse_frac_betas,
        run_for_factors=run_for_factors,
        batch_size=batch_size,
        min_gene_factor_weight=min_gene_factor_weight,
    )


def apply_parsed_gene_phewas_bfs_to_runtime(
    runtime,
    parsed_phewas,
    *,
    anchor_genes=None,
    anchor_phenos=None,
    construct_map_to_ind_fn=None,
    bail_fn=None,
    log_fn=None,
):
    if construct_map_to_ind_fn is None:
        construct_map_to_ind_fn = construct_map_to_ind
    if bail_fn is None:
        bail_fn = _default_bail

    runtime.num_gene_phewas_filtered = parsed_phewas.num_filtered
    phenos = parsed_phewas.phenos
    row = parsed_phewas.row
    col = parsed_phewas.col
    Ys = parsed_phewas.Ys
    combineds = parsed_phewas.combineds
    priors = parsed_phewas.priors

    num_added_phenos = 0
    if runtime.phenos is not None and len(runtime.phenos) < len(phenos):
        num_added_phenos = len(phenos) - len(runtime.phenos)

    if num_added_phenos > 0:
        if runtime.X_phewas_beta is not None:
            runtime.X_phewas_beta = sparse.csc_matrix(
                sparse.vstack(
                    (
                        runtime.X_phewas_beta,
                        sparse.csc_matrix((num_added_phenos, runtime.X_phewas_beta.shape[1])),
                    )
                )
            )
        if runtime.X_phewas_beta_uncorrected is not None:
            runtime.X_phewas_beta_uncorrected = sparse.csc_matrix(
                sparse.vstack(
                    (
                        runtime.X_phewas_beta_uncorrected,
                        sparse.csc_matrix((num_added_phenos, runtime.X_phewas_beta_uncorrected.shape[1])),
                    )
                )
            )

    runtime.phenos = phenos
    runtime.pheno_to_ind = construct_map_to_ind_fn(phenos)

    if combineds is not None:
        runtime.gene_pheno_combined_prior_Ys = sparse.csc_matrix(
            (combineds, (row, col)),
            shape=(len(runtime.genes), len(runtime.phenos)),
        )

    if Ys is not None:
        runtime.gene_pheno_Y = sparse.csc_matrix(
            (Ys, (row, col)),
            shape=(len(runtime.genes), len(runtime.phenos)),
        )

    if priors is not None:
        runtime.gene_pheno_priors = sparse.csc_matrix(
            (priors, (row, col)),
            shape=(len(runtime.genes), len(runtime.phenos)),
        )

    runtime.anchor_gene_mask = None
    if anchor_genes is not None:
        runtime.anchor_gene_mask = np.array([x in anchor_genes for x in runtime.genes])
        if np.sum(runtime.anchor_gene_mask) == 0:
            bail_fn("Couldn't find any match for %s" % list(anchor_genes))

    if log_fn is not None:
        num_pairs = (
            len(runtime.gene_pheno_Y.nonzero()[0])
            if runtime.gene_pheno_Y is not None
            else 0
        )
        log_fn("Read values for %d gene, pheno pairs" % num_pairs)

    runtime.anchor_pheno_mask = None
    if anchor_phenos is not None:
        runtime.anchor_pheno_mask = np.array([x in anchor_phenos for x in runtime.phenos])
        if np.sum(runtime.anchor_pheno_mask) == 0:
            bail_fn("Couldn't find any match for %s" % list(anchor_phenos))


def load_and_apply_gene_phewas_bfs_to_runtime(
    runtime,
    gene_phewas_bfs_in,
    *,
    gene_phewas_bfs_id_col=None,
    gene_phewas_bfs_pheno_col=None,
    anchor_genes=None,
    anchor_phenos=None,
    gene_phewas_bfs_log_bf_col=None,
    gene_phewas_bfs_combined_col=None,
    gene_phewas_bfs_prior_col=None,
    phewas_gene_to_x_gene=None,
    min_value=None,
    max_num_entries_at_once=None,
    open_text_fn=None,
    get_col_fn=None,
    construct_map_to_ind_fn=None,
    warn_fn=None,
    bail_fn=None,
    log_fn=None,
):
    if open_text_fn is None:
        open_text_fn = lambda path: open(path)
    if get_col_fn is None:
        get_col_fn = resolve_column_index
    if bail_fn is None:
        bail_fn = _default_bail
    if warn_fn is None:
        warn_fn = lambda _m: None

    parsed_phewas = parse_gene_phewas_bfs_file(
        gene_phewas_bfs_in,
        gene_phewas_bfs_id_col=gene_phewas_bfs_id_col,
        gene_phewas_bfs_pheno_col=gene_phewas_bfs_pheno_col,
        gene_phewas_bfs_log_bf_col=gene_phewas_bfs_log_bf_col,
        gene_phewas_bfs_combined_col=gene_phewas_bfs_combined_col,
        gene_phewas_bfs_prior_col=gene_phewas_bfs_prior_col,
        min_value=min_value,
        max_num_entries_at_once=max_num_entries_at_once,
        existing_phenos=runtime.phenos,
        existing_pheno_to_ind=runtime.pheno_to_ind,
        gene_to_ind=runtime.gene_to_ind,
        gene_label_map=runtime.gene_label_map,
        phewas_gene_to_x_gene=phewas_gene_to_x_gene,
        open_text_fn=open_text_fn,
        get_col_fn=get_col_fn,
        bail_fn=bail_fn,
        warn_fn=warn_fn,
    )
    apply_parsed_gene_phewas_bfs_to_runtime(
        runtime,
        parsed_phewas,
        anchor_genes=anchor_genes,
        anchor_phenos=anchor_phenos,
        construct_map_to_ind_fn=construct_map_to_ind_fn,
        bail_fn=bail_fn,
        log_fn=log_fn,
    )
    return parsed_phewas


def apply_parsed_gene_set_statistics_to_runtime(
    runtime,
    parsed_stats,
    *,
    return_only_ids=False,
    stats_beta_col=None,
    warn_fn=None,
    bail_fn=None,
    log_fn=None,
):
    if warn_fn is None:
        warn_fn = lambda _m: None
    if bail_fn is None:
        bail_fn = _default_bail
    if log_fn is None:
        log_fn = lambda _m: None

    subset_mask = None
    read_ids = set()
    need_to_take_log = parsed_stats.need_to_take_log
    has_beta_tilde = parsed_stats.has_beta_tilde
    has_p_or_se = parsed_stats.has_p_or_se
    has_beta = parsed_stats.has_beta
    has_beta_uncorrected = parsed_stats.has_beta_uncorrected
    records = parsed_stats.records

    if not return_only_ids:
        if runtime.gene_sets is not None:
            if has_beta_tilde:
                runtime.beta_tildes = np.zeros(len(runtime.gene_sets))
            if has_p_or_se:
                runtime.p_values = np.zeros(len(runtime.gene_sets))
                runtime.ses = np.zeros(len(runtime.gene_sets))
                runtime.z_scores = np.zeros(len(runtime.gene_sets))
            if has_beta:
                runtime.betas = np.zeros(len(runtime.gene_sets))
            if has_beta_uncorrected:
                runtime.betas_uncorrected = np.zeros(len(runtime.gene_sets))
            subset_mask = np.array([False] * len(runtime.gene_sets))
        else:
            if has_beta_tilde:
                runtime.beta_tildes = []
            if has_p_or_se:
                runtime.p_values = []
                runtime.ses = []
                runtime.z_scores = []
            if has_beta:
                runtime.betas = []
            if has_beta_uncorrected:
                runtime.betas_uncorrected = []

    gene_sets = []
    gene_set_to_ind = {}
    ignored = 0

    for gene_set, values in records.items():
        beta_tilde, p, se, z, beta, beta_uncorrected = values
        if runtime.gene_sets is not None:
            if gene_set not in runtime.gene_set_to_ind:
                ignored += 1
                continue
            if return_only_ids:
                read_ids.add(gene_set)
                continue
            gene_set_ind = runtime.gene_set_to_ind[gene_set]
            if gene_set_ind is not None:
                if has_beta_tilde:
                    runtime.beta_tildes[gene_set_ind] = beta_tilde * runtime.scale_factors[gene_set_ind]
                if has_p_or_se:
                    runtime.p_values[gene_set_ind] = p
                    runtime.z_scores[gene_set_ind] = z
                    runtime.ses[gene_set_ind] = se * runtime.scale_factors[gene_set_ind]
                if has_beta:
                    runtime.betas[gene_set_ind] = beta * runtime.scale_factors[gene_set_ind]
                if has_beta_uncorrected:
                    runtime.betas_uncorrected[gene_set_ind] = (
                        beta_uncorrected * runtime.scale_factors[gene_set_ind]
                    )
                subset_mask[gene_set_ind] = True
        else:
            if return_only_ids:
                read_ids.add(gene_set)
                continue
            bail_fn(
                "Not yet implemented this -- no way to convert external beta tilde units reading in into internal units"
            )
            if has_beta_tilde:
                runtime.beta_tildes.append(beta_tilde)
            if has_p_or_se:
                runtime.p_values.append(p)
                runtime.z_scores.append(z)
                runtime.ses.append(se)
            if has_beta:
                runtime.betas.append(beta)
            if has_beta_uncorrected:
                runtime.betas_uncorrected.append(beta_uncorrected)
            gene_set_to_ind[gene_set] = len(gene_sets)
            gene_sets.append(gene_set)

    log_fn("Done reading --stats-in-file")

    if return_only_ids:
        return read_ids

    if runtime.gene_sets is not None:
        log_fn("Subsetting matrices")
        if ignored > 0:
            warn_fn("Ignored %s values from --stats-in file because absent from previously loaded files" % ignored)
        if np.sum(subset_mask) != len(subset_mask):
            warn_fn(
                "Excluding %s values from previously loaded files because absent from --stats-in file"
                % (len(subset_mask) - np.sum(subset_mask))
            )
            if runtime.beta_tildes is not None and not need_to_take_log and np.sum(runtime.beta_tildes < 0) == 0:
                warn_fn(
                    "All beta_tilde values are positive. Are you sure that the values in column %s are not exp(beta_tilde)?"
                    % stats_beta_col
                )
            runtime.subset_gene_sets(subset_mask, keep_missing=True)
        log_fn("Done subsetting matrices")
    else:
        runtime.X_orig_missing_gene_sets = None
        runtime.mean_shifts_missing = None
        runtime.scale_factors_missing = None
        runtime.is_dense_gene_set_missing = None
        runtime.ps_missing = None
        runtime.sigma2s_missing = None

        runtime.beta_tildes_missing = None
        runtime.p_values_missing = None
        runtime.ses_missing = None
        runtime.z_scores_missing = None

        runtime.beta_tildes = np.array(runtime.beta_tildes)
        runtime.p_values = np.array(runtime.p_values)
        runtime.z_scores = np.array(runtime.z_scores)
        runtime.ses = np.array(runtime.ses)
        runtime.gene_sets = gene_sets
        runtime.gene_set_to_ind = gene_set_to_ind

        if has_beta:
            runtime.betas = np.array(runtime.betas)
        if has_beta_uncorrected:
            runtime.betas_uncorrected = np.array(runtime.betas_uncorrected)

        runtime.total_qc_metrics_missing = None
        runtime.mean_qc_metrics_missing = None

    runtime._set_X(runtime.X_orig, runtime.genes, runtime.gene_sets, skip_N=True)
    return None


def load_and_apply_gene_set_statistics_to_runtime(
    runtime,
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
    open_text_fn=None,
    get_col_fn=None,
    warn_fn=None,
    bail_fn=None,
    parse_log_fn=None,
    apply_log_fn=None,
):
    if open_text_fn is None:
        open_text_fn = lambda path: open(path)
    if get_col_fn is None:
        get_col_fn = resolve_column_index
    if bail_fn is None:
        bail_fn = _default_bail
    if warn_fn is None:
        warn_fn = lambda _m: None

    parsed_stats = parse_gene_set_statistics_file(
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
        open_text_fn=open_text_fn,
        get_col_fn=get_col_fn,
        log_fn=parse_log_fn,
        warn_fn=warn_fn,
        bail_fn=bail_fn,
    )
    return apply_parsed_gene_set_statistics_to_runtime(
        runtime,
        parsed_stats,
        return_only_ids=return_only_ids,
        stats_beta_col=stats_beta_col,
        warn_fn=warn_fn,
        bail_fn=bail_fn,
        log_fn=apply_log_fn,
    )


def load_and_apply_gene_set_phewas_statistics_to_runtime(
    runtime,
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
    open_text_fn=None,
    get_col_fn=None,
    construct_map_to_ind_fn=None,
    warn_fn=None,
    bail_fn=None,
    log_fn=None,
):
    if open_text_fn is None:
        open_text_fn = lambda path: open(path)
    if get_col_fn is None:
        get_col_fn = resolve_column_index
    if construct_map_to_ind_fn is None:
        construct_map_to_ind_fn = construct_map_to_ind
    if warn_fn is None:
        warn_fn = lambda _m: None
    if bail_fn is None:
        bail_fn = _default_bail
    if log_fn is None:
        log_fn = lambda _m: None

    if stats_in is None:
        bail_fn("Require --gene-set-stats-in or --gene-set-phewas-stats-in for this operation")

    log_fn("Reading --gene-set-phewas-stats-in file %s" % stats_in)

    for delim in [None, '\t']:
        subset_mask = None
        read_ids = set()
        success = True
        with open_text_fn(stats_in) as stats_fh:
            header_cols = stats_fh.readline().strip('\n').split(delim)
            if len(header_cols) == 1:
                success = False
                continue
            id_col = get_col_fn(stats_id_col, header_cols)
            pheno_col = get_col_fn(stats_pheno_col, header_cols)

            beta_col = None
            if stats_beta_col is not None:
                beta_col = get_col_fn(stats_beta_col, header_cols, True)
            else:
                beta_col = get_col_fn("beta", header_cols, False)

            beta_uncorrected_col = None
            if stats_beta_uncorrected_col is not None:
                beta_uncorrected_col = get_col_fn(stats_beta_uncorrected_col, header_cols, True)
            else:
                beta_uncorrected_col = get_col_fn("beta_uncorrected", header_cols, False)

            if beta_col is None and beta_uncorrected_col is None:
                bail_fn("Require at least beta or beta_uncorrected to read from --gene-set-stats-in")

            if runtime.gene_sets is not None:
                subset_mask = np.array([False] * len(runtime.gene_sets))

            gene_sets = []
            gene_set_to_ind = {}
            phenos = []
            pheno_to_ind = {}
            if max_num_entries_at_once is None:
                max_num_entries_at_once = 200 * 10000

            betas = []
            betas_uncorrected = []
            row = []
            col = []
            betas_chunks = []
            betas_uncorrected_chunks = []
            row_chunks = []
            col_chunks = []

            def __flush_chunks():
                if len(row) == 0:
                    return
                row_chunks.append(np.array(row, dtype=np.int32))
                col_chunks.append(np.array(col, dtype=np.int32))
                betas_chunks.append(np.array(betas, dtype=np.float64))
                betas_uncorrected_chunks.append(np.array(betas_uncorrected, dtype=np.float64))
                row[:] = []
                col[:] = []
                betas[:] = []
                betas_uncorrected[:] = []

            for line in stats_fh:
                beta = None
                beta_uncorrected = None
                cols = line.strip('\n').split(delim)
                if len(cols) != len(header_cols):
                    success = False
                    continue

                if (
                    id_col > len(cols)
                    or pheno_col > len(cols)
                    or (beta_col is not None and beta_col > len(cols))
                    or (beta_uncorrected_col is not None and beta_uncorrected_col > len(cols))
                ):
                    warn_fn("Skipping due to too few columns in line: %s" % line)
                    continue

                gene_set = cols[id_col]
                pheno = cols[pheno_col]
                if phenos_to_match is not None and pheno not in phenos_to_match:
                    continue

                if beta_col is not None:
                    try:
                        beta = float(cols[beta_col])
                        if min_gene_set_beta is not None and beta < min_gene_set_beta:
                            continue
                    except ValueError:
                        if cols[beta_col] != "NA":
                            warn_fn("Skipping unconvertible beta value %s for gene_set %s" % (cols[beta_col], gene_set))
                        continue

                if beta_uncorrected_col is not None:
                    try:
                        beta_uncorrected = float(cols[beta_uncorrected_col])
                        if min_gene_set_beta_uncorrected is not None and beta_uncorrected < min_gene_set_beta_uncorrected:
                            continue
                    except ValueError:
                        if cols[beta_uncorrected_col] != "NA":
                            warn_fn(
                                "Skipping unconvertible beta_uncorrected value %s for gene_set %s"
                                % (cols[beta_uncorrected_col], gene_set)
                            )
                        continue

                if pheno in pheno_to_ind:
                    pheno_ind = pheno_to_ind[pheno]
                else:
                    pheno_ind = len(phenos)
                    pheno_to_ind[pheno] = pheno_ind
                    phenos.append(pheno)

                gene_set_ind = None
                if runtime.gene_sets is not None:
                    if gene_set not in runtime.gene_set_to_ind:
                        continue
                    gene_set_ind = runtime.gene_set_to_ind[gene_set]
                    if gene_set_ind is not None:
                        subset_mask[gene_set_ind] = True
                else:
                    gene_set_to_ind[gene_set] = len(gene_sets)
                    gene_sets.append(gene_set)

                if return_only_ids:
                    read_ids.add(gene_set)
                    continue

                if gene_set_ind is not None:
                    col.append(gene_set_ind)
                    row.append(pheno_ind)
                    if beta_uncorrected is not None:
                        betas_uncorrected.append(beta_uncorrected)
                    else:
                        betas_uncorrected.append(beta)
                    if beta is not None:
                        betas.append(beta)
                    else:
                        betas.append(beta_uncorrected)
                    if len(row) >= max_num_entries_at_once:
                        __flush_chunks()

            __flush_chunks()
            log_fn("Done reading --stats-in-file")
            if success:
                break

    if not success:
        bail_fn("Error: number of columns in header did not match number of columns in lines after header")

    if return_only_ids:
        return read_ids

    if update_X:
        if runtime.gene_sets is not None:
            log_fn("Subsetting matrices")
            if np.sum(subset_mask) != len(subset_mask):
                warn_fn(
                    "Excluding %s values from previously loaded files because absent from --stats-in file"
                    % (len(subset_mask) - np.sum(subset_mask))
                )
                runtime.subset_gene_sets(subset_mask, keep_missing=True)
            log_fn("Done subsetting matrices")

        runtime._set_X(runtime.X_orig, runtime.genes, runtime.gene_sets, skip_N=True)

    if runtime.phenos is not None:
        bail_fn("Bug in code: cannot call this function if phenos have already been read")

    runtime.phenos = phenos
    runtime.pheno_to_ind = construct_map_to_ind_fn(phenos)

    if len(row_chunks) > 0:
        row = np.concatenate(row_chunks)
        col = np.concatenate(col_chunks)
        betas = np.concatenate(betas_chunks)
        betas_uncorrected = np.concatenate(betas_uncorrected_chunks)
    else:
        row = np.array([], dtype=np.int32)
        col = np.array([], dtype=np.int32)
        betas = np.array([], dtype=np.float64)
        betas_uncorrected = np.array([], dtype=np.float64)

    if len(row) > 0:
        key = row.astype(np.int64) * int(len(runtime.gene_sets)) + col.astype(np.int64)
        _, unique_indices = np.unique(key, return_index=True)
    else:
        unique_indices = np.array([], dtype=np.int64)

    if len(unique_indices) < len(row):
        warn_fn("Found %d duplicate values; ignoring duplicates" % (len(row) - len(unique_indices)))

    betas = betas[unique_indices]
    betas_uncorrected = betas_uncorrected[unique_indices]
    row = row[unique_indices]
    col = col[unique_indices]

    runtime.X_phewas_beta = sparse.csc_matrix(
        (betas, (row, col)),
        shape=(len(runtime.phenos), len(runtime.gene_sets)),
    )
    runtime.X_phewas_beta_uncorrected = sparse.csc_matrix(
        (betas_uncorrected, (row, col)),
        shape=(len(runtime.phenos), len(runtime.gene_sets)),
    )

    return None


def read_gene_phewas_stats(path, *, bail_fn=None):
    return read_tsv(path, key_column="Gene", required_columns=["Gene"], bail_fn=bail_fn)


def read_gene_set_phewas_stats(path, *, bail_fn=None):
    return read_tsv(path, key_column="Gene_Set", required_columns=["Gene_Set"], bail_fn=bail_fn)


def read_factor_phewas_stats(path, *, bail_fn=None):
    return read_tsv(path, required_columns=[], bail_fn=bail_fn)


def remove_tag_from_input(x_in, tag_separator=":"):
    tag = None
    if tag_separator in x_in:
        tag_index = x_in.index(tag_separator)
        tag = x_in[:tag_index]
        x_in = x_in[tag_index + 1 :]
        if len(tag) == 0:
            tag = None
    return (x_in, tag)


def add_tag_to_input(x_in, tag, tag_separator=":"):
    if tag is None:
        return x_in
    return tag_separator.join([tag, x_in])


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

    if not (filter_gene_set_p is not None and (correct_betas_mean or correct_betas_var) and runtime.beta_tildes is not None):
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
                assert(runtime.ps is not None)
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

    # `increase_filter_gene_set_p` semantics are "minimum kept fraction".
    # Prefilter logic should relax p-thresholds when too few sets are kept; this
    # post-read hook must not further tighten filtering.
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

    if runtime.total_qc_metrics is not None and filter_gene_set_metric_z:
        filter_mask = np.abs(runtime.mean_qc_metrics) < filter_gene_set_metric_z
        filter_ignore = ~filter_mask
        log_fn("Ignoring %d gene sets due to QC metric filters (kept %d)" % (np.sum(filter_ignore), np.sum(filter_mask)))
        runtime.subset_gene_sets(filter_mask, keep_missing=False, ignore_missing=True, skip_V=True)


def _normalize_input_specs(input_specs):
    if input_specs is None:
        return ([], [])
    if type(input_specs) == str:
        return ([input_specs], [input_specs])
    if type(input_specs) == list:
        return (input_specs, copy.copy(input_specs))
    return ([], [])


def build_xin_to_p_noninf_index_map(
    X_in,
    X_list,
    Xd_in,
    Xd_list,
    p_noninf_values,
    *,
    warn_fn=None,
    bail_fn=None,
):
    if p_noninf_values is None:
        return None

    if warn_fn is None:
        warn_fn = lambda _msg: None
    if bail_fn is None:
        bail_fn = _default_bail

    p_values = p_noninf_values if isinstance(p_noninf_values, list) else [p_noninf_values]
    ordered_specs = []
    ordered_specs += _normalize_input_specs(X_in)[0]
    ordered_specs += _normalize_input_specs(X_list)[0]
    ordered_specs += _normalize_input_specs(Xd_in)[0]
    ordered_specs += _normalize_input_specs(Xd_list)[0]

    xin_to_p_noninf_ind = {}
    if len(p_values) <= 1:
        for spec in ordered_specs:
            if spec in xin_to_p_noninf_ind:
                warn_fn(
                    "You are passing the same file (%s) for two --X-* inputs; are you sure this is what you want to do?"
                    % spec
                )
            xin_to_p_noninf_ind[spec] = 0
        return xin_to_p_noninf_ind

    for index, spec in enumerate(ordered_specs):
        if spec in xin_to_p_noninf_ind:
            warn_fn(
                "You are passing the same file (%s) for two --X-* inputs; are you sure this is what you want to do?"
                % spec
            )
        xin_to_p_noninf_ind[spec] = index

    if len(p_values) != len(ordered_specs):
        bail_fn(
            "Error: if you pass in more than one --p-noninf, you need to have the same number of values as --X-* inputs"
        )
    return xin_to_p_noninf_ind


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


def infer_columns_from_table_file(filename, open_text_fn, *, log_fn=None, bail_fn=None):
    if bail_fn is None:
        bail_fn = _default_bail
    if log_fn is None:
        log_fn = lambda _msg: None

    log_fn("Trying to determine columns from headers and data for %s..." % filename)
    header = None
    with open_text_fn(filename) as fh:
        header = fh.readline().strip("\n")
        orig_header_cols = header.split()

        first_line = fh.readline().strip("\n")
        first_cols = first_line.split()

        if len(orig_header_cols) > len(first_cols):
            orig_header_cols = header.split("\t")

        header_cols = [x.strip('"').strip("'").strip("\n") for x in orig_header_cols]

        def __get_possible_from_headers(_header_cols, possible_headers1, possible_headers2=None):
            possible = np.full(len(_header_cols), False)
            possible_inds = [i for i in range(len(_header_cols)) if _header_cols[i].lower().strip('_"') in possible_headers1]
            if len(possible_inds) == 0 and possible_headers2 is not None:
                possible_inds = [i for i in range(len(_header_cols)) if _header_cols[i].lower() in possible_headers2]
            possible[possible_inds] = True
            return possible

        possible_gene_id_headers = set(["gene", "id"])
        possible_var_id_headers = set(["var", "id", "rs", "varid"])
        possible_chrom_headers = set(["chr", "chrom", "chromosome", "#chrom"])
        possible_pos_headers = set(["pos", "bp", "position", "base_pair_location"])
        possible_locus_headers = set(["variant"])
        possible_p_headers = set(["p-val", "p_val", "pval", "p.value", "p-value", "p_value"])
        possible_p_headers2 = set(["p"])
        possible_beta_headers = set(["beta", "effect"])
        possible_se_headers = set(["se", "std", "stderr", "standard_error"])
        possible_freq_headers = set(["maf", "freq"])
        possible_freq_headers2 = set(["af", "effect_allele_frequency"])
        possible_n_headers = set(["sample", "neff", "TotalSampleSize", "n_samples"])
        possible_n_headers2 = set(["n"])

        possible_gene_id_cols = __get_possible_from_headers(header_cols, possible_gene_id_headers)
        possible_var_id_cols = __get_possible_from_headers(header_cols, possible_var_id_headers)
        possible_chrom_cols = __get_possible_from_headers(header_cols, possible_chrom_headers)
        possible_locus_cols = __get_possible_from_headers(header_cols, possible_locus_headers)
        possible_pos_cols = __get_possible_from_headers(header_cols, possible_pos_headers)
        possible_p_cols = __get_possible_from_headers(header_cols, possible_p_headers, possible_p_headers2)
        possible_beta_cols = __get_possible_from_headers(header_cols, possible_beta_headers)
        possible_se_cols = __get_possible_from_headers(header_cols, possible_se_headers)
        possible_freq_cols = __get_possible_from_headers(header_cols, possible_freq_headers, possible_freq_headers2)
        possible_n_cols = __get_possible_from_headers(header_cols, possible_n_headers, possible_n_headers2)

        missing_vals = set(["", ".", "-", "na"])
        num_read = 0
        max_to_read = 1000

        for line in fh:
            cols = line.strip("\n").split()
            seen_non_missing = False
            if len(cols) != len(header_cols):
                cols = line.strip("\n").split("\t")

            if len(cols) != len(header_cols):
                bail_fn("Error: couldn't parse line into same number of columns as header (%d vs. %d)" % (len(cols), len(header_cols)))

            for i in range(len(cols)):
                token = cols[i].lower()

                if token.lower() in missing_vals:
                    continue

                seen_non_missing = True

                if possible_gene_id_cols[i]:
                    try:
                        val = float(cols[i])
                        if not int(val) == val:
                            possible_gene_id_cols[i] = False
                    except ValueError:
                        pass
                if possible_var_id_cols[i]:
                    if len(token) < 4:
                        possible_var_id_cols[i] = False

                    if "chr" in token or ":" in token or "rs" in token or "_" in token or "-" in token or "var" in token:
                        pass
                    else:
                        possible_var_id_cols[i] = False
                if possible_chrom_cols[i]:
                    if "chr" in token or "x" in token or "y" in token or "m" in token:
                        pass
                    else:
                        try:
                            val = int(cols[i])
                            if val < 1 or val > 26:
                                possible_chrom_cols[i] = False
                        except ValueError:
                            possible_chrom_cols[i] = False
                if possible_locus_cols[i]:
                    if "chr" in token or "x" in token or "y" in token or "m" in token:
                        pass
                    else:
                        try:
                            locus = None
                            for delim in [":", "_"]:
                                if delim in cols[i]:
                                    locus = cols[i].split(delim)
                            if locus is not None and len(locus) >= 2:
                                chrom = int(locus[0])
                                _pos = int(locus[1])
                                if chrom < 1 or chrom > 26:
                                    possible_locus_cols[i] = False
                        except ValueError:
                            possible_locus_cols[i] = False
                if possible_pos_cols[i]:
                    try:
                        if len(token) < 3:
                            possible_pos_cols[i] = False
                        val = float(cols[i])
                        if not int(val) == val:
                            possible_pos_cols[i] = False
                    except ValueError:
                        possible_pos_cols[i] = False

                if possible_p_cols[i]:
                    try:
                        val = float(cols[i])
                        if val > 1 or val < 0:
                            possible_p_cols[i] = False
                    except ValueError:
                        possible_p_cols[i] = False
                if possible_beta_cols[i]:
                    try:
                        _val = float(cols[i])
                    except ValueError:
                        possible_beta_cols[i] = False
                if possible_se_cols[i]:
                    try:
                        val = float(cols[i])
                        if val < 0:
                            possible_se_cols[i] = False
                    except ValueError:
                        possible_se_cols[i] = False
                if possible_freq_cols[i]:
                    try:
                        val = float(cols[i])
                        if val > 1 or val < 0:
                            possible_freq_cols[i] = False
                    except ValueError:
                        possible_freq_cols[i] = False
                if possible_n_cols[i]:
                    if len(token) < 3:
                        possible_n_cols[i] = False
                    else:
                        try:
                            val = float(cols[i])
                            if val < 0:
                                possible_n_cols[i] = False
                        except ValueError:
                            possible_n_cols[i] = False
            if seen_non_missing:
                num_read += 1
                if num_read >= max_to_read:
                    break

    possible_beta_cols[possible_p_cols] = False
    possible_beta_cols[possible_se_cols] = False
    possible_beta_cols[possible_pos_cols] = False

    total_possible = (
        possible_gene_id_cols.astype(int)
        + possible_var_id_cols.astype(int)
        + possible_chrom_cols.astype(int)
        + possible_pos_cols.astype(int)
        + possible_p_cols.astype(int)
        + possible_beta_cols.astype(int)
        + possible_se_cols.astype(int)
        + possible_freq_cols.astype(int)
        + possible_n_cols.astype(int)
    )
    for possible_cols in [
        possible_gene_id_cols,
        possible_var_id_cols,
        possible_chrom_cols,
        possible_pos_cols,
        possible_p_cols,
        possible_beta_cols,
        possible_se_cols,
        possible_freq_cols,
        possible_n_cols,
    ]:
        possible_cols[total_possible > 1] = False

    orig_header_cols = np.array(orig_header_cols)
    return (
        orig_header_cols[possible_gene_id_cols],
        orig_header_cols[possible_var_id_cols],
        orig_header_cols[possible_chrom_cols],
        orig_header_cols[possible_pos_cols],
        orig_header_cols[possible_locus_cols],
        orig_header_cols[possible_p_cols],
        orig_header_cols[possible_beta_cols],
        orig_header_cols[possible_se_cols],
        orig_header_cols[possible_freq_cols],
        orig_header_cols[possible_n_cols],
        header,
    )


def needs_gwas_column_detection(
    gwas_pos_col,
    gwas_chrom_col,
    gwas_locus_col,
    gwas_p_col,
    gwas_beta_col,
    gwas_se_col,
    gwas_n_col,
    gwas_n,
):
    if (gwas_pos_col is None or gwas_chrom_col is None) and gwas_locus_col is None:
        return True

    has_se = gwas_se_col is not None or gwas_n_col is not None or gwas_n is not None
    if (gwas_p_col is not None and gwas_beta_col is not None) or (gwas_p_col is not None and has_se) or (gwas_beta_col is not None and has_se):
        return False
    return True


def autodetect_gwas_columns(
    gwas_in,
    gwas_pos_col,
    gwas_chrom_col,
    gwas_locus_col,
    gwas_p_col,
    gwas_beta_col,
    gwas_se_col,
    gwas_freq_col,
    gwas_n_col,
    gwas_n,
    *,
    infer_columns_fn,
    log_fn=None,
    bail_fn=None,
    debug_just_check_header=False,
):
    if bail_fn is None:
        bail_fn = _default_bail
    if log_fn is None:
        log_fn = lambda _msg: None

    (
        _possible_gene_id_cols,
        _possible_var_id_cols,
        possible_chrom_cols,
        possible_pos_cols,
        possible_locus_cols,
        possible_p_cols,
        possible_beta_cols,
        possible_se_cols,
        possible_freq_cols,
        possible_n_cols,
        header,
    ) = infer_columns_fn(gwas_in)

    if gwas_pos_col is None:
        if len(possible_pos_cols) == 1:
            gwas_pos_col = possible_pos_cols[0]
            log_fn("Using %s for position column; change with --gwas-pos-col if incorrect" % gwas_pos_col)
        else:
            log_fn("Could not determine position column from header %s; specify with --gwas-pos-col" % header)
    if gwas_chrom_col is None:
        if len(possible_chrom_cols) == 1:
            gwas_chrom_col = possible_chrom_cols[0]
            log_fn("Using %s for chrom column; change with --gwas-chrom-col if incorrect" % gwas_chrom_col)
        else:
            log_fn("Could not determine chrom column from header %s; specify with --gwas-chrom-col" % header)
    if (gwas_pos_col is None or gwas_chrom_col is None) and gwas_locus_col is None:
        if len(possible_locus_cols) == 1:
            gwas_locus_col = possible_locus_cols[0]
            log_fn("Using %s for locus column; change with --gwas-locus-col if incorrect" % gwas_locus_col)
        else:
            bail_fn("Could not determine chrom and pos columns from header %s; specify with --gwas-chrom-col and --gwas-pos-col or with --gwas-locus-col" % header)

    if gwas_p_col is None:
        if len(possible_p_cols) == 1:
            gwas_p_col = possible_p_cols[0]
            log_fn("Using %s for p column; change with --gwas-p-col if incorrect" % gwas_p_col)
        else:
            log_fn("Could not determine p column from header %s; if desired specify with --gwas-p-col" % header)
    if gwas_se_col is None:
        if len(possible_se_cols) == 1:
            gwas_se_col = possible_se_cols[0]
            log_fn("Using %s for se column; change with --gwas-se-col if incorrect" % gwas_se_col)
        else:
            log_fn("Could not determine se column from header %s; if desired specify with --gwas-se-col" % header)
    if gwas_beta_col is None:
        if len(possible_beta_cols) == 1:
            gwas_beta_col = possible_beta_cols[0]
            log_fn("Using %s for beta column; change with --gwas-beta-col if incorrect" % gwas_beta_col)
        else:
            log_fn("Could not determine beta column from header %s; if desired specify with --gwas-beta-col" % header)

    if gwas_n_col is None:
        if len(possible_n_cols) == 1:
            gwas_n_col = possible_n_cols[0]
            log_fn("Using %s for N column; change with --gwas-n-col if incorrect" % gwas_n_col)
        else:
            log_fn("Could not determine N column from header %s; if desired specify with --gwas-n-col" % header)

    if gwas_freq_col is None:
        if len(possible_freq_cols) == 1:
            gwas_freq_col = possible_freq_cols[0]
            log_fn("Using %s for freq column; change with --gwas-freq-col if incorrect" % gwas_freq_col)

    has_se = gwas_se_col is not None
    has_n = gwas_n_col is not None or gwas_n is not None
    if (gwas_p_col is not None and gwas_beta_col is not None) or (gwas_p_col is not None and (has_se or has_n)) or (gwas_beta_col is not None and has_se):
        pass
    else:
        bail_fn("Require information about p-value and se or N or beta, or beta and se; specify with --gwas-p-col, --gwas-beta-col, --gwas-se-col, and --gwas-n-col")

    if debug_just_check_header:
        bail_fn("Done checking headers")

    return (
        gwas_pos_col,
        gwas_chrom_col,
        gwas_locus_col,
        gwas_p_col,
        gwas_beta_col,
        gwas_se_col,
        gwas_freq_col,
        gwas_n_col,
    )


def complete_p_beta_se(p, beta, se, *, warn_fn=None):
    if warn_fn is None:
        warn_fn = lambda _message: None

    p_none_mask = np.logical_or(p == None, np.isnan(p))
    beta_none_mask = np.logical_or(beta == None, np.isnan(beta))
    se_none_mask = np.logical_or(se == None, np.isnan(se))

    se_zero_mask = np.logical_and(~se_none_mask, se == 0)
    se_zero_beta_non_zero_mask = np.logical_and(se_zero_mask, np.logical_and(~beta_none_mask, beta != 0))

    if np.sum(se_zero_beta_non_zero_mask) != 0:
        warn_fn("%d variants had zero SEs; setting these to beta zero and se 1" % (np.sum(se_zero_beta_non_zero_mask)))
        beta[se_zero_beta_non_zero_mask] = 0
    se[se_zero_mask] = 1

    bad_mask = np.logical_and(np.logical_and(p_none_mask, beta_none_mask), se_none_mask)
    if np.sum(bad_mask) > 0:
        warn_fn("Couldn't infer p/beta/se at %d positions; setting these to beta zero and se 1" % (np.sum(bad_mask)))
        p[bad_mask] = 1
        beta[bad_mask] = 0
        se[bad_mask] = 1
        p_none_mask[bad_mask] = False
        beta_none_mask[bad_mask] = False
        se_none_mask[bad_mask] = False

    if np.sum(p_none_mask) > 0:
        p[p_none_mask] = 2 * scipy.stats.norm.pdf(-np.abs(beta[p_none_mask] / se[p_none_mask]))
    if np.sum(beta_none_mask) > 0:
        z = np.abs(scipy.stats.norm.ppf(np.array(p[beta_none_mask] / 2)))
        beta[beta_none_mask] = z * se[beta_none_mask]
    if np.sum(se_none_mask) > 0:
        z = np.abs(scipy.stats.norm.ppf(np.array(p[se_none_mask] / 2)))
        z[z == 0] = 1
        se[se_none_mask] = np.abs(beta[se_none_mask] / z)
    return (p, beta, se)


def write_phewas_gene_set_statistics(runtime, output_file, max_no_write_gene_set_beta=None, max_no_write_gene_set_beta_uncorrected=None, basic=False, *, open_text_fn=None, log_fn=None, info_level=0):
    if open_text_fn is None:
        open_text_fn = open
    if log_fn is None:
        log_fn = lambda _msg, _lvl=0: None

    log_fn("Writing phewas gene set stats to %s" % output_file, info_level)
    if runtime.p_values_phewas is None:
        log_fn("No stats available; skipping", info_level)
        return
    with open_text_fn(output_file, 'w') as output_fh:
        if runtime.gene_sets is None:
            return
        header = "Gene_Set"
        if runtime.gene_set_labels is not None:
            header = "%s\t%s" % (header, "label")
        if runtime.phenos is not None:
            header = "%s\t%s" % (header, "trait")
        if runtime.X_orig is not None:
            col_sums = runtime.get_col_sums(runtime.X_orig)
            header = "%s\t%s" % (header, "N")
            header = "%s\t%s" % (header, "scale")
        if runtime.beta_tildes_phewas is not None:
            header = "%s\t%s\t%s\t%s\t%s\t%s" % (header, "beta_tilde", "beta_tilde_internal", "P", "Z", "SE")
        if runtime.betas_phewas is not None:
            header = "%s\t%s\t%s" % (header, "beta", "beta_internal")
        if runtime.betas_uncorrected_phewas is not None and not basic:
            header = "%s\t%s" % (header, "beta_uncorrected")            

        output_fh.write("%s\n" % header)

        for p in range(len(runtime.phenos)):

            ordered_i = range(len(runtime.gene_sets))
            if runtime.betas_uncorrected_phewas is not None:
                ordered_i = sorted(ordered_i, key=lambda k: -runtime.betas_uncorrected_phewas[p,k] / runtime.scale_factors[k])
            elif runtime.p_values_phewas is not None:
                ordered_i = sorted(ordered_i, key=lambda k: runtime.p_values_phewas[p,k])

            for i in ordered_i:

                if max_no_write_gene_set_beta is not None and runtime.betas_phewas is not None and np.abs(runtime.betas_phewas[p,i] / runtime.scale_factors[i]) <= max_no_write_gene_set_beta:
                    continue

                if max_no_write_gene_set_beta_uncorrected is not None and runtime.betas_uncorrected_phewas is not None and np.abs(runtime.betas_uncorrected_phewas[p,i] / runtime.scale_factors[i]) <= max_no_write_gene_set_beta_uncorrected:
                    continue

                line = runtime.gene_sets[i]
                if runtime.gene_set_labels is not None:
                    line = "%s\t%s" % (line, runtime.gene_set_labels[i])
                if runtime.phenos is not None:
                    line = "%s\t%s" % (line, runtime.phenos[p])
                if runtime.X_orig is not None:
                    line = "%s\t%d" % (line, col_sums[i])
                    line = "%s\t%.3g" % (line, runtime.scale_factors[i])

                if runtime.beta_tildes_phewas is not None:
                    line = "%s\t%.3g\t%.3g\t%.3g\t%.3g\t%.3g" % (line, runtime.beta_tildes_phewas[p,i] / runtime.scale_factors[i], runtime.beta_tildes_phewas[p,i], runtime.p_values_phewas[p,i], runtime.z_scores_phewas[p,i], runtime.ses_phewas[p,i] / runtime.scale_factors[i])
                if runtime.betas_phewas is not None:
                    line = "%s\t%.3g\t%.3g" % (line, runtime.betas_phewas[p,i] / runtime.scale_factors[i], runtime.betas_phewas[p,i])
                if runtime.betas_uncorrected_phewas is not None and not basic:
                    line = "%s\t%.3g" % (line, runtime.betas_uncorrected_phewas[p,i] / runtime.scale_factors[i])            
                output_fh.write("%s\n" % line)


def write_gene_statistics(runtime, output_file, *, open_text_fn=None, log_fn=None, info_level=0):
    if open_text_fn is None:
        open_text_fn = open
    if log_fn is None:
        log_fn = lambda _msg, _lvl=0: None
    log_fn("Writing gene stats to %s" % output_file, info_level)

    with open_text_fn(output_file, 'w') as output_fh:
        if runtime.genes is not None:
            genes = runtime.genes
        elif runtime.gene_to_huge_score is not None:
            genes = list(runtime.gene_to_huge_score.keys())
        elif runtime.gene_to_gwas_huge_score is not None:
            genes = list(runtime.gene_to_huge_score.keys())
        elif runtime.gene_to_huge_score is not None:
            genes = list(runtime.gene_to_huge_score.keys())
        else:
            return

        huge_only_genes = set()
        if runtime.gene_to_huge_score is not None:
            huge_only_genes = set(runtime.gene_to_huge_score.keys()) - set(genes)
        if runtime.gene_to_gwas_huge_score is not None:
            huge_only_genes = set(runtime.gene_to_gwas_huge_score.keys()) - set(genes) - set(huge_only_genes)
        if runtime.gene_to_exomes_huge_score is not None:
            huge_only_genes = set(runtime.gene_to_exomes_huge_score.keys()) - set(genes) - set(huge_only_genes)

        if runtime.genes_missing is not None:
            huge_only_genes = huge_only_genes - set(runtime.genes_missing)

        huge_only_genes = list(huge_only_genes)

        write_regression = runtime.Y_for_regression is not None and runtime.Y is not None and np.any(~np.isclose(runtime.Y, runtime.Y_for_regression))
        write_log_bf_diagnostics = False
        if runtime.Y is not None and runtime.Y_r_hat is not None and runtime.Y_mcse is not None and runtime.genes is not None:
            for i in range(len(runtime.genes)):
                if np.isfinite(runtime.Y_mcse[i]) and np.isfinite(runtime.Y_r_hat[i]) and runtime.Y_mcse[i] > 0 and runtime.Y_r_hat[i] > 1:
                    write_log_bf_diagnostics = True
                    break

        header = "Gene"

        if runtime.priors is not None:
            header = "%s\t%s" % (header, "prior")
            if runtime.priors_r_hat is not None:
                header = "%s\t%s\t%s" % (header, "prior_r_hat", "prior_mcse")
        if runtime.priors_adj is not None:
            header = "%s\t%s" % (header, "prior_adj")
        if runtime.combined_prior_Ys is not None:
            header = "%s\t%s" % (header, "combined")
            if runtime.combined_prior_Ys_r_hat is not None:
                header = "%s\t%s\t%s" % (header, "combined_r_hat", "combined_mcse")
        if runtime.combined_prior_Ys_adj is not None:
            header = "%s\t%s" % (header, "combined_adj")
        if runtime.combined_prior_Y_ses is not None:
            header = "%s\t%s" % (header, "combined_se")
        if runtime.combined_Ds is not None:
            header = "%s\t%s" % (header, "combined_D")
        if runtime.gene_to_huge_score is not None:
            header = "%s\t%s" % (header, "huge_score")
        if runtime.gene_to_gwas_huge_score is not None:
            header = "%s\t%s" % (header, "huge_score_gwas")
        if runtime.gene_to_gwas_huge_score_uncorrected is not None:
            header = "%s\t%s" % (header, "huge_score_gwas_uncorrected")
        if runtime.gene_to_exomes_huge_score is not None:
            header = "%s\t%s" % (header, "huge_score_exomes")
        if runtime.gene_to_positive_controls is not None:
            header = "%s\t%s" % (header, "positive_control")
        if runtime.gene_to_case_count_logbf is not None:
            header = "%s\t%s" % (header, "case_count_bf")
        if runtime.Y is not None:
            header = "%s\t%s" % (header, "log_bf")
            if write_log_bf_diagnostics:
                header = "%s\t%s\t%s" % (header, "log_bf_r_hat", "log_bf_mcse")
        if write_regression:
            header = "%s\t%s" % (header, "log_bf_regression")
        if runtime.Y_uncorrected is not None:
            header = "%s\t%s" % (header, "log_bf_uncorrected")
        if runtime.priors_orig is not None:
            header = "%s\t%s" % (header, "prior_orig")
        if runtime.priors_adj_orig is not None:
            header = "%s\t%s" % (header, "prior_adj_orig")
        if runtime.batches is not None:
            header = "%s\t%s" % (header, "batch")
        if runtime.X_orig is not None:
            header = "%s\t%s" % (header, "N")            
        if runtime.gene_to_chrom is not None:
            header = "%s\t%s" % (header, "Chrom")
        if runtime.gene_to_pos is not None:
            header = "%s\t%s\t%s" % (header, "Start", "End")

        if runtime.gene_covariate_zs is not None:
            header = "%s\t%s" % (header, "\t".join(map(lambda x: "%s" % x, [runtime.gene_covariate_names[i] for i in range(len(runtime.gene_covariate_names)) if i != runtime.gene_covariate_intercept_index])))

        output_fh.write("%s\n" % header)

        ordered_i = range(len(runtime.genes))
        if runtime.combined_prior_Ys is not None:
            ordered_i = sorted(ordered_i, key=lambda k: -runtime.combined_prior_Ys[k])
        elif runtime.priors is not None:
            ordered_i = sorted(ordered_i, key=lambda k: -runtime.priors[k])
        elif runtime.Y is not None:
            ordered_i = sorted(ordered_i, key=lambda k: -runtime.Y[k])
        elif write_regression:
            ordered_i = sorted(ordered_i, key=lambda k: -runtime.Y_for_regression[k])

        gene_N = runtime.get_gene_N()
        for i in ordered_i:
            gene = genes[i]
            line = gene
            if runtime.priors is not None:
                line = "%s\t%.3g" % (line, runtime.priors[i])
                if runtime.priors_r_hat is not None:
                    line = "%s\t%.3g\t%.3g" % (line, runtime.priors_r_hat[i], runtime.priors_mcse[i])
            if runtime.priors_adj is not None:
                line = "%s\t%.3g" % (line, runtime.priors_adj[i])
            if runtime.combined_prior_Ys is not None:
                line = "%s\t%.3g" % (line, runtime.combined_prior_Ys[i])
                if runtime.combined_prior_Ys_r_hat is not None:
                    line = "%s\t%.3g\t%.3g" % (line, runtime.combined_prior_Ys_r_hat[i], runtime.combined_prior_Ys_mcse[i])
            if runtime.combined_prior_Ys_adj is not None:
                line = "%s\t%.3g" % (line, runtime.combined_prior_Ys_adj[i])
            if runtime.combined_prior_Y_ses is not None:
                line = "%s\t%.3g" % (line, runtime.combined_prior_Y_ses[i])
            if runtime.combined_Ds is not None:
                line = "%s\t%.3g" % (line, runtime.combined_Ds[i])
            if runtime.gene_to_huge_score is not None:
                if gene in runtime.gene_to_huge_score:
                    line = "%s\t%.3g" % (line, runtime.gene_to_huge_score[gene])
                else:
                    line = "%s\t%s" % (line, "NA")
            if runtime.gene_to_gwas_huge_score is not None:
                if gene in runtime.gene_to_gwas_huge_score:
                    line = "%s\t%.3g" % (line, runtime.gene_to_gwas_huge_score[gene])
                else:
                    line = "%s\t%s" % (line, "NA")
            if runtime.gene_to_gwas_huge_score_uncorrected is not None:
                if gene in runtime.gene_to_gwas_huge_score_uncorrected:
                    line = "%s\t%.3g" % (line, runtime.gene_to_gwas_huge_score_uncorrected[gene])
                else:
                    line = "%s\t%s" % (line, "NA")
            if runtime.gene_to_exomes_huge_score is not None:
                if gene in runtime.gene_to_exomes_huge_score:
                    line = "%s\t%.3g" % (line, runtime.gene_to_exomes_huge_score[gene])
                else:
                    line = "%s\t%s" % (line, "NA")
            if runtime.gene_to_positive_controls is not None:
                if gene in runtime.gene_to_positive_controls:
                    line = "%s\t%.3g" % (line, runtime.gene_to_positive_controls[gene])
                else:
                    line = "%s\t%s" % (line, "NA")

            if runtime.gene_to_case_count_logbf is not None:
                if gene in runtime.gene_to_case_count_logbf:
                    line = "%s\t%.3g" % (line, runtime.gene_to_case_count_logbf[gene])
                else:
                    line = "%s\t%s" % (line, "NA")
            if runtime.Y is not None:
                line = "%s\t%.3g" % (line, runtime.Y[i])
                if write_log_bf_diagnostics:
                    if np.isfinite(runtime.Y_mcse[i]) and np.isfinite(runtime.Y_r_hat[i]) and runtime.Y_mcse[i] > 0 and runtime.Y_r_hat[i] > 1:
                        line = "%s\t%.3g\t%.3g" % (line, runtime.Y_r_hat[i], runtime.Y_mcse[i])
                    else:
                        line = "%s\t%s\t%s" % (line, "NA", "NA")
            if write_regression:
                line = "%s\t%.3g" % (line, runtime.Y_for_regression[i])
            if runtime.Y_uncorrected is not None:
                line = "%s\t%.3g" % (line, runtime.Y_uncorrected[i])
            if runtime.priors_orig is not None:
                line = "%s\t%.3g" % (line, runtime.priors_orig[i])
            if runtime.priors_adj_orig is not None:
                line = "%s\t%.3g" % (line, runtime.priors_adj_orig[i])
            if runtime.batches is not None:
                line = "%s\t%s" % (line, runtime.batches[i])
            if runtime.X_orig is not None:
                line = "%s\t%d" % (line, gene_N[i])
            if runtime.gene_to_chrom is not None:
                line = "%s\t%s" % (line, runtime.gene_to_chrom[gene] if gene in runtime.gene_to_chrom else "NA")
            if runtime.gene_to_pos is not None:
                line = "%s\t%s\t%s" % (line, runtime.gene_to_pos[gene][0] if gene in runtime.gene_to_pos else "NA", runtime.gene_to_pos[gene][1] if gene in runtime.gene_to_pos else "NA")

            if runtime.gene_covariate_zs is not None:
                line = "%s\t%s" % (line, "\t".join(map(lambda x: "%.3g" % x, [runtime.gene_covariate_zs[i,j] for j in range(len(runtime.gene_covariate_names)) if j != runtime.gene_covariate_intercept_index])))

            output_fh.write("%s\n" % line)

        if runtime.genes_missing is not None:
            gene_N_missing = runtime.get_gene_N(get_missing=True)

            for i in range(len(runtime.genes_missing)):
                gene = runtime.genes_missing[i]
                line = gene
                if runtime.priors is not None:
                    line = ("%s\t%.3g" % (line, runtime.priors_missing[i])) if runtime.priors_missing is not None else ("%s\t%s" % (line, "NA"))
                    if runtime.priors_r_hat is not None:
                        line = "%s\t%s\t%s" % (line, "NA", "NA")
                if runtime.priors_adj is not None:
                    line = ("%s\t%.3g" % (line, runtime.priors_adj_missing[i])) if runtime.priors_adj_missing is not None else ("%s\t%s" % (line, "NA"))
                if runtime.combined_prior_Ys is not None:
                    #has no Y of itself so its combined is just the prior
                    line = ("%s\t%.3g" % (line, runtime.priors_missing[i])) if runtime.priors_missing is not None else ("%s\t%s" % (line, "NA"))
                    if runtime.combined_prior_Ys_r_hat is not None:
                        line = "%s\t%s\t%s" % (line, "NA", "NA")
                if runtime.combined_prior_Ys_adj is not None:
                    #has no Y of itself so its combined is just the prior
                    line = ("%s\t%.3g" % (line, runtime.priors_adj_missing[i])) if runtime.priors_adj_missing is not None else ("%s\t%s" % (line, "NA"))
                if runtime.combined_prior_Y_ses is not None:
                    line = "%s\t%s" % (line, "NA")
                if runtime.combined_Ds_missing is not None:
                    line = "%s\t%.3g" % (line, runtime.combined_Ds_missing[i])
                if runtime.gene_to_huge_score is not None:
                    if gene in runtime.gene_to_huge_score:
                        line = "%s\t%.3g" % (line, runtime.gene_to_huge_score[gene])
                    else:
                        line = "%s\t%s" % (line, "NA")
                if runtime.gene_to_gwas_huge_score is not None:
                    if gene in runtime.gene_to_gwas_huge_score:
                        line = "%s\t%.3g" % (line, runtime.gene_to_gwas_huge_score[gene])
                    else:
                        line = "%s\t%s" % (line, "NA")
                if runtime.gene_to_gwas_huge_score_uncorrected is not None:
                    if gene in runtime.gene_to_gwas_huge_score_uncorrected:
                        line = "%s\t%.3g" % (line, runtime.gene_to_gwas_huge_score_uncorrected[gene])
                    else:
                        line = "%s\t%s" % (line, "NA")
                if runtime.gene_to_exomes_huge_score is not None:
                    if gene in runtime.gene_to_exomes_huge_score:
                        line = "%s\t%.3g" % (line, runtime.gene_to_exomes_huge_score[gene])
                    else:
                        line = "%s\t%s" % (line, "NA")
                if runtime.gene_to_positive_controls is not None:
                    if gene in runtime.gene_to_positive_controls:
                        line = "%s\t%.3g" % (line, runtime.gene_to_positive_controls[gene])
                    else:
                        line = "%s\t%s" % (line, "NA")
                if runtime.gene_to_case_count_logbf is not None:
                    if gene in runtime.gene_to_case_count_logbf:
                        line = "%s\t%.3g" % (line, runtime.gene_to_case_count_logbf[gene])
                    else:
                        line = "%s\t%s" % (line, "NA")
                if runtime.Y is not None:
                    line = "%s\t%s" % (line, "NA")
                    if write_log_bf_diagnostics:
                        line = "%s\t%s\t%s" % (line, "NA", "NA")
                if write_regression:
                    line = "%s\t%s" % (line, "NA")
                if runtime.Y_uncorrected is not None:
                    line = "%s\t%s" % (line, "NA")
                if runtime.priors_orig is not None:
                    line = ("%s\t%.3g" % (line, runtime.priors_missing_orig[i])) if runtime.priors_missing_orig is not None else ("%s\t%s" % (line, "NA"))

                if runtime.priors_adj_orig is not None:
                    line = ("%s\t%.3g" % (line, runtime.priors_adj_missing_orig[i])) if runtime.priors_adj_missing_orig is not None else ("%s\t%s" % (line, "NA"))
                if runtime.batches is not None:
                    line = "%s\t%s" % (line, "NA")
                if runtime.X_orig is not None:
                    line = "%s\t%d" % (line, gene_N_missing[i])
                if runtime.gene_to_chrom is not None:
                    line = "%s\t%s" % (line, runtime.gene_to_chrom[gene] if gene in runtime.gene_to_chrom else "NA")
                if runtime.gene_to_pos is not None:
                    line = "%s\t%s\t%s" % (line, runtime.gene_to_pos[gene][0] if gene in runtime.gene_to_pos else "NA", runtime.gene_to_pos[gene][1] if gene in runtime.gene_to_pos else "NA")

                if runtime.gene_covariate_zs is not None:
                    line = "%s\t%s" % (line, "\t".join(["NA" for j in range(len(runtime.gene_covariate_names)) if j != runtime.gene_covariate_intercept_index]))

                output_fh.write("%s\n" % line)

        for i in range(len(huge_only_genes)):
            gene = huge_only_genes[i]
            line = gene
            if runtime.priors is not None:
                line = "%s\t%s" % (line, "NA")
                if runtime.priors_r_hat is not None:
                    line = "%s\t%s\t%s" % (line, "NA", "NA")
            if runtime.priors_adj is not None:
                line = "%s\t%s" % (line, "NA")
            if runtime.combined_prior_Ys is not None:
                line = "%s\t%s" % (line, "NA")
                if runtime.combined_prior_Ys_r_hat is not None:
                    line = "%s\t%s\t%s" % (line, "NA", "NA")
            if runtime.combined_prior_Ys_adj is not None:
                line = "%s\t%s" % (line, "NA")
            if runtime.combined_prior_Y_ses is not None:
                line = "%s\t%s" % (line, "NA")
            if runtime.combined_Ds_missing is not None:
                line = "%s\t%s" % (line, "NA")
            if runtime.gene_to_huge_score is not None:
                if gene in runtime.gene_to_huge_score:
                    line = "%s\t%.3g" % (line, runtime.gene_to_huge_score[gene])
                else:
                    line = "%s\t%s" % (line, "NA")
            if runtime.gene_to_gwas_huge_score is not None:
                if gene in runtime.gene_to_gwas_huge_score:
                    line = "%s\t%.3g" % (line, runtime.gene_to_gwas_huge_score[gene])
                else:
                    line = "%s\t%s" % (line, "NA")
            if runtime.gene_to_gwas_huge_score_uncorrected is not None:
                if gene in runtime.gene_to_gwas_huge_score_uncorrected:
                    line = "%s\t%.3g" % (line, runtime.gene_to_gwas_huge_score_uncorrected[gene])
                else:
                    line = "%s\t%s" % (line, "NA")
            if runtime.gene_to_exomes_huge_score is not None:
                if gene in runtime.gene_to_exomes_huge_score:
                    line = "%s\t%.3g" % (line, runtime.gene_to_exomes_huge_score[gene])
                else:
                    line = "%s\t%s" % (line, "NA")
            if runtime.gene_to_positive_controls is not None:
                if gene in runtime.gene_to_positive_controls:
                    line = "%s\t%.3g" % (line, runtime.gene_to_positive_controls[gene])
                else:
                    line = "%s\t%s" % (line, "NA")
            if runtime.gene_to_case_count_logbf is not None:
                if gene in runtime.gene_to_case_count_logbf:
                    line = "%s\t%.3g" % (line, runtime.gene_to_case_count_logbf[gene])
                else:
                    line = "%s\t%s" % (line, "NA")
            if runtime.Y is not None:
                line = "%s\t%s" % (line, "NA")
                if write_log_bf_diagnostics:
                    line = "%s\t%s\t%s" % (line, "NA", "NA")
            if write_regression:
                line = "%s\t%s" % (line, "NA")
            if runtime.Y_uncorrected is not None:
                line = "%s\t%s" % (line, "NA")
            if runtime.priors_orig is not None:
                line = "%s\t%s" % (line, "NA")
            if runtime.priors_adj_orig is not None:
                line = "%s\t%s" % (line, "NA")
            if runtime.batches is not None:
                line = "%s\t%s" % (line, "NA")
            if runtime.X_orig is not None:
                line = "%s\t%s" % (line, "NA")
            if runtime.gene_to_chrom is not None:
                line = "%s\t%s" % (line, runtime.gene_to_chrom[gene] if gene in runtime.gene_to_chrom else "NA")
            if runtime.gene_to_pos is not None:
                line = "%s\t%s\t%s" % (line, runtime.gene_to_pos[gene][0] if gene in runtime.gene_to_pos else "NA", runtime.gene_to_pos[gene][1] if gene in runtime.gene_to_pos else "NA")
                
            if runtime.gene_covariate_zs is not None:
                line = "%s\t%s" % (line, "\t".join(["NA" for j in range(len(runtime.gene_covariate_names)) if j != runtime.gene_covariate_intercept_index]))

            output_fh.write("%s\n" % line)


def write_gene_gene_set_statistics(runtime, output_file, max_no_write_gene_gene_set_beta=0.0001, write_filter_beta_uncorrected=False, *, open_text_fn=None, log_fn=None, info_level=0):
    if open_text_fn is None:
        open_text_fn = open
    if log_fn is None:
        log_fn = lambda _msg, _lvl=0: None
    log_fn("Writing gene gene set stats to %s" % output_file, info_level)

    if runtime.genes is None or runtime.X_orig is None or (runtime.betas is None and runtime.beta_tildes is None):
        return

    if runtime.gene_to_gwas_huge_score is not None and runtime.gene_to_exomes_huge_score is not None:
        gene_to_huge_score = runtime.gene_to_gwas_huge_score
        huge_score_label = "huge_score_gwas"
        gene_to_huge_score2 = runtime.gene_to_exomes_huge_score
        huge_score2_label = "huge_score_exomes"
    else:
        gene_to_huge_score = runtime.gene_to_huge_score
        huge_score_label = "huge_score"
        gene_to_huge_score2 = None
        huge_score2_label = None
        if gene_to_huge_score is None:
            gene_to_huge_score = runtime.gene_to_gwas_huge_score
            huge_score_label = "huge_score_gwas"
        if gene_to_huge_score is None:
            gene_to_huge_score = runtime.gene_to_exomes_huge_score
            huge_score_label = "huge_score_exomes"

    write_regression = runtime.Y_for_regression is not None and runtime.Y is not None and np.any(~np.isclose(runtime.Y, runtime.Y_for_regression))

    with open_text_fn(output_file, 'w') as output_fh:

        header = "Gene"

        if runtime.priors is not None:
            header = "%s\t%s" % (header, "prior")
        if runtime.combined_prior_Ys is not None:
            header = "%s\t%s" % (header, "combined")
        if runtime.Y is not None:
            header = "%s\t%s" % (header, "log_bf")
        if write_regression:
            header = "%s\t%s" % (header, "log_bf_for_regression")
        if gene_to_huge_score is not None:
            header = "%s\t%s" % (header, huge_score_label)
        if gene_to_huge_score2 is not None:
            header = "%s\t%s" % (header, huge_score2_label)

        header = "%s\t%s\t%s\t%s" % (header, "gene_set", "beta", "weight")

        output_fh.write("%s\n" % header)

        ordered_i = range(len(runtime.genes))
        if runtime.combined_prior_Ys is not None:
            ordered_i = sorted(ordered_i, key=lambda k: -runtime.combined_prior_Ys[k])
        elif runtime.priors is not None:
            ordered_i = sorted(ordered_i, key=lambda k: -runtime.priors[k])
        elif runtime.Y is not None:
            ordered_i = sorted(ordered_i, key=lambda k: -runtime.Y[k])
        elif write_regression is not None:
            ordered_i = sorted(ordered_i, key=lambda k: -runtime.Y_for_regression[k])

        betas_to_use = runtime.betas if runtime.betas is not None else runtime.beta_tildes

        betas_for_filter = betas_to_use 
        if write_filter_beta_uncorrected and runtime.betas_uncorrected is not None:
            betas_for_filter = runtime.betas_uncorrected

        for i in ordered_i:
            gene = runtime.genes[i]

            if np.abs(runtime.X_orig[i,:]).sum() == 0:
                continue

            ordered_j = sorted(runtime.X_orig[i,:].nonzero()[1], key=lambda k: -betas_to_use[k] / runtime.scale_factors[k])

            for j in ordered_j:
                if np.abs(betas_for_filter[j] / runtime.scale_factors[j]) <= max_no_write_gene_gene_set_beta:
                    continue

                line = gene
                if runtime.priors is not None:
                    line = "%s\t%.3g" % (line, runtime.priors[i])
                if runtime.combined_prior_Ys is not None:
                    line = "%s\t%.3g" % (line, runtime.combined_prior_Ys[i])
                if runtime.Y is not None:
                    line = "%s\t%.3g" % (line, runtime.Y[i])
                if write_regression:
                    line = "%s\t%.3g" % (line, runtime.Y_for_regression[i])
                if gene_to_huge_score is not None:
                    huge_score = gene_to_huge_score[gene] if gene in gene_to_huge_score else 0
                    line = "%s\t%.3g" % (line, huge_score)
                if gene_to_huge_score2 is not None:
                    huge_score2 = gene_to_huge_score2[gene] if gene in gene_to_huge_score2 else 0
                    line = "%s\t%.3g" % (line, huge_score2)


                line = "%s\t%s\t%.3g\t%.3g" % (line, runtime.gene_sets[j], betas_to_use[j] / runtime.scale_factors[j], runtime.X_orig[i,j])
                output_fh.write("%s\n" % line)

        ordered_i = range(len(runtime.genes_missing))
        if runtime.priors_missing is not None:
            ordered_i = sorted(ordered_i, key=lambda k: -runtime.priors_missing[k])

        for i in ordered_i:
            gene = runtime.genes_missing[i]

            if np.abs(runtime.X_orig_missing_genes[i,:]).sum() == 0:
                continue

            ordered_j = sorted(runtime.X_orig_missing_genes[i,:].nonzero()[1], key=lambda k: -betas_to_use[k] / runtime.scale_factors[k])

            for j in ordered_j:
                if np.abs(betas_to_use[j] / runtime.scale_factors[j]) <= max_no_write_gene_gene_set_beta:
                    continue
                line = gene
                if runtime.priors is not None:
                    line = ("%s\t%.3g" % (line, runtime.priors_missing[i])) if runtime.priors_missing is not None else ("%s\t%s" % (line, "NA"))
                if runtime.combined_prior_Ys is not None:
                    line = ("%s\t%.3g" % (line, runtime.priors_missing[i])) if runtime.priors_missing is not None else ("%s\t%s" % (line, "NA"))
                if runtime.Y is not None:
                    line = "%s\t%s" % (line, "NA")
                if write_regression:
                    line = "%s\t%s" % (line, "NA")
                if gene_to_huge_score is not None:
                    line = "%s\t%s" % (line, "NA")
                if gene_to_huge_score2 is not None:
                    line = "%s\t%s" % (line, "NA")

                line = "%s\t%s\t%.3g\t%.3g" % (line, runtime.gene_sets[j], betas_to_use[j] / runtime.scale_factors[j], runtime.X_orig_missing_genes[i,j])
                output_fh.write("%s\n" % line)



def write_phewas_statistics(runtime, output_file, *, open_text_fn=None, log_fn=None, info_level=0):
    if open_text_fn is None:
        open_text_fn = open
    if log_fn is None:
        log_fn = lambda _msg, _lvl=0: None
    if runtime.phenos is None or len(runtime.phenos) == 0:
        return

    log_fn("Writing phewas stats to %s" % output_file, info_level)

    with open_text_fn(output_file, 'w') as output_fh:

        header = "Pheno"

        ordered_inds = None

        write = False
        if runtime.pheno_Y_vs_input_combined_prior_Ys_beta is not None:
            write = True
            ordered_inds = sorted(range(len(runtime.phenos)), key=lambda k: -runtime.pheno_Y_vs_input_combined_prior_Ys_beta[k])

        if runtime.pheno_Y_vs_input_Y_beta is not None:
            write = True
            ordered_inds = sorted(range(len(runtime.phenos)), key=lambda k: -runtime.pheno_Y_vs_input_Y_beta[k])

        if runtime.pheno_Y_vs_input_priors_beta is not None:
            write = True
            ordered_inds = sorted(range(len(runtime.phenos)), key=lambda k: -runtime.pheno_Y_vs_input_priors_beta[k])

        if runtime.pheno_combined_prior_Ys_vs_input_combined_prior_Ys_beta is not None:
            write = True
            ordered_inds = sorted(range(len(runtime.phenos)), key=lambda k: -runtime.pheno_combined_prior_Ys_vs_input_combined_prior_Ys_beta[k])

        if runtime.pheno_combined_prior_Ys_vs_input_Y_beta is not None:
            write = True
            ordered_inds = sorted(range(len(runtime.phenos)), key=lambda k: -runtime.pheno_combined_prior_Ys_vs_input_Y_beta[k])

        if runtime.pheno_combined_prior_Ys_vs_input_priors_beta is not None:
            write = True
            ordered_inds = sorted(range(len(runtime.phenos)), key=lambda k: -runtime.pheno_combined_prior_Ys_vs_input_priors_beta[k])

        if write:
            header = "%s\t%s\t%s\t%s\t%s\t%s\t%s" % (header, "analysis", "beta_tilde", "P", "Z", "SE", "beta")


        if ordered_inds is None:
            ordered_inds = range(len(runtime.phenos))                                      

        output_fh.write("%s\n" % header)

        for i in ordered_inds:
            pheno = runtime.phenos[i]
            line = pheno
            if runtime.pheno_Y_vs_input_combined_prior_Ys_beta is not None:
                output_fh.write("%s\t%s\t%.3g\t%.3g\t%.3g\t%.3g\t%.3g\n" % (line, "log_bf_vs_combined", runtime.pheno_Y_vs_input_combined_prior_Ys_beta_tilde[i], runtime.pheno_Y_vs_input_combined_prior_Ys_p_value[i], runtime.pheno_Y_vs_input_combined_prior_Ys_Z[i], runtime.pheno_Y_vs_input_combined_prior_Ys_se[i], runtime.pheno_Y_vs_input_combined_prior_Ys_beta[i]))

            if runtime.pheno_Y_vs_input_Y_beta is not None:
                output_fh.write("%s\t%s\t%.3g\t%.3g\t%.3g\t%.3g\t%.3g\n" % (line, "log_bf_vs_log_bf", runtime.pheno_Y_vs_input_Y_beta_tilde[i], runtime.pheno_Y_vs_input_Y_p_value[i], runtime.pheno_Y_vs_input_Y_Z[i], runtime.pheno_Y_vs_input_Y_se[i], runtime.pheno_Y_vs_input_Y_beta[i]))

            if runtime.pheno_Y_vs_input_priors_beta is not None:
                output_fh.write("%s\t%s\t%.3g\t%.3g\t%.3g\t%.3g\t%.3g\n" % (line, "log_bf_vs_prior", runtime.pheno_Y_vs_input_priors_beta_tilde[i], runtime.pheno_Y_vs_input_priors_p_value[i], runtime.pheno_Y_vs_input_priors_Z[i], runtime.pheno_Y_vs_input_priors_se[i], runtime.pheno_Y_vs_input_priors_beta[i]))

            if runtime.pheno_combined_prior_Ys_vs_input_combined_prior_Ys_beta is not None:
                output_fh.write("%s\t%s\t%.3g\t%.3g\t%.3g\t%.3g\t%.3g\n" % (line, "combined_vs_combined", runtime.pheno_combined_prior_Ys_vs_input_combined_prior_Ys_beta_tilde[i], runtime.pheno_combined_prior_Ys_vs_input_combined_prior_Ys_p_value[i], runtime.pheno_combined_prior_Ys_vs_input_combined_prior_Ys_Z[i], runtime.pheno_combined_prior_Ys_vs_input_combined_prior_Ys_se[i], runtime.pheno_combined_prior_Ys_vs_input_combined_prior_Ys_beta[i]))

            if runtime.pheno_combined_prior_Ys_vs_input_Y_beta is not None:
                output_fh.write("%s\t%s\t%.3g\t%.3g\t%.3g\t%.3g\t%.3g\n" % (line, "combined_vs_log_bf", runtime.pheno_combined_prior_Ys_vs_input_Y_beta_tilde[i], runtime.pheno_combined_prior_Ys_vs_input_Y_p_value[i], runtime.pheno_combined_prior_Ys_vs_input_Y_Z[i], runtime.pheno_combined_prior_Ys_vs_input_Y_se[i], runtime.pheno_combined_prior_Ys_vs_input_Y_beta[i]))

            if runtime.pheno_combined_prior_Ys_vs_input_priors_beta is not None:
                output_fh.write("%s\t%s\t%.3g\t%.3g\t%.3g\t%.3g\t%.3g\n" % (line, "combined_vs_prior", runtime.pheno_combined_prior_Ys_vs_input_priors_beta_tilde[i], runtime.pheno_combined_prior_Ys_vs_input_priors_p_value[i], runtime.pheno_combined_prior_Ys_vs_input_priors_Z[i], runtime.pheno_combined_prior_Ys_vs_input_priors_se[i], runtime.pheno_combined_prior_Ys_vs_input_priors_beta[i]))

#HELPER FUNCTIONS

'''
Read in gene bfs for LOGISTIC or EMPIRICAL mapping
'''


def write_factor_phewas_statistics(runtime, output_file, *, open_text_fn=None, log_fn=None, info_level=0):
    if open_text_fn is None:
        open_text_fn = open
    if log_fn is None:
        log_fn = lambda _msg, _lvl=0: None
    if runtime.phenos is None or len(runtime.phenos) == 0:
        return

    if runtime.factor_labels is None or (runtime.factor_phewas_Y_betas is None and runtime.factor_phewas_combined_prior_Ys_betas is None and runtime.factor_phewas_Y_huber_betas is None and runtime.factor_phewas_combined_prior_Ys_huber_betas is None):
        return 

    log_fn("Writing factor phewas stats to %s" % output_file, info_level)

    with open_text_fn(output_file, 'w') as output_fh:

        header = "%s\t%s\t%s\t%s" % ("Factor", "Label", "Pheno", "analysis")

        header = "%s\t%s\t%s\t%s\t%s\t%s" % (header, "beta", "P", "P_onesided", "Z", "SE")

        output_fh.write("%s\n" % header)

        for f in range(len(runtime.factor_labels)):
            if runtime.factor_phewas_Y_betas is not None:
                ordered_fn = lambda k: runtime.factor_phewas_Y_p_values[f,k]
            else:
                ordered_fn = lambda k: runtime.factor_phewas_combined_prior_Ys_p_values[f,k]

            for i in sorted(range(len(runtime.phenos)), key=ordered_fn):
                pheno = runtime.phenos[i]
                line = "%s\t%s\t%s" % ("Factor%d" % (f + 1), runtime.factor_labels[f], pheno)
                if runtime.factor_phewas_Y_betas is not None:
                    output_fh.write("%s\t%s\t%.3g\t%.3g\t%.3g\t%.3g\t%.3g\n" % (line, "Y", runtime.factor_phewas_Y_betas[f,i], runtime.factor_phewas_Y_p_values[f,i], runtime.factor_phewas_Y_one_sided_p_values[f,i], runtime.factor_phewas_Y_zs[f,i], runtime.factor_phewas_Y_ses[f,i]))
                if runtime.factor_phewas_Y_huber_betas is not None:
                    output_fh.write("%s\t%s\t%.3g\t%.3g\t%.3g\t%.3g\t%.3g\n" % (line, "Y_huber", runtime.factor_phewas_Y_huber_betas[f,i], runtime.factor_phewas_Y_huber_p_values[f,i], runtime.factor_phewas_Y_huber_one_sided_p_values[f,i], runtime.factor_phewas_Y_huber_zs[f,i], runtime.factor_phewas_Y_huber_ses[f,i]))
                if runtime.factor_phewas_combined_prior_Ys_betas is not None:
                    output_fh.write("%s\t%s\t%.3g\t%.3g\t%.3g\t%.3g\t%.3g\n" % (line, "combined", runtime.factor_phewas_combined_prior_Ys_betas[f,i], runtime.factor_phewas_combined_prior_Ys_p_values[f,i], runtime.factor_phewas_combined_prior_Ys_one_sided_p_values[f,i], runtime.factor_phewas_combined_prior_Ys_zs[f,i], runtime.factor_phewas_combined_prior_Ys_ses[f,i]))
                if runtime.factor_phewas_combined_prior_Ys_huber_betas is not None:
                    output_fh.write("%s\t%s\t%.3g\t%.3g\t%.3g\t%.3g\t%.3g\n" % (line, "combined_huber", runtime.factor_phewas_combined_prior_Ys_huber_betas[f,i], runtime.factor_phewas_combined_prior_Ys_huber_p_values[f,i], runtime.factor_phewas_combined_prior_Ys_huber_one_sided_p_values[f,i], runtime.factor_phewas_combined_prior_Ys_huber_zs[f,i], runtime.factor_phewas_combined_prior_Ys_huber_ses[f,i]))



def write_gene_set_statistics(runtime, output_file, max_no_write_gene_set_beta=None, max_no_write_gene_set_beta_uncorrected=None, basic=False, *, open_text_fn=None, log_fn=None, info_level=0, debug_only_avg_huge=False):
    if open_text_fn is None:
        open_text_fn = open
    if log_fn is None:
        log_fn = lambda _msg, _lvl=0: None
    log_fn("Writing gene set stats to %s" % output_file, info_level)
    with open_text_fn(output_file, 'w') as output_fh:
        if runtime.gene_sets is None:
            return
        inf_betas = getattr(runtime, "inf_betas", None)
        inf_betas_orig = getattr(runtime, "inf_betas_orig", None)
        inf_betas_missing = getattr(runtime, "inf_betas_missing", None)
        inf_betas_missing_orig = getattr(runtime, "inf_betas_missing_orig", None)
        header = "Gene_Set"
        if runtime.gene_set_labels is not None:
            header = "%s\t%s" % (header, "label")
        if runtime.X_orig is not None:
            col_sums = runtime.get_col_sums(runtime.X_orig)
            header = "%s\t%s" % (header, "N")
            header = "%s\t%s" % (header, "scale")
        if runtime.beta_tildes is not None:
            header = "%s\t%s\t%s\t%s\t%s\t%s" % (header, "beta_tilde", "beta_tilde_internal", "P", "Z", "SE")
        if inf_betas is not None and not basic:
            header = "%s\t%s" % (header, "inf_beta")            
        if runtime.betas is not None:
            header = "%s\t%s\t%s" % (header, "beta", "beta_internal")
            if runtime.betas_r_hat is not None:
                header = "%s\t%s\t%s" % (header, "beta_r_hat", "beta_mcse")
        if runtime.betas_uncorrected is not None and not basic:
            header = "%s\t%s" % (header, "beta_uncorrected")            
            if runtime.betas_uncorrected_r_hat is not None:
                header = "%s\t%s\t%s" % (header, "beta_uncorrected_r_hat", "beta_uncorrected_mcse")
        if not basic:
            if runtime.non_inf_avg_cond_betas is not None:
                header = "%s\t%s" % (header, "avg_cond_beta")            
            if runtime.non_inf_avg_postps is not None:
                header = "%s\t%s" % (header, "avg_postp")            
            if runtime.beta_tildes_orig is not None:
                header = "%s\t%s\t%s\t%s\t%s\t%s" % (header, "beta_tilde_orig", "beta_tilde_internal_orig", "P_orig", "Z_orig", "SE_orig")
            if inf_betas_orig is not None:
                header = "%s\t%s" % (header, "inf_beta_orig")            
            if runtime.betas_orig is not None:
                header = "%s\t%s\t%s" % (header, "beta_orig", "beta_internal_orig")
            if runtime.betas_uncorrected_orig is not None:
                header = "%s\t%s\t%s" % (header, "beta_uncorrected_orig", "beta_uncorrected_internal_orig")
            if runtime.non_inf_avg_cond_betas_orig is not None:
                header = "%s\t%s" % (header, "avg_cond_beta_orig")            
            if runtime.non_inf_avg_postps_orig is not None:
                header = "%s\t%s" % (header, "avg_postp_orig")            
            if runtime.ps is not None or runtime.p is not None:
                header = "%s\t%s" % (header, "p_used")
            if runtime.sigma2s is not None or runtime.sigma2 is not None:
                header = "%s\t%s" % (header, "sigma2_used")
            if (runtime.sigma2s is not None or runtime.sigma2 is not None) and runtime.sigma_threshold_k is not None and runtime.sigma_threshold_xo is not None:
                header = "%s\t%s" % (header, "sigma2_thresholded")
            if runtime.X_osc is not None:
                header = "%s\t%s\t%s\t%s" % (header, "O", "X_O", "weight")
            if runtime.total_qc_metrics is not None:
                if debug_only_avg_huge:
                    header = "%s\t%s" % (header, "avg_huge_adjustment")
                else:
                    header = "%s\t%s\t%s" % (header, "\t".join(map(lambda x: "avg_%s" % x, [runtime.gene_covariate_names[i] for i in range(len(runtime.gene_covariate_names)) if i != runtime.gene_covariate_intercept_index])), "avg_huge_adjustment")

            if runtime.mean_qc_metrics is not None:
                header = "%s\t%s" % (header, "avg_avg_metric")

        output_fh.write("%s\n" % header)

        ordered_i = range(len(runtime.gene_sets))
        if runtime.betas is not None:
            ordered_i = sorted(ordered_i, key=lambda k: -runtime.betas[k] / runtime.scale_factors[k])
        elif runtime.p_values is not None:
            ordered_i = sorted(ordered_i, key=lambda k: runtime.p_values[k])

        for i in ordered_i:

            if max_no_write_gene_set_beta is not None and runtime.betas is not None and np.abs(runtime.betas[i] / runtime.scale_factors[i]) <= max_no_write_gene_set_beta:
                continue

            if max_no_write_gene_set_beta_uncorrected is not None and runtime.betas_uncorrected is not None and np.abs(runtime.betas_uncorrected[i] / runtime.scale_factors[i]) <= max_no_write_gene_set_beta_uncorrected:
                continue

            line = runtime.gene_sets[i]
            if runtime.gene_set_labels is not None:
                line = "%s\t%s" % (line, runtime.gene_set_labels[i])
            if runtime.X_orig is not None:
                line = "%s\t%d" % (line, col_sums[i])
                line = "%s\t%.3g" % (line, runtime.scale_factors[i])

            if runtime.beta_tildes is not None:
                line = "%s\t%.3g\t%.3g\t%.3g\t%.3g\t%.3g" % (line, runtime.beta_tildes[i] / runtime.scale_factors[i], runtime.beta_tildes[i], runtime.p_values[i], runtime.z_scores[i], runtime.ses[i] / runtime.scale_factors[i])
            if inf_betas is not None and not basic:
                line = "%s\t%.3g" % (line, inf_betas[i] / runtime.scale_factors[i])            
            if runtime.betas is not None:
                line = "%s\t%.3g\t%.3g" % (line, runtime.betas[i] / runtime.scale_factors[i], runtime.betas[i])
                if runtime.betas_r_hat is not None:
                    line = "%s\t%.3g\t%.3g" % (line, runtime.betas_r_hat[i], runtime.betas_mcse[i])
            if runtime.betas_uncorrected is not None and not basic:
                line = "%s\t%.3g" % (line, runtime.betas_uncorrected[i] / runtime.scale_factors[i])            
                if runtime.betas_uncorrected_r_hat is not None:
                    line = "%s\t%.3g\t%.3g" % (line, runtime.betas_uncorrected_r_hat[i], runtime.betas_uncorrected_mcse[i])
            if not basic:
                if runtime.non_inf_avg_cond_betas is not None:
                    line = "%s\t%.3g" % (line, runtime.non_inf_avg_cond_betas[i] / runtime.scale_factors[i])
                if runtime.non_inf_avg_postps is not None:
                    line = "%s\t%.3g" % (line, runtime.non_inf_avg_postps[i])
                if runtime.beta_tildes_orig is not None:
                    line = "%s\t%.3g\t%.3g\t%.3g\t%.3g\t%.3g" % (line, runtime.beta_tildes_orig[i] / runtime.scale_factors[i], runtime.beta_tildes_orig[i], runtime.p_values_orig[i], runtime.z_scores_orig[i], runtime.ses_orig[i] / runtime.scale_factors[i])
                if inf_betas_orig is not None:
                    line = "%s\t%.3g" % (line, inf_betas_orig[i] / runtime.scale_factors[i])            
                if runtime.betas_orig is not None:
                    line = "%s\t%.3g\t%.3g" % (line, runtime.betas_orig[i] / runtime.scale_factors[i], runtime.betas_orig[i])
                if runtime.betas_uncorrected_orig is not None:
                    line = "%s\t%.3g\t%.3g" % (line, runtime.betas_uncorrected_orig[i] / runtime.scale_factors[i], runtime.betas_uncorrected_orig[i])
                if runtime.non_inf_avg_cond_betas_orig is not None:
                    line = "%s\t%.3g" % (line, runtime.non_inf_avg_cond_betas_orig[i] / runtime.scale_factors[i])
                if runtime.non_inf_avg_postps_orig is not None:
                    line = "%s\t%.3g" % (line, runtime.non_inf_avg_postps_orig[i])

                if runtime.ps is not None or runtime.p is not None:
                    line = "%s\t%.3g" % (line, runtime.ps[i] if runtime.ps is not None else runtime.p)
                if runtime.sigma2s is not None or runtime.sigma2 is not None:
                    line = "%s\t%.3g" % (line, runtime.get_scaled_sigma2(runtime.scale_factors[i], runtime.sigma2s[i] if runtime.sigma2s is not None else runtime.sigma2, runtime.sigma_power, None, None))
                if (runtime.sigma2s is not None or runtime.sigma2 is not None) and runtime.sigma_threshold_k is not None and runtime.sigma_threshold_xo is not None:
                    line = "%s\t%.3g" % (line, runtime.get_scaled_sigma2(runtime.scale_factors[i], runtime.sigma2s[i] if runtime.sigma2s is not None else runtime.sigma2, runtime.sigma_power, runtime.sigma_threshold_k, runtime.sigma_threshold_xo))
                if runtime.X_osc is not None:
                    line = "%s\t%.3g\t%.3g\t%.3g" % (line, runtime.osc[i], runtime.X_osc[i], runtime.osc_weights[i])

                if runtime.total_qc_metrics is not None:
                    line = "%s\t%s" % (line, "\t".join(map(lambda x: "%.3g" % x, runtime.total_qc_metrics[i,:])))
                if runtime.mean_qc_metrics is not None:
                    line = "%s\t%.3g" % (line, runtime.mean_qc_metrics[i])


            output_fh.write("%s\n" % line)

        if runtime.gene_sets_missing is not None:
            ordered_i = range(len(runtime.gene_sets_missing))
            if runtime.betas_missing is not None and runtime.scale_factors_missing is not None:
                ordered_i = sorted(ordered_i, key=lambda k: -runtime.betas_missing[k] / runtime.scale_factors_missing[k])
            elif runtime.p_values_missing is not None:
                ordered_i = sorted(ordered_i, key=lambda k: runtime.p_values_missing[k])

            col_sums_missing = runtime.get_col_sums(runtime.X_orig_missing_gene_sets)
            for i in range(len(runtime.gene_sets_missing)):
                if max_no_write_gene_set_beta is not None and runtime.betas_missing is not None and np.abs(runtime.betas_missing[i] / runtime.scale_factors_missing[i]) <= max_no_write_gene_set_beta:
                    continue

                if max_no_write_gene_set_beta_uncorrected is not None and runtime.betas_uncorrected_missing is not None and np.abs(runtime.betas_uncorrected_missing[i] / runtime.scale_factors_missing[i]) <= max_no_write_gene_set_beta_uncorrected:
                    continue

                line = runtime.gene_sets_missing[i]
                if runtime.gene_set_labels is not None:
                    line = "%s\t%s" % (line, runtime.gene_set_labels_missing[i])
                line = "%s\t%d" % (line, col_sums_missing[i])
                line = "%s\t%.3g" % (line, runtime.scale_factors_missing[i])

                if runtime.beta_tildes is not None:
                    line = "%s\t%.3g\t%.3g\t%.3g\t%.3g\t%.3g" % (line, runtime.beta_tildes_missing[i] / runtime.scale_factors_missing[i], runtime.beta_tildes_missing[i], runtime.p_values_missing[i], runtime.z_scores_missing[i], runtime.ses_missing[i] / runtime.scale_factors_missing[i])
                if inf_betas is not None and not basic:
                    line = "%s\t%.3g" % (line, inf_betas_missing[i] / runtime.scale_factors_missing[i])            
                if runtime.betas is not None:
                    line = "%s\t%.3g\t%.3g" % (line, runtime.betas_missing[i] / runtime.scale_factors_missing[i], runtime.betas_missing[i])
                    if runtime.betas_r_hat is not None:
                        line = "%s\t%s\t%s" % (line, "NA", "NA")
                if runtime.betas_uncorrected is not None and not basic:
                    line = "%s\t%.3g" % (line, runtime.betas_uncorrected_missing[i] / runtime.scale_factors_missing[i])            
                    if runtime.betas_uncorrected_r_hat is not None:
                        line = "%s\t%s\t%s" % (line, "NA", "NA")
                if not basic:
                    if runtime.non_inf_avg_cond_betas is not None:
                        line = "%s\t%.3g" % (line, runtime.non_inf_avg_cond_betas_missing[i] / runtime.scale_factors_missing[i])
                    if runtime.non_inf_avg_postps is not None:
                        line = "%s\t%.3g" % (line, runtime.non_inf_avg_postps_missing[i])
                    if runtime.beta_tildes_orig is not None:
                        line = "%s\t%.3g\t%.3g\t%.3g\t%.3g\t%.3g" % (line, runtime.beta_tildes_missing_orig[i] / runtime.scale_factors_missing[i], runtime.beta_tildes_missing_orig[i], runtime.p_values_missing_orig[i], runtime.z_scores_missing_orig[i], runtime.ses_missing_orig[i] / runtime.scale_factors_missing[i])
                    if inf_betas_orig is not None:
                        line = "%s\t%.3g" % (line, inf_betas_missing_orig[i] / runtime.scale_factors_missing[i])            
                    if runtime.betas_orig is not None:
                        line = "%s\t%.3g\t%.3g" % (line, runtime.betas_missing_orig[i] / runtime.scale_factors_missing[i], runtime.betas_missing_orig[i])
                    if runtime.betas_uncorrected_orig is not None:
                        line = "%s\t%.3g\t%.3g" % (line, runtime.betas_uncorrected_missing_orig[i] / runtime.scale_factors_missing[i], runtime.betas_uncorrected_missing_orig[i])
                    if runtime.non_inf_avg_cond_betas_orig is not None:
                        line = "%s\t%.3g" % (line, runtime.non_inf_avg_cond_betas_missing_orig[i] / runtime.scale_factors_missing[i])
                    if runtime.non_inf_avg_postps_orig is not None:
                        line = "%s\t%.3g" % (line, runtime.non_inf_avg_postps_missing_orig[i])

                    if runtime.ps is not None or runtime.p is not None:
                        line = "%s\t%.3g" % (line, runtime.ps_missing[i] if runtime.ps_missing is not None else runtime.p)

                    if runtime.sigma2s is not None or runtime.sigma2 is not None:
                        line = "%s\t%.3g" % (line, runtime.get_scaled_sigma2(runtime.scale_factors_missing[i], runtime.sigma2s_missing[i] if runtime.sigma2s_missing is not None else runtime.sigma2, runtime.sigma_power, None, None))
                    if (runtime.sigma2s is not None or runtime.sigma2 is not None) and runtime.sigma_threshold_k is not None and runtime.sigma_threshold_xo is not None:
                        line = "%s\t%.3g" % (line, runtime.get_scaled_sigma2(runtime.scale_factors_missing[i], runtime.sigma2s_missing[i] if runtime.sigma2s_missing is not None else runtime.sigma2, runtime.sigma_power, runtime.sigma_threshold_k, runtime.sigma_threshold_xo))

                    if runtime.X_osc is not None:
                        line = "%s\t%.3g\t%.3g\t%.3g" % (line, runtime.osc_missing[i], runtime.X_osc_missing[i], runtime.osc_weights_missing[i])

                    if runtime.total_qc_metrics is not None:
                        line = "%s\t%s" % (line, "\t".join(map(lambda x: "%.3g" % x, runtime.total_qc_metrics_missing[i,:])))
                    if runtime.mean_qc_metrics is not None:
                        line = "%s\t%.3g" % (line, runtime.mean_qc_metrics_missing[i])

                output_fh.write("%s\n" % line)



        if runtime.gene_sets_ignored is not None:

            ordered_i = range(len(runtime.gene_sets_ignored))
            if runtime.p_values_ignored is not None:
                ordered_i = sorted(ordered_i, key=lambda k: runtime.p_values_ignored[k])

            for i in ordered_i:
                ignored_beta_value = 0 
                if max_no_write_gene_set_beta is not None and runtime.betas is not None and ignored_beta_value <= max_no_write_gene_set_beta:
                    continue

                ignored_beta_uncorrected_value = 0 
                if max_no_write_gene_set_beta_uncorrected is not None and runtime.betas_uncorrected is not None and ignored_beta_uncorrected_value <= max_no_write_gene_set_beta_uncorrected:
                    continue


                line = "%s" % runtime.gene_sets_ignored[i]
                if runtime.gene_set_labels is not None:
                    line = "%s\t%s" % (line, runtime.gene_set_labels_ignored[i])

                line = "%s\t%d" % (line, runtime.col_sums_ignored[i])
                line = "%s\t%.3g" % (line, runtime.scale_factors_ignored[i])

                scale_factor_denom = runtime.scale_factors_ignored[i] + 1e-20

                if runtime.beta_tildes is not None:
                    if runtime.beta_tildes_ignored is not None:
                        line = "%s\t%.3g\t%.3g\t%.3g\t%.3g\t%.3g" % (line, runtime.beta_tildes_ignored[i] / scale_factor_denom, runtime.beta_tildes_ignored[i], runtime.p_values_ignored[i], runtime.z_scores_ignored[i], runtime.ses_ignored[i] / scale_factor_denom)
                    else:
                        line = "%s\t%s\t%s\t%s\t%s\t%s" % (line, "NA", "NA", "NA", "NA", "NA")
                if inf_betas is not None and not basic:
                    line = "%s\t%.3g" % (line, 0)            
                if runtime.betas is not None:
                    line = "%s\t%.3g\t%.3g" % (line, ignored_beta_value, ignored_beta_value)
                    if runtime.betas_r_hat is not None:
                        line = "%s\t%s\t%s" % (line, "NA", "NA")
                if runtime.betas_uncorrected is not None and not basic:
                    line = "%s\t%.3g" % (line, ignored_beta_uncorrected_value)            
                    if runtime.betas_uncorrected_r_hat is not None:
                        line = "%s\t%s\t%s" % (line, "NA", "NA")
                if not basic:
                    if runtime.non_inf_avg_cond_betas is not None:
                        line = "%s\t%.3g" % (line, 0)
                    if runtime.non_inf_avg_postps is not None:
                        line = "%s\t%.3g" % (line, 0)
                    if runtime.beta_tildes_orig is not None:
                        if runtime.beta_tildes_ignored is not None:
                            line = "%s\t%.3g\t%.3g\t%.3g\t%.3g\t%.3g" % (line, runtime.beta_tildes_ignored[i] / scale_factor_denom, runtime.beta_tildes_ignored[i], runtime.p_values_ignored[i], runtime.z_scores_ignored[i], runtime.ses_ignored[i] / scale_factor_denom)
                        else:
                            line = "%s\t%s\t%s\t%s\t%s\t%s" % (line, "NA", "NA", "NA", "NA", "NA")
                    if inf_betas_orig is not None:
                        line = "%s\t%.3g" % (line, 0)
                    if runtime.betas_orig is not None:
                        line = "%s\t%.3g\t%.3g" % (line, 0, 0)
                    if runtime.betas_uncorrected_orig is not None:
                        line = "%s\t%.3g\t%.3g" % (line, 0, 0)
                    if runtime.non_inf_avg_cond_betas_orig is not None:
                        line = "%s\t%.3g" % (line, 0)
                    if runtime.non_inf_avg_postps_orig is not None:
                        line = "%s\t%.3g" % (line, 0)

                    if runtime.ps is not None or runtime.p is not None:
                        line = "%s\t%s" % (line, "NA")
                    if runtime.sigma2s is not None or runtime.sigma2 is not None:
                        line = "%s\t%s" % (line, "NA")
                    if (runtime.sigma2s is not None or runtime.sigma2 is not None) and runtime.sigma_threshold_k is not None and runtime.sigma_threshold_xo is not None:
                        line = "%s\t%s" % (line, "NA")

                    if runtime.X_osc is not None:
                        line = "%s\t%s\t%s\t%s" % (line, "NA", "NA", "NA")

                    if runtime.total_qc_metrics is not None:
                        line = "%s\t%s" % (line, "\t".join(map(lambda x: "%.3g" % x, runtime.total_qc_metrics_ignored[i,:])))
                    if runtime.mean_qc_metrics is not None:
                        line = "%s\t%.3g" % (line, runtime.mean_qc_metrics_ignored[i])

                output_fh.write("%s\n" % line)
