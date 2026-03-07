#Usage: python calc.py
#
#This script...
#
#Arguments:
# warnings-file[path]: file to output warnings to; default: standard error
#these are for debugging
#import cProfile
#import resource

#import urllib.request #used (below) only if you specify a URL as an input file
#import requests #used (below) only if you specify --lmm-auth-key

import optparse
import sys
import time
import os
import copy
import contextlib
import json
import re
from dataclasses import dataclass, field
import scipy
import scipy.sparse as sparse
import scipy.stats
import numpy as np
import itertools
import gzip
import random

try:
    from pegs_shared.types import (
        XReadConfig as PegsXReadConfig,
        XReadCallbacks as PegsXReadCallbacks,
        XReadPostCallbacks as PegsXReadPostCallbacks,
    )
    from .pegs_cli_errors import (
        DataValidationError,
        PegsCliError,
        handle_cli_exception as pegs_handle_cli_exception,
        handle_unexpected_exception as pegs_handle_unexpected_exception,
    )
    from pegs_shared.cli import (
        apply_cli_config_overrides as pegs_apply_cli_config_overrides,
        callback_set_comma_separated_args as pegs_callback_set_comma_separated_args,
        callback_set_comma_separated_args_as_float as pegs_callback_set_comma_separated_args_as_float,
        coerce_option_int_list as pegs_coerce_option_int_list,
        configure_random_seed as pegs_configure_random_seed,
        emit_stderr_warning as pegs_emit_stderr_warning,
        fail_removed_cli_aliases as pegs_fail_removed_cli_aliases,
        format_removed_option_message as pegs_format_removed_option_message,
        get_tar_write_mode_for_bundle_path as pegs_get_tar_write_mode_for_bundle_path,
        harmonize_cli_mode_args as pegs_harmonize_cli_mode_args,
        initialize_cli_logging as pegs_initialize_cli_logging,
        is_path_like_dest as pegs_is_path_like_dest,
        iter_parser_options as pegs_iter_parser_options,
        json_safe as pegs_json_safe,
        load_json_config as pegs_load_json_config,
        merge_dicts as pegs_merge_dicts,
        resolve_config_path_value as pegs_resolve_config_path_value,
    )
    from pegs_shared.xdata import (
        xdata_from_input_plan as pegs_xdata_from_input_plan,
        build_read_x_ingestion_options as pegs_build_read_x_ingestion_options,
        build_read_x_post_options as pegs_build_read_x_post_options,
        initialize_matrix_and_gene_index_state as pegs_initialize_matrix_and_gene_index_state,
    )
    from pegs_shared.ydata import (
        sync_y_state as pegs_sync_y_state,
        sync_hyperparameter_state as pegs_sync_hyperparameter_state,
        sync_phewas_runtime_state as pegs_sync_phewas_runtime_state,
        sync_runtime_state_bundle as pegs_sync_runtime_state_bundle,
    )
    from .pegs_utils import (
        is_huge_statistics_bundle_path as pegs_is_huge_statistics_bundle_path,
        coerce_runtime_state_dict as pegs_coerce_runtime_state_dict,
        get_huge_statistics_paths_for_prefix as pegs_get_huge_statistics_paths_for_prefix,
        write_numeric_vector_file as pegs_write_numeric_vector_file,
        read_numeric_vector_file as pegs_read_numeric_vector_file,
        build_huge_statistics_matrix_row_genes as pegs_build_huge_statistics_matrix_row_genes,
        build_huge_statistics_score_maps as pegs_build_huge_statistics_score_maps,
        build_huge_statistics_meta as pegs_build_huge_statistics_meta,
        write_huge_statistics_text_tables as pegs_write_huge_statistics_text_tables,
        read_huge_statistics_text_tables as pegs_read_huge_statistics_text_tables,
        resolve_huge_statistics_gene_vectors as pegs_resolve_huge_statistics_gene_vectors,
        read_huge_statistics_covariates_if_present as pegs_read_huge_statistics_covariates_if_present,
        load_huge_statistics_sparse_and_vectors as pegs_load_huge_statistics_sparse_and_vectors,
        apply_huge_statistics_meta_to_runtime as pegs_apply_huge_statistics_meta_to_runtime,
        combine_runtime_huge_scores as pegs_combine_runtime_huge_scores,
        validate_huge_statistics_loaded_shapes as pegs_validate_huge_statistics_loaded_shapes,
        write_huge_statistics_runtime_vectors as pegs_write_huge_statistics_runtime_vectors,
        write_huge_statistics_sparse_components as pegs_write_huge_statistics_sparse_components,
        initialize_read_x_batch_seed_state as pegs_initialize_read_x_batch_seed_state,
        initialize_filtered_gene_set_state as pegs_initialize_filtered_gene_set_state,
        maybe_prepare_filtered_correlation as pegs_maybe_prepare_filtered_correlation,
        resolve_read_x_run_logistic as pegs_resolve_read_x_run_logistic,
        record_read_x_counts as pegs_record_read_x_counts,
        standardize_qc_metrics_after_x_read as pegs_standardize_qc_metrics_after_x_read,
        maybe_correct_gene_set_betas_after_x_read as pegs_maybe_correct_gene_set_betas_after_x_read,
        maybe_limit_initial_gene_sets_by_p as pegs_maybe_limit_initial_gene_sets_by_p,
        maybe_prune_gene_sets_after_x_read as pegs_maybe_prune_gene_sets_after_x_read,
        initialize_hyper_defaults_after_x_read as pegs_initialize_hyper_defaults_after_x_read,
        maybe_adjust_overaggressive_p_filter_after_x_read as pegs_maybe_adjust_overaggressive_p_filter_after_x_read,
        apply_post_read_gene_set_size_and_qc_filters as pegs_apply_post_read_gene_set_size_and_qc_filters,
        prepare_read_x_inputs as pegs_prepare_read_x_inputs,
        build_read_x_pipeline_config as pegs_build_read_x_pipeline_config,
        build_xin_to_p_noninf_index_map as pegs_build_xin_to_p_noninf_index_map,
        load_aligned_gene_bfs as pegs_load_aligned_gene_bfs,
        load_aligned_gene_covariates as pegs_load_aligned_gene_covariates,
        load_and_apply_gene_phewas_bfs_to_runtime as pegs_load_and_apply_gene_phewas_bfs_to_runtime,
        load_and_apply_gene_set_statistics_to_runtime as pegs_load_and_apply_gene_set_statistics_to_runtime,
        set_runtime_y_from_inputs as pegs_set_runtime_y_from_inputs,
        compute_banded_y_corr_cholesky as pegs_compute_banded_y_corr_cholesky,
        whiten_matrix_with_banded_cholesky as pegs_whiten_matrix_with_banded_cholesky,
        calc_shift_scale_for_dense_block as pegs_calc_shift_scale_for_dense_block,
        calc_X_shift_scale as pegs_calc_X_shift_scale,
        calculate_V_internal as pegs_calculate_V_internal,
        set_runtime_x_from_inputs as pegs_set_runtime_x_from_inputs,
        get_num_X_blocks as pegs_get_num_X_blocks,
        iterate_X_blocks_internal as pegs_iterate_X_blocks_internal,
        set_runtime_p as pegs_set_runtime_p,
        set_runtime_sigma as pegs_set_runtime_sigma,
        prepare_phewas_phenos_from_file as pegs_prepare_phewas_phenos_from_file,
        read_phewas_file_batch as pegs_read_phewas_file_batch,
        append_phewas_metric_block as pegs_append_phewas_metric_block,
        accumulate_standard_phewas_outputs as pegs_accumulate_standard_phewas_outputs,
        write_gene_set_statistics as pegs_write_gene_set_statistics,
        write_phewas_gene_set_statistics as pegs_write_phewas_gene_set_statistics,
        write_gene_statistics as pegs_write_gene_statistics,
        write_gene_gene_set_statistics as pegs_write_gene_gene_set_statistics,
        write_phewas_statistics as pegs_write_phewas_statistics,
        finalize_regression_outputs as pegs_finalize_regression_outputs,
        compute_beta_tildes as pegs_compute_beta_tildes,
        compute_logistic_beta_tildes as pegs_compute_logistic_beta_tildes,
        correct_beta_tildes as pegs_correct_beta_tildes,
        compute_multivariate_beta_tildes as pegs_compute_multivariate_beta_tildes,
        build_phewas_stage_config as pegs_build_phewas_stage_config,
        resolve_gene_phewas_input_decision_for_stage as pegs_resolve_gene_phewas_input_decision_for_stage,
        remove_tag_from_input as pegs_remove_tag_from_input,
        clean_chrom_name as pegs_clean_chrom_name,
        parse_gene_map_file as pegs_parse_gene_map_file,
        read_loc_file_with_gene_map as pegs_read_loc_file_with_gene_map,
        infer_columns_from_table_file as pegs_infer_columns_from_table_file,
        needs_gwas_column_detection as pegs_needs_gwas_column_detection,
        autodetect_gwas_columns as pegs_autodetect_gwas_columns,
        complete_p_beta_se as pegs_complete_p_beta_se,
        construct_map_to_ind as pegs_construct_map_to_ind,
        open_text_with_retry as pegs_open_text_with_retry,
        require_existing_nonempty_file as pegs_require_existing_nonempty_file,
        resolve_column_index as pegs_resolve_column_index,
        write_bundle_from_specs as pegs_write_bundle_from_specs,
        write_prefixed_tar_bundle as pegs_write_prefixed_tar_bundle,
        read_prefixed_tar_bundle as pegs_read_prefixed_tar_bundle,
        EAGGL_BUNDLE_SCHEMA as PEGS_EAGGL_BUNDLE_SCHEMA,
    )
except ImportError:
    from pegs_shared.types import (
        XReadConfig as PegsXReadConfig,
        XReadCallbacks as PegsXReadCallbacks,
        XReadPostCallbacks as PegsXReadPostCallbacks,
    )
    from pegs_cli_errors import (
        DataValidationError,
        PegsCliError,
        handle_cli_exception as pegs_handle_cli_exception,
        handle_unexpected_exception as pegs_handle_unexpected_exception,
    )
    from pegs_shared.cli import (
        apply_cli_config_overrides as pegs_apply_cli_config_overrides,
        callback_set_comma_separated_args as pegs_callback_set_comma_separated_args,
        callback_set_comma_separated_args_as_float as pegs_callback_set_comma_separated_args_as_float,
        coerce_option_int_list as pegs_coerce_option_int_list,
        configure_random_seed as pegs_configure_random_seed,
        emit_stderr_warning as pegs_emit_stderr_warning,
        fail_removed_cli_aliases as pegs_fail_removed_cli_aliases,
        format_removed_option_message as pegs_format_removed_option_message,
        get_tar_write_mode_for_bundle_path as pegs_get_tar_write_mode_for_bundle_path,
        harmonize_cli_mode_args as pegs_harmonize_cli_mode_args,
        initialize_cli_logging as pegs_initialize_cli_logging,
        is_path_like_dest as pegs_is_path_like_dest,
        iter_parser_options as pegs_iter_parser_options,
        json_safe as pegs_json_safe,
        load_json_config as pegs_load_json_config,
        merge_dicts as pegs_merge_dicts,
        resolve_config_path_value as pegs_resolve_config_path_value,
    )
    from pegs_shared.xdata import (
        xdata_from_input_plan as pegs_xdata_from_input_plan,
        build_read_x_ingestion_options as pegs_build_read_x_ingestion_options,
        build_read_x_post_options as pegs_build_read_x_post_options,
        initialize_matrix_and_gene_index_state as pegs_initialize_matrix_and_gene_index_state,
    )
    from pegs_shared.ydata import (
        sync_y_state as pegs_sync_y_state,
        sync_hyperparameter_state as pegs_sync_hyperparameter_state,
        sync_phewas_runtime_state as pegs_sync_phewas_runtime_state,
        sync_runtime_state_bundle as pegs_sync_runtime_state_bundle,
    )
    from pegs_utils import (
        is_huge_statistics_bundle_path as pegs_is_huge_statistics_bundle_path,
        coerce_runtime_state_dict as pegs_coerce_runtime_state_dict,
        get_huge_statistics_paths_for_prefix as pegs_get_huge_statistics_paths_for_prefix,
        write_numeric_vector_file as pegs_write_numeric_vector_file,
        read_numeric_vector_file as pegs_read_numeric_vector_file,
        build_huge_statistics_matrix_row_genes as pegs_build_huge_statistics_matrix_row_genes,
        build_huge_statistics_score_maps as pegs_build_huge_statistics_score_maps,
        build_huge_statistics_meta as pegs_build_huge_statistics_meta,
        write_huge_statistics_text_tables as pegs_write_huge_statistics_text_tables,
        read_huge_statistics_text_tables as pegs_read_huge_statistics_text_tables,
        resolve_huge_statistics_gene_vectors as pegs_resolve_huge_statistics_gene_vectors,
        read_huge_statistics_covariates_if_present as pegs_read_huge_statistics_covariates_if_present,
        load_huge_statistics_sparse_and_vectors as pegs_load_huge_statistics_sparse_and_vectors,
        apply_huge_statistics_meta_to_runtime as pegs_apply_huge_statistics_meta_to_runtime,
        combine_runtime_huge_scores as pegs_combine_runtime_huge_scores,
        validate_huge_statistics_loaded_shapes as pegs_validate_huge_statistics_loaded_shapes,
        write_huge_statistics_runtime_vectors as pegs_write_huge_statistics_runtime_vectors,
        write_huge_statistics_sparse_components as pegs_write_huge_statistics_sparse_components,
        initialize_read_x_batch_seed_state as pegs_initialize_read_x_batch_seed_state,
        initialize_filtered_gene_set_state as pegs_initialize_filtered_gene_set_state,
        maybe_prepare_filtered_correlation as pegs_maybe_prepare_filtered_correlation,
        resolve_read_x_run_logistic as pegs_resolve_read_x_run_logistic,
        record_read_x_counts as pegs_record_read_x_counts,
        standardize_qc_metrics_after_x_read as pegs_standardize_qc_metrics_after_x_read,
        maybe_correct_gene_set_betas_after_x_read as pegs_maybe_correct_gene_set_betas_after_x_read,
        maybe_limit_initial_gene_sets_by_p as pegs_maybe_limit_initial_gene_sets_by_p,
        maybe_prune_gene_sets_after_x_read as pegs_maybe_prune_gene_sets_after_x_read,
        initialize_hyper_defaults_after_x_read as pegs_initialize_hyper_defaults_after_x_read,
        maybe_adjust_overaggressive_p_filter_after_x_read as pegs_maybe_adjust_overaggressive_p_filter_after_x_read,
        apply_post_read_gene_set_size_and_qc_filters as pegs_apply_post_read_gene_set_size_and_qc_filters,
        prepare_read_x_inputs as pegs_prepare_read_x_inputs,
        build_read_x_pipeline_config as pegs_build_read_x_pipeline_config,
        build_xin_to_p_noninf_index_map as pegs_build_xin_to_p_noninf_index_map,
        load_aligned_gene_bfs as pegs_load_aligned_gene_bfs,
        load_aligned_gene_covariates as pegs_load_aligned_gene_covariates,
        load_and_apply_gene_phewas_bfs_to_runtime as pegs_load_and_apply_gene_phewas_bfs_to_runtime,
        load_and_apply_gene_set_statistics_to_runtime as pegs_load_and_apply_gene_set_statistics_to_runtime,
        set_runtime_y_from_inputs as pegs_set_runtime_y_from_inputs,
        compute_banded_y_corr_cholesky as pegs_compute_banded_y_corr_cholesky,
        whiten_matrix_with_banded_cholesky as pegs_whiten_matrix_with_banded_cholesky,
        calc_shift_scale_for_dense_block as pegs_calc_shift_scale_for_dense_block,
        calc_X_shift_scale as pegs_calc_X_shift_scale,
        calculate_V_internal as pegs_calculate_V_internal,
        set_runtime_x_from_inputs as pegs_set_runtime_x_from_inputs,
        get_num_X_blocks as pegs_get_num_X_blocks,
        iterate_X_blocks_internal as pegs_iterate_X_blocks_internal,
        set_runtime_p as pegs_set_runtime_p,
        set_runtime_sigma as pegs_set_runtime_sigma,
        prepare_phewas_phenos_from_file as pegs_prepare_phewas_phenos_from_file,
        read_phewas_file_batch as pegs_read_phewas_file_batch,
        append_phewas_metric_block as pegs_append_phewas_metric_block,
        accumulate_standard_phewas_outputs as pegs_accumulate_standard_phewas_outputs,
        write_gene_set_statistics as pegs_write_gene_set_statistics,
        write_phewas_gene_set_statistics as pegs_write_phewas_gene_set_statistics,
        write_gene_statistics as pegs_write_gene_statistics,
        write_gene_gene_set_statistics as pegs_write_gene_gene_set_statistics,
        write_phewas_statistics as pegs_write_phewas_statistics,
        finalize_regression_outputs as pegs_finalize_regression_outputs,
        compute_beta_tildes as pegs_compute_beta_tildes,
        compute_logistic_beta_tildes as pegs_compute_logistic_beta_tildes,
        correct_beta_tildes as pegs_correct_beta_tildes,
        compute_multivariate_beta_tildes as pegs_compute_multivariate_beta_tildes,
        build_phewas_stage_config as pegs_build_phewas_stage_config,
        resolve_gene_phewas_input_decision_for_stage as pegs_resolve_gene_phewas_input_decision_for_stage,
        remove_tag_from_input as pegs_remove_tag_from_input,
        clean_chrom_name as pegs_clean_chrom_name,
        parse_gene_map_file as pegs_parse_gene_map_file,
        read_loc_file_with_gene_map as pegs_read_loc_file_with_gene_map,
        infer_columns_from_table_file as pegs_infer_columns_from_table_file,
        needs_gwas_column_detection as pegs_needs_gwas_column_detection,
        autodetect_gwas_columns as pegs_autodetect_gwas_columns,
        complete_p_beta_se as pegs_complete_p_beta_se,
        construct_map_to_ind as pegs_construct_map_to_ind,
        open_text_with_retry as pegs_open_text_with_retry,
        require_existing_nonempty_file as pegs_require_existing_nonempty_file,
        resolve_column_index as pegs_resolve_column_index,
        write_bundle_from_specs as pegs_write_bundle_from_specs,
        write_prefixed_tar_bundle as pegs_write_prefixed_tar_bundle,
        read_prefixed_tar_bundle as pegs_read_prefixed_tar_bundle,
        EAGGL_BUNDLE_SCHEMA as PEGS_EAGGL_BUNDLE_SCHEMA,
    )

try:
    from pegs_shared.phewas import (
        build_phewas_stage_config as pegs_build_phewas_stage_config,
        resolve_gene_phewas_input_decision_for_stage as pegs_resolve_gene_phewas_input_decision_for_stage,
    )
    from pegs_shared.bundle import (
        get_tar_write_mode_for_bundle_path as pegs_get_tar_write_mode_for_bundle_path,
        require_existing_nonempty_file as pegs_require_existing_nonempty_file,
        write_bundle_from_specs as pegs_write_bundle_from_specs,
        write_prefixed_tar_bundle as pegs_write_prefixed_tar_bundle,
        read_prefixed_tar_bundle as pegs_read_prefixed_tar_bundle,
        EAGGL_BUNDLE_SCHEMA as PEGS_EAGGL_BUNDLE_SCHEMA,
    )
except ImportError:
    from pegs_shared.phewas import (  # type: ignore
        build_phewas_stage_config as pegs_build_phewas_stage_config,
        resolve_gene_phewas_input_decision_for_stage as pegs_resolve_gene_phewas_input_decision_for_stage,
    )
    from pegs_shared.bundle import (  # type: ignore
        get_tar_write_mode_for_bundle_path as pegs_get_tar_write_mode_for_bundle_path,
        require_existing_nonempty_file as pegs_require_existing_nonempty_file,
        write_bundle_from_specs as pegs_write_bundle_from_specs,
        write_prefixed_tar_bundle as pegs_write_prefixed_tar_bundle,
        read_prefixed_tar_bundle as pegs_read_prefixed_tar_bundle,
        EAGGL_BUNDLE_SCHEMA as PEGS_EAGGL_BUNDLE_SCHEMA,
    )

# Canonical suffix tags used when expanding dense gene-set inputs into
# sparse derived sets (top/ext/bottom thresholds).
EXT_TAG = "ext"
BOT_TAG = "bot"
TOP_TAG = "top"

def bail(message):
    raise DataValidationError(message)

try:
    from . import pigean_cli as _pigean_cli
except ImportError:
    import pigean_cli as _pigean_cli

usage = _pigean_cli.usage
parser = _pigean_cli.parser
REMOVED_OPTION_REPLACEMENTS = _pigean_cli.REMOVED_OPTION_REPLACEMENTS
_build_mode_state = _pigean_cli._build_mode_state
_json_safe = _pigean_cli._json_safe

options = None
args = []
mode = None
config_mode = None
cli_specified_dests = set()
config_specified_dests = set()
NONE = 0
INFO = 1
DEBUG = 2
TRACE = 3
debug_level = 1
log_fh = None
warnings_fh = None


def _noop_log(*_args, **_kwargs):
    return None


log = _noop_log
warn = _noop_log

def _build_runtime_state(_options):
    state = PigeanState(background_prior=_options.background_prior, batch_size=_options.batch_size)
    state.debug_old_batch = _options.debug_old_batch
    state.debug_skip_correlation = _options.debug_skip_correlation
    state.debug_skip_phewas_covs = _options.debug_skip_phewas_covs
    state.debug_only_avg_huge = _options.debug_only_avg_huge
    state.debug_just_check_header = _options.debug_just_check_header
    return state


# State-field groups used to make temporary overrides explicit in hot paths.
_STATE_FIELDS_X_INDEXING = (
    "X_orig",
    "X_orig_missing_genes",
    "X_orig_missing_gene_sets",
    "genes",
    "genes_missing",
    "gene_sets",
    "gene_sets_missing",
    "scale_factors",
    "mean_shifts",
)
_STATE_FIELDS_Y_SOURCES = (
    "Y",
    "Y_for_regression",
    "Y_uncorrected",
    "gene_to_huge_score",
    "gene_to_gwas_huge_score",
    "gene_to_gwas_huge_score_uncorrected",
    "gene_to_exomes_huge_score",
)
_STATE_FIELDS_COVARIATE_CORRECTION = (
    "gene_covariates",
    "gene_covariates_mask",
    "gene_covariate_names",
    "gene_covariate_directions",
    "gene_covariate_intercept_index",
    "gene_covariates_mat_inv",
    "gene_covariate_zs",
    "gene_covariate_adjustments",
)
_STATE_FIELDS_SAMPLER_HYPER = (
    "p",
    "ps",
    "sigma2",
    "sigma2s",
    "sigma_power",
)


def _snapshot_state_fields(state, field_names):
    return {field_name: getattr(state, field_name) for field_name in field_names}


def _restore_state_fields(state, snapshot):
    for field_name, field_value in snapshot.items():
        setattr(state, field_name, field_value)


@contextlib.contextmanager
def _temporary_state_fields(state, overrides, restore_fields):
    snapshot = _snapshot_state_fields(state, restore_fields)
    for field_name, field_value in overrides.items():
        setattr(state, field_name, field_value)
    try:
        yield snapshot
    finally:
        _restore_state_fields(state, snapshot)


@contextlib.contextmanager
def _open_optional_gibbs_trace_files(gene_set_stats_trace_out, gene_stats_trace_out):
    with contextlib.ExitStack() as stack:
        gene_set_stats_trace_fh = None
        gene_stats_trace_fh = None
        if gene_set_stats_trace_out is not None:
            gene_set_stats_trace_fh = stack.enter_context(open_gz(gene_set_stats_trace_out, "w"))
            gene_set_stats_trace_fh.write(
                "It\tChain\tGene_Set\tbeta_tilde\tP\tZ\tSE\tbeta_uncorrected\tbeta\tpostp\tbeta_tilde_outlier_z\tR\tSEM\n"
            )
        if gene_stats_trace_out is not None:
            gene_stats_trace_fh = stack.enter_context(open_gz(gene_stats_trace_out, "w"))
            gene_stats_trace_fh.write("It\tChain\tGene\tprior\tcombined\tlog_bf\tD\tpercent_top\tadjust\n")
        yield (gene_set_stats_trace_fh, gene_stats_trace_fh)


def _open_optional_inner_betas_trace_file(betas_trace_out):
    if betas_trace_out is None:
        return None
    betas_trace_fh = open_gz(betas_trace_out, "w")
    betas_trace_fh.write(
        "It\tParallel\tChain\tGene_Set\tbeta_post\tbeta\tpostp\tres_beta_hat\tbeta_tilde\tbeta_internal\tres_beta_hat_internal\tbeta_tilde_internal\tse_internal\tsigma2\tp\tR\tR_weighted\tSEM\n"
    )
    return betas_trace_fh


def _close_optional_inner_betas_trace_file(betas_trace_fh):
    if betas_trace_fh is not None:
        betas_trace_fh.close()


def _return_inner_betas_result(betas_trace_fh, result):
    _close_optional_inner_betas_trace_file(betas_trace_fh)
    return result


def _maybe_unsubset_gene_sets(state, enabled, skip_V=False, skip_scale_factors=False):
    if not enabled:
        return None
    return state._unsubset_gene_sets(skip_V=skip_V, skip_scale_factors=skip_scale_factors)


def _restore_subset_gene_sets(state, subset_mask, keep_missing=True, skip_V=False, skip_scale_factors=False):
    if subset_mask is None:
        return
    state.subset_gene_sets(
        subset_mask,
        keep_missing=keep_missing,
        skip_V=skip_V,
        skip_scale_factors=skip_scale_factors,
    )


@contextlib.contextmanager
def _temporary_unsubset_gene_sets(
    state,
    enabled,
    keep_missing=True,
    skip_V=False,
    skip_scale_factors=False,
):
    subset_mask = _maybe_unsubset_gene_sets(
        state,
        enabled,
        skip_V=skip_V,
        skip_scale_factors=skip_scale_factors,
    )
    try:
        yield subset_mask
    finally:
        _restore_subset_gene_sets(
            state,
            subset_mask,
            keep_missing=keep_missing,
            skip_V=skip_V,
            skip_scale_factors=skip_scale_factors,
        )


def _bootstrap_cli(argv=None):
    global options, args, mode, config_mode, cli_specified_dests, config_specified_dests
    global NONE, INFO, DEBUG, TRACE, debug_level, log_fh, warnings_fh, log, warn

    should_continue = _pigean_cli._bootstrap_cli(argv)
    options = _pigean_cli.options
    args = _pigean_cli.args
    mode = _pigean_cli.mode
    config_mode = _pigean_cli.config_mode
    cli_specified_dests = _pigean_cli.cli_specified_dests
    config_specified_dests = _pigean_cli.config_specified_dests
    NONE = _pigean_cli.NONE
    INFO = _pigean_cli.INFO
    DEBUG = _pigean_cli.DEBUG
    TRACE = _pigean_cli.TRACE
    debug_level = _pigean_cli.debug_level
    log_fh = _pigean_cli.log_fh
    warnings_fh = _pigean_cli.warnings_fh
    log = _pigean_cli.log
    warn = _pigean_cli.warn
    return should_continue


def open_gz(file, flag=None):
    return pegs_open_text_with_retry(
        file,
        flag=flag,
        log_fn=lambda message: log(message, INFO),
        bail_fn=bail,
    )


_HYPERPARAMETER_PROXY_FIELDS = (
    "p",
    "sigma2",
    "sigma_power",
    "sigma2_osc",
    "sigma2_se",
    "sigma2_p",
    "sigma2_total_var",
    "sigma2_total_var_lower",
    "sigma2_total_var_upper",
    "ps",
    "sigma2s",
    "sigma2s_missing",
)


def _bind_hyperparameter_properties(state_cls):
    for field_name in _HYPERPARAMETER_PROXY_FIELDS:
        private_name = "_%s" % field_name

        def _getter(self, _field=field_name, _private=private_name):
            hyper_state = self.__dict__.get("hyperparameter_state")
            if hyper_state is not None:
                return getattr(hyper_state, _field)
            return self.__dict__.get(_private, None)

        def _setter(self, value, _field=field_name, _private=private_name):
            self.__dict__[_private] = value
            hyper_state = self.__dict__.get("hyperparameter_state")
            if hyper_state is not None:
                setattr(hyper_state, _field, value)

        setattr(state_cls, field_name, property(_getter, _setter))


class PigeanState(object):
    '''
    Stores gene and gene set annotations and derived matrices
    It allows reading X or V files and using these to determine the allowed gene sets and genes
    '''
    def __init__(self, background_prior=0.05, batch_size=4500):

        #empirical mean scale factor from mice
        self.MEAN_MOUSE_SCALE = 0.0448373

        if background_prior <= 0 or background_prior >= 1:
            bail("--background-prior must be in (0,1)")
        self.background_prior = background_prior
        self.background_log_bf = np.log(self.background_prior / (1 - self.background_prior))
        self.background_bf = np.exp(self.background_log_bf)
        self.debug_old_batch = False
        self.debug_skip_correlation = False
        self.debug_skip_phewas_covs = False
        self.debug_only_avg_huge = False
        self.debug_just_check_header = False

        pegs_initialize_matrix_and_gene_index_state(self, batch_size=batch_size)
        self._init_phewas_and_label_state()
        self._init_gene_set_regression_state()
        self._init_gene_signal_and_huge_state()
        self._init_model_summary_state()
        self.runtime_state_bundle = pegs_sync_runtime_state_bundle(self)
        self.y_state = self.runtime_state_bundle.y_state
        self.hyperparameter_state = self.runtime_state_bundle.hyperparameter_state
        self.phewas_state = self.runtime_state_bundle.phewas_state

    def _init_phewas_and_label_state(self):
        self.anchor_pheno_mask = None
        self.anchor_gene_mask = None

        self.default_pheno_mask = None

        #how phewas was read in in case we need to redo it
        self.cached_gene_phewas_call = None

        self.gene_pheno_combined_prior_Ys = None
        self.gene_pheno_Y = None
        self.gene_pheno_priors = None

        self.num_gene_phewas_filtered = 0

        #note that these phewas betas are all stored in *external* units (by contrast to the betas which are in internal units)

        #these are for phewas. Regression of pheno values (combined, Y, or priors) against input Y values

        self.pheno_Y_vs_input_Y_beta = None
        self.pheno_Y_vs_input_Y_beta_tilde = None
        self.pheno_Y_vs_input_Y_se = None
        self.pheno_Y_vs_input_Y_Z = None
        self.pheno_Y_vs_input_Y_p_value = None

        self.pheno_combined_prior_Ys_vs_input_Y_beta = None
        self.pheno_combined_prior_Ys_vs_input_Y_beta_tilde = None
        self.pheno_combined_prior_Ys_vs_input_Y_se = None
        self.pheno_combined_prior_Ys_vs_input_Y_Z = None
        self.pheno_combined_prior_Ys_vs_input_Y_p_value = None

        #these are for phewas. Regression of pheno values (combined, Y, or priors) against input combined values
        self.pheno_Y_vs_input_combined_prior_Ys_beta = None
        self.pheno_Y_vs_input_combined_prior_Ys_beta_tilde = None
        self.pheno_Y_vs_input_combined_prior_Ys_se = None
        self.pheno_Y_vs_input_combined_prior_Ys_Z = None
        self.pheno_Y_vs_input_combined_prior_Ys_p_value = None

        self.pheno_combined_prior_Ys_vs_input_combined_prior_Ys_beta = None
        self.pheno_combined_prior_Ys_vs_input_combined_prior_Ys_beta_tilde = None
        self.pheno_combined_prior_Ys_vs_input_combined_prior_Ys_se = None
        self.pheno_combined_prior_Ys_vs_input_combined_prior_Ys_Z = None
        self.pheno_combined_prior_Ys_vs_input_combined_prior_Ys_p_value = None

        #these are for phewas. Regression of pheno values (combined, Y, or priors) against input prior values
        self.pheno_Y_vs_input_priors_beta = None
        self.pheno_Y_vs_input_priors_beta_tilde = None
        self.pheno_Y_vs_input_priors_se = None
        self.pheno_Y_vs_input_priors_Z = None
        self.pheno_Y_vs_input_priors_p_value = None

        self.pheno_combined_prior_Ys_vs_input_priors_beta = None
        self.pheno_combined_prior_Ys_vs_input_priors_beta_tilde = None
        self.pheno_combined_prior_Ys_vs_input_priors_se = None
        self.pheno_combined_prior_Ys_vs_input_priors_Z = None
        self.pheno_combined_prior_Ys_vs_input_priors_p_value = None

        self.gene_to_positive_controls = None
        self.gene_to_case_count_logbf = None

        self.gene_label_map = None

        #only used for running factoring by phenotype
        self.default_pheno = "__default__"
        self.phenos = None
        self.pheno_to_ind = None
        self.phewas_state = pegs_sync_phewas_runtime_state(self)

        #note that these phewas betas are all stored in *external* units (by contrast to the betas which are in internal units)
        self.X_phewas_beta_uncorrected = None
        self.X_phewas_beta = None

    def _init_gene_set_regression_state(self):
        #ordered list of gene sets
        self.gene_sets = None
        self.gene_sets_missing = None
        self.gene_sets_ignored = None
        self.gene_set_to_ind = None

        #gene set association statistics
        #self.max_gene_set_p = None

        #self.is_logistic = None

        self.beta_tildes = None
        self.p_values = None
        self.ses = None
        self.z_scores = None

        self.beta_tildes_phewas = None
        self.p_values_phewas = None
        self.ses_phewas = None
        self.z_scores_phewas = None

        self.beta_tildes_orig = None
        self.p_values_orig = None
        self.ses_orig = None
        self.z_scores_orig = None

        #these store the inflation of SE relative to OLS (if ols_corrected is run)
        self.se_inflation_factors = None
        self.se_inflation_factors_phewas = None

        #these are gene sets we filtered out but need to persist for OSC
        self.beta_tildes_missing = None
        self.p_values_missing = None
        self.ses_missing = None
        self.z_scores_missing = None
        self.se_inflation_factors_missing = None

        #these are gene sets we ignored at the start
        self.col_sums_ignored = None

        self.beta_tildes_ignored = None
        self.p_values_ignored = None
        self.ses_ignored = None
        self.z_scores_ignored = None
        self.se_inflation_factors_ignored = None

        self.beta_tildes_missing_orig = None
        self.p_values_missing_orig = None
        self.ses_missing_orig = None
        self.z_scores_missing_orig = None

    def _init_gene_signal_and_huge_state(self):
        #DO WE NEED THIS???
        #self.y_mean = None
        self.Y = None
        self.Y_exomes = None
        self.Y_positive_controls = None
        self.Y_case_counts = None

        #this is to store altered variables if we detect power
        #these are used for fitting the betas (the indirect support)
        #self.Y is the direct support and is used only for combining with the indirect support to get a D value for this gene
        self.Y_for_regression = None

        #this is where to store the original uncorrected Y values if we have them
        self.Y_uncorrected = None

        self.y_var = 1 #total variance of the Y
        self.Y_orig = None
        self.Y_for_regression_orig = None

        self.gene_locations = None #this stores sort orders for genes, which is populated when fitting correlation matrix from gene loc file

        self.huge_signal_bfs = None
        self.huge_signal_bfs_for_regression = None

        #covariates for genes
        self.gene_covariates = None
        self.gene_covariates_mask = None
        self.gene_covariate_names = None
        self.gene_covariate_directions = None
        self.gene_covariate_intercept_index = None
        self.gene_covariates_mat_inv = None
        self.gene_covariate_zs = None
        self.gene_covariate_adjustments = None

        #for sparse mode
        self.huge_sparse_mode = False
        self.gene_covariate_slope_defaults = None
        self.total_qc_metric_betas_defaults = None
        self.total_qc_metric_intercept_defaults = None
        self.total_qc_metric2_betas_defaults = None
        self.total_qc_metric2_intercept_defaults = None

        self.total_qc_metric_betas = None
        self.total_qc_metric_intercept = None
        self.total_qc_metric2_betas = None
        self.total_qc_metric2_intercept = None
        self.total_qc_metric_desired_var = None

        self.huge_signals = None
        self.huge_signal_posteriors = None
        self.huge_signal_posteriors_for_regression = None
        self.huge_signal_sum_gene_cond_probabilities = None
        self.huge_signal_sum_gene_cond_probabilities_for_regression = None
        self.huge_signal_mean_gene_pos = None
        self.huge_signal_mean_gene_pos_for_regression = None
        self.huge_signal_max_closest_gene_prob = None

        self.huge_cap_region_posterior = True
        self.huge_scale_region_posterior = False
        self.huge_phantom_region_posterior = False
        self.huge_allow_evidence_of_absence = False

        self.y_corr = None #this stores the (banded) correlation matrix for the Y values
        #In addition to storing banded correlation matrix, this signals that we are in partial GLS mode (OLS with inflated SEs)
        self.y_corr_sparse = None #another representation of the banded correlation matrix
        #In addition to storing cholesky decomp, this being set to not None triggers everything to operate in full GLS mode
        self.y_corr_cholesky = None #this stores the cholesky decomposition of the (banded) correlation matrix for the Y values
        #these are the "whitened" ys that are multiplied by sigma^{-1/2}
        self.y_w_var = 1 #total variance of the whitened Y
        self.y_w_mean = 0 #total mean of the whitened Y
        #these are the "full whitened" ys that are multiplied by sigma^{-1}
        self.y_fw_var = 1 #total variance of the whitened Y
        self.y_fw_mean = 0 #total mean of the whitened Y

        #statistics for sigma regression
        self.osc = None
        self.X_osc = None
        self.osc_weights = None

        self.osc_missing = None
        self.X_osc_missing = None
        self.osc_weights_missing = None

        #statistics for gene set qc
        self.total_qc_metrics = None
        self.mean_qc_metrics = None

        self.total_qc_metrics_missing = None
        self.mean_qc_metrics_missing = None

        self.total_qc_metrics_ignored = None
        self.mean_qc_metrics_ignored = None

        self.total_qc_metrics_directions = None
        self.runtime_state_bundle = pegs_sync_runtime_state_bundle(self)
        self.y_state = self.runtime_state_bundle.y_state

    def _init_model_summary_state(self):
        self.p = None
        self.ps = None #this allows gene sets to have different ps
        self.ps_missing = None #this allows gene sets to have different ps
        self.sigma2 = None #sigma2 * np.power(scale_factor, sigma_power) is the prior used for the internal beta
        self.sigma2s = None #this allows gene sets to have different sigma2s
        self.sigma2s_missing = None #this allows gene sets to have different sigma2s

        self.sigma2_osc = None
        self.sigma2_se = None
        self.intercept = None
        self.sigma2_p = None
        self.sigma2_total_var = None
        self.sigma2_total_var_lower = None
        self.sigma2_total_var_upper = None

        #statistics for gene set betas
        self.betas = None
        self.betas_uncorrected = None
        self.betas_r_hat = None
        self.betas_mcse = None
        self.betas_uncorrected_r_hat = None
        self.betas_uncorrected_mcse = None
        self.non_inf_avg_cond_betas = None
        self.non_inf_avg_postps = None

        self.betas_phewas = None
        self.betas_uncorrected_phewas = None

        self.betas_missing = None
        self.betas_uncorrected_missing = None
        self.betas_r_hat_missing = None
        self.betas_mcse_missing = None
        self.betas_uncorrected_r_hat_missing = None
        self.betas_uncorrected_mcse_missing = None
        self.runtime_state_bundle = pegs_sync_runtime_state_bundle(self)
        self.hyperparameter_state = self.runtime_state_bundle.hyperparameter_state
        self.non_inf_avg_cond_betas_missing = None
        self.non_inf_avg_postps_missing = None

        self.betas_orig = None
        self.betas_uncorrected_orig = None
        self.non_inf_avg_cond_betas_orig = None
        self.non_inf_avg_postps_orig = None

        self.betas_missing_orig = None
        self.betas_uncorrected_missing_orig = None
        self.non_inf_avg_cond_betas_missing_orig = None
        self.non_inf_avg_postps_missing_orig = None

        #statistics for genes
        self.priors = None
        self.priors_r_hat = None
        self.priors_mcse = None
        self.priors_adj = None
        self.combined_prior_Ys = None
        self.combined_prior_Ys_r_hat = None
        self.combined_prior_Ys_mcse = None
        self.combined_prior_Ys_for_regression = None

        self.combined_prior_Ys_adj = None
        self.combined_prior_Y_ses = None
        self.combined_Ds = None
        self.combined_Ds_for_regression = None
        self.combined_Ds_missing = None
        self.Y_r_hat = None
        self.Y_mcse = None
        self.priors_missing = None
        self.priors_adj_missing = None

        self.gene_N = None
        self.gene_ignored_N = None #number of ignored gene sets gene is in

        self.gene_N_missing = None #gene_N for genes with missing values for Y
        self.gene_ignored_N_missing = None #gene_N_missing for genes with missing values for Y

        self.batches = None

        self.priors_orig = None
        self.priors_adj_orig = None
        self.priors_missing_orig = None
        self.priors_adj_missing_orig = None

        #model parameters
        self.sigma_power = None

        #soft thresholding of sigmas
        self.sigma_threshold_k = None
        self.sigma_threshold_xo = None

        #stores all parameters used
        self.params = {}
        self.param_keys = []

    def write_V(self, V_out):
        if self.X_orig is not None:
            V = self._get_V()
            log("Writing V matrix to %s" % V_out, INFO)
            np.savetxt(V_out, V, delimiter='\t', fmt="%.2g", comments="#", header="%s" % ("\t".join(self.gene_sets)))
        else:
            warn("V has not been initialized; skipping writing")

    def write_Xd(self, X_out):
        if self.X_orig is not None:
            log("Writing X matrix to %s" % X_out, INFO)
            np.savetxt(X_out, self.X_orig.toarray(), delimiter='\t', fmt="%.3g", comments="#", header="%s" % ("%s\n#%s" % ("\t".join(self.gene_sets), "\t".join(self.genes))))
        else:
            warn("X has not been initialized; skipping writing")

    def write_X(self, X_out):
        if self.genes is None or self.X_orig is None or self.gene_sets is None:
            warn("X has not been initialized; skipping writing")
            return

        log("Writing X sparse matrix to %s" % X_out, INFO)

        with open_gz(X_out, 'w') as output_fh:

            for j in range(len(self.gene_sets)):
                line = self.gene_sets[j]
                nonzero_inds = self.X_orig[:,j].nonzero()[0]
                non_unity = np.sum(self.X_orig[nonzero_inds,j] == 1) < len(nonzero_inds)
                for i in nonzero_inds:
                    if non_unity:
                        line = "%s\t%s:%.2g" % (line, self.genes[i], self.X_orig[i,j])
                    else:
                        line = "%s\t%s" % (line, self.genes[i])

                output_fh.write("%s\n" % line)

    def _initialize_huge_gwas_state(self):
        # Track per-signal HuGE matrices and per-gene covariates for this GWAS run.
        self.huge_signals = []
        self.huge_signal_posteriors = []
        self.huge_signal_posteriors_for_regression = []
        self.huge_signal_sum_gene_cond_probabilities = []
        self.huge_signal_sum_gene_cond_probabilities_for_regression = []
        self.huge_signal_mean_gene_pos = []
        self.huge_signal_mean_gene_pos_for_regression = []
        self.gene_covariates = None
        self.gene_covariates_mask = None
        self.gene_covariate_names = None
        self.gene_covariate_directions = None
        self.gene_covariate_intercept_index = None
        self.gene_covariate_adjustments = None

        return {
            "closest_dist_X": np.array([]),
            "closest_dist_Y": np.array([]),
            "var_all_p": np.array([]),
            "gene_bf_data": [],
            "gene_bf_data_detect": [],
            "gene_prob_rows": [],
            "gene_prob_rows_detect": [],
            "gene_prob_cols": [],
            "gene_prob_cols_detect": [],
            "gene_prob_genes": [],
            "gene_prob_col_num": 0,
            "gene_covariate_genes": [],
        }

    def _remap_huge_gene_probability_rows(self, gene_to_chrom, gene_prob_genes, gene_prob_rows, gene_prob_rows_detect):
        if self.genes is not None:
            genes = self.genes
            gene_to_ind = self.gene_to_ind
        else:
            genes = list(gene_to_chrom.keys())
            gene_to_ind = pegs_construct_map_to_ind(genes)

        # Remap sparse matrix row indices into the final gene ordering.
        extra_genes = []
        extra_gene_to_ind = {}
        for gene_prob_rows_to_process in [gene_prob_rows, gene_prob_rows_detect]:
            for i in range(len(gene_prob_rows_to_process)):
                cur_gene = gene_prob_genes[gene_prob_rows_to_process[i]]

                if cur_gene in gene_to_ind:
                    new_ind = gene_to_ind[cur_gene]
                elif cur_gene in extra_gene_to_ind:
                    new_ind = extra_gene_to_ind[cur_gene]
                else:
                    new_ind = len(extra_genes) + len(genes)
                    extra_genes.append(cur_gene)
                    extra_gene_to_ind[cur_gene] = new_ind
                gene_prob_rows_to_process[i] = new_ind

        # Ensure genes with no retained signal rows still exist in final output vectors.
        for cur_gene in list(gene_to_chrom.keys()) + gene_prob_genes:
            if cur_gene not in gene_to_ind and cur_gene not in extra_gene_to_ind:
                new_ind = len(extra_genes) + len(genes)
                extra_genes.append(cur_gene)
                extra_gene_to_ind[cur_gene] = new_ind

        gene_prob_gene_list = genes + extra_genes
        return (genes, gene_to_ind, extra_genes, extra_gene_to_ind, gene_prob_gene_list)

    def _align_huge_gene_covariates_to_gene_list(self, gene_prob_gene_list, gene_covariate_genes, gene_to_ind, extra_gene_to_ind):
        if self.gene_covariates is None:
            return

        # Sort covariates into the final gene order, filling missing genes with column means.
        sorted_gene_covariates = np.tile(
            np.nanmean(self.gene_covariates, axis=0),
            len(gene_prob_gene_list),
        ).reshape((len(gene_prob_gene_list), self.gene_covariates.shape[1]))

        for i in range(len(gene_covariate_genes)):
            cur_gene = gene_covariate_genes[i]
            assert(cur_gene in gene_to_ind or cur_gene in extra_gene_to_ind)

            if cur_gene in gene_to_ind:
                new_ind = gene_to_ind[cur_gene]
            else:
                new_ind = extra_gene_to_ind[cur_gene]
            noninf_mask = ~np.isnan(self.gene_covariates[i,:])
            sorted_gene_covariates[new_ind,noninf_mask] = self.gene_covariates[i,noninf_mask]

        self.gene_covariates = sorted_gene_covariates

    def _read_huge_s2g_probabilities(
        self,
        s2g_in,
        seen_chrom_pos,
        hold_out_chrom=None,
        s2g_chrom_col=None,
        s2g_pos_col=None,
        s2g_gene_col=None,
        s2g_prob_col=None,
        s2g_normalize_values=None,
    ):
        if s2g_in is None:
            return None

        chrom_pos_to_gene_prob = {}
        log("Reading --s2g-in file %s" % s2g_in, INFO)

        # See if need to determine.
        if s2g_pos_col is None or s2g_chrom_col is None or s2g_gene_col is None:
            (
                possible_s2g_gene_cols,
                possible_s2g_var_id_cols,
                possible_s2g_chrom_cols,
                possible_s2g_pos_cols,
                possible_s2g_locus_cols,
                possible_s2g_p_cols,
                possible_s2g_beta_cols,
                possible_s2g_se_cols,
                possible_s2g_freq_cols,
                possible_s2g_n_cols,
            ) = _determine_columns_from_file(s2g_in)

            if s2g_pos_col is None:
                if len(possible_s2g_pos_cols) == 1:
                    s2g_pos_col = possible_s2g_pos_cols[0]
                    log("Using %s for position column; change with --s2g-pos-col if incorrect" % s2g_pos_col)
                else:
                    bail("Could not determine position column; specify with --s2g-pos-col")
            if s2g_chrom_col is None:
                if len(possible_s2g_chrom_cols) == 1:
                    s2g_chrom_col = possible_s2g_chrom_cols[0]
                    log("Using %s for chromition column; change with --s2g-chrom-col if incorrect" % s2g_chrom_col)
                else:
                    bail("Could not determine chrom column; specify with --s2g-chrom-col")
            if s2g_gene_col is None:
                if len(possible_s2g_gene_cols) == 1:
                    s2g_gene_col = possible_s2g_gene_cols[0]
                    log("Using %s for geneition column; change with --s2g-gene-col if incorrect" % s2g_gene_col)
                else:
                    bail("Could not determine gene column; specify with --s2g-gene-col")

        with open_gz(s2g_in) as s2g_fh:
            header_cols = s2g_fh.readline().strip('\n').split()
            chrom_col = _get_col(s2g_chrom_col, header_cols)
            pos_col = _get_col(s2g_pos_col, header_cols)
            gene_col = _get_col(s2g_gene_col, header_cols)
            prob_col = None
            if s2g_prob_col is not None:
                prob_col = _get_col(s2g_prob_col, header_cols)

            for line in s2g_fh:
                cols = line.strip('\n').split()
                if chrom_col > len(cols) or pos_col > len(cols) or gene_col > len(cols) or (prob_col is not None and prob_col > len(cols)):
                    warn("Skipping due to too few columns in line: %s" % line)
                    continue

                chrom = pegs_clean_chrom_name(cols[chrom_col])
                if hold_out_chrom is not None and chrom == hold_out_chrom:
                    continue

                try:
                    pos = int(cols[pos_col])
                except ValueError:
                    warn("Skipping unconvertible pos value %s" % (cols[pos_col]))
                    continue
                gene = cols[gene_col]

                if self.gene_label_map is not None and gene in self.gene_label_map:
                    gene = self.gene_label_map[gene]

                max_s2g_prob = 0.95
                prob = max_s2g_prob
                if prob_col is not None:
                    try:
                        prob = float(cols[prob_col])
                    except ValueError:
                        warn("Skipping unconvertible prob value %s" % (cols[prob_col]))
                        continue
                if prob > max_s2g_prob:
                    prob = max_s2g_prob

                if chrom in seen_chrom_pos and pos in seen_chrom_pos[chrom]:
                    if chrom not in chrom_pos_to_gene_prob:
                        chrom_pos_to_gene_prob[chrom] = {}
                    if pos not in chrom_pos_to_gene_prob[chrom]:
                        chrom_pos_to_gene_prob[chrom][pos] = []
                    chrom_pos_to_gene_prob[chrom][pos].append((gene, prob))

            if s2g_normalize_values is not None:
                for chrom in chrom_pos_to_gene_prob:
                    for pos in chrom_pos_to_gene_prob[chrom]:
                        prob_sum = sum([x[1] for x in chrom_pos_to_gene_prob[chrom][pos]])
                        if prob_sum > 0:
                            norm_factor = s2g_normalize_values / prob_sum
                            chrom_pos_to_gene_prob[chrom][pos] = [(x[0], x[1] * norm_factor) for x in chrom_pos_to_gene_prob[chrom][pos]]

        return chrom_pos_to_gene_prob

    def _read_huge_input_credible_sets(
        self,
        credible_sets_in,
        seen_chrom_pos,
        chrom_pos_p_beta_se_freq,
        var_p_threshold,
        hold_out_chrom=None,
        credible_sets_id_col=None,
        credible_sets_chrom_col=None,
        credible_sets_pos_col=None,
        credible_sets_ppa_col=None,
    ):
        added_chrom_pos = {}
        input_credible_set_info = {}
        if credible_sets_in is None:
            return (added_chrom_pos, input_credible_set_info)

        log("Reading --credible-sets-in file %s" % credible_sets_in, INFO)

        # See if need to determine.
        if credible_sets_pos_col is None or credible_sets_chrom_col is None:
            (_, _, possible_credible_sets_chrom_cols, possible_credible_sets_pos_cols, _, _, _, _, _, _, header) = _determine_columns_from_file(credible_sets_in)

            if credible_sets_pos_col is None:
                if len(possible_credible_sets_pos_cols) == 1:
                    credible_sets_pos_col = possible_credible_sets_pos_cols[0]
                    log("Using %s for position column; change with --credible-sets-pos-col if incorrect" % credible_sets_pos_col)
                else:
                    bail("Could not determine position column; specify with --credible-sets-pos-col")
            if credible_sets_chrom_col is None:
                if len(possible_credible_sets_chrom_cols) == 1:
                    credible_sets_chrom_col = possible_credible_sets_chrom_cols[0]
                    log("Using %s for chromition column; change with --credible-sets-chrom-col if incorrect" % credible_sets_chrom_col)
                else:
                    bail("Could not determine chrom column; specify with --credible-sets-chrom-col")

        with open_gz(credible_sets_in) as credible_sets_fh:
            header_cols = credible_sets_fh.readline().strip('\n').split()
            chrom_col = _get_col(credible_sets_chrom_col, header_cols)
            pos_col = _get_col(credible_sets_pos_col, header_cols)
            id_col = None
            if credible_sets_id_col is not None:
                id_col = _get_col(credible_sets_id_col, header_cols)
            ppa_col = None
            if credible_sets_ppa_col is not None:
                ppa_col = _get_col(credible_sets_ppa_col, header_cols)

            for line in credible_sets_fh:
                cols = line.strip('\n').split()
                if (id_col is not None and id_col > len(cols)) or (chrom_col is not None and chrom_col > len(cols)) or (pos_col is not None and pos_col > len(cols)) or (ppa_col is not None and ppa_col > len(cols)):
                    warn("Skipping due to too few columns in line: %s" % line)
                    continue

                chrom = pegs_clean_chrom_name(cols[chrom_col])

                if hold_out_chrom is not None and chrom == hold_out_chrom:
                    continue

                try:
                    pos = int(cols[pos_col])
                except ValueError:
                    warn("Skipping unconvertible pos value %s" % (cols[pos_col]))
                    continue

                if id_col is not None:
                    cs_id = cols[id_col]
                else:
                    cs_id = "%s:%s" % (chrom, pos)

                ppa = None
                if ppa_col is not None:
                    try:
                        ppa = float(cols[ppa_col])
                        if ppa > 1:
                            ppa = 0.99
                        elif ppa < 0:
                            ppa = 0
                    except ValueError:
                        warn("Skipping unconvertible ppa value %s" % (cols[ppa_col]))
                        continue

                if chrom in seen_chrom_pos:
                    if pos not in seen_chrom_pos[chrom]:
                        # Make up a beta.
                        assert(var_p_threshold is not None)
                        (p, beta, se, freq) = (var_p_threshold, 1, None, None)
                        chrom_pos_p_beta_se_freq[chrom].append((pos, p, beta, se, freq))
                        seen_chrom_pos[chrom].add(pos)
                        if chrom not in added_chrom_pos:
                            added_chrom_pos[chrom] = set()
                        added_chrom_pos[chrom].add(pos)

                    if chrom not in input_credible_set_info:
                        input_credible_set_info[chrom] = {}
                    if cs_id not in input_credible_set_info[chrom]:
                        input_credible_set_info[chrom][cs_id] = []
                    input_credible_set_info[chrom][cs_id].append((pos, ppa))

        return (added_chrom_pos, input_credible_set_info)

    def _compute_huge_variant_logbf_and_posteriors(
        self,
        var_z,
        allelic_var_k,
        gwas_prior_odds,
        separate_detect=False,
        allelic_var_k_detect=None,
        gwas_prior_odds_detect=None,
    ):
        var_log_bf = -np.log(np.sqrt(1 + allelic_var_k)) + 0.5 * np.square(var_z) * allelic_var_k / (1 + allelic_var_k)

        if separate_detect:
            var_log_bf_detect = -np.log(np.sqrt(1 + allelic_var_k_detect)) + 0.5 * np.square(var_z) * allelic_var_k_detect / (1 + allelic_var_k_detect)
        else:
            var_log_bf_detect = copy.copy(var_log_bf)

        # Convert log-odds to probabilities with a numerical cap for very large values.
        var_posterior = var_log_bf + np.log(gwas_prior_odds)
        if separate_detect:
            var_posterior_detect = var_log_bf_detect + np.log(gwas_prior_odds_detect)
            update_posterior = [var_posterior, var_posterior_detect]
        else:
            var_posterior_detect = copy.copy(var_posterior)
            update_posterior = [var_posterior]

        max_log = 15
        for cur_var_posterior in update_posterior:
            max_mask = cur_var_posterior < max_log
            cur_var_posterior[~max_mask] = 1
            cur_var_posterior[max_mask] = np.exp(cur_var_posterior[max_mask])
            cur_var_posterior[max_mask] = cur_var_posterior[max_mask] / (1 + cur_var_posterior[max_mask])

        if not separate_detect:
            var_posterior_detect = copy.copy(var_posterior)

        return (var_log_bf, var_log_bf_detect, var_posterior, var_posterior_detect)

    def _filter_huge_variants_for_signal_search(
        self,
        var_pos,
        var_p,
        var_beta,
        var_se,
        var_se2,
        var_log_bf,
        var_log_bf_detect,
        var_posterior,
        var_posterior_detect,
        vars_zipped,
        freq_col,
        min_n_ratio,
        mean_n,
        learn_params,
        chrom,
        added_chrom_pos,
    ):
        variants_keep = np.full(len(var_pos), True)
        qc_fail = 1 / var_se2 < min_n_ratio * mean_n
        variants_keep[qc_fail] = False

        # Make sure to add in additional credible set ids.
        if not learn_params and chrom in added_chrom_pos:
            for cur_pos in added_chrom_pos[chrom]:
                variants_keep[var_pos == cur_pos] = True

        # Filter down for efficiency.
        var_pos = var_pos[variants_keep]
        var_p = var_p[variants_keep]
        var_beta = var_beta[variants_keep]
        var_se = var_se[variants_keep]
        var_se2 = var_se2[variants_keep]
        var_log_bf = var_log_bf[variants_keep]
        var_log_bf_detect = var_log_bf_detect[variants_keep]
        var_posterior = var_posterior[variants_keep]
        var_posterior_detect = var_posterior_detect[variants_keep]

        var_logp = -np.log(var_p) / np.log(10)

        var_freq = None
        if freq_col is not None:
            var_freq = np.array(vars_zipped[4], dtype=float)[variants_keep]
            var_freq[var_freq > 0.5] = 1 - var_freq[var_freq > 0.5]

        return (
            var_pos,
            var_p,
            var_beta,
            var_se,
            var_se2,
            var_log_bf,
            var_log_bf_detect,
            var_posterior,
            var_posterior_detect,
            var_logp,
            var_freq,
        )

    def _get_huge_closest_gene_indices(self, gene_pos, region_pos):
        gene_indices = np.searchsorted(gene_pos, region_pos)
        gene_indices[gene_indices == len(gene_pos)] -= 1

        # Look to the left and the right to see which gene is closer.
        lower_mask = np.abs(region_pos - gene_pos[gene_indices - 1]) < np.abs(region_pos - gene_pos[gene_indices])
        gene_indices[lower_mask] = gene_indices[lower_mask] - 1
        return gene_indices

    def _convert_huge_log_rel_bf_to_cond_prob(self, log_rel_bf, max_log):
        cond_prob = log_rel_bf
        cond_prob[cond_prob > max_log] = 1
        cond_prob[cond_prob < max_log] = np.exp(cond_prob[cond_prob < max_log])
        return cond_prob

    def _compute_huge_region_conditional_probabilities(self, var_log_bf, var_log_bf_detect, region_vars, max_log):
        # Use log-sum-exp for numerical stability when normalizing regional variant Bayes factors.
        c = np.max(var_log_bf[region_vars])
        c_detect = np.max(var_log_bf_detect[region_vars])

        log_sum_bf = c + np.log(np.sum(np.exp(var_log_bf[region_vars] - c)))
        log_sum_bf_detect = c_detect + np.log(np.sum(np.exp(var_log_bf_detect[region_vars] - c_detect)))

        log_rel_bf = var_log_bf[region_vars] - log_sum_bf
        log_rel_bf_detect = var_log_bf_detect[region_vars] - log_sum_bf_detect

        cond_prob = self._convert_huge_log_rel_bf_to_cond_prob(log_rel_bf, max_log)
        cond_prob_detect = self._convert_huge_log_rel_bf_to_cond_prob(log_rel_bf_detect, max_log)
        return (cond_prob, cond_prob_detect)

    def _finalize_huge_selected_region(
        self,
        var_pos,
        var_p,
        var_posterior,
        var_posterior_detect,
        variants_left,
        region_vars,
        index_var_chrom_pos_ps,
        chrom,
        lead_index,
    ):
        # Mark selected region as consumed, then summarize the lead signal for downstream gene assignment.
        region_vars = np.logical_and(region_vars, variants_left)
        variants_left[region_vars] = False
        index_var_chrom_pos_ps[chrom].append((var_pos[lead_index], var_p[lead_index]))

        sig_posterior = np.max(var_posterior[region_vars])
        sig_posterior_detect = np.max(var_posterior_detect[region_vars])
        min_pos = np.min(var_pos[region_vars])
        max_pos = np.max(var_pos[region_vars])
        return (variants_left, region_vars, sig_posterior, sig_posterior_detect, min_pos, max_pos)

    def _build_huge_signal_region_mask(
        self,
        var_pos,
        var_p,
        var_logp,
        var_freq,
        lead_index,
        signal_window_size,
        signal_min_sep,
        max_signal_p,
        max_clump_ld,
        signal_max_logp_ratio,
    ):
        # Start with a fixed window around lead SNP and expand until support drops.
        region_vars = np.logical_and(
            var_pos >= var_pos[lead_index] - signal_window_size,
            var_pos <= var_pos[lead_index] + signal_window_size,
        )

        region_inds = np.where(region_vars)[0]
        assert(len(region_inds) > 0)

        increase_ratio = 1.3
        self._record_param("p_value_increase_ratio_for_sep_signal", increase_ratio)

        region_ind = region_inds[0] - 1
        last_significant_snp = region_inds[0]
        while region_ind > 0 and np.abs(var_pos[region_ind] - var_pos[last_significant_snp]) < signal_min_sep:
            if var_p[region_ind] < max_signal_p:
                if var_p[region_ind] < var_p[last_significant_snp]:
                    # Check if it starts to increase after it.
                    cur_block = np.logical_and(
                        np.logical_and(var_pos >= var_pos[region_ind], var_pos < var_pos[region_ind] + signal_min_sep),
                        var_p < max_signal_p,
                    )
                    prev_block = np.logical_and(
                        np.logical_and(var_pos >= var_pos[region_ind] + signal_min_sep, var_pos < var_pos[region_ind] + 2 * signal_min_sep),
                        var_p < max_signal_p,
                    )

                    if np.sum(prev_block) == 0 or np.sum(cur_block) == 0 or np.mean(var_logp[cur_block]) > increase_ratio * np.mean(var_logp[prev_block]):
                        break

                last_significant_snp = region_ind
                region_vars[region_ind:region_inds[0]] = True
            region_ind -= 1

        region_ind = region_inds[-1] + 1
        last_significant_snp = region_inds[0]
        while region_ind < len(var_pos) and np.abs(var_pos[region_ind] - var_pos[last_significant_snp]) < signal_min_sep:
            if var_p[region_ind] < max_signal_p:
                if var_p[region_ind] < var_p[last_significant_snp]:
                    cur_block = np.logical_and(
                        np.logical_and(var_pos <= var_pos[region_ind], var_pos > var_pos[region_ind] - signal_min_sep),
                        var_p < max_signal_p,
                    )
                    prev_block = np.logical_and(
                        np.logical_and(var_pos <= var_pos[region_ind] - signal_min_sep, var_pos > var_pos[region_ind] - 2 * signal_min_sep),
                        var_p < max_signal_p,
                    )
                    if np.sum(prev_block) == 0 or np.sum(cur_block) == 0 or np.mean(var_logp[cur_block]) > increase_ratio * np.mean(var_logp[prev_block]):
                        break

                last_significant_snp = region_ind
                region_vars[region_inds[-1]:region_ind] = True
            region_ind += 1

        # If we have MAF, approximate LD by MAF and clump incompatible variants.
        if var_freq is not None:
            max_ld = np.sqrt((var_freq[lead_index] * (1 - var_freq)) / (var_freq * (1 - var_freq[lead_index])))
            max_ld[var_freq[lead_index] > var_freq] = 1.0 / max_ld[var_freq[lead_index] > var_freq]

            region_vars[max_ld < max_clump_ld] = False

        if signal_max_logp_ratio is not None:
            region_vars[var_logp / var_logp[lead_index] < signal_max_logp_ratio] = False

        return region_vars

    def _append_huge_signal_gene_results(
        self,
        chrom,
        var_pos,
        var_p,
        lead_index,
        is_input_cs,
        sig_posterior,
        sig_posterior_detect,
        cur_gene_prob_causal,
        cur_gene_indices,
        cur_gene_po,
        cur_gene_prob_causal_detect,
        cur_gene_indices_detect,
        cur_gene_po_detect,
        gene_names,
        gene_prob_rows,
        gene_prob_rows_detect,
        gene_prob_cols,
        gene_prob_cols_detect,
        gene_bf_data,
        gene_bf_data_detect,
        gene_prob_genes,
        gene_prob_col_num,
    ):
        gene_prob_rows += list(len(gene_prob_genes) + cur_gene_indices)
        gene_prob_rows_detect += list(len(gene_prob_genes) + cur_gene_indices_detect)

        gene_prob_cols += ([gene_prob_col_num] * len(cur_gene_indices))
        gene_prob_cols_detect += ([gene_prob_col_num] * len(cur_gene_indices_detect))

        gene_bf_data += list(cur_gene_po / self.background_bf)
        gene_bf_data_detect += list(cur_gene_po_detect / self.background_bf)

        self.huge_signals.append((chrom, var_pos[lead_index], var_p[lead_index], is_input_cs))
        self.huge_signal_posteriors.append(sig_posterior)
        self.huge_signal_posteriors_for_regression.append(sig_posterior_detect)

        cur_gene_cond_prob_causal = cur_gene_prob_causal / sig_posterior
        cur_gene_cond_prob_causal_detect = cur_gene_prob_causal_detect / sig_posterior_detect

        sum_cond_prob = np.sum(cur_gene_cond_prob_causal)
        sum_cond_prob_detect = np.sum(cur_gene_cond_prob_causal_detect)
        self.huge_signal_sum_gene_cond_probabilities.append(sum_cond_prob if sum_cond_prob < 1 else 1)
        self.huge_signal_sum_gene_cond_probabilities_for_regression.append(sum_cond_prob_detect if sum_cond_prob_detect < 1 else 1)

        mean_cond_po = np.sum(cur_gene_cond_prob_causal / (1 - cur_gene_cond_prob_causal))
        mean_cond_po_detect = np.sum(cur_gene_cond_prob_causal_detect / (1 - cur_gene_cond_prob_causal_detect))
        self.huge_signal_mean_gene_pos.append(mean_cond_po)
        self.huge_signal_mean_gene_pos_for_regression.append(mean_cond_po_detect)

        gene_prob_col_num += 1
        gene_prob_genes += list(gene_names)
        return gene_prob_col_num

    def _try_use_huge_input_credible_set(
        self,
        learn_params,
        chrom,
        input_credible_set_info,
        variants_left,
        cs_ignore,
        var_pos,
        var_p,
        credible_set_span,
    ):
        result = {
            "handled": False,
            "skip_iteration": False,
            "cond_prob": None,
            "cond_prob_detect": None,
            "is_input_cs": False,
            "region_vars": None,
            "lead_index": None,
            "cs_ignore": cs_ignore,
        }
        if learn_params or chrom not in input_credible_set_info or len(input_credible_set_info[chrom].keys()) == 0:
            return result

        result["handled"] = True
        result["is_input_cs"] = True

        cur_cs_id = list(input_credible_set_info[chrom].keys())[0]
        cur_cs_vars = input_credible_set_info[chrom][cur_cs_id]
        region_vars = np.full(len(var_pos), False)

        cond_prob = np.zeros(len(var_pos))
        mask = None
        for pos_ppa in cur_cs_vars:
            pos = pos_ppa[0]
            ppa = pos_ppa[1]
            mask = np.logical_and(variants_left, var_pos == pos)

            if np.sum(mask) > 0:
                region_vars[mask] = True
                if ppa is not None:
                    cond_prob[mask] = ppa

        # All credible set variants have been used.
        if np.sum(region_vars) == 0:
            del input_credible_set_info[chrom][cur_cs_id]
            result["skip_iteration"] = True
            return result

        cur_cs_ignore = np.logical_and(
            var_pos > np.min(var_pos[region_vars]) - credible_set_span,
            var_pos < np.max(var_pos[region_vars]) + credible_set_span,
        )

        lead_index = None
        cond_prob_region = None
        cond_prob_detect_region = None
        if np.sum(cond_prob) > 0:
            cond_prob /= np.sum(cond_prob)
            lead_index = np.argmax(cond_prob)
            cond_prob_region = cond_prob[region_vars]
            cond_prob_detect_region = copy.copy(cond_prob_region)
        else:
            if mask is not None and np.sum(mask) > 0:
                lead_index = np.where(mask)[0][0]
            else:
                cs_variant_window = np.logical_and(np.logical_and(variants_left, ~cs_ignore), cur_cs_ignore)
                if np.sum(cs_variant_window) > 0:
                    cs_variant_inds = np.where(cs_variant_window)[0]
                    lead_index = cs_variant_inds[np.argmin(var_p[cs_variant_inds])]
                else:
                    result["is_input_cs"] = False

        if result["is_input_cs"]:
            result["cs_ignore"] = np.logical_or(cs_ignore, cur_cs_ignore)
            del input_credible_set_info[chrom][cur_cs_id]

        result["cond_prob"] = cond_prob_region
        result["cond_prob_detect"] = cond_prob_detect_region
        result["region_vars"] = region_vars
        result["lead_index"] = lead_index
        return result

    def _select_huge_lead_variant_index(self, var_p, variants_left):
        variants_left_inds = np.where(variants_left)[0]
        return variants_left_inds[np.argmin(var_p[variants_left_inds])]

    def _add_huge_var_rows(
        self,
        var_inds,
        gene_prob_lists,
        var_offset_prob,
        var_gene_index,
        gene_name_to_index,
        full_prob,
    ):
        # var_inds: indices into var_gene_index and var_offset_probs
        # gene_prob_lists: list of list of (gene, prob) pairs; outer list same length as var_inds
        var_to_seen_genes = {}
        num_added = 0
        for i in range(len(var_inds)):
            cur_var_index = var_inds[i]
            if cur_var_index not in var_to_seen_genes:
                var_to_seen_genes[cur_var_index] = set()
            for cur_gene, cur_prob in gene_prob_lists[i]:
                if cur_gene in gene_name_to_index:
                    cur_gene_index = gene_name_to_index[cur_gene]
                    if cur_gene_index not in var_to_seen_genes[cur_var_index]:
                        var_to_seen_genes[cur_var_index].add(cur_gene_index)
                        if num_added < len(var_to_seen_genes[cur_var_index]):
                            var_offset_prob = np.vstack((var_offset_prob, np.zeros((1, var_offset_prob.shape[1]))))
                            var_gene_index = np.vstack((var_gene_index, np.zeros((1, var_gene_index.shape[1]))))
                            num_added += 1

                        # Scale everything non-zero down to account for likelihood that the variant is coding.
                        var_offset_prob[var_gene_index[:, cur_var_index] == cur_gene_index, cur_var_index] *= (1 - cur_prob)

                        row_index = var_offset_prob.shape[0] - (num_added - len(var_to_seen_genes[cur_var_index])) - 1
                        var_offset_prob[row_index, cur_var_index] = full_prob[cur_var_index] * cur_prob
                        var_gene_index[row_index, cur_var_index] = cur_gene_index

        return (var_offset_prob, var_gene_index)

    def _aggregate_huge_var_gene_index(self, var_gene_index, cur_var_offset_prob, cap=True):
        cur_gene_indices, idx = np.unique(var_gene_index.ravel(), return_inverse=True)
        cur_gene_prob_causal = np.bincount(idx, weights=cur_var_offset_prob.ravel())

        # Remove very low ones.
        non_zero_mask = cur_gene_prob_causal > 0.001 * np.max(cur_gene_prob_causal)
        cur_gene_prob_causal = cur_gene_prob_causal[non_zero_mask]
        cur_gene_indices = cur_gene_indices[non_zero_mask]

        cur_gene_po = None
        if cap:
            cur_gene_prob_causal[cur_gene_prob_causal > 0.999] = 0.999
            cur_gene_po = cur_gene_prob_causal / (1 - cur_gene_prob_causal)

        return (cur_gene_prob_causal, cur_gene_indices, cur_gene_po)

    def _compute_huge_gene_posterior(
        self,
        region_pos,
        full_prob,
        window_fun_slope,
        window_fun_intercept,
        gene_pos,
        gene_index_to_name_index,
        gene_name_to_index,
        scale_raw_closest_gene,
        cap_raw_closest_gene,
        closest_gene_prob,
        exon_interval_tree=None,
        interval_to_gene=None,
        pos_to_gene_prob=None,
        max_offset=20,
        cap=True,
        do_print=True,
    ):
        closest_gene_indices = self._get_huge_closest_gene_indices(gene_pos, region_pos)

        offsets = np.arange(-max_offset, max_offset + 1)
        var_offset_prob = np.zeros((len(offsets), len(region_pos)))
        var_gene_index = np.full(var_offset_prob.shape, -1)
        cur_gene_indices = np.add.outer(offsets, closest_gene_indices)
        cur_gene_indices[cur_gene_indices >= len(gene_pos)] = len(gene_pos) - 1
        cur_gene_indices[cur_gene_indices <= 0] = 0

        prob_causal_odds = np.exp(window_fun_slope * np.abs(gene_pos[cur_gene_indices] - region_pos) + window_fun_intercept)
        cur_prob_causal = full_prob * (prob_causal_odds / (1 + prob_causal_odds))
        cur_prob_causal[cur_prob_causal < 0] = 0

        # Keep only maximum probability per (variant, gene-name) when genes have multiple loci.
        groups = gene_index_to_name_index[cur_gene_indices]
        data = copy.copy(cur_prob_causal)
        order = np.lexsort((data, groups), axis=0)

        order2 = np.arange(groups.shape[1])
        groups2 = groups[order, order2]
        max_by_group_mask = np.empty(groups2.shape, "bool")
        max_by_group_mask[-1, :] = True
        max_by_group_mask[:-1, :] = groups2[1:, :] != groups2[:-1, :]

        rev_order = np.empty_like(order)
        rev_order[order, order2] = np.repeat(np.arange(order.shape[0]), order.shape[1]).reshape(order.shape[0], order.shape[1])
        rev_max_by_group_mask = max_by_group_mask[rev_order, order2]
        cur_prob_causal[~rev_max_by_group_mask] = 0

        var_offset_prob = cur_prob_causal
        var_gene_index = gene_index_to_name_index[cur_gene_indices]

        if exon_interval_tree is not None and interval_to_gene is not None:
            (region_with_overlap_inds, overlapping_interval_starts, overlapping_interval_stops) = exon_interval_tree.find(region_pos, region_pos)
            coding_var_linkage_prob = np.maximum(
                np.exp(window_fun_slope + window_fun_intercept) / (1 + np.exp(window_fun_slope + window_fun_intercept)),
                0.95,
            )

            gene_lists = [interval_to_gene[(overlapping_interval_starts[i], overlapping_interval_stops[i])] for i in range(len(region_with_overlap_inds))]
            gene_prob_lists = []
            for i in range(len(gene_lists)):
                gene_prob_lists.append(list(zip(gene_lists[i], [coding_var_linkage_prob for j in range(len(gene_lists[i]))])))

            var_offset_prob, var_gene_index = self._add_huge_var_rows(
                var_inds=region_with_overlap_inds,
                gene_prob_lists=gene_prob_lists,
                var_offset_prob=var_offset_prob,
                var_gene_index=var_gene_index,
                gene_name_to_index=gene_name_to_index,
                full_prob=full_prob,
            )

        if pos_to_gene_prob is not None:
            gene_prob_lists = []
            for i in range(len(region_pos)):
                probs = []
                if region_pos[i] in pos_to_gene_prob:
                    probs = pos_to_gene_prob[region_pos[i]]
                gene_prob_lists.append(probs)
            var_offset_prob, var_gene_index = self._add_huge_var_rows(
                var_inds=range(len(region_pos)),
                gene_prob_lists=gene_prob_lists,
                var_offset_prob=var_offset_prob,
                var_gene_index=var_gene_index,
                gene_name_to_index=gene_name_to_index,
                full_prob=full_prob,
            )

        var_gene_index = var_gene_index.astype(int)

        if scale_raw_closest_gene or cap_raw_closest_gene:
            var_offset_prob_max = var_offset_prob.max(axis=0)
            var_offset_norm = np.ones(full_prob.shape)
            var_offset_norm[var_offset_prob_max != 0] = full_prob[var_offset_prob_max != 0] * closest_gene_prob / var_offset_prob_max[var_offset_prob_max != 0]

            if cap_raw_closest_gene:
                cap_mask = var_offset_norm > 1
                var_offset_norm[cap_mask] = 1
        else:
            var_offset_norm = 1

        var_offset_prob *= var_offset_norm

        (cur_gene_prob_causal_no_norm, cur_gene_indices_no_norm, cur_gene_po_no_norm) = self._aggregate_huge_var_gene_index(
            var_gene_index=var_gene_index,
            cur_var_offset_prob=var_offset_prob,
            cap=cap,
        )

        var_offset_prob_sum = np.sum(var_offset_prob, axis=0)
        var_offset_prob_sum[var_offset_prob_sum < 1] = 1
        var_offset_prob_norm = var_offset_prob / var_offset_prob_sum
        (cur_gene_prob_causal_norm, cur_gene_indices_norm, cur_gene_po_norm) = self._aggregate_huge_var_gene_index(
            var_gene_index=var_gene_index,
            cur_var_offset_prob=var_offset_prob_norm,
            cap=cap,
        )

        return (
            cur_gene_prob_causal_no_norm,
            cur_gene_indices_no_norm,
            cur_gene_po_no_norm,
            cur_gene_prob_causal_norm,
            cur_gene_indices_norm,
        )

    def _accumulate_huge_window_learning_samples(
        self,
        var_pos,
        var_p,
        gene_pos,
        gene_names,
        gene_index_to_name_index,
        total_num_vars,
        max_closest_gene_dist,
        closest_gene_prob,
        closest_dist_X,
        closest_dist_Y,
        var_all_p,
    ):
        # Randomly subsample variants to estimate distance-to-gene window behavior.
        region_vars = np.full(len(var_pos), False)
        number_needed = 100000
        region_vars[np.random.random(len(region_vars)) < (float(number_needed) / total_num_vars)] = True

        closest_gene_indices = self._get_huge_closest_gene_indices(gene_pos, var_pos[region_vars])
        closest_dists = np.abs(gene_pos[closest_gene_indices] - var_pos[region_vars])
        closest_dists = closest_dists[closest_dists <= max_closest_gene_dist]
        closest_dist_X = np.append(closest_dist_X, closest_dists)
        closest_dist_Y = np.append(closest_dist_Y, np.full(len(closest_dists), closest_gene_prob))
        var_all_p = np.append(var_all_p, var_p)

        max_offset = 200
        offsets = np.arange(-max_offset, max_offset + 1)
        cur_gene_indices = np.add.outer(offsets, closest_gene_indices)
        cur_gene_indices[cur_gene_indices >= len(gene_pos)] = len(gene_pos) - 1
        cur_gene_indices[cur_gene_indices <= 0] = 0

        # Ignore entries whose candidate is the same named gene as closest hit.
        cur_mask = (
            gene_names[gene_index_to_name_index[cur_gene_indices]]
            != gene_names[gene_index_to_name_index[closest_gene_indices]]
        )
        non_closest_dists = np.abs(gene_pos[cur_gene_indices] - var_pos[region_vars])
        cur_mask = np.logical_and(cur_mask, non_closest_dists <= max_closest_gene_dist)

        # Maximum in denominator avoids divide-by-zero when no candidates survive filtering.
        non_closest_probs = (
            np.full(non_closest_dists.shape, (1 - closest_gene_prob) / np.maximum(np.sum(cur_mask, axis=0), 1))
        )[cur_mask]
        non_closest_dists = non_closest_dists[cur_mask]
        closest_dist_X = np.append(closest_dist_X, non_closest_dists)
        closest_dist_Y = np.append(closest_dist_Y, np.full(len(non_closest_dists), non_closest_probs))

        return (closest_dist_X, closest_dist_Y, var_all_p)

    def _accumulate_huge_gene_covariates(
        self,
        gene_names,
        gene_names_non_unique,
        gene_pos,
        gene_index_to_name_index,
        gene_name_to_index,
        window_fun_slope,
        window_fun_intercept,
        scale_raw_closest_gene,
        cap_raw_closest_gene,
        closest_gene_prob,
        gene_covariate_genes,
    ):
        max_gene_offset = 500
        gene_offsets = np.arange(max_gene_offset + 1)

        gene_start_indices = np.zeros(len(gene_names), dtype=int)
        gene_end_indices = np.zeros(len(gene_names), dtype=int)
        gene_num_indices = np.zeros(len(gene_names), dtype=int)

        gene_name_to_ind = pegs_construct_map_to_ind(gene_names)
        for i in range(len(gene_names_non_unique)):
            gene_name_ind = gene_name_to_ind[gene_names_non_unique[i]]
            if gene_start_indices[gene_name_ind] == 0:
                gene_start_indices[gene_name_ind] = i
            gene_end_indices[gene_name_ind] = i
            gene_num_indices[gene_name_ind] += 1

        genes_higher_indices = np.add.outer(gene_offsets, gene_end_indices).astype(int)
        genes_ignore_indices = np.full(genes_higher_indices.shape, False)
        genes_ignore_indices[genes_higher_indices >= len(gene_pos)] = True
        genes_higher_indices[genes_higher_indices >= len(gene_pos)] = len(gene_pos) - 1
        genes_lower_indices = np.add.outer(-gene_offsets, gene_start_indices).astype(int)
        genes_ignore_indices[genes_lower_indices <= 0] = True
        genes_lower_indices[genes_lower_indices <= 0] = 0

        higher_ignore_mask = np.logical_or(
            genes_ignore_indices,
            (gene_names[gene_index_to_name_index[genes_higher_indices]] == gene_names[gene_index_to_name_index[gene_end_indices]]),
        )
        lower_ignore_mask = np.logical_or(
            genes_ignore_indices,
            (gene_names[gene_index_to_name_index[genes_lower_indices]] == gene_names[gene_index_to_name_index[gene_start_indices]]),
        )

        right_dists = (gene_pos[genes_higher_indices] - gene_pos[gene_end_indices]).astype(float)
        right_dists[higher_ignore_mask] = np.inf
        right_dists[right_dists == 0] = 1

        left_dists = (gene_pos[gene_start_indices] - gene_pos[genes_lower_indices]).astype(float)
        left_dists[lower_ignore_mask] = np.inf
        left_dists[left_dists == 0] = 1

        right_sum = np.sum(1.0 / right_dists, axis=0)
        left_sum = np.sum(1.0 / left_dists, axis=0)
        right_left_sum = right_sum + left_sum

        large_dist = 250000
        small_dist = 50000
        num_right_small = np.sum(right_dists < small_dist, axis=0)
        num_left_small = np.sum(left_dists < small_dist, axis=0)
        num_right_large = np.sum(right_dists < large_dist, axis=0)
        num_left_large = np.sum(left_dists < large_dist, axis=0)

        chrom_start = np.max((np.min(gene_pos) - 1e6, 0))
        chrom_end = np.max(gene_pos) + 1e6
        sim_variant_positions = np.linspace(
            chrom_start,
            chrom_end,
            int((chrom_end - chrom_start) / (3e9 / 2e5)),
            dtype=int,
        )

        (
            sim_gene_prob_causal_orig,
            sim_gene_indices,
            sim_gene_po,
            _sim_gene_prob_causal_norm_orig,
            _sim_gene_indices_norm,
        ) = self._compute_huge_gene_posterior(
            region_pos=sim_variant_positions,
            full_prob=np.ones(len(sim_variant_positions)),
            window_fun_slope=window_fun_slope,
            window_fun_intercept=window_fun_intercept,
            gene_pos=gene_pos,
            gene_index_to_name_index=gene_index_to_name_index,
            gene_name_to_index=gene_name_to_index,
            scale_raw_closest_gene=scale_raw_closest_gene,
            cap_raw_closest_gene=cap_raw_closest_gene,
            closest_gene_prob=closest_gene_prob,
            max_offset=20,
            cap=False,
            do_print=False,
        )

        sim_gene_prob_causal = np.zeros(len(gene_names))
        for i in range(len(sim_gene_indices)):
            sim_gene_prob_causal[sim_gene_indices[i]] = sim_gene_prob_causal_orig[i]

        cur_gene_covariates = np.vstack(
            (
                right_left_sum,
                num_right_large,
                num_left_large,
                gene_num_indices,
                sim_gene_prob_causal,
                np.ones(len(gene_names)),
            )
        ).T

        if self.gene_covariates is None:
            self.gene_covariates = cur_gene_covariates
            self.gene_covariate_names = [
                "right_left_sum_inv",
                "num_right_%s" % large_dist,
                "num_left_%s" % large_dist,
                "gene_num_indices",
                "sim_prob_causal",
                "intercept",
            ]
            self.gene_covariate_directions = np.array([-1, -1, -1, 1, 1, 0])

            self.gene_covariate_slope_defaults = np.array(
                [-0.02321564, -0.00182764, -0.00315613, 0.00824289, 0.00316042, 0.08495138]
            )
            self.total_qc_metric_betas_defaults = [
                -0.01659398,
                -0.03525455,
                -0.04813412,
                0.00553828,
                -0.39453483,
                -0.53903559,
            ]
            self.total_qc_metric_intercept_defaults = 0.98859127
            self.total_qc_metric2_betas_defaults = [
                -0.00092923,
                -0.25170301,
                -0.25994094,
                0.13700834,
                -0.10948609,
                -0.510157,
            ]
            self.total_qc_metric2_intercept_defaults = 1.70380708
            self.gene_covariate_intercept_index = len(self.gene_covariate_names) - 1
        else:
            self.gene_covariates = np.vstack((self.gene_covariates, cur_gene_covariates))

        gene_covariate_genes += list(gene_names)
        return gene_covariate_genes

    def _collect_huge_independent_signal_pvalues(self, index_var_chrom_pos_ps):
        index_var_ps = []
        for chrom in index_var_chrom_pos_ps:
            cur_pos = np.array(list(zip(*index_var_chrom_pos_ps[chrom]))[0])
            cur_ps = np.array(list(zip(*index_var_chrom_pos_ps[chrom]))[1])

            indep_window = 1e6
            tree = _IntervalTree([(x - indep_window, x + indep_window) for x in cur_pos])
            start_to_index = dict([(cur_pos[i] - indep_window, i) for i in range(len(cur_pos))])
            (ind_with_overlap_inds, overlapping_interval_starts, overlapping_interval_stops) = tree.find(cur_pos, cur_pos)
            assert(
                np.isclose(
                    overlapping_interval_stops - overlapping_interval_starts - 2 * indep_window,
                    np.zeros(len(overlapping_interval_stops)),
                ).all()
            )

            overlapping_inds = [start_to_index[i] for i in overlapping_interval_starts]
            var_p = cur_ps[ind_with_overlap_inds]
            overlap_var_p = cur_ps[overlapping_inds]
            var_not_best_mask = overlap_var_p < var_p

            indep_mask = np.full(len(cur_pos), True)
            indep_mask[ind_with_overlap_inds[var_not_best_mask]] = False
            index_var_ps += list(cur_ps[indep_mask])

        index_var_ps.sort()
        return np.array(index_var_ps)

    def _adjust_huge_detect_power(
        self,
        index_var_ps,
        num_below_low_p,
        gwas_low_p,
        detect_high_power,
        detect_low_power,
        gwas_high_p,
        gwas_high_p_posterior,
        gwas_low_p_posterior,
        detect_adjust_huge,
        allelic_var_k,
        gwas_prior_odds,
        allelic_var_k_detect,
        gwas_prior_odds_detect,
        separate_detect,
    ):
        if detect_high_power is None and detect_low_power is None:
            return (
                gwas_low_p,
                allelic_var_k,
                gwas_prior_odds,
                allelic_var_k_detect,
                gwas_prior_odds_detect,
                separate_detect,
            )

        target_max_num_variants = detect_high_power
        target_min_num_variants = detect_low_power

        old_low_p = gwas_low_p
        high_or_low = None
        if target_max_num_variants is not None and num_below_low_p > target_max_num_variants:
            gwas_low_p = index_var_ps[target_max_num_variants]
            high_or_low = "high"
        if target_min_num_variants is not None and num_below_low_p < target_min_num_variants:
            if len(index_var_ps) > target_min_num_variants:
                gwas_low_p = index_var_ps[target_min_num_variants]
            elif len(index_var_ps) > 0:
                gwas_low_p = np.min(index_var_ps)
            else:
                gwas_low_p = 0.05
            high_or_low = "low"

        if high_or_low is not None:
            self._record_param("gwas_low_p", gwas_low_p)

            log(
                "Detected %s power (%d variants below p=%.4g); adjusting --gwas-low-p to %.4g"
                % (high_or_low, num_below_low_p, old_low_p, gwas_low_p)
            )
            (allelic_var_k_detect, gwas_prior_odds_detect) = self.compute_allelic_var_and_prior(
                gwas_high_p,
                gwas_high_p_posterior,
                gwas_low_p,
                gwas_low_p_posterior,
            )
            separate_detect = True

            if detect_adjust_huge:
                (allelic_var_k, gwas_prior_odds) = (allelic_var_k_detect, gwas_prior_odds_detect)
                log("Using k=%.3g, po=%.3g for regression and huge scores" % (allelic_var_k_detect, gwas_prior_odds_detect))
                self._record_params({"gwas_allelic_var_k": allelic_var_k, "gwas_prior_odds": gwas_prior_odds})
            else:
                log("Using k=%.3g, po=%.3g for regression only" % (allelic_var_k_detect, gwas_prior_odds_detect))
                self._record_params({"gwas_allelic_var_k_detect": allelic_var_k_detect, "gwas_prior_odds_detect": gwas_prior_odds_detect})

        return (
            gwas_low_p,
            allelic_var_k,
            gwas_prior_odds,
            allelic_var_k_detect,
            gwas_prior_odds_detect,
            separate_detect,
        )

    def _compute_huge_window_function_parameters(
        self,
        learn_window,
        closest_dist_X,
        closest_dist_Y,
        closest_gene_prob,
        max_closest_gene_dist,
    ):
        if learn_window:
            use_logistic_window_function = False
            if use_logistic_window_function:
                num_samples = 5
                window_fun_slope = 0
                window_fun_intercept = 0

                for i in range(num_samples):
                    sample = np.random.random(len(closest_dist_Y)) < closest_dist_Y
                    closest_dist_Y_sample = copy.copy(closest_dist_Y)
                    closest_dist_Y_sample[sample > closest_dist_Y] = 1
                    closest_dist_Y_sample[sample <= closest_dist_Y] = 0

                    (
                        cur_window_fun_slope,
                        se,
                        z,
                        p,
                        se_inflation_factor,
                        cur_window_fun_intercept,
                        diverged,
                    ) = self._compute_logistic_beta_tildes(
                        closest_dist_X[:, np.newaxis],
                        closest_dist_Y_sample,
                        1,
                        0,
                        resid_correlation_matrix=None,
                        convert_to_dichotomous=False,
                        log_fun=lambda x, y=0: 1,
                    )
                    window_fun_slope += cur_window_fun_slope
                    window_fun_intercept += cur_window_fun_intercept

                window_fun_slope /= num_samples
                window_fun_intercept /= num_samples
            else:
                mean_closest_dist_X = np.mean(closest_dist_X[closest_dist_Y == closest_gene_prob])
                mean_non_closest_dist_X = np.mean(closest_dist_X[closest_dist_Y != closest_gene_prob])
                mean_non_closest_dist_Y = np.mean(closest_dist_Y[closest_dist_Y != closest_gene_prob])
                window_fun_slope = (
                    np.log(closest_gene_prob / (1 - closest_gene_prob))
                    - np.log(mean_non_closest_dist_Y / (1 - mean_non_closest_dist_Y))
                ) / (mean_closest_dist_X - mean_non_closest_dist_X)
                window_fun_intercept = np.log(closest_gene_prob / (1 - closest_gene_prob)) - window_fun_slope * mean_closest_dist_X

            if window_fun_slope >= 0:
                warn(
                    "Could not fit decaying linear window function slope for max-closest-gene-dist=%.4g and closest-gene_prob=%.4g; using default"
                    % (max_closest_gene_dist, closest_gene_prob)
                )
                window_fun_slope = -6.983e-06
                window_fun_intercept = -1.934

            log("Fit function %.4g * x + %.4g for closest gene probability" % (window_fun_slope, window_fun_intercept))
        else:
            if max_closest_gene_dist < 3e5:
                window_fun_slope = -5.086e-05
                window_fun_intercept = 2.988
            else:
                window_fun_slope = -5.152e-05
                window_fun_intercept = 4.854
            log("Using %.4g * x + %.4g for closest gene probability" % (window_fun_slope, window_fun_intercept))

        return (window_fun_slope, window_fun_intercept)

    def _build_huge_gwas_gene_outputs(
        self,
        gene_prob_gene_list,
        huge_results,
        huge_results_uncorrected,
        huge_results_for_regression,
        absent_genes,
        absent_log_bf,
        absent_log_bf_for_regression,
    ):
        if self.genes is not None:
            gene_bf = np.array([np.nan] * len(self.genes))
            gene_bf_for_regression = np.array([np.nan] * len(self.genes))
        else:
            gene_bf = np.array([])
            gene_bf_for_regression = np.array([])

        extra_gene_bf = []
        extra_gene_bf_for_regression = []
        extra_genes = []
        self.gene_to_gwas_huge_score = {}
        self.gene_to_gwas_huge_score_uncorrected = {}

        for i in range(len(gene_prob_gene_list)):
            gene = gene_prob_gene_list[i]
            bf = huge_results[i]
            bf_for_regression = huge_results_for_regression[i]
            bf_uncorrected = huge_results_uncorrected[i]
            self.gene_to_gwas_huge_score[gene] = bf
            self.gene_to_gwas_huge_score_uncorrected[gene] = bf_uncorrected
            if self.genes is not None and gene in self.gene_to_ind:
                assert(self.gene_to_ind[gene] == i)
                gene_bf[self.gene_to_ind[gene]] = bf
                gene_bf_for_regression[self.gene_to_ind[gene]] = bf_for_regression
            else:
                extra_gene_bf.append(bf)
                extra_gene_bf_for_regression.append(bf_for_regression)
                extra_genes.append(gene)

        for gene in absent_genes:
            bf = absent_log_bf
            bf_for_regression = absent_log_bf_for_regression
            self.gene_to_gwas_huge_score[gene] = bf
            self.gene_to_gwas_huge_score_uncorrected[gene] = bf
            if self.genes is not None and gene in self.gene_to_ind:
                gene_bf[self.gene_to_ind[gene]] = bf
                gene_bf_for_regression[self.gene_to_ind[gene]] = bf_for_regression
            else:
                extra_gene_bf.append(bf)
                extra_gene_bf_for_regression.append(bf_for_regression)
                extra_genes.append(gene)

        return (
            gene_bf,
            extra_genes,
            np.array(extra_gene_bf),
            gene_bf_for_regression,
            np.array(extra_gene_bf_for_regression),
        )

    def _build_huge_rel_prior_log_bf(self, genes, extra_genes):
        rel_prior_log_bf = None
        if self.Y_exomes is not None:
            assert(len(genes) == len(self.Y_exomes))
            rel_prior_log_bf = np.append(self.Y_exomes, np.zeros(len(extra_genes)))
        if self.Y_positive_controls is not None:
            assert(len(genes) == len(self.Y_positive_controls))
            positive_controls_prior_log_bf = np.append(self.Y_positive_controls, np.zeros(len(extra_genes)))
            if rel_prior_log_bf is None:
                rel_prior_log_bf = positive_controls_prior_log_bf
            else:
                rel_prior_log_bf += positive_controls_prior_log_bf
        if self.Y_case_counts is not None:
            assert(len(genes) == len(self.Y_case_counts))
            case_counts_prior_log_bf = np.append(self.Y_case_counts, np.zeros(len(extra_genes)))
            if rel_prior_log_bf is None:
                rel_prior_log_bf = case_counts_prior_log_bf
            else:
                rel_prior_log_bf += case_counts_prior_log_bf
        return rel_prior_log_bf

    def _finalize_and_distill_huge_signal_state(
        self,
        gene_prob_gene_list,
        gene_prob_col_num,
        gene_bf_data,
        gene_prob_rows,
        gene_prob_cols,
        gene_bf_data_detect,
        gene_prob_rows_detect,
        gene_prob_cols_detect,
        max_closest_gene_prob,
        cap_region_posterior,
        scale_region_posterior,
        phantom_region_posterior,
        allow_evidence_of_absence,
        rel_prior_log_bf,
    ):
        # Normalize collected lists and build sparse signal-by-gene BF matrices.
        self.huge_signal_posteriors = np.array(self.huge_signal_posteriors)
        self.huge_signal_posteriors_for_regression = np.array(self.huge_signal_posteriors_for_regression)
        self.huge_signal_max_closest_gene_prob = max_closest_gene_prob
        self.huge_cap_region_posterior = cap_region_posterior
        self.huge_scale_region_posterior = scale_region_posterior
        self.huge_phantom_region_posterior = phantom_region_posterior
        self.huge_allow_evidence_of_absence = allow_evidence_of_absence

        self.huge_signal_bfs = sparse.csc_matrix(
            (gene_bf_data, (gene_prob_rows, gene_prob_cols)),
            shape=(len(gene_prob_gene_list), gene_prob_col_num),
        )
        self.huge_signal_bfs_for_regression = sparse.csc_matrix(
            (gene_bf_data_detect, (gene_prob_rows_detect, gene_prob_cols_detect)),
            shape=(len(gene_prob_gene_list), gene_prob_col_num),
        )

        self.huge_signal_sum_gene_cond_probabilities = np.array(self.huge_signal_sum_gene_cond_probabilities)
        self.huge_signal_sum_gene_cond_probabilities_for_regression = np.array(
            self.huge_signal_sum_gene_cond_probabilities_for_regression
        )
        self.huge_signal_mean_gene_pos = np.array(self.huge_signal_mean_gene_pos)
        self.huge_signal_mean_gene_pos_for_regression = np.array(self.huge_signal_mean_gene_pos_for_regression)

        (huge_results, huge_results_uncorrected, absent_genes, absent_log_bf) = self._distill_huge_signal_bfs(
            self.huge_signal_bfs,
            self.huge_signal_posteriors,
            self.huge_signal_sum_gene_cond_probabilities,
            self.huge_signal_mean_gene_pos,
            self.huge_signal_max_closest_gene_prob,
            self.huge_cap_region_posterior,
            self.huge_scale_region_posterior,
            self.huge_phantom_region_posterior,
            self.huge_allow_evidence_of_absence,
            None,
            None,
            None,
            None,
            None,
            gene_prob_gene_list,
            total_genes=self.genes,
            rel_prior_log_bf=rel_prior_log_bf,
        )

        (
            huge_results_for_regression,
            huge_results_uncorrected_for_regression,
            absent_genes_for_regression,
            absent_log_bf_for_regression,
        ) = self._distill_huge_signal_bfs(
            self.huge_signal_bfs_for_regression,
            self.huge_signal_posteriors_for_regression,
            self.huge_signal_sum_gene_cond_probabilities_for_regression,
            self.huge_signal_mean_gene_pos_for_regression,
            self.huge_signal_max_closest_gene_prob,
            self.huge_cap_region_posterior,
            self.huge_scale_region_posterior,
            self.huge_phantom_region_posterior,
            self.huge_allow_evidence_of_absence,
            None,
            None,
            None,
            None,
            None,
            gene_prob_gene_list,
            total_genes=self.genes,
            rel_prior_log_bf=rel_prior_log_bf,
        )

        return (
            huge_results,
            huge_results_uncorrected,
            absent_genes,
            absent_log_bf,
            huge_results_for_regression,
            huge_results_uncorrected_for_regression,
            absent_genes_for_regression,
            absent_log_bf_for_regression,
        )

    def _compute_huge_signal_gene_posteriors(
        self,
        region_var_pos,
        full_prob,
        full_prob_detect,
        separate_detect,
        window_fun_slope,
        window_fun_intercept,
        gene_pos,
        gene_index_to_name_index,
        gene_name_to_index,
        scale_raw_closest_gene,
        cap_raw_closest_gene,
        closest_gene_prob,
        exon_interval_tree,
        interval_to_gene,
        pos_to_gene_prob,
        max_closest_gene_dist,
    ):
        # Find max offset needed for this region based on max gene-linking distance.
        gene_index_ranges = self._get_huge_closest_gene_indices(
            gene_pos,
            np.vstack(
                (
                    region_var_pos,
                    region_var_pos - max_closest_gene_dist,
                    region_var_pos + max_closest_gene_dist,
                )
            ),
        )
        max_num_indices = np.maximum(
            np.max(gene_index_ranges[0, :] - gene_index_ranges[1, :]),
            np.max(gene_index_ranges[2, :] - gene_index_ranges[0, :]),
        )

        (
            cur_gene_prob_causal,
            cur_gene_indices,
            cur_gene_po,
            cur_gene_prob_causal_norm,
            cur_gene_indices_norm,
        ) = self._compute_huge_gene_posterior(
            region_pos=region_var_pos,
            full_prob=full_prob,
            window_fun_slope=window_fun_slope,
            window_fun_intercept=window_fun_intercept,
            gene_pos=gene_pos,
            gene_index_to_name_index=gene_index_to_name_index,
            gene_name_to_index=gene_name_to_index,
            scale_raw_closest_gene=scale_raw_closest_gene,
            cap_raw_closest_gene=cap_raw_closest_gene,
            closest_gene_prob=closest_gene_prob,
            exon_interval_tree=exon_interval_tree,
            interval_to_gene=interval_to_gene,
            pos_to_gene_prob=pos_to_gene_prob,
            max_offset=max_num_indices,
        )

        if separate_detect:
            (
                cur_gene_prob_causal_detect,
                cur_gene_indices_detect,
                cur_gene_po_detect,
                cur_gene_prob_causal_norm_detect,
                cur_gene_indices_norm_detect,
            ) = self._compute_huge_gene_posterior(
                region_pos=region_var_pos,
                full_prob=full_prob_detect,
                window_fun_slope=window_fun_slope,
                window_fun_intercept=window_fun_intercept,
                gene_pos=gene_pos,
                gene_index_to_name_index=gene_index_to_name_index,
                gene_name_to_index=gene_name_to_index,
                scale_raw_closest_gene=scale_raw_closest_gene,
                cap_raw_closest_gene=cap_raw_closest_gene,
                closest_gene_prob=closest_gene_prob,
                exon_interval_tree=exon_interval_tree,
                interval_to_gene=interval_to_gene,
                pos_to_gene_prob=pos_to_gene_prob,
                max_offset=max_num_indices,
            )
        else:
            (
                cur_gene_prob_causal_detect,
                cur_gene_indices_detect,
                cur_gene_po_detect,
                cur_gene_prob_causal_norm_detect,
                cur_gene_indices_norm_detect,
            ) = (
                copy.copy(cur_gene_prob_causal),
                copy.copy(cur_gene_indices),
                copy.copy(cur_gene_po),
                copy.copy(cur_gene_prob_causal_norm),
                copy.copy(cur_gene_indices_norm),
            )

        return (
            cur_gene_prob_causal,
            cur_gene_indices,
            cur_gene_po,
            cur_gene_prob_causal_norm,
            cur_gene_indices_norm,
            cur_gene_prob_causal_detect,
            cur_gene_indices_detect,
            cur_gene_po_detect,
            cur_gene_prob_causal_norm_detect,
            cur_gene_indices_norm_detect,
        )

    def _update_huge_learning_phase_parameters(
        self,
        index_var_chrom_pos_ps,
        gwas_low_p,
        detect_high_power,
        detect_low_power,
        gwas_high_p,
        gwas_high_p_posterior,
        gwas_low_p_posterior,
        detect_adjust_huge,
        allelic_var_k,
        gwas_prior_odds,
        allelic_var_k_detect,
        gwas_prior_odds_detect,
        separate_detect,
        learn_window,
        closest_dist_X,
        closest_dist_Y,
        closest_gene_prob,
        max_closest_gene_dist,
    ):
        index_var_ps = self._collect_huge_independent_signal_pvalues(index_var_chrom_pos_ps)
        num_below_low_p = np.sum(index_var_ps < gwas_low_p)
        self._record_param("num_below_initial_low_p", num_below_low_p)
        log(" (%d variants below p=%.4g)" % (num_below_low_p, gwas_low_p))

        (
            gwas_low_p,
            allelic_var_k,
            gwas_prior_odds,
            allelic_var_k_detect,
            gwas_prior_odds_detect,
            separate_detect,
        ) = self._adjust_huge_detect_power(
            index_var_ps=index_var_ps,
            num_below_low_p=num_below_low_p,
            gwas_low_p=gwas_low_p,
            detect_high_power=detect_high_power,
            detect_low_power=detect_low_power,
            gwas_high_p=gwas_high_p,
            gwas_high_p_posterior=gwas_high_p_posterior,
            gwas_low_p_posterior=gwas_low_p_posterior,
            detect_adjust_huge=detect_adjust_huge,
            allelic_var_k=allelic_var_k,
            gwas_prior_odds=gwas_prior_odds,
            allelic_var_k_detect=allelic_var_k_detect,
            gwas_prior_odds_detect=gwas_prior_odds_detect,
            separate_detect=separate_detect,
        )

        log("Using k=%.3g, po=%.3g" % (allelic_var_k, gwas_prior_odds))
        self._record_params({"gwas_allelic_var_k": allelic_var_k, "gwas_prior_odds": gwas_prior_odds})

        (window_fun_slope, window_fun_intercept) = self._compute_huge_window_function_parameters(
            learn_window=learn_window,
            closest_dist_X=closest_dist_X,
            closest_dist_Y=closest_dist_Y,
            closest_gene_prob=closest_gene_prob,
            max_closest_gene_dist=max_closest_gene_dist,
        )
        self._record_params({"window_fun_slope": window_fun_slope, "window_fun_intercept": window_fun_intercept})

        return (
            gwas_low_p,
            allelic_var_k,
            gwas_prior_odds,
            allelic_var_k_detect,
            gwas_prior_odds_detect,
            separate_detect,
            window_fun_slope,
            window_fun_intercept,
        )

    def calculate_huge_scores_gwas(self, gwas_in, gwas_chrom_col=None, gwas_pos_col=None, gwas_p_col=None, gene_loc_file=None, hold_out_chrom=None, exons_loc_file=None, gwas_beta_col=None, gwas_se_col=None, gwas_n_col=None, gwas_n=None, gwas_freq_col=None, gwas_filter_col=None, gwas_filter_value=None, gwas_locus_col=None, gwas_ignore_p_threshold=None, gwas_units=None, gwas_low_p=5e-8, gwas_high_p=1e-2, gwas_low_p_posterior=0.98, gwas_high_p_posterior=0.001, detect_low_power=None, detect_high_power=None, detect_adjust_huge=False, learn_window=False, closest_gene_prob=0.7, max_closest_gene_prob=0.9, scale_raw_closest_gene=True, cap_raw_closest_gene=False, cap_region_posterior=True, scale_region_posterior=False, phantom_region_posterior=False, allow_evidence_of_absence=False, correct_huge=True, max_signal_p=1e-5, signal_window_size=250000, signal_min_sep=100000, signal_max_logp_ratio=None, credible_set_span=25000, max_closest_gene_dist=2.5e5, min_n_ratio=0.5, max_clump_ld=0.2, min_var_posterior=0.01, s2g_in=None, s2g_chrom_col=None, s2g_pos_col=None, s2g_gene_col=None, s2g_prob_col=None, s2g_normalize_values=None, credible_sets_in=None, credible_sets_id_col=None, credible_sets_chrom_col=None, credible_sets_pos_col=None, credible_sets_ppa_col=None, **kwargs):
        (signal_window_size, signal_max_logp_ratio) = _validate_and_normalize_huge_gwas_inputs(
            gwas_in=gwas_in,
            gene_loc_file=gene_loc_file,
            credible_sets_in=credible_sets_in,
            credible_sets_chrom_col=credible_sets_chrom_col,
            credible_sets_pos_col=credible_sets_pos_col,
            signal_window_size=signal_window_size,
            signal_min_sep=signal_min_sep,
            signal_max_logp_ratio=signal_max_logp_ratio,
        )

        self._record_params({"gwas_low_p": gwas_low_p, "gwas_high_p": gwas_high_p, "gwas_low_p_posterior": gwas_low_p_posterior, "gwas_high_p_posterior": gwas_high_p_posterior, "detect_low_power": detect_low_power, "detect_high_power": detect_high_power, "detect_adjust_huge": detect_adjust_huge, "closest_gene_prob": closest_gene_prob, "max_closest_gene_prob": max_closest_gene_prob, "scale_raw_closest_gene": scale_raw_closest_gene, "cap_raw_closest_gene": cap_raw_closest_gene, "cap_region_posterior": cap_region_posterior, "scale_region_posterior": scale_region_posterior, "max_signal_p": max_signal_p, "signal_window_size": signal_window_size, "signal_min_sep": signal_min_sep, "max_closest_gene_dist": max_closest_gene_dist, "min_n_ratio": min_n_ratio})

        need_columns = _needs_gwas_column_detection(
            gwas_pos_col=gwas_pos_col,
            gwas_chrom_col=gwas_chrom_col,
            gwas_locus_col=gwas_locus_col,
            gwas_p_col=gwas_p_col,
            gwas_beta_col=gwas_beta_col,
            gwas_se_col=gwas_se_col,
            gwas_n_col=gwas_n_col,
            gwas_n=gwas_n,
        )
        if need_columns:
            (
                gwas_pos_col,
                gwas_chrom_col,
                gwas_locus_col,
                gwas_p_col,
                gwas_beta_col,
                gwas_se_col,
                gwas_freq_col,
                gwas_n_col,
            ) = _autodetect_gwas_columns(
                gwas_in=gwas_in,
                gwas_pos_col=gwas_pos_col,
                gwas_chrom_col=gwas_chrom_col,
                gwas_locus_col=gwas_locus_col,
                gwas_p_col=gwas_p_col,
                gwas_beta_col=gwas_beta_col,
                gwas_se_col=gwas_se_col,
                gwas_freq_col=gwas_freq_col,
                gwas_n_col=gwas_n_col,
                gwas_n=gwas_n,
                debug_just_check_header=self.debug_just_check_header,
            )

        location_data = _load_huge_gene_and_exon_locations(
            gene_loc_file=gene_loc_file,
            gene_label_map=self.gene_label_map,
            hold_out_chrom=hold_out_chrom,
            exons_loc_file=exons_loc_file,
        )
        gene_chrom_name_pos = location_data["gene_chrom_name_pos"]
        gene_to_chrom = location_data["gene_to_chrom"]
        gene_to_pos = location_data["gene_to_pos"]
        chrom_to_interval_tree = location_data["chrom_to_interval_tree"]

        (allelic_var_k, gwas_prior_odds) = self.compute_allelic_var_and_prior(gwas_high_p, gwas_high_p_posterior, gwas_low_p, gwas_low_p_posterior)
        #this stores the original values, in case we detect low or high power
        (allelic_var_k_detect, gwas_prior_odds_detect) = (allelic_var_k, gwas_prior_odds)
        separate_detect = False

        (var_z_threshold, var_p_threshold) = _compute_huge_variant_thresholds(
            min_var_posterior=min_var_posterior,
            gwas_high_p_posterior=gwas_high_p_posterior,
            allelic_var_k=allelic_var_k,
            gwas_prior_odds=gwas_prior_odds,
        )


        log("Reading --gwas-in file %s" % gwas_in, INFO)

        with open_gz(gwas_in) as gwas_fh:

            split_char = None
            header_line = gwas_fh.readline().strip('\n')
            if '\t' in header_line:
                split_char = '\t'
            header_cols = header_line.split(split_char)
            header_cols = [x for x in header_cols if x != ""]

            chrom_col = None
            pos_col = None
            locus_col = None
            if gwas_chrom_col is not None and gwas_pos_col is not None:
                chrom_col = _get_col(gwas_chrom_col, header_cols)
                pos_col = _get_col(gwas_pos_col, header_cols)
            else:
                locus_col = _get_col(gwas_locus_col, header_cols)

            p_col = None
            if gwas_p_col is not None:
                p_col = _get_col(gwas_p_col, header_cols)

            beta_col = None
            if gwas_beta_col is not None:
                beta_col = _get_col(gwas_beta_col, header_cols)

            n_col = None
            se_col = None
            if gwas_n_col is not None:
                n_col = _get_col(gwas_n_col, header_cols)
            if gwas_se_col is not None:
                se_col = _get_col(gwas_se_col, header_cols)

            freq_col = None
            if gwas_freq_col is not None:
                freq_col = _get_col(gwas_freq_col, header_cols)

            filter_col = None
            if gwas_filter_col is not None:
                filter_col = _get_col(gwas_filter_col, header_cols)

            chrom_pos_p_beta_se_freq = {}
            seen_chrom_pos = {}

            if (chrom_col is None or pos_col is None) and locus_col is None:
                bail("Operation requires --gwas-chrom-col and --gwas-pos-col or --gwas-locus-col")

            #read in the gwas associations
            total_num_vars = 0

            mean_n = 0

            warned_pos = False
            warned_stats = False

            not_enough_info = 0
            for line in gwas_fh:
                cols = line.strip('\n').split(split_char)
                if (chrom_col is not None and chrom_col > len(cols)) or (pos_col is not None and pos_col > len(cols)) or (locus_col is not None and locus_col > len(cols)) or (p_col is not None and p_col > len(cols)) or (se_col is not None and se_col > len(cols)) or (n_col is not None and n_col > len(cols)) or (freq_col is not None and freq_col > len(cols) or (filter_col is not None and filter_col > len(cols))):
                    warn("Skipping line due to too few columns: %s" % line)
                    continue

                if filter_col is not None and gwas_filter_value is not None and cols[filter_col] != gwas_filter_value:
                    continue

                if chrom_col is not None and pos_col is not None:
                    chrom = cols[chrom_col]
                    pos = cols[pos_col]
                else:
                    locus = cols[locus_col]
                    locus_tokens = None
                    for locus_delim in [":", "_"]:
                        if locus_delim in locus:
                            locus_tokens = locus.split(locus_delim)
                            break
                    if locus_tokens is None or len(locus_tokens) <= 2:
                        bail("Could not split locus %s on either : or _" % locus)
                    chrom = locus_tokens[0]
                    pos = locus_tokens[1]

                chrom = pegs_clean_chrom_name(chrom)
                if hold_out_chrom is not None and chrom == hold_out_chrom:
                    continue
                try:
                    pos = int(pos)
                except ValueError:
                    if not warned_pos:
                        warn("Skipping unconvertible pos value %s" % (cols[pos_col]))
                        warned_pos = True
                    continue

                p = None
                if p_col is not None:
                    try:
                        p = float(cols[p_col])
                    except ValueError:
                        if not cols[p_col] == "NA":
                            if not warned_stats:
                                warn("Skipping unconvertible p value %s" % (cols[p_col]))
                                warned_stats = True
                        p = None

                    if p is not None:
                        min_p = 1e-250
                        if p < min_p:
                            p = min_p

                        if p <= 0 or p > 1:
                            if not warned_stats:
                                warn("Skipping invalid p value %s" % (p))
                                warned_stats = True
                            p = None

                        if gwas_ignore_p_threshold is not None and p > gwas_ignore_p_threshold:
                            continue

                beta = None
                if beta_col is not None:
                    try:
                        beta = float(cols[beta_col])
                    except ValueError:
                        if not cols[beta_col] == "NA":
                            if not warned_stats:
                                warn("Skipping unconvertible beta value %s" % (cols[beta_col]))
                                warned_stats = True
                        beta = None

                se = None
                if se_col is not None:
                    try:
                        se = float(cols[se_col])
                    except ValueError:
                        if not cols[se_col] == "NA":
                            if not warned_stats:
                                warn("Skipping unconvertible se value %s" % (cols[se_col]))
                                warned_stats = True
                        se = None

                if se is None:
                    if n_col is not None:
                        try:
                            n = float(cols[n_col])
                            if n <= 0:
                                if not warned_stats:
                                    warn("Skipping invalid N value %s" % (n))
                                    warned_stats = True
                                n = None

                        except ValueError:
                            if not cols[n_col] == "NA":
                                if not warned_stats:
                                    warn("Skipping unconvertible n value %s" % (cols[n_col]))
                                    warned_stats = True
                            n = None

                        if n is not None:
                            se = 1 / np.sqrt(n)

                    elif gwas_n is not None:
                        if gwas_n <= 0:
                            bail("Invalid gwas-n value: %s" % (gwas_n))

                        n = gwas_n
                        se = 1 / np.sqrt(n)


                #make sure have two of the three
                if sum((p is not None, se is not None, beta is not None)) < 2:
                    not_enough_info += 1
                    continue

                if var_z_threshold is not None:
                    if p is not None:
                        if p > var_p_threshold:
                            continue
                    else:
                        if se == 0:
                            continue
                        z = np.abs(beta / se)
                        if z < var_z_threshold:
                            continue

                freq = None
                if freq_col is not None:
                    try:
                        freq = float(cols[freq_col])
                        if freq > 1 or freq < 0:
                            warn("Skipping invalid freq value %s" % freq)
                            freq = None
                    except ValueError:
                        if not cols[freq_col] == "NA":
                            warn("Skipping unconvertible n value %s" % (cols[freq_col]))
                        freq = None


                if chrom not in chrom_pos_p_beta_se_freq:
                    chrom_pos_p_beta_se_freq[chrom] = []

                chrom_pos_p_beta_se_freq[chrom].append((pos, p, beta, se, freq))
                if chrom not in seen_chrom_pos:
                    seen_chrom_pos[chrom] = set()
                seen_chrom_pos[chrom].add(pos)
                total_num_vars += 1

            if not_enough_info > 0:
                warn("Skipped %d variants due to not enough information" % (not_enough_info))

            log("Read in %d variants" % total_num_vars)
            chrom_pos_to_gene_prob = self._read_huge_s2g_probabilities(
                s2g_in=s2g_in,
                seen_chrom_pos=seen_chrom_pos,
                hold_out_chrom=hold_out_chrom,
                s2g_chrom_col=s2g_chrom_col,
                s2g_pos_col=s2g_pos_col,
                s2g_gene_col=s2g_gene_col,
                s2g_prob_col=s2g_prob_col,
                s2g_normalize_values=s2g_normalize_values,
            )

            (added_chrom_pos, input_credible_set_info) = self._read_huge_input_credible_sets(
                credible_sets_in=credible_sets_in,
                seen_chrom_pos=seen_chrom_pos,
                chrom_pos_p_beta_se_freq=chrom_pos_p_beta_se_freq,
                var_p_threshold=var_p_threshold,
                hold_out_chrom=hold_out_chrom,
                credible_sets_id_col=credible_sets_id_col,
                credible_sets_chrom_col=credible_sets_chrom_col,
                credible_sets_pos_col=credible_sets_pos_col,
                credible_sets_ppa_col=credible_sets_ppa_col,
            )

            if total_num_vars == 0:
                bail("Didn't read in any variants!")

            gene_output_data = {}
            total_prob_causal = 0

            # Run through twice: first pass learns the window function, second computes scores.
            huge_buffers = self._initialize_huge_gwas_state()
            closest_dist_Y = huge_buffers["closest_dist_Y"]
            closest_dist_X = huge_buffers["closest_dist_X"]
            var_all_p = huge_buffers["var_all_p"]
            gene_bf_data = huge_buffers["gene_bf_data"]
            gene_bf_data_detect = huge_buffers["gene_bf_data_detect"]
            gene_prob_rows = huge_buffers["gene_prob_rows"]
            gene_prob_rows_detect = huge_buffers["gene_prob_rows_detect"]
            gene_prob_cols = huge_buffers["gene_prob_cols"]
            gene_prob_cols_detect = huge_buffers["gene_prob_cols_detect"]
            gene_prob_genes = huge_buffers["gene_prob_genes"]
            gene_prob_col_num = huge_buffers["gene_prob_col_num"]
            gene_covariate_genes = huge_buffers["gene_covariate_genes"]
            window_fun_intercept = None
            window_fun_slope = None

            #second, compute the huge scores
            for learn_params in [True, False]:
                index_var_chrom_pos_ps = {}
                if learn_params:
                    log("Learning window function and allelic var scale factor")
                else:
                    log("Calculating GWAS HuGE scores")

                for chrom in chrom_pos_p_beta_se_freq:

                    #log("Processing chrom %s" % chrom, TRACE)
                    #convert all of these to np arrays sorted by chromosome
                    #sorted arrays of variant positions and p-values

                    chrom_pos_p_beta_se_freq[chrom].sort(key=lambda k: k[0])
                    vars_zipped = list(zip(*chrom_pos_p_beta_se_freq[chrom]))

                    if len(vars_zipped) == 0:
                        continue

                    var_pos = np.array(vars_zipped[0], dtype=float)
                    var_p = np.array(vars_zipped[1], dtype=float)
                    var_beta = np.array(vars_zipped[2], dtype=float)
                    var_se = np.array(vars_zipped[3], dtype=float)

                    (var_p, var_beta, var_se) = pegs_complete_p_beta_se(
                        var_p,
                        var_beta,
                        var_se,
                        warn_fn=warn,
                    )

                    var_z = var_beta / var_se
                    var_se2 = np.square(var_se)

                    #this will vary slightly by chromosome but probably okay
                    mean_n = np.mean(1 / var_se2)

                    #sorted arrays of gene positions and p-values
                    if chrom not in gene_chrom_name_pos:
                        warn("Could not find chromosome %s in --gene-loc-file; skipping for now" % chrom)
                        continue

                    index_var_chrom_pos_ps[chrom] = []

                    gene_chrom_name_pos[chrom].sort(key=lambda k: k[1])
                    gene_zipped = list(zip(*gene_chrom_name_pos[chrom]))

                    #gene_names is array of the unique gene names
                    #gene_index_to_name_index is an array of the positions (each gene has multiple) and tells us which gene name corresponds to each position
                    gene_names_non_unique = np.array(gene_zipped[0])

                    gene_names, gene_index_to_name_index = np.unique(gene_names_non_unique, return_inverse=True)
                    gene_name_to_index = pegs_construct_map_to_ind(gene_names)
                    gene_pos = np.array(gene_zipped[1])

                    #get a map from position to gene
                    pos_to_gene_prob = None
                    if chrom_pos_to_gene_prob is not None and chrom in chrom_pos_to_gene_prob:
                        pos_to_gene_prob = chrom_pos_to_gene_prob[chrom]                        

                    #gene_prob_causal = np.full(len(gene_names), self.background_prior)

                    exon_interval_tree = None
                    interval_to_gene = None
                    if exons_loc_file is not None and chrom in chrom_to_interval_tree:
                        exon_interval_tree = chrom_to_interval_tree[chrom]
                        interval_to_gene = chrom_interval_to_gene[chrom]

                    if learn_params:
                        (closest_dist_X, closest_dist_Y, var_all_p) = self._accumulate_huge_window_learning_samples(
                            var_pos=var_pos,
                            var_p=var_p,
                            gene_pos=gene_pos,
                            gene_names=gene_names,
                            gene_index_to_name_index=gene_index_to_name_index,
                            total_num_vars=total_num_vars,
                            max_closest_gene_dist=max_closest_gene_dist,
                            closest_gene_prob=closest_gene_prob,
                            closest_dist_X=closest_dist_X,
                            closest_dist_Y=closest_dist_Y,
                            var_all_p=var_all_p,
                        )

                    else:

                        if correct_huge:
                            gene_covariate_genes = self._accumulate_huge_gene_covariates(
                                gene_names=gene_names,
                                gene_names_non_unique=gene_names_non_unique,
                                gene_pos=gene_pos,
                                gene_index_to_name_index=gene_index_to_name_index,
                                gene_name_to_index=gene_name_to_index,
                                window_fun_slope=window_fun_slope,
                                window_fun_intercept=window_fun_intercept,
                                scale_raw_closest_gene=scale_raw_closest_gene,
                                cap_raw_closest_gene=cap_raw_closest_gene,
                                closest_gene_prob=closest_gene_prob,
                                gene_covariate_genes=gene_covariate_genes,
                            )

                    #now onto variants


                    #Z-score based one:
                    #K=-0.439
                    #np.sqrt(1 + K) * np.exp(-np.square(var_z) / 2 * (K) / (1 + K))
                    #or, for which sample size doesn't matter:
                    #K=-0.439 / np.mean(var_n)
                    #np.sqrt(1 + var_n * K) * np.exp(-np.square(var_z) / 2 * (var_n * K) / (1 + var_n * K))

                    # var_log_bf = np.log(np.sqrt(1 + allelic_var_k)) + 0.5 * np.square(var_z) * allelic_var_k / (1 + allelic_var_k)
                    (var_log_bf, var_log_bf_detect, var_posterior, var_posterior_detect) = self._compute_huge_variant_logbf_and_posteriors(
                        var_z=var_z,
                        allelic_var_k=allelic_var_k,
                        gwas_prior_odds=gwas_prior_odds,
                        separate_detect=separate_detect,
                        allelic_var_k_detect=allelic_var_k_detect,
                        gwas_prior_odds_detect=gwas_prior_odds_detect,
                    )
                    max_log = 15

                    (
                        var_pos,
                        var_p,
                        var_beta,
                        var_se,
                        var_se2,
                        var_log_bf,
                        var_log_bf_detect,
                        var_posterior,
                        var_posterior_detect,
                        var_logp,
                        var_freq,
                    ) = self._filter_huge_variants_for_signal_search(
                        var_pos=var_pos,
                        var_p=var_p,
                        var_beta=var_beta,
                        var_se=var_se,
                        var_se2=var_se2,
                        var_log_bf=var_log_bf,
                        var_log_bf_detect=var_log_bf_detect,
                        var_posterior=var_posterior,
                        var_posterior_detect=var_posterior_detect,
                        vars_zipped=vars_zipped,
                        freq_col=freq_col,
                        min_n_ratio=min_n_ratio,
                        mean_n=mean_n,
                        learn_params=learn_params,
                        chrom=chrom,
                        added_chrom_pos=added_chrom_pos,
                    )

                    variants_left = np.full(len(var_pos), True)
                    cs_ignore = np.full(len(var_pos), False)
                    while np.sum(variants_left) > 0:

                        cond_prob = None
                        cond_prob_detect = None
                        is_input_cs = False
                        cs_selection = self._try_use_huge_input_credible_set(
                            learn_params=learn_params,
                            chrom=chrom,
                            input_credible_set_info=input_credible_set_info,
                            variants_left=variants_left,
                            cs_ignore=cs_ignore,
                            var_pos=var_pos,
                            var_p=var_p,
                            credible_set_span=credible_set_span,
                        )
                        if cs_selection["skip_iteration"]:
                            continue
                        if cs_selection["handled"]:
                            cond_prob = cs_selection["cond_prob"]
                            cond_prob_detect = cs_selection["cond_prob_detect"]
                            is_input_cs = cs_selection["is_input_cs"]
                            region_vars = cs_selection["region_vars"]
                            i = cs_selection["lead_index"]
                            cs_ignore = cs_selection["cs_ignore"]

                        #if we didn't have credible set, or it didn't have PPA, then we go through here
                        if cond_prob is None:

                            #if it wasn't a credible set, we select the lead SNP. Otherwise we selected above
                            if not is_input_cs:

                                if not learn_params:
                                    variants_left = np.logical_and(variants_left, ~cs_ignore)

                                # Get the lowest p-value remaining variant.
                                i = self._select_huge_lead_variant_index(var_p, variants_left)

                            #we will do this if there was no credible set, or if the credible set just gave us the top variant
                            #if it just gave us the top variant, then we expand around the lead SNP in the credible set

                            region_vars = self._build_huge_signal_region_mask(
                                var_pos=var_pos,
                                var_p=var_p,
                                var_logp=var_logp,
                                var_freq=var_freq,
                                lead_index=i,
                                signal_window_size=signal_window_size,
                                signal_min_sep=signal_min_sep,
                                max_signal_p=max_signal_p,
                                max_clump_ld=max_clump_ld,
                                signal_max_logp_ratio=signal_max_logp_ratio,
                            )

                        (variants_left, region_vars, sig_posterior, sig_posterior_detect, min_pos, max_pos) = self._finalize_huge_selected_region(
                            var_pos=var_pos,
                            var_p=var_p,
                            var_posterior=var_posterior,
                            var_posterior_detect=var_posterior_detect,
                            variants_left=variants_left,
                            region_vars=region_vars,
                            index_var_chrom_pos_ps=index_var_chrom_pos_ps,
                            chrom=chrom,
                            lead_index=i,
                        )
                        #log("%d-%d (%d)" % (min_pos, max_pos, max_pos - min_pos))
                        #if not learn_params:
                            #log("Index SNP %d=%d; region=%d-%d; logp=%.3g-%.3g" % (i,var_pos[i], np.min(var_pos[region_vars]), np.max(var_pos[region_vars]), np.min(var_logp[region_vars]), np.max(var_logp[region_vars])), TRACE)
                        #log("Variant:",var_pos[i],"P:",var_p[i],"POST:",sig_posterior,"MIN_POS:",min_pos,"MAX_POS:",max_pos,"NUM:",np.sum(region_vars))
                        #m = np.where(var_pos == 84279410.0)[0]


                        if cond_prob is None:
                            (cond_prob, cond_prob_detect) = self._compute_huge_region_conditional_probabilities(
                                var_log_bf=var_log_bf,
                                var_log_bf_detect=var_log_bf_detect,
                                region_vars=region_vars,
                                max_log=max_log,
                            )


                        #this is the final posterior probability of association for all variants in the region
                        full_prob = cond_prob * sig_posterior
                        full_prob_detect = cond_prob_detect * sig_posterior_detect

                        if not learn_params:

                            (
                                cur_gene_prob_causal,
                                cur_gene_indices,
                                cur_gene_po,
                                cur_gene_prob_causal_norm,
                                cur_gene_indices_norm,
                                cur_gene_prob_causal_detect,
                                cur_gene_indices_detect,
                                cur_gene_po_detect,
                                cur_gene_prob_causal_norm_detect,
                                cur_gene_indices_norm_detect,
                            ) = self._compute_huge_signal_gene_posteriors(
                                region_var_pos=var_pos[region_vars],
                                full_prob=full_prob,
                                full_prob_detect=full_prob_detect,
                                separate_detect=separate_detect,
                                window_fun_slope=window_fun_slope,
                                window_fun_intercept=window_fun_intercept,
                                gene_pos=gene_pos,
                                gene_index_to_name_index=gene_index_to_name_index,
                                gene_name_to_index=gene_name_to_index,
                                scale_raw_closest_gene=scale_raw_closest_gene,
                                cap_raw_closest_gene=cap_raw_closest_gene,
                                closest_gene_prob=closest_gene_prob,
                                exon_interval_tree=exon_interval_tree,
                                interval_to_gene=interval_to_gene,
                                pos_to_gene_prob=pos_to_gene_prob,
                                max_closest_gene_dist=max_closest_gene_dist,
                            )

                            gene_prob_col_num = self._append_huge_signal_gene_results(
                                chrom=chrom,
                                var_pos=var_pos,
                                var_p=var_p,
                                lead_index=i,
                                is_input_cs=is_input_cs,
                                sig_posterior=sig_posterior,
                                sig_posterior_detect=sig_posterior_detect,
                                cur_gene_prob_causal=cur_gene_prob_causal,
                                cur_gene_indices=cur_gene_indices,
                                cur_gene_po=cur_gene_po,
                                cur_gene_prob_causal_detect=cur_gene_prob_causal_detect,
                                cur_gene_indices_detect=cur_gene_indices_detect,
                                cur_gene_po_detect=cur_gene_po_detect,
                                gene_names=gene_names,
                                gene_prob_rows=gene_prob_rows,
                                gene_prob_rows_detect=gene_prob_rows_detect,
                                gene_prob_cols=gene_prob_cols,
                                gene_prob_cols_detect=gene_prob_cols_detect,
                                gene_bf_data=gene_bf_data,
                                gene_bf_data_detect=gene_bf_data_detect,
                                gene_prob_genes=gene_prob_genes,
                                gene_prob_col_num=gene_prob_col_num,
                            )

                if learn_params:
                    (
                        gwas_low_p,
                        allelic_var_k,
                        gwas_prior_odds,
                        allelic_var_k_detect,
                        gwas_prior_odds_detect,
                        separate_detect,
                        window_fun_slope,
                        window_fun_intercept,
                    ) = self._update_huge_learning_phase_parameters(
                        index_var_chrom_pos_ps=index_var_chrom_pos_ps,
                        gwas_low_p=gwas_low_p,
                        detect_high_power=detect_high_power,
                        detect_low_power=detect_low_power,
                        gwas_high_p=gwas_high_p,
                        gwas_high_p_posterior=gwas_high_p_posterior,
                        gwas_low_p_posterior=gwas_low_p_posterior,
                        detect_adjust_huge=detect_adjust_huge,
                        allelic_var_k=allelic_var_k,
                        gwas_prior_odds=gwas_prior_odds,
                        allelic_var_k_detect=allelic_var_k_detect,
                        gwas_prior_odds_detect=gwas_prior_odds_detect,
                        separate_detect=separate_detect,
                        learn_window=learn_window,
                        closest_dist_X=closest_dist_X,
                        closest_dist_Y=closest_dist_Y,
                        closest_gene_prob=closest_gene_prob,
                        max_closest_gene_dist=max_closest_gene_dist,
                    )

            #now iterate through all significant variants

            log("Done reading --gwas-in", DEBUG)

            if len(self.huge_signals) == 0:
                 bail("Didn't read in any SNPs for HuGE scores")


            (genes, gene_to_ind, extra_genes, extra_gene_to_ind, gene_prob_gene_list) = self._remap_huge_gene_probability_rows(
                gene_to_chrom=gene_to_chrom,
                gene_prob_genes=gene_prob_genes,
                gene_prob_rows=gene_prob_rows,
                gene_prob_rows_detect=gene_prob_rows_detect,
            )
            self._align_huge_gene_covariates_to_gene_list(
                gene_prob_gene_list=gene_prob_gene_list,
                gene_covariate_genes=gene_covariate_genes,
                gene_to_ind=gene_to_ind,
                extra_gene_to_ind=extra_gene_to_ind,
            )

            exomes_positive_controls_case_counts_prior_log_bf = self._build_huge_rel_prior_log_bf(
                genes=genes,
                extra_genes=extra_genes,
            )

            (
                huge_results,
                huge_results_uncorrected,
                absent_genes,
                absent_log_bf,
                huge_results_for_regression,
                huge_results_uncorrected_for_regression,
                absent_genes_for_regression,
                absent_log_bf_for_regression,
            ) = self._finalize_and_distill_huge_signal_state(
                gene_prob_gene_list=gene_prob_gene_list,
                gene_prob_col_num=gene_prob_col_num,
                gene_bf_data=gene_bf_data,
                gene_prob_rows=gene_prob_rows,
                gene_prob_cols=gene_prob_cols,
                gene_bf_data_detect=gene_bf_data_detect,
                gene_prob_rows_detect=gene_prob_rows_detect,
                gene_prob_cols_detect=gene_prob_cols_detect,
                max_closest_gene_prob=max_closest_gene_prob,
                cap_region_posterior=cap_region_posterior,
                scale_region_posterior=scale_region_posterior,
                phantom_region_posterior=phantom_region_posterior,
                allow_evidence_of_absence=allow_evidence_of_absence,
                rel_prior_log_bf=exomes_positive_controls_case_counts_prior_log_bf,
            )

            (
                gene_bf,
                extra_genes,
                extra_gene_bf,
                gene_bf_for_regression,
                extra_gene_bf_for_regression,
            ) = self._build_huge_gwas_gene_outputs(
                gene_prob_gene_list=gene_prob_gene_list,
                huge_results=huge_results,
                huge_results_uncorrected=huge_results_uncorrected,
                huge_results_for_regression=huge_results_for_regression,
                absent_genes=absent_genes,
                absent_log_bf=absent_log_bf,
                absent_log_bf_for_regression=absent_log_bf_for_regression,
            )

            self.combine_huge_scores()

            total_gene_bfs = np.append(gene_bf_for_regression, extra_gene_bf_for_regression)
            number_same = np.max(np.unique(np.append(gene_bf_for_regression, extra_gene_bf_for_regression), return_counts=True)[1])
            fraction_same = number_same / float(len(total_gene_bfs))
            if fraction_same > 0.4:
                log("Had %d out of %d genes with the the same huge scores; too few genes to run regressions to learn confounder corrections" % (number_same, len(total_gene_bfs)))
                self.huge_sparse_mode = True

            return (gene_bf, extra_genes, extra_gene_bf, gene_bf_for_regression, extra_gene_bf_for_regression)

    def write_huge_statistics(self, huge_statistics_out, gene_bf, extra_genes, extra_gene_bf, gene_bf_for_regression, extra_gene_bf_for_regression):
        if huge_statistics_out is None:
            return

        log("Writing HuGE statistics cache to %s" % huge_statistics_out, INFO)
        pegs_write_prefixed_tar_bundle(
            huge_statistics_out,
            prefix_basename="huge_stats",
            write_prefix_fn=lambda prefix: _write_huge_statistics_bundle(
                self,
                prefix,
                gene_bf,
                extra_genes,
                extra_gene_bf,
                gene_bf_for_regression,
                extra_gene_bf_for_regression,
            ),
            is_bundle_path_fn=pegs_is_huge_statistics_bundle_path,
            option_name="--huge-statistics-out",
            temp_prefix="huge_statistics_out_",
            bail_fn=bail,
        )

    def read_huge_statistics(self, huge_statistics_in):
        if huge_statistics_in is None:
            bail("Require --huge-statistics-in for this operation")

        log("Reading HuGE statistics cache from %s" % huge_statistics_in, INFO)
        return pegs_read_prefixed_tar_bundle(
            huge_statistics_in,
            required_suffix=".huge.meta.json.gz",
            read_prefix_fn=lambda prefix: _read_huge_statistics_bundle(self, prefix),
            is_bundle_path_fn=pegs_is_huge_statistics_bundle_path,
            bundle_flag_name="HuGE cache",
            temp_prefix="huge_statistics_in_",
            bail_fn=bail,
        )

    def calculate_huge_scores_exomes(self, exomes_in, exomes_gene_col=None, exomes_p_col=None, exomes_beta_col=None, exomes_se_col=None, exomes_n_col=None, exomes_n=None, exomes_units=None, allelic_var=0.36, exomes_low_p=2.5e-6, exomes_high_p=0.05, exomes_low_p_posterior=0.95, exomes_high_p_posterior=0.10, hold_out_chrom=None, gene_loc_file=None, **kwargs):

        if exomes_in is None:
            bail("Require --exomes-in for this operation")

        if hold_out_chrom is not None and self.gene_to_chrom is None:
            (self.gene_chrom_name_pos, self.gene_to_chrom, self.gene_to_pos) = pegs_read_loc_file_with_gene_map(
                gene_loc_file,
                gene_label_map=self.gene_label_map,
                clean_chrom_fn=pegs_clean_chrom_name,
                warn_fn=warn,
                bail_fn=bail,
            )

        self._record_params({"exomes_low_p": exomes_low_p, "exomes_high_p": exomes_high_p, "exomes_low_p_posterior": exomes_low_p_posterior, "exomes_high_p_posterior": exomes_high_p_posterior})

        if exomes_gene_col is None:
            need_columns = True

        has_se = exomes_se_col is not None or exomes_n_col is not None or exomes_n is not None
        if exomes_gene_col is not None and ((exomes_p_col is not None and exomes_beta_col is not None) or (exomes_p_col is not None and has_se) or (exomes_beta_col is not None and has_se)):
            need_columns = False
        else:
            need_columns = True

        if need_columns:
            (possible_gene_id_cols, possible_var_id_cols, possible_chrom_cols, possible_pos_cols, possible_locus_cols, possible_p_cols, possible_beta_cols, possible_se_cols, possible_freq_cols, possible_n_cols, header) = _determine_columns_from_file(exomes_in)

            #now recompute
            if exomes_gene_col is None:
                if len(possible_gene_id_cols) == 1:
                    exomes_gene_col = possible_gene_id_cols[0]
                    log("Using %s for gene_id column; change with --exomes-gene-col if incorrect" % exomes_gene_col)
                else:
                    bail("Could not determine gene_id column from header %s; specify with --exomes-gene-col" % header)

            if exomes_p_col is None:
                if len(possible_p_cols) == 1:
                    exomes_p_col = possible_p_cols[0]
                    log("Using %s for p column; change with --exomes-p-col if incorrect" % exomes_p_col)
                else:
                    log("Could not determine p column from header %s; if desired specify with --exomes-p-col" % header)
            if exomes_se_col is None:
                if len(possible_se_cols) == 1:
                    exomes_se_col = possible_se_cols[0]
                    log("Using %s for se column; change with --exomes-se-col if incorrect" % exomes_se_col)
                else:
                    log("Could not determine se column from header %s; if desired specify with --exomes-se-col" % header)
            if exomes_beta_col is None:
                if len(possible_beta_cols) == 1:
                    exomes_beta_col = possible_beta_cols[0]
                    log("Using %s for beta column; change with --exomes-beta-col if incorrect" % exomes_beta_col)
                else:
                    log("Could not determine beta column from header %s; if desired specify with --exomes-beta-col" % header)

            if exomes_n_col is None:
                if len(possible_n_cols) == 1:
                    exomes_n_col = possible_n_cols[0]
                    log("Using %s for N column; change with --exomes-n-col if incorrect" % exomes_n_col)
                else:
                    log("Could not determine N column from header %s; if desired specify with --exomes-n-col" % header)

            has_se = exomes_se_col is not None or exomes_n_col is not None or exomes_n is not None
            if (exomes_p_col is not None and exomes_beta_col is not None) or (exomes_p_col is not None and has_se) or (exomes_beta_col is not None and has_se):
                pass
            else:
                bail("Require information about at least two of p-value, se, and beta; specify with --exomes-p-col, --exomes-beta-col, and --exomes-se-col")

        (allelic_var_k, exomes_prior_odds) = self.compute_allelic_var_and_prior(exomes_high_p, exomes_high_p_posterior, exomes_low_p, exomes_low_p_posterior)

        self._record_params({"exomes_allelic_var_k": allelic_var_k, "exomes_prior_odds": exomes_prior_odds})

        log("Using exomes k=%.3g, po=%.3g" % (allelic_var_k, exomes_prior_odds))

        log("Calculating exomes HuGE scores")

        log("Reading --exomes-in file %s" % exomes_in, INFO)

        seen_genes = set()
        genes = []
        gene_ps = []
        gene_betas = []
        gene_ses = []

        #get the delimiter
        delims = [None, '\t', ' ']
        delim = None
        found_delim = False
        for cur_delim in delims:
            num_cols = None
            good = True
            with open_gz(exomes_in) as exomes_fh:
                for line in exomes_fh:
                    cols = line.strip('\n').split(cur_delim)            
                    if len(cols) == 1 or (num_cols is not None and len(cols) != num_cols):
                        good = False
                        break
                    num_cols = len(cols)
            if good:
                delim = cur_delim
                found_delim = True
                break
        if not found_delim:
            bail("Could not find delimiter for --exomes-in that yielded same number of columns for every row")
        
            
        with open_gz(exomes_in) as exomes_fh:
            header_cols = exomes_fh.readline().strip('\n').split(delim)
            gene_col = _get_col(exomes_gene_col, header_cols)

            p_col = None
            if exomes_p_col is not None:
                p_col = _get_col(exomes_p_col, header_cols)

            beta_col = None
            if exomes_beta_col is not None:
                beta_col = _get_col(exomes_beta_col, header_cols)

            n_col = None
            se_col = None
            if exomes_n_col is not None:
                n_col = _get_col(exomes_n_col, header_cols)
            if exomes_se_col is not None:
                se_col = _get_col(exomes_se_col, header_cols)
            
            chrom_pos_p_se = {}

            #read in the exomes associations
            total_num_genes = 0

            for line in exomes_fh:

                cols = line.strip('\n').split(delim)
                if gene_col > len(cols) or (p_col is not None and p_col > len(cols)) or (se_col is not None and se_col > len(cols)) or (beta_col is not None and beta_col > len(cols)) or (n_col is not None and n_col > len(cols)):
                    warn("Skipping due to too few columns in line: %s" % line)
                    continue

                gene = cols[gene_col]

                if self.gene_label_map is not None and gene in self.gene_label_map:
                    gene = self.gene_label_map[gene]

                if hold_out_chrom is not None and gene in self.gene_to_chrom and self.gene_to_chrom[gene] == hold_out_chrom:
                    continue

                p = None
                beta = None
                se = None
                
                if p_col is not None:
                    try:
                        p = float(cols[p_col])
                    except ValueError:
                        if not cols[p_col] == "NA":
                            warn("Skipping unconvertible p value %s" % (cols[p_col]))
                        continue

                    min_p = 1e-250
                    if p < min_p:
                        p = min_p

                    if p <= 0 or p > 1:
                        warn("Skipping invalid p value %s" % (p))
                        continue

                if beta_col is not None:
                    try:
                        beta = float(cols[beta_col])
                    except ValueError:
                        if not cols[beta_col] == "NA":
                            warn("Skipping unconvertible beta value %s" % (cols[beta_col]))
                        continue

                if se_col is not None:
                    try:
                        se = float(cols[se_col])
                    except ValueError:
                        if not cols[se_col] == "NA":
                            warn("Skipping unconvertible se value %s" % (cols[se_col]))
                        continue
                elif n_col is not None:
                    try:
                        n = float(cols[n_col])
                    except ValueError:
                        if not cols[n_col] == "NA":
                            warn("Skipping unconvertible n value %s" % (cols[n_col]))
                        continue
                        
                    if n <= 0:
                        warn("Skipping invalid N value %s" % (n))
                        continue
                    se = 1 / np.sqrt(n)
                elif exomes_n is not None:
                    if exomes_n <= 0:
                        bail("Invalid exomes-n value: %s" % (exomesa_n))
                        continue
                    n = exomes_n
                    se = 1 / np.sqrt(n)

                total_num_genes += 1

                if gene in seen_genes:
                    warn("Gene %s has been seen before; skipping all but first occurrence" % gene)
                    continue
                
                seen_genes.add(gene)
                genes.append(gene)
                gene_ps.append(p)
                gene_betas.append(beta)
                gene_ses.append(se)

            #determine scale_factor
          
            gene_ps = np.array(gene_ps, dtype=float)
            gene_betas = np.array(gene_betas, dtype=float)
            gene_ses = np.array(gene_ses, dtype=float)

            (gene_ps, gene_betas, gene_ses) = pegs_complete_p_beta_se(
                gene_ps,
                gene_betas,
                gene_ses,
                warn_fn=warn,
            )

            gene_zs = gene_betas / gene_ses

            gene_ses2 = np.square(gene_ses)

            log("Done reading --exomes-in", DEBUG)

            gene_log_bfs = -np.log(np.sqrt(1 + allelic_var_k)) + 0.5 * np.square(gene_zs) * allelic_var_k / (1 + allelic_var_k)

            max_log = 15
            gene_log_bfs[gene_log_bfs > max_log] = max_log

            #set lower bound not here but below; otherwise it gets inflated above background
            #gene_log_bfs[gene_log_bfs < 0] = 0

            gene_post = np.exp(gene_log_bfs + np.log(exomes_prior_odds))
            gene_probs = gene_post / (gene_post + 1)
            gene_probs[gene_probs < self.background_prior] = self.background_prior

            #gene_probs_sum = np.sum(gene_probs)

            absent_genes = set()
            if self.genes is not None:
                #have to account for these
                absent_genes = set(self.genes) - set(genes)
            #gene_probs_sum += self.background_prior * len(absent_genes)

            norm_constant = 1
            #norm_constant = (self.background_prior * (len(gene_probs) + len(absent_genes))) / gene_probs_sum
            #need at least 1000 genes
            #if len(gene_probs) < 1000:
            #    norm_constant = 1
            #gene_probs *= norm_constant


            gene_log_bfs = np.log(gene_probs / (1 - gene_probs)) - self.background_log_bf

            absent_prob = self.background_prior * norm_constant
            absent_log_bf = np.log(absent_prob / (1 - absent_prob)) - self.background_log_bf

            if self.genes is not None:
                gene_bf = np.array([np.nan] * len(self.genes))
            else:
                gene_bf = np.array([])

            extra_gene_bf = []
            extra_genes = []
            self.gene_to_exomes_huge_score = {}

            for i in range(len(genes)):
                gene = genes[i]
                bf = gene_log_bfs[i]
                self.gene_to_exomes_huge_score[gene] = bf
                if self.genes is not None and gene in self.gene_to_ind:
                    gene_bf[self.gene_to_ind[gene]] = bf
                else:
                    extra_gene_bf.append(bf)
                    extra_genes.append(gene)
            for gene in absent_genes:
                bf = absent_log_bf
                self.gene_to_exomes_huge_score[gene] = bf
                if self.genes is not None and gene in self.gene_to_ind:
                    gene_bf[self.gene_to_ind[gene]] = bf
                else:
                    extra_gene_bf.append(bf)
                    extra_genes.append(gene)
            extra_gene_bf = np.array(extra_gene_bf)

            self.combine_huge_scores()
            return (gene_bf, extra_genes, extra_gene_bf)

    def read_positive_controls(self, positive_controls_in, positive_controls_id_col=None, positive_controls_prob_col=None, positive_controls_default_prob=0.95, positive_controls_has_header=True, positive_controls_list=None, positive_controls_all_in=None, positive_controls_all_id_col=None, positive_controls_all_has_header=True, hold_out_chrom=None, gene_loc_file=None, **kwargs):
        if positive_controls_in is None and positive_controls_list is None:
            bail("Require --positive-controls-in or --positive-controls-list for this operation")

        if hold_out_chrom is not None and self.gene_to_chrom is None:
            (self.gene_chrom_name_pos, self.gene_to_chrom, self.gene_to_pos) = pegs_read_loc_file_with_gene_map(
                gene_loc_file,
                gene_label_map=self.gene_label_map,
                clean_chrom_fn=pegs_clean_chrom_name,
                warn_fn=warn,
                bail_fn=bail,
            )

        if positive_controls_default_prob >= 1:
            positive_controls_default_prob = 0.99
        if positive_controls_default_prob <= 0:
            positive_controls_default_prob = 0.01

        self.gene_to_positive_controls = {}
        if positive_controls_list is not None:
            for gene in positive_controls_list:
                self.gene_to_positive_controls[gene] = np.log(positive_controls_default_prob / (1 - positive_controls_default_prob)) - self.background_log_bf

        positive_control_files = []
        if positive_controls_in is not None:
            positive_control_files.append((positive_controls_in, positive_controls_id_col, positive_controls_prob_col, positive_controls_default_prob, positive_controls_has_header))
        if positive_controls_all_in is not None:
            positive_control_files.append((positive_controls_all_in, positive_controls_all_id_col, None, self.background_prior, positive_controls_all_has_header))

        for (cur_positive_controls_in, cur_id_col, cur_prob_col, default_prob, has_header) in positive_control_files:
            log("Reading --positive-controls-in file %s" % cur_positive_controls_in, INFO)

            with open_gz(cur_positive_controls_in) as positive_controls_fh:
                id_col = 0
                prob_col = None
                seen_header = False
                for line in positive_controls_fh:
                    cols = line.strip('\n').split()
                    if not seen_header:
                        seen_header = True
                        if has_header or len(cols) > 1:
                            if len(cols) > 1 and cur_id_col is None:
                                bail("--positive-controls-id-col required for positive control files with more than one column")
                            elif cur_id_col is not None:
                                id_col = _get_col(cur_id_col, cols)

                            if cur_prob_col is not None:
                                prob_col = _get_col(cur_prob_col, cols)

                            if has_header and cur_id_col is not None:
                                continue

                    if id_col >= len(cols):
                        warn("Skipping due to too few columns in line: %s" % line)
                        continue

                    gene = cols[id_col]

                    if self.gene_label_map is not None and gene in self.gene_label_map:
                        gene = self.gene_label_map[gene]

                    if hold_out_chrom is not None and gene in self.gene_to_chrom and self.gene_to_chrom[gene] == hold_out_chrom:
                        continue

                    prob = default_prob
                    if prob_col is not None and prob_col >= len(cols):
                        warn("Skipping due to too few columns in line: %s" % line)
                        continue

                    max_prob = 0.99
                    min_prob = 1e-4 * self.background_prior
                    if prob_col is not None:
                        try:
                            prob = float(cols[prob_col])
                            if prob <= 0 or prob >= 1:
                                if prob > max_prob:
                                    warn("Probabilities must be in (0,1); observed %s for %s and reset to %s" % (prob, gene, max_prob))
                                    prob = max_prob
                                if prob < min_prob:
                                    warn("Probabilities must be in (0,1); observed %s for %s and reset to %s" % (prob, gene, min_prob))
                                    prob = min_prob
                                continue
                        except ValueError:
                            if not cols[prob_col] == "NA":
                                warn("Skipping unconvertible prob value %s for gene %s" % (cols[prob_col], gene))
                            continue

                    if prob > max_prob:
                        prob = max_prob
                    log_bf = np.log(prob / (1 - prob)) - self.background_log_bf
                    if gene not in self.gene_to_positive_controls:
                        self.gene_to_positive_controls[gene] = log_bf

        if self.genes is not None:
            genes = self.genes
            gene_to_ind = self.gene_to_ind
        else:
            genes = []
            gene_to_ind = {}

        positive_controls = np.array([np.nan] * len(genes))
        
        extra_positive_controls = []
        extra_genes = []
        for gene in self.gene_to_positive_controls:
            log_bf = self.gene_to_positive_controls[gene]
            if gene in gene_to_ind:
                positive_controls[gene_to_ind[gene]] = log_bf
            else:
                extra_positive_controls.append(log_bf)
                extra_genes.append(gene)

        return (positive_controls, extra_genes, np.array(extra_positive_controls))

    #written by o3
    def read_count_file(self, case_counts_in, ctrl_counts_in, min_revels=None, mean_rrs=None, case_counts_gene_col=None, ctrl_counts_gene_col=None, case_counts_revel_col=None, ctrl_counts_revel_col=None, case_counts_count_col=None, ctrl_counts_count_col=None, case_counts_tot_col=None, ctrl_counts_tot_col=None, case_counts_max_freq_col=None, ctrl_counts_max_freq_col=None, max_case_freq=0.001, max_ctrl_freq=0.001, syn_revel_threshold=0, syn_fisher_p=1e-4, nu=1, beta=1.0, hold_out_chrom=None, gene_loc_file=None, bound_zero=True, **kwargs):

        if hold_out_chrom is not None and self.gene_to_chrom is None:
            (self.gene_chrom_name_pos, self.gene_to_chrom, self.gene_to_pos) = pegs_read_loc_file_with_gene_map(
                gene_loc_file,
                gene_label_map=self.gene_label_map,
                clean_chrom_fn=pegs_clean_chrom_name,
                warn_fn=warn,
                bail_fn=bail,
            )

        if min_revels is None or len(min_revels) == 0:
            min_revels = [0.4, 0.7, 1]
            mean_rrs = [6, 10, 20]

        if mean_rrs is None or len(mean_rrs) != len(min_revels):
            bail("When you pass --counts-min-revels you must also pass an equally long --counts-mean-rrs")

        #sort
        paired = sorted(zip(min_revels, mean_rrs), key=lambda x: -x[0])
        min_revels, mean_rrs = map(list, zip(*paired))    # unzip back to two lists

        if type(mean_rrs) is list:
            mean_rrs = np.array(mean_rrs)
        if type(min_revels) is list:
            min_revels = np.array(min_revels)


        # -------- helper to read one table and return dict -------------
        def __process_count_file(cur_in, gene_col_in, revel_col_in, count_col_in, total_col_in, tag, max_freq_col_in=None, max_freq=0.001, syn_revel_threshold=0, syn_fisher_p=1e-4):
            log("Reading %s counts from %s" % (tag, cur_in), INFO)
            gmap = {}
            syn_map = {}
            #store gene to max n
            max_n = 0
            min_max_freq = 1
            with open_gz(cur_in) as fh:
                header_cols = fh.readline().strip('\n').split()
                
                if gene_col_in is not None:
                    gene_col = _get_col(gene_col_in, header_cols, True)
                else:
                    gene_col = _get_col("gene", header_cols, False)
                    if gene_col is None:
                        bail("Require gene col for %s counts" % tag)

                if revel_col_in is not None:
                    revel_col = _get_col(revel_col_in, header_cols, True)
                else:
                    revel_col = _get_col("revel", header_cols, False)
                    if revel_col is None:
                        bail("Require revel col for %s counts" % tag)

                if count_col_in is not None:
                    count_col = _get_col(count_col_in, header_cols, True)
                else:
                    count_col = _get_col("count", header_cols, False)
                    if count_col is None:
                        bail("Require count col for %s counts" % tag)

                if total_col_in is not None:
                    total_col = _get_col(total_col_in, header_cols, True)
                else:
                    total_col = _get_col("total", header_cols, False)
                    if total_col is None:
                        bail("Require total col for %s counts" % tag)

                max_freq_col = None
                if max_freq_col_in is not None:
                    max_freq_col = _get_col(max_freq_col_in, header_cols, True)

                #first get max_N
                for line in fh:
                    line = line.strip('\n')
                    cols = line.split('\t')
                    try:
                        N = int(cols[total_col])
                    except ValueError:
                        warn("Skipping unconvertible value %s for %s total" % (cols[total_col], tag))
                        continue
                    if max_freq_col is not None:
                        try:
                            cur_max_freq = float(cols[max_freq_col])
                        except ValueError:
                            warn("Skipping unconvertible value %s for %s max freq" % (cols[max_freq_col], tag))
                            continue
                        
                    if cur_max_freq < min_max_freq:
                        min_max_freq = cur_max_freq

                    if N > max_n:
                        max_n = N

                #update the max_freq threshold as needed
                if max_freq < 1.0 / max_n:
                    max_freq = 2.0 / max_n
                    log("Setting max_freq to 1.0/max_N = %.3g" % max_freq, DEBUG)
                if max_freq < min_max_freq:
                    max_freq = min_max_freq
                    log("Setting max_freq to min max freq = %.3g" % max_freq, DEBUG)

            with open_gz(cur_in) as fh:
                header_cols = fh.readline().strip('\n').split()

                for line in fh:
                    line = line.strip('\n')
                    cols = line.split('\t')

                    gene  = cols[gene_col]
                    try:
                        revel = float(cols[revel_col])
                    except ValueError:
                        warn("Skipping unconvertible value %s for %s revel" % (cols[revel_col], tag))
                        continue
                    try:
                        k = int(cols[count_col])
                    except ValueError:
                        warn("Skipping unconvertible value %s for %s count" % (cols[count_col], tag))
                        continue
                    try:
                        N = int(cols[total_col])
                    except ValueError:
                        warn("Skipping unconvertible value %s for %s total" % (cols[total_col], tag))
                        continue

                    if max_freq_col is not None:
                        try:
                            cur_max_freq = float(cols[max_freq_col])
                        except ValueError:
                            warn("Skipping unconvertible value %s for %s max freq" % (cols[max_freq_col], tag))
                            continue

                        if cur_max_freq > max_freq:
                            continue
                    #if gene == "ACTA1":
                    #    print(gene, k, N, revel)

                    freq = 0
                    if N > 0:
                        freq = float(k) / N

                    #if gene == "ACTA1":
                    #    print(gene, k, N, revel)

                    if self.gene_label_map is not None and gene in self.gene_label_map:
                        gene = self.gene_label_map[gene]
                    if (hold_out_chrom is not None and gene in self.gene_to_chrom and self.gene_to_chrom[gene] == hold_out_chrom):
                        continue

                    if gene not in gmap:
                        gmap[gene] = {}
                        for threshold in sorted(min_revels):
                            gmap[gene][threshold] = [0, 0]
                        syn_map[gene] = [0, 0]

                    if threshold < syn_revel_threshold:
                        syn_map[gene][0] += k
                        syn_map[gene][0] += freq

                    for threshold in min_revels:
                        if revel >= threshold:
                            #store total counts and aggregate frequency
                            gmap[gene][threshold][0] += k
                            gmap[gene][threshold][1] += freq
                            #only add bin to the top one
                            break

            #now convert the second entry in the map to the total count
            for gene in gmap:
                for threshold in min_revels:
                    #count / N = freq -> N = count / freq
                    if gmap[gene][threshold][1] > 0:
                        cur_N = gmap[gene][threshold][0] / gmap[gene][threshold][1]
                        gmap[gene][threshold][1] = cur_N
                    else:
                        gmap[gene][threshold][1] = max_n
                if syn_map[gene][1] > 0:
                    cur_N = syn_map[gene][0] / syn_map[gene][1]
                    syn_map[gene][1] = cur_N
                else:
                    syn_map[gene][1] = max_n

            return gmap, max_n, syn_map

        #read in the case and control counts
        case_map, max_case_n, case_syn_map = __process_count_file(case_counts_in, case_counts_gene_col, case_counts_revel_col, case_counts_count_col, case_counts_tot_col, "case", case_counts_max_freq_col, max_case_freq, syn_revel_threshold)
        ctrl_map, max_ctrl_n, ctrl_syn_map = __process_count_file(ctrl_counts_in, ctrl_counts_gene_col, ctrl_counts_revel_col, ctrl_counts_count_col, ctrl_counts_tot_col, "ctrl", ctrl_counts_max_freq_col, max_ctrl_freq, syn_revel_threshold)

        def __tada_log_bf(
            k_case,    # array shape (G, T): case allele counts
            N_case,    # array shape (G, T): case chromosome totals
            k_ctrl,    # array shape (G, T): control allele counts
            N_ctrl,    # array shape (G, T): control chromosome totals
            mean_rrs,  # array shape (T,): prior mean relative risks per threshold
            rho0, nu0, # floats: Gamma prior shape/rate for q under H0
            rho1, nu1, # floats: Gamma prior shape/rate for q under H1
            beta=1.0,      # float: dispersion parameter for NB under H1
            q_lower=1e-8,
            q_upper=1,
            num_grid=1000,
            debug_genes=None,
            debug_gene=None
        ):

            """
            Vectorized numeric-integration TADA Bayes factor (log10) for case/control.

            Returns
            -------
            BF : array shape (G, T)
                Bayes factor for each gene and threshold.
            """

            if len(np.shape(rho0)) == 1:
                rho0 = rho0[np.newaxis, :]
            if len(np.shape(rho1)) == 1:
                rho1 = rho1[np.newaxis, :]

            # Null model marginals
            p0_ctrl = nu0 / (nu0 + N_ctrl)
            m0_ctrl = scipy.stats.nbinom.pmf(k_ctrl, rho0, p0_ctrl)

            p0_case = (nu0 + N_ctrl) / (nu0 + N_ctrl + N_case)
            m0_case = scipy.stats.nbinom.pmf(k_case, rho0 + k_ctrl, p0_case)

            # Alternative model control marginal
            p1_ctrl = nu1 / (nu1 + N_ctrl)
            m1_ctrl = scipy.stats.nbinom.pmf(k_ctrl, rho1, p1_ctrl)

            # Prepare q integration grid
            u = np.linspace(np.log(q_lower), np.log(q_upper), num_grid)
            q = np.exp(u)

            # Broadcast to 3D arrays (G, T, M)
            kc = k_case[:, :, np.newaxis]
            Nc = N_case[:, :, np.newaxis]
            k0 = k_ctrl[:, :, np.newaxis]
            N0 = N_ctrl[:, :, np.newaxis]
            rho0 = rho0[:, :, np.newaxis]
            rho1 = rho1[:, :, np.newaxis]


            # Case model under H1: NB * Gamma(q) integrated over q
            gamma_mean = mean_rrs[np.newaxis, :, np.newaxis]

            #a = 6.83
            #b = -1.29
            #c = -0.58
            #be ta = np.exp(a * np.power(gamma_mean, b) + c)

            #size = gamma_mean * beta
            size = beta

            #prob = beta / (beta + Nc * q[np.newaxis, np.newaxis, :])
            prob = beta / (beta + gamma_mean * Nc * q[np.newaxis, np.newaxis, :])

            pmf_case = scipy.stats.nbinom.pmf(kc, size, prob)

            a = rho1 + k0
            scale = 1.0 / (nu1 + N0)

            pdf_q = scipy.stats.gamma.pdf(q[np.newaxis, np.newaxis, :], a=a, scale=scale)

            integrand = pmf_case * pdf_q * q[np.newaxis, np.newaxis, :]
            m1_case = np.trapz(integrand, u, axis=2)

            # Bayes factor = (m1_ctrl/m0_ctrl) * (m1_case/m0_case)

            protective_mask = k_case / N_case < k_ctrl / N_ctrl
            #handle underflow in demoninator by setting everything to 1 if numerator is also 0 or if protective

            ctrl_zero_mask = np.logical_and(m0_ctrl == 0, np.logical_or(m1_ctrl == 0, protective_mask))
            m0_ctrl[ctrl_zero_mask] = 1
            m1_ctrl[ctrl_zero_mask] = 1
            case_zero_mask = np.logical_and(m0_case == 0, np.logical_or(m1_case == 0, protective_mask))
            m0_case[case_zero_mask] = 1
            m1_case[case_zero_mask] = 1
            

            #print(m1_ctrl, m0_ctrl, m1_case, m0_case)
            #print(k_case, N_case, k_case/N_case)
            #print(k_ctrl, N_ctrl, k_ctrl/N_ctrl)
            for i in range(len(genes)):
                for j in range(m1_case.shape[1]):
                    if m0_ctrl[i,j] == 0 or m0_case[i,j] == 0:
                        print(genes[i], m1_ctrl[i,j], m0_ctrl[i,j], m1_case[i,j], m0_case[i,j], k_ctrl[i,j]/N_ctrl[i,j], k_case[i,j]/N_case[i,j])

            BF = (m1_ctrl / m0_ctrl) * (m1_case / m0_case)

            #we want to ensure that if the case frequency is below 1, it always has BF = 1
            BF[np.logical_and(BF > 1, protective_mask)] = 1

            max_BF = 1e4
            BF[BF > max_BF] = max_BF

            #print(BF)

            return np.log(BF)


        # ------------- compute per-gene log10 BF -----------------------
        self.gene_to_case_count_logbf = {}

        #now assemble them into four big matrices of gene x threshold
        genes = list(case_map.keys() | ctrl_map.keys())
        count_shape = (len(genes), len(min_revels))
        case_k = np.zeros(count_shape)
        case_N = np.full(count_shape, max_case_n)
        ctrl_k = np.zeros(count_shape)
        ctrl_N = np.full(count_shape, max_ctrl_n)

        case_syn_k = np.zeros(count_shape[0])
        case_syn_N = np.full(count_shape[0], max_case_n)
        ctrl_syn_k = np.zeros(count_shape[0])
        ctrl_syn_N = np.full(count_shape[0], max_ctrl_n)

        for i in range(len(genes)):
            for j in range(len(min_revels)):
                threshold = min_revels[j]
                gene = genes[i]
                if gene in case_map:
                    assert(threshold in case_map[gene])
                    case_k[i,j] = case_map[gene][threshold][0]
                    case_N[i,j] = case_map[gene][threshold][1]
                if gene in ctrl_map:
                    assert(threshold in ctrl_map[gene])
                    ctrl_k[i,j] = ctrl_map[gene][threshold][0]
                    ctrl_N[i,j] = ctrl_map[gene][threshold][1]
            if gene in case_map:
                case_syn_k[i] = case_syn_map[gene][0]
                case_syn_N[i] = case_syn_map[gene][1]
            if gene in ctrl_map:
                ctrl_syn_k[i] = ctrl_syn_map[gene][0]
                ctrl_syn_N[i] = ctrl_syn_map[gene][1]

        #set rho and nu as in TADA
        #rho / nu = mean frequency across genes
        rho = nu * np.mean((case_k + ctrl_k) / (case_N + ctrl_N), axis=0)

        case_count_logbf = __tada_log_bf(case_k, case_N, ctrl_k, ctrl_N, mean_rrs, rho, nu, rho, nu, beta=beta)
        case_count_logbf_tot = case_count_logbf.sum(axis=1)

        #filter out genes that look like artifacts
        remove_mask = np.full(case_count_logbf_tot.shape, False)
        #remove things that fail a fisher test for synonymous variants

        for i in range(len(genes)):
            if ctrl_syn_N[i] == 0 or case_syn_N[i] == 0:
                p_value = 1.0
            else:
                table = [[case_syn_k[i],  case_syn_N[i] - case_syn_k[i]],
                         [ctrl_syn_k[i],  ctrl_syn_N[i] - ctrl_syn_k[i]]]
                _, p_value = scipy.stats.fisher_exact(table, alternative="two-sided")
            if p_value < syn_fisher_p:
                remove_mask[i] = True

        log("Removed %d genes due to failing syn fisher test" % np.sum(remove_mask))

        #remove things that have revel values for only one of them
        remove_mask[np.sum(case_k, axis=1) + np.sum(ctrl_k, axis=1) == 0] = True
        log("Removed %d genes due to lacking revel values" % np.sum(remove_mask))

        case_count_logbf_tot[remove_mask] = 0

        if bound_zero:
            case_count_logbf_tot[case_count_logbf_tot < 0] = 0

        for i in range(len(genes)):
            gene = genes[i]
            if not np.isnan(case_count_logbf_tot[i]):
                self.gene_to_case_count_logbf[gene] = case_count_logbf_tot[i]
            

        # ------------- align to self.genes & return -------------------


        if self.genes is not None:
            case_count_bfs = np.array([np.nan] * len(self.genes))
            for i, g in enumerate(self.genes):
                if g in self.gene_to_case_count_logbf:
                    case_count_bfs[i] = self.gene_to_case_count_logbf[g]
            
        else:
            case_count_bfs = np.array([])

        extra_genes = []
        extra_bfs = []
        for g in self.gene_to_case_count_logbf:
            if self.gene_to_ind is None or g not in self.gene_to_ind:
                extra_genes.append(g)
                extra_bfs.append(self.gene_to_case_count_logbf[g])

        if len(case_count_bfs) == 0 and len(extra_bfs) == 0:
            bail("Error: no genes passed filters")

        return (case_count_bfs, extra_genes, np.array(extra_bfs))


    def compute_allelic_var_and_prior(self, high_p, high_p_posterior, low_p, low_p_posterior):

        if high_p < low_p:
            warn("Swapping high_p and low_p")
            temp = high_p
            high_p = low_p
            low_p = temp

        if high_p == low_p:
            high_p = low_p * 2

        if high_p_posterior >= 1:
            po_high = 0.99/0.01
        elif high_p_posterior <=0 :
            po_high = 0.001/0.999
        else:
            po_high = high_p_posterior / (1 - high_p_posterior)

        if low_p_posterior >= 1:
            po_low = 0.99/0.01
        elif low_p_posterior <=0 :
            po_low = 0.001/0.999
        else:
            po_low = low_p_posterior / (1 - low_p_posterior)

        z_high = np.abs(scipy.stats.norm.ppf(high_p/2))
        z_low = np.abs(scipy.stats.norm.ppf(low_p/2))
        ratio = po_low / po_high

        allelic_var_k = 2 * np.log(ratio) / (np.square(z_low) - np.square(z_high))

        if allelic_var_k > 1:
            #reset high_p_posterior
            max_allelic_var_k = 0.99;
            po_high = po_low / np.exp(max_allelic_var_k * (np.square(z_low) - np.square(z_high)) / 2)
            log("allelic_var_k overflow; adjusting --high-p-posterior to %.4g" % (po_high/(1+po_high)))
            ratio = po_low / po_high
            allelic_var_k = 2 * np.log(ratio) / (np.square(z_low) - np.square(z_high))

        allelic_var_k = allelic_var_k / (1 - allelic_var_k)

        prior_odds = po_low / (np.sqrt(1 / (1 + allelic_var_k)) * np.exp(0.5 * np.square(z_low) * (allelic_var_k / (1 + allelic_var_k))))
        
        return (allelic_var_k, prior_odds)


    def combine_huge_scores(self):
        #combine the huge scores if needed
        if self.gene_to_gwas_huge_score is not None and self.gene_to_exomes_huge_score is not None:
            self.gene_to_huge_score = {}
            genes = list(set().union(self.gene_to_gwas_huge_score, self.gene_to_exomes_huge_score))
            for gene in genes:
                self.gene_to_huge_score[gene] = 0
                if gene in self.gene_to_gwas_huge_score:
                    self.gene_to_huge_score[gene] += self.gene_to_gwas_huge_score[gene]
                if gene in self.gene_to_exomes_huge_score:
                    self.gene_to_huge_score[gene] += self.gene_to_exomes_huge_score[gene]

    def calculate_gene_set_statistics(self, gwas_in=None, exomes_in=None, positive_controls_in=None, positive_controls_list=None, case_counts_in=None, ctrl_counts_in=None, gene_bfs_in=None, Y=None, show_progress=True, max_gene_set_p=None, run_logistic=True, max_for_linear=0.95, run_corrected_ols=False, use_sampling_for_betas=None, correct_betas_mean=True, correct_betas_var=True, gene_loc_file=None, gene_cor_file=None, gene_cor_file_gene_col=1, gene_cor_file_cor_start_col=10, skip_V=False, run_using_phewas=False, **kwargs):
        if self.X_orig is None:
            bail("Error: X is required")
        #now calculate the betas and p-values

        log("Calculating gene set statistics", INFO)

        if run_using_phewas:
            Y = self.gene_pheno_Y.T.toarray()
            if Y is None:
                bail("Need --gene-phewas-bfs in order to run beta calculation with phewas")

        if Y is None:
            Y = self.Y_for_regression

        if Y is None:
            if gwas_in is None and exomes_in is None and gene_bfs_in is None and positive_controls_in is None and positive_controls_list is None and case_counts_in is None and ctrl_counts_in is None:
                bail("Need --gwas-in or --exomes-in or --gene-stats-in or --positive-controls-in or --case-counts_in")

            log("Reading Y within calculate_gene_set_statistics; parameters may not be honored")
            _run_read_y_stage(
                self,
                gwas_in=gwas_in,
                exomes_in=exomes_in,
                positive_controls_in=positive_controls_in,
                positive_controls_list=positive_controls_list,
                case_counts_in=case_counts_in,
                ctrl_counts_in=ctrl_counts_in,
                gene_bfs_in=gene_bfs_in,
                **kwargs
            )
            Y = self.Y_for_regression

        if run_corrected_ols and self.y_corr is None:
            correlation_m = self._read_correlations(gene_cor_file, gene_loc_file, gene_cor_file_gene_col=gene_cor_file_gene_col, gene_cor_file_cor_start_col=gene_cor_file_cor_start_col)

            #convert X and Y to their new values
            min_correlation = 0.05
            self._set_Y(self.Y, self.Y_for_regression, self.Y_exomes, self.Y_positive_controls, self.Y_case_counts, Y_corr_m=correlation_m, store_corr_sparse=run_corrected_ols, skip_V=True, skip_scale_factors=True, min_correlation=min_correlation)

        #subset gene sets to remove empty ones first
        #number of gene sets in each gene set
        col_sums = self.get_col_sums(self.X_orig, num_nonzero=True)
        self.subset_gene_sets(col_sums > 0, keep_missing=False, skip_V=True, skip_scale_factors=True)

        self._set_scale_factors()

        #self.is_logistic = run_logistic

        #if the maximum Y is large, switch to logistic regression (to avoid being too strong)
        Y_to_use = Y
        Y = np.exp(Y_to_use + self.background_log_bf) / (1 + np.exp(Y_to_use + self.background_log_bf))

        if not run_logistic and np.max(Y) > max_for_linear and (use_sampling_for_betas is None or use_sampling_for_betas < 1):
            log("Switching to logistic sampling due to high Y values", DEBUG)
            run_logistic = True
            use_sampling_for_betas = 1

        if use_sampling_for_betas is not None:
            self._record_param("sampling_for_betas", use_sampling_for_betas)

        if use_sampling_for_betas is not None and use_sampling_for_betas > 0:

            #handy option in case we want to see what sampling looks like outside of gibbs
            if run_using_phewas:
                avg_beta_tildes = np.zeros((self.gene_pheno_Y.shape[1],len(self.gene_sets)))
                avg_z_scores = np.zeros((self.gene_pheno_Y.shape[1],len(self.gene_sets)))
            else:
                avg_beta_tildes = np.zeros(len(self.gene_sets))
                avg_z_scores = np.zeros(len(self.gene_sets))
            tot_its = 0
            for iteration_num in range(use_sampling_for_betas):
                log("Sampling iteration %d..." % (iteration_num+1))
                p_sample_m = np.zeros(Y.shape)
                p_sample_m[np.random.random(Y.shape) < Y] = 1
                Y_sample_m = p_sample_m

                (beta_tildes, ses, z_scores, p_values, se_inflation_factors, alpha_tildes, diverged) = self._compute_logistic_beta_tildes(self.X_orig, Y_sample_m, self.scale_factors, self.mean_shifts, resid_correlation_matrix=self.y_corr_sparse)

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
            (beta_tildes, ses, z_scores, p_values, se_inflation_factors, alpha_tildes, diverged) = self._compute_logistic_beta_tildes(self.X_orig, Y, self.scale_factors, self.mean_shifts, resid_correlation_matrix=self.y_corr_sparse)

        else:
            #Technically, we could use the above code for this case, since X_blocks will returned unwhitened matrix
            #But, probably faster to keep sparse multiplication? Might be worth revisiting later to see if there actually is a performance gain
            #We can use original X here because whitening support was removed with GLS.
            assert(not self.scale_is_for_whitened)
            Y = copy.copy(Y)

            if len(Y.shape) > 1:
                y_var = np.var(Y, axis=1)
            else:
                y_var = np.var(Y)

            (beta_tildes, ses, z_scores, p_values, se_inflation_factors) = self._compute_beta_tildes(self.X_orig, Y, y_var, self.scale_factors, self.mean_shifts, resid_correlation_matrix=self.y_corr_sparse)

        if correct_betas_mean or correct_betas_var:
            (beta_tildes, ses, z_scores, p_values, se_inflation_factors) = self._correct_beta_tildes(beta_tildes, ses, se_inflation_factors, self.total_qc_metrics, self.total_qc_metrics_directions, correct_mean=correct_betas_mean, correct_var=correct_betas_var, fit=False)

        if run_using_phewas:
            (self.beta_tildes_phewas, self.z_scores_phewas, self.p_values_phewas, self.ses_phewas, self.se_inflation_factors_phewas) = (beta_tildes, z_scores, p_values, ses, se_inflation_factors)
            if len(self.beta_tildes_phewas.shape) == 1:
                self.beta_tildes_phewas = self.beta_tildes_phewas[np.newaxis,:]
                self.ses_phewas = self.ses_phewas[np.newaxis,:]
                self.z_scores_phewas = self.z_scores_phewas[np.newaxis,:]
                self.p_values_phewas = self.p_values_phewas[np.newaxis,:]
                if self.se_inflation_factors_phewas is not None:
                    self.se_inflation_factors_phewas = self.se_inflation_factors_phewas[np.newaxis,:]
        else:
            (self.beta_tildes, self.z_scores, self.p_values, self.ses, self.se_inflation_factors) = (beta_tildes, z_scores, p_values, ses, se_inflation_factors)

            self.X_orig_missing_gene_sets = None
            self.mean_shifts_missing = None
            self.scale_factors_missing = None
            self.is_dense_gene_set_missing = None
            self.ps_missing = None
            self.sigma2s_missing = None

            self.beta_tildes_missing = None
            self.p_values_missing = None
            self.ses_missing = None
            self.z_scores_missing = None

            self.total_qc_metrics_missing = None
            self.mean_qc_metrics_missing = None

            if max_gene_set_p is not None:
                gene_set_mask = self.p_values <= max_gene_set_p
                if np.sum(gene_set_mask) == 0 and len(self.p_values) > 0:
                    gene_set_mask = self.p_values == np.min(self.p_values)
                log("Keeping %d gene sets that passed threshold of p<%.3g" % (np.sum(gene_set_mask), max_gene_set_p))
                self.subset_gene_sets(gene_set_mask, keep_missing=True, skip_V=True)

                if len(self.gene_sets) < 1:
                    log("No gene sets left!")
                    return

        #self.max_gene_set_p = max_gene_set_p

    def has_gene_sets(self):
        return self.X_orig is not None and self.X_orig.shape[1] > 0

    def set_p(self, p):
        hyper_state = pegs_set_runtime_p(self, p)
        self.hyperparameter_state = hyper_state

    def get_sigma2(self, convert_sigma_to_external_units=False):
        sigma2 = self.sigma2
        sigma_power = self.sigma_power
        if sigma2 is not None and convert_sigma_to_external_units and sigma_power is not None:
            scale_factors = self.scale_factors
            if scale_factors is not None:
                is_dense_gene_set = self.is_dense_gene_set
                if is_dense_gene_set is not None and np.sum(~is_dense_gene_set) > 0:
                    return sigma2 * np.mean(np.power(scale_factors[~is_dense_gene_set], sigma_power - 2))
                else:
                    return sigma2 * np.mean(np.power(scale_factors, sigma_power - 2))
            else:
                return sigma2 * np.power(self.MEAN_MOUSE_SCALE, sigma_power - 2)

        return sigma2

    def get_scaled_sigma2(self, scale_factors, sigma2, sigma_power, sigma_threshold_k=None, sigma_threshold_xo=None):
        threshold = 1
        if sigma_threshold_k is not None and sigma_threshold_xo is not None:
            threshold =  1 / (1 + np.exp(-sigma_threshold_k * (scale_factors - sigma_threshold_xo)))

        zero_mask = None
        if len(scale_factors.shape) == 0:
            if scale_factors == 0:
                return 0
        else:
            zero_mask = scale_factors == 0
            scale_factors[zero_mask] = 1

        result = threshold * sigma2 * np.power(scale_factors, sigma_power)
        if zero_mask is not None:
            result[zero_mask] = 0

        return result

    def set_sigma(self, sigma2, sigma_power, sigma2_osc=None, sigma2_se=None, sigma2_p=None, sigma2_scale_factors=None, convert_sigma_to_internal_units=False):
        hyper_state = pegs_set_runtime_sigma(
            self,
            sigma2,
            sigma_power,
            sigma2_osc=sigma2_osc,
            sigma2_se=sigma2_se,
            sigma2_p=sigma2_p,
            sigma2_scale_factors=sigma2_scale_factors,
            convert_sigma_to_internal_units=convert_sigma_to_internal_units,
        )
        self.hyperparameter_state = hyper_state
        if hyper_state.sigma2 is None:
            return

    def write_params(self, output_file):
        if output_file is not None:
            log("Writing params to %s" % output_file, INFO)
            params_fh = open(output_file, 'w')

            params_fh.write("Parameter\tVersion\tValue\n")
            for param in self.param_keys:
                if type(self.params[param]) == list:
                    values = self.params[param]
                else:
                    values = [self.params[param]]
                for i in range(len(values)):
                    params_fh.write("%s\t%s\t%s\n" % (param, i + 1, values[i]))
                        
            params_fh.close()

    def read_betas(self, betas_in):

        betas_format = "<gene_set_id> <beta>"

        if self.betas_in is None:
            bail("Operation requires --beta-in\nformat: %s" % (self.betas_format))

        log("Reading --betas-in file %s" % self.betas_in, INFO)

        with open_gz(betas_in) as betas_fh:
            id_col = 0
            beta_col = 1

            if self.gene_sets is not None:
                self.betas = np.zeros(len(self.gene_sets))
                subset_mask = np.array([False] * len(self.gene_sets))
            else:
                self.betas = []

            gene_sets = []
            gene_set_to_ind = {}

            ignored = 0
            for line in betas_fh:
                cols = line.strip('\n').split()
                if id_col > len(cols) or beta_col > len(cols):
                    warn("Skipping due to too few columns in line: %s" % line)
                    continue

                gene_set = cols[id_col]
                if gene_set in gene_set_to_ind:
                    warn("Already seen gene set %s; only considering first instance" % (gene_set))
                try:
                    beta = float(cols[beta_col])
                except ValueError:
                    if not cols[beta_col] == "NA":
                        warn("Skipping unconvertible beta value %s for gene_set %s" % (cols[beta_col], gene_set))
                    continue
                
                gene_set_ind = None
                if self.gene_sets is not None:
                    if gene_set not in self.gene_set_to_ind:
                        ignored += 1
                        continue
                    gene_set_ind = self.gene_set_to_ind[gene_set]
                    if gene_set_ind is not None:
                        self.betas[gene_set_ind] = beta
                        subset_mask[gene_set_ind] = True
                else:
                    self.betas.append(beta)
                    #store these in all cases to be able to check for duplicate gene sets in the input
                    gene_set_to_ind[gene_set] = len(gene_sets)
                    gene_sets.append(gene_set)

            if self.gene_sets is not None:
                #need to subset existing marices
                if ignored > 0:
                    warn("Ignored %s values from --betas-in file because absent from previously loaded files" % ignored)
                if sum(subset_mask) != len(subset_mask):
                    warn("Excluding %s values from previously loaded files because absent from --betas-in file" % (len(subset_mask) - sum(subset_mask)))
                    self.subset_gene_sets(subset_mask, keep_missing=False)
            else:
                self.gene_sets = gene_sets
                self.gene_set_to_ind = gene_set_to_ind
                self.betas = np.array(self.betas).flatten()

            if self.normalize_betas:
                self.betas -= np.mean(self.betas)

    def run_cross_val(self, cross_val_num_explore_each_direction, folds=4, cross_val_max_num_tries=2, p=None, max_num_burn_in=1000, max_num_iter=1100, min_num_iter=10, num_chains=4, run_logistic=True, max_for_linear=0.95, run_corrected_ols=False, r_threshold_burn_in=1.01, use_max_r_for_convergence=True, max_frac_sem=0.01, gauss_seidel=False, sparse_solution=False, sparse_frac_betas=None, **kwargs):

        log("Running cross validation", DEBUG)

        if self.sigma2s is not None:
            candidate_sigma2s = self.sigma2s
        elif self.sigma2 is not None:
            candidate_sigma2s = np.array(self.sigma2).reshape((1,))
        else:
           bail("Need to have sigma set before running cross validation")

        if p is None:
           bail("Need to have p set before running cross validation")
        if self.X_orig is None:
           bail("Need to have X_orig set before running cross validation")

        Y_to_use = self.Y_for_regression
        if Y_to_use is None:
            Y_to_use = self.Y

        if Y_to_use is None:
           bail("Need to have Y set before running cross validation")

        
        D = np.exp(Y_to_use + self.background_log_bf) / (1 + np.exp(Y_to_use + self.background_log_bf))
        if not run_logistic and np.max(D) > max_for_linear:
            log("Switching to logistic sampling due to high Y values (max(D) = %.3g" % np.max(D), DEBUG)
            run_logistic = True

        beta_tildes_cv = np.zeros((folds, len(self.gene_sets)))
        alpha_tildes_cv = np.zeros((folds, len(self.gene_sets)))
        ses_cv = np.zeros((folds, len(self.gene_sets)))
        cv_val_masks = np.full((folds, len(Y_to_use)), False)
        for fold in range(folds):
            cv_mask = np.arange(len(Y_to_use)) % folds != fold
            cv_val_masks[fold,:] = ~cv_mask
            X_to_use = self.X_orig[cv_mask,:]
            if run_logistic:
                Y_cv = D[cv_mask]
                (beta_tildes_cv[fold,:], ses_cv[fold,:], _, _, _, alpha_tildes_cv[fold,:], _) = self._compute_logistic_beta_tildes(X_to_use, Y_cv, resid_correlation_matrix=self.y_corr_sparse[cv_mask,:][:,cv_mask])
            else:
                Y_cv = Y_to_use[cv_mask]
                (beta_tildes_cv[fold,:], ses_cv[fold,:], _, _, _) = self._compute_beta_tildes(X_to_use, Y_cv, resid_correlation_matrix=self.y_corr_sparse[cv_mask,:][:,cv_mask])

        #one parallel per sigma value to test
        cross_val_num_explore = cross_val_num_explore_each_direction * 2 + 1
        #for each parallel, need to do it with the different set of Y values
        cross_val_num_explore_with_fold = cross_val_num_explore * folds

        candidate_sigma2s_m = np.tile(candidate_sigma2s, cross_val_num_explore).reshape(cross_val_num_explore, candidate_sigma2s.shape[0])
        candidate_sigma2s_m = (candidate_sigma2s_m.T * np.power(10.0, np.arange(-cross_val_num_explore_each_direction,cross_val_num_explore_each_direction+1))).T
        orig_index = cross_val_num_explore_each_direction

        for try_num in range(cross_val_max_num_tries):

            log("Sigmas to try: %s" % np.mean(candidate_sigma2s_m, axis=1), TRACE)

            #order of parallel is first by explore and then by fold

            #repeat the candidates for each fold
            candidate_sigma2s_m = np.tile(candidate_sigma2s_m, (folds, 1))

            beta_tildes_m = np.repeat(beta_tildes_cv, cross_val_num_explore, axis=0)
            ses_m = np.repeat(ses_cv, cross_val_num_explore, axis=0)
            scale_factors_m = np.tile(self.scale_factors, cross_val_num_explore_with_fold).reshape(cross_val_num_explore_with_fold, len(self.scale_factors))
            mean_shifts_m = np.tile(self.mean_shifts, cross_val_num_explore_with_fold).reshape(cross_val_num_explore_with_fold, len(self.mean_shifts))

            (betas_m, postp_m) = self._calculate_non_inf_betas(initial_p=self.p, beta_tildes=beta_tildes_m, ses=ses_m, scale_factors=scale_factors_m, mean_shifts=mean_shifts_m, sigma2s=candidate_sigma2s_m, max_num_burn_in=max_num_burn_in, max_num_iter=max_num_iter, min_num_iter=min_num_iter, num_chains=num_chains, r_threshold_burn_in=r_threshold_burn_in, use_max_r_for_convergence=use_max_r_for_convergence, max_frac_sem=max_frac_sem, gauss_seidel=gauss_seidel, update_hyper_sigma=False, update_hyper_p=False, sparse_solution=sparse_solution, sparse_frac_betas=sparse_frac_betas, V=self._get_V(), **kwargs)

            rss = np.zeros(cross_val_num_explore)
            num_Y = 0
            #different values for logistic and linear
            Y_val = Y_to_use - np.mean(Y_to_use)

            for fold in range(folds):
                #result is parallel x genes
                output_cv_mask = np.floor(np.arange(betas_m.shape[0]) / cross_val_num_explore) == fold
                cur_pred = self.X_orig[cv_val_masks[fold,:],:].dot((betas_m[output_cv_mask,:] / self.scale_factors).T).T
                rss += np.sum(np.square(cur_pred - Y_val[cv_val_masks[fold,:]]), axis=1)
                num_Y += np.sum(cv_val_masks[fold,:])

            rss /= num_Y
            best_result = np.argmin(rss)
            best_sigma2s = candidate_sigma2s_m[best_result,:]
            log("Got RSS values: %s" % (rss), TRACE)
            log("Best sigma is %.3g" % np.mean(best_sigma2s))
            log("Updating sigma from %.3g to %.3g" % (self.sigma2, np.mean(best_sigma2s)))
            if self.sigma2s is not None:
                self.sigma2s = best_sigma2s
                self.set_sigma(np.mean(best_sigma2s), self.sigma_power)
            else:
                assert(len(best_sigma2s.shape) == 1 and best_sigma2s.shape[0] == 1)
                self.set_sigma(best_sigma2s[0], self.sigma_power)

            if try_num + 1 < cross_val_max_num_tries and (best_result == 0 or best_result == (len(rss) - 1)) and best_result != orig_index:
                log("Expanding search further since best cross validation result was at boundary of search space", DEBUG)
                assert(self.sigma2s is not None or self.sigma2 is not None)
                if self.sigma2s is not None:
                    candidate_sigma2s = self.sigma2s
                else: 
                    candidate_sigma2s = np.array(self.sigma2).reshape((1,))
                candidate_sigma2s_m = np.tile(candidate_sigma2s, cross_val_num_explore).reshape(cross_val_num_explore, candidate_sigma2s.shape[0])
                if best_result == 0:
                    #extend lower
                    candidate_sigma2s_m = (candidate_sigma2s_m.T * np.power(10.0, np.arange(-cross_val_num_explore+1,1))).T
                    orig_index = cross_val_num_explore - 1
                else:
                    #extend higher
                    candidate_sigma2s_m = (candidate_sigma2s_m.T * np.power(10.0, np.arange(cross_val_num_explore))).T
                    orig_index = 0
            else:
                break
        
    def calculate_non_inf_betas(self, p, max_num_burn_in=1000, max_num_iter=1100, min_num_iter=10, num_chains=10, r_threshold_burn_in=1.01, use_max_r_for_convergence=True, max_frac_sem=0.01, gauss_seidel=False, update_hyper_sigma=True, update_hyper_p=True, sparse_solution=False, pre_filter_batch_size=None, pre_filter_small_batch_size=500, sparse_frac_betas=None, betas_trace_out=None, run_betas_using_phewas=False, run_uncorrected_using_phewas=False, **kwargs):

        run_using_phewas = run_betas_using_phewas or run_uncorrected_using_phewas

        log("Calculating betas")
        if run_using_phewas:
            (beta_tildes_to_use, ses_to_use) = (self.beta_tildes_phewas, self.ses_phewas)
        else:
            (beta_tildes_to_use, ses_to_use) = (self.beta_tildes, self.ses)

        if not run_using_phewas or run_uncorrected_using_phewas:
            result_uncorrected = self._calculate_non_inf_betas(p, beta_tildes=beta_tildes_to_use, ses=ses_to_use, max_num_burn_in=max_num_burn_in, max_num_iter=max_num_iter, min_num_iter=min_num_iter, num_chains=num_chains, r_threshold_burn_in=r_threshold_burn_in, use_max_r_for_convergence=use_max_r_for_convergence, max_frac_sem=max_frac_sem, gauss_seidel=gauss_seidel, update_hyper_sigma=False, update_hyper_p=False, sparse_solution=sparse_solution, sparse_frac_betas=sparse_frac_betas, assume_independent=True, V=None, **kwargs)

        avg_betas_v = np.zeros(len(self.gene_sets))
        avg_postp_v = np.zeros(len(self.gene_sets))

        if run_using_phewas:
            initial_run_mask = np.full(len(self.gene_sets), True)
        else:
            (avg_betas_uncorrected_v, avg_postp_uncorrected_v) = result_uncorrected
            initial_run_mask = avg_betas_uncorrected_v != 0

        run_mask = copy.copy(initial_run_mask)

        if pre_filter_batch_size is not None and np.sum(initial_run_mask) > pre_filter_batch_size:
            self._record_param("pre_filter_batch_size_orig", pre_filter_batch_size)

            num_batches = self._get_num_X_blocks(self.X_orig[:,initial_run_mask], batch_size=pre_filter_small_batch_size)
            if num_batches > 1:
                #try to run with small batches to see if we can zero out more
                gene_set_masks = self._compute_gene_set_batches(V=None, X_orig=self.X_orig[:,initial_run_mask], mean_shifts=self.mean_shifts[initial_run_mask], scale_factors=self.scale_factors[initial_run_mask], find_correlated_instead=pre_filter_small_batch_size)
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
                            V_block = self._calculate_V_internal(self.X_orig[:,initial_run_mask][:,gene_set_mask], self.y_corr_cholesky, self.mean_shifts[initial_run_mask][gene_set_mask], self.scale_factors[initial_run_mask][gene_set_mask])
                            orig_indices = np.where(gene_set_mask)[0]
                            V_rows += list(np.repeat(orig_indices, V_block.shape[0]))
                            V_cols += list(np.tile(orig_indices, V_block.shape[0]))
                            V_data += list(V_block.ravel())
                            
                        V_sparse = sparse.csc_matrix((V_data, (V_rows, V_cols)), shape=(np.sum(initial_run_mask), np.sum(initial_run_mask)))

                        log("Running %d blocks to check for zeros..." % len(gene_set_masks), DEBUG)
                        (avg_betas_half_corrected_v, avg_postp_half_corrected_v) = self._calculate_non_inf_betas(p, V=V_sparse, X_orig=None, scale_factors=self.scale_factors[initial_run_mask], mean_shifts=self.mean_shifts[initial_run_mask], is_dense_gene_set=self.is_dense_gene_set[initial_run_mask], ps=self.ps[initial_run_mask], sigma2s=self.sigma2s[initial_run_mask], max_num_burn_in=max_num_burn_in, max_num_iter=max_num_iter, min_num_iter=min_num_iter, num_chains=num_chains, r_threshold_burn_in=r_threshold_burn_in, use_max_r_for_convergence=use_max_r_for_convergence, max_frac_sem=max_frac_sem, gauss_seidel=gauss_seidel, update_hyper_sigma=update_hyper_sigma, update_hyper_p=update_hyper_p, sparse_solution=sparse_solution, sparse_frac_betas=sparse_frac_betas, **kwargs)

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

        if np.sum(run_mask) == 0 and self.p_values is not None:
            run_mask[np.argmax(self.p_values)] = True

        if run_using_phewas:
            (beta_tildes_to_use, ses_to_use) = (self.beta_tildes_phewas[:,run_mask], self.ses_phewas[:,run_mask])
        else:
            (beta_tildes_to_use, ses_to_use) = (self.beta_tildes[run_mask], self.ses[run_mask])

        if not run_using_phewas or run_betas_using_phewas:
            result = self._calculate_non_inf_betas(p, beta_tildes=beta_tildes_to_use, ses=ses_to_use, X_orig=self.X_orig[:,run_mask], scale_factors=self.scale_factors[run_mask], mean_shifts=self.mean_shifts[run_mask], V=None, ps=self.ps[run_mask] if self.ps is not None else None, sigma2s=self.sigma2s[run_mask] if self.sigma2s is not None else None, is_dense_gene_set=self.is_dense_gene_set[run_mask], max_num_burn_in=max_num_burn_in, max_num_iter=max_num_iter, min_num_iter=min_num_iter, num_chains=num_chains, r_threshold_burn_in=r_threshold_burn_in, use_max_r_for_convergence=use_max_r_for_convergence, max_frac_sem=max_frac_sem, gauss_seidel=gauss_seidel, update_hyper_sigma=update_hyper_sigma, update_hyper_p=update_hyper_p, sparse_solution=sparse_solution, sparse_frac_betas=sparse_frac_betas, betas_trace_out=betas_trace_out, betas_trace_gene_sets=[self.gene_sets[i] for i in range(len(self.gene_sets)) if run_mask[i]], debug_gene_sets=[self.gene_sets[i] for i in range(len(self.gene_sets)) if run_mask[i]], **kwargs)

        if run_using_phewas:
            if run_betas_using_phewas:
                self.betas_phewas = copy.copy(result[0])
                if len(self.betas_phewas.shape) == 1:
                    self.betas_phewas = self.betas[np.newaxis,:]
            if run_uncorrected_using_phewas:
                self.betas_uncorrected_phewas = copy.copy(result_uncorrected[0])
                if len(self.betas_uncorrected_phewas.shape) == 1:
                    self.betas_uncorrected_phewas = self.betas_uncorrected_phewas[np.newaxis,:]

        else:
            (avg_betas_v[run_mask], avg_postp_v[run_mask]) = result

            if len(avg_betas_v.shape) == 2:
                avg_betas_v = np.mean(avg_betas_v, axis=0)
                avg_postp_v = np.mean(avg_postp_v, axis=0)

            self.betas = copy.copy(avg_betas_v)
            self.betas_uncorrected = copy.copy(avg_betas_uncorrected_v)

            self.non_inf_avg_postps = copy.copy(avg_postp_v)
            self.non_inf_avg_cond_betas = copy.copy(avg_betas_v)
            self.non_inf_avg_cond_betas[avg_postp_v > 0] /= avg_postp_v[avg_postp_v > 0]

            if self.gene_sets_missing is not None:
                self.betas_missing = np.zeros(len(self.gene_sets_missing))
                self.betas_uncorrected_missing = np.zeros(len(self.gene_sets_missing))
                self.non_inf_avg_postps_missing = np.zeros(len(self.gene_sets_missing))
                self.non_inf_avg_cond_betas_missing = np.zeros(len(self.gene_sets_missing))

    # ==========================================================================
    # Section: Core Inference Orchestration (priors + outer Gibbs).
    # ==========================================================================
    def calculate_priors(self, max_gene_set_p=None, num_gene_batches=None, correct_betas_mean=True, correct_betas_var=True, gene_loc_file=None, gene_cor_file=None, gene_cor_file_gene_col=1, gene_cor_file_cor_start_col=10, p_noninf=None, run_logistic=True, max_for_linear=0.95, adjust_priors=False, tag="", **kwargs):
        # ==========================================================================
        # Prior Phase 0: Validate prerequisites and choose batching strategy.
        # ==========================================================================
        if self.X_orig is None:
            bail("X is required for this operation")
        if self.betas is None:
            bail("betas are required for this operation")

        use_X = False

        assert(self.gene_sets is not None)
        max_num_gene_batches_together = 10000
        #if 0, don't use any V
        num_gene_batches_parallel = int(max_num_gene_batches_together / len(self.gene_sets))
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
            priors = np.array(self.X_orig.dot(self.betas / self.scale_factors) - np.sum(self.mean_shifts * self.betas / self.scale_factors)).flatten()
            self.combined_prior_Ys = None
            self.combined_prior_Ys_for_regression = None
            self.combined_prior_Ys_adj = None
            self.combined_prior_Y_ses = None
            self.combined_Ds = None
            self.batches = None
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
                if self.gene_to_chrom is None:
                    self.gene_to_chrom = {}
                if self.gene_to_pos is None:
                    self.gene_to_pos = {}

                with open_gz(gene_loc_file) as gene_loc_fh:
                    for line in gene_loc_fh:
                        cols = line.strip('\n').split()
                        if len(cols) != 6:
                            bail("Format for --gene-loc-file is:\n\tgene_id\tchrom\tstart\tstop\tstrand\tgene_name\nOffending line:\n\t%s" % line)
                        gene_name = cols[5]
                        if gene_name not in self.gene_to_ind:
                            continue

                        chrom = pegs_clean_chrom_name(cols[1])
                        pos1 = int(cols[2])
                        pos2 = int(cols[3])

                        self.gene_to_chrom[gene_name] = chrom
                        self.gene_to_pos[gene_name] = (pos1,pos2)

                        batches.add(chrom)
                        gene_chromosomes[gene_name] = chrom
                batches = sorted(batches)
                num_gene_batches = len(batches)
            else:
                #need sorted genes and correlation matrix to batch genes
                if self.y_corr is None:
                    correlation_m = self._read_correlations(gene_cor_file, gene_loc_file, gene_cor_file_gene_col=gene_cor_file_gene_col, gene_cor_file_cor_start_col=gene_cor_file_cor_start_col)
                    self._set_Y(self.Y, self.Y_for_regression, self.Y_exomes, self.Y_positive_controls, self.Y_case_counts, Y_corr_m=correlation_m, skip_V=True, skip_scale_factors=True, min_correlation=None)
                batches = range(num_gene_batches)

            gene_batch_size = int(len(self.genes) / float(num_gene_batches) + 1)
            self.batches = [None] * len(self.genes)
            priors = np.zeros(len(self.genes))

            #store a matrix of all beta_tildes across all batches
            full_matrix_shape = (len(batches), len(self.gene_sets) + (len(self.gene_sets_missing) if self.gene_sets_missing is not None else 0))
            full_beta_tildes_m = np.zeros(full_matrix_shape)
            full_ses_m = np.zeros(full_matrix_shape)
            full_z_scores_m = np.zeros(full_matrix_shape)
            full_se_inflation_factors_m = np.zeros(full_matrix_shape)
            full_p_values_m = np.zeros(full_matrix_shape)
            full_scale_factors_m = np.zeros(full_matrix_shape)
            full_ps_m = None
            if self.ps is not None:
                full_ps_m = np.zeros(full_matrix_shape)                
            full_sigma2s_m = None
            if self.sigma2s is not None:
                full_sigma2s_m = np.zeros(full_matrix_shape)                

            full_is_dense_gene_set_m = np.zeros(full_matrix_shape, dtype=bool)
            full_mean_shifts_m = np.zeros(full_matrix_shape)
            full_include_mask_m = np.zeros((len(batches), len(self.genes)), dtype=bool)
            full_priors_mask_m = np.zeros((len(batches), len(self.genes)), dtype=bool)

            # ==========================================================================
            # Prior Phase 2: Per-batch beta-tilde estimation on subsetted genes.
            # ==========================================================================
            # combine X_orig and X_orig_missing for batched prior calculations.
            with _temporary_unsubset_gene_sets(self, self.gene_sets_missing is not None, keep_missing=True, skip_V=True):

                for batch_ind in range(len(batches)):
                    batch = batches[batch_ind]
    
                    #specify:
                    # (a) include_mask: the genes that are used for calculating beta tildes and betas for this batch
                    # (b) priors_mask: the genes that we will calculate priors for
                    #these are not exact complements because we may need to exlude some genes for both (i.e. a buffer)
                    if loco:
                        include_mask = np.array([True] * len(self.genes))
                        priors_mask = np.array([False] * len(self.genes))
                        for i in range(len(self.genes)):
                            if self.genes[i] not in gene_chromosomes:
                                include_mask[i] = False
                                priors_mask[i] = True
                            elif gene_chromosomes[self.genes[i]] == batch:
                                include_mask[i] = False
                                priors_mask[i] = True
                            else:
                                include_mask[i] = True
                                priors_mask[i] = False
                        log("Batch %s: %d genes" % (batch, np.sum(priors_mask)))
                    else:
                        begin = batch * gene_batch_size
                        end = (batch + 1) * gene_batch_size
                        if end > len(self.genes):
                            end = len(self.genes)
                        end = end - 1
                        log("Batch %d: genes %d - %d" % (batch+1, begin, end))
    
    
                        #include only genes not correlated with any in the current batch
                        include_mask = np.array([True] * len(self.genes))
    
                        include_mask_begin = begin - 1
                        while include_mask_begin > 0 and (begin - include_mask_begin) < len(self.y_corr) and self.y_corr[begin - include_mask_begin][include_mask_begin] > 0:
                            include_mask_begin -= 1
                        include_mask_begin += 1
    
                        include_mask_end = end + 1
                        while (include_mask_end - end) < len(self.y_corr) and self.y_corr[include_mask_end - end][end] > 0:
                            include_mask_end += 1
                        include_mask[include_mask_begin:include_mask_end] = False
                        include_mask_end -= 1
    
                        priors_mask = np.array([False] * len(self.genes))
                        priors_mask[begin:(end+1)] = True
    
    
                    for i in range(len(self.genes)):
                        if priors_mask[i]:
                            self.batches[i] = batch
    
                    #now subset Y
                    Y = copy.copy(self.Y_for_regression)
                    y_corr = None
                    y_corr_sparse = None
    
                    if self.y_corr is not None:
                        y_corr = copy.copy(self.y_corr)
                        if not loco:
                            #we cannot rely on chromosome boundaries to zero out correlations, so manually do this
                            for i in range(include_mask_begin - 1, include_mask_begin - y_corr.shape[0], -1):
                                y_corr[include_mask_begin - i:,i] = 0
                        #don't need to zero out anything for include_mask_end because correlations between after end and removed are all stored inside of the removed indices
                        y_corr = y_corr[:,include_mask]
    
                        if self.y_corr_sparse is not None:
                            y_corr_sparse = self.y_corr_sparse[include_mask,:][:,include_mask]
                    
                    Y = Y[include_mask]
                    y_var = np.var(Y)
    
                    #DO WE NEED THIS??
                    #y_mean = np.mean(Y)
                    #Y = Y - y_mean
    
                    (mean_shifts, scale_factors) = self._calc_X_shift_scale(self.X_orig[include_mask,:])
    
                    #if some gene sets became empty!
                    assert(not np.any(np.logical_and(mean_shifts != 0, scale_factors == 0)))
                    mean_shifts[mean_shifts == 0] = 0
                    scale_factors[scale_factors == 0] = 1
    
                    ps = self.ps
                    sigma2s = self.sigma2s
                    is_dense_gene_set = self.is_dense_gene_set
    
                    #max_gene_set_p = self.max_gene_set_p if self.max_gene_set_p is not None else 1
    
                    Y_to_use = Y
                    D = np.exp(Y_to_use + self.background_log_bf) / (1 + np.exp(Y_to_use + self.background_log_bf))
                    if np.max(D) > max_for_linear:
                        run_logistic = True
    
                    #compute special beta tildes here
                    if run_logistic:
                        (beta_tildes, ses, z_scores, p_values, se_inflation_factors, alpha_tildes, diverged) = self._compute_logistic_beta_tildes(self.X_orig[include_mask,:], D, scale_factors, mean_shifts, resid_correlation_matrix=y_corr_sparse)
                    else:
                        (beta_tildes, ses, z_scores, p_values, se_inflation_factors) = self._compute_beta_tildes(self.X_orig[include_mask,:], Y, y_var, scale_factors, mean_shifts, resid_correlation_matrix=y_corr_sparse)
    
                    if correct_betas_mean or correct_betas_var:
                        (beta_tildes, ses, z_scores, p_values, se_inflation_factors) = self._correct_beta_tildes(beta_tildes, ses, se_inflation_factors, self.total_qc_metrics, self.total_qc_metrics_directions, correct_mean=correct_betas_mean, correct_var=correct_betas_var, fit=False)
    
                    #now determine those that have too many genes removed to be accurate
                    mean_reduction = float(num_gene_batches - 1) / float(num_gene_batches)
                    sd_reduction = np.sqrt(mean_reduction * (1 - mean_reduction))
                    reduction = mean_shifts / self.mean_shifts
                    ignore_mask = reduction < mean_reduction - 3 * sd_reduction
                    if sum(ignore_mask) > 0:
                        log("Ignoring %d gene sets because there are too many genes are missing from this batch" % sum(ignore_mask))
                        for ind in np.array(range(len(ignore_mask)))[ignore_mask]:
                            log("%s: %.4g remaining (vs. %.4g +/- %.4g expected)" % (self.gene_sets[ind], reduction[ind], mean_reduction, sd_reduction), TRACE)
                    #also zero out anything above the p-value threshold; this is a convenience for below
                    #note that p-values are still preserved though for below
                    ignore_mask = np.logical_or(ignore_mask, p_values > max_gene_set_p)
    
                    beta_tildes[ignore_mask] = 0
                    ses[ignore_mask] = max(self.ses) * 100
    
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
                    #max_gene_set_p_v[max_gene_set_p_v < (self.max_gene_set_p if self.max_gene_set_p is not None else 1)] = (self.max_gene_set_p if self.max_gene_set_p is not None else 1)
                    max_gene_set_p_v[max_gene_set_p_v < (max_gene_set_p if max_gene_set_p is not None else 1)] = (max_gene_set_p if max_gene_set_p is not None else 1)
    
                    #get the include mask; any batch has p <= threshold
                    new_gene_set_mask = np.max(full_p_values_m[begin:end,:].T <= max_gene_set_p_v, axis=1)
                    num_gene_set_mask = np.sum(new_gene_set_mask)
    
                    #we unsubset genes to aid in batching; this caused sigma and p to be affected
                    fraction_non_missing = np.mean(new_gene_set_mask)
                    missing_scale_factor = self._get_fraction_non_missing() / fraction_non_missing
                    if missing_scale_factor > 1 / self.p:
                        #threshold this here. otherwise set_p will cap p but set_sigma won't cap sigma
                        missing_scale_factor = 1 / self.p
                    
                    #orig_sigma2 = self.sigma2
                    #orig_p = self.p
                    #self.set_sigma(self.sigma2 * missing_scale_factor, self.sigma_power, sigma2_osc=self.sigma2_osc)
                    #self.set_p(self.p * missing_scale_factor)
    
                    #construct the V matrix
                    if not use_X:
                        V_m = np.zeros((end-begin, num_gene_set_mask, num_gene_set_mask))
                        for i,j in zip(range(begin, end),range(end-begin)):
                            include_mask = full_include_mask_m[i,:]
    
                            V_m[j,:,:] = self._calculate_V_internal(self.X_orig[include_mask,:][:,new_gene_set_mask], None, full_mean_shifts_m[i,new_gene_set_mask], full_scale_factors_m[i,new_gene_set_mask])
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
                    (betas, avg_postp) = self._calculate_non_inf_betas(None, beta_tildes=cur_beta_tildes, ses=cur_ses, V=V_m, X_orig=self.X_orig[include_mask,:][:,new_gene_set_mask], scale_factors=cur_scale_factors, mean_shifts=cur_mean_shifts, is_dense_gene_set=cur_is_dense_gene_set, ps=cur_ps, sigma2s=cur_sigma2s, update_hyper_sigma=False, update_hyper_p=False, num_missing_gene_sets=int((1 - fraction_non_missing) * len(self.gene_sets)), **kwargs)
                    if len(betas.shape) == 1:
                        betas = betas[np.newaxis,:]
    
    
                    for i,j in zip(range(begin, end),range(end-begin)):
    
                        priors[full_priors_mask_m[i,:]] = np.array(self.X_orig[full_priors_mask_m[i,:],:][:,new_gene_set_mask].dot(betas[j,:] / cur_scale_factors[j,:]))

                    #now restore the p and sigma
                    #self.set_sigma(orig_sigma2, self.sigma_power, sigma2_osc=self.sigma2_osc)
                    #self.set_p(orig_p)

        # ==========================================================================
        # Prior Phase 4: Merge missing-gene priors, center values, and finalize.
        # ==========================================================================
        #now for the genes that were not included in X
        if self.X_orig_missing_genes is not None:
            #these can use the original betas because they were never included
            self.priors_missing = np.array(self.X_orig_missing_genes.dot(self.betas / self.scale_factors) - np.sum(self.mean_shifts * self.betas / self.scale_factors))
        else:
            self.priors_missing = np.array([])

        #store in member variable
        total_mean = np.mean(np.concatenate((priors, self.priors_missing)))
        self.priors = priors - total_mean
        self.priors_missing -= total_mean

        self.calculate_priors_adj(overwrite_priors=adjust_priors)

    def calculate_priors_adj(self, overwrite_priors=False):
        if self.priors is None:
            return
        
        #do the regression
        gene_N = self.get_gene_N()
        gene_N_missing = self.get_gene_N(get_missing=True)
        all_gene_N = gene_N
        if self.genes_missing is not None:
            assert(gene_N_missing is not None)
            all_gene_N = np.concatenate((all_gene_N, gene_N_missing))

        if self.genes_missing is not None:
            total_priors = np.concatenate((self.priors, self.priors_missing))
        else:
            total_priors = self.priors

        priors_slope = np.cov(total_priors, all_gene_N)[0,1] / np.var(all_gene_N)
        priors_intercept = np.mean(total_priors - all_gene_N * priors_slope)

        log("Adjusting priors with slope %.4g" % priors_slope)
        priors_adj = self.priors - priors_slope * gene_N - priors_intercept
        if overwrite_priors:
            self.priors = priors_adj
        else:
            self.priors_adj = priors_adj
        if self.genes_missing is not None:
            priors_adj_missing = self.priors_missing - priors_slope * gene_N_missing
            if overwrite_priors:
                self.priors_missing = priors_adj_missing
            else:
                self.priors_adj_missing = priors_adj_missing

    def calculate_naive_priors(self, adjust_priors=False):
        if self.X_orig is None:
            bail("X is required for this operation")
        if self.betas is None:
            bail("betas are required for this operation")
        
        self.priors = self.X_orig.dot(self.betas / self.scale_factors)

        if self.X_orig_missing_genes is not None:
            self.priors_missing = self.X_orig_missing_genes.dot(self.betas / self.scale_factors)
        else:
            self.priors_missing = np.array([])

        total_mean = np.mean(np.concatenate((self.priors, self.priors_missing)))
        self.priors -= total_mean
        self.priors_missing -= total_mean

        self.calculate_priors_adj(overwrite_priors=adjust_priors)

        if self.Y is not None:
            if self.priors is not None:
                self.combined_prior_Ys = self.priors + self.Y
            if self.priors_adj is not None:
                self.combined_prior_Ys_adj = self.priors_adj + self.Y

    def run_gibbs(self, max_num_iter=100, total_num_iter=None, max_num_restarts=3, num_chains=10, num_mad=3, r_threshold_burn_in=1.10, use_max_r_for_convergence=True, increase_hyper_if_betas_below=None, experimental_hyper_mutation=False, update_huge_scores=True, top_gene_prior=None, min_num_burn_in=10, max_num_burn_in=None, min_num_post_burn_in=None, max_num_post_burn_in=None, max_num_iter_betas=1100, min_num_iter_betas=10, num_chains_betas=4, r_threshold_burn_in_betas=1.01, use_max_r_for_convergence_betas=True, max_frac_sem_betas=0.01, use_mean_betas=True, warm_start=False, burn_in_rhat_quantile=0.95, burn_in_patience=2, burn_in_stall_window=10, burn_in_stall_delta=0.01, stop_mcse_quantile=0.95, stop_patience=2, stop_top_gene_k=200, stop_min_gene_d=None, max_abs_mcse_d=0.05, max_rel_mcse_beta=0.20, active_beta_top_k=200, active_beta_min_abs=0.01, beta_rel_mcse_denom_floor=0.10, stall_window=8, stall_min_burn_in=50, stall_min_post_burn_in=50, stall_delta_rhat=0.01, stall_delta_mcse=0.01, stall_recent_window=4, stall_recent_eps=0.0, stopping_preset_name="lenient", diag_every=5, sparse_frac_gibbs=0.01, sparse_max_gibbs=0.001, sparse_solution=False, sparse_frac_betas=None, pre_filter_batch_size=None, pre_filter_small_batch_size=500, max_allowed_batch_correlation=None, gauss_seidel_betas=False, gauss_seidel=False, num_batches_parallel=10, max_mb_X_h=200, initial_linear_filter=True, correct_betas_mean=True, correct_betas_var=True, adjust_priors=True, gene_set_stats_trace_out=None, gene_stats_trace_out=None, betas_trace_out=None, debug_zero_sparse=False, eps=0.01):
        # ==========================================================================
        # Gibbs Phase 0: Normalize controls and initialize run-level state.
        # ==========================================================================
        gibbs_controls = _normalize_gibbs_run_controls(
            max_num_iter=max_num_iter,
            total_num_iter=total_num_iter,
            max_num_restarts=max_num_restarts,
            num_chains=num_chains,
            min_num_burn_in=min_num_burn_in,
            max_num_burn_in=max_num_burn_in,
            min_num_post_burn_in=min_num_post_burn_in,
            max_num_post_burn_in=max_num_post_burn_in,
            diag_every=diag_every,
            burn_in_patience=burn_in_patience,
            burn_in_stall_window=burn_in_stall_window,
            burn_in_stall_delta=burn_in_stall_delta,
            stop_patience=stop_patience,
            stop_top_gene_k=stop_top_gene_k,
            stop_min_gene_d=stop_min_gene_d,
            active_beta_top_k=active_beta_top_k,
            active_beta_min_abs=active_beta_min_abs,
            beta_rel_mcse_denom_floor=beta_rel_mcse_denom_floor,
            stall_window=stall_window,
            stall_min_burn_in=stall_min_burn_in,
            stall_min_post_burn_in=stall_min_post_burn_in,
            stall_delta_rhat=stall_delta_rhat,
            stall_delta_mcse=stall_delta_mcse,
            stall_recent_window=stall_recent_window,
            stall_recent_eps=stall_recent_eps,
            burn_in_rhat_quantile=burn_in_rhat_quantile,
            use_max_r_for_convergence=use_max_r_for_convergence,
        )
        run_state = gibbs_controls.run_state
        num_chains = gibbs_controls.num_chains

        # ==========================================================================
        # Gibbs Phase 1: Record configuration and reset diagnostics.
        # ==========================================================================
        gibbs_record_config = _build_gibbs_record_config(
            gibbs_controls=gibbs_controls,
            num_chains_betas=num_chains_betas,
            max_num_iter=max_num_iter,
            use_mean_betas=use_mean_betas,
            warm_start=warm_start,
            stopping_preset_name=stopping_preset_name,
            r_threshold_burn_in=r_threshold_burn_in,
            stop_mcse_quantile=stop_mcse_quantile,
            max_abs_mcse_d=max_abs_mcse_d,
            max_rel_mcse_beta=max_rel_mcse_beta,
            sparse_solution=sparse_solution,
            sparse_frac_gibbs=sparse_frac_gibbs,
            sparse_max_gibbs=sparse_max_gibbs,
            sparse_frac_betas=sparse_frac_betas,
            pre_filter_batch_size=pre_filter_batch_size,
            max_allowed_batch_correlation=max_allowed_batch_correlation,
            initial_linear_filter=initial_linear_filter,
            correct_betas_mean=correct_betas_mean,
            correct_betas_var=correct_betas_var,
            adjust_priors=adjust_priors,
            experimental_hyper_mutation=experimental_hyper_mutation,
            increase_hyper_if_betas_below=increase_hyper_if_betas_below,
        )
        _record_gibbs_configuration_params(self, run_state, gibbs_record_config)
        _log_gibbs_configuration_summary(gibbs_record_config, run_state)

        _reset_gibbs_diagnostics(self)

        # ==========================================================================
        # Gibbs Phase 2: Build static inputs and epoch runtime configs.
        # ==========================================================================
        gibbs_inputs = _prepare_gibbs_run_inputs(
            state=self,
            num_chains=num_chains,
            top_gene_prior=top_gene_prior,
        )

        epoch_aggregates = _new_gibbs_epoch_aggregates()
        epoch_runtime_configs = _build_gibbs_epoch_runtime_configs(
            _build_gibbs_epoch_runtime_config_inputs(
                gibbs_controls,
                _build_gibbs_dynamic_runtime_inputs(
                    gibbs_inputs=gibbs_inputs,
                    use_mean_betas=use_mean_betas,
                    max_mb_X_h=max_mb_X_h,
                    num_mad=num_mad,
                    adjust_priors=adjust_priors,
                    increase_hyper_if_betas_below=increase_hyper_if_betas_below,
                    experimental_hyper_mutation=experimental_hyper_mutation,
                    max_num_iter_betas=max_num_iter_betas,
                    min_num_iter_betas=min_num_iter_betas,
                    num_chains_betas=num_chains_betas,
                    r_threshold_burn_in_betas=r_threshold_burn_in_betas,
                    use_max_r_for_convergence_betas=use_max_r_for_convergence_betas,
                    max_frac_sem_betas=max_frac_sem_betas,
                    max_allowed_batch_correlation=max_allowed_batch_correlation,
                    gauss_seidel_betas=gauss_seidel_betas,
                    sparse_solution=sparse_solution,
                    sparse_frac_betas=sparse_frac_betas,
                    warm_start=warm_start,
                    debug_zero_sparse=debug_zero_sparse,
                    num_batches_parallel=num_batches_parallel,
                    betas_trace_out=betas_trace_out,
                    update_huge_scores=update_huge_scores,
                    sparse_frac_gibbs=sparse_frac_gibbs,
                    sparse_max_gibbs=sparse_max_gibbs,
                    pre_filter_batch_size=pre_filter_batch_size,
                    pre_filter_small_batch_size=pre_filter_small_batch_size,
                    r_threshold_burn_in=r_threshold_burn_in,
                    gauss_seidel=gauss_seidel,
                    eps=eps,
                    stop_mcse_quantile=stop_mcse_quantile,
                    max_rel_mcse_beta=max_rel_mcse_beta,
                    max_abs_mcse_d=max_abs_mcse_d,
                    initial_linear_filter=initial_linear_filter,
                    correct_betas_mean=correct_betas_mean,
                    correct_betas_var=correct_betas_var,
                ),
            )
        )
        epoch_phase_config = epoch_runtime_configs.epoch_phase_config
        epoch_iteration_static_config = epoch_runtime_configs.epoch_iteration_static_config

        # ==========================================================================
        # Gibbs Phase 3: Execute epoch attempts (with optional trace writers).
        # ==========================================================================
        _run_gibbs_epochs_with_optional_traces(
            state=self,
            run_state=run_state,
            epoch_aggregates=epoch_aggregates,
            epoch_phase_config=epoch_phase_config,
            epoch_iteration_static_config=epoch_iteration_static_config,
            gene_set_stats_trace_out=gene_set_stats_trace_out,
            gene_stats_trace_out=gene_stats_trace_out,
            gibbs_inputs=gibbs_inputs,
        )

        # ==========================================================================
        # Gibbs Phase 4: Finalize run-level completion checks.
        # ==========================================================================
        if run_state.num_completed_epochs == 0:
            bail("Gibbs failed to complete any successful epochs within restart/iteration limits")
        log(
            "Aggregated %d Gibbs epoch(s) into %d effective chains"
            % (run_state.num_completed_epochs, run_state.num_completed_epochs * num_chains),
            INFO,
        )

    def _sparse_correlation_with_dot_product_threshold(self, X_sparse, beta, dot_product_threshold=0.01, Y=None):
        """
        Compute the sparse correlation matrix of (X * beta + Y) with dot-product thresholding,
        mean adjustment, and normalization, for k beta vectors in parallel.

        Parameters:
        - X_sparse (scipy.sparse.csc_matrix): Sparse matrix X of shape (n, m).
        - beta (np.array): Dense array of shape (k, m) for k beta vectors.
        - dot_product_threshold (float): Threshold for absolute dot product values.
        - Y (np.array, optional): Dense array of shape (k, n) or None. Defaults to None.

        Returns:
        - scipy.sparse.csc_matrix: Sparse block diagonal correlation matrix for k beta vectors.
        """

        # Handle Y as an optional argument
        if Y is not None:
            if beta.shape[0] != Y.shape[0] or X_sparse.shape[0] != Y.shape[1]:
                raise DataValidationError("Y must have shape (k, n) where k matches beta's rows and n matches X's rows.")
            Y = np.square(Y.flatten())

        # Ensure beta is 2D
        if beta.ndim == 1:
            beta = beta[np.newaxis, :]  # Convert to shape (1, m) if beta is a single vector

        k, m = beta.shape  # Number of beta vectors and features
        n = X_sparse.shape[0]  # Number of rows (samples) in X

        # Step 1: Scale X_sparse by each beta vector and construct block diagonal matrix
        scaled_blocks = [X_sparse.multiply(beta[i, :]) for i in range(k)]
        X_scaled = sparse.block_diag(scaled_blocks, format='csc')  # Shape: (k * n, k * m)

        var_threshold = 0.05
        prior_threshold = 0.1
        X_scaled_sum = X_scaled.sum(axis=1).A1
        keep_mask = np.logical_and((np.square(X_scaled_sum) / ((Y if Y is not None else 0) + np.square(X_scaled_sum) + 1e-20) > var_threshold), (X_scaled_sum > prior_threshold))

        X_scaled = (X_scaled.T.multiply(keep_mask)).T
        X_scaled.eliminate_zeros()

        # Step 2: Compute uncentered second moment for all scaled X_sparse blocks

        X_scaled_dot_X_scaled = X_scaled.dot(X_scaled.T).multiply(1.0 / m).tocsr()  # n x n

        # Retain only the rows, columns, and values that pass the threshold
        threshold_mask = np.abs(X_scaled_dot_X_scaled.data) < (dot_product_threshold / m)
        X_scaled_dot_X_scaled.data[threshold_mask] = 0
        X_scaled_dot_X_scaled.eliminate_zeros()

        #We now have E[XBi*XBj]

        #calculate E[Xbi] and E2[Xbi]

        E_X_scaled = X_scaled.mean(axis=1).A1
        E2_X_scaled = X_scaled_dot_X_scaled.diagonal()

        # Identify block and local indices
        if type(X_scaled_dot_X_scaled) is not sparse.csr_matrix:
            X_scaled_dot_X_scaled = X_scaled_dot_X_scaled.tocsr()

        #get indices of columns
        rows = np.repeat(np.arange(len(X_scaled_dot_X_scaled.indptr) - 1), np.diff(X_scaled_dot_X_scaled.indptr))
        cols = X_scaled_dot_X_scaled.indices  # Directly use indices for rows

        #subtract E[betai]E[betaj]
        X_scaled_dot_X_scaled.data -= E_X_scaled[rows] * E_X_scaled[cols]
        if Y is not None:
            X_scaled_dot_X_scaled.data += Y[rows] * Y[cols]
        #divide by the variances
        X_scaled_dot_X_scaled.data /= (np.sqrt((E2_X_scaled[rows] - np.square(E_X_scaled)[rows] + np.square(Y[rows] if Y is not None else 0)) * (E2_X_scaled[cols] - np.square(E_X_scaled)[cols] + np.square(Y[cols] if Y is not None else 0))) + 1e-20)

        cor_threshold = 0.01
        X_scaled_dot_X_scaled.data[X_scaled_dot_X_scaled.data <= cor_threshold] = 0
        X_scaled_dot_X_scaled.eliminate_zeros()

        # Step 5: Construct sparse block diagonal correlation matrix
        X_scaled_dot_X_scaled = X_scaled_dot_X_scaled + sparse.diags(np.ones(k * n), format="csr")
        X_scaled_dot_X_scaled = X_scaled_dot_X_scaled.multiply(sparse.diags(1.0 / X_scaled_dot_X_scaled.diagonal(), format="csr"))
        sparse_corr_matrix = X_scaled_dot_X_scaled

        # Step 6: Return sparse correlation matrix or list of matrices
        if k == 1:
            return sparse_corr_matrix
        else:
            return [sparse_corr_matrix[i * n:(i + 1) * n, i * n:(i + 1) * n] for i in range(k)]

    def read_gene_phewas(self):
        return self.gene_pheno_Y is not None or  self.gene_pheno_combined_prior_Ys is not None and self.gene_pheno_priors is not None

    def _build_phewas_input_values(self):
        # Build the fixed 3-column input block (Y/combined/prior), then convert
        # from log-odds-style values to probabilities used for PheWAS regression.
        default_value = (
            self.Y[:, np.newaxis]
            if self.Y is not None
            else self.combined_prior_Ys[:, np.newaxis]
            if self.combined_prior_Ys is not None
            else self.priors[:, np.newaxis]
        )
        input_values = np.hstack(
            (
                self.Y[:, np.newaxis] if self.Y is not None else default_value,
                self.combined_prior_Ys[:, np.newaxis] if self.combined_prior_Ys is not None else default_value,
                self.priors[:, np.newaxis] if self.priors is not None else default_value,
            )
        )
        return np.exp(input_values + self.background_bf) / (1 + np.exp(input_values + self.background_bf))

    def _calculate_phewas_block(
        self,
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
        (mean_shifts, scale_factors) = self._calc_X_shift_scale(X_mat)

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
            end = (batch + 1) * cor_batch_size
            if end > beta_tildes.shape[0]:
                end = beta_tildes.shape[0]

            if X_phewas_beta is not None and X_orig is not None and not self.debug_skip_correlation:
                if X_phewas_beta.shape[0] != Y_mat.shape[0]:
                    bail(
                        "When calling this, the phewas_betas must have same number of phenos as Y_mat: shapes are X_phewas=(%d,%d) vs. Y_mat=(%d,%d)"
                        % (X_phewas_beta.shape[0], X_phewas_beta.shape[1], Y_mat.shape[0], Y_mat.shape[1])
                    )
                dot_threshold = 0.01 * 0.01
                log("Calculating correlation matrix for use in residuals", DEBUG)
                cor_matrices = self._sparse_correlation_with_dot_product_threshold(
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
                log("Sparsity of correlation matrix is %d/%d=%.3g (size %.3gMb)" % (nnz, total, float(nnz) / total, nnz * 8 / (1024 * 1024)), DEBUG)

            if multivariate:
                if huber:
                    (beta_tildes[begin:end, :], ses[begin:end, :], z_scores[begin:end, :], p_values[begin:end, :], se_inflation_factors[begin:end, :]) = self._compute_robust_betas(
                        X_mat,
                        Y_mat[begin:end, :],
                        resid_correlation_matrix=cor_matrices,
                        covs=covs if not self.debug_skip_phewas_covs else None,
                    )
                else:
                    (beta_tildes[begin:end, :], ses[begin:end, :], z_scores[begin:end, :], p_values[begin:end, :], se_inflation_factors[begin:end, :]) = self._compute_multivariate_beta_tildes(
                        X_mat,
                        Y_mat[begin:end, :],
                        resid_correlation_matrix=cor_matrices,
                        covs=covs if not self.debug_skip_phewas_covs else None,
                    )
            else:
                (beta_tildes[begin:end, :], ses[begin:end, :], z_scores[begin:end, :], p_values[begin:end, :], se_inflation_factors[begin:end, :]) = self._compute_beta_tildes(
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

        # Run non-inf regression with temporary hyperparameter overrides so this
        # branch never leaks p/sigma state back into the broader run.
        with _temporary_state_fields(
            self,
            overrides={"ps": None, "sigma2s": None},
            restore_fields=_STATE_FIELDS_SAMPLER_HYPER,
        ) as hyper_snapshot:
            orig_p = hyper_snapshot["p"]
            orig_sigma2_internal = hyper_snapshot["sigma2"]
            orig_sigma_power = hyper_snapshot["sigma_power"]

            new_p = 0.5
            new_sigma2_internal = orig_sigma2_internal * (new_p / orig_p)
            self.set_p(new_p)
            self.set_sigma(new_sigma2_internal, orig_sigma_power, convert_sigma_to_internal_units=False)

            (betas_uncorrected, postp_uncorrected) = self._calculate_non_inf_betas(
                initial_p=self.p,
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

    def _append_phewas_metric_block(self, current_beta, current_beta_tilde, current_se, current_z, current_p_value, current_one_sided_p_value, beta, beta_tilde, se, z_score, p_value, one_sided_p_value):
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

    def _prepare_phewas_phenos_from_file(self, gene_phewas_bfs_in, gene_phewas_bfs_id_col=None, gene_phewas_bfs_pheno_col=None, gene_phewas_bfs_log_bf_col=None, gene_phewas_bfs_combined_col=None, gene_phewas_bfs_prior_col=None):
        phenos, pheno_to_ind, col_info = pegs_prepare_phewas_phenos_from_file(
            self,
            gene_phewas_bfs_in,
            gene_phewas_bfs_id_col=gene_phewas_bfs_id_col,
            gene_phewas_bfs_pheno_col=gene_phewas_bfs_pheno_col,
            gene_phewas_bfs_log_bf_col=gene_phewas_bfs_log_bf_col,
            gene_phewas_bfs_combined_col=gene_phewas_bfs_combined_col,
            gene_phewas_bfs_prior_col=gene_phewas_bfs_prior_col,
            open_text_fn=open_gz,
            get_col_fn=_get_col,
            construct_map_to_ind_fn=pegs_construct_map_to_ind,
            warn_fn=warn,
            log_fn=log,
            debug_level=DEBUG,
        )
        return phenos, pheno_to_ind, {
            "id_col": col_info.id_col,
            "pheno_col": col_info.pheno_col,
            "bf_col": col_info.bf_col,
            "combined_col": col_info.combined_col,
            "prior_col": col_info.prior_col,
        }

    def _read_phewas_file_batch(self, gene_phewas_bfs_in, begin, cur_batch_size, pheno_to_ind, id_col, pheno_col, bf_col, combined_col, prior_col):
        col_info = {
            "id_col": id_col,
            "pheno_col": pheno_col,
            "bf_col": bf_col,
            "combined_col": combined_col,
            "prior_col": prior_col,
        }
        return pegs_read_phewas_file_batch(
            self,
            gene_phewas_bfs_in,
            begin=begin,
            cur_batch_size=cur_batch_size,
            pheno_to_ind=pheno_to_ind,
            col_info=col_info,
            open_text_fn=open_gz,
            warn_fn=warn,
        )

    def _accumulate_phewas_outputs(self, output_prefix, beta, beta_tilde, se, z_score, p_value):
        pegs_accumulate_standard_phewas_outputs(
            self,
            output_prefix,
            beta,
            beta_tilde,
            se,
            z_score,
            p_value,
        )

    def run_phewas(self, gene_phewas_bfs_in=None, gene_phewas_bfs_id_col=None, gene_phewas_bfs_pheno_col=None, gene_phewas_bfs_log_bf_col=None, gene_phewas_bfs_combined_col=None, gene_phewas_bfs_prior_col=None, max_num_burn_in=1000, max_num_iter=1100, min_num_iter=10, num_chains=10, r_threshold_burn_in=1.01, use_max_r_for_convergence=True, max_frac_sem=0.01, gauss_seidel=False, sparse_solution=False, sparse_frac_betas=None, batch_size=1500, **kwargs):

        #require X matrix
        if gene_phewas_bfs_in is None and not self.read_gene_phewas():
            bail("Require --gene-stats-in or --gene-phewas-bfs-in with a column for log_bf/Y in this operation")

        if self.genes is None:
            warn("Cannot run phewas without X matrix; skipping")
            return
        if self.Y is None and self.combined_prior_Ys is None and self.priors is None:
            warn("Cannot run phewas without Y values; skipping")
            return

        log("Running phewas", INFO)

        #first get the list of phenotypes
        read_file = gene_phewas_bfs_in is not None

        col_info = None

        if read_file:
            phenos, pheno_to_ind, col_info = self._prepare_phewas_phenos_from_file(
                gene_phewas_bfs_in=gene_phewas_bfs_in,
                gene_phewas_bfs_id_col=gene_phewas_bfs_id_col,
                gene_phewas_bfs_pheno_col=gene_phewas_bfs_pheno_col,
                gene_phewas_bfs_log_bf_col=gene_phewas_bfs_log_bf_col,
                gene_phewas_bfs_combined_col=gene_phewas_bfs_combined_col,
                gene_phewas_bfs_prior_col=gene_phewas_bfs_prior_col,
            )
        else:
            phenos = self.phenos

        #do phewas in batches to save memory
        num_batches = int(np.ceil(len(phenos) / batch_size))
        input_values = self._build_phewas_input_values()
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
                gene_pheno_Y, gene_pheno_combined_prior_Ys, gene_pheno_priors = self._read_phewas_file_batch(
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
                gene_pheno_Y = self.gene_pheno_Y[:,begin:end].toarray() if self.gene_pheno_Y is not None else None
                gene_pheno_combined_prior_Ys = self.gene_pheno_combined_prior_Ys[:,begin:end].toarray() if self.gene_pheno_combined_prior_Ys is not None else None


            if gene_pheno_Y is not None:
                beta, _, beta_tilde, se, Z, p_value, _ = self._calculate_phewas_block(input_values, gene_pheno_Y.T, **phewas_beta_kwargs)
                assert beta.shape[0] == 3, "First dimension of beta should be 3, not (%s, %s)" % (beta.shape[0], beta.shape[1])
                self._accumulate_phewas_outputs("pheno_Y", beta, beta_tilde, se, Z, p_value)

            if gene_pheno_combined_prior_Ys is not None and not self.debug_skip_correlation:
                #we have to use the correlations here
                beta, _, beta_tilde, se, Z, p_value, _ = self._calculate_phewas_block(
                    input_values,
                    gene_pheno_combined_prior_Ys.T,
                    X_orig=self.X_orig,
                    X_phewas_beta=self.X_phewas_beta[begin:end,:] if self.X_phewas_beta is not None else None,
                    Y_resid=gene_pheno_Y.T,
                    **phewas_beta_kwargs
                )
                assert beta.shape[0] == 3, "First dimension of beta should be 3, not (%s, %s)" % (beta.shape[0], beta.shape[1])
                self._accumulate_phewas_outputs("pheno_combined_prior_Ys", beta, beta_tilde, se, Z, p_value)

    def run_sim(self, sigma2, p, sigma_power, log_bf_noise_sigma_mult=0, treat_sigma2_as_sigma2_cond=True, only_positive=False):

        if sigma2 is None or sigma2 <= 0:
            bail("Require positive --sigma2 for simulations")
        if p is None:
            bail("Require --p-noninf for simulations")
        if sigma_power is None:
            bail("Require --sigma-power for simulations")
        if self.X_orig is None:
            bail("Require --X-in for simulations")
        
        log("Simulating gene set and gene values")
        #first simulate the sigmas
        self.betas = np.zeros(len(self.gene_sets))
        non_zero_gene_sets = np.random.random(self.betas.shape) < p

        scaled_sigma2s = self.get_scaled_sigma2(self.scale_factors, sigma2, sigma_power)

        #since we are only simulating for those that have non-zeros, we need to use conditional sigma2
        if treat_sigma2_as_sigma2_cond:
            sigma2_conds = scaled_sigma2s[non_zero_gene_sets]
            log("Using p=%.3g, sigma2_cond=%.3g" % (p, sigma2))
        else:
            sigma2_conds = scaled_sigma2s[non_zero_gene_sets] / p
            log("Using p=%.3g, sigma2_cond=%.3g" % (p, sigma2/p))

        self.betas[non_zero_gene_sets] = scipy.stats.norm.rvs(0, np.sqrt(sigma2_conds), np.sum(non_zero_gene_sets)).ravel()

        if only_positive:
            self.betas = np.abs(self.betas)

        #now simulate the gene values
        self.priors = self.X_orig.dot(self.betas / self.scale_factors)

        if log_bf_noise_sigma_mult > 0:
            #here we don't divide by p since we are adding noise to every beta, not just non zero ones
            noise_add_betas = scipy.stats.norm.rvs(0, np.sqrt(scaled_sigma2s * log_bf_noise_sigma_mult), self.betas.shape)
            self.Y = self.priors + self.X_orig.dot(noise_add_betas / self.scale_factors)
        else:
            self.Y = self.priors

        self._set_Y(self.Y, self.Y, self.Y_exomes, self.Y_positive_controls, self.Y_case_counts)


    def get_col_sums(self, X, num_nonzero=False, axis=0):
        if num_nonzero:
            return X.astype(bool).sum(axis=axis).A1
        else:
            return np.abs(X).sum(axis=axis).A1

    def get_gene_N(self, get_missing=False):
        if get_missing:
            if self.gene_N_missing is None:
                return None
            else:
                return self.gene_N_missing + (self.gene_ignored_N_missing if self.gene_ignored_N_missing is not None else 0)
        else:
            if self.gene_N is None:
                return None
            else:
                return self.gene_N + (self.gene_ignored_N if self.gene_ignored_N is not None else 0)

    def write_gene_set_statistics(self, output_file, max_no_write_gene_set_beta=None, max_no_write_gene_set_beta_uncorrected=None, basic=False):
        return pegs_write_gene_set_statistics(
            self,
            output_file,
            max_no_write_gene_set_beta=max_no_write_gene_set_beta,
            max_no_write_gene_set_beta_uncorrected=max_no_write_gene_set_beta_uncorrected,
            basic=basic,
            open_text_fn=open_gz,
            log_fn=log,
            info_level=INFO,
            debug_only_avg_huge=self.debug_only_avg_huge,
        )

    def write_phewas_gene_set_statistics(self, output_file, max_no_write_gene_set_beta=None, max_no_write_gene_set_beta_uncorrected=None, basic=False):
        return pegs_write_phewas_gene_set_statistics(
            self,
            output_file,
            max_no_write_gene_set_beta=max_no_write_gene_set_beta,
            max_no_write_gene_set_beta_uncorrected=max_no_write_gene_set_beta_uncorrected,
            basic=basic,
            open_text_fn=open_gz,
            log_fn=log,
            info_level=INFO,
        )

    def write_gene_statistics(self, output_file):
        return pegs_write_gene_statistics(
            self,
            output_file,
            open_text_fn=open_gz,
            log_fn=log,
            info_level=INFO,
        )

    def write_gene_gene_set_statistics(self, output_file, max_no_write_gene_gene_set_beta=0.0001, write_filter_beta_uncorrected=False):
        return pegs_write_gene_gene_set_statistics(
            self,
            output_file,
            max_no_write_gene_gene_set_beta=max_no_write_gene_gene_set_beta,
            write_filter_beta_uncorrected=write_filter_beta_uncorrected,
            open_text_fn=open_gz,
            log_fn=log,
            info_level=INFO,
        )

    def write_gene_set_overlap_statistics(self, output_file):
        log("Writing gene set overlap stats to %s" % output_file, INFO)
        with open_gz(output_file, 'w') as output_fh:
            if self.gene_sets is None:
                return
            if self.X_orig is None or self.betas is None or self.betas_uncorrected is None or self.mean_shifts is None or self.scale_factors is None:
                return
            header = "Gene_Set\tbeta\tbeta_uncorrected\tGene_Set_overlap\tV_beta\tV\tbeta_overlap\tbeta_uncorrected_overlap"
            output_fh.write("%s\n" % header)

            print_mask = self.betas_uncorrected != 0
            gene_sets = [self.gene_sets[i] for i in np.where(print_mask)[0]]
            X_to_print = self.X_orig[:,print_mask]
            mean_shifts = self.mean_shifts[print_mask]
            scale_factors = self.scale_factors[print_mask]
            betas_uncorrected = self.betas_uncorrected[print_mask]
            betas = self.betas[print_mask]

            num_batches = self._get_num_X_blocks(X_to_print)

            ordered_i = sorted(range(len(gene_sets)), key=lambda k: -betas[k] / scale_factors[k])

            gene_sets = [gene_sets[i] for i in ordered_i]
            X_to_print = X_to_print[:,ordered_i]
            mean_shifts = mean_shifts[ordered_i]
            scale_factors = scale_factors[ordered_i]
            betas_uncorrected = betas_uncorrected[ordered_i]
            betas = betas[ordered_i]

            for batch in range(num_batches):
                begin = batch * self.batch_size
                end = (batch + 1) * self.batch_size
                if end > X_to_print.shape[1]:
                    end = X_to_print.shape[1]

                X_to_print[:,begin:end]
                mean_shifts[begin:end]
                scale_factors[begin:end]

                cur_V = self._compute_V(X_to_print[:,begin:end], mean_shifts[begin:end], scale_factors[begin:end], X_orig2=X_to_print, mean_shifts2=mean_shifts, scale_factors2=scale_factors)
                cur_V_beta = cur_V * betas

                for i in range(end - begin):
                    outer_ind = int(i + batch * self.batch_size)
                    ordered_j = sorted(np.where(cur_V_beta[i,:] != 0)[0], key=lambda k: -cur_V_beta[i,k] / scale_factors[k])
                    for j in ordered_j:
                        if outer_ind == j:
                            continue
                        output_fh.write("%s\t%.3g\t%.3g\t%s\t%.3g\t%.3g\t%.3g\t%.3g\n" % (gene_sets[outer_ind], betas[outer_ind] / scale_factors[outer_ind], betas_uncorrected[outer_ind] / scale_factors[outer_ind], gene_sets[j], cur_V_beta[i, j] / scale_factors[i], cur_V[i,j], betas[j] / scale_factors[j], betas_uncorrected[j] / scale_factors[j]))


    def write_gene_covariates(self, output_file):
        if self.genes is None or self.gene_covariates is None:
            return

        assert(self.gene_covariates.shape[1] == len(self.gene_covariate_names))
        log("Writing covs to %s" % output_file, INFO)

        with open_gz(output_file, 'w') as output_fh:

            #gene_covariate_betas = 
            #if self.gene_covariate_betas is not None:
            #    value_out = "#betas\tbetas"
            #    for j in range(self.gene_covariates.shape[1]):
            #        value_out += ("\t%.4g" % self.gene_covariate_betas[j])
            #    output_fh.write("%s\n" % value_out)

            header = "%s\t%s" % ("Gene\tin_regression", "\t".join(self.gene_covariate_names))
            output_fh.write("%s\n" % header)

            for i in range(len(self.genes)):
                value_out = "%s\t%s" % (self.genes[i], self.gene_covariates_mask[i])
                for j in range(self.gene_covariates.shape[1]):
                    value_out += ("\t%.4g" % self.gene_covariates[i,j])
                output_fh.write("%s\n" % value_out)


    def write_gene_effectors(self, output_file):
        if self.genes is None or self.huge_signal_bfs is None:
            return

        assert(self.huge_signal_bfs.shape[1] == len(self.huge_signals))
        
        log("Writing gene effectors to %s" % output_file, INFO)

        if self.gene_to_gwas_huge_score is not None and self.gene_to_exomes_huge_score is not None:
            gene_to_huge_score = self.gene_to_gwas_huge_score
        else:
            gene_to_huge_score = self.gene_to_huge_score
            if gene_to_huge_score is None:
                gene_to_huge_score = self.gene_to_gwas_huge_score
            if gene_to_huge_score is None:
                gene_to_huge_score = self.gene_to_exomes_huge_score

        with open_gz(output_file, 'w') as output_fh:

            header = "Lead_locus\tInput\tP\tGene"

            if self.combined_prior_Ys is not None:
                header = "%s\t%s" % (header, "cond_prob_total") #probability of each gene under assumption that only one is causal
            if self.Y is not None and self.priors is not None:
                header = "%s\t%s" % (header, "cond_prob_signal") #probability of each gene under assumption that only one is an effector (more than one could be causal if there are multiple SNPs each with different effectors)
            if self.priors is not None:
                header = "%s\t%s" % (header, "cond_prob_prior") #probability of each gene using only priors (assumption one causal gene)
            if gene_to_huge_score is not None:
                header = "%s\t%s" % (header, "cond_prob_huge") #probability of each gene using only distance/s2g (assumption one causal effector)
            if self.combined_Ds is not None:
                header = "%s\t%s" % (header, "combined_D")

            output_fh.write("%s\n" % header)

            for signal_ind in range(len(self.huge_signals)):

                gene_inds = self.huge_signal_bfs[:, signal_ind].nonzero()[0]

                max_log_bf = 10

                cond_prob_total = None
                if self.combined_prior_Ys is not None:
                    combined_prior_Y_bfs = self.combined_prior_Ys[gene_inds]
                    combined_prior_Y_bfs[combined_prior_Y_bfs > max_log_bf] = max_log_bf                    
                    combined_prior_Y_bfs = np.exp(combined_prior_Y_bfs)
                    cond_prob_total = combined_prior_Y_bfs / np.sum(combined_prior_Y_bfs)

                cond_prob_prior = None
                if self.priors is not None:
                    prior_bfs = self.priors[gene_inds]
                    prior_bfs[prior_bfs > max_log_bf] = max_log_bf                    
                    prior_bfs = np.exp(prior_bfs)
                    cond_prob_prior = prior_bfs / np.sum(prior_bfs)
                    
                    if self.Y is not None:
                        log_bf_bfs = self.huge_signal_bfs[:,signal_ind].todense().A1[gene_inds] * prior_bfs
                        cond_prob_log_bf = log_bf_bfs / np.sum(log_bf_bfs)

                cond_prob_huge = None
                if gene_to_huge_score is not None:
                    cond_prob_huge = self.huge_signal_bfs[:,signal_ind].todense().A1[gene_inds]
                    cond_prob_huge /= np.sum(cond_prob_huge)

                for i in range(len(gene_inds)):
                    gene_ind = gene_inds[i]
                    line = "%s:%d\t%s\t%.3g\t%s" % (self.huge_signals[signal_ind][0], self.huge_signals[signal_ind][1], self.huge_signals[signal_ind][3], self.huge_signals[signal_ind][2], self.genes[gene_ind])

                    if self.combined_prior_Ys is not None:
                        line = "%s\t%.3g" % (line, cond_prob_total[i])
                    if self.Y is not None and self.priors is not None:
                        line = "%s\t%.3g" % (line, cond_prob_log_bf[i])
                    if self.priors is not None:
                        line = "%s\t%.3g" % (line, cond_prob_prior[i])
                    if gene_to_huge_score is not None:
                        line = "%s\t%.3g" % (line, cond_prob_huge[i])
                    if self.combined_Ds is not None:
                        line = "%s\t%.3g" % (line, self.combined_Ds[gene_ind])

                    output_fh.write("%s\n" % line)


    def write_phewas_statistics(self, output_file):
        return pegs_write_phewas_statistics(
            self,
            output_file,
            open_text_fn=open_gz,
            log_fn=log,
            info_level=INFO,
        )

    def _record_param(self, param, value, overwrite=False, record_only_first_time=False):
        if param not in self.params:
            self.param_keys.append(param)
            self.params[param] = value
        elif record_only_first_time:
            return
        elif type(self.params[param]) == list:
            if self.params[param][-1] != value:
                self.params[param].append(value)
        elif self.params[param] != value:
            if overwrite:
                self.params[param] = value
            else:
                self.params[param] = [self.params[param], value]

    def _record_params(self, params, overwrite=False, record_only_first_time=False):
        for param in params:
            if params[param] is not None:
                self._record_param(param, params[param], overwrite=overwrite, record_only_first_time=record_only_first_time)

    def read_gene_bfs(self, gene_bfs_in, gene_bfs_id_col=None, gene_bfs_log_bf_col=None, gene_bfs_combined_col=None, gene_bfs_prob_col=None, gene_bfs_prior_col=None, gene_bfs_sd_col=None, **kwargs):

        #require X matrix
        if self.genes is not None:
            genes = self.genes
            gene_to_ind = self.gene_to_ind
        else:
            genes = []
            gene_to_ind = {}

        aligned_gene_bfs = pegs_load_aligned_gene_bfs(
            gene_bfs_in,
            genes=genes,
            gene_to_ind=gene_to_ind,
            gene_bfs_id_col=gene_bfs_id_col,
            gene_bfs_log_bf_col=gene_bfs_log_bf_col,
            gene_bfs_combined_col=gene_bfs_combined_col,
            gene_bfs_prob_col=gene_bfs_prob_col,
            gene_bfs_prior_col=gene_bfs_prior_col,
            background_log_bf=self.background_log_bf,
            gene_label_map=self.gene_label_map,
            open_text_fn=open_gz,
            get_col_fn=_get_col,
            log_fn=lambda message: log(message, INFO),
            warn_fn=warn,
            bail_fn=bail,
        )
        return (
            aligned_gene_bfs.gene_bfs,
            aligned_gene_bfs.extra_genes,
            aligned_gene_bfs.extra_gene_bfs,
            aligned_gene_bfs.gene_in_combined,
            aligned_gene_bfs.gene_in_priors,
        )

    def read_gene_covs(self, gene_covs_in, gene_covs_id_col=None, gene_covs_cov_cols=None, **kwargs):

        #require X matrix

        if gene_covs_in is None:
            bail("Require --gene-covs-in for this operation")

        log("Reading --gene-covs-in file %s" % gene_covs_in, INFO)
        if self.genes is not None:
            genes = self.genes
            gene_to_ind = self.gene_to_ind
        else:
            genes = []
            gene_to_ind = {}

        aligned_gene_covs = pegs_load_aligned_gene_covariates(
            gene_covs_in,
            genes=genes,
            gene_to_ind=gene_to_ind,
            gene_covs_id_col=gene_covs_id_col,
            open_text_fn=open_gz,
            get_col_fn=_get_col,
            log_fn=lambda message: log(message, DEBUG),
            warn_fn=warn,
            bail_fn=bail,
        )
        cov_names = aligned_gene_covs.cov_names

        if len(cov_names) == 0:
            warn("No covariates in file")
            return
        return (
            cov_names,
            aligned_gene_covs.gene_covs,
            aligned_gene_covs.extra_genes,
            aligned_gene_covs.extra_gene_covs,
        )


    def convert_prior_to_var(self, top_prior, num, frac):
        top_bf = np.log((top_prior) / (1 - top_prior)) - self.background_log_bf
        if top_bf <= 0:
            bail("--top-gene-set-prior must be above background (%.4g)" % self.background_prior)
        if frac is None:
            frac = 1
        if frac <= 0 or frac > 1:
            bail("--frac-gene-sets-for-prior must be in (0,1]")
        var = frac * np.square(top_bf / (-scipy.stats.norm.ppf(1.0 / (num * frac))))
        return var

    def _distill_huge_signal_bfs(self, huge_signal_bfs, huge_signal_posteriors, huge_signal_sum_gene_cond_probabilities, huge_signal_mean_gene_pos, huge_signal_max_closest_gene_prob, cap_region_posterior, scale_region_posterior, phantom_region_posterior, allow_evidence_of_absence, gene_covariates, gene_covariates_mask, gene_covariates_mat_inv, gene_covariate_names, gene_covariate_intercept_index, gene_prob_genes, total_genes=None, rel_prior_log_bf=None):

        if huge_signal_bfs is None:
            return

        if total_genes is not None:
            total_genes = self.genes

        #gene_to_ind = pegs_construct_map_to_ind(gene_prob_genes)


        if rel_prior_log_bf is None:
            prior_log_bf = np.full((1,huge_signal_bfs.shape[0]), self.background_log_bf)
        else:
            prior_log_bf = rel_prior_log_bf + self.background_log_bf
            if len(prior_log_bf.shape) == 1:
                prior_log_bf = prior_log_bf[np.newaxis,:]

        if prior_log_bf.shape[1] != huge_signal_bfs.shape[0]:
            bail("Error: priors shape did not match huge results shape (%s vs. %s)" % (prior_log_bf.shape, huge_signal_bfs.T.shape))

        if phantom_region_posterior:
            #first add an entry at the end to prior that is background_prior
            prior_log_bf = np.hstack((prior_log_bf, np.full((prior_log_bf.shape[0], 1), self.background_log_bf)))

            #then add a row to the bottom of signal_bfs
            phantom_probs = np.zeros(huge_signal_sum_gene_cond_probabilities.shape)
            phantom_mask = np.logical_and(huge_signal_sum_gene_cond_probabilities > 0, huge_signal_sum_gene_cond_probabilities < 1)

            phantom_probs[phantom_mask] = 1.0 - huge_signal_sum_gene_cond_probabilities[phantom_mask]

            #we need to set the BFs such that, when we add BFs below (with uniform prior) and then divide by total, we will get huge_signal_sum_gene_cond_probabilities for the non phantom
            #we *cannot* just convert phantom prob to phantom bf (like we do for signals; e.g. phantom_bfs = (phantom_probs / (1 - phantom_probs)) / self.background_bf) because the signals are defined as marginal probabilities
            #the BF needed to take gene in isolation from 0.05 to posterior
            #for phantom, we don't know the marginal -- it is inherently a joint estimate
            phantom_bfs = np.zeros(phantom_probs.shape)
            phantom_bfs[phantom_mask] = huge_signal_bfs.sum(axis=0).A1[phantom_mask] * (1.0 / huge_signal_sum_gene_cond_probabilities[phantom_mask] - 1.0)

            huge_signal_bfs = sparse.csc_matrix(sparse.vstack((huge_signal_bfs, phantom_bfs)))

            huge_signal_sum_gene_cond_probabilities = huge_signal_sum_gene_cond_probabilities + phantom_probs

        prior_bf = np.exp(prior_log_bf)

        prior = prior_bf / (1 + prior_bf)

        prior[prior == 1] = 1 - 1e-4
        prior[prior == 0] = 1e-4


        #utility sparse matrices to use within loop
        #huge results matrix has posteriors for the region
        signal_log_priors = sparse.csr_matrix(copy.copy(huge_signal_bfs).T)
        sparse_aux = copy.copy(signal_log_priors)

        huge_results = np.zeros(prior_log_bf.shape)
        for i in range(prior_log_bf.shape[0]):

            #need prior * (1 - other_prior)^N in each entry
            #due to sparse matrices, and limiting memory usage, have to overwrite 
            #also, the reason for the complication below is that we have to work in log space, which
            #requires addition rather than subtraction, which we can't do directly on sparse matrices
            #we also need to switch between operating on data (when we do pointwise operations)
            #and operating on matrices (when we sum across axes)

            #priors specific to the signal
            sparse_aux.data = np.ones(len(sparse_aux.data))
            sparse_aux = sparse.csr_matrix(sparse_aux.multiply(np.log(1 - prior[i,:])))
            other_log_priors = sparse_aux.sum(axis=1).A1

            signal_log_priors.data = np.ones(len(signal_log_priors.data))
            signal_log_priors = sparse.csr_matrix(signal_log_priors.multiply(np.log(prior[i,:])))

            #now this has log(prior/(1-prior))
            signal_log_priors.data = signal_log_priors.data - sparse_aux.data

            #now need to add in (1-prior)^N
            sparse_aux.data = np.ones(len(sparse_aux.data))
            sparse_aux = sparse.csr_matrix(sparse_aux.T.multiply(other_log_priors).T)

            #now this has log(prior * (1-other_prior)^N)
            signal_log_priors.data = signal_log_priors.data + sparse_aux.data

            #now normalize
            #log_sum_bf = c + np.log(np.sum(np.exp(var_log_bf[region_vars] - c)))
            #where c is max value

            #to get max, have to add minimum (to get it over zero) and then subtract it
            c = signal_log_priors.min(axis=1)

            #ensure all c are positive (otherwise this will be removed from the sparse matrix and break the subsequent operations on data)
            c = c.toarray()
            c[c == 0] = np.min(c) * 1e-4

            sparse_aux.data = np.ones(len(sparse_aux.data))
            sparse_aux = sparse.csr_matrix(sparse_aux.multiply(c))

            signal_log_priors.data = signal_log_priors.data - sparse_aux.data

            c = signal_log_priors.max(axis=1) + c

            signal_log_priors.data = signal_log_priors.data + sparse_aux.data

            sparse_aux.data = np.ones(len(sparse_aux.data))
            sparse_aux = sparse.csr_matrix(sparse_aux.multiply(c))
            #store the max
            c_data = copy.copy(sparse_aux.data)

            #subtract c
            sparse_aux.data = np.exp(signal_log_priors.data - c_data)

            norms = sparse_aux.sum(axis=1).A1
            norms[norms != 0] = np.log(norms[norms != 0])

            sparse_aux.data = np.ones(len(sparse_aux.data))
            sparse_aux = sparse.csr_matrix(sparse_aux.T.multiply(norms).T)

            sparse_aux.data = c_data + sparse_aux.data

            signal_log_priors.data = signal_log_priors.data - sparse_aux.data

            #finally, we can obtain the priors matrix
            signal_log_priors.data = np.exp(signal_log_priors.data)            
            
            signal_priors = signal_log_priors.T

            #first have to adjust for priors
            #convert to BFs
            #we are overwriting data but drawing from the original (copied) huge_signal_bfs
            #multiply by priors. The final probabilities are proportional to the BFs * prior probabilities

            cur_huge_signal_bfs = huge_signal_bfs.multiply(signal_priors)

            #rescale; these are now posteriors for the signal
            #either:
            #1. sum to 1 (scale_region_posterior)
            #2. reduce (but don't increase) to 1 (cap_region_posterior)
            #3. leave them as is (but scale to be bayes factors before normalizing)

            new_norms = cur_huge_signal_bfs.sum(axis=0).A1

            if not scale_region_posterior and not cap_region_posterior:
                #treat them as bayes factors
                new_norms /= (huge_signal_mean_gene_pos * (np.mean(prior_bf[i:])/self.background_bf))

            #this scales everything to sum to 1

            #in case any have zero
            new_norms[new_norms == 0] = 1

            cur_huge_signal_bfs = sparse.csr_matrix(cur_huge_signal_bfs.multiply(1.0 / new_norms))

            if not scale_region_posterior and not cap_region_posterior:
                #convert them back to probabilities
                cur_huge_signal_bfs.data = cur_huge_signal_bfs.data / (1 + cur_huge_signal_bfs.data)

            #cur_huge_signal_bfs are actually now probabilities that sum to 1 (incorporating priors)
            #signal_cap_norm_factor incorporates both scaling to the signal prob, as well as any capping to reduce the probabilities to their original sum
            if cap_region_posterior:
                signal_cap_norm_factor = huge_signal_posteriors * huge_signal_sum_gene_cond_probabilities
            else:
                signal_cap_norm_factor = copy.copy(huge_signal_posteriors)

            #this is the "fudge factor" that accounts for the fact that the causal gene could be outside of this window
            #we don't need to do it under the phantom gene model because we already added a phantom gene to absorb 1 - max_closest_gene_prob

            max_per_signal = cur_huge_signal_bfs.max(axis=0).todense().A1 * signal_cap_norm_factor
            overflow_mask = max_per_signal > huge_signal_max_closest_gene_prob
            signal_cap_norm_factor[overflow_mask] *= (huge_signal_max_closest_gene_prob / max_per_signal[overflow_mask])

            #rescale to the signal probability (cur_huge_signal_posteriors is the probability that the signal is true)
            cur_huge_signal_bfs = sparse.csr_matrix(cur_huge_signal_bfs.multiply(signal_cap_norm_factor))


            if not allow_evidence_of_absence:
                #this part now ensures that nothing with absence of evidence has evidence of absence
                #consider two coin flips: first, it is causal due to the GWAS signal here
                #second, it is causal for some other reason
                #to not be causal, both need to come up negatuve
                #first has probability equal to huge_signal_posteriors
                #second has probability equal to prior

                #cur_huge_signal_bfs.data = 1 - (1 - np.array(cur_huge_signal_bfs.data)) * (1 - prior[i,:])
                #but, we cannot subtract from sparse matrices so have to do it this way

                cur_huge_signal_bfs.data = 1 - cur_huge_signal_bfs.data
                
                #cur_huge_signal_bfs = sparse.csc_matrix(cur_huge_signal_bfs.T.multiply(1 - prior[i,:]).T)
                cur_huge_signal_bfs = sparse.csc_matrix(cur_huge_signal_bfs.T.multiply(1 - self.background_prior).T)

                cur_huge_signal_bfs.data = 1 - cur_huge_signal_bfs.data            


            if cur_huge_signal_bfs.shape[1] > 0:
                #disable option to sum huge
                #if sum_huge:
                #    cur_huge_signal_bfs.data = np.log(1 - cur_huge_signal_bfs.data)
                #    huge_results[i,:] = 1 - np.exp(cur_huge_signal_bfs.sum(axis=1).A1)

                #This now has strongest signal posterior across all signals
                huge_results[i,:] = cur_huge_signal_bfs.max(axis=1).todense().A1


        #anything that was zero tacitly has probability equal to prior

        huge_results[huge_results == 0] = self.background_prior

        absent_genes = set()
        if total_genes is not None:
            #have to account for these
            absent_genes = set(total_genes) - set(gene_prob_genes)

        total_prob_causal = np.sum(huge_results)
        mean_prob_causal = (total_prob_causal + self.background_prior * len(absent_genes)) / (len(gene_prob_genes) + len(absent_genes)) 
        norm_constant = self.background_prior / mean_prob_causal

        #only normalize if enough genes
        max_prob = 1
        if len(gene_prob_genes) < 1000:
            norm_constant = max_prob
        elif norm_constant >= 1:
            norm_constant = max_prob

        #fix the maximum background prior across all genes
        #max_background_prior = None
        #if max_background_prior is not None and mean_prob_causal > max_background_prior:
        #    norm_constant = max_background_prior / mean_prob_causal
        norm_constant = max_prob

        if norm_constant != 1:
            log("Scaling output probabilities by %.4g" % norm_constant)

        huge_results *= norm_constant
        #now have to subtract out the prior

        okay_mask = huge_results < 1

        #we will add this to the prior to get the final posterior, so just subtract it
        huge_results[okay_mask] = np.log(huge_results[okay_mask] / (1 - huge_results[okay_mask])) - self.background_log_bf

        huge_results[~okay_mask] = np.max(huge_results[okay_mask])

        absent_prob = self.background_prior * norm_constant
        absent_log_bf = np.log(absent_prob / (1 - absent_prob)) - self.background_log_bf

        if phantom_region_posterior:
            huge_results = huge_results[:,:-1]

        huge_results_uncorrected = huge_results
        if gene_covariates is not None:
            (huge_results, huge_results_uncorrected, _) = self._correct_huge(huge_results, gene_covariates, gene_covariates_mask, gene_covariates_mat_inv, gene_covariate_names, gene_covariate_intercept_index)

        huge_results = np.squeeze(huge_results)
        huge_results_uncorrected = np.squeeze(huge_results_uncorrected)

        return (huge_results, huge_results_uncorrected, absent_genes, absent_log_bf)


    def _correct_huge(self, huge_results, gene_covariates, gene_covariates_mask, gene_covariates_mat_inv, gene_covariate_names, gene_covariate_intercept_index):

        if huge_results is None:
            return (None, None, None)

        if len(huge_results.shape) == 1:
            huge_results = huge_results[np.newaxis,:]

        huge_results_uncorrected = copy.copy(huge_results)
        gene_covariate_betas = None

        if gene_covariates is not None:
            assert(gene_covariates_mat_inv is not None)
            assert(gene_covariates_mask is not None)
            assert(gene_covariate_names is not None)
            assert(gene_covariate_intercept_index is not None)

            huge_results_mask = np.all(huge_results < np.mean(huge_results) + 5 * np.std(huge_results), axis=0)
            cur_gene_covariates_mask = np.logical_and(gene_covariates_mask, huge_results_mask)
            #dimensions are num_covariates x chains

            if self.huge_sparse_mode:
                pred_slopes = self.gene_covariate_slope_defaults.repeat(huge_results.shape[0]).reshape((len(self.gene_covariate_slope_defaults), huge_results.shape[0]))
            else:
                pred_slopes = gene_covariates_mat_inv.dot(gene_covariates[cur_gene_covariates_mask,:].T).dot(huge_results[:,cur_gene_covariates_mask].T)

            gene_covariate_betas = np.mean(pred_slopes, axis=1)
            log("Mean slopes are %s" % gene_covariate_betas, TRACE)

            non_intercept_inds = [i for i in range(len(gene_covariate_names)) if i != gene_covariate_intercept_index]

            param_names = ["%s_beta" % gene_covariate_names[i] for i in non_intercept_inds]
            param_values = gene_covariate_betas
            self._record_params(dict(zip(param_names, param_values)), record_only_first_time=True)

            pred_huge_adjusted = huge_results - gene_covariates[:,non_intercept_inds].dot(pred_slopes[non_intercept_inds,:]).T

            #flag those that are very high
            max_huge_change = 1.0

            bad_mask = pred_huge_adjusted - huge_results > max_huge_change

            if np.sum(bad_mask) > 0:
                warn("Not correcting %d genes for covariates due to large swings; there may be a problem with the covariates or input" % np.sum(bad_mask))

            huge_results[~bad_mask] = pred_huge_adjusted[~bad_mask]
            #JASON OLD
            #huge_results[~bad_mask] = pred_huge_residuals[~bad_mask]

        huge_results = np.squeeze(huge_results)
        huge_results_uncorrected = np.squeeze(huge_results_uncorrected)

        return (huge_results, huge_results_uncorrected, gene_covariate_betas)

    def _read_correlations(self, gene_cor_file=None, gene_loc_file=None, gene_cor_file_gene_col=1, gene_cor_file_cor_start_col=10, compute_correlation_distance_function=True):
        if gene_cor_file is not None:
            log("Reading in correlations from %s" % gene_cor_file)
            unique_genes = np.array([True] * len(self.genes))
            correlation_m = [np.ones(len(self.genes))]
            with open(gene_cor_file) as gene_cor_fh:
                gene_cor_file_gene_col = gene_cor_file_gene_col - 1
                gene_cor_file_cor_start_col = gene_cor_file_cor_start_col - 1
                #store the genes in order, which we will need in order to map from each line in the file to correlation
                gene_cor_file_gene_names = []
                new_gene_index = {}
                cor_file_index = 0
                j = 0
                for line in gene_cor_fh:
                    if line[0] == "#":
                        continue
                    cols = line.strip('\n').split()
                    if len(cols) < gene_cor_file_cor_start_col:
                        bail("Not enough columns in --gene-cor-file. Offending line:\n\t%s" % line)
                    gene_name = cols[gene_cor_file_gene_col]
                    if self.gene_label_map is not None and gene_name in self.gene_label_map:
                        gene_name = self.gene_label_map[gene_name]

                    gene_cor_file_gene_names.append(gene_name)
                    i = j - 1
                    if gene_name in self.gene_to_ind:
                        new_gene_index[gene_name] = cor_file_index
                        gene_correlations = [float(x) for x in cols[gene_cor_file_cor_start_col:]]
                        for gc_i in range(1,len(gene_correlations)+1):
                            cur_cor = gene_correlations[-gc_i]
                            if gc_i > cor_file_index:
                                bail("Error in --gene-cor-file: number of correlations is more than the number of genes seen to this point")
                            gene_i = gene_cor_file_gene_names[cor_file_index - gc_i]
                            if gene_i not in self.gene_to_ind:
                                continue
                            if cur_cor >= 1:
                                unique_genes[self.gene_to_ind[gene_i]] = False
                                #log("Excluding %s (correlation=%.4g with %s)" % (gene_i, cur_cor, gene_name), TRACE)

                            #store the values for the regression(s)
                            correlation_m_ind = j - i
                            while correlation_m_ind >= len(correlation_m):
                                correlation_m.append(np.zeros(len(self.genes)))
                            correlation_m[correlation_m_ind][i] = cur_cor
                            i -= 1
                        j += 1
                    cor_file_index += 1
            correlation_m = np.array(correlation_m)

            #now subset down the duplicate locations
            #self._subset_genes(unique_genes, skip_V=True, skip_scale_factors=True)
            #correlation_m = correlation_m[:,unique_genes]
            #log("Excluded %d duplicate genes" % sum(~unique_genes))

            sorted_gene_indices = sorted(range(len(self.genes)), key=lambda k: new_gene_index[self.genes[k]] if self.genes[k] in new_gene_index else 0)
            #sort the X and y values
            self._sort_genes(sorted_gene_indices, skip_V=True, skip_scale_factors=True)

        else:
            if gene_loc_file is None:
                bail("Need --gene-loc-file if don't specify --gene-cor-file")

            self.gene_locations = {}
            log("Reading gene locations")

            if self.gene_to_chrom is None:
                self.gene_to_chrom = {}
            if self.gene_to_pos is None:
                self.gene_to_pos = {}

            unique_genes = np.array([True] * len(self.genes))
            location_genes = {}
            with open(gene_loc_file) as gene_loc_fh:
                for line in gene_loc_fh:
                    cols = line.strip('\n').split()
                    if len(cols) != 6:
                        bail("Format for --gene-loc-file is:\n\tgene_id\tchrom\tstart\tstop\tstrand\tgene_name\nOffending line:\n\t%s" % line)
                    gene_name = cols[5]
                    if gene_name not in self.gene_to_ind:
                        continue

                    chrom = cols[1]
                    start = int(cols[2])
                    end = int(cols[3])

                    self.gene_to_chrom[gene_name] = chrom
                    self.gene_to_pos[gene_name] = (start, end)

                    location = (chrom, start, end)
                    self.gene_locations[gene_name] = location
                    if location in location_genes:
                        #keep the one with highest Y
                        old_ind = self.gene_to_ind[location_genes[location]]
                        new_ind = self.gene_to_ind[gene_name]
                        if self.Y[old_ind] >= self.Y[new_ind]:
                            unique_genes[new_ind] = False
                            log("Excluding %s (duplicate of %s)" % (self.genes[new_ind], self.genes[old_ind]), TRACE)
                        else:
                            unique_genes[new_ind] = True
                            unique_genes[old_ind] = False
                            log("Excluding %s (duplicate of %s)" % (self.genes[old_ind], self.genes[new_ind]), TRACE)
                            location_genes[location] = gene_name
                    else:
                        location_genes[location] = gene_name

            #now subset down the duplicate locations
            self._subset_genes(unique_genes, skip_V=True, skip_scale_factors=True)

            sorted_gene_indices = sorted(range(len(self.genes)), key=lambda k: self.gene_locations[self.genes[k]] if self.genes[k] in self.gene_locations else ("NA", 0))

            #sort the X and y values
            self._sort_genes(sorted_gene_indices, skip_V=True, skip_scale_factors=True)

            #now we have to determine the relationship between distance and correlation
            correlation_m = self._compute_correlations_from_distance(compute_correlation_distance_function=compute_correlation_distance_function)

        #set the diagonal to 1
        correlation_m[0,:] = 1.0

        log("Banded correlation matrix: shape %s, %s" % (correlation_m.shape[0], correlation_m.shape[1]), DEBUG)
        log("Non-zero entries: %s" % sum(sum(correlation_m > 0)), DEBUG)

        return correlation_m

    def _compute_correlations_from_distance(self, Y=None, compute_correlation_distance_function=True):
        if self.genes is None:
            return None

        if Y is None:
            Y = self.Y

        if Y is None or self.gene_locations is None:
            return None

        if self.huge_sparse_mode:
            log("Too few genes from HuGE: using pre-computed correlation function", DEBUG)
            compute_correlation_distance_function = False

        correlation_m = [np.zeros(len(self.genes))]

        max_distance_to_model = 1000000.0
        num_bins = 1000
        distance_num = np.zeros(num_bins)
        distance_denom = np.zeros(num_bins)
        log("Calculating distance/correlation function")
        #this loop does two things
        #first, it stores the distances in a banded matrix -- this will be used later to compute the correlations
        #second, it stores the various distances / empirical covariances in two arrays for doing the regression
        for i in range(len(self.genes)):
            if self.genes[i] in self.gene_locations:
                loc = self.gene_locations[self.genes[i]]
                #traverse in each direction to find pairs within a range
                for j in range(i+1, len(self.genes)):
                    if self.genes[j] in self.gene_locations:
                        loc2 = self.gene_locations[self.genes[j]]
                        if not loc[0] == loc2[0]:
                            continue
                        distance = np.abs(loc2[1] - loc[1])
                        if distance > max_distance_to_model:
                            break
                        #store the values for the regression(s)
                        bin_number = int((distance / max_distance_to_model) * (num_bins - 1))
                        if Y[i] != 0:
                            distance_num[bin_number] += Y[i] * Y[j]
                            distance_denom[bin_number] += Y[i]**2
                        #store the distances for later
                        correlation_m_ind = j - i
                        while correlation_m_ind >= len(correlation_m):
                            correlation_m.append(np.array([np.inf] * len(self.genes)))
                        correlation_m[correlation_m_ind][i] = distance

        correlation_m = np.array(correlation_m)

        # fit function
        slope = -5.229e-07
        intercept = 0.54

        
        bin_Y = distance_num[distance_denom != 0] / distance_denom[distance_denom != 0]
        bin_X = (np.array(range(len(distance_num))) * (max_distance_to_model / num_bins))[distance_denom != 0]

        if compute_correlation_distance_function and np.sum(bin_Y != 0) > 0:
            sd_outlier_threshold = 3
            bin_outlier_max = np.mean(bin_Y) + sd_outlier_threshold * np.std(bin_Y)
            bin_mask = np.logical_and(bin_Y > -bin_outlier_max, bin_Y < bin_outlier_max)
            bin_Y = bin_Y[bin_mask]
            bin_X = bin_X[bin_mask]

            slope = np.cov(bin_X, bin_Y)[0,1] / np.var(bin_X)
            intercept = np.mean(bin_Y - bin_X * slope)
            max_distance = -intercept / slope
            if slope > 0:
                log("Slope was positive; setting all correlations to zero")
                intercept = 0
                slope = 0
            elif intercept < 0:
                log("Incercept was negative; setting all correlations to zero")                
                intercept = 0
                slope = 0
            else:
                log("Fit function from bins: r^2 = %.2g%.4gx; max distance=%d" % (intercept, slope, max_distance))
        else:
            max_distance = -intercept / slope
            log("Using precomputed function: r^2 = %.2g%.4gx; max distance=%d" % (intercept, slope, max_distance))

        if slope < 0:
            max_distance = -intercept / slope
            log("Using function: r^2 = %.2g + %.4g * x; max distance=%d" % (intercept, slope, max_distance))                
            self._record_params({"correlation_slope": slope, "correlation_intercept": intercept, "correlation_max_dist": max_distance})

            #map the values over from raw values to correlations/covariances
            correlation_m = intercept + slope * correlation_m
            correlation_m[correlation_m <= 0] = 0
            correlation_m[0,:] = 1.0
        else:
            correlation_m[0,:] = 1.0
            correlation_m[1:,:] = 0.0

        return correlation_m
            

    def _compute_beta_tildes(self, X, Y, y_var=None, scale_factors=None, mean_shifts=None, resid_correlation_matrix=None, log_fun=log):
        return pegs_compute_beta_tildes(
            X,
            Y,
            y_var=y_var,
            scale_factors=scale_factors,
            mean_shifts=mean_shifts,
            resid_correlation_matrix=resid_correlation_matrix,
            calc_x_shift_scale_fn=self._calc_X_shift_scale,
            finalize_regression_fn=self._finalize_regression,
            bail_fn=bail,
            log_fun=log_fun,
            debug_level=DEBUG,
        )

    def _compute_multivariate_beta_tildes(self, X, Y, resid_correlation_matrix=None, add_intercept=True, covs=None):
        return pegs_compute_multivariate_beta_tildes(
            X,
            Y,
            resid_correlation_matrix=resid_correlation_matrix,
            add_intercept=add_intercept,
            covs=covs,
            finalize_regression_fn=self._finalize_regression,
        )

    def _compute_logistic_beta_tildes(self, X, Y, scale_factors=None, mean_shifts=None, resid_correlation_matrix=None, convert_to_dichotomous=True, rel_tol=0.01, X_stacked=None, append_pseudo=True, log_fun=log):
        return pegs_compute_logistic_beta_tildes(
            X,
            Y,
            scale_factors=scale_factors,
            mean_shifts=mean_shifts,
            resid_correlation_matrix=resid_correlation_matrix,
            convert_to_dichotomous=convert_to_dichotomous,
            rel_tol=rel_tol,
            X_stacked=X_stacked,
            append_pseudo=append_pseudo,
            calc_x_shift_scale_fn=self._calc_X_shift_scale,
            finalize_regression_fn=self._finalize_regression,
            bail_fn=bail,
            log_fun=log_fun,
            debug_level=DEBUG,
            trace_level=TRACE,
            runtime_Y=self.Y,
            runtime_Y_for_regression=self.Y_for_regression,
        )

    def _finalize_regression(self, beta_tildes, ses, se_inflation_factors):
        return pegs_finalize_regression_outputs(
            beta_tildes,
            ses,
            se_inflation_factors,
            log_fn=log,
            warn_fn=warn,
            trace_level=TRACE,
        )

    def _correct_beta_tildes(self, beta_tildes, ses, se_inflation_factors, total_qc_metrics, total_qc_metrics_directions, correct_mean=True, correct_var=True, add_missing=True, add_ignored=True, correct_ignored=False, fit=True):
        return pegs_correct_beta_tildes(
            self,
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
            compute_beta_tildes_fn=self._compute_beta_tildes,
            log_fn=log,
            warn_fn=warn,
            trace_level=TRACE,
            debug_level=DEBUG,
        )

    # ==========================================================================
    # Section: Beta Sampling Core (inner Gibbs for gene-set effects).
    # ==========================================================================
    # there are two levels of parallelization here:
    # 1. num_chains: sample multiple independent chains with the same beta/se/V
    # 2. multiple parallel runs with different beta/se (and potentially V). To do this,
    #    pass in lists of beta and se (must be the same length) and an optional list of V
    #    (V must have same length as beta OR be a single shared matrix).
    #
    # to run this in parallel, pass in a two-dimensional beta_tildes matrix where rows are
    # parallel runs and columns are gene sets. V can also be 3D with first dimension matching
    # parallel runs.
    def _calculate_non_inf_betas(self, initial_p, return_sample=False, max_num_burn_in=None, max_num_iter=1100, min_num_iter=10, num_chains=10, r_threshold_burn_in=1.01, use_max_r_for_convergence=True, eps=0.01, max_frac_sem=0.01, max_allowed_batch_correlation=None, beta_outlier_iqr_threshold=5, gauss_seidel=False, update_hyper_sigma=True, update_hyper_p=True, adjust_hyper_sigma_p=False, only_update_hyper=False, sigma_num_devs_to_top=2.0, p_noninf_inflate=1.0, num_p_pseudo=1, sparse_solution=False, sparse_frac_betas=None, betas_trace_out=None, betas_trace_gene_sets=None, beta_tildes=None, ses=None, V=None, X_orig=None, scale_factors=None, mean_shifts=None, is_dense_gene_set=None, ps=None, sigma2s=None, assume_independent=False, num_missing_gene_sets=None, debug_genes=None, debug_gene_sets=None, init_betas=None, init_postp=None):

        # ==========================================================================
        # Inner Beta Gibbs Phase 0: Validate inputs and set defaults.
        # ==========================================================================
        debug_gene_sets = None

        if max_num_burn_in is None:
            max_num_burn_in = int(max_num_iter * .25)
        if max_num_burn_in >= max_num_iter:
            max_num_burn_in = int(max_num_iter * .25)

        #if (update_hyper_p or update_hyper_sigma) and gauss_seidel:
        #    log("Using Gibbs sampling for betas since update hyper was requested")
        #    gauss_seidel = False

        if ses is None:
            ses = self.ses
        if beta_tildes is None:
            beta_tildes = self.beta_tildes
            
        if X_orig is None and not assume_independent:
            X_orig = self.X_orig
        if scale_factors is None:
            scale_factors = self.scale_factors
        if mean_shifts is None:
            mean_shifts = self.mean_shifts

        use_X = False
        if V is None and not assume_independent:
            if X_orig is None or scale_factors is None or mean_shifts is None:
                bail("Require X, scale, and mean if V is None")
            else:
                use_X = True
                log("Using low memory X instead of V", TRACE)

        if is_dense_gene_set is None:
            is_dense_gene_set = self.is_dense_gene_set
        if ps is None:
            ps = self.ps
        if sigma2s is None:
            sigma2s = self.sigma2s

        if self.sigma2 is None:
            bail("Need sigma to calculate betas!")

        if initial_p is not None:
            self.set_p(initial_p)

        if self.p is None and ps is None:
            bail("Need p to calculate non-inf betas")

        if not len(beta_tildes.shape) == len(ses.shape):
            bail("If running parallel beta inference, beta_tildes and ses must have same shape")

        if len(beta_tildes.shape) == 0 or beta_tildes.shape[0] == 0:
            bail("No gene sets are left!")

        # ==========================================================================
        # Inner Beta Gibbs Phase 1: Normalize all inputs to parallel-friendly shapes.
        # ==========================================================================
        #convert the beta_tildes and ses to matrices -- columns are num_parallel
        #they are always stored as matrices, with 1 column as needed
        #V on the other hand will be a 2-D matrix if it is constant across all parallel (or if there is only 1)
        #checking len(V.shape) can therefore distinguish a constant from variable V

        multiple_V = False
        sparse_V = False

        if len(beta_tildes.shape) > 1:
            num_gene_sets = beta_tildes.shape[1]

            if not beta_tildes.shape[0] == ses.shape[0]:
                bail("beta_tildes and ses must have same number of parallel runs")

            #dimensions should be num_gene_sets, num_parallel
            num_parallel = beta_tildes.shape[0]
            beta_tildes_m = copy.copy(beta_tildes)
            ses_m = copy.copy(ses)

            if V is not None and type(V) is sparse.csc_matrix:
                sparse_V = True
                multiple_V = False
            elif V is not None and len(V.shape) == 3:
                if not V.shape[0] == beta_tildes.shape[0]:
                    bail("V must have same number of parallel runs as beta_tildes")
                multiple_V = True
                sparse_V = False
            else:
                multiple_V = False
                sparse_V = False

        else:
            num_gene_sets = len(beta_tildes)
            if V is not None and type(V) is sparse.csc_matrix:
                num_parallel = 1
                multiple_V = False
                sparse_V = True
                beta_tildes_m = beta_tildes[np.newaxis,:]
                ses_m = ses[np.newaxis,:]
            elif V is not None and len(V.shape) == 3:
                num_parallel = V.shape[0]
                multiple_V = True
                sparse_V = False
                beta_tildes_m = np.tile(beta_tildes, num_parallel).reshape((num_parallel, len(beta_tildes)))
                ses_m = np.tile(ses, num_parallel).reshape((num_parallel, len(ses)))
            else:
                num_parallel = 1
                multiple_V = False
                sparse_V = False
                beta_tildes_m = beta_tildes[np.newaxis,:]
                ses_m = ses[np.newaxis,:]

        if num_parallel == 1 and multiple_V:
            multiple_V = False
            V = V[0,:,:]

        if multiple_V:
            assert(not use_X)

        if scale_factors.shape != mean_shifts.shape:
            bail("scale_factors must have same dimension as mean_shifts")

        def _to_parallel_matrix(values, name):
            values = np.asarray(values)
            if values.ndim == 1:
                if values.shape[0] != num_gene_sets:
                    bail("%s must have length num_gene_sets=%d; got %d" % (name, num_gene_sets, values.shape[0]))
                if num_parallel == 1:
                    return values[np.newaxis, :]
                return np.tile(values, num_parallel).reshape((num_parallel, values.shape[0]))
            if values.ndim == 2:
                if values.shape[0] != num_parallel:
                    bail("%s must have num_parallel=%d rows; got %d" % (name, num_parallel, values.shape[0]))
                if values.shape[1] != num_gene_sets:
                    bail("%s must have num_gene_sets=%d columns; got %d" % (name, num_gene_sets, values.shape[1]))
                return copy.copy(values)
            bail("%s must be a 1D or 2D array" % name)

        scale_factors_m = _to_parallel_matrix(scale_factors, "scale_factors")
        mean_shifts_m = _to_parallel_matrix(mean_shifts, "mean_shifts")
        is_dense_gene_set_m = _to_parallel_matrix(is_dense_gene_set, "is_dense_gene_set")

        if ps is not None:
            ps_m = _to_parallel_matrix(ps, "ps")
        else:
            ps_m = self.p

        if sigma2s is not None:
            orig_sigma2_m = _to_parallel_matrix(sigma2s, "sigma2s")
        else:
            orig_sigma2_m = self.sigma2

        # ==========================================================================
        # Inner Beta Gibbs Phase 2: Build gene-set batches and initialize state.
        # ==========================================================================
        #for efficiency, batch genes to be updated each cycle
        if assume_independent:
            gene_set_masks = [np.full(beta_tildes_m.shape[1], True)]
        else:
            gene_set_masks = self._compute_gene_set_batches(V, X_orig=X_orig, mean_shifts=mean_shifts, scale_factors=scale_factors, use_sum=True, max_allowed_batch_correlation=max_allowed_batch_correlation)
            
        sizes = [float(np.sum(x)) / (num_parallel if multiple_V else 1) for x in gene_set_masks]
        log("Analyzing %d gene sets in %d batches of gene sets; size range %d - %d" % (num_gene_sets, len(gene_set_masks), min(sizes) if len(sizes) > 0 else 0, max(sizes)  if len(sizes) > 0 else 0), DEBUG)

        #get the dimensions of the gene_set_masks to match those of the betas
        if num_parallel == 1:
            assert(not multiple_V)
            #convert the vectors into matrices with one dimension
            gene_set_masks = [x[np.newaxis,:] for x in gene_set_masks]
        elif not multiple_V:
            #we have multiple parallel but only one V
            gene_set_masks = [np.tile(x, num_parallel).reshape((num_parallel, len(x))) for x in gene_set_masks]

        #variables are denoted
        #v: vectors of dimension equal to the number of gene sets
        #m: data that varies by parallel runs and gene sets
        #t: data that varies by chains, parallel runs, and gene sets

        #rules:
        #1. adding a lower dimensional tensor to higher dimenional ones means final dimensions must match. These operations are usually across replicates
        #2. lower dimensional masks on the other hand index from the beginning dimensions (can use :,:,mask to index from end)
        
        tensor_shape = (num_chains, num_parallel, num_gene_sets)
        matrix_shape = (num_parallel, num_gene_sets)

        def _to_initial_tensor(values, name, clamp_unit_interval=False):
            if values is None:
                return None
            values = np.asarray(values)
            if values.ndim == 1:
                if values.shape[0] != num_gene_sets:
                    bail("%s must have length num_gene_sets=%d; got %d" % (name, num_gene_sets, values.shape[0]))
                values_m = np.tile(values, num_parallel).reshape((num_parallel, num_gene_sets))
                values_t = np.tile(values_m, num_chains).reshape(tensor_shape)
            elif values.ndim == 2:
                if values.shape[1] != num_gene_sets:
                    bail("%s must have num_gene_sets=%d columns; got %d" % (name, num_gene_sets, values.shape[1]))
                if values.shape[0] == num_parallel:
                    values_m = values
                elif values.shape[0] == 1 and num_parallel > 1:
                    values_m = np.tile(values, num_parallel).reshape((num_parallel, num_gene_sets))
                else:
                    bail("%s must have num_parallel=%d rows; got %d" % (name, num_parallel, values.shape[0]))
                values_t = np.tile(values_m, num_chains).reshape(tensor_shape)
            elif values.ndim == 3:
                if values.shape != tensor_shape:
                    bail("%s must have shape %s; got %s" % (name, str(tensor_shape), str(values.shape)))
                values_t = values
            else:
                bail("%s must be a 1D, 2D, or 3D array" % name)

            values_t = np.array(values_t, dtype=float)
            values_t = np.nan_to_num(values_t, nan=0.0, posinf=0.0, neginf=0.0)
            if clamp_unit_interval:
                values_t = np.clip(values_t, 0.0, 1.0)
            return values_t

        init_postp_t = _to_initial_tensor(init_postp, "init_postp", clamp_unit_interval=True)
        init_betas_t = _to_initial_tensor(init_betas, "init_betas")

        #these are current posterior means (including p and the conditional beta). They are used to calculate avg_betas
        #using these as the actual betas would yield the Gauss-seidel algorithm
        curr_post_means_t = np.zeros(tensor_shape)
        curr_postp_t = np.ones(tensor_shape)
        if init_postp_t is not None:
            curr_postp_t = init_postp_t

        #these are the current betas to be used in each iteration
        if init_betas_t is not None:
            curr_betas_t = init_betas_t
        else:
            initial_sd = np.std(beta_tildes_m)
            if initial_sd == 0:
                initial_sd = 1
            curr_betas_t = scipy.stats.norm.rvs(0, initial_sd, tensor_shape)

        res_beta_hat_t = np.zeros(tensor_shape)

        avg_betas_m = np.zeros(matrix_shape)
        avg_betas2_m = np.zeros(matrix_shape)
        avg_postp_m = np.zeros(matrix_shape)
        num_avg = 0

        #these are the posterior betas averaged across iterations
        sum_betas_t = np.zeros(tensor_shape)
        sum_betas2_t = np.zeros(tensor_shape)

        # Setting up constants
        #hyperparameters
        #shrinkage prior
        if self.sigma_power is not None:
            #sigma2_m = orig_sigma2_m * np.power(scale_factors_m, self.sigma_power)
            sigma2_m = self.get_scaled_sigma2(scale_factors_m, orig_sigma2_m, self.sigma_power, self.sigma_threshold_k, self.sigma_threshold_xo)

            #for dense gene sets, scaling by size doesn't make sense. So use mean size across sparse gene sets
            if np.sum(is_dense_gene_set_m) > 0:
                if np.sum(~is_dense_gene_set_m) > 0:
                    #sigma2_m[is_dense_gene_set_m] = self.sigma2 * np.power(np.mean(scale_factors_m[~is_dense_gene_set_m]), self.sigma_power)
                    sigma2_m[is_dense_gene_set_m] = self.get_scaled_sigma2(np.mean(scale_factors_m[~is_dense_gene_set_m]), self.sigma2, self.sigma_power, self.sigma_threshold_k, self.sigma_threshold_xo)
                else:
                    #sigma2_m[is_dense_gene_set_m] = self.sigma2 * np.power(np.mean(scale_factors_m), self.sigma_power)
                    sigma2_m[is_dense_gene_set_m] = self.get_scaled_sigma2(np.mean(scale_factors_m), self.sigma2, self.sigma_power, self.sigma_threshold_k, self.sigma_threshold_xo)

        else:
            sigma2_m = orig_sigma2_m

        if ps_m is not None and np.min(ps_m) != np.max(ps_m):
            p_text = "mean p=%.3g (%.3g-%.3g)" % (self.p, np.min(ps_m), np.max(ps_m))
        else:
            p_text = "p=%.3g" % (self.p)
        if np.min(orig_sigma2_m) != np.max(orig_sigma2_m):
            sigma2_text = "mean sigma=%.3g (%.3g-%.3g)" % (self.sigma2, np.min(orig_sigma2_m), np.max(orig_sigma2_m))
        else:
            sigma2_text = "sigma=%.3g" % (self.sigma2)

        if np.min(orig_sigma2_m) != np.max(orig_sigma2_m):
            sigma2_p_text = "mean sigma2/p=%.3g (%.3g-%.3g)" % (self.sigma2/self.p, np.min(orig_sigma2_m/ps_m), np.max(orig_sigma2_m/ps_m))
        else:
            sigma2_p_text = "sigma2/p=%.3g" % (self.sigma2/self.p)


        #maintain 10 most recent ps and sigma
        sigma2_deque = []
        p_deque = []


        tag = ""
        if assume_independent:
            tag = "independent "
        elif sparse_V:
            tag = "partially independent "
            
        log("Calculating %snon-infinitesimal betas with %s, %s; %s" % (tag, p_text, sigma2_text, sigma2_p_text))

        #generate the diagonals to use per replicate
        if assume_independent:
            V_diag_m = None
            account_for_V_diag_m = False
        else:
            if V is not None:
                if num_parallel > 1:
                    #dimensions are num_parallel, num_gene_sets, num_gene_sets
                    if multiple_V:
                        V_diag_m = np.diagonal(V, axis1=1, axis2=2)
                    else:
                        if sparse_V:
                            V_diag = V.diagonal()
                        else:
                            V_diag = np.diag(V)
                        V_diag_m = np.tile(V_diag, num_parallel).reshape((num_parallel, len(V_diag)))
                else:
                    if sparse_V:
                        V_diag_m = V.diagonal()[np.newaxis,:]                        
                    else:
                        V_diag_m = np.diag(V)[np.newaxis,:]

                account_for_V_diag_m = not np.isclose(V_diag_m, np.ones(matrix_shape)).all()
            else:
                #we compute it from X, so we know it is always 1
                V_diag_m = None
                account_for_V_diag_m = False

        se2s_m = np.power(ses_m,2)

        #the below code is based off of the LD-pred code for SNP PRS
        iteration_num = 0
        burn_in_phase_v = np.array([True for i in range(num_parallel)])


        betas_trace_fh = _open_optional_inner_betas_trace_file(betas_trace_out)

        prev_betas_m = None
        sigma_underflow = False
        printed_warning_swing = False
        printed_warning_increase = False

        # ==========================================================================
        # Inner Beta Gibbs Phase 3: Iterative Gibbs updates + convergence checks.
        # ==========================================================================
        while iteration_num < max_num_iter:  #Big iteration

            #if some have not converged, only sample for those that have not converged (for efficiency)
            compute_mask_v = copy.copy(burn_in_phase_v)
            if np.sum(compute_mask_v) == 0:
                compute_mask_v[:] = True

            hdmp_m = (sigma2_m / ps_m)
            hdmpn_m = hdmp_m + se2s_m
            hdmp_hdmpn_m = (hdmp_m / hdmpn_m)

            norm_scale_m = np.sqrt(np.multiply(hdmp_hdmpn_m, se2s_m))
            c_const_m = (ps_m / np.sqrt(hdmpn_m))

            d_const_m = (1 - ps_m) / ses_m

            iteration_num += 1

            #default to 1
            curr_postp_t[:,compute_mask_v,:] = np.ones(tensor_shape)[:,compute_mask_v,:]

            #sample whether each gene set has non-zero effect
            rand_ps_t = np.random.random(tensor_shape)
            #generate normal random variable sampling
            rand_norms_t = scipy.stats.norm.rvs(0, 1, tensor_shape)

            # 3a) Update each gene-set batch conditional on current chain state.
            for gene_set_mask_ind in range(len(gene_set_masks)):
                gene_set_mask_m = gene_set_masks[gene_set_mask_ind]

                # Intersect active parallels with current gene-set batch membership.
                compute_mask_m = np.logical_and(compute_mask_v, gene_set_mask_m.T).T

                # Value to use when determining if we should force an alpha shrink if
                # estimates are way off compared to heritability estimates.
                alpha_shrink = 1

                _update_inner_beta_gene_set_batch(
                    compute_mask_m=compute_mask_m,
                    compute_mask_v=compute_mask_v,
                    alpha_shrink=alpha_shrink,
                    rand_ps_t=rand_ps_t,
                    rand_norms_t=rand_norms_t,
                    hdmp_hdmpn_m=hdmp_hdmpn_m,
                    c_const_m=c_const_m,
                    d_const_m=d_const_m,
                    hdmpn_m=hdmpn_m,
                    se2s_m=se2s_m,
                    norm_scale_m=norm_scale_m,
                    assume_independent=assume_independent,
                    beta_tildes_m=beta_tildes_m,
                    curr_betas_t=curr_betas_t,
                    V=V,
                    multiple_V=multiple_V,
                    sparse_V=sparse_V,
                    use_X=use_X,
                    X_orig=X_orig,
                    scale_factors_m=scale_factors_m,
                    mean_shifts_m=mean_shifts_m,
                    betas_trace_out=betas_trace_out,
                    betas_trace_gene_sets=betas_trace_gene_sets,
                    account_for_V_diag_m=account_for_V_diag_m,
                    V_diag_m=V_diag_m,
                    curr_postp_t=curr_postp_t,
                    curr_post_means_t=curr_post_means_t,
                    gauss_seidel=gauss_seidel,
                    res_beta_hat_t=res_beta_hat_t,
                )

            _apply_inner_beta_sparsity_update(
                sparse_solution=sparse_solution,
                sparse_frac_betas=sparse_frac_betas,
                curr_postp_t=curr_postp_t,
                ps_m=ps_m,
                curr_post_means_t=curr_post_means_t,
                curr_betas_t=curr_betas_t,
                compute_mask_v=compute_mask_v,
            )

            curr_betas_m = np.mean(curr_post_means_t, axis=0)
            curr_postp_m = np.mean(curr_postp_t, axis=0)
            #no state should be preserved across runs, but take a random one just in case
            sample_betas_m = curr_betas_t[int(random.random() * curr_betas_t.shape[0]),:,:]
            sample_postp_m = curr_postp_t[int(random.random() * curr_postp_t.shape[0]),:,:]
            sum_betas_t[:,compute_mask_v,:] = sum_betas_t[:,compute_mask_v,:] + curr_post_means_t[:,compute_mask_v,:]
            sum_betas2_t[:,compute_mask_v,:] = sum_betas2_t[:,compute_mask_v,:] + np.square(curr_post_means_t[:,compute_mask_v,:])

            # 3b) Update convergence diagnostics and stopping conditions.
            #now calculate the convergence metrics
            R_m = np.zeros(matrix_shape)
            beta_weights_m = np.zeros(matrix_shape)
            sem2_m = np.zeros(matrix_shape)
            will_break = False
            if assume_independent:
                burn_in_phase_v[:] = False
            elif gauss_seidel:
                if prev_betas_m is not None:
                    sum_diff = np.sum(np.abs(prev_betas_m - curr_betas_m))
                    sum_prev = np.sum(np.abs(prev_betas_m))
                    tot_diff = sum_diff / sum_prev
                    log("Iteration %d: gauss seidel difference = %.4g / %.4g = %.4g" % (iteration_num+1, sum_diff, sum_prev, tot_diff), TRACE)
                    if iteration_num > min_num_iter and tot_diff < eps:
                        burn_in_phase_v[:] = False
                        log("Converged after %d iterations" % (iteration_num+1), INFO)
                prev_betas_m = curr_betas_m
            elif iteration_num > min_num_iter and np.sum(burn_in_phase_v) > 0:
                (R_m, beta_weights_m, burn_in_phase_v) = _update_inner_beta_rhat_and_outliers(
                    sum_betas_t=sum_betas_t,
                    sum_betas2_t=sum_betas2_t,
                    iteration_num=iteration_num,
                    compute_mask_v=compute_mask_v,
                    use_max_r_for_convergence=use_max_r_for_convergence,
                    beta_outlier_iqr_threshold=beta_outlier_iqr_threshold,
                    curr_betas_t=curr_betas_t,
                    curr_postp_t=curr_postp_t,
                    curr_post_means_t=curr_post_means_t,
                    burn_in_phase_v=burn_in_phase_v,
                    r_threshold_burn_in=r_threshold_burn_in,
                    num_parallel=num_parallel,
                )

            # Burn-in either converged or hit cap: collect samples/summaries.
            if np.sum(burn_in_phase_v) == 0 or iteration_num >= max_num_burn_in:

                #if we only care about parameters, we can return immediately (burn in stops hyper updates)
                if only_update_hyper:
                    return _return_inner_betas_result(betas_trace_fh, (None, None))

                #if we want a sample, first one after convergence will do
                if return_sample:

                    frac_increase = np.sum(np.abs(curr_betas_m) > np.abs(beta_tildes_m)) / curr_betas_m.size
                    if frac_increase > 0.01:
                        warn("A large fraction of betas (%.3g) are larger than beta tildes; this could indicate a problem. Try increasing --prune-gene-sets value or decreasing --sigma2" % frac_increase)
                        printed_warning_increase = True

                    frac_opposite = np.sum(curr_betas_m * beta_tildes_m < 0) / curr_betas_m.size
                    if frac_opposite > 0.01:
                        warn("A large fraction of betas (%.3g) are of opposite signs than the beta tildes; this could indicate a problem. Try increasing --prune-gene-sets value or decreasing --sigma2" % frac_opposite)
                        printed_warning_swing = False

                    if np.sum(burn_in_phase_v) > 0:
                        burn_in_phase_v[:] = False
                        log("Stopping burn in after %d iterations" % (iteration_num), INFO)


                    #max_beta = None
                    #if max_beta is not None:
                    #    threshold_ravel = max_beta * scale_factors_m.ravel()
                    #    if np.sum(sample_betas_m.ravel() > threshold_ravel) > 0:
                    #        log("Capped %d sample betas" % np.sum(sample_betas_m.ravel() > threshold_ravel), DEBUG)
                    #        sample_betas_mask = sample_betas_m.ravel() > threshold_ravel
                    #        sample_betas_m.ravel()[sample_betas_mask] = threshold_ravel[sample_betas_mask]
                    #    if np.sum(curr_betas_m.ravel() > threshold_ravel) > 0:
                    #        log("Capped %d curr betas" % np.sum(curr_betas_m.ravel() > threshold_ravel), DEBUG)
                    #        curr_betas_mask = curr_betas_m.ravel() > threshold_ravel
                    #        curr_betas_m.ravel()[curr_betas_mask] = threshold_ravel[curr_betas_mask]

                    return _return_inner_betas_result(
                        betas_trace_fh,
                        (sample_betas_m, sample_postp_m, curr_betas_m, curr_postp_m),
                    )

                #average over the posterior means instead of samples
                #these differ from sum_betas_v because those include the burn in phase
                avg_betas_m += np.sum(curr_post_means_t, axis=0)
                avg_betas2_m += np.sum(np.power(curr_post_means_t, 2), axis=0)
                avg_postp_m += np.sum(curr_postp_t, axis=0)
                num_avg += curr_post_means_t.shape[0]

                if iteration_num >= min_num_iter and num_avg > 1:
                    if gauss_seidel:
                        will_break = True
                    else:

                        #calculate these here for trace printing
                        avg_m = avg_betas_m
                        avg2_m = avg_betas2_m
                        sem2_m = ((avg2_m / (num_avg - 1)) - np.power(avg_m / num_avg, 2)) / num_avg
                        sem2_v = np.sum(sem2_m, axis=0)
                        zero_sem2_v = sem2_v == 0
                        sem2_v[zero_sem2_v] = 1
                        total_z_v = np.sqrt(np.sum(avg2_m / num_avg, axis=0) / sem2_v)
                        total_z_v[zero_sem2_v] = np.inf

                        log("Iteration %d: sum2=%.4g; sum sem2=%.4g; z=%.3g" % (iteration_num, np.sum(avg2_m / num_avg), np.sum(sem2_m), np.min(total_z_v)), TRACE)

                        min_z_sampling_var = 10
                        if np.all(total_z_v > min_z_sampling_var):
                            log("Desired precision achieved; stopping sampling")
                            will_break=True

            else:
                # 3c) Adapt hyperparameters (p, sigma) while still in burn-in.
                if update_hyper_p or update_hyper_sigma:
                    (new_p, new_sigma2) = _compute_inner_beta_hyper_update_targets(
                        curr_postp_t=curr_postp_t,
                        res_beta_hat_t=res_beta_hat_t,
                        hdmp_hdmpn_m=hdmp_hdmpn_m,
                        se2s_m=se2s_m,
                        curr_betas_m=curr_betas_m,
                        V_diag_m=V_diag_m,
                        num_parallel=num_parallel,
                        use_X=use_X,
                        multiple_V=multiple_V,
                        V=V,
                        sparse_V=sparse_V,
                        num_p_pseudo=num_p_pseudo,
                        curr_postp_m=curr_postp_m,
                        sigma_power=self.sigma_power,
                        scale_factors_m=scale_factors_m,
                        num_gene_sets=num_gene_sets,
                        num_missing_gene_sets=num_missing_gene_sets,
                        p_noninf_inflate=p_noninf_inflate,
                    )

                    sigma2_deque.append(new_sigma2 - self.sigma2)
                    p_deque.append(new_p - self.p)
                    deque_length = 5
                    if len(sigma2_deque) > deque_length:
                        sigma2_deque.pop(0)
                    if len(p_deque) > deque_length:
                        p_deque.pop(0)

                    sigma2_converged = abs(new_sigma2 - self.sigma2) / self.sigma2 < eps or (len(sigma2_deque) == deque_length and abs(np.mean(sigma2_deque)) / self.sigma2 < eps)
                    p_converged = abs(new_p - self.p) / self.p < eps or (len(p_deque) == deque_length and abs(np.mean(p_deque)) / self.p < eps)

                    if (not update_hyper_sigma or sigma2_converged) and (not update_hyper_p or p_converged):
                        log("Sigma converged to %.4g; p converged to %.4g" % (self.sigma2, self.p), TRACE)
                        update_hyper_sigma = False
                        update_hyper_p = False

                        if only_update_hyper:
                            return _return_inner_betas_result(betas_trace_fh, (None, None))

                    else:
                        if update_hyper_p:
                            log("Updating p from %.4g to %.4g" % (self.p, new_p), TRACE)
                            if not update_hyper_sigma and adjust_hyper_sigma_p:
                                #remember, sigma is the *total* variance term. It is equal to p * conditional_sigma.
                                #if we are only updating p, and adjusting sigma, we will leave the conditional_sigma constant, which means scaling the sigma
                                new_sigma2 = self.sigma2 / self.p * new_p
                                log("Updating sigma from %.4g to %.4g to maintain constant sigma/p" % (self.sigma2, new_sigma2), TRACE)
                                #we need to adjust the total sigma to keep the conditional sigma constant
                                self.set_sigma(new_sigma2, self.sigma_power)
                            self.set_p(new_p)
                                
                        if update_hyper_sigma:
                            if not sigma_underflow:
                                log("Updating sigma from %.4g to %.4g ( sqrt(sigma2/p)=%.4g )" % (self.sigma2, new_sigma2, np.sqrt(new_sigma2 / self.p)), TRACE)

                            lower_bound = 2e-3

                            if sigma_underflow or new_sigma2 / self.p < lower_bound:
                                
                                #first, try the heuristic of setting sigma2 so that strongest gene set has maximum possible p_bar

                                max_e_beta2 = np.argmax(beta_tildes_m / ses_m)

                                max_se2 = se2s_m.ravel()[max_e_beta2]
                                max_beta_tilde = beta_tildes_m.ravel()[max_e_beta2]
                                max_beta_tilde2 = np.square(max_beta_tilde)

                                #OLD inference
                                #make sigma/p easily cover the observation
                                #new_sigma2 = (max_beta_tilde2 - max_se2) * self.p
                                #make sigma a little bit smaller so that the top gene set is a little more of an outlier
                                #new_sigma2 /= sigma_num_devs_to_top

                                #NEW inference
                                max_beta = np.sqrt(max_beta_tilde2 - max_se2)
                                correct_sigma2 = self.p * np.square(max_beta / np.abs(scipy.stats.norm.ppf(1 / float(curr_betas_t.shape[2]) * self.p * 2)))
                                new_sigma2 = correct_sigma2

                                if new_sigma2 / self.p <= lower_bound:
                                    new_sigma2_from_top = new_sigma2
                                    new_sigma2 = lower_bound * self.p
                                    log("Sigma underflow including with determination from top gene set (%.4g)! Setting sigma to lower bound (%.4g * %.4g = %.4g) and no updates" % (new_sigma2_from_top, lower_bound, self.p, new_sigma2), TRACE)
                                else:
                                    log("Sigma underflow! Setting sigma determined from top gene set (%.4g) and no updates" % new_sigma2, TRACE)

                                if self.sigma_power is not None:

                                    #gene set specific sigma is internal sigma2 multiplied by scale_factor ** power
                                    #new_sigma2 is final sigma
                                    #so store internal value as final divided by average power

                                    #use power learned from mouse
                                    #using average across gene sets makes it sensitive to distribution of gene sets
                                    #need better solution for learning; since we are hardcoding from top gene set, just use mouse value
                                    new_sigma2 = new_sigma2 / np.power(self.MEAN_MOUSE_SCALE, self.sigma_power)

                                if not update_hyper_p and adjust_hyper_sigma_p:
                                    #remember, sigma is the *total* variance term. It is equal to p * conditional_sigma.
                                    #if we are only sigma p, and adjusting p, we will leave the conditional_sigma constant, which means scaling the p
                                    new_p = self.p / self.sigma2 * new_sigma2
                                    log("Updating p from %.4g to %.4g to maintain constant sigma/p" % (self.p, new_p), TRACE)
                                    #we need to adjust the total sigma to keep the conditional sigma constant
                                    self.set_p(new_p)

                                self.set_sigma(new_sigma2, self.sigma_power)
                                sigma_underflow = True
                            else:
                                self.set_sigma(new_sigma2, self.sigma_power)

                            #update the matrix forms of these variables
                            orig_sigma2_m *= new_sigma2 / np.mean(orig_sigma2_m)
                            if self.sigma_power is not None:
                                sigma2_m = self.get_scaled_sigma2(scale_factors_m, orig_sigma2_m, self.sigma_power, self.sigma_threshold_k, self.sigma_threshold_xo)

                                #for dense gene sets, scaling by size doesn't make sense. So use mean size across sparse gene sets
                                if np.sum(is_dense_gene_set_m) > 0:
                                    if np.sum(~is_dense_gene_set_m) > 0:
                                        sigma2_m[is_dense_gene_set_m] = self.get_scaled_sigma2(np.mean(scale_factors_m[~is_dense_gene_set_m]), orig_sigma2_m, self.sigma_power, self.sigma_threshold_k, self.sigma_threshold_xo)
                                    else:
                                        sigma2_m[is_dense_gene_set_m] = self.get_scaled_sigma2(np.mean(scale_factors_m), orig_sigma2_m, self.sigma_power, self.sigma_threshold_k, self.sigma_threshold_xo)
                            else:
                                sigma2_m = orig_sigma2_m

                            ps_m *= new_p / np.mean(ps_m)

            if betas_trace_fh is not None:
                _write_inner_beta_trace_rows(
                    betas_trace_fh=betas_trace_fh,
                    iteration_num=iteration_num,
                    num_parallel=num_parallel,
                    num_chains=num_chains,
                    num_gene_sets=num_gene_sets,
                    betas_trace_gene_sets=betas_trace_gene_sets,
                    curr_post_means_t=curr_post_means_t,
                    curr_betas_t=curr_betas_t,
                    curr_postp_t=curr_postp_t,
                    res_beta_hat_t=res_beta_hat_t,
                    scale_factors_m=scale_factors_m,
                    beta_tildes_m=beta_tildes_m,
                    ses_m=ses_m,
                    sigma2_m=sigma2_m,
                    ps_m=ps_m,
                    R_m=R_m,
                    beta_weights_m=beta_weights_m,
                    sem2_m=sem2_m,
                )

            if will_break:
                break

        # ==========================================================================
        # Inner Beta Gibbs Phase 4: Finalize posterior summaries and return.
        # ==========================================================================

        #log("%d\t%s" % (iteration_num, "\t".join(["%.3g\t%.3g" % (curr_betas_m[i,0], (np.mean(sum_betas_m, axis=0) / iteration_num)[i]) for i in range(curr_betas_m.shape[0])])), TRACE)

        avg_betas_m /= num_avg
        avg_postp_m /= num_avg

        if num_parallel == 1:
            avg_betas_m = avg_betas_m.flatten()
            avg_postp_m = avg_postp_m.flatten()

        #max_beta = None
        #if max_beta is not None:
        #    threshold_ravel = max_beta * scale_factors_m.ravel()
        #    if np.sum(avg_betas_m.ravel() > threshold_ravel) > 0:
        #        log("Capped %d sample betas" % np.sum(avg_betas_m.ravel() > threshold_ravel), DEBUG)
        #        avg_betas_mask = avg_betas_m.ravel() > threshold_ravel
        #        avg_betas_m.ravel()[avg_betas_mask] = threshold_ravel[avg_betas_mask]

        frac_increase = np.sum(np.abs(curr_betas_m) > np.abs(beta_tildes_m)) / curr_betas_m.size
        if frac_increase > 0.01:
            warn("A large fraction of betas (%.3g) are larger than beta tildes; this could indicate a problem. Try increasing --prune-gene-sets value or decreasing --sigma2" % frac_increase)
            printed_warning_increase = True

        frac_opposite = np.sum(curr_betas_m * beta_tildes_m < 0) / curr_betas_m.size
        if frac_opposite > 0.01:
            warn("A large fraction of betas (%.3g) are of opposite signs than the beta tildes; this could indicate a problem. Try increasing --prune-gene-sets value or decreasing --sigma2" % frac_opposite)
            printed_warning_swing = False

        return _return_inner_betas_result(betas_trace_fh, (avg_betas_m, avg_postp_m))

    #store Y value
    #Y is whitened if Y_corr_m is not null
    def _set_Y(self, Y, Y_for_regression=None, Y_exomes=None, Y_positive_controls=None, Y_case_counts=None, Y_corr_m=None, store_corr_sparse=False, skip_V=False, skip_scale_factors=False, min_correlation=0):
        log("Setting Y", TRACE)
        self.last_X_block = None
        self.y_state = pegs_set_runtime_y_from_inputs(
            runtime=self,
            Y=Y,
            Y_for_regression=Y_for_regression,
            Y_exomes=Y_exomes,
            Y_positive_controls=Y_positive_controls,
            Y_case_counts=Y_case_counts,
            Y_corr_m=Y_corr_m,
            store_corr_sparse=store_corr_sparse,
            min_correlation=min_correlation,
        )

    def _get_y_corr_cholesky(self, Y_corr_m):
        return pegs_compute_banded_y_corr_cholesky(Y_corr_m, diag_add=0.05)

    def _whiten(self, matrix, corr_cholesky, whiten=True, full_whiten=False):
        return pegs_whiten_matrix_with_banded_cholesky(
            matrix,
            corr_cholesky,
            whiten=whiten,
            full_whiten=full_whiten,
        )

    def _get_num_X_blocks(self, X_orig, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size
        return pegs_get_num_X_blocks(X_orig, batch_size)

    def _get_X_size_mb(self, X_orig=None):
        if X_orig is None:
            X_orig = self.X_orig
        return (self.X_orig.data.nbytes + self.X_orig.indptr.nbytes + self.X_orig.indices.nbytes) / 1024 / 1024

    def _get_X_blocks_internal(self, X_orig, y_corr_cholesky, whiten=True, full_whiten=False, start_batch=0, mean_shifts=None, scale_factors=None):
        consider_cache = (
            X_orig is self.X_orig
            and self._get_num_X_blocks(X_orig) == 1
            and mean_shifts is None
            and scale_factors is None
        )
        cache_state = {"last_X_block": self.last_X_block}
        for X_b, begin, end, batch in pegs_iterate_X_blocks_internal(
            X_orig,
            y_corr_cholesky,
            batch_size=self.batch_size,
            log_fn=log,
            trace_level=TRACE,
            is_missing_x=(X_orig is self.X_orig_missing_gene_sets),
            consider_cache=consider_cache,
            cache_state=cache_state,
            whiten_fn=self._whiten,
            whiten=whiten,
            full_whiten=full_whiten,
            start_batch=start_batch,
            mean_shifts=mean_shifts,
            scale_factors=scale_factors,
        ):
            self.last_X_block = cache_state.get("last_X_block")
            yield (X_b, begin, end, batch)
        self.last_X_block = cache_state.get("last_X_block")

    def _get_fraction_non_missing(self):
        if self.gene_sets_missing is not None and self.gene_sets is not None:
            fraction_non_missing = float(len(self.gene_sets)) / float(len(self.gene_sets_missing) + len(self.gene_sets))        
        else:
            fraction_non_missing = 1
        return fraction_non_missing
    
    def _calc_X_shift_scale(self, X, y_corr_cholesky=None):
        return pegs_calc_X_shift_scale(
            X,
            y_corr_cholesky=y_corr_cholesky,
            get_X_blocks_internal_fn=self._get_X_blocks_internal,
            calc_shift_scale_fn=self._calc_shift_scale,
        )

    def _calc_shift_scale(self, X_b):
        return pegs_calc_shift_scale_for_dense_block(X_b)

    #store a (possibly unnormalized) X matrix
    #the passed in X should be a sparse matrix, with 0/1 values
    #does normalization
    def _set_X(self, X_orig, genes, gene_sets, skip_V=False, skip_scale_factors=False, skip_N=True):
        pegs_set_runtime_x_from_inputs(
            self,
            X_orig,
            genes,
            gene_sets,
            skip_scale_factors=skip_scale_factors,
            skip_N=skip_N,
            reread_gene_phewas_bfs_fn=_reread_gene_phewas_bfs,
            get_col_sums_fn=self.get_col_sums,
            set_scale_factors_fn=self._set_scale_factors,
            log_fn=log,
            trace_level=TRACE,
            bail_fn=bail,
        )

    def _set_scale_factors(self):

        log("Calculating scale factors and mean shifts", TRACE)
        (self.mean_shifts, self.scale_factors) = self._calc_X_shift_scale(self.X_orig, self.y_corr_cholesky)

        #flag to indicate whether these scale factors correspond to X_orig or the (implicit) whitened version
        if self.y_corr_cholesky is not None:
            self.scale_is_for_whitened = True
        else:
            self.scale_is_for_whitened = False

    def _get_V(self):
        if self.X_orig is not None:
            log("Calculating internal V", TRACE)
            return self._calculate_V()
        else:
            return None

    def _calculate_V(self, X_orig=None, y_corr_cholesky=None, mean_shifts=None, scale_factors=None):
        if X_orig is None:
            X_orig = self.X_orig
        if mean_shifts is None:
            mean_shifts = self.mean_shifts
        if scale_factors is None:
            scale_factors = self.scale_factors
        if y_corr_cholesky is None:
            y_corr_cholesky = self.y_corr_cholesky
        return self._calculate_V_internal(X_orig, y_corr_cholesky, mean_shifts, scale_factors)

    def _calculate_V_internal(self, X_orig, y_corr_cholesky, mean_shifts, scale_factors, y_corr_sparse=None):
        log("Calculating V for X with dimensions %d x %d" % (X_orig.shape[0], X_orig.shape[1]), TRACE)
        return pegs_calculate_V_internal(
            X_orig,
            y_corr_cholesky,
            mean_shifts,
            scale_factors,
            y_corr_sparse=y_corr_sparse,
            get_num_X_blocks_fn=self._get_num_X_blocks,
            get_X_blocks_internal_fn=self._get_X_blocks_internal,
            compute_V_fn=self._compute_V,
        )

    #calculate V between X_orig and X_orig2
    #X_orig2 can be dense or sparse, but if it is sparse than X_orig must also be sparse
    def _compute_V(self, X_orig, mean_shifts, scale_factors, rows = None, X_orig2 = None, mean_shifts2 = None, scale_factors2 = None, gene_weights=None):
        if X_orig2 is None:
            X_orig2 = X_orig
        if mean_shifts2 is None:
            mean_shifts2 = mean_shifts
        if scale_factors2 is None:
            scale_factors2 = scale_factors
        if rows is None:
            if type(X_orig) is np.ndarray or type(X_orig2) is np.ndarray:
                if gene_weights is not None:
                    dot_product = (X_orig.T.multiply(gene_weights)).dot(X_orig2)
                else:
                    dot_product = X_orig.T.dot(X_orig2)                    
            else:
                if gene_weights is not None:
                    dot_product = (X_orig.T.multiply(gene_weights)).dot(X_orig2).toarray().astype(float)
                else:
                    dot_product = X_orig.T.dot(X_orig2).toarray().astype(float)
        else:
            if type(X_orig) is np.ndarray or type(X_orig2) is np.ndarray:
                if gene_weights is not None:
                    dot_product = (X_orig[:,rows].T.multiply(gene_weights)).dot(X_orig2)
                else:
                    dot_product = X_orig[:,rows].T.dot(X_orig2)
            else:
                if gene_weights is not None:
                    dot_product = (X_orig[:,rows].T.multiply(gene_weights)).dot(X_orig2).toarray().astype(float)
                else:
                    dot_product = X_orig[:,rows].T.dot(X_orig2).toarray().astype(float)                    
            mean_shifts = mean_shifts[rows]
            scale_factors = scale_factors[rows]

        return (dot_product/X_orig.shape[0] - np.outer(mean_shifts, mean_shifts2)) / (np.outer(scale_factors, scale_factors2) + 1e-10)
            

    #by default, find batches of uncorrelated genes (for use in gibbs)
    #option to find batches of correlated (pass in batch size)
    #this is a greedy addition method
    #if have sort_values, will greedily add from lowest value to higher value
    def _compute_gene_set_batches(self, V=None, X_orig=None, mean_shifts=None, scale_factors=None, use_sum=True, max_allowed_batch_correlation=None, find_correlated_instead=None, sort_values=None, resort_as_added=False, stop_at=None, tag="gene sets"):
        gene_set_masks = []

        if max_allowed_batch_correlation is None:
            if use_sum:
                max_allowed_batch_correlation = 0.5
            else:
                max_allowed_batch_correlation = 0.1

        if find_correlated_instead is not None:
            if find_correlated_instead < 1:
                bail("Need batch size of at least 1")

        if use_sum:
            combo_fn = np.sum
        else:
            combo_fn = np.max

        use_X = False
        if V is not None and len(V.shape) == 3:
            num_gene_sets = V.shape[1]
            not_included_gene_sets = np.full((V.shape[0], num_gene_sets), True)
        elif V is not None:
            num_gene_sets = V.shape[0]
            not_included_gene_sets = np.full(num_gene_sets, True)
        else:
            assert(mean_shifts.shape == scale_factors.shape)
            if len(mean_shifts.shape) > 1:
                if mean_shifts.shape[0] == 1:
                    mean_shifts = np.squeeze(mean_shifts, axis=0)
                    scale_factors = np.squeeze(scale_factors, axis=0)
                elif np.all(np.isclose(np.var(mean_shifts, axis=0), 0)):
                    mean_shifts = np.mean(mean_shifts, axis=0)
                    scale_factors = np.mean(scale_factors, axis=0)
                else:
                    bail("Error: can't have different mean shifts across chains")
            if X_orig is None or mean_shifts is None or scale_factors is None:
                bail("Need X_orig or V for this operation")
            num_gene_sets = X_orig.shape[1]
            not_included_gene_sets = np.full(num_gene_sets, True)
            use_X = True

        log("Batching %d %s..." % (num_gene_sets, tag), INFO)
        if use_X:
            log("Using low memory mode", DEBUG)

        indices = np.array(range(num_gene_sets))

        if sort_values is None:
            sort_values = indices
            resort_as_added = False
        orig_sort_values = sort_values
        if resort_as_added:
            sort_values = sort_values.copy()

        total_added = 0

        cur_V = None
        mask_used_for_V = None

        while np.any(not_included_gene_sets):

            if V is not None and len(V.shape) == 3:
                #batches if multiple_V

                current_mask = np.full((V.shape[0], num_gene_sets), False)
                #set the first gene set in each row to True
                for c in range(V.shape[0]):

                    sorted_remaining_indices = sorted(indices[not_included_gene_sets[c,:]], key=lambda k: sort_values[k])
                    #seed with the first gene not already included
                    if len(sorted_remaining_indices) == 0:
                        continue

                    first_gene_set = sorted_remaining_indices[0]
                    current_mask[c,first_gene_set] = True
                    not_included_gene_sets[c,first_gene_set] = False
                    sorted_remaining_indices = sorted_remaining_indices[1:]

                    if find_correlated_instead:
                        #WARNING: THIS HAS NOT BEEN TESTED
                        #sort by decreasing V
                        index_map = np.where(not_included_gene_sets[c,:])[0]
                        ordered_indices = index_map[np.argsort(-V[c,first_gene_set,:])[not_included_gene_sets[c,:]]]
                        indices_to_add = ordered_indices[:find_correlated_instead]
                        current_mask[c,indices_to_add] = True
                    else:
                        for i in sorted_remaining_indices:
                            if combo_fn(V[c,i,current_mask[c,:]]) < max_allowed_batch_correlation:
                                current_mask[c,i] = True
                                not_included_gene_sets[c,i] = False
            else:
                dont_match = orig_sort_values != sort_values
                sorted_remaining_indices = sorted(indices[not_included_gene_sets], key=lambda k: sort_values[k])
                #batches if one V
                current_mask = np.full(num_gene_sets, False)
                #seed with the first gene not already included
                first_gene_set = sorted_remaining_indices[0]
                current_mask[first_gene_set] = True
                not_included_gene_sets[first_gene_set] = False
                sorted_remaining_indices = sorted_remaining_indices[1:]
                if V is not None:
                    if find_correlated_instead:
                        #sort by decreasing V
                        index_map = np.where(not_included_gene_sets)[0]
                        ordered_indices = index_map[np.argsort(-V[first_gene_set,not_included_gene_sets])]
                        #map these to the original ones
                        indices_to_add = ordered_indices[:find_correlated_instead]
                        current_mask[indices_to_add] = True
                        not_included_gene_sets[indices_to_add] = False
                    else:
                        for i in sorted_remaining_indices:
                            if combo_fn(V[i,current_mask]) < max_allowed_batch_correlation:
                                current_mask[i] = True
                                not_included_gene_sets[i] = False

                        if resort_as_added:
                            #adjust the sort values by first multiplying them by the current V
                            #we are only going adjust the ones that were just considered
                            #this will cause us to potentially include gene sets that have not been considered yet, because we are not downweighting them as we should
                            #but this is okay; our goal is to make sure we consider everything (and not just the current batch we are looking at)
                            sort_values -= V[:,current_mask].dot(sort_values)

                else:
                    assert(scale_factors.shape == mean_shifts.shape)

                    if find_correlated_instead:
                        cur_V = self._compute_V(X_orig[:,first_gene_set], mean_shifts[first_gene_set], scale_factors[first_gene_set], X_orig2=X_orig[:,not_included_gene_sets], mean_shifts2=mean_shifts[not_included_gene_sets], scale_factors2=scale_factors[not_included_gene_sets])
                        #sort by decreasing V
                        index_map = np.where(not_included_gene_sets)[0]
                        ordered_indices = index_map[np.argsort(-cur_V[0,:])]
                        indices_to_add = ordered_indices[:find_correlated_instead]
                        current_mask[indices_to_add] = True
                        not_included_gene_sets[indices_to_add] = False
                    else:
                        #cap out at batch_size gene sets to avoid memory of making whole V; this may reduce the batch size relative to optimal
                        #also, only add those not in mask already (since we are searching only these in V)
                        max_to_add = self.batch_size
                        V_to_generate_mask = not_included_gene_sets.copy()
                        if np.sum(V_to_generate_mask) > max_to_add:
                            assert(len(sorted_remaining_indices) == np.sum(not_included_gene_sets))
                            V_to_generate_mask[sort_values > sort_values[sorted_remaining_indices[max_to_add]]] = False

                        V_to_generate_mask[first_gene_set] = True
                        if cur_V is None or self.debug_old_batch:
                            cur_V = self._compute_V(X_orig[:,V_to_generate_mask], mean_shifts[V_to_generate_mask], scale_factors[V_to_generate_mask])
                            working_idx = np.where(V_to_generate_mask)[0]
                        else:
                            new_full_idx   = np.where(V_to_generate_mask)[0]
                            old_full_idx   = idx_used_for_V

                            # sets for convenience
                            new_set        = set(new_full_idx)
                            old_set        = set(old_full_idx)

                            to_add_full    = np.array(sorted(new_set - old_set), dtype=np.int64)

                            keep_mask    = np.isin(old_full_idx, new_full_idx)
                            keep_pos     = np.nonzero(keep_mask)[0]
                            to_keep_full = old_full_idx[keep_pos]

                            cur_V = cur_V[np.ix_(keep_pos, keep_pos)]

                            if to_add_full.size:
                                V_cross = self._compute_V(X_orig[:, to_keep_full], mean_shifts[to_keep_full], scale_factors[to_keep_full], X_orig2 = X_orig[:, to_add_full], mean_shifts2 = mean_shifts[to_add_full], scale_factors2 = scale_factors[to_add_full])
                                V_new = self._compute_V(X_orig[:, to_add_full], mean_shifts[to_add_full], scale_factors[to_add_full])

                                cur_V = np.block([
                                    [cur_V,     V_cross],
                                    [V_cross.T, V_new  ]
                                ])
                            
                            #log("Added %d, kept %d" % (len(to_add_full), len(to_keep_full)), TRACE)
                            working_idx = np.concatenate([to_keep_full, to_add_full])

                        idx_used_for_V = working_idx.copy()

                        assert(np.sum(current_mask[working_idx]) == 1)
                        cur_cor_summary = cur_V[:,current_mask[working_idx]].squeeze(axis=1)

                        sorted_cur_V_indices = sorted(range(cur_V.shape[0]), key=lambda k: sort_values[working_idx[k]])
                        for i in sorted_cur_V_indices:
                            if combo_fn(cur_V[i,current_mask[working_idx]]) < max_allowed_batch_correlation:
                                current_mask[working_idx[i]] = True
                                not_included_gene_sets[working_idx[i]] = False
                                if use_sum:
                                    cur_cor_summary = np.add(cur_cor_summary, cur_V[:,i])
                                else:
                                    cur_cor_summary = np.maximum(cur_cor_summary, cur_V[:,i])

                        if resort_as_added:
                            #adjust the sort values by first multiplying them by the current V
                            #we are only going adjust the ones that were just considered
                            #this will cause us to potentially include gene sets that have not been considered yet, because we are not downweighting them as we should
                            #but this is okay; our goal is to make sure we consider everything (and not just the current batch we are looking at)
                            sort_values[idx_used_for_V] -= cur_V[:,current_mask[idx_used_for_V]].dot(sort_values[idx_used_for_V[current_mask[idx_used_for_V]]])

            gene_set_masks.append(current_mask)
            #log("Batch %d; %d gene sets" % (len(gene_set_masks), sum(current_mask)), TRACE)
            total_added += np.sum(current_mask)
            if stop_at is not None and total_added >= stop_at:
                log("Breaking at %d" % total_added, TRACE)
                break

        denom = 1
        if V is not None and len(V.shape) == 3:
            denom = V.shape[0]

        sizes = [float(np.sum(x)) / denom for x in gene_set_masks]
        log("Batched %d %s into %d batches; size range %d - %d" % (num_gene_sets, tag, len(gene_set_masks), min(sizes) if len(sizes) > 0 else 0, max(sizes)  if len(sizes) > 0 else 0), DEBUG)

        return gene_set_masks

    #sort the genes in the matrices
    #does not alter genes already subseet
    def _sort_genes(self, sorted_gene_indices, skip_V=False, skip_scale_factors=False):

        log("Sorting genes", TRACE)

        self.genes = [self.genes[i] for i in sorted_gene_indices]
        self.gene_to_ind = pegs_construct_map_to_ind(self.genes)

        index_map = {sorted_gene_indices[i]: i for i in range(len(sorted_gene_indices))}

        if self.X_orig is not None:
            #reset the X matrix and scale factors
            self._set_X(sparse.csc_matrix((self.X_orig.data, [index_map[x] for x in self.X_orig.indices], self.X_orig.indptr), shape=self.X_orig.shape), self.genes, self.gene_sets, skip_V=skip_V, skip_scale_factors=skip_scale_factors, skip_N=True)

        if self.X_orig_missing_gene_sets is not None:
            #if we've already removed gene sets, then we need remove the genes from them too
            
            self.X_orig_missing_gene_sets = sparse.csc_matrix((self.X_orig_missing_gene_sets.data, [index_map[x] for x in self.X_orig_missing_gene_sets.indices], self.X_orig_missing_gene_sets.indptr), shape=self.X_orig_missing_gene_sets.shape)
            #need to recompute these
            (self.mean_shifts_missing, self.scale_factors_missing) = self._calc_X_shift_scale(self.X_orig_missing_gene_sets)

        if self.huge_signal_bfs is not None:
            #reset the X matrix and scale factors
            self.huge_signal_bfs = sparse.csc_matrix((self.huge_signal_bfs.data, [index_map[x] for x in self.huge_signal_bfs.indices], self.huge_signal_bfs.indptr), shape=self.huge_signal_bfs.shape)

        if self.huge_signal_bfs_for_regression is not None:
            #reset the X matrix and scale factors
            self.huge_signal_bfs_for_regression = sparse.csc_matrix((self.huge_signal_bfs_for_regression.data, [index_map[x] for x in self.huge_signal_bfs_for_regression.indices], self.huge_signal_bfs_for_regression.indptr), shape=self.huge_signal_bfs_for_regression.shape)

        index_map_rev = {i: sorted_gene_indices[i] for i in range(len(sorted_gene_indices))}

        if self.gene_covariates is not None:
            self.gene_covariates = self.gene_covariates[[index_map_rev[x] for x in range(self.gene_covariates.shape[0])],:]
            self.gene_covariate_zs = self.gene_covariate_zs[[index_map_rev[x] for x in range(self.gene_covariate_zs.shape[0])],:]

        if self.gene_covariate_adjustments is not None:
            self.gene_covariate_adjustments = self.gene_covariate_adjustments[[index_map_rev[x] for x in range(self.gene_covariate_adjustments.shape[0])]]

        if self.gene_covariates_mask is not None:
            self.gene_covariates_mask = self.gene_covariates_mask[[index_map_rev[x] for x in range(self.gene_covariates_mask.shape[0])]]

        if self.gene_pheno_combined_prior_Ys is not None or self.gene_pheno_Y is not None or self.gene_pheno_priors is not None:
            if self.gene_pheno_combined_prior_Ys is not None:
                self.gene_pheno_combined_prior_Ys = self.gene_pheno_combined_prior_Ys[[index_map_rev[x] for x in range(self.gene_pheno_combined_prior_Ys.shape[0])],:]
            if self.gene_pheno_Y is not None:
                self.gene_pheno_Y = self.gene_pheno_Y[[index_map_rev[x] for x in range(self.gene_pheno_Y.shape[0])],:]
            if self.gene_pheno_priors is not None:
                self.gene_pheno_priors = self.gene_pheno_priors[[index_map_rev[x] for x in range(self.gene_pheno_priors.shape[0])],:]

        if self.gene_N is not None:
            self.gene_N = self.gene_N[sorted_gene_indices]
        if self.gene_ignored_N is not None:
            self.gene_ignored_N = self.gene_ignored_N[sorted_gene_indices]

        for x in [self.Y, self.Y_r_hat, self.Y_mcse, self.Y_for_regression, self.Y_uncorrected, self.Y_exomes, self.Y_positive_controls, self.Y_case_counts, self.priors, self.priors_r_hat, self.priors_mcse, self.priors_adj, self.combined_prior_Ys, self.combined_prior_Ys_r_hat, self.combined_prior_Ys_mcse, self.combined_prior_Ys_adj, self.combined_Ds, self.Y_orig, self.Y_for_regression_orig, self.priors_orig, self.priors_adj_orig]:
            if x is not None:
                x[:] = np.array([x[i] for i in sorted_gene_indices])


    def _prune_gene_sets(self, prune_value, prune_deterministically=False, max_size=5000, keep_missing=False, ignore_missing=False, skip_V=False, X_orig=None, gene_sets=None, rank_vector=None, do_internal_pruning=True, gene_weights=None):

        if gene_weights is not None:
            gene_weights = copy.copy(gene_weights)
            gene_weights[gene_weights < 0] = 0

        if X_orig is None and gene_weights is None:
            X_orig = self.X_orig
            mean_shifts = self.mean_shifts
            scale_factors = self.scale_factors
        else:
            if X_orig is None:
                X_orig = self.X_orig
            if gene_weights is None:
                (mean_shifts, scale_factors) = self._calc_X_shift_scale(X_orig)
            else:
                (mean_shifts, scale_factors) = self._calc_X_shift_scale(X_orig.T.multiply(np.sqrt(gene_weights)).T)


        name = ""
        if gene_sets is None:
            gene_sets = self.gene_sets
            name = " gene sets"
        if rank_vector is None:
            rank_vector = self.p_values

        if gene_sets is None or len(gene_sets) == 0:
            return
        if X_orig is None:
            return
        if prune_value > 1:
            return
        keep_mask = np.array([False] * len(gene_sets))

        if len(keep_mask) == 1:
            return keep_mask
        
        log("Pruning%s at %.3g..." % (name, prune_value), DEBUG)

        remove_gene_sets = set()

        #keep total to batch_size ** 2

        batch1_inds = []
        batch2_inds = []

        
        if self.debug_old_batch:
            batch_size = int(max_size ** 2 / X_orig.shape[1])
            num_batches = int(X_orig.shape[1] / batch_size) + 1

            for batch in range(num_batches):
                begin = batch * batch_size
                end = (batch + 1) * batch_size
                if end > X_orig.shape[1]:
                    end = X_orig.shape[1]
            
                batch1_inds.append(np.array(list(range(begin, end))))
                batch2_inds.append(np.array(list(range(X_orig.shape[1]))))
        else:

            max_size_gb = 1
            max_mem_bytes = 1024 * 1024 * 1024 * max_size_gb
            min_num_bins = 20

            X_binary = X_orig.astype(bool)
            num_genes, num_gene_sets = X_binary.shape
            nnz = X_binary.sum(axis=0).A1

            #we are going for each bin to aggregate all of the gene sets that could match its bin into it
            #ensure that even if all gene sets can match, the full V matrix will fit in memory
            max_num_gene_sets_per_bin = int((max_mem_bytes / 8) / num_gene_sets)

            n_bins = max(int(num_gene_sets / max_num_gene_sets_per_bin + 1), min_num_bins)

            mu = nnz / float(num_genes)

            q_nnz = np.linspace(0, 1, n_bins + 1)
            nnz_edges = np.unique(np.quantile(nnz, q_nnz))

            n_bins = len(nnz_edges) - 1

            bin_id = np.clip(np.searchsorted(nnz_edges, nnz, side="right") - 1,0, n_bins - 1)
            idx_by_bin = [np.where(bin_id == b)[0] for b in range(n_bins)]

            mu_max = np.zeros(n_bins)
            mu_min = np.full(n_bins, np.inf)

            # scan once over all sets to fill in extrema
            for i in range(num_gene_sets):
                b            = bin_id[i]
                mu_max[b]    = max(mu_max[b], mu[i])
                mu_min[b]    = min(mu_min[b], mu[i])

            sigma_max = mu_max * (1 - mu_max) + 1e-10
            sigma_min = mu_min * (1 - mu_min) + 1e-10

            must = 0
            can = 0
            for p in range(n_bins):
                all_idx = np.array([], dtype=int)
                for q in range(p,n_bins):
                    #assert(mu_max[p] <= mu_min[q])
                    cor_max = (mu_max[p] - mu_max[p] * mu_min[q]) / np.sqrt(sigma_max[p] * sigma_min[q])
                    can += 1
                    if cor_max > prune_value:
                        must += 1
                        all_idx = np.append(all_idx, idx_by_bin[q])
                if len(all_idx) > 0:
                    batch1_inds.append(idx_by_bin[p])
                    batch2_inds.append(all_idx)

            log("Looking at %d of %d blocks" % (must, can))


        log("Pruning in %d batches" % len(batch1_inds), TRACE)

        for i in range(len(batch1_inds)):

            X_b1  = X_orig[:,batch1_inds[i]]

            log("Constructing matrix of dimensions %d x %d" % (len(batch1_inds[i]), len(batch2_inds[i])))

            V_block = self._compute_V(X_b1, mean_shifts[batch1_inds[i]], scale_factors[batch1_inds[i]], X_orig2=X_orig[:,batch2_inds[i]], mean_shifts2=mean_shifts[batch2_inds[i]], scale_factors2=scale_factors[batch2_inds[i]], gene_weights=gene_weights)

            if rank_vector is not None and False and not prune_deterministically:
                gene_set_key = lambda i: rank_vector[i]
            else:
                gene_set_key = lambda i: np.abs(X_b1[:,i]).sum(axis=0)

            for gene_set_ind in sorted(range(len(batch1_inds[i])), key=gene_set_key):
                absolute_ind = batch1_inds[i][gene_set_ind]
                if absolute_ind in remove_gene_sets:
                    continue
                
                keep_mask[absolute_ind] = True

                rem_local_inds = np.where(np.abs(V_block[gene_set_ind,:]) > prune_value)[0]
                rem_absolute_inds = batch2_inds[i][rem_local_inds]

                #if len(rem_absolute_inds) > 1:
                #    for r in rem_absolute_inds:
                #        print("Removed %s due to %s" % (self.gene_sets[r], self.gene_sets[absolute_ind]))

                remove_gene_sets.update(rem_absolute_inds)

        if np.sum(~keep_mask) > 0:
            if X_orig is self.X_orig and do_internal_pruning:
                self.subset_gene_sets(keep_mask, keep_missing=keep_missing, ignore_missing=ignore_missing, skip_V=skip_V)
                log("Pruning at %.3g resulted in %d%s (of original %d)" % (prune_value, len(self.gene_sets), name, len(keep_mask)))


        return keep_mask

    def _subset_genes(self, gene_mask, skip_V=False, overwrite_missing=False, skip_scale_factors=False, skip_Y=False):

        if not overwrite_missing and sum(np.logical_not(gene_mask)) == 0:
            return
       
        log("Subsetting genes", TRACE)

        if overwrite_missing:
            self.genes_missing = None
            self.priors_missing = None
            self.gene_N_missing = None
            self.gene_ignored_N_missing = None
            self.X_orig_missing_genes = None
            self.X_orig_missing_genes_missing_gene_sets = None

        self.genes_missing = (self.genes_missing if self.genes_missing is not None else []) + [self.genes[i] for i in range(len(self.genes)) if not gene_mask[i]]

        self.gene_missing_to_ind = pegs_construct_map_to_ind(self.genes_missing)
        
        self.genes = [self.genes[i] for i in range(len(self.genes)) if gene_mask[i]]
        self.gene_to_ind = pegs_construct_map_to_ind(self.genes)

        remove_mask = np.logical_not(gene_mask)

        if self.gene_N is not None:
            self.gene_N_missing = np.concatenate((self.gene_N_missing if self.gene_N_missing is not None else np.array([]), self.gene_N[remove_mask]))
        if self.gene_ignored_N is not None:
            self.gene_ignored_N_missing = np.concatenate((self.gene_ignored_N_missing if self.gene_ignored_N_missing is not None else np.array([]), self.gene_ignored_N[remove_mask]))

        if self.X_orig is not None:
            #store the genes that were removed for later
            X_orig_missing_genes = self.X_orig[remove_mask,:]
            if self.X_orig_missing_genes is not None:
                self.X_orig_missing_genes = sparse.csc_matrix(sparse.vstack([self.X_orig_missing_genes, X_orig_missing_genes]))
            else:
                self.X_orig_missing_genes = X_orig_missing_genes

            #reset the X matrix and scale factors
            self._set_X(self.X_orig[gene_mask,:], self.genes, self.gene_sets, skip_V=skip_V, skip_scale_factors=skip_scale_factors, skip_N=True)
            zero = self.X_orig.sum(axis=0).A1

        if self.X_orig_missing_gene_sets is not None:

            X_orig_missing_genes_missing_gene_sets = self.X_orig_missing_gene_sets[remove_mask,:]
            if self.X_orig_missing_genes_missing_gene_sets is not None:
                self.X_orig_missing_genes_missing_gene_sets = sparse.csc_matrix(sparse.vstack([self.X_orig_missing_genes_missing_gene_sets, X_orig_missing_genes_missing_gene_sets]))
            else:
                self.X_orig_missing_genes_missing_gene_sets = X_orig_missing_genes_missing_gene_sets

            #if we've already removed gene sets, then we need remove the genes from them too
            self.X_orig_missing_gene_sets = self.X_orig_missing_gene_sets[gene_mask,:]
            #need to recompute these
            (self.mean_shifts_missing, self.scale_factors_missing) = self._calc_X_shift_scale(self.X_orig_missing_gene_sets, self.y_corr_cholesky)

        if self.gene_N is not None:
            self.gene_N = self.gene_N[gene_mask]
        if self.gene_ignored_N is not None:
            self.gene_ignored_N = self.gene_ignored_N[gene_mask]


        if not skip_Y:
            if self.Y is not None:
                self._set_Y(self.Y[gene_mask], self.Y_for_regression[gene_mask] if self.Y_for_regression is not None else None, self.Y_exomes[gene_mask] if self.Y_exomes is not None else None, self.Y_positive_controls[gene_mask] if self.Y_positive_controls is not None else None, self.Y_case_counts[gene_mask] if self.Y_case_counts is not None else None, Y_corr_m=self.y_corr[:,gene_mask] if self.y_corr is not None else None, store_corr_sparse=self.y_corr_sparse is not None, skip_V=skip_V)

            if self.Y_uncorrected is not None:
                self.Y_uncorrected = self.Y_uncorrected[gene_mask]

            if self.huge_signal_bfs is not None:
                self.huge_signal_bfs = self.huge_signal_bfs[gene_mask,:]
            if self.huge_signal_bfs_for_regression is not None:
                self.huge_signal_bfs_for_regression = self.huge_signal_bfs_for_regression[gene_mask,:]

            if self.gene_covariates is not None:
                self.gene_covariates = self.gene_covariates[gene_mask,:]
            if self.gene_covariate_zs is not None:
                self.gene_covariate_zs = self.gene_covariate_zs[gene_mask,:]
            if self.gene_covariate_adjustments is not None:
                self.gene_covariate_adjustments = self.gene_covariate_adjustments[gene_mask]
            if self.gene_covariates_mask is not None:
                self.gene_covariates_mask = self.gene_covariates_mask[gene_mask]


            if self.priors is not None:
                self.priors_missing = (self.priors_missing if self.priors_missing is not None else []) + [self.priors[i] for i in range(len(self.priors)) if not gene_mask[i]]
                self.priors = self.priors[gene_mask]
            if self.priors_r_hat is not None:
                self.priors_r_hat = self.priors_r_hat[gene_mask]
            if self.priors_mcse is not None:
                self.priors_mcse = self.priors_mcse[gene_mask]
            
            if self.priors_adj is not None:
                self.priors_adj = self.priors_adj[gene_mask]
            if self.combined_prior_Ys is not None:
                self.combined_prior_Ys = self.combined_prior_Ys[gene_mask]
            if self.combined_prior_Ys_r_hat is not None:
                self.combined_prior_Ys_r_hat = self.combined_prior_Ys_r_hat[gene_mask]
            if self.combined_prior_Ys_mcse is not None:
                self.combined_prior_Ys_mcse = self.combined_prior_Ys_mcse[gene_mask]
            if self.combined_prior_Ys_adj is not None:
                self.combined_prior_Ys_adj = self.combined_prior_Ys_adj[gene_mask]
            if self.combined_Ds is not None:
                self.combined_Ds = self.combined_Ds[gene_mask]
            if self.Y_r_hat is not None:
                self.Y_r_hat = self.Y_r_hat[gene_mask]
            if self.Y_mcse is not None:
                self.Y_mcse = self.Y_mcse[gene_mask]
            if self.Y_orig is not None:
                self.Y_orig = self.Y_orig[gene_mask]
            if self.Y_for_regression_orig is not None:
                self.Y_for_regression_orig = self.Y_for_regression_orig[gene_mask]
            if self.priors_orig is not None:
                self.priors_missing_orig = (self.priors_missing_orig if self.priors_missing_orig is not None else []) + [self.priors_orig[i] for i in range(len(self.priors_orig)) if not gene_mask[i]]
                self.priors_orig = self.priors_orig[gene_mask]
            if self.priors_adj_orig is not None:
                self.priors_adj_missing_orig = (self.priors_adj_missing_orig if self.priors_adj_missing_orig is not None else []) + [self.priors_adj_orig[i] for i in range(len(self.priors_adj_orig)) if not gene_mask[i]]
                self.priors_adj_orig = self.priors_adj_orig[gene_mask]

            if self.gene_pheno_combined_prior_Ys is not None:
                self.gene_pheno_combined_prior_Ys = self.gene_pheno_combined_prior_Ys[gene_mask,:]
            if self.gene_pheno_Y is not None:
                self.gene_pheno_Y = self.gene_pheno_Y[gene_mask,:]
            if self.gene_pheno_priors is not None:
                self.gene_pheno_priors = self.gene_pheno_priors[gene_mask,:]


        #    if x is not None:
        #        x[:] = np.concatenate((x[gene_mask], x[~gene_mask]))

    #subset the current state of the class to a reduced set of gene sets
    def subset_gene_sets(self, subset_mask, keep_missing=True, ignore_missing=False, skip_V=False, skip_scale_factors=False):

        if subset_mask is None or np.sum(~subset_mask) == 0:
            return
        if self.gene_sets is None:
            return

        log("Subsetting gene sets", TRACE)

        remove_mask = np.logical_not(subset_mask)

        if ignore_missing:
            keep_missing = False

            if self.gene_sets is not None:
                if self.gene_sets_ignored is None:
                    self.gene_sets_ignored = []
                self.gene_sets_ignored = self.gene_sets_ignored + [self.gene_sets[i] for i in range(len(self.gene_sets)) if remove_mask[i]]

            if self.gene_set_labels is not None:
                if self.gene_set_labels_ignored is None:
                    self.gene_set_labels_ignored = []
                self.gene_set_labels_ignored = np.append(self.gene_set_labels_ignored, self.gene_set_labels[remove_mask])

            if self.scale_factors is not None:
                if self.scale_factors_ignored is None:
                    self.scale_factors_ignored = np.array([])
                self.scale_factors_ignored = np.append(self.scale_factors_ignored, self.scale_factors[remove_mask])

            if self.mean_shifts is not None:
                if self.mean_shifts_ignored is None:
                    self.mean_shifts_ignored = np.array([])
                self.mean_shifts_ignored = np.append(self.mean_shifts_ignored, self.mean_shifts[remove_mask])

            if self.beta_tildes is not None:
                if self.beta_tildes_ignored is None:
                    self.beta_tildes_ignored = np.array([])
                self.beta_tildes_ignored = np.append(self.beta_tildes_ignored, self.beta_tildes[remove_mask])

            if self.p_values is not None:
                if self.p_values_ignored is None:
                    self.p_values_ignored = np.array([])
                self.p_values_ignored = np.append(self.p_values_ignored, self.p_values[remove_mask])

            if self.ses is not None:
                if self.ses_ignored is None:
                    self.ses_ignored = np.array([])
                self.ses_ignored = np.append(self.ses_ignored, self.ses[remove_mask])

            if self.z_scores is not None:
                if self.z_scores_ignored is None:
                    self.z_scores_ignored = np.array([])
                self.z_scores_ignored = np.append(self.z_scores_ignored, self.z_scores[remove_mask])

            if self.se_inflation_factors is not None:
                if self.se_inflation_factors_ignored is None:
                    self.se_inflation_factors_ignored = np.array([])
                self.se_inflation_factors_ignored = np.append(self.se_inflation_factors_ignored, self.se_inflation_factors[remove_mask])

            if self.gene_covariates is not None:
                if self.total_qc_metrics_ignored is None:
                    self.total_qc_metrics_ignored = self.total_qc_metrics[remove_mask,:]
                    self.mean_qc_metrics_ignored = self.mean_qc_metrics[remove_mask]
                else:
                    self.total_qc_metrics_ignored = np.vstack((self.total_qc_metrics_ignored, self.total_qc_metrics[remove_mask,:]))
                    self.mean_qc_metrics_ignored = np.append(self.mean_qc_metrics_ignored, self.mean_qc_metrics[remove_mask])

            #need to record how many ignored
            if self.X_orig is not None:
                if self.col_sums_ignored is None:
                    self.col_sums_ignored = np.array([])
                self.col_sums_ignored = np.append(self.col_sums_ignored, self.get_col_sums(self.X_orig[:,remove_mask]))

                gene_ignored_N = self.get_col_sums(self.X_orig[:,remove_mask], axis=1)
                if self.gene_ignored_N is None:
                    self.gene_ignored_N = gene_ignored_N
                else:
                    self.gene_ignored_N += gene_ignored_N
                if self.gene_N is not None:
                    self.gene_N -= gene_ignored_N

        elif keep_missing:
            self.gene_sets_missing = [self.gene_sets[i] for i in range(len(self.gene_sets)) if remove_mask[i]]

            if self.beta_tildes is not None:
                self.beta_tildes_missing = self.beta_tildes[remove_mask]
            if self.p_values is not None:
                self.p_values_missing = self.p_values[remove_mask]
            if self.z_scores is not None:
                self.z_scores_missing = self.z_scores[remove_mask]
            if self.ses is not None:
                self.ses_missing = self.ses[remove_mask]
            if self.se_inflation_factors is not None:
                self.se_inflation_factors_missing = self.se_inflation_factors[remove_mask]
            if self.beta_tildes_orig is not None:
                self.beta_tildes_missing_orig = self.beta_tildes_orig[remove_mask]
            if self.p_values_orig is not None:
                self.p_values_missing_orig = self.p_values_orig[remove_mask]
            if self.z_scores_orig is not None:
                self.z_scores_missing_orig = self.z_scores_orig[remove_mask]
            if self.ses_orig is not None:
                self.ses_missing_orig = self.ses_orig[remove_mask]

            if self.total_qc_metrics is not None:
                self.total_qc_metrics_missing = self.total_qc_metrics[remove_mask]

            if self.mean_qc_metrics is not None:
                self.mean_qc_metrics_missing = self.mean_qc_metrics[remove_mask]

            if self.betas_uncorrected is not None:
                self.betas_uncorrected_missing = self.betas_uncorrected[remove_mask]
            if self.betas_r_hat is not None:
                self.betas_r_hat_missing = self.betas_r_hat[remove_mask]
            if self.betas_mcse is not None:
                self.betas_mcse_missing = self.betas_mcse[remove_mask]
            if self.betas_uncorrected_r_hat is not None:
                self.betas_uncorrected_r_hat_missing = self.betas_uncorrected_r_hat[remove_mask]
            if self.betas_uncorrected_mcse is not None:
                self.betas_uncorrected_mcse_missing = self.betas_uncorrected_mcse[remove_mask]

            if self.betas is not None:
                self.betas_missing = self.betas[remove_mask]
            if self.non_inf_avg_cond_betas is not None:
                self.non_inf_avg_cond_betas_missing = self.non_inf_avg_cond_betas[remove_mask]
            if self.non_inf_avg_postps is not None:
                self.non_inf_avg_postps_missing = self.non_inf_avg_postps[remove_mask]

            if self.betas_orig is not None:
                self.betas_missing_orig = self.betas_orig[remove_mask]
            if self.betas_uncorrected_orig is not None:
                self.betas_uncorrected_missing_orig = self.betas_uncorrected_orig[remove_mask]
            if self.non_inf_avg_cond_betas_orig is not None:
                self.non_inf_avg_cond_betas_missing_orig = self.non_inf_avg_cond_betas_orig[remove_mask]
            if self.non_inf_avg_postps_orig is not None:
                self.non_inf_avg_postps_missing_orig = self.non_inf_avg_postps_orig[remove_mask]

            if self.is_dense_gene_set is not None:
                self.is_dense_gene_set_missing = self.is_dense_gene_set[remove_mask]

            if self.gene_set_batches is not None:
                self.gene_set_batches_missing = self.gene_set_batches[remove_mask]

            if self.gene_set_labels is not None:
                self.gene_set_labels_missing = self.gene_set_labels[remove_mask]

            if self.ps is not None:
                self.ps_missing = self.ps[remove_mask]
            if self.sigma2s is not None:
                self.sigma2s_missing = self.sigma2s[remove_mask]


            if self.X_orig is not None:
                #store the removed gene sets for later
                if keep_missing:
                    self.X_orig_missing_gene_sets = self.X_orig[:,remove_mask]
                    self.mean_shifts_missing = self.mean_shifts[remove_mask]
                    self.scale_factors_missing = self.scale_factors[remove_mask]

        #now do the subsetting to keep

        if self.beta_tildes is not None:
            self.beta_tildes = self.beta_tildes[subset_mask]
        if self.p_values is not None:
            self.p_values = self.p_values[subset_mask]
        if self.z_scores is not None:
            self.z_scores = self.z_scores[subset_mask]
        if self.ses is not None:
            self.ses = self.ses[subset_mask]
        if self.se_inflation_factors is not None:
            self.se_inflation_factors = self.se_inflation_factors[subset_mask]

        if self.beta_tildes_orig is not None:
            self.beta_tildes_orig = self.beta_tildes_orig[subset_mask]
        if self.p_values_orig is not None:
            self.p_values_orig = self.p_values_orig[subset_mask]
        if self.z_scores_orig is not None:
            self.z_scores_orig = self.z_scores_orig[subset_mask]
        if self.ses_orig is not None:
            self.ses_orig = self.ses_orig[subset_mask]


        if self.total_qc_metrics is not None:
            self.total_qc_metrics = self.total_qc_metrics[subset_mask]

        if self.mean_qc_metrics is not None:
            self.mean_qc_metrics = self.mean_qc_metrics[subset_mask]

        if self.betas_uncorrected is not None:
            self.betas_uncorrected = self.betas_uncorrected[subset_mask]
        if self.betas_r_hat is not None:
            self.betas_r_hat = self.betas_r_hat[subset_mask]
        if self.betas_mcse is not None:
            self.betas_mcse = self.betas_mcse[subset_mask]
        if self.betas_uncorrected_r_hat is not None:
            self.betas_uncorrected_r_hat = self.betas_uncorrected_r_hat[subset_mask]
        if self.betas_uncorrected_mcse is not None:
            self.betas_uncorrected_mcse = self.betas_uncorrected_mcse[subset_mask]

        if self.betas is not None:
            self.betas = self.betas[subset_mask]
        if self.non_inf_avg_cond_betas is not None:
            self.non_inf_avg_cond_betas = self.non_inf_avg_cond_betas[subset_mask]
        if self.non_inf_avg_postps is not None:
            self.non_inf_avg_postps = self.non_inf_avg_postps[subset_mask]

        if self.betas_orig is not None:
            self.betas_orig = self.betas_orig[subset_mask]
        if self.betas_uncorrected_orig is not None:
            self.betas_uncorrected_orig = self.betas_uncorrected_orig[subset_mask]
        if self.non_inf_avg_cond_betas_orig is not None:
            self.non_inf_avg_cond_betas_orig = self.non_inf_avg_cond_betas_orig[subset_mask]
        if self.non_inf_avg_postps_orig is not None:
            self.non_inf_avg_postps_orig = self.non_inf_avg_postps_orig[subset_mask]

        if self.is_dense_gene_set is not None:
            self.is_dense_gene_set = self.is_dense_gene_set[subset_mask]

        if self.gene_set_batches is not None:
            self.gene_set_batches = self.gene_set_batches[subset_mask]

        if self.gene_set_labels is not None:
            self.gene_set_labels = self.gene_set_labels[subset_mask]

        if self.ps is not None:
            self.ps = self.ps[subset_mask]
        if self.sigma2s is not None:
            self.sigma2s = self.sigma2s[subset_mask]

        self.gene_sets = list(itertools.compress(self.gene_sets, subset_mask))
        self.gene_set_to_ind = pegs_construct_map_to_ind(self.gene_sets)

        if self.X_phewas_beta is not None:
            self.X_phewas_beta = self.X_phewas_beta[:,subset_mask]                
        if self.X_phewas_beta_uncorrected is not None:
            self.X_phewas_beta_uncorrected = self.X_phewas_beta_uncorrected[:,subset_mask]                

        if self.X_orig is not None:
            #never update V; if it exists it will be updated below
            self._set_X(self.X_orig[:,subset_mask], self.genes, self.gene_sets, skip_V=True, skip_scale_factors=skip_scale_factors, skip_N=True)

        if self.X_orig_missing_genes is not None:
            #if we've already removed genes, then we need to remove the gene sets from them
            if keep_missing:
                self.X_orig_missing_genes_missing_gene_sets = self.X_orig_missing_genes[:,remove_mask]
            self.X_orig_missing_genes = self.X_orig_missing_genes[:,subset_mask]

        #need to update the scale factor for sigma2
        #sigma2 is always relative to just the non missing gene sets
        if self.sigma2 is not None:
            self.set_sigma(self.sigma2, self.sigma_power, sigma2_osc=self.sigma2_osc)
        if self.p is not None:
            self.set_p(self.p)

    def _unsubset_gene_sets(self, skip_V=False, skip_scale_factors=False):
        if self.gene_sets_missing is None or self.X_orig_missing_gene_sets is None:
            return(np.array([True] * len(self.gene_sets)))

        log("Un-subsetting gene sets", TRACE)

        #need to update the scale factor for sigma2
        #sigma2 is always relative to just the non missing gene sets
        fraction_non_missing = self._get_fraction_non_missing()

        subset_mask = np.array([True] * len(self.gene_sets) + [False] * len(self.gene_sets_missing))

        self.gene_sets += self.gene_sets_missing
        self.gene_sets_missing = None
        self.gene_set_to_ind = pegs_construct_map_to_ind(self.gene_sets)

        if self.beta_tildes_missing is not None:
            self.beta_tildes = np.append(self.beta_tildes, self.beta_tildes_missing)
            self.beta_tildes_missing = None
        if self.p_values_missing is not None:
            self.p_values = np.append(self.p_values, self.p_values_missing)
            self.p_values_missing = None
        if self.z_scores_missing is not None:
            self.z_scores = np.append(self.z_scores, self.z_scores_missing)
            self.z_scores_missing = None
        if self.ses_missing is not None:
            self.ses = np.append(self.ses, self.ses_missing)
            self.ses_missing = None
        if self.se_inflation_factors_missing is not None:
            self.se_inflation_factors = np.append(self.se_inflation_factors, self.se_inflation_factors_missing)
            self.se_inflation_factors_missing = None

        if self.total_qc_metrics_missing is not None:
            self.total_qc_metrics = np.vstack((self.total_qc_metrics, self.total_qc_metrics_missing))
            self.total_qc_metrics_missing = None

        if self.mean_qc_metrics_missing is not None:
            self.mean_qc_metrics = np.append(self.mean_qc_metrics, self.mean_qc_metrics_missing)
            self.mean_qc_metrics_missing = None

        if self.beta_tildes_missing_orig is not None:
            self.beta_tildes_orig = np.append(self.beta_tildes_orig, self.beta_tildes_missing_orig)
            self.beta_tildes_missing_orig = None
        if self.p_values_missing_orig is not None:
            self.p_values_orig = np.append(self.p_values_orig, self.p_values_missing_orig)
            self.p_values_missing_orig = None
        if self.z_scores_missing_orig is not None:
            self.z_scores_orig = np.append(self.z_scores_orig, self.z_scores_missing_orig)
            self.z_scores_missing_orig = None
        if self.ses_missing_orig is not None:
            self.ses_orig = np.append(self.ses_orig, self.ses_missing_orig)
            self.ses_missing_orig = None

        if self.betas_uncorrected_missing is not None:
            self.betas_uncorrected = np.append(self.betas_uncorrected, self.betas_uncorrected_missing)
            self.betas_uncorrected_missing = None
        if self.betas_r_hat_missing is not None:
            self.betas_r_hat = np.append(self.betas_r_hat, self.betas_r_hat_missing)
            self.betas_r_hat_missing = None
        if self.betas_mcse_missing is not None:
            self.betas_mcse = np.append(self.betas_mcse, self.betas_mcse_missing)
            self.betas_mcse_missing = None
        if self.betas_uncorrected_r_hat_missing is not None:
            self.betas_uncorrected_r_hat = np.append(self.betas_uncorrected_r_hat, self.betas_uncorrected_r_hat_missing)
            self.betas_uncorrected_r_hat_missing = None
        if self.betas_uncorrected_mcse_missing is not None:
            self.betas_uncorrected_mcse = np.append(self.betas_uncorrected_mcse, self.betas_uncorrected_mcse_missing)
            self.betas_uncorrected_mcse_missing = None

        if self.betas_missing is not None:
            self.betas = np.append(self.betas, self.betas_missing)
            self.betas_missing = None
        if self.non_inf_avg_cond_betas_missing is not None:
            self.non_inf_avg_cond_betas = np.append(self.non_inf_avg_cond_betas, self.non_inf_avg_cond_betas_missing)
            self.non_inf_avg_cond_betas_missing = None
        if self.non_inf_avg_postps_missing is not None:
            self.non_inf_avg_postps = np.append(self.non_inf_avg_postps, self.non_inf_avg_postps_missing)
            self.non_inf_avg_postps_missing = None

        if self.betas_missing_orig is not None:
            self.betas_orig = np.append(self.betas_orig, self.betas_missing_orig)
            self.betas_missing_orig = None
        if self.betas_uncorrected_missing_orig is not None:
            self.betas_uncorrected_orig = np.append(self.betas_uncorrected_orig, self.betas_uncorrected_missing_orig)
            self.betas_uncorrected_missing_orig = None
        if self.non_inf_avg_cond_betas_missing_orig is not None:
            self.non_inf_avg_cond_betas_orig = np.append(self.non_inf_avg_cond_betas_orig, self.non_inf_avg_cond_betas_missing_orig)
            self.non_inf_avg_cond_betas_missing_orig = None
        if self.non_inf_avg_postps_missing_orig is not None:
            self.non_inf_avg_postps_orig = np.append(self.non_inf_avg_postps_orig, self.non_inf_avg_postps_missing_orig)
            self.non_inf_avg_postps_missing_orig = None

        if self.X_orig_missing_gene_sets is not None:
            self.X_orig = sparse.hstack((self.X_orig, self.X_orig_missing_gene_sets), format="csc")
            self.X_orig_missing_gene_sets = None
            self.mean_shifts = np.append(self.mean_shifts, self.mean_shifts_missing)
            self.mean_shifts_missing = None
            self.scale_factors = np.append(self.scale_factors, self.scale_factors_missing)
            self.scale_factors_missing = None
            self.is_dense_gene_set = np.append(self.is_dense_gene_set, self.is_dense_gene_set_missing)
            self.is_dense_gene_set_missing = None
            self.gene_set_batches = np.append(self.gene_set_batches, self.gene_set_batches_missing)
            self.gene_set_batches_missing = None
            self.gene_set_labels = np.append(self.gene_set_labels, self.gene_set_labels_missing)
            self.gene_set_labels_missing = None


            if self.ps is not None:
                self.ps = np.append(self.ps, self.ps_missing)
                self.ps_missing = None
            if self.sigma2s is not None:
                self.sigma2s = np.append(self.sigma2s, self.sigma2s_missing)
                self.sigma2s_missing = None

        self._set_X(self.X_orig, self.genes, self.gene_sets, skip_V=skip_V, skip_scale_factors=skip_scale_factors, skip_N=False)

        if self.X_orig_missing_genes_missing_gene_sets is not None:
            #if we've already removed genes, then we need to remove the gene sets from them
            self.X_orig_missing_genes = sparse.hstack((self.X_orig_missing_genes, self.X_orig_missing_genes_missing_gene_sets), format="csc")
            self.X_orig_missing_genes_missing_gene_sets = None

        return(subset_mask)

_bind_hyperparameter_properties(PigeanState)


# ==========================================================================
# State-agnostic parsing helpers used by both legacy objects and runtime-state.
# ==========================================================================
def _init_gene_locs(runtime_state, gene_loc_file):
    log("Reading --gene-loc-file %s" % gene_loc_file)
    (
        runtime_state.gene_chrom_name_pos,
        runtime_state.gene_to_chrom,
        runtime_state.gene_to_pos,
    ) = pegs_read_loc_file_with_gene_map(
        gene_loc_file,
        gene_label_map=runtime_state.gene_label_map,
        clean_chrom_fn=pegs_clean_chrom_name,
        warn_fn=warn,
        bail_fn=bail,
    )


def _read_gene_map(runtime_state, gene_map_in, gene_map_orig_gene_col=1, gene_map_new_gene_col=2, allow_multi=False):
    runtime_state.gene_label_map = pegs_parse_gene_map_file(
        gene_map_in,
        gene_map_orig_gene_col=gene_map_orig_gene_col,
        gene_map_new_gene_col=gene_map_new_gene_col,
        allow_multi=allow_multi,
        bail_fn=bail,
    )


def _set_const_Y(runtime_state, value):
    const_Y = np.full(len(runtime_state.genes), value)
    runtime_state._set_Y(const_Y, const_Y, None, None, None, skip_V=True, skip_scale_factors=True)

def _read_Y_from_contract(runtime_state, y_read_contract):
    if y_read_contract is None:
        bail("Bug in code: y_read_contract must be non-None")
    if not hasattr(y_read_contract, "to_read_kwargs"):
        bail("Bug in code: y_read_contract must provide to_read_kwargs()")
    return _read_Y(runtime_state, **y_read_contract.to_read_kwargs())


def _run_read_y_stage(runtime_state, **read_kwargs):
    return _read_Y(runtime_state, **read_kwargs)


def _run_read_y_contract_stage(runtime_state, y_read_contract):
    return _read_Y_from_contract(runtime_state, y_read_contract)


def _read_Y(
    runtime_state,
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
    (
        Y1_exomes,
        Y1_positive_controls,
        Y1_case_counts,
        extra_genes_all,
        extra_Y_exomes,
        extra_Y_positive_controls,
        extra_Y_case_counts,
        missing_value_exomes,
        missing_value_positive_controls,
        missing_value_case_counts,
    ) = _read_and_align_auxiliary_y_components(
        runtime_state,
        exomes_in=exomes_in,
        positive_controls_in=positive_controls_in,
        positive_controls_list=positive_controls_list,
        case_counts_in=case_counts_in,
        ctrl_counts_in=ctrl_counts_in,
        gene_loc_file=gene_loc_file,
        hold_out_chrom=hold_out_chrom,
        **kwargs,
    )

    (
        Y1,
        extra_genes,
        extra_Y,
        Y1_for_regression,
        extra_Y_for_regression,
        missing_value,
        gene_combined_map,
        gene_prior_map,
    ) = _read_primary_y_source(
        runtime_state,
        gwas_in=gwas_in,
        huge_statistics_in=huge_statistics_in,
        huge_statistics_out=huge_statistics_out,
        exomes_in=exomes_in,
        positive_controls_in=positive_controls_in,
        positive_controls_list=positive_controls_list,
        case_counts_in=case_counts_in,
        gene_bfs_in=gene_bfs_in,
        hold_out_chrom=hold_out_chrom,
        gene_loc_file=gene_loc_file,
        Y1_exomes=Y1_exomes,
        Y1_positive_controls=Y1_positive_controls,
        Y1_case_counts=Y1_case_counts,
        **kwargs,
    )

    (
        Y,
        Y_for_regression,
        Y_exomes,
        Y_positive_controls,
        Y_case_counts,
        extra_genes,
        extra_Y,
        extra_Y_for_regression,
        extra_Y_exomes,
        extra_Y_positive_controls,
        extra_Y_case_counts,
    ) = _materialize_y_on_gene_universe(
        runtime_state,
        Y1=Y1,
        Y1_for_regression=Y1_for_regression,
        Y1_exomes=Y1_exomes,
        Y1_positive_controls=Y1_positive_controls,
        Y1_case_counts=Y1_case_counts,
        extra_genes=extra_genes,
        extra_Y=extra_Y,
        extra_Y_for_regression=extra_Y_for_regression,
        extra_genes_all=extra_genes_all,
        extra_Y_exomes=extra_Y_exomes,
        extra_Y_positive_controls=extra_Y_positive_controls,
        extra_Y_case_counts=extra_Y_case_counts,
        missing_value=missing_value,
        missing_value_exomes=missing_value_exomes,
        missing_value_positive_controls=missing_value_positive_controls,
        missing_value_case_counts=missing_value_case_counts,
    )

    _finalize_y_vectors_and_expand_x(
        runtime_state,
        Y=Y,
        Y_for_regression=Y_for_regression,
        Y_exomes=Y_exomes,
        Y_positive_controls=Y_positive_controls,
        Y_case_counts=Y_case_counts,
        extra_genes=extra_genes,
        extra_Y=extra_Y,
        extra_Y_for_regression=extra_Y_for_regression,
        extra_Y_exomes=extra_Y_exomes,
        extra_Y_positive_controls=extra_Y_positive_controls,
        extra_Y_case_counts=extra_Y_case_counts,
    )

    _apply_gene_level_maps_after_read_y(
        runtime_state,
        gene_combined_map=gene_combined_map,
        gene_prior_map=gene_prior_map,
    )
    _apply_gene_covariates_and_correct_huge(runtime_state, gene_covs_in=gene_covs_in, **kwargs)


def _apply_gene_level_maps_after_read_y(runtime_state, gene_combined_map=None, gene_prior_map=None):
    if gene_combined_map is not None:
        runtime_state.combined_prior_Ys = copy.copy(runtime_state.Y)
        for i in range(len(runtime_state.genes)):
            if runtime_state.genes[i] in gene_combined_map:
                runtime_state.combined_prior_Ys[i] = gene_combined_map[runtime_state.genes[i]]

    if gene_prior_map is not None:
        runtime_state.priors = np.zeros(len(runtime_state.genes))
        for i in range(len(runtime_state.genes)):
            if runtime_state.genes[i] in gene_prior_map:
                runtime_state.priors[i] = gene_prior_map[runtime_state.genes[i]]


def _finalize_y_vectors_and_expand_x(
    runtime_state,
    Y,
    Y_for_regression,
    Y_exomes,
    Y_positive_controls,
    Y_case_counts,
    extra_genes,
    extra_Y,
    extra_Y_for_regression,
    extra_Y_exomes,
    extra_Y_positive_controls,
    extra_Y_case_counts,
):
    # Y contains all of the genes in runtime_state.genes that have gene statistics.
    # extra_Y contains additional genes not in runtime_state.genes that have gene statistics.
    if len(extra_Y) > 0:
        Y = np.concatenate((Y, extra_Y))
        Y_for_regression = np.concatenate((Y_for_regression, extra_Y_for_regression))
        Y_exomes = np.concatenate((Y_exomes, extra_Y_exomes))
        Y_positive_controls = np.concatenate((Y_positive_controls, extra_Y_positive_controls))
        Y_case_counts = np.concatenate((Y_case_counts, extra_Y_case_counts))

    if runtime_state.X_orig is not None:
        # Use original X because no whitening has taken place yet.
        log("Expanding matrix", TRACE)
        runtime_state._set_X(
            sparse.csc_matrix(
                (runtime_state.X_orig.data, runtime_state.X_orig.indices, runtime_state.X_orig.indptr),
                shape=(runtime_state.X_orig.shape[0] + len(extra_Y), runtime_state.X_orig.shape[1]),
            ),
            runtime_state.genes,
            runtime_state.gene_sets,
            skip_V=True,
            skip_scale_factors=True,
            skip_N=False,
        )

    if runtime_state.genes is not None:
        runtime_state._set_X(runtime_state.X_orig, runtime_state.genes + extra_genes, runtime_state.gene_sets, skip_N=False)

    runtime_state._set_Y(
        Y,
        Y_for_regression,
        Y_exomes,
        Y_positive_controls,
        Y_case_counts,
        skip_V=True,
        skip_scale_factors=True,
    )


def _read_and_align_auxiliary_y_components(
    runtime_state,
    exomes_in=None,
    positive_controls_in=None,
    positive_controls_list=None,
    case_counts_in=None,
    ctrl_counts_in=None,
    gene_loc_file=None,
    hold_out_chrom=None,
    **kwargs,
):
    Y1_exomes = np.array([])
    extra_genes_all = []
    extra_Y_exomes = []

    if exomes_in is not None:
        (Y1_exomes, extra_genes_exomes, extra_Y_exomes) = runtime_state.calculate_huge_scores_exomes(
            exomes_in,
            hold_out_chrom=hold_out_chrom,
            gene_loc_file=gene_loc_file,
            **kwargs
        )
        if runtime_state.genes is None:
            runtime_state._set_X(runtime_state.X_orig, extra_genes_exomes, runtime_state.gene_sets, skip_N=True, skip_V=True)
            # set this temporarily for use in huge
            runtime_state.Y_exomes = extra_Y_exomes
            Y1_exomes = extra_Y_exomes
            extra_genes_all = []
            extra_Y_exomes = np.array([])
        else:
            # extra_genes_exomes and extra_Y_exomes have genes not yet in runtime_state.genes
            extra_genes_all = extra_genes_exomes

    missing_value_exomes = 0
    missing_value_positive_controls = 0
    missing_value_case_counts = 0

    Y1_positive_controls = np.array([])
    extra_Y_positive_controls = []

    if positive_controls_in is not None or positive_controls_list is not None:
        (Y1_positive_controls, extra_genes_positive_controls, extra_Y_positive_controls) = runtime_state.read_positive_controls(
            positive_controls_in,
            positive_controls_list=positive_controls_list,
            hold_out_chrom=hold_out_chrom,
            gene_loc_file=gene_loc_file,
            **kwargs
        )
        if runtime_state.genes is None:
            # this is the first time genes were read in
            assert(len(Y1_exomes) == 0)
            runtime_state._set_X(runtime_state.X_orig, extra_genes_positive_controls, runtime_state.gene_sets, skip_N=True, skip_V=True)
            # set this temporarily for use in huge
            runtime_state.Y_positive_controls = extra_Y_positive_controls
            Y1_positive_controls = extra_Y_positive_controls
            extra_genes_positive_controls = []
            extra_genes_all = extra_genes_positive_controls
            extra_Y_positive_controls = np.array([])
            Y1_exomes = np.zeros(len(Y1_positive_controls))
        else:
            # exomes is already aligned to runtime_state.genes: Y1_exomes matches runtime_state.genes
            # align so genes includes the union of exomes and positive controls
            extra_genes_all, aligned_existing_values, extra_Y_positive_controls = _align_extra_genes_with_new_source(
                existing_extra_genes=extra_genes_all,
                existing_extra_values=[extra_Y_exomes],
                new_source_genes=extra_genes_positive_controls,
                new_source_values=extra_Y_positive_controls,
                existing_missing_values=[missing_value_exomes],
                new_source_missing_value=missing_value_positive_controls,
            )
            extra_Y_exomes = aligned_existing_values[0]
    else:
        Y1_positive_controls = np.zeros(len(Y1_exomes))
        extra_Y_positive_controls = np.zeros(len(extra_Y_positive_controls))

    assert(len(extra_Y_exomes) == len(extra_genes_all))
    assert(len(extra_Y_exomes) == len(extra_Y_positive_controls))
    assert(len(Y1_exomes) == len(Y1_positive_controls))

    Y1_case_counts = np.array([])
    extra_Y_case_counts = []

    if case_counts_in is not None or ctrl_counts_in is not None:
        if case_counts_in is None or ctrl_counts_in is None:
            bail("If specify one of --case-counts-in or --ctrl-counts-in must specify both of them")

        (Y1_case_counts, extra_genes_case_counts, extra_Y_case_counts) = runtime_state.read_count_file(
            case_counts_in,
            ctrl_counts_in,
            hold_out_chrom=hold_out_chrom,
            gene_loc_file=gene_loc_file,
            **kwargs
        )
        if runtime_state.genes is None:
            assert(len(Y1_exomes) == 0)
            assert(len(Y1_positive_controls) == 0)
            runtime_state._set_X(runtime_state.X_orig, extra_genes_case_counts, runtime_state.gene_sets, skip_N=True, skip_V=True)
            # set this temporarily for use in huge
            runtime_state.Y_case_counts = extra_Y_case_counts
            Y1_case_counts = extra_Y_case_counts
            extra_genes_case_counts = []
            extra_Y_case_counts = np.array([])
            extra_genes_all = extra_genes_case_counts
            Y1_exomes = np.zeros(len(Y1_case_counts))
            Y1_positive_controls = np.zeros(len(Y1_case_counts))
        else:
            # exomes and positive controls are already aligned to runtime_state.genes
            extra_genes_all, aligned_existing_values, extra_Y_case_counts = _align_extra_genes_with_new_source(
                existing_extra_genes=extra_genes_all,
                existing_extra_values=[extra_Y_exomes, extra_Y_positive_controls],
                new_source_genes=extra_genes_case_counts,
                new_source_values=extra_Y_case_counts,
                existing_missing_values=[missing_value_exomes, missing_value_positive_controls],
                new_source_missing_value=missing_value_case_counts,
            )
            extra_Y_exomes = aligned_existing_values[0]
            extra_Y_positive_controls = aligned_existing_values[1]
    else:
        Y1_case_counts = np.zeros(len(Y1_exomes))
        extra_Y_case_counts = np.zeros(len(extra_Y_case_counts))

    assert(len(extra_Y_exomes) == len(extra_genes_all))
    assert(len(extra_Y_exomes) == len(extra_Y_positive_controls))
    assert(len(extra_Y_exomes) == len(extra_Y_case_counts))
    assert(len(Y1_exomes) == len(Y1_positive_controls))
    assert(len(Y1_exomes) == len(Y1_case_counts))

    return (
        Y1_exomes,
        Y1_positive_controls,
        Y1_case_counts,
        extra_genes_all,
        extra_Y_exomes,
        extra_Y_positive_controls,
        extra_Y_case_counts,
        missing_value_exomes,
        missing_value_positive_controls,
        missing_value_case_counts,
    )


def _read_primary_y_source(
    runtime_state,
    gwas_in=None,
    huge_statistics_in=None,
    huge_statistics_out=None,
    exomes_in=None,
    positive_controls_in=None,
    positive_controls_list=None,
    case_counts_in=None,
    gene_bfs_in=None,
    hold_out_chrom=None,
    gene_loc_file=None,
    Y1_exomes=None,
    Y1_positive_controls=None,
    Y1_case_counts=None,
    **kwargs,
):
    missing_value = None
    gene_combined_map = None
    gene_prior_map = None

    huge_or_gwas_source = _read_primary_huge_or_gwas_source(
        runtime_state,
        huge_statistics_in=huge_statistics_in,
        gwas_in=gwas_in,
        huge_statistics_out=huge_statistics_out,
        gene_loc_file=gene_loc_file,
        hold_out_chrom=hold_out_chrom,
        **kwargs,
    )

    if huge_or_gwas_source is not None:
        (
            Y1,
            extra_genes,
            extra_Y,
            Y1_for_regression,
            extra_Y_for_regression,
            missing_value,
        ) = huge_or_gwas_source
    else:
        (
            Y1,
            extra_genes,
            extra_Y,
            Y1_for_regression,
            extra_Y_for_regression,
            gene_combined_map,
            gene_prior_map,
        ) = _read_primary_non_huge_source(
            runtime_state,
            gene_bfs_in=gene_bfs_in,
            exomes_in=exomes_in,
            positive_controls_in=positive_controls_in,
            positive_controls_list=positive_controls_list,
            case_counts_in=case_counts_in,
            Y1_exomes=Y1_exomes,
            Y1_positive_controls=Y1_positive_controls,
            Y1_case_counts=Y1_case_counts,
            hold_out_chrom=hold_out_chrom,
            gene_loc_file=gene_loc_file,
            **kwargs,
        )

    return (
        Y1,
        extra_genes,
        extra_Y,
        Y1_for_regression,
        extra_Y_for_regression,
        missing_value,
        gene_combined_map,
        gene_prior_map,
    )


def _read_primary_huge_or_gwas_source(
    runtime_state,
    huge_statistics_in=None,
    gwas_in=None,
    huge_statistics_out=None,
    gene_loc_file=None,
    hold_out_chrom=None,
    **kwargs,
):
    # Read primary HuGE/gene-level signal from cached statistics or raw GWAS.
    if huge_statistics_in is not None:
        if gwas_in is not None:
            warn("Both --gwas-in and --huge-statistics-in were passed; using --huge-statistics-in")
        (Y1, extra_genes, extra_Y, Y1_for_regression, extra_Y_for_regression) = runtime_state.read_huge_statistics(huge_statistics_in)
        return (Y1, extra_genes, extra_Y, Y1_for_regression, extra_Y_for_regression, 0)

    if gwas_in is None:
        return None

    (Y1, extra_genes, extra_Y, Y1_for_regression, extra_Y_for_regression) = runtime_state.calculate_huge_scores_gwas(
        gwas_in,
        gene_loc_file=gene_loc_file,
        hold_out_chrom=hold_out_chrom,
        **kwargs
    )
    if huge_statistics_out is not None:
        runtime_state.write_huge_statistics(huge_statistics_out, Y1, extra_genes, extra_Y, Y1_for_regression, extra_Y_for_regression)
    return (Y1, extra_genes, extra_Y, Y1_for_regression, extra_Y_for_regression, 0)


def _read_primary_non_huge_source(
    runtime_state,
    gene_bfs_in=None,
    exomes_in=None,
    positive_controls_in=None,
    positive_controls_list=None,
    case_counts_in=None,
    Y1_exomes=None,
    Y1_positive_controls=None,
    Y1_case_counts=None,
    hold_out_chrom=None,
    gene_loc_file=None,
    **kwargs,
):
    runtime_state.huge_signal_bfs = None
    runtime_state.huge_signal_bfs_for_regression = None

    gene_combined_map = None
    gene_prior_map = None
    if gene_bfs_in is not None:
        (Y1, extra_genes, extra_Y, gene_combined_map, gene_prior_map) = runtime_state.read_gene_bfs(
            gene_bfs_in,
            **kwargs
        )
    elif exomes_in is not None:
        (Y1, extra_genes, extra_Y) = (np.zeros(Y1_exomes.shape), [], [])
    elif positive_controls_in is not None or positive_controls_list is not None:
        (Y1, extra_genes, extra_Y) = (np.zeros(Y1_positive_controls.shape), [], [])
    elif case_counts_in is not None:
        (Y1, extra_genes, extra_Y) = (np.zeros(Y1_case_counts.shape), [], [])
    else:
        bail("Need to specify either gene_bfs_in or exomes_in or positive_controls_in or case_counts_in")

    (Y1, extra_genes, extra_Y) = _apply_hold_out_chrom_to_y(
        runtime_state,
        Y1,
        extra_genes,
        extra_Y,
        hold_out_chrom=hold_out_chrom,
        gene_loc_file=gene_loc_file,
    )
    Y1_for_regression = copy.copy(Y1)
    extra_Y_for_regression = copy.copy(extra_Y)

    return (
        Y1,
        extra_genes,
        extra_Y,
        Y1_for_regression,
        extra_Y_for_regression,
        gene_combined_map,
        gene_prior_map,
    )


def _materialize_y_on_gene_universe(
    runtime_state,
    Y1,
    Y1_for_regression,
    Y1_exomes,
    Y1_positive_controls,
    Y1_case_counts,
    extra_genes,
    extra_Y,
    extra_Y_for_regression,
    extra_genes_all,
    extra_Y_exomes,
    extra_Y_positive_controls,
    extra_Y_case_counts,
    missing_value,
    missing_value_exomes,
    missing_value_positive_controls,
    missing_value_case_counts,
):
    if missing_value is None:
        if len(Y1) > 0:
            missing_value = np.nanmean(Y1)
        else:
            missing_value = 0

    if runtime_state.genes is None:
        assert(len(Y1) == 0)
        assert(len(Y1_exomes) == 0)
        assert(len(Y1_positive_controls) == 0)
        assert(len(Y1_case_counts) == 0)
        return _initialize_y_from_new_gene_universe(
            runtime_state,
            extra_genes=extra_genes,
            extra_Y=extra_Y,
            extra_Y_for_regression=extra_Y_for_regression,
            extra_genes_all=extra_genes_all,
            extra_Y_exomes=extra_Y_exomes,
            extra_Y_positive_controls=extra_Y_positive_controls,
            extra_Y_case_counts=extra_Y_case_counts,
            missing_value=missing_value,
            missing_value_exomes=missing_value_exomes,
            missing_value_positive_controls=missing_value_positive_controls,
            missing_value_case_counts=missing_value_case_counts,
        )

    return _merge_y_into_existing_gene_universe(
        runtime_state,
        Y1=Y1,
        Y1_for_regression=Y1_for_regression,
        Y1_exomes=Y1_exomes,
        Y1_positive_controls=Y1_positive_controls,
        Y1_case_counts=Y1_case_counts,
        extra_genes=extra_genes,
        extra_Y=extra_Y,
        extra_Y_for_regression=extra_Y_for_regression,
        extra_genes_all=extra_genes_all,
        extra_Y_exomes=extra_Y_exomes,
        extra_Y_positive_controls=extra_Y_positive_controls,
        extra_Y_case_counts=extra_Y_case_counts,
        missing_value=missing_value,
        missing_value_exomes=missing_value_exomes,
        missing_value_positive_controls=missing_value_positive_controls,
        missing_value_case_counts=missing_value_case_counts,
    )


def _initialize_y_from_new_gene_universe(
    runtime_state,
    extra_genes,
    extra_Y,
    extra_Y_for_regression,
    extra_genes_all,
    extra_Y_exomes,
    extra_Y_positive_controls,
    extra_Y_case_counts,
    missing_value,
    missing_value_exomes,
    missing_value_positive_controls,
    missing_value_case_counts,
):
    # Build the initial gene universe from all gene-level sources.
    genes_union = []
    genes_seen = set()
    for gene in extra_genes + extra_genes_all:
        if gene not in genes_seen:
            genes_union.append(gene)
            genes_seen.add(gene)

    # Populate gene index maps.
    runtime_state._set_X(runtime_state.X_orig, genes_union, runtime_state.gene_sets, skip_N=False)

    # Materialize each Y component on the new gene order.
    Y = np.full(len(runtime_state.genes), missing_value, dtype=float)
    Y_for_regression = np.full(len(runtime_state.genes), missing_value, dtype=float)
    Y_exomes = np.full(len(runtime_state.genes), missing_value_exomes, dtype=float)
    Y_positive_controls = np.full(len(runtime_state.genes), missing_value_positive_controls, dtype=float)
    Y_case_counts = np.full(len(runtime_state.genes), missing_value_case_counts, dtype=float)

    for i in range(len(extra_genes)):
        Y[runtime_state.gene_to_ind[extra_genes[i]]] = extra_Y[i]
        Y_for_regression[runtime_state.gene_to_ind[extra_genes[i]]] = extra_Y_for_regression[i]

    for i in range(len(extra_genes_all)):
        Y_exomes[runtime_state.gene_to_ind[extra_genes_all[i]]] = extra_Y_exomes[i]

    for i in range(len(extra_genes_all)):
        Y_positive_controls[runtime_state.gene_to_ind[extra_genes_all[i]]] = extra_Y_positive_controls[i]

    for i in range(len(extra_genes_all)):
        Y_case_counts[runtime_state.gene_to_ind[extra_genes_all[i]]] = extra_Y_case_counts[i]

    Y += Y_exomes
    Y += Y_positive_controls
    Y += Y_case_counts

    Y_for_regression += Y_exomes
    Y_for_regression += Y_positive_controls
    Y_for_regression += Y_case_counts

    if runtime_state.huge_signal_bfs is not None or runtime_state.gene_covariates is not None:
        if runtime_state.huge_signal_bfs is not None:
            index_map = {i: runtime_state.gene_to_ind[extra_genes[i]] for i in range(len(extra_genes))}
            runtime_state.huge_signal_bfs = sparse.csc_matrix(
                (
                    runtime_state.huge_signal_bfs.data,
                    [index_map[x] for x in runtime_state.huge_signal_bfs.indices],
                    runtime_state.huge_signal_bfs.indptr,
                ),
                shape=runtime_state.huge_signal_bfs.shape,
            )

        if runtime_state.huge_signal_bfs_for_regression is not None:
            index_map = {i: runtime_state.gene_to_ind[extra_genes[i]] for i in range(len(extra_genes))}
            runtime_state.huge_signal_bfs_for_regression = sparse.csc_matrix(
                (
                    runtime_state.huge_signal_bfs_for_regression.data,
                    [index_map[x] for x in runtime_state.huge_signal_bfs_for_regression.indices],
                    runtime_state.huge_signal_bfs_for_regression.indptr,
                ),
                shape=runtime_state.huge_signal_bfs_for_regression.shape,
            )

        if runtime_state.gene_covariates is not None:
            index_map_rev = {runtime_state.gene_to_ind[extra_genes[i]]: i for i in range(len(extra_genes))}
            runtime_state.gene_covariates = runtime_state.gene_covariates[
                [index_map_rev[x] for x in range(runtime_state.gene_covariates.shape[0])], :
            ]

    return (
        Y,
        Y_for_regression,
        Y_exomes,
        Y_positive_controls,
        Y_case_counts,
        [],
        np.array([]),
        np.array([]),
        np.array([]),
        np.array([]),
        np.array([]),
    )


def _merge_y_into_existing_gene_universe(
    runtime_state,
    Y1,
    Y1_for_regression,
    Y1_exomes,
    Y1_positive_controls,
    Y1_case_counts,
    extra_genes,
    extra_Y,
    extra_Y_for_regression,
    extra_genes_all,
    extra_Y_exomes,
    extra_Y_positive_controls,
    extra_Y_case_counts,
    missing_value,
    missing_value_exomes,
    missing_value_positive_controls,
    missing_value_case_counts,
):
    # Combine components on existing genes while preserving original NaN-imputation behavior.
    Y = Y1 + Y1_exomes + Y1_positive_controls + Y1_case_counts
    Y[np.isnan(Y1)] = Y1_exomes[np.isnan(Y1)] + Y1_positive_controls[np.isnan(Y1)] + + Y1_case_counts[np.isnan(Y1)] + missing_value
    Y[np.isnan(Y1_exomes)] = Y1[np.isnan(Y1_exomes)] + Y1_positive_controls[np.isnan(Y1_exomes)] + Y1_case_counts[np.isnan(Y1_exomes)] + missing_value_exomes
    Y[np.isnan(Y1_positive_controls)] = Y1[np.isnan(Y1_positive_controls)] + Y1_exomes[np.isnan(Y1_positive_controls)] + Y1_case_counts[np.isnan(Y1_positive_controls)] + missing_value_positive_controls
    Y[np.isnan(Y1_case_counts)] = Y1[np.isnan(Y1_case_counts)] + Y1_exomes[np.isnan(Y1_case_counts)] + Y1_positive_controls[np.isnan(Y1_case_counts)] + missing_value_case_counts

    Y_for_regression = Y1_for_regression + Y1_exomes + Y1_positive_controls + Y1_case_counts
    Y_for_regression[np.isnan(Y1_for_regression)] = Y1_exomes[np.isnan(Y1_for_regression)] + Y1_positive_controls[np.isnan(Y1_for_regression)] + Y1_case_counts[np.isnan(Y1_for_regression)] + missing_value
    Y_for_regression[np.isnan(Y1_exomes)] = Y1_for_regression[np.isnan(Y1_exomes)] + Y1_positive_controls[np.isnan(Y1_exomes)] + Y1_case_counts[np.isnan(Y1_exomes)] + missing_value_exomes
    Y_for_regression[np.isnan(Y1_positive_controls)] = Y1_for_regression[np.isnan(Y1_positive_controls)] + Y1_exomes[np.isnan(Y1_positive_controls)] + Y1_case_counts[np.isnan(Y1_positive_controls)] + missing_value_positive_controls
    Y_for_regression[np.isnan(Y1_case_counts)] = Y1_for_regression[np.isnan(Y1_case_counts)] + Y1_exomes[np.isnan(Y1_case_counts)] + Y1_positive_controls[np.isnan(Y1_case_counts)] + missing_value_case_counts

    Y_exomes = Y1_exomes
    Y_exomes[np.isnan(Y1_exomes)] = missing_value_exomes

    Y_positive_controls = Y1_positive_controls
    Y_positive_controls[np.isnan(Y1_positive_controls)] = missing_value_positive_controls

    Y_case_counts = Y1_case_counts
    Y_case_counts[np.isnan(Y1_case_counts)] = missing_value_case_counts

    extra_gene_to_ind = pegs_construct_map_to_ind(extra_genes)
    extra_Y = list(extra_Y)
    extra_Y_for_regression = list(extra_Y_for_regression)
    new_extra_Y_exomes = list(np.full(len(extra_Y), missing_value_exomes))
    new_extra_Y_positive_controls = list(np.full(len(extra_Y), missing_value_positive_controls))
    new_extra_Y_case_counts = list(np.full(len(extra_Y), missing_value_case_counts))

    num_add = 0
    for i in range(len(extra_genes_all)):
        if extra_genes_all[i] in extra_gene_to_ind:
            extra_Y[extra_gene_to_ind[extra_genes_all[i]]] += (extra_Y_exomes[i] + extra_Y_positive_controls[i] + extra_Y_case_counts[i])
            extra_Y_for_regression[extra_gene_to_ind[extra_genes_all[i]]] += (extra_Y_exomes[i] + extra_Y_positive_controls[i] + extra_Y_case_counts[i])
            new_extra_Y_exomes[extra_gene_to_ind[extra_genes_all[i]]] = extra_Y_exomes[i]
            new_extra_Y_positive_controls[extra_gene_to_ind[extra_genes_all[i]]] = extra_Y_positive_controls[i]
            new_extra_Y_case_counts[extra_gene_to_ind[extra_genes_all[i]]] = extra_Y_case_counts[i]
        else:
            num_add += 1
            extra_genes.append(extra_genes_all[i])
            extra_Y.append(extra_Y_exomes[i] + extra_Y_positive_controls[i] + extra_Y_case_counts[i])
            extra_Y_for_regression.append(extra_Y_exomes[i] + extra_Y_positive_controls[i] + extra_Y_case_counts[i])
            new_extra_Y_exomes.append(extra_Y_exomes[i])
            new_extra_Y_positive_controls.append(extra_Y_positive_controls[i])
            new_extra_Y_case_counts.append(extra_Y_case_counts[i])

    extra_Y = np.array(extra_Y)
    extra_Y_for_regression = np.array(extra_Y_for_regression)
    extra_Y_exomes = np.array(new_extra_Y_exomes)
    extra_Y_positive_controls = np.array(new_extra_Y_positive_controls)
    extra_Y_case_counts = np.array(new_extra_Y_case_counts)

    if runtime_state.huge_signal_bfs is not None:
        runtime_state.huge_signal_bfs = sparse.csc_matrix(
            (runtime_state.huge_signal_bfs.data, runtime_state.huge_signal_bfs.indices, runtime_state.huge_signal_bfs.indptr),
            shape=(runtime_state.huge_signal_bfs.shape[0] + num_add, runtime_state.huge_signal_bfs.shape[1]),
        )

    if runtime_state.huge_signal_bfs_for_regression is not None:
        runtime_state.huge_signal_bfs_for_regression = sparse.csc_matrix(
            (
                runtime_state.huge_signal_bfs_for_regression.data,
                runtime_state.huge_signal_bfs_for_regression.indices,
                runtime_state.huge_signal_bfs_for_regression.indptr,
            ),
            shape=(runtime_state.huge_signal_bfs_for_regression.shape[0] + num_add, runtime_state.huge_signal_bfs_for_regression.shape[1]),
        )

    if runtime_state.gene_covariates is not None:
        add_gene_covariates = np.tile(np.mean(runtime_state.gene_covariates, axis=0), num_add).reshape((num_add, runtime_state.gene_covariates.shape[1]))
        runtime_state.gene_covariates = np.vstack((runtime_state.gene_covariates, add_gene_covariates))

    return (
        Y,
        Y_for_regression,
        Y_exomes,
        Y_positive_controls,
        Y_case_counts,
        extra_genes,
        extra_Y,
        extra_Y_for_regression,
        extra_Y_exomes,
        extra_Y_positive_controls,
        extra_Y_case_counts,
    )


def _apply_gene_covariates_and_correct_huge(runtime_state, gene_covs_in=None, **kwargs):
    _maybe_append_input_gene_covariates(runtime_state, gene_covs_in=gene_covs_in, **kwargs)

    if runtime_state.gene_covariates is None:
        return

    _prepare_gene_covariate_regression_state(runtime_state)
    _apply_huge_correction_with_covariates(runtime_state)


def _apply_huge_correction_with_covariates(runtime_state):
    # Recompute corrected HuGE scores.
    Y_for_regression = runtime_state.Y_for_regression
    if runtime_state.Y_for_regression is not None:
        (Y_for_regression, _, _) = runtime_state._correct_huge(
            runtime_state.Y_for_regression,
            runtime_state.gene_covariates,
            runtime_state.gene_covariates_mask,
            runtime_state.gene_covariates_mat_inv,
            runtime_state.gene_covariate_names,
            runtime_state.gene_covariate_intercept_index,
        )

    (Y, runtime_state.Y_uncorrected, _) = runtime_state._correct_huge(
        runtime_state.Y,
        runtime_state.gene_covariates,
        runtime_state.gene_covariates_mask,
        runtime_state.gene_covariates_mat_inv,
        runtime_state.gene_covariate_names,
        runtime_state.gene_covariate_intercept_index,
    )

    runtime_state._set_Y(Y, Y_for_regression, runtime_state.Y_exomes, runtime_state.Y_positive_controls, runtime_state.Y_case_counts)
    runtime_state.gene_covariate_adjustments = runtime_state.Y_for_regression - runtime_state.Y_uncorrected

    if runtime_state.gene_to_gwas_huge_score is not None:
        Y_huge = np.zeros(len(runtime_state.Y_for_regression))
        assert(len(Y_huge) == len(runtime_state.genes))
        for i in range(len(runtime_state.genes)):
            if runtime_state.genes[i] in runtime_state.gene_to_gwas_huge_score:
                Y_huge[i] = runtime_state.gene_to_gwas_huge_score[runtime_state.genes[i]]

        (Y_huge, _, _) = runtime_state._correct_huge(
            Y_huge,
            runtime_state.gene_covariates,
            runtime_state.gene_covariates_mask,
            runtime_state.gene_covariates_mat_inv,
            runtime_state.gene_covariate_names,
            runtime_state.gene_covariate_intercept_index,
        )

        for i in range(len(runtime_state.genes)):
            if runtime_state.genes[i] in runtime_state.gene_to_gwas_huge_score:
                runtime_state.gene_to_gwas_huge_score[runtime_state.genes[i]] = Y_huge[i]

        runtime_state.combine_huge_scores()


def _prepare_gene_covariate_regression_state(runtime_state):
    # Remove degenerate / highly collinear columns before linear correction.
    constant_features = np.isclose(np.var(runtime_state.gene_covariates, axis=0), 0)
    if np.sum(constant_features) > 0:
        runtime_state.gene_covariates = runtime_state.gene_covariates[:, ~constant_features]
        runtime_state.gene_covariate_names = [runtime_state.gene_covariate_names[i] for i in np.where(~constant_features)[0]]
        runtime_state.gene_covariate_directions = np.array([runtime_state.gene_covariate_directions[i] for i in np.where(~constant_features)[0]])

    prune_threshold = 0.95
    cor_mat = np.abs(np.corrcoef(runtime_state.gene_covariates.T))
    np.fill_diagonal(cor_mat, 0)

    while True:
        if np.max(cor_mat) < prune_threshold:
            try:
                np.linalg.inv(runtime_state.gene_covariates.T.dot(runtime_state.gene_covariates))
                break
            except np.linalg.LinAlgError:
                pass

        max_index = np.unravel_index(np.argmax(cor_mat), cor_mat.shape)
        if np.max(max_index) == runtime_state.gene_covariate_intercept_index:
            max_index = np.min(max_index)
        else:
            max_index = np.max(max_index)
        log("Removing feature %s" % runtime_state.gene_covariate_names[max_index], TRACE)
        runtime_state.gene_covariates = np.delete(runtime_state.gene_covariates, max_index, axis=1)
        del runtime_state.gene_covariate_names[max_index]
        runtime_state.gene_covariate_directions = np.delete(runtime_state.gene_covariate_directions, max_index)

        cor_mat = np.delete(np.delete(cor_mat, max_index, axis=1), max_index, axis=0)
        if len(runtime_state.gene_covariates) == 0:
            bail("Error: something went wrong with matrix inversion. Still couldn't invert after removing all but one column")

    # Ensure a single intercept is present.
    runtime_state.gene_covariate_intercept_index = np.where(np.isclose(np.var(runtime_state.gene_covariates, axis=0), 0))[0]
    if len(runtime_state.gene_covariate_intercept_index) == 0:
        runtime_state.gene_covariates = np.hstack((runtime_state.gene_covariates, np.ones(runtime_state.gene_covariates.shape[0])[:, np.newaxis]))
        runtime_state.gene_covariate_names.append("intercept")
        runtime_state.gene_covariate_directions = np.append(runtime_state.gene_covariate_directions, 0)
        runtime_state.gene_covariate_intercept_index = len(runtime_state.gene_covariate_names) - 1
    else:
        runtime_state.gene_covariate_intercept_index = runtime_state.gene_covariate_intercept_index[0]

    covariate_means = np.mean(runtime_state.gene_covariates, axis=0)
    covariate_sds = np.std(runtime_state.gene_covariates, axis=0)
    covariate_sds[covariate_sds == 0] = 1

    runtime_state.gene_covariates_mask = np.all(runtime_state.gene_covariates < covariate_means + 5 * covariate_sds, axis=1)
    runtime_state.gene_covariates_mat_inv = np.linalg.inv(
        runtime_state.gene_covariates[runtime_state.gene_covariates_mask, :].T.dot(
            runtime_state.gene_covariates[runtime_state.gene_covariates_mask, :]
        )
    )
    gene_covariate_sds = np.std(runtime_state.gene_covariates, axis=0)
    gene_covariate_sds[gene_covariate_sds == 0] = 1
    runtime_state.gene_covariate_zs = (runtime_state.gene_covariates - np.mean(runtime_state.gene_covariates, axis=0)) / gene_covariate_sds


def _maybe_append_input_gene_covariates(runtime_state, gene_covs_in=None, **kwargs):
    # Load optional covariates and append to any existing covariate matrix.
    if gene_covs_in is None:
        return

    (cov_names, gene_covs, _, _) = runtime_state.read_gene_covs(gene_covs_in, **kwargs)
    cov_dirs = np.array([0] * len(cov_names))

    col_means = np.nanmean(gene_covs, axis=0)
    nan_indices = np.where(np.isnan(gene_covs))
    gene_covs[nan_indices] = np.take(col_means, nan_indices[1])

    if runtime_state.gene_covariates is not None:
        assert(gene_covs.shape[0] == runtime_state.gene_covariates.shape[0])
        runtime_state.gene_covariates = np.hstack((runtime_state.gene_covariates, gene_covs))
        runtime_state.gene_covariate_names = runtime_state.gene_covariate_names + cov_names
        runtime_state.gene_covariate_directions = np.append(runtime_state.gene_covariate_directions, cov_dirs)
    else:
        runtime_state.gene_covariates = gene_covs
        runtime_state.gene_covariate_names = cov_names
        runtime_state.gene_covariate_directions = cov_dirs


def _normalize_dense_gene_rows(mat_info, genes, gene_label_map):
    if gene_label_map is not None:
        genes = list(map(lambda x: gene_label_map[x] if x in gene_label_map else x, genes))

    # Make sure no repeated genes.
    if len(set(genes)) != len(genes):
        seen_genes = set()
        unique_mask = np.full(len(genes), True)
        for i in range(len(genes)):
            if genes[i] in seen_genes:
                unique_mask[i] = False
            else:
                seen_genes.add(genes[i])
        # Subset both matrix rows and gene list to unique genes only.
        mat_info = mat_info[unique_mask, :]
        genes = [genes[i] for i in range(len(genes)) if unique_mask[i]]

    return (mat_info, genes)


def _build_sparse_x_from_dense_input(
    runtime_state,
    mat_info,
    genes,
    gene_sets,
    x_sparsify,
    min_gene_set_size,
    add_ext,
    add_top,
    add_bottom,
    fname=None,
):
    # Check if actually sparse.
    if len(x_sparsify) > 0:
        sparsity_threshold = 1 - np.max(x_sparsify).astype(float) / mat_info.shape[0]
    else:
        sparsity_threshold = 0.95

    orig_dense_gene_sets = gene_sets
    cur_X = None

    # Convert to sparse if (a) many zeros.
    convert_to_sparse = np.sum(mat_info == 0, axis=0) / mat_info.shape[0] > sparsity_threshold

    # Or (b) if all non-zero values are the same.
    abs_mat_info = np.abs(mat_info)
    max_weights = abs_mat_info.max(axis=0)
    all_non_zero_same = np.sum(abs_mat_info * (abs_mat_info != max_weights), axis=0) == 0

    convert_to_sparse = np.logical_or(convert_to_sparse, all_non_zero_same)
    if np.any(convert_to_sparse):
        log("Detected sparse matrix for %d of %d columns" % (np.sum(convert_to_sparse), len(convert_to_sparse)), DEBUG)
        cur_X = sparse.csc_matrix(mat_info[:, convert_to_sparse])
        # Update sparse gene sets and keep dense gene sets for expansions below.
        gene_sets = [gene_sets[i] for i in range(len(gene_sets)) if convert_to_sparse[i]]
        orig_dense_gene_sets = [orig_dense_gene_sets[i] for i in range(len(orig_dense_gene_sets)) if not convert_to_sparse[i]]

        mat_info = mat_info[:, ~convert_to_sparse]
        # Respect min gene size for sparse columns.
        enough_genes = runtime_state.get_col_sums(cur_X, num_nonzero=True) >= min_gene_set_size
        if np.any(~enough_genes):
            log("Excluded %d gene sets due to too small size" % np.sum(~enough_genes), DEBUG)
            cur_X = cur_X[:, enough_genes]
            gene_sets = [gene_sets[i] for i in range(len(gene_sets)) if enough_genes[i]]

    if mat_info.shape[1] > 0:
        mat_sd = np.std(mat_info, axis=0)
        if np.any(mat_sd == 0):
            mat_info = mat_info[:, mat_sd != 0]

        mat_info = (mat_info - np.mean(mat_info, axis=0)) / np.std(mat_info, axis=0)

        subset_mask = np.full(len(genes), True)
        x_for_stats = mat_info
        if runtime_state.Y is not None and runtime_state.genes is not None:
            # Subset down for quantiles using genes seen in Y.
            subset_mask[[i for i in range(len(genes)) if genes[i] not in runtime_state.gene_to_ind]] = False
            x_for_stats = mat_info[subset_mask, :]

        if x_for_stats.shape[0] == 0:
            warn(
                "No genes in --Xd-in %swere seen before so skipping; example genes: %s"
                % ("%s " % fname if fname is not None else "", ",".join(genes[:4]))
            )
            return (None, None, True)

        top_numbers = list(reversed(sorted(x_sparsify)))
        top_fractions = np.array(top_numbers, dtype=float) / x_for_stats.shape[0]

        top_fractions[top_fractions > 1] = 1
        top_fractions[top_fractions < 0] = 0

        if len(top_fractions) == 0:
            bail("No --X-sparsify set so doing nothing")
            return (None, None, True)

        upper_quantiles = np.quantile(x_for_stats, 1 - top_fractions, axis=0)
        lower_quantiles = np.quantile(x_for_stats, top_fractions, axis=0)

        upper = copy.copy(mat_info)
        lower = copy.copy(mat_info)

        assert np.all(upper_quantiles[0, :] == np.min(upper_quantiles, axis=0))
        assert np.all(lower_quantiles[0, :] == np.max(lower_quantiles, axis=0))

        for i in range(len(top_numbers)):
            # Since we are sorted descending, throw away everything below threshold.
            upper_threshold_mask = upper < upper_quantiles[i, :]
            if np.sum(upper_threshold_mask) == 0:
                upper_threshold_mask = upper <= upper_quantiles[i, :]

            lower_threshold_mask = lower > lower_quantiles[i, :]
            if np.sum(lower_threshold_mask) == 0:
                lower_threshold_mask = lower >= lower_quantiles[i, :]

            mat_info[np.logical_and(upper_threshold_mask, lower_threshold_mask)] = 0
            upper[upper_threshold_mask] = 0
            lower[lower_threshold_mask] = 0

            if add_ext:
                temp_X = sparse.csc_matrix(mat_info)
                top_gene_sets = ["%s_%s%d" % (x, EXT_TAG, top_numbers[i]) for x in orig_dense_gene_sets]
                if cur_X is None:
                    cur_X = temp_X
                    gene_sets = top_gene_sets
                else:
                    cur_X = sparse.hstack((cur_X, temp_X))
                    gene_sets = gene_sets + top_gene_sets

            if add_bottom:
                temp_X = sparse.csc_matrix(lower)
                top_gene_sets = ["%s_%s%d" % (x, BOT_TAG, top_numbers[i]) for x in orig_dense_gene_sets]
                if cur_X is None:
                    cur_X = temp_X
                    gene_sets = top_gene_sets
                else:
                    cur_X = sparse.hstack((cur_X, temp_X))
                    gene_sets = gene_sets + top_gene_sets

            if add_top or (not add_ext and not add_bottom):
                temp_X = sparse.csc_matrix(upper)
                top_gene_sets = ["%s_%s%d" % (x, TOP_TAG, top_numbers[i]) for x in orig_dense_gene_sets]
                if cur_X is None:
                    cur_X = temp_X
                    gene_sets = top_gene_sets
                else:
                    gene_sets = gene_sets + top_gene_sets
                    cur_X = sparse.hstack((cur_X, temp_X))

            if cur_X is None:
                return (None, None, True)

            # If all values for a column are negative, flip sign to positive.
            all_negative_mask = ((cur_X < 0).sum(axis=0) == cur_X.astype(bool).sum(axis=0)).A1
            cur_X[:, all_negative_mask] = -cur_X[:, all_negative_mask]
            cur_X.eliminate_zeros()

        if cur_X is None or cur_X.shape[1] == 0:
            return (None, None, True)

    return (cur_X, gene_sets, False)


def _estimate_dense_chunk_size(gene_set_count, only_ids, default_chunk_size=500):
    max_num_at_once = default_chunk_size
    if only_ids and len(only_ids) < gene_set_count:
        # Estimate a larger chunk to maintain enough retained sets after filtering.
        max_num_at_once = int(max_num_at_once / (float(len(only_ids)) / gene_set_count))
    return max_num_at_once


def _record_x_addition(
    runtime_state,
    num_added,
    num_ignored,
    batch_value,
    label_value,
    initial_p_value,
    num_ignored_gene_sets,
    input_index,
    fail_if_first_empty=False,
):
    if fail_if_first_empty and num_added + num_ignored == 0:
        bail("--first-for-hyper was specified but first file had no gene sets")

    runtime_state.gene_set_batches = np.append(runtime_state.gene_set_batches, np.full(num_added, batch_value))
    runtime_state.gene_set_labels = np.append(runtime_state.gene_set_labels, np.full(num_added, label_value))
    if runtime_state.ps is not None and initial_p_value is not None:
        runtime_state.ps = np.append(runtime_state.ps, np.full(num_added, initial_p_value))
    runtime_state.gene_set_labels_ignored = np.append(runtime_state.gene_set_labels_ignored, np.full(num_ignored, label_value))
    num_ignored_gene_sets[input_index] += num_ignored


def _process_dense_x_file(
    runtime_state,
    X_in,
    tag,
    only_ids,
    x_sparsify,
    batch_value,
    label_value,
    initial_p_value,
    num_ignored_gene_sets,
    input_index,
    add_to_x_fn,
):
    with open_gz(X_in) as gene_sets_fh:
        header = gene_sets_fh.readline().strip('\n')
        header = header.lstrip("# \t")
        gene_sets = header.split()
        if len(gene_sets) < 2:
            warn("First line of --Xd-in %s must contain gene column followed by list of gene sets; skipping file" % X_in)
            return False

        # First column is genes so split.
        gene_sets = gene_sets[1:]

        # Maximum number of sets to avoid memory overflow.
        max_num_at_once = _estimate_dense_chunk_size(
            len(gene_sets),
            only_ids=only_ids,
            default_chunk_size=500,
        )

        if len(gene_sets) > max_num_at_once:
            log("Splitting reading of file into chunks to limit memory", DEBUG)
        for j in range(0, len(gene_sets), max_num_at_once):
            if len(gene_sets) > max_num_at_once:
                log("Reading gene sets %d-%d" % (j + 1, j + min(len(gene_sets), j + max_num_at_once + 1)), DEBUG)

            gene_set_indices_to_load = list(range(j, min(len(gene_sets), j + max_num_at_once)))

            gene_set_indices_to_load = _filter_dense_chunk_gene_set_indices(
                gene_sets,
                chunk_indices=gene_set_indices_to_load,
                only_ids=only_ids,
                x_sparsify=x_sparsify,
            )
            if only_ids is not None:
                if len(gene_set_indices_to_load) > 0:
                    log("Will load %d gene sets that were requested" % len(gene_set_indices_to_load), TRACE)
                else:
                    continue

            indices_to_load = [0] + [k + 1 for k in gene_set_indices_to_load]

            cur_X = np.loadtxt(X_in, skiprows=1, dtype=str, usecols=indices_to_load)

            if len(cur_X.shape) == 1:
                cur_X = cur_X[:, np.newaxis]

            if cur_X.shape[1] != len(indices_to_load):
                bail("Xd matrix %s dimensions %s do not match number of gene sets in header line (%s)" % (X_in, cur_X.shape, len(gene_sets)))
            cur_gene_sets = [gene_sets[k] for k in gene_set_indices_to_load]

            genes = cur_X[:, 0]
            if runtime_state.gene_label_map is not None:
                genes = list(map(lambda x: runtime_state.gene_label_map[x] if x in runtime_state.gene_label_map else x, genes))

            mat_info = cur_X[:, 1:].astype(float)
            num_added, num_ignored = add_to_x_fn(
                mat_info,
                genes,
                cur_gene_sets,
                tag,
                skip_scale_factors=False,
            )
            _record_x_addition(
                runtime_state,
                num_added=num_added,
                num_ignored=num_ignored,
                batch_value=batch_value,
                label_value=label_value,
                initial_p_value=initial_p_value,
                num_ignored_gene_sets=num_ignored_gene_sets,
                input_index=input_index,
                fail_if_first_empty=(input_index == 0),
            )

    return True


def _process_sparse_x_file(
    runtime_state,
    X_in,
    tag,
    only_ids,
    min_gene_set_size,
    only_inc_genes,
    fraction_inc_genes,
    ignore_genes,
    max_num_entries_at_once,
    batch_value,
    label_value,
    initial_p_value,
    num_ignored_gene_sets,
    input_index,
    add_to_x_fn,
):
    (
        genes,
        gene_to_ind,
        new_gene_to_ind,
        gene_sets,
        data,
        row,
        col,
        num_read,
        cur_num_read,
    ) = _init_sparse_x_batch_state(runtime_state)
    gene_set_to_ind = {}
    num_too_small = 0
    ignored_for_fraction_inc = 0

    with open_gz(X_in) as gene_sets_fh:
        if max_num_entries_at_once is None:
            max_num_entries_at_once = 200 * 10000

        already_seen = 0
        for line in gene_sets_fh:
            line = line.strip('\n')
            cols = line.split()

            if len(cols) < 2:
                warn("Line does not match format for --X-in: %s" % (line))
                continue
            gs = cols[0]

            if only_ids is not None and gs not in only_ids:
                continue

            if gs in gene_set_to_ind or (runtime_state.gene_set_to_ind is not None and gs in runtime_state.gene_set_to_ind):
                already_seen += 1
                continue

            cur_genes = set(cols[1:])
            if runtime_state.gene_label_map is not None:
                cur_genes = set(map(lambda x: runtime_state.gene_label_map[x] if x in runtime_state.gene_label_map else x, cur_genes))

            if len(cur_genes) < min_gene_set_size:
                # Avoid too small gene sets.
                num_too_small += 1
                continue

            # Initialize a new location for the gene set.
            gene_set_ind = len(gene_sets)
            gene_sets.append(gs)
            # Add this to track duplicates in input file.
            gene_set_to_ind[gs] = gene_set_ind

            if only_inc_genes is not None:
                fraction_match = len(only_inc_genes.intersection(cur_genes)) / float(len(only_inc_genes))
                if fraction_match < (fraction_inc_genes if fraction_inc_genes is not None else 1e-5):
                    ignored_for_fraction_inc += 1
                    continue

            for gene in cur_genes:
                gene_array = gene.split(":")
                gene = gene_array[0]
                if gene in ignore_genes:
                    continue
                if len(gene_array) == 2:
                    try:
                        weight = float(gene_array[1])
                    except ValueError:
                        warn("Couldn't convert weight %s to number so skipping token: %s" % (weight, ":".join(gene_array)))
                        continue
                else:
                    weight = 1.0

                if gene_to_ind is not None and gene in gene_to_ind:
                    # Keep this gene when we harmonize at the end.
                    gene_ind = gene_to_ind[gene]
                else:
                    if gene not in new_gene_to_ind:
                        gene_ind = len(new_gene_to_ind)
                        if gene_to_ind is not None:
                            gene_ind += len(gene_to_ind)

                        new_gene_to_ind[gene] = gene_ind
                        genes.append(gene)
                    else:
                        gene_ind = new_gene_to_ind[gene]

                # Store data for later matrix construction.
                col.append(gene_set_ind)
                row.append(gene_ind)
                data.append(weight)
            num_read += 1
            cur_num_read += 1

            # Add at end or when hit maximum.
            if len(data) >= max_num_entries_at_once:
                log("Batching %d lines to save memory" % cur_num_read)
                num_added, num_ignored = add_to_x_fn((data, row, col), genes, gene_sets, tag, skip_scale_factors=False)
                _record_x_addition(
                    runtime_state,
                    num_added=num_added,
                    num_ignored=num_ignored,
                    batch_value=batch_value,
                    label_value=label_value,
                    initial_p_value=initial_p_value,
                    num_ignored_gene_sets=num_ignored_gene_sets,
                    input_index=input_index,
                    fail_if_first_empty=(input_index == 0),
                )

                # Re-initialize per-batch state.
                (
                    genes,
                    gene_to_ind,
                    new_gene_to_ind,
                    gene_sets,
                    data,
                    row,
                    col,
                    num_read,
                    cur_num_read,
                ) = _init_sparse_x_batch_state(runtime_state)
                log("Continuing reading...")

        if already_seen > 0:
            warn("Skipped second occurrence of %d repeated gene sets" % already_seen)

        mat_info = (data, row, col) if len(data) > 0 else None

    if mat_info is not None:
        num_added, num_ignored = add_to_x_fn(mat_info, genes, gene_sets, tag, skip_scale_factors=False)
        _record_x_addition(
            runtime_state,
            num_added=num_added,
            num_ignored=num_ignored,
            batch_value=batch_value,
            label_value=label_value,
            initial_p_value=initial_p_value,
            num_ignored_gene_sets=num_ignored_gene_sets,
            input_index=input_index,
            fail_if_first_empty=(input_index == 0),
        )

    return (num_too_small, ignored_for_fraction_inc)


def _process_x_input_file(
    runtime_state,
    X_in,
    tag,
    is_dense_input,
    only_ids,
    x_sparsify,
    batch_value,
    label_value,
    initial_p_value,
    num_ignored_gene_sets,
    input_index,
    add_to_x_fn,
    min_gene_set_size,
    only_inc_genes,
    fraction_inc_genes,
    ignore_genes,
    max_num_entries_at_once,
):
    num_too_small = 0
    ignored_for_fraction_inc = 0

    if is_dense_input:
        processed_dense = _process_dense_x_file(
            runtime_state,
            X_in=X_in,
            tag=tag,
            only_ids=only_ids,
            x_sparsify=x_sparsify,
            batch_value=batch_value,
            label_value=label_value,
            initial_p_value=initial_p_value,
            num_ignored_gene_sets=num_ignored_gene_sets,
            input_index=input_index,
            add_to_x_fn=add_to_x_fn,
        )
        if not processed_dense:
            return (num_too_small, ignored_for_fraction_inc, False)
    else:
        num_too_small, ignored_for_fraction_inc = _process_sparse_x_file(
            runtime_state,
            X_in=X_in,
            tag=tag,
            only_ids=only_ids,
            min_gene_set_size=min_gene_set_size,
            only_inc_genes=only_inc_genes,
            fraction_inc_genes=fraction_inc_genes,
            ignore_genes=ignore_genes,
            max_num_entries_at_once=max_num_entries_at_once,
            batch_value=batch_value,
            label_value=label_value,
            initial_p_value=initial_p_value,
            num_ignored_gene_sets=num_ignored_gene_sets,
            input_index=input_index,
            add_to_x_fn=add_to_x_fn,
        )

    return (num_too_small, ignored_for_fraction_inc, True)


def _standardize_qc_metrics_after_x_read(runtime_state):
    pegs_standardize_qc_metrics_after_x_read(runtime_state)


def _maybe_correct_gene_set_betas_after_x_read(
    runtime_state,
    filter_gene_set_p,
    correct_betas_mean,
    correct_betas_var,
    filter_using_phewas,
):
    pegs_maybe_correct_gene_set_betas_after_x_read(
        runtime_state,
        filter_gene_set_p=filter_gene_set_p,
        correct_betas_mean=correct_betas_mean,
        correct_betas_var=correct_betas_var,
        filter_using_phewas=filter_using_phewas,
        log_fn=lambda message: log(message),
    )


def _maybe_limit_initial_gene_sets_by_p(runtime_state, max_num_gene_sets_initial):
    pegs_maybe_limit_initial_gene_sets_by_p(
        runtime_state,
        max_num_gene_sets_initial=max_num_gene_sets_initial,
        log_fn=lambda message: log(message),
    )


def _maybe_prune_gene_sets_after_x_read(
    runtime_state,
    skip_betas,
    prune_gene_sets,
    prune_deterministically,
    weighted_prune_gene_sets,
):
    pegs_maybe_prune_gene_sets_after_x_read(
        runtime_state,
        skip_betas=skip_betas,
        prune_gene_sets=prune_gene_sets,
        prune_deterministically=prune_deterministically,
        weighted_prune_gene_sets=weighted_prune_gene_sets,
    )


def _initialize_hyper_defaults_after_x_read(
    runtime_state,
    initial_p,
    update_hyper_p,
    sigma_power,
    initial_sigma2_cond,
    update_hyper_sigma,
    initial_sigma2,
    sigma_soft_threshold_95,
    sigma_soft_threshold_5,
):
    return pegs_initialize_hyper_defaults_after_x_read(
        runtime_state,
        initial_p=initial_p,
        update_hyper_p=update_hyper_p,
        sigma_power=sigma_power,
        initial_sigma2_cond=initial_sigma2_cond,
        update_hyper_sigma=update_hyper_sigma,
        initial_sigma2=initial_sigma2,
        sigma_soft_threshold_95=sigma_soft_threshold_95,
        sigma_soft_threshold_5=sigma_soft_threshold_5,
        warn_fn=lambda message: warn(message),
        log_fn=lambda message: log(message),
    )


def _learn_hyper_for_gene_set_batch(
    runtime_state,
    gene_sets_for_hyper_mask,
    num_missing_gene_sets,
    update_hyper_p,
    update_hyper_sigma,
    first_for_sigma_cond,
    fixed_sigma_cond,
    ordered_batch_ind,
    max_num_burn_in,
    max_num_iter_betas,
    min_num_iter_betas,
    num_chains_betas,
    r_threshold_burn_in_betas,
    use_max_r_for_convergence_betas,
    max_frac_sem_betas,
    max_allowed_batch_correlation,
    sigma_num_devs_to_top,
    p_noninf_inflate,
    sparse_solution,
    sparse_frac_betas,
    betas_trace_out,
):
    # Keep per-gene-set p/sigma vectors unset while learning a batch-level
    # hyper state, then return explicit learned values to the caller.
    with _temporary_state_fields(
        runtime_state,
        overrides={"ps": None, "sigma2s": None},
        restore_fields=("ps", "sigma2s"),
    ):
        if np.sum(gene_sets_for_hyper_mask) > runtime_state.batch_size:
            V = None
        else:
            V = runtime_state._calculate_V_internal(
                runtime_state.X_orig[:, gene_sets_for_hyper_mask],
                runtime_state.y_corr_cholesky,
                runtime_state.mean_shifts[gene_sets_for_hyper_mask],
                runtime_state.scale_factors[gene_sets_for_hyper_mask],
            )

        # Only add pseudo counts for large values.
        num_p_pseudo = min(1, np.sum(gene_sets_for_hyper_mask) / 1000)

        # Optionally keep sigma/p fixed across batches.
        cur_update_hyper_p = update_hyper_p
        cur_update_hyper_sigma = update_hyper_sigma
        adjust_hyper_sigma_p = False
        if (first_for_sigma_cond and ordered_batch_ind > 0) or fixed_sigma_cond:
            adjust_hyper_sigma_p = True
            if cur_update_hyper_p:
                cur_update_hyper_sigma = False

        runtime_state._calculate_non_inf_betas(
            initial_p=None,
            beta_tildes=runtime_state.beta_tildes[gene_sets_for_hyper_mask],
            ses=runtime_state.ses[gene_sets_for_hyper_mask],
            V=V,
            X_orig=runtime_state.X_orig[:, gene_sets_for_hyper_mask],
            scale_factors=runtime_state.scale_factors[gene_sets_for_hyper_mask],
            mean_shifts=runtime_state.mean_shifts[gene_sets_for_hyper_mask],
            is_dense_gene_set=runtime_state.is_dense_gene_set[gene_sets_for_hyper_mask],
            ps=None,
            max_num_burn_in=max_num_burn_in,
            max_num_iter=max_num_iter_betas,
            min_num_iter=min_num_iter_betas,
            num_chains=num_chains_betas,
            r_threshold_burn_in=r_threshold_burn_in_betas,
            use_max_r_for_convergence=use_max_r_for_convergence_betas,
            max_frac_sem=max_frac_sem_betas,
            max_allowed_batch_correlation=max_allowed_batch_correlation,
            gauss_seidel=False,
            update_hyper_sigma=cur_update_hyper_sigma,
            update_hyper_p=cur_update_hyper_p,
            only_update_hyper=True,
            adjust_hyper_sigma_p=adjust_hyper_sigma_p,
            sigma_num_devs_to_top=sigma_num_devs_to_top,
            p_noninf_inflate=p_noninf_inflate,
            num_p_pseudo=num_p_pseudo,
            num_missing_gene_sets=num_missing_gene_sets,
            sparse_solution=sparse_solution,
            sparse_frac_betas=sparse_frac_betas,
            betas_trace_out=betas_trace_out,
            betas_trace_gene_sets=[
                runtime_state.gene_sets[j]
                for j in range(len(runtime_state.gene_sets))
                if gene_sets_for_hyper_mask[j]
            ],
        )

        return {
            "computed_p": runtime_state.p,
            "computed_sigma2": runtime_state.sigma2,
            "computed_sigma_power": runtime_state.sigma_power,
        }


def _apply_learned_batch_hyper_values(
    runtime_state,
    gene_sets_in_batch_mask,
    computed_p,
    computed_sigma2,
    first_p,
    first_max_p_for_hyper,
):
    updated_first_p = first_p
    adjusted_p = computed_p
    adjusted_sigma2 = computed_sigma2

    if updated_first_p is None:
        updated_first_p = adjusted_p
    elif first_max_p_for_hyper and adjusted_p > updated_first_p:
        # Keep sigma/first_p = sigma/computed_p.
        adjusted_sigma2 = adjusted_sigma2 / adjusted_p * updated_first_p
        adjusted_p = updated_first_p

    runtime_state.ps[gene_sets_in_batch_mask] = adjusted_p
    runtime_state.sigma2s[gene_sets_in_batch_mask] = adjusted_sigma2
    return updated_first_p


def _finalize_batch_hyper_vectors(runtime_state, first_for_hyper):
    assert(len(runtime_state.ps) > 0 and not np.isnan(runtime_state.ps[0]))
    assert(len(runtime_state.sigma2s) > 0 and not np.isnan(runtime_state.sigma2s[0]))

    if first_for_hyper:
        runtime_state.ps[np.isnan(runtime_state.ps)] = runtime_state.ps[0]
        runtime_state.sigma2s[np.isnan(runtime_state.sigma2s)] = runtime_state.sigma2s[0]
    else:
        runtime_state.ps[np.isnan(runtime_state.ps)] = np.mean(runtime_state.ps[~np.isnan(runtime_state.ps)])
        runtime_state.sigma2s[np.isnan(runtime_state.sigma2s)] = np.mean(runtime_state.sigma2s[~np.isnan(runtime_state.sigma2s)])

    runtime_state.set_p(np.mean(runtime_state.ps))
    runtime_state.set_sigma(np.mean(runtime_state.sigma2s), runtime_state.sigma_power)


def _maybe_learn_batch_hyper_after_x_read(
    runtime_state,
    skip_betas,
    update_hyper_p,
    update_hyper_sigma,
    batches,
    num_ignored_gene_sets,
    first_for_hyper,
    max_num_gene_sets_hyper,
    first_for_sigma_cond,
    fixed_sigma_cond,
    first_max_p_for_hyper,
    max_num_burn_in,
    max_num_iter_betas,
    min_num_iter_betas,
    num_chains_betas,
    r_threshold_burn_in_betas,
    use_max_r_for_convergence_betas,
    max_frac_sem_betas,
    max_allowed_batch_correlation,
    sigma_num_devs_to_top,
    p_noninf_inflate,
    sparse_solution,
    sparse_frac_betas,
    betas_trace_out,
):
    if skip_betas or runtime_state.p_values is None or (not update_hyper_p and not update_hyper_sigma) or len(runtime_state.gene_set_batches) == 0:
        return

    # Now learn the hyper values.
    assert(runtime_state.gene_set_batches[0] is not None)
    # First order unique batches; batches has one value per file but we need one per unique batch.
    ordered_batches = [runtime_state.gene_set_batches[0]] + list(set([x for x in runtime_state.gene_set_batches if not x == runtime_state.gene_set_batches[0]]))
    # Get total ignored counts per batch.
    batches_num_ignored = {}
    for i in range(len(batches)):
        if batches[i] not in batches_num_ignored:
            batches_num_ignored[batches[i]] = 0
        batches_num_ignored[batches[i]] += num_ignored_gene_sets[i]

    if update_hyper_p:
        runtime_state.ps = np.full(len(runtime_state.gene_set_batches), np.nan)
    runtime_state.sigma2s = np.full(len(runtime_state.gene_set_batches), np.nan)

    # None batch learns from first; others learn within themselves.
    first_p = None
    for ordered_batch_ind in range(len(ordered_batches)):
        if ordered_batches[ordered_batch_ind] is None:
            assert(first_for_hyper)
            continue

        gene_sets_in_batch_mask = (runtime_state.gene_set_batches == ordered_batches[ordered_batch_ind])
        gene_sets_for_hyper_mask = gene_sets_in_batch_mask.copy()

        if max_num_gene_sets_hyper is not None:
            if np.sum(gene_sets_for_hyper_mask) > max_num_gene_sets_hyper:
                drop_mask = np.random.default_rng().choice(
                    np.where(gene_sets_for_hyper_mask)[0],
                    size=np.sum(gene_sets_for_hyper_mask) - runtime_state.batch_size,
                    replace=False,
                )
                log("Dropping %d gene sets to reduce gene sets used for hyper parameters to %d" % (len(drop_mask), max_num_gene_sets_hyper))
                gene_sets_for_hyper_mask[drop_mask] = False

        if ordered_batch_ind > 0 and np.sum(gene_sets_for_hyper_mask) + batches_num_ignored[ordered_batches[ordered_batch_ind]] < 100:
            log("Skipping learning hyper for batch %s since not enough gene sets" % (ordered_batches[ordered_batch_ind]))
            continue

        hyper_fit = _learn_hyper_for_gene_set_batch(
            runtime_state=runtime_state,
            gene_sets_for_hyper_mask=gene_sets_for_hyper_mask,
            num_missing_gene_sets=batches_num_ignored[ordered_batches[ordered_batch_ind]],
            update_hyper_p=update_hyper_p,
            update_hyper_sigma=update_hyper_sigma,
            first_for_sigma_cond=first_for_sigma_cond,
            fixed_sigma_cond=fixed_sigma_cond,
            ordered_batch_ind=ordered_batch_ind,
            max_num_burn_in=max_num_burn_in,
            max_num_iter_betas=max_num_iter_betas,
            min_num_iter_betas=min_num_iter_betas,
            num_chains_betas=num_chains_betas,
            r_threshold_burn_in_betas=r_threshold_burn_in_betas,
            use_max_r_for_convergence_betas=use_max_r_for_convergence_betas,
            max_frac_sem_betas=max_frac_sem_betas,
            max_allowed_batch_correlation=max_allowed_batch_correlation,
            sigma_num_devs_to_top=sigma_num_devs_to_top,
            p_noninf_inflate=p_noninf_inflate,
            sparse_solution=sparse_solution,
            sparse_frac_betas=sparse_frac_betas,
            betas_trace_out=betas_trace_out,
        )
        computed_p = hyper_fit["computed_p"]
        computed_sigma2 = hyper_fit["computed_sigma2"]
        computed_sigma_power = hyper_fit["computed_sigma_power"]

        log("Learned p=%.4g, sigma2=%.4g (sigma2/p=%.4g)" % (computed_p, computed_sigma2, computed_sigma2 / computed_p))
        runtime_state._record_params(
            {
                "p": computed_p,
                "sigma2": computed_sigma2,
                "sigma2_cond": computed_sigma2 / computed_p,
                "sigma_power": computed_sigma_power,
                "sigma_threshold_k": runtime_state.sigma_threshold_k,
                "sigma_threshold_xo": runtime_state.sigma_threshold_xo,
            }
        )

        first_p = _apply_learned_batch_hyper_values(
            runtime_state=runtime_state,
            gene_sets_in_batch_mask=gene_sets_in_batch_mask,
            computed_p=computed_p,
            computed_sigma2=computed_sigma2,
            first_p=first_p,
            first_max_p_for_hyper=first_max_p_for_hyper,
        )

    _finalize_batch_hyper_vectors(runtime_state=runtime_state, first_for_hyper=first_for_hyper)


def _maybe_adjust_overaggressive_p_filter_after_x_read(
    runtime_state,
    filter_gene_set_p,
    increase_filter_gene_set_p,
    filter_using_phewas,
):
    pegs_maybe_adjust_overaggressive_p_filter_after_x_read(
        runtime_state,
        filter_gene_set_p=filter_gene_set_p,
        increase_filter_gene_set_p=increase_filter_gene_set_p,
        filter_using_phewas=filter_using_phewas,
        log_fn=lambda message: log(message),
    )


def _apply_post_read_gene_set_size_and_qc_filters(
    runtime_state,
    min_gene_set_size,
    max_gene_set_size,
    filter_gene_set_metric_z,
):
    pegs_apply_post_read_gene_set_size_and_qc_filters(
        runtime_state,
        min_gene_set_size=min_gene_set_size,
        max_gene_set_size=max_gene_set_size,
        filter_gene_set_metric_z=filter_gene_set_metric_z,
        log_fn=lambda message: log(message),
    )


def _maybe_filter_zero_uncorrected_betas_after_x_read(
    runtime_state,
    sort_rank,
    skip_betas,
    filter_gene_set_p,
    filter_using_phewas,
    max_num_burn_in,
    max_num_iter_betas,
    min_num_iter_betas,
    num_chains_betas,
    r_threshold_burn_in_betas,
    use_max_r_for_convergence_betas,
    max_frac_sem_betas,
    max_allowed_batch_correlation,
    sparse_solution,
    sparse_frac_betas,
):
    if skip_betas or runtime_state.p_values is None or filter_gene_set_p >= 1 or filter_using_phewas:
        return sort_rank

    # Remove features with identically zero uncorrected effects.
    betas, _avg_postp = runtime_state._calculate_non_inf_betas(
        initial_p=None,
        assume_independent=True,
        max_num_burn_in=max_num_burn_in,
        max_num_iter=max_num_iter_betas,
        min_num_iter=min_num_iter_betas,
        num_chains=num_chains_betas,
        r_threshold_burn_in=r_threshold_burn_in_betas,
        use_max_r_for_convergence=use_max_r_for_convergence_betas,
        max_frac_sem=max_frac_sem_betas,
        max_allowed_batch_correlation=max_allowed_batch_correlation,
        gauss_seidel=False,
        update_hyper_sigma=False,
        update_hyper_p=False,
        adjust_hyper_sigma_p=False,
        sparse_solution=sparse_solution,
        sparse_frac_betas=sparse_frac_betas,
    )

    log("%d have betas uncorrected equal 0" % np.sum(betas == 0))
    log("%d have betas uncorrected below 0.001" % np.sum(betas < 0.001))
    log("%d have betas uncorrected below 0.01" % np.sum(betas < 0.01))

    beta_ignore = betas == 0
    beta_mask = ~beta_ignore
    if np.sum(beta_mask) > 0:
        log("Ignoring %d gene sets due to zero uncorrected betas (kept %d)" % (np.sum(beta_ignore), np.sum(beta_mask)))
        runtime_state.subset_gene_sets(beta_mask, keep_missing=False, ignore_missing=True, skip_V=True)
    else:
        log("Keeping %d gene sets with zero uncorrected betas to avoid having none" % (np.sum(beta_ignore)))

    return -np.abs(betas[beta_mask])


def _maybe_reduce_gene_sets_to_max_after_x_read(
    runtime_state,
    skip_betas,
    max_num_gene_sets,
    sort_rank,
):
    if skip_betas or max_num_gene_sets is None or max_num_gene_sets <= 0:
        return
    if len(runtime_state.gene_sets) <= max_num_gene_sets:
        return

    log(
        "Current %d gene sets is greater than the maximum specified %d; reducing using pruning + small beta removal"
        % (len(runtime_state.gene_sets), max_num_gene_sets),
        DEBUG,
    )
    gene_set_masks = runtime_state._compute_gene_set_batches(
        V=None,
        X_orig=runtime_state.X_orig,
        mean_shifts=runtime_state.mean_shifts,
        scale_factors=runtime_state.scale_factors,
        sort_values=sort_rank,
        resort_as_added=True,
        stop_at=max_num_gene_sets,
    )
    keep_mask = np.full(len(runtime_state.gene_sets), False)
    for gene_set_mask in gene_set_masks:
        keep_mask[gene_set_mask] = True
        log("Adding %d relatively uncorrelated gene sets (total now %d)" % (np.sum(gene_set_mask), np.sum(keep_mask)), TRACE)
        if np.sum(keep_mask) > max_num_gene_sets:
            break
    if np.sum(keep_mask) > max_num_gene_sets:
        keep_indices = np.where(keep_mask)[0]
        if sort_rank is not None:
            keep_indices = keep_indices[np.argsort(sort_rank[keep_indices], kind="stable")]
        trimmed_keep_mask = np.full(len(runtime_state.gene_sets), False)
        trimmed_keep_mask[keep_indices[:max_num_gene_sets]] = True
        keep_mask = trimmed_keep_mask
    if np.sum(~keep_mask) > 0:
        runtime_state.subset_gene_sets(keep_mask, keep_missing=False, ignore_missing=True, skip_V=True)


def _init_sparse_x_batch_state(runtime_state):
    genes = []
    gene_to_ind = None
    if runtime_state.genes is not None:
        # Ensure the batch matrix always contains all currently tracked genes.
        genes = copy.copy(runtime_state.genes)
        if runtime_state.genes_missing is not None:
            genes += runtime_state.genes_missing
        gene_to_ind = pegs_construct_map_to_ind(genes)

    return (
        genes,
        gene_to_ind,
        {},
        [],
        [],
        [],
        [],
        0,
        0,
    )


def _filter_dense_chunk_gene_set_indices(gene_sets, chunk_indices, only_ids, x_sparsify):
    if only_ids is None:
        return chunk_indices

    keep_mask = np.full(len(chunk_indices), False)
    for k in range(len(keep_mask)):
        gs = gene_sets[chunk_indices[k]]
        if gs in only_ids:
            keep_mask[k] = True
        elif x_sparsify is not None:
            for top_number in x_sparsify:
                matched = False
                for sparse_tag in [EXT_TAG, TOP_TAG, BOT_TAG]:
                    if "%s_%s%d" % (gs, sparse_tag, top_number) in only_ids:
                        keep_mask[k] = True
                        matched = True
                        break
                if matched:
                    break

    if np.any(keep_mask):
        return [chunk_indices[i] for i in range(len(keep_mask)) if keep_mask[i]]
    return []


def _normalize_gene_set_weights(runtime_state, cur_X, threshold_weights, cap_weights):
    denom = runtime_state.get_col_sums(cur_X, num_nonzero=True)
    denom[denom == 0] = 1
    avg_weights = np.abs(cur_X).sum(axis=0) / denom
    if np.sum(avg_weights != 1) > 0:
        # this is an option to use the max weight after throwing out outliers as the norm
        # it doesn't look to work as well as avg_weights
        max_weight_devs = None
        if max_weight_devs is not None:
            dev_weights = np.sqrt(np.abs(cur_X).power(2).sum(axis=0) / denom - np.power(avg_weights, 2))
            temp_X = copy.copy(np.abs(cur_X))
            temp_X[temp_X > avg_weights + max_weight_devs * dev_weights] = 0
            weight_norm = temp_X.max(axis=0).todense().A1
        else:
            weight_norm = avg_weights.A1

        weight_norm = np.round(weight_norm, 10)
        weight_norm[weight_norm == 0] = 1

        # assume rows are already normalized if (a) all are below 1 and
        # (b) threshold is None or all are above threshold
        normalize_mask = (np.abs(cur_X) > 1).sum(axis=0).A1 > 0
        if threshold_weights is not None and threshold_weights > 0:
            # check for those that have different number above 0 and above threshold
            normalize_mask = np.logical_or(
                normalize_mask,
                (np.abs(cur_X) >= threshold_weights).sum(axis=0).A1 != (np.abs(cur_X) > 0).sum(axis=0).A1,
            )

        # this uses less memory
        weight_norm[~normalize_mask] = 1.0
        cur_X = sparse.csc_matrix(cur_X.multiply(1.0 / weight_norm))

        # don't do binary; use threshold instead
        if threshold_weights is not None and threshold_weights > 0:
            cur_X.data[np.abs(cur_X.data) < threshold_weights] = 0
            if cap_weights:
                cur_X.data[cur_X.data > 1] = 1
                cur_X.data[cur_X.data < -1] = -1
        cur_X.eliminate_zeros()

    return cur_X


def _maybe_permute_gene_set_rows(runtime_state, cur_X, permute_gene_sets):
    if not permute_gene_sets:
        return cur_X

    # Permute rows while preserving the "observed genes first, missing genes last"
    # contract when Y is already loaded.
    if runtime_state.Y is not None:
        assert len(runtime_state.Y) == len(runtime_state.genes)
        orig_indices = list(range(len(runtime_state.Y)))
        new_indices = random.sample(orig_indices, len(orig_indices))
        if cur_X.shape[0] > len(orig_indices):
            num_to_add = cur_X.shape[0] - len(orig_indices)
            to_add = list(range(len(orig_indices), len(orig_indices) + num_to_add))
            orig_indices += to_add
            new_indices += random.sample(to_add, len(to_add))
    else:
        orig_indices = list(range(cur_X.shape[0]))
        new_indices = random.sample(orig_indices, len(orig_indices))

    index_map = dict(zip(orig_indices, new_indices))
    cur_X = sparse.csc_matrix(cur_X)
    return sparse.csc_matrix(
        (cur_X.data, [index_map[x] for x in cur_X.indices], cur_X.indptr),
        shape=(cur_X.shape[0], cur_X.shape[1]),
    )


def _align_prefilter_gene_set_signs(cur_X, beta_tildes, z_scores):
    # For continuous gene-set weights, orient each feature so pre-filtering is
    # based on the stronger (positive) association direction.
    negative_weights_mask = (cur_X < 0).sum(axis=0).A1 > 0
    if np.sum(negative_weights_mask) > 0:
        flip_mask = np.logical_and(beta_tildes < 0, negative_weights_mask)
        if np.sum(flip_mask) > 0:
            log("Flipped %d gene sets" % np.sum(flip_mask), DEBUG)
            beta_tildes[flip_mask] = -beta_tildes[flip_mask]
            z_scores[flip_mask] = -z_scores[flip_mask]
            cur_X[:, flip_mask] = -cur_X[:, flip_mask]
    return (cur_X, beta_tildes, z_scores)


def _build_prefilter_keep_mask(
    p_values,
    beta_tildes,
    filter_gene_set_p,
    filter_using_phewas=False,
    p_values_phewas=None,
    beta_tildes_phewas=None,
    increase_filter_gene_set_p=None,
    filter_negative=True,
):
    p_value_mask = p_values <= filter_gene_set_p
    if filter_using_phewas:
        p_value_mask = np.logical_or(p_value_mask, np.any(p_values_phewas <= filter_gene_set_p, axis=0))

    if increase_filter_gene_set_p is not None and np.mean(p_value_mask) < increase_filter_gene_set_p:
        # Choose a new more lenient threshold.
        p_from_quantile = np.quantile(p_values, increase_filter_gene_set_p)
        log(
            "Choosing revised p threshold %.3g to ensure keeping %.3g fraction of gene sets"
            % (p_from_quantile, increase_filter_gene_set_p),
            DEBUG,
        )
        p_value_mask = p_values <= p_from_quantile
        if filter_using_phewas:
            # p_values_phewas is shaped (num_phenos, num_gene_sets), so aggregate
            # across phenotypes to keep any gene set passing in at least one pheno.
            p_value_mask = np.logical_or(p_value_mask, np.any(p_values_phewas <= p_from_quantile, axis=0))

        if np.sum(~p_value_mask) > 0:
            log("Ignoring %d gene sets due to p-value filters" % (np.sum(~p_value_mask)))

    if filter_negative:
        negative_beta_tildes_mask = beta_tildes < 0
        if filter_using_phewas:
            negative_beta_tildes_mask = np.logical_and(negative_beta_tildes_mask, np.all(beta_tildes_phewas < 0, axis=0))
        p_value_mask = np.logical_and(p_value_mask, ~negative_beta_tildes_mask)
        if np.sum(negative_beta_tildes_mask) > 0:
            log("Ignoring %d gene sets due to negative beta filters" % (np.sum(negative_beta_tildes_mask)))

    return p_value_mask


def _compute_prefilter_qc_metrics(runtime_state, cur_X):
    total_qc_metrics = None
    mean_qc_metrics = None
    total_qc_metrics_directions = None
    if runtime_state.gene_covariates is None:
        return (total_qc_metrics, mean_qc_metrics, total_qc_metrics_directions)

    cur_X_size = np.abs(cur_X).sum(axis=0)
    cur_X_size[cur_X_size == 0] = 1

    total_qc_metrics = (np.array(cur_X.T.dot(runtime_state.gene_covariate_zs).T / cur_X_size)).T
    total_qc_metrics = np.hstack(
        (
            total_qc_metrics[:, : runtime_state.gene_covariate_intercept_index],
            total_qc_metrics[:, runtime_state.gene_covariate_intercept_index + 1 :],
        )
    )

    total_qc_metrics_directions = np.append(
        runtime_state.gene_covariate_directions[: runtime_state.gene_covariate_intercept_index],
        runtime_state.gene_covariate_directions[runtime_state.gene_covariate_intercept_index + 1 :],
    )

    total_huge_adjustments = (np.array(cur_X.T.dot(runtime_state.gene_covariate_adjustments).T / cur_X_size)).T

    total_qc_metrics = np.hstack((total_qc_metrics, total_huge_adjustments))
    total_qc_metrics_directions = np.append(total_qc_metrics_directions, -1)

    if runtime_state.debug_only_avg_huge:
        total_qc_metrics = total_huge_adjustments
        total_qc_metrics_directions = np.array(-1)

    mean_qc_metrics = total_huge_adjustments.squeeze()
    mean_qc_metrics = total_huge_adjustments
    if len(mean_qc_metrics.shape) == 2 and mean_qc_metrics.shape[1] == 1:
        mean_qc_metrics = mean_qc_metrics.squeeze(axis=1)

    return (total_qc_metrics, mean_qc_metrics, total_qc_metrics_directions)


def _compute_prefilter_assoc_stats(runtime_state, cur_X, run_logistic, filter_using_phewas, mean_shifts, scale_factors):
    Y_to_use = runtime_state.Y_for_regression
    gene_pheno_Y = runtime_state.gene_pheno_Y.T.toarray() if filter_using_phewas else None

    if run_logistic:
        Y = np.exp(Y_to_use + runtime_state.background_log_bf) / (1 + np.exp(Y_to_use + runtime_state.background_log_bf))
        (
            beta_tildes,
            ses,
            z_scores,
            p_values,
            se_inflation_factors,
            _alpha_tildes,
            _diverged,
        ) = runtime_state._compute_logistic_beta_tildes(
            cur_X,
            Y,
            scale_factors,
            mean_shifts,
            resid_correlation_matrix=runtime_state.y_corr_sparse,
        )

        beta_tildes_phewas = None
        p_values_phewas = None
        if filter_using_phewas:
            gene_pheno_Y = np.exp(np.array(gene_pheno_Y) + runtime_state.background_log_bf) / (1 + np.exp(gene_pheno_Y + runtime_state.background_log_bf))
            (
                beta_tildes_phewas,
                _ses_phewas,
                _z_scores_phewas,
                p_values_phewas,
                _se_inflation_factors_phewas,
                _alpha_tildes_phewas,
                _diverged_phewas,
            ) = runtime_state._compute_logistic_beta_tildes(
                cur_X,
                gene_pheno_Y,
                scale_factors,
                mean_shifts,
                resid_correlation_matrix=runtime_state.y_corr_sparse,
            )
    else:
        (
            beta_tildes,
            ses,
            z_scores,
            p_values,
            se_inflation_factors,
        ) = runtime_state._compute_beta_tildes(
            cur_X,
            Y_to_use,
            np.var(Y_to_use),
            scale_factors,
            mean_shifts,
            resid_correlation_matrix=runtime_state.y_corr_sparse,
        )
        beta_tildes_phewas = None
        p_values_phewas = None
        if filter_using_phewas:
            (
                beta_tildes_phewas,
                _ses_phewas,
                _z_scores_phewas,
                p_values_phewas,
                _se_inflation_factors_phewas,
            ) = runtime_state._compute_beta_tildes(
                cur_X,
                gene_pheno_Y,
                None,
                scale_factors,
                mean_shifts,
                resid_correlation_matrix=runtime_state.y_corr_sparse,
            )

    return (
        beta_tildes,
        ses,
        z_scores,
        p_values,
        se_inflation_factors,
        beta_tildes_phewas,
        p_values_phewas,
    )


def _apply_prefilter_and_record(
    runtime_state,
    cur_X,
    gene_sets,
    p_value_mask,
    filter_gene_set_p,
    filter_gene_set_metric_z,
    scale_factors,
    mean_shifts,
    beta_tildes,
    p_values,
    ses,
    z_scores,
    se_inflation_factors,
    total_qc_metrics,
    mean_qc_metrics,
    cur_X_missing_genes_new,
    cur_X_missing_genes_int,
):
    p_value_ignore = np.full(len(p_value_mask), False)
    gene_ignored_N = None
    gene_ignored_N_missing_new = None
    gene_ignored_N_missing_int = None

    if filter_gene_set_p < 1 or filter_gene_set_metric_z is not None:
        p_value_ignore = ~p_value_mask
        if np.sum(p_value_ignore) > 0:
            log("Kept %d gene sets after p-value and beta filters" % (np.sum(p_value_mask)))

        runtime_state.gene_sets_ignored = runtime_state.gene_sets_ignored + [gene_sets[i] for i in range(len(gene_sets)) if p_value_ignore[i]]
        gene_sets = [gene_sets[i] for i in range(len(gene_sets)) if p_value_mask[i]]

        runtime_state.col_sums_ignored = np.append(runtime_state.col_sums_ignored, runtime_state.get_col_sums(cur_X[:, p_value_ignore]))
        runtime_state.scale_factors_ignored = np.append(runtime_state.scale_factors_ignored, scale_factors[p_value_ignore])
        runtime_state.mean_shifts_ignored = np.append(runtime_state.mean_shifts_ignored, mean_shifts[p_value_ignore])
        runtime_state.beta_tildes_ignored = np.append(runtime_state.beta_tildes_ignored, beta_tildes[p_value_ignore])
        runtime_state.p_values_ignored = np.append(runtime_state.p_values_ignored, p_values[p_value_ignore])
        runtime_state.ses_ignored = np.append(runtime_state.ses_ignored, ses[p_value_ignore])
        runtime_state.z_scores_ignored = np.append(runtime_state.z_scores_ignored, z_scores[p_value_ignore])

        runtime_state.beta_tildes = np.append(runtime_state.beta_tildes, beta_tildes[p_value_mask])
        runtime_state.p_values = np.append(runtime_state.p_values, p_values[p_value_mask])
        runtime_state.ses = np.append(runtime_state.ses, ses[p_value_mask])
        runtime_state.z_scores = np.append(runtime_state.z_scores, z_scores[p_value_mask])

        if se_inflation_factors is not None:
            runtime_state.se_inflation_factors_ignored = np.append(
                runtime_state.se_inflation_factors_ignored,
                se_inflation_factors[p_value_ignore],
            )
            if runtime_state.se_inflation_factors is None:
                runtime_state.se_inflation_factors = np.array([])
            runtime_state.se_inflation_factors = np.append(
                runtime_state.se_inflation_factors,
                se_inflation_factors[p_value_mask],
            )

        if runtime_state.gene_covariates is not None:
            if runtime_state.total_qc_metrics_ignored is None:
                runtime_state.total_qc_metrics_ignored = total_qc_metrics[p_value_ignore, :]
                runtime_state.mean_qc_metrics_ignored = mean_qc_metrics[p_value_ignore]
            else:
                runtime_state.total_qc_metrics_ignored = np.vstack((runtime_state.total_qc_metrics_ignored, total_qc_metrics[p_value_ignore, :]))
                runtime_state.mean_qc_metrics_ignored = np.append(runtime_state.mean_qc_metrics_ignored, mean_qc_metrics[p_value_ignore])

            total_qc_metrics = total_qc_metrics[p_value_mask]
            mean_qc_metrics = mean_qc_metrics[p_value_mask]

        # need to record how many ignored
        gene_ignored_N = runtime_state.get_col_sums(cur_X[:, p_value_ignore], axis=1)

        if cur_X_missing_genes_new is not None:
            gene_ignored_N_missing_new = np.array(np.abs(cur_X_missing_genes_new[:, p_value_ignore]).sum(axis=1)).flatten()
            cur_X_missing_genes_new = cur_X_missing_genes_new[:, p_value_mask]

        if cur_X_missing_genes_int is not None:
            gene_ignored_N_missing_int = np.array(np.abs(cur_X_missing_genes_int[:, p_value_ignore]).sum(axis=1)).flatten()
            cur_X_missing_genes_int = cur_X_missing_genes_int[:, p_value_mask]

        cur_X = cur_X[:, p_value_mask]

    return (
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
    )


def _maybe_prefilter_x_block(
    runtime_state,
    cur_X,
    gene_sets,
    run_logistic,
    filter_gene_set_p,
    filter_gene_set_metric_z,
    filter_using_phewas,
    increase_filter_gene_set_p,
    filter_negative,
    cur_X_missing_genes_new,
    gene_ignored_N_missing_new,
    cur_X_missing_genes_int,
    gene_ignored_N_missing_int,
    gene_ignored_N,
):
    p_value_ignore = None
    total_qc_metrics = None
    mean_qc_metrics = None
    total_qc_metrics_directions = None

    if (filter_gene_set_p < 1 or filter_gene_set_metric_z is not None) and runtime_state.Y is not None:
        log("Analyzing gene sets to pre-filter")

        (mean_shifts, scale_factors) = runtime_state._calc_X_shift_scale(cur_X)

        (
            total_qc_metrics,
            mean_qc_metrics,
            total_qc_metrics_directions,
        ) = _compute_prefilter_qc_metrics(runtime_state, cur_X)

        (
            beta_tildes,
            ses,
            z_scores,
            p_values,
            se_inflation_factors,
            beta_tildes_phewas,
            p_values_phewas,
        ) = _compute_prefilter_assoc_stats(
            runtime_state,
            cur_X=cur_X,
            run_logistic=run_logistic,
            filter_using_phewas=filter_using_phewas,
            mean_shifts=mean_shifts,
            scale_factors=scale_factors,
        )

        cur_X, beta_tildes, z_scores = _align_prefilter_gene_set_signs(
            cur_X,
            beta_tildes=beta_tildes,
            z_scores=z_scores,
        )

        p_value_mask = _build_prefilter_keep_mask(
            p_values,
            beta_tildes=beta_tildes,
            filter_gene_set_p=filter_gene_set_p,
            filter_using_phewas=filter_using_phewas,
            p_values_phewas=p_values_phewas if filter_using_phewas else None,
            beta_tildes_phewas=beta_tildes_phewas if filter_using_phewas else None,
            increase_filter_gene_set_p=increase_filter_gene_set_p,
            filter_negative=filter_negative,
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
        ) = _apply_prefilter_and_record(
            runtime_state,
            cur_X=cur_X,
            gene_sets=gene_sets,
            p_value_mask=p_value_mask,
            filter_gene_set_p=filter_gene_set_p,
            filter_gene_set_metric_z=filter_gene_set_metric_z,
            scale_factors=scale_factors,
            mean_shifts=mean_shifts,
            beta_tildes=beta_tildes,
            p_values=p_values,
            ses=ses,
            z_scores=z_scores,
            se_inflation_factors=se_inflation_factors,
            total_qc_metrics=total_qc_metrics,
            mean_qc_metrics=mean_qc_metrics,
            cur_X_missing_genes_new=cur_X_missing_genes_new,
            cur_X_missing_genes_int=cur_X_missing_genes_int,
        )

    return (
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
    )


def _merge_missing_gene_rows(
    runtime_state,
    cur_X,
    genes,
    num_old_gene_sets,
    num_new_gene_sets,
    cur_X_missing_genes_int,
    gene_ignored_N_missing_int,
    cur_X_missing_genes_new,
    gene_ignored_N_missing_new,
    genes_missing_new,
):
    if runtime_state.genes_missing is not None:
        genes += runtime_state.genes_missing

        if runtime_state.X_orig_missing_genes is None:
            X_orig_missing_genes = sparse.csc_matrix(([], ([], [])), shape=(len(runtime_state.genes_missing), num_old_gene_sets))
        else:
            X_orig_missing_genes = copy.copy(runtime_state.X_orig_missing_genes)

        if cur_X_missing_genes_int is not None:
            if runtime_state.gene_ignored_N_missing is not None:
                if gene_ignored_N_missing_int is not None:
                    runtime_state.gene_ignored_N_missing += gene_ignored_N_missing_int
            else:
                runtime_state.gene_ignored_N_missing = gene_ignored_N_missing_int

            cur_X = sparse.vstack((cur_X, sparse.hstack((X_orig_missing_genes, cur_X_missing_genes_int))))
        elif X_orig_missing_genes is not None:
            X_orig_missing_genes.resize((X_orig_missing_genes.shape[0], X_orig_missing_genes.shape[1] + num_new_gene_sets))
            cur_X = sparse.vstack((cur_X, X_orig_missing_genes))

    if cur_X_missing_genes_new is not None:
        cur_X = sparse.vstack(
            (
                cur_X,
                sparse.hstack(
                    (
                        sparse.csc_matrix(([], ([], [])), shape=(cur_X_missing_genes_new.shape[0], num_old_gene_sets)),
                        cur_X_missing_genes_new,
                    )
                ),
            )
        )
        if runtime_state.gene_ignored_N_missing is not None:
            if gene_ignored_N_missing_new is not None:
                runtime_state.gene_ignored_N_missing = np.append(runtime_state.gene_ignored_N_missing, gene_ignored_N_missing_new)
        else:
            runtime_state.gene_ignored_N_missing = gene_ignored_N_missing_new

        genes += genes_missing_new

    return (cur_X, genes)


def _finalize_added_x_block(
    runtime_state,
    cur_X,
    genes,
    gene_sets,
    skip_scale_factors,
    p_value_ignore,
    gene_ignored_N,
    total_qc_metrics,
    mean_qc_metrics,
    total_qc_metrics_directions,
):
    # Save subset mask for later.
    subset_mask = np.full(len(genes), True)
    if runtime_state.gene_to_ind is not None:
        subset_mask[[i for i in range(len(genes)) if genes[i] not in runtime_state.gene_to_ind]] = False

    # Set full X including new and old missing genes.
    num_added = cur_X.shape[1]
    if runtime_state.X_orig is not None:
        num_added -= runtime_state.X_orig.shape[1]
    num_ignored = np.sum(p_value_ignore) if p_value_ignore is not None else 0

    runtime_state._set_X(
        sparse.csc_matrix(cur_X, shape=cur_X.shape),
        genes,
        gene_sets,
        skip_scale_factors=skip_scale_factors,
        skip_V=True,
        skip_N=False,
    )

    # Add ignored_N since this is the only place we have the information.
    if runtime_state.gene_ignored_N is not None:
        if gene_ignored_N is not None:
            runtime_state.gene_ignored_N += gene_ignored_N
    else:
        runtime_state.gene_ignored_N = gene_ignored_N

    if runtime_state.gene_ignored_N is not None and runtime_state.gene_ignored_N_missing is not None:
        runtime_state.gene_ignored_N = np.append(runtime_state.gene_ignored_N, runtime_state.gene_ignored_N_missing)

    # Ensure every data structure gets subsetted (without subsetting Y here).
    runtime_state._subset_genes(
        subset_mask,
        skip_V=True,
        overwrite_missing=True,
        skip_scale_factors=False,
        skip_Y=True,
    )

    if runtime_state.gene_covariates is not None:
        if runtime_state.total_qc_metrics is None:
            runtime_state.total_qc_metrics = total_qc_metrics
            runtime_state.mean_qc_metrics = mean_qc_metrics
        else:
            runtime_state.total_qc_metrics = np.vstack((runtime_state.total_qc_metrics, total_qc_metrics))
            runtime_state.mean_qc_metrics = np.append(runtime_state.mean_qc_metrics, mean_qc_metrics)

        runtime_state.total_qc_metrics_directions = total_qc_metrics_directions

    return (num_added, num_ignored)


def _partition_missing_gene_rows(runtime_state, cur_X, genes, gene_sets):
    gene_ignored_N = None

    # New missing that overlap historical missing genes.
    cur_X_missing_genes_int = None
    gene_ignored_N_missing_int = None

    # New missing that are newly introduced in this call.
    genes_missing_new = []
    cur_X_missing_genes_new = None
    gene_ignored_N_missing_new = None

    if (runtime_state.Y is not None and len(genes) > len(runtime_state.Y)) or (runtime_state.genes is not None):
        genes_missing_old = runtime_state.genes_missing if runtime_state.genes_missing is not None else []
        gene_missing_old_to_ind = pegs_construct_map_to_ind(genes_missing_old)

        # Genes newly missing this round.
        genes_missing_new = [x for x in genes if x not in runtime_state.gene_to_ind and x not in gene_missing_old_to_ind]
        genes_missing_new_set = set(genes_missing_new)

        # Missing genes shared with previous rounds.
        genes_missing_int_set = set([x for x in genes if x in gene_missing_old_to_ind])

        int_mask = np.full(len(genes), False)
        int_mask[[i for i in range(len(genes)) if genes[i] in genes_missing_int_set]] = True
        if np.sum(int_mask) > 0:
            cur_X_missing_genes_int = cur_X[int_mask, :]

        new_mask = np.full(len(genes), False)
        new_mask[[i for i in range(len(genes)) if genes[i] in genes_missing_new_set]] = True
        if np.sum(new_mask) > 0:
            cur_X_missing_genes_new = cur_X[new_mask, :]

        subset_mask = np.full(len(genes), True)
        subset_mask[[i for i in range(len(genes)) if genes[i] not in runtime_state.gene_to_ind]] = False

        cur_X = cur_X[subset_mask, :]
        genes = [x for x in genes if x in runtime_state.gene_to_ind]

        # Remove empty gene sets.
        gene_set_nonempty_mask = runtime_state.get_col_sums(cur_X) > 0
        if np.sum(~gene_set_nonempty_mask) > 0:
            cur_X = cur_X[:, gene_set_nonempty_mask]

            if cur_X_missing_genes_int is not None:
                cur_X_missing_genes_int = cur_X_missing_genes_int[:, gene_set_nonempty_mask]
            if cur_X_missing_genes_new is not None:
                cur_X_missing_genes_new = cur_X_missing_genes_new[:, gene_set_nonempty_mask]
            gene_sets = [gene_sets[i] for i in range(len(gene_sets)) if gene_set_nonempty_mask[i]]

        if runtime_state.Y is not None:
            assert len(genes) == len(runtime_state.Y)

        if cur_X.shape[1] == 0:
            bail("Error: no genes overlapped Y and X; you may have forgotten to map gene names over to a common namespace")

    return (
        cur_X,
        genes,
        gene_sets,
        gene_ignored_N,
        cur_X_missing_genes_int,
        gene_ignored_N_missing_int,
        genes_missing_new,
        cur_X_missing_genes_new,
        gene_ignored_N_missing_new,
    )


def _reindex_x_rows_to_current_genes(runtime_state, cur_X, genes):
    if runtime_state.genes is None:
        return (cur_X, genes)

    # Reorder rows to align with already-initialized runtime gene order and
    # append previously unseen genes at the end.
    old_genes = genes
    genes = runtime_state.genes
    if runtime_state.genes_missing is not None:
        genes += runtime_state.genes_missing
    genes += [
        x
        for x in old_genes
        if (runtime_state.gene_to_ind is None or x not in runtime_state.gene_to_ind)
        and (runtime_state.gene_missing_to_ind is None or x not in runtime_state.gene_missing_to_ind)
    ]
    gene_to_ind = pegs_construct_map_to_ind(genes)
    index_map = {i: gene_to_ind[old_genes[i]] for i in range(len(old_genes))}
    cur_X = sparse.csc_matrix(
        (cur_X.data, [index_map[x] for x in cur_X.indices], cur_X.indptr),
        shape=(len(genes), cur_X.shape[1]),
    )
    return (cur_X, genes)


def _ensure_gene_universe_for_x(
    runtime_state,
    X_ins,
    is_dense,
    add_all_genes,
    only_ids,
    only_inc_genes,
    fraction_inc_genes,
):
    if runtime_state.genes is None or add_all_genes:
        if runtime_state.genes is None:
            log("No genes initialized before reading X: constructing gene list from union of all files", DEBUG)

        # need to set it to the union of all genes
        all_genes = []
        gene_counts = {}
        num_gene_sets = 0
        for i in range(len(X_ins)):
            X_in = X_ins[i]
            (X_in, tag) = pegs_remove_tag_from_input(X_in)

            if is_dense[i]:
                with open_gz(X_in) as gene_sets_fh:
                    num_in_file = None
                    for line in gene_sets_fh:
                        line = line.strip('\n')
                        cols = line.split()
                        if num_in_file is None:
                            num_in_file = len(cols) - 1
                            num_gene_sets += num_in_file
                        elif len(cols) - 1 != num_in_file:
                            bail("Not a square matrix!")

                        if len(cols) > 0:
                            all_genes += cols[0]
                        if cols[0] not in gene_counts:
                            gene_counts[cols[0]] = 0
                        gene_counts[cols[0]] += num_in_file
            else:
                with open_gz(X_in) as gene_sets_fh:
                    it = 0
                    for line in gene_sets_fh:
                        line = line.strip('\n')
                        cols = line.split()
                        if len(cols) < 2:
                            continue

                        cur_genes = set(cols[1:])

                        if only_ids is not None and cols[0] not in only_ids:
                            continue

                        if ":" in line:
                            cur_genes = [gene.split(":")[0] for gene in cur_genes]
                        if runtime_state.gene_label_map is not None:
                            cur_genes = set(map(lambda x: runtime_state.gene_label_map[x] if x in runtime_state.gene_label_map else x, cur_genes))

                        if not add_all_genes and only_inc_genes is not None:
                            fraction_match = len(only_inc_genes.intersection(cur_genes)) / float(len(only_inc_genes))
                            if fraction_match < (fraction_inc_genes if fraction_inc_genes is not None else 1e-5):
                                continue

                        all_genes += cur_genes
                        for gene in cur_genes:
                            if gene not in gene_counts:
                                gene_counts[gene] = 0
                            gene_counts[gene] += 1

                        num_gene_sets += 1
                        it += 1
                        if it % 1000 == 0:
                            all_genes = list(set(all_genes))

            all_genes = list(set(all_genes))

        if runtime_state.genes is not None:
            add_genes = [x for x in all_genes if x not in runtime_state.gene_to_ind]
            log("Adding an additional %d genes from gene sets not in input Y values" % len(add_genes), DEBUG)
            all_genes = runtime_state.genes + add_genes
            new_Y = runtime_state.Y
            if new_Y is not None:
                assert(len(new_Y) == len(runtime_state.genes))
                new_Y = np.append(new_Y, np.zeros(len(add_genes)))
            new_Y_for_regression = runtime_state.Y_for_regression
            if new_Y_for_regression is not None:
                assert(len(new_Y_for_regression) == len(runtime_state.genes))
                new_Y_for_regression = np.append(new_Y_for_regression, np.zeros(len(add_genes)))
            new_Y_exomes = runtime_state.Y_exomes
            if new_Y_exomes is not None:
                assert(len(new_Y_exomes) == len(runtime_state.genes))
                new_Y_exomes = np.append(new_Y_exomes, np.zeros(len(add_genes)))
            new_Y_positive_controls = runtime_state.Y_positive_controls
            if new_Y_positive_controls is not None:
                assert(len(new_Y_positive_controls) == len(runtime_state.genes))
                new_Y_positive_controls = np.append(new_Y_positive_controls, np.zeros(len(add_genes)))

            new_Y_case_counts = runtime_state.Y_case_counts
            if new_Y_case_counts is not None:
                assert(len(new_Y_case_counts) == len(runtime_state.genes))
                new_Y_case_counts = np.append(new_Y_case_counts, np.zeros(len(add_genes)))

            runtime_state._set_Y(new_Y, new_Y_for_regression, new_Y_exomes, new_Y_positive_controls, new_Y_case_counts)

        # really calling this just to set the genes
        runtime_state._set_X(runtime_state.X_orig, list(all_genes), runtime_state.gene_sets, skip_N=False)


def _align_extra_genes_with_new_source(
    existing_extra_genes,
    existing_extra_values,
    new_source_genes,
    new_source_values,
    existing_missing_values,
    new_source_missing_value,
):
    new_source_gene_to_ind = pegs_construct_map_to_ind(new_source_genes)
    merged_extra_genes = list(new_source_genes)
    merged_new_source_values = list(new_source_values)
    aligned_existing_values = [
        list(np.full(len(merged_new_source_values), missing_value))
        for missing_value in existing_missing_values
    ]

    for i in range(len(existing_extra_genes)):
        gene = existing_extra_genes[i]
        if gene in new_source_gene_to_ind:
            source_ind = new_source_gene_to_ind[gene]
            for j in range(len(existing_extra_values)):
                aligned_existing_values[j][source_ind] = existing_extra_values[j][i]
        else:
            merged_extra_genes.append(gene)
            for j in range(len(existing_extra_values)):
                aligned_existing_values[j].append(existing_extra_values[j][i])
            merged_new_source_values.append(new_source_missing_value)

    return (
        merged_extra_genes,
        [np.array(x) for x in aligned_existing_values],
        np.array(merged_new_source_values),
    )


def _apply_hold_out_chrom_to_y(runtime_state, Y, extra_genes, extra_Y, hold_out_chrom, gene_loc_file):
    if hold_out_chrom is None:
        return (Y, extra_genes, extra_Y)

    if runtime_state.gene_to_chrom is None:
        (
            runtime_state.gene_chrom_name_pos,
            runtime_state.gene_to_chrom,
            runtime_state.gene_to_pos,
        ) = pegs_read_loc_file_with_gene_map(
            gene_loc_file,
            gene_label_map=runtime_state.gene_label_map,
            clean_chrom_fn=pegs_clean_chrom_name,
            warn_fn=warn,
            bail_fn=bail,
        )

    extra_Y_mask = np.full(len(extra_Y), True)
    for i in range(len(extra_genes)):
        if extra_genes[i] in runtime_state.gene_to_chrom and runtime_state.gene_to_chrom[extra_genes[i]] == hold_out_chrom:
            extra_Y_mask[i] = False
    if np.sum(~extra_Y_mask) > 0:
        extra_genes = [extra_genes[i] for i in range(len(extra_genes)) if extra_Y_mask[i]]
        extra_Y = extra_Y[extra_Y_mask]

    if runtime_state.genes is not None:
        Y_nan_mask = np.full(len(Y), False)
        for i in range(len(runtime_state.genes)):
            if runtime_state.genes[i] in runtime_state.gene_to_chrom and runtime_state.gene_to_chrom[runtime_state.genes[i]] == hold_out_chrom:
                Y_nan_mask[i] = True
        if np.sum(Y_nan_mask) > 0:
            Y[Y_nan_mask] = np.nan

    return (Y, extra_genes, extra_Y)


def _get_col(col_name_or_index, header_cols, require_match=True):
    return pegs_resolve_column_index(
        col_name_or_index,
        header_cols,
        require_match=require_match,
        bail_fn=bail,
    )


def _determine_columns_from_file(filename):
    return pegs_infer_columns_from_table_file(
        filename,
        open_gz,
        log_fn=lambda message: log(message),
        bail_fn=bail,
    )


def _needs_gwas_column_detection(
    gwas_pos_col,
    gwas_chrom_col,
    gwas_locus_col,
    gwas_p_col,
    gwas_beta_col,
    gwas_se_col,
    gwas_n_col,
    gwas_n,
):
    try:
        from . import pigean_huge as _pigean_huge
    except ImportError:
        import pigean_huge as _pigean_huge

    return _pigean_huge.needs_gwas_column_detection(
        sys.modules[__name__],
        gwas_pos_col,
        gwas_chrom_col,
        gwas_locus_col,
        gwas_p_col,
        gwas_beta_col,
        gwas_se_col,
        gwas_n_col,
        gwas_n,
    )


def _autodetect_gwas_columns(
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
    debug_just_check_header=False,
):
    try:
        from . import pigean_huge as _pigean_huge
    except ImportError:
        import pigean_huge as _pigean_huge

    return _pigean_huge.autodetect_gwas_columns(
        sys.modules[__name__],
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
        debug_just_check_header=debug_just_check_header,
    )


class _IntervalTree(object):
    def __new__(cls, *args, **kwargs):
        try:
            from . import pigean_huge as _pigean_huge
        except ImportError:
            import pigean_huge as _pigean_huge

        _pigean_huge.configure_numpy(np)
        return _pigean_huge.IntervalTree(*args, **kwargs)


def _load_huge_gene_and_exon_locations(gene_loc_file, gene_label_map, hold_out_chrom=None, exons_loc_file=None):
    try:
        from . import pigean_huge as _pigean_huge
    except ImportError:
        import pigean_huge as _pigean_huge

    _pigean_huge.configure_numpy(np)
    return _pigean_huge.load_huge_gene_and_exon_locations(
        sys.modules[__name__],
        gene_loc_file,
        gene_label_map,
        hold_out_chrom=hold_out_chrom,
        exons_loc_file=exons_loc_file,
    )


def _compute_huge_variant_thresholds(min_var_posterior, gwas_high_p_posterior, allelic_var_k, gwas_prior_odds):
    try:
        from . import pigean_huge as _pigean_huge
    except ImportError:
        import pigean_huge as _pigean_huge

    return _pigean_huge.compute_huge_variant_thresholds(
        sys.modules[__name__],
        min_var_posterior,
        gwas_high_p_posterior,
        allelic_var_k,
        gwas_prior_odds,
    )


def _validate_and_normalize_huge_gwas_inputs(
    gwas_in,
    gene_loc_file,
    credible_sets_in=None,
    credible_sets_chrom_col=None,
    credible_sets_pos_col=None,
    signal_window_size=250000,
    signal_min_sep=100000,
    signal_max_logp_ratio=None,
):
    try:
        from . import pigean_huge as _pigean_huge
    except ImportError:
        import pigean_huge as _pigean_huge

    return _pigean_huge.validate_and_normalize_huge_gwas_inputs(
        sys.modules[__name__],
        gwas_in,
        gene_loc_file,
        credible_sets_in=credible_sets_in,
        credible_sets_chrom_col=credible_sets_chrom_col,
        credible_sets_pos_col=credible_sets_pos_col,
        signal_window_size=signal_window_size,
        signal_min_sep=signal_min_sep,
        signal_max_logp_ratio=signal_max_logp_ratio,
    )


def _write_huge_statistics_bundle(runtime_state, prefix, gene_bf, extra_genes, extra_gene_bf, gene_bf_for_regression, extra_gene_bf_for_regression):
    try:
        from . import pigean_huge as _pigean_huge
    except ImportError:
        import pigean_huge as _pigean_huge

    return _pigean_huge.write_huge_statistics_bundle(
        sys.modules[__name__],
        runtime_state,
        prefix,
        gene_bf,
        extra_genes,
        extra_gene_bf,
        gene_bf_for_regression,
        extra_gene_bf_for_regression,
    )


def _read_huge_statistics_bundle(runtime_state, prefix):
    try:
        from . import pigean_huge as _pigean_huge
    except ImportError:
        import pigean_huge as _pigean_huge

    return _pigean_huge.read_huge_statistics_bundle(
        sys.modules[__name__],
        runtime_state,
        prefix,
    )

# ==========================================================================
# Gibbs epoch aggregation helpers.
# ==========================================================================
_GIBBS_EPOCH_SUM_KEYS = (
    "sum_betas_m",
    "sum_betas2_m",
    "sum_betas_uncorrected_m",
    "sum_betas_uncorrected2_m",
    "sum_postp_m",
    "sum_beta_tildes_m",
    "sum_z_scores_m",
    "num_sum_beta_m",
    "sum_Ys_m",
    "sum_Y_raws_m",
    "sum_log_pos_m",
    "sum_log_po_raws_m",
    "sum_log_po_raws2_m",
    "sum_priors_m",
    "sum_priors2_m",
    "sum_Ds_m",
    "sum_D_raws_m",
    "sum_bf_orig_m",
    "sum_bf_orig_raw_m",
    "sum_bf_orig_raw2_m",
    "num_sum_Y_m",
)

_GIBBS_EPOCH_MISSING_SUM_KEYS = (
    "sum_priors_missing_m",
    "sum_Ds_missing_m",
    "num_sum_priors_missing_m",
)

_GIBBS_EPOCH_RUNTIME_SUM_KEYS = (
    "all_sum_betas_m",
    "all_sum_betas2_m",
    "all_num_sum_m",
)


def _new_gibbs_epoch_aggregates():
    epoch_aggregates = {}
    for key in _GIBBS_EPOCH_SUM_KEYS + _GIBBS_EPOCH_MISSING_SUM_KEYS:
        epoch_aggregates[key] = []
    return epoch_aggregates


def _has_gibbs_epoch_aggregates(epoch_aggregates):
    return len(epoch_aggregates["sum_betas_m"]) > 0


def _build_gibbs_diag_sums(epoch_aggregates, sum_betas_m, sum_betas2_m, num_sum_beta_m, sum_Ds_m, num_sum_Y_m):
    if not _has_gibbs_epoch_aggregates(epoch_aggregates):
        return (sum_betas_m, sum_betas2_m, num_sum_beta_m, sum_Ds_m, num_sum_Y_m)
    diag_sum_betas_m = np.vstack(epoch_aggregates["sum_betas_m"] + [sum_betas_m])
    diag_sum_betas2_m = np.vstack(epoch_aggregates["sum_betas2_m"] + [sum_betas2_m])
    diag_num_sum_beta_m = np.vstack(epoch_aggregates["num_sum_beta_m"] + [num_sum_beta_m])
    diag_sum_Ds_m = np.vstack(epoch_aggregates["sum_Ds_m"] + [sum_Ds_m])
    diag_num_sum_Y_m = np.vstack(epoch_aggregates["num_sum_Y_m"] + [num_sum_Y_m])
    return (diag_sum_betas_m, diag_sum_betas2_m, diag_num_sum_beta_m, diag_sum_Ds_m, diag_num_sum_Y_m)


# ========================= Outer Gibbs Control Normalization =========================
def _normalize_gibbs_epoch_iteration_controls(
    max_num_iter,
    min_num_burn_in,
    max_num_burn_in,
    min_num_post_burn_in,
    max_num_post_burn_in,
):
    if min_num_burn_in is None:
        min_num_burn_in = 0
    if min_num_burn_in < 0:
        min_num_burn_in = 0

    if min_num_post_burn_in is None:
        min_num_post_burn_in = 1
    if min_num_post_burn_in < 1:
        min_num_post_burn_in = 1

    if max_num_burn_in is None:
        if max_num_iter is not None and max_num_iter > 0:
            max_num_burn_in = int(np.ceil(max_num_iter * 0.8))
        else:
            max_num_burn_in = min_num_burn_in
        if max_num_burn_in < 1:
            max_num_burn_in = 1
    if max_num_burn_in < 1:
        max_num_burn_in = 1
    if max_num_burn_in < min_num_burn_in:
        max_num_burn_in = min_num_burn_in

    if max_num_post_burn_in is None:
        if max_num_iter is not None and max_num_iter > 0:
            max_num_post_burn_in = max_num_iter - min_num_burn_in
        else:
            max_num_post_burn_in = min_num_post_burn_in
    if max_num_post_burn_in < 1:
        max_num_post_burn_in = 1
    if max_num_post_burn_in < min_num_post_burn_in:
        max_num_post_burn_in = min_num_post_burn_in

    passed_in_max_num_burn_in = max_num_burn_in
    epoch_max_num_iter_config = max_num_burn_in + max_num_post_burn_in
    if epoch_max_num_iter_config < 2:
        epoch_max_num_iter_config = 2

    return {
        "min_num_burn_in": min_num_burn_in,
        "max_num_burn_in": max_num_burn_in,
        "min_num_post_burn_in": min_num_post_burn_in,
        "max_num_post_burn_in": max_num_post_burn_in,
        "passed_in_max_num_burn_in": passed_in_max_num_burn_in,
        "epoch_max_num_iter_config": epoch_max_num_iter_config,
    }


def _sanitize_gibbs_diagnostic_controls(
    num_chains,
    diag_every,
    burn_in_patience,
    burn_in_stall_window,
    burn_in_stall_delta,
    stop_patience,
    stop_top_gene_k,
    stop_min_gene_d,
    active_beta_top_k,
    active_beta_min_abs,
    beta_rel_mcse_denom_floor,
    stall_window,
    stall_min_burn_in,
    stall_min_post_burn_in,
    stall_delta_rhat,
    stall_delta_mcse,
    stall_recent_window,
    stall_recent_eps,
    burn_in_rhat_quantile,
    use_max_r_for_convergence,
):
    if num_chains < 2:
        num_chains = 2
    if diag_every < 1:
        diag_every = 1
    if burn_in_patience < 1:
        burn_in_patience = 1
    if burn_in_stall_window is None or burn_in_stall_window < 2:
        burn_in_stall_window = 0
    if burn_in_stall_delta is None or burn_in_stall_delta < 0:
        burn_in_stall_delta = 0
    if stop_patience < 1:
        stop_patience = 1
    if stop_top_gene_k < 1:
        stop_top_gene_k = 1
    if stop_min_gene_d is not None:
        if stop_min_gene_d < 0:
            stop_min_gene_d = 0
        if stop_min_gene_d > 1:
            stop_min_gene_d = 1
    if active_beta_top_k < 1:
        active_beta_top_k = 1
    if active_beta_min_abs < 0:
        active_beta_min_abs = 0
    if beta_rel_mcse_denom_floor <= 0:
        beta_rel_mcse_denom_floor = 1e-12
    if stall_window is None or stall_window < 2:
        stall_window = 0
    if stall_min_burn_in is None or stall_min_burn_in < 0:
        stall_min_burn_in = 0
    if stall_min_post_burn_in is None or stall_min_post_burn_in < 0:
        stall_min_post_burn_in = 0
    if stall_delta_rhat is None or stall_delta_rhat < 0:
        stall_delta_rhat = 0
    if stall_delta_mcse is None or stall_delta_mcse < 0:
        stall_delta_mcse = 0
    if stall_recent_window is None or stall_recent_window < 2:
        stall_recent_window = 0
    if stall_recent_eps is None or stall_recent_eps < 0:
        stall_recent_eps = 0
    if burn_in_rhat_quantile is None:
        burn_in_rhat_quantile = 1.0
    if use_max_r_for_convergence:
        burn_in_rhat_quantile = 1.0
    if burn_in_rhat_quantile < 0:
        burn_in_rhat_quantile = 0
    elif burn_in_rhat_quantile > 1:
        burn_in_rhat_quantile = 1

    return {
        "num_chains": num_chains,
        "diag_every": diag_every,
        "burn_in_patience": burn_in_patience,
        "burn_in_stall_window": burn_in_stall_window,
        "burn_in_stall_delta": burn_in_stall_delta,
        "stop_patience": stop_patience,
        "stop_top_gene_k": stop_top_gene_k,
        "stop_min_gene_d": stop_min_gene_d,
        "active_beta_top_k": active_beta_top_k,
        "active_beta_min_abs": active_beta_min_abs,
        "beta_rel_mcse_denom_floor": beta_rel_mcse_denom_floor,
        "stall_window": stall_window,
        "stall_min_burn_in": stall_min_burn_in,
        "stall_min_post_burn_in": stall_min_post_burn_in,
        "stall_delta_rhat": stall_delta_rhat,
        "stall_delta_mcse": stall_delta_mcse,
        "stall_recent_window": stall_recent_window,
        "stall_recent_eps": stall_recent_eps,
        "burn_in_rhat_quantile": burn_in_rhat_quantile,
    }


def _normalize_gibbs_run_controls(
    max_num_iter,
    total_num_iter,
    max_num_restarts,
    num_chains,
    min_num_burn_in,
    max_num_burn_in,
    min_num_post_burn_in,
    max_num_post_burn_in,
    diag_every,
    burn_in_patience,
    burn_in_stall_window,
    burn_in_stall_delta,
    stop_patience,
    stop_top_gene_k,
    stop_min_gene_d,
    active_beta_top_k,
    active_beta_min_abs,
    beta_rel_mcse_denom_floor,
    stall_window,
    stall_min_burn_in,
    stall_min_post_burn_in,
    stall_delta_rhat,
    stall_delta_mcse,
    stall_recent_window,
    stall_recent_eps,
    burn_in_rhat_quantile,
    use_max_r_for_convergence,
):
    if max_num_restarts is None or max_num_restarts < 0:
        max_num_restarts = 0
    target_num_epochs = max_num_restarts + 1

    normalized_epoch_controls = _normalize_gibbs_epoch_iteration_controls(
        max_num_iter=max_num_iter,
        min_num_burn_in=min_num_burn_in,
        max_num_burn_in=max_num_burn_in,
        min_num_post_burn_in=min_num_post_burn_in,
        max_num_post_burn_in=max_num_post_burn_in,
    )
    min_num_burn_in = normalized_epoch_controls["min_num_burn_in"]
    max_num_burn_in = normalized_epoch_controls["max_num_burn_in"]
    min_num_post_burn_in = normalized_epoch_controls["min_num_post_burn_in"]
    max_num_post_burn_in = normalized_epoch_controls["max_num_post_burn_in"]
    epoch_max_num_iter_config = normalized_epoch_controls["epoch_max_num_iter_config"]

    if total_num_iter is None:
        total_num_iter = epoch_max_num_iter_config
    if total_num_iter < 1:
        total_num_iter = 1
    run_state = _initialize_gibbs_run_state(total_num_iter, target_num_epochs, max_num_restarts)

    sanitized_diag_controls = _sanitize_gibbs_diagnostic_controls(
        num_chains=num_chains,
        diag_every=diag_every,
        burn_in_patience=burn_in_patience,
        burn_in_stall_window=burn_in_stall_window,
        burn_in_stall_delta=burn_in_stall_delta,
        stop_patience=stop_patience,
        stop_top_gene_k=stop_top_gene_k,
        stop_min_gene_d=stop_min_gene_d,
        active_beta_top_k=active_beta_top_k,
        active_beta_min_abs=active_beta_min_abs,
        beta_rel_mcse_denom_floor=beta_rel_mcse_denom_floor,
        stall_window=stall_window,
        stall_min_burn_in=stall_min_burn_in,
        stall_min_post_burn_in=stall_min_post_burn_in,
        stall_delta_rhat=stall_delta_rhat,
        stall_delta_mcse=stall_delta_mcse,
        stall_recent_window=stall_recent_window,
        stall_recent_eps=stall_recent_eps,
        burn_in_rhat_quantile=burn_in_rhat_quantile,
        use_max_r_for_convergence=use_max_r_for_convergence,
    )

    (
        _,
        first_min_num_burn_in,
        first_max_num_burn_in,
        first_min_num_post_burn_in,
        first_max_num_post_burn_in,
    ) = _resolve_epoch_iteration_budget(
        total_num_iter,
        epoch_max_num_iter_config,
        min_num_burn_in,
        max_num_burn_in,
        min_num_post_burn_in,
        max_num_post_burn_in,
    )

    return GibbsRunControls(
        max_num_restarts=max_num_restarts,
        target_num_epochs=target_num_epochs,
        min_num_burn_in=normalized_epoch_controls["min_num_burn_in"],
        max_num_burn_in=normalized_epoch_controls["max_num_burn_in"],
        min_num_post_burn_in=normalized_epoch_controls["min_num_post_burn_in"],
        max_num_post_burn_in=normalized_epoch_controls["max_num_post_burn_in"],
        passed_in_max_num_burn_in=normalized_epoch_controls["passed_in_max_num_burn_in"],
        epoch_max_num_iter_config=normalized_epoch_controls["epoch_max_num_iter_config"],
        total_num_iter=total_num_iter,
        run_state=run_state,
        num_chains=sanitized_diag_controls["num_chains"],
        diag_every=sanitized_diag_controls["diag_every"],
        burn_in_patience=sanitized_diag_controls["burn_in_patience"],
        burn_in_stall_window=sanitized_diag_controls["burn_in_stall_window"],
        burn_in_stall_delta=sanitized_diag_controls["burn_in_stall_delta"],
        stop_patience=sanitized_diag_controls["stop_patience"],
        stop_top_gene_k=sanitized_diag_controls["stop_top_gene_k"],
        stop_min_gene_d=sanitized_diag_controls["stop_min_gene_d"],
        active_beta_top_k=sanitized_diag_controls["active_beta_top_k"],
        active_beta_min_abs=sanitized_diag_controls["active_beta_min_abs"],
        beta_rel_mcse_denom_floor=sanitized_diag_controls["beta_rel_mcse_denom_floor"],
        stall_window=sanitized_diag_controls["stall_window"],
        stall_min_burn_in=sanitized_diag_controls["stall_min_burn_in"],
        stall_min_post_burn_in=sanitized_diag_controls["stall_min_post_burn_in"],
        stall_delta_rhat=sanitized_diag_controls["stall_delta_rhat"],
        stall_delta_mcse=sanitized_diag_controls["stall_delta_mcse"],
        stall_recent_window=sanitized_diag_controls["stall_recent_window"],
        stall_recent_eps=sanitized_diag_controls["stall_recent_eps"],
        burn_in_rhat_quantile=sanitized_diag_controls["burn_in_rhat_quantile"],
        first_min_num_burn_in=first_min_num_burn_in,
        first_max_num_burn_in=first_max_num_burn_in,
        first_min_num_post_burn_in=first_min_num_post_burn_in,
        first_max_num_post_burn_in=first_max_num_post_burn_in,
    )


# ========================= Outer Gibbs Configuration Logging =========================
def _build_gibbs_record_config(
    gibbs_controls,
    num_chains_betas,
    max_num_iter,
    use_mean_betas,
    warm_start,
    stopping_preset_name,
    r_threshold_burn_in,
    stop_mcse_quantile,
    max_abs_mcse_d,
    max_rel_mcse_beta,
    sparse_solution,
    sparse_frac_gibbs,
    sparse_max_gibbs,
    sparse_frac_betas,
    pre_filter_batch_size,
    max_allowed_batch_correlation,
    initial_linear_filter,
    correct_betas_mean,
    correct_betas_var,
    adjust_priors,
    experimental_hyper_mutation,
    increase_hyper_if_betas_below,
):
    return {
        "num_chains": gibbs_controls.num_chains,
        "max_num_restarts": gibbs_controls.max_num_restarts,
        "target_num_epochs": gibbs_controls.target_num_epochs,
        "total_num_iter": gibbs_controls.total_num_iter,
        "epoch_max_num_iter_config": gibbs_controls.epoch_max_num_iter_config,
        "burn_in_rhat_quantile": gibbs_controls.burn_in_rhat_quantile,
        "burn_in_patience": gibbs_controls.burn_in_patience,
        "first_min_num_burn_in": gibbs_controls.first_min_num_burn_in,
        "first_max_num_burn_in": gibbs_controls.first_max_num_burn_in,
        "first_min_num_post_burn_in": gibbs_controls.first_min_num_post_burn_in,
        "first_max_num_post_burn_in": gibbs_controls.first_max_num_post_burn_in,
        "burn_in_stall_window": gibbs_controls.burn_in_stall_window,
        "burn_in_stall_delta": gibbs_controls.burn_in_stall_delta,
        "active_beta_top_k": gibbs_controls.active_beta_top_k,
        "active_beta_min_abs": gibbs_controls.active_beta_min_abs,
        "stop_patience": gibbs_controls.stop_patience,
        "stop_top_gene_k": gibbs_controls.stop_top_gene_k,
        "stop_min_gene_d": gibbs_controls.stop_min_gene_d,
        "beta_rel_mcse_denom_floor": gibbs_controls.beta_rel_mcse_denom_floor,
        "stall_window": gibbs_controls.stall_window,
        "stall_min_burn_in": gibbs_controls.stall_min_burn_in,
        "stall_min_post_burn_in": gibbs_controls.stall_min_post_burn_in,
        "stall_delta_rhat": gibbs_controls.stall_delta_rhat,
        "stall_delta_mcse": gibbs_controls.stall_delta_mcse,
        "stall_recent_window": gibbs_controls.stall_recent_window,
        "stall_recent_eps": gibbs_controls.stall_recent_eps,
        "diag_every": gibbs_controls.diag_every,
        "num_chains_betas": num_chains_betas,
        "max_num_iter": max_num_iter,
        "use_mean_betas": use_mean_betas,
        "warm_start": warm_start,
        "stopping_preset_name": stopping_preset_name,
        "r_threshold_burn_in": r_threshold_burn_in,
        "stop_mcse_quantile": stop_mcse_quantile,
        "max_abs_mcse_d": max_abs_mcse_d,
        "max_rel_mcse_beta": max_rel_mcse_beta,
        "sparse_solution": sparse_solution,
        "sparse_frac_gibbs": sparse_frac_gibbs,
        "sparse_max_gibbs": sparse_max_gibbs,
        "sparse_frac_betas": sparse_frac_betas,
        "pre_filter_batch_size": pre_filter_batch_size,
        "max_allowed_batch_correlation": max_allowed_batch_correlation,
        "initial_linear_filter": initial_linear_filter,
        "correct_betas_mean": correct_betas_mean,
        "correct_betas_var": correct_betas_var,
        "adjust_priors": adjust_priors,
        "experimental_hyper_mutation": experimental_hyper_mutation,
        "increase_hyper_if_betas_below": increase_hyper_if_betas_below,
    }


def _record_gibbs_configuration_params(state, run_state, config):
    state._record_params(
        {
            "num_chains": config["num_chains"],
            "num_chains_betas": config["num_chains_betas"],
            "max_num_restarts": config["max_num_restarts"],
            "target_num_epochs": config["target_num_epochs"],
            "max_num_attempt_restarts": run_state.max_num_attempt_restarts,
            "max_num_iter": config["max_num_iter"],
            "total_num_iter": config["total_num_iter"],
            "epoch_max_num_iter": config["epoch_max_num_iter_config"],
            "use_mean_betas": config["use_mean_betas"],
            "warm_start": config["warm_start"],
            "stopping_preset_name": config["stopping_preset_name"],
            "r_threshold_burn_in": config["r_threshold_burn_in"],
            "burn_in_rhat_quantile": config["burn_in_rhat_quantile"],
            "burn_in_rhat_quantile_effective": config["burn_in_rhat_quantile"],
            "burn_in_patience": config["burn_in_patience"],
            "min_num_burn_in": config["first_min_num_burn_in"],
            "max_num_burn_in": config["first_max_num_burn_in"],
            "min_num_post_burn_in": config["first_min_num_post_burn_in"],
            "max_num_post_burn_in": config["first_max_num_post_burn_in"],
            "burn_in_stall_window": config["burn_in_stall_window"],
            "burn_in_stall_delta": config["burn_in_stall_delta"],
            "active_beta_top_k": config["active_beta_top_k"],
            "active_beta_min_abs": config["active_beta_min_abs"],
            "stop_mcse_quantile": config["stop_mcse_quantile"],
            "stop_patience": config["stop_patience"],
            "stop_top_gene_k": config["stop_top_gene_k"],
            "stop_min_gene_d": config["stop_min_gene_d"],
            "max_abs_mcse_d": config["max_abs_mcse_d"],
            "max_rel_mcse_beta": config["max_rel_mcse_beta"],
            "beta_rel_mcse_denom_floor": config["beta_rel_mcse_denom_floor"],
            "stall_window": config["stall_window"],
            "stall_min_burn_in": config["stall_min_burn_in"],
            "stall_min_post_burn_in": config["stall_min_post_burn_in"],
            "stall_delta_rhat": config["stall_delta_rhat"],
            "stall_delta_mcse": config["stall_delta_mcse"],
            "stall_recent_window": config["stall_recent_window"],
            "stall_recent_eps": config["stall_recent_eps"],
            "diag_every": config["diag_every"],
            "sparse_solution": config["sparse_solution"],
            "sparse_frac": config["sparse_frac_gibbs"],
            "sparse_max": config["sparse_max_gibbs"],
            "sparse_frac_betas": config["sparse_frac_betas"],
            "pre_filter_batch_size": config["pre_filter_batch_size"],
            "max_allowed_batch_correlation": config["max_allowed_batch_correlation"],
            "initial_linear_filter": config["initial_linear_filter"],
            "correct_betas_mean": config["correct_betas_mean"],
            "correct_betas_var": config["correct_betas_var"],
            "adjust_priors": config["adjust_priors"],
            "experimental_hyper_mutation": config["experimental_hyper_mutation"],
            "increase_hyper_if_betas_below": config["increase_hyper_if_betas_below"],
        }
    )
    state._record_param("min_num_post_burn_in_effective", config["first_min_num_post_burn_in"])
    state._record_param("stall_min_post_burn_samples", config["stall_min_post_burn_in"])


def _log_gibbs_configuration_summary(config, run_state):
    log("Running Gibbs")
    log(
        "Gibbs stopping preset=%s; burn-in: r_threshold=%.4g, rhat_q=%.3g, patience=%d; active betas: topK=%d, min_abs=%.4g"
        % (
            config["stopping_preset_name"],
            config["r_threshold_burn_in"],
            config["burn_in_rhat_quantile"],
            config["burn_in_patience"],
            config["active_beta_top_k"],
            config["active_beta_min_abs"],
        ),
        INFO,
    )
    log(
        "Gibbs restart schedule: target_epochs=%d (max_num_restarts=%d), max_attempts=%d, per-epoch max_num_iter=%d, total_num_iter=%d"
        % (
            config["target_num_epochs"],
            config["max_num_restarts"],
            run_state.max_num_attempt_restarts,
            config["epoch_max_num_iter_config"],
            config["total_num_iter"],
        ),
        INFO,
    )
    log(
        "Gibbs epoch bounds (epoch 1): burn=[%d,%d], post=[%d,%d], stall_window=%d, stall_delta=%.4g"
        % (
            config["first_min_num_burn_in"],
            config["first_max_num_burn_in"],
            config["first_min_num_post_burn_in"],
            config["first_max_num_post_burn_in"],
            config["burn_in_stall_window"],
            config["burn_in_stall_delta"],
        ),
        INFO,
    )
    log(
        "Gibbs stopping thresholds: stop_q=%.3g, stop_patience=%d, max_rel_mcse_beta=%.4g, beta_rel_mcse_denom_floor=%.4g, stop_top_gene_k=%d, stop_min_gene_d=%s, max_abs_mcse_d=%.4g, diag_every=%d"
        % (
            config["stop_mcse_quantile"],
            config["stop_patience"],
            config["max_rel_mcse_beta"],
            config["beta_rel_mcse_denom_floor"],
            config["stop_top_gene_k"],
            ("%.4g" % config["stop_min_gene_d"]) if config["stop_min_gene_d"] is not None else "None",
            config["max_abs_mcse_d"],
            config["diag_every"],
        ),
        INFO,
    )
    log(
        "Gibbs experimental hyper mutation: enabled=%s, threshold=%s"
        % (
            str(config["experimental_hyper_mutation"]),
            ("%.4g" % config["increase_hyper_if_betas_below"])
            if config["increase_hyper_if_betas_below"] is not None
            else "None",
        ),
        INFO,
    )
    log(
        "Gibbs stall controls: window=%d, min_burn=%d, min_post_for_stall=%d, delta_rhat=%.4g, delta_mcse=%.4g, recent_window=%d, recent_eps=%.4g"
        % (
            config["stall_window"],
            config["stall_min_burn_in"],
            config["stall_min_post_burn_in"],
            config["stall_delta_rhat"],
            config["stall_delta_mcse"],
            config["stall_recent_window"],
            config["stall_recent_eps"],
        ),
        INFO,
    )


# ========================= Outer Gibbs Epoch Execution =========================
def _resolve_epoch_iteration_budget(
    remaining_iter,
    epoch_max_num_iter_config,
    min_num_burn_in,
    max_num_burn_in,
    min_num_post_burn_in,
    max_num_post_burn_in,
):
    local_epoch_max_num_iter = min(epoch_max_num_iter_config, remaining_iter)
    if local_epoch_max_num_iter < 1:
        local_epoch_max_num_iter = 1

    local_max_num_burn_in = min(max_num_burn_in, max(1, local_epoch_max_num_iter - 1))
    local_min_num_burn_in = min(min_num_burn_in, local_max_num_burn_in)

    local_max_num_post_burn_in = min(max_num_post_burn_in, max(1, local_epoch_max_num_iter - local_min_num_burn_in))
    local_min_num_post_burn_in = min(min_num_post_burn_in, local_max_num_post_burn_in)

    # Ensure burn-in max still leaves room for at least the local post-burn minimum.
    local_max_num_burn_in = min(local_max_num_burn_in, max(1, local_epoch_max_num_iter - local_min_num_post_burn_in))
    local_min_num_burn_in = min(local_min_num_burn_in, local_max_num_burn_in)
    local_max_num_post_burn_in = min(local_max_num_post_burn_in, max(1, local_epoch_max_num_iter - local_min_num_burn_in))
    local_min_num_post_burn_in = min(local_min_num_post_burn_in, local_max_num_post_burn_in)

    return (
        local_epoch_max_num_iter,
        local_min_num_burn_in,
        local_max_num_burn_in,
        local_min_num_post_burn_in,
        local_max_num_post_burn_in,
    )


@dataclass
class GibbsRunState:
    target_num_epochs: int
    max_num_attempt_restarts: int
    num_p_increases: int = 0
    num_attempts: int = 0
    num_completed_epochs: int = 0
    remaining_total_iter: int = 0


@dataclass
class GibbsRunControls:
    max_num_restarts: int
    target_num_epochs: int
    min_num_burn_in: int
    max_num_burn_in: int
    min_num_post_burn_in: int
    max_num_post_burn_in: int
    passed_in_max_num_burn_in: int
    epoch_max_num_iter_config: int
    total_num_iter: int
    run_state: GibbsRunState
    num_chains: int
    diag_every: int
    burn_in_patience: int
    burn_in_stall_window: int
    burn_in_stall_delta: float
    stop_patience: int
    stop_top_gene_k: int
    stop_min_gene_d: float | None
    active_beta_top_k: int
    active_beta_min_abs: float
    beta_rel_mcse_denom_floor: float
    stall_window: int
    stall_min_burn_in: int
    stall_min_post_burn_in: int
    stall_delta_rhat: float
    stall_delta_mcse: float
    stall_recent_window: int
    stall_recent_eps: float
    burn_in_rhat_quantile: float
    first_min_num_burn_in: int
    first_max_num_burn_in: int
    first_min_num_post_burn_in: int
    first_max_num_post_burn_in: int


@dataclass
class GibbsEpochPhaseConfig:
    total_num_iter: int
    num_chains: int
    num_full_gene_sets: int
    use_mean_betas: bool
    max_mb_X_h: int
    target_num_epochs: int
    num_mad: int
    adjust_priors: bool
    epoch_max_num_iter_config: int
    min_num_burn_in: int
    max_num_burn_in: int
    min_num_post_burn_in: int
    max_num_post_burn_in: int
    increase_hyper_if_betas_below: float | None
    experimental_hyper_mutation: bool


@dataclass
class GibbsIterationUpdateConfig:
    use_mean_betas: bool
    warm_start: bool
    debug_zero_sparse: bool
    num_chains: int
    num_batches_parallel: int
    betas_trace_out: str | None
    update_huge_scores: bool
    compute_Y_raw: bool
    adjust_priors: bool


@dataclass
class GibbsEpochIterationStaticConfig:
    inner_beta_kwargs: dict
    iteration_update_config: GibbsIterationUpdateConfig
    cur_background_log_bf_v: object
    y_var_orig: float
    gauss_seidel: bool
    initial_linear_filter: bool
    sparse_frac_gibbs: float
    sparse_max_gibbs: float
    correct_betas_mean: bool
    correct_betas_var: bool
    prefilter_config: dict
    iteration_progress_config: dict


@dataclass
class GibbsEpochRuntimeConfigs:
    epoch_phase_config: GibbsEpochPhaseConfig
    epoch_iteration_static_config: GibbsEpochIterationStaticConfig


@dataclass
class GibbsIterationCorrectionConfig:
    inner_beta_kwargs: dict
    iteration_update_config: GibbsIterationUpdateConfig
    num_mad: int
    num_attempts: int
    max_num_attempt_restarts: int
    increase_hyper_if_betas_below_for_epoch: float | None
    experimental_hyper_mutation: bool
    num_before_checking_p_increase: int
    p_scale_factor: float


@dataclass
class GibbsEpochIterationLoopConfig:
    epoch_max_num_iter: int
    epoch_total_iter_offset: int
    trace_chain_offset: int
    full_betas_m_shape: tuple
    num_stack_batches: int
    stack_batch_size: int
    X_hstacked: object
    min_num_burn_in_for_epoch: int
    max_num_burn_in_for_epoch: int
    min_num_iter_for_epoch: int
    min_num_post_burn_in_for_epoch: int
    max_num_post_burn_in_for_epoch: int
    post_burn_reset_arrays: list
    post_burn_reset_missing_arrays: list
    inner_beta_kwargs: dict
    iteration_update_config: GibbsIterationUpdateConfig
    cur_background_log_bf_v: object
    y_var_orig: float
    gauss_seidel: bool
    initial_linear_filter: bool
    sparse_frac_gibbs: float
    sparse_max_gibbs: float
    correct_betas_mean: bool
    correct_betas_var: bool
    prefilter_config: dict
    iteration_progress_config: dict
    num_attempts: int
    max_num_attempt_restarts: int
    num_mad: int
    increase_hyper_if_betas_below_for_epoch: float | None
    experimental_hyper_mutation: bool
    num_before_checking_p_increase: int
    p_scale_factor: float


@dataclass
class GibbsIterationProgressRuntimeConfig:
    trace_chain_offset: int
    epoch_total_iter_offset: int
    epoch_max_num_iter: int
    max_num_burn_in_for_epoch: int
    min_num_iter_for_epoch: int
    min_num_burn_in_for_epoch: int
    max_num_post_burn_in_for_epoch: int
    min_num_post_burn_in_for_epoch: int
    post_burn_reset_arrays: list
    post_burn_reset_missing_arrays: list
    iteration_progress_config: dict


@dataclass
class GibbsIterationRuntimeConfigs:
    correction_config: GibbsIterationCorrectionConfig
    progress_runtime_config: GibbsIterationProgressRuntimeConfig
    iteration_state_config: dict


def _initialize_gibbs_run_state(total_num_iter, target_num_epochs, max_num_restarts):
    # Mutable run-level counters shared across epoch attempts.
    return GibbsRunState(
        target_num_epochs=target_num_epochs,
        max_num_attempt_restarts=target_num_epochs + max_num_restarts,
        num_p_increases=0,
        num_attempts=0,
        num_completed_epochs=0,
        remaining_total_iter=int(total_num_iter),
    )


def _prepare_gibbs_epoch_attempt(
    state,
    run_state,
    epoch_phase_config,
):
    # Resolve one epoch attempt's bounds and bookkeeping from run-level state.
    total_num_iter = epoch_phase_config.total_num_iter
    num_chains = epoch_phase_config.num_chains
    target_num_epochs = epoch_phase_config.target_num_epochs
    epoch_max_num_iter_config = epoch_phase_config.epoch_max_num_iter_config
    min_num_burn_in = epoch_phase_config.min_num_burn_in
    max_num_burn_in = epoch_phase_config.max_num_burn_in
    min_num_post_burn_in = epoch_phase_config.min_num_post_burn_in
    max_num_post_burn_in = epoch_phase_config.max_num_post_burn_in
    increase_hyper_if_betas_below = epoch_phase_config.increase_hyper_if_betas_below

    run_state.num_attempts += 1

    (
        epoch_max_num_iter,
        min_num_burn_in_for_epoch,
        max_num_burn_in_for_epoch,
        min_num_post_burn_in_for_epoch,
        max_num_post_burn_in_for_epoch,
    ) = _resolve_epoch_iteration_budget(
        run_state.remaining_total_iter,
        epoch_max_num_iter_config,
        min_num_burn_in,
        max_num_burn_in,
        min_num_post_burn_in,
        max_num_post_burn_in,
    )
    if epoch_max_num_iter < 1:
        return None

    trace_chain_offset = run_state.num_completed_epochs * num_chains
    p_scale_factor = 1 - np.log(state.p) / (2 * np.log(10))

    min_num_iter_for_epoch = min_num_burn_in_for_epoch
    increase_hyper_if_betas_below_for_epoch = increase_hyper_if_betas_below if run_state.num_completed_epochs == 0 else None
    num_before_checking_p_increase = max(min_num_iter_for_epoch, min_num_burn_in_for_epoch)
    if increase_hyper_if_betas_below_for_epoch is not None and num_before_checking_p_increase > min_num_iter_for_epoch:
        # Ensure this check runs before any early break.
        min_num_iter_for_epoch = num_before_checking_p_increase

    state._record_param("num_gibbs_restarts", run_state.num_attempts - 1, overwrite=True)
    state._record_param("num_gibbs_epochs_completed", run_state.num_completed_epochs, overwrite=True)
    if run_state.num_attempts > 1:
        log("Gibbs restart attempt %d" % (run_state.num_attempts - 1))
    log(
        "Gibbs epoch %d/%d: max_num_iter=%d, burn=[%d,%d], post=[%d,%d], remaining_total_iter=%d"
        % (
            run_state.num_completed_epochs + 1,
            target_num_epochs,
            epoch_max_num_iter,
            min_num_burn_in_for_epoch,
            max_num_burn_in_for_epoch,
            min_num_post_burn_in_for_epoch,
            max_num_post_burn_in_for_epoch,
            run_state.remaining_total_iter,
        ),
        INFO,
    )

    epoch_total_iter_offset = total_num_iter - run_state.remaining_total_iter
    return {
        "epoch_max_num_iter": epoch_max_num_iter,
        "min_num_burn_in_for_epoch": min_num_burn_in_for_epoch,
        "max_num_burn_in_for_epoch": max_num_burn_in_for_epoch,
        "min_num_post_burn_in_for_epoch": min_num_post_burn_in_for_epoch,
        "max_num_post_burn_in_for_epoch": max_num_post_burn_in_for_epoch,
        "trace_chain_offset": trace_chain_offset,
        "p_scale_factor": p_scale_factor,
        "min_num_iter_for_epoch": min_num_iter_for_epoch,
        "increase_hyper_if_betas_below_for_epoch": increase_hyper_if_betas_below_for_epoch,
        "experimental_hyper_mutation": epoch_phase_config.experimental_hyper_mutation,
        "num_before_checking_p_increase": num_before_checking_p_increase,
        "epoch_total_iter_offset": epoch_total_iter_offset,
    }


def _trim_stall_histories(
    post_stall_best_beta_rhat_history,
    post_stall_best_D_mcse_history,
    post_stall_snapshots,
    max_stall_history_len,
):
    if len(post_stall_best_beta_rhat_history) > max_stall_history_len:
        del post_stall_best_beta_rhat_history[:-max_stall_history_len]
        del post_stall_best_D_mcse_history[:-max_stall_history_len]
    if len(post_stall_snapshots) > max_stall_history_len:
        del post_stall_snapshots[:-max_stall_history_len]


def _evaluate_post_stall_status(
    num_post_burn_beta,
    num_post_burn_D,
    post_stall_beta_sum_m,
    post_stall_beta_sum2_m,
    post_stall_D_sum_m,
    post_stall_beta_indices,
    post_stall_gene_indices,
    beta_rhat_q_post,
    D_mcse_q,
    stop_mcse_quantile,
    stall_window,
    stall_min_post_burn_in,
    min_num_post_burn_in_for_epoch,
    stall_delta_rhat,
    stall_delta_mcse,
    stall_recent_window,
    stall_recent_eps,
    post_stall_best_beta_rhat_history,
    post_stall_best_D_mcse_history,
    post_stall_snapshots,
    num_chains,
):
    post_stall_plateau = False
    post_stall_recent_worse = False
    post_stall_recent_beta_rhat_q = np.nan
    post_stall_recent_D_mcse_q = np.nan
    min_post_burn_for_stall = max(stall_min_post_burn_in, min_num_post_burn_in_for_epoch)

    if stall_window > 0 and num_post_burn_D >= min_post_burn_for_stall:
        if len(post_stall_best_beta_rhat_history) >= stall_window:
            post_rhat_improvement = post_stall_best_beta_rhat_history[-stall_window] - post_stall_best_beta_rhat_history[-1]
            post_mcse_improvement = post_stall_best_D_mcse_history[-stall_window] - post_stall_best_D_mcse_history[-1]
            if post_rhat_improvement < stall_delta_rhat and post_mcse_improvement < stall_delta_mcse:
                post_stall_plateau = True

        (
            post_stall_recent_beta_rhat_q,
            post_stall_recent_D_mcse_q,
        ) = _compute_recent_post_stall_metrics(
            num_post_burn_beta=num_post_burn_beta,
            num_post_burn_D=num_post_burn_D,
            post_stall_beta_sum_m=post_stall_beta_sum_m,
            post_stall_beta_sum2_m=post_stall_beta_sum2_m,
            post_stall_D_sum_m=post_stall_D_sum_m,
            post_stall_beta_indices=post_stall_beta_indices,
            post_stall_gene_indices=post_stall_gene_indices,
            post_stall_snapshots=post_stall_snapshots,
            stall_recent_window=stall_recent_window,
            stop_mcse_quantile=stop_mcse_quantile,
            num_chains=num_chains,
        )

        if np.isfinite(post_stall_recent_beta_rhat_q) and post_stall_recent_beta_rhat_q > beta_rhat_q_post * (1 + stall_recent_eps):
            post_stall_recent_worse = True
        if np.isfinite(post_stall_recent_D_mcse_q) and post_stall_recent_D_mcse_q > D_mcse_q * (1 + stall_recent_eps):
            post_stall_recent_worse = True

    return (
        post_stall_plateau,
        post_stall_recent_worse,
        post_stall_recent_beta_rhat_q,
        post_stall_recent_D_mcse_q,
    )


def _compute_recent_post_stall_metrics(
    num_post_burn_beta,
    num_post_burn_D,
    post_stall_beta_sum_m,
    post_stall_beta_sum2_m,
    post_stall_D_sum_m,
    post_stall_beta_indices,
    post_stall_gene_indices,
    post_stall_snapshots,
    stall_recent_window,
    stop_mcse_quantile,
    num_chains,
):
    recent_beta_rhat_q = np.nan
    recent_D_mcse_q = np.nan
    if not (stall_recent_window > 0 and len(post_stall_snapshots) >= stall_recent_window + 1):
        return (recent_beta_rhat_q, recent_D_mcse_q)

    old_beta_n, old_beta_sum_m, old_beta_sum2_m, old_D_n, old_D_sum_m = post_stall_snapshots[-(stall_recent_window + 1)]
    recent_beta_n = num_post_burn_beta - old_beta_n
    recent_D_n = num_post_burn_D - old_D_n

    if recent_beta_n > 1 and post_stall_beta_indices.size > 0:
        recent_beta_sum_m = post_stall_beta_sum_m - old_beta_sum_m
        recent_beta_sum2_m = post_stall_beta_sum2_m - old_beta_sum2_m
        _, _, recent_R_beta_v, _ = _calculate_rhat_from_sums(recent_beta_sum_m, recent_beta_sum2_m, recent_beta_n)
        recent_rhat_candidates = recent_R_beta_v[np.logical_and(np.isfinite(recent_R_beta_v), recent_R_beta_v >= 1)]
        recent_beta_rhat_q = _safe_quantile(recent_rhat_candidates, stop_mcse_quantile, 1.0)

    if recent_D_n > 1 and post_stall_gene_indices.size > 0:
        recent_D_sum_m = post_stall_D_sum_m - old_D_sum_m
        recent_D_chain_means_m = recent_D_sum_m / float(recent_D_n)
        recent_D_mcse_v = np.sqrt(np.var(recent_D_chain_means_m, axis=0, ddof=1) / float(num_chains))
        recent_D_mcse_q = _safe_quantile(recent_D_mcse_v, stop_mcse_quantile, np.inf)

    return (recent_beta_rhat_q, recent_D_mcse_q)


def _zero_arrays(*arrays):
    for arr in arrays:
        if arr is not None:
            arr[:] = 0


def _reset_gibbs_post_burn_accumulators(post_burn_reset_arrays, post_burn_reset_missing_arrays):
    _zero_arrays(*(post_burn_reset_arrays + post_burn_reset_missing_arrays))


def _end_gibbs_burn_in(
    post_burn_reset_arrays,
    post_burn_reset_missing_arrays,
    burn_in_pass_streak,
    stop_pass_streak,
    reset_burn_in_pass_streak=False,
):
    in_burn_in = False
    if reset_burn_in_pass_streak:
        burn_in_pass_streak = 0
    stop_pass_streak = 0
    _reset_gibbs_post_burn_accumulators(post_burn_reset_arrays, post_burn_reset_missing_arrays)
    return (in_burn_in, burn_in_pass_streak, stop_pass_streak)


def _evaluate_burn_in_diagnostics(
    epoch_control,
    burn_in_config,
    min_num_burn_in_for_epoch,
    epoch_runtime,
    num_samples,
):
    all_sum_betas_m = epoch_runtime["all_sum_betas_m"]
    all_sum_betas2_m = epoch_runtime["all_sum_betas2_m"]
    all_num_sum_m = epoch_runtime["all_num_sum_m"]
    burn_in_pass_streak = epoch_control["burn_in_pass_streak"]
    burn_in_rhat_history = epoch_control["burn_in_rhat_history"]
    burn_stall_best_beta_rhat_history = epoch_control["burn_stall_best_beta_rhat_history"]
    burn_stall_snapshots = epoch_control["burn_stall_snapshots"]
    burn_stall_beta_indices = epoch_control["burn_stall_beta_indices"]

    active_beta_top_k = burn_in_config["active_beta_top_k"]
    active_beta_min_abs = burn_in_config["active_beta_min_abs"]
    burn_in_rhat_quantile = burn_in_config["burn_in_rhat_quantile"]
    r_threshold_burn_in = burn_in_config["r_threshold_burn_in"]
    stall_window = burn_in_config["stall_window"]
    stall_min_burn_in = burn_in_config["stall_min_burn_in"]
    stall_delta_rhat = burn_in_config["stall_delta_rhat"]
    stall_recent_window = burn_in_config["stall_recent_window"]
    stall_recent_eps = burn_in_config["stall_recent_eps"]
    burn_in_stall_window = burn_in_config["burn_in_stall_window"]
    burn_in_stall_delta = burn_in_config["burn_in_stall_delta"]
    (
        R_beta_v,
        active_beta_mask_v,
        num_active_betas,
        beta_rhat_q,
        beta_rhat_max,
    ) = _compute_burn_in_active_beta_rhat_stats(
        all_sum_betas_m=all_sum_betas_m,
        all_sum_betas2_m=all_sum_betas2_m,
        all_num_sum_m=all_num_sum_m,
        num_samples=num_samples,
        active_beta_top_k=active_beta_top_k,
        active_beta_min_abs=active_beta_min_abs,
        burn_in_rhat_quantile=burn_in_rhat_quantile,
    )

    if beta_rhat_q <= r_threshold_burn_in:
        burn_in_pass_streak += 1
    else:
        burn_in_pass_streak = 0

    burn_in_rhat_history.append(beta_rhat_q)
    burn_best_beta_rhat_q = beta_rhat_q if len(burn_stall_best_beta_rhat_history) == 0 else min(burn_stall_best_beta_rhat_history[-1], beta_rhat_q)
    burn_stall_best_beta_rhat_history.append(burn_best_beta_rhat_q)

    if burn_stall_beta_indices is None:
        burn_stall_beta_indices = _prepare_stall_indices(active_beta_mask_v, np.mean(_means_from_sums(all_sum_betas_m, all_num_sum_m), axis=0), active_beta_top_k)

    if burn_stall_beta_indices.size > 0:
        burn_stall_sum_m = copy.copy(all_sum_betas_m[:,burn_stall_beta_indices])
        burn_stall_sum2_m = copy.copy(all_sum_betas2_m[:,burn_stall_beta_indices])
    else:
        burn_stall_sum_m = np.zeros((all_sum_betas_m.shape[0], 0))
        burn_stall_sum2_m = np.zeros((all_sum_betas2_m.shape[0], 0))
    burn_stall_snapshots.append((num_samples, burn_stall_sum_m, burn_stall_sum2_m, beta_rhat_q))

    max_stall_history_len = max(stall_window + 2, stall_recent_window + 2, 10)
    if len(burn_stall_best_beta_rhat_history) > max_stall_history_len:
        del burn_stall_best_beta_rhat_history[:-max_stall_history_len]
    if len(burn_stall_snapshots) > max_stall_history_len:
        del burn_stall_snapshots[:-max_stall_history_len]

    burn_stall_plateau = False
    burn_stall_recent_worse = False
    min_samples_for_burn_stall = max(min_num_burn_in_for_epoch, stall_min_burn_in)
    if stall_window > 0 and num_samples >= min_samples_for_burn_stall:
        if len(burn_stall_best_beta_rhat_history) >= stall_window:
            burn_rhat_improvement = burn_stall_best_beta_rhat_history[-stall_window] - burn_stall_best_beta_rhat_history[-1]
            if burn_rhat_improvement < stall_delta_rhat:
                burn_stall_plateau = True

        burn_stall_recent_beta_rhat_q = _compute_recent_burn_in_stall_beta_rhat(
            num_samples=num_samples,
            burn_stall_sum_m=burn_stall_sum_m,
            burn_stall_sum2_m=burn_stall_sum2_m,
            burn_stall_beta_indices=burn_stall_beta_indices,
            burn_stall_snapshots=burn_stall_snapshots,
            stall_recent_window=stall_recent_window,
            burn_in_rhat_quantile=burn_in_rhat_quantile,
        )
        if np.isfinite(burn_stall_recent_beta_rhat_q) and burn_stall_recent_beta_rhat_q > beta_rhat_q * (1 + stall_recent_eps):
            burn_stall_recent_worse = True

    burn_stall_detected = burn_stall_plateau or burn_stall_recent_worse
    burn_window_plateau_detected, burn_window_span = _compute_burn_in_window_plateau_status(
        burn_in_rhat_history=burn_in_rhat_history,
        burn_in_stall_window=burn_in_stall_window,
        burn_in_stall_delta=burn_in_stall_delta,
    )

    return {
        "R_beta_v": R_beta_v,
        "burn_in_pass_streak": burn_in_pass_streak,
        "burn_stall_beta_indices": burn_stall_beta_indices,
        "beta_rhat_q": beta_rhat_q,
        "beta_rhat_max": beta_rhat_max,
        "num_active_betas": num_active_betas,
        "burn_stall_plateau": burn_stall_plateau,
        "burn_stall_recent_worse": burn_stall_recent_worse,
        "burn_stall_detected": burn_stall_detected,
        "burn_window_plateau_detected": burn_window_plateau_detected,
        "burn_window_span": burn_window_span,
    }


def _compute_recent_burn_in_stall_beta_rhat(
    num_samples,
    burn_stall_sum_m,
    burn_stall_sum2_m,
    burn_stall_beta_indices,
    burn_stall_snapshots,
    stall_recent_window,
    burn_in_rhat_quantile,
):
    if not (stall_recent_window > 0 and len(burn_stall_snapshots) >= stall_recent_window + 1 and burn_stall_beta_indices.size > 0):
        return np.nan
    old_num_samples, old_sum_m, old_sum2_m, _ = burn_stall_snapshots[-(stall_recent_window + 1)]
    recent_num_samples = num_samples - old_num_samples
    if recent_num_samples <= 1:
        return np.nan
    recent_sum_m = burn_stall_sum_m - old_sum_m
    recent_sum2_m = burn_stall_sum2_m - old_sum2_m
    _, _, burn_recent_R_beta_v, _ = _calculate_rhat_from_sums(recent_sum_m, recent_sum2_m, recent_num_samples)
    burn_recent_candidates = burn_recent_R_beta_v[np.logical_and(np.isfinite(burn_recent_R_beta_v), burn_recent_R_beta_v >= 1)]
    return _safe_quantile(burn_recent_candidates, burn_in_rhat_quantile, 1.0)


def _compute_burn_in_window_plateau_status(
    burn_in_rhat_history,
    burn_in_stall_window,
    burn_in_stall_delta,
):
    if not (burn_in_stall_window > 0 and len(burn_in_rhat_history) >= burn_in_stall_window):
        return (False, np.nan)
    burn_window_v = np.array(burn_in_rhat_history[-burn_in_stall_window:], dtype=float)
    if not np.all(np.isfinite(burn_window_v)):
        return (False, np.nan)
    burn_window_span = np.max(burn_window_v) - np.min(burn_window_v)
    return (burn_window_span < burn_in_stall_delta, burn_window_span)


def _compute_burn_in_active_beta_rhat_stats(
    all_sum_betas_m,
    all_sum_betas2_m,
    all_num_sum_m,
    num_samples,
    active_beta_top_k,
    active_beta_min_abs,
    burn_in_rhat_quantile,
):
    (_, _, R_beta_v, _) = _calculate_rhat_from_sums(all_sum_betas_m, all_sum_betas2_m, num_samples)
    active_beta_mask_v, _, _ = _get_active_beta_mask(all_sum_betas_m, all_num_sum_m, active_beta_top_k, active_beta_min_abs)
    num_active_betas = int(np.sum(active_beta_mask_v))
    if num_active_betas > 0:
        R_beta_active_v = R_beta_v[active_beta_mask_v]
        rhat_candidates = R_beta_active_v[np.logical_and(np.isfinite(R_beta_active_v), R_beta_active_v >= 1)]
        beta_rhat_q = _safe_quantile(rhat_candidates, burn_in_rhat_quantile, 1.0)
        beta_rhat_max = _safe_quantile(R_beta_active_v, 1.0, 1.0)
    else:
        beta_rhat_q = 1.0
        beta_rhat_max = 1.0
    return (R_beta_v, active_beta_mask_v, num_active_betas, beta_rhat_q, beta_rhat_max)


def _handle_gibbs_burn_in_max_iter(
    num_samples,
    post_burn_reset_arrays,
    post_burn_reset_missing_arrays,
    burn_in_pass_streak,
    stop_pass_streak,
):
    in_burn_in, burn_in_pass_streak, stop_pass_streak = _end_gibbs_burn_in(
        post_burn_reset_arrays=post_burn_reset_arrays,
        post_burn_reset_missing_arrays=post_burn_reset_missing_arrays,
        burn_in_pass_streak=burn_in_pass_streak,
        stop_pass_streak=stop_pass_streak,
        reset_burn_in_pass_streak=True,
    )
    log("Stopping Gibbs burn in after %d iterations (per-epoch max burn-in reached)" % (num_samples), INFO)
    return (in_burn_in, burn_in_pass_streak, stop_pass_streak)


def _handle_gibbs_burn_in_gauss_seidel_path(
    num_samples,
    epoch_total_iter_offset,
    min_num_iter_for_epoch,
    eps,
    Y_sample_m,
    prev_Ys_m,
    post_burn_reset_arrays,
    post_burn_reset_missing_arrays,
    burn_in_pass_streak,
    stop_pass_streak,
):
    in_burn_in = True
    if prev_Ys_m is not None:
        sum_diff = np.sum(np.abs(prev_Ys_m - Y_sample_m))
        sum_prev = np.sum(np.abs(prev_Ys_m))
        max_diff_frac = np.max(np.abs((prev_Ys_m - Y_sample_m) / prev_Ys_m))

        tot_diff = sum_diff / sum_prev
        log(
            "Gibbs iteration %d (global %d): mean gauss seidel difference = %.4g / %.4g = %.4g; max frac difference = %.4g"
            % (num_samples, epoch_total_iter_offset + num_samples, sum_diff, sum_prev, tot_diff, max_diff_frac)
        )
        if num_samples > min_num_iter_for_epoch and tot_diff < eps:
            in_burn_in, burn_in_pass_streak, stop_pass_streak = _end_gibbs_burn_in(
                post_burn_reset_arrays=post_burn_reset_arrays,
                post_burn_reset_missing_arrays=post_burn_reset_missing_arrays,
                burn_in_pass_streak=burn_in_pass_streak,
                stop_pass_streak=stop_pass_streak,
                reset_burn_in_pass_streak=True,
            )
            log("Gibbs gauss converged after %d iterations" % num_samples, INFO)
    prev_Ys_m = Y_sample_m
    return (in_burn_in, burn_in_pass_streak, stop_pass_streak, prev_Ys_m)


def _handle_gibbs_burn_in_diag_path(
    num_samples,
    epoch_control,
    burn_in_config,
    min_num_burn_in_for_epoch,
    epoch_runtime,
    post_burn_reset_arrays,
    post_burn_reset_missing_arrays,
    burn_in_pass_streak,
    stop_pass_streak,
):
    in_burn_in = True
    burn_diag = _evaluate_burn_in_diagnostics(
        epoch_control=epoch_control,
        burn_in_config=burn_in_config,
        min_num_burn_in_for_epoch=min_num_burn_in_for_epoch,
        epoch_runtime=epoch_runtime,
        num_samples=num_samples,
    )
    R_beta_v = burn_diag["R_beta_v"]
    burn_in_pass_streak = burn_diag["burn_in_pass_streak"]
    burn_stall_beta_indices = burn_diag["burn_stall_beta_indices"]
    beta_rhat_q = burn_diag["beta_rhat_q"]
    beta_rhat_max = burn_diag["beta_rhat_max"]
    num_active_betas = burn_diag["num_active_betas"]
    burn_stall_plateau = burn_diag["burn_stall_plateau"]
    burn_stall_recent_worse = burn_diag["burn_stall_recent_worse"]
    burn_stall_detected = burn_diag["burn_stall_detected"]
    burn_window_plateau_detected = burn_diag["burn_window_plateau_detected"]
    burn_window_span = burn_diag["burn_window_span"]

    burn_in_rhat_quantile = burn_in_config["burn_in_rhat_quantile"]
    num_full_gene_sets = burn_in_config["num_full_gene_sets"]
    burn_in_patience = burn_in_config["burn_in_patience"]
    stop_patience = burn_in_config["stop_patience"]
    r_threshold_burn_in = burn_in_config["r_threshold_burn_in"]
    burn_in_stall_window = burn_in_config["burn_in_stall_window"]
    burn_in_stall_delta = burn_in_config["burn_in_stall_delta"]

    log(
        "Gibbs burn-in iter %d: beta_Rhat_q(%.2f)=%.4g; beta_Rhat_max=%.4g; active_betas=%d/%d; burn_streak=%d/%d; stop_streak=%d/%d"
        % (
            num_samples,
            burn_in_rhat_quantile,
            beta_rhat_q,
            beta_rhat_max,
            num_active_betas,
            num_full_gene_sets,
            burn_in_pass_streak,
            burn_in_patience,
            stop_pass_streak,
            stop_patience,
        ),
        INFO,
    )
    if burn_in_pass_streak >= burn_in_patience:
        in_burn_in, burn_in_pass_streak, stop_pass_streak = _end_gibbs_burn_in(
            post_burn_reset_arrays=post_burn_reset_arrays,
            post_burn_reset_missing_arrays=post_burn_reset_missing_arrays,
            burn_in_pass_streak=burn_in_pass_streak,
            stop_pass_streak=stop_pass_streak,
        )
        log(
            "Burn-in complete at iter %d (beta R-hat q=%.2f on active betas, threshold=%.4g, patience=%d)"
            % (num_samples, burn_in_rhat_quantile, r_threshold_burn_in, burn_in_patience),
            INFO,
        )
    elif burn_stall_detected:
        in_burn_in, burn_in_pass_streak, stop_pass_streak = _end_gibbs_burn_in(
            post_burn_reset_arrays=post_burn_reset_arrays,
            post_burn_reset_missing_arrays=post_burn_reset_missing_arrays,
            burn_in_pass_streak=burn_in_pass_streak,
            stop_pass_streak=stop_pass_streak,
            reset_burn_in_pass_streak=True,
        )
        log(
            "Stopping Gibbs burn in at iter %d due to stall detectors (plateau=%s, recent_worse=%s)"
            % (num_samples, str(burn_stall_plateau), str(burn_stall_recent_worse)),
            INFO,
        )
    elif burn_window_plateau_detected:
        in_burn_in, burn_in_pass_streak, stop_pass_streak = _end_gibbs_burn_in(
            post_burn_reset_arrays=post_burn_reset_arrays,
            post_burn_reset_missing_arrays=post_burn_reset_missing_arrays,
            burn_in_pass_streak=burn_in_pass_streak,
            stop_pass_streak=stop_pass_streak,
            reset_burn_in_pass_streak=True,
        )
        log(
            "Stopping Gibbs burn in at iter %d due to R-hat plateau (window=%d, span=%.4g < delta=%.4g)"
            % (num_samples, burn_in_stall_window, burn_window_span, burn_in_stall_delta),
            INFO,
        )
    return (in_burn_in, burn_in_pass_streak, stop_pass_streak, burn_stall_beta_indices, R_beta_v)


def _update_gibbs_burn_in_state(
    epoch_control,
    iteration_num,
    epoch_total_iter_offset,
    epoch_max_num_iter,
    max_num_burn_in_for_epoch,
    min_num_iter_for_epoch,
    min_num_burn_in_for_epoch,
    post_burn_reset_arrays,
    post_burn_reset_missing_arrays,
    burn_in_config,
    iter_state,
    epoch_runtime,
):
    in_burn_in = epoch_control["in_burn_in"]
    prev_Ys_m = epoch_control["prev_Ys_m"]
    burn_in_pass_streak = epoch_control["burn_in_pass_streak"]
    burn_stall_beta_indices = epoch_control["burn_stall_beta_indices"]
    stop_pass_streak = epoch_control["stop_pass_streak"]
    R_beta_v = epoch_control["R_beta_v"]

    gauss_seidel = burn_in_config["gauss_seidel"]
    eps = burn_in_config["eps"]
    diag_every = burn_in_config["diag_every"]

    Y_sample_m = iter_state["Y_sample_m"]

    if not in_burn_in:
        return _build_gibbs_burn_in_control_from_epoch(epoch_control)

    num_samples = iteration_num + 1
    if num_samples >= max_num_burn_in_for_epoch:
        in_burn_in, burn_in_pass_streak, stop_pass_streak = _handle_gibbs_burn_in_max_iter(
            num_samples=num_samples,
            post_burn_reset_arrays=post_burn_reset_arrays,
            post_burn_reset_missing_arrays=post_burn_reset_missing_arrays,
            burn_in_pass_streak=burn_in_pass_streak,
            stop_pass_streak=stop_pass_streak,
        )
    elif gauss_seidel:
        in_burn_in, burn_in_pass_streak, stop_pass_streak, prev_Ys_m = _handle_gibbs_burn_in_gauss_seidel_path(
            num_samples=num_samples,
            epoch_total_iter_offset=epoch_total_iter_offset,
            min_num_iter_for_epoch=min_num_iter_for_epoch,
            eps=eps,
            Y_sample_m=Y_sample_m,
            prev_Ys_m=prev_Ys_m,
            post_burn_reset_arrays=post_burn_reset_arrays,
            post_burn_reset_missing_arrays=post_burn_reset_missing_arrays,
            burn_in_pass_streak=burn_in_pass_streak,
            stop_pass_streak=stop_pass_streak,
        )
    elif num_samples >= min_num_burn_in_for_epoch and (num_samples % diag_every == 0 or num_samples == epoch_max_num_iter):
        in_burn_in, burn_in_pass_streak, stop_pass_streak, burn_stall_beta_indices, R_beta_v = _handle_gibbs_burn_in_diag_path(
            num_samples=num_samples,
            epoch_control=epoch_control,
            burn_in_config=burn_in_config,
            min_num_burn_in_for_epoch=min_num_burn_in_for_epoch,
            epoch_runtime=epoch_runtime,
            post_burn_reset_arrays=post_burn_reset_arrays,
            post_burn_reset_missing_arrays=post_burn_reset_missing_arrays,
            burn_in_pass_streak=burn_in_pass_streak,
            stop_pass_streak=stop_pass_streak,
        )

    return _build_gibbs_burn_in_control_update(
        in_burn_in=in_burn_in,
        burn_in_pass_streak=burn_in_pass_streak,
        stop_pass_streak=stop_pass_streak,
        prev_Ys_m=prev_Ys_m,
        burn_stall_beta_indices=burn_stall_beta_indices,
        R_beta_v=R_beta_v,
    )


def _record_gibbs_hyper_mutation_event(state, event):
    event_json = json.dumps(event, sort_keys=True)
    state._record_param("gibbs_hyper_mutation_event", event_json)

    prev_count = state.params.get("gibbs_hyper_mutation_event_count", 0)
    if isinstance(prev_count, list):
        prev_count = prev_count[-1]
    try:
        prev_count = int(prev_count)
    except (TypeError, ValueError):
        prev_count = 0
    state._record_param("gibbs_hyper_mutation_event_count", prev_count + 1, overwrite=True)
    log("Gibbs hyper mutation event: %s" % event_json, INFO)


def _maybe_restart_gibbs_for_low_betas(
    state,
    increase_hyper_if_betas_below_for_epoch,
    experimental_hyper_mutation,
    num_before_checking_p_increase,
    p_scale_factor,
    epoch_runtime,
    epoch_sums,
    num_mad,
    num_attempts,
    max_num_attempt_restarts,
    iteration_num,
):
    all_sum_betas_m = epoch_runtime["all_sum_betas_m"]
    all_num_sum_m = epoch_runtime["all_num_sum_m"]
    num_p_increases = epoch_runtime["num_p_increases"]

    num_sum_beta_m = epoch_sums["num_sum_beta_m"]
    sum_betas_m = epoch_sums["sum_betas_m"]

    gibbs_good = True
    should_break = False

    if increase_hyper_if_betas_below_for_epoch is None:
        return GibbsLowBetaRestartUpdate(
            gibbs_good=gibbs_good,
            num_p_increases=num_p_increases,
            should_break=should_break,
        )

    # Check to make sure that we satisfy the hyperparameter growth criteria.
    if np.any(all_num_sum_m == 0):
        gibbs_good = False
        return GibbsLowBetaRestartUpdate(
            gibbs_good=gibbs_good,
            num_p_increases=num_p_increases,
            should_break=should_break,
        )

    # Check both sums over all iterations and post-burn aggregates.
    _outlier_resistant_mean(all_sum_betas_m, all_num_sum_m, num_mad, record_param_fn=state._record_param)

    fraction_required = 0.001
    state._record_param("fraction_required_to_not_increase_hyper", fraction_required)

    (
        has_post_burn_beta_samples,
        cur_avg_betas_v,
        all_low,
    ) = _evaluate_gibbs_low_beta_condition(
        state=state,
        sum_betas_m=sum_betas_m,
        num_sum_beta_m=num_sum_beta_m,
        num_mad=num_mad,
        increase_hyper_if_betas_below_for_epoch=increase_hyper_if_betas_below_for_epoch,
        fraction_required=fraction_required,
    )

    if has_post_burn_beta_samples:
        top_gene_set = np.argmax(np.mean(sum_betas_m / num_sum_beta_m, axis=0) / state.scale_factors)
        log("Top gene set %s has value %.3g" % (state.gene_sets[top_gene_set], (np.mean(sum_betas_m / num_sum_beta_m, axis=0) / state.scale_factors)[top_gene_set]), TRACE)
        top_gene_set2 = np.argmax(cur_avg_betas_v / state.scale_factors)
        log("Top gene set %s has outlier value %.3g" % (state.gene_sets[top_gene_set2], (cur_avg_betas_v / state.scale_factors)[top_gene_set2]), TRACE)

    if all_low:
        fraction_above_threshold = float(
            np.mean(cur_avg_betas_v / state.scale_factors > increase_hyper_if_betas_below_for_epoch)
        )
        log(
            "Only %.3g of %d (%.3g) are above %.3g"
            % (
                np.sum(cur_avg_betas_v / state.scale_factors > increase_hyper_if_betas_below_for_epoch),
                len(cur_avg_betas_v),
                fraction_above_threshold,
                increase_hyper_if_betas_below_for_epoch,
            )
        )

        # At minimum, guarantee that it will restart unless it gets above this.
        gibbs_good = False
        if not experimental_hyper_mutation:
            state._record_param("gibbs_no_signal_detected", 1, overwrite=True)
            state._record_param(
                "gibbs_no_signal_fraction_above_threshold",
                fraction_above_threshold,
                overwrite=True,
            )
            state._record_param(
                "gibbs_no_signal_threshold",
                increase_hyper_if_betas_below_for_epoch,
                overwrite=True,
            )
            bail(
                "Detected no-signal Gibbs condition (fraction above threshold %.6g < %.6g at threshold %.6g). "
                "Default behavior is explicit failure without hyper mutation. "
                "To enable legacy restart/mutation heuristic, pass --experimental-hyper-mutation with --experimental-increase-hyper-if-betas-below."
                % (
                    fraction_above_threshold,
                    fraction_required,
                    increase_hyper_if_betas_below_for_epoch,
                )
            )
        # Only if above num for checking though that we increase and restart.
        num_p_increases, should_break = _maybe_increase_gibbs_hyper_and_restart(
            state=state,
            increase_hyper_if_betas_below_for_epoch=increase_hyper_if_betas_below_for_epoch,
            fraction_above_threshold=fraction_above_threshold,
            num_before_checking_p_increase=num_before_checking_p_increase,
            p_scale_factor=p_scale_factor,
            num_p_increases=num_p_increases,
            num_attempts=num_attempts,
            max_num_attempt_restarts=max_num_attempt_restarts,
            iteration_num=iteration_num,
        )
    else:
        gibbs_good = True

    return GibbsLowBetaRestartUpdate(
        gibbs_good=gibbs_good,
        num_p_increases=num_p_increases,
        should_break=should_break,
    )


def _evaluate_gibbs_low_beta_condition(
    state,
    sum_betas_m,
    num_sum_beta_m,
    num_mad,
    increase_hyper_if_betas_below_for_epoch,
    fraction_required,
):
    has_post_burn_beta_samples = np.all(num_sum_beta_m > 0)
    cur_avg_betas_v = None
    all_low = False
    if has_post_burn_beta_samples:
        _, cur_avg_betas_v = _outlier_resistant_mean(
            sum_betas_m,
            num_sum_beta_m,
            num_mad,
            record_param_fn=state._record_param,
        )
        all_low = (
            np.mean(cur_avg_betas_v / state.scale_factors > increase_hyper_if_betas_below_for_epoch)
            < fraction_required
        )
    return (has_post_burn_beta_samples, cur_avg_betas_v, all_low)


def _maybe_increase_gibbs_hyper_and_restart(
    state,
    increase_hyper_if_betas_below_for_epoch,
    fraction_above_threshold,
    num_before_checking_p_increase,
    p_scale_factor,
    num_p_increases,
    num_attempts,
    max_num_attempt_restarts,
    iteration_num,
):
    should_break = False
    if iteration_num <= num_before_checking_p_increase:
        return (num_p_increases, should_break)

    old_p = state.p
    old_sigma2 = state.sigma2
    new_p = old_p
    new_sigma2 = old_sigma2

    state._record_param("p_scale_factor", p_scale_factor)

    new_p = state.p * p_scale_factor
    num_p_increases += 1
    if new_p > 1:
        new_p = 1

    break_loop = False
    if new_p != state.p and num_attempts < max_num_attempt_restarts:
        # Update so that new_sigma2 / new_p = self.sigma2 / self.p
        new_sigma2 = state.sigma2 * new_p / state.p

        state.ps *= new_p / state.p
        state.set_p(new_p)
        state._record_param("p_adj", new_p)
        log(
            "Detected all gene set betas below %.3g; increasing p to %.3g and restarting gibbs"
            % (increase_hyper_if_betas_below_for_epoch, state.p)
        )

        # Restart.
        break_loop = True
    if new_sigma2 != state.sigma2 and num_attempts < max_num_attempt_restarts:
        state.sigma2s *= new_sigma2 / state.sigma2
        state._record_param("sigma2_adj", new_sigma2)
        state.set_sigma(new_sigma2, state.sigma_power)
        log(
            "Detected all gene set betas below %.3g; increasing sigma to %.3g and restarting gibbs"
            % (increase_hyper_if_betas_below_for_epoch, state.sigma2)
        )
        break_loop = True
    if break_loop:
        _record_gibbs_hyper_mutation_event(
            state,
            {
                "event": "gibbs_hyper_mutation_restart",
                "trigger": "all_betas_below_threshold",
                "threshold": float(increase_hyper_if_betas_below_for_epoch),
                "fraction_above_threshold": float(fraction_above_threshold),
                "iteration_num": int(iteration_num),
                "num_attempts": int(num_attempts),
                "max_num_attempt_restarts": int(max_num_attempt_restarts),
                "p_scale_factor": float(p_scale_factor),
                "old_p": float(old_p),
                "new_p": float(state.p),
                "old_sigma2": float(old_sigma2),
                "new_sigma2": float(state.sigma2),
            },
        )
        should_break = True
    return (num_p_increases, should_break)


def _safe_quantile(values, q, fallback):
    finite_values = values[np.isfinite(values)]
    if finite_values.size == 0:
        return fallback
    q = min(max(q, 0.0), 1.0)
    return float(np.quantile(finite_values, q))


def _prepare_stall_indices(mask_v, fallback_scores_v, fallback_k):
    if mask_v is not None and np.any(mask_v):
        return np.where(mask_v)[0]
    if fallback_scores_v is None or len(fallback_scores_v) == 0:
        return np.array([], dtype=int)
    k = min(max(fallback_k, 1), len(fallback_scores_v))
    if k >= len(fallback_scores_v):
        return np.arange(len(fallback_scores_v))
    return np.argpartition(-np.abs(fallback_scores_v), k - 1)[:k]


def _means_from_sums(sum_m, num_sum_m):
    return np.divide(sum_m, np.maximum(num_sum_m, 1.0))


def _apply_inner_beta_sparsity_update(
    sparse_solution,
    sparse_frac_betas,
    curr_postp_t,
    ps_m,
    curr_post_means_t,
    curr_betas_t,
    compute_mask_v,
):
    if not sparse_solution:
        return

    sparse_mask_t = curr_postp_t < ps_m
    if sparse_frac_betas is not None:
        # Zero out very small values relative to the top within each chain/parallel slice.
        relative_value = np.max(np.abs(curr_post_means_t), axis=2)
        sparse_mask_t = np.logical_or(
            sparse_mask_t,
            (np.abs(curr_post_means_t).T < sparse_frac_betas * relative_value.T).T,
        )

    # Do not sparsify slices that are currently outside the compute mask.
    sparse_mask_t[:, np.logical_not(compute_mask_v), :] = False
    log(
        "Setting %d entries to zero due to sparsity"
        % np.sum(np.logical_and(sparse_mask_t, curr_betas_t > 0)),
        TRACE,
    )
    curr_betas_t[sparse_mask_t] = 0
    curr_post_means_t[sparse_mask_t] = 0


def _write_inner_beta_trace_rows(
    betas_trace_fh,
    iteration_num,
    num_parallel,
    num_chains,
    num_gene_sets,
    betas_trace_gene_sets,
    curr_post_means_t,
    curr_betas_t,
    curr_postp_t,
    res_beta_hat_t,
    scale_factors_m,
    beta_tildes_m,
    ses_m,
    sigma2_m,
    ps_m,
    R_m,
    beta_weights_m,
    sem2_m,
):
    for parallel_num in range(num_parallel):
        for chain_num in range(num_chains):
            for gene_set_idx in range(num_gene_sets):
                gene_set = gene_set_idx
                if betas_trace_gene_sets is not None and len(betas_trace_gene_sets) == num_gene_sets:
                    gene_set = betas_trace_gene_sets[gene_set_idx]

                betas_trace_fh.write(
                    "%d\t%d\t%d\t%s\t%.4g\t%.4g\t%.4g\t%.4g\t%.4g\t%.4g\t%.4g\t%.4g\t%.4g\t%.4g\t%.4g\t%.4g\t%.4g\t%.4g\n"
                    % (
                        iteration_num,
                        parallel_num + 1,
                        chain_num + 1,
                        gene_set,
                        curr_post_means_t[chain_num, parallel_num, gene_set_idx]
                        / scale_factors_m[parallel_num, gene_set_idx],
                        curr_betas_t[chain_num, parallel_num, gene_set_idx]
                        / scale_factors_m[parallel_num, gene_set_idx],
                        curr_postp_t[chain_num, parallel_num, gene_set_idx],
                        res_beta_hat_t[chain_num, parallel_num, gene_set_idx]
                        / scale_factors_m[parallel_num, gene_set_idx],
                        beta_tildes_m[parallel_num, gene_set_idx]
                        / scale_factors_m[parallel_num, gene_set_idx],
                        curr_betas_t[chain_num, parallel_num, gene_set_idx],
                        res_beta_hat_t[chain_num, parallel_num, gene_set_idx],
                        beta_tildes_m[parallel_num, gene_set_idx],
                        ses_m[parallel_num, gene_set_idx],
                        sigma2_m[parallel_num, gene_set_idx]
                        if len(np.shape(sigma2_m)) > 0
                        else sigma2_m,
                        ps_m[parallel_num, gene_set_idx] if len(np.shape(ps_m)) > 0 else ps_m,
                        R_m[parallel_num, gene_set_idx],
                        R_m[parallel_num, gene_set_idx]
                        * beta_weights_m[parallel_num, gene_set_idx],
                        sem2_m[parallel_num, gene_set_idx],
                    )
                )

    betas_trace_fh.flush()


def _compute_inner_beta_hyper_update_targets(
    curr_postp_t,
    res_beta_hat_t,
    hdmp_hdmpn_m,
    se2s_m,
    curr_betas_m,
    V_diag_m,
    num_parallel,
    use_X,
    multiple_V,
    V,
    sparse_V,
    num_p_pseudo,
    curr_postp_m,
    sigma_power,
    scale_factors_m,
    num_gene_sets,
    num_missing_gene_sets,
    p_noninf_inflate,
):
    # Hyper-updates use Rao-Blackwellized moments to avoid sigma collapse.
    # Conditional slab mean m = hdmp_hdmpn * res_beta_hat.
    cond_mean_t = hdmp_hdmpn_m[np.newaxis, :, :] * res_beta_hat_t
    # Conditional slab variance v = hdmp_hdmpn * se2.
    cond_var_m = hdmp_hdmpn_m * se2s_m
    # E[beta^2] = postp * (m^2 + v).
    e_beta2_m = np.mean(curr_postp_t * (np.square(cond_mean_t) + cond_var_m[np.newaxis, :, :]), axis=0)
    # mu = E[beta].
    mu_m = curr_betas_m
    # Var(beta) = E[beta^2] - (E[beta])^2.
    var_m = e_beta2_m - np.square(mu_m)
    var_m[var_m < 0] = 0.0

    if V_diag_m is None:
        diag_m = np.ones(mu_m.shape)
    else:
        diag_m = V_diag_m

    h2 = 0.0
    for i in range(num_parallel):
        if use_X:
            # In X-implicit mode, use diagonal approximation directly from E[beta^2].
            h2 += float(np.sum(e_beta2_m[i, :]))
        else:
            if multiple_V:
                cur_V = V[i, :, :]
            else:
                cur_V = V

            mu_v = mu_m[i, :]
            if sparse_V:
                muVmu = float(mu_v.dot(cur_V.dot(mu_v)))
            else:
                muVmu = float(mu_v.dot(cur_V).dot(mu_v))
            h2 += muVmu + float(np.sum(diag_m[i, :] * var_m[i, :]))
    h2 /= float(num_parallel)

    # Rao-Blackwellized p update from posterior inclusion probabilities.
    if num_p_pseudo is not None and num_p_pseudo > 0:
        a0 = float(num_p_pseudo)
        b0 = float(num_p_pseudo)
        sum_r = float(np.sum(curr_postp_m))
        m_tot = float(curr_postp_m.size)
        new_p = (a0 + sum_r) / (a0 + b0 + m_tot)
    else:
        new_p = float(np.mean(curr_postp_m))

    if sigma_power is not None:
        new_sigma2 = h2 / np.mean(np.sum(np.power(scale_factors_m, sigma_power), axis=1))
    else:
        new_sigma2 = h2 / num_gene_sets

    if num_missing_gene_sets:
        missing_scale_factor = num_gene_sets / (num_gene_sets + num_missing_gene_sets)
        new_sigma2 *= missing_scale_factor
        new_p *= missing_scale_factor

    if p_noninf_inflate != 1:
        log("Inflating p by %.3g" % p_noninf_inflate, DEBUG)
        new_p *= p_noninf_inflate

    return (new_p, new_sigma2)


def _update_inner_beta_gene_set_batch(
    compute_mask_m,
    compute_mask_v,
    alpha_shrink,
    rand_ps_t,
    rand_norms_t,
    hdmp_hdmpn_m,
    c_const_m,
    d_const_m,
    hdmpn_m,
    se2s_m,
    norm_scale_m,
    assume_independent,
    beta_tildes_m,
    curr_betas_t,
    V,
    multiple_V,
    sparse_V,
    use_X,
    X_orig,
    scale_factors_m,
    mean_shifts_m,
    betas_trace_out,
    betas_trace_gene_sets,
    account_for_V_diag_m,
    V_diag_m,
    curr_postp_t,
    curr_post_means_t,
    gauss_seidel,
    res_beta_hat_t,
):
    # 1) Build residualized beta_tilde for the active batch.
    compute_mask_union = np.any(compute_mask_m, axis=0)
    compute_mask_union_filter_m = compute_mask_m[:, compute_mask_union]

    if assume_independent:
        res_beta_hat_t_flat = beta_tildes_m[compute_mask_m]
    else:
        current_num_parallel = sum(compute_mask_v)

        if multiple_V:
            # Pointwise matmul across active parallels while preserving chain dimension.
            res_beta_hat_union_t = np.einsum(
                "hij,ijk->hik",
                curr_betas_t[:, compute_mask_v, :],
                V[compute_mask_v, :, :][:, :, compute_mask_union],
            ).reshape((curr_betas_t.shape[0], current_num_parallel, np.sum(compute_mask_union)))
        elif sparse_V:
            res_beta_hat_union_t = V[compute_mask_union, :].dot(
                curr_betas_t[:, compute_mask_v, :].T.reshape(
                    (curr_betas_t.shape[2], np.sum(compute_mask_v) * curr_betas_t.shape[0])
                )
            ).reshape((np.sum(compute_mask_union), np.sum(compute_mask_v), curr_betas_t.shape[0])).T
        elif use_X:
            if len(compute_mask_union.shape) == 2:
                assert compute_mask_union.shape[0] == 1
                compute_mask_union = np.squeeze(compute_mask_union)

            curr_betas_filtered_t = curr_betas_t[:, compute_mask_v, :] / scale_factors_m[compute_mask_v, :]
            interm = X_orig.dot(
                curr_betas_filtered_t.T.reshape(
                    (curr_betas_filtered_t.shape[2], curr_betas_filtered_t.shape[0] * curr_betas_filtered_t.shape[1])
                )
            ).reshape((X_orig.shape[0], curr_betas_filtered_t.shape[1], curr_betas_filtered_t.shape[0])) - np.sum(
                mean_shifts_m[compute_mask_v, :] * curr_betas_filtered_t, axis=2
            ).T

            # This path can be sensitive when some parallels converge earlier.
            res_beta_hat_union_t = (
                X_orig[:, compute_mask_union]
                .T.dot(interm.reshape((interm.shape[0], interm.shape[1] * interm.shape[2])))
                .reshape((np.sum(compute_mask_union), interm.shape[1], interm.shape[2]))
                - mean_shifts_m.T[compute_mask_union, :][:, compute_mask_v, np.newaxis] * np.sum(interm, axis=0)
            ).T
            res_beta_hat_union_t /= (X_orig.shape[0] * scale_factors_m[compute_mask_v, :][:, compute_mask_union])
        else:
            res_beta_hat_union_t = curr_betas_t[:, compute_mask_v, :].dot(V[:, compute_mask_union])

        if betas_trace_out is not None and betas_trace_gene_sets is not None:
            cur_sets = [betas_trace_gene_sets[x] for x in range(len(betas_trace_gene_sets)) if compute_mask_union[x]]
            pegs_construct_map_to_ind(betas_trace_gene_sets)
            pegs_construct_map_to_ind(cur_sets)

        res_beta_hat_t_flat = res_beta_hat_union_t[:, compute_mask_union_filter_m[compute_mask_v, :]]
        assert res_beta_hat_t_flat.shape[1] == np.sum(compute_mask_m)
        res_beta_hat_t_flat = beta_tildes_m[compute_mask_m] - res_beta_hat_t_flat

        if account_for_V_diag_m:
            res_beta_hat_t_flat = res_beta_hat_t_flat + V_diag_m[compute_mask_m] * curr_betas_t[:, compute_mask_m]
        else:
            res_beta_hat_t_flat = res_beta_hat_t_flat + curr_betas_t[:, compute_mask_m]

    # 2) Convert residualized effect to inclusion probabilities.
    b2_t_flat = np.power(res_beta_hat_t_flat, 2)
    d_const_b2_exp_t_flat = d_const_m[compute_mask_m] * np.exp(-b2_t_flat / (se2s_m[compute_mask_m] * 2.0))
    numerator_t_flat = c_const_m[compute_mask_m] * np.exp(-b2_t_flat / (2.0 * hdmpn_m[compute_mask_m]))
    numerator_zero_mask_t_flat = numerator_t_flat == 0
    denominator_t_flat = numerator_t_flat + d_const_b2_exp_t_flat
    denominator_t_flat[numerator_zero_mask_t_flat] = 1

    d_imaginary_mask_t_flat = ~np.isreal(d_const_b2_exp_t_flat)
    numerator_imaginary_mask_t_flat = ~np.isreal(numerator_t_flat)

    if np.any(np.logical_or(d_imaginary_mask_t_flat, numerator_imaginary_mask_t_flat)):
        warn("Detected imaginary numbers!")
        denominator_t_flat[d_imaginary_mask_t_flat] = numerator_t_flat[d_imaginary_mask_t_flat]
        numerator_t_flat[np.logical_and(~d_imaginary_mask_t_flat, numerator_imaginary_mask_t_flat)] = 0

    curr_postp_t[:, compute_mask_m] = numerator_t_flat / denominator_t_flat

    # 3) Update conditional means and sampled betas for this batch.
    curr_post_means_t[:, compute_mask_m] = hdmp_hdmpn_m[compute_mask_m] * (
        curr_postp_t[:, compute_mask_m] * res_beta_hat_t_flat
    )

    if gauss_seidel:
        proposed_beta_t_flat = curr_post_means_t[:, compute_mask_m]
    else:
        norm_mean_t_flat = hdmp_hdmpn_m[compute_mask_m] * res_beta_hat_t_flat
        proposed_beta_t_flat = norm_mean_t_flat + norm_scale_m[compute_mask_m] * rand_norms_t[:, compute_mask_m]
        zero_mask_t_flat = rand_ps_t[:, compute_mask_m] >= curr_postp_t[:, compute_mask_m] * alpha_shrink
        proposed_beta_t_flat[zero_mask_t_flat] = 0

    curr_betas_t[:, compute_mask_m] = proposed_beta_t_flat
    res_beta_hat_t[:, compute_mask_m] = res_beta_hat_t_flat


def _update_inner_beta_rhat_and_outliers(
    sum_betas_t,
    sum_betas2_t,
    iteration_num,
    compute_mask_v,
    use_max_r_for_convergence,
    beta_outlier_iqr_threshold,
    curr_betas_t,
    curr_postp_t,
    curr_post_means_t,
    burn_in_phase_v,
    r_threshold_burn_in,
    num_parallel,
):
    # These matrices have convergence statistics in format (num_parallel, num_gene_sets).
    # WARNING: only results for compute_mask_v are valid.
    (B_m, W_m, R_m, avg_W_m, mean_t) = _calculate_r_tensor_from_chain_sums(sum_betas_t, sum_betas2_t, iteration_num)

    beta_weights_m = np.zeros((sum_betas_t.shape[1], sum_betas_t.shape[2]))
    sum_betas_t_mean = np.mean(sum_betas_t)
    if sum_betas_t_mean > 0:
        np.mean(sum_betas_t, axis=0) / sum_betas_t_mean

    # Calculate thresholded/scaled R summary.
    num_R_above_1_v = np.sum(R_m >= 1, axis=1)
    num_R_above_1_v[num_R_above_1_v == 0] = 1

    R_m_above_1 = copy.copy(R_m)
    R_m_above_1[R_m_above_1 < 1] = 0
    mean_thresholded_R_v = np.sum(R_m_above_1, axis=1) / num_R_above_1_v

    # Max R for each parallel run.
    max_index_v = np.argmax(R_m, axis=1)
    max_index_parallel = None
    max_val = None
    for i in range(len(max_index_v)):
        if compute_mask_v[i] and (max_val is None or R_m[i, max_index_v[i]] > max_val):
            max_val = R_m[i, max_index_v[i]]
            max_index_parallel = i
    max_R_v = np.max(R_m, axis=1)

    if use_max_r_for_convergence:
        convergence_statistic_v = max_R_v
    else:
        convergence_statistic_v = mean_thresholded_R_v

    outlier_mask_m = np.full(avg_W_m.shape, False)
    if avg_W_m.shape[0] > 10:
        # Check per-chain oscillation via variance IQR.
        q3, median, q1 = np.percentile(avg_W_m, [75, 50, 25], axis=0)
        iqr_mask = q3 > q1
        chain_iqr_m = np.zeros(avg_W_m.shape)
        chain_iqr_m[:, iqr_mask] = (avg_W_m[:, iqr_mask] - median[iqr_mask]) / (q3 - q1)[iqr_mask]
        # dimensions chain x parallel
        outlier_mask_m = beta_outlier_iqr_threshold
        if np.sum(outlier_mask_m) > 0:
            log("Detected %d outlier chains due to oscillations" % np.sum(outlier_mask_m), DEBUG)

    if np.sum(R_m > 1) > 10:
        # Check extreme high-R outliers.
        q3, median, q1 = np.percentile(R_m[R_m > 1], [75, 50, 25])
        if q3 > q1:
            R_iqr_m = (R_m - median) / (q3 - q1)
            bad_gene_sets_m = np.logical_and(R_iqr_m > 100, R_m > 2.5)
            if np.sum(bad_gene_sets_m) > 0:
                bad_chains = np.argmax(np.abs(mean_t - np.mean(mean_t, axis=0)), axis=0)[bad_gene_sets_m]

                cur_outlier_mask_m = np.zeros(outlier_mask_m.shape)
                cur_outlier_mask_m[bad_chains, np.where(bad_gene_sets_m)[0]] = True

                log(
                    "Found %d outlier chains across %d parallel runs due to %d gene sets with high R (%.4g - %.4g; %.4g - %.4g)"
                    % (
                        np.sum(cur_outlier_mask_m),
                        np.sum(np.any(cur_outlier_mask_m, axis=0)),
                        np.sum(bad_gene_sets_m),
                        np.min(R_m[bad_gene_sets_m]),
                        np.max(R_m[bad_gene_sets_m]),
                        np.min(R_iqr_m[bad_gene_sets_m]),
                        np.max(R_iqr_m[bad_gene_sets_m]),
                    ),
                    DEBUG,
                )
                outlier_mask_m = np.logical_or(outlier_mask_m, cur_outlier_mask_m)

    non_outliers_m = ~outlier_mask_m
    if np.sum(outlier_mask_m) > 0:
        log("Detected %d total outlier chains" % np.sum(outlier_mask_m), DEBUG)
        # Dimensions are num_chains x num_parallel.
        for outlier_parallel in np.where(np.any(outlier_mask_m, axis=0))[0]:
            if np.sum(outlier_mask_m[:, outlier_parallel]) > 0:
                if np.sum(non_outliers_m[:, outlier_parallel]) > 0:
                    replacement_chains = np.random.choice(
                        np.where(non_outliers_m[:, outlier_parallel])[0],
                        size=np.sum(outlier_mask_m[:, outlier_parallel]),
                    )
                    log(
                        "Replaced chains %s with chains %s in parallel %d"
                        % (
                            np.where(outlier_mask_m[:, outlier_parallel])[0],
                            replacement_chains,
                            outlier_parallel,
                        ),
                        DEBUG,
                    )

                    for tensor in [curr_betas_t, curr_postp_t, curr_post_means_t, sum_betas_t, sum_betas2_t]:
                        tensor[outlier_mask_m[:, outlier_parallel], outlier_parallel, :] = copy.copy(
                            tensor[replacement_chains, outlier_parallel, :]
                        )
                else:
                    log("Every chain was an outlier so doing nothing", TRACE)

    log(
        "Iteration %d: max ind=%s; max B=%.3g; max W=%.3g; max R=%.4g; avg R=%.4g; num above=%.4g;"
        % (
            iteration_num,
            (max_index_parallel, max_index_v[max_index_parallel]) if num_parallel > 1 else max_index_v[max_index_parallel],
            B_m[max_index_parallel, max_index_v[max_index_parallel]],
            W_m[max_index_parallel, max_index_v[max_index_parallel]],
            R_m[max_index_parallel, max_index_v[max_index_parallel]],
            np.mean(mean_thresholded_R_v),
            np.sum(R_m > r_threshold_burn_in),
        ),
        TRACE,
    )

    converged_v = convergence_statistic_v < r_threshold_burn_in
    newly_converged_v = np.logical_and(burn_in_phase_v, converged_v)
    if np.sum(newly_converged_v) > 0:
        if num_parallel == 1:
            log("Converged after %d iterations" % iteration_num, INFO)
        else:
            log(
                "Parallel %s converged after %d iterations"
                % (",".join([str(p) for p in np.nditer(np.where(newly_converged_v))]), iteration_num),
                INFO,
            )
        burn_in_phase_v = np.logical_and(burn_in_phase_v, np.logical_not(converged_v))

    return (R_m, beta_weights_m, burn_in_phase_v)


def _calculate_r_tensor_from_chain_sums(sum_betas_t, sum_betas2_t, num):
    num_chains = sum_betas_t.shape[0]
    num_parallel = sum_betas_t.shape[1]
    num_gene_sets = sum_betas_t.shape[2]

    mean_t = sum_betas_t / float(max(num, 1))
    if num <= 1:
        ones_m = np.ones((num_parallel, num_gene_sets))
        avg_W_m = np.ones((num_chains, num_parallel))
        return (ones_m, ones_m, ones_m, avg_W_m, mean_t)

    mean_m = np.mean(mean_t, axis=0)
    var_t = (sum_betas2_t - float(num) * np.square(mean_t)) / (float(num) - 1)
    b_denom = max(num_chains - 1, 1)
    B_m = (float(num) / float(b_denom)) * np.sum(np.square(mean_t - mean_m), axis=0)
    W_m = np.mean(var_t, axis=0)
    var_given_y_m = np.add((float(num) - 1) / float(num) * W_m, (1.0 / float(num)) * B_m)
    var_given_y_m[var_given_y_m < 0] = 0
    R_m = np.ones((num_parallel, num_gene_sets))
    R_non_zero_mask = W_m > 0
    R_m[R_non_zero_mask] = np.sqrt(var_given_y_m[R_non_zero_mask] / W_m[R_non_zero_mask])
    avg_W_m = np.mean(var_t, axis=2)
    return (B_m, W_m, R_m, avg_W_m, mean_t)


def _calculate_rhat_from_sums(sum_m, sum2_m, num):
    if num <= 1:
        default_v = np.ones(sum_m.shape[1])
        return (default_v, default_v, default_v, default_v)
    mean_m = sum_m / float(num)
    mean_v = np.mean(mean_m, axis=0)
    var_m = (sum2_m - float(num) * np.power(mean_m, 2)) / (float(num) - 1)
    B_v = (float(num) / (mean_m.shape[0] - 1)) * np.sum(np.power(mean_m - mean_v, 2), axis=0)
    W_v = (1.0 / float(mean_m.shape[0])) * np.sum(var_m, axis=0)
    var_given_y_v = np.add((float(num) - 1) / float(num) * W_v, (1.0 / float(num)) * B_v)
    var_given_y_v[var_given_y_v < 0] = 0
    R_v = np.ones(len(W_v))
    R_non_zero_mask = W_v > 0
    R_v[R_non_zero_mask] = np.sqrt(var_given_y_v[R_non_zero_mask] / W_v[R_non_zero_mask])
    return (B_v, W_v, R_v, var_given_y_v)


def _get_active_beta_mask(sum_betas_for_diag_m, num_sum_beta_for_diag_m, active_beta_top_k, active_beta_min_abs):
    beta_chain_means_m = _means_from_sums(sum_betas_for_diag_m, num_sum_beta_for_diag_m)
    beta_mean_v = np.mean(beta_chain_means_m, axis=0)
    abs_beta_mean_v = np.abs(beta_mean_v)

    num_beta = len(beta_mean_v)
    active_mask_v = np.zeros(num_beta, dtype=bool)
    if num_beta == 0:
        return (active_mask_v, beta_chain_means_m, beta_mean_v)

    top_k = min(max(active_beta_top_k, 1), num_beta)
    if top_k >= num_beta:
        active_mask_v[:] = True
    else:
        active_idx = np.argpartition(-abs_beta_mean_v, top_k - 1)[:top_k]
        active_mask_v[active_idx] = True

    if active_beta_min_abs > 0:
        filtered_mask_v = np.logical_and(active_mask_v, abs_beta_mean_v >= active_beta_min_abs)
        if np.any(filtered_mask_v):
            active_mask_v = filtered_mask_v

    return (active_mask_v, beta_chain_means_m, beta_mean_v)


def _initialize_gibbs_epoch_state(state, num_chains, num_full_gene_sets, use_mean_betas, max_mb_X_h, log_fun):
    full_betas_m_shape = (num_chains, num_full_gene_sets)
    prev_warm_start_betas_m = None
    prev_warm_start_postp_m = None
    sum_betas_m = np.zeros(full_betas_m_shape)
    sum_betas2_m = np.zeros(full_betas_m_shape)
    sum_betas_uncorrected_m = np.zeros(full_betas_m_shape)
    sum_betas_uncorrected2_m = np.zeros(full_betas_m_shape)
    sum_postp_m = np.zeros(full_betas_m_shape)
    sum_beta_tildes_m = np.zeros(full_betas_m_shape)
    sum_z_scores_m = np.zeros(full_betas_m_shape)
    num_sum_beta_m = np.zeros(full_betas_m_shape)

    Y_m_shape = (num_chains, len(state.Y_for_regression))
    sum_Ys_m = np.zeros(Y_m_shape)
    sum_Y_raws_m = np.zeros(Y_m_shape)
    sum_log_pos_m = np.zeros(Y_m_shape)
    sum_log_po_raws_m = np.zeros(Y_m_shape)
    sum_log_po_raws2_m = np.zeros(Y_m_shape)
    sum_priors_m = np.zeros(Y_m_shape)
    sum_priors2_m = np.zeros(Y_m_shape)
    sum_Ds_m = np.zeros(Y_m_shape)
    sum_D_raws_m = np.zeros(Y_m_shape)
    sum_bf_orig_m = np.zeros(Y_m_shape)
    sum_bf_orig_raw_m = np.zeros(Y_m_shape)
    sum_bf_orig_raw2_m = np.zeros(Y_m_shape)
    num_sum_Y_m = np.zeros(Y_m_shape)

    # Sums across all iterations, not just converged.
    all_sum_betas_m = np.zeros(full_betas_m_shape)
    all_sum_betas2_m = np.zeros(full_betas_m_shape)
    all_num_sum_m = np.zeros(full_betas_m_shape)

    # Initialize per-chain priors.
    priors_sample_m = np.zeros(Y_m_shape)
    priors_mean_m = np.zeros(Y_m_shape)
    priors_percentage_max_sample_m = np.zeros(Y_m_shape)
    priors_percentage_max_mean_m = np.zeros(Y_m_shape)
    priors_adjustment_sample_m = np.zeros(Y_m_shape)
    priors_adjustment_mean_m = np.zeros(Y_m_shape)

    priors_for_Y_m = priors_sample_m
    priors_percentage_max_for_Y_m = priors_percentage_max_sample_m
    priors_adjustment_for_Y_m = priors_adjustment_sample_m
    if use_mean_betas:
        priors_for_Y_m = priors_mean_m
        priors_percentage_max_for_Y_m = priors_percentage_max_mean_m
        priors_adjustment_for_Y_m = priors_adjustment_mean_m

    num_genes_missing = 0
    if state.genes_missing is not None:
        num_genes_missing = len(state.genes_missing)

    sum_priors_missing_m = np.zeros((num_chains, num_genes_missing))
    sum_Ds_missing_m = np.zeros((num_chains, num_genes_missing))
    priors_missing_sample_m = np.zeros(sum_priors_missing_m.shape)
    priors_missing_mean_m = np.zeros(sum_priors_missing_m.shape)
    num_sum_priors_missing_m = np.zeros(sum_priors_missing_m.shape)

    post_burn_reset_arrays = (
        sum_Ys_m,
        sum_Y_raws_m,
        sum_log_pos_m,
        sum_log_po_raws_m,
        sum_log_po_raws2_m,
        sum_priors_m,
        sum_priors2_m,
        sum_Ds_m,
        sum_D_raws_m,
        sum_bf_orig_m,
        sum_bf_orig_raw_m,
        sum_bf_orig_raw2_m,
        num_sum_Y_m,
        sum_betas_m,
        sum_betas2_m,
        sum_betas_uncorrected_m,
        sum_betas_uncorrected2_m,
        sum_postp_m,
        sum_beta_tildes_m,
        sum_z_scores_m,
        num_sum_beta_m,
    )
    post_burn_reset_missing_arrays = ()
    if state.genes_missing is not None:
        post_burn_reset_missing_arrays = (
            sum_priors_missing_m,
            sum_Ds_missing_m,
            num_sum_priors_missing_m,
        )

    _maybe_unsubset_gene_sets(state, state.gene_sets_missing is not None, skip_V=True)

    stack_batch_size = num_chains + 1
    if num_chains > 1:
        X_size_mb = state._get_X_size_mb()
        X_h_size_mb = num_chains * X_size_mb
        if X_h_size_mb <= max_mb_X_h:
            X_hstacked = sparse.hstack([state.X_orig] * num_chains)
        else:
            stack_batch_size = int(max_mb_X_h / X_size_mb)
            if stack_batch_size == 0:
                stack_batch_size = 1
            log_fun(
                "Not building X_hstacked, size would be %d > %d; will instead run %d chains at a time"
                % (X_h_size_mb, max_mb_X_h, stack_batch_size)
            )
            X_hstacked = sparse.hstack([state.X_orig] * stack_batch_size)
    else:
        X_hstacked = state.X_orig

    num_stack_batches = int(np.ceil(num_chains / float(stack_batch_size)))

    return {
        "full_betas_m_shape": full_betas_m_shape,
        "prev_warm_start_betas_m": prev_warm_start_betas_m,
        "prev_warm_start_postp_m": prev_warm_start_postp_m,
        "sum_betas_m": sum_betas_m,
        "sum_betas2_m": sum_betas2_m,
        "sum_betas_uncorrected_m": sum_betas_uncorrected_m,
        "sum_betas_uncorrected2_m": sum_betas_uncorrected2_m,
        "sum_postp_m": sum_postp_m,
        "sum_beta_tildes_m": sum_beta_tildes_m,
        "sum_z_scores_m": sum_z_scores_m,
        "num_sum_beta_m": num_sum_beta_m,
        "sum_Ys_m": sum_Ys_m,
        "sum_Y_raws_m": sum_Y_raws_m,
        "sum_log_pos_m": sum_log_pos_m,
        "sum_log_po_raws_m": sum_log_po_raws_m,
        "sum_log_po_raws2_m": sum_log_po_raws2_m,
        "sum_priors_m": sum_priors_m,
        "sum_priors2_m": sum_priors2_m,
        "sum_Ds_m": sum_Ds_m,
        "sum_D_raws_m": sum_D_raws_m,
        "sum_bf_orig_m": sum_bf_orig_m,
        "sum_bf_orig_raw_m": sum_bf_orig_raw_m,
        "sum_bf_orig_raw2_m": sum_bf_orig_raw2_m,
        "num_sum_Y_m": num_sum_Y_m,
        "all_sum_betas_m": all_sum_betas_m,
        "all_sum_betas2_m": all_sum_betas2_m,
        "all_num_sum_m": all_num_sum_m,
        "priors_sample_m": priors_sample_m,
        "priors_mean_m": priors_mean_m,
        "priors_percentage_max_sample_m": priors_percentage_max_sample_m,
        "priors_percentage_max_mean_m": priors_percentage_max_mean_m,
        "priors_adjustment_sample_m": priors_adjustment_sample_m,
        "priors_adjustment_mean_m": priors_adjustment_mean_m,
        "priors_for_Y_m": priors_for_Y_m,
        "priors_percentage_max_for_Y_m": priors_percentage_max_for_Y_m,
        "priors_adjustment_for_Y_m": priors_adjustment_for_Y_m,
        "sum_priors_missing_m": sum_priors_missing_m,
        "sum_Ds_missing_m": sum_Ds_missing_m,
        "priors_missing_sample_m": priors_missing_sample_m,
        "priors_missing_mean_m": priors_missing_mean_m,
        "num_sum_priors_missing_m": num_sum_priors_missing_m,
        "post_burn_reset_arrays": post_burn_reset_arrays,
        "post_burn_reset_missing_arrays": post_burn_reset_missing_arrays,
        "X_hstacked": X_hstacked,
        "stack_batch_size": stack_batch_size,
        "num_stack_batches": num_stack_batches,
    }


def _initialize_gibbs_epoch_control_state(epoch_state):
    # Mutable per-epoch control/diagnostic state that evolves each iteration.
    num_gene_sets = epoch_state["sum_betas_m"].shape[1]
    num_genes = epoch_state["sum_Ys_m"].shape[1]
    return {
        "in_burn_in": True,
        "burn_in_pass_streak": 0,
        "burn_in_rhat_history": [],
        "stop_pass_streak": 0,
        "R_beta_v": np.ones(num_gene_sets),
        "betas_sem2_v": np.zeros(num_gene_sets),
        "sem2_v": np.zeros(num_genes),
        "stop_due_to_stall": False,
        "stop_due_to_precision": False,
        "restart_due_to_stall": False,
        "burn_stall_best_beta_rhat_history": [],
        "burn_stall_snapshots": [],
        "burn_stall_beta_indices": None,
        "post_stall_best_beta_rhat_history": [],
        "post_stall_best_D_mcse_history": [],
        "post_stall_snapshots": [],
        "post_stall_beta_indices": None,
        "post_stall_gene_indices": None,
        "prev_Ys_m": None,
    }


def _initialize_gibbs_epoch_sums_state(epoch_state, epoch_aggregates):
    # Mutable per-epoch posterior sum arrays accumulated after burn-in.
    epoch_sums = {key: epoch_state[key] for key in _GIBBS_EPOCH_SUM_KEYS}
    for key in _GIBBS_EPOCH_MISSING_SUM_KEYS:
        epoch_sums[key] = epoch_state[key]
    epoch_sums["epoch_aggregates"] = epoch_aggregates
    return epoch_sums


def _initialize_gibbs_epoch_priors_state(epoch_state):
    # Mutable per-epoch priors/warm-start state updated each iteration.
    return {
        "prev_warm_start_betas_m": epoch_state["prev_warm_start_betas_m"],
        "prev_warm_start_postp_m": epoch_state["prev_warm_start_postp_m"],
        "priors_sample_m": epoch_state["priors_sample_m"],
        "priors_mean_m": epoch_state["priors_mean_m"],
        "priors_percentage_max_sample_m": epoch_state["priors_percentage_max_sample_m"],
        "priors_percentage_max_mean_m": epoch_state["priors_percentage_max_mean_m"],
        "priors_adjustment_sample_m": epoch_state["priors_adjustment_sample_m"],
        "priors_adjustment_mean_m": epoch_state["priors_adjustment_mean_m"],
        "priors_for_Y_m": epoch_state["priors_for_Y_m"],
        "priors_percentage_max_for_Y_m": epoch_state["priors_percentage_max_for_Y_m"],
        "priors_adjustment_for_Y_m": epoch_state["priors_adjustment_for_Y_m"],
        "priors_missing_sample_m": epoch_state["priors_missing_sample_m"],
        "priors_missing_mean_m": epoch_state["priors_missing_mean_m"],
    }


def _initialize_gibbs_epoch_runtime_state(epoch_state, num_p_increases):
    # Mutable per-epoch running totals and restart-related flags.
    epoch_runtime = {key: epoch_state[key] for key in _GIBBS_EPOCH_RUNTIME_SUM_KEYS}
    epoch_runtime["gibbs_good"] = True
    epoch_runtime["num_p_increases"] = num_p_increases
    return epoch_runtime


def _start_gibbs_epoch(
    state,
    num_chains,
    num_full_gene_sets,
    use_mean_betas,
    max_mb_X_h,
    log_fun,
    epoch_aggregates,
    num_p_increases,
):
    # Build all mutable epoch containers and stacked-X batching artifacts.
    epoch_state = _initialize_gibbs_epoch_state(
        state,
        num_chains,
        num_full_gene_sets,
        use_mean_betas,
        max_mb_X_h,
        log_fun,
    )
    return {
        "full_betas_m_shape": epoch_state["full_betas_m_shape"],
        "epoch_control": _initialize_gibbs_epoch_control_state(epoch_state),
        "epoch_sums": _initialize_gibbs_epoch_sums_state(epoch_state, epoch_aggregates),
        "epoch_priors": _initialize_gibbs_epoch_priors_state(epoch_state),
        "epoch_runtime": _initialize_gibbs_epoch_runtime_state(epoch_state, num_p_increases),
        "post_burn_reset_arrays": epoch_state["post_burn_reset_arrays"],
        "post_burn_reset_missing_arrays": epoch_state["post_burn_reset_missing_arrays"],
        "X_hstacked": epoch_state["X_hstacked"],
        "stack_batch_size": epoch_state["stack_batch_size"],
        "num_stack_batches": epoch_state["num_stack_batches"],
    }


def _maybe_write_gibbs_gene_stats_trace(
    gene_stats_trace_fh,
    iteration_num,
    trace_chain_offset,
    genes,
    priors_for_Y_m,
    Y_sample_m,
    log_bf_m,
    p_sample_m,
    priors_percentage_max_for_Y_m,
    priors_adjustment_for_Y_m,
):
    if gene_stats_trace_fh is None:
        return
    log("Writing gene stats trace", TRACE)
    _write_gene_stats_trace_rows(
        gene_stats_trace_fh,
        iteration_num,
        trace_chain_offset,
        genes,
        priors_for_Y_m,
        Y_sample_m,
        log_bf_m,
        p_sample_m,
        priors_percentage_max_for_Y_m,
        priors_adjustment_for_Y_m,
    )
    gene_stats_trace_fh.flush()


def _maybe_write_gibbs_gene_set_stats_trace(
    gene_set_stats_trace_fh,
    iteration_num,
    trace_chain_offset,
    gene_sets,
    scale_factors,
    full_beta_tildes_m,
    full_p_values_m,
    full_z_scores_m,
    full_ses_m,
    uncorrected_betas_mean_m,
    uncorrected_betas_sample_m,
    full_betas_mean_m,
    full_betas_sample_m,
    full_postp_mean_m,
    full_postp_sample_m,
    full_z_cur_beta_tildes_m,
    R_beta_v,
    betas_sem2_v,
    use_mean_betas,
):
    if gene_set_stats_trace_fh is None:
        return
    log("Writing gene set stats trace", TRACE)
    _write_gene_set_stats_trace_rows(
        gene_set_stats_trace_fh,
        iteration_num,
        trace_chain_offset,
        gene_sets,
        scale_factors,
        full_beta_tildes_m,
        full_p_values_m,
        full_z_scores_m,
        full_ses_m,
        uncorrected_betas_mean_m,
        uncorrected_betas_sample_m,
        full_betas_mean_m,
        full_betas_sample_m,
        full_postp_mean_m,
        full_postp_sample_m,
        full_z_cur_beta_tildes_m,
        R_beta_v,
        betas_sem2_v,
        use_mean_betas,
    )
    gene_set_stats_trace_fh.flush()


def _outlier_resistant_mean(sum_m, num_sum_m, num_mad, outlier_mask_m=None, record_param_fn=None):
    if outlier_mask_m is None:
        if record_param_fn is not None:
            record_param_fn("mad_threshold", num_mad)

        chain_means_m = sum_m / num_sum_m
        medians_v = np.median(chain_means_m, axis=0)
        mad_m = np.abs(chain_means_m - medians_v)
        mad_median_v = np.median(mad_m, axis=0)
        outlier_mask_m = chain_means_m > medians_v + num_mad * mad_median_v

    num_sum_v = np.sum(~outlier_mask_m, axis=0)
    num_sum_v[num_sum_v == 0] = 1

    copy_sum_m = copy.copy(sum_m)
    copy_sum_m[outlier_mask_m] = 0
    avg_v = np.sum(copy_sum_m / num_sum_m, axis=0) / num_sum_v
    return (outlier_mask_m, avg_v)


def _write_gene_stats_trace_rows(
    fh,
    iteration_num,
    trace_chain_offset,
    genes,
    priors_for_Y_m,
    Y_sample_m,
    log_bf_m,
    p_sample_m,
    priors_percentage_max_for_Y_m,
    priors_adjustment_for_Y_m,
):
    for chain_num in range(priors_for_Y_m.shape[0]):
        for i in range(len(genes)):
            fh.write(
                "%d\t%d\t%s\t%.4g\t%.4g\t%.4g\t%.4g\t%.4g\t%.4g\n"
                % (
                    iteration_num + 1,
                    trace_chain_offset + chain_num + 1,
                    genes[i],
                    priors_for_Y_m[chain_num, i],
                    Y_sample_m[chain_num, i],
                    log_bf_m[chain_num, i],
                    p_sample_m[chain_num, i],
                    priors_percentage_max_for_Y_m[chain_num, i],
                    priors_adjustment_for_Y_m[chain_num, i],
                )
            )


def _write_gene_set_stats_trace_rows(
    fh,
    iteration_num,
    trace_chain_offset,
    gene_sets,
    scale_factors,
    full_beta_tildes_m,
    full_p_values_m,
    full_z_scores_m,
    full_ses_m,
    uncorrected_betas_mean_m,
    uncorrected_betas_sample_m,
    full_betas_mean_m,
    full_betas_sample_m,
    full_postp_mean_m,
    full_postp_sample_m,
    full_z_cur_beta_tildes_m,
    R_beta_v,
    betas_sem2_v,
    use_mean_betas,
):
    for chain_num in range(full_beta_tildes_m.shape[0]):
        for i in range(len(gene_sets)):
            fh.write(
                "%d\t%d\t%s\t%.4g\t%.4g\t%.4g\t%.4g\t%.4g\t%.4g\t%.4g\t%.4g\t%.4g\t%.4g\n"
                % (
                    iteration_num + 1,
                    trace_chain_offset + chain_num + 1,
                    gene_sets[i],
                    full_beta_tildes_m[chain_num, i] / scale_factors[i],
                    full_p_values_m[chain_num, i],
                    full_z_scores_m[chain_num, i],
                    full_ses_m[chain_num, i] / scale_factors[i],
                    (uncorrected_betas_mean_m[chain_num, i] if use_mean_betas else uncorrected_betas_sample_m[chain_num, i]) / scale_factors[i],
                    (full_betas_mean_m[chain_num, i] if use_mean_betas else full_betas_sample_m[chain_num, i]) / scale_factors[i],
                    (full_postp_mean_m[chain_num, i] if use_mean_betas else full_postp_sample_m[chain_num, i]),
                    full_z_cur_beta_tildes_m[chain_num, i],
                    R_beta_v[i],
                    betas_sem2_v[i],
                )
            )


def _calc_priors_from_betas(X, betas_m, mean_shifts, scale_factors):
    # Compute per-chain log-prior odds (relative to background) from betas.
    return np.array(X.dot((betas_m / scale_factors).T) - np.sum(mean_shifts * betas_m / scale_factors, axis=1).T).T


def _combine_optional_gene_bf_terms(Y_exomes, Y_positive_controls, Y_case_counts):
    combined = 0
    for term in (Y_exomes, Y_positive_controls, Y_case_counts):
        if term is not None:
            combined = combined + term
    return combined


def _add_optional_gene_bf_terms(log_bf_m, log_bf_uncorrected_m, log_bf_raw_m, Y_exomes, Y_positive_controls, Y_case_counts):
    # Add external evidence terms back after HuGE distillation.
    for term in (Y_exomes, Y_positive_controls, Y_case_counts):
        if term is not None:
            log_bf_m += term
            log_bf_uncorrected_m += term
            log_bf_raw_m += term


def _get_gibbs_gene_set_mask(uncorrected_betas_mean_m, uncorrected_betas_sample_m, p_values_m, sparse_frac=0.01, sparse_max=0.001):
    # By design we sparsify using posterior means and apply the same mask to
    # both sample and mean matrices so the next Gibbs iteration warm-start is
    # consistent with the retained support.
    uncorrected_betas_m = uncorrected_betas_mean_m
    gene_set_mask_m = uncorrected_betas_m != 0

    if sparse_frac is not None:
        gene_set_mask_m = np.logical_and(
            gene_set_mask_m,
            (np.abs(uncorrected_betas_m).T >= sparse_frac * np.max(np.abs(uncorrected_betas_m), axis=1)).T,
        )
        gene_set_mask_m = np.logical_and(
            gene_set_mask_m,
            (np.abs(uncorrected_betas_m).T >= sparse_max).T,
        )
        uncorrected_betas_sample_m[~gene_set_mask_m] = 0
        uncorrected_betas_mean_m[~gene_set_mask_m] = 0

    if np.sum(gene_set_mask_m) == 0:
        gene_set_mask_m = p_values_m <= np.min(p_values_m)
    return gene_set_mask_m


def _compute_post_burn_beta_diagnostics(
    diag_sum_betas_m,
    diag_sum_betas2_m,
    diag_num_sum_beta_m,
    num_chains_effective_for_diag,
    active_beta_top_k,
    active_beta_min_abs,
    stop_mcse_quantile,
    beta_rel_mcse_denom_floor,
):
    active_beta_mask, beta_chain_means_m, beta_mean_v = _get_active_beta_mask(
        diag_sum_betas_m,
        diag_num_sum_beta_m,
        active_beta_top_k,
        active_beta_min_abs,
    )
    num_active_betas = int(np.sum(active_beta_mask))

    beta_mcse_v = np.sqrt(np.var(beta_chain_means_m, axis=0, ddof=1) / float(num_chains_effective_for_diag))
    num_post_burn_beta = int(np.min(diag_num_sum_beta_m))
    _, _, post_R_beta_v, _ = _calculate_rhat_from_sums(diag_sum_betas_m, diag_sum_betas2_m, num_post_burn_beta)

    if np.any(active_beta_mask):
        post_R_beta_active_v = post_R_beta_v[active_beta_mask]
        post_rhat_candidates = post_R_beta_active_v[np.logical_and(np.isfinite(post_R_beta_active_v), post_R_beta_active_v >= 1)]
        beta_rhat_q_post = _safe_quantile(post_rhat_candidates, stop_mcse_quantile, 1.0)

        beta_ratio_v = beta_mcse_v / np.maximum(np.abs(beta_mean_v), beta_rel_mcse_denom_floor)
        beta_ratio_q = _safe_quantile(beta_ratio_v[active_beta_mask], stop_mcse_quantile, np.inf)
    else:
        beta_rhat_q_post = 1.0
        beta_ratio_q = 0.0

    return {
        "active_beta_mask": active_beta_mask,
        "beta_mean_v": beta_mean_v,
        "num_active_betas": num_active_betas,
        "beta_mcse_v": beta_mcse_v,
        "num_post_burn_beta": num_post_burn_beta,
        "beta_rhat_q_post": beta_rhat_q_post,
        "beta_ratio_q": beta_ratio_q,
    }


def _compute_post_burn_gene_diagnostics(
    diag_sum_Ds_m,
    diag_num_sum_Y_m,
    num_chains_effective_for_diag,
    stop_top_gene_k,
    stop_min_gene_d,
    stop_mcse_quantile,
):
    D_chain_means_m = _means_from_sums(diag_sum_Ds_m, diag_num_sum_Y_m)
    D_mean_v = np.mean(D_chain_means_m, axis=0)
    D_mcse_v = np.sqrt(np.var(D_chain_means_m, axis=0, ddof=1) / float(num_chains_effective_for_diag))

    top_gene_k = min(stop_top_gene_k, len(D_mean_v))
    num_eligible_genes = len(D_mean_v)
    if stop_min_gene_d is not None:
        eligible_gene_indices = np.where(D_mean_v >= stop_min_gene_d)[0]
        num_eligible_genes = len(eligible_gene_indices)
        if num_eligible_genes > 0:
            if top_gene_k >= num_eligible_genes:
                top_gene_indices = eligible_gene_indices
            else:
                top_gene_indices = eligible_gene_indices[np.argpartition(-D_mean_v[eligible_gene_indices], top_gene_k - 1)[:top_gene_k]]
        else:
            if top_gene_k >= len(D_mean_v):
                top_gene_indices = np.arange(len(D_mean_v))
            else:
                top_gene_indices = np.argpartition(-D_mean_v, top_gene_k - 1)[:top_gene_k]
    else:
        if top_gene_k >= len(D_mean_v):
            top_gene_indices = np.arange(len(D_mean_v))
        else:
            top_gene_indices = np.argpartition(-D_mean_v, top_gene_k - 1)[:top_gene_k]

    num_monitored_genes = len(top_gene_indices)
    D_mcse_q = _safe_quantile(D_mcse_v[top_gene_indices], stop_mcse_quantile, np.inf)

    return {
        "D_mean_v": D_mean_v,
        "D_mcse_v": D_mcse_v,
        "top_gene_k": top_gene_k,
        "top_gene_indices": top_gene_indices,
        "num_monitored_genes": num_monitored_genes,
        "num_eligible_genes": num_eligible_genes,
        "D_mcse_q": D_mcse_q,
    }


def _summarize_gibbs_chain_aggregates(
    sum_Ys_m,
    sum_Y_raws_m,
    sum_log_pos_m,
    sum_log_po_raws_m,
    sum_log_po_raws2_m,
    sum_priors_m,
    sum_priors2_m,
    sum_Ds_m,
    sum_D_raws_m,
    sum_bf_orig_m,
    sum_bf_orig_raw_m,
    sum_bf_orig_raw2_m,
    sum_betas_m,
    sum_betas2_m,
    sum_betas_uncorrected_m,
    sum_betas_uncorrected2_m,
    sum_postp_m,
    sum_beta_tildes_m,
    sum_z_scores_m,
    num_sum_Y_m,
    num_sum_beta_m,
    num_chains_effective,
    num_mad,
    record_param_fn=None,
    sum_priors_missing_m=None,
    sum_Ds_missing_m=None,
    num_sum_priors_missing_m=None,
):
    Y_outlier_mask_m, _ = _outlier_resistant_mean(sum_Ys_m, num_sum_Y_m, num_mad, record_param_fn=record_param_fn)
    beta_outlier_mask_m, avg_betas_v = _outlier_resistant_mean(sum_betas_m, num_sum_beta_m, num_mad, record_param_fn=record_param_fn)

    _, _ = _outlier_resistant_mean(sum_Y_raws_m, num_sum_Y_m, num_mad, record_param_fn=record_param_fn)
    _, avg_log_pos_v = _outlier_resistant_mean(sum_log_pos_m, num_sum_Y_m, num_mad, Y_outlier_mask_m)
    _, avg_log_po_raws_v = _outlier_resistant_mean(sum_log_po_raws_m, num_sum_Y_m, num_mad, Y_outlier_mask_m)
    _, avg_Ds_v = _outlier_resistant_mean(sum_Ds_m, num_sum_Y_m, num_mad, Y_outlier_mask_m)
    _, avg_D_raws_v = _outlier_resistant_mean(sum_D_raws_m, num_sum_Y_m, num_mad, Y_outlier_mask_m)
    _, avg_priors_v = _outlier_resistant_mean(sum_priors_m, num_sum_Y_m, num_mad, Y_outlier_mask_m)
    _, avg_bf_orig_v = _outlier_resistant_mean(sum_bf_orig_m, num_sum_Y_m, num_mad, Y_outlier_mask_m)
    _, avg_bf_orig_raw_v = _outlier_resistant_mean(sum_bf_orig_raw_m, num_sum_Y_m, num_mad, Y_outlier_mask_m)

    avg_priors_missing_v = np.array([])
    avg_Ds_missing_v = np.array([])
    if sum_priors_missing_m is not None and sum_Ds_missing_m is not None and num_sum_priors_missing_m is not None:
        priors_missing_outlier_mask_m, avg_priors_missing_v = _outlier_resistant_mean(sum_priors_missing_m, num_sum_priors_missing_m, num_mad, record_param_fn=record_param_fn)
        _, avg_Ds_missing_v = _outlier_resistant_mean(sum_Ds_missing_m, num_sum_priors_missing_m, num_mad, priors_missing_outlier_mask_m)

    _, avg_betas_uncorrected_v = _outlier_resistant_mean(sum_betas_uncorrected_m, num_sum_beta_m, num_mad, beta_outlier_mask_m)
    _, avg_postp_v = _outlier_resistant_mean(sum_postp_m, num_sum_beta_m, num_mad, beta_outlier_mask_m)
    _, avg_beta_tildes_v = _outlier_resistant_mean(sum_beta_tildes_m, num_sum_beta_m, num_mad, beta_outlier_mask_m)
    _, avg_z_scores_v = _outlier_resistant_mean(sum_z_scores_m, num_sum_beta_m, num_mad, beta_outlier_mask_m)

    num_post_burn_in_Y = int(np.min(num_sum_Y_m))
    num_post_burn_in_beta = int(np.min(num_sum_beta_m))

    _, _, prior_r_hat_v, _ = _calculate_rhat_from_sums(sum_priors_m, sum_priors2_m, num_post_burn_in_Y)
    _, _, combined_r_hat_v, _ = _calculate_rhat_from_sums(sum_log_po_raws_m, sum_log_po_raws2_m, num_post_burn_in_Y)
    _, _, log_bf_r_hat_v, _ = _calculate_rhat_from_sums(sum_bf_orig_raw_m, sum_bf_orig_raw2_m, num_post_burn_in_Y)
    _, _, beta_r_hat_v, _ = _calculate_rhat_from_sums(sum_betas_m, sum_betas2_m, num_post_burn_in_beta)
    _, _, beta_uncorrected_r_hat_v, _ = _calculate_rhat_from_sums(sum_betas_uncorrected_m, sum_betas_uncorrected2_m, num_post_burn_in_beta)

    prior_chain_means_m = _means_from_sums(sum_priors_m, num_sum_Y_m)
    combined_chain_means_m = _means_from_sums(sum_log_po_raws_m, num_sum_Y_m)
    log_bf_chain_means_m = _means_from_sums(sum_bf_orig_raw_m, num_sum_Y_m)
    beta_chain_means_m = _means_from_sums(sum_betas_m, num_sum_beta_m)
    beta_uncorrected_chain_means_m = _means_from_sums(sum_betas_uncorrected_m, num_sum_beta_m)

    prior_mcse_v = np.sqrt(np.var(prior_chain_means_m, axis=0, ddof=1) / float(num_chains_effective))
    combined_mcse_v = np.sqrt(np.var(combined_chain_means_m, axis=0, ddof=1) / float(num_chains_effective))
    log_bf_mcse_v = np.sqrt(np.var(log_bf_chain_means_m, axis=0, ddof=1) / float(num_chains_effective))
    beta_mcse_v = np.sqrt(np.var(beta_chain_means_m, axis=0, ddof=1) / float(num_chains_effective))
    beta_uncorrected_mcse_v = np.sqrt(np.var(beta_uncorrected_chain_means_m, axis=0, ddof=1) / float(num_chains_effective))

    return {
        "avg_log_pos_v": avg_log_pos_v,
        "avg_log_po_raws_v": avg_log_po_raws_v,
        "avg_Ds_v": avg_Ds_v,
        "avg_D_raws_v": avg_D_raws_v,
        "avg_priors_v": avg_priors_v,
        "avg_bf_orig_v": avg_bf_orig_v,
        "avg_bf_orig_raw_v": avg_bf_orig_raw_v,
        "avg_priors_missing_v": avg_priors_missing_v,
        "avg_Ds_missing_v": avg_Ds_missing_v,
        "avg_betas_v": avg_betas_v,
        "avg_betas_uncorrected_v": avg_betas_uncorrected_v,
        "avg_postp_v": avg_postp_v,
        "avg_beta_tildes_v": avg_beta_tildes_v,
        "avg_z_scores_v": avg_z_scores_v,
        "prior_r_hat_v": prior_r_hat_v,
        "combined_r_hat_v": combined_r_hat_v,
        "log_bf_r_hat_v": log_bf_r_hat_v,
        "beta_r_hat_v": beta_r_hat_v,
        "beta_uncorrected_r_hat_v": beta_uncorrected_r_hat_v,
        "prior_mcse_v": prior_mcse_v,
        "combined_mcse_v": combined_mcse_v,
        "log_bf_mcse_v": log_bf_mcse_v,
        "beta_mcse_v": beta_mcse_v,
        "beta_uncorrected_mcse_v": beta_uncorrected_mcse_v,
    }


def _apply_gibbs_final_state(state, final_summary, adjust_priors):
    state.beta_tildes = final_summary["avg_beta_tildes_v"]
    state.z_scores = final_summary["avg_z_scores_v"]
    state.p_values = 2 * scipy.stats.norm.cdf(-np.abs(state.z_scores))
    state.ses = np.full(state.beta_tildes.shape, 100.0)
    state.ses[state.z_scores != 0] = np.abs(state.beta_tildes[state.z_scores != 0] / state.z_scores[state.z_scores != 0])

    state.betas = final_summary["avg_betas_v"]
    state.betas_uncorrected = final_summary["avg_betas_uncorrected_v"]
    state.betas_r_hat = final_summary["beta_r_hat_v"]
    state.betas_mcse = final_summary["beta_mcse_v"]
    state.betas_uncorrected_r_hat = final_summary["beta_uncorrected_r_hat_v"]
    state.betas_uncorrected_mcse = final_summary["beta_uncorrected_mcse_v"]
    state.non_inf_avg_cond_betas = None
    state.non_inf_avg_postps = final_summary["avg_postp_v"]

    state.priors = final_summary["avg_priors_v"]
    state.priors_r_hat = final_summary["prior_r_hat_v"]
    state.priors_mcse = final_summary["prior_mcse_v"]
    state.priors_missing = final_summary["avg_priors_missing_v"]
    state.combined_Ds_missing = final_summary["avg_Ds_missing_v"]

    state.Y_for_regression = final_summary["avg_bf_orig_v"]
    state.Y = final_summary["avg_bf_orig_raw_v"]
    state.Y_r_hat = final_summary["log_bf_r_hat_v"]
    state.Y_mcse = final_summary["log_bf_mcse_v"]

    state.combined_Ds_for_regression = final_summary["avg_Ds_v"]
    state.combined_Ds = final_summary["avg_D_raws_v"]

    state.combined_prior_Ys_for_regression = final_summary["avg_log_pos_v"] - state.background_log_bf
    state.combined_prior_Ys = final_summary["avg_log_po_raws_v"] - state.background_log_bf
    state.combined_prior_Ys_r_hat = final_summary["combined_r_hat_v"]
    state.combined_prior_Ys_mcse = final_summary["combined_mcse_v"]

    gene_N = state.get_gene_N()
    gene_N_missing = state.get_gene_N(get_missing=True)

    all_gene_N = gene_N
    if state.genes_missing is not None:
        assert(gene_N_missing is not None)
        all_gene_N = np.concatenate((all_gene_N, gene_N_missing))

    total_priors = np.concatenate((state.priors, state.priors_missing))
    priors_slope = np.cov(total_priors, all_gene_N)[0,1] / np.var(all_gene_N)
    priors_intercept = np.mean(total_priors - all_gene_N * priors_slope)

    if adjust_priors:
        log("Adjusting priors with slope %.4g" % priors_slope)
        state.priors_adj = state.priors - priors_slope * gene_N - priors_intercept
        if state.genes_missing is not None:
            state.priors_adj_missing = state.priors_missing - priors_slope * gene_N_missing

        combined_slope = np.cov(state.combined_prior_Ys, gene_N)[0,1] / np.var(gene_N)
        combined_intercept = np.mean(state.combined_prior_Ys - gene_N * combined_slope)

        log("Adjusting combined with slope %.4g" % combined_slope)
        state.combined_prior_Ys_adj = state.combined_prior_Ys - combined_slope * gene_N - combined_intercept


def _stack_gibbs_epoch_aggregates(epoch_aggregates, include_missing):
    stacked = {}
    for key in _GIBBS_EPOCH_SUM_KEYS:
        stacked[key] = np.vstack(epoch_aggregates[key])
    if include_missing:
        for key in _GIBBS_EPOCH_MISSING_SUM_KEYS:
            stacked[key] = np.vstack(epoch_aggregates[key])
    return stacked


def _build_gibbs_epoch_finalize_context(
    state,
    run_state,
    epoch_phase_config,
    epoch_control,
    epoch_runtime,
    iteration_num,
):
    return {
        "include_missing": (state.genes_missing is not None),
        "gibbs_good": epoch_runtime["gibbs_good"],
        "iterations_run_this_epoch": (iteration_num + 1),
        "remaining_total_iter": run_state.remaining_total_iter,
        "num_completed_epochs": run_state.num_completed_epochs,
        "target_num_epochs": epoch_phase_config.target_num_epochs,
        "num_attempts": run_state.num_attempts,
        "max_num_attempt_restarts": run_state.max_num_attempt_restarts,
        "stop_due_to_stall": epoch_control["stop_due_to_stall"],
        "stop_due_to_precision": epoch_control["stop_due_to_precision"],
        "num_mad": epoch_phase_config.num_mad,
        "adjust_priors": epoch_phase_config.adjust_priors,
    }


def _finalize_gibbs_epoch_attempt(
    state,
    epoch_aggregates,
    epoch_sums,
    finalize_context,
):
    include_missing = finalize_context["include_missing"]
    gibbs_good = finalize_context["gibbs_good"]
    iterations_run_this_epoch = finalize_context["iterations_run_this_epoch"]
    remaining_total_iter = finalize_context["remaining_total_iter"]
    num_completed_epochs = finalize_context["num_completed_epochs"]
    target_num_epochs = finalize_context["target_num_epochs"]
    num_attempts = finalize_context["num_attempts"]
    max_num_attempt_restarts = finalize_context["max_num_attempt_restarts"]
    stop_due_to_stall = finalize_context["stop_due_to_stall"]
    stop_due_to_precision = finalize_context["stop_due_to_precision"]
    num_mad = finalize_context["num_mad"]
    adjust_priors = finalize_context["adjust_priors"]

    remaining_total_iter -= iterations_run_this_epoch
    if remaining_total_iter < 0:
        remaining_total_iter = 0

    if not gibbs_good:
        return {
            "remaining_total_iter": remaining_total_iter,
            "num_completed_epochs": num_completed_epochs,
            "should_continue": True,
        }

    assert(np.all(epoch_sums["num_sum_Y_m"] > 0))
    assert(np.all(epoch_sums["num_sum_beta_m"] > 0))

    for key in _GIBBS_EPOCH_SUM_KEYS:
        epoch_aggregates[key].append(copy.copy(epoch_sums[key]))
    if include_missing:
        for key in _GIBBS_EPOCH_MISSING_SUM_KEYS:
            epoch_aggregates[key].append(copy.copy(epoch_sums[key]))

    num_completed_epochs += 1
    log(
        "Completed Gibbs epoch %d/%d (iter=%d, remaining_total_iter=%d)"
        % (num_completed_epochs, target_num_epochs, iterations_run_this_epoch, remaining_total_iter),
        INFO,
    )

    should_continue = _should_continue_gibbs_epoch_attempts(
        remaining_total_iter=remaining_total_iter,
        num_completed_epochs=num_completed_epochs,
        target_num_epochs=target_num_epochs,
        num_attempts=num_attempts,
        max_num_attempt_restarts=max_num_attempt_restarts,
        stop_due_to_stall=stop_due_to_stall,
        stop_due_to_precision=stop_due_to_precision,
    )
    if should_continue:
        return {
            "remaining_total_iter": remaining_total_iter,
            "num_completed_epochs": num_completed_epochs,
            "should_continue": True,
        }

    stacked = _stack_gibbs_epoch_aggregates(
        epoch_aggregates=epoch_aggregates,
        include_missing=include_missing,
    )
    sum_betas_m = stacked["sum_betas_m"]
    sum_betas2_m = stacked["sum_betas2_m"]
    sum_betas_uncorrected_m = stacked["sum_betas_uncorrected_m"]
    sum_betas_uncorrected2_m = stacked["sum_betas_uncorrected2_m"]
    sum_postp_m = stacked["sum_postp_m"]
    sum_beta_tildes_m = stacked["sum_beta_tildes_m"]
    sum_z_scores_m = stacked["sum_z_scores_m"]
    num_sum_beta_m = stacked["num_sum_beta_m"]
    sum_Ys_m = stacked["sum_Ys_m"]
    sum_Y_raws_m = stacked["sum_Y_raws_m"]
    sum_log_pos_m = stacked["sum_log_pos_m"]
    sum_log_po_raws_m = stacked["sum_log_po_raws_m"]
    sum_log_po_raws2_m = stacked["sum_log_po_raws2_m"]
    sum_priors_m = stacked["sum_priors_m"]
    sum_priors2_m = stacked["sum_priors2_m"]
    sum_Ds_m = stacked["sum_Ds_m"]
    sum_D_raws_m = stacked["sum_D_raws_m"]
    sum_bf_orig_m = stacked["sum_bf_orig_m"]
    sum_bf_orig_raw_m = stacked["sum_bf_orig_raw_m"]
    sum_bf_orig_raw2_m = stacked["sum_bf_orig_raw2_m"]
    num_sum_Y_m = stacked["num_sum_Y_m"]
    if include_missing:
        sum_priors_missing_m = stacked["sum_priors_missing_m"]
        sum_Ds_missing_m = stacked["sum_Ds_missing_m"]
        num_sum_priors_missing_m = stacked["num_sum_priors_missing_m"]
    else:
        sum_priors_missing_m = None
        sum_Ds_missing_m = None
        num_sum_priors_missing_m = None

    num_chains_effective = sum_betas_m.shape[0]
    final_summary = _summarize_gibbs_chain_aggregates(
        sum_Ys_m,
        sum_Y_raws_m,
        sum_log_pos_m,
        sum_log_po_raws_m,
        sum_log_po_raws2_m,
        sum_priors_m,
        sum_priors2_m,
        sum_Ds_m,
        sum_D_raws_m,
        sum_bf_orig_m,
        sum_bf_orig_raw_m,
        sum_bf_orig_raw2_m,
        sum_betas_m,
        sum_betas2_m,
        sum_betas_uncorrected_m,
        sum_betas_uncorrected2_m,
        sum_postp_m,
        sum_beta_tildes_m,
        sum_z_scores_m,
        num_sum_Y_m,
        num_sum_beta_m,
        num_chains_effective,
        num_mad,
        record_param_fn=state._record_param,
        sum_priors_missing_m=sum_priors_missing_m if include_missing else None,
        sum_Ds_missing_m=sum_Ds_missing_m if include_missing else None,
        num_sum_priors_missing_m=num_sum_priors_missing_m if include_missing else None,
    )
    _apply_gibbs_final_state(state, final_summary, adjust_priors)

    return {
        "remaining_total_iter": remaining_total_iter,
        "num_completed_epochs": num_completed_epochs,
        "should_continue": False,
    }


def _compute_gibbs_iteration_y_terms(
    priors_for_Y_m,
    log_bf_m,
    log_bf_raw_m,
    cur_background_log_bf_v,
    max_log=15.0,
    min_D=1e-5,
    max_D=1 - 1e-5,
):
    Y_sample_m = priors_for_Y_m + log_bf_m
    Y_raw_sample_m = priors_for_Y_m + log_bf_raw_m
    y_var = np.var(Y_sample_m, axis=1)

    cur_log_bf_m = Y_sample_m.T + cur_background_log_bf_v
    cur_log_bf_m[cur_log_bf_m > max_log] = max_log
    bf_sample_m = np.exp(cur_log_bf_m).T

    cur_log_bf_raw_m = Y_raw_sample_m.T + cur_background_log_bf_v
    cur_log_bf_raw_m[cur_log_bf_raw_m > max_log] = max_log
    bf_raw_sample_m = np.exp(cur_log_bf_raw_m).T

    D_sample_m = bf_sample_m / (1 + bf_sample_m)
    D_sample_m[D_sample_m > max_D] = max_D
    D_sample_m[D_sample_m < min_D] = min_D
    log_po_sample_m = np.log(D_sample_m / (1 - D_sample_m))

    D_raw_sample_m = bf_raw_sample_m / (1 + bf_raw_sample_m)
    D_raw_sample_m[D_raw_sample_m > max_D] = max_D
    D_raw_sample_m[D_raw_sample_m < min_D] = min_D
    log_po_raw_sample_m = np.log(D_raw_sample_m / (1 - D_raw_sample_m))

    return {
        "Y_sample_m": Y_sample_m,
        "Y_raw_sample_m": Y_raw_sample_m,
        "y_var": y_var,
        "D_sample_m": D_sample_m,
        "log_po_sample_m": log_po_sample_m,
        "D_raw_sample_m": D_raw_sample_m,
        "log_po_raw_sample_m": log_po_raw_sample_m,
    }


def _build_gibbs_post_burn_accumulation_payload(
    iter_state,
    full_betas_mean_m,
    full_postp_sample_m,
    priors_for_Y_m,
    priors_missing_mean_m,
    log_bf_m,
    log_bf_raw_m,
):
    return {
        "Y_sample_m": iter_state["Y_sample_m"],
        "Y_raw_sample_m": iter_state["Y_raw_sample_m"],
        "log_po_sample_m": iter_state["log_po_sample_m"],
        "log_po_raw_sample_m": iter_state["log_po_raw_sample_m"],
        "priors_for_Y_m": priors_for_Y_m,
        "D_sample_m": iter_state["D_sample_m"],
        "D_raw_sample_m": iter_state["D_raw_sample_m"],
        "log_bf_m": log_bf_m,
        "log_bf_raw_m": log_bf_raw_m,
        "full_betas_mean_m": full_betas_mean_m,
        "uncorrected_betas_mean_m": iter_state["uncorrected_betas_mean_m"],
        "full_postp_sample_m": full_postp_sample_m,
        "full_beta_tildes_m": iter_state["full_beta_tildes_m"],
        "full_z_scores_m": iter_state["full_z_scores_m"],
        "priors_missing_mean_m": priors_missing_mean_m,
    }


def _accumulate_gibbs_post_burn_iteration(
    state,
    accumulation_payload,
    epoch_sums,
):
    Y_sample_m = accumulation_payload["Y_sample_m"]
    Y_raw_sample_m = accumulation_payload["Y_raw_sample_m"]
    log_po_sample_m = accumulation_payload["log_po_sample_m"]
    log_po_raw_sample_m = accumulation_payload["log_po_raw_sample_m"]
    priors_for_Y_m = accumulation_payload["priors_for_Y_m"]
    D_sample_m = accumulation_payload["D_sample_m"]
    D_raw_sample_m = accumulation_payload["D_raw_sample_m"]
    log_bf_m = accumulation_payload["log_bf_m"]
    log_bf_raw_m = accumulation_payload["log_bf_raw_m"]
    full_betas_mean_m = accumulation_payload["full_betas_mean_m"]
    uncorrected_betas_mean_m = accumulation_payload["uncorrected_betas_mean_m"]
    full_postp_sample_m = accumulation_payload["full_postp_sample_m"]
    full_beta_tildes_m = accumulation_payload["full_beta_tildes_m"]
    full_z_scores_m = accumulation_payload["full_z_scores_m"]
    priors_missing_mean_m = accumulation_payload["priors_missing_mean_m"]

    # Collect one post-burn Gibbs draw into running sums used for MCSE/R-hat and
    # final epoch aggregation.
    epoch_sums["sum_Ys_m"] += Y_sample_m
    epoch_sums["sum_Y_raws_m"] += Y_raw_sample_m
    epoch_sums["sum_log_pos_m"] += log_po_sample_m
    epoch_sums["sum_log_po_raws_m"] += log_po_raw_sample_m
    epoch_sums["sum_log_po_raws2_m"] += np.power(log_po_raw_sample_m, 2)
    epoch_sums["sum_priors_m"] += priors_for_Y_m
    epoch_sums["sum_priors2_m"] += np.power(priors_for_Y_m, 2)
    epoch_sums["sum_Ds_m"] += D_sample_m
    epoch_sums["sum_D_raws_m"] += D_raw_sample_m
    epoch_sums["sum_bf_orig_m"] += log_bf_m
    epoch_sums["sum_bf_orig_raw_m"] += log_bf_raw_m
    epoch_sums["sum_bf_orig_raw2_m"] += np.power(log_bf_raw_m, 2)
    epoch_sums["num_sum_Y_m"] += 1

    epoch_sums["sum_betas_m"] += full_betas_mean_m
    epoch_sums["sum_betas2_m"] += np.power(full_betas_mean_m, 2)
    epoch_sums["sum_betas_uncorrected_m"] += uncorrected_betas_mean_m
    epoch_sums["sum_betas_uncorrected2_m"] += np.power(uncorrected_betas_mean_m, 2)
    epoch_sums["sum_postp_m"] += full_postp_sample_m
    epoch_sums["sum_beta_tildes_m"] += full_beta_tildes_m
    epoch_sums["sum_z_scores_m"] += full_z_scores_m
    epoch_sums["num_sum_beta_m"] += 1

    if state.genes_missing is not None:
        epoch_sums["sum_priors_missing_m"] += priors_missing_mean_m
        max_log = 15
        cur_log_priors_missing_m = priors_missing_mean_m + state.background_log_bf
        cur_log_priors_missing_m[cur_log_priors_missing_m > max_log] = max_log
        epoch_sums["sum_Ds_missing_m"] += np.exp(cur_log_priors_missing_m) / (1 + np.exp(cur_log_priors_missing_m))
        epoch_sums["num_sum_priors_missing_m"] += 1


def _sample_gibbs_iteration_y_state(
    state,
    priors_for_Y_m,
    log_bf_m,
    log_bf_raw_m,
    cur_background_log_bf_v,
    y_var_orig,
):
    log("Sampling new Ys")
    log("Setting logistic Ys", TRACE)
    y_terms = _compute_gibbs_iteration_y_terms(
        priors_for_Y_m,
        log_bf_m,
        log_bf_raw_m,
        cur_background_log_bf_v,
    )

    if state.y_corr_sparse is not None:
        log("Adjusting correlation matrix")
    y_corr_sparse = _compute_gibbs_y_corr_sparse(state.y_corr_sparse, priors_for_Y_m, y_var_orig)

    y_terms["y_corr_sparse"] = y_corr_sparse
    return y_terms


def _prepare_gibbs_iteration_inputs(
    state,
    iteration_num,
    log_bf_m,
    log_bf_raw_m,
    iteration_input_config,
):
    epoch_total_iter_offset = iteration_input_config["epoch_total_iter_offset"]
    trace_chain_offset = iteration_input_config["trace_chain_offset"]
    cur_background_log_bf_v = iteration_input_config["cur_background_log_bf_v"]
    y_var_orig = iteration_input_config["y_var_orig"]
    epoch_priors = iteration_input_config["epoch_priors"]
    gene_stats_trace_fh = iteration_input_config["gene_stats_trace_fh"]

    priors_for_Y_m = epoch_priors["priors_for_Y_m"]
    priors_percentage_max_for_Y_m = epoch_priors["priors_percentage_max_for_Y_m"]
    priors_adjustment_for_Y_m = epoch_priors["priors_adjustment_for_Y_m"]

    epoch_iter_num = iteration_num + 1
    total_iter_num = epoch_total_iter_offset + epoch_iter_num

    log("Beginning Gibbs iteration %d (global %d)" % (epoch_iter_num, total_iter_num))
    state._record_param("num_gibbs_iter", iteration_num, overwrite=True)
    state._record_param("num_gibbs_iter_total", total_iter_num, overwrite=True)

    y_terms = _sample_gibbs_iteration_y_state(
        state=state,
        priors_for_Y_m=priors_for_Y_m,
        log_bf_m=log_bf_m,
        log_bf_raw_m=log_bf_raw_m,
        cur_background_log_bf_v=cur_background_log_bf_v,
        y_var_orig=y_var_orig,
    )
    (
        Y_sample_m,
        _Y_raw_sample_m,
        y_var,
        D_sample_m,
        _log_po_sample_m,
        _D_raw_sample_m,
        _log_po_raw_sample_m,
        y_corr_sparse,
    ) = (
        y_terms["Y_sample_m"],
        y_terms["Y_raw_sample_m"],
        y_terms["y_var"],
        y_terms["D_sample_m"],
        y_terms["log_po_sample_m"],
        y_terms["D_raw_sample_m"],
        y_terms["log_po_raw_sample_m"],
        y_terms["y_corr_sparse"],
    )

    logistic_config = {
        "full_betas_m_shape": iteration_input_config["full_betas_m_shape"],
        "num_stack_batches": iteration_input_config["num_stack_batches"],
        "stack_batch_size": iteration_input_config["stack_batch_size"],
        "X_hstacked": iteration_input_config["X_hstacked"],
        "inner_beta_kwargs": iteration_input_config["inner_beta_kwargs"],
        "num_chains": len(Y_sample_m),
        "gauss_seidel": iteration_input_config["gauss_seidel"],
        "initial_linear_filter": iteration_input_config["initial_linear_filter"],
        "sparse_frac_gibbs": iteration_input_config["sparse_frac_gibbs"],
        "sparse_max_gibbs": iteration_input_config["sparse_max_gibbs"],
        "correct_betas_mean": iteration_input_config["correct_betas_mean"],
        "correct_betas_var": iteration_input_config["correct_betas_var"],
    }
    logistic_setup = _compute_gibbs_logistic_beta_tildes(
        state=state,
        Y_sample_m=Y_sample_m,
        y_var=y_var,
        D_sample_m=D_sample_m,
        y_corr_sparse=y_corr_sparse,
        logistic_config=logistic_config,
    )

    _maybe_write_gibbs_gene_stats_trace(
        gene_stats_trace_fh,
        iteration_num,
        trace_chain_offset,
        state.genes,
        priors_for_Y_m,
        Y_sample_m,
        log_bf_m,
        logistic_setup["p_sample_m"],
        priors_percentage_max_for_Y_m,
        priors_adjustment_for_Y_m,
    )

    return {
        "epoch_iter_num": epoch_iter_num,
        "total_iter_num": total_iter_num,
        "Y_sample_m": y_terms["Y_sample_m"],
        "Y_raw_sample_m": y_terms["Y_raw_sample_m"],
        "y_var": y_terms["y_var"],
        "D_sample_m": y_terms["D_sample_m"],
        "log_po_sample_m": y_terms["log_po_sample_m"],
        "D_raw_sample_m": y_terms["D_raw_sample_m"],
        "log_po_raw_sample_m": y_terms["log_po_raw_sample_m"],
        "full_scale_factors_m": logistic_setup["full_scale_factors_m"],
        "full_mean_shifts_m": logistic_setup["full_mean_shifts_m"],
        "full_is_dense_gene_set_m": logistic_setup["full_is_dense_gene_set_m"],
        "full_ps_m": logistic_setup["full_ps_m"],
        "full_sigma2s_m": logistic_setup["full_sigma2s_m"],
        "p_sample_m": logistic_setup["p_sample_m"],
        "pre_gene_set_filter_mask": logistic_setup["pre_gene_set_filter_mask"],
        "full_z_cur_beta_tildes_m": logistic_setup["full_z_cur_beta_tildes_m"],
        "full_beta_tildes_m": logistic_setup["full_beta_tildes_m"],
        "full_ses_m": logistic_setup["full_ses_m"],
        "full_z_scores_m": logistic_setup["full_z_scores_m"],
        "full_p_values_m": logistic_setup["full_p_values_m"],
    }


def _prepare_gibbs_iteration_state(
    state,
    iteration_num,
    iteration_state_config,
    log_bf_m,
    log_bf_raw_m,
):
    inner_beta_kwargs = iteration_state_config["inner_beta_kwargs"]
    prefilter_config = iteration_state_config["prefilter_config"]

    # Prepare all iteration-local sampling and masking state before corrected betas.
    iter_setup = _prepare_gibbs_iteration_inputs(
        state=state,
        iteration_num=iteration_num,
        log_bf_m=log_bf_m,
        log_bf_raw_m=log_bf_raw_m,
        iteration_input_config=iteration_state_config,
    )
    iter_state = dict(iter_setup)

    return _augment_gibbs_iteration_state_with_uncorrected_and_mask(
        state=state,
        iter_state=iter_state,
        prefilter_config=prefilter_config,
        inner_beta_kwargs=inner_beta_kwargs,
    )


def _augment_gibbs_iteration_state_with_uncorrected_and_mask(
    state,
    iter_state,
    prefilter_config,
    inner_beta_kwargs,
):
    uncorrected_setup = _compute_gibbs_uncorrected_betas_and_defaults(
        state,
        full_beta_tildes_m=iter_state["full_beta_tildes_m"],
        full_ses_m=iter_state["full_ses_m"],
        full_scale_factors_m=iter_state["full_scale_factors_m"],
        full_mean_shifts_m=iter_state["full_mean_shifts_m"],
        full_is_dense_gene_set_m=iter_state["full_is_dense_gene_set_m"],
        full_ps_m=iter_state["full_ps_m"],
        full_sigma2s_m=iter_state["full_sigma2s_m"],
        **inner_beta_kwargs,
    )
    iter_state.update(uncorrected_setup)

    (
        gene_set_mask_m,
        iter_state["default_betas_sample_m"],
        iter_state["default_postp_sample_m"],
        iter_state["default_betas_mean_m"],
        iter_state["default_postp_mean_m"],
    ) = _prepare_gibbs_gene_set_mask_with_prefilter(
        state,
        iter_state=iter_state,
        prefilter_config=prefilter_config,
        inner_beta_kwargs=inner_beta_kwargs,
        default_betas_sample_m=iter_state["default_betas_sample_m"],
        default_postp_sample_m=iter_state["default_postp_sample_m"],
        default_betas_mean_m=iter_state["default_betas_mean_m"],
        default_postp_mean_m=iter_state["default_postp_mean_m"],
    )

    return (iter_state, gene_set_mask_m)


def _build_non_inf_beta_sampler_kwargs(inner_beta_kwargs):
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


def _snapshot_pre_gibbs_state(state):
    # Preserve pre-Gibbs values so downstream reporting can compare original vs
    # Gibbs-adjusted statistics.
    state.beta_tildes_orig = copy.copy(state.beta_tildes)
    state.p_values_orig = copy.copy(state.p_values)
    state.ses_orig = copy.copy(state.ses)
    state.z_scores_orig = copy.copy(state.z_scores)
    state.beta_tildes_missing_orig = copy.copy(state.beta_tildes_missing)
    state.p_values_missing_orig = copy.copy(state.p_values_missing)
    state.ses_missing_orig = copy.copy(state.ses_missing)
    state.z_scores_missing_orig = copy.copy(state.z_scores_missing)

    state.betas_orig = copy.copy(state.betas)
    state.betas_uncorrected_orig = copy.copy(state.betas_uncorrected)
    state.non_inf_avg_cond_betas_orig = copy.copy(state.non_inf_avg_cond_betas)
    state.non_inf_avg_postps_orig = copy.copy(state.non_inf_avg_postps)
    state.betas_missing_orig = copy.copy(state.betas_missing)
    state.betas_uncorrected_missing_orig = copy.copy(state.betas_uncorrected_missing)
    state.non_inf_avg_cond_betas_missing_orig = copy.copy(state.non_inf_avg_cond_betas_missing)
    state.non_inf_avg_postps_missing_orig = copy.copy(state.non_inf_avg_postps_missing)

    state.Y_orig = copy.copy(state.Y)
    state.Y_for_regression_orig = copy.copy(state.Y_for_regression)
    state.priors_orig = copy.copy(state.priors)
    state.priors_adj_orig = copy.copy(state.priors_adj)
    state.priors_missing_orig = copy.copy(state.priors_missing)
    state.priors_adj_missing_orig = copy.copy(state.priors_adj_missing)


def _maybe_log_gibbs_conditional_variance(state, top_gene_prior):
    # Legacy logging of implied conditional Y variance from prior settings.
    priors_guess = np.array(
        state.X_orig.dot(state.betas / state.scale_factors)
        - np.sum(state.mean_shifts * state.betas / state.scale_factors)
    )
    y_resid = np.var(state.Y_for_regression_orig - priors_guess)
    y_cond_var = y_resid
    if top_gene_prior is not None:
        if top_gene_prior <= 0 or top_gene_prior >= 1:
            bail("--top-gene-prior needs to be in (0,1)")
        y_total_var = state.convert_prior_to_var(top_gene_prior, len(state.genes))
        y_cond_var = y_total_var - state.get_sigma2(convert_sigma_to_external_units=True) * np.mean(state.get_gene_N())
        if y_cond_var < 0:
            y_cond_var = 0.1
        log("Setting Y cond var=%.4g (total var = %.4g) given top gene prior of %.4g" % (y_cond_var, y_total_var, top_gene_prior))


# ========================= Outer Gibbs Input Snapshot + Prep =========================
def _prepare_gibbs_run_inputs(state, num_chains, top_gene_prior):
    _snapshot_pre_gibbs_state(state)

    # We always update correlation relative to the original Y variance.
    y_var_orig = np.var(state.Y_for_regression)

    _maybe_log_gibbs_conditional_variance(state, top_gene_prior)
    bf_orig = np.exp(state.Y_for_regression_orig)
    bf_orig_raw = np.exp(state.Y_orig)

    bf_orig_m = np.tile(bf_orig, num_chains).reshape(num_chains, len(bf_orig))
    log_bf_m = np.log(bf_orig_m)
    log_bf_uncorrected_m = np.log(bf_orig_m)

    bf_orig_raw_m = np.tile(bf_orig_raw, num_chains).reshape(num_chains, len(bf_orig_raw))
    log_bf_raw_m = np.log(bf_orig_raw_m)
    compute_Y_raw = np.any(~np.isclose(log_bf_m, log_bf_raw_m))
    cur_background_log_bf_v = np.tile(state.background_log_bf, num_chains)

    if state.y_corr_cholesky is not None:
        bail("GLS not implemented yet for Gibbs sampling!")

    num_full_gene_sets = len(state.gene_sets)
    if state.gene_sets_missing is not None:
        num_full_gene_sets += len(state.gene_sets_missing)

    return {
        "y_var_orig": y_var_orig,
        "log_bf_m": log_bf_m,
        "log_bf_uncorrected_m": log_bf_uncorrected_m,
        "log_bf_raw_m": log_bf_raw_m,
        "compute_Y_raw": compute_Y_raw,
        "cur_background_log_bf_v": cur_background_log_bf_v,
        "num_full_gene_sets": num_full_gene_sets,
    }


def _reset_gibbs_diagnostics(state):
    # Reset diagnostics that are specific to this Gibbs run.
    state.betas_r_hat = None
    state.betas_mcse = None
    state.betas_uncorrected_r_hat = None
    state.betas_uncorrected_mcse = None
    state.priors_r_hat = None
    state.priors_mcse = None
    state.combined_prior_Ys_r_hat = None
    state.combined_prior_Ys_mcse = None
    state.Y_r_hat = None
    state.Y_mcse = None


# ========================= Outer Gibbs Runtime Config Builders =========================
def _build_gibbs_epoch_runtime_configs(config_inputs):
    # Group per-epoch and per-iteration static knobs so run_gibbs can focus on
    # control flow.
    epoch_phase_config = GibbsEpochPhaseConfig(
        total_num_iter=config_inputs["total_num_iter"],
        num_chains=config_inputs["num_chains"],
        num_full_gene_sets=config_inputs["num_full_gene_sets"],
        use_mean_betas=config_inputs["use_mean_betas"],
        max_mb_X_h=config_inputs["max_mb_X_h"],
        target_num_epochs=config_inputs["target_num_epochs"],
        num_mad=config_inputs["num_mad"],
        adjust_priors=config_inputs["adjust_priors"],
        epoch_max_num_iter_config=config_inputs["epoch_max_num_iter_config"],
        min_num_burn_in=config_inputs["min_num_burn_in"],
        max_num_burn_in=config_inputs["max_num_burn_in"],
        min_num_post_burn_in=config_inputs["min_num_post_burn_in"],
        max_num_post_burn_in=config_inputs["max_num_post_burn_in"],
        increase_hyper_if_betas_below=config_inputs["increase_hyper_if_betas_below"],
        experimental_hyper_mutation=config_inputs["experimental_hyper_mutation"],
    )
    inner_beta_kwargs = {
        "passed_in_max_num_burn_in": config_inputs["passed_in_max_num_burn_in"],
        "max_num_iter_betas": config_inputs["max_num_iter_betas"],
        "min_num_iter_betas": config_inputs["min_num_iter_betas"],
        "num_chains_betas": config_inputs["num_chains_betas"],
        "r_threshold_burn_in_betas": config_inputs["r_threshold_burn_in_betas"],
        "use_max_r_for_convergence_betas": config_inputs["use_max_r_for_convergence_betas"],
        "max_frac_sem_betas": config_inputs["max_frac_sem_betas"],
        "max_allowed_batch_correlation": config_inputs["max_allowed_batch_correlation"],
        "gauss_seidel_betas": config_inputs["gauss_seidel_betas"],
        "sparse_solution": config_inputs["sparse_solution"],
        "sparse_frac_betas": config_inputs["sparse_frac_betas"],
    }
    iteration_update_config = GibbsIterationUpdateConfig(
        use_mean_betas=config_inputs["use_mean_betas"],
        warm_start=config_inputs["warm_start"],
        debug_zero_sparse=config_inputs["debug_zero_sparse"],
        num_chains=config_inputs["num_chains"],
        num_batches_parallel=config_inputs["num_batches_parallel"],
        betas_trace_out=config_inputs["betas_trace_out"],
        update_huge_scores=config_inputs["update_huge_scores"],
        compute_Y_raw=config_inputs["compute_Y_raw"],
        adjust_priors=config_inputs["adjust_priors"],
    )
    prefilter_config = {
        "sparse_frac_gibbs": config_inputs["sparse_frac_gibbs"],
        "sparse_max_gibbs": config_inputs["sparse_max_gibbs"],
        "pre_filter_batch_size": config_inputs["pre_filter_batch_size"],
        "pre_filter_small_batch_size": config_inputs["pre_filter_small_batch_size"],
    }
    burn_in_config = {
        "active_beta_top_k": config_inputs["active_beta_top_k"],
        "active_beta_min_abs": config_inputs["active_beta_min_abs"],
        "burn_in_rhat_quantile": config_inputs["burn_in_rhat_quantile"],
        "r_threshold_burn_in": config_inputs["r_threshold_burn_in"],
        "stall_window": config_inputs["stall_window"],
        "stall_min_burn_in": config_inputs["stall_min_burn_in"],
        "stall_delta_rhat": config_inputs["stall_delta_rhat"],
        "stall_recent_window": config_inputs["stall_recent_window"],
        "stall_recent_eps": config_inputs["stall_recent_eps"],
        "burn_in_stall_window": config_inputs["burn_in_stall_window"],
        "burn_in_stall_delta": config_inputs["burn_in_stall_delta"],
        "gauss_seidel": config_inputs["gauss_seidel"],
        "eps": config_inputs["eps"],
        "diag_every": config_inputs["diag_every"],
        "num_full_gene_sets": config_inputs["num_full_gene_sets"],
        "burn_in_patience": config_inputs["burn_in_patience"],
        "stop_patience": config_inputs["stop_patience"],
    }
    post_burn_diag_config = {
        "num_chains": config_inputs["num_chains"],
        "active_beta_top_k": config_inputs["active_beta_top_k"],
        "active_beta_min_abs": config_inputs["active_beta_min_abs"],
        "stop_mcse_quantile": config_inputs["stop_mcse_quantile"],
        "beta_rel_mcse_denom_floor": config_inputs["beta_rel_mcse_denom_floor"],
        "stop_top_gene_k": config_inputs["stop_top_gene_k"],
        "stop_min_gene_d": config_inputs["stop_min_gene_d"],
        "max_rel_mcse_beta": config_inputs["max_rel_mcse_beta"],
        "max_abs_mcse_d": config_inputs["max_abs_mcse_d"],
        "stop_patience": config_inputs["stop_patience"],
        "stall_window": config_inputs["stall_window"],
        "stall_min_post_burn_in": config_inputs["stall_min_post_burn_in"],
        "stall_delta_rhat": config_inputs["stall_delta_rhat"],
        "stall_delta_mcse": config_inputs["stall_delta_mcse"],
        "stall_recent_window": config_inputs["stall_recent_window"],
        "stall_recent_eps": config_inputs["stall_recent_eps"],
        "num_full_gene_sets": config_inputs["num_full_gene_sets"],
        "burn_in_patience": config_inputs["burn_in_patience"],
    }
    iteration_progress_config = {
        "diag_every": config_inputs["diag_every"],
        "use_mean_betas": config_inputs["use_mean_betas"],
        "post_burn_diag_config": post_burn_diag_config,
        "burn_in_config": burn_in_config,
    }
    epoch_iteration_static_config = GibbsEpochIterationStaticConfig(
        inner_beta_kwargs=inner_beta_kwargs,
        iteration_update_config=iteration_update_config,
        cur_background_log_bf_v=config_inputs["cur_background_log_bf_v"],
        y_var_orig=config_inputs["y_var_orig"],
        gauss_seidel=config_inputs["gauss_seidel"],
        initial_linear_filter=config_inputs["initial_linear_filter"],
        sparse_frac_gibbs=config_inputs["sparse_frac_gibbs"],
        sparse_max_gibbs=config_inputs["sparse_max_gibbs"],
        correct_betas_mean=config_inputs["correct_betas_mean"],
        correct_betas_var=config_inputs["correct_betas_var"],
        prefilter_config=prefilter_config,
        iteration_progress_config=iteration_progress_config,
    )
    return GibbsEpochRuntimeConfigs(
        epoch_phase_config=epoch_phase_config,
        epoch_iteration_static_config=epoch_iteration_static_config,
    )


def _build_gibbs_epoch_runtime_config_inputs(gibbs_controls, dynamic_inputs):
    return {
        "total_num_iter": gibbs_controls.total_num_iter,
        "num_chains": gibbs_controls.num_chains,
        "num_full_gene_sets": dynamic_inputs["num_full_gene_sets"],
        "use_mean_betas": dynamic_inputs["use_mean_betas"],
        "max_mb_X_h": dynamic_inputs["max_mb_X_h"],
        "target_num_epochs": gibbs_controls.target_num_epochs,
        "num_mad": dynamic_inputs["num_mad"],
        "adjust_priors": dynamic_inputs["adjust_priors"],
        "epoch_max_num_iter_config": gibbs_controls.epoch_max_num_iter_config,
        "min_num_burn_in": gibbs_controls.min_num_burn_in,
        "max_num_burn_in": gibbs_controls.max_num_burn_in,
        "min_num_post_burn_in": gibbs_controls.min_num_post_burn_in,
        "max_num_post_burn_in": gibbs_controls.max_num_post_burn_in,
        "increase_hyper_if_betas_below": dynamic_inputs["increase_hyper_if_betas_below"],
        "experimental_hyper_mutation": dynamic_inputs["experimental_hyper_mutation"],
        "warm_start": dynamic_inputs["warm_start"],
        "debug_zero_sparse": dynamic_inputs["debug_zero_sparse"],
        "num_batches_parallel": dynamic_inputs["num_batches_parallel"],
        "betas_trace_out": dynamic_inputs["betas_trace_out"],
        "update_huge_scores": dynamic_inputs["update_huge_scores"],
        "compute_Y_raw": dynamic_inputs["compute_Y_raw"],
        "sparse_frac_gibbs": dynamic_inputs["sparse_frac_gibbs"],
        "sparse_max_gibbs": dynamic_inputs["sparse_max_gibbs"],
        "pre_filter_batch_size": dynamic_inputs["pre_filter_batch_size"],
        "pre_filter_small_batch_size": dynamic_inputs["pre_filter_small_batch_size"],
        "initial_linear_filter": dynamic_inputs["initial_linear_filter"],
        "correct_betas_mean": dynamic_inputs["correct_betas_mean"],
        "correct_betas_var": dynamic_inputs["correct_betas_var"],
        "cur_background_log_bf_v": dynamic_inputs["cur_background_log_bf_v"],
        "y_var_orig": dynamic_inputs["y_var_orig"],
        "stop_mcse_quantile": dynamic_inputs["stop_mcse_quantile"],
        "max_rel_mcse_beta": dynamic_inputs["max_rel_mcse_beta"],
        "max_abs_mcse_d": dynamic_inputs["max_abs_mcse_d"],
        "r_threshold_burn_in": dynamic_inputs["r_threshold_burn_in"],
        "gauss_seidel": dynamic_inputs["gauss_seidel"],
        "eps": dynamic_inputs["eps"],
        "passed_in_max_num_burn_in": gibbs_controls.passed_in_max_num_burn_in,
        "max_num_iter_betas": dynamic_inputs["max_num_iter_betas"],
        "min_num_iter_betas": dynamic_inputs["min_num_iter_betas"],
        "num_chains_betas": dynamic_inputs["num_chains_betas"],
        "r_threshold_burn_in_betas": dynamic_inputs["r_threshold_burn_in_betas"],
        "use_max_r_for_convergence_betas": dynamic_inputs["use_max_r_for_convergence_betas"],
        "max_frac_sem_betas": dynamic_inputs["max_frac_sem_betas"],
        "max_allowed_batch_correlation": dynamic_inputs["max_allowed_batch_correlation"],
        "gauss_seidel_betas": dynamic_inputs["gauss_seidel_betas"],
        "sparse_solution": dynamic_inputs["sparse_solution"],
        "sparse_frac_betas": dynamic_inputs["sparse_frac_betas"],
        "active_beta_top_k": gibbs_controls.active_beta_top_k,
        "active_beta_min_abs": gibbs_controls.active_beta_min_abs,
        "burn_in_rhat_quantile": gibbs_controls.burn_in_rhat_quantile,
        "stall_window": gibbs_controls.stall_window,
        "stall_min_burn_in": gibbs_controls.stall_min_burn_in,
        "stall_delta_rhat": gibbs_controls.stall_delta_rhat,
        "stall_recent_window": gibbs_controls.stall_recent_window,
        "stall_recent_eps": gibbs_controls.stall_recent_eps,
        "burn_in_stall_window": gibbs_controls.burn_in_stall_window,
        "burn_in_stall_delta": gibbs_controls.burn_in_stall_delta,
        "diag_every": gibbs_controls.diag_every,
        "burn_in_patience": gibbs_controls.burn_in_patience,
        "stop_patience": gibbs_controls.stop_patience,
        "beta_rel_mcse_denom_floor": gibbs_controls.beta_rel_mcse_denom_floor,
        "stop_top_gene_k": gibbs_controls.stop_top_gene_k,
        "stop_min_gene_d": gibbs_controls.stop_min_gene_d,
        "stall_min_post_burn_in": gibbs_controls.stall_min_post_burn_in,
        "stall_delta_mcse": gibbs_controls.stall_delta_mcse,
    }


def _build_gibbs_dynamic_runtime_inputs(
    gibbs_inputs,
    use_mean_betas,
    max_mb_X_h,
    num_mad,
    adjust_priors,
    increase_hyper_if_betas_below,
    experimental_hyper_mutation,
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
    warm_start,
    debug_zero_sparse,
    num_batches_parallel,
    betas_trace_out,
    update_huge_scores,
    sparse_frac_gibbs,
    sparse_max_gibbs,
    pre_filter_batch_size,
    pre_filter_small_batch_size,
    r_threshold_burn_in,
    gauss_seidel,
    eps,
    stop_mcse_quantile,
    max_rel_mcse_beta,
    max_abs_mcse_d,
    initial_linear_filter,
    correct_betas_mean,
    correct_betas_var,
):
    return {
        "num_full_gene_sets": gibbs_inputs["num_full_gene_sets"],
        "use_mean_betas": use_mean_betas,
        "max_mb_X_h": max_mb_X_h,
        "num_mad": num_mad,
        "adjust_priors": adjust_priors,
        "increase_hyper_if_betas_below": increase_hyper_if_betas_below,
        "experimental_hyper_mutation": experimental_hyper_mutation,
        "max_num_iter_betas": max_num_iter_betas,
        "min_num_iter_betas": min_num_iter_betas,
        "num_chains_betas": num_chains_betas,
        "r_threshold_burn_in_betas": r_threshold_burn_in_betas,
        "use_max_r_for_convergence_betas": use_max_r_for_convergence_betas,
        "max_frac_sem_betas": max_frac_sem_betas,
        "max_allowed_batch_correlation": max_allowed_batch_correlation,
        "gauss_seidel_betas": gauss_seidel_betas,
        "sparse_solution": sparse_solution,
        "sparse_frac_betas": sparse_frac_betas,
        "warm_start": warm_start,
        "debug_zero_sparse": debug_zero_sparse,
        "num_batches_parallel": num_batches_parallel,
        "betas_trace_out": betas_trace_out,
        "update_huge_scores": update_huge_scores,
        "compute_Y_raw": gibbs_inputs["compute_Y_raw"],
        "sparse_frac_gibbs": sparse_frac_gibbs,
        "sparse_max_gibbs": sparse_max_gibbs,
        "pre_filter_batch_size": pre_filter_batch_size,
        "pre_filter_small_batch_size": pre_filter_small_batch_size,
        "r_threshold_burn_in": r_threshold_burn_in,
        "gauss_seidel": gauss_seidel,
        "eps": eps,
        "stop_mcse_quantile": stop_mcse_quantile,
        "max_rel_mcse_beta": max_rel_mcse_beta,
        "max_abs_mcse_d": max_abs_mcse_d,
        "initial_linear_filter": initial_linear_filter,
        "correct_betas_mean": correct_betas_mean,
        "correct_betas_var": correct_betas_var,
        "cur_background_log_bf_v": gibbs_inputs["cur_background_log_bf_v"],
        "y_var_orig": gibbs_inputs["y_var_orig"],
    }


def _build_gibbs_epoch_iteration_loop_config(
    epoch_context,
    epoch_phase_config,
    epoch_iteration_static_config,
    run_state,
):
    return GibbsEpochIterationLoopConfig(
        epoch_max_num_iter=epoch_context["epoch_max_num_iter"],
        epoch_total_iter_offset=epoch_context["epoch_total_iter_offset"],
        trace_chain_offset=epoch_context["trace_chain_offset"],
        full_betas_m_shape=epoch_context["full_betas_m_shape"],
        num_stack_batches=epoch_context["num_stack_batches"],
        stack_batch_size=epoch_context["stack_batch_size"],
        X_hstacked=epoch_context["X_hstacked"],
        min_num_burn_in_for_epoch=epoch_context["min_num_burn_in_for_epoch"],
        max_num_burn_in_for_epoch=epoch_context["max_num_burn_in_for_epoch"],
        min_num_iter_for_epoch=epoch_context["min_num_iter_for_epoch"],
        min_num_post_burn_in_for_epoch=epoch_context["min_num_post_burn_in_for_epoch"],
        max_num_post_burn_in_for_epoch=epoch_context["max_num_post_burn_in_for_epoch"],
        post_burn_reset_arrays=epoch_context["post_burn_reset_arrays"],
        post_burn_reset_missing_arrays=epoch_context["post_burn_reset_missing_arrays"],
        inner_beta_kwargs=epoch_iteration_static_config.inner_beta_kwargs,
        iteration_update_config=epoch_iteration_static_config.iteration_update_config,
        cur_background_log_bf_v=epoch_iteration_static_config.cur_background_log_bf_v,
        y_var_orig=epoch_iteration_static_config.y_var_orig,
        gauss_seidel=epoch_iteration_static_config.gauss_seidel,
        initial_linear_filter=epoch_iteration_static_config.initial_linear_filter,
        sparse_frac_gibbs=epoch_iteration_static_config.sparse_frac_gibbs,
        sparse_max_gibbs=epoch_iteration_static_config.sparse_max_gibbs,
        correct_betas_mean=epoch_iteration_static_config.correct_betas_mean,
        correct_betas_var=epoch_iteration_static_config.correct_betas_var,
        prefilter_config=epoch_iteration_static_config.prefilter_config,
        iteration_progress_config=epoch_iteration_static_config.iteration_progress_config,
        num_attempts=run_state.num_attempts,
        max_num_attempt_restarts=run_state.max_num_attempt_restarts,
        num_mad=epoch_phase_config.num_mad,
        increase_hyper_if_betas_below_for_epoch=epoch_context["increase_hyper_if_betas_below_for_epoch"],
        experimental_hyper_mutation=epoch_context["experimental_hyper_mutation"],
        num_before_checking_p_increase=epoch_context["num_before_checking_p_increase"],
        p_scale_factor=epoch_context["p_scale_factor"],
    )


def _build_gibbs_iteration_runtime_configs(loop_config, epoch_priors, gene_stats_trace_fh):
    correction_config = GibbsIterationCorrectionConfig(
        inner_beta_kwargs=loop_config.inner_beta_kwargs,
        iteration_update_config=loop_config.iteration_update_config,
        num_mad=loop_config.num_mad,
        num_attempts=loop_config.num_attempts,
        max_num_attempt_restarts=loop_config.max_num_attempt_restarts,
        increase_hyper_if_betas_below_for_epoch=loop_config.increase_hyper_if_betas_below_for_epoch,
        experimental_hyper_mutation=loop_config.experimental_hyper_mutation,
        num_before_checking_p_increase=loop_config.num_before_checking_p_increase,
        p_scale_factor=loop_config.p_scale_factor,
    )
    progress_runtime_config = GibbsIterationProgressRuntimeConfig(
        trace_chain_offset=loop_config.trace_chain_offset,
        epoch_total_iter_offset=loop_config.epoch_total_iter_offset,
        epoch_max_num_iter=loop_config.epoch_max_num_iter,
        max_num_burn_in_for_epoch=loop_config.max_num_burn_in_for_epoch,
        min_num_iter_for_epoch=loop_config.min_num_iter_for_epoch,
        min_num_burn_in_for_epoch=loop_config.min_num_burn_in_for_epoch,
        max_num_post_burn_in_for_epoch=loop_config.max_num_post_burn_in_for_epoch,
        min_num_post_burn_in_for_epoch=loop_config.min_num_post_burn_in_for_epoch,
        post_burn_reset_arrays=loop_config.post_burn_reset_arrays,
        post_burn_reset_missing_arrays=loop_config.post_burn_reset_missing_arrays,
        iteration_progress_config=loop_config.iteration_progress_config,
    )
    iteration_state_config = {
        "epoch_total_iter_offset": loop_config.epoch_total_iter_offset,
        "trace_chain_offset": loop_config.trace_chain_offset,
        "full_betas_m_shape": loop_config.full_betas_m_shape,
        "num_stack_batches": loop_config.num_stack_batches,
        "stack_batch_size": loop_config.stack_batch_size,
        "X_hstacked": loop_config.X_hstacked,
        "inner_beta_kwargs": loop_config.inner_beta_kwargs,
        "cur_background_log_bf_v": loop_config.cur_background_log_bf_v,
        "y_var_orig": loop_config.y_var_orig,
        "gauss_seidel": loop_config.gauss_seidel,
        "initial_linear_filter": loop_config.initial_linear_filter,
        "sparse_frac_gibbs": loop_config.sparse_frac_gibbs,
        "sparse_max_gibbs": loop_config.sparse_max_gibbs,
        "correct_betas_mean": loop_config.correct_betas_mean,
        "correct_betas_var": loop_config.correct_betas_var,
        "prefilter_config": loop_config.prefilter_config,
        "epoch_priors": epoch_priors,
        "gene_stats_trace_fh": gene_stats_trace_fh,
    }
    return GibbsIterationRuntimeConfigs(
        correction_config=correction_config,
        progress_runtime_config=progress_runtime_config,
        iteration_state_config=iteration_state_config,
    )


# ========================= Outer Gibbs Epoch Attempt Orchestration =========================
def _run_single_gibbs_epoch_attempt(
    state,
    run_state,
    epoch_aggregates,
    epoch_phase_config,
    epoch_iteration_static_config,
    gene_set_stats_trace_fh,
    gene_stats_trace_fh,
    log_bf_state,
):
    epoch_attempt = _prepare_gibbs_epoch_attempt(
        state=state,
        run_state=run_state,
        epoch_phase_config=epoch_phase_config,
    )
    if epoch_attempt is None:
        return _build_gibbs_epoch_attempt_result(
            log_bf_state=log_bf_state,
            attempt_started=False,
            should_continue=False,
        )

    return _run_started_gibbs_epoch_attempt(
        state=state,
        run_state=run_state,
        epoch_aggregates=epoch_aggregates,
        epoch_phase_config=epoch_phase_config,
        epoch_iteration_static_config=epoch_iteration_static_config,
        gene_set_stats_trace_fh=gene_set_stats_trace_fh,
        gene_stats_trace_fh=gene_stats_trace_fh,
        log_bf_state=log_bf_state,
        epoch_attempt=epoch_attempt,
    )


def _run_started_gibbs_epoch_attempt(
    state,
    run_state,
    epoch_aggregates,
    epoch_phase_config,
    epoch_iteration_static_config,
    gene_set_stats_trace_fh,
    gene_stats_trace_fh,
    log_bf_state,
    epoch_attempt,
):
    (log_bf_m, log_bf_uncorrected_m, log_bf_raw_m) = log_bf_state
    epoch_attempt_context = _initialize_gibbs_epoch_attempt_context(
        state=state,
        run_state=run_state,
        epoch_aggregates=epoch_aggregates,
        epoch_phase_config=epoch_phase_config,
        epoch_iteration_static_config=epoch_iteration_static_config,
        epoch_attempt=epoch_attempt,
    )
    epoch_context = epoch_attempt_context["epoch_context"]
    epoch_control = epoch_attempt_context["epoch_control"]
    epoch_sums = epoch_attempt_context["epoch_sums"]
    epoch_priors = epoch_attempt_context["epoch_priors"]
    epoch_runtime = epoch_attempt_context["epoch_runtime"]
    loop_config = epoch_attempt_context["loop_config"]
    epoch_loop_update = _run_gibbs_epoch_iterations(
        state=state,
        run_state=run_state,
        epoch_control=epoch_control,
        epoch_sums=epoch_sums,
        epoch_priors=epoch_priors,
        epoch_runtime=epoch_runtime,
        loop_config=loop_config,
        gene_set_stats_trace_fh=gene_set_stats_trace_fh,
        gene_stats_trace_fh=gene_stats_trace_fh,
        log_bf_state=(log_bf_m, log_bf_uncorrected_m, log_bf_raw_m),
    )
    iteration_num = epoch_loop_update["iteration_num"]
    log_bf_m, log_bf_uncorrected_m, log_bf_raw_m = _apply_gibbs_log_bf_update(epoch_loop_update)

    epoch_finalize_update = _finalize_gibbs_epoch_attempt(
        state=state,
        epoch_aggregates=epoch_aggregates,
        epoch_sums=epoch_sums,
        finalize_context=_build_gibbs_epoch_finalize_context(
            state=state,
            run_state=run_state,
            epoch_phase_config=epoch_phase_config,
            epoch_control=epoch_control,
            epoch_runtime=epoch_runtime,
            iteration_num=iteration_num,
        ),
    )
    should_continue = _apply_gibbs_epoch_finalize_update(
        run_state=run_state,
        epoch_runtime=epoch_runtime,
        epoch_finalize_update=epoch_finalize_update,
    )
    return _build_gibbs_epoch_attempt_result(
        log_bf_state=(log_bf_m, log_bf_uncorrected_m, log_bf_raw_m),
        attempt_started=True,
        should_continue=should_continue,
    )


def _initialize_gibbs_epoch_attempt_context(
    state,
    run_state,
    epoch_aggregates,
    epoch_phase_config,
    epoch_iteration_static_config,
    epoch_attempt,
):
    epoch_context = _start_gibbs_epoch(
        state=state,
        num_chains=epoch_phase_config.num_chains,
        num_full_gene_sets=epoch_phase_config.num_full_gene_sets,
        use_mean_betas=epoch_phase_config.use_mean_betas,
        max_mb_X_h=epoch_phase_config.max_mb_X_h,
        log_fun=log,
        epoch_aggregates=epoch_aggregates,
        num_p_increases=run_state.num_p_increases,
    )
    epoch_context.update(epoch_attempt)
    return {
        "epoch_context": epoch_context,
        "epoch_control": epoch_context["epoch_control"],
        "epoch_sums": epoch_context["epoch_sums"],
        "epoch_priors": epoch_context["epoch_priors"],
        "epoch_runtime": epoch_context["epoch_runtime"],
        "loop_config": _build_gibbs_epoch_iteration_loop_config(
            epoch_context=epoch_context,
            epoch_phase_config=epoch_phase_config,
            epoch_iteration_static_config=epoch_iteration_static_config,
            run_state=run_state,
        ),
    }


def _apply_gibbs_epoch_finalize_update(run_state, epoch_runtime, epoch_finalize_update):
    run_state.num_p_increases = epoch_runtime["num_p_increases"]
    run_state.remaining_total_iter = epoch_finalize_update["remaining_total_iter"]
    run_state.num_completed_epochs = epoch_finalize_update["num_completed_epochs"]
    return epoch_finalize_update["should_continue"]


def _apply_gibbs_log_bf_update(update):
    return (
        update["log_bf_m"],
        update["log_bf_uncorrected_m"],
        update["log_bf_raw_m"],
    )


def _build_gibbs_log_bf_payload(log_bf_m, log_bf_uncorrected_m, log_bf_raw_m, **extra):
    payload = {
        "log_bf_m": log_bf_m,
        "log_bf_uncorrected_m": log_bf_uncorrected_m,
        "log_bf_raw_m": log_bf_raw_m,
    }
    payload.update(extra)
    return payload


def _build_gibbs_epoch_attempt_result(log_bf_state, attempt_started, should_continue):
    (log_bf_m, log_bf_uncorrected_m, log_bf_raw_m) = log_bf_state
    return _build_gibbs_log_bf_payload(
        log_bf_m,
        log_bf_uncorrected_m,
        log_bf_raw_m,
        attempt_started=attempt_started,
        should_continue=should_continue,
    )


def _run_gibbs_epoch_phase(
    state,
    run_state,
    epoch_aggregates,
    epoch_phase_config,
    epoch_iteration_static_config,
    gene_set_stats_trace_fh,
    gene_stats_trace_fh,
    log_bf_state,
):
    # Gibbs Phase 1: run one or more epochs (optionally restarting on stalls).
    while _should_continue_gibbs_epoch_loop(run_state):
        (log_bf_state, should_break) = _run_and_apply_gibbs_epoch_attempt(
            state=state,
            run_state=run_state,
            epoch_aggregates=epoch_aggregates,
            epoch_phase_config=epoch_phase_config,
            epoch_iteration_static_config=epoch_iteration_static_config,
            gene_set_stats_trace_fh=gene_set_stats_trace_fh,
            gene_stats_trace_fh=gene_stats_trace_fh,
            log_bf_state=log_bf_state,
        )
        if should_break:
            break

    return None


def _build_initial_gibbs_log_bf_state(gibbs_inputs):
    return (
        gibbs_inputs["log_bf_m"],
        gibbs_inputs["log_bf_uncorrected_m"],
        gibbs_inputs["log_bf_raw_m"],
    )


def _run_gibbs_epochs_with_optional_traces(
    state,
    run_state,
    epoch_aggregates,
    epoch_phase_config,
    epoch_iteration_static_config,
    gene_set_stats_trace_out,
    gene_stats_trace_out,
    gibbs_inputs,
):
    with _open_optional_gibbs_trace_files(
        gene_set_stats_trace_out=gene_set_stats_trace_out,
        gene_stats_trace_out=gene_stats_trace_out,
    ) as (gene_set_stats_trace_fh, gene_stats_trace_fh):
        _run_gibbs_epoch_phase(
            state=state,
            run_state=run_state,
            epoch_aggregates=epoch_aggregates,
            epoch_phase_config=epoch_phase_config,
            epoch_iteration_static_config=epoch_iteration_static_config,
            gene_set_stats_trace_fh=gene_set_stats_trace_fh,
            gene_stats_trace_fh=gene_stats_trace_fh,
            log_bf_state=_build_initial_gibbs_log_bf_state(gibbs_inputs),
        )


def _run_and_apply_gibbs_epoch_attempt(
    state,
    run_state,
    epoch_aggregates,
    epoch_phase_config,
    epoch_iteration_static_config,
    gene_set_stats_trace_fh,
    gene_stats_trace_fh,
    log_bf_state,
):
    epoch_update = _run_single_gibbs_epoch_attempt(
        state=state,
        run_state=run_state,
        epoch_aggregates=epoch_aggregates,
        epoch_phase_config=epoch_phase_config,
        epoch_iteration_static_config=epoch_iteration_static_config,
        gene_set_stats_trace_fh=gene_set_stats_trace_fh,
        gene_stats_trace_fh=gene_stats_trace_fh,
        log_bf_state=log_bf_state,
    )
    return _apply_gibbs_epoch_attempt_update(epoch_update=epoch_update)


def _should_continue_gibbs_epoch_attempts(
    remaining_total_iter,
    num_completed_epochs,
    target_num_epochs,
    num_attempts,
    max_num_attempt_restarts,
    stop_due_to_stall=False,
    stop_due_to_precision=False,
):
    return (
        (not stop_due_to_stall)
        and (not stop_due_to_precision)
        and (num_completed_epochs < target_num_epochs)
        and (remaining_total_iter > 0)
        and (num_attempts < max_num_attempt_restarts)
    )


def _should_continue_gibbs_epoch_loop(run_state):
    return _should_continue_gibbs_epoch_attempts(
        remaining_total_iter=run_state.remaining_total_iter,
        num_completed_epochs=run_state.num_completed_epochs,
        target_num_epochs=run_state.target_num_epochs,
        num_attempts=run_state.num_attempts,
        max_num_attempt_restarts=run_state.max_num_attempt_restarts,
    )


def _apply_gibbs_epoch_attempt_update(epoch_update):
    if not epoch_update["attempt_started"]:
        return (_apply_gibbs_log_bf_update(epoch_update), True)
    return (_apply_gibbs_log_bf_update(epoch_update), not epoch_update["should_continue"])


# ========================= Outer Gibbs Per-Iteration Orchestration =========================
def _run_gibbs_epoch_iterations(
    state,
    run_state,
    epoch_control,
    epoch_sums,
    epoch_priors,
    epoch_runtime,
    loop_config,
    gene_set_stats_trace_fh,
    gene_stats_trace_fh,
    log_bf_state,
):
    epoch_max_num_iter = loop_config.epoch_max_num_iter
    iteration_runtime_configs = _build_gibbs_iteration_runtime_configs(
        loop_config=loop_config,
        epoch_priors=epoch_priors,
        gene_stats_trace_fh=gene_stats_trace_fh,
    )
    correction_config = iteration_runtime_configs.correction_config
    progress_runtime_config = iteration_runtime_configs.progress_runtime_config
    iteration_state_config = iteration_runtime_configs.iteration_state_config

    iteration_num = -1
    for iteration_num in range(epoch_max_num_iter):
        iteration_run = _run_single_gibbs_iteration(
            state=state,
            run_state=run_state,
            epoch_control=epoch_control,
            epoch_sums=epoch_sums,
            epoch_priors=epoch_priors,
            epoch_runtime=epoch_runtime,
            correction_config=correction_config,
            progress_runtime_config=progress_runtime_config,
            iteration_state_config=iteration_state_config,
            gene_set_stats_trace_fh=gene_set_stats_trace_fh,
            iteration_num=iteration_num,
            log_bf_state=log_bf_state,
        )
        log_bf_state, stop_epoch = _apply_gibbs_iteration_loop_update(iteration_run=iteration_run)
        if stop_epoch:
            break

    (log_bf_m, log_bf_uncorrected_m, log_bf_raw_m) = log_bf_state
    return _build_gibbs_log_bf_payload(
        log_bf_m,
        log_bf_uncorrected_m,
        log_bf_raw_m,
        iteration_num=iteration_num,
    )


def _apply_gibbs_iteration_loop_update(iteration_run):
    return (_apply_gibbs_log_bf_update(iteration_run), iteration_run["stop_epoch"])


def _extract_gibbs_iteration_update_state(iteration_update):
    return (_apply_gibbs_log_bf_update(iteration_update), iteration_update["should_break"])


# ========================= Outer Gibbs Iteration Context Builders =========================
def _build_gibbs_iteration_correction_context(
    state,
    iter_state,
    gene_set_mask_m,
    epoch_control,
    correction_config,
    epoch_priors,
    epoch_runtime,
    epoch_sums,
    iteration_num,
    log_bf_state,
):
    return {
        "state": state,
        "iter_state": iter_state,
        "gene_set_mask_m": gene_set_mask_m,
        "epoch_control": epoch_control,
        "correction_config": correction_config,
        "epoch_priors": epoch_priors,
        "epoch_runtime": epoch_runtime,
        "epoch_sums": epoch_sums,
        "iteration_num": iteration_num,
        "log_bf_state": log_bf_state,
    }


def _build_gibbs_iteration_finalize_context(
    state,
    epoch_control,
    run_state,
    progress_runtime_config,
    iter_state,
    iteration_num,
    epoch_sums,
    epoch_priors,
    epoch_runtime,
    gene_set_stats_trace_fh,
    iteration_update,
    should_break,
    log_bf_state,
):
    return {
        "state": state,
        "epoch_control": epoch_control,
        "run_state": run_state,
        "progress_runtime_config": progress_runtime_config,
        "iter_state": iter_state,
        "iteration_num": iteration_num,
        "epoch_sums": epoch_sums,
        "epoch_priors": epoch_priors,
        "epoch_runtime": epoch_runtime,
        "gene_set_stats_trace_fh": gene_set_stats_trace_fh,
        "iteration_update": iteration_update,
        "should_break": should_break,
        "log_bf_state": log_bf_state,
    }


def _build_gibbs_iteration_progress_update_context(
    state,
    epoch_control,
    run_state,
    progress_runtime_config,
    iter_state,
    iteration_num,
    epoch_sums,
    epoch_priors,
    epoch_runtime,
    gene_set_stats_trace_fh,
    iteration_update,
    log_bf_state,
):
    return {
        "state": state,
        "epoch_control": epoch_control,
        "run_state": run_state,
        "progress_runtime_config": progress_runtime_config,
        "iter_state": iter_state,
        "iteration_num": iteration_num,
        "epoch_sums": epoch_sums,
        "epoch_priors": epoch_priors,
        "epoch_runtime": epoch_runtime,
        "gene_set_stats_trace_fh": gene_set_stats_trace_fh,
        "iteration_update": iteration_update,
        "log_bf_state": log_bf_state,
    }


def _run_single_gibbs_iteration(
    state,
    run_state,
    epoch_control,
    epoch_sums,
    epoch_priors,
    epoch_runtime,
    correction_config,
    progress_runtime_config,
    iteration_state_config,
    gene_set_stats_trace_fh,
    iteration_num,
    log_bf_state,
):
    (log_bf_m, log_bf_uncorrected_m, log_bf_raw_m) = log_bf_state
    iter_state, gene_set_mask_m = _prepare_gibbs_iteration_state(
        state=state,
        iteration_num=iteration_num,
        iteration_state_config=iteration_state_config,
        log_bf_m=log_bf_m,
        log_bf_raw_m=log_bf_raw_m,
    )

    iteration_update = _run_gibbs_iteration_correction_and_updates(
        correction_context=_build_gibbs_iteration_correction_context(
            state=state,
            iter_state=iter_state,
            gene_set_mask_m=gene_set_mask_m,
            epoch_control=epoch_control,
            correction_config=correction_config,
            epoch_priors=epoch_priors,
            epoch_runtime=epoch_runtime,
            epoch_sums=epoch_sums,
            iteration_num=iteration_num,
            log_bf_state=log_bf_state,
        ),
    )
    (log_bf_state, should_break) = _extract_gibbs_iteration_update_state(iteration_update)

    return _finalize_gibbs_iteration_after_correction(
        finalize_context=_build_gibbs_iteration_finalize_context(
            state=state,
            epoch_control=epoch_control,
            run_state=run_state,
            progress_runtime_config=progress_runtime_config,
            iter_state=iter_state,
            iteration_num=iteration_num,
            epoch_sums=epoch_sums,
            epoch_priors=epoch_priors,
            epoch_runtime=epoch_runtime,
            gene_set_stats_trace_fh=gene_set_stats_trace_fh,
            iteration_update=iteration_update,
            should_break=should_break,
            log_bf_state=log_bf_state,
        ),
    )


def _finalize_gibbs_iteration_after_correction(finalize_context):
    state = finalize_context["state"]
    epoch_control = finalize_context["epoch_control"]
    run_state = finalize_context["run_state"]
    progress_runtime_config = finalize_context["progress_runtime_config"]
    iter_state = finalize_context["iter_state"]
    iteration_num = finalize_context["iteration_num"]
    epoch_sums = finalize_context["epoch_sums"]
    epoch_priors = finalize_context["epoch_priors"]
    epoch_runtime = finalize_context["epoch_runtime"]
    gene_set_stats_trace_fh = finalize_context["gene_set_stats_trace_fh"]
    iteration_update = finalize_context["iteration_update"]
    should_break = finalize_context["should_break"]
    log_bf_state = finalize_context["log_bf_state"]

    (log_bf_m, log_bf_uncorrected_m, log_bf_raw_m) = log_bf_state
    if should_break:
        return _build_gibbs_log_bf_payload(
            log_bf_m,
            log_bf_uncorrected_m,
            log_bf_raw_m,
            stop_epoch=True,
        )

    iteration_progress_update = _advance_gibbs_iteration_progress(
        progress_update_context=_build_gibbs_iteration_progress_update_context(
            state=state,
            epoch_control=epoch_control,
            run_state=run_state,
            progress_runtime_config=progress_runtime_config,
            iter_state=iter_state,
            iteration_num=iteration_num,
            epoch_sums=epoch_sums,
            epoch_priors=epoch_priors,
            epoch_runtime=epoch_runtime,
            gene_set_stats_trace_fh=gene_set_stats_trace_fh,
            iteration_update=iteration_update,
            log_bf_state=(log_bf_m, log_bf_uncorrected_m, log_bf_raw_m),
        ),
    )
    stop_epoch = iteration_progress_update["done"]
    return _build_gibbs_log_bf_payload(
        log_bf_m,
        log_bf_uncorrected_m,
        log_bf_raw_m,
        stop_epoch=stop_epoch,
    )


def _run_gibbs_corrected_betas_step(
    state,
    iter_state,
    gene_set_mask_m,
    epoch_priors,
    iteration_update_config,
    inner_beta_kwargs,
):
    return _compute_gibbs_corrected_betas_for_gene_set_mask(
        state,
        gene_set_mask_m=gene_set_mask_m,
        default_betas_sample_m=iter_state["default_betas_sample_m"],
        default_postp_sample_m=iter_state["default_postp_sample_m"],
        default_betas_mean_m=iter_state["default_betas_mean_m"],
        default_postp_mean_m=iter_state["default_postp_mean_m"],
        full_beta_tildes_m=iter_state["full_beta_tildes_m"],
        full_ses_m=iter_state["full_ses_m"],
        full_scale_factors_m=iter_state["full_scale_factors_m"],
        full_mean_shifts_m=iter_state["full_mean_shifts_m"],
        full_is_dense_gene_set_m=iter_state["full_is_dense_gene_set_m"],
        full_ps_m=iter_state["full_ps_m"],
        full_sigma2s_m=iter_state["full_sigma2s_m"],
        uncorrected_betas_mean_m=iter_state["uncorrected_betas_mean_m"],
        use_mean_betas=iteration_update_config.use_mean_betas,
        warm_start=iteration_update_config.warm_start,
        prev_warm_start_betas_m=epoch_priors["prev_warm_start_betas_m"],
        prev_warm_start_postp_m=epoch_priors["prev_warm_start_postp_m"],
        debug_zero_sparse=iteration_update_config.debug_zero_sparse,
        num_chains=iteration_update_config.num_chains,
        num_batches_parallel=iteration_update_config.num_batches_parallel,
        **inner_beta_kwargs,
        betas_trace_out=iteration_update_config.betas_trace_out,
    )


def _apply_refresh_update_to_epoch_priors(epoch_priors, refresh_update):
    epoch_priors["prev_warm_start_betas_m"] = refresh_update["prev_warm_start_betas_m"]
    epoch_priors["prev_warm_start_postp_m"] = refresh_update["prev_warm_start_postp_m"]
    epoch_priors["priors_sample_m"] = refresh_update["priors_sample_m"]
    epoch_priors["priors_mean_m"] = refresh_update["priors_mean_m"]
    epoch_priors["priors_missing_sample_m"] = refresh_update["priors_missing_sample_m"]
    epoch_priors["priors_missing_mean_m"] = refresh_update["priors_missing_mean_m"]
    return _apply_gibbs_log_bf_update(refresh_update)


def _apply_prior_update_to_epoch_priors(epoch_priors, prior_update):
    epoch_priors["priors_sample_m"] = prior_update["priors_sample_m"]
    epoch_priors["priors_mean_m"] = prior_update["priors_mean_m"]
    epoch_priors["priors_missing_sample_m"] = prior_update["priors_missing_sample_m"]
    epoch_priors["priors_missing_mean_m"] = prior_update["priors_missing_mean_m"]
    epoch_priors["priors_for_Y_m"] = prior_update["priors_for_Y_m"]
    epoch_priors["priors_percentage_max_for_Y_m"] = prior_update["priors_percentage_max_for_Y_m"]
    epoch_priors["priors_adjustment_for_Y_m"] = prior_update["priors_adjustment_for_Y_m"]


def _build_refresh_gibbs_iteration_inputs(epoch_priors, iteration_update_config, log_bf_state):
    (log_bf_m, log_bf_uncorrected_m, log_bf_raw_m) = log_bf_state
    return {
        "warm_start": iteration_update_config.warm_start,
        "use_mean_betas": iteration_update_config.use_mean_betas,
        "prev_warm_start_betas_m": epoch_priors["prev_warm_start_betas_m"],
        "prev_warm_start_postp_m": epoch_priors["prev_warm_start_postp_m"],
        "priors_missing_sample_m": epoch_priors["priors_missing_sample_m"],
        "priors_missing_mean_m": epoch_priors["priors_missing_mean_m"],
        "priors_for_Y_m": epoch_priors["priors_for_Y_m"],
        "update_huge_scores": iteration_update_config.update_huge_scores,
        "compute_Y_raw": iteration_update_config.compute_Y_raw,
        "log_bf_m": log_bf_m,
        "log_bf_uncorrected_m": log_bf_uncorrected_m,
        "log_bf_raw_m": log_bf_raw_m,
    }


def _build_finalize_gibbs_prior_inputs(epoch_priors, iteration_update_config):
    return {
        "priors_sample_m": epoch_priors["priors_sample_m"],
        "priors_mean_m": epoch_priors["priors_mean_m"],
        "priors_missing_sample_m": epoch_priors["priors_missing_sample_m"],
        "priors_missing_mean_m": epoch_priors["priors_missing_mean_m"],
        "adjust_priors": iteration_update_config.adjust_priors,
        "use_mean_betas": iteration_update_config.use_mean_betas,
        "priors_percentage_max_sample_m": epoch_priors["priors_percentage_max_sample_m"],
        "priors_percentage_max_mean_m": epoch_priors["priors_percentage_max_mean_m"],
        "priors_adjustment_sample_m": epoch_priors["priors_adjustment_sample_m"],
        "priors_adjustment_mean_m": epoch_priors["priors_adjustment_mean_m"],
    }


def _compute_gibbs_iteration_betas_and_priors(
    state,
    iter_state,
    gene_set_mask_m,
    correction_config,
    epoch_priors,
    log_bf_state,
):
    inner_beta_kwargs = correction_config.inner_beta_kwargs
    iteration_update_config = correction_config.iteration_update_config

    (
        full_betas_sample_m,
        full_postp_sample_m,
        full_betas_mean_m,
        full_postp_mean_m,
    ) = _run_gibbs_corrected_betas_step(
        state=state,
        iter_state=iter_state,
        gene_set_mask_m=gene_set_mask_m,
        epoch_priors=epoch_priors,
        iteration_update_config=iteration_update_config,
        inner_beta_kwargs=inner_beta_kwargs,
    )

    refresh_inputs = _build_refresh_gibbs_iteration_inputs(
        epoch_priors,
        iteration_update_config,
        log_bf_state,
    )
    refresh_update = _refresh_gibbs_iteration_priors_and_huge(
        state,
        full_betas_sample_m=full_betas_sample_m,
        full_betas_mean_m=full_betas_mean_m,
        full_postp_sample_m=full_postp_sample_m,
        full_postp_mean_m=full_postp_mean_m,
        **refresh_inputs,
    )
    log_bf_state = _apply_refresh_update_to_epoch_priors(
        epoch_priors,
        refresh_update,
    )

    finalize_inputs = _build_finalize_gibbs_prior_inputs(
        epoch_priors,
        iteration_update_config,
    )
    prior_update = _finalize_gibbs_priors_for_sampling(
        state,
        **finalize_inputs,
    )
    _apply_prior_update_to_epoch_priors(epoch_priors, prior_update)

    (log_bf_m, log_bf_uncorrected_m, log_bf_raw_m) = log_bf_state
    return {
        "full_betas_sample_m": full_betas_sample_m,
        "full_postp_sample_m": full_postp_sample_m,
        "full_betas_mean_m": full_betas_mean_m,
        "full_postp_mean_m": full_postp_mean_m,
        "log_bf_m": log_bf_m,
        "log_bf_uncorrected_m": log_bf_uncorrected_m,
        "log_bf_raw_m": log_bf_raw_m,
    }


def _run_gibbs_iteration_correction_and_updates(correction_context):
    state = correction_context["state"]
    iter_state = correction_context["iter_state"]
    gene_set_mask_m = correction_context["gene_set_mask_m"]
    epoch_control = correction_context["epoch_control"]
    correction_config = correction_context["correction_config"]
    epoch_priors = correction_context["epoch_priors"]
    epoch_runtime = correction_context["epoch_runtime"]
    epoch_sums = correction_context["epoch_sums"]
    iteration_num = correction_context["iteration_num"]
    log_bf_state = correction_context["log_bf_state"]

    (log_bf_m, log_bf_uncorrected_m, log_bf_raw_m) = log_bf_state
    restart_controls = _build_gibbs_low_beta_restart_controls(correction_config)

    # Compute corrected betas, refresh priors/HuGE scores, then update all-iteration
    # sums and restart diagnostics.
    iteration_betas_priors = _compute_gibbs_iteration_betas_and_priors(
        state,
        iter_state=iter_state,
        gene_set_mask_m=gene_set_mask_m,
        correction_config=correction_config,
        epoch_priors=epoch_priors,
        log_bf_state=(log_bf_m, log_bf_uncorrected_m, log_bf_raw_m),
    )
    full_betas_sample_m = iteration_betas_priors["full_betas_sample_m"]
    full_postp_sample_m = iteration_betas_priors["full_postp_sample_m"]
    full_betas_mean_m = iteration_betas_priors["full_betas_mean_m"]
    full_postp_mean_m = iteration_betas_priors["full_postp_mean_m"]
    (log_bf_m, log_bf_uncorrected_m, log_bf_raw_m) = _apply_gibbs_log_bf_update(iteration_betas_priors)

    all_iteration_update = _update_gibbs_all_sums_and_maybe_restart_low_betas(
        state=state,
        epoch_runtime=epoch_runtime,
        epoch_sums=epoch_sums,
        restart_controls=restart_controls,
        iteration_num=iteration_num,
        full_betas_mean_m=full_betas_mean_m,
    )
    should_break = _apply_gibbs_all_iteration_update(
        epoch_runtime=epoch_runtime,
        epoch_control=epoch_control,
        all_iteration_update=all_iteration_update,
    )

    return {
        "full_betas_sample_m": full_betas_sample_m,
        "full_postp_sample_m": full_postp_sample_m,
        "full_betas_mean_m": full_betas_mean_m,
        "full_postp_mean_m": full_postp_mean_m,
        "log_bf_m": log_bf_m,
        "log_bf_uncorrected_m": log_bf_uncorrected_m,
        "log_bf_raw_m": log_bf_raw_m,
        "should_break": should_break,
    }


@dataclass
class GibbsLowBetaRestartControls:
    num_mad: int
    num_attempts: int
    max_num_attempt_restarts: int
    increase_hyper_if_betas_below_for_epoch: float | None
    experimental_hyper_mutation: bool
    num_before_checking_p_increase: int
    p_scale_factor: float


@dataclass
class GibbsLowBetaRestartUpdate:
    gibbs_good: bool
    num_p_increases: int
    should_break: bool


@dataclass
class GibbsAllIterationUpdate:
    all_sum_betas_m: object
    all_sum_betas2_m: object
    all_num_sum_m: object
    R_beta_v: object
    gibbs_good: bool
    num_p_increases: int
    should_break: bool


def _build_gibbs_low_beta_restart_controls(correction_config):
    return GibbsLowBetaRestartControls(
        num_mad=correction_config.num_mad,
        num_attempts=correction_config.num_attempts,
        max_num_attempt_restarts=correction_config.max_num_attempt_restarts,
        increase_hyper_if_betas_below_for_epoch=correction_config.increase_hyper_if_betas_below_for_epoch,
        experimental_hyper_mutation=correction_config.experimental_hyper_mutation,
        num_before_checking_p_increase=correction_config.num_before_checking_p_increase,
        p_scale_factor=correction_config.p_scale_factor,
    )


def _apply_gibbs_all_iteration_update(epoch_runtime, epoch_control, all_iteration_update):
    for key in _GIBBS_EPOCH_RUNTIME_SUM_KEYS:
        epoch_runtime[key] = getattr(all_iteration_update, key)
    epoch_control["R_beta_v"] = all_iteration_update.R_beta_v
    epoch_runtime["gibbs_good"] = all_iteration_update.gibbs_good
    epoch_runtime["num_p_increases"] = all_iteration_update.num_p_increases
    return all_iteration_update.should_break


def _compute_gibbs_y_corr_sparse(y_corr_sparse_base, priors_for_Y_m, y_var_orig):
    if y_corr_sparse_base is None:
        return None

    y_corr_sparse = copy.copy(y_corr_sparse_base)
    y_corr_sparse = y_corr_sparse.multiply(y_var_orig)

    new_y_sd = np.sqrt(np.square(np.mean(priors_for_Y_m, axis=0)) + y_var_orig)[np.newaxis,:]
    new_y_sd[new_y_sd == 0] = 1e-10

    y_corr_sparse = y_corr_sparse.multiply(1 / new_y_sd.T)
    y_corr_sparse = y_corr_sparse.multiply(1 / new_y_sd)
    y_corr_sparse.setdiag(1)
    return y_corr_sparse.tocsc()


def _sample_gibbs_p_targets(Y_sample_m, D_sample_m, gauss_seidel):
    # Keep the legacy initialization shape/type behavior, then overwrite by mode.
    p_sample_m = copy.copy(Y_sample_m)
    if not gauss_seidel:
        p_sample_m = np.zeros(D_sample_m.shape)
        p_sample_m[np.random.random(D_sample_m.shape) < D_sample_m] = 1
    else:
        p_sample_m = D_sample_m
    return p_sample_m


# ========================= Outer Gibbs Logistic Updates =========================


def _compute_gibbs_pre_gene_set_filter_mask(
    state,
    initial_linear_filter,
    Y_sample_m,
    y_var,
    y_corr_sparse,
    inner_beta_kwargs_linear,
    full_scale_factors_m,
    full_mean_shifts_m,
    full_is_dense_gene_set_m,
    full_ps_m,
    full_sigma2s_m,
    sparse_frac_gibbs,
    sparse_max_gibbs,
    num_gene_sets,
):
    if not initial_linear_filter:
        return np.full(num_gene_sets, True)

    (
        linear_uncorrected_betas_sample_m,
        linear_uncorrected_betas_mean_m,
        linear_p_values_m,
    ) = _compute_gibbs_linear_prefilter_betas(
        state=state,
        Y_sample_m=Y_sample_m,
        y_var=y_var,
        y_corr_sparse=y_corr_sparse,
        inner_beta_kwargs_linear=inner_beta_kwargs_linear,
        full_scale_factors_m=full_scale_factors_m,
        full_mean_shifts_m=full_mean_shifts_m,
        full_is_dense_gene_set_m=full_is_dense_gene_set_m,
        full_ps_m=full_ps_m,
        full_sigma2s_m=full_sigma2s_m,
    )
    pre_gene_set_filter_mask_m = _get_gibbs_gene_set_mask(
        linear_uncorrected_betas_mean_m,
        linear_uncorrected_betas_sample_m,
        linear_p_values_m,
        sparse_frac=sparse_frac_gibbs,
        sparse_max=sparse_max_gibbs,
    )
    pre_gene_set_filter_mask = np.any(pre_gene_set_filter_mask_m, axis=0)
    log("Filtered down to %d gene sets using linear pre-filtering" % np.sum(pre_gene_set_filter_mask))
    return pre_gene_set_filter_mask


def _compute_gibbs_linear_prefilter_betas(
    state,
    Y_sample_m,
    y_var,
    y_corr_sparse,
    inner_beta_kwargs_linear,
    full_scale_factors_m,
    full_mean_shifts_m,
    full_is_dense_gene_set_m,
    full_ps_m,
    full_sigma2s_m,
):
    (linear_beta_tildes_m, linear_ses_m, _, linear_p_values_m, _) = state._compute_beta_tildes(
        state.X_orig,
        Y_sample_m,
        y_var,
        state.scale_factors,
        state.mean_shifts,
        resid_correlation_matrix=y_corr_sparse,
    )
    (
        linear_uncorrected_betas_sample_m,
        _,
        linear_uncorrected_betas_mean_m,
        _,
    ) = state._calculate_non_inf_betas(
        assume_independent=True,
        initial_p=None,
        beta_tildes=linear_beta_tildes_m,
        ses=linear_ses_m,
        V=None,
        X_orig=None,
        scale_factors=full_scale_factors_m,
        mean_shifts=full_mean_shifts_m,
        is_dense_gene_set=full_is_dense_gene_set_m,
        ps=full_ps_m,
        sigma2s=full_sigma2s_m,
        return_sample=True,
        update_hyper_sigma=False,
        update_hyper_p=False,
        debug_gene_sets=state.gene_sets,
        **inner_beta_kwargs_linear
    )
    return (
        linear_uncorrected_betas_sample_m,
        linear_uncorrected_betas_mean_m,
        linear_p_values_m,
    )


def _get_gibbs_chain_batch_bounds(batch, stack_batch_size, num_chains):
    begin = batch * stack_batch_size
    end = (batch + 1) * stack_batch_size
    if end > num_chains:
        end = num_chains
    return (begin, end)


def _compute_gibbs_logistic_outputs_for_batches(
    state,
    pre_gene_set_filter_mask,
    p_sample_m,
    y_corr_sparse,
    stack_batch_size,
    num_stack_batches,
    X_hstacked,
    num_chains,
    full_betas_m_shape,
):
    (
        full_beta_tildes_m,
        full_ses_m,
        full_z_scores_m,
        full_p_values_m,
        se_inflation_factors_m,
        diverged_m,
    ) = (
        np.zeros(full_betas_m_shape),
        np.zeros(full_betas_m_shape),
        np.zeros(full_betas_m_shape),
        np.zeros(full_betas_m_shape),
        np.zeros(full_betas_m_shape),
        np.full(full_betas_m_shape, False),
    )

    for batch in range(num_stack_batches):
        (begin, end) = _get_gibbs_chain_batch_bounds(
            batch=batch,
            stack_batch_size=stack_batch_size,
            num_chains=num_chains,
        )

        log("Batch %d: chains %d-%d" % (batch, begin, end), TRACE)
        (
            batch_beta_tildes_m,
            batch_ses_m,
            batch_z_scores_m,
            batch_p_values_m,
            init_se_inflation_factors_m,
            _batch_alpha_tildes_m,
            batch_diverged_m,
        ) = _compute_gibbs_logistic_beta_tildes_batch(
            state=state,
            pre_gene_set_filter_mask=pre_gene_set_filter_mask,
            p_sample_m=p_sample_m,
            begin=begin,
            end=end,
            y_corr_sparse=y_corr_sparse,
            stack_batch_size=stack_batch_size,
            X_hstacked=X_hstacked,
        )
        se_inflation_factors_m = _apply_gibbs_logistic_batch_outputs(
            full_beta_tildes_m=full_beta_tildes_m,
            full_ses_m=full_ses_m,
            full_z_scores_m=full_z_scores_m,
            full_p_values_m=full_p_values_m,
            se_inflation_factors_m=se_inflation_factors_m,
            diverged_m=diverged_m,
            begin=begin,
            end=end,
            pre_gene_set_filter_mask=pre_gene_set_filter_mask,
            batch_beta_tildes_m=batch_beta_tildes_m,
            batch_ses_m=batch_ses_m,
            batch_z_scores_m=batch_z_scores_m,
            batch_p_values_m=batch_p_values_m,
            batch_diverged_m=batch_diverged_m,
            init_se_inflation_factors_m=init_se_inflation_factors_m,
        )

    return {
        "full_beta_tildes_m": full_beta_tildes_m,
        "full_ses_m": full_ses_m,
        "full_z_scores_m": full_z_scores_m,
        "full_p_values_m": full_p_values_m,
        "se_inflation_factors_m": se_inflation_factors_m,
        "diverged_m": diverged_m,
    }


def _apply_gibbs_logistic_batch_outputs(
    full_beta_tildes_m,
    full_ses_m,
    full_z_scores_m,
    full_p_values_m,
    se_inflation_factors_m,
    diverged_m,
    begin,
    end,
    pre_gene_set_filter_mask,
    batch_beta_tildes_m,
    batch_ses_m,
    batch_z_scores_m,
    batch_p_values_m,
    batch_diverged_m,
    init_se_inflation_factors_m,
):
    full_beta_tildes_m[begin:end,pre_gene_set_filter_mask] = batch_beta_tildes_m
    full_ses_m[begin:end,pre_gene_set_filter_mask] = batch_ses_m
    full_z_scores_m[begin:end,pre_gene_set_filter_mask] = batch_z_scores_m
    full_p_values_m[begin:end,pre_gene_set_filter_mask] = batch_p_values_m
    diverged_m[begin:end,pre_gene_set_filter_mask] = batch_diverged_m
    full_ses_m[begin:end,~pre_gene_set_filter_mask] = 100
    full_p_values_m[begin:end,~pre_gene_set_filter_mask] = 1
    if init_se_inflation_factors_m is not None:
        se_inflation_factors_m[begin:end,pre_gene_set_filter_mask] = init_se_inflation_factors_m
    else:
        se_inflation_factors_m = None
    return se_inflation_factors_m


def _log_gibbs_logistic_divergence(diverged_m, gene_sets):
    if np.sum(diverged_m) == 0:
        return
    for c in range(diverged_m.shape[0]):
        if np.sum(diverged_m[c,:] > 0):
            for p in np.nditer(np.where(diverged_m[c,:])):
                log("Chain %d: gene set %s diverged" % (c + 1, gene_sets[p]), DEBUG)


def _compute_gibbs_logistic_beta_tildes_batch(
    state,
    pre_gene_set_filter_mask,
    p_sample_m,
    begin,
    end,
    y_corr_sparse,
    stack_batch_size,
    X_hstacked,
):
    num_cur_stack = (end - begin)
    if num_cur_stack == stack_batch_size:
        cur_X_hstacked = X_hstacked
    else:
        cur_X_hstacked = sparse.hstack([state.X_orig] * num_cur_stack)

    stack_mask = np.tile(pre_gene_set_filter_mask, num_cur_stack)
    return state._compute_logistic_beta_tildes(
        state.X_orig[:,pre_gene_set_filter_mask],
        p_sample_m[begin:end,:],
        state.scale_factors[pre_gene_set_filter_mask],
        state.mean_shifts[pre_gene_set_filter_mask],
        resid_correlation_matrix=y_corr_sparse,
        X_stacked=cur_X_hstacked[:,stack_mask],
    )


def _maybe_correct_gibbs_logistic_beta_tildes(
    state,
    full_beta_tildes_m,
    full_ses_m,
    full_z_scores_m,
    full_p_values_m,
    se_inflation_factors_m,
    correct_betas_mean,
    correct_betas_var,
):
    if not (correct_betas_mean or correct_betas_var):
        return (
            full_beta_tildes_m,
            full_ses_m,
            full_z_scores_m,
            full_p_values_m,
            se_inflation_factors_m,
        )
    return state._correct_beta_tildes(
        full_beta_tildes_m,
        full_ses_m,
        se_inflation_factors_m,
        state.total_qc_metrics,
        state.total_qc_metrics_directions,
        correct_mean=correct_betas_mean,
        correct_var=correct_betas_var,
        fit=False,
    )


def _compute_gibbs_logistic_beta_tildes(
    state,
    Y_sample_m,
    y_var,
    D_sample_m,
    y_corr_sparse,
    logistic_config,
):
    (
        full_betas_m_shape,
        num_stack_batches,
        stack_batch_size,
        X_hstacked,
        inner_beta_kwargs,
        num_chains,
        gauss_seidel,
        initial_linear_filter,
        sparse_frac_gibbs,
        sparse_max_gibbs,
        correct_betas_mean,
        correct_betas_var,
    ) = (
        logistic_config["full_betas_m_shape"],
        logistic_config["num_stack_batches"],
        logistic_config["stack_batch_size"],
        logistic_config["X_hstacked"],
        logistic_config["inner_beta_kwargs"],
        logistic_config["num_chains"],
        logistic_config["gauss_seidel"],
        logistic_config["initial_linear_filter"],
        logistic_config["sparse_frac_gibbs"],
        logistic_config["sparse_max_gibbs"],
        logistic_config["correct_betas_mean"],
        logistic_config["correct_betas_var"],
    )

    inner_beta_kwargs_linear = _build_non_inf_beta_sampler_kwargs(inner_beta_kwargs)
    num_gene_sets = len(state.scale_factors)
    full_scale_factors_m = np.tile(state.scale_factors, num_chains).reshape((num_chains, num_gene_sets))
    full_mean_shifts_m = np.tile(state.mean_shifts, num_chains).reshape((num_chains, num_gene_sets))
    full_is_dense_gene_set_m = np.tile(state.is_dense_gene_set, num_chains).reshape((num_chains, num_gene_sets))

    full_ps_m = None
    if state.ps is not None:
        full_ps_m = np.tile(state.ps, num_chains).reshape((num_chains, num_gene_sets))

    full_sigma2s_m = None
    if state.sigma2s is not None:
        full_sigma2s_m = np.tile(state.sigma2s, num_chains).reshape((num_chains, num_gene_sets))

    if not gauss_seidel:
        log("Sampling Ds for logistic", TRACE)
    else:
        log("Setting Ds to mean probabilities", TRACE)
    p_sample_m = _sample_gibbs_p_targets(Y_sample_m, D_sample_m, gauss_seidel)

    pre_gene_set_filter_mask = _compute_gibbs_pre_gene_set_filter_mask(
        state=state,
        initial_linear_filter=initial_linear_filter,
        Y_sample_m=Y_sample_m,
        y_var=y_var,
        y_corr_sparse=y_corr_sparse,
        inner_beta_kwargs_linear=inner_beta_kwargs_linear,
        full_scale_factors_m=full_scale_factors_m,
        full_mean_shifts_m=full_mean_shifts_m,
        full_is_dense_gene_set_m=full_is_dense_gene_set_m,
        full_ps_m=full_ps_m,
        full_sigma2s_m=full_sigma2s_m,
        sparse_frac_gibbs=sparse_frac_gibbs,
        sparse_max_gibbs=sparse_max_gibbs,
        num_gene_sets=full_betas_m_shape[1],
    )
    logistic_outputs = _compute_gibbs_logistic_outputs_for_batches(
        state=state,
        pre_gene_set_filter_mask=pre_gene_set_filter_mask,
        p_sample_m=p_sample_m,
        y_corr_sparse=y_corr_sparse,
        stack_batch_size=stack_batch_size,
        num_stack_batches=num_stack_batches,
        X_hstacked=X_hstacked,
        num_chains=num_chains,
        full_betas_m_shape=full_betas_m_shape,
    )
    full_beta_tildes_m = logistic_outputs["full_beta_tildes_m"]
    full_ses_m = logistic_outputs["full_ses_m"]
    full_z_scores_m = logistic_outputs["full_z_scores_m"]
    full_p_values_m = logistic_outputs["full_p_values_m"]
    se_inflation_factors_m = logistic_outputs["se_inflation_factors_m"]
    diverged_m = logistic_outputs["diverged_m"]

    # Legacy outlier-z filtering was experimental and is disabled in this path.
    full_z_cur_beta_tildes_m = np.zeros(full_beta_tildes_m.shape)

    _log_gibbs_logistic_divergence(diverged_m, state.gene_sets)
    (
        full_beta_tildes_m,
        full_ses_m,
        full_z_scores_m,
        full_p_values_m,
        se_inflation_factors_m,
    ) = _maybe_correct_gibbs_logistic_beta_tildes(
        state=state,
        full_beta_tildes_m=full_beta_tildes_m,
        full_ses_m=full_ses_m,
        full_z_scores_m=full_z_scores_m,
        full_p_values_m=full_p_values_m,
        se_inflation_factors_m=se_inflation_factors_m,
        correct_betas_mean=correct_betas_mean,
        correct_betas_var=correct_betas_var,
    )

    return {
        "full_scale_factors_m": full_scale_factors_m,
        "full_mean_shifts_m": full_mean_shifts_m,
        "full_is_dense_gene_set_m": full_is_dense_gene_set_m,
        "full_ps_m": full_ps_m,
        "full_sigma2s_m": full_sigma2s_m,
        "p_sample_m": p_sample_m,
        "pre_gene_set_filter_mask": pre_gene_set_filter_mask,
        "full_z_cur_beta_tildes_m": full_z_cur_beta_tildes_m,
        "full_beta_tildes_m": full_beta_tildes_m,
        "full_ses_m": full_ses_m,
        "full_z_scores_m": full_z_scores_m,
        "full_p_values_m": full_p_values_m,
    }


def _build_post_stall_snapshot_arrays(
    sum_betas_m,
    sum_betas2_m,
    sum_Ds_m,
    post_stall_beta_indices,
    post_stall_gene_indices,
    num_chains,
):
    if post_stall_beta_indices.size > 0:
        post_stall_beta_sum_m = copy.copy(sum_betas_m[:,post_stall_beta_indices])
        post_stall_beta_sum2_m = copy.copy(sum_betas2_m[:,post_stall_beta_indices])
    else:
        post_stall_beta_sum_m = np.zeros((num_chains, 0))
        post_stall_beta_sum2_m = np.zeros((num_chains, 0))
    if post_stall_gene_indices.size > 0:
        post_stall_D_sum_m = copy.copy(sum_Ds_m[:,post_stall_gene_indices])
    else:
        post_stall_D_sum_m = np.zeros((num_chains, 0))
    return (post_stall_beta_sum_m, post_stall_beta_sum2_m, post_stall_D_sum_m)


def _update_post_stall_best_histories(
    post_stall_best_beta_rhat_history,
    post_stall_best_D_mcse_history,
    beta_rhat_q_post,
    D_mcse_q,
):
    post_best_beta_rhat_q = beta_rhat_q_post if len(post_stall_best_beta_rhat_history) == 0 else min(post_stall_best_beta_rhat_history[-1], beta_rhat_q_post)
    post_best_D_mcse_q = D_mcse_q if len(post_stall_best_D_mcse_history) == 0 else min(post_stall_best_D_mcse_history[-1], D_mcse_q)
    post_stall_best_beta_rhat_history.append(post_best_beta_rhat_q)
    post_stall_best_D_mcse_history.append(post_best_D_mcse_q)


def _unpack_post_burn_stall_tracking_config(stall_tracking_config):
    return (
        stall_tracking_config["active_beta_top_k"],
        stall_tracking_config["post_stall_best_beta_rhat_history"],
        stall_tracking_config["post_stall_best_D_mcse_history"],
        stall_tracking_config["post_stall_snapshots"],
        stall_tracking_config["post_stall_beta_indices"],
        stall_tracking_config["post_stall_gene_indices"],
        stall_tracking_config["stop_mcse_quantile"],
        stall_tracking_config["stall_window"],
        stall_tracking_config["stall_min_post_burn_in"],
        stall_tracking_config["min_num_post_burn_in_for_epoch"],
        stall_tracking_config["stall_delta_rhat"],
        stall_tracking_config["stall_delta_mcse"],
        stall_tracking_config["stall_recent_window"],
        stall_tracking_config["stall_recent_eps"],
    )


def _resolve_post_stall_indices(
    post_stall_beta_indices,
    post_stall_gene_indices,
    active_beta_mask,
    beta_mean_v,
    active_beta_top_k,
    top_gene_indices,
):
    if post_stall_beta_indices is None:
        post_stall_beta_indices = _prepare_stall_indices(active_beta_mask, beta_mean_v, active_beta_top_k)
    if post_stall_gene_indices is None:
        post_stall_gene_indices = copy.copy(top_gene_indices)
    return (post_stall_beta_indices, post_stall_gene_indices)


def _update_post_burn_stall_tracking(
    sum_betas_m,
    sum_betas2_m,
    sum_Ds_m,
    num_sum_Y_m,
    num_chains,
    beta_rhat_q_post,
    D_mcse_q,
    active_beta_mask,
    beta_mean_v,
    top_gene_indices,
    num_post_burn_beta,
    stall_tracking_config,
):
    (
        active_beta_top_k,
        post_stall_best_beta_rhat_history,
        post_stall_best_D_mcse_history,
        post_stall_snapshots,
        post_stall_beta_indices,
        post_stall_gene_indices,
        stop_mcse_quantile,
        stall_window,
        stall_min_post_burn_in,
        min_num_post_burn_in_for_epoch,
        stall_delta_rhat,
        stall_delta_mcse,
        stall_recent_window,
        stall_recent_eps,
    ) = _unpack_post_burn_stall_tracking_config(stall_tracking_config)

    _update_post_stall_best_histories(
        post_stall_best_beta_rhat_history=post_stall_best_beta_rhat_history,
        post_stall_best_D_mcse_history=post_stall_best_D_mcse_history,
        beta_rhat_q_post=beta_rhat_q_post,
        D_mcse_q=D_mcse_q,
    )
    (post_stall_beta_indices, post_stall_gene_indices) = _resolve_post_stall_indices(
        post_stall_beta_indices=post_stall_beta_indices,
        post_stall_gene_indices=post_stall_gene_indices,
        active_beta_mask=active_beta_mask,
        beta_mean_v=beta_mean_v,
        active_beta_top_k=active_beta_top_k,
        top_gene_indices=top_gene_indices,
    )

    (post_stall_beta_sum_m, post_stall_beta_sum2_m, post_stall_D_sum_m) = _build_post_stall_snapshot_arrays(
        sum_betas_m=sum_betas_m,
        sum_betas2_m=sum_betas2_m,
        sum_Ds_m=sum_Ds_m,
        post_stall_beta_indices=post_stall_beta_indices,
        post_stall_gene_indices=post_stall_gene_indices,
        num_chains=num_chains,
    )
    num_post_burn_D = int(np.min(num_sum_Y_m))
    post_stall_snapshots.append((num_post_burn_beta, post_stall_beta_sum_m, post_stall_beta_sum2_m, num_post_burn_D, post_stall_D_sum_m))

    _trim_post_stall_history_windows(
        post_stall_best_beta_rhat_history=post_stall_best_beta_rhat_history,
        post_stall_best_D_mcse_history=post_stall_best_D_mcse_history,
        post_stall_snapshots=post_stall_snapshots,
        stall_window=stall_window,
        stall_recent_window=stall_recent_window,
    )

    (
        post_stall_plateau,
        post_stall_recent_worse,
        post_stall_recent_beta_rhat_q,
        post_stall_recent_D_mcse_q,
    ) = _evaluate_post_stall_status(
        num_post_burn_beta,
        num_post_burn_D,
        post_stall_beta_sum_m,
        post_stall_beta_sum2_m,
        post_stall_D_sum_m,
        post_stall_beta_indices,
        post_stall_gene_indices,
        beta_rhat_q_post,
        D_mcse_q,
        stop_mcse_quantile,
        stall_window,
        stall_min_post_burn_in,
        min_num_post_burn_in_for_epoch,
        stall_delta_rhat,
        stall_delta_mcse,
        stall_recent_window,
        stall_recent_eps,
        post_stall_best_beta_rhat_history,
        post_stall_best_D_mcse_history,
        post_stall_snapshots,
        num_chains,
    )

    return {
        "post_stall_beta_indices": post_stall_beta_indices,
        "post_stall_gene_indices": post_stall_gene_indices,
        "num_post_burn_D": num_post_burn_D,
        "post_stall_plateau": post_stall_plateau,
        "post_stall_recent_worse": post_stall_recent_worse,
        "post_stall_recent_beta_rhat_q": post_stall_recent_beta_rhat_q,
        "post_stall_recent_D_mcse_q": post_stall_recent_D_mcse_q,
        "post_stall_detected": post_stall_plateau or post_stall_recent_worse,
    }


def _trim_post_stall_history_windows(
    post_stall_best_beta_rhat_history,
    post_stall_best_D_mcse_history,
    post_stall_snapshots,
    stall_window,
    stall_recent_window,
):
    max_stall_history_len = max(stall_window + 2, stall_recent_window + 2, 10)
    _trim_stall_histories(
        post_stall_best_beta_rhat_history,
        post_stall_best_D_mcse_history,
        post_stall_snapshots,
        max_stall_history_len,
    )


def _decide_gibbs_post_burn_action(
    precision_achieved,
    post_stall_detected,
    post_burn_action_config,
):
    num_attempts = post_burn_action_config["num_attempts"]
    max_num_attempt_restarts = post_burn_action_config["max_num_attempt_restarts"]
    epoch_iter_num = post_burn_action_config["epoch_iter_num"]
    total_iter_num = post_burn_action_config["total_iter_num"]
    post_stall_plateau = post_burn_action_config["post_stall_plateau"]
    post_stall_recent_worse = post_burn_action_config["post_stall_recent_worse"]
    beta_rhat_q_post = post_burn_action_config["beta_rhat_q_post"]
    D_mcse_q = post_burn_action_config["D_mcse_q"]
    post_stall_recent_beta_rhat_q = post_burn_action_config["post_stall_recent_beta_rhat_q"]
    post_stall_recent_D_mcse_q = post_burn_action_config["post_stall_recent_D_mcse_q"]

    decision = {
        "done": False,
        "stop_due_to_precision": False,
        "restart_due_to_stall": False,
        "stop_due_to_stall": False,
    }

    if precision_achieved:
        decision["done"] = True
        decision["stop_due_to_precision"] = True
        log("Desired Gibbs precision achieved; stopping sampling", INFO)
        return decision

    if not post_stall_detected:
        return decision

    if num_attempts < max_num_attempt_restarts:
        decision["done"] = True
        decision["restart_due_to_stall"] = True
        # Keep and aggregate this epoch's post-burn samples, then continue
        # with a new epoch to add more effective chain means.
        log(
            "Restarting Gibbs epoch due to post-burn stall at iter %d (global %d) because precision is not yet met (plateau=%s, recent_worse=%s, beta_Rhat_q=%.4g, D_mcse_q=%.4g, recent_beta_Rhat_q=%s, recent_D_mcse_q=%s); aggregating current epoch samples before restart"
            % (
                epoch_iter_num,
                total_iter_num,
                str(post_stall_plateau),
                str(post_stall_recent_worse),
                beta_rhat_q_post,
                D_mcse_q,
                ("%.4g" % post_stall_recent_beta_rhat_q) if np.isfinite(post_stall_recent_beta_rhat_q) else "NA",
                ("%.4g" % post_stall_recent_D_mcse_q) if np.isfinite(post_stall_recent_D_mcse_q) else "NA",
            ),
            INFO,
        )
        return decision

    decision["done"] = True
    decision["stop_due_to_stall"] = True
    log(
        "Post-burn stall detected at iter %d (global %d) and precision is not yet met, but no restart attempts remain; stopping this epoch (beta_Rhat_q=%.4g, D_mcse_q=%.4g)"
        % (epoch_iter_num, total_iter_num, beta_rhat_q_post, D_mcse_q),
        INFO,
    )
    return decision


def _should_run_gibbs_post_burn_diagnostics(
    epoch_sums,
    diag_every,
    epoch_iter_num,
    epoch_max_num_iter,
):
    return (
        np.all(epoch_sums["num_sum_Y_m"] > 1)
        and np.all(epoch_sums["num_sum_beta_m"] > 1)
        and (epoch_iter_num % diag_every == 0 or epoch_iter_num == epoch_max_num_iter)
    )


GIBBS_POST_BURN_CONTROL_KEYS = (
    "stop_pass_streak",
    "post_stall_beta_indices",
    "post_stall_gene_indices",
    "betas_sem2_v",
    "sem2_v",
    "stop_due_to_precision",
    "restart_due_to_stall",
    "stop_due_to_stall",
)


GIBBS_BURN_IN_CONTROL_KEYS = (
    "in_burn_in",
    "burn_in_pass_streak",
    "stop_pass_streak",
    "prev_Ys_m",
    "burn_stall_beta_indices",
    "R_beta_v",
)


def _apply_gibbs_control_update(epoch_control, control_update, control_keys):
    for key in control_keys:
        epoch_control[key] = control_update[key]


def _build_gibbs_burn_in_control_update(
    in_burn_in,
    burn_in_pass_streak,
    stop_pass_streak,
    prev_Ys_m,
    burn_stall_beta_indices,
    R_beta_v,
):
    return {
        "in_burn_in": in_burn_in,
        "burn_in_pass_streak": burn_in_pass_streak,
        "stop_pass_streak": stop_pass_streak,
        "prev_Ys_m": prev_Ys_m,
        "burn_stall_beta_indices": burn_stall_beta_indices,
        "R_beta_v": R_beta_v,
    }


def _build_gibbs_burn_in_control_from_epoch(epoch_control):
    return _build_gibbs_burn_in_control_update(
        in_burn_in=epoch_control["in_burn_in"],
        burn_in_pass_streak=epoch_control["burn_in_pass_streak"],
        stop_pass_streak=epoch_control["stop_pass_streak"],
        prev_Ys_m=epoch_control["prev_Ys_m"],
        burn_stall_beta_indices=epoch_control["burn_stall_beta_indices"],
        R_beta_v=epoch_control["R_beta_v"],
    )


def _apply_gibbs_burn_in_control_update(epoch_control, burn_in_update):
    _apply_gibbs_control_update(
        epoch_control=epoch_control,
        control_update=burn_in_update,
        control_keys=GIBBS_BURN_IN_CONTROL_KEYS,
    )


def _apply_gibbs_post_burn_control_update(epoch_control, post_burn_update):
    _apply_gibbs_control_update(
        epoch_control=epoch_control,
        control_update=post_burn_update,
        control_keys=GIBBS_POST_BURN_CONTROL_KEYS,
    )


def _build_gibbs_post_burn_control_update(
    stop_pass_streak,
    post_stall_beta_indices,
    post_stall_gene_indices,
    betas_sem2_v,
    sem2_v,
    done,
    stop_due_to_precision,
    restart_due_to_stall,
    stop_due_to_stall,
):
    return {
        "stop_pass_streak": stop_pass_streak,
        "post_stall_beta_indices": post_stall_beta_indices,
        "post_stall_gene_indices": post_stall_gene_indices,
        "betas_sem2_v": betas_sem2_v,
        "sem2_v": sem2_v,
        "done": done,
        "stop_due_to_precision": stop_due_to_precision,
        "restart_due_to_stall": restart_due_to_stall,
        "stop_due_to_stall": stop_due_to_stall,
    }


def _build_gibbs_post_burn_control_from_epoch(epoch_control, done=False):
    return _build_gibbs_post_burn_control_update(
        stop_pass_streak=epoch_control["stop_pass_streak"],
        post_stall_beta_indices=epoch_control["post_stall_beta_indices"],
        post_stall_gene_indices=epoch_control["post_stall_gene_indices"],
        betas_sem2_v=epoch_control["betas_sem2_v"],
        sem2_v=epoch_control["sem2_v"],
        done=done,
        stop_due_to_precision=epoch_control["stop_due_to_precision"],
        restart_due_to_stall=epoch_control["restart_due_to_stall"],
        stop_due_to_stall=epoch_control["stop_due_to_stall"],
    )


def _merge_gibbs_post_burn_control_updates(base_update, diag_update):
    return _build_gibbs_post_burn_control_update(
        stop_pass_streak=diag_update["stop_pass_streak"],
        post_stall_beta_indices=diag_update["post_stall_beta_indices"],
        post_stall_gene_indices=diag_update["post_stall_gene_indices"],
        betas_sem2_v=diag_update["betas_sem2_v"],
        sem2_v=diag_update["sem2_v"],
        done=(base_update["done"] or diag_update["done"]),
        stop_due_to_precision=(base_update["stop_due_to_precision"] or diag_update["stop_due_to_precision"]),
        restart_due_to_stall=(base_update["restart_due_to_stall"] or diag_update["restart_due_to_stall"]),
        stop_due_to_stall=(base_update["stop_due_to_stall"] or diag_update["stop_due_to_stall"]),
    )


def _log_gibbs_post_burn_diagnostics(
    epoch_iter_num,
    total_iter_num,
    stop_mcse_quantile,
    beta_rhat_q_post,
    beta_ratio_q,
    max_rel_mcse_beta,
    beta_rel_mcse_denom_floor,
    top_gene_k,
    stop_min_gene_d,
    num_monitored_genes,
    num_eligible_genes,
    D_mcse_q,
    max_abs_mcse_d,
    num_active_betas,
    num_full_gene_sets,
    num_chains_effective_for_diag,
    burn_in_pass_streak,
    burn_in_patience,
    stop_pass_streak,
    stop_patience,
):
    if stop_min_gene_d is None:
        log(
            "Gibbs iteration %d (global %d): beta_Rhat_q(%.2f)=%.4g; beta_rel_mcse_q(%.2f)=%.4g (threshold=%.4g, denom_floor=%.4g); D_mcse_q(%.2f, topK=%d)=%.4g (threshold=%.4g); active_betas=%d/%d; eff_chains=%d; burn_streak=%d/%d; stop_streak=%d/%d"
            % (
                epoch_iter_num,
                total_iter_num,
                stop_mcse_quantile,
                beta_rhat_q_post,
                stop_mcse_quantile,
                beta_ratio_q,
                max_rel_mcse_beta,
                beta_rel_mcse_denom_floor,
                stop_mcse_quantile,
                top_gene_k,
                D_mcse_q,
                max_abs_mcse_d,
                num_active_betas,
                num_full_gene_sets,
                num_chains_effective_for_diag,
                burn_in_pass_streak,
                burn_in_patience,
                stop_pass_streak,
                stop_patience,
            ),
            INFO,
        )
        return

    log(
        "Gibbs iteration %d (global %d): beta_Rhat_q(%.2f)=%.4g; beta_rel_mcse_q(%.2f)=%.4g (threshold=%.4g, denom_floor=%.4g); D_mcse_q(%.2f, topK=%d, minD=%.4g, monitored=%d, eligible=%d)=%.4g (threshold=%.4g); active_betas=%d/%d; eff_chains=%d; burn_streak=%d/%d; stop_streak=%d/%d"
        % (
            epoch_iter_num,
            total_iter_num,
            stop_mcse_quantile,
            beta_rhat_q_post,
            stop_mcse_quantile,
            beta_ratio_q,
            max_rel_mcse_beta,
            beta_rel_mcse_denom_floor,
            stop_mcse_quantile,
            top_gene_k,
            stop_min_gene_d,
            num_monitored_genes,
            num_eligible_genes,
            D_mcse_q,
            max_abs_mcse_d,
            num_active_betas,
            num_full_gene_sets,
            num_chains_effective_for_diag,
            burn_in_pass_streak,
            burn_in_patience,
            stop_pass_streak,
            stop_patience,
        ),
        INFO,
    )


def _build_post_burn_stall_tracking_config(
    diag_config,
    epoch_control,
    min_num_post_burn_in_for_epoch,
):
    return {
        "active_beta_top_k": diag_config["active_beta_top_k"],
        "post_stall_best_beta_rhat_history": epoch_control["post_stall_best_beta_rhat_history"],
        "post_stall_best_D_mcse_history": epoch_control["post_stall_best_D_mcse_history"],
        "post_stall_snapshots": epoch_control["post_stall_snapshots"],
        "post_stall_beta_indices": epoch_control["post_stall_beta_indices"],
        "post_stall_gene_indices": epoch_control["post_stall_gene_indices"],
        "stop_mcse_quantile": diag_config["stop_mcse_quantile"],
        "stall_window": diag_config["stall_window"],
        "stall_min_post_burn_in": diag_config["stall_min_post_burn_in"],
        "min_num_post_burn_in_for_epoch": min_num_post_burn_in_for_epoch,
        "stall_delta_rhat": diag_config["stall_delta_rhat"],
        "stall_delta_mcse": diag_config["stall_delta_mcse"],
        "stall_recent_window": diag_config["stall_recent_window"],
        "stall_recent_eps": diag_config["stall_recent_eps"],
    }


def _build_post_burn_action_config(
    run_state,
    iter_state,
    post_stall_update,
    beta_rhat_q_post,
    D_mcse_q,
):
    return {
        "num_attempts": run_state.num_attempts,
        "max_num_attempt_restarts": run_state.max_num_attempt_restarts,
        "epoch_iter_num": iter_state["epoch_iter_num"],
        "total_iter_num": iter_state["total_iter_num"],
        "post_stall_plateau": post_stall_update["post_stall_plateau"],
        "post_stall_recent_worse": post_stall_update["post_stall_recent_worse"],
        "beta_rhat_q_post": beta_rhat_q_post,
        "D_mcse_q": D_mcse_q,
        "post_stall_recent_beta_rhat_q": post_stall_update["post_stall_recent_beta_rhat_q"],
        "post_stall_recent_D_mcse_q": post_stall_update["post_stall_recent_D_mcse_q"],
    }


# ========================= Outer Gibbs Post-burn Diagnostics =========================
def _compute_gibbs_post_burn_diag_metrics(
    min_num_post_burn_in_for_epoch,
    diag_config,
    epoch_sums,
    epoch_control,
):
    (
        epoch_aggregates,
        sum_betas_m,
        sum_betas2_m,
        num_sum_beta_m,
        sum_Ds_m,
        num_sum_Y_m,
    ) = (
        epoch_sums["epoch_aggregates"],
        epoch_sums["sum_betas_m"],
        epoch_sums["sum_betas2_m"],
        epoch_sums["num_sum_beta_m"],
        epoch_sums["sum_Ds_m"],
        epoch_sums["num_sum_Y_m"],
    )
    num_chains = diag_config["num_chains"]
    stop_mcse_quantile = diag_config["stop_mcse_quantile"]

    # For stopping diagnostics, aggregate previous completed epochs with the
    # current in-progress epoch so MCSE aligns with final reported MCSE.
    (
        diag_sum_betas_m,
        diag_sum_betas2_m,
        diag_num_sum_beta_m,
        diag_sum_Ds_m,
        diag_num_sum_Y_m,
    ) = _build_gibbs_diag_sums(
        epoch_aggregates,
        sum_betas_m,
        sum_betas2_m,
        num_sum_beta_m,
        sum_Ds_m,
        num_sum_Y_m,
    )

    num_chains_effective_for_diag = diag_sum_betas_m.shape[0]
    beta_diag = _compute_post_burn_beta_diagnostics(
        diag_sum_betas_m,
        diag_sum_betas2_m,
        diag_num_sum_beta_m,
        num_chains_effective_for_diag,
        diag_config["active_beta_top_k"],
        diag_config["active_beta_min_abs"],
        stop_mcse_quantile,
        diag_config["beta_rel_mcse_denom_floor"],
    )
    gene_diag = _compute_post_burn_gene_diagnostics(
        diag_sum_Ds_m,
        diag_num_sum_Y_m,
        num_chains_effective_for_diag,
        diag_config["stop_top_gene_k"],
        diag_config["stop_min_gene_d"],
        stop_mcse_quantile,
    )

    post_stall_update = _update_post_burn_stall_tracking(
        sum_betas_m,
        sum_betas2_m,
        sum_Ds_m,
        num_sum_Y_m,
        num_chains,
        beta_diag["beta_rhat_q_post"],
        gene_diag["D_mcse_q"],
        beta_diag["active_beta_mask"],
        beta_diag["beta_mean_v"],
        gene_diag["top_gene_indices"],
        beta_diag["num_post_burn_beta"],
        stall_tracking_config=_build_post_burn_stall_tracking_config(
            diag_config=diag_config,
            epoch_control=epoch_control,
            min_num_post_burn_in_for_epoch=min_num_post_burn_in_for_epoch,
        ),
    )

    return {
        "num_chains_effective_for_diag": num_chains_effective_for_diag,
        "num_active_betas": beta_diag["num_active_betas"],
        "beta_mcse_v": beta_diag["beta_mcse_v"],
        "beta_rhat_q_post": beta_diag["beta_rhat_q_post"],
        "beta_ratio_q": beta_diag["beta_ratio_q"],
        "num_post_burn_beta": beta_diag["num_post_burn_beta"],
        "D_mcse_v": gene_diag["D_mcse_v"],
        "top_gene_k": gene_diag["top_gene_k"],
        "num_monitored_genes": gene_diag["num_monitored_genes"],
        "num_eligible_genes": gene_diag["num_eligible_genes"],
        "D_mcse_q": gene_diag["D_mcse_q"],
        "post_stall_update": post_stall_update,
    }


def _update_gibbs_post_burn_precision_streak(
    stop_pass_streak,
    beta_ratio_q,
    D_mcse_q,
    max_rel_mcse_beta,
    max_abs_mcse_d,
    num_post_burn_D,
    min_num_post_burn_in_for_epoch,
):
    min_post_burn_reached = num_post_burn_D >= min_num_post_burn_in_for_epoch
    precision_pass = beta_ratio_q <= max_rel_mcse_beta and D_mcse_q <= max_abs_mcse_d
    if precision_pass and min_post_burn_reached:
        stop_pass_streak += 1
    else:
        stop_pass_streak = 0
    return (stop_pass_streak, min_post_burn_reached)


def _evaluate_gibbs_post_burn_diagnostics_and_decision(
    min_num_post_burn_in_for_epoch,
    diag_config,
    iter_state,
    epoch_sums,
    epoch_control,
    run_state,
):
    epoch_iter_num = iter_state["epoch_iter_num"]
    total_iter_num = iter_state["total_iter_num"]

    stop_pass_streak = epoch_control["stop_pass_streak"]
    burn_in_pass_streak = epoch_control["burn_in_pass_streak"]

    diag_metrics = _compute_gibbs_post_burn_diag_metrics(
        min_num_post_burn_in_for_epoch=min_num_post_burn_in_for_epoch,
        diag_config=diag_config,
        epoch_sums=epoch_sums,
        epoch_control=epoch_control,
    )
    post_stall_update = diag_metrics["post_stall_update"]

    num_post_burn_D = post_stall_update["num_post_burn_D"]
    post_stall_detected = post_stall_update["post_stall_detected"]

    stop_pass_streak, min_post_burn_reached = _update_gibbs_post_burn_precision_streak(
        stop_pass_streak=stop_pass_streak,
        beta_ratio_q=diag_metrics["beta_ratio_q"],
        D_mcse_q=diag_metrics["D_mcse_q"],
        max_rel_mcse_beta=diag_config["max_rel_mcse_beta"],
        max_abs_mcse_d=diag_config["max_abs_mcse_d"],
        num_post_burn_D=num_post_burn_D,
        min_num_post_burn_in_for_epoch=min_num_post_burn_in_for_epoch,
    )

    _log_gibbs_post_burn_diagnostics(
        epoch_iter_num=epoch_iter_num,
        total_iter_num=total_iter_num,
        stop_mcse_quantile=diag_config["stop_mcse_quantile"],
        beta_rhat_q_post=diag_metrics["beta_rhat_q_post"],
        beta_ratio_q=diag_metrics["beta_ratio_q"],
        max_rel_mcse_beta=diag_config["max_rel_mcse_beta"],
        beta_rel_mcse_denom_floor=diag_config["beta_rel_mcse_denom_floor"],
        top_gene_k=diag_metrics["top_gene_k"],
        stop_min_gene_d=diag_config["stop_min_gene_d"],
        num_monitored_genes=diag_metrics["num_monitored_genes"],
        num_eligible_genes=diag_metrics["num_eligible_genes"],
        D_mcse_q=diag_metrics["D_mcse_q"],
        max_abs_mcse_d=diag_config["max_abs_mcse_d"],
        num_active_betas=diag_metrics["num_active_betas"],
        num_full_gene_sets=diag_config["num_full_gene_sets"],
        num_chains_effective_for_diag=diag_metrics["num_chains_effective_for_diag"],
        burn_in_pass_streak=burn_in_pass_streak,
        burn_in_patience=diag_config["burn_in_patience"],
        stop_pass_streak=stop_pass_streak,
        stop_patience=diag_config["stop_patience"],
    )

    precision_achieved = min_post_burn_reached and stop_pass_streak >= diag_config["stop_patience"]
    decision = _decide_gibbs_post_burn_action(
        precision_achieved=precision_achieved,
        post_stall_detected=post_stall_detected,
        post_burn_action_config=_build_post_burn_action_config(
            run_state=run_state,
            iter_state=iter_state,
            post_stall_update=post_stall_update,
            beta_rhat_q_post=diag_metrics["beta_rhat_q_post"],
            D_mcse_q=diag_metrics["D_mcse_q"],
        ),
    )

    return _build_gibbs_post_burn_control_update(
        stop_pass_streak=stop_pass_streak,
        post_stall_beta_indices=post_stall_update["post_stall_beta_indices"],
        post_stall_gene_indices=post_stall_update["post_stall_gene_indices"],
        betas_sem2_v=np.square(diag_metrics["beta_mcse_v"]),
        sem2_v=np.square(diag_metrics["D_mcse_v"]),
        done=decision["done"],
        stop_due_to_precision=decision["stop_due_to_precision"],
        restart_due_to_stall=decision["restart_due_to_stall"],
        stop_due_to_stall=decision["stop_due_to_stall"],
    )


def _run_optional_gibbs_post_burn_diagnostics(
    min_num_post_burn_in_for_epoch,
    diag_every,
    epoch_iter_num,
    epoch_max_num_iter,
    post_burn_diag_config,
    iter_state,
    epoch_sums,
    epoch_control,
    run_state,
    post_burn_update,
):
    if not _should_run_gibbs_post_burn_diagnostics(
        epoch_sums=epoch_sums,
        diag_every=diag_every,
        epoch_iter_num=epoch_iter_num,
        epoch_max_num_iter=epoch_max_num_iter,
    ):
        return post_burn_update

    return _run_due_gibbs_post_burn_diagnostics(
        min_num_post_burn_in_for_epoch=min_num_post_burn_in_for_epoch,
        post_burn_diag_config=post_burn_diag_config,
        iter_state=iter_state,
        epoch_sums=epoch_sums,
        epoch_control=epoch_control,
        run_state=run_state,
        post_burn_update=post_burn_update,
    )


def _run_due_gibbs_post_burn_diagnostics(
    min_num_post_burn_in_for_epoch,
    post_burn_diag_config,
    iter_state,
    epoch_sums,
    epoch_control,
    run_state,
    post_burn_update,
):
    post_burn_diag = _evaluate_gibbs_post_burn_diagnostics_and_decision(
        min_num_post_burn_in_for_epoch=min_num_post_burn_in_for_epoch,
        diag_config=post_burn_diag_config,
        iter_state=iter_state,
        epoch_sums=epoch_sums,
        epoch_control=epoch_control,
        run_state=run_state,
    )

    return _merge_gibbs_post_burn_control_updates(
        base_update=post_burn_update,
        diag_update=post_burn_diag,
    )


def _maybe_end_gibbs_epoch_for_post_burn_cap(
    done,
    num_sum_Y_m,
    max_num_post_burn_in_for_epoch,
    epoch_iter_num,
    total_iter_num,
):
    num_post_burn_samples_now = int(np.min(num_sum_Y_m))
    if (not done) and num_post_burn_samples_now >= max_num_post_burn_in_for_epoch:
        done = True
        log(
            "Ending Gibbs epoch at iter %d (global %d) after %d post-burn samples (per-epoch max post-burn reached)"
            % (epoch_iter_num, total_iter_num, num_post_burn_samples_now),
            INFO,
        )
    return done


def _update_gibbs_post_burn_state(
    state,
    max_num_post_burn_in_for_epoch,
    min_num_post_burn_in_for_epoch,
    epoch_max_num_iter,
    diag_every,
    post_burn_diag_config,
    iter_state,
    epoch_sums,
    epoch_priors,
    epoch_control,
    run_state,
    log_bf_m,
    log_bf_raw_m,
    full_betas_mean_m,
    full_postp_sample_m,
):
    in_burn_in = epoch_control["in_burn_in"]
    post_burn_update = _build_gibbs_post_burn_control_from_epoch(
        epoch_control,
        done=False,
    )
    epoch_iter_num = iter_state["epoch_iter_num"]
    total_iter_num = iter_state["total_iter_num"]

    priors_for_Y_m = epoch_priors["priors_for_Y_m"]
    priors_missing_mean_m = epoch_priors["priors_missing_mean_m"]

    if in_burn_in:
        return post_burn_update

    return _advance_gibbs_post_burn_state(
        state=state,
        priors_for_Y_m=priors_for_Y_m,
        log_bf_m=log_bf_m,
        log_bf_raw_m=log_bf_raw_m,
        full_betas_mean_m=full_betas_mean_m,
        full_postp_sample_m=full_postp_sample_m,
        priors_missing_mean_m=priors_missing_mean_m,
        epoch_sums=epoch_sums,
        min_num_post_burn_in_for_epoch=min_num_post_burn_in_for_epoch,
        diag_every=diag_every,
        epoch_iter_num=epoch_iter_num,
        epoch_max_num_iter=epoch_max_num_iter,
        post_burn_diag_config=post_burn_diag_config,
        iter_state=iter_state,
        epoch_control=epoch_control,
        run_state=run_state,
        post_burn_update=post_burn_update,
        max_num_post_burn_in_for_epoch=max_num_post_burn_in_for_epoch,
        total_iter_num=total_iter_num,
    )


def _advance_gibbs_post_burn_state(
    state,
    priors_for_Y_m,
    log_bf_m,
    log_bf_raw_m,
    full_betas_mean_m,
    full_postp_sample_m,
    priors_missing_mean_m,
    epoch_sums,
    min_num_post_burn_in_for_epoch,
    diag_every,
    epoch_iter_num,
    epoch_max_num_iter,
    post_burn_diag_config,
    iter_state,
    epoch_control,
    run_state,
    post_burn_update,
    max_num_post_burn_in_for_epoch,
    total_iter_num,
):
    accumulation_payload = _build_gibbs_post_burn_accumulation_payload(
        iter_state=iter_state,
        full_betas_mean_m=full_betas_mean_m,
        full_postp_sample_m=full_postp_sample_m,
        priors_for_Y_m=priors_for_Y_m,
        priors_missing_mean_m=priors_missing_mean_m,
        log_bf_m=log_bf_m,
        log_bf_raw_m=log_bf_raw_m,
    )
    _accumulate_gibbs_post_burn_iteration(
        state=state,
        accumulation_payload=accumulation_payload,
        epoch_sums=epoch_sums,
    )

    post_burn_diag_update = _run_optional_gibbs_post_burn_diagnostics(
        min_num_post_burn_in_for_epoch=min_num_post_burn_in_for_epoch,
        diag_every=diag_every,
        epoch_iter_num=epoch_iter_num,
        epoch_max_num_iter=epoch_max_num_iter,
        post_burn_diag_config=post_burn_diag_config,
        iter_state=iter_state,
        epoch_sums=epoch_sums,
        epoch_control=epoch_control,
        run_state=run_state,
        post_burn_update=post_burn_update,
    )
    post_burn_diag_update["done"] = _maybe_end_gibbs_epoch_for_post_burn_cap(
        done=post_burn_diag_update["done"],
        num_sum_Y_m=epoch_sums["num_sum_Y_m"],
        max_num_post_burn_in_for_epoch=max_num_post_burn_in_for_epoch,
        epoch_iter_num=epoch_iter_num,
        total_iter_num=total_iter_num,
    )
    return post_burn_diag_update


def _build_gibbs_iteration_progress_context(progress_runtime_config, iteration_update):
    return {
        "trace_chain_offset": progress_runtime_config.trace_chain_offset,
        "epoch_total_iter_offset": progress_runtime_config.epoch_total_iter_offset,
        "epoch_max_num_iter": progress_runtime_config.epoch_max_num_iter,
        "max_num_burn_in_for_epoch": progress_runtime_config.max_num_burn_in_for_epoch,
        "min_num_iter_for_epoch": progress_runtime_config.min_num_iter_for_epoch,
        "min_num_burn_in_for_epoch": progress_runtime_config.min_num_burn_in_for_epoch,
        "max_num_post_burn_in_for_epoch": progress_runtime_config.max_num_post_burn_in_for_epoch,
        "min_num_post_burn_in_for_epoch": progress_runtime_config.min_num_post_burn_in_for_epoch,
        "post_burn_reset_arrays": progress_runtime_config.post_burn_reset_arrays,
        "post_burn_reset_missing_arrays": progress_runtime_config.post_burn_reset_missing_arrays,
        "iteration_progress_config": progress_runtime_config.iteration_progress_config,
        "full_betas_sample_m": iteration_update["full_betas_sample_m"],
        "full_postp_sample_m": iteration_update["full_postp_sample_m"],
        "full_betas_mean_m": iteration_update["full_betas_mean_m"],
        "full_postp_mean_m": iteration_update["full_postp_mean_m"],
    }


def _write_gibbs_iteration_gene_set_stats_trace(
    gene_set_stats_trace_fh,
    iteration_num,
    trace_chain_offset,
    state,
    iter_state,
    full_betas_mean_m,
    full_betas_sample_m,
    full_postp_mean_m,
    full_postp_sample_m,
    R_beta_v,
    betas_sem2_v,
    use_mean_betas,
):
    _maybe_write_gibbs_gene_set_stats_trace(
        gene_set_stats_trace_fh,
        iteration_num,
        trace_chain_offset,
        state.gene_sets,
        state.scale_factors,
        iter_state["full_beta_tildes_m"],
        iter_state["full_p_values_m"],
        iter_state["full_z_scores_m"],
        iter_state["full_ses_m"],
        iter_state["uncorrected_betas_mean_m"],
        iter_state["uncorrected_betas_sample_m"],
        full_betas_mean_m,
        full_betas_sample_m,
        full_postp_mean_m,
        full_postp_sample_m,
        iter_state["full_z_cur_beta_tildes_m"],
        R_beta_v,
        betas_sem2_v,
        use_mean_betas,
    )


def _advance_gibbs_iteration_progress(progress_update_context):
    state = progress_update_context["state"]
    epoch_control = progress_update_context["epoch_control"]
    run_state = progress_update_context["run_state"]
    progress_runtime_config = progress_update_context["progress_runtime_config"]
    iter_state = progress_update_context["iter_state"]
    epoch_sums = progress_update_context["epoch_sums"]
    epoch_priors = progress_update_context["epoch_priors"]
    epoch_runtime = progress_update_context["epoch_runtime"]
    iteration_num = progress_update_context["iteration_num"]
    iteration_update = progress_update_context["iteration_update"]
    gene_set_stats_trace_fh = progress_update_context["gene_set_stats_trace_fh"]
    log_bf_state = progress_update_context["log_bf_state"]
    (log_bf_m, _log_bf_uncorrected_m, log_bf_raw_m) = log_bf_state

    progress_context = _build_gibbs_iteration_progress_context(
        progress_runtime_config=progress_runtime_config,
        iteration_update=iteration_update,
    )
    iteration_progress_config = progress_context["iteration_progress_config"]

    burn_in_update = _update_gibbs_burn_in_state(
        epoch_control=epoch_control,
        iteration_num=iteration_num,
        epoch_total_iter_offset=progress_context["epoch_total_iter_offset"],
        epoch_max_num_iter=progress_context["epoch_max_num_iter"],
        max_num_burn_in_for_epoch=progress_context["max_num_burn_in_for_epoch"],
        min_num_iter_for_epoch=progress_context["min_num_iter_for_epoch"],
        min_num_burn_in_for_epoch=progress_context["min_num_burn_in_for_epoch"],
        post_burn_reset_arrays=progress_context["post_burn_reset_arrays"],
        post_burn_reset_missing_arrays=progress_context["post_burn_reset_missing_arrays"],
        burn_in_config=iteration_progress_config["burn_in_config"],
        iter_state=iter_state,
        epoch_runtime=epoch_runtime,
    )
    _apply_gibbs_burn_in_control_update(epoch_control=epoch_control, burn_in_update=burn_in_update)

    post_burn_update = _update_gibbs_post_burn_state(
        state=state,
        max_num_post_burn_in_for_epoch=progress_context["max_num_post_burn_in_for_epoch"],
        min_num_post_burn_in_for_epoch=progress_context["min_num_post_burn_in_for_epoch"],
        epoch_max_num_iter=progress_context["epoch_max_num_iter"],
        diag_every=iteration_progress_config["diag_every"],
        post_burn_diag_config=iteration_progress_config["post_burn_diag_config"],
        iter_state=iter_state,
        epoch_sums=epoch_sums,
        epoch_priors=epoch_priors,
        epoch_control=epoch_control,
        run_state=run_state,
        log_bf_m=log_bf_m,
        log_bf_raw_m=log_bf_raw_m,
        full_betas_mean_m=progress_context["full_betas_mean_m"],
        full_postp_sample_m=progress_context["full_postp_sample_m"],
    )

    return _finalize_gibbs_iteration_progress(
        state=state,
        gene_set_stats_trace_fh=gene_set_stats_trace_fh,
        iteration_num=iteration_num,
        trace_chain_offset=progress_context["trace_chain_offset"],
        iter_state=iter_state,
        full_betas_mean_m=progress_context["full_betas_mean_m"],
        full_betas_sample_m=progress_context["full_betas_sample_m"],
        full_postp_mean_m=progress_context["full_postp_mean_m"],
        full_postp_sample_m=progress_context["full_postp_sample_m"],
        R_beta_v=epoch_control["R_beta_v"],
        betas_sem2_v=post_burn_update["betas_sem2_v"],
        use_mean_betas=iteration_progress_config["use_mean_betas"],
        epoch_control=epoch_control,
        post_burn_update=post_burn_update,
    )


def _finalize_gibbs_iteration_progress(
    state,
    gene_set_stats_trace_fh,
    iteration_num,
    trace_chain_offset,
    iter_state,
    full_betas_mean_m,
    full_betas_sample_m,
    full_postp_mean_m,
    full_postp_sample_m,
    R_beta_v,
    betas_sem2_v,
    use_mean_betas,
    epoch_control,
    post_burn_update,
):
    _write_gibbs_iteration_gene_set_stats_trace(
        gene_set_stats_trace_fh,
        iteration_num,
        trace_chain_offset,
        state,
        iter_state,
        full_betas_mean_m,
        full_betas_sample_m,
        full_postp_mean_m,
        full_postp_sample_m,
        R_beta_v,
        betas_sem2_v,
        use_mean_betas,
    )

    _apply_gibbs_post_burn_control_update(epoch_control=epoch_control, post_burn_update=post_burn_update)
    return {"done": post_burn_update["done"]}


def _finalize_gibbs_priors_for_sampling(
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
):
    # Regress out gene-length trend from priors (when requested), then choose
    # mean/sample priors used for the next iteration's Y sampling.
    total_priors_m = np.hstack((priors_sample_m, priors_missing_sample_m))
    gene_N = state.get_gene_N()
    gene_N_missing = state.get_gene_N(get_missing=True)

    all_gene_N = gene_N
    if state.genes_missing is not None:
        assert(gene_N_missing is not None)
        all_gene_N = np.concatenate((all_gene_N, gene_N_missing))

    priors_slope = total_priors_m.dot(all_gene_N) / (total_priors_m.shape[1] * np.var(all_gene_N))

    if adjust_priors:
        log("Adjusting priors with slopes ranging from %.4g-%.4g" % (np.min(priors_slope), np.max(priors_slope)), TRACE)
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


def _compute_gibbs_uncorrected_betas_and_defaults(
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


def _prepare_gibbs_gene_set_mask_with_prefilter(
    state,
    iter_state,
    prefilter_config,
    inner_beta_kwargs,
    default_betas_sample_m,
    default_postp_sample_m,
    default_betas_mean_m,
    default_postp_mean_m,
):
    (
        uncorrected_betas_mean_m,
        uncorrected_betas_sample_m,
        full_p_values_m,
        full_beta_tildes_m,
        full_ses_m,
        full_scale_factors_m,
        full_mean_shifts_m,
        full_is_dense_gene_set_m,
        full_ps_m,
        full_sigma2s_m,
    ) = (
        iter_state["uncorrected_betas_mean_m"],
        iter_state["uncorrected_betas_sample_m"],
        iter_state["full_p_values_m"],
        iter_state["full_beta_tildes_m"],
        iter_state["full_ses_m"],
        iter_state["full_scale_factors_m"],
        iter_state["full_mean_shifts_m"],
        iter_state["full_is_dense_gene_set_m"],
        iter_state["full_ps_m"],
        iter_state["full_sigma2s_m"],
    )
    (
        sparse_frac_gibbs,
        sparse_max_gibbs,
        pre_filter_batch_size,
        pre_filter_small_batch_size,
    ) = (
        prefilter_config["sparse_frac_gibbs"],
        prefilter_config["sparse_max_gibbs"],
        prefilter_config["pre_filter_batch_size"],
        prefilter_config["pre_filter_small_batch_size"],
    )
    inner_beta_kwargs_linear = _build_non_inf_beta_sampler_kwargs(inner_beta_kwargs)

    # Start from sparsity mask on uncorrected betas, then optionally run a
    # cheaper prefilter pass to mark additional near-zero sets before the
    # corrected beta batching stage.
    gene_set_mask_m = np.full(full_p_values_m.shape, True)
    gene_set_mask_m = _get_gibbs_gene_set_mask(
        uncorrected_betas_mean_m,
        uncorrected_betas_sample_m,
        full_p_values_m,
        sparse_frac=sparse_frac_gibbs,
        sparse_max=sparse_max_gibbs,
    )

    any_gene_set_mask = np.any(gene_set_mask_m, axis=0)
    if pre_filter_batch_size is not None and np.sum(any_gene_set_mask) > pre_filter_batch_size:
        (
            gene_set_mask_m,
            default_betas_sample_m,
            default_postp_sample_m,
            default_betas_mean_m,
            default_postp_mean_m,
        ) = _apply_gibbs_block_prefilter_pruning(
            state=state,
            any_gene_set_mask=any_gene_set_mask,
            pre_filter_small_batch_size=pre_filter_small_batch_size,
            full_beta_tildes_m=full_beta_tildes_m,
            full_ses_m=full_ses_m,
            full_scale_factors_m=full_scale_factors_m,
            full_mean_shifts_m=full_mean_shifts_m,
            full_is_dense_gene_set_m=full_is_dense_gene_set_m,
            full_ps_m=full_ps_m,
            full_sigma2s_m=full_sigma2s_m,
            inner_beta_kwargs_linear=inner_beta_kwargs_linear,
            full_p_values_m=full_p_values_m,
            sparse_frac_gibbs=sparse_frac_gibbs,
            sparse_max_gibbs=sparse_max_gibbs,
            gene_set_mask_m=gene_set_mask_m,
            default_betas_sample_m=default_betas_sample_m,
            default_postp_sample_m=default_postp_sample_m,
            default_betas_mean_m=default_betas_mean_m,
            default_postp_mean_m=default_postp_mean_m,
        )

    gene_set_mask_m = _normalize_gibbs_gene_set_mask_across_chains(gene_set_mask_m)

    return (
        gene_set_mask_m,
        default_betas_sample_m,
        default_postp_sample_m,
        default_betas_mean_m,
        default_postp_mean_m,
    )


def _apply_gibbs_block_prefilter_pruning(
    state,
    any_gene_set_mask,
    pre_filter_small_batch_size,
    full_beta_tildes_m,
    full_ses_m,
    full_scale_factors_m,
    full_mean_shifts_m,
    full_is_dense_gene_set_m,
    full_ps_m,
    full_sigma2s_m,
    inner_beta_kwargs_linear,
    full_p_values_m,
    sparse_frac_gibbs,
    sparse_max_gibbs,
    gene_set_mask_m,
    default_betas_sample_m,
    default_postp_sample_m,
    default_betas_mean_m,
    default_postp_mean_m,
):
    num_batches = state._get_num_X_blocks(
        state.X_orig[:,any_gene_set_mask],
        batch_size=pre_filter_small_batch_size,
    )
    if num_batches <= 1:
        return (
            gene_set_mask_m,
            default_betas_sample_m,
            default_postp_sample_m,
            default_betas_mean_m,
            default_postp_mean_m,
        )

    gene_set_block_masks = state._compute_gene_set_batches(
        V=None,
        X_orig=state.X_orig[:,any_gene_set_mask],
        mean_shifts=state.mean_shifts[any_gene_set_mask],
        scale_factors=state.scale_factors[any_gene_set_mask],
        find_correlated_instead=pre_filter_small_batch_size,
    )

    if len(gene_set_block_masks) == 0:
        return (
            gene_set_mask_m,
            default_betas_sample_m,
            default_postp_sample_m,
            default_betas_mean_m,
            default_postp_mean_m,
        )

    if np.sum(gene_set_block_masks[-1]) == 1 and len(gene_set_block_masks) > 1:
        # Merge singleton tail block into previous block.
        gene_set_block_masks[-2][gene_set_block_masks[-1]] = True
        gene_set_block_masks = gene_set_block_masks[:-1]
    if len(gene_set_block_masks) <= 1 or np.sum(gene_set_block_masks[0]) <= 1:
        return (
            gene_set_mask_m,
            default_betas_sample_m,
            default_postp_sample_m,
            default_betas_mean_m,
            default_postp_mean_m,
        )

    V_sparse = _build_gibbs_prefilter_sparse_V(
        state=state,
        any_gene_set_mask=any_gene_set_mask,
        gene_set_block_masks=gene_set_block_masks,
    )
    log("Running %d blocks to check for zeros..." % len(gene_set_block_masks), DEBUG)
    (
        half_corrected_betas_sample_m,
        half_corrected_postp_sample_m,
        half_corrected_betas_mean_m,
        half_corrected_postp_mean_m,
    ) = state._calculate_non_inf_betas(
        initial_p=None,
        beta_tildes=full_beta_tildes_m[:,any_gene_set_mask],
        ses=full_ses_m[:,any_gene_set_mask],
        V=V_sparse,
        X_orig=None,
        scale_factors=full_scale_factors_m[:,any_gene_set_mask],
        mean_shifts=full_mean_shifts_m[:,any_gene_set_mask],
        is_dense_gene_set=full_is_dense_gene_set_m[:,any_gene_set_mask],
        ps=full_ps_m[:,any_gene_set_mask],
        sigma2s=full_sigma2s_m[:,any_gene_set_mask],
        return_sample=True,
        update_hyper_sigma=False,
        update_hyper_p=False,
        **inner_beta_kwargs_linear,
    )

    add_zero_mask_m = ~(
        _get_gibbs_gene_set_mask(
            half_corrected_betas_mean_m,
            half_corrected_betas_sample_m,
            full_p_values_m,
            sparse_frac=sparse_frac_gibbs,
            sparse_max=sparse_max_gibbs,
        )
    )

    if np.any(add_zero_mask_m):
        map_to_full = np.where(any_gene_set_mask)[0]
        set_to_zero_full = np.where(add_zero_mask_m)
        set_to_zero_full = (set_to_zero_full[0], map_to_full[set_to_zero_full[1]])
        orig_zero = np.sum(np.any(gene_set_mask_m, axis=0))
        gene_set_mask_m[set_to_zero_full] = False
        new_zero = np.sum(np.any(gene_set_mask_m, axis=0))
        log("Found %d additional zero gene sets" % (orig_zero - new_zero), DEBUG)

        default_betas_sample_m[set_to_zero_full] = half_corrected_betas_sample_m[add_zero_mask_m]
        default_postp_sample_m[set_to_zero_full] = half_corrected_postp_sample_m[add_zero_mask_m]
        default_betas_mean_m[set_to_zero_full] = half_corrected_betas_mean_m[add_zero_mask_m]
        default_postp_mean_m[set_to_zero_full] = half_corrected_postp_mean_m[add_zero_mask_m]

    return (
        gene_set_mask_m,
        default_betas_sample_m,
        default_postp_sample_m,
        default_betas_mean_m,
        default_postp_mean_m,
    )


def _build_gibbs_prefilter_sparse_V(state, any_gene_set_mask, gene_set_block_masks):
    V_data = []
    V_rows = []
    V_cols = []
    for gene_set_block_mask in gene_set_block_masks:
        V_block = state._calculate_V_internal(
            state.X_orig[:,any_gene_set_mask][:,gene_set_block_mask],
            state.y_corr_cholesky,
            state.mean_shifts[any_gene_set_mask][gene_set_block_mask],
            state.scale_factors[any_gene_set_mask][gene_set_block_mask],
        )
        orig_indices = np.where(gene_set_block_mask)[0]
        V_rows += list(np.repeat(orig_indices, V_block.shape[0]))
        V_cols += list(np.tile(orig_indices, V_block.shape[0]))
        V_data += list(V_block.ravel())

    return sparse.csc_matrix(
        (V_data, (V_rows, V_cols)),
        shape=(np.sum(any_gene_set_mask), np.sum(any_gene_set_mask)),
    )


def _normalize_gibbs_gene_set_mask_across_chains(gene_set_mask_m):
    if sum(np.any(gene_set_mask_m, axis=0)) == 0:
        gene_set_mask_m[:,0] = 1

    num_non_missing_v = np.sum(gene_set_mask_m, axis=1)
    max_num_non_missing = np.max(num_non_missing_v)
    max_num_non_missing_idx = np.argmax(num_non_missing_v)
    log("Max number of gene sets to keep across all chains is %d" % (max_num_non_missing))
    log("Keeping %d gene sets that had non-zero uncorected betas" % (sum(np.any(gene_set_mask_m, axis=0))))
    for chain_num in range(gene_set_mask_m.shape[0]):
        if num_non_missing_v[chain_num] < max_num_non_missing:
            cur_num = 0
            for index in np.nonzero(
                gene_set_mask_m[max_num_non_missing_idx,:] & ~gene_set_mask_m[chain_num,:]
            )[0]:
                assert(gene_set_mask_m[chain_num,index] == False)
                gene_set_mask_m[chain_num,index] = True
                cur_num += 1
                if cur_num >= max_num_non_missing - num_non_missing_v[chain_num]:
                    break
    return gene_set_mask_m


def _compute_gibbs_corrected_betas_for_gene_set_mask(
    state,
    gene_set_mask_m,
    default_betas_sample_m,
    default_postp_sample_m,
    default_betas_mean_m,
    default_postp_mean_m,
    full_beta_tildes_m,
    full_ses_m,
    full_scale_factors_m,
    full_mean_shifts_m,
    full_is_dense_gene_set_m,
    full_ps_m,
    full_sigma2s_m,
    uncorrected_betas_mean_m,
    use_mean_betas,
    warm_start,
    prev_warm_start_betas_m,
    prev_warm_start_postp_m,
    debug_zero_sparse,
    num_chains,
    num_batches_parallel,
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
    betas_trace_out,
):
    # Estimate corrected non-inf betas in chain batches to keep V construction
    # memory bounded while preserving legacy batching behavior.
    (
        full_betas_mean_m,
        full_betas_sample_m,
        full_postp_mean_m,
        full_postp_sample_m,
    ) = _initialize_gibbs_corrected_beta_output_matrices(
        default_betas_mean_m=default_betas_mean_m,
        default_betas_sample_m=default_betas_sample_m,
        default_postp_mean_m=default_postp_mean_m,
        default_postp_sample_m=default_postp_sample_m,
        debug_zero_sparse=debug_zero_sparse,
    )

    num_calculations = int(np.ceil(num_chains / num_batches_parallel))
    for calc in range(num_calculations):
        (begin, end) = _get_gibbs_chain_batch_bounds(
            batch=calc,
            stack_batch_size=num_batches_parallel,
            num_chains=num_chains,
        )

        (
            cur_gene_set_mask,
            num_missing,
            run_one_V,
            beta_tildes_m,
            ses_m,
            V_m,
            scale_factors_m,
            mean_shifts_m,
            is_dense_gene_set_m,
            ps_m,
            sigma2s_m,
            init_betas_m,
            init_postp_m,
        ) = _prepare_gibbs_corrected_batch_inputs(
            state=state,
            gene_set_mask_m=gene_set_mask_m,
            full_beta_tildes_m=full_beta_tildes_m,
            full_ses_m=full_ses_m,
            full_scale_factors_m=full_scale_factors_m,
            full_mean_shifts_m=full_mean_shifts_m,
            full_is_dense_gene_set_m=full_is_dense_gene_set_m,
            full_ps_m=full_ps_m,
            full_sigma2s_m=full_sigma2s_m,
            warm_start=warm_start,
            prev_warm_start_betas_m=prev_warm_start_betas_m,
            prev_warm_start_postp_m=prev_warm_start_postp_m,
            begin=begin,
            end=end,
            num_chains=num_chains,
        )

        (
            cur_betas_sample_m,
            cur_postp_sample_m,
            cur_betas_mean_m,
            cur_postp_mean_m,
        ) = _run_gibbs_corrected_beta_sampler(
            state=state,
            beta_tildes_m=beta_tildes_m,
            ses_m=ses_m,
            V_m=V_m,
            scale_factors_m=scale_factors_m,
            mean_shifts_m=mean_shifts_m,
            is_dense_gene_set_m=is_dense_gene_set_m,
            ps_m=ps_m,
            sigma2s_m=sigma2s_m,
            passed_in_max_num_burn_in=passed_in_max_num_burn_in,
            max_num_iter_betas=max_num_iter_betas,
            min_num_iter_betas=min_num_iter_betas,
            num_chains_betas=num_chains_betas,
            r_threshold_burn_in_betas=r_threshold_burn_in_betas,
            use_max_r_for_convergence_betas=use_max_r_for_convergence_betas,
            max_frac_sem_betas=max_frac_sem_betas,
            max_allowed_batch_correlation=max_allowed_batch_correlation,
            gauss_seidel_betas=gauss_seidel_betas,
            num_missing=num_missing,
            sparse_solution=sparse_solution,
            sparse_frac_betas=sparse_frac_betas,
            betas_trace_out=betas_trace_out,
            gene_set_mask_m=gene_set_mask_m,
            init_betas_m=init_betas_m,
            init_postp_m=init_postp_m,
        )

        _store_gibbs_corrected_batch_results(
            run_one_V=run_one_V,
            full_betas_sample_m=full_betas_sample_m,
            full_postp_sample_m=full_postp_sample_m,
            full_betas_mean_m=full_betas_mean_m,
            full_postp_mean_m=full_postp_mean_m,
            cur_betas_sample_m=cur_betas_sample_m,
            cur_postp_sample_m=cur_postp_sample_m,
            cur_betas_mean_m=cur_betas_mean_m,
            cur_postp_mean_m=cur_postp_mean_m,
            begin=begin,
            end=end,
            cur_gene_set_mask=cur_gene_set_mask,
            gene_set_mask_m=gene_set_mask_m,
        )

        if run_one_V:
            print_overlapping = None
            _log_gibbs_overlapping_corrected_beta_details(
                print_overlapping=print_overlapping,
                state=state,
                cur_gene_set_mask=cur_gene_set_mask,
                V_m=V_m,
                cur_betas_mean_m=cur_betas_mean_m,
                cur_betas_sample_m=cur_betas_sample_m,
                use_mean_betas=use_mean_betas,
                uncorrected_betas_mean_m=uncorrected_betas_mean_m,
            )

    return (
        full_betas_sample_m,
        full_postp_sample_m,
        full_betas_mean_m,
        full_postp_mean_m,
    )


def _store_gibbs_corrected_batch_results(
    run_one_V,
    full_betas_sample_m,
    full_postp_sample_m,
    full_betas_mean_m,
    full_postp_mean_m,
    cur_betas_sample_m,
    cur_postp_sample_m,
    cur_betas_mean_m,
    cur_postp_mean_m,
    begin,
    end,
    cur_gene_set_mask,
    gene_set_mask_m,
):
    if run_one_V:
        full_betas_sample_m[begin:end,cur_gene_set_mask] = cur_betas_sample_m
        full_postp_sample_m[begin:end,cur_gene_set_mask] = cur_postp_sample_m
        full_betas_mean_m[begin:end,cur_gene_set_mask] = cur_betas_mean_m
        full_postp_mean_m[begin:end,cur_gene_set_mask] = cur_postp_mean_m
        return

    full_betas_sample_m[begin:end,:][gene_set_mask_m[begin:end,:]] = cur_betas_sample_m.ravel()
    full_postp_sample_m[begin:end,:][gene_set_mask_m[begin:end,:]] = cur_postp_sample_m.ravel()
    full_betas_mean_m[begin:end,:][gene_set_mask_m[begin:end,:]] = cur_betas_mean_m.ravel()
    full_postp_mean_m[begin:end,:][gene_set_mask_m[begin:end,:]] = cur_postp_mean_m.ravel()


def _log_gibbs_overlapping_corrected_beta_details(
    print_overlapping,
    state,
    cur_gene_set_mask,
    V_m,
    cur_betas_mean_m,
    cur_betas_sample_m,
    use_mean_betas,
    uncorrected_betas_mean_m,
):
    if print_overlapping is None:
        return
    gene_sets_run = [state.gene_sets[i] for i in range(len(state.gene_sets)) if cur_gene_set_mask[i]]
    gene_set_to_ind = pegs_construct_map_to_ind(gene_sets_run)
    for gene_set in print_overlapping:
        if gene_set in gene_set_to_ind:
            log("For gene set %s" % (gene_set))
            ind = gene_set_to_ind[gene_set]
            values = V_m[ind,:] * (cur_betas_mean_m if use_mean_betas else cur_betas_sample_m)
            indices = np.argsort(values, axis=1)
            for chain in range(values.shape[0]):
                log("Chain %d (uncorrected beta=%.4g, corrected beta=%.4g)" % (chain, uncorrected_betas_mean_m[chain,state.gene_set_to_ind[gene_set]], (cur_betas_mean_m[chain,ind] if use_mean_betas else cur_betas_sample_m[chain,ind])))
                for i in indices[chain,::-1]:
                    if values[chain,i] == 0:
                        break
                    log("%s, V=%.4g, beta=%.4g, prod=%.4g" % (gene_sets_run[i], V_m[ind,i], (cur_betas_mean_m[chain,i] if use_mean_betas else cur_betas_sample_m[chain,i]), values[chain,i]))


def _prepare_gibbs_corrected_batch_inputs(
    state,
    gene_set_mask_m,
    full_beta_tildes_m,
    full_ses_m,
    full_scale_factors_m,
    full_mean_shifts_m,
    full_is_dense_gene_set_m,
    full_ps_m,
    full_sigma2s_m,
    warm_start,
    prev_warm_start_betas_m,
    prev_warm_start_postp_m,
    begin,
    end,
    num_chains,
):
    # Build the superset V matrix once for this chain batch.
    cur_gene_set_mask = np.any(gene_set_mask_m[begin:end,:], axis=0)
    num_gene_set_mask = np.sum(cur_gene_set_mask)
    max_num_gene_set_mask = np.max(np.sum(gene_set_mask_m, axis=1))
    V_superset = state._calculate_V_internal(
        state.X_orig[:,cur_gene_set_mask],
        state.y_corr_cholesky,
        state.mean_shifts[cur_gene_set_mask],
        state.scale_factors[cur_gene_set_mask],
    )

    # If the superset is not too much larger than max-per-chain, reuse one V;
    # otherwise subset V per chain.
    run_one_V = num_gene_set_mask < 5 * max_num_gene_set_mask
    if run_one_V:
        num_non_missing = np.sum(cur_gene_set_mask)
    else:
        num_non_missing = np.max(np.sum(gene_set_mask_m, axis=1))
    num_missing = gene_set_mask_m.shape[1] - num_non_missing

    if run_one_V:
        (
            beta_tildes_m,
            ses_m,
            V_m,
            scale_factors_m,
            mean_shifts_m,
            is_dense_gene_set_m,
            ps_m,
            sigma2s_m,
            init_betas_m,
            init_postp_m,
        ) = _prepare_gibbs_corrected_run_one_v_inputs(
            state=state,
            full_beta_tildes_m=full_beta_tildes_m,
            full_ses_m=full_ses_m,
            cur_gene_set_mask=cur_gene_set_mask,
            begin=begin,
            end=end,
            V_superset=V_superset,
            warm_start=warm_start,
            prev_warm_start_betas_m=prev_warm_start_betas_m,
            prev_warm_start_postp_m=prev_warm_start_postp_m,
        )
    else:
        (
            beta_tildes_m,
            ses_m,
            V_m,
            scale_factors_m,
            mean_shifts_m,
            is_dense_gene_set_m,
            ps_m,
            sigma2s_m,
            init_betas_m,
            init_postp_m,
        ) = _prepare_gibbs_corrected_per_chain_v_inputs(
            gene_set_mask_m=gene_set_mask_m,
            full_beta_tildes_m=full_beta_tildes_m,
            full_ses_m=full_ses_m,
            full_scale_factors_m=full_scale_factors_m,
            full_mean_shifts_m=full_mean_shifts_m,
            full_is_dense_gene_set_m=full_is_dense_gene_set_m,
            full_ps_m=full_ps_m,
            full_sigma2s_m=full_sigma2s_m,
            cur_gene_set_mask=cur_gene_set_mask,
            V_superset=V_superset,
            begin=begin,
            end=end,
            num_chains=num_chains,
            num_non_missing=num_non_missing,
            warm_start=warm_start,
            prev_warm_start_betas_m=prev_warm_start_betas_m,
            prev_warm_start_postp_m=prev_warm_start_postp_m,
        )

    return (
        cur_gene_set_mask,
        num_missing,
        run_one_V,
        beta_tildes_m,
        ses_m,
        V_m,
        scale_factors_m,
        mean_shifts_m,
        is_dense_gene_set_m,
        ps_m,
        sigma2s_m,
        init_betas_m,
        init_postp_m,
    )


def _prepare_gibbs_corrected_run_one_v_inputs(
    state,
    full_beta_tildes_m,
    full_ses_m,
    cur_gene_set_mask,
    begin,
    end,
    V_superset,
    warm_start,
    prev_warm_start_betas_m,
    prev_warm_start_postp_m,
):
    beta_tildes_m = full_beta_tildes_m[begin:end,cur_gene_set_mask]
    ses_m = full_ses_m[begin:end,cur_gene_set_mask]
    V_m = V_superset
    scale_factors_m = state.scale_factors[cur_gene_set_mask]
    mean_shifts_m = state.mean_shifts[cur_gene_set_mask]
    is_dense_gene_set_m = state.is_dense_gene_set[cur_gene_set_mask]
    ps_m = None
    if state.ps is not None:
        ps_m = state.ps[cur_gene_set_mask]
    sigma2s_m = None
    if state.sigma2s is not None:
        sigma2s_m = state.sigma2s[cur_gene_set_mask]

    init_betas_m = None
    init_postp_m = None
    if warm_start and prev_warm_start_betas_m is not None:
        init_betas_m = prev_warm_start_betas_m[begin:end,cur_gene_set_mask]
        init_postp_m = prev_warm_start_postp_m[begin:end,cur_gene_set_mask]

    return (
        beta_tildes_m,
        ses_m,
        V_m,
        scale_factors_m,
        mean_shifts_m,
        is_dense_gene_set_m,
        ps_m,
        sigma2s_m,
        init_betas_m,
        init_postp_m,
    )


def _prepare_gibbs_corrected_per_chain_v_inputs(
    gene_set_mask_m,
    full_beta_tildes_m,
    full_ses_m,
    full_scale_factors_m,
    full_mean_shifts_m,
    full_is_dense_gene_set_m,
    full_ps_m,
    full_sigma2s_m,
    cur_gene_set_mask,
    V_superset,
    begin,
    end,
    num_chains,
    num_non_missing,
    warm_start,
    prev_warm_start_betas_m,
    prev_warm_start_postp_m,
):
    non_missing_matrix_shape = (num_chains, num_non_missing)
    beta_tildes_m = full_beta_tildes_m[gene_set_mask_m].reshape(non_missing_matrix_shape)[begin:end,:]
    ses_m = full_ses_m[gene_set_mask_m].reshape(non_missing_matrix_shape)[begin:end,:]
    scale_factors_m = full_scale_factors_m[gene_set_mask_m].reshape(non_missing_matrix_shape)[begin:end,:]
    mean_shifts_m = full_mean_shifts_m[gene_set_mask_m].reshape(non_missing_matrix_shape)[begin:end,:]
    is_dense_gene_set_m = full_is_dense_gene_set_m[gene_set_mask_m].reshape(non_missing_matrix_shape)[begin:end,:]
    ps_m = None
    if full_ps_m is not None:
        ps_m = full_ps_m[gene_set_mask_m].reshape(non_missing_matrix_shape)[begin:end,:]
    sigma2s_m = None
    if full_sigma2s_m is not None:
        sigma2s_m = full_sigma2s_m[gene_set_mask_m].reshape(non_missing_matrix_shape)[begin:end,:]

    init_betas_m = None
    init_postp_m = None
    if warm_start and prev_warm_start_betas_m is not None:
        init_betas_m = prev_warm_start_betas_m[gene_set_mask_m].reshape(non_missing_matrix_shape)[begin:end,:]
        init_postp_m = prev_warm_start_postp_m[gene_set_mask_m].reshape(non_missing_matrix_shape)[begin:end,:]

    V_m = np.zeros((end-begin, beta_tildes_m.shape[1], beta_tildes_m.shape[1]))
    for i,j in zip(range(begin, end),range(end-begin)):
        gene_set_mask_subset = gene_set_mask_m[i,cur_gene_set_mask]
        V_m[j,:,:] = V_superset[gene_set_mask_subset,:][:,gene_set_mask_subset]

    return (
        beta_tildes_m,
        ses_m,
        V_m,
        scale_factors_m,
        mean_shifts_m,
        is_dense_gene_set_m,
        ps_m,
        sigma2s_m,
        init_betas_m,
        init_postp_m,
    )


def _run_gibbs_corrected_beta_sampler(
    state,
    beta_tildes_m,
    ses_m,
    V_m,
    scale_factors_m,
    mean_shifts_m,
    is_dense_gene_set_m,
    ps_m,
    sigma2s_m,
    passed_in_max_num_burn_in,
    max_num_iter_betas,
    min_num_iter_betas,
    num_chains_betas,
    r_threshold_burn_in_betas,
    use_max_r_for_convergence_betas,
    max_frac_sem_betas,
    max_allowed_batch_correlation,
    gauss_seidel_betas,
    num_missing,
    sparse_solution,
    sparse_frac_betas,
    betas_trace_out,
    gene_set_mask_m,
    init_betas_m,
    init_postp_m,
):
    return state._calculate_non_inf_betas(
        initial_p=None,
        beta_tildes=beta_tildes_m,
        ses=ses_m,
        V=V_m,
        scale_factors=scale_factors_m,
        mean_shifts=mean_shifts_m,
        is_dense_gene_set=is_dense_gene_set_m,
        ps=ps_m,
        sigma2s=sigma2s_m,
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
        num_missing_gene_sets=num_missing,
        sparse_solution=sparse_solution,
        sparse_frac_betas=sparse_frac_betas,
        betas_trace_out=betas_trace_out,
        debug_gene_sets=[state.gene_sets[i] for i in range(len(state.gene_sets)) if gene_set_mask_m[0,i]],
        init_betas=init_betas_m,
        init_postp=init_postp_m,
    )


def _initialize_gibbs_corrected_beta_output_matrices(
    default_betas_mean_m,
    default_betas_sample_m,
    default_postp_mean_m,
    default_postp_sample_m,
    debug_zero_sparse,
):
    if debug_zero_sparse:
        return (
            copy.copy(default_betas_mean_m),
            copy.copy(default_betas_sample_m),
            copy.copy(default_postp_mean_m),
            copy.copy(default_postp_sample_m),
        )
    return (
        np.zeros(default_betas_mean_m.shape),
        np.zeros(default_betas_sample_m.shape),
        np.zeros(default_postp_mean_m.shape),
        np.zeros(default_postp_sample_m.shape),
    )


def _prepare_gibbs_next_warm_start(
    warm_start,
    use_mean_betas,
    prev_warm_start_betas_m,
    prev_warm_start_postp_m,
    full_betas_sample_m,
    full_betas_mean_m,
    full_postp_sample_m,
    full_postp_mean_m,
):
    if warm_start:
        # Filtered-out gene sets remain zero because the full_* matrices are
        # initialized to zero each iteration.
        if use_mean_betas:
            prev_warm_start_betas_m = copy.copy(full_betas_mean_m)
            prev_warm_start_postp_m = copy.copy(full_postp_mean_m)
        else:
            prev_warm_start_betas_m = copy.copy(full_betas_sample_m)
            prev_warm_start_postp_m = copy.copy(full_postp_sample_m)
    return (prev_warm_start_betas_m, prev_warm_start_postp_m)


def _compute_gibbs_iteration_priors_from_betas(
    state,
    full_betas_sample_m,
    full_betas_mean_m,
    priors_missing_sample_m,
    priors_missing_mean_m,
):
    priors_sample_m = _calc_priors_from_betas(state.X_orig, full_betas_sample_m, state.mean_shifts, state.scale_factors)
    priors_mean_m = _calc_priors_from_betas(state.X_orig, full_betas_mean_m, state.mean_shifts, state.scale_factors)
    if state.genes_missing is not None:
        priors_missing_sample_m = _calc_priors_from_betas(state.X_orig_missing_genes, full_betas_sample_m, state.mean_shifts, state.scale_factors)
        priors_missing_mean_m = _calc_priors_from_betas(state.X_orig_missing_genes, full_betas_mean_m, state.mean_shifts, state.scale_factors)
    return (
        priors_sample_m,
        priors_mean_m,
        priors_missing_sample_m,
        priors_missing_mean_m,
    )


def _maybe_refresh_gibbs_huge_scores(
    state,
    update_huge_scores,
    compute_Y_raw,
    priors_for_Y_m,
    log_bf_m,
    log_bf_uncorrected_m,
    log_bf_raw_m,
):
    if not (state.huge_signal_bfs is not None and update_huge_scores):
        return (log_bf_m, log_bf_uncorrected_m, log_bf_raw_m)

    log("Updating HuGE scores")
    combined_optional_bf_terms = _combine_optional_gene_bf_terms(state.Y_exomes, state.Y_positive_controls, state.Y_case_counts)
    rel_prior_log_bf = priors_for_Y_m + combined_optional_bf_terms

    (log_bf_m, log_bf_uncorrected_m, absent_genes, _) = state._distill_huge_signal_bfs(state.huge_signal_bfs_for_regression, state.huge_signal_posteriors_for_regression, state.huge_signal_sum_gene_cond_probabilities_for_regression, state.huge_signal_mean_gene_pos_for_regression, state.huge_signal_max_closest_gene_prob, state.huge_cap_region_posterior, state.huge_scale_region_posterior, state.huge_phantom_region_posterior, state.huge_allow_evidence_of_absence, state.gene_covariates, state.gene_covariates_mask, state.gene_covariates_mat_inv, state.gene_covariate_names, state.gene_covariate_intercept_index, state.genes, rel_prior_log_bf=rel_prior_log_bf)

    if compute_Y_raw:
        (log_bf_raw_m, _, _, _) = state._distill_huge_signal_bfs(state.huge_signal_bfs, state.huge_signal_posteriors, state.huge_signal_sum_gene_cond_probabilities, state.huge_signal_mean_gene_pos, state.huge_signal_max_closest_gene_prob, state.huge_cap_region_posterior, state.huge_scale_region_posterior, state.huge_phantom_region_posterior, state.huge_allow_evidence_of_absence, state.gene_covariates, state.gene_covariates_mask, state.gene_covariates_mat_inv, state.gene_covariate_names, state.gene_covariate_intercept_index, state.genes, rel_prior_log_bf=rel_prior_log_bf)
    else:
        log_bf_raw_m = copy.copy(log_bf_m)

    # Distillation uses optional terms internally for locus resolution;
    # add them back on the total gene-level BF scale.
    _add_optional_gene_bf_terms(
        log_bf_m,
        log_bf_uncorrected_m,
        log_bf_raw_m,
        state.Y_exomes,
        state.Y_positive_controls,
        state.Y_case_counts,
    )

    if len(absent_genes) > 0:
        bail("Error: huge_signal_bfs was incorrectly set and contains extra genes")
    return (log_bf_m, log_bf_uncorrected_m, log_bf_raw_m)


def _refresh_gibbs_iteration_priors_and_huge(
    state,
    warm_start,
    use_mean_betas,
    prev_warm_start_betas_m,
    prev_warm_start_postp_m,
    full_betas_sample_m,
    full_betas_mean_m,
    full_postp_sample_m,
    full_postp_mean_m,
    priors_missing_sample_m,
    priors_missing_mean_m,
    priors_for_Y_m,
    update_huge_scores,
    compute_Y_raw,
    log_bf_m,
    log_bf_uncorrected_m,
    log_bf_raw_m,
):
    (prev_warm_start_betas_m, prev_warm_start_postp_m) = _prepare_gibbs_next_warm_start(
        warm_start=warm_start,
        use_mean_betas=use_mean_betas,
        prev_warm_start_betas_m=prev_warm_start_betas_m,
        prev_warm_start_postp_m=prev_warm_start_postp_m,
        full_betas_sample_m=full_betas_sample_m,
        full_betas_mean_m=full_betas_mean_m,
        full_postp_sample_m=full_postp_sample_m,
        full_postp_mean_m=full_postp_mean_m,
    )
    (
        priors_sample_m,
        priors_mean_m,
        priors_missing_sample_m,
        priors_missing_mean_m,
    ) = _compute_gibbs_iteration_priors_from_betas(
        state=state,
        full_betas_sample_m=full_betas_sample_m,
        full_betas_mean_m=full_betas_mean_m,
        priors_missing_sample_m=priors_missing_sample_m,
        priors_missing_mean_m=priors_missing_mean_m,
    )
    (log_bf_m, log_bf_uncorrected_m, log_bf_raw_m) = _maybe_refresh_gibbs_huge_scores(
        state=state,
        update_huge_scores=update_huge_scores,
        compute_Y_raw=compute_Y_raw,
        priors_for_Y_m=priors_for_Y_m,
        log_bf_m=log_bf_m,
        log_bf_uncorrected_m=log_bf_uncorrected_m,
        log_bf_raw_m=log_bf_raw_m,
    )

    return {
        "prev_warm_start_betas_m": prev_warm_start_betas_m,
        "prev_warm_start_postp_m": prev_warm_start_postp_m,
        "priors_sample_m": priors_sample_m,
        "priors_mean_m": priors_mean_m,
        "priors_missing_sample_m": priors_missing_sample_m,
        "priors_missing_mean_m": priors_missing_mean_m,
        "log_bf_m": log_bf_m,
        "log_bf_uncorrected_m": log_bf_uncorrected_m,
        "log_bf_raw_m": log_bf_raw_m,
    }


def _update_gibbs_all_sums_and_maybe_restart_low_betas(
    state,
    epoch_runtime,
    epoch_sums,
    restart_controls,
    iteration_num,
    full_betas_mean_m,
):
    all_sum_betas_m = epoch_runtime["all_sum_betas_m"]
    all_sum_betas2_m = epoch_runtime["all_sum_betas2_m"]
    all_num_sum_m = epoch_runtime["all_num_sum_m"]

    all_sum_betas_m = np.add(all_sum_betas_m, full_betas_mean_m)
    all_sum_betas2_m = np.add(all_sum_betas2_m, np.power(full_betas_mean_m, 2))
    all_num_sum_m += 1

    R_beta_v = np.zeros(all_sum_betas_m.shape[1])

    low_beta_restart_update = _maybe_restart_gibbs_for_low_betas(
        state=state,
        increase_hyper_if_betas_below_for_epoch=restart_controls.increase_hyper_if_betas_below_for_epoch,
        experimental_hyper_mutation=restart_controls.experimental_hyper_mutation,
        num_before_checking_p_increase=restart_controls.num_before_checking_p_increase,
        p_scale_factor=restart_controls.p_scale_factor,
        epoch_runtime=epoch_runtime,
        epoch_sums=epoch_sums,
        num_mad=restart_controls.num_mad,
        num_attempts=restart_controls.num_attempts,
        max_num_attempt_restarts=restart_controls.max_num_attempt_restarts,
        iteration_num=iteration_num,
    )

    return GibbsAllIterationUpdate(
        all_sum_betas_m=all_sum_betas_m,
        all_sum_betas2_m=all_sum_betas2_m,
        all_num_sum_m=all_num_sum_m,
        R_beta_v=R_beta_v,
        gibbs_good=low_beta_restart_update.gibbs_good,
        num_p_increases=low_beta_restart_update.num_p_increases,
        should_break=low_beta_restart_update.should_break,
    )


def _build_inner_beta_sampler_common_kwargs(options):
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


def _configure_hyperparameters_for_main(state, options):
    sigma2_cond = options.sigma2_cond

    if sigma2_cond is not None:
        # map it with the scale factor
        state.set_sigma(options.sigma2_ext, options.sigma_power, convert_sigma_to_internal_units=False)
        sigma2_cond = state.get_sigma2()
        state.set_sigma(None, state.sigma_power)
    elif options.sigma2_ext is not None:
        state.set_sigma(options.sigma2_ext, options.sigma_power, convert_sigma_to_internal_units=True)
        log("Setting sigma=%.4g (given external=%.4g) " % (state.get_sigma2(), state.get_sigma2(convert_sigma_to_external_units=True)))
    elif options.sigma2 is not None:
        state.set_sigma(options.sigma2, options.sigma_power, convert_sigma_to_internal_units=False)
    elif options.top_gene_set_prior:
        state.set_sigma(
            state.convert_prior_to_var(
                options.top_gene_set_prior,
                options.num_gene_sets_for_prior if options.num_gene_sets_for_prior is not None else len(state.gene_sets),
                options.frac_gene_sets_for_prior,
            ),
            options.sigma_power,
            convert_sigma_to_internal_units=True,
        )
        if options.frac_gene_sets_for_prior == 1:
            # in this case sigma2_cond was specified, not sigma2
            sigma2_cond = state.get_sigma2()
            log("Setting sigma_cond=%.4g (external=%.4g) given top of %d gene sets prior of %.4g" % (state.get_sigma2(), state.get_sigma2(convert_sigma_to_external_units=True), options.num_gene_sets_for_prior, options.top_gene_set_prior))
            state.set_sigma(None, state.sigma_power)
        else:
            log("Setting sigma=%.4g (external=%.4g) given top of %d gene sets prior of %.4g" % (state.get_sigma2(), state.get_sigma2(convert_sigma_to_external_units=True), options.num_gene_sets_for_prior, options.top_gene_set_prior))

    # Legacy behavior: force sigma-power to 2 when sigma is fixed.
    if options.const_sigma:
        options.sigma_power = 2

    update_hyper_mode = options.update_hyper.lower()
    if update_hyper_mode == "both":
        options.update_hyper_p = True
        options.update_hyper_sigma = True
    elif update_hyper_mode == "p":
        options.update_hyper_p = True
        options.update_hyper_sigma = False
    elif update_hyper_mode == "sigma2" or update_hyper_mode == "sigma":
        options.update_hyper_p = False
        options.update_hyper_sigma = True
    elif update_hyper_mode == "none":
        options.update_hyper_p = False
        options.update_hyper_sigma = False
    else:
        bail("Invalid value for --update-hyper (both, p, sigma2, or none)")

    if options.gene_map_in:
        _read_gene_map(
            state,
            gene_map_in=options.gene_map_in,
            gene_map_orig_gene_col=options.gene_map_orig_gene_col,
            gene_map_new_gene_col=options.gene_map_new_gene_col,
        )
    if options.gene_loc_file:
        _init_gene_locs(state, options.gene_loc_file)

    return sigma2_cond

# --------------------------------------------------------------------------
# Main-pipeline Y input contract.
# This keeps source detection and read_Y kwargs assembly in one structured
# object so mode dispatch can reason about "what Y sources are present"
# without re-encoding ad-hoc option checks.
# --------------------------------------------------------------------------
@dataclass
class YPrimaryInputsContract:
    gwas_in: str | None = None
    huge_statistics_in: str | None = None
    exomes_in: str | None = None
    positive_controls_in: str | None = None
    positive_controls_list: list | None = None
    case_counts_in: str | None = None

    def has_any_source(self) -> bool:
        return any([
            self.gwas_in is not None,
            self.huge_statistics_in is not None,
            self.exomes_in is not None,
            self.positive_controls_in is not None,
            self.positive_controls_list is not None,
            self.case_counts_in is not None,
        ])

    def has_only_positive_controls(self) -> bool:
        return (
            self.positive_controls_in is not None
            or self.positive_controls_list is not None
        ) and (
            self.gwas_in is None
            and self.huge_statistics_in is None
            and self.exomes_in is None
            and self.case_counts_in is None
        )


@dataclass
class YReadContract:
    primary_inputs: YPrimaryInputsContract = field(default_factory=YPrimaryInputsContract)
    read_kwargs: dict = field(default_factory=dict)

    def has_any_source(self) -> bool:
        return self.primary_inputs.has_any_source()

    def has_only_positive_controls(self) -> bool:
        return self.primary_inputs.has_only_positive_controls()

    def to_read_kwargs(self) -> dict:
        return dict(self.read_kwargs)


try:
    from . import pigean_pipeline as _pigean_pipeline
except ImportError:
    import pigean_pipeline as _pigean_pipeline


BetaStageResult = _pigean_pipeline.BetaStageResult
PriorsStageResult = _pigean_pipeline.PriorsStageResult
GibbsStageResult = _pigean_pipeline.GibbsStageResult
GibbsStageConfig = _pigean_pipeline.GibbsStageConfig
NonHugePipelineResult = _pigean_pipeline.NonHugePipelineResult
MainPipelineResult = _pigean_pipeline.MainPipelineResult


def _build_main_y_read_contract(options):
    primary_inputs = YPrimaryInputsContract(
        gwas_in=options.gwas_in,
        huge_statistics_in=options.huge_statistics_in,
        exomes_in=options.exomes_in,
        positive_controls_in=options.positive_controls_in,
        positive_controls_list=options.positive_controls_list,
        case_counts_in=options.case_counts_in,
    )
    read_kwargs = dict(
        gwas_in=options.gwas_in,
        huge_statistics_in=options.huge_statistics_in,
        huge_statistics_out=options.huge_statistics_out,
        show_progress=not options.hide_progress,
        gwas_chrom_col=options.gwas_chrom_col,
        gwas_pos_col=options.gwas_pos_col,
        gwas_p_col=options.gwas_p_col,
        gwas_beta_col=options.gwas_beta_col,
        gwas_se_col=options.gwas_se_col,
        gwas_n_col=options.gwas_n_col,
        gwas_n=options.gwas_n,
        gwas_units=options.gwas_units,
        gwas_freq_col=options.gwas_freq_col,
        gwas_filter_col=options.gwas_filter_col,
        gwas_filter_value=options.gwas_filter_value,
        gwas_locus_col=options.gwas_locus_col,
        gwas_ignore_p_threshold=options.gwas_ignore_p_threshold,
        gwas_low_p=options.gwas_low_p,
        gwas_high_p=options.gwas_high_p,
        gwas_low_p_posterior=options.gwas_low_p_posterior,
        gwas_high_p_posterior=options.gwas_high_p_posterior,
        detect_low_power=options.gwas_detect_low_power,
        detect_high_power=options.gwas_detect_high_power,
        detect_adjust_huge=options.gwas_detect_adjust_huge,
        learn_window=options.learn_window,
        closest_gene_prob=options.closest_gene_prob,
        max_closest_gene_prob=options.max_closest_gene_prob,
        scale_raw_closest_gene=options.scale_raw_closest_gene,
        cap_raw_closest_gene=options.cap_raw_closest_gene,
        cap_region_posterior=options.cap_region_posterior,
        scale_region_posterior=options.scale_region_posterior,
        phantom_region_posterior=options.phantom_region_posterior,
        allow_evidence_of_absence=options.allow_evidence_of_absence,
        correct_huge=options.correct_huge,
        gws_prob_true=options.gene_zs_gws_prob_true,
        max_closest_gene_dist=options.max_closest_gene_dist,
        signal_window_size=options.signal_window_size,
        signal_min_sep=options.signal_min_sep,
        signal_max_logp_ratio=options.signal_max_logp_ratio,
        credible_set_span=options.credible_set_span,
        min_n_ratio=options.min_n_ratio,
        max_clump_ld=options.max_clump_ld,
        exomes_in=options.exomes_in,
        exomes_gene_col=options.exomes_gene_col,
        exomes_p_col=options.exomes_p_col,
        exomes_beta_col=options.exomes_beta_col,
        exomes_se_col=options.exomes_se_col,
        exomes_n_col=options.exomes_n_col,
        exomes_n=options.exomes_n,
        exomes_units=options.exomes_units,
        exomes_low_p=options.exomes_low_p,
        exomes_high_p=options.exomes_high_p,
        exomes_low_p_posterior=options.exomes_low_p_posterior,
        exomes_high_p_posterior=options.exomes_high_p_posterior,
        positive_controls_in=options.positive_controls_in,
        positive_controls_id_col=options.positive_controls_id_col,
        positive_controls_prob_col=options.positive_controls_prob_col,
        positive_controls_default_prob=options.positive_controls_default_prob,
        positive_controls_has_header=options.positive_controls_has_header,
        positive_controls_list=options.positive_controls_list,
        positive_controls_all_in=options.positive_controls_all_in,
        positive_controls_all_id_col=options.positive_controls_all_id_col,
        positive_controls_all_has_header=options.positive_controls_all_has_header,
        case_counts_in=options.case_counts_in,
        case_counts_gene_col=options.case_counts_gene_col,
        case_counts_revel_col=options.case_counts_revel_col,
        case_counts_count_col=options.case_counts_count_col,
        case_counts_tot_col=options.case_counts_tot_col,
        case_counts_max_freq_col=options.case_counts_max_freq_col,
        min_revels=options.counts_min_revels,
        mean_rrs=options.counts_mean_rrs,
        max_case_freq=options.counts_max_case_freq,
        ctrl_counts_in=options.ctrl_counts_in,
        ctrl_counts_gene_col=options.ctrl_counts_gene_col,
        ctrl_counts_revel_col=options.ctrl_counts_revel_col,
        ctrl_counts_count_col=options.ctrl_counts_count_col,
        ctrl_counts_tot_col=options.ctrl_counts_tot_col,
        ctrl_counts_max_freq_col=options.ctrl_counts_max_freq_col,
        max_ctrl_freq=options.counts_max_ctrl_freq,
        syn_revel_threshold=options.counts_syn_revel,
        syn_fisher_p=options.counts_syn_fisher_p,
        nu=options.counts_nu,
        beta=options.counts_beta,
        gene_loc_file=options.gene_loc_file_huge if options.gene_loc_file_huge is not None else options.gene_loc_file,
        exons_loc_file=options.exons_loc_file_huge,
        gene_covs_in=options.gene_covs_in,
        hold_out_chrom=options.hold_out_chrom,
        min_var_posterior=options.min_var_posterior,
        s2g_in=options.s2g_in,
        s2g_chrom_col=options.s2g_chrom_col,
        s2g_pos_col=options.s2g_pos_col,
        s2g_gene_col=options.s2g_gene_col,
        s2g_prob_col=options.s2g_prob_col,
        s2g_normalize_values=options.s2g_normalize_values,
        credible_sets_in=options.credible_sets_in,
        credible_sets_id_col=options.credible_sets_id_col,
        credible_sets_chrom_col=options.credible_sets_chrom_col,
        credible_sets_pos_col=options.credible_sets_pos_col,
        credible_sets_ppa_col=options.credible_sets_ppa_col,
    )
    return YReadContract(primary_inputs=primary_inputs, read_kwargs=read_kwargs)


def _run_main_adaptive_read_x(state, options, mode_state, sigma2_cond):
    # 1) Build parser-normalized mapping from each X input spec to its p_noninf index.
    xin_to_p_noninf_ind = pegs_build_xin_to_p_noninf_index_map(
        options.X_in,
        options.X_list,
        options.Xd_in,
        options.Xd_list,
        options.p_noninf,
        warn_fn=warn,
        bail_fn=bail,
    )

    skip_betas = (
        (mode_state["run_huge"] or mode_state["run_beta_tilde"])
        and not (mode_state["run_beta"] or mode_state["run_priors"] or mode_state["run_naive_priors"] or mode_state["run_gibbs"])
    )
    # Retry state for adaptive read_X filtering. Keep this explicit/local so we do
    # not leak transient retry behavior into persistent state.
    read_x_retry_state = {
        "filter_gene_set_p": options.filter_gene_set_p,
        "force_reread": False,
    }
    # 2) Read X and, if needed, adaptively relax filter_gene_set_p until enough
    #    gene sets survive or no further retry is required.
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
        )
        _run_read_x_stage(state, options.X_in, **read_x_kwargs)

        should_reread = False
        new_filter_gene_set_p = read_x_retry_state["filter_gene_set_p"]
        # If too few sets survived filtering, relax filter_gene_set_p and retry once per loop.
        if (
            options.min_num_gene_sets is not None
            and read_x_retry_state["filter_gene_set_p"] is not None
            and read_x_retry_state["filter_gene_set_p"] < 1
            and state.gene_sets is not None
            and len(state.gene_sets) < options.min_num_gene_sets
        ):
            fraction_to_increase = float(options.min_num_gene_sets) / (len(state.gene_sets) + 1)
            if fraction_to_increase > 1:
                # add in a fudge factor
                new_filter_gene_set_p = read_x_retry_state["filter_gene_set_p"] * fraction_to_increase * 1.2
                if new_filter_gene_set_p > 1:
                    new_filter_gene_set_p = 1
                log("Only read in %d gene sets; scaled --filter-gene-set-p to %.3g and re-reading gene sets" % (len(state.gene_sets), new_filter_gene_set_p))
                # Keep sigma stable across adaptive filter retries.
                state.set_sigma(sigma2_internal_before_read, state.sigma_power)
                should_reread = True
        if not should_reread:
            break
        read_x_retry_state["filter_gene_set_p"] = new_filter_gene_set_p
        read_x_retry_state["force_reread"] = True


def _run_read_x_stage(runtime, X_in, **read_x_kwargs):
    read_x_pipeline_config = pegs_build_read_x_pipeline_config(
        X_in,
        read_x_kwargs,
        bail_fn=bail,
    )
    return _read_x_pipeline(runtime, read_x_pipeline_config)


def _read_x_pipeline(runtime, read_x_pipeline_config):
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
        sparse_list_open_fn=open_gz,
        dense_list_open_fn=open,
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
        normalize_dense_gene_rows_fn=_normalize_dense_gene_rows,
        build_sparse_x_from_dense_input_fn=_build_sparse_x_from_dense_input,
        reindex_x_rows_to_current_genes_fn=_reindex_x_rows_to_current_genes,
        normalize_gene_set_weights_fn=_normalize_gene_set_weights,
        partition_missing_gene_rows_fn=_partition_missing_gene_rows,
        maybe_permute_gene_set_rows_fn=_maybe_permute_gene_set_rows,
        maybe_prefilter_x_block_fn=_maybe_prefilter_x_block,
        merge_missing_gene_rows_fn=_merge_missing_gene_rows,
        finalize_added_x_block_fn=_finalize_added_x_block,
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
        ensure_gene_universe_fn=_ensure_gene_universe_for_x,
        process_x_input_file_fn=_process_x_input_file,
        remove_tag_from_input_fn=pegs_remove_tag_from_input,
        log_fn=log,
        info_level=INFO,
        debug_level=DEBUG,
    )

    post_options = pegs_build_read_x_post_options(
        read_x_locals,
        batches=ingestion_state["batches"],
        num_ignored_gene_sets=ingestion_state["num_ignored_gene_sets"],
        ignored_for_fraction_inc=ingestion_state["ignored_for_fraction_inc"],
    )
    post_callbacks = PegsXReadPostCallbacks(
        standardize_qc_metrics_after_x_read_fn=_standardize_qc_metrics_after_x_read,
        maybe_correct_gene_set_betas_after_x_read_fn=_maybe_correct_gene_set_betas_after_x_read,
        maybe_limit_initial_gene_sets_by_p_fn=_maybe_limit_initial_gene_sets_by_p,
        maybe_prune_gene_sets_after_x_read_fn=_maybe_prune_gene_sets_after_x_read,
        initialize_hyper_defaults_after_x_read_fn=_initialize_hyper_defaults_after_x_read,
        maybe_learn_batch_hyper_after_x_read_fn=_maybe_learn_batch_hyper_after_x_read,
        maybe_adjust_overaggressive_p_filter_after_x_read_fn=_maybe_adjust_overaggressive_p_filter_after_x_read,
        apply_post_read_gene_set_size_and_qc_filters_fn=_apply_post_read_gene_set_size_and_qc_filters,
        maybe_filter_zero_uncorrected_betas_after_x_read_fn=_maybe_filter_zero_uncorrected_betas_after_x_read,
        maybe_reduce_gene_sets_to_max_after_x_read_fn=_maybe_reduce_gene_sets_to_max_after_x_read,
        record_read_x_counts_fn=pegs_record_read_x_counts,
    )
    xdata_seed.run_post_stage(
        runtime,
        post_options=post_options,
        post_callbacks=post_callbacks,
        log_fn=log,
        debug_level=DEBUG,
    )


def _mode_requires_gene_scores(mode_state):
    return (
        mode_state["run_huge"]
        or mode_state["run_beta_tilde"]
        or mode_state["run_beta"]
        or mode_state["run_priors"]
        or mode_state["run_naive_priors"]
        or mode_state["run_gibbs"]
    )


def _read_gene_phewas_bfs(
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
    cached.pop("state", None)
    cached.pop("kwargs", None)
    state.cached_gene_phewas_call = cached

    if gene_phewas_bfs_in is None:
        bail("Require --gene-stats-in or --gene-phewas-bfs-in for this operation")

    log("Reading --gene-phewas-bfs-in file %s" % gene_phewas_bfs_in, INFO)
    if state.genes is None:
        bail("Need to initialixe --X before reading gene_phewas")

    phewas_gene_to_X_gene = None
    if phewas_gene_to_X_gene_in is not None:
        phewas_gene_to_X_gene = pegs_parse_gene_map_file(
            phewas_gene_to_X_gene_in,
            allow_multi=True,
            bail_fn=bail,
        )

    pegs_load_and_apply_gene_phewas_bfs_to_runtime(
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
        open_text_fn=open_gz,
        get_col_fn=_get_col,
        construct_map_to_ind_fn=pegs_construct_map_to_ind,
        warn_fn=warn,
        bail_fn=bail,
        log_fn=lambda message: log(message, DEBUG),
    )
    state.phewas_state = pegs_sync_phewas_runtime_state(state)


def _reread_gene_phewas_bfs(state):
    if state.cached_gene_phewas_call is None:
        return
    log("Rereading gene phewas bfs...")
    _read_gene_phewas_bfs(state, **state.cached_gene_phewas_call)


def _load_advanced_set_b_y_inputs(state, options):
    if not options.betas_uncorrected_from_phewas:
        return False
    if not options.gene_phewas_bfs_in:
        bail("Require --gene-phewas-bfs-in for --betas-from-phewas option")
    _read_gene_phewas_bfs(
        state,
        gene_phewas_bfs_in=options.gene_phewas_bfs_in,
        gene_phewas_bfs_id_col=options.gene_phewas_bfs_id_col,
        gene_phewas_bfs_pheno_col=options.gene_phewas_bfs_pheno_col,
        gene_phewas_bfs_log_bf_col=options.gene_phewas_bfs_log_bf_col,
        gene_phewas_bfs_combined_col=options.gene_phewas_bfs_combined_col,
        gene_phewas_bfs_prior_col=options.gene_phewas_bfs_prior_col,
        phewas_gene_to_X_gene_in=options.gene_phewas_id_to_X_id,
        min_value=options.min_gene_phewas_read_value,
        max_num_entries_at_once=options.max_read_entries_at_once,
    )
    return True


def _load_main_Y_inputs(state, options, mode_state):
    if not _mode_requires_gene_scores(mode_state):
        return True

    y_read_contract = _build_main_y_read_contract(options)

    if options.gene_stats_in:
        _run_read_y_stage(
            state,
            gene_bfs_in=options.gene_stats_in,
            show_progress=not options.hide_progress,
            gene_bfs_id_col=options.gene_stats_id_col,
            gene_bfs_log_bf_col=options.gene_stats_log_bf_col,
            gene_bfs_combined_col=options.gene_stats_combined_col,
            gene_bfs_prob_col=options.gene_stats_prob_col,
            gene_bfs_prior_col=options.gene_stats_prior_col,
            gene_covs_in=options.gene_covs_in,
            hold_out_chrom=options.hold_out_chrom,
        )
        return False

    if y_read_contract.has_any_source():
        if y_read_contract.has_only_positive_controls():
            options.ols = True
            if options.positive_controls_all_in is None and not options.add_all_genes:
                bail("Specified positive controls without --positive-controls-all-in; therefore using all genes in gene sets as negatives. This may result in inflated enrichments. If you really want to run this, specify --add-all-genes")
        _run_read_y_contract_stage(state, y_read_contract)
        return False

    if _load_advanced_set_b_y_inputs(state, options):
        return False

    return True

def _run_advanced_set_b_phewas_beta_sampling_if_requested(state, options, beta_sampling_kwargs):
    try:
        from . import pigean_phewas as _pigean_phewas
    except ImportError:
        import pigean_phewas as _pigean_phewas

    return _pigean_phewas.run_advanced_set_b_phewas_beta_sampling_if_requested(
        sys.modules[__name__],
        state,
        options,
        beta_sampling_kwargs,
    )

def _run_advanced_set_b_output_phewas_if_requested(state, options):
    try:
        from . import pigean_phewas as _pigean_phewas
    except ImportError:
        import pigean_phewas as _pigean_phewas

    return _pigean_phewas.run_advanced_set_b_output_phewas_if_requested(
        sys.modules[__name__],
        state,
        options,
    )


def _write_eaggl_bundle_if_requested(state, options, mode):
    try:
        from . import pigean_outputs as _pigean_outputs
    except ImportError:
        import pigean_outputs as _pigean_outputs

    return _pigean_outputs.write_eaggl_bundle_if_requested(
        sys.modules[__name__],
        state,
        options,
        mode,
    )


def _run_main_beta_tilde_stage(state, options, mode_state):
    return _pigean_pipeline.run_main_beta_tilde_stage(
        sys.modules[__name__],
        state,
        options,
        mode_state,
    )


def _run_main_beta_stage(state, options, mode_state):
    return _pigean_pipeline.run_main_beta_stage(
        sys.modules[__name__],
        state,
        options,
        mode_state,
    )


def _run_main_priors_stage(state, options, mode_state):
    return _pigean_pipeline.run_main_priors_stage(
        sys.modules[__name__],
        state,
        options,
        mode_state,
    )


def _build_main_gibbs_stage_config(options):
    return _pigean_pipeline.build_main_gibbs_stage_config(options)


def _run_main_gibbs_stage(state, options, mode_state):
    return _pigean_pipeline.run_main_gibbs_stage(
        sys.modules[__name__],
        state,
        options,
        mode_state,
    )


def _run_main_non_huge_pipeline(state, options, mode_state, sigma2_cond, Y_not_loaded):
    return _pigean_pipeline.run_main_non_huge_pipeline(
        sys.modules[__name__],
        state,
        options,
        mode_state,
        sigma2_cond,
        Y_not_loaded,
    )


def _write_main_outputs_and_optional_phewas(state, options, mode_state, mode):
    try:
        from . import pigean_outputs as _pigean_outputs
    except ImportError:
        import pigean_outputs as _pigean_outputs

    return _pigean_outputs.write_main_outputs_and_optional_phewas(
        sys.modules[__name__],
        state,
        options,
        mode_state,
        mode,
    )


def run_main_pipeline(options, mode):
    try:
        from . import pigean_dispatch as _pigean_dispatch
    except ImportError:
        import pigean_dispatch as _pigean_dispatch

    return _pigean_dispatch.run_main_pipeline(sys.modules[__name__], options, mode)


def main(argv=None):
    try:
        should_continue = _bootstrap_cli(argv)
        if not should_continue:
            return 0
        run_main_pipeline(options, mode)
        return 0
    except PegsCliError as exc:
        return pegs_handle_cli_exception(exc, argv=argv, debug_level=debug_level)
    except Exception as exc:
        return pegs_handle_unexpected_exception(exc, argv=argv, debug_level=debug_level)


if __name__ == '__main__':
    raise SystemExit(main())
