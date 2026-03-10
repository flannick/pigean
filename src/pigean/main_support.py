from __future__ import annotations

import contextlib
import copy
import functools
import gzip
import itertools
import json
import os
import random
import re
import sys
import time
from dataclasses import dataclass

import numpy as np
import scipy
import scipy.sparse as sparse

from pegs_cli_errors import DataValidationError

import pegs_shared.bundle as pegs_bundle
import pegs_shared.cli as pegs_cli
import pegs_shared.gene_io as pegs_gene_io
import pegs_shared.huge_cache as pegs_huge_cache
import pegs_shared.io_common as pegs_io_common
import pegs_shared.phewas as pegs_phewas
import pegs_shared.regression as pegs_regression
import pegs_shared.runtime_matrix as pegs_runtime_matrix
import pegs_shared.types as pegs_types
import pegs_shared.xdata as pegs_xdata
import pegs_shared.ydata as pegs_ydata
import pegs_utils as pegs_utils_mod

from . import cli as pigean_cli
from . import huge as pigean_huge
from . import model as pigean_model
from . import phewas as pigean_phewas
from . import runtime as pigean_runtime
from . import state as pigean_state
from . import x_inputs as pigean_x_inputs
from . import x_inputs_core as pigean_x_inputs_core
from . import y_inputs as pigean_y_inputs
from . import y_inputs_core as pigean_y_inputs_core


_CLI_STATE_FIELDS = (
    "options",
    "args",
    "mode",
    "config_mode",
    "cli_specified_dests",
    "config_specified_dests",
    "NONE",
    "INFO",
    "DEBUG",
    "TRACE",
    "debug_level",
    "log_fh",
    "warnings_fh",
    "log",
    "warn",
)


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
_json_safe = pigean_cli._json_safe

pegs_initialize_matrix_and_gene_index_state = pegs_xdata.initialize_matrix_and_gene_index_state
pegs_sync_runtime_state_bundle = pegs_ydata.sync_runtime_state_bundle
pegs_sync_phewas_runtime_state = pegs_ydata.sync_phewas_runtime_state
pegs_build_read_x_pipeline_config = pegs_xdata.build_read_x_pipeline_config
pegs_build_xin_to_p_noninf_index_map = pegs_utils_mod.build_xin_to_p_noninf_index_map
pegs_remove_tag_from_input = pegs_utils_mod.remove_tag_from_input
pegs_record_read_x_counts = pegs_utils_mod.record_read_x_counts
pegs_standardize_qc_metrics_after_x_read = pegs_utils_mod.standardize_qc_metrics_after_x_read
pegs_maybe_correct_gene_set_betas_after_x_read = pegs_utils_mod.maybe_correct_gene_set_betas_after_x_read
pegs_maybe_limit_initial_gene_sets_by_p = pegs_utils_mod.maybe_limit_initial_gene_sets_by_p
pegs_maybe_prune_gene_sets_after_x_read = pegs_utils_mod.maybe_prune_gene_sets_after_x_read
pegs_initialize_hyper_defaults_after_x_read = pegs_utils_mod.initialize_hyper_defaults_after_x_read
pegs_maybe_learn_batch_hyper_after_x_read = pigean_state._maybe_learn_batch_hyper_after_x_read
pegs_maybe_adjust_overaggressive_p_filter_after_x_read = pegs_utils_mod.maybe_adjust_overaggressive_p_filter_after_x_read
pegs_apply_post_read_gene_set_size_and_qc_filters = pegs_utils_mod.apply_post_read_gene_set_size_and_qc_filters
pegs_load_and_apply_gene_phewas_bfs_to_runtime = pegs_utils_mod.load_and_apply_gene_phewas_bfs_to_runtime
pegs_load_and_apply_gene_set_statistics_to_runtime = pegs_utils_mod.load_and_apply_gene_set_statistics_to_runtime
pegs_set_runtime_y_from_inputs = pegs_utils_mod.set_runtime_y_from_inputs
pegs_write_gene_set_statistics = pegs_utils_mod.write_gene_set_statistics
pegs_write_phewas_gene_set_statistics = pegs_utils_mod.write_phewas_gene_set_statistics
pegs_write_gene_statistics = pegs_utils_mod.write_gene_statistics
pegs_write_gene_gene_set_statistics = pegs_utils_mod.write_gene_gene_set_statistics
pegs_write_phewas_statistics = pegs_utils_mod.write_phewas_statistics
pegs_parse_gene_map_file = pegs_utils_mod.parse_gene_map_file
pegs_read_loc_file_with_gene_map = pegs_utils_mod.read_loc_file_with_gene_map
pegs_clean_chrom_name = pegs_io_common.clean_chrom_name
pegs_infer_columns_from_table_file = pegs_utils_mod.infer_columns_from_table_file
pegs_needs_gwas_column_detection = pegs_utils_mod.needs_gwas_column_detection
pegs_autodetect_gwas_columns = pegs_utils_mod.autodetect_gwas_columns
pegs_complete_p_beta_se = pegs_utils_mod.complete_p_beta_se
pegs_construct_map_to_ind = pegs_utils_mod.construct_map_to_ind
pegs_open_text_with_retry = pegs_utils_mod.open_text_with_retry
pegs_resolve_column_index = pegs_utils_mod.resolve_column_index
pegs_write_bundle_from_specs = pegs_bundle.write_bundle_from_specs
pegs_get_tar_write_mode_for_bundle_path = pegs_bundle.get_tar_write_mode_for_bundle_path
pegs_write_prefixed_tar_bundle = pegs_bundle.write_prefixed_tar_bundle
pegs_read_prefixed_tar_bundle = pegs_bundle.read_prefixed_tar_bundle
pegs_is_huge_statistics_bundle_path = pegs_bundle.is_huge_statistics_bundle_path
PEGS_EAGGL_BUNDLE_SCHEMA = pegs_bundle.EAGGL_BUNDLE_SCHEMA
write_bundle_from_specs = pegs_bundle.write_bundle_from_specs
EAGGL_BUNDLE_SCHEMA = pegs_bundle.EAGGL_BUNDLE_SCHEMA
pegs_load_aligned_gene_bfs = pegs_gene_io.load_aligned_gene_bfs
pegs_load_aligned_gene_covariates = pegs_gene_io.load_aligned_gene_covariates
pegs_apply_huge_statistics_meta_to_runtime = pegs_huge_cache.apply_huge_statistics_meta_to_runtime
pegs_build_huge_statistics_matrix_row_genes = pegs_huge_cache.build_huge_statistics_matrix_row_genes
pegs_build_huge_statistics_meta = pegs_huge_cache.build_huge_statistics_meta
pegs_build_huge_statistics_score_maps = pegs_huge_cache.build_huge_statistics_score_maps
pegs_coerce_runtime_state_dict = pegs_huge_cache.coerce_runtime_state_dict
pegs_combine_runtime_huge_scores = pegs_huge_cache.combine_runtime_huge_scores
pegs_get_huge_statistics_paths_for_prefix = pegs_huge_cache.get_huge_statistics_paths_for_prefix
pegs_load_huge_statistics_sparse_and_vectors = pegs_huge_cache.load_huge_statistics_sparse_and_vectors
pegs_read_huge_statistics_covariates_if_present = pegs_huge_cache.read_huge_statistics_covariates_if_present
pegs_read_huge_statistics_text_tables = pegs_huge_cache.read_huge_statistics_text_tables
pegs_read_numeric_vector_file = pegs_huge_cache.read_numeric_vector_file
pegs_resolve_huge_statistics_gene_vectors = pegs_huge_cache.resolve_huge_statistics_gene_vectors
pegs_validate_huge_statistics_loaded_shapes = pegs_huge_cache.validate_huge_statistics_loaded_shapes
pegs_write_huge_statistics_runtime_vectors = pegs_huge_cache.write_huge_statistics_runtime_vectors
pegs_write_huge_statistics_sparse_components = pegs_huge_cache.write_huge_statistics_sparse_components
pegs_write_huge_statistics_text_tables = pegs_huge_cache.write_huge_statistics_text_tables
pegs_write_numeric_vector_file = pegs_huge_cache.write_numeric_vector_file
pegs_accumulate_standard_phewas_outputs = pegs_phewas.accumulate_standard_phewas_outputs
pegs_append_phewas_metric_block = pegs_phewas.append_phewas_metric_block
pegs_prepare_phewas_phenos_from_file = pegs_phewas.prepare_phewas_phenos_from_file
pegs_read_phewas_file_batch = pegs_phewas.read_phewas_file_batch
pegs_build_phewas_stage_config = pegs_phewas.build_phewas_stage_config
pegs_resolve_gene_phewas_input_decision_for_stage = pegs_phewas.resolve_gene_phewas_input_decision_for_stage
build_phewas_stage_config = pegs_phewas.build_phewas_stage_config
resolve_gene_phewas_input_decision_for_stage = pegs_phewas.resolve_gene_phewas_input_decision_for_stage
load_and_apply_gene_set_statistics_to_runtime = pegs_utils_mod.load_and_apply_gene_set_statistics_to_runtime
pegs_finalize_regression_outputs = pegs_regression.finalize_regression_outputs
pegs_compute_beta_tildes = pegs_regression.compute_beta_tildes
pegs_compute_logistic_beta_tildes = pegs_regression.compute_logistic_beta_tildes
pegs_correct_beta_tildes = pegs_regression.correct_beta_tildes
pegs_compute_multivariate_beta_tildes = pegs_regression.compute_multivariate_beta_tildes
pegs_compute_banded_y_corr_cholesky = pegs_runtime_matrix.compute_banded_y_corr_cholesky
pegs_whiten_matrix_with_banded_cholesky = pegs_runtime_matrix.whiten_matrix_with_banded_cholesky
pegs_calc_shift_scale_for_dense_block = pegs_runtime_matrix.calc_shift_scale_for_dense_block
pegs_calc_X_shift_scale = pegs_runtime_matrix.calc_X_shift_scale
pegs_calculate_V_internal = pegs_runtime_matrix.calculate_V_internal
pegs_set_runtime_x_from_inputs = pegs_runtime_matrix.set_runtime_x_from_inputs
pegs_get_num_X_blocks = pegs_runtime_matrix.get_num_X_blocks
pegs_iterate_X_blocks_internal = pegs_runtime_matrix.iterate_X_blocks_internal
pegs_set_runtime_p = pegs_runtime_matrix.set_runtime_p
pegs_set_runtime_sigma = pegs_runtime_matrix.set_runtime_sigma
_temporary_state_fields = pigean_runtime.temporary_state_fields
_STATE_FIELDS_SAMPLER_HYPER = pigean_runtime.STATE_FIELDS_SAMPLER_HYPER
_open_optional_inner_betas_trace_file = None
_close_optional_inner_betas_trace_file = pigean_runtime.close_optional_inner_betas_trace_file
_return_inner_betas_result = pigean_runtime.return_inner_betas_result
_temporary_unsubset_gene_sets = pigean_runtime.temporary_unsubset_gene_sets
_maybe_unsubset_gene_sets = pigean_runtime.maybe_unsubset_gene_sets
_restore_subset_gene_sets = pigean_runtime.restore_subset_gene_sets


@dataclass(frozen=True)
class PigeanMainServices:
    sys: object
    np: object
    scipy: object
    NONE: int
    INFO: int
    DEBUG: int
    TRACE: int
    log_fn: object
    warn_fn: object
    bail_fn: object

    def log(self, *args, **kwargs):
        return self.log_fn(*args, **kwargs)

    def warn(self, *args, **kwargs):
        return self.warn_fn(*args, **kwargs)

    def bail(self, message):
        return self.bail_fn(message)


def _sync_cli_state() -> None:
    global options, args, mode, config_mode, cli_specified_dests, config_specified_dests
    global NONE, INFO, DEBUG, TRACE, debug_level, log_fh, warnings_fh, log, warn

    options = pigean_cli.options
    args = pigean_cli.args
    mode = pigean_cli.mode
    config_mode = pigean_cli.config_mode
    cli_specified_dests = pigean_cli.cli_specified_dests
    config_specified_dests = pigean_cli.config_specified_dests
    NONE = pigean_cli.NONE
    INFO = pigean_cli.INFO
    DEBUG = pigean_cli.DEBUG
    TRACE = pigean_cli.TRACE
    debug_level = pigean_cli.debug_level
    log_fh = pigean_cli.log_fh
    warnings_fh = pigean_cli.warnings_fh
    log = pigean_cli.log
    warn = pigean_cli.warn


def bail(message):
    raise DataValidationError(message)


def build_cli_services() -> PigeanMainServices:
    _sync_cli_state()
    return PigeanMainServices(
        sys=sys,
        np=np,
        scipy=scipy,
        NONE=NONE,
        INFO=INFO,
        DEBUG=DEBUG,
        TRACE=TRACE,
        log_fn=log,
        warn_fn=warn,
        bail_fn=bail,
    )


def build_mode_state(mode, run_phewas_from_gene_phewas_stats_in):
    return pigean_cli._build_mode_state(mode, run_phewas_from_gene_phewas_stats_in)


def build_runtime_state(options):
    _sync_cli_state()
    pigean_state.configure_runtime_context(cli_module=pigean_cli)
    return pigean_runtime.build_runtime_state(pigean_state.PigeanState, options)


def configure_hyperparameters_for_main(state, options):
    return pigean_runtime.configure_hyperparameters_for_main(
        state,
        options,
        read_gene_map_fn=functools.partial(
            pigean_y_inputs_core.read_gene_map,
            bail_fn=bail,
        ),
        init_gene_locs_fn=functools.partial(
            pigean_y_inputs_core.init_gene_locs,
            warn_fn=warn,
            bail_fn=bail,
            log_fn=log,
        ),
        bail_fn=bail,
        log_fn=log,
    )


def build_inner_beta_sampler_common_kwargs(options):
    return pigean_model.build_inner_beta_sampler_common_kwargs(options)


def open_gz(file, flag=None):
    return pegs_open_text_with_retry(
        file,
        flag=flag,
        log_fn=lambda message: log(message, INFO),
        bail_fn=bail,
    )


def _open_optional_inner_betas_trace_file(betas_trace_out):
    return pigean_runtime.open_optional_inner_betas_trace_file(
        betas_trace_out,
        open_gz=open_gz,
    )


def get_col(*args, **kwargs):
    return pigean_huge.get_col(*args, bail_fn=bail, **kwargs)


def _determine_columns_from_file(*args, **kwargs):
    return pigean_huge.determine_columns_from_file(
        *args,
        open_gz_fn=open_gz,
        log_fn=lambda message: log(message),
        bail_fn=bail,
        **kwargs,
    )


YPrimaryInputsContract = pigean_y_inputs.YPrimaryInputsContract
YReadContract = pigean_y_inputs.YReadContract
_build_main_y_read_contract = pigean_y_inputs.build_main_y_read_contract
_read_Y = functools.partial(
    pigean_y_inputs_core.read_y_pipeline,
    warn_fn=warn,
    bail_fn=bail,
    log_fn=log,
    trace_level=TRACE,
    apply_gene_covariates_and_correct_huge_fn=functools.partial(
        pigean_y_inputs_core.apply_gene_covariates_and_correct_huge,
        log_fn=log,
        trace_level=TRACE,
        bail_fn=bail,
    ),
)
_read_Y_from_contract = functools.partial(
    pigean_y_inputs_core.read_y_from_contract,
    read_y_fn=_read_Y,
    bail_fn=bail,
)
_run_read_y_stage = _read_Y
_run_read_y_contract_stage = _read_Y_from_contract
_set_const_Y = pigean_y_inputs_core.set_const_Y
set_const_Y = pigean_y_inputs_core.set_const_Y


def load_main_y_inputs(state, options, mode_state):
    return pigean_y_inputs.load_main_y_inputs(
        state,
        options,
        mode_state,
        run_read_y_stage_fn=_run_read_y_stage,
        run_read_y_contract_stage_fn=_run_read_y_contract_stage,
        read_gene_phewas_bfs_fn=read_gene_phewas_bfs,
        bail_fn=bail,
    )


def run_main_adaptive_read_x(state, options, mode_state, sigma2_cond):
    return pigean_x_inputs.run_main_adaptive_read_x(
        state,
        options,
        mode_state,
        sigma2_cond,
        build_xin_to_p_noninf_index_map_fn=pegs_build_xin_to_p_noninf_index_map,
        run_read_x_stage_fn=_run_read_x_stage,
        warn_fn=warn,
        bail_fn=bail,
        log_fn=log,
    )


def _run_read_x_stage(runtime, X_in, **read_x_kwargs):
    return pigean_x_inputs.run_read_x_stage(
        runtime,
        X_in,
        read_x_kwargs=read_x_kwargs,
        build_read_x_pipeline_config_fn=pegs_build_read_x_pipeline_config,
        bail_fn=bail,
        read_x_pipeline_fn=_read_x_pipeline,
    )


def _read_x_pipeline(runtime, read_x_pipeline_config):
    return pigean_x_inputs.read_x_pipeline(
        runtime,
        read_x_pipeline_config,
        open_gz_fn=open_gz,
        open_dense_fn=open,
        log_fn=log,
        info_level=INFO,
        debug_level=DEBUG,
        remove_tag_from_input_fn=pegs_remove_tag_from_input,
        record_read_x_counts_fn=pegs_record_read_x_counts,
        ensure_gene_universe_fn=_ensure_gene_universe_for_x,
        process_x_input_file_fn=_process_x_input_file,
        normalize_dense_gene_rows_fn=_normalize_dense_gene_rows,
        build_sparse_x_from_dense_input_fn=_build_sparse_x_from_dense_input,
        reindex_x_rows_to_current_genes_fn=_reindex_x_rows_to_current_genes,
        normalize_gene_set_weights_fn=_normalize_gene_set_weights,
        partition_missing_gene_rows_fn=_partition_missing_gene_rows,
        maybe_permute_gene_set_rows_fn=_maybe_permute_gene_set_rows,
        maybe_prefilter_x_block_fn=_maybe_prefilter_x_block,
        merge_missing_gene_rows_fn=_merge_missing_gene_rows,
        finalize_added_x_block_fn=_finalize_added_x_block,
        standardize_qc_metrics_after_x_read_fn=pegs_standardize_qc_metrics_after_x_read,
        maybe_correct_gene_set_betas_after_x_read_fn=pegs_maybe_correct_gene_set_betas_after_x_read,
        maybe_limit_initial_gene_sets_by_p_fn=pegs_maybe_limit_initial_gene_sets_by_p,
        maybe_prune_gene_sets_after_x_read_fn=pegs_maybe_prune_gene_sets_after_x_read,
        initialize_hyper_defaults_after_x_read_fn=pegs_initialize_hyper_defaults_after_x_read,
        maybe_learn_batch_hyper_after_x_read_fn=pegs_maybe_learn_batch_hyper_after_x_read,
        maybe_adjust_overaggressive_p_filter_after_x_read_fn=pegs_maybe_adjust_overaggressive_p_filter_after_x_read,
        apply_post_read_gene_set_size_and_qc_filters_fn=pegs_apply_post_read_gene_set_size_and_qc_filters,
        maybe_filter_zero_uncorrected_betas_after_x_read_fn=pigean_state._maybe_filter_zero_uncorrected_betas_after_x_read,
        maybe_reduce_gene_sets_to_max_after_x_read_fn=pigean_state._maybe_reduce_gene_sets_to_max_after_x_read,
    )


_normalize_dense_gene_rows = pigean_x_inputs_core.normalize_dense_gene_rows
_build_sparse_x_from_dense_input = functools.partial(
    pigean_x_inputs_core.build_sparse_x_from_dense_input,
    log_fn=log,
    debug_level=DEBUG,
    warn_fn=warn,
    bail_fn=bail,
    ext_tag="ext",
    bot_tag="bot",
    top_tag="top",
)
_process_x_input_file = functools.partial(
    pigean_x_inputs_core.process_x_input_file,
    open_gz_fn=open_gz,
    warn_fn=warn,
    bail_fn=bail,
    log_fn=log,
    debug_level=DEBUG,
    ext_tag="ext",
    top_tag="top",
    bot_tag="bot",
)
_normalize_gene_set_weights = pigean_x_inputs_core.normalize_gene_set_weights
_maybe_permute_gene_set_rows = pigean_x_inputs_core.maybe_permute_gene_set_rows
_build_prefilter_keep_mask = functools.partial(
    pigean_x_inputs_core.build_prefilter_keep_mask,
    log_fn=log,
    debug_level=DEBUG,
)
_maybe_prefilter_x_block = functools.partial(
    pigean_x_inputs_core.maybe_prefilter_x_block,
    log_fn=log,
    debug_level=DEBUG,
)
_merge_missing_gene_rows = pigean_x_inputs_core.merge_missing_gene_rows
_finalize_added_x_block = pigean_x_inputs_core.finalize_added_x_block
_partition_missing_gene_rows = functools.partial(
    pigean_x_inputs_core.partition_missing_gene_rows,
    bail_fn=bail,
)
_reindex_x_rows_to_current_genes = pigean_x_inputs_core.reindex_x_rows_to_current_genes
_ensure_gene_universe_for_x = functools.partial(
    pigean_x_inputs_core.ensure_gene_universe_for_x,
    open_gz_fn=open_gz,
    remove_tag_from_input_fn=pegs_remove_tag_from_input,
    log_fn=log,
    debug_level=DEBUG,
    bail_fn=bail,
)


def read_gene_phewas_bfs(
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
        get_col_fn=get_col,
        construct_map_to_ind_fn=pegs_construct_map_to_ind,
        warn_fn=warn,
        bail_fn=bail,
        log_fn=lambda message: log(message, DEBUG),
    )
    state.phewas_state = pegs_sync_phewas_runtime_state(state)


def _reread_gene_phewas_bfs(state):
    cached_call = getattr(state, "cached_gene_phewas_call", None)
    if cached_call is None:
        return
    read_gene_phewas_bfs(state, **cached_call)


def run_advanced_set_b_output_phewas_if_requested(state, options):
    return pigean_phewas.run_advanced_set_b_output_phewas_if_requested(
        build_cli_services(),
        state,
        options,
    )


pegs_initialize_matrix_and_gene_index_state = pegs_xdata.initialize_matrix_and_gene_index_state
pegs_sync_runtime_state_bundle = pegs_ydata.sync_runtime_state_bundle
pegs_sync_phewas_runtime_state = pegs_ydata.sync_phewas_runtime_state
pegs_build_read_x_pipeline_config = pegs_xdata.build_read_x_pipeline_config
pegs_build_xin_to_p_noninf_index_map = pegs_utils_mod.build_xin_to_p_noninf_index_map
pegs_remove_tag_from_input = pegs_utils_mod.remove_tag_from_input
pegs_record_read_x_counts = pegs_utils_mod.record_read_x_counts
pegs_standardize_qc_metrics_after_x_read = pegs_utils_mod.standardize_qc_metrics_after_x_read
pegs_maybe_correct_gene_set_betas_after_x_read = pegs_utils_mod.maybe_correct_gene_set_betas_after_x_read
pegs_maybe_limit_initial_gene_sets_by_p = pegs_utils_mod.maybe_limit_initial_gene_sets_by_p
pegs_maybe_prune_gene_sets_after_x_read = pegs_utils_mod.maybe_prune_gene_sets_after_x_read
pegs_initialize_hyper_defaults_after_x_read = pegs_utils_mod.initialize_hyper_defaults_after_x_read
pegs_maybe_adjust_overaggressive_p_filter_after_x_read = pegs_utils_mod.maybe_adjust_overaggressive_p_filter_after_x_read
pegs_apply_post_read_gene_set_size_and_qc_filters = pegs_utils_mod.apply_post_read_gene_set_size_and_qc_filters
pegs_load_and_apply_gene_phewas_bfs_to_runtime = pegs_utils_mod.load_and_apply_gene_phewas_bfs_to_runtime
pegs_load_and_apply_gene_set_statistics_to_runtime = pegs_utils_mod.load_and_apply_gene_set_statistics_to_runtime
pegs_set_runtime_y_from_inputs = pegs_utils_mod.set_runtime_y_from_inputs
pegs_write_gene_set_statistics = pegs_utils_mod.write_gene_set_statistics
pegs_write_phewas_gene_set_statistics = pegs_utils_mod.write_phewas_gene_set_statistics
pegs_write_gene_statistics = pegs_utils_mod.write_gene_statistics
pegs_write_gene_gene_set_statistics = pegs_utils_mod.write_gene_gene_set_statistics
pegs_write_phewas_statistics = pegs_utils_mod.write_phewas_statistics
pegs_parse_gene_map_file = pegs_utils_mod.parse_gene_map_file
pegs_read_loc_file_with_gene_map = pegs_utils_mod.read_loc_file_with_gene_map
pegs_infer_columns_from_table_file = pegs_utils_mod.infer_columns_from_table_file
pegs_needs_gwas_column_detection = pegs_utils_mod.needs_gwas_column_detection
pegs_autodetect_gwas_columns = pegs_utils_mod.autodetect_gwas_columns
pegs_complete_p_beta_se = pegs_utils_mod.complete_p_beta_se
pegs_construct_map_to_ind = pegs_utils_mod.construct_map_to_ind
pegs_open_text_with_retry = pegs_utils_mod.open_text_with_retry
pegs_resolve_column_index = pegs_utils_mod.resolve_column_index
pegs_write_bundle_from_specs = pegs_bundle.write_bundle_from_specs
pegs_get_tar_write_mode_for_bundle_path = pegs_bundle.get_tar_write_mode_for_bundle_path
pegs_write_prefixed_tar_bundle = pegs_bundle.write_prefixed_tar_bundle
pegs_read_prefixed_tar_bundle = pegs_bundle.read_prefixed_tar_bundle
PEGS_EAGGL_BUNDLE_SCHEMA = pegs_bundle.EAGGL_BUNDLE_SCHEMA
pegs_load_aligned_gene_bfs = pegs_gene_io.load_aligned_gene_bfs
pegs_apply_huge_statistics_meta_to_runtime = pegs_huge_cache.apply_huge_statistics_meta_to_runtime
pegs_build_huge_statistics_matrix_row_genes = pegs_huge_cache.build_huge_statistics_matrix_row_genes
pegs_build_huge_statistics_meta = pegs_huge_cache.build_huge_statistics_meta
pegs_build_huge_statistics_score_maps = pegs_huge_cache.build_huge_statistics_score_maps
pegs_coerce_runtime_state_dict = pegs_huge_cache.coerce_runtime_state_dict
pegs_combine_runtime_huge_scores = pegs_huge_cache.combine_runtime_huge_scores
pegs_get_huge_statistics_paths_for_prefix = pegs_huge_cache.get_huge_statistics_paths_for_prefix
pegs_load_huge_statistics_sparse_and_vectors = pegs_huge_cache.load_huge_statistics_sparse_and_vectors
pegs_read_huge_statistics_covariates_if_present = pegs_huge_cache.read_huge_statistics_covariates_if_present
pegs_read_huge_statistics_text_tables = pegs_huge_cache.read_huge_statistics_text_tables
pegs_read_numeric_vector_file = pegs_huge_cache.read_numeric_vector_file
pegs_resolve_huge_statistics_gene_vectors = pegs_huge_cache.resolve_huge_statistics_gene_vectors
pegs_validate_huge_statistics_loaded_shapes = pegs_huge_cache.validate_huge_statistics_loaded_shapes
pegs_write_huge_statistics_runtime_vectors = pegs_huge_cache.write_huge_statistics_runtime_vectors
pegs_write_huge_statistics_sparse_components = pegs_huge_cache.write_huge_statistics_sparse_components
pegs_write_huge_statistics_text_tables = pegs_huge_cache.write_huge_statistics_text_tables
pegs_write_numeric_vector_file = pegs_huge_cache.write_numeric_vector_file
pegs_accumulate_standard_phewas_outputs = pegs_phewas.accumulate_standard_phewas_outputs
pegs_append_phewas_metric_block = pegs_phewas.append_phewas_metric_block
pegs_prepare_phewas_phenos_from_file = pegs_phewas.prepare_phewas_phenos_from_file
pegs_read_phewas_file_batch = pegs_phewas.read_phewas_file_batch
pegs_build_phewas_stage_config = pegs_phewas.build_phewas_stage_config
pegs_resolve_gene_phewas_input_decision_for_stage = pegs_phewas.resolve_gene_phewas_input_decision_for_stage
pegs_finalize_regression_outputs = pegs_regression.finalize_regression_outputs
pegs_compute_beta_tildes = pegs_regression.compute_beta_tildes
pegs_compute_logistic_beta_tildes = pegs_regression.compute_logistic_beta_tildes
pegs_correct_beta_tildes = pegs_regression.correct_beta_tildes
pegs_compute_multivariate_beta_tildes = pegs_regression.compute_multivariate_beta_tildes
pegs_compute_banded_y_corr_cholesky = pegs_runtime_matrix.compute_banded_y_corr_cholesky
pegs_whiten_matrix_with_banded_cholesky = pegs_runtime_matrix.whiten_matrix_with_banded_cholesky
pegs_calc_shift_scale_for_dense_block = pegs_runtime_matrix.calc_shift_scale_for_dense_block
pegs_calc_X_shift_scale = pegs_runtime_matrix.calc_X_shift_scale
pegs_calculate_V_internal = pegs_runtime_matrix.calculate_V_internal
pegs_set_runtime_x_from_inputs = pegs_runtime_matrix.set_runtime_x_from_inputs
pegs_get_num_X_blocks = pegs_runtime_matrix.get_num_X_blocks
pegs_iterate_X_blocks_internal = pegs_runtime_matrix.iterate_X_blocks_internal
pegs_set_runtime_p = pegs_runtime_matrix.set_runtime_p
pegs_set_runtime_sigma = pegs_runtime_matrix.set_runtime_sigma
