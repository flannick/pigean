from __future__ import annotations

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
import pegs_shared.gene_io as pegs_gene_io
import pegs_shared.huge_cache as pegs_huge_cache
import pegs_shared.io_common as pegs_io_common
import pegs_shared.phewas as pegs_phewas
import pegs_shared.regression as pegs_regression
import pegs_shared.runtime_matrix as pegs_runtime_matrix
import pegs_shared.xdata as pegs_xdata
import pegs_shared.ydata as pegs_ydata
import pegs_utils as pegs_utils_mod

from pigean import gibbs as pigean_gibbs
from pigean import model as pigean_model
from pigean import phewas_io as pigean_phewas_io
from pigean import runtime as pigean_runtime
from pigean import y_inputs as pigean_y_inputs
from pigean import y_inputs_core as pigean_y_inputs_core

pegs_initialize_matrix_and_gene_index_state = pegs_xdata.initialize_matrix_and_gene_index_state
pegs_sync_runtime_state_bundle = pegs_ydata.sync_runtime_state_bundle
pegs_sync_phewas_runtime_state = pegs_ydata.sync_phewas_runtime_state
pegs_apply_post_read_gene_set_size_and_qc_filters = pegs_utils_mod.apply_post_read_gene_set_size_and_qc_filters
pegs_calc_X_shift_scale = pegs_runtime_matrix.calc_X_shift_scale
pegs_calc_shift_scale_for_dense_block = pegs_runtime_matrix.calc_shift_scale_for_dense_block
pegs_calculate_V_internal = pegs_runtime_matrix.calculate_V_internal
pegs_clean_chrom_name = pegs_io_common.clean_chrom_name
pegs_complete_p_beta_se = pegs_utils_mod.complete_p_beta_se
pegs_coerce_runtime_state_dict = pegs_huge_cache.coerce_runtime_state_dict
pegs_compute_banded_y_corr_cholesky = pegs_runtime_matrix.compute_banded_y_corr_cholesky
pegs_compute_beta_tildes = pegs_regression.compute_beta_tildes
pegs_compute_logistic_beta_tildes = pegs_regression.compute_logistic_beta_tildes
pegs_compute_multivariate_beta_tildes = pegs_regression.compute_multivariate_beta_tildes
pegs_construct_map_to_ind = pegs_utils_mod.construct_map_to_ind
pegs_correct_beta_tildes = pegs_regression.correct_beta_tildes
pegs_finalize_regression_outputs = pegs_regression.finalize_regression_outputs
pegs_get_num_X_blocks = pegs_runtime_matrix.get_num_X_blocks
pegs_apply_huge_statistics_meta_to_runtime = pegs_huge_cache.apply_huge_statistics_meta_to_runtime
pegs_build_huge_statistics_matrix_row_genes = pegs_huge_cache.build_huge_statistics_matrix_row_genes
pegs_build_huge_statistics_meta = pegs_huge_cache.build_huge_statistics_meta
pegs_build_huge_statistics_score_maps = pegs_huge_cache.build_huge_statistics_score_maps
pegs_combine_runtime_huge_scores = pegs_huge_cache.combine_runtime_huge_scores
pegs_get_huge_statistics_paths_for_prefix = pegs_huge_cache.get_huge_statistics_paths_for_prefix
pegs_infer_columns_from_table_file = pegs_utils_mod.infer_columns_from_table_file
pegs_initialize_hyper_defaults_after_x_read = pegs_utils_mod.initialize_hyper_defaults_after_x_read
pegs_is_huge_statistics_bundle_path = pegs_bundle.is_huge_statistics_bundle_path
pegs_iterate_X_blocks_internal = pegs_runtime_matrix.iterate_X_blocks_internal
pegs_load_aligned_gene_bfs = pegs_gene_io.load_aligned_gene_bfs
pegs_load_aligned_gene_covariates = pegs_gene_io.load_aligned_gene_covariates
pegs_load_huge_statistics_sparse_and_vectors = pegs_huge_cache.load_huge_statistics_sparse_and_vectors
pegs_maybe_adjust_overaggressive_p_filter_after_x_read = pegs_utils_mod.maybe_adjust_overaggressive_p_filter_after_x_read
pegs_maybe_correct_gene_set_betas_after_x_read = pegs_utils_mod.maybe_correct_gene_set_betas_after_x_read
pegs_maybe_limit_initial_gene_sets_by_p = pegs_utils_mod.maybe_limit_initial_gene_sets_by_p
pegs_maybe_prune_gene_sets_after_x_read = pegs_utils_mod.maybe_prune_gene_sets_after_x_read
pegs_prepare_phewas_phenos_from_file = pegs_phewas.prepare_phewas_phenos_from_file
pegs_read_huge_statistics_covariates_if_present = pegs_huge_cache.read_huge_statistics_covariates_if_present
pegs_read_huge_statistics_text_tables = pegs_huge_cache.read_huge_statistics_text_tables
pegs_parse_gene_map_file = pegs_io_common.parse_gene_map_file
pegs_read_loc_file_with_gene_map = pegs_utils_mod.read_loc_file_with_gene_map
pegs_read_numeric_vector_file = pegs_huge_cache.read_numeric_vector_file
pegs_read_phewas_file_batch = pegs_phewas.read_phewas_file_batch
pegs_read_prefixed_tar_bundle = pegs_bundle.read_prefixed_tar_bundle
pegs_autodetect_gwas_columns = pegs_utils_mod.autodetect_gwas_columns
pegs_needs_gwas_column_detection = pegs_utils_mod.needs_gwas_column_detection
pegs_resolve_huge_statistics_gene_vectors = pegs_huge_cache.resolve_huge_statistics_gene_vectors
pegs_resolve_column_index = pegs_utils_mod.resolve_column_index
pegs_set_runtime_p = pegs_runtime_matrix.set_runtime_p
pegs_set_runtime_sigma = pegs_runtime_matrix.set_runtime_sigma
pegs_set_runtime_x_from_inputs = pegs_runtime_matrix.set_runtime_x_from_inputs
pegs_set_runtime_y_from_inputs = pegs_utils_mod.set_runtime_y_from_inputs
pegs_standardize_qc_metrics_after_x_read = pegs_utils_mod.standardize_qc_metrics_after_x_read
pegs_validate_huge_statistics_loaded_shapes = pegs_huge_cache.validate_huge_statistics_loaded_shapes
pegs_whiten_matrix_with_banded_cholesky = pegs_runtime_matrix.whiten_matrix_with_banded_cholesky
pegs_write_gene_gene_set_statistics = pegs_utils_mod.write_gene_gene_set_statistics
pegs_write_gene_set_statistics = pegs_utils_mod.write_gene_set_statistics
pegs_write_gene_statistics = pegs_utils_mod.write_gene_statistics
pegs_write_huge_statistics_runtime_vectors = pegs_huge_cache.write_huge_statistics_runtime_vectors
pegs_write_huge_statistics_sparse_components = pegs_huge_cache.write_huge_statistics_sparse_components
pegs_write_huge_statistics_text_tables = pegs_huge_cache.write_huge_statistics_text_tables
pegs_write_numeric_vector_file = pegs_huge_cache.write_numeric_vector_file
pegs_write_phewas_gene_set_statistics = pegs_utils_mod.write_phewas_gene_set_statistics
pegs_write_phewas_statistics = pegs_utils_mod.write_phewas_statistics
pegs_write_prefixed_tar_bundle = pegs_bundle.write_prefixed_tar_bundle
pegs_accumulate_standard_phewas_outputs = pegs_phewas.accumulate_standard_phewas_outputs
pegs_append_phewas_metric_block = pegs_phewas.append_phewas_metric_block
_temporary_state_fields = pigean_runtime.temporary_state_fields
_STATE_FIELDS_SAMPLER_HYPER = pigean_runtime.STATE_FIELDS_SAMPLER_HYPER
_open_optional_inner_betas_trace_file = None
_return_inner_betas_result = pigean_runtime.return_inner_betas_result
_maybe_unsubset_gene_sets = pigean_runtime.maybe_unsubset_gene_sets

NONE = 0
INFO = 1
DEBUG = 2
TRACE = 3
options = None
args = []
mode = None
config_mode = None
cli_specified_dests = set()
config_specified_dests = set()
debug_level = 1
log_fh = None
warnings_fh = None
_json_safe = pegs_utils_mod.json_safe


def log(*_args, **_kwargs):
    return None


def warn(*_args, **_kwargs):
    return None


def bail(message):
    raise DataValidationError(message)


def configure_runtime_context(*, cli_module=None):
    global options, args, mode, config_mode, cli_specified_dests, config_specified_dests
    global NONE, INFO, DEBUG, TRACE, debug_level, log_fh, warnings_fh, log, warn, _json_safe

    if cli_module is None:
        from . import cli as pigean_cli

        cli_module = pigean_cli

    options = cli_module.options
    args = cli_module.args
    mode = cli_module.mode
    config_mode = cli_module.config_mode
    cli_specified_dests = cli_module.cli_specified_dests
    config_specified_dests = cli_module.config_specified_dests
    NONE = cli_module.NONE
    INFO = cli_module.INFO
    DEBUG = cli_module.DEBUG
    TRACE = cli_module.TRACE
    debug_level = cli_module.debug_level
    log_fh = cli_module.log_fh
    warnings_fh = cli_module.warnings_fh
    log = cli_module.log
    warn = cli_module.warn
    _json_safe = cli_module._json_safe


def open_gz(file, flag=None):
    return pegs_utils_mod.open_text_with_retry(
        file,
        flag=flag,
        log_fn=lambda message: log(message, INFO),
        bail_fn=bail,
    )


_open_optional_inner_betas_trace_file = functools.partial(
    pigean_runtime.open_optional_inner_betas_trace_file,
    open_gz=open_gz,
)


@functools.lru_cache(maxsize=1)
def _pigean_huge_module():
    from pigean import huge as pigean_huge

    pigean_huge.configure_numpy(np)
    return pigean_huge


def _get_col(*args, **kwargs):
    return _pigean_huge_module().get_col(*args, bail_fn=bail, **kwargs)


def _determine_columns_from_file(*args, **kwargs):
    return _pigean_huge_module().determine_columns_from_file(
        *args,
        open_gz_fn=open_gz,
        log_fn=lambda message: log(message),
        bail_fn=bail,
        **kwargs,
    )


def _needs_gwas_column_detection(*args, **kwargs):
    return _pigean_huge_module().needs_gwas_column_detection_explicit(
        *args,
        pegs_needs_gwas_column_detection=pegs_needs_gwas_column_detection,
        **kwargs,
    )


def _autodetect_gwas_columns(*args, **kwargs):
    return _pigean_huge_module().autodetect_gwas_columns_explicit(
        *args,
        pegs_autodetect_gwas_columns=pegs_autodetect_gwas_columns,
        infer_columns_fn=_determine_columns_from_file,
        log_fn=log,
        bail_fn=bail,
        **kwargs,
    )


def _load_huge_gene_and_exon_locations(*args, **kwargs):
    return _pigean_huge_module().load_huge_gene_and_exon_locations_explicit(
        *args,
        np_module=np,
        read_loc_file_with_gene_map_fn=pegs_read_loc_file_with_gene_map,
        clean_chrom_fn=pegs_clean_chrom_name,
        log_fn=log,
        warn_fn=warn,
        bail_fn=bail,
        **kwargs,
    )


def _compute_huge_variant_thresholds(*args, **kwargs):
    return _pigean_huge_module().compute_huge_variant_thresholds_explicit(
        *args,
        np_module=np,
        scipy_module=scipy,
        log_fn=log,
        **kwargs,
    )


def _validate_and_normalize_huge_gwas_inputs(*args, **kwargs):
    return _pigean_huge_module().validate_and_normalize_huge_gwas_inputs_explicit(
        *args,
        warn_fn=warn,
        bail_fn=bail,
        **kwargs,
    )


def _write_huge_statistics_bundle(*args, **kwargs):
    return _pigean_huge_module().write_huge_statistics_bundle_explicit(
        *args,
        pegs_coerce_runtime_state_dict=pegs_coerce_runtime_state_dict,
        pegs_get_huge_statistics_paths_for_prefix=pegs_get_huge_statistics_paths_for_prefix,
        pegs_build_huge_statistics_matrix_row_genes=pegs_build_huge_statistics_matrix_row_genes,
        pegs_build_huge_statistics_score_maps=pegs_build_huge_statistics_score_maps,
        pegs_build_huge_statistics_meta=pegs_build_huge_statistics_meta,
        pegs_write_huge_statistics_text_tables=pegs_write_huge_statistics_text_tables,
        pegs_write_huge_statistics_runtime_vectors=pegs_write_huge_statistics_runtime_vectors,
        pegs_write_huge_statistics_sparse_components=pegs_write_huge_statistics_sparse_components,
        pegs_write_numeric_vector_file=pegs_write_numeric_vector_file,
        open_gz_fn=open_gz,
        json_safe_fn=lambda value: _json_safe(value),
        bail_fn=bail,
        **kwargs,
    )


def _read_huge_statistics_bundle(*args, **kwargs):
    return _pigean_huge_module().read_huge_statistics_bundle_explicit(
        *args,
        np_module=np,
        pegs_coerce_runtime_state_dict=pegs_coerce_runtime_state_dict,
        pegs_get_huge_statistics_paths_for_prefix=pegs_get_huge_statistics_paths_for_prefix,
        pegs_read_huge_statistics_text_tables=pegs_read_huge_statistics_text_tables,
        pegs_resolve_huge_statistics_gene_vectors=pegs_resolve_huge_statistics_gene_vectors,
        pegs_load_huge_statistics_sparse_and_vectors=pegs_load_huge_statistics_sparse_and_vectors,
        pegs_apply_huge_statistics_meta_to_runtime=pegs_apply_huge_statistics_meta_to_runtime,
        pegs_read_huge_statistics_covariates_if_present=pegs_read_huge_statistics_covariates_if_present,
        pegs_combine_runtime_huge_scores=pegs_combine_runtime_huge_scores,
        pegs_validate_huge_statistics_loaded_shapes=pegs_validate_huge_statistics_loaded_shapes,
        pegs_read_numeric_vector_file=pegs_read_numeric_vector_file,
        open_gz_fn=open_gz,
        bail_fn=bail,
        **kwargs,
    )


_read_gene_phewas_bfs = functools.partial(
    pigean_phewas_io.read_gene_phewas_bfs,
    parse_gene_map_file_fn=pegs_parse_gene_map_file,
    load_and_apply_gene_phewas_bfs_fn=pegs_utils_mod.load_and_apply_gene_phewas_bfs_to_runtime,
    sync_phewas_runtime_state_fn=pegs_sync_phewas_runtime_state,
    construct_map_to_ind_fn=pegs_construct_map_to_ind,
    open_text_fn=open_gz,
    get_col_fn=_get_col,
    warn_fn=warn,
    bail_fn=bail,
    log_fn=log,
    info_level=INFO,
    debug_level=DEBUG,
)


_reread_gene_phewas_bfs = functools.partial(
    pigean_phewas_io.reread_gene_phewas_bfs,
    read_gene_phewas_bfs_fn=_read_gene_phewas_bfs,
)


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
        return _pigean_huge_module().read_huge_s2g_probabilities(
            self,
            s2g_in,
            seen_chrom_pos,
            hold_out_chrom=hold_out_chrom,
            s2g_chrom_col=s2g_chrom_col,
            s2g_pos_col=s2g_pos_col,
            s2g_gene_col=s2g_gene_col,
            s2g_prob_col=s2g_prob_col,
            s2g_normalize_values=s2g_normalize_values,
            determine_columns_fn=_determine_columns_from_file,
            open_text_fn=open_gz,
            get_col_fn=_get_col,
            clean_chrom_fn=pegs_clean_chrom_name,
            log_fn=log,
            warn_fn=warn,
            bail_fn=bail,
            info_level=INFO,
        )

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
        return _pigean_huge_module().read_huge_input_credible_sets(
            self,
            credible_sets_in,
            seen_chrom_pos,
            chrom_pos_p_beta_se_freq,
            var_p_threshold,
            hold_out_chrom=hold_out_chrom,
            credible_sets_id_col=credible_sets_id_col,
            credible_sets_chrom_col=credible_sets_chrom_col,
            credible_sets_pos_col=credible_sets_pos_col,
            credible_sets_ppa_col=credible_sets_ppa_col,
            determine_columns_fn=_determine_columns_from_file,
            open_text_fn=open_gz,
            get_col_fn=_get_col,
            clean_chrom_fn=pegs_clean_chrom_name,
            log_fn=log,
            warn_fn=warn,
            bail_fn=bail,
            info_level=INFO,
        )

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
            tree = _pigean_huge_module().IntervalTree([(x - indep_window, x + indep_window) for x in cur_pos])
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
        return pigean_model.calculate_gene_set_statistics(
            self,
            gwas_in=gwas_in,
            exomes_in=exomes_in,
            positive_controls_in=positive_controls_in,
            positive_controls_list=positive_controls_list,
            case_counts_in=case_counts_in,
            ctrl_counts_in=ctrl_counts_in,
            gene_bfs_in=gene_bfs_in,
            Y=Y,
            show_progress=show_progress,
            max_gene_set_p=max_gene_set_p,
            run_logistic=run_logistic,
            max_for_linear=max_for_linear,
            run_corrected_ols=run_corrected_ols,
            use_sampling_for_betas=use_sampling_for_betas,
            correct_betas_mean=correct_betas_mean,
            correct_betas_var=correct_betas_var,
            gene_loc_file=gene_loc_file,
            gene_cor_file=gene_cor_file,
            gene_cor_file_gene_col=gene_cor_file_gene_col,
            gene_cor_file_cor_start_col=gene_cor_file_cor_start_col,
            skip_V=skip_V,
            run_using_phewas=run_using_phewas,
            bail_fn=bail,
            warn_fn=warn,
            log_fn=log,
            info_level=INFO,
            debug_level=DEBUG,
            trace_level=TRACE,
            run_read_y_stage_fn=_run_read_y_stage,
            **kwargs
        )

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
        return pigean_model.calculate_non_inf_betas(
            self,
            p,
            max_num_burn_in=max_num_burn_in,
            max_num_iter=max_num_iter,
            min_num_iter=min_num_iter,
            num_chains=num_chains,
            r_threshold_burn_in=r_threshold_burn_in,
            use_max_r_for_convergence=use_max_r_for_convergence,
            max_frac_sem=max_frac_sem,
            gauss_seidel=gauss_seidel,
            update_hyper_sigma=update_hyper_sigma,
            update_hyper_p=update_hyper_p,
            sparse_solution=sparse_solution,
            pre_filter_batch_size=pre_filter_batch_size,
            pre_filter_small_batch_size=pre_filter_small_batch_size,
            sparse_frac_betas=sparse_frac_betas,
            betas_trace_out=betas_trace_out,
            run_betas_using_phewas=run_betas_using_phewas,
            run_uncorrected_using_phewas=run_uncorrected_using_phewas,
            bail_fn=bail,
            warn_fn=warn,
            log_fn=log,
            info_level=INFO,
            debug_level=DEBUG,
            trace_level=TRACE,
            **kwargs
        )

    # ==========================================================================
    # Section: Core Inference Orchestration (priors + outer Gibbs).
    # ==========================================================================
    def calculate_priors(self, max_gene_set_p=None, num_gene_batches=None, correct_betas_mean=True, correct_betas_var=True, gene_loc_file=None, gene_cor_file=None, gene_cor_file_gene_col=1, gene_cor_file_cor_start_col=10, p_noninf=None, run_logistic=True, max_for_linear=0.95, adjust_priors=False, tag="", **kwargs):
        return pigean_model.calculate_priors(
            self,
            max_gene_set_p=max_gene_set_p,
            num_gene_batches=num_gene_batches,
            correct_betas_mean=correct_betas_mean,
            correct_betas_var=correct_betas_var,
            gene_loc_file=gene_loc_file,
            gene_cor_file=gene_cor_file,
            gene_cor_file_gene_col=gene_cor_file_gene_col,
            gene_cor_file_cor_start_col=gene_cor_file_cor_start_col,
            p_noninf=p_noninf,
            run_logistic=run_logistic,
            max_for_linear=max_for_linear,
            adjust_priors=adjust_priors,
            tag=tag,
            bail_fn=bail,
            warn_fn=warn,
            log_fn=log,
            info_level=INFO,
            debug_level=DEBUG,
            trace_level=TRACE,
            **kwargs
        )

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
        from pigean import gibbs as pigean_gibbs
        from pigean import gibbs_callbacks as pigean_gibbs_callbacks

        callbacks = pigean_gibbs_callbacks.build_gibbs_callbacks(
            sys.modules[__name__],
            open_gz_fn=open_gz,
            log_fn=log,
            bail_fn=bail,
            info_level=INFO,
        )
        return pigean_gibbs.run_outer_gibbs(
            self,
            callbacks,
            max_num_iter=max_num_iter,
            total_num_iter=total_num_iter,
            max_num_restarts=max_num_restarts,
            num_chains=num_chains,
            num_mad=num_mad,
            r_threshold_burn_in=r_threshold_burn_in,
            use_max_r_for_convergence=use_max_r_for_convergence,
            increase_hyper_if_betas_below=increase_hyper_if_betas_below,
            experimental_hyper_mutation=experimental_hyper_mutation,
            update_huge_scores=update_huge_scores,
            top_gene_prior=top_gene_prior,
            min_num_burn_in=min_num_burn_in,
            max_num_burn_in=max_num_burn_in,
            min_num_post_burn_in=min_num_post_burn_in,
            max_num_post_burn_in=max_num_post_burn_in,
            max_num_iter_betas=max_num_iter_betas,
            min_num_iter_betas=min_num_iter_betas,
            num_chains_betas=num_chains_betas,
            r_threshold_burn_in_betas=r_threshold_burn_in_betas,
            use_max_r_for_convergence_betas=use_max_r_for_convergence_betas,
            max_frac_sem_betas=max_frac_sem_betas,
            use_mean_betas=use_mean_betas,
            warm_start=warm_start,
            burn_in_rhat_quantile=burn_in_rhat_quantile,
            burn_in_patience=burn_in_patience,
            burn_in_stall_window=burn_in_stall_window,
            burn_in_stall_delta=burn_in_stall_delta,
            stop_mcse_quantile=stop_mcse_quantile,
            stop_patience=stop_patience,
            stop_top_gene_k=stop_top_gene_k,
            stop_min_gene_d=stop_min_gene_d,
            max_abs_mcse_d=max_abs_mcse_d,
            max_rel_mcse_beta=max_rel_mcse_beta,
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
            stopping_preset_name=stopping_preset_name,
            diag_every=diag_every,
            sparse_frac_gibbs=sparse_frac_gibbs,
            sparse_max_gibbs=sparse_max_gibbs,
            sparse_solution=sparse_solution,
            sparse_frac_betas=sparse_frac_betas,
            pre_filter_batch_size=pre_filter_batch_size,
            pre_filter_small_batch_size=pre_filter_small_batch_size,
            max_allowed_batch_correlation=max_allowed_batch_correlation,
            gauss_seidel_betas=gauss_seidel_betas,
            gauss_seidel=gauss_seidel,
            num_batches_parallel=num_batches_parallel,
            max_mb_X_h=max_mb_X_h,
            initial_linear_filter=initial_linear_filter,
            correct_betas_mean=correct_betas_mean,
            correct_betas_var=correct_betas_var,
            adjust_priors=adjust_priors,
            gene_set_stats_trace_out=gene_set_stats_trace_out,
            gene_stats_trace_out=gene_stats_trace_out,
            betas_trace_out=betas_trace_out,
            debug_zero_sparse=debug_zero_sparse,
            eps=eps,
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
    runtime_state.ps = np.array(
        [np.nan if value is None else value for value in runtime_state.ps],
        dtype=float,
    )
    runtime_state.sigma2s = np.array(
        [np.nan if value is None else value for value in runtime_state.sigma2s],
        dtype=float,
    )

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


def _augment_gibbs_iteration_state_with_uncorrected_and_mask(
    state,
    iter_state,
    prefilter_config,
    inner_beta_kwargs,
):
    uncorrected_setup = pigean_model.compute_gibbs_uncorrected_betas_and_defaults(
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
_apply_gibbs_log_bf_update = pigean_gibbs._apply_gibbs_log_bf_update


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
    prior_update = pigean_model.finalize_gibbs_priors_for_sampling(
        state,
        **finalize_inputs,
        log_fn=log,
        trace_level=TRACE,
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

    inner_beta_kwargs_linear = pigean_model.build_non_inf_beta_sampler_kwargs(inner_beta_kwargs)
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
    inner_beta_kwargs_linear = pigean_model.build_non_inf_beta_sampler_kwargs(inner_beta_kwargs)

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
    ) = pigean_model.compute_gibbs_iteration_priors_from_betas(
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


_build_inner_beta_sampler_common_kwargs = pigean_model.build_inner_beta_sampler_common_kwargs


_configure_hyperparameters_for_main = functools.partial(
    pigean_runtime.configure_hyperparameters_for_main,
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


pigean_runtime.bind_hyperparameter_properties(PigeanState)
