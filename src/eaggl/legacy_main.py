#Usage: python calc.py
#
#This script...
#
#Arguments:
# warnings-file[path]: file to output warnings to; default: standard error
#these are for debugging
#import cProfile
#import resource

import urllib.error
import urllib.request

import optparse
import sys
import time
import os
import copy
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
    from .pegs_utils import (
        callback_set_comma_separated_args as pegs_callback_set_comma_separated_args,
        callback_set_comma_separated_args_as_set as pegs_callback_set_comma_separated_args_as_set,
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
        load_aligned_gene_bfs as pegs_load_aligned_gene_bfs,
        load_aligned_gene_covariates as pegs_load_aligned_gene_covariates,
        load_and_apply_gene_phewas_bfs_to_runtime as pegs_load_and_apply_gene_phewas_bfs_to_runtime,
        load_and_apply_gene_set_statistics_to_runtime as pegs_load_and_apply_gene_set_statistics_to_runtime,
        load_and_apply_gene_set_phewas_statistics_to_runtime as pegs_load_and_apply_gene_set_phewas_statistics_to_runtime,
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
        sync_y_state as pegs_sync_y_state,
        sync_hyperparameter_state as pegs_sync_hyperparameter_state,
        sync_phewas_runtime_state as pegs_sync_phewas_runtime_state,
        prepare_phewas_phenos_from_file as pegs_prepare_phewas_phenos_from_file,
        read_phewas_file_batch as pegs_read_phewas_file_batch,
        append_phewas_metric_block as pegs_append_phewas_metric_block,
        accumulate_standard_phewas_outputs as pegs_accumulate_standard_phewas_outputs,
        accumulate_factor_phewas_outputs as pegs_accumulate_factor_phewas_outputs,
        write_gene_set_statistics as pegs_write_gene_set_statistics,
        write_phewas_gene_set_statistics as pegs_write_phewas_gene_set_statistics,
        write_gene_statistics as pegs_write_gene_statistics,
        write_gene_gene_set_statistics as pegs_write_gene_gene_set_statistics,
        write_phewas_statistics as pegs_write_phewas_statistics,
        write_factor_phewas_statistics as pegs_write_factor_phewas_statistics,
        finalize_regression_outputs as pegs_finalize_regression_outputs,
        compute_beta_tildes as pegs_compute_beta_tildes,
        compute_multivariate_beta_tildes as pegs_compute_multivariate_beta_tildes,
        derive_factor_anchor_masks as pegs_derive_factor_anchor_masks,
        resolve_gene_phewas_input_decision_for_stage as pegs_resolve_gene_phewas_input_decision_for_stage,
        build_phewas_stage_config as pegs_build_phewas_stage_config,
        remove_tag_from_input as pegs_remove_tag_from_input,
        XReadConfig as PegsXReadConfig,
        XReadCallbacks as PegsXReadCallbacks,
        XReadPostCallbacks as PegsXReadPostCallbacks,
        xdata_from_input_plan as pegs_xdata_from_input_plan,
        build_read_x_ingestion_options as pegs_build_read_x_ingestion_options,
        build_read_x_post_options as pegs_build_read_x_post_options,
        initialize_matrix_and_gene_index_state as pegs_initialize_matrix_and_gene_index_state,
        apply_cli_config_overrides as pegs_apply_cli_config_overrides,
        harmonize_cli_mode_args as pegs_harmonize_cli_mode_args,
        initialize_cli_logging as pegs_initialize_cli_logging,
        coerce_option_int_list as pegs_coerce_option_int_list,
        configure_random_seed as pegs_configure_random_seed,
        clean_chrom_name as pegs_clean_chrom_name,
        parse_gene_map_file as pegs_parse_gene_map_file,
        read_loc_file_with_gene_map as pegs_read_loc_file_with_gene_map,
        complete_p_beta_se as pegs_complete_p_beta_se,
        construct_map_to_ind as pegs_construct_map_to_ind,
        emit_stderr_warning as pegs_emit_stderr_warning,
        fail_removed_cli_aliases as pegs_fail_removed_cli_aliases,
        format_removed_option_message as pegs_format_removed_option_message,
        is_path_like_dest as pegs_is_path_like_dest,
        iter_parser_options as pegs_iter_parser_options,
        is_remote_path as pegs_is_remote_path,
        json_safe as pegs_json_safe,
        load_json_config as pegs_load_json_config,
        merge_dicts as pegs_merge_dicts,
        open_text_with_retry as pegs_open_text_with_retry,
        resolve_column_index as pegs_resolve_column_index,
        resolve_config_path_value as pegs_resolve_config_path_value,
        load_and_apply_bundle_defaults as pegs_load_and_apply_bundle_defaults,
        EAGGL_BUNDLE_ALLOWED_DEFAULT_INPUTS as PEGS_EAGGL_BUNDLE_ALLOWED_DEFAULT_INPUTS,
        EAGGL_BUNDLE_SCHEMA as PEGS_EAGGL_BUNDLE_SCHEMA,
    )
except ImportError:
    from pegs_utils import (
        callback_set_comma_separated_args as pegs_callback_set_comma_separated_args,
        callback_set_comma_separated_args_as_set as pegs_callback_set_comma_separated_args_as_set,
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
        load_aligned_gene_bfs as pegs_load_aligned_gene_bfs,
        load_aligned_gene_covariates as pegs_load_aligned_gene_covariates,
        load_and_apply_gene_phewas_bfs_to_runtime as pegs_load_and_apply_gene_phewas_bfs_to_runtime,
        load_and_apply_gene_set_statistics_to_runtime as pegs_load_and_apply_gene_set_statistics_to_runtime,
        load_and_apply_gene_set_phewas_statistics_to_runtime as pegs_load_and_apply_gene_set_phewas_statistics_to_runtime,
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
        sync_y_state as pegs_sync_y_state,
        sync_hyperparameter_state as pegs_sync_hyperparameter_state,
        sync_phewas_runtime_state as pegs_sync_phewas_runtime_state,
        prepare_phewas_phenos_from_file as pegs_prepare_phewas_phenos_from_file,
        read_phewas_file_batch as pegs_read_phewas_file_batch,
        append_phewas_metric_block as pegs_append_phewas_metric_block,
        accumulate_standard_phewas_outputs as pegs_accumulate_standard_phewas_outputs,
        accumulate_factor_phewas_outputs as pegs_accumulate_factor_phewas_outputs,
        write_gene_set_statistics as pegs_write_gene_set_statistics,
        write_phewas_gene_set_statistics as pegs_write_phewas_gene_set_statistics,
        write_gene_statistics as pegs_write_gene_statistics,
        write_gene_gene_set_statistics as pegs_write_gene_gene_set_statistics,
        write_phewas_statistics as pegs_write_phewas_statistics,
        write_factor_phewas_statistics as pegs_write_factor_phewas_statistics,
        finalize_regression_outputs as pegs_finalize_regression_outputs,
        compute_beta_tildes as pegs_compute_beta_tildes,
        compute_multivariate_beta_tildes as pegs_compute_multivariate_beta_tildes,
        derive_factor_anchor_masks as pegs_derive_factor_anchor_masks,
        resolve_gene_phewas_input_decision_for_stage as pegs_resolve_gene_phewas_input_decision_for_stage,
        build_phewas_stage_config as pegs_build_phewas_stage_config,
        remove_tag_from_input as pegs_remove_tag_from_input,
        XReadConfig as PegsXReadConfig,
        XReadCallbacks as PegsXReadCallbacks,
        XReadPostCallbacks as PegsXReadPostCallbacks,
        xdata_from_input_plan as pegs_xdata_from_input_plan,
        build_read_x_ingestion_options as pegs_build_read_x_ingestion_options,
        build_read_x_post_options as pegs_build_read_x_post_options,
        initialize_matrix_and_gene_index_state as pegs_initialize_matrix_and_gene_index_state,
        apply_cli_config_overrides as pegs_apply_cli_config_overrides,
        harmonize_cli_mode_args as pegs_harmonize_cli_mode_args,
        initialize_cli_logging as pegs_initialize_cli_logging,
        coerce_option_int_list as pegs_coerce_option_int_list,
        configure_random_seed as pegs_configure_random_seed,
        clean_chrom_name as pegs_clean_chrom_name,
        parse_gene_map_file as pegs_parse_gene_map_file,
        read_loc_file_with_gene_map as pegs_read_loc_file_with_gene_map,
        complete_p_beta_se as pegs_complete_p_beta_se,
        construct_map_to_ind as pegs_construct_map_to_ind,
        emit_stderr_warning as pegs_emit_stderr_warning,
        fail_removed_cli_aliases as pegs_fail_removed_cli_aliases,
        format_removed_option_message as pegs_format_removed_option_message,
        is_path_like_dest as pegs_is_path_like_dest,
        iter_parser_options as pegs_iter_parser_options,
        is_remote_path as pegs_is_remote_path,
        json_safe as pegs_json_safe,
        load_json_config as pegs_load_json_config,
        merge_dicts as pegs_merge_dicts,
        open_text_with_retry as pegs_open_text_with_retry,
        resolve_column_index as pegs_resolve_column_index,
        resolve_config_path_value as pegs_resolve_config_path_value,
        load_and_apply_bundle_defaults as pegs_load_and_apply_bundle_defaults,
        EAGGL_BUNDLE_ALLOWED_DEFAULT_INPUTS as PEGS_EAGGL_BUNDLE_ALLOWED_DEFAULT_INPUTS,
        EAGGL_BUNDLE_SCHEMA as PEGS_EAGGL_BUNDLE_SCHEMA,
    )

try:
    from .pegs_utils_phewas import (
        derive_factor_anchor_masks as pegs_derive_factor_anchor_masks,
        resolve_gene_phewas_input_decision_for_stage as pegs_resolve_gene_phewas_input_decision_for_stage,
        build_phewas_stage_config as pegs_build_phewas_stage_config,
    )
    from .pegs_utils_bundle import (
        load_and_apply_bundle_defaults as pegs_load_and_apply_bundle_defaults,
        EAGGL_BUNDLE_ALLOWED_DEFAULT_INPUTS as PEGS_EAGGL_BUNDLE_ALLOWED_DEFAULT_INPUTS,
        EAGGL_BUNDLE_SCHEMA as PEGS_EAGGL_BUNDLE_SCHEMA,
    )
except ImportError:
    from pegs_utils_phewas import (  # type: ignore
        derive_factor_anchor_masks as pegs_derive_factor_anchor_masks,
        resolve_gene_phewas_input_decision_for_stage as pegs_resolve_gene_phewas_input_decision_for_stage,
        build_phewas_stage_config as pegs_build_phewas_stage_config,
    )
    from pegs_utils_bundle import (  # type: ignore
        load_and_apply_bundle_defaults as pegs_load_and_apply_bundle_defaults,
        EAGGL_BUNDLE_ALLOWED_DEFAULT_INPUTS as PEGS_EAGGL_BUNDLE_ALLOWED_DEFAULT_INPUTS,
        EAGGL_BUNDLE_SCHEMA as PEGS_EAGGL_BUNDLE_SCHEMA,
    )

# Canonical suffix tags used when expanding dense gene-set inputs into
# sparse derived sets (top/ext/bottom thresholds).
EXT_TAG = "ext"
BOT_TAG = "bot"
TOP_TAG = "top"

def bail(message):
    raise ValueError(message)

usage = "usage: eaggl.py [factor|naive_factor] [options]"

get_comma_separated_args = pegs_callback_set_comma_separated_args
get_comma_separated_args_as_set = pegs_callback_set_comma_separated_args_as_set

parser = optparse.OptionParser(usage)
#gene x gene_set matrix
#each specification of these files is a different batch
#can use "," to group multiple files or lists within each --X
#can combine into batches with "@{batch_id}" after the file/list
#by default, the same @{batch_id} is appended to a list, which meansit will be appended to all files in the list that do not already have a batch
#this can be overriden by specifying batches to files within the list
#these batches are used for parameter learning (see below)
parser.add_option("","--X-in",action="append",default=None)
parser.add_option("","--X-list",action="append",default=None)
parser.add_option("","--Xd-in",action="append",default=None)
parser.add_option("","--Xd-list",action="append",default=None)
parser.add_option("","--ignore-genes",action='append',default=["NA"]) #gene names to ignore
parser.add_option("","--batch-separator",default="@") #separator for batches
parser.add_option("","--file-separator",default=None) #separator for multiple files

#model parameters
parser.add_option("","--p-noninf",type=float,default=None,action='append') #initial parameter for p
parser.add_option("","--top-gene-set-prior",type=float,default=None) #specify the top prior efect we are expecting any of the gene sets to have (after all of the calculations). This is the top prior across all gene sets. Precedence 4
parser.add_option("","--num-gene-sets-for-prior",type=int,default=None) #specify the top prior efect we are expecting any of the gene sets to have (after all of the calculations). This is the either the number of non-zero gene sets (by default) or the total number of gene sets (if --frac-gene-sets-for-prior is set to a number below 1).  Precedence 4
parser.add_option("","--sigma-power",type='float',default=None) #multiply sigma times np.power(scale_factors,sigma_power). 2=const_sigma, 0=default. Larger values weight larger gene sets more



parser.add_option("","--update-hyper",type='string',default=None,dest="update_hyper") #update either both,p,sigma,none


parser.add_option("","--background-prior",type=float,default=0.05) #specify background prior

#correlation matrix (otherwise will be calculated from X)
parser.add_option("","--V-in",default=None)

#optional gene name map
parser.add_option("","--gene-map-in",default=None)
parser.add_option("","--gene-map-orig-gene-col",default=1) #1-based column for original gene

#Positive control genes
parser.add_option("","--positive-controls-in",default=None)
parser.add_option("","--positive-controls-list",type="string",action="callback",callback=get_comma_separated_args,default=None) #specify comma separated list of positive controls on the command line
parser.add_option("","--positive-controls-all-in",default=None) #all genes to use in positive control analysis. If specified add these on top of the positive controls

#association statistics for gene bfs in each gene set (if precomputed)
#REMINDER: the betas are all in *external* units
parser.add_option("","--gene-set-stats-in",default=None)
parser.add_option("","--gene-set-stats-id-col",default="Gene_Set")
parser.add_option("","--gene-set-stats-exp-beta-tilde-col",default=None)
parser.add_option("","--gene-set-stats-beta-tilde-col",default=None)
parser.add_option("","--gene-set-stats-beta-col",default=None)
parser.add_option("","--gene-set-stats-beta-uncorrected-col",default=None)
parser.add_option("","--gene-set-stats-se-col",default=None)
parser.add_option("","--gene-set-stats-p-col",default=None)
parser.add_option("","--ignore-negative-exp-beta",action='store_true')

#if you have gene set betas

#gene statistics to use in calculating gene set statistics
parser.add_option("","--gene-stats-in",dest="gene_stats_in",default=None)
parser.add_option("","--gene-stats-id-col",default=None,dest="gene_stats_id_col")
parser.add_option("","--gene-stats-log-bf-col",default=None,dest="gene_stats_log_bf_col")
parser.add_option("","--gene-stats-combined-col",default=None,dest="gene_stats_combined_col")
parser.add_option("","--gene-stats-prior-col",default=None,dest="gene_stats_prior_col")
parser.add_option("","--gene-stats-prob-col",default=None,dest="gene_stats_prob_col")
parser.add_option("","--eaggl-bundle-in",default=None) #read bundled PIGEAN outputs and use as default EAGGL inputs

#locations of genes
#ALL GENE LOC FILES MUST BE IN FORMAT "GENE CHROM START END STRAND GENE" 
parser.add_option("","--gene-loc-file",default=None)
parser.add_option("","--gene-cor-file",default=None)

#additional covariates to use in the model
parser.add_option("","--gene-covs-in",default=None) #extra covariates to correct Y 

#output files for stats
parser.add_option("","--gene-set-stats-out",default=None)
parser.add_option("","--phewas-gene-set-stats-out",default=None)
parser.add_option("","--gene-stats-out",default=None)
parser.add_option("","--gene-gene-set-stats-out",default=None)
parser.add_option("","--gene-set-overlap-stats-out",default=None)
parser.add_option("","--gene-covs-out",default=None)
parser.add_option("","--gene-effectors-out",default=None)
parser.add_option("","--phewas-stats-out",default=None)
parser.add_option("","--factors-out",default=None)
parser.add_option("","--factors-anchor-out",default=None)
parser.add_option("","--gene-set-clusters-out",default=None)
parser.add_option("","--gene-clusters-out",default=None)
parser.add_option("","--pheno-clusters-out",default=None)
parser.add_option("","--gene-set-anchor-clusters-out",default=None)
parser.add_option("","--gene-anchor-clusters-out",default=None)
parser.add_option("","--pheno-anchor-clusters-out",default=None)
parser.add_option("","--factor-phewas-stats-out",default=None)

#for beta calculation against additional traits
parser.add_option("","--betas-from-phewas",action="store_true",default=False)
parser.add_option("","--betas-uncorrected-from-phewas",action="store_true",default=False)


#for pheno factoring
parser.add_option("","--gene-pheno-stats-out",default=None)

#run a phewas against the gene scores
parser.add_option("","--run-phewas-from-gene-phewas-stats-in",default=None) #specify the gene phewas stats to run a phewas against. This is distinct from --factor-phewas-from-gene-phewas-stats-in because it does not do a phewas per any factors; it does a phewas across all of the genes

#apply a multivariate regression post-hoc between the factors and many traits. The output is a separate file with p-values
parser.add_option("","--factor-phewas-from-gene-phewas-stats-in",default=None) #specify the gene phewas stats to run a factor phewas against
parser.add_option("","--factor-phewas-min-gene-factor-weight",type=float,default=0.01) #if genes have max weight across factors less than this, remove them before running phewas

#limit gene sets printed
parser.add_option("","--max-no-write-gene-set-beta",type=float,default=None) #do not write gene sets to gene-set-stats-out that have absolute beta values of this or lower
parser.add_option("","--max-no-write-gene-gene-set-beta",type=float,default=0) #do not write gene sets to gene-gene-set-stats-out that have absolute beta values of this or lower
parser.add_option("","--use-beta-uncorrected-for-gene-gene-set-write-filter",action="store_true",default=False) #filter on beta uncorrected rather than beta when filtering gene/gene set pairs to write
parser.add_option("","--max-no-write-gene-set-beta-uncorrected",type=float,default=None) #do not write gene sets to gene-set-stats-out that have absolute beta values of this or lower
parser.add_option("","--max-no-write-gene-pheno",type=float,default=0) #write only gene-pheno pairs if one value in the row is higher than this

#output for parameters
parser.add_option("","--params-out",default=None)

#control output / logging
parser.add_option("","--log-file",default=None)
parser.add_option("","--warnings-file",default=None)
parser.add_option("","--debug-level",type='int',default=None)
parser.add_option("","--hide-progress",default=False,action='store_true')
parser.add_option("","--hide-opts",default=False,action='store_true')
parser.add_option("","--seed",type='int',default=None) #seed both python random and numpy random generators
parser.add_option("","--deterministic",default=False,action='store_true') #convenience flag for deterministic runs (equivalent to --seed 0 if --seed not set)
parser.add_option("","--config",default=None) #JSON config file. Values from CLI override config values.
parser.add_option("","--print-effective-config",default=False,action='store_true') #print resolved mode/options JSON and exit

#behavior of regression
parser.add_option("","--ols",action='store_true') #run ordinary least squares rather than corrected ordinary least squares
parser.add_option("","--linear",action='store_true',dest="linear",default=None) #run linear regression on odds rather than logistic regression on binary disease status. Applies only to beta_tildes and priors, not gibbs
parser.add_option("","--no-linear",action='store_false',dest="linear",default=None) #run linear regression on odds rather than logistic regression on binary disease status. Applies only to beta_tildes and priors, not gibbs
parser.add_option("","--max-for-linear",type='float',default=None) #if linear regression is specified, it will switch to logistic regression if a probability exceeds this value


#other control
parser.add_option("","--hold-out-chrom",type="string",default=None) #don't use this chromosome for input values (infer only priors, based on other chromosomes)
parser.add_option("","--permute-gene-sets",action='store_true',default=None) #randomly shuffle the genes across gene sets (useful for negative controls)

#parameters for controlling efficiency
#split genes into batches for calculating final statistics via cross-validation
parser.add_option("","--priors-num-gene-batches",type="int",default=20)
parser.add_option("","--gibbs-num-batches-parallel",type="int",default=10)
parser.add_option("","--gibbs-max-mb-X-h",type="int",default=100)
parser.add_option("","--max-gb",type="float",default=2.0) #global memory budget in GB; memory-sensitive batching knobs are derived from this unless explicitly overridden
parser.add_option("","--max-read-entries-at-once",type="int",default=None) #cap for buffered read_X entries before flushing a batch to matrix construction
parser.add_option("","--batch-size",type=int,default=5000) #maximum number of dense X columns to hold in memory at once
parser.add_option("","--pre-filter-batch-size",type=int,default=None) #if more than this number of gene sets are about to go into non inf betas, do pre-filters on smaller batches. Assumes smaller batches will only have higher betas than full batches
parser.add_option("","--pre-filter-small-batch-size",type=int,default=500) #the limit to use for the smaller pre-filtering batches

#parameters for filtering gene sets
parser.add_option("","--min-gene-set-size",type=int,default=None) #ignore genes with fewer genes than this (after removing for other reasons)
parser.add_option("","--filter-gene-set-p",type=float,default=None) #gene sets with p above this are never seen. If this is above --max-gene-set-p, then it will be lowered to match --max-gene-set-p
parser.add_option("","--filter-negative",default=None,action="store_true",dest="filter_negative") #after sparsifying, remove any gene sets with negative beta tilde (under assumption that we added the "wrong" extreme)
parser.add_option("","--no-filter-negative",default=None,action="store_false",dest="filter_negative") #after sparsifying, remove any gene sets with negative beta tilde (under assumption that we added the "wrong" extreme)

parser.add_option("","--max-num-gene-sets-initial",type=int,default=None) #ignore gene sets to reduce to this number. Uses nominal p-values. Happens before expensive operations (pruning, parameter estimation, non-inf betas)
parser.add_option("","--max-num-gene-sets-hyper",type=int,default=5000) #use at most this number of gene sets for hyper parameter estimation (this occurs before the max-num-gene-sets operation)
parser.add_option("","--max-num-gene-sets",type=int,default=5000) #ignore gene sets to reduce to this number. Uses pruning to find independent gene sets with highest betas. Happens afer expensive operations (pruning, parameter estimation) but before gibbs
parser.add_option("","--max-gene-set-read-p",type=float,default=.05) #gene sets with p above this are excluded from the original beta analysis but included in gibbs
parser.add_option("","--min-gene-set-read-beta",type=float,default=1e-20) #gene sets with beta below this are excluded from reading in the gene stats file
parser.add_option("","--min-gene-set-read-beta-uncorrected",type=float,default=1e-20) #gene sets with beta below this are excluded from reading in the gene set stats file
parser.add_option("","--x-sparsify",type="string",action="callback",callback=get_comma_separated_args,default=[50,100,250,1000]) #applies to continuous gene sets, which are converted to dichotomous gene sets internally. For each value N, generate a new dichotomous gene set with the most N extreme genes (see next three options)
parser.add_option("","--add-ext",default=False,action="store_true") #add the top and bottom extremes as a gene set
parser.add_option("","--no-add-top",default=True,action="store_false",dest="add_top") #add the top extremes as a gene set
parser.add_option("","--no-add-bottom",default=True,action="store_false",dest="add_bottom") #add the bottom extremes as a gene set

parser.add_option("","--threshold-weights",type='float',default=0.5) #weights below this fraction of top weight are set to 0
parser.add_option("","--no-cap-weights",default=True,action="store_false",dest="cap_weights") #after normalizing weights by dividing by average, don't set those above 1 to have value 1
parser.add_option("","--max-gene-set-size",type=int,default=30000) #maximum number of genes in a gene set to consider
parser.add_option("","--add-all-genes",default=False,action="store_true") #add all genes from any gene set to the model, as opposed to just genes in the input --gwas-in or --exomes-in etc. Recommended to not normally use, since gene sets often are contaminated with genes that will bias toward significant associations. However, if you are passing in gene-values for only a small number of genes, and implicitly assuming that the remaining genes are zero, this can be used as a convenience feature rather than adding 0s for the desired genes
parser.add_option("","--prune-gene-sets",type=float,default=None) #gene sets with correlation above this threshold with any other gene set are removed (smallest gene set in correlation is retained)
parser.add_option("","--weighted-prune-gene-sets",type=float,default=None) #gene sets with correlation (weighted by Y) above this threshold with any other gene set are removed (smallest gene set in correlation is retained)
parser.add_option("","--prune-deterministically",action="store_true") #prune in order of gene set size, not in order of p-value


#gwas/huge mapping parameter

#huge exomes parametersa

#huge gwas parametersa
#these control how the probability of a SNP to gene link is scaled, independently of how many genes there are nearby
#these parameters control how all genes nearby a signal are scaled
parser.add_option("","--correct-betas-mean",default=None,action='store_true',dest="correct_betas_mean") #don't correct gene set variables (mean Z) for confounding variables (which still may exist even if all genes are corrected)
parser.add_option("","--no-correct-betas-mean",default=None,action='store_false',dest="correct_betas_mean") #don't correct gene set variables (mean Z) for confounding variables (which still may exist even if all genes are corrected)


#sampling parameters
parser.add_option("","--max-num-burn-in",type=int,default=None) #maximum number of burn in iterations to run (defaults to ceil(0.8 * --max-num-iter) for outer Gibbs)

#sparsity parameters
parser.add_option("","--sparse-solution",default=None,action="store_true",dest="sparse_solution") #zero out betas with small p_bar
parser.add_option("","--no-sparse-solution",default=None,action="store_false",dest="sparse_solution") #zero out betas with small p_bar
parser.add_option("","--sparse-frac-betas",default=None,type=float) #zero out betas with with values below this fraction of the top, within each beta_tilde->beta calculation (within gibbs and prior to it). Only applied if sparse-solution is set

#priors parameters
parser.add_option("","--adjust-priors",default=None,action='store_true',dest="adjust_priors") #do correct priors for the number of gene sets a gene is in")
parser.add_option("","--no-adjust-priors",default=None,action='store_false',dest="adjust_priors") #do not correct priors for the number of gene sets a gene is in")

#gibbs parameters

#factor parameters
parser.add_option("","--lmm-auth-key",default=None,type=str) #pass authorization key to enable LLM cluster labelling
parser.add_option("","--lmm-provider",default="openai",type=str) #LLM provider for labeling: openai (implemented), gemini/claude (reserved)
parser.add_option("","--lmm-model",default="gpt-4o-mini",type=str) #choose model
parser.add_option("","--label-gene-sets-only",default=False,action="store_true") #use only gene sets (rather than genes and gene sets) for label
parser.add_option("","--label-include-phenos",default=False,action="store_true") #add phenos to the labels (if --label-gene-sets-only is specified, the labelling will use just gene sets and phenos but skip genes). When doing phenotype based factoring, this (confusingly) adds genes rather than phenos, since phenotypes are the default thing used to label
parser.add_option("","--label-individually",default=False,action="store_true") #generate separate labels from genes, phenos, and gene sets separately
parser.add_option("","--max-num-factors",default=30,type=int) #maximum k for factorization
parser.add_option("","--phi",default=0.05,type=float) #phi prior on factorization. Higher values yield fewer factors.
parser.add_option("","--alpha0",default=10,type=float) #alpha prior on lambda k for factorization (larger makes more sparse)
parser.add_option("","--beta0",default=1,type=float) #beta prior on lambda k for factorization
parser.add_option("","--gene-set-filter-value",type=float,default=0.01) #choose value of filter for gene sets. Will use beta uncorrected if available, otherwise beta, otherwise no filter
parser.add_option("","--gene-filter-value",type=float,default=1) #choose value of filter for genes. Will use combined if available, then priors, then Y, then nothing. Used only when anchoring to a pheno(s) (or default)
parser.add_option("","--pheno-filter-value",type=float,default=1) #choose value of filter for phenos. Used only when anchoring to genes
parser.add_option("","--gene-set-pheno-filter-value",type=float,default=0.01) #choose value of filter for gene set anchoring
parser.add_option("","--no-transpose",action='store_true') #factor original X rather than tranpose
parser.add_option("","--min-lambda-threshold",type=float,default=1e-3) #remove factors with lambdak values below this threshold, or sum(gene loadings) below this threshold, or sum(gene set loadings) below this threshold

#options for controlling factoring behavior
#Factoring decomposes the gene set x gene matrix or gene set x phenotype matrix while weighting matrix entries specific to an "anchor". An "anchor" can be either a single or set of phenotypes, or a single or set of genes
#
#Options for phenotype-based anchoring of factoring
#1. By default, factoring will be performed across the gene set x gene matrix and weighted by the gene combined scores or the gene set beta scores. We refer to this as "single phenotype anchoring" since the weights are based on associations of the genes and gene sets with the phenotype.
#   These will be computed as in a normal PIGEAN run, using --gwas-in or --positive-controls-in or --exomes-in or --gene-stats-in etc.
#   To specify this behavior, simply run factor (or factor_naive to generate betas and combined scores using the incorrect but faster naive approach)
#   Matrix to be factored is (probability gene relevant to anchor phenotype) * (probability gene set contains gene) * (probability that gene set is relevant to anchor phenotype)
#   Gene anchor loadings are (probability gene relevant to anchor phenotype) * (probability pathway contains gene) and gene set anchor loadings (probability that gene set interrogates pathway) * (probability that pathway is relevant to anchor phenotype)
#   Factor relevance scores are probability factor is relevant to anchor phenotype
#2. A special case of number 1 is to factor an input gene list. In this case, the gene list is treated like a "phenotype" and gene/gene set scores are determined that predict membership in it. So even though it is a gene list, it is *unrelated* to the gene sets used to construct the matrix. Semantically, it is equivalent to single phenotype anchoring
#   This is a way to decompose the gene set into distinct mechanisms and then (potentially) project it onto more phenotypes
#   To run this, use the --positive-controls-in (which allows weighting of genes in the set) or the --positive-controls-list options
#3. You can project the results of the "single phenotype anchoring" onto other phenotypes for which gene phewas or gene set phewas results are available. This will create factor loadings for all phenotypes in the file
#   To run this, use the --gene-set-phewas-stats-in option or the --gene-phewas-stats-in options alongside the factor command. If both --gene-set-phewas-stats-in and --gene-phewas-stats-in are specified, --gene-phewas-stats-in will be ignored and --gene-set-phewas-in will be used.
#   The factor command must include a way to obtain betas (e.g. as in a normal PIGEAN run) or you will get different behavior.
#   The interpretation of phenotype anchor loadings file is (probability phenotype associated with pathway) under this setting
#4. You can run a "multiple phenotype anchoring" factoring in which case the factorization will maximize the similarity of the approximated matrix to an input tensor, which is the input gene set x gene matrix projected along a third dimension (of length equal to the number of anchor phenotypes). Each matrix slice is obtained by the gene set x gene matrix (with entries equal to gene set weights) multiplied by the gene probabilities and gene set probabilities for the corresponding anchor phenotype.
#   There will be multiple loadings for each gene, gene set, and factor per anchor phenotype (interpretable identically to the single phenotype anchoring case), each sharing a core indicator loading multiplied by the gene or gene set probability. All phenotypes will be automatically projected as well as a part of this
#   To specify this behavior, run by passing in both --gene-phewas-stats-in and --gene-set-phewas-stats-in alongside factor model. Any other arguments to compute betas will be ignored here. You must also specify --anchor-phenos to determine the subset of phenotypes in the --phewas-stats-in to anchor on
#5. If you want to factor the entire gene by gene set matrix across all phenotypes in the --phewas-stats-in files, you specify the same flags as in the multiple phenotype anchoring case but replace --anchor-phenos with --anchor-any-pheno. This will not actually compute loadings for each phenotype, but rather will construct an "uber" phenotype that represents the probability that the gene or gene set is associated with any of the phenotypes.
#   The interpretation is as above, but instead of terms for (probability that X is relevant to the anchor phenotype) there are terms for (probability that X is relevant to any of the phenotypes)
#   You can use the flags --factor-prune-phenos-num or --factor-prune-phenos-val to reduce the number of phenotypes going into this analysis. --factor-prune-genes and --factor-prune-gene-sets are also important for limiting run time
#
#Options for gene-based anchoring of factoring
#6. To factor a gene set x phenotype matrix, you must anchor to a gene to determine the phenotype relevance scores and which gene sets to include. This is called "single gene anchoring"
#   You specify this behavior by passing in --gene-phewas-stats-in and --gene-set-phewas-stats-in (any flags for computing betas will be ignored) and then specifying --anchor-genes.
#   The entries in the input matrix represent (probability that gene is associated with phenotype) * (probability that gene set is associated with phenotype)
#   The anchor loadings files are, for phenotypes the (probability that the anchor gene is associated with phenotype) * (probability that phenotype is associated with the pathway) and, for gene sets, the (probability that the gene set interrogates the pathway)
#7. You can pass multiple comma separated values to --anchor-genes to run "multiple gene anchoring" factoring which behaves analogously to the mulitple phenotype anchoring. In this case, you can choose how gene sets are included in the matrix (either --add-gene-sets-by-enrichment-p, --add-gene-sets-by-fraction, --add-gene-sets-by-naive, or --add-gene-sets-by-gibbs)
#8. Finally, you can anchor across all genes using --anchor-any-gene. Just as for --anchor-any-pheno, this doesn't actually produce loadings for every anchor gene but instead uses weights for phenotypes corresponding to the probability that they are associated with any gene (these will usually be very close to 1)
#   To reduce the size of the matrix going into the factoring, you can use --factor-prune-phenos-num or --factor-prune-phenos-val which will remove phenotypes just as in the case of --anchor-any-pheno
#
#Options for gene-set-based anchoring of factoring
#9. You can factor the gene set x phenotype matrix if you specify --anchor-gene-set and pass in enough information to run a pheWAS
#   You need to pass in either positive controls, a GWAS, exomes, or gene-bfs, and then specify --run-phewas-from-gene-phewas-stats-in. This will then load the phewas statistics, which will then be used as weights in the factoring.
#   This will produce a single weight for the entire gene set, which is distinct from --anchor-gene [gene set] which will produce weights for each gene in the gene set as part of the factoring.
#   The entries in the input matrix represent (probability input gene set is associated with phenotype) * (probability that the input gene set is associated with gene set).
#   The anchor loadings will be (probability input gene set is associated with phenotype) * (probability that the phenotype is associated with the pathway) and (probability that the gene set interrogates the gene) * (probability that the gene set is associated with the input gene set)

#Note that the signle pheno anchoring with a gene list (option 2) and the multiple gene anchoring with the same gene list (option 7) both take a gene set as input, produce factors, and loadings across genes, phenos, and gene sets. The difference is in the interpretation of the factors. In the former case (option 2), the pathways are chosen to explain why each gene is in each gene set, with genes and gene sets weighted by similarity to the gene list. In the latter case (option 7), the pathways are chosen to explain why each pathway is associated with each phenotype, with phenotypes weighted according to how associated they are with each gene in the gene list


parser.add_option("","--gene-set-phewas-stats-in",default=None)
parser.add_option("","--gene-set-phewas-stats-id-col",default="Gene_Set")
parser.add_option("","--gene-set-phewas-stats-beta-col",default=None)
parser.add_option("","--gene-set-phewas-stats-beta-uncorrected-col",default=None)
parser.add_option("","--gene-set-phewas-stats-pheno-col",default=None)

parser.add_option("","--gene-phewas-bfs-in",default=None)
parser.add_option("","--gene-phewas-stats-in",dest="gene_phewas_bfs_in",default=None)
parser.add_option("","--gene-phewas-bfs-id-col",default=None)
parser.add_option("","--gene-phewas-stats-id-col",default=None,dest="gene_phewas_bfs_id_col")
parser.add_option("","--gene-phewas-bfs-log-bf-col",default=None)
parser.add_option("","--gene-phewas-stats-log-bf-col",default=None,dest="gene_phewas_bfs_log_bf_col")
parser.add_option("","--gene-phewas-bfs-combined-col",default=None)
parser.add_option("","--gene-phewas-stats-combined-col",default=None,dest="gene_phewas_bfs_combined_col")
parser.add_option("","--gene-phewas-bfs-prior-col",default=None)
parser.add_option("","--gene-phewas-stats-prior-col",default=None,dest="gene_phewas_bfs_prior_col")
parser.add_option("","--gene-phewas-bfs-pheno-col",default=None)
parser.add_option("","--gene-phewas-stats-pheno-col",default=None,dest="gene_phewas_bfs_pheno_col")
parser.add_option("","--min-gene-phewas-read-value",type="float",default=1)
parser.add_option("","--gene-phewas-id-to-X-id",default=None) #mapping from gene ids in the phewas file to gene ids in the gmt
parser.add_option("","--project-phenos-from-gene-sets",action='store_true',default=False) #use gene set scores to project pheno loadings rather than gene scores. Note that this will also have the side effect of conditioning the regression only on the gene sets in the model rather than all gene sets

parser.add_option("","--anchor-phenos",type="string",action="callback",callback=get_comma_separated_args_as_set,default=None) #run single or multiple pheno anchoring
parser.add_option("","--anchor-pheno",type="string",action="callback",callback=get_comma_separated_args_as_set,default=None,dest="anchor_phenos") #run single or multiple pheno anchoring
parser.add_option("","--anchor-any-pheno",action="store_true",default=False) #flatten all phenotypes into an uber weight
parser.add_option("","--anchor-genes",type="string",action="callback",callback=get_comma_separated_args_as_set,default=None) #run single or multiple gene anchoring
parser.add_option("","--anchor-gene",type="string",action="callback",callback=get_comma_separated_args_as_set,default=None,dest="anchor_genes") #run single or multiple gene anchoring
parser.add_option("","--anchor-any-gene",action="store_true",default=False) #update phenotype associations to essentially be uniformly 1
parser.add_option("","--anchor-gene-set",action="store_true",default=False) #run gene set anchoring

parser.add_option("","--factor-prune-phenos-num",type='int',default=None) #when running --anchor-any-pheno or --anchor-any gene, reduce phenotypes by including only this many (add an independent set). Phenotypes will be sorted by average probability across genes
parser.add_option("","--factor-prune-phenos-val",type='float',default=None) #when running --anchor-any-pheno or --anchor-any gene, reduce phenotypes by pruning those more correlated than this value. Phenotypes will be sorted by average probability across genes
parser.add_option("","--factor-prune-genes-num",type='int',default=None) #when running --anchor-any-pheno or --anchor-any gene, reduce genes by including only this many (add an independent set). Genes will be sorted by average probability across phenotypes
parser.add_option("","--factor-prune-genes-val",type='float',default=None) #when running --anchor-any-pheno or --anchor-any gene, reduce genes by pruning those more correlated than this value. Genes will be sorted by average probability across phenotypes
parser.add_option("","--factor-prune-gene-sets-num",type='int',default=None) #when running --anchor-any-pheno or --anchor-any gene, reduce gene sets by including only this many (add an independent set). Gene sets will be sorted by maximum association across phenotypes
parser.add_option("","--factor-prune-gene-sets-val",type='float',default=None) #when running --anchor-any-pheno or --anchor-any gene, reduce gene sets by pruning those more correlated than this value. Gene sets will be sorted by maximum assoication across phenotypes


parser.add_option("","--add-gene-sets-by-enrichment-p",type='float',default=None) #when running multiple gene anchoring, add in gene sets that pass the enrichment filters. Filter according to p-value
parser.add_option("","--add-gene-sets-by-fraction",type="float",default=None) #when running multiple gene anchoring, add in gene sets that have this fraction of input genes

#simulation parameters

parser.add_option("","--num-chains",type=int,default=10) #number of chains for gibbs sampling. Larger number uses more memory and compute but produces lower MCSE

#beta sampling parameters
parser.add_option("","--min-num-iter-betas",type=int,default=10) #minimum number of iterations to run for beta sampling
parser.add_option("","--max-num-iter-betas",type=int,default=1100) #maximum number of iterations to run for beta sampling
parser.add_option("","--num-chains-betas",type=int,default=4) #number of chaings for beta sampling
parser.add_option("","--r-threshold-burn-in-betas",type=float,default=1.01) #threshold for R to consider a gene set as converged (that is, stop burn in and start sampling)
parser.add_option("","--gauss-seidel-betas",action="store_true") #run gauss seidel
parser.add_option("","--max-frac-sem-betas",type=float,default=0.01) #the minimum z score (mean/sem) to allow after stopping sampling; continue sampling if this is too large
parser.add_option("","--use-max-r-for-convergence-betas",action="store_true") #use only the maximum R across gene sets to evaluate convergence (most conservative). By default uses mean R


#TEMP DEBUGGING FLAGS
parser.add_option("","--debug-old-batch",action="store_true") #
parser.add_option("","--debug-skip-phewas-covs",action="store_true") #
parser.add_option("","--debug-skip-huber",action="store_true") #
parser.add_option("","--debug-skip-correlation",action="store_true") #
parser.add_option("","--debug-just-check-header",action="store_true") #
parser.add_option("","--debug-only-avg-huge",action="store_true")

_iter_parser_options = pegs_iter_parser_options

_merge_dicts = pegs_merge_dicts

_is_remote_path = pegs_is_remote_path

_is_path_like_dest = pegs_is_path_like_dest

_resolve_config_path_value = pegs_resolve_config_path_value

_early_warn = pegs_emit_stderr_warning

_json_safe = pegs_json_safe

REMOVED_OPTION_REPLACEMENTS = {
    # PIGEAN-owned raw input modes/expansion controls are not supported in EAGGL.
    "gwas_in": "__MOVED_TO_PIGEAN",
    "huge_statistics_in": "__MOVED_TO_PIGEAN",
    "huge_statistics_out": "__MOVED_TO_PIGEAN",
    "credible_sets_in": "__MOVED_TO_PIGEAN",
    "s2g_in": "__MOVED_TO_PIGEAN",
    "exomes_in": "__MOVED_TO_PIGEAN",
    "case_counts_in": "__MOVED_TO_PIGEAN",
    "ctrl_counts_in": "__MOVED_TO_PIGEAN",
    "add_gene_sets_by_naive": "__MOVED_TO_PIGEAN",
    "add_gene_sets_by_gibbs": "__MOVED_TO_PIGEAN",

    "gene_bfs_in": "--gene-stats-in",
    "gene_bfs_id_col": "--gene-stats-id-col",
    "gene_bfs_log_bf_col": "--gene-stats-log-bf-col",
    "gene_bfs_combined_col": "--gene-stats-combined-col",
    "gene_bfs_prior_col": "--gene-stats-prior-col",
    "gene_bfs_prob_col": "--gene-stats-prob-col",
    "gene_percentiles_in": None,
    "gene_percentiles_id_col": None,
    "gene_percentiles_value_col": None,
    "gene_percentiles_top_posterior": None,
    "gene_percentiles_higher_is_better": None,
    "gene_zs_in": None,
    "gene_zs_id_col": None,
    "gene_zs_value_col": None,
    "gene_zs_gws_threshold": None,
    "gene_zs_max_mean_posterior": None,
    "chisq_dynamic": None,
    "desired_intercept_difference": None,
    "chisq_threshold": None,
    "run_gls": None,
    "store_cholesky": None,
}

def _enforce_eaggl_mode_ownership(_mode):
    factor_modes = set(["factor", "naive_factor"])
    if _mode not in factor_modes:
        bail("Mode '%s' belongs to pigean.py; run with pigean.py instead of eaggl.py" % _mode)

_FACTOR_WORKFLOW_STRATEGY_META = {
    "F1": {
        "required_inputs": [],
        "factor_gene_set_x_pheno": False,
        "use_phewas_for_factoring": False,
        "expand_gene_sets": False,
        "warn_ignored_y_inputs_mode": None,
    },
    "F2": {
        "required_inputs": [],
        "factor_gene_set_x_pheno": False,
        "use_phewas_for_factoring": False,
        "expand_gene_sets": False,
        "warn_ignored_y_inputs_mode": None,
    },
    "F3": {
        "required_inputs": [],
        "factor_gene_set_x_pheno": False,
        "use_phewas_for_factoring": False,
        "expand_gene_sets": False,
        "warn_ignored_y_inputs_mode": None,
    },
    "F4": {
        "required_inputs": ["--gene-set-phewas-stats-in", "--gene-phewas-stats-in"],
        "factor_gene_set_x_pheno": False,
        "use_phewas_for_factoring": True,
        "expand_gene_sets": False,
        "warn_ignored_y_inputs_mode": "anchor_phenos",
    },
    "F5": {
        "required_inputs": ["--gene-set-phewas-stats-in", "--gene-phewas-stats-in"],
        "factor_gene_set_x_pheno": False,
        "use_phewas_for_factoring": True,
        "expand_gene_sets": False,
        "warn_ignored_y_inputs_mode": "anchor_phenos",
    },
    "F6": {
        "required_inputs": ["--gene-set-phewas-stats-in", "--gene-phewas-stats-in"],
        "factor_gene_set_x_pheno": True,
        "use_phewas_for_factoring": True,
        "expand_gene_sets": False,
        "warn_ignored_y_inputs_mode": "anchor_genes",
    },
    "F7": {
        "required_inputs": ["--gene-set-phewas-stats-in", "--gene-phewas-stats-in"],
        "factor_gene_set_x_pheno": True,
        "use_phewas_for_factoring": True,
        "expand_gene_sets": True,
        "warn_ignored_y_inputs_mode": "anchor_genes",
    },
    "F8": {
        "required_inputs": ["--gene-set-phewas-stats-in", "--gene-phewas-stats-in"],
        "factor_gene_set_x_pheno": True,
        "use_phewas_for_factoring": True,
        "expand_gene_sets": False,
        "warn_ignored_y_inputs_mode": "anchor_genes",
    },
    "F9": {
        "required_inputs": ["--run-phewas-from-gene-phewas-stats-in"],
        "factor_gene_set_x_pheno": True,
        "use_phewas_for_factoring": False,
        "expand_gene_sets": False,
        "warn_ignored_y_inputs_mode": None,
    },
}


def _workflow_required_inputs_satisfied(_workflow_id, _options):
    required_inputs = _FACTOR_WORKFLOW_STRATEGY_META[_workflow_id]["required_inputs"]
    missing_inputs = []
    for flag in required_inputs:
        if flag == "--gene-set-phewas-stats-in":
            if _options.gene_set_phewas_stats_in is None:
                missing_inputs.append(flag)
        elif flag == "--gene-phewas-stats-in":
            if _options.gene_phewas_bfs_in is None:
                missing_inputs.append(flag)
        elif flag == "--run-phewas-from-gene-phewas-stats-in":
            if _options.run_phewas_from_gene_phewas_stats_in is None:
                missing_inputs.append(flag)
    return missing_inputs


def _build_factor_workflow_error(_workflow_id, _missing_inputs):
    if len(_missing_inputs) == 0:
        return None
    if _workflow_id in ("F4", "F5", "F6", "F7", "F8"):
        return "Require --gene-set-phewas-stats-in and --gene-phewas-stats-in"
    if _workflow_id == "F9":
        return "Require --run-phewas-from-gene-phewas-stats"
    return "Missing required inputs: %s" % ", ".join(_missing_inputs)


def _has_potentially_ignored_factor_inputs(_options):
    return bool(
        _options.gene_set_stats_in
        or _options.positive_controls_in
        or _options.positive_controls_list is not None
    )


def _warn_for_factor_workflow_inputs(_options, _workflow):
    add_gene_set_flags_present = (
        _options.add_gene_sets_by_enrichment_p is not None
        or _options.add_gene_sets_by_fraction is not None
    )
    if add_gene_set_flags_present and not _workflow["expand_gene_sets"]:
        warn("Ignoring options to add gene sets based on association with anchor genes because only 1 anchor gene was specified")

    if _options.anchor_gene_set:
        return
    if not _has_potentially_ignored_factor_inputs(_options):
        return

    warning_mode = _workflow.get("warn_ignored_y_inputs_mode")
    if warning_mode == "anchor_phenos":
        warn("Ignoring all arguments for reading Y or reading betas in --anchor-phenos mode")
    elif warning_mode == "anchor_genes":
        warn("Ignoring all arguments for reading Y or reading betas in --anchor-genes mode")


def _format_anchor_values_for_label(values):
    if values is None:
        return "None"
    if isinstance(values, set):
        values = sorted(list(values))
    elif isinstance(values, (tuple, list)):
        values = list(values)
    else:
        return str(values)
    return "{%s}" % ", ".join(["'%s'" % x for x in values])


def _classify_factor_workflow(_options):
    has_gene_set_phewas = _options.gene_set_phewas_stats_in is not None
    has_gene_phewas = _options.gene_phewas_bfs_in is not None
    projection_source = _options.gene_set_phewas_stats_in if has_gene_set_phewas else _options.gene_phewas_bfs_in

    workflow_id = None
    workflow_label = None

    if _options.anchor_genes is not None and len(_options.anchor_genes) == 1:
        workflow_id = "F6"
        workflow_label = "single gene anchoring (to %s)" % _format_anchor_values_for_label(_options.anchor_genes)
    elif _options.anchor_genes is not None and len(_options.anchor_genes) > 1:
        workflow_id = "F7"
        workflow_label = "multiple gene anchoring (to %s)" % _format_anchor_values_for_label(_options.anchor_genes)
    elif _options.anchor_any_gene:
        workflow_id = "F8"
        workflow_label = "any gene anchoring"
    elif _options.anchor_gene_set:
        workflow_id = "F9"
        workflow_label = "gene set anchoring (to input phenotype/gene set)"
    elif _options.anchor_phenos is not None and len(_options.anchor_phenos) == 1:
        workflow_id = "F4"
        workflow_label = "single phenotype anchoring (to %s) but with phewas statistics used" % _format_anchor_values_for_label(_options.anchor_phenos)
    elif _options.anchor_phenos is not None and len(_options.anchor_phenos) > 1:
        workflow_id = "F4"
        workflow_label = "multiple phenotype anchoring (to %s)" % _format_anchor_values_for_label(_options.anchor_phenos)
    elif _options.anchor_any_pheno:
        workflow_id = "F5"
        workflow_label = "any phenotype anchoring"
    else:
        workflow_label = "single phenotype anchoring (to %s) using default statistics" % _options.anchor_phenos
        if projection_source is not None:
            workflow_id = "F3"
            workflow_label = "%s. Will project using %s" % (workflow_label, projection_source)
        elif _options.positive_controls_in is not None or _options.positive_controls_list is not None:
            workflow_id = "F2"
        else:
            workflow_id = "F1"

    strategy = _FACTOR_WORKFLOW_STRATEGY_META[workflow_id]
    missing_inputs = _workflow_required_inputs_satisfied(workflow_id, _options)

    workflow = {
        "id": workflow_id,
        "label": workflow_label,
        "error": _build_factor_workflow_error(workflow_id, missing_inputs),
        "required_inputs": list(strategy["required_inputs"]),
        "missing_required_inputs": missing_inputs,
        "factor_gene_set_x_pheno": bool(strategy["factor_gene_set_x_pheno"]),
        "use_phewas_for_factoring": bool(strategy["use_phewas_for_factoring"]),
        "expand_gene_sets": bool(strategy["expand_gene_sets"]),
        "warn_ignored_y_inputs_mode": strategy["warn_ignored_y_inputs_mode"],
        "has_gene_set_phewas": has_gene_set_phewas,
        "has_gene_phewas": has_gene_phewas,
    }

    return workflow


def _apply_eaggl_bundle_inputs(_options):
    if _options.eaggl_bundle_in is None:
        return None

    return pegs_load_and_apply_bundle_defaults(
        _options,
        bundle_path=_options.eaggl_bundle_in,
        expected_schema=PEGS_EAGGL_BUNDLE_SCHEMA,
        allowed_default_inputs=PEGS_EAGGL_BUNDLE_ALLOWED_DEFAULT_INPUTS,
        bundle_flag_name="--eaggl-bundle-in",
        manifest_name="manifest.json",
        temp_prefix="eaggl_bundle_in_",
        x_source_option_names=["X_in", "X_list", "Xd_in", "Xd_list"],
        x_default_key="X_in",
        x_target_option_name="X_in",
        scalar_default_option_names=["gene_stats_in", "gene_set_stats_in", "gene_phewas_bfs_in", "gene_set_phewas_stats_in"],
        bail_fn=bail,
    )

argv_parse = sys.argv[1:]
pegs_fail_removed_cli_aliases(
    argv_parse,
    REMOVED_OPTION_REPLACEMENTS,
    format_removed_option_message_fn=pegs_format_removed_option_message,
)

(options, args) = parser.parse_args(argv_parse)
(options, args, config_mode, cli_specified_dests, config_specified_dests) = pegs_apply_cli_config_overrides(
    options,
    args,
    parser,
    argv_parse,
    resolve_path_fn=_resolve_config_path_value,
    is_path_like_dest_fn=_is_path_like_dest,
    early_warn_fn=_early_warn,
    bail_fn=bail,
    removed_option_replacements=REMOVED_OPTION_REPLACEMENTS,
    format_removed_option_message_fn=pegs_format_removed_option_message,
    track_config_specified_dests=True,
)

args = pegs_harmonize_cli_mode_args(args, config_mode, early_warn_fn=_early_warn)

_logging_state = pegs_initialize_cli_logging(options, stderr_stream=sys.stderr, default_debug_level=1)
NONE = _logging_state["NONE"]
INFO = _logging_state["INFO"]
DEBUG = _logging_state["DEBUG"]
TRACE = _logging_state["TRACE"]
debug_level = _logging_state["debug_level"]
log_fh = _logging_state["log_fh"]
warnings_fh = _logging_state["warnings_fh"]
log = _logging_state["log"]
warn = _logging_state["warn"]


def _query_openai_chat_completion(query, auth_key=None, lmm_model=None):
    if auth_key is None:
        bail("Need --lmm-auth-key to use LLM labeling")

    model = lmm_model if lmm_model is not None else "gpt-4o-mini"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": query}],
        "temperature": 0,
    }
    request = urllib.request.Request(
        "https://api.openai.com/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": "Bearer %s" % auth_key,
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response_fh:
            response_payload = json.loads(response_fh.read().decode("utf-8"))
        choices = response_payload.get("choices", [])
        if len(choices) == 0:
            warn("OpenAI response missing choices; skipping LLM labels")
            return None
        message = choices[0].get("message", {})
        content = message.get("content")
        if content is None:
            warn("OpenAI response missing message content; skipping LLM labels")
            return None
        return content
    except urllib.error.HTTPError as e:
        body = ""
        try:
            body = e.read().decode("utf-8")
        except Exception:
            body = str(e)
        warn("OpenAI labeling request failed: HTTP %s %s" % (e.code, body))
        return None
    except urllib.error.URLError as e:
        warn("OpenAI labeling request failed: %s" % e)
        return None
    except Exception as e:
        warn("OpenAI labeling request failed: %s" % e)
        return None


def query_lmm(query, auth_key=None, lmm_model=None, lmm_provider="openai"):
    provider = (lmm_provider if lmm_provider is not None else "openai").strip().lower()
    if provider == "openai":
        return _query_openai_chat_completion(query, auth_key=auth_key, lmm_model=lmm_model)
    if provider == "gemini":
        bail("LLM provider 'gemini' is not implemented yet; use --lmm-provider openai")
    if provider == "claude":
        bail("LLM provider 'claude' is not implemented yet; use --lmm-provider openai")
    bail("Unsupported --lmm-provider '%s'; expected one of: openai, gemini, claude" % provider)


eaggl_bundle_info = _apply_eaggl_bundle_inputs(options)
if eaggl_bundle_info is not None:
    applied = eaggl_bundle_info.applied_defaults
    if len(applied) == 0:
        log("Loaded --eaggl-bundle-in bundle %s (no defaults applied; explicit CLI/config inputs took precedence)" % options.eaggl_bundle_in, INFO)
    else:
        applied_text = ", ".join(["%s=%s" % (k, applied[k]) for k in sorted(applied.keys())])
        log("Loaded --eaggl-bundle-in bundle %s and applied defaults: %s" % (options.eaggl_bundle_in, applied_text), INFO)

pegs_configure_random_seed(options, random, np, log_fn=log, info_level=INFO)

options.x_sparsify = pegs_coerce_option_int_list(options.x_sparsify, "--x-sparsify", bail)

if len(args) < 1:
    bail(usage)

mode = args[0]
_enforce_eaggl_mode_ownership(mode)

run_factor = False
run_phewas = False

run_naive_factor = False
use_phewas_for_factoring = False
factor_gene_set_x_pheno = False
expand_gene_sets = False
factor_workflow = None

if mode == "factor" or mode == "naive_factor": #run factoring, phewas factoring, or pheno factoring
    run_factor = True
    if mode == "naive_factor":
        run_naive_factor = True

    factor_workflow = _classify_factor_workflow(options)
    factor_type = factor_workflow["label"]
    error = factor_workflow["error"]
    factor_gene_set_x_pheno = factor_workflow["factor_gene_set_x_pheno"]
    use_phewas_for_factoring = factor_workflow["use_phewas_for_factoring"]
    expand_gene_sets = factor_workflow["expand_gene_sets"]

    if error is not None:
        bail("Cannot run factoring type: %s. %s" % (factor_type, error))
    else:
        log("Running factoring type: %s [workflow=%s]" % (factor_type, factor_workflow["id"]))
        _warn_for_factor_workflow_inputs(options, factor_workflow)
else:
    bail("Unrecognized mode %s" % mode)

if options.run_phewas_from_gene_phewas_stats_in is not None:
    run_phewas = True

#set defaults (EAGGL supports only factor/naive_factor modes)
options.correct_betas_mean = options.correct_betas_mean if options.correct_betas_mean is not None else True
options.adjust_priors = options.adjust_priors if options.adjust_priors is not None else True
options.p_noninf = options.p_noninf if options.p_noninf is not None else [0.001]
options.sigma_power = options.sigma_power if options.sigma_power is not None else -2
options.update_hyper = options.update_hyper if options.update_hyper is not None else "p"
options.filter_negative = options.filter_negative if options.filter_negative is not None else True
if options.prune_gene_sets is None:
    if run_factor and factor_gene_set_x_pheno:
        options.prune_gene_sets = 0.5
    else:
        options.prune_gene_sets = 0.8

if options.weighted_prune_gene_sets is None:
    if run_factor and factor_gene_set_x_pheno:
        options.weighted_prune_gene_sets = 0.5
    else:
        options.weighted_prune_gene_sets = 0.8

options.top_gene_set_prior = options.top_gene_set_prior if options.top_gene_set_prior is not None else 0.8
options.num_gene_sets_for_prior = options.num_gene_sets_for_prior if options.num_gene_sets_for_prior is not None else 50
options.filter_gene_set_p = options.filter_gene_set_p if options.filter_gene_set_p is not None else 0.01
options.linear = options.linear if options.linear is not None else False
options.max_for_linear = options.max_for_linear if options.max_for_linear is not None else 0.95
options.min_gene_set_size = options.min_gene_set_size if options.min_gene_set_size is not None else 10

if run_factor and factor_gene_set_x_pheno is not None:
    if options.add_gene_sets_by_enrichment_p is not None:
        options.filter_gene_set_p = options.add_gene_sets_by_enrichment_p

options.sparse_frac_betas = options.sparse_frac_betas if options.sparse_frac_betas is not None else 0.001
options.sparse_solution = options.sparse_solution if options.sparse_solution is not None else True

def _is_option_dest_explicit(dest):
    if cli_specified_dests is not None and dest in cli_specified_dests:
        return True
    if config_specified_dests is not None and dest in config_specified_dests:
        return True
    return False

def _derive_memory_controls_from_max_gb():
    if options.max_gb is None:
        options.max_gb = 2.0
    if options.max_gb <= 0:
        bail("Option --max-gb must be > 0")

    total_mb = int(round(options.max_gb * 1024.0))
    baseline_gb = 2.0
    scale = options.max_gb / baseline_gb
    if scale <= 0:
        scale = 1.0

    derived = {}
    clamped = {}

    def _set_with_max_cap(opt_name, implied_max):
        current = getattr(options, opt_name)
        explicit = _is_option_dest_explicit(opt_name)
        if current is None:
            new_value = implied_max
            derived[opt_name] = new_value
        else:
            new_value = min(int(current), int(implied_max))
            if explicit and new_value < int(current):
                clamped[opt_name] = (current, new_value, "max")
            elif not explicit and new_value != int(current):
                derived[opt_name] = new_value
        setattr(options, opt_name, int(new_value))

    def _set_with_min_floor(opt_name, implied_min):
        current = getattr(options, opt_name)
        explicit = _is_option_dest_explicit(opt_name)
        if current is None:
            new_value = implied_min
            derived[opt_name] = new_value
        else:
            new_value = max(int(current), int(implied_min))
            if explicit and new_value > int(current):
                clamped[opt_name] = (current, new_value, "min")
            elif not explicit and new_value != int(current):
                derived[opt_name] = new_value
        setattr(options, opt_name, int(new_value))

    implied_batch_size_max = max(500, int(round(5000 * scale)))
    # Outer-Gibbs stacked-X is only one large buffer among many; keep it conservative.
    implied_gibbs_max_mb_X_h_max = max(32, int(round(total_mb * 0.20)))
    # read_X buffers Python object triplets (data,row,col); keep conservative for low-memory runs.
    implied_max_read_entries_at_once_max = max(100000, int(round(total_mb * 500)))
    implied_gibbs_num_batches_parallel_max = max(1, int(round(10 * scale)))
    if options.num_chains is not None:
        implied_gibbs_num_batches_parallel_max = min(implied_gibbs_num_batches_parallel_max, int(options.num_chains))
    implied_pre_filter_small_batch_size_max = max(100, int(round(500 * scale)))
    implied_pre_filter_batch_size_max = max(implied_pre_filter_small_batch_size_max, int(round(5000 * scale)))
    # For tighter memory budgets, increase gene batches; for looser budgets, reduce batches.
    # This is an inverse memory knob: larger values use less memory.
    implied_priors_num_gene_batches_min = max(1, int(np.ceil(20.0 / scale)))

    _set_with_max_cap("batch_size", implied_batch_size_max)
    _set_with_max_cap("gibbs_max_mb_X_h", implied_gibbs_max_mb_X_h_max)
    _set_with_max_cap("max_read_entries_at_once", implied_max_read_entries_at_once_max)
    _set_with_max_cap("gibbs_num_batches_parallel", implied_gibbs_num_batches_parallel_max)
    _set_with_max_cap("pre_filter_small_batch_size", implied_pre_filter_small_batch_size_max)
    if options.pre_filter_batch_size is not None:
        _set_with_max_cap("pre_filter_batch_size", implied_pre_filter_batch_size_max)
    _set_with_min_floor("priors_num_gene_batches", implied_priors_num_gene_batches_min)

    log("Memory controls: --max-gb=%.3g (%.0f MB total), effective batch controls: max_read_entries_at_once=%d, priors_num_gene_batches=%d, gibbs_num_batches_parallel=%d, gibbs_max_mb_X_h=%d, batch_size=%d, pre_filter_batch_size=%s, pre_filter_small_batch_size=%d" % (options.max_gb, total_mb, options.max_read_entries_at_once, options.priors_num_gene_batches, options.gibbs_num_batches_parallel, options.gibbs_max_mb_X_h, options.batch_size, str(options.pre_filter_batch_size), options.pre_filter_small_batch_size), INFO)
    if len(derived) > 0:
        log("Derived from --max-gb (implicit/default adjustments): %s" % ", ".join(["%s=%s" % (k, derived[k]) for k in sorted(derived.keys())]), DEBUG)
    if len(clamped) > 0:
        log("Clamped by --max-gb: %s" % ", ".join(["%s:%s->%s(%s)" % (k, clamped[k][0], clamped[k][1], clamped[k][2]) for k in sorted(clamped.keys())]), INFO)

_derive_memory_controls_from_max_gb()

if options.gene_cor_file is None and options.gene_loc_file is None and not options.ols:
    warn("Switching to run --ols since --gene-cor-file and --gene-loc-file are unspecified")
    options.ols = True

if options.betas_from_phewas:
    options.betas_uncorrected_from_phewas = True

if options.print_effective_config:
    effective_config = {
        "mode": mode,
        "config": options.config,
        "options": _json_safe(vars(options)),
    }
    if factor_workflow is not None:
        effective_config["factor_workflow"] = _json_safe(factor_workflow)
    if eaggl_bundle_info is not None:
        effective_config["eaggl_bundle"] = _json_safe(eaggl_bundle_info.as_dict())
    sys.stdout.write("%s\n" % json.dumps(effective_config, indent=2, sort_keys=True))
    sys.exit(0)

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


def _append_with_any_user(P):
    if P is None:
        return None
    if sparse.issparse(P):
        P = P.todense()
    return np.hstack((P, 1 - np.prod(1 - P, axis=1)[:, np.newaxis]))


class EagglState(object):
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

        pegs_initialize_matrix_and_gene_index_state(self, batch_size=batch_size)

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

        #note that these phewas betas are all stored in *external* units (by contrast to the betas which are in internal units)
        self.X_phewas_beta_uncorrected = None
        self.X_phewas_beta = None

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
        self.inf_betas = None
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
        self.inf_betas_missing = None
        self.non_inf_avg_cond_betas_missing = None
        self.non_inf_avg_postps_missing = None

        self.betas_orig = None
        self.betas_uncorrected_orig = None
        self.inf_betas_orig = None
        self.non_inf_avg_cond_betas_orig = None
        self.non_inf_avg_postps_orig = None

        self.betas_missing_orig = None
        self.betas_uncorrected_missing_orig = None
        self.inf_betas_missing_orig = None
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

        #stores factored matrices
        self.exp_lambdak = None #anchor-agnostic factor relevance weights (does the factor exist)
        self.factor_anchor_relevance = None #relevance of each factor to each anchor
        self.factor_relevance = None #max relevance of each factor across anchors

        #these are specific to the anchor-agnostic loadings
        self.factor_labels = None
        self.factor_labels_gene_sets = None
        self.factor_labels_genes = None
        self.factor_labels_phenos = None

        self.factor_top_gene_sets = None
        self.factor_top_genes = None
        self.factor_top_phenos = None

        #these are specific to anchors
        self.factor_anchor_top_gene_sets = None
        self.factor_anchor_top_genes = None
        self.factor_anchor_top_phenos = None

        #masks used to select inputs to the factoring
        self.gene_factor_gene_mask = None
        self.gene_set_factor_gene_set_mask = None
        self.pheno_factor_pheno_mask = None  #only used in factor pheno mode or factor phewas mode

        self.exp_gene_factors = None #anchor-agnostic factor loadings
        self.gene_prob_factor_vector = None #outer product of this with factor loadings gives anchor specific loadings

        self.exp_gene_set_factors = None  #anchor-agnostic factor loadings
        self.gene_set_prob_factor_vector = None #outer product of this with factor loadings gives anchor specific loadings

        self.exp_pheno_factors = None #anchor-agnostic factor loadings
        self.pheno_prob_factor_vector = None #outer product of this with factor loadings gives anchor specific loadings

        self.factor_phewas_Y_betas = None #phewas statistics
        self.factor_phewas_Y_ses = None #phewas statistics
        self.factor_phewas_Y_zs = None #phewas statistics
        self.factor_phewas_Y_p_values = None #phewas statistics
        self.factor_phewas_Y_one_sided_p_values = None #phewas statistics

        self.factor_phewas_Y_huber_betas = None #phewas statistics
        self.factor_phewas_Y_huber_ses = None #phewas statistics
        self.factor_phewas_Y_huber_zs = None #phewas statistics
        self.factor_phewas_Y_huber_p_values = None #phewas statistics
        self.factor_phewas_Y_huber_one_sided_p_values = None #phewas statistics

        self.factor_phewas_combined_prior_Ys_betas = None #phewas statistics
        self.factor_phewas_combined_prior_Ys_ses = None #phewas statistics
        self.factor_phewas_combined_prior_Ys_zs = None #phewas statistics
        self.factor_phewas_combined_prior_Ys_p_values = None #phewas statistics
        self.factor_phewas_combined_prior_Ys_one_sided_p_values = None #phewas statistics

        self.factor_phewas_combined_prior_Ys_huber_betas = None #phewas statistics
        self.factor_phewas_combined_prior_Ys_huber_ses = None #phewas statistics
        self.factor_phewas_combined_prior_Ys_huber_zs = None #phewas statistics
        self.factor_phewas_combined_prior_Ys_huber_p_values = None #phewas statistics
        self.factor_phewas_combined_prior_Ys_huber_one_sided_p_values = None #phewas statistics

        self.y_state = pegs_sync_y_state(self)
        self.hyperparameter_state = pegs_sync_hyperparameter_state(self)
        self.phewas_state = pegs_sync_phewas_runtime_state(self)

    def has_gene_sets(self):
        return self.X_orig is not None and self.X_orig.shape[1] > 0

    def set_p(self, p):
        hyper_state = pegs_set_runtime_p(self, p)
        self.hyperparameter_state = hyper_state

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

    def _project_H_with_fixed_W(self, W, V_new, P_gene_set, P_gene_new, phi=0.0, lambdak=None, n_iter=100, tol=1e-5, normalize_genes=False, cap_genes=False, add_intercept=False):
        """
        Projects new genes onto the learned NMF factors W using update rules consistent with the original NMF algorithm.

        Parameters:
        - W: numpy array of shape (N, K), fixed basis matrix from NMF.
        - V_new: numpy array of shape (N, M_new), new genes to project (gene sets x new genes).
        - P_gene_set: numpy array of shape (N, U), gene set weights matrix.
        - P_gene_new: numpy array of shape (M_new, U), gene weights matrix for new genes.
        - phi: regularization parameter (default: 0.0).
        - lambdak: numpy array of shape (K,), ARD weights (default: None). If None, set to ones.
        - n_iter: maximum number of iterations (default: 100).
        - tol: tolerance for convergence (default: 1e-5).

        Returns:
        - H_new: numpy array of shape (M_new, K), the loadings for the new genes.
        """

        eps = 1e-10  # Small constant to prevent division by zero

        if add_intercept:
            W = np.hstack((W, np.ones(W.shape[0])[:,np.newaxis]))

        N, K = W.shape  # N: number of gene sets, K: number of latent factors
        N_v, M_new = V_new.shape  # N_v should be equal to N
        assert N == N_v, "V_new (%s,%s) must have the same number of rows as W (%s,%s)" % (V_new.shape[0], V_new.shape[1], W.shape[0], W.shape[1])

        if sparse.issparse(V_new):
            V_new = V_new.toarray()

        P_gene_set = _append_with_any_user(P_gene_set)
        P_gene_new = _append_with_any_user(P_gene_new)

        use_extended = P_gene_new is not None or P_gene_set is not None
        if use_extended:
            if P_gene_new is not None:
                if P_gene_new.ndim == 1:
                    P_gene_new = P_gene_new[:,np.newaxis]
                U = P_gene_new.shape[1]
            else:
                U = P_gene_set.shape[1] if P_gene_set.ndim > 1 else 1
                P_gene_new = np.ones((M_new, U))

            if P_gene_set is not None:
                if P_gene_set.ndim == 1:
                    P_gene_set = P_gene_set[:,np.newaxis]
                assert P_gene_set.shape[1] == U, "P_gene_new (%s, %s) and P_gene_set (%s, %s) must have the same number of users" % (P_gene_new.shape[0], P_gene_new.shape[1], P_gene_set.shape[0], P_gene_set.shape[1])
            else:
                P_gene_set = np.ones((N, U))

            assert P_gene_set.shape == (N, U), f"P_gene_set should have shape ({N}, {U}), not {P_gene_set.shape}"
            assert P_gene_new.shape == (M_new, U), f"P_gene_new should have shape ({M_new}, {U}), not {P_gene_new.shape}"

            # Compute S_new = P_gene_set @ P_gene_new.T, shape (N, M_new)
            S_new = P_gene_set @ P_gene_new.T  # (N x U) @ (U x M_new) -> (N x M_new)
            if sparse.issparse(S_new):
                S_new = S_new.toarray()

            S_new += eps  # Avoid zeros
        else:
            S_new = np.ones_like(V_new)  # If no weighting, S is a matrix of ones


        # Initialize H_new with positive random values, shape (K, M_new)
        V_max = np.max(V_new)
        H_new = np.random.random((K, M_new)) * V_max

        # Initialize lambdak if not provided
        if lambdak is None:
            lambdak = np.ones(K)
        else:
            lambdak = np.array(lambdak)
            assert lambdak.shape == (K,), "lambdak should have shape (K,)"

        # Initialize V_ap_new
        V_ap_new = W @ H_new  # Shape: (N, M_new)

        for it in range(n_iter):
            # Compute numerator and denominator for H_new

            numerator_H = W.T @ (V_new * S_new)  # Shape: (K, M_new)
            denominator_H = W.T @ (V_ap_new * S_new) + phi * H_new * (1 / lambdak)[:, np.newaxis] + eps  # Shape: (K, M_new)

            # Update H_new
            H_new_update = H_new * (numerator_H / denominator_H)

            # Ensure non-negativity
            H_new_update = np.maximum(H_new_update, 0)

            if normalize_genes:
                H_sum = np.sum(H_new_update, axis=0, keepdims=True)
                H_sum[H_sum < 1] = 1  # Avoid division by zero or small numbers
                H_new_update = H_new_update / H_sum

            if cap_genes:
                H_new_update = np.clip(H_new_update, 0, 1)

            # Check convergence
            diff = np.linalg.norm(H_new_update - H_new, 'fro') / (np.linalg.norm(H_new, 'fro') + eps)
            H_new = H_new_update

            # Update V_ap_new
            V_ap_new = W @ H_new

            if diff < tol:
                break

        if add_intercept:
            H_new = H_new[:-1,:]

        return H_new.T


    def _nnls_project_matrix(self, W, X_new, max_iter=500, tol=1e-5, max_value=None, max_sum=None):
        """
        This code was written by GPT-4.

        Parameters:
        - W: numpy array of shape (n_features, n_components), basis matrix from NMF.
        - X_new: numpy array of shape (n_samples, n_features), each row is a new vector to project.
        - max_iter: maximum number of iterations for the multiplicative update.
        - tol: tolerance for convergence.
        - max_value: maximum allowed value for any entry in H_new.

        Returns:
        - H_new: numpy array of shape (n_samples, n_components), the non-negative loadings for each row in X_new.
        """

        orig_vector = False
        if X_new.ndim == 1:
            orig_vector = True
            X_new = X_new[np.newaxis, :]

        # Initialize H_new with random positive values
        n_components = W.shape[1]
        n_samples = X_new.shape[0]
        H_new = np.random.rand(n_samples, n_components)

        # Precompute W^T * W for efficiency
        WT_W = W.T @ W
        if sparse.issparse(WT_W):
            WT_W = WT_W.toarray()

        # Iterative update
        for i in range(max_iter):
            # Compute numerator and denominator
            numerator = X_new @ W
            denominator = H_new @ WT_W + 1e-10  # Small epsilon to avoid division by zero

            # Update H_new
            if sparse.issparse(numerator):
                H_new_update = (numerator.multiply(1.0 / denominator)).multiply(H_new)
                H_new_update = H_new_update.toarray()
            else:
                H_new_update = H_new * (numerator / denominator)

            H_new_update[H_new_update < 0] = 0


            # Apply maximum value cap if specified
            if max_value is not None:
                H_new_update[H_new_update > max_value] = max_value

            # Apply maximum sum cap if specified
            if max_sum is not None:
                H_sums = H_new_update.sum(axis=1)
                above_sum_mask = H_sums > max_sum
                if np.sum(above_sum_mask) > 0:
                    H_new_update[above_sum_mask,:] = (H_new_update[above_sum_mask,:].T / H_sums[above_sum_mask]).T

            # Check for convergence
            norm = np.linalg.norm(H_new_update - H_new, 'fro')

            if norm < tol:
                break

            H_new = H_new_update

        if orig_vector:
            H_new = np.squeeze(H_new, axis=0)

        return H_new

    def _bayes_nmf_l2_extension(self, V0, P_gene_set=None, P_gene=None, n_iter=10000, a0=10, tol=1e-7, K=15,
                                K0=15, phi=1.0, cap_genes=False, normalize_genes=False, cap_gene_sets=False, normalize_gene_sets=False):
        """
        Bayesian NMF with Automatic Relevance Determination (ARD), using Gaussian (L2) likelihood.
        Extended to handle additional weighting matrices P_gene and P_gene_set without materializing large tensors.

        Parameters:
        - V0: Input data matrix (gene sets x genes), containing Poisson rate parameters.
        - P_gene_set: Gene set weights matrix (gene sets x users), shape (N x U), optional.
        - P_gene: Gene weights matrix (genes x users), shape (M x U), optional.
        - Other parameters as before.

        Returns:
        - W: Gene set factor matrix (gene sets x latent factors).
        - H: Gene factor matrix (latent factors x genes).
        - n_like: Final negative log-likelihood.
        - n_evid: Final evidence value.
        - n_lambda: Final ARD weights.
        - n_error: Final reconstruction error.
        """

        eps = 1e-50
        delambda = 1.0

        # Ensure V0 is non-negative
        V = V0 - np.min(V0)
        N, M = V.shape  # Number of gene sets (N) and genes (M)

        # Initialize W and H with positive random values
        Vmax = np.max(V)
        W = np.random.random((N, K)) * Vmax  # Gene sets x latent factors
        H = np.random.random((K, M)) * Vmax  # Latent factors x genes

        V_ap = W @ H + eps  # Initial approximation of V

        # Initialize ARD parameters
        phi = (np.std(V) ** 2) * phi
        C = (N + M) / 2 + a0 + 1
        b0 = 3.14 * (a0 - 1) * np.mean(V) / (2 * K0)
        lambda_bound = b0 / C
        lambdak = (0.5 * (np.sum(W ** 2, axis=0) + np.sum(H ** 2, axis=1)) + b0) / C
        lambda_cut = lambda_bound * 1.5

        n_like = []
        n_evid = []
        n_error = []
        n_lambda = [lambdak]
        it = 1

        P_gene_set = _append_with_any_user(P_gene_set)
        P_gene = _append_with_any_user(P_gene)

        # Check if P_gene and P_gene_set are specified
        use_extended = P_gene is not None or P_gene_set is not None
        if use_extended:
            if P_gene is not None:
                if P_gene.ndim == 1:
                    P_gene = P_gene[:,np.newaxis]
                U = P_gene.shape[1]
            else:
                U = P_gene_set.shape[1] if P_gene_set.ndim > 1 else 1
                P_gene = np.ones((M, U))

            if P_gene_set is not None:
                if P_gene_set.ndim == 1:
                    P_gene_set = P_gene_set[:,np.newaxis]
                assert P_gene_set.shape[1] == U, f"P_gene ({P_gene.shape[0]},{P_gene.shape[1]}) and P_gene_set ({P_gene_set.shape[0]},{P_gene_set.shape[1]}) must have the same number of users"
            else:
                P_gene_set = np.ones((N, U))

            assert P_gene_set.shape == (N, U), f"P_gene_set should have shape ({N}, {U}), not {P_gene_set.shape}"
            assert P_gene.shape == (M, U), f"P_gene should have shape ({M}, {U}), not {P_gene.shape}"

            # Compute S = P_gene_set @ P_gene.T (N x U) @ (U x M) -> N x M
            S = P_gene_set @ P_gene.T  # Weighting matrix S of shape (N x M)
        else:
            S = np.ones_like(V)  # If no weighting, S is a matrix of ones

        if sparse.issparse(S):
            S = S.toarray()

        while delambda >= tol and it <= n_iter:
            # Update H
            numerator_H = W.T @ (V * S)
            denominator_H = W.T @ (V_ap * S) + phi * H * (1 / lambdak)[:, np.newaxis] + eps
            H *= numerator_H / denominator_H
            H = np.maximum(H, 0)

            if normalize_genes:
                H_sum = np.sum(H, axis=0, keepdims=True)
                H_sum[H_sum < 1] = 1  # Avoid division by zero or small numbers
                H = H / H_sum
            if cap_genes:
                H = np.clip(H, 0, 1)

            V_ap = W @ H + eps  # Update approximation

            # Update W
            numerator_W = (V * S) @ H.T
            denominator_W = (V_ap * S) @ H.T + phi * W * (1 / lambdak)[np.newaxis, :] + eps
            W *= numerator_W / denominator_W
            W = np.maximum(W, 0)

            if normalize_gene_sets:
                W_sum = np.sum(W, axis=1, keepdims=True)
                W_sum[W_sum < 1] = 1  # Avoid division by zero or small numbers
                W = W / W_sum
            if cap_gene_sets:
                W = np.clip(W, 0, 1)

            V_ap = W @ H + eps  # Update approximation

            # Compute Gaussian negative log-likelihood
            like = np.sum(0.5 * S * (V - V_ap) ** 2)

            # Update ARD weights
            lambdak_new = (0.5 * (np.sum(W ** 2, axis=0) +
                                  np.sum(H ** 2, axis=1)) + b0) / C
            delambda = np.max(np.abs(lambdak_new - lambdak) / (lambdak + eps))
            lambdak = lambdak_new

            # Compute evidence and error
            regularization = phi * np.sum((0.5 * (np.sum(W ** 2, axis=0) +
                                                   np.sum(H ** 2, axis=1)) + b0) / lambdak +
                                          C * np.log(lambdak))
            evid = like + regularization
            error = np.sum(S * (V - V_ap) ** 2)

            n_like.append(like)
            n_evid.append(evid)
            n_lambda.append(lambdak)
            n_error.append(error)

            if it % 100 == 0 or it == 1 or delambda < tol:
                factors = np.sum(np.sum(W, axis=0) != 0)
                factors_non_zero = np.sum(lambdak >= lambda_cut)
                log(f"Iteration={it}; evid={evid:.3g}; lik={like:.3g}; err={error:.3g}; delambda={delambda:.3g}; factors={factors}; factors_non_zero={factors_non_zero}")
            it += 1

        # Return the results
        return W, H, n_like[-1], n_evid[-1], n_lambda[-1], n_error[-1]

    #this code is adapted from https://github.com/gwas-partitioning/bnmf-clustering
    def num_factors(self):
        if self.exp_lambdak is None:
            return 0
        else:
            return len(self.exp_lambdak)

    #get raw, specific, or combined factor loadings

    def get_factor_loadings(self, loadings, loading_type='raw', specific_weight=0.5):

        if loadings is None:
            return None
        
        assert(loading_type == 'raw' or loading_type == 'specific' or loading_type == 'combined')
        
        if loading_type == 'raw':
            return loadings 
        else:
            specific_loadings = (loadings.T / (np.sum(loadings, axis=1) + 1e-10)).T
            if loading_type == 'specific':
                return specific_loadings
            else:
                if specific_weight < 0 or specific_weight > 1:
                    bail("Specific weight must be between 0 and 1")
                return (1 - specific_weight) * loadings + specific_weight * specific_loadings


    def run_factor(self, max_num_factors=15, phi=1.0, alpha0=10, beta0=1, gene_set_filter_type=None, gene_set_filter_value=None, gene_or_pheno_filter_type=None, gene_or_pheno_filter_value=None, pheno_prune_value=None, pheno_prune_number=None, gene_prune_value=None, gene_prune_number=None, gene_set_prune_value=None, gene_set_prune_number=None, anchor_pheno_mask=None, anchor_gene_mask=None, anchor_any_pheno=False, anchor_any_gene=False, anchor_gene_set=False, run_transpose=True, max_num_iterations=100, rel_tol=1e-4, min_lambda_threshold=1e-3, lmm_auth_key=None, lmm_model=None, lmm_provider="openai", label_gene_sets_only=False, label_include_phenos=False, label_individually=False, keep_original_loadings=False, project_phenos_from_gene_sets=False):

        if self.X_orig is None:
            bail("Cannot run factoring without X")

        # Persist explicit anchor masks for downstream output writers.
        self.anchor_pheno_mask = np.copy(anchor_pheno_mask) if anchor_pheno_mask is not None else None
        self.anchor_gene_mask = np.copy(anchor_gene_mask) if anchor_gene_mask is not None else None

        if (anchor_any_gene or anchor_any_pheno or anchor_gene_set or anchor_gene_mask is not None or anchor_pheno_mask is not None or pheno_prune_value is not None or pheno_prune_number is not None) and self.X_phewas_beta is None:
            bail("Cannot run factoring without X phewas")

        if anchor_any_gene:
            if anchor_any_pheno:
                warn("Ignoring anchor any pheno since anchor any gene was specified")
            if anchor_gene_mask:
                warn("Ignoring anchor gene since anchor any gene was specified")
            if anchor_pheno_mask:
                warn("Ignoring anchor pheno since anchor any gene was specified")
            if anchor_gene_set:
                warn("Ignoring anchor gene set since anchor any gene was specified")

            self._record_params({"anchor": "any_gene"})
            anchor_any_pheno = False
            anchor_pheno_mask = None
            anchor_gene_mask = np.full(self.X_orig.shape[0], True)
            anchor_gene_set = False

        elif anchor_any_pheno:
            if anchor_gene_mask:
                warn("Ignoring anchor gene since anchor any pheno was specified")
            if anchor_pheno_mask:
                warn("Ignoring anchor pheno since anchor any pheno was specified")
            if anchor_gene_set:
                warn("Ignoring anchor gene set since anchor any pheno was specified")
            anchor_gene_mask = None
            anchor_pheno_mask = np.full(self.X_phewas_beta.shape[0], True)
            anchor_gene_set = False
            self._record_params({"anchor": "any_pheno"})
        elif anchor_gene_set:
            if anchor_gene_mask:
                warn("Ignoring anchor gene since anchor gene set was specified")
            if anchor_pheno_mask:
                warn("Ignoring anchor pheno since anchor gene set was specified")
            anchor_gene_mask = None
            anchor_pheno_mask = None
            self._record_params({"anchor": "gene set"})

        # Record the effective anchor masks after option precedence is resolved.
        self.anchor_pheno_mask = np.copy(anchor_pheno_mask) if anchor_pheno_mask is not None else None
        self.anchor_gene_mask = np.copy(anchor_gene_mask) if anchor_gene_mask is not None else None

        #ensure at most one anchor mask, and initialize the matrix mask accordingly
        #remember that single pheno anchoring mode is implicit and doesn't have the anchor mask defined
        num_users = 1
        anchor_mask = None
        factor_gene_set_x_pheno = False
        pheno_Y = None

        if anchor_gene_mask is not None or anchor_gene_set:
            if anchor_pheno_mask is not None:
                warn("Ignoring anchor pheno since anchor gene or anchor gene set was specified")
                anchor_pheno_mask = None
            gene_or_pheno_mask = np.full(self.X_phewas_beta.shape[0], True)
            gene_set_mask = np.full(self.X_phewas_beta.shape[1], True)
            factor_gene_set_x_pheno = True

            combined_prior_Ys = self.gene_pheno_combined_prior_Ys.T if self.gene_pheno_combined_prior_Ys is not None else None
            priors = self.gene_pheno_priors.T if self.gene_pheno_priors is not None else None
            Y = self.gene_pheno_Y.T if self.gene_pheno_Y is not None else None

            self._record_params({"factor_gene_vectors": "gene_pheno.T"})

            if anchor_gene_mask is not None:
                betas = None
                betas_uncorrected = None

                anchor_mask = anchor_gene_mask
                num_users = np.sum(anchor_mask)
                self._record_params({"factor_gene_set_vectors": "None"})

            else:
                #we need to set things up below
                #we are going to construct a pheno x gene set matrix, using the X_phewas as input
                #we need to have weights for the rows (phenos) and columns (gene sets)
                #the column weights need to be the betas

                anchor_gene_mask = np.full(1, True)
                anchor_mask = anchor_gene_mask
                num_users = 1

                #for the gene set mode, we use the pheno_Y for weights, and do a special setting below
                #we need to keep combined_prior_Y for projecting, but use pheno_Y for weighting
                pheno_Y = self.pheno_Y_vs_input_combined_prior_Ys_beta if self.pheno_Y_vs_input_combined_prior_Ys_beta is not None else self.pheno_Y_vs_input_Y_beta if self.pheno_Y_vs_input_Y_beta is not None else self.pheno_Y_vs_input_priors_beta
                if pheno_Y is not None:
                    pheno_Y = pheno_Y[:,np.newaxis]
                
                #betas are in external units
                betas = (self.betas / self.scale_factors)[:,np.newaxis] if self.betas is not None else None
                betas_uncorrected = (self.betas_uncorrected / self.scale_factors)[:,np.newaxis] if self.betas_uncorrected is not None else None
                self._record_params({"factor_gene_set_vectors": "betas"})

        else:
            if anchor_pheno_mask is not None and anchor_gene_mask is not None:
                warn("Ignoring anchor gene since anchor pheno was specified")
            anchor_gene_mask = None
            gene_or_pheno_mask = np.full(self.X_orig.shape[0], True)
            gene_set_mask = np.full(self.X_orig.shape[1], True)
            if anchor_pheno_mask is not None:

                anchor_mask = anchor_pheno_mask

                combined_prior_Ys = self.gene_pheno_combined_prior_Ys
                priors = self.gene_pheno_priors
                Y = self.gene_pheno_Y

                self._record_params({"factor_gene_vectors": "gene_pheno"})
                betas = self.X_phewas_beta.T if self.X_phewas_beta is not None else None
                betas_uncorrected = self.X_phewas_beta_uncorrected.T if self.X_phewas_beta_uncorrected is not None else None
                self._record_params({"factor_gene_set_vectors": "X_phewas"})

            else:

                combined_prior_Ys = self.combined_prior_Ys[:,np.newaxis] if self.combined_prior_Ys is not None else None
                priors = self.priors[:,np.newaxis] if self.priors is not None else None
                Y = self.Y[:,np.newaxis] if self.Y is not None else None

                self._record_params({"factor_gene_vectors": "Y"})

                betas = (self.betas / self.scale_factors)[:,np.newaxis] if self.betas is not None else None
                betas_uncorrected = (self.betas_uncorrected / self.scale_factors)[:,np.newaxis] if self.betas_uncorrected is not None else None

                self._record_params({"factor_gene_set_vectors": "betas"})


                #when running the original factoring based off the internal betas and gene scores, we are going to emulate the phewas-like behavior by appending these as the only anchor alongside any gene/pheno loaded values
                #this will allow projection to other phenotypes to happen naturally below
                anchor_mask = np.full(1, True)

                have_phewas = False
                if combined_prior_Ys is not None and self.gene_pheno_combined_prior_Ys is not None:
                    combined_prior_Ys = sparse.hstack((self.gene_pheno_combined_prior_Ys, sparse.csc_matrix(combined_prior_Ys))).tocsc()
                    have_phewas = True
                if priors is not None and self.gene_pheno_priors is not None:
                    priors = sparse.hstack((self.gene_pheno_priors, sparse.csc_matrix(priors))).tocsc()
                    have_phewas = True
                if Y is not None and self.gene_pheno_Y is not None:
                    Y = sparse.hstack((self.gene_pheno_Y, sparse.csc_matrix(Y))).tocsc()
                    have_phewas = True

                if betas is not None and self.X_phewas_beta is not None:
                    betas = sparse.hstack((self.X_phewas_beta.T, sparse.csc_matrix(betas))).tocsc()
                    have_phewas = True
                if betas_uncorrected is not None and self.X_phewas_beta_uncorrected is not None:
                    betas_uncorrected = sparse.hstack((self.X_phewas_beta_uncorrected.T, sparse.csc_matrix(betas_uncorrected))).tocsc()
                    have_phewas = True

                if have_phewas:
                    #we have phewas for at least one of combined, prior, or Y
                    #set those that don't to None
                    #otherwise update the internal structures
                    if combined_prior_Ys is not None and combined_prior_Ys.shape[1] == 1:
                        combined_prior_Ys = None
                    else:
                        self.gene_pheno_combined_prior_Ys = combined_prior_Ys
                        
                    if priors is not None and priors.shape[1] == 1:
                        priors = None
                    else:
                        self.gene_pheno_priors = priors

                    if Y is not None and Y.shape[1] == 1:
                        Y = None
                    else:
                        self.gene_pheno_Y = Y
                    if betas is not None and betas.shape[1] == 1:
                        betas = None
                    else:
                        self.X_phewas_beta = betas.T
                    if betas_uncorrected is not None and betas_uncorrected.shape[1] == 1:
                        betas_uncorrected = None
                    else:
                        self.X_phewas_beta_uncorrected = betas_uncorrected.T

                    self.phenos.append(self.default_pheno)
                    self.default_pheno_mask = np.append(np.full(len(self.phenos), False), True)

                    #we need to update these as well
                    self.pheno_Y_vs_input_Y_beta = np.append(self.pheno_Y_vs_input_Y_beta, 0) if self.pheno_Y_vs_input_Y_beta is not None else None
                    self.pheno_Y_vs_input_Y_beta_tilde = np.append(self.pheno_Y_vs_input_Y_beta_tilde, 0) if self.pheno_Y_vs_input_Y_beta_tilde is not None else None
                    self.pheno_Y_vs_input_Y_se = np.append(self.pheno_Y_vs_input_Y_se, 0) if self.pheno_Y_vs_input_Y_se is not None else None
                    self.pheno_Y_vs_input_Y_Z = np.append(self.pheno_Y_vs_input_Y_Z, 0) if self.pheno_Y_vs_input_Y_Z is not None else None
                    self.pheno_Y_vs_input_Y_p_value = np.append(self.pheno_Y_vs_input_Y_p_value, 1) if self.pheno_Y_vs_input_Y_p_value is not None else None

                    self.pheno_combined_prior_Ys_vs_input_Y_beta = np.append(self.pheno_combined_prior_Ys_vs_input_Y_beta, 0) if self.pheno_combined_prior_Ys_vs_input_Y_beta is not None else None
                    self.pheno_combined_prior_Ys_vs_input_Y_beta_tilde = np.append(self.pheno_combined_prior_Ys_vs_input_Y_beta_tilde, 0) if self.pheno_combined_prior_Ys_vs_input_Y_beta_tilde is not None else None
                    self.pheno_combined_prior_Ys_vs_input_Y_se = np.append(self.pheno_combined_prior_Ys_vs_input_Y_se, 0) if self.pheno_combined_prior_Ys_vs_input_Y_se is not None else None
                    self.pheno_combined_prior_Ys_vs_input_Y_Z = np.append(self.pheno_combined_prior_Ys_vs_input_Y_Z, 0) if self.pheno_combined_prior_Ys_vs_input_Y_Z is not None else None
                    self.pheno_combined_prior_Ys_vs_input_Y_p_value = np.append(self.pheno_combined_prior_Ys_vs_input_Y_p_value, 1) if self.pheno_combined_prior_Ys_vs_input_Y_p_value is not None else None

                    self.pheno_Y_vs_input_combined_prior_Ys_beta = np.append(self.pheno_Y_vs_input_combined_prior_Ys_beta, 0) if self.pheno_Y_vs_input_combined_prior_Ys_beta is not None else None
                    self.pheno_Y_vs_input_combined_prior_Ys_beta_tilde = np.append(self.pheno_Y_vs_input_combined_prior_Ys_beta_tilde, 0) if self.pheno_Y_vs_input_combined_prior_Ys_beta_tilde is not None else None
                    self.pheno_Y_vs_input_combined_prior_Ys_se = np.append(self.pheno_Y_vs_input_combined_prior_Ys_se, 0) if self.pheno_Y_vs_input_combined_prior_Ys_se is not None else None
                    self.pheno_Y_vs_input_combined_prior_Ys_Z = np.append(self.pheno_Y_vs_input_combined_prior_Ys_Z, 0) if self.pheno_Y_vs_input_combined_prior_Ys_Z is not None else None
                    self.pheno_Y_vs_input_combined_prior_Ys_p_value = np.append(self.pheno_Y_vs_input_combined_prior_Ys_p_value, 1) if self.pheno_Y_vs_input_combined_prior_Ys_p_value is not None else None

                    self.pheno_combined_prior_Ys_vs_input_combined_prior_Ys_beta = np.append(self.pheno_combined_prior_Ys_vs_input_combined_prior_Ys_beta, 0) if self.pheno_combined_prior_Ys_vs_input_combined_prior_Ys_beta is not None else None
                    self.pheno_combined_prior_Ys_vs_input_combined_prior_Ys_beta_tilde = np.append(self.pheno_combined_prior_Ys_vs_input_combined_prior_Ys_beta_tilde, 0) if self.pheno_combined_prior_Ys_vs_input_combined_prior_Ys_beta_tilde is not None else None
                    self.pheno_combined_prior_Ys_vs_input_combined_prior_Ys_se = np.append(self.pheno_combined_prior_Ys_vs_input_combined_prior_Ys_se, 0) if self.pheno_combined_prior_Ys_vs_input_combined_prior_Ys_se is not None else None
                    self.pheno_combined_prior_Ys_vs_input_combined_prior_Ys_Z = np.append(self.pheno_combined_prior_Ys_vs_input_combined_prior_Ys_Z, 0) if self.pheno_combined_prior_Ys_vs_input_combined_prior_Ys_Z is not None else None
                    self.pheno_combined_prior_Ys_vs_input_combined_prior_Ys_p_value = np.append(self.pheno_combined_prior_Ys_vs_input_combined_prior_Ys_p_value, 1) if self.pheno_combined_prior_Ys_vs_input_combined_prior_Ys_p_value is not None else None

                    self.pheno_Y_vs_input_priors_beta = np.append(self.pheno_Y_vs_input_priors_beta, 0) if self.pheno_Y_vs_input_priors_beta is not None else None
                    self.pheno_Y_vs_input_priors_beta_tilde = np.append(self.pheno_Y_vs_input_priors_beta_tilde, 0) if self.pheno_Y_vs_input_priors_beta_tilde is not None else None
                    self.pheno_Y_vs_input_priors_se = np.append(self.pheno_Y_vs_input_priors_se, 0) if self.pheno_Y_vs_input_priors_se is not None else None
                    self.pheno_Y_vs_input_priors_Z = np.append(self.pheno_Y_vs_input_priors_Z, 0) if self.pheno_Y_vs_input_priors_Z is not None else None
                    self.pheno_Y_vs_input_priors_p_value = np.append(self.pheno_Y_vs_input_priors_p_value, 1) if self.pheno_Y_vs_input_priors_p_value is not None else None

                    self.pheno_combined_prior_Ys_vs_input_priors_beta = np.append(self.pheno_combined_prior_Ys_vs_input_priors_beta, 0) if self.pheno_combined_prior_Ys_vs_input_priors_beta is not None else None
                    self.pheno_combined_prior_Ys_vs_input_priors_beta_tilde = np.append(self.pheno_combined_prior_Ys_vs_input_priors_beta_tilde, 0) if self.pheno_combined_prior_Ys_vs_input_priors_beta_tilde is not None else None
                    self.pheno_combined_prior_Ys_vs_input_priors_se = np.append(self.pheno_combined_prior_Ys_vs_input_priors_se, 0) if self.pheno_combined_prior_Ys_vs_input_priors_se is not None else None
                    self.pheno_combined_prior_Ys_vs_input_priors_Z = np.append(self.pheno_combined_prior_Ys_vs_input_priors_Z, 0) if self.pheno_combined_prior_Ys_vs_input_priors_Z is not None else None
                    self.pheno_combined_prior_Ys_vs_input_priors_p_value = np.append(self.pheno_combined_prior_Ys_vs_input_priors_p_value, 1) if self.pheno_combined_prior_Ys_vs_input_priors_p_value is not None else None

                    if combined_prior_Ys is None and priors is None and Y is None:
                        bail("Need to load gene phewas stats if you are loading gene set phewas stats")
                    if betas is None and betas_uncorrected is None:
                        bail("Need to load gene set phewas stats if you are loading gene phewas stats")
                    
                #the newly appended ones are not anchors
                anchor_mask = np.append(np.full((combined_prior_Ys.shape[1] if combined_prior_Ys is not None else priors.shape[1] if priors is not None else Y.shape[1] if Y is not None else 1) - 1, False), anchor_mask)


            num_users = np.sum(anchor_pheno_mask)

        #get one dimensional vectors with probabilities
        gene_or_pheno_full_vector = combined_prior_Ys if combined_prior_Ys is not None else priors if priors is not None else Y if Y is not None else None

        gene_or_pheno_vector = None
        if anchor_gene_set:
            gene_or_pheno_vector = pheno_Y
        else:
            if gene_or_pheno_full_vector is not None:
                gene_or_pheno_vector = gene_or_pheno_full_vector[:,anchor_mask]

        if gene_or_pheno_vector is not None:
            if sparse.issparse(gene_or_pheno_vector):
                gene_or_pheno_vector = gene_or_pheno_vector.toarray()

        gene_or_pheno_filter_type = "combined_prior_Ys" if combined_prior_Ys is not None else "priors" if priors is not None else "Y" if Y is not None else None        

        #now get the aggregations and masks
        gene_or_pheno_max_vector = np.max(gene_or_pheno_vector, axis=1) if gene_or_pheno_vector is not None else None

        if gene_or_pheno_max_vector is not None and gene_or_pheno_filter_value is not None:
            gene_or_pheno_mask = gene_or_pheno_max_vector > gene_or_pheno_filter_value

        def __combine_prune_masks(prune_masks, prune_number, sort_rank, tag):
            if prune_masks is None or len(prune_masks) == 0:
                return None
            all_prune_mask = np.full(len(prune_masks[0]), False)
            for cur_prune_mask in prune_masks:
                all_prune_mask[cur_prune_mask] = True
                log("Adding %d relatively uncorrelated %ss (total now %d)" % (np.sum(cur_prune_mask), tag, np.sum(all_prune_mask)), TRACE)
                if np.sum(all_prune_mask) > prune_number:
                    break
            if np.sum(all_prune_mask) > prune_number:
                threshold_value = sorted(sort_rank[all_prune_mask])[prune_number - 1]
                all_prune_mask[sort_rank > threshold_value] = False
            if np.sum(~all_prune_mask) > 0:
                log("Found %d %ss remaining after pruning to max number (of %d)" % (np.sum(all_prune_mask), tag, len(self.phenos)))
            return all_prune_mask

        if pheno_prune_value is not None or pheno_prune_number is not None:
            mask_for_pruning = gene_or_pheno_mask if factor_gene_set_x_pheno else anchor_pheno_mask
            if mask_for_pruning is not None:
            
                if factor_gene_set_x_pheno:
                    log("Pruning phenos to reduce matrix size", DEBUG)
                else:
                    log("Pruning phenos to reduce number of anchors", DEBUG)                    

                pheno_sort_rank = -self.X_phewas_beta.mean(axis=1).A1 if self.X_phewas_beta is not None else np.arange(len(mask_for_pruning))
                #now if we request pruning
                if pheno_prune_value is not None:
                    pheno_prune_mask = self._prune_gene_sets(pheno_prune_value, X_orig=self.X_phewas_beta_uncorrected[mask_for_pruning,:].T, gene_sets=[self.phenos[i] for i in np.where(mask_for_pruning)[0]], rank_vector=pheno_sort_rank[mask_for_pruning], do_internal_pruning=False)
                    log("Found %d phenos remaining after pruning (of %d)" % (np.sum(pheno_prune_mask), len(self.phenos)))

                    mask_for_pruning[np.where(mask_for_pruning)[0][~pheno_prune_mask]] = False

                if pheno_prune_number is not None:
                    (mean_shifts, scale_factors) = self._calc_X_shift_scale(self.X_phewas_beta_uncorrected[mask_for_pruning,:].T)
                    pheno_prune_number_masks = self._compute_gene_set_batches(V=None, X_orig=self.X_phewas_beta_uncorrected[mask_for_pruning,:].T, mean_shifts=mean_shifts, scale_factors=scale_factors, sort_values=pheno_sort_rank[mask_for_pruning], stop_at=pheno_prune_number, tag="phenos")
                    all_pheno_prune_mask = __combine_prune_masks(pheno_prune_number_masks, pheno_prune_number, pheno_sort_rank[mask_for_pruning], "pheno")
                    mask_for_pruning[np.where(mask_for_pruning)[0][~all_pheno_prune_mask]] = False
                if mask_for_pruning is anchor_pheno_mask and num_users > 1:
                    #in this case, we may have changed the number of users
                    num_users = np.sum(anchor_pheno_mask)

        if not anchor_gene_set and (gene_prune_value is not None or gene_prune_number is not None):
            mask_for_pruning = gene_or_pheno_mask if not factor_gene_set_x_pheno else anchor_gene_mask
            if mask_for_pruning is not None:
                gene_sort_rank = -self.combined_prior_Ys if self.combined_prior_Ys is not None else -self.Y if self.Y is not None else -self.priors if self.priors is not None else np.arange(len(mask_for_pruning))
                if not factor_gene_set_x_pheno:
                    log("Pruning genes to reduce matrix size", DEBUG)
                else:
                    log("Pruning genes to reduce number of anchors", DEBUG)                    


                #now if we request pruning
                if gene_prune_value is not None:
                    gene_prune_mask = self._prune_gene_sets(gene_prune_value, X_orig=self.X_orig[mask_for_pruning,:].T, gene_sets=[self.genes[i] for i in np.where(mask_for_pruning)[0]], rank_vector=gene_sort_rank[mask_for_pruning], do_internal_pruning=False)
                    log("Found %d genes remaining after pruning (of %d)" % (np.sum(gene_prune_mask), len(self.genes)))

                    mask_for_pruning[np.where(mask_for_pruning)[0][~gene_prune_mask]] = False

                if gene_prune_number is not None:
                    (mean_shifts, scale_factors) = self._calc_X_shift_scale(self.X_orig[mask_for_pruning,:].T)
                    gene_prune_number_masks = self._compute_gene_set_batches(V=None, X_orig=self.X_orig[mask_for_pruning,:].T, mean_shifts=mean_shifts, scale_factors=scale_factors, sort_values=gene_sort_rank[mask_for_pruning], stop_at=gene_prune_number, tag="genes")
                    all_gene_prune_mask = __combine_prune_masks(gene_prune_number_masks, gene_prune_number, gene_sort_rank[mask_for_pruning], "gene")
                    mask_for_pruning[np.where(mask_for_pruning)[0][~all_gene_prune_mask]] = False

                if mask_for_pruning is anchor_gene_mask and num_users > 1:
                    #in this case, we may have changed the number of users
                    num_users = np.sum(anchor_gene_mask)

        #add in the any vectors
        gene_or_pheno_full_prob_vector = None
        if gene_or_pheno_full_vector is not None:
            #we are going to approximate things below the threshold as zero probability, and not fold those in the background prior
            #to get around this we would have to use a dense matrix
            if sparse.issparse(gene_or_pheno_full_vector):
                gene_or_pheno_full_prob_vector_data = np.exp(gene_or_pheno_full_vector.data + self.background_log_bf)
                gene_or_pheno_full_prob_vector_data = gene_or_pheno_full_prob_vector_data / (1 + gene_or_pheno_full_prob_vector_data)
                gene_or_pheno_full_prob_vector = copy.copy(gene_or_pheno_full_vector)
                gene_or_pheno_full_prob_vector.data = gene_or_pheno_full_prob_vector_data
            else:
                gene_or_pheno_full_prob_vector = np.exp(gene_or_pheno_full_vector + self.background_log_bf) / (1 + np.exp(gene_or_pheno_full_vector + self.background_log_bf))

        if anchor_gene_set:
            gene_or_pheno_prob_vector = np.exp(gene_or_pheno_vector + self.background_log_bf) / (1 + np.exp(gene_or_pheno_vector + self.background_log_bf)) if gene_or_pheno_vector is not None else np.ones((len(gene_or_pheno_mask), num_users))
        else:
            gene_or_pheno_prob_vector = gene_or_pheno_full_prob_vector[:,anchor_mask] if gene_or_pheno_full_prob_vector is not None else np.ones((len(gene_or_pheno_mask), num_users))

        if gene_or_pheno_prob_vector is not None and sparse.issparse(gene_or_pheno_prob_vector):
            gene_or_pheno_prob_vector = gene_or_pheno_prob_vector.toarray()

        if anchor_any_gene or anchor_any_pheno:
            #only have one user
            gene_or_pheno_any_prob_vector = 1 - np.prod(1 - gene_or_pheno_prob_vector, axis=1)
            gene_or_pheno_prob_vector = gene_or_pheno_any_prob_vector[:,np.newaxis]

        if factor_gene_set_x_pheno:
            self.pheno_prob_factor_vector = gene_or_pheno_prob_vector
            self.gene_prob_factor_vector = None
        else:
            self.gene_prob_factor_vector = gene_or_pheno_prob_vector
            self.pheno_prob_factor_vector = None

        #now do the gene set vectors and masks
        #normalize
        gene_set_full_vector = betas_uncorrected if betas_uncorrected is not None else betas
        gene_set_vector = None
        if gene_set_full_vector is not None:
            gene_set_vector = gene_set_full_vector[:,anchor_mask]
            if sparse.issparse(gene_set_vector):
                gene_set_vector = gene_set_vector.toarray()

        gene_set_filter_type = "betas_uncorrected" if betas_uncorrected is not None else "betas"
        gene_set_max_vector = np.max(gene_set_vector, axis=1) if gene_set_vector is not None else None

        if gene_set_max_vector is not None and gene_set_filter_value is not None:
            gene_set_mask = gene_set_max_vector > gene_set_filter_value


        gene_set_sort_rank = -(self.X_phewas_beta_uncorrected.mean(axis=0).A1 if self.X_phewas_beta_uncorrected is not None else self.betas)

        if gene_set_prune_value is not None or gene_set_prune_number is not None:
            log("Pruning gene sets to reduce matrix size", DEBUG)

        if gene_set_prune_value is not None:
            gene_set_prune_mask = self._prune_gene_sets(gene_set_prune_value, X_orig=self.X_orig[:,gene_set_mask], gene_sets=[self.gene_sets[i] for i in np.where(gene_set_mask)[0]], rank_vector=gene_set_sort_rank[gene_set_mask], do_internal_pruning=False)

            log("Found %d gene_sets remaining after pruning (of %d)" % (np.sum(gene_set_prune_mask), len(self.gene_sets)))
            gene_set_mask[np.where(gene_set_mask)[0][~gene_set_prune_mask]] = False

        if gene_set_prune_number is not None:
            gene_set_prune_number_masks = self._compute_gene_set_batches(V=None, X_orig=self.X_orig[:,gene_set_mask], mean_shifts=self.mean_shifts[gene_set_mask], scale_factors=self.scale_factors[gene_set_mask], sort_values=gene_set_sort_rank[gene_set_mask], stop_at=pheno_prune_number, tag="gene sets")

            all_gene_set_prune_mask = __combine_prune_masks(gene_set_prune_number_masks, gene_set_prune_number, gene_set_sort_rank[gene_set_mask], "gene set")

            gene_set_mask[np.where(gene_set_mask)[0][~all_gene_set_prune_mask]] = False
        
        gene_set_full_prob_vector = None
        if gene_set_full_vector is not None:
            if sparse.issparse(gene_set_full_vector):
                gene_set_full_prob_vector_data = np.exp(gene_set_full_vector.data + self.background_log_bf)
                gene_set_full_prob_vector_data = gene_set_full_prob_vector_data / (1 + gene_set_full_prob_vector_data)
                gene_set_full_prob_vector = copy.copy(gene_set_full_vector)
                gene_set_full_prob_vector.data = gene_set_full_prob_vector_data
            else:
                gene_set_full_prob_vector = np.exp(gene_set_full_vector + self.background_log_bf) / (1 + np.exp(gene_set_full_vector + self.background_log_bf))

        gene_set_prob_vector = gene_set_full_prob_vector[:,anchor_mask] if gene_set_full_prob_vector is not None else np.ones((len(gene_set_mask), num_users))

        if gene_set_prob_vector is not None and sparse.issparse(gene_set_prob_vector):
            gene_set_prob_vector = gene_set_prob_vector.toarray()

        if anchor_any_gene or anchor_any_pheno:
            #only have one user
            gene_set_any_prob_vector = 1 - np.prod(1 - gene_set_prob_vector, axis=1)
            gene_set_prob_vector = gene_set_any_prob_vector[:,np.newaxis]

        self.gene_set_prob_vector = gene_set_full_prob_vector

        self._record_params({"max_num_factors": max_num_factors, "alpha0": alpha0, "phi": phi, "gene_set_filter_type": gene_set_filter_type, "gene_set_filter_value": gene_set_filter_value, "gene_or_pheno_filter_type": gene_or_pheno_filter_type, "gene_or_pheno_filter_value": gene_or_pheno_filter_value, "pheno_prune_value": pheno_prune_value, "pheno_prune_number": pheno_prune_number, "gene_set_prune_value": gene_set_prune_value, "gene_set_prune_number": gene_set_prune_number, "run_transpose": run_transpose})


        matrix = self.X_phewas_beta_uncorrected.T if factor_gene_set_x_pheno else self.X_orig.T

        matrix = matrix[gene_set_mask,:][:,gene_or_pheno_mask]
        matrix[matrix < 0] = 0
        if not run_transpose:
            matrix = matrix.T

        log("Running matrix factorization")
        if np.sum(~gene_or_pheno_mask) > 0 or np.sum(~gene_set_mask) > 0:
            log("Filtered original matrix from (%s, %s) to (%s, %s)" % (len(gene_or_pheno_mask), len(gene_set_mask), sum(gene_or_pheno_mask), sum(gene_set_mask)))
        log("Matrix to factor shape: (%s, %s)" % (matrix.shape), DEBUG)

        if np.max(matrix.shape) == 0:
            log("Skipping factoring since there aren't enough significant genes and gene sets")
            return

        if np.min(matrix.shape) == 0:
            log("Empty genes or gene sets! Skipping factoring")
            return

        #constrain loadings to be at most 1, but don't require them to sum to 1
        normalize_genes = False
        normalize_gene_sets = False
        cap = True

        result = self._bayes_nmf_l2_extension(matrix.toarray(), gene_set_prob_vector[gene_set_mask,:], gene_or_pheno_prob_vector[gene_or_pheno_mask,:], a0=alpha0, K=max_num_factors, tol=rel_tol, phi=phi, cap_genes=cap, cap_gene_sets=cap, normalize_genes=normalize_genes, normalize_gene_sets=normalize_gene_sets)

        self.exp_lambdak = result[4]
        exp_gene_or_pheno_factors = result[1].T
        self.exp_gene_set_factors = result[0]

        #subset_out the weak factors
        factor_mask = (self.exp_lambdak > min_lambda_threshold) & (np.sum(exp_gene_or_pheno_factors, axis=0) > min_lambda_threshold) & (np.sum(self.exp_gene_set_factors, axis=0) > min_lambda_threshold)
        factor_mask = factor_mask & (np.max(self.exp_gene_set_factors, axis=0) > 1e-5 * np.max(self.exp_gene_set_factors))
        if np.sum(~factor_mask) > 0:
            self.exp_lambdak = self.exp_lambdak[factor_mask]
            exp_gene_or_pheno_factors = exp_gene_or_pheno_factors[:,factor_mask]
            self.exp_gene_set_factors = self.exp_gene_set_factors[:,factor_mask]

        if factor_gene_set_x_pheno:
            self.pheno_factor_pheno_mask = gene_or_pheno_mask
            self.exp_pheno_factors = exp_gene_or_pheno_factors
            self.pheno_prob_factor_vector = gene_or_pheno_prob_vector
            self.gene_prob_factor_vector = None
        else:
            self.gene_factor_gene_mask = gene_or_pheno_mask            
            self.exp_gene_factors = exp_gene_or_pheno_factors
            self.gene_prob_factor_vector = gene_or_pheno_prob_vector
            self.pheno_prob_factor_vector = None

        self.gene_set_prob_factor_vector = gene_set_prob_vector
        self.gene_set_factor_gene_set_mask = gene_set_mask

        #now project the additional genes/phenos/gene sets onto the factors

        log("Projecting factors", TRACE)

        #this code gets the "relevance" values
        #first get the probabilities for either the genotypes or phenotypes (whichever we didn't use to factor)
        #these need to be specific to the anchors
        if factor_gene_set_x_pheno:
            if gene_or_pheno_full_prob_vector is not None:
                self.gene_prob_factor_vector = self._nnls_project_matrix(self.pheno_prob_factor_vector, gene_or_pheno_full_prob_vector.T)
                self._record_params({"factor_gene_prob_from": "phenos"})
            else:
                self.gene_prob_factor_vector = self._nnls_project_matrix(self.gene_set_prob_factor_vector, self.X_orig)
                self._record_params({"factor_gene_prob_from": "gene_sets"})
        else:
            if gene_or_pheno_full_prob_vector is not None:
                self.pheno_prob_factor_vector = self._nnls_project_matrix(self.gene_prob_factor_vector, gene_or_pheno_full_prob_vector.T)
                self._record_params({"factor_pheno_prob_from": "genes"})
            elif self.X_phewas_beta_uncorrected is not None:
                self.pheno_prob_factor_vector = self._nnls_project_matrix(self.gene_set_prob_factor_vector, self.X_phewas_beta_uncorrected)
                self._record_params({"factor_pheno_prob_from": "gene_sets"})

        if self.gene_set_prob_factor_vector is not None and sparse.issparse(self.gene_set_prob_factor_vector):
            self.gene_set_prob_factor_vector = self.gene_set_prob_factor_vector.toarray()
        if self.gene_prob_factor_vector is not None and sparse.issparse(self.gene_prob_factor_vector):
            self.gene_prob_factor_vector = self.gene_prob_factor_vector.toarray()
        if self.pheno_prob_factor_vector is not None and sparse.issparse(self.pheno_prob_factor_vector):
            self.pheno_prob_factor_vector = self.pheno_prob_factor_vector.toarray()

        gene_matrix_to_project = self.X_orig.T
        if not run_transpose:
            gene_matrix_to_project = gene_matrix_to_project.T

        #this code projects to the additional dimensions

        #all gene factor values
        full_gene_factor_values = self._project_H_with_fixed_W(self.exp_gene_set_factors, gene_matrix_to_project[self.gene_set_factor_gene_set_mask,:], self.gene_set_prob_factor_vector[self.gene_set_factor_gene_set_mask,:], self.gene_prob_factor_vector, phi=phi, tol=rel_tol, cap_genes=cap, normalize_genes=normalize_genes)
        if not factor_gene_set_x_pheno and keep_original_loadings:
            full_gene_factor_values[self.gene_factor_gene_mask,:] = self.exp_gene_factors

        #all pheno factor values, either from the phewas used to factor or the phewas passed in to project
        full_pheno_factor_values = self.exp_pheno_factors
        pheno_matrix_to_project = None

        if self.exp_gene_factors is None and self.exp_gene_set_factors is None:
            bail("Something went wrong: both gene factors and gene set factors are empty")

        if self.X_phewas_beta_uncorrected is not None and self.pheno_prob_factor_vector is not None:
            if project_phenos_from_gene_sets or self.exp_gene_factors is None:
                pheno_matrix_to_project = self.X_phewas_beta_uncorrected.T
                if not run_transpose:
                    pheno_matrix_to_project = pheno_matrix_to_project.T

                full_pheno_factor_values = self._project_H_with_fixed_W(self.exp_gene_set_factors, pheno_matrix_to_project if self.gene_set_factor_gene_set_mask is None else pheno_matrix_to_project[self.gene_set_factor_gene_set_mask,:], self.gene_set_prob_factor_vector if self.gene_set_factor_gene_set_mask is None else self.gene_set_prob_factor_vector[self.gene_set_factor_gene_set_mask,:], self.pheno_prob_factor_vector, phi=phi, tol=rel_tol, cap_genes=cap, normalize_genes=normalize_genes)
            else:
                pheno_matrix_to_project = self.gene_pheno_Y
                if not run_transpose:
                    pheno_matrix_to_project = pheno_matrix_to_project.T

                full_pheno_factor_values = self._project_H_with_fixed_W(self.exp_gene_factors, pheno_matrix_to_project if self.gene_factor_gene_mask is None else pheno_matrix_to_project[self.gene_factor_gene_mask,:], self.gene_prob_factor_vector if self.gene_factor_gene_mask is None else self.gene_prob_factor_vector[self.gene_factor_gene_mask,:], self.pheno_prob_factor_vector, phi=phi, tol=rel_tol, cap_genes=cap, normalize_genes=normalize_genes)

                
            if keep_original_loadings:
                full_pheno_factor_values[self.pheno_factor_pheno_mask,:] = self.exp_pheno_factors

        #now gene set factor values, projecting from either phenos or genes depending on what was used
        if factor_gene_set_x_pheno and pheno_matrix_to_project is not None:
            #we have to swap the gene sets and genes, which means transposing the matrix to project and swapping the prios
            full_gene_set_factor_values = self._project_H_with_fixed_W(self.exp_pheno_factors, pheno_matrix_to_project[:,self.pheno_factor_pheno_mask].T if run_transpose else pheno_matrix_to_project[self.pheno_factor_pheno_mask,:].T, self.pheno_prob_factor_vector[self.pheno_factor_pheno_mask,:], self.gene_set_prob_factor_vector, phi=phi, tol=rel_tol, cap_genes=cap, normalize_genes=normalize_gene_sets)
        else:
            full_gene_set_factor_values = self._project_H_with_fixed_W(self.exp_gene_factors, gene_matrix_to_project[:,self.gene_factor_gene_mask].T if run_transpose else gene_matrix_to_project[self.gene_factor_gene_mask,:].T, self.gene_prob_factor_vector[self.gene_factor_gene_mask,:], self.gene_set_prob_factor_vector, phi=phi, tol=rel_tol, cap_genes=cap, normalize_genes=normalize_gene_sets)

        if keep_original_loadings:
            full_gene_set_factor_values[self.gene_set_factor_gene_set_mask,:] = self.exp_gene_set_factors

        #update these to store the imputed as well
        self.exp_gene_factors = full_gene_factor_values
        self.exp_pheno_factors = full_pheno_factor_values
        self.exp_gene_set_factors = full_gene_set_factor_values

        if factor_gene_set_x_pheno:
            exp_gene_or_pheno_factors = self.exp_pheno_factors
        else:
            exp_gene_or_pheno_factors = self.exp_gene_factors

        #now update relevance

        matrix_to_mult = self.exp_pheno_factors if factor_gene_set_x_pheno else self.exp_gene_factors
        vector_to_mult = self.pheno_prob_factor_vector if factor_gene_set_x_pheno else self.gene_prob_factor_vector

        #matrix_to_mult: (genes, factors)
        #vector_to_mult: (users, genes)
        #want: (factors, users)

        self.factor_anchor_relevance = self._nnls_project_matrix(matrix_to_mult, vector_to_mult.T, max_value=1).T
        self.factor_relevance = self._nnls_project_matrix(matrix_to_mult, 1 - np.prod(1 - vector_to_mult, axis=1).T, max_value=1).T

        #gene scores are either for phenos or for genes depending on the mode
        reorder_inds = np.argsort(-self.factor_relevance)

        self.exp_lambdak = self.exp_lambdak[reorder_inds]
        self.factor_anchor_relevance = self.factor_anchor_relevance[reorder_inds,:]
        self.factor_relevance = self.factor_relevance[reorder_inds]
        if self.exp_gene_factors is not None:
            self.exp_gene_factors = self.exp_gene_factors[:,reorder_inds]
        if self.exp_pheno_factors is not None:
            self.exp_pheno_factors = self.exp_pheno_factors[:,reorder_inds]
        self.exp_gene_set_factors = self.exp_gene_set_factors[:,reorder_inds]

        #zero out very low values
        threshold = 1e-5
        if self.num_factors() > 0:
            self.exp_gene_factors[self.exp_gene_factors < np.max(self.exp_gene_factors) * threshold] = 0
            if self.exp_pheno_factors is not None:
                self.exp_pheno_factors[self.exp_pheno_factors < np.max(self.exp_pheno_factors) * threshold] = 0
            self.exp_gene_set_factors[self.exp_gene_set_factors < np.max(self.exp_gene_set_factors) * threshold] = 0

        num_top = 5

        #matries are gene x factor
        #materialize matrix of factor x gene x user, then take argmax over axis 1, then swap axes to get gene x factor x user
        
        #determine whether want highest, most specific, or combined
        exp_gene_factors_for_top = self.get_factor_loadings(self.exp_gene_factors, loading_type='combined')
        exp_pheno_factors_for_top = self.get_factor_loadings(self.exp_pheno_factors, loading_type='combined')
        exp_gene_set_factors_for_top = self.get_factor_loadings(self.exp_gene_set_factors, loading_type='combined')

        #(all_genes, factors)
        #(anchor_genes, users)

        top_anchor_gene_or_pheno_inds = None
        top_anchor_pheno_or_gene_inds = None

        if factor_gene_set_x_pheno:
            top_anchor_gene_or_pheno_inds = np.swapaxes(np.argsort(-(exp_pheno_factors_for_top).T[:,:,np.newaxis] * (self.pheno_prob_factor_vector)[np.newaxis,:,:], axis=1)[:,:num_top,:], 0, 1)
            if exp_gene_factors_for_top is not None:
                top_anchor_pheno_or_gene_inds = np.swapaxes(np.argsort(-(exp_gene_factors_for_top).T[:,:,np.newaxis] * (self.gene_prob_factor_vector)[np.newaxis,:,:], axis=1)[:,:num_top,:], 0, 1)
        else:
            top_anchor_gene_or_pheno_inds = np.swapaxes(np.argsort(-(exp_gene_factors_for_top).T[:,:,np.newaxis] * (self.gene_prob_factor_vector)[np.newaxis,:,:], axis=1)[:,:num_top,:], 0, 1)
            if exp_pheno_factors_for_top is not None:
                top_anchor_pheno_or_gene_inds = np.swapaxes(np.argsort(-(exp_pheno_factors_for_top).T[:,:,np.newaxis] * (self.pheno_prob_factor_vector)[np.newaxis,:,:], axis=1)[:,:num_top,:], 0, 1)

        #old one liner
        #top_anchor_gene_or_pheno_inds = np.swapaxes(np.argsort(-(exp_pheno_factors_for_top if factor_gene_set_x_pheno else exp_gene_factors_for_top).T[:,:,np.newaxis] * (self.pheno_prob_factor_vector if factor_gene_set_x_pheno else self.gene_prob_factor_vector)[np.newaxis,:,:], axis=1)[:,:num_top,:], 0, 1)

        top_anchor_gene_set_inds = np.swapaxes(np.argsort(-exp_gene_set_factors_for_top.T[:,:,np.newaxis] * self.gene_set_prob_factor_vector[np.newaxis,:,:], axis=1)[:,:num_top,:], 0, 1)

        #sort by maximum across phenos
        sort_max_across_phenos = True

        top_gene_or_pheno_inds = None
        top_pheno_or_gene_inds = None

        if factor_gene_set_x_pheno:
            top_gene_or_pheno_inds = np.swapaxes(np.argsort(-(1 - np.prod(1 - ((exp_pheno_factors_for_top).T[:,:,np.newaxis] * (self.pheno_prob_factor_vector)[np.newaxis,:,:]), axis=2)), axis=1)[:,:num_top], 0, 1)
            if exp_gene_factors_for_top is not None:
                top_pheno_or_gene_inds = np.swapaxes(np.argsort(-(1 - np.prod(1 - ((exp_gene_factors_for_top).T[:,:,np.newaxis] * (self.gene_prob_factor_vector)[np.newaxis,:,:]), axis=2)), axis=1)[:,:num_top], 0, 1)                
        else:
            top_gene_or_pheno_inds = np.swapaxes(np.argsort(-(1 - np.prod(1 - ((exp_gene_factors_for_top).T[:,:,np.newaxis] * (self.gene_prob_factor_vector)[np.newaxis,:,:]), axis=2)), axis=1)[:,:num_top], 0, 1)
            if exp_pheno_factors_for_top is not None:
                top_pheno_or_gene_inds = np.swapaxes(np.argsort(-(1 - np.prod(1 - ((exp_pheno_factors_for_top).T[:,:,np.newaxis] * (self.pheno_prob_factor_vector)[np.newaxis,:,:]), axis=2)), axis=1)[:,:num_top], 0, 1)                

        top_gene_set_inds = np.swapaxes(np.argsort(-(1 - np.prod(1 - (exp_gene_set_factors_for_top.T[:,:,np.newaxis] * self.gene_set_prob_factor_vector[np.newaxis,:,:]), axis=2)), axis=1)[:,:num_top], 0, 1)

        self.factor_labels = []
        self.factor_top_gene_sets = []
        top_genes_or_phenos = []
        top_phenos_or_genes = [] if top_pheno_or_gene_inds is not None else None

        self.factor_anchor_top_gene_sets = []
        anchor_top_genes_or_phenos = []
        anchor_top_phenos_or_genes = [] if top_anchor_pheno_or_gene_inds is not None else None

        factor_prompts = []
        for i in range(self.num_factors()):
            self.factor_top_gene_sets.append([self.gene_sets[j] for j in top_gene_set_inds[:,i]])

            self.factor_anchor_top_gene_sets.append([[self.gene_sets[j] for j in top_anchor_gene_set_inds[:,i,k]] for k in range(top_anchor_gene_set_inds.shape[2])])

            if factor_gene_set_x_pheno:
                genes_or_phenos = self.phenos
                phenos_or_genes = self.genes
            else:
                genes_or_phenos = self.genes
                phenos_or_genes = self.phenos

            top_genes_or_phenos.append([genes_or_phenos[j] for j in top_gene_or_pheno_inds[:,i] if not factor_gene_set_x_pheno or genes_or_phenos[j] != self.default_pheno])
            anchor_top_genes_or_phenos.append([[genes_or_phenos[j] for j in top_anchor_gene_or_pheno_inds[:,i,k] if not factor_gene_set_x_pheno or genes_or_phenos[j] != self.default_pheno] for k in range(top_anchor_gene_or_pheno_inds.shape[2])])

            if top_pheno_or_gene_inds is not None:
                top_phenos_or_genes.append([phenos_or_genes[j] for j in top_pheno_or_gene_inds[:,i] if factor_gene_set_x_pheno or phenos_or_genes[j] != self.default_pheno])
            if top_anchor_pheno_or_gene_inds is not None:
                anchor_top_phenos_or_genes.append([[phenos_or_genes[j] for j in top_anchor_pheno_or_gene_inds[:,i,k] if factor_gene_set_x_pheno or phenos_or_genes[j] != self.default_pheno]] for k in range(top_anchor_gene_or_pheno_inds.shape[2]) )
                

            self.factor_labels.append(self.factor_top_gene_sets[i][0] if len(self.factor_top_gene_sets[i]) > 0 else "")
            factor_prompts.append(",".join(self.factor_top_gene_sets[i]))

        if lmm_auth_key is not None and self.num_factors() > 0:
            labels = " ".join(["%d. %s" % (j+1, ",".join(self.factor_top_gene_sets[j] + (top_genes_or_phenos[j] if not label_gene_sets_only else []) + (top_phenos_or_genes[j] if label_include_phenos and top_phenos_or_genes is not None else []))) for j in range(self.num_factors())])

            def _set_factor_labels(factor_labels, num_factors, labels):
                if labels is None or factor_labels is None:
                    return

                if len(labels) == 0:
                    prompt = "Print a label, five words maximum and no quotes, for: %s." % (labels)
                else:
                    prompt = "Print a label, five words maximum, for each group. Print only labels, one per line, label number folowed by text: %s" % (labels)
                log("Querying LMM with prompt: %s" % prompt)
                response = query_lmm(prompt, lmm_auth_key, lmm_model=lmm_model, lmm_provider=lmm_provider)
                if response is not None:
                    try:
                        responses = response.strip('\n').split("\n")
                        responses = [x for x in responses if len(x) > 0]

                        if len(responses) == num_factors:
                            for i in range(num_factors):
                                cur_response = responses[i]
                                cur_response_tokens = cur_response.split()
                                if len(cur_response_tokens) > 1 and cur_response_tokens[0][-1] == ".":
                                    try:
                                        number = int(cur_response_tokens[0][:-1])
                                        cur_response = " ".join(cur_response_tokens[1:])
                                    except ValueError:
                                        pass
                                factor_labels[i] = cur_response
                        else:
                            raise Exception
                    except Exception:
                        log("Couldn't decode LMM response %s; using simple label" % response)
                        pass

            _set_factor_labels(self.factor_labels, self.num_factors(), labels)

            if label_individually:
                self.factor_labels_gene_sets = ["%d. %s" % (j+1, ",".join(self.factor_top_gene_sets[j])) for j in range(self.num_factors())]
                labels_gene_sets = " ".join(self.factor_labels_gene_sets)
                _set_factor_labels(self.factor_labels_gene_sets, self.num_factors(), labels_gene_sets)

                self.factor_labels_genes = ["%d. %s" % (j+1, ",".join(top_genes_or_phenos[j])) for j in range(self.num_factors())]
                labels_genes = " ".join(self.factor_labels_genes)
                _set_factor_labels(self.factor_labels_genes, self.num_factors(), labels_genes)

                self.factor_labels_phenos = ["%d. %s" % (j+1, ",".join(top_phenos_or_genes[j])) for j in range(self.num_factors())] if top_phenos_or_genes is not None else None
                if self.factor_labels_phenos is not None:
                    labels_phenos = " ".join(self.factor_labels_phenos) 
                    _set_factor_labels(self.factor_labels_phenos, self.num_factors(), labels_phenos)

        if factor_gene_set_x_pheno:
            self.factor_top_phenos = top_genes_or_phenos
            self.factor_top_genes = top_phenos_or_genes
            self.factor_anchor_top_phenos = anchor_top_genes_or_phenos
        else:
            self.factor_top_genes = top_genes_or_phenos
            self.factor_top_phenos = top_phenos_or_genes
            self.factor_anchor_top_genes = anchor_top_genes_or_phenos

        log("Found %d factors" % self.num_factors(), INFO)


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
                raise ValueError("Y must have shape (k, n) where k matches beta's rows and n matches X's rows.")
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

    def _build_phewas_input_values(self, run_for_factors=False, min_gene_factor_weight=0):
        if run_for_factors:
            input_values = self.exp_gene_factors
            factor_keep_mask = np.full(input_values.shape[0], True)
            if min_gene_factor_weight > 0:
                factor_keep_mask = np.any(self.exp_gene_factors > min_gene_factor_weight, axis=1)
            return input_values, factor_keep_mask

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
        input_values = np.exp(input_values + self.background_bf) / (1 + np.exp(input_values + self.background_bf))
        return input_values, None

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
            end = min((batch + 1) * cor_batch_size, beta_tildes.shape[0])

            if X_phewas_beta is not None and X_orig is not None and not options.debug_skip_correlation:
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
                        covs=covs if not options.debug_skip_phewas_covs else None,
                    )
                else:
                    (beta_tildes[begin:end, :], ses[begin:end, :], z_scores[begin:end, :], p_values[begin:end, :], se_inflation_factors[begin:end, :]) = self._compute_multivariate_beta_tildes(
                        X_mat,
                        Y_mat[begin:end, :],
                        resid_correlation_matrix=cor_matrices,
                        covs=covs if not options.debug_skip_phewas_covs else None,
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
        orig_ps = self.ps
        orig_sigma2s = self.sigma2s
        orig_p = self.p
        orig_sigma2_internal = self.sigma2
        orig_sigma_power = self.sigma_power
        self.ps = None
        self.sigma2s = None

        try:
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
        finally:
            self.ps = orig_ps
            self.sigma2s = orig_sigma2s
            self.p = orig_p
            self.sigma2 = orig_sigma2_internal
            self.sigma_power = orig_sigma_power

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
            get_col_fn=self._get_col,
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

    def _accumulate_standard_phewas_outputs(self, output_prefix, beta, beta_tilde, se, z_score, p_value):
        pegs_accumulate_standard_phewas_outputs(
            self,
            output_prefix,
            beta,
            beta_tilde,
            se,
            z_score,
            p_value,
        )

    def _accumulate_factor_phewas_outputs(self, output_prefix, beta_tilde, se, z_score, p_value, one_sided_p_value, huber=False):
        pegs_accumulate_factor_phewas_outputs(
            self,
            output_prefix,
            beta_tilde,
            se,
            z_score,
            p_value,
            one_sided_p_value,
            huber=huber,
        )

    def _run_factor_phewas_batch(self, input_values, factor_keep_mask, gene_pheno_Y, gene_pheno_combined_prior_Ys, begin, end, phewas_beta_kwargs):
        if gene_pheno_Y is not None:
            _, _, beta_tilde, se, z_score, p_value, one_sided_p_value = self._calculate_phewas_block(
                input_values[factor_keep_mask, :],
                gene_pheno_Y[factor_keep_mask, :].T,
                multivariate=True,
                covs=self.Y[factor_keep_mask],
                **phewas_beta_kwargs
            )
            self._accumulate_factor_phewas_outputs("Y", beta_tilde, se, z_score, p_value, one_sided_p_value)

            if not options.debug_skip_huber:
                _, _, beta_tilde, se, z_score, p_value, one_sided_p_value = self._calculate_phewas_block(
                    input_values[factor_keep_mask, :],
                    gene_pheno_Y[factor_keep_mask, :].T,
                    multivariate=True,
                    covs=self.Y[factor_keep_mask],
                    huber=True,
                    **phewas_beta_kwargs
                )
                self._accumulate_factor_phewas_outputs("Y", beta_tilde, se, z_score, p_value, one_sided_p_value, huber=True)

        if gene_pheno_combined_prior_Ys is not None and not options.debug_skip_correlation:
            _, _, beta_tilde, se, z_score, p_value, one_sided_p_value = self._calculate_phewas_block(
                input_values[factor_keep_mask, :],
                gene_pheno_combined_prior_Ys[factor_keep_mask, :].T,
                X_orig=self.X_orig[factor_keep_mask, :],
                X_phewas_beta=self.X_phewas_beta[begin:end, :] if self.X_phewas_beta is not None else None,
                Y_resid=gene_pheno_Y[factor_keep_mask, :].T,
                multivariate=True,
                covs=self.combined_prior_Ys[factor_keep_mask] if self.combined_prior_Ys is not None else self.Y[factor_keep_mask],
                **phewas_beta_kwargs
            )
            self._accumulate_factor_phewas_outputs("combined_prior_Ys", beta_tilde, se, z_score, p_value, one_sided_p_value)

            if not options.debug_skip_huber:
                _, _, beta_tilde, se, z_score, p_value, one_sided_p_value = self._calculate_phewas_block(
                    input_values[factor_keep_mask, :],
                    gene_pheno_combined_prior_Ys[factor_keep_mask, :].T,
                    X_orig=self.X_orig[factor_keep_mask, :],
                    X_phewas_beta=self.X_phewas_beta[begin:end, :] if self.X_phewas_beta is not None else None,
                    Y_resid=gene_pheno_Y[factor_keep_mask, :].T,
                    multivariate=True,
                    covs=self.combined_prior_Ys[factor_keep_mask] if self.combined_prior_Ys is not None else self.Y[factor_keep_mask],
                    huber=True,
                    **phewas_beta_kwargs
                )
                self._accumulate_factor_phewas_outputs("combined_prior_Ys", beta_tilde, se, z_score, p_value, one_sided_p_value, huber=True)

    def _run_standard_phewas_batch(self, input_values, gene_pheno_Y, gene_pheno_combined_prior_Ys, begin, end, phewas_beta_kwargs):
        if gene_pheno_Y is not None:
            beta, _, beta_tilde, se, z_score, p_value, _ = self._calculate_phewas_block(
                input_values,
                gene_pheno_Y.T,
                **phewas_beta_kwargs
            )
            assert beta.shape[0] == 3, "First dimension of beta should be 3, not (%s, %s)" % (beta.shape[0], beta.shape[1])
            self._accumulate_standard_phewas_outputs("pheno_Y", beta, beta_tilde, se, z_score, p_value)

        if gene_pheno_combined_prior_Ys is not None and not options.debug_skip_correlation:
            beta, _, beta_tilde, se, z_score, p_value, _ = self._calculate_phewas_block(
                input_values,
                gene_pheno_combined_prior_Ys.T,
                X_orig=self.X_orig,
                X_phewas_beta=self.X_phewas_beta[begin:end, :] if self.X_phewas_beta is not None else None,
                Y_resid=gene_pheno_Y.T,
                **phewas_beta_kwargs
            )
            assert beta.shape[0] == 3, "First dimension of beta should be 3, not (%s, %s)" % (beta.shape[0], beta.shape[1])
            self._accumulate_standard_phewas_outputs("pheno_combined_prior_Ys", beta, beta_tilde, se, z_score, p_value)

    def run_phewas(self, gene_phewas_bfs_in=None, gene_phewas_bfs_id_col=None, gene_phewas_bfs_pheno_col=None, gene_phewas_bfs_log_bf_col=None, gene_phewas_bfs_combined_col=None, gene_phewas_bfs_prior_col=None, run_for_factors=False, max_num_burn_in=1000, max_num_iter=1100, min_num_iter=10, num_chains=10, r_threshold_burn_in=1.01, use_max_r_for_convergence=True, max_frac_sem=0.01, gauss_seidel=False, sparse_solution=False, sparse_frac_betas=None, batch_size=1500, min_gene_factor_weight=0, **kwargs):

        #require X matrix
        if gene_phewas_bfs_in is None and not _has_loaded_gene_phewas(self):
            bail("Require --gene-stats-in or --gene-phewas-bfs-in with a column for log_bf/Y in this operation")

        if run_for_factors:
            if self.exp_gene_set_factors is None:
                warn("Cannot run factor phewas without gene factors; skipping")
                return

            log("Running factor phewas", INFO)
        else:
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
        input_values, factor_keep_mask = self._build_phewas_input_values(
            run_for_factors=run_for_factors,
            min_gene_factor_weight=min_gene_factor_weight,
        )
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

            if run_for_factors:
                self._run_factor_phewas_batch(
                    input_values=input_values,
                    factor_keep_mask=factor_keep_mask,
                    gene_pheno_Y=gene_pheno_Y,
                    gene_pheno_combined_prior_Ys=gene_pheno_combined_prior_Ys,
                    begin=begin,
                    end=end,
                    phewas_beta_kwargs=phewas_beta_kwargs,
                )
            else:
                self._run_standard_phewas_batch(
                    input_values=input_values,
                    gene_pheno_Y=gene_pheno_Y,
                    gene_pheno_combined_prior_Ys=gene_pheno_combined_prior_Ys,
                    begin=begin,
                    end=end,
                    phewas_beta_kwargs=phewas_beta_kwargs,
                )

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
            debug_only_avg_huge=getattr(options, "debug_only_avg_huge", False),
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

    def write_factor_phewas_statistics(self, output_file):
        return pegs_write_factor_phewas_statistics(
            self,
            output_file,
            open_text_fn=open_gz,
            log_fn=log,
            info_level=INFO,
        )

    def write_matrix_factors(self, factors_output_file=None, write_anchor_specific=False):

        if self.num_factors() <= 0:
            return

        anchors = None

        if write_anchor_specific:
            anchor_mask = self.anchor_pheno_mask if self.anchor_pheno_mask is not None else self.anchor_gene_mask

            if anchor_mask is None:
                anchors = ["default"]
            else:
                anchors = self.phenos if self.anchor_pheno_mask is not None else self.genes
                anchors = [anchors[x] for x in np.where(anchor_mask)[0]]

        ordered_inds = range(self.num_factors())

        if factors_output_file is not None:
            log("Writing factors to %s" % factors_output_file, INFO)
            with open_gz(factors_output_file, 'w') as output_fh:
                header = "Factor"
                header = "%s\t%s" % (header, "label")
                if anchors is not None:
                    header = "%s\t%s" % (header, "anchor")
                    header = "%s\t%s" % (header, "relevance")
                else:
                    header = "%s\t%s" % (header, "lambda")
                    header = "%s\t%s" % (header, "any_relevance")

                if self.factor_top_genes is not None or self.factor_anchor_top_genes is not None:
                    header = "%s\t%s" % (header, "top_genes")
                if anchors is None and self.factor_labels_genes is not None:
                    header = "%s\t%s" % (header, "label_genes")
                header = "%s\t%s" % (header, "top_gene_sets")
                if anchors is None and self.factor_labels_gene_sets is not None:
                    header = "%s\t%s" % (header, "label_gene_sets")

                if self.factor_top_phenos is not None or self.factor_anchor_top_phenos is not None:
                    header = "%s\t%s" % (header, "top_phenos")
                    if anchors is None and self.factor_labels_phenos is not None:
                        header = "%s\t%s" % (header, "label_phenos")

                output_fh.write("%s\n" % (header))
                    
                num_users = len(anchors) if anchors is not None else 1
                for j in range(num_users):
                    for i in ordered_inds:
                        line = "Factor%d" % (i+1)
                        line = "%s\t%s" % (line, self.factor_labels[i])
                        if anchors is not None:
                            line = "%s\t%s" % (line, anchors[j])
                            line = "%s\t%.3g" % (line, self.factor_anchor_relevance[i,j])
                            if self.factor_anchor_top_genes is not None:
                                line = "%s\t%s" % (line, ",".join(self.factor_anchor_top_genes[i][j]))
                            line = "%s\t%s" % (line, ",".join(self.factor_anchor_top_gene_sets[i][j]))
                            if self.factor_anchor_top_phenos is not None:
                                line = "%s\t%s" % (line, ",".join(self.factor_anchor_top_phenos[i][j]))
                        else:
                            line = "%s\t%.3g" % (line, self.exp_lambdak[i])
                            line = "%s\t%.3g" % (line, self.factor_relevance[i])
                            if self.factor_top_genes is not None:
                                line = "%s\t%s" % (line, ",".join(self.factor_top_genes[i]))
                                if self.factor_labels_genes:
                                    line = "%s\t%s" % (line, self.factor_labels_genes[i])                                   
                            line = "%s\t%s" % (line, ",".join(self.factor_top_gene_sets[i]))
                            if self.factor_labels_gene_sets:
                                line = "%s\t%s" % (line, self.factor_labels_gene_sets[i])
                            if self.factor_top_phenos is not None:
                                line = "%s\t%s" % (line, ",".join(self.factor_top_phenos[i]))
                                if self.factor_labels_phenos:
                                    line = "%s\t%s" % (line, self.factor_labels_phenos[i])


                        output_fh.write("%s\n" % (line))

    def write_clusters(self, gene_set_clusters_output_file=None, gene_clusters_output_file=None, pheno_clusters_output_file=None, write_anchor_specific=False, anchor_genes=None):

        if self.num_factors() == 0:
            log("No factors; not writing clusters")
            return

        anchors = None
        anchor_inds = None
        pheno_anchors = False
        gene_anchors = False

        if write_anchor_specific:
            anchor_mask = self.anchor_pheno_mask if self.anchor_pheno_mask is not None else self.anchor_gene_mask
            if anchor_mask is None:
                anchors = ["default"]
            else:
                if self.anchor_pheno_mask is not None:
                    anchors = self.phenos
                    pheno_anchors = True
                else:
                    anchors = self.genes
                    gene_anchors = True

                anchor_inds = np.where(anchor_mask)[0]
                anchors = [anchors[x] for x in anchor_inds]

        ordered_inds = range(self.num_factors())
        num_users = len(anchors) if anchors is not None else 1

        if gene_set_clusters_output_file is not None and self.exp_gene_set_factors is not None:
            
            #this uses value relative to others in the cluster
            #values_for_cluster = self.exp_gene_set_factors / np.sum(self.exp_gene_set_factors, axis=0)
            #this uses strongest absolute value
            values_for_cluster = self.exp_gene_set_factors

            log("Writing gene set clusters to %s" % gene_set_clusters_output_file, INFO)
            with open_gz(gene_set_clusters_output_file, 'w') as output_fh:

                gene_set_factor_gene_set_inds = list(range(self.exp_gene_set_factors.shape[0]))
                header = "Gene_Set"
                master_key_fn = None

                any_prob = None
                if anchors is None:
                    if self.betas is None and self.betas_uncorrected is None:
                        any_prob = 1 - np.prod(1 - self.gene_set_prob_factor_vector, axis=1)
                        header = "%s\t%s" % (header, "any_relevance")
                        master_key_fn = lambda k: -any_prob[k]

                if self.betas is not None or (pheno_anchors and self.X_phewas_beta is not None):
                    header = "%s\t%s" % (header, "beta")
                    if self.betas is not None:
                        master_key_fn = lambda k: -self.betas[gene_set_factor_gene_set_inds[k]]
                if self.betas_uncorrected is not None or (pheno_anchors and self.X_phewas_beta_uncorrected is not None):
                    header = "%s\t%s" % (header, "beta_uncorrected")
                    if self.betas_uncorrected is not None and master_key_fn is None:
                        master_key_fn = lambda k: -self.betas_uncorrected_[gene_set_factor_gene_set_inds[k]]

                if anchors is not None:
                    header = "%s\t%s" % (header, "relevance")

                header = "%s\t%s" % (header, "used_to_factor")

                if anchors is not None:
                    header = "%s\t%s" % (header, "anchor")

                output_fh.write("%s\t%s\t%s\t%s\t%s\t%s\n" % (header, "cluster", "label", "\t".join(["Factor%d" % (i+1) for i in ordered_inds]), "\t".join(["Relative_Factor%d" % (i+1) for i in ordered_inds]), "\t".join(["Combined_Factor%d" % (i+1) for i in ordered_inds])))

                if master_key_fn is None:
                    master_key_fn = lambda k: k

                for j in range(num_users):
                    if anchors is not None:
                        key_fn = lambda k: (-self.gene_set_prob_factor_vector[gene_set_factor_gene_set_inds[k],j], master_key_fn(k))
                    else:
                        key_fn = master_key_fn


                    raw_gene_set_factor_loadings = self.get_factor_loadings(self.exp_gene_set_factors, loading_type='raw')
                    specific_gene_set_factor_loadings = self.get_factor_loadings(self.exp_gene_set_factors, loading_type='specific')
                    combined_gene_set_factor_loadings = self.get_factor_loadings(self.exp_gene_set_factors, loading_type='combined')

                    for i in sorted(range(values_for_cluster.shape[0]), key=key_fn):

                        #THINK WE CAN REMOVE THIS; DOES IT PASS ASSERT?
                        orig_i = gene_set_factor_gene_set_inds[i]
                        assert(orig_i == i)

                        line = self.gene_sets[orig_i]

                        if anchors is None:
                            if any_prob is not None:
                                line = "%s\t%.3g" % (line, any_prob[orig_i])
                            if self.betas is not None:
                                line = "%s\t%.3g" % (line, self.betas[orig_i])
                            if self.betas_uncorrected is not None:
                                line = "%s\t%.3g" % (line, self.betas_uncorrected[orig_i])

                        else:
                            if self.X_phewas_beta is not None and pheno_anchors:
                                line = "%s\t%.3g" % (line, self.X_phewas_beta[anchor_inds[j],orig_i])
                            elif self.betas is not None:
                                line = "%s\t%.3g" % (line, self.betas[orig_i])
                                
                            if self.X_phewas_beta_uncorrected is not None and pheno_anchors:
                                line = "%s\t%.3g" % (line, self.X_phewas_beta_uncorrected[anchor_inds[j],orig_i])
                            elif self.betas_uncorrected is not None:
                                line = "%s\t%.3g" % (line, self.betas_uncorrected[orig_i])

                            
                            line = "%s\t%.3g" % (line, self.gene_set_prob_factor_vector[i,j])

                        used_to_factor = self.gene_set_factor_gene_set_mask[i] if self.gene_set_factor_gene_set_mask is not None else False
                        line = "%s\t%s" % (line, used_to_factor)

                        multiplier = 1
                        if anchors is not None:
                            line = "%s\t%s" % (line, anchors[j])
                            multiplier = self.gene_set_prob_factor_vector[orig_i,j]

                        cluster = np.argmax(values_for_cluster[i,:] * multiplier)

                        output_fh.write("%s\tFactor%d\t%s\t%s\t%s\t%s\n" % (line, cluster + 1, self.factor_labels[cluster], "\t".join(["%.4g" % (multiplier * raw_gene_set_factor_loadings[i,k]) for k in ordered_inds]), "\t".join(["%.4g" % (multiplier * specific_gene_set_factor_loadings[i,k]) for k in ordered_inds]), "\t".join(["%.4g" % (multiplier * combined_gene_set_factor_loadings[i,k]) for k in ordered_inds])))

        if gene_clusters_output_file is not None and self.exp_gene_factors is not None:

            #this uses strongest absolute value
            values_for_cluster = self.exp_gene_factors

            log("Writing gene clusters to %s" % (gene_clusters_output_file), INFO)
            with open_gz(gene_clusters_output_file, 'w') as output_fh:
                gene_factor_gene_inds = list(range(self.exp_gene_factors.shape[0]))
                header = "Gene"
                master_key_fn = None

                any_prob = None
                if anchors is None:
                    if self.combined_prior_Ys is None and self.Y is None and self.priors is None:
                        any_prob = 1 - np.prod(1 - self.gene_prob_factor_vector, axis=1)
                        header = "%s\t%s" % (header, "any_relevance")
                        master_key_fn = lambda k: -any_prob[k]

                if self.combined_prior_Ys is not None or (pheno_anchors and self.gene_pheno_combined_prior_Ys is not None):
                    header = "%s\t%s" % (header, "combined")
                    if self.combined_prior_Ys is not None:
                        master_key_fn = lambda k: -self.combined_prior_Ys[gene_factor_gene_inds[k]]
                if self.Y is not None or (pheno_anchors and self.gene_pheno_Y is not None):
                    header = "%s\t%s" % (header, "log_bf")
                    if self.Y is not None and master_key_fn is None:
                        master_key_fn = lambda k: -self.Y[gene_factor_gene_inds[k]]
                if self.priors is not None or (pheno_anchors and self.gene_pheno_priors is not None):
                    header = "%s\t%s" % (header, "prior")
                    if self.priors is not None and master_key_fn is None:
                        master_key_fn = lambda k: -self.priors[gene_factor_gene_inds[k]]

                if anchors is not None:
                    header = "%s\t%s" % (header, "relevance")                    
                    header = "%s\t%s" % (header, "anchor")

                header = "%s\t%s" % (header, "used_to_factor")
                if gene_anchors:
                    header = "%s\t%s" % (header, "is_anchor")                    

                output_fh.write("%s\t%s\t%s\t%s\t%s\t%s\n" % (header, "cluster", "label", "\t".join(["Factor%d" % (i+1) for i in ordered_inds]), "\t".join(["Relative_Factor%d" % (i+1) for i in ordered_inds]), "\t".join(["Combined_Factor%d" % (i+1) for i in ordered_inds])))

                if master_key_fn is None:
                    master_key_fn = lambda k: k
                
                for j in range(num_users):
                    if anchors is not None:
                        key_fn = lambda k: (-self.gene_prob_factor_vector[gene_factor_gene_inds[k],j], master_key_fn(k))
                    else:
                        key_fn = master_key_fn

                    raw_gene_factor_loadings = self.get_factor_loadings(self.exp_gene_factors, loading_type='raw')
                    specific_gene_factor_loadings = self.get_factor_loadings(self.exp_gene_factors, loading_type='specific')
                    combined_gene_factor_loadings = self.get_factor_loadings(self.exp_gene_factors, loading_type='combined')

                    for i in sorted(range(values_for_cluster.shape[0]), key=key_fn):

                        orig_i = gene_factor_gene_inds[i]
                        assert(orig_i == i)

                        line = self.genes[orig_i]

                        if anchors is None and any_prob is not None:
                            line = "%s\t%.3g" % (line, any_prob[orig_i])

                        if self.combined_prior_Ys is not None or (pheno_anchors and self.gene_pheno_combined_prior_Ys is not None):
                            line = "%s\t%.3g" % (line, self.gene_pheno_combined_prior_Ys[orig_i,anchor_inds[j]] if pheno_anchors and self.gene_pheno_combined_prior_Ys is not None else self.combined_prior_Ys[orig_i])
                        if self.Y is not None or (pheno_anchors and self.gene_pheno_Y is not None):
                            line = "%s\t%.3g" % (line, self.gene_pheno_Y[orig_i,anchor_inds[j]] if pheno_anchors and self.gene_pheno_Y is not None else self.Y[orig_i])
                        if self.priors is not None or (pheno_anchors and self.gene_pheno_priors is not None):
                            line = "%s\t%.3g" % (line, self.gene_pheno_priors[orig_i,anchor_inds[j]] if pheno_anchors and self.gene_pheno_priors is not None else self.priors[orig_i])

                        multiplier = 1
                        if anchors is not None:
                            line = "%s\t%.3g" % (line, self.gene_prob_factor_vector[i,j])
                            line = "%s\t%s" % (line, anchors[j])
                            multiplier = self.gene_prob_factor_vector[orig_i,j]

                        used_to_factor = self.gene_factor_gene_mask[i] if self.gene_factor_gene_mask is not None else False
                        line = "%s\t%s" % (line, used_to_factor)

                        if gene_anchors:
                            line = "%s\t%s" % (line, anchor_mask[i])
 
                        cluster = np.argmax(values_for_cluster[i,:] * multiplier)
                        output_fh.write("%s\tFactor%d\t%s\t%s\t%s\t%s\n" % (line, cluster + 1, self.factor_labels[cluster], "\t".join(["%.4g" % (multiplier * raw_gene_factor_loadings[i,k]) for k in ordered_inds]), "\t".join(["%.4g" % (multiplier * specific_gene_factor_loadings[i,k]) for k in ordered_inds]), "\t".join(["%.4g" % (multiplier * combined_gene_factor_loadings[i,k]) for k in ordered_inds])))

        if pheno_clusters_output_file is not None and self.exp_pheno_factors is not None:

            #this uses value relative to others in the cluster
            #this uses strongest absolute value
            values_for_cluster = self.exp_pheno_factors

            pheno_combined_prior_Ys = self.pheno_Y_vs_input_combined_prior_Ys_beta if self.pheno_Y_vs_input_combined_prior_Ys_beta is not None else self.pheno_combined_prior_Ys_vs_input_combined_prior_Ys_beta
            pheno_Y = self.pheno_Y_vs_input_Y_beta if self.pheno_Y_vs_input_Y_beta is not None else self.pheno_combined_prior_Ys_vs_input_Y_beta
            pheno_priors = self.pheno_Y_vs_input_priors_beta if self.pheno_Y_vs_input_priors_beta is not None else self.pheno_combined_prior_Ys_vs_input_priors_beta            

            log("Writing pheno clusters to %s" % (pheno_clusters_output_file), INFO)
            with open_gz(pheno_clusters_output_file, 'w') as output_fh:
                pheno_factor_pheno_inds = list(range(self.exp_pheno_factors.shape[0]))
                header = "Pheno"
                master_key_fn = None

                any_prob = None

                if gene_anchors:
                    if self.gene_pheno_combined_prior_Ys is not None:
                        header = "%s\t%s" % (header, "combined")
                    if self.gene_pheno_Y is not None:
                        header = "%s\t%s" % (header, "log_bf")
                    if self.gene_pheno_priors is not None:
                        header = "%s\t%s" % (header, "prior")
                else:
                    if anchors is None and pheno_combined_prior_Ys is None and pheno_Y is None and pheno_priors is None:
                        any_prob = 1 - np.prod(1 - self.pheno_prob_factor_vector, axis=1)
                        header = "%s\t%s" % (header, "any_relevance")
                        master_key_fn = lambda k: -any_prob[k]

                    if pheno_combined_prior_Ys is not None:
                        header = "%s\t%s" % (header, "combined")
                        master_key_fn = lambda k: -pheno_combined_prior_Ys[pheno_factor_pheno_inds[k]]
                    if pheno_Y is not None:
                        header = "%s\t%s" % (header, "log_bf")
                        if master_key_fn is None:
                            master_key_fn = lambda k: -pheno_Y[pheno_factor_pheno_inds[k]]
                    if pheno_priors is not None:
                        header = "%s\t%s" % (header, "prior")
                        if master_key_fn is None:
                            master_key_fn = lambda k: -pheno_priors[pheno_factor_pheno_inds[k]]

                if anchors is not None:
                    header = "%s\t%s" % (header, "relevance")                    
                    header = "%s\t%s" % (header, "anchor")

                header = "%s\t%s" % (header, "used_to_factor")

                if pheno_anchors:
                    header = "%s\t%s" % (header, "is_anchor")

                output_fh.write("%s\t%s\t%s\t%s\t%s\t%s\n" % (header, "cluster", "label", "\t".join(["Factor%d" % (i+1) for i in ordered_inds]), "\t".join(["Relative_Factor%d" % (i+1) for i in ordered_inds]), "\t".join(["Combined_Factor%d" % (i+1) for i in ordered_inds])))

                if master_key_fn is None:
                    master_key_fn = lambda k: k
                for j in range(num_users):
                    if anchors is not None:
                        key_fn = lambda k: (-self.pheno_prob_factor_vector[pheno_factor_pheno_inds[k],j], master_key_fn(k))
                    else:
                        key_fn = master_key_fn
                    
                    raw_pheno_factor_loadings = self.get_factor_loadings(self.exp_pheno_factors, loading_type='raw')
                    specific_pheno_factor_loadings = self.get_factor_loadings(self.exp_pheno_factors, loading_type='specific')
                    combined_pheno_factor_loadings = self.get_factor_loadings(self.exp_pheno_factors, loading_type='combined')

                    for i in sorted(range(values_for_cluster.shape[0]), key=key_fn):
                        #if np.sum(self.exp_pheno_factors[i,:]) == 0:
                        #    continue

                        orig_i = pheno_factor_pheno_inds[i]
                        assert(orig_i == i)

                        line = self.phenos[orig_i]

                        if not gene_anchors and anchors is None and any_prob is not None:
                            line = "%s\t%.3g" % (line, any_prob[orig_i])

                        if pheno_combined_prior_Ys is not None or (gene_anchors and self.gene_pheno_combined_prior_Ys is not None):
                            line = "%s\t%.3g" % (line, self.gene_pheno_combined_prior_Ys[anchor_inds[j], orig_i] if gene_anchors and self.gene_pheno_combined_prior_Ys is not None else pheno_combined_prior_Ys[orig_i])
                        if pheno_Y is not None or (gene_anchors and self.gene_pheno_Y is not None):
                            line = "%s\t%.3g" % (line, self.gene_pheno_Y[anchor_inds[j], orig_i] if gene_anchors and self.gene_pheno_Y is not None else pheno_Y[orig_i])
                        if pheno_priors is not None or (gene_anchors and self.gene_pheno_priors is not None):
                            line = "%s\t%.3g" % (line, self.gene_pheno_priors[anchor_inds[j], orig_i] if gene_anchors and self.gene_pheno_priors is not None else pheno_priors[orig_i])

                        multiplier = 1
                        if anchors is not None:
                            #relevance
                            line = "%s\t%.3g" % (line, self.pheno_prob_factor_vector[i,j])
                            line = "%s\t%s" % (line, anchors[j])
                            multiplier = self.pheno_prob_factor_vector[orig_i,j]

                        used_to_factor = self.pheno_factor_pheno_mask[i] if self.pheno_factor_pheno_mask is not None else False
                        line = "%s\t%s" % (line, used_to_factor)
                        if pheno_anchors:
                            line = "%s\t%s" % (line, anchor_mask[i])

                        cluster = np.argmax(values_for_cluster[i,:] * multiplier)
                        output_fh.write("%s\tFactor%d\t%s\t%s\t%s\t%s\n" % (line, cluster + 1, self.factor_labels[cluster], "\t".join(["%.4g" % (multiplier * raw_pheno_factor_loadings[i,k]) for k in ordered_inds]), "\t".join(["%.4g" % (multiplier * specific_pheno_factor_loadings[i,k]) for k in ordered_inds]), "\t".join(["%.4g" % (multiplier * combined_pheno_factor_loadings[i,k]) for k in ordered_inds])))                    


    def write_gene_pheno_statistics(self, output_file=None, min_value_to_print=0):
        if self.gene_pheno_Y is None and self.gene_pheno_combined_prior_Ys is None and self.gene_pheno_priors is None:
            return

        if self.genes is None or self.phenos is None:
            return

        log("Writing gene pheno statistics to %s" % output_file)

        with open_gz(output_file, 'w') as output_fh:

            header = "Gene\tPheno"

            if self.gene_pheno_priors is not None:
                header = "%s\t%s" % (header, "prior")
            if self.gene_pheno_combined_prior_Ys is not None:
                header = "%s\t%s" % (header, "combined")
            if self.gene_pheno_Y is not None:
                header = "%s\t%s" % (header, "log_bf")

            output_fh.write("%s\n" % header)

            ordered_i = range(len(self.genes))

            use_for_ordering = None

            if self.gene_pheno_combined_prior_Ys is not None:
                use_for_ordering = self.gene_pheno_combined_prior_Ys
            elif self.gene_pheno_priors is not None:
                use_for_ordering = self.gene_pheno_priors
            elif self.gene_pheno_Y is not None:
                use_for_ordering = self.gene_pheno_Y

            use_for_ordering_genes = use_for_ordering.max(axis=1).toarray().squeeze()

            ordered_i = sorted(ordered_i, key=lambda k: -np.max(use_for_ordering[k]))

            for i in ordered_i:
                gene = self.genes[i]
                ordered_j = range(len(self.phenos))
                ordered_j = sorted(ordered_j, key=lambda k: -use_for_ordering[i,k])
                for j in ordered_j:
                    pheno = self.phenos[j]
                    line = "%s\t%s" % (gene, pheno)
                    print_line = False
                    if self.gene_pheno_priors is not None:
                        line = "%s\t%.3g" % (line, self.gene_pheno_priors[i,j])
                        if self.gene_pheno_priors[i,j] > min_value_to_print:
                            print_line = True
                    if self.gene_pheno_combined_prior_Ys is not None:
                        line = "%s\t%.3g" % (line, self.gene_pheno_combined_prior_Ys[i,j])
                        if self.gene_pheno_combined_prior_Ys[i,j] > min_value_to_print:
                            print_line = True
                    if self.gene_pheno_Y is not None:
                        line = "%s\t%.3g" % (line, self.gene_pheno_Y[i,j])
                        if self.gene_pheno_Y[i,j] > min_value_to_print:
                            print_line = True
                    if print_line:
                        output_fh.write("%s\n" % line)


    #HELPER FUNCTIONS

    '''
    Read in gene bfs for LOGISTIC or EMPIRICAL mapping
    '''
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

    def _read_gene_bfs(self, gene_bfs_in, gene_bfs_id_col=None, gene_bfs_log_bf_col=None, gene_bfs_combined_col=None, gene_bfs_prob_col=None, gene_bfs_prior_col=None, gene_bfs_sd_col=None, **kwargs):

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
            get_col_fn=self._get_col,
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

    def _read_gene_covs(self, gene_covs_in, gene_covs_id_col=None, gene_covs_cov_cols=None, **kwargs):

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
            get_col_fn=self._get_col,
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
        if self.y_corr_cholesky is not None:
            bail("Cannot read/sort correlations after initializing full GLS correlation state (y_corr_cholesky). Re-run correlation setup before enabling full GLS, or disable full GLS for this step.")
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

    def _compute_robust_betas(self, X, Y, resid_correlation_matrix=None, covs=None, add_intercept=True, delta=1.0, max_iter=100, tol=1e-6, rel_tol=0.01):

        log("Calculating robust beta tildes", DEBUG)

        Y = Y.T
        if len(Y.shape) == 1:
            Y = Y[:,np.newaxis]

        n_phenos = Y.shape[1]

        #x is gene x factor
        n_factors = X.shape[1]  # Number of factors (columns in X)

        if add_intercept:
            X = np.hstack((X, np.ones((X.shape[0],1))))

        if covs is not None:
            if len(covs.shape) == 1:
                covs = covs[:,np.newaxis]
            X = np.hstack((X, covs))

        def _huber_loss(residuals, delta):
            return np.where(np.abs(residuals) <= delta, 0.5 * residuals ** 2, delta * (np.abs(residuals) - 0.5 * delta))

        def _huber_weight(residuals, delta):
            residuals[residuals == 0] = delta
            return np.where((np.abs(residuals) > 0) & (np.abs(residuals) <= delta), 1, delta / np.abs(residuals))

        # Initial coefficients
        W = np.linalg.lstsq(X, Y, rcond=None)[0]

        #pheno x gene x factor
        X_x_pheno = np.repeat(X[np.newaxis,:,:], Y.shape[1], axis=0)

        # Iteratively Reweighted Least Squares
        for iteration in range(max_iter):

            Y_pred = np.dot(X, W)
            residuals = Y - Y_pred
            weights = _huber_weight(residuals, delta)

            #unvectorized code for reference
            #W_new = np.zeros_like(W)
            #for i in range(Y.shape[1]):
            #    W_i = weights[:, i]
            #    XTWX = np.dot(X.T, np.multiply(X.T, W_i).T)
            #    XTWY = np.dot(X.T, np.multiply(W_i, Y[:, i]))
            #    W_new[:, i] = np.linalg.solve(XTWX, XTWY)


            #W is factor x phenos
            #weights is gene x phenos
            #Y is gene x phenos
            #X is gene x factor
            #X_x_pheno is pheno x gene x factor
            #X_x_pheno.T is factor x gene x pheno
            #weights are gene x factor

            #per pheno

            #pheno x gene x factor
            X_x_pheno_w = np.multiply(X_x_pheno.T, weights).T
            #X is factor x gene

            #pheno x factor x factor
            XTwX = np.einsum('pgf,gh->pfh', X_x_pheno_w, X)

            #gene x pheno 
            wY = np.multiply(weights, Y)
            #X_x_pheno is pheno x gene x factor

            XTwY = np.einsum('pgf,gp->fp', X_x_pheno, wY)

            #W_new = np.linalg.solve(XTwX, XTwY)

            #pheno x factor x factor
            XTwX_inv = np.linalg.inv(XTwX)
            W_new = np.einsum("phf,fp->hp", XTwX_inv, XTwY)

            if np.linalg.norm(W_new - W, ord='fro') < tol:
                break

            if np.max(np.abs(W_new - W) / (np.abs(W_new) + np.abs(W) + 1e-20)) < rel_tol:
                break


            W = W_new

        Y_pred = np.dot(X, W)
        residuals = Y - Y_pred
        betas = W.T

        # Calculate the variance of the residuals
        n = X.shape[0]
        p = X.shape[1]
        sse = np.sum(_huber_loss(residuals, delta), axis=0)
        #length equal to phenos
        sigma2 = sse / (n - p)


        # We'll also need (X^T X)^{-1} for a quasi-variance approach
        XtX = X.T @ X
        XtX_inv = np.linalg.inv(XtX)

        # 2) "Base" robust standard errors ignoring correlation
        diag_inv = np.diag(XtX_inv)  # (n_pred,)
        base_ses = np.sqrt(sigma2[:, None] * diag_inv[None, :])  # (phenos, n_pred)

        if resid_correlation_matrix is None:
            ses = base_ses
        else:
            if len(resid_correlation_matrix) != n_phenos:
                raise ValueError("resid_correlation_matrix must match number of phenotypes.")

            ses = np.zeros_like(base_ses)  # shape (phenos, n_pred)

            # We'll reuse weights for the final sandwich step, which is shape (genes, phenos)
            for p in range(n_phenos):
                R_p = resid_correlation_matrix[p]  # (genes, genes)
                # robust weights for phenotype p => weights[:, p] => shape (genes,)
                w_vec = np.sqrt(weights[:, p])

                # WeightedX => multiply each row i by w_vec[i]
                WeightedX = X * w_vec[:, None]  # shape (genes, n_pred)

                # Then multiply by R_p => shape => (genes, n_pred)
                if sparse.issparse(R_p):
                    WeightedX_R = R_p.dot(WeightedX)
                else:
                    WeightedX_R = R_p @ WeightedX

                # WeightedX^T * WeightedX_R => (n_pred, n_pred)
                XtRprimeX = WeightedX.T @ WeightedX_R

                var_betas_p = XtX_inv @ XtRprimeX @ XtX_inv
                se_p = np.sqrt(np.diag(var_betas_p))
                ses[p, :] = se_p

        if covs is not None or add_intercept:
            betas = betas[:, :n_factors]  # Only the factor betas
            ses = ses[:, :n_factors]  # Corresponding standard errors

        return self._finalize_regression(betas, ses, se_inflation_factors=None)



    def _finalize_regression(self, beta_tildes, ses, se_inflation_factors):
        return pegs_finalize_regression_outputs(
            beta_tildes,
            ses,
            se_inflation_factors,
            log_fn=log,
            warn_fn=warn,
            trace_level=TRACE,
        )


    def _calculate_non_inf_betas(self, initial_p, return_sample=False, max_num_burn_in=None, max_num_iter=1100, min_num_iter=10, num_chains=10, r_threshold_burn_in=1.01, use_max_r_for_convergence=True, eps=0.01, max_frac_sem=0.01, max_allowed_batch_correlation=None, beta_outlier_iqr_threshold=5, gauss_seidel=False, update_hyper_sigma=True, update_hyper_p=True, adjust_hyper_sigma_p=False, only_update_hyper=False, sigma_num_devs_to_top=2.0, p_noninf_inflate=1.0, num_p_pseudo=1, sparse_solution=False, sparse_frac_betas=None, betas_trace_out=None, betas_trace_gene_sets=None, beta_tildes=None, ses=None, V=None, X_orig=None, scale_factors=None, mean_shifts=None, is_dense_gene_set=None, ps=None, sigma2s=None, assume_independent=False, num_missing_gene_sets=None, debug_genes=None, debug_gene_sets=None, init_betas=None, init_postp=None):

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

        if len(scale_factors.shape) == 2 and not scale_factors.shape[0] == num_parallel:
            bail("scale_factors must have same number of parallel runs as beta_tildes")
        elif len(scale_factors.shape) == 1 and num_parallel == 1:
            scale_factors_m = scale_factors[np.newaxis,:]
            mean_shifts_m = mean_shifts[np.newaxis,:]
        elif len(scale_factors.shape) == 1 and num_parallel > 1:
            scale_factors_m = np.tile(scale_factors, num_parallel).reshape((num_parallel, len(scale_factors)))
            mean_shifts_m = np.tile(mean_shifts, num_parallel).reshape((num_parallel, len(mean_shifts)))
        else:
            scale_factors_m = copy.copy(scale_factors)
            mean_shifts_m = copy.copy(mean_shifts)

        if len(is_dense_gene_set.shape) == 2 and not is_dense_gene_set.shape[0] == num_parallel:
            bail("is_dense_gene_set must have same number of parallel runs as beta_tildes")
        elif len(is_dense_gene_set.shape) == 1 and num_parallel == 1:
            is_dense_gene_set_m = is_dense_gene_set[np.newaxis,:]
        elif len(is_dense_gene_set.shape) == 1 and num_parallel > 1:
            is_dense_gene_set_m = np.tile(is_dense_gene_set, num_parallel).reshape((num_parallel, len(is_dense_gene_set)))
        else:
            is_dense_gene_set_m = copy.copy(is_dense_gene_set)

        if ps is not None:
            if len(ps.shape) == 2 and not ps.shape[0] == num_parallel:
                bail("ps must have same number of parallel runs as beta_tildes")
            elif len(ps.shape) == 1 and num_parallel == 1:
                ps_m = ps[np.newaxis,:]
            elif len(ps.shape) == 1 and num_parallel > 1:
                ps_m = np.tile(ps, num_parallel).reshape((num_parallel, len(ps)))
            else:
                ps_m = copy.copy(ps)
        else:
            ps_m = self.p

        if sigma2s is not None:
            if len(sigma2s.shape) == 2 and not sigma2s.shape[0] == num_parallel:
                bail("sigma2s must have same number of parallel runs as beta_tildes")
            elif len(sigma2s.shape) == 1 and num_parallel == 1:
                orig_sigma2_m = sigma2s[np.newaxis,:]
            elif len(sigma2s.shape) == 1 and num_parallel > 1:
                orig_sigma2_m = np.tile(sigma2s, num_parallel).reshape((num_parallel, len(sigma2s)))
            else:
                orig_sigma2_m = copy.copy(sigma2s)
        else:
            orig_sigma2_m = self.sigma2

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


        if betas_trace_out is not None:
            betas_trace_fh = open_gz(betas_trace_out, 'w')
            betas_trace_fh.write("It\tParallel\tChain\tGene_Set\tbeta_post\tbeta\tpostp\tres_beta_hat\tbeta_tilde\tbeta_internal\tres_beta_hat_internal\tbeta_tilde_internal\tse_internal\tsigma2\tp\tR\tR_weighted\tSEM\n")

        prev_betas_m = None
        sigma_underflow = False
        printed_warning_swing = False
        printed_warning_increase = False
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

            for gene_set_mask_ind in range(len(gene_set_masks)):

                #the challenge here is that gene_set_mask_m produces a ragged (non-square) tensor
                #so we are going to "flatten" the last two dimensions
                #this requires some care, in particular when running einsum, which requires a square tensor

                gene_set_mask_m = gene_set_masks[gene_set_mask_ind]
                
                if debug_gene_sets is not None:
                    cur_debug_gene_sets = [debug_gene_sets[i] for i in range(len(debug_gene_sets)) if gene_set_mask_m[0,i]]

                #intersect compute_max_v with the rows of gene_set_mask (which are the parallel runs)
                compute_mask_m = np.logical_and(compute_mask_v, gene_set_mask_m.T).T

                current_num_parallel = sum(compute_mask_v)

                #Value to use when determining if we should force an alpha shrink if estimates are way off compared to heritability estimates.  (Improves MCMC convergence.)
                #zero_jump_prob=0.05
                #frac_betas_explained = max(0.00001,np.sum(np.apply_along_axis(np.mean, 0, np.power(curr_betas_m,2)))) / self.y_var
                #frac_sigma_explained = self.sigma2_total_var / self.y_var
                #alpha_shrink = min(1 - zero_jump_prob, 1.0 / frac_betas_explained, (frac_sigma_explained + np.mean(np.power(ses[i], 2))) / frac_betas_explained)
                alpha_shrink = 1

                #subtract out the predicted effects of the other betas
                #we need to zero out diagonal of V to do this, but rather than do this we will add it back in

                #1. First take the union of the current_gene_set_mask
                #this is to allow us to run einsum
                #we are going to do it across more gene sets than are needed, and then throw away the computations that are extra for each batch
                compute_mask_union = np.any(compute_mask_m, axis=0)

                #2. Retain how to filter from the union down to each mask
                compute_mask_union_filter_m = compute_mask_m[:,compute_mask_union]

                if assume_independent:
                    res_beta_hat_t_flat = beta_tildes_m[compute_mask_m]
                else:
                    if multiple_V:

                        #3. Do einsum across the union
                        #This does pointwise matrix multiplication of curr_betas_t (sliced on axis 1) with V (sliced on axis 0), maintaining axis 0 for curr_betas_t
                        res_beta_hat_union_t = np.einsum('hij,ijk->hik', curr_betas_t[:,compute_mask_v,:], V[compute_mask_v,:,:][:,:,compute_mask_union]).reshape((num_chains, current_num_parallel, np.sum(compute_mask_union)))

                    elif sparse_V:
                        res_beta_hat_union_t = V[compute_mask_union,:].dot(curr_betas_t[:,compute_mask_v,:].T.reshape((curr_betas_t.shape[2], np.sum(compute_mask_v) * curr_betas_t.shape[0]))).reshape((np.sum(compute_mask_union), np.sum(compute_mask_v), curr_betas_t.shape[0])).T
                    elif use_X:
                        if len(compute_mask_union.shape) == 2:
                            assert(compute_mask_union.shape[0] == 1)
                            compute_mask_union = np.squeeze(compute_mask_union)
                        #curr_betas_t: (num_chains, num_parallel, num_gene_sets)
                        #X_orig: (num_genes, num_gene_sets)
                        #X_orig_t: (num_gene_sets, num_genes)
                        #mean_shifts_m: (num_parallel, num_gene_sets)
                        #curr_betas_filtered_t: (num_chains, num_compute, num_gene_sets)

                        curr_betas_filtered_t = curr_betas_t[:,compute_mask_v,:] / scale_factors_m[compute_mask_v,:]

                        #have to reshape latter two dimensions before multiplying because sparse matrix can only handle 2-D

                        #interm = np.zeros((X_orig.shape[0],np.sum(compute_mask_v),curr_betas_t.shape[0]))
                        #interm[:,compute_mask_v,:] = X_orig.dot(curr_betas_filtered_t.T.reshape((curr_betas_filtered_t.shape[2],curr_betas_filtered_t.shape[0] * curr_betas_filtered_t.shape[1]))).reshape((X_orig.shape[0],curr_betas_filtered_t.shape[1],curr_betas_filtered_t.shape[0])) - np.sum(mean_shifts_m[compute_mask_v,:] * curr_betas_filtered_t, axis=2).T

                        interm = X_orig.dot(curr_betas_filtered_t.T.reshape((curr_betas_filtered_t.shape[2],curr_betas_filtered_t.shape[0] * curr_betas_filtered_t.shape[1]))).reshape((X_orig.shape[0],curr_betas_filtered_t.shape[1],curr_betas_filtered_t.shape[0])) - np.sum(mean_shifts_m[compute_mask_v,:] * curr_betas_filtered_t, axis=2).T

                        #interm: (num_genes, num_parallel remaining, num_chains)

                        #num_gene sets, num_parallel, num_chains

                        #this broke under some circumstances when a parallel chain converged before the others
                        res_beta_hat_union_t = (X_orig[:,compute_mask_union].T.dot(interm.reshape((interm.shape[0],interm.shape[1]*interm.shape[2]))).reshape((np.sum(compute_mask_union),interm.shape[1],interm.shape[2])) - mean_shifts_m.T[compute_mask_union,:][:,compute_mask_v,np.newaxis] * np.sum(interm, axis=0)).T
                        res_beta_hat_union_t /= (X_orig.shape[0] * scale_factors_m[compute_mask_v,:][:,compute_mask_union])

                        #res_beta_hat_union_t = (X_orig[:,compute_mask_union].T.dot(interm.reshape((interm.shape[0],interm.shape[1]*interm.shape[2]))).reshape((np.sum(compute_mask_union),interm.shape[1],interm.shape[2])) - mean_shifts_m.T[compute_mask_union,:][:,:,np.newaxis] * np.sum(interm, axis=0)).T
                        #res_beta_hat_union_t /= (X_orig.shape[0] * scale_factors_m[:,compute_mask_union])

                    else:
                        res_beta_hat_union_t = curr_betas_t[:,compute_mask_v,:].dot(V[:,compute_mask_union])

                    if betas_trace_out is not None and betas_trace_gene_sets is not None:
                        all_map = pegs_construct_map_to_ind(betas_trace_gene_sets)
                        cur_sets = [betas_trace_gene_sets[x] for x in range(len(betas_trace_gene_sets)) if compute_mask_union[x]]
                        cur_map = pegs_construct_map_to_ind(cur_sets)

                    #4. Now restrict to only the actual masks (which flattens things because the compute_mask_m is not square)

                    res_beta_hat_t_flat = res_beta_hat_union_t[:,compute_mask_union_filter_m[compute_mask_v,:]]
                    assert(res_beta_hat_t_flat.shape[1] == np.sum(compute_mask_m))

                    #dimensions of res_beta_hat_t_flat are (num_chains, np.sum(compute_mask_m))
                    #dimensions of beta_tildes_m are (num_parallel, num_gene_sets))
                    #subtraction will subtract matrix from each of the matrices in the tensor

                    res_beta_hat_t_flat = beta_tildes_m[compute_mask_m] - res_beta_hat_t_flat

                    if account_for_V_diag_m:
                        #dimensions of V_diag_m are (num_parallel, num_gene_sets)
                        #curr_betas_t is (num_chains, num_parallel, num_gene_sets)
                        res_beta_hat_t_flat = res_beta_hat_t_flat + V_diag_m[compute_mask_m] * curr_betas_t[:,compute_mask_m]
                    else:
                        res_beta_hat_t_flat = res_beta_hat_t_flat + curr_betas_t[:,compute_mask_m]
                
                b2_t_flat = np.power(res_beta_hat_t_flat, 2)
                d_const_b2_exp_t_flat = d_const_m[compute_mask_m] * np.exp(-b2_t_flat / (se2s_m[compute_mask_m] * 2.0))
                numerator_t_flat = c_const_m[compute_mask_m] * np.exp(-b2_t_flat / (2.0 * hdmpn_m[compute_mask_m]))
                numerator_zero_mask_t_flat = (numerator_t_flat == 0)
                denominator_t_flat = numerator_t_flat + d_const_b2_exp_t_flat
                denominator_t_flat[numerator_zero_mask_t_flat] = 1


                d_imaginary_mask_t_flat = ~np.isreal(d_const_b2_exp_t_flat)
                numerator_imaginary_mask_t_flat = ~np.isreal(numerator_t_flat)

                if np.any(np.logical_or(d_imaginary_mask_t_flat, numerator_imaginary_mask_t_flat)):

                    warn("Detected imaginary numbers!")
                    #if d is imaginary, we set it to 1
                    denominator_t_flat[d_imaginary_mask_t_flat] = numerator_t_flat[d_imaginary_mask_t_flat]
                    #if d is real and numerator is imaginary, we set to 0 (both numerator and denominator will be imaginary)
                    numerator_t_flat[np.logical_and(~d_imaginary_mask_t_flat, numerator_imaginary_mask_t_flat)] = 0

                    #Original code for handling edge cases; adapted above
                    #Commenting these out for now, but they are here in case we ever detect non real numbers
                    #if need them, masked_array is too inefficient -- change to real mask
                    #d_real_mask_t = np.isreal(d_const_b2_exp_t)
                    #numerator_real_mask_t = np.isreal(numerator_t)
                    #curr_postp_t = np.ma.masked_array(curr_postp_t, np.logical_not(d_real_mask_t), fill_value = 1).filled()
                    #curr_postp_t = np.ma.masked_array(curr_postp_t, np.logical_and(d_real_mask_t, np.logical_not(numerator_real_mask_t)), fill_value=0).filled()
                    #curr_postp_t = np.ma.masked_array(curr_postp_t, np.logical_and(np.logical_and(d_real_mask_t, numerator_real_mask_t), numerator_zero_mask_t), fill_value=0).filled()



                curr_postp_t[:,compute_mask_m] = (numerator_t_flat / denominator_t_flat)


                #calculate current posterior means
                #the left hand side, because it is masked, flattens the latter two dimensions into one
                #so we flatten the result of the right hand size to a 1-D array to match up for the assignment
                curr_post_means_t[:,compute_mask_m] = hdmp_hdmpn_m[compute_mask_m] * (curr_postp_t[:,compute_mask_m] * res_beta_hat_t_flat)

                   
                if gauss_seidel:
                    proposed_beta_t_flat = curr_post_means_t[:,compute_mask_m]
                else:
                    norm_mean_t_flat = hdmp_hdmpn_m[compute_mask_m] * res_beta_hat_t_flat

                    #draw from the conditional distribution
                    proposed_beta_t_flat = norm_mean_t_flat + norm_scale_m[compute_mask_m] * rand_norms_t[:,compute_mask_m]

                    #set things to zero that sampled below p
                    zero_mask_t_flat = rand_ps_t[:,compute_mask_m] >= curr_postp_t[:,compute_mask_m] * alpha_shrink
                    proposed_beta_t_flat[zero_mask_t_flat] = 0

                #update betas
                #do this inside loop since this determines the res_beta
                #same idea as above for collapsing
                curr_betas_t[:,compute_mask_m] = proposed_beta_t_flat
                res_beta_hat_t[:,compute_mask_m] = res_beta_hat_t_flat

                #if debug_gene_sets is not None:
                #    my_cur_tensor_shape = (1 if assume_independent else num_chains, current_num_parallel, np.sum(gene_set_mask_m[0,]))
                #    my_cur_tensor_shape2 = (num_chains, current_num_parallel, np.sum(gene_set_mask_m[0,]))
                #    my_res_beta_hat_t = res_beta_hat_t_flat.reshape(my_cur_tensor_shape)
                #    my_proposed_beta_t = proposed_beta_t_flat.reshape(my_cur_tensor_shape2)
                #    my_norm_mean_t = norm_mean_t_flat.reshape(my_cur_tensor_shape)
                #    top_set = [cur_debug_gene_sets[i] for i in range(len(cur_debug_gene_sets)) if np.abs(my_res_beta_hat_t[0,0,i]) == np.max(np.abs(my_res_beta_hat_t[0,0,:]))][0]
                #    log("TOP IS",top_set)
                #    gs = set([ "mp_absent_T_cells", top_set])
                #    ind = [i for i in range(len(cur_debug_gene_sets)) if cur_debug_gene_sets[i] in gs]
                #    for i in ind:
                #        log("BETA_TILDE",cur_debug_gene_sets[i],beta_tildes_m[0,i]/scale_factors_m[0,i])
                #        log("Z",cur_debug_gene_sets[i],beta_tildes_m[0,i]/ses_m[0,i])
                #        log("RES",cur_debug_gene_sets[i],my_res_beta_hat_t[0,0,i]/scale_factors_m[0,i])
                #        #log("RESF",cur_debug_gene_sets[i],res_beta_hat_t_flat[i]/scale_factors_m[0,i])
                #        log("NORM_MEAN",cur_debug_gene_sets[i],my_norm_mean_t[0,0,i])
                #        log("NORM_SCALE_M",cur_debug_gene_sets[i],norm_scale_m[0,i])
                #        log("RAND_NORMS",cur_debug_gene_sets[i],rand_norms_t[0,0,i])
                #        log("PROP",cur_debug_gene_sets[i],my_proposed_beta_t[0,0,i]/scale_factors_m[0,i])
                #        ind2 = [j for j in range(len(debug_gene_sets)) if debug_gene_sets[j] == cur_debug_gene_sets[i]]
                #        for j in ind2:
                #            log("POST",cur_debug_gene_sets[i],curr_post_means_t[0,0,j]/scale_factors_m[0,i])
                #            log("SIGMA",sigma2_m if type(sigma2_m) is float or type(sigma2_m) is np.float64 else sigma2_m[0,i])
                #            log("P",cur_debug_gene_sets[i],curr_postp_t[0,0,j],self.p)
                #            log("HDMP",hdmp_m/np.square(scale_factors_m[0,i]) if type(hdmp_m) is float or type(hdmp_m) is np.float64 else hdmp_m[0,0]/np.square(scale_factors_m[0,i]))
                #            log("SES",se2s_m[0,0]/np.square(scale_factors_m[0,i]))
                #            log("HDMPN",hdmpn_m/np.square(scale_factors_m[0,i]) if type(hdmpn_m) is float or type(hdmpn_m) is np.float64 else hdmpn_m[0,0]/scale_factors_m[0,i])
                #            log("HDMP_HDMPN",hdmp_hdmpn_m if type(hdmp_hdmpn_m) is float or type(hdmp_hdmpn_m) is np.float64 else hdmp_hdmpn_m[0,0])
                #            log("NOW1",debug_gene_sets[j],curr_betas_t[0,0,j]/scale_factors_m[0,i])


            if sparse_solution:
                sparse_mask_t = curr_postp_t < ps_m

                if sparse_frac_betas is not None:
                    #zero out very small values relative to top or median
                    relative_value = np.max(np.abs(curr_post_means_t), axis=2)
                    sparse_mask_t = np.logical_or(sparse_mask_t, (np.abs(curr_post_means_t).T < sparse_frac_betas * relative_value.T).T)

                #don't set anything not currently computed
                sparse_mask_t[:,np.logical_not(compute_mask_v),:] = False
                log("Setting %d entries to zero due to sparsity" % (np.sum(np.logical_and(sparse_mask_t, curr_betas_t > 0))), TRACE)
                curr_betas_t[sparse_mask_t] = 0
                curr_post_means_t[sparse_mask_t] = 0

                if debug_gene_sets is not None:
                    ind = [i for i in range(len(debug_gene_sets)) if debug_gene_sets[i] in gs]

            curr_betas_m = np.mean(curr_post_means_t, axis=0)
            curr_postp_m = np.mean(curr_postp_t, axis=0)
            #no state should be preserved across runs, but take a random one just in case
            sample_betas_m = curr_betas_t[int(random.random() * curr_betas_t.shape[0]),:,:]
            sample_postp_m = curr_postp_t[int(random.random() * curr_postp_t.shape[0]),:,:]
            sum_betas_t[:,compute_mask_v,:] = sum_betas_t[:,compute_mask_v,:] + curr_post_means_t[:,compute_mask_v,:]
            sum_betas2_t[:,compute_mask_v,:] = sum_betas2_t[:,compute_mask_v,:] + np.square(curr_post_means_t[:,compute_mask_v,:])

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
                def __calculate_R_tensor(sum_t, sum2_t, num):

                    #mean of betas across all iterations; psi_dot_j
                    mean_t = sum_t / float(num)

                    #mean of betas across replicates; psi_dot_dot
                    mean_m = np.mean(mean_t, axis=0)
                    #variances of betas across all iterators; s_j
                    var_t = (sum2_t - float(num) * np.power(mean_t, 2)) / (float(num) - 1)
                    #B_v = (float(iteration_num) / (num_chains - 1)) * np.apply_along_axis(np.sum, 0, np.apply_along_axis(lambda x: np.power(x - mean_betas_v, 2), 1, mean_betas_m))
                    B_m = (float(num) / (mean_t.shape[0] - 1)) * np.sum(np.power(mean_t - mean_m, 2), axis=0)
                    W_m = (1.0 / float(mean_t.shape[0])) * np.sum(var_t, axis=0)
                    avg_W_m = (1.0 / float(mean_t.shape[2])) * np.sum(var_t, axis=2)
                    var_given_y_m = np.add((float(num) - 1) / float(num) * W_m, (1.0 / float(num)) * B_m)
                    var_given_y_m[var_given_y_m < 0] = 0

                    R_m = np.ones(W_m.shape)
                    R_non_zero_mask_m = W_m > 0

                    var_given_y_m[var_given_y_m < 0] = 0

                    R_m[R_non_zero_mask_m] = np.sqrt(var_given_y_m[R_non_zero_mask_m] / W_m[R_non_zero_mask_m])
                    
                    return (B_m, W_m, R_m, avg_W_m, mean_t)

                #these matrices have convergence statistics in format (num_parallel, num_gene_sets)
                #WARNING: only the results for compute_mask_v are valid
                (B_m, W_m, R_m, avg_W_m, mean_t) = __calculate_R_tensor(sum_betas_t, sum_betas2_t, iteration_num)

                beta_weights_m = np.zeros((sum_betas_t.shape[1], sum_betas_t.shape[2]))
                sum_betas_t_mean = np.mean(sum_betas_t)
                if sum_betas_t_mean > 0:
                    np.mean(sum_betas_t, axis=0) / sum_betas_t_mean

                #calculate the thresholded / scaled R_v
                num_R_above_1_v = np.sum(R_m >= 1, axis=1)
                num_R_above_1_v[num_R_above_1_v == 0] = 1

                #mean for each parallel run

                R_m_above_1 = copy.copy(R_m)
                R_m_above_1[R_m_above_1 < 1] = 0
                mean_thresholded_R_v = np.sum(R_m_above_1, axis=1) / num_R_above_1_v

                #max for each parallel run
                max_index_v = np.argmax(R_m, axis=1)
                max_index_parallel = None
                max_val = None
                for i in range(len(max_index_v)):
                    if compute_mask_v[i] and (max_val is None or R_m[i,max_index_v[i]] > max_val):
                        max_val = R_m[i,max_index_v[i]]
                        max_index_parallel = i
                max_R_v = np.max(R_m, axis=1)
               

                #TEMP TEMP TEMP
                #if priors_for_convergence:
                #    curr_v = curr_betas_v
                #    s_cur2_v = np.array([curr_v[i] for i in sorted(range(len(curr_v)), key=lambda k: -np.abs(curr_v[k]))])
                #    s_cur2_v = np.square(s_cur2_v - np.mean(s_cur2_v))
                #    cum_cur2_v = np.cumsum(s_cur2_v) / np.sum(s_cur2_v)
                #    top_mask2 = np.array(cum_cur2_v < 0.99)
                #    (B_v2, W_v2, R_v2) = __calculate_R(sum_betas_m[:,top_mask2], sum_betas2_m[:,top_mask2], iteration_num)
                #    max_index2 = np.argmax(R_v2)
                #    log("Iteration %d (betas): max ind=%d; max B=%.3g; max W=%.3g; max R=%.4g; avg R=%.4g; num above=%.4g" % (iteration_num, max_index2, B_v2[max_index2], W_v2[max_index2], R_v2[max_index2], np.mean(R_v2), np.sum(R_v2 > r_threshold_burn_in)), TRACE)
                #END TEMP TEMP TEMP
                    
                if use_max_r_for_convergence:
                    convergence_statistic_v = max_R_v
                else:
                    convergence_statistic_v = mean_thresholded_R_v

                outlier_mask_m = np.full(avg_W_m.shape, False)
                if avg_W_m.shape[0] > 10:
                    #check the variances
                    q3, median, q1 = np.percentile(avg_W_m, [75, 50, 25], axis=0)
                    iqr_mask = q3 > q1
                    chain_iqr_m = np.zeros(avg_W_m.shape)
                    chain_iqr_m[:,iqr_mask] = (avg_W_m[:,iqr_mask] - median[iqr_mask]) / (q3 - q1)[iqr_mask]
                    #dimensions chain x parallel
                    outlier_mask_m = beta_outlier_iqr_threshold
                    if np.sum(outlier_mask_m) > 0:
                        log("Detected %d outlier chains due to oscillations" % np.sum(outlier_mask_m), DEBUG)

                if np.sum(R_m > 1) > 10:
                    #check the Rs
                    q3, median, q1 = np.percentile(R_m[R_m > 1], [75, 50, 25])
                    if q3 > q1:
                        #Z score per parallel, gene
                        R_iqr_m = (R_m - median) / (q3 - q1)
                        #dimensions of parallel x gene sets
                        bad_gene_sets_m = np.logical_and(R_iqr_m > 100, R_m > 2.5)
                        bad_gene_sets_v = np.any(bad_gene_sets_m,0)
                        if np.sum(bad_gene_sets_m) > 0:
                            #now find the bad chains
                            bad_chains = np.argmax(np.abs(mean_t - np.mean(mean_t, axis=0)), axis=0)[bad_gene_sets_m]

                            #np.where bad gene sets[0] lists parallel
                            #bad chains lists the bad chain corresponding to each parallel
                            cur_outlier_mask_m = np.zeros(outlier_mask_m.shape)
                            cur_outlier_mask_m[bad_chains, np.where(bad_gene_sets_m)[0]] = True

                            log("Found %d outlier chains across %d parallel runs due to %d gene sets with high R (%.4g - %.4g; %.4g - %.4g)" % (np.sum(cur_outlier_mask_m), np.sum(np.any(cur_outlier_mask_m, axis=0)), np.sum(bad_gene_sets_m), np.min(R_m[bad_gene_sets_m]), np.max(R_m[bad_gene_sets_m]), np.min(R_iqr_m[bad_gene_sets_m]), np.max(R_iqr_m[bad_gene_sets_m])), DEBUG)
                            outlier_mask_m = np.logical_or(outlier_mask_m, cur_outlier_mask_m)

                            #log("Outlier parallel: %s" % (np.where(bad_gene_sets_m)[0]), DEBUG)
                            #log("Outlier values: %s" % (R_m[bad_gene_sets_m]), DEBUG)
                            #log("Outlier IQR: %s" % (R_iqr_m[bad_gene_sets_m]), DEBUG)
                            #log("Outlier chains: %s" % (bad_chains), DEBUG)


                            #log("Actually in mask: %s" % (str(np.where(outlier_mask_m))))

                non_outliers_m = ~outlier_mask_m
                if np.sum(outlier_mask_m) > 0:
                    log("Detected %d total outlier chains" % np.sum(outlier_mask_m), DEBUG)
                    #dimensions are num_chains x num_parallel
                    for outlier_parallel in np.where(np.any(outlier_mask_m, axis=0))[0]:
                        #find a non-outlier chain and replace the three matrices in the right place
                        if np.sum(outlier_mask_m[:,outlier_parallel]) > 0:
                            if np.sum(non_outliers_m[:,outlier_parallel]) > 0:
                                replacement_chains = np.random.choice(np.where(non_outliers_m[:,outlier_parallel])[0], size=np.sum(outlier_mask_m[:,outlier_parallel]))
                                log("Replaced chains %s with chains %s in parallel %d" % (np.where(outlier_mask_m[:,outlier_parallel])[0], replacement_chains, outlier_parallel), DEBUG)

                                for tensor in [curr_betas_t, curr_postp_t, curr_post_means_t, sum_betas_t, sum_betas2_t]:
                                    tensor[outlier_mask_m[:,outlier_parallel],outlier_parallel,:] = copy.copy(tensor[replacement_chains,outlier_parallel,:])

                            else:
                                log("Every chain was an outlier so doing nothing", TRACE)


                log("Iteration %d: max ind=%s; max B=%.3g; max W=%.3g; max R=%.4g; avg R=%.4g; num above=%.4g;" % (iteration_num, (max_index_parallel, max_index_v[max_index_parallel]) if num_parallel > 1 else max_index_v[max_index_parallel], B_m[max_index_parallel, max_index_v[max_index_parallel]], W_m[max_index_parallel, max_index_v[max_index_parallel]], R_m[max_index_parallel, max_index_v[max_index_parallel]], np.mean(mean_thresholded_R_v), np.sum(R_m > r_threshold_burn_in)), TRACE)

                converged_v = convergence_statistic_v < r_threshold_burn_in
                newly_converged_v = np.logical_and(burn_in_phase_v, converged_v)
                if np.sum(newly_converged_v) > 0:
                    if num_parallel == 1:
                        log("Converged after %d iterations" % iteration_num, INFO)
                    else:
                        log("Parallel %s converged after %d iterations" % (",".join([str(p) for p in np.nditer(np.where(newly_converged_v))]), iteration_num), INFO)
                    burn_in_phase_v = np.logical_and(burn_in_phase_v, np.logical_not(converged_v))

            if np.sum(burn_in_phase_v) == 0 or iteration_num >= max_num_burn_in:

                #if we only care about parameters, we can return immediately (burn in stops hyper updates)
                if only_update_hyper:
                    return (None, None)

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

                    return (sample_betas_m, sample_postp_m, curr_betas_m, curr_postp_m)

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
                if update_hyper_p or update_hyper_sigma:
                    # Hyper-updates use Rao-Blackwellized moments to avoid sigma collapse.
                    # conditional slab mean m = hdmp_hdmpn * res_beta_hat
                    cond_mean_t = hdmp_hdmpn_m[np.newaxis, :, :] * res_beta_hat_t
                    # conditional slab variance v = hdmp_hdmpn * se2
                    cond_var_m = hdmp_hdmpn_m * se2s_m
                    # E[beta^2] = postp * (m^2 + v)
                    e_beta2_m = np.mean(curr_postp_t * (np.square(cond_mean_t) + cond_var_m[np.newaxis, :, :]), axis=0)
                    # mu = E[beta]
                    mu_m = curr_betas_m
                    # Var(beta) = E[beta^2] - (E[beta])^2
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

                    if self.sigma_power is not None:
                        new_sigma2 = h2 / np.mean(np.sum(np.power(scale_factors_m, self.sigma_power), axis=1))
                    else:
                        new_sigma2 = h2 / num_gene_sets

                    if num_missing_gene_sets:
                        missing_scale_factor = num_gene_sets / (num_gene_sets + num_missing_gene_sets)
                        new_sigma2 *= missing_scale_factor
                        new_p *= missing_scale_factor

                    if p_noninf_inflate != 1:
                        log("Inflating p by %.3g" % p_noninf_inflate, DEBUG)
                        new_p *= p_noninf_inflate

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
                            return (None, None)

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

                                    #if np.sum([~is_dense_gene_set_m]) > 0:
                                    #    new_sigma2 = new_sigma2 / np.power(np.mean(scale_factors_m[~is_dense_gene_set_m].ravel()), self.sigma_power)
                                    #else:
                                    #    new_sigma2 = new_sigma2 / np.power(np.mean(scale_factors_m.ravel()), self.sigma_power)

                                    #if is_dense_gene_set_m.ravel()[max_e_beta2]:
                                    #    new_sigma2 = new_sigma2 / np.power(np.mean(scale_factors_m[~is_dense_gene_set_m].ravel()), self.sigma_power)
                                    #else:
                                    #    new_sigma2 = new_sigma2 / np.power(scale_factors_m.ravel()[max_e_beta2], self.sigma_power)

                                if not update_hyper_p and adjust_hyper_sigma_p:
                                    #remember, sigma is the *total* variance term. It is equal to p * conditional_sigma.
                                    #if we are only sigma p, and adjusting p, we will leave the conditional_sigma constant, which means scaling the p
                                    new_p = self.p / self.sigma2 * new_sigma2
                                    log("Updating p from %.4g to %.4g to maintain constant sigma/p" % (self.p, new_p), TRACE)
                                    #we need to adjust the total sigma to keep the conditional sigma constant
                                    self.set_p(new_p)

                                self.set_sigma(new_sigma2, self.sigma_power)
                                sigma_underflow = True

                                #update_hyper_sigma = False
                                #restarting sampling with sigma2 fixed to initial value due to underflow
                                #update_hyper_p = False

                                #reset loop state
                                #iteration_num = 0
                                #curr_post_means_t = np.zeros(tensor_shape)
                                #curr_postp_t = np.ones(tensor_shape)
                                #curr_betas_t = scipy.stats.norm.rvs(0, np.std(beta_tildes_m), tensor_shape)                            
                                #avg_betas_m = np.zeros(matrix_shape)
                                #avg_betas2_m = np.zeros(matrix_shape)
                                #avg_postp_m = np.zeros(matrix_shape)
                                #num_avg = 0
                                #sum_betas_t = np.zeros(tensor_shape)
                                #sum_betas2_t = np.zeros(tensor_shape)
                            else:
                                self.set_sigma(new_sigma2, self.sigma_power)

                            #update the matrix forms of these variables
                            orig_sigma2_m *= new_sigma2 / np.mean(orig_sigma2_m)
                            if self.sigma_power is not None:
                                #sigma2_m = orig_sigma2_m * np.power(scale_factors_m, self.sigma_power)
                                sigma2_m = self.get_scaled_sigma2(scale_factors_m, orig_sigma2_m, self.sigma_power, self.sigma_threshold_k, self.sigma_threshold_xo)

                                #for dense gene sets, scaling by size doesn't make sense. So use mean size across sparse gene sets
                                if np.sum(is_dense_gene_set_m) > 0:
                                    if np.sum(~is_dense_gene_set_m) > 0:
                                        #sigma2_m[is_dense_gene_set_m] = self.sigma2 * np.power(np.mean(scale_factors_m[~is_dense_gene_set_m]), self.sigma_power)
                                        sigma2_m[is_dense_gene_set_m] = self.get_scaled_sigma2(np.mean(scale_factors_m[~is_dense_gene_set_m]), orig_sigma2_m, self.sigma_power, self.sigma_threshold_k, self.sigma_threshold_xo)
                                    else:
                                        #sigma2_m[is_dense_gene_set_m] = self.sigma2 * np.power(np.mean(scale_factors_m), self.sigma_power)
                                        sigma2_m[is_dense_gene_set_m] = self.get_scaled_sigma2(np.mean(scale_factors_m), orig_sigma2_m, self.sigma_power, self.sigma_threshold_k, self.sigma_threshold_xo)
                            else:
                                sigma2_m = orig_sigma2_m

                            ps_m *= new_p / np.mean(ps_m)

            if betas_trace_out is not None:
                for parallel_num in range(num_parallel):
                    for chain_num in range(num_chains):
                        for i in range(num_gene_sets):
                            gene_set = i
                            if betas_trace_gene_sets is not None and len(betas_trace_gene_sets) == num_gene_sets:
                                gene_set = betas_trace_gene_sets[i]

                            betas_trace_fh.write("%d\t%d\t%d\t%s\t%.4g\t%.4g\t%.4g\t%.4g\t%.4g\t%.4g\t%.4g\t%.4g\t%.4g\t%.4g\t%.4g\t%.4g\t%.4g\t%.4g\n" % (iteration_num, parallel_num+1, chain_num+1, gene_set, curr_post_means_t[chain_num,parallel_num,i] / scale_factors_m[parallel_num,i], curr_betas_t[chain_num,parallel_num,i] / scale_factors_m[parallel_num,i], curr_postp_t[chain_num,parallel_num,i], res_beta_hat_t[chain_num,parallel_num,i] / scale_factors_m[parallel_num,i], beta_tildes_m[parallel_num,i] / scale_factors_m[parallel_num,i], curr_betas_t[chain_num,parallel_num,i], res_beta_hat_t[chain_num,parallel_num,i], beta_tildes_m[parallel_num,i], ses_m[parallel_num,i], sigma2_m[parallel_num,i] if len(np.shape(sigma2_m)) > 0 else sigma2_m, ps_m[parallel_num,i] if len(np.shape(ps_m)) > 0 else ps_m, R_m[parallel_num,i], R_m[parallel_num,i] * beta_weights_m[parallel_num,i], sem2_m[parallel_num, i]))

                betas_trace_fh.flush()

            if will_break:
                break


        if betas_trace_out is not None:
            betas_trace_fh.close()

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

        return (avg_betas_m, avg_postp_m)


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

    #return an iterator over chunks of X in dense format
    #useful when want to conduct matrix calculations for which dense arrays are much faster, but don't have enough memory to cast all of X to dense
    #full_whiten (which multiplies by C^{-1} takes precedence over whiten, which multiplies by C^{1/2}, but whiten defaults to true
    #if mean_shifts/scale_factors are passed in, then shift/rescale the blocks. This is done *before* any whitening
    def _get_num_X_blocks(self, X_orig, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size
        return pegs_get_num_X_blocks(X_orig, batch_size)

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
                        if cur_V is None or options.debug_old_batch:
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
        if self.y_corr_cholesky is not None:
            bail("Sorting genes after setting correlation matrix is unsupported")

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
            (self.mean_shifts_missing, self.scale_factors_missing) = self._calc_X_shift_scale(self.X_orig_missing_gene_sets, self.y_corr_cholesky)

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

        if self.gene_factor_gene_mask is not None:
            self.gene_factor_gene_mask = self.gene_factor_gene_mask[[index_map_rev[x] for x in range(self.gene_pheno_combined_prior_Ys.shape[0])]]

        if self.gene_prob_factor_vector is not None:
            self.gene_prob_factor_vector = self.gene_prob_factor_vector[[index_map_rev[x] for x in range(self.gene_prob_factor_vector.shape[0])]]

        if self.exp_gene_factors is not None:
            self.exp_gene_factors = self.exp_gene_factors[[index_map_rev[x] for x in range(self.exp_gene_factors.shape[0])],:]

            
        self.exp_gene_factors = None #anchor-agnostic factor loadings
        self.gene_prob_factor_vector = None #outer product of this with factor loadings gives anchor specific loadings


        if self.gene_N is not None:
            self.gene_N = self.gene_N[sorted_gene_indices]
        if self.gene_ignored_N is not None:
            self.gene_ignored_N = self.gene_ignored_N[sorted_gene_indices]

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

        
        if options.debug_old_batch:
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


        if self.exp_gene_factors is not None:
            self.exp_gene_factors = self.exp_gene_factors[gene_mask,:]
        if self.gene_factor_gene_mask is not None:
            self.gene_factor_gene_mask = self.gene_factor_gene_mask[gene_mask,:]

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

            if self.inf_betas is not None:
                self.inf_betas_missing = self.inf_betas[remove_mask]

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

            if self.inf_betas_orig is not None:
                self.inf_betas_missing_orig = self.inf_betas_orig[remove_mask]
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

        if self.exp_gene_set_factors is not None:
            self.exp_gene_set_factors = self.exp_gene_set_factors[subset_mask,:]
        if self.gene_set_factor_gene_set_mask is not None:
            self.gene_set_factor_gene_set_mask = self.gene_set_factor_gene_set_mask[subset_mask]


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

        if self.inf_betas is not None:
            self.inf_betas = self.inf_betas[subset_mask]

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

        if self.inf_betas_orig is not None:
            self.inf_betas_orig = self.inf_betas_orig[subset_mask]
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

    #utility function to map names or indices to column indicies
    def _get_col(self, col_name_or_index, header_cols, require_match=True):
        return pegs_resolve_column_index(
            col_name_or_index,
            header_cols,
            require_match=require_match,
            bail_fn=bail,
        )

    # inverse_matrix calculations

def _build_main_mode_state():
    return {
        "run_factor": run_factor,
        "run_phewas": run_phewas,
        "run_factor_phewas": options.factor_phewas_from_gene_phewas_stats_in is not None,
        "run_naive_factor": run_naive_factor,
        "use_phewas_for_factoring": use_phewas_for_factoring,
        "factor_gene_set_x_pheno": factor_gene_set_x_pheno,
        "expand_gene_sets": expand_gene_sets,
        "factor_workflow": factor_workflow,
    }


@dataclass
class FactorOnlyStageResult:
    ran: bool = False
    num_gene_sets: int = 0
    factor_input_state: dict = field(default_factory=dict)


@dataclass
class PhewasStageResult:
    ran: bool = False
    output_path: str | None = None


@dataclass
class FactorStageResult:
    ran: bool = False
    workflow_id: str | None = None
    output_plan: object = None


@dataclass
class FactorWorkflow:
    workflow_id: str | None = None
    label: str | None = None
    factor_gene_set_x_pheno: bool = False
    use_phewas_for_factoring: bool = False
    expand_gene_sets: bool = False


@dataclass
class FactorInputs:
    anchor_gene_mask: object = None
    anchor_pheno_mask: object = None


@dataclass
class FactorExecutionConfig:
    max_num_factors: int
    phi: float
    alpha0: float
    beta0: float
    gene_set_filter_value: object = None
    gene_or_pheno_filter_value: object = None
    pheno_prune_value: object = None
    pheno_prune_number: object = None
    gene_prune_value: object = None
    gene_prune_number: object = None
    gene_set_prune_value: object = None
    gene_set_prune_number: object = None
    anchor_pheno_mask: object = None
    anchor_gene_mask: object = None
    anchor_any_pheno: bool = False
    anchor_any_gene: bool = False
    anchor_gene_set: bool = False
    run_transpose: bool = True
    min_lambda_threshold: float = 1e-3
    lmm_auth_key: object = None
    lmm_model: object = None
    lmm_provider: str = "openai"
    label_gene_sets_only: bool = False
    label_include_phenos: bool = False
    label_individually: bool = False
    project_phenos_from_gene_sets: bool = False

    def to_run_kwargs(self):
        return {
            "max_num_factors": self.max_num_factors,
            "phi": self.phi,
            "alpha0": self.alpha0,
            "beta0": self.beta0,
            "gene_set_filter_value": self.gene_set_filter_value,
            "gene_or_pheno_filter_value": self.gene_or_pheno_filter_value,
            "pheno_prune_value": self.pheno_prune_value,
            "pheno_prune_number": self.pheno_prune_number,
            "gene_prune_value": self.gene_prune_value,
            "gene_prune_number": self.gene_prune_number,
            "gene_set_prune_value": self.gene_set_prune_value,
            "gene_set_prune_number": self.gene_set_prune_number,
            "anchor_pheno_mask": self.anchor_pheno_mask,
            "anchor_gene_mask": self.anchor_gene_mask,
            "anchor_any_pheno": self.anchor_any_pheno,
            "anchor_any_gene": self.anchor_any_gene,
            "anchor_gene_set": self.anchor_gene_set,
            "run_transpose": self.run_transpose,
            "min_lambda_threshold": self.min_lambda_threshold,
            "lmm_auth_key": self.lmm_auth_key,
            "lmm_model": self.lmm_model,
            "lmm_provider": self.lmm_provider,
            "label_gene_sets_only": self.label_gene_sets_only,
            "label_include_phenos": self.label_include_phenos,
            "label_individually": self.label_individually,
            "project_phenos_from_gene_sets": self.project_phenos_from_gene_sets,
        }


@dataclass
class FactorOutputPlan:
    factors_out: str | None = None
    factors_anchor_out: str | None = None
    gene_set_clusters_out: str | None = None
    gene_clusters_out: str | None = None
    pheno_clusters_out: str | None = None
    gene_set_anchor_clusters_out: str | None = None
    gene_anchor_clusters_out: str | None = None
    pheno_anchor_clusters_out: str | None = None
    gene_pheno_stats_out: str | None = None
    max_no_write_gene_pheno: object = None


@dataclass
class MainPipelineResult:
    state: object
    mode_state: dict
    factor_only: FactorOnlyStageResult
    phewas: PhewasStageResult = field(default_factory=PhewasStageResult)
    factor: FactorStageResult = field(default_factory=FactorStageResult)
    factor_phewas: PhewasStageResult = field(default_factory=PhewasStageResult)


def _enforce_factor_only_input_boundary(options, mode_state):
    if not mode_state.get("run_factor"):
        return

    has_x_source = any(
        x is not None
        for x in [options.X_in, options.X_list, options.Xd_in, options.Xd_list]
    )
    if not has_x_source:
        bail(
            "EAGGL requires an X matrix input. Provide --X-in/--X-list/--Xd-in/--Xd-list "
            "(or use --eaggl-bundle-in with an X default)."
        )

    workflow = mode_state.get("factor_workflow")
    use_phewas_for_factoring = bool(workflow and workflow.get("use_phewas_for_factoring"))
    if not use_phewas_for_factoring:
        missing = []
        if options.gene_stats_in is None:
            missing.append("--gene-stats-in")
        if options.gene_set_stats_in is None:
            missing.append("--gene-set-stats-in")
        if len(missing) > 0:
            bail(
                "EAGGL factor workflows require precomputed PIGEAN stats: missing %s "
                "(or provide them in --eaggl-bundle-in)." % ", ".join(missing)
            )


def _run_main_factor_only_pipeline(g, options, mode_state):
    current_workflow = mode_state.get("factor_workflow")
    workflow_id = current_workflow.get("id") if isinstance(current_workflow, dict) else None

    # Read IDs first so read_X can skip gene sets outside selected strategy inputs.
    gene_set_ids = None
    factor_uses_phewas_gene_set_ids = workflow_id in set(["F4", "F5", "F6", "F7", "F8"])
    if factor_uses_phewas_gene_set_ids:
        if options.gene_set_phewas_stats_in is None:
            bail("Need --gene-set-phewas-stats-in")
        gene_set_ids = _read_gene_set_phewas_statistics(
            g,
            options.gene_set_phewas_stats_in,
            stats_id_col=options.gene_set_phewas_stats_id_col,
            stats_pheno_col=options.gene_set_phewas_stats_pheno_col,
            stats_beta_col=options.gene_set_phewas_stats_beta_col,
            stats_beta_uncorrected_col=options.gene_set_phewas_stats_beta_uncorrected_col,
            min_gene_set_beta=options.min_gene_set_read_beta,
            min_gene_set_beta_uncorrected=options.min_gene_set_read_beta_uncorrected,
            return_only_ids=True,
            phenos_to_match=options.anchor_phenos,
            max_num_entries_at_once=options.max_read_entries_at_once,
        )
    elif options.gene_set_stats_in is not None:
        gene_set_ids = _read_gene_set_statistics(
            g,
            options.gene_set_stats_in,
            stats_id_col=options.gene_set_stats_id_col,
            stats_exp_beta_tilde_col=options.gene_set_stats_exp_beta_tilde_col,
            stats_beta_tilde_col=options.gene_set_stats_beta_tilde_col,
            stats_p_col=options.gene_set_stats_p_col,
            stats_se_col=options.gene_set_stats_se_col,
            stats_beta_col=options.gene_set_stats_beta_col,
            stats_beta_uncorrected_col=options.gene_set_stats_beta_uncorrected_col,
            ignore_negative_exp_beta=options.ignore_negative_exp_beta,
            max_gene_set_p=options.max_gene_set_read_p,
            min_gene_set_beta=options.min_gene_set_read_beta,
            min_gene_set_beta_uncorrected=options.min_gene_set_read_beta_uncorrected,
            return_only_ids=True,
        )

    if gene_set_ids is not None:
        log("Will read %d gene sets" % (len(gene_set_ids)), DEBUG)

    # Only read matrix inputs in EAGGL. No beta/prior/Gibbs fitting.
    _run_read_x_stage(
        g,
        options.X_in,
        Xd_in=options.Xd_in,
        X_list=options.X_list,
        Xd_list=options.Xd_list,
        V_in=options.V_in,
        min_gene_set_size=options.min_gene_set_size,
        max_gene_set_size=options.max_gene_set_size,
        only_ids=gene_set_ids,
        only_inc_genes=options.anchor_genes if mode_state["use_phewas_for_factoring"] else None,
        fraction_inc_genes=options.add_gene_sets_by_fraction,
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
        filter_gene_set_p=None,
        max_num_gene_sets_initial=options.max_num_gene_sets_initial,
        max_num_gene_sets=options.max_num_gene_sets,
        max_num_gene_sets_hyper=options.max_num_gene_sets_hyper,
        skip_betas=True,
        batch_separator=options.batch_separator,
        ignore_genes=options.ignore_genes,
        file_separator=options.file_separator,
        show_progress=not options.hide_progress,
        max_num_entries_at_once=options.max_read_entries_at_once,
    )

    if not g.has_gene_sets():
        log("No gene sets survived the input filters; stopping")
        sys.exit(0)

    if options.gene_stats_in is not None:
        _run_read_y_stage(
            g,
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

    if options.gene_set_stats_in is not None:
        _read_gene_set_statistics(
            g,
            options.gene_set_stats_in,
            stats_id_col=options.gene_set_stats_id_col,
            stats_exp_beta_tilde_col=options.gene_set_stats_exp_beta_tilde_col,
            stats_beta_tilde_col=options.gene_set_stats_beta_tilde_col,
            stats_p_col=options.gene_set_stats_p_col,
            stats_se_col=options.gene_set_stats_se_col,
            stats_beta_col=options.gene_set_stats_beta_col,
            stats_beta_uncorrected_col=options.gene_set_stats_beta_uncorrected_col,
            ignore_negative_exp_beta=options.ignore_negative_exp_beta,
            max_gene_set_p=options.max_gene_set_read_p,
            min_gene_set_beta=options.min_gene_set_read_beta,
            min_gene_set_beta_uncorrected=options.min_gene_set_read_beta_uncorrected,
        )

    factor_input_state = {
        "anchor_gene_mask": None,
        "anchor_pheno_mask": None,
    }
    if mode_state["run_factor"]:
        factor_input_state = _load_factor_phewas_inputs(g, options)
    return factor_input_state


_bind_hyperparameter_properties(EagglState)


def _run_read_y_stage(runtime, **read_kwargs):
    return _read_y_pipeline(runtime, **read_kwargs)


def _read_y_pipeline(runtime, gwas_in=None, huge_statistics_in=None, huge_statistics_out=None, exomes_in=None, positive_controls_in=None, positive_controls_list=None, case_counts_in=None, ctrl_counts_in=None, gene_bfs_in=None, gene_loc_file=None, gene_covs_in=None, hold_out_chrom=None, **kwargs):
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
        bail(
            "These inputs belong to pigean.py and are not supported in eaggl.py: %s. "
            "Run pigean.py first and pass outputs via --eaggl-bundle-in or --gene-stats-in/--gene-set-stats-in."
            % ", ".join(sorted(unsupported_flags))
        )

    if positive_controls_in is not None or positive_controls_list is not None:
        warn("Ignoring positive-control inputs in eaggl.py read_Y; using --gene-stats-in values")

    if gene_bfs_in is None:
        bail("Require --gene-stats-in for this operation")

    (Y1, extra_genes, extra_Y, gene_combined_map, gene_prior_map) = runtime._read_gene_bfs(gene_bfs_in, **kwargs)

    def _apply_hold_out_chrom(Y_values, extra_gene_names, extra_Y_values):
        if hold_out_chrom is None:
            return (Y_values, extra_gene_names, extra_Y_values)

        if runtime.gene_to_chrom is None:
            if gene_loc_file is None:
                bail("Option --hold-out-chrom requires --gene-loc-file")
            (
                runtime.gene_chrom_name_pos,
                runtime.gene_to_chrom,
                runtime.gene_to_pos,
            ) = pegs_read_loc_file_with_gene_map(
                gene_loc_file,
                gene_label_map=runtime.gene_label_map,
                clean_chrom_fn=pegs_clean_chrom_name,
                warn_fn=warn,
                bail_fn=bail,
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
            assert(gene_covs.shape[0] == runtime.gene_covariates.shape[0])
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
            runtime.gene_covariate_directions = np.array([runtime.gene_covariate_directions[i] for i in np.where(~constant_features)[0]])

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

            log("Removing feature %s" % runtime.gene_covariate_names[max_index], TRACE)
            runtime.gene_covariates = np.delete(runtime.gene_covariates, max_index, axis=1)
            del runtime.gene_covariate_names[max_index]
            runtime.gene_covariate_directions = np.delete(runtime.gene_covariate_directions, max_index)
            cor_mat = np.delete(np.delete(cor_mat, max_index, axis=1), max_index, axis=0)
            if len(runtime.gene_covariates) == 0:
                bail("Error: something went wrong with matrix inversion. Still couldn't invert after removing all but one column")

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
        runtime.gene_covariates_mat_inv = np.linalg.inv(runtime.gene_covariates[runtime.gene_covariates_mask, :].T.dot(runtime.gene_covariates[runtime.gene_covariates_mask, :]))
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


def _log_runtime_environment_if_requested(options):
    if options.hide_opts:
        return
    log("Python version: %s" % sys.version)
    log("Numpy version: %s" % np.__version__)
    log("Scipy version: %s" % scipy.__version__)
    log("Options: %s" % options)


def _read_gene_map(runtime_state, gene_map_in, gene_map_orig_gene_col=1, gene_map_new_gene_col=2, allow_multi=False):
    runtime_state.gene_label_map = pegs_parse_gene_map_file(
        gene_map_in,
        gene_map_orig_gene_col=gene_map_orig_gene_col,
        gene_map_new_gene_col=gene_map_new_gene_col,
        allow_multi=allow_multi,
        bail_fn=bail,
    )


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


def _initialize_main_mappings(g, options):
    if options.gene_map_in:
        _read_gene_map(
            g,
            options.gene_map_in,
            options.gene_map_orig_gene_col,
            options.gene_map_new_gene_col,
        )
    if options.gene_loc_file:
        _init_gene_locs(g, options.gene_loc_file)


def _read_gene_set_statistics(
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
    return pegs_load_and_apply_gene_set_statistics_to_runtime(
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
        open_text_fn=open_gz,
        get_col_fn=runtime_state._get_col,
        parse_log_fn=lambda message: log(message, INFO),
        apply_log_fn=lambda message: log(message, DEBUG),
        warn_fn=warn,
        bail_fn=bail,
    )


def _read_gene_set_phewas_statistics(
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
    return pegs_load_and_apply_gene_set_phewas_statistics_to_runtime(
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
        open_text_fn=open_gz,
        get_col_fn=runtime_state._get_col,
        construct_map_to_ind_fn=pegs_construct_map_to_ind,
        warn_fn=warn,
        bail_fn=bail,
        log_fn=lambda message: log(message, DEBUG),
    )


def _derive_factor_anchor_masks(g, options):
    return pegs_derive_factor_anchor_masks(
        genes=g.genes,
        phenos=g.phenos,
        anchor_genes=options.anchor_genes,
        anchor_phenos=options.anchor_phenos,
        bail_fn=bail,
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
        get_col_fn=state._get_col,
        construct_map_to_ind_fn=pegs_construct_map_to_ind,
        warn_fn=warn,
        bail_fn=bail,
        log_fn=lambda message: log(message, DEBUG),
    )
    state.phewas_state = pegs_sync_phewas_runtime_state(state)


def _has_loaded_gene_phewas(runtime):
    return (
        runtime.gene_pheno_Y is not None
        or runtime.gene_pheno_combined_prior_Ys is not None
        and runtime.gene_pheno_priors is not None
    )


def _reread_gene_phewas_bfs(state):
    if state.cached_gene_phewas_call is None:
        return
    log("Rereading gene phewas bfs...")
    _read_gene_phewas_bfs(state, **state.cached_gene_phewas_call)


def _load_factor_phewas_inputs(g, options):
    # Factor/projection workflows consume these as matrix inputs; this is distinct
    # from standalone PheWAS execution which is handled in a separate stage.
    factor_input_data = _derive_factor_anchor_masks(g, options)
    if options.gene_set_phewas_stats_in is not None:
        _read_gene_set_phewas_statistics(
            g,
            options.gene_set_phewas_stats_in,
            stats_id_col=options.gene_set_phewas_stats_id_col,
            stats_pheno_col=options.gene_set_phewas_stats_pheno_col,
            stats_beta_col=options.gene_set_phewas_stats_beta_col,
            stats_beta_uncorrected_col=options.gene_set_phewas_stats_beta_uncorrected_col,
            min_gene_set_beta=options.min_gene_set_read_beta,
            min_gene_set_beta_uncorrected=options.min_gene_set_read_beta_uncorrected,
            max_num_entries_at_once=options.max_read_entries_at_once,
        )
        factor_input_data.loaded_gene_set_phewas_stats = True

    if options.gene_phewas_bfs_in:
        _read_gene_phewas_bfs(
            g,
            gene_phewas_bfs_in=options.gene_phewas_bfs_in,
            gene_phewas_bfs_id_col=options.gene_phewas_bfs_id_col,
            gene_phewas_bfs_pheno_col=options.gene_phewas_bfs_pheno_col,
            anchor_genes=options.anchor_genes,
            anchor_phenos=options.anchor_phenos,
            gene_phewas_bfs_log_bf_col=options.gene_phewas_bfs_log_bf_col,
            gene_phewas_bfs_combined_col=options.gene_phewas_bfs_combined_col,
            gene_phewas_bfs_prior_col=options.gene_phewas_bfs_prior_col,
            phewas_gene_to_X_gene_in=options.gene_phewas_id_to_X_id,
            min_value=options.min_gene_phewas_read_value,
            max_num_entries_at_once=options.max_read_entries_at_once,
        )
        factor_input_data.loaded_gene_phewas_bfs = True
    return factor_input_data


def _write_main_primary_outputs(g, options):
    if options.gene_set_stats_out:
        g.write_gene_set_statistics(options.gene_set_stats_out, max_no_write_gene_set_beta=options.max_no_write_gene_set_beta, max_no_write_gene_set_beta_uncorrected=options.max_no_write_gene_set_beta_uncorrected)
    if options.phewas_gene_set_stats_out:
        g.write_phewas_gene_set_statistics(options.phewas_gene_set_stats_out, max_no_write_gene_set_beta=options.max_no_write_gene_set_beta, max_no_write_gene_set_beta_uncorrected=options.max_no_write_gene_set_beta_uncorrected)
    if options.gene_stats_out:
        g.write_gene_statistics(options.gene_stats_out)
    if options.gene_gene_set_stats_out:
        g.write_gene_gene_set_statistics(options.gene_gene_set_stats_out, max_no_write_gene_gene_set_beta=options.max_no_write_gene_gene_set_beta, write_filter_beta_uncorrected=options.use_beta_uncorrected_for_gene_gene_set_write_filter)
    if options.gene_set_overlap_stats_out:
        g.write_gene_set_overlap_statistics(options.gene_set_overlap_stats_out)
    if options.gene_covs_out:
        g.write_gene_covariates(options.gene_covs_out)
    if options.gene_effectors_out:
        g.write_gene_effectors(options.gene_effectors_out)


def _resolve_gene_phewas_stage_decision(g, requested_input, reusable_inputs):
    return pegs_resolve_gene_phewas_input_decision_for_stage(
        requested_input=requested_input,
        reusable_inputs=reusable_inputs,
        read_gene_phewas=_has_loaded_gene_phewas(g),
        num_gene_phewas_filtered=g.num_gene_phewas_filtered,
    )


def _run_phewas_with_common_args(g, options, gene_phewas_bfs_in, run_for_factors=False, min_gene_factor_weight=0):
    phewas_config = pegs_build_phewas_stage_config(
        gene_phewas_bfs_in=gene_phewas_bfs_in,
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
        run_for_factors=run_for_factors,
        batch_size=300 if run_for_factors else None,
        min_gene_factor_weight=min_gene_factor_weight,
    )
    run_kwargs = phewas_config.to_run_kwargs()
    g.run_phewas(**run_kwargs)


def _run_main_phewas_stage(g, options):
    decision = _resolve_gene_phewas_stage_decision(
        g,
        options.run_phewas_from_gene_phewas_stats_in,
        [options.gene_phewas_bfs_in],
    )
    log("PheWAS stage 'phewas': mode=%s reason=%s" % (decision.mode, decision.reason), INFO)
    bfs_to_use = decision.resolved_input
    _run_phewas_with_common_args(g, options, bfs_to_use, run_for_factors=False)
    if options.phewas_stats_out:
        g.write_phewas_statistics(options.phewas_stats_out)
    return PhewasStageResult(ran=True, output_path=options.phewas_stats_out)


def _extract_factor_workflow(mode_state):
    workflow = mode_state.get("factor_workflow") if isinstance(mode_state, dict) else None
    if not isinstance(workflow, dict):
        return FactorWorkflow()
    return FactorWorkflow(
        workflow_id=workflow.get("id"),
        label=workflow.get("label"),
        factor_gene_set_x_pheno=bool(workflow.get("factor_gene_set_x_pheno")),
        use_phewas_for_factoring=bool(workflow.get("use_phewas_for_factoring")),
        expand_gene_sets=bool(workflow.get("expand_gene_sets")),
    )


def _extract_factor_inputs(factor_input_state):
    if isinstance(factor_input_state, FactorInputs):
        return factor_input_state
    if factor_input_state is None:
        return FactorInputs()
    return FactorInputs(
        anchor_gene_mask=getattr(factor_input_state, "anchor_gene_mask", None)
        if not isinstance(factor_input_state, dict)
        else factor_input_state.get("anchor_gene_mask"),
        anchor_pheno_mask=getattr(factor_input_state, "anchor_pheno_mask", None)
        if not isinstance(factor_input_state, dict)
        else factor_input_state.get("anchor_pheno_mask"),
    )


def _resolve_factor_gene_or_pheno_filter_value(options, workflow):
    if options.anchor_gene_set:
        return options.gene_set_pheno_filter_value
    if workflow.factor_gene_set_x_pheno:
        return options.pheno_filter_value
    return options.gene_filter_value


def _build_factor_execution_config(options, workflow, factor_inputs):
    return FactorExecutionConfig(
        max_num_factors=options.max_num_factors,
        phi=options.phi,
        alpha0=options.alpha0,
        beta0=options.beta0,
        gene_set_filter_value=options.gene_set_filter_value,
        gene_or_pheno_filter_value=_resolve_factor_gene_or_pheno_filter_value(options, workflow),
        pheno_prune_value=options.factor_prune_phenos_val,
        pheno_prune_number=options.factor_prune_phenos_num,
        gene_prune_value=options.factor_prune_genes_val,
        gene_prune_number=options.factor_prune_genes_num,
        gene_set_prune_value=options.factor_prune_gene_sets_val,
        gene_set_prune_number=options.factor_prune_gene_sets_num,
        anchor_pheno_mask=factor_inputs.anchor_pheno_mask,
        anchor_gene_mask=factor_inputs.anchor_gene_mask,
        anchor_any_pheno=options.anchor_any_pheno,
        anchor_any_gene=options.anchor_any_gene,
        anchor_gene_set=options.anchor_gene_set,
        run_transpose=not options.no_transpose,
        min_lambda_threshold=options.min_lambda_threshold,
        lmm_auth_key=options.lmm_auth_key,
        lmm_model=options.lmm_model,
        lmm_provider=options.lmm_provider,
        label_gene_sets_only=options.label_gene_sets_only,
        label_include_phenos=options.label_include_phenos,
        label_individually=options.label_individually,
        project_phenos_from_gene_sets=options.project_phenos_from_gene_sets,
    )


def _run_factor_model(g, factor_config):
    g.run_factor(**factor_config.to_run_kwargs())


def _run_main_factor_stage(g, options, mode_state, factor_input_state):
    workflow = _extract_factor_workflow(mode_state)
    factor_inputs = _extract_factor_inputs(factor_input_state)
    factor_config = _build_factor_execution_config(options, workflow, factor_inputs)
    _run_factor_model(g, factor_config)
    return FactorStageResult(ran=True, workflow_id=workflow.workflow_id)


def _build_factor_output_plan(options):
    return FactorOutputPlan(
        factors_out=options.factors_out,
        factors_anchor_out=options.factors_anchor_out,
        gene_set_clusters_out=options.gene_set_clusters_out,
        gene_clusters_out=options.gene_clusters_out,
        pheno_clusters_out=options.pheno_clusters_out,
        gene_set_anchor_clusters_out=options.gene_set_anchor_clusters_out,
        gene_anchor_clusters_out=options.gene_anchor_clusters_out,
        pheno_anchor_clusters_out=options.pheno_anchor_clusters_out,
        gene_pheno_stats_out=options.gene_pheno_stats_out,
        max_no_write_gene_pheno=options.max_no_write_gene_pheno,
    )


def _write_factor_outputs_for_plan(g, output_plan):
    if output_plan.factors_out is not None:
        g.write_matrix_factors(output_plan.factors_out)
    if output_plan.factors_anchor_out is not None:
        g.write_matrix_factors(output_plan.factors_anchor_out, write_anchor_specific=True)
    if (
        output_plan.gene_set_clusters_out is not None
        or output_plan.gene_clusters_out is not None
        or output_plan.pheno_clusters_out is not None
    ):
        g.write_clusters(
            output_plan.gene_set_clusters_out,
            output_plan.gene_clusters_out,
            output_plan.pheno_clusters_out,
        )
    if (
        output_plan.gene_set_anchor_clusters_out is not None
        or output_plan.gene_anchor_clusters_out is not None
        or output_plan.pheno_anchor_clusters_out is not None
    ):
        g.write_clusters(
            output_plan.gene_set_anchor_clusters_out,
            output_plan.gene_anchor_clusters_out,
            output_plan.pheno_anchor_clusters_out,
            write_anchor_specific=True,
        )
    if output_plan.gene_pheno_stats_out is not None:
        g.write_gene_pheno_statistics(
            output_plan.gene_pheno_stats_out,
            min_value_to_print=output_plan.max_no_write_gene_pheno,
        )


def _write_main_factor_outputs(g, options):
    output_plan = _build_factor_output_plan(options)
    _write_factor_outputs_for_plan(g, output_plan)


def _run_main_factor_phewas_stage(g, options):
    if g.num_factors() <= 0:
        log("No factors; not performing factor phewas")
        return PhewasStageResult(ran=False, output_path=options.factor_phewas_stats_out)

    decision = _resolve_gene_phewas_stage_decision(
        g,
        options.factor_phewas_from_gene_phewas_stats_in,
        [options.gene_phewas_bfs_in, options.run_phewas_from_gene_phewas_stats_in],
    )
    log(
        "PheWAS stage 'factor_phewas': mode=%s reason=%s" % (decision.mode, decision.reason),
        INFO,
    )
    bfs_to_use = decision.resolved_input
    _run_phewas_with_common_args(
        g,
        options,
        bfs_to_use,
        run_for_factors=True,
        min_gene_factor_weight=options.factor_phewas_min_gene_factor_weight,
    )
    if options.factor_phewas_stats_out:
        g.write_factor_phewas_statistics(options.factor_phewas_stats_out)
    return PhewasStageResult(ran=True, output_path=options.factor_phewas_stats_out)


def _should_run_main_factor_phewas_stage(mode_state):
    return bool(mode_state["run_factor"] and mode_state["run_factor_phewas"])


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
    # hyper state, then restore vector fields before returning explicit values.
    orig_ps = runtime_state.ps
    orig_sigma2s = runtime_state.sigma2s
    runtime_state.ps = None
    runtime_state.sigma2s = None
    try:
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
    finally:
        runtime_state.ps = orig_ps
        runtime_state.sigma2s = orig_sigma2s


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

    # Learn batch-level hyper values.
    assert(runtime_state.gene_set_batches[0] is not None)
    ordered_batches = [runtime_state.gene_set_batches[0]] + list(set([x for x in runtime_state.gene_set_batches if not x == runtime_state.gene_set_batches[0]]))
    batches_num_ignored = {}
    for i in range(len(batches)):
        if batches[i] not in batches_num_ignored:
            batches_num_ignored[batches[i]] = 0
        batches_num_ignored[batches[i]] += num_ignored_gene_sets[i]

    if update_hyper_p:
        runtime_state.ps = np.full(len(runtime_state.gene_set_batches), np.nan)
    runtime_state.sigma2s = np.full(len(runtime_state.gene_set_batches), np.nan)

    first_p = None
    for ordered_batch_ind in range(len(ordered_batches)):
        if ordered_batches[ordered_batch_ind] is None:
            assert(first_for_hyper)
            continue

        gene_sets_in_batch_mask = (runtime_state.gene_set_batches == ordered_batches[ordered_batch_ind])
        gene_sets_for_hyper_mask = gene_sets_in_batch_mask.copy()

        if max_num_gene_sets_hyper is not None and np.sum(gene_sets_for_hyper_mask) > max_num_gene_sets_hyper:
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


GeneSetData = EagglState


def run_main_pipeline(options):

    mode_state = _build_main_mode_state()
    _enforce_factor_only_input_boundary(options, mode_state)
    _log_runtime_environment_if_requested(options)

    g = EagglState(background_prior=options.background_prior, batch_size=options.batch_size)
    _initialize_main_mappings(g, options)
    factor_input_state = _run_main_factor_only_pipeline(g, options, mode_state)
    factor_only_stage_result = FactorOnlyStageResult(
        ran=True,
        num_gene_sets=len(g.gene_sets) if g.gene_sets is not None else 0,
        factor_input_state=factor_input_state,
    )

    _write_main_primary_outputs(g, options)

    phewas_stage_result = PhewasStageResult(ran=False, output_path=options.phewas_stats_out)
    if mode_state["run_phewas"]:
        phewas_stage_result = _run_main_phewas_stage(g, options)

    factor_model_stage_result = FactorStageResult(ran=False, workflow_id=None)
    if mode_state["run_factor"]:
        factor_model_stage_result = _run_main_factor_stage(g, options, mode_state, factor_input_state)

    _write_main_factor_outputs(g, options)

    factor_phewas_stage_result = PhewasStageResult(ran=False, output_path=options.factor_phewas_stats_out)
    if _should_run_main_factor_phewas_stage(mode_state):
        factor_phewas_stage_result = _run_main_factor_phewas_stage(g, options)


    if options.params_out:
        g.write_params(options.params_out)

    return MainPipelineResult(
        state=g,
        mode_state=mode_state,
        factor_only=factor_only_stage_result,
        phewas=phewas_stage_result,
        factor=factor_model_stage_result,
        factor_phewas=factor_phewas_stage_result,
    )


def main():
    run_main_pipeline(options)

if __name__ == '__main__':

    #profiler = cProfile.Profile()
    #profiler.enable()

    #cProfile.run('main()')
    main()

    #profiler.disable()
    #profiler.dump_stats('output.prof')
