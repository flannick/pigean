import json
import os
import shutil
import tarfile
import tempfile
import hashlib
import csv
import gzip
import io
import re
import sys
import time
import urllib.error
import urllib.request
import copy
from dataclasses import dataclass, field

import numpy as np
import scipy.stats
import scipy.linalg
import scipy.sparse as sparse

EAGGL_BUNDLE_SCHEMA = "pigean_eaggl_bundle/v1"
EAGGL_BUNDLE_ALLOWED_DEFAULT_INPUTS = set([
    "X_in",
    "gene_stats_in",
    "gene_set_stats_in",
    "gene_phewas_bfs_in",
    "gene_set_phewas_stats_in",
])

DIG_OPEN_DATA_PREFIX = "dig-open-data:"
DIG_OPEN_DATA_TOKEN_RE = re.compile(r"^[A-Za-z0-9_.+-]+$")


@dataclass
class XData:
    X_orig: object = None
    X_orig_missing_genes: object = None
    X_orig_missing_gene_sets: object = None
    X_orig_missing_genes_missing_gene_sets: object = None
    genes: list = field(default_factory=list)
    genes_missing: list = field(default_factory=list)
    gene_sets: list = field(default_factory=list)
    gene_sets_missing: list = field(default_factory=list)
    gene_sets_ignored: list = field(default_factory=list)
    gene_to_ind: dict = field(default_factory=dict)
    gene_set_to_ind: dict = field(default_factory=dict)
    scale_factors: object = None
    mean_shifts: object = None
    gene_set_batches: object = None
    gene_set_labels: object = None
    is_dense_gene_set: object = None

    @classmethod
    def from_input_plan(cls, input_plan):
        return cls(
            gene_set_batches=np.array(input_plan.batches),
            gene_set_labels=np.array(input_plan.labels),
            is_dense_gene_set=np.array(input_plan.is_dense, dtype=bool),
        )

    def has_gene_sets(self):
        return bool(self.gene_sets is not None and len(self.gene_sets) > 0)

    def num_genes(self):
        return 0 if self.genes is None else len(self.genes)

    def num_gene_sets(self):
        return 0 if self.gene_sets is None else len(self.gene_sets)

    def seed_runtime_read_x_state(self, runtime):
        runtime.gene_set_batches = self.gene_set_batches[:0]
        runtime.gene_set_labels = self.gene_set_labels[:0]
        runtime.gene_sets = []
        runtime.is_dense_gene_set = self.is_dense_gene_set[:0]

    def run_ingestion_stage(
        self,
        runtime,
        input_plan,
        read_config,
        read_callbacks,
        ingestion_options,
        *,
        ensure_gene_universe_fn,
        process_x_input_file_fn,
        remove_tag_from_input_fn,
        log_fn,
        info_level,
        debug_level,
    ):
        initial_ps = input_plan.initial_ps
        X_ins = input_plan.X_ins
        batches = input_plan.batches
        labels = input_plan.labels
        orig_files = input_plan.orig_files
        is_dense = input_plan.is_dense

        batches, num_ignored_gene_sets = initialize_read_x_batch_seed_state(
            runtime=runtime,
            xdata_seed=self,
            batches=batches,
            orig_files=orig_files,
            batch_all_for_hyper=ingestion_options.batch_all_for_hyper,
            first_for_hyper=ingestion_options.first_for_hyper,
            update_hyper_sigma=ingestion_options.update_hyper_sigma,
            update_hyper_p=ingestion_options.update_hyper_p,
            first_for_sigma_cond=ingestion_options.first_for_sigma_cond,
            record_params_fn=runtime._record_params,
            log_fn=log_fn,
        )

        if (
            (ingestion_options.filter_gene_set_p < 1 or ingestion_options.filter_gene_set_metric_z)
            and runtime.Y is not None
        ):
            initialize_filtered_gene_set_state(runtime, update_hyper_p=ingestion_options.update_hyper_p)
            maybe_prepare_filtered_gls_correlation(
                runtime=runtime,
                run_gls=ingestion_options.run_gls,
                run_corrected_ols=ingestion_options.run_corrected_ols,
                gene_cor_file=ingestion_options.gene_cor_file,
                gene_loc_file=ingestion_options.gene_loc_file,
                gene_cor_file_gene_col=ingestion_options.gene_cor_file_gene_col,
                gene_cor_file_cor_start_col=ingestion_options.gene_cor_file_cor_start_col,
            )

        resolved_run_logistic = resolve_read_x_run_logistic(
            runtime=runtime,
            run_logistic=ingestion_options.run_logistic,
            max_for_linear=ingestion_options.max_for_linear,
            background_log_bf=runtime.background_log_bf,
            record_param_fn=runtime._record_param,
            log_fn=lambda message: log_fn(message, debug_level),
        )

        ignored_for_fraction_inc = run_read_x_ingestion(
            runtime,
            X_ins=X_ins,
            is_dense=is_dense,
            batches=batches,
            labels=labels,
            initial_ps=initial_ps,
            num_ignored_gene_sets=num_ignored_gene_sets,
            read_config=read_config,
            read_callbacks=read_callbacks,
            run_logistic=resolved_run_logistic,
            only_ids=ingestion_options.only_ids,
            add_all_genes=ingestion_options.add_all_genes,
            only_inc_genes=ingestion_options.only_inc_genes,
            fraction_inc_genes=ingestion_options.fraction_inc_genes,
            ignore_genes=ingestion_options.ignore_genes,
            max_num_entries_at_once=ingestion_options.max_num_entries_at_once,
            ensure_gene_universe_fn=ensure_gene_universe_fn,
            process_x_input_file_fn=process_x_input_file_fn,
            remove_tag_from_input_fn=remove_tag_from_input_fn,
            log_fn=log_fn,
            info_level=info_level,
            debug_level=debug_level,
        )

        return {
            "batches": batches,
            "num_ignored_gene_sets": num_ignored_gene_sets,
            "ignored_for_fraction_inc": ignored_for_fraction_inc,
            "run_logistic": resolved_run_logistic,
        }

    def run_post_stage(self, runtime, post_options, post_callbacks, *, log_fn, debug_level):
        if post_options.ignored_for_fraction_inc > 0:
            log_fn(
                "Ignored %d gene sets due to too small a fraction of anchor genes"
                % post_options.ignored_for_fraction_inc,
                debug_level,
            )

        if not runtime.has_gene_sets():
            log_fn("No gene sets to analyze; returning")
            return False

        post_callbacks.standardize_qc_metrics_after_x_read_fn(runtime)
        post_callbacks.maybe_correct_gene_set_betas_after_x_read_fn(
            runtime,
            filter_gene_set_p=post_options.filter_gene_set_p,
            correct_betas_mean=post_options.correct_betas_mean,
            correct_betas_var=post_options.correct_betas_var,
            filter_using_phewas=post_options.filter_using_phewas,
        )

        runtime._record_param("gene_set_prune_threshold", post_options.prune_gene_sets)
        runtime._record_param("gene_set_weighted_prune_threshold", post_options.weighted_prune_gene_sets)
        runtime._record_param("gene_set_prune_deterinistically", post_options.prune_deterministically)

        post_callbacks.maybe_limit_initial_gene_sets_by_p_fn(
            runtime,
            max_num_gene_sets_initial=post_options.max_num_gene_sets_initial,
        )
        post_callbacks.maybe_prune_gene_sets_after_x_read_fn(
            runtime,
            skip_betas=post_options.skip_betas,
            prune_gene_sets=post_options.prune_gene_sets,
            prune_deterministically=post_options.prune_deterministically,
            weighted_prune_gene_sets=post_options.weighted_prune_gene_sets,
        )

        fixed_sigma_cond = post_callbacks.initialize_hyper_defaults_after_x_read_fn(
            runtime,
            initial_p=post_options.initial_p,
            update_hyper_p=post_options.update_hyper_p,
            sigma_power=post_options.sigma_power,
            initial_sigma2_cond=post_options.initial_sigma2_cond,
            update_hyper_sigma=post_options.update_hyper_sigma,
            initial_sigma2=post_options.initial_sigma2,
            sigma_soft_threshold_95=post_options.sigma_soft_threshold_95,
            sigma_soft_threshold_5=post_options.sigma_soft_threshold_5,
        )

        post_callbacks.maybe_learn_batch_hyper_after_x_read_fn(
            runtime,
            skip_betas=post_options.skip_betas,
            update_hyper_p=post_options.update_hyper_p,
            update_hyper_sigma=post_options.update_hyper_sigma,
            batches=post_options.batches,
            num_ignored_gene_sets=post_options.num_ignored_gene_sets,
            first_for_hyper=post_options.first_for_hyper,
            max_num_gene_sets_hyper=post_options.max_num_gene_sets_hyper,
            first_for_sigma_cond=post_options.first_for_sigma_cond,
            fixed_sigma_cond=fixed_sigma_cond,
            first_max_p_for_hyper=post_options.first_max_p_for_hyper,
            max_num_burn_in=post_options.max_num_burn_in,
            max_num_iter_betas=post_options.max_num_iter_betas,
            min_num_iter_betas=post_options.min_num_iter_betas,
            num_chains_betas=post_options.num_chains_betas,
            r_threshold_burn_in_betas=post_options.r_threshold_burn_in_betas,
            use_max_r_for_convergence_betas=post_options.use_max_r_for_convergence_betas,
            max_frac_sem_betas=post_options.max_frac_sem_betas,
            max_allowed_batch_correlation=post_options.max_allowed_batch_correlation,
            sigma_num_devs_to_top=post_options.sigma_num_devs_to_top,
            p_noninf_inflate=post_options.p_noninf_inflate,
            sparse_solution=post_options.sparse_solution,
            sparse_frac_betas=post_options.sparse_frac_betas,
            betas_trace_out=post_options.betas_trace_out,
        )

        post_callbacks.maybe_adjust_overaggressive_p_filter_after_x_read_fn(
            runtime,
            filter_gene_set_p=post_options.filter_gene_set_p,
            increase_filter_gene_set_p=post_options.increase_filter_gene_set_p,
            filter_using_phewas=post_options.filter_using_phewas,
        )
        post_callbacks.apply_post_read_gene_set_size_and_qc_filters_fn(
            runtime,
            min_gene_set_size=post_options.min_gene_set_size,
            max_gene_set_size=post_options.max_gene_set_size,
            filter_gene_set_metric_z=post_options.filter_gene_set_metric_z,
        )

        if runtime.p_values is not None:
            sort_rank = -np.sqrt(-np.log(runtime.p_values + 1e-200))
        else:
            sort_rank = None
        sort_rank = post_callbacks.maybe_filter_zero_uncorrected_betas_after_x_read_fn(
            runtime,
            sort_rank=sort_rank,
            skip_betas=post_options.skip_betas,
            filter_gene_set_p=post_options.filter_gene_set_p,
            filter_using_phewas=post_options.filter_using_phewas,
            max_num_burn_in=post_options.max_num_burn_in,
            max_num_iter_betas=post_options.max_num_iter_betas,
            min_num_iter_betas=post_options.min_num_iter_betas,
            num_chains_betas=post_options.num_chains_betas,
            r_threshold_burn_in_betas=post_options.r_threshold_burn_in_betas,
            use_max_r_for_convergence_betas=post_options.use_max_r_for_convergence_betas,
            max_frac_sem_betas=post_options.max_frac_sem_betas,
            max_allowed_batch_correlation=post_options.max_allowed_batch_correlation,
            sparse_solution=post_options.sparse_solution,
            sparse_frac_betas=post_options.sparse_frac_betas,
        )
        post_callbacks.maybe_reduce_gene_sets_to_max_after_x_read_fn(
            runtime,
            skip_betas=post_options.skip_betas,
            max_num_gene_sets=post_options.max_num_gene_sets,
            sort_rank=sort_rank,
        )
        post_callbacks.record_read_x_counts_fn(
            runtime,
            record_param_fn=runtime._record_param,
            log_fn=lambda message: log_fn(message),
        )
        return True


@dataclass
class XInputPlan:
    initial_ps: object
    X_ins: list
    batches: list
    labels: list
    orig_files: list
    is_dense: list


@dataclass
class XReadConfig:
    x_sparsify: object
    min_gene_set_size: int
    add_ext: bool
    add_top: bool
    add_bottom: bool
    threshold_weights: float
    cap_weights: bool
    permute_gene_sets: bool
    filter_gene_set_p: float
    filter_gene_set_metric_z: float
    filter_using_phewas: bool
    increase_filter_gene_set_p: float
    filter_negative: bool


@dataclass
class XReadCallbacks:
    sparse_module: object
    np_module: object
    normalize_dense_gene_rows_fn: object
    build_sparse_x_from_dense_input_fn: object
    reindex_x_rows_to_current_genes_fn: object
    normalize_gene_set_weights_fn: object
    partition_missing_gene_rows_fn: object
    maybe_permute_gene_set_rows_fn: object
    maybe_prefilter_x_block_fn: object
    merge_missing_gene_rows_fn: object
    finalize_added_x_block_fn: object


@dataclass
class XReadIngestionOptions:
    batch_all_for_hyper: bool
    first_for_hyper: bool
    update_hyper_sigma: bool
    update_hyper_p: bool
    first_for_sigma_cond: bool
    run_gls: bool
    run_corrected_ols: bool
    gene_cor_file: object
    gene_loc_file: object
    gene_cor_file_gene_col: object
    gene_cor_file_cor_start_col: object
    run_logistic: bool
    max_for_linear: float
    only_ids: object
    add_all_genes: bool
    only_inc_genes: object
    fraction_inc_genes: object
    ignore_genes: object
    max_num_entries_at_once: object
    filter_gene_set_p: float
    filter_gene_set_metric_z: float
    filter_using_phewas: bool


@dataclass
class XReadPostOptions:
    ignored_for_fraction_inc: int
    filter_gene_set_p: float
    correct_betas_mean: bool
    correct_betas_var: bool
    filter_using_phewas: bool
    prune_gene_sets: float
    weighted_prune_gene_sets: object
    prune_deterministically: bool
    max_num_gene_sets_initial: object
    skip_betas: bool
    initial_p: float
    update_hyper_p: bool
    sigma_power: object
    initial_sigma2_cond: object
    update_hyper_sigma: bool
    initial_sigma2: object
    sigma_soft_threshold_95: object
    sigma_soft_threshold_5: object
    batches: list
    num_ignored_gene_sets: object
    first_for_hyper: bool
    max_num_gene_sets_hyper: object
    first_for_sigma_cond: bool
    first_max_p_for_hyper: bool
    max_num_burn_in: object
    max_num_iter_betas: int
    min_num_iter_betas: int
    num_chains_betas: int
    r_threshold_burn_in_betas: float
    use_max_r_for_convergence_betas: bool
    max_frac_sem_betas: float
    max_allowed_batch_correlation: object
    sigma_num_devs_to_top: float
    p_noninf_inflate: float
    sparse_solution: bool
    sparse_frac_betas: object
    betas_trace_out: object
    increase_filter_gene_set_p: float
    min_gene_set_size: int
    max_gene_set_size: int
    filter_gene_set_metric_z: float
    max_num_gene_sets: object


@dataclass
class XReadPostCallbacks:
    standardize_qc_metrics_after_x_read_fn: object
    maybe_correct_gene_set_betas_after_x_read_fn: object
    maybe_limit_initial_gene_sets_by_p_fn: object
    maybe_prune_gene_sets_after_x_read_fn: object
    initialize_hyper_defaults_after_x_read_fn: object
    maybe_learn_batch_hyper_after_x_read_fn: object
    maybe_adjust_overaggressive_p_filter_after_x_read_fn: object
    apply_post_read_gene_set_size_and_qc_filters_fn: object
    maybe_filter_zero_uncorrected_betas_after_x_read_fn: object
    maybe_reduce_gene_sets_to_max_after_x_read_fn: object
    record_read_x_counts_fn: object

@dataclass
class ParsedGeneSetStats:
    need_to_take_log: bool
    has_beta_tilde: bool
    has_p_or_se: bool
    has_beta: bool
    has_beta_uncorrected: bool
    records: dict


@dataclass
class ParsedGeneBfs:
    gene_in_bfs: dict
    gene_in_combined: object
    gene_in_priors: object


@dataclass
class ParsedGeneCovariates:
    cov_names: list
    gene_to_covs: dict


@dataclass
class AlignedGeneBfs:
    gene_bfs: object
    extra_genes: list
    extra_gene_bfs: object
    gene_in_combined: object
    gene_in_priors: object


@dataclass
class AlignedGeneCovariates:
    cov_names: list
    gene_covs: object
    extra_genes: list
    extra_gene_covs: object


@dataclass
class ParsedGenePhewasBfs:
    phenos: list
    pheno_to_ind: dict
    row: object
    col: object
    Ys: object
    combineds: object
    priors: object
    num_filtered: int


@dataclass
class YData:
    Y: object = None
    Y_for_regression: object = None
    Y_exomes: object = None
    Y_positive_controls: object = None
    Y_case_counts: object = None
    y_var: object = None
    y_corr: object = None
    y_corr_cholesky: object = None
    y_corr_sparse: object = None
    Y_w: object = None
    Y_fw: object = None
    y_w_var: object = None
    y_w_mean: object = None
    y_fw_var: object = None
    y_fw_mean: object = None


@dataclass
class HyperparameterData:
    p: object = None
    sigma2: object = None
    sigma_power: object = None
    sigma2_osc: object = None
    sigma2_se: object = None
    sigma2_p: object = None
    sigma2_total_var: object = None
    sigma2_total_var_lower: object = None
    sigma2_total_var_upper: object = None
    ps: object = None
    sigma2s: object = None
    sigma2s_missing: object = None

    @classmethod
    def from_runtime(cls, runtime):
        return cls(
            p=getattr(runtime, "p", None),
            sigma2=getattr(runtime, "sigma2", None),
            sigma_power=getattr(runtime, "sigma_power", None),
            sigma2_osc=getattr(runtime, "sigma2_osc", None),
            sigma2_se=getattr(runtime, "sigma2_se", None),
            sigma2_p=getattr(runtime, "sigma2_p", None),
            sigma2_total_var=getattr(runtime, "sigma2_total_var", None),
            sigma2_total_var_lower=getattr(runtime, "sigma2_total_var_lower", None),
            sigma2_total_var_upper=getattr(runtime, "sigma2_total_var_upper", None),
            ps=getattr(runtime, "ps", None),
            sigma2s=getattr(runtime, "sigma2s", None),
            sigma2s_missing=getattr(runtime, "sigma2s_missing", None),
        )

    def apply_to_runtime(self, runtime):
        runtime.p = self.p
        runtime.sigma2 = self.sigma2
        runtime.sigma_power = self.sigma_power
        runtime.sigma2_osc = self.sigma2_osc
        runtime.sigma2_se = self.sigma2_se
        runtime.sigma2_p = self.sigma2_p
        runtime.sigma2_total_var = self.sigma2_total_var
        runtime.sigma2_total_var_lower = self.sigma2_total_var_lower
        runtime.sigma2_total_var_upper = self.sigma2_total_var_upper
        runtime.ps = self.ps
        runtime.sigma2s = self.sigma2s
        runtime.sigma2s_missing = self.sigma2s_missing
        runtime.hyperparameter_state = self
        return self

    def get_p(self):
        return self.p

    def set_p(self, p):
        if p is not None:
            if p > 1:
                p = 1
            if p < 0:
                p = 0
        self.p = p
        return self

    def get_sigma2(self):
        return self.sigma2

    def set_sigma(
        self,
        runtime,
        sigma2,
        sigma_power,
        sigma2_osc=None,
        sigma2_se=None,
        sigma2_p=None,
        sigma2_scale_factors=None,
        convert_sigma_to_internal_units=False,
    ):
        self.sigma_power = sigma_power
        if sigma_power is None:
            sigma_power = 2

        if convert_sigma_to_internal_units:
            scale_factors = runtime.scale_factors
            is_dense_gene_set = runtime.is_dense_gene_set
            if scale_factors is not None:
                if is_dense_gene_set is not None and np.sum(~is_dense_gene_set) > 0:
                    self.sigma2 = sigma2 / np.mean(
                        np.power(scale_factors[~is_dense_gene_set], self.sigma_power - 2)
                    )
                else:
                    self.sigma2 = sigma2 / np.mean(
                        np.power(scale_factors, self.sigma_power - 2)
                    )
            else:
                self.sigma2 = sigma2 / np.power(runtime.MEAN_MOUSE_SCALE, self.sigma_power - 2)
        else:
            self.sigma2 = sigma2

        if sigma2_osc is not None:
            self.sigma2_osc = sigma2_osc

        if sigma2_scale_factors is None:
            sigma2_scale_factors = runtime.scale_factors

        if sigma2_se is not None:
            self.sigma2_se = sigma2_se
        if self.sigma2_p is not None:
            self.sigma2_p = sigma2_p

        if self.sigma2 is None and self.sigma2_osc is None:
            return self

        sigma2_for_var = self.sigma2_osc if self.sigma2_osc is not None else self.sigma2
        if sigma2_for_var is not None and sigma2_scale_factors is not None:
            if self.sigma_power is None:
                self.sigma2_total_var = sigma2_for_var * len(sigma2_scale_factors)
            else:
                self.sigma2_total_var = sigma2_for_var * np.sum(np.square(sigma2_scale_factors))

        if self.sigma2_total_var is not None and self.sigma2_se is not None:
            self.sigma2_total_var_lower = self.sigma2_total_var * (
                sigma2_for_var - 1.96 * self.sigma2_se
            ) / sigma2_for_var
            self.sigma2_total_var_upper = self.sigma2_total_var * (
                sigma2_for_var + 1.96 * self.sigma2_se
            ) / sigma2_for_var
        return self


@dataclass
class PhewasRuntimeState:
    phenos: object = None
    pheno_to_ind: object = None
    gene_pheno_Y: object = None
    gene_pheno_combined_prior_Ys: object = None
    gene_pheno_priors: object = None
    X_phewas_beta: object = None
    X_phewas_beta_uncorrected: object = None
    num_gene_phewas_filtered: int = 0
    anchor_gene_mask: object = None
    anchor_pheno_mask: object = None


@dataclass
class FactorInputData:
    anchor_gene_mask: object = None
    anchor_pheno_mask: object = None
    loaded_gene_set_phewas_stats: bool = False
    loaded_gene_phewas_bfs: bool = False


@dataclass
class PhewasStageConfig:
    gene_phewas_bfs_in: object = None
    gene_phewas_bfs_id_col: object = None
    gene_phewas_bfs_pheno_col: object = None
    gene_phewas_bfs_log_bf_col: object = None
    gene_phewas_bfs_combined_col: object = None
    gene_phewas_bfs_prior_col: object = None
    max_num_burn_in: int = 1000
    max_num_iter: int = 1100
    min_num_iter: int = 10
    num_chains: int = 10
    r_threshold_burn_in: float = 1.01
    use_max_r_for_convergence: bool = True
    max_frac_sem: float = 0.01
    gauss_seidel: bool = False
    sparse_solution: bool = False
    sparse_frac_betas: object = None
    run_for_factors: bool = False
    batch_size: int | None = None
    min_gene_factor_weight: float = 0.0

    def to_run_kwargs(self):
        run_kwargs = {
            "gene_phewas_bfs_in": self.gene_phewas_bfs_in,
            "gene_phewas_bfs_id_col": self.gene_phewas_bfs_id_col,
            "gene_phewas_bfs_pheno_col": self.gene_phewas_bfs_pheno_col,
            "gene_phewas_bfs_log_bf_col": self.gene_phewas_bfs_log_bf_col,
            "gene_phewas_bfs_combined_col": self.gene_phewas_bfs_combined_col,
            "gene_phewas_bfs_prior_col": self.gene_phewas_bfs_prior_col,
            "max_num_burn_in": self.max_num_burn_in,
            "max_num_iter": self.max_num_iter,
            "min_num_iter": self.min_num_iter,
            "num_chains": self.num_chains,
            "r_threshold_burn_in": self.r_threshold_burn_in,
            "use_max_r_for_convergence": self.use_max_r_for_convergence,
            "max_frac_sem": self.max_frac_sem,
            "gauss_seidel": self.gauss_seidel,
            "sparse_solution": self.sparse_solution,
            "sparse_frac_betas": self.sparse_frac_betas,
        }
        if self.run_for_factors:
            run_kwargs["run_for_factors"] = True
            if self.batch_size is not None:
                run_kwargs["batch_size"] = self.batch_size
            run_kwargs["min_gene_factor_weight"] = self.min_gene_factor_weight
        return run_kwargs


def _default_bail(message):
    raise ValueError(message)


def merge_dicts(base_value, override_value):
    if not isinstance(base_value, dict):
        base_value = {}
    merged = dict(base_value)
    for key, value in override_value.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = merge_dicts(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_json_config(config_path, bail_fn=None, seen_paths=None):
    if bail_fn is None:
        bail_fn = _default_bail
    if seen_paths is None:
        seen_paths = set()

    abs_path = os.path.abspath(config_path)
    if abs_path in seen_paths:
        bail_fn("Detected circular config include at %s" % abs_path)
    seen_paths.add(abs_path)

    with open(abs_path) as cfg_fh:
        cfg = json.load(cfg_fh)

    if not isinstance(cfg, dict):
        bail_fn("Config file must contain a JSON object: %s" % abs_path)

    includes = cfg.get("include")
    if includes is None:
        return cfg

    include_list = includes if isinstance(includes, list) else [includes]
    merged = {}
    cfg_dir = os.path.dirname(abs_path)
    for include_file in include_list:
        if not isinstance(include_file, str):
            bail_fn("Config include entries must be strings in %s" % abs_path)
        include_path = include_file
        if not os.path.isabs(include_path):
            include_path = os.path.normpath(os.path.join(cfg_dir, include_path))
        include_cfg = load_json_config(include_path, bail_fn=bail_fn, seen_paths=seen_paths)
        merged = merge_dicts(merged, include_cfg)

    cfg = dict(cfg)
    del cfg["include"]
    return merge_dicts(merged, cfg)


def is_remote_path(value):
    if not isinstance(value, str):
        return False
    lower = value.lower()
    return lower.startswith("http:") or lower.startswith("https:") or lower.startswith("ftp:")


def resolve_config_path_value(value, config_dir):
    if not isinstance(value, str):
        return value
    if value == "":
        return value
    if is_remote_path(value):
        return value
    expanded = os.path.expanduser(value)
    if os.path.isabs(expanded):
        return os.path.normpath(expanded)
    return os.path.normpath(os.path.join(config_dir, expanded))


def is_path_like_dest(dest):
    if dest is None:
        return False
    dest_lower = dest.lower()
    return (
        dest_lower.endswith("_in")
        or dest_lower.endswith("_out")
        or dest_lower.endswith("_file")
        or "_file_" in dest_lower
        or dest_lower in ("log_file", "warnings_file", "config")
    )


def emit_stderr_warning(message):
    sys.stderr.write("Warning: %s\n" % message)
    sys.stderr.flush()


def callback_set_comma_separated_args(option, opt, value, parser):
    setattr(parser.values, option.dest, value.split(","))


def callback_set_comma_separated_args_as_float(option, opt, value, parser):
    setattr(parser.values, option.dest, [float(x) for x in value.split(",")])


def callback_set_comma_separated_args_as_set(option, opt, value, parser):
    setattr(parser.values, option.dest, set(value.split(",")))


def open_optional_log_handle(filepath, default_stream=None, mode="w"):
    if filepath is not None:
        return open(filepath, mode)
    if default_stream is not None:
        return default_stream
    return sys.stderr


def urlopen_with_retry(
    file,
    flag=None,
    tries=5,
    delay=60,
    backoff=2,
    *,
    log_fn=None,
    bail_fn=None,
):
    if bail_fn is None:
        bail_fn = _default_bail

    while tries > 1:
        try:
            if flag is not None:
                return urllib.request.urlopen(file, flag)
            return urllib.request.urlopen(file)
        except urllib.error.URLError as e:
            if log_fn is not None:
                log_fn("%s, Retrying in %d seconds..." % (str(e), delay))
            time.sleep(delay)
            tries -= 1
            delay *= backoff
    bail_fn("Couldn't open file after too many retries")


def is_dig_open_data_uri(filepath):
    return isinstance(filepath, str) and filepath.startswith(DIG_OPEN_DATA_PREFIX)


def is_dig_open_data_ancestry_trait_spec(spec):
    if not isinstance(spec, str):
        return False
    if spec.count(":") != 1:
        return False
    ancestry, trait = spec.split(":", 1)
    if len(ancestry) == 0 or len(trait) == 0:
        return False
    if "/" in ancestry or "/" in trait:
        return False
    if not DIG_OPEN_DATA_TOKEN_RE.match(ancestry):
        return False
    if not DIG_OPEN_DATA_TOKEN_RE.match(trait):
        return False
    return True


def open_dig_open_data(uri, flag=None, *, log_fn=None, bail_fn=None):
    if bail_fn is None:
        bail_fn = _default_bail

    if flag is not None and "w" in flag:
        bail_fn("dig-open-data sources are read-only and cannot be opened for writing")

    spec = uri[len(DIG_OPEN_DATA_PREFIX):]
    if len(spec.strip()) == 0:
        bail_fn("Invalid dig-open-data source '%s'; expected dig-open-data:<ancestry>:<trait>" % uri)

    try:
        from dig_open_data import open_text, open_trait
    except ImportError:
        bail_fn("dig_open_data is required to read '%s'. Install https://github.com/flannick/dig-open-data/" % uri)

    if is_dig_open_data_ancestry_trait_spec(spec):
        ancestry, trait = spec.split(":", 1)
        if log_fn is not None:
            log_fn("Reading dig-open-data trait ancestry=%s trait=%s" % (ancestry, trait))
        return open_trait(ancestry, trait)

    if log_fn is not None:
        log_fn("Reading dig-open-data source %s" % spec)
    return open_text(spec)


def is_gz_file(filepath, is_remote, flag=None, *, urlopen_with_retry_fn=None):
    open_url_fn = urlopen_with_retry if urlopen_with_retry_fn is None else urlopen_with_retry_fn

    if len(filepath) >= 3 and (filepath[-3:] == ".gz" or filepath[-4:] == ".bgz") and (flag is None or "w" not in flag):
        try:
            if is_remote:
                test_fh = open_url_fn(filepath)
            else:
                test_fh = gzip.open(filepath, "rb")

            try:
                test_fh.readline()
                test_fh.close()
                return True
            except Exception:
                return False
        except FileNotFoundError:
            return True

    elif flag is None or "w" not in flag:
        test_flag = "rb"
        if is_remote:
            test_fh = open_url_fn(filepath, test_flag)
        else:
            test_fh = open(filepath, test_flag)

        gz_magic = test_fh.read(2) == b"\x1f\x8b"
        test_fh.close()
        return gz_magic

    return filepath[-3:] == ".gz" or filepath[-4:] == ".bgz"


def open_text_auto(
    file,
    flag=None,
    *,
    log_fn=None,
    bail_fn=None,
    urlopen_with_retry_fn=None,
    is_gz_file_fn=None,
):
    if bail_fn is None:
        bail_fn = _default_bail
    open_url_fn = urlopen_with_retry if urlopen_with_retry_fn is None else urlopen_with_retry_fn
    detect_gz_fn = is_gz_file if is_gz_file_fn is None else is_gz_file_fn

    if is_dig_open_data_uri(file):
        return open_dig_open_data(file, flag=flag, log_fn=log_fn, bail_fn=bail_fn)

    is_remote = is_remote_path(file)

    try:
        is_gz = detect_gz_fn(
            file,
            is_remote,
            flag=flag,
            urlopen_with_retry_fn=open_url_fn,
        )
    except TypeError:
        # Backward-compatible path for legacy wrapper call signatures.
        is_gz = detect_gz_fn(file, is_remote, flag=flag)

    if is_gz:
        open_fun = gzip.open
        if flag is not None and len(flag) > 0 and not flag.endswith("t"):
            flag = "%st" % flag
        elif flag is None:
            flag = "rt"
    else:
        open_fun = open

    if is_remote:
        if flag is not None:
            if open_fun is open:
                fh = io.TextIOWrapper(open_url_fn(file, flag))
            else:
                fh = open_fun(open_url_fn(file), flag)
        else:
            if open_fun is open:
                fh = io.TextIOWrapper(open_url_fn(file))
            else:
                fh = open_fun(open_url_fn(file))
    else:
        if flag is not None:
            try:
                fh = open_fun(file, flag, encoding="utf-8")
            except LookupError:
                fh = open_fun(file, flag)
        else:
            try:
                fh = open_fun(file, encoding="utf-8")
            except LookupError:
                fh = open_fun(file)

    return fh


def open_text_with_retry(filepath, flag=None, *, log_fn=None, bail_fn=None):
    if bail_fn is None:
        bail_fn = _default_bail
    if log_fn is None:
        log_fn = lambda _msg: None

    open_url_with_retry_fn = lambda _file, _flag=None: urlopen_with_retry(
        _file,
        flag=_flag,
        log_fn=log_fn,
        bail_fn=bail_fn,
    )
    return open_text_auto(
        filepath,
        flag=flag,
        log_fn=log_fn,
        bail_fn=bail_fn,
        urlopen_with_retry_fn=open_url_with_retry_fn,
    )


class TsvTable(object):
    def __init__(self, columns, rows, key_column=None, by_key=None):
        self.columns = columns
        self.rows = rows
        self.key_column = key_column
        self.by_key = by_key


class GeneStatsTable(TsvTable):
    KEY_COLUMN = "Gene"
    REQUIRED_COLUMNS = ["Gene"]

    @classmethod
    def read(cls, path, *, bail_fn=None):
        table = read_tsv(
            path,
            key_column=cls.KEY_COLUMN,
            required_columns=cls.REQUIRED_COLUMNS,
            bail_fn=bail_fn,
        )
        return cls(
            columns=table.columns,
            rows=table.rows,
            key_column=table.key_column,
            by_key=table.by_key,
        )


class GeneSetStatsTable(TsvTable):
    KEY_COLUMN = "Gene_Set"
    REQUIRED_COLUMNS = ["Gene_Set"]

    @classmethod
    def read(cls, path, *, bail_fn=None):
        table = read_tsv(
            path,
            key_column=cls.KEY_COLUMN,
            required_columns=cls.REQUIRED_COLUMNS,
            bail_fn=bail_fn,
        )
        return cls(
            columns=table.columns,
            rows=table.rows,
            key_column=table.key_column,
            by_key=table.by_key,
        )


def read_tsv(path, key_column=None, required_columns=None, *, bail_fn=None):
    if bail_fn is None:
        bail_fn = _default_bail
    required = set(required_columns or [])

    with open_text_auto(str(path), "rt", bail_fn=bail_fn) as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        if reader.fieldnames is None:
            bail_fn("No header found in TSV: %s" % path)

        missing = required.difference(reader.fieldnames)
        if missing:
            missing_fmt = ", ".join(sorted(missing))
            bail_fn("Missing required columns (%s) in %s" % (missing_fmt, path))

        rows = []
        by_key = {} if key_column else None
        for row in reader:
            rows.append(row)
            if key_column:
                key = row.get(key_column, "")
                if key in by_key:
                    bail_fn("Duplicate key '%s' in %s (%s)" % (key, path, key_column))
                by_key[key] = row

    return TsvTable(columns=list(reader.fieldnames), rows=rows, key_column=key_column, by_key=by_key)


def write_tsv(path, columns, rows):
    path = str(path)
    ensure_parent_dir_for_file(path)
    cols = list(columns)
    if path.endswith(".gz"):
        with gzip.open(path, "wt", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=cols, delimiter="\t", extrasaction="ignore")
            writer.writeheader()
            for row in rows:
                writer.writerow(row)
    else:
        with open(path, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=cols, delimiter="\t", extrasaction="ignore")
            writer.writeheader()
            for row in rows:
                writer.writerow(row)


def parse_gene_set_statistics_file(
    stats_in,
    *,
    stats_id_col,
    stats_exp_beta_tilde_col,
    stats_beta_tilde_col,
    stats_p_col,
    stats_se_col,
    stats_beta_col,
    stats_beta_uncorrected_col,
    ignore_negative_exp_beta,
    max_gene_set_p,
    min_gene_set_beta,
    min_gene_set_beta_uncorrected,
    open_text_fn,
    get_col_fn,
    log_fn=None,
    warn_fn=None,
    bail_fn=None,
):
    if bail_fn is None:
        bail_fn = _default_bail
    if log_fn is None:
        log_fn = lambda _msg: None
    if warn_fn is None:
        warn_fn = lambda _msg: None

    if stats_in is None:
        bail_fn("Require --stats-in for this operation")

    log_fn("Reading --stats-in file %s" % stats_in)
    need_to_take_log = False
    records = {}

    with open_text_fn(stats_in) as stats_fh:
        header_cols = stats_fh.readline().strip("\n").split()
        id_col = get_col_fn(stats_id_col, header_cols)
        beta_tilde_col = None

        if stats_beta_tilde_col is not None:
            beta_tilde_col = get_col_fn(stats_beta_tilde_col, header_cols, False)
        if beta_tilde_col is not None:
            log_fn("Using col %s for beta_tilde values" % stats_beta_tilde_col)
        elif stats_exp_beta_tilde_col is not None:
            beta_tilde_col = get_col_fn(stats_exp_beta_tilde_col, header_cols)
            need_to_take_log = True
            if beta_tilde_col is not None:
                log_fn("Using %s for exp(beta_tilde) values" % stats_exp_beta_tilde_col)
            else:
                bail_fn(
                    "Could not find beta_tilde column %s or %s in header: %s"
                    % (stats_beta_tilde_col, stats_exp_beta_tilde_col, "\t".join(header_cols))
                )

        p_col = None
        if stats_p_col is not None:
            p_col = get_col_fn(stats_p_col, header_cols, False)

        se_col = None
        if stats_se_col is not None:
            se_col = get_col_fn(stats_se_col, header_cols, False)

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

        if (
            se_col is None
            and p_col is None
            and beta_tilde_col is None
            and beta_col is None
            and beta_uncorrected_col is None
        ):
            bail_fn("Require at least something to read from --gene-set-stats-in")

        for line in stats_fh:
            beta_tilde = None
            p = None
            se = None
            z = None
            beta = None
            beta_uncorrected = None

            cols = line.strip("\n").split()
            if (
                id_col > len(cols)
                or (beta_tilde_col is not None and beta_tilde_col > len(cols))
                or (p_col is not None and p_col > len(cols))
                or (se_col is not None and se_col > len(cols))
            ):
                warn_fn("Skipping due to too few columns in line: %s" % line)
                continue

            gene_set = cols[id_col]
            if gene_set in records:
                warn_fn("Already seen gene set %s; only considering first instance" % (gene_set))
                continue

            if beta_tilde_col is not None:
                try:
                    beta_tilde = float(cols[beta_tilde_col])
                except ValueError:
                    if cols[beta_tilde_col] != "NA":
                        warn_fn(
                            "Skipping unconvertible beta_tilde value %s for gene_set %s"
                            % (cols[beta_tilde_col], gene_set)
                        )
                    continue

                if need_to_take_log:
                    if beta_tilde < 0:
                        if ignore_negative_exp_beta:
                            continue
                        bail_fn(
                            "Exp(beta) value %s for gene set %s is < 0; did you mean to specify --stats-beta-col? Otherwise, specify --ignore-negative-exp-beta to ignore these"
                            % (beta_tilde, gene_set)
                        )
                    beta_tilde = np.log(beta_tilde)

            if se_col is not None:
                try:
                    se = float(cols[se_col])
                except ValueError:
                    if cols[se_col] != "NA":
                        warn_fn(
                            "Skipping unconvertible se value %s for gene_set %s"
                            % (cols[se_col], gene_set)
                        )
                    continue

                if beta_tilde_col is not None:
                    z = beta_tilde / se
                    p = 2 * scipy.stats.norm.cdf(-np.abs(z))
                    if max_gene_set_p is not None and p > max_gene_set_p:
                        continue
            elif p_col is not None:
                try:
                    p = float(cols[p_col])
                    if max_gene_set_p is not None and p > max_gene_set_p:
                        continue
                except ValueError:
                    if cols[p_col] != "NA":
                        warn_fn(
                            "Skipping unconvertible p value %s for gene_set %s"
                            % (cols[p_col], gene_set)
                        )
                    continue

                z = np.abs(scipy.stats.norm.ppf(p / 2))
                if z == 0:
                    warn_fn("Skipping gene_set %s due to 0 z-score" % (gene_set))
                    continue

                if beta_tilde_col is not None:
                    se = np.abs(beta_tilde) / z

            if beta_col is not None:
                try:
                    beta = float(cols[beta_col])
                    if min_gene_set_beta is not None and beta < min_gene_set_beta:
                        continue
                except ValueError:
                    if cols[beta_col] != "NA":
                        warn_fn(
                            "Skipping unconvertible beta value %s for gene_set %s"
                            % (cols[beta_col], gene_set)
                        )
                    continue

            if beta_uncorrected_col is not None:
                try:
                    beta_uncorrected = float(cols[beta_uncorrected_col])
                    if (
                        min_gene_set_beta_uncorrected is not None
                        and beta_uncorrected < min_gene_set_beta_uncorrected
                    ):
                        continue
                except ValueError:
                    if cols[beta_uncorrected_col] != "NA":
                        warn_fn(
                            "Skipping unconvertible beta_uncorrected value %s for gene_set %s"
                            % (cols[beta_uncorrected_col], gene_set)
                        )
                    continue

            records[gene_set] = (
                beta_tilde,
                p,
                se,
                z,
                beta,
                beta_uncorrected,
            )

    return ParsedGeneSetStats(
        need_to_take_log=need_to_take_log,
        has_beta_tilde=beta_tilde_col is not None,
        has_p_or_se=(p_col is not None or se_col is not None),
        has_beta=beta_col is not None,
        has_beta_uncorrected=beta_uncorrected_col is not None,
        records=records,
    )


def parse_gene_bfs_file(
    gene_bfs_in,
    *,
    gene_bfs_id_col,
    gene_bfs_log_bf_col,
    gene_bfs_combined_col,
    gene_bfs_prob_col,
    gene_bfs_prior_col,
    background_log_bf,
    gene_label_map,
    open_text_fn,
    get_col_fn,
    log_fn=None,
    warn_fn=None,
    bail_fn=None,
):
    if bail_fn is None:
        bail_fn = _default_bail
    if log_fn is None:
        log_fn = lambda _msg: None
    if warn_fn is None:
        warn_fn = lambda _msg: None

    if gene_bfs_in is None:
        bail_fn("Require --gene-stats-in for this operation")

    log_fn("Reading --gene-stats-in file %s" % gene_bfs_in)
    gene_in_bfs = {}
    gene_in_combined = None
    gene_in_priors = None
    with open_text_fn(gene_bfs_in) as gene_bfs_fh:
        header_cols = gene_bfs_fh.readline().strip("\n").split()
        if gene_bfs_id_col is None:
            gene_bfs_id_col = "Gene"

        id_col = get_col_fn(gene_bfs_id_col, header_cols)

        prob_col = None
        if gene_bfs_prob_col is not None:
            prob_col = get_col_fn(gene_bfs_prob_col, header_cols, True)

        bf_col = None
        if gene_bfs_log_bf_col is not None:
            bf_col = get_col_fn(gene_bfs_log_bf_col, header_cols)
        else:
            if prob_col is None:
                bf_col = get_col_fn("log_bf", header_cols)

        if bf_col is None and prob_col is None:
            bail_fn("--gene-stats-log-bf-col or --gene-stats-prob-col required for this operation")

        combined_col = None
        if gene_bfs_combined_col is not None:
            combined_col = get_col_fn(gene_bfs_combined_col, header_cols, True)
        else:
            combined_col = get_col_fn("combined", header_cols, False)

        prior_col = None
        if gene_bfs_prior_col is not None:
            prior_col = get_col_fn(gene_bfs_prior_col, header_cols, True)
        else:
            prior_col = get_col_fn("prior", header_cols, False)

        if combined_col is not None or prob_col is not None:
            gene_in_combined = {}
        if prior_col is not None:
            gene_in_priors = {}

        for line in gene_bfs_fh:
            cols = line.strip("\n").split()
            if (
                id_col >= len(cols)
                or (bf_col is not None and bf_col >= len(cols))
                or (combined_col is not None and combined_col >= len(cols))
                or (prob_col is not None and prob_col >= len(cols))
                or (prior_col is not None and prior_col >= len(cols))
            ):
                warn_fn("Skipping due to too few columns in line: %s" % line)
                continue

            gene = cols[id_col]
            if gene_label_map is not None and gene in gene_label_map:
                gene = gene_label_map[gene]

            if bf_col is not None:
                try:
                    bf = float(cols[bf_col])
                except ValueError:
                    if cols[bf_col] != "NA":
                        warn_fn("Skipping unconvertible bf value %s for gene %s" % (cols[bf_col], gene))
                    continue
            elif prob_col is not None:
                try:
                    prob = float(cols[prob_col])
                except ValueError:
                    if cols[prob_col] != "NA":
                        warn_fn(
                            "Skipping unconvertible prob value %s for gene %s"
                            % (cols[prob_col], gene)
                        )
                    continue
                if prob <= 0 or prob >= 1:
                    warn_fn("Skipping probability %.3g outside of (0,1)" % (prob))
                    continue
                bf = np.log(prob / (1 - prob)) - background_log_bf

            gene_in_bfs[gene] = bf

            if combined_col is not None:
                try:
                    combined = float(cols[combined_col])
                except ValueError:
                    if cols[combined_col] != "NA":
                        warn_fn(
                            "Skipping unconvertible combined value %s for gene %s"
                            % (cols[combined_col], gene)
                        )
                    continue
                gene_in_combined[gene] = combined

            if prior_col is not None:
                try:
                    prior = float(cols[prior_col])
                except ValueError:
                    if cols[prior_col] != "NA":
                        warn_fn(
                            "Skipping unconvertible prior value %s for gene %s"
                            % (cols[prior_col], gene)
                        )
                    continue
                gene_in_priors[gene] = prior

    return ParsedGeneBfs(
        gene_in_bfs=gene_in_bfs,
        gene_in_combined=gene_in_combined,
        gene_in_priors=gene_in_priors,
    )


def parse_gene_covariates_file(
    gene_covs_in,
    *,
    gene_covs_id_col=None,
    open_text_fn=None,
    get_col_fn=None,
    log_fn=None,
    warn_fn=None,
    bail_fn=None,
):
    if bail_fn is None:
        bail_fn = _default_bail
    if log_fn is None:
        log_fn = lambda _m: None
    if warn_fn is None:
        warn_fn = lambda _m: None
    if open_text_fn is None:
        open_text_fn = lambda path: open(path)
    if get_col_fn is None:
        get_col_fn = resolve_column_index

    log_fn("Reading --gene-covs-in file %s" % gene_covs_in)

    gene_in_covs = {}
    cov_names = []
    with open_text_fn(gene_covs_in) as gene_covs_fh:
        header_cols = gene_covs_fh.readline().strip('\n').split()
        if gene_covs_id_col is None:
            gene_covs_id_col = "Gene"

        id_col = get_col_fn(gene_covs_id_col, header_cols)
        cov_names = [header_cols[i] for i in range(len(header_cols)) if i != id_col]

        if len(cov_names) > 0:
            log_fn("Read covariates %s" % (",".join(cov_names)))
            for line in gene_covs_fh:
                cols = line.strip('\n').split()
                if len(cols) != len(header_cols):
                    warn_fn("Skipping due to too few columns in line: %s" % line)
                    continue

                gene = cols[id_col]
                try:
                    covs = np.array([float(cols[i]) for i in range(len(cols)) if i != id_col])
                except ValueError:
                    continue

                gene_in_covs[gene] = covs

    return ParsedGeneCovariates(cov_names=cov_names, gene_to_covs=gene_in_covs)


def align_gene_scalar_map(gene_to_value, genes=None, gene_to_ind=None, missing_value=np.nan):
    if genes is None:
        genes = []
    if gene_to_ind is None:
        gene_to_ind = {}

    aligned_values = np.array([missing_value] * len(genes))
    extra_values = []
    extra_genes = []
    for gene, value in gene_to_value.items():
        if gene in gene_to_ind:
            aligned_values[gene_to_ind[gene]] = value
        else:
            extra_values.append(value)
            extra_genes.append(gene)

    return (aligned_values, extra_genes, np.array(extra_values))


def align_gene_vector_map(gene_to_values, num_values, genes=None, gene_to_ind=None, missing_value=np.nan):
    if genes is None:
        genes = []
    if gene_to_ind is None:
        gene_to_ind = {}

    aligned_values = np.full((len(genes), num_values), missing_value)
    extra_values = []
    extra_genes = []
    for gene, values in gene_to_values.items():
        if gene in gene_to_ind:
            aligned_values[gene_to_ind[gene], :] = values
        else:
            extra_values.append(values)
            extra_genes.append(gene)

    return (aligned_values, extra_genes, np.array(extra_values))


def load_aligned_gene_bfs(
    gene_bfs_in,
    *,
    genes=None,
    gene_to_ind=None,
    gene_bfs_id_col=None,
    gene_bfs_log_bf_col=None,
    gene_bfs_combined_col=None,
    gene_bfs_prob_col=None,
    gene_bfs_prior_col=None,
    background_log_bf=0.0,
    gene_label_map=None,
    open_text_fn=None,
    get_col_fn=None,
    log_fn=None,
    warn_fn=None,
    bail_fn=None,
):
    parsed_gene_bfs = parse_gene_bfs_file(
        gene_bfs_in,
        gene_bfs_id_col=gene_bfs_id_col,
        gene_bfs_log_bf_col=gene_bfs_log_bf_col,
        gene_bfs_combined_col=gene_bfs_combined_col,
        gene_bfs_prob_col=gene_bfs_prob_col,
        gene_bfs_prior_col=gene_bfs_prior_col,
        background_log_bf=background_log_bf,
        gene_label_map=gene_label_map,
        open_text_fn=open_text_fn,
        get_col_fn=get_col_fn,
        log_fn=log_fn,
        warn_fn=warn_fn,
        bail_fn=bail_fn,
    )
    gene_bfs, extra_genes, extra_gene_bfs = align_gene_scalar_map(
        parsed_gene_bfs.gene_in_bfs,
        genes=genes,
        gene_to_ind=gene_to_ind,
    )
    return AlignedGeneBfs(
        gene_bfs=gene_bfs,
        extra_genes=extra_genes,
        extra_gene_bfs=extra_gene_bfs,
        gene_in_combined=parsed_gene_bfs.gene_in_combined,
        gene_in_priors=parsed_gene_bfs.gene_in_priors,
    )


def load_aligned_gene_covariates(
    gene_covs_in,
    *,
    genes=None,
    gene_to_ind=None,
    gene_covs_id_col=None,
    open_text_fn=None,
    get_col_fn=None,
    log_fn=None,
    warn_fn=None,
    bail_fn=None,
):
    parsed_gene_covs = parse_gene_covariates_file(
        gene_covs_in,
        gene_covs_id_col=gene_covs_id_col,
        open_text_fn=open_text_fn,
        get_col_fn=get_col_fn,
        log_fn=log_fn,
        warn_fn=warn_fn,
        bail_fn=bail_fn,
    )
    gene_covs, extra_genes, extra_gene_covs = align_gene_vector_map(
        parsed_gene_covs.gene_to_covs,
        num_values=len(parsed_gene_covs.cov_names),
        genes=genes,
        gene_to_ind=gene_to_ind,
    )
    return AlignedGeneCovariates(
        cov_names=parsed_gene_covs.cov_names,
        gene_covs=gene_covs,
        extra_genes=extra_genes,
        extra_gene_covs=extra_gene_covs,
    )


def parse_gene_phewas_bfs_file(
    gene_phewas_bfs_in,
    *,
    gene_phewas_bfs_id_col,
    gene_phewas_bfs_pheno_col,
    gene_phewas_bfs_log_bf_col,
    gene_phewas_bfs_combined_col,
    gene_phewas_bfs_prior_col,
    min_value,
    max_num_entries_at_once,
    existing_phenos,
    existing_pheno_to_ind,
    gene_to_ind,
    gene_label_map,
    phewas_gene_to_x_gene,
    open_text_fn,
    get_col_fn,
    bail_fn=None,
    warn_fn=None,
):
    if bail_fn is None:
        bail_fn = _default_bail
    if warn_fn is None:
        warn_fn = lambda _msg: None

    if max_num_entries_at_once is None:
        max_num_entries_at_once = 200 * 10000

    success = False
    num_filtered = 0
    final_phenos = list(existing_phenos) if existing_phenos is not None else []
    final_pheno_to_ind = copy.copy(existing_pheno_to_ind) if existing_pheno_to_ind is not None else {}
    final_row = np.array([], dtype=np.int32)
    final_col = np.array([], dtype=np.int32)
    final_Ys = None
    final_combineds = None
    final_priors = None

    for delim in [None, "\t"]:
        success = True
        Ys = None
        combineds = None
        priors = None

        row = []
        col = []
        row_chunks = []
        col_chunks = []
        Y_chunks = []
        combined_chunks = []
        prior_chunks = []

        with open_text_fn(gene_phewas_bfs_in) as gene_phewas_bfs_fh:
            header_cols = gene_phewas_bfs_fh.readline().strip("\n").split(delim)
            id_col_name = gene_phewas_bfs_id_col if gene_phewas_bfs_id_col is not None else "Gene"
            pheno_col_name = gene_phewas_bfs_pheno_col if gene_phewas_bfs_pheno_col is not None else "Pheno"

            id_col = get_col_fn(id_col_name, header_cols)
            pheno_col = get_col_fn(pheno_col_name, header_cols)

            if gene_phewas_bfs_log_bf_col is not None:
                bf_col = get_col_fn(gene_phewas_bfs_log_bf_col, header_cols)
            else:
                bf_col = get_col_fn("log_bf", header_cols, False)

            if gene_phewas_bfs_combined_col is not None:
                combined_col = get_col_fn(gene_phewas_bfs_combined_col, header_cols, True)
            else:
                combined_col = get_col_fn("combined", header_cols, False)

            if gene_phewas_bfs_prior_col is not None:
                prior_col = get_col_fn(gene_phewas_bfs_prior_col, header_cols, True)
            else:
                prior_col = get_col_fn("prior", header_cols, False)

            if bf_col is not None:
                Ys = []
            if combined_col is not None:
                combineds = []
            if prior_col is not None:
                priors = []

            def _flush_chunks():
                if len(row) == 0:
                    return
                row_chunks.append(np.array(row, dtype=np.int32))
                col_chunks.append(np.array(col, dtype=np.int32))
                if Ys is not None:
                    Y_chunks.append(np.array(Ys, dtype=np.float64))
                    Ys[:] = []
                if combineds is not None:
                    combined_chunks.append(np.array(combineds, dtype=np.float64))
                    combineds[:] = []
                if priors is not None:
                    prior_chunks.append(np.array(priors, dtype=np.float64))
                    priors[:] = []
                row[:] = []
                col[:] = []

            phenos = list(existing_phenos) if existing_phenos is not None else []
            pheno_to_ind = (
                copy.copy(existing_pheno_to_ind) if existing_pheno_to_ind is not None else {}
            )
            num_filtered = 0

            for line in gene_phewas_bfs_fh:
                cols = line.strip("\n").split(delim)
                if len(cols) != len(header_cols):
                    success = False
                    continue

                if (
                    id_col >= len(cols)
                    or pheno_col >= len(cols)
                    or (bf_col is not None and bf_col >= len(cols))
                    or (combined_col is not None and combined_col >= len(cols))
                    or (prior_col is not None and prior_col >= len(cols))
                ):
                    warn_fn("Skipping due to too few columns in line: %s" % line)
                    continue

                gene = cols[id_col]
                pheno = cols[pheno_col]

                cur_combined = None
                if combined_col is not None:
                    try:
                        combined = float(cols[combined_col])
                    except ValueError:
                        if cols[combined_col] != "NA":
                            warn_fn(
                                "Skipping unconvertible value %s for gene_set %s"
                                % (cols[combined_col], gene)
                            )
                        continue

                    if min_value is not None and combined < min_value:
                        num_filtered += 1
                        continue
                    cur_combined = combined

                cur_Y = None
                if bf_col is not None:
                    try:
                        bf = float(cols[bf_col])
                    except ValueError:
                        if cols[bf_col] != "NA":
                            warn_fn(
                                "Skipping unconvertible value %s for gene %s and pheno %s"
                                % (cols[bf_col], gene, pheno)
                            )
                        continue

                    if min_value is not None and combined_col is None and bf < min_value:
                        num_filtered += 1
                        continue
                    cur_Y = bf

                cur_prior = None
                if prior_col is not None:
                    try:
                        prior = float(cols[prior_col])
                    except ValueError:
                        if cols[prior_col] != "NA":
                            warn_fn(
                                "Skipping unconvertible value %s for gene %s"
                                % (cols[prior_col], gene)
                            )
                        continue

                    if min_value is not None and combined_col is None and bf_col is None and prior < min_value:
                        num_filtered += 1
                        continue
                    cur_prior = prior

                if pheno not in pheno_to_ind:
                    pheno_to_ind[pheno] = len(phenos)
                    phenos.append(pheno)
                pheno_ind = pheno_to_ind[pheno]

                if gene_label_map is not None and gene in gene_label_map:
                    gene = gene_label_map[gene]

                mapped_genes = [gene]
                if phewas_gene_to_x_gene is not None and gene in phewas_gene_to_x_gene:
                    mapped_genes = list(phewas_gene_to_x_gene[gene])

                for cur_gene in mapped_genes:
                    if cur_gene not in gene_to_ind:
                        continue
                    if combineds is not None:
                        combineds.append(cur_combined)
                    if Ys is not None:
                        Ys.append(cur_Y)
                    if priors is not None:
                        priors.append(cur_prior)

                    col.append(pheno_ind)
                    row.append(gene_to_ind[cur_gene])
                    if len(row) >= max_num_entries_at_once:
                        _flush_chunks()

            _flush_chunks()

        if success:
            final_phenos = phenos
            final_pheno_to_ind = pheno_to_ind
            if len(row_chunks) > 0:
                row = np.concatenate(row_chunks)
                col = np.concatenate(col_chunks)
            else:
                row = np.array([], dtype=np.int32)
                col = np.array([], dtype=np.int32)

            if len(row) > 0:
                key = row.astype(np.int64) * int(len(phenos)) + col.astype(np.int64)
                _, unique_indices = np.unique(key, return_index=True)
            else:
                unique_indices = np.array([], dtype=np.int64)

            if len(unique_indices) < len(row):
                warn_fn("Found %d duplicate values; ignoring duplicates" % (len(row) - len(unique_indices)))

            final_row = row[unique_indices]
            final_col = col[unique_indices]

            if combineds is not None:
                if len(combined_chunks) > 0:
                    final_combineds = np.concatenate(combined_chunks)[unique_indices]
                else:
                    final_combineds = np.array([], dtype=np.float64)
            else:
                final_combineds = None

            if Ys is not None:
                if len(Y_chunks) > 0:
                    final_Ys = np.concatenate(Y_chunks)[unique_indices]
                else:
                    final_Ys = np.array([], dtype=np.float64)
            else:
                final_Ys = None

            if priors is not None:
                if len(prior_chunks) > 0:
                    final_priors = np.concatenate(prior_chunks)[unique_indices]
                else:
                    final_priors = np.array([], dtype=np.float64)
            else:
                final_priors = None
            break

    if not success:
        bail_fn("Error: different number of columns in header row and non header rows")

    return ParsedGenePhewasBfs(
        phenos=final_phenos,
        pheno_to_ind=final_pheno_to_ind,
        row=final_row,
        col=final_col,
        Ys=final_Ys,
        combineds=final_combineds,
        priors=final_priors,
        num_filtered=num_filtered,
    )


def y_data_from_runtime(runtime):
    return YData(
        Y=getattr(runtime, "Y", None),
        Y_for_regression=getattr(runtime, "Y_for_regression", None),
        Y_exomes=getattr(runtime, "Y_exomes", None),
        Y_positive_controls=getattr(runtime, "Y_positive_controls", None),
        Y_case_counts=getattr(runtime, "Y_case_counts", None),
        y_var=getattr(runtime, "y_var", None),
        y_corr=getattr(runtime, "y_corr", None),
        y_corr_cholesky=getattr(runtime, "y_corr_cholesky", None),
        y_corr_sparse=getattr(runtime, "y_corr_sparse", None),
        Y_w=getattr(runtime, "Y_w", None),
        Y_fw=getattr(runtime, "Y_fw", None),
        y_w_var=getattr(runtime, "y_w_var", None),
        y_w_mean=getattr(runtime, "y_w_mean", None),
        y_fw_var=getattr(runtime, "y_fw_var", None),
        y_fw_mean=getattr(runtime, "y_fw_mean", None),
    )


def build_y_data_from_inputs(
    runtime,
    Y,
    Y_for_regression=None,
    Y_exomes=None,
    Y_positive_controls=None,
    Y_case_counts=None,
    Y_corr_m=None,
    store_cholesky=True,
    store_corr_sparse=False,
    skip_V=False,
    skip_scale_factors=False,
    min_correlation=0,
    get_y_corr_cholesky_fn=None,
    set_X_fn=None,
    calc_X_shift_scale_fn=None,
):
    y_data = y_data_from_runtime(runtime)
    if Y_corr_m is not None:
        y_corr_m = copy.copy(Y_corr_m)
        if min_correlation is not None:
            # preserve existing behavior: zero-out non-positive correlations
            y_corr_m[y_corr_m <= 0] = 0

        keep_mask = np.array([True] * len(y_corr_m))
        for i in range(len(y_corr_m) - 1, -1, -1):
            if np.sum(y_corr_m[i] != 0) == 0:
                keep_mask[i] = False
            else:
                break
        if np.sum(keep_mask) > 0:
            y_corr_m = y_corr_m[keep_mask]

        y_data.y_corr = copy.copy(y_corr_m)

        y_corr_diags = [y_data.y_corr[i, :(len(y_data.y_corr[i, :]) - i)] for i in range(len(y_data.y_corr))]
        y_corr_sparse = sparse.csc_matrix(
            sparse.diags(
                y_corr_diags + y_corr_diags[1:],
                list(range(len(y_corr_diags))) + list(range(-1, -len(y_corr_diags), -1)),
            )
        )

        if store_cholesky:
            if get_y_corr_cholesky_fn is None:
                _default_bail("Expected get_y_corr_cholesky_fn when store_cholesky is True")
            y_data.y_corr_cholesky = get_y_corr_cholesky_fn(y_corr_m)
            y_data.Y_w = scipy.linalg.solve_banded(
                (y_data.y_corr_cholesky.shape[0] - 1, 0), y_data.y_corr_cholesky, Y
            )
            na_mask = ~np.isnan(y_data.Y_w)
            y_data.y_w_var = np.var(y_data.Y_w[na_mask])
            y_data.y_w_mean = np.mean(y_data.Y_w[na_mask])
            y_data.Y_w = y_data.Y_w - y_data.y_w_mean

            y_data.Y_fw = scipy.linalg.cho_solve_banded((y_data.y_corr_cholesky, True), Y)
            na_mask = ~np.isnan(y_data.Y_fw)
            y_data.y_fw_var = np.var(y_data.Y_fw[na_mask])
            y_data.y_fw_mean = np.mean(y_data.Y_fw[na_mask])
            y_data.Y_fw = y_data.Y_fw - y_data.y_fw_mean

            if set_X_fn is not None:
                set_X_fn(runtime.X_orig, runtime.genes, runtime.gene_sets, skip_V=skip_V, skip_scale_factors=skip_scale_factors, skip_N=True)
            if (
                calc_X_shift_scale_fn is not None
                and runtime.X_orig_missing_gene_sets is not None
                and not skip_scale_factors
            ):
                runtime.mean_shifts_missing, runtime.scale_factors_missing = calc_X_shift_scale_fn(
                    runtime.X_orig_missing_gene_sets, y_data.y_corr_cholesky
                )

        if store_corr_sparse:
            y_data.y_corr_sparse = y_corr_sparse

    if Y is not None:
        na_mask = ~np.isnan(Y)
        y_data.y_var = np.var(Y[na_mask])
    else:
        y_data.y_var = None
    y_data.Y = Y
    y_data.Y_for_regression = Y_for_regression
    y_data.Y_exomes = Y_exomes
    y_data.Y_positive_controls = Y_positive_controls
    y_data.Y_case_counts = Y_case_counts
    return y_data


def set_runtime_y_from_inputs(
    runtime,
    Y,
    Y_for_regression=None,
    Y_exomes=None,
    Y_positive_controls=None,
    Y_case_counts=None,
    Y_corr_m=None,
    store_cholesky=True,
    store_corr_sparse=False,
    skip_V=False,
    skip_scale_factors=False,
    min_correlation=0,
    get_y_corr_cholesky_fn=None,
    set_X_fn=None,
    calc_X_shift_scale_fn=None,
):
    y_data = build_y_data_from_inputs(
        runtime=runtime,
        Y=Y,
        Y_for_regression=Y_for_regression,
        Y_exomes=Y_exomes,
        Y_positive_controls=Y_positive_controls,
        Y_case_counts=Y_case_counts,
        Y_corr_m=Y_corr_m,
        store_cholesky=store_cholesky,
        store_corr_sparse=store_corr_sparse,
        skip_V=skip_V,
        skip_scale_factors=skip_scale_factors,
        min_correlation=min_correlation,
        get_y_corr_cholesky_fn=get_y_corr_cholesky_fn,
        set_X_fn=set_X_fn,
        calc_X_shift_scale_fn=calc_X_shift_scale_fn,
    )
    apply_y_data_to_runtime(runtime, y_data)
    return y_data


def apply_y_data_to_runtime(runtime, y_data):
    runtime.Y = y_data.Y
    runtime.Y_for_regression = y_data.Y_for_regression
    runtime.Y_exomes = y_data.Y_exomes
    runtime.Y_positive_controls = y_data.Y_positive_controls
    runtime.Y_case_counts = y_data.Y_case_counts
    runtime.y_var = y_data.y_var
    runtime.y_corr = y_data.y_corr
    runtime.y_corr_cholesky = y_data.y_corr_cholesky
    runtime.y_corr_sparse = y_data.y_corr_sparse
    runtime.Y_w = y_data.Y_w
    runtime.Y_fw = y_data.Y_fw
    runtime.y_w_var = y_data.y_w_var
    runtime.y_w_mean = y_data.y_w_mean
    runtime.y_fw_var = y_data.y_fw_var
    runtime.y_fw_mean = y_data.y_fw_mean


def hyperparameter_data_from_runtime(runtime):
    return HyperparameterData.from_runtime(runtime)


def apply_hyperparameter_data_to_runtime(runtime, hyper_data):
    hyper_data.apply_to_runtime(runtime)


def ensure_hyperparameter_state(runtime):
    hyper_state = getattr(runtime, "hyperparameter_state", None)
    if isinstance(hyper_state, HyperparameterData):
        return hyper_state
    hyper_state = hyperparameter_data_from_runtime(runtime)
    runtime.hyperparameter_state = hyper_state
    return hyper_state


def phewas_runtime_state_from_runtime(runtime):
    return PhewasRuntimeState(
        phenos=getattr(runtime, "phenos", None),
        pheno_to_ind=getattr(runtime, "pheno_to_ind", None),
        gene_pheno_Y=getattr(runtime, "gene_pheno_Y", None),
        gene_pheno_combined_prior_Ys=getattr(runtime, "gene_pheno_combined_prior_Ys", None),
        gene_pheno_priors=getattr(runtime, "gene_pheno_priors", None),
        X_phewas_beta=getattr(runtime, "X_phewas_beta", None),
        X_phewas_beta_uncorrected=getattr(runtime, "X_phewas_beta_uncorrected", None),
        num_gene_phewas_filtered=getattr(runtime, "num_gene_phewas_filtered", 0),
        anchor_gene_mask=getattr(runtime, "anchor_gene_mask", None),
        anchor_pheno_mask=getattr(runtime, "anchor_pheno_mask", None),
    )


def apply_phewas_runtime_state_to_runtime(runtime, phewas_state):
    runtime.phenos = phewas_state.phenos
    runtime.pheno_to_ind = phewas_state.pheno_to_ind
    runtime.gene_pheno_Y = phewas_state.gene_pheno_Y
    runtime.gene_pheno_combined_prior_Ys = phewas_state.gene_pheno_combined_prior_Ys
    runtime.gene_pheno_priors = phewas_state.gene_pheno_priors
    runtime.X_phewas_beta = phewas_state.X_phewas_beta
    runtime.X_phewas_beta_uncorrected = phewas_state.X_phewas_beta_uncorrected
    runtime.num_gene_phewas_filtered = phewas_state.num_gene_phewas_filtered
    runtime.anchor_gene_mask = phewas_state.anchor_gene_mask
    runtime.anchor_pheno_mask = phewas_state.anchor_pheno_mask


def sync_y_state(runtime):
    y_state = y_data_from_runtime(runtime)
    apply_y_data_to_runtime(runtime, y_state)
    return y_state


def sync_hyperparameter_state(runtime):
    hyper_state = ensure_hyperparameter_state(runtime)
    apply_hyperparameter_data_to_runtime(runtime, hyper_state)
    return hyper_state


def sync_phewas_runtime_state(runtime):
    phewas_state = phewas_runtime_state_from_runtime(runtime)
    apply_phewas_runtime_state_to_runtime(runtime, phewas_state)
    return phewas_state


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
    if requested_input is None:
        return None
    if not read_gene_phewas:
        return requested_input
    if num_gene_phewas_filtered != 0:
        return requested_input
    for candidate in reusable_inputs:
        if candidate is not None and requested_input == candidate:
            return None
    return requested_input


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


def set_runtime_p(runtime, p):
    hyper_state = ensure_hyperparameter_state(runtime)
    hyper_state.set_p(p)
    apply_hyperparameter_data_to_runtime(runtime, hyper_state)
    return hyper_state


def set_runtime_sigma(
    runtime,
    sigma2,
    sigma_power,
    sigma2_osc=None,
    sigma2_se=None,
    sigma2_p=None,
    sigma2_scale_factors=None,
    convert_sigma_to_internal_units=False,
):
    hyper_state = ensure_hyperparameter_state(runtime)
    hyper_state.set_sigma(
        runtime,
        sigma2,
        sigma_power,
        sigma2_osc=sigma2_osc,
        sigma2_se=sigma2_se,
        sigma2_p=sigma2_p,
        sigma2_scale_factors=sigma2_scale_factors,
        convert_sigma_to_internal_units=convert_sigma_to_internal_units,
    )
    apply_hyperparameter_data_to_runtime(runtime, hyper_state)
    return hyper_state


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


def json_safe(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return [json_safe(x) for x in value.tolist()]
    if isinstance(value, set):
        return [json_safe(x) for x in sorted(value)]
    if isinstance(value, tuple):
        return [json_safe(x) for x in value]
    if isinstance(value, list):
        return [json_safe(x) for x in value]
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    return value


def resolve_column_index(col_name_or_index, header_cols, require_match=True, *, bail_fn=None):
    if bail_fn is None:
        bail_fn = _default_bail

    try:
        col_ind = int(col_name_or_index)
    except (TypeError, ValueError):
        col_ind = None

    if col_ind is not None:
        if col_ind <= 0:
            bail_fn("All column ids specified as indices are 1-based")
        return col_ind - 1

    matching_cols = [i for i in range(0, len(header_cols)) if header_cols[i] == col_name_or_index]
    if len(matching_cols) == 0:
        if require_match:
            bail_fn("Could not find match for column %s in header: %s" % (col_name_or_index, "\t".join(header_cols)))
        else:
            return None
    if len(matching_cols) > 1:
        bail_fn("Found two matches for column %s in header: %s" % (col_name_or_index, "\t".join(header_cols)))
    return matching_cols[0]


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


def maybe_prepare_filtered_gls_correlation(
    runtime,
    run_gls,
    run_corrected_ols,
    gene_cor_file,
    gene_loc_file,
    gene_cor_file_gene_col,
    gene_cor_file_cor_start_col,
    min_correlation=0.05,
):
    if (run_gls or run_corrected_ols) and runtime.y_corr is None:
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
            store_cholesky=run_gls,
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

    if float(len(runtime.p_values)) / (len(runtime.p_values) + len(runtime.p_values_ignored)) > increase_filter_gene_set_p:
        keep_frac = increase_filter_gene_set_p * float(len(runtime.p_values) + len(runtime.p_values_ignored)) / len(runtime.p_values)
        p_from_quantile = np.quantile(runtime.p_values, keep_frac)
        if p_from_quantile > filter_gene_set_p and not filter_using_phewas:
            overcorrect_ignore = runtime.p_values > p_from_quantile
            if np.sum(overcorrect_ignore) > 0:
                overcorrect_mask = ~overcorrect_ignore
                runtime._record_param("adjusted_filter_gene_set_p", p_from_quantile)
                log_fn(
                    "Ignoring %d gene sets due to p > %.3g (overaggressive adjustment of p-value filters; kept %d)"
                    % (np.sum(overcorrect_ignore), p_from_quantile, np.sum(overcorrect_mask))
                )
                runtime.subset_gene_sets(overcorrect_mask, ignore_missing=True, keep_missing=False, skip_V=True)


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


def _append_initial_p_indices(initial_ps, input_specs, xin_to_p_noninf_ind):
    if initial_ps is None:
        return
    for input_spec in input_specs:
        assert(input_spec in xin_to_p_noninf_ind)
        initial_ps.append(xin_to_p_noninf_ind[input_spec])


def _map_initial_p_indices_to_values(initial_ps, initial_p):
    if initial_ps is None:
        return
    assert(type(initial_p) is list)
    for i in range(len(initial_ps)):
        assert(initial_ps[i]) >= 0 and initial_ps[i] < len(initial_p)
        initial_ps[i] = initial_p[initial_ps[i]]


def _expand_x_inputs(x_inputs, orig_files, batch_separator="@", file_separator=None):
    expanded_inputs = []
    batches = []
    labels = []
    expanded_orig_files = []
    for i in range(len(x_inputs)):
        x_input = x_inputs[i]
        orig_file = orig_files[i]
        batch = None
        label = os.path.basename(orig_file)
        if "." in label:
            label = ".".join(label.split(".")[:-1])
        if batch_separator in x_input:
            batch = x_input.split(batch_separator)[-1]
            label = batch
            x_input = batch_separator.join(x_input.split(batch_separator)[:-1])

        (x_input, tag) = remove_tag_from_input(x_input)
        if tag is not None:
            label = tag

        if file_separator is not None:
            x_to_add = x_input.split(file_separator)
        else:
            x_to_add = [x_input]

        expanded_inputs += x_to_add
        batches += [batch] * len(x_to_add)
        labels += [label] * len(x_to_add)
        expanded_orig_files += [orig_file] * len(x_to_add)
    return (expanded_inputs, batches, labels, expanded_orig_files)


def _append_inputs_from_list_files(
    list_specs,
    dest_inputs,
    dest_orig_files,
    list_open_fn,
    strip_fn,
    resolve_relative_paths=False,
    skip_empty_lines=True,
    initial_ps=None,
    xin_to_p_noninf_ind=None,
    batch_separator="@",
):
    if list_specs is None:
        return

    if type(list_specs) == str:
        list_specs = [list_specs]

    for list_spec in list_specs:
        batch = None
        if batch_separator in list_spec:
            batch = list_spec.split(batch_separator)[-1]
            list_spec = batch_separator.join(list_spec.split(batch_separator)[:-1])

        list_dir = os.path.dirname(os.path.abspath(list_spec))
        with list_open_fn(list_spec) as list_fh:
            for raw_line in list_fh:
                line = strip_fn(raw_line)
                if skip_empty_lines and len(line) == 0:
                    continue

                if resolve_relative_paths:
                    (path, label) = remove_tag_from_input(line)
                    if path and not os.path.isabs(path):
                        path = os.path.normpath(os.path.join(list_dir, path))
                    line = add_tag_to_input(path, label)

                if batch is not None and batch_separator not in line:
                    line = "%s%s%s" % (line, batch_separator, batch)

                dest_inputs.append(line)
                if initial_ps is not None:
                    assert(list_spec in xin_to_p_noninf_ind)
                    initial_ps.append(xin_to_p_noninf_ind[list_spec])
                dest_orig_files.append(list_spec)


def prepare_read_x_inputs(
    X_in,
    X_list,
    Xd_in,
    Xd_list,
    initial_p,
    xin_to_p_noninf_ind,
    batch_separator,
    file_separator,
    *,
    sparse_list_open_fn,
    dense_list_open_fn,
):
    initial_ps = None
    if initial_p is not None:
        if type(initial_p) is not list:
            initial_p = [initial_p]
        initial_ps = []
        assert(xin_to_p_noninf_ind is not None)

    (X_ins, orig_files) = _normalize_input_specs(X_in)
    _append_initial_p_indices(initial_ps, X_ins, xin_to_p_noninf_ind)
    _append_inputs_from_list_files(
        list_specs=X_list,
        dest_inputs=X_ins,
        dest_orig_files=orig_files,
        list_open_fn=sparse_list_open_fn,
        strip_fn=lambda line: line.strip(),
        resolve_relative_paths=True,
        skip_empty_lines=True,
        initial_ps=initial_ps,
        xin_to_p_noninf_ind=xin_to_p_noninf_ind,
        batch_separator=batch_separator,
    )
    X_ins, batches, labels, orig_files = _expand_x_inputs(
        X_ins,
        orig_files,
        batch_separator=batch_separator,
        file_separator=file_separator,
    )
    is_dense = [False for _ in X_ins]

    (Xd_ins, orig_dfiles) = _normalize_input_specs(Xd_in)
    _append_initial_p_indices(initial_ps, Xd_ins, xin_to_p_noninf_ind)
    _append_inputs_from_list_files(
        list_specs=Xd_list,
        dest_inputs=Xd_ins,
        dest_orig_files=orig_dfiles,
        list_open_fn=dense_list_open_fn,
        strip_fn=lambda line: line.strip("\n"),
        resolve_relative_paths=False,
        skip_empty_lines=False,
        initial_ps=initial_ps,
        xin_to_p_noninf_ind=xin_to_p_noninf_ind,
        batch_separator=batch_separator,
    )
    Xd_ins, batches2, labels2, orig_dfiles = _expand_x_inputs(
        Xd_ins,
        orig_dfiles,
        batch_separator=batch_separator,
        file_separator=file_separator,
    )

    _map_initial_p_indices_to_values(initial_ps, initial_p)

    X_ins += Xd_ins
    batches += batches2
    labels += labels2
    orig_files += orig_dfiles
    is_dense += [True for _ in Xd_ins]
    return XInputPlan(
        initial_ps=initial_ps,
        X_ins=X_ins,
        batches=batches,
        labels=labels,
        orig_files=orig_files,
        is_dense=is_dense,
    )


def xdata_from_input_plan(input_plan):
    return XData.from_input_plan(input_plan)


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


def build_read_x_ingestion_options(local_vars):
    return XReadIngestionOptions(
        batch_all_for_hyper=local_vars["batch_all_for_hyper"],
        first_for_hyper=local_vars["first_for_hyper"],
        update_hyper_sigma=local_vars["update_hyper_sigma"],
        update_hyper_p=local_vars["update_hyper_p"],
        first_for_sigma_cond=local_vars["first_for_sigma_cond"],
        run_gls=local_vars["run_gls"],
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
    # genes x gene-set indicator matrices (sparse, unscaled storage)
    runtime.X_orig = None
    runtime.X_binary_packed = None
    runtime.X_orig_missing_genes = None
    runtime.X_orig_missing_genes_missing_gene_sets = None
    runtime.X_orig_missing_gene_sets = None
    runtime.last_X_block = None

    # block and scaling metadata
    runtime.batch_size = batch_size
    runtime.scale_is_for_whitened = False
    runtime.scale_factors = None
    runtime.mean_shifts = None
    runtime.scale_factors_missing = None
    runtime.mean_shifts_missing = None
    runtime.scale_factors_ignored = None
    runtime.mean_shifts_ignored = None

    # gene-set metadata
    runtime.is_dense_gene_set = None
    runtime.is_dense_gene_set_missing = None
    runtime.gene_set_batches = None
    runtime.gene_set_batches_missing = None
    runtime.gene_set_labels = None
    runtime.gene_set_labels_missing = None
    runtime.gene_set_labels_ignored = None

    # gene metadata
    runtime.genes = None
    runtime.genes_missing = None
    runtime.gene_to_ind = None
    runtime.gene_missing_to_ind = None
    runtime.gene_chrom_name_pos = None
    runtime.gene_to_chrom = None
    runtime.gene_to_pos = None
    runtime.gene_to_gwas_huge_score = None
    runtime.gene_to_gwas_huge_score_uncorrected = None
    runtime.gene_to_exomes_huge_score = None
    runtime.gene_to_huge_score = None


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


def clean_chrom_name(chrom):
    if chrom is None:
        return chrom
    if len(chrom) >= 3 and chrom[:3] == "chr":
        return chrom[3:]
    return chrom


def parse_gene_map_file(
    gene_map_in,
    *,
    gene_map_orig_gene_col=1,
    gene_map_new_gene_col=2,
    allow_multi=False,
    bail_fn=None,
):
    if bail_fn is None:
        bail_fn = _default_bail

    gene_label_map = {}
    gene_map_orig_gene_col -= 1
    if gene_map_orig_gene_col < 0:
        bail_fn("--gene-map-orig-gene-col must be greater than 1")
    gene_map_new_gene_col -= 1
    if gene_map_new_gene_col < 0:
        bail_fn("--gene-map-new-gene-col must be greater than 1")

    with open(gene_map_in) as map_fh:
        for line in map_fh:
            cols = line.strip('\n').split()
            if len(cols) <= gene_map_orig_gene_col or len(cols) <= gene_map_new_gene_col:
                bail_fn("Not enough columns in --gene-map-in:\n\t%s" % line)

            # Preserve legacy semantics: currently the parser always consumes
            # columns 1 and 2 after validating requested column indices.
            orig_gene = cols[0]
            new_gene = cols[1]
            if allow_multi:
                if orig_gene not in gene_label_map:
                    gene_label_map[orig_gene] = set()
                gene_label_map[orig_gene].add(new_gene)
            else:
                gene_label_map[orig_gene] = new_gene
    return gene_label_map


def read_loc_file_with_gene_map(
    loc_file,
    *,
    gene_label_map=None,
    return_intervals=False,
    hold_out_chrom=None,
    clean_chrom_fn=None,
    warn_fn=None,
    bail_fn=None,
    split_gene_length=1000000,
):
    if clean_chrom_fn is None:
        clean_chrom_fn = clean_chrom_name
    if warn_fn is None:
        warn_fn = lambda _msg: None
    if bail_fn is None:
        bail_fn = _default_bail

    gene_to_chrom = {}
    gene_to_pos = {}
    gene_chrom_name_pos = {}
    chrom_interval_to_gene = {}

    with open(loc_file) as loc_fh:
        for line in loc_fh:
            cols = line.strip('\n').split()
            if len(cols) != 6:
                bail_fn(
                    "Format for --gene-loc-file is:\n\tgene_id\tchrom\tstart\tstop\tstrand\tgene_name\nOffending line:\n\t%s"
                    % line
                )
            gene = cols[5]
            if gene_label_map is not None and gene in gene_label_map:
                gene = gene_label_map[gene]
            chrom = clean_chrom_fn(cols[1])
            if hold_out_chrom is not None and chrom == hold_out_chrom:
                continue
            pos1 = int(cols[2])
            pos2 = int(cols[3])

            if gene in gene_to_chrom and gene_to_chrom[gene] != chrom:
                warn_fn("Gene %s appears multiple times with different chromosomes; keeping only first" % gene)
                continue

            if gene in gene_to_pos and np.abs(np.mean(gene_to_pos[gene]) - np.mean((pos1, pos2))) > 1e7:
                warn_fn("Gene %s appears multiple times with far away positions; keeping only first" % gene)
                continue

            gene_to_chrom[gene] = chrom
            gene_to_pos[gene] = (pos1, pos2)

            if chrom not in gene_chrom_name_pos:
                gene_chrom_name_pos[chrom] = {}
            if gene not in gene_chrom_name_pos[chrom]:
                gene_chrom_name_pos[chrom][gene] = set()
            gene_chrom_name_pos[chrom][gene].add(pos1)
            gene_chrom_name_pos[chrom][gene].add(pos2)

            if pos2 < pos1:
                pos1, pos2 = pos2, pos1

            if return_intervals:
                if chrom not in chrom_interval_to_gene:
                    chrom_interval_to_gene[chrom] = {}
                if (pos1, pos2) not in chrom_interval_to_gene[chrom]:
                    chrom_interval_to_gene[chrom][(pos1, pos2)] = []
                chrom_interval_to_gene[chrom][(pos1, pos2)].append(gene)

            if pos2 > pos1:
                for posm in range(pos1, pos2, split_gene_length)[1:]:
                    gene_chrom_name_pos[chrom][gene].add(posm)

    if return_intervals:
        return chrom_interval_to_gene
    return (gene_chrom_name_pos, gene_to_chrom, gene_to_pos)


def construct_map_to_ind(values):
    return dict([(values[i], i) for i in range(len(values))])


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


def iter_parser_options(parser):
    for option in parser.option_list:
        if option is not None and option.dest is not None:
            yield option
    for group in parser.option_groups:
        for option in group.option_list:
            if option is not None and option.dest is not None:
                yield option


def collect_cli_specified_dests(argv, parser):
    option_lookup = {}
    for option in iter_parser_options(parser):
        for long_opt in option._long_opts:
            option_lookup[long_opt] = option

    specified_dests = set()
    i = 0
    while i < len(argv):
        arg = argv[i]
        if arg == "--":
            break
        if arg.startswith("--"):
            opt_token = arg.split("=", 1)[0]
            if opt_token in option_lookup:
                opt_obj = option_lookup[opt_token]
                if opt_obj.dest is not None:
                    specified_dests.add(opt_obj.dest)
                if "=" not in arg and opt_obj.takes_value() and i + 1 < len(argv):
                    i += 1
        i += 1
    return specified_dests


def coerce_config_value(option, raw_value, bail_fn=None):
    if bail_fn is None:
        bail_fn = _default_bail

    def _cast_scalar(scalar):
        if scalar is None:
            return None
        if option.type == "int":
            return int(scalar)
        if option.type == "float":
            return float(scalar)
        return scalar

    if option.action == "append":
        values = raw_value if isinstance(raw_value, list) else [raw_value]
        return [_cast_scalar(v) for v in values]

    if option.action in ("store_true", "store_false"):
        if isinstance(raw_value, bool):
            return raw_value
        if isinstance(raw_value, str):
            lower = raw_value.strip().lower()
            if lower in ("1", "true", "yes", "y", "on"):
                return True
            if lower in ("0", "false", "no", "n", "off"):
                return False
        bail_fn("Config value for %s must be boolean" % (option.dest))

    if option.action == "callback":
        return raw_value

    return _cast_scalar(raw_value)


def _format_moved_tool_name(replacement):
    if not isinstance(replacement, str):
        return None
    if not replacement.startswith("__MOVED_TO_"):
        return None
    tool = replacement[len("__MOVED_TO_"):].strip("_")
    if len(tool) == 0:
        return None
    return tool.lower()


def format_removed_option_message(option_name, replacement, context, config_path=None):
    moved_tool = _format_moved_tool_name(replacement)
    if context == "cli":
        if moved_tool is not None:
            return "Error: option %s moved to %s.py after repository split; run this in the %s repository" % (
                option_name,
                moved_tool,
                moved_tool,
            )
        if replacement is None:
            return "Error: option %s has been removed and is no longer supported" % option_name
        return "Error: option %s has been removed; use %s instead" % (option_name, replacement)

    if context == "config":
        if moved_tool is not None:
            return "Config key '%s' moved to %s.py after repository split; run this in the %s repository" % (
                option_name,
                moved_tool,
                moved_tool,
            )
        if replacement is None:
            return "Config key '%s' has been removed in %s and is no longer supported" % (option_name, config_path)
        replacement_config_key = replacement[2:].replace("-", "_") if isinstance(replacement, str) and replacement.startswith("--") else replacement
        return "Config key '%s' has been removed in %s; use '%s' (CLI: %s) instead" % (
            option_name,
            config_path,
            replacement_config_key,
            replacement,
        )

    raise ValueError("Unknown removed-option message context '%s'" % context)


def fail_removed_cli_aliases(
    argv,
    removed_option_replacements,
    *,
    format_removed_option_message_fn=None,
    stderr_write_fn=None,
    exit_fn=None,
):
    formatter = format_removed_option_message if format_removed_option_message_fn is None else format_removed_option_message_fn
    write_fn = sys.stderr.write if stderr_write_fn is None else stderr_write_fn
    terminate_fn = sys.exit if exit_fn is None else exit_fn

    for arg in argv:
        if not isinstance(arg, str) or not arg.startswith("--"):
            continue
        flag = arg.split("=", 1)[0]
        normalized = flag[2:].replace("-", "_")
        if normalized in removed_option_replacements:
            replacement = removed_option_replacements[normalized]
            write_fn("%s\n" % formatter(flag, replacement, context="cli"))
            terminate_fn(2)


def apply_cli_config_overrides(
    options_obj,
    args,
    parser,
    argv,
    *,
    resolve_path_fn,
    is_path_like_dest_fn,
    early_warn_fn,
    bail_fn,
    removed_option_replacements,
    format_removed_option_message_fn,
    track_config_specified_dests=False,
):
    cli_specified_dests = collect_cli_specified_dests(argv, parser)
    config_specified_dests = set() if track_config_specified_dests else None
    config_mode = None

    if getattr(options_obj, "config", None) is not None:
        config_path = resolve_path_fn(options_obj.config, os.getcwd())
        options_obj.config = config_path
        config_dir = os.path.dirname(config_path)
        config_data = load_json_config(config_path, bail_fn=bail_fn, seen_paths=None)
        config_mode = config_data["mode"] if "mode" in config_data else None

        if "options" in config_data:
            config_options = config_data["options"]
            if not isinstance(config_options, dict):
                bail_fn("Config key 'options' must be a JSON object")
        else:
            config_options = dict(config_data)
            config_options.pop("mode", None)
            config_options.pop("include", None)

        apply_config_option_overrides(
            options_obj,
            parser,
            config_options,
            config_path,
            config_dir,
            cli_specified_dests,
            resolve_path_fn=resolve_path_fn,
            is_path_like_dest_fn=is_path_like_dest_fn,
            early_warn_fn=early_warn_fn,
            bail_fn=bail_fn,
            removed_option_replacements=removed_option_replacements,
            format_removed_option_message_fn=format_removed_option_message_fn,
            config_specified_dests=config_specified_dests,
        )

    return options_obj, args, config_mode, cli_specified_dests, config_specified_dests


def harmonize_cli_mode_args(args, config_mode, *, early_warn_fn=None):
    resolved_args = list(args)
    if config_mode is None:
        return resolved_args
    if len(resolved_args) < 1:
        return [config_mode]
    if resolved_args[0] != config_mode and early_warn_fn is not None:
        early_warn_fn("Config mode '%s' differs from CLI mode '%s'; using CLI mode" % (config_mode, resolved_args[0]))
    return resolved_args


def coerce_option_int_list(values, option_name, bail_fn):
    try:
        return [int(x) for x in values]
    except Exception:
        bail_fn("option %s: invalid integer list %s" % (option_name, values))


def initialize_cli_logging(options_obj, *, stderr_stream=None, default_debug_level=1):
    if stderr_stream is None:
        stderr_stream = sys.stderr

    log_fh = open_optional_log_handle(
        getattr(options_obj, "log_file", None),
        default_stream=stderr_stream,
        mode="w",
    )
    warnings_fh = open_optional_log_handle(
        getattr(options_obj, "warnings_file", None),
        default_stream=stderr_stream,
        mode="w",
    )

    NONE = 0
    INFO = 1
    DEBUG = 2
    TRACE = 3
    debug_level = default_debug_level if getattr(options_obj, "debug_level", None) is None else options_obj.debug_level

    def log(message, level=INFO, end_char="\n"):
        if level <= debug_level:
            log_fh.write("%s%s" % (message, end_char))
            log_fh.flush()

    def warn(message):
        if warnings_fh is not None:
            warnings_fh.write("Warning: %s\n" % message)
            warnings_fh.flush()
        log(message, level=INFO)

    return {
        "NONE": NONE,
        "INFO": INFO,
        "DEBUG": DEBUG,
        "TRACE": TRACE,
        "debug_level": debug_level,
        "log_fh": log_fh,
        "warnings_fh": warnings_fh,
        "log": log,
        "warn": warn,
    }


def configure_random_seed(options_obj, random_module, numpy_module, log_fn=None, info_level=None):
    if getattr(options_obj, "deterministic", False) and getattr(options_obj, "seed", None) is None:
        options_obj.seed = 0

    if getattr(options_obj, "seed", None) is not None:
        random_module.seed(options_obj.seed)
        numpy_module.random.seed(options_obj.seed)
        if log_fn is not None:
            if info_level is None:
                log_fn("Using deterministic random seed %d" % options_obj.seed)
            else:
                log_fn("Using deterministic random seed %d" % options_obj.seed, info_level)


def apply_config_option_overrides(
    options_obj,
    parser,
    config_options,
    config_path,
    config_dir,
    cli_specified_dests,
    *,
    resolve_path_fn,
    is_path_like_dest_fn,
    early_warn_fn,
    bail_fn,
    removed_option_replacements=None,
    format_removed_option_message_fn=None,
    config_specified_dests=None,
):
    dest_to_option = {}
    long_key_to_dest = {}
    for opt in iter_parser_options(parser):
        dest_to_option[opt.dest] = opt
        for long_opt in opt._long_opts:
            key = long_opt.lstrip("-")
            long_key_to_dest[key] = opt.dest
            long_key_to_dest[key.replace("-", "_")] = opt.dest

    for raw_key, raw_value in config_options.items():
        if raw_key in ("mode", "options", "include"):
            continue

        normalized_config_key = raw_key
        if isinstance(normalized_config_key, str):
            if normalized_config_key.startswith("--"):
                normalized_config_key = normalized_config_key[2:]
            normalized_config_key = normalized_config_key.replace("-", "_")

        if (
            removed_option_replacements is not None
            and normalized_config_key in removed_option_replacements
        ):
            replacement = removed_option_replacements[normalized_config_key]
            if format_removed_option_message_fn is not None:
                bail_fn(
                    format_removed_option_message_fn(
                        raw_key,
                        replacement,
                        context="config",
                        config_path=config_path,
                    )
                )
            if replacement is None:
                bail_fn(
                    "Config key '%s' has been removed in %s and is no longer supported"
                    % (raw_key, config_path)
                )
            replacement_config_key = (
                replacement[2:].replace("-", "_")
                if isinstance(replacement, str) and replacement.startswith("--")
                else replacement
            )
            bail_fn(
                "Config key '%s' has been removed in %s; use '%s' (CLI: %s) instead"
                % (raw_key, config_path, replacement_config_key, replacement)
            )

        key = raw_key[2:] if isinstance(raw_key, str) and raw_key.startswith("--") else raw_key
        key_norm = key.replace("-", "_") if isinstance(key, str) else key
        if key in dest_to_option:
            dest = key
        elif key_norm in dest_to_option:
            dest = key_norm
        elif key in long_key_to_dest:
            dest = long_key_to_dest[key]
        elif key_norm in long_key_to_dest:
            dest = long_key_to_dest[key_norm]
        else:
            early_warn_fn("Ignoring unknown config key '%s' in %s" % (raw_key, config_path))
            continue

        if dest in cli_specified_dests:
            continue

        opt = dest_to_option[dest]
        coerced_value = coerce_config_value(opt, raw_value, bail_fn=bail_fn)
        if is_path_like_dest_fn(dest):
            if isinstance(coerced_value, list):
                coerced_value = [
                    resolve_path_fn(v, config_dir) if isinstance(v, str) else v
                    for v in coerced_value
                ]
            elif isinstance(coerced_value, str):
                coerced_value = resolve_path_fn(coerced_value, config_dir)

        setattr(options_obj, dest, coerced_value)
        if config_specified_dests is not None:
            config_specified_dests.add(dest)


def get_tar_write_mode_for_bundle_path(bundle_path, option_name="--eaggl-bundle-out", bail_fn=None):
    if bail_fn is None:
        bail_fn = _default_bail
    lower = bundle_path.lower()
    if lower.endswith(".tar.gz") or lower.endswith(".tgz"):
        return "w:gz"
    if lower.endswith(".tar"):
        return "w"
    bail_fn("Option %s must end with .tar, .tar.gz, or .tgz" % option_name)


def _is_unsafe_tar_member_path(member_name):
    if os.path.isabs(member_name):
        return True
    normalized_parts = member_name.replace("\\", "/").split("/")
    return ".." in normalized_parts


def safe_extract_tar_to_temp(bundle_path, temp_prefix="bundle_in_", bundle_flag_name="--bundle-in", bail_fn=None):
    if bail_fn is None:
        bail_fn = _default_bail
    tmp_dir = tempfile.mkdtemp(prefix=temp_prefix)
    try:
        with tarfile.open(bundle_path, "r:*") as tar_fh:
            members = tar_fh.getmembers()
            for member in members:
                if _is_unsafe_tar_member_path(member.name):
                    bail_fn("Refusing to read suspicious path in %s bundle: %s" % (bundle_flag_name, member.name))
            tar_fh.extractall(tmp_dir)
    except Exception:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise
    return tmp_dir


def write_prefixed_tar_bundle(
    out_path,
    *,
    prefix_basename,
    write_prefix_fn,
    is_bundle_path_fn=None,
    option_name="--bundle-out",
    temp_prefix="bundle_out_",
    bail_fn=None,
):
    if bail_fn is None:
        bail_fn = _default_bail
    if is_bundle_path_fn is None:
        is_bundle_path_fn = is_huge_statistics_bundle_path

    if not is_bundle_path_fn(out_path):
        write_prefix_fn(out_path)
        return out_path

    tar_mode = get_tar_write_mode_for_bundle_path(
        out_path,
        option_name=option_name,
        bail_fn=bail_fn,
    )
    ensure_parent_dir_for_file(out_path)

    with tempfile.TemporaryDirectory(prefix=temp_prefix) as stage_dir:
        staged_prefix = os.path.join(stage_dir, prefix_basename)
        write_prefix_fn(staged_prefix)
        staged_names = sorted(
            name for name in os.listdir(stage_dir)
            if name.startswith(prefix_basename + ".")
        )
        if len(staged_names) == 0:
            bail_fn("Cannot write %s: no staged files with prefix %s." % (option_name, prefix_basename))
        with tarfile.open(out_path, tar_mode) as tar_fh:
            for name in staged_names:
                tar_fh.add(os.path.join(stage_dir, name), arcname=name)
    return out_path


def read_prefixed_tar_bundle(
    in_path,
    *,
    required_suffix,
    read_prefix_fn,
    is_bundle_path_fn=None,
    bundle_flag_name="--bundle-in",
    temp_prefix="bundle_in_",
    bail_fn=None,
):
    if bail_fn is None:
        bail_fn = _default_bail
    if is_bundle_path_fn is None:
        is_bundle_path_fn = is_huge_statistics_bundle_path

    if not is_bundle_path_fn(in_path):
        return read_prefix_fn(in_path)

    extract_dir = safe_extract_tar_to_temp(
        in_path,
        temp_prefix=temp_prefix,
        bundle_flag_name=bundle_flag_name,
        bail_fn=bail_fn,
    )
    try:
        marker_files = sorted(
            name for name in os.listdir(extract_dir)
            if name.endswith(required_suffix)
        )
        if len(marker_files) == 0:
            bail_fn("%s bundle did not contain a %s file" % (bundle_flag_name, required_suffix))
        if len(marker_files) > 1:
            bail_fn(
                "%s bundle contained multiple %s files: %s"
                % (bundle_flag_name, required_suffix, ", ".join(marker_files))
            )
        prefix = os.path.join(extract_dir, marker_files[0][:-len(required_suffix)])
        return read_prefix_fn(prefix)
    finally:
        shutil.rmtree(extract_dir, ignore_errors=True)


@dataclass
class BundleManifest:
    manifest: dict
    bundle_path: str | None = None
    extract_dir: str | None = None
    default_inputs: dict = field(default_factory=dict)

    @classmethod
    def load_defaults(
        cls,
        bundle_path,
        expected_schema,
        allowed_default_inputs,
        *,
        bundle_flag_name="--bundle-in",
        manifest_name="manifest.json",
        temp_prefix="bundle_in_",
        bail_fn=None,
    ):
        if bail_fn is None:
            bail_fn = _default_bail

        extract_dir, manifest = load_bundle_manifest(
            bundle_path,
            expected_schema,
            bundle_flag_name=bundle_flag_name,
            manifest_name=manifest_name,
            temp_prefix=temp_prefix,
            bail_fn=bail_fn,
        )
        default_inputs = resolve_bundle_default_inputs(
            manifest.get("default_inputs"),
            extract_dir,
            allowed_default_inputs,
            bundle_flag_name=bundle_flag_name,
            bail_fn=bail_fn,
        )
        return cls(
            manifest=manifest,
            bundle_path=bundle_path,
            extract_dir=extract_dir,
            default_inputs=default_inputs,
        )

    @classmethod
    def build(
        cls,
        schema,
        source_tool,
        source_mode,
        source_argv,
        default_inputs,
        files_metadata,
    ):
        return cls(
            manifest={
                "schema": schema,
                "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "source": {
                    "tool": source_tool,
                    "mode": source_mode,
                    "argv": list(source_argv),
                },
                "default_inputs": dict(default_inputs),
                "files": dict(files_metadata),
            },
            default_inputs=dict(default_inputs),
        )

    def write_manifest(self, stage_dir, manifest_name="manifest.json"):
        manifest_path = os.path.join(stage_dir, manifest_name)
        with open(manifest_path, "w", encoding="utf-8") as out_fh:
            json.dump(self.manifest, out_fh, indent=2, sort_keys=True)
            out_fh.write("\n")
        return manifest_path

    def write_archive(self, out_path, tar_mode, stage_dir, staged_file_names, *, manifest_name="manifest.json"):
        manifest_path = os.path.join(stage_dir, manifest_name)
        with tarfile.open(out_path, tar_mode) as tar_fh:
            tar_fh.add(manifest_path, arcname=manifest_name)
            for bundle_name in sorted(staged_file_names):
                tar_fh.add(os.path.join(stage_dir, bundle_name), arcname=bundle_name)


@dataclass
class BundleDefaultsApplication:
    bundle: BundleManifest
    applied_defaults: dict = field(default_factory=dict)

    def as_dict(self):
        return {
            "bundle_path": self.bundle.bundle_path,
            "extract_dir": self.bundle.extract_dir,
            "schema": self.bundle.manifest.get("schema") if isinstance(self.bundle.manifest, dict) else None,
            "manifest": self.bundle.manifest,
            "default_inputs": self.bundle.default_inputs,
            "applied_defaults": self.applied_defaults,
        }


def load_bundle_manifest(
    bundle_path,
    expected_schema,
    *,
    bundle_flag_name="--bundle-in",
    manifest_name="manifest.json",
    temp_prefix="bundle_in_",
    bail_fn=None,
):
    if bail_fn is None:
        bail_fn = _default_bail
    if not os.path.exists(bundle_path):
        bail_fn("Could not find %s bundle %s" % (bundle_flag_name, bundle_path))

    extract_dir = safe_extract_tar_to_temp(
        bundle_path,
        temp_prefix=temp_prefix,
        bundle_flag_name=bundle_flag_name,
        bail_fn=bail_fn,
    )
    manifest_path = os.path.join(extract_dir, manifest_name)
    if not os.path.exists(manifest_path):
        bail_fn("%s bundle is missing %s: %s" % (bundle_flag_name, manifest_name, bundle_path))

    with open(manifest_path) as in_fh:
        manifest = json.load(in_fh)
    if not isinstance(manifest, dict):
        bail_fn("%s manifest must be a JSON object: %s" % (bundle_flag_name, bundle_path))
    if manifest.get("schema") != expected_schema:
        bail_fn(
            "Unsupported %s schema '%s' in %s (expected %s)"
            % (bundle_flag_name, manifest.get("schema"), bundle_path, expected_schema)
        )
    return extract_dir, manifest


def resolve_bundle_default_inputs(
    raw_default_inputs,
    extract_dir,
    allowed_default_inputs,
    *,
    bundle_flag_name="--bundle-in",
    bail_fn=None,
):
    if bail_fn is None:
        bail_fn = _default_bail

    if not isinstance(raw_default_inputs, dict):
        bail_fn("%s manifest missing required object key 'default_inputs'" % bundle_flag_name)

    resolved_default_inputs = {}
    abs_extract_dir = os.path.abspath(extract_dir)
    for key, rel_path in raw_default_inputs.items():
        if key not in allowed_default_inputs:
            continue
        if not isinstance(rel_path, str) or len(rel_path.strip()) == 0:
            bail_fn("Invalid bundle path for default input '%s'" % key)
        joined = os.path.normpath(os.path.join(extract_dir, rel_path))
        abs_joined = os.path.abspath(joined)
        if os.path.commonpath([abs_extract_dir, abs_joined]) != abs_extract_dir:
            bail_fn("Refusing to resolve path outside %s bundle for key '%s': %s" % (bundle_flag_name, key, rel_path))
        if not os.path.exists(joined):
            bail_fn("%s manifest path for '%s' does not exist: %s" % (bundle_flag_name, key, rel_path))
        resolved_default_inputs[key] = joined
    return resolved_default_inputs


def apply_bundle_defaults_to_options(
    options,
    bundle_manifest,
    *,
    x_source_option_names=None,
    x_default_key="X_in",
    x_target_option_name="X_in",
    scalar_default_option_names=None,
):
    defaults = bundle_manifest.default_inputs
    applied = {}

    if x_source_option_names is None:
        x_source_option_names = ["X_in", "X_list", "Xd_in", "Xd_list"]
    if scalar_default_option_names is None:
        scalar_default_option_names = []

    has_explicit_x_source = any(getattr(options, key, None) is not None for key in x_source_option_names)
    if x_default_key in defaults and not has_explicit_x_source:
        setattr(options, x_target_option_name, [defaults[x_default_key]])
        applied[x_target_option_name] = defaults[x_default_key]

    for key in scalar_default_option_names:
        if key not in defaults:
            continue
        if getattr(options, key, None) is None:
            setattr(options, key, defaults[key])
            applied[key] = defaults[key]

    return BundleDefaultsApplication(bundle=bundle_manifest, applied_defaults=applied)


def load_and_apply_bundle_defaults(
    options,
    *,
    bundle_path,
    expected_schema,
    allowed_default_inputs,
    bundle_flag_name="--bundle-in",
    manifest_name="manifest.json",
    temp_prefix="bundle_in_",
    x_source_option_names=None,
    x_default_key="X_in",
    x_target_option_name="X_in",
    scalar_default_option_names=None,
    bail_fn=None,
):
    bundle = BundleManifest.load_defaults(
        bundle_path=bundle_path,
        expected_schema=expected_schema,
        allowed_default_inputs=allowed_default_inputs,
        bundle_flag_name=bundle_flag_name,
        manifest_name=manifest_name,
        temp_prefix=temp_prefix,
        bail_fn=bail_fn,
    )
    return apply_bundle_defaults_to_options(
        options,
        bundle,
        x_source_option_names=x_source_option_names,
        x_default_key=x_default_key,
        x_target_option_name=x_target_option_name,
        scalar_default_option_names=scalar_default_option_names,
    )


def ensure_parent_dir_for_file(path):
    out_dir = os.path.dirname(os.path.abspath(path))
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)


def require_existing_nonempty_file(
    path,
    label,
    suggestion,
    *,
    option_name="--bundle-out",
    bail_fn=None,
):
    if bail_fn is None:
        bail_fn = _default_bail
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return
    bail_fn("Cannot write %s: missing %s (%s)" % (option_name, label, suggestion))


def stage_file_into_dir(source_path, stage_dir, bundle_name, *, bail_fn=None):
    if bail_fn is None:
        bail_fn = _default_bail
    if source_path is None or not os.path.exists(source_path):
        bail_fn("Cannot stage missing file into bundle: %s" % source_path)
    staged_path = os.path.join(stage_dir, bundle_name)
    with open(source_path, "rb") as in_fh:
        with open(staged_path, "wb") as out_fh:
            shutil.copyfileobj(in_fh, out_fh)
    return staged_path


def write_bundle_from_specs(
    out_path,
    *,
    schema,
    source_tool,
    source_mode,
    source_argv,
    generated_file_specs,
    optional_existing_files=None,
    option_name="--bundle-out",
    temp_prefix="bundle_out_",
    manifest_name="manifest.json",
    bail_fn=None,
):
    if bail_fn is None:
        bail_fn = _default_bail

    tar_mode = get_tar_write_mode_for_bundle_path(
        out_path,
        option_name=option_name,
        bail_fn=bail_fn,
    )
    ensure_parent_dir_for_file(out_path)

    with tempfile.TemporaryDirectory(prefix=temp_prefix) as stage_dir:
        file_map = {}
        file_meta = {}

        for (default_key, bundle_name, write_fn, label, suggestion) in generated_file_specs:
            staged_path = os.path.join(stage_dir, bundle_name)
            write_fn(staged_path)
            require_existing_nonempty_file(
                staged_path,
                label,
                suggestion,
                option_name=option_name,
                bail_fn=bail_fn,
            )
            file_map[default_key] = bundle_name
            file_meta[bundle_name] = collect_file_metadata(staged_path)

        for (default_key, source_path, bundle_name) in optional_existing_files or []:
            if source_path is None or not os.path.exists(source_path):
                continue
            staged_path = stage_file_into_dir(
                source_path,
                stage_dir,
                bundle_name,
                bail_fn=bail_fn,
            )
            file_map[default_key] = bundle_name
            file_meta[bundle_name] = collect_file_metadata(staged_path)

        manifest = BundleManifest.build(
            schema=schema,
            source_tool=source_tool,
            source_mode=source_mode,
            source_argv=source_argv,
            default_inputs=file_map,
            files_metadata=file_meta,
        )
        manifest.write_manifest(stage_dir, manifest_name=manifest_name)
        manifest.write_archive(
            out_path,
            tar_mode,
            stage_dir,
            file_meta.keys(),
            manifest_name=manifest_name,
        )

    return out_path


def hash_file_sha256(path):
    sha = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            chunk = fh.read(1024 * 1024)
            if not chunk:
                break
            sha.update(chunk)
    return sha.hexdigest()


def collect_file_metadata(path):
    return {
        "size_bytes": int(os.path.getsize(path)),
        "sha256": hash_file_sha256(path),
    }


def is_huge_statistics_bundle_path(huge_statistics_file):
    lower = huge_statistics_file.lower()
    return lower.endswith(".tar.gz") or lower.endswith(".tgz") or lower.endswith(".tar")


def coerce_runtime_state_dict(runtime_state, *, bail_fn=None):
    if bail_fn is None:
        bail_fn = _default_bail
    if isinstance(runtime_state, dict):
        return runtime_state
    if hasattr(runtime_state, "__dict__"):
        return runtime_state.__dict__
    bail_fn("Internal error: unsupported runtime state container for HuGE cache IO")


def get_huge_statistics_paths_for_prefix(prefix):
    return {
        "meta": "%s.huge.meta.json.gz" % prefix,
        "cache_genes": "%s.huge.cache_genes.tsv.gz" % prefix,
        "extra_scores": "%s.huge.extra_scores.tsv.gz" % prefix,
        "matrix_row_genes": "%s.huge.matrix_row_genes.tsv.gz" % prefix,
        "gene_scores": "%s.huge.gene_scores.tsv.gz" % prefix,
        "gene_covariates": "%s.huge.gene_covariates.tsv.gz" % prefix,
        "bfs_data": "%s.huge_signal_bfs.data.tsv.gz" % prefix,
        "bfs_indices": "%s.huge_signal_bfs.indices.tsv.gz" % prefix,
        "bfs_indptr": "%s.huge_signal_bfs.indptr.tsv.gz" % prefix,
        "bfs_reg_data": "%s.huge_signal_bfs_for_regression.data.tsv.gz" % prefix,
        "bfs_reg_indices": "%s.huge_signal_bfs_for_regression.indices.tsv.gz" % prefix,
        "bfs_reg_indptr": "%s.huge_signal_bfs_for_regression.indptr.tsv.gz" % prefix,
        "signal_posteriors": "%s.huge_signal_posteriors.tsv.gz" % prefix,
        "signal_posteriors_for_regression": "%s.huge_signal_posteriors_for_regression.tsv.gz" % prefix,
        "signal_sum_gene_cond_probabilities": "%s.huge_signal_sum_gene_cond_probabilities.tsv.gz" % prefix,
        "signal_sum_gene_cond_probabilities_for_regression": "%s.huge_signal_sum_gene_cond_probabilities_for_regression.tsv.gz" % prefix,
        "signal_mean_gene_pos": "%s.huge_signal_mean_gene_pos.tsv.gz" % prefix,
        "signal_mean_gene_pos_for_regression": "%s.huge_signal_mean_gene_pos_for_regression.tsv.gz" % prefix,
    }


def write_numeric_vector_file(out_file, values, *, open_text_fn, value_type=float):
    with open_text_fn(out_file, "w") as out_fh:
        if values is None:
            return
        values = np.ravel(np.array(values))
        for value in values:
            if value_type == int:
                out_fh.write("%d\n" % int(value))
            else:
                out_fh.write("%.18g\n" % float(value))


def read_numeric_vector_file(in_file, *, open_text_fn, value_type=float):
    values = []
    with open_text_fn(in_file) as in_fh:
        for line in in_fh:
            line = line.strip()
            if line == "":
                continue
            if value_type == int:
                values.append(int(line))
            else:
                values.append(float(line))
    if value_type == int:
        return np.array(values, dtype=int)
    return np.array(values, dtype=float)


def build_huge_statistics_matrix_row_genes(cache_genes, extra_genes, num_matrix_rows, *, bail_fn=None):
    if bail_fn is None:
        bail_fn = _default_bail
    if len(cache_genes) > 0:
        if num_matrix_rows < len(cache_genes):
            bail_fn("Error writing HuGE statistics cache: matrix rows %d < number of genes %d" % (num_matrix_rows, len(cache_genes)))
        num_extra_matrix_rows = num_matrix_rows - len(cache_genes)
        if num_extra_matrix_rows > len(extra_genes):
            bail_fn("Error writing HuGE statistics cache: matrix rows require %d extra genes but only %d were provided" % (num_extra_matrix_rows, len(extra_genes)))
        return cache_genes + extra_genes[:num_extra_matrix_rows]
    if num_matrix_rows > len(extra_genes):
        bail_fn("Error writing HuGE statistics cache: matrix rows %d > number of extra genes %d" % (num_matrix_rows, len(extra_genes)))
    return extra_genes[:num_matrix_rows]


def build_huge_statistics_score_maps(runtime_state, cache_genes, extra_genes, gene_bf, extra_gene_bf, gene_bf_for_regression, extra_gene_bf_for_regression):
    gene_to_score = {}
    if runtime_state.get("gene_to_gwas_huge_score") is not None:
        gene_to_score = dict(runtime_state["gene_to_gwas_huge_score"])

    gene_to_score_uncorrected = {}
    if runtime_state.get("gene_to_gwas_huge_score_uncorrected") is not None:
        gene_to_score_uncorrected = dict(runtime_state["gene_to_gwas_huge_score_uncorrected"])

    gene_to_score_for_regression = {}

    for i, gene in enumerate(cache_genes):
        if gene_bf is not None and i < len(gene_bf):
            gene_to_score[gene] = float(gene_bf[i])
        if gene_bf_for_regression is not None and i < len(gene_bf_for_regression):
            gene_to_score_for_regression[gene] = float(gene_bf_for_regression[i])
        elif gene in gene_to_score:
            gene_to_score_for_regression[gene] = float(gene_to_score[gene])

    for i, gene in enumerate(extra_genes):
        if i < len(extra_gene_bf):
            gene_to_score[gene] = float(extra_gene_bf[i])
        if i < len(extra_gene_bf_for_regression):
            gene_to_score_for_regression[gene] = float(extra_gene_bf_for_regression[i])
        elif gene in gene_to_score:
            gene_to_score_for_regression[gene] = float(gene_to_score[gene])
        if gene not in gene_to_score_uncorrected:
            gene_to_score_uncorrected[gene] = float(extra_gene_bf[i])

    return (gene_to_score, gene_to_score_uncorrected, gene_to_score_for_regression)


def build_huge_statistics_meta(runtime_state, huge_signal_bfs, huge_signal_bfs_for_regression, *, json_safe_fn=None):
    if json_safe_fn is None:
        json_safe_fn = json_safe
    return {
        "version": 1,
        "huge_signal_bfs_shape": [int(huge_signal_bfs.shape[0]), int(huge_signal_bfs.shape[1])],
        "huge_signal_bfs_for_regression_shape": [int(huge_signal_bfs_for_regression.shape[0]), int(huge_signal_bfs_for_regression.shape[1])],
        "huge_signal_max_closest_gene_prob": (None if runtime_state.get("huge_signal_max_closest_gene_prob") is None else float(runtime_state.get("huge_signal_max_closest_gene_prob"))),
        "huge_cap_region_posterior": bool(runtime_state.get("huge_cap_region_posterior", True)),
        "huge_scale_region_posterior": bool(runtime_state.get("huge_scale_region_posterior", False)),
        "huge_phantom_region_posterior": bool(runtime_state.get("huge_phantom_region_posterior", False)),
        "huge_allow_evidence_of_absence": bool(runtime_state.get("huge_allow_evidence_of_absence", False)),
        "huge_sparse_mode": bool(runtime_state.get("huge_sparse_mode", False)),
        "huge_signals": [] if runtime_state.get("huge_signals") is None else [[str(x[0]), int(x[1]), float(x[2]), x[3]] for x in runtime_state.get("huge_signals")],
        "gene_covariate_names": (None if runtime_state.get("gene_covariate_names") is None else list(runtime_state.get("gene_covariate_names"))),
        "gene_covariate_directions": (None if runtime_state.get("gene_covariate_directions") is None else list(np.array(runtime_state.get("gene_covariate_directions"), dtype=float))),
        "gene_covariate_intercept_index": runtime_state.get("gene_covariate_intercept_index"),
        "gene_covariate_slope_defaults": (None if runtime_state.get("gene_covariate_slope_defaults") is None else list(np.array(runtime_state.get("gene_covariate_slope_defaults"), dtype=float))),
        "total_qc_metric_betas_defaults": (None if runtime_state.get("total_qc_metric_betas_defaults") is None else list(np.array(runtime_state.get("total_qc_metric_betas_defaults"), dtype=float))),
        "total_qc_metric_intercept_defaults": (None if runtime_state.get("total_qc_metric_intercept_defaults") is None else float(runtime_state.get("total_qc_metric_intercept_defaults"))),
        "total_qc_metric2_betas_defaults": (None if runtime_state.get("total_qc_metric2_betas_defaults") is None else list(np.array(runtime_state.get("total_qc_metric2_betas_defaults"), dtype=float))),
        "total_qc_metric2_intercept_defaults": (None if runtime_state.get("total_qc_metric2_intercept_defaults") is None else float(runtime_state.get("total_qc_metric2_intercept_defaults"))),
        "recorded_params": json_safe_fn(runtime_state.get("params")),
        "recorded_param_keys": json_safe_fn(runtime_state.get("param_keys")),
    }


def write_huge_statistics_text_tables(
    paths,
    runtime_state,
    cache_genes,
    extra_genes,
    extra_gene_bf,
    extra_gene_bf_for_regression,
    matrix_row_genes,
    gene_to_score,
    gene_to_score_uncorrected,
    gene_to_score_for_regression,
    *,
    open_text_fn,
    bail_fn=None,
):
    if bail_fn is None:
        bail_fn = _default_bail
    with open_text_fn(paths["cache_genes"], "w") as out_fh:
        for gene in cache_genes:
            out_fh.write("%s\n" % gene)

    with open_text_fn(paths["extra_scores"], "w") as out_fh:
        out_fh.write("Gene\tlog_bf\tlog_bf_for_regression\n")
        for i in range(len(extra_genes)):
            bf = np.nan
            if i < len(extra_gene_bf):
                bf = extra_gene_bf[i]
            bf_for_regression = np.nan
            if i < len(extra_gene_bf_for_regression):
                bf_for_regression = extra_gene_bf_for_regression[i]
            out_fh.write("%s\t%.18g\t%.18g\n" % (extra_genes[i], bf, bf_for_regression))

    with open_text_fn(paths["matrix_row_genes"], "w") as out_fh:
        for gene in matrix_row_genes:
            out_fh.write("%s\n" % gene)

    ordered_genes = []
    seen = set()
    for gene in cache_genes + extra_genes + list(gene_to_score.keys()) + list(gene_to_score_uncorrected.keys()):
        if gene not in seen:
            seen.add(gene)
            ordered_genes.append(gene)

    with open_text_fn(paths["gene_scores"], "w") as out_fh:
        out_fh.write("Gene\tlog_bf\tlog_bf_uncorrected\tlog_bf_for_regression\n")
        for gene in ordered_genes:
            score = gene_to_score.get(gene, np.nan)
            score_uncorrected = gene_to_score_uncorrected.get(gene, np.nan)
            score_for_regression = gene_to_score_for_regression.get(gene, np.nan)
            out_fh.write("%s\t%.18g\t%.18g\t%.18g\n" % (gene, score, score_uncorrected, score_for_regression))

    gene_covariates = runtime_state.get("gene_covariates")
    if gene_covariates is not None:
        if gene_covariates.shape[0] != len(matrix_row_genes):
            bail_fn("Error writing HuGE statistics cache: gene covariates have %d rows but matrix has %d rows" % (gene_covariates.shape[0], len(matrix_row_genes)))
        with open_text_fn(paths["gene_covariates"], "w") as out_fh:
            out_fh.write("Gene\t%s\n" % ("\t".join(runtime_state.get("gene_covariate_names"))))
            for i in range(len(matrix_row_genes)):
                out_fh.write("%s\t%s\n" % (matrix_row_genes[i], "\t".join(["%.18g" % x for x in gene_covariates[i, :]])))


def read_huge_statistics_text_tables(paths, *, open_text_fn):
    cache_genes = []
    with open_text_fn(paths["cache_genes"]) as in_fh:
        for line in in_fh:
            gene = line.strip()
            if gene != "":
                cache_genes.append(gene)

    extra_genes = []
    extra_gene_bf = []
    extra_gene_bf_for_regression = []
    with open_text_fn(paths["extra_scores"]) as in_fh:
        _header = in_fh.readline()
        for line in in_fh:
            cols = line.strip("\n").split("\t")
            if len(cols) < 3:
                continue
            extra_genes.append(cols[0])
            extra_gene_bf.append(float(cols[1]))
            extra_gene_bf_for_regression.append(float(cols[2]))

    matrix_row_genes = []
    with open_text_fn(paths["matrix_row_genes"]) as in_fh:
        for line in in_fh:
            gene = line.strip()
            if gene != "":
                matrix_row_genes.append(gene)

    gene_to_score = {}
    gene_to_score_uncorrected = {}
    gene_to_score_for_regression = {}
    with open_text_fn(paths["gene_scores"]) as in_fh:
        _header = in_fh.readline()
        for line in in_fh:
            cols = line.strip("\n").split("\t")
            if len(cols) < 4:
                continue
            gene = cols[0]
            gene_to_score[gene] = float(cols[1])
            gene_to_score_uncorrected[gene] = float(cols[2])
            gene_to_score_for_regression[gene] = float(cols[3])

    return (
        cache_genes,
        extra_genes,
        extra_gene_bf,
        extra_gene_bf_for_regression,
        matrix_row_genes,
        gene_to_score,
        gene_to_score_uncorrected,
        gene_to_score_for_regression,
    )


def resolve_huge_statistics_gene_vectors(
    runtime_state,
    cache_genes,
    extra_genes,
    matrix_row_genes,
    gene_to_score,
    gene_to_score_for_regression,
    *,
    bail_fn=None,
):
    if bail_fn is None:
        bail_fn = _default_bail
    genes = runtime_state.get("genes")
    if genes is None:
        if len(cache_genes) > 0:
            bail_fn("HuGE cache was generated with preloaded genes but this run has no preloaded genes")
        if len(matrix_row_genes) > 0 and matrix_row_genes != extra_genes[:len(matrix_row_genes)]:
            bail_fn("HuGE cache is inconsistent: matrix rows do not match extra gene ordering")
        return (np.array([]), np.array([]))

    if cache_genes != genes:
        bail_fn("HuGE cache gene ordering does not match current run. Rebuild cache for this run setup.")
    if len(matrix_row_genes) < len(genes) or matrix_row_genes[:len(genes)] != genes:
        bail_fn("HuGE cache matrix row ordering does not match current run genes")

    gene_bf = np.array([gene_to_score.get(gene, np.nan) for gene in genes])
    gene_bf_for_regression = np.array([gene_to_score_for_regression.get(gene, np.nan) for gene in genes])
    return (gene_bf, gene_bf_for_regression)


def read_huge_statistics_covariates_if_present(runtime_state, paths, *, open_text_fn, exists_fn=None):
    if exists_fn is None:
        exists_fn = os.path.exists
    if not exists_fn(paths["gene_covariates"]):
        return
    covariate_rows = []
    with open_text_fn(paths["gene_covariates"]) as in_fh:
        header = in_fh.readline().strip("\n").split("\t")
        if len(header) > 1 and runtime_state.get("gene_covariate_names") is None:
            runtime_state["gene_covariate_names"] = header[1:]
        for line in in_fh:
            cols = line.strip("\n").split("\t")
            if len(cols) <= 1:
                continue
            covariate_rows.append([float(x) for x in cols[1:]])
    runtime_state["gene_covariates"] = np.array(covariate_rows)


def load_huge_statistics_sparse_and_vectors(runtime_state, paths, meta, *, read_vector_fn):
    huge_signal_bfs_shape = tuple(meta["huge_signal_bfs_shape"])
    huge_signal_bfs_for_regression_shape = tuple(meta["huge_signal_bfs_for_regression_shape"])

    sparse_components = {
        "bfs_data": read_vector_fn(paths["bfs_data"], value_type=float),
        "bfs_indices": read_vector_fn(paths["bfs_indices"], value_type=int),
        "bfs_indptr": read_vector_fn(paths["bfs_indptr"], value_type=int),
        "bfs_reg_data": read_vector_fn(paths["bfs_reg_data"], value_type=float),
        "bfs_reg_indices": read_vector_fn(paths["bfs_reg_indices"], value_type=int),
        "bfs_reg_indptr": read_vector_fn(paths["bfs_reg_indptr"], value_type=int),
    }

    runtime_state["huge_signal_bfs"] = sparse.csc_matrix(
        (sparse_components["bfs_data"], sparse_components["bfs_indices"], sparse_components["bfs_indptr"]),
        shape=huge_signal_bfs_shape,
    )
    runtime_state["huge_signal_bfs_for_regression"] = sparse.csc_matrix(
        (sparse_components["bfs_reg_data"], sparse_components["bfs_reg_indices"], sparse_components["bfs_reg_indptr"]),
        shape=huge_signal_bfs_for_regression_shape,
    )

    runtime_vector_map = (
        ("huge_signal_posteriors", "signal_posteriors"),
        ("huge_signal_posteriors_for_regression", "signal_posteriors_for_regression"),
        ("huge_signal_sum_gene_cond_probabilities", "signal_sum_gene_cond_probabilities"),
        ("huge_signal_sum_gene_cond_probabilities_for_regression", "signal_sum_gene_cond_probabilities_for_regression"),
        ("huge_signal_mean_gene_pos", "signal_mean_gene_pos"),
        ("huge_signal_mean_gene_pos_for_regression", "signal_mean_gene_pos_for_regression"),
    )
    for state_key, path_key in runtime_vector_map:
        runtime_state[state_key] = read_vector_fn(paths[path_key], value_type=float)


def apply_huge_statistics_meta_to_runtime(runtime_state, meta):
    runtime_state["huge_signal_max_closest_gene_prob"] = meta["huge_signal_max_closest_gene_prob"]
    runtime_state["huge_cap_region_posterior"] = bool(meta["huge_cap_region_posterior"])
    runtime_state["huge_scale_region_posterior"] = bool(meta["huge_scale_region_posterior"])
    runtime_state["huge_phantom_region_posterior"] = bool(meta["huge_phantom_region_posterior"])
    runtime_state["huge_allow_evidence_of_absence"] = bool(meta["huge_allow_evidence_of_absence"])
    runtime_state["huge_sparse_mode"] = bool(meta["huge_sparse_mode"])
    runtime_state["huge_signals"] = [tuple(x) for x in meta["huge_signals"]]

    runtime_state["gene_covariates"] = None
    runtime_state["gene_covariates_mask"] = None
    runtime_state["gene_covariate_names"] = meta.get("gene_covariate_names")
    runtime_state["gene_covariate_directions"] = None
    if meta.get("gene_covariate_directions") is not None:
        runtime_state["gene_covariate_directions"] = np.array(meta["gene_covariate_directions"], dtype=float)
    runtime_state["gene_covariate_intercept_index"] = meta.get("gene_covariate_intercept_index")
    runtime_state["gene_covariates_mat_inv"] = None
    runtime_state["gene_covariate_zs"] = None
    runtime_state["gene_covariate_adjustments"] = None

    runtime_state["gene_covariate_slope_defaults"] = None if meta.get("gene_covariate_slope_defaults") is None else np.array(meta.get("gene_covariate_slope_defaults"), dtype=float)
    runtime_state["total_qc_metric_betas_defaults"] = None if meta.get("total_qc_metric_betas_defaults") is None else np.array(meta.get("total_qc_metric_betas_defaults"), dtype=float)
    runtime_state["total_qc_metric_intercept_defaults"] = meta.get("total_qc_metric_intercept_defaults")
    runtime_state["total_qc_metric2_betas_defaults"] = None if meta.get("total_qc_metric2_betas_defaults") is None else np.array(meta.get("total_qc_metric2_betas_defaults"), dtype=float)
    runtime_state["total_qc_metric2_intercept_defaults"] = meta.get("total_qc_metric2_intercept_defaults")


def combine_runtime_huge_scores(runtime_state):
    if runtime_state.get("gene_to_gwas_huge_score") is not None and runtime_state.get("gene_to_exomes_huge_score") is not None:
        runtime_state["gene_to_huge_score"] = {}
        genes = list(set().union(runtime_state["gene_to_gwas_huge_score"], runtime_state["gene_to_exomes_huge_score"]))
        for gene in genes:
            runtime_state["gene_to_huge_score"][gene] = 0
            if gene in runtime_state["gene_to_gwas_huge_score"]:
                runtime_state["gene_to_huge_score"][gene] += runtime_state["gene_to_gwas_huge_score"][gene]
            if gene in runtime_state["gene_to_exomes_huge_score"]:
                runtime_state["gene_to_huge_score"][gene] += runtime_state["gene_to_exomes_huge_score"][gene]


def validate_huge_statistics_loaded_shapes(runtime_state, matrix_row_genes, *, bail_fn=None):
    if bail_fn is None:
        bail_fn = _default_bail
    if runtime_state["huge_signal_bfs"].shape[1] != len(runtime_state["huge_signals"]):
        bail_fn("HuGE cache is inconsistent: huge_signal_bfs has %d columns but found %d signals" % (runtime_state["huge_signal_bfs"].shape[1], len(runtime_state["huge_signals"])))
    if runtime_state["huge_signal_bfs"].shape[0] != len(matrix_row_genes):
        bail_fn("HuGE cache is inconsistent: huge_signal_bfs has %d rows but found %d matrix-row genes" % (runtime_state["huge_signal_bfs"].shape[0], len(matrix_row_genes)))


def write_huge_statistics_runtime_vectors(paths, runtime_state, *, write_vector_fn):
    runtime_vector_map = (
        ("signal_posteriors", "huge_signal_posteriors"),
        ("signal_posteriors_for_regression", "huge_signal_posteriors_for_regression"),
        ("signal_sum_gene_cond_probabilities", "huge_signal_sum_gene_cond_probabilities"),
        ("signal_sum_gene_cond_probabilities_for_regression", "huge_signal_sum_gene_cond_probabilities_for_regression"),
        ("signal_mean_gene_pos", "huge_signal_mean_gene_pos"),
        ("signal_mean_gene_pos_for_regression", "huge_signal_mean_gene_pos_for_regression"),
    )
    for path_key, state_key in runtime_vector_map:
        write_vector_fn(paths[path_key], runtime_state.get(state_key), value_type=float)


def write_huge_statistics_sparse_components(paths, huge_signal_bfs, huge_signal_bfs_for_regression, *, write_vector_fn):
    sparse_vector_map = (
        ("bfs_data", huge_signal_bfs.data, float),
        ("bfs_indices", huge_signal_bfs.indices, int),
        ("bfs_indptr", huge_signal_bfs.indptr, int),
        ("bfs_reg_data", huge_signal_bfs_for_regression.data, float),
        ("bfs_reg_indices", huge_signal_bfs_for_regression.indices, int),
        ("bfs_reg_indptr", huge_signal_bfs_for_regression.indptr, int),
    )
    for path_key, values, value_type in sparse_vector_map:
        write_vector_fn(paths[path_key], values, value_type=value_type)


def finalize_regression_outputs(beta_tildes, ses, se_inflation_factors, *, log_fn=None, warn_fn=None, trace_level=0):
    if se_inflation_factors is not None:
        ses *= se_inflation_factors

    if np.prod(ses.shape) > 0:
        empty_mask = np.logical_and(beta_tildes == 0, ses <= 0)
        max_se = np.max(ses)

        if np.sum(empty_mask) > 0 and log_fn is not None:
            log_fn("Zeroing out %d betas due to negative ses" % (np.sum(empty_mask)), trace_level)

        ses[empty_mask] = max_se * 100 if max_se > 0 else 100
        beta_tildes[ses <= 0] = 0

    z_scores = np.zeros(beta_tildes.shape)
    ses_positive_mask = ses > 0
    z_scores[ses_positive_mask] = beta_tildes[ses_positive_mask] / ses[ses_positive_mask]
    if np.any(~ses_positive_mask) and warn_fn is not None:
        warn_fn("There were %d gene sets with negative ses; setting z-scores to 0" % (np.sum(~ses_positive_mask)))
    p_values = 2 * scipy.stats.norm.cdf(-np.abs(z_scores))
    return (beta_tildes, ses, z_scores, p_values, se_inflation_factors)


def compute_beta_tildes(
    X,
    Y,
    *,
    y_var=None,
    scale_factors=None,
    mean_shifts=None,
    resid_correlation_matrix=None,
    calc_x_shift_scale_fn=None,
    finalize_regression_fn=None,
    bail_fn=None,
    log_fun=None,
    debug_level=0,
):
    if finalize_regression_fn is None:
        finalize_regression_fn = finalize_regression_outputs
    if bail_fn is None:
        bail_fn = _default_bail
    if log_fun is None:
        log_fun = lambda *args, **kwargs: None

    log_fun("Calculating beta tildes")

    if X.shape[0] == 0 or X.shape[1] == 0:
        bail_fn("Can't compute beta tildes on no gene sets!")

    if len(Y.shape) == 2:
        len_Y = Y.shape[1]
        Y_mean = np.mean(Y, axis=1, keepdims=True)
    else:
        len_Y = Y.shape[0]
        Y_mean = np.mean(Y)

    if mean_shifts is None or scale_factors is None:
        if calc_x_shift_scale_fn is None:
            if sparse.issparse(X):
                mean_shifts = np.array(X.mean(axis=0)).ravel()
                scale_factors = np.sqrt(np.array(X.multiply(X).mean(axis=0)).ravel() - np.square(mean_shifts))
            else:
                mean_shifts = np.mean(X, axis=0)
                scale_factors = np.std(X, axis=0)
        else:
            (mean_shifts, scale_factors) = calc_x_shift_scale_fn(X)

    if y_var is None:
        if len(Y.shape) == 1:
            y_var = np.var(Y)
        else:
            y_var = np.var(Y, axis=1)

    if sparse.issparse(X):
        X_sum = X.sum(axis=0).A1.T[:, np.newaxis]
    else:
        X_sum = np.asarray(X.sum(axis=0, keepdims=True).T)

    if len(Y.shape) == 1:
        X_sum = X_sum.squeeze(axis=1)

    dot_product = (X.T.dot(Y.T) - X_sum * Y_mean.T).T / len_Y

    variances = np.power(scale_factors, 2)
    variances[variances == 0] = 1

    beta_tildes = scale_factors * dot_product / variances

    if len(Y.shape) == 2:
        ses = np.outer(np.sqrt(y_var), scale_factors)
    else:
        ses = np.sqrt(y_var) * scale_factors

    if len_Y > 1:
        ses /= (np.sqrt(variances * (len_Y - 1)))

    se_inflation_factors = None
    if resid_correlation_matrix is not None:
        log_fun("Adjusting standard errors for correlations", debug_level)

        if type(resid_correlation_matrix) is list:
            resid_correlation_matrix_list = resid_correlation_matrix
            assert len(resid_correlation_matrix_list) == beta_tildes.shape[0]
        else:
            resid_correlation_matrix_list = [resid_correlation_matrix]

        se_inflation_factors = np.zeros(beta_tildes.shape)

        for i in range(len(resid_correlation_matrix_list)):
            r_X = resid_correlation_matrix_list[i].dot(X)
            if sparse.issparse(X):
                r_X_col_means = r_X.multiply(X).sum(axis=0).A1 / X.shape[0]
            else:
                r_X_col_means = np.sum(r_X * X, axis=0) / X.shape[0]

            cor_variances = r_X_col_means - np.square(r_X_col_means)
            cor_variances[cor_variances < variances] = variances[cor_variances < variances]
            cur_se_inflation_factors = np.sqrt(cor_variances / variances)

            if len(resid_correlation_matrix_list) == 1:
                se_inflation_factors = cur_se_inflation_factors
                if len(beta_tildes.shape) == 2:
                    se_inflation_factors = np.tile(se_inflation_factors, beta_tildes.shape[0]).reshape(beta_tildes.shape)
                break
            else:
                se_inflation_factors[i, :] = cur_se_inflation_factors

    return finalize_regression_fn(beta_tildes, ses, se_inflation_factors)


def compute_logistic_beta_tildes(
    X,
    Y,
    *,
    scale_factors=None,
    mean_shifts=None,
    resid_correlation_matrix=None,
    convert_to_dichotomous=True,
    rel_tol=0.01,
    X_stacked=None,
    append_pseudo=True,
    calc_x_shift_scale_fn=None,
    finalize_regression_fn=None,
    bail_fn=None,
    log_fun=None,
    debug_level=0,
    trace_level=0,
    runtime_Y=None,
    runtime_Y_for_regression=None,
):
    if finalize_regression_fn is None:
        finalize_regression_fn = finalize_regression_outputs
    if bail_fn is None:
        bail_fn = _default_bail
    if log_fun is None:
        log_fun = lambda *args, **kwargs: None

    log_fun("Calculating logistic beta tildes")

    if X.shape[0] == 0 or X.shape[1] == 0:
        bail_fn("Can't compute beta tildes on no gene sets!")

    if runtime_Y is not None and (Y is runtime_Y or Y is runtime_Y_for_regression):
        Y = copy.copy(Y)

    if mean_shifts is None or scale_factors is None:
        if calc_x_shift_scale_fn is None:
            if sparse.issparse(X):
                mean_shifts = np.array(X.mean(axis=0)).ravel()
                scale_factors = np.sqrt(np.array(X.multiply(X).mean(axis=0)).ravel() - np.square(mean_shifts))
            else:
                mean_shifts = np.mean(X, axis=0)
                scale_factors = np.std(X, axis=0)
        else:
            (mean_shifts, scale_factors) = calc_x_shift_scale_fn(X)

    if len(Y.shape) == 1:
        orig_vector = True
        Y = Y[np.newaxis, :]
    else:
        orig_vector = False

    if convert_to_dichotomous:
        if np.sum(np.logical_and(Y != 0, Y != 1)) > 0:
            Y[np.isnan(Y)] = 0
            mult_sum = 1
            Y_sums = np.sum(Y, axis=1).astype(int) * mult_sum
            Y_sorted = np.sort(Y, axis=1)[:, ::-1]
            threshold_val = np.diag(Y_sorted[:, Y_sums])

            true_mask = (Y.T > threshold_val).T
            Y[true_mask] = 1
            Y[~true_mask] = 0
            log_fun("Converting values to dichotomous outcomes; y=1 for input y > %s" % threshold_val, debug_level)

    log_fun("Outcomes: %d=1, %d=0; mean=%.3g" % (np.sum(Y == 1), np.sum(Y == 0), np.mean(Y)), trace_level)

    if np.var(Y) == 0:
        bail_fn("Error: need at least one sample with a different outcome")

    len_Y = Y.shape[1]
    num_chains = Y.shape[0]

    if append_pseudo:
        log_fun("Appending pseudo counts", trace_level)
        Y_means = np.mean(Y, axis=1)[:, np.newaxis]
        Y = np.hstack((Y, Y_means))
        X = sparse.csc_matrix(sparse.vstack((X, sparse.csr_matrix(np.ones((1, X.shape[1]))))))

        if X_stacked is not None:
            X_stacked = sparse.csc_matrix(sparse.vstack((X_stacked, sparse.csr_matrix(np.ones((1, X_stacked.shape[1]))))))

    if X_stacked is None:
        if num_chains > 1:
            X_stacked = sparse.hstack([X] * num_chains)
        else:
            X_stacked = X

    num_non_zero = np.tile((X != 0).sum(axis=0).A1, num_chains)
    num_zero = X_stacked.shape[0] - num_non_zero

    beta_tildes = np.zeros(X.shape[1] * num_chains)
    alpha_tildes = np.zeros(X.shape[1] * num_chains)
    it = 0
    compute_mask = np.full(len(beta_tildes), True)
    diverged_mask = np.full(len(beta_tildes), False)

    def __compute_Y_R(_X, _beta_tildes, _alpha_tildes, max_cap=0.999):
        exp_X_stacked_beta_alpha = _X.multiply(_beta_tildes)
        exp_X_stacked_beta_alpha.data += (_X != 0).multiply(_alpha_tildes).data
        max_val = 100
        overflow_mask = exp_X_stacked_beta_alpha.data > max_val
        exp_X_stacked_beta_alpha.data[overflow_mask] = max_val
        np.exp(exp_X_stacked_beta_alpha.data, out=exp_X_stacked_beta_alpha.data)

        Y_pred = copy.copy(exp_X_stacked_beta_alpha)
        Y_pred.data = Y_pred.data / (1 + Y_pred.data)
        Y_pred.data[Y_pred.data > max_cap] = max_cap
        R = copy.copy(Y_pred)
        R.data = Y_pred.data * (1 - Y_pred.data)
        return (Y_pred, R)

    def __compute_Y_R_zero(_alpha_tildes):
        Y_pred_zero = np.exp(_alpha_tildes)
        Y_pred_zero = Y_pred_zero / (1 + Y_pred_zero)
        R_zero = Y_pred_zero * (1 - Y_pred_zero)
        return (Y_pred_zero, R_zero)

    max_it = 100
    log_fun("Performing IRLS...")
    while True:
        it += 1
        prev_beta_tildes = copy.copy(beta_tildes)
        prev_alpha_tildes = copy.copy(alpha_tildes)

        (Y_pred, R) = __compute_Y_R(X_stacked[:, compute_mask], beta_tildes[compute_mask], alpha_tildes[compute_mask])

        max_val = 100
        overflow_mask = alpha_tildes > max_val
        alpha_tildes[overflow_mask] = max_val

        (Y_pred_zero, R_zero) = __compute_Y_R_zero(alpha_tildes[compute_mask])

        Y_sum_per_chain = np.sum(Y, axis=1)
        Y_sum = np.tile(Y_sum_per_chain, X.shape[1])

        X_r_X_beta = X_stacked[:, compute_mask].power(2).multiply(R).sum(axis=0).A1.ravel()
        X_r_X_alpha = R.sum(axis=0).A1.ravel() + R_zero * num_zero[compute_mask]
        X_r_X_beta_alpha = X_stacked[:, compute_mask].multiply(R).sum(axis=0).A1.ravel()
        denom = X_r_X_beta * X_r_X_alpha - np.square(X_r_X_beta_alpha)

        diverged = np.logical_or(np.logical_or(X_r_X_beta == 0, X_r_X_beta_alpha == 0), denom == 0)

        if np.sum(diverged) > 0:
            log_fun("%d beta_tildes diverged" % np.sum(diverged), trace_level)
            not_diverged = ~diverged
            cur_indices = np.where(compute_mask)[0]
            compute_mask[cur_indices[diverged]] = False
            diverged_mask[cur_indices[diverged]] = True

            Y_pred = sparse.csc_matrix(Y_pred)
            R = sparse.csc_matrix(R)
            Y_pred = Y_pred[:, not_diverged]
            R = R[:, not_diverged]
            Y_pred_zero = Y_pred_zero[not_diverged]
            R_zero = R_zero[not_diverged]
            X_r_X_beta = X_r_X_beta[not_diverged]
            X_r_X_alpha = X_r_X_alpha[not_diverged]
            X_r_X_beta_alpha = X_r_X_beta_alpha[not_diverged]
            denom = denom[not_diverged]

        if np.sum(np.isnan(X_r_X_beta) | np.isnan(X_r_X_alpha) | np.isnan(X_r_X_beta_alpha)) > 0:
            bail_fn("Error: something went wrong")

        R_inv_Y_T_beta = X_stacked[:, compute_mask].multiply(Y_pred).sum(axis=0).A1.ravel() - X.T.dot(Y.T).T.ravel()[compute_mask]
        R_inv_Y_T_alpha = (Y_pred.sum(axis=0).A1.ravel() + Y_pred_zero * num_zero[compute_mask]) - Y_sum[compute_mask]

        beta_tilde_row = (X_r_X_beta * prev_beta_tildes[compute_mask] + X_r_X_beta_alpha * prev_alpha_tildes[compute_mask] - R_inv_Y_T_beta)
        alpha_tilde_row = (X_r_X_alpha * prev_alpha_tildes[compute_mask] + X_r_X_beta_alpha * prev_beta_tildes[compute_mask] - R_inv_Y_T_alpha)

        beta_tildes[compute_mask] = (X_r_X_alpha * beta_tilde_row - X_r_X_beta_alpha * alpha_tilde_row) / denom
        alpha_tildes[compute_mask] = (X_r_X_beta * alpha_tilde_row - X_r_X_beta_alpha * beta_tilde_row) / denom

        diff = np.abs(beta_tildes - prev_beta_tildes)
        diff_denom = np.abs(beta_tildes + prev_beta_tildes)
        diff_denom[diff_denom == 0] = 1
        rel_diff = diff / diff_denom

        compute_mask[np.logical_or(rel_diff < rel_tol, beta_tildes == 0)] = False
        if np.sum(compute_mask) == 0:
            log_fun("Converged after %d iterations" % it, trace_level)
            break
        if it == max_it:
            log_fun("Stopping with %d still not converged" % np.sum(compute_mask), trace_level)
            diverged_mask[compute_mask] = True
            break

    while True:
        if np.sum(diverged_mask) > 0:
            beta_tildes[diverged_mask] = 0
            alpha_tildes[diverged_mask] = Y_sum[diverged_mask] / len_Y

        max_coeff = 100
        (Y_pred, V) = __compute_Y_R(X_stacked, beta_tildes, alpha_tildes)

        params_too_large_mask = np.logical_or(np.abs(alpha_tildes) > max_coeff, np.abs(beta_tildes) > max_coeff)
        alpha_tildes[np.abs(alpha_tildes) > max_coeff] = max_coeff

        p_const = np.exp(alpha_tildes) / (1 + np.exp(alpha_tildes))
        variance_denom = (V.sum(axis=0).A1 + p_const * (1 - p_const) * (len_Y - (X_stacked != 0).sum(axis=0).A1))
        denom_zero = variance_denom == 0
        variance_denom[denom_zero] = 1

        variances = X_stacked.power(2).multiply(V).sum(axis=0).A1 - np.power(X_stacked.multiply(V).sum(axis=0).A1, 2) / variance_denom
        variances[denom_zero] = 100

        additional_diverged_mask = np.logical_and(~diverged_mask, np.logical_or(np.logical_or(variances < 0, denom_zero), params_too_large_mask))
        if np.sum(additional_diverged_mask) > 0:
            diverged_mask = np.logical_or(diverged_mask, additional_diverged_mask)
        else:
            break

    se_inflation_factors = None
    if resid_correlation_matrix is not None:
        if type(resid_correlation_matrix) is list:
            raise NotImplementedError("Vectorized correlations not yet implemented for logistic regression")

        if append_pseudo:
            resid_correlation_matrix = sparse.hstack((resid_correlation_matrix, np.zeros(resid_correlation_matrix.shape[0])[:, np.newaxis]))
            new_bottom_row = np.zeros((1, resid_correlation_matrix.shape[1]))
            new_bottom_row[0, -1] = 1
            resid_correlation_matrix = sparse.vstack((resid_correlation_matrix, new_bottom_row)).tocsc()

        cor_variances = copy.copy(variances)
        r_X = resid_correlation_matrix.dot(X)
        r_X = (X != 0).multiply(r_X)

        cor_variances = sparse.hstack([r_X.multiply(X)] * num_chains).multiply(V).sum(axis=0).A1 - sparse.hstack([r_X] * num_chains).multiply(V).sum(axis=0).A1 / (V.sum(axis=0).A1 + p_const * (1 - p_const) * (len_Y - (X_stacked != 0).sum(axis=0).A1))
        variances[variances == 0] = 1
        se_inflation_factors = np.sqrt(cor_variances / variances)

    if num_chains > 1:
        beta_tildes = beta_tildes.reshape(num_chains, X.shape[1])
        alpha_tildes = alpha_tildes.reshape(num_chains, X.shape[1])
        variances = variances.reshape(num_chains, X.shape[1])
        diverged_mask = diverged_mask.reshape(num_chains, X.shape[1])
        if se_inflation_factors is not None:
            se_inflation_factors = se_inflation_factors.reshape(num_chains, X.shape[1])
    else:
        beta_tildes = beta_tildes[np.newaxis, :]
        alpha_tildes = alpha_tildes[np.newaxis, :]
        variances = variances[np.newaxis, :]
        diverged_mask = diverged_mask[np.newaxis, :]
        if se_inflation_factors is not None:
            se_inflation_factors = se_inflation_factors[np.newaxis, :]

    variances[:, scale_factors == 0] = 1
    beta_tildes = scale_factors * beta_tildes
    variances[variances == 0] = 1e-10
    ses = scale_factors / np.sqrt(variances)

    if orig_vector:
        beta_tildes = np.squeeze(beta_tildes, axis=0)
        alpha_tildes = np.squeeze(alpha_tildes, axis=0)
        variances = np.squeeze(variances, axis=0)
        ses = np.squeeze(ses, axis=0)
        diverged_mask = np.squeeze(diverged_mask, axis=0)

        if se_inflation_factors is not None:
            se_inflation_factors = np.squeeze(se_inflation_factors, axis=0)

    return finalize_regression_fn(beta_tildes, ses, se_inflation_factors) + (alpha_tildes, diverged_mask)


def compute_multivariate_beta_tildes(
    X,
    Y,
    *,
    resid_correlation_matrix=None,
    add_intercept=True,
    covs=None,
    finalize_regression_fn=None,
):
    if finalize_regression_fn is None:
        finalize_regression_fn = finalize_regression_outputs

    if covs is not None:
        if len(covs.shape) == 1:
            covs = covs[:, np.newaxis]
        X_design = np.hstack([X, covs])
    else:
        X_design = X

    if add_intercept:
        ones_col = np.ones((X_design.shape[0], 1))
        X_design = np.hstack([X_design, ones_col])

    n_obs, n_pred = X_design.shape
    n_phenos = Y.shape[0]
    Y_t = Y.T

    XtX = X_design.T @ X_design
    XtX_inv = np.linalg.inv(XtX)
    XtY = X_design.T @ Y_t
    betas = (XtX_inv @ XtY).T

    fitted = X_design @ betas.T
    residuals = Y_t - fitted

    df = n_obs - n_pred
    if df <= 0:
        raise ValueError("Degrees of freedom <= 0. Check the size of your input matrices.")

    sse = np.sum(residuals ** 2, axis=0)
    sigma2 = sse / df

    diag_xtx_inv = np.diag(XtX_inv)
    classical_ses = np.sqrt(sigma2[:, None] * diag_xtx_inv[None, :])
    final_ses = classical_ses.copy()

    if resid_correlation_matrix is not None:
        if len(resid_correlation_matrix) != n_phenos:
            raise ValueError("resid_correlation_matrix must be a list of length == n_phenos.")

        for p in range(n_phenos):
            R_p = resid_correlation_matrix[p]
            if sparse.issparse(R_p):
                XR_p = R_p.dot(X_design)
            else:
                XR_p = R_p @ X_design

            XtR_pX = X_design.T @ XR_p
            var_betas_p = XtX_inv @ XtR_pX @ XtX_inv
            final_ses[p, :] = np.sqrt(np.diag(var_betas_p))

    if covs is not None or add_intercept:
        n_factors = X.shape[1]
        betas = betas[:, :n_factors]
        final_ses = final_ses[:, :n_factors]

    return finalize_regression_fn(betas, final_ses, se_inflation_factors=None)
