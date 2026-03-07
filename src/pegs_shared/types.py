from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import scipy.sparse as sparse


MatrixLike = np.ndarray | sparse.spmatrix
VectorLike = np.ndarray
Callback = Callable[..., Any]


@dataclass
class XData:
    X_orig: MatrixLike | None = None
    X_orig_missing_genes: MatrixLike | None = None
    X_orig_missing_gene_sets: MatrixLike | None = None
    X_orig_missing_genes_missing_gene_sets: MatrixLike | None = None
    genes: list[str] = field(default_factory=list)
    genes_missing: list[str] = field(default_factory=list)
    gene_sets: list[str] = field(default_factory=list)
    gene_sets_missing: list[str] = field(default_factory=list)
    gene_sets_ignored: list[str] = field(default_factory=list)
    gene_to_ind: dict[str, int] = field(default_factory=dict)
    gene_set_to_ind: dict[str, int] = field(default_factory=dict)
    scale_factors: VectorLike | None = None
    mean_shifts: VectorLike | None = None
    gene_set_batches: VectorLike | None = None
    gene_set_labels: VectorLike | None = None
    is_dense_gene_set: VectorLike | None = None

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
        from pegs_utils import (
            initialize_filtered_gene_set_state,
            initialize_read_x_batch_seed_state,
            maybe_prepare_filtered_correlation,
            resolve_read_x_run_logistic,
            run_read_x_ingestion,
        )

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
            maybe_prepare_filtered_correlation(
                runtime=runtime,
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
    initial_ps: Any
    X_ins: list
    batches: list
    labels: list
    orig_files: list
    is_dense: list


@dataclass
class XReadConfig:
    x_sparsify: Any
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
    sparse_module: Any
    np_module: Any
    normalize_dense_gene_rows_fn: Callback
    build_sparse_x_from_dense_input_fn: Callback
    reindex_x_rows_to_current_genes_fn: Callback
    normalize_gene_set_weights_fn: Callback
    partition_missing_gene_rows_fn: Callback
    maybe_permute_gene_set_rows_fn: Callback
    maybe_prefilter_x_block_fn: Callback
    merge_missing_gene_rows_fn: Callback
    finalize_added_x_block_fn: Callback


@dataclass
class XReadIngestionOptions:
    batch_all_for_hyper: bool
    first_for_hyper: bool
    update_hyper_sigma: bool
    update_hyper_p: bool
    first_for_sigma_cond: bool
    run_corrected_ols: bool
    gene_cor_file: Any
    gene_loc_file: Any
    gene_cor_file_gene_col: Any
    gene_cor_file_cor_start_col: Any
    run_logistic: bool
    max_for_linear: float
    only_ids: Any
    add_all_genes: bool
    only_inc_genes: Any
    fraction_inc_genes: Any
    ignore_genes: Any
    max_num_entries_at_once: Any
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
    weighted_prune_gene_sets: Any
    prune_deterministically: bool
    max_num_gene_sets_initial: Any
    skip_betas: bool
    initial_p: float
    update_hyper_p: bool
    sigma_power: Any
    initial_sigma2_cond: Any
    update_hyper_sigma: bool
    initial_sigma2: Any
    sigma_soft_threshold_95: Any
    sigma_soft_threshold_5: Any
    batches: list
    num_ignored_gene_sets: Any
    first_for_hyper: bool
    max_num_gene_sets_hyper: Any
    first_for_sigma_cond: bool
    first_max_p_for_hyper: bool
    max_num_burn_in: Any
    max_num_iter_betas: int
    min_num_iter_betas: int
    num_chains_betas: int
    r_threshold_burn_in_betas: float
    use_max_r_for_convergence_betas: bool
    max_frac_sem_betas: float
    max_allowed_batch_correlation: Any
    sigma_num_devs_to_top: float
    p_noninf_inflate: float
    sparse_solution: bool
    sparse_frac_betas: Any
    betas_trace_out: Any
    increase_filter_gene_set_p: float
    min_gene_set_size: int
    max_gene_set_size: int
    filter_gene_set_metric_z: float
    max_num_gene_sets: Any


@dataclass
class XReadPostCallbacks:
    standardize_qc_metrics_after_x_read_fn: Callback
    maybe_correct_gene_set_betas_after_x_read_fn: Callback
    maybe_limit_initial_gene_sets_by_p_fn: Callback
    maybe_prune_gene_sets_after_x_read_fn: Callback
    initialize_hyper_defaults_after_x_read_fn: Callback
    maybe_learn_batch_hyper_after_x_read_fn: Callback
    maybe_adjust_overaggressive_p_filter_after_x_read_fn: Callback
    apply_post_read_gene_set_size_and_qc_filters_fn: Callback
    maybe_filter_zero_uncorrected_betas_after_x_read_fn: Callback
    maybe_reduce_gene_sets_to_max_after_x_read_fn: Callback
    record_read_x_counts_fn: Callback


@dataclass
class ReadXPipelineConfig:
    X_in: Any = None
    Xd_in: Any = None
    X_list: Any = None
    Xd_list: Any = None
    V_in: Any = None
    skip_V: bool = True
    force_reread: bool = False
    min_gene_set_size: int = 1
    max_gene_set_size: int = 30000
    only_ids: Any = None
    only_inc_genes: Any = None
    fraction_inc_genes: Any = None
    add_all_genes: bool = False
    prune_gene_sets: float = 0.8
    weighted_prune_gene_sets: Any = None
    prune_deterministically: bool = False
    x_sparsify: list = field(default_factory=lambda: [50, 100, 200, 500, 1000])
    add_ext: bool = False
    add_top: bool = True
    add_bottom: bool = True
    filter_negative: bool = True
    threshold_weights: float = 0.5
    cap_weights: bool = True
    permute_gene_sets: bool = False
    max_gene_set_p: Any = None
    filter_gene_set_p: float = 1
    filter_using_phewas: bool = False
    increase_filter_gene_set_p: float = 0.01
    max_num_gene_sets_initial: Any = None
    max_num_gene_sets: Any = None
    max_num_gene_sets_hyper: Any = None
    skip_betas: bool = False
    run_logistic: bool = True
    max_for_linear: float = 0.95
    filter_gene_set_metric_z: float = 2.5
    initial_p: Any = 0.01
    xin_to_p_noninf_ind: Any = None
    initial_sigma2: Any = 1e-3
    initial_sigma2_cond: Any = None
    sigma_power: Any = 0
    sigma_soft_threshold_95: Any = None
    sigma_soft_threshold_5: Any = None
    run_corrected_ols: bool = False
    correct_betas_mean: bool = True
    correct_betas_var: bool = True
    gene_loc_file: Any = None
    gene_cor_file: Any = None
    gene_cor_file_gene_col: int = 1
    gene_cor_file_cor_start_col: int = 10
    update_hyper_p: bool = False
    update_hyper_sigma: bool = False
    batch_all_for_hyper: bool = False
    first_for_hyper: bool = False
    first_max_p_for_hyper: bool = False
    first_for_sigma_cond: bool = False
    sigma_num_devs_to_top: float = 2.0
    p_noninf_inflate: float = 1
    batch_separator: str = "@"
    ignore_genes: set = field(default_factory=lambda: set(["NA"]))
    file_separator: Any = None
    max_num_burn_in: Any = None
    max_num_iter_betas: int = 1100
    min_num_iter_betas: int = 10
    num_chains_betas: int = 10
    r_threshold_burn_in_betas: float = 1.01
    use_max_r_for_convergence_betas: bool = True
    max_frac_sem_betas: float = 0.01
    max_allowed_batch_correlation: Any = None
    sparse_solution: bool = False
    sparse_frac_betas: Any = None
    betas_trace_out: Any = None
    show_progress: bool = True
    max_num_entries_at_once: Any = None


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
    gene_in_combined: Any
    gene_in_priors: Any


@dataclass
class ParsedGeneCovariates:
    cov_names: list
    gene_to_covs: dict


@dataclass
class AlignedGeneBfs:
    gene_bfs: Any
    extra_genes: list
    extra_gene_bfs: Any
    gene_in_combined: Any
    gene_in_priors: Any


@dataclass
class AlignedGeneCovariates:
    cov_names: list
    gene_covs: Any
    extra_genes: list
    extra_gene_covs: Any


@dataclass
class ParsedGenePhewasBfs:
    phenos: list
    pheno_to_ind: dict
    row: Any
    col: Any
    Ys: Any
    combineds: Any
    priors: Any
    num_filtered: int


@dataclass
class PhewasFileColumnInfo:
    id_col: int
    pheno_col: int
    bf_col: Any
    combined_col: Any
    prior_col: Any


@dataclass
class YData:
    Y: Any = None
    Y_for_regression: Any = None
    Y_exomes: Any = None
    Y_positive_controls: Any = None
    Y_case_counts: Any = None
    y_var: Any = None
    y_corr: Any = None
    y_corr_sparse: Any = None


@dataclass
class HyperparameterData:
    p: Any = None
    sigma2: Any = None
    sigma_power: Any = None
    sigma2_osc: Any = None
    sigma2_se: Any = None
    sigma2_p: Any = None
    sigma2_total_var: Any = None
    sigma2_total_var_lower: Any = None
    sigma2_total_var_upper: Any = None
    ps: Any = None
    sigma2s: Any = None
    sigma2s_missing: Any = None

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
        if sigma2_p is not None:
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
    phenos: Any = None
    pheno_to_ind: Any = None
    gene_pheno_Y: Any = None
    gene_pheno_combined_prior_Ys: Any = None
    gene_pheno_priors: Any = None
    X_phewas_beta: Any = None
    X_phewas_beta_uncorrected: Any = None
    num_gene_phewas_filtered: int = 0
    anchor_gene_mask: Any = None
    anchor_pheno_mask: Any = None


@dataclass
class FactorInputData:
    anchor_gene_mask: Any = None
    anchor_pheno_mask: Any = None
    loaded_gene_set_phewas_stats: bool = False
    loaded_gene_phewas_bfs: bool = False


@dataclass
class PhewasStageConfig:
    gene_phewas_bfs_in: Any = None
    gene_phewas_bfs_id_col: Any = None
    gene_phewas_bfs_pheno_col: Any = None
    gene_phewas_bfs_log_bf_col: Any = None
    gene_phewas_bfs_combined_col: Any = None
    gene_phewas_bfs_prior_col: Any = None
    max_num_burn_in: int = 1000
    max_num_iter: int = 1100
    min_num_iter: int = 10
    num_chains: int = 10
    r_threshold_burn_in: float = 1.01
    use_max_r_for_convergence: bool = True
    max_frac_sem: float = 0.01
    gauss_seidel: bool = False
    sparse_solution: bool = False
    sparse_frac_betas: Any = None
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


@dataclass
class PhewasInputResolution:
    requested_input: Any = None
    resolved_input: Any = None
    mode: str = "skip"
    reason: str = "no_input_requested"

    @property
    def should_reuse_loaded_matrix(self):
        return self.mode == "reuse_loaded_matrix"

    @property
    def should_reread_file(self):
        return self.mode == "re_read_file"
