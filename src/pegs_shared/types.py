from __future__ import annotations

from dataclasses import MISSING, dataclass, field, fields
from os import PathLike
from types import ModuleType
from typing import Callable, Literal, TypeAlias

import numpy as np
import scipy.sparse as sparse

from pegs_shared.x_runtime import (
    initialize_filtered_gene_set_state,
    initialize_read_x_batch_seed_state,
    is_metric_qc_filter_active,
    maybe_prepare_filtered_correlation,
    resolve_read_x_run_logistic,
    run_read_x_ingestion,
)


MatrixLike: TypeAlias = np.ndarray | sparse.spmatrix
VectorLike: TypeAlias = np.ndarray
PathLikeStr: TypeAlias = str | PathLike[str]
ColumnSpec: TypeAlias = int | str | None
StringList: TypeAlias = list[str]
OptionalStringList: TypeAlias = list[str] | None
BatchList: TypeAlias = list[str | None]
BoolList: TypeAlias = list[bool]
IndexMap: TypeAlias = dict[str, int]
FloatMap: TypeAlias = dict[str, float]
ChromMap: TypeAlias = dict[str, str]
PositionMap: TypeAlias = dict[str, tuple[int, int]]
ChromGenePosMap: TypeAlias = dict[str, dict[str, set[int]]]
DenseBlockCache: TypeAlias = tuple[MatrixLike, bool, bool, int | None, int | None, str | None]
OptionalVectorLike: TypeAlias = VectorLike | None
OptionalMatrixLike: TypeAlias = MatrixLike | None
OptionalPathLikeStr: TypeAlias = PathLikeStr | None
NumericScalar: TypeAlias = int | float
Callback = Callable[..., object]
PhewasInputMode: TypeAlias = Literal["skip", "reuse_loaded_matrix", "re_read_file"]
PhewasInputReason: TypeAlias = Literal[
    "no_input_requested",
    "matrix_not_loaded",
    "loaded_matrix_filtered",
    "requested_input_matches_loaded_source",
    "requested_input_not_reusable",
]


@dataclass
class XData:
    batch_size: int | None = None
    X_orig: OptionalMatrixLike = None
    X_binary_packed: OptionalMatrixLike = None
    X_orig_missing_genes: OptionalMatrixLike = None
    X_orig_missing_gene_sets: OptionalMatrixLike = None
    X_orig_ignored_gene_sets: OptionalMatrixLike = None
    X_orig_missing_genes_missing_gene_sets: OptionalMatrixLike = None
    last_X_block: DenseBlockCache | None = None
    genes: OptionalStringList = field(default_factory=list)
    genes_missing: OptionalStringList = field(default_factory=list)
    gene_sets: OptionalStringList = field(default_factory=list)
    gene_sets_missing: OptionalStringList = field(default_factory=list)
    gene_sets_ignored: OptionalStringList = field(default_factory=list)
    gene_set_filter_reason_missing: OptionalStringList = field(default_factory=list)
    gene_set_filter_reason_ignored: OptionalStringList = field(default_factory=list)
    gene_set_track_beta_uncorrected_ignored: OptionalVectorLike = None
    gene_to_ind: IndexMap | None = field(default_factory=dict)
    gene_missing_to_ind: IndexMap | None = field(default_factory=dict)
    gene_set_to_ind: IndexMap | None = field(default_factory=dict)
    scale_is_for_whitened: bool = False
    scale_factors: OptionalVectorLike = None
    mean_shifts: OptionalVectorLike = None
    scale_factors_missing: OptionalVectorLike = None
    mean_shifts_missing: OptionalVectorLike = None
    scale_factors_ignored: OptionalVectorLike = None
    mean_shifts_ignored: OptionalVectorLike = None
    gene_set_batches: OptionalVectorLike = None
    gene_set_batches_missing: OptionalVectorLike = None
    gene_set_labels: OptionalVectorLike = None
    gene_set_labels_missing: OptionalVectorLike = None
    gene_set_labels_ignored: OptionalVectorLike = None
    is_dense_gene_set: OptionalVectorLike = None
    is_dense_gene_set_missing: OptionalVectorLike = None
    is_dense_gene_set_ignored: OptionalVectorLike = None
    gene_chrom_name_pos: ChromGenePosMap | None = None
    gene_to_chrom: ChromMap | None = None
    gene_to_pos: PositionMap | None = None
    gene_to_gwas_huge_score: FloatMap | None = None
    gene_to_gwas_huge_score_uncorrected: FloatMap | None = None
    gene_to_exomes_huge_score: FloatMap | None = None
    gene_to_huge_score: FloatMap | None = None

    @classmethod
    def from_input_plan(cls, input_plan):
        return cls(
            gene_set_batches=np.array(input_plan.batches),
            gene_set_labels=np.array(input_plan.labels),
            is_dense_gene_set=np.array(input_plan.is_dense, dtype=bool),
        )

    @classmethod
    def from_runtime(cls, runtime):
        payload = {}
        for data_field in fields(cls):
            if hasattr(runtime, data_field.name):
                payload[data_field.name] = getattr(runtime, data_field.name)
            elif data_field.default is not MISSING:
                payload[data_field.name] = data_field.default
            elif data_field.default_factory is not MISSING:
                payload[data_field.name] = data_field.default_factory()
            else:
                payload[data_field.name] = None
        return cls(**payload)

    @classmethod
    def initialized_runtime_state(cls, batch_size):
        return cls(
            batch_size=batch_size,
            scale_is_for_whitened=False,
            genes=None,
            genes_missing=None,
            gene_sets=None,
            gene_sets_missing=None,
            gene_sets_ignored=None,
            gene_to_ind=None,
            gene_missing_to_ind=None,
            gene_set_to_ind=None,
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

    def apply_to_runtime(self, runtime):
        for data_field in fields(type(self)):
            setattr(runtime, data_field.name, getattr(self, data_field.name))
        runtime.x_state = self
        return self

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
            (
                ingestion_options.filter_gene_set_p < 1
                or is_metric_qc_filter_active(ingestion_options.filter_gene_set_metric_z)
            )
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

        sort_rank = None
        if runtime.p_values is not None:
            try:
                p_values = np.asarray(runtime.p_values, dtype=float)
            except (TypeError, ValueError):
                p_values = None
            if p_values is not None and p_values.size > 0:
                finite_p_values = np.where(np.isfinite(p_values), p_values, 1.0)
                sort_rank = -np.sqrt(-np.log(finite_p_values + 1e-200))
        sort_rank = post_callbacks.maybe_filter_zero_uncorrected_betas_after_x_read_fn(
            runtime,
            sort_rank=sort_rank,
            skip_betas=post_options.skip_betas,
            filter_gene_set_p=post_options.filter_gene_set_p,
            filter_using_phewas=post_options.filter_using_phewas,
            retain_all_beta_uncorrected=post_options.retain_all_beta_uncorrected,
            independent_betas_only=post_options.independent_betas_only,
            track_filtered_beta_uncorrected=post_options.track_filtered_beta_uncorrected,
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
            retain_all_beta_uncorrected=post_options.retain_all_beta_uncorrected,
            independent_betas_only=post_options.independent_betas_only,
            track_filtered_beta_uncorrected=post_options.track_filtered_beta_uncorrected,
        )
        post_callbacks.record_read_x_counts_fn(
            runtime,
            record_param_fn=runtime._record_param,
            log_fn=lambda message: log_fn(message),
        )
        return True


@dataclass
class XInputPlan:
    initial_ps: list[float] | None
    X_ins: StringList
    batches: BatchList
    labels: StringList
    orig_files: StringList
    is_dense: BoolList


@dataclass
class XReadConfig:
    x_sparsify: list[int]
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
    sparse_module: ModuleType
    np_module: ModuleType
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
    gene_cor_file: OptionalPathLikeStr
    gene_loc_file: OptionalPathLikeStr
    gene_cor_file_gene_col: int
    gene_cor_file_cor_start_col: int
    run_logistic: bool
    max_for_linear: float
    only_ids: set[str] | None
    add_all_genes: bool
    only_inc_genes: set[str] | None
    fraction_inc_genes: float | None
    ignore_genes: set[str]
    max_num_entries_at_once: int | None
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
    weighted_prune_gene_sets: float | None
    prune_deterministically: bool
    max_num_gene_sets_initial: int | None
    skip_betas: bool
    initial_p: float
    update_hyper_p: bool
    sigma_power: float | None
    initial_sigma2_cond: float | None
    update_hyper_sigma: bool
    initial_sigma2: float | None
    sigma_soft_threshold_95: float | None
    sigma_soft_threshold_5: float | None
    batches: BatchList
    num_ignored_gene_sets: int
    first_for_hyper: bool
    max_num_gene_sets_hyper: int | None
    first_for_sigma_cond: bool
    first_max_p_for_hyper: bool
    max_num_burn_in: int | None
    max_num_iter_betas: int
    min_num_iter_betas: int
    num_chains_betas: int
    r_threshold_burn_in_betas: float
    use_max_r_for_convergence_betas: bool
    max_frac_sem_betas: float
    max_allowed_batch_correlation: float | None
    sigma_num_devs_to_top: float
    p_noninf_inflate: float
    sparse_solution: bool
    sparse_frac_betas: float | None
    betas_trace_out: OptionalPathLikeStr
    increase_filter_gene_set_p: float
    min_gene_set_size: int
    max_gene_set_size: int
    filter_gene_set_metric_z: float
    max_num_gene_sets: int | None
    retain_all_beta_uncorrected: bool
    independent_betas_only: bool
    track_filtered_beta_uncorrected: bool


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
    X_in: StringList | str | None = None
    Xd_in: StringList | str | None = None
    X_list: StringList | str | None = None
    Xd_list: StringList | str | None = None
    V_in: OptionalPathLikeStr = None
    skip_V: bool = True
    force_reread: bool = False
    min_gene_set_size: int = 1
    max_gene_set_size: int = 30000
    only_ids: set[str] | None = None
    only_inc_genes: set[str] | None = None
    fraction_inc_genes: float | None = None
    add_all_genes: bool = False
    prune_gene_sets: float = 0.8
    weighted_prune_gene_sets: float | None = None
    prune_deterministically: bool = False
    x_sparsify: list[int] = field(default_factory=lambda: [50, 100, 200, 500, 1000])
    add_ext: bool = False
    add_top: bool = True
    add_bottom: bool = True
    filter_negative: bool = True
    threshold_weights: float = 0.5
    cap_weights: bool = True
    permute_gene_sets: bool = False
    max_gene_set_p: float | None = None
    filter_gene_set_p: float = 1
    filter_using_phewas: bool = False
    increase_filter_gene_set_p: float = 0.01
    max_num_gene_sets_initial: int | None = None
    max_num_gene_sets: int | None = None
    max_num_gene_sets_hyper: int | None = None
    skip_betas: bool = False
    run_logistic: bool = True
    max_for_linear: float = 0.95
    filter_gene_set_metric_z: float = 2.5
    initial_p: float | list[float] | None = 0.01
    xin_to_p_noninf_ind: dict[str, int] | None = None
    initial_sigma2: float | None = 1e-3
    initial_sigma2_cond: float | None = None
    sigma_power: float | None = 0
    sigma_soft_threshold_95: float | None = None
    sigma_soft_threshold_5: float | None = None
    run_corrected_ols: bool = False
    correct_betas_mean: bool = True
    correct_betas_var: bool = True
    gene_loc_file: OptionalPathLikeStr = None
    gene_cor_file: OptionalPathLikeStr = None
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
    x_list_unlabeled_batching: str = "per_file"
    ignore_genes: set[str] = field(default_factory=lambda: set(["NA"]))
    file_separator: str | None = None
    max_num_burn_in: int | None = None
    max_num_iter_betas: int = 1100
    min_num_iter_betas: int = 10
    num_chains_betas: int = 10
    r_threshold_burn_in_betas: float = 1.01
    use_max_r_for_convergence_betas: bool = True
    max_frac_sem_betas: float = 0.01
    max_allowed_batch_correlation: float | None = None
    sparse_solution: bool = False
    sparse_frac_betas: float | None = None
    betas_trace_out: OptionalPathLikeStr = None
    show_progress: bool = True
    max_num_entries_at_once: int | None = None
    retain_all_beta_uncorrected: bool = False
    independent_betas_only: bool = False
    track_filtered_beta_uncorrected: bool = False


@dataclass
class ParsedGeneSetStats:
    need_to_take_log: bool
    has_beta_tilde: bool
    has_p_or_se: bool
    has_beta: bool
    has_beta_uncorrected: bool
    records: dict[str, dict[str, float]]


@dataclass
class ParsedGeneBfs:
    gene_in_bfs: FloatMap
    gene_in_combined: FloatMap | None
    gene_in_priors: FloatMap | None


@dataclass
class ParsedGeneCovariates:
    cov_names: StringList
    gene_to_covs: dict[str, list[float]]


@dataclass
class AlignedGeneBfs:
    gene_bfs: VectorLike
    extra_genes: StringList
    extra_gene_bfs: OptionalVectorLike
    gene_in_combined: OptionalVectorLike
    gene_in_priors: OptionalVectorLike


@dataclass
class AlignedGeneCovariates:
    cov_names: StringList
    gene_covs: MatrixLike
    extra_genes: StringList
    extra_gene_covs: OptionalMatrixLike


@dataclass
class ParsedGenePhewasBfs:
    phenos: StringList
    pheno_to_ind: IndexMap
    row: list[int]
    col: list[int]
    Ys: list[float]
    combineds: list[float] | None
    priors: list[float] | None
    num_filtered: int


@dataclass
class PhewasFileColumnInfo:
    id_col: int
    pheno_col: int
    bf_col: ColumnSpec
    combined_col: ColumnSpec
    prior_col: ColumnSpec


@dataclass
class YData:
    Y: OptionalVectorLike = None
    Y_for_regression: OptionalVectorLike = None
    Y_exomes: OptionalVectorLike = None
    Y_positive_controls: OptionalVectorLike = None
    Y_case_counts: OptionalVectorLike = None
    y_var: NumericScalar | None = None
    y_corr: OptionalMatrixLike = None
    y_corr_sparse: sparse.spmatrix | None = None

    @classmethod
    def from_runtime(cls, runtime):
        return cls(
            Y=getattr(runtime, "Y", None),
            Y_for_regression=getattr(runtime, "Y_for_regression", None),
            Y_exomes=getattr(runtime, "Y_exomes", None),
            Y_positive_controls=getattr(runtime, "Y_positive_controls", None),
            Y_case_counts=getattr(runtime, "Y_case_counts", None),
            y_var=getattr(runtime, "y_var", None),
            y_corr=getattr(runtime, "y_corr", None),
            y_corr_sparse=getattr(runtime, "y_corr_sparse", None),
        )

    @classmethod
    def from_inputs(
        cls,
        runtime,
        Y,
        Y_for_regression=None,
        Y_exomes=None,
        Y_positive_controls=None,
        Y_case_counts=None,
        Y_corr_m=None,
        store_corr_sparse=False,
        min_correlation=0,
    ):
        y_data = cls.from_runtime(runtime)
        if Y_corr_m is not None:
            y_corr_m = np.array(Y_corr_m, copy=True)
            if min_correlation is not None:
                y_corr_m[y_corr_m <= 0] = 0

            keep_mask = np.array([True] * len(y_corr_m))
            for i in range(len(y_corr_m) - 1, -1, -1):
                if np.sum(y_corr_m[i] != 0) == 0:
                    keep_mask[i] = False
                else:
                    break
            if np.sum(keep_mask) > 0:
                y_corr_m = y_corr_m[keep_mask]

            y_data.y_corr = np.array(y_corr_m, copy=True)

            y_corr_diags = [y_data.y_corr[i, :(len(y_data.y_corr[i, :]) - i)] for i in range(len(y_data.y_corr))]
            y_corr_sparse = sparse.csc_matrix(
                sparse.diags(
                    y_corr_diags + y_corr_diags[1:],
                    list(range(len(y_corr_diags))) + list(range(-1, -len(y_corr_diags), -1)),
                )
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

    def apply_to_runtime(self, runtime):
        runtime.Y = self.Y
        runtime.Y_for_regression = self.Y_for_regression
        runtime.Y_exomes = self.Y_exomes
        runtime.Y_positive_controls = self.Y_positive_controls
        runtime.Y_case_counts = self.Y_case_counts
        runtime.y_var = self.y_var
        runtime.y_corr = self.y_corr
        runtime.y_corr_sparse = self.y_corr_sparse
        if getattr(runtime, "runtime_state_bundle", None) is not None:
            runtime.runtime_state_bundle.y_state = self
        runtime.y_state = self
        return self


@dataclass
class HyperparameterData:
    p: NumericScalar | None = None
    sigma2: NumericScalar | None = None
    sigma_power: NumericScalar | None = None
    sigma2_osc: NumericScalar | None = None
    sigma2_se: NumericScalar | None = None
    sigma2_p: NumericScalar | None = None
    sigma2_total_var: NumericScalar | None = None
    sigma2_total_var_lower: NumericScalar | None = None
    sigma2_total_var_upper: NumericScalar | None = None
    ps: OptionalVectorLike = None
    sigma2s: OptionalVectorLike = None
    sigma2s_missing: OptionalVectorLike = None

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
        if getattr(runtime, "runtime_state_bundle", None) is not None:
            runtime.runtime_state_bundle.hyperparameter_state = self
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
    phenos: OptionalStringList = None
    pheno_to_ind: IndexMap | None = None
    gene_pheno_Y: OptionalMatrixLike = None
    gene_pheno_combined_prior_Ys: OptionalMatrixLike = None
    gene_pheno_priors: OptionalMatrixLike = None
    X_phewas_beta: OptionalMatrixLike = None
    X_phewas_beta_uncorrected: OptionalMatrixLike = None
    num_gene_phewas_filtered: int = 0
    anchor_gene_mask: OptionalVectorLike = None
    anchor_pheno_mask: OptionalVectorLike = None

    @classmethod
    def from_runtime(cls, runtime):
        return cls(
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

    def apply_to_runtime(self, runtime):
        runtime.phenos = self.phenos
        runtime.pheno_to_ind = self.pheno_to_ind
        runtime.gene_pheno_Y = self.gene_pheno_Y
        runtime.gene_pheno_combined_prior_Ys = self.gene_pheno_combined_prior_Ys
        runtime.gene_pheno_priors = self.gene_pheno_priors
        runtime.X_phewas_beta = self.X_phewas_beta
        runtime.X_phewas_beta_uncorrected = self.X_phewas_beta_uncorrected
        runtime.num_gene_phewas_filtered = self.num_gene_phewas_filtered
        runtime.anchor_gene_mask = self.anchor_gene_mask
        runtime.anchor_pheno_mask = self.anchor_pheno_mask
        if getattr(runtime, "runtime_state_bundle", None) is not None:
            runtime.runtime_state_bundle.phewas_state = self
        runtime.phewas_state = self
        return self


@dataclass
class RuntimeStateBundle:
    y_state: YData
    hyperparameter_state: HyperparameterData
    phewas_state: PhewasRuntimeState

    @classmethod
    def from_runtime(cls, runtime):
        return cls(
            y_state=YData.from_runtime(runtime),
            hyperparameter_state=HyperparameterData.from_runtime(runtime),
            phewas_state=PhewasRuntimeState.from_runtime(runtime),
        )

    def apply_to_runtime(self, runtime):
        self.y_state.apply_to_runtime(runtime)
        self.hyperparameter_state.apply_to_runtime(runtime)
        self.phewas_state.apply_to_runtime(runtime)
        runtime.y_state = self.y_state
        runtime.hyperparameter_state = self.hyperparameter_state
        runtime.phewas_state = self.phewas_state
        runtime.runtime_state_bundle = self
        return self


@dataclass
class FactorInputData:
    anchor_gene_mask: OptionalVectorLike = None
    anchor_pheno_mask: OptionalVectorLike = None
    loaded_gene_set_phewas_stats: bool = False
    loaded_gene_phewas_bfs: bool = False


@dataclass
class PhewasStageConfig:
    gene_phewas_bfs_in: OptionalPathLikeStr = None
    gene_phewas_bfs_id_col: ColumnSpec = None
    gene_phewas_bfs_pheno_col: ColumnSpec = None
    gene_phewas_bfs_log_bf_col: ColumnSpec = None
    gene_phewas_bfs_combined_col: ColumnSpec = None
    gene_phewas_bfs_prior_col: ColumnSpec = None
    min_value: float | None = None
    phewas_comparison_set: str = "matched"
    max_num_burn_in: int = 1000
    max_num_iter: int = 1100
    min_num_iter: int = 10
    num_chains: int = 10
    r_threshold_burn_in: float = 1.01
    use_max_r_for_convergence: bool = True
    max_frac_sem: float = 0.01
    gauss_seidel: bool = False
    sparse_solution: bool = False
    sparse_frac_betas: float | None = None
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
            "min_value": self.min_value,
            "phewas_comparison_set": self.phewas_comparison_set,
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
    requested_input: OptionalPathLikeStr = None
    resolved_input: OptionalPathLikeStr = None
    mode: PhewasInputMode = "skip"
    reason: PhewasInputReason = "no_input_requested"

    @property
    def should_reuse_loaded_matrix(self) -> bool:
        return self.mode == "reuse_loaded_matrix"

    @property
    def should_reread_file(self) -> bool:
        return self.mode == "re_read_file"
