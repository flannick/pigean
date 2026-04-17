from __future__ import annotations

import csv
import re
from dataclasses import dataclass, field

import numpy as np

from . import gene_list_inputs as eaggl_gene_list_inputs
from . import factor_runtime as eaggl_factor_runtime
from . import phewas as eaggl_phewas


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
    loaded_gene_set_phewas_stats: bool = False
    loaded_gene_phewas_bfs: bool = False


@dataclass
class FactorExecutionConfig:
    max_num_factors: int
    phi: float
    alpha0: float
    beta0: float
    gene_set_filter_type: str | None = None
    gene_or_pheno_filter_type: str | None = None
    learn_phi: bool = False
    learn_phi_max_redundancy: float = 0.5
    learn_phi_max_redundancy_q90: float = 0.35
    learn_phi_runs_per_step: int = 1
    learn_phi_min_run_support: float = 0.6
    learn_phi_min_stability: float = 0.85
    learn_phi_max_fit_loss_frac: float = 0.05
    learn_phi_k_band_frac: float = 0.9
    learn_phi_max_steps: int = 5
    learn_phi_expand_factor: float = 2.0
    learn_phi_weight_floor: float | None = None
    learn_phi_mass_floor_frac: float = 0.005
    learn_phi_min_error_gain_per_factor: float = 5.0
    learn_phi_only: bool = False
    learn_phi_report_out: str | None = None
    factor_phi_metrics_out: str | None = None
    factor_backend: str = "full"
    learn_phi_backend: str = "sentinel_pruned"
    blockwise_gene_set_block_size: int = 5000
    blockwise_epochs: int = 3
    blockwise_shuffle_blocks: bool = True
    blockwise_warm_start: bool = True
    blockwise_max_blocks: int | None = None
    blockwise_report_out: str | None = None
    factors_out: str | None = None
    factor_metrics_out: str | None = None
    gene_set_clusters_out: str | None = None
    gene_clusters_out: str | None = None
    learn_phi_prune_genes_num: int | None = 1000
    learn_phi_prune_gene_sets_num: int | None = 1000
    learn_phi_max_num_iterations: int | None = None
    seed: int | None = None
    factor_runs: int = 1
    consensus_nmf: bool = False
    consensus_min_factor_cosine: float = 0.7
    consensus_min_run_support: float = 0.5
    consensus_aggregation: str = "median"
    consensus_stats_out: str | None = None
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
    max_num_iterations: int = 100
    rel_tol: float = 1e-4
    min_lambda_threshold: float = 1e-3
    lmm_auth_key: object = None
    lmm_model: object = None
    lmm_provider: str = "openai"
    label_gene_sets_only: bool = False
    label_include_phenos: bool = False
    label_individually: bool = False
    keep_original_loadings: bool = False
    project_phenos_from_gene_sets: bool = False
    pheno_capture_input: str = "weighted_thresholded"

    def to_run_kwargs(self):
        return {
            "max_num_factors": self.max_num_factors,
            "phi": self.phi,
            "learn_phi": self.learn_phi,
            "learn_phi_max_redundancy": self.learn_phi_max_redundancy,
            "learn_phi_max_redundancy_q90": self.learn_phi_max_redundancy_q90,
            "learn_phi_runs_per_step": self.learn_phi_runs_per_step,
            "learn_phi_min_run_support": self.learn_phi_min_run_support,
            "learn_phi_min_stability": self.learn_phi_min_stability,
            "learn_phi_max_fit_loss_frac": self.learn_phi_max_fit_loss_frac,
            "learn_phi_k_band_frac": self.learn_phi_k_band_frac,
            "learn_phi_max_steps": self.learn_phi_max_steps,
            "learn_phi_expand_factor": self.learn_phi_expand_factor,
            "learn_phi_weight_floor": self.learn_phi_weight_floor,
            "learn_phi_mass_floor_frac": self.learn_phi_mass_floor_frac,
            "learn_phi_min_error_gain_per_factor": self.learn_phi_min_error_gain_per_factor,
            "learn_phi_only": self.learn_phi_only,
            "learn_phi_report_out": self.learn_phi_report_out,
            "factor_phi_metrics_out": self.factor_phi_metrics_out,
            "factor_backend": self.factor_backend,
            "learn_phi_backend": self.learn_phi_backend,
            "blockwise_gene_set_block_size": self.blockwise_gene_set_block_size,
            "blockwise_epochs": self.blockwise_epochs,
            "blockwise_shuffle_blocks": self.blockwise_shuffle_blocks,
            "blockwise_warm_start": self.blockwise_warm_start,
            "blockwise_max_blocks": self.blockwise_max_blocks,
            "blockwise_report_out": self.blockwise_report_out,
            "factors_out": self.factors_out,
            "factor_metrics_out": self.factor_metrics_out,
            "gene_set_clusters_out": self.gene_set_clusters_out,
            "gene_clusters_out": self.gene_clusters_out,
            "learn_phi_prune_genes_num": self.learn_phi_prune_genes_num,
            "learn_phi_prune_gene_sets_num": self.learn_phi_prune_gene_sets_num,
            "learn_phi_max_num_iterations": self.learn_phi_max_num_iterations,
            "alpha0": self.alpha0,
            "beta0": self.beta0,
            "seed": self.seed,
            "factor_runs": self.factor_runs,
            "consensus_nmf": self.consensus_nmf,
            "consensus_min_factor_cosine": self.consensus_min_factor_cosine,
            "consensus_min_run_support": self.consensus_min_run_support,
            "consensus_aggregation": self.consensus_aggregation,
            "consensus_stats_out": self.consensus_stats_out,
            "gene_set_filter_type": self.gene_set_filter_type,
            "gene_set_filter_value": self.gene_set_filter_value,
            "gene_or_pheno_filter_type": self.gene_or_pheno_filter_type,
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
            "max_num_iterations": self.max_num_iterations,
            "rel_tol": self.rel_tol,
            "min_lambda_threshold": self.min_lambda_threshold,
            "lmm_auth_key": self.lmm_auth_key,
            "lmm_model": self.lmm_model,
            "lmm_provider": self.lmm_provider,
            "label_gene_sets_only": self.label_gene_sets_only,
            "label_include_phenos": self.label_include_phenos,
            "label_individually": self.label_individually,
            "keep_original_loadings": self.keep_original_loadings,
            "project_phenos_from_gene_sets": self.project_phenos_from_gene_sets,
            "pheno_capture_input": self.pheno_capture_input,
        }


@dataclass
class MainPipelineResult:
    state: object
    mode_state: dict
    factor_only: FactorOnlyStageResult
    phewas: PhewasStageResult = field(default_factory=PhewasStageResult)
    factor: FactorStageResult = field(default_factory=FactorStageResult)
    pheno_projection: PhewasStageResult = field(default_factory=PhewasStageResult)
    factor_phewas: PhewasStageResult = field(default_factory=PhewasStageResult)


def build_main_mode_state(domain):
    factor_gene_clusters_in = getattr(domain.options, "factor_gene_clusters_in", None)
    factor_gene_set_clusters_in = getattr(domain.options, "factor_gene_set_clusters_in", None)
    return {
        "run_factor": domain.run_factor,
        "run_phewas": domain.run_phewas,
        "run_factor_phewas": bool(domain.options.run_factor_phewas),
        "factor_projection_only": bool(
            factor_gene_clusters_in is not None or factor_gene_set_clusters_in is not None
        ),
        "factor_gene_clusters_in": factor_gene_clusters_in,
        "factor_gene_set_clusters_in": factor_gene_set_clusters_in,
        "factor_phewas_projection_only": bool(
            factor_gene_clusters_in is not None and bool(domain.options.run_factor_phewas)
        ),
        "run_naive_factor": domain.run_naive_factor,
        "use_phewas_for_factoring": domain.use_phewas_for_factoring,
        "factor_gene_set_x_pheno": domain.factor_gene_set_x_pheno,
        "expand_gene_sets": domain.expand_gene_sets,
        "factor_workflow": domain.factor_workflow,
    }


def run_main_factor_only_pipeline(domain, runtime, options, mode_state):
    if mode_state.get("factor_projection_only"):
        if options.factor_gene_clusters_in is not None:
            load_existing_factor_gene_clusters(
                domain,
                runtime,
                options.factor_gene_clusters_in,
            )
        if options.factor_gene_set_clusters_in is not None:
            load_existing_factor_gene_set_clusters(
                domain,
                runtime,
                options.factor_gene_set_clusters_in,
            )
        if options.gene_stats_in is not None:
            domain._run_read_y_stage(
                runtime,
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
        return FactorInputs()

    current_workflow = mode_state.get("factor_workflow")
    workflow_id = current_workflow.get("id") if isinstance(current_workflow, dict) else None

    gene_set_ids = None
    factor_uses_phewas_gene_set_ids = workflow_id in set(["F4", "F5", "F6", "F7", "F8"])
    gene_set_read_p_threshold = None
    if options.max_gene_set_read_p is not None and options.max_gene_set_read_p < 1:
        gene_set_read_p_threshold = options.max_gene_set_read_p
    gene_set_read_beta_uncorrected_threshold = options.min_gene_set_read_beta_uncorrected
    if options.gene_set_filter_value is not None:
        if gene_set_read_beta_uncorrected_threshold is None:
            gene_set_read_beta_uncorrected_threshold = options.gene_set_filter_value
        else:
            gene_set_read_beta_uncorrected_threshold = max(
                gene_set_read_beta_uncorrected_threshold,
                options.gene_set_filter_value,
            )
    if factor_uses_phewas_gene_set_ids:
        if options.gene_set_phewas_stats_in is None:
            domain.bail("Need --gene-set-phewas-stats-in")
        gene_set_ids = domain._read_gene_set_phewas_statistics(
            runtime,
            options.gene_set_phewas_stats_in,
            stats_id_col=options.gene_set_phewas_stats_id_col,
            stats_pheno_col=options.gene_set_phewas_stats_pheno_col,
            stats_beta_col=options.gene_set_phewas_stats_beta_col,
            stats_beta_uncorrected_col=options.gene_set_phewas_stats_beta_uncorrected_col,
            min_gene_set_beta=options.min_gene_set_read_beta,
            min_gene_set_beta_uncorrected=gene_set_read_beta_uncorrected_threshold,
            return_only_ids=True,
            phenos_to_match=options.anchor_phenos,
            max_num_entries_at_once=options.max_read_entries_at_once,
        )
    elif options.gene_set_stats_in is not None:
        gene_set_ids = domain._read_gene_set_statistics(
            runtime,
            options.gene_set_stats_in,
            stats_id_col=options.gene_set_stats_id_col,
            stats_exp_beta_tilde_col=options.gene_set_stats_exp_beta_tilde_col,
            stats_beta_tilde_col=options.gene_set_stats_beta_tilde_col,
            stats_p_col=options.gene_set_stats_p_col,
            stats_se_col=options.gene_set_stats_se_col,
            stats_beta_col=options.gene_set_stats_beta_col,
            stats_beta_uncorrected_col=options.gene_set_stats_beta_uncorrected_col,
            ignore_negative_exp_beta=options.ignore_negative_exp_beta,
            max_gene_set_p=gene_set_read_p_threshold,
            min_gene_set_beta=None,
            min_gene_set_beta_uncorrected=gene_set_read_beta_uncorrected_threshold,
            return_only_ids=True,
        )

    if gene_set_ids is not None:
        domain.log("Will read %d gene sets" % (len(gene_set_ids)), domain.DEBUG)

    domain._run_read_x_stage(
        runtime,
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

    if not runtime.has_gene_sets():
        domain.log("No gene sets survived the input filters; stopping")
        domain.sys.exit(0)

    if workflow_id == "F2":
        eaggl_gene_list_inputs.build_standalone_gene_list_inputs(domain, runtime, options)
    elif options.gene_stats_in is not None:
        domain._run_read_y_stage(
            runtime,
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
        gene_read_threshold = None
        if not mode_state["factor_gene_set_x_pheno"]:
            gene_read_threshold = resolve_factor_gene_or_pheno_filter_value(options, current_workflow)
        if gene_read_threshold is not None and runtime.combined_prior_Ys is not None:
            gene_keep_mask = runtime.combined_prior_Ys > gene_read_threshold
            if np.sum(~gene_keep_mask) > 0:
                domain.log(
                    "Subsetting to %d genes passing combined > %.3g before factorization"
                    % (int(np.sum(gene_keep_mask)), float(gene_read_threshold)),
                    domain.INFO,
                )
                runtime._subset_genes(gene_keep_mask, skip_V=True, skip_scale_factors=True)

    if workflow_id != "F2" and options.gene_set_stats_in is not None:
        domain._read_gene_set_statistics(
            runtime,
            options.gene_set_stats_in,
            stats_id_col=options.gene_set_stats_id_col,
            stats_exp_beta_tilde_col=options.gene_set_stats_exp_beta_tilde_col,
            stats_beta_tilde_col=options.gene_set_stats_beta_tilde_col,
            stats_p_col=options.gene_set_stats_p_col,
            stats_se_col=options.gene_set_stats_se_col,
            stats_beta_col=options.gene_set_stats_beta_col,
            stats_beta_uncorrected_col=options.gene_set_stats_beta_uncorrected_col,
            ignore_negative_exp_beta=options.ignore_negative_exp_beta,
            max_gene_set_p=gene_set_read_p_threshold,
            min_gene_set_beta=None,
            min_gene_set_beta_uncorrected=gene_set_read_beta_uncorrected_threshold,
        )

    factor_input_state = FactorInputs()
    if mode_state["run_factor"]:
        factor_input_state = load_factor_phewas_inputs(domain, runtime, options)
    return factor_input_state


def _parse_factor_number(column_name, prefix="Factor"):
    match = re.fullmatch(r"%s([0-9]+)" % re.escape(prefix), column_name)
    if match is None:
        return None
    return int(match.group(1))


def _coerce_optional_float(raw_value, *, field_name, row_name, domain):
    if raw_value is None or raw_value == "":
        return None
    try:
        return float(raw_value)
    except ValueError:
        domain.bail(
            "Could not parse numeric value for %s in %s: %s"
            % (field_name, row_name, raw_value)
        )


def load_existing_factor_gene_clusters(domain, runtime, gene_clusters_in):
    """Load prior EAGGL gene-cluster factor loadings for projection-only outputs."""
    if gene_clusters_in is None:
        domain.bail("Projection-only factor outputs require --factor-gene-clusters-in")

    genes = []
    gene_to_ind = {}
    loadings = []
    used_to_factor_values = []
    combined_values = []
    y_values = []
    prior_values = []
    labels_by_factor_index = {}

    with domain.open_gz(gene_clusters_in, "r") as input_fh:
        reader = csv.DictReader(input_fh, delimiter="\t")
        if reader.fieldnames is None:
            domain.bail("Empty gene-clusters file: %s" % gene_clusters_in)

        raw_factor_columns = []
        for column_name in reader.fieldnames:
            factor_number = _parse_factor_number(column_name)
            if factor_number is not None:
                raw_factor_columns.append((factor_number, column_name))
        raw_factor_columns.sort(key=lambda x: x[0])
        factor_columns = [column_name for _, column_name in raw_factor_columns]
        if len(factor_columns) == 0:
            domain.bail(
                "Could not find raw Factor1..FactorK loading columns in %s"
                % gene_clusters_in
            )

        for row_number, row in enumerate(reader, start=2):
            gene = row.get("Gene")
            if gene is None or gene == "":
                domain.bail("Missing Gene value in %s at row %d" % (gene_clusters_in, row_number))
            if gene in gene_to_ind:
                domain.bail(
                    "Duplicate Gene value in %s: %s. Projection-only input expects the standard "
                    "non-anchor gene_clusters output, not anchor-specific duplicated rows."
                    % (gene_clusters_in, gene)
                )

            gene_to_ind[gene] = len(genes)
            genes.append(gene)
            row_loadings = []
            for column_name in factor_columns:
                value = _coerce_optional_float(
                    row.get(column_name),
                    field_name=column_name,
                    row_name=gene,
                    domain=domain,
                )
                row_loadings.append(0.0 if value is None else value)
            loadings.append(row_loadings)
            raw_used_to_factor = row.get("used_to_factor")
            if raw_used_to_factor is None or raw_used_to_factor == "":
                used_to_factor_values.append(True)
            else:
                used_to_factor_values.append(str(raw_used_to_factor).strip().lower() in set(["1", "true", "t", "yes", "y"]))

            combined_values.append(
                _coerce_optional_float(row.get("combined"), field_name="combined", row_name=gene, domain=domain)
            )
            y_values.append(
                _coerce_optional_float(row.get("log_bf"), field_name="log_bf", row_name=gene, domain=domain)
            )
            prior_values.append(
                _coerce_optional_float(row.get("prior"), field_name="prior", row_name=gene, domain=domain)
            )

            cluster_name = row.get("cluster")
            label = row.get("label")
            factor_index = _parse_factor_number(cluster_name) if cluster_name is not None else None
            if factor_index is not None and label not in (None, ""):
                labels_by_factor_index.setdefault(factor_index - 1, label)

    if len(genes) == 0:
        domain.bail("No genes found in gene-clusters file: %s" % gene_clusters_in)

    factor_matrix = np.asarray(loadings, dtype=float)
    num_factors = factor_matrix.shape[1]
    runtime.genes = genes
    runtime.gene_to_ind = gene_to_ind
    runtime.exp_gene_factors = factor_matrix
    runtime.exp_lambdak = np.ones(num_factors, dtype=float)
    runtime.gene_factor_gene_mask = np.asarray(used_to_factor_values, dtype=bool)
    runtime.factor_labels = [
        labels_by_factor_index.get(i, "Factor%d" % (i + 1))
        for i in range(num_factors)
    ]

    if all(value is not None for value in combined_values):
        runtime.combined_prior_Ys = np.asarray(combined_values, dtype=float)
    if all(value is not None for value in y_values):
        runtime.Y = np.asarray(y_values, dtype=float)
    if all(value is not None for value in prior_values):
        runtime.priors = np.asarray(prior_values, dtype=float)

    runtime._record_params(
        {
            "factor_gene_clusters_in": gene_clusters_in,
            "factor_projection_only": True,
            "factor_projection_only_num_genes": len(genes),
            "factor_projection_only_num_factors": num_factors,
        },
        overwrite=True,
    )
    domain.log(
        "Loaded %d genes x %d factors from %s for projection-only EAGGL outputs"
        % (len(genes), num_factors, gene_clusters_in),
        domain.INFO,
    )
    return {
        "num_genes": len(genes),
        "num_factors": num_factors,
    }


def load_existing_factor_gene_set_clusters(domain, runtime, gene_set_clusters_in):
    """Load prior EAGGL gene-set-cluster factor loadings for projection-only outputs."""
    if gene_set_clusters_in is None:
        domain.bail("Projection-only gene-set factor outputs require --factor-gene-set-clusters-in")

    gene_sets = []
    gene_set_to_ind = {}
    loadings = []
    used_to_factor_values = []
    beta_values = []
    beta_uncorrected_values = []
    labels_by_factor_index = {}

    with domain.open_gz(gene_set_clusters_in, "r") as input_fh:
        reader = csv.DictReader(input_fh, delimiter="\t")
        if reader.fieldnames is None:
            domain.bail("Empty gene-set-clusters file: %s" % gene_set_clusters_in)

        raw_factor_columns = []
        for column_name in reader.fieldnames:
            factor_number = _parse_factor_number(column_name)
            if factor_number is not None:
                raw_factor_columns.append((factor_number, column_name))
        raw_factor_columns.sort(key=lambda x: x[0])
        factor_columns = [column_name for _, column_name in raw_factor_columns]
        if len(factor_columns) == 0:
            domain.bail(
                "Could not find raw Factor1..FactorK loading columns in %s"
                % gene_set_clusters_in
            )

        for row_number, row in enumerate(reader, start=2):
            gene_set = row.get("Gene_Set")
            if gene_set is None or gene_set == "":
                domain.bail("Missing Gene_Set value in %s at row %d" % (gene_set_clusters_in, row_number))
            if gene_set in gene_set_to_ind:
                domain.bail(
                    "Duplicate Gene_Set value in %s: %s. Projection-only input expects the standard "
                    "non-anchor gene_set_clusters output, not anchor-specific duplicated rows."
                    % (gene_set_clusters_in, gene_set)
                )

            gene_set_to_ind[gene_set] = len(gene_sets)
            gene_sets.append(gene_set)
            row_loadings = []
            for column_name in factor_columns:
                value = _coerce_optional_float(
                    row.get(column_name),
                    field_name=column_name,
                    row_name=gene_set,
                    domain=domain,
                )
                row_loadings.append(0.0 if value is None else value)
            loadings.append(row_loadings)

            raw_used_to_factor = row.get("used_to_factor")
            if raw_used_to_factor is None or raw_used_to_factor == "":
                used_to_factor_values.append(True)
            else:
                used_to_factor_values.append(str(raw_used_to_factor).strip().lower() in set(["1", "true", "t", "yes", "y"]))

            beta_values.append(
                _coerce_optional_float(row.get("beta"), field_name="beta", row_name=gene_set, domain=domain)
            )
            beta_uncorrected_values.append(
                _coerce_optional_float(row.get("beta_uncorrected"), field_name="beta_uncorrected", row_name=gene_set, domain=domain)
            )

            cluster_name = row.get("cluster")
            label = row.get("label")
            factor_index = _parse_factor_number(cluster_name) if cluster_name is not None else None
            if factor_index is not None and label not in (None, ""):
                labels_by_factor_index.setdefault(factor_index - 1, label)

    if len(gene_sets) == 0:
        domain.bail("No gene sets found in gene-set-clusters file: %s" % gene_set_clusters_in)

    factor_matrix = np.asarray(loadings, dtype=float)
    num_factors = factor_matrix.shape[1]
    existing_lambdak = getattr(runtime, "exp_lambdak", None)
    if existing_lambdak is not None and len(existing_lambdak) != num_factors:
        domain.bail(
            "Precomputed gene and gene-set factor tables disagree on factor count: %d vs %d"
            % (len(existing_lambdak), num_factors)
        )

    runtime.gene_sets = gene_sets
    runtime.gene_set_to_ind = gene_set_to_ind
    runtime.exp_gene_set_factors = factor_matrix
    runtime.exp_lambdak = np.ones(num_factors, dtype=float)
    runtime.gene_set_factor_gene_set_mask = np.asarray(used_to_factor_values, dtype=bool)
    if getattr(runtime, "factor_labels", None) is None:
        runtime.factor_labels = [
            labels_by_factor_index.get(i, "Factor%d" % (i + 1))
            for i in range(num_factors)
        ]

    if all(value is not None for value in beta_values):
        runtime.betas = np.asarray(beta_values, dtype=float)
    if all(value is not None for value in beta_uncorrected_values):
        runtime.betas_uncorrected = np.asarray(beta_uncorrected_values, dtype=float)

    runtime._record_params(
        {
            "factor_gene_set_clusters_in": gene_set_clusters_in,
            "factor_projection_only": True,
            "factor_projection_only_num_gene_sets": len(gene_sets),
            "factor_projection_only_gene_set_num_factors": num_factors,
        },
        overwrite=True,
    )
    domain.log(
        "Loaded %d gene sets x %d factors from %s for projection-only EAGGL outputs"
        % (len(gene_sets), num_factors, gene_set_clusters_in),
        domain.INFO,
    )
    return {
        "num_gene_sets": len(gene_sets),
        "num_factors": num_factors,
    }


def load_existing_factor_phewas_gene_clusters(domain, runtime, gene_clusters_in):
    """Compatibility wrapper for the legacy projection-only factor-PheWAS flag."""
    return load_existing_factor_gene_clusters(domain, runtime, gene_clusters_in)


def load_factor_phewas_inputs(domain, runtime, options):
    factor_input_data = domain._derive_factor_anchor_masks(runtime, options)
    if options.gene_set_phewas_stats_in is not None:
        domain._read_gene_set_phewas_statistics(
            runtime,
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
        domain._read_gene_phewas_bfs(
            runtime,
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


def resolve_gene_phewas_stage_decision(domain, runtime, requested_input, reusable_inputs):
    return domain.pegs_resolve_gene_phewas_input_decision_for_stage(
        requested_input=requested_input,
        reusable_inputs=reusable_inputs,
        read_gene_phewas=domain._has_loaded_gene_phewas(runtime),
        num_gene_phewas_filtered=runtime.num_gene_phewas_filtered,
    )


def run_phewas_with_common_args(domain, runtime, options, gene_phewas_bfs_in, run_for_factors=False, min_gene_factor_weight=0):
    phewas_config = domain.pegs_build_phewas_stage_config(
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
    eaggl_phewas.run_phewas(
        runtime,
        **phewas_config.to_run_kwargs(),
        options=options,
        bail_fn=domain.bail,
        warn_fn=domain.warn,
        log_fn=domain.log,
        info_level=domain.INFO,
        debug_level=domain.DEBUG,
        trace_level=domain.TRACE,
    )


def run_main_phewas_stage(domain, runtime, options):
    decision = resolve_gene_phewas_stage_decision(
        domain,
        runtime,
        options.run_phewas_input,
        [options.gene_phewas_bfs_in],
    )
    domain.log("PheWAS stage 'phewas': mode=%s reason=%s" % (decision.mode, decision.reason), domain.INFO)
    bfs_to_use = decision.resolved_input
    run_phewas_with_common_args(domain, runtime, options, bfs_to_use, run_for_factors=False)
    if options.phewas_stats_out:
        runtime.write_phewas_statistics(options.phewas_stats_out)
    return PhewasStageResult(ran=True, output_path=options.phewas_stats_out)


def extract_factor_workflow(mode_state):
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


def extract_factor_inputs(factor_input_state):
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


def resolve_factor_gene_or_pheno_filter_value(options, workflow):
    if options.anchor_gene_set:
        return options.gene_set_pheno_filter_value
    if isinstance(workflow, dict):
        factor_gene_set_x_pheno = workflow.get("factor_gene_set_x_pheno", False)
    else:
        factor_gene_set_x_pheno = workflow.factor_gene_set_x_pheno
    if factor_gene_set_x_pheno:
        return options.pheno_filter_value
    return options.gene_filter_value


def build_factor_execution_config(options, workflow, factor_inputs):
    return FactorExecutionConfig(
        max_num_factors=options.max_num_factors,
        phi=options.phi,
        learn_phi=options.learn_phi,
        learn_phi_max_redundancy=options.learn_phi_max_redundancy,
        learn_phi_max_redundancy_q90=options.learn_phi_max_redundancy_q90,
        learn_phi_runs_per_step=options.learn_phi_runs_per_step,
        learn_phi_min_run_support=options.learn_phi_min_run_support,
        learn_phi_min_stability=options.learn_phi_min_stability,
        learn_phi_max_fit_loss_frac=options.learn_phi_max_fit_loss_frac,
        learn_phi_k_band_frac=options.learn_phi_k_band_frac,
        learn_phi_max_steps=options.learn_phi_max_steps,
        learn_phi_expand_factor=options.learn_phi_expand_factor,
        learn_phi_weight_floor=options.learn_phi_weight_floor,
        learn_phi_mass_floor_frac=getattr(options, "learn_phi_mass_floor_frac", 0.005),
        learn_phi_min_error_gain_per_factor=getattr(options, "learn_phi_min_error_gain_per_factor", 5.0),
        learn_phi_only=getattr(options, "learn_phi_only", False),
        learn_phi_report_out=options.learn_phi_report_out,
        factor_phi_metrics_out=getattr(options, "factor_phi_metrics_out", None),
        factor_backend=getattr(options, "factor_backend", "full"),
        learn_phi_backend=getattr(options, "learn_phi_backend", "sentinel_pruned"),
        blockwise_gene_set_block_size=getattr(options, "blockwise_gene_set_block_size", 5000),
        blockwise_epochs=getattr(options, "blockwise_epochs", 3),
        blockwise_shuffle_blocks=getattr(options, "blockwise_shuffle_blocks", True),
        blockwise_warm_start=getattr(options, "blockwise_warm_start", True),
        blockwise_max_blocks=getattr(options, "blockwise_max_blocks", None),
        blockwise_report_out=getattr(options, "blockwise_report_out", None),
        factors_out=getattr(options, "factors_out", None),
        factor_metrics_out=getattr(options, "factor_metrics_out", None),
        gene_set_clusters_out=getattr(options, "gene_set_clusters_out", None),
        gene_clusters_out=getattr(options, "gene_clusters_out", None),
        learn_phi_prune_genes_num=getattr(options, "learn_phi_prune_genes_num", 1000),
        learn_phi_prune_gene_sets_num=getattr(options, "learn_phi_prune_gene_sets_num", 1000),
        learn_phi_max_num_iterations=getattr(options, "learn_phi_max_num_iterations", None),
        alpha0=options.alpha0,
        beta0=options.beta0,
        seed=options.seed,
        factor_runs=options.factor_runs,
        consensus_nmf=options.consensus_nmf,
        consensus_min_factor_cosine=options.consensus_min_factor_cosine,
        consensus_min_run_support=options.consensus_min_run_support,
        consensus_aggregation=options.consensus_aggregation,
        consensus_stats_out=options.consensus_stats_out,
        gene_set_filter_type="betas_uncorrected",
        gene_set_filter_value=options.gene_set_filter_value,
        gene_or_pheno_filter_type=(
            "gene_set_phewas_betas_uncorrected"
            if options.anchor_gene_set
            else ("gene_phewas_combined" if workflow.factor_gene_set_x_pheno else "combined_prior_Ys")
        ),
        gene_or_pheno_filter_value=resolve_factor_gene_or_pheno_filter_value(options, workflow),
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
        max_num_iterations=getattr(options, "max_num_iterations", 100),
        rel_tol=getattr(options, "rel_tol", 1e-4),
        min_lambda_threshold=options.min_lambda_threshold,
        lmm_auth_key=options.lmm_auth_key,
        lmm_model=options.lmm_model,
        lmm_provider=options.lmm_provider,
        label_gene_sets_only=options.label_gene_sets_only,
        label_include_phenos=options.label_include_phenos,
        label_individually=options.label_individually,
        keep_original_loadings=getattr(options, "keep_original_loadings", False),
        project_phenos_from_gene_sets=options.project_phenos_from_gene_sets,
        pheno_capture_input=getattr(options, "pheno_capture_input", "weighted_thresholded"),
    )


def run_factor_model(runtime, factor_config):
    runtime.run_factor(**factor_config.to_run_kwargs())


def run_main_factor_stage(domain, runtime, options, mode_state, factor_input_state):
    workflow = extract_factor_workflow(mode_state)
    factor_inputs = extract_factor_inputs(factor_input_state)
    factor_config = build_factor_execution_config(options, workflow, factor_inputs)
    run_factor_model(runtime, factor_config)
    return FactorStageResult(ran=True, workflow_id=workflow.workflow_id)


def run_main_pheno_projection_stage(domain, runtime, options):
    if runtime.num_factors() <= 0:
        domain.log("No factors; not projecting pheno clusters")
        return PhewasStageResult(ran=False, output_path=options.pheno_clusters_out)

    if options.project_phenos_from_gene_sets:
        if runtime.X_phewas_beta_uncorrected is None:
            domain._read_gene_set_phewas_statistics(
                runtime,
                options.gene_set_phewas_stats_in,
                stats_id_col=options.gene_set_phewas_stats_id_col,
                stats_pheno_col=options.gene_set_phewas_stats_pheno_col,
                stats_beta_col=options.gene_set_phewas_stats_beta_col,
                stats_beta_uncorrected_col=options.gene_set_phewas_stats_beta_uncorrected_col,
                min_gene_set_beta=getattr(options, "min_gene_set_read_beta", None),
                min_gene_set_beta_uncorrected=getattr(options, "min_gene_set_read_beta_uncorrected", None),
                max_num_entries_at_once=getattr(options, "max_read_entries_at_once", None),
            )
    elif not domain._has_loaded_gene_phewas(runtime):
        domain._read_gene_phewas_bfs(
            runtime,
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

    eaggl_factor_runtime.project_phenos_from_loaded_factors(
        runtime,
        project_phenos_from_gene_sets=options.project_phenos_from_gene_sets,
        pheno_capture_input=options.pheno_capture_input,
        bail_fn=domain.bail,
        log_fn=domain.log,
        info_level=domain.INFO,
    )
    return PhewasStageResult(ran=True, output_path=options.pheno_clusters_out)


def run_main_factor_phewas_stage(domain, runtime, options):
    if runtime.num_factors() <= 0:
        domain.log("No factors; not performing factor phewas")
        return PhewasStageResult(ran=False, output_path=options.factor_phewas_stats_out)

    decision = resolve_gene_phewas_stage_decision(
        domain,
        runtime,
        options.run_factor_phewas_input,
        [options.gene_phewas_bfs_in, options.run_phewas_input],
    )
    domain.log(
        "PheWAS stage 'factor_phewas': mode=%s reason=%s" % (decision.mode, decision.reason),
        domain.INFO,
    )
    requested_modes = eaggl_phewas.resolve_requested_factor_phewas_modes(options)
    runtime._record_params(
        {
            "factor_phewas_mode": options.factor_phewas_mode,
            "factor_phewas_modes": ",".join(requested_modes),
            "factor_phewas_anchor_covariate": options.factor_phewas_anchor_covariate,
            "factor_phewas_thresholded_combined_cutoff": options.factor_phewas_thresholded_combined_cutoff,
            "factor_phewas_se": options.factor_phewas_se,
        },
        overwrite=True,
    )
    bfs_to_use = decision.resolved_input
    run_phewas_with_common_args(
        domain,
        runtime,
        options,
        bfs_to_use,
        run_for_factors=True,
        min_gene_factor_weight=(
            options.factor_phewas_min_gene_factor_weight
            if any(
                mode in set(["legacy_continuous_direct", "legacy_continuous_combined"])
                for mode in requested_modes
            )
            else 0.0
        ),
    )
    if options.factor_phewas_stats_out:
        runtime.write_factor_phewas_statistics(options.factor_phewas_stats_out)
    return PhewasStageResult(ran=True, output_path=options.factor_phewas_stats_out)


def should_run_main_factor_phewas_stage(mode_state):
    return bool(
        mode_state["run_factor_phewas"]
        and (
            mode_state["run_factor"]
            or mode_state.get("factor_projection_only")
            or mode_state.get("factor_phewas_projection_only")
        )
    )


def should_run_main_pheno_projection_stage(mode_state, options):
    return bool(
        mode_state.get("factor_projection_only")
        and options.pheno_clusters_out is not None
    )
