from __future__ import annotations

from dataclasses import dataclass, field

from . import gene_list_inputs as eaggl_gene_list_inputs
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
    learn_phi_max_redundancy: float = 0.6
    learn_phi_runs_per_step: int = 5
    learn_phi_min_run_support: float = 0.6
    learn_phi_min_stability: float = 0.85
    learn_phi_max_fit_loss_frac: float = 0.05
    learn_phi_max_steps: int = 8
    learn_phi_expand_factor: float = 10.0
    learn_phi_weight_floor: float | None = None
    learn_phi_report_out: str | None = None
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
            "learn_phi_runs_per_step": self.learn_phi_runs_per_step,
            "learn_phi_min_run_support": self.learn_phi_min_run_support,
            "learn_phi_min_stability": self.learn_phi_min_stability,
            "learn_phi_max_fit_loss_frac": self.learn_phi_max_fit_loss_frac,
            "learn_phi_max_steps": self.learn_phi_max_steps,
            "learn_phi_expand_factor": self.learn_phi_expand_factor,
            "learn_phi_weight_floor": self.learn_phi_weight_floor,
            "learn_phi_report_out": self.learn_phi_report_out,
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
    factor_phewas: PhewasStageResult = field(default_factory=PhewasStageResult)


def build_main_mode_state(domain):
    return {
        "run_factor": domain.run_factor,
        "run_phewas": domain.run_phewas,
        "run_factor_phewas": domain.options.factor_phewas_from_gene_phewas_stats_in is not None,
        "run_naive_factor": domain.run_naive_factor,
        "use_phewas_for_factoring": domain.use_phewas_for_factoring,
        "factor_gene_set_x_pheno": domain.factor_gene_set_x_pheno,
        "expand_gene_sets": domain.expand_gene_sets,
        "factor_workflow": domain.factor_workflow,
    }


def run_main_factor_only_pipeline(domain, runtime, options, mode_state):
    current_workflow = mode_state.get("factor_workflow")
    workflow_id = current_workflow.get("id") if isinstance(current_workflow, dict) else None

    gene_set_ids = None
    factor_uses_phewas_gene_set_ids = workflow_id in set(["F4", "F5", "F6", "F7", "F8"])
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
            min_gene_set_beta_uncorrected=options.min_gene_set_read_beta_uncorrected,
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
            max_gene_set_p=options.max_gene_set_read_p,
            min_gene_set_beta=options.min_gene_set_read_beta,
            min_gene_set_beta_uncorrected=options.min_gene_set_read_beta_uncorrected,
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
            max_gene_set_p=options.max_gene_set_read_p,
            min_gene_set_beta=options.min_gene_set_read_beta,
            min_gene_set_beta_uncorrected=options.min_gene_set_read_beta_uncorrected,
        )

    factor_input_state = FactorInputs()
    if mode_state["run_factor"]:
        factor_input_state = load_factor_phewas_inputs(domain, runtime, options)
    return factor_input_state


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
        options.run_phewas_from_gene_phewas_stats_in,
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
    if workflow.factor_gene_set_x_pheno:
        return options.pheno_filter_value
    return options.gene_filter_value


def build_factor_execution_config(options, workflow, factor_inputs):
    return FactorExecutionConfig(
        max_num_factors=options.max_num_factors,
        phi=options.phi,
        learn_phi=options.learn_phi,
        learn_phi_max_redundancy=options.learn_phi_max_redundancy,
        learn_phi_runs_per_step=options.learn_phi_runs_per_step,
        learn_phi_min_run_support=options.learn_phi_min_run_support,
        learn_phi_min_stability=options.learn_phi_min_stability,
        learn_phi_max_fit_loss_frac=options.learn_phi_max_fit_loss_frac,
        learn_phi_max_steps=options.learn_phi_max_steps,
        learn_phi_expand_factor=options.learn_phi_expand_factor,
        learn_phi_weight_floor=options.learn_phi_weight_floor,
        learn_phi_report_out=options.learn_phi_report_out,
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


def run_main_factor_phewas_stage(domain, runtime, options):
    if runtime.num_factors() <= 0:
        domain.log("No factors; not performing factor phewas")
        return PhewasStageResult(ran=False, output_path=options.factor_phewas_stats_out)

    decision = resolve_gene_phewas_stage_decision(
        domain,
        runtime,
        options.factor_phewas_from_gene_phewas_stats_in,
        [options.gene_phewas_bfs_in, options.run_phewas_from_gene_phewas_stats_in],
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
    return bool(mode_state["run_factor"] and mode_state["run_factor_phewas"])
