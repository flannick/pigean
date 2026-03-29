from __future__ import annotations

from dataclasses import dataclass, field

from . import main_support as pigean_main_support
from . import phewas as pigean_phewas


@dataclass
class BetaStageResult:
    ran: bool = False
    source: str | None = None


@dataclass
class PriorsStageResult:
    ran: bool = False
    method: str | None = None


@dataclass
class GibbsStageResult:
    ran: bool = False
    num_chains: int | None = None
    total_num_iter: int | None = None


@dataclass
class GibbsStageConfig:
    run_kwargs: dict = field(default_factory=dict)
    num_chains: int | None = None
    total_num_iter: int | None = None


@dataclass
class NonHugePipelineResult:
    beta_tilde: BetaStageResult = field(default_factory=BetaStageResult)
    beta: BetaStageResult = field(default_factory=BetaStageResult)
    priors: PriorsStageResult = field(default_factory=PriorsStageResult)
    gibbs: GibbsStageResult = field(default_factory=GibbsStageResult)


@dataclass
class MainPipelineResult:
    state: object
    mode_state: dict
    sigma2_cond: object
    y_not_loaded: bool
    non_huge: NonHugePipelineResult | None = None


def run_main_beta_tilde_stage(services, state, options, mode_state):
    needs_gene_set_stats = (
        mode_state["run_beta_tilde"]
        or mode_state["run_beta"]
        or mode_state["run_priors"]
        or mode_state["run_naive_priors"]
        or mode_state["run_gibbs"]
        or mode_state["run_sim"]
    )
    if options.const_gene_set_beta is not None:
        state.beta_tildes = services.np.full(len(state.gene_sets), options.const_gene_set_beta)
        return BetaStageResult(ran=True, source="const_gene_set_beta")
    if options.gene_set_stats_in is not None:
        pigean_main_support.load_and_apply_gene_set_statistics_to_runtime(
            state,
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
            return_only_ids=False,
            open_text_fn=pigean_main_support.open_gz,
            get_col_fn=pigean_main_support.get_col,
            parse_log_fn=lambda message: services.log(message, services.INFO),
            apply_log_fn=lambda message: services.log(message, services.DEBUG),
            warn_fn=services.warn,
            bail_fn=services.bail,
        )
        return BetaStageResult(ran=True, source="gene_set_stats_in")
    if needs_gene_set_stats:
        max_gene_set_p = options.filter_gene_set_p if not options.betas_uncorrected_from_phewas else 1
        gene_set_stats_kwargs = dict(
            run_logistic=not options.linear,
            max_for_linear=options.max_for_linear,
            run_corrected_ols=not options.ols,
            use_sampling_for_betas=options.use_sampling_for_betas,
            correct_betas_mean=options.correct_betas_mean,
            correct_betas_var=options.correct_betas_var,
            gene_loc_file=options.gene_loc_file,
            gene_cor_file=options.gene_cor_file,
            gene_cor_file_gene_col=options.gene_cor_file_gene_col,
            gene_cor_file_cor_start_col=options.gene_cor_file_cor_start_col,
        )
        state.calculate_gene_set_statistics(max_gene_set_p=max_gene_set_p, **gene_set_stats_kwargs)
        if options.betas_uncorrected_from_phewas:
            state.calculate_gene_set_statistics(
                max_gene_set_p=1,
                Y=state.gene_pheno_Y,
                run_using_phewas=True,
                **gene_set_stats_kwargs,
            )
        return BetaStageResult(ran=True, source="calculate_gene_set_statistics")
    return BetaStageResult(ran=False, source="skipped")


def run_main_beta_stage(services, state, options, mode_state):
    needs_gene_set_betas = (
        mode_state["run_beta"]
        or mode_state["run_priors"]
        or mode_state["run_naive_priors"]
        or mode_state["run_gibbs"]
    )
    if needs_gene_set_betas and state.sigma2 is None:
        services.bail("Sigma2 was not initialized; provide --sigma2 explicitly")

    if options.cross_val:
        cross_val_kwargs = dict(
            folds=options.cross_val_folds,
            cross_val_max_num_tries=options.cross_val_max_num_tries,
            p=state.p,
            run_logistic=not options.linear,
            max_for_linear=options.max_for_linear,
            run_corrected_ols=not options.ols,
        )
        cross_val_kwargs.update(pigean_main_support.build_inner_beta_sampler_common_kwargs(options))
        state.run_cross_val(options.cross_val_num_explore_each_direction, **cross_val_kwargs)

    if options.const_gene_set_beta is not None:
        state.betas = services.np.full(len(state.gene_sets), options.const_gene_set_beta)
        state.betas_uncorrected = services.np.full(len(state.gene_sets), options.const_gene_set_beta)
        return BetaStageResult(ran=True, source="const_gene_set_beta")
    if options.gene_set_betas_in:
        state.read_betas(options.gene_set_betas_in)
        return BetaStageResult(ran=True, source="gene_set_betas_in")
    if needs_gene_set_betas:
        beta_sampling_kwargs = pigean_main_support.build_inner_beta_sampler_common_kwargs(options)
        beta_sampling_kwargs.update({
            "max_allowed_batch_correlation": options.max_allowed_batch_correlation,
            "update_hyper_sigma": False,
            "update_hyper_p": False,
            "pre_filter_batch_size": options.pre_filter_batch_size,
            "pre_filter_small_batch_size": options.pre_filter_small_batch_size,
            "betas_trace_out": options.betas_trace_out,
            "independent_only": options.independent_betas_only,
        })
        state.calculate_non_inf_betas(state.p, **beta_sampling_kwargs)
        pigean_phewas.run_advanced_set_b_phewas_beta_sampling_if_requested(
            services=services,
            state=state,
            options=options,
            beta_sampling_kwargs=beta_sampling_kwargs,
        )
        return BetaStageResult(ran=True, source="calculate_non_inf_betas")
    return BetaStageResult(ran=False, source="skipped")


def run_main_priors_stage(services, state, options, mode_state):
    if mode_state["run_priors"]:
        priors_kwargs = pigean_main_support.build_inner_beta_sampler_common_kwargs(options)
        priors_kwargs.update({
            "max_gene_set_p": options.filter_gene_set_p,
            "num_gene_batches": options.priors_num_gene_batches,
            "correct_betas_mean": options.correct_betas_mean,
            "correct_betas_var": options.correct_betas_var,
            "gene_loc_file": options.gene_loc_file,
            "gene_cor_file": options.gene_cor_file,
            "gene_cor_file_gene_col": options.gene_cor_file_gene_col,
            "gene_cor_file_cor_start_col": options.gene_cor_file_cor_start_col,
            "p_noninf": state.p,
            "run_logistic": not options.linear,
            "max_for_linear": options.max_for_linear,
            "adjust_priors": options.adjust_priors,
            "max_allowed_batch_correlation": options.max_allowed_batch_correlation,
        })
        state.calculate_priors(**priors_kwargs)
        return PriorsStageResult(ran=True, method="priors")
    if mode_state["run_naive_priors"]:
        state.calculate_naive_priors(adjust_priors=options.adjust_priors)
        return PriorsStageResult(ran=True, method="naive_priors")
    return PriorsStageResult(ran=False, method="skipped")


def build_main_gibbs_stage_config(options):
    return GibbsStageConfig(
        run_kwargs=dict(
            max_num_iter=options.max_num_iter,
            total_num_iter=options.total_num_iter_gibbs,
            max_num_restarts=options.max_num_restarts,
            num_chains=options.num_chains,
            num_mad=options.num_mad,
            r_threshold_burn_in=options.r_threshold_burn_in,
            use_max_r_for_convergence=options.use_max_r_for_convergence,
            increase_hyper_if_betas_below=options.experimental_increase_hyper_if_betas_below,
            experimental_hyper_mutation=options.experimental_hyper_mutation,
            update_huge_scores=options.update_huge_scores,
            top_gene_prior=options.top_gene_prior,
            min_num_burn_in=options.min_num_burn_in,
            max_num_burn_in=options.max_num_burn_in,
            min_num_post_burn_in=options.min_num_post_burn_in,
            max_num_post_burn_in=options.max_num_post_burn_in,
            max_num_iter_betas=options.max_num_iter_betas,
            min_num_iter_betas=options.min_num_iter_betas,
            num_chains_betas=options.num_chains_betas,
            r_threshold_burn_in_betas=options.r_threshold_burn_in_betas,
            use_max_r_for_convergence_betas=options.use_max_r_for_convergence_betas,
            max_frac_sem_betas=options.max_frac_sem_betas,
            use_mean_betas=not options.use_sampled_betas_in_gibbs,
            warm_start=options.warm_start,
            burn_in_rhat_quantile=options.burn_in_rhat_quantile,
            burn_in_patience=options.burn_in_patience,
            burn_in_stall_window=options.burn_in_stall_window,
            burn_in_stall_delta=options.burn_in_stall_delta,
            stop_mcse_quantile=options.stop_mcse_quantile,
            stop_patience=options.stop_patience,
            stop_top_gene_k=options.stop_top_gene_k,
            stop_min_gene_d=options.stop_min_gene_d,
            max_abs_mcse_d=options.max_abs_mcse_d,
            max_rel_mcse_beta=options.max_rel_mcse_beta,
            active_beta_top_k=options.active_beta_top_k,
            active_beta_min_abs=options.active_beta_min_abs,
            beta_rel_mcse_denom_floor=options.beta_rel_mcse_denom_floor,
            stall_window=options.stall_window,
            stall_min_burn_in=options.stall_min_burn_in,
            stall_min_post_burn_in=options.stall_min_post_burn_in,
            stall_delta_rhat=options.stall_delta_rhat,
            stall_delta_mcse=options.stall_delta_mcse,
            stall_recent_window=options.stall_recent_window,
            stall_recent_eps=options.stall_recent_eps,
            stopping_preset_name=options.gibbs_stopping_preset,
            diag_every=options.diag_every,
            sparse_frac_gibbs=options.sparse_frac_gibbs,
            sparse_max_gibbs=options.sparse_max_gibbs,
            sparse_solution=options.sparse_solution,
            sparse_frac_betas=options.sparse_frac_betas,
            pre_filter_batch_size=options.pre_filter_batch_size,
            pre_filter_small_batch_size=options.pre_filter_small_batch_size,
            max_allowed_batch_correlation=options.max_allowed_batch_correlation,
            gauss_seidel=options.gauss_seidel,
            gauss_seidel_betas=options.gauss_seidel_betas,
            num_batches_parallel=options.gibbs_num_batches_parallel,
            max_mb_X_h=options.gibbs_max_mb_X_h,
            initial_linear_filter=options.initial_linear_filter,
            adjust_priors=options.adjust_priors,
            correct_betas_mean=options.correct_betas_mean,
            correct_betas_var=options.correct_betas_var,
            gene_set_stats_trace_out=options.gene_set_stats_trace_out,
            gene_stats_trace_out=options.gene_stats_trace_out,
            betas_trace_out=options.betas_trace_out,
            debug_zero_sparse=options.debug_zero_sparse,
        ),
        num_chains=options.num_chains,
        total_num_iter=options.total_num_iter_gibbs,
    )


def run_main_gibbs_stage(services, state, options, mode_state):
    if not mode_state["run_gibbs"]:
        return GibbsStageResult(ran=False, num_chains=None, total_num_iter=None)
    gibbs_config = build_main_gibbs_stage_config(options)
    state.run_gibbs(**gibbs_config.run_kwargs)
    return GibbsStageResult(
        ran=True,
        num_chains=gibbs_config.num_chains,
        total_num_iter=gibbs_config.total_num_iter,
    )


def run_main_non_huge_pipeline(services, state, options, mode_state, sigma2_cond, y_not_loaded):
    stage_result = NonHugePipelineResult()

    if options.X_in is not None or options.X_list is not None or options.Xd_in is not None or options.Xd_list is not None:
        pigean_main_support.run_main_adaptive_read_x(state, options, mode_state, sigma2_cond)
    elif options.p_noninf is not None:
        if len(options.p_noninf) == 1:
            state.set_p(options.p_noninf[0])
        else:
            services.bail("Multiple --p-noninf is not supported without --X-in inputs")

    if not state.has_gene_sets():
        services.log("No gene sets survived the input filters; stopping")
        services.sys.exit(0)
    assert state.p is not None

    if y_not_loaded and options.const_gene_Y:
        pigean_main_support.set_const_Y(state, options.const_gene_Y)
    if options.X_out:
        state.write_X(options.X_out)
    if options.Xd_out:
        state.write_Xd(options.Xd_out)
    if options.V_out:
        state.write_V(options.V_out)

    if mode_state["run_sim"]:
        state.run_sim(
            sigma2=state.sigma2,
            p=state.p,
            sigma_power=state.sigma_power,
            log_bf_noise_sigma_mult=options.sim_log_bf_noise_sigma_mult,
            treat_sigma2_as_sigma2_cond=False,
            only_positive=options.sim_only_positive,
        )

    stage_result.beta_tilde = run_main_beta_tilde_stage(services, state, options, mode_state)
    stage_result.beta = run_main_beta_stage(services, state, options, mode_state)
    stage_result.priors = run_main_priors_stage(services, state, options, mode_state)
    stage_result.gibbs = run_main_gibbs_stage(services, state, options, mode_state)
    return stage_result
