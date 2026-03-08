from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from .runtime import open_optional_gibbs_trace_files


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


@dataclass(frozen=True)
class GibbsOrchestrationCallbacks:
    prepare_gibbs_run_inputs_fn: Callable[..., dict]
    new_gibbs_epoch_aggregates_fn: Callable[[], dict]
    reset_gibbs_diagnostics_fn: Callable[[object], None]
    start_gibbs_epoch_fn: Callable[..., dict]
    build_gibbs_epoch_finalize_context_fn: Callable[..., dict]
    finalize_gibbs_epoch_attempt_fn: Callable[..., dict]
    prepare_gibbs_iteration_state_fn: Callable[..., tuple]
    run_gibbs_iteration_correction_and_updates_fn: Callable[..., dict]
    advance_gibbs_iteration_progress_fn: Callable[..., dict]
    open_gz_fn: Callable[..., object]
    log_fn: Callable[..., None]
    bail_fn: Callable[[str], None]
    info_level: object


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


def _initialize_gibbs_run_state(total_num_iter, target_num_epochs, max_num_restarts):
    return GibbsRunState(
        target_num_epochs=target_num_epochs,
        max_num_attempt_restarts=target_num_epochs + max_num_restarts,
        num_p_increases=0,
        num_attempts=0,
        num_completed_epochs=0,
        remaining_total_iter=int(total_num_iter),
    )


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


def _log_gibbs_configuration_summary(config, run_state, log_fn, info_level):
    log_fn("Running Gibbs")
    log_fn(
        "Gibbs stopping preset=%s; burn-in: r_threshold=%.4g, rhat_q=%.3g, patience=%d; active betas: topK=%d, min_abs=%.4g"
        % (
            config["stopping_preset_name"],
            config["r_threshold_burn_in"],
            config["burn_in_rhat_quantile"],
            config["burn_in_patience"],
            config["active_beta_top_k"],
            config["active_beta_min_abs"],
        ),
        info_level,
    )
    log_fn(
        "Gibbs restart schedule: target_epochs=%d (max_num_restarts=%d), max_attempts=%d, per-epoch max_num_iter=%d, total_num_iter=%d"
        % (
            config["target_num_epochs"],
            config["max_num_restarts"],
            run_state.max_num_attempt_restarts,
            config["epoch_max_num_iter_config"],
            config["total_num_iter"],
        ),
        info_level,
    )
    log_fn(
        "Gibbs epoch bounds (epoch 1): burn=[%d,%d], post=[%d,%d], stall_window=%d, stall_delta=%.4g"
        % (
            config["first_min_num_burn_in"],
            config["first_max_num_burn_in"],
            config["first_min_num_post_burn_in"],
            config["first_max_num_post_burn_in"],
            config["burn_in_stall_window"],
            config["burn_in_stall_delta"],
        ),
        info_level,
    )
    log_fn(
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
        info_level,
    )
    log_fn(
        "Gibbs experimental hyper mutation: enabled=%s, threshold=%s"
        % (
            str(config["experimental_hyper_mutation"]),
            ("%.4g" % config["increase_hyper_if_betas_below"])
            if config["increase_hyper_if_betas_below"] is not None
            else "None",
        ),
        info_level,
    )
    log_fn(
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
        info_level,
    )


def _build_gibbs_epoch_runtime_configs(config_inputs):
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


def _prepare_gibbs_epoch_attempt(
    state,
    run_state,
    epoch_phase_config,
    log_fn,
    info_level,
):
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
        min_num_iter_for_epoch = num_before_checking_p_increase

    state._record_param("num_gibbs_restarts", run_state.num_attempts - 1, overwrite=True)
    state._record_param("num_gibbs_epochs_completed", run_state.num_completed_epochs, overwrite=True)
    if run_state.num_attempts > 1:
        log_fn("Gibbs restart attempt %d" % (run_state.num_attempts - 1))
    log_fn(
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
        info_level,
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


def _build_gibbs_log_bf_payload(log_bf_m, log_bf_uncorrected_m, log_bf_raw_m, **extra):
    payload = {
        "log_bf_m": log_bf_m,
        "log_bf_uncorrected_m": log_bf_uncorrected_m,
        "log_bf_raw_m": log_bf_raw_m,
    }
    payload.update(extra)
    return payload


def _apply_gibbs_log_bf_update(update):
    return (
        update["log_bf_m"],
        update["log_bf_uncorrected_m"],
        update["log_bf_raw_m"],
    )


def _build_gibbs_epoch_attempt_result(log_bf_state, attempt_started, should_continue):
    (log_bf_m, log_bf_uncorrected_m, log_bf_raw_m) = log_bf_state
    return _build_gibbs_log_bf_payload(
        log_bf_m,
        log_bf_uncorrected_m,
        log_bf_raw_m,
        attempt_started=attempt_started,
        should_continue=should_continue,
    )


def _initialize_gibbs_epoch_attempt_context(
    state,
    run_state,
    epoch_aggregates,
    epoch_phase_config,
    epoch_iteration_static_config,
    epoch_attempt,
    callbacks,
):
    epoch_context = callbacks.start_gibbs_epoch_fn(
        state=state,
        num_chains=epoch_phase_config.num_chains,
        num_full_gene_sets=epoch_phase_config.num_full_gene_sets,
        use_mean_betas=epoch_phase_config.use_mean_betas,
        max_mb_X_h=epoch_phase_config.max_mb_X_h,
        log_fun=callbacks.log_fn,
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


def _build_initial_gibbs_log_bf_state(gibbs_inputs):
    return (
        gibbs_inputs["log_bf_m"],
        gibbs_inputs["log_bf_uncorrected_m"],
        gibbs_inputs["log_bf_raw_m"],
    )


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
    callbacks,
):
    (log_bf_m, log_bf_uncorrected_m, log_bf_raw_m) = log_bf_state
    epoch_attempt_context = _initialize_gibbs_epoch_attempt_context(
        state=state,
        run_state=run_state,
        epoch_aggregates=epoch_aggregates,
        epoch_phase_config=epoch_phase_config,
        epoch_iteration_static_config=epoch_iteration_static_config,
        epoch_attempt=epoch_attempt,
        callbacks=callbacks,
    )
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
        callbacks=callbacks,
    )
    iteration_num = epoch_loop_update["iteration_num"]
    log_bf_m, log_bf_uncorrected_m, log_bf_raw_m = _apply_gibbs_log_bf_update(epoch_loop_update)

    epoch_finalize_update = callbacks.finalize_gibbs_epoch_attempt_fn(
        state=state,
        epoch_aggregates=epoch_aggregates,
        epoch_sums=epoch_sums,
        finalize_context=callbacks.build_gibbs_epoch_finalize_context_fn(
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


def _run_single_gibbs_epoch_attempt(
    state,
    run_state,
    epoch_aggregates,
    epoch_phase_config,
    epoch_iteration_static_config,
    gene_set_stats_trace_fh,
    gene_stats_trace_fh,
    log_bf_state,
    callbacks,
):
    epoch_attempt = _prepare_gibbs_epoch_attempt(
        state=state,
        run_state=run_state,
        epoch_phase_config=epoch_phase_config,
        log_fn=callbacks.log_fn,
        info_level=callbacks.info_level,
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
        callbacks=callbacks,
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
    callbacks,
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
        callbacks=callbacks,
    )
    return _apply_gibbs_epoch_attempt_update(epoch_update=epoch_update)


def _run_gibbs_epoch_phase(
    state,
    run_state,
    epoch_aggregates,
    epoch_phase_config,
    epoch_iteration_static_config,
    gene_set_stats_trace_fh,
    gene_stats_trace_fh,
    log_bf_state,
    callbacks,
):
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
            callbacks=callbacks,
        )
        if should_break:
            break


def _run_gibbs_epochs_with_optional_traces(
    state,
    run_state,
    epoch_aggregates,
    epoch_phase_config,
    epoch_iteration_static_config,
    gene_set_stats_trace_out,
    gene_stats_trace_out,
    gibbs_inputs,
    callbacks,
):
    with open_optional_gibbs_trace_files(
        gene_set_stats_trace_out=gene_set_stats_trace_out,
        gene_stats_trace_out=gene_stats_trace_out,
        open_gz=callbacks.open_gz_fn,
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
            callbacks=callbacks,
        )


def _apply_gibbs_iteration_loop_update(iteration_run):
    return (_apply_gibbs_log_bf_update(iteration_run), iteration_run["stop_epoch"])


def _extract_gibbs_iteration_update_state(iteration_update):
    return (_apply_gibbs_log_bf_update(iteration_update), iteration_update["should_break"])


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


def _finalize_gibbs_iteration_after_correction(finalize_context, callbacks):
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

    iteration_progress_update = callbacks.advance_gibbs_iteration_progress_fn(
        progress_update_context=_build_gibbs_iteration_progress_update_context(
            state=finalize_context["state"],
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
    callbacks,
):
    (log_bf_m, log_bf_uncorrected_m, log_bf_raw_m) = log_bf_state
    iter_state, gene_set_mask_m = callbacks.prepare_gibbs_iteration_state_fn(
        state=state,
        iteration_num=iteration_num,
        iteration_state_config=iteration_state_config,
        log_bf_m=log_bf_m,
        log_bf_raw_m=log_bf_raw_m,
    )

    iteration_update = callbacks.run_gibbs_iteration_correction_and_updates_fn(
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
        callbacks=callbacks,
    )


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
    callbacks,
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
            callbacks=callbacks,
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


def run_outer_gibbs(
    state,
    callbacks,
    *,
    max_num_iter=100,
    total_num_iter=None,
    max_num_restarts=3,
    num_chains=10,
    num_mad=3,
    r_threshold_burn_in=1.10,
    use_max_r_for_convergence=True,
    increase_hyper_if_betas_below=None,
    experimental_hyper_mutation=False,
    update_huge_scores=True,
    top_gene_prior=None,
    min_num_burn_in=10,
    max_num_burn_in=None,
    min_num_post_burn_in=None,
    max_num_post_burn_in=None,
    max_num_iter_betas=1100,
    min_num_iter_betas=10,
    num_chains_betas=4,
    r_threshold_burn_in_betas=1.01,
    use_max_r_for_convergence_betas=True,
    max_frac_sem_betas=0.01,
    use_mean_betas=True,
    warm_start=False,
    burn_in_rhat_quantile=0.95,
    burn_in_patience=2,
    burn_in_stall_window=10,
    burn_in_stall_delta=0.01,
    stop_mcse_quantile=0.95,
    stop_patience=2,
    stop_top_gene_k=200,
    stop_min_gene_d=None,
    max_abs_mcse_d=0.05,
    max_rel_mcse_beta=0.20,
    active_beta_top_k=200,
    active_beta_min_abs=0.01,
    beta_rel_mcse_denom_floor=0.10,
    stall_window=8,
    stall_min_burn_in=50,
    stall_min_post_burn_in=50,
    stall_delta_rhat=0.01,
    stall_delta_mcse=0.01,
    stall_recent_window=4,
    stall_recent_eps=0.0,
    stopping_preset_name="lenient",
    diag_every=5,
    sparse_frac_gibbs=0.01,
    sparse_max_gibbs=0.001,
    sparse_solution=False,
    sparse_frac_betas=None,
    pre_filter_batch_size=None,
    pre_filter_small_batch_size=500,
    max_allowed_batch_correlation=None,
    gauss_seidel_betas=False,
    gauss_seidel=False,
    num_batches_parallel=10,
    max_mb_X_h=200,
    initial_linear_filter=True,
    correct_betas_mean=True,
    correct_betas_var=True,
    adjust_priors=True,
    gene_set_stats_trace_out=None,
    gene_stats_trace_out=None,
    betas_trace_out=None,
    debug_zero_sparse=False,
    eps=0.01,
):
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
    _record_gibbs_configuration_params(state, run_state, gibbs_record_config)
    _log_gibbs_configuration_summary(gibbs_record_config, run_state, callbacks.log_fn, callbacks.info_level)

    callbacks.reset_gibbs_diagnostics_fn(state)

    gibbs_inputs = callbacks.prepare_gibbs_run_inputs_fn(
        state=state,
        num_chains=num_chains,
        top_gene_prior=top_gene_prior,
    )
    epoch_aggregates = callbacks.new_gibbs_epoch_aggregates_fn()
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
    _run_gibbs_epochs_with_optional_traces(
        state=state,
        run_state=run_state,
        epoch_aggregates=epoch_aggregates,
        epoch_phase_config=epoch_runtime_configs.epoch_phase_config,
        epoch_iteration_static_config=epoch_runtime_configs.epoch_iteration_static_config,
        gene_set_stats_trace_out=gene_set_stats_trace_out,
        gene_stats_trace_out=gene_stats_trace_out,
        gibbs_inputs=gibbs_inputs,
        callbacks=callbacks,
    )

    if run_state.num_completed_epochs == 0:
        callbacks.bail_fn("Gibbs failed to complete any successful epochs within restart/iteration limits")
    callbacks.log_fn(
        "Aggregated %d Gibbs epoch(s) into %d effective chains"
        % (run_state.num_completed_epochs, run_state.num_completed_epochs * num_chains),
        callbacks.info_level,
    )
