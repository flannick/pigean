from __future__ import annotations

import copy

import numpy as np

from . import gibbs as pigean_gibbs
from .gibbs import GibbsOrchestrationCallbacks


def build_gibbs_callbacks(legacy_module, *, open_gz_fn, log_fn, bail_fn, info_level):
    def prepare_gibbs_run_inputs(state, num_chains, top_gene_prior):
        legacy_module._snapshot_pre_gibbs_state(state)

        y_var_orig = np.var(state.Y_for_regression)
        legacy_module._maybe_log_gibbs_conditional_variance(state, top_gene_prior)
        bf_orig = np.exp(state.Y_for_regression_orig)
        bf_orig_raw = np.exp(state.Y_orig)

        bf_orig_m = np.tile(bf_orig, num_chains).reshape(num_chains, len(bf_orig))
        log_bf_m = np.log(bf_orig_m)
        log_bf_uncorrected_m = np.log(bf_orig_m)

        bf_orig_raw_m = np.tile(bf_orig_raw, num_chains).reshape(num_chains, len(bf_orig_raw))
        log_bf_raw_m = np.log(bf_orig_raw_m)
        compute_Y_raw = np.any(~np.isclose(log_bf_m, log_bf_raw_m))
        cur_background_log_bf_v = np.tile(state.background_log_bf, num_chains)

        if state.y_corr_cholesky is not None:
            bail_fn("GLS not implemented yet for Gibbs sampling!")

        num_full_gene_sets = len(state.gene_sets)
        if state.gene_sets_missing is not None:
            num_full_gene_sets += len(state.gene_sets_missing)

        return {
            "y_var_orig": y_var_orig,
            "log_bf_m": log_bf_m,
            "log_bf_uncorrected_m": log_bf_uncorrected_m,
            "log_bf_raw_m": log_bf_raw_m,
            "compute_Y_raw": compute_Y_raw,
            "cur_background_log_bf_v": cur_background_log_bf_v,
            "num_full_gene_sets": num_full_gene_sets,
        }

    def new_gibbs_epoch_aggregates():
        epoch_aggregates = {}
        for key in legacy_module._GIBBS_EPOCH_SUM_KEYS + legacy_module._GIBBS_EPOCH_MISSING_SUM_KEYS:
            epoch_aggregates[key] = []
        return epoch_aggregates

    def reset_gibbs_diagnostics(state):
        state.betas_r_hat = None
        state.betas_mcse = None
        state.betas_uncorrected_r_hat = None
        state.betas_uncorrected_mcse = None
        state.priors_r_hat = None
        state.priors_mcse = None
        state.combined_prior_Ys_r_hat = None
        state.combined_prior_Ys_mcse = None
        state.Y_r_hat = None
        state.Y_mcse = None

    def start_gibbs_epoch(
        state,
        num_chains,
        num_full_gene_sets,
        use_mean_betas,
        max_mb_X_h,
        log_fun,
        epoch_aggregates,
        num_p_increases,
    ):
        epoch_state = legacy_module._initialize_gibbs_epoch_state(
            state,
            num_chains,
            num_full_gene_sets,
            use_mean_betas,
            max_mb_X_h,
            log_fun,
        )
        return {
            "full_betas_m_shape": epoch_state["full_betas_m_shape"],
            "epoch_control": legacy_module._initialize_gibbs_epoch_control_state(epoch_state),
            "epoch_sums": legacy_module._initialize_gibbs_epoch_sums_state(epoch_state, epoch_aggregates),
            "epoch_priors": legacy_module._initialize_gibbs_epoch_priors_state(epoch_state),
            "epoch_runtime": legacy_module._initialize_gibbs_epoch_runtime_state(epoch_state, num_p_increases),
            "post_burn_reset_arrays": epoch_state["post_burn_reset_arrays"],
            "post_burn_reset_missing_arrays": epoch_state["post_burn_reset_missing_arrays"],
            "X_hstacked": epoch_state["X_hstacked"],
            "stack_batch_size": epoch_state["stack_batch_size"],
            "num_stack_batches": epoch_state["num_stack_batches"],
        }

    def build_gibbs_epoch_finalize_context(
        state,
        run_state,
        epoch_phase_config,
        epoch_control,
        epoch_runtime,
        iteration_num,
    ):
        return {
            "include_missing": (state.genes_missing is not None),
            "gibbs_good": epoch_runtime["gibbs_good"],
            "iterations_run_this_epoch": (iteration_num + 1),
            "remaining_total_iter": run_state.remaining_total_iter,
            "num_completed_epochs": run_state.num_completed_epochs,
            "target_num_epochs": epoch_phase_config.target_num_epochs,
            "num_attempts": run_state.num_attempts,
            "max_num_attempt_restarts": run_state.max_num_attempt_restarts,
            "stop_due_to_stall": epoch_control["stop_due_to_stall"],
            "stop_due_to_precision": epoch_control["stop_due_to_precision"],
            "num_mad": epoch_phase_config.num_mad,
            "adjust_priors": epoch_phase_config.adjust_priors,
        }

    def finalize_gibbs_epoch_attempt(state, epoch_aggregates, epoch_sums, finalize_context):
        include_missing = finalize_context["include_missing"]
        gibbs_good = finalize_context["gibbs_good"]
        iterations_run_this_epoch = finalize_context["iterations_run_this_epoch"]
        remaining_total_iter = finalize_context["remaining_total_iter"]
        num_completed_epochs = finalize_context["num_completed_epochs"]
        target_num_epochs = finalize_context["target_num_epochs"]
        num_attempts = finalize_context["num_attempts"]
        max_num_attempt_restarts = finalize_context["max_num_attempt_restarts"]
        stop_due_to_stall = finalize_context["stop_due_to_stall"]
        stop_due_to_precision = finalize_context["stop_due_to_precision"]
        num_mad = finalize_context["num_mad"]
        adjust_priors = finalize_context["adjust_priors"]

        remaining_total_iter -= iterations_run_this_epoch
        if remaining_total_iter < 0:
            remaining_total_iter = 0

        if not gibbs_good:
            return {
                "remaining_total_iter": remaining_total_iter,
                "num_completed_epochs": num_completed_epochs,
                "should_continue": True,
            }

        assert np.all(epoch_sums["num_sum_Y_m"] > 0)
        assert np.all(epoch_sums["num_sum_beta_m"] > 0)

        for key in legacy_module._GIBBS_EPOCH_SUM_KEYS:
            epoch_aggregates[key].append(copy.copy(epoch_sums[key]))
        if include_missing:
            for key in legacy_module._GIBBS_EPOCH_MISSING_SUM_KEYS:
                epoch_aggregates[key].append(copy.copy(epoch_sums[key]))

        num_completed_epochs += 1
        log_fn(
            "Completed Gibbs epoch %d/%d (iter=%d, remaining_total_iter=%d)"
            % (num_completed_epochs, target_num_epochs, iterations_run_this_epoch, remaining_total_iter),
            info_level,
        )

        should_continue = pigean_gibbs._should_continue_gibbs_epoch_attempts(
            remaining_total_iter=remaining_total_iter,
            num_completed_epochs=num_completed_epochs,
            target_num_epochs=target_num_epochs,
            num_attempts=num_attempts,
            max_num_attempt_restarts=max_num_attempt_restarts,
            stop_due_to_stall=stop_due_to_stall,
            stop_due_to_precision=stop_due_to_precision,
        )
        if should_continue:
            return {
                "remaining_total_iter": remaining_total_iter,
                "num_completed_epochs": num_completed_epochs,
                "should_continue": True,
            }

        stacked = legacy_module._stack_gibbs_epoch_aggregates(
            epoch_aggregates=epoch_aggregates,
            include_missing=include_missing,
        )
        num_chains_effective = stacked["sum_betas_m"].shape[0]
        final_summary = legacy_module._summarize_gibbs_chain_aggregates(
            stacked["sum_Ys_m"],
            stacked["sum_Y_raws_m"],
            stacked["sum_log_pos_m"],
            stacked["sum_log_po_raws_m"],
            stacked["sum_log_po_raws2_m"],
            stacked["sum_priors_m"],
            stacked["sum_priors2_m"],
            stacked["sum_Ds_m"],
            stacked["sum_D_raws_m"],
            stacked["sum_bf_orig_m"],
            stacked["sum_bf_orig_raw_m"],
            stacked["sum_bf_orig_raw2_m"],
            stacked["sum_betas_m"],
            stacked["sum_betas2_m"],
            stacked["sum_betas_uncorrected_m"],
            stacked["sum_betas_uncorrected2_m"],
            stacked["sum_postp_m"],
            stacked["sum_beta_tildes_m"],
            stacked["sum_z_scores_m"],
            stacked["num_sum_Y_m"],
            stacked["num_sum_beta_m"],
            num_chains_effective,
            num_mad,
            record_param_fn=state._record_param,
            sum_priors_missing_m=stacked.get("sum_priors_missing_m") if include_missing else None,
            sum_Ds_missing_m=stacked.get("sum_Ds_missing_m") if include_missing else None,
            num_sum_priors_missing_m=stacked.get("num_sum_priors_missing_m") if include_missing else None,
        )
        legacy_module._apply_gibbs_final_state(state, final_summary, adjust_priors)
        legacy_module._apply_gibbs_ignored_final_state(state)
        return {
            "remaining_total_iter": remaining_total_iter,
            "num_completed_epochs": num_completed_epochs,
            "should_continue": False,
        }

    def prepare_gibbs_iteration_state(state, iteration_num, iteration_state_config, log_bf_m, log_bf_raw_m):
        iter_state = legacy_module._prepare_gibbs_iteration_inputs(
            state=state,
            iteration_num=iteration_num,
            log_bf_m=log_bf_m,
            log_bf_raw_m=log_bf_raw_m,
            iteration_input_config=iteration_state_config,
        )
        iter_state = dict(iter_state)
        return legacy_module._augment_gibbs_iteration_state_with_uncorrected_and_mask(
            state=state,
            iter_state=iter_state,
            prefilter_config=iteration_state_config["prefilter_config"],
            inner_beta_kwargs=iteration_state_config["inner_beta_kwargs"],
        )

    def run_gibbs_iteration_correction_and_updates(correction_context):
        state = correction_context["state"]
        iter_state = correction_context["iter_state"]
        gene_set_mask_m = correction_context["gene_set_mask_m"]
        epoch_control = correction_context["epoch_control"]
        correction_config = correction_context["correction_config"]
        epoch_priors = correction_context["epoch_priors"]
        epoch_runtime = correction_context["epoch_runtime"]
        epoch_sums = correction_context["epoch_sums"]
        iteration_num = correction_context["iteration_num"]
        log_bf_state = correction_context["log_bf_state"]

        (log_bf_m, log_bf_uncorrected_m, log_bf_raw_m) = log_bf_state
        restart_controls = legacy_module._build_gibbs_low_beta_restart_controls(correction_config)
        iteration_betas_priors = legacy_module._compute_gibbs_iteration_betas_and_priors(
            state,
            iter_state=iter_state,
            gene_set_mask_m=gene_set_mask_m,
            correction_config=correction_config,
            epoch_priors=epoch_priors,
            log_bf_state=(log_bf_m, log_bf_uncorrected_m, log_bf_raw_m),
        )
        full_betas_sample_m = iteration_betas_priors["full_betas_sample_m"]
        full_postp_sample_m = iteration_betas_priors["full_postp_sample_m"]
        full_betas_mean_m = iteration_betas_priors["full_betas_mean_m"]
        full_postp_mean_m = iteration_betas_priors["full_postp_mean_m"]
        (log_bf_m, log_bf_uncorrected_m, log_bf_raw_m) = (
            iteration_betas_priors["log_bf_m"],
            iteration_betas_priors["log_bf_uncorrected_m"],
            iteration_betas_priors["log_bf_raw_m"],
        )

        all_iteration_update = legacy_module._update_gibbs_all_sums_and_maybe_restart_low_betas(
            state=state,
            epoch_runtime=epoch_runtime,
            epoch_sums=epoch_sums,
            restart_controls=restart_controls,
            iteration_num=iteration_num,
            full_betas_mean_m=full_betas_mean_m,
        )
        should_break = legacy_module._apply_gibbs_all_iteration_update(
            epoch_runtime=epoch_runtime,
            epoch_control=epoch_control,
            all_iteration_update=all_iteration_update,
        )
        return {
            "full_betas_sample_m": full_betas_sample_m,
            "full_postp_sample_m": full_postp_sample_m,
            "full_betas_mean_m": full_betas_mean_m,
            "full_postp_mean_m": full_postp_mean_m,
            "log_bf_m": log_bf_m,
            "log_bf_uncorrected_m": log_bf_uncorrected_m,
            "log_bf_raw_m": log_bf_raw_m,
            "should_break": should_break,
        }

    def advance_gibbs_iteration_progress(progress_update_context):
        state = progress_update_context["state"]
        epoch_control = progress_update_context["epoch_control"]
        run_state = progress_update_context["run_state"]
        progress_runtime_config = progress_update_context["progress_runtime_config"]
        iter_state = progress_update_context["iter_state"]
        epoch_sums = progress_update_context["epoch_sums"]
        epoch_priors = progress_update_context["epoch_priors"]
        epoch_runtime = progress_update_context["epoch_runtime"]
        iteration_num = progress_update_context["iteration_num"]
        iteration_update = progress_update_context["iteration_update"]
        gene_set_stats_trace_fh = progress_update_context["gene_set_stats_trace_fh"]
        log_bf_state = progress_update_context["log_bf_state"]
        (log_bf_m, _log_bf_uncorrected_m, log_bf_raw_m) = log_bf_state

        progress_context = legacy_module._build_gibbs_iteration_progress_context(
            progress_runtime_config=progress_runtime_config,
            iteration_update=iteration_update,
        )
        iteration_progress_config = progress_context["iteration_progress_config"]

        burn_in_update = legacy_module._update_gibbs_burn_in_state(
            epoch_control=epoch_control,
            iteration_num=iteration_num,
            epoch_total_iter_offset=progress_context["epoch_total_iter_offset"],
            epoch_max_num_iter=progress_context["epoch_max_num_iter"],
            max_num_burn_in_for_epoch=progress_context["max_num_burn_in_for_epoch"],
            min_num_iter_for_epoch=progress_context["min_num_iter_for_epoch"],
            min_num_burn_in_for_epoch=progress_context["min_num_burn_in_for_epoch"],
            post_burn_reset_arrays=progress_context["post_burn_reset_arrays"],
            post_burn_reset_missing_arrays=progress_context["post_burn_reset_missing_arrays"],
            burn_in_config=iteration_progress_config["burn_in_config"],
            iter_state=iter_state,
            epoch_runtime=epoch_runtime,
        )
        legacy_module._apply_gibbs_burn_in_control_update(epoch_control=epoch_control, burn_in_update=burn_in_update)

        post_burn_update = legacy_module._update_gibbs_post_burn_state(
            state=state,
            max_num_post_burn_in_for_epoch=progress_context["max_num_post_burn_in_for_epoch"],
            min_num_post_burn_in_for_epoch=progress_context["min_num_post_burn_in_for_epoch"],
            epoch_max_num_iter=progress_context["epoch_max_num_iter"],
            diag_every=iteration_progress_config["diag_every"],
            post_burn_diag_config=iteration_progress_config["post_burn_diag_config"],
            iter_state=iter_state,
            epoch_sums=epoch_sums,
            epoch_priors=epoch_priors,
            epoch_control=epoch_control,
            run_state=run_state,
            log_bf_m=log_bf_m,
            log_bf_raw_m=log_bf_raw_m,
            full_betas_mean_m=progress_context["full_betas_mean_m"],
            full_postp_sample_m=progress_context["full_postp_sample_m"],
        )

        return legacy_module._finalize_gibbs_iteration_progress(
            state=state,
            gene_set_stats_trace_fh=gene_set_stats_trace_fh,
            iteration_num=iteration_num,
            trace_chain_offset=progress_context["trace_chain_offset"],
            iter_state=iter_state,
            full_betas_mean_m=progress_context["full_betas_mean_m"],
            full_betas_sample_m=progress_context["full_betas_sample_m"],
            full_postp_mean_m=progress_context["full_postp_mean_m"],
            full_postp_sample_m=progress_context["full_postp_sample_m"],
            R_beta_v=epoch_control["R_beta_v"],
            betas_sem2_v=post_burn_update["betas_sem2_v"],
            use_mean_betas=iteration_progress_config["use_mean_betas"],
            epoch_control=epoch_control,
            post_burn_update=post_burn_update,
        )

    return GibbsOrchestrationCallbacks(
        prepare_gibbs_run_inputs_fn=prepare_gibbs_run_inputs,
        new_gibbs_epoch_aggregates_fn=new_gibbs_epoch_aggregates,
        reset_gibbs_diagnostics_fn=reset_gibbs_diagnostics,
        start_gibbs_epoch_fn=start_gibbs_epoch,
        build_gibbs_epoch_finalize_context_fn=build_gibbs_epoch_finalize_context,
        finalize_gibbs_epoch_attempt_fn=finalize_gibbs_epoch_attempt,
        prepare_gibbs_iteration_state_fn=prepare_gibbs_iteration_state,
        run_gibbs_iteration_correction_and_updates_fn=run_gibbs_iteration_correction_and_updates,
        advance_gibbs_iteration_progress_fn=advance_gibbs_iteration_progress,
        open_gz_fn=open_gz_fn,
        log_fn=log_fn,
        bail_fn=bail_fn,
        info_level=info_level,
    )
