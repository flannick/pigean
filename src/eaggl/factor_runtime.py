from __future__ import annotations

import copy
import gzip
import math

import numpy as np
import scipy
import scipy.sparse as sparse

def _clone_runtime_value(value):
    try:
        return copy.deepcopy(value)
    except Exception:
        return value


def _clone_runtime_state(source):
    cloned = copy.copy(source)
    cloned.__dict__ = {key: _clone_runtime_value(value) for key, value in source.__dict__.items()}
    return cloned


def _replace_runtime_state(target, source):
    target.__dict__.clear()
    target.__dict__.update({key: _clone_runtime_value(value) for key, value in source.__dict__.items()})


def _derive_factor_run_seeds(seed, factor_runs):
    if factor_runs <= 1:
        return [seed]
    base_seed = seed if seed is not None else int(np.random.randint(0, np.iinfo(np.uint32).max))
    children = np.random.SeedSequence(base_seed).spawn(factor_runs)
    return [int(child.generate_state(1, dtype=np.uint32)[0]) for child in children]


def _run_with_numpy_seed(seed, fn):
    if seed is None:
        return fn()
    old_state = np.random.get_state()
    np.random.seed(seed)
    try:
        return fn()
    finally:
        np.random.set_state(old_state)


def _normalize_factor_columns(matrix):
    if matrix is None or matrix.size == 0:
        return None
    norms = np.linalg.norm(matrix, axis=0)
    norms[norms == 0] = 1.0
    return matrix / norms[np.newaxis, :]


def _aggregate_consensus_stack(stacked_values, aggregation):
    if aggregation == "mean":
        return np.mean(stacked_values, axis=0)
    return np.median(stacked_values, axis=0)


def _compute_any_anchor_relevance(factor_anchor_relevance):
    anchor_relevance = np.asarray(factor_anchor_relevance, dtype=float)
    if anchor_relevance.ndim != 2:
        raise ValueError("factor_anchor_relevance must be 2D")
    if anchor_relevance.shape[1] == 0:
        return np.zeros(anchor_relevance.shape[0], dtype=float)
    clipped = np.clip(anchor_relevance, 0.0, 1.0)
    return 1.0 - np.prod(1.0 - clipped, axis=1)


def _open_text_output(path):
    if path.endswith(".gz"):
        return gzip.open(path, "wt", encoding="utf-8")
    return open(path, "w", encoding="utf-8")


def _summarize_mask(mask):
    if mask is None:
        return (False, None, None)
    mask_array = np.asarray(mask)
    if mask_array.size == 0:
        return (True, 0, 0)
    try:
        selected = int(np.sum(mask_array.astype(bool)))
    except Exception:
        selected = int(np.sum(mask_array != 0))
    return (True, int(mask_array.size), selected)


def _build_factor_param_record(
    *,
    max_num_factors,
    phi,
    alpha0,
    beta0,
    seed,
    factor_runs,
    consensus_nmf,
    consensus_min_factor_cosine,
    consensus_min_run_support,
    consensus_aggregation,
    consensus_stats_out,
    learn_phi,
    learn_phi_max_redundancy,
    learn_phi_runs_per_step,
    learn_phi_min_run_support,
    learn_phi_min_stability,
    learn_phi_max_fit_loss_frac,
    learn_phi_max_steps,
    learn_phi_expand_factor,
    learn_phi_weight_floor,
    learn_phi_report_out,
    gene_set_filter_type,
    gene_set_filter_value,
    gene_or_pheno_filter_type,
    gene_or_pheno_filter_value,
    pheno_prune_value,
    pheno_prune_number,
    gene_prune_value,
    gene_prune_number,
    gene_set_prune_value,
    gene_set_prune_number,
    anchor_pheno_mask,
    anchor_gene_mask,
    anchor_any_pheno,
    anchor_any_gene,
    anchor_gene_set,
    run_transpose,
    max_num_iterations,
    rel_tol,
    min_lambda_threshold,
    lmm_auth_key,
    lmm_model,
    lmm_provider,
    label_gene_sets_only,
    label_include_phenos,
    label_individually,
    keep_original_loadings,
    project_phenos_from_gene_sets,
):
    anchor_gene_mask_present, anchor_gene_mask_total, anchor_gene_mask_selected = _summarize_mask(anchor_gene_mask)
    anchor_pheno_mask_present, anchor_pheno_mask_total, anchor_pheno_mask_selected = _summarize_mask(anchor_pheno_mask)
    return {
        "max_num_factors": int(max_num_factors),
        "phi": float(phi),
        "alpha0": float(alpha0),
        "beta0": float(beta0),
        "seed": int(seed) if seed is not None else None,
        "factor_runs": int(factor_runs),
        "consensus_nmf": bool(consensus_nmf),
        "consensus_min_factor_cosine": float(consensus_min_factor_cosine),
        "consensus_min_run_support": float(consensus_min_run_support),
        "consensus_aggregation": consensus_aggregation,
        "consensus_stats_out": consensus_stats_out,
        "learn_phi": bool(learn_phi),
        "learn_phi_max_redundancy": float(learn_phi_max_redundancy),
        "learn_phi_runs_per_step": int(learn_phi_runs_per_step),
        "learn_phi_min_run_support": float(learn_phi_min_run_support),
        "learn_phi_min_stability": float(learn_phi_min_stability),
        "learn_phi_max_fit_loss_frac": float(learn_phi_max_fit_loss_frac),
        "learn_phi_max_steps": int(learn_phi_max_steps),
        "learn_phi_expand_factor": float(learn_phi_expand_factor),
        "learn_phi_weight_floor": None if learn_phi_weight_floor is None else float(learn_phi_weight_floor),
        "learn_phi_report_out": learn_phi_report_out,
        "gene_set_filter_value": gene_set_filter_value,
        "gene_or_pheno_filter_value": gene_or_pheno_filter_value,
        "pheno_prune_value": pheno_prune_value,
        "pheno_prune_number": pheno_prune_number,
        "gene_prune_value": gene_prune_value,
        "gene_prune_number": gene_prune_number,
        "gene_set_prune_value": gene_set_prune_value,
        "gene_set_prune_number": gene_set_prune_number,
        "anchor_any_pheno": bool(anchor_any_pheno),
        "anchor_any_gene": bool(anchor_any_gene),
        "anchor_gene_set": bool(anchor_gene_set),
        "anchor_gene_mask_present": bool(anchor_gene_mask_present),
        "anchor_gene_mask_total": anchor_gene_mask_total,
        "anchor_gene_mask_selected": anchor_gene_mask_selected,
        "anchor_pheno_mask_present": bool(anchor_pheno_mask_present),
        "anchor_pheno_mask_total": anchor_pheno_mask_total,
        "anchor_pheno_mask_selected": anchor_pheno_mask_selected,
        "run_transpose": bool(run_transpose),
        "max_num_iterations": int(max_num_iterations),
        "rel_tol": float(rel_tol),
        "min_lambda_threshold": float(min_lambda_threshold),
        "lmm_auth_key_present": lmm_auth_key is not None,
        "lmm_model": lmm_model,
        "lmm_provider": lmm_provider,
        "label_gene_sets_only": bool(label_gene_sets_only),
        "label_include_phenos": bool(label_include_phenos),
        "label_individually": bool(label_individually),
        "keep_original_loadings": bool(keep_original_loadings),
        "project_phenos_from_gene_sets": bool(project_phenos_from_gene_sets),
    }


def _extract_canonical_factor_matrix(state):
    if state.exp_gene_set_factors is not None and state.exp_gene_set_factors.size > 0:
        return state.exp_gene_set_factors
    if state.exp_gene_factors is not None and state.exp_gene_factors.size > 0:
        return state.exp_gene_factors
    if state.exp_pheno_factors is not None and state.exp_pheno_factors.size > 0:
        return state.exp_pheno_factors
    return None


def _prepare_factor_vector_for_overlap(vector, weight_floor):
    clipped = np.clip(np.asarray(vector, dtype=float), 0.0, None)
    if weight_floor is not None and weight_floor > 0:
        clipped[clipped < weight_floor] = 0.0
    total = float(np.sum(clipped))
    if total <= 0:
        return clipped
    return clipped / total


def _weighted_jaccard_similarity(u, v, weight_floor):
    u_prepared = _prepare_factor_vector_for_overlap(u, weight_floor)
    v_prepared = _prepare_factor_vector_for_overlap(v, weight_floor)
    if float(np.sum(u_prepared)) <= 0 and float(np.sum(v_prepared)) <= 0:
        return 0.0
    denominator = float(np.sum(np.maximum(u_prepared, v_prepared)))
    if denominator <= 0:
        return 0.0
    return float(np.sum(np.minimum(u_prepared, v_prepared)) / denominator)


def _compute_within_run_factor_redundancy(state, weight_floor):
    canonical = _extract_canonical_factor_matrix(state)
    if canonical is None or canonical.shape[1] <= 1:
        return 0.0
    max_redundancy = 0.0
    for left_index in range(canonical.shape[1]):
        for right_index in range(left_index + 1, canonical.shape[1]):
            max_redundancy = max(
                max_redundancy,
                _weighted_jaccard_similarity(
                    canonical[:, left_index],
                    canonical[:, right_index],
                    weight_floor,
                ),
            )
    return float(max_redundancy)


def _collect_run_indices_by_modal_factor_count(run_states, run_summaries):
    factor_count_to_indices = {}
    for index, run_state in enumerate(run_states):
        factor_count_to_indices.setdefault(int(run_summaries[index].get("num_factors", 0)), []).append(index)
    modal_factor_count = max(
        sorted(factor_count_to_indices),
        key=lambda key: (len(factor_count_to_indices[key]), key),
    )
    return modal_factor_count, factor_count_to_indices[modal_factor_count]


def _match_factor_cosines(reference_matrix, other_matrix):
    if reference_matrix is None or other_matrix is None:
        return []
    if reference_matrix.shape[1] == 0 or other_matrix.shape[1] == 0:
        return []
    reference_norm = _normalize_factor_columns(reference_matrix)
    other_norm = _normalize_factor_columns(other_matrix)
    similarity = reference_norm.T @ other_norm
    ref_inds, other_inds = scipy.optimize.linear_sum_assignment(1.0 - similarity)
    return [float(similarity[ref_index, other_index]) for ref_index, other_index in zip(ref_inds, other_inds)]


def _summarize_phi_candidate(run_states, run_summaries, *, phi, weight_floor):
    modal_factor_count, modal_indices = _collect_run_indices_by_modal_factor_count(run_states, run_summaries)
    reference_index = min(modal_indices, key=lambda idx: _best_run_sort_key(run_summaries[idx]))

    best_error = None
    best_evidence = None
    for index in modal_indices:
        reconstruction_error = run_summaries[index].get("reconstruction_error")
        if reconstruction_error is not None:
            best_error = reconstruction_error if best_error is None else min(best_error, reconstruction_error)
        evidence = run_summaries[index].get("evidence")
        if evidence is not None:
            best_evidence = evidence if best_evidence is None else min(best_evidence, evidence)

    matched_cosines = []
    reference_matrix = _extract_canonical_factor_matrix(run_states[reference_index])
    for index in modal_indices:
        if index == reference_index:
            continue
        matched_cosines.extend(
            _match_factor_cosines(reference_matrix, _extract_canonical_factor_matrix(run_states[index]))
        )

    if len(modal_indices) <= 1:
        stability = 1.0
    elif len(matched_cosines) == 0:
        stability = 0.0
    else:
        stability = float(np.mean(np.asarray(matched_cosines, dtype=float)))

    return {
        "phi": float(phi),
        "modal_factor_count": int(modal_factor_count),
        "run_support": float(len(modal_indices)) / float(max(1, len(run_states))),
        "stability": float(stability),
        "redundancy": _compute_within_run_factor_redundancy(run_states[reference_index], weight_floor),
        "best_error": None if best_error is None else float(best_error),
        "best_evidence": None if best_evidence is None else float(best_evidence),
        "reference_run_index": int(reference_index),
        "modal_run_indices": [int(index) for index in modal_indices],
        "matched_cosines": [float(value) for value in matched_cosines],
    }


def _find_candidate_by_phi(candidates_by_phi, phi_value):
    for existing_phi, candidate in candidates_by_phi.items():
        if math.isclose(existing_phi, phi_value, rel_tol=1e-12, abs_tol=1e-15):
            return candidate
    return None


def _evaluate_phi_candidate(
    template_state,
    *,
    phi,
    seed,
    runs_per_step,
    factor_kwargs,
    weight_floor,
    log_fn,
    info_level,
):
    log_fn("Evaluating automatic phi candidate %.6g with %d restart(s)" % (phi, runs_per_step), info_level)
    search_factor_kwargs = dict(factor_kwargs)
    search_factor_kwargs["phi"] = phi
    search_factor_kwargs["lmm_auth_key"] = None
    search_factor_kwargs["label_gene_sets_only"] = False
    search_factor_kwargs["label_include_phenos"] = False
    search_factor_kwargs["label_individually"] = False

    child_seeds = _derive_factor_run_seeds(seed, runs_per_step)
    run_states = []
    run_summaries = []
    for run_index, child_seed in enumerate(child_seeds):
        run_state = _clone_runtime_state(template_state)
        run_summary = _run_factor_with_seed(
            run_state,
            seed=child_seed,
            run_index=run_index,
            factor_kwargs=search_factor_kwargs,
        )
        run_states.append(run_state)
        run_summaries.append(run_summary)

    candidate = _summarize_phi_candidate(
        run_states,
        run_summaries,
        phi=phi,
        weight_floor=weight_floor,
    )
    candidate["run_summaries"] = copy.deepcopy(run_summaries)
    return candidate


def _candidate_sort_key(candidate):
    best_evidence = candidate.get("best_evidence")
    return (
        -int(candidate.get("modal_factor_count", 0)),
        float(candidate.get("phi", float("inf"))),
        float(candidate.get("redundancy", float("inf"))),
        float("inf") if best_evidence is None else float(best_evidence),
    )


def _select_phi_candidate(
    candidates,
    *,
    max_redundancy,
    min_run_support,
    min_stability,
    max_fit_loss_frac,
):
    finite_errors = [float(candidate["best_error"]) for candidate in candidates if candidate.get("best_error") is not None]
    best_global_error = min(finite_errors) if len(finite_errors) > 0 else None

    acceptable = []
    for candidate in candidates:
        if candidate["redundancy"] > max_redundancy:
            continue
        if candidate["run_support"] < min_run_support:
            continue
        if candidate["stability"] < min_stability:
            continue
        if best_global_error is not None and candidate.get("best_error") is not None:
            if float(candidate["best_error"]) > float(best_global_error) * (1.0 + max_fit_loss_frac):
                continue
        acceptable.append(candidate)

    if len(acceptable) > 0:
        return min(acceptable, key=_candidate_sort_key), "max_factor_count_within_constraints"

    def _fallback_sort_key(candidate):
        fit_limit = None if best_global_error is None else float(best_global_error) * (1.0 + max_fit_loss_frac)
        fit_violation = 0.0
        if fit_limit is not None and candidate.get("best_error") is not None:
            fit_violation = max(0.0, float(candidate["best_error"]) - fit_limit)
        best_evidence = candidate.get("best_evidence")
        return (
            max(0.0, float(candidate["redundancy"]) - max_redundancy),
            max(0.0, min_run_support - float(candidate["run_support"])),
            max(0.0, min_stability - float(candidate["stability"])),
            fit_violation,
            -int(candidate["modal_factor_count"]),
            float(candidate["phi"]),
            float(candidate["redundancy"]),
            float("inf") if best_evidence is None else float(best_evidence),
        )

    return min(candidates, key=_fallback_sort_key), "fallback_min_constraint_violation"


def _write_phi_search_report(report_path, candidates, *, selected_phi, selection_reason):
    if report_path is None:
        return
    with _open_text_output(report_path) as output_fh:
        output_fh.write(
            "selected\tselection_reason\tphi\tmodal_factor_count\trun_support\tstability\tredundancy\tbest_error\tbest_evidence\treference_run_index\tmodal_run_indices\tmatched_cosines\n"
        )
        for candidate in sorted(candidates, key=lambda row: float(row["phi"])):
            output_fh.write(
                "%s\t%s\t%.12g\t%d\t%.6g\t%.6g\t%.6g\t%s\t%s\t%d\t%s\t%s\n"
                % (
                    "1" if math.isclose(float(candidate["phi"]), float(selected_phi), rel_tol=1e-12, abs_tol=1e-15) else "0",
                    selection_reason,
                    float(candidate["phi"]),
                    int(candidate["modal_factor_count"]),
                    float(candidate["run_support"]),
                    float(candidate["stability"]),
                    float(candidate["redundancy"]),
                    "" if candidate.get("best_error") is None else "%.12g" % float(candidate["best_error"]),
                    "" if candidate.get("best_evidence") is None else "%.12g" % float(candidate["best_evidence"]),
                    int(candidate["reference_run_index"]),
                    ",".join([str(index) for index in candidate.get("modal_run_indices", [])]),
                    ",".join(["%.6g" % float(value) for value in candidate.get("matched_cosines", [])]),
                )
            )


def _record_phi_search_params(
    state,
    *,
    initial_phi,
    selected_candidate,
    selection_reason,
    candidates,
    weight_floor,
    max_redundancy,
    runs_per_step,
    min_run_support,
    min_stability,
    max_fit_loss_frac,
    max_steps,
    expand_factor,
):
    state._record_params(
        {
            "learn_phi": True,
            "learn_phi_initial_phi": float(initial_phi),
            "learn_phi_selected_phi": float(selected_candidate["phi"]),
            "learn_phi_selection_reason": selection_reason,
            "learn_phi_max_redundancy": float(max_redundancy),
            "learn_phi_runs_per_step": int(runs_per_step),
            "learn_phi_min_run_support": float(min_run_support),
            "learn_phi_min_stability": float(min_stability),
            "learn_phi_max_fit_loss_frac": float(max_fit_loss_frac),
            "learn_phi_max_steps": int(max_steps),
            "learn_phi_expand_factor": float(expand_factor),
            "learn_phi_weight_floor": float(weight_floor),
        },
        overwrite=True,
    )
    metric_map = {
        "learn_phi_candidate_phi": "phi",
        "learn_phi_candidate_modal_factor_count": "modal_factor_count",
        "learn_phi_candidate_run_support": "run_support",
        "learn_phi_candidate_stability": "stability",
        "learn_phi_candidate_redundancy": "redundancy",
        "learn_phi_candidate_best_error": "best_error",
        "learn_phi_candidate_best_evidence": "best_evidence",
    }
    for candidate in sorted(candidates, key=lambda row: float(row["phi"])):
        for param_name, candidate_key in metric_map.items():
            value = candidate.get(candidate_key)
            if value is not None:
                state._record_param(param_name, value)


def _learn_phi(
    state,
    *,
    initial_phi,
    seed,
    runs_per_step,
    max_redundancy,
    min_run_support,
    min_stability,
    max_fit_loss_frac,
    max_steps,
    expand_factor,
    weight_floor,
    report_out,
    factor_kwargs,
    log_fn,
    info_level,
):
    min_phi = float(initial_phi) / 1e4
    max_phi = float(initial_phi) * 1e4
    log_tol = 0.15
    candidates_by_phi = {}

    def _evaluate(phi_value):
        existing = _find_candidate_by_phi(candidates_by_phi, float(phi_value))
        if existing is not None:
            return existing
        candidate = _evaluate_phi_candidate(
            state,
            phi=float(phi_value),
            seed=seed,
            runs_per_step=runs_per_step,
            factor_kwargs=factor_kwargs,
            weight_floor=weight_floor,
            log_fn=log_fn,
            info_level=info_level,
        )
        candidates_by_phi[float(candidate["phi"])] = candidate
        return candidate

    start_candidate = _evaluate(initial_phi)
    lower_candidate = None
    upper_candidate = None

    if start_candidate["redundancy"] > max_redundancy:
        lower_candidate = start_candidate
        current_phi = float(initial_phi)
        while current_phi < max_phi:
            current_phi = min(current_phi * expand_factor, max_phi)
            candidate = _evaluate(current_phi)
            if candidate["redundancy"] <= max_redundancy:
                upper_candidate = candidate
                break
            lower_candidate = candidate
            if math.isclose(current_phi, max_phi, rel_tol=1e-12, abs_tol=1e-15):
                break
    else:
        upper_candidate = start_candidate
        current_phi = float(initial_phi)
        while current_phi > min_phi:
            current_phi = max(current_phi / expand_factor, min_phi)
            candidate = _evaluate(current_phi)
            if candidate["redundancy"] > max_redundancy:
                lower_candidate = candidate
                break
            upper_candidate = candidate
            if math.isclose(current_phi, min_phi, rel_tol=1e-12, abs_tol=1e-15):
                break

    if lower_candidate is not None and upper_candidate is not None:
        for _ in range(max_steps):
            interval = math.log(float(upper_candidate["phi"]) / float(lower_candidate["phi"]))
            if interval <= log_tol:
                break
            midpoint_phi = math.sqrt(float(lower_candidate["phi"]) * float(upper_candidate["phi"]))
            midpoint = _evaluate(midpoint_phi)
            if midpoint["redundancy"] > max_redundancy:
                lower_candidate = midpoint
            else:
                upper_candidate = midpoint

    candidates = list(candidates_by_phi.values())
    selected_candidate, selection_reason = _select_phi_candidate(
        candidates,
        max_redundancy=max_redundancy,
        min_run_support=min_run_support,
        min_stability=min_stability,
        max_fit_loss_frac=max_fit_loss_frac,
    )
    _record_phi_search_params(
        state,
        initial_phi=initial_phi,
        selected_candidate=selected_candidate,
        selection_reason=selection_reason,
        candidates=candidates,
        weight_floor=weight_floor,
        max_redundancy=max_redundancy,
        runs_per_step=runs_per_step,
        min_run_support=min_run_support,
        min_stability=min_stability,
        max_fit_loss_frac=max_fit_loss_frac,
        max_steps=max_steps,
        expand_factor=expand_factor,
    )
    _write_phi_search_report(
        report_out,
        candidates,
        selected_phi=selected_candidate["phi"],
        selection_reason=selection_reason,
    )
    log_fn(
        "Selected phi %.6g by automatic tuning [%s]: K_eff=%d, redundancy=%.3g, stability=%.3g, run_support=%.3g"
        % (
            float(selected_candidate["phi"]),
            selection_reason,
            int(selected_candidate["modal_factor_count"]),
            float(selected_candidate["redundancy"]),
            float(selected_candidate["stability"]),
            float(selected_candidate["run_support"]),
        ),
        info_level,
    )
    return selected_candidate


def _build_factor_run_summary(state, *, run_index, seed, evidence, likelihood, reconstruction_error, factor_gene_set_x_pheno):
    return {
        "run_index": int(run_index),
        "seed": None if seed is None else int(seed),
        "evidence": None if evidence is None else float(evidence),
        "likelihood": None if likelihood is None else float(likelihood),
        "reconstruction_error": None if reconstruction_error is None else float(reconstruction_error),
        "num_factors": int(state.num_factors()),
        "factor_gene_set_x_pheno": bool(factor_gene_set_x_pheno),
    }


def _best_run_sort_key(run_summary):
    evidence = run_summary.get("evidence")
    return (
        float("inf") if evidence is None else float(evidence),
        -int(run_summary.get("num_factors", 0)),
        int(run_summary.get("run_index", 0)),
    )


def _finalize_factor_outputs(
    state,
    *,
    factor_gene_set_x_pheno,
    lmm_auth_key,
    lmm_model,
    lmm_provider,
    label_gene_sets_only,
    label_include_phenos,
    label_individually,
    bail_fn,
    warn_fn,
    log_fn,
    info_level,
    labeling_module,
):
    bail = bail_fn
    warn = warn_fn
    log = log_fn
    INFO = info_level

    reorder_inds = np.argsort(-state.factor_relevance)

    state.exp_lambdak = state.exp_lambdak[reorder_inds]
    state.factor_anchor_relevance = state.factor_anchor_relevance[reorder_inds, :]
    state.factor_relevance = state.factor_relevance[reorder_inds]
    if state.exp_gene_factors is not None:
        state.exp_gene_factors = state.exp_gene_factors[:, reorder_inds]
    if state.exp_pheno_factors is not None:
        state.exp_pheno_factors = state.exp_pheno_factors[:, reorder_inds]
    state.exp_gene_set_factors = state.exp_gene_set_factors[:, reorder_inds]

    threshold = 1e-5
    if state.num_factors() > 0:
        if state.exp_gene_factors is not None and np.max(state.exp_gene_factors) > 0:
            state.exp_gene_factors[state.exp_gene_factors < np.max(state.exp_gene_factors) * threshold] = 0
        if state.exp_pheno_factors is not None and np.max(state.exp_pheno_factors) > 0:
            state.exp_pheno_factors[state.exp_pheno_factors < np.max(state.exp_pheno_factors) * threshold] = 0
        if np.max(state.exp_gene_set_factors) > 0:
            state.exp_gene_set_factors[state.exp_gene_set_factors < np.max(state.exp_gene_set_factors) * threshold] = 0

    num_top = 5
    exp_gene_factors_for_top = state.get_factor_loadings(state.exp_gene_factors, loading_type="combined")
    exp_pheno_factors_for_top = state.get_factor_loadings(state.exp_pheno_factors, loading_type="combined")
    exp_gene_set_factors_for_top = state.get_factor_loadings(state.exp_gene_set_factors, loading_type="combined")

    top_anchor_gene_or_pheno_inds = None
    top_anchor_pheno_or_gene_inds = None

    if factor_gene_set_x_pheno:
        top_anchor_gene_or_pheno_inds = np.swapaxes(
            np.argsort(
                -(exp_pheno_factors_for_top).T[:, :, np.newaxis] * (state.pheno_prob_factor_vector)[np.newaxis, :, :],
                axis=1,
            )[:, :num_top, :],
            0,
            1,
        )
        if exp_gene_factors_for_top is not None:
            top_anchor_pheno_or_gene_inds = np.swapaxes(
                np.argsort(
                    -(exp_gene_factors_for_top).T[:, :, np.newaxis] * (state.gene_prob_factor_vector)[np.newaxis, :, :],
                    axis=1,
                )[:, :num_top, :],
                0,
                1,
            )
    else:
        top_anchor_gene_or_pheno_inds = np.swapaxes(
            np.argsort(
                -(exp_gene_factors_for_top).T[:, :, np.newaxis] * (state.gene_prob_factor_vector)[np.newaxis, :, :],
                axis=1,
            )[:, :num_top, :],
            0,
            1,
        )
        if exp_pheno_factors_for_top is not None:
            top_anchor_pheno_or_gene_inds = np.swapaxes(
                np.argsort(
                    -(exp_pheno_factors_for_top).T[:, :, np.newaxis] * (state.pheno_prob_factor_vector)[np.newaxis, :, :],
                    axis=1,
                )[:, :num_top, :],
                0,
                1,
            )

    top_anchor_gene_set_inds = np.swapaxes(
        np.argsort(-exp_gene_set_factors_for_top.T[:, :, np.newaxis] * state.gene_set_prob_factor_vector[np.newaxis, :, :], axis=1)[:, :num_top, :],
        0,
        1,
    )

    top_gene_or_pheno_inds = None
    top_pheno_or_gene_inds = None

    if factor_gene_set_x_pheno:
        top_gene_or_pheno_inds = np.swapaxes(
            np.argsort(
                -(1 - np.prod(1 - ((exp_pheno_factors_for_top).T[:, :, np.newaxis] * (state.pheno_prob_factor_vector)[np.newaxis, :, :]), axis=2)),
                axis=1,
            )[:, :num_top],
            0,
            1,
        )
        if exp_gene_factors_for_top is not None:
            top_pheno_or_gene_inds = np.swapaxes(
                np.argsort(
                    -(1 - np.prod(1 - ((exp_gene_factors_for_top).T[:, :, np.newaxis] * (state.gene_prob_factor_vector)[np.newaxis, :, :]), axis=2)),
                    axis=1,
                )[:, :num_top],
                0,
                1,
            )
    else:
        top_gene_or_pheno_inds = np.swapaxes(
            np.argsort(
                -(1 - np.prod(1 - ((exp_gene_factors_for_top).T[:, :, np.newaxis] * (state.gene_prob_factor_vector)[np.newaxis, :, :]), axis=2)),
                axis=1,
            )[:, :num_top],
            0,
            1,
        )
        if exp_pheno_factors_for_top is not None:
            top_pheno_or_gene_inds = np.swapaxes(
                np.argsort(
                    -(1 - np.prod(1 - ((exp_pheno_factors_for_top).T[:, :, np.newaxis] * (state.pheno_prob_factor_vector)[np.newaxis, :, :]), axis=2)),
                    axis=1,
                )[:, :num_top],
                0,
                1,
            )

    top_gene_set_inds = np.swapaxes(
        np.argsort(-(1 - np.prod(1 - (exp_gene_set_factors_for_top.T[:, :, np.newaxis] * state.gene_set_prob_factor_vector[np.newaxis, :, :]), axis=2)), axis=1)[:, :num_top],
        0,
        1,
    )

    labeling_module.populate_factor_labels(
        state,
        factor_gene_set_x_pheno=factor_gene_set_x_pheno,
        top_gene_set_inds=top_gene_set_inds,
        top_anchor_gene_set_inds=top_anchor_gene_set_inds,
        top_gene_or_pheno_inds=top_gene_or_pheno_inds,
        top_anchor_gene_or_pheno_inds=top_anchor_gene_or_pheno_inds,
        top_pheno_or_gene_inds=top_pheno_or_gene_inds,
        lmm_auth_key=lmm_auth_key,
        lmm_model=lmm_model,
        lmm_provider=lmm_provider,
        label_gene_sets_only=label_gene_sets_only,
        label_include_phenos=label_include_phenos,
        label_individually=label_individually,
        log_fn=log,
        bail_fn=bail,
        warn_fn=warn,
    )

    log("Found %d factors" % state.num_factors(), INFO)


def _run_factor_single(state, max_num_factors=15, phi=1.0, alpha0=10, beta0=1, gene_set_filter_type=None, gene_set_filter_value=None, gene_or_pheno_filter_type=None, gene_or_pheno_filter_value=None, pheno_prune_value=None, pheno_prune_number=None, gene_prune_value=None, gene_prune_number=None, gene_set_prune_value=None, gene_set_prune_number=None, anchor_pheno_mask=None, anchor_gene_mask=None, anchor_any_pheno=False, anchor_any_gene=False, anchor_gene_set=False, run_transpose=True, max_num_iterations=100, rel_tol=1e-4, min_lambda_threshold=1e-3, lmm_auth_key=None, lmm_model=None, lmm_provider="openai", label_gene_sets_only=False, label_include_phenos=False, label_individually=False, keep_original_loadings=False, project_phenos_from_gene_sets=False, *, bail_fn, warn_fn, log_fn, info_level, debug_level, trace_level, labeling_module):
    bail = bail_fn
    warn = warn_fn
    log = log_fn
    INFO = info_level
    DEBUG = debug_level
    TRACE = trace_level

    if state.X_orig is None:
        bail("Cannot run factoring without X")

    # Persist explicit anchor masks for downstream output writers.
    state.anchor_pheno_mask = np.copy(anchor_pheno_mask) if anchor_pheno_mask is not None else None
    state.anchor_gene_mask = np.copy(anchor_gene_mask) if anchor_gene_mask is not None else None

    if (anchor_any_gene or anchor_any_pheno or anchor_gene_set or anchor_gene_mask is not None or anchor_pheno_mask is not None or pheno_prune_value is not None or pheno_prune_number is not None) and state.X_phewas_beta is None:
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

        state._record_params({"anchor": "any_gene"})
        anchor_any_pheno = False
        anchor_pheno_mask = None
        anchor_gene_mask = np.full(state.X_orig.shape[0], True)
        anchor_gene_set = False

    elif anchor_any_pheno:
        if anchor_gene_mask:
            warn("Ignoring anchor gene since anchor any pheno was specified")
        if anchor_pheno_mask:
            warn("Ignoring anchor pheno since anchor any pheno was specified")
        if anchor_gene_set:
            warn("Ignoring anchor gene set since anchor any pheno was specified")
        anchor_gene_mask = None
        anchor_pheno_mask = np.full(state.X_phewas_beta.shape[0], True)
        anchor_gene_set = False
        state._record_params({"anchor": "any_pheno"})
    elif anchor_gene_set:
        if anchor_gene_mask:
            warn("Ignoring anchor gene since anchor gene set was specified")
        if anchor_pheno_mask:
            warn("Ignoring anchor pheno since anchor gene set was specified")
        anchor_gene_mask = None
        anchor_pheno_mask = None
        state._record_params({"anchor": "gene set"})

    # Record the effective anchor masks after option precedence is resolved.
    state.anchor_pheno_mask = np.copy(anchor_pheno_mask) if anchor_pheno_mask is not None else None
    state.anchor_gene_mask = np.copy(anchor_gene_mask) if anchor_gene_mask is not None else None

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
        gene_or_pheno_mask = np.full(state.X_phewas_beta.shape[0], True)
        gene_set_mask = np.full(state.X_phewas_beta.shape[1], True)
        factor_gene_set_x_pheno = True

        combined_prior_Ys = state.gene_pheno_combined_prior_Ys.T if state.gene_pheno_combined_prior_Ys is not None else None
        priors = state.gene_pheno_priors.T if state.gene_pheno_priors is not None else None
        Y = state.gene_pheno_Y.T if state.gene_pheno_Y is not None else None

        state._record_params({"factor_gene_vectors": "gene_pheno.T"})

        if anchor_gene_mask is not None:
            betas = None
            betas_uncorrected = None

            anchor_mask = anchor_gene_mask
            num_users = np.sum(anchor_mask)
            state._record_params({"factor_gene_set_vectors": "None"})

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
            pheno_Y = state.pheno_Y_vs_input_combined_prior_Ys_beta if state.pheno_Y_vs_input_combined_prior_Ys_beta is not None else state.pheno_Y_vs_input_Y_beta if state.pheno_Y_vs_input_Y_beta is not None else state.pheno_Y_vs_input_priors_beta
            if pheno_Y is not None:
                pheno_Y = pheno_Y[:,np.newaxis]
            
            #betas are in external units
            betas = (state.betas / state.scale_factors)[:,np.newaxis] if state.betas is not None else None
            betas_uncorrected = (state.betas_uncorrected / state.scale_factors)[:,np.newaxis] if state.betas_uncorrected is not None else None
            state._record_params({"factor_gene_set_vectors": "betas"})

    else:
        if anchor_pheno_mask is not None and anchor_gene_mask is not None:
            warn("Ignoring anchor gene since anchor pheno was specified")
        anchor_gene_mask = None
        gene_or_pheno_mask = np.full(state.X_orig.shape[0], True)
        gene_set_mask = np.full(state.X_orig.shape[1], True)
        if anchor_pheno_mask is not None:

            anchor_mask = anchor_pheno_mask

            combined_prior_Ys = state.gene_pheno_combined_prior_Ys
            priors = state.gene_pheno_priors
            Y = state.gene_pheno_Y

            state._record_params({"factor_gene_vectors": "gene_pheno"})
            betas = state.X_phewas_beta.T if state.X_phewas_beta is not None else None
            betas_uncorrected = state.X_phewas_beta_uncorrected.T if state.X_phewas_beta_uncorrected is not None else None
            state._record_params({"factor_gene_set_vectors": "X_phewas"})

        else:

            combined_prior_Ys = state.combined_prior_Ys[:,np.newaxis] if state.combined_prior_Ys is not None else None
            priors = state.priors[:,np.newaxis] if state.priors is not None else None
            Y = state.Y[:,np.newaxis] if state.Y is not None else None

            state._record_params({"factor_gene_vectors": "Y"})

            betas = (state.betas / state.scale_factors)[:,np.newaxis] if state.betas is not None else None
            betas_uncorrected = (state.betas_uncorrected / state.scale_factors)[:,np.newaxis] if state.betas_uncorrected is not None else None

            state._record_params({"factor_gene_set_vectors": "betas"})


            #when running the original factoring based off the internal betas and gene scores, we are going to emulate the phewas-like behavior by appending these as the only anchor alongside any gene/pheno loaded values
            #this will allow projection to other phenotypes to happen naturally below
            anchor_mask = np.full(1, True)

            have_phewas = False
            if combined_prior_Ys is not None and state.gene_pheno_combined_prior_Ys is not None:
                combined_prior_Ys = sparse.hstack((state.gene_pheno_combined_prior_Ys, sparse.csc_matrix(combined_prior_Ys))).tocsc()
                have_phewas = True
            if priors is not None and state.gene_pheno_priors is not None:
                priors = sparse.hstack((state.gene_pheno_priors, sparse.csc_matrix(priors))).tocsc()
                have_phewas = True
            if Y is not None and state.gene_pheno_Y is not None:
                Y = sparse.hstack((state.gene_pheno_Y, sparse.csc_matrix(Y))).tocsc()
                have_phewas = True

            if betas is not None and state.X_phewas_beta is not None:
                betas = sparse.hstack((state.X_phewas_beta.T, sparse.csc_matrix(betas))).tocsc()
                have_phewas = True
            if betas_uncorrected is not None and state.X_phewas_beta_uncorrected is not None:
                betas_uncorrected = sparse.hstack((state.X_phewas_beta_uncorrected.T, sparse.csc_matrix(betas_uncorrected))).tocsc()
                have_phewas = True

            if have_phewas:
                #we have phewas for at least one of combined, prior, or Y
                #set those that don't to None
                #otherwise update the internal structures
                if combined_prior_Ys is not None and combined_prior_Ys.shape[1] == 1:
                    combined_prior_Ys = None
                else:
                    state.gene_pheno_combined_prior_Ys = combined_prior_Ys
                    
                if priors is not None and priors.shape[1] == 1:
                    priors = None
                else:
                    state.gene_pheno_priors = priors

                if Y is not None and Y.shape[1] == 1:
                    Y = None
                else:
                    state.gene_pheno_Y = Y
                if betas is not None and betas.shape[1] == 1:
                    betas = None
                else:
                    state.X_phewas_beta = betas.T
                if betas_uncorrected is not None and betas_uncorrected.shape[1] == 1:
                    betas_uncorrected = None
                else:
                    state.X_phewas_beta_uncorrected = betas_uncorrected.T

                state.phenos.append(state.default_pheno)
                state.default_pheno_mask = np.append(np.full(len(state.phenos), False), True)

                #we need to update these as well
                state.pheno_Y_vs_input_Y_beta = np.append(state.pheno_Y_vs_input_Y_beta, 0) if state.pheno_Y_vs_input_Y_beta is not None else None
                state.pheno_Y_vs_input_Y_beta_tilde = np.append(state.pheno_Y_vs_input_Y_beta_tilde, 0) if state.pheno_Y_vs_input_Y_beta_tilde is not None else None
                state.pheno_Y_vs_input_Y_se = np.append(state.pheno_Y_vs_input_Y_se, 0) if state.pheno_Y_vs_input_Y_se is not None else None
                state.pheno_Y_vs_input_Y_Z = np.append(state.pheno_Y_vs_input_Y_Z, 0) if state.pheno_Y_vs_input_Y_Z is not None else None
                state.pheno_Y_vs_input_Y_p_value = np.append(state.pheno_Y_vs_input_Y_p_value, 1) if state.pheno_Y_vs_input_Y_p_value is not None else None

                state.pheno_combined_prior_Ys_vs_input_Y_beta = np.append(state.pheno_combined_prior_Ys_vs_input_Y_beta, 0) if state.pheno_combined_prior_Ys_vs_input_Y_beta is not None else None
                state.pheno_combined_prior_Ys_vs_input_Y_beta_tilde = np.append(state.pheno_combined_prior_Ys_vs_input_Y_beta_tilde, 0) if state.pheno_combined_prior_Ys_vs_input_Y_beta_tilde is not None else None
                state.pheno_combined_prior_Ys_vs_input_Y_se = np.append(state.pheno_combined_prior_Ys_vs_input_Y_se, 0) if state.pheno_combined_prior_Ys_vs_input_Y_se is not None else None
                state.pheno_combined_prior_Ys_vs_input_Y_Z = np.append(state.pheno_combined_prior_Ys_vs_input_Y_Z, 0) if state.pheno_combined_prior_Ys_vs_input_Y_Z is not None else None
                state.pheno_combined_prior_Ys_vs_input_Y_p_value = np.append(state.pheno_combined_prior_Ys_vs_input_Y_p_value, 1) if state.pheno_combined_prior_Ys_vs_input_Y_p_value is not None else None

                state.pheno_Y_vs_input_combined_prior_Ys_beta = np.append(state.pheno_Y_vs_input_combined_prior_Ys_beta, 0) if state.pheno_Y_vs_input_combined_prior_Ys_beta is not None else None
                state.pheno_Y_vs_input_combined_prior_Ys_beta_tilde = np.append(state.pheno_Y_vs_input_combined_prior_Ys_beta_tilde, 0) if state.pheno_Y_vs_input_combined_prior_Ys_beta_tilde is not None else None
                state.pheno_Y_vs_input_combined_prior_Ys_se = np.append(state.pheno_Y_vs_input_combined_prior_Ys_se, 0) if state.pheno_Y_vs_input_combined_prior_Ys_se is not None else None
                state.pheno_Y_vs_input_combined_prior_Ys_Z = np.append(state.pheno_Y_vs_input_combined_prior_Ys_Z, 0) if state.pheno_Y_vs_input_combined_prior_Ys_Z is not None else None
                state.pheno_Y_vs_input_combined_prior_Ys_p_value = np.append(state.pheno_Y_vs_input_combined_prior_Ys_p_value, 1) if state.pheno_Y_vs_input_combined_prior_Ys_p_value is not None else None

                state.pheno_combined_prior_Ys_vs_input_combined_prior_Ys_beta = np.append(state.pheno_combined_prior_Ys_vs_input_combined_prior_Ys_beta, 0) if state.pheno_combined_prior_Ys_vs_input_combined_prior_Ys_beta is not None else None
                state.pheno_combined_prior_Ys_vs_input_combined_prior_Ys_beta_tilde = np.append(state.pheno_combined_prior_Ys_vs_input_combined_prior_Ys_beta_tilde, 0) if state.pheno_combined_prior_Ys_vs_input_combined_prior_Ys_beta_tilde is not None else None
                state.pheno_combined_prior_Ys_vs_input_combined_prior_Ys_se = np.append(state.pheno_combined_prior_Ys_vs_input_combined_prior_Ys_se, 0) if state.pheno_combined_prior_Ys_vs_input_combined_prior_Ys_se is not None else None
                state.pheno_combined_prior_Ys_vs_input_combined_prior_Ys_Z = np.append(state.pheno_combined_prior_Ys_vs_input_combined_prior_Ys_Z, 0) if state.pheno_combined_prior_Ys_vs_input_combined_prior_Ys_Z is not None else None
                state.pheno_combined_prior_Ys_vs_input_combined_prior_Ys_p_value = np.append(state.pheno_combined_prior_Ys_vs_input_combined_prior_Ys_p_value, 1) if state.pheno_combined_prior_Ys_vs_input_combined_prior_Ys_p_value is not None else None

                state.pheno_Y_vs_input_priors_beta = np.append(state.pheno_Y_vs_input_priors_beta, 0) if state.pheno_Y_vs_input_priors_beta is not None else None
                state.pheno_Y_vs_input_priors_beta_tilde = np.append(state.pheno_Y_vs_input_priors_beta_tilde, 0) if state.pheno_Y_vs_input_priors_beta_tilde is not None else None
                state.pheno_Y_vs_input_priors_se = np.append(state.pheno_Y_vs_input_priors_se, 0) if state.pheno_Y_vs_input_priors_se is not None else None
                state.pheno_Y_vs_input_priors_Z = np.append(state.pheno_Y_vs_input_priors_Z, 0) if state.pheno_Y_vs_input_priors_Z is not None else None
                state.pheno_Y_vs_input_priors_p_value = np.append(state.pheno_Y_vs_input_priors_p_value, 1) if state.pheno_Y_vs_input_priors_p_value is not None else None

                state.pheno_combined_prior_Ys_vs_input_priors_beta = np.append(state.pheno_combined_prior_Ys_vs_input_priors_beta, 0) if state.pheno_combined_prior_Ys_vs_input_priors_beta is not None else None
                state.pheno_combined_prior_Ys_vs_input_priors_beta_tilde = np.append(state.pheno_combined_prior_Ys_vs_input_priors_beta_tilde, 0) if state.pheno_combined_prior_Ys_vs_input_priors_beta_tilde is not None else None
                state.pheno_combined_prior_Ys_vs_input_priors_se = np.append(state.pheno_combined_prior_Ys_vs_input_priors_se, 0) if state.pheno_combined_prior_Ys_vs_input_priors_se is not None else None
                state.pheno_combined_prior_Ys_vs_input_priors_Z = np.append(state.pheno_combined_prior_Ys_vs_input_priors_Z, 0) if state.pheno_combined_prior_Ys_vs_input_priors_Z is not None else None
                state.pheno_combined_prior_Ys_vs_input_priors_p_value = np.append(state.pheno_combined_prior_Ys_vs_input_priors_p_value, 1) if state.pheno_combined_prior_Ys_vs_input_priors_p_value is not None else None

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
            log("Found %d %ss remaining after pruning to max number (of %d)" % (np.sum(all_prune_mask), tag, len(state.phenos)))
        return all_prune_mask

    if pheno_prune_value is not None or pheno_prune_number is not None:
        mask_for_pruning = gene_or_pheno_mask if factor_gene_set_x_pheno else anchor_pheno_mask
        if mask_for_pruning is not None:
        
            if factor_gene_set_x_pheno:
                log("Pruning phenos to reduce matrix size", DEBUG)
            else:
                log("Pruning phenos to reduce number of anchors", DEBUG)                    

            pheno_sort_rank = -state.X_phewas_beta.mean(axis=1).A1 if state.X_phewas_beta is not None else np.arange(len(mask_for_pruning))
            #now if we request pruning
            if pheno_prune_value is not None:
                pheno_prune_mask = state._prune_gene_sets(pheno_prune_value, X_orig=state.X_phewas_beta_uncorrected[mask_for_pruning,:].T, gene_sets=[state.phenos[i] for i in np.where(mask_for_pruning)[0]], rank_vector=pheno_sort_rank[mask_for_pruning], do_internal_pruning=False)
                log("Found %d phenos remaining after pruning (of %d)" % (np.sum(pheno_prune_mask), len(state.phenos)))

                mask_for_pruning[np.where(mask_for_pruning)[0][~pheno_prune_mask]] = False

            if pheno_prune_number is not None:
                (mean_shifts, scale_factors) = state._calc_X_shift_scale(state.X_phewas_beta_uncorrected[mask_for_pruning,:].T)
                pheno_prune_number_masks = state._compute_gene_set_batches(V=None, X_orig=state.X_phewas_beta_uncorrected[mask_for_pruning,:].T, mean_shifts=mean_shifts, scale_factors=scale_factors, sort_values=pheno_sort_rank[mask_for_pruning], stop_at=pheno_prune_number, tag="phenos")
                all_pheno_prune_mask = __combine_prune_masks(pheno_prune_number_masks, pheno_prune_number, pheno_sort_rank[mask_for_pruning], "pheno")
                mask_for_pruning[np.where(mask_for_pruning)[0][~all_pheno_prune_mask]] = False
            if mask_for_pruning is anchor_pheno_mask and num_users > 1:
                #in this case, we may have changed the number of users
                num_users = np.sum(anchor_pheno_mask)

    if not anchor_gene_set and (gene_prune_value is not None or gene_prune_number is not None):
        mask_for_pruning = gene_or_pheno_mask if not factor_gene_set_x_pheno else anchor_gene_mask
        if mask_for_pruning is not None:
            gene_sort_rank = -state.combined_prior_Ys if state.combined_prior_Ys is not None else -state.Y if state.Y is not None else -state.priors if state.priors is not None else np.arange(len(mask_for_pruning))
            if not factor_gene_set_x_pheno:
                log("Pruning genes to reduce matrix size", DEBUG)
            else:
                log("Pruning genes to reduce number of anchors", DEBUG)                    


            #now if we request pruning
            if gene_prune_value is not None:
                gene_prune_mask = state._prune_gene_sets(gene_prune_value, X_orig=state.X_orig[mask_for_pruning,:].T, gene_sets=[state.genes[i] for i in np.where(mask_for_pruning)[0]], rank_vector=gene_sort_rank[mask_for_pruning], do_internal_pruning=False)
                log("Found %d genes remaining after pruning (of %d)" % (np.sum(gene_prune_mask), len(state.genes)))

                mask_for_pruning[np.where(mask_for_pruning)[0][~gene_prune_mask]] = False

            if gene_prune_number is not None:
                (mean_shifts, scale_factors) = state._calc_X_shift_scale(state.X_orig[mask_for_pruning,:].T)
                gene_prune_number_masks = state._compute_gene_set_batches(V=None, X_orig=state.X_orig[mask_for_pruning,:].T, mean_shifts=mean_shifts, scale_factors=scale_factors, sort_values=gene_sort_rank[mask_for_pruning], stop_at=gene_prune_number, tag="genes")
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
            gene_or_pheno_full_prob_vector_data = np.exp(gene_or_pheno_full_vector.data + state.background_log_bf)
            gene_or_pheno_full_prob_vector_data = gene_or_pheno_full_prob_vector_data / (1 + gene_or_pheno_full_prob_vector_data)
            gene_or_pheno_full_prob_vector = copy.copy(gene_or_pheno_full_vector)
            gene_or_pheno_full_prob_vector.data = gene_or_pheno_full_prob_vector_data
        else:
            gene_or_pheno_full_prob_vector = np.exp(gene_or_pheno_full_vector + state.background_log_bf) / (1 + np.exp(gene_or_pheno_full_vector + state.background_log_bf))

    if anchor_gene_set:
        gene_or_pheno_prob_vector = np.exp(gene_or_pheno_vector + state.background_log_bf) / (1 + np.exp(gene_or_pheno_vector + state.background_log_bf)) if gene_or_pheno_vector is not None else np.ones((len(gene_or_pheno_mask), num_users))
    else:
        gene_or_pheno_prob_vector = gene_or_pheno_full_prob_vector[:,anchor_mask] if gene_or_pheno_full_prob_vector is not None else np.ones((len(gene_or_pheno_mask), num_users))

    if gene_or_pheno_prob_vector is not None and sparse.issparse(gene_or_pheno_prob_vector):
        gene_or_pheno_prob_vector = gene_or_pheno_prob_vector.toarray()

    if anchor_any_gene or anchor_any_pheno:
        #only have one user
        gene_or_pheno_any_prob_vector = 1 - np.prod(1 - gene_or_pheno_prob_vector, axis=1)
        gene_or_pheno_prob_vector = gene_or_pheno_any_prob_vector[:,np.newaxis]

    if factor_gene_set_x_pheno:
        state.pheno_prob_factor_vector = gene_or_pheno_prob_vector
        state.gene_prob_factor_vector = None
    else:
        state.gene_prob_factor_vector = gene_or_pheno_prob_vector
        state.pheno_prob_factor_vector = None

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


    gene_set_sort_rank = -(state.X_phewas_beta_uncorrected.mean(axis=0).A1 if state.X_phewas_beta_uncorrected is not None else state.betas)

    if gene_set_prune_value is not None or gene_set_prune_number is not None:
        log("Pruning gene sets to reduce matrix size", DEBUG)

    if gene_set_prune_value is not None:
        gene_set_prune_mask = state._prune_gene_sets(gene_set_prune_value, X_orig=state.X_orig[:,gene_set_mask], gene_sets=[state.gene_sets[i] for i in np.where(gene_set_mask)[0]], rank_vector=gene_set_sort_rank[gene_set_mask], do_internal_pruning=False)

        log("Found %d gene_sets remaining after pruning (of %d)" % (np.sum(gene_set_prune_mask), len(state.gene_sets)))
        gene_set_mask[np.where(gene_set_mask)[0][~gene_set_prune_mask]] = False

    if gene_set_prune_number is not None:
        gene_set_prune_number_masks = state._compute_gene_set_batches(V=None, X_orig=state.X_orig[:,gene_set_mask], mean_shifts=state.mean_shifts[gene_set_mask], scale_factors=state.scale_factors[gene_set_mask], sort_values=gene_set_sort_rank[gene_set_mask], stop_at=pheno_prune_number, tag="gene sets")

        all_gene_set_prune_mask = __combine_prune_masks(gene_set_prune_number_masks, gene_set_prune_number, gene_set_sort_rank[gene_set_mask], "gene set")

        gene_set_mask[np.where(gene_set_mask)[0][~all_gene_set_prune_mask]] = False
    
    gene_set_full_prob_vector = None
    if gene_set_full_vector is not None:
        if sparse.issparse(gene_set_full_vector):
            gene_set_full_prob_vector_data = np.exp(gene_set_full_vector.data + state.background_log_bf)
            gene_set_full_prob_vector_data = gene_set_full_prob_vector_data / (1 + gene_set_full_prob_vector_data)
            gene_set_full_prob_vector = copy.copy(gene_set_full_vector)
            gene_set_full_prob_vector.data = gene_set_full_prob_vector_data
        else:
            gene_set_full_prob_vector = np.exp(gene_set_full_vector + state.background_log_bf) / (1 + np.exp(gene_set_full_vector + state.background_log_bf))

    gene_set_prob_vector = gene_set_full_prob_vector[:,anchor_mask] if gene_set_full_prob_vector is not None else np.ones((len(gene_set_mask), num_users))

    if gene_set_prob_vector is not None and sparse.issparse(gene_set_prob_vector):
        gene_set_prob_vector = gene_set_prob_vector.toarray()

    if anchor_any_gene or anchor_any_pheno:
        #only have one user
        gene_set_any_prob_vector = 1 - np.prod(1 - gene_set_prob_vector, axis=1)
        gene_set_prob_vector = gene_set_any_prob_vector[:,np.newaxis]

    state.gene_set_prob_vector = gene_set_full_prob_vector

    state._record_params({"max_num_factors": max_num_factors, "alpha0": alpha0, "phi": phi, "gene_set_filter_type": gene_set_filter_type, "gene_set_filter_value": gene_set_filter_value, "gene_or_pheno_filter_type": gene_or_pheno_filter_type, "gene_or_pheno_filter_value": gene_or_pheno_filter_value, "pheno_prune_value": pheno_prune_value, "pheno_prune_number": pheno_prune_number, "gene_set_prune_value": gene_set_prune_value, "gene_set_prune_number": gene_set_prune_number, "run_transpose": run_transpose})


    matrix = state.X_phewas_beta_uncorrected.T if factor_gene_set_x_pheno else state.X_orig.T

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

    result = state._bayes_nmf_l2_extension(matrix.toarray(), gene_set_prob_vector[gene_set_mask,:], gene_or_pheno_prob_vector[gene_or_pheno_mask,:], a0=alpha0, K=max_num_factors, tol=rel_tol, phi=phi, cap_genes=cap, cap_gene_sets=cap, normalize_genes=normalize_genes, normalize_gene_sets=normalize_gene_sets)

    state.exp_lambdak = result[4]
    exp_gene_or_pheno_factors = result[1].T
    state.exp_gene_set_factors = result[0]

    #subset_out the weak factors
    factor_mask = (state.exp_lambdak > min_lambda_threshold) & (np.sum(exp_gene_or_pheno_factors, axis=0) > min_lambda_threshold) & (np.sum(state.exp_gene_set_factors, axis=0) > min_lambda_threshold)
    factor_mask = factor_mask & (np.max(state.exp_gene_set_factors, axis=0) > 1e-5 * np.max(state.exp_gene_set_factors))
    if np.sum(~factor_mask) > 0:
        state.exp_lambdak = state.exp_lambdak[factor_mask]
        exp_gene_or_pheno_factors = exp_gene_or_pheno_factors[:,factor_mask]
        state.exp_gene_set_factors = state.exp_gene_set_factors[:,factor_mask]

    if factor_gene_set_x_pheno:
        state.pheno_factor_pheno_mask = gene_or_pheno_mask
        state.exp_pheno_factors = exp_gene_or_pheno_factors
        state.pheno_prob_factor_vector = gene_or_pheno_prob_vector
        state.gene_prob_factor_vector = None
    else:
        state.gene_factor_gene_mask = gene_or_pheno_mask            
        state.exp_gene_factors = exp_gene_or_pheno_factors
        state.gene_prob_factor_vector = gene_or_pheno_prob_vector
        state.pheno_prob_factor_vector = None

    state.gene_set_prob_factor_vector = gene_set_prob_vector
    state.gene_set_factor_gene_set_mask = gene_set_mask

    #now project the additional genes/phenos/gene sets onto the factors

    log("Projecting factors", TRACE)

    #this code gets the "relevance" values
    #first get the probabilities for either the genotypes or phenotypes (whichever we didn't use to factor)
    #these need to be specific to the anchors
    if factor_gene_set_x_pheno:
        if gene_or_pheno_full_prob_vector is not None:
            state.gene_prob_factor_vector = state._nnls_project_matrix(state.pheno_prob_factor_vector, gene_or_pheno_full_prob_vector.T)
            state._record_params({"factor_gene_prob_from": "phenos"})
        else:
            state.gene_prob_factor_vector = state._nnls_project_matrix(state.gene_set_prob_factor_vector, state.X_orig)
            state._record_params({"factor_gene_prob_from": "gene_sets"})
    else:
        if gene_or_pheno_full_prob_vector is not None:
            state.pheno_prob_factor_vector = state._nnls_project_matrix(state.gene_prob_factor_vector, gene_or_pheno_full_prob_vector.T)
            state._record_params({"factor_pheno_prob_from": "genes"})
        elif state.X_phewas_beta_uncorrected is not None:
            state.pheno_prob_factor_vector = state._nnls_project_matrix(state.gene_set_prob_factor_vector, state.X_phewas_beta_uncorrected)
            state._record_params({"factor_pheno_prob_from": "gene_sets"})

    if state.gene_set_prob_factor_vector is not None and sparse.issparse(state.gene_set_prob_factor_vector):
        state.gene_set_prob_factor_vector = state.gene_set_prob_factor_vector.toarray()
    if state.gene_prob_factor_vector is not None and sparse.issparse(state.gene_prob_factor_vector):
        state.gene_prob_factor_vector = state.gene_prob_factor_vector.toarray()
    if state.pheno_prob_factor_vector is not None and sparse.issparse(state.pheno_prob_factor_vector):
        state.pheno_prob_factor_vector = state.pheno_prob_factor_vector.toarray()

    gene_matrix_to_project = state.X_orig.T
    if not run_transpose:
        gene_matrix_to_project = gene_matrix_to_project.T

    #this code projects to the additional dimensions

    #all gene factor values
    full_gene_factor_values = state._project_H_with_fixed_W(state.exp_gene_set_factors, gene_matrix_to_project[state.gene_set_factor_gene_set_mask,:], state.gene_set_prob_factor_vector[state.gene_set_factor_gene_set_mask,:], state.gene_prob_factor_vector, phi=phi, tol=rel_tol, cap_genes=cap, normalize_genes=normalize_genes)
    if not factor_gene_set_x_pheno and keep_original_loadings:
        full_gene_factor_values[state.gene_factor_gene_mask,:] = state.exp_gene_factors

    #all pheno factor values, either from the phewas used to factor or the phewas passed in to project
    full_pheno_factor_values = state.exp_pheno_factors
    pheno_matrix_to_project = None

    if state.exp_gene_factors is None and state.exp_gene_set_factors is None:
        bail("Something went wrong: both gene factors and gene set factors are empty")

    if state.X_phewas_beta_uncorrected is not None and state.pheno_prob_factor_vector is not None:
        if project_phenos_from_gene_sets or state.exp_gene_factors is None:
            pheno_matrix_to_project = state.X_phewas_beta_uncorrected.T
            if not run_transpose:
                pheno_matrix_to_project = pheno_matrix_to_project.T

            full_pheno_factor_values = state._project_H_with_fixed_W(state.exp_gene_set_factors, pheno_matrix_to_project if state.gene_set_factor_gene_set_mask is None else pheno_matrix_to_project[state.gene_set_factor_gene_set_mask,:], state.gene_set_prob_factor_vector if state.gene_set_factor_gene_set_mask is None else state.gene_set_prob_factor_vector[state.gene_set_factor_gene_set_mask,:], state.pheno_prob_factor_vector, phi=phi, tol=rel_tol, cap_genes=cap, normalize_genes=normalize_genes)
        else:
            pheno_matrix_to_project = state.gene_pheno_Y
            if not run_transpose:
                pheno_matrix_to_project = pheno_matrix_to_project.T

            full_pheno_factor_values = state._project_H_with_fixed_W(state.exp_gene_factors, pheno_matrix_to_project if state.gene_factor_gene_mask is None else pheno_matrix_to_project[state.gene_factor_gene_mask,:], state.gene_prob_factor_vector if state.gene_factor_gene_mask is None else state.gene_prob_factor_vector[state.gene_factor_gene_mask,:], state.pheno_prob_factor_vector, phi=phi, tol=rel_tol, cap_genes=cap, normalize_genes=normalize_genes)

            
        if keep_original_loadings:
            full_pheno_factor_values[state.pheno_factor_pheno_mask,:] = state.exp_pheno_factors

    #now gene set factor values, projecting from either phenos or genes depending on what was used
    if factor_gene_set_x_pheno and pheno_matrix_to_project is not None:
        #we have to swap the gene sets and genes, which means transposing the matrix to project and swapping the prios
        full_gene_set_factor_values = state._project_H_with_fixed_W(state.exp_pheno_factors, pheno_matrix_to_project[:,state.pheno_factor_pheno_mask].T if run_transpose else pheno_matrix_to_project[state.pheno_factor_pheno_mask,:].T, state.pheno_prob_factor_vector[state.pheno_factor_pheno_mask,:], state.gene_set_prob_factor_vector, phi=phi, tol=rel_tol, cap_genes=cap, normalize_genes=normalize_gene_sets)
    else:
        full_gene_set_factor_values = state._project_H_with_fixed_W(state.exp_gene_factors, gene_matrix_to_project[:,state.gene_factor_gene_mask].T if run_transpose else gene_matrix_to_project[state.gene_factor_gene_mask,:].T, state.gene_prob_factor_vector[state.gene_factor_gene_mask,:], state.gene_set_prob_factor_vector, phi=phi, tol=rel_tol, cap_genes=cap, normalize_genes=normalize_gene_sets)

    if keep_original_loadings:
        full_gene_set_factor_values[state.gene_set_factor_gene_set_mask,:] = state.exp_gene_set_factors

    #update these to store the imputed as well
    state.exp_gene_factors = full_gene_factor_values
    state.exp_pheno_factors = full_pheno_factor_values
    state.exp_gene_set_factors = full_gene_set_factor_values

    if factor_gene_set_x_pheno:
        exp_gene_or_pheno_factors = state.exp_pheno_factors
    else:
        exp_gene_or_pheno_factors = state.exp_gene_factors

    #now update relevance

    matrix_to_mult = state.exp_pheno_factors if factor_gene_set_x_pheno else state.exp_gene_factors
    vector_to_mult = state.pheno_prob_factor_vector if factor_gene_set_x_pheno else state.gene_prob_factor_vector

    #matrix_to_mult: (genes, factors)
    #vector_to_mult: (users, genes)
    #want: (factors, users)

    state.factor_anchor_relevance = state._nnls_project_matrix(matrix_to_mult, vector_to_mult.T, max_value=1).T
    state.factor_relevance = _compute_any_anchor_relevance(state.factor_anchor_relevance)

    _finalize_factor_outputs(
        state,
        factor_gene_set_x_pheno=factor_gene_set_x_pheno,
        lmm_auth_key=lmm_auth_key,
        lmm_model=lmm_model,
        lmm_provider=lmm_provider,
        label_gene_sets_only=label_gene_sets_only,
        label_include_phenos=label_include_phenos,
        label_individually=label_individually,
        bail_fn=bail,
        warn_fn=warn,
        log_fn=log,
        info_level=INFO,
        labeling_module=labeling_module,
    )

    return _build_factor_run_summary(
        state,
        run_index=0,
        seed=None,
        evidence=result[3],
        likelihood=result[2],
        reconstruction_error=result[5],
        factor_gene_set_x_pheno=factor_gene_set_x_pheno,
    )


def _run_factor_with_seed(state, *, seed, run_index, factor_kwargs):
    def _runner():
        return _run_factor_single(state, **factor_kwargs)

    summary = _run_with_numpy_seed(seed, _runner)
    if summary is None:
        summary = _build_factor_run_summary(
            state,
            run_index=run_index,
            seed=seed,
            evidence=None,
            likelihood=None,
            reconstruction_error=None,
            factor_gene_set_x_pheno=bool(state.exp_pheno_factors is not None and state.gene_factor_gene_mask is None),
        )
    summary["run_index"] = int(run_index)
    summary["seed"] = None if seed is None else int(seed)
    return summary


def _collect_consensus_matches(run_states, run_summaries, reference_index, min_factor_cosine):
    reference_factors = run_states[reference_index].exp_gene_set_factors
    reference_norm = _normalize_factor_columns(reference_factors)
    factor_matches = {idx: [(reference_index, idx, 1.0)] for idx in range(reference_factors.shape[1])}

    for run_index, run_state in enumerate(run_states):
        if run_index == reference_index:
            continue
        run_factors = run_state.exp_gene_set_factors
        if run_factors is None or run_factors.size == 0:
            continue
        run_norm = _normalize_factor_columns(run_factors)
        similarity = reference_norm.T @ run_norm
        ref_inds, run_inds = scipy.optimize.linear_sum_assignment(1.0 - similarity)
        for ref_factor_index, run_factor_index in zip(ref_inds, run_inds):
            cosine = float(similarity[ref_factor_index, run_factor_index])
            if cosine >= min_factor_cosine:
                factor_matches[ref_factor_index].append((run_index, run_factor_index, cosine))
    return factor_matches


def _apply_consensus_solution(
    reference_state,
    run_states,
    run_summaries,
    *,
    min_run_support,
    aggregation,
    min_factor_cosine,
    factor_kwargs,
):
    eligible_indices = [idx for idx, state in enumerate(run_states) if state.exp_gene_set_factors is not None and state.exp_gene_set_factors.size > 0 and state.num_factors() > 0]
    if len(eligible_indices) == 0:
        return _clone_runtime_state(run_states[min(range(len(run_summaries)), key=lambda idx: _best_run_sort_key(run_summaries[idx]))]), {
            "mode": "consensus",
            "reference_run_index": None,
            "run_summaries": copy.deepcopy(run_summaries),
            "factor_support": [],
        }

    factor_count_to_indices = {}
    for idx in eligible_indices:
        factor_count_to_indices.setdefault(run_summaries[idx]["num_factors"], []).append(idx)
    modal_factor_count = max(sorted(factor_count_to_indices), key=lambda key: (len(factor_count_to_indices[key]), key))
    reference_candidates = factor_count_to_indices[modal_factor_count]
    reference_index = min(reference_candidates, key=lambda idx: _best_run_sort_key(run_summaries[idx]))

    consensus_state = _clone_runtime_state(run_states[reference_index])
    factor_matches = _collect_consensus_matches(run_states, run_summaries, reference_index, min_factor_cosine)

    kept_factor_columns = []
    kept_gene_columns = []
    kept_pheno_columns = []
    kept_lambda = []
    kept_anchor_relevance = []
    kept_factor_relevance = []
    factor_support_rows = []
    total_runs = max(1, len(run_states))

    for reference_factor_index in range(run_states[reference_index].exp_gene_set_factors.shape[1]):
        matched = factor_matches.get(reference_factor_index, [])
        support_fraction = float(len(matched)) / float(total_runs)
        factor_support_rows.append(
            {
                "reference_factor_index": int(reference_factor_index),
                "support_runs": int(len(matched)),
                "support_fraction": support_fraction,
                "kept": bool(support_fraction >= min_run_support),
                "matched_run_indices": [int(run_index) for run_index, _, _ in matched],
                "matched_cosines": [float(cosine) for _, _, cosine in matched],
            }
        )
        if support_fraction < min_run_support:
            continue

        gene_set_stack = np.stack(
            [run_states[run_index].exp_gene_set_factors[:, factor_index] for run_index, factor_index, _ in matched],
            axis=0,
        )
        kept_factor_columns.append(_aggregate_consensus_stack(gene_set_stack, aggregation))

        gene_stack = [
            run_states[run_index].exp_gene_factors[:, factor_index]
            for run_index, factor_index, _ in matched
            if run_states[run_index].exp_gene_factors is not None
        ]
        if len(gene_stack) > 0:
            kept_gene_columns.append(_aggregate_consensus_stack(np.stack(gene_stack, axis=0), aggregation))

        pheno_stack = [
            run_states[run_index].exp_pheno_factors[:, factor_index]
            for run_index, factor_index, _ in matched
            if run_states[run_index].exp_pheno_factors is not None
        ]
        if len(pheno_stack) > 0:
            kept_pheno_columns.append(_aggregate_consensus_stack(np.stack(pheno_stack, axis=0), aggregation))

        kept_lambda.append(
            _aggregate_consensus_stack(
                np.array([run_states[run_index].exp_lambdak[factor_index] for run_index, factor_index, _ in matched], dtype=float),
                aggregation,
            )
        )
        kept_anchor_relevance.append(
            _aggregate_consensus_stack(
                np.stack([run_states[run_index].factor_anchor_relevance[factor_index, :] for run_index, factor_index, _ in matched], axis=0),
                aggregation,
            )
        )
        kept_factor_relevance.append(
            _aggregate_consensus_stack(
                np.array([run_states[run_index].factor_relevance[factor_index] for run_index, factor_index, _ in matched], dtype=float),
                aggregation,
            )
        )

    if len(kept_factor_columns) == 0:
        consensus_state = _clone_runtime_state(run_states[reference_index])
        diagnostics = {
            "mode": "consensus",
            "reference_run_index": int(reference_index),
            "run_summaries": copy.deepcopy(run_summaries),
            "factor_support": factor_support_rows,
            "fallback_to_reference": True,
        }
        return consensus_state, diagnostics

    consensus_state.exp_gene_set_factors = np.column_stack(kept_factor_columns)
    consensus_state.exp_gene_factors = np.column_stack(kept_gene_columns) if len(kept_gene_columns) > 0 else None
    consensus_state.exp_pheno_factors = np.column_stack(kept_pheno_columns) if len(kept_pheno_columns) > 0 else None
    consensus_state.exp_lambdak = np.asarray(kept_lambda, dtype=float)
    consensus_state.factor_anchor_relevance = np.vstack(kept_anchor_relevance)
    consensus_state.factor_relevance = np.asarray(kept_factor_relevance, dtype=float)

    _finalize_factor_outputs(
        consensus_state,
        factor_gene_set_x_pheno=bool(run_summaries[reference_index]["factor_gene_set_x_pheno"]),
        lmm_auth_key=factor_kwargs["lmm_auth_key"],
        lmm_model=factor_kwargs["lmm_model"],
        lmm_provider=factor_kwargs["lmm_provider"],
        label_gene_sets_only=factor_kwargs["label_gene_sets_only"],
        label_include_phenos=factor_kwargs["label_include_phenos"],
        label_individually=factor_kwargs["label_individually"],
        bail_fn=factor_kwargs["bail_fn"],
        warn_fn=factor_kwargs["warn_fn"],
        log_fn=factor_kwargs["log_fn"],
        info_level=factor_kwargs["info_level"],
        labeling_module=factor_kwargs["labeling_module"],
    )

    diagnostics = {
        "mode": "consensus",
        "reference_run_index": int(reference_index),
        "run_summaries": copy.deepcopy(run_summaries),
        "factor_support": factor_support_rows,
        "modal_factor_count": int(modal_factor_count),
        "aggregation": aggregation,
        "min_factor_cosine": float(min_factor_cosine),
        "min_run_support": float(min_run_support),
    }
    return consensus_state, diagnostics


def run_factor(state, max_num_factors=15, phi=1.0, alpha0=10, beta0=1, seed=None, factor_runs=1, consensus_nmf=False, consensus_min_factor_cosine=0.7, consensus_min_run_support=0.5, consensus_aggregation="median", consensus_stats_out=None, learn_phi=False, learn_phi_max_redundancy=0.6, learn_phi_runs_per_step=5, learn_phi_min_run_support=0.6, learn_phi_min_stability=0.85, learn_phi_max_fit_loss_frac=0.05, learn_phi_max_steps=8, learn_phi_expand_factor=10.0, learn_phi_weight_floor=None, learn_phi_report_out=None, gene_set_filter_type=None, gene_set_filter_value=None, gene_or_pheno_filter_type=None, gene_or_pheno_filter_value=None, pheno_prune_value=None, pheno_prune_number=None, gene_prune_value=None, gene_prune_number=None, gene_set_prune_value=None, gene_set_prune_number=None, anchor_pheno_mask=None, anchor_gene_mask=None, anchor_any_pheno=False, anchor_any_gene=False, anchor_gene_set=False, run_transpose=True, max_num_iterations=100, rel_tol=1e-4, min_lambda_threshold=1e-3, lmm_auth_key=None, lmm_model=None, lmm_provider="openai", label_gene_sets_only=False, label_include_phenos=False, label_individually=False, keep_original_loadings=False, project_phenos_from_gene_sets=False, *, bail_fn, warn_fn, log_fn, info_level, debug_level, trace_level, labeling_module):
    bail = bail_fn
    log = log_fn
    INFO = info_level

    if factor_runs < 1:
        bail("--factor-runs must be at least 1")
    if consensus_aggregation not in {"median", "mean"}:
        bail("--consensus-aggregation must be one of: median, mean")
    if not (0 < consensus_min_factor_cosine <= 1):
        bail("--consensus-min-factor-cosine must be in (0, 1]")
    if not (0 < consensus_min_run_support <= 1):
        bail("--consensus-min-run-support must be in (0, 1]")
    if consensus_nmf and factor_runs < 2:
        bail("--consensus-nmf requires --factor-runs >= 2")
    if learn_phi:
        if phi <= 0:
            bail("--learn-phi requires --phi > 0")
        if not (0 < learn_phi_max_redundancy <= 1):
            bail("--learn-phi-max-redundancy must be in (0, 1]")
        if learn_phi_runs_per_step < 1:
            bail("--learn-phi-runs-per-step must be at least 1")
        if not (0 < learn_phi_min_run_support <= 1):
            bail("--learn-phi-min-run-support must be in (0, 1]")
        if not (0 < learn_phi_min_stability <= 1):
            bail("--learn-phi-min-stability must be in (0, 1]")
        if learn_phi_max_fit_loss_frac < 0:
            bail("--learn-phi-max-fit-loss-frac must be >= 0")
        if learn_phi_max_steps < 1:
            bail("--learn-phi-max-steps must be at least 1")
        if learn_phi_expand_factor <= 1:
            bail("--learn-phi-expand-factor must be > 1")
        if learn_phi_weight_floor is not None and learn_phi_weight_floor < 0:
            bail("--learn-phi-weight-floor must be >= 0 when set")

    factor_kwargs = {
        "max_num_factors": max_num_factors,
        "phi": phi,
        "alpha0": alpha0,
        "beta0": beta0,
        "gene_set_filter_type": gene_set_filter_type,
        "gene_set_filter_value": gene_set_filter_value,
        "gene_or_pheno_filter_type": gene_or_pheno_filter_type,
        "gene_or_pheno_filter_value": gene_or_pheno_filter_value,
        "pheno_prune_value": pheno_prune_value,
        "pheno_prune_number": pheno_prune_number,
        "gene_prune_value": gene_prune_value,
        "gene_prune_number": gene_prune_number,
        "gene_set_prune_value": gene_set_prune_value,
        "gene_set_prune_number": gene_set_prune_number,
        "anchor_pheno_mask": anchor_pheno_mask,
        "anchor_gene_mask": anchor_gene_mask,
        "anchor_any_pheno": anchor_any_pheno,
        "anchor_any_gene": anchor_any_gene,
        "anchor_gene_set": anchor_gene_set,
        "run_transpose": run_transpose,
        "max_num_iterations": max_num_iterations,
        "rel_tol": rel_tol,
        "min_lambda_threshold": min_lambda_threshold,
        "lmm_auth_key": lmm_auth_key,
        "lmm_model": lmm_model,
        "lmm_provider": lmm_provider,
        "label_gene_sets_only": label_gene_sets_only,
        "label_include_phenos": label_include_phenos,
        "label_individually": label_individually,
        "keep_original_loadings": keep_original_loadings,
        "project_phenos_from_gene_sets": project_phenos_from_gene_sets,
        "bail_fn": bail_fn,
        "warn_fn": warn_fn,
        "log_fn": log_fn,
        "info_level": info_level,
        "debug_level": debug_level,
        "trace_level": trace_level,
        "labeling_module": labeling_module,
    }

    state._record_params(
        _build_factor_param_record(
            max_num_factors=max_num_factors,
            phi=phi,
            alpha0=alpha0,
            beta0=beta0,
            seed=seed,
            factor_runs=factor_runs,
            consensus_nmf=consensus_nmf,
            consensus_min_factor_cosine=consensus_min_factor_cosine,
            consensus_min_run_support=consensus_min_run_support,
            consensus_aggregation=consensus_aggregation,
            consensus_stats_out=consensus_stats_out,
            learn_phi=learn_phi,
            learn_phi_max_redundancy=learn_phi_max_redundancy,
            learn_phi_runs_per_step=learn_phi_runs_per_step,
            learn_phi_min_run_support=learn_phi_min_run_support,
            learn_phi_min_stability=learn_phi_min_stability,
            learn_phi_max_fit_loss_frac=learn_phi_max_fit_loss_frac,
            learn_phi_max_steps=learn_phi_max_steps,
            learn_phi_expand_factor=learn_phi_expand_factor,
            learn_phi_weight_floor=learn_phi_weight_floor,
            learn_phi_report_out=learn_phi_report_out,
            gene_set_filter_type=gene_set_filter_type,
            gene_set_filter_value=gene_set_filter_value,
            gene_or_pheno_filter_type=gene_or_pheno_filter_type,
            gene_or_pheno_filter_value=gene_or_pheno_filter_value,
            pheno_prune_value=pheno_prune_value,
            pheno_prune_number=pheno_prune_number,
            gene_prune_value=gene_prune_value,
            gene_prune_number=gene_prune_number,
            gene_set_prune_value=gene_set_prune_value,
            gene_set_prune_number=gene_set_prune_number,
            anchor_pheno_mask=anchor_pheno_mask,
            anchor_gene_mask=anchor_gene_mask,
            anchor_any_pheno=anchor_any_pheno,
            anchor_any_gene=anchor_any_gene,
            anchor_gene_set=anchor_gene_set,
            run_transpose=run_transpose,
            max_num_iterations=max_num_iterations,
            rel_tol=rel_tol,
            min_lambda_threshold=min_lambda_threshold,
            lmm_auth_key=lmm_auth_key,
            lmm_model=lmm_model,
            lmm_provider=lmm_provider,
            label_gene_sets_only=label_gene_sets_only,
            label_include_phenos=label_include_phenos,
            label_individually=label_individually,
            keep_original_loadings=keep_original_loadings,
            project_phenos_from_gene_sets=project_phenos_from_gene_sets,
        ),
        overwrite=True,
    )
    log(
        "Factor config: phi=%.6g alpha0=%.6g beta0=%.6g max_num_factors=%d factor_runs=%d consensus_nmf=%s learn_phi=%s max_num_iterations=%d rel_tol=%.3g"
        % (
            float(phi),
            float(alpha0),
            float(beta0),
            int(max_num_factors),
            int(factor_runs),
            bool(consensus_nmf),
            bool(learn_phi),
            int(max_num_iterations),
            float(rel_tol),
        ),
        INFO,
    )

    if learn_phi:
        weight_floor = 0.01 if learn_phi_weight_floor is None else float(learn_phi_weight_floor)
        selected_candidate = _learn_phi(
            state,
            initial_phi=phi,
            seed=seed,
            runs_per_step=learn_phi_runs_per_step,
            max_redundancy=learn_phi_max_redundancy,
            min_run_support=learn_phi_min_run_support,
            min_stability=learn_phi_min_stability,
            max_fit_loss_frac=learn_phi_max_fit_loss_frac,
            max_steps=learn_phi_max_steps,
            expand_factor=learn_phi_expand_factor,
            weight_floor=weight_floor,
            report_out=learn_phi_report_out,
            factor_kwargs=factor_kwargs,
            log_fn=log,
            info_level=INFO,
        )
        phi = float(selected_candidate["phi"])
        factor_kwargs["phi"] = phi
        state._record_params({"phi": phi}, overwrite=True)
        log("Using learned phi %.6g for final factorization" % float(phi), INFO)
    else:
        log("Using fixed phi %.6g for factorization" % float(phi), INFO)

    if factor_runs == 1 and not consensus_nmf:
        summary = _run_factor_with_seed(state, seed=seed, run_index=0, factor_kwargs=factor_kwargs)
        state.consensus_mode = "single"
        state.consensus_reference_run = 0
        state.consensus_run_diagnostics = [summary]
        state.consensus_factor_support = []
        return

    child_seeds = _derive_factor_run_seeds(seed, factor_runs)
    run_states = []
    run_summaries = []
    for run_index, child_seed in enumerate(child_seeds):
        log("Running factor restart %d/%d%s" % (run_index + 1, factor_runs, "" if child_seed is None else " [seed=%d]" % child_seed), INFO)
        run_state = _clone_runtime_state(state)
        summary = _run_factor_with_seed(run_state, seed=child_seed, run_index=run_index, factor_kwargs=factor_kwargs)
        run_states.append(run_state)
        run_summaries.append(summary)

    if consensus_nmf:
        selected_state, diagnostics = _apply_consensus_solution(
            state,
            run_states,
            run_summaries,
            min_run_support=consensus_min_run_support,
            aggregation=consensus_aggregation,
            min_factor_cosine=consensus_min_factor_cosine,
            factor_kwargs=factor_kwargs,
        )
    else:
        best_run_index = min(range(len(run_summaries)), key=lambda idx: _best_run_sort_key(run_summaries[idx]))
        selected_state = run_states[best_run_index]
        diagnostics = {
            "mode": "best_of_n",
            "reference_run_index": int(best_run_index),
            "run_summaries": copy.deepcopy(run_summaries),
            "factor_support": [],
        }

    _replace_runtime_state(state, selected_state)
    state.consensus_mode = diagnostics["mode"]
    state.consensus_reference_run = diagnostics["reference_run_index"]
    state.consensus_run_diagnostics = diagnostics["run_summaries"]
    state.consensus_factor_support = diagnostics["factor_support"]
    state.consensus_stats_out = consensus_stats_out
