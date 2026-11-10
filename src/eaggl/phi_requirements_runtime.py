"""Full-panel requirements selection using unchanged native fit kernels."""
import csv
import gzip
import json
from pathlib import Path

import numpy as np

from .phi_requirements import align_candidate, select
from .phi_search import INITIAL_ORDER, next_step, validate


def write_report(path, decision):
    if path is None:
        return
    from .outputs import _json_safe
    opener = gzip.open if str(path).endswith('.gz') else open
    rows = [dict(phi=v['phi'], selected=v['phi'] == decision['selected_phi'],
                 status=decision['status'], selection_qualified=decision['selection_qualified'],
                 **{k: value for k, value in v.items() if k != 'phi'})
            for v in decision['views']]
    with opener(path, 'wt') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter='\t')
        writer.writeheader(); writer.writerows(rows)
    with gzip.open(str(path) + '.details.json.gz', 'wt') as handle:
        json.dump(_json_safe(decision), handle, allow_nan=False)


def run(state, *, seed, factor_kwargs, consensus_stats_out, report_out,
        consensus_min_run_support, consensus_min_factor_cosine,
        search_extension="improvement", search_max_extra=2):
    from . import factor_runtime as fr
    validate(search_extension, search_max_extra)
    candidates, bases, raws, qualified, diagnostics_by_phi = {}, {}, {}, {}, {}
    canonical, target = None, None
    child_seeds = fr._derive_factor_run_seeds(seed, 5)
    pending = list(INITIAL_ORDER)
    search_history, fit_order = [], []
    while pending:
        phi = pending.pop(0)
        fit_order.append(phi)
        kwargs = dict(factor_kwargs, phi=phi, factor_gmt_out=None)
        run_states, summaries = [], []
        for index, child in enumerate(child_seeds):
            kwargs['log_fn']('Requirements phi %.6g restart %d/5' % (phi, index + 1), kwargs['info_level'])
            candidate = fr._clone_runtime_state(state)
            summary = fr._run_factor_with_seed(candidate, seed=child, run_index=index, factor_kwargs=kwargs)
            run_states.append(candidate); summaries.append(summary)
        chosen, consensus = fr._apply_consensus_solution(
            state, run_states, summaries, min_run_support=consensus_min_run_support,
            aggregation='median', min_factor_cosine=consensus_min_factor_cosine,
            factor_kwargs=kwargs, matching='pooled')
        if canonical is None:
            canonical = list(chosen.genes)
        weights = chosen.exp_gene_factors
        if weights is None:
            weights = np.zeros((len(chosen.genes), 0))
        w, t = align_candidate(chosen.genes, weights,
                               chosen._phi_selection_target_matrix,
                               chosen._phi_selection_target_gene_indices, canonical)
        np.fill_diagonal(t, 0)
        if target is None:
            target = t
        elif not np.allclose(target, t, rtol=0, atol=1e-12):
            raise ValueError('Requirements candidates changed the gene-pair target')
        bases[phi] = w; raws[phi] = []
        for candidate in run_states:
            lookup = {g: i for i, g in enumerate(candidate.genes)}
            if set(lookup) != set(canonical):
                raise ValueError('Requirements restart gene identities changed')
            weights = candidate.exp_gene_factors
            if weights is None:
                weights = np.zeros((len(candidate.genes), 0))
            raws[phi].append(weights[[lookup[g] for g in canonical]].copy())
        projections = getattr(chosen, 'internal_projection_diagnostics', [])
        qualified[phi] = bool(all(s['converged'] and not s.get('hit_iteration_cap', False) for s in summaries)
                              and not consensus.get('fallback_to_reference', False)
                              and projections and all(p['converged'] for p in projections)
                              and w.shape[1] > 0)
        diagnostics_by_phi[phi] = dict(
            qualified=qualified[phi],
            restarts=[{k: v for k, v in s.items() if k != 'objective_trace'} for s in summaries],
            projections=projections, consensus_fallback=bool(consensus.get('fallback_to_reference', False)))
        candidates[phi] = (chosen, consensus)
        del run_states
        if not pending:
            decision = select(bases, raws, target, qualified)
            step = next_step(decision, search_extension, search_max_extra)
            search_history.append(step)
            if step['next_phi'] is not None:
                pending.append(step['next_phi'])
    decision['search'] = dict(extension=search_extension, max_extra=search_max_extra,
                             fit_order=fit_order, history=search_history,
                             stop_reason=search_history[-1]['reason'])
    factor_kwargs['log_fn']('Requirements search stopped: %s; %d candidates (%d extra)' %
                           (search_history[-1]['reason'], len(fit_order), len(fit_order)-3), factor_kwargs['info_level'])
    decision['candidate_diagnostics'] = diagnostics_by_phi
    decision['child_seeds'] = child_seeds
    state.phi_requirements_diagnostics = decision
    write_report(report_out, decision)
    if not decision['fallback_qualified']:
        raise ValueError('Requirements comparison has no numerically qualified phi0.01 fallback; inspect the selection report')
    if not decision['selection_qualified']:
        factor_kwargs['warn_fn']('Requirements comparison is unqualified (%s); using qualified fixed phi0.01' % decision['status'])
    phi = decision['selected_phi']
    chosen, consensus = candidates[phi]
    fr._replace_runtime_state(state, chosen)
    state.phi_requirements_diagnostics = decision
    state.consensus_mode = consensus['mode']
    state.consensus_reference_run = consensus['reference_run_index']
    state.consensus_run_diagnostics = consensus['run_summaries']
    state.consensus_factor_support = consensus['factor_support']
    state.consensus_fallback_to_reference = bool(consensus.get('fallback_to_reference', False))
    state.consensus_stats_out = consensus_stats_out
    state._record_params(dict(phi=phi, phi_selection_objective='requirements',
                              phi_selection_status=decision['status'],
                              phi_selection_qualified=decision['selection_qualified'],
                              phi_search_extension=search_extension, phi_search_max_extra=search_max_extra,
                              phi_search_candidates=len(fit_order), phi_search_stop_reason=search_history[-1]['reason'],
                              consensus_final_status=state.consensus_mode,
                              consensus_fallback_to_reference=state.consensus_fallback_to_reference), overwrite=True)
    factor_kwargs['log_fn']('Requirements selection: phi %.6g; status=%s; qualified=%s' %
                           (phi, decision['status'], decision['selection_qualified']), factor_kwargs['info_level'])
    fr._write_final_factor_gmt(state, dict(factor_kwargs, phi=phi))
