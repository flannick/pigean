# Frozen requirements-v4 selection, qualified on six archived trait inputs.
"""Local observed-evidence preservation; all results remain provisional."""
import numpy as np

PHIS = (0.005, 0.01, 0.02)


def align_candidate(genes, weights, target, target_indices, canonical_genes=None):
    """Align the separately ordered target and loadings by explicit gene identity."""
    genes = list(genes)
    canonical = genes if canonical_genes is None else list(canonical_genes)
    indices = np.asarray(target_indices)
    if (len(set(genes)) != len(genes) or len(set(canonical)) != len(canonical)
            or set(genes) != set(canonical)
            or sorted(indices.tolist()) != list(range(len(genes)))):
        raise ValueError('Requirements selection needs identical, unique discovery genes and a target permutation')
    lookup = {g: i for i, g in enumerate(genes)}
    order = [lookup[g] for g in canonical]
    target_order = np.argsort(indices)[order]
    return (np.asarray(weights)[order],
            np.asarray(target)[np.ix_(target_order, target_order)].copy())


def select(bases, raws, target, qualified):
    """Frozen scientific decision with a fail-closed numerical qualification layer.

    Failed fits remain in diagnostics. They never turn a partial comparison into
    a qualified recommendation; a qualified fixed-0.01 fit remains the fallback.
    """
    if any(set(x) != set(PHIS) for x in (bases, raws, qualified)):
        raise ValueError('Requirements selection uses exactly 0.005, 0.01, 0.02')
    target = np.asarray(target, dtype=float).copy()
    if (target.ndim != 2 or target.shape[0] != target.shape[1] or len(target) < 2
            or not np.isfinite(target).all() or (target < 0).any()
            or not np.allclose(target, target.T, rtol=0, atol=1e-12)):
        raise ValueError('Requirements target must be a finite nonnegative symmetric matrix')
    np.fill_diagonal(target, 0)
    for phi in PHIS:
        if len(raws[phi]) != 5:
            raise ValueError('Requirements selection needs five restarts per candidate')
        for w in [bases[phi], *raws[phi]]:
            if (w.ndim != 2 or w.shape[0] != len(target)
                    or not np.isfinite(w).all() or (w < 0).any()):
                raise ValueError('Invalid or misaligned requirements loadings')
    result = _evaluate_frozen(bases, raws, target, qualified)
    result['method'] = 'requirements-v4'
    result['all_candidates_qualified'] = all(qualified.values())
    result['fallback_qualified'] = bool(qualified[0.01])
    result['proposed_phi'] = result['selected_phi']
    if not result['all_candidates_qualified']:
        result['selected_phi'] = 0.01
        result['status'] = 'incomplete_numerical_qualification'
    if result['status'] != 'provisional_resolution':
        result['selected_phi'] = 0.01
    result['selection_qualified'] = bool(result['all_candidates_qualified']
                                         and result['status'] == 'provisional_resolution')
    return result
from ._phi_requirements_profiles import evaluate as old_components

def components(bases, raws, target, qualified, contrast_as_filter=False, **kwargs):
    d = old_components(bases, raws, target, qualified, **kwargs)
    for p in d['profiles']:
        p['contrast_exclusion_control'] = any((x in p['reason'].split(';') for x in ['global_enrichment', 'boundary_undefined', 'local_contrast']))
        p['contrast_descriptive_flag'] = 'undefined_boundary' if not p['boundary_defined'] else 'low_contrast' if p['contrast'] < kwargs.get('contrast', 1.25) else ''
        if not contrast_as_filter:
            reasons = [x for x in p['reason'].split(';') if x in ['numerical_ineligible', 'restart_support', 'cross_phi_persistence']]
            p['eligible'] = not reasons
            p['reason'] = ';'.join(reasons)
    d['supported_profiles'] = sum((p['eligible'] for p in d['profiles']))
    return d

def _evaluate_frozen(bases, raws, target, qualified, cosine=0.9, residual=0.2, contrast=1.25, tolerance=0.2, **kwargs):
    d = components(bases, raws, target, qualified, cosine=cosine, residual=residual, contrast=contrast, **kwargs)
    d.update(claims_unique_partition=False, adequacy_certified=False, interpretation='Preserves relative fit within tested views; neither unique biological partition nor absolute adequacy certified')
    refs = [p for p in d['profiles'] if p['eligible']]
    phis = sorted(bases)
    d['misses'] = []
    d['errors'] = []
    if not refs:
        return d
    H = np.column_stack([bases[p['phi']][:, p['factor'] - 1] / max(bases[p['phi']][:, p['factor'] - 1].max(), 1e-30) for p in refs])
    T = np.array(target, copy=True)
    np.fill_diagonal(T, 0)
    T2 = T * T
    weights = np.outer(H.sum(axis=0), H.sum(axis=0)) - H.T @ H
    energy = H.T @ T2 @ H
    density = np.divide(energy, weights, out=np.zeros_like(energy), where=weights > 1e-12)
    inside = np.diag(density)
    scale = np.maximum(density, np.sqrt(np.outer(inside, inside)))
    valid = (weights > 1e-12) & (scale > 1e-30)
    (ii, jj) = np.triu_indices(len(refs))
    ok = valid[ii, jj]
    all_errors = []
    global_errors = []
    for phi in phis:
        P = bases[phi] @ bases[phi].T
        np.fill_diagonal(P, 0)
        E = (T - P) ** 2
        mean = np.divide(H.T @ E @ H, weights, out=np.zeros_like(energy), where=weights > 1e-12)
        local = np.sqrt(np.maximum(0, np.divide(mean, scale, out=np.zeros_like(mean), where=valid)))
        all_errors.append(local[ii, jj])
        global_errors.append(float(np.sqrt(E.sum() / max(T2.sum(), 1e-30))))
    err = np.array(all_errors)
    good = np.array([qualified[p] for p in phis])
    best = err[good].min(axis=0)
    global_best = min((g for (g, q) in zip(global_errors, good) if q))
    baseviews = {v['phi']: v for v in d['views']}
    views = []
    for (a, phi) in enumerate(phis):
        excess = err[a] - best
        failed = ok & (excess > tolerance + 1e-12)
        global_failed = global_errors[a] > global_best + tolerance + 1e-12
        redundant = baseviews[phi]['redundant']
        admissible = bool(qualified[phi] and ok.any() and (not failed.any()) and (not global_failed) and (not kwargs.get('reject_redundancy', True) or redundant == 0))
        views.append(dict(phi=phi, qualified=bool(qualified[phi]), factors=bases[phi].shape[1], redundant=redundant, failed_regions=int(failed.sum()), failed_within=int((failed & (ii == jj)).sum()), failed_between=int((failed & (ii != jj)).sum()), unevaluable_regions=int((~ok).sum()), max_excess=float(excess[ok].max(initial=0)), global_error=global_errors[a], global_excess=global_errors[a] - global_best, admissible=admissible))
        for (k, (i, j)) in enumerate(zip(ii, jj)):
            row = dict(candidate_phi=phi, source_phi=refs[i]['phi'], source_factor=refs[i]['factor'], other_phi=refs[j]['phi'], other_factor=refs[j]['factor'], kind='within' if i == j else 'between', evaluable=bool(ok[k]), error=float(err[a, k]), best_error=float(best[k]), excess=float(excess[k]), failed=bool(failed[k]))
            d['errors'].append(row)
            if failed[k]:
                d['misses'].append(row)
    d['views'] = views
    admissible = [v['phi'] for v in views if v['admissible']]
    if admissible:
        mid = np.mean(np.log([min(admissible), max(admissible)]))
        dist = {p: abs(np.log(p) - mid) for p in admissible}
        m = min(dist.values())
        chosen = min([p for p in admissible if dist[p] <= m + 1e-12], key=lambda p: (abs(np.log(p / 0.01)), p))
        d.update(selected_phi=chosen, status='provisional_resolution')
    else:
        d.update(selected_phi=0.01, status='unresolved_fallback')
    if refs and all((not p['boundary_defined'] for p in refs)) and (d['status'] == 'provisional_resolution'):
        d['status'] = 'background_or_unresolved'
    d['contrast_used_as_exclusion'] = bool(kwargs.get('contrast_as_filter', False))
    return d
