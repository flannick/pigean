# Frozen requirements-v4 selection, qualified on six archived trait inputs.
"""Three isolated research revisions; no fitted matrices are modified."""
import numpy as np
from ._phi_requirements_metrics import norm, pair_residual, support

def local_evidence(v, target):
    h = v / max(float(v.max(initial=0)), 1e-30)
    g = 1 - h
    inner = h.sum() ** 2 - h @ h
    cross = h.sum() * g.sum() - h @ g
    within = float(h @ target @ h / max(inner, 1e-30))
    boundary = float(h @ target @ g / max(cross, 1e-30))
    return (within, boundary, bool(inner > 1e-12 and cross > 1e-12))

def evaluate(bases, raws, target, qualified, cosine=0.9, residual=0.2, contrast=1.25, require_persistence=False, global_enrichment=False, reject_redundancy=True):
    phis = sorted(bases)
    profiles = []
    refs = []
    bg = target.sum() / (len(target) * (len(target) - 1))
    unique_by_phi = {}
    for phi in phis:
        w = bases[phi]
        s = support(raws[phi], w, cosine)
        unique_by_phi[phi] = []
        for k in range(w.shape[1]):
            col = w[:, k:k + 1]
            v = col[:, 0]
            unique = float(pair_residual(np.delete(w, k, axis=1), col)[0])
            unique_by_phi[phi].append(unique)
            persist = 1 + sum((bool(bases[p].shape[1] and (norm(col).T @ norm(bases[p])).max() >= cosine) for p in phis if p != phi))
            (within, boundary, defined) = local_evidence(v, target)
            evidence = bool(within > 0 and (within >= contrast * bg if global_enrichment else defined and within >= contrast * boundary))
            reasons = []
            if not qualified[phi]:
                reasons.append('numerical_ineligible')
            if s[k] < 3:
                reasons.append('restart_support')
            if require_persistence and persist < 2:
                reasons.append('cross_phi_persistence')
            if not evidence:
                reasons.append('global_enrichment' if global_enrichment else 'boundary_undefined' if not defined else 'local_contrast')
            eligible = not reasons
            profiles.append(dict(phi=phi, factor=k + 1, support=int(s[k]), persistence=persist, uniqueness=unique, within=within, boundary=boundary, boundary_defined=defined, contrast=float(within / max(boundary, 1e-30)), eligible=eligible, reason=';'.join(reasons)))
            if eligible:
                refs.append(col)
    reference = np.column_stack(refs) if refs else np.empty((len(target), 0))
    views = []
    misses = []
    eligible_profiles = [p for p in profiles if p['eligible']]
    for phi in phis:
        errs = pair_residual(bases[phi], reference)
        redundant = sum((u < residual for u in unique_by_phi[phi]))
        bad = errs > residual
        views.append(dict(phi=phi, qualified=bool(qualified[phi]), factors=bases[phi].shape[1], redundant=redundant, missed_supported=int(bad.sum()), max_supported_residual=float(errs.max(initial=0)), admissible=bool(qualified[phi] and len(errs) > 0 and (not bad.any()) and (not reject_redundancy or redundant == 0))))
        misses.extend((dict(candidate_phi=phi, source_phi=p['phi'], source_factor=p['factor'], pair_residual=float(e)) for (p, e) in zip(eligible_profiles, errs) if e > residual))
    admissible = [v['phi'] for v in views if v['admissible']]
    if admissible:
        mid = np.mean(np.log([min(admissible), max(admissible)]))
        distance = {p: abs(np.log(p) - mid) for p in admissible}
        best = min(distance.values())
        ties = [p for p in admissible if distance[p] <= best + 1e-12]
        chosen = min(ties, key=lambda p: (abs(np.log(p / 0.01)), p))
        status = 'provisional_resolution'
    else:
        chosen = 0.01
        status = 'numerical_ineligible' if not any(qualified.values()) else 'unresolved_fallback' if refs else 'background_or_unresolved'
    return dict(selected_phi=chosen, status=status, supported_profiles=len(refs), profiles=profiles, views=views, misses=misses, interpretation='Resolution is provisional; no discrete biological truth inferred')
