"""Useful-leaf v1: constrained parsimony, not a biological probability model.

See plans/EAGGL_leaf_quality_rule_v1_20261118.md for frozen rationale/limits.
No gene names, annotation names, per-gene membership penalty, or refitting.
"""
from dataclasses import dataclass, asdict
import numpy as np
from scipy.optimize import linear_sum_assignment


@dataclass(frozen=True)
class Policy:
    minimum_effective_genes: float = 3.
    minimum_contrast: float = .20
    duplicate_cosine: float = .90
    support_cosine: float = .70
    minimum_support: int = 3
    maximum_flagged_fraction: float = .10
    maximum_flagged_energy: float = .10
    epsilon: float = .02
    replication_cosine: float = .90
    replication_amplitude_min: float = .80
    replication_amplitude_max: float = 1.25

    def __post_init__(self):
        for name, value in asdict(self).items():
            if not np.isfinite(value):
                raise ValueError(f'{name} must be finite')
        if self.epsilon < 0 or self.minimum_effective_genes < 1:
            raise ValueError('Error allowance must be nonnegative and minimum effective genes at least 1')
        for name in ('minimum_contrast', 'maximum_flagged_fraction', 'maximum_flagged_energy', 'replication_cosine'):
            if not 0 <= getattr(self, name) <= 1:
                raise ValueError(f'{name} must be in [0, 1]')
        for name in ('duplicate_cosine', 'support_cosine'):
            if not 0 < getattr(self, name) <= 1:
                raise ValueError(f'{name} must be in (0, 1]')
        if self.minimum_support not in range(1, 6):
            raise ValueError('Minimum restart support must be 1 through 5')
        if not 0 < self.replication_amplitude_min <= 1 <= self.replication_amplitude_max:
            raise ValueError('Replication amplitude bounds must be positive and include 1')

    def to_dict(self):
        return asdict(self)


def normalize(w):
    w = np.asarray(w, dtype=float)
    norm = np.linalg.norm(w, axis=0)
    return np.divide(w, norm, out=np.zeros_like(w), where=norm > 0)


def restart_matches(w, raws, cosine=.7):
    """For each start and consensus leaf, return the matched raw column, or -1."""
    matches = np.full((len(raws), w.shape[1]), -1, dtype=int)
    consensus = normalize(w)
    for row, raw in zip(matches, raws):
        sim = normalize(raw).T @ consensus
        i, j = linear_sum_assignment(-sim)
        keep = sim[i, j] >= cosine
        row[j[keep]] = i[keep]
    return matches


def restart_support(w, raws, cosine=.7):
    return (restart_matches(w, raws, cosine) >= 0).sum(axis=0)


def leaf_metrics(w, counts=None, policy=Policy()):
    w = np.asarray(w, dtype=float)
    if w.ndim != 2 or w.shape[0] < 2 or not np.isfinite(w).all() or (w < 0).any():
        raise ValueError('Expected finite nonnegative gene by factor loadings')
    n, k = w.shape
    energy = (w*w).sum(0)
    fourth = (w**4).sum(0)
    effective = np.divide(energy**2, fourth, out=np.zeros(k), where=fourth > 0)
    contrast = (np.sqrt(n) - np.divide(w.sum(0), np.sqrt(energy),
                out=np.zeros(k), where=energy > 0)) / (np.sqrt(n)-1)
    contrast = np.where(energy > 0, np.clip(contrast, 0., 1.), 0.)
    sim = normalize(w).T @ normalize(w)
    np.fill_diagonal(sim, 0.)
    duplicate = sim.max(1, initial=0.) >= policy.duplicate_cosine
    small = effective < policy.minimum_effective_genes
    uniform = contrast < policy.minimum_contrast
    unsupported = np.ones(k, dtype=bool) if counts is None else np.asarray(counts) < policy.minimum_support
    flagged = small | uniform | duplicate | unsupported
    fraction = float(flagged.mean()) if k else 1.
    energy_fraction = float(energy[flagged].sum()/energy.sum()) if energy.sum() > 0 else 1.
    rows = []
    for j in range(k):
        rows.append(dict(factor=j+1, energy=float(energy[j]), effective_genes=float(effective[j]),
                         contrast=float(contrast[j]), maximum=float(w[:,j].max()),
                         half_maximum_genes=int((w[:,j] >= .5*w[:,j].max()).sum()) if energy[j] else 0,
                         support=None if counts is None else int(counts[j]),
                         nearest_cosine=float(sim[j].max(initial=0.)),
                         small=bool(small[j]), uniform=bool(uniform[j]), duplicate=bool(duplicate[j]),
                         unsupported=bool(unsupported[j]), flagged=bool(flagged[j])))
    summary = dict(factors=k, flagged=int(flagged.sum()), flagged_fraction=fraction,
                   flagged_energy=energy_fraction, small=int(small.sum()), uniform=int(uniform.sum()),
                   duplicate=int(duplicate.sum()), duplicate_pairs=int(np.triu(sim >= policy.duplicate_cosine,1).sum()),
                   unsupported=int(unsupported.sum()), support_known=counts is not None,
                   leaf_usable=bool(k and counts is not None and fraction <= policy.maximum_flagged_fraction+1e-12
                                    and energy_fraction <= policy.maximum_flagged_energy+1e-12))
    return rows, summary


def relationship_error(target, prediction, mask=None):
    t = np.asarray(target, dtype=float).copy()
    p = np.asarray(prediction, dtype=float).copy()
    if t.shape != p.shape or t.ndim != 2 or t.shape[0] != t.shape[1]:
        raise ValueError('Target and prediction must be matching square matrices')
    if not np.isfinite(t).all() or not np.isfinite(p).all():
        raise ValueError('Nonfinite relationship matrix')
    m = np.ones_like(t, dtype=bool) if mask is None else np.asarray(mask, dtype=bool).copy()
    np.fill_diagonal(m, False)
    t[~m] = 0.; p[~m] = 0.
    tnorm = np.linalg.norm(t, axis=1)
    enorm = np.linalg.norm(p-t, axis=1)
    nonzero = tnorm > 0
    if not nonzero.any():
        raise ValueError('No nonzero target relationships')
    rows = enorm[nonzero]/tnorm[nonzero]
    g = float(np.linalg.norm(p-t)/np.linalg.norm(t))
    b = float(rows.mean())
    return dict(global_error=g, balanced_error=b, distortion=max(g,b),
                row_error_q90=float(np.quantile(rows,.9)), nonzero_target_rows=int(nonzero.sum()),
                zero_target_rows=int((~nonzero).sum()), zero_row_prediction_norm=float(np.linalg.norm(p[~nonzero])))


def select(records, policy=Policy()):
    eligible = [r for r in records if r['numerically_qualified'] and r['leaf_usable']]
    if not eligible:
        return dict(selected=None, reason='no_usable_candidate', policy=policy.to_dict())
    best = min(r['distortion'] for r in eligible)
    within = [r for r in eligible if r['distortion'] <= best+policy.epsilon+1e-12]
    winner = min(within, key=lambda r:(r['factors'],r['distortion'],r['phi']))
    phis = [r['phi'] for r in records]
    return dict(selected=winner['id'], selected_phi=winner['phi'], factors=winner['factors'],
                minimum_distortion=best, selected_distortion=winner['distortion'],
                excess_distortion=winner['distortion']-best,
                eligible=[r['id'] for r in eligible], within_tolerance=[r['id'] for r in within],
                sampled_boundary=bool(winner['phi'] in (min(phis),max(phis))),
                reason='fewest_leaves_within_utility_tolerance', policy=policy.to_dict())
