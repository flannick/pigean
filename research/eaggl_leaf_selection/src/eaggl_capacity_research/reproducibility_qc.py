"""Descriptive factor QC; never modifies fitted loadings or predictions.

Inputs are one-to-one matches of marginal pair contributions. Scores and
thresholds are model-agnostic; callers must construct the correct contribution
for their reconstruction link. Factor identifiers are one-based.
"""
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment


def contribution_matches(a, b):
    """Match columns by contribution cosine, using the existing assignment rule."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.ndim != 2 or b.ndim != 2 or a.shape[0] != b.shape[0]:
        raise ValueError('Contributions must share a pair coordinate system')
    if not (np.isfinite(a).all() and np.isfinite(b).all()):
        raise ValueError('Nonfinite contributions')
    na, nb = np.linalg.norm(a, axis=0), np.linalg.norm(b, axis=0)
    if np.any(na <= 0) or np.any(nb <= 0):
        raise ValueError('Zero contributions must be marked unassessed separately')
    c = np.clip((a/na).T @ (b/nb), -1, 1)
    i, j = linear_sum_assignment(-c)
    return pd.DataFrame(dict(factor_A=i+1, factor_B=j+1, cosine=c[i,j],
        amplitude_A_to_B=c[i,j]*nb[j]/na[i],
        amplitude_B_to_A=c[i,j]*na[i]/nb[j]))


def flag_matches(matches, n_a, n_b, cosine_min=.9, amplitude_min=.8,
                 amplitude_max=1.25, comparison_available=True):
    """Return every factor, including unmatched or unassessed factors.

    Reproducible means supported by this comparison, not validated biology or
    a frequentist confidence level. A presentation mask does not prune a model.
    """
    if not (0 <= cosine_min <= 1 and 0 < amplitude_min <= 1 <= amplitude_max):
        raise ValueError('Invalid thresholds')
    pairs = matches.to_dict('records')
    for side, n in [('A', n_a), ('B', n_b)]:
        ids = [p['factor_'+side] for p in pairs]
        if len(set(ids)) != len(ids) or any(int(i) != i or not 1 <= i <= n for i in ids):
            raise ValueError('Matches must be one-to-one with valid factor IDs')
    for p in pairs:
        c, ab, ba = [float(p[k]) for k in ['cosine','amplitude_A_to_B','amplitude_B_to_A']]
        if not np.isfinite([c, ab, ba]).all() or not -1.00001 <= c <= 1.00001:
            raise ValueError('Invalid match scores')
        if not np.isclose(ab*ba, c*c, rtol=1e-5, atol=1e-7):
            raise ValueError('Directional amplitudes disagree with cosine')
    rows=[]
    for side,n in [('A',n_a),('B',n_b)]:
        other='B' if side=='A' else 'A'
        lookup={int(p['factor_'+side]):p for p in pairs}
        for k in range(1,n+1):
            p=lookup.get(k)
            shape=bool(p is not None and p['cosine'] >= cosine_min)
            strength=bool(p is not None and all(amplitude_min <= p[v] <= amplitude_max
                for v in ['amplitude_A_to_B','amplitude_B_to_A']))
            status=('unassessed' if not comparison_available else
                'profile_not_reproduced' if not shape else
                'strength_variable' if not strength else 'reproducible')
            rows.append(dict(ensemble=side,factor=k,
                counterpart=int(p['factor_'+other]) if p else None,
                pair_cosine=p['cosine'] if p else np.nan,
                amplitude_to_other=p['amplitude_'+side+'_to_'+other] if p else np.nan,
                amplitude_from_other=p['amplitude_'+other+'_to_'+side] if p else np.nan,
                status=status,presentation_include=status=='reproducible',
                model_retained=True))
    return pd.DataFrame(rows)
