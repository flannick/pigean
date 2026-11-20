"""Greedy hierarchies over fitted leaf profiles and OR membership summaries."""
import numpy as np
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import cdist, pdist, squareform


RULE_FIELDS = {'cosine': 'min_leaf_cosine', 'shared-mass': 'min_shared_mass',
               'shared-squared-mass': 'min_shared_squared_mass',
               'both-given-either': 'min_both_given_either',
               'prob-containment': 'min_prob_containment',
               'profile-containment': 'min_profile_containment'}


def profile_distances(profiles, rule, other=None):
    """Rows are nonzero profiles. Return condensed distances or cross distances."""
    def distance(x, y, metric):
        return pdist(x, metric) if y is None else cdist(x, y, metric)

    if rule == 'cosine':
        return np.clip(distance(profiles, other, 'cosine'), 0, 1)
    if rule in ('shared-mass', 'shared-squared-mass'):
        def unit_mass(x):
            if rule == 'shared-squared-mass':
                x = x**2
            return x / x.sum(axis=1, keepdims=True)
        return np.clip(distance(unit_mass(profiles), None if other is None else unit_mass(other),
                                'cityblock') / 2, 0, 1)

    y = profiles if other is None else other
    a = profiles.sum(axis=1)[:, None]
    b = y.sum(axis=1)[None, :]
    if rule == 'profile-containment':
        shared = np.array([np.minimum(row, y).sum(axis=1) for row in profiles])
    else:
        shared = profiles @ y.T
    denominator = a+b-shared if rule == 'both-given-either' else np.minimum(a,b)
    result = np.clip(1-shared/denominator, 0, 1)
    return result[np.triu_indices(len(profiles), 1)] if other is None else result


def distances(weights):
    """Condensed distances for the three original leaf-profile comparisons."""
    return {rule: profile_distances(weights.T, rule)
            for rule in ('cosine', 'shared-mass', 'shared-squared-mass')}


def build_tree(weights, leaf_ids, rule='cosine', scope='leaves', *, full_diagnostics=True):
    """Columns follow leaf_ids. Node indices are local; leaf IDs survive permutations."""
    if rule not in RULE_FIELDS:
        raise ValueError(f'Unknown linkage rule {rule}')
    if scope not in ('leaves', 'nodes'):
        raise ValueError(f'Unknown merge scope {scope}')
    w = np.asarray(weights, dtype=float)
    ids = np.asarray(leaf_ids, dtype=int)
    if (w.ndim != 2 or ids.shape != (w.shape[1],) or len(set(ids)) != len(ids)
            or not np.isfinite(w).all() or (w < 0).any() or (w > 1).any()
            or (w.sum(axis=0) <= 0).any()):
        raise ValueError('Expected unique leaf IDs and nonzero, finite gene profiles in [0, 1]')
    order = np.argsort(ids)
    w, ids = w[:, order], ids[order]
    n = len(ids)
    if not n:
        return [], w, []
    matrices = {key: squareform(profile_distances(w.T, key)) for key in (RULE_FIELDS if full_diagnostics else [rule])}
    if scope == 'leaves':
        joins = linkage(squareform(matrices[rule]), method='complete') if n > 1 else []
    else:
        # Upper triangle caches distances between active OR nodes. Node ID breaks ties.
        live = np.full((2*n-1, 2*n-1), np.inf)
        ia, ib = np.triu_indices(n, 1)
        live[ia, ib] = matrices[rule][ia, ib]
        active = set(range(n))
    prefix_floor = 1.
    nodes = []
    members = []
    loadings = np.empty((w.shape[0], 2*n-1))
    for i in range(2*n-1):
        left = right = None
        scores = {RULE_FIELDS[key]: 1. for key in matrices}
        merge_score = 1.
        conditional = dict.fromkeys(('right_given_left', 'left_given_right', 'left_background', 'right_background'), np.nan)
        if i < n:
            indices = [i]
            loadings[:, i] = w[:, i]
        else:
            if scope == 'leaves':
                a, b, distance, _ = joins[i-n]
                left, right = int(a), int(b)
            else:
                left, right = np.unravel_index(np.argmin(live), live.shape)
                left, right = int(left), int(right)
                distance = live[left, right]
                active.difference_update((left, right))
                live[[left, right], :] = np.inf
                live[:, [left, right]] = np.inf
            merge_score = 1 - float(distance)
            prefix_floor = min(prefix_floor, merge_score)
            indices = members[left] + members[right]
            cross = np.ix_(members[left], members[right])
            for other, matrix in matrices.items():
                field = RULE_FIELDS[other]
                scores[field] = min(nodes[left][field], nodes[right][field], 1 - matrix[cross].max())
            x, y = loadings[:, left], loadings[:, right]
            both = x @ y
            conditional = dict(right_given_left=both/x.sum(), left_given_right=both/y.sum(),
                               left_background=x.mean(), right_background=y.mean())
            loadings[:, i] = x + (1 - x)*y
            if scope == 'nodes':
                remaining = sorted(active)
                if remaining:
                    live[remaining, i] = profile_distances(loadings[:, remaining].T, rule,
                                                           loadings[:, i:i+1].T)[:, 0]
                active.add(i)
            nodes[left]['parent'] = nodes[right]['parent'] = i
        energy = np.square(loadings[:, i])
        nodes.append(dict(node=i, parent=None, left=left, right=right,
                          leaf_id=int(ids[i]) if i < n else None, n_leaves=len(indices),
                          **scores, **conditional, merge_score=merge_score,
                          merge_prefix_floor=prefix_floor,
                          gene_equivalents=float(loadings[:, i].sum()),
                          effective_genes=float(energy.sum()**2 / (energy@energy))))
        members.append(indices)

    # Stream subtree pair predictions. Persist only their off-diagonal mass ratio.
    def pair_summary(i):
        node = nodes[i]
        v = loadings[:, i]
        if node['left'] is None:
            pair = np.outer(v, v)
            np.fill_diagonal(pair, 0)
        else:
            a, b = pair_summary(node['left']), pair_summary(node['right'])
            pair = a + (1 - a)*b
        denom = np.outer(v, v)
        np.fill_diagonal(denom, 0)
        mass = denom.sum()
        node['shared_leaf_pair_fraction'] = float(pair.sum()/mass) if mass else np.nan
        return pair

    if full_diagnostics:
        pair_summary(len(nodes)-1)
    return nodes, loadings, [ids[m].tolist() for m in members]


def cut_tree(nodes, floor, metric='min_leaf_cosine'):
    """Return maximal qualifying nodes. Indivisible leaves always remain visible."""
    threshold = floor
    if metric == 'gene_equivalents':
        if not np.isfinite(threshold) or threshold <= 0:
            raise ValueError('Gene-equivalent ceiling must be finite and positive')
    elif (metric not in RULE_FIELDS.values() and metric != 'merge_prefix_floor') or not 0 <= threshold <= 1:
        raise ValueError('Expected a supported metric and a floor in [0, 1]')
    frontier = []
    stack = [len(nodes)-1] if nodes else []
    while stack:
        i = stack.pop()
        node = nodes[i]
        qualifies = node[metric] <= threshold if metric == 'gene_equivalents' else node[metric] >= threshold
        if node['left'] is None or qualifies:
            frontier.append(i)
        else:
            stack.extend((node['right'], node['left']))
    return frontier
