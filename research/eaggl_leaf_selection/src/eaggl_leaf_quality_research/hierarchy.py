"""Complete-linkage families of original leaves and OR membership summaries."""
import numpy as np
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import pdist, squareform


RULE_FIELDS = {'cosine': 'min_leaf_cosine', 'shared-mass': 'min_shared_mass',
               'shared-squared-mass': 'min_shared_squared_mass'}


def distances(weights):
    """Condensed pairwise dissimilarities in the original leaf order."""
    profiles = weights.T
    result = {'cosine': np.clip(pdist(profiles, 'cosine'), 0, 1)}
    for rule, mass in [('shared-mass', profiles), ('shared-squared-mass', profiles**2)]:
        mass = mass / mass.sum(axis=1, keepdims=True)
        result[rule] = np.clip(pdist(mass, 'cityblock') / 2, 0, 1)
    return result


def build_tree(weights, leaf_ids, rule='cosine'):
    """Columns follow leaf_ids. Node indices are local; leaf IDs survive permutations."""
    if rule not in RULE_FIELDS:
        raise ValueError(f'Unknown linkage rule {rule}')
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
    distances_by_rule = distances(w)
    matrices = {key: squareform(d) for key, d in distances_by_rule.items() if key != rule}
    joins = linkage(distances_by_rule[rule], method='complete') if n > 1 else []
    nodes = []
    members = []
    loadings = np.empty((w.shape[0], 2*n-1))
    for i in range(2*n-1):
        left = right = None
        scores = dict.fromkeys(RULE_FIELDS.values(), 1.)
        if i < n:
            indices = [i]
            loadings[:, i] = w[:, i]
        else:
            a, b, distance, _ = joins[i-n]
            left, right = int(a), int(b)
            indices = members[left] + members[right]
            scores[RULE_FIELDS[rule]] = 1 - float(distance)
            cross = np.ix_(members[left], members[right])
            for other, matrix in matrices.items():
                field = RULE_FIELDS[other]
                scores[field] = min(nodes[left][field], nodes[right][field], 1 - matrix[cross].max())
            x, y = loadings[:, left], loadings[:, right]
            loadings[:, i] = x + (1 - x)*y
            nodes[left]['parent'] = nodes[right]['parent'] = i
        energy = np.square(loadings[:, i])
        nodes.append(dict(node=i, parent=None, left=left, right=right,
                          leaf_id=int(ids[i]) if i < n else None, n_leaves=len(indices),
                          **scores,
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

    pair_summary(len(nodes)-1)
    return nodes, loadings, [ids[m].tolist() for m in members]


def cut_tree(nodes, floor, metric='min_leaf_cosine'):
    """Return maximal qualifying nodes. Indivisible leaves always remain visible."""
    threshold = floor
    if metric == 'gene_equivalents':
        if not np.isfinite(threshold) or threshold <= 0:
            raise ValueError('Gene-equivalent ceiling must be finite and positive')
    elif metric not in RULE_FIELDS.values() or not 0 <= threshold <= 1:
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
