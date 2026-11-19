"""Complete-linkage families of original leaves and OR membership summaries."""
import numpy as np
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import pdist, squareform


def distances(weights):
    """One minus cosine, and one minus common mass after unit-sum normalization."""
    profiles = weights.T
    cosine = np.clip(pdist(profiles, 'cosine'), 0, 1)
    mass = profiles / profiles.sum(axis=1, keepdims=True)
    unmatched = np.clip(pdist(mass, 'cityblock') / 2, 0, 1)
    return cosine, squareform(unmatched)


def build_tree(weights, leaf_ids):
    """Columns follow leaf_ids. Node indices are local; leaf IDs survive permutations."""
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
    cosine, unmatched = distances(w)
    joins = linkage(cosine, method='complete') if n > 1 else []
    nodes = []
    members = []
    loadings = np.empty((w.shape[0], 2*n-1))
    for i in range(2*n-1):
        left = right = None
        similarity = overlap = 1.
        if i < n:
            indices = [i]
            loadings[:, i] = w[:, i]
        else:
            a, b, distance, _ = joins[i-n]
            left, right = int(a), int(b)
            indices = members[left] + members[right]
            similarity = 1 - float(distance)
            overlap = min(nodes[left]['min_shared_mass'], nodes[right]['min_shared_mass'],
                          1 - unmatched[np.ix_(members[left], members[right])].max())
            x, y = loadings[:, left], loadings[:, right]
            loadings[:, i] = x + (1 - x)*y
            nodes[left]['parent'] = nodes[right]['parent'] = i
        energy = np.square(loadings[:, i])
        nodes.append(dict(node=i, parent=None, left=left, right=right,
                          leaf_id=int(ids[i]) if i < n else None, n_leaves=len(indices),
                          min_leaf_cosine=similarity, min_shared_mass=float(overlap),
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
    """Return the coarsest non-overlapping frontier satisfying a minimum pair score."""
    if metric not in ('min_leaf_cosine', 'min_shared_mass') or not 0 <= floor <= 1:
        raise ValueError('Expected a supported metric and a floor in [0, 1]')
    frontier = []
    stack = [len(nodes)-1] if nodes else []
    while stack:
        i = stack.pop()
        node = nodes[i]
        if node['left'] is None or node[metric] >= floor:
            frontier.append(i)
        else:
            stack.extend((node['right'], node['left']))
    return frontier
