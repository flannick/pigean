"""Exact sparse covariance edges, optionally reduced to value-group pairs."""
import numpy as np


def compact_edges(X, groups, r, pseudo=False, *, compiled=None):
    """Prepare edges for this correlation only; no cross-iteration weights.

    A reusable row-position array replaces binary searches for edge targets.
    Dense pair reductions are used only when bounded and smaller than the edge
    list. Absolute weights are reduced separately to retain the original
    cancellation scale even when positive and negative edges cancel.
    """
    if compiled is None:
        from .sparse_regression import use_compiled_covariance
        compiled = use_compiled_covariance()
    if compiled:
        from .compiled_covariance import compiled_edges
        return compiled_edges(X, groups, r, pseudo)
    positions_by_row = np.full(r.shape[0], -1, dtype=np.int64)
    left, right, weights, absolute = [], [], [], []
    input_edges = aggregated_columns = 0
    for j in range(X.shape[1]):
        start, end = X.indptr[j:j+2]
        rows, group = X.indices[start:end], groups[start:end]
        if pseudo:
            real = rows < r.shape[0]
            rows, group = rows[real], group[real]
        if not len(rows):
            continue
        positions_by_row[rows] = np.arange(len(rows))
        lengths = r.indptr[rows+1] - r.indptr[rows]
        local_rows = np.repeat(np.arange(len(rows)), lengths)
        starts = r.indptr[rows] - np.r_[0, np.cumsum(lengths[:-1])]
        entries = np.repeat(starts, lengths) + np.arange(len(local_rows))
        positions = positions_by_row[r.indices[entries]]
        kept = np.flatnonzero((positions >= 0) & (positions != local_rows))
        positions_by_row[rows] = -1
        if not len(kept):
            continue
        l, rr, w = group[local_rows[kept]], group[positions[kept]], r.data[entries[kept]]
        input_edges += len(w)
        first = int(group.min())
        num_groups = int(group.max()) - first + 1
        bins = num_groups * num_groups
        if bins <= 65536 and 2*bins < len(w):
            pair = (l-first)*num_groups + (rr-first)
            summed = np.bincount(pair, weights=w, minlength=bins)
            magnitudes = np.bincount(pair, weights=np.abs(w), minlength=bins)
            active = np.flatnonzero(magnitudes)
            l, rr = active//num_groups + first, active%num_groups + first
            w, aw = summed[active], magnitudes[active]
            aggregated_columns += 1
        else:
            aw = np.abs(w)
        left.append(l)
        right.append(rr)
        weights.append(w)
        absolute.append(aw)
    def join(parts, dtype):
        return np.concatenate(parts) if parts else np.array([], dtype=dtype)
    result = (join(left, np.int64), join(right, np.int64),
              join(weights, float), join(absolute, float))
    stats = dict(input_edges=input_edges, stored_pairs=len(result[0]),
                 aggregated_columns=aggregated_columns,
                 edge_payload_bytes=sum(a.nbytes for a in result),
                 lookup_scratch_bytes=positions_by_row.nbytes)
    return result, stats
