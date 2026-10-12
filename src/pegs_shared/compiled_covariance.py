"""Optional serial compilation of the compact covariance preparation loop."""
import numpy as np

try:
    from numba import njit
except ImportError as exc:
    raise ImportError(
        'compiled-covariance requires the optional numba dependency; '
        'use cached-inner for the uncompiled implementation'
    ) from exc


@njit(cache=True, fastmath=False)
def _edge_kernel(xptr, xrows, groups, rptr, rcols, rdata, n, k):
    # Inputs are canonical CSC X and CSR R, with one group index per X entry.
    # R's row entries need not be sorted. Rows >= n are the pseudocount.
    positions = np.full(n, -1, dtype=np.int64)
    lefts = [np.empty(0, dtype=np.int64)]
    rights = [np.empty(0, dtype=np.int64)]
    weights = [np.empty(0, dtype=np.float64)]
    absolutes = [np.empty(0, dtype=np.float64)]
    input_edges = aggregated_columns = total = 0
    for j in range(k):
        start, end = xptr[j], xptr[j + 1]
        first = np.iinfo(np.int64).max
        last = -1
        for pos in range(start, end):
            row = xrows[pos]
            if row < n:
                positions[row] = pos
                first = min(first, groups[pos])
                last = max(last, groups[pos])
        count = 0
        for pos in range(start, end):
            row = xrows[pos]
            if row >= n:
                continue
            for edge in range(rptr[row], rptr[row + 1]):
                target = positions[rcols[edge]]
                if target >= 0 and target != pos:
                    count += 1
        if count:
            input_edges += count
            num_groups = last - first + 1
            bins = num_groups * num_groups
            if bins <= 65536 and 2 * bins < count:
                summed = np.zeros(bins, dtype=np.float64)
                magnitudes = np.zeros(bins, dtype=np.float64)
                # Same row/edge order as the NumPy bincount implementation.
                for pos in range(start, end):
                    row = xrows[pos]
                    if row >= n:
                        continue
                    for edge in range(rptr[row], rptr[row + 1]):
                        target = positions[rcols[edge]]
                        if target >= 0 and target != pos:
                            pair = (groups[pos] - first) * num_groups + groups[target] - first
                            summed[pair] += rdata[edge]
                            magnitudes[pair] += abs(rdata[edge])
                size = 0
                for pair in range(bins):
                    if magnitudes[pair] != 0:
                        size += 1
                l = np.empty(size, dtype=np.int64)
                rr = np.empty(size, dtype=np.int64)
                w = np.empty(size, dtype=np.float64)
                aw = np.empty(size, dtype=np.float64)
                i = 0
                for pair in range(bins):
                    if magnitudes[pair] != 0:
                        l[i], rr[i] = pair // num_groups + first, pair % num_groups + first
                        w[i], aw[i] = summed[pair], magnitudes[pair]
                        i += 1
                aggregated_columns += 1
            else:
                l = np.empty(count, dtype=np.int64)
                rr = np.empty(count, dtype=np.int64)
                w = np.empty(count, dtype=np.float64)
                aw = np.empty(count, dtype=np.float64)
                i = 0
                for pos in range(start, end):
                    row = xrows[pos]
                    if row >= n:
                        continue
                    for edge in range(rptr[row], rptr[row + 1]):
                        target = positions[rcols[edge]]
                        if target >= 0 and target != pos:
                            l[i], rr[i] = groups[pos], groups[target]
                            w[i], aw[i] = rdata[edge], abs(rdata[edge])
                            i += 1
            lefts.append(l)
            rights.append(rr)
            weights.append(w)
            absolutes.append(aw)
            total += len(l)
        for pos in range(start, end):
            row = xrows[pos]
            if row < n:
                positions[row] = -1

    left = np.empty(total, dtype=np.int64)
    right = np.empty(total, dtype=np.int64)
    weight = np.empty(total, dtype=np.float64)
    absolute = np.empty(total, dtype=np.float64)
    offset = 0
    for i in range(len(lefts)):
        size = len(lefts[i])
        left[offset:offset + size] = lefts[i]
        right[offset:offset + size] = rights[i]
        weight[offset:offset + size] = weights[i]
        absolute[offset:offset + size] = absolutes[i]
        offset += size
    return left, right, weight, absolute, input_edges, aggregated_columns


def prepare_compiled_covariance():
    """Warm the usual int32 sparse-index signature; consumes no random draws.

    Int64/mixed-index inputs remain supported and compile on their first use.
    Benchmark callers should check signatures before/after inference when
    distinguishing preparation from warmed timing.
    """
    ptr = np.zeros(2, dtype=np.int32)
    indices = np.empty(0, dtype=np.int32)
    _edge_kernel(ptr, indices, np.empty(0, dtype=np.int64), ptr, indices,
                 np.empty(0, dtype=np.float64), 1, 1)


def compiled_edges(X, groups, r, pseudo=False):
    # Retain the Python implementation for unsupported scalar representations;
    # normal grouped regression supplies canonical float64 matrices.
    if (groups.dtype != np.int64 or r.data.dtype != np.float64
            or X.shape[0] != r.shape[0] + int(pseudo)):
        from .covariance_edges import compact_edges
        return compact_edges(X, groups, r, pseudo, compiled=False)
    left, right, weights, absolute, count, aggregated = _edge_kernel(
        X.indptr, X.indices, groups, r.indptr, r.indices, r.data, r.shape[0], X.shape[1])
    result = left, right, weights, absolute
    return result, dict(input_edges=count, stored_pairs=len(left),
                        aggregated_columns=aggregated,
                        edge_payload_bytes=sum(a.nbytes for a in result),
                        lookup_scratch_bytes=r.shape[0] * np.dtype(np.int64).itemsize)
