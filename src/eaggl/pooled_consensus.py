"""Greedy complete-link consensus across distinct factorization restarts.

Clusters can contain at most one column per restart; every pair must satisfy the
cosine threshold. Content hashes provide deterministic tie-breaking independent
of restart/column order. No reference restart or final factor-count cap is used.
"""
from __future__ import annotations

import hashlib
import numpy as np


def collect_pooled_matches(matrices, run_ids, *, min_cosine=0.7, min_runs=3):
    if len(matrices) != len(run_ids) or len(set(map(str, run_ids))) != len(run_ids):
        raise ValueError("Pooled consensus requires one unique ID per restart")
    if not 0 <= min_cosine <= 1 or min_runs < 1:
        raise ValueError("Invalid pooled consensus support or cosine threshold")
    records = []
    row_count = None
    for run_index, (matrix, run_id) in enumerate(zip(matrices, run_ids)):
        if matrix is None:
            continue
        matrix = np.asarray(matrix, dtype=np.float64)
        if matrix.ndim != 2 or not np.isfinite(matrix).all() or (matrix < 0).any():
            raise ValueError("Consensus factors must be finite nonnegative matrices")
        if row_count is not None and matrix.shape[0] != row_count:
            raise ValueError("Consensus factor rows must share the same feature axis")
        row_count = matrix.shape[0]
        for column in range(matrix.shape[1]):
            values = np.ascontiguousarray(matrix[:, column])
            if np.linalg.norm(values) == 0:
                continue
            digest = hashlib.sha256(values.tobytes()).hexdigest()
            records.append((digest, str(run_id), column, values, run_index))
    records.sort(key=lambda r: (r[0], r[1], r[2]))
    if not records:
        return {}, {"input_factors": 0, "retained_clusters": 0, "unsupported_factors": 0}

    basis = np.column_stack([r[3] for r in records])
    norm = np.linalg.norm(basis, axis=0)
    similarities = (basis / norm).T @ (basis / norm)
    count = len(records)
    clusters = {i: [i] for i in range(count)}
    runs = {i: {records[i][1]} for i in range(count)}
    scores = similarities.copy()

    def signature(nodes):
        return tuple(sorted((records[i][0], records[i][1]) for i in nodes))

    for i in range(count):
        for j in range(count):
            if j <= i or records[i][1] == records[j][1]:
                scores[i, j] = -np.inf
    while True:
        best = float(scores.max())
        if best < min_cosine:
            break
        tied = np.argwhere(scores == best)
        a, b = min(
            (tuple(map(int, pair)) for pair in tied),
            key=lambda pair: tuple(sorted((signature(clusters[pair[0]]), signature(clusters[pair[1]])))),
        )
        clusters[a] += clusters[b]
        runs[a] |= runs[b]
        del clusters[b], runs[b]
        scores[b, :] = scores[:, b] = -np.inf
        scores[a, :] = scores[:, a] = -np.inf
        for other in clusters:
            if other != a and not (runs[a] & runs[other]):
                scores[min(a, other), max(a, other)] = float(
                    similarities[np.ix_(clusters[a], clusters[other])].min())

    kept = sorted((nodes for nodes in clusters.values() if len(nodes) >= min_runs), key=signature)
    matches = {}
    for column, nodes in enumerate(kept):
        minimum = float(similarities[np.ix_(nodes, nodes)].min())
        matches[column] = [(records[i][4], records[i][2], minimum) for i in nodes]
    return matches, {
        "input_factors": count,
        "all_clusters": len(clusters),
        "retained_clusters": len(kept),
        "unsupported_clusters": sum(len(nodes) < min_runs for nodes in clusters.values()),
        "unsupported_factors": sum(len(nodes) for nodes in clusters.values() if len(nodes) < min_runs),
    }
