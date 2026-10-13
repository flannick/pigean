"""Compile the existing greedy batch construction without changing its partition."""
import numpy as np
from numba import njit


@njit(cache=True, fastmath=False)
def pairwise_total(values):
    # NumPy 1.26 float64 contiguous-reduction order: short sequential tails,
    # eight interleaved lanes through 128 values, recursively split larger spans.
    size = len(values)
    if size < 8:
        total = -0.0
        for i in range(size):
            total += values[i]
        return total
    if size <= 128:
        lanes = values[:8].copy()
        end = size - size % 8
        for i in range(8, end, 8):
            for lane in range(8):
                lanes[lane] += values[i+lane]
        total = ((lanes[0]+lanes[1])+(lanes[2]+lanes[3])) + ((lanes[4]+lanes[5])+(lanes[6]+lanes[7]))
        for i in range(end, size):
            total += values[i]
        return total
    split = size//2
    split -= split%8
    return pairwise_total(values[:split])+pairwise_total(values[split:])


@njit(cache=True, fastmath=False)
def greedy_labels(V, threshold, use_sum):
    n = V.shape[0]
    labels = np.full(n, -1, dtype=np.int64)
    selected = np.empty(n, dtype=np.int64)
    values = np.empty(n, dtype=np.float64)
    assigned = 0
    batch = 0
    while assigned < n:
        count = 0
        for candidate in range(n):
            if labels[candidate] >= 0:
                continue
            accept = count == 0
            if count:
                for j in range(count):
                    values[j] = V[candidate, selected[j]]
                if use_sum:
                    total = pairwise_total(values[:count])
                else:
                    total = np.max(values[:count])
                accept = total < threshold
            if accept:
                labels[candidate] = batch
                selected[count] = candidate
                count += 1
                assigned += 1
        batch += 1
    return labels, batch


def compiled_batches(V, threshold, use_sum):
    labels, count = greedy_labels(V, threshold, use_sum)
    return [labels == i for i in range(count)]
