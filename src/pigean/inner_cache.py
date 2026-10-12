"""Bounded reuse of fixed correlation slices during one inner beta fit."""
from collections import OrderedDict

import numpy as np
from scipy import sparse


class InnerCorrelationCache:
    """V is fixed for this object's lifetime; never cache sampled coefficients.

    Keep the legacy slicing expressions and array layouts to preserve reduction
    order. Keys include both masks, since parallel problems can stop separately.
    The cap covers retained array and key payloads, excluding Python overhead.
    Oversize entries are used once and are not retained.
    """

    def __init__(self, V, multiple_V=False, sparse_V=False,
                 max_bytes=16 * 1024 * 1024, max_entries=256):
        self.V = V
        self.multiple_V = multiple_V
        self.sparse_V = sparse_V
        self.max_bytes = max_bytes
        self.max_entries = max_entries
        self.entries = OrderedDict()
        self.payload_bytes = 0
        self.hits = 0
        self.misses = 0

    def get(self, compute_mask_m, compute_mask_v):
        key = (compute_mask_m.shape, compute_mask_m.tobytes(), compute_mask_v.tobytes())
        cached = self.entries.get(key)
        if cached is not None:
            self.entries.move_to_end(key)
            self.hits += 1
            return cached[0]

        self.misses += 1
        union = np.any(compute_mask_m, axis=0)
        union_filter = compute_mask_m[:, union]
        if self.multiple_V:
            selected = self.V[compute_mask_v, :, :][:, :, union]
        elif self.sparse_V:
            selected = self.V[union, :]
        else:
            selected = self.V[:, union]
        value = (union, union_filter, selected)
        size = union.nbytes + union_filter.nbytes + len(key[1]) + len(key[2])
        if sparse.issparse(selected):
            size += selected.data.nbytes + selected.indices.nbytes + selected.indptr.nbytes
        else:
            size += selected.nbytes
        if size <= self.max_bytes and self.max_entries > 0:
            while self.entries and (self.payload_bytes + size > self.max_bytes
                                    or len(self.entries) >= self.max_entries):
                _, (_, old_size) = self.entries.popitem(last=False)
                self.payload_bytes -= old_size
            self.entries[key] = (value, size)
            self.payload_bytes += size
        return value
