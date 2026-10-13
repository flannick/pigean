"""Bounded, exact-content cache for annotation column value groups only."""
from collections import OrderedDict

import numpy as np


class ColumnGroupingCache:
    """One inference invocation owns this cache; never store fitted/covariance state.

    Full byte keys avoid identity reuse, hash collisions, and stale hits after
    in-place mutation. The limits bound retained payload bytes and entry count
    (Python container overhead is additional, bounded by the entry count).
    """

    def __init__(self, max_bytes=64 * 1024 * 1024, max_entries=16384):
        self.max_bytes = max_bytes
        self.max_entries = max_entries
        self.entries = OrderedDict()
        self.payload_bytes = 0
        self.hits = 0
        self.misses = 0

    def column_groups(self, n, rows, data):
        key = (n, rows.dtype.str, data.dtype.str, rows.tobytes(), data.tobytes())
        if key in self.entries:
            self.hits += 1
            self.entries.move_to_end(key)
            return self.entries[key][0]
        self.misses += 1
        result = np.unique(data, return_inverse=True, return_counts=True)
        for array in result:
            array.flags.writeable = False
        size = len(key[-2]) + len(key[-1]) + sum(array.nbytes for array in result)
        if self.max_entries > 0 and size <= self.max_bytes:
            while self.entries and (self.payload_bytes + size > self.max_bytes
                                    or len(self.entries) >= self.max_entries):
                _, (_, removed_bytes) = self.entries.popitem(last=False)
                self.payload_bytes -= removed_bytes
            self.entries[key] = (result, size)
            self.payload_bytes += size
        return result
