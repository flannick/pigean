"""Bounded, content-validated cache of fixed annotation correlations."""
import hashlib
import numpy as np
from scipy import sparse


class StaticCorrelationCache:
    def __init__(self,max_bytes=256*1024*1024,block_columns=256):
        self.max_bytes=max_bytes
        self.block_columns=block_columns
        self.key=None
        self.matrix=None
        self.hits=self.builds=0

    def get(self,state,mask):
        X=state.X_orig
        k=X.shape[1]
        if (state.y_corr_cholesky is not None or not sparse.isspmatrix_csc(X)
                or X.dtype != np.float64 or k*k*8 > self.max_bytes):
            self.key=self.matrix=None
            return None
        # Content, shape and dtype keys detect in-place changes as well as
        # replacement. Never key a mutable scientific input by identity alone.
        digest=hashlib.sha256(str(X.shape).encode())
        for values in (X.indptr,X.indices,X.data,state.mean_shifts,state.scale_factors):
            a=np.ascontiguousarray(values)
            digest.update(str((a.shape,a.dtype.str)).encode())
            digest.update(memoryview(a).cast('B'))
        key=digest.digest()
        if key != self.key:
            self.key=self.matrix=None
            matrix=np.empty((k,k),dtype=np.float64)
            # Bound dense construction temporaries rather than materializing
            # several full-size normalization arrays alongside the cache.
            for start in range(0,k,self.block_columns):
                stop=min(k,start+self.block_columns)
                matrix[start:stop,:]=state._compute_V(X,state.mean_shifts,state.scale_factors,
                                                     rows=np.arange(start,stop))
            matrix.setflags(write=False)
            self.matrix=matrix
            self.key=key
            self.builds+=1
        else:
            self.hits+=1
        indices=np.flatnonzero(mask)
        return self.matrix[np.ix_(indices,indices)]
