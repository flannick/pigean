# Two-thread linear covariance prefilter

Opt in with `--gibbs-engine parallel-prefilter`. This includes compiled covariance
and earlier optimizations. `compiled-covariance` selects the serial parent;
`legacy` remains the default. Optional Numba is required for inherited compilation.

Split linear sparse covariance inflation into two consecutive annotation chunks
at existing block boundaries. A single executor, scoped to the inference engine,
uses at most two threads and is closed on success or error. Workers share the
read-only correlation matrix and own their annotation copies and temporary
arrays. Results are joined in original column order. Sampler updates and random
draws remain on the calling thread. Logistic covariance and single-block inputs
retain the serial path.

This can reduce wall time at the cost of extra CPU and simultaneous working
memory. No process copies or new dependencies are introduced by threading.
The preceding compiler still has startup costs. Benefit depends on chunk sizes,
annotation density, memory bandwidth and scheduling; there is no general
twofold speedup claim. Short-screen evidence is tracked in the analysis project.
