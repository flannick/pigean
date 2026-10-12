# Cached inner Gibbs correlation slices

Opt in with `--gibbs-engine cached-inner`. This includes the five earlier
regression optimizations. `legacy` remains the default; `lean-regression`
selects the immediately preceding engine.

Within each inner beta fit, retain correlation slices for repeated batch and
active-parallel masks. Use the original slicing expressions, layouts and matrix
products. Coefficients, probabilities, hyperparameters and residuals are recomputed
as before; random draws, update order and stopping rules are unchanged.

The cache belongs to one inner fit and cannot survive a change in the outer
regression's correlation matrix. Both masks participate in the key. Dense,
per-parallel dense and sparse correlation matrices are supported; independent
fits and the low-memory X path retain their previous behavior. No dense
correlation matrix is constructed for the X path.

Retained array/key payload is capped at 16 MiB and 256 entries per fit; Python
overhead and transient misses are additional. Oversize entries are not retained;
least-recently-used entries are evicted. More sweeps can amortize preparation;
more annotations can exceed the cap and eliminate the benefit. No approximation
or convergence improvement is intended. Trait timing and memory evidence is
tracked in the analysis project's optimization table; this is a local experiment.
