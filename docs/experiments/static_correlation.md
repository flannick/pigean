# Bounded static annotation-correlation cache

`--gibbs-enable static-correlation` caches the full fixed annotation correlation only for unwhitened float64 CSC inputs whose square matrix fits 256 MiB. Each request hashes the actual matrix structure/data and normalization vectors, so in-place changes invalidate the cache. Whitened, unsupported or oversized inputs use the reference subset calculation. The cache expires with the scoped Gibbs backend and is independent of all other features; every preset leaves it off.

Construction uses 256-row blocks to limit temporary dense arrays. Selected matrices are copies, while the retained full matrix is read-only. This trades quadratic retained memory and cold setup for avoiding repeated subset correlations. Tests verify exact subset matrices, changed coefficients/normalization vectors, cap/whitening fallback and nested-context cleanup. Production-size benchmarking and full memory/runtime checks determine whether the tradeoff is worthwhile.
