# Bounded static annotation-correlation cache

`--gibbs-enable static-correlation` caches the full fixed annotation correlation only for unwhitened float64 CSC inputs whose square matrix fits 256 MiB. Each request hashes the actual matrix structure/data and normalization vectors, so in-place changes invalidate the cache. Whitened, unsupported or oversized inputs use the reference subset calculation. The cache expires with the scoped Gibbs backend and is independent of all other features; every preset leaves it off.

Construction uses 256-row blocks to limit temporary dense arrays. Selected matrices are copies, while the retained full matrix is read-only. This trades quadratic retained memory and cold setup for avoiding repeated subset correlations. Tests verify exact subset matrices, changed coefficients/normalization vectors, cap/whitening fallback and nested-context cleanup. Production-size benchmarking and full memory/runtime checks determine whether the tradeoff is worthwhile.

Real T2D-design benchmark (seeded subsets of 500, 1503 and 2000 annotations): five alternating pairs gave 0.13117 / 0.05083 s median for the three subset requests (2.581x), with bitwise identical matrices. Cold full-cache construction took 0.477 s and retained 193,021,952 bytes (184.1 MiB). Content hashing and subset copying are included in warm timing. This is a modest potential full-run gain with a substantial memory cost; it is not yet part of the recommended stack.

## Completed validation

See [optimization milestone 2](optimization-milestone-2/README.md) for the completed full T2D timings, NAFLD sensitivity checks, memory measurements and retained configuration. This feature remains independently selectable and off in all historical presets. Kernel gains must not be interpreted as whole-run gains. Long-run NAFLD outputs are sensitive despite fixed-input arithmetic checks; no convergence improvement is claimed.
