# Annotation structure cache

`--gibbs-engine cached-grouped-regression` extends `grouped-regression` with run-scoped exact-content caching of annotation value groups. Legacy remains the default; all preceding engines remain selectable. This is an intentional dependent extension of `d7c393a`, independent of fPS and variance-threshold branches.

Each canonical sparse column is keyed by its row count, index/value dtypes, and complete index/value bytes. The cached arrays contain only the distinct values, inverse group assignments, and counts. Subset selection and column reordering can reuse entries; changed values, positions, or row counts cannot silently hit stale entries. Group arrays are assembled with array concatenation instead of expanding nonzero entries into Python lists.

An inference invocation retains at most 64 MiB of key/array payload and 16,384 entries, evicting least recently used entries. Python container overhead is additional and bounded by entry count. Nested backend contexts have isolated caches and restore the enclosing context on normal or exceptional exit. Outside the selected context, there is no persistent cache.

Responses, fitted coefficients, correlation matrices, and covariance weights are never cached across regressions. Each `GroupedDesign` has fresh chain mappings and fresh covariance preparation. No group rounding, approximation, sampler update, tolerance, or random-number change is introduced. Numerical reference and invalidation tests precede bounded paired trait screens; short screens cannot establish convergence or full-run speedup.

Known inherited limitation: grouped and legacy arithmetic can return different SEs for a constant annotation collinear with the intercept, due to cancellation in the singular information matrix. Caching reproduces the grouped result exactly, including this case. Tests require full equality to the grouped parent and legacy parity on identifiable-column SEs and all other fields. This is not a claim of universal legacy SE equivalence; resolve the inherited singular-column behavior before production promotion.
