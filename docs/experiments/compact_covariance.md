# Compact covariance preparation

`--gibbs-engine compact-covariance` extends `cached-grouped-regression` at `3a4f75d`. Legacy remains the default and all preceding engines remain selectable. This branch is independent of the fPS and variance-threshold branches.

Each correlation preparation allocates one temporary gene-to-row-position lookup and reuses it across annotation columns. This replaces binary searches through an annotation's row indices for each covariance edge. The lookup is reset between columns; covariance weights are rebuilt each regression, including when a matrix is mutated in place between regressions.

For repeated annotation values, edges with the same ordered pair of value groups have the same coefficient-dependent influence product. Sum their correlation weights once. Sum absolute weights separately to preserve the numerical cancellation scale; taking the absolute value of a signed sum would incorrectly hide cancellation. This is algebraic aggregation, without thresholding edges or rounding annotation values. Floating-point summation order may differ.

Dense pair aggregation is bounded at 65,536 bins and used only when the bin count is less than half the original edge count. Continuous or sparse-edge columns retain explicit edges. Compared with the preceding engine, the direct lookup adds eight bytes per gene temporarily; each stored edge/pair uses 32 payload bytes instead of 24 because its absolute-weight sum is retained. Compression can more than offset this, but net memory reduction is not guaranteed. Matrix validation, independent pseudocount treatment, correlation-operator fallback, and cancellation fallback remain in place. Diagnostics accumulate counters/maxima only, without retaining per-iteration histories.

No changes to either Gibbs loop, random draws, convergence rules, or inference target. The inherited constant-column SE discrepancy between grouped and legacy arithmetic remains a separate promotion limitation. Validate numerical behavior and paired short inference time/peak RSS before claiming a benefit; short screens do not establish convergence or full-run scaling.
