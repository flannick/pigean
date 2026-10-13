# Compile existing greedy batch construction

`--gibbs-enable inner-batching` compiles the existing ascending-index greedy construction for dense square float64 correlation matrices with default ordering. Every preset leaves it off. It is independent of the other arithmetic flags.

The selected batches and threshold comparison are preserved, including the NumPy 1.26 contiguous float64 reduction order. Other NumPy versions, custom ordering and alternative covariance paths use the reference implementation. This is not the abandoned blockwise redesign. Requires the optional Numba dependency. The compiler adds cold setup time; scratch storage is linear in coefficient count, alongside the existing masks and correlation matrix.

Partition/reduction boundary tests and full inner-fit/RNG tests validate arithmetic. Captured production partition timing and full outer validation must establish usefulness.
