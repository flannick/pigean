# Fused inner conditional arithmetic

`--gibbs-enable inner-fused` independently compiles the elementwise inclusion-probability, conditional-mean and coefficient-draw arithmetic after residual construction. Combine with `--gibbs-enable inner-matmul` or use alone. Every preset keeps this feature off. `--gibbs-disable inner-fused` restores the reference arithmetic. Optional Numba is required only when selected.

The kernel uses float64, no fast-math, the same supplied uniform/normal draws, and the same update equations. It accepts C/F/strided real float64 arrays; other types retain reference safeguards. Scalar exponential/reduction rounding may differ and can change long seeded trajectories. There is no new approximation, stopping change or convergence claim.

72 focused tests passed, including independent/combined switches, dense/sparse/per-parallel/X/independent inner fits, hyperparameter adaptation and subsequent RNG values. A real T2D outer-25 conditional showed 1.85x incremental kernel speedup beyond inner-matmul across five alternating 30-call pairs; compilation was measured separately. The first development guard excluded an F-contiguous SE array and therefore never dispatched; that diagnostic timing is retained but is not a fused-kernel measurement. The corrected benchmark explicitly verifies compiled signatures. Full sampler validation is the next gate.
