# Inner dense matrix product

Enable independently with `--gibbs-enable inner-matmul`, optionally on top of any engine preset. Disable with `--gibbs-disable inner-matmul`; all presets leave it off.

For the shared dense correlation path, reshape the chain-by-parallel-by-coefficient left operand to a matrix, call the two-dimensional BLAS product, then restore the original dimensions. This replaces NumPy's generic three-dimensional `dot` dispatch. Sparse, per-parallel correlation and low-memory design paths retain their established operations. Conditional equations, batches, random draws and stopping are unchanged; BLAS reduction order may change rounding and hence later sampled trajectories.

Six new conditional checks cover Gauss–Seidel, active/inactive parallels and empty updates; 24 existing cache/full-inner cases also pass. No random draws are added. A captured T2D outer-update-25 conditional (four chains, ten parallel fits, 1,503 coefficients) showed 3.54x median conditional speedup over five alternating pairs of 30 calls. Repeated fixed-input conditional arrays differ by less than 1e-12. This is a kernel result, not a full sampler claim. Analysis evidence: `results/autonomous_20261012/matmul_kernel/benchmark.json`.

## Completed validation

See [optimization milestone 2](optimization-milestone-2/README.md) for the completed full T2D timings, NAFLD sensitivity checks, memory measurements and retained configuration. This feature remains independently selectable and off in all historical presets. Kernel gains must not be interpreted as whole-run gains. Long-run NAFLD outputs are sensitive despite fixed-input arithmetic checks; no convergence improvement is claimed.
