# PIGEAN Known Limitations (v1)

- Gibbs inference remains stochastic unless `--deterministic`/`--seed` is set.
- Numerical outputs may differ slightly across platforms due to BLAS/SciPy implementation differences.
- Large-chain runs can still exceed strict RSS targets because Python process overhead and sparse/dense conversion peaks are outside pure matrix batch limits.
- `--correct-huge` relies on covariate-pruning edge cases that are tested, but large custom covariate matrices can still be expensive.
- Advanced Set-B workflows are supported but intentionally less opinionated than the core flow; users should prefer core presets unless needed.
