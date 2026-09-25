# PIGEAN Known Limitations (v1)

- Gibbs inference remains stochastic unless `--deterministic`/`--seed` is set.
- Numerical outputs may differ slightly across platforms due to BLAS/SciPy implementation differences.
- Large-chain runs can still exceed strict RSS targets because Python process overhead and sparse/dense conversion peaks are outside pure matrix batch limits.
- `--correct-huge` relies on covariate-pruning edge cases that are tested, but large custom covariate matrices can still be expensive.
- Advanced Set-B workflows are supported but intentionally less opinionated than the core flow; users should prefer core presets unless needed.

## Historical vectorized logistic gene-set statistics

The September 2026 fix to the shared logistic fitter aligns trait outcome totals with trait-major coefficient order during intercept updates. Older multi-trait logistic batches could therefore produce marginal coefficients, SEs and P values that depended on batch composition. Bayesian estimates derived from those marginal statistics can also be affected and should be regenerated. Single-trait calculations and linear regression do not use the affected ordering. The `--ols` switch controls correlation correction; it does not itself select the linear model. Batch, single-trait and trait/factor permutation equivalence are regression-tested.

## Combined gene-response dependence

The covariance audit corrected marginal linear/logistic and joint linear uncertainty calculations and replaced the annotation-correlation representation. Explicit trait correlations can now be supplied to vectorized factor-GMT inference. The annotation estimator is still a working model and is not automatically calibrated for binary Combined gene hits. Historical independent-gene P values should not be treated as dependence-calibrated. See [gene covariance inputs, equations, and validation](GENE_COVARIANCE.md). Correlated Huber inference now fails explicitly pending a validated robust covariance implementation.
