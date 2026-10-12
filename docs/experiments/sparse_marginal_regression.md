# Sparse marginal-regression arithmetic

Use `--gibbs-engine sparse-regression` to select experimental sparse arithmetic within the existing outer Gibbs loop. `--gibbs-engine legacy` remains the default. The experimental scope is Gibbs iteration only; input preparation retains legacy arithmetic.

The estimator, IRLS stopping rule, pseudocounts, covariance correction, and random-number generation are unchanged. IRLS computes the same nonzero-observation sums directly from sparse entries instead of constructing several intermediate sparse matrices. Marginal covariance correction expresses each influence column as a constant plus sparse deviations, evaluating the same quadratic form without constructing dense gene-by-annotation blocks. Near-cancellation cases and correlation operators retain the legacy covariance calculation. Small floating-point differences can still cause later stochastic trajectories to diverge; numerical tests alone do not establish full-run equivalence.

Early T2D/NAFLD profiles identified marginal regression as the principal cost. Covariance optimization alone cannot reach the requested tenfold speedup, so this candidate also replaces repeated IRLS intermediate calculations. Benchmark this as one opt-in arithmetic backend on an independent branch. It does not introduce a new statistical approximation or improve mixing by design.

Validation proceeds from dense-reference numerical tests to one short paired replay per trait. Full fits are conditional on preserved priors and `beta_uncorrected` plus a credible path to tenfold speedup. Short replays provide no convergence or publication qualification. The oversized legacy matrices are deferred; follow the analysis project's staged execution plan.
