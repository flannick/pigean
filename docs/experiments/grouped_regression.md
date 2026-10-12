# Grouped marginal-regression statistics

`--gibbs-engine grouped-regression` adds reuse of exact repeated-value statistics to the sparse arithmetic experiment. `legacy` remains the default; `sparse-regression` still selects the first sparse implementation. This branch is an intentional dependent extension of sparse arithmetic at `ff4e4ce`, independent of fPS and variance-threshold work.

Each annotation column is partitioned by exactly equal nonzero values. No rounding or binning is introduced. IRLS evaluates the same predicted probabilities and Hessian/gradient sums once per distinct value, weighted by its observation count. Observed-response sums and the existing stopping rule are unchanged.

The one-off linear prefilter retains sparse calculation, with up to 512 columns per sparse operation to reduce repeated dispatch; grouped preparation is worthwhile for the repeated logistic chain fits. Covariance preparation reads compressed sparse entries directly rather than constructing one submatrix per annotation. This changes arithmetic batching only, not Gibbs update groups or sampling order.

For covariance correction, each regression call prepares within-annotation covariance summaries for the value groups. Those summaries are reused across chains sharing the same correlation matrix; group influences are recalculated for every fitted coefficient/intercept. The independent pseudocount is excluded from the correlation matrix but retained in fitting, centering, and independent variance. The cache expires with the regression call, so it cannot leak across Gibbs updates. Correlation operators and numerically unstable quadratic forms use the established fallback.

This is a computational change to the existing estimator and transition algorithm, not a new posterior approximation. Floating-point summation order can differ. Validation requires numerical reference tests, CLI compatibility, then bounded matched trait screens for time, priors, and `beta_uncorrected`. A short trace cannot establish convergence or full-run speedup. Do not launch the deferred full matrices.
