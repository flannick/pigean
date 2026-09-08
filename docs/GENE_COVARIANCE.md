# Gene dependence in factor-GMT PheWAS

The binary Combined gene-hit regression tests gene-list enrichment. Shared annotations can correlate gene responses. Neither a fitted intercept nor the `--ols` flag establishes calibration for that dependence. The historical factor-GMT run did not invoke the separate annotation-derived correlation estimator.

## Explicit correlations

Use `--multi-y-gene-correlation-list MANIFEST.tsv` with `--multi-y-in`, `--multi-y-vectorize-betas`, and betas mode. The manifest has columns `trait`, `matrix`, `genes`, and `response_scale`. Every requested trait must have exactly one row. `matrix` is a SciPy sparse NPZ **correlation** matrix; `genes` is a one-gene-per-line file without a header (optionally gzipped). Paths are relative to the manifest. Gene labels are used to align and subset each matrix to the complete fitted gene universe. Matrix and gene-file SHA-256 values are recorded in parameters for every trait.

`response_scale` must be `binary` for logistic gene-hit regression or `linear` for linear regression. Linear runs must set `--max-for-linear 1` to prevent automatic logistic switching. The input must represent residual correlation for the actual fitted response. A correlation computed from continuous Combined scores is not automatically valid for thresholded gene hits. Labelling a matrix `binary` declares the user's model; it does not prove that model is calibrated.

The explicit input is used even with `--ols`, which otherwise skips automatic correlation-file loading. Other gene-correlation or gene-location inputs cannot be combined with this manifest. Missing traits, duplicate labels, incomplete coverage, nonfinite values, nonunit diagonal, asymmetric or indefinite matrices fail explicitly. Failure to verify positive semidefiniteness also fails explicitly. Loading and validation use a quarter of `--max-gb`; reduce `--multi-y-max-phenos-per-batch` if the matrices exceed that budget. The underlying existing model has additional allocations outside this loader budget.

This interface corrects **marginal** gene-set uncertainty. The subsequent Bayesian factor adjustment retains its existing approximation; it is not a fully covariance-aware joint logistic likelihood.

## Corrected calculations

For a marginal linear fit with intercept, center the predictor, c = x - mean(x). Multiply the independent SE by sqrt(c' R c / c' c). For logistic regression, use c_i = sqrt(p_i (1-p_i)) (x_i - weighted_mean(x)), where the mean is weighted by p_i (1-p_i). These are intercept-adjusted sandwich quadratic forms. Valid negative correlations can reduce SEs; corrections are not artificially clamped to increases only. Identity correlation preserves independent SEs.

The existing logistic pseudocount is treated as an independent synthetic observation, matching the fitted-model convention. A separate information-count bug was fixed: the observation count now includes the appended pseudocount. This slightly changes independent logistic SEs too, while leaving fitted coefficients unchanged.

For simultaneous linear regression, covariance is residual_variance times (B'B)^-1 B' R B (B'B)^-1. The previous correction omitted residual_variance. Identity now agrees with the uncorrected joint calculation and uncertainty scales with outcome units. This retains the existing residual-variance estimator; it is not a full finite-sample GLS fit.

## Annotation-derived working model

The continuous-response PheWAS path builds a centered Gram covariance from each gene's annotation contributions and adds squared Direct support on the diagonal only. It normalizes this matrix to correlation and blends it equally with identity, retaining the historical shrinkage convention and support gates. Genes with zero total variance receive identity correlation. A factorized operator stores annotations and gene vectors, avoiding a dense gene-by-gene matrix. The historical pairwise threshold argument is retained for source compatibility but no longer sparsifies correlations: arbitrary entrywise thresholding can destroy positive semidefiniteness.

This replaces flawed Direct cross-products and variance normalization. It is a **working covariance model**, not posterior uncertainty propagated through annotation training. Its gates, shrinkage, and interpretation of Direct support as residual variance still require calibration. It is not silently connected to binary factor-GMT inference.

The separate Huber-with-correlation branch used an inconsistent weighted covariance and omitted outcome scaling. It now fails explicitly instead of reporting unsupported uncertainty. Correlated Huber inference requires a separately validated robust sandwich implementation. Uncorrelated Huber inference is unchanged and is outside this patch's calibration claim.

## Validation

Tests compare marginal logistic uncertainty against an independently assembled dense sandwich, linear corrections against centered quadratic forms, and joint linear uncertainty against the full simultaneous covariance. Checks cover identity, dense/sparse inputs, trait and gene ordering, degenerate outcomes, single-trait operators, invalid manifests and covariance matrices, memory limits, provenance, and CLI batch equivalence.

A seeded 2,000-replicate clustered-Bernoulli null simulation with known correlation rejected 4.35% at nominal 5%, compared with 34.1% when correlation was ignored. The SE ratio matched the analytic value 2.03715. This validates the calculation under the supplied model, not calibration of the annotation estimator or the real HDL gene-hit P value.

## Empirical robustness audit

The [EAGGL methods supplement](eaggl/methods.tex) includes a [separate gene-dependence section](eaggl/gene_dependence_robustness.tex) and [HDL sensitivity outputs](eaggl/validation/gene_dependence/README.md). It tests fixed structural-overlap correlation assumptions, necessary binary-correlation bounds, source omissions, and gene-group deletions without new simulations. The saved bundle lacks HDL annotation coefficients, so these checks do not reconstruct or validate a fitted HDL-specific annotation covariance. Native EAGGL HC3 errors handle heteroscedasticity, not cross-gene dependence; the native binary fitter does not implement chromosome-clustered uncertainty.
