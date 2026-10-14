# Reliability judgment

**Reliable estimation has not been demonstrated for the current full PIGEAN
procedure or a replacement.** The three validation stages are complete at the
bounded-reference level. Small reference calculations and one target-preserving
mixing method pass, but inner and outer compatibility failures block a full-trait
inference candidate. No production defaults, model or main-branch source changed.

## 1. Intended model versus implemented updates

The methods describe an LDpred-inspired approximation using marginal logistic
estimates. Code standardizes both effects and SEs consistently; this is not a
missing annotation-unit conversion. A raw logistic example, before covariate or
dependence corrections, has SEs .26938/.28006 and incompatible implied precision.
Rescaling its annotation columns changes standardized effects and SEs by less
than 2e-16. Both captured real traits again fail all ten frozen-chain checks,
although their annotation-correlation matrices are exactly symmetric.

The same formulas are present in the legacy monolith. Available documentation
does not justify treating their heterogeneous-SE extension as an exact joint
logistic posterior. Authorial intent cannot be inferred from that absence; no
one-line scaling bug fix has been established. Changing the residual weights
would change the update law, and earlier alternative-target trait screens were
already rejected. They are not revived by this audit.

An analytic two-variable example demonstrates why stationarity is insufficient:
both scan orders have stable stationary laws and identical coefficient means,
but their cross-covariances differ by .20464. This is not Monte Carlo error.
The standard Gaussian precision consistency check remains a necessary inner
condition; it does not establish an outer target or explain all trait pathology.

## 2. Independently solvable references

| Reference | Validation | Scope |
|---|---|---|
| Spike/slab Gaussian model | Enumerate supports and integrate slabs analytically; independent one- and correlated two-dimensional adaptive integration checks pass | Tiny compatible coefficient target |
| Observed-response logistic model | Enumerate supports, Gaussian quadrature at 48/96 nodes; support-weight change 1.7e-16, plus independent one-dimensional adaptive check | Two coefficients, fixed intercept and hyperparameters |
| Fixed-evidence logistic model | Marginalized gene indicators; support-weight refinement change 2.8e-17 | Fixed gene evidence, excludes dynamic HuGE |
| Finite outer model | Enumerate two gene indicators and two three-state coefficients; exact Gibbs stationary joint TV error 6.4e-17 | Discrete slab control, not the production continuous prior |

These are known-answer references, not full-trait replacements. They allow an
algorithm to be judged against independent truth rather than another
non-converged chain. Quadrature refinement is a numerical check on these cases,
not a universal integration error bound.

## 3. Mixing and outer feedback

Eight chains, 10,000 sweeps, 2,000 burn-in, seed 117. The literal production
coefficient-update helper is used with frozen nuisance parameters and explicit
single-coordinate sweeps. The full production stopping controller is excluded.
Support error below means total-variation distance from the enumerated support
probabilities, not relative error in every rare support.

| Case / method | Support error | Maximum beta mean error | Maximum split beta / inclusion R-hat | Decision |
|---|---|---|---|---|
| Easy compatible case, sequential production update | .00122 | .00120 | 1.00002 / 1.00002 | Pass bounded reference |
| Competing supports, sequential production update | .03297 | .04789 | 1.18789 / 1.19285 | Fails predeclared .02 error and 1.05 R-hat gates |
| Same competing target, collapsed support moves plus lazy swap | .00285 | .00403 | 1.00003 / 1.00002 | Pass bounded reference |
| Easy case plus production sparsity projection | .26909 | .07766 | One coefficient stuck; R-hat undefined | Does not recover the unprojected posterior |

The compatible competing target has two single-annotation modes with probability
.49695 each. Gaussian integration removes continuous coefficient barriers from
the support moves. A swap proposal made with probability one half preserves the
target (exact stationarity error 1.1e-16); its spectral gap is .99394, compared
with .00606 for the collapsed single-site support kernel. Worst-initial-state
support error after 500 steps is below 2.5e-14 versus .02391 without swaps.
These exact transition calculations supply evidence beyond a favorable seed.

The initial always-proposed swap also preserved the target, but nearly
alternated modes and did not materially improve the absolute spectral gap.
Both initial and refined records are retained. Do not promote every accepted
swap design simply because its posterior averages look accurate.

Two further controlled checks expose issues that an inner mixing fix cannot
settle:

- Actual default batching groups .2-correlated coefficients. In an all-slab
  Gaussian counterpart, simultaneous updates preserve marginal means/variances
  but change the target cross-covariance from -.03284 to zero. Small errors in
  the easy spike/slab screen therefore do not prove exact joint invariance.
- Even an **exact inner conditional mean**, substituted into the next logistic
  gene draw, changes the finite outer reference distribution by .03785 TV and
  gene-indicator probabilities by up to .03377. More inner samples cannot remove
  this conditional-mean substitution error.

The actual HuGE competition/max and gene-probability helpers pass their fixed-q,
strongest-signal, duplicate-signal and signal-order checks (zero errors; the
independent probability formula agrees within 1.2e-16). Thus the audit preserves
the user-required biology. However, pairing their prior-dependent gene draw
with a plain exact logistic coefficient-given-indicator conditional produces a
mixed log-conditional-ratio error of 1.85938; compatibility requires zero. This
small counterexample shows that merely replacing the inner step with exact
logistic inference would not justify the outer procedure as exact Gibbs.
It is not evidence that HuGE's competition should be turned off.

## Full-trait status and disposition

Existing main-source stable-stopping receipts were re-read, not re-run:
T2D seed 1 R-hat q90 is 1.80388 and NAFLD seed 0 is 11.29083 after 500 updates,
both above 1.25. Their inputs differ, so they are not a matched trait comparison.
No new full trait jobs were launched: there is no justified full outer candidate
to test. No calibration or interval-coverage claim is established by this work.

Retain the diagnostic suite, independent references and modular collapsed/lazy
support-swap implementation as research components. Do not promote any new
inference behavior. The compatibility findings identify inherited approximations;
they do not prove that every existing point estimate is scientifically wrong.

Next, specify the intended full outer target or explicitly define and calibrate
the iterative estimation procedure. Keep fixed signal probabilities, evolving
prior competition and strongest-signal aggregation explicit. Any changed
coefficient update must be derived with those outer rules, and then pass tiny
joint-reference checks before trait tests. Improving mixing alone cannot validate
an unspecified or incompatible target. Shorter runs are not yet the objective.

## Evidence and commits

Research components `025a2ec`, audit driver/derivation `2a52ba4`, and lazy-swap /
batch-invariance checks `a79973e` are separate commits on
`codex/inference-reliability`. Sixteen focused tests pass, including independent
integration, conditional compatibility, exact transitions and HuGE invariants.
Production source is unchanged from baseline `8069e42`.

Compressed receipts and summaries under `evidence/initial/`, `evidence/lazy/`
and the existing full-trait diagnostic snapshot contain parameters and source
revisions. Full draws remain under the two analysis result directories named in
the [reproduction instructions](README.md). No scientific data inputs are modified.
