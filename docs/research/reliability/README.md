# Inference reliability audit

This branch adds research-only validation. Production source, flags, main and
arithmetic PR 21 are unchanged. The inference target, correctness, mixing and
runtime questions are kept separate. No iteration-saving claim is made.

## Derivation audit

`docs/methods.tex` (Fitting the model, lines 291–360; LDpred analogy, 917–945)
asserts a joint gene-indicator/coefficient model and adapts marginal logistic
estimates to an LDpred-style residual update. It does not give a derivation
establishing exact conditional compatibility for heterogeneous logistic SEs.
Its bibliography placeholders limit treating it as a finalized derivation.
The original [LDpred paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC4596916/)
itself describes approximate MCMC; an analogy does not establish exactness here.

Source paths on this branch:

- `src/pegs_shared/regression.py`, lines 511–515: both marginal effects and SEs
  are multiplied by annotation standard deviation. They already use standardized
  annotation units. Logistic Fisher information differs across annotations.
- `src/pigean/state.py`, corrected-input preparation near 13340–13450: V comes
  from the standardized annotation design, whereas fitted/corrected SEs are
  passed separately. No corresponding SE-weighting of V occurs here.
- Inner variance preparation near 5710 and the residual/draw update near 9310:
  slab mean is `shrink[j] * (b[j] - sum(V[k,j] * beta[k]))`, variance is
  `shrink[j] * se[j]^2`. Off-diagonal implied precision is `V[k,j]/se[j]^2`.
  With symmetric V and differing SEs this is generally asymmetric.
- The same construction occurs in `legacy/priors.py`, near 17039–17156. It is
  inherited, not introduced by the recent arithmetic optimizations.

For standardized linear regression with shared noise, equal marginal SEs can
make this construction compatible. Logistic regression has observation weights
that depend on fitted probabilities, and each marginal fit can use different
weights. Gene-dependence and other SE corrections can further change uncertainty.
Even making a symmetric Gaussian precision would not turn the marginal summary
approximation into the full logistic likelihood: its curvature changes with beta.

A consistent invertible change of coefficient units cannot fix incompatible
conditionals. Under `beta = A theta` for diagonal positive A, implied precision
transforms to `A Q A`; its antisymmetric part remains nonzero. Altering only V,
SEs or residual weights changes the update distribution. No one-line scale fix
has been justified by the available documentation. This is evidence of an
incompatible approximation, not evidence about the original author's intent.

Additional approximations have separate source paths: simultaneous correlated
updates (default batch sum threshold .5), optional projection of low-inclusion
coefficients to zero, and mean-coefficient feedback. The experiments isolate
these rather than attributing their effects to a new sampler.

## References and experiment design

The Gaussian exact enumeration and fixed-evidence logistic quadrature modules
are retained research references copied from `codex/inference-reference`
(commit `fe18332b2ace4e3f197b276dab5c98118e7928a2`), under
`research/reliability/`. They are not promoted production models. New independent
checks compare Gaussian support masses and logistic integrals against adaptive
SciPy quadrature, and check exact finite transition matrices.

`checks.py` invokes the literal production coefficient update and optionally its
actual batch builder or sparsity projection. Hyperparameters and outer state
remain fixed, with eight chains, 10,000 sweeps and 2,000 burn-in. These are tests
of update components, not the complete production controller. Both common-SE
Gaussian cases have a justified target; the heterogeneous-SE counterexample is
an analytic autoregressive calculation, without an invented target.

The bounded mixing gate is support total-variation error <=.02 and coefficient
and inclusion split R-hat <=1.05, alongside exact transition invariance and
independent truth. An easy case and one correlated competing-support case are
predeclared. Competing supports are initialized separately across chains.
Collapsed support moves use exact Gaussian integration; the optional symmetric
swap is proposed with probability one half and has a valid Metropolis probability. Both target and chain diagnostics are
checked. There is no claim that a two-annotation solution scales to full traits.

Two-variable continuous logistic references use Gaussian quadrature at 48 and
96 nodes per active coefficient; support-mass change must be below 1e-7. A
separate one-dimensional adaptive integration checks the quadrature machinery.

The outer-loop reference enumerates two gene indicators and two three-state
coefficients (a discrete slab, explicitly different from the continuous
production prior). Exact Gibbs must recover its known fixed-evidence joint.
The conditional-mean feedback experiment uses the exact inner conditional mean,
so finite inner sampling error cannot explain any discrepancy. Actual HuGE
competition/max and gene-probability helpers are evaluated on the same small
coefficient states with fixed signal probabilities. Positive finite conditionals
are compatible only if mixed differences of their log-ratio vanish. Refreshed
HuGE is judged on that compatibility condition, not on agreement with a different
fixed-evidence target. Signal probabilities, duplication, ordering, and
strongest-signal aggregation are tested independently.

## Reproduction

From the analysis project, with the isolated worktree committed and clean:

```sh
MPLBACKEND=Agg MPLCONFIGDIR="$HOME/codex-workspace/.cache/matplotlib" \
../.venv/bin/python worktrees/inference-reliability/scripts/research/run_reliability.py \
  --stage all --capture-root results/post_stability_20261014 \
  --out results/inference_reliability_20261013
```

Use `--swap-probability 0`, `0.5` (default), or `1` to select no swaps, lazy
swaps, or the initial always-proposed reference. Stages `audit`, `reference`,
`mixing`, and `outer` can be selected separately;
outputs refuse overwrite and include source revision and run settings. Existing
trait captures are checksum-verified. No full trait run is justified unless the
reference and outer-target gates support a concrete candidate. Final results
and the reliability judgment will be recorded alongside this derivation.

An initial always-proposed swap preserved the target but produced near-alternating
modes and did not improve worst-start mixing in the exact transition check.
The explicit half-probability proposal removes that periodicity concern. Initial
results are retained separately; the refinement is a new research commit.
