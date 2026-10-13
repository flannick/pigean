# Inner R-hat mean-boundary stability — 2026-10-13

Retain `--gibbs-enable inner-rhat-stable` as an independent opt-in numerical
repair. Keep every preset and the default unchanged. This is not an arithmetic
equivalence claim or a general sparse-mode mixing solution.

## Implementation and integration

Tested implementation: `bd27e8ec36799a7fab485c035877dd3f7ef3e702`, based on remote
main `5ba1ac5`, including EAGGL PR #19. The EAGGL integration does not modify
PIGEAN's sampler or shared regression code. No diagnostic/inference, fPS or
variance-threshold branch is merged into this repair branch.

The legacy inner mean averages only R-hat entries >=1. One and the next float
below one can therefore give materially different denominators. The new flag
snaps finite float64 values within 16 spacings at one (3.55e-15) to exactly one
on a temporary copy used only for that mean. Raw R-hat, max-based stopping and
outlier detection retain their original calculations. Mathematical model and
conditional updates are unchanged; stopping, RNG consumption and finite-run
estimates can change. It is independent of arithmetic backend selection.

## Tests and localized causal check

391 numerical/feature/CLI tests plus four subtests and 46 existing inner-cache
and Gibbs-summary tests pass. Tests exercise actual stopping decisions on a
one-step rounding perturbation, material deviations, degenerate/nonfinite
values, max-based stopping, unchanged raw diagnostics/RNG, default-off behavior
and CLI selection with legacy arithmetic. Commands from the code worktree:

```sh
PYTHONPATH=src ../../../.venv/bin/python -m pytest \
  tests/test_inner_rhat_stability.py tests/test_gibbs_features.py \
  tests/test_pigean_cli_unittest.py -q -o addopts=
PYTHONPATH=src ../../../.venv/bin/python -m pytest \
  tests/test_gibbs_summary_unittest.py tests/test_cached_inner.py -q -o addopts=
```

The earlier spike captured outer update 157. Without stabilization, full
preparation plus correction from identical starting inputs/RNG stopped at
inner index 11 in optimized arithmetic and 12 in legacy arithmetic; corrected
mean beta differed by .0208593. With the production flag, both return at 11,
finish with identical RNG, and corrected mean beta differs by only 2.29e-14
(all outputs <=3.92e-13). This tests the production path; the diagnostic snap
override was not used.

## Matched full-run validation

Exactly two sequential seed-0 runs, 500 outer updates, ten outer/four inner
chains, recommended seven-feature arithmetic stack. The control explicitly
disables the new flag; the second enables it. Fixture checksum:
`f4a455b83c957e9c22ba1abdb61004ab281db059c3ca45e06c8462c7b445727e`.
All inference settings and source commits match. Both final receipts pass the
clean-worktree guard and record `COMPLETE_FIXED_BUDGET`.

The control reproduces every saved historical recommended prior/beta/
uncorrected-beta checkpoint exactly, along with exported priors, both beta
outputs and log BF. Thus EAGGL/current-main integration and the disabled feature
preserve this benchmark.

| Measure | Current-main control | Flag enabled |
|---|---:|---:|
| Inference seconds | 337.981 | 340.433 |
| Fixture-to-output seconds | 354.066 | 356.289 |
| Peak RSS GiB | 1.4052 | 1.4143 |
| Active-panel beta R-hat q90 | 1.84082 | 1.21629 |
| Active coefficients in that panel | 54 | 47 |
| R-hat >1.25 on fixed historical legacy top-200 absolute-beta IDs | 16 | 2 |
| R-hat >1.25 on each run's own top-200 absolute-beta panel | 20 | 4 |
| R-hat >1.25 on each run's own top-200 smallest-P panel | 9 | 0 |
| Relative beta MCSE q90 | .10481 | .06829 |
| D MCSE q90 | .06336 | .04051 |
| Relative prior/beta inconsistency q90 | .00243 | .00320 |

Inference cost +0.73%, wall +0.63%, peak RSS +0.65% in this pair. No incremental
speedup or memory saving is claimed. Against historical legacy's 6,201.42 s,
this enabled run is 18.22x faster, a cross-session arithmetic comparison, not
a gain attributable to this repair. Process RSS includes allocator variation
and trajectory-dependent state; the helper itself adds a temporary array with
one value per parallel/annotation in the inner R-hat panel.

| Output | Enabled vs current control, Pearson r / slope | Enabled vs historical legacy, Pearson r / slope |
|---|---:|---:|
| Prior | .98419 / .99726 | .99389 / .99915 |
| Corrected beta | .83742 / .88480 | .91768 / .94922 |
| Uncorrected beta | .99507 / 1.01300 | .99825 / 1.01018 |
| Log BF | .99815 / .99972 | .99936 / .99947 |

Slopes regress the enabled result on the named baseline with an intercept.
Rows are aligned by 19,746 gene IDs or 4,912 annotation IDs; no duplicate IDs.
Corrected-beta agreement with the current control is lower than the previous
optimized seed-0/seed-1 correlation (.9004). Relative to historical legacy,
agreement is higher than the original optimized run (.8757). Neither
non-converged baseline is a posterior ground truth.

At update 500, the enabled run passes the monitored R-hat, beta/D MCSE and
prior/beta-consistency thresholds, with seven consecutive passing diagnostic
checks. This fixed-budget driver sets stopping patience to 1,000, so it did not
stop by the normal precision rule. Call this an improved endpoint, not proven
general convergence, unbiasedness or fewer required outer iterations. The
active panel changes between runs; the fixed-ID comparison addresses that
particular composition confound. No second seed or NAFLD generalization is
claimed, and no additional jobs were launched.

## Evidence and rerun entry points

Compact receipts and comparisons are in this directory's `evidence/`. Large
captures/traces and input data remain in the analysis workspace under
`results/inner_rhat_stability_20261013/` and the prior spike's `capture157/`.
The local analysis scripts used were:

```sh
bash scripts/run_inner_rhat_validation.sh control
bash scripts/run_inner_rhat_validation.sh stable
../.venv/bin/python scripts/summarize_inner_rhat_validation.py \
  --control results/inner_rhat_stability_20261013/full_control \
  --historical results/autonomous_20261012/T2D_static_full \
  --fixed-panel-baseline results/T2D_full_budget_20261012/legacy \
  --stable results/inner_rhat_stability_20261013/full_stable \
  --out results/inner_rhat_stability_20261013/full_comparison.json.gz
```

Run from the analysis root with its documented environment and trusted fixture.
The launch script refuses to overwrite outputs. Use fresh result paths for a
repeat. Each run's `screen.json` preserves expanded arguments, sampler settings,
source/fixture identity, compiler information, runtime and memory. Future
validation should assess reproducibility across seeds/traits before making this
a default or claiming a mixing improvement.
