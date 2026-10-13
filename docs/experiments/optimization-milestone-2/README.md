# Optimization milestone 2

Performance milestone from the sequential 2026-10-12/13 experiment pass. Tested seven-feature source: `c281eec`; six-feature source: `cfbb7d9`. The final evidence commit adds documentation only. All implementations are separate commits; legacy remains default, no main merge or package-version bump is included, and fPS/variance branches remain separate.

## Full Gibbs results

Each fit executed 500 outer updates with ten outer/four inner chains. Upstream GWAS/input preparation is excluded. Whole replay includes saved-input loading, cold compilation, sampling and exports. The historical legacy comparison is across sessions; the contemporaneous milestone control is the stronger timing comparison.

| T2D configuration | Inference | Whole replay | Peak process RSS |
|---|---:|---:|---:|
| Historical legacy | 6201.42 s | 6215.17 s | 2.479 GiB |
| Current first-milestone control | 852.63 s | 868.48 s | 1.469 GiB |
| Seven additional arithmetic options | 309.42 s | 324.88 s | 1.486 GiB |

T2D: **20.04x inference / 19.13x whole-replay speedup versus historical legacy**, or **2.76x inference versus the contemporaneous first milestone**. Peak RSS is about 40% below historical legacy and 1.2% above the current milestone. Intermediate configurations measured higher RSS; allocation lifetimes and compilation influence this high-water measurement. The static cache itself adds up to 256 MiB retained payload (184 MiB on T2D), so do not interpret this as an isolated cache memory saving.

T2D priors, uncorrected and corrected betas have correlations, ranks and slopes effectively 1 against the first milestone, with matching sampled checkpoints to numerical tolerance. Against original legacy, the first milestone's stochastic differences remain: prior Pearson/slope 0.9870/0.9792, uncorrected beta 0.9965/0.9905, corrected beta 0.8757/0.8573. Beta R-hat q90 stays 1.84082; there is **no convergence improvement**.

NAFLD was tested with the six-feature configuration (excluding static correlation caching): **162.76 → 115.02 s, 1.415x faster**, peak RSS +4.88%. Its long outputs differ: prior Pearson/slope 0.9155/1.0262, uncorrected beta 0.9719/1.0681, corrected beta 0.4180/0.4778. R-hat q90 8.228/9.835 is poor in both. Disabling HuGE normalization reproduces the optimized outputs, so that change is not the trigger. One unchanged-code seed-0/seed-1 control gives corrected-beta correlation 0.4057, comparable to the optimization discrepancy. Early captured NAFLD arithmetic and batch checks pass, but one control does not establish posterior equivalence or reliable mixing.

## Retained options

All are independently enabled with `--gibbs-enable NAME` and omitted/disabled individually. Existing presets retain their historical definitions; unmet prerequisites are explicitly inactive.

| Feature | Change | Key qualification |
|---|---|---|
| `inner-matmul` | Send dense inner residual products through 2-D BLAS | Shared dense path; reduction rounding can differ |
| `inner-fused` | Compile masked conditional probabilities, means and supplied draws | Float64 path; optional Numba, no new RNG draws |
| `inner-batching` | Compile the existing greedy batch construction | Same partition; guarded NumPy 1.26/default dense ordering |
| `compiled-prefilter` | Direct sparse covariance quadratic forms | Requires sparse-regression; preserve cancellation fallback |
| `inner-cache64` | Raise fit-local slice cache from 16 to 64 MiB | Requires cached-inner; bounded capacity tradeoff |
| `huge-prior-softmax` | Normalize signal prior odds directly | Preserve regional/phantom/covariate semantics |
| `static-correlation` | Reuse content-validated fixed annotation correlations | Unwhitened CSC only; 256 MiB cap and fallback |

Recommended T2D suffix:

```text
--gibbs-engine parallel-prefilter --gibbs-enable inner-matmul --gibbs-enable inner-fused --gibbs-enable inner-batching --gibbs-enable compiled-prefilter --gibbs-enable inner-cache64 --gibbs-enable huge-prior-softmax --gibbs-enable static-correlation
```

The NAFLD full test omits the last enabling flag. [Detailed feature controls](../independent_features.md).

## Rejected or deferred

- Incremental residual caching: 0.879x captured-sweep speed, about 14% slower. The committed experiment remains on `codex/inner-residual`, outside the retained integration.
- Earlier inner checks: 1.079x short-screen gain with roughly 6–7% prior/uncorrected-beta scale shifts and changed corrected betas.
- Existing conditional-mean iteration: 1.037x short-screen gain with roughly 4% scale shifts and changed corrected betas. It is not a derived variational posterior.

The latter two remain off-by-default, explicitly experimental switches (`inner-early-check`, `inner-mean`) for reproducibility. Neither warranted a full follow-up. A direct VI substitution, persistent joint chain and valid support moves need a compatible target for the heterogeneous-SE inner conditionals and outer feedback; see the [target audit](logs/2026-10-12_inner_target_audit.md) and [next research plan](plans/next_inference_directions_20261013.md).

## Evidence and reproduction

- [Master tracker snapshot](plans/optimization_tracker.md): flags, defaults, agreement, runtime, memory and scaling for all options.
- [Methods and decisions](logs/2026-10-12_autonomous_optimization.md), [closeout](logs/2026-10-13_optimization_stage2_closeout.md).
- [T2D final increment](results/autonomous_20261012/T2D_static.comparison.json), [NAFLD full pair](results/autonomous_20261012/NAFLD_full_pair/comparison.json), [NAFLD seed control](results/autonomous_20261012/NAFLD_seed_sensitivity.json).
- [Seven-feature profile](results/autonomous_20261012/profile_static_T2D_100/profile.txt) and prior profiles; profile times localize work rather than establish speed.
- [Final 402-check validation](results/autonomous_20261012/final_archive_tests.txt), [earlier arithmetic checks](results/autonomous_20261012/final_arithmetic_tests.txt), and [static-cache checks](results/autonomous_20261012/static_correlation_tests.txt). Preliminary failing metadata checks are retained; subsequent regeneration/validation resolved them.
- [Environment](results/autonomous_20261012/environment.json), [archive manifest](manifest.json), [raw-artifact hashes](raw_artifacts.json).

Archived scripts are exact analysis-driver snapshots and use the original analysis layout, shared Python environment, named worktrees and trusted fixtures. Use fresh output directories; drivers refuse to overwrite completed results. Fixture metadata includes capture commands and hashes. Raw fixtures, large outputs, external inputs and compiler caches are excluded from Git. Historical links to those files require the original analysis directory. This is a reproducibility record, not a portable release bundle.

Nine full fits, four profiling passes, three inference screens and three deliberately stopped captures ran sequentially; no sampler remains queued. Brief unit checks overlapped a few seconds of some full runs, a limitation of the timing controls. No general finite-run interchangeability or convergence claim follows from this performance tag.
