# Optimization milestone 1

User-accepted first major performance improvement, 2026-10-12. Annotated tag: `optimization-milestone-1` on `codex/parallel-prefilter`. The tag adds documentation/evidence to the tested cumulative implementation `f9a36f8`; all eight atomic optimization implementations are ancestors. This is a performance milestone with legacy still the default, not a package major-version release or a convergence claim.

## Measured T2D result

One matched seed; 500 executed outer updates; ten outer/four inner chains. Restarts and stall truncation disabled; original fixed-budget controls preserved. Times begin with a trusted saved pre-Gibbs input state, excluding upstream GWAS/input construction.

| Measure | Legacy | All optimizations | Change |
|---|---:|---:|---:|
| Gibbs inference | 6201.42 s | 893.26 s | 6.94x faster |
| Saved-input-to-output elapsed | 6215.17 s | 909.35 s | 6.83x faster |
| Peak process RSS | 2.48 GiB | 1.48 GiB | 40.4% lower |
| Inner fits | 554.98 s | 589.63 s | 8.9% / 66.0% of inference |

Final prior Pearson correlation/slope: 0.9870 / 0.9792. Final `beta_uncorrected`: 0.9965 / 0.9905. Corrected beta: 0.8757 / 0.8573. Coverage and missingness match; trajectories diverge. Beta R-hat q90: 1.679 / 1.841, both above 1.25. Statistical equivalence is not established by this pair. The user accepted this practical performance gain and broad agreement; original strict numerical/10x gates remain unmet in the immutable comparison JSON. Inner times compare distinct realized trajectories, so they are not an isolated inner-cache benchmark.

## Contents and use

The implementation retains `--gibbs-engine legacy` as default. Select the complete optimization stack with `--gibbs-engine parallel-prefilter`; the compiled path requires the optional Numba dependency. Intermediate engines remain available. No merge, package version bump or default change is part of this milestone. The inherited constant-column standard-error cancellation issue remains documented in the tracker.

- [Master tracker snapshot](plans/optimization_tracker.md): all completed and deferred options.
- [Next directions](plans/next_inference_directions_20261012.md): bounded inner arithmetic, inner approximation, then broader inference choices.
- [Full-run comparison](results/T2D_full_budget_20261012/comparison.json) and [readable summary](results/T2D_full_budget_20261012/summary.md).
- [Run methods](logs/2026-10-12_T2D_full_budget.md), per-engine receipts and expanded command records.
- [113 focused-test output](results/parallel_covariance_20261012/tests.txt); prior method logs provide commands and limitations.
- [Archive manifest](manifest.json): hashes and original analysis location.

Archived scripts are exact analysis-driver snapshots, not new package entry points. Their commands depend on the original analysis layout, shared Python environment, trusted fixture, and named worktrees. To rerun there, use `../.venv/bin/python scripts/run_T2D_full_budget.py` with a fresh output path; the existing driver deliberately refuses to overwrite completed results. Raw fixture and large outputs remain in the analysis directory and are excluded from git. The fixture metadata records its checksum. Links from historical logs to excluded large artifacts require that original analysis directory. No new fits were launched for tagging.
