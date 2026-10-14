# Single-pass grouped moment reductions

Retain as an optional arithmetic optimization. The five weighted reductions in
grouped logistic regression now share one serial compiled traversal. Probability
calculations and each output's multiplication/addition order are unchanged;
fast-math is disabled. Inner and outer Gibbs, HuGE competition/refresh and the
strongest-signal rule are unchanged. There is no convergence improvement claim.

Enable with `--gibbs-enable compiled-moments` alongside grouped regression;
remove the enabling flag or use `--gibbs-disable compiled-moments` to recover
the old reduction. Missing prerequisites are reported inactive. The feature
requires optional Numba and is off by default, absent from all engine presets.
The [complete recommended suffix](../../GIBBS_OPTIMIZATIONS.md) includes it.

## Full-loop validation

All runs use matched saved inputs, 500 outer updates, ten outer/four inner
chains, all previously recommended arithmetic features, default stable stopping,
cProfile, numeric checkpoints and packed support masks. Only one sampler ran
at a time. Upstream GWAS processing is excluded. Compilation during inference
is included, with fresh per-run Numba caches. Whole replay includes preparation
and exports. Budget completion does not establish convergence.

| Trait / seed | Inference seconds off -> on | Inference speedup | Whole replay speedup | Peak RSS bytes off -> on | RSS change |
|---|---|---|---|---|---|
| NAFLD / 0, same-source repeat | 126.236 ->112.576 | 1.121x | 1.103x | 1,632,272,384 ->1,616,248,832 | -0.98% |
| T2D / 1 | 339.138 ->316.652 | 1.071x | 1.070x | 1,647,706,112 ->1,478,950,912 | -10.24% |

All exported prior, beta, uncorrected beta and log-BF values are exact, with
correlation and slope effectively 1. All recorded numeric checkpoints and all
500 sampled/feedback support masks match exactly. Missing entries and IDs are
checked separately. This does not claim a final RNG-state comparison or exact
agreement for every possible input/platform.

Candidate source is `4b4d156`; NAFLD's repeat control uses the same source with
the feature disabled. T2D's control is main `8069e42`; production differences
are confined to this opt-in feature. All run source guards passed. Earlier
NAFLD main control gives 118.494 ->112.576 s (1.053x) but peak RSS 1.197 ->1.616
GB, an apparent 35% increase. This triggered the one same-source repeat above,
which did not reproduce the memory penalty. Preserve both comparisons: timing
and process peaks vary, and these pairs do not establish a general memory gain.

NAFLD profiling attributes grouped moment time 25.498 ->10.511 s across 4,974
calls. Corrected inner-fit time is essentially unchanged (22.277/22.281 s).
Compiler dispatch takes 1.983/2.219 s across all compiled kernels. The screen
receipt's legacy `compilation_during_inference` field monitors only the
covariance kernel; it does not describe new moment-kernel compilation.

Synthetic 512/2,048/8,192-annotation reductions (ten chains, 64 groups each)
give 7.79/6.77/8.19x kernel speedups and exact arrays. They avoid up to four
simultaneous group-sized weight temporaries (10/40/160 MiB in these cases),
while retaining five output vectors. These are not whole-sampler speedups or
measured RSS savings. Savings repeat over regression iterations; compilation
amortizes, and scaling depends on group counts and how much runtime regression
occupies. Do not multiply these results by historical cross-session 20x timings.

## Reproduction and acceptance

The analysis project retains fixtures, full outputs, profiles and support masks
under `results/post_stability_20261014/`. Run sequentially from that project:

```sh
bash scripts/run_compiled_moments_screen.sh NAFLD control 50 0
bash scripts/run_compiled_moments_screen.sh NAFLD candidate 50 0
bash scripts/run_compiled_moments_screen.sh NAFLD candidate 500 0
bash scripts/run_compiled_moments_screen.sh NAFLD control 500 0
bash scripts/run_compiled_moments_screen.sh T2D candidate 500 1
```

Launchers refuse to overwrite existing outputs. The T2D baseline was the
completed `run_post_stability_pair.sh T2D stable 1` run. Compressed receipts and
comparisons under `evidence/` record hashes, seeds, settings, commits and metrics.
The retained kernel benchmark is independently runnable from this checkout.

The predeclared incremental gate required exact outputs, no material RSS
regression and about 5% or greater full inference gain; both traits pass.
The generic comparator retains a historical 10x gate field, which is not the
incremental criterion and remains false. Its promotion field is not overridden.
The arithmetic feature's recommendation is this explicit evidence-based decision.

All 410 focused kernel, grouped-regression, feature and CLI tests pass, including empty
groups, masked/extreme probabilities, covariance and pseudocount fits. The
updated CLI test exercises the complete recommended suffix. The unrelated
existing grouped-pipeline logger fixture repair is a separate test-only commit.
The CLI manifest freshness check and whitespace check also pass.
Main is not changed by this branch or its recommendation.
