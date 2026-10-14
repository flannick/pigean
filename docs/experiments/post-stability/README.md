# Stopping repair replication after PR 20

Two matched pairs on clean main `8069e42`, all retained arithmetic features,
500 outer updates, ten outer/four inner chains, fixed-budget stopping patience.
Each pair shares its trait fixture, seed and settings. All four clean-source
checks passed. cProfile and packed support tracing are enabled in both arms.

| Trait / seed | R-hat q90 off -> on | Fixed control top-200 failures | Profiled inference seconds | Peak RSS change |
|---|---|---|---|---|
| NAFLD / 0 | 9.835 ->11.291 | 18 ->19 | 124.35 ->118.49 | -17.54% |
| T2D / 1 | 1.484 ->1.804 | 17 ->17 | 332.54 ->339.14 | +18.39% |

| Trait | Prior r/slope | Beta r/slope | Uncorrected beta r/slope | Log BF r/slope |
|---|---|---|---|---|
| NAFLD | .8593/.6652 | .4142/.3573 | .9490/.6243 | .9969/.9940 |
| T2D | .9885/.9911 | .8612/.8870 | .9958/1.0007 | .9988/.9965 |

The earlier T2D seed-0 improvement does not generalize in these checks. The
numerical boundary defect remains demonstrated; the repair is not a general
mixing fix. Both new pairs remain non-converged. Keep the historical opt-out and
do not infer required iteration savings from these fixed-budget endpoints.
Memory can change with trajectories and allocations; these are process peaks,
not isolated repair-array costs. Two small unit-test runs overlapped about six
seconds of the T2D control; wall times are descriptive. No sampler jobs overlapped.

Packed nonzero masks record sampled and mean-feedback supports every update.
Terminal 100-update inclusion frequencies and dominant mean-feedback annotations
remain chain dependent. These summaries describe occupancy; dominant annotation
counts are not an independently validated classification of posterior modes.

Compressed comparison and screen receipts are in `evidence/`. The analysis
project retains full profiles, outputs and support traces under
`results/post_stability_20261014/`. Rerun sequentially with
`scripts/run_post_stability_pair.sh TRAIT control SEED` then `stable SEED`,
and summarize using `scripts/compare_post_stability.py`. The driver records its
hash, source commit, fixture checksum, complete settings, output status and
iteration count. Seeds are 0 for NAFLD and 1 for T2D.
