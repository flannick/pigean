# Outer-Gibbs stopping, reruns, and restarts

PIGEAN treats iteration budget, precision stopping, and stall handling as three separate controls:

1. The iteration budget says how long the sampler is allowed to run.
2. Precision stopping says when estimates are already stable enough to stop early.
3. Stall detection is an optional recovery strategy that can end an epoch and start a new one when diagnostics stop improving.

The default is one uninterrupted epoch capped at 500 outer iterations. Precision-based early stopping remains enabled. Stall-triggered exits and restarts are off unless requested.

Independent reruns are a fourth, explicit control. They run several fixed-controller chain batches sequentially and summarize all of their chains together, reducing peak working width when many chains are wanted.

## Default behavior

```bash
PYTHONPATH=src python -m pigean gibbs \
  ...input and output options...
```

This resolves to the following controller behavior:

- `--max-num-iter 500`: at most 500 outer iterations in the epoch.
- `--total-num-iter-gibbs 500`: at most 500 outer iterations over the complete run.
- `--max-num-restarts 0`: use one epoch.
- stall windows are disabled.
- across-chain burn-in and precision diagnostics remain active, so the run may finish before iteration 500.

Use `--print-effective-config` to see the resolved values without starting the analysis. The run's `--params-out` file also records the effective controller and stopping settings.

## Sequential independent reruns

Use `--gibbs-reruns` to obtain more effective chains without widening the matrices used by one Gibbs batch:

```bash
PYTHONPATH=src python -m pigean gibbs \
  --num-chains 20 \
  --gibbs-reruns 3 \
  ...input and output options...
```

This runs three independent 20-chain batches and reports one combined 60-chain result. Every batch starts from the same pre-Gibbs inputs, while the seeded random-number stream advances between batches. A fixed seed therefore reproduces the complete run without making the batches identical.

The implementation retains chain sums, squared sums, and sample counts in memory, then computes the final means, MCSE, confidence intervals, activity probabilities, chain mask, and R-hat once across all effective chains. If burn-in or precision stopping leaves different retained sample counts across batches, pooled R-hat uses those per-chain counts rather than assuming equal lengths.

Precision stopping remains active within each batch. Reaching precision ends that batch but does not cancel later explicitly requested reruns. The default total iteration capacity becomes `--max-num-iter` multiplied by `--gibbs-reruns`; an explicitly supplied `--total-num-iter-gibbs` must provide at least that capacity.

Version 1 supports independent reruns only with the default fixed controller. `--gibbs-reruns` above 1 cannot be combined with `--enable-stall-detection` or a nonzero `--max-num-restarts`. Intermediate reruns are not checkpointed: native outputs are written after the combined summary is complete.

The mental model is a set of chain shards. Increasing `--num-chains` makes one shard wider and can have superlinear runtime costs; increasing `--gibbs-reruns` adds shards sequentially, so peak memory and per-batch matrix width stay near the selected `--num-chains`. More shards improve Monte Carlo precision but do not guarantee convergence; inspect the combined R-hat alongside MCSE.

## Iteration and phase bounds

| Flag | Role |
|---|---|
| `--max-num-iter N` | Maximum outer iterations in one epoch. It is 500 by default and also supplies the default total budget when stalls are disabled. |
| `--gibbs-reruns N` | Number of independent fixed-controller chain batches. The default is 1. |
| `--total-num-iter-gibbs N` | Maximum outer iterations summed across all epochs. An explicit value overrides the default derived from `--max-num-iter`. |
| `--min-num-burn-in N` | Minimum burn-in draws in each epoch before burn-in may end. |
| `--max-num-burn-in N` | Optional burn-in cap within each epoch. |
| `--min-num-post-burn-in N` | Minimum retained post-burn draws in each epoch before precision stopping may occur. |
| `--max-num-post-burn-in N` | Optional retained post-burn cap within each epoch. |
| `--diag-every N` | Evaluate burn-in, precision, and enabled stall diagnostics every N outer iterations. |

For the default uninterrupted controller, increasing only `--total-num-iter-gibbs` does not increase the one-epoch cap. Increase `--max-num-iter` as well when more than 500 iterations are wanted in one epoch.

Example: one uninterrupted trajectory capped at 800 iterations:

```bash
PYTHONPATH=src python -m pigean gibbs \
  --max-num-iter 800 \
  ...input and output options...
```

Example: cap the run at 300 iterations while retaining precision-based early stopping:

```bash
PYTHONPATH=src python -m pigean gibbs \
  --max-num-iter 300 \
  ...input and output options...
```

### Burn-in convergence

Burn-in can end before its cap when the across-chain effect R-hat summary passes its threshold for the required number of consecutive checkpoints.

Burn-in cannot complete while the active-effect diagnostic panel fills its configured top-K capacity. A full panel may be a truncated transient rather than a stable representation of the active effects, so its pass streak is reset until the panel is no longer saturated.

| Flag | Role | Default |
|---|---|---:|
| `--r-threshold-burn-in X` | Maximum burn-in R-hat summary. | 1.10 |
| `--burn-in-rhat-quantile Q` | Quantile across active gene-set effects used for the burn-in R-hat summary. | 0.90 |
| `--burn-in-patience N` | Consecutive passing burn-in checks required. | 2 |
| `--use-max-r-for-convergence` | Use the maximum active-effect R-hat instead of the configured quantile. | off |

## Precision-based early stopping

Precision is evaluated across the parallel outer chains after burn-in. A diagnostic checkpoint passes only when all four monitored quantities meet their thresholds:

- relative Monte Carlo standard error for active gene-set effects;
- absolute Monte Carlo standard error for monitored gene probabilities;
- post-burn R-hat for active gene-set effects;
- consistency between directly summarized priors and priors implied by summarized corrected effects.

The pass must repeat for `--stop-patience` consecutive diagnostic checkpoints, and the epoch must have at least `--min-num-post-burn-in` retained draws. This path is independent of stall detection and remains active under the default 500-iteration controller.

Precision stopping is also blocked while the active-effect diagnostic panel is saturated at `--active-beta-top-k`. Sampling continues until the panel becomes unsaturated or the iteration budget is exhausted.

| Flag | Role | Lenient default | Strict preset |
|---|---|---:|---:|
| `--num-chains N` | Number of parallel chains contributing to convergence and precision estimates. | 10 | 10 |
| `--stop-mcse-quantile Q` | Quantile across monitored effects/genes used for MCSE summaries. | 0.90 | 0.95 |
| `--max-rel-mcse-beta X` | Maximum relative MCSE for active gene-set effects. | 0.20 | 0.05 |
| `--max-abs-mcse-d X` | Maximum absolute MCSE for monitored gene probabilities. | 0.10 | 0.03 |
| `--max-post-beta-rhat X` | Maximum post-burn R-hat summary for active effects. | 1.25 | 1.10 |
| `--max-rel-prior-beta-inconsistency X` | Maximum relative prior/effect-summary discrepancy. | 0.50 | 0.25 |
| `--stop-patience N` | Consecutive passing checkpoints required. | 2 | 2 |

`--strict-stopping` selects the strict values in the table unless an individual threshold is supplied explicitly. It changes the precision target; it does not enable stall detection.

The monitored subsets can be tuned with expert flags including `--active-beta-top-k`, `--active-beta-min-abs`, `--stop-top-gene-k`, `--stop-min-gene-d`, and `--beta-rel-mcse-denom-floor`.

## Optional stall detection

Enable the adaptive stall/restart controller with:

```bash
PYTHONPATH=src python -m pigean gibbs \
  --enable-stall-detection \
  ...input and output options...
```

Unless `--max-num-restarts` is explicitly supplied, this opt-in mode allows up to 10 additional restart epochs. The total iteration budget still limits work across all epochs; by default it follows `--max-num-iter`.

When enabled, PIGEAN watches burn-in and post-burn diagnostics for a plateau or deterioration. If precision has not been achieved, a detected stall may end the current epoch and start another epoch while restart attempts and total iterations remain. Completed post-burn samples are retained and aggregated.

Useful controls are:

| Flag | Role |
|---|---|
| `--max-num-restarts N` | Maximum number of additional epochs. Set 0 to allow stall detection to stop an epoch without restarting it. |
| `--burn-in-stall-window N` | Diagnostic-window length for burn-in plateau detection. |
| `--burn-in-stall-delta X` | Minimum burn-in R-hat improvement expected over that window. |
| `--stall-window N` | Diagnostic-window length for post-burn plateau detection. |
| `--stall-min-burn-in N` | Minimum burn-in age before stall logic can act. |
| `--stall-min-post-burn-samples N` | Minimum retained post-burn sample count before post-burn stall logic can act. |
| `--stall-delta-rhat X` | Minimum best-so-far R-hat improvement over the post-burn stall window. |
| `--stall-delta-mcse X` | Minimum best-so-far MCSE improvement over the post-burn stall window. |
| `--stall-recent-window N` | Window used to compare recent diagnostics with the full epoch. |
| `--stall-recent-eps X` | Tolerance before recent diagnostics count as worse than full-epoch diagnostics. |
| `--disable-stall-detection` | Explicitly select the default no-stall behavior; retained for backward-compatible command lines. |

### When to use stall detection

Stall detection is best treated as an opt-in exploratory controller. It may be useful when local modes are a specific concern and a user wants several shorter attempts within a fixed computational budget. Validate the result against an uninterrupted run for the trait and input configuration being analyzed.

Do not assume that restarts always save time or preserve effect amplitudes. In matched tests, the adaptive controller shortened BMI runtime but deflated effect/prior amplitudes, while for T2D it closely preserved those amplitudes but took longer than the uninterrupted 500-iteration run. Sparse or delayed-activation traits can be especially sensitive to declaring a stall too early. For routine production runs, the uninterrupted default is therefore the safer reference.

## Common command patterns

Sixty effective chains as three sequential 20-chain batches:

```bash
PYTHONPATH=src python -m pigean gibbs \
  --num-chains 20 \
  --gibbs-reruns 3 \
  ...input and output options...
```

Default controller with stricter early stopping:

```bash
PYTHONPATH=src python -m pigean gibbs \
  --strict-stopping \
  ...input and output options...
```

Custom precision target under the default 500-iteration cap:

```bash
PYTHONPATH=src python -m pigean gibbs \
  --max-rel-mcse-beta 0.10 \
  --max-abs-mcse-d 0.05 \
  --max-post-beta-rhat 1.15 \
  --max-rel-prior-beta-inconsistency 0.30 \
  --stop-patience 3 \
  ...input and output options...
```

Stall detection with a smaller restart allowance and an explicit shared budget:

```bash
PYTHONPATH=src python -m pigean gibbs \
  --enable-stall-detection \
  --max-num-restarts 2 \
  --max-num-iter 500 \
  --total-num-iter-gibbs 500 \
  ...input and output options...
```
