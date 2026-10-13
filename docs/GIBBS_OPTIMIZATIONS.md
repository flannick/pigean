# Gibbs arithmetic optimizations

Legacy arithmetic remains the default. To enable all recommended optimizations, append this complete suffix to your existing `pigean gibbs` command:

```sh
--gibbs-engine parallel-prefilter --gibbs-enable inner-matmul,inner-fused,inner-batching,compiled-prefilter,inner-cache64,huge-prior-softmax,static-correlation
```

The full configuration requires the optional `compiled` dependency extra (Numba). The preset enables the first eight features below; the comma-separated list adds the seven later extensions. These change arithmetic, allocation and reuse, preserving the intended inner/outer Gibbs procedure. Floating-point reduction order can change sampled trajectories. They are performance options, not convergence fixes.

## Selecting features

Both `--gibbs-enable` and `--gibbs-disable` accept comma-separated names, repeated flags, or a mixture. For example, `--gibbs-enable inner-matmul,inner-fused` is equivalent to `--gibbs-enable inner-matmul --gibbs-enable inner-fused`. Whitespace within a quoted list is trimmed; duplicates are ignored. Empty names, unknown names, and enabling and disabling the same feature are errors.

Omit an extension's enabling flag to leave it off. Remove a preset feature with, for example, `--gibbs-disable annotation-cache,compiled-covariance`. Missing prerequisites make a feature explicitly inactive; they are never silently re-enabled. For the original baseline, use `--gibbs-engine legacy` without enabling flags. Existing engine presets retain their definitions.

| Feature | Change | Prerequisite or limit |
|---|---|---|
| `sparse-regression` | Sparse regression moments and covariance inflation | Sparse inputs; preserves cancellation fallback |
| `grouped-regression` | Reuse statistics for repeated annotation values | Grouped sparse sufficient statistics |
| `annotation-cache` | Reuse fixed annotation value groups | `grouped-regression` |
| `compact-covariance` | Aggregate covariance edges | `grouped-regression` |
| `lean-regression` | Avoid unused annotation stacks | `grouped-regression` |
| `cached-inner` | Reuse correlation slices within an inner fit | Bounded 16 MiB payload cache |
| `compiled-covariance` | Compile covariance edge preparation | `compact-covariance`; optional Numba |
| `parallel-prefilter` | Two threads for independent prefilter chunks | `sparse-regression` |
| `inner-matmul` | Use 2-D BLAS for dense inner products | Shared dense correlation path |
| `inner-fused` | Compile conditional arithmetic using existing random draws | Supported float64 layouts; optional Numba |
| `inner-batching` | Compile the existing greedy batch construction | Supported dense/default-order inputs and NumPy 1.26; optional Numba |
| `compiled-prefilter` | Direct sparse covariance quadratic forms | `sparse-regression`; optional Numba |
| `inner-cache64` | Raise the inner slice cache from 16 to 64 MiB | `cached-inner` |
| `huge-prior-softmax` | Normalize signal prior odds directly | Preserves regional and covariate adjustments |
| `static-correlation` | Reuse fixed full annotation correlations | Unwhitened float64 CSC; 256 MiB cap; content-validated |

Guarded optimizations use fallback arithmetic for unsupported inputs. Missing Numba raises an import error when a compiled path is used; to run without it, omit `inner-fused`, `inner-batching` and `compiled-prefilter`, and add `--gibbs-disable compiled-covariance` to the preset. Cache payload limits do not bound total process memory. HuGE normalization must be reconciled separately if future fPS changes touch the same calculation.

## Measured performance and limits

The [milestone evidence](https://github.com/flannick/pigean/tree/optimization-milestone-2/docs/experiments/optimization-milestone-2) records 500-update full Gibbs replays from saved inputs, excluding upstream GWAS preparation. T2D inference decreased from historical legacy 6201.42 s to 309.42 s (20.04x), or 2.76x versus a contemporaneous first-milestone control. Exported priors, corrected beta and `beta_uncorrected` were identical to that first milestone. Peak process RSS decreased from historical 2.479 GiB to 1.486 GiB; memory ordering across configurations varies.

NAFLD's full comparison excluded `static-correlation` and measured an additional 1.415x speedup, with peak RSS +4.88%. Outputs differed substantially and chains remained non-converged; an unchanged-code alternate seed showed comparable sensitivity. These measurements do not establish convergence or general finite-run output equivalence. Grouped arithmetic can also differ from legacy standard errors for constant annotations because of cancellation.

Earlier inner stopping, conditional-mean inference, and incremental residual caching were rejected and are not included in these controls. Independent fPS, variance-threshold, and abandoned debug branches are not part of this optimization configuration.

## Optional inner stopping stability

Add `--gibbs-enable inner-rhat-stable` to use a numerically stabilized inner
stopping statistic. This independent feature works with legacy arithmetic or
any optimization subset. It is **off by default and absent from all presets**;
the recommended arithmetic-only suffix above continues to select the old rule.
Omit the flag to retain that rule. It also accepts comma-list selection.

The existing inner mean-R-hat rule averages only entries >=1. A value rounded
just below one is excluded, while exactly one is included, potentially changing
the denominator enough to trigger an additional Gibbs sweep. This feature maps
finite float64 R-hat values within 16 spacings at one (about 3.55e-15) to exactly
one **only for that mean**. Raw R-hat outputs, outlier detection and max-based
stopping remain unchanged. Material deviations from one are not rounded away.

This is a numerical stopping change, not a posterior approximation or a
demonstrated mixing improvement. It can change sweep counts, random-number
consumption and finite-run results. It does not guarantee identical trajectories
across arithmetic backends. Full-run agreement, runtime and memory validation
should accompany promotion; it is separate from the arithmetic recommendations.
