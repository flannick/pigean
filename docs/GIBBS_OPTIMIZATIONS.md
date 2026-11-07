# Gibbs arithmetic optimizations

Legacy arithmetic remains the default. The numerical `inner-rhat-stable` stopping fix and `warm-start-layout` row-mapping correction are enabled by default with every engine. For CLI genetic-input Gibbs, grouped preparation and the `inner-centered-variance` numerical correction are also enabled by default; the latter applies only during Gibbs. See [preparation](PREPARATION_ARITHMETIC.md) and [variance qualification](INNER_CENTERED_VARIANCE.md). To enable all recommended Gibbs arithmetic optimizations, append this complete suffix to your existing `pigean gibbs` command:

```sh
--gibbs-engine parallel-prefilter --gibbs-enable inner-matmul,inner-fused,inner-batching,compiled-prefilter,inner-cache64,huge-prior-softmax,static-correlation,compiled-moments
```

The full configuration requires the optional `compiled` dependency extra (Numba). The preset enables the first eight features below; the comma-separated list adds the eight later extensions. These change arithmetic, allocation and reuse, preserving the intended inner/outer Gibbs procedure. Floating-point reduction order can change sampled trajectories. They are performance options, not convergence fixes.

## Selecting features

Both `--gibbs-enable` and `--gibbs-disable` accept comma-separated names, repeated flags, or a mixture. For example, `--gibbs-enable inner-matmul,inner-fused` is equivalent to `--gibbs-enable inner-matmul --gibbs-enable inner-fused`. Whitespace within a quoted list is trimmed; duplicates are ignored. Empty names, unknown names, and enabling and disabling the same feature are errors.

Omit an arithmetic extension's enabling flag to leave it off. Remove a preset feature with, for example, `--gibbs-disable annotation-cache,compiled-covariance`. Missing prerequisites make a feature explicitly inactive; they are never silently re-enabled. For the original baseline, use `--preparation-engine legacy --gibbs-engine legacy --gibbs-disable inner-rhat-stable,warm-start-layout,inner-centered-variance` without enabling other flags. Existing arithmetic presets retain their definitions; numerical defaults are applied separately.

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
| `compiled-moments` | Combine five grouped moment sums into one traversal | `grouped-regression`; optional Numba |

Guarded optimizations use fallback arithmetic for unsupported inputs. Missing Numba raises an import error when a compiled path is used; to run without it, omit `inner-fused`, `inner-batching`, `compiled-prefilter` and `compiled-moments`, and add `--gibbs-disable compiled-covariance` to the preset. Cache payload limits do not bound total process memory. HuGE normalization must be reconciled separately if future fPS changes touch the same calculation.

## Measured performance and limits

The [milestone evidence](https://github.com/flannick/pigean/tree/optimization-milestone-2/docs/experiments/optimization-milestone-2) records 500-update full Gibbs replays from saved inputs, excluding upstream GWAS preparation. T2D inference decreased from historical legacy 6201.42 s to 309.42 s (20.04x), or 2.76x versus a contemporaneous first-milestone control. Exported priors, corrected beta and `beta_uncorrected` were identical to that first milestone. Peak process RSS decreased from historical 2.479 GiB to 1.486 GiB; memory ordering across configurations varies.

NAFLD's full comparison excluded `static-correlation` and measured an additional 1.415x speedup, with peak RSS +4.88%. Outputs differed substantially and chains remained non-converged; an unchanged-code alternate seed showed comparable sensitivity. These measurements do not establish convergence or general finite-run output equivalence. Grouped arithmetic can also differ from legacy standard errors for constant annotations because of cancellation.

Earlier inner stopping, conditional-mean inference, and incremental residual caching were rejected and are not included in these controls. Independent fPS, variance-threshold, and abandoned debug branches are not part of this optimization configuration.

## Default inner stopping stability

`inner-rhat-stable` is **on by default**, independently of the arithmetic engine.
No extra enabling flag is needed with the recommended suffix above or with
legacy arithmetic. To restore the historical stopping rule, append:

```sh
--gibbs-disable inner-rhat-stable
```

The opt-out covers preliminary fitting and the main Gibbs stage. Direct Python
sampling also uses the numerical default; a `marginal_backend` context with
`disable=['inner-rhat-stable']` restores the old rule. Effective configuration
and Gibbs feature logs report the selected numerical default. Explicit enabling
remains supported, including comma lists.

The existing inner mean-R-hat rule averages only entries >=1. A value rounded
just below one is excluded, while exactly one is included, potentially changing
the denominator enough to trigger an additional Gibbs sweep. This feature maps
finite float64 R-hat values within 16 spacings at one (about 3.55e-15) to exactly
one **only for that mean**. Raw R-hat outputs, outlier detection and max-based
stopping remain unchanged. Material deviations from one are not rounded away.

This is a numerical stopping change, not a posterior approximation or a
demonstrated mixing improvement. It can change sweep counts, random-number
consumption and finite-run results. It does not guarantee identical trajectories
across arithmetic backends. Promotion is supported by the captured numerical defect and the bounded
full-run validation below; the arithmetic recommendations remain independent.

The [bounded T2D validation](experiments/inner-rhat-stability/README.md) passed:
the disabled path reproduced historical optimized outputs exactly; the enabled
path removed the captured numerical split. In one full pair, active-panel
R-hat q90 improved 1.841 -> 1.216 with inference time +0.73% and peak RSS +0.65%.
Enabled versus control prior r/slope was .9842/.9973 and corrected beta
.8374/.8848. These finite-run shifts and the single-seed scope still limit interpretation:
no general mixing or speedup claim is made. The default change reproduces the
previously validated enabled path; the opt-out preserves the historical path.

## Additional grouped-moment optimization

`--gibbs-enable compiled-moments` optionally combines the five grouped-regression
moment reductions into one compiled traversal. It requires `grouped-regression`
and the optional Numba dependency; it is off by default and absent from presets.
Probability calculations and each output's multiplication/addition order are
preserved. Five output vectors remain, while intermediate full-group weight
arrays are avoided. This changes computation only, not sampling or inference.

Matched 500-update replays preserve exported priors, corrected/uncorrected beta,
log BF, recorded numeric trajectories and every recorded support selection
exactly. Profiled inference improves 1.071x for T2D and 1.121x for NAFLD
(1.053x against its earlier control). Peak RSS changes -10.24% and -0.98%
in the selected pairs; process peaks vary and are not an isolated memory-saving
guarantee. This option joins the recommended suffix while remaining off by
default and independently removable. See the [evidence and limits](experiments/compiled-moments/README.md).
A bounded synthetic benchmark is available in
`scripts/experiments/benchmark_compiled_moments.py`.

## Default warm-start row mapping

`warm-start-layout` is **on by default**, independently of the arithmetic engine.
When inner sampling receives a separate warm-start row for each outer problem,
the corrected broadcast copies each row to that problem in every inner chain.
The previous tile/reshape could instead place a different problem's row there.
Both coefficient and inclusion-probability initializers use the corrected mapping;
input validation, sanitization, clipping and subsequent sampling are unchanged.
One-dimensional, single-row and already-expanded inputs keep their behavior.

To reproduce the previous mapping, append:

```sh
--gibbs-disable warm-start-layout
```

Explicit `--gibbs-enable warm-start-layout` is also accepted. The opt-out covers
preliminary fitting and the main Gibbs stage; direct Python callers can use
`marginal_backend(..., disable=['warm-start-layout'])`. Effective configuration
and feature logs report the setting. Arithmetic presets are unchanged.
`--no-warm-start` disables warm starts altogether, which is different from
retaining them with the historical row mapping.

This fixes the initialization contract; it is not a demonstrated mixing or speed
improvement. Finite inner runs can change trajectories and annotation weights.
In two 500-update T2D research histories, prior correlations were .9895/.9862,
while beta correlations were .8198/.8911; all ten prespecified gene controls
retained support. Gene-prior R-hat q90 changed 1.335 to 1.375 and 1.519 to 1.388:
mixing did not improve consistently. An earlier NAFLD continuation also showed
mixed output changes, including four gene-preservation flags. These limited
comparisons do not establish biological equivalence or posterior convergence.
Only the row-mapping correction is promoted here; no research sampler,
calibration, stopping policy or inference approximation is included.
