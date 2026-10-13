# Independent Gibbs feature switches

Existing `--gibbs-engine` presets keep their historical meaning and legacy remains the default. Use repeatable `--gibbs-enable FEATURE` and `--gibbs-disable FEATURE` to override any subset. Unknown names and simultaneously enabling/disabling the same feature are errors.

Examples:

```sh
python -m pigean gibbs --gibbs-engine parallel-prefilter --gibbs-disable cached-inner ...
python -m pigean gibbs --gibbs-engine legacy --gibbs-enable cached-inner ...
python -m pigean gibbs --gibbs-engine legacy --gibbs-enable grouped-regression --gibbs-enable compact-covariance ...
```

| Feature | Prerequisite |
|---|---|
| sparse-regression | None; sparse marginal arithmetic and linear prefilter |
| grouped-regression | None; its own grouped sparse sufficient statistics |
| annotation-cache | grouped-regression |
| compact-covariance | grouped-regression |
| lean-regression | grouped-regression |
| cached-inner | None |
| compiled-covariance | compact-covariance |
| parallel-prefilter | sparse-regression |

Any subset is accepted. A feature whose prerequisite is absent is inactive and falls back to the remaining selected implementation; prerequisites are never silently re-enabled. The CLI reports effective features and inactive prerequisites when overrides are supplied. For example, disabling grouped-regression also makes grouping-cache/compact/lean/compiled features inactive, while leaving inner caching and sparse prefilter choices independent. Grouped statistics inherently use sparse sufficient-statistic arithmetic, even when the separate sparse-regression fallback/prefilter option is disabled.

Pure-Python and compiled covariance traversal, cache lifetime, thread limits and sampler semantics are unchanged by this configuration layer. All 256 original feature subsets receive a small numerical regression check; representative inner sampler tests and existing preset tests cover sampling behavior. These are configuration tests, not performance measurements for every subset.

## Stage 2 options

Four additional arithmetic flags can be mixed independently: `inner-matmul`, `inner-fused`, `inner-batching`, and `compiled-prefilter`. The last requires `sparse-regression`; other inner flags guard their supported execution paths. None is added to an existing preset. Two separately labelled inference probes, `inner-early-check` and `inner-mean`, are also off by default. Incremental residual caching remains on its rejected experiment branch and is excluded from this combination.

Retained candidate command suffix:

```text
--gibbs-engine parallel-prefilter --gibbs-enable inner-matmul --gibbs-enable inner-fused --gibbs-enable inner-batching --gibbs-enable compiled-prefilter
```

Omit any enabling flag to disable that extension. Use `--gibbs-disable` to remove preset features; dependent features become explicitly inactive if their prerequisites are removed.

Final arithmetic candidate also adds `--gibbs-enable inner-cache64` and `--gibbs-enable huge-prior-softmax`. The first requires `cached-inner`; the second independently normalizes HuGE signal prior odds. Both remain off in every preset. The six-feature combination changes arithmetic/cache capacity only; `inner-early-check` and `inner-mean` must be explicitly added to run inference experiments.

The optional seventh arithmetic extension is `--gibbs-enable static-correlation`, which caches fixed full annotation correlations within a 256 MiB cap and validates input content on each request. Unsupported/oversized/whitened inputs fall back. It has full T2D evidence; the NAFLD full comparison uses the six-feature configuration. See the milestone evidence for this distinction and the observed peak-memory variability.

Recommended T2D arithmetic suffix:

```text
--gibbs-engine parallel-prefilter --gibbs-enable inner-matmul --gibbs-enable inner-fused --gibbs-enable inner-batching --gibbs-enable compiled-prefilter --gibbs-enable inner-cache64 --gibbs-enable huge-prior-softmax --gibbs-enable static-correlation
```

The independent `inner-early-check` and `inner-mean` probes remain available but were rejected from the recommended stack after screening. Legacy is still default; no package version or main-branch promotion is made.
