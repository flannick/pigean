# Optional preparation arithmetic

`--preparation-engine grouped-regression` reuses sparse/grouped marginal-regression arithmetic during annotation ingestion and initial marginal fitting. `sparse-regression` is also available; `legacy` remains the default. This selector is independent of `--gibbs-engine`, feedback and retained-sampling settings.

The formulas preserve uncertainty corrections and input filtering rules. Floating-point differences can affect decisions at numerical thresholds; equivalence must be checked on the intended input. Initial inner coefficient fitting, prior calculation and Gibbs retain their separately selected behavior. Existing numerical-default opt-outs remain respected.

Example, adding to an existing command:

```sh
--preparation-engine grouped-regression
```

Return to original preparation with `--preparation-engine legacy`. This experimental opt-in has not yet received full four-trait, end-to-end qualification. It does not alter the memory budget or reduce diagnostic output.

The first matched six-library T2D preparation-only pilot preserved selected annotations, sparse inputs and RNG states exactly. All 125 numeric-array comparisons passed relative tolerance 2e-8 / absolute tolerance 2e-10; initial beta and beta-uncorrected maximum absolute differences were below 4e-14. Preparation time was 162.36 versus 137.32 seconds and peak RSS 3.787 versus 3.670 GB. These are single pilot observations with concurrent test-suite load; clean timing and full-sampler validation remain required before promotion.

Full T2D validation (merged main control, same six libraries and seed 12, 250 discarded/1,000 retained updates) also preserves the exported result: all 66 fields match exactly by ID. The gene file is byte-identical after decompression; annotation rows have identical contents, with two filtered-out pairs exchanging order at equal displayed p-values. Priors, log-BF, beta and beta-uncorrected have correlation/slope 1 and zero exported differences; all top 200 sets match.

Clean preparation time was 161.10 → 131.64 seconds (18.3% lower); total time 974.33 → 920.77 seconds (5.5% lower in one pair). Only 29.46 seconds of the total 53.56-second difference occurs in preparation; wall time in the unchanged Gibbs phase also varies. Peak RSS increased 3.511 → 4.474 GB. A separate Mac allocation diagnostic found a large reusable-allocator contribution to RSS, but the full comparison did not record physical footprint. Treat this as an opt-in startup-time improvement with an unresolved physical-memory tradeoff, not a demonstrated memory optimization. Cross-trait qualification remains outstanding, and no allocator cleanup is included in this feature.
