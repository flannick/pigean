# Optional preparation arithmetic

`--preparation-engine grouped-regression` reuses sparse/grouped marginal-regression arithmetic during annotation ingestion and initial marginal fitting. `sparse-regression` is also available; `legacy` remains the default. This selector is independent of `--gibbs-engine`, feedback and retained-sampling settings.

The formulas preserve uncertainty corrections and input filtering rules. Floating-point differences can affect decisions at numerical thresholds; equivalence must be checked on the intended input. Initial inner coefficient fitting, prior calculation and Gibbs retain their separately selected behavior. Existing numerical-default opt-outs remain respected.

Example, adding to an existing command:

```sh
--preparation-engine grouped-regression
```

Return to original preparation with `--preparation-engine legacy`. This experimental opt-in has not yet received full four-trait, end-to-end qualification. It does not alter the memory budget or reduce diagnostic output.

The first matched six-library T2D preparation-only pilot preserved selected annotations, sparse inputs and RNG states exactly. All 125 numeric-array comparisons passed relative tolerance 2e-8 / absolute tolerance 2e-10; initial beta and beta-uncorrected maximum absolute differences were below 4e-14. Preparation time was 162.36 versus 137.32 seconds and peak RSS 3.787 versus 3.670 GB. These are single pilot observations with concurrent test-suite load; clean timing and full-sampler validation remain required before promotion.
