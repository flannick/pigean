# 2026-10-11 grouped regression experiment

The user requested the next optimization step and a final summary of performance, numerical agreement, and the change. Profiled the first sparse candidate with the same trusted T2D fixture, seed 0, five-iteration budget, 140-second cap, and original chain/inner-sampler controls. Inference completed in 17.305 profiled seconds across four executed outer updates: covariance correction consumed 10.145 seconds, and IRLS moment construction consumed 5.115 seconds. The new profiling stayed within the five-minute budget. Artifacts are under `results/binary_statistics_20261011/`; this directory retains the initial binary-statistics hypothesis name.

Read-only inspection of fixture annotations found 1,740 binary columns among 4,912 T2D columns and 642 among 3,364 NAFLD columns. Most remaining annotations have several repeated values rather than continuously unique values: maxima were 75 distinct nonzero values for T2D and 53 for NAFLD. Generalized the hypothesis from binary sufficient statistics to exact grouping of repeated annotation values; no rounding or discretization is introduced.

Created `codex/grouped-regression-statistics` from the first sparse checkpoint `ff4e4ce`. This is a dependent arithmetic extension, independent of fPS and variance-threshold branches. The implementation groups equal values to evaluate IRLS moments once per group, weighted by counts. It prepares within-annotation covariance summaries once per regression/correlation object and reuses them across chains. Coefficient-dependent influences are recalculated, pseudocounts remain independent, and caches expire before the next regression/Gibbs update. Operator and unstable-quadratic fallbacks remain available. The flag is `--gibbs-engine grouped-regression`; `legacy` remains default and `sparse-regression` remains unchanged.

Initial focused numerical validation passed 29 tests, including weighted annotations, genuinely continuous columns, chain-specific active masks, trait-specific covariance, constant outcomes, and dispatcher isolation. The driver adds only the new backend selection; short matched controls will be refreshed to retain identical driver hashes. This avoids silently comparing receipts from different driver implementations. Four-update screens establish arithmetic and preliminary timing evidence only, not convergence or full-run speedup.

Expanded selection passed 48 numerical, covariance, dispatcher, and replay tests. The CLI manifest check initially failed because the newly added experiment document changed option-reference counts after manifest generation; regenerated the manifest after the document was final, then the remaining CLI check passed (one test, 3.07 seconds). Thus all 49 selected checks passed, with the final manifest check rerun alone. Source checkpoint: `a592c1f`. No source change followed this checkpoint before screening.

Started serial short paired screens with the updated `scripts/screen_inference.py`, using seed 0, five-iteration budget, 140-second cap, and gzip outputs. Each trait receives one refreshed legacy control and one grouped candidate. Existing first-sparse timings remain an exploratory incremental comparator; refreshed controls provide the main old-versus-new comparison.

## Final refinement and screening decision

Initial grouped screens took 7.723876 seconds for T2D inference and 3.076512 seconds for NAFLD. Brief profiles identified repeated sparse submatrix construction and one-off linear-prefilter preparation as avoidable setup costs. The refinement reads CSR arrays directly for within-annotation edges, aggregates group weights using vectorized counts, and retains sparse linear-prefilter arithmetic with 512-column operation batches. This is arithmetic batching, not the abandoned blockwise sampler redesign. Added coverage for unsorted CSR indices and negative correlations. The final 32 targeted numerical, covariance, multi-response, and CLI checks passed in 4.27 seconds. Local source checkpoint `d7c393aa10e4d2311166001530624f5e0b80f2f1` is clean; neither this checkpoint nor its parent was pushed.

Final candidate command, run serially once for each TRAIT in T2D and NAFLD:

```bash
../.venv/bin/python scripts/screen_inference.py --repo worktrees/grouped-regression-statistics --fixture results/feature_foundation_20261011/fixtures/TRAIT/fixture.pkl.gz --out results/binary_statistics_20261011/TRAIT_grouped_final --iterations 5 --seconds 140 --write-outputs --engine grouped-regression
../.venv/bin/python scripts/compare_inference_screen.py --baseline results/binary_statistics_20261011/TRAIT_legacy --candidate results/binary_statistics_20261011/TRAIT_grouped_final --out results/binary_statistics_20261011/comparison_TRAIT_final.json
```

The comparator verifies matching fixture hashes, driver hashes, seeds, inference controls, and executed-update counts. Each screen executed four outer updates with 10 outer and four inner chains; only the outer iteration budget was shortened. Results:

| Trait | Legacy inference seconds | Grouped inference seconds | Speedup | Legacy/grouped screen wall seconds |
|---|---:|---:|---:|---:|
| T2D | 42.469048 | 6.824191 | 6.2233x | 56.943128 / 20.870077 |
| NAFLD | 8.711843 | 1.499349 | 5.8104x | 21.901809 / 14.557456 |

Against the earlier first-sparse timings (18.6060 and 3.66353 seconds), these are approximately 2.73x and 2.44x additional inference gains; those comparisons are exploratory because the driver was refreshed. Screen wall time includes fixture/setup and output costs and is not full end-to-end GWAS runtime.

Exported priors and `beta_uncorrected` match exactly at output precision for both traits, including the top-200 union, with Pearson/Spearman correlations and slopes equal to one within floating-point precision and matching coverage/missingness. Maximum internal iteration differences for prior / `beta_uncorrected` / beta are 8.69083e-13 / 3.72757e-14 / 2.79776e-14 for T2D and 1.78302e-13 / 7.38298e-15 / 7.38298e-15 for NAFLD. All numerical screening gates pass.

Decision: retain an experimental local checkpoint with `--gibbs-engine grouped-regression`; default `legacy` and prior `sparse-regression` remain available. Both final comparisons return `SCREEN_ONLY` and `proceed_to_full_runs=false`, because neither reaches 10x. No full fits, larger matrices, default changes, merges, or pushes were performed. Four early updates do not measure mixing, convergence, or full-run speedup. Machine-readable evidence is in the final comparison files and `results/binary_statistics_20261011/decision.json`.
