# Preparation arithmetic

**Default for CLI genetic-input Gibbs; independently selectable otherwise.** `pigean gibbs --gwas-in ...` and `--huge-statistics-in ...` now select grouped preparation and the [stable inner variance correction](INNER_CENTERED_VARIANCE.md). Other workflows retain legacy preparation unless explicitly selected.

```sh
--preparation-engine grouped-regression
```

Use `--preparation-engine legacy` to restore original preparation. Use `--gibbs-disable inner-centered-variance` independently to restore the original inner variance calculation. Disabling only centered variance retains the known standalone grouped-preparation NAFLD sensitivity documented below.

With centered variance enabled, all1250 RNG states and every exported value match exactly between preparation modes in the tested full NAFLD history. T2D/CAD/Asthma exported values also match exactly. The correction changes NAFLD trajectories relative to original variance, with mixed diagnostics. A matched two-history EAGGL review retains broad biological structure and finds no clear overall old/new winner; old has some more cohesive groups. The numerical fix is defaulted for accuracy, not a demonstrated mixing improvement. Its initial-preparation fitting scope is unchanged.

This reuses existing sparse/grouped marginal-regression arithmetic during annotation ingestion and initial marginal fitting. Repeated annotation values share exact sufficient statistics; values are not discretized. The resolved default is `grouped-regression` for genetic-input Gibbs and `legacy` otherwise; explicit CLI/config choices take precedence. The intermediate `sparse-regression` choice is also available, but the full-trait qualification below concerns `grouped-regression`.

Preparation selection is independent of `--gibbs-engine` and [Gibbs arithmetic features](GIBBS_OPTIMIZATIONS.md). It does not select feedback or an iteration budget. Initial inner coefficient fitting, priors and Gibbs retain their separately selected behavior; numerical-default opt-outs remain respected. Corrections and filtering formulas are unchanged. Floating-point differences can nevertheless alter threshold decisions and subsequent sampling trajectories, as the NAFLD qualification demonstrates.

## Runtime and memory

Matched six-library commands used seed 12, ten outer slots, four inner chains, and 250 discarded/1,000 retained updates. Control is main `f17ebca`; candidate scientific implementation is `24197af` (`12fe9a1` adds documentation). T2D was measured 2026-11-06; the other traits 2026-11-07. Each pair was serial; one history per trait. Both arms used the same recommended optimized Gibbs arithmetic, and only the candidate added grouped preparation.

| Trait | Preparation seconds, original → grouped | Total CLI minutes, original → grouped | Peak RSS GB, original → grouped | Full-run physical peak GB, original → grouped |
|---|---:|---:|---:|---:|
| T2D | 161.10 → 131.64 | 16.24 → 15.35 | 3.511 → 4.474 | Not measured |
| Asthma | 194.75 → 165.82 | 16.54 → 16.50 | 4.270 → 4.397 | 0.981 → 1.105 |
| CAD | 184.32 → 150.00 | 21.72 → 21.34 | 3.271 → 3.074 | 0.910 → 0.755 |
| NAFLD | 166.07 → 128.79 | 7.39 → 6.69 | 3.040 → 3.170 | 0.710 → 0.780 |

Preparation time fell 14.9–22.4%. Whole-command changes were smaller and varied; the unchanged Gibbs phase also varied, and this single-pair design does not isolate the cause of those differences. Do not attribute all end-to-end timing differences to this feature or claim a Gibbs speedup. Preparation savings occur once per invocation, grow with annotation work, and do not grow with the number of Gibbs iterations.

GB is decimal. Physical peaks use macOS kernel lifetime counters, including transient peaks between polls; RSS remains separately reported. The new full runs use a small streaming observer, with no full research trace history, tracemalloc or allocator cleanup. Total CLI timing excludes module imports and includes output writing. Earlier roughly 6 GB Asthma RSS readings did not measure physical footprint and also included qualification observation; they cannot be retrospectively converted into physical RAM usage.

In this matched Asthma pair, peak physical footprint increased 0.981→1.105GB (124 MB, 12.6%), while preparation saved 28.93 s. Physical footprint at Gibbs entry was 0.419→0.432GB. This is a startup-time optimization, not a general memory reduction. CAD's observed physical peak decreased; NAFLD's increased. These are platform-specific single-pair observations.

A separate T2D preparation/20-update repeat gave 157.60→129.82s, RSS 3.556→3.939GB, and physical peak 0.800→0.753GB. That short-prefix physical peak is not a full-run measurement.

## Output qualification

T2D, Asthma and CAD preserved every exported value exactly by stable gene/annotation ID, including priors, log-BF, beta, uncorrected beta, R-hat and MCSE. Primary correlations/slopes are 1 and all top-200 overlaps are 200/200. Gene output bytes match after decompression; 4/2/5 filtered annotation rows respectively reorder at equal displayed P, with identical contents by ID. Strict row-order failure receipts are preserved separately from this formatting-only review.

NAFLD passed initial design/RNG identity and numerical-array checks (maximum discrepancy 2.71e-14), but failed final RNG and output-equivalence gates:

| Quantity | Correlation | Slope, grouped on original | RMSE | Top200 overlap |
|---|---:|---:|---:|---:|
| Gene prior | 0.99271 | 0.96623 | 0.03599 | 195/200 |
| Genetic log-BF | 0.99962 | 0.99795 | 0.00688 | 197/200 |
| Annotation beta | 0.90391 | 0.91210 | 0.01058 | 125/200 |
| Uncorrected beta | 0.99868 | 0.96376 | 0.01683 | 161/200 |

Maximum prior difference is 0.52. PNPLA3 prior 2.170→2.280 and TM6SF2 prior 2.510→2.580 retain support; GCKR 1.470→1.380 and APOB 4.940→4.540 decrease. These selected controls do not establish general biological non-inferiority or override the failed gate.

An unchanged four-update replay reproduced both native log prefixes. RNG states match through outer updates 1–3 and on entry to update 4. At inner sweep 11 of update 4, slot 0's thresholded mean R-hat is 1.0069455 versus 1.0137769, crossing the 1.01 stopping threshold. One coefficient's raw R-hat is 1.022702 versus 1.063690 because the sum-of-squares variance calculation subtracts quantities near 0.364, leaving residuals around 1e-16. Estimated within-chain variance changes from 1.804e-17 to 1.110e-17; one candidate chain's centered sum is negative from cancellation. Mean effects differ only at the last bits. The additional inner sweep changes subsequent RNG consumption and trajectories.

This is a numerical sensitivity in the existing inner stopping calculation, not an added model or sampling method. The existing near-one membership stabilization does not address this variance cancellation. The separately selectable centered-variance implementation now resolves this discrepancy in the tested full NAFLD pair; neither feature widens tolerances or claims to fix statistical convergence. The subsequent bounded review supports defaulting the paired configuration with opt-outs, while preserving the changed-trajectory and biological limitations above. Do not treat the paired pass as a pass for grouped preparation with centered variance disabled.

## Reproducibility and deferred work

Analysis workspace records: `results/preparation_full_20261106/`, `results/preparation_footprint_20261107/`, and `results/preparation_finish_20261107/` (commands, frozen specs, full-output assessments, native progress/memory records and NAFLD split replay). Full runs retain the original failed NAFLD assessment and STOPPED supervisor receipt; all six underlying commands completed. No full-run rerun was used to hide the failure.

The final grouped-preparation profile identified covariance-edge preparation (59.28 s of 169.40 s instrumented) and text parsing (36.06 s own time) as follow-ups. Nested profile timings are not additive or promised savings. Compiled covariance without ingestion-wide annotation caching and versioned parsed-input caching remain separate candidates, outside this PR. No allocator-trim hook is added.
