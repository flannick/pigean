# Stable inner variance

`--gibbs-enable inner-centered-variance` replaces cancellation-prone variance accumulation for active inner Gibbs convergence diagnostics with anchored Welford moments. It is enabled by default for CLI genetic-input Gibbs (`--gwas-in` or `--huge-statistics-in`) and independent of preparation arithmetic. Use `--gibbs-disable inner-centered-variance` for the original calculation. Other CLI workflows and direct Python defaults are unchanged. The option applies within the outer Gibbs stage; initial preparation fitting retains its existing feature scope. Python callers may explicitly enable the feature through the existing backend context.

The model, corrections, draws and 1.01 stopping threshold are unchanged. More accurate diagnostics can nevertheless change stopping, outlier decisions and subsequent random draws. This is a numerical correction, not a claim of improved statistical mixing or identical outputs to the original arithmetic. The existing `inner-rhat-stable` feature concerns membership near R-hat one and remains separately selectable.

The accumulator subtracts each chain's first observation, updates its mean deviation and centered sum of squares online, and computes between-chain variance before reconstructing absolute means. Only active slots accumulate observations. Replacement chains inherit the accumulator history of their source. Inactive-slot handling, short-history defaults and the existing zero-within-variance convention are preserved; outer diagnostic calculations are outside this change.

Persistent overhead is three float64 arrays of inner chain × parallel slot × annotation shape, plus slot counts. For four inner chains, ten outer slots and 5,000 annotations, this is about 4.8 MB. Temporary arithmetic arrays add transient allocation. No full draw history is stored. Work scales with inner sweeps and active annotations.

## Qualification

The matched NAFLD preparation comparison exposed a stopping discrepancy at outer update 4, inner sweep 11. Original sum-of-squares arithmetic produced thresholded mean R-hat 1.0069455 versus 1.0137769, causing an extra sweep in grouped preparation. Observing the actual draws without changing the trajectory gives 1.00911734173 versus 1.00911734171 with the centered reference: both stop at the same 1.01 threshold.

For the offending coefficient, within-chain variance is approximately 1.50841004e-17 and raw R-hat 1.03573288. The online implementation agrees with a separate 70-digit Decimal calculation to relative tolerance 1e-12. Native prefixes and complete RNG records reproduce in both observers. Tests cover independent centered references, masked counts, constant/short sequences, outlier ancestry, full inner-fit integration and context isolation. Full-run qualification is recorded separately; do not infer full-output equivalence from this short replay.

## Full NAFLD result

The same seed-12, six-library NAFLD comparison completed 1,250 updates per arm (250 discarded, 1,000 retained). With centered variance enabled in both arms, legacy and grouped preparation have identical initial identities, all 1,250 post-update RNG states, final RNG states and every exported value by ID. Prior, log-BF, beta and uncorrected-beta correlations/slopes are 1, with 200/200 top-rank overlaps. This resolves the observed preparation discrepancy for the tested history.

Gene rows are byte-identical after decompression. Seven filtered annotation rows reorder at equal displayed P; their complete text is identical by ID. The strict row-order receipt remains separate from this formatting-only review.

| Variance / preparation | Preparation seconds | Gibbs seconds | Total minutes | Peak RSS GB | Physical peak GB |
|---|---:|---:|---:|---:|---:|
| Original / legacy | 166.07 | 265.22 | 7.39 | 3.040 | 0.710 |
| Original / grouped | 128.79 | 260.58 | 6.69 | 3.170 | 0.780 |
| Centered / legacy | 164.37 | 273.05 | 7.50 | 2.772 | 0.731 |
| Centered / grouped | 127.12 | 251.87 | 6.51 | 2.584 | 0.775 |

Historical ordinary controls were measured earlier; this is not a replicated timing experiment. The unchanged sampler trajectory in the centered pair still has different wall times, so do not attribute every timing change to preparation or claim that stable variance speeds up Gibbs. Physical footprint and RSS are separate macOS measurements, in decimal GB.

The centered result differs from original variance under legacy preparation: prior correlation/slope 0.98995/0.92850, log-BF 0.99942/0.99559, beta 0.87728/0.84568 and uncorrected beta 0.99851/0.92809. Top-200 overlaps are 193/197/113/166 respectively. PNPLA3 prior changes 2.17→2.09, TM6SF2 2.51→2.41, GCKR 1.47→1.42 and APOA1 5.02→4.45. The lipid controls retain support, but these selected controls do not establish biological non-inferiority. Beta sum rises 7.815→8.664, with 85→95 coefficients above 0.01.

On the fixed 200 highest original priors, exported native R-hat q90 changes 1.29→1.25 and the fraction above 1.25 changes 15.5%→9.0%. The corresponding beta fraction changes 1%→2%. These mixed, single-history diagnostics do not establish a mixing improvement and are not rank-normalized split-R-hat QC. The subsequent cross-trait and independent-history qualification is below.

Analysis evidence is under `results/inner_variance_20261107/` in the optimization workspace. Implementation is separately committed as `8922754`. These measurements used the opt-in implementation; the subsequent release decision below enables it by default for genetic-input Gibbs. The Gibbs model is unchanged. The original preparation-only NAFLD failure remains relevant when centered variance is disabled.

## Cross-trait and independent-history qualification

Five additional full jobs used fixed source `a9bfd3b` (scientific implementation `8922754`): centered variance for T2D/CAD/Asthma at seed 12, then ordinary and centered NAFLD at fresh seed 14. All retained the six libraries, legacy preparation, ten outer slots, four inner chains, sampled feedback, and 250 discarded plus 1,000 retained updates. Existing seed-12 controls were reused. Every job completed within the fixed 100-minute total / 30-minute per-job / 6 GB RSS limits; total supervisor time was 69.47 minutes. No model tuning or additional seed search was performed.

T2D, CAD and Asthma preserve all 24 gene fields and all 42 annotation fields exactly by ID, including R-hat and MCSE. Primary correlations/slopes are 1 and top-200 overlaps are 200/200. Initial state/RNG checks pass; T2D uses an earlier matching preparation replay because its original full receipt lacks those fields. CMIP, AUTS2 and CCND2 are unchanged.

| NAFLD quantity | Seed 12 correlation / slope | Seed 14 correlation / slope | Top-200 overlap, seed 12 / 14 |
|---|---:|---:|---:|
| Prior | 0.98995 / 0.92850 | 0.98987 / 0.97584 | 193 / 193 |
| Log-BF | 0.99942 / 0.99559 | 0.99918 / 1.00061 | 197 / 196 |
| Beta | 0.87728 / 0.84568 | 0.86784 / 0.83223 | 113 / 119 |
| Uncorrected beta | 0.99851 / 0.92809 | 0.99784 / 0.98073 | 166 / 161 |

For seed 14, PNPLA3 prior changes 2.24→2.19, TM6SF2 2.58→2.56, GCKR 1.50→1.40 and APOA1 4.78→4.97. No prespecified control with prior above 0.1 falls to 0.01 or less. This does not exclude losses elsewhere: JUNB changes 0.548→−0.142 and JUND 1.12→0.481; annotation allocations differ substantially. These exported priors are evidence-scale quantities, not probabilities. Beta sum changes 9.203→8.146 and counts above 0.01 change 96→80, opposite the seed-12 increase. Neither history introduces materially negative coefficients or establishes general biological non-inferiority.

All-gene prior RMSE from the correction is 0.0444/0.0423, comparable to 0.0430 between the ordinary histories. That comparison is less favorable among leading signals: fixed original top-200 prior between-history RMSE rises 0.1228→0.1866 (52%), and uncorrected-beta RMSE rises 0.0481→0.0842 (75%). Native top-200 prior R-hat q90 improves 1.29→1.25 in seed 12 but worsens 1.09→1.25 in seed 14; MCSE q90 rises 0.287→0.308 and 0.166→0.282. Better mixing or absolute reproducibility is not demonstrated. These two histories also cannot establish a systematic deterioration caused by the numerical correction.

| Trait / seed | Total minutes, ordinary → centered | Peak RSS GB, ordinary → centered | Physical peak GB, ordinary → centered |
|---|---:|---:|---:|
| T2D / 12 | 16.24 → 15.75 | 3.511 → 3.202 | Not measured → 0.854 |
| CAD / 12 | 21.72 → 22.20 | 3.271 → 3.520 | 0.910 → 0.926 |
| Asthma / 12 | 16.54 → 16.85 | 4.270 → 3.599 | 0.981 → 1.042 |
| NAFLD / 12 | 7.39 → 7.50 | 3.040 → 2.772 | 0.710 → 0.731 |
| NAFLD / 14 | 7.56 → 7.00 | 3.062 → 3.174 | 0.719 → 0.815 |

No large runtime penalty is evident; small differences are not isolated feature effects. Asthma's 1.042 GB physical peak occurred during preparation before the feature is active. This is not a memory optimization. Full evidence, fixed-panel diagnostics, control changes, largest shifts and breadth tables are under `results/inner_variance_validation_20261107/comparison/` in the optimization workspace.

## Release decision and EAGGL review

The user approved default-on with independent opt-outs and merging if the actual NAFLD EAGGL comparison showed no clear overall biological advantage for old results. Four matched current gene-by-gene EAGGL fits on existing ordinary/centered histories12/14 completed in3.46min; all20 discovery restarts and24 projections pass. Factor counts change23→22 and23→21. Broad lipid, insulin-related, hepatic-transport and lipid-storage structure remains.

There is no clear overall biological winner. Old retains a more cohesive ACLY/APOD/ACAT2/FDFT1 group in both histories; native gene support remains substantial and weighted annotation beta support is0.909×/0.993× (uncorrected0.926×/0.984×) in new. GCKR best factor rank improves6→2 and18→1; PNPLA3 changes38→48 and43→26, while TM6SF2 changes8→14 and10→10. The JUNB/JUND/JUN-rich factor disappears in history14 with an upstream support reduction; it was not reproduced in old history12. Independent-history median best gene-factor cosine is0.925 old versus0.912 new. This supports a release judgment with retained broad biology, not universal non-inferiority, uniformly better factors or improved mixing.

Release defaults add the flag during CLI genetic-input Gibbs option resolution, preserving its tested Gibbs-stage-only scope. It is deliberately excluded from the global default-feature set, initial preparation fitting and direct Python defaults. Explicit disable flags/configuration win; the preparation option remains independently selectable. Other workflows can still enable this component explicitly.

To restore the preparation and variance behavior preceding this PR, use:

```sh
--preparation-engine legacy --gibbs-disable inner-centered-variance
```

Sampled feedback and the250-discarded/1000-retained schedule remain independent defaults from PR23. Numerical/default tests and full result tables are retained in the optimization workspace under `results/NAFLD_eaggl_preparation_release_20261107/`. EAGGL source is unchanged. Earlier failed standalone-preparation and mixed-diagnostic records remain valid and are not replaced by this release decision.

Release checks: 474 focused numerical, feature, scope, configuration, regression and CLI tests pass, plus the native default-versus-explicit recipe test (475 total). The native test verifies 1250 actual updates, exactly1000 retained, and byte-identical gene/annotation exports. A separate compact native comparison verifies both opt-outs reproduce the pre-default checkout byte-for-byte. Generated CLI manifest/help checks pass. No hosted CI result is inferred from these local tests.
