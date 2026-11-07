# Optional stable inner variance

`--gibbs-enable inner-centered-variance` replaces cancellation-prone variance accumulation for active inner Gibbs convergence diagnostics with anchored Welford moments. It is off by default and independent of preparation arithmetic. Omit it or use `--gibbs-disable inner-centered-variance` for the existing calculation. The option applies within the outer Gibbs stage; initial preparation fitting retains its existing feature scope. Python callers may explicitly enable the feature through the existing backend context.

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

On the fixed 200 highest original priors, exported native R-hat q90 changes 1.29→1.25 and the fraction above 1.25 changes 15.5%→9.0%. The corresponding beta fraction changes 1%→2%. These mixed, single-history diagnostics do not establish a mixing improvement and are not rank-normalized split-R-hat QC. No independent-seed or other-trait qualification of centered variance has been performed.

Analysis evidence is under `results/inner_variance_20261107/` in the optimization workspace. Implementation is separately committed as `8922754`. This feature remains opt-in; preparation defaults and the Gibbs model are unchanged. The original preparation-only NAFLD failure remains relevant when centered variance is disabled.
