# Optional stable inner variance

`--gibbs-enable inner-centered-variance` replaces cancellation-prone variance accumulation for active inner Gibbs convergence diagnostics with anchored Welford moments. It is off by default and independent of preparation arithmetic. Omit it or use `--gibbs-disable inner-centered-variance` for the existing calculation. The option applies within the outer Gibbs stage; initial preparation fitting retains its existing feature scope. Python callers may explicitly enable the feature through the existing backend context.

The model, corrections, draws and 1.01 stopping threshold are unchanged. More accurate diagnostics can nevertheless change stopping, outlier decisions and subsequent random draws. This is a numerical correction, not a claim of improved statistical mixing or identical outputs to the original arithmetic. The existing `inner-rhat-stable` feature concerns membership near R-hat one and remains separately selectable.

The accumulator subtracts each chain's first observation, updates its mean deviation and centered sum of squares online, and computes between-chain variance before reconstructing absolute means. Only active slots accumulate observations. Replacement chains inherit the accumulator history of their source. Inactive-slot handling, short-history defaults and the existing zero-within-variance convention are preserved; outer diagnostic calculations are outside this change.

Persistent overhead is three float64 arrays of inner chain × parallel slot × annotation shape, plus slot counts. For four inner chains, ten outer slots and 5,000 annotations, this is about 4.8 MB. Temporary arithmetic arrays add transient allocation. No full draw history is stored. Work scales with inner sweeps and active annotations.

## Qualification

The matched NAFLD preparation comparison exposed a stopping discrepancy at outer update 4, inner sweep 11. Original sum-of-squares arithmetic produced thresholded mean R-hat 1.0069455 versus 1.0137769, causing an extra sweep in grouped preparation. Observing the actual draws without changing the trajectory gives 1.00911734173 versus 1.00911734171 with the centered reference: both stop at the same 1.01 threshold.

For the offending coefficient, within-chain variance is approximately 1.50841004e-17 and raw R-hat 1.03573288. The online implementation agrees with a separate 70-digit Decimal calculation to relative tolerance 1e-12. Native prefixes and complete RNG records reproduce in both observers. Tests cover independent centered references, masked counts, constant/short sequences, outlier ancestry, full inner-fit integration and context isolation. Full-run qualification is recorded separately; do not infer full-output equivalence from this short replay.

Analysis evidence is under `results/inner_variance_20261107/` in the optimization workspace. This feature remains an isolated candidate and is not automatically included in preparation PR #24 or enabled by default.
