# Correlation-SE warmup qualification

This supports an **opt-in initialization option**, not a default change or a claim of global convergence. The tested code is `0b37945`, based on main `7a29be7` (2026-11-11 review).

## Tested behavior

`--gibbs-se-warmup-iterations 125` bypasses the correlation-related SE adjustment during outer updates 1–125. Updates 126–250 restore the full correction and are discarded; updates 251–1250 retain ordinary native inference. The same schedule was used for all four traits, with six annotation libraries, ten outer chains, four inner coefficient chains and sampled coefficient feedback. Original regressions, annotation corrections, SNP signal probabilities, HuGE refresh, spike-and-slab assumptions and retained equations are unchanged.

T2D and NAFLD have two independent RNG histories each. Full CLI replay of seed 12 exactly matches the prepared-state experiment for priors, beta, uncorrected beta, sampled beta and log BF. CAD and Asthma have one matched transfer history each. Every actual retained regression audit reports zero warmup bypasses. Initial data/RNG identities and original regression fields match.

## Equal-window results

These are rank-normalized/folded split **gene-prior** R-hat q90 values on fixed review panels, not the older beta R-hat statistic and not an all-parameter census. Target-trait panels use a frozen review union; transfer panels use the initial top 200 plus controls. Both arms use 1,000 retained updates.

| Trait | Ordinary | Warmup | Native prior correlation, seed 12 | Native prior slope, seed 12 |
| --- | --- | --- | --- | --- |
| T2D, seeds 12 / 14 | 1.060 / 1.047 | 1.051 / 1.042 | 0.9973 | 1.0080 |
| NAFLD, seeds 12 / 14 | 1.249 / 1.301 | 1.064 / 1.042 | 0.9917 | 1.0980 |
| CAD, seed 12 | 1.037 | 1.034 | 0.9970 | 0.9917 |
| Asthma, seed 12 | 1.039 | 1.035 | 0.9976 | 0.9918 |

NAFLD seed 14 has prior correlation 0.9885 and slope 1.1679. Larger evidence alone is not proof of greater accuracy. Independent-history prior correlation improves from 0.9838 to 0.9976 for NAFLD and from 0.9963 to 0.9985 for T2D. Late-window diagnostics also improve, but remain worse than full-window values.

Four additional fully corrected NAFLD starts retain initialization dependence: pooled prior q90 is approximately 1.110 over the full retained window, 1.202 over the late 500 and 1.259 over the final 250. These fresh starts use per-slot mean coefficient configurations, not exact checkpoint continuations. Warmup may select similar basins without improving travel between every basin; it does not solve mixing.

## Biology and downstream factors

Prespecified familiar gene controls retain their main support, including NAFLD PNPLA3/TM6SF2/MTTP/APOB, T2D GCK/PDX1/PPARG/ADIPOQ, CAD LDLR/APOB/PCSK9/LPA/SORT1 and Asthma STAT6/IL4/IL13/GATA3/TSLP/IL33. Small adverse changes remain: MBOAT7 prior 0.0633 to 0.0470 (combined score unchanged), T2D AUTS2 0.135 to 0.0891, and Asthma IL13 4.08 to 3.90. Familiar controls are not an independent accuracy benchmark.

NAFLD annotation weights redistribute and become more concentrated: beta > 0.01 counts 95 to 83; top-ten positive beta mass 41.8% to 49.6%. A kidney-labelled set containing liver/lipid genes and the constitutive-androstane-receptor pathway lose corrected beta. Nevertheless, the main lipoprotein and hepatic drug-metabolism factors remain represented. CAD MIR-target annotation weight increases; no validated negative panel was available, so artifact sentinels and input-defined backgrounds cannot establish a false-positive rate. T2D/NAFLD full-universe review finds no MIR genes in either top 200. One small negative CAD coefficient appears (-0.0129); there is no broad growth of large negative effects.

Unchanged current-main EAGGL, with five restarts and automatic phi, yields ordinary/warmup counts T2D 30/29, CAD 29/30, Asthma 27/25 and NAFLD 30/22. NAFLD chooses different phi values (0.005/0.01); at fixed 0.01 the comparison is 24/22. All twelve completed qualification fits, including targeted attribution fits, pass numerical restart/projection checks with no near-duplicate factor pairs at cosine >= 0.95. Counts and numerical checks alone do not establish biological preservation.

Smaller groups can change. CAD MMP14 and BMP6 cross EAGGL's existing prior > 0.5 input cutoff and their group weakens; the principal lipid, heart-development, endothelial and MMP2/MMP9 groups remain. A NAFLD GGT1/CXCL14 group was already unstable across ordinary histories. An apparent Asthma IL17/IL23-group loss reverses with an independent EAGGL seed: the original group matches fresh ordinary factors at cosine 0.412 and fresh warmup factors at 0.983, while native annotation support is preserved. These findings support core biological preservation, **not universal non-inferiority of every factor**.

## Cost and engineering checks

Paired CAD Gibbs time is 979.5/991.1 seconds; Asthma is 724.9/721.8 seconds. Prepared-state T2D/NAFLD runs improve sampling time by approximately 3–17%, but T2D prepared-state peak RSS increases 14–24% across the two histories. Full CLI peak process-tree RSS is 4.43 GB including startup. This is not a universal speed or memory optimization.

Fourteen focused tests cover actual update boundaries, restored contexts after exceptions/nesting, validation, shortened epochs and CLI/config precedence. Related Gibbs, sampling-default and CLI-manifest checks pass. A 1,250-update compact native default-parity test passes, and the full native replays verify production integration. Zero/off preserves ordinary sampling outputs; parameters additionally record the requested and last-epoch effective warmup lengths.

Study records: `native_warmup_qualification_20261110`, `native_warmup_starts_20261110`, and `native_warmup_release_20261110` in the analysis workspace. They contain commands, input/source hashes, compressed traces/tables, all-factor comparisons, figures and the qualification review package. The initial EAGGL batch hit an agent-selected time bound; the partial Asthma fit was preserved and excluded, and only unfinished work was resumed. No scientific settings were changed to obtain completion.

Leave the option off for unchanged behavior. Its strongest current evidence is as an optional NAFLD initialization aid. Any default-promotion proposal needs a separate decision about residual initialization dependence and weaker-factor changes.
