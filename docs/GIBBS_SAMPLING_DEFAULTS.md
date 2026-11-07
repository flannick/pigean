# GWAS Gibbs sampling defaults

For `pigean gibbs` with `--gwas-in` or `--huge-statistics-in`, the defaults use sampled inner betas for outer feedback and discard 250 updates before retaining 1,000. This gives more reproducible estimates in the tested T2D, CAD and Asthma workflows, at roughly twice the whole-command runtime. It is not a guarantee of convergence: NAFLD remains poorly mixed.

The spike-and-slab prior, marginal regression, standard-error corrections, local HuGE refresh, numerical diagnostics and arithmetic engine are unchanged. Gene-list inputs and other commands retain their previous automatic defaults. Calling the Python inference functions directly also retains their existing defaults; this change is in CLI configuration selection.

Grouped preparation and stable inner variance are separate defaults for these genetic-input commands. Restore their previous behavior with `--preparation-engine legacy --gibbs-disable inner-centered-variance`; this does not change feedback or the iteration schedule. The evidence below concerns the earlier feedback/schedule change; [variance qualification](INNER_CENTERED_VARIANCE.md) records its subsequent independent effects.

## Independent controls and compatibility

| Choice | Flags to add to an existing command |
| --- | --- |
| Qualified combined procedure, explicitly | `--use-sampled-betas-in-gibbs --gibbs-schedule retained` |
| Previous mean feedback and previous budget/controller | `--use-mean-betas-in-gibbs --gibbs-schedule legacy` |
| Sampled feedback with previous budget/controller | `--use-sampled-betas-in-gibbs --gibbs-schedule legacy` |
| Mean feedback with the qualified longer schedule | `--use-mean-betas-in-gibbs --gibbs-schedule retained` |

`--gibbs-schedule auto` is the default. It selects `retained` only for genetic-input Gibbs without explicit budget/controller overrides. An explicit burn-in, post-burn, iteration cap, rerun/restart or stall-window setting instead preserves legacy/custom budget resolution. For example, `--max-num-iter 20` still requests a short run; it does not inherit a 1,000-retained minimum. Feedback selection remains independent, so add the mean flag as well to reproduce old feedback.

`retained` is a fixed schedule. Conflicting explicit controller settings cause a clear error; use `legacy` for a custom schedule. Matching explicit bounds are allowed. Precision/diagnostic thresholds remain configurable but cannot shorten this schedule below its minimum retained count. `retained` requires Gibbs mode; it can be selected explicitly for other Gibbs input types, which were not part of the four-trait qualification.

CLI values take precedence over configuration files. Explicit `use_sampled_betas_in_gibbs: false` is respected; if both feedback flags occur, the last wins. `--print-effective-config` exposes the requested `gibbs_schedule`, resolved `gibbs_schedule_resolved`, feedback and numeric bounds. `legacy` means previous resolution rules, including any explicit custom settings, not that all settings are reset.

The existing native controller uses an inclusive burn boundary. The fixed schedule expands to:

```sh
--min-num-burn-in 251 --max-num-burn-in 251 \
--min-num-post-burn-in 1000 --max-num-post-burn-in 1000 \
--max-num-iter 1251 --total-num-iter-gibbs 1251 \
--max-num-restarts 0 --gibbs-reruns 1 --disable-stall-detection
```

For a normal completed run, actual updates are 1–1250 and retained updates are 251–1250. Reaching that budget does not mean diagnostics passed. The controller implementation and stopping thresholds have not changed.

The independently controlled arithmetic optimizations are described in [Gibbs optimizations](GIBBS_OPTIMIZATIONS.md). Their defaults are not changed here. To reproduce the qualified optimized procedure, add:

```sh
--use-sampled-betas-in-gibbs --gibbs-schedule retained \
--gibbs-engine parallel-prefilter \
--gibbs-enable inner-matmul,inner-fused,inner-batching,compiled-prefilter,inner-cache64,huge-prior-softmax,static-correlation,compiled-moments
```

Remove conflicting old iteration flags when selecting `retained`.

## Qualification and limits

Qualification compared two independent RNG histories (seeds 12 and 13) for each of T2D, CAD, Asthma and NAFLD, using six annotation libraries: mouse 2024, MSigDB without high-power sets, MeSH, human OCR, STRING without text at medium confidence, and sparse small PoPS. Main was pinned at `8d3f5bc`; the qualification checkout `6e5611c` adds an explicit mean-feedback selector. Within each history, the prepared native state and RNG matched between arms. Ten outer slots within a history are coupled; they are not ten independent initialized runs.

The primary diagnostic panel was fixed from the initial top 200 genetic-evidence genes plus controls (203 T2D, 201 CAD, 200 Asthma, 203 NAFLD). Rank-normalized split/folded R-hat was computed on stride-10 traces over each arm's actual retained window. Green is at most 1.01, yellow at most 1.10, red above 1.10. These research readouts do not replace the production diagnostic implementation.

| Trait | Main red %, seeds 12 / 13 | New red %, seeds 12 / 13 | Between-history informative-prior RMSE, Main → new | Whole CLI minutes, seed12 Main → new | New peak whole-process GB, seed12 |
| --- | ---: | ---: | ---: | ---: | ---: |
| T2D | 66.0 / 77.8 | 3.0 / 0.0 | 0.184 → 0.101 | 7.17 → 15.06 | 4.70 |
| CAD | 81.6 / 55.7 | 0.5 / 0.0 | 0.182 → 0.069 | 8.67 → 19.13 | 5.21 |
| Asthma | 66.5 / 75.5 | 0.5 / 0.0 | 0.235 → 0.090 | 7.50 → 16.16 | 6.40 |
| NAFLD | 100 / 100 | 72.4 / 88.7 | 0.448 → 0.176 | 4.55 → 6.82 | 3.81 |

Main's retained windows have only 11 saved observations versus 100 under the new schedule. Most improved genes are yellow, not green. Equal final-250-update windows do not consistently show faster mixing. Thus the table supports the combined estimation procedure; it does not attribute every gain to faster transitions or prove that longer retention alone causes them. Conditional-mean-prior diagnostics also improve for the first three traits, so the result is not solely extra sampled-beta noise. A long mean-feedback attribution arm was not run.

High correlations do not establish absolute agreement. Between independent new histories, informative-prior correlations are 0.9980, 0.9988, 0.9983 and 0.9936 respectively; slopes are 1.015, 1.005, 0.997 and 0.830. NAFLD's slope and remaining red fraction preclude a convergence or stable absolute-scale claim.

| Quantity, seed12 Main versus new | T2D correlation / slope | CAD correlation / slope | Asthma correlation / slope | NAFLD correlation / slope |
| --- | ---: | ---: | ---: | ---: |
| Gene prior | 0.983 / 0.956 | 0.976 / 0.963 | 0.985 / 0.955 | 0.698 / 1.655 |
| Annotation beta | 0.875 / 0.730 | 0.966 / 0.860 | 0.978 / 0.849 | 0.242 / 0.182 |
| `beta_uncorrected` | 0.994 / 0.982 | 0.988 / 0.966 | 0.980 / 0.969 | 0.865 / 2.998 |
| Genetic `log_bf` | 0.998 / 0.992 | 0.998 / 0.993 | 0.999 / 0.989 | 0.981 / 1.005 |

The native biological-support review followed fixed old-axis annotation loadings, marginal and inner-loop evidence, final corrected/uncorrected betas, gene evidence and prespecified controls. Of 84 within-history axes, one Asthma axis had a corrected-beta support ratio below the 0.75 review band. Its apparent loss was driven by one annotation with Main beta 0.619/1.520 versus new 0.632/0.665 across histories; initial marginal evidence was unchanged. Common finite-loading coverage was 82.94%, explicitly excluding missing values. This was not a replicated native-support loss. No axis had an uncorrected-beta ratio below 0.75. These are review bands, not formal non-inferiority margins.

Core positive controls retain support, but individual weak priors can remain unstable. For example, T2D CMIP prior is 0.412/0.431 in Main versus 0.314/−0.0135 in the new histories; its new genetic log-BF remains 3.59/3.56 and combined-score rank 111/134. NAFLD retains lipid support but remains poorly mixed. EAGGL axis representation changes require separate review and are not altered by this PR. None of these checks independently establishes probability calibration or universal biological non-inferiority.

Whole-process memory includes preparation and qualification observation; it is not comparable to earlier Gibbs-only replay measurements below 2 GB. This change does not enable full trace saving by default. Production accumulation remains streaming; running longer costs time without retaining a dense history of all samples.

Analysis provenance (separate analysis workspace): `results/long_eaggl_20261105/comparison/`, `results/independent_axes_20261105/comparison/`, and `results/pigean_default_finish_20261105/`. These include run commands, compressed tables, native retained-count receipts and the detailed support review. Release tests cover scope/precedence/compatibility plus a compact real-GWAS native run that checks actual retained counts and fixed-seed equality to the explicit qualified recipe.
