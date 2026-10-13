# 2026-10-11 T2D and NAFLD convergence and runtime research

## Source and implementation audit

Read the August 29 NAFLD sampler pathology log, retained study README, exact default command, runtime and cross-trait tables, September 5 corrected validation log, and September 12 T2D record. The August study is a distinct configuration from September validation and the 2025 GWAS T2D run. Its T2D ordinary control has R-hat q90 2.899, not 1.655 or 2.140. No new inference was run.

Inspected the clean local `pigean/` checkout at `388dfc7`, especially `src/pigean/state.py`, `src/pigean/model.py`, `src/pigean/gibbs.py`, `src/pigean/gibbs_callbacks.py`, `src/pegs_shared/regression.py`, and `src/pigean/cli.py`. This is a local code identity, not a claim to have fetched the latest upstream release. No production source was edited.

Verified operations:

- Gene posterior probabilities combine gene prior log odds, background log odds, and log Bayes factors, followed by clipping. The ordinary outer update draws Bernoulli gene indicators.
- Each iteration computes marginal logistic effects/SEs, applies corrections, screens gene sets, and invokes Gaussian spike-and-slab coefficient updates with residualization through gene-set correlations.
- The default feeds mean inner coefficients back into gene priors. The CLI explicitly documents this as smoother but more prone to incomplete exploration. Retained NAFLD params confirm `use_mean_betas=True`.
- HuGE evidence is refreshed using prior-dependent locus allocation, and prior adjustment and correlation/SE corrections are additional operations. A textbook logistic likelihood is therefore not established as an exact joint target for all production updates.
- The Bernoulli draw is the first observed chain-specific divergence; this does not prove that it alone causes persistent trapping.

## Retained timing extraction

Created and ran `scripts/summarize_convergence_runtime_20261011.py` using `../.venv/bin/python`. It reads three retained compressed logs, checks that every timer has 500 unique iterations, and records source SHA-256 hashes. Outputs: `results/convergence_runtime_research_20261011/retained_runtime.tsv` and `provenance.json`.

| Run | Outer iterations, seconds | Input preparation | Corrected coefficients and priors |
|---|---:|---:|---:|
| NAFLD default seed 0 | 495.28 | 77.61% | 17.41% |
| NAFLD forced 500 seed 0 | 431.13 | 78.48% | 16.61% |
| T2D forced 500 seed 0 | 1371.06 | 66.13% | 32.92% |

Timers are rounded to 0.01 seconds and nested; these columns are coarse disjoint stages, not isolated operation profiles. Input preparation includes repeated logistic fitting, prefiltering, response construction, and other work. Corrected coefficients and priors include inner sampling, prior construction, and HuGE refresh. Do not attribute the entire first stage to logistic fitting or the entire second to inner Gibbs.

Even eliminating the entire corrected-coefficients/prior stage gives only about 1.2-fold NAFLD and 1.5-fold T2D outer-loop acceleration. End-to-end gains are smaller. NAFLD default wall time 592.73 seconds minus 495.28 seconds in outer iterations leaves about 97 seconds outside the loop, already greater than the 59-second tenfold target. The matched forced-500 rerun has a similar floor. Any tenfold end-to-end proposal needs setup reuse or optimization as well as inference changes. These are deductions from historical timings, not new speed benchmarks.

## Literature review and model implications

Primary sources consulted:

1. Carbonetto and Stephens (2012), [Scalable variational inference for Bayesian variable selection](https://pcarbo.github.io/bvs.pdf), and the authors' [varbvs documentation](https://pcarbo.github.io/varbvs/varbvs-R/docs/). Logistic spike-and-slab variational inference is established; it is not precluded by either nonconjugacy or the point mass at zero.
2. Polson, Scott, and Windle (2013), [Pólya–Gamma augmentation](https://arxiv.org/abs/1205.0310). Conditional Gaussian structure removes the logistic nonconjugacy obstacle but does not itself resolve support multimodality.
3. Zanella and Roberts (2019), [Scalable importance tempering](https://arxiv.org/abs/1805.00541). Strong statistical-efficiency gains in variable-selection examples motivate tempering; they are not PIGEAN runtime measurements.
4. Jankowiak, [Fast Bayesian variable selection in binomial and negative binomial regression](https://arxiv.org/html/2106.14981v2) and [Bayesian variable selection in a million dimensions](https://arxiv.org/abs/2208.01180). Combines Pólya–Gamma augmentation, coefficient marginalization, and weighted tempered support sampling; the first paper directly tests mutually substitutable correlated predictors. Importance weights and the augmented target matter: this is not an arbitrary heating of the current update loop.
5. Authors' [gSuSiE documentation](https://yizenglistat.github.io/glmsusie/index.html) and [SuSiE paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC10201948/). A structured single-effect representation is attractive for uncertainty among overlapping annotations but changes the coefficient prior, and published logistic software does not directly accept the whole PIGEAN hierarchy.
6. Yao, Vehtari, and Gelman, [Stacking for non-mixing Bayesian computations](https://arxiv.org/abs/2006.12335). Combining modes for prediction can be useful; predictive stacking weights are not posterior mode probabilities.
7. Vehtari et al., [Rank-normalization, folding, and localization](https://arxiv.org/abs/1903.08008). Use modern split rank-normalized diagnostics, effective sample sizes, and fixed scientific estimands, rather than only the legacy active-panel q90.

Under an explicitly simplified model with fixed independent gene evidence, let B_g be the gene Bayes factor and eta_g the background-plus-annotation prior log odds. Summing the two states of each gene indicator yields an observed-data likelihood proportional to the product of `(1 + exp(eta_g) B_g) / (1 + exp(eta_g))`. Its log is a sum of `softplus(eta_g + log(B_g)) - softplus(eta_g)`. Its derivative with respect to eta is `sigmoid(eta + log(B)) - sigmoid(eta)`. This identity is an algebraic derivation, not a claim that the full production pipeline has this likelihood. It removes explicit gene-indicator sampling, but the likelihood need not be concave and correlated coefficient supports still compete.

The fixed-evidence derivation cannot be applied by simply substituting refreshed prior-dependent Bayes factors. Locus competition must be incorporated into a joint likelihood, or freezing it must be labeled a model simplification. Likewise regression/QC corrections and gene dependence need an explicit treatment. Existing expected-D/Gauss–Seidel behavior is not a derivation of a joint ELBO or a correct collapsed posterior sampler.

Preferred research branches:

- Speed branch: derive joint latent-indicator/coefficient variational inference (or a collapsed-evidence objective), begin with a mean-field diagnostic baseline, then retain alternatives through structured blocks or mixture components. A multistart MAP plus within-mode Laplace approximation is a useful fast control, not sufficient uncertainty accounting. No tenfold PIGEAN speedup is established.
- Mixing branch: define the target, then compare valid support swaps/blocks and Pólya–Gamma/weighted tempered sampling. Standard tempering or SMC can provide a reference at additional cost. Tempering may reduce time to reliable estimates while increasing time per iteration.
- Architecture branch: carry persistent coefficient and support state through one joint chain. One invariant conditional sweep need not fully converge internally on each outer step, but this principle does not justify merely shortening the current refit-and-average inner procedure.
- Model branch: gSuSiE-style single effects can express uncertainty between interchangeable gene sets. Treat it as a prior change and check gene-level calibration and annotation-family behavior.

Mean-field optimization may be reproducible while confidently missing modes. Multiple optimizer starts alone do not supply valid posterior mode weights. The 32/40 NAFLD chain count is an observed initialization/occupancy outcome, not verified 80% posterior mass.

## Proposed validation boundaries

Pin code, evidence, annotation universe, corrections, and hyperparameters within each trait; use the same inference settings across T2D and NAFLD while retaining their trait-specific data. Distinguish a frozen-evidence research fixture from the full dynamic HuGE pipeline. First assess whether mode differences change gene-level probabilities, annotation-family contribution, or just interchangeable coefficient identities.

Compare candidates at matched wall time and matched error, counting all optimizer starts, temperature replicas, and setup. Report both cached inference and cold end-to-end time. Validate against simulated and reduced exactly enumerable/numerically integrable targets; the long NAFLD run is a stress reference, not posterior truth. Measure gene score calibration/scale, set and family inclusion, uncertainty coverage, support transitions, repeated-run stability, and ESS per second. Use held-out loci or chromosomes for predictive checks to reduce dependence leakage.

For posterior sampling, propose rank-normalized split R-hat near 1.01 on prespecified important continuous estimands, explicit diagnostics for sparse indicators and modes, and adequate ESS/MCSE. Do not treat a deterministic approximation as converged MCMC. For approximations require calibration and cross-start stability, with disagreements reported rather than averaged using arbitrary weights.

Work completed: source audit, literature review, retained timing extraction, and experimental design. No new sampler, approximation, or tenfold runtime claim has been validated.
