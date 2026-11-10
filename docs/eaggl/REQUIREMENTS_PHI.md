# Requirements-based phi selection

Fresh recommended gene-by-gene discovery uses this selector by default. No algorithm flags are required. To request its audit outputs, add:

```bash
--learn-phi-report-out results/phi_selection.tsv.gz \
--numerical-diagnostics-out results/numerical_diagnostics.json.gz
```

Use `--no-learn-phi` to retain fixed phi 0.01. An explicit `--phi VALUE` also fixes that value unless `--learn-phi` is explicitly enabled. CLI flags override configuration values; `"learn_phi": false` is the configuration opt-out. Explicit custom recipes outside the supported search scope require this opt-out or an explicit older selector, rather than silently running an unqualified search.

Create the output directory first. This selector starts at phi 0.01, then fits 0.005 and 0.02, with five restarts each and the same child seeds across candidates. Inputs load once. The chosen complete fit is reused, including annotation projections; selection never adds a separate final refit. Bounded extension can add at most two further candidate fits by default.

## What it tries to preserve

The older composite selector combines metrics using weights. This selector checks explicit requirements before choosing a representative acceptable resolution. Local gene-pair evidence cannot be traded away merely for a better aggregate score. Restart-supported weak profiles enter the comparison even when their contrast with the dominant signal is small. Cross-phi persistence is corroborating evidence, not a requirement that would automatically reject fine-only programs. Redundancy is checked separately. A separate error allowance still limits sensitivity to arbitrarily small effects.

The frozen rule uses support from at least three of five restarts at cosine 0.9, a redundancy threshold of 0.2, and regional/global error-excess tolerance of 0.2. Among acceptable candidates it chooses the one closest to 0.01 in log space, breaking a tie toward lower phi. Extending a flat acceptable range therefore does not move the selected resolution. These are empirically evaluated settings, not a calibrated probability of biological correctness.

## Bounded extension and runtime

The default `--phi-search-extension improvement` extends only from an admissible edge that materially improves evidence preservation over every interior candidate. Specifically, each interior must be inadmissible and its direct local or global normalized error must exceed the edge's by more than the existing 0.2 tolerance. This is a practical effect-size rule, not a statistical significance test. Numerical failures and redundancy-only rejection do not trigger extension. Every assessment includes all tested candidates and their supported evidence; nothing is discarded to make an edge pass.

An eligible lower edge is halved; an eligible upper edge is doubled. If both are eligible, the edge closer to 0.01 runs first, then lower phi breaks a tie. Reassess after each fit. A rejected edge stops that direction; a new acceptable edge must show a further meaningful gain over the best interior alternative to continue. Flat or marginal gains stop without extra fitting.

| Option | Behavior |
|---|---|
| `--phi-search-extension improvement` | Default; require material evidence improvement |
| `--phi-search-extension noninferior` | Explore acceptable edges even without material gain |
| `--phi-search-extension none` | Exactly the initial three candidates |
| `--phi-search-max-extra 2` | Default total extra-candidate cap across both directions; zero disables extension |

Both extension modes share this cap. Users can explicitly increase it. Normal default work is 15 restart fits and the maximum is 25, with projections for each candidate. This bounds candidate count, not wall time: individual fits can take different amounts of time. A numerically failed extra candidate stops the comparison and follows the unqualified fixed-baseline fallback below; it is never silently ignored. The report includes fit order, edge/interior gain comparisons, actual candidate count and stop reason. A budget-exhausted search may select a qualified tested view but does not establish that exploration beyond the edge is unnecessary. Adequacy remains limited to tested views.

## Scope and diagnostics

This initial interface supports the recommended full gene-by-gene recipe: 30 starting factors per restart, five restarts, scale-aware retention, pooled median consensus with support 0.5 and cosine 0.7, adaptive projected-gradient fitting and verified quadratic annotation projection. The usual numerical iteration limits and tolerances remain in force. Setting the initial `--phi` to anything other than 0.01, changing the starting rank or restart/consensus recipe, or supplying a custom phi grid is rejected. Legacy target-size controls, scout-only execution, inline LLM labeling and legacy scout output tables are unsupported. Composite-specific scoring flags do not configure this method.

`--learn-phi-report-out` writes a per-candidate TSV plus a compressed `.details.json.gz` sidecar containing the complete decision. `--numerical-diagnostics-out` includes the same decision under `phi_selection`, together with per-candidate restart and projection checks. Use the recorded status and `selection_qualified`, not just the selected phi:

- A provisional resolution passes the configured numerical and comparison checks. It is not a unique biological partition or a global-optimality certificate.
- An unresolved comparison uses fixed phi 0.01 and remains explicitly unqualified.
- If any candidate fails numerical qualification, the comparison remains incomplete; a numerically qualified phi 0.01 is the fallback. If that baseline also fails, execution raises an error after writing the requested selection report.

An exception during fitting stops execution; it is not silently converted into a successful reduced-grid comparison. Request the selection report to retain comparison diagnostics.

## Evidence and remaining limits

The frozen research rule was evaluated on archived six-library inputs for CAD, T2D, Asthma, NAFLD, BMI and FEV1/FVC. It selected 0.01 for five traits and 0.005 for NAFLD. The first four traits were also checked with a second optimizer seed; five NAFLD additions/regroupings repeated, while an extra profile remained seed-sensitive. These are checks of reproducibility and evidence preservation, not independent biological truth or qualification on every current PIGEAN input.

Package decision parity is exact on all ten saved trait/seed grids. Full-workflow parity is evaluated separately from the scientific comparison. The research implementation cost about 2.7–2.9 times one fixed-phi workflow on BMI/FEV1-FVC, and about 2.3 times the old automatic selector. Integration reuses input loading and the selected fit but still performs fifteen restart fits.

The limited grid and 30-factor starting capacity remain caveats; a supported final consensus may exceed 30. Neither the absence of flagged duplicates nor configured numerical convergence proves absence of fragmentation. Earlier numerical-stopping failures and biological exceptions remain documented limitations; numerical and comparison failures trigger the explicit fallback policy above.

Plain `--learn-phi` explicitly enables the requirements selector for fresh gene-by-gene discovery. Use `--learn-phi --phi-selection-objective composite` for the previous weighted-score selector or `--learn-phi --phi-selection-objective legacy` for target-size selection. Rectangular, supplied-factor and other modes keep their previous mode-specific defaults. Direct Python state APIs are unchanged.

## Package integration replay

The [qualification receipt](requirements_qualification_20261109.json) records one full NAFLD seed-202 replay: phi 0.005,27 factors,15 numerically converged restart fits and18 converged projection checks. Discovery factors, annotation projections, lambdas, pair target and all four exported result tables exactly match the saved standalone phi-0.005 result. The integrated workflow took120seconds (122seconds including supervision) and0.41GiB peak process-tree RSS. Saved standalone fixed-0.01 and complete three-phi workflows took42 and171seconds respectively; these historical timings are descriptive, not a controlled benchmark. A preliminary eight-second attempt caught and fixed a configuration handoff before fitting. This receipt predates the default-policy change and qualifies implementation parity. The default release adds interface/override checks and a replay with algorithm flags omitted.


## Default-policy qualification and separate follow-up

The original qualification used the three-point grid; the bounded extension amendment retains it as the initial comparison. Wider saved sweeps found actual preservation failures (CAD0.16, T2D0.32, NAFLD0.0025), but did not find lower boundaries for CAD/T2D; they do not define a replacement expanded-grid policy. Choosing the acceptable resolution closest to 0.01 is a reproducible convention, not a uniquely optimal biological resolution.

A saved T2D loading audit found GCK/PDX1 and PPARG/ADIPOQ share a dominant factor across phi, optimizer seeds and retention/consensus controls. Across phi0.0003125–0.32, dominant shares varied by less than one percentage point for each gene. Broad membership therefore remains a separate input/model interpretability investigation, not a defect resolved by phi selection. Evidence preservation and lack of near-duplicates do not establish that each gene has one dominant mechanism.


The original three-candidate [default-command qualification receipt](requirements_default_qualification_20261110.json) records the release replay with all algorithm flags omitted: NAFLD selected 0.005 / 27 factors with exact selected-factor, projection, target and four-table parity against both the prior opt-in integration and standalone reference. All 15 restarts and 18 projection checks qualified; execution took112 seconds and 0.38 GiB peak process-tree RSS. The full EAGGL suite passed 596 tests and 46 subtests; one stale generated-manifest check passed after regeneration and one optional sibling-checkout test was skipped. No unresolved failures remain. Fixed-mode runtime, CLI/config precedence and other-mode defaults are covered.


## Bounded-extension qualification

The [extension receipt](requirements_extension_qualification_20261110.json) records 616 passing EAGGL tests, 50 passing subtests and one optional sibling-checkout skip. Ten saved initial trait/seed grids retain their previous decisions exactly. Saved CAD, T2D, Asthma, BMI and FEV1/FVC comparisons stop after the initial three candidates under the default rule. Saved noninferior CAD/T2D controls reach the two-extra-candidate cap while keeping the selected resolution at 0.01.

Two complete default-command checks passed with exact initial-candidate arrays and selected gene/annotation loadings, targets and four exported tables:

| Trait and optimizer seed | Candidates evaluated | Selected phi / factors | Workflow time | Peak process-tree RSS |
|---|---|---|---|---|
| T2D 202 | 0.01, 0.005, 0.02 | 0.01 / 29 | 334 s | 0.64 GiB |
| NAFLD 101 | 0.01, 0.005, 0.02, 0.0025 | 0.005 / 26 | 182 s | 0.39 GiB |

For NAFLD seed101, 0.0025 remains admissible but improves local normalized error over 0.005 by at most 0.0754, below the 0.2 material-gain threshold. The default stops; noninferior mode would request 0.00125, which was not fitted in this qualification. In the earlier seed202 saved fits, 0.0025 fails preservation instead. Thus both seeds stop after four default candidates and retain 0.005, but the boundary rejection itself does not replicate. This supports bounded default behavior, not a universal lower boundary.

All 35 live restarts and 42 projection checks qualified. An observer filename mismatch after T2D completed was corrected by checking its saved outputs; T2D was not refitted. Both fits used 518 seconds of the shared 1,200-second allocation and stayed below the 4 GiB ceiling. These timings are descriptive; neither the candidate cap nor these archived-input checks guarantees a fixed runtime on other datasets.
