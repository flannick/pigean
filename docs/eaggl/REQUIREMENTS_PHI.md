# Experimental requirements-based phi selection

The recommended default remains fixed phi 0.01. To try the frozen requirements selector, add these flags to the recommended gene-by-gene factor command:

```bash
--learn-phi --phi-selection-objective requirements \
--learn-phi-report-out results/phi_selection.tsv.gz \
--numerical-diagnostics-out results/numerical_diagnostics.json.gz
```

Create the output directory first. This selector fits phi 0.005, 0.01 and 0.02 with five restarts each, using the same child seeds across candidates. Inputs load once. The chosen complete fit is reused, including annotation projections; there is no fourth fit.

## What it tries to preserve

The older composite selector combines metrics using weights. This experimental alternative checks explicit requirements before choosing a representative acceptable resolution. Local gene-pair evidence cannot be traded away merely for a better aggregate score. Restart-supported weak profiles enter the comparison even when their contrast with the dominant signal is small. Cross-phi persistence is corroborating evidence, not a requirement that would automatically reject fine-only programs. Redundancy is checked separately. A separate error allowance still limits sensitivity to arbitrarily small effects.

The frozen rule uses support from at least three of five restarts at cosine 0.9, a redundancy threshold of 0.2, and regional/global error-excess tolerance of 0.2. Among acceptable candidates it chooses the middle log-phi resolution, breaking a tie toward 0.01. These are research settings, not a calibrated probability of biological correctness.

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

The limited grid and 30-factor starting capacity remain caveats; a supported final consensus may exceed 30. Neither the absence of flagged duplicates nor configured numerical convergence proves absence of fragmentation. Earlier numerical-stopping failures and biological exceptions remain reasons not to default the selector without further review.

Plain `--learn-phi` retains the composite selector. `--phi-selection-objective legacy` retains the historical target-size alternative. Neither automatic path changes merely by installing this opt-in feature.

## Package integration replay

The [qualification receipt](requirements_qualification_20261109.json) records one full NAFLD seed-202 replay: phi 0.005,27 factors,15 numerically converged restart fits and18 converged projection checks. Discovery factors, annotation projections, lambdas, pair target and all four exported result tables exactly match the saved standalone phi-0.005 result. The integrated workflow took120seconds (122seconds including supervision) and0.41GiB peak process-tree RSS. Saved standalone fixed-0.01 and complete three-phi workflows took42 and171seconds respectively; these historical timings are descriptive, not a controlled benchmark. A preliminary eight-second attempt caught and fixed a configuration handoff before fitting. This qualifies implementation parity, not a new default.
