# Recommended EAGGL workflow and defaults

Fresh `python -m eaggl factor` discovery uses the workflow below without extra algorithm flags. Choose inputs, output paths and a seed for reproducibility. The recommendation is requirements-based automatic selection starting at phi 0.005, 0.01 and 0.02, extending only for meaningful evidence improvement with at most two extra candidates; the selected fit is reused.

```bash
PYTHONPATH=src python -m eaggl factor \
  --X-in annotations.gmt.gz \
  --gene-stats-in pigean.gene_stats.out.gz \
  --gene-set-stats-in pigean.gene_set_stats.out.gz \
  --seed 101 \
  --factors-out results/factors.out.gz \
  --gene-clusters-out results/gene_clusters.out.gz \
  --gene-set-clusters-out results/gene_set_clusters.out.gz \
  --params-out results/params.out.gz
```

The output directory must exist. `--eaggl-bundle-in` can replace the three input arguments. `--print-effective-config` prints the resolved choices without fitting.

| Choice | Recommended CLI default | Independent alternative |
|---|---|---|
| Discovery target | `--discovery-model gene_by_gene`, corrected beta pair evidence | `--discovery-model gene_by_annotation` restores rectangular-mode defaults |
| Sparsity parameter | `--learn-phi --phi-selection-objective requirements` (implicit) | `--no-learn-phi` fixes 0.01; `--phi VALUE` fixes another value unless learning is explicitly enabled |
| Numerical fit | Projected gradient, adaptive stopping, maximum 20,000 updates, relative tolerance 1e-6 | Explicit solver/stopping/iteration flags |
| Retention | `--factor-retention backend_scale`: lambda threshold is the smaller of the requested threshold and the backend's scale-aware cutoff | `--factor-retention legacy` |
| Restarts | `--factor-runs 5` | `--factor-runs 1` also disables implicit consensus |
| Consensus | `--consensus-nmf --consensus-matching pooled --consensus-aggregation median` | `--consensus-matching reference`; or `--no-consensus-nmf` for best-of-N |
| Support | All pairwise gene-loading cosines at least 0.7; support from at least half the restarts (three of five) | Explicit cosine/support settings |
| Factor budget | 30 starting factors per restart; final supported pool may exceed 30 | `--max-num-factors` changes the starting budget |
| Annotation projection | `--fixed-basis-projection-method verified_quadratic`, 2,000 cycles, stationarity tolerance 1e-8 | `--fixed-basis-projection-method legacy` |
| Output and ranking | All retained consensus factors, row threshold 0, top entities by raw loading | `--factor-output-scope primary`, row threshold and ranking flags |
| Dashboard gene display | Discovery gene loadings | Explicit full-gene source selector |

Pooling uses deterministic greedy complete-link grouping over all restarts, with at most one factor from each restart in a group. It does not optimize a global clustering objective or reoptimize consensus loadings. If no group meets support, the existing reference-run fallback is retained and recorded in consensus diagnostics; that is not a successful supported consensus.

Explicit CLI or configuration values override defaults. The requirements search supports the qualified recipe below; custom restart, rank, retention, consensus or solver recipes require `--no-learn-phi` or an explicit older selector. Rectangular discovery, `naive_factor`, labeling and supplied-factor projection retain their previous mode-specific defaults. Direct Python state APIs retain their prior defaults. This recommendation does not silently change those APIs.

## Evidence and limitations

Qualification compared current versus proposed retention/consensus at fixed phi 0.01 and automatic phi, across T2D, CAD, Asthma and NAFLD, with two independent master seeds and five final restarts each (32 configurations). Inputs were matched six-library PIGEAN outputs. Fixed-phi factor counts were:

| Trait | Previous workflow, seeds 101 / 202 | Proposed workflow, seeds 101 / 202 |
|---|---:|---:|
| T2D | 20 / 23 | 30 / 29 |
| CAD | 24 / 23 | 31 / 31 |
| Asthma | 17 / 15 | 26 / 26 |
| NAFLD | 22 / 22 | 22 / 22 |

The fixed-phi comparisons preserved all reproducible reference profiles under the study's matching gates and recovered additional structure; some weak additions remained seed-sensitive. No within-output near-duplicates reached cosine 0.95. These checks support this configuration, not universal superiority or a claim that every extra factor is a separate biological mechanism.

The previous composite automatic selector had a known exception: one T2D endocrine profile retained by reference-centered consensus was rejected by complete-link support in one seed. The raw restart fits were identical; a third hormone-related fit met the reference match but not every pairwise match. Automatic search also selected its upper boundary in 14 of 16 configurations. The replacement requirements selector explicitly checks local evidence preservation before selecting; see its separate qualification below.

Discovery gene loadings plus annotation projections are the qualified display route. Optional full-gene projections through annotations remain available but have not passed the same biological-preservation qualification. Multi-anchor runs, annotation exclusions and larger starting-factor budgets need their own biological review.

Numerical stopping and projection diagnostics do not establish biological validity. Request `--numerical-diagnostics-out` and `--consensus-stats-out` for an audit; reaching an iteration cap or using a fallback requires review.

## Implementation validation

[Qualification receipt](defaults_qualification_20261109.json) records the frozen control and output hashes. Production pooling reproduces group membership, ordering and minimum pair cosine in all 12 saved stage-capture configurations. One full CAD seed-202 run using minimal defaults reproduces all four qualified output tables exactly after decompression: 31 factors, five numerically converged restarts and six converged projections, 118 seconds and 0.54 GiB peak process-group RSS. CLI parity, explicit overrides, alternate modes and dashboard/launcher tests cover the release interface.

A bare discovery command normally performs fifteen restart fits (five at each of three phi values); bounded improvement-triggered extension permits at most ten additional restart fits by default. Inputs load once and the selected fit is reused. The prior five-restart fixed-phi workflow is available with `--no-learn-phi`. The integrated NAFLD comparison measured about122seconds/0.41GiB; cost varies with inputs.

## Reproduce previous choices

To restore fixed phi and the earlier retention/reference-consensus choices:

```bash
--no-learn-phi --factor-retention legacy --consensus-matching reference
```

To reproduce the previous raw CLI gene-by-gene defaults:

```bash
--discovery-model gene_by_gene --phi 0.01 \
--no-learn-phi --factor-retention legacy --consensus-matching reference \
--factor-runs 1 --no-consensus-nmf \
--fixed-basis-projection-method legacy \
--factor-output-scope primary --cluster-row-min-max-loading 0.01 \
--factor-top-loading-type euclidean
```

For the previous default rectangular model, select `--discovery-model gene_by_annotation`; omitted mode-specific values resolve to the previous single-run configuration and phi 0.05.

Automatic composite selection is available with `--learn-phi --phi-selection-objective composite`. Do not add `--learn-phi-target-gene-effective-support` merely to enable it: that legacy control can also alter the pair prior unless the latter is explicit. Inputs and scientific settings should be held fixed when comparing selection policies.

The default [requirements-based selector](REQUIREMENTS_PHI.md) compares three complete candidate resolutions using explicit evidence-preservation checks. On archived six-library CAD, T2D, Asthma, NAFLD, BMI and FEV1/FVC inputs it selected0.01 except NAFLD0.005. Saved decisions match exactly across ten trait/seed grids; one integrated NAFLD replay exactly matches selected native factors, projections and exported tables. These empirical checks support the bounded policy, not a claim of biological superiority or a unique optimal phi. Unresolved comparisons explicitly fall back to numerically qualified0.01; a failed baseline raises an error. Request selection/numerical reports to inspect this status.

Broad T2D gene-loading overlap persists across phi and remains a separate investigation. The default change does not fix that overlap or establish exclusive gene memberships. The extension policy and its shared budget are described in the selector documentation. Arbitrary starting-factor budgets remain outside the qualified selection interface.
