# EAGGL Factor Workflows (F1-F4)

This document maps each supported factoring workflow to:

1. required inputs
2. workflow-selection flags
3. a minimal runnable command pattern

All workflows run through `factor` (or `naive_factor`), and the selected workflow ID is visible with `--print-effective-config`.
Optional labeling stays attached to the same factor command; EAGGL does not have a separate `label` mode.
Canonical trait linkage is the primary annotation layer and reports support-normalized projection coefficients from the same internal trait-factor matching step. Raw trait support and raw factor loadings keep their original totals; only copied internal vectors are normalized for matching. Factor-PheWAS is a secondary expert-only enrichment regression.

Optional LLM/provider-based factor labeling is documented separately in `docs/eaggl/LABELING.md`. Workflow selection and factor execution do not require labeling.

Related docs:
- `docs/eaggl/CLI_REFERENCE.md`: curated run manual and main flag guide
- `docs/eaggl/methods.tex`: theory and model formalization
- `docs/eaggl/CLI_OPTIONS.md`: exhaustive generated inventory

## Common Setup

Use the local venv from the analysis workspace:

```bash
PYTHON=../../.venv/bin/python
```

Useful common outputs:

```bash
--factors-out results/factors.out \
--gene-set-clusters-out results/gene_set_clusters.out \
--gene-clusters-out results/gene_clusters.out \
--trait-factor-links-out results/trait_factor_links.out \
--params-out results/params.out
```

Discovery-stage note:

1. EAGGL now distinguishes retained annotations, discovery annotations, and projected annotations.
2. By default, factor discovery is learned on redundancy-balanced discovery families using `--discovery-similarity-threshold 0.35`, and the resulting outputs mark discovery rows with `in_discovery`.
3. Discovery-family diagnostics now include `discovery_family_mean_similarity` and `discovery_family_effective_size` alongside `discovery_family_size` and `discovery_weight`.
4. The default discovery weighting is `effective_size`, which uses representative support multiplied by an effective family size instead of the removed family-average weighting.
5. All retained gene sets are still projected and written after discovery, so adding correlated annotations mostly deepens annotation rather than redefining `W`.
6. `--discovery-model gene_by_annotation` is the default retained-annotation workflow described above.
7. `--discovery-model gene_by_gene` instead builds a retained-gene by retained-gene pairwise matrix from corrected annotation betas, fits a symmetric nonnegative factorization in gene space, and then projects retained gene sets onto the learned gene factors.
8. In `gene_by_gene` mode, all retained gene sets contribute to pairwise evidence; discovery-family leader subsetting and redundancy weighting are ignored.
9. In `gene_by_gene` mode, v1 currently requires `--factor-backend full`, `--learn-phi-backend sentinel_pruned`, and the default transposed factor matrix.
10. In `gene_by_gene` mode, if `--phi` is not explicitly set, EAGGL starts factor learning and any `--learn-phi` search from `0.01` rather than `0.05`.
11. In multi-anchor `gene_by_gene` mode, `--gene-gene-anchor-aggregation multi` is the default weighted multi-view shared-factor objective, and `any` is noisy-OR union evidence.
12. `gene_by_gene` still learns one shared gene-factor basis and projects annotations onto that basis afterward; it does not write anchor-specific cluster outputs.

Debug workflow selection without running factorization:

```bash
$PYTHON -m eaggl factor --print-effective-config [workflow flags ...]
```

## Input Contracts

Core matrix/stat inputs (direct mode):

1. `--X-in` or `--X-list`
2. `--gene-stats-in`
3. `--gene-set-stats-in`

Use `--X-in` for a direct `.gmt` sparse matrix file. Use `--X-list` only for a text file that lists matrix inputs. If a direct `.gmt` is passed to `--X-list`, EAGGL accepts it for compatibility but warns and treats it like `--X-in`.

Consensus cNMF is part of the normal factor workflow surface: add `--factor-runs N --consensus-nmf` to any of the factor workflows below when you want restart aggregation instead of a single fitted run.
Automatic phi tuning is also part of the normal factor workflow surface: add `--learn-phi --learn-phi-target-gene-effective-support S` to any of the factor workflows below when you want EAGGL to search for a `phi` whose primary factors have median effective gene support near the requested size `S`. The selected `phi`, target-size diagnostics, the redundancy basis, and the search thresholds are written to `--params-out`, and `--learn-phi-report-out` writes the full per-candidate diagnostics table. Redundancy is measured on gene loadings when they are available, with fallback to gene-set or phenotype loadings only when gene loadings are absent. By default, phi-selection redundancy and repeat-stability metrics are computed only on primary factors, ignoring the low-mass ARD tail; `--learn-phi-mass-floor-frac` defines that primary-factor mass threshold across the whole phi-search metric surface, and `--learn-phi-metric-factor-scope all` is only for auditing all fitted columns. The search records both the worst nearest-neighbor overlap (`redundancy_max`) and a global tail-overlap summary (`redundancy_q90`), treats capped solutions explicitly, and computes factor-mass diagnostics (`raw_factor_count`, `primary_factor_count`, and `effective_factor_count`) plus convergence diagnostics (`final_delambda`, `final_iterations`, and hit-iteration-cap summaries) for each candidate. Among acceptable candidates, selection minimizes log-scale mismatch to the target median primary-factor gene effective support; candidates within the configured size tolerance prefer the largest `phi`. Ordinary fit loss is warning-level through `--learn-phi-fit-loss-warning-frac`, while `--learn-phi-max-severe-fit-loss-frac` remains a hard guard against pathological underfit and also rejects candidates missing a finite fit error. The default redundancy thresholds are `--learn-phi-max-redundancy 0.5` and `--learn-phi-max-redundancy-q90 0.35`.

EAGGL now exposes two scalable phi-search backends:

1. `--learn-phi-backend sentinel_pruned`
   - the legacy shortcut
   - evaluates candidates on a correlation-pruned sentinel panel of up to `1000` genes and `1000` gene sets by default
2. `--learn-phi-backend blockwise_global_w`
   - the preferred scalable approximation to a full retained-panel search
   - keeps one shared global gene-factor basis and shared ARD state across block-local gene-set solves
   - evaluates all retained gene sets in blocks instead of replacing them with a single decorrelated sentinel panel

Likewise, the final factorization can run with either:

1. `--factor-backend full`
   - the original in-memory solve
2. `--factor-backend blockwise_global_w`
   - a blockwise solve over the full retained gene-set collection with one shared global gene-factor basis

The symmetric `--discovery-model gene_by_gene` mode is separate from those rectangular backends:

1. it uses corrected `beta` values by default (`--gene-gene-beta-source beta`)
2. it converts retained shared-annotation evidence into pairwise probabilities before factoring
3. it currently supports only `--factor-backend full`
4. it ignores discovery-family subsetting and weighting flags because pairwise evidence is built from all retained gene sets
5. with multiple anchor traits, `multi` fits one shared gene-factor basis with a weighted multi-view objective over per-anchor pair targets, while `any` uses noisy-OR over per-anchor target matrices

Use `--blockwise-gene-set-block-size`, `--blockwise-epochs`, `--blockwise-shuffle-blocks`, `--blockwise-warm-start`, `--blockwise-max-blocks`, and `--blockwise-report-out` to tune or audit the blockwise backend. When `--learn-phi-backend blockwise_global_w` is used, neighboring phi candidates are warm-started from the closest previously fitted phi on the log scale when possible. Use `--factor-phi-metrics-out`, `--factor-phi-factors-out`, `--factor-phi-gene-set-clusters-out`, and `--factor-phi-gene-clusters-out` when you want audit tables for every tested phi candidate. Cluster output rows whose maximum raw factor loading is below `--cluster-row-min-max-loading` are omitted from reported cluster files.

Factor output policy:

1. `factors.out`, `gene_set_clusters.out`, and `gene_clusters.out` print only primary factors by default.
2. Primary factors have `combined_mass_fraction >= 0.005`; secondary factors have `combined_mass_fraction >= 0.001`; smaller factors are treated as filtered tail factors.
3. Use `--factor-output-scope primary_secondary` or `--factor-output-scope all` when you need to inspect the secondary or full ARD tail.
4. `factor_metrics.out` and `factor_phi_metrics.out` remain exhaustive diagnostics over all raw fitted factors.
5. Factor identifiers are preserved rather than renumbered after filtering.

PheWAS matrix inputs:

1. `--gene-phewas-stats-in`
2. `--gene-set-phewas-stats-in`

Phenotype annotation policy:

1. use canonical trait linkage for the primary public phenotype annotation layer
2. write the long-form linkage table with `--trait-factor-links-out`
3. interpret trait linkage as linkage of the thresholded high-confidence phenotype support shape, not of a fully observed unthresholded phenotype surface or a biological probability distribution
4. canonical linkage forms a masked full-space target (`s_mask / A`) by dividing masked thresholded trait support by total thresholded trait support before masking, then solves the joint/marginal projections in that full objective space
5. use `--trait-factor-links-output-detail main` for the concise default table (`trait`, `factor`, `is_anchor`, `joint_fraction`, `marginal_fraction`, `marginal_overlap`, `joint_support_mass`, `marginal_support_mass`, `marginal_overlap_support_mass`, `low_retention_flag`, `trait_neff`, `retained_n_eff`) and `--trait-factor-links-output-detail full` when additional retained diagnostics are needed
6. use the retained diagnostics in full-detail `trait_factor_links.out.gz` to judge whether a trait is poorly represented or highly concentrated on the current factor basis:
   - `trait_total_support`
   - `retained_trait_support`
   - `retained_fraction`
   - `trait_n_eff`
   - `retained_n_eff`
   - `total_feature_count`
   - `retained_feature_count`
   - `low_retention_flag`
   - `joint_coefficient_support_mass`
   - `marginal_coefficient_support_mass`
   - `marginal_overlap`
   - `marginal_overlap_support_mass`
7. use `trait_n_eff` and `retained_n_eff` when raw thresholded feature counts overstate breadth; these effective-size diagnostics shrink toward the number of genes carrying most of the support mass
8. use `factor_n_eff`, `factor_top_share`, `factor_top10_share`, and `broad_factor_flag` in `factors.out` to identify broad factors; `broad_factor_flag` marks `factor_n_eff >= 500` and `factor_top_share <= 0.01`
9. for factor -> trait interpretation, filter on trait QC and rank by `joint_coefficient`, using `marginal_coefficient` or `marginal_overlap` as secondary context
10. for trait -> factor interpretation, rank by `joint_coefficient`, not marginal alone; optionally require `broad_factor_flag = 0` or inspect `factor_n_eff`
11. use `--pheno-capture-input weighted_thresholded` by default and `binary_thresholded` only as an expert sensitivity mode
12. default to `--trait-linkage-source combined` with `--trait-linkage-threshold 1.0` (strict `source_value > 1.0`); use `--trait-linkage-source auto` only when you explicitly want fallback resolution (`combined`, then `log_bf`, then `prior`)
13. default to `--trait-linkage-computation-mode sparse_full` for sparse-aware full-space linkage; use `dense_full` only as an expert/debug comparison backend
14. use `--project-phenos-from-gene-sets` only when the gene-set basis is the intended expert or fallback basis
15. treat `--run-factor-phewas` as a secondary expert workflow
16. by default factor-PheWAS uses `--factor-phewas-mode marginal_anchor_adjusted_binary`
17. by default factor-PheWAS uses `--factor-phewas-anchor-covariate direct`
18. use `--factor-phewas-modes mode1,mode2,...` only for explicit expert comparisons; the requested models are appended into one `factor_phewas_stats.out` table
19. add `--factor-phewas-full-output` only when you explicitly want the broader legacy continuous and sensitivity diagnostics
20. to rerun canonical trait linkage from existing EAGGL factors on the gene basis, pass `--factor-gene-clusters-in results/gene_clusters.out.gz`; add `--trait-factor-links-out ...` to write the canonical long-form linkage table, `--run-factor-phewas --factor-phewas-stats-out ...` to write factor-PheWAS, or both in the same command
21. `--factor-phewas-gene-clusters-in` remains accepted as a compatibility alias for the factor-PheWAS-only projection path, but `--factor-gene-clusters-in` is the canonical precomputed-factor input
22. to rerun expert trait linkage from the gene-set basis, pass `--project-phenos-from-gene-sets --factor-gene-set-clusters-in results/gene_set_clusters.out.gz --gene-set-phewas-stats-in ... --trait-factor-links-out ...`; this uses the same projection basis as normal EAGGL factorization
23. to project full genes from existing gene factors, pass `--factor-gene-clusters-in`, `--X-in`, `--gene-set-stats-in`, and `--gene-clusters-full-out`; this reconstructs direct beta-weighted gene-gene evidence
24. to project full genes via an existing gene-set factor basis, pass `--factor-gene-set-clusters-in`, `--X-in`, and `--gene-clusters-full-out`; EAGGL does not infer this basis from `gene_clusters.out.gz`

Projection-only trait linkage from the gene basis:

```bash
$PYTHON -m eaggl factor \
  --factor-gene-clusters-in results/gene_clusters.out.gz \
  --gene-phewas-stats-in /path/to/gene_phewas_stats.out.gz \
  --trait-factor-links-out results/trait_factor_links.projected.out.gz
```

Projection-only trait linkage from the gene-set basis:

```bash
$PYTHON -m eaggl factor \
  --project-phenos-from-gene-sets \
  --factor-gene-set-clusters-in results/gene_set_clusters.out.gz \
  --gene-set-phewas-stats-in /path/to/gene_set_phewas_stats.out.gz \
  --trait-factor-links-out results/trait_factor_links.projected.out.gz
```

Projection-only trait linkage plus factor-PheWAS:

```bash
$PYTHON -m eaggl factor \
  --factor-gene-clusters-in results/gene_clusters.out.gz \
  --gene-phewas-stats-in /path/to/gene_phewas_stats.out.gz \
  --trait-factor-links-out results/trait_factor_links.projected.out.gz \
  --run-factor-phewas \
  --factor-phewas-stats-out results/factor_phewas_stats.out.gz
```

Projection-only full-gene output from gene factors:

```bash
$PYTHON -m eaggl factor \
  --factor-gene-clusters-in results/gene_clusters.out.gz \
  --X-in /path/to/annotations.gmt \
  --gene-set-stats-in /path/to/gene_set_stats.out.gz \
  --gene-clusters-full-out results/gene_clusters_full.projected.out.gz
```

Projection-only full-gene output from gene-set factors:

```bash
$PYTHON -m eaggl factor \
  --factor-gene-set-clusters-in results/gene_set_clusters.out.gz \
  --X-in /path/to/annotations.gmt \
  --gene-clusters-full-out results/gene_clusters_full.projected.out.gz
```

Bundle mode:

1. `--eaggl-bundle-in <bundle.tar.gz>` can provide defaults for core inputs
2. explicit CLI flags always override bundle defaults

## Workflow Matrix

### F1: Single Phenotype Anchoring (default stats path)

Required:

1. no special anchor flags
2. standard factor inputs (`X + gene stats + gene set stats`)

Command:

```bash
$PYTHON -m eaggl factor \
  --X-in /path/to/X.tsv.gz \
  --gene-stats-in /path/to/gene_stats.out \
  --gene-set-stats-in /path/to/gene_set_stats.out \
  --factors-out results/F1.factors.out
```

Gene-by-gene variant:

```bash
$PYTHON -m eaggl factor \
  --X-in /path/to/X.tsv.gz \
  --gene-stats-in /path/to/gene_stats.out \
  --gene-set-stats-in /path/to/gene_set_stats.out \
  --discovery-model gene_by_gene \
  --factor-backend full \
  --learn-phi-backend sentinel_pruned \
  --factors-out results/F1.gene_gene.factors.out
```

### F2: Standalone Gene-list Enrichment

Required:

1. `--gene-list` or `--gene-list-in`
2. `--X-in` or another X-matrix source

Behavior:

1. EAGGL uses the loaded X-gene universe as the enrichment background
2. it runs a hypergeometric test for each loaded gene set against the input gene list
3. it keeps gene sets with Benjamini-Hochberg `q <= --gene-list-max-fdr-q` (default `0.05`)
4. retained gene sets are weighted by `-log(P) / sqrt(gene_set_size)`
5. genes are unweighted and all genes from the retained gene sets are brought into the final factoring matrix

Compatibility aliases:

1. `--positive-controls-list`
2. `--positive-controls-in`

Command:

```bash
$PYTHON -m eaggl factor \
  --X-in /path/to/X.tsv.gz \
  --gene-list INS,GCK,HNF1A \
  --gene-set-stats-out results/F2.gene_set_stats.out \
  --gene-stats-out results/F2.gene_stats.out \
  --factors-out results/F2.factors.out
```

### F3: Single Phenotype + Projection from PheWAS

Required:

1. base F1 inputs
2. either `--gene-phewas-stats-in` or `--gene-set-phewas-stats-in` (or both)

Command:

```bash
$PYTHON -m eaggl factor \
  --X-in /path/to/X.tsv.gz \
  --gene-stats-in /path/to/gene_stats.out \
  --gene-set-stats-in /path/to/gene_set_stats.out \
  --gene-phewas-stats-in /path/to/gene_phewas_stats.out \
  --factors-out results/F3.factors.out
```

### F4: Phenotype-Input Anchoring

Required:

1. matched gene phenotype statistics and gene-set phenotype statistics
2. either repeated PheWAS tables with trait labels, or repeated labeled single-trait stats files using `LABEL=path`
3. every loaded trait must have both gene and gene-set evidence after filtering

Command:

```bash
$PYTHON -m eaggl factor \
  --X-in /path/to/X.tsv.gz \
  --gene-phewas-stats-in /path/to/gene_phewas_stats.out \
  --gene-set-phewas-stats-in /path/to/gene_set_phewas_stats.out \
  --factors-out results/F4.factors.out
```

Single-trait stats can also be supplied as matched labeled pairs:

```bash
$PYTHON -m eaggl factor \
  --X-in /path/to/X.tsv.gz \
  --gene-stats-in T2D=/path/to/t2d.gene_stats.out \
  --gene-set-stats-in T2D=/path/to/t2d.gene_set_stats.out \
  --gene-stats-in CAD=/path/to/cad.gene_stats.out \
  --gene-set-stats-in CAD=/path/to/cad.gene_set_stats.out \
  --factors-out results/F4.factors.out
```

EAGGL treats all complete input traits as anchors. Legacy explicit phenotype, gene, and gene-set anchor flags have been removed; use phenotype-resolved inputs instead.

## Workflow Validation and Guardrails

1. Use `--print-effective-config` first to verify the selected workflow ID and required inputs.
2. Missing required inputs hard-fail with an actionable message.
3. For overlapping flags, EAGGL applies workflow-specific precedence and logs warnings for ignored inputs.
4. PheWAS stages log explicit input-I/O mode:
   - `mode=reuse_loaded_matrix` when a compatible loaded matrix is reused.
   - `mode=re_read_file` when stage inputs must be read from file again.

## Removed Legacy GLS Path

EAGGL does not support the historical GLS/whitened-Y path.

1. Removed aliases hard-fail if passed (`--run-gls`, `run_gls`, `store_cholesky`).
2. Correlation-aware behavior in EAGGL uses the retained corrected-OLS path only.

## Gene-set Filter Relaxation

When EAGGL invokes shared read-X filtering, `--increase-filter-gene-set-p` is treated
as a minimum kept-fraction target used to relax prefiltering if needed.
Post-read filtering does not tighten this threshold.

## References

1. Deterministic workflow baseline generator: `scripts/freeze_factor_workflow_effective_configs.sh`
2. Effective-config fixtures: `tests/data/reference/factor_workflow_effective_config/`
3. PIGEAN handoff details: `docs/eaggl/INTEROP.md`
