# EAGGL Factor Workflows (F1-F9)

This document maps each supported factoring workflow to:

1. required inputs
2. workflow-selection flags
3. a minimal runnable command pattern

All workflows run through `factor` (or `naive_factor`), and the selected workflow ID is visible with `--print-effective-config`.
Optional labeling stays attached to the same factor command; EAGGL does not have a separate `label` mode.
Canonical trait linkage is the primary annotation layer and reports support-capture fractions from the same internal trait-factor matching step. Raw trait support and raw factor loadings keep their original totals; only copied internal vectors are normalized for matching. Factor-PheWAS is a secondary expert-only enrichment regression.

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
Automatic phi tuning is also part of the normal factor workflow surface: add `--learn-phi` to any of the factor workflows below when you want EAGGL to search for a less redundant, restart-stable `phi` before the final reported factorization. The selected `phi`, the redundancy basis, and the search thresholds are written to `--params-out`, and `--learn-phi-report-out` writes the full per-candidate diagnostics table. Redundancy is measured on gene loadings when they are available, with fallback to gene-set or phenotype loadings only when gene loadings are absent. The search records both the worst nearest-neighbor overlap (`redundancy_max`) and a global tail-overlap summary (`redundancy_q90`), treats capped solutions explicitly, and computes factor-mass diagnostics (`effective_factor_count` and `mass_ge_floor_factor_count`) plus convergence diagnostics (`final_delambda`, `final_iterations`, and hit-iteration-cap summaries) for each candidate. Among acceptable uncapped candidates, selection now follows a fit/complexity frontier: EAGGL orders candidates by increasing retained mechanism complexity and keeps adding complexity only while the improvement in reconstruction error per additional retained mechanism stays above the marginal-gain threshold. This avoids the older near-maximal-K bias once redundancy is no longer the active constraint. The default redundancy thresholds are `--learn-phi-max-redundancy 0.5` and `--learn-phi-max-redundancy-q90 0.35`.

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

Use `--blockwise-gene-set-block-size`, `--blockwise-epochs`, `--blockwise-shuffle-blocks`, `--blockwise-warm-start`, `--blockwise-max-blocks`, and `--blockwise-report-out` to tune or audit the blockwise backend. When `--learn-phi-backend blockwise_global_w` is used, neighboring phi candidates are warm-started from the closest previously fitted phi on the log scale when possible.

PheWAS matrix inputs (for phenotype/gene anchor workflows):

1. `--gene-phewas-stats-in`
2. `--gene-set-phewas-stats-in`

Phenotype annotation policy:

1. use canonical trait linkage for the primary public phenotype annotation layer
2. write the long-form linkage table with `--trait-factor-links-out`; `--pheno-clusters-out` remains accepted as a compatibility alias for one release
3. interpret trait linkage as linkage of the thresholded high-confidence phenotype support shape, not of a fully observed unthresholded phenotype surface or a biological probability distribution
4. canonical linkage forms a masked full-space target (`s_mask / A`) by dividing masked thresholded trait support by total thresholded trait support before masking, then solves the joint/marginal projections in that full objective space
5. use the retained diagnostics in `trait_factor_links.out.gz` to judge whether a trait is poorly represented on the current factor basis:
   - `trait_total_support`
   - `retained_trait_support`
   - `retained_fraction`
   - `trait_n_eff`
   - `retained_n_eff`
   - `total_feature_count`
   - `retained_feature_count`
   - `low_retention_flag`
   - `joint_support_mass`
   - `marginal_support_mass`
6. use `trait_n_eff` and `retained_n_eff` when raw thresholded feature counts overstate breadth; these effective-size diagnostics shrink toward the number of genes carrying most of the support mass
7. use `--pheno-capture-input weighted_thresholded` by default and `binary_thresholded` only as an expert sensitivity mode
8. default to `--trait-linkage-source combined` with `--trait-linkage-threshold 1.0` (strict `source_value > 1.0`); use `--trait-linkage-source auto` only when you explicitly want fallback resolution (`combined`, then `log_bf`, then `prior`)
9. use `--project-phenos-from-gene-sets` only when the gene-set basis is the intended expert or fallback basis
10. treat `--run-factor-phewas` as a secondary expert workflow
11. by default factor-PheWAS uses `--factor-phewas-mode marginal_anchor_adjusted_binary`
12. by default factor-PheWAS uses `--factor-phewas-anchor-covariate direct`
13. use `--factor-phewas-modes mode1,mode2,...` only for explicit expert comparisons; the requested models are appended into one `factor_phewas_stats.out` table
13. add `--factor-phewas-full-output` only when you explicitly want the broader legacy continuous and sensitivity diagnostics
14. to rerun canonical trait linkage from existing EAGGL factors on the gene basis, pass `--factor-gene-clusters-in results/gene_clusters.out.gz`; add `--trait-factor-links-out ...` to write the canonical long-form linkage table, `--run-factor-phewas --factor-phewas-stats-out ...` to write factor-PheWAS, or both in the same command
15. `--factor-phewas-gene-clusters-in` remains accepted as a compatibility alias for the factor-PheWAS-only projection path, but `--factor-gene-clusters-in` is the canonical precomputed-factor input
16. to rerun expert trait linkage from the gene-set basis, pass `--project-phenos-from-gene-sets --factor-gene-set-clusters-in results/gene_set_clusters.out.gz --gene-set-phewas-stats-in ... --trait-factor-links-out ...`; this uses the same projection basis as normal EAGGL factorization

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
  --factor-gene-set-clusters-in results/gene_set_clusters.out.gz \
  --project-phenos-from-gene-sets \
  --gene-set-phewas-stats-in /path/to/gene_set_phewas_stats.out.gz \
  --gene-phewas-stats-in /path/to/gene_phewas_stats.out.gz \
  --trait-factor-links-out results/trait_factor_links.projected.out.gz \
  --run-factor-phewas \
  --factor-phewas-stats-out results/factor_phewas_stats.out.gz
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

### F4: Multi-Phenotype Anchoring

Required:

1. `--anchor-phenos <comma-separated phenotypes>`
2. `--gene-phewas-stats-in`
3. `--gene-set-phewas-stats-in`

Command:

```bash
$PYTHON -m eaggl factor \
  --X-in /path/to/X.tsv.gz \
  --anchor-phenos T2D,T2D_ALT \
  --gene-phewas-stats-in /path/to/gene_phewas_stats.out \
  --gene-set-phewas-stats-in /path/to/gene_set_phewas_stats.out \
  --factors-out results/F4.factors.out
```

### F5: Any-Phenotype Anchoring

Required:

1. `--anchor-any-pheno`
2. `--gene-phewas-stats-in`
3. `--gene-set-phewas-stats-in`

Command:

```bash
$PYTHON -m eaggl factor \
  --X-in /path/to/X.tsv.gz \
  --anchor-any-pheno \
  --gene-phewas-stats-in /path/to/gene_phewas_stats.out \
  --gene-set-phewas-stats-in /path/to/gene_set_phewas_stats.out \
  --factors-out results/F5.factors.out
```

### F6: Single-Gene Anchoring

Required:

1. `--anchor-genes <GENE>`
2. `--gene-phewas-stats-in`
3. `--gene-set-phewas-stats-in`

Command:

```bash
$PYTHON -m eaggl factor \
  --X-in /path/to/X.tsv.gz \
  --anchor-genes INS \
  --gene-phewas-stats-in /path/to/gene_phewas_stats.out \
  --gene-set-phewas-stats-in /path/to/gene_set_phewas_stats.out \
  --factors-out results/F6.factors.out
```

### F7: Multi-Gene Anchoring

Required:

1. `--anchor-genes <comma-separated genes>`
2. `--gene-phewas-stats-in`
3. `--gene-set-phewas-stats-in`

Optional expansion knobs:

1. `--add-gene-sets-by-enrichment-p`
2. `--add-gene-sets-by-fraction`
3. `--add-gene-sets-by-naive`
4. `--add-gene-sets-by-gibbs`

Command:

```bash
$PYTHON -m eaggl factor \
  --X-in /path/to/X.tsv.gz \
  --anchor-genes INS,GCK \
  --gene-phewas-stats-in /path/to/gene_phewas_stats.out \
  --gene-set-phewas-stats-in /path/to/gene_set_phewas_stats.out \
  --factors-out results/F7.factors.out
```

### F8: Any-Gene Anchoring

Required:

1. `--anchor-any-gene`
2. `--gene-phewas-stats-in`
3. `--gene-set-phewas-stats-in`

Command:

```bash
$PYTHON -m eaggl factor \
  --X-in /path/to/X.tsv.gz \
  --anchor-any-gene \
  --gene-phewas-stats-in /path/to/gene_phewas_stats.out \
  --gene-set-phewas-stats-in /path/to/gene_set_phewas_stats.out \
  --factors-out results/F8.factors.out
```

### F9: Gene-set Anchoring

Required:

1. `--anchor-gene-set`
2. `--run-phewas`
3. `--gene-phewas-stats-in`
4. enough evidence input to compute gene/gene-set scores if not precomputed

Command:

```bash
$PYTHON -m eaggl factor \
  --X-in /path/to/X.tsv.gz \
  --anchor-gene-set \
  --gene-stats-in /path/to/gene_stats.out \
  --gene-set-stats-in /path/to/gene_set_stats.out \
  --run-phewas \
  --gene-phewas-stats-in /path/to/gene_phewas_stats.out \
  --factors-out results/F9.factors.out
```

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
