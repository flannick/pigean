# EAGGL Factor Workflows (F1-F4)

This document maps each supported factoring workflow to:

1. required inputs
2. workflow-selection flags
3. a minimal runnable command pattern

All workflows run through `factor` (or `naive_factor`), and the selected workflow ID is visible with `--print-effective-config`.
Optional labeling stays attached to the same factor command; EAGGL does not have a separate `label` mode. Use `--gene-sets-for-labeling` one or more times to limit factor-label candidates to selected gene-set libraries without changing the fitted loadings.
Simplified trait linkage is the primary annotation layer. It reports fixed-W projection loadings from probability-transformed trait support vectors onto the factor basis. Factor-as-gene-set regression statistics are produced by exporting factors with `--factor-gmt-out` and running PIGEAN `multi-y`, not by EAGGL.

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
11. `--gene-gene-profligate-correction linear` is an opt-in gene-by-gene diagnostic/production control that subtracts a simple fitted retained-annotation-count effect from raw pair log evidence before the usual pair-probability calibration. The default `none` path is unchanged.
12. `--annotation-bridge-metrics-out` is an optional gene-by-gene post-processing output that reports which annotations bridge otherwise distinct fitted gene factors.
13. `--anchor-aggregation multi` is the default multi-anchor mode for both discovery models; `any` uses explicit noisy-OR union evidence. With one anchor trait, both modes reduce exactly to ordinary single-trait anchoring.
14. `gene_by_gene` still learns one shared gene-factor basis and projects annotations onto that basis afterward; it does not write anchor-specific cluster outputs.

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
Automatic phi tuning is also part of the normal factor workflow surface: add `--learn-phi` to any factor workflow when you want EAGGL to search over candidate `phi` values before the final fit. The default `--phi-selection-objective composite` evaluates each candidate with a weighted score over factor size, non-overlap, entity concentration, high-priority gene/annotation coverage, reconstruction, coherence, factor balance, and annotation bridge QC when available. All component scores are scaled to `[0,1]`, unavailable components are skipped with weight renormalization, and the selected `phi` plus component diagnostics are written to `--params-out`. `--phi-selection-metrics-wide-out` and `--phi-selection-metrics-long-out` write explicit per-candidate audit tables. Use `--phi-selection-objective legacy --learn-phi-target-gene-effective-support S` when you want the previous target-size selector. Use `--learn-phi-values` with a comma-separated list when you want to evaluate a manual candidate grid while keeping the selected objective's scoring and tie-breakers.

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

1. it uses corrected `beta` values for pairwise evidence (`--gene-gene-beta-source beta`)
2. it converts retained shared-annotation evidence into pairwise probabilities before factoring
3. it currently supports only `--factor-backend full`
4. it ignores discovery-family subsetting and weighting flags because pairwise evidence is built from all retained gene sets
5. with multiple anchor traits, each trait-specific pair target is built exactly as in single-trait mode; `multi` factors the equal average of all per-anchor target matrices, while `any` uses noisy-OR union over those matrices
6. optional `--gene-gene-profligate-correction linear` uses retained/scored annotation counts from the annotations entering the gene-gene matrix to correct the raw pair log evidence for broadly annotated genes

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

1. use fixed-W trait linkage for the primary public phenotype annotation layer
2. write the long-form linkage table with `--trait-factor-links-out`
3. interpret `nnls_loading` as fixed-W projection of the selected trait support surface after converting combined/log-BF values to probabilities with `background_prior`
4. interpret `beta`, `beta_uncorrected`, `beta_tilde`, `se`, `z`, and `p_value` as factor-as-gene-set association summaries
5. use `--trait-factor-linkage-factor-gene-threshold` to set the minimum factor gene loading included when exporting factor GMTs for PIGEAN enrichment
6. use `--factor-gmt-out` to export thresholded factors for a manual PIGEAN `multi-y` run

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

Full-gene projection policy for gene-by-gene workflows:

1. The recommended/default display projection is gene-set-routed full-gene projection, written with `--gene-clusters-full-via-gene-sets-out`.
2. Direct full-gene projection, written with `--gene-clusters-full-out`, remains available as a diagnostic and comparison output.
3. Dashboards and factor graphs prefer `gene_clusters_full_via_gene_sets.out.gz` when both full-gene outputs are present.

Projection-only recommended full-gene output from gene-set factors:

```bash
$PYTHON -m eaggl factor \
  --factor-gene-set-clusters-in results/gene_set_clusters.out.gz \
  --X-in /path/to/annotations.gmt \
  --gene-clusters-full-via-gene-sets-out results/gene_clusters_full.via_gene_sets.out.gz
```

Projection-only diagnostic direct full-gene output from gene factors:

```bash
$PYTHON -m eaggl factor \
  --factor-gene-clusters-in results/gene_clusters.out.gz \
  --X-in /path/to/annotations.gmt \
  --gene-set-stats-in /path/to/gene_set_stats.out.gz \
  --gene-clusters-full-out results/gene_clusters_full.direct.out.gz
```

Projection-only direct and gene-set-mediated full-gene outputs in one command:

```bash
$PYTHON -m eaggl factor \
  --factor-gene-clusters-in results/gene_clusters.out.gz \
  --factor-gene-set-clusters-in results/gene_set_clusters.out.gz \
  --X-in /path/to/annotations.gmt \
  --gene-set-stats-in /path/to/gene_set_stats.out.gz \
  --gene-clusters-full-out results/gene_clusters_full.direct.out.gz \
  --gene-clusters-full-via-gene-sets-out results/gene_clusters_full.via_gene_sets.out.gz
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
  --factors-out results/F1.gene_gene.factors.out \
  --annotation-bridge-metrics-out results/F1.annotation_bridge_metrics.tsv.gz \
  --annotation-bridge-suggested-exclude-out results/F1.annotation_bridge_suggested_exclude.txt
```

`annotation_bridge_metrics.tsv.gz` includes review diagnostics and one conservative
automatic exclusion flag. `flag_review` / `flag_bridge_candidate` mark broad bridge
candidates for inspection. `flag_suggest_exclude` marks high-confidence diffuse
bridges from empirically broad/noisy sources and is the only flag used to write
`annotation_bridge_suggested_exclude.txt`. The same rule can be rerun after a
full-Gibbs exclusion refit; newly suggested annotations reflect the changed model,
not a different exclusion mode.

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

### Simplified Trait Linkage

EAGGL trait-factor linkage is intentionally limited to fixed-W projection loadings. Combined/log-BF phenotype values are converted to probabilities using the run `background_prior`; sparse inputs preserve implicit zeros as zero probability unless dense computation is requested. Factors are exported as weighted gene sets after zeroing gene loadings below `--trait-factor-linkage-factor-gene-threshold` (default `0.05`). `--factor-gmt-out` writes those thresholded factor gene sets for PIGEAN `multi-y` runs when beta statistics are needed.

Gene, gene-set, and phenotype cluster tables now include raw `Factor*` loadings, `Cosine_Factor*`, and `Euclidean_Factor*` columns. `Cosine_FactorK` is the cosine similarity between the row's factor-loading vector and the indicator vector for factor `K`; `Euclidean_FactorK` is the Euclidean distance to that same one-hot factor indicator.
