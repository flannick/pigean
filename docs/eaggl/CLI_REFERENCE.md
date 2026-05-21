# EAGGL CLI Reference

This is the canonical human-written manual for the stable, routinely used `python -m eaggl` command surface.

Use this document for practical command shapes, workflow selection, and the meaning of the main EAGGL flags.
Use `docs/eaggl/CLI_OPTIONS.md` for the exhaustive machine-generated parser inventory.
Use `README.md` for the full repository documentation map.
Optional downstream analyses use explicit `--run-*` booleans with separate `--*-in` / `--*-out` flags. Older hybrid flags remain accepted as compatibility aliases but are not the canonical documented surface.

Scope rules for this document:
- only stable user-facing workflows and flag groups are described here
- exhaustive option coverage lives in `docs/eaggl/CLI_OPTIONS.md`
- every documented flag in this file should have direct regression coverage or explicit mapping to an existing EAGGL CLI test
- niche, debug-only, or transitional flags belong in the generated inventory, not in this manual

## Entry points and modes

Primary entrypoint:

```bash
PYTHONPATH=src python -m eaggl <mode> [...options]
```

Common modes:
- `factor`: canonical EAGGL factor workflow with F1-F4 workflow selection
- `naive_factor`: simpler baseline factorization path using the same high-level contracts

Typical user workflow:

1. build or load the matrix and PIGEAN-derived evidence to factor
2. choose the anchoring workflow
3. fit the ARD nonnegative factor model
4. optionally annotate factors with canonical trait-linkage weights
5. optionally run factor-PheWAS as a secondary expert enrichment analysis
6. optionally label the factors

## Common command shapes

Default factor workflow from direct inputs:

```bash
PYTHONPATH=src python -m eaggl factor \
  --X-in bundles/current/model_small/data/gene_set_list_mouse_2024.txt \
  --gene-stats-in path/to/gene_stats.out \
  --gene-set-stats-in path/to/gene_set_stats.out \
  --factors-out results/factors.out \
  --gene-set-clusters-out results/gene_set_clusters.out \
  --gene-clusters-out results/gene_clusters.out \
  --params-out results/params.out
```

Bundle-driven factor workflow:

```bash
PYTHONPATH=src python -m eaggl factor \
  --eaggl-bundle-in path/to/pigean_to_eaggl.tar.gz \
  --factors-out results/factors.out \
  --gene-set-clusters-out results/gene_set_clusters.out \
  --gene-clusters-out results/gene_clusters.out \
  --params-out results/params.out
```

Consensus cNMF workflow from bundled PIGEAN outputs:

```bash
PYTHONPATH=src python -m eaggl factor \
  --eaggl-bundle-in path/to/pigean_to_eaggl.tar.gz \
  --factor-runs 3 \
  --consensus-nmf \
  --factors-out results/factors.out \
  --gene-set-clusters-out results/gene_set_clusters.out \
  --gene-clusters-out results/gene_clusters.out \
  --params-out results/params.out \
  --consensus-stats-out results/consensus.tsv
```

Automatic phi-tuning workflow from bundled PIGEAN outputs:

```bash
PYTHONPATH=src python -m eaggl factor \
  --eaggl-bundle-in path/to/pigean_to_eaggl.tar.gz \
  --learn-phi \
  --learn-phi-target-gene-effective-support 100 \
  --learn-phi-max-num-iterations 50 \
  --learn-phi-report-out results/phi_search.tsv \
  --factors-out results/factors.out \
  --gene-set-clusters-out results/gene_set_clusters.out \
  --gene-clusters-out results/gene_clusters.out \
  --params-out results/params.out
```

`--params-out` is the resolved run record. For factor runs it writes the effective factor configuration, including restart and consensus settings, anchor/filter choices, labeling settings, the final `phi` used for fitting, and any `--learn-phi` search diagnostics. Current `--learn-phi` diagnostics include the target factor size, primary-factor gene-support summaries, the redundancy basis, candidate capped-status, nearest-neighbor overlap summaries (`redundancy_max` and `redundancy_q90`), factor-mass summaries (`effective_factor_count` and `primary_factor_count`), and convergence diagnostics such as final `delambda` and whether the scout fit hit the iteration cap.

Scalable backend note:
- `--factor-backend full|blockwise_global_w` selects the final factorization backend.
- `--learn-phi-backend sentinel_pruned|blockwise_global_w` selects the phi-search backend.
- `blockwise_global_w` keeps one shared global gene-factor basis and one shared ARD state across block-local gene-set solves, so all retained gene sets can contribute during phi search and final factorization without requiring one giant in-memory solve.

Phenotype-input anchoring workflow:

```bash
PYTHONPATH=src python -m eaggl factor \
  --X-in bundles/current/model_small/data/gene_set_list_mouse_2024.txt \
  --gene-phewas-stats-in path/to/gene_phewas_stats.out \
  --gene-set-phewas-stats-in path/to/gene_set_phewas_stats.out \
  --factors-out results/factors.out \
  --gene-set-clusters-out results/gene_set_clusters.out \
  --gene-clusters-out results/gene_clusters.out \
  --trait-factor-links-out results/trait_factor_links.out \
  --params-out results/params.out
```

Before a large run, inspect the selected workflow and resolved defaults:

```bash
PYTHONPATH=src python -m eaggl factor --print-effective-config [...workflow flags...]
```

## Workflow map

The supported workflow families are documented in detail in `docs/eaggl/WORKFLOWS.md`.

At a high level:
- `F1`: default single-phenotype factoring from PIGEAN gene/gene-set stats
- `F2`: standalone gene-list enrichment and factoring
- `F3`: default factorization with phenotype projection from PheWAS inputs
- `F4`: phenotype-input anchoring from matched gene and gene-set phenotype statistics

Use `--print-effective-config` to confirm which workflow the CLI selected.

## Documented option groups

### Runtime and reproducibility

| Flag | Meaning |
|---|---|
| `--config` | load a config profile before applying CLI overrides |
| `--deterministic` | force deterministic seeds and deterministic runtime behavior where supported |
| `--seed` | explicit RNG seed |
| `--debug-level` | increase debug logging |
| `--max-gb` | set memory budget used for batching heuristics |
| `--print-effective-config` | print the fully resolved config/options and selected workflow and exit |

### Core matrix and handoff inputs

| Flag | Meaning |
|---|---|
| `--X-in` | read one or more sparse gene-set matrix files |
| `--X-list` | read a file listing sparse matrix inputs |
| `--Xd-in` | read one or more dense matrix files |
| `--Xd-list` | read a file listing dense matrix inputs |
| `--gene-stats-in` | read PIGEAN gene-level statistics |
| `--gene-set-stats-in` | read PIGEAN gene-set statistics |
| `--eaggl-bundle-in` | load a bundled PIGEAN-to-EAGGL handoff |
| `--gene-map-in` | map input gene identifiers onto the runtime gene space when needed |
| `--gene-loc-file` | gene location file used by shared read-X/runtime paths when needed |

### Workflow Selectors

| Flag | Meaning |
|---|---|
| `--gene-list-in` | read a standalone input gene list from a file and let EAGGL synthesize enrichment weights internally |
| `--gene-list` | provide a standalone input gene list directly on the command line |
| `--gene-list-id-col` | choose the gene column from a standalone gene-list file when it has multiple columns |
| `--gene-list-no-header` | declare that the standalone gene-list file has no header row |
| `--gene-list-max-fdr-q` | retain gene sets up to this FDR threshold in standalone gene-list mode |
| `--positive-controls-in` | legacy compatibility alias for `--gene-list-in` |
| `--positive-controls-list` | legacy compatibility alias for `--gene-list` |
| `--positive-controls-all-in` | legacy compatibility alias; standalone gene-list mode still uses the loaded X-gene universe as the background |

Notes:
- `--gene-list-in` / `--gene-list` are the primary standalone EAGGL workflow selectors
- standalone gene-list mode uses the loaded X-gene universe as the enrichment background
- retained standalone gene sets are weighted by `-log(P) / sqrt(gene_set_size)` and filtered by `--gene-list-max-fdr-q`
- `--gene-list` expects a comma-separated list, not a file path
- `--positive-controls-in` / `--positive-controls-list` remain compatibility aliases for the standalone gene-list workflow and are hidden from the primary help surface
- phenotype-input anchoring is selected by matched gene and gene-set phenotype inputs; EAGGL uses all complete input phenotypes as anchors

### PheWAS and projection inputs

| Flag | Meaning |
|---|---|
| `--gene-phewas-stats-in` | load gene-by-phenotype statistics |
| `--gene-set-phewas-stats-in` | load gene-set-by-phenotype statistics |
| `--run-phewas` | run a gene-level PheWAS stage from `--gene-phewas-stats-in` |
| `--run-factor-phewas` | compute factor-level phenotype enrichment regression from `--gene-phewas-stats-in` |
| `--factor-gene-clusters-in` | load an existing `gene_clusters.out(.gz)` factor loading table and run projection-only simplified trait linkage, factor-PheWAS, or both without refitting factors |
| `--factor-gene-set-clusters-in` | load an existing `gene_set_clusters.out(.gz)` factor loading table for projection-only simplified trait linkage from the gene-set basis |
| `--factor-phewas-gene-clusters-in` | compatibility alias for the older factor-PheWAS-only projection command |
| `--project-phenos-from-gene-sets` | compute simplified trait linkage on the gene-set basis instead of the gene basis |
| `--pheno-capture-input` | choose whether simplified trait linkage uses retained weighted thresholded support or binary thresholded hits |
| `--trait-factor-links-out` | write the simplified long-form trait-factor linkage table |
| `--trait-linkage-source` | choose the support surface for simplified trait linkage; default is `combined` (expert overrides: `auto`, `log_bf`, `prior`) |
| `--trait-linkage-threshold` | strict threshold for simplified trait linkage support (`source_value > threshold`) |
| `--trait-linkage-computation-mode` | choose the linkage computation backend: `sparse_full` by default, or `dense_full` as a debug comparison backend |
| `--trait-factor-links-output-detail` | accepted for command compatibility; simplified linkage writes one concise schema |
| `--no-trait-linkage` | disable simplified trait linkage even when trait inputs are available |
| `--factor-phewas-modes` | expert override: run multiple factor-PheWAS model surfaces in one pass and append them into one output table |
| `--factor-phewas-full-output` | expose the full expert factor-PheWAS surface, including combined and Huber variants |

Operational notes:
- simplified trait linkage writes one long table with one row per `(trait, factor)`
- `nnls_loading` is the NNLS projection of the thresholded trait support vector onto the factor-loading basis
- `beta`, `beta_uncorrected`, `beta_tilde`, `se`, `z`, and `p_value` treat each EAGGL factor as a weighted gene set after applying `--trait-factor-linkage-factor-gene-threshold`
- use `--factor-gmt-out` to export those thresholded weighted factors for manual PIGEAN `multi-y` runs
- raw trait support and raw factor loadings are not forced to sum to `1`
- `--pheno-capture-input weighted_thresholded` is the default and uses retained source-support values that strictly exceed `--trait-linkage-threshold`; `binary_thresholded` is an expert sensitivity mode
- if `--trait-linkage-source auto` is requested, one support surface is chosen per run in the order `combined`, then `log_bf`, then `prior`
- factor-PheWAS is a secondary expert analysis for factor-specific phenotype enrichment
- the default factor-PheWAS mode is `marginal_anchor_adjusted_binary`, which regresses thresholded phenotype-hit membership on one factor at a time while adjusting for direct anchor support
- projection-only gene-basis phenotype clusters and factor-PheWAS use the raw `Factor1..FactorK` columns from `gene_clusters.out(.gz)` as the gene-factor loading matrix; any `combined`, `log_bf`, or `prior` columns in that file are reused as anchor covariates unless overridden by `--gene-stats-in`
- projection-only gene-basis trait linkage requires `--gene-phewas-stats-in` and writes the long-form `trait_factor_links.out(.gz)` table
- projection-only full-gene output from `--factor-gene-clusters-in` uses direct beta-weighted gene-gene evidence and therefore requires `--X-in` plus `--gene-set-stats-in`; write it with `--gene-clusters-full-out`
- projection-only gene-set-basis trait linkage uses the raw `Factor1..FactorK` columns from `gene_set_clusters.out(.gz)` with `--gene-set-phewas-stats-in`; request this with `--project-phenos-from-gene-sets`
- projection-only full-gene output from precomputed gene-set factors requires `--factor-gene-set-clusters-in` plus `--X-in`; write it with `--gene-clusters-full-via-gene-sets-out`, or with `--gene-clusters-full-out` when a gene-set factor basis is the only factor input
- direct and gene-set-mediated full-gene projections can be written in one projection-only command by passing both `--factor-gene-clusters-in` and `--factor-gene-set-clusters-in` plus separate `--gene-clusters-full-out` and `--gene-clusters-full-via-gene-sets-out` paths
- projection-only reuse expects the standard non-anchor `gene_clusters.out(.gz)` and `gene_set_clusters.out(.gz)` tables with one row per gene or gene set
- if you request multiple factor-PheWAS models in one run, `factor_phewas_stats.out` appends them together and labels each row with `model_name`, `factor_model_scope`, `outcome_surface`, and `anchor_covariate`
- `--factor-phewas-full-output` restores the broader legacy continuous and sensitivity outputs for expert diagnostics
- compatibility aliases remain accepted but are not the canonical public interface:
  - `--run-phewas-from-gene-phewas-stats-in <file>`
  - `--factor-phewas-from-gene-phewas-stats-in <file>`
  - each behaves like the corresponding `--run-*` flag plus `--gene-phewas-stats-in <file>`

### Input schema and column selectors

Use these only when your files do not match the expected default headers.

| Selector family | Meaning |
|---|---|
| gene-stats column selectors such as `--gene-stats-id-col` and `--gene-stats-prior-col` | choose the gene-level score columns used from `--gene-stats-in` |
| gene-set-stats column selectors such as `--gene-set-stats-id-col` and `--gene-set-stats-beta-uncorrected-col` | choose the gene-set score columns used from `--gene-set-stats-in` |
| gene-PheWAS column selectors such as `--gene-phewas-stats-id-col` and `--gene-phewas-stats-pheno-col` | choose the gene-PheWAS columns used from `--gene-phewas-stats-in` |
| gene-set-PheWAS column selectors such as `--gene-set-phewas-stats-id-col` and `--gene-set-phewas-stats-pheno-col` | choose the gene-set-PheWAS columns used from `--gene-set-phewas-stats-in` |
| `--gene-phewas-id-to-X-id` | map gene IDs in the PheWAS input onto the X-matrix gene IDs |

Operational note:
- Use `--X-in` for a direct `.gmt` sparse matrix file.
- `--X-list` is for a text file that lists sparse matrix inputs one per line.
- For compatibility, a direct `.gmt` or `.gmt.gz` path passed to `--X-list` is accepted and treated like `--X-in`, but EAGGL emits a warning and `--X-in` remains the canonical form.

### Core factor model controls

| Flag | Meaning |
|---|---|
| `--max-num-factors` | upper bound on the number of latent factors |
| `--phi` | primary sparsity / concentration control for the factor model; default initial value is `0.05` for `gene_by_annotation` and `0.01` for `gene_by_gene` when `--phi` is not explicitly set |
| `--alpha0` | ARD hyperparameter controlling factor shrinkage |
| `--beta0` | companion ARD hyperparameter controlling factor shrinkage scale |
| `--min-lambda-threshold` | drop weak factors whose relevance falls below this threshold |
| `--no-transpose` | keep the original matrix orientation instead of the default transposed view |

### Restart and consensus controls

These are first-tier factorization controls in the normal public EAGGL interface.

| Flag | Meaning |
|---|---|
| `--factor-runs` | number of random restarts for factorization; if greater than `1` without consensus enabled, EAGGL keeps the best-evidence run |
| `--consensus-nmf` | aggregate multiple restarts into a consensus factorization instead of selecting a single best run |
| `--consensus-min-factor-cosine` | minimum cosine similarity required to match a restart factor to the reference factor during consensus building |
| `--consensus-min-run-support` | minimum fraction of restart runs that must support a consensus factor for it to be kept |
| `--consensus-aggregation` | aggregation rule for matched factor loadings across supporting runs (`median` or `mean`) |

Operational note:
- `--consensus-nmf` requires `--factor-runs >= 2`.
- If `--factor-runs > 1` and `--consensus-nmf` is not set, EAGGL performs multi-start factorization and keeps only the best-evidence run.

### Automatic phi tuning

These are first-tier factorization controls when you want EAGGL to choose a better `phi` automatically rather than trusting a single fixed guess.

| Flag | Meaning |
|---|---|
| `--learn-phi` | enable structural auto-tuning of `phi` before the final reported factorization |
| `--learn-phi-target-gene-effective-support` | required with `--learn-phi`; target median effective gene support among primary factors |
| `--learn-phi-size-tolerance-frac` | fractional tolerance around the target primary-factor gene effective support; defaults to `0.25` |
| `--learn-phi-min-primary-factors` | minimum primary factor count required for a candidate; defaults to `3` |
| `--learn-phi-max-primary-gene-max-weight-q90` | optional spike guardrail: maximum 90th percentile of primary-factor maximum gene weight |
| `--learn-phi-max-redundancy` | maximum within-run weighted Jaccard overlap allowed between metric-scope factors in the selected solution, measured on gene loadings when available; the default `0.5` is intended as a rough \"share at most about half\" rule |
| `--learn-phi-runs-per-step` | number of restart fits used to score each tested `phi` candidate; defaults to `1` for cheaper search, with larger values as an expert stability check |
| `--learn-phi-min-run-support` | minimum fraction of restart runs that must agree on the modal retained factor count |
| `--learn-phi-min-stability` | minimum mean matched-factor cosine across the modal restart runs |
| `--learn-phi-fit-loss-warning-frac` | reconstruction-loss warning threshold used during target-size phi selection; the old `--learn-phi-max-fit-loss-frac` spelling remains accepted as a compatibility alias |
| `--learn-phi-max-severe-fit-loss-frac` | hard severe-underfit threshold relative to the best phi-search candidate; blocks pathological high-phi underfit even when target size is satisfied |
| `--learn-phi-max-steps` | maximum number of additional `phi` candidates to evaluate after the initial `--phi`; defaults to `5` |
| `--learn-phi-values` | comma-separated explicit candidate `phi` values to evaluate; when set, EAGGL skips adaptive candidate generation but still applies the normal phi-selection criteria |
| `--learn-phi-backend` | choose between the legacy sentinel-pruned phi search and the blockwise-global-W phi search over all retained gene sets |
| `--learn-phi-expand-factor` | multiplicative factor used when widening the search bracket away from the initial `--phi` |
| `--learn-phi-weight-floor` | factor weights below this are treated as zero when computing redundancy |
| `--learn-phi-metric-factor-scope` | choose whether phi-selection redundancy and repeat-stability metrics use `primary` factors (default) or `all` fitted factors; independent of printed output scope |
| `--learn-phi-report-out` | optional per-candidate diagnostics table for all tested `phi` values |
| `--factor-phi-metrics-out` | optional per-factor diagnostics table for each tested `phi` |
| `--factor-phi-factors-out` | optional `factors.out`-style rows for each tested `phi`, with a leading `phi` column |
| `--factor-phi-gene-set-clusters-out` | optional `gene_set_clusters.out`-style rows for each tested `phi`, with a leading `phi` column |
| `--factor-phi-gene-clusters-out` | optional `gene_clusters.out`-style rows for each tested `phi`, with a leading `phi` column |
| `--learn-phi-prune-genes-num` | expert shortcut: during phi search only, correlation-prune to at most this many representative genes before candidate NMF evaluation; defaults to `1000` |
| `--learn-phi-prune-gene-sets-num` | deprecated expert knob retained for compatibility; phi search now uses the same discovery plan as the final fit and this option is ignored |
| `--learn-phi-max-num-iterations` | expert shortcut: during phi search only, cap candidate NMF iterations separately from the final factorization |

Operational notes:
- `--phi` remains the initial guess. With `--learn-phi`, EAGGL treats it as the starting point for search rather than the final fixed value.
- The default `--phi-selection-objective composite` scores each candidate using factor size, non-overlap, entity concentration, high-priority coverage, reconstruction, coherence, factor balance, and annotation bridge QC when available. Each component is scaled to `[0,1]`; unavailable components are skipped and the remaining weights are renormalized.
- Use `--phi-selection-objective legacy` to recover the previous target-size heuristic. In legacy mode, `--learn-phi` still requires one of the `--learn-phi-target-*` options.
- Composite weights can be overridden with `--phi-selection-composite-weights factor_size=0.15,nonoverlap=0.15,entity_concentration=0.15,coverage=0.25,reconstruction=0.10,coherence=0.10,factor_balance=0.05,annotation_bridge_qc=0.05`. Unknown names and negative weights are rejected.
- `--phi-selection-target-factor-gene-mass` defaults to `100` and controls the composite factor-size score. If no legacy target is supplied, the same value is also used as the adaptive search-size target.
- `--phi-selection-metrics-wide-out` writes one row per candidate with component scores, weights, contributions, and the final composite score. `--phi-selection-metrics-long-out` writes one row per candidate-component pair, including availability and unavailable reasons.
- Use `--learn-phi-values 0.005,0.01,0.02` to manually specify the phi candidate grid. This overrides adaptive expansion/refinement, but candidate scoring and selection are unchanged.
- When `--discovery-model gene_by_gene` is selected and `--phi` is not explicitly provided, EAGGL starts the search from `0.01` rather than the rectangular-model default `0.05`.
- Composite ties within `--phi-selection-tie-tolerance` prefer fewer factors, then higher coverage, then lower `phi`. Legacy ties keep the previous target-size tie-breakers.
- Auto-tuning uses `--learn-phi-runs-per-step` during search, then runs the normal final factorization with the selected `phi`. The default search uses one restart per candidate; increase this only when you explicitly want a more expensive restart-stability check.
- The selected `phi`, objective, component scores, search settings, and per-candidate diagnostics are written to both the run log and `--params-out`. Use `--learn-phi-report-out`, `--factor-phi-metrics-out`, and the composite wide/long outputs when you also want separate audit artifacts.
- Candidate factor mass is still reported. `effective_factor_count` is the inverse-participation-ratio style mass summary `(sum m_k)^2 / sum m_k^2`, and `primary_factor_count` counts factors whose mass fraction exceeds `--learn-phi-mass-floor-frac`.
- The search report also records whether the scout factorization converged before the candidate iteration cap, via `final_delambda`, `final_iterations`, `converged_fraction`, and `hit_iteration_cap_fraction`.
- By default, `--learn-phi-backend sentinel_pruned` evaluates candidates on a lightweight sentinel panel. `--learn-phi-prune-genes-num` still affects that gene-side shortcut, but gene-set discovery now follows the same retained-to-discovery plan used by the final fit.
- `--learn-phi-backend blockwise_global_w` instead evaluates all retained gene sets in blocks while keeping one shared global gene-factor basis and one shared ARD state.
- `--learn-phi-prune-genes-num` and `--learn-phi-max-num-iterations` apply only while scoring phi candidates. Gene-set discovery selection itself is shared between learn-phi and the final reported factorization.

### Factor pruning, weighting, and post-processing

| Flag | Meaning |
|---|---|
| `--max-num-discovery-gene-sets` | cap the number of discovery family leaders used to learn the latent basis `W`; all retained annotations are still projected afterward |
| `--no-auto-discovery-subset` | disable the default family-leader discovery subset and instead fit discovery on all retained gene sets |
| `--discovery-redundancy-weighting-mode` | choose leader-family discovery weighting: `effective_size`, `log_effective_size`, or `none` |
| `--no-discovery-redundancy-weighting` | disable the default redundancy-balanced discovery weighting |
| `--discovery-similarity-threshold` | similarity threshold used when assigning retained gene sets to discovery families; defaults to `0.35` |
| `--discovery-model` | choose whether discovery is learned from the default rectangular gene-by-annotation matrix (`gene_by_annotation`) or from a symmetric gene-by-gene pairwise matrix (`gene_by_gene`) |
| `--factor-prune-gene-sets-num` / `--factor-prune-gene-sets-val` | deprecated factor-stage discovery controls kept only as compatibility aliases; use the discovery flags above instead |
| `--factor-prune-genes-num` / `--factor-prune-genes-val` | prune weak gene memberships from factor outputs |
| `--factor-prune-phenos-num` / `--factor-prune-phenos-val` | prune weak phenotype memberships from factor outputs |
| `--factor-backend` | choose the final factorization backend: `full` or `blockwise_global_w` |
| `--blockwise-gene-set-block-size` | set the retained gene-set block size used by `blockwise_global_w` |
| `--blockwise-epochs` | number of global block passes used by `blockwise_global_w` |
| `--blockwise-shuffle-blocks` | shuffle gene-set block order between blockwise epochs |
| `--blockwise-warm-start` | warm-start neighboring phi candidates in blockwise phi search |
| `--blockwise-max-blocks` | optional debugging cap on the number of processed blocks per epoch |
| `--blockwise-report-out` | write per-epoch blockwise diagnostics |
| `--cluster-row-min-max-loading` | minimum row-wise maximum raw factor loading required to print gene and gene-set cluster rows; defaults to `0.01` |
| `--factor-output-scope` | choose which factor tiers are printed in user-facing factor and cluster outputs: `primary` (default), `primary_secondary`, or `all` |

Gene-by-gene expert controls:

| Flag | Meaning |
|---|---|
| `--gene-gene-beta-source` | gene-by-gene pairwise evidence source; corrected `beta` is required |
| `--gene-gene-pair-prior` | set the direct prior probability that two retained genes share a mechanism before observing shared annotation evidence |
| `--gene-gene-pair-prior-effective-size` | set the effective mechanism size used to derive the pair prior when no direct prior is supplied |
| `--gene-gene-logbf-base` | declare whether the shared annotation evidence is already in natural-log units or in `log10` units before logistic calibration |
| `--anchor-aggregation` | choose how multiple anchor traits are combined: `multi` is the default shared multi-trait mode, while `any` uses explicit noisy-OR union; with one anchor, both reduce exactly to single-trait anchoring |
| `--gene-gene-diagonal-weight` | set the diagonal fitting weight in the symmetric objective; the default `0.0` suppresses self-pairs |
| `--gene-gene-excess-probability` / `--no-gene-gene-excess-probability` | factor excess pairwise probability above the pair prior (default) or the raw calibrated pairwise probability |
| `--gene-gene-row-sum-cap` / `--no-gene-gene-row-sum-cap` | keep each gene’s mechanism memberships approximately disjoint by capping the row sum of `W` at `1` after each update |
| `--gene-gene-sparsity` | optional L1 penalty on the symmetric gene-by-gene loading matrix |
| `--gene-gene-profligate-correction` | optional correction for profligate genes before pair-probability calibration; `none` is the default and `linear` subtracts a simple retained-annotation-count effect from raw pair log evidence |
| `--annotation-bridge-metrics-out` | write per-annotation bridge diagnostics for the retained gene-by-gene evidence kernel, including review candidates and the conservative source-adaptive suggested-exclude flag |
| `--annotation-bridge-suggested-exclude-out` | write annotation IDs whose bridge diagnostics pass the conservative suggested-exclusion flag |
| `--gene-factor-annotation-contribs-out` | write top annotation contributions explaining gene-factor loadings |
| `--gene-factor-annotation-contribs-top-n` | cap contribution rows per gene-factor-anchor combination; defaults to `10` |

Notes:

- The discovery similarity threshold is denoted `rho_disc` in the methods documentation. The default is `0.35` for the gene-by-gene-set discovery model.
- The current default weighting mode is `effective_size`: each family leader uses representative support multiplied by the family effective size, `|F| / (1 + (|F|-1) * mean_similarity_to_leader)`.
- `log_effective_size` remains the conservative fallback, and `none` disables redundancy weighting entirely.
- `--no-discovery-redundancy-weighting` is a compatibility shortcut for `--discovery-redundancy-weighting-mode none`.
- `--no-auto-discovery-subset` currently disables weighted leader-family corrections and falls back to unweighted retained-row discovery.
- `--discovery-model gene_by_gene` uses all retained gene sets to construct pairwise gene evidence, so discovery-family subsetting and redundancy weighting flags are ignored in that mode.
- `--discovery-model gene_by_gene` currently requires `--factor-backend full`, `--learn-phi-backend sentinel_pruned`, and the default transposed factor matrix.
- In `gene_by_gene` mode, corrected `beta` is the default pairwise evidence surface and no additional gene-set-size normalization is applied before symmetric factorization.
- `--gene-gene-profligate-correction linear` is opt-in. It fits `L_ij = a + b * (log1p(c_i) + log1p(c_j))` on off-diagonal pairs, where `c_i` is the retained/scored annotation count entering the gene-gene evidence matrix, subtracts the fitted degree effect from `L_ij`, and then applies the same pair-prior calibration and target construction as the default path.
- `--annotation-bridge-metrics-out` is post-factorization and output-only. It decomposes the fitted gene-by-gene evidence kernel into each retained annotation's exact rank-one contribution, summarizes within-factor and between-factor bridge mass, and reports source-adaptive review/exclusion diagnostics using empirical source/global bridge percentiles rather than hardcoded source names.
- `--gene-factor-annotation-contribs-out` can be large; by default it writes only the top `10` positive annotation contributions per gene-factor-anchor combination.
- For multiple anchor traits in `gene_by_annotation` mode, `--anchor-aggregation multi` uses `sum_t p_i,t q_j,t` as the cell weight. `--anchor-aggregation any` uses cell-level noisy-OR, `1 - product_t (1 - p_i,t q_j,t)`, so gene support from one trait and annotation support from another do not create cross-trait weight.
- For multiple anchor traits in `gene_by_gene` mode, EAGGL first constructs each per-trait pair target exactly as it would for a single-trait run. `multi` uses the equal average of all per-trait targets, so zero targets dilute support; `any` uses noisy-OR union when support from any anchor should be sufficient.
- No hidden any pseudo-anchor is appended during fitting; `any` is used only when explicitly requested with `--anchor-aggregation any`.
- By default, `factors.out` and cluster outputs print only primary factors, defined by `combined_mass_fraction >= 0.005`. Use `--factor-output-scope primary_secondary` to include factors with `combined_mass_fraction >= 0.001`, or `--factor-output-scope all` to audit the full ARD tail.
- Automatic phi-selection metrics are also primary-scoped by default: redundancy and repeat-stability matching ignore the low-mass ARD tail unless `--learn-phi-metric-factor-scope all` is set. This option is separate from `--factor-output-scope`, which only controls printed factor and cluster rows.
- `factor_metrics.out` and `factor_phi_metrics.out` remain exhaustive over all raw fitted factors. The optional `factor_phi_*` output tables follow `--factor-output-scope`, matching the final user-facing output policy for each tested phi.
- Factor labels are not renumbered when output filtering is active: if `Factor7` is primary, it remains `Factor7`.

### Factor-PheWAS controls

| Flag | Meaning |
|---|---|
| `--factor-phewas-mode` | choose the factor-PheWAS model class; default is marginal binary enrichment with direct anchor adjustment |
| `--factor-phewas-modes` | expert comma-separated list of model classes to run in one pass and append into the same factor-PheWAS output file |
| `--factor-phewas-anchor-covariate` | choose the anchor covariate for factor-PheWAS; default is `direct`, with `combined` and `none` as expert options |
| `--factor-phewas-thresholded-combined-cutoff` | cutoff used to define thresholded phenotype hits for the binary factor-PheWAS modes |
| `--factor-phewas-se` | choose the uncertainty estimator for factor-PheWAS; default is robust |
| `--factor-phewas-min-gene-factor-weight` | minimum gene-factor weight kept for legacy continuous factor-PheWAS modes |
| `--factor-phewas-full-output` | write the broader expert factor-PheWAS surface in addition to the default binary output |
| `--threshold-weights` | threshold very small weights during post-processing |

Operational notes:
- Public default factor-PheWAS:
  - `--factor-phewas-mode marginal_anchor_adjusted_binary`
  - `--factor-phewas-anchor-covariate direct`
  - `--factor-phewas-se robust`
- To compare multiple model surfaces in one run, use:
  - `--factor-phewas-modes marginal_anchor_adjusted_binary,joint_anchor_adjusted_binary`
  - each requested model is appended into one `factor_phewas_stats.out` table with explicit model-identifying columns
- Expert binary modes:
  - `marginal_unconditional_binary`
  - `joint_anchor_adjusted_binary`
- Legacy continuous modes remain available for compatibility:
  - `legacy_continuous_direct`
  - `legacy_continuous_combined`
- Combined anchor adjustment is expert-only because both the binary outcome and the combined anchor covariate inherit gene-set-mediated indirect structure; coefficient estimates can still be useful, but p-values are more approximate.

### Labeling and optional LLM integration

| Flag | Meaning |
|---|---|
| `--lmm-provider` | choose the optional LLM provider used for factor labeling |
| `--lmm-model` | choose the optional LLM model used for factor labeling |
| `--lmm-auth-key` | provider credential used for optional labeling |
| `--label-gene-sets-only` | label from gene-set content only |
| `--label-include-phenos` | include phenotype context in labeling prompts |
| `--label-individually` | label factors independently instead of in one batch |
| `--gene-sets-for-labeling` | restrict gene-set candidates used for factor labels to identifiers from one or more files; the flag may be repeated |
| `--gene-sets-for-labeling-id-col` | optional header column for `--gene-sets-for-labeling`; by default the first column is read, so GMT files can be passed directly |

If `--gene-sets-for-labeling` is supplied, EAGGL filters gene-set label candidates before ranking top terms for each factor. If no supplied identifier has positive loading for a factor, EAGGL emits a warning and falls back to the unfiltered gene-set candidates for that factor.

Labeling details and the rationale for keeping labeling integrated into `factor` are documented in `docs/eaggl/LABELING.md`.

### Outputs

| Flag | Meaning |
|---|---|
| `--factors-out` | main factor output table |
| `--gene-set-clusters-out` | gene-set cluster output |
| `--gene-clusters-out` | gene cluster output |
| `--trait-factor-links-out` | canonical long-form trait-factor linkage output |
| `--factor-phewas-stats-out` | factor-level PheWAS output |
| `--gene-pheno-stats-out` | gene-phenotype output |
| `--consensus-stats-out` | per-run and per-factor diagnostics for restart or consensus factorization |
| `--params-out` | params and diagnostics output |

## Relationship to the theory doc

The mathematical model and workflow formalization live in:
- `docs/eaggl/methods.tex`

For post-factor phenotype interpretation:
- `trait_factor_links.out` is the primary phenotype annotation artifact
- `trait_factor_links.out` writes `nnls_loading`, which projects each selected trait support vector onto the EAGGL factor gene-loading basis with NNLS
- `trait_neff = (\sum_g s_t(g))^2 / \sum_g s_t(g)^2` reports the support-weighted effective number of genes contributing to the thresholded trait signal
- `factor_num_genes` and `factor_weight_sum` describe the thresholded factor gene set used for linkage
- `trait_response_source` records the selected phenotype support column, usually `combined`
- `--factor-gmt-out` exports the same thresholded factor gene sets for a PIGEAN `multi-y` run; factor-to-trait regression statistics such as `beta`, `beta_uncorrected`, `beta_tilde`, `se`, `z`, and `p_value` are intentionally produced by PIGEAN, not by EAGGL
- `factor_total_mass` in `factors.out` reports the raw total mass of each factor on the canonical linkage basis used for that run
- `factor_tier` and `combined_mass_fraction` in `factors.out` report the post-fit interpretability tier next to `lambda`; the default user-facing outputs include only primary factors unless `--factor-output-scope` is widened
- `factor_n_eff`, `factor_top_share`, `factor_top10_share`, and `broad_factor_flag` in `factors.out` summarize factor breadth on the retained projection basis; `broad_factor_flag` marks factors with `factor_n_eff >= 500` and `factor_top_share <= 0.01`
- `--gene-stats-in` / `--gene-set-stats-in` runs treat the input statistics as an implicit `input_gene_stats` anchor
- for factor-to-trait interpretation in EAGGL, rank by `nnls_loading`; for regression-style trait association strength, export `--factor-gmt-out` and run PIGEAN `multi-y`
- `pheno_clusters.out` remains accepted as a compatibility alias for one release and writes the same long-form canonical linkage payload
- `factor_phewas_stats.out` is a secondary enrichment table rather than the main phenotype-labeling surface

Use this split:
- `docs/eaggl/CLI_REFERENCE.md`: how to run EAGGL and what the main flags do
- `docs/eaggl/WORKFLOWS.md`: workflow-by-workflow command patterns
- `docs/eaggl/methods.tex`: theory and model formalization
- `docs/eaggl/LABELING.md`: optional labeling behavior and provider usage
- `docs/eaggl/CLI_OPTIONS.md`: exhaustive generated inventory

## Testing expectations for this reference

This document is intentionally smaller than the full parser surface.

Current reference tests should cover:
- help and routing behavior: `tests/eaggl/test_eaggl_cli_unittest.py`
- workflow ID selection and bundle defaults: `tests/eaggl/test_eaggl_cli_unittest.py`
- curated EAGGL CLI reference coverage: `tests/eaggl/test_eaggl_cli_reference_unittest.py`
- generated manifest freshness: `tests/eaggl/test_cli_manifest_unittest.py`

### Simplified Trait-Factor Linkage

`trait_factor_links.out(.gz)` now contains only native EAGGL NNLS factor-to-trait projections. `nnls_loading` projects each trait support vector onto the EAGGL factor gene-loading basis after zeroing factor gene loadings below `--trait-factor-linkage-factor-gene-threshold` (default `0.05`). Use `--factor-gmt-out` to export the same thresholded factor gene sets for an external PIGEAN `multi-y` run when factor-as-gene-set beta statistics are needed.

Cluster files report raw `Factor*` loadings, `Cosine_Factor*`, and `Euclidean_Factor*`. Cosine is the row loading vector's cosine similarity to the one-hot factor indicator; Euclidean is the row loading vector's distance to that indicator. The previous `Relative_Factor*` and `Combined_Factor*` cluster columns are no longer written.

`--factor-top-loading-type` controls the metric used to choose top entities for `factors.out` summaries and labels. It defaults to `euclidean`; valid values are `euclidean`, `cosine`, and `raw`.
