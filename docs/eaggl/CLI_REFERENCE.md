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
- `factor`: canonical EAGGL factor workflow with F1-F9 workflow selection
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
  --learn-phi-max-num-iterations 50 \
  --learn-phi-report-out results/phi_search.tsv \
  --factors-out results/factors.out \
  --gene-set-clusters-out results/gene_set_clusters.out \
  --gene-clusters-out results/gene_clusters.out \
  --params-out results/params.out
```

`--params-out` is the resolved run record. For factor runs it writes the effective factor configuration, including restart and consensus settings, anchor/filter choices, labeling settings, the final `phi` used for fitting, and any `--learn-phi` search diagnostics. Current `--learn-phi` diagnostics include the redundancy basis, candidate capped-status, nearest-neighbor overlap summaries (`redundancy_max` and `redundancy_q90`), factor-mass summaries (`effective_factor_count` and `mass_ge_floor_factor_count`), and convergence diagnostics such as final `delambda` and whether the scout fit hit the iteration cap.

Scalable backend note:
- `--factor-backend full|blockwise_global_w` selects the final factorization backend.
- `--learn-phi-backend sentinel_pruned|blockwise_global_w` selects the phi-search backend.
- `blockwise_global_w` keeps one shared global gene-factor basis and one shared ARD state across block-local gene-set solves, so all retained gene sets can contribute during phi search and final factorization without requiring one giant in-memory solve.

Phenotype-anchored workflow:

```bash
PYTHONPATH=src python -m eaggl factor \
  --X-in bundles/current/model_small/data/gene_set_list_mouse_2024.txt \
  --anchor-phenos T2D,T2D_ALT \
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
- `F4`: explicit phenotype anchoring
- `F5`: any-phenotype anchoring
- `F6`: single-gene anchoring
- `F7`: multi-gene anchoring
- `F8`: any-gene anchoring
- `F9`: gene-set anchoring

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

### Workflow selectors and anchors

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
| `--anchor-phenos` | anchor to one or more named phenotypes |
| `--anchor-any-pheno` | anchor to an aggregate any-phenotype signal |
| `--anchor-genes` | anchor to one or more genes |
| `--anchor-any-gene` | anchor to an aggregate any-gene signal |
| `--anchor-gene-set` | anchor to the input gene set itself |

Notes:
- `--gene-list-in` / `--gene-list` are the primary standalone EAGGL workflow selectors
- standalone gene-list mode uses the loaded X-gene universe as the enrichment background
- retained standalone gene sets are weighted by `-log(P) / sqrt(gene_set_size)` and filtered by `--gene-list-max-fdr-q`
- `--gene-list` expects a comma-separated list, not a file path
- `--positive-controls-in` / `--positive-controls-list` remain compatibility aliases for the standalone gene-list workflow and are hidden from the primary help surface
- anchored phenotype and gene workflows generally require PheWAS inputs in addition to the anchor flag itself

### PheWAS and projection inputs

| Flag | Meaning |
|---|---|
| `--gene-phewas-stats-in` | load gene-by-phenotype statistics |
| `--gene-set-phewas-stats-in` | load gene-set-by-phenotype statistics |
| `--run-phewas` | run a gene-level PheWAS stage from `--gene-phewas-stats-in`; also required by the gene-set-anchored workflow |
| `--run-factor-phewas` | compute factor-level phenotype enrichment regression from `--gene-phewas-stats-in` |
| `--factor-gene-clusters-in` | load an existing `gene_clusters.out(.gz)` factor loading table and run projection-only canonical trait linkage, factor-PheWAS, or both without refitting factors |
| `--factor-gene-set-clusters-in` | load an existing `gene_set_clusters.out(.gz)` factor loading table for projection-only canonical trait linkage from the gene-set basis |
| `--factor-phewas-gene-clusters-in` | compatibility alias for the older factor-PheWAS-only projection command |
| `--project-phenos-from-gene-sets` | compute canonical trait linkage on the gene-set basis instead of the gene basis |
| `--pheno-capture-input` | choose whether canonical trait linkage uses retained weighted thresholded support or binary thresholded hits |
| `--trait-factor-links-out` | write the canonical long-form trait-factor linkage table |
| `--trait-linkage-source` | choose the support surface for canonical trait linkage; default is `combined` (expert overrides: `auto`, `log_bf`, `prior`) |
| `--trait-linkage-threshold` | strict threshold for canonical trait linkage support (`source_value > threshold`) |
| `--trait-linkage-computation-mode` | choose the linkage computation backend: `sparse_full` by default, or `dense_full` as a debug comparison backend |
| `--trait-factor-links-output-detail` | choose `trait_factor_links.out` column detail: `main` for concise coefficients, `full`/`debug` for retained-support diagnostics |
| `--no-trait-linkage` | disable canonical trait linkage even when trait inputs are available |
| `--factor-phewas-modes` | expert override: run multiple factor-PheWAS model surfaces in one pass and append them into one output table |
| `--factor-phewas-full-output` | expose the full expert factor-PheWAS surface, including combined and Huber variants |

Operational notes:
- canonical trait linkage is the primary user-facing phenotype annotation layer and is interpreted as support-normalized trait-factor projection coefficients, not calibrated posterior probability or exact captured-support mass
- canonical linkage writes one long table with one row per `(trait, factor)` and reports both `marginal_coefficient` and `joint_coefficient` from the same internal matching inputs: `marginal_coefficient` is the one-factor bounded projection and `joint_coefficient` is the all-factor constrained projection
- `--trait-factor-links-output-detail main` is the default concise schema: `trait`, `factor`, `is_anchor`, `joint_fraction`, `marginal_fraction`, `marginal_overlap`, `joint_support_mass`, `marginal_support_mass`, `marginal_overlap_support_mass`, `low_retention_flag`, `trait_neff`, and `retained_n_eff`; use `full` or `debug` to include additional retained-support diagnostics and explicit coefficient names
- the target profile is normalized by total thresholded trait strength before masking, not by retained masked strength
- raw trait support and raw factor loadings are not required to sum to `1`; only copied internal vectors are normalized for matching
- trait linkage operates on the thresholded phenotype support file, not on a fully observed unthresholded phenotype surface
- `--pheno-capture-input weighted_thresholded` is the default and uses retained source-support values that strictly exceed `--trait-linkage-threshold`; `binary_thresholded` is an expert sensitivity mode
- canonical linkage defaults to `--trait-linkage-source combined` and `--trait-linkage-threshold 1.0` (strict `> 1.0`)
- canonical linkage defaults to `--trait-linkage-computation-mode sparse_full`; `dense_full` remains available as an expert/debug comparison backend without changing the corrected linkage math
- if `--trait-linkage-source auto` is requested, one support surface is chosen per run in the order `combined`, then `log_bf`, then `prior`
- factor-PheWAS is a secondary expert analysis for factor-specific phenotype enrichment
- the default factor-PheWAS mode is `marginal_anchor_adjusted_binary`, which regresses thresholded phenotype-hit membership on one factor at a time while adjusting for direct anchor support
- projection-only gene-basis phenotype clusters and factor-PheWAS use the raw `Factor1..FactorK` columns from `gene_clusters.out(.gz)` as the gene-factor loading matrix; any `combined`, `log_bf`, or `prior` columns in that file are reused as anchor covariates unless overridden by `--gene-stats-in`
- projection-only gene-basis trait linkage requires `--gene-phewas-stats-in` and writes the long-form `trait_factor_links.out(.gz)` table
- projection-only gene-set-basis trait linkage uses the raw `Factor1..FactorK` columns from `gene_set_clusters.out(.gz)` with `--gene-set-phewas-stats-in`; request this with `--project-phenos-from-gene-sets`
- `--pheno-clusters-out` remains accepted as a compatibility alias for one release and writes the same long-form canonical linkage payload
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
| `--phi` | primary sparsity / concentration control for the factor model |
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
| `--learn-phi-max-redundancy` | maximum within-run weighted Jaccard overlap allowed between retained factors in the selected solution, measured on gene loadings when available; the default `0.5` is intended as a rough \"share at most about half\" rule |
| `--learn-phi-runs-per-step` | number of restart fits used to score each tested `phi` candidate; defaults to `1` for cheaper search, with larger values as an expert stability check |
| `--learn-phi-min-run-support` | minimum fraction of restart runs that must agree on the modal retained factor count |
| `--learn-phi-min-stability` | minimum mean matched-factor cosine across the modal restart runs |
| `--learn-phi-max-fit-loss-frac` | legacy fallback fit-loss guard used only when no candidate satisfies the primary redundancy/restart criteria |
| `--learn-phi-max-steps` | maximum number of additional `phi` candidates to evaluate after the initial `--phi`; defaults to `5` |
| `--learn-phi-backend` | choose between the legacy sentinel-pruned phi search and the blockwise-global-W phi search over all retained gene sets |
| `--learn-phi-expand-factor` | multiplicative factor used when widening the search bracket away from the initial `--phi` |
| `--learn-phi-weight-floor` | factor weights below this are treated as zero when computing redundancy |
| `--learn-phi-report-out` | optional per-candidate diagnostics table for all tested `phi` values |
| `--learn-phi-prune-genes-num` | expert shortcut: during phi search only, correlation-prune to at most this many representative genes before candidate NMF evaluation; defaults to `1000` |
| `--learn-phi-prune-gene-sets-num` | deprecated expert knob retained for compatibility; phi search now uses the same discovery plan as the final fit and this option is ignored |
| `--learn-phi-max-num-iterations` | expert shortcut: during phi search only, cap candidate NMF iterations separately from the final factorization |

Operational notes:
- `--phi` remains the initial guess. With `--learn-phi`, EAGGL treats it as the starting point for search rather than the final fixed value.
- Auto-tuning uses `--learn-phi-runs-per-step` during search, then runs the normal final factorization with the selected `phi`.
- The default search uses one restart per candidate `phi`. Increase `--learn-phi-runs-per-step` only when you explicitly want a more expensive restart-stability check during selection.
- The selected `phi`, the search thresholds, the redundancy basis, and per-candidate diagnostics are written to both the run log and `--params-out`. Use `--learn-phi-report-out` when you also want the full candidate table as a separate artifact.
- The default search is structural model selection, not held-out cross-validation. It now gates on redundancy and restart behavior, then chooses `phi` from the resulting fit/complexity frontier using the marginal improvement in reconstruction error per additional retained mechanism, rather than a near-maximal factor-count band.
- Candidate complexity is summarized using both the raw retained factor count and factor-mass diagnostics. In particular, `effective_factor_count` is the inverse-participation-ratio style mass summary `(sum m_k)^2 / sum m_k^2`, and `mass_ge_floor_factor_count` counts factors whose mass fraction is at least `1%`.
- The search report also records whether the scout factorization converged before the candidate iteration cap, via `final_delambda`, `final_iterations`, `converged_fraction`, and `hit_iteration_cap_fraction`.
- By default, `--learn-phi-backend sentinel_pruned` evaluates candidates on a lightweight sentinel panel. `--learn-phi-prune-genes-num` still affects that gene-side shortcut, but gene-set discovery now follows the same retained-to-discovery plan used by the final fit.
- `--learn-phi-backend blockwise_global_w` instead evaluates all retained gene sets in blocks while keeping one shared global gene-factor basis and one shared ARD state.
- The default search budget is `5` additional candidate evaluations after the initial `--phi`. Those evaluations may be spent on upward expansion, downward expansion, or midpoint refinement once the search brackets a capped or zero-factor boundary.
- `--learn-phi-prune-genes-num` and `--learn-phi-max-num-iterations` apply only while scoring phi candidates. Gene-set discovery selection itself is shared between learn-phi and the final reported factorization.

### Factor pruning, weighting, and post-processing

| Flag | Meaning |
|---|---|
| `--max-num-discovery-gene-sets` | cap the number of discovery family leaders used to learn the latent basis `W`; all retained annotations are still projected afterward |
| `--no-auto-discovery-subset` | disable the default family-leader discovery subset and instead fit discovery on all retained gene sets |
| `--discovery-redundancy-weighting-mode` | choose leader-family discovery weighting: `effective_size`, `log_effective_size`, or `none` |
| `--no-discovery-redundancy-weighting` | disable the default redundancy-balanced discovery weighting |
| `--discovery-redundancy-threshold` | similarity threshold used when assigning retained gene sets to discovery families; defaults to `0.35` |
| `--factor-prune-gene-sets-num` / `--factor-prune-gene-sets-val` | deprecated factor-stage discovery controls kept only as compatibility aliases; use the discovery flags above instead |
| `--factor-prune-genes-num` / `--factor-prune-genes-val` | prune weak gene memberships from factor outputs |
| `--factor-prune-phenos-num` / `--factor-prune-phenos-val` | prune weak phenotype memberships from factor outputs |
| `--factor-backend` | choose the final factorization backend: `full` or `blockwise_global_w` |
| `--blockwise-gene-set-block-size` | set the retained gene-set block size used by `blockwise_global_w` |
| `--blockwise-epochs` | number of global block passes used by `blockwise_global_w` |
| `--blockwise-shuffle-blocks` | shuffle gene-set block order between blockwise epochs |
| `--blockwise-warm-start` | warm-start neighboring phi candidates in blockwise phi search |
| `--blockwise-max-blocks` | optional debugging cap on the number of processed blocks per epoch |

Notes:

- The current default weighting mode is `effective_size`. `log_effective_size` remains the conservative fallback, and `none` disables redundancy weighting entirely.
- `--no-discovery-redundancy-weighting` is a compatibility shortcut for `--discovery-redundancy-weighting-mode none`.
- `--no-auto-discovery-subset` currently disables weighted leader-family corrections and falls back to unweighted retained-row discovery.
| `--blockwise-report-out` | write per-epoch blockwise diagnostics |
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

Labeling details and the rationale for keeping labeling integrated into `factor` are documented in `docs/eaggl/LABELING.md`.

### Outputs

| Flag | Meaning |
|---|---|
| `--factors-out` | main factor output table |
| `--factors-anchor-out` | anchor-specific factor output |
| `--gene-set-clusters-out` | gene-set cluster output |
| `--gene-clusters-out` | gene cluster output |
| `--trait-factor-links-out` | canonical long-form trait-factor linkage output |
| `--pheno-clusters-out` | compatibility alias that writes the same canonical trait-factor linkage payload |
| `--gene-set-anchor-clusters-out` | anchor-side gene-set clusters |
| `--gene-anchor-clusters-out` | anchor-side gene clusters |
| `--pheno-anchor-clusters-out` | anchor-side phenotype clusters |
| `--factor-phewas-stats-out` | factor-level PheWAS output |
| `--gene-pheno-stats-out` | gene-phenotype output |
| `--consensus-stats-out` | per-run and per-factor diagnostics for restart or consensus factorization |
| `--params-out` | params and diagnostics output |

## Relationship to the theory doc

The mathematical model and workflow formalization live in:
- `docs/eaggl/methods.tex`

For post-factor phenotype interpretation:
- `trait_factor_links.out` is the primary phenotype annotation artifact
- `--trait-factor-links-output-detail main` writes `trait`, `factor`, `is_anchor`, `joint_fraction`, `marginal_fraction`, `marginal_overlap`, `joint_support_mass`, `marginal_support_mass`, `marginal_overlap_support_mass`, `low_retention_flag`, `trait_neff`, and `retained_n_eff`
- `--trait-factor-links-output-detail full` adds retained-support diagnostics, effective-size diagnostics, coefficient-scaled support totals, and joint residual
- raw trait support and raw factor loadings are not forced to sum to `1`; EAGGL preserves total support and total factor mass separately and only normalizes copied internal vectors for matching
- it reports both `marginal_coefficient` and `joint_coefficient` from the same internal matching step, with `marginal_coefficient` treating each factor alone and `joint_coefficient` letting all factors compete under a shared sum constraint
- `marginal_overlap = q_t^T b_k` is a direct shape-overlap metric; unlike `marginal_coefficient`, it is not divided by the factor self-norm and is less sensitive to factor breadth
- `trait_total_support` is the full thresholded trait support before masking, or the total thresholded hit count under binary capture mode
- `retained_trait_support` is the masked retained support available on the fitted factor basis
- `retained_fraction = retained_trait_support / trait_total_support`
- `trait_n_eff = (\sum_g s_t(g))^2 / \sum_g s_t(g)^2` reports the support-weighted effective number of genes contributing to the thresholded trait signal
- `retained_n_eff` applies the same effective-size calculation after masking to the factorized gene universe
- `trait_n_eff` and `retained_n_eff` are concentration diagnostics: they can be much smaller than `total_feature_count` or `retained_feature_count` when a small number of genes carries most of the support
- `joint_coefficient_support_mass = trait_total_support * joint_coefficient`, a coefficient-scaled support total rather than exact captured-support mass
- `marginal_coefficient_support_mass = trait_total_support * marginal_coefficient`, a coefficient-scaled support total rather than exact captured-support mass
- `marginal_overlap_support_mass = trait_total_support * marginal_overlap`, an overlap-scaled support total
- `total_feature_count` and `retained_feature_count` report the same diagnostic on thresholded feature counts
- `low_retention_flag` marks traits whose retained support is very sparse or highly concentrated on the current factor basis
- `joint_residual` is the uncaptured normalized trait mass after the joint competitive projection
- `factor_total_mass` in `factors.out` reports the raw total mass of each factor on the canonical linkage basis used for that run
- `factor_n_eff`, `factor_top_share`, `factor_top10_share`, and `broad_factor_flag` in `factors.out` summarize factor breadth on the retained projection basis; `broad_factor_flag` marks factors with `factor_n_eff >= 500` and `factor_top_share <= 0.01`
- `--gene-stats-in` / `--gene-set-stats-in` runs treat the input statistics as an implicit `input_gene_stats` anchor; `factors.out` writes `anchor_any_joint` / `anchor_any_marginal` only when both canonical anchor joint and marginal summaries are available; the legacy `any_relevance` alias is no longer emitted
- for factor-to-trait interpretation, filter on trait QC and rank by `joint_coefficient`, using `marginal_coefficient` and `marginal_overlap` as secondary context
- for trait-to-factor interpretation, rank by `joint_coefficient` rather than marginal alone, and inspect or filter broad factors with `broad_factor_flag`
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
