# EAGGL CLI Reference

This is the canonical human-written manual for the stable, routinely used `python -m eaggl` command surface.

Use this document for practical command shapes, workflow selection, and the meaning of the main EAGGL flags.
Use `docs/eaggl/CLI_OPTIONS.md` for the exhaustive machine-generated parser inventory.
Use `README.md` for the full repository documentation map.

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
4. optionally project phenotypes or run factor-PheWAS
5. optionally label the factors

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
  --learn-phi-report-out results/phi_search.tsv \
  --factors-out results/factors.out \
  --gene-set-clusters-out results/gene_set_clusters.out \
  --gene-clusters-out results/gene_clusters.out \
  --params-out results/params.out
```

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
  --pheno-clusters-out results/pheno_clusters.out \
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
- `F2`: positive-control gene-list anchoring
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
| `--positive-controls-in` | read positive-control genes from a file |
| `--positive-controls-list` | provide positive-control genes directly on the command line |
| `--positive-controls-all-in` | provide the background gene universe for positive-control workflows |
| `--anchor-phenos` | anchor to one or more named phenotypes |
| `--anchor-any-pheno` | anchor to an aggregate any-phenotype signal |
| `--anchor-genes` | anchor to one or more genes |
| `--anchor-any-gene` | anchor to an aggregate any-gene signal |
| `--anchor-gene-set` | anchor to the input gene set itself |

Notes:
- `--positive-controls-list` expects a comma-separated list, not a file path
- anchored phenotype and gene workflows generally require PheWAS inputs in addition to the anchor flag itself

### PheWAS and projection inputs

| Flag | Meaning |
|---|---|
| `--gene-phewas-stats-in` | load gene-by-phenotype statistics |
| `--gene-set-phewas-stats-in` | load gene-set-by-phenotype statistics |
| `--run-phewas-from-gene-phewas-stats-in` | run a gene-level PheWAS stage from precomputed gene-PheWAS stats; also required by the gene-set-anchored workflow |
| `--factor-phewas-from-gene-phewas-stats-in` | compute factor-level PheWAS from precomputed gene-PheWAS stats |
| `--project-phenos-from-gene-sets` | project phenotype loadings from gene-set scores instead of gene scores |

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
| `--learn-phi-max-redundancy` | maximum within-run weighted Jaccard overlap allowed between retained factors in the selected solution |
| `--learn-phi-runs-per-step` | number of restart fits used to score each tested `phi` candidate |
| `--learn-phi-min-run-support` | minimum fraction of restart runs that must agree on the modal retained factor count |
| `--learn-phi-min-stability` | minimum mean matched-factor cosine across the modal restart runs |
| `--learn-phi-max-fit-loss-frac` | maximum allowed reconstruction-error loss relative to the best candidate tested |
| `--learn-phi-max-steps` | maximum number of log-space search steps after bracketing the redundancy transition |
| `--learn-phi-expand-factor` | multiplicative factor used when widening the search bracket away from the initial `--phi` |
| `--learn-phi-weight-floor` | factor weights below this are treated as zero when computing redundancy |
| `--learn-phi-report-out` | optional per-candidate diagnostics table for all tested `phi` values |

Operational notes:
- `--phi` remains the initial guess. With `--learn-phi`, EAGGL treats it as the starting point for search rather than the final fixed value.
- Auto-tuning uses `--learn-phi-runs-per-step` during search, then runs the normal final factorization with the selected `phi`.
- The default search is structural model selection, not held-out cross-validation. It prefers the smallest acceptable `phi` that keeps factors non-redundant, stable across restarts, and close to the best fit seen during search.

### Factor pruning, weighting, and post-processing

| Flag | Meaning |
|---|---|
| `--factor-prune-gene-sets-num` / `--factor-prune-gene-sets-val` | prune weak gene-set memberships from factor outputs |
| `--factor-prune-genes-num` / `--factor-prune-genes-val` | prune weak gene memberships from factor outputs |
| `--factor-prune-phenos-num` / `--factor-prune-phenos-val` | prune weak phenotype memberships from factor outputs |
| `--factor-phewas-min-gene-factor-weight` | minimum gene-factor weight kept for factor-PheWAS |
| `--threshold-weights` | threshold very small weights during post-processing |

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
| `--pheno-clusters-out` | phenotype cluster output |
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
