# Advanced Set B Workflows

This page documents the retained advanced workflows in `python -m pigean`.
Each block lists required inputs and expected outputs.

## 1) Precomputed gene statistics input (`--gene-stats-in`)

Purpose: Skip raw evidence loading and ingest gene-level scores directly.

Required inputs:
- Mode in main path (`beta_tildes`, `betas`, `priors`, `naive_priors`, or `gibbs`)
- Gene-set input (`--X-in` or `--X-list`)
- `--gene-stats-in <file>`
- Column mappings:
  - `--gene-stats-id-col`
  - `--gene-stats-log-bf-col`
  - optional `--gene-stats-combined-col`, `--gene-stats-prior-col`, `--gene-stats-prob-col`

Primary outputs:
- Standard main-path outputs (`--gene-set-stats-out`, `--gene-stats-out`, `--params-out`) for the selected mode.

Notes:
- This path bypasses raw `--gwas-in` / `--exomes-in` Y loading.
- In pure `betas` runs on large expanded X collections, two expert controls are available for decoupling cheap independent shrinkage from the expensive corrected-beta solve:
  - `--retain-all-beta-uncorrected`
    - keep real independent `beta_uncorrected` values for gene sets dropped only by `--max-num-gene-sets`
    - corrected `beta` remains limited to the capped retained subset
  - `--independent-betas-only`
    - skip the covariance-backed corrected-beta solve entirely
    - write only the cheap independent `beta_uncorrected` path
    - implies `--retain-all-beta-uncorrected`
- In `betas` and `gibbs`, `--track-filtered-beta-uncorrected-mode` controls which ignored rows still get tracked independent `beta_uncorrected` sidecars:
  - `cap_only` (default): only rows dropped by `--max-num-gene-sets`
  - `all`: every ignored row
  - `none`: disable tracked ignored-sidecar updates
- The boolean aliases `--track-filtered-beta-uncorrected` and `--no-track-filtered-beta-uncorrected` remain as compatibility shims for `all` and `none`.
- `--retain-all-beta-uncorrected` and `--independent-betas-only` still apply only to pure `betas` mode, not `priors` or outer `gibbs`.

## PIGEAN beta-only rerun bundles (`--pigean-rerun-bundle-out` / `--pigean-rerun-bundle-in`)

Purpose: rerun only the joint gene-set beta stage after annotation review, using the original run's fixed combined gene scores and active X matrix.

Initial full run:
- Run a normal PIGEAN workflow, usually `gibbs`.
- Request ordinary outputs plus:
  - `--pigean-rerun-bundle-out <bundle.tar.gz>`

Rerun:

```bash
PYTHONPATH=src python -m pigean betas \
  --pigean-rerun-bundle-in original.rerun_bundle.tar.gz \
  --gene-set-exclude-in annotation_bridge_suggested_exclude.txt \
  --gene-set-stats-out reduced.gene_set_stats.out.gz \
  --gene-stats-out reduced.gene_stats.out.gz \
  --eaggl-bundle-out reduced.eaggl_bundle.tar.gz \
  --params-out reduced.params.out.gz
```

Bundle contents:
- `X.tsv.gz`: active gene x annotation matrix after read/mapping/filtering.
- `gene_stats.tsv.gz`: original gene stats, including `combined`, used as fixed Y.
- `gene_universe.tsv.gz`: exact active analysis gene universe.
- `params.tsv.gz`: resolved parameter snapshot.
- `gene_set_stats.tsv.gz`: optional reference copy only; it is not loaded as beta input.
- `manifest.json`: schema, source argv, file hashes, column mappings, learned hyperparameters, and beta-stage defaults.

Semantics:
- `--pigean-rerun-bundle-in` is valid only with mode `betas`.
- The rerun does not run outer Gibbs.
- The rerun uses `Y = gene_stats[combined]`, not prior or direct log-BF columns.
- Annotation IDs listed in `--gene-set-exclude-in` are removed before beta-tildes and joint betas are recomputed.
- Bundle defaults are applied unless explicitly overridden by CLI/config.
- With no excluded annotations, rerun beta estimates should agree closely with the source beta estimates. Exact equality is not promised when the source was a Gibbs posterior summary.

Older runs may not have a rerun bundle. In that case, use the ordinary params file directly and provide the fixed Y and X inputs explicitly:

```bash
PYTHONPATH=src python -m pigean betas \
  --pigean-params-in original.params.out \
  --gene-stats-in original.gene_stats.out.gz \
  --gene-stats-id-col Gene \
  --gene-stats-combined-col combined \
  --gene-universe-from-y \
  --X-in library1.gmt \
  --X-in library2.gmt \
  --gene-set-exclude-in annotation_bridge_suggested_exclude.txt \
  --gene-set-stats-out reduced.gene_set_stats.out.gz \
  --gene-stats-out reduced.gene_stats.out.gz \
  --params-out reduced.params.out
```

`--pigean-params-in` replays learned `p` and `sigma2` rows from the params file and maps vector values back onto retained gene sets by annotation-library label. It also defaults to `--update-hyper none` unless the user explicitly overrides it. This is a parameter replay mechanism, not an input bundle; the fixed Y input, gene universe, and X inputs still need to be supplied.

## 2) Precomputed gene-set statistics input (`--gene-set-stats-in`)

Purpose: Reuse precomputed gene-set association statistics instead of recomputing from X and Y.

Required inputs:
- Mode requiring gene-set stats (`beta_tildes`, `betas`, `priors`, `naive_priors`, `gibbs`)
- `--gene-set-stats-in <file>`
- `--gene-set-stats-id-col`
- At least one metric column mapping, usually:
  - `--gene-set-stats-beta-tilde-col`
  - optionally `--gene-set-stats-beta-col`, `--gene-set-stats-beta-uncorrected-col`, `--gene-set-stats-se-col`, `--gene-set-stats-p-col`

Primary outputs:
- Downstream mode outputs (for example, betas, priors, or Gibbs outputs) using ingested gene-set statistics.

Notes:
- Rows not present in currently loaded gene sets are ignored.

## 3) HuGE cache write/read (`--huge-statistics-out` / `--huge-statistics-in`)

Purpose: Cache expensive HuGE preprocessing and replay it quickly.

Required inputs (cache write):
- `--gwas-in <sumstats>`
- HuGE mapping inputs (for example `--gene-loc-file-huge`, optional S2G/credible-set flags as needed)
- `--huge-statistics-out <prefix-or-tar>`

Required inputs (cache read):
- `--huge-statistics-in <prefix-or-tar>`
- Same downstream mode inputs (X inputs and mode flags) as normal run

Primary outputs:
- Cache artifacts on write
- Normal mode outputs on read

Notes:
- Use `--deterministic` (or fixed `--seed`) for cache-vs-raw parity checks.

## 4) Optional gene-level PheWAS output (`--run-phewas`)

Purpose: Produce gene-level PheWAS summary output from precomputed gene-by-phenotype statistics.

Required inputs:
- Main mode run that computes input features (commonly `beta_tildes` or later modes)
- `--run-phewas`
- `--gene-phewas-stats-in <file>`
- optional: `--phewas-comparison-set matched|diagnostic`
- Column mappings:
  - `--gene-phewas-stats-id-col`
  - `--gene-phewas-stats-pheno-col`
  - `--gene-phewas-stats-log-bf-col` for direct phenotype support
  - `--gene-phewas-stats-combined-col` for combined phenotype support
- `--phewas-stats-out <file>`

Primary outputs:
- `--phewas-stats-out` table.

Notes:
- This is distinct from factor-based PheWAS (moved to `eaggl`).
- Runtime logs one explicit I/O decision for this stage before running the output step.
- Default `--phewas-comparison-set matched` writes only the two matched comparisons:
  - `pheno_Y_vs_input_Y`
  - `pheno_combined_prior_Ys_vs_input_combined_prior_Ys`
- `--phewas-comparison-set diagnostic` additionally enables the four cross-family contrasts:
  - `pheno_Y_vs_input_combined_prior_Ys`
  - `pheno_Y_vs_input_priors`
  - `pheno_combined_prior_Ys_vs_input_Y`
  - `pheno_combined_prior_Ys_vs_input_priors`
- The phenotype-side prior family is intentionally not part of this stage.
- Only the combined phenotype-support family receives the sparse residual-correlation correction.
- The later non-infinitesimal shrinkage step still uses the independent approximation.
- If the stage cannot reuse an already loaded sparse matrix, it now stages the requested file once and slices phenotype batches from that staged sparse representation instead of rereading the raw file for every batch.

Decision table:

| Requested input | Loaded gene-PheWAS matrix | Filtered after load | Runtime mode | Reason | File is re-read |
| --- | --- | --- | --- | --- | --- |
| no | yes or no | yes or no | `skip` | `no_input_requested` | no |
| yes | no | no | `re_read_file` | `matrix_not_loaded` | yes |
| yes | yes | yes | `re_read_file` | `loaded_matrix_filtered` | yes |
| yes, same normalized path as loaded source | yes | no | `reuse_loaded_matrix` | `requested_input_matches_loaded_source` | no |
| yes, different source from loaded matrix | yes | no | `re_read_file` | requested input differs from loaded source | yes |

Operational notes:
- Reuse happens only when the requested file resolves to the same normalized path as one of the reusable loaded sources.
- Any post-load filtering of the loaded matrix disables reuse, because the output PheWAS stage expects the full requested matrix.
- The non-reuse path still reads the raw file once up front, but it no longer rereads it once per phenotype batch.
- The logging marker is the decision source of truth for debugging:
  - `mode=skip`
  - `mode=re_read_file`
  - `mode=reuse_loaded_matrix`
- The current regression coverage for this behavior lives in:
  - `tests/test_phewas_stage_reuse_unittest.py`
  - `tests/test_pegs_utils_bundle_unittest.py`

## 5) Native multi-Y trait batching (`--multi-y-in`)

Purpose: Run the current `pigean` package once per trait from a long-form multi-trait gene-statistics table, then append trait-labelled outputs into one aggregated file. This replaces the retired PheWAS-as-Y beta-sampling flags.

Supported modes:
- `betas`
- `gibbs`

Required inputs:
- Mode `betas` or `gibbs`
- Gene-set input (`--X-in` or `--X-list`)
- `--multi-y-in <file>`
- `--gene-set-stats-out <file>`
- Schema mapping:
  - optional `--multi-y-id-col` (default `Gene`)
  - optional `--multi-y-pheno-col` (auto-detects `Trait` then `Pheno`)
  - optional `--multi-y-log-bf-col` (auto-detects `log_bf` then `Direct`)
  - optional `--multi-y-combined-col` (auto-detects `combined` then `Combined`)
  - optional `--multi-y-prior-col` (auto-detects `prior` then `Prior`)

Optional expert controls:
- `--multi-y-max-phenos-per-batch <n>`
  - overrides the automatic trait chunk size
  - if omitted, PIGEAN estimates a trait batch size from `--max-gb`
- `--multi-y-vectorize-betas`
  - beta-mode only optimization
  - reads each trait batch once and maps traits to the existing parallel-run dimension of the beta sampler
  - preserves multiple chains within each trait
  - requires `--no-filter-negative`, because negative beta-tilde filtering is trait-specific in the ordinary per-trait X read path
  - requires disabled gene-set pruning (`--prune-gene-sets > 1` and `--weighted-prune-gene-sets > 1`), because pruning also occurs during X read
  - if hyperparameter updates are enabled, the update is shared across traits in the vectorized batch and PIGEAN emits a warning

Primary outputs:
- `--gene-set-stats-out`
  - aggregated across all requested traits
  - includes a `trait` column
- `--gene-stats-out`
  - only for `gibbs`
  - aggregated across all requested traits
  - includes a `trait` column
- `--params-out`
  - records the resolved multi-Y settings and completed trait count

Semantics:
- Each trait is materialized onto the current X gene universe and then run through the ordinary current-package `betas` or `gibbs` path.
- This is a native orchestration workflow, not a second legacy code path.
- If a single trait leaves no surviving gene sets after filtering, that trait is skipped and the rest of the batch continues.
- Because the long-form multi-Y inputs are continuous support vectors, the CLI auto-enables `--linear` unless you explicitly pass `--no-linear`.
- `--multi-y-in` cannot be combined with other primary Y sources such as `--gwas-in`, `--gene-stats-in`, or gene-list inputs.

Retired interface:
- The old `--betas-from-phewas` and `--betas-uncorrected-from-phewas` flags have been removed. Use `--multi-y-in` for multi-trait beta workflows.

## 6) Simulation mode (`sim`)

Purpose: Simulate gene and gene-set signal from configured hyperparameters.

Required inputs:
- Mode `sim`
- `--X-in` / `--X-list`
- Hyperparameters:
  - `--p-noninf`
  - one sigma input (`--sigma2-cond`, `--sigma2-ext`, or `--sigma2`)
  - `--sigma-power`
- Optional simulation controls:
  - `--sim-log-bf-noise-sigma-mult`
  - `--sim-only-positive`

Primary outputs:
- Standard outputs requested on CLI (`--gene-stats-out`, `--gene-set-stats-out`, `--params-out`).

Notes:
- `sim` is retained for testing and controlled benchmarking workflows.

## 7) PoPS-style prior modes (`pops`, `naive_pops`)

Purpose: Run PoPS-style settings on PIGEAN pipeline branches.

Required inputs:
- Mode `pops` or `naive_pops`
- Typical main-path inputs (`--X-in`/`--X-list`, plus gene evidence or precomputed gene stats)
- Optional overrides to mode defaults if needed

Primary outputs:
- `pops`: prior path outputs (for example gene priors, gene-set stats)
- `naive_pops`: naive-prior path outputs

Notes:
- Mode defaults are applied in `_apply_mode_and_runtime_defaults(...)`.
- Use `--print-effective-config` to inspect resolved defaults.

## Filter Relaxation Semantics

`--increase-filter-gene-set-p` is treated as a **minimum kept fraction** target during
prefiltering. If too few gene sets pass the current `--filter-gene-set-p`, PIGEAN
relaxes the threshold to keep at least that fraction.

Post-read filtering no longer tightens this threshold, so this option has one
canonical direction: avoid overly strict filtering.

## Removed Legacy GLS Path

The historical full-GLS/whitened-Y path is no longer supported in `python -m pigean`.

- Removed aliases now hard-fail:
  - `--run-gls` / `run_gls`
  - `store_cholesky`
- Supported linear path is corrected OLS by default; use `--ols` to disable correlation correction.
