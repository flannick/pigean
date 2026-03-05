# Advanced Set B Workflows

This page documents the retained advanced workflows in `src/pigean.py`.
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

## 4) Optional gene-level PheWAS output (`--run-phewas-from-gene-phewas-stats-in`)

Purpose: Produce gene-level PheWAS summary output from precomputed gene-by-phenotype statistics.

Required inputs:
- Main mode run that computes input features (commonly `beta_tildes` or later modes)
- `--run-phewas-from-gene-phewas-stats-in <file>`
- Column mappings:
  - `--gene-phewas-bfs-id-col`
  - `--gene-phewas-bfs-pheno-col`
  - one or more of `--gene-phewas-bfs-log-bf-col`, `--gene-phewas-bfs-combined-col`, `--gene-phewas-bfs-prior-col`
- `--phewas-stats-out <file>`

Primary outputs:
- `--phewas-stats-out` table.

Notes:
- This is distinct from factor-based PheWAS (moved to `eaggl`).
- Runtime now logs one explicit I/O decision for this stage:
  - `mode=reuse_loaded_matrix` when loaded matrix state can be reused safely.
  - `mode=re_read_file` when the matrix must be re-read (for example when not preloaded or filtered).

## 5) Simulation mode (`sim`)

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

## 6) PoPS-style prior modes (`pops`, `naive_pops`)

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

The historical full-GLS/whitened-Y path is no longer supported in `src/pigean.py`.

- Removed aliases now hard-fail:
  - `--run-gls` / `run_gls`
  - `store_cholesky`
- Supported linear path is corrected OLS by default; use `--ols` to disable correlation correction.
