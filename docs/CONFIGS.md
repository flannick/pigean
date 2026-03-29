# Config strategy

Use profile configs for static resources and defaults.

Users should provide:
1. `--config config/profiles/<profile>.default.json`
2. exactly one runtime input file (`--gwas-in`, `--exomes-in`, `--huge-statistics-in`, or `--gene-list-in`)

`config/profiles/common.factor.json` uses `__BUNDLE_ROOT__` placeholders.
After downloading bundles, replace `__BUNDLE_ROOT__` with your bundle root (usually `<repo>/bundles/current`).

For `--X-list` behavior parity with separate `--X-in`, each line should include explicit batch labels:

```text
mouse:/path/gene_set_list_mouse_2024.txt@mouse
msigdb:/path/gene_set_list_msigdb_nohp.txt@msigdb
```

- `label:` controls display label
- `@batch` controls hyperparameter pooling (p/sigma sharing)

Batching and per-input `p` behavior:

1. Each `--X-in`, `--X-list`, `--Xd-in`, or `--Xd-list` specification is a separate hyper-learning input by default.
2. Inputs that share the same `@batch` label share learned `p` and `sigma`.
3. `--batch-all-for-hyper` forces all unlabeled inputs into one shared hyper-learning batch.
4. `--first-for-hyper` learns hyperparameters on the first batch and reuses them for later unlabeled batches.
5. `--first-for-sigma-cond` fixes the learned `sigma2 / p` ratio from the first batch for later batches.
6. `--first-max-p-for-hyper` caps later learned `p` values at the first batch's learned `p`.
7. `--p-noninf` may be passed more than once. When multiple values are supplied, they are assigned in CLI input order across `--X-in`, `--X-list`, `--Xd-in`, and `--Xd-list`. If you pass more than one `--p-noninf`, the number of values must match the number of `--X-*` specifications.

## Core vs advanced options

`python -m pigean --help` shows the curated default interface. Use `python -m pigean --help-expert` to show Set B workflows, cache I/O, and expert tuning flags.

Core configs should target the main path:

1. raw evidence input (`--gwas-in` / `--exomes-in` / positive controls)
2. gene-set matrix input (`--X-in` / `--X-list`)
3. Gibbs output files

Advanced configs can layer in:

1. precomputed input ingestion (`--gene-stats-in`, `--gene-set-stats-in`)
2. HuGE cache workflows (`--huge-statistics-in/out`)
3. optional PheWAS output workflow
4. specialized modes (`sim`, `pops`, `naive_pops`)

For concise required-input/output blocks for each retained advanced workflow, see:

1. `docs/ADVANCED_SET_B.md`
