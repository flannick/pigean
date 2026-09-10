# PIGEAN Results Portal

`python -m pigean.portal` is a lightweight viewer for PIGEAN outputs. It has two steps:

1. `build` reads one or more runs (gene stats, gene-set stats, and the gene x gene-set
   loading table), applies thresholds, and writes a SQLite file.
2. `serve` hosts that file behind a small JSON API and a single-page UI: a scatter of each
   gene's direct (`log_bf`) vs. indirect (`prior`) score, a ranked gene-set table, and the
   gene loadings for whichever gene set (or gene) you click.

Like `pigean.dashboard`, it is a post-processing tool that never reruns PIGEAN, and it uses
only the standard library (`sqlite3`, `http.server`). The UI loads Plotly from a CDN by
default; pass `serve --plotly-js` for offline use.

## Build

```bash
PYTHONPATH=src python -m pigean.portal build \
  --db results/portal.sqlite \
  --run t2d:results/t2d/pigean \
  --run-title t2d:"Type 2 diabetes" \
  --gene-filter 'prior>1' --gene-filter 'log_bf>1' \
  --gene-set-filter 'beta>0.01'
```

- `--run RUN_ID:DIR` locates the tables in a directory. Both the dashboard-style names
  (`pigean.gene_stats.out.gz`, `pigean.gene_set_stats.out.gz`, `pigean.gene_gene_set_stats.out.gz`)
  and LAP-style names (`*.gene_stats.tsv`, `*.gene_set_stats.tsv`, `*.gene_gene_set_stats.tsv`)
  are recognised. Repeat for several runs.
- `--run-files RUN_ID:gene_stats=PATH,gene_set_stats=PATH[,gene_gene_set_stats=PATH]` gives
  explicit paths when the files live elsewhere.
- `--gene-filter`, `--gene-set-filter`, `--loading-filter` take `column op number`
  expressions (`>`, `>=`, `<`, `<=`, `==`, `!=`) against the normalised column names below
  and are repeatable. `--filter-mode any` (default) keeps a row if *any* filter on that table
  passes, so `--gene-filter 'prior>1' --gene-filter 'log_bf>1'` keeps genes with either
  strong indirect or strong direct support; `--filter-mode all` requires every filter.
- Loadings are kept only for gene sets that passed the gene-set filters (so the loading
  table stays small); the genes in those loadings do not need to pass the gene filters.
  `--keep-all-loadings` keeps loadings for every gene set instead.
- `--append` adds or replaces runs in an existing database instead of recreating it.

Normalised columns:

| table | columns (aliases accepted from the raw files) |
|---|---|
| genes | `gene`, `prior`, `combined`, `log_bf`, `huge_score` (`huge_score_gwas`), `n`, `chrom`, `start`, `end` |
| gene_sets | `gene_set`, `label`, `n`, `beta`, `beta_uncorrected`, `p_orig`, `z_orig` |
| gene_gene_sets | `gene`, `gene_set`, `beta`, `weight`, `prior`, `combined`, `log_bf`, `huge_score` |

Every original column of the gene and gene-set tables is also stored as JSON and shown in the
detail panel, so `--output-detail full` runs lose nothing.

## Serve

```bash
PYTHONPATH=src python -m pigean.portal serve --db results/portal.sqlite --host 127.0.0.1 --port 8765
```

Open `http://127.0.0.1:8765/`. The database is opened read-only; rebuild with `build` and
restart to pick up changes. Use `--host 0.0.0.0` to expose it on a network (there is no
authentication, so keep it behind SSH forwarding or a trusted network).

### JSON API

| endpoint | parameters |
|---|---|
| `GET /api/runs` | — |
| `GET /api/genes` | `run`, optional `min_prior`, `min_log_bf`, `min_combined`, `search`, `sort` (`combined`/`prior`/`log_bf`/`huge_score`/`gene`), `limit` |
| `GET /api/gene_sets` | `run`, optional `min_beta`, `min_beta_uncorrected`, `search` (id or label), `sort` (`beta`/`beta_uncorrected`/`n`/`gene_set`), `limit` |
| `GET /api/gene_set` | `run`, `id`, optional `limit` — the gene set plus its gene loadings (sorted by weight, then combined) |
| `GET /api/gene` | `run`, `id`, optional `limit` — the gene plus the gene sets it loads on |
| `GET /healthz` | — |

## Tests

```bash
PYTHONPATH=src python -m pytest tests/test_portal_unittest.py
```
