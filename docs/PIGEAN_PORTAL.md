# PIGEAN Results Portal

`python -m pigean.portal` is a lightweight viewer for PIGEAN outputs. It has three commands (the served page opens on a landing search — trait, model, run, optional gene — and then a results view):

1. `build` reads one or more runs (gene stats, gene-set stats, and the gene x gene-set
   loading table), applies thresholds, and writes a SQLite file.
2. `serve` hosts that file behind a small JSON API and a single-page UI: model → trait → run
   selectors plus a gene search, a scatter of each gene's direct (`log_bf`) vs. indirect
   (`prior`) score with a sortable gene table, a ranked gene-set table, and a collapsible
   detail sheet that opens with a gene set's gene loadings (its genes are ringed on the scatter while the
   rest fade) or a gene's gene-set memberships. Each sheet has two tabs: details for the
   current run, and "Across traits" — a vertical trait plot (one point per run, coloured by
   portal trait group, metric selectable) of that gene or gene set in every run of the
   database. Build thresholds, input paths and run parameters sit behind toggles.
3. `html` writes the same UI as a static page whose JavaScript calls a `serve` instance at a
   fixed URL, so the page can be hosted from a bucket or any static file server.

Like `pigean.dashboard`, it is a post-processing tool that never reruns PIGEAN, and it uses
only the standard library (`sqlite3`, `http.server`). The UI loads Plotly from a CDN by
default; pass `serve --plotly-js` for offline use.

## Build

```bash
PYTHONPATH=src python -m pigean.portal build \
  --db results/portal.sqlite \
  --package model=large,trait=T2D,gene_stats=results/T2D/gs.out,gene_set_stats=results/T2D/gss.out,gene_gene_set_stats=results/T2D/ggss.out,params=results/T2D/params.out \
  --package model=large,trait=IBD,gene_stats=results/IBD/gs.out,gene_set_stats=results/IBD/gss.out,gene_gene_set_stats=results/IBD/ggss.out \
  --phenotype-file ../dig-portal-data-models/versions/phenotype/v0.0.1/portal_phenotypes_flat.tsv \
  --gene-filter 'prior>1' --gene-filter 'log_bf>1' \
  --gene-set-filter 'beta>0.01'
```

Every PIGEAN result has a model and a trait, and there may be several runs per model/trait
(seeds, parameter sweeps). A `--package` describes one result:

- `model=` the model id and optional `model_title=` a display name;
- `trait=` the trait, as a legacy portal phenotype id when possible so it can be joined with
  the portal phenotype file;
- `gene_stats=`, `gene_set_stats=`, optional `gene_gene_set_stats=` (needed for loadings) and
  optional `params=` (the `params.out` / `params.tsv` written by PIGEAN, stored as run
  provenance and shown under "Run parameters");
- optional `run=` label. When omitted the run is called `main` if the model/trait pair occurs
  once, or `run1`, `run2`, … in input order when it occurs several times. The run id is always
  `<model>__<trait>__<run>`, and `title=` overrides the display title.

Older forms are still accepted:

- `--run RUN_ID:DIR` locates the tables in a directory. Both the dashboard-style names
  (`pigean.gene_stats.out.gz`, `pigean.gene_set_stats.out.gz`, `pigean.gene_gene_set_stats.out.gz`)
  and LAP-style names (`*.gene_stats.tsv`, `*.gene_set_stats.tsv`, `*.gene_gene_set_stats.tsv`)
  are recognised. Repeat for several runs.
- `--run-files RUN_ID:gene_stats=PATH,gene_set_stats=PATH[,gene_gene_set_stats=PATH]` gives
  explicit paths when the files live elsewhere.
- `--run-meta RUN_ID:model=NAME,trait=NAME,seed=N[,title=TEXT]` labels a run for the UI's
  model → trait → run selectors. Omitted labels are inferred from run ids shaped like
  `<model>__<trait>[__s<seed>]` (the LAP naming used by the inferiority pipeline).
- `--gene-filter`, `--gene-set-filter`, `--loading-filter` take `column op number`
  expressions (`>`, `>=`, `<`, `<=`, `==`, `!=`) against the normalised column names below
  and are repeatable. `--filter-mode any` (default) keeps a row if *any* filter on that table
  passes, so `--gene-filter 'prior>1' --gene-filter 'log_bf>1'` keeps genes with either
  strong indirect or strong direct support; `--filter-mode all` requires every filter.
- Loadings are kept only for gene sets that passed the gene-set filters (so the loading
  table stays small); the genes in those loadings do not need to pass the gene filters.
  `--keep-all-loadings` keeps loadings for every gene set instead.
- `--phenotype-file PATH` points at dig-portal-data-models'
  `versions/phenotype/v*/portal_phenotypes_flat.tsv`. Each run's `trait` (a legacy portal
  phenotype id such as `T2D`) is looked up there and the portal name, portal id, trait group,
  type and every ontology mapping (MESH / MONDO / EFO / DOID / … with predicate and confidence)
  are stored in `phenotypes` / `phenotype_mappings` and returned with `/api/runs`. The UI shows
  the name and portal id on the run card, lets you search traits by name or portal id, and lists
  the mappings under a "Phenotype mappings" toggle. Traits missing from the file are warned about.
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

## Static page (`html`)

```bash
PYTHONPATH=src python -m pigean.portal html \
  --api-url http://localhost:8765 \
  --out results/portal.html \
  --db results/portal.sqlite   # optional: only checked for existence, so a pipeline can make the page depend on the database
```

The page is self-contained apart from Plotly (CDN, or `--plotly-js` to embed) and calls
`<api-url>/api/...` for every request. Host it anywhere static files can be served (an S3/GCS
bucket, GitHub Pages, `python -m http.server`) and keep a `serve` instance reachable at the
API URL from the viewer's browser. For a server bound to `localhost`, that means the viewer
opens the page on the same machine or uses an SSH tunnel (`ssh -L 8765:localhost:8765 host`).

`serve` sends `Access-Control-Allow-Origin: *` on API responses so the cross-origin page can
read them (the API is read-only). Restrict it with `serve --cors-origin https://my-bucket.example`
or disable with `--cors-origin ""`.

### JSON API

| endpoint | parameters |
|---|---|
| `GET /api/runs` | — (each run carries `model`, `trait`, `seed`, build counts/filters and, when built with `--phenotype-file`, a `phenotype` object with `name`, `portal_id`, `trait_group`, `mappings[]`) |
| `GET /api/genes` | `run`, optional `min_prior`, `min_log_bf`, `min_combined`, `search`, `sort` (`combined`/`prior`/`log_bf`/`huge_score`/`gene`), `limit` |
| `GET /api/gene_sets` | `run`, optional `min_beta`, `min_beta_uncorrected`, `search` (id or label), `sort` (`beta`/`beta_uncorrected`/`n`/`gene_set`), `limit` |
| `GET /api/gene_set` | `run`, `id`, optional `limit` — the gene set plus its gene loadings (sorted by weight, then combined) |
| `GET /api/gene` | `run`, `id`, optional `limit` — the gene plus the gene sets it loads on |
| `GET /api/gene_across` | `id`, optional `model` — the gene's `combined` / `log_bf` / `prior` / `huge_score` in every run where it passed the build thresholds, with each run's trait, phenotype name and trait group |
| `GET /api/gene_set_across` | `id`, optional `model` — likewise `beta` / `beta_uncorrected` for a gene set |
| `GET /api/run_params` | `run` — the PIGEAN parameters stored from the package's `params=` file |
| `GET /healthz` | — |

## Tests

```bash
PYTHONPATH=src python -m pytest tests/test_portal_unittest.py
```
