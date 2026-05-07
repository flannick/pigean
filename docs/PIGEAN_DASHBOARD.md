# PIGEAN/EAGGL Dashboard

`python -m pigean.dashboard` is a post-processing tool for building a standalone HTML dashboard from existing PIGEAN and optional EAGGL outputs. It does not rerun PIGEAN or EAGGL.

## Basic Usage

```bash
PYTHONPATH=src python -m pigean.dashboard \
  --pigean-run t2d:results/t2d/pigean \
  --eaggl-run t2d:gene_x_gene:results/t2d/eaggl/gene_x_gene \
  --html-out results/t2d/dashboard.html \
  --json-out results/t2d/dashboard.json
```

Runs are explicit and stable:

- `--pigean-run RUN_ID:DIR` points to a directory containing PIGEAN outputs.
- `--eaggl-run RUN_ID:MODE_ID:DIR` points to an EAGGL output directory associated with a PIGEAN `RUN_ID`.
- Repeat both flags to compare multiple runs or EAGGL modes.

The dashboard is tolerant of partial output directories. Missing expected files or optional columns are recorded as warnings in the dashboard status tab instead of causing the command to fail. Invalid command-line syntax and a command with no supplied runs still fail.

## Expected Inputs

For PIGEAN directories, the dashboard looks for:

- `pigean.gene_stats.out.gz`
- `pigean.gene_set_stats.out.gz`
- optional `pigean.params.out`, `pigean.run.log`, and `pigean.warnings.log`

For EAGGL directories, the dashboard looks for:

- `factors.out.gz`
- `gene_clusters.out.gz`
- `gene_set_clusters.out.gz`
- optional `trait_factor_links.out.gz`
- optional `factor_graph.html` and `factor_graph.json`
- optional `params.out`, `eaggl.run.log`, and `eaggl.warnings.log`

Pass `--x-input PATH` one or more times to enable gene/gene-set membership expansions from GMT-like gene-set input files.

## Useful Options

```bash
--title "T2D EAGGL dashboard"
--run-title t2d:"Type 2 diabetes"
--trait-id t2d:"Type_2_diabetes_(T2D)"
--gene-threshold 1
--gene-set-threshold 0.01
--factor-loading-within-max 0.05
--trait-min-neff 200
--max-genes-per-run 5000
--max-gene-sets-per-run 2500
--max-factor-genes 150
--max-factor-gene-sets 150
--max-provenance-rows-per-entry 50
```

The HTML file embeds the dashboard data and needs no backend. If an EAGGL `factor_graph.html` is present, it is embedded in the factor section.

By default, the embedded payload is size-controlled: genes must have combined support at least `1`, gene sets must have `beta_uncorrected` at least `0.01` when that column is available, factor gene/gene-set loadings are retained when they are within `0.05` of the factor-specific maximum loading, phenotype projections require effective size at least `200`, and nested provenance tables are capped at `50` rows per expanded entry.
