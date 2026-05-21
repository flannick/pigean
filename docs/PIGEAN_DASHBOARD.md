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
- optional `factor_metrics.out.gz`
- `gene_clusters.out.gz`
- optional `gene_clusters_full.out.gz`
- optional `gene_clusters_full_via_gene_sets.out.gz`
- `gene_set_clusters.out.gz`
- optional `trait_factor_links.out.gz`
- optional `factor_graph.html` and `factor_graph.json`
- optional source-specific factor graphs:
  - `factor_graph.full_direct.html`
  - `factor_graph.full_via_gene_sets.html`
- optional `params.out`, `eaggl.run.log`, and `eaggl.warnings.log`
- optional phi-selection reports such as `phi_selection_metrics_wide.out.gz`, `learn_phi_report.out.gz`, or `summary.tsv`

Pass `--x-input PATH` one or more times to enable gene/gene-set membership expansions from GMT-like gene-set input files.

## Useful Options

```bash
--title "T2D EAGGL dashboard"
--run-title t2d:"Type 2 diabetes"
--trait-id t2d:"Type_2_diabetes_(T2D)"
--gene-threshold 1
--gene-set-threshold 0.01
--factor-loading-min-max-frac 0.05
--trait-min-neff 200
--max-genes-per-run 5000
--max-gene-sets-per-run 2500
--max-factor-genes 150
--max-factor-gene-sets 150
--max-provenance-rows-per-entry 50
```

The HTML file embeds the dashboard data and needs no backend. If an EAGGL `factor_graph.html` is present, it is embedded in the factor section. If full-gene projection tables are present, the EAGGL panel exposes a gene-loading-source selector so the gene loading table and available source-specific factor graph can switch between discovery genes, direct full-gene projection, and gene-set-routed full-gene projection.

Use `--eaggl-phi-sweep RUN_ID:MODE_ID:DIR` to point the dashboard at either a directory containing per-phi EAGGL output subdirectories such as `phi_0p005/eaggl/`, `phi_0p01/eaggl/`, and `phi_0p02/eaggl/`, or a compact learn-phi output directory containing aggregate `factor_phi_*` tables. Each phi is loaded as a separate EAGGL run and automatically grouped under one EAGGL group selector. Directory-based phis can include optional per-phi artifacts such as factor graphs and full projections. Aggregate-table phis include factors, factor metrics, gene loadings, and gene-set loadings when the corresponding `factor_phi_*` files were written. When per-phi metrics are available, the group panel shows a metric heatmap with the composite score delineated and column maxima starred.

Standalone EAGGL runs can be grouped explicitly:

```bash
--eaggl-group RUN_ID:MODE_ID:GROUP_ID[:GROUP_TITLE]
```

If no useful groups are present, the dashboard keeps the simpler single EAGGL-run dropdown behavior.

The EAGGL Factors tab reads `factor_metrics.out.gz` when available and shows factor-level metrics together with selected phi/composite metrics. Column-header info icons describe each metric. Missing metrics are displayed as `NA`.

Text column filters use case-insensitive browser regular expressions. Common Perl-style patterns such as `PPARG|PDX1`, `^GOBP_`, and `insulin.*signal` are supported; invalid regex patterns simply match no rows. Numeric column filters continue to use the selected threshold operator.

By default, the embedded payload is size-controlled: genes must have combined support at least `1`, gene sets must have `beta_uncorrected` at least `0.01` when that column is available, factor gene/gene-set loadings are retained when they are at least `0.05` times the factor-specific maximum loading, phenotype projections require effective size at least `200`, and nested provenance tables are capped at `50` rows per expanded entry.
