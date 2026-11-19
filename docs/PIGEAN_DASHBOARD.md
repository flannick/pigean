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
- `--pigean-group GROUP_ID:RUN_ID[:GROUP_TITLE]` assigns PIGEAN runs to a top-level dashboard group. This is useful for multi-trait EAGGL runs where one EAGGL result is built from several trait-specific PIGEAN runs.
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
- optional `gene_clusters_full.out.gz` (gene-set-routed by default when produced by EAGGL)
- optional `gene_clusters_full_via_gene_sets.out.gz` (optional full-gene output)
- `gene_set_clusters.out.gz`
- optional `trait_factor_links.out.gz`
- optional `factor_graph.html` and `factor_graph.json`
- optional source-specific factor graphs:
  - `factor_graph.full_direct.html`
  - `factor_graph.full_via_gene_sets.html`
- optional `params.out`, `eaggl.run.log`, and `eaggl.warnings.log`
- optional phi-selection reports such as `phi_selection_metrics_wide.out.gz`, `learn_phi_report.out.gz`, or `summary.tsv`

Pass `--x-input PATH` one or more times to enable gene/gene-set membership expansions from GMT-like gene-set input files.

### Gene and gene-set evidence

The gene plot shows indirect support (`prior`, horizontal) and direct support after updating gene-set priors (`log_bf`, vertical). The gene-set plot shows the marginal effect (`beta_uncorrected`, horizontal) and joint effect (`beta`, vertical). Both plots follow the column filters across all pages. Hover over a point for its name and values.

For run-specific provenance, repeat `--run-x-input RUN_ID:FILE` for the annotation libraries recorded by the run and pass its gene alias map with `--run-gene-map RUN_ID:FILE`. **Show gene sets** lists fitted sets containing the gene, ranked by joint effect and then marginal effect. Up to 50 are shown by default. **Show genes** under a gene set displays its scored members.

Pass the run’s HuGE archive with `--run-huge-statistics RUN_ID:FILE` and its known assembly with `--run-genome-build RUN_ID:BUILD`. **Show SNPs** marks the linked signal with the lowest GWAS P value against the selected gene and neighboring intervals. Equal P values are resolved by the larger stored gene–signal Bayes factor. Expand the signal list to see all linked lead positions, P values and stored Bayes factors. The coordinates identify the lead variants in archives that omit rsIDs.

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

The HTML file embeds the dashboard data and needs no backend. If an EAGGL `factor_graph.html` is present, it is embedded in the factor section. If full-gene projection tables are present, the EAGGL panel exposes a gene-loading-source selector so the gene loading table and available source-specific factor graph can switch between discovery genes, direct full-gene projection, and gene-set-routed full-gene projection. The default source is discovery gene loadings, the qualified display route. Use `--default-gene-loading-source full_via_gene_sets` to explicitly select the optional gene-set-routed projection; it remains available in the selector when present.

Use `--eaggl-phi-sweep RUN_ID:MODE_ID:DIR` to point the dashboard at either a directory containing per-phi EAGGL output subdirectories such as `phi_0p005/eaggl/`, `phi_0p01/eaggl/`, and `phi_0p02/eaggl/`, or a compact learn-phi output directory containing aggregate `factor_phi_*` tables. Each phi is loaded as a separate EAGGL run and automatically grouped under one EAGGL group selector. Directory-based phis can include optional per-phi artifacts such as factor graphs and full projections. Aggregate-table phis include factors, factor metrics, gene loadings, and gene-set loadings when the corresponding `factor_phi_*` files were written. When per-phi metrics are available, the group panel shows a metric heatmap with the composite score delineated and column maxima starred.

PIGEAN runs can be grouped explicitly for multi-trait dashboards:

```bash
--pigean-group multi_pre:t2d_pre:"T2D + MODY pre-exclusion"
--pigean-group multi_pre:mody_pre:"T2D + MODY pre-exclusion"
--pigean-group multi_post:t2d_post:"T2D + MODY post-exclusion"
--pigean-group multi_post:mody_post:"T2D + MODY post-exclusion"
```

When PIGEAN groups are present, the dashboard shows a top-level PIGEAN group selector and then a nested selector for the PIGEAN runs in that group. If an EAGGL run has the same run ID as the selected PIGEAN group, the EAGGL panel follows the group rather than the individual trait run.

Standalone EAGGL runs can be grouped explicitly:

```bash
--eaggl-group RUN_ID:MODE_ID:GROUP_ID[:GROUP_TITLE]
```

If no useful groups are present, the dashboard keeps the simpler single EAGGL-run dropdown behavior.

The EAGGL Factors tab reads `factor_metrics.out.gz` when available and shows factor-level metrics together with selected phi/composite metrics. Column-header info icons describe each metric. Missing metrics are displayed as `NA`.

Text column filters use case-insensitive browser regular expressions. Common Perl-style patterns such as `PPARG|PDX1`, `^GOBP_`, and `insulin.*signal` are supported; invalid regex patterns simply match no rows. Numeric column filters continue to use the selected threshold operator.

By default, the embedded payload is size-controlled: genes must have combined support at least `1`, gene sets must have `beta_uncorrected` at least `0.01` when that column is available, factor gene/gene-set loadings are retained when they are at least `0.05` times the factor-specific maximum loading, phenotype projections require effective size at least `200`, and nested provenance tables are capped at `50` rows per expanded entry.

In both the gene and gene-set loading heatmaps, **Order factors by total loading of matching genes** (or **gene sets**) puts the largest column sums first. Totals use raw loadings from all matching embedded rows, across pages, even when another display metric is selected. Ties keep the original factor order. Click a factor header to sort the rows separately.

### Research factor hierarchy

Supply a hierarchy export per trait. Relative source paths in the export are resolved against the analysis directory. For example, from the analysis directory:

```bash
python -m pigean.dashboard \
  --eaggl-hierarchy results/EAGGL_hierarchy_v1/main_default_review/T2D \
  --eaggl-hierarchy results/EAGGL_hierarchy_v1/main_default_review/CAD \
  --hierarchy-source-root . --html-out hierarchy.html
```

The original dashboard opens with run selection, the PIGEAN tables and the EAGGL panel. Select **Research consensus and hierarchy** or a supplied Main run in the EAGGL selector. **Hierarchy** shows the saved cut and full tree, **Factors** includes flagged leaves, and **Phi selection** explains the recorded candidate decision.

The research MDS graph, gene and gene-set heatmaps, and factor PheWAS include all leaves, internal families and flagged factors. **Plot nodes** opens at the saved overview cut and excludes flagged leaves until **Include flagged factors** is checked. Choose **Leaves only** for the fitted factors or **All nodes** for every level. The gene, gene-set and phenotype matrices support case-insensitive regular expressions such as `^(GCK|PPARG)$`. **At the cut** keeps the maximal qualifying family on each path, including indivisible leaves. **At or above the cut** also keeps its ancestors. Adjust the overlap floor or maximum gene equivalents, or toggle flagged leaves separately. The cut filters the existing coordinates and assay results. A gene disappears from the graph when it loses its last visible factor connection. The initial map fits classical MDS to the full discovery-gene profiles of QC-passing leaves. Each leaf is normalized to unit length, so squared distance equals twice one minus the leaf cosine. Families use their own OR-combined gene profiles and project onto the same two axes. Flagged leaves also project onto those axes. A displayed gene sits at the mean of its fitted leaf positions weighted by its original loadings. Positions describe similarity in two dimensions, and the displayed edges show selected memberships. Physics and dragging can move the displayed nodes; reset restores the fitted map. The graph uses up to three leading representatives per factor and cross-links those genes at half their strongest loading (minimum 0.01). Parent gene-set loadings summarize the fitted leaf values by OR.

To include **Factor PheWAS**, supply a matched `--pigean-run TRAIT:DIR` for each trait and `--hierarchy-phenotype-reference FILE.gz` with columns `Trait_Internal`, `Gene` and `Combined`. Each node is tested separately on the discovery genes, with a binary response (Combined > 1), the node loading, the anchor's direct log BF and an intercept. The dashboard uses EAGGL's marginal binary regression with HC3 standard errors. The displayed two-sided nominal P values use its normal reference. A constant phenotype response or unidentified node coefficient is shown as missing. Search, sort a node column, or switch heatmap metrics. The standalone HTML embeds the complete matrices. The underlying assays use the original input precision. The embedded values use float32.
