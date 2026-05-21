# EAGGL Factor Graph

`python -m eaggl.factor_graph` is a post-processing tool for visualizing existing EAGGL factor outputs. It does not refit factors and is not part of the `eaggl factor` pipeline.

## Basic Usage

```bash
PYTHONPATH=src python -m eaggl.factor_graph \
  --eaggl-dir results/eaggl_seed000 \
  --html-out results/eaggl_seed000/factor_graph.html \
  --json-out results/eaggl_seed000/factor_graph.json
```

`--pdf-out` is also available when `matplotlib` is installed. HTML output is interactive by default:

- Drag nodes to manually pin and inspect local structure.
- Drag blank space to pan and use the `+` / `-` buttons to zoom.
- Browser-side force physics is enabled on load by default. Use **Disable Physics** to stop the relaxation and keep the current coordinates.
- Use **Reset Layout** to return to the deterministic layout.
- Use the filter bar to target factors, genes, or phenotypes. Text filters hide unmatched nodes of the targeted types while dimming other node types for context.
- Use the **hide unmatched** checkbox to switch between hiding unmatched targeted nodes and keeping them visible but dimmed. It is unchecked by default.
- Add one or more case-insensitive text filters to match nodes whose ID, label, or type contains any requested substring; comma-separated entries are treated as OR filters.
- Use **Add node** to add omitted genes from the embedded cluster-file candidates by autocomplete. The standalone HTML contains the candidate nodes and edges, so this works without a server.
- Click a node to show embedded provenance. Gene nodes can show per-anchor direct, indirect, and combined support when the original `--gene-phewas-stats-in` file is also passed to `eaggl.factor_graph`, plus all factor loadings within 0.01 of that gene's top factor loading. Factor nodes show per-anchor relevance from `trait_factor_links.out` plus the top five gene and gene-set loadings.
- Hover over an edge to show its weight, source table, and source field.
- Node labels are truncated to 20 characters by default, with the full label shown on hover. Use `--label-max-chars` to change the displayed length, or `--label-max-chars 0` to disable truncation.
- Trait nodes use `--trait-layout-mode anchored_top_factor` by default. This keeps phenotype nodes near the factor or factor pair with strongest linkage and enforces a small minimum radius from the factor centroid, avoiding collapse when trait linkage vectors are broad. Use `--trait-layout-mode mds` to restore raw MDS placement.
- Interactive physics uses shorter factor-trait springs by default with `--trait-edge-length-scale 0.2`, so anchor phenotypes stay near the factor/gene structure.

To start with physics disabled:

```bash
PYTHONPATH=src python -m eaggl.factor_graph \
  --eaggl-dir results/eaggl_seed000 \
  --html-out results/eaggl_seed000/factor_graph.html \
  --no-html-physics
```

To write a static SVG-only HTML file:

```bash
PYTHONPATH=src python -m eaggl.factor_graph \
  --eaggl-dir results/eaggl_seed000 \
  --html-out results/eaggl_seed000/factor_graph.static.html \
  --no-html-interactive
```

With `--eaggl-dir`, the tool looks for standard EAGGL outputs:

- `factors.out.gz`
- `gene_clusters_full.out.gz` or `gene_clusters.out.gz`
- `trait_factor_links.out.gz` or `pheno_clusters.out.gz`

Phenotype nodes are included when a trait-factor linkage file is present in `--eaggl-dir` or passed explicitly with `--trait-factor-links-in`.
Factor-node trait provenance is ranked by `beta` by default. Use `--trait-factor-rank-field beta_uncorrected` or `--trait-factor-rank-field nnls` when the graph should prioritize independent enrichment or NNLS projection loadings instead.

You can also pass explicit paths:

```bash
PYTHONPATH=src python -m eaggl.factor_graph \
  --factors-in results/factors.out.gz \
  --gene-clusters-in results/gene_clusters.out.gz \
  --gene-set-clusters-in results/gene_set_clusters.out.gz \
  --trait-factor-links-in results/trait_factor_links.out.gz \
  --gene-phewas-stats-in data/gene_phewas_stats.tsv.gz \
  --gene-set-phewas-stats-in data/gene_set_phewas_stats.tsv.gz \
  --html-out results/factor_graph.html
```

To show a bounded number of phenotype nodes per factor:

```bash
PYTHONPATH=src python -m eaggl.factor_graph \
  --eaggl-dir results/eaggl_seed000 \
  --trait-factor-links-in results/trait_factor_links.out.gz \
  --max-num-trait-nodes-per-factor 3 \
  --html-out results/eaggl_seed000/factor_graph.with_traits.html
```

By default, phenotype nodes are also filtered to traits with `trait_neff > 25` when `trait_neff` or `trait_n_eff` is available in the linkage table. Override this with `--trait-min-neff`, for example `--trait-min-neff 0` to disable the effective-size filter.

## Visual Encoding

- Factor nodes are squares.
- Gene nodes are circles. By default, gene nodes are labeled with the gene ID column; pass `--gene-label-col` only if a separate display-label column is desired.
- Trait nodes are diamonds.
- Factor colors are blended into gene and trait nodes according to their factor loadings.
- The layout uses a deterministic MDS-style projection of factor-loading profiles, preserving the legacy factor/genes/traits visual arrangement.
- Optional browser-side physics keeps the deterministic coordinates as anchors while allowing connected nodes to relax interactively.

## Useful Filters

```bash
--max-num-factor-nodes 50
--max-num-gene-nodes-per-factor 3
--max-num-trait-nodes-per-factor 3
--gene-min-loading 0.01
--trait-min-loading 0.005
--trait-min-neff 25
--trait-factor-rank-field beta
--label-max-chars 20
--trait-layout-mode anchored_top_factor
--trait-coordinate-scale 0.2
--trait-min-centroid-distance-frac 0.35
--trait-edge-length-scale 0.2
--gene-min-loading-frac 0.5
--trait-min-loading-frac 0.5
```

The `*-min-loading-frac` options keep only near-best loadings for each retained gene or trait, which helps make factor-specific structure visible.
