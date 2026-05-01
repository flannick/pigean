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
- Use **Enable Physics** to relax the deterministic layout with browser-side force physics.
- Use **Reset Layout** to return to the deterministic layout.

To start with physics enabled:

```bash
PYTHONPATH=src python -m eaggl.factor_graph \
  --eaggl-dir results/eaggl_seed000 \
  --html-out results/eaggl_seed000/factor_graph.html \
  --html-physics
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

You can also pass explicit paths:

```bash
PYTHONPATH=src python -m eaggl.factor_graph \
  --factors-in results/factors.out.gz \
  --gene-clusters-in results/gene_clusters.out.gz \
  --trait-factor-links-in results/trait_factor_links.out.gz \
  --html-out results/factor_graph.html
```

## Visual Encoding

- Factor nodes are squares.
- Gene nodes are circles. By default, gene nodes are labeled with the gene ID column; pass `--gene-label-col` only if a separate display-label column is desired.
- Trait nodes are diamonds.
- Factor colors are blended into gene and trait nodes according to their factor loadings.
- The layout uses a deterministic MDS-style projection of factor-loading profiles, preserving the legacy factor/genes/traits visual arrangement.
- Optional browser-side physics keeps the deterministic coordinates as anchors while allowing connected nodes to relax interactively.

## Useful Filters

```bash
--max-num-gene-nodes-per-factor 10
--max-num-trait-nodes-per-factor 10
--gene-min-loading 0.01
--trait-min-loading 0.005
--gene-min-loading-frac 0.5
--trait-min-loading-frac 0.5
```

The `*-min-loading-frac` options keep only near-best loadings for each retained gene or trait, which helps make factor-specific structure visible.
