# PIGEAN to EAGGL Handoff

This document describes how to package PIGEAN outputs for direct EAGGL consumption.

## Bundle Handoff (`--eaggl-out`)

`pigean.py` can write a single tarball containing the minimum default EAGGL inputs.

### Command pattern

```bash
PYTHON=../../.venv/bin/python

$PYTHON src/pigean.py gibbs \
  --config config/profiles/gene_list.default.json \
  --positive-controls-list INS,GCK,HNF1A \
  --gene-stats-out results/pigean.gene_stats.out \
  --gene-set-stats-out results/pigean.gene_set_stats.out \
  --eaggl-out results/pigean_to_eaggl.tar.gz
```

### Output requirements

1. `--eaggl-out` must end with `.tar`, `.tar.gz`, or `.tgz`.
2. The run must have usable X/gene/gene-set outputs available to bundle.
3. PIGEAN hard-fails if required handoff files are missing.

### Bundle schema

Schema ID:

1. `pigean_eaggl_bundle/v1`

Manifest file:

1. `manifest.json` with source info and file metadata

Default input files:

1. `X.tsv.gz`
2. `gene_stats.tsv.gz`
3. `gene_set_stats.tsv.gz`

Optional files (included when available):

1. `gene_phewas_stats.tsv.gz`
2. `gene_set_phewas_stats.tsv.gz`

## Consuming in EAGGL

In the `eaggl` repo:

```bash
PYTHON=../../.venv/bin/python

$PYTHON src/eaggl.py factor \
  --eaggl-in /abs/path/to/pigean/results/pigean_to_eaggl.tar.gz \
  --factors-out results/factors.out
```

You can override any bundled default by passing explicit EAGGL CLI flags.

## Reference

For EAGGL-side workflow details and fallback separate-file inputs, see:

1. `../../eaggl/docs/INTEROP.md`
2. `../../eaggl/docs/WORKFLOWS.md`
