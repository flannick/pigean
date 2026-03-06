# PIGEAN -> EAGGL Interop

This document describes the supported handoff paths from `pigean.py` outputs into `eaggl.py`.

## Recommended Path: Bundle Handoff

### Step 1: Write handoff bundle from PIGEAN

In the `pigean` repo:

```bash
PYTHON=../../.venv/bin/python

$PYTHON src/pigean.py gibbs \
  --config config/profiles/gene_list.default.json \
  --positive-controls-list INS,GCK,HNF1A \
  --gene-stats-out results/pigean.gene_stats.out \
  --gene-set-stats-out results/pigean.gene_set_stats.out \
  --eaggl-bundle-out results/pigean_to_eaggl.tar.gz
```

Bundle output requirements:

1. output path must end in `.tar`, `.tar.gz`, or `.tgz`
2. core tables are included:
   - `X.tsv.gz`
   - `gene_stats.tsv.gz`
   - `gene_set_stats.tsv.gz`
3. optional PheWAS tables are included when present in the same run:
   - `gene_phewas_stats.tsv.gz`
   - `gene_set_phewas_stats.tsv.gz`

### Step 2: Run EAGGL from bundle

In the `eaggl` repo:

```bash
PYTHON=../../.venv/bin/python

$PYTHON src/eaggl.py factor \
  --eaggl-bundle-in /abs/path/to/pigean/results/pigean_to_eaggl.tar.gz \
  --factors-out results/factors.out \
  --gene-set-clusters-out results/gene_set_clusters.out
```

## Fallback Path: Separate Files

If you do not use `--eaggl-bundle-in`, provide files directly:

```bash
PYTHON=../../.venv/bin/python

$PYTHON src/eaggl.py factor \
  --X-in /path/to/X.tsv.gz \
  --gene-stats-in /path/to/gene_stats.out \
  --gene-set-stats-in /path/to/gene_set_stats.out \
  --factors-out results/factors.out
```

## Override Precedence

When both bundle and explicit flags are present:

1. `--eaggl-bundle-in` loads defaults from bundle manifest
2. explicit CLI/config file flags override those defaults
3. workflow validation runs after merge and hard-fails on missing required inputs

Example (override bundled gene stats):

```bash
$PYTHON src/eaggl.py factor \
  --eaggl-bundle-in /path/to/handoff.tar.gz \
  --gene-stats-in /path/to/override_gene_stats.out \
  --factors-out results/factors.out
```

## Sanity Check Command

Use this to inspect resolved inputs and workflow selection before expensive runs:

```bash
$PYTHON src/eaggl.py factor \
  --eaggl-bundle-in /path/to/handoff.tar.gz \
  --print-effective-config
```

The JSON includes:

1. selected workflow (`factor_workflow.id`)
2. final merged options (`options`)
3. bundle metadata (`eaggl_bundle`)
