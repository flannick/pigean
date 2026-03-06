# EAGGL Factor Workflows (F1-F9)

This document maps each supported factoring workflow to:

1. required inputs
2. workflow-selection flags
3. a minimal runnable command pattern

All workflows run through `factor` (or `naive_factor`), and the selected workflow ID is visible with `--print-effective-config`.

## Common Setup

Use the local venv from the analysis workspace:

```bash
PYTHON=../../.venv/bin/python
```

Useful common outputs:

```bash
--factors-out results/factors.out \
--gene-set-clusters-out results/gene_set_clusters.out \
--gene-clusters-out results/gene_clusters.out \
--pheno-clusters-out results/pheno_clusters.out \
--params-out results/params.out
```

Debug workflow selection without running factorization:

```bash
$PYTHON src/eaggl.py factor --print-effective-config [workflow flags ...]
```

## Input Contracts

Core matrix/stat inputs (direct mode):

1. `--X-in` or `--X-list`
2. `--gene-stats-in`
3. `--gene-set-stats-in`

PheWAS matrix inputs (for phenotype/gene anchor workflows):

1. `--gene-phewas-stats-in`
2. `--gene-set-phewas-stats-in`

Bundle mode:

1. `--eaggl-bundle-in <bundle.tar.gz>` can provide defaults for core inputs
2. explicit CLI flags always override bundle defaults

## Workflow Matrix

### F1: Single Phenotype Anchoring (default stats path)

Required:

1. no special anchor flags
2. standard factor inputs (`X + gene stats + gene set stats`)

Command:

```bash
$PYTHON src/eaggl.py factor \
  --X-in /path/to/X.tsv.gz \
  --gene-stats-in /path/to/gene_stats.out \
  --gene-set-stats-in /path/to/gene_set_stats.out \
  --factors-out results/F1.factors.out
```

### F2: Gene-list-as-phenotype Anchoring

Required:

1. `--positive-controls-list` or `--positive-controls-in`
2. data path to compute/obtain factor weights

Command:

```bash
$PYTHON src/eaggl.py factor \
  --X-in /path/to/X.tsv.gz \
  --positive-controls-list INS,GCK,HNF1A \
  --gene-set-stats-out results/F2.gene_set_stats.out \
  --gene-stats-out results/F2.gene_stats.out \
  --factors-out results/F2.factors.out
```

### F3: Single Phenotype + Projection from PheWAS

Required:

1. base F1 inputs
2. either `--gene-phewas-stats-in` or `--gene-set-phewas-stats-in` (or both)

Command:

```bash
$PYTHON src/eaggl.py factor \
  --X-in /path/to/X.tsv.gz \
  --gene-stats-in /path/to/gene_stats.out \
  --gene-set-stats-in /path/to/gene_set_stats.out \
  --gene-phewas-stats-in /path/to/gene_phewas_stats.out \
  --factors-out results/F3.factors.out
```

### F4: Multi-Phenotype Anchoring

Required:

1. `--anchor-phenos <comma-separated phenotypes>`
2. `--gene-phewas-stats-in`
3. `--gene-set-phewas-stats-in`

Command:

```bash
$PYTHON src/eaggl.py factor \
  --X-in /path/to/X.tsv.gz \
  --anchor-phenos T2D,T2D_ALT \
  --gene-phewas-stats-in /path/to/gene_phewas_stats.out \
  --gene-set-phewas-stats-in /path/to/gene_set_phewas_stats.out \
  --factors-out results/F4.factors.out
```

### F5: Any-Phenotype Anchoring

Required:

1. `--anchor-any-pheno`
2. `--gene-phewas-stats-in`
3. `--gene-set-phewas-stats-in`

Command:

```bash
$PYTHON src/eaggl.py factor \
  --X-in /path/to/X.tsv.gz \
  --anchor-any-pheno \
  --gene-phewas-stats-in /path/to/gene_phewas_stats.out \
  --gene-set-phewas-stats-in /path/to/gene_set_phewas_stats.out \
  --factors-out results/F5.factors.out
```

### F6: Single-Gene Anchoring

Required:

1. `--anchor-gene <GENE>` (or `--anchor-genes <single_gene>`)
2. `--gene-phewas-stats-in`
3. `--gene-set-phewas-stats-in`

Command:

```bash
$PYTHON src/eaggl.py factor \
  --X-in /path/to/X.tsv.gz \
  --anchor-gene INS \
  --gene-phewas-stats-in /path/to/gene_phewas_stats.out \
  --gene-set-phewas-stats-in /path/to/gene_set_phewas_stats.out \
  --factors-out results/F6.factors.out
```

### F7: Multi-Gene Anchoring

Required:

1. `--anchor-genes <comma-separated genes>`
2. `--gene-phewas-stats-in`
3. `--gene-set-phewas-stats-in`

Optional expansion knobs:

1. `--add-gene-sets-by-enrichment-p`
2. `--add-gene-sets-by-fraction`
3. `--add-gene-sets-by-naive`
4. `--add-gene-sets-by-gibbs`

Command:

```bash
$PYTHON src/eaggl.py factor \
  --X-in /path/to/X.tsv.gz \
  --anchor-genes INS,GCK \
  --gene-phewas-stats-in /path/to/gene_phewas_stats.out \
  --gene-set-phewas-stats-in /path/to/gene_set_phewas_stats.out \
  --factors-out results/F7.factors.out
```

### F8: Any-Gene Anchoring

Required:

1. `--anchor-any-gene`
2. `--gene-phewas-stats-in`
3. `--gene-set-phewas-stats-in`

Command:

```bash
$PYTHON src/eaggl.py factor \
  --X-in /path/to/X.tsv.gz \
  --anchor-any-gene \
  --gene-phewas-stats-in /path/to/gene_phewas_stats.out \
  --gene-set-phewas-stats-in /path/to/gene_set_phewas_stats.out \
  --factors-out results/F8.factors.out
```

### F9: Gene-set Anchoring

Required:

1. `--anchor-gene-set`
2. `--run-phewas-from-gene-phewas-stats-in`
3. enough evidence input to compute gene/gene-set scores if not precomputed

Command:

```bash
$PYTHON src/eaggl.py factor \
  --X-in /path/to/X.tsv.gz \
  --anchor-gene-set \
  --gene-stats-in /path/to/gene_stats.out \
  --gene-set-stats-in /path/to/gene_set_stats.out \
  --run-phewas-from-gene-phewas-stats-in /path/to/gene_phewas_stats.out \
  --factors-out results/F9.factors.out
```

## Workflow Validation and Guardrails

1. Use `--print-effective-config` first to verify the selected workflow ID and required inputs.
2. Missing required inputs hard-fail with an actionable message.
3. For overlapping flags, EAGGL applies workflow-specific precedence and logs warnings for ignored inputs.
4. PheWAS stages log explicit input-I/O mode:
   - `mode=reuse_loaded_matrix` when a compatible loaded matrix is reused.
   - `mode=re_read_file` when stage inputs must be read from file again.

## Removed Legacy GLS Path

EAGGL does not support the historical GLS/whitened-Y path.

1. Removed aliases hard-fail if passed (`--run-gls`, `run_gls`, `store_cholesky`).
2. Correlation-aware behavior in EAGGL uses the retained corrected-OLS path only.

## Gene-set Filter Relaxation

When EAGGL invokes shared read-X filtering, `--increase-filter-gene-set-p` is treated
as a minimum kept-fraction target used to relax prefiltering if needed.
Post-read filtering does not tighten this threshold.

## References

1. Deterministic workflow baseline generator: `scripts/freeze_factor_workflow_effective_configs.sh`
2. Effective-config fixtures: `tests/data/reference/factor_workflow_effective_config/`
3. PIGEAN handoff details: `docs/INTEROP.md`
