# PIGEAN

PIGEAN codebase split into:
- `src/`: new cleaned implementation (`pigean.py` and future modules)
- `legacy/`: frozen legacy script(s)
- `config/profiles/`: default run profiles
- `scripts/`: bundle download/packaging helpers
- `catalog/`: bundle catalog(s)
- `docs/`: bundle/release/bootstrap docs
- `tests/`: smoke/unit tests for scripts

## Quick start

1. Populate `catalog/bundles.json` from `catalog/bundles.example.json`.
2. Download required bundles:

```bash
python scripts/fetch_bundles.py --catalog catalog/bundles.json --profile minimal --mode gene_list
```

3. Edit `config/profiles/common.factor.json` and replace `__BUNDLE_ROOT__` with your bundle root (printed by fetch script, usually `<repo>/bundles/current`).

4. Run core workflow (`src/pigean.py`) with config + runtime input:

```bash
GENE_CSV=$(awk 'NF && $1 !~ /^#/ {print $1}' data/mody.gene.list | awk '!seen[$1]++' | paste -sd ',' -)

python src/pigean.py gibbs \
  --config config/profiles/gene_list.default.json \
  --positive-controls-list "$GENE_CSV" \
  --gene-stats-out results/MODY.gene_stats.out \
  --gene-set-stats-out results/MODY.gene_set_stats.out \
  --params-out results/MODY.params.out
```

## Core workflow

Core path is:

1. Load gene evidence (`--gwas-in`, `--exomes-in`, and/or positive controls)
2. Read/filter gene sets (`--X-in` / `--X-list`)
3. Estimate initial betas and run outer Gibbs
4. Write gene and gene-set outputs

## Advanced workflows (Set B)

Supported advanced paths are explicit and tagged as `[advanced]` in `--help`:

1. Precomputed input ingestion:
   - `--gene-stats-in`
   - `--gene-set-stats-in`
2. HuGE cache I/O:
   - `--huge-statistics-out`
   - `--huge-statistics-in`
3. Optional PheWAS output path:
   - `--run-phewas-from-gene-phewas-stats-in`
4. Specialized modes:
   - `sim`
   - `pops`
   - `naive_pops`

For required inputs and expected outputs per retained advanced workflow, see:
- `docs/ADVANCED_SET_B.md`

Legacy script is retained in `legacy/priors.py` for historical reference, but active refactor/testing targets `src/pigean.py`.

See `docs/REPO_BOOTSTRAP.md` for full setup and release steps.
