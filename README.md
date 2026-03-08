# PIGEAN

PIGEAN codebase split into:
- `src/`: current implementation plus canonical in-repo EAGGL snapshot
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

4. Run core workflow (`python -m pigean`) with config + runtime input:

```bash
GENE_CSV=$(awk 'NF && $1 !~ /^#/ {print $1}' data/mody.gene.list | awk '!seen[$1]++' | paste -sd ',' -)

PYTHONPATH=src python -m pigean gibbs \
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

For PIGEAN -> EAGGL handoff bundle usage, see:
- `docs/EAGGL_INTEROP.md`

For optional single-file stitched artifacts built from the modular source tree, see:
- `docs/STITCHED_ARTIFACTS.md`

For the canonical in-repo EAGGL snapshot and transition notes, see:
- `docs/CANONICAL_SOURCE.md`
- `docs/eaggl/README.md`
- `docs/eaggl/TRANSITION.md`

For curated CLI surface and category inventory (auto-generated):
- `docs/CLI_OPTIONS.md`
- `docs/cli_option_manifest.json`

Legacy script is retained in `legacy/priors.py` for historical reference, but active refactor/testing targets `python -m pigean`.

For current architecture, the package modules under `src/pigean/` are the primary edit locations for the CLI and stage-level flow. The main remaining flat-file exception is the deeper legacy core in `src/pigean_legacy_main.py`.

See `docs/REPO_BOOTSTRAP.md` for full setup and release steps.
