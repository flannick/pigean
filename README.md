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

4. Edit `config/profiles/common.factor.json` and replace `__BUNDLE_ROOT__` with your bundle root (printed by fetch script, usually `<repo>/bundles/current`).

5. Run legacy directly with config + input:

```bash
GENE_CSV=$(awk 'NF && $1 !~ /^#/ {print $1}' data/mody.gene.list | awk '!seen[$1]++' | paste -sd ',' -)

python legacy/priors.py \
  --config config/profiles/gene_list.default.json \
  --positive-controls-list "$GENE_CSV" \
  --gene-stats-out results/MODY.gene_stats.out \
  --gene-set-stats-out results/MODY.gene_set_stats.out \
  --params-out results/MODY.params.out
```

See `docs/REPO_BOOTSTRAP.md` for full setup and release steps.
