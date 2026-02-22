# pigean

PIGEAN codebase split into:
- `src/`: new cleaned implementation (`pigean.py` and future modules)
- `legacy/`: frozen legacy script(s)
- `config/profiles/`: default run profiles
- `scripts/`: bundle download/packaging + run helpers
- `catalog/`: bundle catalog(s)
- `docs/`: bundle/release/bootstrap docs
- `tests/`: smoke/unit tests for scripts

## Quick start

1. Create and activate a venv.
2. Populate `catalog/bundles.json` from `catalog/bundles.example.json`.
3. Download required bundles:

```bash
python scripts/fetch_bundles.py --catalog catalog/bundles.json --profile minimal --mode gene_list
```

4. Run legacy with profile + one input file:

```bash
python scripts/run_legacy.py \
  --config config/profiles/gene_list.default.json \
  --gene-list-in data/mody.gene.list \
  --out-dir results \
  --run-name MODY
```

See `docs/REPO_BOOTSTRAP.md` for full setup and release steps.
