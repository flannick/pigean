# Bundles

Bundle goals:
- Keep large static resources out of git
- Version resource sets immutably
- Let configs point to stable symlinks in `bundles/current/`

## Expected bundle groups

- `core_small`: maps and small loc files
- `x_panels`: large `--X-in`/`--X-list` files
- `phewas_large`: very large PheWAS stats files

## Build a bundle

```bash
python scripts/package_bundle.py \
  --name core_small \
  --version 2026.02.0 \
  --source-dir /path/to/staging/core_small \
  --out-dir dist/bundles
```

## Publish

Upload `*.tar.gz` to versioned object storage paths (S3/GCS/R2), keep immutable.

## Install bundles

```bash
python scripts/fetch_bundles.py \
  --catalog catalog/bundles.json \
  --profile minimal \
  --mode gene_list
```

This installs into `bundles/<name>-<version>/` and updates symlinks under `bundles/current/<name>`.
