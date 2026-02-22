# Bootstrap + release checklist

## 1) Initialize repo

```bash
cd pigean
git init
```

## 2) Stage legacy script snapshot

```bash
python scripts/stage_legacy.py --source /ABS/PATH/TO/priors.warm_stopping.py
```

## 3) Create/update bundle catalog

```bash
cp catalog/bundles.example.json catalog/bundles.json
# edit URLs + SHA256
```

## 4) Download minimal bundles for local smoke tests

```bash
python scripts/fetch_bundles.py --catalog catalog/bundles.json --profile minimal --mode gene_list
```

## 5) Run a profile

```bash
python scripts/run_legacy.py \
  --config config/profiles/gene_list.default.json \
  --gene-list-in data/mody.gene.list \
  --out-dir results \
  --run-name MODY
```

## 6) Publish bundles

Recommended:
- code: GitHub repo `pigean`
- bundles: object storage (S3/GCS/R2) at immutable, versioned URLs
- checksums: store in `catalog/bundles.json`
- optional mirror: GitHub Releases for small bundles only

## 7) Push code

```bash
git add .
git commit -m "Initialize pigean repo scaffold with legacy runner, bundle tooling, and configs"
git branch -M main
git remote add origin git@github.com:<ORG>/pigean.git
git push -u origin main
```
