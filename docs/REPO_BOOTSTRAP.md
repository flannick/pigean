# Bootstrap + release checklist

## 1) Initialize repo

```bash
cd pigean
git init
```

## 2) Stage legacy script snapshot

```bash
python scripts/stage_legacy.py --source /ABS/PATH/TO/priors.py
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
## edit config/profiles/common.factor.json first:
## replace __BUNDLE_ROOT__ with your bundle root, e.g. /abs/path/to/pigean/bundles/current

GENE_CSV=$(awk 'NF && $1 !~ /^#/ {print $1}' data/mody.gene.list | awk '!seen[$1]++' | paste -sd ',' -)

../../.venv/bin/python legacy/priors.py \
  --config config/profiles/gene_list.default.json \
  --positive-controls-list "$GENE_CSV" \
  --gene-stats-out results/MODY.gene_stats.out \
  --gene-set-stats-out results/MODY.gene_set_stats.out \
  --params-out results/MODY.params.out
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
