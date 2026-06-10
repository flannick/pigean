# Bundles

Bundle goals:
- Keep large static resources out of git
- Version resource sets immutably
- Keep configs pointed at the checked-in, versioned bundle unless users explicitly override paths

## Repo-tracked bundle

The current repository ships one default model bundle:

```text
bundles/model_large-2026.02.22/data/
```

Important files include:

- `gene_set_list_mouse_2024.txt`
- `gene_set_list_msigdb_nohp.txt`
- `gene_set_list_ocr_human.txt`
- `gene_set_list_string_notext_medium.txt`
- `portal_gencode.gene.map`
- `NCBI37.3.plink.gene.loc`
- `NCBI37.3.plink.gene.exons.loc`

The gene-set files are named `.txt`, but they are valid `--X-in` sparse gene-set-list inputs. A `.gmt` extension is not required.

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

External bundle installers may install into `bundles/<name>-<version>/`; command examples in this repo use the checked-in `bundles/model_large-2026.02.22/data/` paths.
