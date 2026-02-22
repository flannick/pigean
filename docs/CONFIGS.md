# Config strategy

Use profile configs for static resources and defaults.

Users should provide:
1. `--config config/profiles/<profile>.default.json`
2. exactly one runtime input file (`--gwas-in`, `--exomes-in`, `--huge-statistics-in`, or `--gene-list-in`)

For `--X-list` behavior parity with separate `--X-in`, each line should include explicit batch labels:

```text
mouse:/path/gene_set_list_mouse_2024.txt@mouse
msigdb:/path/gene_set_list_msigdb_nohp.txt@msigdb
```

- `label:` controls display label
- `@batch` controls hyperparameter pooling (p/sigma sharing)
