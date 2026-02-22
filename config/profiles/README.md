# Default profiles

These profiles provide all static bundle-backed resources so users only pass:
- a profile config
- one runtime input (`--gwas-in`, `--exomes-in`, `--huge-statistics-in`, or `--gene-list-in`)

Use via wrapper:

```bash
python scripts/run_legacy.py --config config/profiles/gwas.default.json --gwas-in <file>
```
