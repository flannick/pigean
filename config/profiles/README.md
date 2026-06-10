# Default profiles

These profiles provide static bundle-backed resources for the PIGEAN repository.

The default profile paths point at the repo-tracked bundle:

```text
bundles/model_large-2026.02.22/data/
```

Run commands from the repository root, or override the resource paths on the command line.

Use directly with `python -m pigean`:

```bash
PYTHONPATH=src python -m pigean gibbs \
  --config config/profiles/gwas.default.json \
  --gwas-in <file>
```

The PIGEAN profiles must not include EAGGL-only factorization or PheWAS projection options. Downstream EAGGL runs should receive the PIGEAN outputs or an EAGGL bundle explicitly.
