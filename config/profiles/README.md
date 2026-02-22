# Default profiles

These profiles provide static bundle-backed resources.

Before running, edit `config/profiles/common.factor.json`:
- replace `__BUNDLE_ROOT__` with your installed bundle root (typically `<repo>/bundles/current`).

Use directly with `legacy/priors.py`:

```bash
../../.venv/bin/python legacy/priors.py \
  --config config/profiles/gwas.default.json \
  --gwas-in <file>
```
