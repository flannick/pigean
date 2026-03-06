# EAGGL Canonical Transition

This directory is the canonical documentation home for EAGGL inside the `pigean/` repository.

Current transition state:
- canonical source snapshot: `src/eaggl/`
- canonical scripts snapshot: `scripts/eaggl/`
- canonical tests snapshot: `tests/eaggl/`
- standalone `../eaggl/` remains untouched and is treated as downstream during this migration

Temporary in-repo launcher strategy:

```bash
PYTHONPATH=src ../../.venv/bin/python -m eaggl factor --help
```

Why this is temporary:
- `src/pigean.py` still exists as a flat script and blocks the final package-based `src/pigean/` layout
- EAGGL can use `python -m eaggl` already because its canonical code is now under `src/eaggl/`
- PIGEAN will follow once the next entrypoint/module-safety milestone converts it to the same package pattern
