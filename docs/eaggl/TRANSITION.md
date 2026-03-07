# EAGGL Canonical Source And Export Policy

This directory is the canonical documentation home for EAGGL inside the `pigean/` repository.

Canonical source state:
- canonical source: `src/eaggl/`
- canonical scripts: `scripts/eaggl/`
- canonical tests: `tests/eaggl/`
- standalone `../eaggl/` is a downstream export target only

Temporary in-repo launcher strategy:

```bash
PYTHONPATH=src ../../.venv/bin/python -m eaggl factor --help
```

Why this is temporary:
- PIGEAN now has a package entrypoint via `python -m pigean`, but some flat compatibility modules still exist and will be retired in later cleanup milestones
- EAGGL can use `python -m eaggl` already because its canonical code is now under `src/eaggl/`
- later milestones will move more flat shared and compatibility modules under package namespaces

If the standalone `../eaggl/` checkout still needs to be published separately, refresh it from the canonical tree with:

```bash
cd pigean
../.venv/bin/python scripts/eaggl/export_standalone_eaggl.py ../eaggl
```
