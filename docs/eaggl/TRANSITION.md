# EAGGL Canonical Source And Export Policy

This directory is the canonical documentation home for EAGGL inside the `pigean/` repository.

Canonical source state:
- canonical source: `src/eaggl/`
- canonical scripts: `scripts/eaggl/`
- canonical tests: `tests/eaggl/`
- standalone `../eaggl/` is a downstream export target only

Current in-repo launcher strategy:

```bash
PYTHONPATH=src ../../.venv/bin/python -m eaggl factor --help
```

Current state:
- PIGEAN uses `python -m pigean`
- EAGGL uses `python -m eaggl`
- both flat legacy runtime files have been retired
- package-owned entry/runtime surfaces now live under `src/pigean/` and `src/eaggl/`
- some flat compatibility shim modules still exist for import compatibility, but runtime ownership is package-local

If the standalone `../eaggl/` checkout still needs to be published separately, refresh it from the canonical tree with:

```bash
cd pigean
../.venv/bin/python scripts/eaggl/export_standalone_eaggl.py ../eaggl
```
