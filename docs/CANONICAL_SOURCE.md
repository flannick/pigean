# Canonical Source Layout

The `pigean/` repository is the canonical source of truth for both tools:

- PIGEAN
- EAGGL

Canonical ownership policy:
- active architecture work happens in this repo first
- the standalone local `eaggl/` checkout is a downstream export target only
- no source changes should originate in the standalone `eaggl/` checkout
- if standalone EAGGL needs to stay available, refresh it from this repo rather than maintaining a co-canonical codebase

Recommended downstream refresh command:

```bash
cd pigean
../.venv/bin/python scripts/eaggl/export_standalone_eaggl.py ../eaggl
```

Current state:
- PIGEAN runtime entrypoint is `python -m pigean`
- canonical in-repo EAGGL sources live under `src/eaggl/`
- canonical docs are being organized under `docs/pigean/` and `docs/eaggl/`
- canonical tests are being organized under `tests/pigean/` and `tests/eaggl/`
