# Canonical Source Layout

The `pigean/` repository is the canonical source of truth for both tools:

- PIGEAN
- EAGGL

Transition policy:
- active architecture work happens in this repo first
- the standalone local `eaggl/` checkout remains untouched during the migration
- if standalone EAGGL needs to stay available, it should be treated as a downstream export/mirror rather than a co-canonical codebase

Current state:
- PIGEAN runtime entrypoint remains `src/pigean.py`
- canonical in-repo EAGGL sources live under `src/eaggl/`
- canonical docs are being organized under `docs/pigean/` and `docs/eaggl/`
- canonical tests are being organized under `tests/pigean/` and `tests/eaggl/`

Important constraint:
- `src/pigean.py` currently blocks creation of a sibling `src/pigean/` package directory with the same name
- that move is deferred to the next entrypoint/module-safety milestone, where the script entrypoint will be converted into a package-based launcher
