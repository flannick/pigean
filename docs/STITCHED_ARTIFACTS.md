# Stitched Single-File Artifacts

The modular source tree under `src/` is the only editable source of truth.

Stitched artifacts exist only as an optional build product for:

- single-file browsing
- single-file handoff
- environments where `python -m pigean` or `python -m eaggl` is inconvenient

They are not checked in and should not be edited directly.

## Build

Build both artifacts:

```bash
cd pigean
../.venv/bin/python scripts/build_stitched_artifacts.py
```

Build only one artifact:

```bash
cd pigean
../.venv/bin/python scripts/build_stitched_artifacts.py --artifact pigean
../.venv/bin/python scripts/build_stitched_artifacts.py --artifact eaggl
```

By default, outputs are written to:

```text
dist/stitched/
```

You can override that with:

```bash
../.venv/bin/python scripts/build_stitched_artifacts.py --out-dir /tmp/stitched
```

## Output files

- `dist/stitched/pigean_stitched.py`
- `dist/stitched/eaggl_stitched.py`

Each artifact embeds:

- the relevant runtime modules
- shared `pegs_shared` modules
- a small in-memory importer so the stitched file can run without `PYTHONPATH`

## Runtime examples

```bash
../.venv/bin/python dist/stitched/pigean_stitched.py gibbs --help
../.venv/bin/python dist/stitched/eaggl_stitched.py factor --help
```

## Guarantees

- generated content is deterministic for a fixed source tree
- artifacts are rebuilt from modular source, never used as canonical input
- tests cover both reproducibility and basic runtime smoke behavior

## Limitations

- tracebacks from the stitched file reference stitched line numbers, not original module paths
- artifacts are intended for convenience, not as the preferred development target
