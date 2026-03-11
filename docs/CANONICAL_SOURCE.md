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
- EAGGL runtime entrypoint is `python -m eaggl`
- package modules under `src/pigean/` own the CLI, stage-level orchestration, and runtime edit path
- `src/pigean/app.py` is the package-owned entry module for the normal PIGEAN runtime path
- `src/pigean/main_support.py` is the package-owned runtime wiring/support layer for PIGEAN
- `src/pigean/state.py` is the remaining deep runtime-coupled PIGEAN module
- package modules under `src/eaggl/` own the CLI, stage-level orchestration, and runtime edit path
- `src/eaggl/app.py` is the package-owned entry module for the normal EAGGL runtime path
- `src/eaggl/main_support.py` is the package-owned runtime wiring/support layer for EAGGL
- `src/eaggl/state.py` is the remaining deep runtime-coupled EAGGL module
- `src/pigean_legacy_main.py` has been retired
- `src/eaggl/legacy_main.py` has been retired
- flat `src/pigean_*.py` modules are compatibility shims around package-owned code
- canonical in-repo EAGGL sources live under `src/eaggl/`
- canonical docs are being organized under `docs/pigean/` and `docs/eaggl/`
- canonical tests are being organized under `tests/pigean/` and `tests/eaggl/`
- optional stitched single-file artifacts are generated on demand from modular source and are not checked in

Legacy-core retirement policy:
- `src/pigean_legacy_main.py` has been retired
- `src/eaggl/legacy_main.py` has been retired
- `src/pigean/main_support.py` and `src/eaggl/main_support.py` are package-owned support layers, not flat legacy runtimes
- `src/pigean/state.py` and `src/eaggl/state.py` are active package modules and remain the deepest runtime-coupled code paths
- `src/pigean/state.py` and `src/eaggl/state.py` are the canonical deep engines for now; do not split them further without a concrete ownership seam
- `src/pegs_utils.py` is no longer the catch-all deep owner and should continue to shrink only by moving stable helpers into explicit shared modules
- new logic should land in package modules first; support-layer or state-layer changes should stay narrow and should not reintroduce flat catch-all runtime files

For a concise developer map from `docs/methods.tex` to the owning package modules, see:
- `docs/pigean/METHODS_TO_CODE.md`

Stitched artifact build command:

```bash
cd pigean
../.venv/bin/python scripts/build_stitched_artifacts.py
```

See `docs/STITCHED_ARTIFACTS.md` for the generated outputs and limitations.

For a concise summary of the retired runtime references and their package-owned replacements, see:
- `docs/LEGACY_RETIREMENT_REPORT.md`
