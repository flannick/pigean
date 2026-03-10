# Legacy Retirement Report

Purpose:
- record what retired runtime references were removed
- document what package-owned modules replaced them
- give a current snapshot of the remaining intentional deep implementation files

Current retired runtime files:
- `src/pigean_legacy_main.py`
- `src/eaggl/legacy_main.py`

Current status:
- both files are deleted
- `python -m pigean` runs through package-owned modules
- `python -m eaggl` runs through package-owned modules
- the legacy symbol report shows zero direct imports and zero dynamic imports of retired runtime modules

Removed reference classes:
- direct imports of `pigean_legacy_main`
  - replacement: package-owned modules under `src/pigean/`
- direct imports of `eaggl.legacy_main`
  - replacement: package-owned modules under `src/eaggl/`
- dynamic imports of retired runtime modules
  - replacement: explicit package imports and package-owned support/state modules
- module-object dispatch patterns
  - replacement: explicit support objects and package-owned service wiring

Key package-owned replacements:
- PIGEAN entry/runtime:
  - `src/pigean/app.py`
  - `src/pigean/dispatch.py`
  - `src/pigean/main_support.py`
  - `src/pigean/state.py`
- EAGGL entry/runtime:
  - `src/eaggl/app.py`
  - `src/eaggl/dispatch.py`
  - `src/eaggl/main_support.py`
  - `src/eaggl/state.py`

Current legacy-symbol snapshot:
- `src/pigean_legacy_main.py`: absent
- `src/eaggl/legacy_main.py`: absent
- direct import sites:
  - `pigean_legacy_main`: `0`
  - `eaggl.legacy_main`: `0`
- dynamic import sites:
  - `pigean_legacy_main`: `0`
  - `eaggl.legacy_main`: `0`
- module-object dispatch counts:
  - `dispatch.run_main_pipeline(_legacy_main)`: `0`
  - `dispatch.run_main_pipeline(sys.modules[__name__])`: `0`
  - `_eaggl_dispatch.run_main_pipeline(_build_main_domain())`: `0`

What remains intentionally deep:
- `src/pigean/state.py`
  - still owns the main remaining PIGEAN inner implementation and runtime-coupled helpers
- `src/eaggl/state.py`
  - still owns the main remaining EAGGL runtime-coupled inner implementation

These are active package modules, not retired compatibility files.
