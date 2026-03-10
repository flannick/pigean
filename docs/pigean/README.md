# PIGEAN Docs Namespace

This directory is the target home for PIGEAN-specific documentation as the repo is reorganized around:

- `docs/pigean/`
- `docs/eaggl/`

Current authoritative docs still include top-level files such as:

- `docs/ADVANCED_SET_B.md`
- `docs/EAGGL_INTEROP.md`
- `docs/CLI_OPTIONS.md`

These remain authoritative until they are moved or regenerated in the namespaced layout.

Current package-owned PIGEAN runtime structure:
- entry/app: `src/pigean/app.py`
- runtime wiring/support: `src/pigean/main_support.py`
- deep runtime-coupled implementation: `src/pigean/state.py`
- stage modules: `src/pigean/dispatch.py`, `src/pigean/pipeline.py`, `src/pigean/gibbs.py`, `src/pigean/huge.py`, `src/pigean/model.py`, `src/pigean/outputs.py`, `src/pigean/phewas.py`, `src/pigean/x_inputs.py`, `src/pigean/y_inputs.py`

Developer-facing methods ownership map:
- `docs/pigean/METHODS_TO_CODE.md`
