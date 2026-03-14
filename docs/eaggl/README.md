# EAGGL Docs Namespace

This directory is the canonical in-repo home for EAGGL documentation inside the `pigean/` repository.

Canonical ownership:
- docs here are the source of truth for EAGGL
- the standalone `eaggl/` repo is downstream only
- if a separate standalone distribution is still needed, refresh it from this tree rather than editing it directly

## Start here

Use these documents in order:

1. `docs/eaggl/CLI_REFERENCE.md`
   - practical manual for running `python -m eaggl`
   - curated explanation of the main workflows, anchors, factor controls, labeling flags, and outputs
2. `docs/eaggl/WORKFLOWS.md`
   - F1-F9 workflow map with minimal runnable command patterns
3. `docs/eaggl/methods.tex`
   - theory and mathematical formalization of the EAGGL factor model and workflow families
4. `docs/eaggl/LABELING.md`
   - optional factor-labeling behavior and why labeling remains part of `factor`
5. `docs/eaggl/CLI_OPTIONS.md`
   - exhaustive machine-generated option inventory

## Related docs

- `docs/eaggl/INTEROP.md`: interoperability notes
- `docs/eaggl/SHARED_CODE.md`: shared-code model and ownership expectations
- `docs/eaggl/TRANSITION.md`: transition and package-ownership notes
- `docs/eaggl/KNOWN_LIMITATIONS.md`: known limitations
- `docs/eaggl/RELEASE_CHECKLIST.md`: release checklist
- `docs/eaggl/RELEASE_STATUS.md`: release status notes
