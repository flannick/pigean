# EAGGL Docs Namespace

This directory is the canonical in-repo home for EAGGL documentation inside the `pigean/` repository.

Canonical ownership:
- docs here are the source of truth for EAGGL
- the standalone `eaggl/` repo is downstream only
- if a separate standalone distribution is still needed, refresh it from this tree rather than editing it directly

## Start here

Start with [Analyze one trait](WORKFLOWS.md#analyze-one-trait) to run EAGGL on a PIGEAN result and view the factor dashboard. Use the [effective defaults](RECOMMENDED_DEFAULTS.md) and [CLI reference](CLI_REFERENCE.md) to change a setting. The [option inventory](CLI_OPTIONS.md) lists every parser option.

For interpretation, see [gene-set evidence and optional mechanism names](LABELING.md) and the [supplementary methods](methods.pdf).

## Related docs

- `docs/eaggl/INTEROP.md`: interoperability notes
- `docs/eaggl/SHARED_CODE.md`: shared-code model and ownership expectations
- `docs/eaggl/TRANSITION.md`: transition and package-ownership notes
- `docs/eaggl/KNOWN_LIMITATIONS.md`: known limitations
- `docs/eaggl/RELEASE_CHECKLIST.md`: release checklist
- `docs/eaggl/RELEASE_STATUS.md`: release status notes

## Gene-dependence robustness

The [methods supplement](methods.tex) now distinguishes native HC3 binary enrichment, exported factor-GMT logistic statistics, and annotation-derived working covariance. Its [gene-dependence section](gene_dependence_robustness.tex) contains the deterministic HDL Factor 4/7 robustness audit; [compact results and reproduction details](validation/gene_dependence/README.md) accompany it. Shared-annotation sensitivity is not a claim of calibrated binary-hit significance.
