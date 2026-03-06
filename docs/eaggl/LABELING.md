# EAGGL Optional Labeling

Optional factor labeling is intentionally separate from core factorization.

## Default Behavior

If `--lmm-auth-key` is not provided, EAGGL does not call any external labeling provider.
Factor labels fall back to deterministic labels derived from top gene sets, genes, and phenotypes already present in the factor results.

This means:

1. core factorization does not require network access
2. core factorization tests do not depend on provider code paths
3. provider failures do not affect factor computation itself

## Current Provider Support

Production-enabled provider:

1. `openai`

Reserved but not implemented:

1. `gemini`
2. `claude`

If one of the reserved providers is requested, EAGGL fails fast with a clear CLI error.

## Relevant Flags

1. `--lmm-auth-key`
2. `--lmm-model`
3. `--lmm-provider`
4. `--label-gene-sets-only`
5. `--label-include-phenos`
6. `--label-individually`

## Provider Boundary

Core label construction lives in `src/eaggl/labeling.py`.

Provider adapters live in `src/eaggl/labeling_providers.py`.

The provider module is loaded lazily only when LLM labeling is actually requested. Non-LLM runs should not import or exercise provider adapters.

## Extending Providers

To add a new provider:

1. implement a provider class in `src/eaggl/labeling_providers.py`
2. add it to `resolve_labeling_provider(...)`
3. add unit tests for provider selection and failure behavior
4. keep provider-specific request formatting out of `src/eaggl/factor.py`
5. preserve the rule that factorization remains valid without provider imports
