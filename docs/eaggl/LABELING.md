# EAGGL Optional Labeling

Optional factor labeling is separate from core factorization. The mental model is:
factorization learns a basis; labeling summarizes an existing basis. EAGGL supports
both doing this at factor-run finalization time and doing it later from saved
loading tables.

## Label-Only Mode

Use `python -m eaggl label` when the factor loadings already exist and you only
want to add or refresh labels.

Minimal gene-set example:

```bash
PYTHONPATH=src python -m eaggl label \
  --label-gene-set-clusters-in results/gene_set_clusters.out.gz \
  --factors-out results/factors.labeled.out.gz \
  --gene-set-clusters-out results/gene_set_clusters.labeled.out.gz \
  --params-out results/label.params.out.gz
```

Supported loading inputs:

1. `--label-gene-clusters-in`
2. `--label-gene-set-clusters-in`
3. `--label-pheno-clusters-in`
4. `--label-trait-factor-links-in`

At least one loading input is required. If multiple loading inputs are supplied,
their raw `Factor1..FactorK` columns must match exactly. This prevents accidentally
labeling gene, gene-set, and phenotype tables from different EAGGL runs.

Phenotype inputs can be wide (`--label-pheno-clusters-in`) or long
(`--label-trait-factor-links-in`). Long trait-factor links default to the
`nnls_loading` column; override with `--label-trait-factor-link-loading-col`.

Label-only outputs:

1. `--factors-out`
2. `--factor-metrics-out`
3. `--gene-clusters-out`
4. `--gene-set-clusters-out`
5. `--label-pheno-clusters-out`
6. `--trait-factor-links-out`
7. `--params-out`

Label-only mode does not refit factors, choose phi, run projections, or run
PIGEAN. It annotates the supplied factor columns in place. Unless
`--factor-output-scope` is explicitly supplied, label-only mode reports all
supplied factors.

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
7. `--gene-sets-for-labeling`
8. `--factor-top-loading-type`

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
