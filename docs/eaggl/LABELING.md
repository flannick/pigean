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

## Naming leaves within a fit

With a labeling provider, `--label-strategy cross-factor` (the default on this
branch) gives the labeler an anonymous roster of leaves from the same fit.
Each row lists up to five ranked names per selected evidence source. Each
request identifies one target leaf and supplies its detailed evidence. The
roster helps distinguish closely related processes, and leaves may share a
name when their evidence supports the same interpretation. Each additional
source combination gets its own roster. Leaves missing an explicitly required
source are omitted from that roster.

Use `--label-strategy independent` for the previous per-leaf prompt, or to
reduce prompt size for a very large fit. Both strategies work during factor
finalization and on previously saved loading tables. There is one request
per eligible leaf and distinct source combination. Requests in cross-factor
mode repeat the compact roster, so input token use increases with fit size.

The research dashboard's single-trait hierarchy also names internal nodes
bottom-up. Each parent sees the names of every descendant leaf, grouped by
immediate child. Identical names within a child carry a count. Counts describe
how many fitted leaves share a name. The prompt also includes each child's
name and leading evidence plus the pooled gene and gene-set evidence. Broad
families can receive umbrella or paired names. Stable node identifiers
continue to identify leaves or families that share a name.

## Optional study context

Both `eaggl factor` and `eaggl label` accept free text, for example
`--label-context "Samples are human pancreatic islets"`. Omit this flag to
use the selected loading evidence alone. The labeler does not infer a disease
from the run name. Context can clarify terminology when the evidence supports
more than one interpretation. It applies to every model-generated label in
the run, including additional source combinations. When holding out a phenotype
for validation, omit its identity and related clues from the context as well.
The flag alone does not enable a provider or change the fit.

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

## Gene-set evidence and mechanism names

The default `--gene-set-clusters-format evidence` keeps each raw `FactorN`
coefficient. For each factor, the table also reports the following quantities.

| Field | Interpretation |
|---|---|
| Coverage | Fraction of the factor's total gene loading carried by members of this set |
| Enrichment | Coverage divided by the fraction of basis genes belonging to the set. One means background coverage |
| MeanMemberLoading | Average factor loading among set members |
| CumulativeCoverage | Fraction of factor mass in the union of this set and all preceding positive-loading sets. Sort by descending coefficient, breaking ties by set ID. Genes count once. The value at rank X gives the coverage of the top X sets before row or labeling filters |
| SupportingGenes | Up to three set members with the largest positive factor loadings |
| BasisGenes, MemberGenes | Gene universe size and number of set members in that universe |

For discovery fits the universe is the discovery genes. For projection from
supplied factors it is the genes shared by the input matrix and the supplied
basis. Positive annotation entries count as membership. An undefined ratio is
written as NA. The overlap columns describe the basis even in anchor-specific
output. The fitted coefficients retain the existing anchor multiplier.
A fitted coefficient describes the strength of the projection, with competition
between factors in joint mode. Coverage has a sampling interpretation if one
draws a gene in proportion to its factor loading. In saved label-only inputs
without membership evidence, unavailable fields are reported as NA. To obtain
the previous raw, cosine and Euclidean output columns and gene-set ranking
policy, use `--gene-set-clusters-format legacy`.

## Choose evidence for the primary and additional labels

Both `factor` and `label` use up to five ranked items per selected source. The
default primary label uses genes and gene sets. Use `--label-sources` to change
the primary label. Repeat `--additional-label-sources` for each additional
label you want to compare on the same factors. Each option takes any nonempty
combination of `genes`, `gene-sets` and `phenotypes`, separated by commas.

For example, this labels the same saved factors three ways. The primary label
holds phenotypes out, the second uses phenotypes alone and the third uses all
three sources.

```bash
python -m eaggl label \
  --label-gene-clusters-in gene_clusters.out.gz \
  --label-gene-set-clusters-in gene_set_clusters.out.gz \
  --label-pheno-clusters-in pheno_clusters.out.gz \
  --additional-label-sources phenotypes \
  --additional-label-sources genes,gene-sets,phenotypes \
  --lmm-auth-key env --factors-out factors.labeled.out.gz
```

The factor table stores the primary name in `label`, its requested evidence in
`label_sources` and additional names in columns such as `label_phenos` or
`label_genes_gene_sets_phenos`. A gene-set prompt includes the set names, fitted
loadings, coverage and enrichment. Per-set supporting genes can appear only if
genes are selected and the gene is among that factor's top five. A genes-only
or phenotypes-only prompt contains just the corresponding names. Gene-set names
are preserved as supplied and may themselves mention genes or phenotypes.
Holding a source out of naming tests interpretation of an already fitted factor.
Inputs used during discovery or projection can still influence its loadings.

Each distinct combination makes at most one request per final factor. Reordering
or repeating a combination reuses its label. For an explicitly requested
combination, a factor lacking any selected source receives `NA` and uses zero
requests. With the implicit default, either available primary source can supply
a label. Without credentials, or if an ordinary provider call fails, the
deterministic fallback uses only the requested sources. The strict portal
handoff requires a valid response for every requested label. The older
`--label-gene-sets-only`, `--label-include-phenos` and `--label-individually`
options remain available.

For optional live labeling, set `OPENAI_API_KEY` in the environment running
EAGGL, then add `--lmm-auth-key env`. The environment variable alone does
not activate provider calls in the factor or label CLI. The small default
model is `gpt-6-luna`; use `--lmm-model gpt-6.2-sol` to try a larger model. For example:

```bash
python -m eaggl label \
  --label-gene-set-clusters-in gene_set_clusters.out.gz \
  --label-gene-clusters-in gene_clusters.out.gz \
  --lmm-auth-key env --factors-out factors.labeled.out.gz
```
