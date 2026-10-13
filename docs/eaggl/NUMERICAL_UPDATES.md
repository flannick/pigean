# Numerical Backends and Output Consistency

## Mode-specific CLI defaults

`--symnmf-update-method auto` selects optimized projected-gradient discovery for
`gene_by_gene`, and legacy discovery for `gene_by_annotation`. An omitted iteration
ceiling resolves to 20,000 for projected gradient or 100 for legacy; an omitted
relative tolerance resolves to 1e-6 or 1e-4 respectively. Explicit settings are
respected. The default discovery matrix remains `gene_by_annotation`.
Direct Python state APIs retain their legacy defaults; automatic selection
occurs in the CLI factor stage, not in every numerical API.

Projected gradient supports symmetric nonnegative single-view targets, zero
diagonal weight and nonnegative loadings with row sums at most one. Unsupported
objectives fail explicitly. It optimizes the logged half-squared off-diagonal
loss plus profiled ARD regularization, not the implicit doubled legacy penalty.
The method owner has delegated the objective convention. Phi values need not
have equivalent effects under the two solvers.

Cached line search reuses target products, with a conservative screening guard
and the original cancellation-controlled calculation near a decision boundary.
The guard is empirical, not a formal roundoff certificate. Fixed-budget endpoints
matched the original solver bit-for-bit for 65 pilot fits at 10,000 and 20,000
updates; ten additional toys check original line-search decisions.

## Adaptive stopping

`--symnmf-stopping adaptive` is the projected-gradient default. Diagnostics are
evaluated every 25 updates. A fit must pass stationarity at the requested
tolerance and total/strongest-five loading drift <=1e-4 over 1,000 updates for
five consecutive checks, then complete a 2,000-update confirmation. Confirmation
requires drift <=1e-3, relative objective gain <=1e-6 and an unchanged scale-aware
pre-projection candidate count. Failed checks reset the confirmation.
The iteration ceiling is a limit, not an instruction to perform 20,000 updates.
`--symnmf-stopping fixed` disables early stopping for reference runs.

This is an explicitly approximate fast mode. In development validation, 64/65
adaptive endpoints agreed with fixed 10,000 and 61/65 with fixed 20,000 under
the continuation gates. Four starts had later magnitude changes of 6-9% despite
very similar factor directions. Passing configured checks does not certify a
final loading magnitude, local/global optimum, or biological validity.
An adaptive fit that reaches its cap without confirmation is unconverged even
if endpoint-only checks pass. Line-search failure cannot count as convergence.

Consensus is matching plus median/mean aggregation and support filtering;
there is no post-consensus reoptimization. Raw CLI consensus remains opt-in
(`--factor-runs 1`, `--consensus-nmf` false). The reviewed portal launcher explicitly
requests five independently seeded restarts and median consensus.

## Checked projections and saved context

Fixed-basis projection remains legacy in the raw CLI for compatibility.
The reviewed launcher explicitly requests:

```bash
--discovery-model gene_by_gene \
--symnmf-update-method projected_gradient \
--symnmf-stopping adaptive \
--factor-max-iterations 20000 --factor-rel-tol 1e-6 \
--factor-runs 5 --consensus-nmf \
--fixed-basis-projection-method verified_quadratic \
--projection-max-cycles 2000 --projection-stationarity-tol 1e-8 \
--factor-output-scope all --cluster-row-min-max-loading 0 \
--gene-clusters-full-via-gene-sets-out results/gene_clusters_full_via_gene_sets.out.gz \
--projection-context-out results/projection_context.npz \
--numerical-diagnostics-out results/numerical_diagnostics.json
```

Checked projections solve the requested constrained nonnegative quadratic
objective and record per-output stationarity. They do not establish calibrated
probabilities. Near-duplicate bases can still fail the strict gate, especially
with zero regularization and a sum constraint. Such calls remain unconverged.
The launcher refuses success if any checked projection is unconverged. Discovery
caps or consensus fallback produce a review-required marker and exit status 4
rather than success. Exact projection context is saved when additional phenotype
profiles are supplied, not when there is no phenotype projection to reproduce.

`consensus_stats` preserves its leading columns and appends stopping diagnostics;
consumers must use column names. `--numerical-diagnostics-out` also saves restart
and projection diagnostics as JSON. Additional phenotype projections do not
become discovery anchors. Factor-indexed files follow final factor ordering.
Empty discovery returns exit status 3, not success with empty outputs.

A rounded loading table alone does not preserve weights, regularization or
discovery masks. To reproduce gene-basis phenotype projections from supplied
factors, use the same inputs/settings and the exact sidecar:

```bash
--factor-gene-clusters-in results/gene_clusters.out.gz \
--projection-context-in results/projection_context.npz
```

The sidecar validates gene/factor identities, rounded values, settings and array
integrity. It does not support gene-set-basis reload or continuation of discovery.

## Final-factor labels and handoff

`scripts/eaggl/run_portal_trait.py` requires live LLM labels by default, after
final consensus and projections are written. Set `EAGGL_LLM_MODEL` and
`EAGGL_LLM_AUTH_KEY` (or `OPENAI_API_KEY`) in the environment. OpenAI is the
implemented provider in this pinned code; no particular model has been validated
as best. Reuse a known-working supported model for the live smoke test.

The helper uses top-five raw-loading genes and projected gene sets, without
phenotype names. It preserves noncontiguous factor IDs and rejects missing,
misnumbered or oversized responses rather than silently accepting fallback
names. It saves model/prompt provenance and an ID-aligned label table. The
launcher publishes labels consistently into factor and cluster tables, changing
text only; the original factor summary is retained as `factors.unlabeled.out.gz`.

Missing credentials fail before numerical work. Failed labeling retains numerical
outputs but cannot create a success marker. `--mock-labels` is explicitly a
software test, not biological annotation, and never creates `_SUCCESS`.
The launcher refuses existing output directories and writes locally only.
It may download explicit S3 inputs but never uploads or overwrites S3 results.
