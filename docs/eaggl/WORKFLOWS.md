# EAGGL factor workflows

## Analyze one trait

Start with a completed PIGEAN run for your trait and the annotation matrix used in that run. The example below assumes an installed package and PIGEAN results in `runs/my_trait/pigean`. Replace the paths with yours. A single run supplies everything needed for factor discovery and its dashboard.

```bash
mkdir -p runs/my_trait/eaggl
python -m eaggl factor \
  --X-in annotations.gmt.gz \
  --gene-stats-in runs/my_trait/pigean/pigean.gene_stats.out.gz \
  --gene-set-stats-in runs/my_trait/pigean/pigean.gene_set_stats.out.gz \
  --factors-out runs/my_trait/eaggl/factors.out.gz \
  --gene-clusters-out runs/my_trait/eaggl/gene_clusters.out.gz \
  --gene-set-clusters-out runs/my_trait/eaggl/gene_set_clusters.out.gz

python -m pigean.dashboard \
  --pigean-run my_trait:runs/my_trait/pigean \
  --eaggl-run my_trait:gene_x_gene:runs/my_trait/eaggl \
  --html-out runs/my_trait/dashboard.html
```

Open `runs/my_trait/dashboard.html` to inspect the factors, their leading genes and the gene sets that support each interpretation. Fresh discovery uses the additive gene-pair model, selects phi automatically, and pools reproducible factors across five restarts per candidate. [Effective defaults](RECOMMENDED_DEFAULTS.md) · [Optional mechanism naming](LABELING.md) · [Dashboard inputs and views](../PIGEAN_DASHBOARD.md)

If you already have qualified union-model fits for this trait, use the [saved-fit guide](../../research/eaggl_leaf_selection/README.md) to select phi, inspect leaf QC and build its hierarchy. Those steps require the discovery capture and fitted candidates. The fresh discovery command above produces additive factor tables.

The remaining sections describe alternate inputs, projections, and command options.

## Common Setup

The reference examples below use the Python interpreter where EAGGL is installed:

```bash
PYTHON=python
```

Useful common outputs:

```bash
--factors-out results/factors.out \
--gene-set-clusters-out results/gene_set_clusters.out \
--gene-clusters-out results/gene_clusters.out \
--trait-factor-links-out results/trait_factor_links.out \
--params-out results/params.out
```

Discovery-stage note:

1. EAGGL now distinguishes retained annotations, discovery annotations, and projected annotations.
2. By default, factor discovery is learned on redundancy-balanced discovery families using `--discovery-similarity-threshold 0.35`, and the resulting outputs mark discovery rows with `in_discovery`.
3. Discovery-family diagnostics now include `discovery_family_mean_similarity` and `discovery_family_effective_size` alongside `discovery_family_size` and `discovery_weight`.
4. The default discovery weighting is `effective_size`, which uses representative support multiplied by an effective family size instead of the removed family-average weighting.
5. All retained gene sets are still projected and written after discovery, so adding correlated annotations mostly deepens annotation rather than redefining `W`.
6. `--discovery-model gene_by_annotation` explicitly selects the alternative rectangular retained-annotation workflow described above.
7. The default `--discovery-model gene_by_gene` builds a retained-gene by retained-gene pairwise matrix from corrected annotation betas, fits a symmetric nonnegative factorization in gene space, and then projects retained gene sets onto the learned gene factors.
8. In `gene_by_gene` mode, all retained gene sets contribute to pairwise evidence; discovery-family leader subsetting and redundancy weighting are ignored.
9. In `gene_by_gene` mode, v1 currently requires `--factor-backend full`, `--learn-phi-backend sentinel_pruned`, and the default transposed factor matrix.
10. In `gene_by_gene` mode, if `--phi` is not explicitly set, EAGGL starts factor learning and any `--learn-phi` search from `0.01` rather than `0.05`.
11. `--gene-gene-profligate-correction linear` is an opt-in gene-by-gene diagnostic/production control that subtracts a simple fitted retained-annotation-count effect from raw pair log evidence before the usual pair-probability calibration. The default `none` path is unchanged.
12. `--annotation-bridge-metrics-out` is an optional gene-by-gene post-processing output that reports which annotations bridge otherwise distinct fitted gene factors.
13. `--anchor-aggregation multi` is the default multi-anchor mode for both discovery models; `any` uses explicit noisy-OR union evidence. With one anchor trait, both modes reduce exactly to ordinary single-trait anchoring.
14. `gene_by_gene` still learns one shared gene-factor basis and projects annotations onto that basis afterward; it does not write anchor-specific cluster outputs.

Debug workflow selection without running factorization:

```bash
$PYTHON -m eaggl factor --print-effective-config [workflow flags ...]
```

## Input Contracts

Core matrix/stat inputs (direct mode):

1. `--X-in` or `--X-list`
2. `--gene-stats-in`
3. `--gene-set-stats-in`

Use `--X-in` for a direct `.gmt` sparse matrix file. Use `--X-list` only for a text file that lists matrix inputs. If a direct `.gmt` is passed to `--X-list`, EAGGL accepts it for compatibility but warns and treats it like `--X-in`.

Fresh gene-by-gene discovery defaults to five restarts and pooled median consensus. Use `--factor-runs 1` for a single fit, `--no-consensus-nmf` for best-of-N selection, or `--consensus-matching reference` for the previous reference-centered matching. Rectangular discovery retains single-run defaults.
Fresh gene-by-gene discovery chooses phi automatically with the requirements selector. It starts at 0.01, evaluates 0.005 and 0.02, and permits at most two further candidates when a boundary materially improves evidence preservation. The selected fit is reused. Specify `--phi 0.01` for a fixed comparison. The composite and target-size selectors are explicit alternatives. [Selection rules and audit outputs](REQUIREMENTS_PHI.md)

For the older search objectives, EAGGL also provides two scalable phi-search backends:

1. `--learn-phi-backend sentinel_pruned`
   - the legacy shortcut
   - evaluates candidates on a correlation-pruned sentinel panel of up to `1000` genes and `1000` gene sets by default
2. `--learn-phi-backend blockwise_global_w`
   - an alternative rectangular approximation to a full retained-panel search (requires `--discovery-model gene_by_annotation`)
   - keeps one shared global gene-factor basis and shared ARD state across block-local gene-set solves
   - evaluates all retained gene sets in blocks instead of replacing them with a single decorrelated sentinel panel

Likewise, the final factorization can run with either:

1. `--factor-backend full`
   - the original in-memory solve
2. `--factor-backend blockwise_global_w`
   - a blockwise solve over the full retained gene-set collection with one shared global gene-factor basis

The symmetric `--discovery-model gene_by_gene` mode is separate from those rectangular backends:

1. it uses corrected `beta` values for pairwise evidence (`--gene-gene-beta-source beta`)
2. it converts retained shared-annotation evidence into pairwise probabilities before factoring
3. it currently supports only `--factor-backend full`
4. it ignores discovery-family subsetting and weighting flags because pairwise evidence is built from all retained gene sets
5. with multiple anchor traits, each trait-specific pair target is built exactly as in single-trait mode; `multi` factors the equal average of all per-anchor target matrices, while `any` uses noisy-OR union over those matrices
6. optional `--gene-gene-profligate-correction linear` uses retained/scored annotation counts from the annotations entering the gene-gene matrix to correct the raw pair log evidence for broadly annotated genes

Use `--blockwise-gene-set-block-size`, `--blockwise-epochs`, `--blockwise-shuffle-blocks`, `--blockwise-warm-start`, `--blockwise-max-blocks`, and `--blockwise-report-out` to tune or audit the blockwise backend. When `--learn-phi-backend blockwise_global_w` is used, neighboring phi candidates are warm-started from the closest previously fitted phi on the log scale when possible. Use `--factor-phi-metrics-out`, `--factor-phi-factors-out`, `--factor-phi-gene-set-clusters-out`, and `--factor-phi-gene-clusters-out` when you want audit tables for every tested phi candidate. Cluster output rows whose maximum raw factor loading is below `--cluster-row-min-max-loading` are omitted from reported cluster files.

Factor output policy:

1. Fresh gene-by-gene discovery prints all retained consensus factors by default, including their tier labels. Other modes retain primary-only output.
2. Primary factors have `combined_mass_fraction >= 0.005`; secondary factors have `combined_mass_fraction >= 0.001`; smaller factors are treated as filtered tail factors.
3. Use `--factor-output-scope primary` or `primary_secondary` to restrict the displayed tiers; `all` does not override consensus support requirements.
4. `factor_metrics.out` and `factor_phi_metrics.out` remain exhaustive diagnostics over all raw fitted factors.
5. Factor identifiers are preserved rather than renumbered after filtering.

PheWAS matrix inputs:

1. `--gene-phewas-stats-in`
2. `--gene-set-phewas-stats-in`

Phenotype annotation policy:

1. use fixed-W trait linkage for the primary public phenotype annotation layer
2. write the long-form linkage table with `--trait-factor-links-out`
3. interpret `nnls_loading` as fixed-W projection of the selected trait support surface after converting combined/log-BF values to probabilities with `background_prior`
4. interpret `beta`, `beta_uncorrected`, `beta_tilde`, `se`, `z`, and `p_value` as factor-as-gene-set association summaries
5. use `--trait-factor-linkage-factor-gene-threshold` to set the minimum factor gene loading included when exporting factor GMTs for PIGEAN enrichment
6. use `--factor-gmt-out` to export thresholded factors for a manual PIGEAN `multi-y` run

Projection-only trait linkage from the gene basis:

```bash
$PYTHON -m eaggl factor \
  --factor-gene-clusters-in results/gene_clusters.out.gz \
  --gene-phewas-stats-in /path/to/gene_phewas_stats.out.gz \
  --trait-factor-links-out results/trait_factor_links.projected.out.gz
```

Projection-only trait linkage from the gene-set basis:

```bash
$PYTHON -m eaggl factor \
  --project-phenos-from-gene-sets \
  --factor-gene-set-clusters-in results/gene_set_clusters.out.gz \
  --gene-set-phewas-stats-in /path/to/gene_set_phewas_stats.out.gz \
  --trait-factor-links-out results/trait_factor_links.projected.out.gz
```

Projection-only trait linkage plus factor-PheWAS:

```bash
$PYTHON -m eaggl factor \
  --factor-gene-clusters-in results/gene_clusters.out.gz \
  --gene-phewas-stats-in /path/to/gene_phewas_stats.out.gz \
  --trait-factor-links-out results/trait_factor_links.projected.out.gz \
  --run-factor-phewas \
  --factor-phewas-stats-out results/factor_phewas_stats.out.gz
```

Full-gene projection policy:

1. When `--gene-clusters-full-out` is requested, the default projection basis is gene-set-routed (`gene_set_loadings`). This keeps full-gene display loadings tied to the retained gene-set factor representation.
2. The explicit `--gene-clusters-full-via-gene-sets-out` output always uses gene-set-routed projection and is optional; it has not passed the discovery workflow biological-preservation qualification.
3. Use `--full-gene-projection-method direct_gene_gene` with `--gene-clusters-full-out` only for a direct gene-gene diagnostic or comparison; `auto` preserves the legacy dispatch behavior.
4. Dashboards default to discovery gene loadings. Explicit full-gene sources remain selectable; factor graphs use the input table specified.

Optional projection-only full-gene output from gene-set factors:

```bash
$PYTHON -m eaggl factor \
  --factor-gene-set-clusters-in results/gene_set_clusters.out.gz \
  --X-in /path/to/annotations.gmt \
  --gene-clusters-full-via-gene-sets-out results/gene_clusters_full.via_gene_sets.out.gz
```

Projection-only diagnostic direct full-gene output from gene factors:

```bash
$PYTHON -m eaggl factor \
  --factor-gene-clusters-in results/gene_clusters.out.gz \
  --X-in /path/to/annotations.gmt \
  --gene-set-stats-in /path/to/gene_set_stats.out.gz \
  --full-gene-projection-method direct_gene_gene \
  --gene-clusters-full-out results/gene_clusters_full.direct.out.gz
```

Projection-only direct and gene-set-mediated full-gene outputs in one command:

```bash
$PYTHON -m eaggl factor \
  --factor-gene-clusters-in results/gene_clusters.out.gz \
  --factor-gene-set-clusters-in results/gene_set_clusters.out.gz \
  --X-in /path/to/annotations.gmt \
  --gene-set-stats-in /path/to/gene_set_stats.out.gz \
  --full-gene-projection-method direct_gene_gene \
  --gene-clusters-full-out results/gene_clusters_full.direct.out.gz \
  --gene-clusters-full-via-gene-sets-out results/gene_clusters_full.via_gene_sets.out.gz
```

Bundle mode:

1. `--eaggl-bundle-in <bundle.tar.gz>` can provide defaults for core inputs
2. explicit CLI flags always override bundle defaults

## Workflow Matrix

### F1: Single Phenotype Anchoring (default stats path)

Required:

1. no special anchor flags
2. standard factor inputs (`X + gene stats + gene set stats`)

Command:

```bash
$PYTHON -m eaggl factor \
  --X-in /path/to/X.tsv.gz \
  --gene-stats-in /path/to/gene_stats.out \
  --gene-set-stats-in /path/to/gene_set_stats.out \
  --factors-out results/F1.factors.out
```

Gene-by-gene variant:

```bash
$PYTHON -m eaggl factor \
  --X-in /path/to/X.tsv.gz \
  --gene-stats-in /path/to/gene_stats.out \
  --gene-set-stats-in /path/to/gene_set_stats.out \
  --discovery-model gene_by_gene \
  --factor-backend full \
  --learn-phi-backend sentinel_pruned \
  --factors-out results/F1.gene_gene.factors.out \
  --annotation-bridge-metrics-out results/F1.annotation_bridge_metrics.tsv.gz \
  --annotation-bridge-suggested-exclude-out results/F1.annotation_bridge_suggested_exclude.txt
```

`annotation_bridge_metrics.tsv.gz` includes review diagnostics and one conservative
automatic exclusion flag. `flag_review` / `flag_bridge_candidate` mark broad bridge
candidates for inspection. `flag_suggest_exclude` marks high-confidence diffuse
bridges from empirically broad/noisy sources and is the only flag used to write
`annotation_bridge_suggested_exclude.txt`. The same rule can be rerun after a
full-Gibbs exclusion refit; newly suggested annotations reflect the changed model,
not a different exclusion mode.

### F2: Standalone Gene-list Enrichment

Required:

1. `--gene-list` or `--gene-list-in`
2. `--X-in` or another X-matrix source

Behavior:

1. EAGGL uses the loaded X-gene universe as the enrichment background
2. it runs a hypergeometric test for each loaded gene set against the input gene list
3. it keeps gene sets with Benjamini-Hochberg `q <= --gene-list-max-fdr-q` (default `0.05`)
4. retained gene sets are weighted by `-log(P) / sqrt(gene_set_size)`
5. genes are unweighted and all genes from the retained gene sets are brought into the final factoring matrix

Compatibility aliases:

1. `--positive-controls-list`
2. `--positive-controls-in`

Command:

```bash
$PYTHON -m eaggl factor \
  --X-in /path/to/X.tsv.gz \
  --gene-list INS,GCK,HNF1A \
  --gene-set-stats-out results/F2.gene_set_stats.out \
  --gene-stats-out results/F2.gene_stats.out \
  --factors-out results/F2.factors.out
```

### F3: Single Phenotype + Projection from PheWAS

Required:

1. base F1 inputs
2. either `--gene-phewas-stats-in` or `--gene-set-phewas-stats-in` (or both)

Command:

```bash
$PYTHON -m eaggl factor \
  --X-in /path/to/X.tsv.gz \
  --gene-stats-in /path/to/gene_stats.out \
  --gene-set-stats-in /path/to/gene_set_stats.out \
  --gene-phewas-stats-in /path/to/gene_phewas_stats.out \
  --factors-out results/F3.factors.out
```

### F4: Phenotype-Input Anchoring

Required:

1. matched gene phenotype statistics and gene-set phenotype statistics
2. either repeated PheWAS tables with trait labels, or repeated labeled single-trait stats files using `LABEL=path`
3. every loaded trait must have both gene and gene-set evidence after filtering

Command:

```bash
$PYTHON -m eaggl factor \
  --X-in /path/to/X.tsv.gz \
  --gene-phewas-stats-in /path/to/gene_phewas_stats.out \
  --gene-set-phewas-stats-in /path/to/gene_set_phewas_stats.out \
  --factors-out results/F4.factors.out
```

Single-trait stats can also be supplied as matched labeled pairs:

```bash
$PYTHON -m eaggl factor \
  --X-in /path/to/X.tsv.gz \
  --gene-stats-in T2D=/path/to/t2d.gene_stats.out \
  --gene-set-stats-in T2D=/path/to/t2d.gene_set_stats.out \
  --gene-stats-in CAD=/path/to/cad.gene_stats.out \
  --gene-set-stats-in CAD=/path/to/cad.gene_set_stats.out \
  --factors-out results/F4.factors.out
```

EAGGL treats all complete input traits as anchors. Legacy explicit phenotype, gene, and gene-set anchor flags have been removed; use phenotype-resolved inputs instead.

## Workflow Validation and Guardrails

1. Use `--print-effective-config` first to verify the selected workflow ID and required inputs.
2. Missing required inputs hard-fail with an actionable message.
3. For overlapping flags, EAGGL applies workflow-specific precedence and logs warnings for ignored inputs.
4. PheWAS stages log explicit input-I/O mode:
   - `mode=reuse_loaded_matrix` when a compatible loaded matrix is reused.
   - `mode=re_read_file` when stage inputs must be read from file again.

## Removed Legacy GLS Path

EAGGL does not support the historical GLS/whitened-Y path.

1. Removed aliases hard-fail if passed (`--run-gls`, `run_gls`, `store_cholesky`).
2. Correlation-aware behavior in EAGGL uses the retained corrected-OLS path only.

## Gene-set Filter Relaxation

When EAGGL invokes shared read-X filtering, `--increase-filter-gene-set-p` is treated
as a minimum kept-fraction target used to relax prefiltering if needed.
Post-read filtering does not tighten this threshold.

## References

1. Deterministic workflow baseline generator: `scripts/freeze_factor_workflow_effective_configs.sh`
2. Effective-config fixtures: `tests/data/reference/factor_workflow_effective_config/`
3. PIGEAN handoff details: `docs/eaggl/INTEROP.md`

### Simplified Trait Linkage

EAGGL trait-factor linkage is intentionally limited to fixed-W projection loadings. Combined/log-BF phenotype values are converted to probabilities using the run `background_prior`; sparse inputs preserve implicit zeros as zero probability unless dense computation is requested. Factors are exported as weighted gene sets after zeroing gene loadings below `--trait-factor-linkage-factor-gene-threshold` (default `0.05`). `--factor-gmt-out` writes those thresholded factor gene sets for PIGEAN `multi-y` runs when beta statistics are needed.

Gene and phenotype cluster tables, and gene-set tables in legacy format, include raw `Factor*` loadings, `Cosine_Factor*`, and `Euclidean_Factor*` columns. `Cosine_FactorK` is the cosine similarity between the row's factor-loading vector and the indicator vector for factor `K`; `Euclidean_FactorK` is the Euclidean distance to that same one-hot factor indicator.

The default gene-set evidence format retains the raw coefficients and adds coverage, enrichment, mean member loading, cumulative coverage and supporting genes. [Field definitions and naming workflow](LABELING.md#gene-set-evidence-and-mechanism-names)

## Merge results from multiple runs

If you have independently analyzed two or more traits, you can group their factors in a shared hierarchy. Pass one gene-by-leaf table per trait. For the single-trait workflow above, use the `gene_clusters.out.gz` output from each run. For runs using the saved union-fit workflow, use its `leaf_loadings.tsv.gz` of QC-passing consensus leaves.
Each file needs a unique `Gene` column and raw `Factor1`, `Factor2`, … columns
bounded between zero and one. Original factor numbers may have gaps. Annotation
summaries (`--factors-out`), normalized display columns and parent nodes are
unsuitable as inputs. Use the same gene identifier system and loading type
throughout, preferably complete tables without display-based row filtering.

```bash
python -m eaggl hierarchy \
  --leaf-file T2D runs/T2D/eaggl/gene_clusters.out.gz \
  --leaf-file CAD runs/CAD/eaggl/gene_clusters.out.gz \
  --output runs/T2D_CAD_hierarchy
```

Repeat `--leaf-file TRAIT PATH` for additional traits. The command aligns rows
by gene ID on the union of the input panels. A missing row supplies zero
observed loading in that trait; absence can reflect discovery-gene selection.
Different panels can therefore conceal a shared mechanism. The panel-overlap
table and the per-trait observation mask record which genes were measured.

The tree uses complete linkage on cosine similarity. At the default
`--min-leaf-cosine 0.50` overview cut, every pair of leaves in a family meets
the threshold. Parent loadings use the same OR rule as single-trait families.
These summarize the supplied profiles; repeated observations from related
traits can increase a parent's loading. Interpret it as a descriptive summary.

The separate redundancy screen lists **cross-trait pairs** with full-profile
cosine at least `--redundant-cosine 0.90`. Every such pair is flagged as a redundant loading profile, consistent with
the single-trait QC threshold. A separate comparable-strength flag requires the smaller
loading-vector norm to be at least `--min-norm-ratio 0.80` times the larger.
This additional heuristic distinguishes similar patterns at comparable strength
from much weaker copies. Every qualifying pair is reported, even across different
overview groups. Both source leaves stay in the tree.

| Output | Contents |
|---|---|
| `factor_hierarchy.tsv.gz`, `hierarchy_loadings.npz` | Tree edges, leaf IDs, merge scores, aligned node loadings, gene and node axes, observed-gene mask by trait |
| `leaf_sources.tsv.gz` | Integrated ID, source trait and factor, redundancy flag and partner count |
| `overview.tsv.gz` | One row per leaf giving its family at the requested cut |
| `redundancy_pairs.tsv.gz` | Similar pairs, cosine, norm ratio, fraction of each factor's squared loadings on shared genes, and profile/strength flags |
| `gene_panel_overlap.tsv.gz`, `hierarchy.json` | Trait-pair coverage and maximum cosine, source file hashes, settings and counts |

To display the integrated leaves in the offline PIGEAN dashboard, add
`--factor-atlas runs/T2D_CAD_hierarchy` to the dashboard build alongside the
matching per-trait `--eaggl-hierarchy` inputs. Install the optional
`umap-learn` dependency in the build environment. The browser needs no network.
Under **Factor graph**, switch from **MDS graph** to **Cross-trait atlas**.
One point represents one input leaf. Positions come from a seeded cosine UMAP
of the aligned gene profiles and stay fixed when coloring by gene, gene set or
phenotype. A search uses all embedded rows. Gene and gene-set colors show raw
loadings. If a matched phenotype reference was supplied, phenotype colors show
signed marginal Factor PheWAS coefficients adjusted for discovery-trait genetic
evidence. Hollow points mean the measurement is unavailable. Point details
include the integrated family, source factor, leading genes and overlay value.

Use **EAGGL anchor trait** to choose the discovery fit independently of the
PIGEAN evidence panel. **Match PIGEAN run** follows the top selector. An
explicit trait pins the EAGGL graphs, hierarchy and tables to that fit until
you select another anchor. In the research atlas, it also limits points and
rankings to that trait. **All anchor traits (combined atlas)** shows the
integrated leaves across fits. Single-fit tables and MDS are available by
selecting a trait. Switching atlas anchors preserves the searched overlay, map
coordinates and atlas-wide color scale.
