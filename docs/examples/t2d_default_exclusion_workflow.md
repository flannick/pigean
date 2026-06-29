# T2D Default Annotation-Exclusion Workflow

This example runs a complete T2D PIGEAN/EAGGL workflow:

1. Run PIGEAN from `dig-open-data` GWAS input.
2. Run EAGGL `gene_by_gene` with learn-phi, restart-consensus NMF, and aggregate per-phi outputs.
3. Use EAGGL annotation bridge diagnostics to write a suggested annotation-exclude list.
4. Re-run PIGEAN with the suggested exclusions.
5. Re-run EAGGL with learn-phi and restart-consensus NMF after exclusions.
6. For the selected EAGGL runs, compute phenotype NNLS projection and PIGEAN factor-trait enrichment.
7. Build factor graphs and a combined dashboard.

The commands assume they are run from the `pigean/` repository root.

## Input Files

The example uses files under `bundles/model_large-2026.02.22/data/`. The bundle includes the mouse/MSigDB libraries, the retained large libraries used below, reference gene files, and a large gene-PHEWAS file used for phenotype projection and factor-trait enrichment. The bundled gene-PHEWAS has traits containing `HP_` or `exomes_` removed.

```bash
ROOT="$PWD"
PY="../../.venv/bin/python"
DIG_OPEN_DATA_SRC="/Users/flannick/codex-workspace/analysis/resources/repos/dig-open-data/src"
export PYTHONPATH="$ROOT/src:$DIG_OPEN_DATA_SRC${PYTHONPATH:+:$PYTHONPATH}"
export MPLBACKEND=Agg
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-t2d-example}"
mkdir -p "$MPLCONFIGDIR"

BUNDLE="$ROOT/bundles/model_large-2026.02.22"
DATA="$BUNDLE/data"
BASE="$ROOT/results/example_t2d_default_exclusion"
mkdir -p "$BASE"

GWAS="dig-open-data:Mixed:T2D"
GENE_MAP="$DATA/portal_gencode.gene.map"
GENE_LOC="$DATA/NCBI37.3.plink.gene.loc"
GENE_LOC_EXONS="$DATA/NCBI37.3.plink.gene.exons.loc"
PHEWAS="$DATA/all.gene_stats.large.gt1.out.gz"

X_MOUSE="$DATA/gene_set_list_mouse_2024.txt"
X_MSIGDB="$DATA/gene_set_list_msigdb_nohp.txt"
X_OCR="$DATA/gene_set_list_ocr_human.txt"
X_STRING="$DATA/gene_set_list_string_notext_medium.txt"

PIGEAN0="$BASE/no_exclusions/pigean"
EAGGL0="$BASE/no_exclusions/eaggl_learn_phi"
PIGEAN1="$BASE/default_exclusions/pigean"
EAGGL1="$BASE/default_exclusions/eaggl_learn_phi"
DASH="$BASE/dashboard"

mkdir -p "$PIGEAN0" "$EAGGL0" "$PIGEAN1" "$EAGGL1" "$DASH"

PIGEAN0_GENE_STATS="$PIGEAN0/pigean.gene_stats.out.gz"
PIGEAN0_GENE_SET_STATS="$PIGEAN0/pigean.gene_set_stats.out.gz"
PIGEAN0_PARAMS="$PIGEAN0/pigean.params.out.gz"
PIGEAN0_RERUN_BUNDLE="$PIGEAN0/pigean.rerun_bundle.tar.gz"

PIGEAN1_GENE_STATS="$PIGEAN1/pigean.gene_stats.out.gz"
PIGEAN1_GENE_SET_STATS="$PIGEAN1/pigean.gene_set_stats.out.gz"
PIGEAN1_PARAMS="$PIGEAN1/pigean.params.out.gz"
PIGEAN1_RERUN_BUNDLE="$PIGEAN1/pigean.rerun_bundle.tar.gz"

EAGGL0_FACTORS="$EAGGL0/factors.out.gz"
EAGGL0_FACTOR_METRICS="$EAGGL0/factor_metrics.out.gz"
EAGGL0_GENE_CLUSTERS="$EAGGL0/gene_clusters.out.gz"
EAGGL0_GENE_SET_CLUSTERS="$EAGGL0/gene_set_clusters.out.gz"
EAGGL0_GENE_CLUSTERS_FULL="$EAGGL0/gene_clusters_full.out.gz"
EAGGL0_GENE_CLUSTERS_FULL_VIA_GENE_SETS="$EAGGL0/gene_clusters_full_via_gene_sets.out.gz"
EAGGL0_ANNOTATION_BRIDGE_METRICS="$EAGGL0/annotation_bridge_metrics.out.gz"
EAGGL0_SUGGESTED_EXCLUDE="$EAGGL0/annotation_bridge_suggested_exclude.txt"
EAGGL0_ANNOTATION_CONTRIBS="$EAGGL0/gene_factor_annotation_contribs.out.gz"
EAGGL0_LEARN_PHI_REPORT="$EAGGL0/learn_phi_report.out.gz"
EAGGL0_FACTOR_PHI_METRICS="$EAGGL0/factor_phi_metrics.out.gz"
EAGGL0_FACTOR_PHI_FACTORS="$EAGGL0/factor_phi_factors.out.gz"
EAGGL0_FACTOR_PHI_GENE_SET_CLUSTERS="$EAGGL0/factor_phi_gene_set_clusters.out.gz"
EAGGL0_FACTOR_PHI_GENE_CLUSTERS="$EAGGL0/factor_phi_gene_clusters.out.gz"
EAGGL0_PHI_SELECTION_WIDE="$EAGGL0/phi_selection_metrics_wide.out.gz"
EAGGL0_PHI_SELECTION_LONG="$EAGGL0/phi_selection_metrics_long.out.gz"
EAGGL0_PARAMS="$EAGGL0/params.out.gz"
EAGGL0_CONSENSUS_STATS="$EAGGL0/consensus_stats.out.gz"
EAGGL0_FACTOR_GMT="$EAGGL0/factors_as_gene_sets.gmt.gz"
EAGGL0_TRAIT_NNLS="$EAGGL0/trait_factor_links.nnls.out.gz"
EAGGL0_FACTOR_TRAIT_ENRICHMENTS="$EAGGL0/factor_trait_pigean_enrichments.out.gz"
EAGGL0_GRAPH_HTML="$EAGGL0/factor_graph.full_via_gene_sets.html"
EAGGL0_GRAPH_JSON="$EAGGL0/factor_graph.full_via_gene_sets.json"

EAGGL1_FACTORS="$EAGGL1/factors.out.gz"
EAGGL1_FACTOR_METRICS="$EAGGL1/factor_metrics.out.gz"
EAGGL1_GENE_CLUSTERS="$EAGGL1/gene_clusters.out.gz"
EAGGL1_GENE_SET_CLUSTERS="$EAGGL1/gene_set_clusters.out.gz"
EAGGL1_GENE_CLUSTERS_FULL="$EAGGL1/gene_clusters_full.out.gz"
EAGGL1_GENE_CLUSTERS_FULL_VIA_GENE_SETS="$EAGGL1/gene_clusters_full_via_gene_sets.out.gz"
EAGGL1_ANNOTATION_BRIDGE_METRICS="$EAGGL1/annotation_bridge_metrics.out.gz"
EAGGL1_SUGGESTED_EXCLUDE="$EAGGL1/annotation_bridge_suggested_exclude.txt"
EAGGL1_ANNOTATION_CONTRIBS="$EAGGL1/gene_factor_annotation_contribs.out.gz"
EAGGL1_LEARN_PHI_REPORT="$EAGGL1/learn_phi_report.out.gz"
EAGGL1_FACTOR_PHI_METRICS="$EAGGL1/factor_phi_metrics.out.gz"
EAGGL1_FACTOR_PHI_FACTORS="$EAGGL1/factor_phi_factors.out.gz"
EAGGL1_FACTOR_PHI_GENE_SET_CLUSTERS="$EAGGL1/factor_phi_gene_set_clusters.out.gz"
EAGGL1_FACTOR_PHI_GENE_CLUSTERS="$EAGGL1/factor_phi_gene_clusters.out.gz"
EAGGL1_PHI_SELECTION_WIDE="$EAGGL1/phi_selection_metrics_wide.out.gz"
EAGGL1_PHI_SELECTION_LONG="$EAGGL1/phi_selection_metrics_long.out.gz"
EAGGL1_PARAMS="$EAGGL1/params.out.gz"
EAGGL1_CONSENSUS_STATS="$EAGGL1/consensus_stats.out.gz"
EAGGL1_FACTOR_GMT="$EAGGL1/factors_as_gene_sets.gmt.gz"
EAGGL1_TRAIT_NNLS="$EAGGL1/trait_factor_links.nnls.out.gz"
EAGGL1_FACTOR_TRAIT_ENRICHMENTS="$EAGGL1/factor_trait_pigean_enrichments.out.gz"
EAGGL1_GRAPH_HTML="$EAGGL1/factor_graph.full_via_gene_sets.html"
EAGGL1_GRAPH_JSON="$EAGGL1/factor_graph.full_via_gene_sets.json"

DASH_HTML="$DASH/t2d_default_exclusion_dashboard.html"
DASH_JSON="$DASH/t2d_default_exclusion_dashboard.json"
```

## 1. Run PIGEAN Without Annotation Exclusions

```bash
"$PY" -m pigean gibbs \
  --hide-progress \
  --gwas-in "$GWAS" \
  --gene-map-in "$GENE_MAP" \
  --gene-loc-file "$GENE_LOC" \
  --gene-loc-file-huge "$GENE_LOC_EXONS" \
  --exons-loc-file-huge "$GENE_LOC_EXONS" \
  --X-in "$X_MOUSE" \
  --X-in "$X_MSIGDB" \
  --X-in "$X_OCR" \
  --X-in "$X_STRING" \
  --gene-stats-out "$PIGEAN0_GENE_STATS" \
  --gene-set-stats-out "$PIGEAN0_GENE_SET_STATS" \
  --params-out "$PIGEAN0_PARAMS" \
  --pigean-rerun-bundle-out "$PIGEAN0_RERUN_BUNDLE" \
  --log-file "$PIGEAN0/pigean.run.log.gz" \
  --warnings-file "$PIGEAN0/pigean.warnings.log.gz" \
  > "$PIGEAN0/stdout.txt" \
  2> "$PIGEAN0/stderr.txt"
```

## 2. Run EAGGL Learn-Phi With Consensus NMF Without Annotation Exclusions

```bash
"$PY" -m eaggl factor \
  --deterministic \
  --seed 0 \
  --hide-progress \
  --max-gb 2 \
  --discovery-model gene_by_gene \
  --phi-selection-objective composite \
  --learn-phi \
  --factor-runs 10 \
  --consensus-nmf \
  --consensus-stats-out "$EAGGL0_CONSENSUS_STATS" \
  --max-num-factors 200 \
  --factor-output-scope all \
  --cluster-row-min-max-loading 0 \
  --X-in "$X_MOUSE" \
  --X-in "$X_MSIGDB" \
  --X-in "$X_OCR" \
  --X-in "$X_STRING" \
  --gene-sets-for-labeling "$X_MOUSE" \
  --gene-sets-for-labeling "$X_MSIGDB" \
  --gene-stats-in "$PIGEAN0_GENE_STATS" \
  --gene-stats-id-col Gene \
  --gene-stats-combined-col combined \
  --gene-stats-log-bf-col log_bf \
  --gene-stats-prior-col prior \
  --gene-set-stats-in "$PIGEAN0_GENE_SET_STATS" \
  --gene-set-stats-id-col Gene_Set \
  --gene-set-stats-beta-col beta \
  --gene-set-stats-beta-uncorrected-col beta_uncorrected \
  --factors-out "$EAGGL0_FACTORS" \
  --factor-metrics-out "$EAGGL0_FACTOR_METRICS" \
  --gene-clusters-out "$EAGGL0_GENE_CLUSTERS" \
  --gene-set-clusters-out "$EAGGL0_GENE_SET_CLUSTERS" \
  --gene-clusters-full-out "$EAGGL0_GENE_CLUSTERS_FULL" \
  --gene-clusters-full-via-gene-sets-out "$EAGGL0_GENE_CLUSTERS_FULL_VIA_GENE_SETS" \
  --annotation-bridge-metrics-out "$EAGGL0_ANNOTATION_BRIDGE_METRICS" \
  --annotation-bridge-suggested-exclude-out "$EAGGL0_SUGGESTED_EXCLUDE" \
  --gene-factor-annotation-contribs-out "$EAGGL0_ANNOTATION_CONTRIBS" \
  --gene-factor-annotation-contribs-top-n 10 \
  --learn-phi-report-out "$EAGGL0_LEARN_PHI_REPORT" \
  --factor-phi-metrics-out "$EAGGL0_FACTOR_PHI_METRICS" \
  --factor-phi-factors-out "$EAGGL0_FACTOR_PHI_FACTORS" \
  --factor-phi-gene-set-clusters-out "$EAGGL0_FACTOR_PHI_GENE_SET_CLUSTERS" \
  --factor-phi-gene-clusters-out "$EAGGL0_FACTOR_PHI_GENE_CLUSTERS" \
  --phi-selection-metrics-wide-out "$EAGGL0_PHI_SELECTION_WIDE" \
  --phi-selection-metrics-long-out "$EAGGL0_PHI_SELECTION_LONG" \
  --params-out "$EAGGL0_PARAMS" \
  --log-file "$EAGGL0/eaggl.run.log.gz" \
  --warnings-file "$EAGGL0/eaggl.warnings.log.gz" \
  > "$EAGGL0/stdout.txt" \
  2> "$EAGGL0/stderr.txt"
```

## 3. Re-Run PIGEAN With EAGGL Suggested Exclusions

```bash
EXCLUDE_DEFAULT="$EAGGL0_SUGGESTED_EXCLUDE"

"$PY" -m pigean gibbs \
  --hide-progress \
  --gwas-in "$GWAS" \
  --gene-map-in "$GENE_MAP" \
  --gene-loc-file "$GENE_LOC" \
  --gene-loc-file-huge "$GENE_LOC_EXONS" \
  --exons-loc-file-huge "$GENE_LOC_EXONS" \
  --X-in "$X_MOUSE" \
  --X-in "$X_MSIGDB" \
  --X-in "$X_OCR" \
  --X-in "$X_STRING" \
  --gene-set-exclude-in "$EXCLUDE_DEFAULT" \
  --gene-stats-out "$PIGEAN1_GENE_STATS" \
  --gene-set-stats-out "$PIGEAN1_GENE_SET_STATS" \
  --params-out "$PIGEAN1_PARAMS" \
  --pigean-rerun-bundle-out "$PIGEAN1_RERUN_BUNDLE" \
  --log-file "$PIGEAN1/pigean.run.log.gz" \
  --warnings-file "$PIGEAN1/pigean.warnings.log.gz" \
  > "$PIGEAN1/stdout.txt" \
  2> "$PIGEAN1/stderr.txt"
```

## 4. Re-Run EAGGL Learn-Phi With Consensus NMF After Exclusions

```bash
"$PY" -m eaggl factor \
  --deterministic \
  --seed 0 \
  --hide-progress \
  --max-gb 2 \
  --discovery-model gene_by_gene \
  --phi-selection-objective composite \
  --learn-phi \
  --factor-runs 10 \
  --consensus-nmf \
  --consensus-stats-out "$EAGGL1_CONSENSUS_STATS" \
  --max-num-factors 200 \
  --factor-output-scope all \
  --cluster-row-min-max-loading 0 \
  --X-in "$X_MOUSE" \
  --X-in "$X_MSIGDB" \
  --X-in "$X_OCR" \
  --X-in "$X_STRING" \
  --gene-sets-for-labeling "$X_MOUSE" \
  --gene-sets-for-labeling "$X_MSIGDB" \
  --gene-stats-in "$PIGEAN1_GENE_STATS" \
  --gene-stats-id-col Gene \
  --gene-stats-combined-col combined \
  --gene-stats-log-bf-col log_bf \
  --gene-stats-prior-col prior \
  --gene-set-stats-in "$PIGEAN1_GENE_SET_STATS" \
  --gene-set-stats-id-col Gene_Set \
  --gene-set-stats-beta-col beta \
  --gene-set-stats-beta-uncorrected-col beta_uncorrected \
  --factors-out "$EAGGL1_FACTORS" \
  --factor-metrics-out "$EAGGL1_FACTOR_METRICS" \
  --gene-clusters-out "$EAGGL1_GENE_CLUSTERS" \
  --gene-set-clusters-out "$EAGGL1_GENE_SET_CLUSTERS" \
  --gene-clusters-full-out "$EAGGL1_GENE_CLUSTERS_FULL" \
  --gene-clusters-full-via-gene-sets-out "$EAGGL1_GENE_CLUSTERS_FULL_VIA_GENE_SETS" \
  --annotation-bridge-metrics-out "$EAGGL1_ANNOTATION_BRIDGE_METRICS" \
  --annotation-bridge-suggested-exclude-out "$EAGGL1_SUGGESTED_EXCLUDE" \
  --gene-factor-annotation-contribs-out "$EAGGL1_ANNOTATION_CONTRIBS" \
  --gene-factor-annotation-contribs-top-n 10 \
  --learn-phi-report-out "$EAGGL1_LEARN_PHI_REPORT" \
  --factor-phi-metrics-out "$EAGGL1_FACTOR_PHI_METRICS" \
  --factor-phi-factors-out "$EAGGL1_FACTOR_PHI_FACTORS" \
  --factor-phi-gene-set-clusters-out "$EAGGL1_FACTOR_PHI_GENE_SET_CLUSTERS" \
  --factor-phi-gene-clusters-out "$EAGGL1_FACTOR_PHI_GENE_CLUSTERS" \
  --phi-selection-metrics-wide-out "$EAGGL1_PHI_SELECTION_WIDE" \
  --phi-selection-metrics-long-out "$EAGGL1_PHI_SELECTION_LONG" \
  --params-out "$EAGGL1_PARAMS" \
  --log-file "$EAGGL1/eaggl.run.log.gz" \
  --warnings-file "$EAGGL1/eaggl.warnings.log.gz" \
  > "$EAGGL1/stdout.txt" \
  2> "$EAGGL1/stderr.txt"
```

## 5. Extract Gene Universes

The factor-trait enrichment step uses the exact gene universe from each PIGEAN run. The bundled PHEWAS input already excludes traits containing `HP_` or `exomes_`, so no runtime trait blacklist is needed.

```bash
mkdir -p "$PIGEAN0/rerun_bundle_extract" "$PIGEAN1/rerun_bundle_extract"
tar -xzf "$PIGEAN0_RERUN_BUNDLE" -C "$PIGEAN0/rerun_bundle_extract" gene_universe.tsv.gz
tar -xzf "$PIGEAN1_RERUN_BUNDLE" -C "$PIGEAN1/rerun_bundle_extract" gene_universe.tsv.gz

GENE_UNIVERSE0="$PIGEAN0/rerun_bundle_extract/gene_universe.tsv.gz"
GENE_UNIVERSE1="$PIGEAN1/rerun_bundle_extract/gene_universe.tsv.gz"
```

## 6. Phenotype NNLS Projection For The Selected EAGGL Runs

These commands compute fixed-W phenotype projection and export factors as GMT files for PIGEAN factor-trait enrichment.

```bash
"$PY" -m eaggl factor \
  --deterministic \
  --seed 0 \
  --hide-progress \
  --max-gb 2 \
  --factor-gene-clusters-in "$EAGGL0_GENE_CLUSTERS" \
  --gene-phewas-stats-in "$PHEWAS" \
  --gene-phewas-stats-id-col Gene \
  --gene-phewas-stats-combined-col Combined \
  --gene-phewas-stats-log-bf-col Direct \
  --gene-phewas-stats-prior-col Indirect \
  --gene-phewas-stats-pheno-col Trait_Internal \
  --trait-linkage-source combined \
  --trait-factor-linkage-factor-gene-threshold 0.05 \
  --trait-factor-linkage-nnls-min-loading 0.5 \
  --trait-factor-linkage-nnls-max-value 1.0 \
  --factor-gmt-out "$EAGGL0_FACTOR_GMT" \
  --trait-factor-links-output-detail full \
  --trait-factor-links-out "$EAGGL0_TRAIT_NNLS" \
  --params-out "$EAGGL0/trait_projection.params.out.gz" \
  --log-file "$EAGGL0/trait_projection.run.log.gz" \
  --warnings-file "$EAGGL0/trait_projection.warnings.log.gz" \
  > "$EAGGL0/trait_projection.stdout.txt" \
  2> "$EAGGL0/trait_projection.stderr.txt"

"$PY" -m eaggl factor \
  --deterministic \
  --seed 0 \
  --hide-progress \
  --max-gb 2 \
  --factor-gene-clusters-in "$EAGGL1_GENE_CLUSTERS" \
  --gene-phewas-stats-in "$PHEWAS" \
  --gene-phewas-stats-id-col Gene \
  --gene-phewas-stats-combined-col Combined \
  --gene-phewas-stats-log-bf-col Direct \
  --gene-phewas-stats-prior-col Indirect \
  --gene-phewas-stats-pheno-col Trait_Internal \
  --trait-linkage-source combined \
  --trait-factor-linkage-factor-gene-threshold 0.05 \
  --trait-factor-linkage-nnls-min-loading 0.5 \
  --trait-factor-linkage-nnls-max-value 1.0 \
  --factor-gmt-out "$EAGGL1_FACTOR_GMT" \
  --trait-factor-links-output-detail full \
  --trait-factor-links-out "$EAGGL1_TRAIT_NNLS" \
  --params-out "$EAGGL1/trait_projection.params.out.gz" \
  --log-file "$EAGGL1/trait_projection.run.log.gz" \
  --warnings-file "$EAGGL1/trait_projection.warnings.log.gz" \
  > "$EAGGL1/trait_projection.stdout.txt" \
  2> "$EAGGL1/trait_projection.stderr.txt"
```

## 7. PIGEAN Factor-Trait Enrichment

This treats EAGGL factors as gene sets and computes PIGEAN `beta`, `beta_uncorrected`, `beta_tilde`, standard error, and p-value for each external trait/factor pair.

```bash
"$PY" -m pigean betas \
  --X-in "$EAGGL0_FACTOR_GMT" \
  --gene-universe-in "$GENE_UNIVERSE0" \
  --gene-universe-id-col Gene \
  --multi-y-in "$PHEWAS" \
  --multi-y-id-col Gene \
  --multi-y-pheno-col Trait_Internal \
  --multi-y-combined-col Combined \
  --multi-y-log-bf-col Direct \
  --multi-y-prior-col Indirect \
  --multi-y-response-col combined \
  --update-hyper none \
  --multi-y-max-phenos-per-batch 200 \
  --multi-y-vectorize-betas \
  --no-filter-negative \
  --prune-gene-sets 2 \
  --weighted-prune-gene-sets 2 \
  --min-gene-set-size 1 \
  --filter-gene-set-p 1 \
  --output-detail full \
  --gene-set-stats-out "$EAGGL0_FACTOR_TRAIT_ENRICHMENTS" \
  --params-out "$EAGGL0/factor_trait_pigean_enrichments.params.out.gz" \
  --log-file "$EAGGL0/factor_trait_pigean_enrichments.run.log.gz" \
  --warnings-file "$EAGGL0/factor_trait_pigean_enrichments.warnings.log.gz" \
  > "$EAGGL0/factor_trait_pigean_enrichments.stdout.txt" \
  2> "$EAGGL0/factor_trait_pigean_enrichments.stderr.txt"

"$PY" -m pigean betas \
  --X-in "$EAGGL1_FACTOR_GMT" \
  --gene-universe-in "$GENE_UNIVERSE1" \
  --gene-universe-id-col Gene \
  --multi-y-in "$PHEWAS" \
  --multi-y-id-col Gene \
  --multi-y-pheno-col Trait_Internal \
  --multi-y-combined-col Combined \
  --multi-y-log-bf-col Direct \
  --multi-y-prior-col Indirect \
  --multi-y-response-col combined \
  --update-hyper none \
  --multi-y-max-phenos-per-batch 200 \
  --multi-y-vectorize-betas \
  --no-filter-negative \
  --prune-gene-sets 2 \
  --weighted-prune-gene-sets 2 \
  --min-gene-set-size 1 \
  --filter-gene-set-p 1 \
  --output-detail full \
  --gene-set-stats-out "$EAGGL1_FACTOR_TRAIT_ENRICHMENTS" \
  --params-out "$EAGGL1/factor_trait_pigean_enrichments.params.out.gz" \
  --log-file "$EAGGL1/factor_trait_pigean_enrichments.run.log.gz" \
  --warnings-file "$EAGGL1/factor_trait_pigean_enrichments.warnings.log.gz" \
  > "$EAGGL1/factor_trait_pigean_enrichments.stdout.txt" \
  2> "$EAGGL1/factor_trait_pigean_enrichments.stderr.txt"
```

## 8. Build Factor Graphs

```bash
"$PY" -m eaggl.factor_graph \
  --eaggl-dir "$EAGGL0" \
  --gene-clusters-in "$EAGGL0_GENE_CLUSTERS_FULL_VIA_GENE_SETS" \
  --gene-set-clusters-in "$EAGGL0_GENE_SET_CLUSTERS" \
  --trait-factor-links-in "$EAGGL0_TRAIT_NNLS" \
  --factor-trait-enrichments-in "$EAGGL0_FACTOR_TRAIT_ENRICHMENTS" \
  --gene-phewas-stats-in "$PHEWAS" \
  --gene-phewas-stats-id-col Gene \
  --gene-phewas-stats-pheno-col Trait_Internal \
  --gene-phewas-stats-combined-col Combined \
  --gene-phewas-stats-log-bf-col Direct \
  --gene-phewas-stats-prior-col Indirect \
  --trait-min-neff 25 \
  --html-out "$EAGGL0_GRAPH_HTML" \
  --json-out "$EAGGL0_GRAPH_JSON" \
  > "$EAGGL0/factor_graph.stdout.txt" \
  2> "$EAGGL0/factor_graph.stderr.txt"

"$PY" -m eaggl.factor_graph \
  --eaggl-dir "$EAGGL1" \
  --gene-clusters-in "$EAGGL1_GENE_CLUSTERS_FULL_VIA_GENE_SETS" \
  --gene-set-clusters-in "$EAGGL1_GENE_SET_CLUSTERS" \
  --trait-factor-links-in "$EAGGL1_TRAIT_NNLS" \
  --factor-trait-enrichments-in "$EAGGL1_FACTOR_TRAIT_ENRICHMENTS" \
  --gene-phewas-stats-in "$PHEWAS" \
  --gene-phewas-stats-id-col Gene \
  --gene-phewas-stats-pheno-col Trait_Internal \
  --gene-phewas-stats-combined-col Combined \
  --gene-phewas-stats-log-bf-col Direct \
  --gene-phewas-stats-prior-col Indirect \
  --trait-min-neff 25 \
  --html-out "$EAGGL1_GRAPH_HTML" \
  --json-out "$EAGGL1_GRAPH_JSON" \
  > "$EAGGL1/factor_graph.stdout.txt" \
  2> "$EAGGL1/factor_graph.stderr.txt"
```

## 9. Build The Dashboard

The dashboard reads the aggregate learn-phi outputs directly through `--eaggl-phi-sweep`, so it does not require separate fixed-phi EAGGL directories for every candidate phi. The selected-phi `--eaggl-run` entries add the factor graph and phenotype projection/enrichment outputs.

```bash
"$PY" -m pigean.dashboard \
  --title "T2D default annotation-exclusion learn-phi gene-by-gene comparison" \
  --pigean-run no_exclusions:"$PIGEAN0" \
  --pigean-run default_exclusions:"$PIGEAN1" \
  --run-title no_exclusions:"T2D full Gibbs, no annotation exclusions" \
  --run-title default_exclusions:"T2D full Gibbs, EAGGL suggested annotation exclusions" \
  --eaggl-phi-sweep no_exclusions:gene_by_gene_learn_phi:"$EAGGL0" \
  --eaggl-phi-sweep default_exclusions:gene_by_gene_learn_phi:"$EAGGL1" \
  --eaggl-run no_exclusions:gene_by_gene_selected:"$EAGGL0" \
  --eaggl-run default_exclusions:gene_by_gene_selected:"$EAGGL1" \
  --eaggl-group no_exclusions:gene_by_gene_selected:gene_by_gene_learn_phi:"gene by gene learn phi" \
  --eaggl-group default_exclusions:gene_by_gene_selected:gene_by_gene_learn_phi:"gene by gene learn phi" \
  --x-input "$X_MOUSE" \
  --x-input "$X_MSIGDB" \
  --x-input "$X_OCR" \
  --x-input "$X_STRING" \
  --default-gene-loading-source full_via_gene_sets \
  --html-out "$DASH_HTML" \
  --json-out "$DASH_JSON" \
  > "$DASH/stdout.txt" \
  2> "$DASH/stderr.txt"
```

Final output:

```bash
echo "$DASH_HTML"
```
