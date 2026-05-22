# T2D/MODY Multi-Trait EAGGL Workflow

This example runs a two-anchor T2D/MODY PIGEAN/EAGGL workflow:

1. Run T2D PIGEAN from `dig-open-data` GWAS input.
2. Run MODY PIGEAN from a MODY gene list.
3. Run EAGGL `gene_by_gene` with both T2D and MODY as labeled anchors, using learn-phi and composite phi selection.
4. Use EAGGL annotation bridge diagnostics to write one suggested annotation-exclude list for the two-anchor run.
5. Re-run T2D and MODY PIGEAN with that exclude list.
6. Re-run two-anchor EAGGL after exclusions.
7. For both selected EAGGL runs, compute phenotype NNLS projection, export factors as a GMT, run PIGEAN multi-Y factor-trait enrichment, build factor graphs, and build one dashboard.

The commands assume they are run from the `pigean/` repository root. They use only files in `bundles/model_large-2026.02.22/data/` plus the small MODY gene-list fixture in `tests/data/t2d_smoke/mody.gene.list`.

## Input Files

```bash
ROOT="$PWD"
PY="../../.venv/bin/python"
DIG_OPEN_DATA_SRC="/Users/flannick/codex-workspace/analysis/resources/repos/dig-open-data/src"
export PYTHONPATH="$ROOT/src:$DIG_OPEN_DATA_SRC${PYTHONPATH:+:$PYTHONPATH}"
export MPLBACKEND=Agg
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-t2d-mody-example}"
mkdir -p "$MPLCONFIGDIR"

BUNDLE="$ROOT/bundles/model_large-2026.02.22"
DATA="$BUNDLE/data"
BASE="$ROOT/results/example_t2d_mody_multitrait"
mkdir -p "$BASE"

GWAS_T2D="dig-open-data:Mixed:T2D"
MODY_LIST="$ROOT/tests/data/t2d_smoke/mody.gene.list"
GENE_MAP="$DATA/portal_gencode.gene.map"
GENE_LOC="$DATA/NCBI37.3.plink.gene.loc"
GENE_LOC_EXONS="$DATA/NCBI37.3.plink.gene.exons.loc"
PHEWAS="$DATA/all.gene_stats.large.gt1.out.gz"

X_MOUSE="$DATA/gene_set_list_mouse_2024.txt"
X_MSIGDB="$DATA/gene_set_list_msigdb_nohp.txt"
X_OCR="$DATA/gene_set_list_ocr_human.txt"
X_STRING="$DATA/gene_set_list_string_notext_medium.txt"

T2D_PRE="$BASE/t2d_pre_exclusion/pigean"
MODY_PRE="$BASE/mody_pre_exclusion/pigean"
EAGGL_PRE="$BASE/t2d_mody_pre_exclusion/eaggl_learn_phi"
T2D_POST="$BASE/t2d_post_exclusion/pigean"
MODY_POST="$BASE/mody_post_exclusion/pigean"
EAGGL_POST="$BASE/t2d_mody_post_exclusion/eaggl_learn_phi"
DASH="$BASE/dashboard"
mkdir -p "$T2D_PRE" "$MODY_PRE" "$EAGGL_PRE" "$T2D_POST" "$MODY_POST" "$EAGGL_POST" "$DASH"
```

## 1. Run T2D PIGEAN Before Exclusions

```bash
"$PY" -m pigean gibbs \
  --hide-progress \
  --gwas-in "$GWAS_T2D" \
  --gene-map-in "$GENE_MAP" \
  --gene-loc-file "$GENE_LOC" \
  --gene-loc-file-huge "$GENE_LOC_EXONS" \
  --exons-loc-file-huge "$GENE_LOC_EXONS" \
  --X-in "$X_MOUSE" \
  --X-in "$X_MSIGDB" \
  --X-in "$X_OCR" \
  --X-in "$X_STRING" \
  --gene-stats-out "$T2D_PRE/pigean.gene_stats.out.gz" \
  --gene-set-stats-out "$T2D_PRE/pigean.gene_set_stats.out.gz" \
  --params-out "$T2D_PRE/pigean.params.out.gz" \
  --pigean-rerun-bundle-out "$T2D_PRE/pigean.rerun_bundle.tar.gz" \
  --log-file "$T2D_PRE/pigean.run.log.gz" \
  --warnings-file "$T2D_PRE/pigean.warnings.log.gz" \
  > "$T2D_PRE/stdout.txt" \
  2> "$T2D_PRE/stderr.txt"
```

## 2. Run MODY PIGEAN Before Exclusions

```bash
"$PY" -m pigean gibbs \
  --hide-progress \
  --gene-list-in "$MODY_LIST" \
  --gene-list-no-header \
  --gene-list-all-in "$GENE_LOC" \
  --gene-list-all-id-col 6 \
  --gene-list-all-no-header \
  --gene-map-in "$GENE_MAP" \
  --gene-loc-file "$GENE_LOC" \
  --X-in "$X_MOUSE" \
  --X-in "$X_MSIGDB" \
  --X-in "$X_OCR" \
  --X-in "$X_STRING" \
  --gene-stats-out "$MODY_PRE/pigean.gene_stats.out.gz" \
  --gene-set-stats-out "$MODY_PRE/pigean.gene_set_stats.out.gz" \
  --params-out "$MODY_PRE/pigean.params.out.gz" \
  --pigean-rerun-bundle-out "$MODY_PRE/pigean.rerun_bundle.tar.gz" \
  --log-file "$MODY_PRE/pigean.run.log.gz" \
  --warnings-file "$MODY_PRE/pigean.warnings.log.gz" \
  > "$MODY_PRE/stdout.txt" \
  2> "$MODY_PRE/stderr.txt"
```

## 3. Run Two-Anchor EAGGL Before Exclusions

```bash
"$PY" -m eaggl factor \
  --deterministic \
  --seed 0 \
  --hide-progress \
  --max-gb 2 \
  --discovery-model gene_by_gene \
  --anchor-aggregation multi \
  --phi-selection-objective composite \
  --learn-phi \
  --max-num-factors 200 \
  --factor-output-scope all \
  --cluster-row-min-max-loading 0 \
  --X-in "$X_MOUSE" \
  --X-in "$X_MSIGDB" \
  --X-in "$X_OCR" \
  --X-in "$X_STRING" \
  --gene-sets-for-labeling "$X_MOUSE" \
  --gene-sets-for-labeling "$X_MSIGDB" \
  --gene-stats-in "T2D=$T2D_PRE/pigean.gene_stats.out.gz" \
  --gene-stats-in "MODY=$MODY_PRE/pigean.gene_stats.out.gz" \
  --gene-stats-id-col Gene \
  --gene-stats-combined-col combined \
  --gene-stats-log-bf-col log_bf \
  --gene-stats-prior-col prior \
  --gene-set-stats-in "T2D=$T2D_PRE/pigean.gene_set_stats.out.gz" \
  --gene-set-stats-in "MODY=$MODY_PRE/pigean.gene_set_stats.out.gz" \
  --gene-set-stats-id-col Gene_Set \
  --gene-set-stats-beta-col beta \
  --gene-set-stats-beta-uncorrected-col beta_uncorrected \
  --factors-out "$EAGGL_PRE/factors.out.gz" \
  --factor-metrics-out "$EAGGL_PRE/factor_metrics.out.gz" \
  --gene-clusters-out "$EAGGL_PRE/gene_clusters.out.gz" \
  --gene-set-clusters-out "$EAGGL_PRE/gene_set_clusters.out.gz" \
  --gene-clusters-full-out "$EAGGL_PRE/gene_clusters_full.out.gz" \
  --annotation-bridge-metrics-out "$EAGGL_PRE/annotation_bridge_metrics.out.gz" \
  --annotation-bridge-suggested-exclude-out "$EAGGL_PRE/annotation_bridge_suggested_exclude.txt" \
  --gene-factor-annotation-contribs-out "$EAGGL_PRE/gene_factor_annotation_contribs.out.gz" \
  --gene-factor-annotation-contribs-top-n 10 \
  --learn-phi-report-out "$EAGGL_PRE/learn_phi_report.out.gz" \
  --factor-phi-metrics-out "$EAGGL_PRE/factor_phi_metrics.out.gz" \
  --factor-phi-factors-out "$EAGGL_PRE/factor_phi_factors.out.gz" \
  --factor-phi-gene-set-clusters-out "$EAGGL_PRE/factor_phi_gene_set_clusters.out.gz" \
  --factor-phi-gene-clusters-out "$EAGGL_PRE/factor_phi_gene_clusters.out.gz" \
  --phi-selection-metrics-wide-out "$EAGGL_PRE/phi_selection_metrics_wide.out.gz" \
  --phi-selection-metrics-long-out "$EAGGL_PRE/phi_selection_metrics_long.out.gz" \
  --params-out "$EAGGL_PRE/params.out.gz" \
  --log-file "$EAGGL_PRE/eaggl.run.log.gz" \
  --warnings-file "$EAGGL_PRE/eaggl.warnings.log.gz" \
  > "$EAGGL_PRE/stdout.txt" \
  2> "$EAGGL_PRE/stderr.txt"
```

## 4. Re-Run T2D And MODY PIGEAN With The Two-Anchor Exclude List

```bash
EXCLUDE_T2D_MODY="$EAGGL_PRE/annotation_bridge_suggested_exclude.txt"

"$PY" -m pigean gibbs \
  --hide-progress \
  --gwas-in "$GWAS_T2D" \
  --gene-map-in "$GENE_MAP" \
  --gene-loc-file "$GENE_LOC" \
  --gene-loc-file-huge "$GENE_LOC_EXONS" \
  --exons-loc-file-huge "$GENE_LOC_EXONS" \
  --X-in "$X_MOUSE" \
  --X-in "$X_MSIGDB" \
  --X-in "$X_OCR" \
  --X-in "$X_STRING" \
  --gene-set-exclude-in "$EXCLUDE_T2D_MODY" \
  --gene-stats-out "$T2D_POST/pigean.gene_stats.out.gz" \
  --gene-set-stats-out "$T2D_POST/pigean.gene_set_stats.out.gz" \
  --params-out "$T2D_POST/pigean.params.out.gz" \
  --pigean-rerun-bundle-out "$T2D_POST/pigean.rerun_bundle.tar.gz" \
  --log-file "$T2D_POST/pigean.run.log.gz" \
  --warnings-file "$T2D_POST/pigean.warnings.log.gz" \
  > "$T2D_POST/stdout.txt" \
  2> "$T2D_POST/stderr.txt"

"$PY" -m pigean gibbs \
  --hide-progress \
  --gene-list-in "$MODY_LIST" \
  --gene-list-no-header \
  --gene-list-all-in "$GENE_LOC" \
  --gene-list-all-id-col 6 \
  --gene-list-all-no-header \
  --gene-map-in "$GENE_MAP" \
  --gene-loc-file "$GENE_LOC" \
  --X-in "$X_MOUSE" \
  --X-in "$X_MSIGDB" \
  --X-in "$X_OCR" \
  --X-in "$X_STRING" \
  --gene-set-exclude-in "$EXCLUDE_T2D_MODY" \
  --gene-stats-out "$MODY_POST/pigean.gene_stats.out.gz" \
  --gene-set-stats-out "$MODY_POST/pigean.gene_set_stats.out.gz" \
  --params-out "$MODY_POST/pigean.params.out.gz" \
  --pigean-rerun-bundle-out "$MODY_POST/pigean.rerun_bundle.tar.gz" \
  --log-file "$MODY_POST/pigean.run.log.gz" \
  --warnings-file "$MODY_POST/pigean.warnings.log.gz" \
  > "$MODY_POST/stdout.txt" \
  2> "$MODY_POST/stderr.txt"
```

## 5. Re-Run Two-Anchor EAGGL After Exclusions

```bash
"$PY" -m eaggl factor \
  --deterministic \
  --seed 0 \
  --hide-progress \
  --max-gb 2 \
  --discovery-model gene_by_gene \
  --anchor-aggregation multi \
  --phi-selection-objective composite \
  --learn-phi \
  --max-num-factors 200 \
  --factor-output-scope all \
  --cluster-row-min-max-loading 0 \
  --X-in "$X_MOUSE" \
  --X-in "$X_MSIGDB" \
  --X-in "$X_OCR" \
  --X-in "$X_STRING" \
  --gene-sets-for-labeling "$X_MOUSE" \
  --gene-sets-for-labeling "$X_MSIGDB" \
  --gene-stats-in "T2D=$T2D_POST/pigean.gene_stats.out.gz" \
  --gene-stats-in "MODY=$MODY_POST/pigean.gene_stats.out.gz" \
  --gene-stats-id-col Gene \
  --gene-stats-combined-col combined \
  --gene-stats-log-bf-col log_bf \
  --gene-stats-prior-col prior \
  --gene-set-stats-in "T2D=$T2D_POST/pigean.gene_set_stats.out.gz" \
  --gene-set-stats-in "MODY=$MODY_POST/pigean.gene_set_stats.out.gz" \
  --gene-set-stats-id-col Gene_Set \
  --gene-set-stats-beta-col beta \
  --gene-set-stats-beta-uncorrected-col beta_uncorrected \
  --factors-out "$EAGGL_POST/factors.out.gz" \
  --factor-metrics-out "$EAGGL_POST/factor_metrics.out.gz" \
  --gene-clusters-out "$EAGGL_POST/gene_clusters.out.gz" \
  --gene-set-clusters-out "$EAGGL_POST/gene_set_clusters.out.gz" \
  --gene-clusters-full-out "$EAGGL_POST/gene_clusters_full.out.gz" \
  --annotation-bridge-metrics-out "$EAGGL_POST/annotation_bridge_metrics.out.gz" \
  --annotation-bridge-suggested-exclude-out "$EAGGL_POST/annotation_bridge_suggested_exclude.txt" \
  --gene-factor-annotation-contribs-out "$EAGGL_POST/gene_factor_annotation_contribs.out.gz" \
  --gene-factor-annotation-contribs-top-n 10 \
  --learn-phi-report-out "$EAGGL_POST/learn_phi_report.out.gz" \
  --factor-phi-metrics-out "$EAGGL_POST/factor_phi_metrics.out.gz" \
  --factor-phi-factors-out "$EAGGL_POST/factor_phi_factors.out.gz" \
  --factor-phi-gene-set-clusters-out "$EAGGL_POST/factor_phi_gene_set_clusters.out.gz" \
  --factor-phi-gene-clusters-out "$EAGGL_POST/factor_phi_gene_clusters.out.gz" \
  --phi-selection-metrics-wide-out "$EAGGL_POST/phi_selection_metrics_wide.out.gz" \
  --phi-selection-metrics-long-out "$EAGGL_POST/phi_selection_metrics_long.out.gz" \
  --params-out "$EAGGL_POST/params.out.gz" \
  --log-file "$EAGGL_POST/eaggl.run.log.gz" \
  --warnings-file "$EAGGL_POST/eaggl.warnings.log.gz" \
  > "$EAGGL_POST/stdout.txt" \
  2> "$EAGGL_POST/stderr.txt"
```

## 6. Compute Phenotype Projection And PIGEAN Factor-Trait Enrichment

Run this block once for the pre-exclusion EAGGL run and once for the post-exclusion EAGGL run.

```bash
for EAGGL_DIR in "$EAGGL_PRE" "$EAGGL_POST"; do
  "$PY" -m eaggl factor \
    --deterministic \
    --seed 0 \
    --hide-progress \
    --max-gb 2 \
    --factor-gene-clusters-in "$EAGGL_DIR/gene_clusters.out.gz" \
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
    --factor-gmt-out "$EAGGL_DIR/factors_as_gene_sets.gmt.gz" \
    --trait-factor-links-output-detail full \
    --trait-factor-links-out "$EAGGL_DIR/trait_factor_links.nnls.out.gz" \
    --params-out "$EAGGL_DIR/trait_projection.params.out.gz" \
    --log-file "$EAGGL_DIR/trait_projection.run.log.gz" \
    --warnings-file "$EAGGL_DIR/trait_projection.warnings.log.gz" \
    > "$EAGGL_DIR/trait_projection.stdout.txt" \
    2> "$EAGGL_DIR/trait_projection.stderr.txt"

  "$PY" -m pigean betas \
    --X-in "$EAGGL_DIR/factors_as_gene_sets.gmt.gz" \
    --gene-universe-from-x \
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
    --gene-set-stats-out "$EAGGL_DIR/factor_trait_pigean_enrichments.out.gz" \
    --params-out "$EAGGL_DIR/factor_trait_pigean_enrichments.params.out.gz" \
    --log-file "$EAGGL_DIR/factor_trait_pigean_enrichments.run.log.gz" \
    --warnings-file "$EAGGL_DIR/factor_trait_pigean_enrichments.warnings.log.gz" \
    > "$EAGGL_DIR/factor_trait_pigean_enrichments.stdout.txt" \
    2> "$EAGGL_DIR/factor_trait_pigean_enrichments.stderr.txt"
done
```

## 7. Build Factor Graphs

```bash
for EAGGL_DIR in "$EAGGL_PRE" "$EAGGL_POST"; do
  "$PY" -m eaggl.factor_graph \
    --eaggl-dir "$EAGGL_DIR" \
    --gene-clusters-in "$EAGGL_DIR/gene_clusters_full.out.gz" \
    --gene-set-clusters-in "$EAGGL_DIR/gene_set_clusters.out.gz" \
    --trait-factor-links-in "$EAGGL_DIR/trait_factor_links.nnls.out.gz" \
    --factor-trait-enrichments-in "$EAGGL_DIR/factor_trait_pigean_enrichments.out.gz" \
    --color-by auto \
    --html-out "$EAGGL_DIR/factor_graph.full_direct.html" \
    --json-out "$EAGGL_DIR/factor_graph.full_direct.json" \
    > "$EAGGL_DIR/factor_graph.stdout.txt" \
    2> "$EAGGL_DIR/factor_graph.stderr.txt"
done
```

## 8. Build The Dashboard

The dashboard reads aggregate learn-phi outputs through `--eaggl-phi-sweep`; it does not require separate fixed-phi directories for each candidate phi.

```bash
"$PY" -m pigean.dashboard \
  --title "T2D + MODY multi-trait gene-by-gene EAGGL workflow" \
  --pigean-run "t2d_pre:$T2D_PRE" \
  --pigean-run "mody_pre:$MODY_PRE" \
  --pigean-run "t2d_post:$T2D_POST" \
  --pigean-run "mody_post:$MODY_POST" \
  --run-title "t2d_pre:T2D pre-exclusion PIGEAN" \
  --run-title "mody_pre:MODY pre-exclusion PIGEAN" \
  --run-title "t2d_post:T2D post-exclusion PIGEAN" \
  --run-title "mody_post:MODY post-exclusion PIGEAN" \
  --pigean-group "multi_pre:t2d_pre:T2D + MODY pre-exclusion" \
  --pigean-group "multi_pre:mody_pre:T2D + MODY pre-exclusion" \
  --pigean-group "multi_post:t2d_post:T2D + MODY post-exclusion" \
  --pigean-group "multi_post:mody_post:T2D + MODY post-exclusion" \
  --eaggl-phi-sweep "multi_pre:gene_by_gene_multi:$EAGGL_PRE" \
  --eaggl-phi-sweep "multi_post:gene_by_gene_multi:$EAGGL_POST" \
  --run-title "multi_pre:T2D+MODY pre-exclusion EAGGL multi-anchor" \
  --run-title "multi_post:T2D+MODY post-exclusion EAGGL multi-anchor" \
  --x-input "$X_MOUSE" \
  --x-input "$X_MSIGDB" \
  --x-input "$X_OCR" \
  --x-input "$X_STRING" \
  --html-out "$DASH/t2d_mody_multitrait_dashboard.html" \
  --json-out "$DASH/t2d_mody_multitrait_dashboard.json" \
  > "$DASH/stdout.txt" \
  2> "$DASH/stderr.txt"

echo "$DASH/t2d_mody_multitrait_dashboard.html"
```
