#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
REPO="$ROOT/eaggl_redundancy"
PYTHON="/Users/flannick/codex-workspace/analysis/.venv/bin/python"
DIG_SRC="/Users/flannick/codex-workspace/analysis/resources/repos/dig-open-data/src"
OUT_ROOT="$ROOT/results/fev1tofvc_discovery_cap_benchmark_2026-04-17"
BASE="$ROOT/results/fev1tofvc_xin_matrix_fresh_2026-04-15/e_current_large_xin_fixed500_track"
X1="/Users/flannick/codex-workspace/analysis/resources/pigean/data/small/gene_set_list_mouse_2024.txt"
X2="/Users/flannick/codex-workspace/analysis/resources/pigean/data/small/gene_set_list_msigdb_nohp.txt"
X3="/Users/flannick/codex-workspace/analysis/resources/pigean/data/large/gene_set_list_mesh.txt"
X4="/Users/flannick/codex-workspace/analysis/resources/pigean/data/large/gene_set_list_ocr_human.txt"
X5="/Users/flannick/codex-workspace/analysis/resources/pigean/data/large/gene_set_list_string_notext_medium.txt"
X6="/Users/flannick/codex-workspace/analysis/resources/pigean/data/large/gene_set_list_pops_sparse_small.txt"
GENE_SET_STATS="$BASE/pigean/pigean.gene_set_stats.out.gz"
GENE_STATS="$BASE/pigean/pigean.gene_stats.out.gz"
GENE_LOC="/Users/flannick/codex-workspace/analysis/resources/pigean/data/reference/NCBI37.3.plink.gene.loc"
SEQ_LOG="$OUT_ROOT/sequence.log"
SUMMARY_TSV="$OUT_ROOT/benchmark_summary.tsv"
SUMMARY_JSON="$OUT_ROOT/benchmark_summary.json"

mkdir -p "$OUT_ROOT"

run_case() {
  local label="$1"
  local cap="$2"
  local out_dir="$OUT_ROOT/$label"
  mkdir -p "$out_dir"

  if [[ -f "$out_dir/factor_metrics.out.gz" && -f "$out_dir/gene_set_clusters.out.gz" ]]; then
    echo "[$(date '+%F %T')] SKIP $label existing final outputs" | tee -a "$SEQ_LOG"
    return
  fi

  echo "[$(date '+%F %T')] START $label cap=$cap" | tee -a "$SEQ_LOG"
  (
    cd "$REPO"
    export PYTHONPATH="src:$DIG_SRC"
    export MPLBACKEND=Agg
    export MPLCONFIGDIR="$HOME/codex-workspace/.cache/matplotlib"
    cmd=(
      "$PYTHON" -m eaggl factor
      --deterministic
      --hide-progress
      --phi 0.05
      --max-num-factors 200
      --X-in "$X1"
      --X-in "$X2"
      --X-in "$X3"
      --X-in "$X4"
      --X-in "$X5"
      --X-in "$X6"
      --gene-set-stats-in "$GENE_SET_STATS"
      --gene-stats-in "$GENE_STATS"
      --gene-stats-id-col Gene
      --gene-stats-log-bf-col log_bf
      --gene-stats-combined-col combined
      --gene-loc-file "$GENE_LOC"
      --factors-out "$out_dir/factors.out.gz"
      --factor-metrics-out "$out_dir/factor_metrics.out.gz"
      --gene-set-clusters-out "$out_dir/gene_set_clusters.out.gz"
      --gene-clusters-out "$out_dir/gene_clusters.out.gz"
      --params-out "$out_dir/params.out.gz"
      --log-file "$out_dir/eaggl.run.log.gz"
      --warnings-file "$out_dir/eaggl.warnings.log.gz"
    )
    if [[ "$cap" != "uncapped" ]]; then
      cmd+=(--max-num-discovery-gene-sets "$cap")
    fi
    printf '%q ' "${cmd[@]}" > "$out_dir/command.sh"
    printf '\n' >> "$out_dir/command.sh"
    chmod +x "$out_dir/command.sh"
    "${cmd[@]}"
  )
  echo "[$(date '+%F %T')] END $label cap=$cap" | tee -a "$SEQ_LOG"
}

summarize() {
  OUT_ROOT="$OUT_ROOT" SUMMARY_TSV="$SUMMARY_TSV" SUMMARY_JSON="$SUMMARY_JSON" "$PYTHON" - <<'PY'
import csv, gzip, json, math, os
from pathlib import Path
out_root = Path(os.environ['OUT_ROOT'])
summary_tsv = Path(os.environ['SUMMARY_TSV'])
summary_json = Path(os.environ['SUMMARY_JSON'])
rows = []
for run_dir in sorted(p for p in out_root.iterdir() if p.is_dir()):
    fm = run_dir / 'factor_metrics.out.gz'
    gsc = run_dir / 'gene_set_clusters.out.gz'
    if not (fm.exists() and gsc.exists()):
        continue
    with gzip.open(fm, 'rt') as f:
        r = list(csv.DictReader(f, delimiter='\t'))
    masses = []
    primary = 0
    median_unique = []
    median_g = []
    median_gs = []
    for row in r:
        try:
            masses.append(float(row['combined_mass_fraction']))
        except Exception:
            pass
        if row.get('factor_mass_floor_0p5pct') == '1':
            primary += 1
        for key, bucket in [
            ('combined_unique_fraction', median_unique),
            ('gene_max_jaccard', median_g),
            ('gene_set_max_jaccard', median_gs),
        ]:
            try:
                bucket.append(float(row[key]))
            except Exception:
                pass
    total = sum(masses)
    if total > 0:
        probs = [m/total for m in masses if m > 0]
        effective = math.exp(-sum(p*math.log(p) for p in probs))
    else:
        effective = float('nan')
    with gzip.open(gsc, 'rt') as f:
        rr = list(csv.DictReader(f, delimiter='\t'))
    retained = len(rr)
    in_disc = sum(str(row.get('in_discovery', '')).lower() in {'1','true','t','yes'} for row in rr)
    def med(xs):
        if not xs:
            return float('nan')
        xs = sorted(xs)
        n = len(xs)
        m = n//2
        return xs[m] if n%2 else 0.5*(xs[m-1]+xs[m])
    rows.append({
        'run': run_dir.name,
        'raw_factor_count': len(r),
        'primary_factor_count': primary,
        'effective_factor_count': round(effective, 6) if effective == effective else None,
        'retained_gene_sets': retained,
        'in_discovery_gene_sets': in_disc,
        'median_combined_unique_fraction': round(med(median_unique), 6) if median_unique else None,
        'median_gene_max_jaccard': round(med(median_g), 6) if median_g else None,
        'median_gene_set_max_jaccard': round(med(median_gs), 6) if median_gs else None,
    })
fields = ['run','raw_factor_count','primary_factor_count','effective_factor_count','retained_gene_sets','in_discovery_gene_sets','median_combined_unique_fraction','median_gene_max_jaccard','median_gene_set_max_jaccard']
with summary_tsv.open('w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=fields, delimiter='\t')
    w.writeheader()
    w.writerows(rows)
with summary_json.open('w') as f:
    json.dump(rows, f, indent=2)
print(summary_tsv)
PY
}

caps=(2000 4000 8000 12000 uncapped)
for cap in "${caps[@]}"; do
  label="cap_${cap}"
  run_case "$label" "$cap"
  summarize | tee -a "$SEQ_LOG"
done
