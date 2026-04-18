#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
REPO="$ROOT/eaggl_redundancy"
PYTHON="/Users/flannick/codex-workspace/analysis/.venv/bin/python"
DIG_SRC="/Users/flannick/codex-workspace/analysis/resources/repos/dig-open-data/src"
BASE="$ROOT/results/fev1tofvc_xin_matrix_fresh_2026-04-15/e_current_large_xin_fixed500_track"
GENE_SET_STATS="$BASE/pigean/pigean.gene_set_stats.out.gz"
GENE_STATS="$BASE/pigean/pigean.gene_stats.out.gz"
GENE_LOC="/Users/flannick/codex-workspace/analysis/resources/pigean/data/reference/NCBI37.3.plink.gene.loc"
OUT_ROOT="$ROOT/results/fev1tofvc_discovery_threshold_sweep_2026-04-17"
SEQ_LOG="$OUT_ROOT/sequence.log"
mkdir -p "$OUT_ROOT"
XARGS=(
  --X-in /Users/flannick/codex-workspace/analysis/resources/pigean/data/small/gene_set_list_mouse_2024.txt
  --X-in /Users/flannick/codex-workspace/analysis/resources/pigean/data/small/gene_set_list_msigdb_nohp.txt
  --X-in /Users/flannick/codex-workspace/analysis/resources/pigean/data/large/gene_set_list_mesh.txt
  --X-in /Users/flannick/codex-workspace/analysis/resources/pigean/data/large/gene_set_list_ocr_human.txt
  --X-in /Users/flannick/codex-workspace/analysis/resources/pigean/data/large/gene_set_list_string_notext_medium.txt
  --X-in /Users/flannick/codex-workspace/analysis/resources/pigean/data/large/gene_set_list_pops_sparse_small.txt
)
run_case() {
  local label="$1"; shift
  local out_dir="$OUT_ROOT/$label"
  mkdir -p "$out_dir"
  echo "[$(date '+%F %T')] START $label" | tee -a "$SEQ_LOG"
  (
    cd "$REPO"
    export PYTHONPATH="src:$DIG_SRC" MPLBACKEND=Agg MPLCONFIGDIR="$HOME/codex-workspace/.cache/matplotlib"
    cmd=(
      "$PYTHON" -m eaggl factor --deterministic --hide-progress --phi 0.05 --max-num-factors 200
      "${XARGS[@]}"
      --gene-set-stats-in "$GENE_SET_STATS"
      --gene-stats-in "$GENE_STATS"
      --gene-stats-id-col Gene --gene-stats-log-bf-col log_bf --gene-stats-combined-col combined
      --gene-loc-file "$GENE_LOC"
      --factors-out "$out_dir/factors.out.gz"
      --factor-metrics-out "$out_dir/factor_metrics.out.gz"
      --gene-set-clusters-out "$out_dir/gene_set_clusters.out.gz"
      --gene-clusters-out "$out_dir/gene_clusters.out.gz"
      --params-out "$out_dir/params.out.gz"
      --log-file "$out_dir/eaggl.run.log.gz"
      --warnings-file "$out_dir/eaggl.warnings.log.gz"
      "$@"
    )
    printf '%q ' "${cmd[@]}" > "$out_dir/command.sh"; printf '\n' >> "$out_dir/command.sh"; chmod +x "$out_dir/command.sh"
    "${cmd[@]}"
  )
  echo "[$(date '+%F %T')] END $label" | tee -a "$SEQ_LOG"
}
thresholds=(0.2 0.35 0.5 0.65 0.8)
for t in "${thresholds[@]}"; do
  run_case "default_t${t//./}" --discovery-redundancy-threshold "$t"
  run_case "noweight_t${t//./}" --discovery-redundancy-threshold "$t" --no-discovery-redundancy-weighting
  OUT_ROOT="$OUT_ROOT" "$PYTHON" - <<'PY'
import csv, gzip, json, math, os
from pathlib import Path
root=Path(os.environ['OUT_ROOT'])
rows=[]
for run_dir in sorted(p for p in root.iterdir() if p.is_dir()):
    params=run_dir/'params.out.gz'
    vals={}
    if params.exists():
        with params.open() as f:
            next(f)
            for line in f:
                parts=line.rstrip('\n').split('\t')
                if len(parts)==3:
                    vals[parts[0]]=parts[2]
    fm=run_dir/'factor_metrics.out.gz'
    gsc=run_dir/'gene_set_clusters.out.gz'
    raw=primary=0; eff=None
    if fm.exists():
        with gzip.open(fm,'rt') as f: fr=list(csv.DictReader(f, delimiter='\t'))
        raw=len(fr); primary=sum(r.get('factor_mass_floor_0p5pct')=='1' for r in fr)
        probs=[float(r['combined_mass_fraction']) for r in fr if r.get('combined_mass_fraction')]
        s=sum(probs)
        if s>0:
            probs=[p/s for p in probs if p>0]
            eff=math.exp(-sum(p*math.log(p) for p in probs))
    retained=int(vals.get('num_retained_gene_sets','0')) if vals else 0
    disc=int(vals.get('num_discovery_gene_sets','0')) if vals else 0
    status='completed' if fm.exists() and gsc.exists() else ('collapsed_or_no_final_outputs' if params.exists() else 'running')
    rows.append({'run':run_dir.name,'raw_factor_count':raw,'primary_factor_count':primary,'effective_factor_count':eff,'retained_gene_sets':retained,'in_discovery_gene_sets':disc,'status':status})
fields=list(rows[0].keys()) if rows else ['run']
with (root/'threshold_summary.tsv').open('w', newline='') as f:
    w=csv.DictWriter(f, fieldnames=fields, delimiter='\t'); w.writeheader(); w.writerows(rows)
with (root/'threshold_summary.json').open('w') as f: json.dump(rows, f, indent=2)
PY
done
