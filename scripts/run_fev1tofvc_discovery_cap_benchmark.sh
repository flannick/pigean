#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
REPO="$ROOT/eaggl_redundancy"
PYTHON="/Users/flannick/codex-workspace/analysis/.venv/bin/python"
DIG_SRC="/Users/flannick/codex-workspace/analysis/resources/repos/dig-open-data/src"
OUT_ROOT="$ROOT/results/fev1tofvc_discovery_binding_cap_benchmark_2026-04-19"
BASE="$ROOT/results/fev1tofvc_xin_matrix_fresh_2026-04-15/e_current_large_xin_fixed500_track"
GENE_SET_STATS="$BASE/pigean/pigean.gene_set_stats.out.gz"
GENE_STATS="$BASE/pigean/pigean.gene_stats.out.gz"
GENE_LOC="/Users/flannick/codex-workspace/analysis/resources/pigean/data/reference/NCBI37.3.plink.gene.loc"
SEQ_LOG="$OUT_ROOT/sequence.log"
SUMMARY_TSV="$OUT_ROOT/benchmark_summary.tsv"
SUMMARY_JSON="$OUT_ROOT/benchmark_summary.json"
THRESHOLD="0.5"

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
  local label="$1"
  local mode="$2"
  local cap="$3"
  local out_dir="$OUT_ROOT/$label"
  mkdir -p "$out_dir"

  echo "[$(date '+%F %T')] START $label mode=$mode cap=$cap" | tee -a "$SEQ_LOG"
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
      --discovery-redundancy-threshold "$THRESHOLD"
      --discovery-redundancy-weighting-mode "$mode"
      "${XARGS[@]}"
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
  echo "[$(date '+%F %T')] END $label mode=$mode cap=$cap" | tee -a "$SEQ_LOG"
}

summarize() {
  OUT_ROOT="$OUT_ROOT" SUMMARY_TSV="$SUMMARY_TSV" SUMMARY_JSON="$SUMMARY_JSON" "$PYTHON" - <<'PY'
import csv, gzip, json, math, os
from pathlib import Path
import numpy as np
import scipy.optimize

out_root = Path(os.environ['OUT_ROOT'])
summary_tsv = Path(os.environ['SUMMARY_TSV'])
summary_json = Path(os.environ['SUMMARY_JSON'])

def open_auto(path):
    with open(path, 'rb') as fh:
        magic = fh.read(2)
    if magic == b'\x1f\x8b':
        return gzip.open(path, 'rt')
    return open(path, 'r', encoding='utf-8')

def median(values):
    values = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    if not values:
        return None
    return float(np.median(np.asarray(values, dtype=float)))

def percentile(values, q):
    values = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    if not values:
        return None
    return float(np.quantile(np.asarray(values, dtype=float), q))

def load_params(path):
    vals = {}
    if not path.exists():
        return vals
    with open_auto(path) as f:
        next(f)
        for line in f:
            parts = line.rstrip('\n').split('\t')
            if len(parts) == 3:
                vals[parts[0]] = parts[2]
    return vals

def factor_columns(fieldnames):
    cols = []
    for name in fieldnames:
        if not name.startswith('Factor'):
            continue
        suffix = name[len('Factor'):]
        if suffix.isdigit():
            cols.append((int(suffix), name))
    cols.sort()
    return [name for _, name in cols]

def load_gene_matrix(path):
    if not path.exists():
        return None
    with open_auto(path) as f:
        reader = csv.DictReader(f, delimiter='\t')
        cols = factor_columns(reader.fieldnames or [])
        if not cols:
            return None
        rows = [[float(row.get(col, 0.0) or 0.0) for col in cols] for row in reader]
    if not rows:
        return np.zeros((0, 0), dtype=float)
    return np.asarray(rows, dtype=float)

def normalize_cols(matrix):
    if matrix is None or matrix.size == 0:
        return None
    norms = np.linalg.norm(matrix, axis=0)
    norms[norms == 0] = 1.0
    return matrix / norms[np.newaxis, :]

def matched_cosines(ref, other):
    if ref is None or other is None or ref.size == 0 or other.size == 0:
        return []
    ref_n = normalize_cols(ref)
    other_n = normalize_cols(other)
    sim = np.clip(ref_n.T @ other_n, -1.0, 1.0)
    ri, oi = scipy.optimize.linear_sum_assignment(1.0 - sim)
    return [float(sim[r, o]) for r, o in zip(ri, oi)]

rows = []
matrices = {}
for run_dir in sorted(p for p in out_root.iterdir() if p.is_dir()):
    params = load_params(run_dir / 'params.out.gz')
    factor_metrics_path = run_dir / 'factor_metrics.out.gz'
    gene_clusters_path = run_dir / 'gene_clusters.out.gz'
    if not (factor_metrics_path.exists() and gene_clusters_path.exists()):
        continue
    with open_auto(factor_metrics_path) as f:
        metrics = list(csv.DictReader(f, delimiter='\t'))
    masses = []
    gene_overlap = []
    gene_set_overlap = []
    primary = 0
    for row in metrics:
        try:
            masses.append(float(row['combined_mass_fraction']))
        except Exception:
            pass
        if row.get('factor_mass_floor_0p5pct') == '1':
            primary += 1
        try:
            gene_overlap.append(float(row.get('gene_max_jaccard', 'nan')))
        except Exception:
            pass
        try:
            gene_set_overlap.append(float(row.get('gene_set_max_jaccard', 'nan')))
        except Exception:
            pass
    masses = [m for m in masses if math.isfinite(m) and m > 0]
    total_mass = sum(masses)
    effective = None
    if total_mass > 0:
        probs = [m / total_mass for m in masses]
        effective = float(math.exp(-sum(p * math.log(p) for p in probs)))
    discovery_mean = []
    discovery_eff = []
    with open_auto(run_dir / 'gene_set_clusters.out.gz') as f:
        for row in csv.DictReader(f, delimiter='\t'):
            try:
                discovery_mean.append(float(row.get('discovery_family_mean_similarity', 'nan')))
            except Exception:
                pass
            try:
                discovery_eff.append(float(row.get('discovery_family_effective_size', 'nan')))
            except Exception:
                pass
    matrices[run_dir.name] = load_gene_matrix(gene_clusters_path)
    rows.append({
        'run': run_dir.name,
        'weighting_mode': params.get('discovery_redundancy_weighting_mode'),
        'threshold': params.get('discovery_redundancy_threshold'),
        'max_num_discovery_gene_sets': params.get('max_num_discovery_gene_sets', 'uncapped') or 'uncapped',
        'raw_factor_count': len(metrics),
        'primary_factor_count': primary,
        'effective_factor_count': effective,
        'retained_gene_sets': int(params.get('num_retained_gene_sets', '0') or 0),
        'in_discovery_gene_sets': int(params.get('num_discovery_gene_sets', '0') or 0),
        'mean_discovery_family_mean_similarity': median(discovery_mean),
        'mean_discovery_family_effective_size': median(discovery_eff),
        'top_factor_mass_fraction': max(masses) if masses else None,
        'top5_mass_fraction': float(sum(sorted(masses, reverse=True)[:5])) if masses else None,
        'median_gene_max_jaccard': median(gene_overlap),
        'median_gene_set_max_jaccard': median(gene_set_overlap),
    })

reference_effective = next((row['run'] for row in rows if row['weighting_mode'] == 'effective_size' and row['max_num_discovery_gene_sets'] == 'uncapped'), None)
reference_none = next((row['run'] for row in rows if row['weighting_mode'] == 'none' and row['max_num_discovery_gene_sets'] == 'uncapped'), None)
for row in rows:
    cur = matrices.get(row['run'])
    for suffix, ref_name in [('effective_size_ref', reference_effective), ('none_ref', reference_none)]:
        ref = matrices.get(ref_name)
        cosines = matched_cosines(ref, cur) if ref_name is not None else []
        ref_k = ref.shape[1] if ref is not None else None
        cur_k = cur.shape[1] if cur is not None else None
        row[f'reference_run_{suffix}'] = ref_name
        row[f'matched_cosine_median_{suffix}'] = median(cosines)
        row[f'matched_cosine_p10_{suffix}'] = percentile(cosines, 0.1)
        row[f'unmatched_factor_count_{suffix}'] = abs(ref_k - cur_k) if ref_k is not None and cur_k is not None else None

fields = [
    'run','weighting_mode','threshold','max_num_discovery_gene_sets','raw_factor_count','primary_factor_count',
    'effective_factor_count','retained_gene_sets','in_discovery_gene_sets','mean_discovery_family_mean_similarity',
    'mean_discovery_family_effective_size','top_factor_mass_fraction','top5_mass_fraction',
    'median_gene_max_jaccard','median_gene_set_max_jaccard','reference_run_effective_size_ref',
    'matched_cosine_median_effective_size_ref','matched_cosine_p10_effective_size_ref','unmatched_factor_count_effective_size_ref',
    'reference_run_none_ref','matched_cosine_median_none_ref','matched_cosine_p10_none_ref','unmatched_factor_count_none_ref'
]
with summary_tsv.open('w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=fields, delimiter='\t')
    writer.writeheader()
    writer.writerows(rows)
with summary_json.open('w') as f:
    json.dump(rows, f, indent=2)
print(summary_tsv)
PY
}

run_case effective_size_cap100 effective_size 100
summarize | tee -a "$SEQ_LOG"
run_case effective_size_cap250 effective_size 250
summarize | tee -a "$SEQ_LOG"
run_case effective_size_cap500 effective_size 500
summarize | tee -a "$SEQ_LOG"
run_case effective_size_cap750 effective_size 750
summarize | tee -a "$SEQ_LOG"
run_case effective_size_cap1000 effective_size 1000
summarize | tee -a "$SEQ_LOG"
run_case effective_size_uncapped effective_size uncapped
summarize | tee -a "$SEQ_LOG"
run_case none_cap500 none 500
summarize | tee -a "$SEQ_LOG"
run_case none_uncapped none uncapped
summarize | tee -a "$SEQ_LOG"
run_case log_effective_size_uncapped log_effective_size uncapped
summarize | tee -a "$SEQ_LOG"
