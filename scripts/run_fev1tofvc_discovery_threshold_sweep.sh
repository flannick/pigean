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
OUT_ROOT="$ROOT/results/fev1tofvc_discovery_threshold_sweep_2026-04-19"
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
thresholds=(0.5 0.65 0.8)
weighting_modes=(none effective_size log_effective_size)
for t in "${thresholds[@]}"; do
  for mode in "${weighting_modes[@]}"; do
    run_case "${mode}_t${t//./}" \
      --discovery-redundancy-threshold "$t" \
      --discovery-redundancy-weighting-mode "$mode"
  done
done
OUT_ROOT="$OUT_ROOT" "$PYTHON" - <<'PY'
import csv, gzip, json, math, os
from pathlib import Path
import numpy as np
import scipy.optimize

root = Path(os.environ['OUT_ROOT'])

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
        rows = []
        for row in reader:
            rows.append([float(row.get(col, 0.0) or 0.0) for col in cols])
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

def load_factor_metrics(path):
    if not path.exists():
        return []
    with open_auto(path) as f:
        return list(csv.DictReader(f, delimiter='\t'))

runs = []
matrices = {}
for run_dir in sorted(p for p in root.iterdir() if p.is_dir()):
    params = load_params(run_dir / 'params.out.gz')
    metrics = load_factor_metrics(run_dir / 'factor_metrics.out.gz')
    gene_matrix = load_gene_matrix(run_dir / 'gene_clusters.out.gz')
    matrices[run_dir.name] = gene_matrix
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
    if (run_dir / 'gene_set_clusters.out.gz').exists():
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
    threshold = params.get('discovery_redundancy_threshold')
    mode = params.get('discovery_redundancy_weighting_mode')
    runs.append({
        'run': run_dir.name,
        'threshold': threshold,
        'weighting_mode': mode,
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
        'status': 'completed' if metrics and gene_matrix is not None else ('collapsed_or_no_final_outputs' if params else 'running'),
    })

reference_by_threshold = {
    row['threshold']: row['run']
    for row in runs
    if row['weighting_mode'] == 'none' and row['status'] == 'completed'
}
for row in runs:
    ref_name = reference_by_threshold.get(row['threshold'])
    ref = matrices.get(ref_name)
    cur = matrices.get(row['run'])
    cosines = matched_cosines(ref, cur) if ref_name is not None else []
    ref_k = ref.shape[1] if ref is not None else None
    cur_k = cur.shape[1] if cur is not None else None
    row['reference_run_vs_none'] = ref_name
    row['matched_cosine_median_vs_none'] = median(cosines)
    row['matched_cosine_p10_vs_none'] = percentile(cosines, 0.1)
    row['unmatched_factor_count_vs_none'] = (abs(ref_k - cur_k) if ref_k is not None and cur_k is not None else None)

fields = [
    'run','threshold','weighting_mode','raw_factor_count','primary_factor_count','effective_factor_count',
    'retained_gene_sets','in_discovery_gene_sets','mean_discovery_family_mean_similarity',
    'mean_discovery_family_effective_size','top_factor_mass_fraction','top5_mass_fraction',
    'median_gene_max_jaccard','median_gene_set_max_jaccard','reference_run_vs_none',
    'matched_cosine_median_vs_none','matched_cosine_p10_vs_none','unmatched_factor_count_vs_none','status'
]
with (root/'threshold_summary.tsv').open('w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=fields, delimiter='\t')
    writer.writeheader()
    writer.writerows(runs)
with (root/'threshold_summary.json').open('w') as f:
    json.dump(runs, f, indent=2)
PY
