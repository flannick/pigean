#!/bin/bash
set -eo pipefail
cd "$(dirname "$0")/.."
export PYTHONHASHSEED=0 PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export MPLBACKEND=Agg MPLCONFIGDIR="$HOME/codex-workspace/.cache/matplotlib"
base=results/autonomous_20261012
flags=(--gibbs-enable inner-matmul --gibbs-enable inner-fused --gibbs-enable inner-batching --gibbs-enable compiled-prefilter --gibbs-enable inner-cache64 --gibbs-enable huge-prior-softmax)
../.venv/bin/python scripts/screen_inference.py --repo worktrees/optimization-stage-2 \
    --fixture results/feature_foundation_20261011/fixtures/T2D/fixture.pkl.gz \
    --out "$base/T2D_final_full" --iterations 500 --seconds 1800 --seed 0 \
    --engine parallel-prefilter --prepare-engine --write-outputs \
    --fixed-budget-validation --trace-stride 100 --monitor "${flags[@]}" > "$base/T2D_final_full.console.log" 2>&1
for control in milestone inner; do
    ../.venv/bin/python scripts/compare_inference_screen.py \
        --baseline "$base/T2D_full_pair/$control" --candidate "$base/T2D_final_full" \
        --out "$base/T2D_final_vs_$control.json" > "$base/T2D_final_vs_$control.console.log"
done
../.venv/bin/python scripts/compare_inference_screen.py --baseline "$base/T2D_stage2_full" --candidate "$base/T2D_final_full" \
    --out "$base/T2D_final_vs_four.json" > "$base/T2D_final_vs_four.console.log"
../.venv/bin/python scripts/screen_inference.py --repo worktrees/optimization-stage-2 \
    --fixture results/feature_foundation_20261011/fixtures/T2D/fixture.pkl.gz \
    --out "$base/profile_final_T2D_100" --iterations 101 --seconds 300 --seed 0 \
    --engine parallel-prefilter --prepare-engine --profile --monitor "${flags[@]}" > "$base/profile_final_T2D_100.console.log" 2>&1
