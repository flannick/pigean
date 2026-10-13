#!/bin/bash
set -eo pipefail
cd "$(dirname "$0")/.."
export PYTHONHASHSEED=0 PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export MPLBACKEND=Agg MPLCONFIGDIR="$HOME/codex-workspace/.cache/matplotlib"
base=results/autonomous_20261012/T2D_full_pair
mkdir "$base"
for label in milestone inner; do
    extra=()
    if [[ "$label" == milestone ]]; then
        repo=worktrees/parallel-prefilter
    else
        repo=worktrees/inner-fused
        extra=(--gibbs-enable inner-matmul --gibbs-enable inner-fused)
    fi
    ../.venv/bin/python scripts/screen_inference.py \
        --repo "$repo" --fixture results/feature_foundation_20261011/fixtures/T2D/fixture.pkl.gz \
        --out "$base/$label" --iterations 500 --seconds 1800 --seed 0 \
        --engine parallel-prefilter --prepare-engine --write-outputs \
        --fixed-budget-validation --trace-stride 100 --monitor "${extra[@]}" \
        > "$base/$label.console.log" 2>&1
done
../.venv/bin/python scripts/compare_inference_screen.py \
    --baseline "$base/milestone" --candidate "$base/inner" \
    --out "$base/comparison.json" > "$base/comparison.console.log"
