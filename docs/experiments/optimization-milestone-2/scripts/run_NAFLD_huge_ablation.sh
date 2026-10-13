#!/bin/bash
set -eo pipefail
cd "$(dirname "$0")/.."
export PYTHONHASHSEED=0 PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export MPLBACKEND=Agg MPLCONFIGDIR="$HOME/codex-workspace/.cache/matplotlib"
base=results/autonomous_20261012
../.venv/bin/python scripts/screen_inference.py --repo worktrees/optimization-stage-2 \
    --fixture results/feature_foundation_20261011/fixtures/NAFLD/fixture.pkl.gz \
    --out "$base/NAFLD_without_huge" --iterations 500 --seconds 1800 --seed 0 \
    --engine parallel-prefilter --prepare-engine --write-outputs --fixed-budget-validation --trace-stride 100 --monitor \
    --gibbs-enable inner-matmul --gibbs-enable inner-fused --gibbs-enable inner-batching \
    --gibbs-enable compiled-prefilter --gibbs-enable inner-cache64 --gibbs-disable huge-prior-softmax \
    > "$base/NAFLD_without_huge.console.log" 2>&1
../.venv/bin/python scripts/compare_inference_screen.py --baseline "$base/NAFLD_full_pair/milestone" \
    --candidate "$base/NAFLD_without_huge" --out "$base/NAFLD_without_huge.comparison.json" \
    > "$base/NAFLD_without_huge.comparison.console.log"
