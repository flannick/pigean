#!/bin/bash
set -eo pipefail
cd "$(dirname "$0")/.."
export PYTHONHASHSEED=0 PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export MPLBACKEND=Agg MPLCONFIGDIR="$HOME/codex-workspace/.cache/matplotlib"
base=results/autonomous_20261012
../.venv/bin/python scripts/screen_inference.py --repo worktrees/parallel-prefilter \
    --fixture results/feature_foundation_20261011/fixtures/NAFLD/fixture.pkl.gz \
    --out "$base/NAFLD_milestone_seed1" --iterations 500 --seconds 1800 --seed 1 \
    --engine parallel-prefilter --prepare-engine --write-outputs --fixed-budget-validation --trace-stride 100 --monitor \
    > "$base/NAFLD_milestone_seed1.console.log" 2>&1
../.venv/bin/python scripts/compare_output_sensitivity.py --reference "$base/NAFLD_full_pair/milestone" \
    --candidate "$base/NAFLD_milestone_seed1" --out "$base/NAFLD_seed_sensitivity.json" \
    > "$base/NAFLD_seed_sensitivity.console.log"
