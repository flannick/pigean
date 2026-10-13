#!/bin/bash
set -eo pipefail
cd "$(dirname "$0")/.."
export PYTHONHASHSEED=0 PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export MPLBACKEND=Agg MPLCONFIGDIR="$HOME/codex-workspace/.cache/matplotlib"
label=${1:?Choose reference, early, or mean}
base=results/autonomous_20261012/effort_T2D
mkdir -p "$base"
flags=(--gibbs-enable inner-matmul --gibbs-enable inner-fused --gibbs-enable inner-batching --gibbs-enable compiled-prefilter --gibbs-enable inner-cache64 --gibbs-enable huge-prior-softmax --gibbs-enable static-correlation)
case "$label" in
    reference) ;;
    early) flags+=(--gibbs-enable inner-early-check) ;;
    mean) flags+=(--gibbs-enable inner-mean) ;;
    *) exit 2 ;;
esac
../.venv/bin/python scripts/screen_inference.py --repo worktrees/optimization-stage-2 \
    --fixture results/feature_foundation_20261011/fixtures/T2D/fixture.pkl.gz \
    --out "$base/$label" --iterations 101 --seconds 300 --seed 0 \
    --engine parallel-prefilter --prepare-engine --write-outputs \
    --trace-stride 25 --monitor "${flags[@]}" > "$base/$label.console.log" 2>&1
if [[ "$label" != reference ]]; then
    ../.venv/bin/python scripts/compare_inference_screen.py --baseline "$base/reference" --candidate "$base/$label" \
        --out "$base/$label.comparison.json" > "$base/$label.comparison.console.log"
fi
