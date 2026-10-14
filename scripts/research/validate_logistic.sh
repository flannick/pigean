#!/usr/bin/env bash
set -euo pipefail
code_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
export PYTHONPATH="$code_root/src:$code_root/research${PYTHONPATH:+:$PYTHONPATH}"
export MPLBACKEND=Agg MPLCONFIGDIR="$HOME/codex-workspace/.cache/matplotlib"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
cd "$code_root"
../../../.venv/bin/python -m pytest -q tests/test_logistic_sampling.py tests/test_gibbs_default_scope.py tests/test_gibbs_features.py "$@"
