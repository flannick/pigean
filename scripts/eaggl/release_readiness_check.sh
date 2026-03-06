#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
OUT_DIR="${ROOT_DIR}/reports/release_v1/eaggl"

PYTHON_CMD=""
for candidate in "${ROOT_DIR}/../.venv/bin/python" "${ROOT_DIR}/../../.venv/bin/python"; do
    if [ -x "${candidate}" ]; then
        PYTHON_CMD="${candidate}"
        break
    fi
done

if [ -z "${PYTHON_CMD}" ]; then
    echo "[release] ERROR: could not find venv python at ../.venv/bin/python or ../../.venv/bin/python" >&2
    exit 1
fi

mkdir -p "${OUT_DIR}"

echo "[release] ROOT_DIR=${ROOT_DIR}"
echo "[release] OUT_DIR=${OUT_DIR}"

echo "[release] Running canonical eaggl test suite"
"${PYTHON_CMD}" "${ROOT_DIR}/scripts/eaggl/run_with_metrics.py" --metrics-out "${OUT_DIR}/pytest.full.metrics.json" -- "${PYTHON_CMD}" -m pytest -q tests/eaggl > "${OUT_DIR}/pytest.full.out" 2> "${OUT_DIR}/pytest.full.err"

echo "[release] Running finalize regression checks"
"${PYTHON_CMD}" "${ROOT_DIR}/scripts/eaggl/run_with_metrics.py" --metrics-out "${OUT_DIR}/finalize_checks.metrics.json" -- "${ROOT_DIR}/scripts/eaggl/finalize_regression_checks.sh" > "${OUT_DIR}/finalize_checks.out" 2> "${OUT_DIR}/finalize_checks.err"

echo "[release] Completed. Logs are in ${OUT_DIR}"
