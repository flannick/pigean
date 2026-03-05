#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
OUT_DIR="${ROOT_DIR}/reports/release_v1"

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

echo "[release] Running full pigean test suite"
"${PYTHON_CMD}" "${ROOT_DIR}/scripts/run_with_metrics.py" --metrics-out "${OUT_DIR}/pytest.full.metrics.json" -- "${PYTHON_CMD}" -m pytest -q > "${OUT_DIR}/pytest.full.out" 2> "${OUT_DIR}/pytest.full.err"

echo "[release] Running MODY core regression tests"
"${PYTHON_CMD}" "${ROOT_DIR}/scripts/run_with_metrics.py" --metrics-out "${OUT_DIR}/pytest.mody_core.metrics.json" -- "${PYTHON_CMD}" -m pytest -q tests/test_mody_core_modes_regression_unittest.py > "${OUT_DIR}/pytest.mody_core.out" 2> "${OUT_DIR}/pytest.mody_core.err"

echo "[release] Running MODY Gibbs regression tests"
"${PYTHON_CMD}" "${ROOT_DIR}/scripts/run_with_metrics.py" --metrics-out "${OUT_DIR}/pytest.mody_gibbs.metrics.json" -- "${PYTHON_CMD}" -m pytest -q tests/test_mody_gibbs_regression_unittest.py > "${OUT_DIR}/pytest.mody_gibbs.out" 2> "${OUT_DIR}/pytest.mody_gibbs.err"

echo "[release] Running HuGE cache regression tests"
"${PYTHON_CMD}" "${ROOT_DIR}/scripts/run_with_metrics.py" --metrics-out "${OUT_DIR}/pytest.huge_cache.metrics.json" -- "${PYTHON_CMD}" -m pytest -q tests/test_huge_statistics_cache_regression_unittest.py > "${OUT_DIR}/pytest.huge_cache.out" 2> "${OUT_DIR}/pytest.huge_cache.err"

echo "[release] Running HuGE GWAS regression tests"
"${PYTHON_CMD}" "${ROOT_DIR}/scripts/run_with_metrics.py" --metrics-out "${OUT_DIR}/pytest.huge_gwas.metrics.json" -- "${PYTHON_CMD}" -m pytest -q tests/test_huge_real_gwas_regression_unittest.py > "${OUT_DIR}/pytest.huge_gwas.out" 2> "${OUT_DIR}/pytest.huge_gwas.err"

echo "[release] Completed. Logs are in ${OUT_DIR}"
