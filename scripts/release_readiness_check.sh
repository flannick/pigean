#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON_CMD="${ROOT_DIR}/../.venv/bin/python"
OUT_DIR="${ROOT_DIR}/reports/release_v1"

mkdir -p "${OUT_DIR}"

echo "[release] ROOT_DIR=${ROOT_DIR}"
echo "[release] OUT_DIR=${OUT_DIR}"

echo "[release] Running full pigean test suite"
/usr/bin/time -l "${PYTHON_CMD}" -m pytest -q > "${OUT_DIR}/pytest.full.out" 2> "${OUT_DIR}/pytest.full.time"

echo "[release] Running MODY core regression tests"
/usr/bin/time -l "${PYTHON_CMD}" -m pytest -q tests/test_mody_core_modes_regression_unittest.py > "${OUT_DIR}/pytest.mody_core.out" 2> "${OUT_DIR}/pytest.mody_core.time"

echo "[release] Running MODY Gibbs regression tests"
/usr/bin/time -l "${PYTHON_CMD}" -m pytest -q tests/test_mody_gibbs_regression_unittest.py > "${OUT_DIR}/pytest.mody_gibbs.out" 2> "${OUT_DIR}/pytest.mody_gibbs.time"

echo "[release] Running HuGE cache regression tests"
/usr/bin/time -l "${PYTHON_CMD}" -m pytest -q tests/test_huge_statistics_cache_regression_unittest.py > "${OUT_DIR}/pytest.huge_cache.out" 2> "${OUT_DIR}/pytest.huge_cache.time"

echo "[release] Running HuGE GWAS regression tests"
/usr/bin/time -l "${PYTHON_CMD}" -m pytest -q tests/test_huge_real_gwas_regression_unittest.py > "${OUT_DIR}/pytest.huge_gwas.out" 2> "${OUT_DIR}/pytest.huge_gwas.time"

echo "[release] Completed. Logs are in ${OUT_DIR}"
