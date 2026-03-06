# EAGGL v1 Release Status

Date: 2026-03-05

## Checklist Gates
- `./scripts/check_shared_utils_sync.py`: PASS
- `../../.venv/bin/python scripts/generate_cli_manifest.py --check`: PASS
- `./scripts/release_readiness_check.sh`: PASS

## Test/Regression Summary
- Full suite: PASS (`105 passed, 18 subtests passed`)
- `scripts/finalize_regression_checks.sh`: PASS

## Runtime/Memory (from `reports/release_v1/*.metrics.json`)
- `pytest.full`: 18.81s, 106272 KB max RSS
- `finalize_checks`: 25.60s, 106544 KB max RSS

## Notes
- During N10, frozen effective-config references were refreshed and committed to match current behavior.
- Raw logs and metrics are under `reports/release_v1/`.
