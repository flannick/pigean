# EAGGL v1 Release Checklist

## Scope
- Branch is clean except for known untracked local artifacts.
- CLI manifest generation/check passes.
- Shared utils sync check passes.
- Full test suite passes.
- Factor regression finalization checks pass.
- Runtime/memory checks are captured using `scripts/release_readiness_check.sh`.

## Commands
```bash
cd eaggl
./scripts/check_shared_utils_sync.py
./scripts/generate_cli_manifest.py --check
./scripts/release_readiness_check.sh
```

## Pass/Fail Criteria
- All commands above exit with status `0`.
- No drift in generated artifacts after running checks.
- Regression and workflow checks pass in `reports/release_v1/` logs.

## Artifacts
- `reports/release_v1/pytest.full.out`
- `reports/release_v1/finalize_checks.out`
- matching `*.metrics.json` files with runtime/memory metrics
- matching `*.err` files for stderr capture
