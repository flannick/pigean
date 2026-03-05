# PIGEAN v1 Release Checklist

## Scope
- Branch is clean except for known untracked local artifacts.
- CLI manifest generation/check passes.
- Shared utils sync check passes.
- Full test suite passes.
- Legacy parity/regression tests pass (MODY + HuGE cache + HuGE GWAS).
- Runtime/memory checks are captured using `scripts/release_readiness_check.sh`.

## Commands
```bash
cd pigean
./scripts/check_shared_utils_sync.py
./scripts/generate_cli_manifest.py --check
./scripts/release_readiness_check.sh
```

## Pass/Fail Criteria
- All commands above exit with status `0`.
- No drift in generated artifacts after running checks.
- No parity regression failures in test logs under `reports/release_v1/`.

## Artifacts
- `reports/release_v1/pytest.full.out`
- `reports/release_v1/pytest.mody_core.out`
- `reports/release_v1/pytest.mody_gibbs.out`
- `reports/release_v1/pytest.huge_cache.out`
- `reports/release_v1/pytest.huge_gwas.out`
- matching `*.metrics.json` files with runtime/memory metrics
- matching `*.err` files for stderr capture
