# PIGEAN v1 Release Status

Date: 2026-03-05

## Checklist Gates
- `./scripts/check_shared_utils_sync.py`: PASS
- `../../.venv/bin/python scripts/generate_cli_manifest.py --check`: PASS
- `./scripts/release_readiness_check.sh`: PASS

## Test/Regression Summary
- Full suite: PASS (`147 passed`)
- MODY core regression suite: PASS
- MODY Gibbs regression suite: PASS
- HuGE cache regression suite: PASS
- HuGE GWAS regression suite: PASS

## Runtime/Memory (from `reports/release_v1/*.metrics.json`)
- `pytest.full`: 119.38s, 1734672 KB max RSS
- `pytest.mody_core`: 9.62s, 319072 KB max RSS
- `pytest.mody_gibbs`: 6.52s, 249136 KB max RSS
- `pytest.huge_cache`: 5.49s, 134752 KB max RSS
- `pytest.huge_gwas`: 45.84s, 1449472 KB max RSS

## Notes
- This status is tied to current `main` HEAD after N10 hardening commits.
- Raw logs and metrics are under `reports/release_v1/`.
