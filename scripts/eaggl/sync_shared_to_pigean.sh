#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
TARGET_EAGGL_ROOT="${1:-${REPO_ROOT}/../eaggl}"

echo "scripts/eaggl/sync_shared_to_pigean.sh is deprecated." >&2
echo "Use scripts/eaggl/export_standalone_eaggl.py to refresh the downstream standalone eaggl repo." >&2
exec "${REPO_ROOT}/../.venv/bin/python" "${REPO_ROOT}/scripts/eaggl/export_standalone_eaggl.py" "${TARGET_EAGGL_ROOT}"
