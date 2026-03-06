#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
if [[ -n "${PYTHON_CMD:-}" ]]; then
  PYTHON_BIN="$PYTHON_CMD"
elif [[ -x "$REPO_ROOT/../.venv/bin/python" ]]; then
  PYTHON_BIN="$REPO_ROOT/../.venv/bin/python"
else
  PYTHON_BIN="$REPO_ROOT/../../.venv/bin/python"
fi
OUT_DIR="$REPO_ROOT/tests/data/reference/eaggl_factor_workflow_effective_config"

mkdir -p "$OUT_DIR"

run_case() {
  local name="$1"
  shift
  PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}" \
    "$PYTHON_BIN" -m eaggl factor --hide-opts --deterministic --print-effective-config "$@" > "$OUT_DIR/${name}.json"
}

run_case "F1_default_single_pheno"
run_case "F2_gene_list_like" --positive-controls-list INS
run_case "F3_projection_from_phewas" --gene-phewas-stats-in dummy_gene_phewas.tsv
run_case "F4_multi_pheno_anchor" \
  --anchor-phenos T2D,T2D_ALT \
  --gene-phewas-stats-in dummy_gene_phewas.tsv \
  --gene-set-phewas-stats-in dummy_gene_set_phewas.tsv
run_case "F5_any_pheno_anchor" \
  --anchor-any-pheno \
  --gene-phewas-stats-in dummy_gene_phewas.tsv \
  --gene-set-phewas-stats-in dummy_gene_set_phewas.tsv
run_case "F6_single_gene_anchor" \
  --anchor-gene INS \
  --gene-phewas-stats-in dummy_gene_phewas.tsv \
  --gene-set-phewas-stats-in dummy_gene_set_phewas.tsv
run_case "F7_multi_gene_anchor" \
  --anchor-genes INS,GCK \
  --gene-phewas-stats-in dummy_gene_phewas.tsv \
  --gene-set-phewas-stats-in dummy_gene_set_phewas.tsv
run_case "F8_any_gene_anchor" \
  --anchor-any-gene \
  --gene-phewas-stats-in dummy_gene_phewas.tsv \
  --gene-set-phewas-stats-in dummy_gene_set_phewas.tsv
run_case "F9_gene_set_anchor" \
  --anchor-gene-set \
  --run-phewas-from-gene-phewas-stats-in dummy_gene_phewas.tsv

cat > "$OUT_DIR/README.md" <<'EOF'
# Factor Workflow Effective Config Baselines

These JSON files are deterministic snapshots (`--print-effective-config`) for EAGGL factor workflow classification.

Regenerate with:

```bash
./scripts/eaggl/freeze_factor_workflow_effective_configs.sh
```
EOF

echo "Wrote workflow effective-config baselines to $OUT_DIR"
