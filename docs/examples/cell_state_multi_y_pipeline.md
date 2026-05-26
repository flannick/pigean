# Cell-State Multi-Y PIGEAN Pipeline

This example records the PIGEAN commands used for the all-cell-state full-universe run with HPO/exomes traits removed.

The original executed script was:

```text
/Users/flannick/codex-workspace/analysis/blanc_screen/results/pigean_all_cell_states_full_universe_default_no_hpo_exomes/run_pigean_all_cell_states_full_universe_default_no_hpo_exomes.sh
```

The repo-local version below uses artifacts copied into `dat/` and the bundled large PIGEAN data.

## Inputs

```bash
ROOT="$(pwd)"
PYTHON_BIN="../../.venv/bin/python"
PIGEAN_SRC="$ROOT/src"
BUNDLE="$ROOT/bundles/model_large-2026.02.22"
DATA="$BUNDLE/data"

MULTI_Y_IN="$DATA/all.gene_stats.large.gt1.out.gz"
GENE_UNIVERSE="$DATA/NCBI37.3.plink.gene.loc"
TRAIT_BLACKLIST="$ROOT/dat/trait_blacklists/trait_blacklist_exomes_hp.txt"
GMT_ROOT="$ROOT/dat/cell_states/pankbase_pancreas_reusable/by_cell_type/cycling_alpha/state_gmts/gmt"
OUT_ROOT="$ROOT/results/pigean_all_cell_states_full_universe_default_no_hpo_exomes"
```

## Commands

Run from the repository root (`pigean/`).

```bash
mkdir -p "$OUT_ROOT"

run_pigean() {
  local method="$1"
  local gmt="${GMT_ROOT}/${method}.gmt"
  local out_dir="${OUT_ROOT}/${method}"

  mkdir -p "${out_dir}"

  env PYTHONPATH="${PIGEAN_SRC}" \
  "${PYTHON_BIN}" -m pigean betas \
    --X-in "${gmt}" \
    --multi-y-in "${MULTI_Y_IN}" \
    --multi-y-id-col Gene \
    --multi-y-pheno-col Trait_Internal \
    --multi-y-log-bf-col Direct \
    --multi-y-combined-col Combined \
    --multi-y-prior-col Indirect \
    --multi-y-trait-blacklist-in "${TRAIT_BLACKLIST}" \
    --gene-universe-in "${GENE_UNIVERSE}" \
    --gene-universe-id-col 6 \
    --gene-universe-no-header \
    --gene-set-stats-out "${out_dir}/gene_set_stats.debug.out.gz" \
    --params-out "${out_dir}/params.out.gz" \
    --log-file "${out_dir}/run.log" \
    --warnings-file "${out_dir}/warnings.log" \
    --output-detail debug \
    --deterministic \
    --hide-progress \
    --min-gene-set-size 1 \
    --filter-gene-set-p 1 \
    --max-gene-set-read-p 1 \
    --no-filter-negative \
    --prune-gene-sets 1.1 \
    --weighted-prune-gene-sets 1.1
}

run_pigean original_markers
run_pigean top_absolute_expression
run_pigean top_specific_fc
run_pigean top_specific_logp
```

## Notes

- `--multi-y-in` uses the bundled large trait gene-statistics table.
- `--multi-y-trait-blacklist-in` removes traits listed in `dat/trait_blacklists/trait_blacklist_exomes_hp.txt`, including HPO and exomes traits.
- `--gene-universe-in` uses the full NCBI gene-location file and column 6 as the gene ID.
- The output paths intentionally use `.gz` for the large tables.
