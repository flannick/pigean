# Data Artifacts

This directory contains small repo-local data artifacts used by documented example workflows.

## Cell-State Artifacts

`dat/cell_states/out/` contains copied `.gmt` and `.xlsx` cell-state marker outputs originally produced under:

```text
/Users/flannick/codex-workspace/analysis/cell_states/out/
```

The directory structure preserves the source tissue/QC subdirectories.

`dat/cell_states/pankbase_pancreas_reusable/by_cell_type/cycling_alpha/state_gmts/gmt/` contains the four GMT files used by the all-cell-state PIGEAN multi-Y example:

- `original_markers.gmt`
- `top_absolute_expression.gmt`
- `top_specific_fc.gmt`
- `top_specific_logp.gmt`

These were copied from:

```text
/Users/flannick/codex-workspace/analysis/cell_states/results/pankbase_pancreas_reusable/by_cell_type/cycling_alpha/state_gmts/gmt/
```

`dat/trait_blacklists/trait_blacklist_exomes_hp.txt` is the HPO/exomes trait blacklist used by the same example workflow. It was copied from the completed `analysis/blanc_screen` PIGEAN run directory.
