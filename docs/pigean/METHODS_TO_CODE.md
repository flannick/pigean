# PIGEAN Methods To Code

Purpose:
- help a developer start from `docs/methods.tex`
- find the primary module to edit for a methods-level change
- identify the regression tests that should move with that change

This is intentionally short. It is a stage-level ownership map, not a function-by-function inventory.

## How To Use This

1. Read the relevant section in `docs/methods.tex`.
2. Find the matching method area below.
3. Edit the package module listed under `Primary module` first.
4. Only drop below the stage modules if the change reaches deeper runtime-coupled logic in `src/pigean/state.py`.
5. Run the listed regression tests before widening to the full suite.

## Primary Edit Order

For most PIGEAN changes, start here in order:

1. `src/pigean/dispatch.py`
2. `src/pigean/pipeline.py`
3. one of:
   - `src/pigean/y_inputs.py`
   - `src/pigean/x_inputs.py`
   - `src/pigean/gibbs.py`
   - `src/pigean/huge.py`
   - `src/pigean/outputs.py`
   - `src/pigean/phewas.py`
4. `src/pigean/main_support.py` if the change is strictly entry/runtime wiring
5. `src/pegs_shared/types.py` / related shared modules if the data contract changes
6. `src/pigean/state.py` only for deeper runtime-coupled logic

## Methods Crosswalk

### 1. Trait-relevance model and gene evidence inputs

Methods area:
- gene-level evidence inputs
- direct genetic support
- combining GWAS, exomes, positive controls, case/control inputs

Primary module:
- `src/pigean/y_inputs.py`

Related lower-level module:
- `src/pigean/state.py`
  - deeper runtime-coupled readers and inner helpers still live here

Key state/contracts:
- `src/pegs_shared/types.py`
  - `YData`
  - `HyperparameterData`

Primary regressions:
- `tests/test_mody_core_modes_regression_unittest.py`
- `tests/test_huge_statistics_cache_regression_unittest.py`
- `tests/test_huge_real_gwas_regression_unittest.py`

### 2. Gene-set matrix ingest and filtering

Methods area:
- reading `X`
- gene-set filtering
- thresholding and pruning before inference

Primary module:
- `src/pigean/x_inputs.py`

Related lower-level module:
- `src/pigean/state.py`
  - matrix callbacks and deeper filtering internals still live here

Key state/contracts:
- `src/pegs_shared/types.py`
  - `XData`
- `src/pegs_shared/xdata.py`

Primary regressions:
- `tests/test_mody_core_modes_regression_unittest.py`
- `tests/test_shared_module_boundaries_unittest.py`

### 3. Initial beta-tilde / beta / priors stages

Methods area:
- stagewise PIGEAN pipeline before outer Gibbs
- beta-tilde estimation
- non-infinitesimal beta estimation
- priors construction

Primary module:
- `src/pigean/pipeline.py`

Related lower-level module:
- `src/pigean/state.py`
  - lower-level beta estimation math still lives here

Primary regressions:
- `tests/test_mody_core_modes_regression_unittest.py`
- `tests/test_set_b_smoke_unittest.py`

### 4. Outer Gibbs control and stopping/restart policy

Methods area:
- outer Gibbs epochs
- burn-in logic
- MCSE / R-hat stopping
- restart scheduling

Primary module:
- `src/pigean/gibbs.py`

Related lower-level module:
- `src/pigean/state.py`
  - inner beta update math and sampler internals still live here

Primary regressions:
- `tests/test_mody_gibbs_regression_unittest.py`
- `tests/test_shared_module_boundaries_unittest.py`
- `tests/test_gibbs_hyper_mutation_unittest.py`

### 5. HuGE score generation and cache I/O

Methods area:
- GWAS to HuGE preprocessing
- HuGE cache read/write
- HuGE correction and related input preparation

Primary module:
- `src/pigean/huge.py`

Related lower-level module:
- `src/pigean/state.py`
  - lower-level score generation details still live here

Primary regressions:
- `tests/test_huge_statistics_cache_regression_unittest.py`
- `tests/test_huge_real_gwas_regression_unittest.py`

### 6. Output writing and PheWAS stages

Methods area:
- gene stats / gene set stats outputs
- optional PheWAS stages
- output bundling for downstream tools

Primary modules:
- `src/pigean/outputs.py`
- `src/pigean/phewas.py`

Primary regressions:
- `tests/test_phewas_stage_reuse_unittest.py`
- `tests/test_set_b_smoke_unittest.py`

## Current Boundary To Remember

The package layout is now truthful for stage-level ownership.

The main remaining deep implementation file is:
- `src/pigean/state.py`

The main remaining runtime wiring/support module is:
- `src/pigean/main_support.py`

Use it when a change touches:
- low-level sampler math
- deep reader internals
- matrix callbacks still shared by the extracted orchestration modules
- runtime-owned helper methods on `PigeanState`

Use `src/pigean/main_support.py` when a change only affects:
- entry/runtime wiring
- CLI-to-runtime service setup
- package-level helper routing between stage modules and `PigeanState`

If a change only alters stage wiring, stopping policy, input contracts, or output routing, it should usually land in the package modules above instead.

## Minimal Update Checklist

When a methods-level change is made:

1. Update `docs/methods.tex` if the scientific method changed.
2. Update the owning package module first.
3. Update `src/pigean/main_support.py` only if runtime wiring changes.
4. Update `src/pigean/state.py` only if the change reaches deeper runtime-coupled internals.
5. Run the narrow regression tests for the touched stage.
6. Run the full suite before merge.
