# PIGEAN

PIGEAN is the package-owned runtime for gene-set enrichment from GWAS, exomes, positive controls, and case/control burden counts. The active entrypoints are:

- `PYTHONPATH=src python -m pigean`
- `PYTHONPATH=src python -m eaggl`

The current code layout is:
- `src/`: active PIGEAN and EAGGL implementation
- `legacy/`: frozen historical scripts
- `config/profiles/`: reusable config profiles
- `scripts/`: fetch/build/release helpers
- `catalog/`: bundle catalogs
- `docs/`: user, architecture, release, and interoperability docs
- `tests/`: unit, regression, and bundled-fixture coverage

## Documentation Map

Use this as the index for the repo documentation set.

### Main entry docs

- `README.md`: project entrypoint, architecture summary, test tiers, and documentation index
- `docs/PIGEAN_CLI_REFERENCE.md`: human-written manual for how to run PIGEAN
- `docs/CLI_OPTIONS.md`: machine-generated exhaustive PIGEAN CLI inventory
- `docs/eaggl/CLI_OPTIONS.md`: machine-generated exhaustive EAGGL CLI inventory

### Scientific method docs

- `docs/methods.tex`: primary PIGEAN methods writeup
- `docs/eaggl/methods.tex`: primary EAGGL methods writeup
- `docs/pigean/METHODS_TO_CODE.md`: developer map from methods-level concepts to owning code modules

### User workflow and interoperability docs

- `docs/ADVANCED_SET_B.md`: supported advanced PIGEAN workflows, especially precomputed inputs and PheWAS-related paths
- `docs/EAGGL_INTEROP.md`: PIGEAN to EAGGL handoff bundle workflow
- `docs/eaggl/INTEROP.md`: EAGGL-specific interoperability notes
- `docs/eaggl/WORKFLOWS.md`: human-written EAGGL workflow guide
- `docs/eaggl/LABELING.md`: how optional EAGGL labeling works and why it remains integrated into `factor`
- `docs/pigean/README.md`: focused PIGEAN package notes for developers working inside `src/pigean/`
- `docs/eaggl/README.md`: focused EAGGL package notes for developers working inside `src/eaggl/`

### Architecture and transition docs

- `docs/CANONICAL_SOURCE.md`: canonical source-of-truth and active package architecture
- `docs/LEGACY_RETIREMENT_REPORT.md`: summary of the legacy-runtime retirement work
- `docs/eaggl/TRANSITION.md`: EAGGL transition and package-ownership notes
- `docs/eaggl/SHARED_CODE.md`: how EAGGL shares code with the main repo instead of maintaining a separate duplicated core
- `legacy/README.md`: what remains in the frozen legacy area and how it should be interpreted

### Setup, release, and repo operations docs

- `docs/REPO_BOOTSTRAP.md`: repository setup and bundle bootstrap steps
- `docs/BUNDLES.md`: bundle structure and handling
- `docs/CONFIGS.md`: config-profile structure and usage
- `docs/RELEASE_CHECKLIST.md`: PIGEAN release checklist
- `docs/RELEASE_STATUS.md`: PIGEAN release-status tracking notes
- `docs/eaggl/RELEASE_CHECKLIST.md`: EAGGL release checklist
- `docs/eaggl/RELEASE_STATUS.md`: EAGGL release-status tracking notes
- `docs/STITCHED_ARTIFACTS.md`: optional stitched single-file artifact generation

### Limitations and caveats

- `docs/KNOWN_LIMITATIONS.md`: known PIGEAN limitations
- `docs/eaggl/KNOWN_LIMITATIONS.md`: known EAGGL limitations

### Config and fixture docs

- `config/profiles/README.md`: profile layout and usage conventions
- `tests/data/t2d_smoke/README.md`: bundled toy/validation T2D fixtures used in repo tests
- `tests/data/reference/eaggl_factor_workflow_effective_config/README.md`: reference effective-config fixture used by EAGGL tests

## Quick start

Minimal setup:

1. Populate `catalog/bundles.json` from `catalog/bundles.example.json`.
2. Download the required bundles:

```bash
python scripts/fetch_bundles.py --catalog catalog/bundles.json --profile minimal --mode gene_list
```

3. Edit `config/profiles/common.factor.json` and replace `__BUNDLE_ROOT__` with your bundle root.
4. Then use `docs/PIGEAN_CLI_REFERENCE.md` for actual command shapes and workflow-specific flags.

Minimal example:

```bash
GENE_CSV=$(awk 'NF && $1 !~ /^#/ {print $1}' data/mody.gene.list | awk '!seen[$1]++' | paste -sd ',' -)

PYTHONPATH=src python -m pigean gibbs \
  --config config/profiles/gene_list.default.json \
  --positive-controls-list "$GENE_CSV" \
  --gene-stats-out results/MODY.gene_stats.out \
  --gene-set-stats-out results/MODY.gene_set_stats.out \
  --params-out results/MODY.params.out
```

For the practical run manual, use:
- `docs/PIGEAN_CLI_REFERENCE.md`

For the exhaustive generated parser surface, use:
- `docs/CLI_OPTIONS.md`
- `docs/cli_option_manifest.json`

## Test tiers

Two bundled T2D fixture tiers are available for repo-tracked testing:

- Toy tier:
  - inputs live under `tests/data/t2d_smoke/`
  - includes compact GWAS + exomes + MODY positive controls + synthetic case/control counts
  - uses `tests/data/t2d_smoke/gene_set_list_mouse_t2d_toy.txt`
  - intended for quick regression coverage of mixed input parsing and staged Y assembly
- Validation tier:
  - uses the same compact bundled GWAS fixture
  - uses the real mouse gene-set file `tests/data/model_small/gene_set_list_mouse_2024.txt`
  - intended for slower but more faithful gene-set validation without the toy QC-filter override

Run the toy tier:

```bash
cd pigean
PYTHONPATH=src ../../.venv/bin/python -m pytest \
  tests/test_t2d_toy_bundle_inputs_unittest.py
```

Run the validation tier:

```bash
cd pigean
PYTHONPATH=src ../../.venv/bin/python -m pytest \
  tests/test_t2d_validation_bundle_inputs_unittest.py \
  tests/test_huge_real_gwas_regression_unittest.py
```

Run the broader focused slice covering bundled fixtures plus MODY/CLI paths:

```bash
cd pigean
PYTHONPATH=src ../../.venv/bin/python -m pytest \
  tests/test_t2d_toy_bundle_inputs_unittest.py \
  tests/test_t2d_validation_bundle_inputs_unittest.py \
  tests/test_huge_real_gwas_regression_unittest.py \
  tests/test_mody_core_modes_regression_unittest.py \
  tests/test_mody_gibbs_regression_unittest.py \
  tests/test_pigean_cli_unittest.py
```

## Architecture summary

Current architecture:
- `src/pigean/app.py` and `src/eaggl/app.py` are the package-owned runtime entry modules
- `src/pigean/dispatch.py`, `src/pigean/pipeline.py`, `src/pigean/gibbs.py`, `src/pigean/huge.py`, and `src/pigean/model.py` own the stage-level PIGEAN flow
- `src/eaggl/dispatch.py`, `src/eaggl/factor.py`, `src/eaggl/phewas.py`, `src/eaggl/regression.py`, and `src/eaggl/io.py` own the stage-level EAGGL flow
- `src/pigean/main_support.py` and `src/eaggl/main_support.py` are narrow package-owned support layers for entry/runtime wiring
- `src/pigean/state.py` and `src/eaggl/state.py` are the remaining deep runtime-coupled modules and the canonical deep engines
- `src/pigean/state.py` is organized around:
  - `PhewasLabelState`
  - `GeneSetRegressionState`
  - `GeneSignalHugeState`
  - `ModelSummaryState`
- `src/eaggl/state.py` is organized around:
  - `PhewasPhenoState`
  - `GeneSetRegressionState`
  - `GeneSignalHugeState`
  - `FactorModelState`
- `src/pegs_utils.py` is no longer the catch-all owner for shared runtime behavior and continues to shrink toward a narrow transitional shim
