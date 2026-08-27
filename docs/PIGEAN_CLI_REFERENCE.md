# PIGEAN CLI Reference

This is the canonical human-written manual for the stable, routinely used `python -m pigean` command surface.

Use this document for practical command shapes and option semantics.
Use `docs/CLI_OPTIONS.md` for the exhaustive machine-generated parser inventory.
Use `docs/GIBBS_STOPPING.md` for the outer-Gibbs iteration budget, precision stopping, and optional stall/restart controller.
Use `README.md` for the full repository documentation map.
Use `docs/ADVANCED_SET_B.md` for retained advanced workflows such as HuGE cache I/O, precomputed stats ingestion, deprecated expert output PheWAS, PheWAS-as-Y beta sampling, and native multi-Y trait batching.
The gene-level PheWAS output stage (`--run-phewas`) is deprecated expert compatibility functionality. It regresses each phenotype's gene-level support surface against the current run's PIGEAN gene-level results, producing summaries such as `log_bf_vs_log_bf` and `combined_vs_combined`. It is not the same as native `--multi-y-in`, which treats each trait as a separate response and estimates gene-set beta outputs for the supplied gene-set matrix. Prefer native `--multi-y-in` for trait-wise PIGEAN beta workflows and EAGGL trait-linkage outputs for factor-to-trait summaries. If this legacy stage is used, the default is the matched comparison set; use `--phewas-comparison-set diagnostic` only when you explicitly want the cross-family diagnostics.
Optional downstream analyses use explicit `--run-*` booleans with separate `--*-in` / `--*-out` flags. Older hybrid flags remain compatibility aliases but are not the canonical documented surface.

Scope rules for this document:
- only stable user-facing options are included here
- every documented flag should have direct regression coverage
- niche, debug-only, or transitional flags belong in `docs/CLI_OPTIONS.md`, not here

## Entry points and modes

Primary entrypoint:

```bash
PYTHONPATH=src python -m pigean <mode> [...options]
```

Common modes:
- `gibbs`: full workflow
- `beta_tildes`: marginal gene-set effects only
- `betas`: joint gene-set effects without outer Gibbs
- `priors`: priors from precomputed or direct-support inputs
- `huge`: HuGE-only gene evidence / cache generation

Typical user workflow:

1. choose a config and input evidence sources
2. run `gibbs`, `betas`, `beta_tildes`, `priors`, or `huge`
3. inspect `gene_stats`, `gene_set_stats`, and `params`
4. optionally pass the resulting bundle or outputs into downstream EAGGL workflows

## Common command shapes

Full workflow from raw inputs:

```bash
mkdir -p results

PYTHONPATH=src python -m pigean gibbs \
  --config config/profiles/gwas.default.json \
  --X-in bundles/model_large-2026.02.22/data/gene_set_list_mouse_2024.txt \
  --gene-map-in bundles/model_large-2026.02.22/data/portal_gencode.gene.map \
  --gene-loc-file bundles/model_large-2026.02.22/data/NCBI37.3.plink.gene.loc \
  --gene-loc-file-huge bundles/model_large-2026.02.22/data/NCBI37.3.plink.gene.exons.loc \
  --gwas-in path/to/trait.sumstats.tsv.gz \
  --gwas-chrom-col CHROM \
  --gwas-pos-col POS \
  --gwas-p-col P \
  --gwas-n-col N \
  --gene-stats-out results/trait.gene_stats.out.gz \
  --gene-set-stats-out results/trait.gene_set_stats.out.gz \
  --pigean-rerun-bundle-out results/trait.pigean_rerun_bundle.tar.gz \
  --params-out results/trait.params.out.gz
```

`--params-out` is the resolved run record. It includes learned/internal quantities such as `p`, `sigma2`, Gibbs diagnostics, and other stage-specific outputs, and it also includes the resolved CLI/config state under `option_*` rows so the effective run settings can be reconstructed after the fact.

HuGE cache build:

```bash
PYTHONPATH=src python -m pigean huge \
  --gwas-in path/to/trait.sumstats.tsv.gz \
  --gwas-chrom-col CHROM \
  --gwas-pos-col POS \
  --gwas-p-col P \
  --gwas-n-col N \
  --gene-loc-file-huge bundles/model_large-2026.02.22/data/NCBI37.3.plink.gene.exons.loc \
  --huge-statistics-out results/trait.huge_statistics.tar.gz \
  --gene-stats-out results/trait.huge_gene_stats.out \
  --params-out results/trait.huge.params.out
```

Priors from precomputed gene stats:

```bash
PYTHONPATH=src python -m pigean priors \
  --X-in bundles/model_large-2026.02.22/data/gene_set_list_mouse_2024.txt \
  --gene-map-in bundles/model_large-2026.02.22/data/portal_gencode.gene.map \
  --gene-loc-file bundles/model_large-2026.02.22/data/NCBI37.3.plink.gene.loc \
  --gene-stats-in path/to/gene_stats.tsv \
  --gene-stats-id-col GENE \
  --gene-stats-log-bf-col log_bf \
  --gene-stats-combined-col combined \
  --gene-stats-prior-col prior \
  --gene-stats-out results/trait.gene_stats.out \
  --gene-set-stats-out results/trait.gene_set_stats.out \
  --params-out results/trait.params.out
```

## Documented option groups

### Runtime and reproducibility

| Flag | Meaning |
|---|---|
| `--config` | load a config profile before applying CLI overrides |
| `--deterministic` | force deterministic seeds and deterministic runtime behavior where supported |
| `--hide-opts` | suppress option echo in normal output |
| `--seed` | explicit RNG seed |
| `--debug-level` | increase debug logging |
| `--max-gb` | set memory budget used for batching heuristics |
| `--print-effective-config` | print the fully resolved config/options and exit |

### Outer-Gibbs stopping

The default `gibbs` controller runs one uninterrupted epoch for at most 500 outer iterations. It can still stop earlier when the across-chain precision checks pass. Stall-triggered epoch exits and restarts are disabled by default and are available with `--enable-stall-detection`.

The practical controls are summarized in `docs/GIBBS_STOPPING.md`. Use `--max-num-iter` to change the default one-epoch cap, the MCSE/R-hat flags to define acceptable precision, and `--strict-stopping` for the stricter bundled precision preset.

### Gene-set matrix inputs

| Flag | Meaning |
|---|---|
| `--X-in` | read one or more sparse gene-set files |
| `--X-list` | read a file listing sparse gene-set inputs |
| `--Xd-in` | read one or more dense gene-set files |
| `--Xd-list` | read a file listing dense gene-set inputs |
| `--gene-map-in` | map input gene identifiers onto the runtime gene space |
| `--gene-loc-file` | gene-location file for general runtime alignment |
| `--gene-loc-file-huge` | gene/exon location file for HuGE |
| `--exons-loc-file-huge` | explicit exon-location file for HuGE if separate from `--gene-loc-file-huge` |

### GWAS inputs

| Flag | Meaning |
|---|---|
| `--gwas-in` | GWAS summary-statistics input |
| `--gwas-chrom-col` | chromosome column |
| `--gwas-pos-col` | base-pair position column |
| `--gwas-p-col` | p-value column |
| `--gwas-beta-col` | effect-size column when present |
| `--gwas-se-col` | standard-error column when present |
| `--gwas-n-col` | sample-size column when present |

Notes:
- prefer `--gwas-se-col` when the file provides it
- if you provide `beta` without `se`, PIGEAN may need to infer z-scores conservatively from p-values instead

### Exome inputs

| Flag | Meaning |
|---|---|
| `--exomes-in` | exome burden-statistics input |
| `--exomes-gene-col` | gene column |
| `--exomes-p-col` | p-value column |
| `--exomes-beta-col` | effect-size column |
| `--exomes-se-col` | standard-error column |
| `--exomes-n-col` | sample-size column if `se` is not supplied |

Exome inputs must provide enough information to recover effect size scale: in practice use `p + beta + se`, or `p + beta + n`.

### Gene-list inputs

| Flag | Meaning |
|---|---|
| `--gene-list-in` | read gene-list inputs from a file |
| `--gene-list-id-col` | choose the gene ID column in `--gene-list-in` |
| `--gene-list-prob-col` | choose the probability column in `--gene-list-in` |
| `--gene-list-default-prob` | default probability to use when `--gene-list-prob-col` is absent |
| `--gene-list-no-header` | declare that `--gene-list-in` has no header row |
| `--gene-list` | comma-separated genes on the command line |
| `--gene-universe-in` | explicit gene universe file applied consistently across Y-input modes |
| `--gene-universe-id-col` | gene ID column in `--gene-universe-in` |
| `--gene-universe-no-header` | declare that `--gene-universe-in` has no header row |
| `--gene-universe-from-y` | use only genes present in the input Y source as the universe |
| `--gene-universe-from-x` | use all genes observed across `--X-in` / `--X-list` as the universe |

Important:
- `--gene-list` expects comma-separated gene symbols, not a file path
- GWAS / HuGE-derived inputs default to the HuGE gene list if no explicit universe is supplied, and this is recorded in the log
- other Y-input modes should provide one of `--gene-universe-in`, `--gene-universe-from-y`, or `--gene-universe-from-x`
- `--gene-list-all-*` and `--positive-controls-all-*` remain compatibility aliases for the canonical gene-universe surface

### Case/control burden-count inputs

| Flag | Meaning |
|---|---|
| `--case-counts-in` | case burden-count table |
| `--ctrl-counts-in` | control burden-count table |
| `--case-counts-max-freq-col` | optional max-frequency column for case filtering |
| `--ctrl-counts-max-freq-col` | optional max-frequency column for control filtering |

Count tables are expected to contain `gene`, `revel`, `count`, and `total`, plus optional max-frequency columns when those flags are used.

### Precomputed and cache workflows

| Flag | Meaning |
|---|---|
| `--gene-stats-in` | precomputed gene-level statistics input |
| `--gene-stats-id-col` | gene ID column |
| `--gene-stats-log-bf-col` | log-BF column |
| `--gene-stats-combined-col` | combined-score column |
| `--gene-stats-prior-col` | prior column |
| `--gene-stats-output-scope` | choose whether `gene_stats.out` writes the active universe only or the legacy expanded missing-gene view |
| `--huge-statistics-out` | write a HuGE cache tarball |
| `--huge-statistics-in` | read a HuGE cache tarball |
| `--pigean-rerun-bundle-out` | write a fixed-Y beta-stage rerun bundle for later annotation-exclusion sensitivity analyses |
| `--pigean-rerun-bundle-in` | load a fixed-Y beta-stage rerun bundle in `betas` mode |
| `--pigean-params-in` | replay beta-stage parameters from an existing PIGEAN `params.out` file in `betas` mode |
| `--gene-set-exclude-in` | remove listed annotations/gene sets before beta-tildes and joint betas are recomputed |
| `--gene-set-exclude-id-col` | ID column for `--gene-set-exclude-in`; default reads the first column |
| `--gene-set-exclude-no-header` | declare that `--gene-set-exclude-in` has no header row |

PIGEAN rerun bundles use schema `pigean_rerun_bundle/v1` and are distinct from EAGGL handoff bundles. A typical sensitivity workflow is:

```bash
PYTHONPATH=src python -m pigean betas \
  --pigean-rerun-bundle-in results/trait.pigean_rerun_bundle.tar.gz \
  --gene-set-exclude-in annotation_bridge_suggested_exclude.txt \
  --gene-set-stats-out results/trait.reduced.gene_set_stats.out.gz \
  --gene-stats-out results/trait.reduced.gene_stats.out.gz \
  --eaggl-bundle-out results/trait.reduced.eaggl_bundle.tar.gz \
  --params-out results/trait.reduced.params.out.gz
```

Rerun bundles replay the original active X matrix, active gene universe, fixed `combined` gene-score surface, learned hyperparameters, and beta-stage settings. They do not run outer Gibbs and they do not load the original gene-set statistics as beta input; gene-set beta-tildes and joint betas are recomputed. With no exclusion list, the rerun should closely match the original beta estimates, but exact equality is not guaranteed when the source run was a Gibbs posterior summary.

For older runs that did not write a rerun bundle, `--pigean-params-in previous.params.out` can replay beta-stage settings from the ordinary params file. The user must still provide the fixed Y input, gene universe mode, and X inputs. In `betas` mode, params replay uses the learned `p` and `sigma2` rows from the params file, maps vector values back onto loaded gene sets by annotation-library label, and defaults to `--update-hyper none` unless explicitly overridden.

### Expert beta-stage controls

| Flag | Meaning |
|---|---|
| `--retain-all-beta-uncorrected` | in pure `betas` runs, preserve independent `beta_uncorrected` values for gene sets dropped only by the expensive `--max-num-gene-sets` cap |
| `--independent-betas-only` | in pure `betas` runs, compute only independent `beta_uncorrected` and skip the covariance-backed corrected-beta solve |
| `--track-filtered-beta-uncorrected-mode` | in `betas`/`gibbs` runs, choose which ignored gene sets keep tracked `beta_uncorrected` sidecars: `none`, `all`, or `cap_only` |

Notes:
- These flags are aimed at large expanded-X seeded `betas` reruns where the raw regression stage is cheap but the final covariance-backed beta solve is what forces aggressive top-N truncation.
- `--retain-all-beta-uncorrected` keeps the expensive corrected `beta` path capped, but it still writes real independent `beta_uncorrected` values for capped-out rows in `gene_set_stats.out`.
- `--independent-betas-only` implies `--retain-all-beta-uncorrected`.
- `--track-filtered-beta-uncorrected-mode cap_only` is the default in `betas` and `gibbs`. It preserves tracked `beta_uncorrected` sidecars only for gene sets that were dropped by the expensive `--max-num-gene-sets` cap, not for rows filtered earlier by p-value or zero-beta prefilters.
- `--track-filtered-beta-uncorrected-mode all` restores the broader legacy behavior for every ignored row that makes it into the ignored sidecar pool.
- `--track-filtered-beta-uncorrected-mode none` disables tracked ignored-sidecar beta updates.
- `--retain-all-beta-uncorrected` and `--independent-betas-only` currently support only pure `betas` mode.
- `--track-filtered-beta-uncorrected-mode` currently supports `betas` and `gibbs`.

### Core filters and outputs

| Flag | Meaning |
|---|---|
| `--min-gene-set-size` | minimum retained gene-set size after runtime filtering |
| `--filter-gene-set-metric-z` | post-read gene-set QC metric threshold; values `<= 0` disable the QC-metric filter |
| `--filter-gene-set-p` | prefilter gene sets by association p-value |
| `--max-gene-set-read-p` | max p-value to keep in the initial read/beta stage |
| `--no-filter-negative` | keep negative beta-tilde gene sets rather than dropping them |
| `--gene-stats-out` | gene-level output table |
| `--gene-set-stats-out` | gene-set output table |
| `--max-no-write-gene-combined` | optional write-time filter for `gene_stats.out` based on absolute combined score |
| `--params-out` | params/diagnostics output table |

Notes:
- `--gene-stats-output-scope universe` is the default and writes only genes in the active analysis universe.
- `--gene-stats-output-scope current` restores the older behavior that also appends missing-gene rows after the active universe block.

## Testing expectations for this reference

This document is intentionally smaller than the full parser surface.

Testing tiers:
- toy tier:
  - uses `tests/data/t2d_smoke/gene_set_list_mouse_t2d_toy.txt`
  - exercises bundled GWAS + exomes + positive controls + case/control counts
  - uses `--filter-gene-set-metric-z 0` to disable QC-metric filtering on the intentionally tiny toy gene-set file
- validation tier:
  - uses the real mouse gene-set file from `bundles/model_large-2026.02.22/data/gene_set_list_mouse_2024.txt` when present
  - uses bundled GWAS without the toy QC override

Current reference tests:
- toy bundled-input path: `tests/test_t2d_toy_bundle_inputs_unittest.py`
- validation bundled-input path: `tests/test_t2d_validation_bundle_inputs_unittest.py`
- precomputed gene-stats path: `tests/test_mody_core_modes_regression_unittest.py`
- full Gibbs MODY regression path: `tests/test_mody_gibbs_regression_unittest.py`
- parser/config/error handling: `tests/test_pigean_cli_unittest.py`
- bundled real-GWAS HuGE regression: `tests/test_huge_real_gwas_regression_unittest.py`
