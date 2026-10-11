# Independent feature experiments

The protected initial baseline is `ca12beeba4a80fb829695ec992718e05465c4e73`. The registry records active fPS and effect-priority work, already merged prerequisites, and abandoned experiments. Blockwise redesign and old debug experiments are abandoned and are never integration dependencies. Preserve their historical files.

Use a dedicated worktree per feature. Ports are siblings of the foundation; independent future inference features must not incorporate either port. Combine selected commits only in disposable integration worktrees. Local checkpoint commits are allowed. Push only exact commits with passing functional and benchmark evidence. Never automatically merge or change defaults.

## Runner

Invoke `scripts/experiments/bench.py` with the project's configured Python interpreter. Commands are `inventory`, `freeze`, `run`, `compare`, `test`, `integrate`, `verify`, `publication`, and `publish`; each provides `--help`. The runner requires clean committed source and frozen input hashes. Commands in JSON are argument arrays, not shell strings. Supported substitutions are `{python}`, `{repo}`, `{output}`, and `{seed}`.

Specifications contain `schema_version: 1`, a trait, a benchmark level, command array, an `inputs` mapping of names to path/SHA256 records, and expected output filenames. Freeze once to a new file; never overwrite a baseline. The supplied analysis-level recipes contain local paths and remain outside this repository. Include every command input in the input manifest.

Each run archives its runner and command, pins its source commit, verifies input hashes before and after execution, checks gzip integrity, and records wall time, peak child-process RSS, and summed rounded outer-iteration timers when present. Run one benchmark process at a time using the same `--lock` path. The host and thread counts must match. A completed run can resume only with identical identities. An interrupted/failed run requires a new output directory. Saved receipts remain valid only while source worktrees stay at their recorded commits; use detached worktrees for long-term references.

`test` writes a commit-bound pytest receipt. `integrate` creates a new worktree from an explicit base and merges only the listed feature refs. Conflicts leave that integration worktree for manual resolution. Original feature worktrees are not modified. Regenerate CLI manifests after resolving merges.

## Fixed-state inference

`scripts/experiments/inference.py capture --fixture FILE -- gibbs ...` intercepts the pre-Gibbs boundary, saves state, arguments, and RNG state, and exits before sampling. `replay --fixture FILE --out DIRECTORY --seed N` runs the recorded inference arguments and writes gene/set summaries. Use only trusted locally generated fixtures: these are Python pickle files. Include both fixture and its SHA256 sidecar in benchmark inputs. Never use a fixture from an untrusted source.

A fixed-state fixture isolates inference from stochastic setup. A full cached-input run includes loading and finalization, but does not benchmark raw-GWAS processing. Keep these claims separate. Current behavior remains explicitly selectable with `--gibbs-engine legacy`; unknown engines fail. Signal weighting and downstream fPS are independent switches.

## Comparisons and promotion

Compare `prior` by `Gene` and `beta_uncorrected` by `Gene_Set`, across the full finite universe and the union of the two top-200 absolute-value panels. Report Pearson, Spearman, through-zero slope, fitted slope/intercept, errors, finite counts, and universe differences. Never impute missing values. Seed aggregation requires consistent universes and finite values in every replicate.

Use seeds 0–4 for candidates, 5–9 for baseline repeatability, and three alternating fixed-seed baseline/candidate timing confirmations. Test T2D and NAFLD separately. Complete cached-input baseline controls use 10 outer chains, four inner chains, 500 iterations, and disabled early stopping; additionally test production defaults.

- `neutral`: identical numerical summaries and no >5% median runtime regression.
- `repeatability`: Pearson/Spearman >=0.99 and through-zero slope 0.95–1.05. This evaluates two independent baseline panels.
- `agreement`: same strict agreement gates, requiring a passing baseline repeatability receipt.
- `speed`: strict agreement plus >=10-fold median speedup, requiring baseline repeatability. Name the benchmark level explicitly.
- `inference`: always inconclusive in the generic comparator. A new statistical target requires additional preregistered reference/calibration criteria; do not automatically promote it from resemblance to an unconverged baseline.

An inconclusive or failed result is not a publishable improvement. A feature port has two obligations: preserve disabled behavior on current main and verify its enabled calculations with the preserved source and feature-specific tests. Intended statistical changes are not expected to match association-only output.

A publication manifest contains the exact commit, PASS status, both traits, `change_kind` (`foundation` or `speed`), and hashed test/comparison receipts. Statistical changes fail closed until their preregistered scientific acceptance gate is implemented. `publication` rechecks these before a normal feature-branch push. Check latest upstream separately and repeat integration validation if it changed; never force-push. Abandoned branches are excluded from combinations.

The four first-phase combinations are baseline, fPS, effect-priority, and both. Future inference experiments compare each combination with its matching legacy-engine control. Shared coverage includes cached HuGE correction/provenance, effector metadata, CLI options/defaults, output identifiers/schema, cache reload, numerical parity, fPS export/reload, and effect-weighting formulas.

## Executing a complete panel

`matrix.py` runs both traits serially, binds source and recipe identities, and resumes completed runs without replacing them. A phase can be `baseline`, `candidate`, `timing`, `compare`, or `all`. Each full matrix contains 10 baseline seeds, five candidate seeds, and six alternating timing runs per trait. Use a fresh output root when changing a recipe, runner, or commit. Keep test jobs and other analyses off the timing host during the timing phase.

```bash
"$PYTHON_CMD" scripts/experiments/matrix.py \
  --spec /absolute/project/config/experiments_20261011/T2D.json \
  --spec /absolute/project/config/experiments_20261011/NAFLD.json \
  --baseline-repo /absolute/project/worktrees/experiment-baseline \
  --candidate-repo /absolute/project/worktrees/experiment-foundation \
  --candidate-arg=--gibbs-engine --candidate-arg=legacy \
  --track neutral --phase all \
  --out /absolute/project/results/foundation_matrix \
  --lock /absolute/project/results/benchmark.lock
```

Repeat this at the `fixed_state_inference` level using replay specifications whose inputs include the captured fixture, its sidecar, and the replay driver. The confirmation gate uses recorded inference seconds at that level, and subprocess wall time for complete cached-input runs. Capture on the protected baseline, freeze the resulting files, and replay the same fixture in both source worktrees. Seed overrides intentionally reset inference RNGs after setup; they do not claim to reproduce a full run using the same seed.

The comparator writes an incomplete panel as INCONCLUSIVE. It verifies that repeatability refers to the same baseline and records its checksum. Neutral comparisons also require identical full gene and gene-set tables for each paired seed, including column names and structural missing values. Strict agreement rejects missing values and undefined correlations. Timing confirmations must be six distinct runs, in alternating serial order, at seed zero and the same source/specification as the numerical panels.

Publication additionally requires an `integration` entry with `repo`, `upstream_commit`, `tests`, and `tests_sha256`. The integration commit must contain both the candidate and recorded upstream, and its commit-bound test receipt must pass. `publication` checks against the locally fetched upstream reference; `publish` first refreshes upstream, verifies all evidence, and performs an ordinary push of the current `codex/` branch. Neither command changes main or enables experimental defaults. Statistical ports remain checkpoints until separately justified enabled-behavior evidence is available; neutral disabled-mode parity alone does not establish scientific improvement.
