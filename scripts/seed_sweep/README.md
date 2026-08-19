# PIGEAN seed-replication sweeps

Run one PIGEAN command under several random seeds, then fold the per-seed stats
tables into a single table carrying a **mean, a standard deviation and a rank
spread for every score**. That gives consensus rankings of genes and gene sets
instead of whichever ranking one seed happened to produce, and it makes
run-to-run instability measurable rather than anecdotal.

Everything here is local. No Akleao pipeline, no Dock workflow.

## Quick start

```bash
python scripts/seed_sweep/run_seed_sweep.py all \
  --config scripts/seed_sweep/configs/t2d_mouse_msigdb.json \
  --out-dir results/seed_sweep/t2d_mouse_msigdb \
  --seeds 0,1,2 --strategy parallel --workers 3
```

Subcommands:

| command | does |
|---|---|
| `run` | the seeded PIGEAN jobs only |
| `aggregate` | rebuild the cross-seed tables from an existing sweep directory |
| `all` | both |
| `compare` | put several finished sweeps side by side |

## Ablation arms

`--set FLAG=VALUE` (repeatable) overrides or adds one PIGEAN flag on top of the
config, so an arm is an override rather than a near-duplicate config file —
which matters, because every line the two arms share has to stay identical for
the comparison to mean anything. `true`/`false` give a bare switch, `null`
removes a flag the config set, and a repeated key accumulates into a list.
Overrides are recorded in the sweep's `manifest.json`, so a directory says
which arm it is without the command line.

```bash
python scripts/seed_sweep/run_seed_sweep.py all \
  --config scripts/seed_sweep/configs/t2d_bottomline_mouse_msigdb.json \
  --out-dir results/seed_sweep/t2d_bl_strict \
  --cache-dir results/seed_sweep/_cache \
  --seeds 0,1,2 --set strict-stopping=true
```

Then read the arms against each other:

```bash
python scripts/seed_sweep/run_seed_sweep.py compare \
  --sweeps baseline=results/seed_sweep/t2d_bottomline_mouse_msigdb \
           pinned=results/seed_sweep/t2d_bl_pinned \
           strict=results/seed_sweep/t2d_bl_strict \
  --out results/seed_sweep/arm_comparison.tsv
```

`compare` leads with the **Gibbs sampling budget** across seeds, because that
is the check that says whether a knob did what it claimed:

```
Gibbs sampling budget across seeds (min..max; 'pinned' = identical in every seed)
                      baseline    no_stall    pinned
epochs                1..3        0           0
iters                 120..268    492..500    499
chains                20..39      10          10
verdict               varies      varies      pinned
```

An arm that silently failed to pin the budget would otherwise be read as
evidence about the model.

## Turning off Gibbs early stopping

PIGEAN stops each seed at a different point, so seeds differ in *how much
sampling they got*, not only in which draws they got — that is the mechanism
behind most of the run-to-run spread this harness measures. Three exits have to
be closed, and `--disable-stall-detection` only closes the first two:

| exit | closed by |
|---|---|
| stall detectors + restart epochs | `--disable-stall-detection` (zeroes the stall windows, sets `max_num_restarts=0`) |
| MCSE / R-hat stop (`_GIBBS_STOPPING_PRESETS`, `cli.py`) | `--min-num-post-burn-in` == `--max-num-post-burn-in` |
| adaptive burn-in, free within `burn=[10,400]` | `--min-num-burn-in` == `--max-num-burn-in` |

So the fully pinned arm is:

```bash
  --set disable-stall-detection=true --set max-num-iter=500 \
  --set min-num-burn-in=100      --set max-num-burn-in=100 \
  --set min-num-post-burn-in=400 --set max-num-post-burn-in=400
```

Collapsing each phase's min onto its max leaves no room to stop early. Confirm
it worked with `compare` — the verdict must read `pinned`.

This pins the *budget*, not convergence. It is the right diagnostic for
attributing spread to the stopping rule, and not necessarily the right
production setting; `--strict-stopping` (tighter thresholds, machinery intact)
and a higher `--num-chains` are the candidate shippable fixes.

Useful flags: `--strategy sequential` (one run at a time), `--workers N`,
`--seeds 0-9` (inclusive range), `--num-seeds 10`, `--resume` (skip seeds that
already exited 0), `--dry-run`, `--threads-per-worker N`.

## Watching the workers

`--stream` merges every worker's output into the console, one tagged line per
worker, while still writing each run's full log to `runs/seed_N/run.log`:

```bash
python scripts/seed_sweep/run_seed_sweep.py run \
  --config scripts/seed_sweep/configs/t2d_mouse_msigdb.json \
  --out-dir results/seed_sweep/t2d_mouse_msigdb \
  --seeds 0,1,2 --workers 3 \
  --stream --stream-grep 'Gibbs epoch [0-9]|Aggregated|Writing'
```

```
[seed 0] Gibbs epoch 1/11: max_num_iter=890, burn=[10,400], post=[10,490]
[seed 1] Gibbs epoch 1/11: max_num_iter=890, burn=[10,400], post=[10,490]
[seed 0] Completed Gibbs epoch 1/11 (iter=80, remaining_total_iter=810)
[seed 2] Completed Gibbs epoch 1/11 (iter=64, remaining_total_iter=826)
```

`--stream-grep` filters only what reaches the terminal; the log file always
gets every line, so grepping never costs you the record. Without a grep the raw
PIGEAN log is several megabytes per run and will bury the terminal — the
epoch/restart lines above are what you actually want to watch, because
epoch-count divergence between seeds is itself a reproducibility signal.

Streaming is output-identical to the non-streaming path (verified by `cmp`);
the only difference is that the runner tees the log itself instead of handing
PIGEAN a `--log-file`, since that flag diverts progress off stderr entirely and
would leave nothing to stream. Note the log lands at `run.log` (plain) rather
than `run.log.gz` in this mode.

## Layout of a sweep directory

```
<out-dir>/
  manifest.json                 # config, strategy, per-seed rc and wall time
  runs/seed_0/
    gene_stats.tsv.gz
    gene_set_stats.tsv.gz
    gene_gene_set_stats.tsv.gz
    params.tsv
    command.txt  status.json  stdout.txt  stderr.txt  run.log.gz
  runs/seed_1/ ...
  aggregate/
    gene_stats.seed_agg.tsv.gz
    gene_set_stats.seed_agg.tsv.gz
    gene_gene_set_stats.seed_agg.tsv.gz
    stability_summary.tsv
    stability_summary.json
```

## The aggregated table format

Key columns first (`Gene`, or `Gene_Set`, or `Gene`+`gene_set`), then:

| column | meaning |
|---|---|
| `n_runs`, `run_frequency` | how many seeds emitted this id at all |
| `consensus_rank`, `consensus_rank_sd` | mean and sd of the id's rank on the table's primary metric (`combined` for genes, `beta` for gene sets) |
| `primary_metric` | which metric `consensus_rank` was computed on |
| `<metric>_mean` | mean of the score across seeds |
| `<metric>_sd` | sample sd (ddof=1); `NA` when only one seed had it |
| `<metric>_cv` | `sd / abs(mean)` — scale-free spread |
| `<metric>_min`, `<metric>_max` | range, for spotting one outlier seed |
| `<metric>_n` | seeds contributing to this metric |
| `<metric>_rank_mean`, `<metric>_rank_sd` | mean and sd of the within-run rank (1 = highest) |

Rows are sorted by the primary metric's mean, descending, so the top of the
file *is* the consensus ranking.

Every numeric column in the source table gets the full set, so the exact
metric list follows whatever the run wrote — `combined`, `prior`, `log_bf`,
`huge_score_gwas`, `beta`, `beta_uncorrected`, `avg_postp`, and so on. Nothing
is hardcoded: columns are classified numeric vs. categorical by inspecting the
values, so `label` and `filter_reason` come through as the modal value plus a
`_n_values` count instead of being averaged.

Trim the width with `--stats mean,sd,n,rank_mean,rank_sd`, and drop unstable
ids entirely with `--min-runs 3`.

### Reading the three tables

* **`gene_stats`** — the main event. `combined_sd` and `combined_rank_sd` are
  the numbers to quote for gene-level reproducibility.
* **`gene_set_stats`** — note that most rows carry `beta = NA` because the run
  filtered them; the stability numbers for `beta` are therefore computed on the
  kept subset, and `n_runs` tells you whether a set was kept by every seed.
* **`gene_gene_set_stats`** — its `beta` is the *gene set's* beta repeated on
  every member row, not a pair-specific score, so per-pair score spread mostly
  restates the gene-set table. The informative quantity here is
  `run_frequency`: which (gene, gene set) pairs survive the write filter in
  every seed and which come and go.

## The stability summary

`aggregate/stability_summary.tsv` is the "did it replicate?" answer, in long
format (`table`, `metric`, `statistic`, `value`):

* `spearman[seed_a|seed_b]` and `spearman_mean`/`spearman_min` — rank agreement
  between each pair of runs over the ids they share.
* `jaccard_top_K_mean`/`_min` for K in 10/50/100/500 — would two seeds hand a
  reader the same top-K list. Ties at the K-th value are broken by id so the
  comparison is fair; `jaccard_top_K_boundary_tied` flags when the cut lands
  inside a block of equal values and the number is partly about the tie-break.
* `cv_median`/`cv_p90` and `rank_sd_median`/`rank_sd_p90` — the id-level spread
  distribution.
* `id_presence_stability` — fraction of ids emitted by *every* run.

Per-id sd answers "how noisy is this number". Top-K Jaccard answers "would a
reader get the same list". Both are needed; the second is usually much worse
than the first.

A headline table is printed to the console at the end of every aggregation.

## Sweep configs

A config is a JSON object naming the mode and the PIGEAN flags. Flag keys are
written without the leading `--` (underscores or dashes both work); a list
value repeats the flag; `true` makes it a bare switch. Relative paths are
resolved against the repo root, so configs stay portable.

```json
{
  "name": "t2d_mouse_msigdb",
  "mode": "gibbs",
  "seeds": [0, 1, 2],
  "outputs": ["gene_stats", "gene_set_stats", "gene_gene_set_stats", "params"],
  "args": {
    "X-in": ["bundles/.../gene_set_list_mouse_2024.txt",
             "bundles/.../gene_set_list_msigdb_nohp.txt"],
    "gwas-in": "tests/data/t2d_smoke/T2D.p_lt_1e-6.chrom_pos.sumstats.tsv.gz"
  }
}
```

Shipped configs:

| config | GWAS | libraries | cost |
|---|---|---|---|
| `t2d_bottomline_mouse_msigdb.json` | real bottom-line T2D from S3 (8.3 GB gz) | mouse_2024 + msigdb_nohp | ~1 h a seed |
| `t2d_mouse_msigdb.json` | repo-tracked T2D fixture | mouse_2024 + msigdb_nohp | ~70 s a seed |
| `t2d_mouse_only.json` | repo-tracked T2D fixture | mouse_2024 | ~30 s a seed |

The two fixture configs use
`tests/data/t2d_smoke/T2D.p_lt_1e-6.chrom_pos.sumstats.tsv.gz` (the P < 1e-6
subset of the portal T2D sumstats), so they run with nothing checked out beyond
this repo — good for iterating on the harness.

`t2d_bottomline_mouse_msigdb.json` is the production read. It reproduces the
command `dock/pigean-validation.workflow.yaml` runs for `model=mouse_msigdb`
under the `dock_current` param set — `--update-hyper none`, `--sigma-power 0`,
`--max-num-gene-sets 5000`, and the per-library `--p-noninf` priors from
`dock/config/models.yaml` — so a difference against a Dock run is the seed and
not the configuration.

### Remote inputs

Any config value may be an `http(s)` URL; PIGEAN reads those natively. The
harness downloads each one **once** before any seed starts and rewrites the
flag to the local copy:

```
fetching https://dig-open-bottom-line-analysis.s3.amazonaws.com/... (8.3 GB)
fetched T2D.sumstats.tsv.gz in 604s
```

This is not just a speed-up. Three seeds streaming the T2D sumstats would pull
25 GB and twenty would pull 166 GB, with every run gated on the network instead
of on PIGEAN; and fetching once removes any doubt that all seeds saw identical
input bytes, which is the whole premise of attributing differences to the seed.

`--cache-dir` picks the location (default `<out-dir>/_inputs`) — point several
sweeps at one directory to share the download. A complete prior copy is reused
after a Content-Length check; an interrupted fetch stays a `.part` file and is
never mistaken for a usable cache entry. `--stream-remote` opts out and lets
each seed pull its own copy.

## Two things the harness does on your behalf

**Both `--deterministic` and `--seed N`.** `--seed` alone only pins the legacy
global `random`/`numpy.random` state. Several pre-Gibbs paths — gene-set
batching, hyper subsampling — draw from an unseeded `default_rng`, and without
`--deterministic` those draws vary independently of the seed. Run-to-run spread
would then be a mixture of the seed effect and un-pinned sampling, which is not
the quantity anyone wants to report.

**`PYTHONHASHSEED=0`.** PIGEAN iterates over sets in a few places. Unpinned hash
randomization is a second uncontrolled source of run-to-run variation on top of
the seed.

## Checking the control

Before reading any spread as a seed effect, confirm a *same-seed* rerun is
identical — otherwise something in the stack is non-deterministic and the whole
comparison is confounded:

```bash
python scripts/seed_sweep/run_seed_sweep.py run \
  --config scripts/seed_sweep/configs/t2d_mouse_only.json \
  --out-dir results/seed_sweep/control --seeds 0,1,2 --workers 3

for s in 0 1 2; do
  cmp <(gzcat results/seed_sweep/t2d_mouse_only/runs/seed_$s/gene_stats.tsv.gz) \
      <(gzcat results/seed_sweep/control/runs/seed_$s/gene_stats.tsv.gz) \
    && echo "seed $s: identical"
done
```

## Sequential vs. parallel

The strategy changes wall-clock and peak memory only — per-run outputs are
identical either way, because each seed is its own `python -m pigean`
subprocess. Parallel workers are threads blocked in `subprocess.run`, and each
run's BLAS/OpenMP thread count is capped at `--threads-per-worker` (default 1)
so W workers do not each spawn N threads and thrash the box. Use
`--strategy sequential` when a single run is already using the whole machine.
