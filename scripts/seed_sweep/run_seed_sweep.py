#!/usr/bin/env python
"""Run PIGEAN under several random seeds and aggregate the results.

    # run 3 seeds in 3 workers, then aggregate
    python scripts/seed_sweep/run_seed_sweep.py all \
      --config scripts/seed_sweep/configs/t2d_mouse_msigdb.json \
      --out-dir results/seed_sweep/t2d_mouse_msigdb \
      --seeds 0,1,2 --strategy parallel --workers 3

    # same seeds, one at a time
    ... all --strategy sequential

    # re-aggregate an existing sweep without re-running PIGEAN
    python scripts/seed_sweep/run_seed_sweep.py aggregate \
      --out-dir results/seed_sweep/t2d_mouse_msigdb

Subcommands: ``run`` (seeds only), ``aggregate`` (tables only), ``all`` (both).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import aggregate as agg  # noqa: E402
import runner  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]


def _parse_seeds(text: str) -> list[int]:
    """Accept ``0,1,2`` or an inclusive range ``0-9``."""
    seeds: list[int] = []
    for chunk in text.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "-" in chunk.lstrip("-"):
            lo, hi = chunk.split("-", 1)
            seeds.extend(range(int(lo), int(hi) + 1))
        else:
            seeds.append(int(chunk))
    return seeds


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    def add_common(p):
        p.add_argument("--out-dir", required=True, type=Path, help="sweep output root")

    def add_run_args(p):
        p.add_argument("--config", required=True, type=Path, help="sweep config JSON")
        p.add_argument("--seeds", help="seed list, e.g. 0,1,2 or 0-9 (overrides the config)")
        p.add_argument("--num-seeds", type=int, help="use seeds seed-start..seed-start+N-1")
        p.add_argument("--seed-start", type=int, default=0)
        p.add_argument(
            "--strategy",
            choices=("sequential", "parallel"),
            default="parallel",
            help="run seeds one at a time or concurrently (default: parallel)",
        )
        p.add_argument("--workers", type=int, default=3, help="max concurrent runs when parallel")
        p.add_argument(
            "--threads-per-worker",
            type=int,
            default=1,
            help="BLAS/OMP threads per run; 0 leaves the environment untouched",
        )
        p.add_argument("--python", help="interpreter for the PIGEAN subprocesses (default: this one)")
        p.add_argument("--resume", action="store_true", help="skip seeds that already exited 0")
        p.add_argument("--dry-run", action="store_true", help="print commands without running them")
        p.add_argument(
            "--stream",
            action="store_true",
            help="interleave every worker's log on the console, each line tagged [seed N]; "
                 "the full log is still written to <run_dir>/run.log",
        )
        p.add_argument(
            "--stream-grep",
            metavar="REGEX",
            help="with --stream, only echo matching lines (the file still gets everything). "
                 "Try --stream-grep 'epoch|chain|Writing|restart' for a progress-only view.",
        )

    def add_agg_args(p):
        p.add_argument(
            "--tables",
            help="comma-separated subset of: %s" % ",".join(agg.TABLE_SPECS),
        )
        p.add_argument(
            "--stats",
            default=",".join(agg.DEFAULT_STATS),
            help="per-metric stats to emit, from: %s" % ",".join(agg.ALL_STATS),
        )
        p.add_argument(
            "--min-runs",
            type=int,
            default=1,
            help="drop ids seen in fewer than this many runs (default: keep all)",
        )

    p_run = sub.add_parser("run", help="run the seeded PIGEAN jobs only")
    add_common(p_run)
    add_run_args(p_run)

    p_agg = sub.add_parser("aggregate", help="aggregate an existing sweep directory")
    add_common(p_agg)
    add_agg_args(p_agg)

    p_all = sub.add_parser("all", help="run the seeds, then aggregate")
    add_common(p_all)
    add_run_args(p_all)
    add_agg_args(p_all)

    return parser


def _resolve_seeds(args, config) -> list[int]:
    if args.seeds:
        return _parse_seeds(args.seeds)
    if args.num_seeds:
        return list(range(args.seed_start, args.seed_start + args.num_seeds))
    if config.seeds:
        return config.seeds
    raise SystemExit("no seeds: pass --seeds/--num-seeds or set 'seeds' in the config")


def _do_run(args) -> None:
    config = runner.SweepConfig.load(args.config)
    config.seeds = _resolve_seeds(args, config)
    runner.run_sweep(
        config,
        args.out_dir,
        strategy=args.strategy,
        workers=args.workers,
        python=args.python,
        repo_root=REPO_ROOT,
        threads_per_worker=args.threads_per_worker,
        resume=args.resume,
        dry_run=args.dry_run,
        stream=args.stream,
        stream_grep=args.stream_grep,
    )


def _do_aggregate(args) -> None:
    tables = args.tables.split(",") if args.tables else None
    stats = tuple(s.strip() for s in args.stats.split(",") if s.strip())
    unknown = [s for s in stats if s not in agg.ALL_STATS]
    if unknown:
        raise SystemExit("unknown --stats %s; known: %s" % (unknown, ",".join(agg.ALL_STATS)))
    agg.aggregate_sweep(args.out_dir, tables=tables, stats=stats, min_runs=args.min_runs)


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.command in ("run", "all"):
        _do_run(args)
    if args.command in ("aggregate", "all"):
        if getattr(args, "dry_run", False):
            print("dry run: skipping aggregation")
            return 0
        _do_aggregate(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
