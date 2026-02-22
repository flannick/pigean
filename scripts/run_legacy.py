#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import subprocess
import sys
from pathlib import Path


def read_gene_list(path: Path) -> str:
    genes = []
    seen = set()
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            token = line.split()[0]
            if token not in seen:
                genes.append(token)
                seen.add(token)
    if not genes:
        raise ValueError(f"no genes found in {path}")
    return ",".join(genes)


def main() -> int:
    ap = argparse.ArgumentParser(description="Run legacy PIGEAN with profile + one input")
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--python", type=str, default=sys.executable)
    ap.add_argument("--legacy-script", type=Path, default=Path("legacy/priors.warm_stopping.py"))
    ap.add_argument("--out-dir", type=Path, default=Path("results"))
    ap.add_argument("--run-name", type=str, default=None)
    ap.add_argument("--gwas-in", type=Path, default=None)
    ap.add_argument("--exomes-in", type=Path, default=None)
    ap.add_argument("--huge-statistics-in", type=Path, default=None)
    ap.add_argument("--gene-list-in", type=Path, default=None)
    ap.add_argument("--dry-run", action="store_true", default=False)
    ap.add_argument("--extra-arg", action="append", default=[])
    args = ap.parse_args()

    inputs = [
        ("--gwas-in", args.gwas_in),
        ("--exomes-in", args.exomes_in),
        ("--huge-statistics-in", args.huge_statistics_in),
        ("--gene-list-in", args.gene_list_in),
    ]
    provided = [(k, v) for (k, v) in inputs if v is not None]
    if len(provided) != 1:
        raise SystemExit("provide exactly one of --gwas-in/--exomes-in/--huge-statistics-in/--gene-list-in")

    repo_root = Path(__file__).resolve().parents[1]
    config_path = args.config if args.config.is_absolute() else (repo_root / args.config)
    legacy_script = args.legacy_script if args.legacy_script.is_absolute() else (repo_root / args.legacy_script)

    if not config_path.exists():
        raise SystemExit(f"config not found: {config_path}")
    if not legacy_script.exists():
        raise SystemExit(f"legacy script not found: {legacy_script}")

    run_name = args.run_name or dt.datetime.now().strftime("PIGEAN_%Y%m%d_%H%M%S")
    out_dir = args.out_dir / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    stem = out_dir / run_name

    cmd = [
        args.python,
        str(legacy_script),
        "--config",
        str(config_path),
        "--gene-stats-out",
        str(stem.with_suffix(".gene_stats.out")),
        "--gene-set-stats-out",
        str(stem.with_suffix(".gene_set_stats.out")),
        "--phewas-stats-out",
        str(stem.with_suffix(".phewas_stats.out")),
        "--factors-out",
        str(stem.with_suffix(".factors.out")),
        "--gene-clusters-out",
        str(stem.with_suffix(".gene_clusters.out")),
        "--gene-set-clusters-out",
        str(stem.with_suffix(".gene_set_clusters.out")),
        "--pheno-clusters-out",
        str(stem.with_suffix(".pheno_clusters.out")),
        "--params-out",
        str(stem.with_suffix(".params.out")),
    ]

    input_flag, input_path = provided[0]
    if input_flag == "--gene-list-in":
        cmd.extend(["--positive-controls-list", read_gene_list(input_path)])
    else:
        cmd.extend([input_flag.replace("-in", "-in"), str(input_path)])

    for ea in args.extra_arg:
        cmd.append(ea)

    cmd_file = out_dir / f"{run_name}.cmd.txt"
    cmd_file.write_text(" ".join(cmd) + "\n", encoding="utf-8")

    log_file = out_dir / f"{run_name}.run.log"

    if args.dry_run:
        print("DRY-RUN")
        print("Command:")
        print(" ".join(cmd))
        print(f"Out dir: {out_dir}")
        return 0

    with log_file.open("w", encoding="utf-8") as lf:
        subprocess.run(cmd, check=True, stdout=lf, stderr=subprocess.STDOUT)

    print(f"Run complete: {out_dir}")
    print(f"Log: {log_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
