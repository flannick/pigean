"""PIGEAN results portal: threshold PIGEAN outputs into SQLite and serve a lightweight viewer.

    PYTHONPATH=src python -m pigean.portal build --db results/portal.sqlite \\
        --run t2d:results/t2d/pigean --gene-filter 'prior>1' --gene-filter 'log_bf>1' \\
        --gene-set-filter 'beta>0.01'

    PYTHONPATH=src python -m pigean.portal serve --db results/portal.sqlite --port 8765

See docs/PIGEAN_PORTAL.md. Standard library only; the UI loads Plotly from a CDN unless
`serve --plotly-js` points at a local copy.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from . import portal_db
from .portal_db import BuildOptions, RunFiles, parse_filter, resolve_run_dir
from .portal_server import serve

DEFAULT_PLOTLY_CDN = "https://cdn.plot.ly/plotly-2.35.2.min.js"


def _parse_run_spec(value: str) -> tuple[str, Path]:
    if ":" not in value:
        raise argparse.ArgumentTypeError(f"--run expects RUN_ID:DIR, got '{value}'")
    run_id, directory = value.split(":", 1)
    if not run_id:
        raise argparse.ArgumentTypeError(f"--run has empty RUN_ID: '{value}'")
    return run_id, Path(directory)


def _parse_run_files_spec(value: str) -> RunFiles:
    """RUN_ID:gene_stats=PATH,gene_set_stats=PATH[,gene_gene_set_stats=PATH]"""
    if ":" not in value:
        raise argparse.ArgumentTypeError(f"--run-files expects RUN_ID:key=PATH,..., got '{value}'")
    run_id, rest = value.split(":", 1)
    parts: dict[str, Path] = {}
    for item in rest.split(","):
        if "=" not in item:
            raise argparse.ArgumentTypeError(f"--run-files item '{item}' must be key=PATH")
        key, path = item.split("=", 1)
        parts[key.strip()] = Path(path.strip())
    for required in ("gene_stats", "gene_set_stats"):
        if required not in parts:
            raise argparse.ArgumentTypeError(f"--run-files for '{run_id}' is missing {required}=PATH")
    unknown = set(parts) - {"gene_stats", "gene_set_stats", "gene_gene_set_stats"}
    if unknown:
        raise argparse.ArgumentTypeError(f"--run-files for '{run_id}' has unknown keys: {sorted(unknown)}")
    files = RunFiles(run_id=run_id, gene_stats=parts["gene_stats"], gene_set_stats=parts["gene_set_stats"],
                     gene_gene_set_stats=parts.get("gene_gene_set_stats"))
    if files.gene_gene_set_stats is None:
        files.warnings.append(f"run '{run_id}': no gene_gene_set_stats given; loadings will be empty")
    return files


def _parse_title(value: str) -> tuple[str, str]:
    if ":" not in value:
        raise argparse.ArgumentTypeError(f"--run-title expects RUN_ID:TITLE, got '{value}'")
    run_id, title = value.split(":", 1)
    return run_id, title


def _filter_arg(value: str):
    try:
        return parse_filter(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m pigean.portal", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    build = sub.add_parser("build", help="Threshold PIGEAN outputs into a SQLite database")
    build.add_argument("--db", required=True, type=Path, help="SQLite file to write")
    build.add_argument("--run", action="append", default=[], type=_parse_run_spec, metavar="RUN_ID:DIR",
                       help="Run directory containing gene_stats / gene_set_stats / gene_gene_set_stats tables "
                            "(dashboard-style pigean.*.out.gz or LAP-style *.gene_stats.tsv names). Repeatable.")
    build.add_argument("--run-files", action="append", default=[], type=_parse_run_files_spec,
                       metavar="RUN_ID:gene_stats=PATH,gene_set_stats=PATH[,gene_gene_set_stats=PATH]",
                       help="Explicit file paths for one run. Repeatable.")
    build.add_argument("--run-title", action="append", default=[], type=_parse_title, metavar="RUN_ID:TITLE")
    build.add_argument("--gene-filter", action="append", default=[], type=_filter_arg, metavar="EXPR",
                       help="Gene threshold such as 'prior>1' or 'log_bf>=1'. Repeatable.")
    build.add_argument("--gene-set-filter", action="append", default=[], type=_filter_arg, metavar="EXPR",
                       help="Gene-set threshold such as 'beta>0.01' or 'beta_uncorrected>1'. Repeatable.")
    build.add_argument("--loading-filter", action="append", default=[], type=_filter_arg, metavar="EXPR",
                       help="Gene x gene-set loading threshold such as 'weight>0'. Repeatable.")
    build.add_argument("--filter-mode", choices=["any", "all"], default="any",
                       help="How repeated filters on one table combine (default: any = keep a row if at least "
                            "one threshold passes)")
    build.add_argument("--keep-all-loadings", action="store_true",
                       help="Keep loadings for every gene set, not only those that passed --gene-set-filter")
    build.add_argument("--append", action="store_true", help="Add/replace runs in an existing database")

    run = sub.add_parser("serve", help="Serve the portal UI + JSON API for a built database")
    run.add_argument("--db", required=True, type=Path)
    run.add_argument("--host", default="127.0.0.1")
    run.add_argument("--port", type=int, default=8765)
    run.add_argument("--title", default="PIGEAN results portal")
    run.add_argument("--plotly-js", type=Path, default=None,
                     help="Local plotly.min.js to embed instead of loading from the CDN (offline use)")
    return parser


def _collect_runs(args: argparse.Namespace) -> list[RunFiles]:
    titles = dict(args.run_title)
    runs: list[RunFiles] = []
    for run_id, directory in args.run:
        runs.append(resolve_run_dir(run_id, directory))
    runs.extend(args.run_files)
    seen: set[str] = set()
    for files in runs:
        if files.run_id in seen:
            raise ValueError(f"duplicate run id '{files.run_id}'")
        seen.add(files.run_id)
        files.title = titles.get(files.run_id, files.run_id)
        for path in (files.gene_stats, files.gene_set_stats, files.gene_gene_set_stats):
            if path is not None and not path.exists():
                raise FileNotFoundError(f"run '{files.run_id}': {path} does not exist")
    return runs


def run_build(args: argparse.Namespace) -> int:
    runs = _collect_runs(args)
    if not runs:
        logging.error("no runs supplied; use --run RUN_ID:DIR or --run-files")
        return 2
    options = BuildOptions(
        gene_filters=args.gene_filter, gene_set_filters=args.gene_set_filter, loading_filters=args.loading_filter,
        filter_mode=args.filter_mode, append=args.append, restrict_loadings_to_gene_sets=not args.keep_all_loadings,
    )
    summaries = portal_db.build_database(args.db, runs, options)
    for summary in summaries:
        logging.info(
            "run %s: genes %d/%d, gene sets %d/%d, loadings %d/%d", summary["run_id"],
            summary["n_genes"], summary["n_genes_input"], summary["n_gene_sets"], summary["n_gene_sets_input"],
            summary["n_loadings"], summary["n_loadings_input"],
        )
        for warning in summary["warnings"]:
            logging.warning(warning)
    logging.info("wrote %s", args.db)
    return 0


def run_serve(args: argparse.Namespace) -> int:
    if args.plotly_js is not None:
        plotly_src = "data:text/javascript;base64," + _b64(args.plotly_js)
    else:
        plotly_src = DEFAULT_PLOTLY_CDN
    serve(args.db, host=args.host, port=args.port, title=args.title, plotly_src=plotly_src)
    return 0


def _b64(path: Path) -> str:
    import base64

    return base64.b64encode(path.read_bytes()).decode("ascii")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    try:
        if args.command == "build":
            return run_build(args)
        return run_serve(args)
    except (FileNotFoundError, ValueError) as exc:
        logging.error(str(exc))
        return 1


if __name__ == "__main__":
    sys.exit(main())
