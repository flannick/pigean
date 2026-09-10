"""PIGEAN results portal: threshold PIGEAN outputs into SQLite and serve a lightweight viewer.

    PYTHONPATH=src python -m pigean.portal build --db results/portal.sqlite \\
        --run t2d:results/t2d/pigean --gene-filter 'prior>1' --gene-filter 'log_bf>1' \\
        --gene-set-filter 'beta>0.01'

    PYTHONPATH=src python -m pigean.portal serve --db results/portal.sqlite --port 8765

    PYTHONPATH=src python -m pigean.portal html --api-url http://localhost:8765 --out portal.html

`html` writes a static copy of the UI that talks to a running `serve` instance, for hosting
from a bucket or any static file server. Standard library only; the UI loads Plotly from a
CDN unless `--plotly-js` points at a local copy. See docs/PIGEAN_PORTAL.md.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from . import portal_db
from .portal_db import BuildOptions, RunFiles, parse_filter, resolve_run_dir
from .portal_assets import render_portal_html
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


def _parse_run_meta(value: str) -> tuple[str, dict]:
    """RUN_ID:model=NAME,trait=NAME,seed=N[,title=TEXT]"""
    if ":" not in value:
        raise argparse.ArgumentTypeError(f"--run-meta expects RUN_ID:key=value,..., got '{value}'")
    run_id, rest = value.split(":", 1)
    meta: dict = {}
    for item in rest.split(","):
        if "=" not in item:
            raise argparse.ArgumentTypeError(f"--run-meta item '{item}' must be key=value")
        key, val = item.split("=", 1)
        key = key.strip()
        if key not in ("model", "trait", "seed", "title"):
            raise argparse.ArgumentTypeError(f"--run-meta for '{run_id}' has unknown key '{key}'")
        meta[key] = val.strip()
    return run_id, meta


PACKAGE_KEYS = {"model", "model_title", "trait", "run", "title", "gene_stats", "gene_set_stats", "gene_gene_set_stats", "params"}


def _parse_package(value: str) -> dict:
    """model=NAME,trait=NAME,gene_stats=PATH,gene_set_stats=PATH[,gene_gene_set_stats=PATH][,params=PATH][,run=LABEL][,model_title=..][,title=..]"""
    spec: dict = {}
    for item in value.split(","):
        if "=" not in item:
            raise argparse.ArgumentTypeError(f"--package item '{item}' must be key=value")
        key, val = item.split("=", 1)
        key = key.strip()
        if key not in PACKAGE_KEYS:
            raise argparse.ArgumentTypeError(f"--package has unknown key '{key}' (allowed: {sorted(PACKAGE_KEYS)})")
        spec[key] = val.strip()
    for required in ("model", "trait", "gene_stats", "gene_set_stats"):
        if not spec.get(required):
            raise argparse.ArgumentTypeError(f"--package is missing {required}=...")
    return spec


def packages_to_runs(packages: list[dict]) -> list[RunFiles]:
    """
    Turn --package specs into RunFiles with constructed run ids.

    The run label within a (model, trait) pair is `run=` when given; otherwise "main" when the
    pair occurs once, or run1, run2, ... in input order when it occurs several times. The run id
    is `<model>__<trait>__<label>`.
    """
    counts: dict[tuple[str, str], int] = {}
    for spec in packages:
        key = (spec["model"], spec["trait"])
        counts[key] = counts.get(key, 0) + 1
    seen: dict[tuple[str, str], int] = {}
    runs: list[RunFiles] = []
    for spec in packages:
        key = (spec["model"], spec["trait"])
        seen[key] = seen.get(key, 0) + 1
        label = spec.get("run") or ("main" if counts[key] == 1 else f"run{seen[key]}")
        run_id = f"{spec['model']}__{spec['trait']}__{label}"
        files = RunFiles(
            run_id=run_id, gene_stats=Path(spec["gene_stats"]), gene_set_stats=Path(spec["gene_set_stats"]),
            gene_gene_set_stats=Path(spec["gene_gene_set_stats"]) if spec.get("gene_gene_set_stats") else None,
            params=Path(spec["params"]) if spec.get("params") else None,
            model=spec["model"], model_title=spec.get("model_title", ""), trait=spec["trait"], seed=label,
            title=spec.get("title") or (f"{spec['trait']} / {spec['model']}" + (f" / {label}" if label != "main" else "")),
        )
        if files.gene_gene_set_stats is None:
            files.warnings.append(f"run '{run_id}': no gene_gene_set_stats given; loadings will be empty")
        runs.append(files)
    return runs


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
    build.add_argument("--package", action="append", default=[], type=_parse_package,
                       metavar="model=NAME,trait=NAME,gene_stats=PATH,gene_set_stats=PATH[,gene_gene_set_stats=PATH][,params=PATH][,run=LABEL][,model_title=TEXT][,title=TEXT]",
                       help="One PIGEAN result package: the model it was run with, the trait, the result tables and "
                            "optionally the params file. Run id = <model>__<trait>__<run>; run defaults to 'main', or "
                            "run1, run2, ... when the same model/trait is given several times. Repeatable.")
    build.add_argument("--run-files", action="append", default=[], type=_parse_run_files_spec,
                       metavar="RUN_ID:gene_stats=PATH,gene_set_stats=PATH[,gene_gene_set_stats=PATH]",
                       help="Explicit file paths for one run. Repeatable.")
    build.add_argument("--run-title", action="append", default=[], type=_parse_title, metavar="RUN_ID:TITLE")
    build.add_argument("--run-meta", action="append", default=[], type=_parse_run_meta,
                       metavar="RUN_ID:model=NAME,trait=NAME,seed=N[,title=TEXT]",
                       help="Model / trait / seed labels for the portal's selectors. Inferred from run ids of the "
                            "form <model>__<trait>[__s<seed>] when omitted. Repeatable.")
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
    build.add_argument("--phenotype-file", type=Path, default=None,
                       help="dig-portal-data-models portal_phenotypes_flat.tsv: adds each trait's portal name, "
                            "portal id and ontology mappings (matched on the legacy phenotype id = run trait)")

    run = sub.add_parser("serve", help="Serve the portal UI + JSON API for a built database")
    run.add_argument("--db", required=True, type=Path)
    run.add_argument("--host", default="127.0.0.1")
    run.add_argument("--port", type=int, default=8765)
    run.add_argument("--title", default="PIGEAN Portal")
    run.add_argument("--plotly-js", type=Path, default=None,
                     help="Local plotly.min.js to embed instead of loading from the CDN (offline use)")
    run.add_argument("--cors-origin", default="*",
                     help="Access-Control-Allow-Origin value for the API (default '*'; empty disables CORS)")

    page = sub.add_parser("html", help="Write a static portal page that calls a running `serve` instance")
    page.add_argument("--api-url", required=True, help="Base URL of the portal server, e.g. http://localhost:8765")
    page.add_argument("--out", required=True, type=Path, help="HTML file to write")
    page.add_argument("--title", default="PIGEAN Portal")
    page.add_argument("--plotly-js", type=Path, default=None, help="Embed a local plotly.min.js instead of the CDN")
    page.add_argument("--db", type=Path, default=None,
                      help="Optional: the SQLite file this page is meant to browse; only checked for existence "
                           "(lets pipelines declare the database as a dependency of the page)")
    return parser


def _collect_runs(args: argparse.Namespace) -> list[RunFiles]:
    titles = dict(args.run_title)
    metas = dict(args.run_meta)
    runs: list[RunFiles] = packages_to_runs(args.package)
    legacy: list[RunFiles] = []
    for run_id, directory in args.run:
        legacy.append(resolve_run_dir(run_id, directory))
    legacy.extend(args.run_files)
    seen: set[str] = set()
    for files in legacy:
        meta = metas.get(files.run_id, {})
        files.model, files.trait, files.seed = meta.get("model", ""), meta.get("trait", ""), meta.get("seed", "")
        files.infer_metadata()
        files.title = titles.get(files.run_id) or meta.get("title") or files.run_id
    runs.extend(legacy)
    for files in runs:
        if files.run_id in seen:
            raise ValueError(f"duplicate run id '{files.run_id}'")
        seen.add(files.run_id)
        for path in (files.gene_stats, files.gene_set_stats, files.gene_gene_set_stats, files.params):
            if path is not None and not path.exists():
                raise FileNotFoundError(f"run '{files.run_id}': {path} does not exist")
    return runs


def run_build(args: argparse.Namespace) -> int:
    runs = _collect_runs(args)
    if not runs:
        logging.error("no runs supplied; use --package, --run RUN_ID:DIR or --run-files")
        return 2
    options = BuildOptions(
        gene_filters=args.gene_filter, gene_set_filters=args.gene_set_filter, loading_filters=args.loading_filter,
        filter_mode=args.filter_mode, append=args.append, restrict_loadings_to_gene_sets=not args.keep_all_loadings,
        phenotype_file=args.phenotype_file,
    )
    if args.phenotype_file is not None and not args.phenotype_file.exists():
        raise FileNotFoundError(f"phenotype file not found: {args.phenotype_file}")
    summaries = portal_db.build_database(args.db, runs, options)
    for summary in summaries:
        logging.info(
            "run %s: genes %d/%d, gene sets %d/%d, loadings %d/%d", summary["run_id"],
            summary["n_genes"], summary["n_genes_input"], summary["n_gene_sets"], summary["n_gene_sets_input"],
            summary["n_loadings"], summary["n_loadings_input"],
        )
        for warning in summary["warnings"]:
            logging.warning(warning)
        if "phenotypes" in summary:
            logging.info("phenotypes: %d traits, %d ontology mappings", summary["phenotypes"]["n_phenotypes"],
                         summary["phenotypes"]["n_mappings"])
    logging.info("wrote %s", args.db)
    return 0


def _plotly_src(plotly_js: Path | None) -> str:
    if plotly_js is not None:
        return "data:text/javascript;base64," + _b64(plotly_js)
    return DEFAULT_PLOTLY_CDN


def run_serve(args: argparse.Namespace) -> int:
    serve(args.db, host=args.host, port=args.port, title=args.title, plotly_src=_plotly_src(args.plotly_js),
          cors_origin=args.cors_origin)
    return 0


def run_html(args: argparse.Namespace) -> int:
    api_url = args.api_url.strip()
    if not api_url.startswith(("http://", "https://")):
        raise ValueError(f"--api-url must start with http:// or https://, got '{api_url}'")
    if args.db is not None and not args.db.exists():
        raise FileNotFoundError(f"database not found: {args.db}")
    page = render_portal_html(title=args.title, plotly_src=_plotly_src(args.plotly_js), api_base=api_url)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(page, encoding="utf-8")
    logging.info("wrote %s (API %s)", args.out, api_url)
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
        if args.command == "html":
            return run_html(args)
        return run_serve(args)
    except (FileNotFoundError, ValueError) as exc:
        logging.error(str(exc))
        return 1


if __name__ == "__main__":
    sys.exit(main())
