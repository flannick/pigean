from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

from .dashboard_assets import render_html


@dataclass(frozen=True)
class PigeanRunSpec:
    run_id: str
    path: Path


@dataclass(frozen=True)
class EagglRunSpec:
    run_id: str
    mode_id: str
    path: Path
    group_id: str | None = None
    group_title: str | None = None


@dataclass(frozen=True)
class EagglPhiSweepSpec:
    run_id: str
    mode_id: str
    path: Path


@dataclass(frozen=True)
class EagglGroupSpec:
    run_id: str
    mode_id: str
    group_id: str
    group_title: str


def open_text(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8", newline="")
    return path.open("rt", encoding="utf-8", newline="")


def parse_float(value, default=None):
    if value in (None, "", "NA", "NaN", "nan", "None"):
        return default
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(number):
        return default
    return number


def read_tsv(path: Path, warnings: list[str]) -> list[dict[str, str]]:
    try:
        with open_text(path) as handle:
            return list(csv.DictReader(handle, delimiter="\t"))
    except FileNotFoundError:
        warnings.append(f"missing file: {path}")
    except OSError as exc:
        warnings.append(f"could not read {path}: {exc}")
    except csv.Error as exc:
        warnings.append(f"could not parse TSV {path}: {exc}")
    return []


def read_optional_text(path: Path, warnings: list[str], *, max_chars: int | None = 200000) -> str:
    if not path.exists():
        return ""
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        warnings.append(f"could not read {path}: {exc}")
        return ""
    if max_chars is not None and len(text) > max_chars:
        warnings.append(f"truncated embedded text from {path} to {max_chars} characters")
        return text[:max_chars]
    return text


def _first(row: dict[str, str], names: Iterable[str], default=""):
    for name in names:
        if name in row and row[name] not in (None, ""):
            return row[name]
    return default


def factor_columns(header: list[str] | None) -> list[str]:
    if not header:
        return []
    return [name for name in header if name.startswith("Factor") and not name.startswith(("Relative_", "Combined_"))]


def parse_run_spec(value: str) -> PigeanRunSpec:
    if ":" not in value:
        raise argparse.ArgumentTypeError("expected RUN_ID:DIR")
    run_id, path = value.split(":", 1)
    run_id = run_id.strip()
    if not run_id or not path:
        raise argparse.ArgumentTypeError("expected RUN_ID:DIR")
    return PigeanRunSpec(run_id=run_id, path=Path(path))


def parse_eaggl_spec(value: str) -> EagglRunSpec:
    parts = value.split(":", 2)
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("expected RUN_ID:MODE_ID:DIR")
    run_id, mode_id, path = [part.strip() for part in parts]
    if not run_id or not mode_id or not path:
        raise argparse.ArgumentTypeError("expected RUN_ID:MODE_ID:DIR")
    return EagglRunSpec(run_id=run_id, mode_id=mode_id, path=Path(path))


def parse_eaggl_phi_sweep_spec(value: str) -> EagglPhiSweepSpec:
    parts = value.split(":", 2)
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("expected RUN_ID:MODE_ID:DIR")
    run_id, mode_id, path = [part.strip() for part in parts]
    if not run_id or not mode_id or not path:
        raise argparse.ArgumentTypeError("expected RUN_ID:MODE_ID:DIR")
    return EagglPhiSweepSpec(run_id=run_id, mode_id=mode_id, path=Path(path))


def parse_eaggl_group_spec(value: str) -> EagglGroupSpec:
    parts = value.split(":", 3)
    if len(parts) not in {3, 4}:
        raise argparse.ArgumentTypeError("expected RUN_ID:MODE_ID:GROUP_ID[:GROUP_TITLE]")
    run_id, mode_id, group_id = [part.strip() for part in parts[:3]]
    group_title = parts[3].strip() if len(parts) == 4 else group_id.replace("_", " ")
    if not run_id or not mode_id or not group_id:
        raise argparse.ArgumentTypeError("expected RUN_ID:MODE_ID:GROUP_ID[:GROUP_TITLE]")
    return EagglGroupSpec(run_id=run_id, mode_id=mode_id, group_id=group_id, group_title=group_title or group_id)


def parse_run_title(value: str) -> tuple[str, str]:
    if ":" not in value:
        raise argparse.ArgumentTypeError("expected RUN_ID:TITLE")
    run_id, title = value.split(":", 1)
    if not run_id.strip() or not title.strip():
        raise argparse.ArgumentTypeError("expected RUN_ID:TITLE")
    return run_id.strip(), title.strip()


def _parse_phi_from_name(name: str):
    match = re.search(r"(?:^|[_-])phi[_-]?([0-9]+(?:p[0-9]+)?(?:\\.[0-9]+)?(?:e[-+]?[0-9]+)?)(?:$|[_-])", name, flags=re.IGNORECASE)
    if not match:
        match = re.search(r"([0-9]+p[0-9]+|[0-9]+\\.[0-9]+)", name)
    if not match:
        return None
    raw = match.group(1).replace("p", ".")
    try:
        return float(raw)
    except ValueError:
        return None


def _read_selected_phi_from_report(path: Path):
    if not path.exists():
        return None
    rows = read_tsv(path, [])
    for row in rows:
        selected = str(_first(row, ["selected", "is_selected"], "")).strip().lower()
        if selected in {"1", "true", "t", "yes", "y"}:
            return parse_float(_first(row, ["phi", "candidate_phi"]))
    return None


def _metric_value(value: str):
    number = parse_float(value)
    if number is not None:
        return number
    if value in (None, ""):
        return None
    return value


def _first_existing(path: Path, names: Iterable[str]) -> Path | None:
    for name in names:
        candidate = path / name
        if candidate.exists():
            return candidate
    return None


def read_factor_metrics(path: Path, warnings: list[str]) -> dict[str, dict]:
    if not path.exists():
        return {}
    rows = read_tsv(path, warnings)
    metrics: dict[str, dict] = {}
    for row in rows:
        factor = _first(row, ["Factor", "factor"])
        if not factor:
            continue
        metrics[factor] = {
            key: _metric_value(value)
            for key, value in row.items()
            if key not in {"Factor", "factor", "label", "Label"} and value not in (None, "")
        }
    return metrics


def read_selected_phi_metrics(path: Path, warnings: list[str]) -> dict:
    if not path.exists():
        return {}
    rows = read_tsv(path, warnings)
    if not rows:
        return {}
    selected = None
    for row in rows:
        flag = str(_first(row, ["selected", "is_selected"], "")).strip().lower()
        if flag in {"1", "true", "t", "yes", "y"}:
            selected = row
            break
    selected = selected or rows[0]
    return {
        key: _metric_value(value)
        for key, value in selected.items()
        if value not in (None, "")
    }


def _discover_phi_sweep_runs(spec: EagglPhiSweepSpec) -> list[EagglRunSpec]:
    root = spec.path
    candidates: list[tuple[float, Path]] = []
    if not root.exists():
        return [EagglRunSpec(spec.run_id, f"{spec.mode_id}_missing", root)]
    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        run_dir = child / "eaggl" if (child / "eaggl").is_dir() else child
        if not any((run_dir / name).exists() for name in ("factors.out.gz", "params.out.gz", "params.out", "eaggl.run.log")):
            continue
        phi = _parse_phi_from_name(child.name) or _parse_phi_from_name(run_dir.name)
        if phi is None:
            continue
        candidates.append((phi, run_dir))
    if not candidates and (root / "factors.out.gz").exists():
        phi = _parse_phi_from_name(root.name)
        candidates.append((float("nan") if phi is None else phi, root))
    selected_phi = None
    for report_name in ("learn_phi_report.tsv", "learn_phi_report.out", "learn_phi_report.out.gz", "phi_report.tsv", "summary.tsv"):
        selected_phi = _read_selected_phi_from_report(root / report_name)
        if selected_phi is not None:
            break
    specs = []
    for phi, run_dir in sorted(candidates, key=lambda item: (math.inf if math.isnan(item[0]) else item[0])):
        phi_label = "unknown" if math.isnan(phi) else ("%g" % phi)
        mode_id = f"{spec.mode_id}_phi_{phi_label.replace('.', 'p')}"
        specs.append(
            EagglRunSpec(
                spec.run_id,
                mode_id,
                run_dir,
                group_id=spec.mode_id,
                group_title=f"{spec.mode_id.replace('_', ' ')} phi sweep",
            )
        )
    if selected_phi is not None:
        specs.sort(key=lambda item: (
            0 if _parse_phi_from_name(item.mode_id) is not None and math.isclose(float(_parse_phi_from_name(item.mode_id)), float(selected_phi), rel_tol=1e-12, abs_tol=1e-15) else 1,
            _parse_phi_from_name(item.mode_id) if _parse_phi_from_name(item.mode_id) is not None else math.inf,
        ))
    return specs


def normalize_gene_rows(rows: list[dict[str, str]], warnings: list[str], *, combined_threshold: float, max_rows: int) -> list[dict]:
    normalized = []
    if rows and not any(key in rows[0] for key in ("Gene", "gene")):
        warnings.append("gene stats table lacks Gene/gene column")
    for row in rows:
        gene = _first(row, ["Gene", "gene", "id", "ID"])
        if not gene:
            continue
        combined = parse_float(_first(row, ["combined", "Combined", "combined_log_bf"]))
        if combined is not None and combined < combined_threshold:
            continue
        normalized.append(
            {
                "gene": gene,
                "label": _first(row, ["label", "Label"], gene),
                "combined": combined,
                "log_bf": parse_float(_first(row, ["log_bf", "Direct", "direct", "direct_log_bf"])),
                "prior": parse_float(_first(row, ["prior", "Indirect", "indirect", "indirect_log_bf"])),
                "huge_score": parse_float(_first(row, ["huge_score_gwas", "positive_control", "huge_score"])),
                "n": parse_float(_first(row, ["N", "n"])),
                "chrom": _first(row, ["Chrom", "chrom", "chr"]),
                "start": parse_float(_first(row, ["Start", "start"])),
                "end": parse_float(_first(row, ["End", "end"])),
            }
        )
    normalized.sort(key=lambda item: (item.get("combined") is not None, item.get("combined") or -1e300), reverse=True)
    return normalized if max_rows < 0 else normalized[:max_rows]


def normalize_gene_set_rows(rows: list[dict[str, str]], warnings: list[str], *, beta_threshold: float, max_rows: int) -> list[dict]:
    normalized = []
    if rows and not any(key in rows[0] for key in ("Gene_Set", "gene_set")):
        warnings.append("gene-set stats table lacks Gene_Set/gene_set column")
    for row in rows:
        gene_set = _first(row, ["Gene_Set", "gene_set", "id", "ID"])
        if not gene_set:
            continue
        beta = parse_float(_first(row, ["beta", "Beta"]))
        beta_uncorrected = parse_float(_first(row, ["beta_uncorrected", "Beta_uncorrected"]))
        filter_value = beta_uncorrected if beta_uncorrected is not None else beta
        if filter_value is not None and filter_value < beta_threshold:
            continue
        normalized.append(
            {
                "gene_set": gene_set,
                "label": _first(row, ["label", "Label"], gene_set),
                "filter_reason": _first(row, ["filter_reason", "Filter", "filter"]),
                "n": parse_float(_first(row, ["N", "n"])),
                "beta": beta,
                "beta_uncorrected": beta_uncorrected,
                "p_orig": parse_float(_first(row, ["P_orig", "p_orig", "P"])),
                "z_orig": parse_float(_first(row, ["Z_orig", "z_orig", "Z"])),
            }
        )
    normalized.sort(
        key=lambda item: (
            item.get("beta") is not None,
            item.get("beta") or -1e300,
            item.get("beta_uncorrected") or -1e300,
        ),
        reverse=True,
    )
    return normalized if max_rows < 0 else normalized[:max_rows]


def read_gene_set_membership(paths: list[Path], warnings: list[str]) -> dict[str, set[str]]:
    membership: dict[str, set[str]] = defaultdict(set)
    for path in paths:
        if not path.exists():
            warnings.append(f"missing X input for membership expansion: {path}")
            continue
        try:
            with open_text(path) as handle:
                for line in handle:
                    parts = line.rstrip("\n").split("\t")
                    if len(parts) < 2:
                        continue
                    gene_set = parts[0].strip()
                    if not gene_set:
                        continue
                    start = 2 if len(parts) > 2 and parts[1].strip().lower() in {"", "na", "nan", "description"} else 1
                    for token in parts[start:]:
                        gene = token.split(":", 1)[0].strip()
                        if gene:
                            membership[gene_set].add(gene)
        except OSError as exc:
            warnings.append(f"could not read X input {path}: {exc}")
    return dict(membership)


def build_gene_expansions(
    genes: list[dict],
    gene_sets: list[dict],
    membership: dict[str, set[str]],
    beta_threshold: float,
    *,
    max_rows_per_entry: int,
) -> dict[str, list[dict]]:
    gene_index = {row["gene"] for row in genes}
    expansions: dict[str, list[dict]] = defaultdict(list)
    for row in gene_sets:
        beta = row.get("beta_uncorrected") if row.get("beta_uncorrected") is not None else row.get("beta")
        if beta is not None and beta < beta_threshold:
            continue
        for gene in membership.get(row["gene_set"], set()):
            if gene in gene_index:
                expansions[gene].append({"gene_set": row["gene_set"], "label": row.get("label", ""), "beta": row.get("beta"), "beta_uncorrected": row.get("beta_uncorrected"), "n": row.get("n")})
    for gene in expansions:
        expansions[gene].sort(key=lambda item: item.get("beta_uncorrected") or item.get("beta") or -1e300, reverse=True)
        if max_rows_per_entry >= 0:
            expansions[gene] = expansions[gene][:max_rows_per_entry]
    return dict(expansions)


def build_gene_set_expansions(
    genes: list[dict],
    gene_sets: list[dict],
    membership: dict[str, set[str]],
    combined_threshold: float,
    *,
    max_rows_per_entry: int,
) -> dict[str, list[dict]]:
    gene_by_id = {row["gene"]: row for row in genes if row.get("combined") is None or row.get("combined") >= combined_threshold}
    expansions: dict[str, list[dict]] = {}
    for row in gene_sets:
        members = []
        for gene in membership.get(row["gene_set"], set()):
            if gene in gene_by_id:
                g = gene_by_id[gene]
                members.append({"gene": gene, "combined": g.get("combined"), "log_bf": g.get("log_bf"), "prior": g.get("prior")})
        members.sort(key=lambda item: item.get("combined") or -1e300, reverse=True)
        if max_rows_per_entry >= 0:
            members = members[:max_rows_per_entry]
        if members:
            expansions[row["gene_set"]] = members
    return expansions


def load_pigean_run(spec: PigeanRunSpec, args: argparse.Namespace, membership: dict[str, set[str]]) -> dict:
    warnings: list[str] = []
    path = spec.path
    if not path.exists():
        warnings.append(f"PIGEAN directory does not exist: {path}")
    gene_path = path / "pigean.gene_stats.out.gz"
    gene_set_path = path / "pigean.gene_set_stats.out.gz"
    genes = normalize_gene_rows(
        read_tsv(gene_path, warnings) if gene_path.exists() else [],
        warnings,
        combined_threshold=args.gene_threshold,
        max_rows=args.max_genes_per_run,
    )
    if not gene_path.exists():
        warnings.append(f"missing PIGEAN gene stats: {gene_path}")
    gene_sets = normalize_gene_set_rows(
        read_tsv(gene_set_path, warnings) if gene_set_path.exists() else [],
        warnings,
        beta_threshold=args.gene_set_threshold,
        max_rows=args.max_gene_sets_per_run,
    )
    if not gene_set_path.exists():
        warnings.append(f"missing PIGEAN gene-set stats: {gene_set_path}")
    return {
        "run_id": spec.run_id,
        "title": args.run_titles.get(spec.run_id, spec.run_id.replace("_", " ")),
        "summary": "PIGEAN run supplied on the dashboard command line.",
        "trait_id": args.trait_ids.get(spec.run_id, ""),
        "paths": {
            "run_dir": str(path),
            "gene_stats": str(gene_path),
            "gene_set_stats": str(gene_set_path),
            "params": str(path / "pigean.params.out"),
            "run_log": str(path / "pigean.run.log"),
            "warnings_log": str(path / "pigean.warnings.log"),
        },
        "warnings": warnings,
        "genes": genes,
        "gene_sets": gene_sets,
        "gene_expansions": build_gene_expansions(
            genes,
            gene_sets,
            membership,
            args.gene_set_threshold,
            max_rows_per_entry=args.max_provenance_rows_per_entry,
        ),
        "gene_set_expansions": build_gene_set_expansions(
            genes,
            gene_sets,
            membership,
            args.gene_threshold,
            max_rows_per_entry=args.max_provenance_rows_per_entry,
        ),
    }


def placeholder_pigean_run(run_id: str, args: argparse.Namespace, reason: str) -> dict:
    return {
        "run_id": run_id,
        "title": args.run_titles.get(run_id, run_id.replace("_", " ")),
        "summary": "Placeholder run created so supplied EAGGL outputs are visible without a matching PIGEAN directory.",
        "trait_id": args.trait_ids.get(run_id, ""),
        "paths": {},
        "warnings": [reason],
        "genes": [],
        "gene_sets": [],
        "gene_expansions": {},
        "gene_set_expansions": {},
    }


def read_cluster_table(
    path: Path,
    id_key: str,
    threshold: float,
    warnings: list[str],
    *,
    factor_loading_min_max_frac: float | None,
    max_rows_per_factor: int,
) -> tuple[list[str], dict[str, list[dict]]]:
    rows = read_tsv(path, warnings)
    factors = factor_columns(list(rows[0].keys()) if rows else [])
    by_factor: dict[str, list[dict]] = defaultdict(list)
    id_names = [id_key, id_key.lower(), "id", "ID"]
    for row in rows:
        entity_id = _first(row, id_names)
        if not entity_id:
            continue
        for factor in factors:
            loading = parse_float(row.get(factor), 0.0) or 0.0
            record = {
                "id": entity_id,
                "gene" if id_key == "Gene" else "gene_set": entity_id,
                "loading": loading,
                "relative_loading": parse_float(row.get(f"Relative_{factor}")),
                "combined_loading": parse_float(row.get(f"Combined_{factor}")),
                "cluster": _first(row, ["cluster", "Cluster"]),
                "label": _first(row, ["label", "Label"], entity_id),
                "combined": parse_float(_first(row, ["combined", "Combined"])),
                "log_bf": parse_float(_first(row, ["log_bf", "Direct"])),
                "prior": parse_float(_first(row, ["prior", "Indirect"])),
                "beta": parse_float(_first(row, ["beta", "Beta"])),
                "beta_uncorrected": parse_float(_first(row, ["beta_uncorrected", "Beta_uncorrected"])),
                "in_discovery": _first(row, ["in_discovery", "In_Discovery"]),
            }
            by_factor[factor].append(record)
    for factor, records in by_factor.items():
        records.sort(key=lambda item: item.get("loading") or -1e300, reverse=True)
        max_loading = records[0].get("loading") if records else None
        if max_loading is not None and factor_loading_min_max_frac is not None and factor_loading_min_max_frac >= 0:
            records = [
                record
                for record in records
                if (record.get("loading") or 0.0) >= max(threshold, max_loading * factor_loading_min_max_frac)
            ]
        else:
            records = [record for record in records if (record.get("loading") or 0.0) >= threshold]
        if max_rows_per_factor >= 0:
            records = records[:max_rows_per_factor]
        by_factor[factor] = records
    return factors, dict(by_factor)


def _is_truthy(value) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "t", "yes", "y"}


def _anchor_column_name(trait: str, existing: set[str]) -> str:
    base = "anchor_" + re.sub(r"[^A-Za-z0-9]+", "_", trait).strip("_").lower()
    if not base or base == "anchor_":
        base = "anchor_trait"
    candidate = base
    index = 2
    while candidate in existing:
        candidate = f"{base}_{index}"
        index += 1
    existing.add(candidate)
    return candidate


def read_trait_links(path: Path, min_trait_neff: float, warnings: list[str]) -> tuple[dict[str, list[dict]], list[dict], dict[str, dict[str, float]]]:
    if not path.exists():
        return {}, [], {}
    rows = read_tsv(path, warnings)
    by_factor: dict[str, list[dict]] = defaultdict(list)
    anchor_traits: dict[str, dict] = {}
    anchor_values_by_factor: dict[str, dict[str, float]] = defaultdict(dict)
    used_columns: set[str] = set()
    for row in rows:
        factor = _first(row, ["factor", "Factor"])
        trait = _first(row, ["trait", "Trait", "pheno", "Phenotype"])
        if not factor or not trait:
            continue
        neff = parse_float(_first(row, ["trait_neff", "trait_n_eff", "retained_n_eff"]), 0.0) or 0.0
        if neff < min_trait_neff:
            continue
        record = {
            "trait": trait,
            "factor": factor,
            "is_anchor": _first(row, ["is_anchor"]),
            "joint_fraction": parse_float(row.get("joint_fraction")),
            "marginal_fraction": parse_float(row.get("marginal_fraction")),
            "marginal_overlap": parse_float(row.get("marginal_overlap")),
            "trait_neff": neff,
            "retained_n_eff": parse_float(row.get("retained_n_eff")),
            "joint_support_mass": parse_float(row.get("joint_support_mass")),
            "marginal_support_mass": parse_float(row.get("marginal_support_mass")),
            "joint_residual": parse_float(row.get("joint_residual")),
        }
        by_factor[factor].append(record)
        if _is_truthy(record["is_anchor"]):
            if trait not in anchor_traits:
                anchor_traits[trait] = {
                    "trait": trait,
                    "label": trait,
                    "column": _anchor_column_name(trait, used_columns),
                    "metric": "joint_fraction",
                }
            value = record.get("joint_fraction")
            if value is not None:
                anchor_values_by_factor[factor][anchor_traits[trait]["column"]] = value
    for factor in by_factor:
        by_factor[factor].sort(key=lambda item: item.get("joint_fraction") or -1e300, reverse=True)
    return dict(by_factor), list(anchor_traits.values()), dict(anchor_values_by_factor)


def _metric_summary_for_group(eaggl_run: dict) -> dict:
    metrics = dict(eaggl_run.get("selected_phi_metrics") or {})
    if "phi" not in metrics and eaggl_run.get("phi") is not None:
        metrics["phi"] = eaggl_run.get("phi")
    if "num_factors" not in metrics and eaggl_run.get("factors") is not None:
        metrics["num_factors"] = len(eaggl_run.get("factors") or [])
    if "modal_factor_count" not in metrics and "num_factors" in metrics:
        metrics["modal_factor_count"] = metrics["num_factors"]
    return metrics


def load_eaggl_run(spec: EagglRunSpec, args: argparse.Namespace) -> dict:
    warnings: list[str] = []
    path = spec.path
    if not path.exists():
        warnings.append(f"EAGGL directory does not exist: {path}")
    factors_path = path / "factors.out.gz"
    factor_metrics_path = _first_existing(
        path,
        [
            "factor_metrics.out.gz",
            "factor_metrics.out",
            "factor_metrics.tsv.gz",
            "factor_metrics.tsv",
        ],
    )
    phi_selection_metrics_path = _first_existing(
        path,
        [
            "phi_selection_metrics_wide.out.gz",
            "phi_selection_metrics_wide.out",
            "phi_selection_metrics_wide.tsv.gz",
            "phi_selection_metrics_wide.tsv",
            "phi_selection_metrics.out.gz",
            "phi_selection_metrics.out",
            "learn_phi_report.out.gz",
            "learn_phi_report.out",
            "learn_phi_report.tsv.gz",
            "learn_phi_report.tsv",
            "phi_report.tsv.gz",
            "phi_report.tsv",
            "summary.tsv.gz",
            "summary.tsv",
        ],
    )
    gene_clusters_path = path / "gene_clusters.out.gz"
    gene_loading_source_specs = [
        ("discovery", "Discovery genes", gene_clusters_path, path / "factor_graph.html"),
        ("full_direct", "Full genes: direct projection", path / "gene_clusters_full.out.gz", path / "factor_graph.full_direct.html"),
        ("full_via_gene_sets", "Full genes: via gene sets", path / "gene_clusters_full_via_gene_sets.out.gz", path / "factor_graph.full_via_gene_sets.html"),
    ]
    gene_set_clusters_path = path / "gene_set_clusters.out.gz"
    factors_rows = read_tsv(factors_path, warnings) if factors_path.exists() else []
    factor_metrics = read_factor_metrics(factor_metrics_path, warnings) if factor_metrics_path is not None else {}
    selected_phi_metrics = read_selected_phi_metrics(phi_selection_metrics_path, warnings) if phi_selection_metrics_path is not None else {}
    if not factors_path.exists():
        warnings.append(f"missing EAGGL factors: {factors_path}")
    gene_loading_sources: dict[str, dict] = {}
    for source_id, source_label, source_path, graph_path in gene_loading_source_specs:
        if not source_path.exists():
            continue
        _, source_by_factor = read_cluster_table(
            source_path,
            "Gene",
            args.factor_loading_threshold,
            warnings,
            factor_loading_min_max_frac=args.factor_loading_min_max_frac,
            max_rows_per_factor=args.max_factor_genes,
        )
        gene_loading_sources[source_id] = {
            "id": source_id,
            "label": source_label,
            "path": str(source_path),
            "factor_graph_html_path": str(graph_path),
            "factor_graph_available": graph_path.exists(),
            "factor_graph_html": read_optional_text(graph_path, warnings, max_chars=None),
            "by_factor": source_by_factor,
        }
    if not gene_clusters_path.exists():
        warnings.append(f"missing EAGGL gene clusters: {gene_clusters_path}")
    genes_by_factor = gene_loading_sources.get("discovery", {}).get("by_factor", {})
    _, gene_sets_by_factor = (
        read_cluster_table(
            gene_set_clusters_path,
            "Gene_Set",
            args.factor_loading_threshold,
            warnings,
            factor_loading_min_max_frac=args.factor_loading_min_max_frac,
            max_rows_per_factor=args.max_factor_gene_sets,
        )
        if gene_set_clusters_path.exists()
        else ([], {})
    )
    if not gene_set_clusters_path.exists():
        warnings.append(f"missing EAGGL gene-set clusters: {gene_set_clusters_path}")
    trait_links, anchor_traits, anchor_values = read_trait_links(path / "trait_factor_links.out.gz", args.trait_min_neff, warnings)
    factors = []
    if factors_rows and not any(key in factors_rows[0] for key in ("Factor", "factor")):
        warnings.append("factors table lacks Factor/factor column")
    for row in factors_rows:
        factor = _first(row, ["Factor", "factor"])
        if not factor:
            continue
        factor_record = {
                "factor": factor,
                "label": _first(row, ["label", "Label"], factor),
                "lambda": parse_float(row.get("lambda")),
                "factor_tier": _first(row, ["factor_tier", "tier"]),
                "combined_mass_fraction": parse_float(row.get("combined_mass_fraction")),
                "top_genes": [item.strip() for item in (_first(row, ["top_genes"])).replace(";", ",").split(",") if item.strip()],
                "top_gene_sets": [item.strip() for item in (_first(row, ["top_gene_sets"])).replace(";", ",").split(",") if item.strip()],
                "genes": genes_by_factor.get(factor, []),
                "gene_sets": gene_sets_by_factor.get(factor, []),
                "phenotypes": trait_links.get(factor, []),
                "metrics": factor_metrics.get(factor, {}),
            }
        factor_record.update(factor_metrics.get(factor, {}))
        factor_record.update(anchor_values.get(factor, {}))
        factors.append(factor_record)
    factors.sort(key=lambda item: item.get("combined_mass_fraction") or -1e300, reverse=True)
    factor_graph_html_path = path / "factor_graph.html"
    phi_value = _parse_phi_from_name(spec.mode_id) or _parse_phi_from_name(path.name) or _parse_phi_from_name(path.parent.name)
    return {
        "run_id": spec.run_id,
        "mode_id": spec.mode_id,
        "group_id": spec.group_id or spec.mode_id,
        "group_title": spec.group_title or (spec.group_id or spec.mode_id).replace("_", " "),
        "title": ("%s (phi %g)" % (spec.mode_id.replace("_", " "), phi_value)) if phi_value is not None else spec.mode_id.replace("_", " "),
        "phi": phi_value,
        "summary": "EAGGL run supplied on the dashboard command line.",
        "paths": {
            "run_dir": str(path),
            "factors": str(factors_path),
            "factor_metrics": str(factor_metrics_path) if factor_metrics_path is not None else str(path / "factor_metrics.out.gz"),
            "phi_selection_metrics": str(phi_selection_metrics_path) if phi_selection_metrics_path is not None else "",
            "gene_clusters": str(gene_clusters_path),
            "gene_clusters_full": str(path / "gene_clusters_full.out.gz"),
            "gene_clusters_full_via_gene_sets": str(path / "gene_clusters_full_via_gene_sets.out.gz"),
            "gene_set_clusters": str(gene_set_clusters_path),
            "trait_factor_links": str(path / "trait_factor_links.out.gz"),
            "factor_graph_html": str(factor_graph_html_path),
            "factor_graph_json": str(path / "factor_graph.json"),
            "params": str(path / "params.out"),
            "run_log": str(path / "eaggl.run.log"),
            "warnings_log": str(path / "eaggl.warnings.log"),
        },
        "warnings": warnings,
        "factor_graph_available": factor_graph_html_path.exists(),
        "factor_graph_html": read_optional_text(factor_graph_html_path, warnings, max_chars=None),
        "gene_loading_sources": gene_loading_sources,
        "anchor_traits": anchor_traits,
        "selected_phi_metrics": selected_phi_metrics,
        "factors": factors,
    }


def build_payload(args: argparse.Namespace) -> dict:
    warnings: list[str] = []
    membership = read_gene_set_membership(args.x_input or [], warnings)
    pigean_runs = [load_pigean_run(spec, args, membership) for spec in args.pigean_run]
    seen_pigean_run_ids = {run["run_id"] for run in pigean_runs}
    eaggl_runs = {}
    group_overrides = {
        (spec.run_id, spec.mode_id): spec
        for spec in args.eaggl_group or []
    }
    eaggl_specs = list(args.eaggl_run or [])
    for sweep_spec in args.eaggl_phi_sweep or []:
        sweep_runs = _discover_phi_sweep_runs(sweep_spec)
        if not sweep_runs:
            warnings.append(f"no EAGGL phi-sweep runs found under {sweep_spec.path}")
        eaggl_specs.extend(sweep_runs)
    for spec in eaggl_specs:
        group_override = group_overrides.get((spec.run_id, spec.mode_id))
        if group_override is not None:
            spec = EagglRunSpec(
                spec.run_id,
                spec.mode_id,
                spec.path,
                group_id=group_override.group_id,
                group_title=group_override.group_title,
            )
        if spec.run_id not in seen_pigean_run_ids:
            pigean_runs.append(
                placeholder_pigean_run(
                    spec.run_id,
                    args,
                    f"no --pigean-run was supplied for EAGGL run {spec.run_id}",
                )
            )
            seen_pigean_run_ids.add(spec.run_id)
        loaded = load_eaggl_run(spec, args)
        eaggl_runs[f"{spec.run_id}::{spec.mode_id}"] = loaded
    eaggl_groups_by_run: dict[str, dict[str, dict]] = defaultdict(dict)
    for eaggl in eaggl_runs.values():
        group_id = eaggl.get("group_id") or eaggl.get("mode_id")
        group = eaggl_groups_by_run[eaggl["run_id"]].setdefault(
            group_id,
            {
                "run_id": eaggl["run_id"],
                "group_id": group_id,
                "title": eaggl.get("group_title") or str(group_id).replace("_", " "),
                "mode_ids": [],
                "metrics_by_mode": {},
            },
        )
        group["mode_ids"].append(eaggl["mode_id"])
        group["metrics_by_mode"][eaggl["mode_id"]] = _metric_summary_for_group(eaggl)
    eaggl_groups = {
        run_id: list(groups.values())
        for run_id, groups in eaggl_groups_by_run.items()
    }
    payload = {
        "schema": "pigean_dashboard/v1",
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "run_root": str(Path.cwd()),
        "title": args.title,
        "default_gene_loading_source": args.default_gene_loading_source,
        "thresholds": {
            "gene_combined": args.gene_threshold,
            "gene_set_beta_uncorrected": args.gene_set_threshold,
            "factor_loading": args.factor_loading_threshold,
            "factor_loading_min_max_frac": args.factor_loading_min_max_frac,
            "trait_neff": args.trait_min_neff,
            "max_provenance_rows_per_entry": args.max_provenance_rows_per_entry,
        },
        "warnings": warnings,
        "x_inputs": [str(path) for path in args.x_input or []],
        "gene_set_membership_count": len(membership),
        "pigean_runs": pigean_runs,
        "eaggl_runs": eaggl_runs,
        "eaggl_groups": eaggl_groups,
    }
    return payload


def write_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def write_html(payload: dict, path: Path, *, title: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_html(payload, title=title), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a standalone HTML dashboard from supplied PIGEAN and EAGGL outputs.")
    parser.add_argument("--pigean-run", action="append", type=parse_run_spec, default=[], help="PIGEAN run as RUN_ID:DIR; repeat for multiple runs.")
    parser.add_argument("--eaggl-run", action="append", type=parse_eaggl_spec, default=[], help="EAGGL run as RUN_ID:MODE_ID:DIR; repeat for multiple modes/runs.")
    parser.add_argument("--eaggl-phi-sweep", action="append", type=parse_eaggl_phi_sweep_spec, default=[], help="EAGGL phi sweep bundle as RUN_ID:MODE_ID:DIR. The directory is scanned for per-phi EAGGL output directories.")
    parser.add_argument("--eaggl-group", action="append", type=parse_eaggl_group_spec, default=[], help="Assign a standalone EAGGL run to a dashboard group as RUN_ID:MODE_ID:GROUP_ID[:GROUP_TITLE]; repeatable.")
    parser.add_argument("--x-input", action="append", type=Path, default=None, help="Optional GMT/gene-set input for gene/gene-set membership expansions; repeatable.")
    parser.add_argument("--title", default="PIGEAN/EAGGL Dashboard")
    parser.add_argument("--run-title", action="append", type=parse_run_title, default=[], help="Display title as RUN_ID:TITLE; repeatable.")
    parser.add_argument("--trait-id", action="append", type=parse_run_title, default=[], help="Optional trait identifier as RUN_ID:TRAIT; repeatable.")
    parser.add_argument("--html-out", default=None)
    parser.add_argument("--json-out", default=None)
    parser.add_argument("--gene-threshold", type=float, default=1.0)
    parser.add_argument("--gene-set-threshold", type=float, default=0.01)
    parser.add_argument("--factor-loading-threshold", type=float, default=0.0)
    parser.add_argument("--factor-loading-min-max-frac", type=float, default=0.05, help="Keep factor gene/gene-set rows with loading at least this fraction of the factor-specific maximum; use a negative value to disable.")
    parser.add_argument("--trait-min-neff", type=float, default=200.0)
    parser.add_argument(
        "--default-gene-loading-source",
        choices=("discovery", "full_direct", "full_via_gene_sets"),
        default="discovery",
        help="Initial EAGGL gene-loading source when multiple projections are available.",
    )
    parser.add_argument("--max-genes-per-run", type=int, default=5000)
    parser.add_argument("--max-gene-sets-per-run", type=int, default=2500)
    parser.add_argument("--max-factor-genes", type=int, default=150)
    parser.add_argument("--max-factor-gene-sets", type=int, default=150)
    parser.add_argument("--max-provenance-rows-per-entry", type=int, default=50)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.run_titles = dict(args.run_title or [])
    args.trait_ids = dict(args.trait_id or [])
    if not args.pigean_run and not args.eaggl_run and not args.eaggl_phi_sweep:
        parser.error("Need at least one --pigean-run, --eaggl-run, or --eaggl-phi-sweep")
    if args.html_out is None and args.json_out is None:
        parser.error("Need at least one of --html-out or --json-out")
    payload = build_payload(args)
    if args.json_out is not None:
        write_json(payload, Path(args.json_out))
    if args.html_out is not None:
        write_html(payload, Path(args.html_out), title=args.title)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
