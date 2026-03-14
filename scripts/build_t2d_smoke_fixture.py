from __future__ import annotations

import csv
import gzip
import json
import math
import random
import shutil
from pathlib import Path

import numpy as np

GWAS_P_THRESHOLD = 1e-6
EXOMES_P_THRESHOLD = 1e-4
CASE_TOTAL = 10_000
CTRL_TOTAL = 20_000
RNG_SEED = 20260314
TOY_GENE_SET_IDS = [
    "mp_abnormal_glucose_homeostasis",
    "mp_abnormal_circulating_glucose_level",
    "mp_abnormal_circulating_insulin_level",
    "mp_abnormal_insulin_secretion",
    "mp_abnormal_pancreas_morphology",
    "mp_abnormal_pancreas_secretion",
    "mp_abnormal_endocrine_pancreas_morphology",
    "mp_abnormal_endocrine_pancreas_physiology",
    "mp_decreased_circulating_glucose_level",
    "mp_abnormal_pancreatic_alpha_cell_morphology",
]


def _read_mody_genes(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _format_float(value: float, digits: int = 8) -> str:
    return f"{value:.{digits}g}"


def build_fixture() -> dict[str, object]:
    repo_root = Path(__file__).resolve().parents[1]
    analysis_root = repo_root.parent
    out_dir = repo_root / "tests" / "data" / "t2d_smoke"
    out_dir.mkdir(parents=True, exist_ok=True)

    src_gwas = analysis_root / "data" / "T2D.chrom_pos.sumstats.gz"
    src_exomes = analysis_root / "data" / "T2D.exomes.txt"
    src_mody = analysis_root / "data" / "mody.gene.list"
    src_mouse_gene_sets = repo_root / "tests" / "data" / "model_small" / "gene_set_list_mouse_2024.txt"
    if not src_gwas.exists() or not src_exomes.exists() or not src_mody.exists() or not src_mouse_gene_sets.exists():
        missing = [str(p) for p in [src_gwas, src_exomes, src_mody, src_mouse_gene_sets] if not p.exists()]
        raise SystemExit(f"Missing source inputs: {', '.join(missing)}")

    mody_genes = _read_mody_genes(src_mody)
    mody_gene_set = set(mody_genes)

    out_gwas = out_dir / "T2D.p_lt_1e-6.chrom_pos.sumstats.tsv.gz"
    out_exomes = out_dir / "T2D.exomes.p_lt_1e-4_or_mody.tsv"
    out_mody = out_dir / "mody.gene.list"
    out_case = out_dir / "mody_case_counts.tsv"
    out_ctrl = out_dir / "mody_ctrl_counts.tsv"
    out_toy_gene_sets = out_dir / "gene_set_list_mouse_t2d_toy.txt"
    out_summary = out_dir / "fixture_summary.json"
    out_readme = out_dir / "README.md"

    gwas_rows = 0
    with gzip.open(src_gwas, "rt") as src, gzip.open(out_gwas, "wt") as dst:
        header = src.readline()
        dst.write(header)
        for line in src:
            cols = line.rstrip("\n").split()
            if len(cols) < 4:
                continue
            try:
                p = float(cols[2])
            except ValueError:
                continue
            if p < GWAS_P_THRESHOLD:
                dst.write("\t".join(cols) + "\n")
                gwas_rows += 1

    exome_rows = 0
    kept_mody_rows = 0
    with src_exomes.open("r", encoding="utf-8") as src, out_exomes.open("w", encoding="utf-8", newline="") as dst:
        reader = csv.DictReader(src, delimiter="\t")
        extra_fields = ["beta", "se", "n"]
        writer = csv.DictWriter(
            dst,
            fieldnames=[*(reader.fieldnames or []), *extra_fields],
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        for row in reader:
            gene = (row.get("GeneSymbol") or "").strip()
            try:
                p = float(row["P-value"])
                z = float(row["Zscore"])
                n = float(row["Weight"])
            except (ValueError, KeyError, TypeError):
                continue
            if p < EXOMES_P_THRESHOLD or gene in mody_gene_set:
                se = 1.0 / math.sqrt(n)
                beta = z * se
                row["beta"] = _format_float(beta, 8)
                row["se"] = _format_float(se, 8)
                row["n"] = _format_float(n, 8)
                writer.writerow(row)
                exome_rows += 1
                if gene in mody_gene_set:
                    kept_mody_rows += 1

    shutil.copyfile(src_mody, out_mody)

    kept_gene_sets = 0
    with src_mouse_gene_sets.open("r", encoding="utf-8") as src, out_toy_gene_sets.open("w", encoding="utf-8") as dst:
        for line in src:
            gene_set_id = line.split("\t", 1)[0].strip()
            if gene_set_id in TOY_GENE_SET_IDS:
                dst.write(line)
                kept_gene_sets += 1
    if kept_gene_sets != len(TOY_GENE_SET_IDS):
        raise SystemExit(
            "Did not find all requested toy gene sets; "
            f"kept {kept_gene_sets} of {len(TOY_GENE_SET_IDS)}"
        )

    rng = random.Random(RNG_SEED)
    np_rng = np.random.default_rng(RNG_SEED)
    count_rows: list[dict[str, object]] = []
    for gene in mody_genes:
        rr = 1.0 + rng.expovariate(1.0 / 4.0)
        ctrl_freq = rng.uniform(5e-5, 1.5e-4)
        case_freq = min(ctrl_freq * rr, 9e-4)
        revel = rng.uniform(0.45, 0.99)
        ctrl_count = int(np_rng.binomial(CTRL_TOTAL, ctrl_freq))
        case_count = int(np_rng.binomial(CASE_TOTAL, case_freq))
        if case_count == 0:
            case_count = 1
        count_rows.append(
            {
                "gene": gene,
                "revel": revel,
                "rr": rr,
                "case_count": case_count,
                "case_total": CASE_TOTAL,
                "case_max_freq": case_freq,
                "ctrl_count": ctrl_count,
                "ctrl_total": CTRL_TOTAL,
                "ctrl_max_freq": ctrl_freq,
            }
        )

    case_fields = ["gene", "revel", "count", "total", "max_freq", "rr"]
    ctrl_fields = ["gene", "revel", "count", "total", "max_freq", "rr"]
    with out_case.open("w", encoding="utf-8", newline="") as case_fh, out_ctrl.open("w", encoding="utf-8", newline="") as ctrl_fh:
        case_writer = csv.DictWriter(case_fh, fieldnames=case_fields, delimiter="\t", lineterminator="\n")
        ctrl_writer = csv.DictWriter(ctrl_fh, fieldnames=ctrl_fields, delimiter="\t", lineterminator="\n")
        case_writer.writeheader()
        ctrl_writer.writeheader()
        for row in count_rows:
            case_writer.writerow(
                {
                    "gene": row["gene"],
                    "revel": _format_float(float(row["revel"]), 6),
                    "count": int(row["case_count"]),
                    "total": int(row["case_total"]),
                    "max_freq": _format_float(float(row["case_max_freq"]), 6),
                    "rr": _format_float(float(row["rr"]), 6),
                }
            )
            ctrl_writer.writerow(
                {
                    "gene": row["gene"],
                    "revel": _format_float(float(row["revel"]), 6),
                    "count": int(row["ctrl_count"]),
                    "total": int(row["ctrl_total"]),
                    "max_freq": _format_float(float(row["ctrl_max_freq"]), 6),
                    "rr": _format_float(float(row["rr"]), 6),
                }
            )

    summary = {
        "source_gwas": str(src_gwas.relative_to(analysis_root)),
        "source_exomes": str(src_exomes.relative_to(analysis_root)),
        "source_mody": str(src_mody.relative_to(analysis_root)),
        "gwas_p_threshold": GWAS_P_THRESHOLD,
        "exomes_p_threshold": EXOMES_P_THRESHOLD,
        "gwas_rows": gwas_rows,
        "exome_rows": exome_rows,
        "exome_rows_for_mody_genes": kept_mody_rows,
        "mody_gene_count": len(mody_genes),
        "toy_gene_set_count": kept_gene_sets,
        "case_total": CASE_TOTAL,
        "ctrl_total": CTRL_TOTAL,
        "rng_seed": RNG_SEED,
    }
    out_summary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    size_lines = []
    for path in [out_gwas, out_exomes, out_mody, out_case, out_ctrl, out_toy_gene_sets, out_summary]:
        size_lines.append(f"- `{path.name}`: {path.stat().st_size} bytes")
    out_readme.write_text(
        "# T2D Smoke Fixtures\n\n"
        "Deterministic compact fixtures for repo-tracked PIGEAN tests.\n\n"
        "Contents:\n"
        f"- `T2D.p_lt_1e-6.chrom_pos.sumstats.tsv.gz`: subset of `{src_gwas.relative_to(analysis_root)}` with `P < 1e-6`\n"
        f"- `T2D.exomes.p_lt_1e-4_or_mody.tsv`: subset of `{src_exomes.relative_to(analysis_root)}` with `P-value < 1e-4` plus all MODY genes\n"
        "- `mody.gene.list`: copied MODY positive-control list\n"
        "- `mody_case_counts.tsv` / `mody_ctrl_counts.tsv`: synthetic burden-count fixtures for the MODY genes\n"
        f"- `gene_set_list_mouse_t2d_toy.txt`: 10 curated diabetes/pancreas lines copied from `{src_mouse_gene_sets.relative_to(repo_root)}` for fast toy tests\n"
        "- `fixture_summary.json`: thresholds and row counts\n\n"
        "Synthetic counts were generated with RNG seed `20260314`, cohort sizes 10k cases / 20k controls, and per-gene relative risk sampled as `1 + Exponential(scale=4)` so the expected mean RR is 5 with a minimum of 1.\n\n"
        "File sizes:\n"
        + "\n".join(size_lines)
        + "\n",
        encoding="utf-8",
    )

    return summary


if __name__ == "__main__":
    print(json.dumps(build_fixture(), indent=2, sort_keys=True))
