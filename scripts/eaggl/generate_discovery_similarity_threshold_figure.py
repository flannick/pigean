#!/usr/bin/env python3
"""Generate the supplementary discovery-similarity-threshold tuning figure."""

from __future__ import annotations

import csv
import math
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_DIR = ROOT / "results" / "joint_tau_phi_confirmation_2026-04-21" / "artifacts"
GROUP_SUMMARY = ARTIFACT_DIR / "joint_tau_phi_group_summary.tsv"
MATCH_SUMMARY = ARTIFACT_DIR / "tau_neighbor_matched_w_summary.tsv"
OUT_DIR = ROOT / "for_paper" / "similarity_threshold"
OUT_STEM = OUT_DIR / "supp_discovery_similarity_threshold_tuning"


TRAIT_ORDER = ["Asthma", "CAD", "FEV1toFVC"]
PHI_ORDER = [0.1, 0.2, 0.4]
RHO_ORDER = [0.2, 0.3, 0.35]
COLORS = {
    0.1: "#0072B2",
    0.2: "#E69F00",
    0.4: "#009E73",
}
MARKERS = {
    "Asthma": "o",
    "CAD": "s",
    "FEV1toFVC": "^",
}


def _read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def _float(row: dict[str, str], key: str) -> float:
    value = row.get(key, "")
    if value in {"", "NA", "nan", "None"}:
        return math.nan
    return float(value)


def _median(values: list[float]) -> float:
    clean = sorted(value for value in values if not math.isnan(value))
    if not clean:
        return math.nan
    mid = len(clean) // 2
    if len(clean) % 2:
        return clean[mid]
    return 0.5 * (clean[mid - 1] + clean[mid])


def _spread(values: list[float]) -> tuple[float, float]:
    clean = [value for value in values if not math.isnan(value)]
    if not clean:
        return math.nan, math.nan
    center = _median(clean)
    return center - min(clean), max(clean) - center


def _configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 8,
            "legend.fontsize": 7,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _panel_label(ax, label: str) -> None:
    ax.text(
        -0.18,
        1.08,
        label,
        transform=ax.transAxes,
        fontsize=10,
        fontweight="bold",
        va="top",
        ha="left",
    )


def _plot_discovery_rows(ax, rows: list[dict[str, str]]) -> None:
    by_trait_rho: dict[tuple[str, float], list[float]] = defaultdict(list)
    for row in rows:
        by_trait_rho[(row["trait"], _float(row, "tau"))].append(_float(row, "discovery_rows_median"))

    for trait in TRAIT_ORDER:
        ys = [_median(by_trait_rho[(trait, rho)]) for rho in RHO_ORDER]
        ax.plot(
            RHO_ORDER,
            ys,
            marker=MARKERS[trait],
            linewidth=1.4,
            markersize=4,
            label=trait,
        )

    ax.set_xlabel(r"Discovery similarity threshold $\rho_{\mathrm{disc}}$")
    ax.set_ylabel("Discovery families")
    ax.set_xticks(RHO_ORDER)
    ax.grid(axis="y", color="#d9d9d9", linewidth=0.5)
    ax.legend(frameon=False, loc="upper left")


def _plot_metric_by_phi(
    ax,
    rows: list[dict[str, str]],
    metric: str,
    ylabel: str,
) -> None:
    by_phi_rho: dict[tuple[float, float], list[float]] = defaultdict(list)
    for row in rows:
        by_phi_rho[(_float(row, "phi"), _float(row, "tau"))].append(_float(row, metric))

    for phi in PHI_ORDER:
        centers = [_median(by_phi_rho[(phi, rho)]) for rho in RHO_ORDER]
        lows = []
        highs = []
        for rho, center in zip(RHO_ORDER, centers):
            low, high = _spread(by_phi_rho[(phi, rho)])
            lows.append(low)
            highs.append(high)
        ax.errorbar(
            RHO_ORDER,
            centers,
            yerr=[lows, highs],
            color=COLORS[phi],
            marker="o",
            linewidth=1.4,
            markersize=4,
            capsize=2,
            label=f"phi={phi:g}",
        )

    ax.set_xlabel(r"Discovery similarity threshold $\rho_{\mathrm{disc}}$")
    ax.set_ylabel(ylabel)
    ax.set_xticks(RHO_ORDER)
    ax.grid(axis="y", color="#d9d9d9", linewidth=0.5)
    ax.legend(frameon=False, loc="best")


def _plot_matched_w(ax, rows: list[dict[str, str]]) -> None:
    pairs = ["0.2_vs_0.3", "0.3_vs_0.35"]
    x_base = {pair: index for index, pair in enumerate(pairs)}
    offsets = {0.1: -0.18, 0.2: 0.0, 0.4: 0.18}
    labels_seen: set[float] = set()

    for row in rows:
        phi = _float(row, "phi")
        pair = row["tau_pair"]
        x = x_base[pair] + offsets.get(phi, 0.0)
        y = _float(row, "matched_w_cosine_median_across_seeds")
        ax.scatter(
            x,
            y,
            s=25,
            color=COLORS[phi],
            marker=MARKERS.get(row["trait"], "o"),
            edgecolor="white",
            linewidth=0.4,
            label=f"phi={phi:g}" if phi not in labels_seen else None,
        )
        labels_seen.add(phi)

    for pair in pairs:
        values = [
            _float(row, "matched_w_cosine_median_across_seeds")
            for row in rows
            if row["tau_pair"] == pair
        ]
        center = _median(values)
        ax.plot(
            [x_base[pair] - 0.27, x_base[pair] + 0.27],
            [center, center],
            color="black",
            linewidth=1.1,
        )

    ax.set_xticks([x_base[pair] for pair in pairs])
    ax.set_xticklabels([r"0.2 -> 0.3", r"0.3 -> 0.35"])
    ax.set_ylabel("Matched W cosine")
    ax.set_xlabel(r"Neighboring $\rho_{\mathrm{disc}}$ comparison")
    ax.set_ylim(-0.02, 1.02)
    ax.grid(axis="y", color="#d9d9d9", linewidth=0.5)
    ax.legend(frameon=False, loc="upper right")


def main() -> None:
    _configure_matplotlib()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    group_rows = _read_tsv(GROUP_SUMMARY)
    match_rows = _read_tsv(MATCH_SUMMARY)

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.4), constrained_layout=True)
    axes = axes.ravel()

    _plot_discovery_rows(axes[0], group_rows)
    _plot_metric_by_phi(
        axes[1],
        group_rows,
        "primary_factor_count_median",
        "Primary factors",
    )
    _plot_metric_by_phi(
        axes[2],
        group_rows,
        "effective_factor_count_median",
        "Effective factor count",
    )
    _plot_matched_w(axes[3], match_rows)

    for label, ax in zip(["a", "b", "c", "d"], axes):
        _panel_label(ax, label)

    fig.savefig(OUT_STEM.with_suffix(".pdf"))
    fig.savefig(OUT_STEM.with_suffix(".png"), dpi=300)
    print(f"Wrote {OUT_STEM.with_suffix('.pdf')}")
    print(f"Wrote {OUT_STEM.with_suffix('.png')}")


if __name__ == "__main__":
    main()
