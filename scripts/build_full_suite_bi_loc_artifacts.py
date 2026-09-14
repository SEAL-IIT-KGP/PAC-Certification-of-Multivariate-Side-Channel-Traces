#!/usr/bin/env python3
"""Build compact tables/plots for full-suite BI_loc sweeps."""

from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path

import pandas as pd

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", tempfile.mkdtemp(prefix="full_suite_loc_mpl_"))
os.environ.setdefault("XDG_CACHE_HOME", tempfile.mkdtemp(prefix="full_suite_loc_fontcache_"))

import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, NullFormatter


DATASET_LABELS = {
    "aes_rd": "AES-RD",
    "ascad_desync_0": "ASCAD d0",
    "ascad_desync_50": "ASCAD d50",
    "ascad_desync_100": "ASCAD d100",
}

COLORS = {
    "aes_rd": "#1f4e79",
    "ascad_desync_0": "#d95f02",
    "ascad_desync_50": "#2ca25f",
    "ascad_desync_100": "#756bb1",
}

ZERO_PROXY = 3e-6


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 14,
            "axes.labelsize": 20,
            "legend.fontsize": 13,
            "xtick.labelsize": 16,
            "ytick.labelsize": 16,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def format_axis(ax: plt.Axes) -> None:
    ax.grid(True, linestyle="--", alpha=0.25, axis="y")
    ax.grid(True, linestyle="--", alpha=0.18, axis="x")
    ax.spines["left"].set_linewidth(1.2)
    ax.spines["bottom"].set_linewidth(1.2)
    ax.tick_params(width=1.1, length=5)


def read_summaries(input_dir: Path) -> pd.DataFrame:
    paths = sorted(input_dir.glob("*_full_suite_bi_loc_summary.csv"))
    if not paths:
        raise FileNotFoundError(f"no full-suite summary CSVs under {input_dir}")
    frames = [pd.read_csv(path) for path in paths]
    return pd.concat(frames, ignore_index=True)


def build_compact_table(df: pd.DataFrame) -> pd.DataFrame:
    selected = df[df["epsilon"].astype(float).isin([0.0, 1e-4, 3e-4, 1e-3, 1e-2])].copy()
    cols = [
        "dataset",
        "M",
        "epsilon",
        "n_te",
        "S_loc_max",
        "p_hat_loc_max",
        "p_loc_plus",
        "BI_suite_plus",
        "BI_loc_plus",
        "active_architecture",
        "nonvacuous",
    ]
    return selected[cols].sort_values(["dataset", "epsilon"])


def plot_contrast(df: pd.DataFrame, out_dir: Path) -> None:
    setup_style()
    fig, ax = plt.subplots(figsize=(6.4, 5.8))
    for dataset, group in df.groupby("dataset", sort=False):
        group = group.sort_values("epsilon")
        x = group["epsilon"].astype(float).mask(group["epsilon"].astype(float) == 0.0, ZERO_PROXY)
        ax.plot(
            x,
            group["BI_loc_plus"],
            marker="o",
            linewidth=2.7,
            markersize=8,
            color=COLORS.get(dataset, "#6b7280"),
            label=DATASET_LABELS.get(dataset, dataset),
        )
        suite_y = float(group["BI_suite_plus"].iloc[0])
        ax.axhline(suite_y, linestyle=":", linewidth=1.8, color=COLORS.get(dataset, "#6b7280"), alpha=0.55)

    ax.set_xscale("log")
    ax.set_xlim(2.4e-6, 4.0e-2)
    y_min = max(0.0, float(df["BI_loc_plus"].min()) - 0.25)
    y_max = min(8.25, max(8.25, float(df["BI_loc_plus"].max()) + 0.25))
    ax.set_ylim(y_min, y_max)
    ax.set_xticks([ZERO_PROXY, 1e-5, 1e-4, 1e-3, 1e-2])
    ax.set_xticklabels(["0", r"$10^{-5}$", r"$10^{-4}$", r"$10^{-3}$", r"$10^{-2}$"])
    ax.xaxis.set_minor_locator(LogLocator(base=10.0, subs=tuple(range(2, 10))))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel(r"Score-local radius $\epsilon$")
    ax.set_ylabel(r"$\mathrm{BI}^{\mathrm{loc},+}$ [bits]")
    format_axis(ax)
    ax.legend(loc="upper left", frameon=True, edgecolor="#cccccc", framealpha=0.96)
    fig.tight_layout(pad=0.45)
    fig.savefig(out_dir / "full_suite_bi_loc_contrast.pdf", bbox_inches="tight")
    fig.savefig(out_dir / "full_suite_bi_loc_contrast.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=Path("results/bi_scope_certificates/full_suite_loc"))
    parser.add_argument("--out-dir", type=Path, default=Path("results/bi_scope_certificates/full_suite_loc"))
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary = read_summaries(args.input_dir)
    summary.to_csv(args.out_dir / "full_suite_bi_loc_summary_all.csv", index=False)
    build_compact_table(summary).to_csv(args.out_dir / "full_suite_bi_loc_compact_table.csv", index=False)
    plot_contrast(summary, args.out_dir)
    print(args.out_dir / "full_suite_bi_loc_summary_all.csv")
    print(args.out_dir / "full_suite_bi_loc_contrast.pdf")


if __name__ == "__main__":
    main()
