#!/usr/bin/env python3
"""Plot BI_loc radius curves using the paper figure style."""

from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path

import pandas as pd

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", tempfile.mkdtemp(prefix="bi_loc_mpl_"))
os.environ.setdefault("XDG_CACHE_HOME", tempfile.mkdtemp(prefix="bi_loc_fontcache_"))

import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, NullFormatter


DATASET_LABELS = {
    "ascad_desync_50": "ASCAD d50",
    "ascad_desync_100": "ASCAD d100",
    "aes_rd": "AES-RD",
    "ascad_random_key": "ASCAD random",
}

DATASET_ORDER = ["ascad_desync_50", "ascad_desync_100", "ascad_random_key", "aes_rd"]

COLORS = {
    "MLP": "#1f4e79",
    "CNN": "#d95f02",
    "Transformer": "#2ca25f",
}
ARCH_ORDER = ["MLP", "CNN", "Transformer"]

TICK_SIZE = 16
AXIS_TITLE_SIZE = 20
LEGEND_SIZE = 22
FIG_SIZE = (6.2, 6.2)
PANEL_FIG_SIZE = (10.4, 7.6)
LINE_WIDTH = 2.7
MARKER_SIZE = 9
AUDITED_MARKER_SIZE = 170


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 14,
            "axes.labelsize": AXIS_TITLE_SIZE,
            "legend.fontsize": LEGEND_SIZE * 0.58,
            "xtick.labelsize": TICK_SIZE,
            "ytick.labelsize": TICK_SIZE,
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


def load_audited_points(path: Path | None) -> pd.DataFrame:
    if path is None or not path.exists():
        return pd.DataFrame()
    points = pd.read_csv(path)
    required = {"dataset", "architecture", "audited_epsilon", "BI_loc_plus"}
    missing = required.difference(points.columns)
    if missing:
        raise ValueError(f"{path} is missing required audited-point columns: {sorted(missing)}")
    return points


def audited_x(values: pd.Series, zero_proxy: float) -> pd.Series:
    return values.astype(float).mask(values.astype(float) == 0.0, zero_proxy)


def draw_audited_points(ax: plt.Axes, points: pd.DataFrame, zero_proxy: float) -> None:
    if points.empty:
        return
    for _, point in points.iterrows():
        arch = str(point["architecture"])
        ax.scatter(
            audited_x(pd.Series([point["audited_epsilon"]]), zero_proxy).iloc[0],
            float(point["BI_loc_plus"]),
            marker="*",
            s=AUDITED_MARKER_SIZE,
            color=COLORS.get(arch, "#6b7280"),
            edgecolor="black",
            linewidth=0.75,
            zorder=8,
        )


def iter_arch_groups(panel: pd.DataFrame):
    seen = set(panel["architecture"].astype(str))
    ordered = [arch for arch in ARCH_ORDER if arch in seen]
    ordered.extend([arch for arch in panel["architecture"].drop_duplicates().astype(str) if arch not in ordered])
    for arch in ordered:
        yield arch, panel[panel["architecture"].astype(str) == arch]


def add_audited_legend(ax: plt.Axes) -> None:
    handles, labels = ax.get_legend_handles_labels()
    if "audited radius" not in labels:
        handles.append(
            plt.Line2D(
                [],
                [],
                marker="*",
                linestyle="None",
                markersize=11,
                markerfacecolor="#ffffff",
                markeredgecolor="black",
                label="audited radius",
            )
        )
        labels.append("audited radius")
    ax.legend(handles, labels, loc="lower right", frameon=True, edgecolor="#cccccc", framealpha=0.96, fontsize=11)


def plot(csv_path: Path, out_dir: Path, audited_points_path: Path | None = None) -> None:
    setup_style()
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(csv_path)
    audited_points = load_audited_points(audited_points_path)
    single_dataset = df["dataset"].nunique() == 1

    if not single_dataset:
        plot_panels(df, out_dir, audited_points)
        return

    fig, ax = plt.subplots(figsize=FIG_SIZE)
    zero_proxy = 3e-6

    dataset = str(df["dataset"].iloc[0])
    for arch, group in iter_arch_groups(df):
        group = group.sort_values("epsilon")
        x = group["epsilon"].astype(float).mask(group["epsilon"].astype(float) == 0.0, zero_proxy)
        label = str(arch) if single_dataset else f"{arch}: {DATASET_LABELS.get(dataset, dataset)}"
        ax.plot(
            x,
            group["BI_loc_plus"],
            marker="o",
            color=COLORS.get(arch, "#6b7280"),
            linewidth=LINE_WIDTH,
            markersize=MARKER_SIZE,
            label=label,
        )

    panel_points = (
        audited_points[audited_points["dataset"].astype(str) == dataset]
        if not audited_points.empty
        else pd.DataFrame()
    )
    if not panel_points.empty:
        draw_audited_points(ax, panel_points, zero_proxy)

    ax.set_xscale("log")
    ax.set_xlim(2.4e-6, 4.0e-2)
    y_min = max(0.0, float(df["BI_loc_plus"].min()) - 0.25)
    y_max = min(8.25, max(8.25, float(df["BI_loc_plus"].max()) + 0.25))
    ax.set_ylim(y_min, y_max)
    ax.set_xticks([zero_proxy, 1e-5, 1e-4, 1e-3, 1e-2])
    ax.set_xticklabels(["0", r"$10^{-5}$", r"$10^{-4}$", r"$10^{-3}$", r"$10^{-2}$"])
    ax.xaxis.set_minor_locator(LogLocator(base=10.0, subs=tuple(range(2, 10))))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel(r"Score-local radius $\epsilon$")
    ax.set_ylabel(r"$\mathrm{BI}^{\mathrm{loc},+}$ [bits]")
    format_axis(ax)
    if panel_points.empty:
        ax.legend(loc="upper left", frameon=True, edgecolor="#cccccc", framealpha=0.96)
    else:
        handles, labels = ax.get_legend_handles_labels()
        handles.append(
            plt.Line2D(
                [],
                [],
                marker="*",
                linestyle="None",
                markersize=11,
                markerfacecolor="#ffffff",
                markeredgecolor="black",
                label="audited radius",
            )
        )
        ax.legend(handles, labels + ["audited radius"], loc="upper left", frameon=True, edgecolor="#cccccc", framealpha=0.96)

    fig.tight_layout(pad=0.4)
    fig.savefig(out_dir / "bi_loc_radius_curves.pdf", bbox_inches="tight")
    fig.savefig(out_dir / "bi_loc_radius_curves.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_panels(df: pd.DataFrame, out_dir: Path, audited_points: pd.DataFrame | None = None) -> None:
    audited_points = audited_points if audited_points is not None else pd.DataFrame()
    datasets = [d for d in DATASET_ORDER if d in set(df["dataset"])]
    datasets.extend([d for d in df["dataset"].drop_duplicates() if d not in datasets])
    fig, axes = plt.subplots(2, 2, figsize=PANEL_FIG_SIZE, sharex=True, sharey=True)
    axes_flat = list(axes.ravel())
    zero_proxy = 3e-6
    y_min = max(0.0, float(df["BI_loc_plus"].min()) - 0.25)
    y_max = min(8.25, max(8.25, float(df["BI_loc_plus"].max()) + 0.25))

    for ax, dataset in zip(axes_flat, datasets):
        panel = df[df["dataset"] == dataset]
        for arch, group in iter_arch_groups(panel):
            group = group.sort_values("epsilon")
            x = group["epsilon"].astype(float).mask(group["epsilon"].astype(float) == 0.0, zero_proxy)
            ax.plot(
                x,
                group["BI_loc_plus"],
                marker="o",
                color=COLORS.get(str(arch), "#6b7280"),
                linewidth=LINE_WIDTH,
                markersize=MARKER_SIZE * 0.72,
                label=str(arch),
            )
        panel_points = audited_points[audited_points["dataset"] == dataset] if not audited_points.empty else pd.DataFrame()
        if not panel_points.empty:
            draw_audited_points(ax, panel_points, zero_proxy)
        ax.set_xscale("log")
        ax.set_xlim(2.4e-6, 4.0e-2)
        ax.set_ylim(y_min, y_max)
        ax.set_xticks([zero_proxy, 1e-5, 1e-4, 1e-3, 1e-2])
        ax.set_xticklabels(["0", r"$10^{-5}$", r"$10^{-4}$", r"$10^{-3}$", r"$10^{-2}$"])
        ax.xaxis.set_minor_locator(LogLocator(base=10.0, subs=tuple(range(2, 10))))
        ax.xaxis.set_minor_formatter(NullFormatter())
        format_axis(ax)
        ax.text(
            0.04,
            0.93,
            DATASET_LABELS.get(dataset, dataset),
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=14,
            bbox={"facecolor": "white", "edgecolor": "#dddddd", "boxstyle": "round,pad=0.24", "alpha": 0.96},
        )
        if panel_points.empty:
            ax.legend(loc="lower right", frameon=True, edgecolor="#cccccc", framealpha=0.96, fontsize=11)
        else:
            add_audited_legend(ax)

    for ax in axes_flat[len(datasets) :]:
        ax.axis("off")

    fig.supxlabel(r"Score-local radius $\epsilon$", fontsize=AXIS_TITLE_SIZE)
    fig.supylabel(r"$\mathrm{BI}^{\mathrm{loc},+}$ [bits]", fontsize=AXIS_TITLE_SIZE)
    fig.tight_layout(pad=0.7)
    fig.savefig(out_dir / "bi_loc_radius_curves.pdf", bbox_inches="tight")
    fig.savefig(out_dir / "bi_loc_radius_curves.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("results/bi_scope_certificates/bi_loc_radius_curves.csv"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("results/bi_scope_certificates"),
    )
    parser.add_argument(
        "--audited-points-csv",
        type=Path,
        default=None,
        help="Optional CSV of concrete audited radius points to overlay with star markers.",
    )
    args = parser.parse_args()
    plot(args.csv, args.out_dir, args.audited_points_csv)
    print(args.out_dir / "bi_loc_radius_curves.pdf")
    print(args.out_dir / "bi_loc_radius_curves.png")


if __name__ == "__main__":
    main()
