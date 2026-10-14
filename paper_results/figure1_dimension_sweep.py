#!/usr/bin/env python3
"""Build Figure 1: CHES-CTF-2025 dimension sweep of BI_suite,+ and the density-style estimators.

BI_suite,+ is recomputed for every trace window (100 to 7,000 samples) from the
per-checkpoint attack success counts of the 11-member EstraNet suite trained on
that window (two-sided KL-binomial interval, delta / (2M)).

- Figure 1 (`figure1_dimension_sweep.pdf`): the BI line at delta = 0.05, as in the
  paper. The check compares the six plotted BI values and the range quoted in
  the text (0.3 to 0.4 bits).
- Additional result (`figure1_dimension_sweep_delta1e-6.pdf`): the same counts at
  delta = 1e-6, the confidence level of Table 7; its 3,000-sample value is the
  CHES-CTF-2025 row of Table 7.

The GKOV MI, MLP-PI and ePI/PI values are the measured means (three seeds) in
`paper_results/archived/ches_dimension_sweep_nonbi_estimators.csv`, replaced by the
cells that `dimension_stability/ches_nonbi_metric_cell.py` recomputes from the
CHES-CTF-2025 profiling traces.
"""

from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path

from bi_suite_8datasets.evaluate_tches20_pretrained_fixed_split import summarize_counts
from paper_results.common import ARCHIVED, DEFAULT_OUT, DELTA, EXPECTED, PaperCheck, read_rows, write_rows

DIMENSIONS = [100, 200, 500, 1000, 3000, 7000]
ESTIMATORS = [("GKOV_MI", "GKOV MI", "#2ca25f"), ("MLP_PI", "MLP-PI", "#7570b3"), ("ePI_or_PI", "ePI/PI", "#c51b8a")]
BI_COLOR = "#d95f02"
# Confidence level of the BI line in Figure 1 of the paper; Table 7 uses DELTA (1e-6).
FIGURE1_DELTA = 0.05


def window_counts(recomputed_dir: Path) -> tuple[dict[int, list[dict[str, object]]], dict[int, str]]:
    by_dim: dict[int, list[dict[str, object]]] = {}
    for row in read_rows(ARCHIVED / "ches_window_success_counts.csv"):
        by_dim.setdefault(int(row["dimension"]), []).append(
            {
                "dataset": "ches_ctf_2025",
                "split": "attack",
                "model_id": row["model_id"],
                "n_te": int(row["n_te"]),
                "successes": int(row["successes"]),
            }
        )
    sources = {dim: "archived" for dim in by_dim}
    if recomputed_dir.exists():
        for dim in DIMENSIONS:
            matches = sorted(recomputed_dir.rglob(f"ches_ctf_2025_estranet_w{dim}_success.csv"))
            if matches:
                reference = {str(r["model_id"]): int(r["successes"]) for r in by_dim[dim]}
                by_dim[dim] = [
                    {
                        "dataset": "ches_ctf_2025",
                        "split": "attack",
                        "model_id": r["model_id"],
                        "n_te": int(r["n_te"]),
                        "successes": int(r["successes"]),
                    }
                    for r in read_rows(matches[0])
                ]
                sources[dim] = "recomputed"
                same = sum(1 for r in by_dim[dim] if reference.get(str(r["model_id"])) == int(r["successes"]))
                largest = max(abs(int(r["successes"]) - reference.get(str(r["model_id"]), 0)) for r in by_dim[dim])
                print(
                    f"window {dim}: {same}/{len(by_dim[dim])} recomputed per-checkpoint counts equal the archived counts (largest difference {largest} traces)"
                )
    return by_dim, sources


def plot(estimators: dict[str, dict[int, tuple[float, str]]], bi_line: dict[int, float], stem: Path) -> None:
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("MPLCONFIGDIR", tempfile.mkdtemp(prefix="figure1_mpl_"))
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 14,
            "axes.labelsize": 20,
            "xtick.labelsize": 16,
            "ytick.labelsize": 16,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    fig, ax = plt.subplots(figsize=(6.2, 6.0))
    for key, label, color in ESTIMATORS:
        ax.plot(
            DIMENSIONS,
            [estimators[key][dim][0] for dim in DIMENSIONS],
            color=color,
            linewidth=2.4,
            marker="o",
            markersize=7,
            label=label,
            zorder=4,
        )
    ax.plot(
        DIMENSIONS,
        [bi_line[dim] for dim in DIMENSIONS],
        color=BI_COLOR,
        linewidth=3.0,
        marker="o",
        markersize=10,
        label=r"$\mathrm{BI}^{\mathrm{suite},+}$",
        zorder=5,
    )
    ax.axhline(0.0, color="black", linewidth=1.0)
    ax.set_xscale("log")
    ax.set_xticks(DIMENSIONS)
    ax.set_xticklabels([str(dim) for dim in DIMENSIONS], rotation=35, ha="right")
    ax.minorticks_off()
    lowest = min(estimators[key][dim][0] for key, _, _ in ESTIMATORS for dim in DIMENSIONS)
    ax.set_ylim(min(-1.25, lowest - 0.1), 0.95)
    ax.set_xlabel(r"Trace dimension $d'$")
    ax.set_ylabel("Bits")
    ax.grid(True, linestyle="--", alpha=0.25, axis="y")
    ax.legend(loc="lower left", frameon=True, edgecolor="#cccccc", framealpha=0.96, fontsize=12)
    fig.tight_layout(pad=0.4)
    fig.savefig(f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(f"{stem}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"{stem}.pdf")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--recomputed-dir", type=Path, default=DEFAULT_OUT / "recomputed")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--no-plot", action="store_true", help="Write the CSV file only.")
    args = parser.parse_args()

    counts, sources = window_counts(args.recomputed_dir)
    rows = []
    bi_lines: dict[float, dict[int, float]] = {}
    for delta in (FIGURE1_DELTA, DELTA):
        bi_lines[delta] = {}
        for dim in DIMENSIONS:
            summary = summarize_counts(counts[dim], delta)
            bi_lines[delta][dim] = float(summary["BI_plus"])
            rows.append(
                {
                    "dimension": dim,
                    "metric": "BI_suite_plus",
                    "delta": delta,
                    "value_bits": summary["BI_plus"],
                    "status": f"certificate from {sources[dim]} counts",
                    "M": summary["M"],
                    "n_te": summary["n_te"],
                    "successes": max(int(r["successes"]) for r in counts[dim]),
                }
            )
    bi_figure, bi_table7 = bi_lines[FIGURE1_DELTA], bi_lines[DELTA]
    estimators: dict[str, dict[int, tuple[float, str]]] = {}
    for row in read_rows(ARCHIVED / "ches_dimension_sweep_nonbi_estimators.csv"):
        estimators.setdefault(row["metric"], {})[int(row["dimension"])] = (float(row["value_bits"]), "archived")
    # Estimator cells recomputed by dimension_stability.ches_nonbi_metric_cell replace the archived means.
    fresh_cells = (
        sorted(args.recomputed_dir.rglob("ches_nonbi_highdim_checkpoint.csv")) if args.recomputed_dir.exists() else []
    )
    if fresh_cells:
        by_cell: dict[tuple[str, int], list[float]] = {}
        for row in read_rows(fresh_cells[0]):
            if row.get("status") == "complete" and row.get("metric") in estimators:
                by_cell.setdefault((row["metric"], int(float(row["dimension"]))), []).append(float(row["value"]))
        for (metric, dim), values in sorted(by_cell.items()):
            mean = sum(values) / len(values)
            estimators[metric][dim] = (mean, f"recomputed ({len(values)} seeds)")
            print(f"{metric} at d={dim}: recomputed mean {mean:.3f} from {len(values)} seed(s)")
    for metric, cells in estimators.items():
        missing = [dim for dim in DIMENSIONS if dim not in cells]
        if missing:
            raise SystemExit(f"{metric}: no value for dimension(s) {missing}")
        for dim, (value, status) in sorted(cells.items()):
            rows.append(
                {
                    "dimension": dim,
                    "metric": metric,
                    "delta": "",
                    "value_bits": value,
                    "status": status,
                    "M": "",
                    "n_te": "",
                    "successes": "",
                }
            )
    write_rows(args.out_dir / "figure1_dimension_sweep.csv", rows)

    print(
        f"dimension  BI+ ({FIGURE1_DELTA:g})  BI+ ({DELTA:g})   "
        + "   ".join(f"{label:>10}" for _, label, _ in ESTIMATORS)
    )
    for dim in DIMENSIONS:
        cells = [f"{estimators[key][dim][0]:>9.3f} " for key, _, _ in ESTIMATORS]
        print(f"{dim:>9}  {bi_figure[dim]:>10.3f}  {bi_table7[dim]:>11.3f}   " + "   ".join(cells))

    expected = {row["claim"]: row["value_bits"] for row in read_rows(EXPECTED / "figure1_dimension_sweep.csv")}
    # Recomputed window counts are compared to 0.01 bits.
    slack = 0.01 if any(source == "recomputed" for source in sources.values()) else 0.0
    check = PaperCheck("Figure 1 (dimension sweep)")
    for dim in DIMENSIONS:
        check.number(f"BI_suite,+ at d={dim}", bi_figure[dim], expected[f"bi_suite_plus_d{dim}"], slack=slack)
    check.number("smallest BI_suite,+ over the sweep", min(bi_figure.values()), expected["bi_suite_plus_minimum"])
    check.number("largest BI_suite,+ over the sweep", max(bi_figure.values()), expected["bi_suite_plus_maximum"])
    gkov = [estimators["GKOV_MI"][dim][0] for dim in DIMENSIONS]
    check.exact("GKOV MI is negative at the largest dimension", gkov[-1] < 0, True)
    check.exact("BI_suite,+ is positive at every dimension", all(v > 0 for v in bi_figure.values()), True)
    extra = PaperCheck("Figure 1 at delta = 1e-6 (additional result, confidence level of Table 7)")
    extra.number(
        "BI_suite,+ at d=3000 (Table 7)", bi_table7[3000], expected["table7_bi_suite_plus_at_3000"], slack=slack
    )
    extra.exact("BI_suite,+ is positive at every dimension", all(v > 0 for v in bi_table7.values()), True)

    if not args.no_plot:
        plot(estimators, bi_figure, args.out_dir / "figure1_dimension_sweep")
        plot(estimators, bi_table7, args.out_dir / "figure1_dimension_sweep_delta1e-6")
    passed = [check.report(), extra.report()]
    raise SystemExit(0 if all(passed) else 1)


if __name__ == "__main__":
    main()
