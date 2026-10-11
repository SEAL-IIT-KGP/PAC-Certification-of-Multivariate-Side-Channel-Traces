#!/usr/bin/env python3
"""Build Figure 1: CHES-CTF-2025 dimension sweep of BI_suite,+ and the density-style estimators.

BI_suite,+ is recomputed for every trace window (100 to 7,000 samples) from the
per-checkpoint attack success counts of the 11-member EstraNet suite trained on
that window (two-sided KL-binomial interval, delta / (2M)).

The GKOV MI, MLP-PI, and ePI/PI values are read from
`paper_results/archived/ches_dimension_sweep_nonbi_estimators.csv`.
`dimension_stability/ches_nonbi_metric_cell.py` measures these cells from the
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--recomputed-dir", type=Path, default=DEFAULT_OUT / "recomputed")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--delta", type=float, default=DELTA)
    parser.add_argument("--no-plot", action="store_true", help="Write the CSV file only.")
    args = parser.parse_args()

    counts, sources = window_counts(args.recomputed_dir)
    rows = []
    bi_plus = {}
    for dim in DIMENSIONS:
        summary = summarize_counts(counts[dim], args.delta)
        bi_plus[dim] = float(summary["BI_plus"])
        rows.append(
            {
                "dimension": dim,
                "metric": "BI_suite_plus",
                "value_bits": summary["BI_plus"],
                "status": f"certificate from {sources[dim]} counts",
                "M": summary["M"],
                "n_te": summary["n_te"],
                "successes": max(int(r["successes"]) for r in counts[dim]),
            }
        )
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
            archived_value, _ = estimators[metric].get(dim, (float("nan"), ""))
            mean = sum(values) / len(values)
            estimators[metric][dim] = (mean, f"recomputed ({len(values)} seeds)")
            print(
                f"{metric} at d={dim}: recomputed mean {mean:.3f} from {len(values)} seed(s), archived {archived_value:.3f}"
            )
    for metric, cells in estimators.items():
        for dim, (value, status) in sorted(cells.items()):
            rows.append(
                {
                    "dimension": dim,
                    "metric": metric,
                    "value_bits": value,
                    "status": status,
                    "M": "",
                    "n_te": "",
                    "successes": "",
                }
            )
    write_rows(args.out_dir / "figure1_dimension_sweep.csv", rows)

    print("dimension  BI_suite,+   " + "   ".join(f"{label:>10}" for _, label, _ in ESTIMATORS))
    for dim in DIMENSIONS:
        cells = [f"{estimators[key][dim][0]:>9.3f} " for key, _, _ in ESTIMATORS]
        print(f"{dim:>9}  {bi_plus[dim]:>10.3f}   " + "   ".join(cells))

    expected = {row["claim"]: row["value_bits"] for row in read_rows(EXPECTED / "figure1_dimension_sweep.csv")}
    check = PaperCheck("Figure 1 (dimension sweep)")
    # Recomputed window counts are compared to 0.01 bits.
    slack = 0.01 if any(source == "recomputed" for source in sources.values()) else 0.0
    check.number(
        "smallest BI_suite,+ over the sweep", min(bi_plus.values()), expected["bi_suite_plus_minimum"], slack=slack
    )
    check.number(
        "largest BI_suite,+ over the sweep", max(bi_plus.values()), expected["bi_suite_plus_maximum"], slack=slack
    )
    check.number("BI_suite,+ at d=3000", bi_plus[3000], expected["bi_suite_plus_at_3000"], slack=slack)
    gkov = [estimators["GKOV_MI"][dim][0] for dim in DIMENSIONS]
    check.exact("GKOV MI is negative at the largest dimension", gkov[-1] < 0, True)
    check.exact("BI_suite,+ is positive at every dimension", all(v > 0 for v in bi_plus.values()), True)

    if not args.no_plot:
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
            [bi_plus[dim] for dim in DIMENSIONS],
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
        fig.savefig(args.out_dir / "figure1_dimension_sweep.pdf", bbox_inches="tight")
        fig.savefig(args.out_dir / "figure1_dimension_sweep.png", dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(args.out_dir / "figure1_dimension_sweep.pdf")
    raise SystemExit(0 if check.report() else 1)


if __name__ == "__main__":
    main()
