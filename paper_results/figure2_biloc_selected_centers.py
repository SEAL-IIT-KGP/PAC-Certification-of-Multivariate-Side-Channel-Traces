#!/usr/bin/env python3
"""Build Figure 2: selected-center local-family curves and the calibrated radius.

Twelve reference models (MLP, CNN, Transformer on ASCAD d50, ASCAD d100, ASCAD
random key, and AES-RD) are profiled by `bi_scope_certificates.profile_epsilon_calibration`.
Each attack set is split once (seed 20260721) into 20% calibration traces and
80% certification traces. This script

1. applies the calibration rule on the calibration split to select the radius
   (the first grid radius that changes the relaxed-margin count of at least 75%
   of the reference models and covers the observed FP32 score variation), and
2. computes BI_loc,+(eps) on the certification split with a one-sided
   KL-binomial endpoint and the confidence split delta / (12 * |grid|).

Counts are read from `paper_results/archived/selected_center_relaxed_margin_counts.csv`
and `selected_center_score_variation.csv`, or from the `epsilon_margin_counts.csv`
and `epsilon_variant_distances.csv` files found under `--recomputed-dir`.

Section 5.3 also quotes the twelve bounds at eps = 1e-4. Those values use every
attack trace of each reference model (no calibration split) with the confidence
split delta / |grid| per model, as computed by
`bi_scope_certificates.compute_bi_loc_margin_curves`. They are checked here from
`paper_results/archived/selected_center_full_attack_relaxed_margin_counts.csv`.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import pandas as pd

from bi_scope_certificates.compute_bi_loc_margin_curves import bi_from_p, kl_upper
from paper_results.common import ARCHIVED, DEFAULT_OUT, DELTA, EPS_GRID, EXPECTED, PaperCheck, read_rows

COVERAGE_FRACTION = 0.75


def load_counts(recomputed_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, str]]:
    """Relaxed-margin counts (recomputed where available) and score-variation profiles.

    The archived score variation was measured on the GPU runs behind the paper,
    where Transformer scores change with the batch size by up to 1.9e-4. It is
    always kept, with `run = archived`; the variation measured by a recomputation
    is added with `run = recomputed`. The calibrated radius covers both.
    """
    counts = pd.read_csv(ARCHIVED / "selected_center_relaxed_margin_counts.csv")
    distances = pd.read_csv(ARCHIVED / "selected_center_score_variation.csv")
    distances["run"] = "archived"
    sources = {dataset: "archived" for dataset in counts["dataset"].unique()}
    if recomputed_dir.exists():
        for margin_path in sorted(recomputed_dir.rglob("epsilon_margin_counts.csv")):
            fresh = pd.read_csv(margin_path)
            fresh_distances = pd.read_csv(margin_path.with_name("epsilon_variant_distances.csv"))
            for dataset in fresh["dataset"].unique():
                archived_centers = set(counts.loc[counts["dataset"] == dataset, "center_id"])
                fresh_centers = set(fresh.loc[fresh["dataset"] == dataset, "center_id"])
                # Replace only the reference models that were recomputed.
                counts = counts[~((counts["dataset"] == dataset) & counts["center_id"].isin(fresh_centers))]
                sources[dataset] = (
                    "recomputed" if fresh_centers >= archived_centers else "recomputed (public models) + archived"
                )
            counts = pd.concat([counts, fresh[counts.columns.intersection(fresh.columns)]], ignore_index=True)
            fresh_distances = fresh_distances[distances.columns.intersection(fresh_distances.columns)].copy()
            fresh_distances["run"] = "recomputed"
            distances = pd.concat([distances, fresh_distances], ignore_index=True)
    return counts, distances, sources


def select_radius(counts: pd.DataFrame, distances: pd.DataFrame) -> dict[str, object]:
    calibration = counts[counts["split"] == "calibration"]
    by_center_eps = {(r.center_id, float(r.epsilon)): int(r.successes) for r in calibration.itertuples()}
    centers = sorted(calibration["center_id"].unique())
    resolution = []
    resolved = None
    for eps in EPS_GRID[1:]:
        changed = sum(by_center_eps[(center, eps)] != by_center_eps[(center, 0.0)] for center in centers)
        resolution.append({"epsilon": eps, "centers_changed": changed, "centers_total": len(centers)})
        if resolved is None and changed / len(centers) >= COVERAGE_FRACTION:
            resolved = eps
    fp32 = distances[distances["variant_group"] == "fp32_repro"]
    fp32_by_run = {run: float(group["distance_max"].max()) for run, group in fp32.groupby("run")}
    fp32_max = max(fp32_by_run.values())
    covering = next((eps for eps in EPS_GRID[1:] if fp32_max <= eps), None)
    selected = None if resolved is None or covering is None else max(resolved, covering)
    return {
        "resolution": resolution,
        "resolved_epsilon": resolved,
        "fp32_max_distance": fp32_max,
        "fp32_max_distance_by_run": fp32_by_run,
        "fp32_covering_epsilon": covering,
        "selected_epsilon": selected,
    }


def full_attack_curves(delta: float) -> pd.DataFrame:
    curves = pd.read_csv(ARCHIVED / "selected_center_full_attack_relaxed_margin_counts.csv")
    alpha = delta / curves["epsilon"].nunique()
    curves["p_plus"] = [kl_upper(int(s), int(n), alpha) for s, n in zip(curves["S_loc"], curves["n_te"])]
    curves["BI_loc_plus"] = [bi_from_p(p) for p in curves["p_plus"]]
    curves["delta_effective"] = alpha
    return curves


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--recomputed-dir", type=Path, default=DEFAULT_OUT / "recomputed")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--delta", type=float, default=DELTA)
    parser.add_argument("--no-plot", action="store_true", help="Write the CSV files only.")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    counts, distances, sources = load_counts(args.recomputed_dir)
    centers = sorted(counts["center_id"].unique())
    if len(centers) != 12:
        raise SystemExit(f"expected 12 reference models, found {len(centers)}")
    selection = select_radius(counts, distances)
    selected = selection["selected_epsilon"]
    print("Calibration split: reference models whose relaxed-margin count changes at each radius")
    for row in selection["resolution"]:
        print(f"  eps={row['epsilon']:<8g} {row['centers_changed']}/{row['centers_total']}")
    print(f"First radius changing >= {COVERAGE_FRACTION:.0%} of the models: {selection['resolved_epsilon']}")
    labels = {"archived": "the paper's GPU runs (archived)", "recomputed": "this recomputation"}
    for run, value in selection["fp32_max_distance_by_run"].items():
        print(f"Largest FP32 repeat/batch score distance, {labels.get(run, run)}: {value:.3e}")
    print(f"Smallest grid radius covering it: {selection['fp32_covering_epsilon']}")
    print(f"Calibrated radius: {selected}")

    alpha = args.delta / (len(centers) * len(EPS_GRID))
    evaluation = counts[counts["split"] == "evaluation"].copy()
    evaluation["p_plus"] = [kl_upper(int(s), int(n), alpha) for s, n in zip(evaluation["successes"], evaluation["n"])]
    evaluation["BI_loc_plus"] = [bi_from_p(p) for p in evaluation["p_plus"]]
    evaluation["delta_effective"] = alpha
    evaluation["counts_source"] = evaluation["dataset"].map(sources)
    evaluation = evaluation.sort_values(["dataset", "architecture", "epsilon"])
    curves_path = args.out_dir / "figure2_biloc_selected_centers.csv"
    evaluation.to_csv(curves_path, index=False)

    at_selected = evaluation[[math.isclose(float(e), float(selected)) for e in evaluation["epsilon"]]]
    points = at_selected[["dataset", "architecture", "center_id", "n", "successes", "BI_loc_plus"]].copy()
    points.insert(3, "audited_epsilon", float(selected))
    points_path = args.out_dir / "figure2_calibrated_radius_points.csv"
    points.to_csv(points_path, index=False)
    print()
    print(points.to_string(index=False, float_format=lambda v: f"{v:.3f}" if v > 1e-3 else f"{v:g}"))
    print()

    any_recomputed = any(source != "archived" for source in sources.values())
    slack = 0.01 if any_recomputed else 0.0
    check = PaperCheck("Figure 2 (selected-center local-family certificates)")
    check.number("calibrated radius", float(selected), "0.0003")
    full_attack = full_attack_curves(args.delta)
    full_attack.to_csv(args.out_dir / "figure2_biloc_selected_centers_full_attack_set.csv", index=False)
    for printed in read_rows(EXPECTED / "figure2_selected_centers.csv"):
        eps = float(printed["epsilon"])
        curves = evaluation if printed["traces"] == "certification_split" else full_attack
        at_eps = curves[[math.isclose(float(e), eps) for e in curves["epsilon"]]]
        if printed["claim"] == "point":
            row = at_eps[
                (at_eps["dataset"] == printed["dataset"]) & (at_eps["architecture"] == printed["architecture"])
            ]
            value = float(row["BI_loc_plus"].iloc[0])
            label = f"{printed['dataset']} {printed['architecture']} at eps={eps:g} ({printed['location']})"
        else:
            ascad = at_eps[at_eps["dataset"].str.startswith("ascad")]["BI_loc_plus"]
            value = float(ascad.min() if printed["claim"] == "ascad_minimum" else ascad.max())
            label = f"{printed['claim']} at eps={eps:g} ({printed['location']})"
        check.number(
            label, value, printed["BI_loc_plus"], slack=slack if printed["traces"] == "certification_split" else 0.0
        )
    nonvacuous = int((at_selected["p_plus"] < 1.0).sum())
    check.exact("non-vacuous certificates at the calibrated radius", nonvacuous, 12)

    if not args.no_plot:
        from scripts import plot_bi_loc_radius_curves as plotter

        plotter.POINT_LABEL = "calibrated radius"
        plotter.OUTPUT_STEM = "figure2_biloc_selected_centers"
        plotter.plot(curves_path, args.out_dir, points_path)
        print(args.out_dir / "figure2_biloc_selected_centers.pdf")
    raise SystemExit(0 if check.report() else 1)


if __name__ == "__main__":
    main()
