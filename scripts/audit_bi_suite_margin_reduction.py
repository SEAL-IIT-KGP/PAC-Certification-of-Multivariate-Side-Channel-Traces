#!/usr/bin/env python3
"""Audit ways to reduce BI-suite finite-sample margin slack.

This script compares the current table against two non-mutating alternatives:

1. pooling the reusable seed rows before computing the KL endpoint; and
2. estimating the additional attack-set size needed to reach target
   R_margin values at the observed success rate.

The pooled rows are diagnostic unless the seed-level holdout splits are known
to be independent/disjoint. The reusable seed-row generator redraws random splits
from the same finite dataset, so these numbers should not replace the paper
table without a separate independence/provenance argument.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd


N_CLASSES = 256


def bernoulli_kl(p: float, q: float) -> float:
    if p == q:
        return 0.0
    if q <= 0.0:
        return 0.0 if p <= 0.0 else math.inf
    if q >= 1.0:
        return 0.0 if p >= 1.0 else math.inf
    out = 0.0
    if p > 0.0:
        out += p * math.log(p / q)
    if p < 1.0:
        out += (1.0 - p) * math.log((1.0 - p) / (1.0 - q))
    return out


def kl_endpoint(successes: int, n: int, c: float, *, upper: bool) -> float:
    p_hat = successes / float(n)
    if upper:
        if successes >= n:
            return 1.0
        lo, hi = p_hat, 1.0
        for _ in range(90):
            mid = (lo + hi) / 2.0
            if bernoulli_kl(p_hat, mid) <= c:
                lo = mid
            else:
                hi = mid
        return lo

    if successes <= 0:
        return 0.0
    lo, hi = 0.0, p_hat
    for _ in range(90):
        mid = (lo + hi) / 2.0
        if bernoulli_kl(p_hat, mid) <= c:
            hi = mid
        else:
            lo = mid
    return hi


def bi_bits(p: float, n_classes: int = N_CLASSES) -> float:
    return max(0.0, math.log2(max(p, 0.0) * n_classes))


def r_margin(p_obs: float, p_plus: float, n_classes: int = N_CLASSES) -> float:
    denom = p_obs - (1.0 / n_classes)
    slack = p_plus - p_obs
    return math.inf if denom <= 0.0 and slack > 0.0 else slack / max(denom, 1e-300)


def suite_row_from_counts(
    dataset: str,
    scope_id: str,
    counts: dict[str, int],
    n_te: int,
    delta: float,
    n_classes: int,
) -> dict[str, Any]:
    m = len(counts)
    c = math.log((2.0 * m) / delta) / float(n_te)
    best_model, best_successes = max(counts.items(), key=lambda kv: kv[1])
    lowers = [kl_endpoint(s, n_te, c, upper=False) for s in counts.values()]
    uppers = [kl_endpoint(s, n_te, c, upper=True) for s in counts.values()]
    p_obs = best_successes / float(n_te)
    p_minus = max(lowers)
    p_plus = max(uppers)
    return {
        "dataset": dataset,
        "scope_id": scope_id,
        "M": m,
        "n_te": n_te,
        "p_obs": p_obs,
        "p_minus": p_minus,
        "p_plus": p_plus,
        "BI_minus": bi_bits(p_minus, n_classes),
        "BI_obs": bi_bits(p_obs, n_classes),
        "BI_plus": bi_bits(p_plus, n_classes),
        "slack": p_plus - p_obs,
        "R_margin": r_margin(p_obs, p_plus, n_classes),
        "best_model": best_model,
    }


def parse_success_json(value: Any) -> dict[str, float]:
    if isinstance(value, dict):
        return {str(k): float(v) for k, v in value.items()}
    if not isinstance(value, str) or not value.strip():
        return {}
    return {str(k): float(v) for k, v in json.loads(value).items()}


def pooled_reusable_rows(
    reusable_suite_path: Path,
    datasets: set[str],
    delta: float,
    n_classes: int,
) -> list[dict[str, Any]]:
    df = pd.read_csv(reusable_suite_path)
    rows: list[dict[str, Any]] = []
    for dataset, group in df[df["dataset"].isin(datasets)].groupby("dataset", sort=True):
        if group.empty:
            continue
        n_values = sorted({int(x) for x in group["n_holdout"].dropna().unique()})
        model_sets = [set(parse_success_json(v).keys()) for v in group["per_model_success"]]
        common_models = set.intersection(*model_sets) if model_sets else set()
        union_models = set.union(*model_sets) if model_sets else set()
        if not common_models or common_models != union_models:
            rows.append(
                {
                    "dataset": dataset,
                    "pooled_status": "not_poolable_model_set_mismatch",
                    "n_seed_rows": len(group),
                    "model_intersection": len(common_models),
                    "model_union": len(union_models),
                }
            )
            continue

        counts = {model: 0 for model in sorted(common_models)}
        n_total = 0
        for record in group.itertuples(index=False):
            n = int(record.n_holdout)
            n_total += n
            success = parse_success_json(record.per_model_success)
            for model in counts:
                counts[model] += int(round(float(success[model]) * n))
        row = suite_row_from_counts(
            dataset=str(dataset),
            scope_id="pooled_seed_diagnostic_" + str(group["dataset"].iloc[0]),
            counts=counts,
            n_te=n_total,
            delta=delta,
            n_classes=n_classes,
        )
        row.update(
            {
                "pooled_status": (
                    "diagnostic_only_random_splits_reuse_same_underlying_dataset"
                ),
                "n_seed_rows": int(len(group)),
                "per_seed_n_te_values": ";".join(map(str, n_values)),
                "model_intersection": len(common_models),
                "model_union": len(union_models),
            }
        )
        rows.append(row)
    return rows


def p_plus_from_rate(p_obs: float, n_te: int, m: int, delta: float) -> float:
    successes = int(round(p_obs * n_te))
    c = math.log((2.0 * m) / delta) / float(n_te)
    return kl_endpoint(successes, n_te, c, upper=True)


def needed_n(p_obs: float, m: int, delta: float, target_r: float, n_classes: int) -> int | None:
    if p_obs <= 1.0 / n_classes:
        return None

    def current_r(n: int) -> float:
        p_plus = p_plus_from_rate(p_obs, n, m, delta)
        return r_margin(p_obs, p_plus, n_classes)

    hi = 1
    while current_r(hi) > target_r:
        hi *= 2
        if hi > 10_000_000_000:
            return None
    lo = hi // 2
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        if current_r(mid) <= target_r:
            hi = mid
        else:
            lo = mid
    return hi


def build_comparison(current: pd.DataFrame, pooled: pd.DataFrame, delta: float, n_classes: int) -> pd.DataFrame:
    pooled_by_dataset = {str(r["dataset"]): r for _, r in pooled.iterrows()} if not pooled.empty else {}
    rows: list[dict[str, Any]] = []
    for record in current.itertuples(index=False):
        dataset = str(record.dataset)
        base = {
            "dataset": dataset,
            "scope_id": str(record.scope_id),
            "M": int(record.M),
            "current_n_te": int(record.n_te),
            "current_p_obs": float(record.p_obs),
            "current_p_plus": float(record.p_plus),
            "current_BI_plus": float(record.BI_plus),
            "current_R_margin": float(record.R_margin),
            "n_for_R_le_1_at_current_p": needed_n(
                float(record.p_obs), int(record.M), delta, 1.0, n_classes
            ),
            "n_for_R_le_0p5_at_current_p": needed_n(
                float(record.p_obs), int(record.M), delta, 0.5, n_classes
            ),
        }
        pooled_row = pooled_by_dataset.get(dataset)
        if pooled_row is not None and "R_margin" in pooled_row and not pd.isna(pooled_row["R_margin"]):
            base.update(
                {
                    "pooled_status": pooled_row["pooled_status"],
                    "pooled_seed_rows": int(pooled_row["n_seed_rows"]),
                    "pooled_n_te": int(pooled_row["n_te"]),
                    "pooled_p_obs": float(pooled_row["p_obs"]),
                    "pooled_p_plus": float(pooled_row["p_plus"]),
                    "pooled_BI_plus": float(pooled_row["BI_plus"]),
                    "pooled_R_margin": float(pooled_row["R_margin"]),
                    "delta_R_margin": float(pooled_row["R_margin"]) - float(record.R_margin),
                    "pooled_best_model": str(pooled_row["best_model"]),
                }
            )
        else:
            base.update(
                {
                    "pooled_status": (
                        pooled_row.get("pooled_status", "no_reusable_seed_pool")
                        if pooled_row is not None
                        else "no_reusable_seed_pool"
                    ),
                    "pooled_seed_rows": (
                        int(pooled_row.get("n_seed_rows", 0)) if pooled_row is not None else 0
                    ),
                    "pooled_n_te": math.nan,
                    "pooled_p_obs": math.nan,
                    "pooled_p_plus": math.nan,
                    "pooled_BI_plus": math.nan,
                    "pooled_R_margin": math.nan,
                    "delta_R_margin": math.nan,
                    "pooled_best_model": "",
                }
            )
        rows.append(base)
    return pd.DataFrame(rows)


def write_markdown(path: Path, comparison: pd.DataFrame) -> None:
    high = comparison[comparison["current_R_margin"] > 1.0].copy()
    cols = [
        "dataset",
        "current_R_margin",
        "current_n_te",
        "pooled_status",
        "pooled_n_te",
        "pooled_R_margin",
        "n_for_R_le_1_at_current_p",
        "n_for_R_le_0p5_at_current_p",
    ]
    lines = [
        "# BI-suite margin reduction audit",
        "",
        "This audit does not overwrite the manuscript table. It compares the current",
        "`R_margin` values against pooled seed-row diagnostics and sample-size targets.",
        "",
        "Important provenance note: `dimension_stability/multivariate_stability_driver.py` forms each seed row by",
        "drawing a fresh random train/holdout split from the same finite dataset. Those",
        "seed rows are therefore diagnostic only unless a separate provenance check",
        "establishes that the evaluated attack examples are independent/disjoint.",
        "",
        "## High-margin rows",
        "",
        high[cols].to_markdown(index=False, floatfmt=".4g"),
        "",
        "## Recommendation",
        "",
        "Use the pooled rows only as a direction-finding diagnostic. For a paper-safe",
        "reduction, run a fresh fixed-suite evaluation on a larger declared attack set",
        "or prove that the pooled seed rows correspond to independent/disjoint attack",
        "examples. DPAv4 is the cheapest target: the current observed rate only needs",
        "about 21.5k attack examples for `R_margin <= 1` at `M=80`, versus 2.25k now.",
    ]
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--current-table",
        default=(
            "results/bi_scope_certificates/mainpaper_package/"
            "data/table1_bi_suite_eight_benchmarks.csv"
        ),
        type=Path,
    )
    parser.add_argument(
        "--reusable-suite-rows",
        default="results/reusable_suite_rows/reusable_suite_success.csv",
        type=Path,
    )
    parser.add_argument(
        "--out-dir",
        default="results/bi_margin_reduction_audit",
        type=Path,
    )
    parser.add_argument("--delta", default=1e-6, type=float)
    parser.add_argument("--n-classes", default=256, type=int)
    args = parser.parse_args()

    current = pd.read_csv(args.current_table)
    reusable = set(current[current["source_artifact"].astype(str).str.contains("reusable_suite")]["dataset"])
    pooled_rows = pooled_reusable_rows(args.reusable_suite_rows, reusable, args.delta, args.n_classes)
    pooled = pd.DataFrame(pooled_rows)
    comparison = build_comparison(current, pooled, args.delta, args.n_classes)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    pooled.to_csv(args.out_dir / "pooled_seed_diagnostic_rows.csv", index=False, quoting=csv.QUOTE_MINIMAL)
    comparison.to_csv(args.out_dir / "margin_reduction_comparison.csv", index=False, quoting=csv.QUOTE_MINIMAL)
    write_markdown(args.out_dir / "MARGIN_REDUCTION_AUDIT.md", comparison)

    display_cols = [
        "dataset",
        "current_R_margin",
        "pooled_R_margin",
        "pooled_status",
        "n_for_R_le_1_at_current_p",
    ]
    print(comparison[display_cols].to_string(index=False))
    print(f"Wrote {args.out_dir}")


if __name__ == "__main__":
    main()
