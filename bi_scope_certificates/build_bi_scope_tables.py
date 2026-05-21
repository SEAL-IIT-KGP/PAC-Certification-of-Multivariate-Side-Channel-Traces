#!/usr/bin/env python3
"""Build BI^sing / BI^suite / BI^loc reporting artifacts.

This script is intentionally postprocessing-only.  The expensive exact
local-margin curves are produced by ``compute_bi_loc_margin_curves.py``; this
builder merges those outputs with the current eight-dataset suite artifacts and
writes paper-facing CSV/Markdown summaries.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Iterable

import pandas as pd


K = 256
CHANCE = 1.0 / K
DATASET_ORDER = {
    "ascad_desync_50": 0,
    "ascad_desync_100": 1,
    "ascad_random_key": 2,
    "aes_rd": 3,
}
ARCH_ORDER = {"MLP": 0, "CNN": 1, "Transformer": 2}


def kl_bernoulli(q: float, p: float) -> float:
    eps = 1e-15
    q = min(1.0 - eps, max(eps, float(q)))
    p = min(1.0 - eps, max(eps, float(p)))
    return q * math.log(q / p) + (1.0 - q) * math.log((1.0 - q) / (1.0 - p))


def kl_endpoint(successes: int, n: int, alpha: float, upper: bool) -> float:
    q = successes / n
    if upper:
        if successes >= n:
            return 1.0
        lo, hi = q, 1.0 - 1e-15
    else:
        if successes <= 0:
            return 0.0
        lo, hi = 1e-15, q
    target = math.log(1.0 / alpha) / n
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        too_far = kl_bernoulli(q, mid) > target
        if upper:
            if too_far:
                hi = mid
            else:
                lo = mid
        else:
            if too_far:
                lo = mid
            else:
                hi = mid
    return hi if upper else lo


def bi_from_p(p_success: float) -> float:
    return max(0.0, math.log2(1.0 + K * max(0.0, float(p_success) - CHANCE)))


def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def sort_loc_rows(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    out = df.copy()
    out["_dataset_order"] = out["dataset"].map(DATASET_ORDER).fillna(99)
    out["_arch_order"] = out["architecture"].map(ARCH_ORDER).fillna(99)
    out = out.sort_values(["_dataset_order", "dataset", "_arch_order", "architecture", "epsilon"])
    return out.drop(columns=["_dataset_order", "_arch_order"])


def compact_loc_table(loc_table: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "dataset",
        "architecture",
        "n_te",
        "epsilon",
        "S_loc",
        "p_hat_loc",
        "p_loc_plus",
        "BI_suite_plus",
        "BI_loc_plus",
        "nonvacuous",
    ]
    if loc_table.empty:
        return pd.DataFrame(columns=columns)
    return sort_loc_rows(loc_table)[columns].copy()


def compact_loc_curves(loc_curves: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "dataset",
        "architecture",
        "epsilon",
        "S_loc",
        "p_hat_loc",
        "p_loc_plus",
        "BI_loc_plus",
    ]
    if loc_curves.empty:
        return pd.DataFrame(columns=columns)
    return sort_loc_rows(loc_curves)[columns].copy()


def build_singleton(suite: pd.DataFrame, delta: float) -> pd.DataFrame:
    alpha = delta / max(1, len(suite))
    rows = []
    for row in suite.itertuples(index=False):
        n_te = int(row.n_te)
        successes = int(round(float(row.p_obs) * n_te))
        p_hat = successes / n_te
        p_minus = kl_endpoint(successes, n_te, alpha, upper=False)
        p_plus = kl_endpoint(successes, n_te, alpha, upper=True)
        rows.append(
            {
                "dataset": row.dataset,
                "attacker": row.best_model,
                "n_te": n_te,
                "successes": successes,
                "p_hat": p_hat,
                "p_minus": p_minus,
                "p_plus": p_plus,
                "BI_obs": bi_from_p(p_hat),
                "BI_minus": bi_from_p(p_minus),
                "BI_plus": bi_from_p(p_plus),
                "delta_effective": alpha,
                "source_scope_id": row.scope_id,
                "source_artifact": row.source_artifact,
                "note": "singleton endpoint for the current artifact-selected frozen attacker",
            }
        )
    return pd.DataFrame(rows)


def select_loc_rows(curves: pd.DataFrame, preferred_eps: Iterable[float]) -> pd.DataFrame:
    if curves.empty:
        return pd.DataFrame()
    rows = []
    for (dataset, arch, center), group in curves.groupby(["dataset", "architecture", "center_id"], sort=False):
        group = group.sort_values("epsilon")
        chosen = None
        for eps in preferred_eps:
            idx = (group["epsilon"].astype(float) - float(eps)).abs().idxmin()
            candidate = group.loc[idx]
            if math.isfinite(float(candidate["BI_loc_plus"])):
                chosen = candidate
                if float(candidate["epsilon"]) > 0:
                    break
        if chosen is None:
            chosen = group.iloc[0]
        rows.append(chosen.to_dict())
    return pd.DataFrame(rows)


def markdown_table(df: pd.DataFrame, columns: list[str], formats: dict[str, str] | None = None) -> list[str]:
    formats = formats or {}
    out = []
    out.append("| " + " | ".join(columns) + " |")
    out.append("| " + " | ".join(["---"] * len(columns)) + " |")
    for row in df[columns].itertuples(index=False):
        vals = []
        for col, val in zip(columns, row):
            if col in formats and pd.notna(val):
                vals.append(format(float(val), formats[col]))
            else:
                vals.append(str(val))
        out.append("| " + " | ".join(vals) + " |")
    return out


def write_summary(out_dir: Path, singleton: pd.DataFrame, suite: pd.DataFrame, loc_table: pd.DataFrame, loc_curves: pd.DataFrame) -> None:
    lines = [
        "# BI Scope Certificate Results",
        "",
        "Generated from the current frozen-suite artifacts plus exact score-local margin curves.",
        "",
        "## Execution Plan",
        "",
        "1. Preserve the existing eight-dataset `BI^suite` table as the main finite-suite evidence.",
        "2. Add an appendix-facing `BI^sing` table by applying a singleton KL-binomial endpoint to the frozen attacker recorded for each dataset.",
        "3. Add exact `BI^loc` curves using declared normalized scores and relaxed-margin counts `rho >= -2 epsilon`.",
        "4. Keep older product-norm / `BIarch` artifacts as diagnostics, not as architecture-family certificates.",
        "",
        "## BI^sing Table",
        "",
    ]
    lines.extend(
        markdown_table(
            singleton,
            ["dataset", "attacker", "n_te", "p_hat", "p_minus", "p_plus", "BI_obs", "BI_plus"],
            {"p_hat": ".6f", "p_minus": ".6f", "p_plus": ".6f", "BI_obs": ".3f", "BI_plus": ".3f"},
        )
    )
    lines.extend(["", "## BI^suite Table", ""])
    lines.extend(
        markdown_table(
            suite,
            ["dataset", "scope_id", "M", "n_te", "p_obs", "p_plus", "BI_obs", "BI_plus", "R_margin"],
            {"p_obs": ".6f", "p_plus": ".6f", "BI_obs": ".3f", "BI_plus": ".3f", "R_margin": ".3f"},
        )
    )
    lines.extend(["", "## BI^loc Table", ""])
    if loc_table.empty:
        lines.append("Pending: run `compute_bi_loc_margin_curves.py` and rerun this builder.")
    else:
        lines.extend(
            markdown_table(
                loc_table,
                [
                    "dataset",
                    "architecture",
                    "center_id",
                    "M",
                    "BI_suite_plus",
                    "epsilon",
                    "p_hat_loc",
                    "p_loc_plus",
                    "BI_loc_plus",
                    "nonvacuous",
                ],
                {
                    "BI_suite_plus": ".3f",
                    "epsilon": ".6g",
                    "p_hat_loc": ".6f",
                    "p_loc_plus": ".6f",
                    "BI_loc_plus": ".3f",
                },
            )
        )
        lines.extend(
            [
                "",
                "Full radius curves are in `bi_loc_radius_curves.csv`; the table above selects a small nonzero radius when available.",
                "Score normalization is row-specific and recorded in `bi_loc_radius_curves.csv`; Keras rows use probability-simplex scores (`A=1`) and the Transformer row uses calibrated raw logits.",
            ]
        )
    lines.extend(["", "## Outputs", ""])
    for name in [
        "bi_singleton_table.csv",
        "bi_suite_table.csv",
        "bi_loc_table.csv",
        "bi_loc_table_paper.csv",
        "bi_loc_radius_curves.csv",
        "bi_loc_radius_curves_paper.csv",
    ]:
        lines.append(f"- `{name}`")
    (out_dir / "BI_SCOPE_REORG_EXECUTION_PLAN.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite-csv", type=Path, default=Path("results/bi_suite_8datasets/bi_suite_dataset_summary.csv"))
    parser.add_argument("--loc-curves-csv", type=Path, default=Path("results/bi_scope_certificates/bi_loc_radius_curves.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/bi_scope_certificates"))
    parser.add_argument("--delta", type=float, default=1e-6)
    parser.add_argument("--preferred-eps", default="1e-4,3e-4,1e-3,0")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    suite = pd.read_csv(args.suite_csv)
    singleton = build_singleton(suite, args.delta)
    write_csv(singleton, args.output_dir / "bi_singleton_table.csv")
    write_csv(suite, args.output_dir / "bi_suite_table.csv")

    if args.loc_curves_csv.exists():
        loc_curves = pd.read_csv(args.loc_curves_csv)
    else:
        loc_curves = pd.DataFrame()
    preferred_eps = [float(x) for x in args.preferred_eps.split(",") if x.strip()]
    loc_table = sort_loc_rows(select_loc_rows(loc_curves, preferred_eps))
    write_csv(loc_table, args.output_dir / "bi_loc_table.csv")
    write_csv(compact_loc_table(loc_table), args.output_dir / "bi_loc_table_paper.csv")
    write_csv(compact_loc_curves(loc_curves), args.output_dir / "bi_loc_radius_curves_paper.csv")
    write_summary(args.output_dir, singleton, suite, loc_table, loc_curves)
    print(args.output_dir / "BI_SCOPE_REORG_EXECUTION_PLAN.md")


if __name__ == "__main__":
    main()
