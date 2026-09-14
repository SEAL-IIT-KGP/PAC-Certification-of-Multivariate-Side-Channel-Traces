#!/usr/bin/env python3
"""Merge per-seed ASCAD d0 fixed-split MLP challenger outputs."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


# KL-binomial suite endpoint of ASCAD d0 on the PROFILING split (S=301 successes,
# n=50,000, M=80, delta=1e-6). It is not the paper's attack-split endpoint
# (n=10,000, p_hat=0.0063, p_plus=0.01247); pass --suite-p-plus to compare against another endpoint.
DEFAULT_SUITE_P_PLUS = 0.00840171215280455
# Same default as --output-dir of attacker_scope_diagnostics/evaluate_ascad_d0_challenger_fixed_split.py.
# That evaluator writes directly into --output-dir; this merge expects one evaluator run per seed
# in <base-dir>/seed_<k>/ (for example --seeds <k> --output-dir <base-dir>/seed_<k>).
DEFAULT_BASE = Path("results/ascad_fixed_split_challenger")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=DEFAULT_BASE,
        help="Directory containing seed_*/ascad_d0_challenger_fixed_split_rows.csv (one evaluator run per seed).",
    )
    parser.add_argument(
        "--suite-p-plus",
        type=float,
        default=DEFAULT_SUITE_P_PLUS,
        help="Suite endpoint to compare against (default: ASCAD d0 profiling-split endpoint, not the attack-split endpoint).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    base = args.base_dir
    suite_p_plus = float(args.suite_p_plus)
    row_files = sorted(base.glob("seed_*/ascad_d0_challenger_fixed_split_rows.csv"))
    if not row_files:
        raise FileNotFoundError(f"no seed row files under {base}")
    rows = pd.concat([pd.read_csv(path) for path in row_files], ignore_index=True)
    rows.to_csv(base / "ascad_d0_challenger_fixed_split_rows_all.csv", index=False)

    best = rows.sort_values("p_fixed", ascending=False).iloc[0]
    exceeds_suite_p_plus = bool(float(best["p_fixed"]) > suite_p_plus)
    if suite_p_plus == DEFAULT_SUITE_P_PLUS:
        endpoint_note = (
            "default ASCAD d0 profiling-split endpoint: S=301, n=50,000, M=80, delta=1e-6; "
            "not the attack-split endpoint used in the paper"
        )
    else:
        endpoint_note = "user-supplied via --suite-p-plus"
    summary = pd.DataFrame(
        [
            {
                "dataset": best["dataset"],
                "n_rows": int(len(rows)),
                "n_seeds": int(rows["seed"].nunique()),
                "n_fixed_eval": int(best["n_fixed_eval"]),
                "best_scope_level": int(best["scope_level"]),
                "best_scope_name": best["scope_name"],
                "best_seed": int(best["seed"]),
                "best_model": best["best_model_fixed"],
                "best_challenger_p_hat": float(best["p_fixed"]),
                "best_challenger_BI_obs": float(best["bi_fixed_obs"]),
                "old_holdout_reproduction_p_hat": float(best["p_old_holdout"]),
                "old_holdout_best_model": best["best_model_old_holdout"],
                "suite_p_plus": suite_p_plus,
                "exceeds_suite_p_plus": exceeds_suite_p_plus,
                "protocol": best["protocol"],
            }
        ]
    )
    summary.to_csv(base / "ascad_d0_challenger_fixed_split_summary_all.csv", index=False)

    lines = [
        "# ASCAD d0 MLP Challenger Fixed-Split Rerun",
        "",
        f"Rows evaluated: {len(rows)} (5 seeds x scopes 2/3/4; MLP family only)",
        f"Fixed evaluation population: {int(best['n_fixed_eval'])} profiling traces",
        "",
        "## Best Matched Challenger",
        "",
        f"- Scope: {best['scope_name']} (level {int(best['scope_level'])})",
        f"- Seed: {int(best['seed'])}",
        f"- Model: `{best['best_model_fixed']}`",
        f"- Matched fixed-profiling success (in-sample): `{float(best['p_fixed']):.6f}`",
        f"- Observed BI from point success: `{float(best['bi_fixed_obs']):.6f}` bits",
        f"- Old-holdout reproduction success for this fitted pool: `{float(best['p_old_holdout']):.6f}`",
        "",
        "## Outcome Against ASCAD d0 Suite Endpoint",
        "",
        f"- Suite endpoint ({endpoint_note}): `p_suite^+ = {suite_p_plus:.15f}`",
        f"- The matched MLP challenger {'exceeds' if exceeds_suite_p_plus else 'does not exceed'} this endpoint.",
        "- `p_fixed` from `evaluate_ascad_d0_challenger_fixed_split.py` is scored on the full profiling set, "
        "including the challengers' own training traces (in-sample), so this is not an out-of-sample comparison.",
        "",
        "## Protocol",
        "",
        str(best["protocol"]),
        "",
    ]
    (base / "ASCAD_D0_CHALLENGER_FIXED_SPLIT_SUMMARY.md").write_text("\n".join(lines))
    print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
