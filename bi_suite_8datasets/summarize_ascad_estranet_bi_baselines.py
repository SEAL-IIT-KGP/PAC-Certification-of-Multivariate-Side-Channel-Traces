#!/usr/bin/env python3
"""Summarize matched ASCAD EstraNet BI checkpoint-suite evaluations."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Iterable

import pandas as pd


N_CLASSES = 256


def kl_bernoulli(q: float, p: float) -> float:
    eps = 1e-15
    q = min(max(q, eps), 1.0 - eps)
    p = min(max(p, eps), 1.0 - eps)
    return q * math.log(q / p) + (1.0 - q) * math.log((1.0 - q) / (1.0 - p))


def endpoint(successes: int, n: int, c: float, *, upper: bool) -> float:
    q = successes / float(n)
    if upper:
        lo, hi = q, 1.0
        for _ in range(80):
            mid = 0.5 * (lo + hi)
            if kl_bernoulli(q, mid) <= c:
                lo = mid
            else:
                hi = mid
        return lo
    lo, hi = 0.0, q
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if kl_bernoulli(q, mid) <= c:
            hi = mid
        else:
            lo = mid
    return hi


def bi_bits(p: float) -> float:
    return max(0.0, math.log2(max(p, 0.0) * N_CLASSES))


def summarize_file(path: Path, delta: float) -> dict[str, object]:
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"{path} has no rows")
    if df["n_te"].nunique() != 1:
        raise ValueError(f"{path} has mixed n_te values")

    dataset = str(df["dataset"].iloc[0])
    scope_id = str(df["scope_id"].iloc[0])
    n_te = int(df["n_te"].iloc[0])
    m = int(len(df))
    c = math.log((2.0 * m) / delta) / float(n_te)

    per_model_counts = {
        str(row.model_id): int(row.successes)
        for row in df.itertuples(index=False)
    }
    best_model, best_successes = max(per_model_counts.items(), key=lambda kv: kv[1])
    p_obs = best_successes / float(n_te)
    lowers = [endpoint(s, n_te, c, upper=False) for s in per_model_counts.values()]
    uppers = [endpoint(s, n_te, c, upper=True) for s in per_model_counts.values()]
    p_minus = max(lowers)
    p_plus = max(uppers)
    slack = p_plus - p_obs
    denom = p_obs - (1.0 / N_CLASSES)
    r_margin = math.inf if denom <= 0.0 and slack > 0.0 else slack / max(denom, 1e-300)

    return {
        "dataset": dataset,
        "scope_id": scope_id,
        "M": m,
        "n_te": n_te,
        "p_obs": p_obs,
        "p_minus": p_minus,
        "p_plus": p_plus,
        "BI_minus": bi_bits(p_minus),
        "BI_obs": bi_bits(p_obs),
        "BI_plus": bi_bits(p_plus),
        "slack": slack,
        "R_margin": r_margin,
        "best_model": best_model,
        "source_artifact": str(path),
        "status": "complete",
        "per_model_success": json.dumps(
            {k: v / float(n_te) for k, v in per_model_counts.items()},
            sort_keys=True,
        ),
    }


def iter_success_files(input_dir: Path) -> Iterable[Path]:
    yield from sorted(input_dir.glob("ascad_desync_*_estranet_success.csv"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--output-csv", required=True, type=Path)
    parser.add_argument("--delta", default=1e-6, type=float)
    args = parser.parse_args()

    rows = [summarize_file(path, args.delta) for path in iter_success_files(args.input_dir)]
    if not rows:
        raise SystemExit(f"No ASCAD EstraNet success CSVs found in {args.input_dir}")

    order = {"ascad_desync_0": 0, "ascad_desync_50": 1, "ascad_desync_100": 2}
    rows.sort(key=lambda row: order.get(str(row["dataset"]), 99))

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output_csv, index=False)
    print(f"Wrote {args.output_csv}")
    print(pd.DataFrame(rows)[["dataset", "M", "n_te", "p_obs", "BI_obs", "BI_plus", "best_model"]].to_string(index=False))


if __name__ == "__main__":
    main()
