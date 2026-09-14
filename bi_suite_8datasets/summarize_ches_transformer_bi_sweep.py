#!/usr/bin/env python3
"""Summarize CHES-CTF Transformer window BI sweep success CSVs."""

from __future__ import annotations

import argparse
import csv
import math
import re
from collections import defaultdict
from pathlib import Path


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


def kl_lower_endpoint(successes: int, n: int, c: float) -> float:
    p_hat = successes / n
    if successes <= 0:
        return 0.0
    lo, hi = 0.0, p_hat
    for _ in range(80):
        mid = (lo + hi) / 2.0
        if bernoulli_kl(p_hat, mid) > c:
            lo = mid
        else:
            hi = mid
    return hi


def kl_upper_endpoint(successes: int, n: int, c: float) -> float:
    p_hat = successes / n
    if successes >= n:
        return 1.0
    lo, hi = p_hat, 1.0
    for _ in range(80):
        mid = (lo + hi) / 2.0
        if bernoulli_kl(p_hat, mid) > c:
            hi = mid
        else:
            lo = mid
    return lo


def bi_bits(p: float, n_classes: int) -> float:
    return max(0.0, math.log2(max(p, 0.0) * n_classes))


def dimension_from_scope(scope_id: str) -> int:
    match = re.search(r"_w(\d+)_checkpoint_suite$", scope_id)
    if not match:
        return -1
    return int(match.group(1))


def read_success_rows(input_dir: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in sorted(input_dir.glob("ches_ctf_2025_estranet_w*_success.csv")):
        with path.open(newline="") as f:
            for row in csv.DictReader(f):
                row["_source_csv"] = str(path)
                rows.append(row)
    return rows


def summarize(rows: list[dict[str, str]], delta: float, n_classes: int) -> list[dict[str, object]]:
    groups: dict[tuple[str, str, str, int], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        key = (
            row["dataset"],
            row["scope_id"],
            row.get("seed", "0"),
            int(float(row["n_te"])),
        )
        groups[key].append(row)

    out: list[dict[str, object]] = []
    for (dataset, scope_id, seed, n_te), group in sorted(
        groups.items(), key=lambda item: dimension_from_scope(item[0][1])
    ):
        counts = {
            row["model_id"]: int(float(row.get("successes") or float(row["success_rate"]) * n_te))
            for row in group
        }
        m = len(counts)
        c = math.log((2.0 * m) / delta) / float(n_te)
        best_model, best_successes = max(counts.items(), key=lambda kv: kv[1])
        lowers = [kl_lower_endpoint(s, n_te, c) for s in counts.values()]
        uppers = [kl_upper_endpoint(s, n_te, c) for s in counts.values()]
        p_obs = best_successes / n_te
        p_minus = max(lowers)
        p_plus = max(uppers)
        slack = p_plus - p_obs
        denom = p_obs - (1.0 / n_classes)
        r_margin = math.inf if denom <= 0.0 and slack > 0.0 else slack / max(denom, 1e-300)
        out.append(
            {
                "dataset": dataset,
                "dimension": dimension_from_scope(scope_id),
                "scope_id": scope_id,
                "seed": seed,
                "M": m,
                "n_te": n_te,
                "p_obs": p_obs,
                "p_minus": p_minus,
                "p_plus": p_plus,
                "BI_minus": bi_bits(p_minus, n_classes),
                "BI_obs": bi_bits(p_obs, n_classes),
                "BI_plus": bi_bits(p_plus, n_classes),
                "slack": slack,
                "R_margin": r_margin,
                "best_model": best_model,
                "source_csv": ";".join(sorted({row["_source_csv"] for row in group})),
            }
        )
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "results/ches_transformer_bi_dim_sweep/ches_transformer_bi_dim_sweep_summary.csv",
    )
    parser.add_argument("--delta", type=float, default=1e-6)
    parser.add_argument("--n-classes", type=int, default=256)
    args = parser.parse_args()

    rows = summarize(read_success_rows(args.input_dir), args.delta, args.n_classes)
    if not rows:
        raise SystemExit(f"No CHES Transformer BI success CSVs found in {args.input_dir}")

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {args.output_csv}")


if __name__ == "__main__":
    main()
