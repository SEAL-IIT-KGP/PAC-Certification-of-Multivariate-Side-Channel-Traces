#!/usr/bin/env python3
"""Compose the four best raw-ASCAD byte attackers into exact four-byte key recovery.

Byte attackers are selected on their validation key rank and top-1 accuracy
only; the four-byte enumeration rank is then measured on the attack split over
independent campaigns of shuffled trace subsets (full-version table on the
raw-ASCAD four-byte composition).
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Sequence

import numpy as np

from multibyte_composition.train_raw_ascad_byte_attacker import AES_SBOX


KEYS = np.arange(256, dtype=np.uint8)


def parse_ints(text: str) -> list[int]:
    vals = [int(part.strip()) for part in text.replace(";", ",").split(",") if part.strip()]
    if not vals:
        raise argparse.ArgumentTypeError("expected at least one integer")
    return vals


def write_csv(path: Path, rows: Sequence[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"no rows for {path}")
    fieldnames: list[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fieldnames.append(key)
                seen.add(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def key_scores_for_campaign(probs: np.ndarray, plaintext_byte: np.ndarray, indices: np.ndarray) -> np.ndarray:
    labels = AES_SBOX[np.bitwise_xor(plaintext_byte[indices, None].astype(np.uint8), KEYS[None, :])].astype(np.int64)
    logp = np.log(probs[indices].astype(np.float64) + 1e-40)
    return np.take_along_axis(logp, labels, axis=1).sum(axis=0)


def rank0(scores: np.ndarray, true_key: int) -> int:
    return int(np.sum(scores > float(scores[int(true_key)])))


def full_rank_4byte(score_arrays: Sequence[np.ndarray], correct_keys: Sequence[int]) -> int:
    correct = sum(float(scores[int(key)]) for scores, key in zip(score_arrays, correct_keys))
    pair_a = (score_arrays[0][:, None] + score_arrays[1][None, :]).reshape(-1)
    pair_b = (score_arrays[2][:, None] + score_arrays[3][None, :]).reshape(-1)
    pair_b.sort()
    tol = 1e-6
    count = 0
    for val in pair_a:
        count += int(len(pair_b) - np.searchsorted(pair_b, correct + tol - val, side="right"))
    return count


def kl_bernoulli(p: float, q: float) -> float:
    eps = 1e-15
    p = min(max(float(p), eps), 1.0 - eps)
    q = min(max(float(q), eps), 1.0 - eps)
    return p * math.log(p / q) + (1.0 - p) * math.log((1.0 - p) / (1.0 - q))


def kl_upper(successes: int, n: int, delta: float) -> float:
    if successes >= n:
        return 1.0
    p_hat = successes / n
    target = math.log(1.0 / delta) / n
    lo, hi = p_hat, 1.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if kl_bernoulli(p_hat, mid) <= target:
            lo = mid
        else:
            hi = mid
    return lo


def select_bytes(input_dir: Path, candidates: Sequence[int], fixed: str) -> tuple[list[int], list[dict[str, object]]]:
    rows: list[dict[str, object]] = []
    for byte in candidates:
        path = input_dir / f"byte_{byte:02d}_model_summary.csv"
        if not path.exists():
            continue
        row = read_csv_rows(path)[0]
        out = {
            "target_byte": int(row["target_byte"]),
            "correct_key": int(row["correct_key"]),
            "val_top1": float(row["val_top1"]),
            "attack_top1": float(row["attack_top1"]),
            "val_rank_at_5000": int(row["val_rank_at_5000"]),
            "attack_rank_at_5000": int(row["attack_rank_at_5000"]),
            "features": int(row["features"]),
            "posterior_npz": row["posterior_npz"],
        }
        rows.append(out)
    if fixed.strip().lower() != "auto":
        selected = parse_ints(fixed)
    else:
        ordered = sorted(
            rows, key=lambda r: (int(r["val_rank_at_5000"]) != 0, int(r["val_rank_at_5000"]), -float(r["val_top1"]))
        )
        selected = [int(r["target_byte"]) for r in ordered[:4]]
    if len(selected) != 4:
        raise ValueError(f"expected exactly four selected bytes, got {selected}")
    return selected, rows


def main() -> None:
    repo = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input-dir", type=Path, default=repo / "results/multibyte_composition/byte_attackers")
    parser.add_argument("--candidate-bytes", type=parse_ints, default=parse_ints("0,1,2,3,4,5,8,15"))
    parser.add_argument("--target-bytes", default="auto")
    parser.add_argument(
        "--trace-counts", type=parse_ints, default=parse_ints("1,2,5,10,20,50,100,200,500,1000,2000,5000,10000")
    )
    parser.add_argument("--campaigns", type=int, default=250)
    parser.add_argument("--delta", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=20260520)
    args = parser.parse_args()

    args.input_dir.mkdir(parents=True, exist_ok=True)
    selected, inventory_rows = select_bytes(args.input_dir, args.candidate_bytes, args.target_bytes)
    byte_data = []
    for byte in selected:
        npz = np.load(args.input_dir / f"byte_{byte:02d}_posteriors.npz")
        byte_data.append(
            {
                "byte": byte,
                "probs": np.asarray(npz["probs"], dtype=np.float32),
                "plaintext": np.asarray(npz["plaintext"], dtype=np.uint8),
                "correct_key": int(np.asarray(npz["correct_key"]).reshape(-1)[0]),
            }
        )
    n_avail = min(len(d["probs"]) for d in byte_data)
    trace_counts = [int(c) for c in args.trace_counts if int(c) <= n_avail]
    max_count = max(trace_counts)
    rng = np.random.default_rng(int(args.seed))
    campaign_rows: list[dict[str, object]] = []
    per_byte_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    campaign_indices = [rng.choice(n_avail, max_count, replace=False) for _ in range(int(args.campaigns))]
    for trace_count in trace_counts:
        full_success = 0
        full_ranks = []
        byte_success = {d["byte"]: 0 for d in byte_data}
        byte_ranks = {d["byte"]: [] for d in byte_data}
        for campaign, base_idx in enumerate(campaign_indices):
            idx = base_idx[:trace_count]
            score_arrays = [key_scores_for_campaign(d["probs"], d["plaintext"][:, d["byte"]], idx) for d in byte_data]
            correct_keys = [d["correct_key"] for d in byte_data]
            ranks = [rank0(scores, key) for scores, key in zip(score_arrays, correct_keys)]
            tops = [int(np.argmax(scores)) for scores in score_arrays]
            for d, r, top, key in zip(byte_data, ranks, tops, correct_keys):
                byte_success[d["byte"]] += int(r == 0 and top == key)
                byte_ranks[d["byte"]].append(r)
            full_rank = full_rank_4byte(score_arrays, correct_keys)
            ok = int(full_rank == 0)
            full_success += ok
            full_ranks.append(full_rank)
            campaign_rows.append(
                {
                    "campaign": campaign,
                    "trace_count": trace_count,
                    "selected_bytes": ",".join(str(x) for x in selected),
                    "full_rank": full_rank,
                    "full_success": ok,
                    **{f"byte_{d['byte']:02d}_rank": r for d, r in zip(byte_data, ranks)},
                }
            )
        p_emp = []
        p_upper = []
        for d in byte_data:
            successes = byte_success[d["byte"]]
            p = successes / int(args.campaigns)
            up = kl_upper(successes, int(args.campaigns), float(args.delta))
            p_emp.append(p)
            p_upper.append(up)
            per_byte_rows.append(
                {
                    "target_byte": d["byte"],
                    "trace_count": trace_count,
                    "campaigns": int(args.campaigns),
                    "successes": successes,
                    "empirical_success": p,
                    "kl_binom_upper": up,
                    "rank0_mean": float(np.mean(byte_ranks[d["byte"]])),
                    "rank0_median": float(np.median(byte_ranks[d["byte"]])),
                }
            )
        ranks_arr = np.asarray(full_ranks, dtype=np.float64)
        summary_rows.append(
            {
                "trace_count": trace_count,
                "campaigns": int(args.campaigns),
                "selected_bytes": ",".join(str(x) for x in selected),
                "full_successes": full_success,
                "empirical_full_success": full_success / int(args.campaigns),
                "full_kl_binom_upper": kl_upper(full_success, int(args.campaigns), float(args.delta)),
                "dc_product_empirical_byte_success": float(np.prod(p_emp)),
                "dc_product_kl_upper_byte_success": float(np.prod(p_upper)),
                "mean_full_rank": float(np.mean(ranks_arr)),
                "median_full_rank": float(np.median(ranks_arr)),
                "p90_full_rank": float(np.quantile(ranks_arr, 0.9)),
                "min_full_rank": int(np.min(ranks_arr)),
                "max_full_rank": int(np.max(ranks_arr)),
            }
        )
    write_csv(args.input_dir / "posterior_artifact_inventory.csv", inventory_rows)
    write_csv(args.input_dir / "multibyte_campaign_rank_rows.csv", campaign_rows)
    write_csv(args.input_dir / "multibyte_per_byte_summary.csv", per_byte_rows)
    write_csv(args.input_dir / "multibyte_composition_summary.csv", summary_rows)
    (args.input_dir / "selected_bytes.txt").write_text(",".join(str(x) for x in selected) + "\n")
    readme = [
        "# Raw ASCAD Two-Share Multi-Byte Composition",
        "",
        f"Selected bytes: `{','.join(str(x) for x in selected)}`",
        "",
        "This artifact trains compact nonlinear byte attackers on raw ASCAD",
        "windows built from unmasked-S-box, masked-S-box, and mask POIs. The",
        "selected bytes are chosen using validation key rank/top-1 only; exact",
        "four-byte enumeration is then measured on the held-out attack split.",
        "",
        "| traces | exact full success | product byte success | median full rank |",
        "|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        readme.append(
            "| {trace_count} | {empirical_full_success:.6g} | {dc_product_empirical_byte_success:.6g} | "
            "{median_full_rank:.6g} |".format(**row)
        )
    (args.input_dir / "README.md").write_text("\n".join(readme) + "\n")
    print(f"Selected bytes: {selected}", flush=True)
    for row in summary_rows:
        print(row, flush=True)


if __name__ == "__main__":
    main()
