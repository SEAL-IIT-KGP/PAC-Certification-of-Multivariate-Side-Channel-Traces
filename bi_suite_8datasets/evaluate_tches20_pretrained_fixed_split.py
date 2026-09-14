#!/usr/bin/env python3
"""Evaluate TCHES20 pretrained suites on declared fixed data splits.

The reusable suite artifact evaluates random holdout subsets from the
profiling split and then averages seed-level certificates. This evaluator is a
margin-reduction audit: it evaluates each frozen pretrained model once on a
declared split (`profiling`, `attack`, or `both`) and writes raw success counts
plus a KL-binomial BI-suite summary.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import types
import warnings
from pathlib import Path
from typing import Iterable

import numpy as np


DATASET_KEYWORDS = {
    "ascad_desync_0": ["ascad_desync_0", "ascad"],
    "ascad_desync_50": ["ascad_desync_50", "desync50"],
    "ascad_desync_100": ["ascad_desync_100", "desync100"],
    "aes_hd": ["aes_hd"],
    "aes_rd": ["aes_rd"],
    "dpav4": ["dpav4", "dpa_v4"],
}

N_CLASSES = 256


def detect_dataset_tag(filename: str) -> str | None:
    name = filename.lower()
    for tag, keywords in sorted(
        DATASET_KEYWORDS.items(),
        key=lambda kv: -max(len(k) for k in kv[1]),
    ):
        if any(keyword in name for keyword in keywords):
            return tag
    return None


def discover_model_paths(models_dir: Path, dataset: str) -> list[Path]:
    paths = []
    for path in sorted(models_dir.rglob("*.hdf5")):
        if detect_dataset_tag(path.name) == dataset:
            paths.append(path)
    return paths


def load_data(dataset: str, dataset_dir: Path, tches20_src_dir: Path):
    try:
        import tqdm  # noqa: F401
    except ImportError:
        tqdm_stub = types.ModuleType("tqdm")

        def _tqdm(iterable=None, **_kwargs):
            return iterable if iterable is not None else []

        tqdm_stub.tqdm = _tqdm
        sys.modules["tqdm"] = tqdm_stub

    sys.path.insert(0, str(tches20_src_dir))
    from dataLoaders import load_aes_hd, load_aes_rd, load_ascad, load_dpav4  # type: ignore

    loader_map = {
        "ascad_desync_0": lambda: load_ascad(
            str(dataset_dir / "ASCAD_dataset" / "ASCAD.h5")
        ),
        "ascad_desync_50": lambda: load_ascad(
            str(dataset_dir / "ASCAD_dataset" / "ASCAD_desync50.h5")
        ),
        "ascad_desync_100": lambda: load_ascad(
            str(dataset_dir / "ASCAD_dataset" / "ASCAD_desync100.h5")
        ),
        "aes_hd": lambda: load_aes_hd(str(dataset_dir / "AES_HD_dataset") + os.sep),
        "aes_rd": lambda: load_aes_rd(str(dataset_dir / "AES_RD_dataset") + os.sep),
        "dpav4": lambda: load_dpav4(str(dataset_dir / "DPAv4_dataset") + os.sep),
    }
    if dataset not in loader_map:
        raise ValueError(f"unsupported dataset: {dataset}")

    x_profile, y_profile, x_attack, attack_targets, real_key = loader_map[dataset]()
    y_profile = np.asarray(y_profile).reshape(-1).astype(np.int64)
    y_attack = np.asarray(attack_targets[:, int(real_key)]).reshape(-1).astype(np.int64)
    return (
        np.asarray(x_profile),
        y_profile,
        np.asarray(x_attack),
        y_attack,
    )


def choose_split(split: str, data) -> tuple[np.ndarray, np.ndarray]:
    x_profile, y_profile, x_attack, y_attack = data
    if split == "profiling":
        return x_profile, y_profile
    if split == "attack":
        return x_attack, y_attack
    if split == "both":
        return np.concatenate([x_profile, x_attack], axis=0), np.concatenate(
            [y_profile, y_attack], axis=0
        )
    raise ValueError(f"unsupported split: {split}")


def model_input_shape(model) -> tuple[int, ...]:
    return tuple(d for d in model.input_shape[1:] if d is not None)


def prepare_input(model, x: np.ndarray) -> np.ndarray:
    expected = model_input_shape(model)
    out = x
    if len(expected) == 2 and expected[-1] == 1 and out.ndim == 2:
        target_len = expected[0]
        if out.shape[1] > target_len:
            out = out[:, :target_len]
        elif out.shape[1] < target_len:
            out = np.hstack([out, np.zeros((out.shape[0], target_len - out.shape[1]))])
        return out[..., np.newaxis]
    if len(expected) == 1 and out.ndim == 2:
        target_len = expected[0]
        if out.shape[1] > target_len:
            return out[:, :target_len]
        if out.shape[1] < target_len:
            return np.hstack([out, np.zeros((out.shape[0], target_len - out.shape[1]))])
    return out


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


def bi_bits(p: float) -> float:
    return max(0.0, math.log2(max(p, 0.0) * N_CLASSES))


def summarize_counts(rows: list[dict[str, object]], delta: float) -> dict[str, object]:
    n_te_values = {int(row["n_te"]) for row in rows}
    if len(n_te_values) != 1:
        raise ValueError(f"mixed n_te values: {n_te_values}")
    n_te = n_te_values.pop()
    m = len(rows)
    c = math.log((2.0 * m) / delta) / float(n_te)
    best = max(rows, key=lambda row: int(row["successes"]))
    p_obs = int(best["successes"]) / float(n_te)
    lowers = [kl_endpoint(int(row["successes"]), n_te, c, upper=False) for row in rows]
    uppers = [kl_endpoint(int(row["successes"]), n_te, c, upper=True) for row in rows]
    p_minus = max(lowers)
    p_plus = max(uppers)
    slack = p_plus - p_obs
    denom = p_obs - 1.0 / N_CLASSES
    r_margin = math.inf if denom <= 0.0 and slack > 0.0 else slack / max(denom, 1e-300)
    return {
        "dataset": best["dataset"],
        "split": best["split"],
        "scope_id": f"{best['dataset']}_tches20_m{m}_{best['split']}_fixed_split",
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
        "best_model": best["model_id"],
        "status": "complete",
    }


def evaluate(args: argparse.Namespace) -> tuple[list[dict[str, object]], dict[str, object]]:
    import tensorflow as tf

    model_paths = discover_model_paths(args.models_dir, args.dataset)
    if args.max_models:
        model_paths = model_paths[: args.max_models]
    if not model_paths:
        raise FileNotFoundError(f"no .hdf5 models found for {args.dataset} in {args.models_dir}")
    data = load_data(args.dataset, args.dataset_dir, args.tches20_src_dir)
    x_eval, y_eval = choose_split(args.split, data)
    if args.max_eval_traces is not None:
        x_eval = x_eval[: args.max_eval_traces]
        y_eval = y_eval[: args.max_eval_traces]
    print(f"dataset={args.dataset} split={args.split} x={x_eval.shape} models={len(model_paths)}")

    rows = []
    for index, path in enumerate(model_paths, start=1):
        tf.keras.backend.clear_session()
        print(f"[{index}/{len(model_paths)}] {path.name}", flush=True)
        try:
            model = tf.keras.models.load_model(str(path), compile=False)
            x_model = prepare_input(model, x_eval)
            preds = np.argmax(
                model.predict(x_model, batch_size=args.batch_size, verbose=0),
                axis=-1,
            ).astype(np.int64)
            successes = int(np.sum(preds == y_eval[: len(preds)]))
            status = "complete"
            error = ""
        except Exception as exc:  # pragma: no cover - runtime diagnostic path.
            warnings.warn(f"failed {path}: {exc}")
            successes = 0
            status = "failed"
            error = str(exc)
        rows.append(
            {
                "dataset": args.dataset,
                "split": args.split,
                "model_id": path.stem,
                "model_path": str(path),
                "n_te": int(len(y_eval)),
                "successes": successes,
                "success_rate": successes / float(len(y_eval)),
                "status": status,
                "error": error,
            }
        )
    complete_rows = [row for row in rows if row["status"] == "complete"]
    if not complete_rows:
        raise RuntimeError("all model evaluations failed")
    if len(complete_rows) < len(model_paths):
        warnings.warn(
            f"only {len(complete_rows)} of {len(model_paths)} declared models were evaluated; "
            f"the summary certificate covers M={len(complete_rows)} models"
        )
    summary = summarize_counts(complete_rows, args.delta)
    for row in rows:
        row["scope_id"] = summary["scope_id"]
    return rows, summary


def write_rows(path: Path, rows: Iterable[dict[str, object]]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=sorted(DATASET_KEYWORDS))
    parser.add_argument(
        "--split",
        required=True,
        choices=["profiling", "attack", "both"],
        help="Evaluation split. Use 'attack' for certificates; 'profiling' and 'both' include the "
        "models' own training traces and are in-sample diagnostics only.",
    )
    parser.add_argument("--models-dir", required=True, type=Path)
    parser.add_argument("--dataset-dir", required=True, type=Path)
    parser.add_argument("--tches20-src-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--delta", default=1e-6, type=float)
    parser.add_argument("--batch-size", default=1024, type=int)
    parser.add_argument("--max-models", default=0, type=int)
    parser.add_argument(
        "--max-eval-traces",
        default=None,
        type=int,
        help="Evaluate only the first N records of the selected split (default: all records). "
        "Use 12500 with --dataset aes_rd --split attack to match the paper's AES-RD attack set.",
    )
    args = parser.parse_args()
    if args.max_eval_traces is not None and args.max_eval_traces <= 0:
        parser.error("--max-eval-traces must be a positive integer")
    if args.split != "attack":
        warnings.warn(
            f"--split {args.split} evaluates the pretrained models on their own profiling (training) "
            "traces; the result is in-sample and is not a valid certificate. Use --split attack."
        )

    rows, summary = evaluate(args)
    stem = f"{args.dataset}_{args.split}"
    success_path = args.output_dir / f"{stem}_success.csv"
    summary["source_artifact"] = str(success_path)
    write_rows(success_path, rows)
    write_rows(args.output_dir / f"{stem}_summary.csv", [summary])
    print(summary)


if __name__ == "__main__":
    main()
