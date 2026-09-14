#!/usr/bin/env python3
"""Compute full M80 score-local BI_loc sweeps for frozen TCHES20 suites."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import tempfile
import warnings
from pathlib import Path
from typing import Any

import numpy as np

from bi_scope_certificates.compute_bi_loc_margin_curves import (
    adapt_input,
    bi_from_p,
    kl_upper,
    load_tches20_attack,
    margins_from_probs,
    normalized_probs,
)

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", tempfile.mkdtemp(prefix="full_suite_loc_mpl_"))
os.environ.setdefault("XDG_CACHE_HOME", tempfile.mkdtemp(prefix="full_suite_loc_font_"))


DATASET_PATTERNS = {
    "aes_rd": "*aes_rd*.hdf5",
    "ascad_desync_0": "*ascad_desync_0*.hdf5",
    "ascad_desync_50": "*ascad_desync_50*.hdf5",
    "ascad_desync_100": "*ascad_desync_100*.hdf5",
}

DEFAULT_MAX_ATTACK = {
    "aes_rd": 12500,
    "ascad_desync_0": 25000,
    "ascad_desync_50": 10000,
    "ascad_desync_100": 10000,
}


def read_suite_plus(path: Path, dataset: str) -> float:
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            if row["dataset"] == dataset:
                return float(row["BI_plus"])
    raise KeyError(f"dataset {dataset!r} not found in {path}")


def model_arch(name: str) -> str:
    if name.startswith("noConv1_"):
        return "MLP"
    if name.startswith("zaid_"):
        return "CNN"
    return "Other"


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"no rows to write for {path}")
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def summarize_dataset(
    *,
    dataset: str,
    margins_by_center: dict[str, np.ndarray],
    eps_grid: list[float],
    delta: float,
    suite_plus: float,
    output_dir: Path,
    score_normalization: str,
) -> tuple[Path, Path, Path]:
    m_count = len(margins_by_center)
    if m_count == 0:
        raise ValueError(f"no margins for {dataset}")
    n = len(next(iter(margins_by_center.values())))
    alpha = delta / max(1, m_count * len(eps_grid))
    scope = f"{'full' if m_count == 80 else 'partial'} TCHES20-M{m_count} score-local suite"

    per_center_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    tie_rows: list[dict[str, Any]] = []

    for center_id, margins in margins_by_center.items():
        correct_strict = int(np.sum(margins > 0.0))
        correct_tie_inclusive = int(np.sum(margins >= 0.0))
        tie_rows.append(
            {
                "dataset": dataset,
                "center_id": center_id,
                "architecture": model_arch(center_id),
                "n_te": n,
                "correct_strict_margin_gt_0": correct_strict,
                "correct_tie_inclusive_margin_ge_0": correct_tie_inclusive,
                "exact_tie_count": correct_tie_inclusive - correct_strict,
                "p_strict": correct_strict / n,
                "p_tie_inclusive": correct_tie_inclusive / n,
            }
        )
        for eps in eps_grid:
            successes = int(np.sum(margins >= -2.0 * eps))
            p_hat = successes / n
            p_plus = kl_upper(successes, n, alpha)
            per_center_rows.append(
                {
                    "dataset": dataset,
                    "scope": scope,
                    "center_id": center_id,
                    "architecture": model_arch(center_id),
                    "M": m_count,
                    "epsilon": eps,
                    "n_te": n,
                    "S_loc": successes,
                    "p_hat_loc": p_hat,
                    "p_loc_plus": p_plus,
                    "BI_obs_loc": bi_from_p(p_hat),
                    "BI_loc_plus": bi_from_p(p_plus),
                    "BI_suite_plus": suite_plus,
                    "score_normalization": score_normalization,
                    "A": 1.0,
                    "delta_effective": alpha,
                    "margin_min": float(np.min(margins)),
                    "margin_median": float(np.median(margins)),
                    "margin_q05": float(np.quantile(margins, 0.05)),
                    "margin_q95": float(np.quantile(margins, 0.95)),
                }
            )

    for eps in eps_grid:
        candidates = [
            (
                int(np.sum(margins >= -2.0 * eps)),
                center_id,
                margins,
            )
            for center_id, margins in margins_by_center.items()
        ]
        best_s, active_center, active_margins = max(candidates, key=lambda item: item[0])
        p_hat = best_s / n
        p_plus = kl_upper(best_s, n, alpha)
        summary_rows.append(
            {
                "dataset": dataset,
                "scope": scope,
                "M": m_count,
                "epsilon": eps,
                "n_te": n,
                "S_loc_max": best_s,
                "p_hat_loc_max": p_hat,
                "p_loc_plus": p_plus,
                "BI_obs_loc": bi_from_p(p_hat),
                "BI_loc_plus": bi_from_p(p_plus),
                "BI_suite_plus": suite_plus,
                "active_center": active_center,
                "active_architecture": model_arch(active_center),
                "score_normalization": score_normalization,
                "A": 1.0,
                "delta_effective": alpha,
                "active_margin_min": float(np.min(active_margins)),
                "active_margin_median": float(np.median(active_margins)),
                "active_margin_q05": float(np.quantile(active_margins, 0.05)),
                "active_margin_q95": float(np.quantile(active_margins, 0.95)),
                "nonvacuous": bool(p_plus < 1.0),
            }
        )

    prefix = output_dir / dataset
    per_center_path = prefix.with_name(f"{dataset}_full_suite_bi_loc_per_center.csv")
    summary_path = prefix.with_name(f"{dataset}_full_suite_bi_loc_summary.csv")
    tie_path = prefix.with_name(f"{dataset}_full_suite_tie_audit.csv")
    write_rows(per_center_path, per_center_rows)
    write_rows(summary_path, summary_rows)
    write_rows(tie_path, tie_rows)
    return per_center_path, summary_path, tie_path


def main() -> None:
    repo = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=sorted(DATASET_PATTERNS), required=True)
    parser.add_argument(
        "--tches-root",
        type=Path,
        default=repo / "datasets/raw/tches20/TCHES20V3_CNN_SCA",
    )
    parser.add_argument("--suite-csv", type=Path, default=repo / "results/bi_suite_8datasets/bi_suite_dataset_summary.csv")
    parser.add_argument("--output-dir", type=Path, default=repo / "results/bi_scope_certificates/full_suite_loc")
    parser.add_argument("--delta", type=float, default=1e-6)
    parser.add_argument("--eps-grid", default="0,1e-5,3e-5,1e-4,3e-4,1e-3,3e-3,1e-2,3e-2")
    parser.add_argument("--max-attack", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=512)
    args = parser.parse_args()

    import tensorflow as tf

    eps_grid = [float(x) for x in args.eps_grid.split(",") if x.strip()]
    max_attack = args.max_attack if args.max_attack > 0 else DEFAULT_MAX_ATTACK[args.dataset]
    suite_plus = read_suite_plus(args.suite_csv, args.dataset)
    model_dir = args.tches_root / "models" / "pretrained_models" / "models"
    model_paths = sorted(model_dir.glob(DATASET_PATTERNS[args.dataset]))
    if len(model_paths) != 80:
        raise RuntimeError(f"expected 80 models for {args.dataset}, found {len(model_paths)}")

    print(f"dataset={args.dataset} models={len(model_paths)} max_attack={max_attack}", flush=True)
    x_attack, labels = load_tches20_attack(args.tches_root, args.dataset, max_attack)
    margins_by_center: dict[str, np.ndarray] = {}
    errors: list[dict[str, str]] = []
    for idx, model_path in enumerate(model_paths, start=1):
        center_id = model_path.stem
        try:
            print(f"[{idx:02d}/{len(model_paths)}] loading {center_id}", flush=True)
            model = tf.keras.models.load_model(str(model_path), compile=False)
            x_model = adapt_input(model, x_attack)
            raw = model.predict(x_model, batch_size=args.batch_size, verbose=0)
            probs = normalized_probs(raw[0] if isinstance(raw, (list, tuple)) else raw)
            margins_by_center[center_id] = margins_from_probs(probs, labels)
            tf.keras.backend.clear_session()
        except Exception as exc:
            errors.append({"dataset": args.dataset, "center_id": center_id, "error": repr(exc)})
            print(f"ERROR {center_id}: {exc!r}", flush=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if errors:
        err_path = args.output_dir / f"{args.dataset}_full_suite_errors.csv"
        write_rows(err_path, errors)
        if not margins_by_center:
            raise SystemExit(1)
    if len(margins_by_center) < 80:
        warnings.warn(
            f"{args.dataset}: only {len(margins_by_center)} of 80 TCHES20 models were evaluated; "
            f"the certificate covers M={len(margins_by_center)} models, not the full TCHES20-M80 suite"
        )
    per_center_path, summary_path, tie_path = summarize_dataset(
        dataset=args.dataset,
        margins_by_center=margins_by_center,
        eps_grid=eps_grid,
        delta=args.delta,
        suite_plus=suite_plus,
        output_dir=args.output_dir,
        score_normalization="probability_simplex_A=1",
    )
    manifest = {
        "dataset": args.dataset,
        "M_requested": 80,
        "M_completed": len(margins_by_center),
        "n_te": int(len(labels)),
        "eps_grid": eps_grid,
        "delta": args.delta,
        "outputs": [str(per_center_path), str(summary_path), str(tie_path)],
        "errors": errors,
    }
    manifest_path = args.output_dir / f"{args.dataset}_full_suite_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(summary_path)


if __name__ == "__main__":
    main()
