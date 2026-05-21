#!/usr/bin/env python3
"""Projected-dimension real-dataset finite-suite BI runner for Plot B.

Each array task evaluates one dataset/dimension/seed cell.  The lower
dimensional points are not obtained by feeding truncated traces to the
pretrained full-trace CNNs; instead they certify a fixed sklearn attack suite
trained on the projected trace representation.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import math
import os
import socket
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


LOADER_BACKED_DATASETS = [
    "ascad_desync_0",
    "ascad_desync_50",
    "ascad_desync_100",
    "aes_hd",
    "aes_rd",
    "dpav4",
]

CHECKPOINT_COLUMNS = [
    "dataset",
    "dimension",
    "seed",
    "status",
    "n_total_used",
    "n_train",
    "n_holdout",
    "window_method",
    "M",
    "p_obs",
    "p_minus",
    "p_plus",
    "BI_minus",
    "BI_obs",
    "BI_plus",
    "slack",
    "R_margin",
    "best_model",
    "per_model_success",
    "source_artifact",
    "started_at",
    "finished_at",
    "elapsed_sec",
    "hostname",
    "job_id",
    "array_task_id",
    "error",
]


def parse_csv_ints(value: str) -> List[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_csv_strings(value: str) -> List[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    repo = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="results/plot_b_projected_bi/manifest.csv")
    parser.add_argument("--output-dir", default="results/plot_b_projected_bi")
    parser.add_argument("--datasets", default="ascad_desync_0")
    parser.add_argument("--dimensions", default="10,50,100")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--write-manifest-only", action="store_true")
    parser.add_argument("--task-id", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--n-samples-total", type=int, default=50000)
    parser.add_argument("--holdout-size", type=int, default=25000)
    parser.add_argument("--n-classes", type=int, default=256)
    parser.add_argument("--delta", type=float, default=1e-6)
    parser.add_argument("--bi-max-iter", type=int, default=250)
    parser.add_argument("--window-method", default="center", choices=["center", "start", "random"])
    parser.add_argument(
        "--tches20-dataset-dir",
        default=str(repo / "datasets/raw/tches20/TCHES20V3_CNN_SCA/datasets"),
    )
    parser.add_argument(
        "--tches20-src-dir",
        default=str(repo / "datasets/raw/tches20/TCHES20V3_CNN_SCA/src"),
    )
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def bi_bits(p: float, n_classes: int) -> float:
    if not np.isfinite(p):
        return math.nan
    return max(0.0, math.log2(max(0.0, float(p)) * n_classes))


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


def suite_bracket(per_model_success: Dict[str, float], n: int, delta: float, n_classes: int) -> Dict[str, float]:
    if not per_model_success:
        raise ValueError("empty attack suite")
    m = len(per_model_success)
    counts = {
        name: int(round(max(0.0, min(1.0, float(rate))) * n))
        for name, rate in per_model_success.items()
    }
    c = math.log((2.0 * m) / delta) / float(n)
    best_model, best_successes = max(counts.items(), key=lambda item: item[1])
    p_obs = best_successes / n
    p_minus = max(kl_lower_endpoint(s, n, c) for s in counts.values())
    p_plus = max(kl_upper_endpoint(s, n, c) for s in counts.values())
    slack = p_plus - p_obs
    denom = p_obs - (1.0 / n_classes)
    r_margin = math.inf if denom <= 0.0 and slack > 0.0 else slack / max(denom, 1e-300)
    return {
        "M": float(m),
        "p_obs": p_obs,
        "p_minus": p_minus,
        "p_plus": p_plus,
        "BI_minus": bi_bits(p_minus, n_classes),
        "BI_obs": bi_bits(p_obs, n_classes),
        "BI_plus": bi_bits(p_plus, n_classes),
        "slack": slack,
        "R_margin": r_margin,
        "best_model": best_model,
    }


def write_manifest(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = Path(args.manifest)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    task_id = 0
    for dataset in parse_csv_strings(args.datasets):
        if dataset not in LOADER_BACKED_DATASETS:
            raise ValueError(f"unsupported loader-backed dataset: {dataset}")
        for dimension in parse_csv_ints(args.dimensions):
            for seed in parse_csv_ints(args.seeds):
                rows.append(
                    {
                        "task_id": task_id,
                        "dataset": dataset,
                        "dimension": dimension,
                        "seed": seed,
                        "status": "scheduled",
                        "note": "projected-window finite-suite BI for Plot B",
                    }
                )
                task_id += 1
    pd.DataFrame(rows).to_csv(manifest_path, index=False)
    print(f"Wrote manifest with {len(rows)} cells: {manifest_path}")


def read_task(manifest: Path, task_id: int) -> Dict[str, object]:
    df = pd.read_csv(manifest)
    match = df[df["task_id"].astype(int) == int(task_id)]
    if match.empty:
        raise IndexError(f"task_id {task_id} not present in {manifest}")
    return match.iloc[0].to_dict()


def load_dataset_window(args: argparse.Namespace, dataset: str, dimension: int) -> Tuple[np.ndarray, np.ndarray, str]:
    from dimension_stability.multivariate_stability_driver import create_windowed_representation, try_load_tches20_data

    traces, labels = try_load_tches20_data(
        dataset,
        args.tches20_dataset_dir,
        args.tches20_src_dir,
    )
    n = min(int(args.n_samples_total), int(len(labels)))
    traces = np.asarray(traces[:n])
    labels = np.asarray(labels[:n], dtype=np.int64).reshape(-1)
    if dimension and int(dimension) != int(traces.shape[1]):
        traces = create_windowed_representation(traces, int(dimension), method=args.window_method)
    return np.asarray(traces, dtype=np.float32), labels, f"{args.tches20_dataset_dir}:{dataset}"


def split_train_holdout(
    traces: np.ndarray,
    labels: np.ndarray,
    seed: int,
    holdout_size: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n_total = int(len(labels))
    n_holdout = min(int(holdout_size), n_total // 2)
    n_train = n_total - n_holdout
    np.random.seed(int(seed))
    indices = np.random.permutation(n_total)
    train_idx = indices[:n_train]
    holdout_idx = indices[n_train : n_train + n_holdout]
    return traces[train_idx], labels[train_idx], traces[holdout_idx], labels[holdout_idx]


def train_projected_suite(
    X_train: np.ndarray,
    y_train: np.ndarray,
    seed: int,
    max_iter: int,
) -> List[Tuple[str, object]]:
    models: List[Tuple[str, object]] = []

    candidates = [
        (
            "logistic_fast",
            make_pipeline(
                StandardScaler(),
                LogisticRegression(
                    solver="saga",
                    max_iter=max_iter,
                    n_jobs=-1,
                    multi_class="multinomial",
                    random_state=int(seed),
                ),
            ),
        ),
        (
            "lda_shrinkage",
            make_pipeline(
                StandardScaler(),
                LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"),
            ),
        ),
        (
            "mlp_fast",
            make_pipeline(
                StandardScaler(),
                MLPClassifier(
                    hidden_layer_sizes=(32,),
                    max_iter=min(60, max_iter),
                    early_stopping=True,
                    validation_fraction=0.1,
                    n_iter_no_change=3,
                    batch_size=512,
                    random_state=int(seed),
                ),
            ),
        ),
    ]

    for name, model in candidates:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(X_train, y_train)
            models.append((name, model))
        except Exception as exc:
            warnings.warn(f"{name} failed: {exc}")

    if not models:
        raise RuntimeError("all projected BI attack models failed")
    return models


def evaluate_suite(models: Iterable[Tuple[str, object]], X_holdout: np.ndarray, y_holdout: np.ndarray) -> Dict[str, float]:
    per_model: Dict[str, float] = {}
    y_true = np.asarray(y_holdout).reshape(-1)
    for name, model in models:
        try:
            y_pred = np.asarray(model.predict(X_holdout)).reshape(-1)
            per_model[name] = float(np.mean(y_pred == y_true[: len(y_pred)]))
        except Exception as exc:
            warnings.warn(f"evaluation failed for {name}: {exc}")
            per_model[name] = math.nan
    return {name: rate for name, rate in per_model.items() if np.isfinite(rate)}


def normalize_checkpoint(path: Path) -> pd.DataFrame:
    if path.exists() and path.stat().st_size > 0:
        df = pd.read_csv(path)
    else:
        df = pd.DataFrame(columns=CHECKPOINT_COLUMNS)
    for col in CHECKPOINT_COLUMNS:
        if col not in df.columns:
            df[col] = "" if col in {"status", "error", "source_artifact", "per_model_success"} else np.nan
    return df[CHECKPOINT_COLUMNS]


def write_summary(df: pd.DataFrame, output_dir: Path) -> None:
    rows = []
    numeric_cols = ["p_obs", "p_minus", "p_plus", "BI_minus", "BI_obs", "BI_plus", "slack", "R_margin", "M"]
    for (dataset, dimension), group in df.groupby(["dataset", "dimension"], dropna=False):
        ok = group["status"].astype(str).eq("complete")
        row = {
            "dataset": dataset,
            "dimension": int(dimension),
            "status": "complete" if bool(ok.any()) else "failed",
            "success_count": int(ok.sum()),
            "run_count": int(len(group)),
            "source_artifact": str(output_dir / "real_projected_bi_dimcap_checkpoint.csv"),
            "note": "projected-window finite-suite BI; KL-binomial endpoints",
        }
        for col in numeric_cols:
            vals = pd.to_numeric(group.loc[ok, col], errors="coerce").dropna()
            row[f"{col}_mean"] = float(vals.mean()) if len(vals) else math.nan
            row[f"{col}_std"] = float(vals.std(ddof=1)) if len(vals) > 1 else math.nan
        rows.append(row)
    pd.DataFrame(rows).to_csv(output_dir / "real_projected_bi_dimcap_summary.csv", index=False)


def upsert_result(output_dir: Path, result: Dict[str, object]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = output_dir / "real_projected_bi_dimcap_checkpoint.csv"
    lock_path = checkpoint.with_suffix(checkpoint.suffix + ".lock")
    with open(lock_path, "w") as lock_f:
        fcntl.flock(lock_f, fcntl.LOCK_EX)
        df = normalize_checkpoint(checkpoint)
        mask = (
            (df["dataset"].astype(str) == str(result["dataset"]))
            & (pd.to_numeric(df["dimension"], errors="coerce") == int(result["dimension"]))
            & (pd.to_numeric(df["seed"], errors="coerce") == int(result["seed"]))
        )
        df = df.loc[~mask].copy()
        row = {col: result.get(col, "") for col in CHECKPOINT_COLUMNS}
        df = pd.concat([df, pd.DataFrame([row], columns=CHECKPOINT_COLUMNS)], ignore_index=True)
        df = df.sort_values(["dataset", "dimension", "seed"]).reset_index(drop=True)
        tmp = checkpoint.with_suffix(checkpoint.suffix + ".tmp")
        df.to_csv(tmp, index=False, columns=CHECKPOINT_COLUMNS)
        os.replace(tmp, checkpoint)
        write_summary(df, output_dir)


def row_complete(args: argparse.Namespace, dataset: str, dimension: int, seed: int) -> bool:
    if args.force:
        return False
    checkpoint = Path(args.output_dir) / "real_projected_bi_dimcap_checkpoint.csv"
    if not checkpoint.exists() or checkpoint.stat().st_size == 0:
        return False
    df = pd.read_csv(checkpoint)
    mask = (
        (df["dataset"].astype(str) == dataset)
        & (pd.to_numeric(df["dimension"], errors="coerce") == int(dimension))
        & (pd.to_numeric(df["seed"], errors="coerce") == int(seed))
    )
    return bool(mask.any() and str(df.loc[mask].iloc[-1].get("status", "")) == "complete")


def run_task(args: argparse.Namespace) -> None:
    task_id = args.task_id
    if task_id is None:
        task_id = int(os.environ.get("ARRAY_TASK_ID", "0"))
    task = read_task(Path(args.manifest), task_id)
    dataset = str(task["dataset"])
    dimension = int(task["dimension"])
    seed = int(task["seed"])

    if row_complete(args, dataset, dimension, seed):
        print(f"[skip] existing complete row for dataset={dataset} d={dimension} seed={seed}")
        return

    started = utc_now()
    tic = time.monotonic()
    output_dir = Path(args.output_dir)
    base = {
        "dataset": dataset,
        "dimension": dimension,
        "seed": seed,
        "status": "failed",
        "window_method": args.window_method,
        "started_at": started,
        "hostname": socket.gethostname(),
        "job_id": os.environ.get("JOB_ID", ""),
        "array_task_id": os.environ.get("ARRAY_TASK_ID", str(task_id)),
    }
    print(f"[run] task={task_id} dataset={dataset} d={dimension} seed={seed}")

    try:
        traces, labels, source = load_dataset_window(args, dataset, dimension)
        X_train, y_train, X_holdout, y_holdout = split_train_holdout(
            traces, labels, seed, args.holdout_size
        )
        models = train_projected_suite(X_train, y_train, seed, args.bi_max_iter)
        per_model = evaluate_suite(models, X_holdout, y_holdout)
        bracket = suite_bracket(per_model, len(y_holdout), args.delta, args.n_classes)
        result = dict(base)
        result.update(bracket)
        result.update(
            {
                "status": "complete",
                "n_total_used": int(len(labels)),
                "n_train": int(len(y_train)),
                "n_holdout": int(len(y_holdout)),
                "per_model_success": json.dumps(per_model, sort_keys=True),
                "source_artifact": source,
                "error": "",
            }
        )
    except BaseException as exc:
        result = dict(base)
        result.update({"error": repr(exc)})

    result["finished_at"] = utc_now()
    result["elapsed_sec"] = round(time.monotonic() - tic, 3)
    upsert_result(output_dir, result)
    print(
        f"[done] dataset={dataset} d={dimension} seed={seed} "
        f"status={result.get('status')} BI_plus={result.get('BI_plus')} "
        f"elapsed={result['elapsed_sec']}s"
    )


def main() -> None:
    args = parse_args()
    if args.write_manifest_only:
        write_manifest(args)
        return
    run_task(args)


if __name__ == "__main__":
    main()
