#!/usr/bin/env python3
"""Metric-split CHES-CTF non-BI cells for Plot B extension.

Each array task computes one (dimension, seed, metric) cell and writes a
long-format checkpoint.  This is intentionally separate from
real_nonbi_dimcap_baselines.py because the high-dimensional estimators can fail
or time out independently, and the paper plot only needs metric-level means.
"""

from __future__ import annotations

import argparse
import fcntl
import math
import os
import socket
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

from dimension_stability.real_nonbi_dimcap_baselines import (
    load_dataset_window,
    metric_attempts,
    parse_csv_ints,
    parse_csv_strings,
    split_train_val,
)


METRIC_TO_COLUMN = {
    "ePI_or_PI": ("pi", "pi_success", "pi_error"),
    "eHI_or_HI": ("hi", "hi_success", "hi_error"),
    "MLP_PI": ("mlp_pi", "mlp_pi_success", "mlp_pi_error"),
    "GKOV_MI": ("mi", "mi_success", "mi_error"),
}

METRIC_TO_ARG = {
    "ePI_or_PI": "pi",
    "eHI_or_HI": "hi",
    "MLP_PI": "mlp_pi",
    "GKOV_MI": "mi",
}

CHECKPOINT_COLUMNS = [
    "dataset",
    "dimension",
    "seed",
    "metric",
    "status",
    "value",
    "success",
    "error",
    "n_total_used",
    "n_train",
    "n_val",
    "window_method",
    "pi_method",
    "source_artifact",
    "started_at",
    "finished_at",
    "elapsed_sec",
    "hostname",
    "job_id",
    "array_task_id",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def parse_args() -> argparse.Namespace:
    repo = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="results/ches_nonbi_highdim/manifest.csv")
    parser.add_argument("--output-dir", default="results/ches_nonbi_highdim")
    parser.add_argument("--dimensions", default="500,1000,3000,7000")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--metrics", default="ePI_or_PI,eHI_or_HI,MLP_PI,GKOV_MI")
    parser.add_argument("--write-manifest-only", action="store_true")
    parser.add_argument("--task-id", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--n-samples-total", type=int, default=30000)
    parser.add_argument("--n-val", type=int, default=6000)
    parser.add_argument("--n-classes", type=int, default=256)
    parser.add_argument("--max-dim-mi", type=int, default=7000)
    parser.add_argument("--pi-method", default="fast", choices=["original", "fast"])
    parser.add_argument("--pi-max-iter", type=int, default=250)
    parser.add_argument("--max-mi-samples", type=int, default=8000)
    parser.add_argument("--window-method", default="center", choices=["center", "start", "random"])
    parser.add_argument(
        "--tches20-dataset-dir",
        default=str(repo / "datasets/raw/tches20/TCHES20V3_CNN_SCA/datasets"),
    )
    parser.add_argument(
        "--tches20-src-dir",
        default=str(repo / "datasets/raw/tches20/TCHES20V3_CNN_SCA/src"),
    )
    parser.add_argument(
        "--ches-h5",
        default=str(repo / "datasets/raw/ches_ctf_2025/CHES_Challenge.h5"),
    )
    return parser.parse_args()


def write_manifest(args: argparse.Namespace) -> None:
    rows = []
    task_id = 0
    for dimension in parse_csv_ints(args.dimensions):
        for seed in parse_csv_ints(args.seeds):
            for metric in parse_csv_strings(args.metrics):
                if metric not in METRIC_TO_COLUMN:
                    raise ValueError(f"unknown metric: {metric}")
                rows.append(
                    {
                        "task_id": task_id,
                        "dataset": "ches_ctf_2025",
                        "dimension": dimension,
                        "seed": seed,
                        "metric": metric,
                        "status": "scheduled",
                        "note": "CHES high-dimensional non-BI metric-split Plot B pass",
                    }
                )
                task_id += 1
    manifest = Path(args.manifest)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(manifest, index=False)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    print(f"Wrote {len(rows)} tasks to {manifest}")


def read_task(manifest: Path, task_id: int) -> Dict[str, object]:
    df = pd.read_csv(manifest)
    match = df[df["task_id"].astype(int).eq(int(task_id))]
    if match.empty:
        raise IndexError(f"task_id {task_id} absent from {manifest}")
    return match.iloc[0].to_dict()


def normalize_checkpoint(path: Path) -> pd.DataFrame:
    if path.exists() and path.stat().st_size > 0:
        df = pd.read_csv(path)
    else:
        df = pd.DataFrame(columns=CHECKPOINT_COLUMNS)
    for col in CHECKPOINT_COLUMNS:
        if col not in df.columns:
            df[col] = ""
    return df[CHECKPOINT_COLUMNS]


def write_summary(df: pd.DataFrame, output_dir: Path) -> None:
    rows = []
    if df.empty:
        pd.DataFrame(columns=[
            "dataset",
            "dimension",
            "metric",
            "status",
            "success_fraction",
            "value_mean",
            "value_std",
            "success_count",
            "run_count",
            "source_artifact",
            "note",
        ]).to_csv(output_dir / "ches_nonbi_highdim_summary.csv", index=False)
        return
    for (dataset, dimension, metric), group in df.groupby(["dataset", "dimension", "metric"], dropna=False):
        values = pd.to_numeric(group["value"], errors="coerce")
        success = group["success"].fillna(False).astype(bool) & values.notna()
        rows.append(
            {
                "dataset": dataset,
                "dimension": int(dimension),
                "metric": metric,
                "status": "numeric" if bool(success.any()) else "failed",
                "success_fraction": float(success.mean()) if len(success) else 0.0,
                "value_mean": float(values[success].mean()) if bool(success.any()) else math.nan,
                "value_std": float(values[success].std(ddof=1)) if int(success.sum()) > 1 else math.nan,
                "success_count": int(success.sum()) if len(success) else 0,
                "run_count": int(len(group)),
                "source_artifact": str(output_dir / "ches_nonbi_highdim_checkpoint.csv"),
                "note": "CHES high-dimensional non-BI metric-split Plot B pass",
            }
        )
    pd.DataFrame(rows).sort_values(["dimension", "metric"]).to_csv(
        output_dir / "ches_nonbi_highdim_summary.csv", index=False
    )


def upsert_result(output_dir: Path, result: Dict[str, object]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = output_dir / "ches_nonbi_highdim_checkpoint.csv"
    lock_path = checkpoint.with_suffix(checkpoint.suffix + ".lock")
    with lock_path.open("w") as lock_f:
        fcntl.flock(lock_f, fcntl.LOCK_EX)
        df = normalize_checkpoint(checkpoint)
        mask = (
            df["dataset"].astype(str).eq(str(result["dataset"]))
            & pd.to_numeric(df["dimension"], errors="coerce").eq(int(result["dimension"]))
            & pd.to_numeric(df["seed"], errors="coerce").eq(int(result["seed"]))
            & df["metric"].astype(str).eq(str(result["metric"]))
        )
        df = df.loc[~mask].copy()
        df = pd.concat(
            [df, pd.DataFrame([{col: result.get(col, "") for col in CHECKPOINT_COLUMNS}])],
            ignore_index=True,
        )
        df = df.sort_values(["dataset", "dimension", "seed", "metric"]).reset_index(drop=True)
        tmp = checkpoint.with_suffix(checkpoint.suffix + ".tmp")
        df.to_csv(tmp, index=False, columns=CHECKPOINT_COLUMNS)
        os.replace(tmp, checkpoint)
        write_summary(df, output_dir)


def row_done(args: argparse.Namespace, dataset: str, dimension: int, seed: int, metric: str) -> bool:
    if args.force:
        return False
    checkpoint = Path(args.output_dir) / "ches_nonbi_highdim_checkpoint.csv"
    if not checkpoint.exists() or checkpoint.stat().st_size == 0:
        return False
    df = pd.read_csv(checkpoint)
    mask = (
        df["dataset"].astype(str).eq(dataset)
        & pd.to_numeric(df["dimension"], errors="coerce").eq(int(dimension))
        & pd.to_numeric(df["seed"], errors="coerce").eq(int(seed))
        & df["metric"].astype(str).eq(metric)
    )
    return bool(mask.any())


def run_task(args: argparse.Namespace) -> None:
    task_id = args.task_id
    if task_id is None:
        task_id = int(os.environ.get("ARRAY_TASK_ID", "0"))
    task = read_task(Path(args.manifest), task_id)
    dataset = str(task["dataset"])
    dimension = int(task["dimension"])
    seed = int(task["seed"])
    metric = str(task["metric"])
    if metric not in METRIC_TO_COLUMN:
        raise ValueError(f"unknown metric: {metric}")

    if row_done(args, dataset, dimension, seed, metric):
        print(f"[skip] existing row for dataset={dataset} d={dimension} seed={seed} metric={metric}")
        return

    started = utc_now()
    tic = time.monotonic()
    output_dir = Path(args.output_dir)
    result: Dict[str, object] = {
        "dataset": dataset,
        "dimension": dimension,
        "seed": seed,
        "metric": metric,
        "status": "failed",
        "value": math.nan,
        "success": False,
        "error": "",
        "n_total_used": math.nan,
        "n_train": math.nan,
        "n_val": math.nan,
        "window_method": args.window_method,
        "pi_method": args.pi_method,
        "source_artifact": "",
        "started_at": started,
        "hostname": socket.gethostname(),
        "job_id": os.environ.get("JOB_ID", ""),
        "array_task_id": os.environ.get("ARRAY_TASK_ID", str(task_id)),
    }

    try:
        print(f"[run] task={task_id} dataset={dataset} d={dimension} seed={seed} metric={metric}", flush=True)
        traces, labels, source = load_dataset_window(args, dataset, dimension)
        x_train, y_train, x_val, y_val = split_train_val(traces, labels, seed, args.n_val)
        args.metrics = METRIC_TO_ARG[metric]
        metric_result = metric_attempts(args, x_train, y_train, x_val, y_val, dimension)
        value_col, success_col, error_col = METRIC_TO_COLUMN[metric]
        success = bool(metric_result.get(success_col, False))
        value = metric_result.get(value_col, math.nan)
        result.update(
            {
                "status": "complete" if success else "failed",
                "value": value,
                "success": success,
                "error": metric_result.get(error_col, ""),
                "n_total_used": int(len(labels)),
                "n_train": int(len(y_train)),
                "n_val": int(len(y_val)),
                "source_artifact": source,
            }
        )
    except BaseException as exc:
        result.update({"status": "failed", "success": False, "error": repr(exc)})

    result["finished_at"] = utc_now()
    result["elapsed_sec"] = round(time.monotonic() - tic, 3)
    upsert_result(output_dir, result)
    print(
        f"[done] dataset={dataset} d={dimension} seed={seed} metric={metric} "
        f"success={result['success']} value={result['value']} elapsed={result['elapsed_sec']}s",
        flush=True,
    )


def main() -> None:
    args = parse_args()
    if args.write_manifest_only:
        write_manifest(args)
        return
    run_task(args)


if __name__ == "__main__":
    main()
