#!/usr/bin/env python3
"""Run a fresh fair non-BI dimension-capped baseline pass on real datasets.

Each run computes one dataset/dimension/seed cell and upserts it into a
fresh local checkpoint.
"""

from __future__ import annotations

import argparse
import fcntl
import math
import os
import socket
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


LOADER_BACKED_DATASETS = [
    "ascad_desync_0",
    "ascad_desync_50",
    "ascad_desync_100",
    "aes_hd",
    "aes_rd",
    "dpav4",
    "ches_ctf_2025",
]

UNSUPPORTED_DATASETS = {
    "ascad_random_key": (
        "no_loader_backed_random-key H5 was found locally/remotely; the visible ASCAD.h5 "
        "asset has fixed key metadata"
    )
}

CHECKPOINT_COLUMNS = [
    "dataset",
    "dimension",
    "seed",
    "status",
    "n_total_used",
    "n_train",
    "n_val",
    "window_method",
    "pi_method",
    "source_artifact",
    "pi",
    "ti",
    "pi_success",
    "pi_error",
    "hi",
    "hi_success",
    "hi_error",
    "mlp_pi",
    "mlp_ti",
    "mlp_pi_success",
    "mlp_pi_error",
    "mi",
    "mi_success",
    "mi_error",
    "started_at",
    "finished_at",
    "elapsed_sec",
    "hostname",
    "job_id",
    "array_task_id",
]


def parse_csv_ints(value: str) -> List[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_csv_strings(value: str) -> List[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_metrics(value: str) -> set[str]:
    aliases = {
        "all": "all",
        "pi": "pi",
        "epi": "pi",
        "epi_or_pi": "pi",
        "hi": "hi",
        "ehi": "hi",
        "ehi_or_hi": "hi",
        "mlp": "mlp_pi",
        "mlp_pi": "mlp_pi",
        "mi": "mi",
        "gkov": "mi",
        "gkov_mi": "mi",
    }
    selected = set()
    for item in parse_csv_strings(value.lower()):
        metric = aliases.get(item)
        if metric is None:
            raise ValueError(f"unknown metric in --metrics: {item}")
        selected.add(metric)
    if not selected or "all" in selected:
        return {"pi", "hi", "mlp_pi", "mi"}
    return selected


def parse_args() -> argparse.Namespace:
    repo = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="results/real_nonbi_dimcap/manifest.csv")
    parser.add_argument("--output-dir", default="results/real_nonbi_dimcap")
    parser.add_argument("--datasets", default=",".join(LOADER_BACKED_DATASETS))
    parser.add_argument("--dimensions", default="10,50,100")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--write-manifest-only", action="store_true")
    parser.add_argument("--task-id", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--n-samples-total", type=int, default=30000)
    parser.add_argument("--n-val", type=int, default=6000)
    parser.add_argument("--n-classes", type=int, default=256)
    parser.add_argument("--max-dim-mi", type=int, default=100)
    parser.add_argument("--pi-method", default="original", choices=["original", "fast"])
    parser.add_argument("--pi-max-iter", type=int, default=250)
    parser.add_argument(
        "--max-mi-samples",
        type=int,
        default=8000,
        help="Cap training samples used by the MI estimator to avoid O(n^2) memory blowups. "
        "Set to 0 to disable capping.",
    )
    parser.add_argument("--metrics", default="all", help="Comma-separated subset: pi,hi,mlp_pi,mi or all")
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


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def safe_float(value) -> float:
    try:
        return float(value)
    except Exception:
        return math.nan


def write_manifest(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = Path(args.manifest)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    task_id = 0
    for dataset in parse_csv_strings(args.datasets):
        if dataset not in LOADER_BACKED_DATASETS:
            continue
        loader = "ches_h5" if dataset == "ches_ctf_2025" else "tches20_loader"
        for dimension in parse_csv_ints(args.dimensions):
            for seed in parse_csv_ints(args.seeds):
                rows.append(
                    {
                        "task_id": task_id,
                        "dataset": dataset,
                        "dimension": dimension,
                        "seed": seed,
                        "loader": loader,
                        "status": "scheduled",
                        "note": "fresh fair dimension-capped non-BI baseline pass",
                    }
                )
                task_id += 1
    pd.DataFrame(rows).to_csv(manifest_path, index=False)

    inventory_rows = [
        {
            "dataset": dataset,
            "status": "scheduled" if dataset in LOADER_BACKED_DATASETS else "unsupported",
            "dimensions": args.dimensions if dataset in LOADER_BACKED_DATASETS else "",
            "seeds": args.seeds if dataset in LOADER_BACKED_DATASETS else "",
            "source_artifact": (
                args.ches_h5 if dataset == "ches_ctf_2025" else args.tches20_dataset_dir
            )
            if dataset in LOADER_BACKED_DATASETS
            else "",
            "note": "loader-backed real-dataset pass"
            if dataset in LOADER_BACKED_DATASETS
            else UNSUPPORTED_DATASETS.get(dataset, "unsupported"),
        }
        for dataset in LOADER_BACKED_DATASETS + list(UNSUPPORTED_DATASETS)
    ]
    pd.DataFrame(inventory_rows).to_csv(output_dir / "real_nonbi_dimcap_inventory.csv", index=False)
    print(f"Wrote manifest with {len(rows)} scheduled cells: {manifest_path}")


def read_task(manifest: Path, task_id: int) -> Dict[str, object]:
    df = pd.read_csv(manifest)
    if "task_id" in df.columns:
        match = df[df["task_id"].astype(int) == int(task_id)]
        if match.empty:
            raise IndexError(f"task_id {task_id} not present in {manifest}")
        return match.iloc[0].to_dict()
    if task_id < 0 or task_id >= len(df):
        raise IndexError(f"task_id {task_id} outside manifest with {len(df)} rows")
    return df.iloc[task_id].to_dict()


def load_ches_window(
    h5_path: Path,
    dimension: int,
    n_samples_total: int,
    window_method: str,
) -> Tuple[np.ndarray, np.ndarray, str]:
    import h5py

    with h5py.File(h5_path, "r") as handle:
        traces_ds = handle["Profiling_traces/traces"]
        metadata_ds = handle["Profiling_traces/metadata"]
        n = min(int(n_samples_total), int(traces_ds.shape[0]))
        n_features = int(traces_ds.shape[1])
        if dimension >= n_features:
            start = 0
            stop = n_features
        elif window_method == "start":
            start = 0
            stop = dimension
        elif window_method == "random":
            rng = np.random.default_rng(0)
            start = int(rng.integers(0, n_features - dimension + 1))
            stop = start + dimension
        else:
            start = (n_features - dimension) // 2
            stop = start + dimension
        traces = np.asarray(traces_ds[:n, start:stop], dtype=np.float32)
        labels = np.asarray(metadata_ds[:n]["labels"], dtype=np.int64).reshape(-1)
    return traces, labels, f"{h5_path}:Profiling_traces/traces[:,{start}:{stop}]"


def load_dataset_window(args: argparse.Namespace, dataset: str, dimension: int) -> Tuple[np.ndarray, np.ndarray, str]:
    if dataset == "ches_ctf_2025":
        return load_ches_window(Path(args.ches_h5), dimension, args.n_samples_total, args.window_method)

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


def split_train_val(
    traces: np.ndarray,
    labels: np.ndarray,
    seed: int,
    n_val_target: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n_total = int(len(labels))
    n_val = min(int(n_val_target), max(1, n_total // 5))
    if n_total - n_val < 256:
        raise ValueError(f"not enough samples after validation split: n_total={n_total}, n_val={n_val}")
    rng = np.random.default_rng(int(seed))
    indices = rng.permutation(n_total)
    val_idx = indices[-n_val:]
    train_idx = indices[:-n_val]
    return traces[train_idx], labels[train_idx], traces[val_idx], labels[val_idx]


def compute_pi_logistic_fast(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    n_classes: int,
    max_iter: int,
) -> Tuple[float, float]:
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_train)
    X_vl = scaler.transform(X_val)
    clf = LogisticRegression(
        solver="saga",
        max_iter=max_iter,
        n_jobs=-1,
        multi_class="multinomial",
        random_state=0,
    )
    clf.fit(X_tr, y_train)

    log_proba_val = clf.predict_log_proba(X_vl)
    log_proba_train = clf.predict_log_proba(X_tr)
    class_to_col = {int(cls): idx for idx, cls in enumerate(clf.classes_)}
    score_val = 0.0
    score_train = 0.0
    for cls in range(n_classes):
        col = class_to_col.get(cls)
        if col is None:
            continue
        idx_val = np.where(y_val == cls)[0]
        idx_train = np.where(y_train == cls)[0]
        if len(idx_val) > 0:
            score_val += np.mean(log_proba_val[idx_val, col])
        if len(idx_train) > 0:
            score_train += np.mean(log_proba_train[idx_train, col])
    log2_n = np.log2(n_classes)
    pi = log2_n + (score_val / n_classes) * np.log2(np.e)
    ti = log2_n + (score_train / n_classes) * np.log2(np.e)
    return pi, ti


def metric_attempts(
    args: argparse.Namespace,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    dimension: int,
) -> Dict[str, object]:
    from core.baseline_metrics import compute_hi, compute_mi, compute_pi
    from dimension_stability.multivariate_stability_driver import compute_mlp_pi_ti

    result: Dict[str, object] = {}
    selected = parse_metrics(args.metrics)

    if "pi" in selected:
        try:
            if args.pi_method == "fast":
                pi, ti = compute_pi_logistic_fast(
                    X_train,
                    y_train,
                    X_val,
                    y_val,
                    n_classes=args.n_classes,
                    max_iter=args.pi_max_iter,
                )
            else:
                pi, ti = compute_pi(
                    X_train,
                    y_train,
                    X_val,
                    y_val,
                    method="logistic",
                    n_classes=args.n_classes,
                )
            result.update({"pi": safe_float(pi), "ti": safe_float(ti), "pi_success": True, "pi_error": ""})
        except SystemExit as exc:
            code = getattr(exc, "code", 1)
            result.update({"pi": math.nan, "ti": math.nan, "pi_success": False, "pi_error": f"SystemExit({code})"})
        except Exception as exc:
            result.update({"pi": math.nan, "ti": math.nan, "pi_success": False, "pi_error": repr(exc)})
    else:
        result.update({"pi": math.nan, "ti": math.nan, "pi_success": False, "pi_error": "skipped by --metrics"})

    if "hi" in selected:
        try:
            hi = compute_hi(X_train, y_train, X_val, y_val, n_classes=args.n_classes)
            result.update({"hi": safe_float(hi), "hi_success": True, "hi_error": ""})
        except SystemExit as exc:
            code = getattr(exc, "code", 1)
            result.update({"hi": math.nan, "hi_success": False, "hi_error": f"SystemExit({code})"})
        except Exception as exc:
            result.update({"hi": math.nan, "hi_success": False, "hi_error": repr(exc)})
    else:
        result.update({"hi": math.nan, "hi_success": False, "hi_error": "skipped by --metrics"})

    if "mlp_pi" in selected:
        try:
            mlp_pi, mlp_ti = compute_mlp_pi_ti(
                X_train,
                y_train,
                X_val,
                y_val,
                n_classes=args.n_classes,
            )
            result.update(
                {
                    "mlp_pi": safe_float(mlp_pi),
                    "mlp_ti": safe_float(mlp_ti),
                    "mlp_pi_success": True,
                    "mlp_pi_error": "",
                }
            )
        except SystemExit as exc:
            code = getattr(exc, "code", 1)
            result.update(
                {"mlp_pi": math.nan, "mlp_ti": math.nan, "mlp_pi_success": False, "mlp_pi_error": f"SystemExit({code})"}
            )
        except Exception as exc:
            result.update({"mlp_pi": math.nan, "mlp_ti": math.nan, "mlp_pi_success": False, "mlp_pi_error": repr(exc)})
    else:
        result.update(
            {"mlp_pi": math.nan, "mlp_ti": math.nan, "mlp_pi_success": False, "mlp_pi_error": "skipped by --metrics"}
        )

    if "mi" in selected:
        if dimension <= args.max_dim_mi:
            try:
                X_mi = X_train
                y_mi = y_train
                max_mi_samples = int(getattr(args, "max_mi_samples", 0) or 0)
                if max_mi_samples > 0 and int(len(y_mi)) > max_mi_samples:
                    rng = np.random.default_rng(0)
                    idx = rng.choice(int(len(y_mi)), size=max_mi_samples, replace=False)
                    X_mi = X_mi[idx]
                    y_mi = y_mi[idx]
                mi, ok = compute_mi(X_mi, y_mi, max_dim_for_estimate=args.max_dim_mi)
                err = "" if ok else "estimator returned not stable"
                if max_mi_samples > 0 and int(len(y_train)) > max_mi_samples:
                    suffix = f" (MI subsampled {max_mi_samples}/{len(y_train)})"
                    err = (err + suffix).strip()
                result.update({"mi": safe_float(mi) if ok else math.nan, "mi_success": bool(ok), "mi_error": err})
            except SystemExit as exc:
                code = getattr(exc, "code", 1)
                result.update({"mi": math.nan, "mi_success": False, "mi_error": f"SystemExit({code})"})
            except Exception as exc:
                result.update({"mi": math.nan, "mi_success": False, "mi_error": repr(exc)})
        else:
            result.update({"mi": math.nan, "mi_success": False, "mi_error": f"Dimension {dimension} > max {args.max_dim_mi}"})
    else:
        result.update({"mi": math.nan, "mi_success": False, "mi_error": "skipped by --metrics"})

    return result

def row_complete(row: pd.Series) -> bool:
    if row.empty:
        return False
    return str(row.get("status", "")).strip() == "complete"


def normalize_checkpoint(path: Path) -> pd.DataFrame:
    if path.exists() and path.stat().st_size > 0:
        df = pd.read_csv(path)
    else:
        df = pd.DataFrame(columns=CHECKPOINT_COLUMNS)
    for col in CHECKPOINT_COLUMNS:
        if col not in df.columns:
            df[col] = "" if col.endswith("_error") or col in {"status", "source_artifact"} else np.nan
    return df[CHECKPOINT_COLUMNS]


def write_summary(df: pd.DataFrame, output_dir: Path) -> None:
    rows = []
    metrics = [
        ("pi", "ePI_or_PI", "pi_success"),
        ("hi", "eHI_or_HI", "hi_success"),
        ("mlp_pi", "MLP_PI", "mlp_pi_success"),
        ("mi", "GKOV_MI", "mi_success"),
    ]
    for (dataset, dimension), group in df.groupby(["dataset", "dimension"], dropna=False):
        for metric, label, success_col in metrics:
            values = pd.to_numeric(group[metric], errors="coerce") if metric in group.columns else pd.Series(dtype=float)
            success = group[success_col].fillna(False).astype(bool) & values.notna()
            rows.append(
                {
                    "dataset": dataset,
                    "dimension": int(dimension),
                    "metric": label,
                    "status": "numeric" if bool(success.any()) else "failed",
                    "success_fraction": float(success.mean()) if len(success) else 0.0,
                    "value_mean": float(values[success].mean()) if bool(success.any()) else math.nan,
                    "value_std": float(values[success].std(ddof=1)) if int(success.sum()) > 1 else math.nan,
                    "success_count": int(success.sum()) if len(success) else 0,
                    "run_count": int(len(group)),
                    "source_artifact": str(output_dir / "real_nonbi_dimcap_checkpoint.csv"),
                    "note": "fresh real-dataset dimension-capped baseline pass",
                }
            )
    pd.DataFrame(rows).to_csv(output_dir / "real_nonbi_dimcap_summary.csv", index=False)


def upsert_result(output_dir: Path, result: Dict[str, object]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = output_dir / "real_nonbi_dimcap_checkpoint.csv"
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


def maybe_skip_existing(args: argparse.Namespace, dataset: str, dimension: int, seed: int) -> bool:
    if args.force:
        return False
    checkpoint = Path(args.output_dir) / "real_nonbi_dimcap_checkpoint.csv"
    if not checkpoint.exists() or checkpoint.stat().st_size == 0:
        return False
    df = pd.read_csv(checkpoint)
    if df.empty:
        return False
    mask = (
        (df["dataset"].astype(str) == dataset)
        & (pd.to_numeric(df["dimension"], errors="coerce") == int(dimension))
        & (pd.to_numeric(df["seed"], errors="coerce") == int(seed))
    )
    if not mask.any():
        return False
    return row_complete(df.loc[mask].iloc[-1])


def run_task(args: argparse.Namespace) -> None:
    task_id = args.task_id
    if task_id is None:
        task_id = int(os.environ.get("ARRAY_TASK_ID", "0"))
    task = read_task(Path(args.manifest), task_id)
    dataset = str(task["dataset"])
    dimension = int(task["dimension"])
    seed = int(task["seed"])

    if maybe_skip_existing(args, dataset, dimension, seed):
        print(f"[skip] existing complete row for dataset={dataset} d={dimension} seed={seed}")
        return

    started = utc_now()
    tic = time.monotonic()
    print(f"[run] task={task_id} dataset={dataset} d={dimension} seed={seed}")
    output_dir = Path(args.output_dir)
    base_result = {
        "dataset": dataset,
        "dimension": dimension,
        "seed": seed,
        "status": "failed",
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
        traces, labels, source = load_dataset_window(args, dataset, dimension)
        X_train, y_train, X_val, y_val = split_train_val(traces, labels, seed, args.n_val)
        result = dict(base_result)
        result.update(
            {
                "status": "complete",
                "n_total_used": int(len(labels)),
                "n_train": int(len(y_train)),
                "n_val": int(len(y_val)),
                "source_artifact": source,
            }
        )
        result.update(metric_attempts(args, X_train, y_train, X_val, y_val, dimension))
    except BaseException as exc:
        result = dict(base_result)
        result.update(
            {
                "pi_success": False,
                "pi_error": repr(exc),
                "hi_success": False,
                "hi_error": repr(exc),
                "mlp_pi_success": False,
                "mlp_pi_error": repr(exc),
                "mi_success": False,
                "mi_error": repr(exc),
            }
        )
    result["finished_at"] = utc_now()
    result["elapsed_sec"] = round(time.monotonic() - tic, 3)
    upsert_result(output_dir, result)
    flags = ", ".join(f"{name}={result.get(name + '_success')}" for name in ["pi", "hi", "mlp_pi", "mi"])
    print(f"[done] dataset={dataset} d={dimension} seed={seed} {flags} elapsed={result['elapsed_sec']}s status={result.get('status')}")


def main() -> None:
    args = parse_args()
    if args.write_manifest_only:
        write_manifest(args)
        return
    run_task(args)


if __name__ == "__main__":
    main()
