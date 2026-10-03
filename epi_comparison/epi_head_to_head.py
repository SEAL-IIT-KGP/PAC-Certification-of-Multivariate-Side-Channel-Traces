#!/usr/bin/env python3
"""Same-model BI versus IUH+22 EPI on native long-dimensional SCA traces.

The protocol uses three disjoint, class-balanced profiling subsets:
  1. selection: choose one frozen model from the declared TCHES20 suite;
  2. calibration: fit the single IUH+22 power parameter beta;
  3. evaluation: report both BI and EPI from the same frozen model.

BI is an exact finite-benchmark lower bound for the balanced evaluation records.
EPI is a held-out Shannon cross-information estimate.  The two bit values have
different entropy semantics and are not treated as an ordinal horse race.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import os
import platform
import socket
import sys
import time
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar


DATASETS = ("ascad_desync_0", "ascad_desync_50", "ascad_desync_100", "aes_hd", "aes_rd", "dpav4")

KEYWORDS = {
    "ascad_desync_100": ("ascad_desync_100", "ascad_desync100"),
    "ascad_desync_50": ("ascad_desync_50", "ascad_desync50"),
    "ascad_desync_0": ("ascad_desync_0", "ascad_desync0"),
    "aes_hd": ("aes_hd",),
    "aes_rd": ("aes_rd",),
    "dpav4": ("dpav4", "dpa_v4", "dpav4_"),
}


def logsumexp(array: np.ndarray, axis: int, keepdims: bool = False) -> np.ndarray:
    """Stable NumPy-only log-sum-exp, so SciPy and JAX never have to be mixed in one process."""
    values = np.asarray(array, dtype=np.float64)
    maximum = np.max(values, axis=axis, keepdims=True)
    result = maximum + np.log(np.sum(np.exp(values - maximum), axis=axis, keepdims=True))
    if not keepdims:
        result = np.squeeze(result, axis=axis)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-index", type=int, default=None)
    parser.add_argument("--dataset", choices=DATASETS)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--tches20-src-dir", type=Path, required=True)
    parser.add_argument("--models-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260721)
    parser.add_argument("--max-models", type=int, default=0)
    parser.add_argument("--selection-fraction", type=float, default=0.2)
    parser.add_argument("--calibration-fraction", type=float, default=0.2)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--beta-max", type=float, default=64.0)
    parser.add_argument("--scarcity-reps", type=int, default=20)
    parser.add_argument("--bootstrap-reps", type=int, default=2000)
    return parser.parse_args()


def dataset_name(args: argparse.Namespace) -> str:
    if args.dataset:
        return args.dataset
    idx = args.dataset_index
    if idx is None:
        idx = int(os.environ.get("SLURM_ARRAY_TASK_ID", "0"))
    if not 0 <= idx < len(DATASETS):
        raise IndexError(f"dataset index {idx} outside 0..{len(DATASETS) - 1}")
    return DATASETS[idx]


def load_dataset(name: str, dataset_dir: Path, src_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    sys.path.insert(0, str(src_dir))
    from dataLoaders import load_aes_hd, load_aes_rd, load_ascad, load_dpav4

    loaders = {
        "ascad_desync_0": lambda: load_ascad(str(dataset_dir / "ASCAD_dataset/ASCAD.h5")),
        "ascad_desync_50": lambda: load_ascad(str(dataset_dir / "ASCAD_dataset/ASCAD_desync50.h5")),
        "ascad_desync_100": lambda: load_ascad(str(dataset_dir / "ASCAD_dataset/ASCAD_desync100.h5")),
        # The upstream loaders concatenate filenames onto these strings and
        # therefore require a trailing separator.
        "aes_hd": lambda: load_aes_hd(str(dataset_dir / "AES_HD_dataset") + os.sep),
        "aes_rd": lambda: load_aes_rd(str(dataset_dir / "AES_RD_dataset") + os.sep),
        "dpav4": lambda: load_dpav4(str(dataset_dir / "DPAv4_dataset") + os.sep),
    }
    payload = loaders[name]()
    traces = np.asarray(payload[0], dtype=np.float32)
    labels = np.asarray(payload[1], dtype=np.int64).reshape(-1)
    if traces.ndim != 2 or len(traces) != len(labels):
        raise AssertionError(f"invalid dataset shapes traces={traces.shape} labels={labels.shape}")
    return traces, labels


def discover_models(root: Path, dataset: str, max_models: int) -> list[Path]:
    paths = []
    for path in sorted(root.rglob("*.hdf5")):
        lower = path.name.lower()
        if any(token in lower for token in KEYWORDS[dataset]):
            paths.append(path)
    if max_models > 0:
        paths = paths[:max_models]
    if not paths:
        raise FileNotFoundError(f"no {dataset} .hdf5 models below {root}")
    return paths


def balanced_three_way_split(
    labels: np.ndarray, seed: int, selection_fraction: float, calibration_fraction: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    classes, counts = np.unique(labels, return_counts=True)
    if len(classes) != 256 or not np.array_equal(classes, np.arange(256)):
        raise AssertionError(f"expected all 256 byte classes, got {len(classes)}")
    per_class = int(np.min(counts))
    n_select = max(1, int(math.floor(per_class * selection_fraction)))
    n_cal = max(1, int(math.floor(per_class * calibration_fraction)))
    n_eval = per_class - n_select - n_cal
    if n_eval < 2:
        raise ValueError(
            f"not enough records/class for three-way split: min={per_class}, selection={n_select}, calibration={n_cal}"
        )
    rng = np.random.default_rng(seed)
    select_parts, cal_parts, eval_parts = [], [], []
    for cls in classes:
        idx = np.flatnonzero(labels == cls)
        rng.shuffle(idx)
        select_parts.append(idx[:n_select])
        cal_parts.append(idx[n_select : n_select + n_cal])
        eval_parts.append(idx[n_select + n_cal : n_select + n_cal + n_eval])
    select = np.concatenate(select_parts)
    calibration = np.concatenate(cal_parts)
    evaluation = np.concatenate(eval_parts)
    rng.shuffle(select)
    rng.shuffle(calibration)
    rng.shuffle(evaluation)
    return select, calibration, evaluation, classes, per_class


def prepare_input(model, traces: np.ndarray) -> np.ndarray:
    expected = tuple(int(x) for x in model.input_shape[1:] if x is not None)
    x = traces
    if len(expected) == 2 and expected[-1] == 1 and x.ndim == 2:
        width = expected[0]
        if x.shape[1] > width:
            x = x[:, :width]
        elif x.shape[1] < width:
            x = np.pad(x, ((0, 0), (0, width - x.shape[1])))
        x = x[..., None]
    elif len(expected) == 1 and x.ndim == 2:
        width = expected[0]
        if x.shape[1] > width:
            x = x[:, :width]
        elif x.shape[1] < width:
            x = np.pad(x, ((0, 0), (0, width - x.shape[1])))
    return np.asarray(x, dtype=np.float32)


def predict_logp(model, traces: np.ndarray, batch_size: int) -> np.ndarray:
    raw = model.predict(prepare_input(model, traces), batch_size=batch_size, verbose=0)
    if isinstance(raw, (tuple, list)):
        if len(raw) != 1:
            raise ValueError(f"expected one model output, got {len(raw)}")
        raw = raw[0]
    raw = np.asarray(raw, dtype=np.float64)
    if raw.ndim != 2 or raw.shape[1] != 256:
        raise AssertionError(f"expected (n,256) output, got {raw.shape}")
    row_sums = raw.sum(axis=1)
    if np.all(raw >= 0.0) and np.all(np.isfinite(row_sums)) and np.median(np.abs(row_sums - 1.0)) < 1e-3:
        logp = np.log(np.maximum(raw, np.finfo(np.float64).tiny))
    else:
        logp = raw
    return logp - logsumexp(logp, axis=1, keepdims=True)


def information_records(logp: np.ndarray, labels: np.ndarray, beta: float) -> np.ndarray:
    scaled = float(beta) * logp
    selected = scaled[np.arange(len(labels)), labels]
    normalized = selected - logsumexp(scaled, axis=1)
    return 8.0 + normalized / math.log(2.0)


def fit_beta(logp: np.ndarray, labels: np.ndarray, beta_max: float) -> tuple[float, float]:
    def nll(beta: float) -> float:
        return float(8.0 - np.mean(information_records(logp, labels, beta)))

    result = minimize_scalar(
        nll, bounds=(0.0, float(beta_max)), method="bounded", options={"xatol": 1e-8, "maxiter": 500}
    )
    candidates = [(0.0, nll(0.0)), (float(result.x), float(result.fun)), (float(beta_max), nll(beta_max))]
    beta, loss = min(candidates, key=lambda item: item[1])
    if not np.isfinite(loss):
        raise FloatingPointError("non-finite optimized ECE")
    return beta, loss


def bootstrap_mean_ci(values: np.ndarray, reps: int, seed: int) -> tuple[float, float]:
    if reps <= 0:
        return math.nan, math.nan
    rng = np.random.default_rng(seed)
    means = np.empty(reps, dtype=np.float64)
    for start in range(0, reps, 200):
        stop = min(reps, start + 200)
        idx = rng.integers(0, len(values), size=(stop - start, len(values)))
        means[start:stop] = np.mean(values[idx], axis=1)
    lo, hi = np.quantile(means, (0.025, 0.975))
    return float(lo), float(hi)


def scarcity_rows(
    logp_cal: np.ndarray,
    y_cal: np.ndarray,
    logp_eval: np.ndarray,
    y_eval: np.ndarray,
    beta_max: float,
    reps: int,
    seed: int,
) -> list[dict]:
    by_class = {cls: np.flatnonzero(y_cal == cls) for cls in range(256)}
    max_per_class = min(len(idx) for idx in by_class.values())
    sizes = [x for x in (1, 2, 4, 8, 16, 32, 64, 128) if x <= max_per_class]
    if max_per_class not in sizes:
        sizes.append(max_per_class)
    rng = np.random.default_rng(seed)
    rows = []
    for size in sizes:
        betas, epis = [], []
        for _ in range(reps):
            idx = np.concatenate([rng.choice(by_class[cls], size=size, replace=False) for cls in range(256)])
            beta, _ = fit_beta(logp_cal[idx], y_cal[idx], beta_max)
            epi = float(np.mean(information_records(logp_eval, y_eval, beta)))
            betas.append(beta)
            epis.append(epi)
        rows.append(
            {
                "calibration_per_class": int(size),
                "calibration_records": int(size * 256),
                "repetitions": int(reps),
                "beta_mean": float(np.mean(betas)),
                "beta_std": float(np.std(betas, ddof=1)) if reps > 1 else 0.0,
                "epi_mean_bits": float(np.mean(epis)),
                "epi_std_bits": float(np.std(epis, ddof=1)) if reps > 1 else 0.0,
                "epi_min_bits": float(np.min(epis)),
                "epi_max_bits": float(np.max(epis)),
            }
        )
    return rows


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    name = dataset_name(args)
    started = time.time()
    out_dir = args.output_dir / name
    out_dir.mkdir(parents=True, exist_ok=True)

    traces, labels = load_dataset(name, args.dataset_dir, args.tches20_src_dir)
    select_idx, cal_idx, eval_idx, classes, min_count = balanced_three_way_split(
        labels, args.seed, args.selection_fraction, args.calibration_fraction
    )
    model_paths = discover_models(args.models_dir, name, args.max_models)

    import tensorflow as tf

    selection_rows = []
    best = None
    for ordinal, model_path in enumerate(model_paths):
        model = tf.keras.models.load_model(model_path, compile=False)
        logp = predict_logp(model, traces[select_idx], args.batch_size)
        accuracy = float(np.mean(np.argmax(logp, axis=1) == labels[select_idx]))
        row = {"ordinal": ordinal, "model": model_path.name, "path": str(model_path), "selection_accuracy": accuracy}
        selection_rows.append(row)
        if best is None or (accuracy, model_path.name) > (best[0], best[1].name):
            best = (accuracy, model_path)
        del model, logp
        tf.keras.backend.clear_session()
        gc.collect()
        print(
            f"{name} selection {ordinal + 1}/{len(model_paths)} accuracy={accuracy:.8f} {model_path.name}", flush=True
        )

    assert best is not None
    best_accuracy, best_path = best
    model = tf.keras.models.load_model(best_path, compile=False)
    logp_cal = predict_logp(model, traces[cal_idx], args.batch_size)
    logp_eval = predict_logp(model, traces[eval_idx], args.batch_size)
    y_cal = labels[cal_idx]
    y_eval = labels[eval_idx]

    raw_records = information_records(logp_eval, y_eval, 1.0)
    beta_hat, cal_ece = fit_beta(logp_cal, y_cal, args.beta_max)
    epi_records = information_records(logp_eval, y_eval, beta_hat)
    predictions = np.argmax(logp_eval, axis=1)
    exact_success = predictions == y_eval
    success = float(np.mean(exact_success))
    finite_bi = max(0.0, math.log2(256.0 * success)) if success > 0 else 0.0
    raw_ci = bootstrap_mean_ci(raw_records, args.bootstrap_reps, args.seed + 1)
    epi_ci = bootstrap_mean_ci(epi_records, args.bootstrap_reps, args.seed + 2)

    temperature_rows = []
    for temperature in (0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 10.0, 20.0):
        stressed_cal = logp_cal / temperature
        stressed_eval = logp_eval / temperature
        # Under q -> H_(1/T)[q], the EPI optimum maps as beta -> beta*T.
        # Scale the numerical search ceiling so the stress test does not
        # manufacture a boundary artifact at large temperatures.
        beta_t, _ = fit_beta(stressed_cal, y_cal, args.beta_max * max(1.0, temperature))
        epi_t = float(np.mean(information_records(stressed_eval, y_eval, beta_t)))
        pred_t = np.argmax(stressed_eval, axis=1)
        temperature_rows.append(
            {
                "temperature": temperature,
                "raw_pi_bits": float(np.mean(information_records(stressed_eval, y_eval, 1.0))),
                "beta_hat": beta_t,
                "epi_bits": epi_t,
                "argmax_preserved": bool(np.array_equal(pred_t, predictions)),
                "finite_bi_bits": finite_bi,
            }
        )

    scarcity = scarcity_rows(logp_cal, y_cal, logp_eval, y_eval, args.beta_max, args.scarcity_reps, args.seed + 3)

    selection_csv = out_dir / "model_selection.csv"
    with selection_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(selection_rows[0]))
        writer.writeheader()
        writer.writerows(selection_rows)
    temp_csv = out_dir / "temperature_stress.csv"
    with temp_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(temperature_rows[0]))
        writer.writeheader()
        writer.writerows(temperature_rows)
    scarcity_csv = out_dir / "calibration_scarcity.csv"
    with scarcity_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(scarcity[0]))
        writer.writeheader()
        writer.writerows(scarcity)

    score_path = out_dir / "heldout_scores.npz"
    np.savez_compressed(
        score_path,
        calibration_logp=logp_cal.astype(np.float32),
        calibration_labels=y_cal.astype(np.int16),
        evaluation_logp=logp_eval.astype(np.float32),
        evaluation_labels=y_eval.astype(np.int16),
        evaluation_predictions=predictions.astype(np.int16),
        evaluation_success=exact_success.astype(np.uint8),
    )

    result = {
        "schema": "epi_longdim_head_to_head_v1",
        "dataset": name,
        "native_dimension": int(traces.shape[1]),
        "total_records": int(len(labels)),
        "classes": int(len(classes)),
        "minimum_records_per_class": min_count,
        "selection_records": int(len(select_idx)),
        "calibration_records": int(len(cal_idx)),
        "evaluation_records": int(len(eval_idx)),
        "selection_calibration_evaluation_disjoint": bool(
            not np.intersect1d(select_idx, cal_idx).size
            and not np.intersect1d(select_idx, eval_idx).size
            and not np.intersect1d(cal_idx, eval_idx).size
        ),
        "suite_size": len(model_paths),
        "selected_model": best_path.name,
        "selected_model_path": str(best_path),
        "selection_accuracy": best_accuracy,
        "evaluation_accuracy": success,
        "finite_benchmark_bi_lower_bits": finite_bi,
        "raw_pi_bits": float(np.mean(raw_records)),
        "raw_pi_ci95": list(raw_ci),
        "epi_beta_hat": beta_hat,
        "calibration_ece_bits": cal_ece,
        "epi_bits": float(np.mean(epi_records)),
        "epi_ci95": list(epi_ci),
        "metric_scope": {
            "BI": "exact min-entropy leakage lower bound on uniform finite evaluation records",
            "EPI": "IUH+22 power-optimized Shannon cross-information; beta fit on disjoint calibration records",
        },
        "artifacts": {
            "model_selection": selection_csv.name,
            "temperature_stress": temp_csv.name,
            "calibration_scarcity": scarcity_csv.name,
            "heldout_scores": score_path.name,
        },
        "reproducibility": {
            "seed": args.seed,
            "beta_max": args.beta_max,
            "selection_fraction_per_class": args.selection_fraction,
            "calibration_fraction_per_class": args.calibration_fraction,
            "bootstrap_reps": args.bootstrap_reps,
            "scarcity_reps": args.scarcity_reps,
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "python": sys.version,
            "numpy": np.__version__,
            "tensorflow": tf.__version__,
            "slurm_job_id": os.environ.get("SLURM_JOB_ID", ""),
            "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID", ""),
            "elapsed_sec": time.time() - started,
        },
    }
    result_path = out_dir / "result.json"
    result_path.write_text(json.dumps(result, indent=2) + "\n")
    checksum_path = out_dir / "SHA256SUMS"
    primary = [result_path, selection_csv, temp_csv, scarcity_csv, score_path]
    checksum_path.write_text("".join(f"{file_sha256(path)}  {path.name}\n" for path in primary))
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
