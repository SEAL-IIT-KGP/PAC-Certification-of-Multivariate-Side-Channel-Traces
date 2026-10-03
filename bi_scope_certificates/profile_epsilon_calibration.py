#!/usr/bin/env python3
"""Profile score variation and relaxed-margin counts for the radius calibration.

For each of the twelve selected reference models (MLP, CNN, Transformer on
ASCAD d50, ASCAD d100, ASCAD random key, and AES-RD), the attack set is
shuffled once (seed 20260721) and split into 20% calibration traces and 80%
certification traces. The script writes

- `epsilon_margin_counts.csv`: relaxed-margin counts S_loc(eps) on both splits
  for the nine predeclared radii, and
- `epsilon_variant_distances.csv`: the largest normalized-score change on the
  calibration traces under repeated FP32 inference, other batch sizes, FP16
  storage, quantization, and additive input noise.

`paper_results.figure2_biloc_selected_centers` reads both files, applies the
calibration rule, and certifies on the 80% split.

The six MLP/CNN reference models on ASCAD d50, ASCAD d100, and AES-RD are
public TCHES20 pretrained models (`--public-only`). The ASCAD random-key
MLP/CNN and the four EstraNet Transformer checkpoints are the released
reference checkpoints (see `paper_results/README.md`).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import platform
import re
import sys
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np

from bi_scope_certificates.compute_bi_loc_margin_curves import (
    adapt_input,
    build_transformer,
    ensure_jax_stub,
    import_estranet,
    labels_from_dataset,
    load_ascad_random_attack,
    load_tches20_attack,
    margins_from_probs,
    margins_from_scores,
    normalized_probs,
)


EPS_GRID = [0.0, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2]
DATASET_SEED_OFFSET = {"ascad_desync_50": 50, "ascad_desync_100": 100, "ascad_random_key": 200, "aes_rd": 300}


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def checkpoint_path(spec: dict[str, Any]) -> str:
    return str(Path(spec["checkpoint_dir"]) / f"trans_long-{int(spec['checkpoint_index'])}")


def dataset_specs(dataset: str, args: argparse.Namespace) -> list[dict[str, Any]]:
    tches_root = args.tches_root
    models = tches_root / "models/pretrained_models/models"
    trained = args.trained_models_dir
    estranet = args.estranet_root
    checkpoints = args.checkpoint_root
    common_keras: dict[str, list[tuple[str, str, Path]]] = {
        "ascad_desync_50": [
            (
                "MLP",
                "noConv1_ascad_desync_50_no_preprocessing_0",
                models / "noConv1_ascad_desync_50_no_preprocessing_0.hdf5",
            ),
            ("CNN", "zaid_ascad_desync_50_no_preprocessing_3", models / "zaid_ascad_desync_50_no_preprocessing_3.hdf5"),
        ],
        "ascad_desync_100": [
            (
                "MLP",
                "noConv1_ascad_desync_100_no_preprocessing_6",
                models / "noConv1_ascad_desync_100_no_preprocessing_6.hdf5",
            ),
            (
                "CNN",
                "zaid_ascad_desync_100_no_preprocessing_5",
                models / "zaid_ascad_desync_100_no_preprocessing_5.hdf5",
            ),
        ],
        "aes_rd": [
            ("MLP", "noConv1_aes_rd_no_preprocessing_2", models / "noConv1_aes_rd_no_preprocessing_2.hdf5"),
            ("CNN", "zaid_aes_rd_no_preprocessing_9", models / "zaid_aes_rd_no_preprocessing_9.hdf5"),
        ],
        "ascad_random_key": [
            ("MLP", "ascad_random_key_mlp_seed0", trained / "ascad_random_key_mlp_seed0_best.hdf5"),
            ("CNN", "ascad_random_key_cnn_seed0", trained / "ascad_random_key_cnn_seed0_best.hdf5"),
        ],
    }
    specs: list[dict[str, Any]] = []
    for arch, center, model in common_keras[dataset]:
        specs.append(
            {
                "kind": "keras",
                "dataset": dataset,
                "architecture": arch,
                "center_id": center,
                "model_path": str(model),
                "tches_root": str(tches_root),
                "random_h5": str(args.ascad_random_h5),
                "max_samples": 100000 if dataset == "ascad_random_key" else (12500 if dataset == "aes_rd" else 25000),
            }
        )

    transformer_specs: dict[str, dict[str, Any]] = {
        "ascad_desync_50": {
            "center_id": "ascad_desync_50_estranet_ckpt60",
            "data_path": args.ascad_fixed_dir / "ASCAD_desync50.h5",
            "checkpoint_dir": checkpoints / "ascadf_desync50",
            "checkpoint_index": 60,
            "input_length": 700,
            "pool_size": 5,
            "beta_hat_2": 150,
            "A": 24.324125289916992,
        },
        "ascad_desync_100": {
            "center_id": "ascad_desync_100_estranet_ckpt40",
            "data_path": args.ascad_fixed_dir / "ASCAD_desync100.h5",
            "checkpoint_dir": checkpoints / "ascadf_desync100",
            "checkpoint_index": 40,
            "input_length": 700,
            "pool_size": 5,
            "beta_hat_2": 150,
            "A": 14.253807067871094,
        },
        "ascad_random_key": {
            "center_id": "ascad_random_key_estranet_ckpt80",
            "data_path": args.ascad_random_h5,
            "checkpoint_dir": checkpoints / "ascadr",
            "checkpoint_index": 80,
            "input_length": 1400,
            "pool_size": 10,
            "beta_hat_2": 50,
            "A": 32.28862380981445,
        },
        "aes_rd": {
            "center_id": "aes_rd_estranet_fstd_pool5_lr5e4_6977529_2_ckpt11",
            "data_path": args.aesrd_estranet_h5,
            "checkpoint_dir": checkpoints / "aes_rd_fstd_pool5_lr5e4",
            "checkpoint_index": 11,
            "input_length": 3500,
            "pool_size": 5,
            "beta_hat_2": 150,
            "A": 102.09283447265625,
        },
    }
    t_spec = dict(transformer_specs[dataset])
    t_spec.update(
        {
            "kind": "estranet",
            "dataset": dataset,
            "architecture": "Transformer",
            "project_root": str(estranet),
            "max_samples": 100000 if dataset == "ascad_random_key" else None,
        }
    )
    specs.append(t_spec)
    return specs


def split_indices(n: int, dataset: str, seed: int, fraction: float) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed + DATASET_SEED_OFFSET[dataset])
    order = rng.permutation(n)
    n_cal = max(1, min(n - 1, int(round(n * fraction))))
    return np.sort(order[:n_cal]), np.sort(order[n_cal:])


def load_keras_data(spec: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    limit = int(spec["max_samples"])
    if spec["dataset"] == "ascad_random_key":
        return load_ascad_random_attack(Path(spec["random_h5"]), limit)
    return load_tches20_attack(Path(spec["tches_root"]), spec["dataset"], limit)


def load_estranet_data(spec: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, Any]:
    ensure_jax_stub()
    data_utils, Transformer = import_estranet(Path(spec["project_root"]))
    dataset = data_utils.Dataset(
        data_path=str(spec["data_path"]), split="test", input_length=int(spec["input_length"]), data_desync=0
    )
    traces = np.concatenate([np.asarray(x) for x in dataset.traces], axis=0)
    labels = labels_from_dataset(dataset)
    limit = spec.get("max_samples")
    if limit:
        traces = traces[: int(limit)]
        labels = labels[: int(limit)]
    return traces, labels, Transformer


def keras_predictor(spec: dict[str, Any], x: np.ndarray) -> tuple[Callable[[np.ndarray, int], np.ndarray], str, float]:
    import tensorflow as tf

    model = tf.keras.models.load_model(spec["model_path"], compile=False)

    def predict(values: np.ndarray, batch_size: int) -> np.ndarray:
        raw = model.predict(adapt_input(model, values), batch_size=batch_size, verbose=0)
        raw = raw[0] if isinstance(raw, (list, tuple)) else raw
        return normalized_probs(np.asarray(raw)).astype(np.float64)

    return predict, "probability_simplex_A=1", 1.0


def estranet_predictor(
    spec: dict[str, Any], Transformer: Any
) -> tuple[Callable[[np.ndarray, int], np.ndarray], str, float]:
    import tensorflow as tf

    model = build_transformer(spec, Transformer)
    optimizer = tf.keras.optimizers.Adam(learning_rate=2.5e-4)
    checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)
    ckpt = checkpoint_path(spec)
    checkpoint.restore(
        ckpt, options=tf.train.CheckpointOptions(experimental_io_device="/job:localhost")
    ).expect_partial()
    A = float(spec["A"])

    def predict(values: np.ndarray, batch_size: int) -> np.ndarray:
        n_values = len(values)
        if n_values == 0:
            return np.empty((0, 256), dtype=np.float64)
        pad = (-n_values) % batch_size
        fixed = values.astype(np.float32, copy=False)
        if pad:
            fixed = np.concatenate([fixed, np.repeat(fixed[-1:], pad, axis=0)], axis=0)
        outputs = []
        for start in range(0, len(fixed), batch_size):
            raw = model(tf.convert_to_tensor(fixed[start : start + batch_size]), training=False)
            raw = raw[0] if isinstance(raw, (list, tuple)) else raw
            outputs.append(np.asarray(raw))
        return np.concatenate(outputs, axis=0)[:n_values].astype(np.float64) / A

    return predict, "raw_logits_A=frozen_train_split_max_abs", A


def quantize_scores(scores: np.ndarray, bits: int, probability: bool) -> np.ndarray:
    levels = float((1 << bits) - 1)
    if probability:
        out = np.round(np.clip(scores, 0.0, 1.0) * levels) / levels
        sums = np.sum(out, axis=1, keepdims=True)
        bad = sums[:, 0] <= 0.0
        if np.any(bad):
            out[bad] = 1.0 / out.shape[1]
            sums = np.sum(out, axis=1, keepdims=True)
        return out / sums
    return np.round(np.clip(scores, -1.0, 1.0) * levels) / levels


def quantize_inputs(values: np.ndarray, bits: int) -> np.ndarray:
    x = values.astype(np.float64)
    lo = float(np.min(x))
    hi = float(np.max(x))
    if not math.isfinite(lo) or not math.isfinite(hi) or hi <= lo:
        return values.astype(np.float32)
    levels = float((1 << bits) - 1)
    return (np.round((x - lo) * levels / (hi - lo)) * (hi - lo) / levels + lo).astype(np.float32)


def distance_row(
    spec: dict[str, Any],
    variant: str,
    group: str,
    baseline: np.ndarray,
    candidate: np.ndarray,
    score_normalization: str,
    A: float,
) -> tuple[dict[str, Any], np.ndarray]:
    n = min(len(baseline), len(candidate))
    per_trace = np.max(np.abs(baseline[:n] - candidate[:n]), axis=1)
    return (
        {
            "dataset": spec["dataset"],
            "architecture": spec["architecture"],
            "center_id": spec["center_id"],
            "variant": variant,
            "variant_group": group,
            "score_normalization": score_normalization,
            "A": A,
            "n_cal": n,
            "distance_mean": float(np.mean(per_trace)),
            "distance_p99": float(np.quantile(per_trace, 0.99)),
            "distance_p999": float(np.quantile(per_trace, 0.999)),
            "distance_max": float(np.max(per_trace)),
            "coverage_1e-5": float(np.mean(per_trace <= 1e-5)),
            "coverage_3e-5": float(np.mean(per_trace <= 3e-5)),
            "coverage_1e-4": float(np.mean(per_trace <= 1e-4)),
            "coverage_3e-4": float(np.mean(per_trace <= 3e-4)),
            "coverage_1e-3": float(np.mean(per_trace <= 1e-3)),
        },
        per_trace,
    )


def center_profile(
    spec: dict[str, Any], output_dir: Path, seed: int, cal_fraction: float
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    started = time.time()
    if spec["kind"] == "keras":
        x, labels = load_keras_data(spec)
        predict, score_normalization, A = keras_predictor(spec, x)
        base_batch, small_batch, large_batch = 512, 64, 2048
        model_path = spec["model_path"]
    else:
        x, labels, Transformer = load_estranet_data(spec)
        predict, score_normalization, A = estranet_predictor(spec, Transformer)
        base_batch, small_batch, large_batch = 32, 8, 64
        model_path = checkpoint_path(spec)

    n = min(len(x), len(labels))
    x = x[:n]
    labels = labels[:n].astype(np.int64)
    cal_idx, eval_idx = split_indices(n, spec["dataset"], seed, cal_fraction)
    print(f"[{spec['center_id']}] baseline n={n} cal={len(cal_idx)} eval={len(eval_idx)}", flush=True)
    baseline_all = predict(x, base_batch)
    n = min(n, len(baseline_all))
    x = x[:n]
    labels = labels[:n]
    baseline_all = baseline_all[:n]
    cal_idx = cal_idx[cal_idx < n]
    eval_idx = eval_idx[eval_idx < n]
    x_cal = x[cal_idx]
    base_cal = baseline_all[cal_idx]
    probability = spec["kind"] == "keras"

    variants: list[tuple[str, str, np.ndarray]] = []
    variants.append(("repeat_base_batch", "fp32_repro", predict(x_cal, base_batch)))
    variants.append((f"batch_{small_batch}", "fp32_repro", predict(x_cal, small_batch)))
    variants.append((f"batch_{large_batch}", "fp32_repro", predict(x_cal, large_batch)))
    variants.append(
        ("input_fp16_storage", "fp16_storage", predict(x_cal.astype(np.float16).astype(np.float32), base_batch))
    )
    fp16_scores = base_cal.astype(np.float16).astype(np.float64)
    if probability:
        fp16_scores = normalized_probs(fp16_scores).astype(np.float64)
    variants.append(("score_fp16_storage", "fp16_storage", fp16_scores))
    for bits in (12, 13, 14, 16):
        variants.append((f"score_quant_{bits}bit", "score_quantization", quantize_scores(base_cal, bits, probability)))
    for bits in (8, 10, 12, 14):
        variants.append(
            (f"input_quant_{bits}bit", "input_quantization", predict(quantize_inputs(x_cal, bits), base_batch))
        )
    sigma = float(np.std(x_cal.astype(np.float64)))
    for rel in (1e-4, 3e-4, 1e-3):
        rng = np.random.default_rng(seed + DATASET_SEED_OFFSET[spec["dataset"]] + int(rel * 1e8))
        noisy = x_cal.astype(np.float32) + rng.normal(0.0, rel * sigma, size=x_cal.shape).astype(np.float32)
        variants.append((f"input_noise_{rel:g}sigma", "input_noise", predict(noisy, base_batch)))

    distance_rows: list[dict[str, Any]] = []
    fp32_distances: list[np.ndarray] = []
    for name, group, candidate in variants:
        row, per_trace = distance_row(spec, name, group, base_cal, candidate, score_normalization, A)
        distance_rows.append(row)
        if group == "fp32_repro":
            fp32_distances.append(per_trace)

    if probability:
        cal_margins = margins_from_probs(base_cal.astype(np.float32), labels[cal_idx])
        eval_margins = margins_from_probs(baseline_all[eval_idx].astype(np.float32), labels[eval_idx])
    else:
        cal_margins = margins_from_scores(base_cal, labels[cal_idx])
        eval_margins = margins_from_scores(baseline_all[eval_idx], labels[eval_idx])

    margin_rows: list[dict[str, Any]] = []
    for split, margins in (("calibration", cal_margins), ("evaluation", eval_margins)):
        for eps in EPS_GRID:
            successes = int(np.sum(margins >= -2.0 * eps))
            margin_rows.append(
                {
                    "dataset": spec["dataset"],
                    "architecture": spec["architecture"],
                    "center_id": spec["center_id"],
                    "split": split,
                    "epsilon": eps,
                    "n": len(margins),
                    "successes": successes,
                    "p_hat": successes / len(margins),
                    "score_normalization": score_normalization,
                    "A": A,
                    "model_path": model_path,
                }
            )

    safe_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(spec["center_id"]))
    np.savez_compressed(
        output_dir / f"profile_{safe_id}.npz",
        cal_margins=np.asarray(cal_margins, dtype=np.float32),
        eval_margins=np.asarray(eval_margins, dtype=np.float32),
        fp32_distance_max_per_trace=np.max(np.vstack(fp32_distances), axis=0).astype(np.float32),
    )
    meta = {
        "dataset": spec["dataset"],
        "architecture": spec["architecture"],
        "center_id": spec["center_id"],
        "kind": spec["kind"],
        "model_path": model_path,
        "n_total": n,
        "n_calibration": len(cal_idx),
        "n_evaluation": len(eval_idx),
        "score_normalization": score_normalization,
        "A": A,
        "elapsed_seconds": time.time() - started,
    }
    return distance_rows, margin_rows, meta


def main() -> None:
    repo = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True, choices=sorted(DATASET_SEED_OFFSET))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tches-root", type=Path, default=repo / "datasets/raw/tches20/TCHES20V3_CNN_SCA")
    parser.add_argument("--estranet-root", type=Path, default=repo / "external/EstraNet")
    parser.add_argument("--checkpoint-root", type=Path, default=repo / "checkpoints/estranet")
    parser.add_argument("--trained-models-dir", type=Path, default=repo / "checkpoints/keras_ascad_random_key")
    parser.add_argument("--ascad-random-h5", type=Path, default=repo / "datasets/raw/ascad_random_key/ASCAD.h5")
    parser.add_argument(
        "--ascad-fixed-dir",
        type=Path,
        default=repo / "datasets/raw/tches20/TCHES20V3_CNN_SCA/datasets/ASCAD_dataset",
        help="Directory with ASCAD_desync50.h5 and ASCAD_desync100.h5 (the TCHES20 copies are the ANSSI files).",
    )
    parser.add_argument(
        "--aesrd-estranet-h5", type=Path, default=repo / "datasets/raw/aes_rd/AES_RD_estranet_feature_standardized.h5"
    )
    parser.add_argument(
        "--public-only",
        action="store_true",
        help="Profile only the reference models that are public TCHES20 pretrained models.",
    )
    parser.add_argument("--seed", type=int, default=20260721)
    parser.add_argument("--calibration-fraction", type=float, default=0.2)
    parser.add_argument("--only-architecture", choices=("MLP", "CNN", "Transformer"))
    parser.add_argument("--max-samples", type=int)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_distances: list[dict[str, Any]] = []
    all_margins: list[dict[str, Any]] = []
    centers: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    specs = dataset_specs(args.dataset, args)
    if args.only_architecture:
        specs = [spec for spec in specs if spec["architecture"] == args.only_architecture]
    if args.public_only:
        public_root = str(args.tches_root / "models")
        specs = [spec for spec in specs if spec["kind"] == "keras" and str(spec["model_path"]).startswith(public_root)]
        if not specs:
            print(f"{args.dataset}: no public TCHES20 reference model; nothing to do", flush=True)
            return
    if args.max_samples:
        for spec in specs:
            spec["max_samples"] = args.max_samples
    for spec in specs:
        try:
            distances, margins, meta = center_profile(spec, args.output_dir, args.seed, args.calibration_fraction)
            all_distances.extend(distances)
            all_margins.extend(margins)
            centers.append(meta)
            write_rows(args.output_dir / "epsilon_variant_distances.csv", all_distances)
            write_rows(args.output_dir / "epsilon_margin_counts.csv", all_margins)
        except Exception as exc:
            errors.append({"center_id": str(spec.get("center_id")), "error": repr(exc)})
            print(f"ERROR {spec.get('center_id')}: {exc!r}", flush=True)

    write_rows(args.output_dir / "epsilon_variant_distances.csv", all_distances)
    write_rows(args.output_dir / "epsilon_margin_counts.csv", all_margins)
    error_path = args.output_dir / "errors.csv"
    if errors:
        write_rows(error_path, errors)
    elif error_path.exists():
        error_path.unlink()
    manifest = {
        "dataset": args.dataset,
        "seed": args.seed,
        "calibration_fraction": args.calibration_fraction,
        "epsilon_grid": EPS_GRID,
        "python": sys.version,
        "platform": platform.platform(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID", ""),
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID", ""),
        "centers": centers,
        "errors": errors,
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps({"dataset": args.dataset, "centers": len(centers), "errors": len(errors)}, sort_keys=True),
        flush=True,
    )
    if errors or len(centers) != len(specs):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
