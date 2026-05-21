"""Shared helpers for the lightweight model-training examples."""

from __future__ import annotations

import argparse
import json
import pickle
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np


def add_dataset_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--dataset",
        default="ascad",
        choices=[
            "ascad",
            "aes_hd",
            "ascad_desync_0",
            "ascad_desync_50",
            "ascad_desync_100",
            "aes_rd",
            "dpav4",
        ],
        help="Dataset identifier understood by core.data_loader.",
    )
    parser.add_argument("--data-path", default="", help="Optional direct path to a dataset file.")
    parser.add_argument("--data-dir", default="datasets/raw", help="Root directory for prepared datasets.")
    parser.add_argument("--tches20-src-dir", default="", help="Optional source directory containing TCHES20 dataLoaders.py.")
    parser.add_argument("--max-samples", type=int, default=20000, help="Optional cap for quick local example training.")
    parser.add_argument("--window-start", type=int, default=None, help="Optional inclusive trace-window start.")
    parser.add_argument("--window-end", type=int, default=None, help="Optional exclusive trace-window end.")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--train-size", type=float, default=0.6)
    parser.add_argument("--val-size", type=float, default=0.2)
    parser.add_argument("--holdout-size", type=float, default=0.2)
    parser.add_argument("--stratify", action="store_true", help="Use stratified splits when every class has enough samples.")
    parser.add_argument("--out-dir", default="checkpoints/model_training", help="Local directory for checkpoints and metadata.")


def load_standardized_split(args: argparse.Namespace):
    from sklearn.preprocessing import StandardScaler

    from core.data_loader import create_data_split, load_dataset

    traces, labels = load_dataset(
        args.dataset,
        filepath=args.data_path or None,
        data_dir=args.data_dir,
        tches20_src_dir=args.tches20_src_dir,
    )
    traces = np.asarray(traces, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.int64)

    if args.window_start is not None or args.window_end is not None:
        start = 0 if args.window_start is None else args.window_start
        end = traces.shape[1] if args.window_end is None else args.window_end
        traces = traces[:, start:end]

    if args.max_samples and len(labels) > args.max_samples:
        rng = np.random.default_rng(args.seed)
        index = rng.choice(len(labels), size=args.max_samples, replace=False)
        traces = traces[index]
        labels = labels[index]

    split = create_data_split(
        traces,
        labels,
        train_size=args.train_size,
        val_size=args.val_size,
        holdout_size=args.holdout_size,
        random_state=args.seed,
        stratify=args.stratify,
    )

    scaler = StandardScaler()
    x_train = scaler.fit_transform(split.X_train.reshape(len(split.X_train), -1)).astype(np.float32)
    x_val = scaler.transform(split.X_val.reshape(len(split.X_val), -1)).astype(np.float32)
    x_holdout = scaler.transform(split.X_holdout.reshape(len(split.X_holdout), -1)).astype(np.float32)
    return x_train, split.y_train.astype(np.int64), x_val, split.y_val.astype(np.int64), x_holdout, split.y_holdout.astype(np.int64), scaler


def save_pickle(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(payload, handle)


def save_metadata(path: Path, args: argparse.Namespace, metrics: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"args": vars(args), "metrics": metrics}
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def metrics_dict(**kwargs: Any) -> dict[str, Any]:
    return {key: (asdict(value) if hasattr(value, "__dataclass_fields__") else value) for key, value in kwargs.items()}
