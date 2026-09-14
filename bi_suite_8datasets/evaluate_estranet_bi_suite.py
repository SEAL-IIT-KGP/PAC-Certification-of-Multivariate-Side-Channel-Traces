#!/usr/bin/env python3
"""Evaluate EstraNet checkpoints as a declared BI^{suite} success-count set.

The script imports the existing EstraNet project code, restores each declared
checkpoint, computes single-target
classification success on the attack split, and writes one CSV row per model.
Key-rank/GE evidence is intentionally not used as BI input.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from pathlib import Path
from typing import Iterable, List

import numpy as np
import tensorflow as tf


def parse_checkpoint_indices(text: str) -> List[int]:
    out = []
    for part in text.replace(";", ",").split(","):
        part = part.strip()
        if not part:
            continue
        out.append(int(part))
    return out


def discover_indices(checkpoint_dir: Path) -> List[int]:
    indices = []
    for path in checkpoint_dir.glob("trans_long-*.index"):
        match = re.search(r"-(\d+)\.index$", path.name)
        if match:
            indices.append(int(match.group(1)))
    return sorted(set(indices))


def load_modules(project_root: Path):
    sys.path.insert(0, str(project_root))
    import data_utils  # type: ignore
    import data_utils_ches25  # type: ignore
    from transformer import Transformer  # type: ignore

    return data_utils, data_utils_ches25, Transformer


def labels_from_dataset(dataset) -> np.ndarray:
    labels = np.concatenate([np.asarray(chunk).reshape(-1) for chunk in dataset.labels])
    return labels.astype(np.int64)


def build_model(args, Transformer):
    model = Transformer(
        n_layer=args.n_layer,
        d_model=args.d_model,
        d_head=args.d_head,
        n_head=args.n_head,
        d_inner=args.d_inner,
        n_head_softmax=args.n_head_softmax,
        d_head_softmax=args.d_head_softmax,
        dropout=args.dropout,
        n_classes=256,
        conv_kernel_size=args.conv_kernel_size,
        n_conv_layer=args.n_conv_layer,
        pool_size=args.pool_size,
        d_kernel_map=args.d_kernel_map,
        beta_hat_2=args.beta_hat_2,
        model_normalization=args.model_normalization,
        head_initialization=args.head_initialization,
        softmax_attn=args.softmax_attn,
        output_attn=False,
    )
    dummy = tf.zeros([1, args.input_length], dtype=tf.float32)
    model(dummy, training=False)
    return model


def restore_checkpoint(args, model, checkpoint_index: int) -> str:
    optimizer = tf.keras.optimizers.Adam(learning_rate=args.learning_rate)
    checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)
    options = tf.train.CheckpointOptions(experimental_io_device="/job:localhost")
    if checkpoint_index <= 0:
        checkpoint_path = tf.train.latest_checkpoint(str(args.checkpoint_dir))
    else:
        checkpoint_path = str(args.checkpoint_dir / f"trans_long-{checkpoint_index}")
    if not checkpoint_path:
        raise FileNotFoundError(f"No checkpoint found in {args.checkpoint_dir}")
    if checkpoint_index > 0 and not Path(checkpoint_path + ".index").exists():
        raise FileNotFoundError(f"Checkpoint index file not found: {checkpoint_path}.index")
    checkpoint.restore(checkpoint_path, options=options).expect_partial()
    return checkpoint_path


def predict_success(args, dataset, labels: np.ndarray, model) -> tuple[int, int, float]:
    tfrecords = dataset.GetTFRecords(args.eval_batch_size, training=False)
    output = model.predict(tfrecords, verbose=1)
    logits = output[0] if isinstance(output, (list, tuple)) else output
    preds = np.argmax(logits, axis=1).astype(np.int64)
    labels = labels[: len(preds)]
    successes = int(np.sum(preds == labels))
    n_te = int(len(labels))
    return successes, n_te, successes / float(n_te)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", required=True, type=Path)
    parser.add_argument("--dataset", choices=["ASCAD", "CHES25"], required=True)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--scope-id", required=True)
    parser.add_argument("--data-path", required=True, type=Path)
    parser.add_argument("--checkpoint-dir", required=True, type=Path)
    parser.add_argument("--checkpoint-indices", default="latest")
    parser.add_argument("--skip-missing-checkpoints", action="store_true")
    parser.add_argument("--output-csv", required=True, type=Path)
    parser.add_argument("--source-note", default="")
    parser.add_argument("--input-length", type=int, required=True)
    parser.add_argument("--data-desync", type=int, default=0)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2.5e-4)

    parser.add_argument("--n-layer", type=int, default=2)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--d-head", type=int, default=32)
    parser.add_argument("--n-head", type=int, default=8)
    parser.add_argument("--d-inner", type=int, default=256)
    parser.add_argument("--n-head-softmax", type=int, default=8)
    parser.add_argument("--d-head-softmax", type=int, default=16)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--conv-kernel-size", type=int, default=3)
    parser.add_argument("--n-conv-layer", type=int, default=2)
    parser.add_argument("--pool-size", type=int, required=True)
    parser.add_argument("--d-kernel-map", type=int, default=512)
    parser.add_argument("--beta-hat-2", type=int, required=True)
    parser.add_argument("--model-normalization", default="preLC")
    parser.add_argument("--head-initialization", default="forward")
    parser.add_argument("--softmax-attn", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    data_utils, data_utils_ches25, Transformer = load_modules(args.project_root)

    if args.dataset == "ASCAD":
        dataset = data_utils.Dataset(
            data_path=str(args.data_path),
            split="test",
            input_length=args.input_length,
            data_desync=args.data_desync,
        )
    else:
        dataset = data_utils_ches25.Dataset(
            data_path=str(args.data_path),
            split="test",
            input_length=args.input_length,
            data_desync=args.data_desync,
        )
    labels = labels_from_dataset(dataset)

    if args.checkpoint_indices.strip().lower() == "latest":
        indices = [0]
    elif args.checkpoint_indices.strip().lower() == "all":
        indices = discover_indices(args.checkpoint_dir)
    else:
        indices = parse_checkpoint_indices(args.checkpoint_indices)
    if not indices:
        raise ValueError(f"No checkpoint indices selected for {args.checkpoint_dir}")

    rows = []
    for checkpoint_index in indices:
        tf.keras.backend.clear_session()
        model = build_model(args, Transformer)
        try:
            checkpoint_path = restore_checkpoint(args, model, checkpoint_index)
        except FileNotFoundError as exc:
            if args.skip_missing_checkpoints:
                print(f"SKIP checkpoint {checkpoint_index}: {exc}", flush=True)
                continue
            raise
        successes, n_te, success_rate = predict_success(args, dataset, labels, model)
        model_id = f"{args.dataset_name}_estranet_ckpt{checkpoint_index or 'latest'}"
        print(
            f"{model_id}: successes={successes} n_te={n_te} "
            f"success_rate={success_rate:.8f} checkpoint={checkpoint_path}",
            flush=True,
        )
        rows.append(
            {
                "dataset": args.dataset_name,
                "scope_id": args.scope_id,
                "seed": 0,
                "model_id": model_id,
                "checkpoint_index": checkpoint_index,
                "checkpoint_path": checkpoint_path,
                "n_te": n_te,
                "successes": successes,
                "success_rate": success_rate,
                "source_artifact": args.source_note or checkpoint_path,
            }
        )

    if not rows:
        raise RuntimeError(f"No checkpoints were evaluated for {args.dataset_name}")

    with args.output_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary_path = args.output_csv.with_suffix(".summary.json")
    summary_path.write_text(
        json.dumps(
            {
                "dataset": args.dataset_name,
                "scope_id": args.scope_id,
                "M": len(rows),
                "n_te": rows[0]["n_te"],
                "best_model": max(rows, key=lambda r: r["successes"])["model_id"],
                "best_success_rate": max(float(r["success_rate"]) for r in rows),
                "output_csv": str(args.output_csv),
            },
            indent=2,
        )
        + "\n"
    )
    print(f"Wrote {args.output_csv}")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
