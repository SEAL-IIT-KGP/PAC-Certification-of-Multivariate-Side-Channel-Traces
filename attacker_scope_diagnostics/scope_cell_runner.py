#!/usr/bin/env python3
"""
Targeted runner for attacker-scope diagnostic cells.

This script wraps `attacker_scope_diagnostics/attacker_scope_diagnostic.py` without changing the existing
checkpoint schema. It runs exactly one (dataset, seed, scope) cell per task,
appends under a file lock, and skips rows that are already present.

Two runtime fixes are applied for the aes_rd scope-diagnostic cells:
  1. high-dimensional datasets can be reduced with deterministic unsupervised
     variance feature selection so sklearn baselines checkpoint within an 8h
     allocation;
  2. the PyTorch DataLoader path keeps tensors on CPU and moves mini-batches to
     CUDA inside the training loop, avoiding CUDA tensor multiprocessing issues.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Iterable, Set, Tuple

import numpy as np
import pandas as pd

from attacker_scope_diagnostics import attacker_scope_diagnostic as scope_module


Key = Tuple[str, int, int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--scope", type=int, required=True, choices=[1, 2, 3, 4])
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--tches20-src-dir", default="")
    parser.add_argument("--output-dir", default="results/attacker_scope_diagnostic")
    parser.add_argument("--checkpoint", default="results/attacker_scope_diagnostic/scope_checkpoint.csv")
    parser.add_argument("--n-seeds", type=int, default=5)
    parser.add_argument("--holdout-size", type=int, default=25000)
    parser.add_argument("--delta", type=float, default=1e-6)
    parser.add_argument("--n-classes", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--target-cnn-params", type=int, default=50000)
    parser.add_argument("--cnn-epochs", type=int, default=12)
    parser.add_argument("--cnn-patience", type=int, default=3)
    parser.add_argument("--cnn-lr", type=float, default=1e-3)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-features", type=int, default=700)
    parser.add_argument("--feature-select", choices=["variance", "stride"], default="variance")
    return parser.parse_args()


def checkpoint_keys(checkpoint: Path) -> Set[Key]:
    if not checkpoint.exists() or checkpoint.stat().st_size == 0:
        return set()
    try:
        df = pd.read_csv(checkpoint)
    except pd.errors.EmptyDataError:
        return set()
    required = {"dataset", "scope_level", "seed"}
    if not required.issubset(df.columns):
        return set()
    return {
        (str(row.dataset), int(row.scope_level), int(row.seed))
        for row in df.itertuples(index=False)
    }


def append_checkpoint(result: scope_module.ExperimentResult, checkpoint: Path) -> None:
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    lock_path = checkpoint.with_suffix(checkpoint.suffix + ".lock")
    row_key = (result.dataset, int(result.scope_level), int(result.seed))
    with open(lock_path, "w") as lock_f:
        fcntl.flock(lock_f, fcntl.LOCK_EX)
        if row_key in checkpoint_keys(checkpoint):
            print(f"[checkpoint] {row_key} already present; not appending duplicate", flush=True)
            return
        row = pd.DataFrame([asdict(result)])
        write_header = not checkpoint.exists() or checkpoint.stat().st_size == 0
        row.to_csv(checkpoint, mode="a", index=False, header=write_header)
        print(f"[checkpoint] appended {row_key} to {checkpoint}", flush=True)


def select_features(X: np.ndarray, max_features: int, method: str) -> np.ndarray:
    if max_features <= 0 or X.ndim != 2 or X.shape[1] <= max_features:
        return X
    if method == "stride":
        idx = np.linspace(0, X.shape[1] - 1, max_features, dtype=int)
    else:
        scores = np.nan_to_num(np.var(X, axis=0), nan=0.0, posinf=0.0, neginf=0.0)
        idx = np.argsort(scores)[-max_features:]
        idx.sort()
    return X[:, idx]


def patch_loader(max_features: int, feature_select: str) -> None:
    original_load_dataset = scope_module.load_dataset

    def load_dataset_capped(dataset: str, data_dir: str, tches20_src_dir: str = ""):
        X, y = original_load_dataset(dataset, data_dir=data_dir, tches20_src_dir=tches20_src_dir)
        X = np.asarray(X)
        y = np.asarray(y)
        n_features_before = X.shape[1] if X.ndim == 2 else 0
        X = select_features(X, max_features=max_features, method=feature_select)
        n_features_after = X.shape[1] if X.ndim == 2 else 0
        if n_features_after != n_features_before:
            print(
                f"[feature-select] {dataset}: {n_features_before} -> {n_features_after} "
                f"features via {feature_select}",
                flush=True,
            )
        return X, y

    scope_module.load_dataset = load_dataset_capped


def patch_torch_training() -> None:
    if not scope_module.HAS_TORCH:
        return

    torch = scope_module.torch
    nn = scope_module.nn
    optim = scope_module.optim
    TensorDataset = scope_module.TensorDataset
    DataLoader = scope_module.DataLoader

    def train_torch_model_fixed(
        model,
        X_train,
        y_train,
        epochs=30,
        batch_size=256,
        lr=1e-3,
        patience=5,
        device="cpu",
    ):
        if str(device).startswith("cuda") and not torch.cuda.is_available():
            device = "cpu"
        use_cuda = str(device).startswith("cuda")
        model = model.to(device)

        n_train = len(X_train)
        n_val = max(1, int(0.15 * n_train))
        n_train_actual = n_train - n_val
        idx = np.random.permutation(n_train)
        idx_tr = idx[:n_train_actual]
        idx_val = idx[n_train_actual:]

        X_tr_t = torch.as_tensor(X_train[idx_tr], dtype=torch.float32)
        y_tr_t = torch.as_tensor(y_train[idx_tr], dtype=torch.long)
        X_v_t = torch.as_tensor(X_train[idx_val], dtype=torch.float32)
        y_v_t = torch.as_tensor(y_train[idx_val], dtype=torch.long)

        train_ds = TensorDataset(X_tr_t, y_tr_t)
        train_dl = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            pin_memory=use_cuda,
            num_workers=0,
        )
        val_ds = TensorDataset(X_v_t, y_v_t)
        val_dl = DataLoader(
            val_ds,
            batch_size=max(batch_size, 512),
            shuffle=False,
            pin_memory=use_cuda,
            num_workers=0,
        )

        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        best_val_loss = float("inf")
        no_improve = 0

        for _epoch in range(epochs):
            model.train()
            for xb, yb in train_dl:
                xb = xb.to(device, non_blocking=use_cuda)
                yb = yb.to(device, non_blocking=use_cuda)
                optimizer.zero_grad(set_to_none=True)
                loss = criterion(model(xb), yb)
                loss.backward()
                optimizer.step()

            model.eval()
            val_loss = 0.0
            val_count = 0
            with torch.no_grad():
                for xb, yb in val_dl:
                    xb = xb.to(device, non_blocking=use_cuda)
                    yb = yb.to(device, non_blocking=use_cuda)
                    batch_loss = criterion(model(xb), yb).item()
                    val_loss += batch_loss * len(yb)
                    val_count += len(yb)
            val_loss /= max(1, val_count)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                no_improve = 0
            else:
                no_improve += 1
            if no_improve >= patience:
                break

        model.eval()
        return model

    def predict_torch_fixed(model, X, device="cpu"):
        if str(device).startswith("cuda") and not torch.cuda.is_available():
            device = "cpu"
        use_cuda = str(device).startswith("cuda")
        model.eval()
        ds = TensorDataset(torch.as_tensor(X, dtype=torch.float32))
        dl = DataLoader(ds, batch_size=1024, shuffle=False, pin_memory=use_cuda, num_workers=0)
        preds = []
        with torch.no_grad():
            for (xb,) in dl:
                xb = xb.to(device, non_blocking=use_cuda)
                preds.append(model(xb).argmax(dim=1).cpu().numpy())
        return np.concatenate(preds) if preds else np.array([], dtype=np.int64)

    scope_module.train_torch_model = train_torch_model_fixed
    scope_module.predict_torch = predict_torch_fixed


def non_target_keys(dataset: str, target_seed: int, target_scope: int, n_seeds: int) -> Iterable[Key]:
    for seed in range(n_seeds):
        if seed != target_seed:
            yield (dataset, target_scope, seed)


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    checkpoint = Path(args.checkpoint)
    output_dir.mkdir(parents=True, exist_ok=True)

    patch_loader(args.max_features, args.feature_select)
    patch_torch_training()

    target = (args.dataset, args.scope, args.seed)
    completed = checkpoint_keys(checkpoint)
    if target in completed:
        print(f"[skip] {target} already complete in {checkpoint}", flush=True)
        return 0

    completed.update(non_target_keys(args.dataset, args.seed, args.scope, args.n_seeds))

    config = {
        "target": {"dataset": args.dataset, "seed": args.seed, "scope": args.scope},
        "max_features": args.max_features,
        "feature_select": args.feature_select,
        "cnn_epochs": args.cnn_epochs,
        "cnn_patience": args.cnn_patience,
        "target_cnn_params": args.target_cnn_params,
        "device": args.device,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "pid": os.getpid(),
    }
    with open(output_dir / "scope_run_config.jsonl", "a") as f:
        f.write(json.dumps(config, sort_keys=True) + "\n")
    print("[config] " + json.dumps(config, sort_keys=True), flush=True)

    scope_module.run_experiment(
        dataset=args.dataset,
        dataset_dir=args.dataset_dir,
        tches20_src_dir=args.tches20_src_dir,
        scopes=[args.scope],
        n_seeds=args.n_seeds,
        holdout_size=args.holdout_size,
        delta=args.delta,
        n_classes=args.n_classes,
        batch_size=args.batch_size,
        target_cnn_params=args.target_cnn_params,
        cnn_epochs=args.cnn_epochs,
        cnn_patience=args.cnn_patience,
        cnn_lr=args.cnn_lr,
        device=args.device,
        gpu_workers=1,
        completed_keys=completed,
        scope_checkpoint_callback=lambda result: append_checkpoint(result, checkpoint),
    )

    if target not in checkpoint_keys(checkpoint):
        print(f"[error] target {target} did not produce a checkpoint row", file=sys.stderr, flush=True)
        return 2
    print(f"[done] target {target} complete", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
