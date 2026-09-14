#!/usr/bin/env python3
"""
Budgeted seed-level runner for AES-RD attacker-scope diagnostics.

The original scope-level resume path retrains lower scopes independently for
scope 3 and scope 4, which can be slow on aes_rd. This
runner processes one seed per job, trains a compact nested suite once, and
checkpoints every missing scope row derived from that seed.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import math
import os
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from attacker_scope_diagnostics import attacker_scope_diagnostic as scope_module


Key = Tuple[str, int, int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="aes_rd")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--tches20-src-dir", default="")
    parser.add_argument("--output-dir", default="results/attacker_scope_diagnostic")
    parser.add_argument("--checkpoint", default="results/attacker_scope_diagnostic/aesrd_budget_checkpoint.csv")
    parser.add_argument("--holdout-size", type=int, default=25000)
    parser.add_argument("--delta", type=float, default=1e-6)
    parser.add_argument("--n-classes", type=int, default=256)
    parser.add_argument("--max-features", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--cnn-epochs", type=int, default=4)
    parser.add_argument("--cnn-patience", type=int, default=2)
    parser.add_argument("--cnn-lr", type=float, default=1e-3)
    parser.add_argument("--cnn-channels", type=int, default=16)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def checkpoint_keys(checkpoint: Path) -> set[Key]:
    if not checkpoint.exists() or checkpoint.stat().st_size == 0:
        return set()
    try:
        df = pd.read_csv(checkpoint)
    except pd.errors.EmptyDataError:
        return set()
    if not {"dataset", "scope_level", "seed"}.issubset(df.columns):
        return set()
    return {
        (str(row.dataset), int(row.scope_level), int(row.seed))
        for row in df.itertuples(index=False)
    }


def append_checkpoint(result: scope_module.ExperimentResult, checkpoint: Path) -> None:
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    lock_path = checkpoint.with_suffix(checkpoint.suffix + ".lock")
    key = (result.dataset, int(result.scope_level), int(result.seed))
    with open(lock_path, "w") as lock_f:
        fcntl.flock(lock_f, fcntl.LOCK_EX)
        if key in checkpoint_keys(checkpoint):
            print(f"[checkpoint] {key} already present; skip duplicate", flush=True)
            return
        row = pd.DataFrame([asdict(result)])
        write_header = not checkpoint.exists() or checkpoint.stat().st_size == 0
        row.to_csv(checkpoint, mode="a", index=False, header=write_header)
        print(f"[checkpoint] appended {key}", flush=True)


def select_features(X: np.ndarray, max_features: int) -> np.ndarray:
    if X.ndim != 2 or X.shape[1] <= max_features:
        return X
    scores = np.nan_to_num(np.var(X, axis=0), nan=0.0, posinf=0.0, neginf=0.0)
    idx = np.argsort(scores)[-max_features:]
    idx.sort()
    return X[:, idx]


def split_seed(X: np.ndarray, y: np.ndarray, seed: int, holdout_size: int):
    rng = np.random.RandomState(seed)
    n_total = len(X)
    train_size = int(0.6 * n_total)
    val_size = int(0.2 * n_total)
    holdout_size_actual = min(holdout_size, n_total - train_size - val_size)
    idx = rng.permutation(n_total)
    idx_train = idx[:train_size]
    idx_val = idx[train_size:train_size + val_size]
    idx_holdout = idx[train_size + val_size:train_size + val_size + holdout_size_actual]
    X_train_full = np.vstack([X[idx_train], X[idx_val]])
    y_train_full = np.hstack([y[idx_train], y[idx_val]])
    return X_train_full, y_train_full, X[idx_holdout], y[idx_holdout]


def build_budgeted_suites(n_features: int, n_classes: int, cnn_channels: int):
    linear = [
        ("LDA", LinearDiscriminantAnalysis()),
        ("LogReg_C1.0_budget", LogisticRegression(solver="lbfgs", max_iter=80, C=1.0, n_jobs=-1)),
        ("GNB", GaussianNB()),
        ("Ridge", RidgeClassifier(alpha=1.0)),
    ]
    mlp = [
        ("MLP_64_budget", MLPClassifier(hidden_layer_sizes=(64,), max_iter=80, early_stopping=True, validation_fraction=0.1, n_iter_no_change=5, batch_size=512, random_state=42)),
        ("MLP_128_64_budget", MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=80, early_stopping=True, validation_fraction=0.1, n_iter_no_change=5, batch_size=512, random_state=43)),
    ]
    cnn_local = []
    cnn_wide = []
    if scope_module.HAS_TORCH and n_features >= 3:
        for rf in [9, 17, 33]:
            cnn_local.append((f"CNN_RL{rf}_ch{cnn_channels}_budget", scope_module.ControlledReceptiveFieldCNN(n_features, n_classes, rf, cnn_channels)))
        for rf in [65, 129]:
            if rf <= n_features:
                cnn_wide.append((f"CNN_RL{rf}_ch{cnn_channels}_budget", scope_module.ControlledReceptiveFieldCNN(n_features, n_classes, rf, cnn_channels)))
    return {1: linear, 2: mlp, 3: cnn_local, 4: cnn_wide}


def train_cnn(model, X_train, y_train, X_holdout, y_holdout, args) -> float:
    torch = scope_module.torch
    nn = scope_module.nn
    optim = scope_module.optim
    TensorDataset = scope_module.TensorDataset
    DataLoader = scope_module.DataLoader

    device = args.device if str(args.device).startswith("cuda") and torch.cuda.is_available() else "cpu"
    use_cuda = str(device).startswith("cuda")
    model = model.to(device)

    n_train = len(X_train)
    n_val = max(1, int(0.1 * n_train))
    idx = np.random.permutation(n_train)
    tr_idx = idx[:-n_val]
    val_idx = idx[-n_val:]

    train_ds = TensorDataset(
        torch.as_tensor(X_train[tr_idx], dtype=torch.float32),
        torch.as_tensor(y_train[tr_idx], dtype=torch.long),
    )
    val_x = torch.as_tensor(X_train[val_idx], dtype=torch.float32)
    val_y = torch.as_tensor(y_train[val_idx], dtype=torch.long)

    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, pin_memory=use_cuda, num_workers=0)
    optimizer = optim.Adam(model.parameters(), lr=args.cnn_lr)
    criterion = nn.CrossEntropyLoss()
    best = float("inf")
    stale = 0
    for _epoch in range(args.cnn_epochs):
        model.train()
        for xb, yb in train_dl:
            xb = xb.to(device, non_blocking=use_cuda)
            yb = yb.to(device, non_blocking=use_cuda)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()

        model.eval()
        losses = []
        with torch.no_grad():
            for start in range(0, len(val_x), max(args.batch_size, 1024)):
                xb = val_x[start:start + max(args.batch_size, 1024)].to(device, non_blocking=use_cuda)
                yb = val_y[start:start + max(args.batch_size, 1024)].to(device, non_blocking=use_cuda)
                losses.append(criterion(model(xb), yb).item())
        val_loss = float(np.mean(losses)) if losses else best
        if val_loss < best:
            best = val_loss
            stale = 0
        else:
            stale += 1
        if stale >= args.cnn_patience:
            break

    preds = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(X_holdout), 1024):
            xb = torch.as_tensor(X_holdout[start:start + 1024], dtype=torch.float32).to(device, non_blocking=use_cuda)
            preds.append(model(xb).argmax(dim=1).cpu().numpy())
    pred = np.concatenate(preds) if preds else np.array([], dtype=np.int64)
    return float(np.mean(pred == y_holdout))


def train_sklearn_model(model, X_train_scaled, y_train, X_holdout_scaled, y_holdout) -> float:
    model.fit(X_train_scaled, y_train)
    if hasattr(model, "predict_proba"):
        pred = np.argmax(model.predict_proba(X_holdout_scaled), axis=1)
    else:
        pred = model.predict(X_holdout_scaled)
    return float(np.mean(pred == y_holdout))


def make_result(dataset: str, seed: int, scope: int, successes: Dict[str, float], n_train: int, n_holdout: int, args) -> scope_module.ExperimentResult:
    M = len(successes)
    p_max = max(successes.values()) if successes else 0.0
    best_model = max(successes, key=successes.get) if successes else "none"
    margin = math.sqrt(math.log(max(M, 1) / args.delta) / (2 * n_holdout))
    certified_success = min(1.0, p_max + margin)
    eps_star = max(0.0, certified_success - 1.0 / args.n_classes)
    bi = min(math.log2(1 + args.n_classes * eps_star), math.log2(args.n_classes))
    return scope_module.ExperimentResult(
        dataset=dataset,
        scope_level=scope,
        scope_name=scope_module.SCOPE_NAMES[scope],
        seed=seed,
        M=M,
        n_train=n_train,
        n_holdout=n_holdout,
        bi=bi,
        bi_margin=eps_star,
        p_max=p_max,
        margin_term=margin,
        best_model=best_model,
        per_model_success=json.dumps(successes, sort_keys=True),
        pi=np.nan,
        ti=np.nan,
        hi=np.nan,
        mlp_pi=np.nan,
        mlp_ti=np.nan,
    )


def main() -> int:
    args = parse_args()
    checkpoint = Path(args.checkpoint)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    print(f"[config] {json.dumps(vars(args), sort_keys=True)}", flush=True)

    completed = checkpoint_keys(checkpoint)
    missing_scopes = [scope for scope in [1, 2, 3, 4] if (args.dataset, scope, args.seed) not in completed]
    if not missing_scopes:
        print(f"[done] all scopes already complete for {args.dataset} seed={args.seed}", flush=True)
        return 0

    X, y = scope_module.load_dataset(args.dataset, data_dir=args.dataset_dir, tches20_src_dir=args.tches20_src_dir)
    before = X.shape[1]
    X = select_features(np.asarray(X), args.max_features)
    y = np.asarray(y)
    print(f"[feature-select] {args.dataset}: {before} -> {X.shape[1]}", flush=True)

    X_train, y_train, X_holdout, y_holdout = split_seed(X, y, args.seed, args.holdout_size)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_holdout_scaled = scaler.transform(X_holdout)
    suites = build_budgeted_suites(X.shape[1], args.n_classes, args.cnn_channels)

    successes: Dict[str, float] = {}
    for scope in [1, 2, 3, 4]:
        t0 = time.time()
        print(f"[scope-start] seed={args.seed} scope={scope} missing={scope in missing_scopes}", flush=True)
        for name, model in suites[scope]:
            mt0 = time.time()
            try:
                if scope_module.HAS_TORCH and isinstance(model, scope_module.nn.Module):
                    success = train_cnn(model, X_train, y_train, X_holdout, y_holdout, args)
                else:
                    success = train_sklearn_model(model, X_train_scaled, y_train, X_holdout_scaled, y_holdout)
                successes[name] = success
                print(f"[model] {name} success={success:.5f} seconds={time.time()-mt0:.1f}", flush=True)
            except Exception as exc:
                print(f"[model-failed] {name}: {exc}", flush=True)

        if scope in missing_scopes:
            result = make_result(args.dataset, args.seed, scope, dict(successes), len(X_train), len(X_holdout), args)
            append_checkpoint(result, checkpoint)
            print(f"[scope-done] seed={args.seed} scope={scope} BI={result.bi:.4f} M={result.M} seconds={time.time()-t0:.1f}", flush=True)
        else:
            print(f"[scope-skip-append] seed={args.seed} scope={scope} seconds={time.time()-t0:.1f}", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
