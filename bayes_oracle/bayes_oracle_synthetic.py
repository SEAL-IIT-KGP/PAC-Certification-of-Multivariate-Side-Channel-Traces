#!/usr/bin/env python3
"""Oracle-aware synthetic known-posterior diagnostic.

The experiment is intentionally synthetic because the unrestricted Bayes
attacker is unknown on real side-channel traces. Here the generative law is
known, so the script computes the Bayes posterior, Bayes guessing probability,
mutual information, posterior margins, span-specific expressivity gaps, and
finite trained-model gaps.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import math
import os
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", tempfile.mkdtemp(prefix="bayes_oracle_mpl_"))
os.environ.setdefault("XDG_CACHE_HOME", tempfile.mkdtemp(prefix="bayes_oracle_cache_"))

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
except Exception:  # pragma: no cover - handled at runtime
    LogisticRegression = None
    StandardScaler = None

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
except Exception:  # pragma: no cover - handled at runtime
    torch = None
    nn = None
    optim = None
    DataLoader = None
    TensorDataset = None


K = 2
CHANCE = 0.5


@dataclass(frozen=True)
class DataSplit:
    x_train: np.ndarray
    y_train: np.ndarray
    x_val: np.ndarray
    y_val: np.ndarray
    x_hold: np.ndarray
    y_hold: np.ndarray
    llr_hold: np.ndarray
    starts: tuple[int, ...]


class RawPatchMLP(nn.Module):
    def __init__(self, input_dim: int, width: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, width),
            nn.ReLU(),
            nn.Linear(width, width),
            nn.ReLU(),
            nn.Linear(width, K),
        )

    def forward(self, x):
        return self.net(x)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def parse_int_list(value: str) -> list[int]:
    out = [int(v.strip()) for v in str(value).split(",") if v.strip()]
    if not out:
        raise argparse.ArgumentTypeError("expected at least one integer")
    return out


def parse_float_list(value: str) -> list[float]:
    out = [float(v.strip()) for v in str(value).split(",") if v.strip()]
    if not out:
        raise argparse.ArgumentTypeError("expected at least one float")
    return out


def sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    out = np.empty_like(x, dtype=np.float64)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    ex = np.exp(x[~pos])
    out[~pos] = ex / (1.0 + ex)
    return out


def binary_entropy_bits(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-15, 1.0 - 1e-15)
    return -(p * np.log2(p) + (1.0 - p) * np.log2(1.0 - p))


def bi_from_success(p_success: float) -> float:
    adv = max(0.0, float(p_success) - CHANCE)
    return min(1.0, math.log2(1.0 + K * adv))


def kl_bernoulli(q: float, p: float) -> float:
    eps = 1e-15
    q = min(1.0 - eps, max(eps, float(q)))
    p = min(1.0 - eps, max(eps, float(p)))
    return q * math.log(q / p) + (1.0 - q) * math.log((1.0 - q) / (1.0 - p))


def kl_upper(successes: int, n: int, alpha: float) -> float:
    q = successes / n
    if successes >= n:
        return 1.0
    target = math.log(1.0 / alpha) / n
    lo, hi = q, 1.0 - 1e-15
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        if kl_bernoulli(q, mid) > target:
            hi = mid
        else:
            lo = mid
    return hi


def true_span(separation: int) -> int:
    return int(separation) + 1


def planted_starts(trace_length: int, separation: int, n_repetitions: int) -> np.ndarray:
    span = true_span(separation)
    if n_repetitions < 1:
        raise ValueError("n_repetitions must be positive")
    required = n_repetitions * span
    if trace_length < required:
        raise ValueError(
            f"trace_length={trace_length} is too short for {n_repetitions} "
            f"pairs of span {span}; need at least {required}"
        )
    if n_repetitions == 1:
        return np.array([0], dtype=np.int64)
    return np.arange(n_repetitions, dtype=np.int64) * span


def bayes_llr_coefficient(signal_amplitude: float, noise_std: float) -> float:
    a2 = float(signal_amplitude) ** 2
    s2 = float(noise_std) ** 2
    return 2.0 * a2 / (s2 * (2.0 * a2 + s2))


def generate_split(
    *,
    seed: int,
    n_samples: int,
    holdout_size: int,
    separation: int,
    n_repetitions: int,
    signal_amplitude: float,
    noise_std: float,
    train_frac: float,
    val_frac: float,
) -> DataSplit:
    span = true_span(separation)
    trace_length = span * n_repetitions
    starts = planted_starts(trace_length, separation, n_repetitions)
    rng = np.random.default_rng(seed)

    y = rng.integers(0, 2, size=n_samples, dtype=np.int64)
    signs = (2 * y - 1).astype(np.float64)
    x = rng.normal(0.0, noise_std, size=(n_samples, trace_length)).astype(np.float64)
    llr = np.zeros(n_samples, dtype=np.float64)
    coeff = bayes_llr_coefficient(signal_amplitude, noise_std)

    for start in starts:
        p1 = int(start)
        p2 = int(start + separation)
        z = rng.normal(0.0, 1.0, size=n_samples)
        share_1 = signal_amplitude * z
        share_2 = signal_amplitude * signs * z
        x[:, p1] += share_1
        x[:, p2] += share_2
        llr += coeff * x[:, p1] * x[:, p2]

    n_hold = min(int(holdout_size), n_samples - 2)
    n_remaining = n_samples - n_hold
    n_train = max(1, int(train_frac * n_remaining))
    n_val = max(1, int(val_frac * n_remaining))
    if n_train + n_val > n_remaining:
        n_val = n_remaining - n_train
    if n_val <= 0:
        raise ValueError("not enough samples for train/val/holdout split")

    idx = rng.permutation(n_samples)
    idx_train = idx[:n_train]
    idx_val = idx[n_train:n_train + n_val]
    idx_hold = idx[n_train + n_val:n_train + n_val + n_hold]

    mean = x[idx_train].mean(axis=0, keepdims=True)
    std = x[idx_train].std(axis=0, keepdims=True)
    std[std < 1e-8] = 1.0
    x_norm = ((x - mean) / std).astype(np.float32)

    return DataSplit(
        x_train=x_norm[idx_train],
        y_train=y[idx_train],
        x_val=x_norm[idx_val],
        y_val=y[idx_val],
        x_hold=x_norm[idx_hold],
        y_hold=y[idx_hold],
        llr_hold=llr[idx_hold],
        starts=tuple(int(s) for s in starts.tolist()),
    )


def oracle_metrics(y_hold: np.ndarray, llr_hold: np.ndarray) -> dict[str, float]:
    posterior_one = sigmoid(llr_hold)
    pred = (posterior_one >= 0.5).astype(np.int64)
    p_guess = float(np.mean(np.maximum(posterior_one, 1.0 - posterior_one)))
    map_success = float(np.mean(pred == y_hold))
    mi_bits = float(1.0 - np.mean(binary_entropy_bits(posterior_one)))
    score_margin = np.abs(llr_hold)
    posterior_margin = np.abs(2.0 * posterior_one - 1.0)
    return {
        "oracle_p_guess": p_guess,
        "oracle_map_success": map_success,
        "oracle_mi_bits": mi_bits,
        "oracle_score_margin_mean": float(np.mean(score_margin)),
        "oracle_score_margin_q01": float(np.quantile(score_margin, 0.01)),
        "oracle_score_margin_q05": float(np.quantile(score_margin, 0.05)),
        "oracle_score_margin_median": float(np.median(score_margin)),
        "oracle_posterior_margin_mean": float(np.mean(posterior_margin)),
        "oracle_posterior_margin_q01": float(np.quantile(posterior_margin, 0.01)),
        "oracle_posterior_margin_q05": float(np.quantile(posterior_margin, 0.05)),
    }


def product_lag_features(x: np.ndarray, receptive_field: int) -> np.ndarray:
    max_lag = min(max(0, int(receptive_field) - 1), x.shape[1] - 1)
    if max_lag <= 0:
        return np.zeros((len(x), 1), dtype=np.float32)
    feats = np.empty((len(x), max_lag), dtype=np.float32)
    for lag in range(1, max_lag + 1):
        feats[:, lag - 1] = np.mean(x[:, :-lag] * x[:, lag:], axis=1)
    return feats


def logistic_product_lag_success(
    split: DataSplit,
    receptive_field: int,
    seed: int,
) -> tuple[float, float, int, str]:
    x_train = product_lag_features(split.x_train, receptive_field)
    x_val = product_lag_features(split.x_val, receptive_field)
    x_hold = product_lag_features(split.x_hold, receptive_field)

    if LogisticRegression is None or StandardScaler is None:
        lag = min(max(0, int(receptive_field) - 2), max(0, x_hold.shape[1] - 1))
        if x_hold.shape[1] <= 1:
            pred_hold = np.full(len(split.y_hold), int(np.mean(split.y_train) >= 0.5), dtype=np.int64)
            pred_val = np.full(len(split.y_val), int(np.mean(split.y_train) >= 0.5), dtype=np.int64)
            return float(np.mean(pred_val == split.y_val)), float(np.mean(pred_hold == split.y_hold)), 0, "majority_fallback"
        score_val = x_val[:, min(lag, x_val.shape[1] - 1)]
        score_hold = x_hold[:, min(lag, x_hold.shape[1] - 1)]
        corr = np.corrcoef(score_val, split.y_val)[0, 1]
        sign = 1.0 if np.nan_to_num(corr) >= 0 else -1.0
        pred_val = (sign * score_val >= 0.0).astype(np.int64)
        pred_hold = (sign * score_hold >= 0.0).astype(np.int64)
        return float(np.mean(pred_val == split.y_val)), float(np.mean(pred_hold == split.y_hold)), x_hold.shape[1], "numpy_threshold_fallback"

    scaler = StandardScaler()
    x_train_s = scaler.fit_transform(x_train)
    x_val_s = scaler.transform(x_val)
    x_hold_s = scaler.transform(x_hold)
    model = LogisticRegression(
        solver="lbfgs",
        max_iter=300,
        C=10.0,
        random_state=seed,
    )
    model.fit(x_train_s, split.y_train)
    val_success = float(np.mean(model.predict(x_val_s) == split.y_val))
    hold_success = float(np.mean(model.predict(x_hold_s) == split.y_hold))
    return val_success, hold_success, int(x_hold.shape[1]), "logistic_product_lag"


def make_loader(x: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool, device, seed: int):
    x_t = torch.as_tensor(x, dtype=torch.float32)
    y_t = torch.as_tensor(y, dtype=torch.long)
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        TensorDataset(x_t, y_t),
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=(device.type == "cuda"),
        generator=generator if shuffle else None,
    )


def evaluate_torch(model, x: np.ndarray, y: np.ndarray, batch_size: int, device) -> tuple[float, np.ndarray]:
    model.eval()
    preds = []
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            xb = torch.as_tensor(x[start:start + batch_size], dtype=torch.float32).to(device, non_blocking=True)
            logits = model(xb)
            preds.append(logits.argmax(dim=1).cpu().numpy())
    pred = np.concatenate(preds) if preds else np.array([], dtype=np.int64)
    return float(np.mean(pred == y)), pred


def train_raw_mlp(
    split: DataSplit,
    *,
    width: int,
    restart: int,
    seed: int,
    batch_size: int,
    epochs: int,
    patience: int,
    lr: float,
    device_name: str,
) -> dict[str, object]:
    if torch is None:
        raise RuntimeError("torch is unavailable; use --skip-mlp")

    torch.manual_seed(seed + restart * 10_007 + width * 101)
    np.random.seed(seed + restart * 10_007 + width * 101)
    device = torch.device("cuda" if device_name.startswith("cuda") and torch.cuda.is_available() else "cpu")
    model = RawPatchMLP(split.x_train.shape[1], width).to(device)
    train_loader = make_loader(split.x_train, split.y_train, batch_size, True, device, seed + restart)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    best_state = None
    best_val = -1.0
    best_epoch = 0
    stale = 0
    for epoch in range(1, epochs + 1):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()

        val_success, _ = evaluate_torch(model, split.x_val, split.y_val, batch_size, device)
        if val_success > best_val:
            best_val = val_success
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    hold_success, _ = evaluate_torch(model, split.x_hold, split.y_hold, batch_size, device)
    return {
        "model_kind": "raw_patch_mlp",
        "width": int(width),
        "restart": int(restart),
        "best_epoch": int(best_epoch),
        "val_success": float(best_val),
        "holdout_success": float(hold_success),
        "n_params": int(model.count_parameters()),
        "device_used": str(device),
    }


def finite_cert_row(success: float, n: int, delta: float) -> dict[str, float]:
    successes = int(round(success * n))
    p_obs = successes / n
    p_plus = kl_upper(successes, n, delta)
    return {
        "success_count": successes,
        "p_obs": p_obs,
        "p_plus_kl": p_plus,
        "BI_obs": bi_from_success(p_obs),
        "BI_plus_kl": bi_from_success(p_plus),
        "kl_slack": p_plus - p_obs,
    }


def task_ids(args: argparse.Namespace) -> list[tuple[str, int, Optional[int], Optional[int]]]:
    seeds = list(range(args.n_seeds))
    tasks: list[tuple[str, int, Optional[int], Optional[int]]] = []
    for seed in seeds:
        tasks.append(("oracle", seed, None, None))
        for rf in args.receptive_fields:
            tasks.append(("span", seed, rf, None))
        if not args.skip_mlp:
            for width in args.widths:
                for restart in range(args.restarts):
                    tasks.append(("width", seed, width, restart))
    return [
        task for idx, task in enumerate(tasks)
        if idx % args.num_shards == args.shard_index
    ]


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def append_row(path: Path, row: dict[str, object], key_cols: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_suffix(path.suffix + ".lock")
    with open(lock_path, "a+") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        try:
            df = read_csv(path)
            df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
            df = df.drop_duplicates(subset=key_cols, keep="last")
            df.to_csv(path, index=False)
        finally:
            fcntl.flock(lock_file, fcntl.LOCK_UN)


def write_margin_rows(
    path: Path,
    *,
    seed: int,
    split: DataSplit,
    oracle: dict[str, float],
    args: argparse.Namespace,
) -> None:
    posterior = sigmoid(split.llr_hold)
    score_margin = np.abs(split.llr_hold)
    posterior_margin = np.abs(2.0 * posterior - 1.0)
    rows = []
    for gamma in args.margin_gammas:
        rows.append(
            {
                "seed": seed,
                "gamma": float(gamma),
                "n_holdout": int(len(split.y_hold)),
                "score_margin_le_gamma": float(np.mean(score_margin <= gamma)),
                "posterior_margin_le_gamma": float(np.mean(posterior_margin <= min(gamma, 1.0))),
                "oracle_p_guess": oracle["oracle_p_guess"],
                "oracle_mi_bits": oracle["oracle_mi_bits"],
            }
        )
    for row in rows:
        append_row(path, row, ["seed", "gamma"])


def summarize_results(out_dir: Path) -> None:
    rows_path = out_dir / "bayes_oracle_rows.csv"
    margin_path = out_dir / "bayes_margin_curve.csv"
    if not rows_path.exists():
        return
    rows = pd.read_csv(rows_path)
    metric_cols = [
        "oracle_p_guess",
        "oracle_map_success",
        "oracle_mi_bits",
        "family_oracle_p_guess",
        "family_expressivity_gap",
        "holdout_success",
        "gap_to_oracle_p_guess",
        "p_plus_kl",
        "BI_obs",
        "BI_plus_kl",
    ]
    group_cols = [
        "row_type",
        "model_kind",
        "receptive_field",
        "width",
    ]
    present_metrics = [c for c in metric_cols if c in rows.columns]
    summary = (
        rows.groupby(group_cols, dropna=False)[present_metrics]
        .agg(["mean", "std", "min", "max"])
        .reset_index()
    )
    summary.columns = [
        "_".join(str(x) for x in col).rstrip("_") if isinstance(col, tuple) else col
        for col in summary.columns
    ]
    summary.to_csv(out_dir / "bayes_oracle_summary.csv", index=False)

    if margin_path.exists():
        margins = pd.read_csv(margin_path)
        margin_summary = (
            margins.groupby("gamma", dropna=False)[
                ["score_margin_le_gamma", "posterior_margin_le_gamma"]
            ]
            .agg(["mean", "std"])
            .reset_index()
        )
        margin_summary.columns = [
            "_".join(str(x) for x in col).rstrip("_") if isinstance(col, tuple) else col
            for col in margin_summary.columns
        ]
        margin_summary.to_csv(out_dir / "bayes_margin_summary.csv", index=False)

    write_plots(out_dir, rows)
    write_readme(out_dir, rows)


def write_plots(out_dir: Path, rows: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt

    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    span = rows[rows["row_type"] == "span_sweep"].copy()
    if not span.empty:
        grouped = span.groupby("receptive_field")
        rf = np.array(sorted(grouped.groups.keys()), dtype=float)
        mean_success = grouped["holdout_success"].mean().reindex(rf).to_numpy()
        std_success = grouped["holdout_success"].std().fillna(0.0).reindex(rf).to_numpy()
        mean_family = grouped["family_oracle_p_guess"].mean().reindex(rf).to_numpy()
        mean_oracle = grouped["oracle_p_guess"].mean().reindex(rf).to_numpy()
        true_span_val = int(span["true_span"].iloc[0])

        fig, ax = plt.subplots(figsize=(6.2, 3.6))
        ax.plot(rf, mean_oracle, color="black", linestyle="--", label="Bayes oracle")
        ax.plot(rf, mean_family, color="#1f77b4", marker="o", label="oracle in declared span")
        ax.errorbar(rf, mean_success, yerr=std_success, color="#d62728", marker="s", label="trained product-lag")
        ax.axvline(true_span_val, color="0.35", linestyle=":", label=f"true span = {true_span_val}")
        ax.set_xlabel("Declared receptive field / span")
        ax.set_ylabel("Single-trace success")
        ax.set_ylim(0.45, min(1.0, max(0.8, float(np.nanmax(mean_oracle)) + 0.08)))
        ax.legend(frameon=False, fontsize=8)
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(fig_dir / "bayes_oracle_success_vs_span.pdf")
        fig.savefig(fig_dir / "bayes_oracle_success_vs_span.png", dpi=200)
        plt.close(fig)

    width = rows[rows["row_type"] == "width_sweep"].copy()
    if not width.empty:
        best = width.sort_values("val_success").groupby(["seed", "width"], as_index=False).tail(1)
        grouped = best.groupby("width")
        widths = np.array(sorted(grouped.groups.keys()), dtype=float)
        mean_success = grouped["holdout_success"].mean().reindex(widths).to_numpy()
        std_success = grouped["holdout_success"].std().fillna(0.0).reindex(widths).to_numpy()
        oracle = float(best["oracle_p_guess"].mean())

        fig, ax = plt.subplots(figsize=(5.4, 3.4))
        ax.axhline(oracle, color="black", linestyle="--", label="Bayes oracle")
        ax.errorbar(widths, mean_success, yerr=std_success, color="#2ca02c", marker="o", label="best restart")
        ax.set_xscale("log", base=2)
        ax.set_xlabel("MLP width")
        ax.set_ylabel("Single-trace success")
        ax.set_ylim(0.45, min(1.0, max(0.8, oracle + 0.08)))
        ax.legend(frameon=False, fontsize=8)
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(fig_dir / "bayes_oracle_width_gap.pdf")
        fig.savefig(fig_dir / "bayes_oracle_width_gap.png", dpi=200)
        plt.close(fig)

    margin_summary_path = out_dir / "bayes_margin_summary.csv"
    if margin_summary_path.exists():
        margins = pd.read_csv(margin_summary_path)
        fig, ax = plt.subplots(figsize=(5.4, 3.4))
        ax.plot(
            margins["gamma"],
            margins["score_margin_le_gamma_mean"],
            marker="o",
            label="score margin",
        )
        ax.plot(
            margins["gamma"],
            margins["posterior_margin_le_gamma_mean"],
            marker="s",
            label="posterior margin",
        )
        ax.set_xscale("log")
        ax.set_xlabel("Margin threshold gamma")
        ax.set_ylabel("Fraction <= gamma")
        ax.set_ylim(0.0, 1.0)
        ax.legend(frameon=False, fontsize=8)
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(fig_dir / "bayes_oracle_margin_curve.pdf")
        fig.savefig(fig_dir / "bayes_oracle_margin_curve.png", dpi=200)
        plt.close(fig)


def write_readme(out_dir: Path, rows: pd.DataFrame) -> None:
    span = rows[rows["row_type"] == "span_sweep"].copy()
    width = rows[rows["row_type"] == "width_sweep"].copy()
    oracle = rows[rows["row_type"] == "oracle"].copy()

    lines = [
        "# Bayes-Oracle Synthetic Results",
        "",
        "This artifact uses a synthetic distribution whose Bayes posterior is known.",
        "",
        "## Files",
        "",
        "- `bayes_oracle_rows.csv`: oracle, span-sweep, and width-sweep rows.",
        "- `bayes_oracle_summary.csv`: grouped means/std/min/max.",
        "- `bayes_margin_curve.csv`: per-seed near-margin fractions.",
        "- `bayes_margin_summary.csv`: margin curve aggregated over seeds.",
        "- `figures/`: generated PDF/PNG plots.",
        "",
    ]
    if not oracle.empty:
        lines.extend(
            [
                "## Oracle",
                "",
                f"- Mean Bayes `P_guess`: {oracle['oracle_p_guess'].mean():.6f}",
                f"- Mean Bayes MAP success: {oracle['oracle_map_success'].mean():.6f}",
                f"- Mean true MI estimate: {oracle['oracle_mi_bits'].mean():.6f} bits",
                "",
            ]
        )
    if not span.empty:
        true_span_val = int(span["true_span"].iloc[0])
        below = span[span["receptive_field"] < true_span_val]
        at = span[span["receptive_field"] == true_span_val]
        above = span[span["receptive_field"] > true_span_val]
        lines.extend(["## Span Sweep", ""])
        lines.append(f"- True exploitable span: `{true_span_val}`.")
        if not below.empty:
            lines.append(
                f"- Below span mean trained success: {below['holdout_success'].mean():.6f}; "
                f"mean expressivity gap: {below['family_expressivity_gap'].mean():.6f}."
            )
        if not at.empty:
            lines.append(
                f"- At span mean trained success: {at['holdout_success'].mean():.6f}; "
                f"mean gap to Bayes `P_guess`: {at['gap_to_oracle_p_guess'].mean():.6f}."
            )
        if not above.empty:
            lines.append(
                f"- Above span mean trained success: {above['holdout_success'].mean():.6f}; "
                f"mean gap to Bayes `P_guess`: {above['gap_to_oracle_p_guess'].mean():.6f}."
            )
        lines.append("")
    if not width.empty:
        best = width.sort_values("val_success").groupby(["seed", "width"], as_index=False).tail(1)
        best_width = best.groupby("width")["holdout_success"].mean().idxmax()
        best_rows = best[best["width"] == best_width]
        lines.extend(
            [
                "## Width Sweep",
                "",
                f"- Best mean width: `{int(best_width)}`.",
                f"- Best-width mean trained success: {best_rows['holdout_success'].mean():.6f}.",
                f"- Best-width mean gap to Bayes `P_guess`: {best_rows['gap_to_oracle_p_guess'].mean():.6f}.",
                "",
            ]
        )
    lines.extend(
        [
            "## Interpretation",
            "",
            "The span sweep is the direct oracle-coverage check: spans below the",
            "planted interaction cannot see both shares and therefore have a positive",
            "expressivity gap, while the true span contains the Bayes score feature.",
            "The margin curve supports the regularity condition by showing that the",
            "continuous oracle has no atom at an exact posterior tie.",
            "",
        ]
    )
    (out_dir / "README.md").write_text("\n".join(lines))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="results/bayes_oracle_synthetic")
    parser.add_argument("--n-seeds", type=int, default=5)
    parser.add_argument("--n-samples", type=int, default=30000)
    parser.add_argument("--holdout-size", type=int, default=10000)
    parser.add_argument("--separation", type=int, default=64)
    parser.add_argument("--n-repetitions", type=int, default=1)
    parser.add_argument("--signal-amplitude", type=float, default=1.5)
    parser.add_argument("--noise-std", type=float, default=1.0)
    parser.add_argument("--train-frac", type=float, default=0.70)
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--receptive-fields", type=parse_int_list, default=parse_int_list("5,9,17,33,49,65,81,97,129"))
    parser.add_argument("--widths", type=parse_int_list, default=parse_int_list("8,16,32,64,128"))
    parser.add_argument("--restarts", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--delta", type=float, default=1e-6)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--skip-mlp", action="store_true")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--margin-gammas", type=parse_float_list, default=None)
    parser.add_argument("--margin-gamma-grid", default="0.001,0.003,0.01,0.03,0.1,0.3,1,3,10")
    args = parser.parse_args()
    if args.margin_gammas is None:
        args.margin_gammas = [float(v) for v in args.margin_gamma_grid.split(",") if v.strip()]
    if args.num_shards < 1:
        raise ValueError("--num-shards must be positive")
    if not (0 <= args.shard_index < args.num_shards):
        raise ValueError("--shard-index must be in [0, num_shards)")
    if torch is None and not args.skip_mlp:
        print("[warn] torch unavailable; skipping MLP width sweep", flush=True)
        args.skip_mlp = True
    return args


def main() -> int:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows_path = out_dir / "bayes_oracle_rows.csv"
    margin_path = out_dir / "bayes_margin_curve.csv"

    print(f"[config] {json.dumps({k: str(v) for k, v in vars(args).items()}, sort_keys=True)}", flush=True)
    print(f"[theory] true_span={true_span(args.separation)}", flush=True)

    split_cache: dict[int, tuple[DataSplit, dict[str, float]]] = {}
    t0 = time.time()
    for task_type, seed, value, restart in task_ids(args):
        if seed not in split_cache:
            split = generate_split(
                seed=seed,
                n_samples=args.n_samples,
                holdout_size=args.holdout_size,
                separation=args.separation,
                n_repetitions=args.n_repetitions,
                signal_amplitude=args.signal_amplitude,
                noise_std=args.noise_std,
                train_frac=args.train_frac,
                val_frac=args.val_frac,
            )
            oracle = oracle_metrics(split.y_hold, split.llr_hold)
            split_cache[seed] = (split, oracle)
            write_margin_rows(margin_path, seed=seed, split=split, oracle=oracle, args=args)
        split, oracle = split_cache[seed]
        span = true_span(args.separation)
        base = {
            "seed": seed,
            "n_samples": args.n_samples,
            "n_train": int(len(split.y_train)),
            "n_val": int(len(split.y_val)),
            "n_holdout": int(len(split.y_hold)),
            "separation": args.separation,
            "true_span": span,
            "n_repetitions": args.n_repetitions,
            "signal_amplitude": args.signal_amplitude,
            "noise_std": args.noise_std,
            "starts_json": json.dumps(split.starts),
            "delta": args.delta,
            **oracle,
        }

        if task_type == "oracle":
            cert = finite_cert_row(oracle["oracle_map_success"], len(split.y_hold), args.delta)
            row = {
                **base,
                "row_type": "oracle",
                "model_kind": "bayes_map_oracle",
                "receptive_field": span,
                "width": np.nan,
                "restart": np.nan,
                "contains_true_pair": True,
                "bayes_complete": True,
                "family_oracle_p_guess": oracle["oracle_p_guess"],
                "family_oracle_mi_bits": oracle["oracle_mi_bits"],
                "family_expressivity_gap": 0.0,
                "val_success": np.nan,
                "holdout_success": oracle["oracle_map_success"],
                "gap_to_oracle_p_guess": oracle["oracle_p_guess"] - oracle["oracle_map_success"],
                **cert,
            }
            append_row(rows_path, row, ["row_type", "model_kind", "seed", "receptive_field", "width", "restart"])
            print(f"[oracle] seed={seed} p_guess={oracle['oracle_p_guess']:.4f} mi={oracle['oracle_mi_bits']:.4f}", flush=True)
            continue

        if task_type == "span":
            rf = int(value)
            contains = rf >= span
            family_p = oracle["oracle_p_guess"] if contains else CHANCE
            family_mi = oracle["oracle_mi_bits"] if contains else 0.0
            val_success, hold_success, n_features, model_kind = logistic_product_lag_success(split, rf, seed)
            cert = finite_cert_row(hold_success, len(split.y_hold), args.delta)
            row = {
                **base,
                "row_type": "span_sweep",
                "model_kind": model_kind,
                "receptive_field": rf,
                "width": np.nan,
                "restart": np.nan,
                "contains_true_pair": contains,
                "bayes_complete": contains,
                "family_oracle_p_guess": family_p,
                "family_oracle_mi_bits": family_mi,
                "family_expressivity_gap": oracle["oracle_p_guess"] - family_p,
                "val_success": val_success,
                "holdout_success": hold_success,
                "gap_to_oracle_p_guess": oracle["oracle_p_guess"] - hold_success,
                "n_features": n_features,
                **cert,
            }
            append_row(rows_path, row, ["row_type", "model_kind", "seed", "receptive_field", "width", "restart"])
            print(
                f"[span] seed={seed} R={rf} hold={hold_success:.4f} "
                f"family_gap={row['family_expressivity_gap']:.4f}",
                flush=True,
            )
            continue

        if task_type == "width":
            width = int(value)
            restart_i = int(restart or 0)
            mlp = train_raw_mlp(
                split,
                width=width,
                restart=restart_i,
                seed=seed,
                batch_size=args.batch_size,
                epochs=args.epochs,
                patience=args.patience,
                lr=args.lr,
                device_name=args.device,
            )
            cert = finite_cert_row(float(mlp["holdout_success"]), len(split.y_hold), args.delta)
            row = {
                **base,
                "row_type": "width_sweep",
                "receptive_field": span,
                "contains_true_pair": True,
                "bayes_complete": True,
                "family_oracle_p_guess": oracle["oracle_p_guess"],
                "family_oracle_mi_bits": oracle["oracle_mi_bits"],
                "family_expressivity_gap": 0.0,
                "gap_to_oracle_p_guess": oracle["oracle_p_guess"] - float(mlp["holdout_success"]),
                **mlp,
                **cert,
            }
            append_row(rows_path, row, ["row_type", "model_kind", "seed", "receptive_field", "width", "restart"])
            print(
                f"[width] seed={seed} W={width} restart={restart_i} "
                f"hold={mlp['holdout_success']:.4f}",
                flush=True,
            )

    summarize_results(out_dir)
    print(f"[done] wrote {rows_path}", flush=True)
    print(f"[done] elapsed_sec={time.time() - t0:.1f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
