#!/usr/bin/env python3
"""Gaussian byte-oracle suite and calibration diagnostic.

This synthetic known-posterior diagnostic uses a multiclass distribution,
a declared trained suite, and posterior quality metrics. The distribution is
deliberately simple and SCA-like: each byte class is represented by its eight
bits, mapped to +/- amplitude, then observed through shared-covariance
Gaussian noise.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", tempfile.mkdtemp(prefix="gauss_oracle_mpl_"))
os.environ.setdefault("XDG_CACHE_HOME", tempfile.mkdtemp(prefix="gauss_oracle_cache_"))


K = 256
CHANCE = 1.0 / K


def class_means(amplitude: float) -> np.ndarray:
    vals = np.arange(K, dtype=np.uint16)
    bits = ((vals[:, None] >> np.arange(7, -1, -1)) & 1).astype(np.float64)
    return float(amplitude) * (2.0 * bits - 1.0)


def generate(labels: np.ndarray, means: np.ndarray, sigma: float, rng: np.random.Generator) -> np.ndarray:
    return means[labels] + rng.normal(0.0, float(sigma), size=(len(labels), means.shape[1]))


def logsumexp(a: np.ndarray, axis: int = 1) -> np.ndarray:
    m = np.max(a, axis=axis, keepdims=True)
    return (m + np.log(np.sum(np.exp(a - m), axis=axis, keepdims=True))).squeeze(axis)


def posterior_probs(x: np.ndarray, means: np.ndarray, sigma: float) -> np.ndarray:
    sq = ((x[:, None, :] - means[None, :, :]) ** 2).sum(axis=2)
    logits = -0.5 * sq / (float(sigma) ** 2)
    logits -= logsumexp(logits, axis=1)[:, None]
    return np.exp(logits)


def estimate_template_means(x: np.ndarray, y: np.ndarray, prior_means: np.ndarray) -> np.ndarray:
    out = prior_means.copy()
    for cls in range(K):
        mask = y == cls
        if np.any(mask):
            out[cls] = x[mask].mean(axis=0)
    return out


def kl_bernoulli(q: float, p: float) -> float:
    eps = 1e-15
    q = min(1.0 - eps, max(eps, float(q)))
    p = min(1.0 - eps, max(eps, float(p)))
    return q * math.log(q / p) + (1.0 - q) * math.log((1.0 - q) / (1.0 - p))


def kl_upper(successes: int, n: int, alpha: float) -> float:
    q = successes / n
    if successes >= n:
        return 1.0
    lo, hi = q, 1.0 - 1e-15
    target = math.log(1.0 / alpha) / n
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        if kl_bernoulli(q, mid) > target:
            hi = mid
        else:
            lo = mid
    return hi


def bi_bits(p_success: float) -> float:
    return min(math.log2(K), max(0.0, math.log2(K * max(float(p_success), 1e-300))))


def multiclass_metrics(probs: np.ndarray, y: np.ndarray, n_bins: int) -> dict[str, float]:
    n = len(y)
    pred = np.argmax(probs, axis=1)
    top_sorted = np.argsort(probs, axis=1)
    true_probs = np.clip(probs[np.arange(n), y], 1e-15, 1.0)
    conf = probs[np.arange(n), pred]
    correct = pred == y
    onehot_term = np.sum(probs * probs, axis=1) - 2.0 * true_probs + 1.0

    ece = 0.0
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    for lo, hi in zip(edges[:-1], edges[1:]):
        if hi == 1.0:
            mask = (conf >= lo) & (conf <= hi)
        else:
            mask = (conf >= lo) & (conf < hi)
        if np.any(mask):
            ece += float(np.mean(mask)) * abs(float(np.mean(correct[mask])) - float(np.mean(conf[mask])))

    return {
        "top1_success": float(np.mean(correct)),
        "top5_success": float(np.mean([yy in row[-5:] for yy, row in zip(y, top_sorted)])),
        "top10_success": float(np.mean([yy in row[-10:] for yy, row in zip(y, top_sorted)])),
        "nll": float(-np.mean(np.log(true_probs))),
        "brier": float(np.mean(onehot_term)),
        "ece": ece,
        "mean_confidence": float(np.mean(conf)),
        "mean_true_probability": float(np.mean(true_probs)),
    }


def run_one(seed: int, args: argparse.Namespace, means: np.ndarray) -> list[dict[str, object]]:
    rng = np.random.default_rng(seed)
    y_train = rng.integers(0, K, size=args.n_train, dtype=np.int64)
    y_attack = rng.integers(0, K, size=args.n_attack, dtype=np.int64)
    x_train = generate(y_train, means, args.sigma, rng)
    x_attack = generate(y_attack, means, args.sigma, rng)

    learned_means = estimate_template_means(x_train, y_train, means)
    models = [
        ("bayes_oracle_known_means", means, "known posterior reference"),
        ("learned_gaussian_template", learned_means, "declared trained template suite"),
    ]
    rows: list[dict[str, object]] = []
    oracle_probs = posterior_probs(x_attack, means, args.sigma)
    oracle_success = float(np.mean(np.argmax(oracle_probs, axis=1) == y_attack))
    for model_name, model_means, scope in models:
        probs = posterior_probs(x_attack, model_means, args.sigma)
        metrics = multiclass_metrics(probs, y_attack, args.ece_bins)
        successes = int(round(metrics["top1_success"] * args.n_attack))
        p_plus = kl_upper(successes, args.n_attack, args.delta)
        rows.append(
            {
                "seed": seed,
                "scenario": "gaussian_byte_bitcode_shared_covariance",
                "K": K,
                "d": int(means.shape[1]),
                "amplitude": float(args.amplitude),
                "sigma": float(args.sigma),
                "n_train": int(args.n_train),
                "n_attack": int(args.n_attack),
                "model": model_name,
                "scope": scope,
                "M": 1,
                "p_bayes": oracle_success,
                "p_suite": metrics["top1_success"],
                "p_suite_plus": p_plus,
                "delta_bayes": oracle_success - metrics["top1_success"],
                "BI_bayes": bi_bits(oracle_success),
                "BI_suite_obs": bi_bits(metrics["top1_success"]),
                "BI_suite_plus": bi_bits(p_plus),
                **metrics,
            }
        )
    return rows


def summarize(rows: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [
        "p_bayes",
        "p_suite",
        "p_suite_plus",
        "delta_bayes",
        "BI_bayes",
        "BI_suite_obs",
        "BI_suite_plus",
        "top1_success",
        "top5_success",
        "top10_success",
        "nll",
        "brier",
        "ece",
        "mean_confidence",
        "mean_true_probability",
    ]
    out = rows.groupby(["scenario", "model", "scope"], dropna=False)[metric_cols].agg(["mean", "std"]).reset_index()
    out.columns = ["_".join(str(x) for x in col).rstrip("_") if isinstance(col, tuple) else col for col in out.columns]
    out["seed_count"] = rows.groupby(["scenario", "model", "scope"], dropna=False).size().to_numpy()
    return out


def write_readme(out_dir: Path, summary: pd.DataFrame, args: argparse.Namespace) -> None:
    learned = summary[summary["model"] == "learned_gaussian_template"].iloc[0]
    oracle = summary[summary["model"] == "bayes_oracle_known_means"].iloc[0]
    lines = [
        "# Gaussian Byte Oracle Suite",
        "",
        "Known-distribution oracle diagnostic for declared-suite calibration.",
        "",
        "Synthetic model: `X in {0,...,255}` is encoded as eight +/- amplitude",
        "bit means and observed through shared spherical Gaussian noise.",
        f"The target alphabet has K={K}, so the maximum byte-scale BI is log2(K)=8 bits.",
        "",
        "## Configuration",
        "",
        f"- amplitude: {args.amplitude}",
        f"- sigma: {args.sigma}",
        f"- seeds: {args.n_seeds}",
        f"- n_train per seed: {args.n_train}",
        f"- n_attack per seed: {args.n_attack}",
        "",
        "## Main Result",
        "",
        f"- Bayes oracle success: {oracle['p_suite_mean']:.6f}",
        f"- Bayes oracle `BI_bayes`: {oracle['BI_bayes_mean']:.6f} bits",
        f"- Learned template-suite success: {learned['p_suite_mean']:.6f}",
        f"- Bayes gap: {learned['delta_bayes_mean']:.6f}",
        f"- Learned template `BI_suite_plus`: {learned['BI_suite_plus_mean']:.6f} bits",
        f"- Learned template NLL: {learned['nll_mean']:.6f}",
        f"- Learned template Brier: {learned['brier_mean']:.6f}",
        f"- Learned template ECE: {learned['ece_mean']:.6f}",
        "",
        "## Files",
        "",
        "- `gaussian_oracle_suite_rows.csv`: per-seed oracle and learned-template rows.",
        "- `gaussian_oracle_suite_summary.csv`: mean/std over seeds.",
        "",
    ]
    (out_dir / "README.md").write_text("\n".join(lines))


def write_plot(out_dir: Path, summary: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt

    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    plot = summary.sort_values("model")
    x = np.arange(len(plot))
    fig, ax = plt.subplots(figsize=(5.8, 3.4))
    ax.bar(x - 0.18, plot["BI_suite_obs_mean"], width=0.36, label="BI obs")
    ax.bar(x + 0.18, plot["BI_suite_plus_mean"], width=0.36, label="BI+")
    ax.set_xticks(x)
    ax.set_xticklabels(["Bayes oracle" if m.startswith("bayes") else "Learned template" for m in plot["model"]])
    ax.set_ylabel("BI bits")
    ax.set_ylim(0, 8)
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(fig_dir / "gaussian_oracle_suite_bi.pdf")
    fig.savefig(fig_dir / "gaussian_oracle_suite_bi.png", dpi=200)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="results/bayes_gaussian_oracle_suite")
    parser.add_argument("--n-seeds", type=int, default=3)
    parser.add_argument("--n-train", type=int, default=102400)
    parser.add_argument("--n-attack", type=int, default=50000)
    parser.add_argument("--amplitude", type=float, default=2.0)
    parser.add_argument("--sigma", type=float, default=1.0)
    parser.add_argument("--delta", type=float, default=1e-6)
    parser.add_argument("--ece-bins", type=int, default=15)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    means = class_means(args.amplitude)
    rows: list[dict[str, object]] = []
    for seed in range(args.n_seeds):
        rows.extend(run_one(seed, args, means))
    row_df = pd.DataFrame(rows)
    summary = summarize(row_df)
    row_df.to_csv(out_dir / "gaussian_oracle_suite_rows.csv", index=False)
    summary.to_csv(out_dir / "gaussian_oracle_suite_summary.csv", index=False)
    write_readme(out_dir, summary, args)
    write_plot(out_dir, summary)
    print(json.dumps({"out_dir": str(out_dir), "rows": len(row_df)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
