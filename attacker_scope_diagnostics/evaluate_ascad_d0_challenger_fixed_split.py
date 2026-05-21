#!/usr/bin/env python3
"""Re-evaluate ASCAD d0 challengers on the fixed profiling split.

The original challenger row for ASCAD d0 came from the
train/validation/holdout split over the ASCAD profiling traces. This script
keeps that challenger construction fixed, but evaluates each trained challenger
on the declared fixed profiling population used by the corrected Table 1 row.
It writes a separate artifact instead of overwriting the scope-diagnostic
checkpoint.
"""

from __future__ import annotations

import argparse
import json
import math
import time
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from attacker_scope_diagnostics import attacker_scope_diagnostic as scope_module
from core.data_loader import load_dataset


K = 256


@dataclass
class FixedSplitRow:
    dataset: str
    seed: int
    scope_level: int
    scope_name: str
    M: int
    n_train: int
    n_old_holdout: int
    n_fixed_eval: int
    best_model_fixed: str
    p_fixed: float
    bi_fixed_obs: float
    best_model_old_holdout: str
    p_old_holdout: float
    per_model_success_fixed: str
    per_model_success_old_holdout: str
    protocol: str


def parse_scopes(text: str) -> list[int]:
    scopes = [int(part.strip()) for part in text.split(",") if part.strip()]
    bad = [scope for scope in scopes if scope not in {1, 2, 3, 4}]
    if bad:
        raise argparse.ArgumentTypeError(f"invalid scope levels: {bad}")
    return scopes


def bi_from_p(p: float, k: int = K) -> float:
    if not math.isfinite(p):
        return float("nan")
    return max(0.0, min(math.log2(k), math.log2(1.0 + k * max(p - 1.0 / k, 0.0))))


def select_features(x: np.ndarray, max_features: int, method: str) -> np.ndarray:
    if max_features <= 0 or x.ndim != 2 or x.shape[1] <= max_features:
        return x
    if method == "stride":
        idx = np.linspace(0, x.shape[1] - 1, max_features, dtype=int)
    else:
        scores = np.nan_to_num(np.var(x, axis=0), nan=0.0, posinf=0.0, neginf=0.0)
        idx = np.argsort(scores)[-max_features:]
        idx.sort()
    return x[:, idx]


def predict_sklearn(model, x_scaled: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return np.argmax(model.predict_proba(x_scaled), axis=1)
    return model.predict(x_scaled)


def evaluate_models(
    suite,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_fixed: np.ndarray,
    y_fixed: np.ndarray,
    x_old_holdout: np.ndarray,
    y_old_holdout: np.ndarray,
) -> tuple[dict[str, float], dict[str, float]]:
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_fixed_scaled = scaler.transform(x_fixed)
    x_hold_scaled = scaler.transform(x_old_holdout)

    y_fixed = np.asarray(y_fixed).reshape(-1)
    y_old_holdout = np.asarray(y_old_holdout).reshape(-1)
    fixed_success: dict[str, float] = {}
    holdout_success: dict[str, float] = {}

    for name, model in suite:
        if scope_module.HAS_TORCH and isinstance(model, scope_module.nn.Module):
            warnings.warn(f"skipping torch model {name}; matched ASCAD challenger checkpoint had no CNN rows")
            continue
        try:
            model.fit(x_train_scaled, y_train)
            pred_fixed = np.asarray(predict_sklearn(model, x_fixed_scaled)).reshape(-1)
            pred_hold = np.asarray(predict_sklearn(model, x_hold_scaled)).reshape(-1)
            fixed_success[name] = float(np.mean(pred_fixed == y_fixed[: len(pred_fixed)]))
            holdout_success[name] = float(np.mean(pred_hold == y_old_holdout[: len(pred_hold)]))
        except Exception as exc:
            warnings.warn(f"failed {name}: {exc}")

    return fixed_success, holdout_success


def best_item(scores: dict[str, float]) -> tuple[str, float]:
    if not scores:
        return "none", float("nan")
    return max(scores.items(), key=lambda kv: kv[1])


def filter_suite(suite, model_family: str):
    if model_family == "all":
        return suite
    if model_family == "mlp":
        return [(name, model) for name, model in suite if name.startswith("MLP_")]
    raise ValueError(f"unsupported model family: {model_family}")


def scope_model_names(scope: int, n_features: int, args: argparse.Namespace) -> list[str]:
    suite = scope_module.build_scope(
        scope,
        n_features,
        args.n_classes,
        args.target_cnn_params,
        args.device,
    )
    suite = filter_suite(suite, args.model_family)
    return [name for name, model in suite if not (scope_module.HAS_TORCH and isinstance(model, scope_module.nn.Module))]


def run(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    # The existing ASCAD challenger checkpoint has M=12 for these scopes and no CNN.
    # per-model entries. Disabling CNN additions keeps this rerun matched.
    if args.disable_cnn:
        scope_module.HAS_TORCH = False

    x, y = load_dataset(
        args.dataset,
        data_dir=args.dataset_dir,
        tches20_src_dir=args.tches20_src_dir,
    )
    x = select_features(np.asarray(x), args.max_features, args.feature_select)
    y = np.asarray(y).reshape(-1).astype(np.int64)
    if len(x) != len(y):
        raise ValueError(f"trace/label length mismatch: {len(x)} vs {len(y)}")

    rows: list[FixedSplitRow] = []
    n_total = len(y)
    train_size = int(0.6 * n_total)
    val_size = int(0.2 * n_total)
    old_holdout_size = min(args.holdout_size, n_total - train_size - val_size)
    protocol = (
        "Scope-diagnostic challenger construction over ASCAD profiling traces; "
        "train+val subset used for fitting, old holdout retained for "
        "reproduction check, fixed_eval is the full corrected profiling split."
    )
    if args.model_family != "all":
        protocol += f" Targeted model-family rerun: {args.model_family}."

    for seed in args.seeds:
        rng = np.random.RandomState(seed)
        split_idx = rng.permutation(n_total)
        idx_train = split_idx[:train_size]
        idx_val = split_idx[train_size : train_size + val_size]
        idx_hold = split_idx[train_size + val_size : train_size + val_size + old_holdout_size]
        idx_train_full = np.concatenate([idx_train, idx_val])

        x_train = x[idx_train_full]
        y_train = y[idx_train_full]
        x_hold = x[idx_hold]
        y_hold = y[idx_hold]

        if args.disable_cnn:
            t0 = time.time()
            full_suite = scope_module.build_scope(
                2,
                x.shape[1],
                args.n_classes,
                args.target_cnn_params,
                args.device,
            )
            full_suite = filter_suite(full_suite, args.model_family)
            full_fixed_success, full_holdout_success = evaluate_models(
                full_suite,
                x_train,
                y_train,
                x,
                y,
                x_hold,
                y_hold,
            )
            print(
                f"seed={seed} trained matched linear+MLP challenger pool "
                f"M={len(full_fixed_success)} time={time.time() - t0:.1f}s",
                flush=True,
            )
        else:
            full_fixed_success = {}
            full_holdout_success = {}

        for scope in args.scopes:
            t0 = time.time()
            if args.disable_cnn:
                effective_scope = 1 if scope == 1 else 2
                names = scope_model_names(effective_scope, x.shape[1], args)
                fixed_success = {name: full_fixed_success[name] for name in names if name in full_fixed_success}
                holdout_success = {name: full_holdout_success[name] for name in names if name in full_holdout_success}
            else:
                suite = scope_module.build_scope(
                    scope,
                    x.shape[1],
                    args.n_classes,
                    args.target_cnn_params,
                    args.device,
                )
                suite = filter_suite(suite, args.model_family)
                fixed_success, holdout_success = evaluate_models(
                    suite,
                    x_train,
                    y_train,
                    x,
                    y,
                    x_hold,
                    y_hold,
                )
            best_fixed_model, p_fixed = best_item(fixed_success)
            best_hold_model, p_hold = best_item(holdout_success)
            print(
                f"seed={seed} scope={scope} M={len(fixed_success)} "
                f"fixed={p_fixed:.6f} ({best_fixed_model}) "
                f"old_holdout={p_hold:.6f} ({best_hold_model}) "
                f"time={time.time() - t0:.1f}s",
                flush=True,
            )
            rows.append(
                FixedSplitRow(
                    dataset=args.dataset,
                    seed=seed,
                    scope_level=scope,
                    scope_name=scope_module.SCOPE_NAMES[scope],
                    M=len(fixed_success),
                    n_train=len(y_train),
                    n_old_holdout=len(y_hold),
                    n_fixed_eval=len(y),
                    best_model_fixed=best_fixed_model,
                    p_fixed=p_fixed,
                    bi_fixed_obs=bi_from_p(p_fixed, args.n_classes),
                    best_model_old_holdout=best_hold_model,
                    p_old_holdout=p_hold,
                    per_model_success_fixed=json.dumps(fixed_success, sort_keys=True),
                    per_model_success_old_holdout=json.dumps(holdout_success, sort_keys=True),
                    protocol=protocol,
                )
            )

    row_df = pd.DataFrame([asdict(row) for row in rows])
    if row_df.empty:
        raise RuntimeError("no challenger rows were produced")
    best = row_df.sort_values("p_fixed", ascending=False).iloc[0]
    summary = pd.DataFrame(
        [
            {
                "dataset": args.dataset,
                "n_rows": int(len(row_df)),
                "n_fixed_eval": int(best["n_fixed_eval"]),
                "best_scope_level": int(best["scope_level"]),
                "best_scope_name": best["scope_name"],
                "best_seed": int(best["seed"]),
                "best_model": best["best_model_fixed"],
                "best_challenger_p_hat": float(best["p_fixed"]),
                "best_challenger_BI_obs": float(best["bi_fixed_obs"]),
                "old_holdout_reproduction_p_hat": float(best["p_old_holdout"]),
                "old_holdout_best_model": best["best_model_old_holdout"],
                "protocol": best["protocol"],
            }
        ]
    )
    return row_df, summary


def write_summary_md(summary: pd.DataFrame, rows: pd.DataFrame, path: Path) -> None:
    best = summary.iloc[0]
    old_best = rows.sort_values("p_old_holdout", ascending=False).iloc[0]
    lines = [
        "# ASCAD d0 Challenger Fixed-Split Rerun",
        "",
        f"Rows evaluated: {int(best['n_rows'])}",
        f"Fixed evaluation population: {int(best['n_fixed_eval'])} profiling traces",
        "",
        "## Best Fixed-Split Challenger",
        "",
        f"- Scope: {best['best_scope_name']} (level {int(best['best_scope_level'])})",
        f"- Seed: {int(best['best_seed'])}",
        f"- Model: `{best['best_model']}`",
        f"- Matched fixed-split success: `{float(best['best_challenger_p_hat']):.6f}`",
        f"- Observed BI from point success: `{float(best['best_challenger_BI_obs']):.6f}` bits",
        "",
        "## Old-Holdout Reproduction Check",
        "",
        f"- Strongest old-holdout row in this rerun: scope {int(old_best['scope_level'])}, "
        f"seed {int(old_best['seed'])}, model `{old_best['best_model_old_holdout']}`, "
        f"success `{float(old_best['p_old_holdout']):.6f}`",
        "",
        "## Protocol",
        "",
        str(best["protocol"]),
        "",
    ]
    path.write_text("\n".join(lines))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="ascad_desync_0")
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--tches20-src-dir", required=True)
    parser.add_argument("--output-dir", default="results/ascad_fixed_split_challenger")
    parser.add_argument("--seeds", type=lambda s: [int(x) for x in s.split(",") if x], default=[0, 1, 2, 3, 4])
    parser.add_argument("--scopes", type=parse_scopes, default=[1, 2, 3, 4])
    parser.add_argument("--holdout-size", type=int, default=25000)
    parser.add_argument("--n-classes", type=int, default=256)
    parser.add_argument("--target-cnn-params", type=int, default=50000)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-features", type=int, default=700)
    parser.add_argument("--feature-select", choices=["variance", "stride"], default="variance")
    parser.add_argument("--model-family", choices=["all", "mlp"], default="all")
    parser.add_argument("--disable-cnn", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows, summary = run(args)
    rows.to_csv(output_dir / "ascad_d0_challenger_fixed_split_rows.csv", index=False)
    summary.to_csv(output_dir / "ascad_d0_challenger_fixed_split_summary.csv", index=False)
    write_summary_md(summary, rows, output_dir / "ASCAD_D0_CHALLENGER_FIXED_SPLIT_SUMMARY.md")
    print(f"wrote {len(rows)} rows to {output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
