#!/usr/bin/env python3
"""Build paper-safe BI-suite artifacts for the updated AsiaCRYPT evidence.

This script recomputes the Section 3/4 KL-binomial BI^{suite} bracket from
per-model holdout success rates. It never reuses legacy scalar `bi`
column except as provenance in the seed-level CSV.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import tempfile
import warnings
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning, module="pandas")
os.environ.setdefault("MPLCONFIGDIR", tempfile.mkdtemp(prefix="bi_suite_mpl_"))


TARGET_DATASETS = [
    "ascad_desync_0",
    "ascad_desync_50",
    "ascad_desync_100",
    "ascad_random_key",
    "aes_hd",
    "aes_rd",
    "dpav4",
    "ches_ctf_2025",
]

REUSABLE_DATASETS = {
    "ascad_desync_0": "tches20_m80_pretrained_suite",
    "ascad_desync_50": "tches20_m80_pretrained_suite",
    "ascad_desync_100": "tches20_m80_pretrained_suite",
    "aes_hd": "tches20_m80_pretrained_suite",
    "aes_rd": "tches20_m80_pretrained_suite",
    "dpav4": "tches20_m80_pretrained_suite",
}

PENDING_BI_ROWS = {
    "ascad_random_key": {
        "scope_id": "ascad_random_estranet_checkpoint_suite",
        "source_artifact": (
            "checkpoints/estranet/ascadr"
        ),
        "status": "pending_remote_bi_suite_success_counts",
    },
    "ches_ctf_2025": {
        "scope_id": "ches_ctf_2025_estranet_w3000_checkpoint_suite",
        "source_artifact": (
            "checkpoints/estranet/ches25_lx_w3000"
        ),
        "status": "pending_remote_bi_suite_success_counts",
    },
}

BI_TABLE_COLUMNS = [
    "dataset",
    "scope_id",
    "M",
    "n_te",
    "p_obs",
    "p_minus",
    "p_plus",
    "BI_minus",
    "BI_obs",
    "BI_plus",
    "slack",
    "R_margin",
    "best_model",
    "source_artifact",
    "status",
]


def bernoulli_kl(p: float, q: float) -> float:
    """D_Ber(p || q), with boundary handling."""
    if p == q:
        return 0.0
    if q <= 0.0:
        return 0.0 if p <= 0.0 else math.inf
    if q >= 1.0:
        return 0.0 if p >= 1.0 else math.inf
    out = 0.0
    if p > 0.0:
        out += p * math.log(p / q)
    if p < 1.0:
        out += (1.0 - p) * math.log((1.0 - p) / (1.0 - q))
    return out


def kl_lower_endpoint(successes: int, n: int, c: float) -> float:
    p_hat = successes / n
    if successes <= 0:
        return 0.0
    lo, hi = 0.0, p_hat
    for _ in range(80):
        mid = (lo + hi) / 2.0
        if bernoulli_kl(p_hat, mid) > c:
            lo = mid
        else:
            hi = mid
    return hi


def kl_upper_endpoint(successes: int, n: int, c: float) -> float:
    p_hat = successes / n
    if successes >= n:
        return 1.0
    lo, hi = p_hat, 1.0
    for _ in range(80):
        mid = (lo + hi) / 2.0
        if bernoulli_kl(p_hat, mid) > c:
            hi = mid
        else:
            lo = mid
    return lo


def bi_bits(p: float, n_classes: int) -> float:
    if pd.isna(p):
        return math.nan
    return max(0.0, math.log2(max(p, 0.0) * n_classes))


def parse_success_json(value: Any) -> Dict[str, float]:
    if isinstance(value, dict):
        return {str(k): float(v) for k, v in value.items()}
    if not isinstance(value, str) or not value.strip():
        return {}
    return {str(k): float(v) for k, v in json.loads(value).items()}


def compute_bi_row(
    *,
    dataset: str,
    scope_id: str,
    seed: Any,
    n_te: int,
    per_model_success: Dict[str, float],
    source_artifact: str,
    delta: float,
    n_classes: int,
    old_bi: Optional[float] = None,
) -> Dict[str, Any]:
    if not per_model_success:
        raise ValueError(f"{dataset} seed={seed} has no per-model success values")

    m = len(per_model_success)
    c = math.log((2.0 * m) / delta) / float(n_te)
    per_model_counts = {
        model_id: int(round(float(success_rate) * n_te))
        for model_id, success_rate in per_model_success.items()
    }

    best_model, best_successes = max(per_model_counts.items(), key=lambda kv: kv[1])
    p_obs = best_successes / n_te
    lowers = [kl_lower_endpoint(s, n_te, c) for s in per_model_counts.values()]
    uppers = [kl_upper_endpoint(s, n_te, c) for s in per_model_counts.values()]
    p_minus = max(lowers)
    p_plus = max(uppers)
    slack = p_plus - p_obs
    denom = p_obs - (1.0 / n_classes)
    r_margin = math.inf if denom <= 0.0 and slack > 0.0 else slack / max(denom, 1e-300)

    return {
        "dataset": dataset,
        "scope_id": scope_id,
        "seed": seed,
        "M": m,
        "n_te": int(n_te),
        "p_obs": p_obs,
        "p_minus": p_minus,
        "p_plus": p_plus,
        "BI_minus": bi_bits(p_minus, n_classes),
        "BI_obs": bi_bits(p_obs, n_classes),
        "BI_plus": bi_bits(p_plus, n_classes),
        "slack": slack,
        "R_margin": r_margin,
        "best_model": best_model,
        "source_artifact": source_artifact,
        "status": "complete",
        "old_bi_column": old_bi,
        "per_model_success": json.dumps(per_model_success, sort_keys=True),
    }


def load_reusable_suite_rows(
    reusable_suite_path: Path,
    *,
    delta: float,
    n_classes: int,
) -> List[Dict[str, Any]]:
    df = pd.read_csv(reusable_suite_path)
    rows: List[Dict[str, Any]] = []
    for _, record in df.iterrows():
        dataset = str(record["dataset"])
        if dataset not in REUSABLE_DATASETS:
            continue
        per_model_success = parse_success_json(record["per_model_success"])
        rows.append(
            compute_bi_row(
                dataset=dataset,
                scope_id=REUSABLE_DATASETS[dataset],
                seed=record.get("seed", ""),
                n_te=int(record["n_holdout"]),
                per_model_success=per_model_success,
                source_artifact=str(reusable_suite_path),
                delta=delta,
                n_classes=n_classes,
                old_bi=float(record["bi"]) if "bi" in record else None,
            )
        )
    return rows


def load_extra_raw_success(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    df = pd.read_csv(path)
    if df.empty:
        return []

    if "per_model_success" in df.columns:
        out = []
        for _, record in df.iterrows():
            out.append(
                {
                    "dataset": str(record["dataset"]),
                    "scope_id": str(record["scope_id"]),
                    "seed": record.get("seed", 0),
                    "n_te": int(record["n_te"]),
                    "per_model_success": parse_success_json(record["per_model_success"]),
                    "source_artifact": str(record.get("source_artifact", path)),
                }
            )
        return out

    required = {"dataset", "scope_id", "model_id", "n_te"}
    if not required.issubset(df.columns):
        raise ValueError(f"{path} is missing required raw success columns: {required}")

    if "success_rate" not in df.columns:
        if "successes" not in df.columns:
            raise ValueError(f"{path} needs either success_rate or successes")
        df["success_rate"] = df["successes"] / df["n_te"]

    if "seed" not in df.columns:
        df["seed"] = 0
    out = []
    group_cols = ["dataset", "scope_id", "seed", "n_te"]
    for key, group in df.groupby(group_cols, dropna=False):
        dataset, scope_id, seed, n_te = key
        per_model_success = {
            str(row["model_id"]): float(row["success_rate"])
            for _, row in group.iterrows()
        }
        out.append(
            {
                "dataset": str(dataset),
                "scope_id": str(scope_id),
                "seed": seed,
                "n_te": int(n_te),
                "per_model_success": per_model_success,
                "source_artifact": str(path),
            }
        )
    return out


def summarize_seed_rows(seed_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for dataset in TARGET_DATASETS:
        group = seed_df[seed_df["dataset"] == dataset]
        if group.empty:
            pending = PENDING_BI_ROWS.get(dataset, {})
            rows.append(
                {
                    "dataset": dataset,
                    "scope_id": pending.get("scope_id", ""),
                    "M": np.nan,
                    "n_te": np.nan,
                    "p_obs": np.nan,
                    "p_minus": np.nan,
                    "p_plus": np.nan,
                    "BI_minus": np.nan,
                    "BI_obs": np.nan,
                    "BI_plus": np.nan,
                    "slack": np.nan,
                    "R_margin": np.nan,
                    "best_model": "",
                    "source_artifact": pending.get("source_artifact", ""),
                    "status": pending.get("status", "missing"),
                    "n_runs": 0,
                }
            )
            continue

        mode_models = Counter(str(x) for x in group["best_model"]).most_common(1)
        best_model = mode_models[0][0] if mode_models else ""
        source_artifact = ";".join(sorted(set(map(str, group["source_artifact"]))))
        row = {
            "dataset": dataset,
            "scope_id": ";".join(sorted(set(map(str, group["scope_id"])))),
            "M": int(group["M"].max()),
            "n_te": int(group["n_te"].median()),
            "best_model": best_model,
            "source_artifact": source_artifact,
            "status": "complete",
            "n_runs": int(len(group)),
        }
        for col in [
            "p_obs",
            "p_minus",
            "p_plus",
            "BI_minus",
            "BI_obs",
            "BI_plus",
            "slack",
            "R_margin",
        ]:
            row[col] = float(group[col].mean())
            row[f"{col}_std"] = float(group[col].std(ddof=1)) if len(group) > 1 else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def status_from_metric(group: pd.DataFrame, metric: str) -> Dict[str, Any]:
    success_col = f"{metric}_success"
    error_col = f"{metric}_error"
    values = pd.to_numeric(group.get(metric, pd.Series(dtype=float)), errors="coerce")
    successes = group.get(success_col, pd.Series([False] * len(group))).fillna(False)
    errors = (
        group.get(error_col, pd.Series([""] * len(group)))
        .fillna("")
        .astype(str)
        .str.strip()
    )

    numeric_mask = values.notna()
    success_mask = successes.astype(bool) & numeric_mask
    error_text = "; ".join(sorted(set(e for e in errors if e and e.lower() != "nan")))

    if success_mask.any():
        status = "numeric"
    elif error_text and "Dimension" in error_text:
        status = "dimension_capped_failed"
    elif numeric_mask.any():
        status = "numeric_not_marked_success"
    elif error_text:
        status = "failed"
    else:
        status = "not_available"

    return {
        "status": status,
        "value_mean": float(values[success_mask].mean()) if success_mask.any() else np.nan,
        "value_std": float(values[success_mask].std(ddof=1)) if success_mask.sum() > 1 else np.nan,
        "success_count": int(success_mask.sum()),
        "run_count": int(len(group)),
        "error": error_text,
    }


def build_baseline_status(
    target_datasets: Iterable[str],
    baseline_checkpoint: Path,
    reusable_suite_rows: Path,
) -> pd.DataFrame:
    metrics = [
        ("pi", "ePI_or_PI"),
        ("hi", "eHI_or_HI"),
        ("mlp_pi", "MLP_PI"),
        ("mi", "GKOV_MI"),
    ]
    rows = []
    checkpoint_df = pd.read_csv(baseline_checkpoint) if baseline_checkpoint.exists() else pd.DataFrame()
    pretrained_df = pd.read_csv(reusable_suite_rows) if reusable_suite_rows.exists() else pd.DataFrame()

    for dataset in target_datasets:
        ck_group = (
            checkpoint_df[checkpoint_df["dataset"] == dataset]
            if not checkpoint_df.empty and "dataset" in checkpoint_df.columns
            else pd.DataFrame()
        )
        pretrained_has_dataset = (
            not pretrained_df.empty
            and "dataset" in pretrained_df.columns
            and dataset in set(pretrained_df["dataset"].astype(str))
        )
        for metric, label in metrics:
            if not ck_group.empty and metric in ck_group.columns:
                status = status_from_metric(ck_group, metric)
                source = str(baseline_checkpoint)
                note = "artifact-backed from a reusable baseline checkpoint"
            elif pretrained_has_dataset:
                status = {
                    "status": "not_available_in_reusable_bi_artifact",
                    "value_mean": np.nan,
                    "value_std": np.nan,
                    "success_count": 0,
                    "run_count": 0,
                    "error": "",
                }
                source = str(reusable_suite_rows)
                note = "reusable suite rows have per-model BI success only"
            else:
                status = {
                    "status": "no_real_dataset_artifact",
                    "value_mean": np.nan,
                    "value_std": np.nan,
                    "success_count": 0,
                    "run_count": 0,
                    "error": "",
                }
                source = ""
                note = "needs fresh dimension-capped baseline pass"
            rows.append(
                {
                    "dataset": dataset,
                    "metric": label,
                    "status": status["status"],
                    "value_mean": status["value_mean"],
                    "value_std": status["value_std"],
                    "success_count": status["success_count"],
                    "run_count": status["run_count"],
                    "source_artifact": source,
                    "error": status["error"],
                    "note": note,
                }
            )
    return pd.DataFrame(rows)


def first_trace_with_full_recovery(stats_path: Path, tau: int) -> Dict[str, Any]:
    if not stats_path.exists():
        return {}
    df = pd.read_csv(stats_path)
    group = df[df["tau"] == tau].sort_values("trace_count")
    if group.empty:
        return {}
    stable = group[group["success_rank0_frac"] >= 1.0]
    final = group.iloc[-1]
    return {
        "traces_to_rank_zero": int(stable.iloc[0]["trace_count"]) if not stable.empty else np.nan,
        "final_rank": float(final["rank0_median"]),
        "mean_GE": float(final["rank0_mean"]),
        "stable_zero_rank_reached": bool(not stable.empty),
    }


def parse_value(pattern: str, text: str, cast=float, default=np.nan):
    match = re.search(pattern, text, flags=re.IGNORECASE)
    if not match:
        return default
    return cast(match.group(1))


def parse_markdown_table_line(text: str, first_cell: str) -> Optional[List[str]]:
    for line in text.splitlines():
        if not line.strip().startswith("|"):
            continue
        parts = [p.strip() for p in line.strip().strip("|").split("|")]
        if parts and parts[0] == first_cell:
            return parts
    return None


def build_scsca_status(repo_root: Path) -> pd.DataFrame:
    rows = []
    ascad_stats = repo_root / "results/keyrank_ascad_meanfill/ascad_keyrank_stats.csv"
    ascad_full = first_trace_with_full_recovery(ascad_stats, 700)
    if ascad_full:
        rows.append(
            {
                "dataset": "ascad_desync_0",
                "baseline_scope": "single-byte_key_rank_cnn_meanfill_tau700",
                **ascad_full,
                "source_artifact": str(ascad_stats),
                "status": "artifact_backed",
                "note": "real ASCAD key-rank curve; not part of single-target BI table",
            }
        )

    estranet_status = repo_root / "results/estranet_keyrank/RECOVERY_STATUS.md"
    if estranet_status.exists():
        text = estranet_status.read_text()
        parts = parse_markdown_table_line(text, "ASCADf")
        if parts and len(parts) >= 7:
            rows.append(
                {
                    "dataset": "ascad_desync_0",
                    "baseline_scope": "estranet_key_rank_curve",
                    "traces_to_rank_zero": float(parts[3]),
                    "final_rank": float(parts[4]),
                    "mean_GE": float(parts[4]),
                    "stable_zero_rank_reached": True,
                    "source_artifact": str(estranet_status),
                    "status": "artifact_backed",
                    "note": "EstraNet ASCAD fixed-key key-rank evidence; separate from BI suite",
                }
            )
        parts = parse_markdown_table_line(text, "ASCADr")
        if parts and len(parts) >= 7:
            rows.append(
                {
                    "dataset": "ascad_random_key",
                    "baseline_scope": "estranet_key_rank_curve",
                    "traces_to_rank_zero": float(parts[3]),
                    "final_rank": float(parts[4]),
                    "mean_GE": float(parts[4]),
                    "stable_zero_rank_reached": True,
                    "source_artifact": str(estranet_status),
                    "status": "artifact_backed",
                    "note": "EstraNet end-to-end key-rank evidence; separate from BI suite",
                }
            )
        parts = parse_markdown_table_line(text, "3")
        if parts and len(parts) >= 5:
            rows.append(
                {
                    "dataset": "ches_ctf_2025",
                    "baseline_scope": "estranet_w3000_key_rank_ckpt3",
                    "traces_to_rank_zero": np.nan,
                    "final_rank": float(parts[3]),
                    "mean_GE": float(parts[3]),
                    "stable_zero_rank_reached": False,
                    "source_artifact": str(estranet_status),
                    "status": "artifact_backed_not_stable",
                    "note": "Best EstraNet checkpoint sweep did not reach stable rank zero",
                }
            )

    ches_status = repo_root / "results/ches2025_official_keyrank/RECOVERY_STATUS.md"
    if ches_status.exists():
        text = ches_status.read_text()
        ntge = parse_value(r"\| Test NTGE \|\s*([0-9.]+)\s*\|", text)
        rows.append(
            {
                "dataset": "ches_ctf_2025",
                "baseline_scope": "official_ches2025_key_rank_curve",
                "traces_to_rank_zero": ntge,
                "final_rank": 0.0,
                "mean_GE": 0.0,
                "stable_zero_rank_reached": True,
                "source_artifact": str(ches_status),
                "status": "artifact_backed",
                "note": "Official CHES run reaches zero rank; separate from BI suite",
            }
        )
    return pd.DataFrame(rows)


def build_scsca_inventory(scsca_status: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for dataset in TARGET_DATASETS:
        group = (
            scsca_status[scsca_status["dataset"] == dataset]
            if not scsca_status.empty and "dataset" in scsca_status.columns
            else pd.DataFrame()
        )
        if group.empty:
            rows.append(
                {
                    "dataset": dataset,
                    "status": "no_posterior_artifact_found",
                    "artifact_count": 0,
                    "baseline_scopes": "",
                    "source_artifacts": "",
                    "note": "No posterior/probability outputs with compatible metadata were found in current artifacts",
                }
            )
            continue
        rows.append(
            {
                "dataset": dataset,
                "status": "artifact_backed",
                "artifact_count": int(len(group)),
                "baseline_scopes": ";".join(group["baseline_scope"].astype(str).tolist()),
                "source_artifacts": ";".join(sorted(set(group["source_artifact"].astype(str)))),
                "note": "SCSCA/SASCA-style end-to-end rank/GE evidence is available; keep separate from BI table",
            }
        )
    return pd.DataFrame(rows)


def write_markdown_summary(
    path: Path,
    paper_table: pd.DataFrame,
    baseline_status: pd.DataFrame,
    scsca_status: pd.DataFrame,
    scsca_inventory: Optional[pd.DataFrame] = None,
) -> None:
    complete = paper_table[paper_table["status"] == "complete"]
    pending = paper_table[paper_table["status"] != "complete"]
    lines = [
        "# BI Suite Eight-Dataset Paper Artifacts",
        "",
        "This directory is generated from current repo artifacts. The primary BI rows",
        "use the KL-binomial finite-suite bracket from Sections 3/4.",
        "",
        "## BI Suite Status",
        "",
        f"- Complete BI rows: {len(complete)}/{len(paper_table)}",
        f"- Pending BI rows: {len(pending)}/{len(paper_table)}",
        "",
        "### Paper Table",
        "",
        dataframe_to_markdown(
            paper_table[
                [
                    "dataset",
                    "scope_id",
                    "M",
                    "n_te",
                    "p_obs",
                    "p_minus",
                    "p_plus",
                    "BI_obs",
                    "BI_plus",
                    "R_margin",
                    "status",
                ]
            ]
        ),
        "",
        "## Non-BI Baseline Policy",
        "",
        "Existing non-BI values are reused only when a metric value and success flag",
        "are present in the artifact. Dimension-capped failures are explicitly marked.",
        "",
        dataframe_to_markdown(
            baseline_status[
                [
                    "dataset",
                    "metric",
                    "status",
                    "value_mean",
                    "success_count",
                    "run_count",
                    "note",
                ]
            ].head(40)
        ),
        "",
        "## SCSCA/SASCA Evidence",
        "",
        "These rows are end-to-end rank/GE evidence and are intentionally separate",
        "from the single-target BI^{suite} table.",
        "",
        dataframe_to_markdown(scsca_status),
        "",
        "### SCSCA/SASCA Extension Inventory",
        "",
        dataframe_to_markdown(scsca_inventory) if scsca_inventory is not None else "_No inventory._",
        "",
        "## Manuscript Guardrails",
        "",
        "- Do not claim eight complete BI^{suite} rows until pending rows are filled.",
        "- Do not merge CHES/ASCAD-random EstraNet scopes into the TCHES20 M=80 scope.",
        "- Remove old BI^{fam}, broad five-scope Transformer, and unsupported CHES BI claims unless regenerated here.",
    ]
    path.write_text("\n".join(lines) + "\n")


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows._"
    display = df.copy()
    for col in display.columns:
        if pd.api.types.is_float_dtype(display[col]):
            display[col] = display[col].map(lambda x: "" if pd.isna(x) else f"{x:.6g}")
    rows = [list(map(str, display.columns))]
    rows += display.fillna("").astype(str).values.tolist()
    widths = [max(len(row[i]) for row in rows) for i in range(len(rows[0]))]
    out = []
    out.append("| " + " | ".join(rows[0][i].ljust(widths[i]) for i in range(len(widths))) + " |")
    out.append("| " + " | ".join("-" * widths[i] for i in range(len(widths))) + " |")
    for row in rows[1:]:
        out.append("| " + " | ".join(row[i].ljust(widths[i]) for i in range(len(widths))) + " |")
    return "\n".join(out)


def write_figures(out_dir: Path, paper_table: pd.DataFrame, baseline_status: pd.DataFrame) -> None:
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - depends on environment
        (fig_dir / "FIGURE_GENERATION_SKIPPED.txt").write_text(str(exc) + "\n")
        return

    complete = paper_table[paper_table["status"] == "complete"].copy()
    if not complete.empty:
        complete = complete.sort_values("BI_plus", ascending=False)
        x = np.arange(len(complete))
        y = complete["BI_obs"].astype(float).values
        lo = y - complete["BI_minus"].astype(float).values
        hi = complete["BI_plus"].astype(float).values - y
        fig, ax = plt.subplots(figsize=(10.0, 4.8))
        ax.errorbar(x, y, yerr=[lo, hi], fmt="o", capsize=4, color="#1f4e79")
        ax.set_xticks(x)
        ax.set_xticklabels(complete["dataset"], rotation=30, ha="right")
        ax.set_ylabel("BI bits")
        ax.set_title("KL-binomial BI-suite brackets from generated artifacts")
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        fig.savefig(fig_dir / "bi_suite_kl_brackets.pdf")
        fig.savefig(fig_dir / "bi_suite_kl_brackets.png", dpi=220)
        plt.close(fig)

    if not baseline_status.empty:
        counts = baseline_status.groupby(["dataset", "status"]).size().unstack(fill_value=0)
        fig, ax = plt.subplots(figsize=(10.0, 4.8))
        counts.plot(kind="bar", stacked=True, ax=ax, colormap="tab20")
        ax.set_ylabel("Metric count")
        ax.set_title("Non-BI baseline artifact status")
        ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0), fontsize=8)
        fig.tight_layout()
        fig.savefig(fig_dir / "baseline_status.pdf")
        fig.savefig(fig_dir / "baseline_status.png", dpi=220)
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--reusable-suite-rows",
        default="results/reusable_suite_rows/reusable_suite_success.csv",
    )
    parser.add_argument("--baseline-checkpoint", default="results/baseline_metrics/baseline_checkpoint.csv")
    parser.add_argument("--out-dir", default="results/bi_suite_8datasets")
    parser.add_argument("--extra-raw", action="append", default=[])
    parser.add_argument("--delta", type=float, default=1e-6)
    parser.add_argument("--n-classes", type=int, default=256)
    args = parser.parse_args()

    repo_root = Path.cwd()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    remote_dir = out_dir / "remote_bi_success"
    remote_dir.mkdir(parents=True, exist_ok=True)

    reusable_suite_rows = Path(args.reusable_suite_rows)
    seed_rows = load_reusable_suite_rows(
        reusable_suite_rows,
        delta=args.delta,
        n_classes=args.n_classes,
    )

    extra_paths = [Path(p) for p in args.extra_raw]
    extra_paths += sorted(remote_dir.glob("*.csv"))
    for path in extra_paths:
        for record in load_extra_raw_success(path):
            seed_rows.append(
                compute_bi_row(
                    dataset=record["dataset"],
                    scope_id=record["scope_id"],
                    seed=record.get("seed", 0),
                    n_te=int(record["n_te"]),
                    per_model_success=record["per_model_success"],
                    source_artifact=record.get("source_artifact", str(path)),
                    delta=args.delta,
                    n_classes=args.n_classes,
                )
            )

    seed_df = pd.DataFrame(seed_rows)
    seed_csv = out_dir / "bi_suite_seed_rows.csv"
    seed_df.to_csv(seed_csv, index=False, quoting=csv.QUOTE_MINIMAL)

    summary = summarize_seed_rows(seed_df)
    paper_table = summary[BI_TABLE_COLUMNS + [c for c in summary.columns if c not in BI_TABLE_COLUMNS]]
    paper_table.to_csv(out_dir / "bi_suite_dataset_summary.csv", index=False)
    paper_table[BI_TABLE_COLUMNS].to_csv(out_dir / "bi_suite_paper_table.csv", index=False)

    baseline_status = build_baseline_status(
        TARGET_DATASETS,
        Path(args.baseline_checkpoint),
        reusable_suite_rows,
    )
    baseline_status.to_csv(out_dir / "baseline_status.csv", index=False)

    scsca_status = build_scsca_status(repo_root)
    scsca_status.to_csv(out_dir / "scsca_sasca_status.csv", index=False)
    scsca_inventory = build_scsca_inventory(scsca_status)
    scsca_inventory.to_csv(out_dir / "scsca_sasca_inventory.csv", index=False)

    write_figures(out_dir, paper_table, baseline_status)
    write_markdown_summary(
        out_dir / "BI_SUITE_8DATASETS_SUMMARY.md",
        paper_table,
        baseline_status,
        scsca_status,
        scsca_inventory,
    )

    print(f"Wrote {seed_csv}")
    print(f"Wrote {out_dir / 'bi_suite_paper_table.csv'}")
    print(f"Wrote {out_dir / 'baseline_status.csv'}")
    print(f"Wrote {out_dir / 'scsca_sasca_status.csv'}")
    print(f"Wrote {out_dir / 'scsca_sasca_inventory.csv'}")


if __name__ == "__main__":
    main()
