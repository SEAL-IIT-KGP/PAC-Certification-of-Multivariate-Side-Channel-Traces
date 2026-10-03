#!/usr/bin/env python3
"""Evaluate the selected TCHES20 suite member of each benchmark on its attack traces.

For each of the six TCHES20 benchmarks, the selected member is the pretrained
model with the largest attack success in the 80-member suite. This script runs
inference only and reports, from the same predictions,

- the single-trace success count and its BI_suite,+ endpoint (M = 80 kept for the selection),
- EPI: the information of the temperature-rescaled model on the attack traces,
  with the inverse temperature fitted on at most 10,000 profiling traces
  (seed 20260908), and
- key-byte recovery over 100 shuffled orders of the fixed attack pool (Table 8).

AES-RD uses the first 12,500 attack traces, as in the suite evaluation.
Outputs go to `<out-dir>/selected_models/<dataset>/summary.json` and are picked
up by `paper_results.table7_bi_suite` and `paper_results.table8_key_recovery`.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import types
from pathlib import Path

import numpy as np

from bi_suite_8datasets.evaluate_tches20_pretrained_fixed_split import bi_bits, kl_endpoint
from paper_results.common import ARCHIVED, DEFAULT_OUT, DELTA, TCHES20_DATASETS, read_rows
from paper_results.matched_metrics import (
    BETA_MAX,
    PERMUTATION_SEED,
    calibration_indices,
    file_sha256,
    fit_beta,
    information_records,
    key_rank_campaigns,
    predict_logp,
)

MAX_ATTACK = {"aes_rd": 12500}


def selected_models(recomputed_dir: Path) -> dict[str, dict[str, object]]:
    """Suite maximum per dataset: from recomputed counts when available, else from the archived counts."""
    selected: dict[str, dict[str, object]] = {}
    rows_by_dataset: dict[str, list[dict[str, str]]] = {}
    for row in read_rows(ARCHIVED / "suite_success_counts.csv"):
        rows_by_dataset.setdefault(row["dataset"], []).append(row)
    for dataset in TCHES20_DATASETS:
        rows, source = rows_by_dataset[dataset], "archived counts"
        matches = sorted(recomputed_dir.rglob(f"{dataset}_attack_success.csv")) if recomputed_dir.exists() else []
        if matches:
            rows, source = read_rows(matches[0]), "recomputed counts"
        best = max(rows, key=lambda row: int(row["successes"]))
        selected[dataset] = {
            "model": best["model_id"],
            "n_te": int(best["n_te"]),
            "suite_successes": int(best["successes"]),
            "M": len(rows),
            "selection_source": source,
        }
    return selected


def load_benchmark(dataset: str, tches_root: Path):
    try:
        import tqdm  # noqa: F401
    except ImportError:
        stub = types.ModuleType("tqdm")
        stub.tqdm = lambda iterable=None, **_kwargs: iterable if iterable is not None else []
        sys.modules["tqdm"] = stub
    sys.path.insert(0, str(tches_root / "src"))
    from dataLoaders import load_aes_hd, load_aes_rd, load_ascad, load_dpav4  # type: ignore

    data = tches_root / "datasets"
    loaders = {
        "ascad_desync_0": lambda: load_ascad(str(data / "ASCAD_dataset" / "ASCAD.h5")),
        "ascad_desync_50": lambda: load_ascad(str(data / "ASCAD_dataset" / "ASCAD_desync50.h5")),
        "ascad_desync_100": lambda: load_ascad(str(data / "ASCAD_dataset" / "ASCAD_desync100.h5")),
        "aes_hd": lambda: load_aes_hd(str(data / "AES_HD_dataset") + os.sep),
        "aes_rd": lambda: load_aes_rd(str(data / "AES_RD_dataset") + os.sep),
        "dpav4": lambda: load_dpav4(str(data / "DPAv4_dataset") + os.sep),
    }
    return loaders[dataset]()


def evaluate(dataset: str, spec: dict[str, object], args: argparse.Namespace) -> dict[str, object]:
    import tensorflow as tf

    started = time.time()
    out = args.out_dir / "selected_models" / dataset
    out.mkdir(parents=True, exist_ok=True)
    x_profile, y_profile, x_attack, targets, key = load_benchmark(dataset, args.tches_root)
    y_profile = np.asarray(y_profile, dtype=np.int64).reshape(-1)
    targets = np.asarray(targets, dtype=np.int64)
    available = len(x_attack)
    limit = MAX_ATTACK.get(dataset)
    if limit:
        x_attack, targets = x_attack[:limit], targets[:limit]
    labels = targets[:, int(key)]
    if len(x_attack) != int(spec["n_te"]):
        raise RuntimeError(f"{dataset}: {len(x_attack)} attack traces loaded, suite counts use {spec['n_te']}")

    model_paths = list((args.tches_root / "models" / "pretrained_models").rglob(f"{spec['model']}.hdf5"))
    if len(model_paths) != 1:
        raise FileNotFoundError(f"{dataset}: expected one file named {spec['model']}.hdf5, found {len(model_paths)}")
    model = tf.keras.models.load_model(str(model_paths[0]), compile=False)

    # The inverse temperature is fitted on profiling traces only.
    cal_idx = calibration_indices(len(x_profile))
    cal_logp = predict_logp(model, np.asarray(x_profile)[cal_idx], args.batch_size)
    beta, _ = fit_beta(cal_logp, y_profile[cal_idx], BETA_MAX)
    logp = predict_logp(model, np.asarray(x_attack), args.batch_size)
    if not np.isfinite(logp).all():
        raise FloatingPointError(f"{dataset}: non-finite log-probabilities")

    successes = int(np.sum(logp.argmax(axis=1) == labels))
    n_te = len(labels)
    m_suite = int(spec["M"])
    upper = kl_endpoint(successes, n_te, np.log(2.0 * m_suite / args.delta) / n_te, upper=True)
    keyrank, ranks, budgets = key_rank_campaigns(logp, targets, int(key))
    result = {
        "dataset": dataset,
        "model": spec["model"],
        "selection": f"largest attack success in the M={m_suite} suite ({spec['selection_source']})",
        "model_sha256": file_sha256(model_paths[0]),
        "n_te": n_te,
        "available_attack_traces": available,
        "evaluation_split": "first 12,500 attack traces" if limit else "all attack traces",
        "successes": successes,
        "suite_successes": spec["suite_successes"],
        "count_difference": successes - int(spec["suite_successes"]),
        "M": m_suite,
        "delta": args.delta,
        "p_obs": successes / n_te,
        "BI_obs": bi_bits(successes / n_te),
        "p_plus": upper,
        "BI_plus": bi_bits(upper),
        "n_calibration": int(len(cal_idx)),
        "calibration_split": "profiling traces only",
        "beta": beta,
        "raw_PI": float(np.mean(information_records(logp, labels, 1.0))),
        "EPI": float(np.mean(information_records(logp, labels, beta))),
        "keyrank": keyrank,
        "permutation_seed": PERMUTATION_SEED,
        "tensorflow": tf.__version__,
        "numpy": np.__version__,
        "seconds": time.time() - started,
    }
    if args.save_predictions:
        np.savez_compressed(
            out / "predictions.npz",
            logp=logp,
            labels=labels,
            targets=targets,
            key=int(key),
            calibration_indices=cal_idx,
            calibration_logp=cal_logp,
            ranks=ranks,
            budgets=budgets,
        )
    (out / "summary.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    tf.keras.backend.clear_session()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tches-root", type=Path, default=Path("datasets/raw/tches20/TCHES20V3_CNN_SCA"))
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT / "recomputed")
    parser.add_argument("--datasets", nargs="+", default=TCHES20_DATASETS, choices=TCHES20_DATASETS)
    parser.add_argument("--delta", type=float, default=DELTA)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument(
        "--save-predictions",
        action="store_true",
        help="Also store the log-probability matrices (about 200 MB in total).",
    )
    args = parser.parse_args()

    specs = selected_models(args.out_dir)
    archived_epi = {row["dataset"]: row for row in read_rows(ARCHIVED / "selected_model_epi.csv")}
    for dataset in args.datasets:
        result = evaluate(dataset, specs[dataset], args)
        reference = archived_epi[dataset]
        sr = {row["traces"]: row["successes"] for row in result["keyrank"]}
        print(
            f"{dataset}: model={result['model']} successes={result['successes']}/{result['n_te']} "
            f"(suite count {result['suite_successes']}) EPI={result['EPI']:.4f} "
            f"(archived {float(reference['EPI']):.4f}) key recovery at 100/500/1000/2000 traces = "
            f"{sr.get(100, '-')}/{sr.get(500, '-')}/{sr.get(1000, '-')}/{sr.get(2000, '-')} of 100 "
            f"[{result['seconds']:.0f} s]",
            flush=True,
        )


if __name__ == "__main__":
    main()
