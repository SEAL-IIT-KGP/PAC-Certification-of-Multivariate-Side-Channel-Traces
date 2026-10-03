#!/usr/bin/env python3
"""Recompute the EstraNet checkpoint-suite results from the released checkpoints.

Suites (11 checkpoints each, evaluated on the 100,000 attack traces):

- `ascad_random_key`: ASCAD random-key, 1,400-sample input (Table 7).
- `ches_ctf_2025`: CHES-CTF-2025, 3,000-sample window (Table 7).
- `ches_windows`: CHES-CTF-2025 windows of 100, 200, 500, 1,000, and 7,000
  samples (BI_suite,+ line of Figure 1; the 3,000-sample window is the
  `ches_ctf_2025` suite).

For the first two suites, the checkpoint with the largest attack success is
also evaluated for EPI and key-byte recovery (Tables 7 and 8), with the inverse
temperature fitted on 10,000 profiling traces (seed 20260908).

Inputs: the EstraNet sources (`paper_results/fetch_estranet.sh`), the released
checkpoints, the ASCAD random-key database `ASCAD.h5` (1,400-sample window),
and the CHES-CTF-2025 file `CHES_Challenge.h5`. See `paper_results/README.md`.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from paper_results.common import DEFAULT_OUT, DELTA, REPO, read_rows
from paper_results.matched_metrics import (
    BETA_MAX,
    PERMUTATION_SEED,
    calibration_indices,
    fit_beta,
    information_records,
    key_rank_campaigns,
    logsumexp,
)

ASCADR_CHECKPOINTS = "1,10,20,30,40,50,60,70,80,90,99"
CHES_CHECKPOINTS = "1,2,3,4,5,10,20,50,100,150,201"
# CHES-CTF-2025 window -> (checkpoint directory, pooling size used in training)
CHES_WINDOWS = {
    100: ("ches25_lx_w100", 5),
    200: ("ches25_lx_w200", 5),
    500: ("ches25_lx_w500", 10),
    1000: ("ches25_lx_w1000", 20),
    3000: ("ches25_lx_w3000", 20),
    7000: ("ches25_lx", 20),
}
# CHES-CTF-2025 preprocessing used for training and evaluation: notch
# denoising, shift of the attack mean toward the profiling traces, and
# per-sample standardization fitted on the profiling traces.
CHES_ENV = {
    "CHES25_PREPROCESS": "lucianoxu",
    "CHES25_LX_DENOISE": "1",
    "CHES25_LX_PROFILE_TRACES": "450000",
    "CHES25_LX_ATTACK_TRACES": "-1",
    "CHES25_STANDARDIZE": "1",
    "CHES25_STANDARDIZE_FIT_TRACES": "45000",
}
KEY_RECOVERY_BUDGETS = [100, 500, 1000, 2000]
TARGET_BYTE = {"ascad_random_key": 2, "ches_ctf_2025": 0}


def suite_counts(
    args: argparse.Namespace,
    *,
    dataset: str,
    kind: str,
    data_path: Path,
    checkpoint_dir: Path,
    indices: str,
    input_length: int,
    pool_size: int,
    beta_hat_2: int,
    scope: str,
    output: Path,
) -> None:
    if output.exists() and not args.force:
        print(f"{output.name}: already recomputed")
        return
    if not any(checkpoint_dir.glob("trans_long-*.index")):
        raise SystemExit(f"no EstraNet checkpoints in {checkpoint_dir}")
    command = [
        sys.executable,
        "-m",
        "bi_suite_8datasets.evaluate_estranet_bi_suite",
        "--project-root",
        str(args.estranet_root),
        "--dataset",
        kind,
        "--dataset-name",
        dataset,
        "--scope-id",
        scope,
        "--data-path",
        str(data_path),
        "--checkpoint-dir",
        str(checkpoint_dir),
        "--checkpoint-indices",
        indices,
        "--input-length",
        str(input_length),
        "--pool-size",
        str(pool_size),
        "--beta-hat-2",
        str(beta_hat_2),
        "--output-csv",
        str(output),
        "--source-note",
        checkpoint_dir.name,
    ]
    env = dict(os.environ)
    if kind == "CHES25":
        env.update(CHES_ENV)
    print("$ " + " ".join(command), flush=True)
    subprocess.check_call(command, cwd=REPO, env=env)


def selected_checkpoint(
    args: argparse.Namespace,
    dataset: str,
    counts_csv: Path,
    data_path: Path,
    checkpoint_dir: Path,
    input_length: int,
    pool_size: int,
    beta_hat_2: int,
) -> None:
    """EPI and key-byte recovery for the suite member with the largest attack success."""
    import h5py
    import tensorflow as tf

    out = args.out_dir / "selected_models" / dataset
    if (out / "summary.json").exists() and not args.force:
        print(f"{dataset}: selected checkpoint already evaluated")
        return
    out.mkdir(parents=True, exist_ok=True)
    started = time.time()
    rows = read_rows(counts_csv)
    best = max(rows, key=lambda row: int(row["successes"]))
    checkpoint_index = int(best["checkpoint_index"])
    ches = dataset == "ches_ctf_2025"
    if ches:
        os.environ.update(CHES_ENV)

    sys.path.insert(0, str(args.estranet_root))
    import data_utils  # type: ignore
    import data_utils_ches25  # type: ignore
    from transformer import Transformer  # type: ignore

    from bi_suite_8datasets.evaluate_estranet_bi_suite import build_model

    options = SimpleNamespace(
        n_layer=2,
        d_model=128,
        d_head=32,
        n_head=8,
        d_inner=256,
        n_head_softmax=8,
        d_head_softmax=16,
        dropout=0.05,
        conv_kernel_size=3,
        n_conv_layer=2,
        pool_size=pool_size,
        d_kernel_map=512,
        beta_hat_2=beta_hat_2,
        model_normalization="preLC",
        head_initialization="forward",
        softmax_attn=True,
        input_length=input_length,
    )
    model = build_model(options, Transformer)
    status = tf.train.Checkpoint(model=model).restore(str(checkpoint_dir / f"trans_long-{checkpoint_index}"))
    status.assert_existing_objects_matched()
    status.expect_partial()

    batch = 16

    @tf.function(input_signature=[tf.TensorSpec([batch, input_length], tf.float32)])
    def forward(x):
        return model(x, training=False)[0]

    def infer(traces: np.ndarray) -> np.ndarray:
        if len(traces) % batch:
            raise ValueError("the number of traces must be a multiple of the batch size")
        logits = np.concatenate(
            [
                forward(tf.convert_to_tensor(traces[i : i + batch], tf.float32)).numpy()
                for i in range(0, len(traces), batch)
            ]
        ).astype(np.float64)
        return logits - logsumexp(logits, axis=1, keepdims=True)

    # Calibration traces for the inverse temperature: profiling partition only.
    with h5py.File(data_path, "r") as corpus:
        profiling = corpus["Profiling_traces"]
        cal_idx = calibration_indices(len(profiling["traces"]))
        x_cal = profiling["traces"][cal_idx, :input_length]
        metadata = profiling["metadata"][cal_idx]
        if "labels" in profiling:
            y_cal = profiling["labels"][cal_idx].reshape(-1).astype(np.int64)
        elif metadata.dtype.names and "labels" in metadata.dtype.names:
            y_cal = metadata["labels"].reshape(-1).astype(np.int64)
        else:
            y_cal = data_utils_ches25.AES_SBOX[metadata["plaintext"][:, 0] ^ metadata["key"][:, 0]].astype(np.int64)
        if ches:
            stats = data_utils_ches25._compute_or_load_lucianoxu_stats(corpus, str(data_path), input_length)
            x_cal = data_utils_ches25._apply_lucianoxu_preprocess(x_cal, "train", stats, input_length)
        attack_metadata = corpus["Attack_traces"]["metadata"][:]
    beta, _ = fit_beta(infer(x_cal), y_cal, BETA_MAX)

    loader = data_utils_ches25 if ches else data_utils
    attack = loader.Dataset(str(data_path), "test", input_length, 0)
    labels = np.concatenate(attack.labels).reshape(-1).astype(np.int64)
    logp = infer(np.concatenate(attack.traces, axis=0))
    successes = int(np.sum(logp.argmax(axis=1) == labels))

    byte = TARGET_BYTE[dataset]
    keys = np.unique(attack_metadata["key"][:, byte])
    keyrank: list[dict[str, object]] = []
    if len(keys) == 1:
        sbox = np.asarray(data_utils_ches25.AES_SBOX)
        plaintexts = attack_metadata["plaintext"][:, byte].astype(np.int64)
        targets = sbox[plaintexts[:, None] ^ np.arange(256)[None, :]].astype(np.int64)
        if not np.array_equal(targets[:, int(keys[0])], labels):
            raise RuntimeError(f"{dataset}: labels do not match Sbox[plaintext ^ key]")
        keyrank, _, _ = key_rank_campaigns(logp, targets, int(keys[0]), budgets=KEY_RECOVERY_BUDGETS)

    result = {
        "dataset": dataset,
        "model": best["model_id"],
        "checkpoint_index": checkpoint_index,
        "selection": f"largest attack success in the M={len(rows)} checkpoint suite",
        "M": len(rows),
        "delta": DELTA,
        "n_te": int(len(labels)),
        "successes": successes,
        "suite_successes": int(best["successes"]),
        "count_difference": successes - int(best["successes"]),
        "p_obs": successes / len(labels),
        "n_calibration": int(len(cal_idx)),
        "calibration_split": "profiling traces only",
        "beta": beta,
        "raw_PI": float(np.mean(information_records(logp, labels, 1.0))),
        "EPI": float(np.mean(information_records(logp, labels, beta))),
        "keyrank": keyrank,
        "permutation_seed": PERMUTATION_SEED,
        "preprocessing": "notch denoising, attack-mean shift, profiling standardization" if ches else "raw traces",
        "tensorflow": tf.__version__,
        "numpy": np.__version__,
        "seconds": time.time() - started,
    }
    if args.save_predictions:
        np.savez_compressed(out / "predictions.npz", logp=logp, labels=labels)
    (out / "summary.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    sr = {row["traces"]: row["successes"] for row in keyrank}
    print(
        f"{dataset}: checkpoint {checkpoint_index} successes={successes}/{len(labels)} EPI={result['EPI']:.4f} "
        f"key recovery at 100/500/1000/2000 traces = {[sr.get(b) for b in KEY_RECOVERY_BUDGETS]} of 100",
        flush=True,
    )
    tf.keras.backend.clear_session()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--suites",
        nargs="+",
        default=["ascad_random_key", "ches_ctf_2025"],
        choices=["ascad_random_key", "ches_ctf_2025", "ches_windows"],
    )
    parser.add_argument("--estranet-root", type=Path, default=REPO / "external/EstraNet")
    parser.add_argument("--checkpoint-root", type=Path, default=REPO / "checkpoints/estranet")
    parser.add_argument("--ascad-random-h5", type=Path, default=REPO / "datasets/raw/ascad_random_key/ASCAD.h5")
    parser.add_argument("--ches-h5", type=Path, default=REPO / "datasets/raw/ches_ctf_2025/CHES_Challenge.h5")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT / "recomputed")
    parser.add_argument("--skip-selected", action="store_true", help="Only recompute the suite success counts.")
    parser.add_argument("--save-predictions", action="store_true")
    parser.add_argument("--force", action="store_true", help="Recompute results that already exist.")
    args = parser.parse_args()
    counts_dir = args.out_dir / "estranet"
    counts_dir.mkdir(parents=True, exist_ok=True)

    if "ascad_random_key" in args.suites:
        counts = counts_dir / "ascad_random_key_estranet_success.csv"
        suite_counts(
            args,
            dataset="ascad_random_key",
            kind="ASCAD",
            data_path=args.ascad_random_h5,
            checkpoint_dir=args.checkpoint_root / "ascadr",
            indices=ASCADR_CHECKPOINTS,
            input_length=1400,
            pool_size=10,
            beta_hat_2=50,
            scope="ascad_random_estranet_checkpoint_suite",
            output=counts,
        )
        if not args.skip_selected:
            selected_checkpoint(
                args, "ascad_random_key", counts, args.ascad_random_h5, args.checkpoint_root / "ascadr", 1400, 10, 50
            )

    windows = []
    if "ches_ctf_2025" in args.suites:
        windows.append(3000)
    if "ches_windows" in args.suites:
        windows += [100, 200, 500, 1000, 7000]
    for window in windows:
        directory, pool = CHES_WINDOWS[window]
        counts = counts_dir / f"ches_ctf_2025_estranet_w{window}_success.csv"
        suite_counts(
            args,
            dataset="ches_ctf_2025",
            kind="CHES25",
            data_path=args.ches_h5,
            checkpoint_dir=args.checkpoint_root / directory,
            indices=CHES_CHECKPOINTS,
            input_length=window,
            pool_size=pool,
            beta_hat_2=150,
            scope=f"ches_ctf_2025_estranet_w{window}_checkpoint_suite",
            output=counts,
        )
        if window == 3000:
            # The 3,000-sample suite is the CHES-CTF-2025 row of Table 7.
            shutil.copyfile(counts, counts_dir / "ches_ctf_2025_estranet_success.csv")
            if not args.skip_selected:
                selected_checkpoint(
                    args, "ches_ctf_2025", counts, args.ches_h5, args.checkpoint_root / directory, 3000, pool, 150
                )


if __name__ == "__main__":
    main()
