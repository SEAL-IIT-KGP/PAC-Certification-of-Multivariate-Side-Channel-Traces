#!/usr/bin/env python3
"""Compute exact score-local BI curves from relaxed attack-set margins."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import math
import os
import sys
import tempfile
import types
from pathlib import Path
from typing import Any

import numpy as np

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", tempfile.mkdtemp(prefix="bi_loc_mpl_"))
os.environ.setdefault("XDG_CACHE_HOME", tempfile.mkdtemp(prefix="bi_loc_font_"))


K = 256
CHANCE = 1.0 / K


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


def bi_from_p(p_success: float) -> float:
    return max(0.0, math.log2(1.0 + K * max(0.0, float(p_success) - CHANCE)))


def softmax(x: np.ndarray) -> np.ndarray:
    z = x.astype(np.float64)
    z -= np.max(z, axis=1, keepdims=True)
    exp = np.exp(z)
    return (exp / np.sum(exp, axis=1, keepdims=True)).astype(np.float32)


def normalized_probs(scores: np.ndarray) -> np.ndarray:
    scores = np.asarray(scores)
    if scores.ndim != 2:
        raise ValueError(f"expected 2-D score array, got {scores.shape}")
    row_sums = np.sum(scores, axis=1)
    if np.nanmin(scores) >= -1e-8 and np.allclose(row_sums, 1.0, atol=1e-3):
        probs = scores.astype(np.float32)
    else:
        probs = softmax(scores)
    probs = np.nan_to_num(probs, nan=1.0 / K, posinf=1.0 / K, neginf=0.0)
    row_sums = np.sum(probs, axis=1, keepdims=True)
    bad = row_sums[:, 0] <= 0
    if np.any(bad):
        probs[bad, :] = 1.0 / K
        row_sums = np.sum(probs, axis=1, keepdims=True)
    return probs / row_sums


def margins_from_probs(probs: np.ndarray, labels: np.ndarray) -> np.ndarray:
    labels = labels.astype(np.int64)[: len(probs)]
    true_scores = probs[np.arange(len(labels)), labels]
    masked = probs.copy()
    masked[np.arange(len(labels)), labels] = -np.inf
    return true_scores - np.max(masked, axis=1)


def margins_from_scores(scores: np.ndarray, labels: np.ndarray) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float64)
    if scores.ndim != 2:
        raise ValueError(f"expected 2-D score array, got {scores.shape}")
    labels = labels.astype(np.int64)[: len(scores)]
    true_scores = scores[np.arange(len(labels)), labels]
    masked = scores.copy()
    masked[np.arange(len(labels)), labels] = -np.inf
    return true_scores - np.max(masked, axis=1)


def curve_rows(
    *,
    dataset: str,
    architecture: str,
    center_id: str,
    scope: str,
    margins: np.ndarray,
    eps_grid: list[float],
    delta: float,
    suite_plus: float,
    source: str,
    model_path: str,
    m_count: int = 1,
    score_normalization: str = "probability_simplex_A=1",
    A: float = 1.0,
) -> list[dict[str, Any]]:
    rows = []
    n = int(len(margins))
    alpha = delta / max(1, m_count * len(eps_grid))
    for eps in eps_grid:
        successes = int(np.sum(margins >= -2.0 * float(eps)))
        p_hat = successes / n
        p_plus = kl_upper(successes, n, alpha)
        rows.append(
            {
                "dataset": dataset,
                "architecture": architecture,
                "center_id": center_id,
                "M": m_count,
                "scope": scope,
                "score_normalization": score_normalization,
                "A": float(A),
                "epsilon": float(eps),
                "n_te": n,
                "S_loc": successes,
                "p_hat_loc": p_hat,
                "p_loc_plus": p_plus,
                "BI_suite_plus": float(suite_plus),
                "BI_obs_loc": bi_from_p(p_hat),
                "BI_loc_plus": bi_from_p(p_plus),
                "nonvacuous": bool(p_plus < 1.0),
                "margin_min": float(np.min(margins)),
                "margin_median": float(np.median(margins)),
                "margin_q05": float(np.quantile(margins, 0.05)),
                "margin_q95": float(np.quantile(margins, 0.95)),
                "delta_effective": alpha,
                "source_artifact": source,
                "model_path": model_path,
            }
        )
    return rows


def adapt_input(model: Any, x: np.ndarray) -> np.ndarray:
    expected = tuple(d for d in model.input_shape[1:] if d is not None)
    out = x
    if len(expected) == 2 and expected[-1] == 1:
        t = int(expected[0])
        if out.shape[1] > t:
            out = out[:, :t]
        elif out.shape[1] < t:
            out = np.hstack([out, np.zeros((out.shape[0], t - out.shape[1]), dtype=out.dtype)])
        out = out[..., None]
    elif len(expected) == 1:
        t = int(expected[0])
        if out.shape[1] > t:
            out = out[:, :t]
        elif out.shape[1] < t:
            out = np.hstack([out, np.zeros((out.shape[0], t - out.shape[1]), dtype=out.dtype)])
    return out


def load_tches20_attack(tches_root: Path, dataset: str, max_attack: int | None) -> tuple[np.ndarray, np.ndarray]:
    src_dir = tches_root / "src"
    sys.path.insert(0, str(src_dir))
    if "tqdm" not in sys.modules:
        tqdm_mod = types.ModuleType("tqdm")

        def _tqdm(iterable=None, *args, **kwargs):
            return iterable if iterable is not None else []

        tqdm_mod.tqdm = _tqdm
        sys.modules["tqdm"] = tqdm_mod
        sys.modules.setdefault("tqdm.auto", tqdm_mod)
    from dataLoaders import load_ascad, load_aes_rd  # type: ignore

    data_dir = tches_root / "datasets"
    if dataset == "ascad_desync_0":
        _, _, x_attack, targets, key = load_ascad(str(data_dir / "ASCAD_dataset" / "ASCAD.h5"))
    elif dataset == "ascad_desync_50":
        _, _, x_attack, targets, key = load_ascad(str(data_dir / "ASCAD_dataset" / "ASCAD_desync50.h5"))
    elif dataset == "ascad_desync_100":
        _, _, x_attack, targets, key = load_ascad(str(data_dir / "ASCAD_dataset" / "ASCAD_desync100.h5"))
    elif dataset == "aes_rd":
        _, _, x_attack, targets, key = load_aes_rd(str(data_dir / "AES_RD_dataset") + "/")
    else:
        raise ValueError(f"unsupported TCHES20 attack dataset: {dataset}")
    labels = np.asarray(targets[:, int(key)], dtype=np.int64)
    x_attack = np.asarray(x_attack, dtype=np.float32)
    if max_attack and max_attack > 0:
        x_attack = x_attack[:max_attack]
        labels = labels[:max_attack]
    return x_attack, labels


def load_ascad_random_attack(data_path: Path, max_attack: int | None) -> tuple[np.ndarray, np.ndarray]:
    import h5py

    with h5py.File(data_path, "r") as h5:
        x_attack = h5["Attack_traces"]["traces"][()]
        labels = h5["Attack_traces"]["labels"][()]
    x_attack = np.asarray(x_attack, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.int64)
    if max_attack and max_attack > 0:
        x_attack = x_attack[:max_attack]
        labels = labels[:max_attack]
    return x_attack, labels


def keras_spec(
    *,
    dataset: str,
    architecture: str,
    center_id: str,
    model_path: Path,
    tches_root: Path,
    max_attack: int,
    suite_plus: float,
    data_path: Any = None,
    source_artifact: str = "TCHES20 pretrained model + attack split",
) -> dict[str, Any]:
    arch_scope = "noConv1 MLP" if architecture == "MLP" else "Zaid CNN"
    return {
        "kind": "keras",
        "dataset": dataset,
        "architecture": architecture,
        "center_id": center_id,
        "scope": f"score-local posterior ball around frozen {arch_scope}",
        "model_path": str(model_path),
        "tches20_root": str(tches_root),
        "data_path": str(data_path) if data_path is not None else "",
        "max_attack": max_attack,
        "suite_plus": suite_plus,
        "source_artifact": source_artifact,
    }


def run_keras_center(spec: dict[str, Any], eps_grid: list[float], delta: float) -> list[dict[str, Any]]:
    import tensorflow as tf

    model_path = Path(spec["model_path"])
    print(f"[keras] loading {model_path}", flush=True)
    model = tf.keras.models.load_model(str(model_path), compile=False)
    if spec["dataset"] == "ascad_random_key":
        x_attack, labels = load_ascad_random_attack(Path(spec["data_path"]), int(spec["max_attack"]))
    else:
        x_attack, labels = load_tches20_attack(Path(spec["tches20_root"]), spec["dataset"], int(spec["max_attack"]))
    x_model = adapt_input(model, x_attack)
    raw = model.predict(x_model, batch_size=int(spec.get("batch_size", 512)), verbose=1)
    probs = normalized_probs(raw[0] if isinstance(raw, (list, tuple)) else raw)
    margins = margins_from_probs(probs, labels)
    return curve_rows(
        dataset=spec["dataset"],
        architecture=spec["architecture"],
        center_id=spec["center_id"],
        scope=spec["scope"],
        margins=margins,
        eps_grid=eps_grid,
        delta=delta,
        suite_plus=float(spec["suite_plus"]),
        source=spec["source_artifact"],
        model_path=str(model_path),
    )


def ensure_jax_stub() -> None:
    if "jax" in sys.modules:
        return
    jax_stub = types.ModuleType("jax")

    def _xla_unavailable(*_args, **_kwargs):
        raise RuntimeError("JAX is unavailable in this audit environment")

    jax_stub.xla_computation = _xla_unavailable
    sys.modules["jax"] = jax_stub


def import_estranet(project_root: Path):
    sys.path.insert(0, str(project_root))
    import data_utils  # type: ignore
    from transformer import Transformer  # type: ignore

    return data_utils, Transformer


def labels_from_dataset(dataset: Any) -> np.ndarray:
    labels = np.concatenate([np.asarray(chunk).reshape(-1) for chunk in dataset.labels])
    return labels.astype(np.int64)


def build_transformer(spec: dict[str, Any], Transformer: Any):
    model = Transformer(
        n_layer=2,
        d_model=128,
        d_head=32,
        n_head=8,
        d_inner=256,
        n_head_softmax=8,
        d_head_softmax=16,
        dropout=0.05,
        n_classes=256,
        conv_kernel_size=3,
        n_conv_layer=2,
        pool_size=int(spec["pool_size"]),
        d_kernel_map=512,
        beta_hat_2=int(spec["beta_hat_2"]),
        model_normalization="preLC",
        head_initialization="forward",
        softmax_attn=True,
        output_attn=False,
    )
    import tensorflow as tf

    model(tf.zeros([1, int(spec["input_length"])], dtype=tf.float32), training=False)
    return model


def run_estranet_center(spec: dict[str, Any], eps_grid: list[float], delta: float) -> list[dict[str, Any]]:
    ensure_jax_stub()
    import tensorflow as tf

    project_root = Path(spec["project_root"])
    data_utils, Transformer = import_estranet(project_root)
    dataset = data_utils.Dataset(
        data_path=str(spec["data_path"]),
        split="test",
        input_length=int(spec["input_length"]),
        data_desync=0,
    )
    labels = labels_from_dataset(dataset)
    model = build_transformer(spec, Transformer)
    optimizer = tf.keras.optimizers.Adam(learning_rate=2.5e-4)
    checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)
    checkpoint_index = spec.get("checkpoint_index", "latest")
    if str(checkpoint_index).lower() == "latest":
        checkpoint_path = tf.train.latest_checkpoint(str(Path(spec["checkpoint_dir"])))
        if not checkpoint_path:
            raise FileNotFoundError(f"no checkpoint found in {spec['checkpoint_dir']}")
    else:
        checkpoint_path = str(Path(spec["checkpoint_dir"]) / f"trans_long-{int(checkpoint_index)}")
    checkpoint.restore(checkpoint_path, options=tf.train.CheckpointOptions(experimental_io_device="/job:localhost")).expect_partial()

    score_mode = str(spec.get("score_mode", "posterior_simplex"))
    score_normalization = "probability_simplex_A=1"
    A = 1.0
    if score_mode == "raw_logits_calibrated":
        calibration = data_utils.Dataset(
            data_path=str(spec["data_path"]),
            split=str(spec.get("calibration_split", "train")),
            input_length=int(spec["input_length"]),
            data_desync=0,
        )
        cal_records = calibration.GetTFRecords(int(spec.get("batch_size", 16)), training=False)
        cal_raw = model.predict(cal_records, verbose=1)
        cal_scores = np.asarray(cal_raw[0] if isinstance(cal_raw, (list, tuple)) else cal_raw, dtype=np.float64)
        A = float(np.nanmax(np.abs(cal_scores)))
        if not math.isfinite(A) or A <= 0.0:
            raise ValueError(f"invalid calibration logit bound A={A}")
        score_normalization = f"raw_logits_A=max_abs_{spec.get('calibration_split', 'train')}_split"

    tfrecords = dataset.GetTFRecords(int(spec.get("batch_size", 16)), training=False)
    raw = model.predict(tfrecords, verbose=1)
    scores = np.asarray(raw[0] if isinstance(raw, (list, tuple)) else raw)
    if score_mode == "raw_logits_calibrated":
        scores = scores.astype(np.float64) / A
        labels = labels[: len(scores)]
        margins = margins_from_scores(scores, labels)
    else:
        probs = normalized_probs(scores)
        labels = labels[: len(probs)]
        margins = margins_from_probs(probs, labels)
    return curve_rows(
        dataset=spec["dataset"],
        architecture=spec["architecture"],
        center_id=spec["center_id"],
        scope=spec["scope"],
        margins=margins,
        eps_grid=eps_grid,
        delta=delta,
        suite_plus=float(spec["suite_plus"]),
        source=spec["source_artifact"],
        model_path=checkpoint_path,
        score_normalization=score_normalization,
        A=A,
    )


def write_summary(rows: list[dict[str, Any]], out_dir: Path) -> None:
    table = []
    seen = set()
    for row in sorted(rows, key=lambda x: (x["dataset"], x["architecture"], x["center_id"], float(x["epsilon"]))):
        key = (row["dataset"], row["architecture"], row["center_id"])
        if key in seen or float(row["epsilon"]) <= 0.0:
            continue
        seen.add(key)
        table.append(row)
    if not table:
        for row in sorted(rows, key=lambda x: (x["dataset"], x["architecture"], x["center_id"], float(x["epsilon"]))):
            key = (row["dataset"], row["architecture"], row["center_id"])
            if key in seen:
                continue
            seen.add(key)
            table.append(row)
    lines = [
        "# BI_loc Relaxed-Margin Audit",
        "",
        "Rows use declared normalized scores and count relaxed successes with `rho >= -2 epsilon`.",
        "Keras rows use probability-simplex scores (`A=1`); Transformer rows use raw logits normalized by a train-split max-absolute logit bound.",
        "",
        "| dataset | arch | center | epsilon | n_te | S_loc | p_hat_loc | p_loc_plus | BI_loc_plus |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in table:
        lines.append(
            f"| {row['dataset']} | {row['architecture']} | {row['center_id']} | {float(row['epsilon']):.6g} | "
            f"{int(row['n_te'])} | {int(row['S_loc'])} | {float(row['p_hat_loc']):.6f} | "
            f"{float(row['p_loc_plus']):.6f} | {float(row['BI_loc_plus']):.3f} |"
        )
    lines.extend(["", "Full curve rows are in `bi_loc_radius_curves.csv`.", ""])
    (out_dir / "BI_LOC_MARGIN_AUDIT_SUMMARY.md").write_text("\n".join(lines))


def main() -> None:
    repo = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=repo / "results/bi_scope_certificates")
    parser.add_argument("--tches-root", type=Path, default=repo / "datasets/raw/tches20/TCHES20V3_CNN_SCA")
    parser.add_argument("--estranet-root", type=Path, default=repo / "external/EstraNet")
    parser.add_argument("--suite-csv", type=Path, default=repo / "results/bi_suite_8datasets/bi_suite_dataset_summary.csv")
    parser.add_argument("--estranet-summary", type=Path, default=repo / "results/ascad_estranet_bi_baseline/ascad_estranet_bi_summary.csv")
    parser.add_argument("--ascad-random-h5", type=Path, default=repo / "datasets/raw/ascad_random_key/ASCAD.h5")
    parser.add_argument("--aesrd-estranet-h5", type=Path, default=repo / "datasets/raw/aes_rd/AES_RD_estranet.h5")
    parser.add_argument("--aesrd-checkpoint-dir", type=Path, default=repo / "checkpoints/estranet/aes_rd")
    parser.add_argument("--delta", type=float, default=1e-6)
    parser.add_argument("--eps-grid", default="0,1e-5,3e-5,1e-4,3e-4,1e-3,3e-3,1e-2,3e-2")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    eps_grid = [float(x) for x in args.eps_grid.split(",") if x.strip()]

    tches_root = args.tches_root
    models = tches_root / "models" / "pretrained_models" / "models"
    suite_plus: dict[str, float] = {}
    with args.suite_csv.open(newline="") as f:
        for row in csv.DictReader(f):
            suite_plus[row["dataset"]] = float(row["BI_plus"])
    estranet_suite_plus: dict[str, float] = {}
    estranet_summary = args.estranet_summary
    if estranet_summary.exists():
        with estranet_summary.open(newline="") as f:
            for row in csv.DictReader(f):
                estranet_suite_plus[row["dataset"]] = float(row["BI_plus"])
    estranet_suite_plus.setdefault("ascad_desync_50", 1.5609413912724257)
    estranet_suite_plus.setdefault("ascad_desync_100", 1.6907041265379728)
    estranet_suite_plus.setdefault("ascad_random_key", suite_plus["ascad_random_key"])

    trained_models = args.output_dir / "trained_models"
    estranet_root = args.estranet_root
    ascad_random_h5 = args.ascad_random_h5
    ascad_random_mlp = trained_models / "ascad_random_key_mlp_seed0_best.hdf5"
    ascad_random_cnn = trained_models / "ascad_random_key_cnn_seed0_best.hdf5"
    aesrd_estranet_h5 = args.aesrd_estranet_h5
    aesrd_estranet_ckp = args.aesrd_checkpoint_dir

    specs: list[dict[str, Any]] = [
        keras_spec(
            dataset="ascad_desync_50",
            architecture="MLP",
            center_id="noConv1_ascad_desync_50_no_preprocessing_0",
            model_path=models / "noConv1_ascad_desync_50_no_preprocessing_0.hdf5",
            tches_root=tches_root,
            max_attack=25000,
            suite_plus=suite_plus["ascad_desync_50"],
        ),
        keras_spec(
            dataset="ascad_desync_50",
            architecture="CNN",
            center_id="zaid_ascad_desync_50_no_preprocessing_3",
            model_path=models / "zaid_ascad_desync_50_no_preprocessing_3.hdf5",
            tches_root=tches_root,
            max_attack=25000,
            suite_plus=suite_plus["ascad_desync_50"],
        ),
        keras_spec(
            dataset="ascad_desync_100",
            architecture="MLP",
            center_id="noConv1_ascad_desync_100_no_preprocessing_6",
            model_path=models / "noConv1_ascad_desync_100_no_preprocessing_6.hdf5",
            tches_root=tches_root,
            max_attack=25000,
            suite_plus=suite_plus["ascad_desync_100"],
        ),
        keras_spec(
            dataset="ascad_desync_100",
            architecture="CNN",
            center_id="zaid_ascad_desync_100_no_preprocessing_5",
            model_path=models / "zaid_ascad_desync_100_no_preprocessing_5.hdf5",
            tches_root=tches_root,
            max_attack=25000,
            suite_plus=suite_plus["ascad_desync_100"],
        ),
        keras_spec(
            dataset="aes_rd",
            architecture="MLP",
            center_id="noConv1_aes_rd_no_preprocessing_2",
            model_path=models / "noConv1_aes_rd_no_preprocessing_2.hdf5",
            tches_root=tches_root,
            max_attack=12500,
            suite_plus=suite_plus["aes_rd"],
        ),
        keras_spec(
            dataset="aes_rd",
            architecture="CNN",
            center_id="zaid_aes_rd_no_preprocessing_9",
            model_path=models / "zaid_aes_rd_no_preprocessing_9.hdf5",
            tches_root=tches_root,
            max_attack=12500,
            suite_plus=suite_plus["aes_rd"],
        ),
        {
            "kind": "estranet",
            "dataset": "ascad_desync_50",
            "architecture": "Transformer",
            "center_id": "ascad_desync_50_estranet_ckpt60",
            "scope": "score-local raw-logit ball around frozen EstraNet checkpoint",
            "project_root": str(estranet_root),
            "data_path": str(repo / "datasets/raw/ascad/ASCAD_data/ASCAD_databases/ASCAD_desync50.h5"),
            "checkpoint_dir": str(repo / "checkpoints/estranet/ascadf_desync50"),
            "checkpoint_index": 60,
            "input_length": 700,
            "pool_size": 5,
            "beta_hat_2": 150,
            "score_mode": "raw_logits_calibrated",
            "calibration_split": "train",
            "suite_plus": estranet_suite_plus["ascad_desync_50"],
            "source_artifact": "EstraNet ASCAD-desync-50 checkpoint suite",
        },
        {
            "kind": "estranet",
            "dataset": "ascad_desync_100",
            "architecture": "Transformer",
            "center_id": "ascad_desync_100_estranet_ckpt40",
            "scope": "score-local raw-logit ball around frozen EstraNet checkpoint",
            "project_root": str(estranet_root),
            "data_path": str(repo / "datasets/raw/ascad/ASCAD_data/ASCAD_databases/ASCAD_desync100.h5"),
            "checkpoint_dir": str(repo / "checkpoints/estranet/ascadf_desync100"),
            "checkpoint_index": 40,
            "input_length": 700,
            "pool_size": 5,
            "beta_hat_2": 150,
            "score_mode": "raw_logits_calibrated",
            "calibration_split": "train",
            "suite_plus": estranet_suite_plus["ascad_desync_100"],
            "source_artifact": "EstraNet ASCAD-desync-100 checkpoint suite",
        },
        {
            "kind": "estranet",
            "dataset": "ascad_random_key",
            "architecture": "Transformer",
            "center_id": "ascad_random_key_estranet_ckpt80",
            "scope": "score-local raw-logit ball around frozen EstraNet checkpoint",
            "project_root": str(estranet_root),
            "data_path": str(ascad_random_h5),
            "checkpoint_dir": str(repo / "checkpoints/estranet/ascadr"),
            "checkpoint_index": 80,
            "input_length": 1400,
            "pool_size": 10,
            "beta_hat_2": 50,
            "score_mode": "raw_logits_calibrated",
            "calibration_split": "train",
            "suite_plus": estranet_suite_plus["ascad_random_key"],
            "source_artifact": "EstraNet ASCAD-random checkpoint suite",
        },
    ]

    optional_specs: list[dict[str, Any]] = []
    if ascad_random_mlp.exists():
        optional_specs.append(
            keras_spec(
                dataset="ascad_random_key",
                architecture="MLP",
                center_id="ascad_random_key_mlp_seed0",
                model_path=ascad_random_mlp,
                tches_root=tches_root,
                data_path=ascad_random_h5,
                max_attack=100000,
                suite_plus=suite_plus["ascad_random_key"],
                source_artifact="locally trained ASCAD-random Keras MLP + attack split",
            )
        )
    if ascad_random_cnn.exists():
        optional_specs.append(
            keras_spec(
                dataset="ascad_random_key",
                architecture="CNN",
                center_id="ascad_random_key_cnn_seed0",
                model_path=ascad_random_cnn,
                tches_root=tches_root,
                data_path=ascad_random_h5,
                max_attack=100000,
                suite_plus=suite_plus["ascad_random_key"],
                source_artifact="locally trained ASCAD-random Keras CNN + attack split",
            )
        )
    if aesrd_estranet_h5.exists() and any(aesrd_estranet_ckp.glob("trans_long-*.index")):
        optional_specs.append(
            {
                "kind": "estranet",
                "dataset": "aes_rd",
                "architecture": "Transformer",
                "center_id": "aes_rd_estranet_latest",
                "scope": "score-local raw-logit ball around frozen AES-RD EstraNet checkpoint",
                "project_root": str(estranet_root),
                "data_path": str(aesrd_estranet_h5),
                "checkpoint_dir": str(aesrd_estranet_ckp),
                "checkpoint_index": "latest",
                "input_length": 3500,
                "pool_size": 10,
                "beta_hat_2": 150,
                "score_mode": "raw_logits_calibrated",
                "calibration_split": "train",
                "suite_plus": suite_plus["aes_rd"],
                "source_artifact": "locally trained EstraNet AES-RD checkpoint",
            }
        )
    specs.extend(optional_specs)

    rows: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    for spec in specs:
        try:
            if spec["kind"] == "keras":
                rows.extend(run_keras_center(spec, eps_grid, args.delta))
            elif spec["kind"] == "estranet":
                rows.extend(run_estranet_center(spec, eps_grid, args.delta))
            else:
                raise ValueError(f"unknown spec kind: {spec['kind']}")
        except Exception as exc:
            errors.append({"center_id": str(spec.get("center_id")), "error": repr(exc)})
            print(f"ERROR {spec.get('center_id')}: {exc!r}", flush=True)

    if rows:
        path = args.output_dir / "bi_loc_radius_curves.csv"
        with path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        write_summary(rows, args.output_dir)
        print(path)
    with (args.output_dir / "bi_loc_margin_errors.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["center_id", "error"])
        writer.writeheader()
        writer.writerows(errors)
    if errors:
        if not rows:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
