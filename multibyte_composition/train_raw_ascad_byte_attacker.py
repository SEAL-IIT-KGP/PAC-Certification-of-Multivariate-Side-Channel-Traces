#!/usr/bin/env python3
"""Train one raw-ASCAD two-share byte attacker.

The raw ASCAD scan showed strong target/mask points in the full 100k-sample
trace.  This trainer builds a compact byte-specific raw window by taking the
top POIs for unmasked S-box, masked S-box, and mask leakage, expands each POI
into a small local window, and trains a nonlinear MLP to predict the unmasked
S-box output Sbox(P_i xor K_i).

The output posterior schema is intentionally compatible with exact key-rank
post-processing: probabilities are over 256 S-box-output classes, and the NPZ
stores plaintext/key metadata for the held-out validation and attack splits.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Sequence

import h5py
import numpy as np


AES_SBOX = np.array(
    [
        0x63,
        0x7C,
        0x77,
        0x7B,
        0xF2,
        0x6B,
        0x6F,
        0xC5,
        0x30,
        0x01,
        0x67,
        0x2B,
        0xFE,
        0xD7,
        0xAB,
        0x76,
        0xCA,
        0x82,
        0xC9,
        0x7D,
        0xFA,
        0x59,
        0x47,
        0xF0,
        0xAD,
        0xD4,
        0xA2,
        0xAF,
        0x9C,
        0xA4,
        0x72,
        0xC0,
        0xB7,
        0xFD,
        0x93,
        0x26,
        0x36,
        0x3F,
        0xF7,
        0xCC,
        0x34,
        0xA5,
        0xE5,
        0xF1,
        0x71,
        0xD8,
        0x31,
        0x15,
        0x04,
        0xC7,
        0x23,
        0xC3,
        0x18,
        0x96,
        0x05,
        0x9A,
        0x07,
        0x12,
        0x80,
        0xE2,
        0xEB,
        0x27,
        0xB2,
        0x75,
        0x09,
        0x83,
        0x2C,
        0x1A,
        0x1B,
        0x6E,
        0x5A,
        0xA0,
        0x52,
        0x3B,
        0xD6,
        0xB3,
        0x29,
        0xE3,
        0x2F,
        0x84,
        0x53,
        0xD1,
        0x00,
        0xED,
        0x20,
        0xFC,
        0xB1,
        0x5B,
        0x6A,
        0xCB,
        0xBE,
        0x39,
        0x4A,
        0x4C,
        0x58,
        0xCF,
        0xD0,
        0xEF,
        0xAA,
        0xFB,
        0x43,
        0x4D,
        0x33,
        0x85,
        0x45,
        0xF9,
        0x02,
        0x7F,
        0x50,
        0x3C,
        0x9F,
        0xA8,
        0x51,
        0xA3,
        0x40,
        0x8F,
        0x92,
        0x9D,
        0x38,
        0xF5,
        0xBC,
        0xB6,
        0xDA,
        0x21,
        0x10,
        0xFF,
        0xF3,
        0xD2,
        0xCD,
        0x0C,
        0x13,
        0xEC,
        0x5F,
        0x97,
        0x44,
        0x17,
        0xC4,
        0xA7,
        0x7E,
        0x3D,
        0x64,
        0x5D,
        0x19,
        0x73,
        0x60,
        0x81,
        0x4F,
        0xDC,
        0x22,
        0x2A,
        0x90,
        0x88,
        0x46,
        0xEE,
        0xB8,
        0x14,
        0xDE,
        0x5E,
        0x0B,
        0xDB,
        0xE0,
        0x32,
        0x3A,
        0x0A,
        0x49,
        0x06,
        0x24,
        0x5C,
        0xC2,
        0xD3,
        0xAC,
        0x62,
        0x91,
        0x95,
        0xE4,
        0x79,
        0xE7,
        0xC8,
        0x37,
        0x6D,
        0x8D,
        0xD5,
        0x4E,
        0xA9,
        0x6C,
        0x56,
        0xF4,
        0xEA,
        0x65,
        0x7A,
        0xAE,
        0x08,
        0xBA,
        0x78,
        0x25,
        0x2E,
        0x1C,
        0xA6,
        0xB4,
        0xC6,
        0xE8,
        0xDD,
        0x74,
        0x1F,
        0x4B,
        0xBD,
        0x8B,
        0x8A,
        0x70,
        0x3E,
        0xB5,
        0x66,
        0x48,
        0x03,
        0xF6,
        0x0E,
        0x61,
        0x35,
        0x57,
        0xB9,
        0x86,
        0xC1,
        0x1D,
        0x9E,
        0xE1,
        0xF8,
        0x98,
        0x11,
        0x69,
        0xD9,
        0x8E,
        0x94,
        0x9B,
        0x1E,
        0x87,
        0xE9,
        0xCE,
        0x55,
        0x28,
        0xDF,
        0x8C,
        0xA1,
        0x89,
        0x0D,
        0xBF,
        0xE6,
        0x42,
        0x68,
        0x41,
        0x99,
        0x2D,
        0x0F,
        0xB0,
        0x54,
        0xBB,
        0x16,
    ],
    dtype=np.uint8,
)
KEYS = np.arange(256, dtype=np.uint8)


def parse_ints(text: str) -> list[int]:
    vals = [int(part.strip()) for part in text.replace(";", ",").split(",") if part.strip()]
    if not vals:
        raise argparse.ArgumentTypeError("expected at least one integer")
    return vals


def write_csv(path: Path, rows: Sequence[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"no rows for {path}")
    keys: list[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                keys.append(key)
                seen.add(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def labels_for_byte(plaintext: np.ndarray, key: np.ndarray, byte: int) -> np.ndarray:
    return AES_SBOX[np.bitwise_xor(plaintext[:, byte], np.uint8(key[byte]))].astype(np.int64)


def load_top_positions(scan_topk: Path, byte: int, top_per_model: int) -> list[int]:
    positions: list[int] = []
    with scan_topk.open(newline="") as f:
        reader = csv.DictReader(f)
        rows = [
            row
            for row in reader
            if int(row["target_byte"]) == byte
            and row["label_model"] in {"unmasked_sbox_hw", "masked_sbox_hw", "mask_hw"}
            and int(row["rank"]) <= int(top_per_model)
        ]
    rows.sort(key=lambda r: (r["label_model"], int(r["rank"])))
    for row in rows:
        positions.append(int(row["feature_index"]))
    if not positions:
        raise ValueError(f"no POI positions found for byte {byte} in {scan_topk}")
    return positions


def expand_positions(positions: Sequence[int], radius: int, trace_len: int) -> np.ndarray:
    out = set()
    for pos in positions:
        for j in range(max(0, int(pos) - int(radius)), min(trace_len, int(pos) + int(radius) + 1)):
            out.add(j)
    return np.asarray(sorted(out), dtype=np.int64)


def key_rank_from_probs(probs: np.ndarray, plaintext_byte: np.ndarray, true_key: int, count: int) -> tuple[int, int]:
    count = min(int(count), len(probs))
    labels = AES_SBOX[np.bitwise_xor(plaintext_byte[:count, None].astype(np.uint8), KEYS[None, :])].astype(np.int64)
    logp = np.log(probs[:count].astype(np.float64) + 1e-40)
    scores = np.take_along_axis(logp, labels, axis=1).sum(axis=0)
    top = int(np.argmax(scores))
    rank = int(np.sum(scores > float(scores[int(true_key)])))
    return rank, top


def predict_probs(model, x: np.ndarray, batch_size: int, device: str) -> np.ndarray:
    import torch
    import torch.nn.functional as F

    model.eval()
    chunks = []
    with torch.no_grad():
        for start in range(0, len(x), int(batch_size)):
            xb = torch.from_numpy(x[start : start + int(batch_size)]).to(device=device, dtype=torch.float32)
            probs = F.softmax(model(xb), dim=1).detach().cpu().numpy().astype(np.float32)
            chunks.append(probs)
    probs = np.concatenate(chunks, axis=0)
    probs = np.nan_to_num(probs, nan=1.0 / 256.0, posinf=1.0 / 256.0, neginf=0.0)
    row_sum = probs.sum(axis=1, keepdims=True)
    bad = row_sum[:, 0] <= 0
    if np.any(bad):
        probs[bad] = 1.0 / 256.0
        row_sum = probs.sum(axis=1, keepdims=True)
    return probs / row_sum


def main() -> None:
    repo = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw-h5", type=Path, default=repo / "datasets/raw/ascad_raw/ATMega8515_raw_traces.h5")
    parser.add_argument(
        "--scan-topk", type=Path, default=repo / "results/multibyte_composition/poi_scan/raw_ascad_poi_scan_topk.csv"
    )
    parser.add_argument("--output-dir", type=Path, default=repo / "results/multibyte_composition/byte_attackers")
    parser.add_argument("--target-byte", type=int, required=True)
    parser.add_argument("--train-count", type=int, default=40000)
    parser.add_argument("--val-count", type=int, default=10000)
    parser.add_argument("--attack-count", type=int, default=10000)
    parser.add_argument("--top-per-model", type=int, default=8)
    parser.add_argument("--window-radius", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=45)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--predict-batch-size", type=int, default=1024)
    parser.add_argument("--hidden", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=20260520)
    args = parser.parse_args()

    t0 = time.time()
    byte = int(args.target_byte)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    torch.manual_seed(int(args.seed) + byte)
    np.random.seed(int(args.seed) + byte)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"byte={byte} device={device}", flush=True)

    with h5py.File(args.raw_h5, "r") as h5:
        trace_len = int(h5["traces"].shape[1])
        n_total = int(h5["traces"].shape[0])
        need = int(args.train_count) + int(args.val_count) + int(args.attack_count)
        if need > n_total:
            raise ValueError(f"need {need} traces, have {n_total}")
        positions = load_top_positions(args.scan_topk, byte, args.top_per_model)
        features = expand_positions(positions, args.window_radius, trace_len)
        print(
            f"raw={args.raw_h5} n_total={n_total} trace_len={trace_len} "
            f"base_pois={len(positions)} expanded_features={len(features)}",
            flush=True,
        )
        traces = np.asarray(h5["traces"][:, features], dtype=np.float32)
        md = h5["metadata"][:]
        plaintext = md["plaintext"].astype(np.uint8)
        key = md["key"][0].astype(np.uint8)
    labels = labels_for_byte(plaintext, key, byte)

    rng = np.random.default_rng(int(args.seed))
    perm = rng.permutation(n_total)
    train_idx = perm[: args.train_count]
    val_idx = perm[args.train_count : args.train_count + args.val_count]
    attack_idx = perm[args.train_count + args.val_count : args.train_count + args.val_count + args.attack_count]

    x_train = traces[train_idx]
    y_train = labels[train_idx]
    x_val = traces[val_idx]
    y_val = labels[val_idx]
    x_attack = traces[attack_idx]
    y_attack = labels[attack_idx]
    p_val = plaintext[val_idx]
    p_attack = plaintext[attack_idx]
    del traces, plaintext, labels

    mean = np.mean(x_train, axis=0, dtype=np.float64).astype(np.float32)
    std = np.std(x_train, axis=0, dtype=np.float64).astype(np.float32)
    std[std <= 1e-6] = 1.0
    x_train = ((x_train - mean[None, :]) / std[None, :]).astype(np.float32)
    x_val = ((x_val - mean[None, :]) / std[None, :]).astype(np.float32)
    x_attack = ((x_attack - mean[None, :]) / std[None, :]).astype(np.float32)

    model = nn.Sequential(
        nn.Linear(x_train.shape[1], int(args.hidden)),
        nn.BatchNorm1d(int(args.hidden)),
        nn.ReLU(),
        nn.Dropout(float(args.dropout)),
        nn.Linear(int(args.hidden), int(args.hidden)),
        nn.BatchNorm1d(int(args.hidden)),
        nn.ReLU(),
        nn.Dropout(float(args.dropout)),
        nn.Linear(int(args.hidden), 256),
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()
    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train.astype(np.int64))),
        batch_size=int(args.batch_size),
        shuffle=True,
        num_workers=0,
    )
    x_val_t = torch.from_numpy(x_val).to(device=device, dtype=torch.float32)
    y_val_t = torch.from_numpy(y_val.astype(np.int64)).to(device=device)
    best_state = None
    best_val_acc = -1.0
    best_val_loss = float("inf")
    history: list[dict[str, object]] = []
    patience = 8
    stale = 0
    for epoch in range(int(args.epochs)):
        model.train()
        total_loss = 0.0
        total_ok = 0
        total_seen = 0
        for xb, yb in train_loader:
            xb = xb.to(device=device, dtype=torch.float32)
            yb = yb.to(device=device)
            opt.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()
            total_loss += float(loss.item()) * len(yb)
            total_ok += int((logits.argmax(dim=1) == yb).sum().item())
            total_seen += len(yb)
        model.eval()
        with torch.no_grad():
            val_logits = model(x_val_t)
            val_loss = float(loss_fn(val_logits, y_val_t).item())
            val_acc = float((val_logits.argmax(dim=1) == y_val_t).float().mean().item())
        row = {
            "epoch": epoch + 1,
            "train_loss": total_loss / total_seen,
            "train_acc": total_ok / total_seen,
            "val_loss": val_loss,
            "val_acc": val_acc,
        }
        history.append(row)
        print(row, flush=True)
        if val_acc > best_val_acc or (val_acc == best_val_acc and val_loss < best_val_loss):
            best_val_acc = val_acc
            best_val_loss = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                print(f"early stopping at epoch {epoch + 1}", flush=True)
                break
    if best_state is not None:
        model.load_state_dict(best_state)

    val_probs = predict_probs(model, x_val, args.predict_batch_size, device)
    attack_probs = predict_probs(model, x_attack, args.predict_batch_size, device)
    val_top1 = float(np.mean(np.argmax(val_probs, axis=1) == y_val))
    attack_top1 = float(np.mean(np.argmax(attack_probs, axis=1) == y_attack))
    val_rank, val_top_key = key_rank_from_probs(val_probs, p_val[:, byte], int(key[byte]), min(5000, len(val_probs)))
    attack_rank, attack_top_key = key_rank_from_probs(
        attack_probs, p_attack[:, byte], int(key[byte]), min(5000, len(attack_probs))
    )

    posterior_path = args.output_dir / f"byte_{byte:02d}_posteriors.npz"
    np.savez_compressed(
        posterior_path,
        probs=attack_probs.astype(np.float16),
        val_probs=val_probs.astype(np.float16),
        labels=y_attack.astype(np.uint8),
        val_labels=y_val.astype(np.uint8),
        plaintext=p_attack.astype(np.uint8),
        val_plaintext=p_val.astype(np.uint8),
        key=np.tile(key[None, :], (len(p_attack), 1)).astype(np.uint8),
        correct_key=np.array([int(key[byte])], dtype=np.uint8),
        target_byte=np.array([byte], dtype=np.int64),
        feature_indices=features.astype(np.int64),
        base_poi_positions=np.asarray(positions, dtype=np.int64),
    )
    model_path = args.output_dir / f"byte_{byte:02d}_twoshare_mlp.pt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "feature_indices": features,
            "mean": mean,
            "std": std,
            "target_byte": byte,
            "correct_key": int(key[byte]),
        },
        model_path,
    )
    write_csv(args.output_dir / f"byte_{byte:02d}_training_history.csv", history)
    summary = [
        {
            "target_byte": byte,
            "correct_key": int(key[byte]),
            "device": device,
            "features": int(len(features)),
            "base_pois": int(len(positions)),
            "val_top1": val_top1,
            "attack_top1": attack_top1,
            "val_rank_at_5000": val_rank,
            "val_top_key_at_5000": val_top_key,
            "attack_rank_at_5000": attack_rank,
            "attack_top_key_at_5000": attack_top_key,
            "selected_epoch_val_acc": best_val_acc,
            "selected_epoch_val_loss": best_val_loss,
            "posterior_npz": str(posterior_path),
            "model_path": str(model_path),
            "elapsed_sec": time.time() - t0,
            "finished_at": datetime.now().isoformat(timespec="seconds"),
            "history_json": json.dumps(history, sort_keys=True),
        }
    ]
    write_csv(args.output_dir / f"byte_{byte:02d}_model_summary.csv", summary)
    print(summary[0], flush=True)


if __name__ == "__main__":
    main()
