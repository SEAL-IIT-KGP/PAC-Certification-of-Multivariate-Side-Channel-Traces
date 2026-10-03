#!/usr/bin/env python3
"""Scan raw ASCAD traces for multi-byte leakage points.

The pre-cut ASCAD/TCHES benchmarks are mostly single-target artifacts.  For a
positive multi-target composition experiment we need a source trace that carries
several byte operations.  The raw ASCAD file has full 100k-sample traces and
16-byte metadata, so this scanner checks whether several AES S-box bytes expose
usable points of interest before launching heavier byte-wise attackers.

For each byte it reports top correlations against three leakage labels:

* unmasked_sbox_hw: HW(Sbox(P_i xor K_i)); this is the target used for key rank.
* masked_sbox_hw: HW(Sbox(P_i xor K_i) xor M_i); useful to locate masked S-box
  operations in the raw trace.
* mask_hw: HW(M_i); useful because successful ASCAD attackers often need both
  mask and masked-value evidence.
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
HW = np.array([bin(i).count("1") for i in range(256)], dtype=np.float32)


def write_csv(path: Path, rows: Sequence[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"no rows for {path}")
    fieldnames: list[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fieldnames.append(key)
                seen.add(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_label_matrix(
    plaintext: np.ndarray, key: np.ndarray, masks: np.ndarray
) -> tuple[np.ndarray, list[tuple[str, int]]]:
    cols = []
    names: list[tuple[str, int]] = []
    for byte in range(16):
        sbox = AES_SBOX[np.bitwise_xor(plaintext[:, byte], np.uint8(key[byte]))]
        cols.append(HW[sbox])
        names.append(("unmasked_sbox_hw", byte))
    for byte in range(16):
        sbox = AES_SBOX[np.bitwise_xor(plaintext[:, byte], np.uint8(key[byte]))]
        cols.append(HW[np.bitwise_xor(sbox, masks[:, byte])])
        names.append(("masked_sbox_hw", byte))
    for byte in range(16):
        cols.append(HW[masks[:, byte]])
        names.append(("mask_hw", byte))
    y = np.stack(cols, axis=1).astype(np.float32)
    y -= np.mean(y, axis=0, dtype=np.float64).astype(np.float32)[None, :]
    y_std = np.std(y, axis=0, dtype=np.float64).astype(np.float32)
    y_std[y_std <= 1e-7] = 1.0
    y /= y_std[None, :]
    return y, names


def update_top(
    top_abs: np.ndarray, top_corr: np.ndarray, top_pos: np.ndarray, abs_corr: np.ndarray, corr: np.ndarray, offset: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    k = top_abs.shape[0]
    combined_abs = np.concatenate([top_abs, abs_corr], axis=0)
    combined_corr = np.concatenate([top_corr, corr], axis=0)
    chunk_pos = np.arange(offset, offset + abs_corr.shape[0], dtype=np.int64)[:, None]
    chunk_pos = np.repeat(chunk_pos, abs_corr.shape[1], axis=1)
    combined_pos = np.concatenate([top_pos, chunk_pos], axis=0)
    keep = np.argpartition(combined_abs, -k, axis=0)[-k:, :]
    new_abs = np.take_along_axis(combined_abs, keep, axis=0)
    new_corr = np.take_along_axis(combined_corr, keep, axis=0)
    new_pos = np.take_along_axis(combined_pos, keep, axis=0)
    order = np.argsort(new_abs, axis=0)[::-1, :]
    return (
        np.take_along_axis(new_abs, order, axis=0),
        np.take_along_axis(new_corr, order, axis=0),
        np.take_along_axis(new_pos, order, axis=0),
    )


def main() -> None:
    repo = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw-h5", type=Path, default=repo / "datasets/raw/ascad_raw/ATMega8515_raw_traces.h5")
    parser.add_argument("--output-dir", type=Path, default=repo / "results/multibyte_composition/poi_scan")
    parser.add_argument("--samples", type=int, default=20000)
    parser.add_argument("--chunk-cols", type=int, default=2048)
    parser.add_argument("--top-k", type=int, default=20)
    args = parser.parse_args()

    t0 = time.time()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Scanning {args.raw_h5}", flush=True)
    with h5py.File(args.raw_h5, "r") as h5:
        traces = h5["traces"]
        n = min(int(args.samples), traces.shape[0])
        d = int(traces.shape[1])
        md = h5["metadata"][:n]
        key = md["key"][0].astype(np.uint8)
        plaintext = md["plaintext"].astype(np.uint8)
        masks = md["masks"].astype(np.uint8)
        y, names = build_label_matrix(plaintext, key, masks)
        top_abs = np.full((int(args.top_k), y.shape[1]), -np.inf, dtype=np.float32)
        top_corr = np.zeros_like(top_abs)
        top_pos = np.full((int(args.top_k), y.shape[1]), -1, dtype=np.int64)
        print(f"samples={n} trace_len={d} label_columns={y.shape[1]} key={key.tolist()}", flush=True)
        for start in range(0, d, int(args.chunk_cols)):
            end = min(start + int(args.chunk_cols), d)
            x = np.asarray(traces[:n, start:end], dtype=np.float32)
            x -= np.mean(x, axis=0, dtype=np.float64).astype(np.float32)[None, :]
            x_std = np.std(x, axis=0, dtype=np.float64).astype(np.float32)
            x_std[x_std <= 1e-7] = 1.0
            corr = (x.T @ y) / float(n)
            corr /= x_std[:, None]
            top_abs, top_corr, top_pos = update_top(top_abs, top_corr, top_pos, np.abs(corr), corr, start)
            if start == 0 or end == d or (start // int(args.chunk_cols)) % 10 == 0:
                print(f"processed columns {start}:{end}", flush=True)

    rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    for col, (label_model, byte) in enumerate(names):
        for rank in range(int(args.top_k)):
            rows.append(
                {
                    "label_model": label_model,
                    "target_byte": byte,
                    "rank": rank + 1,
                    "feature_index": int(top_pos[rank, col]),
                    "corr": float(top_corr[rank, col]),
                    "abs_corr": float(top_abs[rank, col]),
                }
            )
        summary_rows.append(
            {
                "label_model": label_model,
                "target_byte": byte,
                "max_abs_corr": float(top_abs[0, col]),
                "best_feature_index": int(top_pos[0, col]),
                "best_corr": float(top_corr[0, col]),
            }
        )
    write_csv(args.output_dir / "raw_ascad_poi_scan_topk.csv", rows)
    write_csv(args.output_dir / "raw_ascad_poi_scan_summary.csv", summary_rows)
    (args.output_dir / "metadata.json").write_text(
        json.dumps(
            {
                "raw_h5": str(args.raw_h5),
                "samples": n,
                "trace_len": d,
                "top_k": int(args.top_k),
                "chunk_cols": int(args.chunk_cols),
                "key": key.tolist(),
                "elapsed_sec": time.time() - t0,
                "finished_at": datetime.now().isoformat(timespec="seconds"),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    print(f"Done. Wrote {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
