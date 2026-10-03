#!/usr/bin/env python3
"""Convert the TCHES20 AES-RD NumPy arrays into an ASCAD-like HDF5 file for EstraNet.

The AES-RD Transformer reference model of Figure 2 was trained and evaluated on
the `feature_standardize` variant: every sample is standardized with the mean
and standard deviation of the profiling traces.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np


METADATA_DTYPE = np.dtype(
    [("plaintext", "u1", (16,)), ("key", "u1", (16,)), ("masks", "u1", (18,)), ("desync", "<u4", (1,))]
)


def make_metadata(plaintext: np.ndarray, key: np.ndarray, target_byte: int) -> np.ndarray:
    metadata = np.zeros(len(plaintext), dtype=METADATA_DTYPE)
    plaintext = plaintext.astype(np.uint8)
    key = key.astype(np.uint8)
    metadata["plaintext"] = plaintext
    metadata["key"] = np.broadcast_to(key, (len(plaintext), 16))
    # EstraNet's ASCAD evaluator reads metadata byte 2. Keep the HDF5 labels as
    # the training target, but make byte 2 metadata agree with the AES-RD target.
    metadata["plaintext"][:, 2] = plaintext[:, target_byte]
    metadata["key"][:, 2] = key[target_byte]
    metadata["masks"] = 0
    metadata["desync"] = 0
    return metadata


def transform_traces(
    profiling_traces: np.ndarray, attack_traces: np.ndarray, trace_mode: str
) -> tuple[np.ndarray, np.ndarray]:
    if trace_mode == "raw":
        return profiling_traces, attack_traces
    profiling = profiling_traces.astype(np.float32)
    attack = attack_traces.astype(np.float32)
    if trace_mode == "feature_standardize":
        mean = profiling.mean(axis=0, keepdims=True)
        std = profiling.std(axis=0, keepdims=True)
        std = np.where(std < 1e-6, 1.0, std)
        return (profiling - mean) / std, (attack - mean) / std
    if trace_mode == "global_standardize":
        mean = float(profiling.mean())
        std = float(profiling.std())
        if std < 1e-6:
            std = 1.0
        return (profiling - mean) / std, (attack - mean) / std
    raise ValueError(f"unknown trace mode: {trace_mode}")


def write_split(
    h5: h5py.File,
    name: str,
    traces: np.ndarray,
    labels: np.ndarray,
    plaintext: np.ndarray,
    key: np.ndarray,
    target_byte: int,
) -> None:
    group = h5.create_group(name)
    trace_data = traces.astype(np.int16) if np.issubdtype(traces.dtype, np.integer) else traces.astype(np.float32)
    group.create_dataset("traces", data=trace_data, chunks=True)
    group.create_dataset("labels", data=labels.astype(np.int64), chunks=True)
    group.create_dataset("metadata", data=make_metadata(plaintext, key, target_byte), chunks=True)


def main() -> None:
    repo = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--aesrd-root", type=Path, default=repo / "datasets/raw/tches20/TCHES20V3_CNN_SCA/datasets/AES_RD_dataset"
    )
    parser.add_argument(
        "--output", type=Path, default=repo / "datasets/raw/aes_rd/AES_RD_estranet_feature_standardized.h5"
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--trace-mode", choices=["raw", "feature_standardize", "global_standardize"], default="feature_standardize"
    )
    parser.add_argument("--target-byte", type=int, default=0)
    args = parser.parse_args()

    if args.output.exists() and not args.force:
        print(f"exists={args.output}")
        return

    args.output.parent.mkdir(parents=True, exist_ok=True)
    key = np.load(args.aesrd_root / "key.npy").astype(np.uint8)
    profiling_traces = np.load(args.aesrd_root / "profiling_traces_AES_RD.npy")
    profiling_labels = np.load(args.aesrd_root / "profiling_labels_AES_RD.npy")
    profiling_plaintext = np.load(args.aesrd_root / "profiling_plaintext_AES_RD.npy")
    attack_traces = np.load(args.aesrd_root / "attack_traces_AES_RD.npy")
    attack_labels = np.load(args.aesrd_root / "attack_labels_AES_RD.npy")
    attack_plaintext = np.load(args.aesrd_root / "attack_plaintext_AES_RD.npy")
    profiling_traces, attack_traces = transform_traces(profiling_traces, attack_traces, args.trace_mode)

    tmp = args.output.with_suffix(args.output.suffix + ".tmp")
    if tmp.exists():
        tmp.unlink()
    with h5py.File(tmp, "w") as h5:
        h5.attrs["trace_mode"] = args.trace_mode
        h5.attrs["target_byte"] = args.target_byte
        write_split(
            h5, "Profiling_traces", profiling_traces, profiling_labels, profiling_plaintext, key, args.target_byte
        )
        write_split(h5, "Attack_traces", attack_traces, attack_labels, attack_plaintext, key, args.target_byte)
    tmp.replace(args.output)
    print(f"wrote={args.output}")
    print(f"profile={profiling_traces.shape} attack={attack_traces.shape}")
    print(f"trace_mode={args.trace_mode} target_byte={args.target_byte}")


if __name__ == "__main__":
    main()
