#!/usr/bin/env python3
"""Train the Keras MLP and CNN reference models of ASCAD random key (Figure 2).

Each model is trained on the profiling traces only: a seeded 90/10 split gives the
training and validation traces, and the epoch with the best validation accuracy is kept
as `ascad_random_key_<arch>_seed<seed>_best.hdf5`, the file read by
`profile_epsilon_calibration.py` and `compute_bi_loc_margin_curves.py`. The defaults are
the settings of the released models (seed 0, 30 epochs, batch size 256, learning rate 1e-3).
"""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

import h5py
import numpy as np
import tensorflow as tf


def set_seed(seed: int) -> None:
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def build_mlp(input_length: int) -> tf.keras.Model:
    inp = tf.keras.Input(shape=(input_length, 1), name="trace")
    x = tf.keras.layers.Flatten(name="flatten")(inp)
    x = tf.keras.layers.Dense(256, activation="selu", kernel_initializer="lecun_normal", name="fc1")(x)
    x = tf.keras.layers.Dense(256, activation="selu", kernel_initializer="lecun_normal", name="fc2")(x)
    x = tf.keras.layers.Dense(128, activation="selu", kernel_initializer="lecun_normal", name="fc3")(x)
    out = tf.keras.layers.Dense(256, activation="softmax", name="predictions")(x)
    return tf.keras.Model(inp, out, name="ascad_random_mlp")


def build_cnn(input_length: int) -> tf.keras.Model:
    inp = tf.keras.Input(shape=(input_length, 1), name="trace")
    x = inp
    for i, (filters, kernel, pool) in enumerate([(32, 11, 2), (64, 25, 10), (128, 3, 5)], start=1):
        x = tf.keras.layers.Conv1D(
            filters, kernel, padding="same", activation="selu", kernel_initializer="he_uniform", name=f"conv{i}"
        )(x)
        x = tf.keras.layers.BatchNormalization(name=f"bn{i}")(x)
        x = tf.keras.layers.AveragePooling1D(pool, strides=pool, name=f"pool{i}")(x)
    x = tf.keras.layers.Flatten(name="flatten")(x)
    x = tf.keras.layers.Dense(64, activation="selu", kernel_initializer="he_uniform", name="fc1")(x)
    x = tf.keras.layers.Dense(64, activation="selu", kernel_initializer="he_uniform", name="fc2")(x)
    out = tf.keras.layers.Dense(256, activation="softmax", name="predictions")(x)
    return tf.keras.Model(inp, out, name="ascad_random_cnn")


def main() -> None:
    repo = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--arch", choices=["MLP", "CNN"], required=True)
    parser.add_argument("--data-path", type=Path, default=repo / "datasets/raw/ascad_random_key/ASCAD.h5")
    parser.add_argument("--output-dir", type=Path, default=repo / "checkpoints/keras_ascad_random_key_retrained")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    with h5py.File(args.data_path, "r") as h5:
        x = np.asarray(h5["Profiling_traces"]["traces"][()], dtype=np.float32)[..., None]
        y = np.asarray(h5["Profiling_traces"]["labels"][()], dtype=np.int64)
    idx = np.random.default_rng(args.seed).permutation(len(y))
    n_val = max(1, int(round(len(y) * args.val_fraction)))
    val_idx, train_idx = idx[:n_val], idx[n_val:]
    input_length = int(x.shape[1])

    model = (build_mlp if args.arch == "MLP" else build_cnn)(input_length)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=args.learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    name = f"ascad_random_key_{args.arch.lower()}_seed{args.seed}"
    best_path = args.output_dir / f"{name}_best.hdf5"
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(str(best_path), monitor="val_accuracy", mode="max", save_best_only=True),
        tf.keras.callbacks.CSVLogger(str(args.output_dir / f"{name}_training.csv")),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-5, verbose=1),
    ]

    print(f"training {name}: train={len(train_idx)} val={len(val_idx)} input_length={input_length}")
    model.fit(
        x[train_idx],
        y[train_idx],
        validation_data=(x[val_idx], y[val_idx]),
        epochs=args.epochs,
        batch_size=args.batch_size,
        verbose=2,
        callbacks=callbacks,
    )

    metadata = {
        "dataset": "ascad_random_key",
        "architecture": args.arch,
        "model_name": name,
        "seed": args.seed,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "train_size": int(len(train_idx)),
        "val_size": int(len(val_idx)),
        "input_length": input_length,
    }
    (args.output_dir / f"{name}_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"saved_best_model={best_path}")


if __name__ == "__main__":
    main()
