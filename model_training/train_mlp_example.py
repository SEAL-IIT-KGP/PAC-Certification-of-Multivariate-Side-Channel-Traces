#!/usr/bin/env python3
"""Train a lightweight MLP profiled attacker for BI-suite experiments."""

from __future__ import annotations

import argparse
from pathlib import Path

from model_training.common import add_dataset_args, load_standardized_split, save_metadata, save_pickle


def parse_hidden_layers(value: str) -> tuple[int, ...]:
    return tuple(int(part) for part in value.split(",") if part.strip())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_dataset_args(parser)
    parser.add_argument("--hidden-layers", default="512,256")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--alpha", type=float, default=1e-4)
    args = parser.parse_args()

    from sklearn.metrics import accuracy_score
    from sklearn.neural_network import MLPClassifier

    x_train, y_train, x_val, y_val, x_holdout, y_holdout, scaler = load_standardized_split(args)
    model = MLPClassifier(
        hidden_layer_sizes=parse_hidden_layers(args.hidden_layers),
        activation="relu",
        solver="adam",
        alpha=args.alpha,
        batch_size=args.batch_size,
        learning_rate_init=args.learning_rate,
        max_iter=args.epochs,
        early_stopping=True,
        random_state=args.seed,
        verbose=True,
    )
    model.fit(x_train, y_train)

    val_acc = accuracy_score(y_val, model.predict(x_val))
    holdout_acc = accuracy_score(y_holdout, model.predict(x_holdout))
    out_dir = Path(args.out_dir) / args.dataset / "mlp"
    save_pickle(out_dir / "model.pkl", {"model": model, "scaler": scaler})
    save_metadata(out_dir / "metadata.json", args, {"val_accuracy": val_acc, "holdout_accuracy": holdout_acc})
    print(f"validation_accuracy={val_acc:.6f}")
    print(f"holdout_accuracy={holdout_acc:.6f}")
    print(f"wrote {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
