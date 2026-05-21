#!/usr/bin/env python3
"""Train a compact 1D CNN profiled attacker for side-channel traces."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from model_training.common import add_dataset_args, load_standardized_split, save_metadata, save_pickle


class TraceCNN(nn.Module):
    def __init__(self, trace_len: int, n_classes: int = 256):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=11, padding=5),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(16),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 16, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


def make_loader(x: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    tensor_x = torch.from_numpy(x[:, None, :]).float()
    tensor_y = torch.from_numpy(y).long()
    return DataLoader(TensorDataset(tensor_x, tensor_y), batch_size=batch_size, shuffle=shuffle)


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred = model(x).argmax(dim=1)
            correct += int((pred == y).sum().item())
            total += int(y.numel())
    return correct / max(total, 1)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_dataset_args(parser)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    x_train, y_train, x_val, y_val, x_holdout, y_holdout, scaler = load_standardized_split(args)
    train_loader = make_loader(x_train, y_train, args.batch_size, shuffle=True)
    val_loader = make_loader(x_val, y_val, args.batch_size, shuffle=False)
    holdout_loader = make_loader(x_holdout, y_holdout, args.batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TraceCNN(trace_len=x_train.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = nn.CrossEntropyLoss()

    best_val = -1.0
    best_state = None
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            running_loss += float(loss.item()) * int(y.numel())
        val_acc = evaluate(model, val_loader, device)
        if val_acc > best_val:
            best_val = val_acc
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        print(f"epoch={epoch} loss={running_loss / max(len(y_train), 1):.6f} val_accuracy={val_acc:.6f}")

    if best_state is not None:
        model.load_state_dict(best_state)
    holdout_acc = evaluate(model, holdout_loader, device)
    out_dir = Path(args.out_dir) / args.dataset / "cnn"
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": model.state_dict(), "trace_len": x_train.shape[1]}, out_dir / "model.pt")
    save_pickle(out_dir / "scaler.pkl", scaler)
    save_metadata(out_dir / "metadata.json", args, {"best_val_accuracy": best_val, "holdout_accuracy": holdout_acc})
    print(f"holdout_accuracy={holdout_acc:.6f}")
    print(f"wrote {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
