#!/usr/bin/env python3
"""Train a compact Transformer-style attacker for long side-channel traces."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from model_training.common import add_dataset_args, load_standardized_split, save_metadata, save_pickle


class TraceTransformer(nn.Module):
    def __init__(self, trace_len: int, patch_len: int = 20, d_model: int = 96, n_classes: int = 256):
        super().__init__()
        self.patch = nn.Conv1d(1, d_model, kernel_size=patch_len, stride=patch_len)
        n_tokens = max(1, trace_len // patch_len)
        self.positional = nn.Parameter(torch.zeros(1, n_tokens, d_model))
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=4,
            dim_feedforward=4 * d_model,
            dropout=0.1,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=2)
        self.classifier = nn.Linear(d_model, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.patch(x).transpose(1, 2)
        tokens = tokens + self.positional[:, : tokens.shape[1], :]
        encoded = self.encoder(tokens)
        return self.classifier(encoded.mean(dim=1))


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
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--patch-len", type=int, default=20)
    parser.add_argument("--d-model", type=int, default=96)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    x_train, y_train, x_val, y_val, x_holdout, y_holdout, scaler = load_standardized_split(args)
    train_loader = make_loader(x_train, y_train, args.batch_size, shuffle=True)
    val_loader = make_loader(x_val, y_val, args.batch_size, shuffle=False)
    holdout_loader = make_loader(x_holdout, y_holdout, args.batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TraceTransformer(trace_len=x_train.shape[1], patch_len=args.patch_len, d_model=args.d_model).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
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
    out_dir = Path(args.out_dir) / args.dataset / "transformer"
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "trace_len": x_train.shape[1],
            "patch_len": args.patch_len,
            "d_model": args.d_model,
        },
        out_dir / "model.pt",
    )
    save_pickle(out_dir / "scaler.pkl", scaler)
    save_metadata(out_dir / "metadata.json", args, {"best_val_accuracy": best_val, "holdout_accuracy": holdout_acc})
    print(f"holdout_accuracy={holdout_acc:.6f}")
    print(f"wrote {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
