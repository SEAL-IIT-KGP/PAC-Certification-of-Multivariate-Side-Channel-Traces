#!/usr/bin/env python3
"""
Attacker Scope Comparison (AsiaCRYPT 2026 appendix diagnostic)

Compares BI certification across four hierarchical attack scopes:

1. SCOPE 1 - Linear (M ≈ 5-8):
   - Linear Discriminant Analysis (1)
   - Logistic Regression variants (3)
   - Gaussian Naive Bayes (1)
   - Ridge Classifier (1)

2. SCOPE 2 - MLP-bottleneck (M ≈ 15-20):
   - All of Scope 1
   - MLP with architectures: (64,), (128,), (256,), (128,64), (256,128), (128,64,32)

3. SCOPE 3 - CNN τ-local (M ≈ 25-30):
   - All of Scope 2
   - CNN with R_L=9, R_L=17, R_L=33 (3 channel variants each)

4. SCOPE 4 - CNN-wide (M ≈ 35-40):
   - All of Scope 3
   - CNN with R_L=65, R_L=129 (3 channel variants each)

Key insight: Each scope is a superset of the previous. By monotonicity of BI
(Proposition 4), BI(Scope_i) <= BI(Scope_j) for i < j. The experiment validates:
- BI is monotonically non-decreasing across scopes
- τ-local scope captures most achievable leakage at lower cost
- Suite correction penalty grows only as √ln(M), not linearly

Usage examples:

  # Single dataset, all scopes (default)
  python -m attacker_scope_diagnostics.attacker_scope_diagnostic \\
      --dataset ascad_desync_0 \\
      --tches20-src-dir ./datasets

  # Multiple datasets, specific scopes
  python -m attacker_scope_diagnostics.attacker_scope_diagnostic \\
      --datasets ascad_desync_0 aes_rd \\
      --scopes 1,3,4 \\
      --tches20-src-dir ./datasets \\
      --n-seeds 3

  # Full run with all parameters
  python -m attacker_scope_diagnostics.attacker_scope_diagnostic \\
      --datasets all \\
      --scopes 1,2,3,4 \\
      --tches20-src-dir ./datasets \\
      --output-dir results/attacker_scope_diagnostic \\
      --n-seeds 5 \\
      --holdout-size 25000
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field, asdict
import warnings
import json
import argparse
import signal
import sys
import os
from datetime import datetime
from tqdm import tqdm
import traceback
import math
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

# Sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.exceptions import ConvergenceWarning

# PyTorch for CNN
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import TensorDataset, DataLoader
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("[WARN] PyTorch not available — CNN models disabled")

from core.data_loader import load_dataset, generate_synthetic_data
from core.baseline_metrics import compute_pi, compute_hi

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore', category=ConvergenceWarning)
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

# Lock for thread-safe numpy RNG access in train_torch_model()
_np_rng_lock = threading.Lock()


# ===================================================================
# Constants
# ===================================================================

ALL_DATASETS = [
    'ascad_desync_0',
    'ascad_desync_50',
    'ascad_desync_100',
    'aes_hd',
    'aes_rd',
    'dpav4',
]

SCOPE_NAMES = {
    1: 'Linear',
    2: 'MLP-bottleneck',
    3: 'CNN τ-local',
    4: 'CNN-wide',
}

SCOPE_DESCRIPTIONS = {
    1: 'LDA, LogReg, GNB, Ridge',
    2: 'Scope 1 + MLP variants',
    3: 'Scope 2 + CNNs (R_L=9,17,33)',
    4: 'Scope 3 + CNNs (R_L=65,129)',
}


# ===================================================================
# ControlledReceptiveFieldCNN
# ===================================================================

class ControlledReceptiveFieldCNN(nn.Module):
    """
    1D CNN with controlled receptive field via stacked Conv1d(k=3, pad=1) layers.

    Architecture:
    - Input: (batch, seq_len)
    - Reshape to (batch, 1, seq_len)
    - Stack N conv layers with k=3, padding=1 to achieve R_L = 1 + 2N
    - BatchNorm + SELU after each conv
    - Global average pooling
    - FC layer to output

    Args:
        seq_len: Input sequence length
        n_classes: Number of output classes
        receptive_field: Desired receptive field size (1, 3, 5, 9, 17, 33, 65, 129, ...)
        n_channels: Base number of channels (adjusted per layer to keep params ~target)
        target_params: Target parameter count (default 50000)
    """
    def __init__(
        self,
        seq_len: int,
        n_classes: int = 256,
        receptive_field: int = 9,
        n_channels: int = 32,
        target_params: int = 50000,
    ):
        super().__init__()

        self.seq_len = seq_len
        self.n_classes = n_classes
        self.receptive_field = receptive_field

        # Compute number of conv layers: R_L = 1 + 2 * num_layers
        # Solve: receptive_field = 1 + 2 * num_layers
        num_layers = max(1, (receptive_field - 1) // 2)

        # Build conv stack
        layers = []
        in_channels = 1
        out_channels = n_channels

        for i in range(num_layers):
            layers.append(nn.Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                padding=1,
                bias=False
            ))
            layers.append(nn.BatchNorm1d(out_channels))
            layers.append(nn.SELU(inplace=False))
            in_channels = out_channels

        self.conv_stack = nn.Sequential(*layers)

        # Global average pooling (output: (batch, out_channels))
        self.pool = nn.AdaptiveAvgPool1d(1)

        # FC classifier
        self.fc = nn.Linear(in_channels, n_classes)

    def forward(self, x):
        # x: (batch, seq_len)
        x = x.unsqueeze(1)  # (batch, 1, seq_len)
        x = self.conv_stack(x)  # (batch, channels, seq_len)
        x = self.pool(x)  # (batch, channels, 1)
        x = x.squeeze(-1)  # (batch, channels)
        x = self.fc(x)  # (batch, n_classes)
        return x


def compute_mlp_pi_ti(X_train, y_train, X_val, y_val, n_classes=256):
    """MLP-based Perceived/Training Information."""
    try:
        from sklearn.neural_network import MLPClassifier
        from sklearn.preprocessing import StandardScaler
        import math

        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_train)
        X_vl = scaler.transform(X_val)

        clf = MLPClassifier(
            hidden_layer_sizes=(64, 32, 16, 4),
            activation='relu', max_iter=200,
            early_stopping=True, validation_fraction=0.15,
            batch_size=256,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            clf.fit(X_tr, y_train)

        log_proba_val = clf.predict_log_proba(X_vl)
        log_proba_train = clf.predict_log_proba(X_tr)
        classes = clf.classes_

        score_val, score_train, n_active = 0.0, 0.0, 0
        for x in classes:
            idx_val = np.where(y_val == x)[0]
            idx_train = np.where(y_train == x)[0]
            class_col = np.where(classes == x)[0]
            if len(class_col) == 0: continue
            class_col = class_col[0]
            if len(idx_val) > 0:
                score_val += np.mean(log_proba_val[idx_val, class_col])
                n_active += 1
            if len(idx_train) > 0:
                score_train += np.mean(log_proba_train[idx_train, class_col])

        K = n_classes
        if n_active > 0:
            score_val /= n_active
            score_train /= n_active
        PI = np.log2(K) + score_val * np.log2(math.e)
        TI = np.log2(K) + score_train * np.log2(math.e)
        return PI, TI
    except Exception:
        return np.nan, np.nan


def count_cnn_params(model: nn.Module) -> int:
    """Count total parameters in a PyTorch model."""
    return sum(p.numel() for p in model.parameters())


def adjust_cnn_channels_for_params(
    seq_len: int,
    receptive_field: int,
    n_classes: int,
    target_params: int = 50000,
) -> int:
    """
    Estimate appropriate n_channels to achieve target_params.

    Simple heuristic: try a few values and pick the closest.
    """
    candidates = [8, 16, 32, 48, 64, 96, 128]
    best_n_ch = 32
    best_diff = float('inf')

    for n_ch in candidates:
        model = ControlledReceptiveFieldCNN(
            seq_len=seq_len,
            n_classes=n_classes,
            receptive_field=receptive_field,
            n_channels=n_ch,
        )
        params = count_cnn_params(model)
        diff = abs(params - target_params)
        if diff < best_diff:
            best_diff = diff
            best_n_ch = n_ch

    return best_n_ch


# ===================================================================
# Model training utilities
# ===================================================================

def train_torch_model(
    model: nn.Module,
    X_train: np.ndarray,
    y_train: np.ndarray,
    epochs: int = 30,
    batch_size: int = 256,
    lr: float = 1e-3,
    patience: int = 5,
    device: str = 'cpu',
) -> nn.Module:
    """
    Train a PyTorch model with early stopping on validation loss.

    Args:
        model: PyTorch nn.Module
        X_train: Training features (n_samples, seq_len)
        y_train: Training labels
        epochs: Max epochs
        batch_size: Batch size
        lr: Learning rate
        patience: Early stopping patience
        device: 'cpu' or 'cuda'

    Returns:
        Trained model in eval mode
    """
    model = model.to(device)

    # Split into train/val (thread-safe RNG access)
    n_train = len(X_train)
    val_split = 0.15
    n_val = max(1, int(val_split * n_train))
    n_train_actual = n_train - n_val

    with _np_rng_lock:
        idx = np.random.permutation(n_train)
    idx_tr = idx[:n_train_actual]
    idx_val = idx[n_train_actual:]

    X_tr = X_train[idx_tr]
    y_tr = y_train[idx_tr]
    X_v = X_train[idx_val]
    y_v = y_train[idx_val]

    X_tr_t = torch.FloatTensor(X_tr).to(device)
    y_tr_t = torch.LongTensor(y_tr).to(device)
    X_v_t = torch.FloatTensor(X_v).to(device)
    y_v_t = torch.LongTensor(y_v).to(device)

    use_pin = (str(device) != 'cpu')
    train_ds = TensorDataset(X_tr_t, y_tr_t)
    train_dl = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        pin_memory=use_pin, num_workers=2 if use_pin else 0,
    )

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    best_val_loss = float('inf')
    no_improve = 0

    model.train()
    for epoch in range(epochs):
        # Train
        for xb, yb in train_dl:
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()

        # Validate
        model.eval()
        with torch.no_grad():
            val_out = model(X_v_t)
            val_loss = criterion(val_out, y_v_t).item()
        model.train()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= patience:
            break

    model.eval()
    return model


def predict_torch(model: nn.Module, X: np.ndarray, device: str = 'cpu') -> np.ndarray:
    """Get predictions from a PyTorch model."""
    model.eval()
    with torch.no_grad():
        X_t = torch.FloatTensor(X).to(device)
        out = model(X_t)
        preds = out.argmax(dim=1).cpu().numpy()
    return preds


# ===================================================================
# Scope builders
# ===================================================================

def build_linear_suite(n_features: int) -> List[Tuple[str, Any]]:
    """Scope 1: Linear models (M ≈ 5-8)."""
    suite = []

    # LDA
    suite.append(('LDA', LinearDiscriminantAnalysis()))

    # Logistic Regression
    for C in [0.1, 1.0, 10.0]:
        suite.append((f'LogReg_C{C}', LogisticRegression(
            solver='lbfgs', max_iter=500, C=C, n_jobs=-1
        )))

    # Gaussian Naive Bayes
    suite.append(('GNB', GaussianNB()))

    # Ridge Classifier
    suite.append(('Ridge', RidgeClassifier(alpha=1.0)))

    return suite


def build_mlp_suite(n_features: int) -> List[Tuple[str, Any]]:
    """Scope 2: MLP models to add to Scope 1 (adds ~8-12)."""
    architectures = [
        (64,),
        (128,),
        (256,),
        (128, 64),
        (256, 128),
        (128, 64, 32),
    ]

    suite = []
    for arch in architectures:
        name = 'MLP_' + '_'.join(str(a) for a in arch)
        suite.append((name, MLPClassifier(
            hidden_layer_sizes=arch,
            max_iter=300,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=10,
            batch_size=256,
            random_state=42,
        )))

    return suite


def build_cnn_local_suite(
    n_features: int,
    n_classes: int = 256,
    target_cnn_params: int = 50000,
    device: str = 'cpu',
) -> List[Tuple[str, nn.Module]]:
    """
    Scope 3: CNN models with τ-local receptive fields to add to Scope 2.

    Adds ~6-9 models:
    - R_L=9 with 3 channel variants
    - R_L=17 with 3 channel variants
    - R_L=33 with 3 channel variants
    """
    if not HAS_TORCH or n_features < 3:
        return []

    suite = []
    receptive_fields = [9, 17, 33]
    channel_multipliers = [0.8, 1.0, 1.2]

    for r_l in receptive_fields:
        base_ch = adjust_cnn_channels_for_params(
            seq_len=n_features,
            receptive_field=r_l,
            n_classes=n_classes,
            target_params=target_cnn_params,
        )

        for mult in channel_multipliers:
            n_ch = max(8, int(base_ch * mult))
            name = f'CNN_RL{r_l}_ch{n_ch}'
            model = ControlledReceptiveFieldCNN(
                seq_len=n_features,
                n_classes=n_classes,
                receptive_field=r_l,
                n_channels=n_ch,
            )
            suite.append((name, model))

    return suite


def build_cnn_wide_suite(
    n_features: int,
    n_classes: int = 256,
    target_cnn_params: int = 50000,
    device: str = 'cpu',
) -> List[Tuple[str, nn.Module]]:
    """
    Scope 4: Wider CNN models to add to Scope 3.

    Adds ~6 models:
    - R_L=65 with 3 channel variants
    - R_L=129 with 3 channel variants
    """
    if not HAS_TORCH or n_features < 3:
        return []

    suite = []
    receptive_fields = [65, 129]
    channel_multipliers = [0.8, 1.0, 1.2]

    for r_l in receptive_fields:
        # Skip if receptive field > sequence length
        if r_l > n_features:
            continue

        base_ch = adjust_cnn_channels_for_params(
            seq_len=n_features,
            receptive_field=r_l,
            n_classes=n_classes,
            target_params=target_cnn_params,
        )

        for mult in channel_multipliers:
            n_ch = max(8, int(base_ch * mult))
            name = f'CNN_RL{r_l}_ch{n_ch}'
            model = ControlledReceptiveFieldCNN(
                seq_len=n_features,
                n_classes=n_classes,
                receptive_field=r_l,
                n_channels=n_ch,
            )
            suite.append((name, model))

    return suite


def build_scope(
    level: int,
    n_features: int,
    n_classes: int = 256,
    target_cnn_params: int = 50000,
    device: str = 'cpu',
) -> List[Tuple[str, Any]]:
    """
    Build attack suite for a given scope level (1-4).

    Each level is a superset of the previous:
    - Level 1: Linear models only
    - Level 2: Linear + MLP variants
    - Level 3: Linear + MLP + CNN τ-local
    - Level 4: Linear + MLP + CNN τ-local + CNN wide

    Args:
        level: Scope level 1-4
        n_features: Dimensionality of input features
        n_classes: Number of classes
        target_cnn_params: Target parameter count for CNN models
        device: Device for PyTorch models ('cpu' or 'cuda')

    Returns:
        List of (name, model) tuples
    """
    suite = []

    # Always start with Scope 1
    suite.extend(build_linear_suite(n_features))

    if level >= 2:
        suite.extend(build_mlp_suite(n_features))

    if level >= 3:
        suite.extend(build_cnn_local_suite(
            n_features, n_classes, target_cnn_params, device
        ))

    if level >= 4:
        suite.extend(build_cnn_wide_suite(
            n_features, n_classes, target_cnn_params, device
        ))

    return suite


# ===================================================================
# Training and evaluation
# ===================================================================

def train_and_evaluate_suite(
    suite: List[Tuple[str, Any]],
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_holdout: np.ndarray,
    y_holdout: np.ndarray,
    scaler: StandardScaler = None,
    cnn_epochs: int = 30,
    cnn_patience: int = 5,
    cnn_lr: float = 1e-3,
    batch_size: int = 256,
    device: str = 'cpu',
) -> Tuple[Dict[str, float], List[str], int]:
    """
    Train all models in a suite and evaluate on holdout.

    Args:
        suite: List of (name, model) tuples
        X_train: Training features (n, d)
        y_train: Training labels
        X_holdout: Holdout features
        y_holdout: Holdout labels
        scaler: Fitted StandardScaler for non-CNN models
        cnn_epochs: Epochs for CNN training
        cnn_patience: Early stopping patience for CNN
        cnn_lr: Learning rate for CNN
        batch_size: Batch size for CNN training
        device: Device for PyTorch models

    Returns:
        Tuple of:
        - per_model_success: Dict {model_name -> success_rate}
        - trained_names: List of successfully trained model names
        - best_idx: Index of best performing model
    """
    per_model_success = {}
    best_success = 0.0
    best_idx = 0

    # Scale data for sklearn models
    if scaler is None:
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
    else:
        X_train_scaled = scaler.transform(X_train)

    X_holdout_scaled = scaler.transform(X_holdout)

    for idx, (name, model) in enumerate(suite):
        try:
            if isinstance(model, nn.Module):
                # PyTorch CNN model
                model = train_torch_model(
                    model,
                    X_train,
                    y_train,
                    epochs=cnn_epochs,
                    batch_size=batch_size,
                    lr=cnn_lr,
                    patience=cnn_patience,
                    device=device,
                )
                preds = predict_torch(model, X_holdout, device=device)
            else:
                # sklearn model
                model.fit(X_train_scaled, y_train)
                if hasattr(model, 'predict_proba'):
                    y_pred_proba = model.predict_proba(X_holdout_scaled)
                    preds = np.argmax(y_pred_proba, axis=1)
                else:
                    preds = model.predict(X_holdout_scaled)

            # Compute success rate
            success = np.mean(preds == y_holdout)
            per_model_success[name] = success

            if success > best_success:
                best_success = success
                best_idx = idx

        except Exception as e:
            warnings.warn(f"Failed to train {name}: {str(e)[:50]}")
            continue

    trained_names = list(per_model_success.keys())
    return per_model_success, trained_names, best_idx


# ===================================================================
# Experiment runner
# ===================================================================

@dataclass
class ExperimentResult:
    """Container for a single experiment result row."""
    dataset: str
    scope_level: int
    scope_name: str
    seed: int
    M: int
    n_train: int
    n_holdout: int
    bi: float
    bi_margin: float
    p_max: float
    margin_term: float
    best_model: str
    per_model_success: str  # JSON string
    pi: float = np.nan
    ti: float = np.nan
    hi: float = np.nan
    mlp_pi: float = np.nan
    mlp_ti: float = np.nan


def run_experiment(
    dataset: str,
    dataset_dir: str,
    tches20_src_dir: str,
    scopes: List[int] = [1, 2, 3, 4],
    n_seeds: int = 5,
    holdout_size: int = 25000,
    delta: float = 1e-6,
    n_classes: int = 256,
    batch_size: int = 256,
    target_cnn_params: int = 50000,
    cnn_epochs: int = 30,
    cnn_patience: int = 5,
    cnn_lr: float = 1e-3,
    device: str = 'cpu',
    gpu_workers: int = 1,
    completed_keys: Optional[set] = None,
    scope_checkpoint_callback=None,
) -> List[ExperimentResult]:
    """
    Run experiment for a single dataset across all seeds and scopes.

    Args:
        dataset: Dataset name
        dataset_dir: Path to dataset directory
        tches20_src_dir: Path to TCHES20 source directory
        scopes: List of scope levels to evaluate
        n_seeds: Number of random seeds
        holdout_size: Size of holdout set
        delta: Failure probability for BI certification
        n_classes: Number of classes
        batch_size: Batch size for CNN training
        target_cnn_params: Target parameter count for CNNs
        cnn_epochs: Max epochs for CNN training
        cnn_patience: Early stopping patience for CNN
        cnn_lr: Learning rate for CNN
        device: Device for PyTorch models
        completed_keys: Set of (dataset, scope_level, seed) already done
        scope_checkpoint_callback: Called with ExperimentResult after each scope

    Returns:
        List of ExperimentResult objects
    """
    if completed_keys is None:
        completed_keys = set()

    results = []

    # Load dataset
    try:
        X, y = load_dataset(
            dataset,
            data_dir=dataset_dir,
            tches20_src_dir=tches20_src_dir,
        )
    except Exception as e:
        warnings.warn(f"Failed to load {dataset}: {str(e)}")
        return results

    n_features = X.shape[1]
    n_total = len(X)

    # ── Sequential: iterate over seeds, then scopes ──
    for seed in range(n_seeds):
        # Thread-safe RNG
        rng = np.random.RandomState(seed)

        # Split: 60% train, 20% val, 20% holdout
        train_size = int(0.6 * n_total)
        val_size = int(0.2 * n_total)
        holdout_size_actual = min(holdout_size, n_total - train_size - val_size)

        idx = rng.permutation(n_total)
        idx_train = idx[:train_size]
        idx_val = idx[train_size:train_size + val_size]
        idx_holdout = idx[train_size + val_size:train_size + val_size + holdout_size_actual]

        X_train = X[idx_train]
        y_train = y[idx_train]
        X_val = X[idx_val]
        y_val = y[idx_val]
        X_holdout = X[idx_holdout]
        y_holdout = y[idx_holdout]

        # Combine train and val for actual training
        X_train_full = np.vstack([X_train, X_val])
        y_train_full = np.hstack([y_train, y_val])

        # Fit scaler on full training data
        scaler = StandardScaler()
        scaler.fit(X_train_full)

        # Evaluate each scope
        for scope_level in scopes:
            # Skip if already completed
            if (dataset, scope_level, seed) in completed_keys:
                print(f"    seed={seed} scope={scope_level} already in checkpoint, skipping")
                continue

            import time as _time
            t0 = _time.time()
            print(f"    seed={seed} scope={scope_level} ({SCOPE_NAMES[scope_level]})...", end='', flush=True)

            # Build suite for this scope
            suite = build_scope(
                scope_level,
                n_features,
                n_classes,
                target_cnn_params,
                device,
            )

            if len(suite) == 0:
                warnings.warn(f"Empty suite for scope {scope_level}, skipping")
                continue

            # Train and evaluate
            per_model_success, trained_names, best_idx = train_and_evaluate_suite(
                suite,
                X_train_full,
                y_train_full,
                X_holdout,
                y_holdout,
                scaler=scaler,
                cnn_epochs=cnn_epochs,
                cnn_patience=cnn_patience,
                cnn_lr=cnn_lr,
                batch_size=batch_size,
                device=device,
            )

            dt = _time.time() - t0

            if len(per_model_success) == 0:
                warnings.warn(f"No successful models for {dataset} scope {scope_level} seed {seed}")
                print(f" FAILED ({dt:.0f}s)")
                continue

            # Compute BI certificate
            M = len(per_model_success)
            p_max = max(per_model_success.values())
            best_model = trained_names[best_idx] if best_idx < len(trained_names) else "unknown"

            # BI computation
            m = len(y_holdout)
            margin = math.sqrt(math.log(M / delta) / (2 * m))
            certified_success = min(1.0, p_max + margin)
            eps_star = max(0.0, certified_success - 1.0 / n_classes)
            bi = min(math.log2(1 + n_classes * eps_star), math.log2(n_classes))

            # Baseline metrics (using train/val split)
            n_val_bl = min(10000, len(y_train_full) // 5)
            X_tr_bl = X_train_full[:-n_val_bl]
            y_tr_bl = y_train_full[:-n_val_bl]
            X_vl_bl = X_train_full[-n_val_bl:]
            y_vl_bl = y_train_full[-n_val_bl:]

            try:
                pi_val, ti_val = compute_pi(X_tr_bl, y_tr_bl, X_vl_bl, y_vl_bl, n_classes=n_classes)
            except Exception:
                pi_val, ti_val = np.nan, np.nan

            try:
                hi_val = compute_hi(X_tr_bl, y_tr_bl, X_vl_bl, y_vl_bl, n_classes=n_classes)
            except Exception:
                hi_val = np.nan

            mlp_pi_val, mlp_ti_val = compute_mlp_pi_ti(X_tr_bl, y_tr_bl, X_vl_bl, y_vl_bl, n_classes=n_classes)

            # Create result row
            result = ExperimentResult(
                dataset=dataset,
                scope_level=scope_level,
                scope_name=SCOPE_NAMES[scope_level],
                seed=seed,
                M=M,
                n_train=len(X_train_full),
                n_holdout=m,
                bi=bi,
                bi_margin=eps_star,
                p_max=p_max,
                margin_term=margin,
                best_model=best_model,
                per_model_success=json.dumps(per_model_success),
                pi=pi_val,
                ti=ti_val,
                hi=hi_val,
                mlp_pi=mlp_pi_val,
                mlp_ti=mlp_ti_val,
            )
            results.append(result)
            print(f" BI={bi:.4f} p_max={p_max:.4f} M={M} ({dt:.0f}s)")

            # Checkpoint after every (dataset, scope, seed)
            if scope_checkpoint_callback:
                scope_checkpoint_callback(result)

    return results


# ===================================================================
# Plotting
# ===================================================================

def plot_results(df: pd.DataFrame, output_dir: Path, scopes: List[int]):
    """
    Generate summary plots for the attacker-scope comparison.

    Args:
        df: Results DataFrame with columns: dataset, scope_level, scope_name,
            seed, M, bi, p_max, pi, ti, hi, mlp_pi, mlp_ti, margin_term
        output_dir: Output directory for plots
        scopes: List of scope levels evaluated
    """
    # ── Paper-quality formatting constants ──
    TITLE_SIZE = 18
    LABEL_SIZE = 16
    TICK_SIZE = 14
    LEGEND_SIZE = 13
    LINE_WIDTH = 2.5
    MARKER_SIZE = 10
    CAP_SIZE = 5

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    datasets_in_results = sorted(df['dataset'].unique())
    scope_colors = {1: '#3498db', 2: '#e74c3c', 3: '#2ecc71', 4: '#9b59b6'}

    # ---- Plot 1: BI by Scope Level (grouped bar chart per dataset) ----
    fig, ax = plt.subplots(figsize=(max(12, len(datasets_in_results) * 3), 8))

    x = np.arange(len(datasets_in_results))
    n_scopes = len(scopes)
    width = 0.8 / max(n_scopes, 1)

    for idx, scope in enumerate(sorted(scopes)):
        means, stds = [], []
        for ds in datasets_in_results:
            df_sub = df[(df['dataset'] == ds) & (df['scope_level'] == scope)]
            bi_vals = df_sub['bi'].dropna()
            means.append(bi_vals.mean() if len(bi_vals) > 0 else 0)
            stds.append(bi_vals.std() if len(bi_vals) > 1 else 0)

        offset = (idx - n_scopes / 2 + 0.5) * width
        color = scope_colors.get(scope, '#333333')
        ax.bar(x + offset, means, width, yerr=stds,
               label=f'Scope {scope}: {SCOPE_NAMES.get(scope, "")}',
               color=color, alpha=0.85, edgecolor='white',
               capsize=CAP_SIZE, error_kw={'linewidth': 1.5})

    ax.set_xticks(x)
    ax.set_xticklabels(datasets_in_results, rotation=30, ha='right', fontsize=TICK_SIZE)
    ax.set_ylabel('BI (bits)', fontsize=LABEL_SIZE)
    ax.set_title('BI Certification by Attack Scope', fontsize=TITLE_SIZE)
    ax.tick_params(axis='both', labelsize=TICK_SIZE)
    ax.legend(fontsize=LEGEND_SIZE, loc='best')
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(output_dir / 'attacker_scope_bi_by_scope.pdf', dpi=300)
    plt.savefig(output_dir / 'attacker_scope_bi_by_scope.png', dpi=150)
    plt.close()

    # ---- Plot 2: BI vs Scope Level per dataset (line plot with error bars) ----
    fig, ax = plt.subplots(figsize=(12, 8))

    ds_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    for di, ds in enumerate(datasets_in_results):
        df_ds = df[df['dataset'] == ds]
        scope_means, scope_stds, scope_levels = [], [], []
        for scope in sorted(scopes):
            bi_vals = df_ds[df_ds['scope_level'] == scope]['bi'].dropna()
            if len(bi_vals) > 0:
                scope_levels.append(scope)
                scope_means.append(bi_vals.mean())
                scope_stds.append(bi_vals.std() if len(bi_vals) > 1 else 0)

        if scope_levels:
            color = ds_colors[di % len(ds_colors)]
            ax.errorbar(scope_levels, scope_means, yerr=scope_stds,
                        marker='o', capsize=CAP_SIZE, color=color,
                        linewidth=LINE_WIDTH, markersize=MARKER_SIZE,
                        label=ds)

    ax.set_xlabel('Scope Level', fontsize=LABEL_SIZE)
    ax.set_ylabel('BI (bits)', fontsize=LABEL_SIZE)
    ax.set_title('BI Monotonicity Across Attack Scopes', fontsize=TITLE_SIZE)
    ax.tick_params(axis='both', labelsize=TICK_SIZE)
    ax.set_xticks(sorted(scopes))
    ax.set_xticklabels([f'{s}\n{SCOPE_NAMES.get(s, "")}' for s in sorted(scopes)],
                       fontsize=TICK_SIZE - 2)
    ax.legend(fontsize=LEGEND_SIZE, loc='best')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'attacker_scope_bi_monotonicity.pdf', dpi=300)
    plt.savefig(output_dir / 'attacker_scope_bi_monotonicity.png', dpi=150)
    plt.close()

    # ---- Plot 3: All metrics by scope (for each dataset) ----
    metrics_to_plot = [
        ('bi', 'BI', '#2ecc71', 'o'),
        ('pi', 'ePI', '#3498db', 'D'),
        ('ti', 'TI', '#3498db', 'v'),
        ('hi', 'eHI', '#e74c3c', 's'),
        ('mlp_pi', 'MLP-PI', '#9b59b6', 'p'),
    ]

    for ds in datasets_in_results:
        df_ds = df[df['dataset'] == ds]

        fig, ax = plt.subplots(figsize=(12, 8))

        for metric, label, color, marker in metrics_to_plot:
            if metric not in df_ds.columns:
                continue
            scope_means, scope_stds, scope_levels = [], [], []
            for scope in sorted(scopes):
                vals = df_ds[df_ds['scope_level'] == scope][metric].dropna()
                if len(vals) > 0:
                    scope_levels.append(scope)
                    scope_means.append(vals.mean())
                    scope_stds.append(vals.std() if len(vals) > 1 else 0)

            if scope_levels:
                ls = '-' if metric == 'bi' else '--'
                lw = LINE_WIDTH if metric == 'bi' else LINE_WIDTH - 0.5
                ax.errorbar(scope_levels, scope_means, yerr=scope_stds,
                            marker=marker, capsize=CAP_SIZE, color=color,
                            linewidth=lw, markersize=MARKER_SIZE,
                            label=label, linestyle=ls)

        ax.set_xlabel('Scope Level', fontsize=LABEL_SIZE)
        ax.set_ylabel('Bits', fontsize=LABEL_SIZE)
        ax.set_title(f'Leakage Metrics by Scope: {ds}', fontsize=TITLE_SIZE)
        ax.tick_params(axis='both', labelsize=TICK_SIZE)
        ax.set_xticks(sorted(scopes))
        ax.set_xticklabels([f'{s}\n{SCOPE_NAMES.get(s, "")}' for s in sorted(scopes)],
                           fontsize=TICK_SIZE - 2)
        ax.legend(fontsize=LEGEND_SIZE, loc='best')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / f'attacker_scope_metrics_by_scope_{ds}.pdf', dpi=300)
        plt.savefig(output_dir / f'attacker_scope_metrics_by_scope_{ds}.png', dpi=150)
        plt.close()

    # ---- Plot 4: Suite size M and margin term by scope ----
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    # Left: Suite size M
    for di, ds in enumerate(datasets_in_results):
        df_ds = df[df['dataset'] == ds]
        scope_levels, m_means = [], []
        for scope in sorted(scopes):
            m_vals = df_ds[df_ds['scope_level'] == scope]['M'].dropna()
            if len(m_vals) > 0:
                scope_levels.append(scope)
                m_means.append(m_vals.mean())
        if scope_levels:
            color = ds_colors[di % len(ds_colors)]
            ax1.plot(scope_levels, m_means, marker='o', color=color,
                     linewidth=LINE_WIDTH, markersize=MARKER_SIZE, label=ds)

    ax1.set_xlabel('Scope Level', fontsize=LABEL_SIZE)
    ax1.set_ylabel('Suite Size M', fontsize=LABEL_SIZE)
    ax1.set_title('Suite Size by Scope', fontsize=TITLE_SIZE)
    ax1.tick_params(axis='both', labelsize=TICK_SIZE)
    ax1.set_xticks(sorted(scopes))
    ax1.legend(fontsize=LEGEND_SIZE, loc='best')
    ax1.grid(True, alpha=0.3)

    # Right: Margin term
    for di, ds in enumerate(datasets_in_results):
        df_ds = df[df['dataset'] == ds]
        scope_levels, margin_means = [], []
        for scope in sorted(scopes):
            margin_vals = df_ds[df_ds['scope_level'] == scope]['margin_term'].dropna()
            if len(margin_vals) > 0:
                scope_levels.append(scope)
                margin_means.append(margin_vals.mean())
        if scope_levels:
            color = ds_colors[di % len(ds_colors)]
            ax2.plot(scope_levels, margin_means, marker='s', color=color,
                     linewidth=LINE_WIDTH, markersize=MARKER_SIZE, label=ds)

    ax2.set_xlabel('Scope Level', fontsize=LABEL_SIZE)
    ax2.set_ylabel('Margin Term', fontsize=LABEL_SIZE)
    ax2.set_title(r'Confidence Slack Diagnostic', fontsize=TITLE_SIZE)
    ax2.tick_params(axis='both', labelsize=TICK_SIZE)
    ax2.set_xticks(sorted(scopes))
    ax2.legend(fontsize=LEGEND_SIZE, loc='best')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'attacker_scope_suite_size_and_margin.pdf', dpi=300)
    plt.savefig(output_dir / 'attacker_scope_suite_size_and_margin.png', dpi=150)
    plt.close()

    print(f"Plots saved to {output_dir}")


# ===================================================================
# Main
# ===================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Attacker scope comparison'
    )
    parser.add_argument(
        '--datasets',
        type=str,
        default='ascad_desync_0 aes_rd',
        help='Space-separated list of datasets or "all"'
    )
    parser.add_argument(
        '--dataset-dir',
        type=str,
        default='./datasets',
        help='Path to datasets directory'
    )
    parser.add_argument(
        '--tches20-src-dir',
        type=str,
        default='./datasets',
        help='Path to TCHES20 source data'
    )
    parser.add_argument(
        '--scopes',
        type=str,
        default='1,2,3,4',
        help='Comma-separated scope levels to evaluate'
    )
    parser.add_argument(
        '--n-seeds',
        type=int,
        default=5,
        help='Number of random seeds'
    )
    parser.add_argument(
        '--holdout-size',
        type=int,
        default=25000,
        help='Size of holdout set'
    )
    parser.add_argument(
        '--delta',
        type=float,
        default=1e-6,
        help='Failure probability for BI certification'
    )
    parser.add_argument(
        '--n-classes',
        type=int,
        default=256,
        help='Number of classes'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=256,
        help='Batch size for CNN training'
    )
    parser.add_argument(
        '--target-cnn-params',
        type=int,
        default=50000,
        help='Target parameter count for CNN models'
    )
    parser.add_argument(
        '--cnn-epochs',
        type=int,
        default=30,
        help='Max epochs for CNN training'
    )
    parser.add_argument(
        '--cnn-patience',
        type=int,
        default=5,
        help='Early stopping patience for CNN'
    )
    parser.add_argument(
        '--cnn-lr',
        type=float,
        default=1e-3,
        help='Learning rate for CNN'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='results/attacker_scope_diagnostic',
        help='Output directory'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu' if HAS_TORCH else 'cpu',
        help='Device: cpu or cuda'
    )
    parser.add_argument(
        '--gpu-workers',
        type=int,
        default=1,
        help='Concurrent GPU training workers (seeds trained in parallel '
             'via CUDA streams). Set to 4-5 for full single-GPU utilization. '
             'Default 1 = sequential.'
    )
    parser.add_argument(
        '--plot-only',
        action='store_true',
        help='Only generate plots from existing checkpoint; do not run experiments'
    )
    parser.add_argument(
        '--no-resume',
        action='store_true',
        help='Force fresh run (ignore existing checkpoint)'
    )

    args = parser.parse_args()

    # Parse datasets
    if args.datasets.lower() == 'all':
        datasets = ALL_DATASETS
    else:
        datasets = args.datasets.split()

    # Parse scopes
    scopes = [int(s.strip()) for s in args.scopes.split(',')]

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("ATTACKER SCOPE COMPARISON")
    print("=" * 80)
    print(f"Datasets: {datasets}")
    print(f"Scopes: {scopes}")
    print(f"Seeds: {args.n_seeds}")
    print(f"Device: {args.device}")
    print(f"Output: {output_dir}")
    print("=" * 80)

    # ---- CHECKPOINT/RESUME LOGIC ----
    checkpoint_path = output_dir / 'scope_checkpoint.csv'
    existing_df = None
    completed_keys = set()  # (dataset, scope_level, seed) tuples

    if checkpoint_path.exists() and not args.no_resume:
        try:
            existing_df = pd.read_csv(checkpoint_path)
            print(f"\n[CHECKPOINT] Found checkpoint with {len(existing_df)} rows")
            for _, row in existing_df.iterrows():
                completed_keys.add((row['dataset'], int(row['scope_level']), int(row['seed'])))
            # Summarize what's done
            ds_counts = {}
            for (d, sc, s) in completed_keys:
                ds_counts[d] = ds_counts.get(d, 0) + 1
            expected_per_ds = len(scopes) * args.n_seeds
            for d, c in sorted(ds_counts.items()):
                status = "COMPLETE" if c >= expected_per_ds else f"{c}/{expected_per_ds}"
                print(f"  {d}: {status}")
        except Exception as e:
            print(f"  Warning: failed to load checkpoint: {e}")

    # If --plot-only mode, skip experiments and go straight to plotting
    if args.plot_only:
        print("\n[PLOT-ONLY MODE] Skipping experiments, generating plots from checkpoint...")
        if checkpoint_path.exists():
            df = pd.read_csv(checkpoint_path)
            if not df.empty:
                figures_dir = output_dir / 'figures'
                try:
                    plot_results(df, figures_dir, scopes)
                    print(f"\nPlots generated successfully in {figures_dir}")
                except Exception as e:
                    print(f"ERROR during plot generation: {e}")
                    traceback.print_exc()
                    sys.exit(1)
            else:
                print("ERROR: Checkpoint file is empty")
                sys.exit(1)
        else:
            print(f"ERROR: Checkpoint not found at {checkpoint_path}")
            sys.exit(1)
        sys.exit(0)

    # Accumulated results for this run
    accumulated_results = []

    def _save_checkpoint_result(result: ExperimentResult):
        """Save checkpoint after every (dataset, scope, seed)."""
        accumulated_results.append(result)
        new_df = pd.DataFrame([asdict(r) for r in accumulated_results])
        if existing_df is not None:
            combined = pd.concat([existing_df, new_df], ignore_index=True)
            combined = combined.drop_duplicates(
                subset=['dataset', 'scope_level', 'seed'],
                keep='last'
            )
        else:
            combined = new_df
        combined.to_csv(checkpoint_path, index=False)
        print(f"      [checkpoint: {len(combined)} total rows]")

    def _sigterm_handler(signum, frame):
        """Handle SIGTERM and SIGUSR1 by saving checkpoint and attempting plots."""
        print(f"\n  SIGNAL {signum} received - saving partial results...")
        # Final save
        if accumulated_results:
            new_df = pd.DataFrame([asdict(r) for r in accumulated_results])
            if existing_df is not None:
                combined = pd.concat([existing_df, new_df], ignore_index=True)
                combined = combined.drop_duplicates(
                    subset=['dataset', 'scope_level', 'seed'], keep='last'
                )
            else:
                combined = new_df
            combined.to_csv(checkpoint_path, index=False)
        # Try to generate plots
        try:
            if checkpoint_path.exists():
                df = pd.read_csv(checkpoint_path)
                if df is not None and not df.empty:
                    figures_dir = output_dir / 'figures'
                    plot_results(df, figures_dir, scopes)
                    print(f"  Plots saved to {figures_dir}")
        except Exception as e:
            print(f"  [WARN] Plot generation failed: {e}")
        print(f"  Checkpoint saved. Re-run to resume.")
        sys.exit(0)

    # Register signal handlers
    signal.signal(signal.SIGTERM, _sigterm_handler)
    signal.signal(signal.SIGUSR1, _sigterm_handler)

    # Run experiment for each dataset
    for dataset in datasets:
        # Check if ALL (scope, seed) combos are done for this dataset
        ds_done = sum(1 for (d, sc, s) in completed_keys if d == dataset)
        expected = len(scopes) * args.n_seeds
        if ds_done >= expected:
            print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Skipping {dataset} (all {expected} entries in checkpoint)")
            continue

        print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Processing {dataset} ({ds_done}/{expected} done)...")

        results = run_experiment(
            dataset=dataset,
            dataset_dir=args.dataset_dir,
            tches20_src_dir=args.tches20_src_dir,
            scopes=scopes,
            n_seeds=args.n_seeds,
            holdout_size=args.holdout_size,
            delta=args.delta,
            n_classes=args.n_classes,
            batch_size=args.batch_size,
            target_cnn_params=args.target_cnn_params,
            cnn_epochs=args.cnn_epochs,
            cnn_patience=args.cnn_patience,
            cnn_lr=args.cnn_lr,
            device=args.device,
            gpu_workers=args.gpu_workers,
            completed_keys=completed_keys,
            scope_checkpoint_callback=_save_checkpoint_result,
        )

        print(f"  Completed: {len(results)} new result rows")

    # Load final results from checkpoint or accumulated data
    if checkpoint_path.exists():
        df = pd.read_csv(checkpoint_path)
        print(f"\n[FINAL] Loaded {len(df)} total rows from checkpoint")
    elif accumulated_results:
        df = pd.DataFrame([asdict(r) for r in accumulated_results])
        print(f"\n[FINAL] Using {len(df)} accumulated results")
    else:
        df = None

    # Save results to CSV and generate plots
    if df is not None and not df.empty:

        # Print summary
        print("\n" + "=" * 80)
        print("SUMMARY TABLE: Mean BI by Scope")
        print("=" * 80)

        summary = df.groupby(['dataset', 'scope_level']).agg({
            'M': 'mean',
            'bi': ['mean', 'std'],
            'pi': ['mean', 'std'],
            'ti': ['mean', 'std'],
            'hi': ['mean', 'std'],
            'mlp_pi': ['mean', 'std'],
            'p_max': 'mean',
            'margin_term': 'mean',
        }).round(4)

        print(summary)

        # Verify monotonicity
        print("\n" + "=" * 80)
        print("MONOTONICITY CHECK: BI(Scope_i) <= BI(Scope_j) for i < j")
        print("=" * 80)

        for dataset in datasets:
            df_ds = df[df['dataset'] == dataset]
            for seed in df_ds['seed'].unique():
                df_seed = df_ds[df_ds['seed'] == seed].sort_values('scope_level')
                if len(df_seed) < 2:
                    continue

                bi_vals = df_seed['bi'].values
                is_monotonic = all(bi_vals[i] <= bi_vals[i+1] for i in range(len(bi_vals)-1))

                if not is_monotonic:
                    print(f"WARNING: {dataset} seed {seed} NOT monotonic: {bi_vals}")

        print("Monotonicity check complete.")
        print("\n" + "=" * 80)
        print("BI SCALING ANALYSIS")
        print("=" * 80)

        for scope_level in sorted(scopes):
            df_scope = df[df['scope_level'] == scope_level]
            M_mean = df_scope['M'].mean()
            margin_sqrt_ln = math.sqrt(math.log(M_mean / args.delta) / 2)

            print(f"\nScope {scope_level} ({SCOPE_NAMES[scope_level]}):")
            print(f"  M (avg): {M_mean:.1f}")
            print(f"  √ln(M/δ)/√2 term: {margin_sqrt_ln:.6f}")
            print(f"  BI (mean ± std): {df_scope['bi'].mean():.4f} ± {df_scope['bi'].std():.4f}")
            print(f"  p_max (mean): {df_scope['p_max'].mean():.6f}")

        csv_path = output_dir / 'scope_results.csv'
        df.to_csv(csv_path, index=False)
        print(f"Results saved to {csv_path}")

        # Generate plots
        figures_dir = output_dir / 'figures'
        plot_results(df, figures_dir, scopes)

    else:
        print("\nERROR: No results generated. Check dataset paths and configuration.")
        sys.exit(1)


if __name__ == '__main__':
    main()
