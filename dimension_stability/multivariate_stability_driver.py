"""
Multivariate Stability: BI vs PI/HI/MI Instability

Loads pretrained Keras (.hdf5) models from the TCHES20 repository and/or
trains sklearn attack suites to compare BI certification with traditional
leakage metrics across datasets and dimensions.

Part A: Cross-dataset certification with 80 pretrained models
  - Bar chart of BI across TCHES20 datasets
  - Per-model success breakdown
  - Comparison with ePI, TI, MLP-PI, eHI

Part B: Dimension sweep stability analysis
  - Sweep d' from 10 to 1000
  - Track which metrics break (produce NaN/diverge)
  - Bar chart of max tolerated dimension per metric
  - BI remains stable while PI/HI/MI fail

Supported datasets (from Wouters et al., TCHES 2020):
  ascad_desync_0, ascad_desync_50, ascad_desync_100,
  aes_hd, aes_rd, dpav4

Usage examples:
  # Cross-dataset certification with pretrained models
  python -m dimension_stability.multivariate_stability_driver \\
      --models-dir ./models/pretrained_models/models \\
      --datasets all --include-baselines \\
      --output-dir results/dimension_stability

  # Dimension sweep on synthetic data
  python -m dimension_stability.multivariate_stability_driver \\
      --dataset synthetic --sweep-dimensions \\
      --dimensions 10,20,50,100,200,500,1000 \\
      --include-baselines --output-dir results/dimension_stability
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import warnings
import json
import argparse
import sys
import os
import signal
from datetime import datetime
from tqdm import tqdm
import traceback
import math

# Local imports
from core.data_loader import (
    load_dataset, create_data_split, generate_synthetic_data, DataSplit
)
from core.baseline_metrics import compute_pi, compute_hi, compute_mi
from core.bi_certificate import compute_bi_certificate

# Sklearn imports
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler


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

# Substrings used to identify each dataset in model filenames.
# Longer/more specific patterns are checked first (see detect_dataset_tag).
DATASET_KEYWORDS = {
    'ascad_desync_100': ['ascad_desync_100', 'ascad_desync100'],
    'ascad_desync_50':  ['ascad_desync_50',  'ascad_desync50'],
    'ascad_desync_0':   ['ascad_desync_0',   'ascad_desync0'],
    'aes_hd':           ['aes_hd'],
    'aes_rd':           ['aes_rd'],
    'dpav4':            ['dpav4', 'dpa_v4', 'dpav4_'],
}


# ===================================================================
# Model discovery and loading
# ===================================================================

def detect_dataset_tag(filename: str) -> Optional[str]:
    """
    Infer the dataset tag from a model filename.

    Checks longer keywords first so that, for example,
    'ascad_desync_100' is matched before bare 'ascad'.
    """
    name_lower = filename.lower()
    for tag, keywords in sorted(
        DATASET_KEYWORDS.items(),
        key=lambda kv: -max(len(k) for k in kv[1]),
    ):
        for kw in keywords:
            if kw in name_lower:
                return tag
    return None


def discover_pretrained_models(
    models_dir: str,
    extension: str = '.hdf5',
) -> Dict[str, List[Path]]:
    """
    Recursively scan *models_dir* for model files and group by dataset.

    Returns a dict mapping dataset tag to a sorted list of Paths.
    """
    root = Path(models_dir)
    if not root.exists():
        raise FileNotFoundError(f"Models directory not found: {root}")

    grouped: Dict[str, List[Path]] = {}
    for path in sorted(root.rglob(f'*{extension}')):
        tag = detect_dataset_tag(path.name)
        if tag is None:
            warnings.warn(
                f"Unrecognised dataset in filename, skipping: {path.name}"
            )
            continue
        grouped.setdefault(tag, []).append(path)

    return grouped


def load_keras_model(model_path: Path):
    """Load a Keras/TF model from an .hdf5 file (compile=False)."""
    try:
        import tensorflow as tf
        return tf.keras.models.load_model(str(model_path), compile=False)
    except ImportError:
        pass
    try:
        import keras
        return keras.models.load_model(str(model_path), compile=False)
    except ImportError:
        raise ImportError(
            "Neither tensorflow nor keras is installed. "
            "Install one to load .hdf5 models."
        )


def get_model_input_shape(model) -> Tuple[int, ...]:
    """Return expected input shape excluding the batch dimension."""
    return tuple(d for d in model.input_shape[1:] if d is not None)


def predict_with_model(
    model,
    X: np.ndarray,
    batch_size: int = 1024,
) -> np.ndarray:
    """
    Run prediction and return hard class labels.

    Reshapes X to match the model's expected input (adds a channel
    dimension if needed, truncates or pads the trace length).
    """
    expected = get_model_input_shape(model)

    # Model expects (T, 1) but X is (n, T)
    if len(expected) == 2 and expected[-1] == 1 and X.ndim == 2:
        T = expected[0]
        if X.shape[1] > T:
            X = X[:, :T]
        elif X.shape[1] < T:
            pad = np.zeros((X.shape[0], T - X.shape[1]))
            X = np.hstack([X, pad])
        X = X[..., np.newaxis]
    elif len(expected) == 1:
        T = expected[0]
        if X.shape[1] > T:
            X = X[:, :T]
        elif X.shape[1] < T:
            pad = np.zeros((X.shape[0], T - X.shape[1]))
            X = np.hstack([X, pad])

    probs = model.predict(X, batch_size=batch_size, verbose=0)
    return np.argmax(probs, axis=-1)


def load_pretrained_suite(
    model_paths: List[Path],
    verbose: bool = True,
) -> List[Tuple[str, object]]:
    """Load a list of Keras models, skipping any that fail."""
    suite = []
    iterator = model_paths
    if verbose:
        iterator = tqdm(model_paths, desc="Loading models", unit="model")
    for path in iterator:
        name = path.stem
        try:
            model = load_keras_model(path)
            suite.append((name, model))
        except Exception as e:
            warnings.warn(f"Failed to load {path.name}: {e}")
    if verbose:
        print(f"  Successfully loaded {len(suite)}/{len(model_paths)} models")
    return suite


# ===================================================================
# TCHES20 data loading helpers
# ===================================================================

def try_load_tches20_data(
    dataset_tag: str,
    dataset_dir: str,
    tches20_src_dir: str = '',
):
    """
    Attempt to load data using the TCHES20 repo's own dataLoaders.

    Falls back to the project's generic load_dataset if the TCHES20
    loader is not available.

    Returns (traces, labels) as numpy arrays.
    """
    # Try TCHES20 dataLoaders first
    if tches20_src_dir:
        sys.path.insert(0, tches20_src_dir)
    try:
        from dataLoaders import (
            load_ascad, load_aes_hd, load_aes_rd, load_dpav4,
        )

        loader_map = {
            'ascad_desync_0':   lambda: load_ascad(
                os.path.join(dataset_dir, 'ASCAD_dataset', 'ASCAD.h5')),
            'ascad_desync_50':  lambda: load_ascad(
                os.path.join(dataset_dir, 'ASCAD_dataset', 'ASCAD_desync50.h5')),
            'ascad_desync_100': lambda: load_ascad(
                os.path.join(dataset_dir, 'ASCAD_dataset', 'ASCAD_desync100.h5')),
            'aes_hd':           lambda: load_aes_hd(
                os.path.join(dataset_dir, 'AES_HD_dataset/')),
            'aes_rd':           lambda: load_aes_rd(
                os.path.join(dataset_dir, 'AES_RD_dataset/')),
            'dpav4':            lambda: load_dpav4(
                os.path.join(dataset_dir, 'DPAv4_dataset/')),
        }

        if dataset_tag in loader_map:
            result = loader_map[dataset_tag]()
            # TCHES20 loaders return:
            #   (profiling_traces, profiling_labels,
            #    attack_traces, attack_labels_per_key, correct_key)
            # We use the profiling set for BI certification (train + holdout).
            profiling_traces = result[0]
            profiling_labels = result[1]
            print(f"  Loaded via TCHES20 dataLoaders: "
                  f"{profiling_traces.shape[0]} profiling traces, "
                  f"{profiling_traces.shape[1]} features")
            return profiling_traces, profiling_labels

    except ImportError:
        pass

    # Fallback to generic loader
    print(f"  TCHES20 dataLoaders not found, using generic load_dataset")
    traces, labels = load_dataset(dataset_tag, data_dir=dataset_dir, tches20_src_dir=tches20_src_dir)
    return traces, labels


# ===================================================================
# Windowing helper (from both files)
# ===================================================================

def create_windowed_representation(
    traces: np.ndarray,
    target_dim: int,
    method: str = 'center',
) -> np.ndarray:
    """
    Create a representation with specified dimension by windowing.

    Args:
        traces: Full trace array (n_samples, n_features)
        target_dim: Target dimension
        method: 'center', 'random', or 'start'

    Returns:
        Windowed traces (n_samples, target_dim)
    """
    n_samples, n_features = traces.shape

    if target_dim >= n_features:
        # Pad with noise if needed
        if target_dim > n_features:
            padding = np.random.randn(n_samples, target_dim - n_features) * 0.1
            return np.hstack([traces, padding])
        return traces

    if method == 'center':
        start = (n_features - target_dim) // 2
    elif method == 'start':
        start = 0
    elif method == 'random':
        start = np.random.randint(0, n_features - target_dim)
    else:
        start = 0

    return traces[:, start:start + target_dim]


# ===================================================================
# Attack suite training
# ===================================================================

def train_attack_suite(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_classes: int = 256
) -> List[Tuple[str, object]]:
    """
    Train a suite of attack models.

    Args:
        X_train: Training features
        y_train: Training labels
        n_classes: Number of classes

    Returns:
        List of (model_name, trained_model) tuples
    """
    models = []

    # Logistic Regression
    try:
        lr = LogisticRegression(
            solver='lbfgs',
            max_iter=500,
            n_jobs=-1
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            lr.fit(X_train, y_train)
        models.append(('logistic', lr))
    except Exception as e:
        warnings.warn(f"Logistic Regression failed: {e}")

    # LDA
    try:
        lda = LinearDiscriminantAnalysis()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            lda.fit(X_train, y_train)
        models.append(('lda', lda))
    except Exception as e:
        warnings.warn(f"LDA failed: {e}")

    # MLP (small for speed)
    try:
        mlp = MLPClassifier(
            hidden_layer_sizes=(32,),
            max_iter=100,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=5
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mlp.fit(X_train, y_train)
        models.append(('mlp', mlp))
    except Exception as e:
        warnings.warn(f"MLP failed: {e}")

    return models


# ===================================================================
# Attacker evaluation (merged from both files)
# ===================================================================

def evaluate_suite_on_holdout(
    models: List[Tuple[str, object]],
    X_holdout: np.ndarray,
    y_holdout: np.ndarray,
    is_keras: bool = False,
    batch_size: int = 1024,
) -> Tuple[float, str, Dict[str, float]]:
    """
    Evaluate every model in the suite on the holdout set.

    Args:
        models: List of (name, model) tuples
        X_holdout: Holdout features
        y_holdout: Holdout labels
        is_keras: Whether to use Keras prediction (with reshaping)
        batch_size: Batch size for Keras predictions

    Returns:
        (best_success_rate, best_model_name, per_model_dict)
    """
    best_success = 0.0
    best_name = None
    per_model: Dict[str, float] = {}
    y_true = np.asarray(y_holdout).reshape(-1)

    for name, model in models:
        try:
            if is_keras:
                y_pred = predict_with_model(model, X_holdout, batch_size)
            else:
                y_pred = model.predict(X_holdout)
            y_pred = np.asarray(y_pred).reshape(-1)
            success = float(np.mean(y_pred == y_true[: len(y_pred)]))
            per_model[name] = success
            if success > best_success:
                best_success = success
                best_name = name
        except Exception as e:
            warnings.warn(f"Evaluation failed for {name}: {e}")
            per_model[name] = float('nan')

    return best_success, best_name, per_model


# ===================================================================
# MLP-based PI/TI computation (NEW)
# ===================================================================

def compute_mlp_pi_ti(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    n_classes: int = 256,
) -> Tuple[float, float]:
    """
    MLP-based (latent) Perceived Information and Training Information.
    Uses sklearn MLPClassifier with architecture (64, 32, 16, 4).

    Args:
        X_train: Training traces
        y_train: Training labels
        X_val: Validation traces
        y_val: Validation labels
        n_classes: Number of classes

    Returns:
        Tuple of (PI_mlp, TI_mlp) in bits
    """
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_train)
    X_vl = scaler.transform(X_val)

    clf = MLPClassifier(
        hidden_layer_sizes=(64, 32, 16, 4),
        activation='relu',
        max_iter=200,
        early_stopping=True,
        validation_fraction=0.15,
        batch_size=256,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        clf.fit(X_tr, y_train)

    log_proba_val = clf.predict_log_proba(X_vl)
    log_proba_train = clf.predict_log_proba(X_tr)

    classes = clf.classes_
    score_val = 0.0
    score_train = 0.0
    n_active = 0

    for x in classes:
        idx_val = np.where(y_val == x)[0]
        idx_train = np.where(y_train == x)[0]
        class_col = np.where(classes == x)[0]
        if len(class_col) == 0:
            continue
        class_col = class_col[0]
        if len(idx_val) > 0:
            score_val += np.mean(log_proba_val[idx_val, class_col])
            n_active += 1
        if len(idx_train) > 0:
            score_train += np.mean(log_proba_train[idx_train, class_col])

    if n_active > 0:
        score_val /= n_active
        score_train /= n_active

    PI = np.log2(n_classes) + score_val * np.log2(math.e)
    TI = np.log2(n_classes) + score_train * np.log2(math.e)
    return PI, TI


# ===================================================================
# Configuration
# ===================================================================

@dataclass
class ExperimentConfig:
    """Configuration for the multivariate stability diagnostics."""
    # Dataset selection
    dataset_name: str = 'synthetic'
    dataset_path: str = ''
    datasets: List[str] = field(default_factory=lambda: ['aes_hd'])

    # Paths (for pretrained models)
    models_dir: str = ''
    dataset_dir: str = ''
    tches20_src_dir: str = ''

    # Dimension sweep
    dimensions: List[int] = field(default_factory=lambda: [10, 20, 50, 100, 200, 500, 1000])
    sweep_dimensions: bool = False

    # Fixed certification parameters
    holdout_size: int = 50000
    suite_size: int = 3
    delta: float = 1e-6
    n_classes: int = 256

    # Experiment parameters
    n_seeds: int = 10
    n_samples_total: int = 100000
    batch_size: int = 1024

    # Data split
    train_ratio: float = 0.5
    val_ratio: float = 0.0

    # Baselines
    include_baselines: bool = True
    max_dim_for_mi: int = 100

    # Mode: 'synthetic' | 'pretrained' (auto-detected)
    use_pretrained_models: bool = False


# ===================================================================
# Single-seed runner (merged from both files)
# ===================================================================

def run_single_seed(
    traces: np.ndarray,
    labels: np.ndarray,
    pretrained_suite: Optional[List[Tuple[str, object]]],
    config: ExperimentConfig,
    dataset_tag: str,
    seed: int,
    dimension: Optional[int] = None,
) -> Dict:
    """
    Run one seed for one dataset (optionally at a fixed window width).

    Args:
        traces: Trace data
        labels: Label data
        pretrained_suite: Optional list of pre-trained Keras models
        config: Experiment configuration
        dataset_tag: Dataset identifier
        seed: Random seed
        dimension: Optional target dimension for windowing

    Returns:
        Dictionary with all metric results
    """
    np.random.seed(seed)

    X = traces.copy()
    if dimension is not None:
        X = create_windowed_representation(X, dimension)

    n_total = len(labels)
    n_holdout = min(config.holdout_size, n_total // 2)
    n_train = n_total - n_holdout

    indices = np.random.permutation(n_total)
    train_idx = indices[:n_train]
    holdout_idx = indices[n_train:n_train + n_holdout]

    X_train, y_train = X[train_idx], labels[train_idx]
    X_holdout, y_holdout = X[holdout_idx], labels[holdout_idx]

    results = {
        'dataset': dataset_tag,
        'dimension': dimension if dimension is not None else X.shape[1],
        'seed': seed,
        'n_train': len(y_train),
        'n_holdout': len(y_holdout),
    }

    # ---- BI Certificate ----
    # Use pretrained suite if available, otherwise train attack suite
    if pretrained_suite is not None:
        models = pretrained_suite
        is_keras = True
        M = len(models)
    else:
        models = train_attack_suite(X_train, y_train, config.n_classes)
        is_keras = False
        M = len(models)

    try:
        best_success, best_model, per_model = evaluate_suite_on_holdout(
            models, X_holdout, y_holdout,
            is_keras=is_keras, batch_size=config.batch_size,
        )
        cert = compute_bi_certificate(
            holdout_success=best_success,
            m=len(y_holdout),
            M=M,
            delta=config.delta,
            n_classes=config.n_classes,
        )
        results.update({
            'bi': cert.bi_bits,
            'bi_margin': cert.margin,
            'bi_success': True,
            'best_model': best_model,
            'observed_success': best_success,
            'M': M,
            'per_model_success': json.dumps(per_model),
        })
    except Exception as e:
        results.update({
            'bi': np.nan,
            'bi_margin': np.nan,
            'bi_success': False,
            'bi_error': str(e),
        })

    # ---- PI / HI / MI / MLP-PI / MLP-TI (optional) ----
    if config.include_baselines:
        n_val = min(10000, len(y_train) // 5)
        X_tr_bl, y_tr_bl = X_train[:-n_val], y_train[:-n_val]
        X_vl_bl, y_vl_bl = X_train[-n_val:], y_train[-n_val:]

        # Logistic PI (ePI)
        try:
            pi, ti = compute_pi(
                X_tr_bl, y_tr_bl, X_vl_bl, y_vl_bl,
                method='logistic', n_classes=config.n_classes,
            )
            results['pi'] = pi
            results['ti'] = ti
            results['pi_success'] = True
        except Exception as e:
            results['pi'] = np.nan
            results['ti'] = np.nan
            results['pi_success'] = False
            results['pi_error'] = str(e)

        # HI (eHI)
        try:
            hi = compute_hi(
                X_tr_bl, y_tr_bl, X_vl_bl, y_vl_bl,
                n_classes=config.n_classes,
            )
            results['hi'] = hi
            results['hi_success'] = True
        except Exception as e:
            results['hi'] = np.nan
            results['hi_success'] = False
            results['hi_error'] = str(e)

        # MLP-based PI/TI
        try:
            mlp_pi, mlp_ti = compute_mlp_pi_ti(
                X_tr_bl, y_tr_bl, X_vl_bl, y_vl_bl,
                n_classes=config.n_classes,
            )
            results['mlp_pi'] = mlp_pi
            results['mlp_ti'] = mlp_ti
            results['mlp_pi_success'] = True
        except Exception as e:
            results['mlp_pi'] = np.nan
            results['mlp_ti'] = np.nan
            results['mlp_pi_success'] = False
            results['mlp_pi_error'] = str(e)

        # MI
        eff_dim = dimension if dimension is not None else X.shape[1]
        if eff_dim <= config.max_dim_for_mi:
            try:
                mi, ok = compute_mi(
                    X_train, y_train,
                    max_dim_for_estimate=config.max_dim_for_mi,
                )
                results['mi'] = mi if ok else np.nan
                results['mi_success'] = ok
            except Exception as e:
                results['mi'] = np.nan
                results['mi_success'] = False
                results['mi_error'] = str(e)
        else:
            results['mi'] = np.nan
            results['mi_success'] = False
            results['mi_error'] = f"Dimension {eff_dim} > max {config.max_dim_for_mi}"

    return results


# ===================================================================
# Per-dataset runner
# ===================================================================

def run_for_dataset(
    dataset_tag: str,
    pretrained_suite: Optional[List[Tuple[str, object]]],
    config: ExperimentConfig,
    completed_seeds: Optional[set] = None,
    seed_checkpoint_callback=None,
    preloaded_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
) -> pd.DataFrame:
    """
    Run all seeds (and optionally dimension sweep) for one dataset.

    Args:
        dataset_tag: Dataset identifier
        pretrained_suite: Optional list of pretrained models
        config: Experiment configuration
        completed_seeds: Set of (dataset, dimension, seed) tuples to skip
        seed_checkpoint_callback: Called with result dict after each seed
        preloaded_data: Optional (traces, labels) tuple to skip data loading
                        (used for synthetic data mode)

    Returns:
        DataFrame with all results
    """
    print(f"\n{'=' * 60}")
    if pretrained_suite:
        print(f"Dataset: {dataset_tag}   |   Suite size M = {len(pretrained_suite)}")
    else:
        print(f"Dataset: {dataset_tag}   |   Training sklearn suite")
    print(f"{'=' * 60}")

    if completed_seeds is None:
        completed_seeds = set()

    # Load data (or use preloaded)
    if preloaded_data is not None:
        traces, labels = preloaded_data
        print(f"  Using preloaded data: {traces.shape}")
    else:
        traces, labels = try_load_tches20_data(
            dataset_tag, config.dataset_dir, config.tches20_src_dir,
        )
    print(f"  Traces shape: {traces.shape}")

    all_results = []

    if config.sweep_dimensions and config.dimensions:
        for dim in config.dimensions:
            print(f"\n  Dimension d' = {dim}")
            for seed in tqdm(range(config.n_seeds), desc=f"  d={dim}", leave=False):
                if (dataset_tag, dim, seed) in completed_seeds:
                    print(f"    seed {seed} already in checkpoint, skipping")
                    continue
                try:
                    r = run_single_seed(
                        traces, labels, pretrained_suite,
                        config, dataset_tag, seed, dimension=dim,
                    )
                    all_results.append(r)
                    if seed_checkpoint_callback:
                        seed_checkpoint_callback(r)
                except Exception as e:
                    print(f"    Seed {seed} failed: {e}")
    else:
        for seed in tqdm(range(config.n_seeds), desc=f"  {dataset_tag}", leave=False):
            if (dataset_tag, None, seed) in completed_seeds:
                print(f"    seed {seed} already in checkpoint, skipping")
                continue
            try:
                r = run_single_seed(
                    traces, labels, pretrained_suite,
                    config, dataset_tag, seed, dimension=None,
                )
                all_results.append(r)
                if seed_checkpoint_callback:
                    seed_checkpoint_callback(r)
            except Exception as e:
                print(f"    Seed {seed} failed: {e}")

    df = pd.DataFrame(all_results)

    # Quick summary
    if not df.empty and 'bi' in df.columns:
        valid = df['bi'].dropna()
        if len(valid) > 0:
            print(f"\n  BI: {valid.mean():.4f} +/- {valid.std():.4f}  "
                  f"(n = {len(valid)}/{len(df)})")

    return df


# ===================================================================
# Main orchestrator
# ===================================================================

def run_experiment(
    config: ExperimentConfig,
    completed_seeds: Optional[set] = None,
    seed_checkpoint_callback=None,
) -> pd.DataFrame:
    """
    Run the experiment across all requested datasets.

    Args:
        config: Experiment configuration
        completed_seeds: Set of (dataset, dimension, seed) tuples to skip
        seed_checkpoint_callback: Called with result dict after each seed completes

    Returns:
        DataFrame with all results
    """
    if completed_seeds is None:
        completed_seeds = set()
    print("=" * 70)
    print("Multivariate leakage stability certification")
    print("=" * 70)

    # Determine mode: pretrained or synthetic
    use_pretrained = (config.models_dir and config.dataset_dir)

    if use_pretrained:
        print(f"Mode: Pretrained Model Evaluation")
        print(f"Models dir:       {config.models_dir}")
        print(f"Dataset dir:      {config.dataset_dir}")
        print(f"Requested:        {config.datasets}")
    else:
        print(f"Mode: Synthetic Data with Sklearn Attacks")
        print(f"Dataset:          {config.dataset_name}")
        if config.sweep_dimensions:
            print(f"Dim sweep:        {config.dimensions}")

    print(f"Seeds:            {config.n_seeds}")
    print(f"Holdout m:        {config.holdout_size}")
    print(f"delta:            {config.delta}")
    print(f"Include baselines: {config.include_baselines}")
    print()

    all_dfs = []

    if use_pretrained:
        # Discover all models
        grouped = discover_pretrained_models(config.models_dir)
        print("Discovered model groups:")
        for tag in sorted(grouped):
            print(f"  {tag:25s}  {len(grouped[tag]):3d} models")
        print()

        # Resolve 'all'
        requested = config.datasets
        if 'all' in requested:
            requested = [t for t in ALL_DATASETS if t in grouped]
            print(f"Resolved 'all' to: {requested}")

        # Check availability
        missing = [d for d in requested if d not in grouped]
        if missing:
            available = ', '.join(sorted(grouped.keys()))
            raise ValueError(
                f"No pretrained models found for: {missing}.  "
                f"Available: {available}"
            )

        # Run each dataset
        for dataset_tag in requested:
            # Check if all seeds for this dataset are already done
            ds_seeds_done = {s for (d, dim, s) in completed_seeds if d == dataset_tag}
            if len(ds_seeds_done) >= config.n_seeds:
                print(f"\nSkipping {dataset_tag} (all {config.n_seeds} seeds in checkpoint)")
                continue

            model_paths = grouped[dataset_tag]
            print(f"\nLoading {len(model_paths)} models for '{dataset_tag}'...")
            suite = load_pretrained_suite(model_paths, verbose=True)
            if not suite:
                print(f"  WARNING: all models failed to load, skipping {dataset_tag}")
                continue

            df = run_for_dataset(
                dataset_tag, suite, config,
                completed_seeds=completed_seeds,
                seed_checkpoint_callback=seed_checkpoint_callback,
            )
            all_dfs.append(df)

            # Free GPU memory between datasets
            del suite
            try:
                import tensorflow as tf
                tf.keras.backend.clear_session()
            except Exception:
                pass

        if not all_dfs:
            raise RuntimeError("No datasets produced results.")

    else:
        # Synthetic mode: train sklearn models per seed
        print("Generating or loading data...")
        if config.dataset_name == 'synthetic':
            max_dim = max(config.dimensions) if config.dimensions else 100
            traces, labels = generate_synthetic_data(
                n_samples=config.n_samples_total,
                n_features=max_dim + 100,
                n_classes=config.n_classes,
                snr=0.5,
                n_leaky_points=min(20, max_dim // 5),
                random_state=0
            )
        else:
            traces, labels = load_dataset(
                config.dataset_name,
                config.dataset_path
            )

        print(f"Data shape: {traces.shape}")
        print()

        # Run synthetic experiment with no pretrained suite
        # Pass preloaded data so run_for_dataset doesn't try to re-load 'synthetic'
        df = run_for_dataset(
            config.dataset_name, None, config,
            preloaded_data=(traces, labels),
        )
        all_dfs.append(df)

    if not all_dfs:
        raise RuntimeError("No datasets produced results.")

    return pd.concat(all_dfs, ignore_index=True)


# ===================================================================
# Analysis (merged and extended)
# ===================================================================

def analyze_results(df: pd.DataFrame) -> Dict:
    """
    Compute per-dataset (and per-dimension) statistics.

    Args:
        df: Results DataFrame

    Returns:
        Analysis dictionary
    """
    analysis = {}
    rows = []

    for dataset_tag in df['dataset'].unique():
        df_ds = df[df['dataset'] == dataset_tag]
        for dim in sorted(df_ds['dimension'].unique()):
            df_dd = df_ds[df_ds['dimension'] == dim]
            row = {'dataset': dataset_tag, 'dimension': dim}

            # Metrics to analyze: BI, PI, HI, MI, MLP-PI, MLP-TI, TI
            for metric in ['bi', 'pi', 'hi', 'mi', 'mlp_pi', 'mlp_ti', 'ti']:
                if metric not in df_dd.columns:
                    continue
                valid = df_dd[metric].dropna()
                n_total = len(df_dd)
                if len(valid) > 0:
                    row[f'{metric}_mean'] = valid.mean()
                    row[f'{metric}_std'] = valid.std()
                    row[f'{metric}_cv'] = (
                        valid.std() / valid.mean() if valid.mean() != 0
                        else np.nan
                    )
                    row[f'{metric}_n_valid'] = len(valid)
                else:
                    row[f'{metric}_mean'] = np.nan
                    row[f'{metric}_std'] = np.nan
                    row[f'{metric}_cv'] = np.nan
                    row[f'{metric}_n_valid'] = 0
                row[f'{metric}_n_total'] = n_total

            rows.append(row)

    analysis['stats'] = pd.DataFrame(rows)
    return analysis


def analyze_stability(df: pd.DataFrame) -> Dict:
    """
    Analyze metric stability across dimensions.

    Args:
        df: Results DataFrame

    Returns:
        Analysis dictionary with stability metrics
    """
    analysis = {}

    # Group by dimension and compute statistics
    stats = []
    for dim in sorted(df['dimension'].unique()):
        df_dim = df[df['dimension'] == dim]
        row = {'dimension': dim}

        for metric in ['bi', 'pi', 'hi', 'mi', 'mlp_pi', 'mlp_ti', 'ti']:
            if metric in df_dim.columns:
                valid = df_dim[metric].dropna()
                if len(valid) > 0:
                    row[f'{metric}_mean'] = valid.mean()
                    row[f'{metric}_std'] = valid.std()
                    row[f'{metric}_cv'] = valid.std() / valid.mean() if valid.mean() != 0 else np.nan
                    row[f'{metric}_n_valid'] = len(valid)
                    row[f'{metric}_n_total'] = len(df_dim)
                else:
                    row[f'{metric}_mean'] = np.nan
                    row[f'{metric}_std'] = np.nan
                    row[f'{metric}_cv'] = np.nan
                    row[f'{metric}_n_valid'] = 0
                    row[f'{metric}_n_total'] = len(df_dim)

        stats.append(row)

    analysis['dimension_stats'] = pd.DataFrame(stats)

    # Identify failure thresholds
    for metric in ['pi', 'hi', 'mi', 'mlp_pi']:
        col = f'{metric}_n_valid'
        if col in analysis['dimension_stats'].columns:
            total_col = f'{metric}_n_total'
            for _, row in analysis['dimension_stats'].iterrows():
                if row.get(total_col, 0) > 0:
                    success_rate = row.get(col, 0) / row.get(total_col, 1)
                    if success_rate < 0.5:
                        analysis[f'{metric}_failure_threshold'] = row['dimension']
                        break

    return analysis


# ===================================================================
# Plotting (merged and extended)
# ===================================================================

def plot_results(df: pd.DataFrame, analysis: Dict, output_dir: Path):
    """
    Generate summary plots for the multivariate stability diagnostics.

    Args:
        df: Results DataFrame
        analysis: Analysis dictionary
        output_dir: Output directory
    """
    # ── Paper-quality formatting constants ──
    TITLE_SIZE = 18
    LABEL_SIZE = 16
    TICK_SIZE = 14
    LEGEND_SIZE = 13
    LINE_WIDTH = 2.5
    MARKER_SIZE = 8
    CAP_SIZE = 5

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    stats = analysis['stats']
    datasets_in_results = sorted(df['dataset'].unique())

    # ---- Plot 1: BI/PI/HI/TI/MLP-PI bar chart across datasets ----
    if len(datasets_in_results) > 1:
        metrics_to_plot = ['bi', 'pi', 'hi', 'ti', 'mlp_pi']
        colors = ['#2ecc71', '#3498db', '#e74c3c', '#9b59b6', '#f39c12']
        labels_map = {'bi': 'BI', 'pi': 'ePI', 'hi': 'eHI', 'ti': 'TI', 'mlp_pi': 'MLP-PI'}

        fig, ax = plt.subplots(
            figsize=(max(14, len(datasets_in_results) * 2.5), 8)
        )

        x = np.arange(len(datasets_in_results))
        width = 0.15

        for idx, (metric, color) in enumerate(zip(metrics_to_plot, colors)):
            mean_col = f'{metric}_mean'
            std_col = f'{metric}_std'

            if mean_col not in stats.columns:
                continue

            means, stds = [], []
            for ds in datasets_in_results:
                s = stats[stats['dataset'] == ds]
                if mean_col in s.columns and s[mean_col].notna().any():
                    means.append(s[mean_col].iloc[0])
                    stds.append(s[std_col].iloc[0] if std_col in s.columns else 0)
                else:
                    means.append(0)
                    stds.append(0)

            offset = (idx - 2) * width
            ax.bar(x + offset, means, width, label=labels_map.get(metric, metric.upper()),
                   color=color, alpha=0.85, edgecolor='white')

        ax.set_xticks(x)
        ax.set_xticklabels(datasets_in_results, rotation=30, ha='right', fontsize=TICK_SIZE)
        ax.set_ylabel('Bits', fontsize=LABEL_SIZE)
        ax.set_title(
            'Leakage Metrics Across Datasets (BI vs PI/HI/TI)',
            fontsize=TITLE_SIZE,
        )
        ax.tick_params(axis='both', labelsize=TICK_SIZE)
        ax.legend(fontsize=LEGEND_SIZE, loc='best')
        ax.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.savefig(output_dir / 'dimension_metrics_across_datasets.pdf', dpi=300)
        plt.savefig(output_dir / 'dimension_metrics_across_datasets.png', dpi=150)
        plt.close()

    # ---- Plot 2: Per-model success breakdown per dataset (Keras suite) ----
    for ds in datasets_in_results:
        df_ds = df[df['dataset'] == ds]
        raw = df_ds.get('per_model_success', pd.Series(dtype=str)).dropna()
        if raw.empty:
            continue

        try:
            per_model = json.loads(raw.iloc[0])
        except:
            continue

        names = list(per_model.keys())
        accs = [per_model[n] for n in names]

        # Sort by accuracy descending
        order = np.argsort(accs)[::-1]
        names = [names[i] for i in order]
        accs = [accs[i] for i in order]

        fig, ax = plt.subplots(figsize=(max(12, len(names) * 0.7), 8))
        ax.barh(range(len(names)), accs, color='#3498db', edgecolor='white')
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names, fontsize=max(10, TICK_SIZE - 2))
        ax.set_xlabel('Holdout Success Rate', fontsize=LABEL_SIZE)
        ax.set_title(f'Per-Model Success: {ds} (Pretrained Suite)', fontsize=TITLE_SIZE)
        ax.tick_params(axis='x', labelsize=TICK_SIZE)
        ax.axvline(
            x=1.0 / 256, color='gray', linestyle='--', alpha=0.5,
            label='Random guess (1/256)',
        )
        ax.legend(fontsize=LEGEND_SIZE)
        ax.invert_yaxis()
        plt.tight_layout()
        plt.savefig(output_dir / f'per_model_success_{ds}.pdf', dpi=300)
        plt.savefig(output_dir / f'per_model_success_{ds}.png', dpi=150)
        plt.close()

    # ---- Plot 3: Dimension sweep stability ----
    for ds in datasets_in_results:
        ds_stats = stats[stats['dataset'] == ds]
        if len(ds_stats) <= 1:
            continue

        fig, ax = plt.subplots(figsize=(12, 8))
        metrics = ['bi', 'pi', 'hi', 'mi', 'mlp_pi']
        colors = ['#2ecc71', '#3498db', '#e74c3c', '#9b59b6', '#f39c12']
        labels_map = {'bi': 'BI', 'pi': 'ePI', 'hi': 'eHI', 'mi': 'MI', 'mlp_pi': 'MLP-PI'}

        for metric, color in zip(metrics, colors):
            mc = f'{metric}_mean'
            sc = f'{metric}_std'
            if mc not in ds_stats.columns:
                continue
            valid = ds_stats[ds_stats[mc].notna()]
            if valid.empty:
                continue
            ax.errorbar(
                valid['dimension'], valid[mc], yerr=valid[sc],
                marker='o', capsize=CAP_SIZE, color=color, linewidth=LINE_WIDTH,
                markersize=MARKER_SIZE,
                label=labels_map.get(metric, metric.upper()),
            )

        ax.set_xlabel("Dimension (d')", fontsize=LABEL_SIZE)
        ax.set_ylabel('Bits', fontsize=LABEL_SIZE)
        ax.set_title(f'Metric Stability vs Dimension: {ds}', fontsize=TITLE_SIZE)
        ax.tick_params(axis='both', labelsize=TICK_SIZE)
        if len(ds_stats) > 3:
            ax.set_xscale('log')
            ax.xaxis.set_major_formatter(ScalarFormatter())
        ax.legend(fontsize=LEGEND_SIZE, loc='best')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / f'dimension_stability_{ds}.pdf', dpi=300)
        plt.savefig(output_dir / f'dimension_stability_{ds}.png', dpi=150)
        plt.close()

    # ---- Plot 4: Coefficient of Variation comparison ----
    fig, ax = plt.subplots(figsize=(12, 8))
    metrics = ['bi', 'pi', 'hi', 'mi', 'mlp_pi']
    colors = ['#2ecc71', '#3498db', '#e74c3c', '#9b59b6', '#f39c12']
    labels_map = {'bi': 'BI', 'pi': 'ePI', 'hi': 'eHI', 'mi': 'MI', 'mlp_pi': 'MLP-PI'}

    if 'dimension_stats' in analysis:
        stats_by_dim = analysis['dimension_stats']
        for metric, color in zip(metrics, colors):
            cv_col = f'{metric}_cv'
            if cv_col in stats_by_dim.columns:
                valid = stats_by_dim[stats_by_dim[cv_col].notna()]
                if len(valid) > 0:
                    ax.plot(valid['dimension'], valid[cv_col],
                           marker='o', color=color, label=labels_map.get(metric, metric.upper()),
                           linewidth=LINE_WIDTH, markersize=MARKER_SIZE)

        ax.set_xlabel('Dimension (d\')', fontsize=LABEL_SIZE)
        ax.set_ylabel('Coefficient of Variation (std/mean)', fontsize=LABEL_SIZE)
        ax.set_title('Metric Stability: Coefficient of Variation vs Dimension', fontsize=TITLE_SIZE)
        ax.tick_params(axis='both', labelsize=TICK_SIZE)
        if len(stats_by_dim) > 3:
            ax.set_xscale('log')
            ax.xaxis.set_major_formatter(ScalarFormatter())
        ax.legend(fontsize=LEGEND_SIZE, loc='best')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / 'coefficient_of_variation.pdf', dpi=300)
        plt.savefig(output_dir / 'coefficient_of_variation.png', dpi=150)
        plt.close()

    # ---- Plot 5: Success rate vs dimension ----
    fig, ax = plt.subplots(figsize=(12, 8))
    if 'dimension_stats' in analysis:
        stats_by_dim = analysis['dimension_stats']
        for metric, color in zip(metrics, colors):
            valid_col = f'{metric}_n_valid'
            total_col = f'{metric}_n_total'

            if valid_col in stats_by_dim.columns and total_col in stats_by_dim.columns:
                valid = stats_by_dim[[valid_col, total_col, 'dimension']].dropna()
                if len(valid) > 0:
                    success_rate = valid[valid_col] / valid[total_col]
                    ax.plot(valid['dimension'], success_rate * 100,
                           marker='s', color=color, label=labels_map.get(metric, metric.upper()),
                           linewidth=LINE_WIDTH, markersize=MARKER_SIZE)

        ax.axhline(y=50, color='gray', linestyle='--', alpha=0.5, label='50% threshold')
        ax.set_xlabel('Dimension (d\')', fontsize=LABEL_SIZE)
        ax.set_ylabel('Computation Success Rate (%)', fontsize=LABEL_SIZE)
        ax.set_title('Metric Computability vs Dimension', fontsize=TITLE_SIZE)
        ax.tick_params(axis='both', labelsize=TICK_SIZE)
        if len(stats_by_dim) > 3:
            ax.set_xscale('log')
            ax.xaxis.set_major_formatter(ScalarFormatter())
        ax.set_ylim(0, 105)
        ax.legend(fontsize=LEGEND_SIZE, loc='best')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / 'success_rate.pdf', dpi=300)
        plt.savefig(output_dir / 'success_rate.png', dpi=150)
        plt.close()

    print(f"Plots saved to {output_dir}")


# ===================================================================
# CLI
# ===================================================================

def main():
    """Main entry point for multivariate stability diagnostics."""
    parser = argparse.ArgumentParser(
        description='Multivariate leakage stability certification '
                    '(BI vs PI/HI/MI Stability)',
    )

    # Pretrained model evaluation
    parser.add_argument(
        '--models-dir', type=str, default='',
        help='Root directory containing pretrained .hdf5 files',
    )
    parser.add_argument(
        '--datasets', type=str, nargs='+', default=['aes_hd'],
        help='Dataset tags to evaluate. Use "all" for all TCHES20 datasets.',
    )
    parser.add_argument(
        '--dataset-dir', type=str, default='./datasets',
        help='Root directory containing dataset folders',
    )
    parser.add_argument(
        '--tches20-src-dir', type=str, default='',
        help='Path to TCHES20 repo src/ directory',
    )

    # Synthetic data mode
    parser.add_argument(
        '--dataset', type=str, default='synthetic',
        help='Dataset name for synthetic mode (synthetic, ascad, aes_hd)',
    )
    parser.add_argument(
        '--dataset-path', type=str, default='',
        help='Path to dataset file',
    )

    # Dimension sweep
    parser.add_argument(
        '--sweep-dimensions', action='store_true',
        help='Sweep over --dimensions instead of using full traces',
    )
    parser.add_argument(
        '--dimensions', type=str, default='10,20,50,100,200,500,1000',
        help='Comma-separated dimension list',
    )

    # Certification parameters
    parser.add_argument('--holdout-size', type=int, default=50000)
    parser.add_argument('--delta', type=float, default=1e-6)
    parser.add_argument('--n-classes', type=int, default=256)
    parser.add_argument('--n-seeds', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=1024)

    # Baselines
    parser.add_argument(
        '--include-baselines', action='store_true', default=True,
        help='Compute PI/HI/MI/MLP-PI baselines (default: True)',
    )
    parser.add_argument(
        '--no-baselines', action='store_true',
        help='Disable baseline metrics',
    )
    parser.add_argument('--max-dim-mi', type=int, default=100)

    # Output
    parser.add_argument(
        '--output-dir', type=str, default='results/dimension_stability',
        help='Output directory',
    )

    # Checkpoint and resume
    parser.add_argument(
        '--resume', action='store_true', default=True,
        help='Resume from checkpoint if it exists (default: True)',
    )
    parser.add_argument(
        '--no-resume', action='store_true',
        help='Do not resume from checkpoint',
    )
    parser.add_argument(
        '--plot-only', action='store_true',
        help='Load checkpoint and generate plots only (skip experiments)',
    )

    args = parser.parse_args()

    # Override baselines flag
    include_baselines = not args.no_baselines

    # Parse resume flag
    do_resume = args.resume and not args.no_resume

    # Parse dimensions
    dimensions = [int(d.strip()) for d in args.dimensions.split(',')]

    # Determine mode
    use_pretrained = bool(args.models_dir and args.dataset_dir)

    if use_pretrained:
        config = ExperimentConfig(
            datasets=args.datasets,
            models_dir=args.models_dir,
            dataset_dir=args.dataset_dir,
            tches20_src_dir=args.tches20_src_dir,
            dimensions=dimensions,
            sweep_dimensions=args.sweep_dimensions,
            holdout_size=args.holdout_size,
            delta=args.delta,
            n_classes=args.n_classes,
            n_seeds=args.n_seeds,
            batch_size=args.batch_size,
            include_baselines=include_baselines,
            max_dim_for_mi=args.max_dim_mi,
            use_pretrained_models=True,
        )
    else:
        config = ExperimentConfig(
            dataset_name=args.dataset,
            dataset_path=args.dataset_path,
            dimensions=dimensions,
            sweep_dimensions=args.sweep_dimensions,
            holdout_size=args.holdout_size,
            delta=args.delta,
            n_classes=args.n_classes,
            n_seeds=args.n_seeds,
            batch_size=args.batch_size,
            include_baselines=include_baselines,
            max_dim_for_mi=args.max_dim_mi,
            use_pretrained_models=False,
        )

    # Setup output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Checkpoint path
    checkpoint_path = output_dir / 'dimension_stability_checkpoint.csv'

    # Load existing checkpoint if resuming
    existing_df = None
    completed_seeds = set()  # (dataset, dimension, seed) tuples
    if do_resume and checkpoint_path.exists():
        try:
            existing_df = pd.read_csv(checkpoint_path)
            for _, row in existing_df.iterrows():
                dim = row.get('dimension', None)
                if pd.isna(dim):
                    dim = None
                else:
                    dim = int(dim) if dim == int(dim) else dim
                completed_seeds.add((row['dataset'], dim, int(row['seed'])))
            print(f"  Resuming: found checkpoint with {len(completed_seeds)} completed (dataset, dim, seed) entries")
            ds_counts = {}
            for (d, dim, s) in completed_seeds:
                ds_counts[d] = ds_counts.get(d, 0) + 1
            for d, c in sorted(ds_counts.items()):
                print(f"    {d}: {c} seeds done")
            print()
        except Exception as e:
            print(f"  Warning: failed to load checkpoint: {e}")
            print()

    # If plot-only mode, skip experiment and go directly to analysis/plotting
    if args.plot_only:
        if existing_df is None or existing_df.empty:
            print("ERROR: --plot-only requires an existing checkpoint file.")
            sys.exit(1)
        df = existing_df
        print(f"Plot-only mode: loaded {len(df)} results from checkpoint.")
    else:
        # Accumulate new results for incremental saving
        new_rows = []

        def _seed_checkpoint(result_dict):
            """Save checkpoint after every single seed."""
            new_rows.append(result_dict)
            new_df = pd.DataFrame(new_rows)
            if existing_df is not None:
                combined = pd.concat([existing_df, new_df], ignore_index=True)
                combined = combined.drop_duplicates(
                    subset=['dataset', 'dimension', 'seed'],
                    keep='last'
                )
            else:
                combined = new_df
            combined.to_csv(checkpoint_path, index=False)
            print(f"      [checkpoint saved: {len(combined)} total rows]")

        # Define SIGTERM/SIGUSR1 handler
        def _sigterm_handler(signum, frame):
            print(f"\n  SIGNAL {signum} received - saving partial results...")
            if new_rows:
                new_df = pd.DataFrame(new_rows)
                if existing_df is not None:
                    combined = pd.concat([existing_df, new_df], ignore_index=True)
                    combined = combined.drop_duplicates(
                        subset=['dataset', 'dimension', 'seed'],
                        keep='last'
                    )
                else:
                    combined = new_df
                combined.to_csv(checkpoint_path, index=False)
            print(f"  Checkpoint saved to {checkpoint_path}. Re-run to resume.")
            sys.exit(0)

        signal.signal(signal.SIGTERM, _sigterm_handler)
        signal.signal(signal.SIGUSR1, _sigterm_handler)

        # Run experiment with seed-level checkpoint support
        df = run_experiment(config, completed_seeds, _seed_checkpoint)

        # Merge with existing checkpoint for final df
        if existing_df is not None and not existing_df.empty:
            if not df.empty:
                df = pd.concat([existing_df, df], ignore_index=True)
                df = df.drop_duplicates(
                    subset=['dataset', 'dimension', 'seed'],
                    keep='last'
                )
            else:
                df = existing_df

    # Analyze results
    if config.sweep_dimensions or len(df['dimension'].unique()) > 1:
        analysis = analyze_stability(df)
    else:
        analysis = analyze_results(df)

    # Save results (only if not plot-only)
    if not args.plot_only:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        df.to_csv(output_dir / f'dimension_stability_results_{timestamp}.csv', index=False)

        if 'dimension_stats' in analysis:
            analysis['dimension_stats'].to_csv(
                output_dir / f'dimension_stability_stats_{timestamp}.csv', index=False
            )
        elif 'stats' in analysis:
            analysis['stats'].to_csv(
                output_dir / f'dimension_stability_stats_{timestamp}.csv', index=False
            )
    else:
        # For plot-only mode, still save analysis stats (without timestamp)
        if 'dimension_stats' in analysis:
            analysis['dimension_stats'].to_csv(
                output_dir / 'dimension_stability_stats_plot_only.csv', index=False
            )
        elif 'stats' in analysis:
            analysis['stats'].to_csv(
                output_dir / 'dimension_stability_stats_plot_only.csv', index=False
            )

    # Print summary
    print("\n" + "=" * 70)
    print("MULTIVARIATE STABILITY SUMMARY")
    print("=" * 70)

    for ds in sorted(df['dataset'].unique()):
        df_ds = df[df['dataset'] == ds]
        bi_vals = df_ds['bi'].dropna()
        if bi_vals.empty:
            print(f"  {ds:25s}  BI: ALL FAILED")
            continue
        M = df_ds['M'].iloc[0] if 'M' in df_ds.columns else None
        m_str = f"  M={int(M)}" if M is not None else ""
        print(f"  {ds:25s}  BI: {bi_vals.mean():.4f} +/- {bi_vals.std():.4f}  "
              f"(n={len(bi_vals)}){m_str}")

        # Top models (if available)
        raw = df_ds.get('per_model_success', pd.Series(dtype=str)).dropna()
        if not raw.empty:
            try:
                per_model = json.loads(raw.iloc[0])
                top3 = sorted(per_model.items(), key=lambda kv: -kv[1])[:3]
                for name, acc in top3:
                    print(f"    {name}: {acc:.4f}")
            except:
                pass

    print()

    # Generate plots
    plot_results(df, analysis, output_dir / 'figures')

    # Stability check
    if 'stats' in analysis or 'dimension_stats' in analysis:
        s = analysis.get('stats') or analysis.get('dimension_stats')
        for ds in sorted(df['dataset'].unique()):
            bi_cv = s.loc[s['dataset'] == ds, 'bi_cv'].dropna() if 'dataset' in s.columns else pd.Series()
            if len(bi_cv) == 0:
                bi_cv = s['bi_cv'].dropna() if 'bi_cv' in s.columns else pd.Series()

            if len(bi_cv) > 0:
                ok = bi_cv.max() < 0.5
                marker = 'PASS' if ok else 'NEEDS REVIEW'
                print(f"  {ds}: BI stability {marker} (max CV = {bi_cv.max():.4f})")

    print(f"\nResults saved to {output_dir}")


if __name__ == '__main__':
    main()
