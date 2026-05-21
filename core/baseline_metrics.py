"""
Baseline Metrics for Leakage Certification

This module provides unified interfaces for computing:
- PI (Perceived Information) via trained classifiers
- HI (Hypothetical Information) via Gaussian templates
- MI (Mutual Information) via GKOV/KSG estimators

Adapted from:
- Leakage-Certification-Made-Simple (GKOV_MI.py, Plugin_MI.py, main.py)
- Leakage_Certification_Revisited (Models.py, MI_computation.py)
"""

import numpy as np
import scipy.special
from scipy.spatial import cKDTree
from scipy.stats import multivariate_normal
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.naive_bayes import GaussianNB
from sklearn.covariance import LedoitWolf
from typing import Tuple, Optional, List, Dict, Union
import warnings


# =============================================================================
# MUTUAL INFORMATION ESTIMATORS (from GKOV_MI.py)
# =============================================================================

def gkov_mi_univariate(traces: np.ndarray, labels: np.ndarray, k: int = None) -> float:
    """
    GKOV MI estimator for univariate traces.

    Adapted from: Leakage-Certification-Made-Simple/GKOV_MI.py::MI_mixt_gao

    Args:
        traces: Univariate traces (n_samples,) or (n_samples, 1)
        labels: Target labels (n_samples,)
        k: Nearest neighbor parameter (default: log(n))

    Returns:
        MI estimate in bits
    """
    n = len(labels)
    if k is None:
        k = max(1, int(np.log(n)))

    # Ensure proper shapes
    traces = np.asarray(traces).flatten()
    labels = np.asarray(labels).flatten()

    # Combine into joint space
    joint = np.column_stack((traces, labels))

    # Build KD-trees
    tree_joint = cKDTree(joint)
    tree_traces = cKDTree(traces.reshape(-1, 1))
    tree_labels = cKDTree(labels.reshape(-1, 1))

    # Find k-th nearest neighbor distances
    dist = tree_joint.query(joint, k=k+1, p=float('inf'), workers=-1)[0][:, k]

    # Initialize counts
    kp_arr = np.full(n, k, dtype=int)
    m_arr = np.zeros(n, dtype=int)
    n_arr = np.zeros(n, dtype=int)

    # Handle zero distances (identical points)
    zero_mask = (dist == 0)
    if np.any(zero_mask):
        zero_idx = np.where(zero_mask)[0]
        zero_pts = joint[zero_idx]
        kp_arr[zero_mask] = tree_joint.query_ball_point(
            zero_pts, 1e-15, p=float('inf'), workers=-1, return_length=True
        )
        m_arr[zero_mask] = tree_traces.query_ball_point(
            zero_pts[:, 0:1], 1e-15, p=float('inf'), workers=-1, return_length=True
        )
        n_arr[zero_mask] = tree_labels.query_ball_point(
            zero_pts[:, 1:2], 1e-15, p=float('inf'), workers=-1, return_length=True
        )

    # Handle non-zero distances
    nonzero_mask = ~zero_mask
    if np.any(nonzero_mask):
        nonzero_idx = np.where(nonzero_mask)[0]
        m_arr[nonzero_mask] = tree_traces.query_ball_point(
            traces[nonzero_idx].reshape(-1, 1),
            dist[nonzero_idx],
            p=float('inf'), workers=-1, return_length=True
        )
        n_arr[nonzero_mask] = tree_labels.query_ball_point(
            labels[nonzero_idx].reshape(-1, 1),
            dist[nonzero_idx],
            p=float('inf'), workers=-1, return_length=True
        )

    # Compute MI estimate
    digamma_kp = scipy.special.digamma(kp_arr)
    mi = np.mean(digamma_kp + np.log(n) - np.log(m_arr + 1) - np.log(n_arr + 1))

    # Convert to bits
    return mi * np.log2(np.e)


def gkov_mi_multivariate(traces: np.ndarray, labels: np.ndarray, k: int = None) -> float:
    """
    GKOV MI estimator for multivariate traces.

    Adapted from: Leakage-Certification-Made-Simple/GKOV_MI.py::MI_gao_multi

    Args:
        traces: Multivariate traces (n_samples, n_features)
        labels: Target labels (n_samples,)
        k: Nearest neighbor parameter (default: log(n))

    Returns:
        MI estimate in bits
    """
    n = len(labels)
    if k is None:
        k = max(1, int(np.log(n)))

    traces = np.atleast_2d(traces)
    if traces.shape[0] == 1:
        traces = traces.T

    labels = np.asarray(labels).reshape(-1, 1)

    # Combine into joint space
    joint = np.column_stack((traces, labels))

    # Build KD-trees
    tree_joint = cKDTree(joint)
    tree_traces = cKDTree(traces)
    tree_labels = cKDTree(labels)

    # Find k-th nearest neighbor distances
    dist = tree_joint.query(joint, k=k+1, p=float('inf'), workers=-1)[0][:, k]

    # Adjust distances for boundary handling
    cond_dist = np.where(dist == 0, dist, dist - 2e-15)

    # Count neighbors in marginal spaces
    m_arr = tree_traces.query_ball_point(
        traces, cond_dist + 1e-15,
        p=float('inf'), workers=-1, return_length=True
    )
    n_arr = tree_labels.query_ball_point(
        labels, cond_dist + 1e-15,
        p=float('inf'), workers=-1, return_length=True
    )

    # Compute MI with zero-distance handling
    mi = 0.0
    for i in range(n):
        kp = k
        if dist[i] == 0:
            kp = tree_joint.query_ball_point(
                joint[i], 1e-15, p=float('inf'), workers=-1, return_length=True
            )
        mi += (scipy.special.digamma(kp) - np.log(m_arr[i] + 1) - np.log(n_arr[i] + 1)) / n

    mi += np.log(n)

    # Convert to bits
    return mi * np.log2(np.e)


def compute_mi(
    traces: np.ndarray,
    labels: np.ndarray,
    k: int = None,
    max_dim_for_estimate: int = 100
) -> Tuple[float, bool]:
    """
    Compute MI with automatic selection of univariate/multivariate estimator.

    Args:
        traces: Trace data (n_samples,) or (n_samples, n_features)
        labels: Target labels
        k: Nearest neighbor parameter
        max_dim_for_estimate: Maximum dimension for reliable MI estimation

    Returns:
        Tuple of (MI estimate in bits, success flag)
    """
    traces = np.atleast_2d(traces)
    if traces.shape[0] < traces.shape[1]:
        traces = traces.T

    n_samples, n_features = traces.shape

    # Check if dimension is too high for reliable estimation
    if n_features > max_dim_for_estimate:
        warnings.warn(
            f"MI estimation unreliable for d={n_features} > {max_dim_for_estimate}. "
            "Consider dimensionality reduction."
        )
        return np.nan, False

    try:
        if n_features == 1:
            mi = gkov_mi_univariate(traces.flatten(), labels, k)
        else:
            mi = gkov_mi_multivariate(traces, labels, k)
        return mi, True
    except Exception as e:
        warnings.warn(f"MI estimation failed: {e}")
        return np.nan, False


# =============================================================================
# PERCEIVED INFORMATION (PI) via Classifiers
# =============================================================================

def compute_pi_logistic(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    n_classes: int = 256
) -> Tuple[float, float]:
    """
    Compute PI and TI using logistic regression.

    Adapted from: Leakage-Certification-Made-Simple/main.py::run_lr

    Args:
        X_train: Training traces
        y_train: Training labels
        X_val: Validation traces
        y_val: Validation labels
        n_classes: Number of classes (default 256)

    Returns:
        Tuple of (PI on validation, TI on training) in bits
    """
    # Train logistic regression
    clf = LogisticRegression(
    solver='lbfgs',
    max_iter=1000,
    n_jobs=-1
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        clf.fit(X_train, y_train)

    # Compute log-probability scores
    score_val = 0.0
    score_train = 0.0

    for i in range(n_classes):
        # Validation set
        X_val_i = X_val[y_val == i]
        if len(X_val_i) > 0:
            log_proba = clf.predict_log_proba(X_val_i)
            # Get probability for correct class
            if i < log_proba.shape[1]:
                score_val += np.mean(log_proba[:, i])

        # Training set
        X_train_i = X_train[y_train == i]
        if len(X_train_i) > 0:
            log_proba = clf.predict_log_proba(X_train_i)
            if i < log_proba.shape[1]:
                score_train += np.mean(log_proba[:, i])

    # Convert to bits: PI = log2(n_classes) + score (in natural log) * log2(e)
    log2_n = np.log2(n_classes)
    PI = log2_n + (score_val / n_classes) * np.log2(np.e)
    TI = log2_n + (score_train / n_classes) * np.log2(np.e)

    return PI, TI


def compute_pi_lda(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    n_classes: int = 256
) -> Tuple[float, float]:
    """
    Compute PI using Linear Discriminant Analysis.

    Args:
        X_train: Training traces
        y_train: Training labels
        X_val: Validation traces
        y_val: Validation labels
        n_classes: Number of classes

    Returns:
        Tuple of (PI on validation, TI on training) in bits
    """
    clf = LDA()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        clf.fit(X_train, y_train)

    score_val = 0.0
    score_train = 0.0

    # Get log probabilities
    log_proba_val = np.log(np.clip(clf.predict_proba(X_val), 1e-10, 1.0))
    log_proba_train = np.log(np.clip(clf.predict_proba(X_train), 1e-10, 1.0))

    for i in range(n_classes):
        idx_val = np.where(y_val == i)[0]
        idx_train = np.where(y_train == i)[0]

        if len(idx_val) > 0 and i < log_proba_val.shape[1]:
            score_val += np.mean(log_proba_val[idx_val, i])

        if len(idx_train) > 0 and i < log_proba_train.shape[1]:
            score_train += np.mean(log_proba_train[idx_train, i])

    log2_n = np.log2(n_classes)
    PI = log2_n + (score_val / n_classes) * np.log2(np.e)
    TI = log2_n + (score_train / n_classes) * np.log2(np.e)

    return PI, TI


def compute_pi(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    method: str = 'logistic',
    n_classes: int = 256
) -> Tuple[float, float]:
    """
    Unified PI computation interface.

    Args:
        X_train: Training traces
        y_train: Training labels
        X_val: Validation traces
        y_val: Validation labels
        method: 'logistic' or 'lda'
        n_classes: Number of classes

    Returns:
        Tuple of (PI, TI) in bits
    """
    if method == 'logistic':
        return compute_pi_logistic(X_train, y_train, X_val, y_val, n_classes)
    elif method == 'lda':
        return compute_pi_lda(X_train, y_train, X_val, y_val, n_classes)
    else:
        raise ValueError(f"Unknown method: {method}")


# =============================================================================
# HYPOTHETICAL INFORMATION (HI) via Gaussian Templates
# =============================================================================

class GaussianTemplate:
    """
    Gaussian Template model for HI computation.

    Adapted from: Leakage_Certification_Revisited/Models.py::GT
    """

    def __init__(self, n_classes: int = 256, n_dims: int = 1):
        self.n_classes = n_classes
        self.n_dims = n_dims
        self.means = None
        self.covs = None
        self.priors = None
        self._fitted = False

    def fit(self, traces: np.ndarray, labels: np.ndarray, regularize: bool = True):
        """
        Fit Gaussian templates to training data.

        Args:
            traces: Training traces (n_samples, n_dims)
            labels: Training labels (n_samples,)
            regularize: Whether to use Ledoit-Wolf covariance estimation
        """
        traces = np.atleast_2d(traces)
        if traces.shape[0] < traces.shape[1]:
            traces = traces.T

        self.n_dims = traces.shape[1]
        self.means = np.zeros((self.n_classes, self.n_dims))
        self.covs = np.zeros((self.n_classes, self.n_dims, self.n_dims))
        self.priors = np.zeros(self.n_classes)

        for k in range(self.n_classes):
            idx = np.where(labels == k)[0]
            self.priors[k] = len(idx)

            if len(idx) > 0:
                self.means[k] = np.mean(traces[idx], axis=0)

                if len(idx) > 1:
                    if regularize and self.n_dims > 1:
                        try:
                            lw = LedoitWolf()
                            lw.fit(traces[idx])
                            self.covs[k] = lw.covariance_
                        except:
                            self.covs[k] = np.cov(traces[idx].T) + 1e-6 * np.eye(self.n_dims)
                    else:
                        cov = np.cov(traces[idx].T)
                        if self.n_dims == 1:
                            cov = np.array([[cov]])
                        self.covs[k] = cov + 1e-6 * np.eye(self.n_dims)
                else:
                    self.covs[k] = np.eye(self.n_dims) * 1e-2

        # Normalize priors
        self.priors = self.priors / np.sum(self.priors)
        self._fitted = True

    def pdf(self, x: np.ndarray, k: int) -> np.ndarray:
        """Compute PDF of class k at points x."""
        if not self._fitted:
            raise RuntimeError("Model not fitted")

        try:
            return multivariate_normal.pdf(x, self.means[k], self.covs[k])
        except:
            return np.zeros(len(x)) + 1e-100


def compute_hi_gaussian_template(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    n_classes: int = 256
) -> float:
    """
    Compute Hypothetical Information using Gaussian templates.

    HI uses the same model for both oracle (true distribution) and
    the estimation model.

    Adapted from: Leakage_Certification_Revisited/MI_computation.py::MI_sampling

    Args:
        X_train: Training traces
        y_train: Training labels
        X_val: Validation traces (used as "oracle" samples)
        y_val: Validation labels
        n_classes: Number of classes

    Returns:
        HI estimate in bits
    """
    # Fit Gaussian template
    gt = GaussianTemplate(n_classes=n_classes)
    gt.fit(X_train, y_train)

    # Compute HI via sampling
    H_Y = 0.0  # Entropy of labels
    S = 0.0    # Conditional term

    for k in range(n_classes):
        if gt.priors[k] == 0:
            continue

        H_Y -= gt.priors[k] * np.log2(gt.priors[k] + 1e-100)

        # Get samples for this class
        idx = np.where(y_val == k)[0]
        if len(idx) == 0:
            continue

        samples = X_val[idx]

        # Compute p(l|k) for all classes using the model
        pr_l_k = np.zeros((n_classes, len(samples)))
        for j in range(n_classes):
            pr_l_k[j] = gt.pdf(samples, j)

        # Compute p(k|l) via Bayes
        pr_all = np.sum(gt.priors.reshape(-1, 1) * pr_l_k, axis=0)
        pr_k_l = (gt.priors[k] * pr_l_k[k]) / (pr_all + 1e-100)

        # Add to sum
        S += gt.priors[k] * np.mean(np.log2(pr_k_l + 1e-100))

    return H_Y + S


def compute_hi(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    n_classes: int = 256
) -> float:
    """
    Unified HI computation interface.

    Args:
        X_train: Training traces
        y_train: Training labels
        X_val: Validation traces
        y_val: Validation labels
        n_classes: Number of classes

    Returns:
        HI estimate in bits
    """
    return compute_hi_gaussian_template(X_train, y_train, X_val, y_val, n_classes)


# =============================================================================
# UNIFIED INTERFACE
# =============================================================================

def compute_all_metrics(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    compute_mi_flag: bool = True,
    max_dim_for_mi: int = 20,
    n_classes: int = 256
) -> Dict[str, float]:
    """
    Compute all baseline metrics: PI, TI, HI, and optionally MI.

    Args:
        X_train: Training traces
        y_train: Training labels
        X_val: Validation traces
        y_val: Validation labels
        compute_mi_flag: Whether to compute MI
        max_dim_for_mi: Maximum dimension for MI computation
        n_classes: Number of classes

    Returns:
        Dictionary with all metrics in bits
    """
    results = {}

    # PI and TI
    try:
        pi, ti = compute_pi(X_train, y_train, X_val, y_val, n_classes=n_classes)
        results['PI'] = pi
        results['TI'] = ti
    except Exception as e:
        warnings.warn(f"PI computation failed: {e}")
        results['PI'] = np.nan
        results['TI'] = np.nan

    # HI (eHI - empirical HI)
    try:
        hi = compute_hi(X_train, y_train, X_val, y_val, n_classes=n_classes)
        results['HI'] = hi
    except Exception as e:
        warnings.warn(f"HI computation failed: {e}")
        results['HI'] = np.nan

    # MI (only if dimension is reasonable)
    if compute_mi_flag:
        X_combined = np.vstack([X_train, X_val])
        y_combined = np.concatenate([y_train, y_val])
        n_dims = X_combined.shape[1] if X_combined.ndim > 1 else 1

        if n_dims <= max_dim_for_mi:
            mi, success = compute_mi(X_combined, y_combined)
            results['MI'] = mi if success else np.nan
            results['MI_success'] = success
        else:
            results['MI'] = np.nan
            results['MI_success'] = False
            results['MI_note'] = f"Skipped: dim={n_dims} > max={max_dim_for_mi}"
    else:
        results['MI'] = np.nan
        results['MI_success'] = False

    return results


if __name__ == "__main__":
    # Example usage
    print("Baseline Metrics Module - Example Usage")
    print("=" * 50)

    # Generate synthetic data
    np.random.seed(42)
    n_samples = 5000
    n_classes = 256
    n_dims = 5

    # Simple leakage model: trace = HW(label) + noise
    labels = np.random.randint(0, n_classes, n_samples)
    hw = np.array([bin(x).count('1') for x in labels])
    traces = hw.reshape(-1, 1) + np.random.randn(n_samples, 1) * 0.5

    # Multi-dimensional traces
    traces_multi = np.column_stack([
        traces,
        np.random.randn(n_samples, n_dims - 1)  # Add noise dimensions
    ])

    # Split data
    split = int(0.7 * n_samples)
    X_train, X_val = traces_multi[:split], traces_multi[split:]
    y_train, y_val = labels[:split], labels[split:]

    # Compute metrics
    print("\nComputing metrics for synthetic data...")
    results = compute_all_metrics(
        X_train, y_train, X_val, y_val,
        compute_mi_flag=True,
        max_dim_for_mi=10
    )

    print(f"\nResults:")
    for key, value in results.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f} bits")
        else:
            print(f"  {key}: {value}")
